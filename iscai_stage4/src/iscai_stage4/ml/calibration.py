from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
import torch

from .gaussian_math import (
    CHI_SQUARE_3_THRESHOLDS,
    CONFIDENCE_LEVELS,
)


POSITION_DIM = 3
HORIZON_COUNT = 4

MIN_VARIANCE_SCALE = 0.05
MAX_VARIANCE_SCALE = 20.0


@dataclass(frozen=True)
class CovarianceScaleCalibration:
    """
    Per-horizon scalar calibration.

    Predictive mean is untouched.

    For horizon h:

        Sigma_cal,h = alpha_h * Sigma_raw,h
        L_cal,h     = sqrt(alpha_h) * L_raw,h

    This modifies predictive uncertainty only.

    Stage2 measurement covariance R_t remains
    an upstream conditioning input and is not
    calibrated here.
    """

    variance_scale: tuple[
        float,
        float,
        float,
        float,
    ]

    def __post_init__(self) -> None:
        if len(
            self.variance_scale
        ) != HORIZON_COUNT:
            raise ValueError(
                "Expected four horizon "
                "variance scales."
            )

        for value in (
            self.variance_scale
        ):
            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    "Calibration scale "
                    "must be finite."
                )

            if not (
                MIN_VARIANCE_SCALE
                <=
                float(value)
                <=
                MAX_VARIANCE_SCALE
            ):
                raise ValueError(
                    "Calibration scale "
                    "outside frozen bounds."
                )

    def to_dict(self) -> dict:
        return {
            "method":
                "per_horizon_scalar_covariance_scaling",

            "variance_scale":
                [
                    float(x)
                    for x in (
                        self.variance_scale
                    )
                ],

            "standard_deviation_scale":
                [
                    math.sqrt(
                        float(x)
                    )
                    for x in (
                        self.variance_scale
                    )
                ],

            "mean_modified":
                False,

            "measurement_covariance_R_t_modified":
                False,

            "predictive_covariance_modified":
                True,
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode("utf-8")

        return sha256(
            encoded
        ).hexdigest()


def apply_variance_scale(
    scale_tril: torch.Tensor,
    variance_scale,
) -> torch.Tensor:
    """
    Apply alpha_h to predictive covariance by
    multiplying Cholesky factor by sqrt(alpha_h).
    """

    if scale_tril.ndim != 4:
        raise ValueError(
            "scale_tril must be "
            "[B,H,3,3]."
        )

    if scale_tril.shape[1:] != (
        HORIZON_COUNT,
        POSITION_DIM,
        POSITION_DIM,
    ):
        raise ValueError(
            "Predictive Cholesky "
            "shape changed."
        )

    scale = torch.as_tensor(
        variance_scale,
        dtype=scale_tril.dtype,
        device=scale_tril.device,
    )

    if scale.shape != (
        HORIZON_COUNT,
    ):
        raise ValueError(
            "variance_scale must "
            "have shape [4]."
        )

    if not bool(
        torch.isfinite(
            scale
        ).all()
    ):
        raise ValueError(
            "Variance scale contains "
            "non-finite values."
        )

    if bool(
        torch.any(
            scale
            <
            MIN_VARIANCE_SCALE
        )
    ):
        raise ValueError(
            "Variance scale below "
            "frozen lower bound."
        )

    if bool(
        torch.any(
            scale
            >
            MAX_VARIANCE_SCALE
        )
    ):
        raise ValueError(
            "Variance scale above "
            "frozen upper bound."
        )

    return (
        scale_tril
        *
        torch.sqrt(
            scale
        )[
            None,
            :,
            None,
            None,
        ]
    )


def _as_numpy(
    value,
) -> np.ndarray:
    if torch.is_tensor(
        value
    ):
        return (
            value.detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value
    )


def _validate_q_mask(
    mahalanobis_squared,
    validity_mask,
):
    q = _as_numpy(
        mahalanobis_squared
    ).astype(
        np.float64,
        copy=False,
    )

    mask = _as_numpy(
        validity_mask
    )

    if q.ndim != 2:
        raise ValueError(
            "Mahalanobis tensor "
            "must be [N,H]."
        )

    if q.shape[1] != (
        HORIZON_COUNT
    ):
        raise ValueError(
            "Expected four horizons."
        )

    if mask.shape != q.shape:
        raise ValueError(
            "Validity-mask shape "
            "mismatch."
        )

    valid = (
        mask
        >
        0.5
    )

    if not np.all(
        np.isfinite(
            q[
                valid
            ]
        )
    ):
        raise ValueError(
            "Valid Mahalanobis values "
            "must be finite."
        )

    if np.any(
        q[
            valid
        ]
        <
        0.0
    ):
        raise ValueError(
            "Squared Mahalanobis "
            "distance cannot be negative."
        )

    if np.any(
        valid.sum(
            axis=0
        )
        <=
        0
    ):
        raise ValueError(
            "Every horizon must have "
            "at least one valid label."
        )

    return (
        q,
        valid,
    )


def scaled_mahalanobis_squared(
    mahalanobis_squared,
    variance_scale,
):
    q = _as_numpy(
        mahalanobis_squared
    ).astype(
        np.float64,
        copy=False,
    )

    scale = np.asarray(
        variance_scale,
        dtype=np.float64,
    )

    if scale.shape != (
        HORIZON_COUNT,
    ):
        raise ValueError(
            "variance_scale must "
            "have four values."
        )

    if not np.all(
        np.isfinite(
            scale
        )
    ):
        raise ValueError(
            "Non-finite variance scale."
        )

    if np.any(
        scale
        <=
        0.0
    ):
        raise ValueError(
            "Variance scale must "
            "be positive."
        )

    return (
        q
        /
        scale[
            None,
            :
        ]
    )


def reliability_metrics(
    mahalanobis_squared,
    validity_mask,
    *,
    variance_scale=(
        1.0,
        1.0,
        1.0,
        1.0,
    ),
):
    """
    Reliability definition frozen in Block4.5.

    Primary coverage ECE / coverage MAE:

        macro mean over:
            4 horizons x 5 confidence levels

        | empirical coverage - nominal coverage |

    Thus each horizon and confidence level has
    equal weight.

    Brier semantics:

        For every valid (actor,horizon,level):

            y = 1 if ground truth lies inside
                the nominal predictive ellipsoid

            Brier = (y - nominal_probability)^2

        Report mean across all valid events.
    """

    (
        q,
        valid,
    ) = _validate_q_mask(
        mahalanobis_squared,
        validity_mask,
    )

    q_scaled = (
        scaled_mahalanobis_squared(
            q,
            variance_scale,
        )
    )

    horizon_names = (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    )

    by_horizon = {}

    absolute_errors = []
    brier_sum = 0.0
    brier_count = 0

    global_inside = {
        level: 0
        for level
        in CONFIDENCE_LEVELS
    }

    global_valid = int(
        valid.sum()
    )

    for horizon in range(
        HORIZON_COUNT
    ):
        hmask = (
            valid[
                :,
                horizon
            ]
        )

        hq = (
            q_scaled[
                hmask,
                horizon
            ]
        )

        count = int(
            hq.shape[0]
        )

        level_metrics = {}

        for level in (
            CONFIDENCE_LEVELS
        ):
            threshold = float(
                CHI_SQUARE_3_THRESHOLDS[
                    level
                ]
            )

            inside = (
                hq
                <=
                threshold
            )

            inside_count = int(
                inside.sum()
            )

            empirical = float(
                inside_count
                /
                count
            )

            absolute_error = abs(
                empirical
                -
                float(level)
            )

            brier = float(
                np.mean(
                    (
                        inside.astype(
                            np.float64
                        )
                        -
                        float(level)
                    )
                    ** 2
                )
            )

            absolute_errors.append(
                absolute_error
            )

            brier_sum += (
                brier
                *
                count
            )

            brier_count += (
                count
            )

            global_inside[
                level
            ] += (
                inside_count
            )

            level_metrics[
                f"{level:.2f}"
            ] = {
                "nominal":
                    float(level),

                "empirical":
                    empirical,

                "absolute_error":
                    float(
                        absolute_error
                    ),

                "coverage_event_Brier":
                    brier,

                "inside":
                    inside_count,

                "valid":
                    count,

                "chi_square_df3_threshold":
                    threshold,
            }

        by_horizon[
            horizon_names[
                horizon
            ]
        ] = {
            "valid":
                count,

            "levels":
                level_metrics,

            "coverage_ECE":
                float(
                    np.mean(
                        [
                            value[
                                "absolute_error"
                            ]
                            for value
                            in (
                                level_metrics
                                .values()
                            )
                        ]
                    )
                ),
        }

    global_levels = {}

    for level in (
        CONFIDENCE_LEVELS
    ):
        empirical = float(
            global_inside[
                level
            ]
            /
            global_valid
        )

        global_levels[
            f"{level:.2f}"
        ] = {
            "nominal":
                float(level),

            "empirical":
                empirical,

            "absolute_error":
                abs(
                    empirical
                    -
                    float(level)
                ),

            "inside":
                int(
                    global_inside[
                        level
                    ]
                ),

            "valid":
                global_valid,
        }

    macro_ece = float(
        np.mean(
            absolute_errors
        )
    )

    coverage_event_brier = float(
        brier_sum
        /
        brier_count
    )

    return {
        "coverage_ECE_macro":
            macro_ece,

        "coverage_MAE_macro":
            macro_ece,

        "coverage_event_Brier":
            coverage_event_brier,

        "confidence_levels":
            [
                float(x)
                for x in (
                    CONFIDENCE_LEVELS
                )
            ],

        "horizon_weighting":
            "equal_macro",

        "confidence_level_weighting":
            "equal_macro",

        "by_horizon":
            by_horizon,

        "global_levels":
            global_levels,
    }


def _horizon_coverage_mae(
    q_values: np.ndarray,
    alpha: float,
) -> float:
    scaled = (
        q_values
        /
        float(alpha)
    )

    errors = []

    for level in (
        CONFIDENCE_LEVELS
    ):
        empirical = float(
            np.mean(
                scaled
                <=
                float(
                    CHI_SQUARE_3_THRESHOLDS[
                        level
                    ]
                )
            )
        )

        errors.append(
            abs(
                empirical
                -
                float(level)
            )
        )

    return float(
        np.mean(
            errors
        )
    )


def _horizon_scale_nll_term(
    q_values: np.ndarray,
    alpha: float,
) -> float:
    """
    Alpha-dependent portion of mean Gaussian NLL:

        0.5 * [
            d log(alpha)
            +
            E[q] / alpha
        ]

    Used only as deterministic tie-break.
    """

    return float(
        0.5
        *
        (
            POSITION_DIM
            *
            math.log(
                float(alpha)
            )
            +
            float(
                np.mean(
                    q_values
                )
            )
            /
            float(alpha)
        )
    )


def _candidate_scales(
    q_values: np.ndarray,
):
    candidates = {
        1.0,
    }

    # Gaussian-NLL optimum for scalar covariance
    # scaling in d dimensions.
    nll_alpha = float(
        np.mean(
            q_values
        )
        /
        POSITION_DIM
    )

    candidates.add(
        nll_alpha
    )

    coverage_alphas = []

    for level in (
        CONFIDENCE_LEVELS
    ):
        quantile = float(
            np.quantile(
                q_values,
                float(level),
                method="linear",
            )
        )

        threshold = float(
            CHI_SQUARE_3_THRESHOLDS[
                level
            ]
        )

        alpha = (
            quantile
            /
            threshold
        )

        coverage_alphas.append(
            alpha
        )

        candidates.add(
            alpha
        )

    candidates.add(
        float(
            np.median(
                coverage_alphas
            )
        )
    )

    positive = [
        max(
            float(value),
            1e-12,
        )
        for value in (
            coverage_alphas
        )
    ]

    candidates.add(
        float(
            math.exp(
                np.mean(
                    np.log(
                        positive
                    )
                )
            )
        )
    )

    # Deterministic local interpolation around
    # the principal candidate scales.
    base = sorted(
        candidates
    )

    for left, right in zip(
        base[:-1],
        base[1:],
    ):
        if (
            left > 0.0
            and
            right > 0.0
        ):
            candidates.add(
                math.sqrt(
                    left
                    *
                    right
                )
            )

    clipped = {
        min(
            MAX_VARIANCE_SCALE,
            max(
                MIN_VARIANCE_SCALE,
                float(value),
            ),
        )
        for value in candidates
        if math.isfinite(
            float(value)
        )
    }

    # Identity MUST remain available.
    clipped.add(
        1.0
    )

    return tuple(
        sorted(
            clipped
        )
    )


def fit_per_horizon_covariance_scale(
    mahalanobis_squared,
    validity_mask,
):
    """
    Fit on calibration partition only.

    Primary objective:
        per-horizon mean absolute coverage error
        over 50/80/90/95/99%.

    Tie-break:
        scalar-dependent Gaussian NLL term.

    Identity alpha=1 is always a candidate, so
    calibration-set coverage MAE cannot be worse
    than raw for any horizon.
    """

    (
        q,
        valid,
    ) = _validate_q_mask(
        mahalanobis_squared,
        validity_mask,
    )

    selected = []
    horizon_diagnostics = {}

    names = (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    )

    for horizon in range(
        HORIZON_COUNT
    ):
        values = (
            q[
                valid[
                    :,
                    horizon
                ],
                horizon
            ]
        )

        candidates = (
            _candidate_scales(
                values
            )
        )

        scored = []

        for alpha in candidates:
            coverage_mae = (
                _horizon_coverage_mae(
                    values,
                    alpha,
                )
            )

            nll_term = (
                _horizon_scale_nll_term(
                    values,
                    alpha,
                )
            )

            scored.append(
                (
                    coverage_mae,
                    nll_term,
                    abs(
                        math.log(
                            alpha
                        )
                    ),
                    alpha,
                )
            )

        scored.sort()

        (
            best_mae,
            best_nll_term,
            _,
            best_alpha,
        ) = scored[0]

        raw_mae = (
            _horizon_coverage_mae(
                values,
                1.0,
            )
        )

        if (
            best_mae
            >
            raw_mae
            +
            1e-15
        ):
            raise RuntimeError(
                "Calibration candidate "
                "selection violated "
                "non-worsening property."
            )

        selected.append(
            float(
                best_alpha
            )
        )

        horizon_diagnostics[
            names[
                horizon
            ]
        ] = {
            "valid":
                int(
                    values.shape[0]
                ),

            "candidate_count":
                len(
                    candidates
                ),

            "raw_coverage_MAE":
                float(
                    raw_mae
                ),

            "selected_coverage_MAE":
                float(
                    best_mae
                ),

            "selected_NLL_tiebreak_term":
                float(
                    best_nll_term
                ),

            "selected_variance_scale":
                float(
                    best_alpha
                ),

            "selected_std_scale":
                math.sqrt(
                    float(
                        best_alpha
                    )
                ),
        }

    calibrator = (
        CovarianceScaleCalibration(
            variance_scale=tuple(
                selected
            )
        )
    )

    raw_metrics = (
        reliability_metrics(
            q,
            valid,
        )
    )

    calibrated_metrics = (
        reliability_metrics(
            q,
            valid,
            variance_scale=(
                calibrator
                .variance_scale
            ),
        )
    )

    if (
        calibrated_metrics[
            "coverage_ECE_macro"
        ]
        >
        raw_metrics[
            "coverage_ECE_macro"
        ]
        +
        1e-15
    ):
        raise RuntimeError(
            "Macro calibration ECE "
            "became worse despite "
            "identity fallback."
        )

    return {
        "calibrator":
            calibrator,

        "horizon_diagnostics":
            horizon_diagnostics,

        "raw_reliability":
            raw_metrics,

        "calibrated_reliability":
            calibrated_metrics,

        "coverage_ECE_nonworsening":
            True,
    }

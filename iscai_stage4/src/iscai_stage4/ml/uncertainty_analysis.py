from __future__ import annotations

import math

import numpy as np
import torch

from .ablation_runtime import (
    ABLATION_FULL,
    apply_normalized_ablation,
)

from .gaussian_math import (
    covariance_from_scale_tril,
    denormalize_gaussian,
    gaussian_nll_per_horizon,
)


R_XX = 6
R_XY = 7
R_XZ = 8
R_YY = 9
R_YZ = 10
R_ZZ = 11

OBSERVED_MASK_INDEX = 12

EXPECTED_R_INDICES = (
    R_XX,
    R_XY,
    R_XZ,
    R_YY,
    R_YZ,
    R_ZZ,
)

HORIZON_NAMES = (
    "0.1",
    "0.3",
    "0.5",
    "1.0",
)


def latest_observed_indices(
    target_history,
):
    """
    target_history:
        raw cache array [N,11,14]
    """

    value = np.asarray(
        target_history
    )

    if (
        value.ndim != 3
        or
        value.shape[1:] != (
            11,
            14,
        )
    ):
        raise ValueError(
            "Expected raw target history [N,11,14]."
        )

    observed = (
        value[
            :,
            :,
            OBSERVED_MASK_INDEX
        ]
        >
        0.5
    )

    if not np.all(
        observed.any(
            axis=1
        )
    ):
        raise ValueError(
            "At least one sample has "
            "no causal observation."
        )

    grid = np.arange(
        value.shape[1],
        dtype=np.int64,
    )

    masked = np.where(
        observed,
        grid[
            None,
            :
        ],
        -1,
    )

    return masked.max(
        axis=1
    )


def latest_measurement_covariance(
    target_history,
):
    value = np.asarray(
        target_history,
        dtype=np.float64,
    )

    indices = latest_observed_indices(
        value
    )

    batch = np.arange(
        value.shape[0],
        dtype=np.int64,
    )

    latest = value[
        batch,
        indices,
        :
    ]

    covariance = np.zeros(
        (
            value.shape[0],
            3,
            3,
        ),
        dtype=np.float64,
    )

    covariance[
        :,
        0,
        0
    ] = latest[
        :,
        R_XX
    ]

    covariance[
        :,
        0,
        1
    ] = latest[
        :,
        R_XY
    ]

    covariance[
        :,
        1,
        0
    ] = latest[
        :,
        R_XY
    ]

    covariance[
        :,
        0,
        2
    ] = latest[
        :,
        R_XZ
    ]

    covariance[
        :,
        2,
        0
    ] = latest[
        :,
        R_XZ
    ]

    covariance[
        :,
        1,
        1
    ] = latest[
        :,
        R_YY
    ]

    covariance[
        :,
        1,
        2
    ] = latest[
        :,
        R_YZ
    ]

    covariance[
        :,
        2,
        1
    ] = latest[
        :,
        R_YZ
    ]

    covariance[
        :,
        2,
        2
    ] = latest[
        :,
        R_ZZ
    ]

    if not np.all(
        np.isfinite(
            covariance
        )
    ):
        raise ValueError(
            "Latest measurement covariance "
            "contains non-finite values."
        )

    return covariance


def latest_measurement_sigma_rms_m(
    target_history,
):
    covariance = (
        latest_measurement_covariance(
            target_history
        )
    )

    trace = np.trace(
        covariance,
        axis1=-2,
        axis2=-1,
    )

    if np.any(
        trace < -1e-9
    ):
        raise ValueError(
            "Measurement covariance has "
            "negative trace."
        )

    sigma = np.sqrt(
        np.maximum(
            trace,
            0.0,
        )
        /
        3.0
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    return {
        "sigma_rms_m":
            sigma,

        "minimum_eigenvalue":
            float(
                np.min(
                    eigenvalues
                )
            ),

        "PSD_violation_count":
            int(
                np.sum(
                    np.min(
                        eigenvalues,
                        axis=-1,
                    )
                    <
                    -1e-5
                )
            ),
    }


def _average_rank(
    values,
):
    """
    Dependency-free average ranks,
    equivalent to rankdata(method='average').
    """

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.ndim != 1:
        raise ValueError(
            "Rank input must be 1-D."
        )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    sorted_x = x[
        order
    ]

    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    start = 0

    while start < len(x):
        stop = start + 1

        while (
            stop < len(x)
            and
            sorted_x[
                stop
            ]
            ==
            sorted_x[
                start
            ]
        ):
            stop += 1

        rank = (
            start
            +
            stop
            -
            1
        ) / 2.0

        ranks[
            order[
                start:stop
            ]
        ] = rank

        start = stop

    return ranks


def pearson_correlation(
    x,
    y,
):
    a = np.asarray(
        x,
        dtype=np.float64,
    )

    b = np.asarray(
        y,
        dtype=np.float64,
    )

    if (
        a.ndim != 1
        or
        b.ndim != 1
        or
        a.shape != b.shape
    ):
        raise ValueError(
            "Correlation inputs must "
            "have equal 1-D shape."
        )

    if len(a) < 2:
        return float("nan")

    a_centered = (
        a - np.mean(a)
    )

    b_centered = (
        b - np.mean(b)
    )

    denominator = math.sqrt(
        float(
            np.sum(
                a_centered ** 2
            )
            *
            np.sum(
                b_centered ** 2
            )
        )
    )

    if denominator == 0.0:
        return 0.0

    return float(
        np.sum(
            a_centered
            *
            b_centered
        )
        /
        denominator
    )


def spearman_correlation(
    x,
    y,
):
    return pearson_correlation(
        _average_rank(
            x
        ),
        _average_rank(
            y
        ),
    )


def balanced_rank_bins(
    values,
    *,
    bin_count=5,
):
    """
    Equal-count rank bins.

    Avoids empty uncertainty bins when
    physical sigma values contain ties.
    """

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.ndim != 1:
        raise ValueError(
            "Bin input must be 1-D."
        )

    if len(x) < bin_count:
        raise ValueError(
            "Not enough samples for bins."
        )

    order = np.argsort(
        x,
        kind="mergesort",
    )

    bins = np.empty(
        len(x),
        dtype=np.int64,
    )

    for rank, index in enumerate(
        order
    ):
        bin_id = min(
            bin_count - 1,
            (
                rank
                *
                bin_count
            )
            //
            len(x),
        )

        bins[
            index
        ] = bin_id

    return bins


def scale_physical_R_in_normalized_history(
    history,
    *,
    feature_mean,
    feature_std,
    variance_scale,
    covariance_indices=EXPECTED_R_INDICES,
):
    """
    The input is already standardized.

    For a covariance feature r:

        n = (r - mu) / sigma

    Physical covariance scaling r' = c*r gives:

        n' = c*n + (c-1)*mu/sigma

    Missing history rows remain exactly zero.
    """

    c = float(
        variance_scale
    )

    if (
        not math.isfinite(c)
        or
        c <= 0.0
    ):
        raise ValueError(
            "Physical R scale must "
            "be finite and positive."
        )

    value = history.clone()

    indices = tuple(
        int(index)
        for index in covariance_indices
    )

    if indices != EXPECTED_R_INDICES:
        raise ValueError(
            "Frozen R_t feature indices changed."
        )

    mean = feature_mean[
        list(
            indices
        )
    ].to(
        dtype=value.dtype,
        device=value.device,
    )

    std = feature_std[
        list(
            indices
        )
    ].to(
        dtype=value.dtype,
        device=value.device,
    )

    observed = (
        value[
            ...,
            OBSERVED_MASK_INDEX:
            OBSERVED_MASK_INDEX + 1
        ]
        >
        0.5
    ).to(
        dtype=value.dtype
    )

    current = value[
        ...,
        list(
            indices
        )
    ]

    scaled = (
        c
        *
        current
        +
        (
            c - 1.0
        )
        *
        (
            mean
            /
            std
        )
    )

    value[
        ...,
        list(
            indices
        )
    ] = (
        scaled
        *
        observed
    )

    return value


def apply_physical_R_scale(
    target,
    neighbors,
    *,
    feature_mean,
    feature_std,
    variance_scale,
):
    return (
        scale_physical_R_in_normalized_history(
            target,
            feature_mean=feature_mean,
            feature_std=feature_std,
            variance_scale=variance_scale,
        ),

        scale_physical_R_in_normalized_history(
            neighbors,
            feature_mean=feature_mean,
            feature_std=feature_std,
            variance_scale=variance_scale,
        ),
    )


def predictive_sigma_rms_m(
    scale_tril_metric,
):
    covariance = (
        covariance_from_scale_tril(
            scale_tril_metric
        )
    )

    diagonal = torch.diagonal(
        covariance,
        dim1=-2,
        dim2=-1,
    )

    trace = diagonal.sum(
        dim=-1
    )

    return torch.sqrt(
        trace.clamp_min(
            0.0
        )
        /
        3.0
    )


def evaluate_gaussian_condition(
    model,
    arrays,
    normalizer,
    *,
    device,
    ablation_mode=ABLATION_FULL,
    R_variance_scale=1.0,
    covariance_indices=EXPECTED_R_INDICES,
    batch_size=1024,
    return_sample_arrays=False,
):
    n = int(
        arrays[
            "target"
        ].shape[0]
    )

    nll_sum = 0.0
    valid_count = 0

    planar_sum = 0.0

    horizon_nll_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_planar_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_sigma_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_count = np.zeros(
        4,
        dtype=np.int64,
    )

    sigma_parts = []
    planar_parts = []
    validity_parts = []

    SPD_failures = 0

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            n,
            batch_size,
        ):
            stop = min(
                n,
                start + batch_size,
            )

            indices = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            batch = normalizer.prepare(
                arrays,
                indices,
                device=device,
            )

            (
                target,
                neighbors,
                neighbor_mask,
                map_context,
            ) = apply_normalized_ablation(
                batch[
                    "target"
                ],
                batch[
                    "neighbors"
                ],
                batch[
                    "neighbor_mask"
                ],
                batch[
                    "map_context"
                ],
                mode=ablation_mode,
                covariance_indices=(
                    covariance_indices
                ),
            )

            if (
                ablation_mode
                ==
                ABLATION_FULL
                and
                float(
                    R_variance_scale
                )
                !=
                1.0
            ):
                (
                    target,
                    neighbors,
                ) = apply_physical_R_scale(
                    target,
                    neighbors,
                    feature_mean=(
                        normalizer.feature_mean
                    ),
                    feature_std=(
                        normalizer.feature_std
                    ),
                    variance_scale=(
                        R_variance_scale
                    ),
                )

            output = model(
                target,
                neighbors,
                neighbor_mask,
                map_context,
            )

            (
                mean_metric,
                scale_metric,
            ) = denormalize_gaussian(
                output.mean,
                output.scale_tril,
                normalizer.label_mean,
                normalizer.label_std,
            )

            truth = batch[
                "true_displacement"
            ]

            valid = (
                batch[
                    "future_mask"
                ]
                >
                0.5
            )

            nll = gaussian_nll_per_horizon(
                mean_metric,
                scale_metric,
                truth,
            )

            planar = torch.sqrt(
                (
                    mean_metric[
                        :,
                        :,
                        0
                    ]
                    -
                    truth[
                        :,
                        :,
                        0
                    ]
                ).square()
                +
                (
                    mean_metric[
                        :,
                        :,
                        1
                    ]
                    -
                    truth[
                        :,
                        :,
                        1
                    ]
                ).square()
            )

            sigma = predictive_sigma_rms_m(
                scale_metric
            )

            if not bool(
                torch.isfinite(
                    nll[
                        valid
                    ]
                ).all()
            ):
                raise RuntimeError(
                    "Non-finite Gaussian NLL."
                )

            covariance = (
                covariance_from_scale_tril(
                    scale_metric
                )
            )

            (
                _,
                info,
            ) = torch.linalg.cholesky_ex(
                covariance
            )

            SPD_failures += int(
                (
                    info != 0
                ).sum().item()
            )

            nll_sum += float(
                nll[
                    valid
                ].sum().item()
            )

            planar_sum += float(
                planar[
                    valid
                ].sum().item()
            )

            valid_count += int(
                valid.sum().item()
            )

            for horizon in range(4):
                mask = valid[
                    :,
                    horizon
                ]

                count = int(
                    mask.sum().item()
                )

                if count <= 0:
                    continue

                horizon_count[
                    horizon
                ] += count

                horizon_nll_sum[
                    horizon
                ] += float(
                    nll[
                        :,
                        horizon
                    ][
                        mask
                    ].sum().item()
                )

                horizon_planar_sum[
                    horizon
                ] += float(
                    planar[
                        :,
                        horizon
                    ][
                        mask
                    ].sum().item()
                )

                horizon_sigma_sum[
                    horizon
                ] += float(
                    sigma[
                        :,
                        horizon
                    ][
                        mask
                    ].sum().item()
                )

            if return_sample_arrays:
                sigma_parts.append(
                    sigma.detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                planar_parts.append(
                    planar.detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                validity_parts.append(
                    valid.detach()
                    .cpu()
                    .numpy()
                )

    if valid_count <= 0:
        raise RuntimeError(
            "No valid evaluation labels."
        )

    if SPD_failures != 0:
        raise RuntimeError(
            "Predictive covariance SPD failure."
        )

    result = {
        "metric_Gaussian_NLL":
            float(
                nll_sum
                /
                valid_count
            ),

        "planar_ADE_m":
            float(
                planar_sum
                /
                valid_count
            ),

        "horizon_NLL": {
            HORIZON_NAMES[h]:
                float(
                    horizon_nll_sum[h]
                    /
                    horizon_count[h]
                )
            for h in range(4)
        },

        "horizon_planar_error_m": {
            HORIZON_NAMES[h]:
                float(
                    horizon_planar_sum[h]
                    /
                    horizon_count[h]
                )
            for h in range(4)
        },

        "mean_predictive_sigma_RMS_m": {
            HORIZON_NAMES[h]:
                float(
                    horizon_sigma_sum[h]
                    /
                    horizon_count[h]
                )
            for h in range(4)
        },

        "valid_actor_horizon_points":
            int(
                valid_count
            ),

        "predictive_covariance_SPD_failures":
            0,
    }

    if return_sample_arrays:
        result[
            "sample_arrays"
        ] = {
            "predictive_sigma_RMS_m":
                np.concatenate(
                    sigma_parts,
                    axis=0,
                ),

            "planar_error_m":
                np.concatenate(
                    planar_parts,
                    axis=0,
                ),

            "validity":
                np.concatenate(
                    validity_parts,
                    axis=0,
                ),
        }

    return result


def uncertainty_relation_report(
    measurement_sigma,
    predictive_sigma,
    planar_error,
    validity,
    *,
    calibrated_std_scale=None,
):
    measurement = np.asarray(
        measurement_sigma,
        dtype=np.float64,
    )

    predictive = np.asarray(
        predictive_sigma,
        dtype=np.float64,
    )

    error = np.asarray(
        planar_error,
        dtype=np.float64,
    )

    valid = np.asarray(
        validity,
        dtype=bool,
    )

    if (
        predictive.shape
        !=
        error.shape
        or
        predictive.shape
        !=
        valid.shape
        or
        predictive.ndim != 2
        or
        predictive.shape[1] != 4
        or
        measurement.shape
        !=
        (
            predictive.shape[0],
        )
    ):
        raise ValueError(
            "Uncertainty-relation shape mismatch."
        )

    if calibrated_std_scale is None:
        scales = np.ones(
            4,
            dtype=np.float64,
        )

    else:
        scales = np.asarray(
            calibrated_std_scale,
            dtype=np.float64,
        )

        if scales.shape != (4,):
            raise ValueError(
                "Expected four calibration std scales."
            )

    predictive_calibrated = (
        predictive
        *
        scales[
            None,
            :
        ]
    )

    bins = balanced_rank_bins(
        measurement,
        bin_count=5,
    )

    horizons = {}

    for horizon, name in enumerate(
        HORIZON_NAMES
    ):
        mask = valid[
            :,
            horizon
        ]

        horizons[
            name
        ] = {
            "Pearson_measurement_vs_predictive_sigma":
                pearson_correlation(
                    measurement[
                        mask
                    ],
                    predictive_calibrated[
                        mask,
                        horizon
                    ],
                ),

            "Spearman_measurement_vs_predictive_sigma":
                spearman_correlation(
                    measurement[
                        mask
                    ],
                    predictive_calibrated[
                        mask,
                        horizon
                    ],
                ),

            "Pearson_measurement_vs_planar_error":
                pearson_correlation(
                    measurement[
                        mask
                    ],
                    error[
                        mask,
                        horizon
                    ],
                ),

            "Spearman_measurement_vs_planar_error":
                spearman_correlation(
                    measurement[
                        mask
                    ],
                    error[
                        mask,
                        horizon
                    ],
                ),
        }

    bin_report = {}

    for bin_id in range(5):
        selection = (
            bins
            ==
            bin_id
        )

        payload = {
            "sample_count":
                int(
                    selection.sum()
                ),

            "measurement_sigma_RMS_m": {
                "min":
                    float(
                        np.min(
                            measurement[
                                selection
                            ]
                        )
                    ),

                "mean":
                    float(
                        np.mean(
                            measurement[
                                selection
                            ]
                        )
                    ),

                "max":
                    float(
                        np.max(
                            measurement[
                                selection
                            ]
                        )
                    ),
            },

            "horizons":
                {},
        }

        for horizon, name in enumerate(
            HORIZON_NAMES
        ):
            mask = (
                selection
                &
                valid[
                    :,
                    horizon
                ]
            )

            payload[
                "horizons"
            ][
                name
            ] = {
                "valid":
                    int(
                        mask.sum()
                    ),

                "mean_calibrated_predictive_sigma_RMS_m":
                    float(
                        np.mean(
                            predictive_calibrated[
                                mask,
                                horizon
                            ]
                        )
                    ),

                "mean_planar_error_m":
                    float(
                        np.mean(
                            error[
                                mask,
                                horizon
                            ]
                        )
                    ),
            }

        bin_report[
            str(
                bin_id + 1
            )
        ] = payload

    return {
        "measurement_uncertainty_definition":
            (
                "latest causal target R_t: "
                "sqrt(trace(R_t)/3)"
            ),

        "predictive_uncertainty_definition":
            (
                "sqrt(trace(Sigma_pred)/3)"
            ),

        "predictive_sigma_uses_Block45_calibration_scale":
            (
                calibrated_std_scale
                is not None
            ),

        "correlations":
            horizons,

        "equal_count_measurement_uncertainty_bins":
            bin_report,
    }

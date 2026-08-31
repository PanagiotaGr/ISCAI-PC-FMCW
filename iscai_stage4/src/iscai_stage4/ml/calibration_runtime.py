from __future__ import annotations

from hashlib import sha256
import math

import numpy as np
import torch

from iscai_stage4.ml.calibration import (
    apply_variance_scale,
)

from iscai_stage4.ml.gaussian_gru import (
    GaussianTrajectoryGRU,
)

from iscai_stage4.ml.gaussian_math import (
    covariance_from_scale_tril,
    denormalize_gaussian,
    gaussian_nll_per_horizon,
    mahalanobis_squared,
)

from iscai_stage4.ml.training_utils import (
    actor_class_id,
)


POSITION_DIM = 3
HORIZON_COUNT = 4


class CalibrationNormalizer:
    """
    Frozen Block4.3 fit-only normalization.

    Calibration data never contributes
    normalization statistics.
    """

    def __init__(
        self,
        statistics,
        *,
        device,
    ):
        self.feature_mean = torch.tensor(
            statistics[
                "continuous_feature_mean"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.feature_std = torch.tensor(
            statistics[
                "continuous_feature_std"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.map_mean = torch.tensor(
            statistics[
                "map_mean"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.map_std = torch.tensor(
            statistics[
                "map_std"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.label_mean = torch.tensor(
            statistics[
                "label_displacement_mean"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.label_std = torch.tensor(
            statistics[
                "label_displacement_std"
            ],
            dtype=torch.float32,
            device=device,
        )

        for name, value in (
            (
                "feature_std",
                self.feature_std,
            ),
            (
                "map_std",
                self.map_std,
            ),
            (
                "label_std",
                self.label_std,
            ),
        ):
            if bool(
                torch.any(
                    value <= 0.0
                )
            ):
                raise ValueError(
                    f"{name} must be positive."
                )

    @staticmethod
    def _latest_position(
        target,
    ):
        observed = (
            target[:, :, 12]
            >
            0.5
        )

        if not bool(
            observed.any(
                dim=1
            ).all()
        ):
            raise ValueError(
                "Target without causal "
                "observation."
            )

        grid = torch.arange(
            target.shape[1],
            device=target.device,
            dtype=torch.long,
        )

        grid = (
            grid[None, :]
            .expand_as(
                observed
            )
        )

        indices = torch.where(
            observed,
            grid,
            torch.full_like(
                grid,
                -1,
            ),
        ).max(
            dim=1
        ).values

        batch = torch.arange(
            target.shape[0],
            device=target.device,
        )

        return target[
            batch,
            indices,
            :3,
        ]

    def prepare(
        self,
        arrays,
        *,
        device,
    ):
        target = torch.from_numpy(
            np.asarray(
                arrays["target"],
                dtype=np.float32,
            )
        ).to(device)

        neighbors = torch.from_numpy(
            np.asarray(
                arrays["neighbors"],
                dtype=np.float32,
            )
        ).to(device)

        neighbor_mask = torch.from_numpy(
            np.asarray(
                arrays["neighbor_mask"],
                dtype=np.float32,
            )
        ).to(device)

        map_context = torch.from_numpy(
            np.asarray(
                arrays["map_context"],
                dtype=np.float32,
            )
        ).to(device)

        future = torch.from_numpy(
            np.asarray(
                arrays["future"],
                dtype=np.float32,
            )
        ).to(device)

        future_mask = torch.from_numpy(
            np.asarray(
                arrays["future_mask"],
                dtype=np.float32,
            )
        ).to(device)

        class_id = torch.from_numpy(
            np.asarray(
                arrays["class_id"],
                dtype=np.int64,
            )
        ).to(device)

        target_observed = (
            target[
                :,
                :,
                12:13
            ]
        )

        target_cont = (
            (
                target[
                    :,
                    :,
                    :12
                ]
                -
                self.feature_mean
            )
            /
            self.feature_std
        )

        target_cont = (
            target_cont
            *
            target_observed
        )

        target_normalized = torch.cat(
            (
                target_cont,
                target[
                    :,
                    :,
                    12:
                ],
            ),
            dim=-1,
        )

        neighbor_observed = (
            neighbors[
                :,
                :,
                :,
                12:13
            ]
            *
            neighbor_mask[
                :,
                :,
                None,
                None
            ]
        )

        neighbor_cont = (
            (
                neighbors[
                    :,
                    :,
                    :,
                    :12
                ]
                -
                self.feature_mean
            )
            /
            self.feature_std
        )

        neighbor_cont = (
            neighbor_cont
            *
            neighbor_observed
        )

        neighbors_normalized = torch.cat(
            (
                neighbor_cont,

                neighbors[
                    :,
                    :,
                    :,
                    12:
                ]
                *
                neighbor_mask[
                    :,
                    :,
                    None,
                    None
                ],
            ),
            dim=-1,
        )

        map_normalized = (
            (
                map_context
                -
                self.map_mean
            )
            /
            self.map_std
        )

        origin = self._latest_position(
            target
        )

        displacement = (
            future
            -
            origin[
                :,
                None,
                :
            ]
        )

        return {
            "target":
                target_normalized,

            "neighbors":
                neighbors_normalized,

            "neighbor_mask":
                neighbor_mask,

            "map_context":
                map_normalized,

            "true_displacement":
                displacement,

            "future_mask":
                future_mask,

            "class_id":
                class_id,
        }


def build_gaussian_model(
    configuration,
    *,
    device,
):
    model_config = (
        configuration[
            "model"
        ]
    )

    return GaussianTrajectoryGRU(
        target_hidden_dim=int(
            model_config[
                "target_hidden_dim"
            ]
        ),
        neighbor_hidden_dim=int(
            model_config[
                "neighbor_hidden_dim"
            ]
        ),
        map_hidden_dim=int(
            model_config[
                "map_hidden_dim"
            ]
        ),
        fusion_hidden_dim=int(
            model_config[
                "fusion_hidden_dim"
            ]
        ),
        use_neighbors=True,
        use_map=True,
    ).to(device)


def samples_to_arrays(
    samples,
):
    count = len(samples)

    if count == 0:
        return {
            "target":
                np.zeros(
                    (
                        0,
                        11,
                        14,
                    ),
                    dtype=np.float32,
                ),

            "neighbors":
                np.zeros(
                    (
                        0,
                        8,
                        11,
                        14,
                    ),
                    dtype=np.float32,
                ),

            "neighbor_mask":
                np.zeros(
                    (
                        0,
                        8,
                    ),
                    dtype=np.float32,
                ),

            "map_context":
                np.zeros(
                    (
                        0,
                        10,
                    ),
                    dtype=np.float32,
                ),

            "future":
                np.zeros(
                    (
                        0,
                        4,
                        3,
                    ),
                    dtype=np.float32,
                ),

            "future_mask":
                np.zeros(
                    (
                        0,
                        4,
                    ),
                    dtype=np.float32,
                ),

            "class_id":
                np.zeros(
                    (0,),
                    dtype=np.int64,
                ),
        }

    return {
        "target":
            np.asarray(
                [
                    sample
                    .model_input
                    .target_history
                    for sample in samples
                ],
                dtype=np.float32,
            ),

        "neighbors":
            np.asarray(
                [
                    sample
                    .model_input
                    .neighbor_histories
                    for sample in samples
                ],
                dtype=np.float32,
            ),

        "neighbor_mask":
            np.asarray(
                [
                    sample
                    .model_input
                    .neighbor_mask
                    for sample in samples
                ],
                dtype=np.float32,
            ),

        "map_context":
            np.asarray(
                [
                    sample
                    .model_input
                    .map_context
                    for sample in samples
                ],
                dtype=np.float32,
            ),

        "future":
            np.asarray(
                [
                    sample
                    .future_label
                    .positions_H0_m
                    for sample in samples
                ],
                dtype=np.float32,
            ),

        "future_mask":
            np.asarray(
                [
                    sample
                    .future_label
                    .valid_mask
                    for sample in samples
                ],
                dtype=np.float32,
            ),

        "class_id":
            np.asarray(
                [
                    actor_class_id(
                        sample.actor_class
                    )
                    for sample in samples
                ],
                dtype=np.int64,
            ),
    }


def prediction_sha256(
    mean,
    scale_tril,
):
    digest = sha256()

    for tensor in (
        mean,
        scale_tril,
    ):
        value = (
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        digest.update(
            value.tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def infer_scene_statistics(
    model,
    samples,
    normalizer,
    *,
    device,
):
    arrays = samples_to_arrays(
        samples
    )

    count = int(
        arrays[
            "target"
        ].shape[0]
    )

    if count == 0:
        return {
            "mahalanobis_squared":
                np.zeros(
                    (
                        0,
                        4,
                    ),
                    dtype=np.float32,
                ),

            "validity":
                np.zeros(
                    (
                        0,
                        4,
                    ),
                    dtype=np.float32,
                ),

            "raw_metric_nll":
                np.zeros(
                    (
                        0,
                        4,
                    ),
                    dtype=np.float32,
                ),

            "planar_error_m":
                np.zeros(
                    (
                        0,
                        4,
                    ),
                    dtype=np.float32,
                ),

            "class_id":
                np.zeros(
                    (0,),
                    dtype=np.int64,
                ),

            "prediction_sha256":
                sha256(
                    b""
                ).hexdigest(),
        }

    batch = normalizer.prepare(
        arrays,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        output = model(
            batch["target"],
            batch["neighbors"],
            batch[
                "neighbor_mask"
            ],
            batch[
                "map_context"
            ],
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

        q = mahalanobis_squared(
            mean_metric,
            scale_metric,
            truth,
        )

        raw_nll = (
            gaussian_nll_per_horizon(
                mean_metric,
                scale_metric,
                truth,
            )
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

        valid = batch[
            "future_mask"
        ]

        valid_bool = (
            valid
            >
            0.5
        )

        if not bool(
            torch.isfinite(
                q[
                    valid_bool
                ]
            ).all()
        ):
            raise RuntimeError(
                "Non-finite calibration "
                "Mahalanobis values."
            )

        if not bool(
            torch.isfinite(
                raw_nll[
                    valid_bool
                ]
            ).all()
        ):
            raise RuntimeError(
                "Non-finite calibration "
                "raw NLL values."
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

        if bool(
            torch.any(
                info != 0
            )
        ):
            raise RuntimeError(
                "Raw Gaussian calibration "
                "inference produced "
                "non-SPD covariance."
            )

        prediction_hash = (
            prediction_sha256(
                output.mean,
                output.scale_tril,
            )
        )

    return {
        "mahalanobis_squared":
            q.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "validity":
            valid.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "raw_metric_nll":
            raw_nll.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "planar_error_m":
            planar.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            ),

        "class_id":
            batch[
                "class_id"
            ].detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            ),

        "prediction_sha256":
            prediction_hash,
    }


def calibrated_nll_per_horizon(
    raw_metric_nll,
    mahalanobis_squared_raw,
    variance_scale,
):
    """
    Exact NLL adjustment for:

        Sigma_cal = alpha * Sigma_raw

    in d=3:

      NLL_cal =
        NLL_raw
        + 0.5 * [
            3 log(alpha)
            + q_raw * (1/alpha - 1)
          ]
    """

    raw = np.asarray(
        raw_metric_nll,
        dtype=np.float64,
    )

    q = np.asarray(
        mahalanobis_squared_raw,
        dtype=np.float64,
    )

    alpha = np.asarray(
        variance_scale,
        dtype=np.float64,
    )

    if raw.shape != q.shape:
        raise ValueError(
            "raw NLL/q shape mismatch."
        )

    if raw.ndim != 2:
        raise ValueError(
            "Expected [N,H]."
        )

    if raw.shape[1] != 4:
        raise ValueError(
            "Expected four horizons."
        )

    if alpha.shape != (4,):
        raise ValueError(
            "Expected four "
            "variance scales."
        )

    if np.any(
        alpha <= 0.0
    ):
        raise ValueError(
            "Variance scales "
            "must be positive."
        )

    adjustment = (
        0.5
        *
        (
            POSITION_DIM
            *
            np.log(
                alpha
            )[
                None,
                :
            ]
            +
            q
            *
            (
                1.0
                /
                alpha[
                    None,
                    :
                ]
                -
                1.0
            )
        )
    )

    return (
        raw
        +
        adjustment
    )


def masked_nll_metrics(
    nll,
    validity,
):
    value = np.asarray(
        nll,
        dtype=np.float64,
    )

    mask = (
        np.asarray(
            validity
        )
        >
        0.5
    )

    if value.shape != mask.shape:
        raise ValueError(
            "NLL/mask shape mismatch."
        )

    if (
        value.ndim != 2
        or
        value.shape[1] != 4
    ):
        raise ValueError(
            "Expected NLL [N,4]."
        )

    if not np.all(
        np.isfinite(
            value[
                mask
            ]
        )
    ):
        raise ValueError(
            "Valid NLL values "
            "must be finite."
        )

    horizon_names = (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    )

    by_horizon = {}

    for horizon, name in enumerate(
        horizon_names
    ):
        hmask = mask[
            :,
            horizon
        ]

        if not np.any(
            hmask
        ):
            raise ValueError(
                f"No valid labels "
                f"at horizon {name}."
            )

        by_horizon[
            name
        ] = float(
            np.mean(
                value[
                    hmask,
                    horizon
                ]
            )
        )

    return {
        "metric_Gaussian_NLL":
            float(
                np.mean(
                    value[
                        mask
                    ]
                )
            ),

        "by_horizon":
            by_horizon,

        "valid_actor_horizon_points":
            int(
                mask.sum()
            ),
    }


def masked_planar_ade(
    planar_error,
    validity,
):
    error = np.asarray(
        planar_error,
        dtype=np.float64,
    )

    mask = (
        np.asarray(
            validity
        )
        >
        0.5
    )

    if error.shape != mask.shape:
        raise ValueError(
            "Planar-error/mask "
            "shape mismatch."
        )

    if not np.any(
        mask
    ):
        raise ValueError(
            "No valid planar errors."
        )

    horizon_names = (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    )

    return {
        "planar_ADE_m":
            float(
                np.mean(
                    error[
                        mask
                    ]
                )
            ),

        "horizon_planar_error_m": {
            name:
                float(
                    np.mean(
                        error[
                            mask[
                                :,
                                horizon
                            ],
                            horizon
                        ]
                    )
                )
            for horizon, name
            in enumerate(
                horizon_names
            )
        },
    }


def probe_calibrated_output(
    model,
    samples,
    normalizer,
    variance_scale,
    *,
    device,
):
    arrays = samples_to_arrays(
        samples
    )

    if (
        arrays[
            "target"
        ].shape[0]
        ==
        0
    ):
        raise ValueError(
            "Probe requires "
            "non-empty samples."
        )

    batch = normalizer.prepare(
        arrays,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        output = model(
            batch["target"],
            batch["neighbors"],
            batch[
                "neighbor_mask"
            ],
            batch[
                "map_context"
            ],
        )

        raw_mean = output.mean.clone()

        calibrated_scale = (
            apply_variance_scale(
                output.scale_tril,
                variance_scale,
            )
        )

        # Calibration has no mean operation.
        calibrated_mean = (
            output.mean
        )

        mean_exact = torch.equal(
            raw_mean,
            calibrated_mean,
        )

        (
            _,
            calibrated_scale_metric,
        ) = denormalize_gaussian(
            calibrated_mean,
            calibrated_scale,
            normalizer.label_mean,
            normalizer.label_std,
        )

        covariance = (
            covariance_from_scale_tril(
                calibrated_scale_metric
            )
        )

        (
            _,
            info,
        ) = torch.linalg.cholesky_ex(
            covariance
        )

        spd = bool(
            torch.all(
                info == 0
            )
        )

    return {
        "mean_exactly_unchanged":
            mean_exact,

        "calibrated_covariance_SPD":
            spd,
    }

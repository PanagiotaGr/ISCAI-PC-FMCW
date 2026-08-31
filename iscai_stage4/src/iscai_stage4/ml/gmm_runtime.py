from __future__ import annotations

from collections import Counter
from hashlib import sha256

import numpy as np
import torch

from .gmm_math import (
    gmm_joint_nll_per_sample,
    map_component_mean,
    mixture_mean,
)


HORIZON_NAMES = (
    "0.1",
    "0.3",
    "0.5",
    "1.0",
)


class GMMNormalizer:
    """
    Frozen Block4.3 fit-only normalization.

    No development/formal/calibration statistics
    are introduced here.
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

        for name, tensor in (
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
                    tensor <= 0.0
                )
            ):
                raise ValueError(
                    f"{name} must be positive."
                )

    @staticmethod
    def latest_position(
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
                "Target has no causal observation."
            )

        steps = torch.arange(
            target.shape[1],
            device=target.device,
            dtype=torch.long,
        )

        steps = steps[
            None,
            :
        ].expand_as(
            observed
        )

        indices = torch.where(
            observed,
            steps,
            torch.full_like(
                steps,
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
        indices,
        *,
        device,
    ):
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        target = torch.from_numpy(
            np.asarray(
                arrays["target"][indices],
                dtype=np.float32,
            )
        ).to(device)

        neighbors = torch.from_numpy(
            np.asarray(
                arrays["neighbors"][indices],
                dtype=np.float32,
            )
        ).to(device)

        neighbor_mask = torch.from_numpy(
            np.asarray(
                arrays[
                    "neighbor_mask"
                ][indices],
                dtype=np.float32,
            )
        ).to(device)

        map_context = torch.from_numpy(
            np.asarray(
                arrays[
                    "map_context"
                ][indices],
                dtype=np.float32,
            )
        ).to(device)

        future = torch.from_numpy(
            np.asarray(
                arrays["future"][indices],
                dtype=np.float32,
            )
        ).to(device)

        future_mask = torch.from_numpy(
            np.asarray(
                arrays[
                    "future_mask"
                ][indices],
                dtype=np.float32,
            )
        ).to(device)

        class_id = torch.from_numpy(
            np.asarray(
                arrays[
                    "class_id"
                ][indices],
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

        origin = self.latest_position(
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

        label_normalized = (
            (
                displacement
                -
                self.label_mean[
                    None,
                    :,
                    :
                ]
            )
            /
            self.label_std[
                None,
                :,
                :
            ]
        )

        label_normalized = (
            label_normalized
            *
            future_mask[
                :,
                :,
                None
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

            "label_normalized":
                label_normalized,

            "true_displacement":
                displacement,

            "future_mask":
                future_mask,

            "class_id":
                class_id,
        }


def denormalize_gmm(
    means_normalized,
    scale_tril_normalized,
    label_mean,
    label_std,
):
    """
    means:
        [B,K,H,3]

    scale:
        [B,K,H,3,3]
    """

    if means_normalized.ndim != 4:
        raise ValueError(
            "GMM means must be [B,K,H,3]."
        )

    if scale_tril_normalized.ndim != 5:
        raise ValueError(
            "GMM Cholesky must be [B,K,H,3,3]."
        )

    mean_metric = (
        means_normalized
        *
        label_std[
            None,
            None,
            :,
            :
        ]
        +
        label_mean[
            None,
            None,
            :,
            :
        ]
    )

    scale_metric = (
        scale_tril_normalized
        *
        label_std[
            None,
            None,
            :,
            :,
            None
        ]
    )

    return (
        mean_metric,
        scale_metric,
    )


def prediction_sha256(
    mixture_logits,
    means,
    scale_tril,
):
    digest = sha256()

    for tensor in (
        mixture_logits,
        means,
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


def evaluate_gmm(
    model,
    arrays,
    normalizer,
    *,
    device,
    class_id_to_name,
    batch_size=1024,
):
    model.eval()

    sample_count = int(
        arrays[
            "target"
        ].shape[0]
    )

    joint_nll_sum = 0.0
    active_samples = 0

    mixture_mean_error_sum = 0.0
    map_error_sum = 0.0
    valid_horizon_points = 0

    horizon_mixture_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_map_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_valid = np.zeros(
        4,
        dtype=np.int64,
    )

    minade_sum = 0.0
    minade_count = 0

    minfde_sum = 0.0
    minfde_count = 0

    entropy_sum = 0.0
    effective_mode_sum = 0.0

    map_mode_counts = Counter()

    probability_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    pairwise_1s_sum = 0.0
    pairwise_1s_count = 0

    class_error_sum = Counter()
    class_valid = Counter()

    spd_failures = 0

    with torch.inference_mode():
        for start in range(
            0,
            sample_count,
            batch_size,
        ):
            stop = min(
                sample_count,
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
                means_metric,
                scale_metric,
            ) = denormalize_gmm(
                output.means,
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

            active = valid.any(
                dim=-1
            )

            active_count = int(
                active.sum().item()
            )

            if active_count <= 0:
                continue

            sample_nll = (
                gmm_joint_nll_per_sample(
                    output.mixture_logits,
                    means_metric,
                    scale_metric,
                    truth,
                    batch[
                        "future_mask"
                    ],
                )
            )

            if not bool(
                torch.isfinite(
                    sample_nll[
                        active
                    ]
                ).all()
            ):
                raise RuntimeError(
                    "Non-finite development GMM NLL."
                )

            joint_nll_sum += float(
                sample_nll[
                    active
                ].sum().item()
            )

            active_samples += (
                active_count
            )

            weighted_mean = mixture_mean(
                output.mixture_logits,
                means_metric,
            )

            map_mean = map_component_mean(
                output.mixture_logits,
                means_metric,
            )

            mixture_planar = torch.sqrt(
                (
                    weighted_mean[
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
                    weighted_mean[
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

            map_planar = torch.sqrt(
                (
                    map_mean[
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
                    map_mean[
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

            mixture_mean_error_sum += float(
                mixture_planar[
                    valid
                ].sum().item()
            )

            map_error_sum += float(
                map_planar[
                    valid
                ].sum().item()
            )

            valid_horizon_points += int(
                valid.sum().item()
            )

            for horizon in range(4):
                hmask = valid[
                    :,
                    horizon
                ]

                hcount = int(
                    hmask.sum().item()
                )

                if hcount <= 0:
                    continue

                horizon_valid[
                    horizon
                ] += hcount

                horizon_mixture_sum[
                    horizon
                ] += float(
                    mixture_planar[
                        :,
                        horizon
                    ][
                        hmask
                    ].sum().item()
                )

                horizon_map_sum[
                    horizon
                ] += float(
                    map_planar[
                        :,
                        horizon
                    ][
                        hmask
                    ].sum().item()
                )

            # ------------------------------------------------
            # Standard multimodal minADE:
            # choose the best complete component trajectory
            # for each sample, then average over samples.
            # ------------------------------------------------

            component_planar = torch.sqrt(
                (
                    means_metric[
                        :,
                        :,
                        :,
                        0
                    ]
                    -
                    truth[
                        :,
                        None,
                        :,
                        0
                    ]
                ).square()
                +
                (
                    means_metric[
                        :,
                        :,
                        :,
                        1
                    ]
                    -
                    truth[
                        :,
                        None,
                        :,
                        1
                    ]
                ).square()
            )

            mask_float = (
                valid
                .to(
                    dtype=component_planar.dtype
                )
                .unsqueeze(1)
            )

            denominator = (
                valid.sum(
                    dim=-1
                )
                .clamp_min(1)
                .to(
                    dtype=component_planar.dtype
                )
                .unsqueeze(1)
            )

            component_ade = (
                (
                    component_planar
                    *
                    mask_float
                ).sum(
                    dim=-1
                )
                /
                denominator
            )

            minade = component_ade.min(
                dim=1
            ).values

            minade_sum += float(
                minade[
                    active
                ].sum().item()
            )

            minade_count += (
                active_count
            )

            # minFDE strictly at frozen 1.0-s horizon.
            final_valid = valid[
                :,
                3
            ]

            if bool(
                final_valid.any()
            ):
                component_fde = (
                    component_planar[
                        :,
                        :,
                        3
                    ]
                )

                minfde = component_fde.min(
                    dim=1
                ).values

                count = int(
                    final_valid.sum().item()
                )

                minfde_sum += float(
                    minfde[
                        final_valid
                    ].sum().item()
                )

                minfde_count += count

            probabilities = (
                output
                .mixture_probabilities
            )

            p = probabilities.clamp_min(
                1e-12
            )

            entropy = -(
                p
                *
                torch.log(p)
            ).sum(
                dim=-1
            )

            effective = torch.exp(
                entropy
            )

            entropy_sum += float(
                entropy[
                    active
                ].sum().item()
            )

            effective_mode_sum += float(
                effective[
                    active
                ].sum().item()
            )

            probability_sum += (
                probabilities[
                    active
                ]
                .sum(
                    dim=0
                )
                .cpu()
                .numpy()
            )

            map_indices = torch.argmax(
                output.mixture_logits,
                dim=-1,
            )

            for value in (
                map_indices[
                    active
                ]
                .detach()
                .cpu()
                .tolist()
            ):
                map_mode_counts[
                    int(value)
                ] += 1

            # Pairwise mode separation at 1.0 s.
            final_xy = means_metric[
                :,
                :,
                3,
                :2
            ]

            pairs = (
                (0, 1),
                (0, 2),
                (1, 2),
            )

            for left, right in pairs:
                distance = torch.linalg.vector_norm(
                    final_xy[
                        :,
                        left
                    ]
                    -
                    final_xy[
                        :,
                        right
                    ],
                    dim=-1,
                )

                pairwise_1s_sum += float(
                    distance[
                        active
                    ].sum().item()
                )

                pairwise_1s_count += (
                    active_count
                )

            classes = batch[
                "class_id"
            ]

            for class_id in torch.unique(
                classes
            ).detach().cpu().tolist():
                class_id = int(
                    class_id
                )

                class_mask = (
                    (
                        classes
                        ==
                        class_id
                    )[
                        :,
                        None
                    ]
                    &
                    valid
                )

                count = int(
                    class_mask.sum().item()
                )

                if count <= 0:
                    continue

                name = class_id_to_name.get(
                    class_id,
                    str(class_id),
                )

                class_error_sum[
                    name
                ] += float(
                    mixture_planar[
                        class_mask
                    ].sum().item()
                )

                class_valid[
                    name
                ] += count

            covariance = (
                scale_metric
                @
                scale_metric.transpose(
                    -1,
                    -2,
                )
            )

            (
                _,
                info,
            ) = torch.linalg.cholesky_ex(
                covariance
            )

            spd_failures += int(
                (
                    info != 0
                ).sum().item()
            )

    if active_samples <= 0:
        raise RuntimeError(
            "Development set has no active GMM samples."
        )

    if valid_horizon_points <= 0:
        raise RuntimeError(
            "Development set has no valid horizon points."
        )

    if minade_count <= 0:
        raise RuntimeError(
            "No samples available for minADE."
        )

    if minfde_count <= 0:
        raise RuntimeError(
            "No 1.0-s samples available for minFDE."
        )

    if spd_failures != 0:
        raise RuntimeError(
            f"GMM covariance SPD failures={spd_failures}"
        )

    horizon_mixture = {}
    horizon_map = {}

    for horizon, name in enumerate(
        HORIZON_NAMES
    ):
        count = int(
            horizon_valid[
                horizon
            ]
        )

        if count <= 0:
            raise RuntimeError(
                f"No valid labels at {name}s."
            )

        horizon_mixture[
            name
        ] = float(
            horizon_mixture_sum[
                horizon
            ]
            /
            count
        )

        horizon_map[
            name
        ] = float(
            horizon_map_sum[
                horizon
            ]
            /
            count
        )

    class_ade = {
        name:
            float(
                class_error_sum[
                    name
                ]
                /
                class_valid[
                    name
                ]
            )
        for name in sorted(
            class_valid
        )
    }

    mean_probabilities = (
        probability_sum
        /
        active_samples
    )

    return {
        "joint_GMM_NLL_per_sample":
            float(
                joint_nll_sum
                /
                active_samples
            ),

        "mean_valid_horizons_per_active_sample":
            float(
                valid_horizon_points
                /
                active_samples
            ),

        "mixture_mean_planar_ADE_m":
            float(
                mixture_mean_error_sum
                /
                valid_horizon_points
            ),

        "MAP_component_planar_ADE_m":
            float(
                map_error_sum
                /
                valid_horizon_points
            ),

        "mixture_mean_horizon_error_m":
            horizon_mixture,

        "MAP_component_horizon_error_m":
            horizon_map,

        "mixture_mean_one_second_error_m":
            horizon_mixture[
                "1.0"
            ],

        "MAP_component_one_second_error_m":
            horizon_map[
                "1.0"
            ],

        "min_component_ADE_m":
            float(
                minade_sum
                /
                minade_count
            ),

        "min_component_FDE_1.0s_m":
            float(
                minfde_sum
                /
                minfde_count
            ),

        "mixture_entropy_nats":
            float(
                entropy_sum
                /
                active_samples
            ),

        "effective_mode_count":
            float(
                effective_mode_sum
                /
                active_samples
            ),

        "mean_mixture_probability":
            [
                float(value)
                for value in (
                    mean_probabilities
                    .tolist()
                )
            ],

        "MAP_mode_counts":
            {
                str(component):
                    int(
                        map_mode_counts[
                            component
                        ]
                    )
                for component in range(3)
            },

        "mean_pairwise_mode_separation_1.0s_m":
            float(
                pairwise_1s_sum
                /
                pairwise_1s_count
            ),

        "class_mixture_mean_ADE_m":
            class_ade,

        "active_samples":
            int(
                active_samples
            ),

        "valid_actor_horizon_points":
            int(
                valid_horizon_points
            ),

        "component_covariance_SPD_failures":
            0,
    }

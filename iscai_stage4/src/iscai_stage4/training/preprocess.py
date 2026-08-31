from __future__ import annotations

from dataclasses import dataclass
import math

import torch


CONTINUOUS_DIM = 12
OBSERVED_INDEX = 12
VELOCITY_VALID_INDEX = 13

POSITION_DIMS = (
    0,
    1,
    2,
)

VELOCITY_DIMS = (
    3,
    4,
    5,
)


@dataclass(frozen=True)
class NormalizationStats:
    feature_mean: tuple[
        float,
        ...,
    ]

    feature_std: tuple[
        float,
        ...,
    ]

    map_mean: tuple[
        float,
        ...,
    ]

    map_std: tuple[
        float,
        ...,
    ]

    def __post_init__(self):
        if len(
            self.feature_mean
        ) != CONTINUOUS_DIM:
            raise ValueError(
                "feature_mean must "
                "have 12 values."
            )

        if len(
            self.feature_std
        ) != CONTINUOUS_DIM:
            raise ValueError(
                "feature_std must "
                "have 12 values."
            )

        if len(
            self.map_mean
        ) != 10:
            raise ValueError(
                "map_mean must "
                "have 10 values."
            )

        if len(
            self.map_std
        ) != 10:
            raise ValueError(
                "map_std must "
                "have 10 values."
            )

        if any(
            value <= 0.0
            for value
            in self.feature_std
        ):
            raise ValueError(
                "feature std must "
                "be positive."
            )

        if any(
            value <= 0.0
            for value
            in self.map_std
        ):
            raise ValueError(
                "map std must "
                "be positive."
            )

    def to_dict(self):
        return {
            "feature_mean":
                list(
                    self.feature_mean
                ),

            "feature_std":
                list(
                    self.feature_std
                ),

            "map_mean":
                list(
                    self.map_mean
                ),

            "map_std":
                list(
                    self.map_std
                ),
        }

    @classmethod
    def from_dict(
        cls,
        value,
    ):
        return cls(
            feature_mean=tuple(
                float(x)
                for x in value[
                    "feature_mean"
                ]
            ),
            feature_std=tuple(
                float(x)
                for x in value[
                    "feature_std"
                ]
            ),
            map_mean=tuple(
                float(x)
                for x in value[
                    "map_mean"
                ]
            ),
            map_std=tuple(
                float(x)
                for x in value[
                    "map_std"
                ]
            ),
        )


def normalize_history(
    history: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    """
    Normalize continuous features while
    preserving missing rows as exact zeros
    and leaving masks unchanged.
    """

    result = history.clone()

    mean = torch.tensor(
        stats.feature_mean,
        dtype=result.dtype,
        device=result.device,
    )

    std = torch.tensor(
        stats.feature_std,
        dtype=result.dtype,
        device=result.device,
    )

    observed = (
        result[
            ...,
            OBSERVED_INDEX
        ]
        >
        0.5
    )

    velocity_valid = (
        result[
            ...,
            VELOCITY_VALID_INDEX
        ]
        >
        0.5
    )

    for dim in range(
        CONTINUOUS_DIM
    ):
        valid = (
            velocity_valid
            if dim in (
                VELOCITY_DIMS
            )
            else
            observed
        )

        values = (
            result[
                ...,
                dim
            ]
        )

        normalized = (
            values
            -
            mean[dim]
        ) / std[dim]

        result[
            ...,
            dim
        ] = torch.where(
            valid,
            normalized,
            torch.zeros_like(
                normalized
            ),
        )

    return result


def normalize_map_context(
    value: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    mean = torch.tensor(
        stats.map_mean,
        dtype=value.dtype,
        device=value.device,
    )

    std = torch.tensor(
        stats.map_std,
        dtype=value.dtype,
        device=value.device,
    )

    return (
        value
        -
        mean
    ) / std


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        prediction.shape
        !=
        target.shape
    ):
        raise ValueError(
            "Prediction/target shape mismatch."
        )

    if (
        valid_mask.shape
        !=
        prediction.shape[:2]
    ):
        raise ValueError(
            "Mask shape mismatch."
        )

    mask = (
        valid_mask
        .to(
            prediction.dtype
        )
        .unsqueeze(-1)
    )

    element = torch.nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=1.0,
    )

    numerator = (
        element
        *
        mask
    ).sum()

    denominator = (
        mask.sum()
        *
        prediction.shape[-1]
    )

    if float(
        denominator.detach().cpu()
    ) <= 0.0:
        return prediction.sum() * 0.0

    return (
        numerator
        /
        denominator
    )


def planar_ade(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    error = torch.linalg.vector_norm(
        prediction[
            ...,
            :2
        ]
        -
        target[
            ...,
            :2
        ],
        dim=-1,
    )

    mask = (
        valid_mask.bool()
    )

    if not bool(
        mask.any()
    ):
        return float(
            "nan"
        )

    return float(
        error[
            mask
        ].mean()
        .detach()
        .cpu()
        .item()
    )


def planar_fde(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    errors = []

    for row in range(
        prediction.shape[0]
    ):
        indices = torch.nonzero(
            valid_mask[row],
            as_tuple=False,
        ).flatten()

        if indices.numel() == 0:
            continue

        index = int(
            indices[-1].item()
        )

        error = (
            torch.linalg.vector_norm(
                prediction[
                    row,
                    index,
                    :2,
                ]
                -
                target[
                    row,
                    index,
                    :2,
                ]
            )
        )

        errors.append(
            error
        )

    if not errors:
        return float(
            "nan"
        )

    return float(
        torch.stack(
            errors
        ).mean()
        .detach()
        .cpu()
        .item()
    )


def class_sampling_weights(
    class_ids: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[int, float],
]:
    """
    Deterministic inverse-square-root
    class balancing.

    Uses current actor class as sampler
    metadata only, never model input.
    """

    if (
        class_ids.ndim
        !=
        1
    ):
        raise ValueError(
            "class_ids must be 1-D."
        )

    unique, counts = (
        torch.unique(
            class_ids,
            return_counts=True,
        )
    )

    count_map = {
        int(key.item()):
            int(value.item())
        for key, value
        in zip(
            unique,
            counts,
        )
    }

    maximum = max(
        count_map.values()
    )

    weight_map = {}

    for key, count in (
        count_map.items()
    ):
        raw = math.sqrt(
            maximum
            /
            count
        )

        weight_map[key] = min(
            raw,
            8.0,
        )

    sample_weights = torch.tensor(
        [
            weight_map[
                int(value.item())
            ]
            for value
            in class_ids
        ],
        dtype=torch.float64,
    )

    return (
        sample_weights,
        weight_map,
    )

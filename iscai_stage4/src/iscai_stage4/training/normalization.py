from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path

import numpy as np

from iscai_stage4.data import (
    FEATURE_DIM,
    MAP_CONTEXT_DIM,
)


CONTINUOUS_HISTORY_DIMS = 12
OBSERVED_MASK_INDEX = 12
VELOCITY_MASK_INDEX = 13

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

COVARIANCE_DIMS = (
    6,
    7,
    8,
    9,
    10,
    11,
)


@dataclass(frozen=True)
class FitNormalizationStats:
    history_mean: tuple[float, ...]

    history_std: tuple[float, ...]

    history_count: tuple[int, ...]

    map_mean: tuple[float, ...]

    map_std: tuple[float, ...]

    map_count: tuple[int, ...]

    label_mean: tuple[
            tuple[float, ...],
            ...,
        ]

    label_std: tuple[
            tuple[float, ...],
            ...,
        ]

    label_count: tuple[
            tuple[int, ...],
            ...,
        ]

    def __post_init__(self):
        if len(
            self.history_mean
        ) != CONTINUOUS_HISTORY_DIMS:
            raise ValueError(
                "History normalization "
                "dimension changed."
            )

        if len(
            self.history_std
        ) != CONTINUOUS_HISTORY_DIMS:
            raise ValueError(
                "History normalization "
                "dimension changed."
            )

        if len(
            self.map_mean
        ) != MAP_CONTEXT_DIM:
            raise ValueError(
                "Map normalization "
                "dimension changed."
            )

        if len(
            self.label_mean
        ) != 4:
            raise ValueError(
                "Label horizon count changed."
            )

        if any(
            len(row) != 3
            for row in self.label_mean
        ):
            raise ValueError(
                "Label dimension changed."
            )

        if any(
            float(value) <= 0.0
            for value
            in self.history_std
        ):
            raise ValueError(
                "History std must be > 0."
            )

        if any(
            float(value) <= 0.0
            for value
            in self.map_std
        ):
            raise ValueError(
                "Map std must be > 0."
            )

        if any(
            float(value) <= 0.0
            for row in self.label_std
            for value in row
        ):
            raise ValueError(
                "Label std must be > 0."
            )

    def to_dict(self):
        return {
            "history_mean":
                list(
                    self.history_mean
                ),

            "history_std":
                list(
                    self.history_std
                ),

            "history_count":
                list(
                    self.history_count
                ),

            "map_mean":
                list(
                    self.map_mean
                ),

            "map_std":
                list(
                    self.map_std
                ),

            "map_count":
                list(
                    self.map_count
                ),

            "label_mean": [
                list(row)
                for row
                in self.label_mean
            ],

            "label_std": [
                list(row)
                for row
                in self.label_std
            ],

            "label_count": [
                list(row)
                for row
                in self.label_count
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data,
    ):
        return cls(
            history_mean=tuple(
                float(x)
                for x in data[
                    "history_mean"
                ]
            ),

            history_std=tuple(
                float(x)
                for x in data[
                    "history_std"
                ]
            ),

            history_count=tuple(
                int(x)
                for x in data[
                    "history_count"
                ]
            ),

            map_mean=tuple(
                float(x)
                for x in data[
                    "map_mean"
                ]
            ),

            map_std=tuple(
                float(x)
                for x in data[
                    "map_std"
                ]
            ),

            map_count=tuple(
                int(x)
                for x in data[
                    "map_count"
                ]
            ),

            label_mean=tuple(
                tuple(
                    float(x)
                    for x in row
                )
                for row in data[
                    "label_mean"
                ]
            ),

            label_std=tuple(
                tuple(
                    float(x)
                    for x in row
                )
                for row in data[
                    "label_std"
                ]
            ),

            label_count=tuple(
                tuple(
                    int(x)
                    for x in row
                )
                for row in data[
                    "label_count"
                ]
            ),
        )

    def write_json(
        self,
        path: Path,
    ):
        path.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
            )
            +
            "\n",
            encoding="utf-8",
        )


class FitStatsAccumulator:

    def __init__(self):
        self.history_sum = (
            np.zeros(
                CONTINUOUS_HISTORY_DIMS,
                dtype=np.float64,
            )
        )

        self.history_sumsq = (
            np.zeros(
                CONTINUOUS_HISTORY_DIMS,
                dtype=np.float64,
            )
        )

        self.history_count = (
            np.zeros(
                CONTINUOUS_HISTORY_DIMS,
                dtype=np.int64,
            )
        )

        self.map_sum = (
            np.zeros(
                MAP_CONTEXT_DIM,
                dtype=np.float64,
            )
        )

        self.map_sumsq = (
            np.zeros(
                MAP_CONTEXT_DIM,
                dtype=np.float64,
            )
        )

        self.map_count = (
            np.zeros(
                MAP_CONTEXT_DIM,
                dtype=np.int64,
            )
        )

        self.label_sum = (
            np.zeros(
                (4, 3),
                dtype=np.float64,
            )
        )

        self.label_sumsq = (
            np.zeros(
                (4, 3),
                dtype=np.float64,
            )
        )

        self.label_count = (
            np.zeros(
                (4, 3),
                dtype=np.int64,
            )
        )

        self.sample_count = 0

    def _update_history(
        self,
        history,
    ):
        values = np.asarray(
            history,
            dtype=np.float64,
        )

        if values.shape != (
            11,
            FEATURE_DIM,
        ):
            raise ValueError(
                "History tensor shape changed."
            )

        observed = (
            values[
                :,
                OBSERVED_MASK_INDEX
            ]
            >
            0.5
        )

        velocity_valid = (
            values[
                :,
                VELOCITY_MASK_INDEX
            ]
            >
            0.5
        )

        for dim in (
            POSITION_DIMS
            +
            COVARIANCE_DIMS
        ):
            selected = (
                values[
                    observed,
                    dim,
                ]
            )

            if selected.size:
                self.history_sum[
                    dim
                ] += selected.sum()

                self.history_sumsq[
                    dim
                ] += (
                    selected
                    .square()
                    .sum()
                )

                self.history_count[
                    dim
                ] += selected.size

        for dim in VELOCITY_DIMS:
            selected = (
                values[
                    velocity_valid,
                    dim,
                ]
            )

            if selected.size:
                self.history_sum[
                    dim
                ] += selected.sum()

                self.history_sumsq[
                    dim
                ] += (
                    selected
                    .square()
                    .sum()
                )

                self.history_count[
                    dim
                ] += selected.size

    def update_sample(
        self,
        sample,
    ):
        payload = (
            sample.model_input
        )

        self._update_history(
            payload.target_history
        )

        for active, history in zip(
            payload.neighbor_mask,
            payload.neighbor_histories,
        ):
            if float(active) > 0.5:
                self._update_history(
                    history
                )

        map_values = np.asarray(
            payload.map_context,
            dtype=np.float64,
        )

        self.map_sum += map_values

        self.map_sumsq += (
            map_values.square()
        )

        self.map_count += 1

        target = np.asarray(
            payload.target_history,
            dtype=np.float64,
        )

        observed = np.where(
            target[
                :,
                OBSERVED_MASK_INDEX
            ]
            >
            0.5
        )[0]

        if not len(observed):
            raise ValueError(
                "Target has no observed frame."
            )

        origin = target[
            observed[-1],
            :3,
        ]

        positions = np.asarray(
            sample
            .future_label
            .positions_H0_m,
            dtype=np.float64,
        )

        valid = np.asarray(
            sample
            .future_label
            .valid_mask,
            dtype=bool,
        )

        displacement = (
            positions
            -
            origin[
                None,
                :,
            ]
        )

        for horizon in range(4):
            if not valid[
                horizon
            ]:
                continue

            self.label_sum[
                horizon
            ] += displacement[
                horizon
            ]

            self.label_sumsq[
                horizon
            ] += (
                displacement[
                    horizon
                ].square()
            )

            self.label_count[
                horizon
            ] += 1

        self.sample_count += 1

    @staticmethod
    def _finalize(
        total,
        sumsq,
        count,
    ):
        if np.any(
            count <= 0
        ):
            raise RuntimeError(
                "Normalization statistic "
                "has zero support."
            )

        mean = (
            total
            /
            count
        )

        variance = (
            sumsq
            /
            count
            -
            mean.square()
        )

        variance = np.maximum(
            variance,
            1e-12,
        )

        std = np.sqrt(
            variance
        )

        std = np.maximum(
            std,
            1e-6,
        )

        return (
            mean,
            std,
        )

    def finalize(
        self,
    ) -> FitNormalizationStats:
        (
            history_mean,
            history_std,
        ) = self._finalize(
            self.history_sum,
            self.history_sumsq,
            self.history_count,
        )

        (
            map_mean,
            map_std,
        ) = self._finalize(
            self.map_sum,
            self.map_sumsq,
            self.map_count,
        )

        (
            label_mean,
            label_std,
        ) = self._finalize(
            self.label_sum,
            self.label_sumsq,
            self.label_count,
        )

        return FitNormalizationStats(
            history_mean=tuple(
                float(x)
                for x in history_mean
            ),

            history_std=tuple(
                float(x)
                for x in history_std
            ),

            history_count=tuple(
                int(x)
                for x in self.history_count
            ),

            map_mean=tuple(
                float(x)
                for x in map_mean
            ),

            map_std=tuple(
                float(x)
                for x in map_std
            ),

            map_count=tuple(
                int(x)
                for x in self.map_count
            ),

            label_mean=tuple(
                tuple(
                    float(x)
                    for x in row
                )
                for row in label_mean
            ),

            label_std=tuple(
                tuple(
                    float(x)
                    for x in row
                )
                for row in label_std
            ),

            label_count=tuple(
                tuple(
                    int(x)
                    for x in row
                )
                for row in self.label_count
            ),
        )


def normalized_sample_arrays(
    sample,
    stats:
        FitNormalizationStats,
):
    history_mean = np.asarray(
        stats.history_mean,
        dtype=np.float32,
    )

    history_std = np.asarray(
        stats.history_std,
        dtype=np.float32,
    )

    map_mean = np.asarray(
        stats.map_mean,
        dtype=np.float32,
    )

    map_std = np.asarray(
        stats.map_std,
        dtype=np.float32,
    )

    label_mean = np.asarray(
        stats.label_mean,
        dtype=np.float32,
    )

    label_std = np.asarray(
        stats.label_std,
        dtype=np.float32,
    )

    def normalize_history(
        history,
    ):
        x = np.asarray(
            history,
            dtype=np.float32,
        ).copy()

        observed = (
            x[
                :,
                OBSERVED_MASK_INDEX
            :
                OBSERVED_MASK_INDEX
                +
                1
            ]
        )

        velocity_valid = (
            x[
                :,
                VELOCITY_MASK_INDEX
            :
                VELOCITY_MASK_INDEX
                +
                1
            ]
        )

        x[
            :,
            :
            CONTINUOUS_HISTORY_DIMS
        ] = (
            x[
                :,
                :
                CONTINUOUS_HISTORY_DIMS
            ]
            -
            history_mean[
                None,
                :,
            ]
        ) / (
            history_std[
                None,
                :,
            ]
        )

        for dim in (
            POSITION_DIMS
            +
            COVARIANCE_DIMS
        ):
            x[
                :,
                dim
            ] *= (
                observed[
                    :,
                    0
                ]
            )

        for dim in VELOCITY_DIMS:
            x[
                :,
                dim
            ] *= (
                velocity_valid[
                    :,
                    0
                ]
            )

        return x

    target = (
        normalize_history(
            sample
            .model_input
            .target_history
        )
    )

    neighbors = np.stack(
        [
            normalize_history(
                history
            )
            for history
            in (
                sample
                .model_input
                .neighbor_histories
            )
        ],
        axis=0,
    )

    neighbor_mask = np.asarray(
        sample
        .model_input
        .neighbor_mask,
        dtype=np.float32,
    )

    map_context = (
        np.asarray(
            sample
            .model_input
            .map_context,
            dtype=np.float32,
        )
        -
        map_mean
    ) / map_std

    raw_target = np.asarray(
        sample
        .model_input
        .target_history,
        dtype=np.float32,
    )

    observed = np.where(
        raw_target[
            :,
            OBSERVED_MASK_INDEX
        ]
        >
        0.5
    )[0]

    origin = raw_target[
        observed[-1],
        :3,
    ]

    positions = np.asarray(
        sample
        .future_label
        .positions_H0_m,
        dtype=np.float32,
    )

    valid = np.asarray(
        sample
        .future_label
        .valid_mask,
        dtype=np.float32,
    )

    displacement = (
        positions
        -
        origin[
            None,
            :,
        ]
    )

    label_normalized = (
        displacement
        -
        label_mean
    ) / label_std

    label_normalized *= (
        valid[
            :,
            None
        ]
    )

    return {
        "target":
            target.astype(
                np.float32
            ),

        "neighbors":
            neighbors.astype(
                np.float32
            ),

        "neighbor_mask":
            neighbor_mask.astype(
                np.float32
            ),

        "map":
            map_context.astype(
                np.float32
            ),

        "label":
            label_normalized.astype(
                np.float32
            ),

        "valid":
            valid.astype(
                np.float32
            ),

        "origin":
            origin.astype(
                np.float32
            ),

        "actor_class":
            str(
                sample.actor_class
            ),
    }


def denormalize_displacement(
    value,
    stats:
        FitNormalizationStats,
):
    value = np.asarray(
        value,
        dtype=np.float32,
    )

    return (
        value
        *
        np.asarray(
            stats.label_std,
            dtype=np.float32,
        )
        +
        np.asarray(
            stats.label_mean,
            dtype=np.float32,
        )
    )

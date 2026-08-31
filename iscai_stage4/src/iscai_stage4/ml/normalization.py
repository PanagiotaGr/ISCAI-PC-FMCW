from __future__ import annotations

from hashlib import sha256
import json

import numpy as np


CONTINUOUS_FEATURE_DIM = 12
EPS_STD = 1e-6


def _canonical_sha(
    value,
) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def latest_observed_position(
    target_history,
):
    target = np.asarray(
        target_history
    )

    if target.ndim != 3:
        raise ValueError(
            "target history must "
            "be [N,T,F]."
        )

    observed = (
        target[:, :, 12]
        >
        0.5
    )

    if not np.all(
        observed.any(
            axis=1
        )
    ):
        raise ValueError(
            "A target history has "
            "no observation."
        )

    grid = np.arange(
        target.shape[1],
        dtype=np.int64,
    )[None, :]

    indices = np.where(
        observed,
        grid,
        -1,
    ).max(
        axis=1
    )

    return target[
        np.arange(
            target.shape[0]
        ),
        indices,
        :3,
    ].astype(
        np.float64
    )


def compute_fit_normalization(
    arrays,
    *,
    chunk_size: int = 4096,
):
    """
    Fit partition only.

    Continuous history statistics use only
    actual observed rows.

    Zero-padded missing timesteps and
    padded neighbour slots never enter
    mean/std estimates.

    Label statistics use only valid future
    labels and are computed as displacement
    from the latest causal observed target
    position.
    """

    target = arrays[
        "target"
    ]

    neighbors = arrays[
        "neighbors"
    ]

    neighbor_mask = arrays[
        "neighbor_mask"
    ]

    map_context = arrays[
        "map_context"
    ]

    future = arrays[
        "future"
    ]

    future_mask = arrays[
        "future_mask"
    ]

    sample_count = int(
        target.shape[0]
    )

    if sample_count <= 0:
        raise ValueError(
            "Empty fit cache."
        )

    feature_sum = np.zeros(
        CONTINUOUS_FEATURE_DIM,
        dtype=np.float64,
    )

    feature_sumsq = np.zeros(
        CONTINUOUS_FEATURE_DIM,
        dtype=np.float64,
    )

    feature_count = 0

    map_sum = np.zeros(
        map_context.shape[1],
        dtype=np.float64,
    )

    map_sumsq = np.zeros(
        map_context.shape[1],
        dtype=np.float64,
    )

    map_count = 0

    label_sum = np.zeros(
        (
            future.shape[1],
            future.shape[2],
        ),
        dtype=np.float64,
    )

    label_sumsq = np.zeros_like(
        label_sum
    )

    label_count = np.zeros(
        future.shape[1],
        dtype=np.int64,
    )

    for start in range(
        0,
        sample_count,
        chunk_size,
    ):
        stop = min(
            sample_count,
            start
            +
            chunk_size,
        )

        t = np.asarray(
            target[
                start:stop
            ],
            dtype=np.float64,
        )

        n = np.asarray(
            neighbors[
                start:stop
            ],
            dtype=np.float64,
        )

        nm = np.asarray(
            neighbor_mask[
                start:stop
            ],
            dtype=np.float64,
        )

        m = np.asarray(
            map_context[
                start:stop
            ],
            dtype=np.float64,
        )

        y = np.asarray(
            future[
                start:stop
            ],
            dtype=np.float64,
        )

        ym = np.asarray(
            future_mask[
                start:stop
            ],
            dtype=np.float64,
        )

        target_observed = (
            t[:, :, 12]
            >
            0.5
        )

        target_values = (
            t[:, :, :12][
                target_observed
            ]
        )

        if target_values.size:
            feature_sum += (
                target_values
                .sum(
                    axis=0
                )
            )

            feature_sumsq += (
                np.square(
                    target_values
                )
                .sum(
                    axis=0
                )
            )

            feature_count += (
                target_values
                .shape[0]
            )

        neighbor_observed = (
            (
                n[:, :, :, 12]
                >
                0.5
            )
            &
            (
                nm[:, :, None]
                >
                0.5
            )
        )

        neighbor_values = (
            n[:, :, :, :12][
                neighbor_observed
            ]
        )

        if neighbor_values.size:
            feature_sum += (
                neighbor_values
                .sum(
                    axis=0
                )
            )

            feature_sumsq += (
                np.square(
                    neighbor_values
                )
                .sum(
                    axis=0
                )
            )

            feature_count += (
                neighbor_values
                .shape[0]
            )

        map_sum += (
            m.sum(
                axis=0
            )
        )

        map_sumsq += (
            np.square(
                m
            ).sum(
                axis=0
            )
        )

        map_count += (
            m.shape[0]
        )

        origin = (
            latest_observed_position(
                t
            )
        )

        displacement = (
            y
            -
            origin[:, None, :]
        )

        for horizon in range(
            y.shape[1]
        ):
            valid = (
                ym[:, horizon]
                >
                0.5
            )

            values = (
                displacement[
                    valid,
                    horizon,
                    :
                ]
            )

            if values.size:
                label_sum[
                    horizon
                ] += (
                    values.sum(
                        axis=0
                    )
                )

                label_sumsq[
                    horizon
                ] += (
                    np.square(
                        values
                    ).sum(
                        axis=0
                    )
                )

                label_count[
                    horizon
                ] += (
                    values.shape[0]
                )

    if feature_count <= 0:
        raise RuntimeError(
            "No observed history rows "
            "for normalization."
        )

    if map_count <= 0:
        raise RuntimeError(
            "No map rows for "
            "normalization."
        )

    if np.any(
        label_count <= 0
    ):
        raise RuntimeError(
            "At least one horizon "
            "has zero fit labels."
        )

    feature_mean = (
        feature_sum
        /
        feature_count
    )

    feature_var = (
        feature_sumsq
        /
        feature_count
        -
        np.square(
            feature_mean
        )
    )

    feature_std = np.sqrt(
        np.maximum(
            feature_var,
            EPS_STD**2,
        )
    )

    map_mean = (
        map_sum
        /
        map_count
    )

    map_var = (
        map_sumsq
        /
        map_count
        -
        np.square(
            map_mean
        )
    )

    map_std = np.sqrt(
        np.maximum(
            map_var,
            EPS_STD**2,
        )
    )

    label_mean = (
        label_sum
        /
        label_count[:, None]
    )

    label_var = (
        label_sumsq
        /
        label_count[:, None]
        -
        np.square(
            label_mean
        )
    )

    label_std = np.sqrt(
        np.maximum(
            label_var,
            EPS_STD**2,
        )
    )

    result = {
        "source":
            "fit_only",

        "sample_count":
            sample_count,

        "observed_history_row_count":
            int(
                feature_count
            ),

        "continuous_feature_mean":
            feature_mean.tolist(),

        "continuous_feature_std":
            feature_std.tolist(),

        "map_mean":
            map_mean.tolist(),

        "map_std":
            map_std.tolist(),

        "label_displacement_mean":
            label_mean.tolist(),

        "label_displacement_std":
            label_std.tolist(),

        "label_valid_count_per_horizon":
            label_count.tolist(),

        "missing_history_rows_in_statistics":
            False,

        "padded_neighbor_slots_in_statistics":
            False,

        "development_used":
            False,

        "calibration_used":
            False,

        "validation_used":
            False,
    }

    result[
        "sha256"
    ] = _canonical_sha(
        result
    )

    return result

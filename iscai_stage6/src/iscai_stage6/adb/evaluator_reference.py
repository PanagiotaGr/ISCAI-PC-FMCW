from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FROZEN_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

FROZEN_ADB_ACTOR_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)


@dataclass(frozen=True)
class EvaluatorReferenceSemantics:
    oracle_type: str
    measured_adb_ground_truth: bool
    controller_future_truth_access: bool
    parameter_tuning_future_truth_access: bool
    oracle_grid_semantics: str
    road_roi_semantics: str
    final_metric_intensity: str


FROZEN_EVALUATOR_REFERENCE_SEMANTICS = (
    EvaluatorReferenceSemantics(
        oracle_type=(
            "constructed_future_reference"
        ),

        measured_adb_ground_truth=False,

        controller_future_truth_access=False,

        parameter_tuning_future_truth_access=False,

        oracle_grid_semantics=(
            "same_frozen_theta_range_headlamp_"
            "actuator_grid_as_controller"
        ),

        road_roi_semantics=(
            "all_valid_cells_of_frozen_theta_range_"
            "headlamp_actuator_grid"
        ),

        final_metric_intensity=(
            "RateLimitedSchedule.illumination"
        ),
    )
)


def frozen_road_roi(
    grid_shape,
) -> np.ndarray:
    """
    Frozen Block6.8 evaluator implementation choice.

    The Stage-6 actuator grid is the frozen theta x range
    headlamp road-domain grid.

    The road-illumination-retention ROI is therefore every
    valid cell of that frozen actuator grid.

    This ROI:
      - is deterministic,
      - does not depend on actors,
      - does not depend on future truth,
      - does not depend on development outcomes,
      - does not alter the controller.
    """

    shape = tuple(
        int(value)
        for value in grid_shape
    )

    if len(shape) != 2:
        raise ValueError(
            "grid_shape must be "
            "(theta_cells, range_cells)"
        )

    if (
        shape[0] <= 0
        or
        shape[1] <= 0
    ):
        raise ValueError(
            "grid dimensions must be > 0"
        )

    return np.ones(
        shape,
        dtype=np.bool_,
    )


def _actor_support_array(
    value,
    *,
    grid_shape,
) -> np.ndarray:

    array = np.asarray(
        value
    )

    if (
        array.dtype
        !=
        np.bool_
    ):
        raise TypeError(
            "actor support masks must be boolean"
        )

    expected = (
        len(
            FROZEN_HORIZONS_S
        ),
        int(
            grid_shape[0]
        ),
        int(
            grid_shape[1]
        ),
    )

    if (
        tuple(
            array.shape
        )
        !=
        expected
    ):
        raise ValueError(
            "actor support shape must be "
            "[4, theta, range]"
        )

    return array


def aggregate_constructed_oracle_supports(
    actor_supports: Iterable[np.ndarray],
    *,
    grid_shape,
) -> np.ndarray:
    """
    O_tau is the union of evaluator-only future actor
    shadow supports on the frozen controller grid.

    Future truth is consumed only by the evaluator after
    controller output has already been produced.
    """

    shape = tuple(
        int(value)
        for value in grid_shape
    )

    if len(shape) != 2:
        raise ValueError(
            "grid_shape must be "
            "(theta_cells, range_cells)"
        )

    result = np.zeros(
        (
            len(
                FROZEN_HORIZONS_S
            ),
            shape[0],
            shape[1],
        ),
        dtype=np.bool_,
    )

    for support in actor_supports:

        result |= _actor_support_array(
            support,
            grid_shape=shape,
        )

    return result


def aggregate_class_future_region(
    actor_regions: Iterable[np.ndarray],
    *,
    grid_shape,
) -> np.ndarray:
    """
    Union of evaluator-only projected future-GT full-box
    footprints for one actor class.

    Used for pedestrian/cyclist visibility support and,
    with the vehicle-specific constructed surrogate,
    evaluator protection regions.
    """

    return aggregate_constructed_oracle_supports(
        actor_regions,
        grid_shape=grid_shape,
    )


def validate_final_metric_illumination(
    illumination,
    *,
    grid_shape,
) -> np.ndarray:
    """
    Validate the frozen evaluator-facing post-actuation output:
        RateLimitedSchedule.illumination
    """

    array = np.asarray(
        illumination,
        dtype=np.float64,
    )

    expected = (
        len(
            FROZEN_HORIZONS_S
        ),
        int(
            grid_shape[0]
        ),
        int(
            grid_shape[1]
        ),
    )

    if (
        tuple(
            array.shape
        )
        !=
        expected
    ):
        raise ValueError(
            "final illumination must have "
            "shape [4, theta, range]"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "final illumination must be finite"
        )

    if (
        np.any(
            array < 0.0
        )
        or
        np.any(
            array > 1.0
        )
    ):
        raise ValueError(
            "final illumination must lie in [0,1]"
        )

    return array

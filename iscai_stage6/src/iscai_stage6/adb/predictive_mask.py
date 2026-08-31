from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from iscai_stage6.adb.illumination import (
    ReactiveShadowRegion,
    reactive_adb_map,
)

from iscai_stage6.adb.probabilistic_occupancy import (
    ActorOccupancyProbability,
)


HORIZON_COUNT = 4


def _readonly(
    value,
    *,
    dtype,
) -> np.ndarray:
    result = np.array(
        value,
        dtype=dtype,
        copy=True,
    )

    result.setflags(
        write=False
    )

    return result


def _validated_gamma(
    gamma: float,
) -> float:
    value = float(
        gamma
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            "gamma must be finite."
        )

    if (
        value < 0.0
        or
        value > 1.0
    ):
        raise ValueError(
            "gamma must lie in [0,1]."
        )

    return value


def _validated_grid_shape(
    grid_shape,
) -> tuple[
    int,
    int,
]:
    try:
        values = tuple(
            int(value)
            for value in grid_shape
        )
    except BaseException as exc:
        raise ValueError(
            "grid_shape must contain two integers."
        ) from exc

    if len(
        values
    ) != 2:
        raise ValueError(
            "grid_shape must be (N_theta,N_range)."
        )

    if (
        values[0] <= 0
        or
        values[1] <= 0
    ):
        raise ValueError(
            "grid_shape dimensions must be positive."
        )

    return values


def _validated_probability(
    actor_occupancy:
        ActorOccupancyProbability,
) -> np.ndarray:
    probability = np.asarray(
        actor_occupancy
        .occupancy_probability,
        dtype=np.float64,
    )

    if probability.ndim != 3:
        raise ValueError(
            "Actor occupancy probability must "
            "have shape [4,N_theta,N_range]."
        )

    if probability.shape[
        0
    ] != HORIZON_COUNT:
        raise ValueError(
            "Actor occupancy probability must "
            "contain exactly four horizons."
        )

    if not np.all(
        np.isfinite(
            probability
        )
    ):
        raise ValueError(
            "Actor occupancy probability "
            "must be finite."
        )

    if (
        np.any(
            probability < 0.0
        )
        or
        np.any(
            probability > 1.0
        )
    ):
        raise ValueError(
            "Actor occupancy probability "
            "must lie in [0,1]."
        )

    return probability


@dataclass(
    frozen=True
)
class BinaryActorPredictiveMask:
    """
    Per-actor thresholded predictive mask

        M_i,tau = 1[P_occ,i > gamma]

    The comparison is deliberately STRICT greater-than.
    """

    prediction_id: str

    gamma: float

    mask: np.ndarray

    @property
    def shape(
        self,
    ) -> tuple[
        int,
        int,
        int,
    ]:
        return tuple(
            int(value)
            for value in self.mask.shape
        )


@dataclass(
    frozen=True
)
class ClassAgnosticPredictiveMaskPlan:
    """
    Block6.5 Part1 representation.

    predictive_mask:
        exact binary multi-actor predictive aggregation
        [4,N_theta,N_range].

    reactive_fallback_illumination:
        exact frozen Block6.2 original-reactive intensity
        map [N_theta,N_range], or None when no reactive
        grid was requested.

    These are deliberately kept separate. Part1 does not
    invent a final dimming floor or destroy the Part-A
    raised-cosine reactive fallback by binarizing it.
    """

    gamma: float

    actor_masks: tuple[
        BinaryActorPredictiveMask,
        ...,
    ]

    predictive_mask: np.ndarray

    reactive_fallback_illumination: (
        np.ndarray
        |
        None
    )

    reactive_fallback_region_count: int

    predictive_aggregation_semantics: str = (
        "1_minus_product_i_of_1_minus_M_i_tau"
    )

    reactive_fallback_semantics: str = (
        "exact_frozen_Block6.2_reactive_adb_map"
    )

    final_illumination_composition_applied: bool = False

    class_aware_policy_applied: bool = False

    temporal_smoothing_applied: bool = False

    actuation_rate_limit_applied: bool = False


def threshold_actor_occupancy(
    actor_occupancy:
        ActorOccupancyProbability,
    *,
    gamma: float,
) -> BinaryActorPredictiveMask:
    """
    Compute exactly:

        M_i,tau = 1[P_occ,i > gamma]

    No >= substitution and no class-specific adjustment.
    """

    value = _validated_gamma(
        gamma
    )

    probability = _validated_probability(
        actor_occupancy
    )

    mask = np.greater(
        probability,
        value,
    )

    return BinaryActorPredictiveMask(
        prediction_id=str(
            actor_occupancy.prediction_id
        ),
        gamma=value,
        mask=_readonly(
            mask,
            dtype=bool,
        ),
    )


def aggregate_binary_actor_masks(
    actor_masks:
        Iterable[
            BinaryActorPredictiveMask
        ],
    *,
    grid_shape,
) -> np.ndarray:
    """
    Compute exactly:

        M_tau = 1 - product_i(1 - M_i,tau)

    for binary per-actor masks.

    The implementation also verifies exact equivalence to
    logical OR. Empty actor sets produce the all-false mask.
    """

    theta_count, range_count = (
        _validated_grid_shape(
            grid_shape
        )
    )

    expected_shape = (
        HORIZON_COUNT,
        theta_count,
        range_count,
    )

    items = tuple(
        actor_masks
    )

    if not items:
        return _readonly(
            np.zeros(
                expected_shape,
                dtype=bool,
            ),
            dtype=bool,
        )

    arrays = []

    for index, item in enumerate(
        items
    ):
        array = np.asarray(
            item.mask
        )

        if array.dtype != np.bool_:
            raise ValueError(
                "Binary actor masks must have bool dtype."
            )

        if array.shape != expected_shape:
            raise ValueError(
                "Binary actor mask shape mismatch "
                f"at index {index}: "
                f"{array.shape} != {expected_shape}."
            )

        arrays.append(
            array
        )

    stacked = np.stack(
        arrays,
        axis=0,
    )

    integer = stacked.astype(
        np.uint8
    )

    formula_integer = (
        1
        -
        np.prod(
            (
                1
                -
                integer
            ),
            axis=0,
            dtype=np.uint8,
        )
    )

    formula_mask = (
        formula_integer
        >
        0
    )

    logical_or = np.any(
        stacked,
        axis=0,
    )

    if not np.array_equal(
        formula_mask,
        logical_or,
    ):
        raise RuntimeError(
            "Binary aggregation lost exact OR "
            "equivalence."
        )

    return _readonly(
        formula_mask,
        dtype=bool,
    )


def exact_original_reactive_fallback_map(
    illumination_grid,
    reactive_regions:
        Iterable[
            ReactiveShadowRegion
        ],
) -> np.ndarray:
    """
    Exact call-through to the frozen Block6.2 reactive
    illumination implementation.

    No binary conversion is performed. The returned map
    therefore retains the Part-A raised-cosine radial
    transition and intensity-floor semantics.
    """

    regions = tuple(
        reactive_regions
    )

    result = reactive_adb_map(
        illumination_grid,
        regions,
    )

    array = np.asarray(
        result,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise RuntimeError(
            "Frozen reactive ADB map must be 2-D."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise RuntimeError(
            "Frozen reactive ADB map contains "
            "non-finite intensity."
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
        raise RuntimeError(
            "Frozen reactive ADB intensity "
            "left [0,1]."
        )

    return _readonly(
        array,
        dtype=np.float64,
    )


def build_class_agnostic_predictive_mask_plan(
    actor_occupancies:
        Iterable[
            ActorOccupancyProbability
        ],
    *,
    gamma: float,
    grid_shape,
    reactive_grid: Any | None = None,
    reactive_fallback_regions:
        Iterable[
            ReactiveShadowRegion
        ] = (),
) -> ClassAgnosticPredictiveMaskPlan:
    """
    Build the class-agnostic Block6.5 Part1 mask layer.

    Matched actors:
        calibrated probabilistic P_occ -> strict gamma
        threshold -> exact multi-actor binary union.

    Unmatched actors:
        exact frozen original-reactive Block6.2
        illumination map.

    No final predictive/reactive intensity composition is
    performed in Part1 because class-specific dimming floors
    and policy depths have not yet been frozen.
    """

    value = _validated_gamma(
        gamma
    )

    shape = _validated_grid_shape(
        grid_shape
    )

    occupancies = tuple(
        actor_occupancies
    )

    prediction_ids = tuple(
        str(
            item.prediction_id
        )
        for item in occupancies
    )

    if len(
        set(
            prediction_ids
        )
    ) != len(
        prediction_ids
    ):
        raise ValueError(
            "Duplicate prediction_id in "
            "actor occupancies."
        )

    actor_masks = tuple(
        threshold_actor_occupancy(
            item,
            gamma=value,
        )
        for item in occupancies
    )

    expected_probability_shape = (
        HORIZON_COUNT,
        shape[
            0
        ],
        shape[
            1
        ],
    )

    for index, item in enumerate(
        actor_masks
    ):
        if item.mask.shape != (
            expected_probability_shape
        ):
            raise ValueError(
                "Thresholded actor occupancy "
                f"shape mismatch at index {index}: "
                f"{item.mask.shape} != "
                f"{expected_probability_shape}."
            )

    predictive_mask = (
        aggregate_binary_actor_masks(
            actor_masks,
            grid_shape=shape,
        )
    )

    regions = tuple(
        reactive_fallback_regions
    )

    fallback = None

    if reactive_grid is None:
        if regions:
            raise ValueError(
                "reactive_grid is required when "
                "reactive fallback regions are supplied."
            )

    else:
        fallback = (
            exact_original_reactive_fallback_map(
                reactive_grid,
                regions,
            )
        )

        if fallback.shape != shape:
            raise ValueError(
                "Predictive/fallback grid shape "
                "mismatch: "
                f"{shape} vs {fallback.shape}."
            )

    return ClassAgnosticPredictiveMaskPlan(
        gamma=value,
        actor_masks=actor_masks,
        predictive_mask=predictive_mask,
        reactive_fallback_illumination=(
            fallback
        ),
        reactive_fallback_region_count=len(
            regions
        ),
    )

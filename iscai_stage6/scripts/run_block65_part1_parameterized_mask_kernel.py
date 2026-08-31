from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

BLOCK64_HANDOFF = (
    S6
    / "artifacts"
    / "block64"
    / "block64_to_block65_handoff.json"
)

BLOCK64_FREEZE = (
    S6
    / "artifacts"
    / "block64"
    / "block64_part2b2_numeric_freeze_manifest.json"
)

BLOCK64_NUMERIC = (
    S6
    / "configs"
    / "block64_mc_grid_numeric_freeze.json"
)

BLOCK64_REPORT = (
    S6
    / "reports"
    / "block64_part2b2_development_numeric_freeze.json"
)

P_OCC_MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "probabilistic_occupancy.py"
)

BLOCK62_CLOSURE = (
    S6
    / "reports"
    / "block62_closure.json"
)

ILLUMINATION_MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "illumination.py"
)

MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "predictive_mask.py"
)

TEST = (
    S6
    / "tests"
    / "test_block65_part1_parameterized_mask.py"
)

CONTRACT = (
    S6
    / "configs"
    / "block65_part1_parameterized_mask_contract.json"
)

REPORT = (
    S6
    / "reports"
    / "block65_part1_parameterized_mask_kernel.json"
)


EXPECTED = {
    "block64_handoff":
        (
            "50c914e017d8d2aaa487c75077b49098"
            "c3ae5f2165b3048834b3a4bcc17719ae"
        ),

    "block64_freeze":
        (
            "bf0e7ee51cd9aadfced923f769dccf6d"
            "c2ff374179d6ba01428e0c891bbe1ad5"
        ),

    "block64_numeric":
        (
            "993c4248a902e7dff3a4383ac343e722"
            "2cc43ef9b9372ed2efd0ee08d6d20a73"
        ),

    "block64_report":
        (
            "06cb2487cc097a5df5b7f4966dded850"
            "9fa4771b70755146180cc13d1e64333f"
        ),

    "p_occ_module":
        (
            "d4206761fc70495563d53a6ab58aefe6"
            "db5d8bd1b2d09736bfc5bbc6b6f50e3e"
        ),

    "block62_closure":
        (
            "23b299b4dc6a123ce64a6ab1350b3ed9"
            "cb149c8fed176b37e9715d9935704d92"
        ),
}


MODULE_SOURCE = r'''from __future__ import annotations

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
'''


TEST_SOURCE = r'''from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from iscai_stage6.adb.grid import (
    IlluminationGridSpec,
)

from iscai_stage6.adb.illumination import (
    ReactiveShadowRegion,
    reactive_adb_map,
    static_adb_map,
)

from iscai_stage6.adb.probabilistic_occupancy import (
    ActorOccupancyProbability,
)

from iscai_stage6.adb.predictive_mask import (
    BinaryActorPredictiveMask,
    aggregate_binary_actor_masks,
    build_class_agnostic_predictive_mask_plan,
    exact_original_reactive_fallback_map,
    threshold_actor_occupancy,
)


HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)


def occupancy(
    prediction_id: str,
    probability,
):
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    counts = np.zeros(
        probability.shape,
        dtype=np.uint32,
    )

    return ActorOccupancyProbability(
        prediction_id=str(
            prediction_id
        ),
        sample_count=10,
        horizons_s=HORIZONS,
        occupancy_counts=counts,
        occupancy_probability=probability,
    )


def blank_probability(
    *,
    theta=3,
    radial=4,
):
    return np.zeros(
        (
            4,
            theta,
            radial,
        ),
        dtype=np.float64,
    )


def reactive_grid():
    return IlluminationGridSpec(
        theta_min_rad=-0.5,
        theta_max_rad=0.5,
        range_min_m=0.0,
        range_max_m=60.0,
        n_theta=11,
        n_range=61,
    )


def reactive_region():
    return ReactiveShadowRegion(
        theta_center_rad=0.0,
        theta_span_rad=0.4,
        r_shadow_end_m=20.0,
        r_transition_end_m=40.0,
        intensity_floor=0.0,
    )


class TestBlock65Part1ParameterizedMask(
    unittest.TestCase
):
    def test_01_strict_gamma_equality_is_not_active(
        self,
    ):
        probability = blank_probability()

        probability[
            0,
            1,
            1
        ] = 0.5

        probability[
            0,
            1,
            2
        ] = np.nextafter(
            0.5,
            1.0,
        )

        result = threshold_actor_occupancy(
            occupancy(
                "actor-a",
                probability,
            ),
            gamma=0.5,
        )

        self.assertFalse(
            result.mask[
                0,
                1,
                1
            ]
        )

        self.assertTrue(
            result.mask[
                0,
                1,
                2
            ]
        )

    def test_02_below_gamma_is_not_active(
        self,
    ):
        probability = blank_probability()

        probability[
            1,
            0,
            0
        ] = 0.499

        result = threshold_actor_occupancy(
            occupancy(
                "actor-a",
                probability,
            ),
            gamma=0.5,
        )

        self.assertFalse(
            result.mask[
                1,
                0,
                0
            ]
        )

    def test_03_gamma_zero_is_strict_positive_support(
        self,
    ):
        probability = blank_probability()

        probability[
            2,
            2,
            3
        ] = 0.1

        result = threshold_actor_occupancy(
            occupancy(
                "actor-a",
                probability,
            ),
            gamma=0.0,
        )

        self.assertEqual(
            int(
                np.count_nonzero(
                    result.mask
                )
            ),
            1,
        )

    def test_04_gamma_one_produces_empty_mask(
        self,
    ):
        probability = np.ones(
            (
                4,
                3,
                4,
            ),
            dtype=np.float64,
        )

        result = threshold_actor_occupancy(
            occupancy(
                "actor-a",
                probability,
            ),
            gamma=1.0,
        )

        self.assertFalse(
            np.any(
                result.mask
            )
        )

    def test_05_invalid_gamma_is_rejected(
        self,
    ):
        item = occupancy(
            "actor-a",
            blank_probability(),
        )

        for gamma in (
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
        ):
            with self.assertRaises(
                ValueError
            ):
                threshold_actor_occupancy(
                    item,
                    gamma=gamma,
                )

    def test_06_actor_mask_shape_and_bool_dtype(
        self,
    ):
        result = threshold_actor_occupancy(
            occupancy(
                "actor-a",
                blank_probability(),
            ),
            gamma=0.5,
        )

        self.assertEqual(
            result.shape,
            (
                4,
                3,
                4,
            ),
        )

        self.assertEqual(
            result.mask.dtype,
            np.bool_,
        )

    def test_07_exact_multi_actor_formula(
        self,
    ):
        first = np.zeros(
            (
                4,
                2,
                3,
            ),
            dtype=bool,
        )

        second = np.zeros_like(
            first
        )

        first[
            0,
            0,
            0
        ] = True

        first[
            1,
            0,
            1
        ] = True

        second[
            1,
            0,
            1
        ] = True

        second[
            3,
            1,
            2
        ] = True

        actor_masks = (
            BinaryActorPredictiveMask(
                prediction_id="a",
                gamma=0.5,
                mask=first,
            ),
            BinaryActorPredictiveMask(
                prediction_id="b",
                gamma=0.5,
                mask=second,
            ),
        )

        result = aggregate_binary_actor_masks(
            actor_masks,
            grid_shape=(
                2,
                3,
            ),
        )

        stacked = np.stack(
            (
                first,
                second,
            ),
            axis=0,
        ).astype(
            np.uint8
        )

        exact_formula = (
            1
            -
            np.prod(
                1 - stacked,
                axis=0,
                dtype=np.uint8,
            )
        ).astype(
            bool
        )

        np.testing.assert_array_equal(
            result,
            exact_formula,
        )

    def test_08_binary_formula_equals_logical_or(
        self,
    ):
        rng = np.random.default_rng(
            7
        )

        masks = []

        raw = []

        for index in range(
            5
        ):
            array = (
                rng.random(
                    (
                        4,
                        3,
                        4,
                    )
                )
                >
                0.7
            )

            raw.append(
                array
            )

            masks.append(
                BinaryActorPredictiveMask(
                    prediction_id=f"a{index}",
                    gamma=0.5,
                    mask=array,
                )
            )

        result = aggregate_binary_actor_masks(
            masks,
            grid_shape=(
                3,
                4,
            ),
        )

        expected = np.logical_or.reduce(
            raw
        )

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_09_aggregation_is_order_invariant(
        self,
    ):
        p1 = blank_probability()

        p2 = blank_probability()

        p1[
            0,
            0,
            0
        ] = 0.8

        p2[
            2,
            1,
            2
        ] = 0.9

        m1 = threshold_actor_occupancy(
            occupancy(
                "a",
                p1,
            ),
            gamma=0.5,
        )

        m2 = threshold_actor_occupancy(
            occupancy(
                "b",
                p2,
            ),
            gamma=0.5,
        )

        first = aggregate_binary_actor_masks(
            (
                m1,
                m2,
            ),
            grid_shape=(
                3,
                4,
            ),
        )

        second = aggregate_binary_actor_masks(
            (
                m2,
                m1,
            ),
            grid_shape=(
                3,
                4,
            ),
        )

        np.testing.assert_array_equal(
            first,
            second,
        )

    def test_10_empty_predictive_actor_set_is_all_false(
        self,
    ):
        result = aggregate_binary_actor_masks(
            (),
            grid_shape=(
                3,
                4,
            ),
        )

        self.assertEqual(
            result.shape,
            (
                4,
                3,
                4,
            ),
        )

        self.assertFalse(
            np.any(
                result
            )
        )

    def test_11_reactive_fallback_is_exact_block62_callthrough(
        self,
    ):
        grid = reactive_grid()

        region = reactive_region()

        direct = reactive_adb_map(
            grid,
            (
                region,
            ),
        )

        wrapped = (
            exact_original_reactive_fallback_map(
                grid,
                (
                    region,
                ),
            )
        )

        np.testing.assert_array_equal(
            wrapped,
            direct,
        )

    def test_12_reactive_fallback_is_not_binarized(
        self,
    ):
        result = (
            exact_original_reactive_fallback_map(
                reactive_grid(),
                (
                    reactive_region(),
                ),
            )
        )

        self.assertTrue(
            np.any(
                (
                    result > 0.0
                )
                &
                (
                    result < 1.0
                )
            )
        )

    def test_13_empty_reactive_fallback_equals_static(
        self,
    ):
        grid = reactive_grid()

        fallback = (
            exact_original_reactive_fallback_map(
                grid,
                (),
            )
        )

        static = static_adb_map(
            grid
        )

        np.testing.assert_array_equal(
            fallback,
            static,
        )

    def test_14_plan_keeps_predictive_and_reactive_outputs_separate(
        self,
    ):
        grid = reactive_grid()

        probability = np.zeros(
            (
                4,
                *grid.shape,
            ),
            dtype=np.float64,
        )

        probability[
            0,
            5,
            10
        ] = 0.9

        plan = (
            build_class_agnostic_predictive_mask_plan(
                (
                    occupancy(
                        "matched-a",
                        probability,
                    ),
                ),
                gamma=0.5,
                grid_shape=grid.shape,
                reactive_grid=grid,
                reactive_fallback_regions=(
                    reactive_region(),
                ),
            )
        )

        self.assertEqual(
            plan.predictive_mask.shape,
            (
                4,
                *grid.shape,
            ),
        )

        self.assertEqual(
            plan
            .reactive_fallback_illumination
            .shape,
            grid.shape,
        )

        self.assertTrue(
            plan.predictive_mask[
                0,
                5,
                10
            ]
        )

        self.assertFalse(
            plan
            .final_illumination_composition_applied
        )

    def test_15_duplicate_prediction_ids_are_rejected(
        self,
    ):
        item = occupancy(
            "duplicate",
            blank_probability(),
        )

        with self.assertRaises(
            ValueError
        ):
            build_class_agnostic_predictive_mask_plan(
                (
                    item,
                    item,
                ),
                gamma=0.5,
                grid_shape=(
                    3,
                    4,
                ),
            )

    def test_16_predictive_reactive_shape_mismatch_rejected(
        self,
    ):
        grid = reactive_grid()

        with self.assertRaises(
            ValueError
        ):
            build_class_agnostic_predictive_mask_plan(
                (),
                gamma=0.5,
                grid_shape=(
                    3,
                    4,
                ),
                reactive_grid=grid,
            )

    def test_17_gamma_has_no_default(
        self,
    ):
        signature = inspect.signature(
            threshold_actor_occupancy
        )

        gamma = signature.parameters[
            "gamma"
        ]

        self.assertIs(
            gamma.default,
            inspect.Parameter.empty,
        )

    def test_18_class_policy_not_in_part1_api(
        self,
    ):
        functions = (
            threshold_actor_occupancy,
            build_class_agnostic_predictive_mask_plan,
        )

        forbidden = (
            "class",
            "margin",
            "floor",
            "closing_speed",
            "pedestrian",
            "cyclist",
        )

        for function in functions:
            names = tuple(
                inspect.signature(
                    function
                ).parameters
            )

            self.assertFalse(
                any(
                    token
                    in
                    name.lower()
                    for name in names
                    for token in forbidden
                )
            )


if __name__ == "__main__":
    unittest.main()
'''


class ControlledBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(
        condition
    ):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_exact_text(
    path: Path,
    text: str,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = path.read_text(
            encoding="utf-8"
        )

        require(
            existing == text,
            (
                "Refusing to overwrite non-identical "
                f"existing file: {path}"
            ),
        )

        return "EXACT_EXISTING"

    path.write_text(
        text,
        encoding="utf-8",
    )

    return "CREATED"


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return sha256_file(
        path
    )


def run_tests(
    pattern: str,
) -> dict[str, Any]:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            pattern,
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "tests":
            (
                int(
                    match.group(
                        1
                    )
                )
                if match
                else None
            ),

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -40:
                ]
            ),
    }


def main() -> None:
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.5 PART1 — PARAMETERIZED CLASS-AGNOSTIC MASK KERNEL"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # A. Frozen upstream seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN BLOCK6.4 / BLOCK6.2 SEAL ====="
    )

    for path, expected, label in (
        (
            BLOCK64_HANDOFF,
            EXPECTED[
                "block64_handoff"
            ],
            "Block6.4 handoff",
        ),
        (
            BLOCK64_FREEZE,
            EXPECTED[
                "block64_freeze"
            ],
            "Block6.4 freeze",
        ),
        (
            BLOCK64_NUMERIC,
            EXPECTED[
                "block64_numeric"
            ],
            "Block6.4 numerics",
        ),
        (
            BLOCK64_REPORT,
            EXPECTED[
                "block64_report"
            ],
            "Block6.4 report",
        ),
        (
            P_OCC_MODULE,
            EXPECTED[
                "p_occ_module"
            ],
            "P_occ module",
        ),
        (
            BLOCK62_CLOSURE,
            EXPECTED[
                "block62_closure"
            ],
            "Block6.2 closure",
        ),
    ):
        require(
            path.is_file(),
            f"Missing {label}: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:20s} = EXACT PASS"
        )

    numeric = json.loads(
        BLOCK64_NUMERIC.read_text(
            encoding="utf-8"
        )
    )

    require(
        numeric[
            "MC"
        ][
            "runtime_sample_count"
        ]
        ==
        8192,
        (
            "Frozen MC runtime N changed."
        ),
    )

    require(
        numeric[
            "MC"
        ][
            "runtime_seed"
        ]
        ==
        20260821,
        (
            "Frozen MC seed changed."
        ),
    )

    require(
        numeric[
            "scientific_illumination_grid"
        ][
            "shape"
        ]
        ==
        [
            501,
            301,
        ],
        (
            "Frozen scientific grid changed."
        ),
    )

    print(
        "MC N / seed        = 8192 / 20260821"
    )

    print(
        "scientific grid    = 501 x 301"
    )

    illumination_sha = sha256_file(
        ILLUMINATION_MODULE
    )

    print(
        "illumination module SHA256 =",
        illumination_sha,
    )

    # ========================================================
    # B. Frozen reactive API runtime structure gate
    # ========================================================

    print()
    print(
        "===== B. EXACT REACTIVE FALLBACK API GATE ====="
    )

    import iscai_stage6.adb.illumination as illumination

    for name in (
        "ReactiveShadowRegion",
        "reactive_adb_map",
        "static_adb_map",
    ):
        require(
            hasattr(
                illumination,
                name,
            ),
            (
                "Frozen Block6.2 API missing: "
                f"{name}"
            ),
        )

    reactive_signature = (
        inspect.signature(
            illumination.reactive_adb_map
        )
    )

    forbidden = {
        "posterior",
        "covariance",
        "future_gt",
        "codebook",
        "beam_id",
        "receiver",
    }

    require(
        set(
            reactive_signature.parameters
        ).isdisjoint(
            forbidden
        ),
        (
            "Frozen reactive API now exposes "
            "predictive/communication inputs."
        ),
    )

    print(
        "reactive_adb_map signature =",
        reactive_signature,
    )

    print(
        "reactive predictive input  = NO"
    )

    print(
        "reactive communication input = NO"
    )

    # ========================================================
    # C. Materialize implementation
    # ========================================================

    print()
    print(
        "===== C. MATERIALIZE BLOCK6.5 PART1 ====="
    )

    module_status = write_exact_text(
        MODULE,
        MODULE_SOURCE,
    )

    test_status = write_exact_text(
        TEST,
        TEST_SOURCE,
    )

    compile_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(
                MODULE
            ),
            str(
                TEST
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    require(
        compile_result.returncode
        ==
        0,
        (
            "Block6.5 Part1 compile failed:\n"
            +
            compile_result.stdout
        ),
    )

    module_sha = sha256_file(
        MODULE
    )

    test_sha = sha256_file(
        TEST
    )

    print(
        "module status =",
        module_status,
    )

    print(
        "module =",
        MODULE,
    )

    print(
        "module SHA256 =",
        module_sha,
    )

    print(
        "test status =",
        test_status,
    )

    print(
        "test SHA256 =",
        test_sha,
    )

    print(
        "compile = PASS"
    )

    # ========================================================
    # D. Dedicated tests
    # ========================================================

    print()
    print(
        "===== D. BLOCK6.5 PART1 DEDICATED TESTS ====="
    )

    dedicated = run_tests(
        "test_block65_part1_parameterized_mask.py"
    )

    print(
        dedicated[
            "tail"
        ]
    )

    require(
        dedicated[
            "returncode"
        ]
        ==
        0,
        (
            "Block6.5 Part1 dedicated "
            "tests failed."
        ),
    )

    require(
        dedicated[
            "tests"
        ]
        ==
        18,
        (
            "Expected 18 Block6.5 Part1 "
            f"tests, got {dedicated['tests']}."
        ),
    )

    print(
        "dedicated tests = PASS | 18"
    )

    # ========================================================
    # E. Full regression
    # ========================================================

    print()
    print(
        "===== E. FULL STAGE6 REGRESSION ====="
    )

    regression = run_tests(
        "test_*.py"
    )

    print(
        regression[
            "tail"
        ]
    )

    require(
        regression[
            "returncode"
        ]
        ==
        0,
        (
            "Full Stage6 regression failed."
        ),
    )

    require(
        regression[
            "tests"
        ]
        ==
        145,
        (
            "Expected frozen 127 + new 18 "
            "Stage6 tests = 145; got "
            f"{regression['tests']}."
        ),
    )

    print(
        "full regression = PASS | 145"
    )

    # ========================================================
    # F. Contract
    # ========================================================

    print()
    print(
        "===== F. BLOCK6.5 PART1 CONTRACT ====="
    )

    contract_payload = {
        "stage":
            6,

        "block":
            "6.5-Part1",

        "status":
            (
                "PASS_PARAMETERIZED_CLASS_AGNOSTIC_"
                "PREDICTIVE_MASK_CONTRACT"
            ),

        "frozen_upstream": {
            "block64_handoff_sha256":
                EXPECTED[
                    "block64_handoff"
                ],

            "block64_freeze_sha256":
                EXPECTED[
                    "block64_freeze"
                ],

            "block64_numeric_sha256":
                EXPECTED[
                    "block64_numeric"
                ],

            "block64_report_sha256":
                EXPECTED[
                    "block64_report"
                ],

            "P_occ_module_sha256":
                EXPECTED[
                    "p_occ_module"
                ],

            "block62_closure_sha256":
                EXPECTED[
                    "block62_closure"
                ],

            "illumination_module_sha256":
                illumination_sha,
        },

        "frozen_numeric_context": {
            "MC_N":
                8192,

            "MC_seed":
                20260821,

            "grid_shape":
                [
                    501,
                    301,
                ],
        },

        "per_actor_mask": {
            "equation":
                (
                    "M_i_tau = "
                    "1[P_occ_i(theta,r,tau) > gamma]"
                ),

            "comparison":
                "STRICT_GREATER_THAN",

            "equal_to_gamma_is_active":
                False,

            "gamma_default":
                None,

            "gamma_frozen":
                False,

            "allowed_runtime_domain":
                "[0,1]",
        },

        "multi_actor_predictive_mask": {
            "equation":
                (
                    "M_tau = "
                    "1 - product_i(1 - M_i_tau)"
                ),

            "binary_OR_equivalence":
                True,

            "actor_order_invariant":
                True,

            "empty_actor_set":
                "all_false",
        },

        "reactive_fallback": {
            "applies_to":
                "unmatched_causal_ADB_actors",

            "implementation":
                (
                    "exact_call_through_to_frozen_"
                    "Block6.2_reactive_adb_map"
                ),

            "output":
                "continuous_normalized_intensity_[0,1]",

            "binary_conversion":
                False,

            "raised_cosine_preserved":
                True,

            "PartA_floor_semantics_preserved":
                True,

            "multi_region_rule":
                "frozen_Block6.2_elementwise_minimum",
        },

        "representation_boundary": {
            "predictive_binary_mask_shape":
                "[4,N_theta,N_range]",

            "reactive_fallback_shape":
                "[N_theta,N_range]",

            "kept_separate":
                True,

            "final_predictive_reactive_intensity_composition":
                False,

            "reason":
                (
                    "do_not_invent_dimming_depth_or_"
                    "destroy_reactive_raised_cosine_"
                    "before_policy_freeze"
                ),
        },

        "class_policy": {
            "class_agnostic":
                True,

            "class_margin":
                False,

            "class_specific_gamma":
                False,

            "class_specific_floor":
                False,

            "pedestrian_policy":
                False,

            "cyclist_policy":
                False,

            "vehicle_policy":
                False,
        },

        "still_deferred": {
            "development_gamma_selection":
                True,

            "class_aware_policy":
                True,

            "dimming_floors":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,

            "formal_evaluation":
                True,
        },

        "scientific_execution": {
            "real_posterior_sampling":
                False,

            "model_forward":
                False,

            "formal_evaluation":
                False,

            "training":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,
        },

        "implementation": {
            "module":
                str(
                    MODULE
                ),

            "module_sha256":
                module_sha,

            "test":
                str(
                    TEST
                ),

            "test_sha256":
                test_sha,
        },
    }

    contract_sha = write_json(
        CONTRACT,
        contract_payload,
    )

    print(
        "contract =",
        CONTRACT,
    )

    print(
        "contract SHA256 =",
        contract_sha,
    )

    # ========================================================
    # G. Report
    # ========================================================

    print()
    print(
        "===== G. BLOCK6.5 PART1 REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.5-Part1",

        "status":
            (
                "PASS_BLOCK65_PART1_"
                "PARAMETERIZED_MASK_KERNEL"
            ),

        "implementation": {
            "module":
                str(
                    MODULE
                ),

            "module_sha256":
                module_sha,

            "test":
                str(
                    TEST
                ),

            "test_sha256":
                test_sha,
        },

        "contract": {
            "path":
                str(
                    CONTRACT
                ),

            "sha256":
                contract_sha,
        },

        "tests": {
            "dedicated":
                {
                    "status":
                        "PASS",

                    "count":
                        18,
                },

            "full_stage6":
                {
                    "status":
                        "PASS",

                    "count":
                        145,
                },
        },

        "semantics": {
            "strict_gamma_threshold":
                True,

            "gamma_equal_probability_inactive":
                True,

            "exact_multi_actor_formula":
                True,

            "binary_OR_equivalent":
                True,

            "class_agnostic":
                True,

            "original_reactive_fallback_exact":
                True,

            "reactive_fallback_binarized":
                False,

            "final_intensity_composition":
                False,
        },

        "scope": {
            "gamma_frozen":
                False,

            "class_policy_frozen":
                False,

            "temporal_smoothing":
                False,

            "actuation_rate_limits":
                False,

            "formal_evaluation":
                False,

            "model_forward":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "next":
            (
                "BLOCK6.5_PART2_REAL_DEVELOPMENT_"
                "GAMMA_CURVES_AND_PREREGISTRATION"
            ),
    }

    report_sha = write_json(
        REPORT,
        report_payload,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    # ========================================================
    # H. Final
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.5 PART1 — FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "STATUS = "
        "PASS_BLOCK65_PART1_PARAMETERIZED_MASK_KERNEL"
    )

    print(
        "per-actor mask             = "
        "1[P_occ > gamma]"
    )

    print(
        "comparison                 = STRICT >"
    )

    print(
        "P_occ == gamma             = INACTIVE"
    )

    print(
        "multi-actor aggregation    = "
        "1 - product(1-M_i)"
    )

    print(
        "binary OR equivalence      = PASS"
    )

    print(
        "class-agnostic             = YES"
    )

    print(
        "reactive fallback          = "
        "EXACT FROZEN BLOCK6.2"
    )

    print(
        "reactive fallback binary   = NO"
    )

    print(
        "raised cosine preserved    = YES"
    )

    print(
        "final intensity composition= NOT YET"
    )

    print(
        "gamma frozen               = NO"
    )

    print(
        "class policy frozen        = NO"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "model forward              = NO"
    )

    print(
        "training/recalibration     = NO / NO"
    )

    print(
        "dedicated tests            = PASS | 18"
    )

    print(
        "full Stage6 regression     = PASS | 145"
    )

    print(
        "module SHA256              =",
        module_sha,
    )

    print(
        "contract SHA256            =",
        contract_sha,
    )

    print(
        "report SHA256              =",
        report_sha,
    )

    print(
        "NEXT = BLOCK6.5 PART2 "
        "REAL DEVELOPMENT GAMMA CURVES + PREREGISTRATION"
    )

    print(
        "terminal remains open = YES"
    )

    print(
        "=" * 72
    )


try:
    main()

except BaseException as exc:
    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK6.5 PART1 — CONTROLLED BLOCK"
    )

    print(
        "=" * 72
    )

    print(
        "exception type =",
        type(
            exc
        ).__name__,
    )

    print(
        "exception      =",
        str(
            exc
        ),
    )

    print()

    traceback.print_exc(
        limit=18
    )

    print()

    print(
        "gamma frozen          = NO"
    )

    print(
        "formal evaluation     = NO"
    )

    print(
        "model forward         = NO"
    )

    print(
        "training/recalibration= NO / NO"
    )

    print(
        "upstream modification = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

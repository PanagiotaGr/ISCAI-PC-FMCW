from __future__ import annotations

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

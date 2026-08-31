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
    angular_shadow_mask,
    combine_illumination_maps_minimum,
    part_a_radial_profile,
    reactive_adb_map,
    single_region_illumination_map,
    static_adb_map,
)


class TestBlock62StaticReactiveKernel(
    unittest.TestCase
):

    def grid(self):

        return IlluminationGridSpec(
            theta_min_rad=-math.pi,
            theta_max_rad=math.pi,
            range_min_m=0.0,
            range_max_m=100.0,
            n_theta=72,
            n_range=100,
        )

    def test_static_adb_is_full_illumination(self):

        result = static_adb_map(
            self.grid()
        )

        self.assertEqual(
            result.shape,
            (72, 100),
        )

        self.assertTrue(
            np.all(
                result == 1.0
            )
        )

    def test_exact_part_a_floor_zero_shadow(self):

        values = (
            part_a_radial_profile(
                np.asarray(
                    [
                        0.0,
                        10.0,
                        20.0,
                    ]
                ),
                r_shadow_end_m=20.0,
                r_transition_end_m=40.0,
                intensity_floor=0.0,
            )
        )

        np.testing.assert_allclose(
            values,
            np.zeros(3),
            atol=1.0e-15,
        )

    def test_exact_part_a_transition_midpoint(self):

        value = part_a_radial_profile(
            30.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=0.0,
        )

        expected = (
            0.5
            *
            (
                1.0
                -
                math.cos(
                    math.pi * 0.5
                )
            )
        )

        self.assertAlmostEqual(
            value,
            expected,
            places=14,
        )

    def test_exact_part_a_transition_endpoint(self):

        value = part_a_radial_profile(
            40.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=0.0,
        )

        self.assertEqual(
            value,
            1.0,
        )

    def test_outside_transition_is_full_light(self):

        values = part_a_radial_profile(
            np.asarray(
                [
                    40.0,
                    50.0,
                    100.0,
                ]
            ),
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        np.testing.assert_allclose(
            values,
            np.ones(3),
            atol=0.0,
        )

    def test_nonzero_floor_lifts_same_curve(self):

        floor = 0.25

        start = part_a_radial_profile(
            20.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=floor,
        )

        middle = part_a_radial_profile(
            30.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=floor,
        )

        end = part_a_radial_profile(
            40.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=floor,
        )

        self.assertAlmostEqual(
            start,
            0.25,
        )

        self.assertAlmostEqual(
            middle,
            0.625,
        )

        self.assertAlmostEqual(
            end,
            1.0,
        )

    def test_invalid_radial_thresholds_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            part_a_radial_profile(
                20.0,
                r_shadow_end_m=30.0,
                r_transition_end_m=30.0,
            )

    def test_negative_range_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            part_a_radial_profile(
                -1.0,
                r_shadow_end_m=20.0,
                r_transition_end_m=40.0,
            )

    def test_angular_shadow_wraps_across_pi(self):

        theta = np.asarray(
            [
                math.pi - 0.05,
                -math.pi + 0.05,
                0.0,
            ]
        )

        mask = angular_shadow_mask(
            theta,
            theta_center_rad=(
                math.pi
            ),
            theta_span_rad=0.2,
        )

        self.assertTrue(
            bool(mask[0])
        )

        self.assertTrue(
            bool(mask[1])
        )

        self.assertFalse(
            bool(mask[2])
        )

    def test_outside_angular_region_stays_full(self):

        grid = self.grid()

        region = ReactiveShadowRegion(
            theta_center_rad=0.0,
            theta_span_rad=0.1,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        result = (
            single_region_illumination_map(
                grid,
                region,
            )
        )

        far_angle_index = int(
            np.argmax(
                np.abs(
                    grid.theta_centers_rad
                )
            )
        )

        self.assertTrue(
            np.all(
                result[
                    far_angle_index,
                    :
                ]
                ==
                1.0
            )
        )

    def test_single_region_map_has_dimming(self):

        grid = self.grid()

        region = ReactiveShadowRegion(
            theta_center_rad=0.0,
            theta_span_rad=0.5,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        result = (
            single_region_illumination_map(
                grid,
                region,
            )
        )

        self.assertLess(
            float(
                np.min(
                    result
                )
            ),
            1.0,
        )

        self.assertGreaterEqual(
            float(
                np.min(
                    result
                )
            ),
            0.0,
        )

        self.assertLessEqual(
            float(
                np.max(
                    result
                )
            ),
            1.0,
        )

    def test_multi_actor_combination_is_elementwise_minimum(self):

        grid = self.grid()

        first = ReactiveShadowRegion(
            theta_center_rad=-0.2,
            theta_span_rad=0.4,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        second = ReactiveShadowRegion(
            theta_center_rad=0.2,
            theta_span_rad=0.4,
            r_shadow_end_m=30.0,
            r_transition_end_m=50.0,
        )

        first_map = (
            single_region_illumination_map(
                grid,
                first,
            )
        )

        second_map = (
            single_region_illumination_map(
                grid,
                second,
            )
        )

        combined = (
            reactive_adb_map(
                grid,
                (
                    first,
                    second,
                ),
            )
        )

        expected = np.minimum(
            first_map,
            second_map,
        )

        np.testing.assert_allclose(
            combined,
            expected,
            atol=0.0,
        )

    def test_multi_actor_combination_order_invariant(self):

        grid = self.grid()

        first = ReactiveShadowRegion(
            theta_center_rad=-0.3,
            theta_span_rad=0.3,
            r_shadow_end_m=15.0,
            r_transition_end_m=35.0,
        )

        second = ReactiveShadowRegion(
            theta_center_rad=0.3,
            theta_span_rad=0.3,
            r_shadow_end_m=25.0,
            r_transition_end_m=45.0,
        )

        a = reactive_adb_map(
            grid,
            (
                first,
                second,
            ),
        )

        b = reactive_adb_map(
            grid,
            (
                second,
                first,
            ),
        )

        np.testing.assert_array_equal(
            a,
            b,
        )

    def test_empty_reactive_regions_equal_static_baseline(self):

        grid = self.grid()

        reactive = reactive_adb_map(
            grid,
            (),
        )

        static = static_adb_map(
            grid
        )

        np.testing.assert_array_equal(
            reactive,
            static,
        )

    def test_combiner_rejects_out_of_range_intensity(self):

        with self.assertRaises(
            ValueError
        ):
            combine_illumination_maps_minimum(
                (
                    np.asarray(
                        [
                            [1.1]
                        ]
                    ),
                )
            )

    def test_reactive_region_rejects_future_semantics(self):

        with self.assertRaises(
            ValueError
        ):
            ReactiveShadowRegion(
                theta_center_rad=0.0,
                theta_span_rad=0.2,
                r_shadow_end_m=20.0,
                r_transition_end_m=40.0,
                source_semantics="future_gt",
            )

    def test_reactive_api_has_no_predictive_or_communication_input(self):

        signature = inspect.signature(
            reactive_adb_map
        )

        names = set(
            signature.parameters
        )

        forbidden = {
            "posterior",
            "covariance",
            "future_gt",
            "codebook",
            "beam_id",
            "receiver",
        }

        self.assertTrue(
            names.isdisjoint(
                forbidden
            )
        )

    def test_region_has_no_class_policy_field(self):

        fields = set(
            ReactiveShadowRegion
            .__dataclass_fields__
        )

        forbidden = {
            "actor_class",
            "class_margin",
            "confidence_threshold",
            "closing_speed",
            "prediction_horizon",
        }

        self.assertTrue(
            fields.isdisjoint(
                forbidden
            )
        )


if __name__ == "__main__":
    unittest.main()

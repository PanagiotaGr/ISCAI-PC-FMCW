from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
import unittest

import numpy as np

from iscai_stage6.adb.part_a_reactive import (
    CAUSAL_BOX_CENTER_FIELD,
    CAUSAL_BOX_CLASS_FIELD,
    PART_A_CONTROL_TRANSITION_LENGTH_M,
    PART_A_FIXED_DEMO_EPSILON_DEG,
    PART_A_LATERAL_SAFETY_MARGIN_M,
    PART_A_RADIAL_SAFETY_MARGIN_M,
    PART_A_TARGET_WIDTH_M,
    PART_A_TRZ_TRANSITION_LENGTH_UNITS,
    PartAReferenceGrid,
    h0_center_to_part_a_xy,
    part_a_current_vehicle_centers_from_causal_boxes,
    part_a_direct_source_reference_map_from_h0_centers,
    part_a_dynamic_geometry_from_h0_center,
    part_a_original_reactive_map_from_h0_centers,
)


class TestBlock62ExactPartAReactive(
    unittest.TestCase
):

    def test_frozen_exact_parameters(self):

        self.assertEqual(
            PART_A_TARGET_WIDTH_M,
            1.9,
        )

        self.assertEqual(
            PART_A_LATERAL_SAFETY_MARGIN_M,
            1.0,
        )

        self.assertEqual(
            PART_A_RADIAL_SAFETY_MARGIN_M,
            4.0,
        )

        self.assertEqual(
            PART_A_CONTROL_TRANSITION_LENGTH_M,
            20.0,
        )

        self.assertEqual(
            PART_A_TRZ_TRANSITION_LENGTH_UNITS,
            8.0,
        )

        self.assertEqual(
            PART_A_FIXED_DEMO_EPSILON_DEG,
            0.75,
        )

    def test_control_and_visual_transition_not_conflated(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertAlmostEqual(
            (
                geometry.control_r_max_m
                -
                geometry.control_r_min_m
            ),
            20.0,
        )

        self.assertAlmostEqual(
            (
                geometry.trz_visual_r_max_m
                -
                geometry.trz_visual_r_min_m
            ),
            8.0,
        )

    def test_h0_to_part_a_coordinate_conversion(self):

        x_A, y_A = (
            h0_center_to_part_a_xy(
                (
                    10.0,
                    2.0,
                    0.0,
                )
            )
        )

        self.assertEqual(
            x_A,
            -2.0,
        )

        self.assertEqual(
            y_A,
            10.0,
        )

    def test_angle_sign_relation(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    10.0,
                    2.0,
                    0.0,
                )
            )
        )

        theta_H = math.degrees(
            math.atan2(
                2.0,
                10.0,
            )
        )

        self.assertAlmostEqual(
            geometry.theta_c_partA_deg,
            -theta_H,
            places=13,
        )

    def test_dynamic_epsilon_exact_width_margin_formula(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        expected = math.degrees(
            math.atan(
                1.95
                /
                50.0
            )
        )

        self.assertAlmostEqual(
            geometry.epsilon_dynamic_deg,
            expected,
            places=14,
        )

    def test_fixed_demo_epsilon_not_forced_into_dynamic_path(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertNotAlmostEqual(
            geometry.epsilon_dynamic_deg,
            PART_A_FIXED_DEMO_EPSILON_DEG,
            places=6,
        )

    def test_dynamic_control_radial_thresholds(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertAlmostEqual(
            geometry.control_r_min_m,
            54.0,
        )

        self.assertAlmostEqual(
            geometry.control_r_max_m,
            74.0,
        )

    def test_reference_grid_is_exact_part_a_axis_sampling(self):

        grid = PartAReferenceGrid()

        theta = np.rad2deg(
            grid.theta_centers_rad
        )

        ranges = (
            grid.range_centers_m
        )

        self.assertEqual(
            grid.shape,
            (
                501,
                361,
            ),
        )

        self.assertAlmostEqual(
            float(theta[0]),
            -25.0,
        )

        self.assertAlmostEqual(
            float(theta[-1]),
            25.0,
        )

        self.assertAlmostEqual(
            float(theta[1] - theta[0]),
            0.1,
            places=12,
        )

        self.assertAlmostEqual(
            float(ranges[0]),
            0.0,
        )

        self.assertAlmostEqual(
            float(ranges[-1]),
            180.0,
        )

        self.assertAlmostEqual(
            float(
                ranges[1]
                -
                ranges[0]
            ),
            0.5,
            places=14,
        )

    def test_generic_kernel_matches_direct_source_transcription(self):

        grid = PartAReferenceGrid()

        centers = (
            (
                40.0,
                2.0,
                0.0,
            ),
            (
                60.0,
                -4.0,
                0.0,
            ),
        )

        implementation = (
            part_a_original_reactive_map_from_h0_centers(
                grid,
                centers,
            )
        )

        reference = (
            part_a_direct_source_reference_map_from_h0_centers(
                grid,
                centers,
            )
        )

        np.testing.assert_allclose(
            implementation,
            reference,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_empty_original_reactive_equals_static(self):

        grid = PartAReferenceGrid()

        result = (
            part_a_original_reactive_map_from_h0_centers(
                grid,
                (),
            )
        )

        np.testing.assert_array_equal(
            result,
            np.ones(
                grid.shape,
                dtype=float,
            ),
        )

    def test_original_reactive_actor_extraction_vehicle_only(self):

        vehicle = SimpleNamespace(
            **{
                CAUSAL_BOX_CENTER_FIELD:
                    (
                        20.0,
                        1.0,
                        0.0,
                    ),

                CAUSAL_BOX_CLASS_FIELD:
                    "TYPE_VEHICLE",
            }
        )

        pedestrian = SimpleNamespace(
            **{
                CAUSAL_BOX_CENTER_FIELD:
                    (
                        20.0,
                        2.0,
                        0.0,
                    ),

                CAUSAL_BOX_CLASS_FIELD:
                    "TYPE_PEDESTRIAN",
            }
        )

        cyclist = SimpleNamespace(
            **{
                CAUSAL_BOX_CENTER_FIELD:
                    (
                        20.0,
                        3.0,
                        0.0,
                    ),

                CAUSAL_BOX_CLASS_FIELD:
                    "TYPE_CYCLIST",
            }
        )

        centers = (
            part_a_current_vehicle_centers_from_causal_boxes(
                (
                    vehicle,
                    pedestrian,
                    cyclist,
                )
            )
        )

        self.assertEqual(
            centers,
            (
                (
                    20.0,
                    1.0,
                    0.0,
                ),
            ),
        )

    def test_original_reactive_api_has_no_future_posterior_covariance(self):

        signature = inspect.signature(
            part_a_original_reactive_map_from_h0_centers
        )

        forbidden = {
            "future",
            "posterior",
            "covariance",
            "class_margin",
            "codebook",
            "beam",
        }

        self.assertTrue(
            set(
                signature.parameters
            ).isdisjoint(
                forbidden
            )
        )

    def test_dynamic_geometry_api_is_center_only_legacy_baseline(self):

        signature = inspect.signature(
            part_a_dynamic_geometry_from_h0_center
        )

        self.assertEqual(
            tuple(
                signature.parameters
            ),
            (
                "center_H0_m",
            ),
        )

    def test_no_actual_box_width_is_used_in_legacy_formula(self):

        geometry_a = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    30.0,
                    0.0,
                    0.0,
                )
            )
        )

        expected = math.degrees(
            math.atan(
                (
                    1.9 / 2.0 + 1.0
                )
                /
                30.0
            )
        )

        self.assertAlmostEqual(
            geometry_a.epsilon_dynamic_deg,
            expected,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()

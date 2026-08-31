from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.class_aware_policy import (
    TYPE_CYCLIST,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
    angular_margin_cells,
    apply_actuation_rate_limit,
    compose_class_aware_illumination,
    compute_class_aware_margin,
    dilate_mask_theta,
    predictive_mask_to_illumination,
    temporal_smooth_schedule,
    threshold_occupancy_counts_strict_k,
)
from iscai_stage6.adb.part_a_reactive import (
    PART_A_LATERAL_SAFETY_MARGIN_M,
)


class StrictThresholdTests(unittest.TestCase):

    def test_strict_k_boundary(self):
        counts = np.asarray(
            [0, 1, 2, 8191, 8192],
            dtype=np.uint32,
        )

        result = (
            threshold_occupancy_counts_strict_k(
                counts,
                k=1,
                sample_count=8192,
            )
        )

        np.testing.assert_array_equal(
            result,
            np.asarray(
                [
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            ),
        )

    def test_k_zero_keeps_positive_counts(self):
        counts = np.asarray(
            [0, 1, 2],
            dtype=np.uint32,
        )

        result = (
            threshold_occupancy_counts_strict_k(
                counts,
                k=0,
            )
        )

        np.testing.assert_array_equal(
            result,
            [
                False,
                True,
                True,
            ],
        )

    def test_invalid_k_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            threshold_occupancy_counts_strict_k(
                np.zeros(
                    3,
                    dtype=np.uint32,
                ),
                k=8192,
            )


class MarginTests(unittest.TestCase):

    def test_vehicle_uncertainty_and_closing(self):
        result = compute_class_aware_margin(
            TYPE_VEHICLE,
            current_position_H0_m=(
                10.0,
                0.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                -2.0,
                0.0,
                0.0,
            ),
            covariance_H0_m2=np.diag(
                [
                    0.25,
                    4.0,
                    0.01,
                ]
            ),
            horizon_s=0.5,
            base_margin_m=0.5,
            uncertainty_multiplier=1.0,
            motion_multiplier=0.1,
        )

        self.assertAlmostEqual(
            result.lateral_predictive_sigma_m,
            2.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.closing_speed_mps,
            4.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.additional_class_margin_m,
            2.7,
            places=12,
        )

        self.assertAlmostEqual(
            result.total_lateral_margin_m,
            (
                float(
                    PART_A_LATERAL_SAFETY_MARGIN_M
                )
                +
                2.7
            ),
            places=12,
        )

    def test_pedestrian_has_no_motion_term(self):
        first = compute_class_aware_margin(
            TYPE_PEDESTRIAN,
            current_position_H0_m=(
                10.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                -5.0,
                3.0,
            ),
            covariance_H0_m2=np.eye(
                2
            ),
            horizon_s=0.5,
            base_margin_m=0.25,
            uncertainty_multiplier=2.0,
            motion_multiplier=0.0,
        )

        second = compute_class_aware_margin(
            TYPE_PEDESTRIAN,
            current_position_H0_m=(
                10.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                -5.0,
                3.0,
            ),
            covariance_H0_m2=np.eye(
                2
            ),
            horizon_s=0.5,
            base_margin_m=0.25,
            uncertainty_multiplier=2.0,
            motion_multiplier=99.0,
        )

        self.assertAlmostEqual(
            first.additional_class_margin_m,
            second.additional_class_margin_m,
            places=12,
        )

    def test_cyclist_lateral_response(self):
        result = compute_class_aware_margin(
            TYPE_CYCLIST,
            current_position_H0_m=(
                10.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                0.0,
                2.0,
            ),
            covariance_H0_m2=np.eye(
                2
            ),
            horizon_s=0.5,
            base_margin_m=0.0,
            uncertainty_multiplier=1.0,
            motion_multiplier=0.5,
        )

        self.assertAlmostEqual(
            result.cyclist_lateral_rate_mps,
            4.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.additional_class_margin_m,
            2.0,
            places=12,
        )

    def test_unknown_class_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            compute_class_aware_margin(
                "TYPE_UNKNOWN",
                current_position_H0_m=(
                    1.0,
                    0.0,
                ),
                mean_displacement_H0_m=(
                    0.0,
                    0.0,
                ),
                covariance_H0_m2=np.eye(
                    2
                ),
                horizon_s=0.1,
                base_margin_m=0.0,
                uncertainty_multiplier=1.0,
            )

    def test_angular_margin_uses_ceil(self):
        cells = angular_margin_cells(
            total_margin_m=1.0,
            predicted_range_m=10.0,
            theta_step_rad=0.01,
        )

        expected = int(
            math.ceil(
                math.atan2(
                    1.0,
                    10.0,
                )
                /
                0.01
            )
        )

        self.assertEqual(
            cells,
            expected,
        )


class MaskAndIlluminationTests(unittest.TestCase):

    def test_theta_dilation_only(self):
        mask = np.zeros(
            (
                1,
                5,
                4,
            ),
            dtype=bool,
        )

        mask[
            0,
            2,
            1,
        ] = True

        result = dilate_mask_theta(
            mask,
            dilation_cells=1,
        )

        expected = np.zeros_like(
            mask
        )

        expected[
            0,
            1:4,
            1,
        ] = True

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_theta_dilation_clips_at_boundary(self):
        mask = np.zeros(
            (
                1,
                3,
                2,
            ),
            dtype=bool,
        )

        mask[
            0,
            0,
            0,
        ] = True

        result = dilate_mask_theta(
            mask,
            dilation_cells=2,
        )

        self.assertTrue(
            np.all(
                result[
                    0,
                    :,
                    0,
                ]
            )
        )

        self.assertFalse(
            np.any(
                result[
                    0,
                    :,
                    1,
                ]
            )
        )

    def test_predictive_illumination_preserves_inactive_theta(self):
        mask = np.zeros(
            (
                1,
                2,
                5,
            ),
            dtype=bool,
        )

        mask[
            0,
            0,
            1,
        ] = True

        ranges = np.asarray(
            [
                0.0,
                5.0,
                10.0,
                15.0,
                20.0,
            ]
        )

        output = predictive_mask_to_illumination(
            mask,
            range_centers_m=ranges,
            intensity_floor=0.2,
        )

        np.testing.assert_allclose(
            output[
                0,
                1,
                :,
            ],
            1.0,
            rtol=0.0,
            atol=0.0,
        )

        self.assertTrue(
            np.all(
                output
                >=
                0.2
            )
        )

        self.assertTrue(
            np.all(
                output
                <=
                1.0
            )
        )

    def test_predictive_illumination_uses_nonzero_floor(self):
        mask = np.zeros(
            (
                1,
                1,
                4,
            ),
            dtype=bool,
        )

        mask[
            0,
            0,
            0,
        ] = True

        ranges = np.asarray(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        )

        output = predictive_mask_to_illumination(
            mask,
            range_centers_m=ranges,
            intensity_floor=0.4,
        )

        self.assertAlmostEqual(
            float(
                output[
                    0,
                    0,
                    0,
                ]
            ),
            0.4,
            places=12,
        )


class CompositionTests(unittest.TestCase):

    def test_vehicle_only_uses_minimum(self):
        vehicle = np.full(
            (
                1,
                1,
                2,
            ),
            0.2,
        )

        result = compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    vehicle,
            },
            class_active_masks={},
            class_floors={},
        )

        np.testing.assert_allclose(
            result.raw_class_aware_illumination,
            vehicle,
            rtol=0.0,
            atol=0.0,
        )

    def test_pedestrian_floor_overrides_vehicle_blackout(self):
        vehicle = np.zeros(
            (
                1,
                1,
                2,
            )
        )

        pedestrian = np.full(
            (
                1,
                1,
                2,
            ),
            0.2,
        )

        active = np.asarray(
            [
                [
                    [
                        True,
                        False,
                    ]
                ]
            ],
            dtype=bool,
        )

        result = compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    vehicle,

                TYPE_PEDESTRIAN:
                    pedestrian,
            },

            class_active_masks={
                TYPE_PEDESTRIAN:
                    active,
            },

            class_floors={
                TYPE_PEDESTRIAN:
                    0.4,
            },
        )

        self.assertAlmostEqual(
            float(
                result.raw_class_aware_illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.4,
            places=12,
        )

        self.assertAlmostEqual(
            float(
                result.raw_class_aware_illumination[
                    0,
                    0,
                    1,
                ]
            ),
            0.0,
            places=12,
        )

    def test_maximum_active_vru_floor_wins(self):
        shape = (
            1,
            1,
            1,
        )

        active = np.ones(
            shape,
            dtype=bool,
        )

        result = compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    np.zeros(
                        shape
                    ),

                TYPE_PEDESTRIAN:
                    np.zeros(
                        shape
                    ),

                TYPE_CYCLIST:
                    np.zeros(
                        shape
                    ),
            },

            class_active_masks={
                TYPE_PEDESTRIAN:
                    active,

                TYPE_CYCLIST:
                    active,
            },

            class_floors={
                TYPE_PEDESTRIAN:
                    0.3,

                TYPE_CYCLIST:
                    0.6,
            },
        )

        self.assertAlmostEqual(
            float(
                result.raw_class_aware_illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.6,
            places=12,
        )


class TemporalTests(unittest.TestCase):

    def test_temporal_smoothing_exact_first_step(self):
        raw = np.ones(
            (
                1,
                1,
                1,
            )
        )

        current = np.zeros(
            (
                1,
                1,
            )
        )

        output = temporal_smooth_schedule(
            raw,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            time_constant_s=0.1,
        )

        expected = (
            1.0
            -
            math.exp(
                -1.0
            )
        )

        self.assertAlmostEqual(
            float(
                output[
                    0,
                    0,
                    0,
                ]
            ),
            expected,
            places=12,
        )

    def test_temporal_smoothing_is_recursive(self):
        raw = np.asarray(
            [
                [
                    [
                        0.0,
                    ]
                ],
                [
                    [
                        1.0,
                    ]
                ],
            ]
        )

        current = np.ones(
            (
                1,
                1,
            )
        )

        output = temporal_smooth_schedule(
            raw,
            current_illumination=current,
            horizons_s=(
                0.1,
                0.3,
            ),
            time_constant_s=0.2,
        )

        alpha1 = (
            1.0
            -
            math.exp(
                -0.1
                /
                0.2
            )
        )

        first = (
            1.0
            -
            alpha1
        )

        alpha2 = (
            1.0
            -
            math.exp(
                -0.2
                /
                0.2
            )
        )

        expected_second = (
            alpha2
            +
            (
                1.0
                -
                alpha2
            )
            *
            first
        )

        self.assertAlmostEqual(
            float(
                output[
                    1,
                    0,
                    0,
                ]
            ),
            expected_second,
            places=12,
        )


class RateLimitTests(unittest.TestCase):

    def test_finite_dimming_rate(self):
        smooth = np.zeros(
            (
                1,
                1,
                1,
            )
        )

        current = np.ones(
            (
                1,
                1,
            )
        )

        result = apply_actuation_rate_limit(
            smooth,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            rho_dim_per_s=2.0,
            rho_bright_per_s=1.0,
        )

        self.assertAlmostEqual(
            float(
                result.illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.8,
            places=12,
        )

    def test_finite_brightening_rate(self):
        smooth = np.ones(
            (
                1,
                1,
                1,
            )
        )

        current = np.zeros(
            (
                1,
                1,
            )
        )

        result = apply_actuation_rate_limit(
            smooth,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            rho_dim_per_s=2.0,
            rho_bright_per_s=1.0,
        )

        self.assertAlmostEqual(
            float(
                result.illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.1,
            places=12,
        )

    def test_vru_floor_safety_override_is_counted(self):
        smooth = np.zeros(
            (
                1,
                1,
                2,
            )
        )

        current = np.ones(
            (
                1,
                2,
            )
        )

        guard = np.asarray(
            [
                [
                    [
                        0.9,
                        0.0,
                    ]
                ]
            ]
        )

        result = apply_actuation_rate_limit(
            smooth,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            rho_dim_per_s=2.0,
            rho_bright_per_s=1.0,
            vru_floor_guard=guard,
        )

        self.assertAlmostEqual(
            float(
                result.illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.9,
            places=12,
        )

        self.assertEqual(
            result.safety_override_count,
            1,
        )

        self.assertAlmostEqual(
            result.safety_override_fraction,
            0.5,
            places=12,
        )

    def test_invalid_rate_order_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            apply_actuation_rate_limit(
                np.zeros(
                    (
                        1,
                        1,
                        1,
                    )
                ),
                current_illumination=
                    np.ones(
                        (
                            1,
                            1,
                        )
                    ),
                horizons_s=(
                    0.1,
                ),
                rho_dim_per_s=1.0,
                rho_bright_per_s=2.0,
            )


if __name__ == "__main__":
    unittest.main()

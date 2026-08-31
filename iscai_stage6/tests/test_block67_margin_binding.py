from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.class_aware_policy import (
    MarginComputation,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
    angular_margin_cells,
    compute_class_aware_margin,
)

from iscai_stage6.adb.class_margin_binding import (
    bind_predictive_class_margin,
)


class Block67MarginBindingTest(
    unittest.TestCase
):

    def _margin(
        self,
        *,
        extra: float,
        generic: float = 1.0,
        actor_class: str = TYPE_VEHICLE,
        actor_range: float = 20.0,
    ) -> MarginComputation:

        return MarginComputation(
            actor_class=actor_class,

            predicted_center_H0_xy_m=(
                actor_range,
                0.0,
            ),

            predicted_range_m=actor_range,

            lateral_predictive_sigma_m=0.0,

            closing_speed_mps=0.0,

            cyclist_lateral_rate_mps=0.0,

            part_a_generic_lateral_margin_m=(
                generic
            ),

            additional_class_margin_m=(
                extra
            ),

            total_lateral_margin_m=(
                generic
                +
                extra
            ),
        )

    def test_runtime_uses_extra_only(
        self,
    ):

        margin = self._margin(
            extra=0.5
        )

        binding = (
            bind_predictive_class_margin(
                margin,
                theta_step_rad=0.01,
            )
        )

        expected = angular_margin_cells(
            total_margin_m=0.5,
            predicted_range_m=20.0,
            theta_step_rad=0.01,
        )

        wrong_double_count = (
            angular_margin_cells(
                total_margin_m=1.5,
                predicted_range_m=20.0,
                theta_step_rad=0.01,
            )
        )

        self.assertEqual(
            binding.predictive_dilation_margin_m,
            0.5,
        )

        self.assertEqual(
            binding.predictive_dilation_cells,
            expected,
        )

        self.assertEqual(
            binding.total_margin_reference_cells,
            wrong_double_count,
        )

        self.assertNotEqual(
            binding.predictive_dilation_cells,
            wrong_double_count,
        )

        self.assertFalse(
            binding.generic_margin_reapplied
        )

    def test_zero_extra_means_no_predictive_dilation(
        self,
    ):

        margin = self._margin(
            extra=0.0
        )

        binding = (
            bind_predictive_class_margin(
                margin,
                theta_step_rad=0.01,
            )
        )

        self.assertEqual(
            binding.predictive_dilation_margin_m,
            0.0,
        )

        self.assertEqual(
            binding.predictive_dilation_cells,
            0,
        )

        self.assertGreater(
            binding.total_margin_reference_cells,
            0,
        )

    def test_real_margin_computation_binds_extra(
        self,
    ):

        covariance = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        margin = compute_class_aware_margin(
            TYPE_PEDESTRIAN,

            current_position_H0_m=(
                20.0,
                0.0,
                0.0,
            ),

            mean_displacement_H0_m=(
                0.0,
                0.0,
                0.0,
            ),

            covariance_H0_m2=covariance,

            horizon_s=0.5,

            base_margin_m=0.0,

            uncertainty_multiplier=1.0,

            motion_multiplier=0.0,
        )

        self.assertAlmostEqual(
            margin.additional_class_margin_m,
            0.5,
            places=12,
        )

        binding = (
            bind_predictive_class_margin(
                margin,
                theta_step_rad=0.01,
            )
        )

        self.assertAlmostEqual(
            binding.predictive_dilation_margin_m,
            margin.additional_class_margin_m,
            places=12,
        )

        self.assertFalse(
            binding.generic_margin_reapplied
        )

    def test_inconsistent_total_rejected(
        self,
    ):

        malformed = MarginComputation(
            actor_class=TYPE_VEHICLE,

            predicted_center_H0_xy_m=(
                20.0,
                0.0,
            ),

            predicted_range_m=20.0,

            lateral_predictive_sigma_m=0.0,

            closing_speed_mps=0.0,

            cyclist_lateral_rate_mps=0.0,

            part_a_generic_lateral_margin_m=1.0,

            additional_class_margin_m=0.5,

            total_lateral_margin_m=9.0,
        )

        with self.assertRaises(
            ValueError
        ):
            bind_predictive_class_margin(
                malformed,
                theta_step_rad=0.01,
            )

    def test_cell_rule_remains_frozen_ceil(
        self,
    ):

        margin = self._margin(
            extra=0.5,
            actor_range=20.0,
        )

        binding = (
            bind_predictive_class_margin(
                margin,
                theta_step_rad=0.01,
            )
        )

        expected = int(
            math.ceil(
                math.atan2(
                    0.5,
                    20.0,
                )
                /
                0.01
            )
        )

        self.assertEqual(
            binding.predictive_dilation_cells,
            expected,
        )


if __name__ == "__main__":
    unittest.main()

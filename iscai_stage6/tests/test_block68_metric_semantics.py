from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.metric_semantics import (
    binary_dim_support,
    cyclist_visibility_proxy,
    false_dimming,
    flicker_change_rate,
    glare_risk_exposure,
    mask_iou,
    normalized_energy_consumption,
    over_masking_area,
    pedestrian_visibility_proxy,
    road_illumination_retention,
    summarize_actuation_latency_s,
    temporal_smoothness,
    vehicle_shadow_zone_violation,
)


class Block68MetricSemanticsTest(
    unittest.TestCase
):

    def test_01_binary_dim_support(
        self,
    ):

        value = np.asarray(
            [
                1.0,
                0.9,
                0.0,
            ]
        )

        np.testing.assert_array_equal(
            binary_dim_support(
                value
            ),
            np.asarray(
                [
                    False,
                    True,
                    True,
                ]
            ),
        )

    def test_02_mask_iou_empty(
        self,
    ):

        empty = np.zeros(
            4,
            dtype=bool,
        )

        self.assertEqual(
            mask_iou(
                empty,
                empty,
            ),
            1.0,
        )

    def test_03_vehicle_violation(
        self,
    ):

        dim = np.asarray(
            [
                True,
                False,
                False,
            ],
            dtype=bool,
        )

        oracle_vehicle = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            vehicle_shadow_zone_violation(
                dim,
                oracle_vehicle,
            ),
            0.5,
        )

    def test_04_vehicle_empty_is_na(
        self,
    ):

        dim = np.zeros(
            3,
            dtype=bool,
        )

        oracle_vehicle = np.zeros(
            3,
            dtype=bool,
        )

        self.assertTrue(
            math.isnan(
                vehicle_shadow_zone_violation(
                    dim,
                    oracle_vehicle,
                )
            )
        )

    def test_05_glare_reference_region(
        self,
    ):

        intensity = np.asarray(
            [
                0.2,
                0.4,
                1.0,
            ]
        )

        region = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertAlmostEqual(
            glare_risk_exposure(
                intensity,
                region,
            ),
            0.3,
            places=12,
        )

    def test_06_overmask_full_grid_denominator(
        self,
    ):

        dim = np.asarray(
            [
                True,
                True,
                False,
                False,
            ],
            dtype=bool,
        )

        oracle = np.asarray(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            over_masking_area(
                dim,
                oracle,
            ),
            0.25,
        )

    def test_07_road_retention(
        self,
    ):

        intensity = np.asarray(
            [
                1.0,
                0.5,
                0.0,
            ]
        )

        road = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            road_illumination_retention(
                intensity,
                road,
            ),
            0.75,
        )

    def test_08_pedestrian_visibility(
        self,
    ):

        intensity = np.asarray(
            [
                0.2,
                0.8,
                1.0,
            ]
        )

        region = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            pedestrian_visibility_proxy(
                intensity,
                region,
            ),
            0.5,
        )

    def test_09_cyclist_visibility(
        self,
    ):

        intensity = np.asarray(
            [
                0.4,
                0.6,
                1.0,
            ]
        )

        region = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            cyclist_visibility_proxy(
                intensity,
                region,
            ),
            0.5,
        )

    def test_10_false_dimming_support_ratio(
        self,
    ):

        intensity = np.asarray(
            [
                0.0,
                0.5,
                1.0,
                1.0,
            ]
        )

        oracle = np.asarray(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=bool,
        )

        # D = [1,1,0,0]
        # D \ O = one cell
        # => 1/2
        self.assertEqual(
            false_dimming(
                intensity,
                oracle,
            ),
            0.5,
        )

    def test_11_false_dimming_no_dim_is_zero(
        self,
    ):

        intensity = np.ones(
            4
        )

        oracle = np.zeros(
            4,
            dtype=bool,
        )

        self.assertEqual(
            false_dimming(
                intensity,
                oracle,
            ),
            0.0,
        )

    def test_12_temporal_change_rate(
        self,
    ):

        sequence = np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    0.2,
                    0.0,
                ],
            ]
        )

        # Mean grid absolute change = 0.1.
        # dt = 0.5 s.
        # change rate = 0.2 / s.
        self.assertAlmostEqual(
            temporal_smoothness(
                sequence,
                delta_t_s=0.5,
            ),
            0.2,
            places=12,
        )

    def test_13_flicker_exact_inequality(
        self,
    ):

        sequence = np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0e-12,
                ],
            ]
        )

        self.assertEqual(
            flicker_change_rate(
                sequence
            ),
            0.5,
        )

    def test_14_energy_full_grid_mean(
        self,
    ):

        sequence = np.asarray(
            [
                [
                    0.0,
                    1.0,
                ],
                [
                    1.0,
                    0.0,
                ],
            ]
        )

        self.assertEqual(
            normalized_energy_consumption(
                sequence
            ),
            0.5,
        )

    def test_15_latency_seconds(
        self,
    ):

        summary = (
            summarize_actuation_latency_s(
                [
                    0.001,
                    0.002,
                    0.003,
                    0.004,
                ]
            )
        )

        self.assertEqual(
            summary.sample_count,
            4,
        )

        self.assertAlmostEqual(
            summary.mean_s,
            0.0025,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()

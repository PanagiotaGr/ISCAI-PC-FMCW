from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.metric_semantics import (
    FLICKER_DELTA_THRESHOLD,
    cyclist_visibility_proxy,
    false_dimming,
    flicker_change_rate,
    glare_risk_exposure,
    mask_iou,
    normalized_energy_consumption,
    over_masking_area,
    pedestrian_visibility_proxy,
    road_illumination_retention,
    summarize_actuation_latency_ms,
    temporal_smoothness,
    vehicle_shadow_zone_violation,
)


class Block68MetricSemanticsTest(
    unittest.TestCase
):

    def test_01_mask_requires_boolean(
        self,
    ):
        with self.assertRaises(
            TypeError
        ):
            mask_iou(
                np.asarray(
                    [0, 1],
                    dtype=np.int64,
                ),
                np.asarray(
                    [False, True],
                    dtype=bool,
                ),
            )

    def test_02_intensity_bounds_enforced(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            glare_risk_exposure(
                np.asarray(
                    [0.0, 1.1]
                ),
                np.asarray(
                    [True, True],
                    dtype=bool,
                ),
            )

    def test_03_mask_iou_perfect(
        self,
    ):
        mask = np.asarray(
            [True, False, True],
            dtype=bool,
        )

        self.assertEqual(
            mask_iou(
                mask,
                mask,
            ),
            1.0,
        )

    def test_04_mask_iou_empty_is_one(
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

    def test_05_vehicle_violation(
        self,
    ):
        candidate = np.asarray(
            [True, False, False, False],
            dtype=bool,
        )

        vehicle = np.asarray(
            [True, True, False, False],
            dtype=bool,
        )

        self.assertEqual(
            vehicle_shadow_zone_violation(
                candidate,
                vehicle,
            ),
            0.5,
        )

    def test_06_glare_exposure(
        self,
    ):
        intensity = np.asarray(
            [0.1, 0.3, 1.0]
        )

        vehicle = np.asarray(
            [True, True, False],
            dtype=bool,
        )

        self.assertAlmostEqual(
            glare_risk_exposure(
                intensity,
                vehicle,
            ),
            0.2,
            places=12,
        )

    def test_07_overmasking(
        self,
    ):
        candidate = np.asarray(
            [True, True, False, False],
            dtype=bool,
        )

        oracle = np.asarray(
            [True, False, False, False],
            dtype=bool,
        )

        road = np.asarray(
            [True, True, True, True],
            dtype=bool,
        )

        self.assertEqual(
            over_masking_area(
                candidate,
                oracle,
                road,
            ),
            0.25,
        )

    def test_08_road_retention(
        self,
    ):
        intensity = np.asarray(
            [1.0, 0.5, 0.0]
        )

        road = np.asarray(
            [True, True, False],
            dtype=bool,
        )

        self.assertEqual(
            road_illumination_retention(
                intensity,
                road,
            ),
            0.75,
        )

    def test_09_pedestrian_visibility(
        self,
    ):
        intensity = np.asarray(
            [0.2, 0.8, 1.0]
        )

        region = np.asarray(
            [True, True, False],
            dtype=bool,
        )

        self.assertEqual(
            pedestrian_visibility_proxy(
                intensity,
                region,
            ),
            0.5,
        )

        absent = np.zeros(
            3,
            dtype=bool,
        )

        self.assertTrue(
            math.isnan(
                pedestrian_visibility_proxy(
                    intensity,
                    absent,
                )
            )
        )

    def test_10_cyclist_visibility(
        self,
    ):
        intensity = np.asarray(
            [0.4, 0.6, 1.0]
        )

        region = np.asarray(
            [True, True, False],
            dtype=bool,
        )

        self.assertEqual(
            cyclist_visibility_proxy(
                intensity,
                region,
            ),
            0.5,
        )

    def test_11_false_dimming(
        self,
    ):
        intensity = np.asarray(
            [0.0, 0.5, 1.0, 1.0]
        )

        oracle = np.asarray(
            [True, False, False, False],
            dtype=bool,
        )

        road = np.ones(
            4,
            dtype=bool,
        )

        # Outside oracle = cells 1,2,3.
        # dimming = 0.5,0,0 -> mean 1/6.
        self.assertAlmostEqual(
            false_dimming(
                intensity,
                oracle,
                road,
            ),
            1.0 / 6.0,
            places=12,
        )

    def test_12_temporal_smoothness_stable(
        self,
    ):
        sequence = np.ones(
            (
                3,
                2,
                2,
            ),
            dtype=np.float64,
        )

        self.assertEqual(
            temporal_smoothness(
                sequence
            ),
            1.0,
        )

    def test_13_flicker_threshold(
        self,
    ):
        self.assertEqual(
            FLICKER_DELTA_THRESHOLD,
            0.10,
        )

        sequence = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.09],
            ],
            dtype=np.float64,
        )

        self.assertEqual(
            flicker_change_rate(
                sequence
            ),
            0.5,
        )

    def test_14_energy_proxy(
        self,
    ):
        sequence = np.asarray(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.assertEqual(
            normalized_energy_consumption(
                sequence
            ),
            0.5,
        )

    def test_15_latency_summary(
        self,
    ):
        summary = (
            summarize_actuation_latency_ms(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            )
        )

        self.assertEqual(
            summary.sample_count,
            4,
        )

        self.assertEqual(
            summary.mean_ms,
            2.5,
        )

        self.assertEqual(
            summary.median_ms,
            2.5,
        )

        self.assertGreaterEqual(
            summary.p95_ms,
            3.0,
        )

        self.assertEqual(
            summary.max_ms,
            4.0,
        )


if __name__ == "__main__":
    unittest.main()

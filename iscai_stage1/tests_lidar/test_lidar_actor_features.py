import math
import unittest

from iscai_stage1.lidar.actor_features import (
    DecodedActorLidarPoint,
    summarize_actor_lidar_points,
)


class TestActorLidarFeatures(unittest.TestCase):
    def test_two_point_statistics(self):
        points = (
            DecodedActorLidarPoint(
                point_H0_m=(1.0, 2.0, 3.0),
                range_m=10.0,
                intensity=2.0,
                elongation=0.5,
            ),
            DecodedActorLidarPoint(
                point_H0_m=(3.0, 4.0, 5.0),
                range_m=14.0,
                intensity=4.0,
                elongation=1.5,
            ),
        )

        result = summarize_actor_lidar_points(points)

        self.assertEqual(result.point_count, 2)
        self.assertTrue(result.has_points)

        self.assertEqual(result.range_stats.mean, 12.0)
        self.assertEqual(result.range_stats.median, 12.0)
        self.assertEqual(result.range_stats.variance, 4.0)

        self.assertEqual(result.intensity_stats.mean, 3.0)
        self.assertEqual(result.intensity_stats.variance, 1.0)

        self.assertEqual(result.elongation_stats.mean, 1.0)
        self.assertEqual(
            result.elongation_stats.variance,
            0.25,
        )

        self.assertEqual(
            result.spatial_std_H0_m,
            (1.0, 1.0, 1.0),
        )

    def test_single_point_has_zero_spatial_spread(self):
        result = summarize_actor_lidar_points(
            (
                DecodedActorLidarPoint(
                    point_H0_m=(1.0, 2.0, 3.0),
                    range_m=5.0,
                    intensity=7.0,
                    elongation=0.2,
                ),
            )
        )

        self.assertEqual(
            result.spatial_std_H0_m,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(result.range_stats.variance, 0.0)

    def test_zero_points_are_retained_explicitly(self):
        result = summarize_actor_lidar_points(())

        self.assertEqual(result.point_count, 0)
        self.assertFalse(result.has_points)
        self.assertIsNone(result.range_stats)
        self.assertIsNone(result.intensity_stats)
        self.assertIsNone(result.elongation_stats)
        self.assertIsNone(result.spatial_std_H0_m)

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_actor_lidar_points(
                (
                    DecodedActorLidarPoint(
                        point_H0_m=(math.nan, 0.0, 0.0),
                        range_m=1.0,
                        intensity=1.0,
                        elongation=1.0,
                    ),
                )
            )

    def test_negative_range_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_actor_lidar_points(
                (
                    DecodedActorLidarPoint(
                        point_H0_m=(0.0, 0.0, 0.0),
                        range_m=-1.0,
                        intensity=1.0,
                        elongation=1.0,
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
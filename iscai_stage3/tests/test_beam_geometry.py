import math
import unittest

from iscai_stage3.geometry.beam import (
    BeamOrigin,
    point_H0_to_beam,
)


class TestBeamGeometry(unittest.TestCase):

    def test_forward_target(self):

        origin = BeamOrigin(
            position_H0_m=(0.0,0.0,0.0)
        )

        result = point_H0_to_beam(
            (10.0,0.0,0.0),
            origin,
        )

        self.assertEqual(
            result.range_m,
            10.0,
        )

        self.assertEqual(
            result.azimuth_rad,
            0.0,
        )

        self.assertEqual(
            result.elevation_rad,
            0.0,
        )


    def test_left_target(self):

        origin = BeamOrigin(
            position_H0_m=(0.0,0.0,0.0)
        )

        result = point_H0_to_beam(
            (10.0,10.0,0.0),
            origin,
        )

        self.assertAlmostEqual(
            result.azimuth_rad,
            math.pi/4,
        )


    def test_vertical_target(self):

        origin = BeamOrigin(
            position_H0_m=(0.0,0.0,0.0)
        )

        result = point_H0_to_beam(
            (10.0,0.0,10.0),
            origin,
        )

        self.assertAlmostEqual(
            result.elevation_rad,
            math.pi/4,
        )


    def test_zero_range_rejected(self):

        origin = BeamOrigin(
            position_H0_m=(0.0,0.0,0.0)
        )

        with self.assertRaises(ValueError):
            point_H0_to_beam(
                (0.0,0.0,0.0),
                origin,
            )


if __name__ == "__main__":
    unittest.main()

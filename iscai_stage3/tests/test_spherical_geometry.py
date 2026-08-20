import unittest
from math import sqrt

from iscai_stage3.geometry.spherical import (
    spherical_to_cartesian,
    spherical_position_jacobian,
)


class TestSphericalGeometry(unittest.TestCase):

    def test_forward_axis(self):

        xyz = spherical_to_cartesian(
            10.0,
            0.0,
            0.0,
        )

        self.assertEqual(
            xyz,
            (10.0,0.0,0.0),
        )


    def test_jacobian_shape(self):

        J = spherical_position_jacobian(
            20.0,
            0.1,
            0.05,
        )

        self.assertEqual(
            len(J),
            3,
        )

        self.assertEqual(
            len(J[0]),
            3,
        )


    def test_small_angle_consistency(self):

        xyz = spherical_to_cartesian(
            1.0,
            0.0,
            0.0,
        )

        self.assertAlmostEqual(
            sqrt(sum(x*x for x in xyz)),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

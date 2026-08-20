import unittest

from iscai_stage3.uncertainty.beam_covariance import (
    beam_jacobian,
    propagate_cartesian_covariance_to_beam,
)


class TestBeamCovariance(unittest.TestCase):

    def test_forward_jacobian(self):

        J = beam_jacobian(
            (10.0,0.0,0.0)
        )

        self.assertAlmostEqual(
            J[0][0],
            1.0,
        )

        self.assertAlmostEqual(
            J[1][1],
            0.1,
        )


    def test_identity_covariance(self):

        result = (
            propagate_cartesian_covariance_to_beam(
                position_xyz=(
                    10.0,
                    0.0,
                    0.0,
                ),
                covariance_xyz=(
                    (1.0,0.0,0.0),
                    (0.0,1.0,0.0),
                    (0.0,0.0,1.0),
                ),
            )
        )

        self.assertGreater(
            result.range_variance,
            0.0,
        )

        self.assertGreater(
            result.azimuth_variance,
            0.0,
        )


    def test_symmetry(self):

        result = (
            propagate_cartesian_covariance_to_beam(
                position_xyz=(
                    5.0,
                    2.0,
                    1.0,
                ),
                covariance_xyz=(
                    (1.0,0.1,0.0),
                    (0.1,1.0,0.0),
                    (0.0,0.0,1.0),
                ),
            )
        )

        M = result.covariance_matrix

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    M[i][j],
                    M[j][i],
                )

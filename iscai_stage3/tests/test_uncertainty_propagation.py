import unittest

from iscai_stage3.uncertainty.propagation import (
    propagate_position_covariance,
    validate_covariance,
)


class TestPropagation(unittest.TestCase):

    def test_identity_propagation(self):

        J = (
            (1.0,0.0,0.0),
            (0.0,1.0,0.0),
            (0.0,0.0,1.0),
        )


        Sigma = (
            (1.0,0.0,0.0),
            (0.0,2.0,0.0),
            (0.0,0.0,3.0),
        )


        out = propagate_position_covariance(
            J,
            Sigma,
        )

        self.assertEqual(
            out,
            Sigma,
        )


    def test_symmetry_required(self):

        with self.assertRaises(ValueError):

            validate_covariance(
                (
                    (1.0,2.0),
                    (0.0,1.0),
                )
            )


    def test_negative_variance_rejected(self):

        with self.assertRaises(ValueError):

            validate_covariance(
                (
                    (-1.0,0.0),
                    (0.0,1.0),
                )
            )


if __name__ == "__main__":
    unittest.main()

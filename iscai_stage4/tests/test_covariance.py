import unittest

from iscai_stage4.uncertainty import (
    validate_covariance,
)


class TestCovariance(unittest.TestCase):

    def test_valid_covariance(self):

        cov = (
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.1),
        )

        validate_covariance(
            cov
        )


    def test_negative_variance(self):

        cov = (
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

        with self.assertRaises(
            ValueError
        ):
            validate_covariance(
                cov
            )


if __name__ == "__main__":
    unittest.main()

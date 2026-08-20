import unittest

from iscai_stage3.probability.beam_confidence import (
    gaussian_confidence,
    beam_confidence_from_covariance,
)



class TestBeamConfidence(unittest.TestCase):


    def test_center_probability(self):

        p = gaussian_confidence(
            error_rad=0.0,
            sigma_rad=0.1,
        )

        self.assertEqual(
            p,
            1.0,
        )


    def test_error_reduces_confidence(self):

        center = gaussian_confidence(
            error_rad=0.0,
            sigma_rad=0.1,
        )

        off = gaussian_confidence(
            error_rad=0.1,
            sigma_rad=0.1,
        )

        self.assertLess(
            off,
            center,
        )


    def test_covariance_confidence(self):

        result = (
            beam_confidence_from_covariance(
                angular_error_rad=0.0,
                azimuth_variance=0.01,
                elevation_variance=0.01,
            )
        )

        self.assertEqual(
            result.confidence,
            1.0,
        )


    def test_invalid_sigma(self):

        with self.assertRaises(
            ValueError
        ):
            gaussian_confidence(
                error_rad=0.1,
                sigma_rad=0.0,
            )

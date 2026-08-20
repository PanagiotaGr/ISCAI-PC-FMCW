import unittest

from iscai_stage3.beam.probability import (
    beam_cross_track_error,
    gaussian_beam_confidence,
)



class TestBeamProbability(unittest.TestCase):


    def test_center_has_zero_error(self):

        error = beam_cross_track_error(
            (10.0,0.0,0.0),
            (0.0,0.0,0.0),
            (1.0,0.0,0.0),
        )

        self.assertEqual(
            error,
            0.0,
        )


    def test_side_error(self):

        error = beam_cross_track_error(
            (10.0,2.0,0.0),
            (0.0,0.0,0.0),
            (1.0,0.0,0.0),
        )

        self.assertEqual(
            error,
            2.0,
        )


    def test_confidence_center(self):

        score = gaussian_beam_confidence(
            0.0,
            1.0,
        )

        self.assertEqual(
            score,
            1.0,
        )


    def test_invalid_sigma(self):

        with self.assertRaises(ValueError):

            gaussian_beam_confidence(
                1.0,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()

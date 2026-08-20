import unittest


from iscai_stage4.predictor.calibration import (
    evaluate_calibration,
)



class TestCalibration(
    unittest.TestCase
):


    def test_metrics(self):

        metrics = evaluate_calibration(

            errors=(

                0.1,
                0.2,
                0.3,
            ),

            variance=0.25,

            predicted=(

                0.8,
                0.7,
                0.9,
            ),

            observed=(

                1.0,
                1.0,
                1.0,
            ),
        )


        self.assertGreaterEqual(
            metrics.nll,
            0.0,
        )


        self.assertGreaterEqual(
            metrics.brier,
            0.0,
        )


        self.assertGreaterEqual(
            metrics.empirical_coverage,
            0.0,
        )


        self.assertLessEqual(
            metrics.empirical_coverage,
            1.0,
        )


if __name__ == "__main__":

    unittest.main()

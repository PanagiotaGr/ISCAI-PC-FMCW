import unittest


from iscai_stage3.baselines.imm_evaluator import (
    IMMModeMeasurement,
    evaluate_mode,
)


class TestIMMEvaluator(unittest.TestCase):


    def test_best_prediction_gets_best_score(self):

        covariance = (
            (1.0,0.0,0.0,0.0),
            (0.0,1.0,0.0,0.0),
            (0.0,0.0,1.0,0.0),
            (0.0,0.0,0.0,1.0),
        )


        good = evaluate_mode(
            mode=IMMModeMeasurement(
                mode_name="CV",
                predicted_measurement=(
                    10.0,
                    1.0,
                    0.1,
                    0.0,
                ),
            ),
            measured=(
                10.0,
                1.0,
                0.1,
                0.0,
            ),
            covariance=covariance,
        )


        bad = evaluate_mode(
            mode=IMMModeMeasurement(
                mode_name="CA",
                predicted_measurement=(
                    20.0,
                    5.0,
                    0.5,
                    0.5,
                ),
            ),
            measured=(
                10.0,
                1.0,
                0.1,
                0.0,
            ),
            covariance=covariance,
        )


        self.assertGreater(
            good.log_likelihood,
            bad.log_likelihood,
        )


    def test_mode_name_preserved(self):

        result = evaluate_mode(
            mode=IMMModeMeasurement(
                mode_name="CTRV",
                predicted_measurement=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
            ),
            measured=(
                1.0,
                1.0,
                1.0,
                1.0,
            ),
            covariance=(
                (1.0,0,0,0),
                (0,1.0,0,0),
                (0,0,1.0,0),
                (0,0,0,1.0),
            ),
        )

        self.assertEqual(
            result.mode_name,
            "CTRV",
        )


if __name__ == "__main__":
    unittest.main()

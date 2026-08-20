import unittest

from iscai_stage3.baselines.imm_likelihood import (
    gaussian_log_likelihood,
    normalized_likelihoods_from_logs,
)


class TestIMMLikelihood(unittest.TestCase):

    def test_zero_innovation_best(self):

        covariance = (
            (1.0, 0.0),
            (0.0, 1.0),
        )

        zero = gaussian_log_likelihood(
            innovation=(0.0, 0.0),
            innovation_covariance=covariance,
        )

        nonzero = gaussian_log_likelihood(
            innovation=(1.0, 0.0),
            innovation_covariance=covariance,
        )

        self.assertGreater(
            zero,
            nonzero,
        )


    def test_larger_residual_lower_likelihood(self):

        covariance = (
            (1.0, 0.0),
            (0.0, 1.0),
        )

        small = gaussian_log_likelihood(
            innovation=(0.1, 0.1),
            innovation_covariance=covariance,
        )

        large = gaussian_log_likelihood(
            innovation=(3.0, 3.0),
            innovation_covariance=covariance,
        )

        self.assertGreater(
            small,
            large,
        )


    def test_normalized_likelihoods_sum_to_one(self):

        values = normalized_likelihoods_from_logs(
            (
                -1.0,
                -2.0,
                -3.0,
            )
        )

        self.assertAlmostEqual(
            sum(values),
            1.0,
        )


    def test_best_log_likelihood_gets_best_weight(self):

        cv, ca, ctrv = (
            normalized_likelihoods_from_logs(
                (
                    -5.0,
                    -1.0,
                    -10.0,
                )
            )
        )

        self.assertGreater(
            ca,
            cv,
        )

        self.assertGreater(
            ca,
            ctrv,
        )


    def test_asymmetric_covariance_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            gaussian_log_likelihood(
                innovation=(0.0, 0.0),
                innovation_covariance=(
                    (1.0, 0.5),
                    (0.0, 1.0),
                ),
            )


    def test_singular_covariance_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            gaussian_log_likelihood(
                innovation=(0.0, 0.0),
                innovation_covariance=(
                    (1.0, 0.0),
                    (0.0, 0.0),
                ),
            )


if __name__ == "__main__":
    unittest.main()

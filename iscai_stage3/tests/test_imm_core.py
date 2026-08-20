import unittest

from iscai_stage3.baselines.imm_core import (
    IMMProbabilities,
    imm_mixing_probabilities,
    update_imm_probabilities,
)


TRANSITION = (
    (0.90, 0.05, 0.05),
    (0.05, 0.90, 0.05),
    (0.05, 0.05, 0.90),
)


class TestIMMCore(unittest.TestCase):

    def test_probabilities_sum_to_one(self):

        probabilities = IMMProbabilities(
            cv=0.6,
            ca=0.3,
            ctrv=0.1,
        )

        self.assertAlmostEqual(
            sum(
                probabilities.as_tuple()
            ),
            1.0,
        )


    def test_mixing_columns_sum_to_one(self):

        result = imm_mixing_probabilities(
            probabilities=(
                IMMProbabilities(
                    cv=0.6,
                    ca=0.3,
                    ctrv=0.1,
                )
            ),
            transition_matrix=(
                TRANSITION
            ),
        )

        matrix = (
            result.conditional_mixing
        )

        for destination in range(3):

            self.assertAlmostEqual(
                sum(
                    matrix[source][
                        destination
                    ]
                    for source
                    in range(3)
                ),
                1.0,
            )


    def test_equal_likelihood_preserves_prediction(self):

        result = imm_mixing_probabilities(
            probabilities=(
                IMMProbabilities(
                    cv=0.6,
                    ca=0.3,
                    ctrv=0.1,
                )
            ),
            transition_matrix=(
                TRANSITION
            ),
        )

        updated = (
            update_imm_probabilities(
                predicted_probabilities=(
                    result
                    .predicted_probabilities
                ),
                measurement_likelihoods=(
                    1.0,
                    1.0,
                    1.0,
                ),
            )
        )

        for actual, expected in zip(
            updated.as_tuple(),
            (
                result
                .predicted_probabilities
                .as_tuple()
            ),
        ):
            self.assertAlmostEqual(
                actual,
                expected,
            )


    def test_high_ca_likelihood_increases_ca(self):

        predicted = IMMProbabilities(
            cv=0.4,
            ca=0.3,
            ctrv=0.3,
        )

        updated = update_imm_probabilities(
            predicted_probabilities=predicted,
            measurement_likelihoods=(
                0.1,
                1.0,
                0.1,
            ),
        )

        self.assertGreater(
            updated.ca,
            predicted.ca,
        )

        self.assertGreater(
            updated.ca,
            updated.cv,
        )

        self.assertGreater(
            updated.ca,
            updated.ctrv,
        )


    def test_invalid_probability_sum_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            IMMProbabilities(
                cv=0.5,
                ca=0.5,
                ctrv=0.5,
            )


    def test_invalid_transition_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            imm_mixing_probabilities(
                probabilities=(
                    IMMProbabilities(
                        cv=1.0,
                        ca=0.0,
                        ctrv=0.0,
                    )
                ),
                transition_matrix=(
                    (0.8, 0.1, 0.0),
                    (0.1, 0.8, 0.1),
                    (0.1, 0.1, 0.8),
                ),
            )


if __name__ == "__main__":
    unittest.main()

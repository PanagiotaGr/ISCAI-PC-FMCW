from __future__ import annotations

import unittest

from iscai_stage5.formal_acceptance import (
    FORMAL_EMPIRICAL_ALPHA,
    NOMINAL_COVERAGE_Q,
    binomial_lower_tail,
    empirical_coverage_decision,
    formal_acceptance_policy_dict,
    minimum_accepted_hits,
)


class TestBlock58FormalAcceptance(
    unittest.TestCase
):
    def test_01_binomial_n2_h0(
        self,
    ):
        self.assertAlmostEqual(
            binomial_lower_tail(
                0,
                2,
                0.5,
            ),
            0.25,
            places=14,
        )

    def test_02_binomial_n2_h1(
        self,
    ):
        self.assertAlmostEqual(
            binomial_lower_tail(
                1,
                2,
                0.5,
            ),
            0.75,
            places=14,
        )

    def test_03_all_hits_tail_is_one(
        self,
    ):
        self.assertEqual(
            binomial_lower_tail(
                20,
                20,
                0.95,
            ),
            1.0,
        )

    def test_04_tail_is_monotonic(
        self,
    ):
        previous = 0.0

        for hits in range(
            0,
            101,
        ):
            current = (
                binomial_lower_tail(
                    hits,
                    100,
                    0.95,
                )
            )

            self.assertGreaterEqual(
                current + 1e-15,
                previous,
            )

            previous = current

    def test_05_clear_shortfall_fails(
        self,
    ):
        decision = (
            empirical_coverage_decision(
                hits=90,
                trials=100,
                requested_q=0.95,
            )
        )

        self.assertFalse(
            decision.passed
        )

        self.assertLess(
            decision.lower_tail_p_value,
            FORMAL_EMPIRICAL_ALPHA,
        )

    def test_06_all_hits_pass(
        self,
    ):
        decision = (
            empirical_coverage_decision(
                hits=100,
                trials=100,
                requested_q=0.95,
            )
        )

        self.assertTrue(
            decision.passed
        )

    def test_07_empirical_coverage_exact(
        self,
    ):
        decision = (
            empirical_coverage_decision(
                hits=19,
                trials=20,
                requested_q=0.95,
            )
        )

        self.assertAlmostEqual(
            decision.empirical_coverage,
            0.95,
            places=15,
        )

    def test_08_minimum_hit_threshold(
        self,
    ):
        minimum = (
            minimum_accepted_hits(
                100,
                0.95,
                0.05,
            )
        )

        self.assertEqual(
            minimum,
            91,
        )

        self.assertLess(
            binomial_lower_tail(
                minimum - 1,
                100,
                0.95,
            ),
            0.05,
        )

        self.assertGreaterEqual(
            binomial_lower_tail(
                minimum,
                100,
                0.95,
            ),
            0.05,
        )

    def test_09_zero_trials_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            empirical_coverage_decision(
                hits=0,
                trials=0,
                requested_q=0.95,
            )

    def test_10_invalid_hits_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            empirical_coverage_decision(
                hits=11,
                trials=10,
                requested_q=0.95,
            )

    def test_11_invalid_q_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            empirical_coverage_decision(
                hits=10,
                trials=10,
                requested_q=1.0,
            )

    def test_12_policy_is_prefrozen_nominal_95(
        self,
    ):
        policy = (
            formal_acceptance_policy_dict()
        )

        self.assertEqual(
            policy["status"],
            "FROZEN_PRE_FORMAL",
        )

        self.assertEqual(
            policy[
                "coverage_targets"
            ][
                "nominal_primary_q"
            ],
            NOMINAL_COVERAGE_Q,
        )

        self.assertFalse(
            policy[
                "formal_leakage_guards"
            ][
                "formal_results_used_to_choose_threshold"
            ]
        )


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest

import numpy as np

from iscai_stage5.adaptive_topk import (
    ADAPTIVE_SELECTION_RULE,
    ADAPTIVE_TIE_BREAK,
    FROZEN_COVERAGE_TARGETS,
    FROZEN_NOMINAL_COVERAGE,
    KMAX_POLICY_STATUS,
    OUTSIDE_SUPPORT_POLICY,
    adaptive_topk_probability_mass,
    all_frozen_coverage_targets,
    descending_probability_ranking,
    nominal_adaptive_topk,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)


class AdaptiveTopKTestBase(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

    def probability(
        self,
        masses,
        *,
        outside=0.0,
    ):
        masses = np.asarray(
            masses,
            dtype=np.float64,
        )

        return BeamProbabilityMass(
            masses=tuple(
                float(
                    value
                )
                for value in masses
            ),

            inside_support_mass=float(
                np.sum(
                    masses
                )
            ),

            outside_support_mass=float(
                outside
            ),

            sample_count=2048,
        )


class TestBlock55FrozenContract(
    AdaptiveTopKTestBase
):

    def test_01_coverage_targets(self):
        self.assertEqual(
            FROZEN_COVERAGE_TARGETS,
            (
                0.90,
                0.95,
                0.975,
                0.99,
            ),
        )

    def test_02_nominal_coverage(self):
        self.assertEqual(
            FROZEN_NOMINAL_COVERAGE,
            0.95,
        )

    def test_03_selection_rule_explicit(self):
        self.assertIn(
            "smallest",
            ADAPTIVE_SELECTION_RULE,
        )

        self.assertEqual(
            ADAPTIVE_TIE_BREAK,
            "lower_beam_index",
        )

    def test_04_outside_support_not_renormalized(self):
        self.assertIn(
            "cannot_be_claimed_as_covered",
            OUTSIDE_SUPPORT_POLICY,
        )

    def test_05_kmax_not_yet_numeric_frozen(self):
        self.assertEqual(
            KMAX_POLICY_STATUS,
            (
                "SUPPORTED_NOT_NUMERICALLY_"
                "FROZEN_IN_PART1"
            ),
        )


class TestBlock55AdaptiveSelection(
    AdaptiveTopKTestBase
):

    def test_06_minimum_prefix_reaches_95_percent(self):
        masses = np.zeros(
            16
        )

        masses[
            8
        ] = 0.60

        masses[
            7
        ] = 0.25

        masses[
            9
        ] = 0.10

        masses[
            6
        ] = 0.05

        result = (
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,
            )
        )

        self.assertEqual(
            result.beam_indices,
            (
                8,
                7,
                9,
            ),
        )

        self.assertEqual(
            result.k,
            3,
        )

        self.assertTrue(
            result.achieved_requested_coverage
        )

        self.assertGreaterEqual(
            result.selected_in_support_mass,
            0.95,
        )

    def test_07_stops_at_smallest_prefix(self):
        masses = np.zeros(
            16
        )

        masses[
            2
        ] = 0.96

        masses[
            3
        ] = 0.04

        result = nominal_adaptive_topk(
            probability=(
                self.probability(
                    masses
                )
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            result.k,
            1,
        )

        self.assertEqual(
            result.beam_indices,
            (
                2,
            ),
        )

    def test_08_tie_break_lower_index(self):
        masses = np.zeros(
            16
        )

        masses[
            5
        ] = 0.50

        masses[
            4
        ] = 0.50

        ranking = (
            descending_probability_ranking(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            ranking[
                :2
            ],
            (
                4,
                5,
            ),
        )

    def test_09_all_frozen_targets_present(self):
        masses = np.zeros(
            16
        )

        masses[
            8
        ] = 1.0

        results = (
            all_frozen_coverage_targets(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            tuple(
                results.keys()
            ),
            FROZEN_COVERAGE_TARGETS,
        )

        for result in (
            results.values()
        ):

            self.assertEqual(
                result.k,
                1,
            )

            self.assertTrue(
                result
                .achieved_requested_coverage
            )

    def test_10_monotonic_K_across_coverage_targets(self):
        masses = np.asarray(
            [
                0.30,
                0.25,
                0.20,
                0.15,
                0.05,
                0.025,
                0.015,
                0.01,
            ]
            +
            [
                0.0
            ]
            *
            8
        )

        results = (
            all_frozen_coverage_targets(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        ks = [
            results[
                q
            ].k
            for q in (
                FROZEN_COVERAGE_TARGETS
            )
        ]

        self.assertEqual(
            ks,
            sorted(
                ks
            ),
        )


class TestBlock55OutsideSupport(
    AdaptiveTopKTestBase
):

    def test_11_unattainable_if_inside_mass_below_q(self):
        masses = np.zeros(
            16
        )

        masses[
            8
        ] = 0.60

        masses[
            7
        ] = 0.30

        result = (
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses,
                        outside=0.10,
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,
            )
        )

        self.assertFalse(
            result
            .achieved_requested_coverage
        )

        self.assertTrue(
            result
            .unattainable_due_to_outside_support
        )

        self.assertAlmostEqual(
            result.inside_support_mass,
            0.90,
        )

        self.assertAlmostEqual(
            result.outside_support_mass,
            0.10,
        )

    def test_12_outside_mass_is_not_renormalized(self):
        masses = np.zeros(
            16
        )

        masses[
            8
        ] = 0.80

        result = (
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses,
                        outside=0.20,
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,
            )
        )

        self.assertAlmostEqual(
            result
            .selected_in_support_mass,
            0.80,
        )

        self.assertFalse(
            result
            .achieved_requested_coverage
        )


class TestBlock55Kmax(
    AdaptiveTopKTestBase
):

    def test_13_kmax_can_limit_selection(self):
        masses = np.ones(
            16
        ) / 16.0

        result = (
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,

                kmax=5,
            )
        )

        self.assertEqual(
            result.k,
            5,
        )

        self.assertTrue(
            result.kmax_limited
        )

        self.assertFalse(
            result.achieved_requested_coverage
        )

    def test_14_kmax_none_allows_full_codebook(self):
        masses = np.ones(
            16
        ) / 16.0

        result = (
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,

                kmax=None,
            )
        )

        self.assertEqual(
            result.k,
            16,
        )

        self.assertTrue(
            result.achieved_requested_coverage
        )

    def test_15_invalid_kmax_rejected(self):
        masses = np.ones(
            16
        ) / 16.0

        with self.assertRaises(
            ValueError
        ):
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,

                kmax=0,
            )


class TestBlock55ValidationAndLeakage(
    AdaptiveTopKTestBase
):

    def test_16_invalid_coverage_rejected(self):
        masses = np.ones(
            16
        ) / 16.0

        with self.assertRaises(
            ValueError
        ):
            adaptive_topk_probability_mass(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=1.1,
            )

    def test_17_probability_accounting_mismatch_rejected(self):
        masses = np.ones(
            16
        ) / 16.0

        probability = BeamProbabilityMass(
            masses=tuple(
                masses
            ),

            inside_support_mass=0.9,

            outside_support_mass=0.0,

            sample_count=2048,
        )

        with self.assertRaises(
            ValueError
        ):
            adaptive_topk_probability_mass(
                probability=(
                    probability
                ),

                codebook=(
                    self.codebook
                ),

                requested_coverage=0.95,
            )

    def test_18_controller_API_has_no_truth_or_oracle(self):
        signature = str(
            inspect.signature(
                adaptive_topk_probability_mass
            )
        ).lower()

        for token in (
            "future",
            "ground_truth",
            "realized",
            "oracle",
            "tracks_to_predict",
            "objects_of_interest",
        ):

            self.assertNotIn(
                token,
                signature,
            )

    def test_19_nominal_API_has_no_truth_or_oracle(self):
        signature = str(
            inspect.signature(
                nominal_adaptive_topk
            )
        ).lower()

        for token in (
            "future",
            "ground_truth",
            "realized",
            "oracle",
            "tracks_to_predict",
            "objects_of_interest",
        ):

            self.assertNotIn(
                token,
                signature,
            )

    def test_20_exact_repeat(self):
        masses = np.asarray(
            [
                0.40,
                0.25,
                0.15,
                0.10,
                0.05,
                0.03,
                0.02,
            ]
            +
            [
                0.0
            ]
            *
            9
        )

        probability = (
            self.probability(
                masses
            )
        )

        first = (
            nominal_adaptive_topk(
                probability=(
                    probability
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        second = (
            nominal_adaptive_topk(
                probability=(
                    probability
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            first,
            second,
        )


if __name__ == "__main__":
    unittest.main()

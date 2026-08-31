import inspect
import unittest

import numpy as np

from iscai_stage5.adaptive_topk import (
    FROZEN_COVERAGE_TARGETS,
)

from iscai_stage5.adaptive_topk_temporal import (
    COARSER_CODEBOOK,
    FROZEN_MC_SAMPLE_COUNT,
    FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE,
    HYSTERESIS_RULE,
    KMAX_RULE,
    LOSS_OF_LOCK_RECOVERY,
    LOSS_OF_LOCK_RULE,
    NEIGHBOR_SWEEP_RULE,
    SWITCH_PENALTY_RULE,
    WIDENED_FALLBACK_RULE,
    adaptive_temporal_step,
    all_coverage_temporal_decisions,
    frozen_kmax_for_codebook,
    immediate_neighbor_indices,
    initial_adaptive_temporal_state,
    reset_adaptive_temporal_state,
    widened_fallback_for_primary,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)


class Block55Part2Base(
    unittest.TestCase
):

    def probability(
        self,
        beam_count,
        entries,
        *,
        outside=0.0,
    ):
        masses = np.zeros(
            beam_count,
            dtype=np.float64,
        )

        for index, value in entries:
            masses[
                index
            ] = value

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


class TestBlock55Part2FrozenParameters(
    Block55Part2Base
):

    def test_01_mc_resolution_tolerance(self):
        self.assertEqual(
            FROZEN_MC_SAMPLE_COUNT,
            2048,
        )

        self.assertEqual(
            FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE,
            1.0 / 2048.0,
        )

    def test_02_kmax_rule_is_full_codebook(self):
        self.assertEqual(
            KMAX_RULE,
            "full_codebook_size",
        )

        for count in (
            16,
            32,
            64,
        ):
            codebook = (
                build_frozen_directional_codebook(
                    count
                )
            )

            self.assertEqual(
                frozen_kmax_for_codebook(
                    codebook
                ),
                count,
            )

    def test_03_coarser_codebook_chain(self):
        self.assertEqual(
            COARSER_CODEBOOK,
            {
                64:
                    32,

                32:
                    16,

                16:
                    None,
            },
        )

    def test_04_temporal_rules_are_explicit(self):
        self.assertIn(
            "retain_previous_primary",
            HYSTERESIS_RULE,
        )

        self.assertIn(
            "unit_switch_event",
            SWITCH_PENALTY_RULE,
        )

        self.assertIn(
            "immediate_codebook_graph_neighbors",
            NEIGHBOR_SWEEP_RULE,
        )

        self.assertIn(
            "next_coarser_frozen_codebook",
            WIDENED_FALLBACK_RULE,
        )

        self.assertIn(
            "requested_coverage_not_achievable",
            LOSS_OF_LOCK_RULE,
        )

        self.assertIn(
            "exhaustive_current_codebook",
            LOSS_OF_LOCK_RECOVERY,
        )


class TestBlock55Part2Hysteresis(
    Block55Part2Base
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

    def test_05_first_step_uses_highest_probability_primary(self):
        probability = self.probability(
            16,
            (
                (
                    8,
                    0.60,
                ),
                (
                    7,
                    0.25,
                ),
                (
                    9,
                    0.10,
                ),
                (
                    6,
                    0.05,
                ),
            ),
        )

        decision, state = (
            adaptive_temporal_step(
                state=(
                    initial_adaptive_temporal_state()
                ),

                probability=(
                    probability
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            decision.primary_beam_index,
            8,
        )

        self.assertFalse(
            decision
            .previous_primary_retained
        )

        self.assertEqual(
            state.previous_primary_beam_index,
            8,
        )

    def test_06_previous_primary_retained_if_still_in_set(self):
        first_probability = self.probability(
            16,
            (
                (
                    8,
                    0.60,
                ),
                (
                    7,
                    0.35,
                ),
                (
                    9,
                    0.05,
                ),
            ),
        )

        _, state = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                first_probability
            ),

            codebook=(
                self.codebook
            ),
        )

        second_probability = self.probability(
            16,
            (
                (
                    7,
                    0.55,
                ),
                (
                    8,
                    0.40,
                ),
                (
                    9,
                    0.05,
                ),
            ),
        )

        decision, state = (
            adaptive_temporal_step(
                state=(
                    state
                ),

                probability=(
                    second_probability
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            decision.primary_beam_index,
            8,
        )

        self.assertTrue(
            decision
            .previous_primary_retained
        )

        self.assertFalse(
            decision
            .primary_switched
        )

        self.assertEqual(
            decision.switch_penalty_units,
            0,
        )

    def test_07_selected_mass_covering_set_is_not_changed_by_hysteresis(self):
        first_probability = self.probability(
            16,
            (
                (
                    8,
                    0.60,
                ),
                (
                    7,
                    0.35,
                ),
                (
                    9,
                    0.05,
                ),
            ),
        )

        _, state = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                first_probability
            ),

            codebook=(
                self.codebook
            ),
        )

        second_probability = self.probability(
            16,
            (
                (
                    7,
                    0.55,
                ),
                (
                    8,
                    0.40,
                ),
                (
                    9,
                    0.05,
                ),
            ),
        )

        decision, _ = adaptive_temporal_step(
            state=(
                state
            ),

            probability=(
                second_probability
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            set(
                decision
                .ordered_beam_indices
            ),
            set(
                decision
                .adaptive_selection
                .beam_indices
            ),
        )

    def test_08_switch_occurs_if_previous_leaves_set(self):
        first_probability = self.probability(
            16,
            (
                (
                    8,
                    0.95,
                ),
                (
                    7,
                    0.05,
                ),
            ),
        )

        _, state = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                first_probability
            ),

            codebook=(
                self.codebook
            ),
        )

        second_probability = self.probability(
            16,
            (
                (
                    3,
                    0.95,
                ),
                (
                    4,
                    0.05,
                ),
            ),
        )

        decision, state = (
            adaptive_temporal_step(
                state=(
                    state
                ),

                probability=(
                    second_probability
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            decision.primary_beam_index,
            3,
        )

        self.assertTrue(
            decision.primary_switched
        )

        self.assertEqual(
            decision.switch_penalty_units,
            1,
        )

        self.assertEqual(
            state.cumulative_switch_events,
            1,
        )


class TestBlock55Part2Recovery(
    Block55Part2Base
):

    def test_09_immediate_neighbors_middle(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        self.assertEqual(
            immediate_neighbor_indices(
                primary_beam_index=8,
                codebook=(
                    codebook
                ),
            ),
            (
                7,
                9,
            ),
        )

    def test_10_immediate_neighbors_edge(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        self.assertEqual(
            immediate_neighbor_indices(
                primary_beam_index=0,
                codebook=(
                    codebook
                ),
            ),
            (
                1,
            ),
        )

    def test_11_widened_64_to_32(self):
        codebook = (
            build_frozen_directional_codebook(
                64
            )
        )

        fallback = (
            widened_fallback_for_primary(
                primary_beam_index=40,
                codebook=(
                    codebook
                ),
            )
        )

        self.assertTrue(
            fallback.available
        )

        self.assertEqual(
            fallback.fallback_beam_count,
            32,
        )

    def test_12_widened_32_to_16(self):
        codebook = (
            build_frozen_directional_codebook(
                32
            )
        )

        fallback = (
            widened_fallback_for_primary(
                primary_beam_index=20,
                codebook=(
                    codebook
                ),
            )
        )

        self.assertTrue(
            fallback.available
        )

        self.assertEqual(
            fallback.fallback_beam_count,
            16,
        )

    def test_13_no_coarser_than_16(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        fallback = (
            widened_fallback_for_primary(
                primary_beam_index=8,
                codebook=(
                    codebook
                ),
            )
        )

        self.assertFalse(
            fallback.available
        )

        self.assertIsNone(
            fallback.fallback_beam_count
        )

    def test_14_outside_support_loss_of_lock_triggers_exhaustive(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        probability = self.probability(
            16,
            (
                (
                    8,
                    0.55,
                ),
                (
                    7,
                    0.30,
                ),
                (
                    9,
                    0.05,
                ),
            ),
            outside=0.10,
        )

        decision, _ = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )

        self.assertTrue(
            decision.loss_of_lock
        )

        self.assertTrue(
            decision
            .exhaustive_fallback_active
        )

        self.assertEqual(
            decision
            .exhaustive_fallback_indices,
            tuple(
                range(
                    16
                )
            ),
        )

        self.assertEqual(
            decision.loss_of_lock_reason,
            (
                "outside_support_mass_prevents_"
                "requested_coverage"
            ),
        )


class TestBlock55Part2CoverageAndLeakage(
    Block55Part2Base
):

    def test_15_all_q_values_supported(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        probability = self.probability(
            16,
            (
                (
                    8,
                    0.70,
                ),
                (
                    7,
                    0.20,
                ),
                (
                    9,
                    0.08,
                ),
                (
                    6,
                    0.02,
                ),
            ),
        )

        decisions = (
            all_coverage_temporal_decisions(
                state=(
                    initial_adaptive_temporal_state()
                ),

                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),
            )
        )

        self.assertEqual(
            tuple(
                decisions.keys()
            ),
            FROZEN_COVERAGE_TARGETS,
        )

    def test_16_requested_mass_is_not_renormalized(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        probability = self.probability(
            16,
            (
                (
                    8,
                    0.80,
                ),
            ),
            outside=0.20,
        )

        decision, _ = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )

        self.assertFalse(
            decision
            .adaptive_selection
            .achieved_requested_coverage
        )

        self.assertAlmostEqual(
            decision
            .adaptive_selection
            .inside_support_mass,
            0.80,
        )

    def test_17_controller_API_has_no_truth_or_oracle(self):
        signature = str(
            inspect.signature(
                adaptive_temporal_step
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

    def test_18_reset_is_clean(self):
        self.assertEqual(
            reset_adaptive_temporal_state(),
            initial_adaptive_temporal_state(),
        )

    def test_19_exact_repeat(self):
        codebook = (
            build_frozen_directional_codebook(
                32
            )
        )

        probability = self.probability(
            32,
            (
                (
                    16,
                    0.50,
                ),
                (
                    15,
                    0.30,
                ),
                (
                    17,
                    0.15,
                ),
                (
                    14,
                    0.05,
                ),
            ),
        )

        first = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )

        second = adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )

        self.assertEqual(
            first,
            second,
        )

    def test_20_kmax_never_below_codebook_size(self):
        for count in (
            16,
            32,
            64,
        ):
            codebook = (
                build_frozen_directional_codebook(
                    count
                )
            )

            self.assertEqual(
                frozen_kmax_for_codebook(
                    codebook
                ),
                count,
            )


if __name__ == "__main__":
    unittest.main()

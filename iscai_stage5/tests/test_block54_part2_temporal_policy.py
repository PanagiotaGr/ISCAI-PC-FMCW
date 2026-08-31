import inspect
import unittest

import numpy as np

from iscai_stage5.beam_baseline_temporal import (
    ORACLE_ACCESS_POLICY,
    PERSISTENCE_INITIALIZATION,
    PERSISTENCE_RESET,
    PERSISTENCE_STATE_SEMANTICS,
    PERSISTENCE_UPDATE,
    REALIZED_RECEIVER_AZIMUTH_POLICY,
    PersistenceState,
    fixed_probability_baseline_suite,
    frozen_temporal_baseline_contract,
    initial_persistence_state,
    persistence_step,
    reset_persistence_state,
    validate_controller_selection,
    validate_evaluator_oracle,
)

from iscai_stage5.beam_baselines import (
    exhaustive_sweep,
    oracle_best_gain_beam,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)


class TestBlock54Part2Persistence(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                32
            )
        )

    def test_01_initial_state_empty(self):
        state = (
            initial_persistence_state()
        )

        self.assertIsNone(
            state.previous_beam_index
        )

        self.assertEqual(
            state.step_count,
            0,
        )

    def test_02_first_step_uses_geometry_fallback(self):
        target = (
            self.codebook.cells[
                18
            ].center_azimuth_rad
        )

        selection, state = (
            persistence_step(
                state=(
                    initial_persistence_state()
                ),

                causal_predicted_azimuth_rad=(
                    target
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            selection.beam_indices,
            (
                18,
            ),
        )

        self.assertTrue(
            selection.fallback_used
        )

        self.assertEqual(
            state.previous_beam_index,
            18,
        )

        self.assertEqual(
            state.step_count,
            1,
        )

    def test_03_second_step_reuses_previous(self):
        first_angle = (
            self.codebook.cells[
                18
            ].center_azimuth_rad
        )

        _, state = persistence_step(
            state=(
                initial_persistence_state()
            ),

            causal_predicted_azimuth_rad=(
                first_angle
            ),

            codebook=(
                self.codebook
            ),
        )

        very_different_angle = (
            self.codebook.cells[
                4
            ].center_azimuth_rad
        )

        selection, state = (
            persistence_step(
                state=(
                    state
                ),

                causal_predicted_azimuth_rad=(
                    very_different_angle
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            selection.beam_indices,
            (
                18,
            ),
        )

        self.assertFalse(
            selection.fallback_used
        )

        self.assertEqual(
            state.previous_beam_index,
            18,
        )

        self.assertEqual(
            state.step_count,
            2,
        )

    def test_04_reset_clears_previous_beam(self):
        reset = (
            reset_persistence_state()
        )

        self.assertEqual(
            reset,
            PersistenceState(
                previous_beam_index=None,
                step_count=0,
            ),
        )

    def test_05_reset_next_step_reinitializes_geometry(self):
        target = (
            self.codebook.cells[
                7
            ].center_azimuth_rad
        )

        selection, _ = persistence_step(
            state=(
                reset_persistence_state()
            ),

            causal_predicted_azimuth_rad=(
                target
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            selection.beam_indices,
            (
                7,
            ),
        )

        self.assertTrue(
            selection.fallback_used
        )

    def test_06_invalid_previous_index_rejected(self):
        state = PersistenceState(
            previous_beam_index=100,
            step_count=1,
        )

        with self.assertRaises(
            ValueError
        ):
            persistence_step(
                state=(
                    state
                ),

                causal_predicted_azimuth_rad=0.0,

                codebook=(
                    self.codebook
                ),
            )


class TestBlock54Part2FixedSuite(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

    def test_07_fixed_suite_has_exact_K_1_3_5(self):
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
        ] = 0.15

        probability = BeamProbabilityMass(
            masses=tuple(
                masses
            ),

            inside_support_mass=1.0,

            outside_support_mass=0.0,

            sample_count=2048,
        )

        suite = (
            fixed_probability_baseline_suite(
                probability=(
                    probability
                ),

                codebook=(
                    self.codebook
                ),
            )
        )

        self.assertEqual(
            tuple(
                suite.keys()
            ),
            (
                1,
                3,
                5,
            ),
        )

        self.assertEqual(
            suite[
                1
            ].probing_count,
            1,
        )

        self.assertEqual(
            suite[
                3
            ].probing_count,
            3,
        )

        self.assertEqual(
            suite[
                5
            ].probing_count,
            5,
        )

    def test_08_fixed_suite_is_not_adaptive(self):
        contract = (
            frozen_temporal_baseline_contract()
        )

        self.assertFalse(
            contract[
                "adaptive_TopK"
            ]
        )


class TestBlock54Part2OracleIsolation(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

    def test_09_controller_validator_accepts_exhaustive(self):
        selection = exhaustive_sweep(
            codebook=(
                self.codebook
            )
        )

        validate_controller_selection(
            selection=(
                selection
            ),

            codebook=(
                self.codebook
            ),
        )

    def test_10_controller_validator_rejects_oracle(self):
        oracle = oracle_best_gain_beam(
            realized_azimuth_rad=(
                self.codebook.cells[
                    4
                ].center_azimuth_rad
            ),

            codebook=(
                self.codebook
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            validate_controller_selection(
                selection=(
                    oracle
                ),

                codebook=(
                    self.codebook
                ),
            )

    def test_11_evaluator_validator_accepts_oracle(self):
        oracle = oracle_best_gain_beam(
            realized_azimuth_rad=(
                self.codebook.cells[
                    4
                ].center_azimuth_rad
            ),

            codebook=(
                self.codebook
            ),
        )

        validate_evaluator_oracle(
            selection=(
                oracle
            ),

            codebook=(
                self.codebook
            ),
        )

    def test_12_evaluator_validator_rejects_controller_selection(self):
        selection = exhaustive_sweep(
            codebook=(
                self.codebook
            )
        )

        with self.assertRaises(
            ValueError
        ):
            validate_evaluator_oracle(
                selection=(
                    selection
                ),

                codebook=(
                    self.codebook
                ),
            )


class TestBlock54Part2Contract(
    unittest.TestCase
):

    def test_13_temporal_semantics_strings_frozen(self):
        self.assertIn(
            "previous_selected_beam",
            PERSISTENCE_STATE_SEMANTICS,
        )

        self.assertIn(
            "geometry_nearest",
            PERSISTENCE_INITIALIZATION,
        )

        self.assertIn(
            "selected_beam",
            PERSISTENCE_UPDATE,
        )

        self.assertEqual(
            PERSISTENCE_RESET,
            "clear_previous_beam_state",
        )

    def test_14_oracle_access_is_evaluator_only(self):
        self.assertEqual(
            ORACLE_ACCESS_POLICY,
            (
                "evaluator_only_after_"
                "controller_decision"
            ),
        )

    def test_15_realized_angle_is_not_controller_input(self):
        self.assertEqual(
            REALIZED_RECEIVER_AZIMUTH_POLICY,
            (
                "evaluation_only_never_"
                "controller_input"
            ),
        )

    def test_16_blockage_aware_not_started(self):
        contract = (
            frozen_temporal_baseline_contract()
        )

        self.assertFalse(
            contract[
                "blockage_aware"
            ]
        )

    def test_17_persistence_API_has_no_future_truth(self):
        signature = str(
            inspect.signature(
                persistence_step
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

    def test_18_persistence_sequence_exact_repeat(self):
        angles = [
            self.codebook_angle(
                10
            ),
            self.codebook_angle(
                3
            ),
            self.codebook_angle(
                14
            ),
        ]

        def run():
            codebook = (
                build_frozen_directional_codebook(
                    16
                )
            )

            state = (
                initial_persistence_state()
            )

            selected = []

            for angle in angles:

                selection, state = (
                    persistence_step(
                        state=(
                            state
                        ),

                        causal_predicted_azimuth_rad=(
                            angle
                        ),

                        codebook=(
                            codebook
                        ),
                    )
                )

                selected.append(
                    selection.beam_indices[
                        0
                    ]
                )

            return (
                tuple(
                    selected
                ),
                state,
            )

        first = run()
        second = run()

        self.assertEqual(
            first,
            second,
        )

    @staticmethod
    def codebook_angle(
        index,
    ):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        return (
            codebook.cells[
                index
            ].center_azimuth_rad
        )


if __name__ == "__main__":
    unittest.main()

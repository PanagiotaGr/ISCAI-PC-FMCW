import inspect
import math
import unittest

import numpy as np

from iscai_stage5.beam_baselines import (
    CONTROLLER_BASELINE_NAMES,
    EVALUATOR_ONLY_BASELINE_NAMES,
    FIXED_TOP_K_VALUES,
    FROZEN_BASELINE_NAMES,
    exhaustive_sweep,
    fixed_top_k_probability,
    frozen_baseline_registry,
    geometry_nearest_beam,
    oracle_best_gain_beam,
    previous_beam_persistence,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)


class TestBlock54BaselineRegistry(
    unittest.TestCase
):

    def test_01_required_baselines_present(self):
        self.assertEqual(
            FROZEN_BASELINE_NAMES,
            (
                "exhaustive_sweep",
                "previous_beam_persistence",
                "geometry_nearest",
                "fixed_top1_probability",
                "fixed_top3_probability",
                "fixed_top5_probability",
                "oracle_best_gain_eval_only",
            ),
        )

    def test_02_fixed_k_values(self):
        self.assertEqual(
            FIXED_TOP_K_VALUES,
            (
                1,
                3,
                5,
            ),
        )

    def test_03_oracle_not_controller_baseline(self):
        self.assertNotIn(
            "oracle_best_gain_eval_only",
            CONTROLLER_BASELINE_NAMES,
        )

        self.assertIn(
            "oracle_best_gain_eval_only",
            EVALUATOR_ONLY_BASELINE_NAMES,
        )

    def test_04_adaptive_topk_not_in_registry(self):
        registry = (
            frozen_baseline_registry()
        )

        self.assertFalse(
            registry[
                "adaptive_TopK_included"
            ]
        )


class TestBlock54ExhaustiveAndFixedTopK(
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
        return BeamProbabilityMass(
            masses=tuple(
                float(
                    value
                )
                for value in masses
            ),

            inside_support_mass=float(
                sum(
                    masses
                )
            ),

            outside_support_mass=float(
                outside
            ),

            sample_count=2048,
        )

    def test_05_exhaustive_probes_all_16(self):
        result = exhaustive_sweep(
            codebook=self.codebook
        )

        self.assertEqual(
            result.beam_indices,
            tuple(
                range(
                    16
                )
            ),
        )

        self.assertEqual(
            result.probing_count,
            16,
        )

    def test_06_fixed_top1_selects_max_Pb(self):
        masses = np.zeros(
            16
        )

        masses[
            7
        ] = 0.6

        masses[
            3
        ] = 0.4

        result = fixed_top_k_probability(
            probability=(
                self.probability(
                    masses
                )
            ),

            codebook=(
                self.codebook
            ),

            k=1,
        )

        self.assertEqual(
            result.beam_indices,
            (
                7,
            ),
        )

    def test_07_fixed_top3_ordered_by_Pb(self):
        masses = np.zeros(
            16
        )

        masses[
            8
        ] = 0.50

        masses[
            2
        ] = 0.30

        masses[
            11
        ] = 0.20

        result = fixed_top_k_probability(
            probability=(
                self.probability(
                    masses
                )
            ),

            codebook=(
                self.codebook
            ),

            k=3,
        )

        self.assertEqual(
            result.beam_indices,
            (
                8,
                2,
                11,
            ),
        )

    def test_08_fixed_top5_deterministic_tie_break(self):
        masses = np.zeros(
            16
        )

        masses[
            2
        ] = 0.25

        masses[
            3
        ] = 0.25

        masses[
            4
        ] = 0.25

        masses[
            5
        ] = 0.25

        result = fixed_top_k_probability(
            probability=(
                self.probability(
                    masses
                )
            ),

            codebook=(
                self.codebook
            ),

            k=5,
        )

        self.assertEqual(
            result.beam_indices[
                :4
            ],
            (
                2,
                3,
                4,
                5,
            ),
        )

        self.assertEqual(
            result.beam_indices[
                4
            ],
            0,
        )

    def test_09_invalid_fixed_k_rejected(self):
        masses = np.ones(
            16
        ) / 16.0

        with self.assertRaises(
            ValueError
        ):
            fixed_top_k_probability(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                k=2,
            )

    def test_10_probability_size_mismatch_rejected(self):
        masses = np.ones(
            15
        ) / 15.0

        with self.assertRaises(
            ValueError
        ):
            fixed_top_k_probability(
                probability=(
                    self.probability(
                        masses
                    )
                ),

                codebook=(
                    self.codebook
                ),

                k=1,
            )

    def test_11_outside_support_flag_propagated(self):
        masses = np.ones(
            16
        ) * (
            0.75
            /
            16.0
        )

        result = fixed_top_k_probability(
            probability=(
                self.probability(
                    masses,
                    outside=0.25,
                )
            ),

            codebook=(
                self.codebook
            ),

            k=3,
        )

        self.assertTrue(
            result.outside_support
        )


class TestBlock54GeometryAndPersistence(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

    def test_12_geometry_exact_center(self):
        cell = (
            self.codebook.cells[
                9
            ]
        )

        result = geometry_nearest_beam(
            predicted_azimuth_rad=(
                cell.center_azimuth_rad
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            result.beam_indices,
            (
                9,
            ),
        )

        self.assertFalse(
            result.outside_support
        )

    def test_13_geometry_boundary_tie_is_lower_index(self):
        left = (
            self.codebook.cells[
                7
            ]
        )

        right = (
            self.codebook.cells[
                8
            ]
        )

        boundary = 0.5 * (
            left.center_azimuth_rad
            +
            right.center_azimuth_rad
        )

        result = geometry_nearest_beam(
            predicted_azimuth_rad=(
                boundary
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            result.beam_indices,
            (
                7,
            ),
        )

    def test_14_geometry_outside_clips_to_edge_and_flags(self):
        result = geometry_nearest_beam(
            predicted_azimuth_rad=(
                math.radians(
                    30.0
                )
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            result.beam_indices,
            (
                15,
            ),
        )

        self.assertTrue(
            result.outside_support
        )

    def test_15_persistence_reuses_previous(self):
        result = previous_beam_persistence(
            previous_beam_index=6,

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            result.beam_indices,
            (
                6,
            ),
        )

        self.assertFalse(
            result.fallback_used
        )

    def test_16_persistence_initialization_uses_geometry(self):
        target = (
            self.codebook.cells[
                10
            ].center_azimuth_rad
        )

        result = previous_beam_persistence(
            previous_beam_index=None,

            codebook=(
                self.codebook
            ),

            fallback_predicted_azimuth_rad=(
                target
            ),
        )

        self.assertEqual(
            result.beam_indices,
            (
                10,
            ),
        )

        self.assertTrue(
            result.fallback_used
        )

    def test_17_persistence_without_state_or_fallback_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            previous_beam_persistence(
                previous_beam_index=None,

                codebook=(
                    self.codebook
                ),
            )


class TestBlock54OracleAndLeakage(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

    def test_18_oracle_best_gain_at_center(self):
        cell = (
            self.codebook.cells[
                5
            ]
        )

        result = oracle_best_gain_beam(
            realized_azimuth_rad=(
                cell.center_azimuth_rad
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertEqual(
            result.beam_indices,
            (
                5,
            ),
        )

        self.assertTrue(
            result.evaluation_only
        )

    def test_19_oracle_outside_support_is_explicit(self):
        result = oracle_best_gain_beam(
            realized_azimuth_rad=(
                math.radians(
                    30.0
                )
            ),

            codebook=(
                self.codebook
            ),
        )

        self.assertTrue(
            result.outside_support
        )

        self.assertTrue(
            result.evaluation_only
        )

        self.assertEqual(
            result.beam_indices,
            (
                15,
            ),
        )

    def test_20_controller_APIs_have_no_truth_or_oracle_inputs(self):
        controller_APIs = (
            exhaustive_sweep,
            fixed_top_k_probability,
            geometry_nearest_beam,
            previous_beam_persistence,
        )

        forbidden = (
            "future",
            "ground_truth",
            "realized",
            "oracle",
            "tracks_to_predict",
            "objects_of_interest",
        )

        for function in (
            controller_APIs
        ):

            signature = str(
                inspect.signature(
                    function
                )
            ).lower()

            for token in forbidden:

                self.assertNotIn(
                    token,
                    signature,
                )


if __name__ == "__main__":
    unittest.main()

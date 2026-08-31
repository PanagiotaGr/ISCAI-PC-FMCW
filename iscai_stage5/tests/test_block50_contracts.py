import unittest

from iscai_stage5.contracts import (
    BEAM_BASELINES,
    BEAM_PROBABILITY_AND_EXPECTED_GAIN_ARE_DISTINCT,
    CODEBOOK_SIZES,
    DEFAULT_PROBABILITY_MASS_TARGET,
    EFFECTIVE_RATE_FORMULA_STATUS,
    EFFECTIVE_RATE_PROVENANCE,
    FORMAL_POPULATION_N,
    FORMAL_TUNING_ALLOWED,
    FUTURE_GROUND_TRUTH_HEADING_ALLOWED_IN_CONTROLLER,
    FUTURE_TRUTH_IS_RECEIVER_SELECTOR,
    HORIZONS_S,
    NEW_ARBITRARY_OPTICAL_MODEL_ALLOWED,
    OPTICAL_LINK_CHAIN,
    PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED,
    PRIMARY_RECEIVER_POLICY,
    PROBABILITY_MASS_TARGETS,
    RECEIVER_GEOMETRY_MODES,
    STAGE4_DEFAULT_POSTERIOR,
    STAGE4_POSTHOC_GMM_SELECTION_ALLOWED,
    STAGE4_RECALIBRATION_ALLOWED,
    STAGE4_RETRAINING_ALLOWED,
    TRACKS_TO_PREDICT_IS_RECEIVER_SELECTOR,
    contract_dict,
)


class TestBlock50Contracts(
    unittest.TestCase
):

    def test_01_frozen_stage4_posterior(self):
        self.assertEqual(
            STAGE4_DEFAULT_POSTERIOR,
            "calibrated_Gaussian_GRU",
        )

        self.assertFalse(
            STAGE4_RETRAINING_ALLOWED
        )

        self.assertFalse(
            STAGE4_RECALIBRATION_ALLOWED
        )

        self.assertFalse(
            STAGE4_POSTHOC_GMM_SELECTION_ALLOWED
        )

    def test_02_horizons(self):
        self.assertEqual(
            HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

    def test_03_receiver_policy_is_causal(self):
        self.assertEqual(
            PRIMARY_RECEIVER_POLICY,
            "nearest_causal_vehicle_ahead",
        )

        self.assertFalse(
            TRACKS_TO_PREDICT_IS_RECEIVER_SELECTOR
        )

        self.assertFalse(
            FUTURE_TRUTH_IS_RECEIVER_SELECTOR
        )

    def test_04_receiver_geometry_modes(self):
        self.assertEqual(
            RECEIVER_GEOMETRY_MODES,
            (
                "centroid_baseline",
                "known_receiver_offset",
                "uncertain_receiver_offset",
            ),
        )

        self.assertFalse(
            FUTURE_GROUND_TRUTH_HEADING_ALLOWED_IN_CONTROLLER
        )

    def test_05_codebook_sizes(self):
        self.assertEqual(
            CODEBOOK_SIZES,
            (
                16,
                32,
                64,
            ),
        )

    def test_06_probability_mass_targets(self):
        self.assertEqual(
            PROBABILITY_MASS_TARGETS,
            (
                0.90,
                0.95,
                0.975,
                0.99,
            ),
        )

        self.assertEqual(
            DEFAULT_PROBABILITY_MASS_TARGET,
            0.95,
        )

    def test_07_beam_probability_not_gain(self):
        self.assertTrue(
            BEAM_PROBABILITY_AND_EXPECTED_GAIN_ARE_DISTINCT
        )

    def test_08_required_baselines(self):
        self.assertEqual(
            set(
                BEAM_BASELINES
            ),
            {
                "exhaustive_sweep",
                "previous_beam_persistence",
                "geometry_nearest_beam",
                "fixed_Top1",
                "fixed_Top3",
                "fixed_Top5",
                "oracle_beam_set",
            },
        )

    def test_09_optical_link_chain(self):
        self.assertEqual(
            OPTICAL_LINK_CHAIN,
            (
                "pointing_error",
                "optical_beam_gain",
                "received_power",
                "SNR",
                "DPSK_BER",
                "effective_rate",
            ),
        )

        self.assertFalse(
            NEW_ARBITRARY_OPTICAL_MODEL_ALLOWED
        )

        self.assertFalse(
            PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED
        )

        self.assertIn(
            "Stage5_derived",
            EFFECTIVE_RATE_PROVENANCE,
        )

        self.assertEqual(
            EFFECTIVE_RATE_FORMULA_STATUS,
            "MUST_FREEZE_IN_BLOCK5.6_BEFORE_FORMAL",
        )

    def test_10_formal_is_evaluation_only(self):
        self.assertEqual(
            FORMAL_POPULATION_N,
            120,
        )

        self.assertFalse(
            FORMAL_TUNING_ALLOWED
        )

    def test_11_scope_boundary(self):
        contract = contract_dict()

        self.assertIn(
            "predictive_class_aware_ADB",
            contract[
                "out_of_scope"
            ],
        )

        self.assertIn(
            "DeepSense_external_mmWave_validation",
            contract[
                "out_of_scope"
            ],
        )

        self.assertEqual(
            contract[
                "next_stage"
            ],
            6,
        )

    def test_12_development_freeze_before_formal(self):
        contract = contract_dict()

        freeze = contract[
            "development_freeze"
        ]

        self.assertTrue(
            freeze[
                "must_complete_before_formal"
            ]
        )

        self.assertEqual(
            freeze[
                "source"
            ],
            "non_formal_development_partition_only",
        )

        self.assertIn(
            "formal_empirical_coverage_tolerance",
            freeze[
                "fields"
            ],
        )


if __name__ == "__main__":
    unittest.main()

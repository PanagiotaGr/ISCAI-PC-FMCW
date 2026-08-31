import inspect
import math
import unittest

import numpy as np

from iscai_stage5.beam_codebook import (
    BEAM_PROBABILITY_SEMANTICS,
    DECISION_CELL_SEMANTICS,
    EXPECTED_GAIN_SCORE_SEMANTICS,
    PHYSICAL_GAIN_SEMANTICS,
    SUPPORTED_BEAM_COUNTS,
    beam_probability_mass_from_samples,
    build_uniform_azimuth_codebook,
    expected_gain_scores_from_samples,
)


class TestBlock53Codebook(
    unittest.TestCase
):

    def test_01_supported_beam_counts(self):
        self.assertEqual(
            SUPPORTED_BEAM_COUNTS,
            (
                16,
                32,
                64,
            ),
        )

    def test_02_uniform_16_beam_width(self):
        codebook = (
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=-1.0,
                support_max_azimuth_rad=1.0,
            )
        )

        self.assertAlmostEqual(
            codebook.decision_width_rad,
            2.0
            /
            16.0,
        )

        self.assertEqual(
            len(
                codebook.cells
            ),
            16,
        )

    def test_03_centers_are_midpoints(self):
        codebook = (
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=-1.0,
                support_max_azimuth_rad=1.0,
            )
        )

        for cell in codebook.cells:

            self.assertAlmostEqual(
                cell.center_azimuth_rad,
                0.5
                *
                (
                    cell.lower_azimuth_rad
                    +
                    cell.upper_azimuth_rad
                ),
            )

    def test_04_decision_cells_contiguous(self):
        codebook = (
            build_uniform_azimuth_codebook(
                beam_count=32,
                support_min_azimuth_rad=-1.2,
                support_max_azimuth_rad=1.3,
            )
        )

        for left, right in zip(
            codebook.cells[
                :-1
            ],
            codebook.cells[
                1:
            ],
        ):
            self.assertAlmostEqual(
                left.upper_azimuth_rad,
                right.lower_azimuth_rad,
            )

    def test_05_cells_cover_exact_support(self):
        codebook = (
            build_uniform_azimuth_codebook(
                beam_count=64,
                support_min_azimuth_rad=-0.8,
                support_max_azimuth_rad=0.9,
            )
        )

        self.assertAlmostEqual(
            codebook.cells[
                0
            ].lower_azimuth_rad,
            -0.8,
        )

        self.assertAlmostEqual(
            codebook.cells[
                -1
            ].upper_azimuth_rad,
            0.9,
        )

    def test_06_invalid_beam_count_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            build_uniform_azimuth_codebook(
                beam_count=24,
                support_min_azimuth_rad=-1.0,
                support_max_azimuth_rad=1.0,
            )

    def test_07_invalid_support_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=1.0,
                support_max_azimuth_rad=-1.0,
            )

    def test_08_support_above_2pi_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=0.0,
                support_max_azimuth_rad=(
                    2.0
                    *
                    math.pi
                    +
                    0.1
                ),
            )

    def test_09_semantics_separate_cells_and_gain(self):
        codebook = (
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=-1.0,
                support_max_azimuth_rad=1.0,
            )
        )

        self.assertEqual(
            codebook.decision_cell_semantics,
            DECISION_CELL_SEMANTICS,
        )

        self.assertEqual(
            codebook.physical_gain_semantics,
            PHYSICAL_GAIN_SEMANTICS,
        )

        self.assertNotEqual(
            DECISION_CELL_SEMANTICS,
            PHYSICAL_GAIN_SEMANTICS,
        )


class TestBlock53ProbabilityMass(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=-1.0,
                support_max_azimuth_rad=1.0,
            )
        )

    def test_10_all_inside_mass_sums_to_one(self):
        samples = np.linspace(
            -0.99,
            0.99,
            1600,
        )

        result = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
            )
        )

        self.assertAlmostEqual(
            sum(
                result.masses
            ),
            1.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.outside_support_mass,
            0.0,
            places=12,
        )

    def test_11_outside_mass_is_explicit(self):
        samples = np.asarray(
            [
                -2.0,
                -0.5,
                0.5,
                2.0,
            ]
        )

        result = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
            )
        )

        self.assertAlmostEqual(
            result.inside_support_mass,
            0.5,
        )

        self.assertAlmostEqual(
            result.outside_support_mass,
            0.5,
        )

    def test_12_final_upper_boundary_is_assigned(self):
        samples = np.asarray(
            [
                1.0,
            ]
        )

        result = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
            )
        )

        self.assertAlmostEqual(
            result.masses[
                -1
            ],
            1.0,
        )

    def test_13_mass_plus_outside_is_one(self):
        samples = np.asarray(
            [
                -1.5,
                -0.75,
                -0.25,
                0.25,
                0.75,
                1.5,
            ]
        )

        result = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
            )
        )

        self.assertAlmostEqual(
            (
                sum(
                    result.masses
                )
                +
                result.outside_support_mass
            ),
            1.0,
            places=12,
        )

    def test_14_sample_order_does_not_change_mass(self):
        samples = np.asarray(
            [
                -0.9,
                -0.1,
                0.2,
                0.8,
                1.5,
            ]
        )

        first = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
            )
        )

        second = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples[
                    ::-1
                ],
                codebook=self.codebook,
            )
        )

        np.testing.assert_allclose(
            first.masses,
            second.masses,
            atol=0.0,
            rtol=0.0,
        )

        self.assertEqual(
            first.outside_support_mass,
            second.outside_support_mass,
        )

    def test_15_probability_semantics_explicit(self):
        result = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=np.asarray(
                    [
                        0.0,
                    ]
                ),
                codebook=self.codebook,
            )
        )

        self.assertEqual(
            result.semantics,
            BEAM_PROBABILITY_SEMANTICS,
        )


class TestBlock53GainScore(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_uniform_azimuth_codebook(
                beam_count=16,
                support_min_azimuth_rad=-1.0,
                support_max_azimuth_rad=1.0,
            )
        )

    def test_16_constant_gain_scores(self):
        samples = np.asarray(
            [
                -0.8,
                -0.2,
                0.4,
                0.9,
            ]
        )

        def constant_gain(
            cell,
            values,
        ):
            return np.full_like(
                values,
                float(
                    cell.index
                    +
                    1
                ),
            )

        result = (
            expected_gain_scores_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
                gain_evaluator=constant_gain,
            )
        )

        self.assertAlmostEqual(
            result.scores[
                0
            ],
            1.0,
        )

        self.assertAlmostEqual(
            result.scores[
                -1
            ],
            16.0,
        )

        self.assertEqual(
            result.semantics,
            EXPECTED_GAIN_SCORE_SEMANTICS,
        )

    def test_17_gain_score_can_overlap_decision_cells(self):
        samples = np.asarray(
            [
                -0.02,
                0.00,
                0.02,
            ]
        )

        probability = (
            beam_probability_mass_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
            )
        )

        def overlapping_gain(
            cell,
            values,
        ):
            difference = np.abs(
                values
                -
                cell.center_azimuth_rad
            )

            return np.exp(
                -0.5
                *
                (
                    difference
                    /
                    0.30
                )
                **
                2
            )

        gain = (
            expected_gain_scores_from_samples(
                azimuth_samples_rad=samples,
                codebook=self.codebook,
                gain_evaluator=overlapping_gain,
            )
        )

        positive_probability_beams = sum(
            value
            >
            0.0
            for value in (
                probability.masses
            )
        )

        positive_gain_beams = sum(
            value
            >
            1e-6
            for value in (
                gain.scores
            )
        )

        self.assertGreater(
            positive_gain_beams,
            positive_probability_beams,
        )

    def test_18_public_APIs_have_no_future_or_oracle_input(self):
        APIs = (
            build_uniform_azimuth_codebook,
            beam_probability_mass_from_samples,
            expected_gain_scores_from_samples,
        )

        forbidden = (
            "future_truth",
            "ground_truth",
            "tracks_to_predict",
            "oracle",
            "objects_of_interest",
            "adb",
        )

        for function in APIs:

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

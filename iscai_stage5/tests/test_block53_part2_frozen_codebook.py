import inspect
import math
import unittest

import numpy as np

from iscai_stage5.beam_directional_gain import (
    FROZEN_GAIN_IS_MEASURED,
    FROZEN_GAIN_IS_PARTA_OPTICAL_LINK_GAIN,
    FROZEN_GAIN_MODEL,
    FROZEN_GAIN_ROLE,
    FROZEN_HPBW_RULE,
    FROZEN_SUPPORT_HALF_ANGLE_DEG,
    FROZEN_SUPPORT_IS_MEASURED,
    FROZEN_SUPPORT_IS_PAPER_GIVEN,
    FROZEN_SUPPORT_MAX_DEG,
    FROZEN_SUPPORT_MIN_DEG,
    FROZEN_SUPPORT_PROVENANCE,
    build_frozen_directional_codebook,
    frozen_beam_scores_from_samples,
    normalized_gaussian_directional_gain,
)


class TestBlock53Part2Support(
    unittest.TestCase
):

    def test_01_support_is_plus_minus_12_deg(self):
        self.assertEqual(
            FROZEN_SUPPORT_HALF_ANGLE_DEG,
            12.0,
        )

        self.assertEqual(
            FROZEN_SUPPORT_MIN_DEG,
            -12.0,
        )

        self.assertEqual(
            FROZEN_SUPPORT_MAX_DEG,
            12.0,
        )

    def test_02_support_provenance_is_constructed(self):
        self.assertIn(
            "constructed_Stage5",
            FROZEN_SUPPORT_PROVENANCE,
        )

        self.assertFalse(
            FROZEN_SUPPORT_IS_MEASURED
        )

        self.assertFalse(
            FROZEN_SUPPORT_IS_PAPER_GIVEN
        )

    def test_03_16_beam_width_is_1p5_deg(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        self.assertAlmostEqual(
            math.degrees(
                codebook.decision_width_rad
            ),
            1.5,
            places=12,
        )

    def test_04_32_beam_width_is_0p75_deg(self):
        codebook = (
            build_frozen_directional_codebook(
                32
            )
        )

        self.assertAlmostEqual(
            math.degrees(
                codebook.decision_width_rad
            ),
            0.75,
            places=12,
        )

    def test_05_64_beam_width_is_0p375_deg(self):
        codebook = (
            build_frozen_directional_codebook(
                64
            )
        )

        self.assertAlmostEqual(
            math.degrees(
                codebook.decision_width_rad
            ),
            0.375,
            places=12,
        )

    def test_06_codebook_is_symmetric(self):
        codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        centers = np.asarray(
            [
                cell.center_azimuth_rad
                for cell
                in codebook.cells
            ]
        )

        np.testing.assert_allclose(
            centers,
            -centers[
                ::-1
            ],
            atol=1e-15,
            rtol=0.0,
        )


class TestBlock53Part2Gain(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        self.cell = (
            self.codebook.cells[
                8
            ]
        )

    def test_07_gain_model_is_constructed_not_PartA_link_gain(self):
        self.assertIn(
            "constructed",
            FROZEN_GAIN_MODEL,
        )

        self.assertEqual(
            FROZEN_GAIN_ROLE,
            (
                "codebook_expected_gain_"
                "scoring_only"
            ),
        )

        self.assertFalse(
            FROZEN_GAIN_IS_PARTA_OPTICAL_LINK_GAIN
        )

        self.assertFalse(
            FROZEN_GAIN_IS_MEASURED
        )

    def test_08_HPBW_rule_is_explicit(self):
        self.assertEqual(
            FROZEN_HPBW_RULE,
            (
                "HPBW_equals_uniform_"
                "decision_cell_width"
            ),
        )

    def test_09_center_gain_is_one(self):
        gain = (
            normalized_gaussian_directional_gain(
                self.cell,

                np.asarray(
                    [
                        self.cell
                        .center_azimuth_rad
                    ]
                ),
            )
        )

        self.assertAlmostEqual(
            gain[
                0
            ],
            1.0,
            places=15,
        )

    def test_10_lower_boundary_is_half_power(self):
        gain = (
            normalized_gaussian_directional_gain(
                self.cell,

                np.asarray(
                    [
                        self.cell
                        .lower_azimuth_rad
                    ]
                ),
            )
        )

        self.assertAlmostEqual(
            gain[
                0
            ],
            0.5,
            places=14,
        )

    def test_11_upper_boundary_is_half_power(self):
        gain = (
            normalized_gaussian_directional_gain(
                self.cell,

                np.asarray(
                    [
                        self.cell
                        .upper_azimuth_rad
                    ]
                ),
            )
        )

        self.assertAlmostEqual(
            gain[
                0
            ],
            0.5,
            places=14,
        )

    def test_12_adjacent_beams_overlap_at_boundary(self):
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

        boundary = (
            left.upper_azimuth_rad
        )

        left_gain = (
            normalized_gaussian_directional_gain(
                left,
                np.asarray(
                    [
                        boundary
                    ]
                ),
            )[
                0
            ]
        )

        right_gain = (
            normalized_gaussian_directional_gain(
                right,
                np.asarray(
                    [
                        boundary
                    ]
                ),
            )[
                0
            ]
        )

        self.assertAlmostEqual(
            left_gain,
            0.5,
            places=14,
        )

        self.assertAlmostEqual(
            right_gain,
            0.5,
            places=14,
        )

    def test_13_gain_is_finite_nonnegative(self):
        values = np.linspace(
            math.radians(
                -30.0
            ),
            math.radians(
                30.0
            ),
            1001,
        )

        gain = (
            normalized_gaussian_directional_gain(
                self.cell,
                values,
            )
        )

        self.assertTrue(
            np.isfinite(
                gain
            ).all()
        )

        self.assertTrue(
            np.all(
                gain
                >=
                0.0
            )
        )

        self.assertLessEqual(
            float(
                np.max(
                    gain
                )
            ),
            1.0
            +
            1e-15,
        )


class TestBlock53Part2Scores(
    unittest.TestCase
):

    def test_14_probability_plus_outside_is_one(self):
        samples = np.radians(
            np.asarray(
                [
                    -20.0,
                    -10.0,
                    -1.0,
                    0.0,
                    1.0,
                    10.0,
                    20.0,
                ]
            )
        )

        result = (
            frozen_beam_scores_from_samples(
                azimuth_samples_rad=(
                    samples
                ),

                beam_count=(
                    16
                ),
            )
        )

        total = (
            sum(
                result
                .probability
                .masses
            )
            +
            result
            .probability
            .outside_support_mass
        )

        self.assertAlmostEqual(
            total,
            1.0,
            places=12,
        )

    def test_15_outside_support_is_not_silently_dropped(self):
        samples = np.radians(
            np.asarray(
                [
                    -30.0,
                    0.0,
                    30.0,
                ]
            )
        )

        result = (
            frozen_beam_scores_from_samples(
                azimuth_samples_rad=(
                    samples
                ),

                beam_count=(
                    16
                ),
            )
        )

        self.assertAlmostEqual(
            result
            .probability
            .outside_support_mass,
            2.0
            /
            3.0,
        )

    def test_16_P_and_S_have_same_beam_count_but_distinct_values(self):
        samples = np.radians(
            np.asarray(
                [
                    -0.2,
                    0.0,
                    0.2,
                ]
            )
        )

        result = (
            frozen_beam_scores_from_samples(
                azimuth_samples_rad=(
                    samples
                ),

                beam_count=(
                    16
                ),
            )
        )

        self.assertEqual(
            len(
                result
                .probability
                .masses
            ),
            16,
        )

        self.assertEqual(
            len(
                result
                .expected_gain
                .scores
            ),
            16,
        )

        self.assertFalse(
            np.allclose(
                np.asarray(
                    result
                    .probability
                    .masses
                ),

                np.asarray(
                    result
                    .expected_gain
                    .scores
                ),
            )
        )

    def test_17_scoring_is_exactly_repeatable(self):
        samples = np.radians(
            np.linspace(
                -15.0,
                15.0,
                2048,
            )
        )

        first = (
            frozen_beam_scores_from_samples(
                azimuth_samples_rad=(
                    samples
                ),

                beam_count=(
                    32
                ),
            )
        )

        second = (
            frozen_beam_scores_from_samples(
                azimuth_samples_rad=(
                    samples
                ),

                beam_count=(
                    32
                ),
            )
        )

        np.testing.assert_array_equal(
            np.asarray(
                first
                .probability
                .masses
            ),

            np.asarray(
                second
                .probability
                .masses
            ),
        )

        np.testing.assert_array_equal(
            np.asarray(
                first
                .expected_gain
                .scores
            ),

            np.asarray(
                second
                .expected_gain
                .scores
            ),
        )

    def test_18_public_APIs_have_no_formal_or_oracle_inputs(self):
        APIs = (
            build_frozen_directional_codebook,
            normalized_gaussian_directional_gain,
            frozen_beam_scores_from_samples,
        )

        forbidden = (
            "future_truth",
            "ground_truth",
            "tracks_to_predict",
            "objects_of_interest",
            "oracle",
            "formal",
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

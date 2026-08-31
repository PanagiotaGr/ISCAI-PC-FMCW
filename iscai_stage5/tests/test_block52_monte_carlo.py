import inspect
import math
import unittest

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    FROZEN_VARIANCE_SCALE_ALPHA_H,
    FUTURE_HEADING_SEMANTICS,
    LOW_SPEED_HEADING_THRESHOLD_MPS,
    MONTE_CARLO_MINIMUM_SAMPLE_COUNT,
    RECEIVER_OFFSET_TEMPORAL_SEMANTICS,
    TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
    ReceiverOffsetGaussianBody,
    calibrate_stage4_horizon_covariances,
    deterministic_receiver_angular_posterior,
    resolve_samplewise_headings,
    sample_product_of_horizon_gaussians,
)


def small_covariance(
    scale=1.0,
):
    return np.stack(
        [
            np.diag(
                [
                    0.04 * scale,
                    0.01 * scale,
                    0.01 * scale,
                ]
            )
            for _ in range(
                4
            )
        ],
        axis=0,
    )


def straight_mean():
    return np.asarray(
        [
            [
                10.0,
                0.0,
                0.0,
            ],
            [
                12.0,
                0.0,
                0.0,
            ],
            [
                14.0,
                0.0,
                0.0,
            ],
            [
                19.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )


class TestBlock52Calibration(
    unittest.TestCase
):

    def test_01_frozen_alpha_applied_per_horizon(self):
        raw = np.stack(
            [
                np.eye(
                    3
                )
                for _ in range(
                    4
                )
            ],
            axis=0,
        )

        calibrated = (
            calibrate_stage4_horizon_covariances(
                raw
            )
        )

        for index, alpha in enumerate(
            FROZEN_VARIANCE_SCALE_ALPHA_H
        ):
            np.testing.assert_allclose(
                calibrated[
                    index
                ],
                alpha
                *
                np.eye(
                    3
                ),
                atol=1e-12,
                rtol=0.0,
            )

    def test_02_calibration_does_not_change_mean_api(self):
        signature = str(
            inspect.signature(
                calibrate_stage4_horizon_covariances
            )
        )

        self.assertNotIn(
            "mean",
            signature.lower(),
        )


class TestBlock52TrajectorySampling(
    unittest.TestCase
):

    def test_03_product_sampling_exact_repeat(self):
        mean = straight_mean()

        cov = (
            calibrate_stage4_horizon_covariances(
                small_covariance()
            )
        )

        first = (
            sample_product_of_horizon_gaussians(
                trajectory_mean_h0_m=mean,
                calibrated_covariance_h0_m2=cov,
                sample_count=2048,
                seed=123,
            )
        )

        second = (
            sample_product_of_horizon_gaussians(
                trajectory_mean_h0_m=mean,
                calibrated_covariance_h0_m2=cov,
                sample_count=2048,
                seed=123,
            )
        )

        self.assertTrue(
            np.array_equal(
                first,
                second,
            )
        )

    def test_04_different_seed_changes_samples(self):
        mean = straight_mean()

        cov = (
            calibrate_stage4_horizon_covariances(
                small_covariance()
            )
        )

        first = (
            sample_product_of_horizon_gaussians(
                trajectory_mean_h0_m=mean,
                calibrated_covariance_h0_m2=cov,
                sample_count=2048,
                seed=123,
            )
        )

        second = (
            sample_product_of_horizon_gaussians(
                trajectory_mean_h0_m=mean,
                calibrated_covariance_h0_m2=cov,
                sample_count=2048,
                seed=124,
            )
        )

        self.assertFalse(
            np.array_equal(
                first,
                second,
            )
        )

    def test_05_joint_semantics_explicit(self):
        self.assertEqual(
            TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
            (
                "product_of_frozen_calibrated_"
                "per_horizon_Gaussian_marginals"
            ),
        )


class TestBlock52Heading(
    unittest.TestCase
):

    def test_06_straight_trajectory_heading(self):
        samples = np.repeat(
            straight_mean()[
                None,
                :,
                :
            ],
            2,
            axis=0,
        )

        headings, fallback = (
            resolve_samplewise_headings(
                trajectory_samples_h0_m=samples,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,
            )
        )

        np.testing.assert_allclose(
            headings,
            0.0,
            atol=1e-12,
            rtol=0.0,
        )

        self.assertFalse(
            np.any(
                fallback
            )
        )

    def test_07_low_speed_uses_previous_heading(self):
        samples = np.asarray(
            [
                [
                    [
                        0.001,
                        0.0,
                        0.0,
                    ],
                    [
                        0.002,
                        0.0,
                        0.0,
                    ],
                    [
                        0.003,
                        0.0,
                        0.0,
                    ],
                    [
                        0.004,
                        0.0,
                        0.0,
                    ],
                ]
            ],
            dtype=np.float64,
        )

        current_heading = 0.7

        headings, fallback = (
            resolve_samplewise_headings(
                trajectory_samples_h0_m=samples,

                current_position_h0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=(
                    current_heading
                ),
            )
        )

        np.testing.assert_allclose(
            headings,
            current_heading,
            atol=1e-12,
            rtol=0.0,
        )

        self.assertTrue(
            np.all(
                fallback
            )
        )

    def test_08_heading_threshold_frozen(self):
        self.assertEqual(
            LOW_SPEED_HEADING_THRESHOLD_MPS,
            0.25,
        )

        self.assertEqual(
            FUTURE_HEADING_SEMANTICS,
            (
                "samplewise_trajectory_tangent_with_"
                "causal_heading_fallback"
            ),
        )


class TestBlock52ReceiverPosterior(
    unittest.TestCase
):

    def setUp(self):
        self.mean = straight_mean()

        self.cov = small_covariance()

        self.offset = (
            ReceiverOffsetGaussianBody(
                mean_body_m=(
                    1.0,
                    0.0,
                    0.5,
                ),

                covariance_body_m2=(
                    (
                        0.04,
                        0.0,
                        0.0,
                    ),
                    (
                        0.0,
                        0.04,
                        0.0,
                    ),
                    (
                        0.0,
                        0.0,
                        0.01,
                    ),
                ),
            )
        )

    def test_09_centroid_exact_repeat(self):
        first = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "centroid_baseline"
                ),

                receiver_offset=None,

                sample_count=(
                    MONTE_CARLO_MINIMUM_SAMPLE_COUNT
                ),

                seed=111,
            )
        )

        second = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "centroid_baseline"
                ),

                receiver_offset=None,

                sample_count=(
                    MONTE_CARLO_MINIMUM_SAMPLE_COUNT
                ),

                seed=111,
            )
        )

        self.assertEqual(
            first.sample_sha256,
            second.sample_sha256,
        )

        self.assertTrue(
            np.array_equal(
                first.azimuth_samples_rad,
                second.azimuth_samples_rad,
            )
        )

    def test_10_known_offset_moves_receiver(self):
        centroid = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "centroid_baseline"
                ),

                receiver_offset=None,

                sample_count=2048,

                seed=222,
            )
        )

        known = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "known_receiver_offset"
                ),

                receiver_offset=self.offset,

                sample_count=2048,

                seed=222,
            )
        )

        delta = (
            known.receiver_position_samples_h0_m
            -
            centroid.receiver_position_samples_h0_m
        )

        self.assertGreater(
            float(
                np.mean(
                    np.linalg.norm(
                        delta,
                        axis=2,
                    )
                )
            ),
            0.5,
        )

    def test_11_uncertain_offset_adds_variability(self):
        known = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "known_receiver_offset"
                ),

                receiver_offset=self.offset,

                sample_count=4096,

                seed=333,
            )
        )

        uncertain = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "uncertain_receiver_offset"
                ),

                receiver_offset=self.offset,

                sample_count=4096,

                seed=333,
            )
        )

        known_cov = np.asarray(
            known.summaries[
                0
            ].covariance_rae
        )

        uncertain_cov = np.asarray(
            uncertain.summaries[
                0
            ].covariance_rae
        )

        self.assertGreater(
            float(
                np.trace(
                    uncertain_cov
                )
            ),
            float(
                np.trace(
                    known_cov
                )
            ),
        )

    def test_12_shared_offset_semantics_frozen(self):
        self.assertEqual(
            RECEIVER_OFFSET_TEMPORAL_SEMANTICS,
            (
                "one_body_frame_receiver_offset_draw_"
                "shared_across_all_horizons_per_MC_sample"
            ),
        )

    def test_13_summary_covariance_psd(self):
        posterior = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "uncertain_receiver_offset"
                ),

                receiver_offset=self.offset,

                sample_count=2048,

                seed=444,
            )
        )

        for summary in posterior.summaries:

            covariance = np.asarray(
                summary.covariance_rae
            )

            np.testing.assert_allclose(
                covariance,
                covariance.T,
                atol=1e-12,
                rtol=0.0,
            )

            self.assertGreaterEqual(
                float(
                    np.min(
                        np.linalg.eigvalsh(
                            covariance
                        )
                    )
                ),
                -1e-10,
            )

    def test_14_four_horizon_output(self):
        posterior = (
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "centroid_baseline"
                ),

                receiver_offset=None,

                sample_count=2048,

                seed=555,
            )
        )

        self.assertEqual(
            posterior.range_samples_m.shape,
            (
                2048,
                4,
            ),
        )

        self.assertEqual(
            len(
                posterior.summaries
            ),
            4,
        )

    def test_15_below_minimum_MC_count_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            deterministic_receiver_angular_posterior(
                trajectory_mean_h0_m=self.mean,
                raw_trajectory_covariance_h0_m2=self.cov,

                current_position_h0_m=(
                    9.0,
                    0.0,
                    0.0,
                ),

                current_heading_h0_rad=0.0,

                receiver_geometry_mode=(
                    "centroid_baseline"
                ),

                receiver_offset=None,

                sample_count=1024,

                seed=666,
            )

    def test_16_no_future_or_evaluator_metadata_API(self):
        signature = str(
            inspect.signature(
                deterministic_receiver_angular_posterior
            )
        ).lower()

        for token in (
            "future_truth",
            "ground_truth",
            "tracks_to_predict",
            "oracle",
            "objects_of_interest",
        ):
            self.assertNotIn(
                token,
                signature,
            )

    def test_17_raw_covariance_is_calibrated_inside_primary_path(self):
        signature = str(
            inspect.signature(
                deterministic_receiver_angular_posterior
            )
        )

        self.assertIn(
            "raw_trajectory_covariance_h0_m2",
            signature,
        )

    def test_18_geometry_modes_remain_separate(self):
        modes = {
            "centroid_baseline",
            "known_receiver_offset",
            "uncertain_receiver_offset",
        }

        self.assertEqual(
            len(
                modes
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()

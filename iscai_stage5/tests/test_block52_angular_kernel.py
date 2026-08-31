import inspect
import math
import unittest

import numpy as np

from iscai_stage5.angular_posterior import (
    GaussianPositionH0,
    ANALYTIC_CROSSCHECK_METHOD,
    PRIMARY_STAGE52_METHOD,
    circular_angle_difference,
    combine_independent_position_uncertainties,
    linearized_position_gaussian_to_angular,
    position_h0_to_range_azimuth_elevation,
    range_azimuth_elevation_jacobian,
)


class TestBlock52AngularGeometry(
    unittest.TestCase
):

    def test_01_positive_x_axis(self):
        result = (
            position_h0_to_range_azimuth_elevation(
                (
                    10.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertAlmostEqual(
            result.range_m,
            10.0,
        )

        self.assertAlmostEqual(
            result.azimuth_rad,
            0.0,
        )

        self.assertAlmostEqual(
            result.elevation_rad,
            0.0,
        )

    def test_02_positive_y_axis(self):
        result = (
            position_h0_to_range_azimuth_elevation(
                (
                    0.0,
                    10.0,
                    0.0,
                )
            )
        )

        self.assertAlmostEqual(
            result.range_m,
            10.0,
        )

        self.assertAlmostEqual(
            result.azimuth_rad,
            math.pi
            /
            2.0,
        )

        self.assertAlmostEqual(
            result.elevation_rad,
            0.0,
        )

    def test_03_nonzero_elevation(self):
        result = (
            position_h0_to_range_azimuth_elevation(
                (
                    3.0,
                    4.0,
                    5.0,
                )
            )
        )

        self.assertAlmostEqual(
            result.range_m,
            math.sqrt(
                50.0
            ),
        )

        self.assertAlmostEqual(
            result.azimuth_rad,
            math.atan2(
                4.0,
                3.0,
            ),
        )

        self.assertAlmostEqual(
            result.elevation_rad,
            math.atan2(
                5.0,
                5.0,
            ),
        )

    def test_04_undefined_azimuth_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            position_h0_to_range_azimuth_elevation(
                (
                    0.0,
                    0.0,
                    5.0,
                )
            )

    def test_05_circular_difference_wrap(self):
        difference = (
            circular_angle_difference(
                -math.pi
                +
                0.1,

                math.pi
                -
                0.1,
            )
        )

        self.assertAlmostEqual(
            difference,
            0.2,
            places=12,
        )


class TestBlock52Jacobian(
    unittest.TestCase
):

    def test_06_jacobian_matches_finite_difference(self):
        point = np.asarray(
            [
                12.0,
                3.0,
                1.5,
            ],
            dtype=np.float64,
        )

        analytic = (
            range_azimuth_elevation_jacobian(
                point
            )
        )

        epsilon = (
            1e-6
        )

        numeric = np.zeros(
            (
                3,
                3,
            ),
            dtype=np.float64,
        )

        def vector(
            value,
        ):
            result = (
                position_h0_to_range_azimuth_elevation(
                    value
                )
            )

            return np.asarray(
                [
                    result.range_m,
                    result.azimuth_rad,
                    result.elevation_rad,
                ],
                dtype=np.float64,
            )

        for dimension in range(
            3
        ):
            positive = point.copy()

            negative = point.copy()

            positive[
                dimension
            ] += epsilon

            negative[
                dimension
            ] -= epsilon

            numeric[
                :,
                dimension
            ] = (
                vector(
                    positive
                )
                -
                vector(
                    negative
                )
            ) / (
                2.0
                *
                epsilon
            )

        np.testing.assert_allclose(
            analytic,
            numeric,
            atol=2e-8,
            rtol=2e-8,
        )

    def test_07_jacobian_axis_case(self):
        jacobian = (
            range_azimuth_elevation_jacobian(
                (
                    10.0,
                    0.0,
                    0.0,
                )
            )
        )

        expected = np.asarray(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.1,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.1,
                ],
            ]
        )

        np.testing.assert_allclose(
            jacobian,
            expected,
            atol=1e-12,
            rtol=0.0,
        )


class TestBlock52GaussianPropagation(
    unittest.TestCase
):

    def test_08_zero_covariance_stays_zero(self):
        posterior = (
            linearized_position_gaussian_to_angular(
                GaussianPositionH0(
                    mean_h0_m=(
                        10.0,
                        0.0,
                        0.0,
                    ),

                    covariance_h0_m2=(
                        (
                            0.0,
                            0.0,
                            0.0,
                        ),
                        (
                            0.0,
                            0.0,
                            0.0,
                        ),
                        (
                            0.0,
                            0.0,
                            0.0,
                        ),
                    ),
                )
            )
        )

        np.testing.assert_allclose(
            posterior.covariance,
            np.zeros(
                (
                    3,
                    3,
                )
            ),
            atol=1e-15,
            rtol=0.0,
        )

    def test_09_axis_covariance_has_expected_scaling(self):
        posterior = (
            linearized_position_gaussian_to_angular(
                GaussianPositionH0(
                    mean_h0_m=(
                        10.0,
                        0.0,
                        0.0,
                    ),

                    covariance_h0_m2=(
                        (
                            4.0,
                            0.0,
                            0.0,
                        ),
                        (
                            0.0,
                            1.0,
                            0.0,
                        ),
                        (
                            0.0,
                            0.0,
                            0.25,
                        ),
                    ),
                )
            )
        )

        covariance = np.asarray(
            posterior.covariance
        )

        self.assertAlmostEqual(
            covariance[
                0,
                0
            ],
            4.0,
        )

        self.assertAlmostEqual(
            covariance[
                1,
                1
            ],
            0.01,
        )

        self.assertAlmostEqual(
            covariance[
                2,
                2
            ],
            0.0025,
        )

    def test_10_propagated_covariance_is_psd(self):
        posterior = (
            linearized_position_gaussian_to_angular(
                GaussianPositionH0(
                    mean_h0_m=(
                        20.0,
                        3.0,
                        1.0,
                    ),

                    covariance_h0_m2=(
                        (
                            2.0,
                            0.2,
                            0.1,
                        ),
                        (
                            0.2,
                            1.0,
                            0.05,
                        ),
                        (
                            0.1,
                            0.05,
                            0.5,
                        ),
                    ),
                )
            )
        )

        covariance = np.asarray(
            posterior.covariance
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

    def test_11_non_psd_input_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            linearized_position_gaussian_to_angular(
                GaussianPositionH0(
                    mean_h0_m=(
                        10.0,
                        0.0,
                        0.0,
                    ),

                    covariance_h0_m2=(
                        (
                            1.0,
                            0.0,
                            0.0,
                        ),
                        (
                            0.0,
                            -1.0,
                            0.0,
                        ),
                        (
                            0.0,
                            0.0,
                            1.0,
                        ),
                    ),
                )
            )

    def test_12_position_uncertainties_add_only_at_position_level(self):
        result = (
            combine_independent_position_uncertainties(
                trajectory_mean_h0_m=(
                    10.0,
                    2.0,
                    1.0,
                ),

                trajectory_covariance_h0_m2=(
                    (
                        1.0,
                        0.0,
                        0.0,
                    ),
                    (
                        0.0,
                        2.0,
                        0.0,
                    ),
                    (
                        0.0,
                        0.0,
                        3.0,
                    ),
                ),

                receiver_offset_mean_h0_m=(
                    2.0,
                    0.0,
                    0.5,
                ),

                receiver_placement_covariance_h0_m2=(
                    (
                        0.25,
                        0.0,
                        0.0,
                    ),
                    (
                        0.0,
                        0.5,
                        0.0,
                    ),
                    (
                        0.0,
                        0.0,
                        0.04,
                    ),
                ),
            )
        )

        np.testing.assert_allclose(
            result.mean_h0_m,
            (
                12.0,
                2.0,
                1.5,
            ),
            atol=1e-12,
            rtol=0.0,
        )

        np.testing.assert_allclose(
            result.covariance_h0_m2,
            np.diag(
                [
                    1.25,
                    2.5,
                    3.04,
                ]
            ),
            atol=1e-12,
            rtol=0.0,
        )

        self.assertIn(
            "Stage4_predictive",
            result.covariance_semantics,
        )

    def test_13_primary_and_crosscheck_methods_distinct(self):
        self.assertEqual(
            PRIMARY_STAGE52_METHOD,
            "deterministic_monte_carlo",
        )

        self.assertEqual(
            ANALYTIC_CROSSCHECK_METHOD,
            "first_order_Jacobian_Gaussian",
        )

        self.assertNotEqual(
            PRIMARY_STAGE52_METHOD,
            ANALYTIC_CROSSCHECK_METHOD,
        )

    def test_14_APIs_have_no_future_truth_or_evaluator_metadata(self):
        APIs = (
            position_h0_to_range_azimuth_elevation,
            range_azimuth_elevation_jacobian,
            linearized_position_gaussian_to_angular,
            combine_independent_position_uncertainties,
        )

        forbidden = (
            "future_truth",
            "ground_truth",
            "tracks_to_predict",
            "oracle",
            "objects_of_interest",
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

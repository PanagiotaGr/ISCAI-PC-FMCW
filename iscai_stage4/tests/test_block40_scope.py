import unittest

from iscai_stage4.contracts import (
    CONFIDENCE_LEVELS,
    GaussianHorizonPosterior,
    PRIMARY_HORIZONS_S,
    Stage4ScopeContract,
    TrajectoryGaussianPosterior,
)


class TestStage4Scope(unittest.TestCase):

    def test_primary_horizons(self):
        self.assertEqual(
            PRIMARY_HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

    def test_confidence_levels(self):
        self.assertEqual(
            CONFIDENCE_LEVELS,
            (
                0.50,
                0.80,
                0.90,
                0.95,
                0.99,
            ),
        )

    def test_measurement_covariance_is_predictor_input(self):
        contract = Stage4ScopeContract()

        self.assertTrue(
            contract.measurement_covariance_is_input
        )

        self.assertTrue(
            contract.measurement_uncertainty_distinct_from_predictive
        )

    def test_no_future_or_oracle_realistic_inputs(self):
        contract = Stage4ScopeContract()

        self.assertFalse(
            contract.future_states_are_model_input
        )
        self.assertFalse(
            contract.future_validity_is_model_input
        )
        self.assertFalse(
            contract.future_track_duration_is_model_input
        )
        self.assertFalse(
            contract.future_lidar_is_model_input
        )
        self.assertFalse(
            contract.tracks_to_predict_is_model_input
        )
        self.assertFalse(
            contract.objects_of_interest_is_model_input
        )
        self.assertFalse(
            contract.annotated_velocity_is_realistic_input
        )
        self.assertFalse(
            contract.perfect_track_id_is_numeric_feature
        )

    def test_stage5_plus_scope_excluded(self):
        contract = Stage4ScopeContract()

        self.assertFalse(
            contract.receiver_angular_posterior_in_stage4
        )
        self.assertFalse(
            contract.beam_controller_in_stage4
        )
        self.assertFalse(
            contract.optical_link_in_stage4
        )
        self.assertFalse(
            contract.adb_in_stage4
        )
        self.assertFalse(
            contract.deepsense_in_stage4
        )

    def test_valid_spd_gaussian_contract(self):
        item = GaussianHorizonPosterior(
            horizon_s=0.1,
            mean_H0_m=(
                1.0,
                2.0,
                0.5,
            ),
            covariance_H0_m2=(
                (1.0, 0.1, 0.0),
                (0.1, 2.0, 0.0),
                (0.0, 0.0, 0.5),
            ),
        )

        self.assertEqual(
            item.horizon_s,
            0.1,
        )

    def test_non_symmetric_covariance_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GaussianHorizonPosterior(
                horizon_s=0.1,
                mean_H0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                covariance_H0_m2=(
                    (1.0, 0.3, 0.0),
                    (0.1, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )

    def test_non_positive_definite_covariance_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            GaussianHorizonPosterior(
                horizon_s=0.1,
                mean_H0_m=(
                    0.0,
                    0.0,
                    0.0,
                ),
                covariance_H0_m2=(
                    (1.0, 2.0, 0.0),
                    (2.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )

    def test_full_trajectory_contract(self):
        covariance = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

        posterior = (
            TrajectoryGaussianPosterior(
                prediction_id="demo",
                horizons=tuple(
                    GaussianHorizonPosterior(
                        horizon_s=h,
                        mean_H0_m=(
                            h,
                            0.0,
                            0.0,
                        ),
                        covariance_H0_m2=(
                            covariance
                        ),
                    )
                    for h
                    in PRIMARY_HORIZONS_S
                ),
            )
        )

        self.assertEqual(
            len(posterior.horizons),
            4,
        )


if __name__ == "__main__":
    unittest.main()

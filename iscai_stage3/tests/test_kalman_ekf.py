import math
import unittest

import numpy as np

from iscai_stage2.observations.contracts import (
    MeasurementCovariance,
    PcfmcwLikeObservation,
)

from iscai_stage3.baselines.kalman_ekf import (
    initialize_ekf_from_measurement,
    measurement_function,
    measurement_jacobian,
    predict_ekf_cv,
    update_ekf_with_stage2_measurement,
)


def observation(
    *,
    timestamp_s=0.0,
    range_m=10.0,
    radial_velocity_mps=1.0,
    azimuth_rad=0.0,
    elevation_rad=0.0,
):
    return PcfmcwLikeObservation(
        scenario_id="test",
        track_id="metadata_only",
        object_class="TYPE_VEHICLE",

        time_index=int(
            round(
                timestamp_s * 10.0
            )
        ),

        timestamp_s=timestamp_s,

        range_m=range_m,
        radial_velocity_mps=(
            radial_velocity_mps
        ),
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,

        covariance=MeasurementCovariance(
            matrix=(
                (0.01, 0.0, 0.0, 0.0),
                (0.0, 0.04, 0.0, 0.0),
                (0.0, 0.0, 0.001, 0.0),
                (0.0, 0.0, 0.0, 0.001),
            )
        ),

        measurement_valid=True,
    )


class TestKalmanEKF(unittest.TestCase):

    def test_measurement_function_forward(self):

        z = measurement_function(
            (
                10.0, 0.0, 0.0,
                2.0, 0.0, 0.0,
            )
        )

        self.assertAlmostEqual(
            z[0],
            10.0,
        )

        self.assertAlmostEqual(
            z[1],
            2.0,
        )

        self.assertAlmostEqual(
            z[2],
            0.0,
        )

        self.assertAlmostEqual(
            z[3],
            0.0,
        )


    def test_jacobian_shape(self):

        H = measurement_jacobian(
            (
                10.0, 1.0, 0.5,
                2.0, 0.1, 0.0,
            )
        )

        self.assertEqual(
            H.shape,
            (4, 6),
        )


    def test_initialize(self):

        state = (
            initialize_ekf_from_measurement(
                observation()
            )
        )

        self.assertAlmostEqual(
            state.mean[0],
            10.0,
        )

        self.assertAlmostEqual(
            state.mean[3],
            1.0,
        )


    def test_prediction_advances_time(self):

        state = (
            initialize_ekf_from_measurement(
                observation()
            )
        )

        predicted = predict_ekf_cv(
            state,
            timestamp_s=0.1,
        )

        self.assertAlmostEqual(
            predicted.timestamp_s,
            0.1,
        )

        self.assertGreater(
            predicted.mean[0],
            state.mean[0],
        )


    def test_update_uses_measurement(self):

        state = (
            initialize_ekf_from_measurement(
                observation()
            )
        )

        predicted = predict_ekf_cv(
            state,
            timestamp_s=0.1,
        )

        measured = observation(
            timestamp_s=0.1,
            range_m=10.2,
            radial_velocity_mps=1.0,
        )

        before_error = abs(
            predicted.mean[0]
            - 10.2
        )

        updated = (
            update_ekf_with_stage2_measurement(
                predicted=predicted,
                observation=measured,
            )
        )

        after_error = abs(
            updated.mean[0]
            - 10.2
        )

        self.assertLess(
            after_error,
            before_error,
        )


    def test_posterior_covariance_symmetric(self):

        state = (
            initialize_ekf_from_measurement(
                observation()
            )
        )

        predicted = predict_ekf_cv(
            state,
            timestamp_s=0.1,
        )

        updated = (
            update_ekf_with_stage2_measurement(
                predicted=predicted,
                observation=observation(
                    timestamp_s=0.1,
                    range_m=10.1,
                ),
            )
        )

        P = np.asarray(
            updated.covariance
        )

        self.assertTrue(
            np.allclose(
                P,
                P.T,
                atol=1e-10,
            )
        )


if __name__ == "__main__":
    unittest.main()

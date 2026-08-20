import unittest

from iscai_stage2.observations.contracts import (
    MEASURED_FMCW,
    MEASUREMENT_COMPONENTS,
    OBSERVATION_FRAME,
    MeasurementCovariance,
    PcfmcwLikeObservation,
)


def diag_cov():
    return MeasurementCovariance(
        matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 2.0, 0.0, 0.0),
            (0.0, 0.0, 0.01, 0.0),
            (0.0, 0.0, 0.0, 0.02),
        )
    )


class TestObservationContracts(unittest.TestCase):

    def test_measurement_order_is_frozen(self):
        self.assertEqual(
            MEASUREMENT_COMPONENTS,
            (
                "range_m",
                "radial_velocity_mps",
                "azimuth_rad",
                "elevation_rad",
            ),
        )

    def test_frame_is_dynamic_headlamp(self):
        self.assertEqual(
            OBSERVATION_FRAME,
            "Ht",
        )

    def test_womd_is_not_measured_fmcw(self):
        self.assertFalse(MEASURED_FMCW)

    def test_valid_covariance(self):
        self.assertIsInstance(
            diag_cov(),
            MeasurementCovariance,
        )

    def test_asymmetric_covariance_rejected(self):
        with self.assertRaises(ValueError):
            MeasurementCovariance(
                matrix=(
                    (1.0, 1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )

    def test_non_psd_covariance_rejected(self):
        # Positive diagonal alone is not sufficient:
        # [[1,2],[2,1]] has a negative eigenvalue.
        with self.assertRaises(ValueError):
            MeasurementCovariance(
                matrix=(
                    (1.0, 2.0, 0.0, 0.0),
                    (2.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )

    def test_observation_contract(self):
        obs = PcfmcwLikeObservation(
            scenario_id="s",
            track_id="t",
            object_class="TYPE_VEHICLE",
            time_index=10,
            timestamp_s=1.0,
            range_m=25.0,
            radial_velocity_mps=-3.0,
            azimuth_rad=0.1,
            elevation_rad=0.02,
            covariance=diag_cov(),
            measurement_valid=True,
        )

        self.assertEqual(
            obs.measurement_vector(),
            (25.0, -3.0, 0.1, 0.02),
        )

        self.assertFalse(obs.measured_fmcw)
        self.assertEqual(obs.frame_name, "Ht")


if __name__ == "__main__":
    unittest.main()

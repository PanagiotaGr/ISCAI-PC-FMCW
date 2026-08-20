import unittest

from iscai_stage1.geometry.rigid import (
    RigidTransform,
)
from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
    clean_crlb_conditioned_observation,
)
from iscai_stage2.observations.ideal import (
    ideal_causal_observable,
)


IDENTITY = RigidTransform(
    rotation=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation=(0.0, 0.0, 0.0),
)


CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,
    azimuth_std_rad=0.01,
    elevation_std_rad=0.02,
)


def valid_ideal():
    return ideal_causal_observable(
        time_index=1,
        timestamp_s=0.1,

        actor_position_W_m=(
            10.0, 1.0, 0.5
        ),
        actor_position_valid=True,

        actor_velocity_W_mps=(
            -2.0, 0.0, 0.0
        ),
        actor_velocity_valid=True,

        headlamp_velocity_W_mps=(
            0.0, 0.0, 0.0
        ),

        T_Ht_from_W=IDENTITY,
    )


class TestCleanMeasurement(unittest.TestCase):

    def test_valid_ideal_produces_measurement(self):
        ideal = valid_ideal()

        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=ideal,
                config=CONFIG,
            )
        )

        self.assertTrue(
            record.measurement_valid
        )

        self.assertIsNotNone(
            record.measurement
        )

    def test_clean_mean_equals_ideal(self):
        ideal = valid_ideal()

        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=ideal,
                config=CONFIG,
            )
        )

        measurement = record.measurement

        self.assertIsNotNone(
            measurement
        )

        self.assertEqual(
            measurement.range_m,
            ideal.range_m,
        )

        self.assertEqual(
            measurement.radial_velocity_mps,
            ideal.radial_velocity_mps,
        )

        self.assertEqual(
            measurement.azimuth_rad,
            ideal.azimuth_rad,
        )

        self.assertEqual(
            measurement.elevation_rad,
            ideal.elevation_rad,
        )

    def test_fixed_snr_attached(self):
        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=valid_ideal(),
                config=CONFIG,
            )
        )

        self.assertIsNotNone(
            record.sensing_snr
        )

        self.assertAlmostEqual(
            record.sensing_snr.snr_db,
            20.0,
        )

    def test_crlb_covariance_attached(self):
        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=valid_ideal(),
                config=CONFIG,
            )
        )

        covariance = (
            record.measurement
            .covariance.matrix
        )

        self.assertGreater(
            covariance[0][0],
            0.0,
        )

        self.assertGreater(
            covariance[1][1],
            0.0,
        )

        self.assertAlmostEqual(
            covariance[2][2],
            0.01 ** 2,
        )

        self.assertAlmostEqual(
            covariance[3][3],
            0.02 ** 2,
        )

    def test_no_noise_applied(self):
        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=valid_ideal(),
                config=CONFIG,
            )
        )

        self.assertFalse(
            record.sensor_noise_applied
        )

        self.assertFalse(
            record.missed_detection_applied
        )

        self.assertFalse(
            record.false_alarm_applied
        )

    def test_invalid_geometry_is_retained(self):
        ideal = ideal_causal_observable(
            time_index=0,
            timestamp_s=0.0,

            actor_position_W_m=(
                0.0, 0.0, 0.0
            ),
            actor_position_valid=False,

            actor_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            actor_velocity_valid=False,

            headlamp_velocity_W_mps=None,

            T_Ht_from_W=IDENTITY,
        )

        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=ideal,
                config=CONFIG,
            )
        )

        self.assertFalse(
            record.measurement_valid
        )

        self.assertIsNone(
            record.measurement
        )

        self.assertEqual(
            record.invalid_reason,
            "invalid_geometry",
        )

    def test_invalid_vr_does_not_fake_measurement(self):
        ideal = ideal_causal_observable(
            time_index=1,
            timestamp_s=0.1,

            actor_position_W_m=(
                10.0, 0.0, 0.0
            ),
            actor_position_valid=True,

            actor_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),
            actor_velocity_valid=False,

            headlamp_velocity_W_mps=(
                0.0, 0.0, 0.0
            ),

            T_Ht_from_W=IDENTITY,
        )

        record = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=ideal,
                config=CONFIG,
            )
        )

        self.assertFalse(
            record.measurement_valid
        )

        self.assertEqual(
            record.invalid_reason,
            "invalid_radial_velocity",
        )


if __name__ == "__main__":
    unittest.main()

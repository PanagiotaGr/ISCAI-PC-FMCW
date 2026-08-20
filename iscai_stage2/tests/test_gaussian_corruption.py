import math
import unittest

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
    clean_crlb_conditioned_observation,
)
from iscai_stage2.observations.gaussian_corruption import (
    GaussianCorruptionConfig,
    deterministic_standard_normal,
    gaussian_corrupt_clean_record,
)
from iscai_stage2.observations.ideal import (
    ideal_causal_observable,
)
from iscai_stage1.geometry.rigid import (
    RigidTransform,
)


IDENTITY = RigidTransform(
    rotation=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation=(0.0, 0.0, 0.0),
)


def clean_record():
    ideal = ideal_causal_observable(
        time_index=5,
        timestamp_s=0.5,

        actor_position_W_m=(
            25.0, 2.0, 1.0
        ),
        actor_position_valid=True,

        actor_velocity_W_mps=(
            -3.0, 0.5, 0.0
        ),
        actor_velocity_valid=True,

        headlamp_velocity_W_mps=(
            0.0, 0.0, 0.0
        ),

        T_Ht_from_W=IDENTITY,
    )

    return clean_crlb_conditioned_observation(
        scenario_id="scenario",
        track_id="track",
        object_class="TYPE_VEHICLE",
        ideal=ideal,

        config=CleanObservationConfig(
            sensing_snr_db=20.0,
            azimuth_std_rad=0.01,
            elevation_std_rad=0.02,
        ),
    )


class TestGaussianCorruption(unittest.TestCase):

    def test_standard_normal_is_deterministic(self):
        a = deterministic_standard_normal(
            seed=123,
            scenario_id="s",
            track_id="t",
            time_index=4,
            component="range",
        )

        b = deterministic_standard_normal(
            seed=123,
            scenario_id="s",
            track_id="t",
            time_index=4,
            component="range",
        )

        self.assertEqual(a, b)

    def test_seed_changes_noise(self):
        a = deterministic_standard_normal(
            seed=1,
            scenario_id="s",
            track_id="t",
            time_index=4,
            component="range",
        )

        b = deterministic_standard_normal(
            seed=2,
            scenario_id="s",
            track_id="t",
            time_index=4,
            component="range",
        )

        self.assertNotEqual(a, b)

    def test_corruption_is_deterministic(self):
        clean = clean_record()

        a = gaussian_corrupt_clean_record(
            scenario_id="scenario",
            track_id="track",
            object_class="TYPE_VEHICLE",
            clean=clean,
            config=GaussianCorruptionConfig(
                seed=20260810
            ),
        )

        b = gaussian_corrupt_clean_record(
            scenario_id="scenario",
            track_id="track",
            object_class="TYPE_VEHICLE",
            clean=clean,
            config=GaussianCorruptionConfig(
                seed=20260810
            ),
        )

        self.assertEqual(a, b)

    def test_noise_is_actually_applied(self):
        clean = clean_record()

        noisy = gaussian_corrupt_clean_record(
            scenario_id="scenario",
            track_id="track",
            object_class="TYPE_VEHICLE",
            clean=clean,
            config=GaussianCorruptionConfig(
                seed=20260810
            ),
        )

        self.assertTrue(
            noisy.measurement_valid
        )

        self.assertIsNotNone(
            noisy.noise_vector
        )

        self.assertTrue(
            any(
                value != 0.0
                for value in noisy.noise_vector
            )
        )

    def test_covariance_is_preserved(self):
        clean = clean_record()

        noisy = gaussian_corrupt_clean_record(
            scenario_id="scenario",
            track_id="track",
            object_class="TYPE_VEHICLE",
            clean=clean,
            config=GaussianCorruptionConfig(
                seed=123
            ),
        )

        self.assertEqual(
            noisy.measurement.covariance,
            clean.measurement.covariance,
        )

    def test_invalid_clean_record_stays_invalid(self):
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

        clean = (
            clean_crlb_conditioned_observation(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                ideal=ideal,
                config=CleanObservationConfig(
                    sensing_snr_db=20.0,
                    azimuth_std_rad=0.01,
                    elevation_std_rad=0.01,
                ),
            )
        )

        noisy = gaussian_corrupt_clean_record(
            scenario_id="s",
            track_id="t",
            object_class="TYPE_VEHICLE",
            clean=clean,
            config=GaussianCorruptionConfig(
                seed=1
            ),
        )

        self.assertFalse(
            noisy.measurement_valid
        )

        self.assertIsNone(
            noisy.measurement
        )

    def test_gaussian_generator_empirical_rmse(self):
        n = 20_000

        values = [
            deterministic_standard_normal(
                seed=777,
                scenario_id="mc",
                track_id=str(i),
                time_index=i,
                component="range",
            )
            for i in range(n)
        ]

        mean = sum(values) / n

        rmse = math.sqrt(
            sum(
                value * value
                for value in values
            ) / n
        )

        self.assertLess(
            abs(mean),
            0.03,
        )

        # Unit Gaussian -> RMSE approximately 1.
        self.assertLess(
            abs(rmse - 1.0),
            0.03,
        )

    def test_no_detection_logic_in_3a(self):
        noisy = gaussian_corrupt_clean_record(
            scenario_id="scenario",
            track_id="track",
            object_class="TYPE_VEHICLE",
            clean=clean_record(),
            config=GaussianCorruptionConfig(
                seed=5
            ),
        )

        self.assertTrue(
            noisy.sensor_noise_applied
        )

        self.assertFalse(
            noisy.missed_detection_applied
        )

        self.assertFalse(
            noisy.false_alarm_applied
        )


if __name__ == "__main__":
    unittest.main()

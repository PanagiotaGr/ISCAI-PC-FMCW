import unittest

from iscai_stage1.geometry.rigid import (
    RigidTransform,
)
from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
    clean_crlb_conditioned_observation,
)
from iscai_stage2.observations.detection import (
    DETECTION_MODEL_SEMANTICS,
    DetectionProbabilityConfig,
    apply_missed_detection,
    detection_probability,
    deterministic_detection_uniform,
)
from iscai_stage2.observations.gaussian_corruption import (
    GaussianCorruptionConfig,
    gaussian_corrupt_clean_record,
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


PD_CONFIG = DetectionProbabilityConfig(
    snr_midpoint_db=10.0,
    transition_width_db=2.0,
)


def valid_gaussian_record():
    ideal = ideal_causal_observable(
        time_index=5,
        timestamp_s=0.5,

        actor_position_W_m=(
            20.0, 1.0, 0.2
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

    clean = (
        clean_crlb_conditioned_observation(
            scenario_id="scenario",
            track_id="track",
            object_class="TYPE_VEHICLE",
            ideal=ideal,

            config=CleanObservationConfig(
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.01,
            ),
        )
    )

    return gaussian_corrupt_clean_record(
        scenario_id="scenario",
        track_id="track",
        object_class="TYPE_VEHICLE",
        clean=clean,
        config=GaussianCorruptionConfig(
            seed=123
        ),
    )


class TestDetection(unittest.TestCase):

    def test_midpoint_probability(self):
        pd = detection_probability(
            sensing_snr_db=10.0,
            config=PD_CONFIG,
        )

        self.assertAlmostEqual(
            pd,
            0.5,
            places=15,
        )

    def test_probability_increases_with_snr(self):
        low = detection_probability(
            sensing_snr_db=0.0,
            config=PD_CONFIG,
        )

        high = detection_probability(
            sensing_snr_db=20.0,
            config=PD_CONFIG,
        )

        self.assertLess(
            low,
            high,
        )

    def test_floor_and_ceiling_respected(self):
        config = DetectionProbabilityConfig(
            snr_midpoint_db=10.0,
            transition_width_db=1.0,
            pd_floor=0.1,
            pd_ceiling=0.9,
        )

        very_low = detection_probability(
            sensing_snr_db=-100.0,
            config=config,
        )

        very_high = detection_probability(
            sensing_snr_db=100.0,
            config=config,
        )

        self.assertGreaterEqual(
            very_low,
            0.1,
        )

        self.assertLessEqual(
            very_high,
            0.9,
        )

    def test_detection_uniform_is_deterministic(self):
        a = deterministic_detection_uniform(
            seed=7,
            scenario_id="s",
            track_id="t",
            time_index=3,
        )

        b = deterministic_detection_uniform(
            seed=7,
            scenario_id="s",
            track_id="t",
            time_index=3,
        )

        self.assertEqual(a, b)

    def test_detection_decision_is_deterministic(self):
        gaussian = valid_gaussian_record()

        a = apply_missed_detection(
            scenario_id="scenario",
            track_id="track",
            gaussian=gaussian,
            sensing_snr_db=10.0,
            probability_config=PD_CONFIG,
            seed=55,
        )

        b = apply_missed_detection(
            scenario_id="scenario",
            track_id="track",
            gaussian=gaussian,
            sensing_snr_db=10.0,
            probability_config=PD_CONFIG,
            seed=55,
        )

        self.assertEqual(a, b)

    def test_pd_zero_always_misses(self):
        gaussian = valid_gaussian_record()

        result = apply_missed_detection(
            scenario_id="scenario",
            track_id="track",
            gaussian=gaussian,
            sensing_snr_db=20.0,

            probability_config=(
                DetectionProbabilityConfig(
                    snr_midpoint_db=0.0,
                    transition_width_db=1.0,
                    pd_floor=0.0,
                    pd_ceiling=0.0,
                )
            ),

            seed=1,
        )

        self.assertFalse(
            result.measurement_valid
        )

        self.assertTrue(
            result.missed_detection_applied
        )

        self.assertEqual(
            result.invalid_reason,
            "missed_detection",
        )

        self.assertIsNone(
            result.measurement
        )

    def test_pd_one_never_misses(self):
        gaussian = valid_gaussian_record()

        result = apply_missed_detection(
            scenario_id="scenario",
            track_id="track",
            gaussian=gaussian,
            sensing_snr_db=-100.0,

            probability_config=(
                DetectionProbabilityConfig(
                    snr_midpoint_db=0.0,
                    transition_width_db=1.0,
                    pd_floor=1.0,
                    pd_ceiling=1.0,
                )
            ),

            seed=1,
        )

        self.assertTrue(
            result.measurement_valid
        )

        self.assertFalse(
            result.missed_detection_applied
        )

        self.assertIsNotNone(
            result.measurement
        )

    def test_empirical_detection_fraction_matches_pd(self):
        pd_target = 0.35

        config = DetectionProbabilityConfig(
            snr_midpoint_db=0.0,
            transition_width_db=1.0,
            pd_floor=pd_target,
            pd_ceiling=pd_target,
        )

        n = 20_000

        detected = 0

        for i in range(n):
            uniform = (
                deterministic_detection_uniform(
                    seed=999,
                    scenario_id="mc",
                    track_id=str(i),
                    time_index=i,
                )
            )

            pd = detection_probability(
                sensing_snr_db=0.0,
                config=config,
            )

            if uniform < pd:
                detected += 1

        fraction = detected / n

        self.assertLess(
            abs(
                fraction - pd_target
            ),
            0.015,
        )

    def test_invalid_upstream_not_relabelled_as_miss(self):
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

        gaussian = (
            gaussian_corrupt_clean_record(
                scenario_id="s",
                track_id="t",
                object_class="TYPE_VEHICLE",
                clean=clean,
                config=GaussianCorruptionConfig(
                    seed=1
                ),
            )
        )

        result = apply_missed_detection(
            scenario_id="s",
            track_id="t",
            gaussian=gaussian,
            sensing_snr_db=0.0,
            probability_config=PD_CONFIG,
            seed=1,
        )

        self.assertFalse(
            result.upstream_measurement_valid
        )

        self.assertFalse(
            result.missed_detection_applied
        )

        self.assertEqual(
            result.invalid_reason,
            "invalid_geometry",
        )

    def test_model_is_explicit_stage2_assumption(self):
        result = apply_missed_detection(
            scenario_id="scenario",
            track_id="track",
            gaussian=valid_gaussian_record(),
            sensing_snr_db=10.0,
            probability_config=PD_CONFIG,
            seed=42,
        )

        self.assertEqual(
            result.detection_model_semantics,
            DETECTION_MODEL_SEMANTICS,
        )


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
    clean_crlb_conditioned_observation,
)
from iscai_stage2.observations.clean_scene import (
    ActorCleanObservationSeries,
    CleanObservationScene,
)
from iscai_stage2.observations.degraded_scene import (
    DegradedObservationConfig,
    build_degraded_observation_scene,
    degraded_algorithm_sha256,
    degraded_truth_sha256,
)
from iscai_stage2.observations.detection import (
    DetectionProbabilityConfig,
)
from iscai_stage2.observations.detection_set import (
    FALSE_ALARM,
    TRUE_DETECTION,
    FalseAlarmConfig,
)
from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


def clean_record(
    time_index: int,
):
    ideal = IdealCausalObservable(
        time_index=time_index,
        timestamp_s=0.1 * time_index,

        actor_position_Ht_m=(
            20.0,
            1.0,
            0.0,
        ),

        range_m=20.0,
        azimuth_rad=0.05,
        elevation_rad=0.0,

        radial_velocity_mps=-2.0,

        geometry_valid=True,
        radial_velocity_valid=True,
    )

    return clean_crlb_conditioned_observation(
        scenario_id="scenario",
        track_id="oracle-track",
        object_class="TYPE_VEHICLE",
        ideal=ideal,

        config=CleanObservationConfig(
            sensing_snr_db=20.0,
            azimuth_std_rad=0.01,
            elevation_std_rad=0.01,
        ),
    )


def clean_scene():
    records = tuple(
        clean_record(i)
        for i in range(3)
    )

    return CleanObservationScene(
        scenario_id="scenario",
        anchor_index=2,

        actors=(
            ActorCleanObservationSeries(
                scenario_id="scenario",
                track_index=1,
                track_id="oracle-track",
                object_class="TYPE_VEHICLE",
                records=records,
            ),
        ),
    )


def no_false_alarms():
    return FalseAlarmConfig(
        seed=3,

        mean_false_alarms_per_frame=0.0,

        range_min_m=1.0,
        range_max_m=100.0,

        radial_velocity_min_mps=-30.0,
        radial_velocity_max_mps=30.0,

        azimuth_min_rad=-math.pi,
        azimuth_max_rad=math.pi,

        elevation_min_rad=-0.3,
        elevation_max_rad=0.3,

        range_std_m=1.0,
        radial_velocity_std_mps=2.0,
        azimuth_std_rad=0.05,
        elevation_std_rad=0.05,
    )


def config(
    *,
    pd: float,
    gaussian_seed: int = 10,
):
    return DegradedObservationConfig(
        gaussian_seed=gaussian_seed,
        detection_seed=20,

        detection_probability=(
            DetectionProbabilityConfig(
                snr_midpoint_db=0.0,
                transition_width_db=1.0,
                pd_floor=pd,
                pd_ceiling=pd,
            )
        ),

        false_alarms=(
            no_false_alarms()
        ),
    )


class TestDegradedScene(unittest.TestCase):

    def test_scene_has_expected_frames(self):
        scene = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(pd=1.0),
            )
        )

        self.assertEqual(
            len(scene.frames),
            3,
        )

        self.assertEqual(
            len(scene.truth_sidecars),
            3,
        )

    def test_pd_one_keeps_true_measurements(self):
        scene = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(pd=1.0),
            )
        )

        self.assertTrue(
            all(
                len(frame.detections)
                == 1
                for frame in scene.frames
            )
        )

        self.assertTrue(
            all(
                sidecar.entries[0]
                .source_type
                == TRUE_DETECTION
                for sidecar
                in scene.truth_sidecars
            )
        )

    def test_pd_zero_removes_true_measurements(self):
        scene = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(pd=0.0),
            )
        )

        self.assertTrue(
            all(
                len(frame.detections)
                == 0
                for frame in scene.frames
            )
        )

    def test_algorithm_frames_do_not_expose_truth(self):
        scene = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(pd=1.0),
            )
        )

        frame_text = repr(
            scene.frames
        )

        self.assertNotIn(
            "oracle-track",
            frame_text,
        )

        self.assertNotIn(
            "TYPE_VEHICLE",
            frame_text,
        )

        self.assertNotIn(
            TRUE_DETECTION,
            frame_text,
        )

        self.assertNotIn(
            FALSE_ALARM,
            frame_text,
        )

    def test_scene_is_reproducible(self):
        a = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(pd=1.0),
            )
        )

        b = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(pd=1.0),
            )
        )

        self.assertEqual(
            degraded_algorithm_sha256(a),
            degraded_algorithm_sha256(b),
        )

        self.assertEqual(
            degraded_truth_sha256(a),
            degraded_truth_sha256(b),
        )

    def test_gaussian_seed_changes_algorithm_hash(self):
        a = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(
                    pd=1.0,
                    gaussian_seed=10,
                ),
            )
        )

        b = (
            build_degraded_observation_scene(
                clean_scene=clean_scene(),
                config=config(
                    pd=1.0,
                    gaussian_seed=11,
                ),
            )
        )

        self.assertNotEqual(
            degraded_algorithm_sha256(a),
            degraded_algorithm_sha256(b),
        )


if __name__ == "__main__":
    unittest.main()

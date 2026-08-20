from __future__ import annotations

import json
import math
from pathlib import Path

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
)
from iscai_stage2.observations.clean_scene import (
    build_clean_observation_scene,
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
from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage2/"
    "reports/block3d_degraded_scene_smoke.json"
)


CLEAN_CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,

    azimuth_std_rad=math.radians(
        1.0
    ),
    elevation_std_rad=math.radians(
        1.0
    ),
)


DEGRADED_CONFIG = DegradedObservationConfig(
    gaussian_seed=20260810,
    detection_seed=20260811,

    # Controlled degraded smoke:
    # at the current fixed 20-dB clean baseline,
    # P_D = 0.5 exactly.
    detection_probability=(
        DetectionProbabilityConfig(
            snr_midpoint_db=20.0,
            transition_width_db=2.0,
            pd_floor=0.05,
            pd_ceiling=0.95,
        )
    ),

    false_alarms=(
        FalseAlarmConfig(
            seed=20260812,

            mean_false_alarms_per_frame=2.0,

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

            azimuth_std_rad=math.radians(
                5.0
            ),
            elevation_std_rad=math.radians(
                5.0
            ),
        )
    ),
)


def build(raw):
    adapted = (
        adapt_causal_womd_scenario(
            raw
        )
    )

    ideal = (
        build_real_ideal_observation_scene(
            raw_scenario=raw,
            adapted=adapted,
            include_sdc=False,
        )
    )

    clean = (
        build_clean_observation_scene(
            ideal_scene=ideal,
            config=CLEAN_CONFIG,
        )
    )

    degraded = (
        build_degraded_observation_scene(
            clean_scene=clean,
            config=DEGRADED_CONFIG,
        )
    )

    return clean, degraded


scenario = read_first_scenario(
    MOTION
)

if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected pilot scenario."
    )


clean, degraded = build(
    scenario
)


if len(degraded.frames) != 11:
    raise RuntimeError(
        "Expected 11 causal detection frames."
    )


# ------------------------------------------------------------
# Frozen upstream-valid count from Block 2C
# ------------------------------------------------------------

upstream_valid = sum(
    record.measurement_valid
    for actor in clean.actors
    for record in actor.records
)

if upstream_valid != 86:
    raise RuntimeError(
        "Expected frozen clean-valid count 86, "
        f"got {upstream_valid}."
    )


# ------------------------------------------------------------
# Count evaluator truth
# ------------------------------------------------------------

true_detections = 0
false_alarms = 0

for sidecar in (
    degraded.truth_sidecars
):
    for entry in sidecar.entries:
        if (
            entry.source_type
            == TRUE_DETECTION
        ):
            true_detections += 1

        elif (
            entry.source_type
            == FALSE_ALARM
        ):
            false_alarms += 1

        else:
            raise RuntimeError(
                "Unknown evaluator truth source."
            )


missed_detections = (
    upstream_valid
    - true_detections
)


if not (
    0
    < missed_detections
    < upstream_valid
):
    raise RuntimeError(
        "Controlled degraded mode did not "
        "produce both detections and misses."
    )


if false_alarms <= 0:
    raise RuntimeError(
        "Controlled degraded mode produced "
        "no false alarms."
    )


algorithm_detection_count = sum(
    len(frame.detections)
    for frame in degraded.frames
)

if algorithm_detection_count != (
    true_detections + false_alarms
):
    raise RuntimeError(
        "Algorithm detection count mismatch."
    )


# ------------------------------------------------------------
# Algorithm-facing leakage gate
# ------------------------------------------------------------

algorithm_json = json.dumps(
    [
        {
            "scenario_id": frame.scenario_id,
            "time_index": frame.time_index,
            "timestamp_s": frame.timestamp_s,

            "detections": [
                {
                    "detection_key":
                        detection.detection_key,

                    "range_m":
                        detection.range_m,

                    "radial_velocity_mps":
                        detection.radial_velocity_mps,

                    "azimuth_rad":
                        detection.azimuth_rad,

                    "elevation_rad":
                        detection.elevation_rad,

                    "covariance":
                        detection.covariance.matrix,
                }
                for detection
                in frame.detections
            ],
        }
        for frame in degraded.frames
    ],
    sort_keys=True,
)


for forbidden in (
    "track_id",
    "object_class",
    "TYPE_VEHICLE",
    "true_detection",
    "false_alarm",
):
    if forbidden in algorithm_json:
        raise RuntimeError(
            "Oracle/truth information leaked into "
            f"algorithm frame: {forbidden}"
        )


# ------------------------------------------------------------
# Deterministic reproducibility
# ------------------------------------------------------------

algorithm_hash_1 = (
    degraded_algorithm_sha256(
        degraded
    )
)

truth_hash_1 = (
    degraded_truth_sha256(
        degraded
    )
)

_, degraded_repeat = build(
    scenario
)

algorithm_hash_2 = (
    degraded_algorithm_sha256(
        degraded_repeat
    )
)

truth_hash_2 = (
    degraded_truth_sha256(
        degraded_repeat
    )
)

if algorithm_hash_1 != algorithm_hash_2:
    raise RuntimeError(
        "Degraded algorithm frames are not "
        "reproducible."
    )

if truth_hash_1 != truth_hash_2:
    raise RuntimeError(
        "Degraded evaluator truth is not "
        "reproducible."
    )


# ------------------------------------------------------------
# Future mutation invariance
# ------------------------------------------------------------

mutated = scenario_pb2.Scenario()
mutated.CopyFrom(
    scenario
)

future_index = (
    int(mutated.current_time_index)
    + 1
)

future = (
    mutated.tracks[0]
    .states[future_index]
)

future.center_x += 9000.0
future.center_y -= 8000.0

future.velocity_x += 700.0
future.velocity_y -= 600.0

future.heading += 1.0
future.valid = not future.valid


_, degraded_mutated = build(
    mutated
)

if (
    degraded_algorithm_sha256(
        degraded_mutated
    )
    != algorithm_hash_1
):
    raise RuntimeError(
        "Future mutation leaked into algorithm "
        "detection frames."
    )

if (
    degraded_truth_sha256(
        degraded_mutated
    )
    != truth_hash_1
):
    raise RuntimeError(
        "Future mutation leaked into evaluator "
        "truth sidecars."
    )


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

report = {
    "status": "PASS",

    "block": (
        "stage2_block3d_real_degraded_scene"
    ),

    "scenario_id": SID,

    "causal_frames": 11,

    "clean_upstream_valid": (
        upstream_valid
    ),

    "true_detections": (
        true_detections
    ),

    "missed_detections": (
        missed_detections
    ),

    "false_alarms": (
        false_alarms
    ),

    "algorithm_detection_count": (
        algorithm_detection_count
    ),

    "gaussian_noise": {
        "enabled": True,
        "seed": 20260810,
        "source": (
            "same_measurement_covariance_R"
        ),
    },

    "detection_model": {
        "enabled": True,
        "seed": 20260811,

        "model": (
            "logistic_pd_from_sensing_snr"
        ),

        "snr_midpoint_db": 20.0,
        "transition_width_db": 2.0,
        "pd_floor": 0.05,
        "pd_ceiling": 0.95,

        "current_fixed_snr_pd": 0.5,

        "semantics": (
            "controlled_stage2_degraded_smoke_"
            "not_calibrated_part_a_pd"
        ),
    },

    "false_alarm_model": {
        "enabled": True,
        "seed": 20260812,

        "mean_per_frame": 2.0,

        "model": (
            "poisson_uniform_measurement_volume_v1"
        ),

        "semantics": (
            "configurable_stage2_degraded_"
            "assumption_not_part_a_cfar_calibration"
        ),
    },

    "algorithm_input": {
        "unlabeled": True,
        "track_id_present": False,
        "object_class_present": False,
        "truth_source_present": False,
    },

    "truth_sidecar": {
        "evaluator_only": True,
    },

    "measured_fmcw": False,

    "reproducible": True,
    "future_dependency": False,

    "future_mutation_hash_identical":
        True,

    "algorithm_sha256":
        algorithm_hash_1,

    "truth_sha256":
        truth_hash_1,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage2 degraded real-scene smoke ====="
)

print(
    "causal frames        =",
    len(degraded.frames),
)

print(
    "clean upstream valid =",
    upstream_valid,
)

print(
    "true detections      =",
    true_detections,
)

print(
    "missed detections    =",
    missed_detections,
)

print(
    "false alarms         =",
    false_alarms,
)

print(
    "algorithm detections =",
    algorithm_detection_count,
)

print(
    "algorithm unlabeled  = YES"
)

print(
    "oracle truth sidecar = EVALUATOR ONLY"
)

print(
    "reproducible         = YES"
)

print(
    "future hash identical= True"
)

print(
    "measured FMCW        = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)

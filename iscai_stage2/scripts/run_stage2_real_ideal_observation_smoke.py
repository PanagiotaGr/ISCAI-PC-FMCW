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
from iscai_stage2.observations.womd_ideal_adapter import (
    build_causal_dynamic_headlamp_frames,
    build_real_ideal_observation_scene,
    ideal_observation_scene_sha256,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/"
    "reports/stage2/"
    "block1c_real_ideal_observations.json"
)


def transform_close(a, b, tol=1e-12):
    for i in range(3):
        for j in range(3):
            if abs(
                a.rotation[i][j]
                - b.rotation[i][j]
            ) > tol:
                return False

        if abs(
            a.translation[i]
            - b.translation[i]
        ) > tol:
            return False

    return True


# ============================================================
# Original real scenario
# ============================================================

scenario = read_first_scenario(
    MOTION
)

if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected real pilot scenario."
    )

adapted = adapt_causal_womd_scenario(
    scenario
)

scene = build_real_ideal_observation_scene(
    raw_scenario=scenario,
    adapted=adapted,
    include_sdc=False,
)

scene_hash_before = (
    ideal_observation_scene_sha256(
        scene
    )
)

if len(scene.actors) != (
    len(scenario.tracks) - 1
):
    raise RuntimeError(
        "Default Stage-2 target set must "
        "exclude exactly the SDC."
    )


# ============================================================
# Ht(anchor) == frozen H0
# ============================================================

frames, headlamp_velocities = (
    build_causal_dynamic_headlamp_frames(
        adapted
    )
)

if len(frames) != 11:
    raise RuntimeError(
        f"Expected 11 causal Ht frames, "
        f"got {len(frames)}"
    )

anchor_frame = frames[
    adapted.anchor_index
]

if anchor_frame is None:
    raise RuntimeError(
        "Anchor dynamic headlamp frame invalid."
    )

if not transform_close(
    anchor_frame.T_Ht_from_W,
    adapted.frames.T_H0_from_W,
):
    raise RuntimeError(
        "Ht(anchor) != frozen H0."
    )

if headlamp_velocities[0] is not None:
    raise RuntimeError(
        "t=0 headlamp velocity must be invalid: "
        "strict adjacent rule."
    )


# ============================================================
# Observation gates
# ============================================================

total_records = 0
geometry_valid = 0
radial_valid = 0

ranges = []
radial_velocities = []
azimuths = []
elevations = []

class_counts = {}
class_geometry_valid = {}

for actor in scene.actors:
    if actor.is_sdc:
        raise RuntimeError(
            "SDC leaked into Stage-2 target set."
        )

    class_counts[
        actor.object_class
    ] = (
        class_counts.get(
            actor.object_class,
            0,
        )
        + 1
    )

    class_geometry_valid.setdefault(
        actor.object_class,
        0,
    )

    if len(actor.observations) != 11:
        raise RuntimeError(
            "Actor observation history "
            "must contain 11 causal frames."
        )

    for obs in actor.observations:
        total_records += 1

        if obs.measured_fmcw:
            raise RuntimeError(
                "WOMD observation falsely marked "
                "as measured FMCW."
            )

        if obs.frame_name != "Ht":
            raise RuntimeError(
                "Observation is not in Ht."
            )

        if obs.geometry_valid:
            geometry_valid += 1

            class_geometry_valid[
                actor.object_class
            ] += 1

            if (
                obs.range_m is None
                or obs.azimuth_rad is None
                or obs.elevation_rad is None
            ):
                raise RuntimeError(
                    "Valid geometry has missing fields."
                )

            if obs.range_m <= 0.0:
                raise RuntimeError(
                    "Non-positive valid range."
                )

            if not (
                -math.pi
                <= obs.azimuth_rad
                <= math.pi
            ):
                raise RuntimeError(
                    "Azimuth out of bounds."
                )

            if not (
                -math.pi / 2.0
                <= obs.elevation_rad
                <= math.pi / 2.0
            ):
                raise RuntimeError(
                    "Elevation out of bounds."
                )

            ranges.append(
                obs.range_m
            )
            azimuths.append(
                obs.azimuth_rad
            )
            elevations.append(
                obs.elevation_rad
            )

        if obs.radial_velocity_valid:
            radial_valid += 1

            if (
                obs.radial_velocity_mps
                is None
            ):
                raise RuntimeError(
                    "Valid radial velocity missing."
                )

            if not math.isfinite(
                obs.radial_velocity_mps
            ):
                raise RuntimeError(
                    "Non-finite radial velocity."
                )

            radial_velocities.append(
                obs.radial_velocity_mps
            )


if total_records != (
    len(scene.actors) * 11
):
    raise RuntimeError(
        "Unexpected Stage-2 record count."
    )

if geometry_valid <= 0:
    raise RuntimeError(
        "No valid real geometry observations."
    )

if radial_valid <= 0:
    raise RuntimeError(
        "No valid geometry-derived radial velocities."
    )


# ============================================================
# Future mutation MUST NOT affect ideal observables
# ============================================================

mutated = scenario_pb2.Scenario()
mutated.CopyFrom(scenario)

future_index = (
    int(mutated.current_time_index)
    + 1
)

if future_index >= len(
    mutated.tracks[0].states
):
    raise RuntimeError(
        "No future state available for mutation."
    )

future_state = (
    mutated.tracks[0]
    .states[future_index]
)

future_state.center_x += 4321.0
future_state.center_y -= 1234.0
future_state.velocity_x += 999.0
future_state.velocity_y -= 999.0
future_state.heading += 1.0
future_state.valid = (
    not future_state.valid
)

adapted_mutated = (
    adapt_causal_womd_scenario(
        mutated
    )
)

scene_mutated = (
    build_real_ideal_observation_scene(
        raw_scenario=mutated,
        adapted=adapted_mutated,
        include_sdc=False,
    )
)

scene_hash_after = (
    ideal_observation_scene_sha256(
        scene_mutated
    )
)

if scene_hash_before != scene_hash_after:
    raise RuntimeError(
        "Future mutation leaked into "
        "Stage-2 ideal observables."
    )


# ============================================================
# Report
# ============================================================

report = {
    "status": "PASS",
    "block": (
        "stage2_block1c_real_ideal_observations"
    ),
    "scenario_id": SID,

    "target_policy": (
        "all_non_sdc_actors_with_per_time_validity_masks"
    ),

    "causal_frames": 11,
    "target_actor_count": len(
        scene.actors
    ),
    "total_records": total_records,

    "geometry_valid_records": (
        geometry_valid
    ),
    "radial_velocity_valid_records": (
        radial_valid
    ),

    "class_actor_counts": (
        class_counts
    ),
    "class_geometry_valid_records": (
        class_geometry_valid
    ),

    "range_m": {
        "min": min(ranges),
        "max": max(ranges),
    },

    "radial_velocity_mps": {
        "min": min(
            radial_velocities
        ),
        "max": max(
            radial_velocities
        ),
        "sign_convention": (
            "negative_approaching_positive_receding"
        ),
    },

    "azimuth_rad": {
        "min": min(azimuths),
        "max": max(azimuths),
        "positive_direction": "left",
    },

    "elevation_rad": {
        "min": min(elevations),
        "max": max(elevations),
        "positive_direction": "up",
    },

    "frame": "Ht",
    "Ht_anchor_equals_H0": True,

    "headlamp_velocity": {
        "method": (
            "strict_adjacent_backward_difference"
        ),
        "t0_valid": False,
        "valid_count": sum(
            value is not None
            for value in headlamp_velocities
        ),
    },

    "actor_velocity": {
        "source": (
            "stage1a_strict_adjacent_past_only"
        ),
        "womd_annotated_velocity_used": False,
    },

    "radial_velocity_source": (
        "geometry_derived_from_causal_womd_trajectory"
    ),

    "angle_source": (
        "scene_perception_geometry"
    ),

    "measured_fmcw": False,

    "sensor_noise_applied": False,
    "snr_model_applied": False,
    "crlb_applied": False,

    "future_dependency": False,
    "future_mutation_hash_identical": (
        scene_hash_before
        == scene_hash_after
    ),

    "ideal_observation_sha256": (
        scene_hash_before
    ),
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
    "===== Stage2 real ideal-observation smoke ====="
)
print(
    "target actors        =",
    len(scene.actors),
)
print(
    "causal frames        =",
    11,
)
print(
    "total records        =",
    total_records,
)
print(
    "geometry valid       =",
    geometry_valid,
)
print(
    "radial velocity valid=",
    radial_valid,
)
print(
    "classes              =",
    class_counts,
)
print(
    "range [m]            =",
    min(ranges),
    max(ranges),
)
print(
    "vr [m/s]             =",
    min(radial_velocities),
    max(radial_velocities),
)
print(
    "Ht(anchor)==H0       = True"
)
print(
    "future hash identical=",
    scene_hash_before
    == scene_hash_after,
)
print(
    "measured FMCW        = NO"
)
print(
    "SNR/CRLB/noise       = NO"
)
print("STATUS = PASS")
print("report =", REPORT)

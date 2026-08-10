from __future__ import annotations

import json
import shutil
from pathlib import Path

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)
from iscai_stage1.lidar.actor_features import (
    extract_causal_actor_lidar,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

LIDAR = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/lidar/b8/"
    "b85e1bd6cc8e74c0.tfrecord"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/"
    "reports/stage1a/"
    "block6b_real_actor_lidar.json"
)


motion = read_first_scenario(MOTION)
lidar = read_first_scenario(LIDAR)

if motion.scenario_id != SID:
    raise RuntimeError(
        "Motion scenario-id mismatch."
    )

if LIDAR.stem != SID:
    raise RuntimeError(
        "Frozen LiDAR filename association mismatch."
    )

if lidar.scenario_id != "":
    print(
        "NOTE: sidecar embedded scenario_id is non-empty:",
        repr(lidar.scenario_id),
    )

artifacts = extract_causal_actor_lidar(
    motion_scenario=motion,
    lidar_sidecar_scenario=lidar,
)

if len(artifacts) != len(motion.tracks):
    raise RuntimeError(
        "Actor/LiDAR artifact count mismatch."
    )

if len(artifacts) != 14:
    raise RuntimeError(
        f"Expected 14 actors, got {len(artifacts)}"
    )

for artifact in artifacts:
    if len(artifact.frames) != 11:
        raise RuntimeError(
            "Expected 11 causal LiDAR feature frames."
        )

    if artifact.sensor_realistic:
        raise RuntimeError(
            "Stage1A LiDAR artifact must remain "
            "sensor_realistic=false."
        )


rows = []

total_assigned_points = 0
nonempty_actor_frames = 0

for artifact in artifacts:
    track = motion.tracks[
        artifact.track_index
    ]

    frame_rows = []

    for feature in artifact.frames:
        total_assigned_points += (
            feature.point_count
        )

        if feature.point_count > 0:
            nonempty_actor_frames += 1

        frame_rows.append(
            {
                "time_index": feature.time_index,
                "timestamp_s": feature.timestamp_s,
                "actor_state_valid": (
                    feature.actor_state_valid
                ),
                "point_count": feature.point_count,
                "range_m": (
                    vars(feature.range_m)
                    if feature.range_m
                    is not None
                    else None
                ),
                "intensity": (
                    vars(feature.intensity)
                    if feature.intensity
                    is not None
                    else None
                ),
                "elongation": (
                    vars(feature.elongation)
                    if feature.elongation
                    is not None
                    else None
                ),
                "spatial_std_actor_m": (
                    list(
                        feature
                        .spatial_std_actor_m
                    )
                    if feature
                    .spatial_std_actor_m
                    is not None
                    else None
                ),
            }
        )

    rows.append(
        {
            "track_index": (
                artifact.track_index
            ),
            "track_id": artifact.track_id,
            "object_type": int(
                track.object_type
            ),
            "association_mode": (
                artifact.association_mode
            ),
            "assignment_mode": (
                artifact
                .lidar_actor_assignment_mode
            ),
            "frames": frame_rows,
        }
    )


if total_assigned_points <= 0:
    raise RuntimeError(
        "Real pilot yielded zero total actor-assigned "
        "LiDAR points."
    )

if nonempty_actor_frames <= 0:
    raise RuntimeError(
        "No actor frame received LiDAR points."
    )


free_gib = (
    shutil.disk_usage(
        "/home/agni/waymo/data"
    ).free
    / (1024 ** 3)
)

if free_gib < 250.0:
    raise RuntimeError(
        f"Hard reserve violated: {free_gib:.3f} GiB"
    )


report = {
    "status": "PASS",
    "block": "stage1a_block6b_real_actor_lidar",
    "scenario_id": SID,
    "association_contract": {
        "association_mode": (
            "oracle_womd_track_id"
        ),
        "lidar_actor_assignment_mode": (
            "oracle_causal_box"
        ),
        "sidecar_identity_source": (
            "frozen_manifest_and_filename"
        ),
        "embedded_sidecar_scenario_id": (
            lidar.scenario_id
        ),
    },
    "artifact_semantics": (
        "causal_womd_annotation_upstream"
    ),
    "sensor_realistic": False,
    "causal_frames": 11,
    "actor_count": len(artifacts),
    "total_actor_assigned_points": (
        total_assigned_points
    ),
    "nonempty_actor_frames": (
        nonempty_actor_frames
    ),
    "mvp_features": [
        "point_count",
        "range_stats",
        "intensity_stats",
        "elongation_stats",
        "actor_local_spatial_std",
    ],
    "all_lasers": True,
    "both_returns": True,
    "future_lidar_used": False,
    "future_boxes_used": False,
    "free_gib": free_gib,
    "actors": rows,
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

print("===== Stage1A real actor LiDAR =====")
print("actors =", len(artifacts))
print("causal_frames = 11")
print(
    "total_actor_assigned_points =",
    total_assigned_points,
)
print(
    "nonempty_actor_frames =",
    nonempty_actor_frames,
)
print(f"free_gib = {free_gib:.3f}")
print("future LiDAR used = NO")
print("future boxes used = NO")
print("STATUS = PASS")
print("report =", REPORT)

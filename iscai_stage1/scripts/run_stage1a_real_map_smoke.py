from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage1.maps.hashing import (
    causal_map_sha256,
)
from iscai_stage1.maps.womd_adapter import (
    adapt_causal_womd_map,
)


SCENARIO_ID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/"
    "reports/stage1a/"
    "block5b_real_map_adapter.json"
)


scenario = read_first_scenario(MOTION)

if scenario.scenario_id != SCENARIO_ID:
    raise RuntimeError(
        f"Unexpected scenario {scenario.scenario_id}"
    )

actors = adapt_causal_womd_scenario(
    scenario
)

maps = adapt_causal_womd_map(
    scenario,
    actors.frames.T_H0_from_W,
)


if maps.scenario_id != SCENARIO_ID:
    raise RuntimeError(
        "Map scenario-id mismatch."
    )

if maps.anchor_index != 10:
    raise RuntimeError(
        f"Unexpected anchor={maps.anchor_index}"
    )

if len(maps.static_features) != 139:
    raise RuntimeError(
        "Expected 139 static features, got "
        f"{len(maps.static_features)}"
    )

if len(maps.dynamic_frames) != 11:
    raise RuntimeError(
        "Expected 11 causal dynamic frames, got "
        f"{len(maps.dynamic_frames)}"
    )


counts = Counter(
    feature.kind
    for feature in maps.static_features
)

expected_counts = {
    "driveway": 12,
    "lane": 92,
    "road_edge": 19,
    "road_line": 11,
    "speed_bump": 4,
    "stop_sign": 1,
}

if dict(counts) != expected_counts:
    raise RuntimeError(
        "Unexpected pilot static-map counts: "
        f"{dict(counts)}"
    )


dynamic_lane_counts = [
    len(frame.lane_states)
    for frame in maps.dynamic_frames
]

if dynamic_lane_counts != [0] * 11:
    raise RuntimeError(
        "Unexpected causal dynamic lane states: "
        f"{dynamic_lane_counts}"
    )


# Verify W -> H0 -> W on the first available map point.
checked_roundtrip = False
max_roundtrip_error = 0.0

for feature in maps.static_features:
    if not feature.points_W_m:
        continue

    p_W = feature.points_W_m[0]
    p_H0 = feature.points_H0_m[0]

    recovered = (
        actors.frames.T_W_from_H0
        .apply_point(p_H0)
    )

    error = max(
        abs(a - b)
        for a, b in zip(
            p_W,
            recovered,
        )
    )

    max_roundtrip_error = max(
        max_roundtrip_error,
        error,
    )

    checked_roundtrip = True
    break

if not checked_roundtrip:
    raise RuntimeError(
        "No static map point available "
        "for frame round-trip gate."
    )

if max_roundtrip_error > 1e-8:
    raise RuntimeError(
        "Map W/H0 round-trip failure: "
        f"{max_roundtrip_error}"
    )


digest = causal_map_sha256(
    maps
)

report = {
    "status": "PASS",
    "block": "stage1a_block5b_real_map_adapter",
    "scenario_id": maps.scenario_id,
    "anchor_index": maps.anchor_index,
    "artifact_semantics": maps.artifact_semantics,
    "sensor_realistic": maps.sensor_realistic,
    "static_feature_count": len(
        maps.static_features
    ),
    "static_kind_counts": dict(
        sorted(counts.items())
    ),
    "causal_dynamic_frame_count": len(
        maps.dynamic_frames
    ),
    "causal_dynamic_lane_counts": (
        dynamic_lane_counts
    ),
    "future_dynamic_map_inspected": False,
    "map_frame_roundtrip_max_error_m": (
        max_roundtrip_error
    ),
    "causal_map_sha256": digest,
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

print("===== Stage1A real causal map =====")
print(
    "static_features =",
    report["static_feature_count"],
)
print(
    "kinds           =",
    report["static_kind_counts"],
)
print(
    "dynamic_frames  =",
    report["causal_dynamic_frame_count"],
)
print(
    "lane_states     =",
    report["causal_dynamic_lane_counts"],
)
print(
    "roundtrip_error =",
    report["map_frame_roundtrip_max_error_m"],
)
print(
    "map_sha256      =",
    digest,
)
print("STATUS = PASS")
print("report =", REPORT)

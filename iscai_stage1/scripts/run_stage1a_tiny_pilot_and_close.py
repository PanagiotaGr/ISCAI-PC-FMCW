from __future__ import annotations

import hashlib
import json
import shutil
import unittest
from collections import defaultdict
from pathlib import Path

from iscai_stage0.womd_proto_io import (
    iter_scenarios,
    read_first_scenario,
)
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage1.artifacts.hashing import (
    causal_actor_sha256,
)
from iscai_stage1.artifacts.scenario_hashing import (
    combined_causal_artifact_sha256,
)
from iscai_stage1.lidar.actor_features import (
    extract_causal_actor_lidar,
)
from iscai_stage1.lidar.hashing import (
    causal_actor_lidar_sha256,
)
from iscai_stage1.maps.hashing import (
    causal_map_sha256,
)
from iscai_stage1.maps.womd_adapter import (
    adapt_causal_womd_map,
)


ROOT = Path("/home/agni/waymo")
STAGE1 = ROOT / "iscai_stage1"

PAIRED = (
    ROOT / "data/paired_womd_lidar_v1_3_0"
)

VAL_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
    "selected_validation.jsonl"
)

DATASET_MANIFEST = (
    PAIRED / "manifests/dataset_manifest.json"
)

REPORT_DIR = (
    STAGE1 / "reports/stage1a"
)

ARTIFACT_DIR = (
    STAGE1 / "artifacts/stage1a/manifests"
)

PILOT_MANIFEST = (
    ARTIFACT_DIR / "tiny_pilot_validation.jsonl"
)

CLOSURE_REPORT = (
    REPORT_DIR / "stage1a_closure_report.json"
)

IMPLEMENTATION_LOG = (
    ROOT
    / "iscai_stage0/docs/implementation_log.md"
)

EXPECTED_VAL_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

PILOT_COUNT = 3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def json_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


# ============================================================
# 1. Frozen provenance gates
# ============================================================

if sha256_file(VAL_MANIFEST) != EXPECTED_VAL_SHA:
    raise RuntimeError(
        "Frozen validation selection manifest hash mismatch."
    )

dataset_manifest = json.loads(
    DATASET_MANIFEST.read_text()
)

if dataset_manifest.get("status") != "COMPLETE_FROZEN":
    raise RuntimeError(
        "Canonical paired dataset is not COMPLETE_FROZEN."
    )

if (
    dataset_manifest.get("association_key")
    != "exact_scenario_id"
):
    raise RuntimeError(
        "Unexpected paired-dataset association key."
    )


# ============================================================
# 2. Deterministic first-three validation records
# ============================================================

entries = []
raw_lines = []

with VAL_MANIFEST.open(
    "rb"
) as f:
    for line_number, raw in enumerate(
        f,
        start=1,
    ):
        if not raw.strip():
            continue

        entry = json.loads(
            raw.decode("utf-8")
        )

        entries.append(entry)
        raw_lines.append(raw)

        if len(entries) == PILOT_COUNT:
            break

if len(entries) != PILOT_COUNT:
    raise RuntimeError(
        f"Expected {PILOT_COUNT} pilot records."
    )

scenario_ids = [
    str(entry["scenario_id"])
    for entry in entries
]

if len(set(scenario_ids)) != PILOT_COUNT:
    raise RuntimeError(
        "Duplicate scenario-id in tiny pilot."
    )


ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with PILOT_MANIFEST.open(
    "wb"
) as f:
    for raw in raw_lines:
        f.write(raw)

pilot_manifest_sha = sha256_file(
    PILOT_MANIFEST
)


print("===== Deterministic tiny pilot =====")
for i, sid in enumerate(
    scenario_ids,
    start=1,
):
    print(i, sid)

print(
    "pilot_manifest_sha256 =",
    pilot_manifest_sha,
)


# ============================================================
# 3. Resolve compact motion shards
# ============================================================

wanted_by_motion_path = defaultdict(set)

for entry in entries:
    split = str(entry["split"])

    if split != "validation":
        raise RuntimeError(
            f"Unexpected pilot split {split!r}"
        )

    source_name = Path(
        str(entry["source_shard"])
    ).name

    motion_path = (
        PAIRED
        / split
        / "motion"
        / f"paired-from-{source_name}"
    )

    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Missing compact motion shard: {motion_path}"
        )

    wanted_by_motion_path[
        motion_path
    ].add(
        str(entry["scenario_id"])
    )


# ============================================================
# 4. Read only involved motion shards; early stop
# ============================================================

motion_scenarios = {}

for motion_path, wanted_ids in (
    wanted_by_motion_path.items()
):
    remaining = set(wanted_ids)

    for scenario in iter_scenarios(
        motion_path,
        limit=None,
    ):
        sid = str(scenario.scenario_id)

        if sid in remaining:
            motion_scenarios[sid] = scenario
            remaining.remove(sid)

        if not remaining:
            break

    if remaining:
        raise RuntimeError(
            "Pilot scenarios missing from compact motion shard "
            f"{motion_path}: {sorted(remaining)}"
        )

if set(motion_scenarios) != set(
    scenario_ids
):
    raise RuntimeError(
        "Tiny-pilot motion resolution mismatch."
    )


# ============================================================
# 5. Full Stage1A extraction for 3 scenarios
# ============================================================

pilot_rows = []

for entry in entries:
    sid = str(entry["scenario_id"])
    motion = motion_scenarios[sid]

    lidar_path = (
        PAIRED
        / "validation"
        / "lidar"
        / sid[:2]
        / f"{sid}.tfrecord"
    )

    if not lidar_path.is_file():
        raise FileNotFoundError(
            f"Missing exact LiDAR sidecar: {lidar_path}"
        )

    if lidar_path.stat().st_size != int(
        entry["lidar_bytes"]
    ):
        raise RuntimeError(
            f"LiDAR byte mismatch for {sid}"
        )

    lidar = read_first_scenario(
        lidar_path
    )

    if lidar_path.stem != sid:
        raise RuntimeError(
            "LiDAR filename association mismatch."
        )

    actors = adapt_causal_womd_scenario(
        motion
    )

    maps = adapt_causal_womd_map(
        motion,
        actors.frames.T_H0_from_W,
    )

    lidar_artifacts = (
        extract_causal_actor_lidar(
            motion_scenario=motion,
            lidar_sidecar_scenario=lidar,
        )
    )

    if len(actors.actors) != len(
        motion.tracks
    ):
        raise RuntimeError(
            f"Actor mismatch for {sid}"
        )

    if len(lidar_artifacts) != len(
        motion.tracks
    ):
        raise RuntimeError(
            f"LiDAR actor mismatch for {sid}"
        )

    if len(maps.dynamic_frames) != (
        int(motion.current_time_index) + 1
    ):
        raise RuntimeError(
            f"Dynamic-map causal-prefix mismatch for {sid}"
        )

    actor_hashes = tuple(
        causal_actor_sha256(
            actor.artifact
        )
        for actor in actors.actors
    )

    map_hash = causal_map_sha256(
        maps
    )

    lidar_hashes = tuple(
        causal_actor_lidar_sha256(
            artifact
        )
        for artifact in lidar_artifacts
    )

    actor_map_hash = (
        combined_causal_artifact_sha256(
            scenario_id=sid,
            anchor_index=int(
                motion.current_time_index
            ),
            actor_hashes=actor_hashes,
            map_hash=map_hash,
        )
    )

    full_stage1a_hash = json_sha256(
        {
            "scenario_id": sid,
            "anchor_index": int(
                motion.current_time_index
            ),
            "actor_map_hash": actor_map_hash,
            "lidar_hashes": lidar_hashes,
            "artifact_semantics":
                "causal_womd_annotation_upstream",
            "sensor_realistic": False,
        }
    )

    total_assigned = sum(
        frame.point_count
        for artifact in lidar_artifacts
        for frame in artifact.frames
    )

    nonempty_actor_frames = sum(
        frame.point_count > 0
        for artifact in lidar_artifacts
        for frame in artifact.frames
    )

    role_counts = {
        "context": sum(
            a.artifact.roles.is_context_actor
            for a in actors.actors
        ),
        "anchor_valid": sum(
            a.artifact.roles.is_anchor_valid
            for a in actors.actors
        ),
        "forecast_target_candidate": sum(
            a.artifact.roles
            .is_forecasting_target_candidate
            for a in actors.actors
        ),
        "receiver_candidate": sum(
            a.artifact.roles.is_receiver_candidate
            for a in actors.actors
        ),
    }

    pilot_rows.append(
        {
            "scenario_id": sid,
            "source_shard": str(
                entry["source_shard"]
            ),
            "motion_record_offset":
                int(entry["record_offset"]),
            "motion_payload_length":
                int(entry["payload_length"]),
            "lidar_path": str(lidar_path),
            "lidar_bytes":
                lidar_path.stat().st_size,
            "actors": len(actors.actors),
            "static_map_features":
                len(maps.static_features),
            "causal_dynamic_frames":
                len(maps.dynamic_frames),
            "causal_lidar_frames":
                len(
                    lidar
                    .compressed_frame_laser_data
                ),
            "role_counts": role_counts,
            "total_actor_assigned_points":
                total_assigned,
            "nonempty_actor_frames":
                nonempty_actor_frames,
            "actor_map_sha256":
                actor_map_hash,
            "full_stage1a_causal_sha256":
                full_stage1a_hash,
        }
    )

    print()
    print("PASS", sid)
    print(
        "  actors =",
        len(actors.actors),
    )
    print(
        "  lidar points =",
        total_assigned,
    )
    print(
        "  full hash =",
        full_stage1a_hash,
    )


# ============================================================
# 6. Final regression inside capable environment
# ============================================================

loader = unittest.TestLoader()

core_suite = loader.discover(
    str(STAGE1 / "tests"),
    pattern="test_*.py",
)

lidar_suite = loader.discover(
    str(STAGE1 / "tests_lidar"),
    pattern="test_*.py",
)

runner = unittest.TextTestRunner(
    verbosity=1
)

print()
print("===== Final core regression =====")
core_result = runner.run(
    core_suite
)

print()
print("===== Final LiDAR regression =====")
lidar_result = runner.run(
    lidar_suite
)

if not core_result.wasSuccessful():
    raise RuntimeError(
        "Final core regression failed."
    )

if not lidar_result.wasSuccessful():
    raise RuntimeError(
        "Final LiDAR regression failed."
    )

core_tests = core_result.testsRun
lidar_tests = lidar_result.testsRun


# ============================================================
# 7. Previous mandatory gate reports
# ============================================================

required_reports = [
    "block4c_real_scenario_smoke.json",
    "block4d_real_actor_adapter.json",
    "block5b_real_map_adapter.json",
    "block6b_real_actor_lidar.json",
    "block7a_future_mutation_gate.json",
    "block8a_visual_gate.json",
]

gate_reports = {}

for name in required_reports:
    path = REPORT_DIR / name

    if not path.is_file():
        raise FileNotFoundError(
            f"Mandatory Stage1A report missing: {path}"
        )

    payload = json.loads(
        path.read_text()
    )

    if payload.get("status") != "PASS":
        raise RuntimeError(
            f"Mandatory gate not PASS: {path}"
        )

    gate_reports[name] = {
        "status": "PASS",
        "sha256": sha256_file(path),
    }


# ============================================================
# 8. Visual artifact existence/hashes
# ============================================================

visual_report = json.loads(
    (
        REPORT_DIR
        / "block8a_visual_gate.json"
    ).read_text()
)

for item in visual_report[
    "visualizations"
].values():
    path = Path(item["path"])

    if not path.is_file():
        raise FileNotFoundError(
            f"Visual artifact missing: {path}"
        )

    if sha256_file(path) != item["sha256"]:
        raise RuntimeError(
            f"Visual hash mismatch: {path}"
        )


# ============================================================
# 9. Storage reserve
# ============================================================

free_gib = (
    shutil.disk_usage(PAIRED).free
    / (1024 ** 3)
)

if free_gib < 250.0:
    raise RuntimeError(
        f"Hard reserve violated: "
        f"{free_gib:.3f} GiB"
    )


# ============================================================
# 10. Closure report
# ============================================================

closure = {
    "status": "PASS_COMPLETE",
    "stage": "Stage 1A",
    "scope": (
        "causal preprocessing / canonical representation / "
        "frames / receiver geometry / static+causal dynamic map / "
        "causal annotation-assisted LiDAR MVP"
    ),
    "dataset_release": "WOMD/WOMD-LiDAR v1.3.0",
    "canonical_paired_corpus": str(PAIRED),
    "artifact_semantics":
        "causal_womd_annotation_upstream",
    "sensor_realistic": False,

    "tiny_pilot": {
        "policy":
            "first_3_records_of_frozen_validation_jsonl",
        "scenario_count": PILOT_COUNT,
        "scenario_ids": scenario_ids,
        "manifest": str(PILOT_MANIFEST),
        "manifest_sha256":
            pilot_manifest_sha,
        "scenarios": pilot_rows,
    },

    "regression": {
        "core_tests": core_tests,
        "lidar_tests": lidar_tests,
        "total_tests":
            core_tests + lidar_tests,
        "pass": True,
    },

    "mandatory_gate_reports":
        gate_reports,

    "causality": {
        "future_actor_state_dependency":
            False,
        "future_dynamic_map_dependency":
            False,
        "future_lidar_dependency":
            False,
        "future_box_dependency":
            False,
        "tracks_to_predict_as_realistic_input":
            False,
        "objects_of_interest_as_realistic_input":
            False,
        "annotated_womd_velocity_as_realistic_input":
            False,
        "future_mutation_gate_pass":
            True,
    },

    "geometry": {
        "W": "WOMD world/global",
        "E0_origin":
            "SDC box center at anchor",
        "E0_axes":
            "+x forward, +y left, +z up",
        "H0":
            "front_face_midpoint_surrogate",
        "receiver_geometry_mode":
            "centroid_baseline",
    },

    "lidar": {
        "association_mode":
            "oracle_womd_track_id",
        "actor_assignment_mode":
            "oracle_causal_box",
        "all_lasers": True,
        "both_returns": True,
        "mvp_features": [
            "point_count",
            "range_stats",
            "intensity_stats",
            "elongation_stats",
            "spatial_spread",
        ],
    },

    "visual_gate": {
        "headlamp_centric": True,
        "elevation_3d": True,
    },

    "storage": {
        "free_gib_at_closure":
            free_gib,
        "hard_reserve_gib":
            250.0,
        "hard_reserve_gate_pass":
            True,
    },

    "not_in_stage1a": [
        "PC-FMCW observation noise/CRLB",
        "radial-velocity sensing model",
        "forecast predictor/training",
        "CV/CA/CTRV/Kalman/IMM/MHT baselines",
        "beam codebook/adaptive Top-K",
        "optical BER/effective-rate chain",
        "predictive ADB",
        "DeepSense validation",
    ],

    "next_stage":
        "Official Stage 2: PC-FMCW-like observations "
        "+ measurement covariance",
}


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLOSURE_REPORT.write_text(
    json.dumps(
        closure,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# 11. Append-only canonical implementation log
# ============================================================

marker = (
    "## Stage 1A closure — COMPLETE/FROZEN"
)

existing_log = (
    IMPLEMENTATION_LOG.read_text()
    if IMPLEMENTATION_LOG.exists()
    else ""
)

if marker not in existing_log:
    with IMPLEMENTATION_LOG.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write("\n")
        f.write(marker + "\n\n")
        f.write(
            "- Status: COMPLETE/FROZEN.\n"
        )
        f.write(
            "- Artifact semantics: "
            "`causal_womd_annotation_upstream`; "
            "`sensor_realistic = false`.\n"
        )
        f.write(
            "- Final deterministic tiny pilot: "
            "first 3 records of the frozen validation "
            "selection manifest.\n"
        )
        f.write(
            f"- Final regression: "
            f"{core_tests} core + "
            f"{lidar_tests} LiDAR tests PASS.\n"
        )
        f.write(
            "- Real gates PASS: actor adapter, causal maps, "
            "causal actor-LiDAR MVP, strict future-mutation "
            "hash invariance, headlamp-centric visualization, "
            "3-D/elevation visualization.\n"
        )
        f.write(
            f"- Free space at closure: "
            f"{free_gib:.3f} GiB; "
            "250 GiB hard reserve PASS.\n"
        )
        f.write(
            "- Stage 1A does not claim sensor-realistic "
            "PC-FMCW measurements; those begin in official "
            "Stage 2.\n"
        )
        f.write(
            "- Next stage: PC-FMCW-like observations + "
            "measurement covariance.\n"
        )


print()
print("===== STAGE 1A CLOSURE =====")
print(
    "pilot scenarios =",
    scenario_ids,
)
print(
    "core tests      =",
    core_tests,
)
print(
    "LiDAR tests     =",
    lidar_tests,
)
print(
    "total tests     =",
    core_tests + lidar_tests,
)
print(
    f"free_gib        = {free_gib:.3f}"
)
print("STATUS = PASS_COMPLETE")
print(
    "closure report =",
    CLOSURE_REPORT,
)

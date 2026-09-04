#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/agni/waymo")
S7 = ROOT / "iscai_stage7"
S3 = ROOT / "iscai_stage3"
AUDIT = ROOT / "audits/stage7_pdf_alignment"

LAST_BINDING_ZIP = AUDIT / "block75c3_last_binding_read.zip"
B75B0_LOG = AUDIT / "block75b0_final_preformal_schema_slice_binding.log"
STAGE1_LIDAR_FEATURES = ROOT / "iscai_stage1/src/iscai_stage1/lidar/actor_features.py"
STAGE1_LIDAR_CONTRACTS = ROOT / "iscai_stage1/src/iscai_stage1/lidar/contracts.py"

SCRIPT = S7 / "scripts/run_block75c3a_core_statistics.py"

B75B = S7 / "configs/stage7_block75b_final_formal_evaluator_contract.json"
B74 = S7 / "configs/stage7_block74_MINIMAL_HANDOFF.json"
FORMAL = S3 / "artifacts/block38e/formal_validation_120.jsonl"
VALIDATION_MANIFEST = ROOT / "iscai_data_prep/manifests/selected_validation.jsonl"
PAIRED_ROOT = ROOT / "data/paired_womd_lidar_v1_3_0"

C1_MANIFEST = S7 / "artifacts/block75c1_nonoracle_lock_manifest.json"
C1_RUNNER = S7 / "scripts/run_block75c1_formal_causal_output_lock.py"
C1_CHECKER = S7 / "scripts/check_block75c1_nonoracle_lock.py"

C2_RUNNER = S7 / "scripts/run_block75c2_final_formal_evaluator.py"
C2_MANIFEST = S7 / "artifacts/block75c2_final_formal_raw_manifest.json"
C2_ROWS = S7 / "artifacts/block75c2_final_joint_rows.jsonl"
C2_REPORT = S7 / "reports/stage7_block75c2_final_formal_evaluation_report.json"
C2_SEAL = S7 / "artifacts/stage7_block75c2_final_frozen_seal.json"

OUT_ROOT = S7 / "artifacts/block75c3a_core_statistics"
SLICE_RAW = OUT_ROOT / "slice_membership_raw"
AGGREGATE = OUT_ROOT / "five_system_aggregate.json"
BOOTSTRAP = OUT_ROOT / "paired_bootstrap_10000.json"
FAILURES = OUT_ROOT / "failure_slices.json"
TRADEOFF = OUT_ROOT / "common_tradeoff_core.json"
MANIFEST = OUT_ROOT / "block75c3a_manifest.json"
REPORT = S7 / "reports/stage7_block75c3a_core_statistics_report.json"

EXPECTED = {
    LAST_BINDING_ZIP: "444f7a50da27eca2174ba8afea83b6d600a4bbdb18dac0838123cfb2f0b5f255",
    B75B0_LOG: "c11d70ad24311ff5e8233276d34867edab543a0c52277434d9a3d6b5014f049a",
    STAGE1_LIDAR_FEATURES: "d7337726b0b2ad60e1dab2a5dd62e00cf3a55d1ff5a643580f26a5b370b05752",
    STAGE1_LIDAR_CONTRACTS: "01150645798b76613b858c456b5ab850e6440e7911be654aaa4e9c102a5e2806",
    B75B: "5f79169138653b8223a795fae625adc5859a7a046a67c79fa98d357e256d7029",
    B74: "1351ac6d08f42e8a6627b6631aad438e89f416006feab800cd45c74d7f14eefa",
    FORMAL: "2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46",
    VALIDATION_MANIFEST: "dc10609ef18a2ba881657eb3da3a3df7a81bdcc8345ecbc2227102ab16b8833c",
    C1_MANIFEST: "d6792d2bcf304bd51260e98d99e8f8fa1e0aa470849a2d1989bdeac5ad74bfe5",
    C1_RUNNER: "077237d85e62468f90ebf21a70fe6cdf3ad095fe83dba2985d0b4cb799379275",
    C1_CHECKER: "b9bfb6447fbfe1982c6aa46e0cb5053d527112207bfc68be5acea29de095725e",
    C2_RUNNER: "3ccbed33cc66f89a2b366e50b923e06af7c71f0d9ad2baeedf691eef099af454",
    C2_MANIFEST: "adae0abde4377e0e6a7c9e993540b22ad53b7b0ebff038b4313d946d57684175",
    C2_ROWS: "6498e4bfe78597bca43aea7b7a7e72d2ea13befeadd1eaee4216917c26f16830",
    C2_REPORT: "616a7ae86574889ded6213b2b5f4363b80f20b45e183040282d658de65b244e7",
    C2_SEAL: "743541e5fb6b7718ef6d6634e09f13c6f7b879d106a5bcd9c0e7e8dad7d5d82c",
}

SYSTEMS = (
    "shared_trajectory_posterior",
    "independent_models",
    "direct_beam_classifier",
    "direct_ADB_predictor",
    "deterministic_shared_trajectory",
)
SHARED = "shared_trajectory_posterior"
CORE_CLASSES = {"TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"}
BOOTSTRAP_SEED = 2338514766
BOOTSTRAP_N = 10000

class FailClosed(RuntimeError):
    pass

def require(x: bool, message: str) -> None:
    if not x:
        raise FailClosed(message)

def sha256_file(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ) + "\n").encode("utf-8")

def atomic_immutable(path: Path, obj: Any) -> str:
    data = canonical_bytes(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = path.read_bytes()
        require(old == data, f"Immutable artifact differs: {path}")
        return hashlib.sha256(old).hexdigest()
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        os.chmod(path, 0o444)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if temp.exists():
            temp.unlink()
    return hashlib.sha256(data).hexdigest()

def finite(x: Any) -> float | None:
    if x is None:
        return None
    y = float(x)
    return y if math.isfinite(y) else None

def mean_finite(values) -> float | None:
    xs = [float(v) for v in values if finite(v) is not None]
    return None if not xs else float(np.mean(np.asarray(xs, dtype=np.float64)))

def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing JSONL: {path}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

# ---------------------------------------------------------------------------
# Causal-only failure slice membership.
# This function receives no C2 outcome metrics.
# ---------------------------------------------------------------------------

_RUNTIME: dict[str, Any] = {}
_VALIDATION_BY_ID: dict[str, Any] = {}

def init_slice_worker() -> None:
    global _RUNTIME, _VALIDATION_BY_ID
    from iscai_stage3.validation.womd_access import read_validation_manifest, read_motion_scenario
    from iscai_stage1.actors.womd_adapter import adapt_causal_womd_scenario, object_type_name
    from iscai_stage0.womd_proto_io import read_first_scenario
    from iscai_stage1.lidar.actor_features import extract_causal_actor_lidar

    rows = read_validation_manifest(VALIDATION_MANIFEST)
    _VALIDATION_BY_ID = {str(r.scenario_id): r for r in rows}
    require(len(_VALIDATION_BY_ID) == 44097, "Validation runtime count changed.")
    _RUNTIME = {
        "read_motion_scenario": read_motion_scenario,
        "adapt_causal_womd_scenario": adapt_causal_womd_scenario,
        "object_type_name": object_type_name,
        "read_first_scenario": read_first_scenario,
        "extract_causal_actor_lidar": extract_causal_actor_lidar,
    }

def state_position_h0(state: Any, transform: Any) -> np.ndarray:
    return np.asarray(transform.apply_point((
        float(state.center_x),
        float(state.center_y),
        float(state.center_z),
    )), dtype=np.float64)

def velocity_h0(track: Any, timestamps: list[float], transform: Any, t: int) -> np.ndarray | None:
    if t <= 0 or t >= len(track.states):
        return None
    a = track.states[t - 1]
    b = track.states[t]
    if not bool(a.valid) or not bool(b.valid):
        return None
    dt = float(timestamps[t]) - float(timestamps[t - 1])
    if not math.isfinite(dt) or dt <= 0.0:
        return None
    pa = state_position_h0(a, transform)
    pb = state_position_h0(b, transform)
    v = (pb - pa) / dt
    return v if np.all(np.isfinite(v)) else None

def has_temporary_missing(validity: list[bool], anchor: int) -> bool:
    if anchor <= 1 or not validity[anchor]:
        return False
    for i in range(1, anchor):
        if validity[i]:
            continue
        if any(validity[:i]) and any(validity[i + 1:anchor + 1]):
            return True
    return False

def beam_index_32(theta_deg: float) -> int | None:
    if theta_deg < -12.0 or theta_deg > 12.0:
        return None
    if theta_deg == 12.0:
        return 31
    i = int(math.floor((theta_deg + 12.0) / 0.75))
    return min(31, max(0, i))

def process_slice_membership(task: dict[str, Any]) -> dict[str, Any]:
    sid = str(task["scenario_id"])
    idx = int(task["formal_index"])
    formal = task["formal"]
    receiver = task["receiver"]
    c1_lock_sha = str(task["c1_lock_sha256"])
    runner_sha = str(task["runner_sha256"])

    vr = _VALIDATION_BY_ID.get(sid)
    require(vr is not None, f"Validation runtime row missing: {sid}")
    scenario = _RUNTIME["read_motion_scenario"](
        vr,
        paired_root=PAIRED_ROOT,
        compact_record_offset=int(formal["compact_record_offset"]),
    )
    require(str(scenario.scenario_id) == sid, f"Scenario ID mismatch: {sid}")
    anchor = int(scenario.current_time_index)
    require(anchor == 10, f"Anchor changed: {sid}")

    adapted = _RUNTIME["adapt_causal_womd_scenario"](scenario)
    transform = adapted.frames.T_H0_from_W
    timestamps = [float(x) for x in scenario.timestamps_seconds]

    actor_rows = []
    for ti, track in enumerate(scenario.tracks):
        if ti == int(scenario.sdc_track_index):
            continue
        cls = str(_RUNTIME["object_type_name"](track))
        if cls not in CORE_CLASSES:
            continue
        if anchor >= len(track.states) or not bool(track.states[anchor].valid):
            continue
        pos = state_position_h0(track.states[anchor], transform)
        vcur = velocity_h0(track, timestamps, transform, anchor)

        vstates = []
        for t in range(1, anchor + 1):
            v = velocity_h0(track, timestamps, transform, t)
            if v is not None:
                vstates.append((t, v))

        actor_rows.append({
            "track_index": ti,
            "track_id": str(track.id),
            "actor_class": cls,
            "position_H0_m": pos,
            "velocity_current_H0_mps": vcur,
            "velocity_states": vstates,
            "validity": [bool(track.states[t].valid) for t in range(anchor + 1)],
        })

    memberships: dict[str, bool] = {}

    # Selected communication receiver slices.
    if receiver:
        p = receiver.get("current_position_H0_m")
        if p is None:
            receiver_theta_deg = None
        else:
            receiver_theta_deg = math.degrees(math.atan2(float(p[1]), float(p[0])))
    else:
        receiver_theta_deg = None

    if receiver_theta_deg is None or not (-12.0 <= receiver_theta_deg <= 12.0):
        memberships["actor_near_beam_boundary"] = False
        memberships["receiver_near_FoV_edge"] = False
    else:
        internal = [-12.0 + 0.75 * k for k in range(1, 32)]
        nearest = min(abs(receiver_theta_deg - b) for b in internal)
        memberships["actor_near_beam_boundary"] = bool(nearest <= 0.18750000000000003)
        edge = min(abs(receiver_theta_deg + 12.0), abs(12.0 - receiver_theta_deg))
        memberships["receiver_near_FoV_edge"] = bool(edge <= 0.7500000000000001)

    # Scenario-level existential application of frozen actor predicates.
    braking = False
    ped_cross = False
    cyc_lat = False
    temp_missing = False

    for a in actor_rows:
        vs = a["velocity_states"]
        if len(vs) >= 2 and int(vs[-1][0]) == anchor:
            current_speed = float(np.linalg.norm(vs[-1][1][:2]))
            previous_speed = float(np.linalg.norm(vs[-2][1][:2]))
            if current_speed < previous_speed:
                braking = True

        vcur = a["velocity_current_H0_mps"]
        if vcur is not None:
            speed = float(np.linalg.norm(vcur[:2]))
            lateral = abs(float(vcur[1]))
            longitudinal = abs(float(vcur[0]))
            if speed >= 0.25 and lateral > longitudinal:
                if a["actor_class"] == "TYPE_PEDESTRIAN":
                    ped_cross = True
                if a["actor_class"] == "TYPE_CYCLIST":
                    cyc_lat = True

        if has_temporary_missing(a["validity"], anchor):
            temp_missing = True

    memberships["braking"] = braking
    memberships["pedestrian_crossing"] = ped_cross
    memberships["cyclist_lateral_maneuver"] = cyc_lat
    memberships["temporary_missing_track"] = temp_missing

    # Same / neighboring primary-32 beam.
    beam_indices = []
    for a in actor_rows:
        x, y = float(a["position_H0_m"][0]), float(a["position_H0_m"][1])
        b = beam_index_32(math.degrees(math.atan2(y, x)))
        if b is not None:
            beam_indices.append((a["track_index"], b))
    two_close = any(
        abs(b1 - b2) <= 1
        for i, (t1, b1) in enumerate(beam_indices)
        for t2, b2 in beam_indices[i + 1:]
        if t1 != t2
    )
    memberships["two_actors_same_or_neighboring_beam"] = bool(two_close)

    # Exact frozen Stage1 causal actor-LiDAR point count at anchor.
    lidar_path = PAIRED_ROOT / "validation" / "lidar" / sid[:2] / f"{sid}.tfrecord"
    require(lidar_path.is_file(), f"Missing paired LiDAR sidecar: {sid}")
    lidar = _RUNTIME["read_first_scenario"](lidar_path)
    artifacts = _RUNTIME["extract_causal_actor_lidar"](
        motion_scenario=scenario,
        lidar_sidecar_scenario=lidar,
    )
    require(len(artifacts) == len(scenario.tracks), f"LiDAR actor count mismatch: {sid}")
    low_zero = False
    actor_by_index = {int(a["track_index"]): a for a in actor_rows}
    for artifact in artifacts:
        ti = int(artifact.track_index)
        if ti not in actor_by_index:
            continue
        require(len(artifact.frames) == anchor + 1, f"LiDAR causal frame count changed: {sid}")
        frame = artifact.frames[anchor]
        require(int(frame.time_index) == anchor, f"LiDAR anchor index mismatch: {sid}")
        if bool(frame.actor_state_valid) and int(frame.point_count) == 0:
            low_zero = True
            break
    memberships["low_LiDAR_point_count"] = low_zero

    expected_evaluable = {
        "actor_near_beam_boundary",
        "receiver_near_FoV_edge",
        "braking",
        "pedestrian_crossing",
        "cyclist_lateral_maneuver",
        "temporary_missing_track",
        "low_LiDAR_point_count",
        "two_actors_same_or_neighboring_beam",
    }
    require(set(memberships) == expected_evaluable, f"Slice membership schema changed: {sid}")

    return {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3A",
        "status": "IMMUTABLE_CAUSAL_FAILURE_SLICE_MEMBERSHIP",
        "formal_index": idx,
        "scenario_id": sid,
        "runner_sha256": runner_sha,
        "block75b_contract_sha256": EXPECTED[B75B],
        "c1_lock_json_sha256": c1_lock_sha,
        "formal_outcomes_used_for_membership": False,
        "future_state_used_for_membership": False,
        "memberships": memberships,
    }

def bootstrap_mean(rng: np.random.Generator, values: np.ndarray) -> tuple[float, float, float]:
    require(values.ndim == 1 and len(values) > 0, "Empty bootstrap vector.")
    n = len(values)
    draws = rng.integers(0, n, size=(BOOTSTRAP_N, n), endpoint=False)
    means = values[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)

def main() -> int:
    print("=" * 88)
    print("STAGE 7 — BLOCK 7.5C3A")
    print("FROZEN C2 DERIVED STATISTICS + PAIRED 10K BOOTSTRAP + FAILURE SLICES")
    print("NO MODEL FORWARD / NO CONTROLLER RETUNING / NO C1-C2 MODIFICATION")
    print("=" * 88)

    runner_sha = sha256_file(SCRIPT)

    print("\n===== A. FROZEN AUTHORITY / C2 SEAL GATE =====")
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        require(actual == expected, f"SHA mismatch: {path}: {actual} != {expected}")
    contract = read_json(B75B)
    require(contract.get("status") == "FROZEN_FINAL_FORMAL_EVALUATOR_CONTRACT", "7.5B not frozen.")
    require(contract["statistics"]["resamples"] == BOOTSTRAP_N, "Bootstrap N changed.")
    require(contract["statistics"]["seed"] == BOOTSTRAP_SEED, "Bootstrap seed changed.")
    c2seal = read_json(C2_SEAL)
    require(c2seal.get("status") in ("FROZEN_COMPLETE_C2", "PASS", "FROZEN"), "C2 seal status invalid.")
    print("C1/C2/7.5B exact identities = PASS")

    c1 = read_json(C1_MANIFEST)
    require(c1.get("status") == "FROZEN_COMPLETE_NONORACLE_OUTPUT_LOCK", "C1 manifest status invalid.")
    c1_entries = c1["scenario_locks"]
    require(len(c1_entries) == 120, "C1 lock count changed.")

    formal_rows = read_jsonl(FORMAL)
    require(len(formal_rows) == 120, "Formal N changed.")
    formal_ids = [str(r["scenario_id"]) for r in formal_rows]
    order_sha = hashlib.sha256(json.dumps(
        formal_ids, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    require(order_sha == "4ba67f109c67f1306374423429654415ef884217dd93371f33e3d8fe27d948b7",
            "Formal order changed.")
    require(formal_ids == [str(e["scenario_id"]) for e in c1_entries], "C1/formal order mismatch.")

    # Verify every C1 lock before causal slice extraction.
    tasks = []
    for formal, entry in zip(formal_rows, c1_entries):
        lock_path = Path(entry["lock_json"])
        lock_sha = sha256_file(lock_path)
        require(lock_sha == entry["lock_json_sha256"], f"C1 lock SHA mismatch: {entry['scenario_id']}")
        lock = read_json(lock_path)
        require(lock.get("future_GT_accessed") is False, "C1 future flag changed.")
        require(lock.get("formal_metrics_computed") is False, "C1 metric flag changed.")
        require(lock.get("runner_sha256") == EXPECTED[C1_RUNNER], "C1 lock runner identity changed.")
        tasks.append({
            "formal_index": int(entry["formal_index"]),
            "scenario_id": str(entry["scenario_id"]),
            "formal": formal,
            "receiver": lock.get("receiver"),
            "c1_lock_sha256": lock_sha,
            "runner_sha256": runner_sha,
        })
    print("C1 120/120 locks reverified = PASS")

    print("\n===== B. CAUSAL FAILURE-SLICE MEMBERSHIP =====")
    SLICE_RAW.mkdir(parents=True, exist_ok=True)
    membership_docs: dict[str, dict[str, Any]] = {}
    pending = []
    for task in tasks:
        path = SLICE_RAW / f"{task['formal_index']:03d}_{task['scenario_id']}.json"
        if path.exists():
            doc = read_json(path)
            require(doc.get("runner_sha256") == runner_sha, f"Existing C3A runner mismatch: {path}")
            require(doc.get("scenario_id") == task["scenario_id"], f"Existing C3A SID mismatch: {path}")
            require(doc.get("c1_lock_json_sha256") == task["c1_lock_sha256"], f"Existing C3A C1 binding mismatch: {path}")
            membership_docs[task["scenario_id"]] = doc
        else:
            pending.append(task)

    workers = max(1, int(os.environ.get("C3_WORKERS", "12")))
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_slice_worker) as pool:
            futures = {pool.submit(process_slice_membership, t): t for t in pending}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    doc = fut.result()
                except Exception as exc:
                    raise FailClosed(
                        f"Slice membership scenario {task['formal_index']:03d} failed: {exc}"
                    ) from exc
                path = SLICE_RAW / f"{task['formal_index']:03d}_{task['scenario_id']}.json"
                atomic_immutable(path, doc)
                membership_docs[task["scenario_id"]] = doc
                i = int(task["formal_index"])
                if i % 20 == 0 or i == 119:
                    print(f"slice membership {i:03d}/119 complete", flush=True)

    require(len(membership_docs) == 120, "C3A slice membership count != 120.")
    print("120/120 causal slice memberships = PASS")

    print("\n===== C. VERIFY FINAL C2 RAW RECORD SET =====")
    c2m = read_json(C2_MANIFEST)
    require(c2m.get("status") == "FROZEN_COMPLETE_C2_RAW_FORMAL_RECORDS", "C2 manifest status invalid.")
    require(c2m.get("scenario_count") == 120 and c2m.get("joint_row_count") == 600, "C2 cardinality changed.")
    require(c2m.get("merged_joint_rows_sha256") == EXPECTED[C2_ROWS], "C2 rows binding changed.")
    for item in c2m["scenario_records"]:
        require(sha256_file(Path(item["record_path"])) == item["record_sha256"],
                f"C2 record drift: {item['scenario_id']}")
        require(sha256_file(Path(item["reference_path"])) == item["reference_sha256"],
                f"C2 reference drift: {item['scenario_id']}")
    print("C2 120/120 records + references = PASS")

    rows = read_jsonl(C2_ROWS)
    require(len(rows) == 600, "C2 joint row count changed.")
    by_sid_system = {(str(r["scenario_id"]), str(r["system"])): r for r in rows}
    require(len(by_sid_system) == 600, "Duplicate C2 scenario/system row.")
    for sid in formal_ids:
        require(all((sid, s) in by_sid_system for s in SYSTEMS), f"Missing system row: {sid}")

    metric_categories = {
        k: list(v) for k, v in contract["joint_metrics"].items()
        if k in (
            "communication_reliability",
            "beam_overhead",
            "glare_protection",
            "VRU_visibility",
            "over_masking",
        )
    }
    primary_metrics = []
    for category in (
        "communication_reliability",
        "beam_overhead",
        "glare_protection",
        "VRU_visibility",
        "over_masking",
    ):
        for metric in metric_categories[category]:
            if metric not in primary_metrics:
                primary_metrics.append(metric)

    aggregate_doc: dict[str, Any] = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3A",
        "status": "PASS",
        "aggregation": "equal-weight macro mean over finite scenario-level metric values",
        "scenario_count": 120,
        "systems": list(SYSTEMS),
        "metric_categories": metric_categories,
        "table": {},
    }
    for system in SYSTEMS:
        table = {}
        for metric in primary_metrics:
            vals = []
            for sid in formal_ids:
                v = finite(by_sid_system[(sid, system)]["metrics"].get(metric))
                if v is not None:
                    vals.append(v)
            table[metric] = {
                "mean": None if not vals else float(np.mean(np.asarray(vals, dtype=np.float64))),
                "finite_scenarios": len(vals),
            }
        aggregate_doc["table"][system] = table

    aggregate_sha = atomic_immutable(AGGREGATE, aggregate_doc)
    print("five-system aggregate table = PASS")

    print("\n===== D. PAIRED NONPARAMETRIC 10,000 BOOTSTRAP =====")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_doc: dict[str, Any] = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3A",
        "status": "PASS",
        "method": "paired_nonparametric_percentile_bootstrap",
        "resampling_unit": "scenario_id",
        "resamples": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "confidence": 0.95,
        "interval": "2.5th_and_97.5th_percentiles",
        "system_intervals": {},
        "paired_delta_vs_shared": {},
    }

    for metric in primary_metrics:
        bootstrap_doc["system_intervals"][metric] = {}
        for system in SYSTEMS:
            vals = [
                finite(by_sid_system[(sid, system)]["metrics"].get(metric))
                for sid in formal_ids
            ]
            arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
            if len(arr) == 0:
                result = {"mean": None, "ci95": [None, None], "n": 0}
            else:
                mean, lo, hi = bootstrap_mean(rng, arr)
                result = {"mean": mean, "ci95": [lo, hi], "n": int(len(arr))}
            bootstrap_doc["system_intervals"][metric][system] = result

        bootstrap_doc["paired_delta_vs_shared"][metric] = {}
        for system in SYSTEMS:
            if system == SHARED:
                paired_n = sum(
                    finite(by_sid_system[(sid, SHARED)]["metrics"].get(metric)) is not None
                    for sid in formal_ids
                )
                bootstrap_doc["paired_delta_vs_shared"][metric][system] = {
                    "mean_delta_system_minus_shared": 0.0 if paired_n else None,
                    "ci95": [0.0, 0.0] if paired_n else [None, None],
                    "paired_n": int(paired_n),
                }
                continue
            diffs = []
            for sid in formal_ids:
                a = finite(by_sid_system[(sid, system)]["metrics"].get(metric))
                b = finite(by_sid_system[(sid, SHARED)]["metrics"].get(metric))
                if a is not None and b is not None:
                    diffs.append(a - b)
            arr = np.asarray(diffs, dtype=np.float64)
            if len(arr) == 0:
                result = {
                    "mean_delta_system_minus_shared": None,
                    "ci95": [None, None],
                    "paired_n": 0,
                }
            else:
                mean, lo, hi = bootstrap_mean(rng, arr)
                result = {
                    "mean_delta_system_minus_shared": mean,
                    "ci95": [lo, hi],
                    "paired_n": int(len(arr)),
                }
            bootstrap_doc["paired_delta_vs_shared"][metric][system] = result

    bootstrap_sha = atomic_immutable(BOOTSTRAP, bootstrap_doc)
    print("paired 10,000 bootstrap = PASS")

    print("\n===== E. FAILURE-CASE TABLE — ALL 14 FROZEN NAMES =====")
    failure_contract = contract["failure_slices"]
    require(len(failure_contract) == 14, "Frozen failure slice count changed.")
    failure_doc: dict[str, Any] = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3A",
        "status": "PASS",
        "membership_outcome_independent": True,
        "post_formal_threshold_change": False,
        "slices": {},
    }

    for name in sorted(failure_contract):
        spec = failure_contract[name]
        if str(spec["status"]).startswith("NOT_EVALUABLE"):
            failure_doc["slices"][name] = {
                "status": "NOT_EVALUABLE",
                "reason": spec["reason"],
                "scenario_count": None,
                "system_metrics": None,
            }
            continue

        selected_ids = [
            sid for sid in formal_ids
            if bool(membership_docs[sid]["memberships"][name])
        ]
        sys_table = {}
        for system in SYSTEMS:
            mt = {}
            for metric in primary_metrics:
                vals = [
                    finite(by_sid_system[(sid, system)]["metrics"].get(metric))
                    for sid in selected_ids
                ]
                vals = [v for v in vals if v is not None]
                mt[metric] = {
                    "mean": None if not vals else float(np.mean(np.asarray(vals, dtype=np.float64))),
                    "finite_scenarios": len(vals),
                }
            sys_table[system] = mt

        failure_doc["slices"][name] = {
            "status": spec["status"],
            "frozen_definition": spec.get("definition"),
            "scenario_count": len(selected_ids),
            "scenario_ids": selected_ids,
            "system_metrics": sys_table,
        }

    require(set(failure_doc["slices"]) == set(failure_contract), "Failure slice name mismatch.")
    failures_sha = atomic_immutable(FAILURES, failure_doc)
    print("14/14 failure slice names retained = PASS")

    print("\n===== F. COMMON COMMUNICATION–ILLUMINATION TRADE-OFF CORE =====")
    tradeoff_metrics = (
        "outage_probability",
        "probability_coverage",
        "selected_K",
        "probing_overhead_fraction",
        "effective_rate_bps",
        "effective_rate_loss_bps",
        "vehicle_shadow_zone_violation",
        "glare_risk_exposure",
        "over_masking_area",
        "road_illumination_retention",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    )
    tradeoff_doc = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3A",
        "status": "PASS_CORE_BEFORE_RESOURCE_LATENCY_APPEND",
        "systems": {},
    }
    for system in SYSTEMS:
        tradeoff_doc["systems"][system] = {
            m: aggregate_doc["table"][system][m]
            for m in tradeoff_metrics
        }
    tradeoff_sha = atomic_immutable(TRADEOFF, tradeoff_doc)
    print("common trade-off core = PASS")

    membership_records = []
    for task in tasks:
        p = SLICE_RAW / f"{task['formal_index']:03d}_{task['scenario_id']}.json"
        membership_records.append({
            "formal_index": task["formal_index"],
            "scenario_id": task["scenario_id"],
            "path": str(p),
            "sha256": sha256_file(p),
        })

    manifest_doc = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3A",
        "status": "FROZEN_COMPLETE_C3A_CORE_STATISTICS",
        "runner_path": str(SCRIPT),
        "runner_sha256": runner_sha,
        "block75b_contract_sha256": EXPECTED[B75B],
        "c1_manifest_sha256": EXPECTED[C1_MANIFEST],
        "c2_manifest_sha256": EXPECTED[C2_MANIFEST],
        "c2_joint_rows_sha256": EXPECTED[C2_ROWS],
        "c2_seal_sha256": EXPECTED[C2_SEAL],
        "bootstrap": {
            "resamples": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "slice_membership_records": membership_records,
        "outputs": {
            str(AGGREGATE): aggregate_sha,
            str(BOOTSTRAP): bootstrap_sha,
            str(FAILURES): failures_sha,
            str(TRADEOFF): tradeoff_sha,
        },
        "causality": {
            "C1_modified": False,
            "C2_modified": False,
            "new_model_forward": False,
            "controller_retuning": False,
            "post_outcome_threshold_change": False,
            "failure_membership_uses_formal_outcomes": False,
            "failure_membership_future_state_use": False,
        },
        "next": "7.5C3B frozen codebook/coverage/uncertainty/latency sweeps; then C3C isolated resource-latency and final reproducibility seal",
    }
    manifest_sha = atomic_immutable(MANIFEST, manifest_doc)

    report_doc = {
        "status": "PASS",
        "stage": 7,
        "block": "7.5C3A",
        "runner_sha256": runner_sha,
        "manifest_sha256": manifest_sha,
        "scenario_count": 120,
        "joint_rows": 600,
        "bootstrap_resamples": BOOTSTRAP_N,
        "failure_slice_names": 14,
        "evaluable_failure_slices": 8,
        "not_evaluable_failure_slices": 6,
        "C1_C2_modified": False,
        "new_model_forward": False,
        "post_outcome_tuning": False,
        "C3_final_seal": False,
    }
    report_sha = atomic_immutable(REPORT, report_doc)

    print("\n" + "=" * 88)
    print("BLOCK 7.5C3A = FULLY VERIFIED / FROZEN")
    print("120/120 CAUSAL FAILURE-SLICE MEMBERSHIPS = PASS")
    print("600/600 FINAL C2 JOINT ROWS VERIFIED = PASS")
    print("PAIRED 10,000 BOOTSTRAP = PASS")
    print("14/14 FAILURE SLICE NAMES = PASS")
    print("COMMON TRADE-OFF CORE = PASS")
    print("C1 MODIFIED = NO")
    print("C2 MODIFIED = NO")
    print("NEW MODEL FORWARD = NO")
    print("POST-OUTCOME TUNING = NO")
    print("C3 FINAL SEAL = NO")
    print("C3A manifest SHA256 =", manifest_sha)
    print("C3A report SHA256   =", report_sha)
    print("C3A runner SHA256   =", runner_sha)
    print("STATUS = FROZEN_COMPLETE_C3A")
    print("=" * 88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!" * 88)
        print("BLOCK 7.5C3A FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("C1/C2 MUST REMAIN UNCHANGED.")
        print("NO C3 FINAL SEAL.")
        print("!" * 88)
        traceback.print_exc()
        raise SystemExit(1)

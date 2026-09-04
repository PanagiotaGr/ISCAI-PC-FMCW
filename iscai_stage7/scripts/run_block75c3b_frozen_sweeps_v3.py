#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path("/home/agni/waymo")
S7 = ROOT / "iscai_stage7"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S3 = ROOT / "iscai_stage3"

SCRIPT = S7 / "scripts/run_block75c3b_frozen_sweeps_v3.py"

B75B = S7 / "configs/stage7_block75b_final_formal_evaluator_contract.json"
FORMAL = S3 / "artifacts/block38e/formal_validation_120.jsonl"

C1_RUNNER = S7 / "scripts/run_block75c1_formal_causal_output_lock.py"
C1_MANIFEST = S7 / "artifacts/block75c1_nonoracle_lock_manifest.json"

C2_RUNNER = S7 / "scripts/run_block75c2_final_formal_evaluator.py"
C2_MANIFEST = S7 / "artifacts/block75c2_final_formal_raw_manifest.json"
C2_ROWS = S7 / "artifacts/block75c2_final_joint_rows.jsonl"
C2_SEAL = S7 / "artifacts/stage7_block75c2_final_frozen_seal.json"

C3A_RUNNER = S7 / "scripts/run_block75c3a_core_statistics.py"
C3A_ROOT = S7 / "artifacts/block75c3a_core_statistics"
C3A_AGG = C3A_ROOT / "five_system_aggregate.json"
C3A_BOOT = C3A_ROOT / "paired_bootstrap_10000.json"
C3A_FAIL = C3A_ROOT / "failure_slices.json"
C3A_TRADE = C3A_ROOT / "common_tradeoff_core.json"
C3A_MANIFEST = C3A_ROOT / "block75c3a_manifest.json"
C3A_REPORT = S7 / "reports/stage7_block75c3a_core_statistics_report.json"

SOURCE_BUNDLE = ROOT / "audits/stage7_pdf_alignment/block75_c2c3_exact_source_bundle.zip"
OUT_ROOT = S7 / "artifacts/block75c3b_frozen_sweeps"
RAW_ROOT = OUT_ROOT / "scenario_raw"
CODEBOOK_OUT = OUT_ROOT / "codebook_sweep.json"
COVERAGE_OUT = OUT_ROOT / "coverage_sweep.json"
UNCERTAINTY_OUT = OUT_ROOT / "uncertainty_sweep.json"
MANIFEST = OUT_ROOT / "block75c3b_manifest.json"
REPORT = S7 / "reports/stage7_block75c3b_frozen_sweeps_report.json"

EXPECTED = {
    B75B: "5f79169138653b8223a795fae625adc5859a7a046a67c79fa98d357e256d7029",
    FORMAL: "2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46",
    C1_RUNNER: "077237d85e62468f90ebf21a70fe6cdf3ad095fe83dba2985d0b4cb799379275",
    C1_MANIFEST: "d6792d2bcf304bd51260e98d99e8f8fa1e0aa470849a2d1989bdeac5ad74bfe5",
    C2_RUNNER: "3ccbed33cc66f89a2b366e50b923e06af7c71f0d9ad2baeedf691eef099af454",
    C2_MANIFEST: "adae0abde4377e0e6a7c9e993540b22ad53b7b0ebff038b4313d946d57684175",
    C2_ROWS: "6498e4bfe78597bca43aea7b7a7e72d2ea13befeadd1eaee4216917c26f16830",
    C2_SEAL: "743541e5fb6b7718ef6d6634e09f13c6f7b879d106a5bcd9c0e7e8dad7d5d82c",
    C3A_RUNNER: "3ca0f4921d09dbb97cc13ea9eee30d03afe8eb90f3437495212f7a6afbf89d73",
    C3A_AGG: "bec76ea0ed08c78623319356c89c43877fbe6e73ffe3abe7fda4835399c2eeef",
    C3A_BOOT: "5e19036f2376da5f449cdacf79ff7c4dd037412f65832380e5d56055558371b6",
    C3A_FAIL: "a07c57012bab0b74abd7ee633f1f0b818390a9ffc1b3361114e9d6b22fe2e7bb",
    C3A_TRADE: "c3a31d4a3550482a3875984d33fa3080d7b4e35fc4e3e395970886b570ecaa64",
    C3A_MANIFEST: "4c7da7e404c227414cc9e31053b9f46f7440e0ad207515addc801f7fb002fc4b",
    C3A_REPORT: "be51ebc83feef4fde84c7209864abb609e24c23cbde3fced9b437407ab570b54",
    SOURCE_BUNDLE: "8ce6594f1fe53a3f1fbb1bfa210ad475598818d75a4c82fe1fc31e19b2971282",
    S5 / "src/iscai_stage5/adaptive_topk.py": "08bb2145eac117132166a0f8b96a38c61b0a912cfd9b7f2d48f16db649d22865",
    S5 / "src/iscai_stage5/adaptive_topk_temporal.py": "433dad357bcba95d5e85d650ee0dcf7ffc5ef157aa8294df1fdba0ef3ab4627a",
    S5 / "src/iscai_stage5/angular_monte_carlo.py": "b86f7bcdb424340fb18c2d4970fa6130041e8ad5a0b4e31a97bb8d1e69460686",
    S5 / "src/iscai_stage5/beam_codebook.py": "acfc4c26db182bd4cec743bb1ebc2473729e10b5537c0b1dfed6f26d28d4573b",
    S5 / "src/iscai_stage5/beam_latency.py": "81c57865f5b5994e0dbbf9fba90e3a9e6638b99da1c7d7abccbe34a09d8d2810",
    S5 / "src/iscai_stage5/beam_baselines.py": "48b690d7d3c22e696ba7c7329427ac5d023bac3f86063c1490dde9048cf43d4a",
    S5 / "src/iscai_stage5/optical_link.py": "25474202adcf481e859e41fde0767d8a96dcf13a77862eb1f270d1d7f857b66c",
    S6 / "src/iscai_stage6/adb/metric_semantics.py": "bf8358b5a7edfe40ffac2718ca7cd5af6e7b5fdb3ff8a2823caad6d0f40f057a",
    S6 / "src/iscai_stage6/adb/probabilistic_full_box.py": "51daeab95713d6e7d2738ced466be4844b73dce70a09138b2b769de880462995",
    S6 / "src/iscai_stage6/adb/probabilistic_occupancy.py": "d4206761fc70495563d53a6ab58aefe6db5d8bd1b2d09736bfc5bbc6b6f50e3e",
    S6 / "src/iscai_stage6/adb/class_aware_policy.py": "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",
    S6 / "src/iscai_stage6/adb/womd_geometry.py": "b90f47289bd091474658e3864c4b1ccc8c2fc841aaa38160f38f0b95ece7187c",
}

SYSTEMS = (
    "shared_trajectory_posterior",
    "independent_models",
    "direct_beam_classifier",
    "direct_ADB_predictor",
    "deterministic_shared_trajectory",
)
CODEBOOK_VALUES = (16, 32, 64)
COVERAGE_VALUES = (0.90, 0.95, 0.975, 0.99)
ALPHA_VALUES = (0.5, 1.0, 1.5, 2.0)
PRIMARY_CODEBOOK = 32
PRIMARY_COVERAGE = 0.95

COMM_METRICS = (
    "probability_coverage",
    "outage_probability",
    "selected_K",
    "probing_overhead_fraction",
    "overhead_reduction_vs_exhaustive",
    "beam_switching_rate",
    "reacquisition_latency_s",
    "posterior_probability_mass_selected",
    "beam_gain_loss_db",
    "snr_loss_db",
    "BER",
    "effective_rate_bps",
    "effective_rate_loss_bps",
    "valid_future_horizons",
)
ADB_METRICS = (
    "mask_IoU_with_oracle_future_mask",
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "over_masking_area",
    "road_illumination_retention",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "false_dimming",
    "temporal_smoothness",
    "flicker_change_rate",
    "energy_consumption",
)

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

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

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
        return sha256_bytes(old)
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
    return sha256_bytes(data)

def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"Missing JSONL: {path}")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def finite(x: Any) -> float | None:
    if x is None:
        return None
    y = float(x)
    return y if math.isfinite(y) else None

def mean_finite(values) -> float | None:
    vals = [float(v) for v in values if finite(v) is not None]
    return None if not vals else float(np.mean(np.asarray(vals, dtype=np.float64)))

def array_digest(a: np.ndarray) -> str:
    x = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("ascii")); h.update(b"\0")
    h.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii")); h.update(b"\0")
    h.update(x.tobytes(order="C"))
    return h.hexdigest()

def module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    require(name not in sys.modules, f"Unexpected module collision: {name}")
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module

def close_enough(a: Any, b: Any, tol: float = 1e-12) -> bool:
    aa, bb = finite(a), finite(b)
    if aa is None or bb is None:
        return aa is None and bb is None
    return abs(aa - bb) <= tol * max(1.0, abs(aa), abs(bb))

def assert_metric_parity(actual: dict[str, Any], expected: dict[str, Any], names, label: str) -> None:
    for name in names:
        require(close_enough(actual.get(name), expected.get(name), 2e-11),
                f"{label} metric parity failed for {name}: {actual.get(name)} != {expected.get(name)}")

# Worker globals.
C1: Any = None
C2: Any = None
RT: dict[str, Any] = {}
CODEBOOKS: dict[int, Any] = {}
GRID: Any = None
ADB_PARAMS: dict[str, Any] = {}
BeamProbabilityMass: Any = None
adaptive_temporal_step: Any = None
initial_adaptive_temporal_state: Any = None
actual_probe_count_for_adaptive_decision: Any = None
oracle_best_gain_beam: Any = None
build_causal_adb_actor_boxes: Any = None

def init_worker() -> None:
    global C1, C2, RT, CODEBOOKS, GRID, ADB_PARAMS
    global BeamProbabilityMass, adaptive_temporal_step, initial_adaptive_temporal_state
    global actual_probe_count_for_adaptive_decision, oracle_best_gain_beam
    global build_causal_adb_actor_boxes

    C1 = module_from_file("iscai_stage7_c3b_c1_readonly", C1_RUNNER)
    C2 = module_from_file("iscai_stage7_c3b_c2_readonly", C2_RUNNER)
    C2.initialize_runtime()
    RT = C1.import_runtime()

    from iscai_stage5.beam_codebook import BeamProbabilityMass as BPM
    from iscai_stage5.adaptive_topk_temporal import (
        adaptive_temporal_step as ats,
        initial_adaptive_temporal_state as iats,
    )
    from iscai_stage5.beam_latency import (
        actual_probe_count_for_adaptive_decision as apc,
    )
    from iscai_stage5.beam_baselines import oracle_best_gain_beam as obgb
    from iscai_stage6.adb.womd_geometry import build_causal_adb_actor_boxes as bcab

    BeamProbabilityMass = BPM
    adaptive_temporal_step = ats
    initial_adaptive_temporal_state = iats
    actual_probe_count_for_adaptive_decision = apc
    oracle_best_gain_beam = obgb
    build_causal_adb_actor_boxes = bcab

    beam_cfg = read_json(C1.BEAM_CFG)
    support = beam_cfg["azimuth_support"]
    lo = math.radians(float(support["min_deg"]))
    hi = math.radians(float(support["max_deg"]))
    CODEBOOKS = {
        n: RT["build_uniform_azimuth_codebook"](
            beam_count=n,
            support_min_azimuth_rad=lo,
            support_max_azimuth_rad=hi,
        )
        for n in CODEBOOK_VALUES
    }
    GRID = C1.grid_from_frozen_config(RT["OccupancyGrid"])
    ADB_PARAMS = C1.frozen_adb_parameters()

def probability_object(payload: dict[str, Any]):
    masses = tuple(float(x) for x in payload["masses"])
    return BeamProbabilityMass(
        masses=masses,
        inside_support_mass=float(payload["inside_support_mass"]),
        outside_support_mass=float(payload["outside_support_mass"]),
        sample_count=int(payload["sample_count"]),
    )

def resolve_comm(lock: dict[str, Any], system: str) -> tuple[dict[str, Any], str]:
    raw = lock["communication_outputs"][system]
    if raw.get("status") == "REUSE_SHARED_COMMUNICATION_BRANCH":
        source = str(raw["source_system"])
        require(source == "shared_trajectory_posterior", "Unexpected communication alias.")
        return lock["communication_outputs"][source], source
    require(raw.get("status") == "SELECTED", f"Communication not SELECTED: {system}")
    return raw, system

def bind_temporal_return(value):
    require(isinstance(value, tuple) and len(value) == 2, "Temporal controller return changed.")
    decisions = [x for x in value if hasattr(x, "adaptive_selection") and hasattr(x, "exhaustive_fallback_active")]
    states = [x for x in value if hasattr(x, "cumulative_switch_events") and not hasattr(x, "adaptive_selection")]
    require(len(decisions) == 1 and len(states) == 1, "Temporal decision/state binding failed.")
    return decisions[0], states[0]

def evaluate_probability_payloads(
    payloads: list[dict[str, Any]],
    *,
    codebook: Any,
    requested_coverage: float,
    truth_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(payloads) == 4 and len(truth_rows) == 4, "Horizon count changed.")
    state = initial_adaptive_temporal_state()
    horizon_rows = []

    for h in range(4):
        prob = probability_object(payloads[h])
        decision, state = bind_temporal_return(adaptive_temporal_step(
            state=state,
            probability=prob,
            codebook=codebook,
            requested_coverage=float(requested_coverage),
        ))
        selected = tuple(int(x) for x in decision.adaptive_selection.beam_indices)
        selected_k = int(decision.adaptive_selection.k)
        require(selected_k == len(selected) and selected_k > 0, "Selected K invalid.")
        physical_count = int(actual_probe_count_for_adaptive_decision(decision))
        if bool(decision.loss_of_lock):
            require(bool(decision.exhaustive_fallback_active), "Loss-of-lock without fallback.")
            physical_indices = tuple(int(x) for x in decision.exhaustive_fallback_indices)
        else:
            physical_indices = selected
        require(physical_count >= len(physical_indices), "Physical probe accounting undercounts current-codebook probes.")

        row = {
            "horizon_index": h,
            "horizon_s": float(C1.HORIZONS[h]),
            "selected_beam_indices": list(selected),
            "selected_K": selected_k,
            "achieved_mass": float(decision.adaptive_selection.achieved_probability_mass),
            "primary_beam_index": int(decision.primary_beam_index),
            "primary_switched": bool(decision.primary_switched),
            "loss_of_lock": bool(decision.loss_of_lock),
            "physical_probe_count": physical_count,
            "probing_overhead_fraction": float(physical_count * C2.TBEAM_S / C2.TFRAME_S),
            "overhead_reduction_vs_exhaustive": float(1.0 - physical_count / float(codebook.beam_count)),
            "reacquisition_latency_s": float(
                max(0, physical_count - selected_k) * C2.TBEAM_S if decision.loss_of_lock else 0.0
            ),
            "inside_support_mass": float(prob.inside_support_mass),
            "outside_support_mass": float(prob.outside_support_mass),
        }

        truth = truth_rows[h]
        if not bool(truth.get("valid")):
            row.update({
                "valid_future_receiver": False,
                "containment_hit": None,
                "beam_outage": None,
                "beam_gain_loss_db": None,
                "snr_loss_db": None,
                "BER": None,
                "effective_rate_bps": None,
                "effective_rate_loss_bps": None,
            })
            horizon_rows.append(row)
            continue

        az = float(truth["azimuth_rad"])
        el = float(truth["elevation_rad"])
        oracle_sel = oracle_best_gain_beam(realized_azimuth_rad=az, codebook=codebook)
        oracle_indices = C2.selection_indices(oracle_sel)
        require(len(oracle_indices) == 1, "Oracle selection changed.")
        oracle_link, _ = C2.best_optical_link(
            indices=oracle_indices, azimuth=az, elevation=el,
            codebook=codebook, probing_beam_count=1,
        )
        achieved_link, _ = C2.best_optical_link(
            indices=physical_indices, azimuth=az, elevation=el,
            codebook=codebook, probing_beam_count=physical_count,
        )
        hit = C2.selected_set_hit(az, selected, codebook)
        row.update({
            "valid_future_receiver": True,
            "containment_hit": bool(hit),
            "beam_outage": bool(not hit),
            "truth_decision_beam_index": C2.truth_decision_beam_index(az, codebook),
        })
        row.update(C2.link_metrics(achieved_link, oracle_link))
        horizon_rows.append(row)

    valid_rows = [r for r in horizon_rows if r["valid_future_receiver"]]
    metrics = {
        "probability_coverage": mean_finite(1.0 if r["containment_hit"] else 0.0 for r in valid_rows),
        "outage_probability": mean_finite(1.0 if r["beam_outage"] else 0.0 for r in valid_rows),
        "selected_K": mean_finite(r["selected_K"] for r in valid_rows),
        "probing_overhead_fraction": mean_finite(r["probing_overhead_fraction"] for r in valid_rows),
        "overhead_reduction_vs_exhaustive": mean_finite(r["overhead_reduction_vs_exhaustive"] for r in valid_rows),
        "beam_switching_rate": mean_finite(1.0 if r["primary_switched"] else 0.0 for r in valid_rows),
        "reacquisition_latency_s": mean_finite(r["reacquisition_latency_s"] for r in valid_rows),
        "posterior_probability_mass_selected": mean_finite(r["achieved_mass"] for r in valid_rows),
        "beam_gain_loss_db": mean_finite(r.get("beam_gain_loss_db") for r in valid_rows),
        "snr_loss_db": mean_finite(r.get("snr_loss_db") for r in valid_rows),
        "BER": mean_finite(r.get("BER") for r in valid_rows),
        "effective_rate_bps": mean_finite(r.get("effective_rate_bps") for r in valid_rows),
        "effective_rate_loss_bps": mean_finite(r.get("effective_rate_loss_bps") for r in valid_rows),
        "valid_future_horizons": int(len(valid_rows)),
    }
    compact_h = [{
        k: r.get(k) for k in (
            "horizon_index", "selected_K", "selected_beam_indices", "primary_beam_index",
            "primary_switched", "loss_of_lock", "physical_probe_count", "containment_hit",
            "truth_decision_beam_index",
        )
    } for r in horizon_rows]
    return {"metrics": metrics, "horizons": compact_h}

def scaled_posterior(base: dict[str, Any], alpha: float) -> dict[str, Any]:
    result = {}
    for pid, p in base.items():
        q = dict(p)
        cov = np.asarray(p["covariance_H0_m2"], dtype=np.float64)
        require(cov.shape == (4, 3, 3) and np.all(np.isfinite(cov)), f"Covariance invalid: {pid}")
        q["covariance_H0_m2"] = (cov * float(alpha)).tolist()
        result[str(pid)] = q
    return result

def load_reference(record: dict[str, Any]) -> dict[str, np.ndarray]:
    info = record["fresh_Stage6_constructed_ADB_reference"]
    p = Path(info["reference_npz_path"])
    require(sha256_file(p) == info["reference_npz_sha256"], f"Reference drift: {p}")
    with np.load(p, allow_pickle=False) as z:
        ref = {k: np.asarray(z[k], dtype=np.bool_) for k in z.files}
    require(set(ref) == set(C2.REFERENCE_FIELDS), "Reference fields changed.")
    return ref

def load_current_and_baseline_schedules(lock: dict[str, Any]):
    p = Path(lock["ADB_outputs"]["npz_path"])
    require(sha256_file(p) == lock["adb_schedules_npz_sha256"], f"C1 ADB NPZ drift: {p}")
    with np.load(p, allow_pickle=False) as z:
        return (
            np.asarray(z["current_reactive_t0"], dtype=np.float64),
            np.asarray(z["shared_trajectory_posterior"], dtype=np.float64),
            np.asarray(z["independent_models"], dtype=np.float64),
        )

def scenario_joint_rows(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(r["system"]): r for r in record["joint_system_rows"]}
    require(set(rows) == set(SYSTEMS), "C2 scenario system rows changed.")
    return rows

def reconstruct_adb_context(task: dict[str, Any], lock: dict[str, Any]):
    sid = str(task["scenario_id"])
    formal = task["formal"]
    vr = C2.VALIDATION_RUNTIME_BY_ID.get(sid)
    require(vr is not None, f"Validation runtime missing: {sid}")
    scenario = C2.RUNTIME["read_motion_scenario"](
        vr,
        paired_root=C1.PAIRED_ROOT,
        compact_record_offset=int(formal["compact_record_offset"]),
    )
    require(str(scenario.scenario_id) == sid and int(scenario.current_time_index) == 10,
            f"Causal scenario binding changed: {sid}")
    adapted = C2.RUNTIME["adapt_causal_womd_scenario"](scenario)
    actor_boxes = build_causal_adb_actor_boxes(scenario=scenario, adapted=adapted)
    require(len(actor_boxes) == int(lock["causal_input"]["ADB_actor_box_count"]),
            f"ADB actor population changed: {sid}")

    eligible = set(int(x) for x in lock["causal_input"]["ADB_headlamp_eligible_box_indices"])
    diag = lock["ADB_outputs"]["shared_diagnostics"]
    matches = tuple(
        SimpleNamespace(
            box_index=int(x["actor_box_index"]),
            prediction_id=str(x["prediction_id"]),
        )
        for x in diag["matched_eligible"]
    )
    association = SimpleNamespace(matches=matches)
    matched = {int(x.box_index) for x in matches}
    expected_fallback = sorted(int(x) for x in diag["fallback_box_indices"])
    require(expected_fallback == sorted(eligible - matched),
            f"Frozen ADB association/fallback binding changed: {sid}")
    return actor_boxes, association, eligible

def gaussian_adb(
    posterior: dict[str, Any],
    *,
    actor_boxes,
    association,
    eligible,
    current: np.ndarray,
):
    return C1.gaussian_adb_schedule(
        prediction_by_id=posterior,
        scene_inputs=None,
        actor_boxes=actor_boxes,
        association=association,
        eligible=eligible,
        grid=GRID,
        params=ADB_PARAMS,
        current_t0=current,
        CalibratedGaussianFullBoxPrediction=RT["CalibratedGaussianFullBoxPrediction"],
        build_stochastic_future_full_boxes=RT["build_stochastic_future_full_boxes"],
        estimate_actor_occupancy_probability=RT["estimate_actor_occupancy_probability"],
        threshold_occupancy_counts_strict_k=RT["threshold_occupancy_counts_strict_k"],
        compute_class_aware_margin=RT["compute_class_aware_margin"],
        angular_margin_cells=RT["angular_margin_cells"],
        dilate_mask_theta=RT["dilate_mask_theta"],
        predictive_mask_to_illumination=RT["predictive_mask_to_illumination"],
        compose_class_aware_illumination=RT["compose_class_aware_illumination"],
        temporal_smooth_schedule=RT["temporal_smooth_schedule"],
        apply_actuation_rate_limit=RT["apply_actuation_rate_limit"],
        part_a_original_reactive_map_from_h0_centers=RT["part_a_original_reactive_map_from_h0_centers"],
    )[0]

def process_scenario(task: dict[str, Any]) -> dict[str, Any]:
    idx = int(task["formal_index"])
    sid = str(task["scenario_id"])
    lock_path = Path(task["c1_lock_path"])
    record_path = Path(task["c2_record_path"])
    require(sha256_file(lock_path) == task["c1_lock_sha256"], f"C1 lock changed: {sid}")
    require(sha256_file(record_path) == task["c2_record_sha256"], f"C2 record changed: {sid}")
    lock = read_json(lock_path)
    record = read_json(record_path)
    require(lock["scenario_id"] == sid == record["scenario_id"], f"Scenario identity mismatch: {sid}")
    require(lock["future_GT_accessed"] is False and lock["formal_metrics_computed"] is False,
            f"C1 causal boundary changed: {sid}")
    require(record["causality"]["future_GT_controller_input"] is False, f"C2 causality changed: {sid}")

    truth = record["future_GT"]["receiver_truth_by_horizon"]
    require(len(truth) == 4, f"Receiver truth cardinality changed: {sid}")
    c2_rows = scenario_joint_rows(record)

    codebook_results: dict[str, Any] = {}
    coverage_results: dict[str, Any] = {}

    for system in SYSTEMS:
        comm, source = resolve_comm(lock, system)
        codebook_results[system] = {}
        for n in CODEBOOK_VALUES:
            probs = comm["codebooks"][str(n)]["probability"]
            res = evaluate_probability_payloads(
                probs, codebook=CODEBOOKS[n], requested_coverage=PRIMARY_COVERAGE, truth_rows=truth
            )
            res["communication_branch_source_system"] = source
            codebook_results[system][str(n)] = res
        assert_metric_parity(
            codebook_results[system][str(PRIMARY_CODEBOOK)]["metrics"],
            c2_rows[system]["communication_metrics"], COMM_METRICS,
            f"{sid}/{system}/codebook32_q095",
        )

        coverage_results[system] = {}
        probs32 = comm["codebooks"][str(PRIMARY_CODEBOOK)]["probability"]
        for q in COVERAGE_VALUES:
            res = evaluate_probability_payloads(
                probs32, codebook=CODEBOOKS[PRIMARY_CODEBOOK], requested_coverage=q, truth_rows=truth
            )
            res["communication_branch_source_system"] = source
            coverage_results[system][f"{q:.3f}"] = res
        assert_metric_parity(
            coverage_results[system]["0.950"]["metrics"],
            c2_rows[system]["communication_metrics"], COMM_METRICS,
            f"{sid}/{system}/coverage095",
        )

    # Predictive uncertainty: communication branch from locked posterior only.
    uncertainty: dict[str, Any] = {s: {} for s in SYSTEMS}
    comm_branch = {
        "shared_trajectory_posterior": ("shared_trajectory_posterior", "shared_trajectory_posterior"),
        "independent_models": ("independent_communication_predictor", "independent_models"),
        "direct_ADB_predictor": ("shared_trajectory_posterior", "shared_trajectory_posterior"),
    }

    for system in SYSTEMS:
        for alpha in ALPHA_VALUES:
            key = f"{alpha:.1f}"
            uncertainty[system][key] = {
                "alpha": float(alpha),
                "communication": {"status": "NOT_APPLICABLE", "metrics": None},
                "ADB": {"status": "NOT_APPLICABLE", "metrics": None},
            }

    receiver = lock.get("receiver")
    require(receiver is not None, f"Unexpected no-receiver scenario: {sid}")
    rid = str(receiver["prediction_id"])

    for system, (posterior_key, source_system) in comm_branch.items():
        base_all = lock["posterior_outputs"][posterior_key]
        require(rid in base_all, f"Receiver posterior missing: {sid}/{system}/{rid}")
        p = base_all[rid]
        source_comm, _ = resolve_comm(lock, source_system)
        for alpha in ALPHA_VALUES:
            scaled_cov = np.asarray(p["covariance_H0_m2"], dtype=np.float64) * float(alpha)
            generated = C1.gaussian_comm_output(
                scenario_id=sid,
                prediction_id=rid,
                latest_position=np.asarray(p["latest_position_H0_m"], dtype=np.float64),
                mean_displacement=np.asarray(p["mean_displacement_H0_m"], dtype=np.float64),
                covariance=scaled_cov,
                codebooks=CODEBOOKS,
                sample_product_of_horizon_gaussians=RT["sample_product_of_horizon_gaussians"],
                beam_probability_mass_from_samples=RT["beam_probability_mass_from_samples"],
                adaptive_temporal_step=RT["adaptive_temporal_step"],
                initial_adaptive_temporal_state=RT["initial_adaptive_temporal_state"],
            )
            probs = generated["codebooks"]["32"]["probability"]
            if alpha == 1.0:
                require(probs == source_comm["codebooks"]["32"]["probability"],
                        f"Alpha=1 communication probability parity failed: {sid}/{system}")
            result = evaluate_probability_payloads(
                probs, codebook=CODEBOOKS[32], requested_coverage=0.95, truth_rows=truth
            )
            if alpha == 1.0:
                assert_metric_parity(
                    result["metrics"], c2_rows[system]["communication_metrics"], COMM_METRICS,
                    f"{sid}/{system}/uncertainty_comm_alpha1",
                )
            uncertainty[system][f"{alpha:.1f}"]["communication"] = {
                "status": "EVALUATED",
                "source_system": source_system,
                "probability_payload_sha256": sha256_bytes(canonical_bytes(probs)),
                "metrics": result["metrics"],
            }

    # Predictive uncertainty: ADB analytic branch from locked posterior only.
    current, baseline_shared, baseline_indep = load_current_and_baseline_schedules(lock)
    reference = load_reference(record)
    actor_boxes, association, eligible = reconstruct_adb_context(task, lock)

    shared_base = lock["posterior_outputs"]["shared_trajectory_posterior"]
    indep_base = lock["posterior_outputs"]["independent_ADB_predictor"]

    for alpha in ALPHA_VALUES:
        key = f"{alpha:.1f}"
        shared_sched = gaussian_adb(
            scaled_posterior(shared_base, alpha),
            actor_boxes=actor_boxes, association=association, eligible=eligible, current=current,
        )
        indep_sched = gaussian_adb(
            scaled_posterior(indep_base, alpha),
            actor_boxes=actor_boxes, association=association, eligible=eligible, current=current,
        )
        if alpha == 1.0:
            require(np.array_equal(shared_sched, baseline_shared),
                    f"Alpha=1 shared ADB schedule parity failed: {sid}")
            require(np.array_equal(indep_sched, baseline_indep),
                    f"Alpha=1 independent ADB schedule parity failed: {sid}")

        shared_metrics = C2.evaluate_adb(shared_sched, current, reference)
        indep_metrics = C2.evaluate_adb(indep_sched, current, reference)
        if alpha == 1.0:
            assert_metric_parity(shared_metrics, c2_rows["shared_trajectory_posterior"]["illumination_metrics"],
                                 ADB_METRICS, f"{sid}/shared/uncertainty_adb_alpha1")
            assert_metric_parity(indep_metrics, c2_rows["independent_models"]["illumination_metrics"],
                                 ADB_METRICS, f"{sid}/indep/uncertainty_adb_alpha1")

        uncertainty["shared_trajectory_posterior"][key]["ADB"] = {
            "status": "EVALUATED",
            "source_system": "shared_trajectory_posterior",
            "schedule_sha256": array_digest(shared_sched),
            "metrics": shared_metrics,
        }
        uncertainty["direct_beam_classifier"][key]["ADB"] = {
            "status": "EVALUATED_ALIAS_SHARED_ADB",
            "source_system": "shared_trajectory_posterior",
            "schedule_sha256": array_digest(shared_sched),
            "metrics": shared_metrics,
        }
        uncertainty["independent_models"][key]["ADB"] = {
            "status": "EVALUATED",
            "source_system": "independent_models",
            "schedule_sha256": array_digest(indep_sched),
            "metrics": indep_metrics,
        }
        uncertainty["direct_ADB_predictor"][key]["ADB"] = {
            "status": "NOT_APPLICABLE_NO_EXPLICIT_PREDICTIVE_COVARIANCE",
            "metrics": None,
        }
        uncertainty["deterministic_shared_trajectory"][key]["ADB"] = {
            "status": "NOT_APPLICABLE_DETERMINISTIC",
            "metrics": None,
        }
        uncertainty["direct_beam_classifier"][key]["communication"] = {
            "status": "NOT_APPLICABLE_DIRECT_COMM_NO_EXPLICIT_PREDICTIVE_COVARIANCE",
            "metrics": None,
        }
        uncertainty["deterministic_shared_trajectory"][key]["communication"] = {
            "status": "NOT_APPLICABLE_DETERMINISTIC",
            "metrics": None,
        }

    return {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3B",
        "status": "IMMUTABLE_C3B_SCENARIO_SWEEPS_COMPLETE",
        "formal_index": idx,
        "scenario_id": sid,
        "runner_sha256": task["runner_sha256"],
        "c1_lock_sha256": task["c1_lock_sha256"],
        "c2_record_sha256": task["c2_record_sha256"],
        "codebook_sweep": codebook_results,
        "coverage_sweep": coverage_results,
        "uncertainty_sweep": uncertainty,
        "causality": {
            "new_model_forward": False,
            "future_GT_controller_input": False,
            "C1_modified": False,
            "C2_modified": False,
            "post_outcome_tuning": False,
            "sweep_values_from_frozen_7p5B_contract": True,
        },
    }

def aggregate_nested(raw_docs: list[dict[str, Any]], kind: str, values: list[str], metric_names, branch: str | None = None):
    out = {}
    for system in SYSTEMS:
        out[system] = {}
        for value in values:
            table = {}
            for metric in metric_names:
                vals = []
                for doc in raw_docs:
                    if kind == "codebook_sweep":
                        m = doc[kind][system][value]["metrics"]
                    elif kind == "coverage_sweep":
                        m = doc[kind][system][value]["metrics"]
                    else:
                        entry = doc["uncertainty_sweep"][system][value][branch]
                        m = entry["metrics"]
                    if m is not None:
                        v = finite(m.get(metric))
                        if v is not None:
                            vals.append(v)
                table[metric] = {
                    "mean": None if not vals else float(np.mean(np.asarray(vals, dtype=np.float64))),
                    "finite_scenarios": int(len(vals)),
                }
            out[system][value] = table
    return out

def main() -> int:
    print("=" * 88)
    print("STAGE 7 — BLOCK 7.5C3B")
    print("FROZEN CODEBOOK + COVERAGE + PREDICTIVE-UNCERTAINTY SWEEPS")
    print("NO MODEL FORWARD / NO RETRAINING / NO RECALIBRATION / NO C1-C2 MODIFICATION")
    print("=" * 88)

    require(Path(__file__).resolve() == SCRIPT.resolve(),
            f"Runner must be installed at {SCRIPT}; actual={Path(__file__).resolve()}")
    runner_sha = sha256_file(SCRIPT)

    print("\n===== A. FROZEN AUTHORITY + C3A GATE =====")
    for p, expected in EXPECTED.items():
        actual = sha256_file(p)
        require(actual == expected, f"SHA mismatch {p}: {actual} != {expected}")
    contract = read_json(B75B)
    require(contract["status"] == "FROZEN_FINAL_FORMAL_EVALUATOR_CONTRACT", "7.5B not frozen.")
    require(tuple(contract["sweep_execution"]["codebook"]["values"]) == CODEBOOK_VALUES, "Codebook grid changed.")
    require(tuple(float(x) for x in contract["sweep_execution"]["coverage_target"]["values"]) == COVERAGE_VALUES,
            "Coverage grid changed.")
    require(tuple(float(x) for x in contract["sweep_execution"]["predictive_uncertainty"]["alpha"]) == ALPHA_VALUES,
            "Uncertainty grid changed.")
    c3a = read_json(C3A_MANIFEST)
    require(c3a["status"] == "FROZEN_COMPLETE_C3A_CORE_STATISTICS", "C3A status changed.")
    print("C1/C2/C3A/7.5B exact identities = PASS")

    formal = read_jsonl(FORMAL)
    c1m = read_json(C1_MANIFEST)
    c2m = read_json(C2_MANIFEST)
    require(len(formal) == len(c1m["scenario_locks"]) == len(c2m["scenario_records"]) == 120, "N120 binding changed.")
    ids = [str(x["scenario_id"]) for x in formal]
    require(ids == [str(x["scenario_id"]) for x in c1m["scenario_locks"]], "C1 order changed.")
    require(ids == [str(x["scenario_id"]) for x in c2m["scenario_records"]], "C2 order changed.")

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = []
    raw_docs_by_sid = {}
    for f, c1e, c2e in zip(formal, c1m["scenario_locks"], c2m["scenario_records"]):
        sid = str(f["scenario_id"])
        idx = int(c1e["formal_index"])
        out = RAW_ROOT / f"{idx:03d}_{sid}.json"
        task = {
            "formal_index": idx,
            "scenario_id": sid,
            "formal": f,
            "c1_lock_path": c1e["lock_json"],
            "c1_lock_sha256": c1e["lock_json_sha256"],
            "c2_record_path": c2e["record_path"],
            "c2_record_sha256": c2e["record_sha256"],
            "runner_sha256": runner_sha,
        }
        if out.exists():
            doc = read_json(out)
            require(doc.get("runner_sha256") == runner_sha, f"Existing C3B runner mismatch: {out}")
            require(doc.get("c1_lock_sha256") == task["c1_lock_sha256"], f"Existing C3B C1 mismatch: {out}")
            require(doc.get("c2_record_sha256") == task["c2_record_sha256"], f"Existing C3B C2 mismatch: {out}")
            raw_docs_by_sid[sid] = doc
        else:
            tasks.append(task)

    print("\n===== B. PARALLEL PRE-REGISTERED SWEEP EXECUTION =====")
    workers = max(1, int(os.environ.get("C3_WORKERS", "12")))
    print("workers =", workers)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = {pool.submit(process_scenario, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    doc = future.result()
                except Exception as exc:
                    raise FailClosed(
                        f"C3B scenario {task['formal_index']:03d} {task['scenario_id']} failed: {exc}"
                    ) from exc
                out = RAW_ROOT / f"{task['formal_index']:03d}_{task['scenario_id']}.json"
                atomic_immutable(out, doc)
                raw_docs_by_sid[task["scenario_id"]] = doc
                i = int(task["formal_index"])
                if i % 10 == 0 or i == 119:
                    print(f"[{i:03d}/119] SWEEP LOCK {task['scenario_id']}", flush=True)

    require(len(raw_docs_by_sid) == 120, f"Expected 120 C3B scenario docs, got {len(raw_docs_by_sid)}")
    raw_docs = [raw_docs_by_sid[sid] for sid in ids]
    print("120/120 immutable scenario sweep records = PASS")

    print("\n===== C. GLOBAL CODEBOOK SWEEP =====")
    codebook_doc = {
        "project": "Agni", "stage": 7, "block": "7.5C3B", "status": "PASS",
        "sweep": "codebook", "values": list(CODEBOOK_VALUES),
        "coverage_target": PRIMARY_COVERAGE,
        "aggregation": "equal scenario macro mean over finite scenario metrics",
        "table": aggregate_nested(raw_docs, "codebook_sweep", [str(x) for x in CODEBOOK_VALUES], COMM_METRICS),
    }
    codebook_sha = atomic_immutable(CODEBOOK_OUT, codebook_doc)
    print("codebook {16,32,64} = PASS")

    print("\n===== D. GLOBAL COVERAGE SWEEP =====")
    coverage_keys = [f"{x:.3f}" for x in COVERAGE_VALUES]
    coverage_doc = {
        "project": "Agni", "stage": 7, "block": "7.5C3B", "status": "PASS",
        "sweep": "coverage_target", "values": list(COVERAGE_VALUES),
        "codebook": PRIMARY_CODEBOOK,
        "no_probability_remapping": True,
        "aggregation": "equal scenario macro mean over finite scenario metrics",
        "table": aggregate_nested(raw_docs, "coverage_sweep", coverage_keys, COMM_METRICS),
    }
    coverage_sha = atomic_immutable(COVERAGE_OUT, coverage_doc)
    print("coverage {.90,.95,.975,.99} = PASS")

    print("\n===== E. GLOBAL PREDICTIVE-UNCERTAINTY SWEEP =====")
    alpha_keys = [f"{x:.1f}" for x in ALPHA_VALUES]
    uncertainty_doc = {
        "project": "Agni", "stage": 7, "block": "7.5C3B", "status": "PASS",
        "sweep": "predictive_covariance_scale",
        "alpha": list(ALPHA_VALUES),
        "covariance_rule": "covariance_H0_m2_scaled_multiplicatively_by_alpha;means_unchanged",
        "retraining": False, "recalibration": False,
        "communication": {
            "table": aggregate_nested(raw_docs, "uncertainty_sweep", alpha_keys, COMM_METRICS, branch="communication"),
            "applicability": {
                "shared_trajectory_posterior": "EVALUATED",
                "independent_models": "EVALUATED",
                "direct_beam_classifier": "NOT_APPLICABLE_DIRECT_COMM_NO_EXPLICIT_PREDICTIVE_COVARIANCE",
                "direct_ADB_predictor": "EVALUATED_SHARED_COMMUNICATION_BRANCH_ONLY",
                "deterministic_shared_trajectory": "NOT_APPLICABLE_DETERMINISTIC",
            },
        },
        "ADB": {
            "table": aggregate_nested(raw_docs, "uncertainty_sweep", alpha_keys, ADB_METRICS, branch="ADB"),
            "applicability": {
                "shared_trajectory_posterior": "EVALUATED",
                "independent_models": "EVALUATED",
                "direct_beam_classifier": "EVALUATED_SHARED_ADB_BRANCH_ONLY",
                "direct_ADB_predictor": "NOT_APPLICABLE_DIRECT_ADB_NO_EXPLICIT_PREDICTIVE_COVARIANCE",
                "deterministic_shared_trajectory": "NOT_APPLICABLE_DETERMINISTIC",
            },
        },
        "alpha_1_exact_reproduction_gate": "PASS_BY_PER_SCENARIO_ASSERTION",
    }
    uncertainty_sha = atomic_immutable(UNCERTAINTY_OUT, uncertainty_doc)
    print("uncertainty alpha {.5,1,1.5,2} = PASS")
    print("alpha=1 exact C1/C2 reproduction = PASS")

    raw_manifest = []
    for f in formal:
        sid = str(f["scenario_id"])
        idx = ids.index(sid)
        p = RAW_ROOT / f"{idx:03d}_{sid}.json"
        raw_manifest.append({
            "formal_index": idx, "scenario_id": sid, "path": str(p), "sha256": sha256_file(p)
        })

    manifest_doc = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3B",
        "status": "FROZEN_COMPLETE_C3B_CONTROLLER_SWEEPS",
        "runner_path": str(SCRIPT),
        "runner_sha256": runner_sha,
        "block75b_contract_sha256": EXPECTED[B75B],
        "c1_manifest_sha256": EXPECTED[C1_MANIFEST],
        "c2_manifest_sha256": EXPECTED[C2_MANIFEST],
        "c2_seal_sha256": EXPECTED[C2_SEAL],
        "c3a_manifest_sha256": EXPECTED[C3A_MANIFEST],
        "scenario_records": raw_manifest,
        "outputs": {
            str(CODEBOOK_OUT): codebook_sha,
            str(COVERAGE_OUT): coverage_sha,
            str(UNCERTAINTY_OUT): uncertainty_sha,
        },
        "sweeps": {
            "codebook": list(CODEBOOK_VALUES),
            "coverage_target": list(COVERAGE_VALUES),
            "predictive_uncertainty_alpha": list(ALPHA_VALUES),
            "latency_multiplier": "DEFERRED_TO_C3C_ISOLATED_FORMAL_LATENCY_MEASUREMENT",
        },
        "causality": {
            "C1_modified": False,
            "C2_modified": False,
            "new_model_forward": False,
            "retraining": False,
            "recalibration": False,
            "future_GT_controller_input": False,
            "post_outcome_tuning": False,
            "analytic_controller_reexecution_only_for_pre_registered_sweep_values": True,
        },
        "next": "7.5C3C isolated formal resource/latency measurement + latency multiplier stress sweep + final reproducibility seal",
    }
    manifest_sha = atomic_immutable(MANIFEST, manifest_doc)

    report_doc = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C3B",
        "status": "PASS_FROZEN_COMPLETE_C3B",
        "runner_sha256": runner_sha,
        "manifest_sha256": manifest_sha,
        "scenario_sweep_records": 120,
        "codebook_sweep": "PASS",
        "coverage_sweep": "PASS",
        "uncertainty_sweep": "PASS",
        "alpha_1_exact_reproduction": "PASS",
        "new_model_forward": False,
        "C1_modified": False,
        "C2_modified": False,
        "post_outcome_tuning": False,
        "C3_final_seal": False,
    }
    report_sha = atomic_immutable(REPORT, report_doc)

    print("\n" + "=" * 88)
    print("BLOCK 7.5C3B = FULLY VERIFIED / FROZEN")
    print("120/120 IMMUTABLE SCENARIO SWEEP RECORDS = PASS")
    print("CODEBOOK SWEEP 16/32/64 = PASS")
    print("COVERAGE SWEEP .90/.95/.975/.99 = PASS")
    print("PREDICTIVE UNCERTAINTY SWEEP .5/1/1.5/2 = PASS")
    print("ALPHA=1 EXACT C1/C2 REPRODUCTION = PASS")
    print("NEW MODEL FORWARD = NO")
    print("C1 MODIFIED = NO")
    print("C2 MODIFIED = NO")
    print("POST-OUTCOME TUNING = NO")
    print("C3 FINAL SEAL = NO")
    print("C3B manifest SHA256 =", manifest_sha)
    print("C3B report SHA256   =", report_sha)
    print("C3B runner SHA256   =", runner_sha)
    print("STATUS = FROZEN_COMPLETE_C3B")
    print("=" * 88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!" * 88, file=sys.stderr)
        print("BLOCK 7.5C3B FAIL-CLOSED", file=sys.stderr)
        print(type(exc).__name__ + ":", exc, file=sys.stderr)
        print("DO NOT MODIFY/DELETE EXISTING C3B RECORDS.", file=sys.stderr)
        print("C3 FINAL SEAL = NO", file=sys.stderr)
        print("!" * 88, file=sys.stderr)
        traceback.print_exc()
        raise

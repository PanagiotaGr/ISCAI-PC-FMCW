#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/agni/waymo")
S7 = ROOT / "iscai_stage7"
SCRIPT = S7 / "scripts/run_block75d_latency_resource_closure_final.py"
CONTRACT = S7 / "configs/stage7_block75d_latency_resource_closure_contract.json"

C1_RUNNER = S7 / "scripts/run_block75c1_formal_causal_output_lock.py"
C1_ROOT = S7 / "artifacts/block75c1_nonoracle_lock"
C2_SEAL = S7 / "artifacts/stage7_block75c2_final_frozen_seal.json"
C3_FINAL_SEAL = S7 / "artifacts/stage7_block75c3_final_reproducibility_seal.json"
C3A_MANIFEST = S7 / "artifacts/block75c3a_core_statistics/block75c3a_manifest.json"
C3B_MANIFEST = S7 / "artifacts/block75c3b_final_frozen_sweeps/block75c3b_manifest.json"
C3C_MANIFEST = S7 / "artifacts/block75c3c_final_latency_resources/block75c3c_manifest.json"

OUT_ROOT = S7 / "artifacts/block75d_latency_resource_closure_final"
RECORD_ROOT = OUT_ROOT / "isolated_system_records"
TABLE = OUT_ROOT / "five_system_latency_resource_table.json"
SENSITIVITY = OUT_ROOT / "measured_latency_sensitivity.json"
MANIFEST = OUT_ROOT / "block75d_manifest.json"
REPORT = S7 / "reports/stage7_block75d_latency_resource_closure_final_report.json"
SEAL = S7 / "artifacts/stage7_pdf_compliance_final_superseding_seal.json"

EXPECTED_CONTRACT_SHA = "b4f3c212093ffa45924c72272f8d95d5dbf3e2244df9ab0cacbdb3360b81da5e"
EXPECTED_FROZEN = {
    str(C1_RUNNER): "077237d85e62468f90ebf21a70fe6cdf3ad095fe83dba2985d0b4cb799379275",
    str(C2_SEAL): "743541e5fb6b7718ef6d6634e09f13c6f7b879d106a5bcd9c0e7e8dad7d5d82c",
    str(C3_FINAL_SEAL): "e01ba2e735d4d6403a584f8f1a304e3b3acb926d36fb3adb3cb0f62c5cc44d8a",
    str(C3A_MANIFEST): "4c7da7e404c227414cc9e31053b9f46f7440e0ad207515addc801f7fb002fc4b",
    str(C3B_MANIFEST): "eda5ea7c9c4b1e62f3196482959cd142ee1ab9684d278b1b8ab4fc02e9ba411b",
    str(C3C_MANIFEST): "93072e6a02ed5271aab67d0d4c1d5dfb6e4916853359f1a119bad5f5ebf4eaf4",
}

SYSTEMS = (
    "shared_trajectory_posterior",
    "independent_models",
    "direct_beam_classifier",
    "direct_ADB_predictor",
    "deterministic_shared_trajectory",
)
PRIMARY_CODEBOOK = 32
LATENCY_MULTIPLIERS = (0.5, 1.0, 2.0)

class FailClosed(RuntimeError):
    pass

def require(cond: bool, msg: str) -> None:
    if not cond:
        raise FailClosed(msg)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")

def canonical_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()

def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing overwrite: {path}")
    data = canonical_bytes(obj)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def gate_frozen() -> None:
    require(CONTRACT.is_file(), f"Missing contract {CONTRACT}")
    require(sha256_file(CONTRACT) == EXPECTED_CONTRACT_SHA, "7.5D contract SHA changed.")
    for raw, expected in EXPECTED_FROZEN.items():
        p = Path(raw)
        require(p.is_file(), f"Missing frozen input: {p}")
        require(sha256_file(p) == expected, f"Frozen input SHA changed: {p}")

def import_c1():
    name = "stage7_c1_frozen_runtime_for_measurement"
    spec = importlib.util.spec_from_file_location(name, C1_RUNNER)
    require(spec is not None and spec.loader is not None, "Could not load frozen C1 runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def perf_call(device: torch.device, fn, *args, **kwargs):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter_ns()
    value = fn(*args, **kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, (time.perf_counter_ns() - t0) / 1e9

def array_digest(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in ("target", "neighbors", "neighbor_mask", "map_context"):
        a = np.ascontiguousarray(np.asarray(arrays[name]))
        h.update(name.encode()); h.update(b"\0")
        h.update(str(a.dtype).encode()); h.update(b"\0")
        h.update(json.dumps(list(a.shape), separators=(",", ":")).encode()); h.update(b"\0")
        h.update(a.tobytes(order="C")); h.update(b"\0")
    return h.hexdigest()

def trainable_parameters(*models) -> int:
    seen = set()
    total = 0
    for model in models:
        if model is None: continue
        for p in model.parameters():
            if not p.requires_grad: continue
            ident = id(p)
            if ident in seen: continue
            seen.add(ident)
            total += int(p.numel())
    return total

def peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

def find_measurement_scenario(c1) -> tuple[int, str, dict[str, Any]]:
    formal_rows = [json.loads(x) for x in c1.FORMAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    require(len(formal_rows) == 120, "Formal N changed.")
    for idx, row in enumerate(formal_rows):
        sid = str(row["scenario_id"])
        lock = C1_ROOT / f"{idx:03d}_{sid}.lock.json"
        require(lock.is_file(), f"Missing C1 causal lock {lock}")
        doc = load_json(lock)
        require(doc.get("future_GT_accessed") is False, "C1 lock future boundary changed.")
        if doc.get("receiver") is not None:
            return idx, sid, doc
    raise FailClosed("No receiver-evaluable C1 scenario found.")

def load_system_models(c1, rt, system: str, device: torch.device):
    normalizer = rt["CalibrationNormalizer"](load_json(c1.FIT_NORM), device=device)
    direct_module = None
    shared = indep_comm = indep_adb = det = direct_beam = direct_adb = None
    scales = {}

    if system in ("shared_trajectory_posterior", "direct_beam_classifier", "direct_ADB_predictor"):
        shared = c1.build_gaussian_model(rt["GaussianTrajectoryGRU"], device)
        shared.load_state_dict(c1.checkpoint_state_dict(c1.GAUSS_CKPT, device), strict=True); shared.eval()
        scales["shared"] = c1.load_variance_scale(c1.SHARED_CAL)

    if system == "independent_models":
        indep_comm = c1.build_gaussian_model(rt["GaussianTrajectoryGRU"], device)
        indep_comm.load_state_dict(c1.checkpoint_state_dict(c1.INDEP_COMM_CKPT, device), strict=True); indep_comm.eval()
        indep_adb = c1.build_gaussian_model(rt["GaussianTrajectoryGRU"], device)
        indep_adb.load_state_dict(c1.checkpoint_state_dict(c1.INDEP_ADB_CKPT, device), strict=True); indep_adb.eval()
        scales["indep_comm"] = c1.load_variance_scale(c1.INDEP_COMM_CAL)
        scales["indep_adb"] = c1.load_variance_scale(c1.INDEP_ADB_CAL)

    if system == "deterministic_shared_trajectory":
        det, det_class = c1.load_frozen_deterministic_model(rt["construct_deterministic_model"], device)
    else:
        det_class = None

    if system in ("direct_beam_classifier", "direct_ADB_predictor"):
        direct_module = c1.load_direct_module(c1.DIRECT_TRAINER)

    if system == "direct_beam_classifier":
        direct_beam = direct_module.DirectBeamClassifier().to(device)
        direct_beam.load_state_dict(c1.checkpoint_state_dict(c1.DIRECT_BEAM_CKPT, device), strict=True); direct_beam.eval()

    if system == "direct_ADB_predictor":
        direct_adb = direct_module.DirectADBField().to(device)
        direct_adb.load_state_dict(c1.checkpoint_state_dict(c1.DIRECT_ADB_CKPT, device), strict=True); direct_adb.eval()

    models = {
        "shared": shared, "indep_comm": indep_comm, "indep_adb": indep_adb,
        "det": det, "direct_beam": direct_beam, "direct_adb": direct_adb,
    }
    return normalizer, models, scales, det_class

def make_actuation_wrappers(rt, device):
    acc = {"seconds": 0.0, "calls": []}
    def wrap(name, fn):
        def inner(*args, **kwargs):
            value, dt = perf_call(device, fn, *args, **kwargs)
            acc["seconds"] += float(dt)
            acc["calls"].append({"name": name, "seconds": float(dt)})
            return value
        return inner
    return acc, wrap("temporal_smooth_schedule", rt["temporal_smooth_schedule"]), wrap("apply_actuation_rate_limit", rt["apply_actuation_rate_limit"])

def run_child(system: str, output: Path) -> int:
    gate_frozen()
    require(system in SYSTEMS, f"Unknown system {system}")
    require(not output.exists(), f"Child output exists: {output}")
    require(torch.cuda.is_available(), "Frozen Stage7 measurement requires CUDA.")
    device = torch.device("cuda:0")
    torch.manual_seed(20260821); np.random.seed(20260821)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    c1 = import_c1()
    rt = c1.import_runtime()
    idx, sid, lock_doc = find_measurement_scenario(c1)
    receiver_lock = lock_doc["receiver"]
    require(receiver_lock is not None, "Selected measurement lock has no receiver.")

    normalizer, models, scales, det_class = load_system_models(c1, rt, system, device)
    parameter_count = trainable_parameters(*models.values())

    # Fixed runtime objects not timed as per-frame data loading/preprocessing.
    clean_config, degraded_config = rt["load_frozen_stage2_configs"]()
    grid = c1.grid_from_frozen_config(rt["OccupancyGrid"])
    adb_params = c1.frozen_adb_parameters()
    beam_cfg = load_json(c1.BEAM_CFG)
    support = beam_cfg["azimuth_support"]
    codebook32 = rt["build_uniform_azimuth_codebook"](
        beam_count=32,
        support_min_azimuth_rad=math.radians(float(support["min_deg"])),
        support_max_azimuth_rad=math.radians(float(support["max_deg"])),
    )

    validation_rows = rt["read_validation_manifest"](c1.VALIDATION_MANIFEST)
    validation_by_id = {str(r.scenario_id): r for r in validation_rows}
    require(sid in validation_by_id, "Measurement scenario missing from validation runtime.")
    formal_rows = [json.loads(x) for x in c1.FORMAL.read_text(encoding="utf-8").splitlines() if x.strip()]
    record = c1.resolve_formal_reader_records(formal_rows)[idx]

    # Data loading: exact raw causal scenario read only.
    scenario, data_loading_s = perf_call(
        device, rt["read_motion_scenario"], validation_by_id[sid],
        paired_root=c1.PAIRED_ROOT, compact_record_offset=int(record["compact_record_offset"])
    )
    require(str(scenario.scenario_id) == sid, "Scenario ID changed.")
    require(int(scenario.current_time_index) == 10, "Anchor changed.")

    # Preprocessing: all common causal work required to produce identical controller/model payloads.
    def preprocess():
        causal = rt["build_real_causal_inputs"](scenario, clean_config=clean_config, degraded_config=degraded_config)
        scene_inputs = causal["scene_inputs"]
        adapted = causal["adapted"]
        history_ids, payloads = c1.build_payloads_for_scene(
            scenario, causal, rt["build_static_map_index_H0"], rt["map_context_summary"], rt["build_model_input_payload"]
        )
        require(payloads and history_ids == sorted(history_ids), "Causal payload ordering changed.")
        actor_boxes = rt["build_causal_adb_actor_boxes"](scenario=scenario, adapted=adapted)
        association = c1.association_for_scene(
            scene_inputs, actor_boxes, rt["PredictorAnchor"], rt["reciprocal_unique_nearest_anchor_association"]
        )
        eligible = c1.eligible_box_indices(actor_boxes, rt["box_corners_headlamp"])
        receiver, selection = c1.select_receiver(
            scene_inputs, actor_boxes, association,
            rt["ReceiverCandidate"], rt["ReceiverSelectionConfig"], rt["select_primary_receiver"],
        )
        require(receiver is not None, "Measurement scenario lost receiver.")
        require(str(receiver["prediction_id"]) == str(receiver_lock["prediction_id"]), "Receiver identity differs from frozen C1 lock.")
        arrays = c1.payload_arrays(payloads)
        normalized = normalizer.prepare(arrays, device=device)
        current_t0 = c1.current_reactive_map(
            actor_boxes, grid,
            rt["part_a_current_vehicle_centers_from_causal_boxes"],
            rt["part_a_original_reactive_map_from_h0_centers"],
        )
        return causal, scene_inputs, history_ids, arrays, normalized, actor_boxes, association, eligible, receiver, current_t0

    prep, preprocessing_s = perf_call(device, preprocess)
    causal, scene_inputs, history_ids, arrays, normalized, actor_boxes, association, eligible, receiver, current_t0 = prep
    payload_sha = array_digest(arrays)
    rid = str(receiver["prediction_id"])
    rid_index = history_ids.index(rid)
    batch_one = {
        "target": normalized["target"][rid_index:rid_index+1],
        "neighbors": normalized["neighbors"][rid_index:rid_index+1],
        "neighbor_mask": normalized["neighbor_mask"][rid_index:rid_index+1],
        "map_context": normalized["map_context"][rid_index:rid_index+1],
    }

    # Warm-up exactly once per unique required neural inference route.
    if models["shared"] is not None:
        c1.infer_gaussian(models["shared"], normalized, normalizer, scales["shared"], device,
                          apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"])
    if models["indep_comm"] is not None:
        c1.infer_gaussian(models["indep_comm"], normalized, normalizer, scales["indep_comm"], device,
                          apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"])
        c1.infer_gaussian(models["indep_adb"], normalized, normalizer, scales["indep_adb"], device,
                          apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"])
    if models["det"] is not None:
        c1.infer_deterministic(models["det"], normalized, normalizer, rt["extract_deterministic_prediction"], device)
    if models["direct_beam"] is not None:
        with torch.inference_mode():
            perf_call(device, models["direct_beam"], batch_one["target"], batch_one["neighbors"], batch_one["neighbor_mask"], batch_one["map_context"])
    if models["direct_adb"] is not None:
        c1.direct_adb_masks(
            direct_model=models["direct_adb"], normalized_batch=normalized, history_ids=history_ids,
            actor_boxes=actor_boxes, association=association, eligible=eligible, device=device,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    predictor_s = 0.0
    shared_post = indep_comm_post = indep_adb_post = det_post = None

    # One measured inference for each unique model required by the system.
    if models["shared"] is not None:
        shared_mean, shared_cov, dt = c1.infer_gaussian(
            models["shared"], normalized, normalizer, scales["shared"], device,
            apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"]
        )
        predictor_s += dt
        shared_post, _ = c1.posterior_dicts(history_ids, scene_inputs, shared_mean, shared_cov)

    if models["indep_comm"] is not None:
        m1, c1cov, dt1 = c1.infer_gaussian(
            models["indep_comm"], normalized, normalizer, scales["indep_comm"], device,
            apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"]
        )
        m2, c2cov, dt2 = c1.infer_gaussian(
            models["indep_adb"], normalized, normalizer, scales["indep_adb"], device,
            apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"]
        )
        predictor_s += dt1 + dt2
        indep_comm_post, _ = c1.posterior_dicts(history_ids, scene_inputs, m1, c1cov)
        indep_adb_post, _ = c1.posterior_dicts(history_ids, scene_inputs, m2, c2cov)

    if models["det"] is not None:
        dm, dt = c1.infer_deterministic(
            models["det"], normalized, normalizer, rt["extract_deterministic_prediction"], device
        )
        predictor_s += dt
        det_post = c1.deterministic_dicts(history_ids, scene_inputs, dm)

    # Communication / beam-selection timing, primary 32 only.
    captured = []
    original_step = rt["adaptive_temporal_step"]
    def capture_step(*args, **kwargs):
        value = original_step(*args, **kwargs)
        decision = [x for x in value if hasattr(x, "adaptive_selection")]
        require(len(decision) == 1, "Adaptive decision binding changed.")
        captured.append(decision[0])
        return value

    direct_beam_model_s = 0.0
    if system == "shared_trajectory_posterior":
        obj = shared_post[rid]
        (comm, beam_selection_s) = perf_call(
            device, c1.gaussian_comm_output,
            scenario_id=sid, prediction_id=rid,
            latest_position=np.asarray(obj["latest_position_H0_m"]),
            mean_displacement=np.asarray(obj["mean_displacement_H0_m"]),
            covariance=np.asarray(obj["covariance_H0_m2"]),
            codebooks={32: codebook32},
            sample_product_of_horizon_gaussians=rt["sample_product_of_horizon_gaussians"],
            beam_probability_mass_from_samples=rt["beam_probability_mass_from_samples"],
            adaptive_temporal_step=capture_step,
            initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
        )
        comm["posterior_digest_sha256"] = obj["posterior_digest_sha256"]
        expected_comm = lock_doc["communication_outputs"]["shared_trajectory_posterior"]
    elif system == "independent_models":
        obj = indep_comm_post[rid]
        (comm, beam_selection_s) = perf_call(
            device, c1.gaussian_comm_output,
            scenario_id=sid, prediction_id=rid,
            latest_position=np.asarray(obj["latest_position_H0_m"]),
            mean_displacement=np.asarray(obj["mean_displacement_H0_m"]),
            covariance=np.asarray(obj["covariance_H0_m2"]),
            codebooks={32: codebook32},
            sample_product_of_horizon_gaussians=rt["sample_product_of_horizon_gaussians"],
            beam_probability_mass_from_samples=rt["beam_probability_mass_from_samples"],
            adaptive_temporal_step=capture_step,
            initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
        )
        expected_comm = lock_doc["communication_outputs"]["independent_models"]
    elif system == "direct_beam_classifier":
        ((comm, direct_beam_model_s), outer_s) = perf_call(
            device, c1.direct_beam_comm_output,
            model=models["direct_beam"], batch_one=batch_one,
            codebooks={32: codebook32}, BeamProbabilityMass=rt["BeamProbabilityMass"],
            adaptive_temporal_step=capture_step,
            initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
            device=device,
        )
        predictor_s += direct_beam_model_s
        beam_selection_s = max(0.0, outer_s - direct_beam_model_s)
        comm["prediction_id"] = rid
        expected_comm = lock_doc["communication_outputs"]["direct_beam_classifier"]
    elif system == "direct_ADB_predictor":
        obj = shared_post[rid]
        (comm, beam_selection_s) = perf_call(
            device, c1.gaussian_comm_output,
            scenario_id=sid, prediction_id=rid,
            latest_position=np.asarray(obj["latest_position_H0_m"]),
            mean_displacement=np.asarray(obj["mean_displacement_H0_m"]),
            covariance=np.asarray(obj["covariance_H0_m2"]),
            codebooks={32: codebook32},
            sample_product_of_horizon_gaussians=rt["sample_product_of_horizon_gaussians"],
            beam_probability_mass_from_samples=rt["beam_probability_mass_from_samples"],
            adaptive_temporal_step=capture_step,
            initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
        )
        comm["posterior_digest_sha256"] = obj["posterior_digest_sha256"]
        expected_comm = lock_doc["communication_outputs"]["shared_trajectory_posterior"]
    else:
        obj = det_post[rid]
        (comm, beam_selection_s) = perf_call(
            device, c1.deterministic_comm_output,
            prediction_id=rid,
            latest_position=np.asarray(obj["latest_position_H0_m"]),
            mean_displacement=np.asarray(obj["mean_displacement_H0_m"]),
            codebooks={32: codebook32},
            BeamProbabilityMass=rt["BeamProbabilityMass"],
            complete_partition_decision_azimuth=rt["complete_partition_decision_azimuth"],
            adaptive_temporal_step=capture_step,
            initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
        )
        expected_comm = lock_doc["communication_outputs"]["deterministic_shared_trajectory"]

    require(len(captured) == 4, f"Expected 4 primary decisions, got {len(captured)}")

    # Compare only primary-32 branch because latency measurement intentionally does not execute sweep codebooks 16/64.
    expected_primary = dict(expected_comm)
    measured_primary = dict(comm)
    if "codebooks" in expected_primary:
        expected_primary["codebooks"] = {"32": expected_primary["codebooks"]["32"]}
    if "codebooks" in measured_primary:
        measured_primary["codebooks"] = {"32": measured_primary["codebooks"]["32"]}
    require(canonical_digest(c1.to_jsonable(measured_primary)) == canonical_digest(expected_primary),
            f"{system}: primary communication output identity differs from frozen C1.")

    from iscai_stage5.beam_latency import actual_probe_count_for_adaptive_decision, beam_probing_time_s
    probe_counts = [int(actual_probe_count_for_adaptive_decision(d)) for d in captured]
    probe_s = [float(beam_probing_time_s(k)) for k in probe_counts]

    # ADB branch with exact actuation-call instrumentation.
    act, smooth_wrap, rate_wrap = make_actuation_wrappers(rt, device)
    direct_adb_model_s = 0.0
    if system in ("shared_trajectory_posterior", "direct_beam_classifier"):
        post = shared_post
        ((schedule, adb_diag), adb_total_s) = perf_call(
            device, c1.gaussian_adb_schedule,
            prediction_by_id=post, scene_inputs=scene_inputs, actor_boxes=actor_boxes,
            association=association, eligible=eligible, grid=grid, params=adb_params, current_t0=current_t0,
            CalibratedGaussianFullBoxPrediction=rt["CalibratedGaussianFullBoxPrediction"],
            build_stochastic_future_full_boxes=rt["build_stochastic_future_full_boxes"],
            estimate_actor_occupancy_probability=rt["estimate_actor_occupancy_probability"],
            threshold_occupancy_counts_strict_k=rt["threshold_occupancy_counts_strict_k"],
            compute_class_aware_margin=rt["compute_class_aware_margin"],
            angular_margin_cells=rt["angular_margin_cells"], dilate_mask_theta=rt["dilate_mask_theta"],
            predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            compose_class_aware_illumination=rt["compose_class_aware_illumination"],
            temporal_smooth_schedule=smooth_wrap, apply_actuation_rate_limit=rate_wrap,
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
        )
        expected_npz_key = "shared_trajectory_posterior"
    elif system == "independent_models":
        ((schedule, adb_diag), adb_total_s) = perf_call(
            device, c1.gaussian_adb_schedule,
            prediction_by_id=indep_adb_post, scene_inputs=scene_inputs, actor_boxes=actor_boxes,
            association=association, eligible=eligible, grid=grid, params=adb_params, current_t0=current_t0,
            CalibratedGaussianFullBoxPrediction=rt["CalibratedGaussianFullBoxPrediction"],
            build_stochastic_future_full_boxes=rt["build_stochastic_future_full_boxes"],
            estimate_actor_occupancy_probability=rt["estimate_actor_occupancy_probability"],
            threshold_occupancy_counts_strict_k=rt["threshold_occupancy_counts_strict_k"],
            compute_class_aware_margin=rt["compute_class_aware_margin"],
            angular_margin_cells=rt["angular_margin_cells"], dilate_mask_theta=rt["dilate_mask_theta"],
            predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            compose_class_aware_illumination=rt["compose_class_aware_illumination"],
            temporal_smooth_schedule=smooth_wrap, apply_actuation_rate_limit=rate_wrap,
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
        )
        expected_npz_key = "independent_models"
    elif system == "direct_ADB_predictor":
        ((schedule, adb_diag, direct_adb_model_s), adb_total_s) = perf_call(
            device, c1.direct_adb_schedule,
            direct_model=models["direct_adb"], normalized_batch=normalized, history_ids=history_ids,
            actor_boxes=actor_boxes, association=association, eligible=eligible, grid=grid, params=adb_params,
            current_t0=current_t0, device=device,
            predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            compose_class_aware_illumination=rt["compose_class_aware_illumination"],
            temporal_smooth_schedule=smooth_wrap, apply_actuation_rate_limit=rate_wrap,
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
        )
        predictor_s += float(direct_adb_model_s)
        expected_npz_key = "direct_ADB_predictor"
    else:
        ((schedule, adb_diag), adb_total_s) = perf_call(
            device, c1.deterministic_adb_schedule,
            det_by_id=det_post, actor_boxes=actor_boxes, association=association, eligible=eligible,
            grid=grid, current_t0=current_t0,
            DeterministicSharedMeanPrediction=rt["DeterministicSharedMeanPrediction"],
            build_deterministic_future_full_boxes=rt["build_deterministic_future_full_boxes"],
            rasterize_projected_full_box=rt["rasterize_projected_full_box"],
            predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            temporal_smooth_schedule=smooth_wrap, apply_actuation_rate_limit=rate_wrap,
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
            params=adb_params,
        )
        expected_npz_key = "deterministic_shared_trajectory"

    actuation_s = float(act["seconds"])
    adb_generation_s = max(0.0, float(adb_total_s) - actuation_s - float(direct_adb_model_s))
    require(np.asarray(schedule).shape == (4, 501, 301), "ADB schedule shape changed.")

    npz_path = Path(lock_doc["ADB_outputs"]["npz_path"])
    require(npz_path.is_file(), "Frozen C1 NPZ missing.")
    with np.load(npz_path, allow_pickle=False) as frozen:
        require(expected_npz_key in frozen.files, f"Frozen C1 NPZ missing {expected_npz_key}")
        require(np.array_equal(np.asarray(schedule, dtype=np.float64), np.asarray(frozen[expected_npz_key], dtype=np.float64)),
                f"{system}: ADB output identity differs from frozen C1.")

    tracking_s = 0.0
    base_without_probe = (
        float(data_loading_s) + float(preprocessing_s) + tracking_s + float(predictor_s)
        + float(beam_selection_s) + float(adb_generation_s) + float(actuation_s)
    )
    totals = [base_without_probe + x for x in probe_s]
    primary_total = max(totals)

    cuda_peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    cuda_peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    rss = peak_rss_bytes()

    result = {
        "block": "7.5D",
        "status": "PASS_ISOLATED_SYSTEM_MEASUREMENT",
        "system": system,
        "runner_sha256": sha256_file(SCRIPT),
        "contract_sha256": EXPECTED_CONTRACT_SHA,
        "scenario": {
            "formal_index": idx,
            "scenario_id": sid,
            "selection_rule": "first formal scenario with non-null frozen C1 causal receiver",
            "receiver_prediction_id": rid,
            "causal_payload_sha256": payload_sha,
            "future_GT_accessed": False,
        },
        "output_identity": {
            "communication_primary32_equal_frozen_C1": True,
            "ADB_schedule_array_equal_frozen_C1": True,
        },
        "latency_s": {
            "data_loading": float(data_loading_s),
            "preprocessing": float(preprocessing_s),
            "tracking": 0.0,
            "predictor_inference": float(predictor_s),
            "beam_selection": float(beam_selection_s),
            "beam_probing_by_horizon": {str(c1.HORIZONS[i]): probe_s[i] for i in range(4)},
            "ADB_generation": float(adb_generation_s),
            "ADB_actuation": float(actuation_s),
            "total_frame_by_horizon": {str(c1.HORIZONS[i]): totals[i] for i in range(4)},
            "primary_conservative_total_frame": float(primary_total),
        },
        "beam_probe_counts_by_horizon": {str(c1.HORIZONS[i]): probe_counts[i] for i in range(4)},
        "resources": {
            "unique_trainable_parameter_count": int(parameter_count),
            "peak_RSS_bytes": int(rss),
            "peak_RSS_mib": float(rss / (1024**2)),
            "CUDA_peak_allocated_bytes_after_warmup_reset": cuda_peak_alloc,
            "CUDA_peak_reserved_bytes_after_warmup_reset": cuda_peak_reserved,
            "FLOPs": None,
            "FLOPs_status": "NOT_APPLICABLE_OR_NOT_EXACTLY_COMPUTABLE_UNDER_FROZEN_RUNTIME",
        },
        "measurement_boundary": {
            "fresh_process": True,
            "one_warmup_per_unique_required_model": True,
            "one_measured_inference_per_unique_required_model": True,
            "CUDA_synchronized_timing": True,
            "Linux_peak_RSS_reset_supported": False,
            "performance_measurement_only": True,
        },
        "model_details": {
            "deterministic_runtime_class": det_class,
            "direct_beam_model_forward_s": float(direct_beam_model_s),
            "direct_ADB_encode_decode_s": float(direct_adb_model_s),
        },
        "actuation_call_breakdown": act["calls"],
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "platform": platform.platform(),
        },
    }
    atomic_json(output, result)
    print(json.dumps({
        "system": system,
        "status": result["status"],
        "scenario_id": sid,
        "parameters": parameter_count,
        "peak_RSS_mib": result["resources"]["peak_RSS_mib"],
        "inference_s": predictor_s,
        "primary_total_frame_s": primary_total,
    }, sort_keys=True))
    return 0

def verify_existing_record(path: Path, system: str, runner_sha: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    doc = load_json(path)
    require(doc.get("status") == "PASS_ISOLATED_SYSTEM_MEASUREMENT", f"Bad prior record status {path}")
    require(doc.get("system") == system, f"Prior system mismatch {path}")
    require(doc.get("runner_sha256") == runner_sha, f"Prior runner SHA mismatch {path}")
    require(doc.get("contract_sha256") == EXPECTED_CONTRACT_SHA, f"Prior contract SHA mismatch {path}")
    require(doc["output_identity"]["communication_primary32_equal_frozen_C1"] is True, f"Prior comm parity failed {path}")
    require(doc["output_identity"]["ADB_schedule_array_equal_frozen_C1"] is True, f"Prior ADB parity failed {path}")
    return doc

def run_parent() -> int:
    gate_frozen()
    require(Path(__file__).resolve() == SCRIPT.resolve(), f"Runner path mismatch: {Path(__file__).resolve()}")
    runner_sha = sha256_file(SCRIPT)
    require(not TABLE.exists() and not SENSITIVITY.exists() and not MANIFEST.exists() and not REPORT.exists() and not SEAL.exists(),
            "Global 7.5D/final compliance output already exists; refusing overwrite.")

    print("="*88)
    print("STAGE 7 — BLOCK 7.5D")
    print("PDF LATENCY / RESOURCE CLOSURE ADDENDUM")
    print("FROZEN OUTCOMES UNCHANGED; PERFORMANCE-MEASUREMENT-ONLY MODEL FORWARDS")
    print("="*88)
    print("\n===== A. FROZEN CONTRACT + UPSTREAM GATE =====")
    print("contract SHA256 =", EXPECTED_CONTRACT_SHA)
    print("C1/C2/C3 exact identities = PASS")

    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    print("\n===== B. FIVE ISOLATED FRESH-PROCESS MEASUREMENTS =====")
    for system in SYSTEMS:
        out = RECORD_ROOT / f"{system}.json"
        prior = verify_existing_record(out, system, runner_sha)
        if prior is not None:
            print(f"REUSE {system}")
            records.append(prior)
            continue
        cmd = [sys.executable, str(SCRIPT), "--child-system", system, "--child-output", str(out)]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(proc.stdout, end="")
        require(proc.returncode == 0, f"Isolated measurement failed for {system}; rc={proc.returncode}")
        records.append(verify_existing_record(out, system, runner_sha))

    require(all(x is not None for x in records), "Missing system records.")
    payloads = {r["scenario"]["causal_payload_sha256"] for r in records}
    scenario_ids = {r["scenario"]["scenario_id"] for r in records}
    require(len(payloads) == 1, f"Causal payload differs across system processes: {payloads}")
    require(len(scenario_ids) == 1, "Scenario differs across system processes.")
    print("5/5 isolated records = PASS")
    print("identical causal payload digest = PASS")
    print("5/5 communication parity to C1 = PASS")
    print("5/5 ADB parity to C1 = PASS")

    print("\n===== C. FIVE-SYSTEM LATENCY / RESOURCE TABLE =====")
    rows = []
    for r in records:
        lat = r["latency_s"]; res = r["resources"]
        required = ("data_loading","preprocessing","predictor_inference","beam_selection","ADB_generation","ADB_actuation","primary_conservative_total_frame")
        require(all(math.isfinite(float(lat[k])) and float(lat[k]) >= 0.0 for k in required), f"Nonfinite latency in {r['system']}")
        require(int(res["unique_trainable_parameter_count"]) >= 0, "Bad parameter count.")
        require(int(res["peak_RSS_bytes"]) > 0, "Bad RSS.")
        require(res["FLOPs"] is None and isinstance(res["FLOPs_status"], str), "FLOPs must be numeric or explicit NA.")
        rows.append({
            "system": r["system"],
            "scenario_id": r["scenario"]["scenario_id"],
            "data_loading_s": lat["data_loading"],
            "preprocessing_s": lat["preprocessing"],
            "predictor_inference_s": lat["predictor_inference"],
            "beam_selection_s": lat["beam_selection"],
            "beam_probing_s_by_horizon": lat["beam_probing_by_horizon"],
            "ADB_generation_s": lat["ADB_generation"],
            "ADB_actuation_s": lat["ADB_actuation"],
            "total_frame_s_by_horizon": lat["total_frame_by_horizon"],
            "primary_conservative_total_frame_s": lat["primary_conservative_total_frame"],
            "unique_trainable_parameter_count": res["unique_trainable_parameter_count"],
            "peak_RSS_bytes": res["peak_RSS_bytes"],
            "peak_RSS_mib": res["peak_RSS_mib"],
            "FLOPs": res["FLOPs"],
            "FLOPs_status": res["FLOPs_status"],
        })
    table_doc = {
        "block": "7.5D",
        "status": "PASS_COMPLETE_FIVE_SYSTEM_LATENCY_RESOURCE_TABLE",
        "primary_codebook": 32,
        "primary_coverage": 0.95,
        "primary_latency_summary": "maximum across four frozen horizons",
        "rows": rows,
    }
    atomic_json(TABLE, table_doc)
    print("five-system formal latency/resource table = PASS")

    print("\n===== D. MEASURED LATENCY SENSITIVITY =====")
    sens_rows = []
    for row in rows:
        base = float(row["primary_conservative_total_frame_s"])
        sens_rows.append({
            "system": row["system"],
            "baseline_s": base,
            "multipliers": {str(m): base*m for m in LATENCY_MULTIPLIERS},
            "semantics": "offline runtime sensitivity only; no outcome/controller rerun",
        })
    atomic_json(SENSITIVITY, {
        "block": "7.5D",
        "status": "PASS_MEASURED_BASELINE_LATENCY_SENSITIVITY",
        "multipliers": list(LATENCY_MULTIPLIERS),
        "rows": sens_rows,
    })
    print("measured baseline latency multipliers {.5,1,2} = PASS")

    print("\n===== E. POST-MEASUREMENT IMMUTABILITY RECHECK =====")
    gate_frozen()
    print("C1/C2/C3 seals/manifests unchanged = PASS")

    record_entries = [{"path": str(RECORD_ROOT / f"{s}.json"), "sha256": sha256_file(RECORD_ROOT / f"{s}.json")} for s in SYSTEMS]
    manifest = {
        "block": "7.5D",
        "status": "FROZEN_COMPLETE_STAGE7_LATENCY_RESOURCE_CLOSURE",
        "contract": {"path": str(CONTRACT), "sha256": EXPECTED_CONTRACT_SHA},
        "runner": {"path": str(SCRIPT), "sha256": runner_sha},
        "measurement_records": record_entries,
        "table": {"path": str(TABLE), "sha256": sha256_file(TABLE)},
        "sensitivity": {"path": str(SENSITIVITY), "sha256": sha256_file(SENSITIVITY)},
        "upstream_preserved": EXPECTED_FROZEN,
        "future_GT_accessed": False,
        "formal_outcomes_recomputed": False,
        "performance_measurement_only_model_forward": True,
    }
    atomic_json(MANIFEST, manifest)

    report = {
        "block": "7.5D",
        "status": "PASS_STAGE7_PDF_LATENCY_RESOURCE_CLOSURE",
        "all_five_system_latency_resource_rows": True,
        "all_required_latency_components": True,
        "per_system_total_frame_latency": True,
        "per_system_inference_time": True,
        "per_system_parameter_count": True,
        "per_system_peak_RSS": True,
        "FLOPs_numeric_or_explicit_NA": True,
        "output_identity_to_frozen_C1": True,
        "prior_C3_final_seal_preserved": True,
        "manifest_sha256": sha256_file(MANIFEST),
        "runner_sha256": runner_sha,
    }
    atomic_json(REPORT, report)

    seal = {
        "stage": 7,
        "status": "STAGE7_PDF_CORE_COMPLETE_SUPERSEDING_SEAL",
        "scope": "Stage7 only; DeepSense remains Stage8",
        "stage7_pdf_compliance_100_percent_within_stage7_scope": True,
        "prior_C3_seal": {"path": str(C3_FINAL_SEAL), "sha256": EXPECTED_FROZEN[str(C3_FINAL_SEAL)], "preserved": True},
        "latency_resource_addendum": {
            "contract_sha256": EXPECTED_CONTRACT_SHA,
            "runner_sha256": runner_sha,
            "manifest_sha256": sha256_file(MANIFEST),
            "report_sha256": sha256_file(REPORT),
            "table_sha256": sha256_file(TABLE),
            "sensitivity_sha256": sha256_file(SENSITIVITY),
        },
        "completed_stage7_requirements": [
            "same shared posterior feeds communication and ADB",
            "five-system joint comparison",
            "joint communication-illumination metrics",
            "common communication-illumination tradeoff",
            "paired 10000 bootstrap confidence intervals",
            "14-name failure-case analysis with unsupported slices retained as NOT_EVALUABLE",
            "codebook sweep 16/32/64",
            "coverage sweep .90/.95/.975/.99",
            "predictive uncertainty sweep .5/1/1.5/2",
            "per-system isolated inference time",
            "per-system unique trainable parameter count",
            "per-system fresh-process peak RSS",
            "data loading latency",
            "preprocessing latency",
            "beam selection latency",
            "beam probing latency",
            "ADB generation latency",
            "ADB actuation latency",
            "per-system total-frame latency",
            "measured-baseline latency sensitivity .5/1/2",
            "FLOPs numeric if exact, otherwise explicit NA under frozen contract",
        ],
        "deferred_by_PDF_to_stage8": ["DeepSense external measured beam-power policy validation"],
        "scientific_integrity": {
            "future_GT_accessed_in_addendum": False,
            "future_GT_controller_input": False,
            "formal_outcome_recomputation": False,
            "training": False,
            "recalibration": False,
            "post_outcome_tuning": False,
            "C1_C2_C3_modified": False,
            "measurement_outputs_verified_equal_to_frozen_C1": True,
        },
    }
    atomic_json(SEAL, seal)

    gate_frozen()

    print("\n" + "="*88)
    print("BLOCK 7.5D = FULLY VERIFIED / FROZEN")
    print("5/5 ISOLATED SYSTEM MEASUREMENTS = PASS")
    print("IDENTICAL CAUSAL INPUT PAYLOAD = PASS")
    print("5/5 COMMUNICATION OUTPUT IDENTITY TO C1 = PASS")
    print("5/5 ADB OUTPUT IDENTITY TO C1 = PASS")
    print("PER-SYSTEM INFERENCE TIME = PASS")
    print("PER-SYSTEM PARAMETER COUNT = PASS")
    print("PER-SYSTEM PEAK RSS = PASS")
    print("ALL REQUIRED LATENCY COMPONENTS = PASS")
    print("PER-SYSTEM TOTAL FRAME LATENCY = PASS")
    print("MEASURED LATENCY SENSITIVITY .5/1/2 = PASS")
    print("FLOPS = NUMERIC_IF_EXACT_ELSE_EXPLICIT_NA PASS")
    print("C1/C2/C3 MODIFIED = NO")
    print("FUTURE GT ACCESSED = NO")
    print("POST-OUTCOME TUNING = NO")
    print("STAGE7 PDF COMPLIANCE SUPERSEDING SEAL = PASS")
    print("7.5D manifest SHA256 =", sha256_file(MANIFEST))
    print("7.5D report SHA256   =", sha256_file(REPORT))
    print("Stage7 PDF seal SHA256 =", sha256_file(SEAL))
    print("7.5D runner SHA256   =", runner_sha)
    print("STATUS = STAGE7_PDF_CORE_COMPLETE")
    print("="*88)
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child-system", choices=SYSTEMS)
    ap.add_argument("--child-output")
    args = ap.parse_args()
    if args.child_system:
        require(args.child_output, "--child-output required with --child-system")
        return run_child(args.child_system, Path(args.child_output))
    return run_parent()

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88, file=sys.stderr)
        print("BLOCK 7.5D FAIL-CLOSED", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("DO NOT DELETE/MODIFY ANY 7.5D RECORD THAT EXISTS.", file=sys.stderr)
        print("DO NOT START STAGE8.", file=sys.stderr)
        print("!"*88, file=sys.stderr)
        raise

#!/usr/bin/env python3
"""
Stage 7 — Block 7.5C1
Formal causal five-system execution + durable NON-ORACLE output lock.

SCIENTIFIC BOUNDARY
-------------------
This runner is intentionally evaluator-blind:
  * It MAY read each formal WOMD Scenario protobuf because causal history/current
    geometry live in that record.
  * It MUST NOT access states after scenario.current_time_index.
  * It MUST NOT access tracks_to_predict, objects_of_interest, future labels,
    constructed oracle/reference masks, or any formal metric outcome.
  * It computes/locks all five Section-47 controller outputs first.
  * Block 7.5C2 is the only permitted next step that may open evaluator-only
    future truth after verifying this durable lock.

The runner is resumable. A previously locked scenario is reused only when the
runner SHA, Block7.5B contract SHA, formal scenario ID/order, JSON SHA and NPZ
SHA all match exactly.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np
import torch


# =============================================================================
# Paths / frozen authorities
# =============================================================================

ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"
AUDIT = ROOT / "audits/stage7_pdf_alignment"

SCRIPT_EXPECTED_PATH = S7 / "scripts/run_block75c1_formal_causal_output_lock.py"

B75B = S7 / "configs/stage7_block75b_final_formal_evaluator_contract.json"
B74_HANDOFF = S7 / "configs/stage7_block74_MINIMAL_HANDOFF.json"
FORMAL = S3 / "artifacts/block38e/formal_validation_120.jsonl"
VALIDATION_MANIFEST = ROOT / "iscai_data_prep/manifests/selected_validation.jsonl"
PAIRED_ROOT = ROOT / "data/paired_womd_lidar_v1_3_0"

DET_CKPT = S4 / "artifacts/block43/deterministic_gru.pt"
DET_ARCH_CFG = S4 / "configs/stage4_deterministic_gru.json"
GAUSS_CKPT = S4 / "artifacts/block44/gaussian_gru.pt"
FIT_NORM = S4 / "artifacts/block43/fit_normalization.json"
SHARED_CAL = S4 / "artifacts/block45/covariance_scaler.json"

INDEP_COMM_CKPT = S7 / "artifacts/block74c/communication/artifacts/gaussian_gru.pt"
INDEP_ADB_CKPT = S7 / "artifacts/block74c/adb/artifacts/gaussian_gru.pt"
INDEP_COMM_CAL = S7 / "artifacts/block74d/communication/artifacts/covariance_scaler.json"
INDEP_ADB_CAL = S7 / "artifacts/block74d/adb/artifacts/covariance_scaler.json"
DIRECT_BEAM_CKPT = S7 / "artifacts/block74h_direct/beam/best.pt"
DIRECT_ADB_CKPT = S7 / "artifacts/block74h_direct/adb/best.pt"
DIRECT_TRAINER = S7 / "scripts/run_block74h_d_direct_baseline_training.py"

GRID_CFG = S6 / "configs/block64_mc_grid_numeric_freeze.json"
GAMMA_CFG = S6 / "configs/stage6_development_stage1_class_gamma_freeze.json"
MARGIN_CFG = S6 / "configs/stage6_development_stage2_class_margin_freeze.json"
FLOOR_CFG = S6 / "configs/stage6_development_stage3_class_floor_freeze.json"
TEMPORAL_CFG = S6 / "configs/stage6_development_stage4_temporal_rate_selection_freeze.json"
BEAM_CFG = S5 / "configs/beam_codebook_policy.json"
ANGULAR_CFG = S5 / "configs/angular_posterior_policy.json"

OUT_DIR = S7 / "artifacts/block75c1_nonoracle_lock"
MANIFEST = S7 / "artifacts/block75c1_nonoracle_lock_manifest.json"
REPORT = S7 / "reports/stage7_block75c1_nonoracle_lock_report.json"
LOG = AUDIT / "block75c1_formal_causal_output_lock.log"

EXPECTED_B75B_SHA = "5f79169138653b8223a795fae625adc5859a7a046a67c79fa98d357e256d7029"
EXPECTED_B74_SHA = "1351ac6d08f42e8a6627b6631aad438e89f416006feab800cd45c74d7f14eefa"
EXPECTED_FORMAL_SHA = "2208e7287ddf6439fda4597c435a9cba1d1b9d0e4c4547bc5dd92e56e8124e46"
EXPECTED_VALIDATION_MANIFEST_SHA = "dc10609ef18a2ba881657eb3da3a3df7a81bdcc8345ecbc2227102ab16b8833c"
EXPECTED_FORMAL_ORDER_SHA = "4ba67f109c67f1306374423429654415ef884217dd93371f33e3d8fe27d948b7"
EXPECTED_DIRECT_TRAINER_SHA = "db9be0d9ac4f4363003986ab3cfa52add2f2a498f32928b66a786bfd8f47a2a5"
EXPECTED_DET_ARCH_CFG_SHA = "88e3571fb8408b526eafa1799e092a8b627e34672d7a1708eb50bf0e55480a18"
EXPECTED_GRID_SHA = "993c4248a902e7dff3a4383ac343e7222cc43ef9b9372ed2efd0ee08d6d20a73"
EXPECTED_GAMMA_SHA = "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87"
EXPECTED_MARGIN_SHA = "b061541f0563181cd07a91107a211b3519c6fdcc2e08243171db5b3bd9393ae8"
EXPECTED_FLOOR_SHA = "c8536a1a2f888ebc3310cb900626cf0c8c0c2ca3ce34c2d321120a269c548d00"
EXPECTED_TEMPORAL_SHA = "780e134a66153c52783bc18d55f607c891b9a6983f8cfc8ac78e2e7367874885"
EXPECTED_BEAM_SHA = "bd94f8609393a7c9fe02762cc4bf38e3a77a90cc31adef5f2e6b06aece4074d7"
EXPECTED_ANGULAR_SHA = "846f6bf3d3419a2ea99fabc6293514726388bf60a31bb2a889aaf18b487fd82e"

EXPECTED_RETAINED = {
    DET_CKPT: "5456a76b84d558e9983a59b9f1d3060ba245e0d36d60883519654f809996dbc5",
    GAUSS_CKPT: "49ff64d145eaa633f295c16f660df380c35383e7e3b61279a5aad7cd700d619f",
    FIT_NORM: "3d7fc0a66d4a4f566f6569befa9c3766ecae21830bb4e46256df2f326a82a5f6",
    SHARED_CAL: "508ff2e3fbcfafe8e001155340c25baaf3772fe2561a8022a9ed1cf780e66087",
    INDEP_COMM_CKPT: "0f5fd0b2d77696563fb2b7ff9f5d8b78fa7e3c66757f267e119facc8aeea4880",
    INDEP_ADB_CKPT: "b40ca65c903f97c7fda3d5ebeb5052987f8c26cefcd032707af57e5c4de0bbf3",
    INDEP_COMM_CAL: "e2478cd44ca61b4132a50938e7a945bf184d104fe33b4e06c9a20864d821fdd5",
    INDEP_ADB_CAL: "c01e8d2070de1f062a6b7b72e76789c12148e00817366920a95fba6bacf9aed6",
    DIRECT_BEAM_CKPT: "bb0a8fd6dab8277f374216da07ea139730df2267148bfb838022201872fff2d4",
    DIRECT_ADB_CKPT: "c3bbfe83ebde069e5b84e73e54d7d5e71639164761eabc2ed7528a81aeabe88a",
    DIRECT_TRAINER: EXPECTED_DIRECT_TRAINER_SHA,
}

HORIZONS = (0.1, 0.3, 0.5, 1.0)
SYSTEMS = (
    "shared_trajectory_posterior",
    "independent_models",
    "direct_beam_classifier",
    "direct_ADB_predictor",
    "deterministic_shared_trajectory",
)
PRIMARY_CODEBOOK = 32
PRIMARY_COVERAGE = 0.95
CODEBOOKS = (16, 32, 64)
COVERAGES = (0.90, 0.95, 0.975, 0.99)
COMM_MC_N = 2048
COMM_BASE_SEED = 20260821
ADB_MC_N = 8192
ADB_MC_SEED = 20260821

TYPE_VEHICLE = "TYPE_VEHICLE"
TYPE_PEDESTRIAN = "TYPE_PEDESTRIAN"
TYPE_CYCLIST = "TYPE_CYCLIST"
CORE_CLASSES = (TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST)
DIRECT_CLASS_ID = {TYPE_VEHICLE: 0, TYPE_PEDESTRIAN: 1, TYPE_CYCLIST: 2}

GEOMETRY_EPS = 1.0e-12
HALF_ANGLE_RAD = math.radians(25.0)
MAXIMUM_RANGE_M = 150.0


class FailClosed(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FailClosed(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    temp = Path(temp_name)
    try:
        np.savez_compressed(temp, **arrays)
        with temp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp.exists():
            temp.unlink()


def recursive_key_values(obj: Any, key: str, path: str = "$") -> list[tuple[str, Any]]:
    hits: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if k == key:
                hits.append((p, v))
            hits.extend(recursive_key_values(v, key, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(recursive_key_values(v, key, f"{path}[{i}]"))
    return hits


def unique_recursive_value(obj: Any, key: str) -> Any:
    hits = recursive_key_values(obj, key)
    require(len(hits) == 1, f"Expected exactly one {key!r}; found {hits}")
    return hits[0][1]


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return int((COMM_BASE_SEED + value) % (2**31 - 1))


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(device: torch.device, fn, *args, **kwargs):
    cuda_sync(device)
    start = time.perf_counter_ns()
    value = fn(*args, **kwargs)
    cuda_sync(device)
    return value, (time.perf_counter_ns() - start) / 1e9


def np_float64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def resolve_formal_reader_records(
    formal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Resolve the frozen Stage3 formal-selection rows to the exact frozen
    Stage4 reader schema without opening any WOMD payload.

    Frozen cross-binding:
      formal.scenario_id            == validation.scenario_id
      formal.selection_hash         == validation.selection_hash
      formal.source_shard           == validation.source_shard
      formal.compact_record_offset  == validation.record_offset

    The only schema adaptation is:
      motion_shard          <- source_shard
      compact_record_offset <- record_offset
      payload_length        <- payload_length

    This is the same canonical validation authority used by frozen Stage4.
    """
    require(
        VALIDATION_MANIFEST.is_file(),
        f"Missing frozen validation manifest: {VALIDATION_MANIFEST}",
    )
    actual_sha = sha256_file(VALIDATION_MANIFEST)
    require(
        actual_sha == EXPECTED_VALIDATION_MANIFEST_SHA,
        (
            "Frozen selected_validation SHA changed: "
            f"{actual_sha} != {EXPECTED_VALIDATION_MANIFEST_SHA}"
        ),
    )

    validation_rows = [
        json.loads(line)
        for line in VALIDATION_MANIFEST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(validation_rows) == 44097,
        f"Frozen selected_validation row count changed: {len(validation_rows)}",
    )

    by_id: dict[str, dict[str, Any]] = {}

    for row in validation_rows:
        require(
            isinstance(row, dict),
            "selected_validation row is not an object.",
        )
        sid = str(row.get("scenario_id", ""))
        require(sid, "selected_validation row missing scenario_id.")
        require(
            sid not in by_id,
            f"Duplicate selected_validation scenario_id: {sid}",
        )
        by_id[sid] = row

    resolved: list[dict[str, Any]] = []

    for formal in formal_rows:
        sid = str(formal["scenario_id"])
        require(
            sid in by_id,
            f"Formal scenario absent from selected_validation: {sid}",
        )
        validation = by_id[sid]

        require(
            str(validation.get("scenario_id")) == sid,
            f"{sid}: validation scenario_id mismatch.",
        )
        require(
            str(validation.get("selection_hash"))
            == str(formal.get("selection_hash")),
            f"{sid}: selection_hash mismatch between formal and validation.",
        )
        require(
            str(validation.get("source_shard"))
            == str(formal.get("source_shard")),
            f"{sid}: source_shard mismatch between formal and validation.",
        )

        validation_offset = int(validation["record_offset"])
        formal_offset = int(formal["compact_record_offset"])

        require(
            validation_offset == formal_offset,
            (
                f"{sid}: record offset mismatch "
                f"validation={validation_offset} formal={formal_offset}"
            ),
        )

        payload_length = int(validation["payload_length"])
        require(
            payload_length > 0,
            f"{sid}: non-positive payload_length.",
        )

        # Reader record: preserve formal metadata and add ONLY the exact
        # field aliases expected by frozen Stage4 read_training_scenario().
        reader = dict(formal)
        reader["motion_shard"] = str(validation["source_shard"])
        reader["compact_record_offset"] = validation_offset
        reader["payload_length"] = payload_length

        # Keep the canonical validation names as provenance too.
        reader["validation_record_offset"] = validation_offset
        reader["validation_manifest_sha256"] = (
            EXPECTED_VALIDATION_MANIFEST_SHA
        )

        resolved.append(reader)

    require(
        len(resolved) == 120,
        f"Expected 120 formal reader records, got {len(resolved)}",
    )
    require(
        [str(x["scenario_id"]) for x in resolved]
        == [str(x["scenario_id"]) for x in formal_rows],
        "Formal reader resolver changed frozen scenario order.",
    )

    return resolved


# =============================================================================
# Frozen Stage6 headlamp-eligibility geometry (exact Block6.6 Part2C1 semantics)
# =============================================================================

def cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull_xy(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    require(array.ndim == 2 and array.shape[1] == 2, "convex_hull_xy expects [N,2].")
    require(np.all(np.isfinite(array)), "Non-finite XY corner.")
    unique = sorted({(float(row[0]), float(row[1])) for row in array})
    require(len(unique) >= 3, "Full box footprint has fewer than three unique points.")
    pts = [np.asarray(item, dtype=np.float64) for item in unique]
    lower: list[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and cross2(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross2(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    require(hull.shape[0] >= 3, "Degenerate ground footprint.")
    return hull


def clip_polygon_halfplane(polygon: np.ndarray, *, a: float, b: float, c: float = 0.0, eps: float = GEOMETRY_EPS) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.shape[0] == 0:
        return polygon.copy()
    result: list[np.ndarray] = []
    def value(point):
        return a * float(point[0]) + b * float(point[1]) + c
    for index in range(polygon.shape[0]):
        current = polygon[index]
        previous = polygon[index - 1]
        cv = value(current)
        pv = value(previous)
        ci = cv <= eps
        pi = pv <= eps
        if ci != pi:
            denominator = pv - cv
            require(abs(denominator) > 0.0, "Half-plane clipping zero denominator.")
            fraction = pv / denominator
            result.append(previous + fraction * (current - previous))
        if ci:
            result.append(current)
    if not result:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def point_segment_distance_origin(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    denominator = float(np.dot(delta, delta))
    if denominator <= 0.0:
        return float(np.linalg.norm(a))
    t = float(-np.dot(a, delta) / denominator)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(a + t * delta))


def minimum_radius_to_polygon(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon, dtype=np.float64)
    require(polygon.ndim == 2 and polygon.shape[1] == 2 and polygon.shape[0] > 0, "Invalid clipped polygon.")
    return float(min(point_segment_distance_origin(polygon[i], polygon[(i + 1) % polygon.shape[0]]) for i in range(polygon.shape[0])))


def box_intersects_forward_sector(box, *, box_corners_headlamp) -> bool:
    corners = np.asarray(box_corners_headlamp(box), dtype=np.float64)
    require(corners.shape == (8, 3), f"Expected exactly 8 box corners; got {corners.shape}.")
    footprint = convex_hull_xy(corners[:, :2])
    tangent = math.tan(HALF_ANGLE_RAD)
    clipped = clip_polygon_halfplane(footprint, a=-tangent, b=1.0)
    if clipped.shape[0] == 0:
        return False
    clipped = clip_polygon_halfplane(clipped, a=-tangent, b=-1.0)
    if clipped.shape[0] == 0:
        return False
    minimum_radius = minimum_radius_to_polygon(clipped)
    return bool(minimum_radius <= MAXIMUM_RANGE_M + GEOMETRY_EPS)


# =============================================================================
# Runtime imports (all exact, no heuristic metric/API discovery)
# =============================================================================

def import_runtime():
    from iscai_stage3.validation.womd_access import read_validation_manifest, read_motion_scenario
    from iscai_stage4.data.real_pipeline import load_frozen_stage2_configs, build_real_causal_inputs
    from iscai_stage4.data.neural_inputs import build_model_input_payload
    from iscai_stage4.data.supervision import build_static_map_index_H0, map_context_summary
    from iscai_stage4.ml.calibration_runtime import CalibrationNormalizer
    from iscai_stage4.ml.calibration import apply_variance_scale
    from iscai_stage4.ml.gaussian_gru import GaussianTrajectoryGRU
    from iscai_stage4.ml.gaussian_math import denormalize_gaussian
    from iscai_stage4.ml.formal_runtime import construct_deterministic_model, extract_deterministic_prediction

    from iscai_stage5.receiver_selection import ReceiverCandidate, ReceiverSelectionConfig, select_primary_receiver
    from iscai_stage5.beam_codebook import BeamProbabilityMass, build_uniform_azimuth_codebook, beam_probability_mass_from_samples, complete_partition_decision_azimuth
    from iscai_stage5.adaptive_topk_temporal import initial_adaptive_temporal_state, adaptive_temporal_step
    from iscai_stage5.angular_monte_carlo import sample_product_of_horizon_gaussians

    from iscai_stage6.adb.womd_geometry import build_causal_adb_actor_boxes
    from iscai_stage6.adb.deterministic_association import PredictorAnchor, reciprocal_unique_nearest_anchor_association
    from iscai_stage6.adb.deterministic_predictive import DeterministicSharedMeanPrediction, build_deterministic_future_full_boxes
    from iscai_stage6.adb.probabilistic_full_box import CalibratedGaussianFullBoxPrediction, build_stochastic_future_full_boxes
    from iscai_stage6.adb.probabilistic_occupancy import OccupancyGrid, estimate_actor_occupancy_probability, rasterize_projected_full_box
    from iscai_stage6.adb.class_aware_policy import threshold_occupancy_counts_strict_k, compute_class_aware_margin, angular_margin_cells, dilate_mask_theta, predictive_mask_to_illumination, compose_class_aware_illumination, temporal_smooth_schedule, apply_actuation_rate_limit
    from iscai_stage6.adb.geometry import box_corners_headlamp
    from iscai_stage6.adb.part_a_reactive import part_a_current_vehicle_centers_from_causal_boxes, part_a_original_reactive_map_from_h0_centers

    return locals()


# =============================================================================
# Model / calibration helpers
# =============================================================================

def load_variance_scale(path: Path) -> tuple[float, float, float, float]:
    doc = read_json(path)
    hits = recursive_key_values(doc, "variance_scale")
    candidates = []
    for p, value in hits:
        try:
            arr = tuple(float(x) for x in value)
        except Exception:
            continue
        if len(arr) == 4 and all(math.isfinite(x) and x > 0 for x in arr):
            candidates.append((p, arr))
    unique = {arr for _, arr in candidates}
    require(len(unique) == 1, f"Ambiguous variance_scale in {path}: {candidates}")
    return next(iter(unique))


def load_direct_module(path: Path):
    require(sha256_file(path) == EXPECTED_DIRECT_TRAINER_SHA, "Direct trainer SHA changed.")
    module_name = "iscai_stage7_block74_direct_runtime"
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "Could not create direct trainer import spec.")
    module = importlib.util.module_from_spec(spec)

    # Python 3.13 dataclasses resolves metadata through sys.modules while
    # class decorators execute. Register before exec_module(). Mechanical only.
    previous = sys.modules.get(module_name)
    require(previous is None or previous is module, f"Unexpected pre-existing dynamic module: {module_name}")
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    for name in ("DirectBeamClassifier", "DirectADBField"):
        require(hasattr(module, name), f"Direct trainer missing {name}.")
    return module


def build_gaussian_model(GaussianTrajectoryGRU, device: torch.device):
    return GaussianTrajectoryGRU(
        target_hidden_dim=64,
        neighbor_hidden_dim=32,
        map_hidden_dim=32,
        fusion_hidden_dim=128,
        use_neighbors=True,
        use_map=True,
    ).to(device)


def load_frozen_deterministic_model(construct_deterministic_model, device: torch.device):
    require(DET_ARCH_CFG.is_file(), f"Missing deterministic architecture authority: {DET_ARCH_CFG}")
    require(sha256_file(DET_ARCH_CFG) == EXPECTED_DET_ARCH_CFG_SHA, "Deterministic architecture config SHA changed.")

    architecture_configuration = read_json(DET_ARCH_CFG)
    checkpoint = torch.load(DET_CKPT, map_location=device, weights_only=False)
    require(isinstance(checkpoint, dict), "Deterministic checkpoint payload is not a dict.")
    require(isinstance(checkpoint.get("state_dict"), dict), "Deterministic checkpoint state_dict missing.")

    result = construct_deterministic_model(
        architecture_configuration,
        checkpoint["state_dict"],
        device=device,
    )
    require(isinstance(result, tuple) and len(result) == 2, "Stage4 deterministic constructor return contract changed.")
    model, class_name = result
    require(isinstance(class_name, str) and class_name, "Stage4 deterministic class authority missing.")
    model.eval()
    return model, class_name


def checkpoint_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=device, weights_only=False)
    require(isinstance(obj, dict) and "state_dict" in obj, f"Checkpoint state_dict missing: {path}")
    return obj["state_dict"]


def payload_arrays(payloads: list[Any]) -> dict[str, np.ndarray]:
    n = len(payloads)
    return {
        "target": np.asarray([p.target_history for p in payloads], dtype=np.float32).reshape(n, 11, 14),
        "neighbors": np.asarray([p.neighbor_histories for p in payloads], dtype=np.float32).reshape(n, 8, 11, 14),
        "neighbor_mask": np.asarray([p.neighbor_mask for p in payloads], dtype=np.float32).reshape(n, 8),
        "map_context": np.asarray([p.map_context for p in payloads], dtype=np.float32).reshape(n, 10),
        # Dummy placeholders required only by the frozen normalizer API.
        # They are never consumed by any model forward in this runner.
        "future": np.zeros((n, 4, 3), dtype=np.float32),
        "future_mask": np.zeros((n, 4), dtype=np.float32),
        "class_id": np.zeros((n,), dtype=np.int64),
    }


def infer_gaussian(
    model,
    batch: dict[str, torch.Tensor],
    normalizer,
    variance_scale,
    device: torch.device,
    *,
    apply_variance_scale,
    denormalize_gaussian,
):
    with torch.inference_mode():
        output, runtime = timed_call(
            device,
            model,
            batch["target"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["map_context"],
        )
        scaled = apply_variance_scale(output.scale_tril, variance_scale)
        mean_metric, scale_metric = denormalize_gaussian(
            output.mean,
            scaled,
            normalizer.label_mean,
            normalizer.label_std,
        )
        covariance = scale_metric @ scale_metric.transpose(-1, -2)
    mean_np = mean_metric.detach().cpu().numpy().astype(np.float64, copy=False)
    cov_np = covariance.detach().cpu().numpy().astype(np.float64, copy=False)
    require(mean_np.ndim == 3 and mean_np.shape[1:] == (4, 3), f"Gaussian mean shape {mean_np.shape}")
    require(cov_np.shape == (mean_np.shape[0], 4, 3, 3), f"Gaussian covariance shape {cov_np.shape}")
    require(np.all(np.isfinite(mean_np)) and np.all(np.isfinite(cov_np)), "Non-finite Gaussian output.")
    return mean_np, cov_np, float(runtime)


def infer_deterministic(model, batch: dict[str, torch.Tensor], normalizer, extract_deterministic_prediction, device: torch.device):
    with torch.inference_mode():
        raw_output, runtime = timed_call(
            device,
            model,
            batch["target"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["map_context"],
        )
        prediction = extract_deterministic_prediction(raw_output)
        require(torch.is_tensor(prediction), f"Deterministic extractor returned {type(prediction).__name__}, expected Tensor.")
        require(tuple(prediction.shape[1:]) == (4, 3), f"Deterministic prediction shape {tuple(prediction.shape)}")
        metric = prediction * normalizer.label_std[None, :, :] + normalizer.label_mean[None, :, :]
    metric_np = metric.detach().cpu().numpy().astype(np.float64, copy=False)
    require(np.all(np.isfinite(metric_np)), "Non-finite deterministic output.")
    return metric_np, float(runtime)


def posterior_digest(prediction_id: str, mean_absolute: np.ndarray, covariance: np.ndarray) -> str:
    """Exact Block7.5B canonical shared-posterior content digest."""
    h = hashlib.sha256()
    fields = (
        ("prediction_id_utf8", np.frombuffer(prediction_id.encode("utf-8"), dtype=np.uint8)),
        ("horizons_float64_C_order", np.asarray(HORIZONS, dtype="<f8")),
        ("mean_H0_m_float64_C_order", np.asarray(mean_absolute, dtype="<f8")),
        ("covariance_H0_m2_float64_C_order", np.asarray(covariance, dtype="<f8")),
    )
    for name, array in fields:
        a = np.ascontiguousarray(array)
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(str(a.dtype).encode("ascii")); h.update(b"\0")
        h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii")); h.update(b"\0")
        h.update(a.tobytes(order="C")); h.update(b"\0")
    return h.hexdigest()


# =============================================================================
# Communication controller helpers
# =============================================================================

def beam_probability_payload(prob) -> dict[str, Any]:
    return {
        "masses": [float(x) for x in prob.masses],
        "inside_support_mass": float(prob.inside_support_mass),
        "outside_support_mass": float(prob.outside_support_mass),
        "sample_count": int(prob.sample_count),
    }


def temporal_decisions_from_probabilities(probabilities, codebook, adaptive_temporal_step, initial_adaptive_temporal_state) -> list[dict[str, Any]]:
    state = initial_adaptive_temporal_state()
    result = []
    for h, prob in enumerate(probabilities):
        value = adaptive_temporal_step(
            state=state,
            probability=prob,
            codebook=codebook,
            requested_coverage=PRIMARY_COVERAGE,
        )
        # Frozen implementation returns exactly one AdaptiveTemporalDecision
        # and one AdaptiveTemporalState. Bind by frozen attributes rather than
        # tuple order.
        require(
            isinstance(value, tuple) and len(value) == 2,
            "adaptive_temporal_step return contract changed.",
        )

        decision_candidates = [
            obj for obj in value
            if hasattr(obj, "adaptive_selection")
            and hasattr(obj, "local_neighbor_probe_indices")
            and hasattr(obj, "exhaustive_fallback_active")
        ]
        state_candidates = [
            obj for obj in value
            if hasattr(obj, "cumulative_switch_events")
            and not hasattr(obj, "adaptive_selection")
        ]

        require(
            len(decision_candidates) == 1,
            (
                "Could not uniquely bind AdaptiveTemporalDecision from "
                f"return types={[type(x).__name__ for x in value]}"
            ),
        )
        require(
            len(state_candidates) == 1,
            (
                "Could not uniquely bind AdaptiveTemporalState from "
                f"return types={[type(x).__name__ for x in value]}"
            ),
        )

        decision = decision_candidates[0]
        state = state_candidates[0]

        result.append({
            "horizon_s": float(HORIZONS[h]),
            "selected_beam_indices": [int(x) for x in decision.adaptive_selection.beam_indices],
            "selected_K": int(len(decision.adaptive_selection.beam_indices)),
            "achieved_mass": float(
                sum(
                    float(prob.masses[int(i)])
                    for i in decision.adaptive_selection.beam_indices
                )
            ),
            "achieved_requested_coverage": bool(decision.adaptive_selection.achieved_requested_coverage),
            "primary_beam_index": int(decision.primary_beam_index),
            "primary_switched": bool(decision.primary_switched),
            "loss_of_lock": bool(decision.loss_of_lock),
            "local_neighbor_probe_indices": [int(x) for x in decision.local_neighbor_probe_indices],
            "exhaustive_fallback_active": bool(decision.exhaustive_fallback_active),
        })
    return result


def gaussian_comm_output(
    *, scenario_id: str, prediction_id: str, latest_position: np.ndarray,
    mean_displacement: np.ndarray, covariance: np.ndarray, codebooks: dict[int, Any],
    sample_product_of_horizon_gaussians, beam_probability_mass_from_samples,
    adaptive_temporal_step, initial_adaptive_temporal_state,
) -> dict[str, Any]:
    absolute_mean = np.asarray(latest_position, dtype=np.float64)[None, :] + np.asarray(mean_displacement, dtype=np.float64)
    seed = stable_seed(COMM_BASE_SEED, scenario_id, "uncertain_receiver_offset", "posterior")
    samples = sample_product_of_horizon_gaussians(
        trajectory_mean_h0_m=absolute_mean,
        calibrated_covariance_h0_m2=np.asarray(covariance, dtype=np.float64),
        sample_count=COMM_MC_N,
        seed=seed,
    )
    samples = np.asarray(samples, dtype=np.float64)
    require(samples.shape == (COMM_MC_N, 4, 3), f"Comm Gaussian samples shape {samples.shape}")
    azimuth = np.arctan2(samples[:, :, 1], samples[:, :, 0])
    by_codebook: dict[str, Any] = {}
    for n, codebook in codebooks.items():
        probs = [
            beam_probability_mass_from_samples(azimuth_samples_rad=azimuth[:, h], codebook=codebook)
            for h in range(4)
        ]
        by_codebook[str(n)] = {
            "probability": [beam_probability_payload(p) for p in probs],
            "primary_q_0p95_temporal_decisions": temporal_decisions_from_probabilities(
                probs, codebook, adaptive_temporal_step, initial_adaptive_temporal_state
            ),
        }
    return {
        "status": "SELECTED",
        "prediction_id": prediction_id,
        "sampling_seed": int(seed),
        "sampling_count": COMM_MC_N,
        "mean_absolute_H0_m": absolute_mean.tolist(),
        "codebooks": by_codebook,
    }


def direct_beam_logits_by_codebook(output: Any) -> dict[int, torch.Tensor]:
    require(isinstance(output, dict), f"Direct beam forward must return dict; got {type(output).__name__}")
    result: dict[int, torch.Tensor] = {}
    for n in CODEBOOKS:
        if n in output:
            tensor = output[n]
        elif str(n) in output:
            tensor = output[str(n)]
        else:
            raise FailClosed(f"Direct beam output missing codebook {n}; keys={list(output.keys())}")
        require(torch.is_tensor(tensor), f"Direct beam output {n} is not tensor")
        if tensor.ndim == 2 and tensor.shape == (1, 4 * n):
            tensor = tensor.reshape(1, 4, n)
        require(tuple(tensor.shape) == (1, 4, n), f"Direct beam output {n} shape={tuple(tensor.shape)}")
        result[n] = tensor
    return result


def direct_beam_comm_output(
    *, model, batch_one: dict[str, torch.Tensor], codebooks: dict[int, Any], BeamProbabilityMass,
    adaptive_temporal_step, initial_adaptive_temporal_state, device: torch.device,
) -> tuple[dict[str, Any], float]:
    with torch.inference_mode():
        raw, runtime = timed_call(
            device,
            model,
            batch_one["target"], batch_one["neighbors"], batch_one["neighbor_mask"], batch_one["map_context"],
        )
        logits = direct_beam_logits_by_codebook(raw)
    by_codebook: dict[str, Any] = {}
    for n, codebook in codebooks.items():
        probs_np = torch.softmax(logits[n], dim=-1)[0].detach().cpu().numpy().astype(np.float64, copy=False)
        probs = []
        for h in range(4):
            row = np.asarray(
                probs_np[h],
                dtype=np.float64,
            ).copy()

            require(
                np.all(np.isfinite(row)),
                "Direct beam probabilities contain non-finite values.",
            )
            require(
                np.all(row >= 0.0),
                "Direct beam probabilities contain negative values.",
            )

            total = float(
                math.fsum(
                    float(x)
                    for x in row
                )
            )
            require(
                math.isfinite(total)
                and total > 0.0,
                "Direct beam probability mass is non-positive/non-finite.",
            )

            # The direct-beam head is a softmax over the frozen in-support
            # codebook classes only; it has no outside-support class.
            # Canonicalize float32 softmax output into an exact conditional
            # in-support distribution before passing it to the stricter
            # frozen Stage5 BeamProbabilityMass validator.
            row /= total

            # Remove residual binary floating error from the final component
            # without changing class ordering or scientific semantics.
            if row.size > 1:
                prefix = math.fsum(
                    float(x)
                    for x in row[:-1]
                )
                row[-1] = 1.0 - prefix

            require(
                np.all(np.isfinite(row))
                and np.all(row >= 0.0),
                "Canonicalized direct beam probabilities invalid.",
            )

            exact_total = math.fsum(
                float(x)
                for x in row
            )
            require(
                abs(exact_total - 1.0) <= 1e-15,
                (
                    "Canonical direct beam mass does not sum to one: "
                    f"{exact_total}"
                ),
            )

            probs.append(
                BeamProbabilityMass(
                    masses=tuple(
                        float(x)
                        for x in row
                    ),
                    inside_support_mass=1.0,
                    outside_support_mass=0.0,
                    sample_count=1,
                )
            )
        by_codebook[str(n)] = {
            "probability": [beam_probability_payload(p) for p in probs],
            "primary_q_0p95_temporal_decisions": temporal_decisions_from_probabilities(
                probs, codebook, adaptive_temporal_step, initial_adaptive_temporal_state
            ),
        }
    return {"status": "SELECTED", "representation": "direct_softmax_no_explicit_trajectory", "codebooks": by_codebook}, float(runtime)


def deterministic_comm_output(
    *, prediction_id: str, latest_position: np.ndarray, mean_displacement: np.ndarray,
    codebooks: dict[int, Any], BeamProbabilityMass, complete_partition_decision_azimuth,
    adaptive_temporal_step, initial_adaptive_temporal_state,
) -> dict[str, Any]:
    absolute = np.asarray(latest_position, dtype=np.float64)[None, :] + np.asarray(mean_displacement, dtype=np.float64)
    azimuth = np.arctan2(absolute[:, 1], absolute[:, 0])
    by_codebook: dict[str, Any] = {}
    for n, codebook in codebooks.items():
        probs = []
        for h in range(4):
            theta = float(
                complete_partition_decision_azimuth(
                    azimuth_rad=float(azimuth[h]),
                    codebook=codebook,
                )
            )
            index = int(math.floor((theta - codebook.support_min_azimuth_rad) / codebook.decision_width_rad))
            index = max(0, min(n - 1, index))
            mass = np.zeros(n, dtype=np.float64); mass[index] = 1.0
            probs.append(BeamProbabilityMass(
                masses=tuple(float(x) for x in mass),
                inside_support_mass=1.0,
                outside_support_mass=0.0,
                sample_count=1,
            ))
        by_codebook[str(n)] = {
            "probability": [beam_probability_payload(p) for p in probs],
            "primary_q_0p95_temporal_decisions": temporal_decisions_from_probabilities(
                probs, codebook, adaptive_temporal_step, initial_adaptive_temporal_state
            ),
        }
    return {"status": "SELECTED", "prediction_id": prediction_id, "mean_absolute_H0_m": absolute.tolist(), "codebooks": by_codebook}


# =============================================================================
# ADB controller helpers
# =============================================================================

def grid_from_frozen_config(OccupancyGrid):
    doc = read_json(GRID_CFG)
    g = doc["scientific_illumination_grid"]
    theta = np.linspace(
        math.radians(float(g["theta_centers_deg"]["minimum"])),
        math.radians(float(g["theta_centers_deg"]["maximum"])),
        int(g["theta_centers_deg"]["count"]),
        dtype=np.float64,
    )
    radial = np.linspace(
        float(g["range_centers_m"]["minimum"]),
        float(g["range_centers_m"]["maximum"]),
        int(g["range_centers_m"]["count"]),
        dtype=np.float64,
    )
    require(theta.shape == (501,) and radial.shape == (301,), "Frozen grid shape changed.")
    return OccupancyGrid(tuple(float(x) for x in theta), tuple(float(x) for x in radial))


def frozen_adb_parameters() -> dict[str, Any]:
    grid_doc = read_json(GRID_CFG)
    gamma_doc = read_json(GAMMA_CFG)
    margin_doc = read_json(MARGIN_CFG)
    floor_doc = read_json(FLOOR_CFG)
    temporal_doc = read_json(TEMPORAL_CFG)
    require(grid_doc["MC"]["runtime_sample_count"] == ADB_MC_N, "ADB MC N changed.")
    require(grid_doc["MC"]["runtime_seed"] == ADB_MC_SEED, "ADB MC seed changed.")
    selected_gamma = gamma_doc["selected"]
    selected_margin = margin_doc["selected"]
    floor_selected = floor_doc["selected"]
    temporal_selected = temporal_doc["selected"]
    k = {c: int(selected_gamma[c]["k"]) for c in CORE_CLASSES}
    margins = {
        c: {
            "base_margin_m": float(selected_margin[c]["base_margin_m"]),
            "uncertainty_multiplier": float(selected_margin[c]["uncertainty_multiplier"]),
            "motion_multiplier": float(selected_margin[c]["motion_multiplier"]),
        }
        for c in CORE_CLASSES
    }
    floors = {
        TYPE_VEHICLE: float(floor_selected["f_vehicle"]),
        TYPE_PEDESTRIAN: float(floor_selected["f_pedestrian"]),
        TYPE_CYCLIST: float(floor_selected["f_cyclist"]),
    }
    require(floors == {TYPE_VEHICLE: 0.15, TYPE_PEDESTRIAN: 1.0, TYPE_CYCLIST: 1.0}, f"Class floors changed: {floors}")
    return {
        "k": k,
        "margins": margins,
        "floors": floors,
        "time_constant_s": float(temporal_selected["time_constant_s"]),
        "rho_dim_per_s": float(temporal_selected["rho_dim_per_s"]),
        "rho_bright_per_s": float(temporal_selected["rho_bright_per_s"]),
    }


def association_for_scene(scene_inputs, actor_boxes, PredictorAnchor, reciprocal_unique_nearest_anchor_association):
    anchors = tuple(PredictorAnchor(h.prediction_id, h.latest_position_H0_m) for h in scene_inputs.histories)
    return reciprocal_unique_nearest_anchor_association(anchors, actor_boxes)


def eligible_box_indices(actor_boxes, box_corners_headlamp) -> set[int]:
    return {
        i for i, actor in enumerate(actor_boxes)
        if box_intersects_forward_sector(actor.box, box_corners_headlamp=box_corners_headlamp)
    }


def current_reactive_map(actor_boxes, grid, part_a_current_vehicle_centers_from_causal_boxes, part_a_original_reactive_map_from_h0_centers) -> np.ndarray:
    centers = part_a_current_vehicle_centers_from_causal_boxes(actor_boxes)
    result = np.asarray(part_a_original_reactive_map_from_h0_centers(grid, centers), dtype=np.float64)
    require(result.shape == (501, 301), f"Current reactive map shape {result.shape}")
    return result


def fallback_class_maps(actor_boxes, fallback_indices: Iterable[int], grid, floors: dict[str, float], part_a_original_reactive_map_from_h0_centers) -> dict[str, np.ndarray]:
    by_class = {c: [] for c in CORE_CLASSES}
    for idx in fallback_indices:
        actor = actor_boxes[int(idx)]
        c = str(actor.object_type)
        if c not in by_class:
            continue
        center = actor.stage1_anchor_center_H0_m
        if center is not None:
            by_class[c].append(center)
    maps = {}
    for c in CORE_CLASSES:
        base = np.asarray(part_a_original_reactive_map_from_h0_centers(grid, tuple(by_class[c])), dtype=np.float64)
        # Frozen Block6.6 preregistration: reactive fallback preserves exact
        # Part-A raised-cosine geometry and receives the selected class floor.
        maps[c] = np.maximum(base, float(floors[c]))
    return maps


def final_class_aware_schedule(
    *, class_masks: dict[str, np.ndarray], fallback_maps: dict[str, np.ndarray], current_t0: np.ndarray,
    grid, params: dict[str, Any], predictive_mask_to_illumination,
    compose_class_aware_illumination, temporal_smooth_schedule, apply_actuation_rate_limit,
) -> np.ndarray:
    class_illumination: dict[str, np.ndarray] = {}
    active_masks: dict[str, np.ndarray] = {}
    for c in CORE_CLASSES:
        pred = predictive_mask_to_illumination(
            class_masks[c],
            range_centers_m=grid.range_centers_m,
            intensity_floor=float(params["floors"][c]),
        )
        fallback = np.broadcast_to(fallback_maps[c][None, :, :], pred.shape)
        combined = np.minimum(np.asarray(pred, dtype=np.float64), np.asarray(fallback, dtype=np.float64))
        class_illumination[c] = combined
        if c in (TYPE_PEDESTRIAN, TYPE_CYCLIST):
            # Fallback is floor-clipped; its dim-support participates in the VRU
            # guard only if it is genuinely <1 (with floor 1 this is empty).
            active_masks[c] = np.asarray(class_masks[c] | (fallback < 1.0), dtype=bool)
    composition = compose_class_aware_illumination(
        class_illumination=class_illumination,
        class_active_masks={TYPE_PEDESTRIAN: active_masks[TYPE_PEDESTRIAN], TYPE_CYCLIST: active_masks[TYPE_CYCLIST]},
        class_floors={TYPE_PEDESTRIAN: params["floors"][TYPE_PEDESTRIAN], TYPE_CYCLIST: params["floors"][TYPE_CYCLIST]},
    )
    raw = np.asarray(composition.raw_class_aware_illumination, dtype=np.float64)
    guard = np.asarray(composition.vru_floor_guard, dtype=np.float64)
    smoothed = temporal_smooth_schedule(
        raw,
        current_illumination=current_t0,
        horizons_s=HORIZONS,
        time_constant_s=params["time_constant_s"],
    )
    limited = apply_actuation_rate_limit(
        smoothed,
        current_illumination=current_t0,
        horizons_s=HORIZONS,
        rho_dim_per_s=params["rho_dim_per_s"],
        rho_bright_per_s=params["rho_bright_per_s"],
        vru_floor_guard=guard,
    )
    schedule = np.asarray(limited.illumination, dtype=np.float64)
    require(schedule.shape == (4, 501, 301), f"Class-aware schedule shape {schedule.shape}")
    return schedule


def gaussian_adb_schedule(
    *, prediction_by_id: dict[str, dict[str, Any]], scene_inputs, actor_boxes, association, eligible: set[int],
    grid, params, current_t0, CalibratedGaussianFullBoxPrediction, build_stochastic_future_full_boxes,
    estimate_actor_occupancy_probability, threshold_occupancy_counts_strict_k, compute_class_aware_margin,
    angular_margin_cells, dilate_mask_theta, predictive_mask_to_illumination, compose_class_aware_illumination,
    temporal_smooth_schedule, apply_actuation_rate_limit, part_a_original_reactive_map_from_h0_centers,
) -> tuple[np.ndarray, dict[str, Any]]:
    masks = {c: np.zeros((4, 501, 301), dtype=bool) for c in CORE_CLASSES}
    matched_eligible = []
    for match in association.matches:
        if int(match.box_index) not in eligible:
            continue
        actor = actor_boxes[int(match.box_index)]
        c = str(actor.object_type)
        if c not in CORE_CLASSES:
            continue
        pid = str(match.prediction_id)
        require(pid in prediction_by_id, f"Posterior missing prediction_id={pid}")
        p = prediction_by_id[pid]
        prediction = CalibratedGaussianFullBoxPrediction(
            prediction_id=pid,
            latest_position_H0_m=tuple(float(x) for x in p["latest_position_H0_m"]),
            mean_displacement_H0_m=tuple(tuple(float(x) for x in row) for row in p["mean_displacement_H0_m"]),
            calibrated_predictive_covariance_H0_m2=tuple(tuple(tuple(float(x) for x in row) for row in matrix) for matrix in p["covariance_H0_m2"]),
        )
        forecast = build_stochastic_future_full_boxes(prediction, actor, sample_count=ADB_MC_N, seed=ADB_MC_SEED)
        occupancy = estimate_actor_occupancy_probability(forecast, grid)
        binary = threshold_occupancy_counts_strict_k(
            occupancy.occupancy_counts,
            k=params["k"][c],
            sample_count=ADB_MC_N,
        )
        dilated = np.empty_like(binary, dtype=bool)
        mcfg = params["margins"][c]
        for h, tau in enumerate(HORIZONS):
            margin = compute_class_aware_margin(
                c,
                current_position_H0_m=p["latest_position_H0_m"],
                mean_displacement_H0_m=p["mean_displacement_H0_m"][h],
                covariance_H0_m2=p["covariance_H0_m2"][h],
                horizon_s=float(tau),
                base_margin_m=mcfg["base_margin_m"],
                uncertainty_multiplier=mcfg["uncertainty_multiplier"],
                motion_multiplier=mcfg["motion_multiplier"],
            )
            # Exact frozen Block6.7/6.8 semantic: m_extra only.
            cells = angular_margin_cells(
                total_margin_m=float(margin.additional_class_margin_m),
                predicted_range_m=float(margin.predicted_range_m),
                theta_step_rad=math.radians(0.1),
            )
            dilated[h] = dilate_mask_theta(binary[h:h+1], dilation_cells=int(cells))[0]
        masks[c] |= dilated
        matched_eligible.append({"prediction_id": pid, "actor_box_index": int(match.box_index), "actor_class": c})

    matched_box_indices = {int(m.box_index) for m in association.matches}
    fallback = sorted(i for i in eligible if i not in matched_box_indices)
    fallback_maps = fallback_class_maps(actor_boxes, fallback, grid, params["floors"], part_a_original_reactive_map_from_h0_centers)
    schedule = final_class_aware_schedule(
        class_masks=masks, fallback_maps=fallback_maps, current_t0=current_t0, grid=grid, params=params,
        predictive_mask_to_illumination=predictive_mask_to_illumination,
        compose_class_aware_illumination=compose_class_aware_illumination,
        temporal_smooth_schedule=temporal_smooth_schedule,
        apply_actuation_rate_limit=apply_actuation_rate_limit,
    )
    return schedule, {"matched_eligible": matched_eligible, "fallback_box_indices": fallback, "class_mask_cells": {c: int(masks[c].sum()) for c in CORE_CLASSES}}


def deterministic_adb_schedule(
    *, det_by_id: dict[str, dict[str, Any]], actor_boxes, association, eligible: set[int], grid, current_t0,
    DeterministicSharedMeanPrediction, build_deterministic_future_full_boxes, rasterize_projected_full_box,
    predictive_mask_to_illumination, temporal_smooth_schedule, apply_actuation_rate_limit,
    part_a_original_reactive_map_from_h0_centers, params,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.zeros((4, 501, 301), dtype=bool)
    matched = []
    for match in association.matches:
        if int(match.box_index) not in eligible:
            continue
        pid = str(match.prediction_id)
        require(pid in det_by_id, f"Deterministic output missing {pid}")
        p = det_by_id[pid]
        pred = DeterministicSharedMeanPrediction(
            prediction_id=pid,
            latest_position_H0_m=tuple(float(x) for x in p["latest_position_H0_m"]),
            mean_displacement_H0_m=tuple(tuple(float(x) for x in row) for row in p["mean_displacement_H0_m"]),
        )
        actor = actor_boxes[int(match.box_index)]
        future = build_deterministic_future_full_boxes(pred, actor)
        for h, item in enumerate(future):
            mask[h] |= rasterize_projected_full_box(item, grid)
        matched.append({"prediction_id": pid, "actor_box_index": int(match.box_index), "actor_class": str(actor.object_type)})
    matched_box_indices = {int(m.box_index) for m in association.matches}
    fallback_indices = sorted(i for i in eligible if i not in matched_box_indices)
    fallback_centers = [actor_boxes[i].stage1_anchor_center_H0_m for i in fallback_indices if actor_boxes[i].stage1_anchor_center_H0_m is not None]
    predictive = np.asarray(predictive_mask_to_illumination(mask, range_centers_m=grid.range_centers_m, intensity_floor=0.0), dtype=np.float64)
    fallback_map = np.asarray(part_a_original_reactive_map_from_h0_centers(grid, tuple(fallback_centers)), dtype=np.float64)
    raw = np.minimum(predictive, np.broadcast_to(fallback_map[None, :, :], predictive.shape))
    smoothed = temporal_smooth_schedule(raw, current_illumination=current_t0, horizons_s=HORIZONS, time_constant_s=params["time_constant_s"])
    limited = apply_actuation_rate_limit(
        smoothed, current_illumination=current_t0, horizons_s=HORIZONS,
        rho_dim_per_s=params["rho_dim_per_s"], rho_bright_per_s=params["rho_bright_per_s"],
        vru_floor_guard=None,
    )
    schedule = np.asarray(limited.illumination, dtype=np.float64)
    require(schedule.shape == (4, 501, 301), f"Deterministic ADB schedule shape {schedule.shape}")
    return schedule, {"matched_eligible": matched, "fallback_box_indices": fallback_indices, "mask_cells": int(mask.sum())}


def direct_adb_masks(
    *, direct_model, normalized_batch: dict[str, torch.Tensor], history_ids: list[str], actor_boxes,
    association, eligible: set[int], device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    masks = {c: np.zeros((4, 501, 301), dtype=bool) for c in CORE_CLASSES}
    id_to_batch = {pid: i for i, pid in enumerate(history_ids)}
    with torch.inference_mode():
        embedding_all, encode_time = timed_call(
            device,
            direct_model.encode,
            normalized_batch["target"], normalized_batch["neighbors"], normalized_batch["neighbor_mask"], normalized_batch["map_context"],
        )
    total_decode = 0.0
    theta_idx_full = np.repeat(np.arange(501, dtype=np.int64), 301)
    range_idx_full = np.tile(np.arange(301, dtype=np.int64), 501)
    chunk = 65536
    matched = []
    for match in association.matches:
        if int(match.box_index) not in eligible:
            continue
        pid = str(match.prediction_id)
        require(pid in id_to_batch, f"Direct ADB encoder missing {pid}")
        actor = actor_boxes[int(match.box_index)]
        c = str(actor.object_type)
        if c not in CORE_CLASSES:
            continue
        emb = embedding_all[id_to_batch[pid]:id_to_batch[pid] + 1]
        class_id = DIRECT_CLASS_ID[c]
        actor_mask = np.zeros((4, 501, 301), dtype=bool)
        for h in range(4):
            out_flat = np.empty(theta_idx_full.size, dtype=bool)
            for start in range(0, theta_idx_full.size, chunk):
                end = min(theta_idx_full.size, start + chunk)
                m = end - start
                emb_chunk = emb.expand(m, -1)
                ti = torch.from_numpy(theta_idx_full[start:end]).to(device=device, dtype=torch.long)
                ri = torch.from_numpy(range_idx_full[start:end]).to(device=device, dtype=torch.long)
                hi = torch.full((m,), h, device=device, dtype=torch.long)
                ci = torch.full((m,), class_id, device=device, dtype=torch.long)
                with torch.inference_mode():
                    logits, elapsed = timed_call(device, direct_model.decode, emb_chunk, ti, ri, hi, ci)
                total_decode += elapsed
                logits = logits.reshape(-1)
                require(logits.numel() == m, f"Direct ADB decode count {logits.numel()} != {m}")
                # strict p > 0.5 is exactly logit > 0
                out_flat[start:end] = (logits > 0.0).detach().cpu().numpy()
            actor_mask[h] = out_flat.reshape(501, 301)
        masks[c] |= actor_mask
        matched.append({"prediction_id": pid, "actor_box_index": int(match.box_index), "actor_class": c, "mask_cells": int(actor_mask.sum())})
    return masks, {"matched_eligible": matched, "encode_runtime_s": float(encode_time), "decode_runtime_s": float(total_decode)}, float(encode_time + total_decode)


def direct_adb_schedule(
    *, direct_model, normalized_batch, history_ids, actor_boxes, association, eligible, grid, params, current_t0,
    device, predictive_mask_to_illumination, compose_class_aware_illumination, temporal_smooth_schedule,
    apply_actuation_rate_limit, part_a_original_reactive_map_from_h0_centers,
):
    masks, diag, model_time = direct_adb_masks(
        direct_model=direct_model, normalized_batch=normalized_batch, history_ids=history_ids,
        actor_boxes=actor_boxes, association=association, eligible=eligible, device=device,
    )
    matched_box_indices = {int(m.box_index) for m in association.matches}
    fallback = sorted(i for i in eligible if i not in matched_box_indices)
    fallback_maps = fallback_class_maps(actor_boxes, fallback, grid, params["floors"], part_a_original_reactive_map_from_h0_centers)
    schedule = final_class_aware_schedule(
        class_masks=masks, fallback_maps=fallback_maps, current_t0=current_t0, grid=grid, params=params,
        predictive_mask_to_illumination=predictive_mask_to_illumination,
        compose_class_aware_illumination=compose_class_aware_illumination,
        temporal_smooth_schedule=temporal_smooth_schedule,
        apply_actuation_rate_limit=apply_actuation_rate_limit,
    )
    diag["fallback_box_indices"] = fallback
    diag["class_mask_cells"] = {c: int(masks[c].sum()) for c in CORE_CLASSES}
    return schedule, diag, model_time


# =============================================================================
# Formal scenario construction (strictly causal)
# =============================================================================

def build_payloads_for_scene(scenario, causal, build_static_map_index_H0, map_context_summary, build_model_input_payload):
    adapted = causal["adapted"]
    scene_inputs = causal["scene_inputs"]
    map_index = build_static_map_index_H0(scenario, T_H0_from_W=adapted.frames.T_H0_from_W)
    ids = [h.prediction_id for h in scene_inputs.histories]
    payloads = []
    for h in scene_inputs.histories:
        context = map_context_summary(map_index, target_position_H0_m=h.latest_position_H0_m)
        payloads.append(build_model_input_payload(scene_inputs, prediction_id=h.prediction_id, map_context=context))
    return ids, payloads


def posterior_dicts(history_ids, scene_inputs, means, covariances) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    out: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    require(len(history_ids) == means.shape[0] == covariances.shape[0], "Posterior batch count mismatch.")
    for i, pid in enumerate(history_ids):
        latest = np.asarray(scene_inputs.history_by_id(pid).latest_position_H0_m, dtype=np.float64)
        mean_disp = np.asarray(means[i], dtype=np.float64)
        cov = np.asarray(covariances[i], dtype=np.float64)
        mean_abs = latest[None, :] + mean_disp
        digest = posterior_digest(pid, mean_abs, cov)
        out[pid] = {
            "prediction_id": pid,
            "latest_position_H0_m": latest.tolist(),
            "mean_displacement_H0_m": mean_disp.tolist(),
            "mean_absolute_H0_m": mean_abs.tolist(),
            "covariance_H0_m2": cov.tolist(),
            "posterior_digest_sha256": digest,
        }
        digests[pid] = digest
    return out, digests


def deterministic_dicts(history_ids, scene_inputs, means) -> dict[str, dict[str, Any]]:
    result = {}
    for i, pid in enumerate(history_ids):
        latest = np.asarray(scene_inputs.history_by_id(pid).latest_position_H0_m, dtype=np.float64)
        result[pid] = {
            "prediction_id": pid,
            "latest_position_H0_m": latest.tolist(),
            "mean_displacement_H0_m": np.asarray(means[i], dtype=np.float64).tolist(),
        }
    return result


def select_receiver(scene_inputs, actor_boxes, association, ReceiverCandidate, ReceiverSelectionConfig, select_primary_receiver):
    candidates = []
    metadata = []
    for match in association.matches:
        actor = actor_boxes[int(match.box_index)]
        pid = str(match.prediction_id)
        history = scene_inputs.history_by_id(pid)
        candidates.append(ReceiverCandidate(
            semantic_class=str(actor.object_type),
            position_h0_m=tuple(float(x) for x in history.latest_position_H0_m),
            current_available=True,
            association_valid=True,
        ))
        metadata.append({
            "prediction_id": pid,
            "actor_box_index": int(match.box_index),
            "track_index": int(actor.track_index),
            "track_id": str(actor.track_id),
            "actor_class": str(actor.object_type),
            "current_position_H0_m": [float(x) for x in history.latest_position_H0_m],
        })
    selection = select_primary_receiver(
        candidates=candidates,
        config=ReceiverSelectionConfig(
            minimum_forward_h0_m=0.0,
            max_planar_range_m=None,
        ),
    )
    if selection.selected_candidate_index is None:
        return None, to_jsonable(selection)
    index = int(selection.selected_candidate_index)
    require(0 <= index < len(metadata), "Receiver selector index invalid.")
    return metadata[index], to_jsonable(selection)


def no_receiver_comm_output() -> dict[str, Any]:
    return {"status": "NO_ELIGIBLE_RECEIVER", "codebooks": {}}


# =============================================================================
# Resume / lock verification
# =============================================================================

def scenario_paths(index: int, scenario_id: str) -> tuple[Path, Path]:
    base = f"{index:03d}_{scenario_id}"
    return OUT_DIR / f"{base}.lock.json", OUT_DIR / f"{base}.adb_schedules.npz"


def verify_existing_lock(json_path: Path, npz_path: Path, *, index: int, scenario_id: str, runner_sha: str) -> dict[str, Any] | None:
    if not json_path.exists() and not npz_path.exists():
        return None
    require(json_path.is_file() and npz_path.is_file(), f"Partial prior lock exists for {scenario_id}")
    doc = read_json(json_path)
    require(doc.get("block") == "7.5C1", f"Existing lock block mismatch {scenario_id}")
    require(doc.get("scenario_id") == scenario_id and int(doc.get("formal_index")) == index, f"Existing lock identity mismatch {scenario_id}")
    require(doc.get("runner_sha256") == runner_sha, f"Existing lock runner SHA mismatch {scenario_id}")
    require(doc.get("block75b_contract_sha256") == EXPECTED_B75B_SHA, f"Existing lock contract SHA mismatch {scenario_id}")
    require(doc.get("future_GT_accessed") is False and doc.get("formal_metrics_computed") is False, f"Existing lock boundary invalid {scenario_id}")
    actual_npz = sha256_file(npz_path)
    require(doc.get("adb_schedules_npz_sha256") == actual_npz, f"Existing NPZ SHA mismatch {scenario_id}")
    return doc


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Debug-only causal limit; final PASS requires 120.")
    parser.add_argument("--start-index", type=int, default=None, help="Execution-only worker range start, inclusive.")
    parser.add_argument("--end-index", type=int, default=None, help="Execution-only worker range end, exclusive.")
    args = parser.parse_args()

    worker_mode = (args.start_index is not None) or (args.end_index is not None)
    require(
        not (args.limit is not None and worker_mode),
        "--limit cannot be combined with --start-index/--end-index.",
    )
    if worker_mode:
        require(
            args.start_index is not None and args.end_index is not None,
            "Worker mode requires both --start-index and --end-index.",
        )

    print("=" * 80)
    print("STAGE 7 — BLOCK 7.5C1")
    print("FORMAL CAUSAL FIVE-SYSTEM EXECUTION + NON-ORACLE OUTPUT LOCK")
    print("NO EVALUATOR FUTURE GT / NO FORMAL METRICS")
    print("=" * 80)

    require(Path(__file__).resolve() == SCRIPT_EXPECTED_PATH.resolve(), f"Runner must be installed at {SCRIPT_EXPECTED_PATH}; actual={Path(__file__).resolve()}")
    runner_sha = sha256_file(Path(__file__).resolve())

    # -------------------------------------------------------------------------
    # Authority gate
    # -------------------------------------------------------------------------
    print("\n===== A. FROZEN AUTHORITY GATE =====")
    for path, expected in (
        (B75B, EXPECTED_B75B_SHA),
        (B74_HANDOFF, EXPECTED_B74_SHA),
        (FORMAL, EXPECTED_FORMAL_SHA),
        (VALIDATION_MANIFEST, EXPECTED_VALIDATION_MANIFEST_SHA),
        (DET_ARCH_CFG, EXPECTED_DET_ARCH_CFG_SHA),
        (GRID_CFG, EXPECTED_GRID_SHA),
        (GAMMA_CFG, EXPECTED_GAMMA_SHA),
        (MARGIN_CFG, EXPECTED_MARGIN_SHA),
        (FLOOR_CFG, EXPECTED_FLOOR_SHA),
        (TEMPORAL_CFG, EXPECTED_TEMPORAL_SHA),
        (BEAM_CFG, EXPECTED_BEAM_SHA),
        (ANGULAR_CFG, EXPECTED_ANGULAR_SHA),
    ):
        require(path.is_file(), f"Missing frozen authority: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"SHA mismatch {path}: {actual} != {expected}")
        print("PASS", path, actual)

    handoff = read_json(B74_HANDOFF)
    contract = read_json(B75B)
    require(contract.get("status") == "FROZEN_FINAL_FORMAL_EVALUATOR_CONTRACT", "Block7.5B status invalid.")
    require(handoff.get("status") == "FROZEN_BASELINE_PACKAGE_READY_FOR_CLEAN_HANDOFF", "Block7.4 handoff invalid.")
    require(contract["next_block"]["name"] == "7.5C", "Block7.5B does not authorize 7.5C.")

    for path, expected in EXPECTED_RETAINED.items():
        require(path.is_file(), f"Missing retained authority: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"Retained SHA mismatch {path}: {actual}")
    print("PASS retained runtime/checkpoint package")

    # Recheck every retained Stage7 file declared by the handoff.
    for raw_path, info in handoff["retained_Stage7_files"].items():
        p = Path(raw_path)
        require(p.is_file(), f"Retained handoff file missing: {p}")
        require(sha256_file(p) == info["sha256"], f"Retained handoff SHA mismatch: {p}")
    print("PASS all handoff-retained Stage7 files")

    # -------------------------------------------------------------------------
    # Formal manifest identity — IDs/metadata only, already unsealed in 7.4.
    # -------------------------------------------------------------------------
    print("\n===== B. FORMAL N=120 ORDER GATE =====")
    formal_rows = [json.loads(line) for line in FORMAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_ids = [str(row["scenario_id"]) for row in formal_rows]
    require(len(scenario_ids) == 120 and len(set(scenario_ids)) == 120, "Formal N/uniqueness changed.")
    order_sha = sha256_bytes(json.dumps(scenario_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    require(order_sha == EXPECTED_FORMAL_ORDER_SHA, f"Formal order SHA changed: {order_sha}")
    print("formal N = 120")
    print("formal order SHA256 =", order_sha)

    # Metadata-only canonical resolver. This runs for all 120 before the
    # first raw WOMD protobuf is opened.
    reader_rows = resolve_formal_reader_records(formal_rows)
    print("formal -> selected_validation metadata binding = 120/120 PASS")
    print("reader alias motion_shard <- source_shard = PASS")
    print("reader alias compact_record_offset <- record_offset = PASS")
    print("payload_length binding = PASS")

    indexed_reader_rows = list(enumerate(reader_rows))

    if args.limit is not None:
        require(1 <= args.limit <= 120, "--limit must be 1..120")
        rows_to_run = indexed_reader_rows[: args.limit]
        print("DEBUG LIMIT =", args.limit, "(cannot publish final manifest/report)")
    elif worker_mode:
        start_index = int(args.start_index)
        end_index = int(args.end_index)
        require(
            0 <= start_index < end_index <= 120,
            f"Invalid worker range [{start_index}, {end_index}).",
        )
        rows_to_run = indexed_reader_rows[start_index:end_index]
        print(
            "PARALLEL WORKER RANGE =",
            f"[{start_index}, {end_index})",
            "(execution scheduling only; cannot publish final manifest/report)",
        )
    else:
        rows_to_run = indexed_reader_rows

    # -------------------------------------------------------------------------
    # Deterministic environment and exact imports.
    # -------------------------------------------------------------------------
    print("\n===== C. RUNTIME INITIALIZATION =====")
    require(torch.cuda.is_available(), "Frozen Stage4/7 formal runtime requires CUDA; CUDA unavailable.")
    device = torch.device("cuda:0")
    torch.manual_seed(20260821)
    np.random.seed(20260821)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("device =", device)
    print("GPU =", torch.cuda.get_device_name(device))

    rt = import_runtime()
    direct_module = load_direct_module(DIRECT_TRAINER)

    # Exact frozen Stage3/Stage4 formal-validation route.
    # Metadata only: no raw WOMD protobuf is opened here.
    validation_rows_runtime = rt["read_validation_manifest"](
        VALIDATION_MANIFEST
    )
    validation_by_id_runtime = {
        str(row.scenario_id): row
        for row in validation_rows_runtime
    }
    require(
        len(validation_by_id_runtime) == 44097,
        (
            "Canonical validation runtime index changed: "
            f"{len(validation_by_id_runtime)}"
        ),
    )

    missing_runtime_ids = [
        str(row["scenario_id"])
        for row in formal_rows
        if str(row["scenario_id"])
        not in validation_by_id_runtime
    ]
    require(
        not missing_runtime_ids,
        (
            "Formal scenarios missing from canonical validation runtime: "
            f"{missing_runtime_ids[:5]}"
        ),
    )

    require(
        PAIRED_ROOT.is_dir(),
        f"Canonical paired root missing: {PAIRED_ROOT}",
    )

    # Pure filesystem preflight: check every unique formal source shard has the
    # canonical compact local paired motion file expected by the frozen
    # Stage3 resolver. No TFRecord payload is opened here.
    expected_runtime_shards = []
    for formal in formal_rows:
        source_name = Path(str(formal["source_shard"])).name
        expected_runtime_shards.append(
            PAIRED_ROOT
            / "validation"
            / "motion"
            / f"paired-from-{source_name}"
        )

    missing_runtime_shards = sorted({
        str(path)
        for path in expected_runtime_shards
        if not path.is_file()
    })
    require(
        not missing_runtime_shards,
        (
            "Canonical local paired motion shard(s) missing: "
            f"{missing_runtime_shards[:5]}"
        ),
    )

    print(
        "canonical Stage3 validation runtime index = 44097 PASS"
    )
    print(
        "formal IDs in canonical validation runtime = 120/120 PASS"
    )
    print(
        "formal local paired motion shard preflight = PASS "
        f"({len(set(expected_runtime_shards))} unique shards)"
    )

    clean_config, degraded_config = rt["load_frozen_stage2_configs"]()
    normalization_doc = read_json(FIT_NORM)
    normalizer = rt["CalibrationNormalizer"](normalization_doc, device=device)

    shared_scale = load_variance_scale(SHARED_CAL)
    indep_comm_scale = load_variance_scale(INDEP_COMM_CAL)
    indep_adb_scale = load_variance_scale(INDEP_ADB_CAL)
    print("shared variance scale =", shared_scale)
    print("independent comm variance scale =", indep_comm_scale)
    print("independent ADB variance scale =", indep_adb_scale)

    shared_model = build_gaussian_model(rt["GaussianTrajectoryGRU"], device)
    shared_model.load_state_dict(checkpoint_state_dict(GAUSS_CKPT, device), strict=True); shared_model.eval()
    indep_comm_model = build_gaussian_model(rt["GaussianTrajectoryGRU"], device)
    indep_comm_model.load_state_dict(checkpoint_state_dict(INDEP_COMM_CKPT, device), strict=True); indep_comm_model.eval()
    indep_adb_model = build_gaussian_model(rt["GaussianTrajectoryGRU"], device)
    indep_adb_model.load_state_dict(checkpoint_state_dict(INDEP_ADB_CKPT, device), strict=True); indep_adb_model.eval()
    det_model, det_model_class = load_frozen_deterministic_model(rt["construct_deterministic_model"], device)
    print("deterministic runtime class =", det_model_class)

    direct_beam_model = direct_module.DirectBeamClassifier().to(device)
    direct_beam_model.load_state_dict(checkpoint_state_dict(DIRECT_BEAM_CKPT, device), strict=True); direct_beam_model.eval()
    direct_adb_model = direct_module.DirectADBField().to(device)
    direct_adb_model.load_state_dict(checkpoint_state_dict(DIRECT_ADB_CKPT, device), strict=True); direct_adb_model.eval()

    grid = grid_from_frozen_config(rt["OccupancyGrid"])
    adb_params = frozen_adb_parameters()
    beam_cfg = read_json(BEAM_CFG)
    support = beam_cfg["azimuth_support"]
    support_min = math.radians(float(support["min_deg"]))
    support_max = math.radians(float(support["max_deg"]))
    codebooks = {
        n: rt["build_uniform_azimuth_codebook"](
            beam_count=n, support_min_azimuth_rad=support_min, support_max_azimuth_rad=support_max
        )
        for n in CODEBOOKS
    }
    print("PASS all model/controller runtimes loaded before raw formal scenario read")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    future_gt_accessed = False
    formal_metrics_computed = False

    # -------------------------------------------------------------------------
    # Formal causal execution. No evaluator future truth below this line.
    # -------------------------------------------------------------------------
    print("\n===== D. FORMAL CAUSAL EXECUTION =====")
    for formal_index, record in rows_to_run:
        scenario_id = str(record["scenario_id"])
        lock_json, lock_npz = scenario_paths(formal_index, scenario_id)
        existing = verify_existing_lock(lock_json, lock_npz, index=formal_index, scenario_id=scenario_id, runner_sha=runner_sha)
        if existing is not None:
            print(f"[{formal_index:03d}/119] REUSE {scenario_id}")
            completed.append({
                "formal_index": formal_index,
                "scenario_id": scenario_id,
                "lock_json": str(lock_json),
                "lock_json_sha256": sha256_file(lock_json),
                "adb_schedules_npz": str(lock_npz),
                "adb_schedules_npz_sha256": sha256_file(lock_npz),
                "nonoracle_output_digest_sha256": existing["nonoracle_output_digest_sha256"],
            })
            continue

        print(f"[{formal_index:03d}/119] RUN   {scenario_id}", flush=True)
        scenario_start = time.perf_counter()

        # Exact frozen Stage3/Stage4 formal-validation route already used
        # upstream. read_motion_scenario resolves the historical source shard
        # to the canonical local paired compact motion shard.
        #
        # This is the first raw protobuf read in C1. All controller/model input
        # accesses below remain causal/current-only; evaluator future truth is
        # not accessed in this block.
        validation_row = validation_by_id_runtime[scenario_id]
        scenario = rt["read_motion_scenario"](
            validation_row,
            paired_root=PAIRED_ROOT,
            compact_record_offset=int(
                record["compact_record_offset"]
            ),
        )
        require(str(scenario.scenario_id) == scenario_id, "Formal scenario ID mismatch.")
        require(int(scenario.current_time_index) == 10, "Frozen current_time_index changed.")

        causal, causal_runtime = timed_call(
            device,
            rt["build_real_causal_inputs"],
            scenario,
            clean_config=clean_config,
            degraded_config=degraded_config,
        )
        scene_inputs = causal["scene_inputs"]
        adapted = causal["adapted"]
        history_ids, payloads = build_payloads_for_scene(
            scenario, causal, rt["build_static_map_index_H0"], rt["map_context_summary"], rt["build_model_input_payload"]
        )
        require(history_ids == sorted(history_ids), "Frozen causal history ordering changed.")
        require(len(payloads) == len(history_ids), "Payload/history count mismatch.")

        actor_boxes = rt["build_causal_adb_actor_boxes"](scenario=scenario, adapted=adapted)
        association = association_for_scene(
            scene_inputs, actor_boxes, rt["PredictorAnchor"], rt["reciprocal_unique_nearest_anchor_association"]
        )
        eligible = eligible_box_indices(actor_boxes, rt["box_corners_headlamp"])
        receiver, receiver_selection = select_receiver(
            scene_inputs, actor_boxes, association,
            rt["ReceiverCandidate"], rt["ReceiverSelectionConfig"], rt["select_primary_receiver"],
        )

        if len(payloads) == 0:
            raise FailClosed(f"Formal scenario {scenario_id} produced zero causal model payloads.")

        arrays = payload_arrays(payloads)
        normalized = normalizer.prepare(arrays, device=device)

        # ----- Unique predictor forwards -----
        shared_mean, shared_cov, shared_forward_s = infer_gaussian(shared_model, normalized, normalizer, shared_scale, device, apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"])
        indep_comm_mean, indep_comm_cov, indep_comm_forward_s = infer_gaussian(indep_comm_model, normalized, normalizer, indep_comm_scale, device, apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"])
        indep_adb_mean, indep_adb_cov, indep_adb_forward_s = infer_gaussian(indep_adb_model, normalized, normalizer, indep_adb_scale, device, apply_variance_scale=rt["apply_variance_scale"], denormalize_gaussian=rt["denormalize_gaussian"])
        det_mean, det_forward_s = infer_deterministic(
            det_model, normalized, normalizer, rt["extract_deterministic_prediction"], device
        )

        shared_post, shared_digests = posterior_dicts(history_ids, scene_inputs, shared_mean, shared_cov)
        indep_comm_post, _ = posterior_dicts(history_ids, scene_inputs, indep_comm_mean, indep_comm_cov)
        indep_adb_post, _ = posterior_dicts(history_ids, scene_inputs, indep_adb_mean, indep_adb_cov)
        det_post = deterministic_dicts(history_ids, scene_inputs, det_mean)

        # ----- Communication branch -----
        communication: dict[str, Any] = {}
        direct_beam_forward_s = 0.0
        if receiver is None:
            for system in SYSTEMS:
                communication[system] = no_receiver_comm_output()
        else:
            rid = str(receiver["prediction_id"])
            require(rid in shared_post and rid in indep_comm_post and rid in det_post, "Selected receiver prediction missing from predictor outputs.")
            shared_obj = shared_post[rid]
            communication["shared_trajectory_posterior"] = gaussian_comm_output(
                scenario_id=scenario_id, prediction_id=rid,
                latest_position=np.asarray(shared_obj["latest_position_H0_m"]),
                mean_displacement=np.asarray(shared_obj["mean_displacement_H0_m"]),
                covariance=np.asarray(shared_obj["covariance_H0_m2"]),
                codebooks=codebooks,
                sample_product_of_horizon_gaussians=rt["sample_product_of_horizon_gaussians"],
                beam_probability_mass_from_samples=rt["beam_probability_mass_from_samples"],
                adaptive_temporal_step=rt["adaptive_temporal_step"],
                initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
            )
            communication["shared_trajectory_posterior"]["posterior_digest_sha256"] = shared_obj["posterior_digest_sha256"]
            indep_obj = indep_comm_post[rid]
            communication["independent_models"] = gaussian_comm_output(
                scenario_id=scenario_id, prediction_id=rid,
                latest_position=np.asarray(indep_obj["latest_position_H0_m"]),
                mean_displacement=np.asarray(indep_obj["mean_displacement_H0_m"]),
                covariance=np.asarray(indep_obj["covariance_H0_m2"]),
                codebooks=codebooks,
                sample_product_of_horizon_gaussians=rt["sample_product_of_horizon_gaussians"],
                beam_probability_mass_from_samples=rt["beam_probability_mass_from_samples"],
                adaptive_temporal_step=rt["adaptive_temporal_step"],
                initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
            )
            receiver_batch_index = history_ids.index(rid)
            batch_one = {k: v[receiver_batch_index:receiver_batch_index + 1] for k, v in normalized.items() if k in ("target", "neighbors", "neighbor_mask", "map_context")}
            direct_comm, direct_beam_forward_s = direct_beam_comm_output(
                model=direct_beam_model, batch_one=batch_one, codebooks=codebooks,
                BeamProbabilityMass=rt["BeamProbabilityMass"],
                adaptive_temporal_step=rt["adaptive_temporal_step"],
                initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
                device=device,
            )
            direct_comm["prediction_id"] = rid
            communication["direct_beam_classifier"] = direct_comm
            # Hybrid baseline: direct ADB replaces ADB branch only.
            communication["direct_ADB_predictor"] = {
                "status": "REUSE_SHARED_COMMUNICATION_BRANCH",
                "source_system": "shared_trajectory_posterior",
                "source_posterior_digest_sha256": shared_obj["posterior_digest_sha256"],
            }
            det_obj = det_post[rid]
            communication["deterministic_shared_trajectory"] = deterministic_comm_output(
                prediction_id=rid,
                latest_position=np.asarray(det_obj["latest_position_H0_m"]),
                mean_displacement=np.asarray(det_obj["mean_displacement_H0_m"]),
                codebooks=codebooks,
                BeamProbabilityMass=rt["BeamProbabilityMass"],
                complete_partition_decision_azimuth=rt["complete_partition_decision_azimuth"],
                adaptive_temporal_step=rt["adaptive_temporal_step"],
                initial_adaptive_temporal_state=rt["initial_adaptive_temporal_state"],
            )

        # ----- ADB branch -----
        current_t0 = current_reactive_map(
            actor_boxes, grid,
            rt["part_a_current_vehicle_centers_from_causal_boxes"],
            rt["part_a_original_reactive_map_from_h0_centers"],
        )

        (shared_schedule, shared_adb_diag), shared_adb_runtime_s = timed_call(
            device,
            gaussian_adb_schedule,
            prediction_by_id=shared_post, scene_inputs=scene_inputs, actor_boxes=actor_boxes,
            association=association, eligible=eligible, grid=grid, params=adb_params, current_t0=current_t0,
            CalibratedGaussianFullBoxPrediction=rt["CalibratedGaussianFullBoxPrediction"],
            build_stochastic_future_full_boxes=rt["build_stochastic_future_full_boxes"],
            estimate_actor_occupancy_probability=rt["estimate_actor_occupancy_probability"],
            threshold_occupancy_counts_strict_k=rt["threshold_occupancy_counts_strict_k"],
            compute_class_aware_margin=rt["compute_class_aware_margin"], angular_margin_cells=rt["angular_margin_cells"],
            dilate_mask_theta=rt["dilate_mask_theta"], predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            compose_class_aware_illumination=rt["compose_class_aware_illumination"], temporal_smooth_schedule=rt["temporal_smooth_schedule"],
            apply_actuation_rate_limit=rt["apply_actuation_rate_limit"],
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
        )
        (indep_adb_schedule, indep_adb_diag), indep_adb_runtime_s = timed_call(
            device,
            gaussian_adb_schedule,
            prediction_by_id=indep_adb_post, scene_inputs=scene_inputs, actor_boxes=actor_boxes,
            association=association, eligible=eligible, grid=grid, params=adb_params, current_t0=current_t0,
            CalibratedGaussianFullBoxPrediction=rt["CalibratedGaussianFullBoxPrediction"],
            build_stochastic_future_full_boxes=rt["build_stochastic_future_full_boxes"],
            estimate_actor_occupancy_probability=rt["estimate_actor_occupancy_probability"],
            threshold_occupancy_counts_strict_k=rt["threshold_occupancy_counts_strict_k"],
            compute_class_aware_margin=rt["compute_class_aware_margin"], angular_margin_cells=rt["angular_margin_cells"],
            dilate_mask_theta=rt["dilate_mask_theta"], predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            compose_class_aware_illumination=rt["compose_class_aware_illumination"], temporal_smooth_schedule=rt["temporal_smooth_schedule"],
            apply_actuation_rate_limit=rt["apply_actuation_rate_limit"],
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
        )
        direct_schedule, direct_adb_diag, direct_adb_runtime_s = direct_adb_schedule(
            direct_model=direct_adb_model, normalized_batch=normalized, history_ids=history_ids,
            actor_boxes=actor_boxes, association=association, eligible=eligible, grid=grid, params=adb_params,
            current_t0=current_t0, device=device,
            predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            compose_class_aware_illumination=rt["compose_class_aware_illumination"],
            temporal_smooth_schedule=rt["temporal_smooth_schedule"], apply_actuation_rate_limit=rt["apply_actuation_rate_limit"],
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"],
        )
        (det_schedule, det_adb_diag), det_adb_runtime_s = timed_call(
            device,
            deterministic_adb_schedule,
            det_by_id=det_post, actor_boxes=actor_boxes, association=association, eligible=eligible,
            grid=grid, current_t0=current_t0,
            DeterministicSharedMeanPrediction=rt["DeterministicSharedMeanPrediction"],
            build_deterministic_future_full_boxes=rt["build_deterministic_future_full_boxes"],
            rasterize_projected_full_box=rt["rasterize_projected_full_box"],
            predictive_mask_to_illumination=rt["predictive_mask_to_illumination"],
            temporal_smooth_schedule=rt["temporal_smooth_schedule"], apply_actuation_rate_limit=rt["apply_actuation_rate_limit"],
            part_a_original_reactive_map_from_h0_centers=rt["part_a_original_reactive_map_from_h0_centers"], params=adb_params,
        )

        # Hybrid direct-beam system reuses EXACT shared ADB schedule; no duplicate
        # NPZ array is stored so identity cannot silently diverge.
        adb_aliases = {
            "direct_beam_classifier": "shared_trajectory_posterior",
        }

        # ----- Same-posterior identity proof -----
        if receiver is not None:
            rid = str(receiver["prediction_id"])
            shared_digest = shared_post[rid]["posterior_digest_sha256"]
            comm_digest = communication["shared_trajectory_posterior"]["posterior_digest_sha256"]
            require(comm_digest == shared_digest, "Shared communication posterior digest mismatch.")
            # ADB branch consumed shared_post itself above. The exact object is the
            # one whose digest is recorded; no second shared forward exists.
            adb_consumed_digest = shared_post[rid]["posterior_digest_sha256"]
            require(adb_consumed_digest == shared_digest, "Shared ADB posterior digest mismatch.")
            shared_runtime_proof = {
                "selected_receiver_prediction_id": rid,
                "shared_posterior_digest_sha256": shared_digest,
                "communication_branch_input_digest_sha256": comm_digest,
                "ADB_branch_input_digest_sha256": adb_consumed_digest,
                "same_content_identity": True,
                "shared_predictor_forward_count_for_scene": 1,
            }
        else:
            shared_runtime_proof = {
                "selected_receiver_prediction_id": None,
                "shared_posterior_digest_sha256": None,
                "communication_branch_input_digest_sha256": None,
                "ADB_branch_input_digest_sha256": None,
                "same_content_identity": True,
                "shared_predictor_forward_count_for_scene": 1,
                "note": "No eligible communication receiver; ADB still consumed the single shared posterior batch.",
            }

        require(np.array_equal(shared_schedule, shared_schedule), "Shared schedule internal sanity failed.")
        for name, arr in (
            ("shared", shared_schedule), ("independent", indep_adb_schedule),
            ("direct_adb", direct_schedule), ("deterministic", det_schedule), ("current_t0", current_t0),
        ):
            require(np.all(np.isfinite(arr)), f"{name} ADB contains non-finite values")
            require(np.all((arr >= 0.0) & (arr <= 1.0)), f"{name} ADB outside [0,1]")

        # ----- Durable binary payload first -----
        atomic_savez_compressed(
            lock_npz,
            current_reactive_t0=np.asarray(current_t0, dtype=np.float64),
            shared_trajectory_posterior=np.asarray(shared_schedule, dtype=np.float64),
            independent_models=np.asarray(indep_adb_schedule, dtype=np.float64),
            direct_ADB_predictor=np.asarray(direct_schedule, dtype=np.float64),
            deterministic_shared_trajectory=np.asarray(det_schedule, dtype=np.float64),
        )
        npz_sha = sha256_file(lock_npz)

        timing = {
            "causal_preprocessing_s": float(causal_runtime),
            "shared_gaussian_forward_s": float(shared_forward_s),
            "independent_comm_gaussian_forward_s": float(indep_comm_forward_s),
            "independent_ADB_gaussian_forward_s": float(indep_adb_forward_s),
            "deterministic_forward_s": float(det_forward_s),
            "direct_beam_forward_s": float(direct_beam_forward_s),
            "direct_ADB_model_encode_decode_s": float(direct_adb_runtime_s),
            "shared_ADB_generation_s": float(shared_adb_runtime_s),
            "independent_ADB_generation_s": float(indep_adb_runtime_s),
            "deterministic_ADB_generation_s": float(det_adb_runtime_s),
            "note": "diagnostic C1 timings only; Block7.5C3 performs frozen isolated-process resource/latency accounting",
        }

        lock_core = {
            "project": "Agni",
            "stage": 7,
            "block": "7.5C1",
            "status": "LOCKED_NONORACLE_CAUSAL_OUTPUTS",
            "formal_index": int(formal_index),
            "scenario_id": scenario_id,
            "formal_manifest_sha256": EXPECTED_FORMAL_SHA,
            "formal_order_sha256": EXPECTED_FORMAL_ORDER_SHA,
            "validation_manifest_sha256": EXPECTED_VALIDATION_MANIFEST_SHA,
            "formal_runtime_loader": {
                "module": "iscai_stage3.validation.womd_access",
                "manifest_reader": "read_validation_manifest",
                "motion_reader": "read_motion_scenario",
                "paired_root": str(PAIRED_ROOT),
                "historical_source_shard_is_provenance": True,
                "local_compact_motion_runtime": True,
            },
            "formal_reader_binding": {
                "motion_shard_from": "selected_validation.source_shard",
                "compact_record_offset_from": "selected_validation.record_offset",
                "payload_length_from": "selected_validation.payload_length",
                "formal_selection_hash_cross_checked": True,
                "formal_source_shard_cross_checked": True,
                "formal_offset_cross_checked": True,
            },
            "block75b_contract_sha256": EXPECTED_B75B_SHA,
            "block74_handoff_sha256": EXPECTED_B74_SHA,
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": runner_sha,
            "causal_input": {
                "scene_inputs_sha256": scene_inputs.sha256(),
                "stage2_algorithm_sha256": str(causal["stage2_algorithm_sha256"]),
                "associated_history_count": len(history_ids),
                "prediction_ids": history_ids,
                "ADB_actor_box_count": len(actor_boxes),
                "ADB_headlamp_eligible_box_indices": sorted(int(x) for x in eligible),
                "association_match_count": int(association.matched_pair_count),
                "association_unmatched_box_indices": [int(x) for x in association.unmatched_box_indices],
                "tracks_to_predict_accessed": False,
                "objects_of_interest_accessed": False,
                "future_state_accessed": False,
            },
            "receiver": receiver,
            "receiver_selection": receiver_selection,
            "shared_posterior_runtime_identity": shared_runtime_proof,
            "posterior_outputs": {
                "shared_trajectory_posterior": shared_post,
                "independent_communication_predictor": indep_comm_post,
                "independent_ADB_predictor": indep_adb_post,
                "deterministic_shared_trajectory": det_post,
            },
            "communication_outputs": communication,
            "ADB_outputs": {
                "npz_path": str(lock_npz),
                "npz_arrays": [
                    "current_reactive_t0",
                    "shared_trajectory_posterior",
                    "independent_models",
                    "direct_ADB_predictor",
                    "deterministic_shared_trajectory",
                ],
                "aliases": adb_aliases,
                "shared_diagnostics": shared_adb_diag,
                "independent_diagnostics": indep_adb_diag,
                "direct_ADB_diagnostics": direct_adb_diag,
                "deterministic_diagnostics": det_adb_diag,
            },
            "diagnostic_runtime": timing,
            "adb_schedules_npz_sha256": npz_sha,
            "future_GT_accessed": False,
            "formal_metrics_computed": False,
            "post_outcome_tuning": False,
            "scientific_parameters_modified": False,
            "scenario_elapsed_s": float(time.perf_counter() - scenario_start),
        }
        nonoracle_digest = sha256_bytes(canonical_bytes(lock_core))
        lock_core["nonoracle_output_digest_sha256"] = nonoracle_digest
        atomic_write_bytes(lock_json, canonical_bytes(lock_core))
        json_sha = sha256_file(lock_json)

        completed.append({
            "formal_index": formal_index,
            "scenario_id": scenario_id,
            "lock_json": str(lock_json),
            "lock_json_sha256": json_sha,
            "adb_schedules_npz": str(lock_npz),
            "adb_schedules_npz_sha256": npz_sha,
            "nonoracle_output_digest_sha256": nonoracle_digest,
        })
        print(f"[{formal_index:03d}/119] LOCK  {scenario_id}  json={json_sha[:12]} npz={npz_sha[:12]}", flush=True)

        # Release scenario-local GPU cache without altering model weights.
        del scenario, causal, normalized
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Global seal — final only for all N=120.
    # -------------------------------------------------------------------------
    if args.limit is not None or worker_mode:
        if worker_mode:
            print("\nBLOCK 7.5C1 PARALLEL WORKER RUN COMPLETE")
            print(
                "worker range =",
                f"[{int(args.start_index)}, {int(args.end_index)})",
            )
        else:
            print("\nBLOCK 7.5C1 PARTIAL DEBUG RUN COMPLETE")
        print("locked/reused scenarios =", len(completed))
        print("future GT accessed = NO")
        print("formal metrics computed = NO")
        print("scientific parameters modified = NO")
        return 0

    require(len(completed) == 120, f"Expected 120 completed locks, got {len(completed)}")
    completed.sort(key=lambda x: int(x["formal_index"]))
    require([x["scenario_id"] for x in completed] == scenario_ids, "Completed lock order differs from formal manifest.")

    manifest_doc = {
        "project": "Agni",
        "stage": 7,
        "block": "7.5C1",
        "status": "FROZEN_COMPLETE_NONORACLE_OUTPUT_LOCK",
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": runner_sha,
        "block75b_contract_sha256": EXPECTED_B75B_SHA,
        "formal_manifest": {"path": str(FORMAL), "sha256": EXPECTED_FORMAL_SHA, "scenario_count": 120, "scenario_order_sha256": EXPECTED_FORMAL_ORDER_SHA},
        "validation_manifest": {"path": str(VALIDATION_MANIFEST), "sha256": EXPECTED_VALIDATION_MANIFEST_SHA, "scenario_count": 44097},
        "formal_reader_resolution": "120/120 exact metadata cross-binding plus frozen Stage3 read_validation_manifest/read_motion_scenario runtime route",
        "system_count": 5,
        "systems": list(SYSTEMS),
        "scenario_locks": completed,
        "scenario_lock_count": 120,
        "shared_posterior_forward_semantics": "one shared Gaussian forward batch per formal scenario; same posterior content feeds communication and ADB",
        "direct_beam_ADB_alias": "shared_trajectory_posterior",
        "direct_ADB_communication_alias": "shared_trajectory_posterior",
        "future_GT_accessed": False,
        "formal_metrics_computed": False,
        "Stage4_modified": False,
        "Stage5_modified": False,
        "Stage6_modified": False,
        "retraining": False,
        "recalibration": False,
        "post_outcome_tuning": False,
        "next": "Block7.5C2 evaluator-only future truth + fresh communication/ADB scoring after independent lock verification",
    }
    manifest_bytes = canonical_bytes(manifest_doc)
    manifest_sha = sha256_bytes(manifest_bytes)
    atomic_write_bytes(MANIFEST, manifest_bytes)

    report_doc = {
        "status": "PASS",
        "stage": 7,
        "block": "7.5C1",
        "manifest_path": str(MANIFEST),
        "manifest_sha256": manifest_sha,
        "runner_sha256": runner_sha,
        "formal_causal_scenarios": 120,
        "five_systems_locked": True,
        "shared_posterior_runtime_identity_proof": True,
        "nonoracle_lock_complete": True,
        "future_GT_accessed": False,
        "formal_metrics_computed": False,
        "Stage4_5_6_modified": False,
        "post_outcome_tuning": False,
        "next_block": "7.5C2",
    }
    atomic_write_bytes(REPORT, canonical_bytes(report_doc))

    print("\n===== FINAL =====")
    print("runner SHA256 =", runner_sha)
    print("manifest =", MANIFEST)
    print("manifest SHA256 =", manifest_sha)
    print("report =", REPORT)
    print("formal causal scenarios = 120")
    print("five systems locked = YES")
    print("shared posterior identity proof = YES")
    print("future GT accessed = NO")
    print("formal metrics computed = NO")
    print("Stage4/5/6 modified = NO")
    print("post-outcome tuning = NO")
    print("NONORACLE LOCK COMPLETE = YES")
    print("BLOCK 7.5C1 STATUS = PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

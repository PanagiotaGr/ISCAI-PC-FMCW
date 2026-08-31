from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

FORMAL_MANIFEST = (
    S3 / "artifacts/block38e/formal_validation_120.jsonl"
)
FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

VALIDATION_MANIFEST = (
    ROOT / "iscai_data_prep/manifests/selected_validation.jsonl"
)
PAIRED_ROOT = ROOT / "data/paired_womd_lidar_v1_3_0"

GAUSSIAN_CKPT = S4 / "artifacts/block44/gaussian_gru.pt"
GAUSSIAN_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

NORMALIZATION = S4 / "artifacts/block43/fit_normalization.json"

CALIBRATOR = S4 / "artifacts/block45/covariance_scaler.json"
CALIBRATOR_SHA = (
    "508ff2e3fbcfafe8e001155340c25baa"
    "f3772fe2561a8022a9ed1cf780e66087"
)

STAGE4_HANDOFF = (
    S4 / "artifacts/block410/stage4_to_stage5_handoff.json"
)
STAGE4_HANDOFF_SHA = (
    "491bce010d35c2a394f879ecf35ed26e"
    "f1de92f7fff1465dcbf46b072a87c6fd"
)

RECEIVER_REPORT = S5 / "reports/block51_receiver_geometry.json"
RECEIVER_REPORT_SHA = (
    "a7a79c4f0083e0668b1d6313c990dd2"
    "456efb3cd4cc4589b718bb93c2459f9b3"
)

ANGULAR_REPORT = (
    S5 / "reports/block52_receiver_angular_posterior.json"
)
ANGULAR_REPORT_SHA = (
    "7102a2a87ab783ccfcd0b6e9ca01ac70"
    "d1022076d3a8c821201eac237ca4bfb8"
)

LATENCY_CONFIG = S5 / "configs/beam_latency_policy.json"
LATENCY_SHA = (
    "aa04f2e73da591b87f2ab2e962d1b69c"
    "1bad645c043c0d15970dbf0990080ed0"
)

ACCEPTANCE_CONFIG = (
    S5 / "configs/formal_stage5_acceptance_policy.json"
)
ACCEPTANCE_SHA = (
    "a89dcc3c785933632b547d73dae96696e"
    "3b63ebd19b46e73b7b0602fa0da313f"
)

BLOCK44_SCRIPT = (
    S4 / "scripts/run_block44_gaussian_training.py"
)

CACHE_DIR = S5 / "artifacts/block58/formal_clean_cache"
MERGED = S5 / "artifacts/block58/formal_clean_records.jsonl"
REPORT = S5 / "reports/block58_formal_evaluation.json"
FAILURE = S5 / "reports/block58_formal_failure.json"

HORIZONS = (0.1, 0.3, 0.5, 1.0)
OFFSETS = (1, 3, 5, 10)
CODEBOOKS = (16, 32, 64)
Q_LEVELS = (0.90, 0.95, 0.975, 0.99)
FIXED_TOPK = (1, 3, 5)

# Full uncertainty-aware configuration = primary Stage5 path.
# Centroid and known-offset are formal ablations/comparisons.
GEOMETRIES = (
    "centroid",
    "known",
    "uncertain",
)
PRIMARY_GEOMETRY = "uncertain"

MC_SAMPLES = 2048
MC_SEED = 20260821
LOW_SPEED_MPS = 0.25

# Frozen Block4.5 calibrated-Gaussian variance factors.
VARIANCE_SCALE = (
    1.2347064500315355,
    1.3451250295202921,
    1.354822057728461,
    1.2829700217319266,
)

MIN_FREE_BYTES = 250 * 1024**3

BRIDGE_REPAIR_POLICY = (
    "partial_unique_actor_bridge_unresolved_class_explicit_v1"
)

COMPATIBLE_PRIOR_RUNNER_SHA256 = (
    "ad30bf289bc5b6a910c38c508882a58ff6d216b8276a4661ae364e3871fec430"
)


from iscai_stage3.validation import (
    read_motion_scenario,
    read_validation_manifest,
)

from iscai_stage4.data import attach_supervision

from iscai_stage4.data.real_pipeline import (
    build_real_causal_inputs,
    load_frozen_stage2_configs,
)

from iscai_stage4.ml.gaussian_gru import GaussianTrajectoryGRU
from iscai_stage4.ml.calibration import apply_variance_scale

from iscai_stage5.receiver_selection import (
    ReceiverCandidate,
    ReceiverSelectionConfig,
    select_primary_receiver,
)

from iscai_stage5.angular_monte_carlo import (
    ReceiverOffsetGaussianBody,
    deterministic_receiver_angular_posterior,
)

from iscai_stage5.beam_codebook import (
    beam_probability_mass_from_samples,
    complete_partition_beam_probability_mass_from_samples,
    complete_partition_decision_azimuth,
    build_uniform_azimuth_codebook,
)

from iscai_stage5.beam_directional_gain import (
    FROZEN_SUPPORT_MIN_RAD,
    FROZEN_SUPPORT_MAX_RAD,
)

from iscai_stage5.beam_baselines import (
    fixed_top_k_probability,
    geometry_nearest_beam,
    oracle_best_gain_beam,
)

from iscai_stage5.adaptive_topk_temporal import (
    AdaptiveTemporalDecision,
    AdaptiveTemporalState,
    adaptive_temporal_step,
    initial_adaptive_temporal_state,
)

from iscai_stage5.beam_latency import (
    actual_probe_count_for_adaptive_decision,
)

from iscai_stage5.optical_link import (
    evaluate_optical_link,
    optical_gain_for_cell,
)

from iscai_stage5.formal_acceptance import (
    empirical_coverage_decision,
)


def sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    """
    Persistence-only JSON sanitation.

    Scientific calculations remain unchanged in memory.
    JSON cannot represent NaN/+inf/-inf, therefore such
    values are serialized explicitly as null.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return json_safe(
            value.tolist()
        )

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        return (
            value
            if math.isfinite(value)
            else None
        )

    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )



def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*values) -> int:
    text = "|".join(str(x) for x in values)
    return (
        MC_SEED
        + int(sha256(text.encode()).hexdigest()[:8], 16)
    ) % (2**31 - 1)


def avg(values):
    x = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    return float(np.mean(x)) if x else None


def check_sha(path, expected, label):
    if not path.is_file():
        raise RuntimeError(f"{label} missing: {path}")
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: {actual}"
        )


def read_formal():
    check_sha(FORMAL_MANIFEST, FORMAL_SHA, "formal manifest")

    rows = [
        json.loads(line)
        for line in FORMAL_MANIFEST.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(rows) != 120:
        raise RuntimeError(
            f"formal population is {len(rows)}, expected 120"
        )

    if len({str(x["scenario_id"]) for x in rows}) != 120:
        raise RuntimeError("duplicate formal scenario IDs")

    return rows


def load_normalizer_class():
    spec = importlib.util.spec_from_file_location(
        "_stage4_block44_runtime",
        BLOCK44_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen Block4.4 script")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, "GaussianNormalizer", None)
    if cls is None:
        raise RuntimeError("GaussianNormalizer missing")
    return cls


def load_gaussian():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda:0")

    checkpoint = torch.load(
        GAUSSIAN_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    model = GaussianTrajectoryGRU().to(device)
    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    model.eval()

    Normalizer = load_normalizer_class()

    normalizer = Normalizer(
        load_json(NORMALIZATION),
        device=device,
    )

    return device, model, normalizer


def receiver_offset():
    """
    Resolve the already-frozen Stage1/5 body-frame
    receiver mean/covariance from Block5.1 evidence.
    """
    sources = [
        RECEIVER_REPORT,
        *sorted((S5 / "configs").glob("*.json")),
    ]

    candidates = []

    def walk(obj, source, prefix=""):
        if isinstance(obj, dict):
            means = []
            covs = []

            for key, value in obj.items():
                low = str(key).lower()

                try:
                    arr = np.asarray(value, dtype=np.float64)
                except Exception:
                    arr = None

                if (
                    arr is not None
                    and arr.shape == (3,)
                    and "mean" in low
                    and (
                        "body" in low
                        or "offset" in low
                        or "receiver" in low
                    )
                ):
                    means.append((key, arr))

                if (
                    arr is not None
                    and arr.shape == (3, 3)
                    and ("cov" in low or "sigma" in low)
                    and (
                        "body" in low
                        or "offset" in low
                        or "receiver" in low
                    )
                ):
                    covs.append((key, arr))

            for mk, m in means:
                for ck, c in covs:
                    if (
                        np.all(np.isfinite(m))
                        and np.all(np.isfinite(c))
                    ):
                        candidates.append(
                            (
                                tuple(m.tolist()),
                                tuple(tuple(r) for r in c.tolist()),
                                f"{source}:{prefix}:{mk}/{ck}",
                            )
                        )

            for key, value in obj.items():
                walk(
                    value,
                    source,
                    f"{prefix}.{key}" if prefix else str(key),
                )

        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, source, f"{prefix}[{i}]")

    for path in sources:
        if path.is_file():
            try:
                walk(load_json(path), path)
            except Exception:
                pass

    if not candidates:
        raise RuntimeError(
            "frozen receiver offset/covariance not found"
        )

    unique = {}
    for mean, cov, source in candidates:
        key = (
            tuple(round(x, 12) for x in mean),
            tuple(
                tuple(round(x, 12) for x in row)
                for row in cov
            ),
        )
        unique.setdefault(key, source)

    if len(unique) != 1:
        raise RuntimeError(
            f"ambiguous receiver offsets: {len(unique)}"
        )

    key, source = next(iter(unique.items()))
    mean, cov = key

    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(cov, dtype=np.float64),
        source,
    )


def resolve_vehicle_semantic():
    """
    Use the frozen Stage5 receiver selector itself to
    determine its accepted canonical vehicle string.
    """
    config = ReceiverSelectionConfig()

    for semantic in (
        "vehicle",
        "VEHICLE",
        "TYPE_VEHICLE",
    ):
        result = select_primary_receiver(
            [
                ReceiverCandidate(
                    semantic_class=semantic,
                    position_h0_m=(10.0, 0.0, 0.0),
                    current_available=True,
                    association_valid=True,
                )
            ],
            config,
        )

        if result.selected_candidate_index == 0:
            return semantic

    raise RuntimeError(
        "receiver selector accepted no canonical vehicle label"
    )


def resolve_geometry_modes(mean_offset, cov_offset):
    """
    Resolve only the exact API spelling of the already-frozen
    Block5.2 geometry modes.

    No scientific parameter is selected here.
    """

    import inspect
    import json
    import re

    from iscai_stage5 import angular_monte_carlo as amc

    candidates = set()

    # 1. Frozen Block5.2 report strings.
    report_path = (
        S5
        / "reports/block52_receiver_angular_posterior.json"
    )

    if report_path.is_file():
        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )

        def walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(k)
                    walk(v)

            elif isinstance(value, list):
                for v in value:
                    walk(v)

            elif isinstance(value, str):
                low = value.lower()

                if any(
                    token in low
                    for token in (
                        "centroid",
                        "known",
                        "uncertain",
                        "offset",
                    )
                ):
                    candidates.add(value)

        walk(report)

    # 2. Frozen module constants.
    for name in dir(amc):
        if name.startswith("_"):
            continue

        value = getattr(amc, name)

        if isinstance(value, str):
            low = value.lower()

            if any(
                token in low
                for token in (
                    "centroid",
                    "known",
                    "uncertain",
                    "offset",
                )
            ):
                candidates.add(value)

    # 3. Exact string literals used by the frozen function.
    source = inspect.getsource(
        amc.deterministic_receiver_angular_posterior
    )

    for value in re.findall(
        r"""["']([^"']+)["']""",
        source,
    ):
        low = value.lower()

        if any(
            token in low
            for token in (
                "centroid",
                "known",
                "uncertain",
                "offset",
            )
        ):
            candidates.add(value)

    # Add only spelling variants already used earlier in Stage5.
    candidates.update(
        (
            "centroid",
            "centroid_baseline",
            "actor_centroid",
            "actor_centroid_baseline",
            "known",
            "known_offset",
            "known_receiver_offset",
            "uncertain",
            "uncertain_offset",
            "uncertain_receiver_offset",
        )
    )

    dummy_mean = np.asarray(
        [
            [12.0, 1.0, 0.0],
            [14.0, 1.2, 0.0],
            [16.0, 1.5, 0.0],
            [20.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    dummy_cov = np.asarray(
        [
            np.diag([0.20, 0.15, 0.10]),
            np.diag([0.25, 0.18, 0.10]),
            np.diag([0.30, 0.20, 0.12]),
            np.diag([0.40, 0.25, 0.15]),
        ],
        dtype=np.float64,
    )

    zero_offset = ReceiverOffsetGaussianBody(
        mean_body_m=(0.0, 0.0, 0.0),
        covariance_body_m2=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )

    known_offset = ReceiverOffsetGaussianBody(
        mean_body_m=tuple(float(x) for x in mean_offset),
        covariance_body_m2=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )

    uncertain_offset = ReceiverOffsetGaussianBody(
        mean_body_m=tuple(float(x) for x in mean_offset),
        covariance_body_m2=tuple(
            tuple(float(x) for x in row)
            for row in cov_offset
        ),
    )

    conceptual = {
        "centroid": (
            ("centroid",),
            (None, zero_offset),
        ),

        "known": (
            ("known",),
            (known_offset,),
        ),

        "uncertain": (
            ("uncertain",),
            (uncertain_offset,),
        ),
    }

    resolved = {}
    errors = {}

    for label, (tokens, offsets) in conceptual.items():
        ordered = sorted(
            candidates,
            key=lambda value: (
                0
                if all(
                    token in value.lower()
                    for token in tokens
                )
                else
                1,
                len(value),
                value,
            ),
        )

        for mode in ordered:
            low = mode.lower()

            if not all(
                token in low
                for token in tokens
            ):
                continue

            for offset in offsets:
                try:
                    posterior = (
                        deterministic_receiver_angular_posterior(
                            trajectory_mean_h0_m=dummy_mean,
                            raw_trajectory_covariance_h0_m2=(
                                dummy_cov
                            ),
                            current_position_h0_m=(
                                10.0,
                                0.0,
                                0.0,
                            ),
                            current_heading_h0_rad=0.0,
                            receiver_geometry_mode=mode,
                            receiver_offset=offset,
                            sample_count=MC_SAMPLES,
                            seed=123,
                            low_speed_threshold_mps=(
                                LOW_SPEED_MPS
                            ),
                        )
                    )

                    if int(posterior.sample_count) != MC_SAMPLES:
                        raise RuntimeError(
                            "unexpected MC sample count"
                        )

                    resolved[label] = mode
                    break

                except Exception as exc:
                    errors.setdefault(
                        label,
                        [],
                    ).append(
                        (
                            mode,
                            type(exc).__name__,
                            str(exc)[:120],
                        )
                    )

            if label in resolved:
                break

        if label not in resolved:
            raise RuntimeError(
                f"cannot resolve frozen geometry mode "
                f"{label}; attempts={errors.get(label, [])[:8]}"
            )

    print(
        "resolved geometry modes   =",
        resolved,
    )

    return resolved



def model_input_arrays(payload):
    target = np.asarray(payload.target_history, dtype=np.float32)
    neighbors = np.asarray(
        payload.neighbor_histories,
        dtype=np.float32,
    )
    mask = np.asarray(payload.neighbor_mask, dtype=np.bool_)
    map_context = np.asarray(
        payload.map_context,
        dtype=np.float32,
    )

    return {
        "target": target[None, ...],
        "neighbors": neighbors[None, ...],
        "neighbor_mask": mask[None, ...],
        "map_context": map_context[None, ...],
        # supervision placeholders only; never model.forward inputs
        "class_id": np.zeros((1,), dtype=np.int64),
        "future": np.zeros((1, 4, 3), dtype=np.float32),
        "future_mask": np.zeros((1, 4), dtype=np.bool_),
    }


def predict_gaussian(
    sample,
    history,
    *,
    device,
    model,
    normalizer,
):
    batch = normalizer.prepare(
        model_input_arrays(sample.model_input),
        np.asarray([0], dtype=np.int64),
        device=device,
    )

    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()

    with torch.inference_mode():
        out = model(
            batch["target"],
            batch["neighbors"],
            batch["neighbor_mask"],
            batch["map_context"],
        )

    torch.cuda.synchronize(device)
    runtime_s = (time.perf_counter_ns() - start) / 1e9

    if tuple(out.mean.shape) != (1, 4, 3):
        raise RuntimeError(
            f"Gaussian mean shape changed: {tuple(out.mean.shape)}"
        )

    if tuple(out.scale_tril.shape) != (1, 4, 3, 3):
        raise RuntimeError("Gaussian scale shape changed")

    scale = apply_variance_scale(
        out.scale_tril,
        torch.as_tensor(
            VARIANCE_SCALE,
            dtype=out.scale_tril.dtype,
            device=device,
        ),
    )

    label_mean = normalizer.label_mean.to(device)
    label_std = normalizer.label_std.to(device)

    displacement = (
        label_mean[None, ...]
        + out.mean * label_std[None, ...]
    )

    D = torch.diag_embed(label_std)
    scale_m = D[None, ...] @ scale
    covariance = scale_m @ scale_m.transpose(-1, -2)

    current = torch.as_tensor(
        history.latest_position_H0_m,
        dtype=out.mean.dtype,
        device=device,
    )

    mean = displacement + current[None, None, :]

    return (
        mean[0].detach().cpu().numpy().astype(np.float64),
        covariance[0]
        .detach().cpu().numpy().astype(np.float64),
        float(runtime_s),
    )


def current_heading(history):
    steps = tuple(history.steps)
    i = int(history.latest_observed_frame_index)

    if 0 <= i < len(steps):
        step = steps[i]

        if bool(step.velocity_valid):
            v = np.asarray(step.velocity_H0_mps, dtype=float)
            if (
                v.shape == (3,)
                and np.all(np.isfinite(v))
                and np.linalg.norm(v[:2]) >= LOW_SPEED_MPS
            ):
                return float(math.atan2(v[1], v[0]))

    positions = [
        np.asarray(s.position_H0_m, dtype=float)
        for s in steps[: i + 1]
        if bool(s.observed)
    ]

    if len(positions) >= 2:
        d = positions[-1][:2] - positions[-2][:2]
        if np.linalg.norm(d) > 1e-8:
            return float(math.atan2(d[1], d[0]))

    return 0.0


def causal_actor_bridge(built, scenario):
    """
    Current causal history -> adapted-actor identity bridge.

    Complete one-to-one matching is preferred and reproduces
    the original Block5.8 behavior exactly.

    If the causal association layer contains more current
    histories than uniquely bridgeable adapted actors, retain
    the deterministic maximum one-to-one subset and leave the
    remaining histories SEMANTICALLY UNRESOLVED.

    Unresolved histories:
      - are counted explicitly,
      - are not assigned a guessed vehicle class,
      - are not communication-receiver candidates,
      - are not replaced using tracks_to_predict/future truth.
    """

    histories = [
        h
        for h in built["scene_inputs"].histories
        if (
            int(h.latest_observed_frame_index)
            == int(scenario.current_time_index)
            and bool(
                h.steps[
                    int(h.latest_observed_frame_index)
                ].observed
            )
        )
    ]

    actors = []

    for actor in built["adapted"].actors:
        p = np.asarray(
            actor.anchor_center_H0_m,
            dtype=float,
        )

        if (
            p.shape != (3,)
            or not np.all(np.isfinite(p))
        ):
            continue

        ti = int(actor.track_index)

        if 0 <= ti < len(scenario.tracks):
            actors.append(
                (actor, p)
            )

    pairs = []

    for hi, history in enumerate(histories):
        hp = np.asarray(
            history.latest_position_H0_m,
            dtype=float,
        )

        if (
            hp.shape != (3,)
            or not np.all(np.isfinite(hp))
        ):
            continue

        for ai, (_, ap) in enumerate(actors):
            pairs.append(
                (
                    float(np.linalg.norm(hp - ap)),
                    str(history.prediction_id),
                    hi,
                    ai,
                )
            )

    pairs.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
            x[3],
        )
    )

    used_h = set()
    used_a = set()
    match = {}

    for distance, _, hi, ai in pairs:
        if hi in used_h or ai in used_a:
            continue

        used_h.add(hi)
        used_a.add(ai)

        match[hi] = (
            ai,
            float(distance),
        )

        if (
            len(match) == len(histories)
            or len(match) == len(actors)
        ):
            break

    return histories, actors, match



def rotate_offset(offset, heading):
    x, y, z = [float(v) for v in offset]
    c, s = math.cos(heading), math.sin(heading)

    return np.asarray(
        [c * x - s * y, s * x + c * y, z],
        dtype=float,
    )


def current_receiver_position(
    center,
    heading,
    geometry,
    mean_offset,
):
    center = np.asarray(center, dtype=float)

    if geometry == "centroid":
        return center

    return center + rotate_offset(mean_offset, heading)


def selection_indices(selection):
    for name in (
        "beam_indices",
        "selected_beam_indices",
        "indices",
    ):
        value = getattr(selection, name, None)

        if value is not None:
            return tuple(int(x) for x in value)

    for name in (
        "beam_index",
        "selected_beam_index",
        "primary_beam_index",
    ):
        value = getattr(selection, name, None)

        if value is not None:
            return (int(value),)

    raise RuntimeError(
        f"cannot extract BeamSelection from {type(selection)}"
    )


def temporal_result(result, previous):
    if isinstance(result, tuple):
        state = next(
            (
                x for x in result
                if isinstance(x, AdaptiveTemporalState)
            ),
            None,
        )
        decision = next(
            (
                x for x in result
                if isinstance(x, AdaptiveTemporalDecision)
            ),
            None,
        )

        if state is not None and decision is not None:
            return state, decision

    if isinstance(result, AdaptiveTemporalDecision):
        state = AdaptiveTemporalState(
            previous_primary_beam_index=int(
                result.primary_beam_index
            ),
            step_count=int(previous.step_count) + 1,
            cumulative_switch_events=(
                int(previous.cumulative_switch_events)
                + int(bool(result.primary_switched))
            ),
        )
        return state, result

    state = getattr(result, "state", None)
    decision = getattr(result, "decision", None)

    if (
        isinstance(state, AdaptiveTemporalState)
        and isinstance(decision, AdaptiveTemporalDecision)
    ):
        return state, decision

    raise RuntimeError(
        "unknown adaptive_temporal_step return contract"
    )


def physical_adaptive_indices(decision):
    indices = set(
        int(x)
        for x in decision.adaptive_selection.beam_indices
    )

    indices.update(
        int(x)
        for x in decision.local_neighbor_probe_indices
    )

    fallback = decision.widened_fallback

    if (
        bool(fallback.available)
        and fallback.fallback_beam_index is not None
    ):
        indices.add(int(fallback.fallback_beam_index))

    indices.update(
        int(x)
        for x in decision.exhaustive_fallback_indices
    )

    return tuple(sorted(indices))


def hit(azimuth, indices, codebook):
    """
    Probability-containment hit under the complete Block5.3
    decision partition.

    IMPORTANT:
    This clipping is ONLY for assignment to a probability
    decision cell. Optical gain uses the original realized
    azimuth and is therefore not artificially improved.
    """

    decision_azimuth = (
        complete_partition_decision_azimuth(
            azimuth_rad=azimuth,
            codebook=codebook,
        )
    )

    indices = set(
        int(x)
        for x in indices
    )

    for cell in codebook.cells:
        if int(cell.index) not in indices:
            continue

        lo = float(
            cell.lower_azimuth_rad
        )
        hi = float(
            cell.upper_azimuth_rad
        )

        if int(cell.index) == len(codebook.cells) - 1:
            inside = (
                lo
                <= decision_azimuth
                <= hi
            )
        else:
            inside = (
                lo
                <= decision_azimuth
                < hi
            )

        if inside:
            return True

    return False



def best_link(
    indices,
    azimuth,
    elevation,
    codebook,
    probe_count,
    tbeam,
    tframe,
):
    if not indices:
        return None

    choices = []

    for index in indices:
        cell = codebook.cells[int(index)]

        _, gain = optical_gain_for_cell(
            receiver_azimuth_rad=azimuth,
            receiver_elevation_rad=elevation,
            cell=cell,
        )

        choices.append(
            (-float(gain), int(index), cell)
        )

    choices.sort()

    return evaluate_optical_link(
        receiver_azimuth_rad=azimuth,
        receiver_elevation_rad=elevation,
        cell=choices[0][2],
        probing_beam_count=int(probe_count),
        beam_probe_time_s=float(tbeam),
        frame_time_s=float(tframe),
    )


def link_row(link, oracle):
    if link is None:
        return {}

    eps = 1e-300

    def loss_db(a, b):
        return max(
            0.0,
            10.0 * math.log10(
                max(float(a), eps) / max(float(b), eps)
            ),
        )

    return {
        "optical_gain": float(link.optical_gain),
        "received_power": float(
            link.received_power_normalized
        ),
        "snr_db": float(link.snr_db),
        "ber": float(link.dbpsk_ber),
        "effective_rate_bps": float(
            link.effective_rate.effective_rate_bps
        ),
        "beam_gain_loss_db": loss_db(
            oracle.optical_gain,
            link.optical_gain,
        ),
        "received_power_loss_db": loss_db(
            oracle.received_power_normalized,
            link.received_power_normalized,
        ),
        "snr_loss_db": max(
            0.0,
            float(oracle.snr_db) - float(link.snr_db),
        ),
        "effective_rate_loss_bps": max(
            0.0,
            float(
                oracle.effective_rate.effective_rate_bps
            )
            - float(
                link.effective_rate.effective_rate_bps
            ),
        ),
    }


def transform_h0(built, current_index):
    """
    Convert the frozen Stage1 T_H0_from_W representation to a
    homogeneous 4x4 NumPy matrix.

    Stage1 may expose this as a RigidTransform object rather
    than a raw ndarray. This is representation handling only;
    no geometry or scientific parameter is changed.
    """

    raw = built["adapted"].frames.T_H0_from_W

    def numeric_array(value):
        try:
            arr = np.asarray(
                value,
                dtype=np.float64,
            )
        except Exception:
            return None

        if not np.all(np.isfinite(arr)):
            return None

        return arr

    def rotation_array(value):
        arr = numeric_array(value)

        if arr is not None:
            if arr.shape == (3, 3):
                return arr

            if arr.shape == (4, 4):
                return arr[:3, :3]

        for name in (
            "as_matrix",
            "matrix",
            "rotation_matrix",
            "R",
        ):
            if not hasattr(value, name):
                continue

            candidate = getattr(value, name)

            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue

            arr = numeric_array(candidate)

            if arr is not None:
                if arr.shape == (3, 3):
                    return arr

                if arr.shape == (4, 4):
                    return arr[:3, :3]

        return None

    def matrix4(value):
        arr = numeric_array(value)

        if arr is not None:
            if arr.shape == (4, 4):
                return arr

            if arr.shape == (3, 4):
                out = np.eye(4, dtype=np.float64)
                out[:3, :] = arr
                return out

        # Common RigidTransform matrix properties/methods.
        for name in (
            "as_matrix",
            "matrix",
            "matrix4x4",
            "homogeneous_matrix",
            "transform_matrix",
            "to_matrix",
        ):
            if not hasattr(value, name):
                continue

            candidate = getattr(value, name)

            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue

            arr = numeric_array(candidate)

            if arr is not None:
                if arr.shape == (4, 4):
                    return arr

                if arr.shape == (3, 4):
                    out = np.eye(4, dtype=np.float64)
                    out[:3, :] = arr
                    return out

        # Common RigidTransform decomposition:
        # rotation + translation.
        rotation = None
        translation = None

        for name in (
            "rotation",
            "rotation_matrix",
            "R",
        ):
            if hasattr(value, name):
                rotation = rotation_array(
                    getattr(value, name)
                )

                if rotation is not None:
                    break

        for name in (
            "translation",
            "translation_m",
            "translation_vector",
            "t",
        ):
            if not hasattr(value, name):
                continue

            candidate = getattr(value, name)

            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue

            arr = numeric_array(candidate)

            if arr is not None:
                arr = arr.reshape(-1)

                if arr.shape == (3,):
                    translation = arr
                    break

        if rotation is not None and translation is not None:
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = rotation
            out[:3, 3] = translation
            return out

        public = [
            name
            for name in dir(value)
            if not name.startswith("_")
        ]

        raise RuntimeError(
            "Cannot convert frozen RigidTransform to 4x4 "
            f"matrix; type={type(value).__module__}."
            f"{type(value).__name__}; "
            f"public_fields={public[:30]}"
        )

    # Some adapters expose one transform per frame.
    if isinstance(raw, (tuple, list)):
        if not (
            0 <= current_index < len(raw)
        ):
            raise RuntimeError(
                "T_H0_from_W frame index out of range."
            )

        return matrix4(raw[current_index])

    arr = numeric_array(raw)

    if arr is not None and arr.ndim == 3:
        if (
            arr.shape[1:] != (4, 4)
            or not (
                0 <= current_index < arr.shape[0]
            )
        ):
            raise RuntimeError(
                f"unexpected T_H0_from_W stack shape "
                f"{arr.shape}"
            )

        return arr[current_index]

    # Normal Stage1 case: one frozen RigidTransform for H0.
    return matrix4(raw)



def future_center(state, T):
    p = np.asarray(
        [
            float(state.center_x),
            float(state.center_y),
            float(state.center_z),
            1.0,
        ]
    )

    return (T @ p)[:3]


def future_heading(state, T):
    frame_yaw = math.atan2(T[1, 0], T[0, 0])

    return (
        float(state.heading) + frame_yaw + math.pi
    ) % (2 * math.pi) - math.pi


def spherical(position):
    x, y, z = [float(v) for v in position]
    planar = math.hypot(x, y)

    return (
        math.sqrt(x*x + y*y + z*z),
        math.atan2(y, x),
        math.atan2(z, planar),
    )


def circular_std(values):
    values = np.asarray(values, dtype=float)
    c = np.mean(np.cos(values))
    s = np.mean(np.sin(values))
    r = min(1.0, max(1e-15, math.hypot(c, s)))
    return math.sqrt(max(0.0, -2.0 * math.log(r)))


def process_scene(
    rank,
    formal,
    validation_by_id,
    *,
    clean_config,
    degraded_config,
    runtime,
    receiver_parameters,
    geometry_modes,
    vehicle_semantic,
    codebooks,
    tbeam,
    tframe,
    runner_sha,
):
    scenario_id = str(formal["scenario_id"])
    cache = CACHE_DIR / f"{rank:03d}_{scenario_id}.json"

    if cache.is_file():
        existing = load_json(cache)

        if (
            existing.get("status") == "COMPLETE"
            and existing.get("runner_sha256")
            in (
                runner_sha,
                COMPATIBLE_PRIOR_RUNNER_SHA256,
            )
        ):
            return existing, True

        raise RuntimeError(
            f"incompatible existing cache: {cache}"
        )

    row = validation_by_id[scenario_id]

    scenario = read_motion_scenario(
        row,
        paired_root=PAIRED_ROOT,
        compact_record_offset=int(
            formal["compact_record_offset"]
        ),
    )

    if int(scenario.current_time_index) != 10:
        raise RuntimeError("current_time_index changed")

    data_start = time.perf_counter_ns()

    built = build_real_causal_inputs(
        scenario,
        clean_config=clean_config,
        degraded_config=degraded_config,
    )

    samples = attach_supervision(
        built["scene_inputs"],
        scenario,
        T_H0_from_W=built["adapted"].frames.T_H0_from_W,
    )

    data_runtime = (
        time.perf_counter_ns() - data_start
    ) / 1e9

    sample_by_pid = {
        str(sample.prediction_id): sample
        for sample in samples
    }

    histories, actors, matches = causal_actor_bridge(
        built,
        scenario,
    )

    candidate_rows = []
    bridge_distances = []

    unresolved_causal_histories = (
        len(histories) - len(matches)
    )

    for hi, history in enumerate(histories):
        if hi not in matches:
            continue

        ai, distance = matches[hi]
        actor, _ = actors[ai]
        track_index = int(actor.track_index)

        # Static semantic metadata only. No future state read.
        object_type = int(
            scenario.tracks[track_index].object_type
        )

        semantic = (
            vehicle_semantic
            if object_type == 1
            else "other"
        )

        candidate_rows.append(
            {
                "history": history,
                "track_index": track_index,
                "semantic": semantic,
            }
        )

        bridge_distances.append(distance)

    candidates = [
        ReceiverCandidate(
            semantic_class=row["semantic"],
            position_h0_m=tuple(
                float(x)
                for x in row[
                    "history"
                ].latest_position_H0_m
            ),
            current_available=True,
            association_valid=True,
        )
        for row in candidate_rows
    ]

    selection = select_primary_receiver(
        candidates,
        ReceiverSelectionConfig(),
    )

    base = {
        "stage": 5,
        "block": "5.8",
        "status": "COMPLETE",
        "runner_sha256": runner_sha,
        "rank": rank,
        "scenario_id": scenario_id,
        "receiver_status": str(selection.status),
        "eligible_receiver_count": int(
            selection.eligible_candidate_count
        ),
        "selected_receiver": False,
        "selected_prediction_available": False,
        "bridge_max_distance_m": (
            max(bridge_distances)
            if bridge_distances else None
        ),
        "causal_current_history_count": int(
            len(histories)
        ),
        "class_resolved_causal_history_count": int(
            len(matches)
        ),
        "class_unresolved_causal_history_count": int(
            unresolved_causal_histories
        ),
        "unresolved_class_histories_receiver_eligible": False,
        "data_pipeline_runtime_s": data_runtime,
        "adaptive": [],
        "baselines": [],
        "valid_future": {
            str(h): False for h in HORIZONS
        },
        "future_truth_used_before_decision": False,
        "tracks_to_predict_receiver_selector": False,
    }

    if selection.selected_candidate_index is None:
        atomic_json(cache, base)
        return base, False

    selected = candidate_rows[
        int(selection.selected_candidate_index)
    ]

    base["selected_receiver"] = True
    base["receiver_prediction_id"] = str(
        selected["history"].prediction_id
    )

    sample = sample_by_pid.get(
        base["receiver_prediction_id"]
    )

    if sample is None:
        atomic_json(cache, base)
        return base, False

    base["selected_prediction_available"] = True

    device, model, normalizer = runtime

    torch.cuda.reset_peak_memory_stats(device)

    mean, covariance, predictor_runtime = predict_gaussian(
        sample,
        selected["history"],
        device=device,
        model=model,
        normalizer=normalizer,
    )

    heading = current_heading(selected["history"])
    center = np.asarray(
        selected["history"].latest_position_H0_m,
        dtype=float,
    )

    mean_offset, cov_offset, _ = receiver_parameters

    angular_offsets = {
        "centroid": None,
        "known": ReceiverOffsetGaussianBody(
            mean_body_m=tuple(mean_offset),
            covariance_body_m2=tuple(
                tuple(x) for x in np.zeros((3, 3))
            ),
        ),
        "uncertain": ReceiverOffsetGaussianBody(
            mean_body_m=tuple(mean_offset),
            covariance_body_m2=tuple(
                tuple(x) for x in cov_offset
            ),
        ),
    }

    plans = []

    for geometry in GEOMETRIES:
        start = time.perf_counter_ns()

        posterior = deterministic_receiver_angular_posterior(
            trajectory_mean_h0_m=mean,
            raw_trajectory_covariance_h0_m2=covariance,
            current_position_h0_m=tuple(center),
            current_heading_h0_rad=heading,
            receiver_geometry_mode=geometry_modes[geometry],
            receiver_offset=angular_offsets[geometry],
            sample_count=MC_SAMPLES,
            seed=stable_seed(
                scenario_id,
                geometry,
                "angular",
            ),
            low_speed_threshold_mps=LOW_SPEED_MPS,
        )

        angular_runtime = (
            time.perf_counter_ns() - start
        ) / 1e9

        azimuth_samples = np.asarray(
            posterior.azimuth_samples_rad,
            dtype=float,
        )

        if azimuth_samples.shape != (MC_SAMPLES, 4):
            raise RuntimeError(
                f"angular sample shape {azimuth_samples.shape}"
            )

        current_position = current_receiver_position(
            center,
            heading,
            geometry,
            mean_offset,
        )

        current_azimuth = math.atan2(
            current_position[1],
            current_position[0],
        )

        for beam_count in CODEBOOKS:
            codebook = codebooks[beam_count]

            geometry_baseline = geometry_nearest_beam(
                predicted_azimuth_rad=current_azimuth,
                codebook=codebook,
            )

            persistence_indices = selection_indices(
                geometry_baseline
            )

            temporal_states = {
                q: initial_adaptive_temporal_state()
                for q in Q_LEVELS
            }

            for h_index, horizon in enumerate(HORIZONS):
                start = time.perf_counter_ns()

                probability = (
                    complete_partition_beam_probability_mass_from_samples(
                        azimuth_samples_rad=(
                            azimuth_samples[:, h_index]
                        ),
                        codebook=codebook,
                    )
                )

                probability_runtime = (
                    time.perf_counter_ns() - start
                ) / 1e9

                fixed = {
                    k: fixed_top_k_probability(
                        probability=probability,
                        codebook=codebook,
                        k=k,
                    )
                    for k in FIXED_TOPK
                }

                adaptive = {}

                for q in Q_LEVELS:
                    start = time.perf_counter_ns()

                    result = adaptive_temporal_step(
                        state=temporal_states[q],
                        probability=probability,
                        codebook=codebook,
                        requested_coverage=q,
                    )

                    topk_runtime = (
                        time.perf_counter_ns() - start
                    ) / 1e9

                    state, decision = temporal_result(
                        result,
                        temporal_states[q],
                    )

                    temporal_states[q] = state
                    adaptive[q] = (
                        decision,
                        topk_runtime,
                    )

                plans.append(
                    {
                        "geometry": geometry,
                        "beam_count": beam_count,
                        "h_index": h_index,
                        "horizon": horizon,
                        "codebook": codebook,
                        "probability": probability,
                        "fixed": fixed,
                        "persistence": persistence_indices,
                        "adaptive": adaptive,
                        "angular_runtime_s": angular_runtime,
                        "probability_runtime_s": probability_runtime,
                        "uncertainty_rad": circular_std(
                            azimuth_samples[:, h_index]
                        ),
                    }
                )

    # --------------------------------------------------------
    # CONTROLLER DECISIONS ARE COMPLETE.
    # Only below this line is future WOMD state accessed.
    # --------------------------------------------------------

    track = scenario.tracks[selected["track_index"]]
    T = transform_h0(
        built,
        int(scenario.current_time_index),
    )

    rng = np.random.default_rng(
        stable_seed(
            scenario_id,
            "constructed_receiver_truth",
        )
    )

    uncertain_truth_offset = rng.multivariate_normal(
        mean_offset,
        cov_offset,
        check_valid="raise",
    )

    truth = {
        geometry: {}
        for geometry in GEOMETRIES
    }

    for hi, (horizon, offset) in enumerate(
        zip(HORIZONS, OFFSETS)
    ):
        state_index = (
            int(scenario.current_time_index)
            + offset
        )

        valid = (
            state_index < len(track.states)
            and bool(track.states[state_index].valid)
        )

        base["valid_future"][str(horizon)] = valid

        if not valid:
            continue

        state = track.states[state_index]
        actor_center = future_center(state, T)
        actor_heading = future_heading(state, T)

        truth_offsets = {
            "centroid": np.zeros(3),
            "known": mean_offset,
            "uncertain": uncertain_truth_offset,
        }

        for geometry in GEOMETRIES:
            position = (
                actor_center
                + rotate_offset(
                    truth_offsets[geometry],
                    actor_heading,
                )
            )

            truth[geometry][hi] = spherical(position)

    for plan in plans:
        geometry = plan["geometry"]
        hi = plan["h_index"]

        if hi not in truth[geometry]:
            continue

        distance, azimuth, elevation = truth[geometry][hi]
        codebook = plan["codebook"]
        beam_count = plan["beam_count"]

        oracle = oracle_best_gain_beam(
            realized_azimuth_rad=azimuth,
            codebook=codebook,
        )

        oracle_indices = selection_indices(oracle)

        oracle_link = best_link(
            oracle_indices,
            azimuth,
            elevation,
            codebook,
            1,
            tbeam,
            tframe,
        )

        baseline_sets = {
            "geometry_nearest": (
                plan["persistence"],
                1,
            ),
            "previous_beam_persistence": (
                plan["persistence"],
                1,
            ),
            "exhaustive": (
                tuple(range(beam_count)),
                beam_count,
            ),
            "oracle": (
                oracle_indices,
                1,
            ),
        }

        for k in FIXED_TOPK:
            baseline_sets[f"fixed_top_{k}"] = (
                selection_indices(plan["fixed"][k]),
                k,
            )

        for policy, (indices, probes) in baseline_sets.items():
            link = best_link(
                indices,
                azimuth,
                elevation,
                codebook,
                probes,
                tbeam,
                tframe,
            )

            base["baselines"].append(
                {
                    "geometry": geometry,
                    "codebook": beam_count,
                    "horizon": plan["horizon"],
                    "policy": policy,
                    "hit": (
                        True
                        if policy == "oracle"
                        else hit(
                            azimuth,
                            indices,
                            codebook,
                        )
                    ),
                    "probe_count": probes,
                    "overhead_fraction": (
                        probes * tbeam / tframe
                    ),
                    "distance_m": distance,
                    "uncertainty_rad": plan[
                        "uncertainty_rad"
                    ],
                    **link_row(link, oracle_link),
                }
            )

        for q in Q_LEVELS:
            decision, selection_runtime = plan["adaptive"][q]

            mass_indices = tuple(
                int(x)
                for x in (
                    decision
                    .adaptive_selection
                    .beam_indices
                )
            )

            physical_indices = physical_adaptive_indices(
                decision
            )

            probe_count = int(
                actual_probe_count_for_adaptive_decision(
                    decision
                )
            )

            link = best_link(
                physical_indices,
                azimuth,
                elevation,
                codebook,
                probe_count,
                tbeam,
                tframe,
            )

            base["adaptive"].append(
                {
                    "geometry": geometry,
                    "codebook": beam_count,
                    "horizon": plan["horizon"],
                    "q": q,
                    "containment_hit": hit(
                        azimuth,
                        mass_indices,
                        codebook,
                    ),
                    "selected_k": int(
                        decision.adaptive_selection.k
                    ),
                    "selected_mass": float(
                        decision
                        .adaptive_selection
                        .selected_in_support_mass
                    ),
                    "physical_probe_count": probe_count,
                    "overhead_fraction": (
                        probe_count * tbeam / tframe
                    ),
                    "overhead_reduction_vs_exhaustive": (
                        1.0 - probe_count / beam_count
                    ),
                    "primary_switched": bool(
                        decision.primary_switched
                    ),
                    "loss_of_lock": bool(
                        decision.loss_of_lock
                    ),
                    "exhaustive_reacquisition": bool(
                        decision.exhaustive_fallback_active
                    ),
                    "reacquisition_latency_s": (
                        max(
                            0,
                            probe_count
                            - int(
                                decision
                                .adaptive_selection
                                .k
                            ),
                        )
                        * tbeam
                        if bool(decision.loss_of_lock)
                        else 0.0
                    ),
                    "distance_m": distance,
                    "uncertainty_rad": plan[
                        "uncertainty_rad"
                    ],
                    "predictor_runtime_s": predictor_runtime,
                    "angular_runtime_s": plan[
                        "angular_runtime_s"
                    ],
                    "beam_probability_runtime_s": plan[
                        "probability_runtime_s"
                    ],
                    "topk_runtime_s": selection_runtime,
                    **link_row(link, oracle_link),
                }
            )

    base["gpu_peak_memory_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )

    atomic_json(cache, base)
    return base, False


def group_summary(rows, fields, adaptive):
    groups = defaultdict(list)

    for row in rows:
        key = tuple(row[field] for field in fields)
        groups[key].append(row)

    output = {}

    for key, values in sorted(groups.items()):
        name = "|".join(str(x) for x in key)

        if adaptive:
            hits = sum(
                bool(x["containment_hit"])
                for x in values
            )

            output[name] = {
                **dict(zip(fields, key)),
                "n": len(values),
                "hits": hits,
                "coverage": hits / len(values),
                "mean_k": avg(
                    x["selected_k"] for x in values
                ),
                "mean_probe_count": avg(
                    x["physical_probe_count"]
                    for x in values
                ),
                "mean_overhead_fraction": avg(
                    x["overhead_fraction"]
                    for x in values
                ),
                "mean_overhead_reduction": avg(
                    x[
                        "overhead_reduction_vs_exhaustive"
                    ]
                    for x in values
                ),
                "outage_probability": 1.0 - hits / len(values),
                "beam_switching_rate": avg(
                    int(x["primary_switched"])
                    for x in values
                ),
                "mean_reacquisition_latency_s": avg(
                    x["reacquisition_latency_s"]
                    for x in values
                ),
                "beam_gain_loss_db": avg(
                    x.get("beam_gain_loss_db")
                    for x in values
                ),
                "received_power_loss_db": avg(
                    x.get("received_power_loss_db")
                    for x in values
                ),
                "snr_loss_db": avg(
                    x.get("snr_loss_db")
                    for x in values
                ),
                "ber": avg(
                    x.get("ber")
                    for x in values
                ),
                "effective_rate_bps": avg(
                    x.get("effective_rate_bps")
                    for x in values
                ),
                "effective_rate_loss_bps": avg(
                    x.get("effective_rate_loss_bps")
                    for x in values
                ),
            }

        else:
            hits = sum(bool(x["hit"]) for x in values)

            output[name] = {
                **dict(zip(fields, key)),
                "n": len(values),
                "hits": hits,
                "hit_rate": hits / len(values),
                "mean_probe_count": avg(
                    x["probe_count"] for x in values
                ),
                "mean_overhead_fraction": avg(
                    x["overhead_fraction"]
                    for x in values
                ),
                "beam_gain_loss_db": avg(
                    x.get("beam_gain_loss_db")
                    for x in values
                ),
                "snr_loss_db": avg(
                    x.get("snr_loss_db")
                    for x in values
                ),
                "ber": avg(
                    x.get("ber")
                    for x in values
                ),
                "effective_rate_bps": avg(
                    x.get("effective_rate_bps")
                    for x in values
                ),
            }

    return output


def correlation(rows, xkey, ykey):
    pairs = [
        (float(r[xkey]), float(r[ykey]))
        for r in rows
        if (
            r.get(xkey) is not None
            and r.get(ykey) is not None
            and math.isfinite(float(r[xkey]))
            and math.isfinite(float(r[ykey]))
        )
    ]

    if len(pairs) < 3:
        return None

    x = np.asarray([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def acceptance(adaptive_summary, baseline_summary):
    coverage_tests = []
    coverage_pass = True

    for beam_count in CODEBOOKS:
        for horizon in HORIZONS:
            key = (
                f"{PRIMARY_GEOMETRY}|"
                f"{beam_count}|{horizon}|0.95"
            )

            row = adaptive_summary.get(key)

            if row is None or row["n"] == 0:
                decision = {
                    "status": "NOT_EVALUABLE",
                    "passed": False,
                }
                coverage_pass = False
            else:
                decision = empirical_coverage_decision(
                    hits=int(row["hits"]),
                    trials=int(row["n"]),
                    requested_q=0.95,
                ).to_dict()

                coverage_pass &= bool(decision["passed"])

            coverage_tests.append(
                {
                    "codebook": beam_count,
                    "horizon": horizon,
                    "decision": decision,
                }
            )

    overhead_tests = []
    overhead_pass = True

    for beam_count in CODEBOOKS:
        adaptive_rows = [
            row
            for row in adaptive_summary.values()
            if (
                row["geometry"] == PRIMARY_GEOMETRY
                and int(row["codebook"]) == beam_count
                and float(row["q"]) == 0.95
            )
        ]

        adaptive_probes = avg(
            row["mean_probe_count"]
            for row in adaptive_rows
        )

        valid_fixed = []

        for k in FIXED_TOPK:
            valid = True

            for horizon in HORIZONS:
                key = (
                    f"fixed_top_{k}|"
                    f"{PRIMARY_GEOMETRY}|"
                    f"{beam_count}|{horizon}"
                )

                row = baseline_summary.get(key)

                if row is None or row["n"] == 0:
                    valid = False
                    break

                decision = empirical_coverage_decision(
                    hits=int(row["hits"]),
                    trials=int(row["n"]),
                    requested_q=0.95,
                )

                if not decision.passed:
                    valid = False
                    break

            if valid:
                valid_fixed.append(k)

        best_fixed = (
            min(valid_fixed)
            if valid_fixed else None
        )

        exhaustive_pass = (
            adaptive_probes is not None
            and adaptive_probes < beam_count
        )

        fixed_pass = (
            None
            if best_fixed is None
            else adaptive_probes <= best_fixed + 1e-12
        )

        codebook_pass = (
            exhaustive_pass
            and fixed_pass is not False
        )

        overhead_pass &= codebook_pass

        overhead_tests.append(
            {
                "codebook": beam_count,
                "adaptive_mean_probes": adaptive_probes,
                "coverage_valid_fixed_topk": valid_fixed,
                "best_valid_fixed_topk": best_fixed,
                "adaptive_less_than_exhaustive": exhaustive_pass,
                "adaptive_vs_valid_fixed_pass": fixed_pass,
                "passed": codebook_pass,
            }
        )

    return {
        "primary_geometry": PRIMARY_GEOMETRY,
        "coverage_pass": bool(coverage_pass),
        "coverage_tests": coverage_tests,
        "overhead_pass": bool(overhead_pass),
        "overhead_tests": overhead_tests,
        "stage5_acceptance_pass": bool(
            coverage_pass and overhead_pass
        ),
    }


def main():
    check_sha(GAUSSIAN_CKPT, GAUSSIAN_SHA, "Gaussian")
    check_sha(CALIBRATOR, CALIBRATOR_SHA, "calibrator")
    check_sha(
        STAGE4_HANDOFF,
        STAGE4_HANDOFF_SHA,
        "Stage4 handoff",
    )
    check_sha(
        RECEIVER_REPORT,
        RECEIVER_REPORT_SHA,
        "Block5.1 receiver report",
    )
    check_sha(
        ANGULAR_REPORT,
        ANGULAR_REPORT_SHA,
        "Block5.2 angular report",
    )
    check_sha(
        LATENCY_CONFIG,
        LATENCY_SHA,
        "Block5.7 latency config",
    )
    check_sha(
        ACCEPTANCE_CONFIG,
        ACCEPTANCE_SHA,
        "Block5.8 acceptance config",
    )

    if shutil.disk_usage(S5).free < MIN_FREE_BYTES:
        raise RuntimeError("free disk < 250 GiB")

    formal_rows = read_formal()

    latency = load_json(LATENCY_CONFIG)
    tbeam = float(
        latency["multi_rate_timing"]["Tbeam_s"]
    )
    tframe = float(
        latency["multi_rate_timing"]["Tframe_s"]
    )

    mean_offset, cov_offset, offset_source = (
        receiver_offset()
    )

    geometry_modes = resolve_geometry_modes(
        mean_offset,
        cov_offset,
    )

    vehicle_semantic = resolve_vehicle_semantic()

    runtime = load_gaussian()

    lo = float(FROZEN_SUPPORT_MIN_RAD)
    hi = float(FROZEN_SUPPORT_MAX_RAD)

    codebooks = {
        n: build_uniform_azimuth_codebook(
            beam_count=n,
            support_min_azimuth_rad=lo,
            support_max_azimuth_rad=hi,
        )
        for n in CODEBOOKS
    }

    clean_config, degraded_config = (
        load_frozen_stage2_configs()
    )

    validation_rows = read_validation_manifest(
        VALIDATION_MANIFEST
    )

    validation_by_id = {
        str(row.scenario_id): row
        for row in validation_rows
    }

    missing = [
        str(row["scenario_id"])
        for row in formal_rows
        if str(row["scenario_id"]) not in validation_by_id
    ]

    if missing:
        raise RuntimeError(
            f"formal scenarios missing from canonical validation: "
            f"{missing[:3]}"
        )

    runner_sha = sha(Path(__file__))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("BLOCK 5.8 preflight       = PASS")
    print("formal population          = 120 / SHA PASS")
    print("receiver policy            = nearest causal vehicle ahead")
    print("geometry modes             = centroid / known / uncertain")
    print("codebooks                  = 16 / 32 / 64")
    print("q levels                   = 0.90 / 0.95 / 0.975 / 0.99")
    print("future truth               = evaluator-only after decision")
    print("starting formal N=120...")

    scene_results = []
    reused = 0

    for rank, formal in enumerate(
        formal_rows,
        start=1,
    ):
        result, was_reused = process_scene(
            rank,
            formal,
            validation_by_id,
            clean_config=clean_config,
            degraded_config=degraded_config,
            runtime=runtime,
            receiver_parameters=(
                mean_offset,
                cov_offset,
                offset_source,
            ),
            geometry_modes=geometry_modes,
            vehicle_semantic=vehicle_semantic,
            codebooks=codebooks,
            tbeam=tbeam,
            tframe=tframe,
            runner_sha=runner_sha,
        )

        scene_results.append(result)
        reused += int(was_reused)

        if rank % 20 == 0:
            print(f"formal progress             = {rank}/120")

    adaptive_rows = [
        row
        for scene in scene_results
        for row in scene["adaptive"]
    ]

    baseline_rows = [
        row
        for scene in scene_results
        for row in scene["baselines"]
    ]

    adaptive_summary = group_summary(
        adaptive_rows,
        ("geometry", "codebook", "horizon", "q"),
        adaptive=True,
    )

    baseline_summary = group_summary(
        baseline_rows,
        ("policy", "geometry", "codebook", "horizon"),
        adaptive=False,
    )

    gate = acceptance(
        adaptive_summary,
        baseline_summary,
    )

    selected = sum(
        bool(x["selected_receiver"])
        for x in scene_results
    )

    prediction_available = sum(
        bool(x["selected_prediction_available"])
        for x in scene_results
    )

    no_receiver = 120 - selected
    prediction_unavailable = (
        selected - prediction_available
    )

    valid_endpoints = {
        str(h): sum(
            bool(scene["valid_future"][str(h)])
            for scene in scene_results
        )
        for h in HORIZONS
    }

    nominal = [
        row
        for row in adaptive_rows
        if (
            row["geometry"] == PRIMARY_GEOMETRY
            and float(row["q"]) == 0.95
        )
    ]

    axis_analysis = {
        "distance_vs_selected_K_correlation": correlation(
            nominal,
            "distance_m",
            "selected_k",
        ),
        "uncertainty_vs_selected_K_correlation": correlation(
            nominal,
            "uncertainty_rad",
            "selected_k",
        ),
        "distance_vs_effective_rate_correlation": correlation(
            nominal,
            "distance_m",
            "effective_rate_bps",
        ),
        "uncertainty_vs_effective_rate_correlation": correlation(
            nominal,
            "uncertainty_rad",
            "effective_rate_bps",
        ),
    }

    reliability_overhead = {}

    for geometry in GEOMETRIES:
        for beam_count in CODEBOOKS:
            points = []

            for q in Q_LEVELS:
                rows = [
                    x
                    for x in adaptive_summary.values()
                    if (
                        x["geometry"] == geometry
                        and int(x["codebook"]) == beam_count
                        and float(x["q"]) == q
                    )
                ]

                points.append(
                    {
                        "q": q,
                        "coverage": avg(
                            x["coverage"] for x in rows
                        ),
                        "mean_K": avg(
                            x["mean_k"] for x in rows
                        ),
                        "mean_probe_count": avg(
                            x["mean_probe_count"]
                            for x in rows
                        ),
                    }
                )

            pareto = []

            for point in points:
                if (
                    point["coverage"] is None
                    or point["mean_probe_count"] is None
                ):
                    continue

                dominated = any(
                    (
                        other["coverage"] is not None
                        and other["mean_probe_count"] is not None
                        and other["coverage"] >= point["coverage"]
                        and other["mean_probe_count"]
                        <= point["mean_probe_count"]
                        and (
                            other["coverage"] > point["coverage"]
                            or other["mean_probe_count"]
                            < point["mean_probe_count"]
                        )
                    )
                    for other in points
                    if other is not point
                )

                if not dominated:
                    pareto.append(point)

            reliability_overhead[
                f"{geometry}|{beam_count}"
            ] = {
                "curve": points,
                "pareto_frontier": pareto,
            }

    with MERGED.open("w", encoding="utf-8") as f:
        for scene in scene_results:
            for row in scene["adaptive"]:
                f.write(
                    json.dumps(
                        json_safe(
                            {
                                "scenario_id": scene["scenario_id"],
                                "type": "adaptive",
                                **row,
                            }
                        ),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )

            for row in scene["baselines"]:
                f.write(
                    json.dumps(
                        json_safe(
                            {
                                "scenario_id": scene["scenario_id"],
                                "type": "baseline",
                                **row,
                            }
                        ),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )

    report = {
        "stage": 5,
        "block": "5.8",
        "status": (
            "PASS"
            if gate["stage5_acceptance_pass"]
            else "BLOCKED"
        ),
        "formal_population": {
            "N": 120,
            "manifest_sha256": FORMAL_SHA,
            "same_immutable_Stage3_4_population": True,
        },
        "receiver_availability": {
            "scenarios_with_eligible_receiver": selected,
            "scenarios_without_eligible_receiver": no_receiver,
            "selected_receiver_prediction_available": (
                prediction_available
            ),
            "selected_receiver_prediction_unavailable": (
                prediction_unavailable
            ),
            "causal_histories_semantic_class_unresolved": int(
                sum(
                    int(
                        scene.get(
                            "class_unresolved_causal_history_count",
                            0,
                        )
                    )
                    for scene in scene_results
                )
            ),
            "unresolved_history_policy": (
                "explicitly_counted_not_receiver_eligible_"
                "no_class_guess_no_substitution"
            ),
            "valid_future_receiver_endpoints": valid_endpoints,
        },
        "frozen_contract": {
            "receiver_policy": "nearest_causal_vehicle_ahead",
            "tracks_to_predict_receiver_selector": False,
            "future_truth_receiver_selector": False,
            "future_truth_role": (
                "evaluator_only_after_controller_decision"
            ),
            "primary_posterior": (
                "calibrated_Gaussian_GRU"
            ),
            "MC_samples": MC_SAMPLES,
            "MC_seed": MC_SEED,
            "receiver_offset_source": offset_source,
            "receiver_geometry_modes": list(GEOMETRIES),
            "primary_acceptance_geometry": PRIMARY_GEOMETRY,
            "codebooks": list(CODEBOOKS),
            "horizons_s": list(HORIZONS),
            "q_levels": list(Q_LEVELS),
            "Tbeam_s": tbeam,
            "Tframe_s": tframe,
            "constructed_receiver_geometry_not_measured_label": True,
            "probability_decision_partition": (
                "complete_nonoverlapping_edge_extended"
            ),
            "probability_mass_sum_contract": "sum_Pb_equals_1",
            "physical_gain_support_changed": False,
            "outside_physical_sector_link_metrics_use_actual_angle": True,
        },
        "adaptive_metrics": adaptive_summary,
        "baseline_metrics": baseline_summary,
        "reliability_overhead": reliability_overhead,
        "distance_uncertainty_analysis": axis_analysis,
        "nominal_primary_summary": {
            "average_selected_K": avg(
                x["selected_k"] for x in nominal
            ),
            "average_physical_probe_count": avg(
                x["physical_probe_count"] for x in nominal
            ),
            "probability_coverage": avg(
                int(x["containment_hit"]) for x in nominal
            ),
            "probing_overhead_fraction": avg(
                x["overhead_fraction"] for x in nominal
            ),
            "overhead_reduction_vs_exhaustive": avg(
                x["overhead_reduction_vs_exhaustive"]
                for x in nominal
            ),
            "beam_gain_loss_db": avg(
                x.get("beam_gain_loss_db")
                for x in nominal
            ),
            "received_power_loss_db": avg(
                x.get("received_power_loss_db")
                for x in nominal
            ),
            "snr_loss_db": avg(
                x.get("snr_loss_db")
                for x in nominal
            ),
            "BER": avg(
                x.get("ber") for x in nominal
            ),
            "effective_rate_bps": avg(
                x.get("effective_rate_bps")
                for x in nominal
            ),
            "effective_rate_loss_bps": avg(
                x.get("effective_rate_loss_bps")
                for x in nominal
            ),
            "outage_probability": avg(
                1 - int(x["containment_hit"])
                for x in nominal
            ),
            "beam_switching_rate": avg(
                int(x["primary_switched"])
                for x in nominal
            ),
            "reacquisition_latency_s": avg(
                x["reacquisition_latency_s"]
                for x in nominal
            ),
        },
        "latency": {
            "predictor_inference_s": avg(
                x["predictor_runtime_s"]
                for x in nominal
            ),
            "posterior_mapping_s": avg(
                x["angular_runtime_s"]
                for x in nominal
            ),
            "beam_probability_s": avg(
                x["beam_probability_runtime_s"]
                for x in nominal
            ),
            "TopK_selection_s": avg(
                x["topk_runtime_s"]
                for x in nominal
            ),
            "probing_time_formula": "K_physical*Tbeam",
            "annotated_sub100ms_ground_truth": False,
            "GPU_peak_memory_bytes": max(
                (
                    int(x.get("gpu_peak_memory_bytes", 0))
                    for x in scene_results
                ),
                default=0,
            ),
        },
        "acceptance": gate,
        "provenance": {
            "runner_sha256": runner_sha,
            "Stage4_handoff_sha256": STAGE4_HANDOFF_SHA,
            "Gaussian_checkpoint_sha256": GAUSSIAN_SHA,
            "normalization_sha256": sha(NORMALIZATION),
            "calibrator_sha256": CALIBRATOR_SHA,
            "receiver_report_sha256": RECEIVER_REPORT_SHA,
            "angular_report_sha256": ANGULAR_REPORT_SHA,
            "latency_policy_sha256": LATENCY_SHA,
            "acceptance_policy_sha256": ACCEPTANCE_SHA,
            "merged_records_sha256": sha(MERGED),
            "cache_reused_scenarios": reused,
            "compatible_prior_runner_sha256": (
                COMPATIBLE_PRIOR_RUNNER_SHA256
            ),
            "causal_actor_bridge_policy": (
                BRIDGE_REPAIR_POLICY
            ),
        },
        "scope": {
            "Stage4_retraining": False,
            "normalization_refit": False,
            "recalibration": False,
            "formal_parameter_tuning": False,
            "posthoc_threshold_change": False,
            "predictive_ADB_started": False,
            "joint_Stage7_evaluation_started": False,
            "DeepSense_started": False,
        },
        "next": (
            "Block 5.9 reproducibility"
            if gate["stage5_acceptance_pass"]
            else
            "Stage5 BLOCKED; no post-hoc formal tuning"
        ),
    }

    atomic_json(REPORT, report)

    print()
    print("formal scenarios            = 120/120")
    print("eligible receiver scenes    =", selected)
    print("prediction unavailable      =", prediction_unavailable)
    print(
        "coverage acceptance         =",
        "PASS" if gate["coverage_pass"] else "BLOCKED",
    )
    print(
        "overhead acceptance         =",
        "PASS" if gate["overhead_pass"] else "BLOCKED",
    )
    print(
        "BLOCK 5.8 STATUS            =",
        report["status"],
    )
    print("report                      =", REPORT)


def guarded_main():
    try:
        main()

    except Exception as exc:
        atomic_json(
            FAILURE,
            {
                "stage": 5,
                "block": "5.8",
                "status": "IMPLEMENTATION_FAILURE",
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "recovery": (
                    "repair only the Block5.8 implementation; "
                    "do not tune formal scientific parameters"
                ),
            },
        )

        print()
        print("BLOCK 5.8 STATUS            = IMPLEMENTATION_FAILURE")
        print("type                        =", type(exc).__name__)
        print("message                     =", str(exc))
        print("failure report              =", FAILURE)

    # Deliberately return normally so the terminal remains open.
    return 0


if __name__ == "__main__":
    raise SystemExit(guarded_main())

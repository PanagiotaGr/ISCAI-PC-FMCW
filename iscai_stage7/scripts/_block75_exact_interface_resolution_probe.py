from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"

FILES = [
    S4 / "src/iscai_stage4/data/__init__.py",
    S4 / "src/iscai_stage4/data/neural_inputs.py",
    S4 / "src/iscai_stage4/data/real_pipeline.py",
    S4 / "src/iscai_stage4/ml/calibration_runtime.py",
    S4 / "src/iscai_stage4/ml/normalization.py",

    S5 / "scripts/run_block58_part2_formal_evaluation.py",
    S5 / "src/iscai_stage5/receiver_selection.py",
    S5 / "src/iscai_stage5/receiver_policy.py",
    S5 / "src/iscai_stage5/adaptive_topk.py",
    S5 / "src/iscai_stage5/codebook_probability.py",
    S5 / "src/iscai_stage5/angular_monte_carlo.py",
    S5 / "src/iscai_stage5/optical_link.py",
    S5 / "src/iscai_stage5/beam_latency.py",
    S5 / "src/iscai_stage5/beam_baselines.py",

    S6 / "scripts/run_block63_part2b1_frozen_gaussian_forward.py",
    S6 / "src/iscai_stage6/adb/geometry.py",
    S6 / "src/iscai_stage6/adb/probabilistic_full_box.py",
    S6 / "src/iscai_stage6/adb/probabilistic_occupancy.py",
    S6 / "src/iscai_stage6/adb/deterministic_predictive.py",
    S6 / "src/iscai_stage6/adb/class_aware_policy.py",
    S6 / "src/iscai_stage6/adb/predictive_mask.py",
    S6 / "src/iscai_stage6/adb/part_a_reactive.py",
    S6 / "src/iscai_stage6/adb/metric_semantics.py",

    S7 / "scripts/run_block74h_d_direct_baseline_training.py",
]

KEYWORD_GROUPS = {
    "S4_CAUSAL_SAMPLE_CONSTRUCTION": (
        "build_real_causal_inputs",
        "build_causal_track_history",
        "map_context_summary",
        "neighbor",
        "target",
        "feature",
        "measurement_covariance",
        "latest_position",
        "prediction_id",
    ),
    "S4_MODEL_PREPARE_INFERENCE": (
        "CalibrationNormalizer",
        "prepare",
        "infer_scene_statistics",
        "GaussianTrajectoryGRU",
        "future_mask",
    ),
    "S6_FROZEN_GAUSSIAN_BLUEPRINT": (
        "build_real_causal_inputs",
        "build_causal_track_history",
        "map_context_summary",
        "attach_supervision",
        "raw_covariance",
        "calibrated_covariance",
        "predictive_mean",
        "CalibrationNormalizer",
        "GaussianTrajectoryGRU",
    ),
    "S5_CAUSAL_RECEIVER_AND_BEAM": (
        "receiver",
        "candidate",
        "select",
        "adaptive_topk_probability_mass",
        "deterministic_receiver_angular_posterior",
        "codebook",
        "probability",
        "future_truth_used_before_decision",
    ),
    "S5_EXACT_LINK_METRICS": (
        "optical_gain",
        "received_power",
        "snr",
        "ber",
        "effective_rate",
        "probing",
        "outage",
        "link_row",
    ),
    "S6_CAUSAL_ADB_GEOMETRY_CONTROLLER": (
        "CausalADBActorBox",
        "reactive",
        "current",
        "build_stochastic_future_full_boxes",
        "estimate_actor_occupancy_probability",
        "build_deterministic_future_full_boxes",
        "compose_class_aware_illumination",
        "temporal",
        "rate_limit",
    ),
    "S6_EVALUATOR_ONLY_REFERENCES": (
        "oracle_all",
        "oracle_vehicle",
        "pedestrian_region",
        "cyclist_region",
        "vehicle_surrogate",
        "future_full_box",
        "evaluator",
    ),
    "S7_DIRECT_MODEL_INTERFACE": (
        "DirectBeam",
        "DirectADB",
        "forward",
        "encoder",
        "checkpoint",
    ),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparse-failed>"

def signature_text(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    pieces = []

    pos = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)

    posonly_n = len(args.posonlyargs)
    for i, (a, d) in enumerate(zip(pos, defaults)):
        s = a.arg
        if a.annotation is not None:
            s += ": " + safe_unparse(a.annotation)
        if d is not None:
            s += " = " + safe_unparse(d)
        pieces.append(s)
        if posonly_n and i + 1 == posonly_n:
            pieces.append("/")

    if args.vararg is not None:
        s = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            s += ": " + safe_unparse(args.vararg.annotation)
        pieces.append(s)
    elif args.kwonlyargs:
        pieces.append("*")

    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        s = a.arg
        if a.annotation is not None:
            s += ": " + safe_unparse(a.annotation)
        if d is not None:
            s += " = " + safe_unparse(d)
        pieces.append(s)

    if args.kwarg is not None:
        s = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            s += ": " + safe_unparse(args.kwarg.annotation)
        pieces.append(s)

    ret = ""
    if node.returns is not None:
        ret = " -> " + safe_unparse(node.returns)

    return f"{node.name}({', '.join(pieces)}){ret}"

def node_source(lines: list[str], node: ast.AST, pad: int = 2) -> str:
    start = max(1, getattr(node, "lineno", 1) - pad)
    end = min(len(lines), getattr(node, "end_lineno", getattr(node, "lineno", 1)) + pad)
    return "\n".join(
        f"{i:05d}: {lines[i-1]}"
        for i in range(start, end + 1)
    )

def token_score(text: str, tokens: Iterable[str]) -> tuple[int, list[str]]:
    low = text.lower()
    hits = [t for t in tokens if t.lower() in low]
    return len(hits), hits

def top_nodes(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(path))
    nodes = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            src = ast.get_source_segment(text, n) or ""
            nodes.append((n, src))
    return text, lines, tree, nodes

def print_header(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

print_header("BLOCK 7.5 EXACT INTERFACE RESOLUTION PROBE")
print("READ ONLY")
print("Formal manifest/data opened          = NO")
print("Formal protobuf scenario opened      = NO")
print("Model checkpoint loaded              = NO")
print("Model inference                      = NO")
print("Future GT read                       = NO")
print("Scientific artifact modified         = NO")
print("Only this temporary probe source and audit log are written.")

missing = [str(p) for p in FILES if not p.is_file()]
if missing:
    print_header("FAIL_CLOSED — REQUIRED SOURCE FILES MISSING")
    for p in missing:
        print(p)
    raise SystemExit(2)

print_header("A. SOURCE SHA INVENTORY")
for p in FILES:
    print(f"{sha256(p)}  {p}")

print_header("B. TOP-LEVEL FUNCTION / CLASS REGISTRY")
for p in FILES:
    print()
    print("-" * 100)
    print(p)
    print("-" * 100)
    text, lines, tree, nodes = top_nodes(p)
    for n, src in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"{getattr(n,'lineno',0):05d}  FUNCTION  {signature_text(n)}")
        else:
            bases = ", ".join(safe_unparse(b) for b in n.bases)
            print(f"{getattr(n,'lineno',0):05d}  CLASS     {n.name}({bases})")

print_header("C. RELEVANT API CANDIDATES WITH EXACT SOURCE")
for group, tokens in KEYWORD_GROUPS.items():
    print()
    print("#" * 100)
    print(group)
    print("TOKENS =", json.dumps(tokens))
    print("#" * 100)
    ranked = []
    for p in FILES:
        text, lines, tree, nodes = top_nodes(p)
        for n, src in nodes:
            score, hits = token_score(src, tokens)
            if score:
                # Weight exact core tokens more than generic words.
                core_bonus = sum(
                    3 for t in hits
                    if t in {
                        "build_real_causal_inputs",
                        "build_causal_track_history",
                        "map_context_summary",
                        "CalibrationNormalizer",
                        "infer_scene_statistics",
                        "adaptive_topk_probability_mass",
                        "deterministic_receiver_angular_posterior",
                        "compose_class_aware_illumination",
                        "build_stochastic_future_full_boxes",
                        "estimate_actor_occupancy_probability",
                        "build_deterministic_future_full_boxes",
                        "CausalADBActorBox",
                        "oracle_all",
                    }
                )
                ranked.append((score + core_bonus, score, p, n, hits, src))
    ranked.sort(key=lambda x: (-x[0], str(x[2]), getattr(x[3], "lineno", 0)))
    for total, score, p, n, hits, src in ranked[:20]:
        print()
        print(f"FILE = {p}")
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print("KIND = function")
            print("SIGNATURE =", signature_text(n))
        else:
            print("KIND = class")
            print("NAME =", n.name)
        print("LINES =", getattr(n, "lineno", None), "-", getattr(n, "end_lineno", None))
        print("HITS =", hits)
        print("SOURCE:")
        print(node_source(p.read_text(encoding='utf-8').splitlines(), n, pad=1))

print_header("D. STAGE4 neural_inputs.py — FULL DEF/CLASS BODIES NEAR CAUSAL MODEL INPUT")
p = S4 / "src/iscai_stage4/data/neural_inputs.py"
text, lines, tree, nodes = top_nodes(p)
for n, src in nodes:
    low = src.lower()
    if any(k in low for k in (
        "measurement_covariance_h0_m2",
        "latest_position_h0_m",
        "map_context",
        "neighbor",
        "feature_vector",
        "build_causal_track_history",
    )):
        print()
        print(f"{type(n).__name__} {getattr(n,'name','?')} lines {n.lineno}-{n.end_lineno}")
        print(node_source(lines, n, pad=0))

print_header("E. STAGE6 FROZEN GAUSSIAN BLUEPRINT — CRITICAL LINE WINDOWS")
p = S6 / "scripts/run_block63_part2b1_frozen_gaussian_forward.py"
lines = p.read_text(encoding="utf-8").splitlines()
for start, end in (
    (720, 1325),
    (1710, 2075),
    (2250, 2595),
    (2820, 3225),
    (3520, 3785),
):
    print()
    print(f"----- {p.name} lines {start}-{end} -----")
    for i in range(start, min(end, len(lines)) + 1):
        print(f"{i:05d}: {lines[i-1]}")

print_header("F. STAGE5 FORMAL EXECUTOR — CAUSAL DECISION / LINK WINDOWS")
p = S5 / "scripts/run_block58_part2_formal_evaluation.py"
lines = p.read_text(encoding="utf-8").splitlines()
# Print function bodies whose names/source show receiver/controller/link logic.
text, lines2, tree, nodes = top_nodes(p)
for n, src in nodes:
    low_name = getattr(n, "name", "").lower()
    low = src.lower()
    if (
        any(k in low_name for k in (
            "receiver", "candidate", "link", "beam", "controller", "posterior",
        ))
        or (
            "future_truth_access_after_controller_decisions" in low
            or "future_label_read_by_controller" in low
            or "tracks_to_predict_receiver_selection" in low
        )
    ):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print()
            print("SIGNATURE =", signature_text(n))
            print("LINES =", n.lineno, "-", n.end_lineno)
            print(node_source(lines2, n, pad=0))

print_header("G. STAGE6 EXACT ADB / EVALUATOR CANDIDATE FUNCTIONS")
for p in [
    S6 / "src/iscai_stage6/adb/geometry.py",
    S6 / "src/iscai_stage6/adb/probabilistic_full_box.py",
    S6 / "src/iscai_stage6/adb/probabilistic_occupancy.py",
    S6 / "src/iscai_stage6/adb/deterministic_predictive.py",
    S6 / "src/iscai_stage6/adb/class_aware_policy.py",
    S6 / "src/iscai_stage6/adb/predictive_mask.py",
    S6 / "src/iscai_stage6/adb/part_a_reactive.py",
]:
    text, lines, tree, nodes = top_nodes(p)
    for n, src in nodes:
        low = src.lower()
        if any(k in low for k in (
            "causaladbactorbox",
            "build_stochastic_future_full_boxes",
            "estimate_actor_occupancy_probability",
            "build_deterministic_future_full_boxes",
            "compose_class_aware_illumination",
            "temporal_smooth",
            "rate_limit",
            "reactive_adb_map",
        )):
            print()
            print(f"FILE = {p}")
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                print("SIGNATURE =", signature_text(n))
            else:
                print("CLASS =", n.name)
            print("LINES =", n.lineno, "-", n.end_lineno)
            print(node_source(lines, n, pad=0))

print_header("H. STATIC CALL GRAPH CLUES — EXACT CROSS-MODULE NAMES")
targets = {
    "build_real_causal_inputs",
    "build_causal_track_history",
    "map_context_summary",
    "attach_supervision",
    "adaptive_topk_probability_mass",
    "deterministic_receiver_angular_posterior",
    "optical_gain_for_cell",
    "normalized_received_power",
    "effective_rate",
    "build_latency_budget",
    "build_stochastic_future_full_boxes",
    "estimate_actor_occupancy_probability",
    "build_deterministic_future_full_boxes",
    "compose_class_aware_illumination",
}
for p in FILES:
    text = p.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(p))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name in targets:
                hits.append((getattr(node, "lineno", None), name))
    if hits:
        print()
        print(p)
        for lineno, name in sorted(hits):
            print(f"  line {lineno:05d}: CALL {name}")

print_header("I. SAFETY / SCOPE ASSERTIONS")
forbidden_data_paths = [
    ROOT / "iscai_stage3/artifacts/block38e/formal_validation_120.jsonl",
    ROOT / "iscai_stage4/artifacts/block48/formal_cache_manifest.json",
    ROOT / "iscai_stage4/artifacts/block48/formal_neural_outputs.npz",
]
print("Formal data paths intentionally NOT opened:")
for p in forbidden_data_paths:
    print(" ", p)
print("No TFRecord opened                         = YES")
print("No protobuf scenario parsed                = YES")
print("No checkpoint loaded                       = YES")
print("No model inference                         = YES")
print("No future GT read                          = YES")
print("No Stage4/5/6 scientific file modified     = YES")
print("No Stage7 scientific artifact modified     = YES")
print("No threshold/policy/model selection change = YES")
print("Probe status                               = PASS_READ_ONLY_INTERFACE_DISCOVERY")

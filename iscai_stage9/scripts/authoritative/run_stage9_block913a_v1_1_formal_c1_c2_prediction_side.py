#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

for candidate in (
    S9 / "src",
    ROOT / "iscai_stage5/src",
    ROOT / "iscai_stage4/src",
    ROOT / "iscai_stage3/src",
    ROOT / "iscai_stage2/src",
    ROOT / "iscai_stage1/src",
    ROOT / "iscai_stage0/src",
    ROOT / "iscai_stage5",
    ROOT / "iscai_stage4",
    ROOT / "iscai_stage3",
    ROOT / "iscai_stage2",
    ROOT / "iscai_stage1",
    ROOT / "iscai_stage0",
):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

SCRIPT = S9 / "scripts/run_stage9_block913a_v1_1_formal_c1_c2_prediction_side.py"
CONTRACT = S9 / "configs/stage9_block913a_v1_1_formal_c1_c2_prediction_side_contract.json"

AUTH_LOG = ROOT / "audits/stage9/block913_v1_1_formal_c1_c2_authority_gate.log"
B913_CONTRACT = S9 / "configs/stage9_block913_v1_1_formal_c1_c2_execution_contract.json"
B912_SEAL = S9 / "artifacts/block912_v1_1_c1_c2_core_preformal_freeze/stage9_block912_v1_1_c1_c2_core_preformal_freeze_seal.json"
FORMAL_IDS = S9 / "artifacts/block91b5_opaque_scenario_id_cohort_freeze/stage9_FORMAL_scenario_ids.txt"

B94_MAP = S9 / "artifacts/block94_v1_1_1_calibration/stage9_v1_1_1_block94_isotonic_map.json"
B95_SEAL = S9 / "artifacts/block95_v1_1_1_joint_beam_criticality/stage9_v1_1_1_block95_joint_beam_criticality_seal.json"
B97_CONTRACT = S9 / "configs/stage9_v1_1_1_block97_c2_interface_contract.json"
B97_SEAL = S9 / "artifacts/block97_v1_1_1_c2_policy_cal/stage9_v1_1_1_block97_c2_policy_seal.json"
B96_SEAL = S9 / "artifacts/block96_v1_1_1_heuristic_q_baseline/stage9_v1_1_1_block96_heuristic_q_baseline_seal.json"
BASELINE_COMPARISON = S9 / "artifacts/block97_baseline_comparison/stage9_block97_baseline_comparison_v1.json"

G93_SCRIPT = S9 / "scripts/run_stage9_block93g_full_cal_truth_free_criticality_materialization.py"
M95_SCRIPT = S9 / "scripts/run_stage9_block95_joint_beam_criticality_posterior.py"
S5_SCRIPT = ROOT / "iscai_stage5/scripts/run_block58_part2_formal_evaluation.py"
CRIT_GEOM = S9 / "src/iscai_stage9/criticality_geometry_v1.py"
PATH_ADAPTER = S9 / "src/iscai_stage9/path_adapter_v1_1.py"
VM = ROOT / "iscai_data_prep/manifests/selected_validation.jsonl"
CHECKPOINT = ROOT / "iscai_stage4/artifacts/block44/gaussian_gru.pt"
SCALER = ROOT / "iscai_stage4/artifacts/block45/covariance_scaler.json"

OUT = S9 / "artifacts/block913a_v1_1_formal_c1_c2_prediction_side"
CACHE = OUT / "scenario_cache"
MANIFEST = OUT / "formal_c1_c2_prediction_side_by_scenario.jsonl"
SUMMARY = OUT / "stage9_block913a_v1_1_prediction_side_summary.json"
SEAL = OUT / "stage9_block913a_v1_1_prediction_side_seal.json"
REPORT = S9 / "reports/stage9_block913a_v1_1_prediction_side_report.json"

EXPECTED = {
    str(CONTRACT): "20f824d81eab19d225febc184a9f9baf16dde90297ba2d35ab3e5d5690bd4f77",
    str(AUTH_LOG): "999d5bd0c6bceee3cebe5e8f892517543f132e333122106e660e94c594eb1360",
    str(B913_CONTRACT): "e26d94167268fcc28e2ffca53054de199a951162620286e8d9ac7982a9dbfd1f",
    str(B912_SEAL): "91b873d9620226e036a60fd99d72ae6e4a674596e457b5291f19735b2ca71914",
    str(FORMAL_IDS): "7d3066f08c466cfd2cba64eecdd24474a7e83130a4e90d77e9cdee349e22f03f",
    str(B94_MAP): "6337640f5831c4c1cbca9a7e6922e170ab391fe4437ecd4aae701e0519697f5a",
    str(B95_SEAL): "68a7f6430faab77e00f41da4bf1d999243b8d7ff8be118b0b666faa3b5241ae8",
    str(B97_CONTRACT): "190eee5f3ec3ba7302e3eb44b9000d6951af30a259c650673780be858f89a79d",
    str(B97_SEAL): "4291d58359e5138c152661398fcb29afa49af72e978cb98c680b75930793f18b",
    str(B96_SEAL): "b49461895dbe7395a09762e81051a86701ee3181a075c94d5bb7754076a95508",
    str(BASELINE_COMPARISON): "defade336884e7c30517009dca8af51b999df47eed9621cbfa25d9a2f3aa48a6",
    str(G93_SCRIPT): "1fa95c8c5eaa865c8b93fbe5aec6649d361e74bb4c2e92f7bfa00d76ba8773ee",
    str(M95_SCRIPT): "4f4b28f1e112203831d5ed5aea134faa506466e6b3f0468e6dc30c27ee279b92",
    str(S5_SCRIPT): "16495f3c46ed5f95145918b25b169f8aa33408b2ef84ffc19cb66af1db3c189a",
    str(CRIT_GEOM): "e53b8bea562d0f26989c11cfa1c20ef658f277eb6be53872727227c3204c00a5",
    str(PATH_ADAPTER): "e9cebecba583459fad7017bd6806e826fbcc75c4156e77c39b381834792d70c1",
    str(VM): "dc10609ef18a2ba881657eb3da3a3df7a81bdcc8345ecbc2227102ab16b8833c",
    str(CHECKPOINT): "49ff64d145eaa633f295c16f660df380c35383e7e3b61279a5aad7cd700d619f",
    str(SCALER): "508ff2e3fbcfafe8e001155340c25baaf3772fe2561a8022a9ed1cf780e66087",
}

M = 4096
FORMAL_N = 26209
HORIZON_INDEX = 3
HORIZON_S = 1.0
CLEARANCE_M = 1.0
CODEBOOKS = (16, 32, 64)
NOMINAL_MISS_CAP = 204
NOMINAL_SELECTED_MIN = 3892

class FailClosed(RuntimeError):
    pass

def req(cond, msg):
    if not cond:
        raise FailClosed(msg)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def canonical_bytes(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

def canonical_sha(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()

def atomic_json(path, obj, readonly=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    req(not path.exists(), f"Refusing overwrite: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(
                obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            ).encode("utf-8"))
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if readonly:
        os.chmod(path, 0o444)

def atomic_jsonl_from_cache(path, formal_ids, fingerprint):
    req(not path.exists(), f"Refusing overwrite: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as out:
            for ordinal, sid in enumerate(formal_ids):
                p = cache_path(ordinal, sid)
                req(p.is_file(), f"Missing sealed scenario cache: {p}")
                obj = json.loads(p.read_text(encoding="utf-8"))
                validate_cached(obj, ordinal, sid, fingerprint)
                out.write(canonical_bytes(obj) + b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o444)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    req(spec is not None and spec.loader is not None, f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def apply_isotonic_count(cal_map, critical_count):
    """
    Frozen Block94 semantics:
      left-continuous step on the discrete critical_count grid;
      clamp below first and above last observed support.
    """
    c = int(critical_count)
    req(0 <= c <= M, f"critical_count outside [0,{M}]: {c}")
    blocks = cal_map["blocks"]
    req(isinstance(blocks, list) and blocks, "Block94 map has no blocks")
    prev_q = float(blocks[0]["q"])
    first_min = int(blocks[0]["raw_count_min"])
    if c < first_min:
        return prev_q
    last_max = None
    for b in blocks:
        lo = int(b["raw_count_min"])
        hi = int(b["raw_count_max"])
        q = float(b["q"])
        req(lo <= hi, "Malformed Block94 count interval")
        if last_max is not None:
            req(lo > last_max, "Block94 count intervals overlap/out of order")
        if c < lo:
            return prev_q
        if c <= hi:
            return q
        prev_q = q
        last_max = hi
    return prev_q

def selected_tuple_from_missed(B, missed):
    ms = set(missed)
    return tuple(i for i in range(B) if i not in ms)

def solve_c2_counts(beam_counts, critical_counts):
    """
    Exact sealed v1.1.1 C2 equations.
    We maximize the number of excluded beams under:
      missed nominal <= 204
      missed critical <= floor(C/100), if C>0
    which is equivalent to minimum selected K.

    Tie break among equal K:
      1) minimum nominal missed count
      2) minimum raw critical missed count
      3) lexicographically smallest sorted selected beam tuple
    """
    n = tuple(int(x) for x in beam_counts)
    c = tuple(int(x) for x in critical_counts)
    B = len(n)
    req(B in CODEBOOKS, f"Unexpected codebook B={B}")
    req(len(c) == B, "Critical-count length mismatch")
    req(all(x >= 0 for x in n), "Negative nominal beam count")
    req(all(x >= 0 for x in c), "Negative critical beam count")
    req(sum(n) == M, f"Nominal beam counts do not sum to {M}")
    C = sum(c)
    req(0 <= C <= M, "Critical total outside bounds")
    req(all(c[i] <= n[i] for i in range(B)), "critical count exceeds nominal count")

    ccap = C // 100 if C > 0 else 0

    # state (excluded_k, missed_nominal, missed_critical) -> lexicographically
    # largest missed tuple. For equal cardinality this is equivalent to the
    # lexicographically smallest selected complement.
    states = {(0, 0, 0): ()}
    for i, (ni, ci) in enumerate(zip(n, c)):
        nxt = dict(states)
        for (k, ns, cs), missed in states.items():
            n2 = ns + ni
            c2 = cs + ci
            if n2 > NOMINAL_MISS_CAP or c2 > ccap:
                continue
            key = (k + 1, n2, c2)
            cand = missed + (i,)
            old = nxt.get(key)
            if old is None or cand > old:
                nxt[key] = cand
        states = nxt

    req(states, "C2 dynamic program produced no states")
    max_excluded = max(k for (k, _, _) in states)
    candidates = [
        (ns, cs, missed)
        for (k, ns, cs), missed in states.items()
        if k == max_excluded
    ]
    min_ns = min(ns for ns, _, _ in candidates)
    candidates = [x for x in candidates if x[0] == min_ns]
    min_cs = min(cs for _, cs, _ in candidates)
    candidates = [x for x in candidates if x[1] == min_cs]

    # lexicographically smallest selected complement
    best = min(
        candidates,
        key=lambda x: selected_tuple_from_missed(B, x[2]),
    )
    ns, cs, missed = best
    selected = selected_tuple_from_missed(B, missed)

    selected_nominal = M - ns
    selected_critical = C - cs
    req(selected_nominal >= NOMINAL_SELECTED_MIN, "C2 nominal constraint violation")
    if C > 0:
        req(cs <= C // 100, "C2 critical constraint violation")

    return {
        "selected_beam_indices": list(selected),
        "K": len(selected),
        "selected_nominal_sample_count": selected_nominal,
        "nominal_missed_sample_count": ns,
        "selected_p_mass": selected_nominal / M,
        "nominal_missed_mass": ns / M,
        "critical_count": C,
        "critical_miss_cap": ccap,
        "selected_critical_sample_count": selected_critical,
        "critical_missed_sample_count": cs,
        "selected_u_raw_mass": selected_critical / M,
        "raw_critical_missed_mass": cs / M,
        "raw_conditional_critical_coverage": (
            selected_critical / C if C > 0 else None
        ),
        "raw_conditional_critical_miss": (
            cs / C if C > 0 else None
        ),
        "fallback_used": False,
        "fallback_reason": None,
    }

def c2_with_fallback(beam_counts, critical_counts):
    B = len(beam_counts)
    try:
        return solve_c2_counts(beam_counts, critical_counts)
    except Exception as exc:
        # Frozen local fallback. This is not a global provenance/hash failure.
        n = [int(x) for x in beam_counts]
        c = [int(x) for x in critical_counts]
        if B not in CODEBOOKS or len(c) != B:
            raise
        if any(x < 0 for x in n + c):
            raise
        nsum = sum(n)
        csum = sum(c)
        if nsum != M or not (0 <= csum <= M):
            raise
        return {
            "selected_beam_indices": list(range(B)),
            "K": B,
            "selected_nominal_sample_count": M,
            "nominal_missed_sample_count": 0,
            "selected_p_mass": 1.0,
            "nominal_missed_mass": 0.0,
            "critical_count": csum,
            "critical_miss_cap": csum // 100 if csum > 0 else 0,
            "selected_critical_sample_count": csum,
            "critical_missed_sample_count": 0,
            "selected_u_raw_mass": csum / M,
            "raw_critical_missed_mass": 0.0,
            "raw_conditional_critical_coverage": 1.0 if csum > 0 else None,
            "raw_conditional_critical_miss": 0.0 if csum > 0 else None,
            "fallback_used": True,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }

def cache_path(ordinal, sid):
    return CACHE / f"{ordinal:05d}_{sid}.json"

def validate_cached(obj, ordinal, sid, fingerprint):
    req(obj.get("schema") == "stage9_block913a_v1_1_formal_prediction_side_scenario_v1",
        f"Cache schema mismatch: {sid}")
    req(obj.get("run_fingerprint") == fingerprint, f"Cache fingerprint mismatch: {sid}")
    req(int(obj.get("formal_ordinal")) == int(ordinal), f"Cache ordinal mismatch: {sid}")
    req(str(obj.get("scenario_id")) == str(sid), f"Cache scenario mismatch: {sid}")
    req(obj.get("status") in ("COMPLETE", "NO_CURRENT_CAUSAL_HISTORIES"),
        f"Cache status invalid: {sid}")
    req(obj.get("future_GT_access") is False, f"Cache future-GT flag invalid: {sid}")
    req(obj.get("FORMAL_outcomes_computed") is False, f"Cache outcome flag invalid: {sid}")
    req(obj.get("C3_FORMAL_testing") is False, f"Cache C3 flag invalid: {sid}")
    return obj

def main():
    print("=" * 92)
    print("STAGE 9 — BLOCK 9.13A-v1.1")
    print("FORMAL C1+C2 PREDICTION-SIDE MATERIALIZATION")
    print("NO FUTURE GT | NO FORMAL METRICS | NO C3 | NO TUNING")
    print("=" * 92)

    req(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch")
    req(not SEAL.exists(), "9.13A already sealed; refusing rerun")

    for raw, expected in EXPECTED.items():
        p = Path(raw)
        req(p.is_file(), f"Missing frozen authority: {p}")
        actual = sha_file(p)
        req(actual == expected, f"SHA mismatch: {p}\nexpected={expected}\nactual={actual}")

    authority_text = AUTH_LOG.read_text(encoding="utf-8", errors="replace")
    for marker in (
        "PRE_FORMAL_AUTHORITY_GATE = PASS",
        "FORMAL_COHORT_OPENED = TRUE",
        "FORMAL_OUTCOMES_COMPUTED = FALSE",
        "BLOCK913_AUTHORITY_GATE = PASS",
        "EXECUTION_STATUS = FORMAL_COHORT_OPENED_NO_OUTCOMES_COMPUTED",
    ):
        req(marker in authority_text, f"Missing authority marker: {marker}")

    c913 = json.loads(B913_CONTRACT.read_text(encoding="utf-8"))
    req(c913["status"] == "FROZEN_PRE_EXECUTION_FORMAL_C1_C2", "9.13 contract status drift")
    req(c913["scope"]["primary_confirmatory_scope"] == "C1+C2", "scope drift")
    req(c913["scope"]["C3_FORMAL_confirmatory_testing"] is False, "C3 FORMAL must be disabled")
    req(c913["formal_opening"]["expected_FORMAL_scenario_count"] == FORMAL_N, "FORMAL N drift")
    req(c913["frozen_C1"]["critical_event_horizon_s"] == HORIZON_S, "C1 horizon drift")
    req(c913["frozen_C1"]["critical_clearance_m"] == CLEARANCE_M, "C1 clearance drift")
    req(c913["frozen_C1"]["same_MC_sample_reuse_required"] is True, "same-sample rule drift")
    req(c913["frozen_C2"]["objective"] == "minimum cardinality beam probing", "C2 objective drift")
    req(c913["frozen_C2"]["solver_retuning"] is False, "C2 solver retuning forbidden")
    req(c913["frozen_C2"]["threshold_retuning"] is False, "C2 threshold retuning forbidden")
    req(c913["future_GT_boundary"]["evaluator_only"] is True, "future-GT boundary drift")
    req(c913["future_GT_boundary"]["controller_access"] is False, "controller GT access forbidden")

    c_impl = json.loads(CONTRACT.read_text(encoding="utf-8"))
    req(c_impl["scientific_change"] is False, "Implementation contract cannot change science")
    req(c_impl["source_identity_claim"] is False, "No source-identity claim allowed")

    cal_map = json.loads(B94_MAP.read_text(encoding="utf-8"))
    req(cal_map["status"] == "FROZEN_CAL_FIT", "Block94 map status drift")
    req(cal_map["FORMAL_refit_allowed"] is False, "Block94 FORMAL refit forbidden")
    req(int(cal_map["M"]) == M, "Block94 M drift")
    req(
        cal_map["application_extension"]
        == "left-continuous step on discrete critical_count grid; clamp below first and above last observed support",
        "Block94 application semantics drift",
    )

    b97 = json.loads(B97_CONTRACT.read_text(encoding="utf-8"))
    req(b97["status"] == "FROZEN_PRE_EXECUTION_STAGE9_V1_1_1_BLOCK97", "Block97 interface status drift")
    req(b97["scientific_role"]["primary_receiver_binding_resolved"] is False,
        "Current v1.1.1 C2 unexpectedly resolves primary receiver")
    pc = b97["primary_constraints"]
    req(int(pc["M"]) == M, "Block97 M drift")
    req(float(pc["nominal_q"]) == 0.95, "Block97 nominal q drift")
    req(float(pc["critical_conditional_q"]) == 0.99, "Block97 critical q drift")
    req(int(pc["nominal_max_missed_samples"]) == NOMINAL_MISS_CAP, "nominal cap drift")
    req(b97["why_scalar_calibration_not_redistributed"]["u_b_cal_defined"] is False,
        "u_b_cal must remain undefined")
    req(b97["primary_solver_inputs"]["Pcrit_cal"]["solver_input"] is False,
        "Pcrit_cal must not enter C2 solver")

    formal_ids = [
        x.strip()
        for x in FORMAL_IDS.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    req(len(formal_ids) == FORMAL_N, f"FORMAL N mismatch: {len(formal_ids)}")
    req(len(set(formal_ids)) == FORMAL_N, "FORMAL IDs are not unique")

    # Load immutable implementation carriers only after SHA gates.
    g93 = load_module("_stage9_913a_g93", G93_SCRIPT)
    m95 = load_module("_stage9_913a_m95", M95_SCRIPT)

    import iscai_stage4.data.neural_inputs as ni
    import iscai_stage4.data.supervision as sup
    from iscai_stage4.data.real_pipeline import build_real_causal_inputs, load_frozen_stage2_configs
    from iscai_stage9.criticality_geometry_v1 import (
        world_ego_row_to_h0,
        bind_current_ego_ctrv,
        propagate_ctrv,
    )

    s5 = g93.load_stage5()
    s5_joint = m95.load_stage5_authority(ROOT)

    def forbidden(*args, **kwargs):
        raise FailClosed("Forbidden future-evaluator/controller route called during 9.13A")

    # Explicitly prevent future-truth/evaluator/receiver-selection routes.
    s5.attach_supervision = forbidden
    s5.resolve_sample_truth_track_index = forbidden
    s5.choose_primary_receiver = forbidden
    s5.process_scenario = forbidden
    s5.read_formal_rows = forbidden
    sup.attach_supervision = forbidden
    if hasattr(sup, "_build_future_label"):
        sup._build_future_label = forbidden

    rows = s5.read_validation_manifest(VM)
    by_id = {str(r.scenario_id): r for r in rows}
    req(len(by_id) == 44097, f"Validation manifest unique-ID count drift: {len(by_id)}")
    missing = [sid for sid in formal_ids if sid not in by_id]
    req(not missing, f"FORMAL IDs missing from validation manifest: first={missing[:5]}")

    runtime = s5.build_gaussian_runtime()
    g93.install_prepare_compatibility_shim(runtime)
    clean_cfg, degraded_cfg = load_frozen_stage2_configs()

    provenance = {
        "contract_sha256": sha_file(CONTRACT),
        "authority_log_sha256": sha_file(AUTH_LOG),
        "block913_contract_sha256": sha_file(B913_CONTRACT),
        "block912_seal_sha256": sha_file(B912_SEAL),
        "formal_ids_sha256": sha_file(FORMAL_IDS),
        "block94_map_sha256": sha_file(B94_MAP),
        "block95_seal_sha256": sha_file(B95_SEAL),
        "block97_interface_contract_sha256": sha_file(B97_CONTRACT),
        "block97_seal_sha256": sha_file(B97_SEAL),
        "block96_seal_sha256": sha_file(B96_SEAL),
        "baseline_comparison_sha256": sha_file(BASELINE_COMPARISON),
        "g93_carrier_sha256": sha_file(G93_SCRIPT),
        "m95_mapper_carrier_sha256": sha_file(M95_SCRIPT),
        "stage5_runner_sha256": sha_file(S5_SCRIPT),
        "criticality_geometry_sha256": sha_file(CRIT_GEOM),
        "path_adapter_sha256": sha_file(PATH_ADAPTER),
        "validation_manifest_sha256": sha_file(VM),
        "gaussian_checkpoint_sha256": sha_file(CHECKPOINT),
        "covariance_scaler_sha256": sha_file(SCALER),
        "source_identity_claim": False,
    }
    scientific_config = {
        "cohort": "Stage9 frozen FORMAL IDs",
        "FORMAL_N": FORMAL_N,
        "M": M,
        "horizon_s": HORIZON_S,
        "horizon_index": HORIZON_INDEX,
        "clearance_m": CLEARANCE_M,
        "sample_id_rule": "scenario_id|prediction_id",
        "seed_rule": "first64(SHA256('stage9-mc-v1|'+sample_id))",
        "same_sample_reuse": True,
        "codebooks": list(CODEBOOKS),
        "C2_nominal_miss_cap": NOMINAL_MISS_CAP,
        "C2_critical_rule": "missed_critical <= floor(C/100) when C>0",
        "Pcrit_cal_solver_input": False,
        "u_b_cal_defined": False,
        "future_GT_access": False,
        "C3_FORMAL_testing": False,
        "receiver_semantics": "actor_conditioned_intermediate_only",
    }
    fingerprint = canonical_sha({
        "provenance": provenance,
        "scientific_config": scientific_config,
    })

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    reused = 0
    processed = 0
    total_actors = 0
    nonempty_scenarios = 0
    empty_scenarios = 0
    fallback_actor_codebooks = 0

    for ordinal, sid in enumerate(formal_ids):
        p = cache_path(ordinal, sid)

        if p.exists():
            obj = validate_cached(
                json.loads(p.read_text(encoding="utf-8")),
                ordinal, sid, fingerprint
            )
            reused += 1
        else:
            scenario, read_route = g93.read_cal_scenario(s5, by_id[sid])
            current_index = int(scenario.current_time_index)

            built = build_real_causal_inputs(
                scenario,
                clean_config=clean_cfg,
                degraded_config=degraded_cfg,
            )
            scene_inputs = built["scene_inputs"]
            T = built["adapted"].frames.T_H0_from_W

            histories = []
            for h in tuple(scene_inputs.histories):
                latest = int(h.latest_observed_frame_index)
                if latest != current_index:
                    continue
                steps = tuple(h.steps)
                req(0 <= latest < len(steps), f"Invalid current history index: {sid}")
                if bool(steps[latest].observed):
                    histories.append(h)
            histories.sort(key=lambda h: str(h.prediction_id))

            if not histories:
                obj = {
                    "schema": "stage9_block913a_v1_1_formal_prediction_side_scenario_v1",
                    "run_fingerprint": fingerprint,
                    "formal_ordinal": ordinal,
                    "scenario_id": sid,
                    "read_route": read_route,
                    "current_index": current_index,
                    "status": "NO_CURRENT_CAUSAL_HISTORIES",
                    "conditioning": "actor_conditioned_intermediate_only",
                    "actor_prediction_count": 0,
                    "actors": [],
                    "receiver_selection_executed": False,
                    "receiver_substitution_used": False,
                    "primary_receiver_binding_resolved": False,
                    "future_GT_access": False,
                    "FORMAL_outcomes_computed": False,
                    "C3_FORMAL_testing": False,
                }
            else:
                ego_rows_W, max_ego_index = g93.causal_ego_history_rows(
                    scenario, current_index
                )
                req(max_ego_index == current_index, f"Unexpected ego index: {sid}")
                ego_rows_H0 = [
                    world_ego_row_to_h0(r, T)
                    for r in ego_rows_W
                    if bool(r["valid"])
                ]
                req(ego_rows_H0, f"No valid causal ego history: {sid}")
                ego_future = propagate_ctrv(
                    bind_current_ego_ctrv(ego_rows_H0),
                    horizon_s=HORIZON_S,
                )

                map_index = sup.build_static_map_index_H0(
                    scenario, T_H0_from_W=T
                )

                actors = []
                seen_pids = set()

                for h in histories:
                    pid = str(h.prediction_id)
                    req(pid not in seen_pids, f"Duplicate prediction_id in {sid}: {pid}")
                    seen_pids.add(pid)

                    ctx = sup.map_context_summary(
                        map_index,
                        target_position_H0_m=h.latest_position_H0_m,
                    )
                    payload = ni.build_model_input_payload(
                        scene_inputs,
                        prediction_id=pid,
                        map_context=ctx,
                    )

                    mean4, cov4, _runtime_s = s5.calibrated_receiver_trajectory(
                        runtime,
                        SimpleNamespace(model_input=payload, history=h),
                    )
                    mean4 = np.asarray(mean4, dtype=np.float64)
                    cov4 = np.asarray(cov4, dtype=np.float64)
                    req(mean4.shape == (4, 3), f"Mean shape drift in {sid}|{pid}")
                    req(cov4.shape == (4, 3, 3), f"Cov shape drift in {sid}|{pid}")

                    mean = mean4[HORIZON_INDEX]
                    cov = cov4[HORIZON_INDEX]
                    sample_id = f"{sid}|{pid}"
                    seed = g93.first64_seed(sample_id)
                    samples = g93.sample_endpoint(mean, cov, M, seed)
                    samples_sha = hashlib.sha256(samples.tobytes()).hexdigest()

                    bits = g93.vectorized_critical_bits(
                        samples, ego_future, clearance=CLEARANCE_M
                    )
                    req(bits.shape == (M,), "Critical bit vector shape drift")
                    critical_count = int(np.count_nonzero(bits))
                    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
                    req(packed.size == M // 8, "Packed critical bitset size drift")
                    critical_bitset_sha = hashlib.sha256(packed.tobytes()).hexdigest()

                    pcrit_raw = critical_count / M
                    pcrit_cal = apply_isotonic_count(cal_map, critical_count)
                    req(math.isfinite(pcrit_cal) and 0.0 <= pcrit_cal <= 1.0,
                        "Non-finite/out-of-range Pcrit_cal")

                    # Temporary in-memory carrier expected by the exact historical
                    # Stage9 same-sample beam-cell mapper. The full bitset is NOT
                    # written to the final 9.13A artifact.
                    actor_carrier = {
                        "scenario_id": sid,
                        "prediction_id": pid,
                        "sample_id": sample_id,
                        "seed": int(seed),
                        "mean_1s_H0_m": [float(x) for x in mean],
                        "covariance_1s_H0_m2": [
                            [float(x) for x in rr] for rr in cov
                        ],
                        "samples_sha256": samples_sha,
                        "critical_bitset_hex_little_endian": packed.tobytes().hex(),
                        "critical_count": critical_count,
                        "pcrit_raw": float(pcrit_raw),
                    }

                    joint = m95.joint_for_actor(actor_carrier, s5_joint, M)
                    req(int(joint["critical_count"]) == critical_count,
                        "Joint mapper critical-count mismatch")
                    req(abs(float(joint["pcrit_raw"]) - pcrit_raw) <= 1e-15,
                        "Joint mapper Pcrit_raw mismatch")
                    req(str(joint["samples_sha256"]) == samples_sha,
                        "Joint mapper sample-SHA mismatch")

                    codebook_state = {}
                    for B in CODEBOOKS:
                        j = joint["codebooks"][str(B)]
                        beam_counts = [int(x) for x in j["beam_sample_counts"]]
                        crit_counts = [int(x) for x in j["critical_beam_sample_counts"]]
                        req(sum(beam_counts) == M, f"sum beam counts drift B={B}")
                        req(sum(crit_counts) == critical_count,
                            f"sum critical counts drift B={B}")
                        req(all(crit_counts[i] <= beam_counts[i] for i in range(B)),
                            f"critical>nominal beam count B={B}")

                        decision = c2_with_fallback(beam_counts, crit_counts)
                        fallback_actor_codebooks += int(decision["fallback_used"])

                        codebook_state[str(B)] = {
                            "beam_count": B,
                            "beam_sample_counts": beam_counts,
                            "critical_beam_sample_counts": crit_counts,
                            "sum_p_b": 1.0,
                            "sum_u_b_raw": float(pcrit_raw),
                            "C2": decision,
                        }

                    actors.append({
                        "scenario_id": sid,
                        "prediction_id": pid,
                        "sample_id": sample_id,
                        "seed": int(seed),
                        "payload_sha256": payload.sha256(),
                        "mean_1s_H0_m": [float(x) for x in mean],
                        "covariance_1s_H0_m2": [
                            [float(x) for x in rr] for rr in cov
                        ],
                        "samples_sha256": samples_sha,
                        "critical_bitset_sha256": critical_bitset_sha,
                        "critical_count": critical_count,
                        "Pcrit_raw": float(pcrit_raw),
                        "Pcrit_cal_descriptive_only": float(pcrit_cal),
                        "Pcrit_cal_solver_input": False,
                        "u_b_cal_defined": False,
                        "codebooks": codebook_state,
                    })

                obj = {
                    "schema": "stage9_block913a_v1_1_formal_prediction_side_scenario_v1",
                    "run_fingerprint": fingerprint,
                    "formal_ordinal": ordinal,
                    "scenario_id": sid,
                    "read_route": read_route,
                    "current_index": current_index,
                    "status": "COMPLETE",
                    "conditioning": "actor_conditioned_intermediate_only",
                    "actor_prediction_count": len(actors),
                    "ego_prediction_1s": {
                        "center_xy_H0_m": [
                            float(x) for x in ego_future["center_xy_H0_m"]
                        ],
                        "heading_H0_rad": float(ego_future["heading_H0_rad"]),
                        "length_m": float(ego_future["length_m"]),
                        "width_m": float(ego_future["width_m"]),
                        "mode": str(ego_future["mode"]),
                    },
                    "actors": actors,
                    "receiver_selection_executed": False,
                    "receiver_substitution_used": False,
                    "primary_receiver_binding_resolved": False,
                    "future_GT_access": False,
                    "FORMAL_outcomes_computed": False,
                    "C3_FORMAL_testing": False,
                }

            atomic_json(p, obj, readonly=True)
            processed += 1

        n = int(obj["actor_prediction_count"])
        total_actors += n
        if n:
            nonempty_scenarios += 1
        else:
            empty_scenarios += 1

        done = ordinal + 1
        if done % 100 == 0 or done == FORMAL_N:
            elapsed = time.time() - started
            print(
                f"9.13A progress = {done}/{FORMAL_N} ; "
                f"new={processed} ; reused={reused} ; "
                f"actor_predictions={total_actors} ; elapsed_s={elapsed:.1f}",
                flush=True,
            )

    req(nonempty_scenarios + empty_scenarios == FORMAL_N, "Scenario accounting mismatch")

    atomic_jsonl_from_cache(MANIFEST, formal_ids, fingerprint)

    # Accounting only. No performance metrics / no hypothesis outcomes.
    summary = {
        "schema": "stage9_block913a_v1_1_prediction_side_summary_v1",
        "stage": 9,
        "block": "9.13A-v1.1",
        "status": "FORMAL_PREDICTION_SIDE_MATERIALIZED_NO_GT",
        "FORMAL_scenario_count": FORMAL_N,
        "scenario_rows_materialized": FORMAL_N,
        "scenarios_with_current_causal_histories": nonempty_scenarios,
        "scenarios_without_current_causal_histories": empty_scenarios,
        "actor_prediction_count": total_actors,
        "C2_fallback_actor_codebook_count": fallback_actor_codebooks,
        "conditioning": "actor_conditioned_intermediate_only",
        "primary_receiver_binding_resolved": False,
        "receiver_selection_executed": False,
        "receiver_substitution_used": False,
        "future_GT_access": False,
        "FORMAL_outcomes_computed": False,
        "scientific_metrics_computed": False,
        "hypothesis_decisions_computed": False,
        "C3_FORMAL_testing": False,
        "post_FORMAL_retuning": False,
        "run_fingerprint": fingerprint,
        "manifest_sha256": sha_file(MANIFEST),
    }
    atomic_json(SUMMARY, summary, readonly=True)

    report = {
        "schema": "stage9_block913a_v1_1_prediction_side_report_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK913A_PREDICTION_SIDE_NO_GT",
        "scientific_role": (
            "Implementation-only first subpass of frozen Block9.13; "
            "causal C1/C2 prediction-side state and C2 decisions only."
        ),
        "source_identity_claim": False,
        "sealed_equations_implemented": True,
        "future_GT_access": False,
        "FORMAL_outcomes_computed": False,
        "metrics_or_claim_selection_performed": False,
        "C3_FORMAL_testing": False,
        "summary_sha256": sha_file(SUMMARY),
        "manifest_sha256": sha_file(MANIFEST),
        "next": "9.13B frozen communication-baseline decision materialization, still without future GT",
    }
    atomic_json(REPORT, report, readonly=True)

    seal = {
        "schema": "stage9_block913a_v1_1_prediction_side_seal_v1",
        "stage": 9,
        "block": "9.13A-v1.1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK913A_PREDICTION_SIDE_NO_GT",
        "runner_sha256": sha_file(SCRIPT),
        "contract_sha256": sha_file(CONTRACT),
        "authority_log_sha256": sha_file(AUTH_LOG),
        "block913_contract_sha256": sha_file(B913_CONTRACT),
        "block912_seal_sha256": sha_file(B912_SEAL),
        "FORMAL_ids_sha256": sha_file(FORMAL_IDS),
        "block94_map_sha256": sha_file(B94_MAP),
        "block95_seal_sha256": sha_file(B95_SEAL),
        "block97_interface_contract_sha256": sha_file(B97_CONTRACT),
        "block97_seal_sha256": sha_file(B97_SEAL),
        "block96_seal_sha256": sha_file(B96_SEAL),
        "baseline_comparison_sha256": sha_file(BASELINE_COMPARISON),
        "g93_carrier_sha256": sha_file(G93_SCRIPT),
        "historical_block95_mapper_carrier_sha256": sha_file(M95_SCRIPT),
        "stage5_runner_sha256": sha_file(S5_SCRIPT),
        "run_fingerprint": fingerprint,
        "manifest_sha256": sha_file(MANIFEST),
        "summary_sha256": sha_file(SUMMARY),
        "report_sha256": sha_file(REPORT),
        "FORMAL_scenario_count": FORMAL_N,
        "future_GT_access": False,
        "FORMAL_outcomes_computed": False,
        "scientific_metrics_computed": False,
        "hypothesis_decisions_computed": False,
        "C3_FORMAL_testing": False,
        "source_identity_claim": False,
        "post_seal_retuning_allowed": False,
        "next": "9.13B_FROZEN_BASELINE_DECISION_MATERIALIZATION_NO_GT",
    }
    atomic_json(SEAL, seal, readonly=True)

    print()
    print("===== 9.13A FINAL =====")
    print("STATUS = FROZEN_COMPLETE_STAGE9_BLOCK913A_PREDICTION_SIDE_NO_GT")
    print("FORMAL_SCENARIO_COUNT =", FORMAL_N)
    print("FORMAL_OUTCOMES_COMPUTED = FALSE")
    print("FUTURE_GT_ACCESSED = FALSE")
    print("SCIENTIFIC_METRICS_COMPUTED = FALSE")
    print("HYPOTHESIS_DECISIONS_COMPUTED = FALSE")
    print("C3_FORMAL_TESTING = FALSE")
    print("SOURCE_IDENTITY_CLAIM = FALSE")
    print("MANIFEST_SHA256 =", sha_file(MANIFEST))
    print("SUMMARY_SHA256 =", sha_file(SUMMARY))
    print("REPORT_SHA256 =", sha_file(REPORT))
    print("SEAL_SHA256 =", sha_file(SEAL))
    print("NEXT = 9.13B FROZEN BASELINE DECISIONS, STILL NO FUTURE GT")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("!" * 92)
        print("STAGE9 BLOCK 9.13A FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("PARTIAL SEALED SCENARIO CACHES, IF ANY, ARE IMMUTABLE EVIDENCE.")
        print("DO NOT RETUNE. DO NOT COMPUTE FORMAL OUTCOMES.")
        print("!" * 92)
        raise

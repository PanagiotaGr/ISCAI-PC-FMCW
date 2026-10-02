#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

FORMAL_N = 26209
VALID_COMM_SCENES = 24917
CODEBOOKS = (16, 32, 64)
POLICIES = (
    "fixed_top_1",
    "fixed_top_3",
    "fixed_top_5",
    "fixed_q_0.95",
    "fixed_q_0.99",
    "exhaustive",
    "uncertainty_only_adaptive_topk",
    "heuristic_q",
    "proposed_C2",
    "oracle_eval_only",
)

# -------------------------------------------------------------------------------------------------
# Frozen upstream authorities.
# -------------------------------------------------------------------------------------------------
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
OUTCOME_DIR = B913 / "scenario_outcomes"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_CONTRACT = B915A / "stage9_block915a_v2R4_statistics_contract.json"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

B915C = S9 / "artifacts/block915c_v2_robustness"
P915C_SEAL = B915C / "stage9_block915c_v2_robustness_seal.json"

B915D = S9 / "artifacts/block915d_v2R1_failure_case_analysis"
P915D_AUDIT = B915D / "stage9_block915d_v2R1_failure_slice_authority_audit.json"
P915D_REPORT = B915D / "stage9_block915d_v2R1_failure_case_report.json"
P915D_SEAL = B915D / "stage9_block915d_v2R1_failure_case_seal.json"

EXPECTED = {
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(P915A_CONTRACT): "91362b13413565501d3d33905d78bc12f14bb9c8b3739783e899cbc6088c7b03",
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",
    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",
    str(P915C_SEAL): "a8fa3276e29e7093dd0e8d1131c8fd4708500d3b9eef7bd17a7740885c962a5b",
    str(P915D_AUDIT): "a7d5bc05b806e5c9ff2b0e88f022bc05217645a4dabe7316812da24a14eae13c",
    str(P915D_REPORT): "937e211b00ba4966ddb85935f0fc07f047ee6c4aacaa8fe56b141a551a6fc297",
    str(P915D_SEAL): "465e1ab9a8fc06190780121708628531953d2576d02a86d22669714e8cc4f1ca",
}

OUT = S9 / "artifacts/block915e_v2R1_latency_resources"
AUDIT = OUT / "stage9_block915e_v2R1_latency_resource_carrier_audit.json"
REPORT = OUT / "stage9_block915e_v2R1_latency_resources_report.json"
SEAL = OUT / "stage9_block915e_v2R1_latency_resources_seal.json"


class FailClosed(RuntimeError):
    pass


def req(c: bool, msg: str) -> None:
    if not c:
        raise FailClosed(msg)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(obj: Any) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, obj: Any) -> None:
    data = canonical_bytes(obj)
    if path.exists():
        req(path.read_bytes() == data, f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def hash_gate() -> None:
    for raw, wanted in EXPECTED.items():
        p = Path(raw)
        req(p.is_file(), f"missing frozen authority: {p}")
        got = sha256_path(p)
        req(got == wanted, f"SHA256 drift {p}: {got} != {wanted}")
        print("EXACT PASS", p)


def descriptive(x: list[float]) -> dict[str, Any]:
    req(len(x) > 0, "descriptive() called with empty vector")
    a = np.asarray(x, dtype=np.float64)
    req(np.all(np.isfinite(a)), "non-finite value in descriptive carrier")
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95, method="linear")),
        "max": float(np.max(a)),
    }


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15E-v2R1 — LATENCY / RESOURCES")
    print("RECORDED FORMAL CARRIERS ONLY | NO TIMING RERUN | NO MODEL/POLICY/SOLVER RERUN | NO NEW RNG")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    a_contract = load_json(P915A_CONTRACT)
    a_seal = load_json(P915A_SEAL)
    b = load_json(P915B_SUMMARY)
    c = load_json(P915C_SEAL)
    d = load_json(P915D_SEAL)

    req(
        a_seal.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915A_V2R4_PRE_INFERENCE_CONTRACT",
        "9.15A status drift",
    )
    req(
        b.get("status") == "CURRENT_FORMAL_H1_H6_STATISTICS_COMPLETE",
        "9.15B status drift",
    )
    req(
        c.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915C_V2_ROBUSTNESS",
        "9.15C status drift",
    )
    req(
        d.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915D_V2R1_FAILURE_CASE_ANALYSIS",
        "9.15D status drift",
    )

    # 9.15A-v2R4 is hash-bound above.  Its frozen JSON schema does not contain a
    # top-level "latency" section.  Do not mutate/supersede 9.15A merely to add one.
    #
    # The Stage-9 no-rerun boundary is enforced here operationally: this 9.15E runner
    # reads only already-materialized final 9.13 outcomes and computes descriptive
    # summaries.  It imports/executes no predictor, no policy, no C1 MC, no C2 solver,
    # no C3 controller, and creates no timing experiment.
    contract_latency_section_present = isinstance(a_contract.get("latency"), dict)
    req(
        a_contract.get("status") == "FROZEN_POST_FORMAL_PRE_CURRENT_INFERENCE_CONTRACT_REPAIRED",
        "9.15A contract status drift",
    )
    req(
        a_contract.get("block") == "9.15A-v2R4",
        "9.15A contract block identity drift",
    )

    print("\n===== A. FULL FORMAL LATENCY / RESOURCE CARRIER AUDIT =====")

    files = sorted(OUTCOME_DIR.glob("*.json"))
    req(len(files) == FORMAL_N, f"FORMAL outcome cardinality drift: {len(files)}")

    status_counts = Counter()
    policy_codebook_counts = Counter()

    # Per policy / codebook resource carriers.
    K = defaultdict(list)
    probes = defaultdict(list)
    overhead = defaultdict(list)
    effective_rate = defaultdict(list)

    # Timing carrier accounting.
    runtime_finite = defaultdict(list)
    runtime_null = Counter()
    runtime_nonfinite = Counter()

    # Global checks for any top-level or nested timing-like carrier other than solver_runtime_s.
    timing_like_key_counts = Counter()

    h6_count_values = []
    h5_support_n_total = 0

    def scan_keys(x: Any) -> None:
        if isinstance(x, dict):
            for kk, vv in x.items():
                kl = str(kk).lower()
                if any(t in kl for t in ("runtime", "latency", "elapsed", "timing")):
                    timing_like_key_counts[str(kk)] += 1
                scan_keys(vv)
        elif isinstance(x, list):
            for vv in x:
                scan_keys(vv)

    for i, path in enumerate(files):
        j = load_json(path)
        req(int(j.get("formal_ordinal", -1)) == i, f"formal ordinal drift: {path.name}")
        req(j.get("status") == "COMPLETE_FINAL_EVALUATOR_H1_H6",
            f"outcome status drift: {path.name}")
        req(j.get("C1_MC_recomputed") is False, f"C1 recomputed: {path.name}")
        req(j.get("C2_solver_recomputed") is False, f"C2 solver recomputed: {path.name}")
        req(j.get("model_forward_executed") is False, f"model forward drift: {path.name}")
        req(j.get("policy_retuned") is False, f"policy retuning drift: {path.name}")
        req(j.get("thresholds_retuned") is False, f"threshold drift: {path.name}")

        scan_keys(j)

        primary = j.get("primary")
        req(isinstance(primary, dict), f"primary missing: {path.name}")
        pstatus = str(primary.get("status"))
        status_counts[pstatus] += 1

        comm = primary.get("communication_records", [])
        req(isinstance(comm, list), f"communication_records not list: {path.name}")

        if pstatus == "EVALUATED_FUTURE_GT":
            by_key = {}
            for r in comm:
                pol = str(r["policy"])
                cb = int(r["codebook_size"])
                req(pol in POLICIES, f"unexpected communication policy {pol}: {path.name}")
                req(cb in CODEBOOKS, f"unexpected codebook {cb}: {path.name}")
                key = (pol, cb)
                req(key not in by_key, f"duplicate policy/codebook: {path.name} {key}")
                by_key[key] = r

                kval = int(r["K"])
                pval = int(r["physical_probe_count"])
                oval = float(r["probing_overhead_fraction"])
                rate = float(r["effective_rate_bps"])

                req(kval >= 0, f"negative K: {path.name} {key}")
                req(pval >= 0, f"negative probe count: {path.name} {key}")
                req(math.isfinite(oval) and oval >= 0.0, f"invalid overhead: {path.name} {key}")
                req(math.isfinite(rate) and rate >= 0.0, f"invalid effective rate: {path.name} {key}")

                K[key].append(kval)
                probes[key].append(pval)
                overhead[key].append(oval)
                effective_rate[key].append(rate)
                policy_codebook_counts[key] += 1

                rt = r.get("solver_runtime_s")
                if rt is None:
                    runtime_null[key] += 1
                else:
                    rt = float(rt)
                    if math.isfinite(rt) and rt >= 0.0:
                        runtime_finite[key].append(rt)
                    else:
                        runtime_nonfinite[key] += 1

            expected_keys = {(p, cb) for p in POLICIES for cb in CODEBOOKS}
            req(set(by_key) == expected_keys,
                f"communication policy/codebook completeness drift: {path.name}")

        elif pstatus in (
            "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID",
            "NOT_EVALUATED_NO_ELIGIBLE_RECEIVER",
        ):
            req(comm == [], f"non-communication scene has communication records: {path.name}")
        else:
            raise FailClosed(f"unexpected primary status {pstatus!r}: {path.name}")

        h6n = j.get("H6_matched_eligible_count")
        req(isinstance(h6n, int) and h6n >= 0, f"H6 count invalid: {path.name}")
        h6_count_values.append(h6n)

        h5 = j.get("H5_raw_counts")
        req(isinstance(h5, dict), f"H5 counts missing: {path.name}")
        # Count only one baseline/profile carrier per scene to avoid calling six duplicated
        # common-support representations six times the computational workload.
        q = h5["H_nominal"]["original_reactive_ADB"]
        h5_support_n_total += int(q["n"])

        if (i + 1) % 5000 == 0:
            print(f"LATENCY/RESOURCE AUDIT {i+1}/{FORMAL_N}", flush=True)

    req(status_counts["EVALUATED_FUTURE_GT"] == VALID_COMM_SCENES,
        "valid communication scene count drift")
    for key in ((p, cb) for p in POLICIES for cb in CODEBOOKS):
        req(policy_codebook_counts[key] == VALID_COMM_SCENES,
            f"policy/codebook record count drift: {key} {policy_codebook_counts[key]}")

    # Timing-like keys are expected to include solver_runtime_s only in the final outcome carrier.
    timing_keys = dict(sorted(timing_like_key_counts.items()))
    unexpected_timing_keys = sorted(
        k for k in timing_keys if k != "solver_runtime_s"
    )
    req(
        not unexpected_timing_keys,
        f"unexpected FORMAL timing-like carrier(s) found; manual review required: {unexpected_timing_keys}",
    )

    # C2 solver timing completeness.
    c2_total = VALID_COMM_SCENES * len(CODEBOOKS)
    c2_finite = sum(len(runtime_finite[("proposed_C2", cb)]) for cb in CODEBOOKS)
    c2_null = sum(runtime_null[("proposed_C2", cb)] for cb in CODEBOOKS)
    c2_nonfinite = sum(runtime_nonfinite[("proposed_C2", cb)] for cb in CODEBOOKS)
    req(c2_finite + c2_null + c2_nonfinite == c2_total, "C2 runtime accounting drift")

    if c2_finite == 0:
        c2_latency_status = "NOT_EVALUABLE_NO_RECORDED_FORMAL_SOLVER_TIMING"
    elif c2_finite == c2_total and c2_nonfinite == 0:
        c2_latency_status = "FORMAL_RECORDED_SOLVER_TIMING_COMPLETE"
    else:
        c2_latency_status = "PARTIAL_RECORDED_TIMING_DESCRIPTIVE_ONLY"

    resource_by_policy_codebook = {}
    for pol in POLICIES:
        for cb in CODEBOOKS:
            key = (pol, cb)
            out_key = f"{pol}__cb{cb}"
            resource_by_policy_codebook[out_key] = {
                "K": descriptive(K[key]),
                "physical_probe_count": descriptive(probes[key]),
                "probing_overhead_fraction": descriptive(overhead[key]),
                "effective_rate_bps": descriptive(effective_rate[key]),
            }

    c2_timing_by_codebook = {}
    for cb in CODEBOOKS:
        key = ("proposed_C2", cb)
        finite = runtime_finite[key]
        c2_timing_by_codebook[str(cb)] = {
            "total_records": VALID_COMM_SCENES,
            "finite_runtime_count": len(finite),
            "null_runtime_count": int(runtime_null[key]),
            "nonfinite_runtime_count": int(runtime_nonfinite[key]),
            "recorded_runtime_s_descriptive": descriptive(finite) if finite else None,
        }

    # Resource deltas are descriptive raw FORMAL differences; inference for H3 already lives in 9.15B.
    c2_resource_deltas = {}
    for cb in CODEBOOKS:
        proposed_K = np.asarray(K[("proposed_C2", cb)], dtype=float)
        q99_K = np.asarray(K[("fixed_q_0.99", cb)], dtype=float)
        unc_K = np.asarray(K[("uncertainty_only_adaptive_topk", cb)], dtype=float)

        proposed_O = np.asarray(overhead[("proposed_C2", cb)], dtype=float)
        q99_O = np.asarray(overhead[("fixed_q_0.99", cb)], dtype=float)
        unc_O = np.asarray(overhead[("uncertainty_only_adaptive_topk", cb)], dtype=float)

        c2_resource_deltas[str(cb)] = {
            "proposed_minus_q099_mean_K": float(np.mean(proposed_K - q99_K)),
            "proposed_minus_q099_mean_overhead_fraction": float(np.mean(proposed_O - q99_O)),
            "proposed_minus_uncertainty_only_mean_K": float(np.mean(proposed_K - unc_K)),
            "proposed_minus_uncertainty_only_mean_overhead_fraction": float(np.mean(proposed_O - unc_O)),
            "role": "DESCRIPTIVE_RESOURCE_CONTEXT_ONLY_INFERENCE_ALREADY_IN_9_15B",
        }

    h6_arr = np.asarray(h6_count_values, dtype=float)

    audit = {
        "schema": "stage9_block915e_v2R1_latency_resource_carrier_audit_v1",
        "status": "PASS_FORMAL_LATENCY_RESOURCE_CARRIER_AUDIT",
        "authority": {
            "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage915a_contract_sha256": EXPECTED[str(P915A_CONTRACT)],
            "stage915a_seal_sha256": EXPECTED[str(P915A_SEAL)],
            "stage915b_seal_sha256": EXPECTED[str(P915B_SEAL)],
            "stage915c_seal_sha256": EXPECTED[str(P915C_SEAL)],
            "stage915d_seal_sha256": EXPECTED[str(P915D_SEAL)],
        },
        "repair_provenance": {
            "failed_v2_runner_sha256": "c9c435a8c2010c830304f10dcecd19b5519a0458a727d9a0f933099a8da6d17c",
            "failed_v2_reason": (
                "validator incorrectly required a top-level 9.15A latency section "
                "that is absent from the exact hash-bound v2R4 contract schema"
            ),
            "scientific_endpoint_or_disposition_changed": False,
        },
        "stage915a_contract_latency_section_present": contract_latency_section_present,
        "no_rerun_guard_source": (
            "READ_ONLY_9_15E_EXECUTION_PATH_PLUS_EXACT_HASH_BOUND_UPSTREAM_AUTHORITIES"
        ),
        "FORMAL_scene_count": FORMAL_N,
        "primary_status_counts": dict(sorted(status_counts.items())),
        "communication_valid_scene_count": VALID_COMM_SCENES,
        "communication_policy_count": len(POLICIES),
        "codebook_sizes": list(CODEBOOKS),
        "timing_like_keys_in_final_outcomes": timing_keys,
        "unexpected_timing_like_keys": unexpected_timing_keys,
        "C2_solver_timing": {
            "status": c2_latency_status,
            "total_proposed_C2_records": c2_total,
            "finite_runtime_count": c2_finite,
            "null_runtime_count": c2_null,
            "nonfinite_runtime_count": c2_nonfinite,
            "by_codebook": c2_timing_by_codebook,
        },
        "C3_FORMAL_timing": {
            "status": "NOT_EVALUABLE_NO_FORMAL_C3_RUNTIME_CARRIER_IN_FINAL_OUTCOMES",
            "formal_runtime_field_found": False,
            "CAL_candidate_action_runtime_may_exist_but_is_not_FORMAL_latency": True,
            "posthoc_C3_rerun_for_timing": False,
        },
        "resource_carriers_complete": True,
        "new_timing_measurement_executed": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "model_policy_solver_rerun": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT, audit)

    report = {
        "schema": "stage9_block915e_v2R1_latency_resources_report_v1",
        "status": "STAGE9_915E_LATENCY_RESOURCES_COMPLETE",
        "authority_audit_sha256": sha256_path(AUDIT),
        "repair_provenance": {
            "failed_v2_runner_sha256": "c9c435a8c2010c830304f10dcecd19b5519a0458a727d9a0f933099a8da6d17c",
            "validator_repair_only": True,
            "scientific_endpoint_or_disposition_changed": False,
        },
        "latency": {
            "C2_solver": {
                "status": c2_latency_status,
                "by_codebook": c2_timing_by_codebook,
                "FORMAL_system_latency_claim_authorized": (
                    c2_latency_status == "FORMAL_RECORDED_SOLVER_TIMING_COMPLETE"
                ),
            },
            "C3_recovery_ADB": {
                "status": "NOT_EVALUABLE_NO_FORMAL_C3_RUNTIME_CARRIER",
                "FORMAL_system_latency_claim_authorized": False,
                "CAL_runtime_generalized_to_FORMAL": False,
            },
            "end_to_end_stage9_latency": {
                "status": "NOT_EVALUABLE_NO_COMPLETE_END_TO_END_FORMAL_TIMING_CARRIER",
                "claim_authorized": False,
            },
        },
        "resources": {
            "communication_resource_statistics_by_policy_codebook": resource_by_policy_codebook,
            "proposed_C2_descriptive_deltas": c2_resource_deltas,
            "H3_inference_authority": "9.15B",
            "H6_matched_eligible_actor_count_per_scene": {
                "n_scenes": FORMAL_N,
                "mean": float(np.mean(h6_arr)),
                "median": float(np.median(h6_arr)),
                "p95": float(np.quantile(h6_arr, 0.95, method="linear")),
                "max": int(np.max(h6_arr)),
                "role": "EVALUATOR_WORKLOAD_CONTEXT_NOT_LATENCY",
            },
            "H5_nominal_common_support_evaluation_count_total": int(h5_support_n_total),
            "H5_support_count_role": "EVALUATOR_WORKLOAD_CONTEXT_NOT_WALL_CLOCK_LATENCY",
        },
        "claim_boundary": {
            "additional_safety_reasoning_without_second_learned_network": (
                "ARCHITECTURAL_FACT_ONLY_NOT_A_LATENCY_OR_SPEEDUP_CLAIM"
            ),
            "low_latency_claim": "NOT_AUTHORIZED_UNLESS_RECORDED_FORMAL_TIMING_SUPPORTS_IT",
            "speedup_claim": "NOT_AUTHORIZED",
            "hardware_realtime_claim": "NOT_AUTHORIZED",
            "CAL_candidate_runtime_as_FORMAL_substitute": False,
        },
        "new_timing_measurement_executed": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "retuning": False,
        "model_policy_solver_rerun": False,
        "next_block": "9.15F_FINAL_915_CLOSURE_SEAL",
    }
    atomic_json(REPORT, report)

    seal = {
        "schema": "stage9_block915e_v2R1_latency_resources_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915E_V2R1_LATENCY_RESOURCES",
        "stage915d_seal_sha256": EXPECTED[str(P915D_SEAL)],
        "failed_v2_runner_sha256": "c9c435a8c2010c830304f10dcecd19b5519a0458a727d9a0f933099a8da6d17c",
        "validator_repair_only": True,
        "scientific_endpoint_or_disposition_changed": False,
        "audit_sha256": sha256_path(AUDIT),
        "report_sha256": sha256_path(REPORT),
        "C2_latency_status": c2_latency_status,
        "C3_latency_status": "NOT_EVALUABLE_NO_FORMAL_C3_RUNTIME_CARRIER",
        "end_to_end_latency_status": "NOT_EVALUABLE_NO_COMPLETE_END_TO_END_FORMAL_TIMING_CARRIER",
        "resource_statistics_complete": True,
        "new_timing_measurement_executed": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "post_FORMAL_timing_rerun": False,
        "retuning": False,
        "next_block": "9.15F_FINAL_915_CLOSURE_SEAL",
    }
    atomic_json(SEAL, seal)

    print("LATENCY_RESOURCE_CARRIER_AUDIT = PASS")
    print("STAGE915A_TOP_LEVEL_LATENCY_SECTION_PRESENT =", contract_latency_section_present)
    print("NO_RERUN_GUARD = READ_ONLY_9_15E_EXECUTION_PATH")
    print("C2_FORMAL_SOLVER_LATENCY =", c2_latency_status)
    print("C2_FINITE_RUNTIME_RECORDS =", c2_finite, "/", c2_total)
    print("C3_FORMAL_LATENCY = NOT_EVALUABLE_NO_FORMAL_C3_RUNTIME_CARRIER")
    print("END_TO_END_STAGE9_LATENCY = NOT_EVALUABLE_NO_COMPLETE_END_TO_END_FORMAL_TIMING_CARRIER")
    print("RESOURCE_STATISTICS_COMPLETE = TRUE")
    print("CAL_RUNTIME_USED_AS_FORMAL_SUBSTITUTE = FALSE")
    print("NEW_TIMING_MEASUREMENT_EXECUTED = FALSE")
    print("NEW_RNG = FALSE")
    print("NEW_BOOTSTRAP = FALSE")
    print("STAGE 9.15E-v2R1 = COMPLETE")
    print("AUDIT_SHA256 =", sha256_path(AUDIT))
    print("REPORT_SHA256 =", sha256_path(REPORT))
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15F_FINAL_915_CLOSURE_SEAL")
    print("=" * 124)


if __name__ == "__main__":
    main()

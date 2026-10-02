#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

# -------------------------------------------------------------------------------------------------
# Immutable upstream Stage 9 authorities.
# -------------------------------------------------------------------------------------------------
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

B914 = S9 / "artifacts/block914_v2_external_deepsense_boundary"
P914_SEAL = B914 / "stage9_block914_v2_external_boundary_seal.json"

# 9.15A authoritative repaired pre-inference contract.
B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_EVIDENCE = B915A / "stage9_block915a_v2R4_h1_schema_status_censoring_evidence.json"
P915A_EST = B915A / "stage9_block915a_v2R4_estimand_registry.json"
P915A_CONTRACT = B915A / "stage9_block915a_v2R4_statistics_contract.json"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

# 9.15B statistics.
B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_PREFLIGHT = B915B / "stage9_block915b_v2_prebootstrap_schema_gate.json"
P915B_REPS = B915B / "stage9_block915b_v2_bootstrap_replicates.npz"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

# 9.15C robustness.
B915C = S9 / "artifacts/block915c_v2_robustness"
P915C_REPORT = B915C / "stage9_block915c_v2_robustness_report.json"
P915C_SEAL = B915C / "stage9_block915c_v2_robustness_seal.json"

# 9.15D failure-case analysis.
B915D = S9 / "artifacts/block915d_v2R1_failure_case_analysis"
P915D_AUDIT = B915D / "stage9_block915d_v2R1_failure_slice_authority_audit.json"
P915D_REPORT = B915D / "stage9_block915d_v2R1_failure_case_report.json"
P915D_SEAL = B915D / "stage9_block915d_v2R1_failure_case_seal.json"

# 9.15E latency/resources.
B915E = S9 / "artifacts/block915e_v2R1_latency_resources"
P915E_AUDIT = B915E / "stage9_block915e_v2R1_latency_resource_carrier_audit.json"
P915E_REPORT = B915E / "stage9_block915e_v2R1_latency_resources_report.json"
P915E_SEAL = B915E / "stage9_block915e_v2R1_latency_resources_seal.json"

EXPECTED = {
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(P914_SEAL): "9d9ec48e9c75307861a5b895e1b26365328abe748e43cca5598021ea70031ddd",

    str(P915A_EVIDENCE): "c96b9f969c5e6d1f1e760313d5c58d279746044fb3846cba63c67a1a7bc785a0",
    str(P915A_EST): "4ecc05f12b5c11e46a976b161fd7866d30bda76f287d4904d7cd52e95a71e37c",
    str(P915A_CONTRACT): "91362b13413565501d3d33905d78bc12f14bb9c8b3739783e899cbc6088c7b03",
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",

    str(P915B_PREFLIGHT): "3fff3acb783b74cde2b9b28fb38901930ba154ab7af63732235daad58aaaa7ef",
    str(P915B_REPS): "ef31ae620f02f208bd38e8b87b10b19078e23e1b68c9a8d044897f7f3fc4d0bc",
    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",

    str(P915C_REPORT): "a42be7cc37e5c891d4c26a0ac6ceea472074fca915bdd36d33896b15b472cb6c",
    str(P915C_SEAL): "a8fa3276e29e7093dd0e8d1131c8fd4708500d3b9eef7bd17a7740885c962a5b",

    str(P915D_AUDIT): "a7d5bc05b806e5c9ff2b0e88f022bc05217645a4dabe7316812da24a14eae13c",
    str(P915D_REPORT): "937e211b00ba4966ddb85935f0fc07f047ee6c4aacaa8fe56b141a551a6fc297",
    str(P915D_SEAL): "465e1ab9a8fc06190780121708628531953d2576d02a86d22669714e8cc4f1ca",

    str(P915E_AUDIT): "3178bd85018c47378590c9250e1df341ffef6c55b97c43f5cc286dad6204599f",
    str(P915E_REPORT): "7e83fbd79c5c9f79c555dd44b32e490cd38d8d73f5a39ba7031271a163e5e496",
    str(P915E_SEAL): "c204eb243ac093b370e3e988be29217d82e7e779f078d0d7e8ce482fd9f8ab12",
}

OUT = S9 / "artifacts/block915f_v2_final_stage915_closure"
SUMMARY = OUT / "stage9_block915f_v2_final_stage915_summary.json"
SEAL = OUT / "stage9_block915f_v2_final_stage915_seal.json"


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
        req(p.is_file(), f"missing authoritative artifact: {p}")
        got = sha256_path(p)
        req(got == wanted, f"SHA256 drift {p}: {got} != {wanted}")
        print("EXACT PASS", p)


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15F-v2 — FINAL 9.15 CLOSURE SEAL")
    print("HASH / STATUS / CLAIM-BOUNDARY FREEZE ONLY — NO OUTCOME SCAN / NO RNG / NO NEW STATISTICS / NO RERUN")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    a = load_json(P915A_SEAL)
    b = load_json(P915B_SUMMARY)
    bs = load_json(P915B_SEAL)
    c = load_json(P915C_REPORT)
    cs = load_json(P915C_SEAL)
    d = load_json(P915D_REPORT)
    ds = load_json(P915D_SEAL)
    e = load_json(P915E_REPORT)
    es = load_json(P915E_SEAL)

    # ---------------------------------------------------------------------------------------------
    # Exact completed-subblock identities.
    # ---------------------------------------------------------------------------------------------
    req(
        a.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915A_V2R4_PRE_INFERENCE_CONTRACT",
        "9.15A status drift",
    )
    req(
        b.get("status") == "CURRENT_FORMAL_H1_H6_STATISTICS_COMPLETE",
        "9.15B summary status drift",
    )
    req(
        bs.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915B_V2_CURRENT_FORMAL_H1_H6_STATISTICS",
        "9.15B seal status drift",
    )
    req(
        cs.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915C_V2_ROBUSTNESS",
        "9.15C seal status drift",
    )
    req(
        ds.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915D_V2R1_FAILURE_CASE_ANALYSIS",
        "9.15D seal status drift",
    )
    req(
        es.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915E_V2R1_LATENCY_RESOURCES",
        "9.15E seal status drift",
    )

    # ---------------------------------------------------------------------------------------------
    # Freeze the actual scientific dispositions.  These are not re-estimated here.
    # ---------------------------------------------------------------------------------------------
    req(
        b["H1"]["status"] == "SUPPORTED_CALIBRATION_REDUCES_MACRO_ABS_COVERAGE_ERROR",
        "H1 disposition drift",
    )
    req(
        b["H2"]["status"] == "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H2 disposition drift",
    )
    req(
        b["H3"]["status"] == "PARTIALLY_EVALUABLE_RESOURCE_COMPONENT_ONLY",
        "H3 disposition drift",
    )
    req(
        b["H4"]["status"] == "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H4 disposition drift",
    )
    req(
        b["H5"]["status"] == "SECONDARY_FORMAL_SENSITIVITY_ALL_SIX_REPORTED",
        "H5 disposition drift",
    )
    req(
        b["H6"]["status"] == "PASS_FROZEN_STAGE6_CORE_GATES_WITH_BOOTSTRAP_CONTEXT",
        "H6 disposition drift",
    )

    req(
        c["human_profile_robustness"]["status"] == "PROFILE_SENSITIVE_PAIRWISE_H5_CONCLUSIONS",
        "human-profile robustness disposition drift",
    )
    req(
        c["calibration_robustness"]["status"]
        == "NOT_EVALUABLE_WITHOUT_PREFROZEN_NUMERIC_PERTURBATION_RULE",
        "calibration robustness disposition drift",
    )

    req(
        d["requested_failure_case_analysis"]["requested_slice_count"] == 17,
        "failure-slice count drift",
    )
    req(
        d["requested_failure_case_analysis"]["evaluable_requested_slice_count"] == 0,
        "failure-slice evaluability drift",
    )
    req(
        d["requested_failure_case_analysis"]["metric_or_slice_substitution"] is False,
        "failure-slice substitution drift",
    )

    req(
        e["latency"]["C2_solver"]["status"]
        == "NOT_EVALUABLE_NO_RECORDED_FORMAL_SOLVER_TIMING",
        "C2 latency disposition drift",
    )
    req(
        e["latency"]["C3_recovery_ADB"]["status"]
        == "NOT_EVALUABLE_NO_FORMAL_C3_RUNTIME_CARRIER",
        "C3 latency disposition drift",
    )
    req(
        e["latency"]["end_to_end_stage9_latency"]["status"]
        == "NOT_EVALUABLE_NO_COMPLETE_END_TO_END_FORMAL_TIMING_CARRIER",
        "end-to-end latency disposition drift",
    )
    req(
        es.get("resource_statistics_complete") is True,
        "resource statistics completeness drift",
    )

    # ---------------------------------------------------------------------------------------------
    # Non-negotiable post-FORMAL integrity guards.
    # ---------------------------------------------------------------------------------------------
    req(bs.get("zero_support_metric_substitution") is False,
        "zero-support metric substitution drift")
    req(bs.get("post_FORMAL_retuning") is False, "9.15B retuning drift")
    req(cs.get("new_bootstrap_draws") is False, "9.15C new bootstrap drift")
    req(cs.get("new_RNG_used") is False, "9.15C new RNG drift")
    req(ds.get("post_FORMAL_new_thresholds_defined") is False,
        "9.15D new threshold drift")
    req(ds.get("post_FORMAL_new_membership_rules_defined") is False,
        "9.15D new membership-rule drift")
    req(es.get("new_timing_measurement_executed") is False,
        "9.15E new timing experiment drift")
    req(es.get("post_FORMAL_timing_rerun") is False,
        "9.15E timing rerun drift")
    req(es.get("retuning") is False, "9.15E retuning drift")

    rare = b["rare_event_support"]
    req(int(rare["critical_label_valid_denominator"]) == 24917,
        "critical denominator drift")
    req(int(rare["observed_Ccrit_GT_events"]) == 0,
        "critical-event count drift")
    req(rare["zero_observed_events_interpreted_as_zero_risk"] is False,
        "zero-event interpretation drift")

    # ---------------------------------------------------------------------------------------------
    # Final Stage 9.15 summary.
    # ---------------------------------------------------------------------------------------------
    summary = {
        "schema": "stage9_block915f_v2_final_stage915_summary_v1",
        "status": "STAGE9_BLOCK915_STATISTICS_ROBUSTNESS_FAILURE_CASES_LATENCY_COMPLETE",
        "block_sequence_role": "9.15_STATISTICS_ROBUSTNESS_FAILURE_CASES_LATENCY",
        "authoritative_upstream": {
            "stage913_final_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage914_v2_external_boundary_seal_sha256": EXPECTED[str(P914_SEAL)],
        },
        "subblocks": {
            "9.15A": {
                "role": "pre_inference_statistics_contract",
                "status": a["status"],
                "seal_sha256": EXPECTED[str(P915A_SEAL)],
            },
            "9.15B": {
                "role": "current_FORMAL_H1_H6_statistics_10k_cluster_bootstrap",
                "status": bs["status"],
                "seal_sha256": EXPECTED[str(P915B_SEAL)],
                "bootstrap_replicates_sha256": EXPECTED[str(P915B_REPS)],
            },
            "9.15C": {
                "role": "robustness",
                "status": cs["status"],
                "seal_sha256": EXPECTED[str(P915C_SEAL)],
            },
            "9.15D": {
                "role": "failure_case_analysis",
                "status": ds["status"],
                "seal_sha256": EXPECTED[str(P915D_SEAL)],
            },
            "9.15E": {
                "role": "latency_resources",
                "status": es["status"],
                "seal_sha256": EXPECTED[str(P915E_SEAL)],
            },
        },
        "statistics_H1_H6": {
            "H1": b["H1"]["status"],
            "H2": b["H2"]["status"],
            "H3": b["H3"]["status"],
            "H4": b["H4"]["status"],
            "H5": b["H5"]["status"],
            "H6": b["H6"]["status"],
            "bootstrap_replicates": int(b["statistics"]["replicates"]),
            "bootstrap_seed": int(b["statistics"]["seed"]),
            "RNG_engine": b["statistics"]["RNG_engine"],
            "critical_event_numerator": int(rare["observed_Ccrit_GT_events"]),
            "critical_event_denominator": int(rare["critical_label_valid_denominator"]),
            "critical_event_Wilson_95pct": rare["Wilson_95pct"],
            "zero_events_not_zero_risk": True,
        },
        "robustness": {
            "human_profile": c["human_profile_robustness"]["status"],
            "full_three_way_policy_ranking":
                c["human_profile_robustness"]["full_three_way_policy_ranking"],
            "calibration_perturbation": c["calibration_robustness"]["status"],
            "new_calibration_perturbation_executed": False,
        },
        "failure_cases": {
            "requested_slice_count":
                int(d["requested_failure_case_analysis"]["requested_slice_count"]),
            "evaluable_requested_slice_count":
                int(d["requested_failure_case_analysis"]["evaluable_requested_slice_count"]),
            "status": d["status"],
            "new_post_FORMAL_slice_thresholds": False,
            "new_post_FORMAL_membership_rules": False,
            "slice_substitution": False,
        },
        "latency_resources": {
            "C2_FORMAL_solver_latency": e["latency"]["C2_solver"]["status"],
            "C3_FORMAL_latency": e["latency"]["C3_recovery_ADB"]["status"],
            "end_to_end_FORMAL_latency": e["latency"]["end_to_end_stage9_latency"]["status"],
            "resource_statistics_complete": True,
            "CAL_runtime_used_as_FORMAL_substitute": False,
            "low_latency_claim_authorized": False,
            "speedup_claim_authorized": False,
            "realtime_claim_authorized": False,
        },
        "external_validation_boundary": {
            "stage914_v2_status": "FROZEN_COMPLETE",
            "DeepSense_role": "supporting_measured_mmWave_beam_policy_evidence_only",
            "DeepSense_independent_new_Stage9_confirmation_claim": False,
            "DeepSense_optical_headlamp_validation_claim": False,
            "DeepSense_glare_or_collision_validation_claim": False,
        },
        "integrity": {
            "FORMAL_population_changed": False,
            "H1_H6_hypotheses_changed": False,
            "policy_changed": False,
            "baseline_changed": False,
            "threshold_changed": False,
            "C3_reselected": False,
            "post_FORMAL_retuning": False,
            "zero_support_metric_substitution": False,
            "post_FORMAL_failure_slice_redefinition": False,
            "post_FORMAL_latency_rerun": False,
            "legacy_old_9_14_statistical_outputs_used_as_current_evidence": False,
            "new_scientific_computation_in_9_15F": False,
        },
        "limitations_carried_forward_to_9_16": [
            "H2 not evaluable because 0 observed critical events among 24917 valid critical labels",
            "H3 reliability component not evaluable because 0 observed critical events; resource component only",
            "H4 not evaluable because the critical-conditioned endpoint has zero observed critical-event support",
            "H5 conclusions are human-profile sensitive",
            "calibration under/over-confidence robustness not evaluable without a prefrozen numeric perturbation rule",
            "17 requested failure-case slices lack prefrozen exact membership authority in the final carrier",
            "FORMAL C2/C3/end-to-end latency claims are not supported by recorded timing carriers",
        ],
        "next_block": "9.16_PAPER_READY_REPORTING_NOVELTY_CLAIM_AUDIT_FINAL_SEAL",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(SUMMARY, summary)
    summary_sha = sha256_path(SUMMARY)

    seal = {
        "schema": "stage9_block915f_v2_final_stage915_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915_V2_STATISTICS_ROBUSTNESS_FAILURE_CASES_LATENCY",
        "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
        "stage914_v2_seal_sha256": EXPECTED[str(P914_SEAL)],
        "stage915a_seal_sha256": EXPECTED[str(P915A_SEAL)],
        "stage915b_seal_sha256": EXPECTED[str(P915B_SEAL)],
        "stage915c_seal_sha256": EXPECTED[str(P915C_SEAL)],
        "stage915d_seal_sha256": EXPECTED[str(P915D_SEAL)],
        "stage915e_seal_sha256": EXPECTED[str(P915E_SEAL)],
        "summary_sha256": summary_sha,
        "statistics_complete": True,
        "robustness_complete": True,
        "failure_case_disposition_complete": True,
        "latency_resource_disposition_complete": True,
        "new_outcome_scan": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "new_statistics": False,
        "new_timing_measurement": False,
        "scientific_reexecution": False,
        "post_FORMAL_retuning": False,
        "legacy_old_9_14_used_as_current_evidence": False,
        "next_block": "9.16_PAPER_READY_REPORTING_NOVELTY_CLAIM_AUDIT_FINAL_SEAL",
    }
    atomic_json(SEAL, seal)

    print("STAGE 9.15A = COMPLETE")
    print("STAGE 9.15B = COMPLETE")
    print("STAGE 9.15C = COMPLETE")
    print("STAGE 9.15D = COMPLETE")
    print("STAGE 9.15E = COMPLETE")
    print("NEW_OUTCOME_SCAN = FALSE")
    print("NEW_RNG = FALSE")
    print("NEW_BOOTSTRAP = FALSE")
    print("NEW_STATISTICS = FALSE")
    print("NEW_TIMING_MEASUREMENT = FALSE")
    print("SCIENTIFIC_REEXECUTION = FALSE")
    print("POST_FORMAL_RETUNING = FALSE")
    print("BLOCK_9_15 = FROZEN_COMPLETE")
    print("SUMMARY_SHA256 =", summary_sha)
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.16_PAPER_READY_REPORTING_NOVELTY_CLAIM_AUDIT_FINAL_SEAL")
    print("=" * 124)


if __name__ == "__main__":
    main()

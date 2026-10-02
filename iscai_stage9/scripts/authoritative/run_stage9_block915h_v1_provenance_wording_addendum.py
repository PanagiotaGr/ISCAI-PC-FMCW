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
# Current 9.15 primary + mandatory secondary sensitivity.
# -------------------------------------------------------------------------------------------------
B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

B915E = S9 / "artifacts/block915e_v2R1_latency_resources"
P915E_REPORT = B915E / "stage9_block915e_v2R1_latency_resources_report.json"
P915E_SEAL = B915E / "stage9_block915e_v2R1_latency_resources_seal.json"

B915F = S9 / "artifacts/block915f_v2_final_stage915_closure"
P915F_SUMMARY = B915F / "stage9_block915f_v2_final_stage915_summary.json"
P915F_SEAL = B915F / "stage9_block915f_v2_final_stage915_seal.json"

B915G = S9 / "artifacts/block915g_v1_technical_recovery_exclusion_sensitivity"
P915G_PREFLIGHT = B915G / "stage9_block915g_v1_prebootstrap_gate.json"
P915G_REPS = B915G / "stage9_block915g_v1_bootstrap_replicates.npz"
P915G_SUMMARY = B915G / "stage9_block915g_v1_technical_recovery_sensitivity_summary.json"
P915G_SEAL = B915G / "stage9_block915g_v1_technical_recovery_sensitivity_seal.json"

# -------------------------------------------------------------------------------------------------
# H1 preoutcome confirmatory boundary.
# -------------------------------------------------------------------------------------------------
BH1 = S9 / "artifacts/block913b2c_h1_formal_disposition_pre_gt"
P_H1_SEAL = BH1 / "stage9_block913b2c_h1_formal_disposition_seal.json"

# H4 preoutcome implementation-semantics correction.
BH4 = S9 / "artifacts/block913b3_R2_h4_preoutcome_authority_correction"
P_H4_REPORT = BH4 / "stage9_block913b3_R2_h4_preoutcome_authority_correction_report.json"
P_H4_SEAL = BH4 / "stage9_block913b3_R2_h4_preoutcome_authority_correction_seal.json"

# Block 9.4 current event-calibration authority.
B94A = S9 / "artifacts/block94a_evaluator_label_and_calibration_contract_freeze"
P94A_CONTRACT = B94A / "evaluator_label_and_calibration_contract_v1.json"

B94 = S9 / "artifacts/block94b2a_contract_self_hash_binding_repair"
P94_LABEL_SUMMARY = B94 / "cal_evaluator_label_summary.json"
P94_DIAGNOSTICS = B94 / "cal_fit_set_calibration_diagnostics.json"
P94_CALIBRATOR = B94 / "critical_event_monotone_logistic_calibrator.json"
P94_SEAL = B94 / "stage9_block94b2a_contract_self_hash_binding_repair_seal.json"

# C3 scope / component runtime authority.
P_C3_CONTRACT = S9 / "configs/stage9_block913_v1_2_formal_c1_c2_primary_c3_secondary_execution_contract.json"
BC3 = S9 / "artifacts/block913_v1_3_c3_phase3_DIRECT_FAST_SHARDED_R3_EXACT_OCC"
P_C3_PHASE3_SEAL = BC3 / "stage9_block913_v1_3_c3_phase3_DIRECT_seal.json"

# Old Stage-9.16 package that predates the current 9.15F/G chain.
B916_OLD = S9 / "artifacts/block916_R1_final_paper_ready_package"
P916_OLD_MANIFEST = B916_OLD / "stage9_block916_paper_ready_manifest.json"
P916_OLD_SEAL = B916_OLD / "stage9_block916_final_reproducibility_seal.json"

EXPECTED = {
    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",

    str(P915E_REPORT): "7e83fbd79c5c9f79c555dd44b32e490cd38d8d73f5a39ba7031271a163e5e496",
    str(P915E_SEAL): "c204eb243ac093b370e3e988be29217d82e7e779f078d0d7e8ce482fd9f8ab12",

    str(P915F_SUMMARY): "15f974474c9e455ee48c72ebb7c05d5be5c9b87a7691f6eda0b14ac9bb5bd169",
    str(P915F_SEAL): "02d4f36951254ca0f140b6682ee3d0716110e362c6c922d715b44ef3888bf065",

    str(P915G_PREFLIGHT): "96ac75802104b16cf2283ede7fe50c75218a410cbdf9955fcb3f4e4c210d8d67",
    str(P915G_REPS): "4545d6b52d5171986b994d8c19d225aaa1484f7300f3e76674e035c90a77d920",
    str(P915G_SUMMARY): "6d9c66876dc4eab5ea45c5932294a92370c7bede8bf68db8acde2a4bd0d20a63",
    str(P915G_SEAL): "d80053cf02413f558edd99df18ca444bd130f339b1780ece6183d0d31c32c291",

    str(P_H1_SEAL): "f86829c71ab5b410155e0bbeabe256bd280b17d7037576356bde9253d0306442",

    str(P_H4_REPORT): "6a844e0f8fbd6df94f792b5d9b8943e686a468f35a02d718d675c8ac87cfa306",
    str(P_H4_SEAL): "5128db258d5972c537fe5de4c3a1ebd834ca34c2a9c144fcd8753c3436fd9e12",

    str(P94A_CONTRACT): "2b893497dc792d033da7dac06817f89c70ec1edb390de6e264664b1a3ec79e43",
    str(P94_LABEL_SUMMARY): "969407a744ab2885ebae3205622baef6262b85463d03a3481d71eb33da9d017b",
    str(P94_DIAGNOSTICS): "35da53a736c5831c54b31bf579c3cda45710351098a41d59cb38a7deba985bd7",
    str(P94_CALIBRATOR): "0f5cc8fc7410545891401e1b0bfcb9ddd64493dbf910f4ebb1f1792dc7b15861",
    str(P94_SEAL): "34b71b31639c3ea2b9722bfa6c83e88fedc79f3636b4df809d8de6d172e5d82b",

    str(P_C3_CONTRACT): "991349ff9029b197652411e7ac60767458debf15674fcd103bfa56cb16288001",
    str(P_C3_PHASE3_SEAL): "33068597382dfe853b73b0180f44f74d586171fc0212f7e381774de0ff6ba0d3",

    str(P916_OLD_MANIFEST): "cc14e78c2f2f89ef283119f25a4dd016ce1f4d4f988819352c2557f2b993bad1",
    str(P916_OLD_SEAL): "d52cb626352c2d552c11cd482be9a9387e2000490005b1330eba3d9cfebf1a79",
}

OUT = S9 / "artifacts/block915h_v1_provenance_wording_addendum"
ADDENDUM = OUT / "stage9_block915h_v1_provenance_wording_addendum.json"
SEAL = OUT / "stage9_block915h_v1_provenance_wording_addendum_seal.json"


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


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15H-v1 — APPEND-ONLY PROVENANCE / WORDING ADDENDUM")
    print("NO OUTCOME SCAN | NO RNG | NO BOOTSTRAP | NO NEW STATISTICS | NO SCIENTIFIC REEXECUTION")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    b = load_json(P915B_SUMMARY)
    e = load_json(P915E_REPORT)
    f = load_json(P915F_SUMMARY)
    g = load_json(P915G_SUMMARY)
    h1 = load_json(P_H1_SEAL)
    h4r = load_json(P_H4_REPORT)
    h4s = load_json(P_H4_SEAL)
    c94 = load_json(P94A_CONTRACT)
    l94 = load_json(P94_LABEL_SUMMARY)
    d94 = load_json(P94_DIAGNOSTICS)
    cal94 = load_json(P94_CALIBRATOR)
    c3 = load_json(P_C3_CONTRACT)
    old916m = load_json(P916_OLD_MANIFEST)
    old916s = load_json(P916_OLD_SEAL)

    # ---------------------------------------------------------------------------------------------
    # H1 authority boundary.
    # ---------------------------------------------------------------------------------------------
    req(
        h1["H1"]["confirmatory_status"] == "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "H1 confirmatory authority drift",
    )
    req(h1["H1"]["posthoc_FORMAL_raw_arm_allowed"] is False, "H1 raw-arm prohibition drift")
    req(h1["H1"]["new_model_forward_allowed"] is False, "H1 model-forward prohibition drift")
    req(
        b["H1"]["status"] == "SUPPORTED_CALIBRATION_REDUCES_MACRO_ABS_COVERAGE_ERROR",
        "9.15B numerical H1 status drift",
    )
    req(
        g["H1"]["confirmatory_status"] == "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "9.15G H1 boundary drift",
    )
    req(g["H1"]["confirmatory_reactivation"] is False, "9.15G H1 reactivation drift")

    # ---------------------------------------------------------------------------------------------
    # H4 exact preoutcome implementation authority.
    # ---------------------------------------------------------------------------------------------
    req(h4r["H4"]["FORMAL_outcomes_used_for_correction"] is False, "H4 correction used FORMAL outcomes")
    req(h4r["H4"]["comparator_selected_from_FORMAL"] is False, "H4 comparator FORMAL-selected")
    req(
        h4r["H4"]["comparator_semantics"]
        == "CAL_FROZEN_CLOSEST_MEAN_K_UNCERTAINTY_ONLY_FIXED_Q_TOPK",
        "H4 comparator semantics drift",
    )
    req(h4r["H4"]["comparator"] == "fixed_q_0.95", "H4 exact comparator drift")
    req(h4s["H4"]["baseline"] == "fixed_q_0.95", "H4 seal baseline drift")
    req(
        b["H4"]["status"] == "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "current H4 disposition drift",
    )

    # ---------------------------------------------------------------------------------------------
    # Block 9.4 event-calibration reporting authority.
    # ---------------------------------------------------------------------------------------------
    req(c94["diagnostics"]["ECE_bins"] == 15, "Block9.4 current ECE-bin count drift")
    req(c94["diagnostics"]["ECE_type"] == "fixed equal-width [0,1]", "Block9.4 ECE type drift")
    req(c94["calibrator"]["FORMAL_used_for_fit"] is False, "Block9.4 FORMAL-used-for-fit drift")
    req(l94["status"] == "CAL_EVALUATOR_LABELS_COMPLETE", "Block9.4 label summary status drift")
    req(d94["status"] == "CAL_FIT_SET_DIAGNOSTICS", "Block9.4 diagnostics status drift")
    req(cal94["status"] == "FROZEN_MONOTONE_LOGISTIC_EVENT_CALIBRATOR", "Block9.4 calibrator drift")

    # ---------------------------------------------------------------------------------------------
    # C3 scope / latency wording.
    # ---------------------------------------------------------------------------------------------
    req(c3["scope"]["primary_confirmatory"] == "C1+C2", "C3 scope primary drift")
    req(c3["scope"]["secondary"] == "C3_PROFILE_SENSITIVITY", "C3 secondary scope drift")
    req(c3["scope"]["C3_global_success_claim"] is False, "C3 global-success scope drift")
    req(c3["prohibitions"]["FAST_exclusion"] is True, "C3 FAST prohibition drift")
    req(c3["prohibitions"]["policy_retuning"] is True, "C3 retuning prohibition drift")

    c3_latency = e["latency"]["C3_recovery_ADB"]
    req(
        c3_latency["status"] == "NOT_EVALUABLE_NO_FORMAL_C3_RUNTIME_CARRIER",
        "9.15E stored C3 latency status drift",
    )
    req(c3_latency["FORMAL_system_latency_claim_authorized"] is False,
        "9.15E system-latency claim boundary drift")

    # ---------------------------------------------------------------------------------------------
    # 9.15G is supplemental, not a replacement for 9.15F.
    # ---------------------------------------------------------------------------------------------
    req(f["status"] == "STAGE9_BLOCK915_STATISTICS_ROBUSTNESS_FAILURE_CASES_LATENCY_COMPLETE",
        "9.15F status drift")
    req(g["primary_full_cohort_result_remains_authoritative"] is True,
        "9.15G primary-authority drift")
    req(g["may_override_primary_full_cohort_result"] is False,
        "9.15G override boundary drift")
    req(g["population"]["excluded_documented_reread_scene_count"] == 22,
        "9.15G excluded-scene count drift")
    req(g["population"]["secondary_reduced_scene_count"] == 26187,
        "9.15G reduced population drift")
    req(g["statistics"]["new_RNG_draws_authorized_before_primary_inference"] is True,
        "9.15G prefrozen-RNG authority drift")

    # ---------------------------------------------------------------------------------------------
    # The pre-existing 9.16 package is an older chain and cannot be the new current package.
    # ---------------------------------------------------------------------------------------------
    req(
        old916s["status"] == "FROZEN_COMPLETE_STAGE9_916_C1_C2_CORE_WITH_DISCLOSED_PACKAGE_GAPS",
        "old 9.16 seal status drift",
    )
    req(old916s["final_dispositions"]["H1"] == "NOT_TESTED", "old 9.16 H1 status drift")
    old_manifest_text = P916_OLD_MANIFEST.read_text(encoding="utf-8")
    old_seal_text = P916_OLD_SEAL.read_text(encoding="utf-8")
    req(EXPECTED[str(P915F_SEAL)] not in old_manifest_text, "old 9.16 unexpectedly binds current 9.15F seal")
    req(EXPECTED[str(P915F_SEAL)] not in old_seal_text, "old 9.16 seal unexpectedly binds current 9.15F seal")
    req(EXPECTED[str(P915G_SEAL)] not in old_manifest_text, "old 9.16 unexpectedly binds current 9.15G seal")
    req(EXPECTED[str(P915G_SEAL)] not in old_seal_text, "old 9.16 seal unexpectedly binds current 9.15G seal")

    addendum = {
        "schema": "stage9_block915h_v1_provenance_wording_addendum_v1",
        "status": "STAGE9_BLOCK915H_PROVENANCE_WORDING_ADDENDUM_COMPLETE",
        "role": "APPEND_ONLY_REPORTING_AND_PROVENANCE_CORRECTION_WITHOUT_SCIENTIFIC_REEXECUTION",
        "authority": {
            "stage915b_primary_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
            "stage915e_latency_resources_report_sha256": EXPECTED[str(P915E_REPORT)],
            "stage915f_primary_closure_summary_sha256": EXPECTED[str(P915F_SUMMARY)],
            "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
            "stage915g_secondary_sensitivity_summary_sha256": EXPECTED[str(P915G_SUMMARY)],
            "stage915g_secondary_sensitivity_seal_sha256": EXPECTED[str(P915G_SEAL)],
            "H1_preoutcome_not_tested_seal_sha256": EXPECTED[str(P_H1_SEAL)],
            "H4_preoutcome_authority_correction_report_sha256": EXPECTED[str(P_H4_REPORT)],
            "H4_preoutcome_authority_correction_seal_sha256": EXPECTED[str(P_H4_SEAL)],
            "block94_contract_sha256": EXPECTED[str(P94A_CONTRACT)],
            "block94_cal_label_summary_sha256": EXPECTED[str(P94_LABEL_SUMMARY)],
            "block94_cal_diagnostics_sha256": EXPECTED[str(P94_DIAGNOSTICS)],
            "block94_calibrator_sha256": EXPECTED[str(P94_CALIBRATOR)],
            "block94_seal_sha256": EXPECTED[str(P94_SEAL)],
            "C3_secondary_execution_contract_sha256": EXPECTED[str(P_C3_CONTRACT)],
            "C3_phase3_preoutcome_action_seal_sha256": EXPECTED[str(P_C3_PHASE3_SEAL)],
            "old_block916_manifest_sha256": EXPECTED[str(P916_OLD_MANIFEST)],
            "old_block916_seal_sha256": EXPECTED[str(P916_OLD_SEAL)],
        },
        "H1_reporting_correction": {
            "confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
            "confirmatory_scientific_result": "NOT_TESTED",
            "posthoc_confirmatory_reactivation_allowed": False,
            "current_9_15B_numerical_status": b["H1"]["status"],
            "current_9_15B_numerical_result_role":
                "NONCONFIRMATORY_POST_PREOUTCOME_AUTHORITY_NUMERICAL_EVIDENCE",
            "stage915g_numerical_sensitivity_role":
                "SECONDARY_NONCONFIRMATORY_NUMERICAL_SENSITIVITY_ONLY",
            "stage915g_numerical_sensitivity_status":
                g["H1"]["numerical_sensitivity_status"],
            "paper_wording_boundary":
                "Do not state that confirmatory H1 was supported. State that H1 was not tested "
                "confirmatorily under the frozen preoutcome authority; later materialized RAW-vs-CAL "
                "records show a numerical reduction in macro absolute coverage error, including the "
                "22-scene-exclusion sensitivity, and must be labeled non-confirmatory.",
        },
        "H4_reporting_correction": {
            "hypothesis_comparator_family": "uncertainty_only_adaptive_topk",
            "exact_preoutcome_executable_comparator": "fixed_q_0.95",
            "exact_preoutcome_comparator_semantics":
                "CAL_FROZEN_CLOSEST_MEAN_K_UNCERTAINTY_ONLY_FIXED_Q_TOPK",
            "correction_class": h4r["correction_class"],
            "FORMAL_outcomes_used_for_correction": False,
            "comparator_selected_from_FORMAL": False,
            "current_scientific_disposition": b["H4"]["status"],
            "paper_wording_boundary":
                "Describe fixed_q_0.95 as the preoutcome CAL-frozen executable implementation of "
                "the matched uncertainty-only comparator, not as a comparator selected after FORMAL. "
                "H4 remains not evaluable because the critical-conditioned endpoint has zero observed "
                "critical-event support.",
        },
        "block94_event_calibration_reporting": {
            "current_contract_ECE_bins": int(c94["diagnostics"]["ECE_bins"]),
            "current_contract_ECE_type": c94["diagnostics"]["ECE_type"],
            "FORMAL_used_for_calibrator_fit": False,
            "CAL_valid_label_prevalence": float(l94["valid_label_prevalence"]),
            "CAL_raw_Brier": float(d94["raw"]["Brier"]),
            "CAL_calibrated_Brier": float(d94["calibrated"]["Brier"]),
            "CAL_raw_ECE15_equal_width": float(d94["raw"]["ECE15_equal_width"]),
            "CAL_calibrated_ECE15_equal_width": float(d94["calibrated"]["ECE15_equal_width"]),
            "legacy_ECE20_is_current_reporting_authority": False,
            "paper_wording_boundary":
                "Report the repaired/current Block-9.4 15-bin equal-width event-calibration diagnostics. "
                "Do not substitute the older legacy ECE20 calibration artifact.",
        },
        "C3_scope_and_latency_wording": {
            "primary_confirmatory_scope": "C1+C2",
            "C3_role": "C3_PROFILE_SENSITIVITY",
            "C3_global_success_claim_allowed": False,
            "stored_9_15E_C3_latency_status": c3_latency["status"],
            "preoutcome_action_materialization_runtime_component_carrier_exists": True,
            "preoutcome_action_materialization_runtime_component_scene_count": 26209,
            "validated_FORMAL_system_or_end_to_end_C3_latency_carrier_exists": False,
            "low_latency_claim_authorized": False,
            "realtime_claim_authorized": False,
            "speedup_claim_authorized": False,
            "paper_wording_boundary":
                "Interpret the 9.15E C3 latency status narrowly: no validated FORMAL C3 system/end-to-end "
                "latency carrier exists. Preoutcome per-scene action-materialization runtime_s values exist "
                "as component provenance for all 26,209 scenes, but they do not establish system latency, "
                "real-time operation, or speedup.",
        },
        "technical_recovery_sensitivity_reporting": {
            "primary_full_cohort_scene_count": 26209,
            "primary_full_cohort_remains_authoritative": True,
            "mandatory_secondary_excluded_scene_count": 22,
            "mandatory_secondary_scene_count": 26187,
            "new_RNG_draws_executed": True,
            "new_RNG_draws_prefrozen_authorized": True,
            "primary_result_override_allowed": False,
            "paper_wording_boundary":
                "Report the exact 22-scene exclusion as a mandatory prefrozen secondary technical-recovery "
                "sensitivity. It supplements and cannot replace the full-cohort primary result.",
        },
        "old_stage916_package_disposition": {
            "old_package_status": old916s["status"],
            "old_package_manifest_sha256": EXPECTED[str(P916_OLD_MANIFEST)],
            "old_package_seal_sha256": EXPECTED[str(P916_OLD_SEAL)],
            "binds_current_stage915f_seal": False,
            "binds_current_stage915g_seal": False,
            "current_reporting_authority": False,
            "disposition": "LEGACY_SUPERSEDED_FOR_NEW_STAGE916_PACKAGE",
            "paper_package_rule":
                "Do not write the manuscript from the old block916_R1 package. A new Stage-9.16 package "
                "must bind the current 9.15F primary closure, 9.15G mandatory secondary sensitivity, and "
                "this 9.15H provenance/wording addendum.",
        },
        "integrity": {
            "new_outcome_scan": False,
            "new_RNG": False,
            "new_bootstrap": False,
            "new_statistics": False,
            "model_rerun": False,
            "policy_rerun": False,
            "solver_rerun": False,
            "retuning": False,
            "threshold_change": False,
            "baseline_change": False,
            "primary_scientific_result_changed": False,
            "append_only_reporting_correction": True,
        },
        "next_block": "9.16_NEW_PAPER_READY_REPORTING_NOVELTY_CLAIM_AUDIT_FINAL_SEAL",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(ADDENDUM, addendum)
    addendum_sha = sha256_path(ADDENDUM)

    seal = {
        "schema": "stage9_block915h_v1_provenance_wording_addendum_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915H_V1_PROVENANCE_WORDING_ADDENDUM",
        "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
        "stage915g_secondary_sensitivity_seal_sha256": EXPECTED[str(P915G_SEAL)],
        "addendum_sha256": addendum_sha,
        "H1_confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "H1_confirmatory_reactivation": False,
        "H4_current_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "block94_current_ECE_bin_count": 15,
        "C3_system_end_to_end_latency_claim_authorized": False,
        "old_block916_R1_current_reporting_authority": False,
        "new_outcome_scan": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "new_statistics": False,
        "scientific_reexecution": False,
        "retuning": False,
        "next_block": "9.16_NEW_PAPER_READY_REPORTING_NOVELTY_CLAIM_AUDIT_FINAL_SEAL",
    }
    atomic_json(SEAL, seal)

    print("H1_CONFIRMATORY_STATUS = NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM")
    print("H1_915B_NUMERICAL_EVIDENCE_ROLE = NONCONFIRMATORY")
    print("H1_915G_SENSITIVITY_ROLE = SECONDARY_NONCONFIRMATORY")
    print("H4_PREOUTCOME_EXECUTABLE_COMPARATOR = fixed_q_0.95")
    print("H4_COMPARATOR_SEMANTICS = CAL_FROZEN_CLOSEST_MEAN_K_UNCERTAINTY_ONLY_FIXED_Q_TOPK")
    print("BLOCK94_CURRENT_ECE_BINS = 15")
    print("BLOCK94_RAW_ECE15 =", d94["raw"]["ECE15_equal_width"])
    print("BLOCK94_CALIBRATED_ECE15 =", d94["calibrated"]["ECE15_equal_width"])
    print("C3_COMPONENT_RUNTIME_CARRIER_EXISTS = TRUE")
    print("C3_VALIDATED_SYSTEM_END_TO_END_LATENCY_CARRIER_EXISTS = FALSE")
    print("OLD_BLOCK916_R1_CURRENT_AUTHORITY = FALSE")
    print("NEW_OUTCOME_SCAN = FALSE")
    print("NEW_RNG = FALSE")
    print("NEW_BOOTSTRAP = FALSE")
    print("NEW_STATISTICS = FALSE")
    print("SCIENTIFIC_REEXECUTION = FALSE")
    print("STAGE 9.15H-v1 = COMPLETE")
    print("ADDENDUM_SHA256 =", addendum_sha)
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.16_NEW_PAPER_READY_REPORTING_NOVELTY_CLAIM_AUDIT_FINAL_SEAL")
    print("=" * 124)


if __name__ == "__main__":
    main()

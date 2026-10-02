#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

# -------------------------------------------------------------------------------------------------
# Current authoritative chain.
# -------------------------------------------------------------------------------------------------
P913_SEAL = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3/stage9_block913_v1_3_FINAL_H1_H6_seal.json"
P914_SEAL = S9 / "artifacts/block914_v2_external_deepsense_boundary/stage9_block914_v2_external_boundary_seal.json"

B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_EVIDENCE = B915A / "stage9_block915a_v2R4_h1_schema_status_censoring_evidence.json"
P915A_EST = B915A / "stage9_block915a_v2R4_estimand_registry.json"
P915A_CONTRACT = B915A / "stage9_block915a_v2R4_statistics_contract.json"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

B915C = S9 / "artifacts/block915c_v2_robustness"
P915C_REPORT = B915C / "stage9_block915c_v2_robustness_report.json"
P915C_SEAL = B915C / "stage9_block915c_v2_robustness_seal.json"

B915D = S9 / "artifacts/block915d_v2R1_failure_case_analysis"
P915D_REPORT = B915D / "stage9_block915d_v2R1_failure_case_report.json"
P915D_SEAL = B915D / "stage9_block915d_v2R1_failure_case_seal.json"

B915E = S9 / "artifacts/block915e_v2R1_latency_resources"
P915E_REPORT = B915E / "stage9_block915e_v2R1_latency_resources_report.json"
P915E_SEAL = B915E / "stage9_block915e_v2R1_latency_resources_seal.json"

B915F = S9 / "artifacts/block915f_v2_final_stage915_closure"
P915F_SUMMARY = B915F / "stage9_block915f_v2_final_stage915_summary.json"
P915F_SEAL = B915F / "stage9_block915f_v2_final_stage915_seal.json"

B915G = S9 / "artifacts/block915g_v1_technical_recovery_exclusion_sensitivity"
P915G_SUMMARY = B915G / "stage9_block915g_v1_technical_recovery_sensitivity_summary.json"
P915G_SEAL = B915G / "stage9_block915g_v1_technical_recovery_sensitivity_seal.json"

B915H = S9 / "artifacts/block915h_v1_provenance_wording_addendum"
P915H_ADDENDUM = B915H / "stage9_block915h_v1_provenance_wording_addendum.json"
P915H_SEAL = B915H / "stage9_block915h_v1_provenance_wording_addendum_seal.json"

# Scope / methods authorities.
P_C3_CONTRACT = S9 / "configs/stage9_block913_v1_2_formal_c1_c2_primary_c3_secondary_execution_contract.json"
P94_CONTRACT = S9 / "artifacts/block94a_evaluator_label_and_calibration_contract_freeze/evaluator_label_and_calibration_contract_v1.json"
P94_DIAGNOSTICS = S9 / "artifacts/block94b2a_contract_self_hash_binding_repair/cal_fit_set_calibration_diagnostics.json"
P94_SEAL = S9 / "artifacts/block94b2a_contract_self_hash_binding_repair/stage9_block94b2a_contract_self_hash_binding_repair_seal.json"

# Legacy package explicitly superseded for current reporting.
B916_OLD = S9 / "artifacts/block916_R1_final_paper_ready_package"
P916_OLD_MANIFEST = B916_OLD / "stage9_block916_paper_ready_manifest.json"
P916_OLD_SEAL = B916_OLD / "stage9_block916_final_reproducibility_seal.json"

EXPECTED = {
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(P914_SEAL): "9d9ec48e9c75307861a5b895e1b26365328abe748e43cca5598021ea70031ddd",

    str(P915A_EVIDENCE): "c96b9f969c5e6d1f1e760313d5c58d279746044fb3846cba63c67a1a7bc785a0",
    str(P915A_EST): "4ecc05f12b5c11e46a976b161fd7866d30bda76f287d4904d7cd52e95a71e37c",
    str(P915A_CONTRACT): "91362b13413565501d3d33905d78bc12f14bb9c8b3739783e899cbc6088c7b03",
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",

    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",

    str(P915C_REPORT): "a42be7cc37e5c891d4c26a0ac6ceea472074fca915bdd36d33896b15b472cb6c",
    str(P915C_SEAL): "a8fa3276e29e7093dd0e8d1131c8fd4708500d3b9eef7bd17a7740885c962a5b",

    str(P915D_REPORT): "937e211b00ba4966ddb85935f0fc07f047ee6c4aacaa8fe56b141a551a6fc297",
    str(P915D_SEAL): "465e1ab9a8fc06190780121708628531953d2576d02a86d22669714e8cc4f1ca",

    str(P915E_REPORT): "7e83fbd79c5c9f79c555dd44b32e490cd38d8d73f5a39ba7031271a163e5e496",
    str(P915E_SEAL): "c204eb243ac093b370e3e988be29217d82e7e779f078d0d7e8ce482fd9f8ab12",

    str(P915F_SUMMARY): "15f974474c9e455ee48c72ebb7c05d5be5c9b87a7691f6eda0b14ac9bb5bd169",
    str(P915F_SEAL): "02d4f36951254ca0f140b6682ee3d0716110e362c6c922d715b44ef3888bf065",

    str(P915G_SUMMARY): "6d9c66876dc4eab5ea45c5932294a92370c7bede8bf68db8acde2a4bd0d20a63",
    str(P915G_SEAL): "d80053cf02413f558edd99df18ca444bd130f339b1780ece6183d0d31c32c291",

    str(P915H_ADDENDUM): "de0a1c6f117f4178fa285db7532c53899642f3674f19348f99214782d15f0b2d",
    str(P915H_SEAL): "21c39a416602252fe2fdbda1e89b31f13cb17966bcdb416b9643b6460706bc6a",

    str(P_C3_CONTRACT): "991349ff9029b197652411e7ac60767458debf15674fcd103bfa56cb16288001",
    str(P94_CONTRACT): "2b893497dc792d033da7dac06817f89c70ec1edb390de6e264664b1a3ec79e43",
    str(P94_DIAGNOSTICS): "35da53a736c5831c54b31bf579c3cda45710351098a41d59cb38a7deba985bd7",
    str(P94_SEAL): "34b71b31639c3ea2b9722bfa6c83e88fedc79f3636b4df809d8de6d172e5d82b",

    str(P916_OLD_MANIFEST): "cc14e78c2f2f89ef283119f25a4dd016ce1f4d4f988819352c2557f2b993bad1",
    str(P916_OLD_SEAL): "d52cb626352c2d552c11cd482be9a9387e2000490005b1330eba3d9cfebf1a79",
}

OUT = S9 / "artifacts/block916_v2_new_paper_ready_package"

METHODS = OUT / "stage9_block916_v2_methods_problem_statement.json"
PROVENANCE = OUT / "stage9_block916_v2_dataset_provenance.json"
CALIBRATION = OUT / "stage9_block916_v2_calibration_tables.json"
RESULTS = OUT / "stage9_block916_v2_paper_ready_results.json"
ROBUSTNESS = OUT / "stage9_block916_v2_robustness_failure_latency.json"
CLAIMS = OUT / "stage9_block916_v2_claim_evidence_matrix.json"
LIMITATIONS = OUT / "stage9_block916_v2_limitations.json"
SUPERSESSION = OUT / "stage9_block916_v2_supersession_map.json"
SOURCE_MANIFEST = OUT / "stage9_block916_v2_source_config_manifest.json"

CSV_H3 = OUT / "stage9_block916_v2_table_H3_resource.csv"
CSV_H5 = OUT / "stage9_block916_v2_table_H5_profile_sensitivity.csv"
CSV_H6 = OUT / "stage9_block916_v2_table_H6_side_effects.csv"
CSV_CAL = OUT / "stage9_block916_v2_table_calibration.csv"

SUMMARY_MD = OUT / "stage9_block916_v2_paper_ready_summary.md"
MANIFEST = OUT / "stage9_block916_v2_paper_ready_manifest.json"
SEAL = OUT / "stage9_block916_v2_final_reproducibility_seal.json"


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
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def atomic_bytes(path: Path, data: bytes) -> None:
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


def atomic_json(path: Path, obj: Any) -> None:
    atomic_bytes(path, canonical_bytes(obj))


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    import io
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    atomic_text(path, s.getvalue())


def hash_gate() -> None:
    for raw, wanted in EXPECTED.items():
        p = Path(raw)
        req(p.is_file(), f"missing authoritative artifact: {p}")
        got = sha256_path(p)
        req(got == wanted, f"SHA256 drift {p}: {got} != {wanted}")
        print("EXACT PASS", p)


def find_status_counts(x: Any) -> dict[str, int] | None:
    target = {
        "EVALUATED_FUTURE_GT",
        "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID",
        "NOT_EVALUATED_NO_ELIGIBLE_RECEIVER",
    }
    if isinstance(x, dict):
        if target.issubset(set(x.keys())):
            vals = {k: x[k] for k in target}
            if all(isinstance(v, int) for v in vals.values()):
                return vals
        for v in x.values():
            z = find_status_counts(v)
            if z is not None:
                return z
    elif isinstance(x, list):
        for v in x:
            z = find_status_counts(v)
            if z is not None:
                return z
    return None


def main() -> None:
    print("=" * 124)
    print("STAGE 9.16-v2 — NEW PAPER-READY REPORTING + NOVELTY/CLAIM AUDIT + FINAL SEAL")
    print("SEALED ARTIFACT SYNTHESIS ONLY | NO OUTCOME SCAN | NO RNG | NO NEW STATISTICS | NO SCIENTIFIC REEXECUTION")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    aev = load_json(P915A_EVIDENCE)
    b = load_json(P915B_SUMMARY)
    c = load_json(P915C_REPORT)
    d = load_json(P915D_REPORT)
    e = load_json(P915E_REPORT)
    f = load_json(P915F_SUMMARY)
    g = load_json(P915G_SUMMARY)
    h = load_json(P915H_ADDENDUM)
    c3 = load_json(P_C3_CONTRACT)
    c94 = load_json(P94_CONTRACT)
    d94 = load_json(P94_DIAGNOSTICS)

    # -------------------------------------------------------------------------------------------------
    # Current reporting truth gates.
    # -------------------------------------------------------------------------------------------------
    req(h["H1_reporting_correction"]["confirmatory_status"]
        == "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM", "H1 reporting boundary drift")
    req(h["H1_reporting_correction"]["posthoc_confirmatory_reactivation_allowed"] is False,
        "H1 confirmatory reactivation drift")
    req(h["H4_reporting_correction"]["current_scientific_disposition"]
        == "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS", "H4 disposition drift")
    req(h["block94_event_calibration_reporting"]["current_contract_ECE_bins"] == 15,
        "Block9.4 ECE authority drift")
    req(h["C3_scope_and_latency_wording"]["validated_FORMAL_system_or_end_to_end_C3_latency_carrier_exists"] is False,
        "C3 latency boundary drift")
    req(h["old_stage916_package_disposition"]["current_reporting_authority"] is False,
        "old 9.16 supersession drift")
    req(g["primary_full_cohort_result_remains_authoritative"] is True,
        "9.15G primary-authority boundary drift")
    req(g["may_override_primary_full_cohort_result"] is False,
        "9.15G override boundary drift")
    req(c3["scope"]["primary_confirmatory"] == "C1+C2", "C3 scope drift")
    req(c3["scope"]["secondary"] == "C3_PROFILE_SENSITIVITY", "C3 secondary scope drift")
    req(c3["scope"]["C3_global_success_claim"] is False, "C3 global-success boundary drift")

    status_counts = find_status_counts(aev)
    req(status_counts is not None, "could not locate authoritative primary status counts in 9.15A evidence")
    req(sum(status_counts.values()) == 26209, "FORMAL status count total drift")
    req(status_counts["EVALUATED_FUTURE_GT"] == 24917, "FORMAL evaluated count drift")
    req(status_counts["EVALUATED_FUTURE_GT_ACTOR_1S_INVALID"] == 322, "FORMAL actor-invalid count drift")
    req(status_counts["NOT_EVALUATED_NO_ELIGIBLE_RECEIVER"] == 970, "FORMAL no-receiver count drift")

    rare = b["rare_event_support"]
    req(rare["observed_Ccrit_GT_events"] == 0, "critical event numerator drift")
    req(rare["critical_label_valid_denominator"] == 24917, "critical event denominator drift")
    req(rare["zero_observed_events_interpreted_as_zero_risk"] is False, "zero-risk interpretation drift")

    # -------------------------------------------------------------------------------------------------
    # Frozen methods specification / mathematical problem statement.
    # -------------------------------------------------------------------------------------------------
    methods = {
        "schema": "stage9_block916_v2_methods_problem_statement_v1",
        "status": "PAPER_READY_METHODS_AND_PROBLEM_STATEMENT_NO_NEW_METHOD",
        "frozen_scope": {
            "primary_confirmatory": "C1+C2",
            "secondary": "C3_PROFILE_SENSITIVITY",
            "C3_policy": c3["scope"]["C3_policy"],
            "rho_dim_per_s": c3["scope"]["rho_dim_per_s"],
            "rho_bright_per_s": c3["scope"]["rho_bright_per_s"],
            "FAST_exclusion_forbidden": c3["prohibitions"]["FAST_exclusion"],
            "policy_retuning_forbidden": c3["prohibitions"]["policy_retuning"],
            "threshold_retuning_forbidden": c3["prohibitions"]["threshold_retuning"],
        },
        "problem_statement": {
            "shared_future_state": (
                "Use the frozen sensing-informed future-motion posterior as the common future-state carrier "
                "for beam uncertainty and traffic-criticality reasoning."
            ),
            "communication_C2": (
                "Select a minimum-overhead beam-probing action under the frozen criticality/reliability "
                "constraints and compare against the frozen communication baselines."
            ),
            "ADB_C3": (
                "Evaluate the frozen recovery-aware predictive ADB policy as a secondary human-profile "
                "sensitivity extension with modeled glare/reaction/recovery effects."
            ),
            "critical_event_label": {
                "name": c94["label"]["name"],
                "horizon_s": c94["label"]["horizon_s"],
                "clearance_m": c94["label"]["clearance_m"],
                "positive_rule": c94["label"]["positive_rule"],
                "not_crash_probability": c94["label"]["not_crash_probability"],
                "future_GT_role": "EVALUATOR_ONLY",
            },
        },
        "novelty_boundary": {
            "new_collision_predictor_claim": False,
            "new_uncertainty_calibration_method_claim": False,
            "population_level_crash_reduction_claim": False,
            "universal_glare_recovery_time_claim": False,
            "low_latency_or_realtime_claim": False,
            "safe_method_story": (
                "The contribution is the frozen shared future-state interface and safety-consequence-aware "
                "decision layer: criticality-constrained communication plus a secondary human-factors-informed "
                "recovery-aware ADB extension, not another predictor."
            ),
        },
        "authority": {
            "C3_execution_contract_sha256": EXPECTED[str(P_C3_CONTRACT)],
            "block94_contract_sha256": EXPECTED[str(P94_CONTRACT)],
            "stage915h_addendum_sha256": EXPECTED[str(P915H_ADDENDUM)],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(METHODS, methods)

    # -------------------------------------------------------------------------------------------------
    # Dataset split / provenance.
    # -------------------------------------------------------------------------------------------------
    provenance = {
        "schema": "stage9_block916_v2_dataset_provenance_v1",
        "status": "PAPER_READY_DATASET_SPLIT_AND_PROVENANCE",
        "WOMD_FORMAL": {
            "frozen_order_scene_count": 26209,
            "primary_status_counts": dict(sorted(status_counts.items())),
            "critical_label_valid_denominator": 24917,
            "observed_critical_events": 0,
            "critical_event_Wilson_95pct": rare["Wilson_95pct"],
            "zero_events_not_zero_risk": True,
            "primary_full_cohort_remains_authoritative": True,
        },
        "technical_recovery": {
            "documented_reread_scene_count": 22,
            "extra_physical_future_GT_read_events": 24,
            "mandatory_secondary_reduced_scene_count": 26187,
            "secondary_sensitivity_may_override_primary": False,
            "stage915g_seal_sha256": EXPECTED[str(P915G_SEAL)],
        },
        "external_DeepSense": f["external_validation_boundary"],
        "DeepSense_claim_boundary": (
            "DeepSense supports communication-policy evidence only; it does not validate the optical headlamp, "
            "glare, collision, or C3 human-safety claims."
        ),
        "authority": {
            "stage913_final_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage914_v2_external_boundary_seal_sha256": EXPECTED[str(P914_SEAL)],
            "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
            "stage915g_secondary_sensitivity_seal_sha256": EXPECTED[str(P915G_SEAL)],
        },
    }
    atomic_json(PROVENANCE, provenance)

    # -------------------------------------------------------------------------------------------------
    # Calibration tables.
    # -------------------------------------------------------------------------------------------------
    calibration = {
        "schema": "stage9_block916_v2_calibration_tables_v1",
        "status": "PAPER_READY_CALIBRATION_TABLES",
        "critical_event_calibration_CAL": {
            "ECE_bins": h["block94_event_calibration_reporting"]["current_contract_ECE_bins"],
            "ECE_type": "fixed equal-width [0,1]",
            "raw_Brier": h["block94_event_calibration_reporting"]["CAL_raw_Brier"],
            "calibrated_Brier": h["block94_event_calibration_reporting"]["CAL_calibrated_Brier"],
            "raw_ECE15_equal_width": h["block94_event_calibration_reporting"]["CAL_raw_ECE15_equal_width"],
            "calibrated_ECE15_equal_width": h["block94_event_calibration_reporting"]["CAL_calibrated_ECE15_equal_width"],
            "valid_label_prevalence": h["block94_event_calibration_reporting"]["CAL_valid_label_prevalence"],
            "FORMAL_used_for_fit": False,
            "legacy_ECE20_is_current_reporting_authority": False,
        },
        "H1_numerical_coverage_evidence": {
            "confirmatory_status": h["H1_reporting_correction"]["confirmatory_status"],
            "reporting_role": h["H1_reporting_correction"]["current_9_15B_numerical_result_role"],
            "macro_abs_coverage_error_RAW": b["H1"]["point_estimates"]["macro_abs_coverage_error_RAW"],
            "macro_abs_coverage_error_CALIBRATED": b["H1"]["point_estimates"]["macro_abs_coverage_error_CALIBRATED"],
            "delta_CAL_minus_RAW": b["H1"]["point_estimates"]["primary_delta_CALIBRATED_minus_RAW"],
            "paired_bootstrap_CI95": b["H1"]["primary_delta_CI95"],
            "absolute_calibration_interpretation": (
                "The numerical error reduction is supported, but absolute requested-vs-empirical coverage "
                "error remains large; do not describe the posterior as demonstrated well-calibrated on FORMAL."
            ),
            "technical_recovery_sensitivity": {
                "role": g["H1"]["reporting_role"],
                "numerical_status": g["H1"]["numerical_sensitivity_status"],
                "delta_CAL_minus_RAW": g["H1"]["point_estimates"]["primary_delta_CALIBRATED_minus_RAW"],
                "paired_bootstrap_CI95": g["H1"]["primary_delta_CI95"],
            },
        },
        "authority": {
            "block94_diagnostics_sha256": EXPECTED[str(P94_DIAGNOSTICS)],
            "stage915b_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
            "stage915h_addendum_sha256": EXPECTED[str(P915H_ADDENDUM)],
        },
    }
    atomic_json(CALIBRATION, calibration)

    # -------------------------------------------------------------------------------------------------
    # Paper-ready result tables: no new computation.
    # -------------------------------------------------------------------------------------------------
    results = {
        "schema": "stage9_block916_v2_paper_ready_results_v1",
        "status": "PAPER_READY_RESULTS_FROM_SEALED_STAGE9_ARTIFACTS_NO_NEW_ANALYSIS",
        "hypothesis_dispositions": {
            "H1_confirmatory": h["H1_reporting_correction"]["confirmatory_status"],
            "H1_numerical_evidence": b["H1"]["status"],
            "H2": b["H2"]["status"],
            "H3": b["H3"]["status"],
            "H4": b["H4"]["status"],
            "H5": b["H5"]["status"],
            "H6": b["H6"]["status"],
        },
        "rare_event_support": rare,
        "H3_resource_results_by_codebook": b["H3"]["resource_component_by_codebook"],
        "H5_secondary_FORMAL_profile_sensitivity": b["H5"],
        "H6_side_effect_and_core_gate_context": b["H6"],
        "mandatory_22_scene_exclusion_sensitivity": {
            "role": g["scientific_role"],
            "primary_full_cohort_remains_authoritative": True,
            "H1": g["H1"],
            "H2": g["H2"],
            "H3": g["H3"],
            "H4": g["H4"],
            "H5": g["H5"],
            "H6": g["H6"],
        },
        "H4_comparator_provenance": h["H4_reporting_correction"],
        "C3_scope": {
            "primary_confirmatory": c3["scope"]["primary_confirmatory"],
            "secondary": c3["scope"]["secondary"],
            "global_success_claim_allowed": False,
        },
        "authority": {
            "stage915b_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
            "stage915g_summary_sha256": EXPECTED[str(P915G_SUMMARY)],
            "stage915h_addendum_sha256": EXPECTED[str(P915H_ADDENDUM)],
        },
    }
    atomic_json(RESULTS, results)

    # CSV: calibration.
    atomic_csv(
        CSV_CAL,
        ["quantity", "raw", "calibrated", "role"],
        [
            {
                "quantity": "critical_event_Brier_CAL",
                "raw": h["block94_event_calibration_reporting"]["CAL_raw_Brier"],
                "calibrated": h["block94_event_calibration_reporting"]["CAL_calibrated_Brier"],
                "role": "current_Block9.4_CAL_event_calibration",
            },
            {
                "quantity": "critical_event_ECE15_equal_width_CAL",
                "raw": h["block94_event_calibration_reporting"]["CAL_raw_ECE15_equal_width"],
                "calibrated": h["block94_event_calibration_reporting"]["CAL_calibrated_ECE15_equal_width"],
                "role": "current_Block9.4_CAL_event_calibration",
            },
            {
                "quantity": "macro_abs_beam_coverage_error_FORMAL_nonconfirmatory",
                "raw": b["H1"]["point_estimates"]["macro_abs_coverage_error_RAW"],
                "calibrated": b["H1"]["point_estimates"]["macro_abs_coverage_error_CALIBRATED"],
                "role": "nonconfirmatory_numerical_H1_evidence",
            },
        ],
    )

    # CSV: H3 resources.
    h3_rows = []
    for cb in ("16", "32", "64"):
        q = b["H3"]["resource_component_by_codebook"][cb]
        h3_rows.append({
            "codebook_size": cb,
            "delta_mean_K_proposed_minus_q099": q["delta_mean_K_proposed_minus_q099"],
            "delta_mean_K_CI95_lo": q["delta_mean_K_CI95"][0],
            "delta_mean_K_CI95_hi": q["delta_mean_K_CI95"][1],
            "delta_mean_overhead_proposed_minus_q099": q["delta_mean_overhead_proposed_minus_q099"],
            "delta_mean_overhead_CI95_lo": q["delta_mean_overhead_CI95"][0],
            "delta_mean_overhead_CI95_hi": q["delta_mean_overhead_CI95"][1],
            "resource_reduction_supported": q["resource_reduction_supported"],
            "full_H3_confirmatory_pass_allowed": False,
        })
    atomic_csv(
        CSV_H3,
        [
            "codebook_size",
            "delta_mean_K_proposed_minus_q099",
            "delta_mean_K_CI95_lo",
            "delta_mean_K_CI95_hi",
            "delta_mean_overhead_proposed_minus_q099",
            "delta_mean_overhead_CI95_lo",
            "delta_mean_overhead_CI95_hi",
            "resource_reduction_supported",
            "full_H3_confirmatory_pass_allowed",
        ],
        h3_rows,
    )

    # CSV: H5.
    h5_rows = []
    for name, q in sorted(b["H5"]["comparisons"].items()):
        h5_rows.append({
            "comparison": name,
            "n": q["n"],
            "baseline_violations": q["baseline_violations"],
            "recovery_violations": q["recovery_violations"],
            "delta_recovery_minus_baseline": q["point_delta_rate_recovery_minus_baseline"],
            "CI95_lo": q["CI95"][0],
            "CI95_hi": q["CI95"][1],
            "decision": q["decision"],
            "role": "secondary_FORMAL_profile_sensitivity",
        })
    atomic_csv(
        CSV_H5,
        ["comparison", "n", "baseline_violations", "recovery_violations",
         "delta_recovery_minus_baseline", "CI95_lo", "CI95_hi", "decision", "role"],
        h5_rows,
    )

    # CSV: H6.
    h6_rows = []
    for name, q in sorted(b["H6"]["paired_deltas"].items()):
        h6_rows.append({
            "comparison": name,
            "paired_finite_scene_count": q["paired_finite_scene_count"],
            "point_delta": q["point_delta"],
            "CI95_lo": q["CI95"][0],
            "CI95_hi": q["CI95"][1],
            "role": "side_effect_and_bootstrap_context",
        })
    atomic_csv(
        CSV_H6,
        ["comparison", "paired_finite_scene_count", "point_delta", "CI95_lo", "CI95_hi", "role"],
        h6_rows,
    )

    # -------------------------------------------------------------------------------------------------
    # Robustness / failure cases / latency-resources.
    # -------------------------------------------------------------------------------------------------
    robustness = {
        "schema": "stage9_block916_v2_robustness_failure_latency_v1",
        "status": "PAPER_READY_ROBUSTNESS_FAILURE_LATENCY_RESOURCE_CONTEXT",
        "human_profile_robustness": c["human_profile_robustness"],
        "calibration_perturbation_robustness": c["calibration_robustness"],
        "technical_recovery_exclusion_sensitivity": {
            "primary_full_cohort_remains_authoritative": True,
            "excluded_scene_count": 22,
            "secondary_scene_count": 26187,
            "H1_confirmatory_reactivation": False,
            "H2": g["H2"]["secondary_sensitivity_status"],
            "H3": g["H3"]["secondary_sensitivity_status"],
            "H4": g["H4"]["secondary_sensitivity_status"],
            "H5": g["H5"]["secondary_sensitivity_status"],
            "H6": g["H6"]["secondary_sensitivity_core_gate_status"],
        },
        "failure_case_analysis": d["requested_failure_case_analysis"],
        "failure_case_reporting_boundary": (
            "This is a completed fail-closed disposition audit, not an evaluation of 17 failure slices. "
            "No requested slice had a prefrozen exact membership rule."
        ),
        "latency": e["latency"],
        "resources": e["resources"],
        "latency_wording_correction": h["C3_scope_and_latency_wording"],
        "authority": {
            "stage915c_report_sha256": EXPECTED[str(P915C_REPORT)],
            "stage915d_report_sha256": EXPECTED[str(P915D_REPORT)],
            "stage915e_report_sha256": EXPECTED[str(P915E_REPORT)],
            "stage915g_summary_sha256": EXPECTED[str(P915G_SUMMARY)],
            "stage915h_addendum_sha256": EXPECTED[str(P915H_ADDENDUM)],
        },
    }
    atomic_json(ROBUSTNESS, robustness)

    # -------------------------------------------------------------------------------------------------
    # Claim-evidence / novelty audit.
    # -------------------------------------------------------------------------------------------------
    claims = {
        "schema": "stage9_block916_v2_claim_evidence_matrix_v1",
        "status": "STRICT_FINAL_CLAIM_EVIDENCE_AUDIT_COMPLETE",
        "rule": "Unsupported claims are removed; no post-hoc experiment is introduced to rescue them.",
        "claims": [
            {
                "claim_id": "METHOD_SHARED_POSTERIOR_CRITICALITY",
                "claim": (
                    "Stage 9 uses the frozen sensing-informed future-motion posterior as a shared carrier "
                    "for beam uncertainty and traffic-criticality reasoning."
                ),
                "status": "ALLOWED_METHOD_CLAIM",
                "evidence": ["frozen Stage-9 method/scope authorities", "current execution chain"],
                "boundary": "Does not claim a new predictor or prove absolute calibration quality.",
            },
            {
                "claim_id": "H1_CONFIRMATORY",
                "claim": "Confirmatory H1 was supported on FORMAL.",
                "status": "FORBIDDEN",
                "evidence": [h["H1_reporting_correction"]["confirmatory_status"]],
                "boundary": h["H1_reporting_correction"]["paper_wording_boundary"],
            },
            {
                "claim_id": "H1_NUMERICAL",
                "claim": "Later materialized RAW-vs-CAL records show a reduction in macro absolute coverage error.",
                "status": "ALLOWED_NONCONFIRMATORY_NUMERICAL_CLAIM",
                "evidence": [b["H1"]["status"], g["H1"]["numerical_sensitivity_status"]],
                "boundary": "Must be labeled non-confirmatory and must not be described as well-calibrated absolute coverage.",
            },
            {
                "claim_id": "H2_CRITICAL_RELIABILITY",
                "claim": "C2 reduces realized critical-event beam miss versus fixed q=.95.",
                "status": "NOT_ESTABLISHED",
                "evidence": [b["H2"]["status"], f"0/{rare['critical_label_valid_denominator']} critical events"],
                "boundary": "No substitute endpoint; zero observed events are not zero risk.",
            },
            {
                "claim_id": "H3_FULL",
                "claim": "C2 approaches q=.99 critical reliability with lower resources.",
                "status": "PARTIAL_ONLY",
                "evidence": [b["H3"]["status"]],
                "boundary": (
                    "Only the resource component is evaluable/supported; the critical-reliability component "
                    "is not evaluable and full H3 confirmation is not allowed."
                ),
            },
            {
                "claim_id": "H4_MATCHED_CRITICAL",
                "claim": "C2 reduces critical miss/outage versus the CAL-frozen matched uncertainty-only comparator.",
                "status": "NOT_ESTABLISHED",
                "evidence": [b["H4"]["status"], h["H4_reporting_correction"]["exact_preoutcome_executable_comparator"]],
                "boundary": "H4 is not evaluable because the critical-conditioned endpoint has zero event support.",
            },
            {
                "claim_id": "H5_C3_PROFILE",
                "claim": "Recovery-aware C3 improves modeled safety risk across all human profiles.",
                "status": "FORBIDDEN_AS_GLOBAL_CLAIM",
                "evidence": [b["H5"]["comparisons"], c["human_profile_robustness"]["status"]],
                "boundary": (
                    "Report all six secondary comparisons. FAST shows no clear difference; NOMINAL/SLOW show "
                    "supported reductions. Conclusions are human-profile sensitive."
                ),
            },
            {
                "claim_id": "H6_CORE_SIDE_EFFECT",
                "claim": "The recovery-aware policy satisfies the frozen Stage-6 core H6 side-effect gates.",
                "status": "ALLOWED_WITH_SCOPE",
                "evidence": [b["H6"]["frozen_core_gate_results"], b["H6"]["frozen_core_gate_PASS"]],
                "boundary": "Do not convert this into improvement claims for every metric/baseline or invent new NI thresholds.",
            },
            {
                "claim_id": "CALIBRATION_ROBUSTNESS",
                "claim": "Results are robust to underconfident/overconfident calibration perturbations.",
                "status": "NOT_ESTABLISHED",
                "evidence": [c["calibration_robustness"]["status"]],
                "boundary": "No prefrozen numeric perturbation transformation existed; no post-FORMAL perturbation was executed.",
            },
            {
                "claim_id": "FAILURE_CASE_ROBUSTNESS",
                "claim": "The method was evaluated across 17 requested failure-case slices.",
                "status": "FORBIDDEN",
                "evidence": [d["requested_failure_case_analysis"]],
                "boundary": "0/17 had prefrozen exact membership authority; only the fail-closed disposition audit was completed.",
            },
            {
                "claim_id": "LATENCY_REALTIME",
                "claim": "Stage 9 is low-latency, real-time, or faster than alternatives.",
                "status": "FORBIDDEN",
                "evidence": [e["latency"], h["C3_scope_and_latency_wording"]],
                "boundary": "No validated system/end-to-end FORMAL timing carrier establishes these claims.",
            },
            {
                "claim_id": "NO_SECOND_LEARNED_RISK_NETWORK",
                "claim": "The additional safety-aware reasoning does not require a second learned risk network.",
                "status": "ALLOWED_ARCHITECTURAL_FACT",
                "evidence": ["frozen Stage-9 architecture and execution chain"],
                "boundary": "This is not a latency, speedup, or real-time claim.",
            },
            {
                "claim_id": "DEEPSENSE",
                "claim": "DeepSense validates the communication policy.",
                "status": "ALLOWED_ONLY_WITH_BOUNDARY",
                "evidence": [f["external_validation_boundary"]],
                "boundary": "Does not validate optical headlamp, glare, collision, or C3 human-safety claims.",
            },
            {
                "claim_id": "CRASH_REDUCTION",
                "claim": "C3 reduces real-world crash incidence.",
                "status": "FORBIDDEN",
                "evidence": ["C3 is a human-factors-informed modeled safety extension"],
                "boundary": "No population-level crash-reduction claim.",
            },
            {
                "claim_id": "NOVEL_PREDICTOR_OR_CALIBRATOR",
                "claim": "Stage 9 contributes a new collision predictor or new uncertainty calibration method.",
                "status": "FORBIDDEN_NOVELTY_CLAIM",
                "evidence": ["frozen novelty boundary"],
                "boundary": "Novelty is in the shared interface/decision layer, not a new predictor/calibrator.",
            },
        ],
        "safe_central_story": (
            "Stage 9 reuses a frozen sensing-informed future-motion posterior to reason jointly about where a "
            "vehicular receiver may be and which predicted futures are traffic-critical. This supports a "
            "criticality-constrained communication decision layer and a secondary human-factors-informed "
            "recovery-aware ADB extension, with all performance claims restricted to the evidence actually observed."
        ),
    }
    atomic_json(CLAIMS, claims)

    # -------------------------------------------------------------------------------------------------
    # Limitations.
    # -------------------------------------------------------------------------------------------------
    limitations = {
        "schema": "stage9_block916_v2_limitations_v1",
        "status": "PAPER_READY_LIMITATIONS_FROZEN",
        "limitations": [
            {
                "id": "H1_AUTHORITY",
                "text": (
                    "H1 was not evaluated confirmatorily under the frozen preoutcome authority because no "
                    "prefrozen FORMAL uncalibrated arm was available. Later RAW-vs-CAL numerical results are "
                    "non-confirmatory evidence."
                ),
            },
            {
                "id": "RARE_CRITICAL_EVENTS",
                "text": (
                    f"No ground-truth critical events were observed among {rare['critical_label_valid_denominator']} "
                    "valid critical labels. H2 and H4 are not evaluable and H3 reliability is not evaluable. "
                    "Zero observed events must not be interpreted as zero risk."
                ),
            },
            {
                "id": "H5_PROFILE_SENSITIVITY",
                "text": "C3 modeled safety conclusions are sensitive to the frozen FAST/NOMINAL/SLOW human-response profile.",
            },
            {
                "id": "CALIBRATION_PERTURBATION",
                "text": (
                    "Under/over-confidence calibration robustness is not evaluable because no prefrozen numeric "
                    "perturbation rule existed."
                ),
            },
            {
                "id": "FAILURE_SLICES",
                "text": (
                    "The 17 requested failure-case slices could not be evaluated without inventing post-FORMAL "
                    "membership rules; therefore only a fail-closed authority audit is reported."
                ),
            },
            {
                "id": "LATENCY",
                "text": (
                    "C2 has no recorded FORMAL solver timing; C3 has preoutcome action-materialization component "
                    "runtimes but no validated FORMAL system/end-to-end timing carrier. Low-latency, real-time, "
                    "and speedup claims are not supported."
                ),
            },
            {
                "id": "TECHNICAL_RECOVERY",
                "text": (
                    "Twenty-two scenes required documented physical future-GT rereads. The full 26,209-scene "
                    "cohort remains primary; the exact 22-scene exclusion is a mandatory secondary sensitivity."
                ),
            },
            {
                "id": "DEEPSENSE_BOUNDARY",
                "text": (
                    "No untouched locally evaluable DeepSense split remains for a new independent Stage-9 "
                    "confirmation. DeepSense evidence is limited to communication-policy support and does not "
                    "validate optical/headlamp, glare, collision, or C3 safety claims."
                ),
            },
            {
                "id": "C3_MODELED_EXTENSION",
                "text": (
                    "C3 is a human-factors-informed modeled safety extension, not evidence of population-level "
                    "crash reduction or a universal glare-recovery time."
                ),
            },
        ],
        "authority": {
            "stage915f_summary_sha256": EXPECTED[str(P915F_SUMMARY)],
            "stage915g_summary_sha256": EXPECTED[str(P915G_SUMMARY)],
            "stage915h_addendum_sha256": EXPECTED[str(P915H_ADDENDUM)],
        },
    }
    atomic_json(LIMITATIONS, limitations)

    # -------------------------------------------------------------------------------------------------
    # Supersession map.
    # -------------------------------------------------------------------------------------------------
    supersession = {
        "schema": "stage9_block916_v2_supersession_map_v1",
        "status": "CURRENT_VS_LEGACY_REPORTING_AUTHORITY_FROZEN",
        "current_authority": {
            "9.13": EXPECTED[str(P913_SEAL)],
            "9.14_v2": EXPECTED[str(P914_SEAL)],
            "9.15F_primary_closure": EXPECTED[str(P915F_SEAL)],
            "9.15G_mandatory_secondary_sensitivity": EXPECTED[str(P915G_SEAL)],
            "9.15H_provenance_wording_addendum": EXPECTED[str(P915H_SEAL)],
            "new_9.16_package": "THIS_PACKAGE",
        },
        "legacy_or_superseded": {
            "block916_R1_manifest_sha256": EXPECTED[str(P916_OLD_MANIFEST)],
            "block916_R1_seal_sha256": EXPECTED[str(P916_OLD_SEAL)],
            "block916_R1_current_reporting_authority": False,
            "reason": (
                "The legacy block916_R1 package predates and does not bind the current 9.15F/G/H chain; "
                "its package gaps and dispositions are not the current Stage-9 reporting authority."
            ),
        },
    }
    atomic_json(SUPERSESSION, supersession)

    # -------------------------------------------------------------------------------------------------
    # Source/config manifest.
    # -------------------------------------------------------------------------------------------------
    source_paths = [
        Path(__file__).resolve(),
        S9 / "scripts/run_stage9_block915b_v2_current_formal_H1_H6_bootstrap.py",
        S9 / "scripts/run_stage9_block915c_v2_robustness.py",
        S9 / "scripts/run_stage9_block915d_v2R1_failure_case_analysis.py",
        S9 / "scripts/run_stage9_block915e_v2R1_latency_resources.py",
        S9 / "scripts/run_stage9_block915f_v2_final_stage915_closure.py",
        S9 / "scripts/run_stage9_block915g_v1_technical_recovery_sensitivity.py",
        S9 / "scripts/run_stage9_block915h_v1_provenance_wording_addendum.py",
        P_C3_CONTRACT,
        P94_CONTRACT,
    ]
    source_entries = []
    for p in source_paths:
        req(p.is_file(), f"missing current source/config file: {p}")
        source_entries.append({
            "path": str(p),
            "sha256": sha256_path(p),
            "bytes": p.stat().st_size,
        })
    source_manifest = {
        "schema": "stage9_block916_v2_source_config_manifest_v1",
        "status": "CURRENT_SOURCE_CONFIG_MANIFEST_COMPLETE",
        "entries": source_entries,
    }
    atomic_json(SOURCE_MANIFEST, source_manifest)

    # -------------------------------------------------------------------------------------------------
    # Deterministic paper-ready Markdown summary.
    # -------------------------------------------------------------------------------------------------
    h1p = b["H1"]["point_estimates"]
    md = f"""# Stage 9 — Current Paper-Ready Result Summary

## Scope

Primary confirmatory scope: **C1+C2**.  
C3: **secondary profile sensitivity**; no global C3 success claim.

## Formal population

- Frozen WOMD FORMAL scenarios: **26,209**
- Valid critical labels: **24,917**
- Observed ground-truth critical events: **0**
- Wilson 95% CI for event prevalence: **{rare['Wilson_95pct']}**
- Zero observed events are **not** interpreted as zero risk.
- Mandatory technical-recovery sensitivity excludes exactly **22** documented reread scenes, leaving **26,187** scenes; the full cohort remains primary.

## H1

Confirmatory H1: **NOT TESTED** (`NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM`).

Later materialized RAW-vs-CAL records provide **non-confirmatory numerical evidence**:
- macro absolute coverage error RAW: {h1p['macro_abs_coverage_error_RAW']}
- macro absolute coverage error CALIBRATED: {h1p['macro_abs_coverage_error_CALIBRATED']}
- CAL-RAW delta: {h1p['primary_delta_CALIBRATED_minus_RAW']}
- paired bootstrap 95% CI: {b['H1']['primary_delta_CI95']}

This must not be described as confirmatory H1 support or as demonstrating good absolute calibration.

## H2–H4

- H2: **NOT EVALUABLE** — zero observed critical-event support.
- H3: **PARTIALLY EVALUABLE** — resource component only; full H3 confirmation is not allowed.
- H4: **NOT EVALUABLE** — zero observed critical-event support.
- H4 executable comparator provenance: preoutcome CAL-frozen closest-mean-K uncertainty-only fixed-q Top-K, `fixed_q_0.95`.

## C3 / H5–H6

H5 is a **secondary FORMAL profile sensitivity**:
- FAST: no clear difference in both frozen baseline comparisons.
- NOMINAL: reduction supported in both comparisons.
- SLOW: reduction supported in both comparisons.
- Therefore there is **no global 6/6 H5 success claim**.

H6: **{b['H6']['status']}**. The frozen Stage-6 core side-effect gates pass; this is not a claim of improvement for every metric and baseline.

## Calibration

Current Block-9.4 critical-event calibration authority uses **15 equal-width bins**:
- raw Brier: {h['block94_event_calibration_reporting']['CAL_raw_Brier']}
- calibrated Brier: {h['block94_event_calibration_reporting']['CAL_calibrated_Brier']}
- raw ECE15: {h['block94_event_calibration_reporting']['CAL_raw_ECE15_equal_width']}
- calibrated ECE15: {h['block94_event_calibration_reporting']['CAL_calibrated_ECE15_equal_width']}

The older ECE20 artifact is not the current reporting authority.

## Robustness and failure analysis

- Human-profile conclusions: **profile-sensitive**.
- Calibration under/over-confidence perturbation: **not evaluable** without a prefrozen numeric perturbation rule.
- Requested failure slices: **0/17 evaluable** with prefrozen exact membership authority. This is a fail-closed disposition audit, not a 17-slice empirical analysis.
- The mandatory exact 22-scene technical-recovery exclusion sensitivity supplements and cannot replace the primary full-cohort result.

## Latency / resources

- H3 communication resource statistics are reportable with the frozen paired CIs.
- No recorded FORMAL C2 solver timing supports a system-latency claim.
- C3 has per-scene preoutcome action-materialization `runtime_s` component carriers, but no validated FORMAL system/end-to-end timing carrier.
- Therefore low-latency, real-time, and speedup claims are not authorized.
- The architecture may be described as adding safety-aware reasoning **without a second learned risk network**; that is an architectural fact, not a timing claim.

## External validation boundary

DeepSense supports the **communication policy**, not the optical headlamp, glare, collision, or C3 human-safety claims. No untouched locally evaluable DeepSense split remains for a new independent Stage-9 confirmation.

## Safe central story

Stage 9 reuses a frozen sensing-informed future-motion posterior to reason jointly about where a vehicular receiver may be and which predicted futures are traffic-critical. This supports a criticality-constrained communication decision layer and a secondary human-factors-informed recovery-aware ADB extension, with all performance claims restricted to the evidence actually observed.

## Supersession

The old `block916_R1_final_paper_ready_package` is **legacy/superseded for current reporting**. Manuscript writing must use this new Stage-9.16-v2 package and its bound 9.15F/G/H authorities.
"""
    atomic_text(SUMMARY_MD, md)

    # -------------------------------------------------------------------------------------------------
    # Package manifest and final seal.
    # -------------------------------------------------------------------------------------------------
    component_paths = [
        METHODS, PROVENANCE, CALIBRATION, RESULTS, ROBUSTNESS, CLAIMS, LIMITATIONS,
        SUPERSESSION, SOURCE_MANIFEST, CSV_CAL, CSV_H3, CSV_H5, CSV_H6, SUMMARY_MD,
    ]
    components = {
        p.name: {"path": str(p), "sha256": sha256_path(p), "bytes": p.stat().st_size}
        for p in component_paths
    }

    manifest = {
        "schema": "stage9_block916_v2_paper_ready_manifest_v1",
        "status": "FROZEN_CURRENT_STAGE9_PAPER_READY_PACKAGE_MANIFEST",
        "package_role": "NEW_CURRENT_AUTHORITY_SUPERSEDING_LEGACY_BLOCK916_R1_FOR_REPORTING",
        "input_authorities": {
            "stage913_final_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage914_v2_external_boundary_seal_sha256": EXPECTED[str(P914_SEAL)],
            "stage915a_contract_seal_sha256": EXPECTED[str(P915A_SEAL)],
            "stage915b_statistics_seal_sha256": EXPECTED[str(P915B_SEAL)],
            "stage915c_robustness_seal_sha256": EXPECTED[str(P915C_SEAL)],
            "stage915d_failure_case_seal_sha256": EXPECTED[str(P915D_SEAL)],
            "stage915e_latency_resources_seal_sha256": EXPECTED[str(P915E_SEAL)],
            "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
            "stage915g_secondary_sensitivity_seal_sha256": EXPECTED[str(P915G_SEAL)],
            "stage915h_provenance_wording_seal_sha256": EXPECTED[str(P915H_SEAL)],
        },
        "components": components,
        "package_completeness": {
            "frozen_methods_specification": True,
            "mathematical_problem_statement": True,
            "dataset_split_provenance": True,
            "calibration_tables": True,
            "critical_event_calibration": True,
            "communication_tables": True,
            "ADB_tables": True,
            "FAST_NOMINAL_SLOW_sensitivity": True,
            "paired_CIs": True,
            "failure_case_disposition": True,
            "latency_resources": True,
            "claim_evidence_matrix": True,
            "limitations": True,
            "source_code_configs_manifest": True,
            "final_reproducibility_seal": "PENDING_AT_MANIFEST_WRITE",
        },
        "new_scientific_analysis": False,
    }
    atomic_json(MANIFEST, manifest)
    manifest_sha = sha256_path(MANIFEST)

    seal = {
        "schema": "stage9_block916_v2_final_reproducibility_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_916_V2_CURRENT_PAPER_READY_PACKAGE",
        "stage913_final_seal_sha256": EXPECTED[str(P913_SEAL)],
        "stage914_v2_external_boundary_seal_sha256": EXPECTED[str(P914_SEAL)],
        "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
        "stage915g_secondary_sensitivity_seal_sha256": EXPECTED[str(P915G_SEAL)],
        "stage915h_provenance_wording_seal_sha256": EXPECTED[str(P915H_SEAL)],
        "paper_ready_manifest_sha256": manifest_sha,
        "component_hashes": {name: meta["sha256"] for name, meta in components.items()},
        "old_block916_R1_current_reporting_authority": False,
        "H1_confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "H2_status": b["H2"]["status"],
        "H3_status": b["H3"]["status"],
        "H4_status": b["H4"]["status"],
        "H5_role": "SECONDARY_FORMAL_PROFILE_SENSITIVITY",
        "H6_status": b["H6"]["status"],
        "primary_full_cohort_scene_count": 26209,
        "mandatory_secondary_recovery_exclusion_scene_count": 22,
        "new_outcome_scan": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "new_statistics": False,
        "new_model_policy_solver_execution": False,
        "retuning": False,
        "threshold_change": False,
        "baseline_change": False,
        "claim_rescue_experiment": False,
        "next": "MANUSCRIPT_WRITING_FROM_FROZEN_STAGE9_916_V2_PACKAGE_ONLY",
    }
    atomic_json(SEAL, seal)

    print("METHODS_PROBLEM_STATEMENT = COMPLETE")
    print("DATASET_PROVENANCE = COMPLETE")
    print("CALIBRATION_TABLES = COMPLETE")
    print("COMMUNICATION_RESULTS = COMPLETE")
    print("ADB_RESULTS = COMPLETE")
    print("FAST_NOMINAL_SLOW_SENSITIVITY = COMPLETE")
    print("PAIRED_CI_TABLES = COMPLETE")
    print("FAILURE_CASE_DISPOSITION = COMPLETE")
    print("LATENCY_RESOURCES = COMPLETE")
    print("CLAIM_EVIDENCE_MATRIX = COMPLETE")
    print("LIMITATIONS = COMPLETE")
    print("SOURCE_CONFIG_MANIFEST = COMPLETE")
    print("OLD_BLOCK916_R1_CURRENT_AUTHORITY = FALSE")
    print("NEW_OUTCOME_SCAN = FALSE")
    print("NEW_RNG = FALSE")
    print("NEW_BOOTSTRAP = FALSE")
    print("NEW_STATISTICS = FALSE")
    print("SCIENTIFIC_REEXECUTION = FALSE")
    print("STAGE 9.16-v2 = COMPLETE")
    print("MANIFEST_SHA256 =", manifest_sha)
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT = MANUSCRIPT_WRITING_FROM_FROZEN_STAGE9_916_V2_PACKAGE_ONLY")
    print("=" * 124)


if __name__ == "__main__":
    main()

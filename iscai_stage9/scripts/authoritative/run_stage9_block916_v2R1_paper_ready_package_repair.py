#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

# =================================================================================================
# Stage 9.16-v2R1
#
# Append-only paper-ready PACKAGING / REPORTING repair.
#
# STRICTLY FORBIDDEN HERE:
#   * scenario_outcome scan
#   * future-GT read
#   * RNG / bootstrap
#   * new statistic / new inferential endpoint
#   * model / predictor / C1 / C2 / C3 execution
#   * policy / baseline / threshold change
#   * H1 confirmatory reactivation
#   * reaction-margin distribution construction
#
# This runner only packages already-sealed aggregate evidence and fixes the independently-audited
# 9.16-v2 completeness defects.
# =================================================================================================

# Current Stage-9 scientific authorities.
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
P913_SUMMARY = B913 / "stage9_block913_v1_3_FINAL_H1_H6_summary.json"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

B913A = S9 / "artifacts/block913a_v1_2_c1_c2_prediction_side_continuation"
P913A_SEAL = B913A / "stage9_block913a_v1_2_continuation_seal.json"
P913A_EXECUTED_SOURCE = S9 / "scripts/run_stage9_block913a_v1_1_formal_c1_c2_prediction_side.py"

P_R4_SOURCE = S9 / "scripts/run_stage9_block913_v1_3_primary_pretruth_R4.py"
P_R4_SEAL = S9 / "artifacts/block913_v1_3_primary_pretruth_R4/stage9_block913_v1_3_pretruth_seal.json"

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

P_H1_OLD_SEAL = (
    S9
    / "artifacts/block913b2c_h1_formal_disposition_pre_gt"
    / "stage9_block913b2c_h1_formal_disposition_seal.json"
)
P_H4_REPORT = (
    S9
    / "artifacts/block913b3_R2_h4_preoutcome_authority_correction"
    / "stage9_block913b3_R2_h4_preoutcome_authority_correction_report.json"
)
P_H4_SEAL = (
    S9
    / "artifacts/block913b3_R2_h4_preoutcome_authority_correction"
    / "stage9_block913b3_R2_h4_preoutcome_authority_correction_seal.json"
)

# Calibration.
P94_CONTRACT = (
    S9
    / "artifacts/block94a_evaluator_label_and_calibration_contract_freeze"
    / "evaluator_label_and_calibration_contract_v1.json"
)
P94_DIAGNOSTICS = (
    S9
    / "artifacts/block94b2a_contract_self_hash_binding_repair"
    / "cal_fit_set_calibration_diagnostics.json"
)
P94_SEAL = (
    S9
    / "artifacts/block94b2a_contract_self_hash_binding_repair"
    / "stage9_block94b2a_contract_self_hash_binding_repair_seal.json"
)

# C3 scope.
P_C3_CONTRACT = S9 / "configs/stage9_block913_v1_2_formal_c1_c2_primary_c3_secondary_execution_contract.json"

# Executed C2 semantics and historical non-primary branch provenance.
P97_INTERFACE = S9 / "configs/stage9_v1_1_1_block97_c2_interface_contract.json"
P97_CONDITIONAL_CONTRACT = S9 / "configs/stage9_block97_criticality_constrained_beam_policy_contract.json"
P97_JOINT_V2 = S9 / "configs/stage9_block97_pdf_joint_critical_constraint_amendment_v2.json"
P97_JOINT_V3 = S9 / "configs/stage9_block97_selected_beam_tiebreak_amendment_v3.json"

# Previous 9.16-v2 candidate package. It is scientifically compatible, but independently audited as incomplete.
V2 = S9 / "artifacts/block916_v2_new_paper_ready_package"
P916V2_MANIFEST = V2 / "stage9_block916_v2_paper_ready_manifest.json"
P916V2_SEAL = V2 / "stage9_block916_v2_final_reproducibility_seal.json"
P916V2_RUNNER = S9 / "scripts/run_stage9_block916_v2_new_paper_ready_package.py"

EXPECTED = {
    str(P913_SUMMARY): "292d515a3f52c82ad08460e1122a805d236a8aeaefa82271ff84664c2cddfe60",
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",

    str(P913A_SEAL): "a67b1ee879cdaf6dc4516834c439fbc0bdd78243281c30802f71d4b8ea0cea95",
    str(P913A_EXECUTED_SOURCE): "df82015ea8ef5f090fbcbbd7242d83ba72ca011f8fd19c73bcf2d6d0590f8e8c",
    str(P_R4_SOURCE): "5b1dd54f8be489c6c727842b719eca5da76c564e44070960b9eb799ab9970050",
    str(P_R4_SEAL): "ab3c20b7282c8e3a2978f25e3c16e7d1ffe700354a27b8a96f7049f45cb3090f",

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

    str(P_H1_OLD_SEAL): "f86829c71ab5b410155e0bbeabe256bd280b17d7037576356bde9253d0306442",
    str(P_H4_REPORT): "6a844e0f8fbd6df94f792b5d9b8943e686a468f35a02d718d675c8ac87cfa306",
    str(P_H4_SEAL): "5128db258d5972c537fe5de4c3a1ebd834ca34c2a9c144fcd8753c3436fd9e12",

    str(P94_CONTRACT): "2b893497dc792d033da7dac06817f89c70ec1edb390de6e264664b1a3ec79e43",
    str(P94_DIAGNOSTICS): "35da53a736c5831c54b31bf579c3cda45710351098a41d59cb38a7deba985bd7",
    str(P94_SEAL): "34b71b31639c3ea2b9722bfa6c83e88fedc79f3636b4df809d8de6d172e5d82b",

    str(P_C3_CONTRACT): "991349ff9029b197652411e7ac60767458debf15674fcd103bfa56cb16288001",

    str(P97_INTERFACE): "190eee5f3ec3ba7302e3eb44b9000d6951af30a259c650673780be858f89a79d",
    str(P97_CONDITIONAL_CONTRACT): "1851ca01f6e43b1aa8a55d32e702e690ac8aad9815bd9a71b04806c98ac2500a",
    str(P97_JOINT_V2): "1be1ecd4517338063490eb83f45b5a6a740ac2e5d66654945029090086ca98e2",
    str(P97_JOINT_V3): "b0309c6248c192012ede754203cb5f3d39e99c31d14346ff3da1fe893def1681",

    str(P916V2_MANIFEST): "b610da3d57065ab6fb2421d903f040284030ef4ea6006f0d5a1284a79615ef1e",
    str(P916V2_SEAL): "7184893977d0acb9e0b9947438cc016ede65c823bcf943f847220c3813d7bd52",
    str(P916V2_RUNNER): "8053684ccfc5953e50b7575f52ed8e1651ce39e74f2b4aeb2b6495a452c9df55",
}

OUT = S9 / "artifacts/block916_v2R1_paper_ready_package_repair"

METHODS = OUT / "stage9_block916_v2R1_methods_mathematical_problem.json"
PROVENANCE = OUT / "stage9_block916_v2R1_dataset_provenance.json"
CALIBRATION = OUT / "stage9_block916_v2R1_calibration_tables.json"
ECE_JSON = OUT / "stage9_block916_v2R1_ece15_reliability_bins.json"
ECE_CSV = OUT / "stage9_block916_v2R1_ece15_reliability_bins.csv"

COMM_JSON = OUT / "stage9_block916_v2R1_full_communication_table.json"
COMM_CSV = OUT / "stage9_block916_v2R1_full_communication_table.csv"

RESULTS = OUT / "stage9_block916_v2R1_paper_ready_results.json"
ROBUSTNESS = OUT / "stage9_block916_v2R1_robustness_failure_latency.json"
ADB_BOUNDARY = OUT / "stage9_block916_v2R1_adb_reporting_boundary.json"

CLAIMS = OUT / "stage9_block916_v2R1_claim_evidence_matrix.json"
LIMITATIONS = OUT / "stage9_block916_v2R1_limitations.json"
SUPERSESSION = OUT / "stage9_block916_v2R1_supersession_map.json"
SOURCE_MANIFEST = OUT / "stage9_block916_v2R1_canonical_source_config_manifest.json"
AUTHORITY_MANIFEST = OUT / "stage9_block916_v2R1_authority_manifest.json"

H3_CSV = OUT / "stage9_block916_v2R1_table_H3_resource.csv"
H5_CSV = OUT / "stage9_block916_v2R1_table_H5_profile_sensitivity.csv"
H6_CSV = OUT / "stage9_block916_v2R1_table_H6_side_effects.csv"
CAL_SUMMARY_CSV = OUT / "stage9_block916_v2R1_table_calibration_summary.csv"

SUMMARY_MD = OUT / "stage9_block916_v2R1_paper_ready_summary.md"
MANIFEST = OUT / "stage9_block916_v2R1_paper_ready_manifest.json"
SEAL = OUT / "stage9_block916_v2R1_final_reproducibility_seal.json"

COMM_POLICIES = (
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
CODEBOOKS = ("16", "32", "64")
COMM_REQUIRED_METRICS = (
    "beam_outage_rate",
    "coverage_rate",
    "joint_critical_miss_rate",
    "mean_K",
    "mean_beam_gain_loss_db",
    "mean_effective_rate_bps",
    "mean_probing_overhead_fraction",
    "mean_received_power_loss_db",
)


class FailClosed(RuntimeError):
    pass


def req(cond: bool, msg: str) -> None:
    if not cond:
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
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    atomic_text(path, s.getvalue())


def hash_gate() -> None:
    for raw, wanted in EXPECTED.items():
        p = Path(raw)
        req(p.is_file(), f"missing current authority: {p}")
        got = sha256_path(p)
        req(got == wanted, f"SHA256 drift {p}: {got} != {wanted}")
        print("EXACT PASS", p)


def verify_v2_components(v2_manifest: dict[str, Any], v2_seal: dict[str, Any]) -> dict[str, Path]:
    req(
        v2_seal["paper_ready_manifest_sha256"] == EXPECTED[str(P916V2_MANIFEST)],
        "9.16-v2 seal does not bind expected v2 manifest",
    )
    out: dict[str, Path] = {}
    for name, meta in v2_manifest["components"].items():
        p = Path(meta["path"])
        req(p.is_file(), f"missing 9.16-v2 component: {p}")
        got = sha256_path(p)
        req(got == meta["sha256"], f"9.16-v2 manifest component drift: {name}")
        req(
            v2_seal["component_hashes"].get(name) == got,
            f"9.16-v2 seal component drift: {name}",
        )
        out[name] = p
    return out


def evidence_ref(role: str, path: Path) -> dict[str, Any]:
    req(path.is_file(), f"missing evidence path: {path}")
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256_path(path),
    }


def copy_exact(src: Path, dst: Path) -> None:
    req(src.is_file(), f"missing copy source: {src}")
    atomic_bytes(dst, src.read_bytes())


def scalar_row(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return out


def main() -> None:
    print("=" * 124)
    print("STAGE 9.16-v2R1 — APPEND-ONLY PAPER-READY PACKAGE REPAIR")
    print("PACKAGING/REPORTING ONLY | NO OUTCOME SCAN | NO RNG | NO NEW STATISTICS | NO SCIENTIFIC REEXECUTION")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE =", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    v2_manifest = load_json(P916V2_MANIFEST)
    v2_seal = load_json(P916V2_SEAL)
    v2_components = verify_v2_components(v2_manifest, v2_seal)

    s913 = load_json(P913_SUMMARY)
    s913a = load_json(P913A_SEAL)
    r4seal = load_json(P_R4_SEAL)
    b = load_json(P915B_SUMMARY)
    c = load_json(P915C_REPORT)
    d = load_json(P915D_REPORT)
    e = load_json(P915E_REPORT)
    f = load_json(P915F_SUMMARY)
    g = load_json(P915G_SUMMARY)
    h = load_json(P915H_ADDENDUM)
    d94 = load_json(P94_DIAGNOSTICS)
    c3 = load_json(P_C3_CONTRACT)
    p97if = load_json(P97_INTERFACE)
    p97joint2 = load_json(P97_JOINT_V2)
    p97joint3 = load_json(P97_JOINT_V3)

    # -------------------------------------------------------------------------------------------------
    # A. SCIENTIFIC BOUNDARY GATES.
    # -------------------------------------------------------------------------------------------------
    req(
        s913a["authority"]["frozen_C1_C2_carrier_sha256"]
        == EXPECTED[str(P913A_EXECUTED_SOURCE)],
        "9.13A seal no longer binds executed C1/C2 source",
    )
    req(r4seal["9_13A_seal_sha256"] == EXPECTED[str(P913A_SEAL)], "R4 -> 9.13A seal drift")
    req(r4seal["9_13A_manifest_sha256"] == s913a["authority"]["combined_manifest_sha256"],
        "R4 -> 9.13A manifest binding drift")

    src913a = P913A_EXECUTED_SOURCE.read_text(encoding="utf-8")
    req("missed critical <= floor(C/100), if C>0" in src913a, "executed C2 source semantics drift")
    req(
        '"C2_critical_rule": "missed_critical <= floor(C/100) when C>0"'
        in src913a,
        "executed C2 frozen rule string drift",
    )

    req(
        p97if["historical_v1_absolute_joint_variant"]["status"]
        == "preserved_historical_lineage_not_primary_v1_1_1",
        "Block9.7 historical joint-variant disposition drift",
    )
    req(
        p97if["historical_v1_absolute_joint_variant"]["may_be_reported_as_secondary_sensitivity_without_replacing_primary"]
        is True,
        "Block9.7 historical joint-variant scope drift",
    )
    req(
        p97if["primary_constraints"]["critical_integer_equivalent"]
        == "for critical_count C>0: sum missed critical_beam_sample_counts <= floor(C/100)",
        "executed conditional critical integer form drift",
    )
    req(p97if["primary_solver_inputs"]["Pcrit_cal"]["solver_input"] is False,
        "Pcrit_cal solver-input boundary drift")
    req(
        p97joint3["scientific_authority"]["primary_constraint_semantics"]
        == "joint_absolute_critical_miss",
        "historical joint-v3 semantics drift",
    )

    req(h["H1_reporting_correction"]["confirmatory_status"]
        == "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM", "H1 boundary drift")
    req(g["primary_full_cohort_result_remains_authoritative"] is True, "9.15G primary authority drift")
    req(g["may_override_primary_full_cohort_result"] is False, "9.15G override drift")
    req(c3["scope"]["primary_confirmatory"] == "C1+C2", "C3 scope drift")
    req(c3["scope"]["secondary"] == "C3_PROFILE_SENSITIVITY", "C3 secondary role drift")

    OUT.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------------------------------
    # B. FULL COMMUNICATION TABLE — exact extraction from sealed current 9.13 aggregate summary.
    # -------------------------------------------------------------------------------------------------
    req(
        "H2_H3_H4_communication_point_estimates" in s913,
        "sealed 9.13 summary missing communication point-estimate branch",
    )
    comm = s913["H2_H3_H4_communication_point_estimates"]
    req(isinstance(comm, dict), "communication branch is not a dict")

    comm_rows: list[dict[str, Any]] = []
    for cb in CODEBOOKS:
        req(cb in comm and isinstance(comm[cb], dict), f"missing communication codebook {cb}")
        table = comm[cb]
        missing_p = [x for x in COMM_POLICIES if x not in table]
        req(not missing_p, f"codebook {cb} missing policies: {missing_p}")
        for policy in COMM_POLICIES:
            q = table[policy]
            req(isinstance(q, dict), f"communication row not dict: {cb}/{policy}")
            missing_m = [x for x in COMM_REQUIRED_METRICS if x not in q]
            req(not missing_m, f"communication row {cb}/{policy} missing metrics: {missing_m}")
            row = {"codebook_size": int(cb), "policy": policy}
            row.update(scalar_row(q))
            comm_rows.append(row)

    req(len(comm_rows) == 30, f"expected 30 communication rows, got {len(comm_rows)}")

    comm_json = {
        "schema": "stage9_block916_v2R1_full_communication_table_v1",
        "status": "EXACT_EXTRACTION_FROM_CURRENT_SEALED_9_13_AGGREGATE_NO_RECOMPUTATION",
        "source": evidence_ref("current_9_13_FORMAL_aggregate_summary", P913_SUMMARY),
        "policy_count_per_codebook": 10,
        "codebooks": [16, 32, 64],
        "required_metrics": list(COMM_REQUIRED_METRICS),
        "critical_metric_boundary": (
            "joint_critical_miss_rate is carried from the sealed aggregate table but cannot rescue H2/H4; "
            "there were zero observed ground-truth critical events in the valid FORMAL critical-label population."
        ),
        "table": comm,
    }
    atomic_json(COMM_JSON, comm_json)

    comm_fields = ["codebook_size", "policy"] + sorted(
        {k for row in comm_rows for k in row.keys()} - {"codebook_size", "policy"}
    )
    atomic_csv(COMM_CSV, comm_fields, comm_rows)

    # -------------------------------------------------------------------------------------------------
    # C. CURRENT BLOCK-9.4 BIN-LEVEL ECE15 RELIABILITY TABLE.
    # -------------------------------------------------------------------------------------------------
    req(len(d94["raw"]["ECE_bins"]) == 15, "raw ECE bin count drift")
    req(len(d94["calibrated"]["ECE_bins"]) == 15, "calibrated ECE bin count drift")
    for arm in ("raw", "calibrated"):
        for i, q in enumerate(d94[arm]["ECE_bins"]):
            req(int(q["bin"]) == i, f"{arm} ECE bin ordinal drift at {i}")
            req(
                set(q) == {
                    "absolute_gap", "bin", "count", "empirical_rate",
                    "lower", "mean_probability", "upper",
                },
                f"{arm} ECE bin schema drift at {i}",
            )

    ece_json = {
        "schema": "stage9_block916_v2R1_ece15_reliability_bins_v1",
        "status": "EXACT_COPY_FROM_CURRENT_BLOCK94_DIAGNOSTICS_NO_RECOMPUTATION",
        "source": evidence_ref("current_Block94_CAL_fit_diagnostics", P94_DIAGNOSTICS),
        "ECE_type": "fixed equal-width [0,1]",
        "bin_count": 15,
        "raw_ECE15_equal_width": d94["raw"]["ECE15_equal_width"],
        "calibrated_ECE15_equal_width": d94["calibrated"]["ECE15_equal_width"],
        "raw": d94["raw"]["ECE_bins"],
        "calibrated": d94["calibrated"]["ECE_bins"],
    }
    atomic_json(ECE_JSON, ece_json)

    ece_rows: list[dict[str, Any]] = []
    for arm in ("raw", "calibrated"):
        for q in d94[arm]["ECE_bins"]:
            row = {"arm": arm}
            row.update(q)
            ece_rows.append(row)
    atomic_csv(
        ECE_CSV,
        ["arm", "bin", "lower", "upper", "count", "mean_probability", "empirical_rate", "absolute_gap"],
        ece_rows,
    )

    # -------------------------------------------------------------------------------------------------
    # D. TRUE EXECUTED C2 MATHEMATICAL PROBLEM STATEMENT.
    # -------------------------------------------------------------------------------------------------
    methods_v2 = load_json(v2_components["stage9_block916_v2_methods_problem_statement.json"])

    methods = {
        "schema": "stage9_block916_v2R1_methods_mathematical_problem_v1",
        "status": "PAPER_READY_EXECUTED_C2_MATHEMATICAL_PROBLEM_WITH_LINEAGE_BOUNDARY",
        "scope": methods_v2["frozen_scope"],
        "notation": {
            "M": 4096,
            "z_b": "binary beam-selection decision variable; z_b in {0,1}",
            "p_b": "raw same-sample nominal beam probability mass",
            "u_b_raw": "raw same-sample joint critical-beam probability mass",
            "Pcrit_raw": "sum_b u_b_raw",
            "C": "total raw critical Monte-Carlo sample count",
            "K": "sum_b z_b",
        },
        "objective": {
            "primary": "minimize K = sum_b z_b",
            "lexicographic_tiebreak": [
                "minimize nominal missed sample count",
                "minimize raw critical missed sample count",
                "lexicographically smallest sorted selected_beam_indices",
            ],
        },
        "constraints": {
            "nominal_probability": "sum_b p_b z_b >= 0.95",
            "nominal_exact_M4096": (
                "sum selected beam_sample_counts >= 3892; equivalently "
                "sum missed beam_sample_counts <= 204"
            ),
            "critical_probability_when_positive": (
                "if Pcrit_raw > 0: sum_b u_b_raw z_b >= 0.99 * Pcrit_raw"
            ),
            "critical_exact_integer_when_C_positive": (
                "sum missed critical_beam_sample_counts <= floor(C/100)"
            ),
            "zero_raw_criticality": (
                "if Pcrit_raw == 0 the critical constraint is vacuous; "
                "the nominal constraint remains active"
            ),
        },
        "solver_input_boundary": {
            "Pcrit_raw": "solver input through same-sample raw critical-beam mass",
            "Pcrit_cal_available": True,
            "Pcrit_cal_solver_input": False,
            "Pcrit_cal_role": "descriptive/provenance event-calibration quantity only",
        },
        "executed_authority": {
            "stage913A_continuation_seal": evidence_ref("exact_9_13A_seal_bound_into_R4", P913A_SEAL),
            "executed_C1_C2_source": evidence_ref("frozen_C1_C2_carrier_bound_by_9_13A_seal", P913A_EXECUTED_SOURCE),
            "R4_pretruth_seal": evidence_ref("R4_reuses_9_13A_without_C2_recompute", P_R4_SEAL),
            "R4_source": evidence_ref("R4_pretruth_source", P_R4_SOURCE),
            "interface_contract": evidence_ref("v1_1_1_primary_conditional_semantics_and_historical_lineage", P97_INTERFACE),
        },
        "historical_nonprimary_joint_absolute_branch": {
            "status": "HISTORICAL_LINEAGE_NOT_EXECUTED_PRIMARY_FORMAL_C2",
            "joint_v2": evidence_ref("historical_joint_absolute_amendment_v2", P97_JOINT_V2),
            "joint_v3": evidence_ref("historical_joint_absolute_tiebreak_v3", P97_JOINT_V3),
            "joint_form": "P(Ccrit intersect Bmiss | Dt) <= 0.01; M=4096 integer cap <= 40",
            "paper_rule": (
                "Do not present the joint-absolute <=40 formulation as the C2 formulation executed "
                "in the current 9.13A -> R4 -> FORMAL chain."
            ),
        },
        "novelty_boundary": methods_v2["novelty_boundary"],
    }
    atomic_json(METHODS, methods)

    # -------------------------------------------------------------------------------------------------
    # E. INHERITED CURRENT RESULTS / PROVENANCE, now explicitly pointing to the full tables.
    # -------------------------------------------------------------------------------------------------
    provenance = load_json(v2_components["stage9_block916_v2_dataset_provenance.json"])
    provenance["schema"] = "stage9_block916_v2R1_dataset_provenance_v1"
    provenance["status"] = "PAPER_READY_DATASET_SPLIT_AND_PROVENANCE_REPAIRED_PACKAGE"
    provenance["package_repair_note"] = (
        "Scientific population/provenance is unchanged from 9.16-v2; only paper-ready packaging is repaired."
    )
    atomic_json(PROVENANCE, provenance)

    calibration = load_json(v2_components["stage9_block916_v2_calibration_tables.json"])
    calibration["schema"] = "stage9_block916_v2R1_calibration_tables_v1"
    calibration["status"] = "PAPER_READY_CALIBRATION_TABLES_WITH_BIN_LEVEL_ECE15"
    calibration["critical_event_calibration_CAL"]["ECE15_bin_table_artifact"] = {
        "json_path": str(ECE_JSON),
        "csv_path": str(ECE_CSV),
    }
    calibration["critical_event_calibration_CAL"]["bin_level_reliability_packaged"] = True
    atomic_json(CALIBRATION, calibration)

    results = load_json(v2_components["stage9_block916_v2_paper_ready_results.json"])
    results["schema"] = "stage9_block916_v2R1_paper_ready_results_v1"
    results["status"] = "PAPER_READY_RESULTS_REPAIRED_PACKAGE_NO_NEW_ANALYSIS"
    results["full_communication_table"] = {
        "json_path": str(COMM_JSON),
        "csv_path": str(COMM_CSV),
        "source_9_13_summary_sha256": EXPECTED[str(P913_SUMMARY)],
        "policy_count_per_codebook": 10,
        "codebooks": [16, 32, 64],
        "new_statistics": False,
    }
    results["executed_C2_math_artifact"] = str(METHODS)
    atomic_json(RESULTS, results)

    robustness = load_json(v2_components["stage9_block916_v2_robustness_failure_latency.json"])
    robustness["schema"] = "stage9_block916_v2R1_robustness_failure_latency_v1"
    robustness["status"] = "PAPER_READY_ROBUSTNESS_FAILURE_LATENCY_WITH_ADB_REPORTING_BOUNDARY"
    robustness["ADB_reaction_margin_reporting_boundary"] = {
        "reaction_margin_distribution_status": "NOT_MATERIALIZED_AS_CURRENT_SEALED_AGGREGATE",
        "C_late_pairwise_H5_results_remain_authoritative": True,
        "post_FORMAL_distribution_construction_allowed": False,
    }
    atomic_json(ROBUSTNESS, robustness)

    adb_boundary = {
        "schema": "stage9_block916_v2R1_adb_reporting_boundary_v1",
        "status": "FAIL_CLOSED_ADB_REPORTING_BOUNDARY",
        "H5_current_authority": evidence_ref("current_H1_H6_statistics_summary", P915B_SUMMARY),
        "H5_secondary_profile_sensitivity": b["H5"],
        "H6_current_authority": b["H6"],
        "reaction_margin_distribution": {
            "status": "NOT_MATERIALIZED_AS_CURRENT_SEALED_AGGREGATE",
            "representative_preoutcome_action_leaf_audit": "NO reaction_margin / C_late leaf in audited representative sealed action JSON",
            "new_post_FORMAL_distribution": False,
            "paper_rule": (
                "Report the frozen H5 P(C_late)-based violation comparisons and H6 side-effect metrics. "
                "Do not claim or construct a separate reaction-margin distribution."
            ),
        },
        "integrity": {
            "new_outcome_scan": False,
            "new_statistics": False,
            "C3_rerun": False,
        },
    }
    atomic_json(ADB_BOUNDARY, adb_boundary)

    # -------------------------------------------------------------------------------------------------
    # F. CLAIM-EVIDENCE MATRIX — every row gets explicit artifact path + SHA traceability.
    # -------------------------------------------------------------------------------------------------
    claims_v2 = load_json(v2_components["stage9_block916_v2_claim_evidence_matrix.json"])

    refs = {
        "METHOD_SHARED_POSTERIOR_CRITICALITY": [
            evidence_ref("executed_C1_C2_source", P913A_EXECUTED_SOURCE),
            evidence_ref("R4_pretruth_seal", P_R4_SEAL),
            evidence_ref("C3_secondary_execution_contract", P_C3_CONTRACT),
        ],
        "H1_CONFIRMATORY": [
            evidence_ref("preoutcome_H1_not_tested_seal", P_H1_OLD_SEAL),
            evidence_ref("reporting_correction", P915H_ADDENDUM),
        ],
        "H1_NUMERICAL": [
            evidence_ref("current_FORMAL_statistics", P915B_SUMMARY),
            evidence_ref("mandatory_22_scene_sensitivity", P915G_SUMMARY),
            evidence_ref("reporting_correction", P915H_ADDENDUM),
        ],
        "H2_CRITICAL_RELIABILITY": [
            evidence_ref("current_FORMAL_statistics", P915B_SUMMARY),
            evidence_ref("mandatory_22_scene_sensitivity", P915G_SUMMARY),
        ],
        "H3_FULL": [
            evidence_ref("current_FORMAL_statistics", P915B_SUMMARY),
            evidence_ref("full_communication_aggregate", P913_SUMMARY),
        ],
        "H4_MATCHED_CRITICAL": [
            evidence_ref("current_FORMAL_statistics", P915B_SUMMARY),
            evidence_ref("preoutcome_H4_authority_correction", P_H4_REPORT),
            evidence_ref("reporting_correction", P915H_ADDENDUM),
        ],
        "H5_C3_PROFILE": [
            evidence_ref("current_FORMAL_statistics", P915B_SUMMARY),
            evidence_ref("human_profile_robustness", P915C_REPORT),
            evidence_ref("mandatory_22_scene_sensitivity", P915G_SUMMARY),
        ],
        "H6_CORE_SIDE_EFFECT": [
            evidence_ref("current_FORMAL_statistics", P915B_SUMMARY),
            evidence_ref("mandatory_22_scene_sensitivity", P915G_SUMMARY),
        ],
        "CALIBRATION_ROBUSTNESS": [
            evidence_ref("robustness_report", P915C_REPORT),
            evidence_ref("current_Block94_diagnostics", P94_DIAGNOSTICS),
        ],
        "FAILURE_CASE_ROBUSTNESS": [
            evidence_ref("failure_case_disposition", P915D_REPORT),
        ],
        "LATENCY_REALTIME": [
            evidence_ref("latency_resources_report", P915E_REPORT),
            evidence_ref("reporting_correction", P915H_ADDENDUM),
        ],
        "NO_SECOND_LEARNED_RISK_NETWORK": [
            evidence_ref("C3_secondary_execution_contract", P_C3_CONTRACT),
            evidence_ref("executed_C2_source", P913A_EXECUTED_SOURCE),
        ],
        "DEEPSENSE": [
            evidence_ref("current_DeepSense_boundary_seal", P914_SEAL),
            evidence_ref("Stage915_closure_boundary", P915F_SUMMARY),
        ],
        "CRASH_REDUCTION": [
            evidence_ref("C3_secondary_execution_contract", P_C3_CONTRACT),
            evidence_ref("reporting_correction", P915H_ADDENDUM),
        ],
        "NOVEL_PREDICTOR_OR_CALIBRATOR": [
            evidence_ref("event_calibration_contract", P94_CONTRACT),
            evidence_ref("reporting_correction", P915H_ADDENDUM),
            evidence_ref("executed_C2_source", P913A_EXECUTED_SOURCE),
        ],
    }

    repaired_claim_rows = []
    for row in claims_v2["claims"]:
        cid = row["claim_id"]
        req(cid in refs, f"no explicit repair evidence mapping for claim {cid}")
        q = dict(row)
        q["evidence_refs"] = refs[cid]
        repaired_claim_rows.append(q)

    claims = {
        "schema": "stage9_block916_v2R1_claim_evidence_matrix_v1",
        "status": "STRICT_FINAL_CLAIM_EVIDENCE_AUDIT_WITH_EXPLICIT_SHA_TRACEABILITY",
        "rule": claims_v2["rule"],
        "claims": repaired_claim_rows,
        "safe_central_story": claims_v2["safe_central_story"],
    }
    atomic_json(CLAIMS, claims)

    # -------------------------------------------------------------------------------------------------
    # G. LIMITATIONS — preserve v2, add explicit reaction-margin and math-lineage limitations.
    # -------------------------------------------------------------------------------------------------
    limitations = load_json(v2_components["stage9_block916_v2_limitations.json"])
    limitations["schema"] = "stage9_block916_v2R1_limitations_v1"
    limitations["status"] = "PAPER_READY_LIMITATIONS_REPAIRED_PACKAGE"

    existing_ids = {x["id"] for x in limitations["limitations"]}
    if "REACTION_MARGIN_DISTRIBUTION" not in existing_ids:
        limitations["limitations"].append({
            "id": "REACTION_MARGIN_DISTRIBUTION",
            "text": (
                "No current sealed aggregate reaction-margin distribution was materialized. "
                "The frozen H5 P(C_late)-based pairwise comparisons remain reportable, but no "
                "post-FORMAL reaction-margin distribution is constructed."
            ),
        })
    if "C2_MATH_LINEAGE" not in existing_ids:
        limitations["limitations"].append({
            "id": "C2_MATH_LINEAGE",
            "text": (
                "The server contains a historical joint-absolute Block-9.7 branch. The C2 formulation "
                "executed in the 9.13A -> R4 -> FORMAL chain is the conditional 99% critical-coverage "
                "constraint missed<=floor(C/100); manuscript equations must follow the executed chain."
            ),
        })
    atomic_json(LIMITATIONS, limitations)

    # -------------------------------------------------------------------------------------------------
    # H. SUPERSESSION / VERSION MAP — make the server-version ambiguity explicit.
    # -------------------------------------------------------------------------------------------------
    supersession = {
        "schema": "stage9_block916_v2R1_supersession_map_v1",
        "status": "CURRENT_REPORTING_LINEAGE_AND_HISTORICAL_BRANCH_MAP",
        "scientific_current_chain": {
            "stage913_final_seal": evidence_ref("current_FORMAL_outcomes_seal", P913_SEAL),
            "stage914_v2_seal": evidence_ref("current_external_boundary_seal", P914_SEAL),
            "stage915F_primary_closure": evidence_ref("current_9_15_primary_closure", P915F_SEAL),
            "stage915G_secondary_recovery_sensitivity": evidence_ref("mandatory_secondary_sensitivity", P915G_SEAL),
            "stage915H_reporting_correction": evidence_ref("provenance_wording_addendum", P915H_SEAL),
        },
        "paper_package_lineage": {
            "stage916_v2": {
                "manifest": evidence_ref("scientifically_compatible_but_incomplete_v2_manifest", P916V2_MANIFEST),
                "seal": evidence_ref("scientifically_compatible_but_incomplete_v2_seal", P916V2_SEAL),
                "current_manuscript_authority_after_v2R1_independent_certification": False,
                "reason": "independent audit found packaging/completeness defects, not scientific-result mismatch",
            },
            "stage916_v2R1": {
                "role": "APPEND_ONLY_PACKAGING_REPAIR",
                "supersedes_v2_for_manuscript_only_after_independent_certification": True,
            },
        },
        "C2_mathematical_lineage": {
            "executed_primary_FORMAL_semantics": {
                "name": "conditional_99pct_critical_coverage_when_Pcrit_raw_positive",
                "source": evidence_ref("executed_9_13A_C1_C2_source", P913A_EXECUTED_SOURCE),
                "9_13A_seal": evidence_ref("seal_binding_executed_source", P913A_SEAL),
                "equation": "sum_b u_b_raw z_b >= 0.99 * Pcrit_raw",
                "integer_equivalent": "missed_critical_sample_count <= floor(C/100)",
            },
            "historical_joint_absolute_branch": {
                "status": "PRESERVED_HISTORICAL_LINEAGE_NOT_PRIMARY_EXECUTED_FORMAL_C2",
                "joint_v2": evidence_ref("historical_joint_absolute_v2", P97_JOINT_V2),
                "joint_v3": evidence_ref("historical_joint_absolute_v3", P97_JOINT_V3),
                "historical_equation": "P(Ccrit intersect Bmiss | Dt) <= 0.01",
                "historical_M4096_cap": 40,
            },
        },
        "rule": (
            "Choose authorities by explicit binding into the executed current chain, not by filename recency "
            "or version-number appearance."
        ),
    }
    atomic_json(SUPERSESSION, supersession)

    # -------------------------------------------------------------------------------------------------
    # I. CANONICAL SOURCE / CONFIG MANIFEST — current chain only, not all historical server files.
    # -------------------------------------------------------------------------------------------------
    old_source_manifest = load_json(v2_components["stage9_block916_v2_source_config_manifest.json"])
    source_entries_by_path: dict[str, dict[str, Any]] = {}

    for q in old_source_manifest["entries"]:
        pp = Path(q["path"])
        req(pp.is_file(), f"v2 canonical source entry disappeared: {pp}")
        req(sha256_path(pp) == q["sha256"], f"v2 source entry drift: {pp}")
        source_entries_by_path[str(pp.resolve())] = {
            "path": str(pp.resolve()),
            "sha256": q["sha256"],
            "bytes": pp.stat().st_size,
            "binding_status": "INHERITED_FROM_SEALED_9_16_V2_SOURCE_MANIFEST",
        }

    canonical_additions = [
        (
            P913A_EXECUTED_SOURCE,
            "EXECUTION_BOUND_BY_9_13A_SEAL",
        ),
        (
            P_R4_SOURCE,
            "EXACT_CURRENT_R4_SOURCE_HASH_GATED",
        ),
        (
            S9 / "scripts/run_stage9_block913_v1_3_FINAL_EVALUATOR_H1_H6_R3.py",
            "CURRENT_SERVER_CANONICAL_FINAL_EVALUATOR_SOURCE_SNAPSHOT",
        ),
        (
            S9 / "scripts/run_stage9_block913_v1_3_FINAL_EVALUATOR_H1_H6_R4_RECOVERY.py",
            "CURRENT_SERVER_CANONICAL_RECOVERY_SOURCE_SNAPSHOT",
        ),
        (
            S9 / "scripts/run_stage9_block913_v1_3_FINAL_EVALUATOR_H1_H6_R5_RECOVERY2.py",
            "CURRENT_SERVER_CANONICAL_RECOVERY2_SOURCE_SNAPSHOT",
        ),
        (
            S9 / "scripts/run_stage9_block913_phase3_DIRECT_closure.py",
            "CURRENT_SERVER_CANONICAL_C3_PREOUTCOME_CLOSURE_SOURCE_SNAPSHOT",
        ),
        (
            S9 / "scripts/run_stage9_block914_v2a_external_deepsense_boundary_freeze.py",
            "CURRENT_SERVER_CANONICAL_9_14_V2_SOURCE_SNAPSHOT",
        ),
        (
            S9 / "scripts/run_stage9_block915a_v2R4_h1_status_censoring_contract_repair.py",
            "CURRENT_SERVER_CANONICAL_9_15A_V2R4_SOURCE_SNAPSHOT",
        ),
        (
            P97_INTERFACE,
            "CURRENT_PRIMARY_C2_SEMANTICS_AND_HISTORICAL_LINEAGE_CONFIG",
        ),
        (
            P97_JOINT_V2,
            "HISTORICAL_NONPRIMARY_C2_BRANCH_CONFIG",
        ),
        (
            P97_JOINT_V3,
            "HISTORICAL_NONPRIMARY_C2_BRANCH_CONFIG",
        ),
        (
            Path(__file__).resolve(),
            "CURRENT_9_16_V2R1_PACKAGING_REPAIR_RUNNER",
        ),
    ]

    for pp, status in canonical_additions:
        req(pp.is_file(), f"missing canonical source/config: {pp}")
        source_entries_by_path[str(pp.resolve())] = {
            "path": str(pp.resolve()),
            "sha256": sha256_path(pp),
            "bytes": pp.stat().st_size,
            "binding_status": status,
        }

    source_manifest = {
        "schema": "stage9_block916_v2R1_canonical_source_config_manifest_v1",
        "status": "CANONICAL_CURRENT_CHAIN_ONLY_NOT_ALL_DISCOVERABLE_SERVER_VERSIONS",
        "selection_rule": (
            "Include the explicit current scientific chain plus configurations required to explain "
            "current-vs-historical semantics. Do not include every historical/failed/intermediate Stage-9 script."
        ),
        "entries": sorted(source_entries_by_path.values(), key=lambda x: x["path"]),
    }
    atomic_json(SOURCE_MANIFEST, source_manifest)

    # -------------------------------------------------------------------------------------------------
    # J. AUTHORITY MANIFEST — exact scientific/reporting inputs.
    # -------------------------------------------------------------------------------------------------
    authority_paths = [
        P913_SUMMARY, P913_SEAL, P913A_SEAL, P_R4_SEAL, P914_SEAL,
        P915A_EVIDENCE, P915A_EST, P915A_CONTRACT, P915A_SEAL,
        P915B_SUMMARY, P915B_SEAL, P915C_REPORT, P915C_SEAL,
        P915D_REPORT, P915D_SEAL, P915E_REPORT, P915E_SEAL,
        P915F_SUMMARY, P915F_SEAL, P915G_SUMMARY, P915G_SEAL,
        P915H_ADDENDUM, P915H_SEAL, P_H1_OLD_SEAL, P_H4_REPORT, P_H4_SEAL,
        P94_CONTRACT, P94_DIAGNOSTICS, P94_SEAL, P_C3_CONTRACT,
        P97_INTERFACE, P97_JOINT_V2, P97_JOINT_V3,
        P916V2_MANIFEST, P916V2_SEAL,
    ]
    authority_manifest = {
        "schema": "stage9_block916_v2R1_authority_manifest_v1",
        "status": "EXACT_CURRENT_AUTHORITY_MANIFEST",
        "entries": [
            {
                "path": str(pp),
                "sha256": sha256_path(pp),
                "bytes": pp.stat().st_size,
            }
            for pp in authority_paths
        ],
    }
    atomic_json(AUTHORITY_MANIFEST, authority_manifest)

    # -------------------------------------------------------------------------------------------------
    # K. COPY EXISTING PAPER TABLES BYTE-FOR-BYTE FROM SCIENTIFICALLY COMPATIBLE 9.16-v2.
    # -------------------------------------------------------------------------------------------------
    copy_exact(v2_components["stage9_block916_v2_table_H3_resource.csv"], H3_CSV)
    copy_exact(v2_components["stage9_block916_v2_table_H5_profile_sensitivity.csv"], H5_CSV)
    copy_exact(v2_components["stage9_block916_v2_table_H6_side_effects.csv"], H6_CSV)
    copy_exact(v2_components["stage9_block916_v2_table_calibration.csv"], CAL_SUMMARY_CSV)

    # -------------------------------------------------------------------------------------------------
    # L. SUMMARY.
    # -------------------------------------------------------------------------------------------------
    h1p = b["H1"]["point_estimates"]
    md = f"""# Stage 9.16-v2R1 — Paper-Ready Package Repair

## Repair scope

This package is an append-only reporting/packaging repair of the scientifically compatible Stage-9.16-v2 package.
It performs no outcome scan, RNG, bootstrap, new statistic, model run, policy run, solver run, threshold change,
baseline change, or hypothesis redefinition.

## Executed C2 mathematical formulation

The C2 formulation actually executed in the frozen 9.13A -> R4 -> FORMAL chain is:

- binary decision: `z_b in {{0,1}}`
- objective: `minimize K = sum_b z_b`
- nominal constraint: `sum_b p_b z_b >= 0.95`
- critical constraint when `Pcrit_raw > 0`: `sum_b u_b_raw z_b >= 0.99 * Pcrit_raw`
- exact critical integer form: `missed critical samples <= floor(C/100)`
- `Pcrit_cal` is descriptive/provenance only and is **not** a C2 solver input.

The server's joint-absolute `P(Ccrit ∩ Bmiss | Dt) <= 0.01` / `<=40 of 4096` branch is preserved as
historical non-primary lineage and must not be presented as the executed FORMAL C2 formulation.

## Full communication table

A dedicated 30-row table is packaged for all 10 frozen policies across codebooks 16/32/64.
It is copied exactly from the current sealed Stage-9.13 aggregate summary and includes coverage, joint critical
miss, mean K, beam-gain loss, received-power loss, probing overhead, outage, effective rate, and the other
already-materialized aggregate communication quantities.

No communication statistic is recomputed here.

## H1

Confirmatory status: **NOT TESTED** (`NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM`).

Later non-confirmatory numerical evidence:
- RAW macro absolute coverage error: {h1p['macro_abs_coverage_error_RAW']}
- CALIBRATED macro absolute coverage error: {h1p['macro_abs_coverage_error_CALIBRATED']}
- CAL-RAW delta: {h1p['primary_delta_CALIBRATED_minus_RAW']}
- paired bootstrap 95% CI: {b['H1']['primary_delta_CI95']}

The mandatory exact 22-scene technical-recovery exclusion remains secondary and cannot replace the full cohort.

## Critical events / H2-H4

Observed FORMAL ground-truth critical events: **0 / 24,917 valid critical labels**.
Zero observed events are not interpreted as zero risk.

- H2: not evaluable.
- H3: resource component only; full reliability claim is not established.
- H4: not evaluable.

The communication table's zero aggregate joint-critical-miss entries do not rescue these critical-conditioned claims.

## C3 / H5-H6

H5 remains a secondary profile sensitivity: FAST 0/2 clear reductions, NOMINAL 2/2, SLOW 2/2.
There is no global 6/6 H5 success claim.

H6 retains the frozen Stage-6 core-gate interpretation.

No current sealed aggregate reaction-margin distribution was materialized. No post-FORMAL reaction-margin
distribution is constructed in this repair; only the already-sealed H5 `P(C_late)` comparisons are reported.

## Event calibration

Current Block-9.4 authority is 15-bin equal-width ECE. This package includes the exact raw and calibrated
15-bin reliability tables from the sealed CAL diagnostics, in JSON and CSV form.

## Version-control / provenance rule

Authorities are selected by explicit binding into the current executed chain, not by the most recent-looking
filename or version number. Historical, failed, superseded, and intermediate Stage-9 scripts are not bulk-added
to the canonical source manifest.

## Manuscript authority

Stage-9.16-v2R1 is a repaired candidate package. It should supersede 9.16-v2 for manuscript writing only after
an independent read-only certification audit passes.
"""
    atomic_text(SUMMARY_MD, md)

    # -------------------------------------------------------------------------------------------------
    # M. MANIFEST + SEAL. No self-certification: independent audit is explicitly still required.
    # -------------------------------------------------------------------------------------------------
    component_paths = [
        METHODS, PROVENANCE, CALIBRATION, ECE_JSON, ECE_CSV,
        COMM_JSON, COMM_CSV, RESULTS, ROBUSTNESS, ADB_BOUNDARY,
        CLAIMS, LIMITATIONS, SUPERSESSION, SOURCE_MANIFEST, AUTHORITY_MANIFEST,
        H3_CSV, H5_CSV, H6_CSV, CAL_SUMMARY_CSV, SUMMARY_MD,
    ]
    components = {
        pp.name: {
            "path": str(pp),
            "sha256": sha256_path(pp),
            "bytes": pp.stat().st_size,
        }
        for pp in component_paths
    }

    manifest = {
        "schema": "stage9_block916_v2R1_paper_ready_manifest_v1",
        "status": "PACKAGING_REPAIR_COMPLETE_AWAITING_INDEPENDENT_CERTIFICATION",
        "supersedes_for_manuscript_after_independent_certification": {
            "stage916_v2_manifest_sha256": EXPECTED[str(P916V2_MANIFEST)],
            "stage916_v2_seal_sha256": EXPECTED[str(P916V2_SEAL)],
        },
        "scientific_results_changed": False,
        "repair_scope": [
            "executed_C2_mathematical_problem_statement",
            "full_10_policy_3_codebook_communication_table",
            "bin_level_ECE15_reliability_table",
            "reaction_margin_fail_closed_reporting_boundary",
            "explicit_path_SHA_claim_evidence_traceability",
            "canonical_current_source_config_manifest",
            "manifest_final_seal_state_fix",
            "conditional_vs_joint_C2_lineage_supersession_map",
        ],
        "components": components,
        "package_completeness": {
            "frozen_methods_specification": True,
            "formal_mathematical_problem_statement": True,
            "dataset_split_provenance": True,
            "critical_event_calibration_summary": True,
            "critical_event_ECE15_bin_table": True,
            "full_communication_table_all_frozen_baselines": True,
            "communication_metrics": True,
            "ADB_tables": True,
            "reaction_margin_distribution": "NOT_MATERIALIZED_AS_CURRENT_SEALED_AGGREGATE_NO_POST_FORMAL_CONSTRUCTION",
            "FAST_NOMINAL_SLOW_sensitivity": True,
            "paired_CIs": True,
            "failure_case_disposition": True,
            "latency_resources": True,
            "claim_evidence_matrix_with_explicit_SHA_paths": True,
            "limitations": True,
            "canonical_source_config_manifest": True,
            "authority_manifest": True,
            "final_reproducibility_seal": "EMITTED_AFTER_MANIFEST_AND_BINDS_THIS_MANIFEST",
            "independent_certification": "REQUIRED_AFTER_THIS_RUNNER",
        },
        "integrity": {
            "new_outcome_scan": False,
            "future_GT_read": False,
            "new_RNG": False,
            "new_bootstrap": False,
            "new_statistics": False,
            "new_model_policy_solver_execution": False,
            "retuning": False,
            "threshold_change": False,
            "baseline_change": False,
            "H1_confirmatory_reactivation": False,
            "reaction_margin_post_FORMAL_construction": False,
        },
    }
    atomic_json(MANIFEST, manifest)
    manifest_sha = sha256_path(MANIFEST)

    seal = {
        "schema": "stage9_block916_v2R1_final_reproducibility_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_916_V2R1_PACKAGING_REPAIR_AWAITING_INDEPENDENT_CERTIFICATION",
        "paper_ready_manifest_sha256": manifest_sha,
        "component_hashes": {name: meta["sha256"] for name, meta in components.items()},
        "stage913_final_seal_sha256": EXPECTED[str(P913_SEAL)],
        "stage913A_executed_C1_C2_source_sha256": EXPECTED[str(P913A_EXECUTED_SOURCE)],
        "stage913A_continuation_seal_sha256": EXPECTED[str(P913A_SEAL)],
        "R4_pretruth_seal_sha256": EXPECTED[str(P_R4_SEAL)],
        "stage914_v2_external_boundary_seal_sha256": EXPECTED[str(P914_SEAL)],
        "stage915F_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
        "stage915G_secondary_recovery_sensitivity_seal_sha256": EXPECTED[str(P915G_SEAL)],
        "stage915H_provenance_wording_seal_sha256": EXPECTED[str(P915H_SEAL)],
        "prior_stage916_v2_manifest_sha256": EXPECTED[str(P916V2_MANIFEST)],
        "prior_stage916_v2_seal_sha256": EXPECTED[str(P916V2_SEAL)],
        "executed_C2_critical_semantics": "conditional_99pct_Pcrit_raw",
        "executed_C2_integer_critical_rule": "missed_critical <= floor(C/100) when C>0",
        "historical_joint_absolute_branch_is_executed_primary_FORMAL_C2": False,
        "H1_confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "primary_full_cohort_scene_count": 26209,
        "mandatory_secondary_recovery_exclusion_scene_count": 22,
        "reaction_margin_distribution_materialized_as_current_sealed_aggregate": False,
        "independent_certification_required": True,
        "new_outcome_scan": False,
        "future_GT_read": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "new_statistics": False,
        "new_model_policy_solver_execution": False,
        "retuning": False,
        "claim_rescue_experiment": False,
        "next": "INDEPENDENT_READ_ONLY_9_16_V2R1_CERTIFICATION_AUDIT_BEFORE_MANUSCRIPT",
    }
    atomic_json(SEAL, seal)

    print("EXECUTED_C2_MATH = CONDITIONAL_99PCT_PCRIT_RAW")
    print("EXECUTED_C2_INTEGER_RULE = MISSED_CRITICAL_LE_FLOOR_C_DIV_100")
    print("HISTORICAL_JOINT_ABSOLUTE_BRANCH_PRIMARY_FORMAL = FALSE")
    print("FULL_10_POLICY_COMMUNICATION_TABLE = PACKAGED")
    print("COMMUNICATION_ROW_COUNT = 30")
    print("ECE15_BIN_LEVEL_TABLE = PACKAGED")
    print("ECE15_ROW_COUNT = 30")
    print("REACTION_MARGIN_DISTRIBUTION = NOT_MATERIALIZED_NO_POST_FORMAL_CONSTRUCTION")
    print("CLAIM_EVIDENCE_EXPLICIT_SHA_PATHS = COMPLETE")
    print("CANONICAL_SOURCE_CONFIG_MANIFEST = COMPLETE")
    print("MANIFEST_FINAL_SEAL_STATE = EMITTED_AFTER_MANIFEST_AND_BINDS_THIS_MANIFEST")
    print("SCIENTIFIC_RESULTS_CHANGED = FALSE")
    print("NEW_OUTCOME_SCAN = FALSE")
    print("FUTURE_GT_READ = FALSE")
    print("NEW_RNG = FALSE")
    print("NEW_BOOTSTRAP = FALSE")
    print("NEW_STATISTICS = FALSE")
    print("SCIENTIFIC_REEXECUTION = FALSE")
    print("STAGE 9.16-v2R1 PACKAGING REPAIR = COMPLETE")
    print("MANIFEST_SHA256 =", manifest_sha)
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT = INDEPENDENT_READ_ONLY_9_16_V2R1_CERTIFICATION_AUDIT_BEFORE_MANUSCRIPT")
    print("=" * 124)


if __name__ == "__main__":
    main()

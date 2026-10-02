#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"
S9 = ROOT / "iscai_stage9"

OUT = S9 / "artifacts/block914_v2_external_deepsense_boundary"

# -------------------------------------------------------------------------------------------------
# Current Stage 9 FORMAL authority (completed H1-H6 materialization).
# -------------------------------------------------------------------------------------------------
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
B913_MANIFEST = B913 / "stage9_block913_v1_3_FINAL_H1_H6_manifest.jsonl"
B913_SUMMARY = B913 / "stage9_block913_v1_3_FINAL_H1_H6_summary.json"
B913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

EXPECTED_913 = {
    str(B913_MANIFEST): "752600ddf126cabd87010d8277b7fe040d18b3e8d8f2490adcc2e9ca0ceb63c4",
    str(B913_SUMMARY): "292d515a3f52c82ad08460e1122a805d236a8aeaefa82271ff84664c2cddfe60",
    str(B913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
}

# -------------------------------------------------------------------------------------------------
# Immutable Stage 8 DeepSense authority.
# Only small JSON/seal/reporting artifacts are opened here.
# Raw LiDAR/mmWave samples and formal measured-power records are NOT reopened.
# -------------------------------------------------------------------------------------------------
B82B = S8 / "artifacts/block82b_canonical_corpus_schema_split"
SPLITS = B82B / "frozen_split_roles_and_disjointness.json"
REFS = B82B / "resolved_reference_integrity.json"
SEAL82B = B82B / "stage8_block82b_canonical_corpus_schema_split_seal.json"

B88C = S8 / "artifacts/block88c_corrected_formal_measured_deepsense_evaluation"
SUMMARY88C = B88C / "formal_metrics_summary.json"
SEAL88C = B88C / "stage8_block88c_corrected_formal_measured_deepsense_evaluation_seal.json"

B810 = S8 / "artifacts/block810_final_reporting_reproducibility"
SEP810 = B810 / "optical_mmwave_reporting_separation.json"
TABLE810 = B810 / "deepsense_measured_mmwave_final_table.json"
MATRIX810 = B810 / "stage8_completion_matrix.json"
SEAL810 = B810 / "stage8_block810_final_reproducibility_seal.json"
SUPER810 = B810 / "stage8_pdf_compliance_complete_superseding_seal.json"

EXPECTED_S8 = {
    str(SPLITS): "67e7624cdb633bf83c0fedab19b6d4c5b41d70ccf0d69c9875873c24b3fe20ce",
    str(REFS): "11aa52287c0e2cb2fee7d8e0f83590517d97aefa72cb276d516cd8547ab6f098",
    str(SEAL82B): "c84eb7ef45d16a1feb3ef068bdaf498e5e4298514f3c4807ffe10db9b1112ba3",
    str(SUMMARY88C): "8f29fd2b3882ffb28b66fdef87a53490eeaf64fe3b501d788495710a989fa0ac",
    str(SEAL88C): "f287399f7ddc6070c45f5c42502ae90fc4db0622f6cf42a845aa94d2b8ff8cb4",
    str(SEP810): "19170f933ddf604009cf1a1587eb2c0d1dc0785a8eeafba0eb5033e245ff0fff",
    str(TABLE810): "afb4736f2cb9fdf780a2662d37e41a368b0f5dbe54dc451c6c534b1d26359da5",
    str(MATRIX810): "8f2a778a302ed3ab16546affeba6ec2596eb01c53271c5185d44fe9e5cabd3ed",
    str(SEAL810): "c1c9c06f7ccc80160247ca9eeadd8d64312201168ae1eae26fa6b924ca8b2c17",
    str(SUPER810): "497d94565019238e71afa3d1ff05c45d42ba56a507aa7f61706f97bd4df313cc",
}

# Historical Stage8 formal measured-power record hash is bound THROUGH the frozen seals;
# the record file itself is deliberately not opened or hashed in this Stage9 block.
EXPECTED_STAGE8_FORMAL_RECORDS_SHA = (
    "1984d777f8e649089c7f14ed80df8cd105838496ca03c70adce54fe124dd164b"
)

REPORT = OUT / "stage9_block914_v2_external_boundary_report.json"
MANIFEST = OUT / "stage9_block914_v2_external_boundary_manifest.json"
SEAL = OUT / "stage9_block914_v2_external_boundary_seal.json"


class FailClosed(RuntimeError):
    pass


def req(c: bool, msg: str) -> None:
    if not c:
        raise FailClosed(msg)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
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
        req(path.read_bytes() == data, f"existing immutable artifact content drift: {path}")
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


def exact_hash_gate(expected: dict[str, str]) -> None:
    for raw, wanted in expected.items():
        p = Path(raw)
        req(p.is_file(), f"missing authority artifact: {p}")
        got = sha256_path(p)
        req(got == wanted, f"SHA256 drift {p}: {got} != {wanted}")
        print("EXACT PASS", p)


def main() -> None:
    print("=" * 120)
    print("STAGE 9.14-v2A — EXTERNAL COMMUNICATION / DEEPSENSE BOUNDARY FREEZE")
    print("READ-ONLY AUTHORITY AUDIT")
    print("NO DEEPSENSE SAMPLE READ | NO RAW POWER READ | NO MODEL FORWARD | NO POLICY EXECUTION | NO METRICS")
    print("=" * 120)

    req(not SEAL.exists(), f"9.14-v2A already sealed: {SEAL}")

    print("\n===== A. CURRENT STAGE9 9.13 AUTHORITY =====")
    exact_hash_gate(EXPECTED_913)

    s913 = load_json(B913_SEAL)
    q913 = load_json(B913_SUMMARY)
    req(
        s913.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK913_V1_3_H1_H6_FORMAL_OUTCOMES",
        "Stage9 9.13 final seal status drift",
    )
    req(
        q913.get("status") == "FORMAL_RAW_OUTCOMES_H1_H6_COMPLETE",
        "Stage9 9.13 summary status drift",
    )
    req(
        s913.get("bootstrap_or_significance_performed_in_9_13") is False,
        "9.13 unexpectedly contains bootstrap/significance inference",
    )
    req(
        s913.get("outcome_based_tuning_after_failed_attempt") is False,
        "9.13 outcome-based tuning flag drift",
    )
    req(
        s913.get("technical_recovery_documented") is True,
        "9.13 technical recovery provenance missing",
    )
    print("STAGE9_9_13_FINAL = PASS")
    print("STAGE9_9_13_STATISTICAL_INFERENCE = DEFERRED")

    print("\n===== B. IMMUTABLE STAGE8 DEEPSENSE AUTHORITY =====")
    exact_hash_gate(EXPECTED_S8)

    splits = load_json(SPLITS)
    refs = load_json(REFS)
    s82 = load_json(SEAL82B)
    sum88 = load_json(SUMMARY88C)
    s88 = load_json(SEAL88C)
    sep = load_json(SEP810)
    table = load_json(TABLE810)
    matrix = load_json(MATRIX810)
    s810 = load_json(SEAL810)
    super810 = load_json(SUPER810)

    expected_roles = {
        "official_dev_train": "TRAIN",
        "official_dev_val": "CALIBRATION",
        "official_dev_test": "FORMAL_EVALUATION",
        "challenge_dataset": "EXCLUDED_FROM_LOCAL_CORE_HIDDEN_GT_SUBMISSION_ONLY",
    }
    req(splits.get("status") == "FROZEN_SPLIT_ROLES_AND_DISJOINTNESS", "Stage8 split status drift")
    req(splits.get("role_mapping") == expected_roles, "Stage8 split role mapping drift")
    req(splits.get("exact_lidar_reference_disjointness") == "PASS_ZERO_OVERLAP", "LiDAR split overlap drift")
    req(splits.get("exact_mmwave_reference_disjointness") == "PASS_ZERO_OVERLAP", "mmWave split overlap drift")
    req(splits.get("formal_test_used_for_model_selection") is False, "Stage8 formal test selection boundary drift")

    req(refs.get("status") == "FROZEN_REFERENCE_INTEGRITY", "Stage8 reference-integrity status drift")
    rs = refs.get("splits")
    req(isinstance(rs, list) and len(rs) == 6, "Stage8 canonical split count drift")
    formal = [x for x in rs if x.get("role") == "FORMAL_EVALUATION"]
    train = [x for x in rs if x.get("role") == "TRAIN"]
    cal = [x for x in rs if x.get("role") == "CALIBRATION"]
    req(len(formal) == len(train) == len(cal) == 2, "Stage8 role cardinality drift")
    req(sum(int(x["row_count"]) for x in formal) == 1030, "Stage8 formal row count drift")
    req(sum(int(x["row_count"]) for x in train) == 7030, "Stage8 train row count drift")
    req(sum(int(x["row_count"]) for x in cal) == 1947, "Stage8 calibration row count drift")

    req(s82.get("status") == "FROZEN_COMPLETE_BLOCK82B", "Stage8 Block82B seal drift")
    req(s82.get("splits_sha256") == EXPECTED_S8[str(SPLITS)], "Stage8 split SHA binding drift")
    req(s82.get("references_sha256") == EXPECTED_S8[str(REFS)], "Stage8 reference SHA binding drift")

    req(sum88.get("status") == "FORMAL_MEASURED_DEEPSENSE_POINT_ESTIMATES", "Stage8 formal summary status drift")
    req(sum88.get("formal_power_accessed") is True, "Stage8 formal power-open status drift")
    req(int(sum88.get("formal_rows", -1)) == 1030, "Stage8 formal summary row count drift")
    req(sum88.get("post_outcome_tuning") is False, "Stage8 post-outcome tuning flag drift")

    req(s88.get("status") == "FROZEN_COMPLETE_BLOCK88C", "Stage8 Block88C seal status drift")
    req(s88.get("formal_power_accessed") is True, "Stage8 Block88C formal access flag drift")
    req(s88.get("controller_reexecution_after_power_open") is False, "Stage8 controller reexecution flag drift")
    req(s88.get("measured_power_controller_input") is False, "Stage8 controller power-input boundary drift")
    req(s88.get("post_outcome_tuning") is False, "Stage8 Block88C tuning flag drift")
    req(s88.get("formal_records_sha256") == EXPECTED_STAGE8_FORMAL_RECORDS_SHA,
        "Stage8 formal-record hash binding drift")

    req(sep.get("status") == "FROZEN_OPTICAL_MMWAVE_REPORTING_SEPARATION",
        "Stage8 optical/mmWave separation status drift")
    req(sep.get("required_statement") == "DeepSense validates the policy, not the optical headlamp.",
        "Stage8 required reporting statement drift")
    req(sep.get("direct_optical_hardware_validation_claim_from_deepsense") is False,
        "Stage8 direct optical-validation boundary drift")
    req(sep.get("cross_modal_pooling") is False, "Stage8 cross-modal pooling drift")
    req(sep.get("tables_must_remain_separate") is True, "Stage8 reporting-table separation drift")
    req(sep["deepsense_mmwave"].get("result_type") == "measured DeepSense mmWave beam-policy validation",
        "Stage8 DeepSense result type drift")
    req(sep["deepsense_mmwave"].get("formal_records_sha256") == EXPECTED_STAGE8_FORMAL_RECORDS_SHA,
        "Stage8 DeepSense separation formal-record binding drift")

    req(table.get("dataset") == "DeepSense primary LiDAR-aided beam prediction, Scenario8/9 official dev_test",
        "Stage8 final-table dataset identity drift")
    req(int(table.get("formal_rows", -1)) == 1030, "Stage8 final-table row count drift")

    req(matrix.get("status_summary") == "STAGE8_PDF_CORE_COMPLETE", "Stage8 completion matrix drift")
    req(matrix["requirements"].get("DeepSense interpreted as policy validation, not optical-headlamp validation") is True,
        "Stage8 DeepSense interpretation boundary drift")
    req(matrix["requirements"].get("official train/calibration/formal split frozen") is True,
        "Stage8 split-freeze completion drift")

    req(s810.get("status") == "FROZEN_COMPLETE_BLOCK810", "Stage8 Block810 seal status drift")
    req(s810.get("stage8_core_status") == "STAGE8_PDF_CORE_COMPLETE", "Stage8 core status drift")
    req(s810.get("optical_mmwave_separate") is True, "Stage8 optical/mmWave separation seal drift")
    req(s810.get("cross_modal_pooling") is False, "Stage8 cross-modal pooling seal drift")
    req(s810.get("post_outcome_tuning") is False, "Stage8 final tuning flag drift")
    req(s810.get("scientific_reexecution") is False, "Stage8 scientific reexecution flag drift")
    req(s810.get("raw_power_files_reopened") is False, "Stage8 raw-power reopen flag drift")
    req(s810.get("formal_records_sha256") == EXPECTED_STAGE8_FORMAL_RECORDS_SHA,
        "Stage8 final formal-record binding drift")
    req(s810.get("reporting_separation_sha256") == EXPECTED_S8[str(SEP810)],
        "Stage8 reporting-separation binding drift")

    req(super810.get("status") == "STAGE8_PDF_CORE_COMPLETE", "Stage8 superseding seal drift")
    req(super810.get("scientific_reexecution") is False, "Stage8 superseding scientific-reexecution drift")
    req(super810.get("post_outcome_tuning") is False, "Stage8 superseding tuning flag drift")

    print("STAGE8_DEEPSENSE_FROZEN_CHAIN = PASS")
    print("STAGE8_FORMAL_DEV_TEST_ALREADY_OPENED = TRUE")
    print("STAGE8_FORMAL_ROWS = 1030")
    print("RAW_POWER_FILES_OPENED_BY_9_14_V2A = FALSE")

    print("\n===== C. STAGE9 9.14-v2 SCIENTIFIC BOUNDARY =====")

    # The locally canonical DeepSense corpus has already assigned every locally evaluable
    # official split: train -> TRAIN, val -> CALIBRATION, test -> FORMAL_EVALUATION.
    # The challenge set is hidden-GT submission-only. Therefore no new *locally evaluable*
    # untouched DeepSense split is established by the frozen Stage8 corpus.
    locally_evaluable_untouched_split_established = False

    report = {
        "schema": "stage9_block914_v2_external_deepsense_boundary_report_v1",
        "status": "STAGE9_914_V2_EXTERNAL_COMMUNICATION_BOUNDARY_COMPLETE",
        "stage": 9,
        "block": "9.14-v2",
        "subblock": "9.14-v2A",
        "purpose": "secondary external communication validation / DeepSense boundary",
        "current_stage9_authority": {
            "block913_manifest_sha256": EXPECTED_913[str(B913_MANIFEST)],
            "block913_summary_sha256": EXPECTED_913[str(B913_SUMMARY)],
            "block913_seal_sha256": EXPECTED_913[str(B913_SEAL)],
            "block913_status": s913["status"],
            "block913_outcomes_used_to_select_external_dataset_or_split": False,
        },
        "stage8_deepsense_authority": {
            "split_roles_sha256": EXPECTED_S8[str(SPLITS)],
            "reference_integrity_sha256": EXPECTED_S8[str(REFS)],
            "block82b_seal_sha256": EXPECTED_S8[str(SEAL82B)],
            "block88c_seal_sha256": EXPECTED_S8[str(SEAL88C)],
            "formal_summary_sha256": EXPECTED_S8[str(SUMMARY88C)],
            "formal_records_sha256_bound_without_reopening_records": EXPECTED_STAGE8_FORMAL_RECORDS_SHA,
            "reporting_separation_sha256": EXPECTED_S8[str(SEP810)],
            "final_table_sha256": EXPECTED_S8[str(TABLE810)],
            "block810_final_seal_sha256": EXPECTED_S8[str(SEAL810)],
            "block810_superseding_seal_sha256": EXPECTED_S8[str(SUPER810)],
            "formal_rows": 1030,
            "formal_scenarios": {"Scenario8": 437, "Scenario9": 593},
        },
        "canonical_deepsense_split_roles": expected_roles,
        "independence_assessment": {
            "stage8_official_dev_test_already_opened": True,
            "genuinely_untouched_locally_evaluable_deepsense_split_established": (
                locally_evaluable_untouched_split_established
            ),
            "challenge_dataset_local_confirmatory_use": "NOT_AVAILABLE_HIDDEN_GT_SUBMISSION_ONLY",
            "new_stage9_policy_independent_confirmation_on_reused_dev_test": "NOT_AUTHORIZED",
            "reused_dev_test_future_stage9_analysis_if_any": "EXPLORATORY_SECONDARY_ONLY",
            "other_compatible_measured_beam_dataset": "NOT_ASSESSED_IN_THIS_BLOCK",
        },
        "retained_external_evidence": {
            "status": "IMMUTABLE_STAGE8_SUPPORTING_EVIDENCE",
            "claim": (
                "the previously frozen Stage8 beam-policy machinery was evaluated "
                "against measured DeepSense mmWave powers"
            ),
            "new_stage9_policy_tested_on_deepsense_in_9_14_v2": False,
            "stage8_scientific_reexecution": False,
            "stage8_results_modified": False,
        },
        "scope_boundary": {
            "allowed": [
                "measured DeepSense mmWave beam-policy evidence",
                "communication-side coverage / probing-overhead / measured-power-loss evidence",
                "historical external support for the underlying beam-policy machinery",
            ],
            "forbidden_as_deepsense_validation": [
                "optical headlamp hardware",
                "optical PC-FMCW performance",
                "glare model",
                "human visual recovery model",
                "collision or traffic-criticality model",
                "C3 human-safety effect",
                "population-level crash reduction",
            ],
            "required_statement": "DeepSense validates the policy, not the optical headlamp.",
            "cross_modal_pooling": False,
        },
        "execution": {
            "dataset_samples_read": False,
            "raw_lidar_files_read": False,
            "raw_mmwave_power_files_read": False,
            "stage8_formal_record_jsonl_reopened": False,
            "model_forward_executed": False,
            "stage9_policy_executed_on_deepsense": False,
            "metrics_recomputed": False,
            "bootstrap_executed": False,
            "significance_test_executed": False,
            "training": False,
            "calibration_refit": False,
            "retuning": False,
        },
        "legacy_stage9_914_statistical_artifacts": {
            "status": "PRESERVED_AS_LEGACY_PROVENANCE_NOT_CURRENT_9_14",
            "reason": (
                "legacy 9.14A/9.14B were statistical analyses bound to an older "
                "Stage9 9.13 outcome authority; current statistics belong to 9.15"
            ),
        },
        "block914_v2_completion": {
            "fresh_external_confirmatory_experiment_required_for_closure": False,
            "closure_mode": "BOUNDARY_FREEZE_WITH_IMMUTABLE_STAGE8_EXTERNAL_EVIDENCE",
            "new_deepsense_experiment_run": False,
            "next_block": "9.15_STATS_ROBUSTNESS_FAILURE_CASES_LATENCY",
        },
    }

    manifest = {
        "schema": "stage9_block914_v2_external_deepsense_boundary_manifest_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK914_V2_EXTERNAL_BOUNDARY",
        "inputs": {
            "stage9_913": EXPECTED_913,
            "stage8_small_authority_artifacts": EXPECTED_S8,
            "stage8_formal_records_sha256_bound_without_record_reopen": (
                EXPECTED_STAGE8_FORMAL_RECORDS_SHA
            ),
        },
        "report_sha256": None,
        "no_new_deepsense_experiment": True,
        "no_stage9_policy_execution_on_deepsense": True,
        "no_raw_data_access": True,
        "no_metrics_or_statistics": True,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT, report)
    manifest["report_sha256"] = sha256_path(REPORT)
    atomic_json(MANIFEST, manifest)

    seal = {
        "schema": "stage9_block914_v2_external_deepsense_boundary_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK914_V2_EXTERNAL_COMMUNICATION_BOUNDARY",
        "stage9_block913_final_seal_sha256": EXPECTED_913[str(B913_SEAL)],
        "stage8_block810_final_seal_sha256": EXPECTED_S8[str(SEAL810)],
        "stage8_block810_superseding_seal_sha256": EXPECTED_S8[str(SUPER810)],
        "stage8_block88c_formal_seal_sha256": EXPECTED_S8[str(SEAL88C)],
        "stage8_split_roles_sha256": EXPECTED_S8[str(SPLITS)],
        "stage8_formal_records_sha256_bound_without_reopening": EXPECTED_STAGE8_FORMAL_RECORDS_SHA,
        "reporting_separation_sha256": EXPECTED_S8[str(SEP810)],
        "report_sha256": sha256_path(REPORT),
        "manifest_sha256": sha256_path(MANIFEST),
        "stage8_dev_test_already_opened": True,
        "genuinely_untouched_locally_evaluable_deepsense_split_established": False,
        "new_stage9_policy_independent_confirmation_on_reused_dev_test": False,
        "reused_dev_test_stage9_role": "EXPLORATORY_SECONDARY_ONLY_IF_USED",
        "stage8_external_evidence_role": "IMMUTABLE_SUPPORTING_MEASURED_MMWAVE_EVIDENCE",
        "direct_optical_glare_collision_validation_from_deepsense": False,
        "stage9_913_outcomes_used_for_external_split_selection": False,
        "new_deepsense_experiment_executed": False,
        "raw_deepsense_data_accessed": False,
        "metrics_or_statistics_executed": False,
        "retuning": False,
        "next_block": "9.15_STATS_ROBUSTNESS_FAILURE_CASES_LATENCY",
    }
    atomic_json(SEAL, seal)

    print("DEEPSENSE_DEV_TEST_ALREADY_OPENED = TRUE")
    print("UNTOUCHED_LOCALLY_EVALUABLE_DEEPSENSE_SPLIT_ESTABLISHED = FALSE")
    print("NEW_STAGE9_POLICY_CONFIRMATION_ON_REUSED_DEV_TEST = NOT_AUTHORIZED")
    print("STAGE8_EXTERNAL_EVIDENCE = IMMUTABLE_SUPPORTING_MEASURED_MMWAVE_EVIDENCE")
    print("DEEPSENSE_SCOPE = MEASURED_MMWAVE_COMMUNICATION_ONLY")
    print("OPTICAL_GLARE_COLLISION_VALIDATION_FROM_DEEPSENSE = FALSE")
    print("NEW_DEEPSENSE_EXPERIMENT_EXECUTED = FALSE")
    print("STAGE9_9_13_OUTCOMES_USED_FOR_EXTERNAL_SPLIT_SELECTION = FALSE")
    print("BLOCK_9_14_V2 = COMPLETE")
    print("REPORT_SHA256 =", sha256_path(REPORT))
    print("MANIFEST_SHA256 =", sha256_path(MANIFEST))
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15_STATS_ROBUSTNESS_FAILURE_CASES_LATENCY")
    print("=" * 120)


if __name__ == "__main__":
    main()

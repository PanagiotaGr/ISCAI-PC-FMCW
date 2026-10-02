#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
OUTCOME_DIR = B913 / "scenario_outcomes"
P913_MANIFEST = B913 / "stage9_block913_v1_3_FINAL_H1_H6_manifest.jsonl"
P913_SUMMARY = B913 / "stage9_block913_v1_3_FINAL_H1_H6_summary.json"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

B914 = S9 / "artifacts/block914_v2_external_deepsense_boundary"
P914_SEAL = B914 / "stage9_block914_v2_external_boundary_seal.json"

OLD = S9 / "artifacts/block915a_v2_current_formal_statistics_contract"
OLD_EST = OLD / "stage9_block915a_v2_estimand_registry.json"
OLD_CONTRACT = OLD / "stage9_block915a_v2_statistics_contract.json"
OLD_SEAL = OLD / "stage9_block915a_v2_statistics_contract_seal.json"

EXPECTED = {
    str(P913_MANIFEST): "752600ddf126cabd87010d8277b7fe040d18b3e8d8f2490adcc2e9ca0ceb63c4",
    str(P913_SUMMARY): "292d515a3f52c82ad08460e1122a805d236a8aeaefa82271ff84664c2cddfe60",
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(P914_SEAL): "9d9ec48e9c75307861a5b895e1b26365328abe748e43cca5598021ea70031ddd",
    str(OLD_EST): "18291ae57cb087f36a4d499b4be3327c3fd67e2a0170f36264b28f2a7dacfe8e",
    str(OLD_CONTRACT): "d6d1e3c52f9b64da053749ded9fd803a0a428edeaf1da540a3c78e7078120234",
    str(OLD_SEAL): "852209992563ce7b72bcd502288f1b545540137b2a2348defb249fd6676feb21",
}

FAILED_R1_RUNNER_SHA = "9cc98609b5e85d5fd7c2a36b7d2fa6d839678b94b0725845b661f508816a3ae1"
FAILED_R2_RUNNER_SHA = "f743f543aac5d31364a7c76e4f7726863b278a5b31a268427bb8f3876bd1a9a1"
FAILED_R3_RUNNER_SHA = "6a3c0261e3a6213e0bbf2ba6127b1b5c30f41de90eb1a283ff00fe7761bcae11"

OUT = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
EVIDENCE = OUT / "stage9_block915a_v2R4_h1_schema_status_censoring_evidence.json"
EST = OUT / "stage9_block915a_v2R4_estimand_registry.json"
CONTRACT = OUT / "stage9_block915a_v2R4_statistics_contract.json"
SEAL = OUT / "stage9_block915a_v2R4_statistics_contract_seal.json"


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
        path = Path(raw)
        req(path.is_file(), f"missing frozen authority: {path}")
        got = sha256_path(path)
        req(got == wanted, f"SHA256 drift {path}: {got} != {wanted}")
        print("EXACT PASS", path)


def scan_h1_schema_status_censoring() -> dict[str, Any]:
    files = sorted(OUTCOME_DIR.glob("*.json"))
    req(len(files) == 26209, f"outcome cardinality drift: {len(files)} != 26209")

    allowed_h = {0.1, 0.3, 0.5, 1.0}
    allowed_c = {16, 32, 64}
    allowed_a = {"RAW", "CALIBRATED"}
    known_statuses = {
        "NOT_EVALUATED_NO_ELIGIBLE_RECEIVER",
        "EVALUATED_FUTURE_GT",
        "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID",
    }

    status_counts = Counter()
    record_count_counts = Counter()
    record_count_by_status = {}
    horizon_subset_counts = Counter()
    horizon_subset_by_status = {}
    per_horizon_scene_counts = Counter()
    per_horizon_record_counts = Counter()

    all_arms = set()
    all_codebooks = set()
    all_horizons = set()
    q_values = set()

    no_receiver_nonempty_h1 = 0
    status_record_count_drift = 0
    duplicate_combo_scenes = 0
    incomplete_horizon_block_scenes = 0
    actor1s_invalid_has_h1_1s = 0
    actor1s_invalid_semantic_drift = 0
    full_eval_missing_h1_1s = 0
    full_eval_semantic_drift = 0

    for i, path in enumerate(files):
        j = load_json(path)
        req(int(j.get("formal_ordinal", -1)) == i,
            f"formal ordinal/path ordering drift at {path.name}")
        req(j.get("status") == "COMPLETE_FINAL_EVALUATOR_H1_H6",
            f"outer outcome status drift: {path.name}")

        primary = j.get("primary")
        req(isinstance(primary, dict), f"missing primary object: {path.name}")
        status = str(primary.get("status"))
        status_counts[status] += 1

        recs = primary.get("H1_records", [])
        req(isinstance(recs, list), f"H1_records not a list: {path.name}")

        if status == "NOT_EVALUATED_NO_ELIGIBLE_RECEIVER":
            if recs:
                no_receiver_nonempty_h1 += 1
            continue

        record_count_counts[len(recs)] += 1
        record_count_by_status.setdefault(status, Counter())[len(recs)] += 1

        # Status-specific structural cardinality.
        # Full 1s-valid evaluation must have at least the 1.0s block.
        # Actor-1s-invalid may legitimately have zero H1 records if no earlier horizon is valid.
        if status == "EVALUATED_FUTURE_GT":
            if len(recs) not in (6, 12, 18, 24):
                status_record_count_drift += 1
        elif status == "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID":
            if len(recs) not in (0, 6, 12, 18):
                status_record_count_drift += 1

        combo_seen = set()
        horizons_here = set()
        block_counts = Counter()

        for r in recs:
            a = str(r["arm"])
            c = int(r["codebook_size"])
            h = float(r["horizon_s"])
            q = float(r["requested_q"])

            req(a in allowed_a, f"unexpected H1 arm {a}: {path.name}")
            req(c in allowed_c, f"unexpected H1 codebook {c}: {path.name}")
            req(h in allowed_h, f"unexpected H1 horizon {h}: {path.name}")
            req(abs(q - 0.95) <= 1e-15, f"unexpected H1 requested_q {q}: {path.name}")

            key = (a, c, h)
            if key in combo_seen:
                duplicate_combo_scenes += 1
            combo_seen.add(key)

            horizons_here.add(h)
            block_counts[h] += 1
            all_arms.add(a)
            all_codebooks.add(c)
            all_horizons.add(h)
            q_values.add(q)
            per_horizon_record_counts[h] += 1

        expected_combo = {
            (a, c, h)
            for h in horizons_here
            for a in allowed_a
            for c in allowed_c
        }
        if combo_seen != expected_combo or any(block_counts[h] != 6 for h in horizons_here):
            incomplete_horizon_block_scenes += 1

        subset_key = ",".join(f"{h:.1f}" for h in sorted(horizons_here)) if horizons_here else "EMPTY"
        horizon_subset_counts[subset_key] += 1
        horizon_subset_by_status.setdefault(status, Counter())[subset_key] += 1
        for h in horizons_here:
            per_horizon_scene_counts[h] += 1

        if status == "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID":
            if 1.0 in horizons_here:
                actor1s_invalid_has_h1_1s += 1
            label = primary.get("criticality_label")
            if not (
                primary.get("actor_future_valid_1s") is False
                and primary.get("actor_center_truth_1s") is None
                and isinstance(label, dict)
                and label.get("label_valid") is False
                and label.get("Y_crit_1s") is None
                and label.get("missing_reason") == "actor_future_unavailable_or_invalid"
                and primary.get("communication_records") == []
            ):
                actor1s_invalid_semantic_drift += 1

        elif status == "EVALUATED_FUTURE_GT":
            if 1.0 not in horizons_here:
                full_eval_missing_h1_1s += 1
            label = primary.get("criticality_label")
            if not (
                primary.get("actor_future_valid_1s") is True
                and primary.get("actor_center_truth_1s") is not None
                and isinstance(label, dict)
                and label.get("label_valid") is True
                and label.get("Y_crit_1s") in (0, 1)
            ):
                full_eval_semantic_drift += 1

        if (i + 1) % 5000 == 0:
            print(f"H1 schema/status/censoring verify {i+1}/26209", flush=True)

    print("PRIMARY_STATUS_COUNTS =", dict(sorted(status_counts.items())), flush=True)
    print("H1_RECORD_COUNT_DISTRIBUTION =", dict(sorted(record_count_counts.items())), flush=True)
    print(
        "H1_RECORD_COUNT_BY_STATUS =",
        {k: dict(sorted(v.items())) for k, v in sorted(record_count_by_status.items())},
        flush=True,
    )
    print("H1_HORIZON_SUBSET_DISTRIBUTION =", dict(sorted(horizon_subset_counts.items())), flush=True)
    print(
        "H1_HORIZON_SUBSET_BY_STATUS =",
        {k: dict(sorted(v.items())) for k, v in sorted(horizon_subset_by_status.items())},
        flush=True,
    )

    unknown = sorted(set(status_counts) - known_statuses)
    req(not unknown, f"unknown primary status(es) after full scan: {unknown}")

    req(status_counts["NOT_EVALUATED_NO_ELIGIBLE_RECEIVER"] == 970,
        f"no-receiver count drift: {status_counts['NOT_EVALUATED_NO_ELIGIBLE_RECEIVER']} != 970")
    req(
        status_counts["EVALUATED_FUTURE_GT"]
        + status_counts["EVALUATED_FUTURE_GT_ACTOR_1S_INVALID"]
        == 25239,
        "selected/evaluated primary scene count drift",
    )
    req(no_receiver_nonempty_h1 == 0,
        f"no-receiver scenes unexpectedly contain H1 records: {no_receiver_nonempty_h1}")
    req(status_record_count_drift == 0,
        f"status-specific H1 record-count drift scenes: {status_record_count_drift}")
    req(duplicate_combo_scenes == 0,
        f"duplicate H1 combo scenes: {duplicate_combo_scenes}")
    req(incomplete_horizon_block_scenes == 0,
        f"incomplete arm/codebook blocks at available horizon: {incomplete_horizon_block_scenes}")
    req(actor1s_invalid_has_h1_1s == 0,
        f"actor-1s-invalid scenes unexpectedly contain H1 1.0s records: {actor1s_invalid_has_h1_1s}")
    req(actor1s_invalid_semantic_drift == 0,
        f"actor-1s-invalid semantic drift scenes: {actor1s_invalid_semantic_drift}")
    req(full_eval_missing_h1_1s == 0,
        f"full-evaluation scenes missing H1 1.0s support: {full_eval_missing_h1_1s}")
    req(full_eval_semantic_drift == 0,
        f"full-evaluation semantic drift scenes: {full_eval_semantic_drift}")

    req(all_arms == allowed_a, f"H1 arm authority mismatch: {sorted(all_arms)}")
    req(all_codebooks == allowed_c, f"H1 codebook authority mismatch: {sorted(all_codebooks)}")
    req(all_horizons == allowed_h, f"H1 horizon authority mismatch: {sorted(all_horizons)}")
    req(q_values == {0.95}, f"H1 q authority mismatch: {sorted(q_values)}")

    return {
        "schema": "stage9_block915a_v2R4_h1_schema_status_censoring_evidence_v1",
        "status": "PASS_EXACT_CURRENT_913_H1_SCHEMA_STATUS_AND_CENSORING",
        "authority_913_seal_sha256": EXPECTED[str(P913_SEAL)],
        "FORMAL_scene_count": 26209,
        "primary_status_counts": dict(sorted(status_counts.items())),
        "H1_record_count_distribution": {str(k): int(v) for k, v in sorted(record_count_counts.items())},
        "H1_record_count_by_status": {
            k: {str(kk): int(vv) for kk, vv in sorted(v.items())}
            for k, v in sorted(record_count_by_status.items())
        },
        "H1_horizon_subset_distribution": dict(sorted(horizon_subset_counts.items())),
        "H1_horizon_subset_by_status": {
            k: dict(sorted(v.items())) for k, v in sorted(horizon_subset_by_status.items())
        },
        "H1_available_scene_count_by_horizon": {
            f"{k:.1f}": int(v) for k, v in sorted(per_horizon_scene_counts.items())
        },
        "H1_record_count_by_horizon": {
            f"{k:.1f}": int(v) for k, v in sorted(per_horizon_record_counts.items())
        },
        "H1_arms": sorted(all_arms),
        "H1_codebook_sizes": sorted(all_codebooks),
        "H1_horizons_s": sorted(all_horizons),
        "H1_requested_q": 0.95,
        "status_specific_cardinality_rule": {
            "NOT_EVALUATED_NO_ELIGIBLE_RECEIVER": "0 H1 records",
            "EVALUATED_FUTURE_GT": "6/12/18/24 records, complete six-record blocks, must include 1.0s",
            "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID": (
                "0/6/12/18 records, complete six-record blocks, must exclude 1.0s; "
                "zero records is valid when no earlier horizon has valid future truth"
            ),
        },
        "horizon_censoring_rule": (
            "Each materialized horizon contributes exactly the complete "
            "RAW/CALIBRATED x 16/32/64 six-record block. Missing horizons are not imputed."
        ),
        "statistics_computed": False,
        "bootstrap_draws_executed": False,
        "RNG_used": False,
        "policy_or_threshold_selection_from_outcomes": False,
    }


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15A-v2R4 — H1 STATUS + HORIZON CENSORING CONTRACT REPAIR")
    print("SCHEMA/STATUS VALIDATION ONLY — NO BOOTSTRAP / NO RNG / NO CI / NO P-VALUE / NO RETUNING")
    print("=" * 124)

    req(not SEAL.exists(), f"9.15A-v2R4 already sealed: {SEAL}")
    hash_gate()

    old_seal = load_json(OLD_SEAL)
    req(old_seal.get("bootstrap_executed") is False, "old 9.15A unexpectedly executed bootstrap")
    req(old_seal.get("RNG_used") is False, "old 9.15A unexpectedly used RNG")
    req(old_seal.get("CI_computed") is False, "old 9.15A unexpectedly computed CI")
    req(old_seal.get("p_value_computed") is False, "old 9.15A unexpectedly computed p-value")

    print("\n===== A. EXACT CURRENT 9.13 H1 SCHEMA / STATUS / CENSORING VALIDATION =====")
    evidence = scan_h1_schema_status_censoring()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(EVIDENCE, evidence)

    print("H1_HORIZONS_CURRENT_913 =", evidence["H1_horizons_s"])
    print("H1_CODEBOOKS_CURRENT_913 =", evidence["H1_codebook_sizes"])
    print("H1_ARMS_CURRENT_913 =", evidence["H1_arms"])
    print("H1_AVAILABLE_SCENES_BY_HORIZON =", evidence["H1_available_scene_count_by_horizon"])

    print("\n===== B. SUPERSEDING PRE-INFERENCE CONTRACT =====")
    old_est = load_json(OLD_EST)
    old_contract = load_json(OLD_CONTRACT)

    new_est = json.loads(json.dumps(old_est))
    new_est["schema"] = "stage9_block915a_v2R4_estimand_registry_v1"
    new_est["status"] = "FROZEN_CURRENT_FORMAL_ESTIMAND_REGISTRY_PRE_INFERENCE_REPAIRED"
    new_est["supersedes_estimand_registry_sha256"] = EXPECTED[str(OLD_EST)]
    new_est["repair"] = {
        "failed_R1_runner_sha256": FAILED_R1_RUNNER_SHA,
        "failed_R2_runner_sha256": FAILED_R2_RUNNER_SHA,
        "failed_R3_runner_sha256": FAILED_R3_RUNNER_SHA,
        "R1_failure": "incorrectly required 24 H1 records in every evaluated scene",
        "R2_failure": "incorrectly treated EVALUATED_FUTURE_GT_ACTOR_1S_INVALID as unexpected",
        "R3_failure": "correctly recognized actor-1s-invalid status but incorrectly rejected its valid zero-record case",
        "correct_horizons_s": [0.1, 0.3, 0.5, 1.0],
        "actor_1s_invalid_zero_H1_records_is_valid_when_no_earlier_truth_support": True,
        "schema_status_censoring_evidence_sha256": sha256_path(EVIDENCE),
        "bootstrap_seen_before_repair": False,
        "hypothesis_changed": False,
        "scientific_endpoint_changed": False,
        "population_selection_changed": False,
        "missing_horizon_imputation_added": False,
    }

    h1 = new_est["hypotheses"]["H1"]
    h1["fixed_strata"]["horizons_s"] = [0.1, 0.3, 0.5, 1.0]
    h1["population"] = (
        "frozen H1 RAW/CALIBRATED records with future truth available at the specific horizon; "
        "denominator is horizon-specific and fixed by the materialized 9.13 carrier"
    )
    h1["horizon_censoring"] = {
        "missing_future_truth_at_horizon": "EXCLUDE_FROM_THAT_HORIZON_STRATUM_ONLY",
        "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID": (
            "valid frozen status; 1.0s unavailable; may contain zero earlier-horizon H1 records "
            "if no earlier future truth is valid"
        ),
        "imputation": False,
        "drop_entire_scene_because_one_horizon_missing": False,
        "arm_or_codebook_specific_missingness_allowed": False,
        "required_block_when_horizon_available": "2 arms x 3 codebooks = 6 records",
        "bootstrap_replicate_zero_denominator_rule": (
            "FAIL_CLOSED_NO_REDRAW if any of the 12 required H1 strata has zero denominator"
        ),
    }
    h1["primary_estimand"] = (
        "Within each bootstrap replicate, compute empirical coverage separately for each "
        "arm x codebook x horizon stratum using only sampled scenarios with frozen future-truth "
        "support at that horizon; compute absolute error from q=0.95; macro-average equally "
        "over the 12 codebook x horizon strata within each arm. Primary delta is "
        "macro_error(CALIBRATED) - macro_error(RAW)."
    )
    new_est["robustness"]["H1_horizons_s"] = [0.1, 0.3, 0.5, 1.0]

    atomic_json(EST, new_est)

    new_contract = json.loads(json.dumps(old_contract))
    new_contract["schema"] = "stage9_block915a_v2R4_statistics_contract_v1"
    new_contract["status"] = "FROZEN_POST_FORMAL_PRE_CURRENT_INFERENCE_CONTRACT_REPAIRED"
    new_contract["block"] = "9.15A-v2R4"
    new_contract["supersedes_contract_sha256"] = EXPECTED[str(OLD_CONTRACT)]
    new_contract["supersedes_seal_sha256"] = EXPECTED[str(OLD_SEAL)]
    new_contract["repair"] = {
        "failed_R1_runner_sha256": FAILED_R1_RUNNER_SHA,
        "failed_R2_runner_sha256": FAILED_R2_RUNNER_SHA,
        "failed_R3_runner_sha256": FAILED_R3_RUNNER_SHA,
        "kind": "H1_HORIZON_STATUS_AND_FROZEN_CENSORING_ALIGNMENT",
        "correct_horizons_s": [0.1, 0.3, 0.5, 1.0],
        "actor_1s_invalid_zero_H1_records_is_valid": True,
        "schema_status_censoring_evidence_sha256": sha256_path(EVIDENCE),
        "timing": "BEFORE_ANY_CURRENT_FINAL_9_13_BOOTSTRAP_DRAW",
        "bootstrap_results_seen_before_repair": False,
        "current_final_9_13_CI_seen_before_repair": False,
        "hypothesis_changed": False,
        "scientific_endpoint_changed": False,
        "policy_baseline_threshold_change": False,
        "future_truth_missingness_imputed": False,
    }
    new_contract["estimand_registry_sha256"] = sha256_path(EST)
    new_contract["timing_disclosure"]["current_final_9_13_bootstrap_draws_before_this_contract"] = False
    new_contract["timing_disclosure"]["current_final_9_13_significance_tests_before_this_contract"] = False
    new_contract["timing_disclosure"]["current_9_13_outcome_JSON_opened_for_schema_status_censoring_validation"] = True
    new_contract["next_block"] = "9.15B_CURRENT_FORMAL_PRIMARY_BOOTSTRAP_AND_H1_H6_INFERENCE"
    atomic_json(CONTRACT, new_contract)

    seal = {
        "schema": "stage9_block915a_v2R4_statistics_contract_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915A_V2R4_PRE_INFERENCE_CONTRACT",
        "stage9_913_seal_sha256": EXPECTED[str(P913_SEAL)],
        "stage9_914_v2_seal_sha256": EXPECTED[str(P914_SEAL)],
        "superseded_915A_estimand_registry_sha256": EXPECTED[str(OLD_EST)],
        "superseded_915A_contract_sha256": EXPECTED[str(OLD_CONTRACT)],
        "superseded_915A_seal_sha256": EXPECTED[str(OLD_SEAL)],
        "failed_R1_runner_sha256": FAILED_R1_RUNNER_SHA,
        "failed_R2_runner_sha256": FAILED_R2_RUNNER_SHA,
        "failed_R3_runner_sha256": FAILED_R3_RUNNER_SHA,
        "H1_schema_status_censoring_evidence_sha256": sha256_path(EVIDENCE),
        "estimand_registry_sha256": sha256_path(EST),
        "contract_sha256": sha256_path(CONTRACT),
        "H1_horizons_s": [0.1, 0.3, 0.5, 1.0],
        "H1_codebook_sizes": [16, 32, 64],
        "H1_arms": ["RAW", "CALIBRATED"],
        "H1_requested_q": 0.95,
        "actor_1s_invalid_zero_record_case_preserved": True,
        "H1_horizon_specific_future_truth_censoring_preserved": True,
        "H1_missing_horizon_imputation": False,
        "bootstrap_executed": False,
        "RNG_used": False,
        "CI_computed": False,
        "p_value_computed": False,
        "outcome_json_opened_for_schema_status_censoring_validation": True,
        "outcome_values_used_for_policy_or_threshold_selection": False,
        "hypothesis_changed": False,
        "scientific_endpoint_changed": False,
        "retuning": False,
        "policy_baseline_threshold_change": False,
        "next_block": "9.15B_CURRENT_FORMAL_PRIMARY_BOOTSTRAP_AND_H1_H6_INFERENCE",
    }
    atomic_json(SEAL, seal)

    print("H1_STATUS_HORIZON_CENSORING_REPAIR = PASS")
    print("CURRENT_EXACT_HORIZONS =", [0.1, 0.3, 0.5, 1.0])
    print("ACTOR_1S_INVALID_ZERO_H1_RECORD_CASE = PRESERVED_VALID_FROZEN_STATUS")
    print("PER_AVAILABLE_HORIZON_COMPLETE_BLOCK = 6_RECORDS")
    print("MISSING_HORIZON_IMPUTATION = FALSE")
    print("BOOTSTRAP_EXECUTED = FALSE")
    print("RNG_USED = FALSE")
    print("CI_COMPUTED = FALSE")
    print("HYPOTHESIS_CHANGED = FALSE")
    print("SCIENTIFIC_ENDPOINT_CHANGED = FALSE")
    print("POLICY_BASELINE_THRESHOLD_CHANGE = FALSE")
    print("BLOCK_9_15A_V2R4 = COMPLETE")
    print("SCHEMA_STATUS_CENSORING_EVIDENCE_SHA256 =", sha256_path(EVIDENCE))
    print("ESTIMAND_REGISTRY_SHA256 =", sha256_path(EST))
    print("CONTRACT_SHA256 =", sha256_path(CONTRACT))
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15B_CURRENT_FORMAL_PRIMARY_BOOTSTRAP_AND_H1_H6_INFERENCE")
    print("=" * 124)


if __name__ == "__main__":
    main()

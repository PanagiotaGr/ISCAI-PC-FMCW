#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

# -------------------------------------------------------------------------------------------------
# Frozen upstream authorities.
# -------------------------------------------------------------------------------------------------
B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_PREFLIGHT = B915B / "stage9_block915b_v2_prebootstrap_schema_gate.json"
P915B_REPLICATES = B915B / "stage9_block915b_v2_bootstrap_replicates.npz"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

EXPECTED = {
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",
    str(P915B_PREFLIGHT): "3fff3acb783b74cde2b9b28fb38901930ba154ab7af63732235daad58aaaa7ef",
    str(P915B_REPLICATES): "ef31ae620f02f208bd38e8b87b10b19078e23e1b68c9a8d044897f7f3fc4d0bc",
    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",
}

OUT = S9 / "artifacts/block915c_v2_robustness"
REPORT = OUT / "stage9_block915c_v2_robustness_report.json"
SEAL = OUT / "stage9_block915c_v2_robustness_seal.json"

PROFILES = ("H_fast", "H_nominal", "H_slow")
BASELINES = ("original_reactive_ADB", "Stage6_V3_predictive_ADB")
COMPARISONS = tuple((p, b) for p in PROFILES for b in BASELINES)
IDX = {k: i for i, k in enumerate(COMPARISONS)}
PROFILE_PAIRS = (
    ("H_fast", "H_nominal"),
    ("H_fast", "H_slow"),
    ("H_nominal", "H_slow"),
)


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


def q95(x: np.ndarray) -> list[float]:
    q = np.quantile(np.asarray(x, dtype=np.float64), [0.025, 0.975], method="linear")
    return [float(q[0]), float(q[1])]


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15C-v2 — ROBUSTNESS")
    print("HUMAN-PROFILE ROBUSTNESS + CALIBRATION-PERTURBATION DISPOSITION")
    print("REUSE 9.15B BOOTSTRAP ONLY | NO NEW RNG | NO POLICY/MODEL RERUN | NO POST-FORMAL PERTURBATION INVENTION")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    a = load_json(P915A_SEAL)
    b = load_json(P915B_SEAL)
    s = load_json(P915B_SUMMARY)

    req(
        a.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915A_V2R4_PRE_INFERENCE_CONTRACT",
        "9.15A authority drift",
    )
    req(
        b.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915B_V2_CURRENT_FORMAL_H1_H6_STATISTICS",
        "9.15B authority drift",
    )
    req(int(b.get("bootstrap_replicates", -1)) == 10000, "9.15B bootstrap count drift")
    req(int(b.get("bootstrap_seed", -1)) == 20260910, "9.15B bootstrap seed drift")

    h5 = s.get("H5")
    req(isinstance(h5, dict), "9.15B H5 summary missing")
    req(h5.get("status") == "SECONDARY_FORMAL_SENSITIVITY_ALL_SIX_REPORTED",
        "9.15B H5 status drift")
    req(int(h5.get("comparison_count", -1)) == 6, "H5 comparison count drift")
    req(h5.get("FAST_excluded") is False, "FAST exclusion drift")
    req(h5.get("profile_or_baseline_dropped") is False, "H5 profile/baseline drop drift")

    expected_keys = {f"{p}__{bb}" for p, bb in COMPARISONS}
    comps = h5.get("comparisons")
    req(isinstance(comps, dict) and set(comps) == expected_keys,
        "H5 comparison-key schema drift")

    with np.load(P915B_REPLICATES, allow_pickle=False) as z:
        req("H5_delta_violation_rate" in z.files, "9.15B H5 replicate carrier missing")
        reps = np.asarray(z["H5_delta_violation_rate"], dtype=np.float64)

    req(reps.shape == (10000, 6), f"H5 replicate shape drift: {reps.shape}")
    req(np.all(np.isfinite(reps)), "H5 replicate carrier contains non-finite values")

    print("\n===== A. HUMAN-PROFILE ROBUSTNESS =====")

    by_baseline: dict[str, Any] = {}
    for baseline in BASELINES:
        profile_rows = {}
        decisions = []
        points = []

        for profile in PROFILES:
            key = f"{profile}__{baseline}"
            row = comps[key]
            point = float(row["point_delta_rate_recovery_minus_baseline"])
            ci = [float(x) for x in row["CI95"]]
            decision = str(row["decision"])
            req(decision in {"REDUCTION_SUPPORTED", "INCREASE_SUPPORTED", "NO_CLEAR_DIFFERENCE"},
                f"unexpected H5 decision: {key}")
            profile_rows[profile] = {
                "n": int(row["n"]),
                "baseline_violations": int(row["baseline_violations"]),
                "recovery_violations": int(row["recovery_violations"]),
                "point_delta_rate_recovery_minus_baseline": point,
                "CI95": ci,
                "decision": decision,
            }
            decisions.append(decision)
            points.append(point)

        # Secondary descriptive cross-profile contrasts using exactly the already-frozen
        # 9.15B replicate matrix. No new bootstrap draws are generated.
        contrasts = {}
        for p1, p2 in PROFILE_PAIRS:
            c1 = IDX[(p1, baseline)]
            c2 = IDX[(p2, baseline)]
            d = reps[:, c1] - reps[:, c2]
            contrasts[f"{p1}_minus_{p2}"] = {
                "point_difference_of_recovery_minus_baseline_effects": (
                    float(comps[f"{p1}__{baseline}"]["point_delta_rate_recovery_minus_baseline"])
                    - float(comps[f"{p2}__{baseline}"]["point_delta_rate_recovery_minus_baseline"])
                ),
                "CI95_from_reused_915B_bootstrap": q95(d),
                "role": "SECONDARY_DESCRIPTIVE_PROFILE_SENSITIVITY",
            }

        cols = np.stack([reps[:, IDX[(p, baseline)]] for p in PROFILES], axis=1)
        span = np.max(cols, axis=1) - np.min(cols, axis=1)

        by_baseline[baseline] = {
            "profiles": profile_rows,
            "same_H5_decision_across_FAST_NOMINAL_SLOW": len(set(decisions)) == 1,
            "recovery_reduction_supported_in_all_three_profiles": all(
                d == "REDUCTION_SUPPORTED" for d in decisions
            ),
            "point_effect_range_across_profiles": [float(min(points)), float(max(points))],
            "point_effect_span_across_profiles": float(max(points) - min(points)),
            "bootstrap_profile_effect_span_CI95": q95(span),
            "pairwise_profile_effect_contrasts": contrasts,
        }

    same_decision_both = all(
        x["same_H5_decision_across_FAST_NOMINAL_SLOW"] for x in by_baseline.values()
    )
    reduction_all = all(
        x["recovery_reduction_supported_in_all_three_profiles"] for x in by_baseline.values()
    )

    if reduction_all:
        human_status = "RECOVERY_REDUCTION_SUPPORTED_ACROSS_ALL_FROZEN_HUMAN_PROFILES_AND_BOTH_BASELINES"
    elif not same_decision_both:
        human_status = "PROFILE_SENSITIVE_PAIRWISE_H5_CONCLUSIONS"
    else:
        human_status = "PROFILE_CONCLUSION_DIRECTION_STABLE_BUT_NOT_UNIVERSALLY_SUPPORTED_AS_REDUCTION"

    # Important methodological boundary:
    # H5 is baseline-referenced paired common-finite support. That supports the two
    # recovery-vs-baseline comparisons per profile, but it does NOT automatically provide
    # one common support on which to claim a full three-way policy ranking.
    full_three_way_ranking = "NOT_AUTHORIZED_FROM_BASELINE_REFERENCED_H5_SUPPORT"

    print("HUMAN_PROFILE_STATUS =", human_status)
    for baseline, row in by_baseline.items():
        print(
            baseline,
            "SAME_DECISION_ACROSS_PROFILES =",
            row["same_H5_decision_across_FAST_NOMINAL_SLOW"],
            "ALL_THREE_REDUCTION_SUPPORTED =",
            row["recovery_reduction_supported_in_all_three_profiles"],
        )

    print("\n===== B. CALIBRATION ROBUSTNESS DISPOSITION =====")

    # The Stage-9 design document names the qualitative perturbation states
    # underconfident / calibrated / overconfident, but the frozen executable chain
    # supplied to 9.15 does not establish a numerical covariance/temperature/scale mapping.
    #
    # After FORMAL opening, inventing such a mapping would be a new post-outcome
    # sensitivity definition. Therefore we do not execute it.
    calibration = {
        "requested_by_stage9_design": True,
        "requested_states": ["underconfident", "calibrated", "overconfident"],
        "design_document_provides_numeric_transform": False,
        "prefrozen_executable_numeric_perturbation_authority_established_for_9_15": False,
        "status": "NOT_EVALUABLE_WITHOUT_PREFROZEN_NUMERIC_PERTURBATION_RULE",
        "underconfident_execution": "NOT_PERFORMED",
        "overconfident_execution": "NOT_PERFORMED",
        "central_calibrated_FORMAL_condition": "ALREADY_EVALUATED_IN_9_13_9_15B",
        "RAW_H1_arm_used_as_under_or_overconfident_substitute": False,
        "critical_beam_outage_sensitivity": "NOT_EVALUABLE_WITHOUT_PREFROZEN_NUMERIC_PERTURBATION_RULE",
        "ADB_human_safety_risk_sensitivity": "NOT_EVALUABLE_WITHOUT_PREFROZEN_NUMERIC_PERTURBATION_RULE",
        "policy_or_calibrator_rerun": False,
        "post_FORMAL_scale_selection": False,
        "reason": (
            "The design specifies qualitative underconfident/calibrated/overconfident states "
            "but no frozen numerical perturbation transformation is established in the current "
            "pre-inference authority. A post-FORMAL covariance/temperature/scale factor is not invented."
        ),
    }
    print("CALIBRATION_ROBUSTNESS_STATUS =", calibration["status"])
    print("NEW_CALIBRATION_PERTURBATION_EXECUTED = FALSE")

    report = {
        "schema": "stage9_block915c_v2_robustness_report_v1",
        "status": "STAGE9_915C_ROBUSTNESS_COMPLETE_WITH_CALIBRATION_PERTURBATION_NOT_EVALUABLE",
        "authority": {
            "stage915a_v2R4_seal_sha256": EXPECTED[str(P915A_SEAL)],
            "stage915b_preflight_sha256": EXPECTED[str(P915B_PREFLIGHT)],
            "stage915b_bootstrap_replicates_sha256": EXPECTED[str(P915B_REPLICATES)],
            "stage915b_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
            "stage915b_seal_sha256": EXPECTED[str(P915B_SEAL)],
        },
        "human_profile_robustness": {
            "profiles": list(PROFILES),
            "underlying_trajectory_changed": False,
            "new_human_profile_created": False,
            "bootstrap_reused_from_9_15B": True,
            "new_bootstrap_draws": False,
            "new_RNG": False,
            "analysis_role": "SECONDARY_FORMAL_ROBUSTNESS",
            "status": human_status,
            "by_baseline": by_baseline,
            "full_three_way_policy_ranking": full_three_way_ranking,
            "interpretation_boundary": (
                "Report recovery-vs-reactive and recovery-vs-Stage6 profile robustness separately. "
                "Do not convert baseline-referenced paired supports into an unsupported universal "
                "three-way ranking."
            ),
        },
        "calibration_robustness": calibration,
        "H6_context": {
            "stage915b_H6_status": s["H6"]["status"],
            "profile_specific_H6_carrier_available": False,
            "disposition": (
                "H6 remains the frozen side-effect context from 9.15B; no profile-specific "
                "H6 metric is fabricated."
            ),
        },
        "retuning": False,
        "policy_changed": False,
        "human_profile_definition_changed": False,
        "calibration_mapping_changed": False,
        "new_FORMAL_model_or_policy_execution": False,
        "next_block": "9.15D_FAILURE_CASE_ANALYSIS",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT, report)

    seal = {
        "schema": "stage9_block915c_v2_robustness_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915C_V2_ROBUSTNESS",
        "stage915b_seal_sha256": EXPECTED[str(P915B_SEAL)],
        "stage915b_bootstrap_replicates_sha256": EXPECTED[str(P915B_REPLICATES)],
        "report_sha256": sha256_path(REPORT),
        "human_profile_robustness_complete": True,
        "human_profile_status": human_status,
        "all_profiles_and_both_baselines_retained": True,
        "new_bootstrap_draws": False,
        "new_RNG_used": False,
        "calibration_robustness_status": calibration["status"],
        "new_calibration_perturbation_executed": False,
        "post_FORMAL_perturbation_rule_invented": False,
        "retuning": False,
        "next_block": "9.15D_FAILURE_CASE_ANALYSIS",
    }
    atomic_json(SEAL, seal)

    print("\n" + "=" * 124)
    print("STAGE 9.15C-v2 = COMPLETE")
    print("HUMAN_PROFILE_ROBUSTNESS =", human_status)
    print("FULL_THREE_WAY_POLICY_RANKING =", full_three_way_ranking)
    print("CALIBRATION_ROBUSTNESS =", calibration["status"])
    print("NEW_BOOTSTRAP_DRAWS = FALSE")
    print("NEW_RNG_USED = FALSE")
    print("NEW_CALIBRATION_PERTURBATION_EXECUTED = FALSE")
    print("REPORT_SHA256 =", sha256_path(REPORT))
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15D_FAILURE_CASE_ANALYSIS")
    print("=" * 124)


if __name__ == "__main__":
    main()

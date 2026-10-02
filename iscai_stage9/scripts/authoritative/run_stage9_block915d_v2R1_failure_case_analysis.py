#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

# -------------------------------------------------------------------------------------------------
# Frozen authorities.
# -------------------------------------------------------------------------------------------------
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
OUTCOME_DIR = B913 / "scenario_outcomes"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_EST = B915A / "stage9_block915a_v2R4_estimand_registry.json"
P915A_CONTRACT = B915A / "stage9_block915a_v2R4_statistics_contract.json"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_PREFLIGHT = B915B / "stage9_block915b_v2_prebootstrap_schema_gate.json"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

B915C = S9 / "artifacts/block915c_v2_robustness"
P915C_REPORT = B915C / "stage9_block915c_v2_robustness_report.json"
P915C_SEAL = B915C / "stage9_block915c_v2_robustness_seal.json"

EXPECTED = {
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(P915A_EST): "4ecc05f12b5c11e46a976b161fd7866d30bda76f287d4904d7cd52e95a71e37c",
    str(P915A_CONTRACT): "91362b13413565501d3d33905d78bc12f14bb9c8b3739783e899cbc6088c7b03",
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",
    str(P915B_PREFLIGHT): "3fff3acb783b74cde2b9b28fb38901930ba154ab7af63732235daad58aaaa7ef",
    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",
    str(P915C_REPORT): "a42be7cc37e5c891d4c26a0ac6ceea472074fca915bdd36d33896b15b472cb6c",
    str(P915C_SEAL): "a8fa3276e29e7093dd0e8d1131c8fd4708500d3b9eef7bd17a7740885c962a5b",
}

OUT = S9 / "artifacts/block915d_v2R1_failure_case_analysis"
AUDIT = OUT / "stage9_block915d_v2R1_failure_slice_authority_audit.json"
REPORT = OUT / "stage9_block915d_v2R1_failure_case_report.json"
SEAL = OUT / "stage9_block915d_v2R1_failure_case_seal.json"

REQUESTED_SLICES = (
    "beam_boundary",
    "FoV_edge",
    "lane_change",
    "braking",
    "intersection",
    "pedestrian_crossing",
    "cyclist_maneuver_or_crossing",
    "occlusion",
    "missing_tracks",
    "low_LiDAR_count",
    "high_angular_velocity",
    "dynamic_blocker",
    "low_criticality_high_beam_uncertainty",
    "high_criticality_low_beam_uncertainty",
    "high_criticality_high_beam_uncertainty",
    "oncoming_vehicle",
    "glare_source_near_target_angular_location",
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


def recursively_collect_keys(x: Any, out: Counter[str]) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            out[str(k)] += 1
            recursively_collect_keys(v, out)
    elif isinstance(x, list):
        for v in x:
            recursively_collect_keys(v, out)


def disposition(
    status: str,
    reason: str,
    *,
    carrier_present: bool = False,
    threshold_present: bool = False,
    analysis_executed: bool = False,
) -> dict[str, Any]:
    return {
        "status": status,
        "carrier_present": carrier_present,
        "prefrozen_membership_or_threshold_present": threshold_present,
        "analysis_executed": analysis_executed,
        "reason": reason,
        "post_FORMAL_membership_rule_invented": False,
    }


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15D-v2R1 — FAILURE-CASE ANALYSIS")
    print("REQUESTED-SLICE AUTHORITY AUDIT | NO NEW THRESHOLDS | NO NEW RNG | NO MODEL/POLICY RERUN")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    est = load_json(P915A_EST)
    contract = load_json(P915A_CONTRACT)
    bsum = load_json(P915B_SUMMARY)
    cseal = load_json(P915C_SEAL)

    req(
        est.get("status") == "FROZEN_CURRENT_FORMAL_ESTIMAND_REGISTRY_PRE_INFERENCE_REPAIRED",
        "9.15A estimand status drift",
    )
    fs = est.get("robustness", {}).get("failure_slice_registry")
    req(isinstance(fs, dict), "9.15A failure-slice registry missing")
    req(set(fs.get("requested_by_stage9_plan", [])) == set(REQUESTED_SLICES),
        "9.15A failure-slice registry drift")
    req(
        fs.get("post_FORMAL_slice_threshold_invention_forbidden") is True,
        "9.15A post-FORMAL slice-threshold guard drift",
    )
    req(
        bsum.get("status") == "CURRENT_FORMAL_H1_H6_STATISTICS_COMPLETE",
        "9.15B summary status drift",
    )
    req(
        cseal.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915C_V2_ROBUSTNESS",
        "9.15C seal status drift",
    )

    print("\n===== A. FULL CURRENT 9.13 FAILURE-CARRIER AUDIT =====")

    files = sorted(OUTCOME_DIR.glob("*.json"))
    req(len(files) == 26209, f"outcome cardinality drift: {len(files)}")

    top_keys = Counter()
    all_keys = Counter()
    primary_status = Counter()
    h1_geometry_mode = Counter()
    h6_class_scene_presence = Counter()
    h6_class_actor_total = Counter()

    scenes_with_h5_support = 0
    scenes_with_h6_zero_eligible = 0
    h6_role_diagnostics_present = 0
    h6_role_diagnostics_missing = 0
    scenes_with_c3_target_among_diagnostic_present = 0
    scenes_with_h6_pedestrian = 0
    scenes_with_h6_cyclist = 0
    scenes_with_h6_vehicle = 0

    # Explicit requested-slice names/near-synonyms that would count as materialized
    # membership carriers if they existed in the final outcome schema.
    requested_field_tokens = {
        "beam_boundary": ("beam_boundary", "beam_edge", "boundary_receiver"),
        "FoV_edge": ("fov_edge", "field_of_view_edge"),
        "lane_change": ("lane_change", "lane_changing"),
        "braking": ("braking", "sudden_braking", "hard_brake"),
        "intersection": ("intersection",),
        "pedestrian_crossing": ("pedestrian_crossing", "ped_crossing"),
        "cyclist_maneuver_or_crossing": ("cyclist_crossing", "cyclist_maneuver"),
        "occlusion": ("occlusion", "occluded"),
        "missing_tracks": ("missing_track", "track_missing"),
        "low_LiDAR_count": ("lidar_count", "low_lidar"),
        "high_angular_velocity": ("angular_velocity", "yaw_rate", "high_angular"),
        "dynamic_blocker": ("dynamic_blocker", "blocker_dynamic"),
        "low_criticality_high_beam_uncertainty": ("low_criticality_high_beam_uncertainty",),
        "high_criticality_low_beam_uncertainty": ("high_criticality_low_beam_uncertainty",),
        "high_criticality_high_beam_uncertainty": ("high_criticality_high_beam_uncertainty",),
        "oncoming_vehicle": ("oncoming", "oncoming_vehicle"),
        "glare_source_near_target_angular_location": (
            "glare_source_near_target",
            "glare_target_angular_separation",
        ),
    }

    for i, path in enumerate(files):
        j = load_json(path)
        req(int(j.get("formal_ordinal", -1)) == i,
            f"formal ordinal drift: {path.name}")
        req(j.get("status") == "COMPLETE_FINAL_EVALUATOR_H1_H6",
            f"outcome status drift: {path.name}")

        top_keys.update(j.keys())
        recursively_collect_keys(j, all_keys)

        p = j.get("primary")
        req(isinstance(p, dict), f"primary missing: {path.name}")
        primary_status[str(p.get("status"))] += 1

        for r in p.get("H1_records", []):
            gm = r.get("geometry_mode")
            if gm is not None:
                h1_geometry_mode[str(gm)] += 1

        class_counts = j.get("H6_matched_eligible_class_counts")
        req(isinstance(class_counts, dict), f"H6 class counts missing: {path.name}")
        for cls, n0 in class_counts.items():
            n = int(n0)
            req(n >= 0, f"negative H6 class count: {path.name}")
            h6_class_actor_total[str(cls)] += n
            if n > 0:
                h6_class_scene_presence[str(cls)] += 1

        if int(class_counts.get("TYPE_PEDESTRIAN", 0)) > 0:
            scenes_with_h6_pedestrian += 1
        if int(class_counts.get("TYPE_CYCLIST", 0)) > 0:
            scenes_with_h6_cyclist += 1
        if int(class_counts.get("TYPE_VEHICLE", 0)) > 0:
            scenes_with_h6_vehicle += 1

        # H6_matched_eligible_count is the final top-level H6 carrier and was already
        # validated across all 26,209 scenes by 9.15B.  H6_role_binding_diagnostics is
        # supplemental provenance from the role-binding recovery and is not mandatory
        # in every final outcome JSON.
        top_h6_n = j.get("H6_matched_eligible_count")
        req(isinstance(top_h6_n, int) and top_h6_n >= 0,
            f"top-level H6_matched_eligible_count missing/invalid: {path.name}")
        req(top_h6_n == sum(int(v) for v in class_counts.values()),
            f"H6 top-level/class-count mismatch: {path.name}")
        if top_h6_n == 0:
            scenes_with_h6_zero_eligible += 1

        h6d = j.get("H6_role_binding_diagnostics")
        if h6d is None:
            h6_role_diagnostics_missing += 1
        else:
            req(isinstance(h6d, dict), f"H6 diagnostics type drift: {path.name}")
            h6_role_diagnostics_present += 1
            req(
                int(h6d.get("H6_matched_eligible_count", -1)) == top_h6_n,
                f"H6 diagnostic/top-level count mismatch: {path.name}",
            )
            req(h6d.get("H6_metric_universe_changed") is False,
                f"H6 metric universe drift: {path.name}")
            req(h6d.get("role_sets_forced_equal") is False,
                f"H6 role-set equality drift: {path.name}")
            if int(h6d.get("C3_target_count", 0)) > 0:
                scenes_with_c3_target_among_diagnostic_present += 1

        h5 = j.get("H5_raw_counts")
        req(isinstance(h5, dict), f"H5 counts missing: {path.name}")
        if any(
            int(q.get("n", 0)) > 0
            for by_baseline in h5.values()
            for q in by_baseline.values()
        ):
            scenes_with_h5_support += 1

        if (i + 1) % 5000 == 0:
            print(f"FAILURE-CARRIER AUDIT {i+1}/26209", flush=True)

    # After the COMPLETE scan, discover any schema-key names that look like requested
    # failure-slice carriers. We do NOT auto-authorize them: semantic equivalence must be
    # reviewed before analysis. This avoids both silent false positives and false negatives.
    all_key_names_lower = {k.lower() for k in all_keys.keys()}
    semantic_candidate_hits = {}
    for slice_name, tokens in requested_field_tokens.items():
        hits = sorted({
            actual
            for actual in all_key_names_lower
            for token in tokens
            if token in actual
        })
        semantic_candidate_hits[slice_name] = hits

    candidates = {k: v for k, v in semantic_candidate_hits.items() if v}
    print("REQUESTED_SLICE_SCHEMA_CANDIDATE_FIELDS =", candidates, flush=True)
    req(
        not candidates,
        "candidate requested-slice schema fields were found; fail closed for manual semantic review "
        "instead of declaring the slice non-evaluable automatically",
    )

    # For authorization we require exact semantic membership authority, not proxies.
    audit_diagnostics = {
        "FORMAL_scene_count": 26209,
        "primary_status_counts": dict(sorted(primary_status.items())),
        "H1_geometry_mode_values": dict(sorted(h1_geometry_mode.items())),
        "H6_class_scene_presence": dict(sorted(h6_class_scene_presence.items())),
        "H6_class_actor_totals": dict(sorted(h6_class_actor_total.items())),
        "scenes_with_H6_vehicle_presence": scenes_with_h6_vehicle,
        "scenes_with_H6_pedestrian_presence": scenes_with_h6_pedestrian,
        "scenes_with_H6_cyclist_presence": scenes_with_h6_cyclist,
        "scenes_with_H6_zero_matched_eligible": scenes_with_h6_zero_eligible,
        "H6_role_binding_diagnostics_present_scene_count": h6_role_diagnostics_present,
        "H6_role_binding_diagnostics_missing_scene_count": h6_role_diagnostics_missing,
        "scenes_with_C3_target_count_gt_0_among_diagnostic_present_only":
            scenes_with_c3_target_among_diagnostic_present,
        "C3_target_scene_count_is_complete_FORMAL_statistic": False,
        "scenes_with_any_H5_support": scenes_with_h5_support,
        "requested_slice_schema_candidate_fields": semantic_candidate_hits,
    }

    # ---------------------------------------------------------------------------------------------
    # Requested slice dispositions.
    #
    # Important: "presence" is not "crossing"; invalid/no-receiver is not "missing tracks";
    # C3 target count is not "dynamic blocker"; H1 geometry_mode is not beam-boundary membership.
    # ---------------------------------------------------------------------------------------------
    slices: dict[str, Any] = {}

    slices["beam_boundary"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_MEMBERSHIP_RULE",
        "Final 9.13 carrier has H1 geometry_mode but no exact beam-boundary membership field "
        "or frozen numerical distance-to-beam-boundary band for failure slicing.",
        carrier_present=False,
        threshold_present=False,
    )
    slices["FoV_edge"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_MEMBERSHIP_RULE",
        "No exact FoV-edge membership field or frozen edge-band width is bound in the 9.15 authority.",
    )
    slices["lane_change"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_MEMBERSHIP_RULE",
        "No frozen lane-change slice label is materialized in the final 9.13 outcome carrier.",
    )
    slices["braking"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_MEMBERSHIP_RULE",
        "No frozen sudden-braking/braking slice label or preregistered kinematic threshold is materialized.",
    )
    slices["intersection"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_MEMBERSHIP_RULE",
        "No exact frozen intersection-membership field is materialized in final 9.13 outcomes.",
    )
    slices["pedestrian_crossing"] = disposition(
        "NOT_EVALUABLE_PRESENCE_IS_NOT_CROSSING",
        "H6 carries TYPE_PEDESTRIAN presence counts, but presence does not establish pedestrian crossing.",
        carrier_present=True,
        threshold_present=False,
    )
    slices["cyclist_maneuver_or_crossing"] = disposition(
        "NOT_EVALUABLE_PRESENCE_IS_NOT_MANEUVER_OR_CROSSING",
        "H6 carries TYPE_CYCLIST presence counts, but presence does not establish maneuver/crossing.",
        carrier_present=True,
        threshold_present=False,
    )
    slices["occlusion"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_MEMBERSHIP_RULE",
        "No exact frozen occlusion slice label is materialized in the final 9.13 carrier.",
    )
    slices["missing_tracks"] = disposition(
        "NOT_EVALUABLE_STATUS_IS_NOT_EQUIVALENT_TO_MISSING_TRACKS",
        "NO_ELIGIBLE_RECEIVER and ACTOR_1S_INVALID are frozen evaluator dispositions, "
        "not an authorized substitute for a missing-track slice.",
        carrier_present=True,
        threshold_present=False,
    )
    slices["low_LiDAR_count"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_NUMERIC_THRESHOLD",
        "No frozen low-LiDAR-count threshold and no authorized slice-membership field are bound in 9.15A.",
    )
    slices["high_angular_velocity"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_NUMERIC_THRESHOLD",
        "No frozen angular-velocity threshold or authorized high-angular-velocity membership field is bound.",
    )
    slices["dynamic_blocker"] = disposition(
        "NOT_EVALUABLE_C3_TARGET_IS_NOT_DYNAMIC_BLOCKER",
        "C3 target-related role-binding diagnostics exist for some final outcomes, but they are "
        "supplemental/incomplete as a FORMAL scene-membership carrier and are not semantically "
        "identical to the requested dynamic-blocker slice.",
        carrier_present=True,
        threshold_present=False,
    )
    slices["low_criticality_high_beam_uncertainty"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_LOW_HIGH_SLICE_THRESHOLDS",
        "The design names this slice but the current authority does not bind low/high thresholds for both axes.",
    )
    slices["high_criticality_low_beam_uncertainty"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_LOW_HIGH_SLICE_THRESHOLDS",
        "The design names this slice but the current authority does not bind low/high thresholds for both axes.",
    )
    slices["high_criticality_high_beam_uncertainty"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_LOW_HIGH_SLICE_THRESHOLDS",
        "The design names this slice but the current authority does not bind low/high thresholds for both axes.",
    )
    slices["oncoming_vehicle"] = disposition(
        "NOT_EVALUABLE_NO_MATERIALIZED_FROZEN_SCENE_MEMBERSHIP",
        "The final Stage9 outcome carrier contains vehicle presence counts but no oncoming-vehicle scene membership. "
        "A new actor-level reconstruction is not introduced here after FORMAL.",
        carrier_present=True,
        threshold_present=False,
    )
    slices["glare_source_near_target_angular_location"] = disposition(
        "NOT_EVALUABLE_NO_PREFROZEN_ANGULAR_PROXIMITY_RULE",
        "No frozen angular-nearness threshold or exact scene-membership field is bound in the current 9.15 authority.",
    )

    req(set(slices) == set(REQUESTED_SLICES), "requested-slice disposition coverage drift")
    req(not any(v["analysis_executed"] for v in slices.values()),
        "unexpected requested-slice analysis executed")

    audit = {
        "schema": "stage9_block915d_v2R1_failure_slice_authority_audit_v1",
        "status": "PASS_FAIL_CLOSED_FAILURE_SLICE_AUTHORITY_AUDIT",
        "authority": {
            "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage915a_estimand_registry_sha256": EXPECTED[str(P915A_EST)],
            "stage915a_contract_sha256": EXPECTED[str(P915A_CONTRACT)],
            "stage915a_seal_sha256": EXPECTED[str(P915A_SEAL)],
            "stage915b_seal_sha256": EXPECTED[str(P915B_SEAL)],
            "stage915c_seal_sha256": EXPECTED[str(P915C_SEAL)],
        },
        "requested_slices": list(REQUESTED_SLICES),
        "requested_slice_count": len(REQUESTED_SLICES),
        "outcome_carrier_audit": audit_diagnostics,
        "slice_dispositions": slices,
        "repair_provenance": {
            "failed_v2_runner_sha256": "42b247fb7d641564eea417623f741579e2c9c83a99b144b26a6b77f4c804f641",
            "failed_v2_reason": (
                "validator incorrectly required supplemental H6_role_binding_diagnostics "
                "in every final 9.13 outcome"
            ),
            "scientific_dispositions_changed": False,
        },
        "post_FORMAL_new_thresholds_defined": False,
        "post_FORMAL_new_membership_rules_defined": False,
        "model_or_policy_rerun": False,
        "new_RNG": False,
        "new_bootstrap": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT, audit)

    # Supplemental materialized diagnostics are reported because they are real frozen failure /
    # support states, but explicitly NOT as replacements for the requested failure-case slices.
    report = {
        "schema": "stage9_block915d_v2R1_failure_case_report_v1",
        "status": "FAILURE_CASE_BLOCK_COMPLETE_REQUESTED_SLICES_NOT_EVALUABLE_WITHOUT_NEW_POST_FORMAL_DEFINITIONS",
        "authority_audit_sha256": sha256_path(AUDIT),
        "requested_failure_case_analysis": {
            "requested_slice_count": len(REQUESTED_SLICES),
            "evaluable_requested_slice_count": 0,
            "not_evaluable_requested_slice_count": len(REQUESTED_SLICES),
            "metric_or_slice_substitution": False,
            "dispositions": slices,
        },
        "materialized_diagnostics_not_substitutes": {
            "primary_status_counts": audit_diagnostics["primary_status_counts"],
            "H1_geometry_mode_values": audit_diagnostics["H1_geometry_mode_values"],
            "H6_class_scene_presence": audit_diagnostics["H6_class_scene_presence"],
            "H6_class_actor_totals": audit_diagnostics["H6_class_actor_totals"],
            "scenes_with_H6_zero_matched_eligible": scenes_with_h6_zero_eligible,
            "H6_role_binding_diagnostics_present_scene_count": h6_role_diagnostics_present,
            "H6_role_binding_diagnostics_missing_scene_count": h6_role_diagnostics_missing,
            "scenes_with_C3_target_count_gt_0_among_diagnostic_present_only":
                scenes_with_c3_target_among_diagnostic_present,
            "scenes_with_any_H5_support": scenes_with_h5_support,
            "interpretation": (
                "These are frozen carrier/support diagnostics only. They are not relabeled "
                "as beam-boundary, crossing, occlusion, missing-track, dynamic-blocker, "
                "oncoming, or other requested failure-case memberships."
            ),
        },
        "scientific_interpretation": (
            "The Stage-9 design requests a broad failure-slice analysis, but the current frozen "
            "pre-FORMAL authority does not bind exact membership rules/thresholds for those slices. "
            "After FORMAL opening, new thresholds or semantic proxies are not introduced. "
            "The limitation is retained for Stage 9.16 reporting."
        ),
        "repair_provenance": {
            "failed_v2_runner_sha256": "42b247fb7d641564eea417623f741579e2c9c83a99b144b26a6b77f4c804f641",
            "scientific_dispositions_changed": False,
        },
        "retuning": False,
        "new_thresholds": False,
        "new_membership_rules": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "model_or_policy_rerun": False,
        "next_block": "9.15E_LATENCY_RESOURCES",
    }
    atomic_json(REPORT, report)

    seal = {
        "schema": "stage9_block915d_v2R1_failure_case_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915D_V2R1_FAILURE_CASE_ANALYSIS",
        "stage915b_seal_sha256": EXPECTED[str(P915B_SEAL)],
        "stage915c_seal_sha256": EXPECTED[str(P915C_SEAL)],
        "authority_audit_sha256": sha256_path(AUDIT),
        "report_sha256": sha256_path(REPORT),
        "requested_slice_count": len(REQUESTED_SLICES),
        "evaluable_requested_slice_count": 0,
        "failed_v2_runner_sha256": "42b247fb7d641564eea417623f741579e2c9c83a99b144b26a6b77f4c804f641",
        "validator_repair_only": True,
        "scientific_dispositions_changed": False,
        "requested_slices_not_silently_redefined": True,
        "metric_or_slice_substitution": False,
        "post_FORMAL_new_thresholds_defined": False,
        "post_FORMAL_new_membership_rules_defined": False,
        "new_RNG": False,
        "new_bootstrap": False,
        "retuning": False,
        "next_block": "9.15E_LATENCY_RESOURCES",
    }
    atomic_json(SEAL, seal)

    print("FAILURE_SLICE_AUTHORITY_AUDIT = PASS")
    print("REQUESTED_SLICE_COUNT =", len(REQUESTED_SLICES))
    print("EVALUABLE_REQUESTED_SLICE_COUNT = 0")
    print("REQUESTED_SLICES_SILENTLY_REDEFINED = FALSE")
    print("POST_FORMAL_NEW_THRESHOLDS = FALSE")
    print("POST_FORMAL_NEW_MEMBERSHIP_RULES = FALSE")
    print("NEW_RNG = FALSE")
    print("NEW_BOOTSTRAP = FALSE")
    print("STAGE 9.15D-v2R1 = COMPLETE")
    print("AUDIT_SHA256 =", sha256_path(AUDIT))
    print("REPORT_SHA256 =", sha256_path(REPORT))
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15E_LATENCY_RESOURCES")
    print("=" * 124)


if __name__ == "__main__":
    main()

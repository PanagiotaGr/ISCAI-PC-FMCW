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

import numpy as np

ROOT = Path("/home/agni/waymo")
S9 = ROOT / "iscai_stage9"

FORMAL_N = 26209
EXPECTED_EXCLUDED_N = 22
REDUCED_N = FORMAL_N - EXPECTED_EXCLUDED_N

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260910
CI_Q = (0.025, 0.975)
CHUNK = 25

# -------------------------------------------------------------------------------------------------
# Frozen final outcome authority.
# -------------------------------------------------------------------------------------------------
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
OUTCOME_DIR = B913 / "scenario_outcomes"
P913_MANIFEST = B913 / "stage9_block913_v1_3_FINAL_H1_H6_manifest.jsonl"
P913_SUMMARY = B913 / "stage9_block913_v1_3_FINAL_H1_H6_summary.json"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"
REC1 = B913 / "FINAL_EVALUATOR_TECHNICAL_RECOVERY_AMENDMENT.json"
REC2 = B913 / "FINAL_EVALUATOR_TECHNICAL_RECOVERY_AMENDMENT_R2.json"

# Frozen 9.15A pre-inference authority.
B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_EVIDENCE = B915A / "stage9_block915a_v2R4_h1_schema_status_censoring_evidence.json"
P915A_EST = B915A / "stage9_block915a_v2R4_estimand_registry.json"
P915A_CONTRACT = B915A / "stage9_block915a_v2R4_statistics_contract.json"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

# Current full-cohort primary statistical authority. 9.15G must never override it.
B915B = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
P915B_SUMMARY = B915B / "stage9_block915b_v2_H1_H6_statistics_summary.json"
P915B_SEAL = B915B / "stage9_block915b_v2_statistics_seal.json"

# Frozen 9.15 closure remains primary.
B915F = S9 / "artifacts/block915f_v2_final_stage915_closure"
P915F_SUMMARY = B915F / "stage9_block915f_v2_final_stage915_summary.json"
P915F_SEAL = B915F / "stage9_block915f_v2_final_stage915_seal.json"

# H1 confirmatory authority boundary.
BH1OLD = S9 / "artifacts/block913b2c_h1_formal_disposition_pre_gt"
P_H1_OLD_SEAL = BH1OLD / "stage9_block913b2c_h1_formal_disposition_seal.json"

# Exact 9.15B implementation identity whose inferential design is being reapplied.
P915B_RUNNER = S9 / "scripts/run_stage9_block915b_v2_current_formal_H1_H6_bootstrap.py"

EXPECTED = {
    str(P913_MANIFEST): "752600ddf126cabd87010d8277b7fe040d18b3e8d8f2490adcc2e9ca0ceb63c4",
    str(P913_SUMMARY): "292d515a3f52c82ad08460e1122a805d236a8aeaefa82271ff84664c2cddfe60",
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(REC1): "021f9e1b200fb8b4fed2b5fdbf22fc224740f670d9aff647b16c4c721cfc0756",
    str(REC2): "ea05e6771e4b47d3938e3bf6621a748bf206d3dc52c3c4c4644db500dce00e02",

    str(P915A_EVIDENCE): "c96b9f969c5e6d1f1e760313d5c58d279746044fb3846cba63c67a1a7bc785a0",
    str(P915A_EST): "4ecc05f12b5c11e46a976b161fd7866d30bda76f287d4904d7cd52e95a71e37c",
    str(P915A_CONTRACT): "91362b13413565501d3d33905d78bc12f14bb9c8b3739783e899cbc6088c7b03",
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",

    str(P915B_SUMMARY): "a203141a1fcf0af5355fc35718493c1317e384445f8c12d437c8f15a1ebed175",
    str(P915B_SEAL): "8b5f8ded62e6c5fe35420830e4c5a5608c33afdad2fea4d07e16205ece7aae67",
    str(P915B_RUNNER): "f5275a5b7aa66a1f7a590440bfcc7e8c0884f820ba7ba8fa04c4086c238b31f0",

    str(P915F_SUMMARY): "15f974474c9e455ee48c72ebb7c05d5be5c9b87a7691f6eda0b14ac9bb5bd169",
    str(P915F_SEAL): "02d4f36951254ca0f140b6682ee3d0716110e362c6c922d715b44ef3888bf065",

    str(P_H1_OLD_SEAL): "f86829c71ab5b410155e0bbeabe256bd280b17d7037576356bde9253d0306442",
}

OUT = S9 / "artifacts/block915g_v1_technical_recovery_exclusion_sensitivity"
PREFLIGHT = OUT / "stage9_block915g_v1_prebootstrap_gate.json"
REPLICATES = OUT / "stage9_block915g_v1_bootstrap_replicates.npz"
SUMMARY = OUT / "stage9_block915g_v1_technical_recovery_sensitivity_summary.json"
SEAL = OUT / "stage9_block915g_v1_technical_recovery_sensitivity_seal.json"

H1_ARMS = ("RAW", "CALIBRATED")
CODEBOOKS = (16, 32, 64)
HORIZONS = (0.1, 0.3, 0.5, 1.0)
H1_STRATA = tuple((cb, h) for cb in CODEBOOKS for h in HORIZONS)
H1_INDEX = {k: i for i, k in enumerate(H1_STRATA)}

H5_PROFILES = ("H_fast", "H_nominal", "H_slow")
H5_BASELINES = ("original_reactive_ADB", "Stage6_V3_predictive_ADB")
H5_COMPARISONS = tuple((p, b) for p in H5_PROFILES for b in H5_BASELINES)
H5_INDEX = {k: i for i, k in enumerate(H5_COMPARISONS)}

H6_ACTIONS = ("reactive", "Stage6_V3", "recovery")
H6_BASELINES = ("reactive", "Stage6_V3")
H6_METRICS = (
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "over_masking_area",
    "road_illumination_retention",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
)
H6_COMPARISONS = tuple((b, m) for b in H6_BASELINES for m in H6_METRICS)
H6_INDEX = {k: i for i, k in enumerate(H6_COMPARISONS)}

REQUIRED_COMM_POLICIES = (
    "proposed_C2",
    "fixed_q_0.95",
    "fixed_q_0.99",
    "uncertainty_only_adaptive_topk",
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


def atomic_npz(path: Path, **arrays) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        actual = Path(tmp)
        if not actual.exists() and Path(tmp + ".npz").exists():
            actual = Path(tmp + ".npz")
        os.replace(actual, path)
    finally:
        for q in (Path(tmp), Path(tmp + ".npz")):
            if q.exists():
                q.unlink()


def hash_gate() -> None:
    for raw, wanted in EXPECTED.items():
        p = Path(raw)
        req(p.is_file(), f"missing frozen authority: {p}")
        got = sha256_path(p)
        req(got == wanted, f"SHA256 drift {p}: {got} != {wanted}")
        print("EXACT PASS", p)


def quantile_ci(x: np.ndarray) -> tuple[float, float]:
    q = np.quantile(np.asarray(x, dtype=np.float64), CI_Q, method="linear")
    return float(q[0]), float(q[1])


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    req(n > 0, "Wilson interval requires positive denominator")
    phat = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / den
    half = z * math.sqrt((phat * (1.0 - phat) / n) + z2 / (4.0 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def h1_point(hits_raw: np.ndarray, hits_cal: np.ndarray, den: np.ndarray) -> dict[str, Any]:
    req(np.all(den > 0), "H1 point estimate has zero-support stratum")
    cov_raw = hits_raw / den
    cov_cal = hits_cal / den
    err_raw = np.abs(cov_raw - 0.95)
    err_cal = np.abs(cov_cal - 0.95)
    delta = err_cal - err_raw
    return {
        "macro_abs_coverage_error_RAW": float(np.mean(err_raw)),
        "macro_abs_coverage_error_CALIBRATED": float(np.mean(err_cal)),
        "primary_delta_CALIBRATED_minus_RAW": float(np.mean(delta)),
        "strata": {
            f"cb{cb}_h{h:.1f}": {
                "denominator": int(den[i]),
                "coverage_RAW": float(cov_raw[i]),
                "coverage_CALIBRATED": float(cov_cal[i]),
                "abs_error_RAW": float(err_raw[i]),
                "abs_error_CALIBRATED": float(err_cal[i]),
                "delta_abs_error_CALIBRATED_minus_RAW": float(delta[i]),
            }
            for i, (cb, h) in enumerate(H1_STRATA)
        },
    }


def recovery_scene_set() -> tuple[set[str], dict[str, Any]]:
    r1 = load_json(REC1)
    r2 = load_json(REC2)

    req(r1.get("status") == "FROZEN_TECHNICAL_RECOVERY_AUTHORIZATION_BEFORE_REREAD",
        "Recovery1 status drift")
    req(r2.get("status") == "FROZEN_TECHNICAL_RECOVERY2_AUTHORIZATION_BEFORE_REREAD",
        "Recovery2 status drift")
    req(r1.get("baselines_changed") is False and r2.get("baselines_changed") is False,
        "recovery baseline-change drift")
    req(r1.get("policy_retuned") is False and r2.get("policy_retuned") is False,
        "recovery policy-retuning drift")
    req(r1.get("thresholds_retuned") is False and r2.get("thresholds_retuned") is False,
        "recovery threshold-retuning drift")
    req(r1.get("outcome_based_tuning") is False and r2.get("outcome_based_tuning") is False,
        "recovery outcome-tuning drift")

    s1 = {str(x["scenario_id"]) for x in r1["authorized_recovery_scenes"]}
    s2 = {str(x["scenario_id"]) for x in r2["authorized_recovery2_scenes"]}
    union = s1 | s2

    req(len(s1) == 12, f"Recovery1 scene count drift: {len(s1)}")
    req(len(s2) == 12, f"Recovery2 authorization count drift: {len(s2)}")
    req(len(union) == EXPECTED_EXCLUDED_N, f"recovery union drift: {len(union)} != 22")

    return union, {
        "recovery1_scene_count": len(s1),
        "recovery2_authorization_scene_count": len(s2),
        "union_unique_scene_count": len(union),
        "overlap_scene_count": len(s1 & s2),
        "recovery1_sha256": EXPECTED[str(REC1)],
        "recovery2_sha256": EXPECTED[str(REC2)],
    }


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15G-v1 — PREFROZEN TECHNICAL-RECOVERY EXCLUSION SENSITIVITY")
    print("EXCLUDE EXACTLY 22 DOCUMENTED REREAD SCENES | SECONDARY ONLY | SAME FROZEN 10k BOOTSTRAP DESIGN")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    est = load_json(P915A_EST)
    contract = load_json(P915A_CONTRACT)
    aseal = load_json(P915A_SEAL)
    primary_summary = load_json(P915B_SUMMARY)
    closure = load_json(P915F_SUMMARY)
    old_h1 = load_json(P_H1_OLD_SEAL)

    req(
        aseal.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915A_V2R4_PRE_INFERENCE_CONTRACT",
        "9.15A status drift",
    )
    req(
        est["robustness"]["technical_recovery"]["mandatory_secondary_sensitivity"]
        == (
            "repeat headline estimands after excluding the 22 documented physical-reread scenes; "
            "label strictly secondary and never let it override the full-cohort result"
        ),
        "technical-recovery sensitivity authority drift",
    )
    req(
        est["robustness"]["technical_recovery"]["exclude_recovery_scenes_from_primary"] is False,
        "primary exclusion authority drift",
    )
    req(
        est["robustness"]["technical_recovery"]["known_from_9_13_seal"]["unique_scenes_with_physical_reread"]
        == 22,
        "recovery scene-count authority drift",
    )
    req(
        est["robustness"]["technical_recovery"]["known_from_9_13_seal"]["extra_physical_future_GT_read_events"]
        == 24,
        "extra physical GT read count authority drift",
    )

    stats = contract["statistics"]
    req(int(stats["bootstrap_replicates"]) == BOOTSTRAP_REPS, "bootstrap replicate count drift")
    req(int(stats["bootstrap_seed"]) == BOOTSTRAP_SEED, "bootstrap seed drift")
    req(stats["CI_quantiles"] == [0.025, 0.975], "CI quantile drift")
    req(stats["numpy_quantile_method"] == "linear", "quantile method drift")

    req(
        old_h1["H1"]["confirmatory_status"] == "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "old H1 confirmatory boundary drift",
    )
    req(old_h1["H1"]["posthoc_FORMAL_raw_arm_allowed"] is False,
        "H1 posthoc raw-arm prohibition drift")
    req(old_h1["H1"]["new_model_forward_allowed"] is False,
        "H1 model-forward prohibition drift")

    recovery_sids, recovery_meta = recovery_scene_set()

    print("\n===== A. FULL-CARRIER GATE + EXACT 22-SCENE EXCLUSION =====")

    files = sorted(OUTCOME_DIR.glob("*.json"))
    req(len(files) == FORMAL_N, f"FORMAL outcome cardinality drift: {len(files)}")

    # First pass: verify exact recovery membership against final outcomes.
    flagged = set()
    seen_recovery_union = set()
    physical_read_count_hist = Counter()

    for full_i, path in enumerate(files):
        j = load_json(path)
        req(int(j.get("formal_ordinal", -1)) == full_i, f"formal ordinal drift: {path.name}")
        sid = str(j["scenario_id"])
        rn = j.get("future_GT_physical_read_number")
        physical_read_count_hist[rn] += 1

        if j.get("technical_recovery_reread") is True:
            flagged.add(sid)
        if sid in recovery_sids:
            seen_recovery_union.add(sid)

    req(flagged == recovery_sids,
        f"final-outcome reread marker set != amendment union: flagged={len(flagged)} union={len(recovery_sids)}")
    req(seen_recovery_union == recovery_sids, "not all recovery-authorized scenes found in final outcomes")

    extra_reads = sum(
        (int(k) - 1) * int(v)
        for k, v in physical_read_count_hist.items()
        if isinstance(k, int) and k >= 1
    )
    req(extra_reads == 24, f"extra physical read event drift: {extra_reads} != 24")

    included_files = []
    excluded_rows = []
    for path in files:
        j = load_json(path)
        sid = str(j["scenario_id"])
        if sid in recovery_sids:
            excluded_rows.append({
                "formal_ordinal": int(j["formal_ordinal"]),
                "scenario_id": sid,
                "future_GT_physical_read_number": int(j["future_GT_physical_read_number"]),
                "technical_recovery_amendment_sha256": j.get("technical_recovery_amendment_sha256"),
            })
        else:
            included_files.append(path)

    req(len(excluded_rows) == EXPECTED_EXCLUDED_N, "excluded row count drift")
    req(len(included_files) == REDUCED_N, "reduced cohort size drift")

    # Reduced-cohort paired carriers.
    h1_den = np.zeros((REDUCED_N, len(H1_STRATA)), dtype=np.uint8)
    h1_raw = np.zeros_like(h1_den)
    h1_cal = np.zeros_like(h1_den)

    h3_k_diff = np.full((REDUCED_N, len(CODEBOOKS)), np.nan, dtype=np.float64)
    h3_overhead_diff = np.full_like(h3_k_diff, np.nan)

    h5_n = np.zeros((REDUCED_N, len(H5_COMPARISONS)), dtype=np.int32)
    h5_base = np.zeros_like(h5_n)
    h5_recovery = np.zeros_like(h5_n)

    h6_delta = np.full((REDUCED_N, len(H6_COMPARISONS)), np.nan, dtype=np.float64)

    status_counts = Counter()
    critical_valid = 0
    critical_events = 0
    communication_valid_scenes = 0
    h1_record_count = 0

    required_h1_keys = {
        "arm", "codebook_size", "horizon_s", "requested_q", "containment_hit",
        "selected_k", "physical_probe_count",
    }
    required_comm_keys = {
        "policy", "codebook_size", "K", "probing_overhead_fraction",
        "beam_outage", "critical_label_valid", "Y_crit_1s",
    }

    for ri, path in enumerate(included_files):
        j = load_json(path)
        req(j.get("status") == "COMPLETE_FINAL_EVALUATOR_H1_H6",
            f"outer outcome status drift: {path.name}")
        req(j.get("C1_MC_recomputed") is False, f"C1 recomputed: {path.name}")
        req(j.get("C2_solver_recomputed") is False, f"C2 recomputed: {path.name}")
        req(j.get("model_forward_executed") is False, f"model forward drift: {path.name}")
        req(j.get("policy_retuned") is False, f"policy retuned: {path.name}")
        req(j.get("thresholds_retuned") is False, f"threshold retuned: {path.name}")

        primary = j.get("primary")
        req(isinstance(primary, dict), f"missing primary object: {path.name}")
        pstatus = str(primary.get("status"))
        status_counts[pstatus] += 1

        recs = primary.get("H1_records", [])
        req(isinstance(recs, list), f"H1_records not list: {path.name}")
        seen_h1 = set()
        for r in recs:
            req(required_h1_keys.issubset(r), f"H1 record schema drift: {path.name}")
            arm = str(r["arm"])
            cb = int(r["codebook_size"])
            h = float(r["horizon_s"])
            q = float(r["requested_q"])
            req(arm in H1_ARMS, f"unexpected H1 arm: {path.name}")
            req((cb, h) in H1_INDEX, f"unexpected H1 stratum: {path.name}")
            req(abs(q - 0.95) <= 1e-15, f"H1 requested_q drift: {path.name}")
            key = (arm, cb, h)
            req(key not in seen_h1, f"duplicate H1 record: {path.name} {key}")
            seen_h1.add(key)
            s = H1_INDEX[(cb, h)]
            h1_den[ri, s] = 1
            if bool(r["containment_hit"]):
                if arm == "RAW":
                    h1_raw[ri, s] = 1
                else:
                    h1_cal[ri, s] = 1
            h1_record_count += 1

        for cb, h in H1_STRATA:
            has_raw = ("RAW", cb, h) in seen_h1
            has_cal = ("CALIBRATED", cb, h) in seen_h1
            req(has_raw == has_cal, f"H1 arm-specific missingness: {path.name} cb={cb} h={h}")

        label = primary.get("criticality_label")
        comm = primary.get("communication_records", [])
        req(isinstance(comm, list), f"communication_records not list: {path.name}")

        if pstatus == "EVALUATED_FUTURE_GT":
            req(isinstance(label, dict) and label.get("label_valid") is True,
                f"valid-eval critical label drift: {path.name}")
            y = int(label["Y_crit_1s"])
            req(y in (0, 1), f"invalid Y_crit_1s: {path.name}")
            critical_valid += 1
            critical_events += y

            by_key = {}
            for r in comm:
                req(required_comm_keys.issubset(r), f"communication schema drift: {path.name}")
                key = (str(r["policy"]), int(r["codebook_size"]))
                req(key not in by_key, f"duplicate communication policy/codebook: {path.name} {key}")
                by_key[key] = r
                req(bool(r["critical_label_valid"]) is True, f"communication label-valid drift: {path.name}")
                req(int(r["Y_crit_1s"]) == y, f"communication Ycrit drift: {path.name}")

            for cb_i, cb in enumerate(CODEBOOKS):
                for policy in REQUIRED_COMM_POLICIES:
                    req((policy, cb) in by_key,
                        f"missing required communication record: {path.name} {policy}/{cb}")

                rp = by_key[("proposed_C2", cb)]
                rq = by_key[("fixed_q_0.99", cb)]
                h3_k_diff[ri, cb_i] = float(rp["K"]) - float(rq["K"])
                h3_overhead_diff[ri, cb_i] = (
                    float(rp["probing_overhead_fraction"])
                    - float(rq["probing_overhead_fraction"])
                )
            communication_valid_scenes += 1

        elif pstatus == "EVALUATED_FUTURE_GT_ACTOR_1S_INVALID":
            req(isinstance(label, dict) and label.get("label_valid") is False,
                f"actor-invalid label drift: {path.name}")
            req(comm == [], f"actor-invalid scene unexpectedly has communication records: {path.name}")

        elif pstatus == "NOT_EVALUATED_NO_ELIGIBLE_RECEIVER":
            req(label is None, f"no-receiver critical label drift: {path.name}")
            req(comm == [], f"no-receiver communication drift: {path.name}")
        else:
            raise FailClosed(f"unknown primary status {pstatus!r}: {path.name}")

        h5 = j.get("H5_raw_counts")
        req(isinstance(h5, dict) and set(h5) == set(H5_PROFILES),
            f"H5 profile schema drift: {path.name}")
        for profile in H5_PROFILES:
            req(set(h5[profile]) == set(H5_BASELINES),
                f"H5 baseline schema drift: {path.name}/{profile}")
            for baseline in H5_BASELINES:
                q = h5[profile][baseline]
                req(set(q) >= {"n", "baseline_violations", "recovery_violations"},
                    f"H5 count schema drift: {path.name}/{profile}/{baseline}")
                c = H5_INDEX[(profile, baseline)]
                n = int(q["n"])
                bv = int(q["baseline_violations"])
                rv = int(q["recovery_violations"])
                req(0 <= bv <= n and 0 <= rv <= n,
                    f"H5 count bounds drift: {path.name}/{profile}/{baseline}")
                h5_n[ri, c] = n
                h5_base[ri, c] = bv
                h5_recovery[ri, c] = rv

        h6 = j.get("H6_metrics")
        req(isinstance(h6, dict) and set(h6) == set(H6_ACTIONS),
            f"H6 action schema drift: {path.name}")
        for action in H6_ACTIONS:
            req(set(h6[action]) == set(H6_METRICS),
                f"H6 metric schema drift: {path.name}/{action}")

        for baseline in H6_BASELINES:
            for metric in H6_METRICS:
                c = H6_INDEX[(baseline, metric)]
                a = float(h6["recovery"][metric])
                b = float(h6[baseline][metric])
                if math.isfinite(a) and math.isfinite(b):
                    h6_delta[ri, c] = a - b

        if (ri + 1) % 5000 == 0:
            print(
                f"REDUCED PREFLIGHT {ri+1}/{REDUCED_N} "
                f"critical_valid={critical_valid} critical_events={critical_events}",
                flush=True,
            )

    req(sum(status_counts.values()) == REDUCED_N, "reduced status count total drift")
    req(critical_events == 0, f"excluded-cohort critical support unexpectedly nonzero: {critical_events}")

    h1_den_tot = h1_den.sum(axis=0, dtype=np.int64)
    h1_raw_tot = h1_raw.sum(axis=0, dtype=np.int64)
    h1_cal_tot = h1_cal.sum(axis=0, dtype=np.int64)
    req(np.all(h1_den_tot > 0), "H1 reduced cohort has zero-support stratum")

    h5_total_n = h5_n.sum(axis=0, dtype=np.int64)
    req(np.all(h5_total_n > 0), "H5 reduced comparison has zero aggregate support")
    h6_finite_support = np.sum(np.isfinite(h6_delta), axis=0, dtype=np.int64)

    for metric in (
        "vehicle_shadow_zone_violation",
        "over_masking_area",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    ):
        c = H6_INDEX[("reactive", metric)]
        req(h6_finite_support[c] > 0, f"H6 reduced core metric zero paired support: {metric}")

    wil_lo, wil_hi = wilson_interval(critical_events, critical_valid)

    preflight = {
        "schema": "stage9_block915g_v1_prebootstrap_gate_v1",
        "status": "PASS_PREFROZEN_TECHNICAL_RECOVERY_EXCLUSION_SENSITIVITY_GATE",
        "role": "MANDATORY_SECONDARY_SENSITIVITY_ONLY",
        "authority": {
            "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
            "recovery1_amendment_sha256": EXPECTED[str(REC1)],
            "recovery2_amendment_sha256": EXPECTED[str(REC2)],
            "stage915a_estimand_registry_sha256": EXPECTED[str(P915A_EST)],
            "stage915a_statistics_contract_sha256": EXPECTED[str(P915A_CONTRACT)],
            "stage915b_primary_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
            "stage915b_primary_seal_sha256": EXPECTED[str(P915B_SEAL)],
            "stage915f_primary_closure_summary_sha256": EXPECTED[str(P915F_SUMMARY)],
            "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
            "old_H1_not_tested_seal_sha256": EXPECTED[str(P_H1_OLD_SEAL)],
            "stage915b_runner_sha256": EXPECTED[str(P915B_RUNNER)],
        },
        "population": {
            "primary_full_cohort_scene_count": FORMAL_N,
            "primary_full_cohort_remains_authoritative": True,
            "excluded_documented_reread_scene_count": EXPECTED_EXCLUDED_N,
            "reduced_secondary_scene_count": REDUCED_N,
            "excluded_scene_ids": sorted(recovery_sids),
            "excluded_rows": sorted(excluded_rows, key=lambda x: x["formal_ordinal"]),
            "extra_physical_future_GT_read_events": extra_reads,
            "recovery_authorization": recovery_meta,
        },
        "reduced_primary_status_counts": dict(sorted(status_counts.items())),
        "H1_total_materialized_records": int(h1_record_count),
        "H1_support_by_stratum": {
            f"cb{cb}_h{h:.1f}": int(h1_den_tot[k])
            for k, (cb, h) in enumerate(H1_STRATA)
        },
        "critical_label_valid_denominator": int(critical_valid),
        "observed_Ccrit_GT_events": int(critical_events),
        "Ccrit_Wilson_95pct": [wil_lo, wil_hi],
        "H3_resource_paired_scene_count": int(np.sum(np.all(np.isfinite(h3_k_diff), axis=1))),
        "H5_support_n": {
            f"{p}__{b}": int(h5_total_n[k])
            for k, (p, b) in enumerate(H5_COMPARISONS)
        },
        "H6_finite_paired_scene_support": {
            f"recovery_minus_{b}__{m}": int(h6_finite_support[k])
            for k, (b, m) in enumerate(H6_COMPARISONS)
        },
        "bootstrap_design": {
            "paired_scenario_cluster_bootstrap": True,
            "replicates": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "RNG_engine": "numpy.random.PCG64",
            "CI_method": "equal_tailed_percentile_bootstrap",
            "CI_level": 0.95,
            "numpy_quantile_method": "linear",
            "raw_primary_resample_indices_reused": False,
            "reason_indices_not_reused": "9.15B replicate NPZ stores statistics, not resample indices",
            "new_draws_authority": "PREFROZEN_MANDATORY_SECONDARY_SENSITIVITY_PLUS_FROZEN_9_15A_STATISTICS_CONTRACT",
        },
        "H1_boundary": {
            "confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
            "sensitivity_role": "SECONDARY_NONCONFIRMATORY_NUMERICAL_SENSITIVITY_ONLY",
            "confirmatory_reactivation_forbidden": True,
        },
        "RNG_not_created_at_gate_write": True,
        "bootstrap_not_started_at_gate_write": True,
        "model_policy_solver_rerun": False,
        "retuning": False,
        "threshold_change": False,
        "baseline_change": False,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(PREFLIGHT, preflight)
    preflight_sha = sha256_path(PREFLIGHT)

    print("RECOVERY_EXCLUSION_PREFLIGHT = PASS")
    print("PRIMARY_FULL_COHORT_N =", FORMAL_N)
    print("EXCLUDED_REREAD_SCENES =", EXPECTED_EXCLUDED_N)
    print("REDUCED_SECONDARY_N =", REDUCED_N)
    print("CRITICAL_EVENTS_REDUCED =", critical_events, "/", critical_valid)
    print("PREFLIGHT_SHA256 =", preflight_sha)

    # Point estimates before RNG.
    h1_pt = h1_point(h1_raw_tot, h1_cal_tot, h1_den_tot)

    h3_pt_k = np.nanmean(h3_k_diff, axis=0)
    h3_pt_o = np.nanmean(h3_overhead_diff, axis=0)

    h5_base_tot = h5_base.sum(axis=0, dtype=np.int64)
    h5_rec_tot = h5_recovery.sum(axis=0, dtype=np.int64)
    h5_pt = (h5_rec_tot / h5_total_n) - (h5_base_tot / h5_total_n)

    h6_pt = np.nanmean(h6_delta, axis=0)

    print("\n===== B. PREFROZEN-AUTHORIZED 10,000-REPLICATE SECONDARY BOOTSTRAP =====")
    print("NEW_RNG_DRAWS = TRUE")
    print("AUTHORITY = PREFROZEN_MANDATORY_SECONDARY_SENSITIVITY")
    print("RNG_ENGINE = numpy.random.PCG64")
    print("BOOTSTRAP_SEED =", BOOTSTRAP_SEED)

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))

    r_h1 = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    r_h1_strata = np.empty((BOOTSTRAP_REPS, len(H1_STRATA)), dtype=np.float64)
    r_h3_k = np.empty((BOOTSTRAP_REPS, len(CODEBOOKS)), dtype=np.float64)
    r_h3_o = np.empty_like(r_h3_k)
    r_h5 = np.empty((BOOTSTRAP_REPS, len(H5_COMPARISONS)), dtype=np.float64)
    r_h6 = np.empty((BOOTSTRAP_REPS, len(H6_COMPARISONS)), dtype=np.float64)

    rep0 = 0
    while rep0 < BOOTSTRAP_REPS:
        bsz = min(CHUNK, BOOTSTRAP_REPS - rep0)
        idx = rng.integers(0, REDUCED_N, size=(bsz, REDUCED_N), dtype=np.int32)

        den = h1_den[idx].sum(axis=1, dtype=np.int64)
        raw = h1_raw[idx].sum(axis=1, dtype=np.int64)
        cal = h1_cal[idx].sum(axis=1, dtype=np.int64)
        req(np.all(den > 0), "H1 sensitivity bootstrap replicate zero-support stratum")
        er = np.abs(raw / den - 0.95)
        ec = np.abs(cal / den - 0.95)
        ds = ec - er
        r_h1[rep0:rep0+bsz] = np.mean(ds, axis=1)
        r_h1_strata[rep0:rep0+bsz, :] = ds

        ksel = h3_k_diff[idx]
        osel = h3_overhead_diff[idx]
        with np.errstate(invalid="ignore"):
            r_h3_k[rep0:rep0+bsz, :] = np.nanmean(ksel, axis=1)
            r_h3_o[rep0:rep0+bsz, :] = np.nanmean(osel, axis=1)

        n = h5_n[idx].sum(axis=1, dtype=np.int64)
        bv = h5_base[idx].sum(axis=1, dtype=np.int64)
        rv = h5_recovery[idx].sum(axis=1, dtype=np.int64)
        req(np.all(n > 0), "H5 sensitivity bootstrap replicate zero denominator")
        r_h5[rep0:rep0+bsz, :] = (rv / n) - (bv / n)

        d = h6_delta[idx]
        with np.errstate(invalid="ignore"):
            x = np.nanmean(d, axis=1)
        req(np.all(np.isfinite(x)), "H6 sensitivity bootstrap replicate zero paired finite support")
        r_h6[rep0:rep0+bsz, :] = x

        rep0 += bsz
        if rep0 % 500 == 0 or rep0 == BOOTSTRAP_REPS:
            print(f"SENSITIVITY_BOOTSTRAP {rep0}/{BOOTSTRAP_REPS}", flush=True)

    atomic_npz(
        REPLICATES,
        H1_primary_delta=r_h1,
        H1_stratum_delta_abs_error=r_h1_strata,
        H3_delta_mean_K=r_h3_k,
        H3_delta_mean_overhead=r_h3_o,
        H5_delta_violation_rate=r_h5,
        H6_delta_scene_mean=r_h6,
    )
    reps_sha = sha256_path(REPLICATES)

    h1_ci = quantile_ci(r_h1)
    h1_stratum_summary = {}
    for k, (cb, h) in enumerate(H1_STRATA):
        lo, hi = quantile_ci(r_h1_strata[:, k])
        h1_stratum_summary[f"cb{cb}_h{h:.1f}"] = {
            "point_delta_abs_error_CALIBRATED_minus_RAW": (
                h1_pt["strata"][f"cb{cb}_h{h:.1f}"]["delta_abs_error_CALIBRATED_minus_RAW"]
            ),
            "CI95": [lo, hi],
        }

    h3_summary = {}
    for k, cb in enumerate(CODEBOOKS):
        klo, khi = quantile_ci(r_h3_k[:, k])
        olo, ohi = quantile_ci(r_h3_o[:, k])
        h3_summary[str(cb)] = {
            "delta_mean_K_proposed_minus_q099": float(h3_pt_k[k]),
            "delta_mean_K_CI95": [klo, khi],
            "delta_mean_overhead_proposed_minus_q099": float(h3_pt_o[k]),
            "delta_mean_overhead_CI95": [olo, ohi],
            "resource_reduction_supported_in_secondary_sensitivity": bool(khi < 0.0 and ohi < 0.0),
        }

    h5_summary = {}
    for k, (profile, baseline) in enumerate(H5_COMPARISONS):
        lo, hi = quantile_ci(r_h5[:, k])
        if hi < 0.0:
            decision = "REDUCTION_SUPPORTED"
        elif lo > 0.0:
            decision = "INCREASE_SUPPORTED"
        else:
            decision = "NO_CLEAR_DIFFERENCE"
        h5_summary[f"{profile}__{baseline}"] = {
            "n": int(h5_total_n[k]),
            "baseline_violations": int(h5_base_tot[k]),
            "recovery_violations": int(h5_rec_tot[k]),
            "point_delta_rate_recovery_minus_baseline": float(h5_pt[k]),
            "CI95": [lo, hi],
            "secondary_sensitivity_decision": decision,
        }

    h6_summary = {}
    for k, (baseline, metric) in enumerate(H6_COMPARISONS):
        lo, hi = quantile_ci(r_h6[:, k])
        h6_summary[f"recovery_minus_{baseline}__{metric}"] = {
            "paired_finite_scene_count": int(h6_finite_support[k]),
            "point_delta": float(h6_pt[k]),
            "CI95": [lo, hi],
        }

    d_vehicle = float(h6_pt[H6_INDEX[("reactive", "vehicle_shadow_zone_violation")]])
    d_overmask = float(h6_pt[H6_INDEX[("reactive", "over_masking_area")]])
    d_ped = float(h6_pt[H6_INDEX[("reactive", "pedestrian_visibility_proxy")]])
    d_cyc = float(h6_pt[H6_INDEX[("reactive", "cyclist_visibility_proxy")]])

    h6_core = {
        "vehicle_shadow_recovery_minus_reactive_lt_0": bool(d_vehicle < 0.0),
        "over_masking_recovery_minus_reactive_le_0p02": bool(d_overmask <= 0.02),
        "pedestrian_visibility_recovery_minus_reactive_ge_minus0p05": bool(d_ped >= -0.05),
        "cyclist_visibility_recovery_minus_reactive_ge_minus0p05": bool(d_cyc >= -0.05),
    }
    h6_core_pass = bool(all(h6_core.values()))

    h1_numerical_status = (
        "NUMERICAL_REDUCTION_SUPPORTED_BY_95PCT_BOOTSTRAP_CI"
        if h1_ci[1] < 0.0
        else "NUMERICAL_REDUCTION_NOT_SUPPORTED_BY_95PCT_BOOTSTRAP_CI"
    )

    summary = {
        "schema": "stage9_block915g_v1_technical_recovery_sensitivity_summary_v1",
        "status": "SECONDARY_TECHNICAL_RECOVERY_EXCLUSION_SENSITIVITY_COMPLETE",
        "scientific_role": "MANDATORY_PREFROZEN_SECONDARY_SENSITIVITY",
        "primary_full_cohort_result_remains_authoritative": True,
        "may_override_primary_full_cohort_result": False,
        "authority": {
            "prebootstrap_gate_sha256": preflight_sha,
            "bootstrap_replicates_npz_sha256": reps_sha,
            "stage915b_primary_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
            "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
            "recovery1_amendment_sha256": EXPECTED[str(REC1)],
            "recovery2_amendment_sha256": EXPECTED[str(REC2)],
        },
        "population": {
            "primary_full_cohort_scene_count": FORMAL_N,
            "excluded_documented_reread_scene_count": EXPECTED_EXCLUDED_N,
            "secondary_reduced_scene_count": REDUCED_N,
        },
        "statistics": {
            "independent_unit": "FORMAL_SCENARIO_SEGMENT",
            "paired_scenario_cluster_bootstrap": True,
            "replicates": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "RNG_engine": "numpy.random.PCG64",
            "CI_method": "equal_tailed_percentile_bootstrap",
            "CI_level": 0.95,
            "numpy_quantile_method": "linear",
            "same_global_reduced_scenario_draw_used_for_all_evaluable_endpoints_per_replicate": True,
            "new_RNG_draws_executed": True,
            "new_RNG_draws_authorized_before_primary_inference": True,
            "primary_resample_indices_reused": False,
            "scipy_used": False,
        },
        "rare_event_support": {
            "critical_label_valid_denominator": int(critical_valid),
            "observed_Ccrit_GT_events": int(critical_events),
            "prevalence_point": 0.0,
            "Wilson_95pct": [wil_lo, wil_hi],
            "zero_observed_events_interpreted_as_zero_risk": False,
        },
        "H1": {
            "confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
            "confirmatory_reactivation": False,
            "reporting_role": "SECONDARY_NONCONFIRMATORY_NUMERICAL_SENSITIVITY_ONLY",
            "numerical_sensitivity_status": h1_numerical_status,
            "point_estimates": h1_pt,
            "primary_delta_CI95": list(h1_ci),
            "stratum_delta_CI95": h1_stratum_summary,
        },
        "H2": {
            "secondary_sensitivity_status": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
            "confirmatory_PASS_allowed": False,
            "critical_event_numerator": 0,
            "critical_event_denominator": int(critical_valid),
            "metric_substitution": False,
        },
        "H3": {
            "secondary_sensitivity_status": "PARTIALLY_EVALUABLE_RESOURCE_COMPONENT_ONLY",
            "critical_reliability_component": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
            "full_confirmatory_PASS_allowed": False,
            "resource_component_by_codebook": h3_summary,
            "new_noninferiority_margin_introduced": False,
        },
        "H4": {
            "secondary_sensitivity_status": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
            "confirmatory_PASS_allowed": False,
            "matched_comparator_inference_executed": False,
            "reason": "zero critical-event support persists after exact 22-scene exclusion",
            "FORMAL_based_comparator_selection": False,
        },
        "H5": {
            "secondary_sensitivity_status": "TECHNICAL_RECOVERY_EXCLUSION_ALL_SIX_REPORTED",
            "comparison_count": 6,
            "comparisons": h5_summary,
            "FAST_excluded": False,
            "profile_or_baseline_dropped": False,
            "all_six_reduction_supported": bool(
                all(v["secondary_sensitivity_decision"] == "REDUCTION_SUPPORTED"
                    for v in h5_summary.values())
            ),
        },
        "H6": {
            "secondary_sensitivity_core_gate_status": (
                "PASS_FROZEN_STAGE6_CORE_GATES"
                if h6_core_pass
                else "FAIL_FROZEN_STAGE6_CORE_GATE"
            ),
            "paired_deltas": h6_summary,
            "frozen_core_gate_results": h6_core,
            "frozen_core_gate_PASS": h6_core_pass,
            "road_illumination_new_threshold_introduced": False,
            "glare_new_threshold_introduced": False,
        },
        "integrity": {
            "model_rerun": False,
            "policy_rerun": False,
            "solver_rerun": False,
            "retuning": False,
            "policy_changed": False,
            "baseline_changed": False,
            "threshold_changed": False,
            "C3_reselected": False,
            "primary_FORMAL_population_changed": False,
            "secondary_sensitivity_population_is_prefrozen_exact_exclusion": True,
            "H1_confirmatory_reactivation": False,
        },
        "next_block": "9.15H_APPEND_ONLY_PROVENANCE_WORDING_ADDENDUM",
    }

    atomic_json(SUMMARY, summary)
    summary_sha = sha256_path(SUMMARY)

    seal = {
        "schema": "stage9_block915g_v1_technical_recovery_sensitivity_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915G_V1_TECHNICAL_RECOVERY_EXCLUSION_SENSITIVITY",
        "role": "APPEND_ONLY_MANDATORY_SECONDARY_SENSITIVITY",
        "stage915f_primary_closure_seal_sha256": EXPECTED[str(P915F_SEAL)],
        "stage915b_primary_summary_sha256": EXPECTED[str(P915B_SUMMARY)],
        "prebootstrap_gate_sha256": preflight_sha,
        "bootstrap_replicates_sha256": reps_sha,
        "summary_sha256": summary_sha,
        "excluded_scene_count": EXPECTED_EXCLUDED_N,
        "secondary_reduced_scene_count": REDUCED_N,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "new_RNG_draws_executed": True,
        "new_RNG_draws_prefrozen_authorized": True,
        "primary_resample_indices_reused": False,
        "primary_full_cohort_remains_authoritative": True,
        "sensitivity_may_override_primary": False,
        "H1_confirmatory_status": "NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM",
        "H1_confirmatory_reactivation": False,
        "post_FORMAL_retuning": False,
        "model_policy_solver_rerun": False,
        "next_block": "9.15H_APPEND_ONLY_PROVENANCE_WORDING_ADDENDUM",
    }
    atomic_json(SEAL, seal)

    print("\n" + "=" * 124)
    print("STAGE 9.15G-v1 = COMPLETE")
    print("ROLE = MANDATORY_PREFROZEN_SECONDARY_SENSITIVITY")
    print("PRIMARY_FULL_COHORT_REMAINS_AUTHORITATIVE = TRUE")
    print("EXCLUDED_DOCUMENTED_REREAD_SCENES =", EXPECTED_EXCLUDED_N)
    print("SECONDARY_REDUCED_N =", REDUCED_N)
    print("NEW_RNG_DRAWS_EXECUTED = TRUE")
    print("NEW_RNG_DRAWS_PREFROZEN_AUTHORIZED = TRUE")
    print("PRIMARY_RESAMPLE_INDICES_REUSED = FALSE")
    print("H1_CONFIRMATORY_STATUS = NOT_TESTED_MISSING_PREFROZEN_UNCALIBRATED_ARM")
    print("H1_CONFIRMATORY_REACTIVATION = FALSE")
    print("H1_NUMERICAL_SENSITIVITY_STATUS =", h1_numerical_status)
    print("H2_SENSITIVITY_STATUS =", summary["H2"]["secondary_sensitivity_status"])
    print("H3_SENSITIVITY_STATUS =", summary["H3"]["secondary_sensitivity_status"])
    print("H4_SENSITIVITY_STATUS =", summary["H4"]["secondary_sensitivity_status"])
    print("H5_SENSITIVITY_STATUS =", summary["H5"]["secondary_sensitivity_status"])
    print("H6_SENSITIVITY_STATUS =", summary["H6"]["secondary_sensitivity_core_gate_status"])
    print("PREFLIGHT_SHA256 =", preflight_sha)
    print("BOOTSTRAP_REPLICATES_SHA256 =", reps_sha)
    print("SUMMARY_SHA256 =", summary_sha)
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15H_APPEND_ONLY_PROVENANCE_WORDING_ADDENDUM")
    print("=" * 124)


if __name__ == "__main__":
    main()

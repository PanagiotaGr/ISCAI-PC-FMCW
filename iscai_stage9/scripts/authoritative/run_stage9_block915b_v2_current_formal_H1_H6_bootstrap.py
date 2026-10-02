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
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260910
CI_Q = (0.025, 0.975)
CHUNK = 25

# -------------------------------------------------------------------------------------------------
# Frozen 9.13 final raw-outcome authority.
# -------------------------------------------------------------------------------------------------
B913 = S9 / "artifacts/block913_v1_3_FINAL_EVALUATOR_H1_H6_R3"
OUTCOME_DIR = B913 / "scenario_outcomes"
P913_MANIFEST = B913 / "stage9_block913_v1_3_FINAL_H1_H6_manifest.jsonl"
P913_SUMMARY = B913 / "stage9_block913_v1_3_FINAL_H1_H6_summary.json"
P913_SEAL = B913 / "stage9_block913_v1_3_FINAL_H1_H6_seal.json"

# Authoritative repaired 9.15A pre-inference contract.
B915A = S9 / "artifacts/block915a_v2R4_h1_status_censoring_contract_repair"
P915A_EVIDENCE = B915A / "stage9_block915a_v2R4_h1_schema_status_censoring_evidence.json"
P915A_EST = B915A / "stage9_block915a_v2R4_estimand_registry.json"
P915A_CONTRACT = B915A / "stage9_block915a_v2R4_statistics_contract.json"
P915A_SEAL = B915A / "stage9_block915a_v2R4_statistics_contract_seal.json"

EXPECTED = {
    str(P913_MANIFEST): "752600ddf126cabd87010d8277b7fe040d18b3e8d8f2490adcc2e9ca0ceb63c4",
    str(P913_SUMMARY): "292d515a3f52c82ad08460e1122a805d236a8aeaefa82271ff84664c2cddfe60",
    str(P913_SEAL): "fdbb3a4c9266336f6f150ed8b0c3677a08419e51488d018cad2494c72b428a64",
    str(P915A_EVIDENCE): "c96b9f969c5e6d1f1e760313d5c58d279746044fb3846cba63c67a1a7bc785a0",
    str(P915A_EST): "4ecc05f12b5c11e46a976b161fd7866d30bda76f287d4904d7cd52e95a71e37c",
    str(P915A_CONTRACT): "91362b13413565501d3d33905d78bc12f14bb9c8b3739783e899cbc6088c7b03",
    str(P915A_SEAL): "e0606b3aeb016e573de995032e5c251583978d96426f9177408d0d290b9d7a37",
}

OUT = S9 / "artifacts/block915b_v2_current_formal_H1_H6_bootstrap"
PREFLIGHT = OUT / "stage9_block915b_v2_prebootstrap_schema_gate.json"
REPLICATES = OUT / "stage9_block915b_v2_bootstrap_replicates.npz"
SUMMARY = OUT / "stage9_block915b_v2_H1_H6_statistics_summary.json"
SEAL = OUT / "stage9_block915b_v2_statistics_seal.json"

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
        # numpy may append .npz if suffix handling changes; normalize.
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


def finite_float_or_none(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


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


def main() -> None:
    print("=" * 124)
    print("STAGE 9.15B-v2 — CURRENT FORMAL H1-H6 STATISTICS + 10k PAIRED SCENARIO-CLUSTER BOOTSTRAP")
    print("FINAL 9.13 OUTCOMES ONLY | NO MODEL/POLICY/SOLVER RERUN | NO RETUNING | NO SCIPY")
    print("=" * 124)

    if SEAL.exists():
        print("ALREADY_COMPLETE", SEAL)
        print("SEAL_SHA256 =", sha256_path(SEAL))
        return

    hash_gate()

    aseal = load_json(P915A_SEAL)
    req(
        aseal.get("status") == "FROZEN_COMPLETE_STAGE9_BLOCK915A_V2R4_PRE_INFERENCE_CONTRACT",
        "9.15A-v2R4 seal status drift",
    )
    req(aseal.get("bootstrap_executed") is False, "9.15A already executed bootstrap")
    req(aseal.get("RNG_used") is False, "9.15A already used RNG")
    req(aseal.get("H1_horizons_s") == [0.1, 0.3, 0.5, 1.0], "9.15A H1 horizons drift")

    print("\n===== A. PRE-BOOTSTRAP FULL H1-H6 SCHEMA / SUPPORT GATE =====")

    files = sorted(OUTCOME_DIR.glob("*.json"))
    req(len(files) == FORMAL_N, f"FORMAL outcome cardinality drift: {len(files)}")

    # Additive / paired carriers by FORMAL scenario.
    h1_den = np.zeros((FORMAL_N, len(H1_STRATA)), dtype=np.uint8)
    h1_raw = np.zeros_like(h1_den)
    h1_cal = np.zeros_like(h1_den)

    h3_k_diff = np.full((FORMAL_N, len(CODEBOOKS)), np.nan, dtype=np.float64)
    h3_overhead_diff = np.full_like(h3_k_diff, np.nan)

    h5_n = np.zeros((FORMAL_N, len(H5_COMPARISONS)), dtype=np.int32)
    h5_base = np.zeros_like(h5_n)
    h5_recovery = np.zeros_like(h5_n)

    h6_delta = np.full((FORMAL_N, len(H6_COMPARISONS)), np.nan, dtype=np.float64)

    status_counts = Counter()
    critical_valid = 0
    critical_events = 0
    communication_valid_scenes = 0
    h1_record_count = 0
    h5_total_n = np.zeros(len(H5_COMPARISONS), dtype=np.int64)
    h6_finite_support = np.zeros(len(H6_COMPARISONS), dtype=np.int64)

    required_h1_keys = {
        "arm", "codebook_size", "horizon_s", "requested_q", "containment_hit",
        "selected_k", "physical_probe_count",
    }
    required_comm_keys = {
        "policy", "codebook_size", "K", "probing_overhead_fraction",
        "beam_outage", "critical_label_valid", "Y_crit_1s",
    }

    for i, path in enumerate(files):
        j = load_json(path)
        req(j.get("status") == "COMPLETE_FINAL_EVALUATOR_H1_H6",
            f"outer outcome status drift: {path.name}")
        req(int(j.get("formal_ordinal", -1)) == i, f"formal ordinal drift: {path.name}")
        req(j.get("C1_MC_recomputed") is False, f"C1 recomputed: {path.name}")
        req(j.get("C2_solver_recomputed") is False, f"C2 recomputed: {path.name}")
        req(j.get("model_forward_executed") is False, f"model forward drift: {path.name}")
        req(j.get("policy_retuned") is False, f"policy retuned: {path.name}")
        req(j.get("thresholds_retuned") is False, f"threshold retuned: {path.name}")

        primary = j.get("primary")
        req(isinstance(primary, dict), f"missing primary object: {path.name}")
        pstatus = str(primary.get("status"))
        status_counts[pstatus] += 1

        # H1 carrier.
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
            h1_den[i, s] = 1
            if bool(r["containment_hit"]):
                if arm == "RAW":
                    h1_raw[i, s] = 1
                else:
                    h1_cal[i, s] = 1
            h1_record_count += 1

        # For any available H1 stratum, both arms must be present.
        for cb, h in H1_STRATA:
            has_raw = ("RAW", cb, h) in seen_h1
            has_cal = ("CALIBRATED", cb, h) in seen_h1
            req(has_raw == has_cal, f"H1 arm-specific missingness: {path.name} cb={cb} h={h}")

        # Criticality / communication carrier.
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
                h3_k_diff[i, cb_i] = float(rp["K"]) - float(rq["K"])
                h3_overhead_diff[i, cb_i] = (
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

        # H5 exact six comparisons.
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
                h5_n[i, c] = n
                h5_base[i, c] = bv
                h5_recovery[i, c] = rv

        # H6 paired scene-level metrics.
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
                    h6_delta[i, c] = a - b

        if (i + 1) % 5000 == 0:
            print(
                f"PREFLIGHT {i+1}/{FORMAL_N} "
                f"critical_valid={critical_valid} critical_events={critical_events}",
                flush=True,
            )

    req(sum(status_counts.values()) == FORMAL_N, "primary status count total drift")
    req(status_counts["NOT_EVALUATED_NO_ELIGIBLE_RECEIVER"] == 970, "no-receiver count drift")
    req(critical_valid == 24917, f"critical-valid denominator drift: {critical_valid} != 24917")
    req(communication_valid_scenes == 24917, "communication-valid scene count drift")

    # This is an already-known frozen point-estimate fact, not a newly selected endpoint.
    req(
        critical_events == 0,
        f"current final 9.13 critical support changed: observed {critical_events}, expected frozen 0",
    )

    h1_den_tot = h1_den.sum(axis=0, dtype=np.int64)
    h1_raw_tot = h1_raw.sum(axis=0, dtype=np.int64)
    h1_cal_tot = h1_cal.sum(axis=0, dtype=np.int64)
    req(np.all(h1_den_tot > 0), "H1 has zero-support stratum")

    h5_total_n = h5_n.sum(axis=0, dtype=np.int64)
    req(np.all(h5_total_n > 0), "H5 comparison has zero aggregate support")
    h6_finite_support = np.sum(np.isfinite(h6_delta), axis=0, dtype=np.int64)

    # H6 core gates require finite paired support.
    for metric in (
        "vehicle_shadow_zone_violation",
        "over_masking_area",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    ):
        c = H6_INDEX[("reactive", metric)]
        req(h6_finite_support[c] > 0, f"H6 core metric has zero paired support: {metric}")

    wil_lo, wil_hi = wilson_interval(critical_events, critical_valid)

    preflight = {
        "schema": "stage9_block915b_v2_prebootstrap_schema_gate_v1",
        "status": "PASS_FULL_H1_H6_SCHEMA_AND_SUPPORT_GATE",
        "authority": {
            "stage913_manifest_sha256": EXPECTED[str(P913_MANIFEST)],
            "stage913_summary_sha256": EXPECTED[str(P913_SUMMARY)],
            "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage915a_v2R4_contract_sha256": EXPECTED[str(P915A_CONTRACT)],
            "stage915a_v2R4_seal_sha256": EXPECTED[str(P915A_SEAL)],
        },
        "FORMAL_scene_count": FORMAL_N,
        "primary_status_counts": dict(sorted(status_counts.items())),
        "H1_total_materialized_records": int(h1_record_count),
        "H1_support_by_stratum": {
            f"cb{cb}_h{h:.1f}": int(h1_den_tot[k])
            for k, (cb, h) in enumerate(H1_STRATA)
        },
        "critical_label_valid_denominator": int(critical_valid),
        "observed_Ccrit_GT_events": int(critical_events),
        "Ccrit_prevalence": 0.0,
        "Ccrit_Wilson_95pct": [wil_lo, wil_hi],
        "H2_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H3_reliability_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H4_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H4_comparator_resolution": (
            "NOT_REQUIRED_FOR_CURRENT_INFERENCE_BECAUSE_FROZEN_CRITICAL_CONDITIONED_DENOMINATOR_IS_ZERO"
        ),
        "H3_resource_paired_scene_count": int(np.sum(np.all(np.isfinite(h3_k_diff), axis=1))),
        "H5_support_n": {
            f"{p}__{b}": int(h5_total_n[k])
            for k, (p, b) in enumerate(H5_COMPARISONS)
        },
        "H6_finite_paired_scene_support": {
            f"recovery_minus_{b}__{m}": int(h6_finite_support[k])
            for k, (b, m) in enumerate(H6_COMPARISONS)
        },
        "bootstrap_not_started_at_gate_write": True,
        "RNG_not_created_at_gate_write": True,
        "retuning": False,
        "metric_substitution": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(PREFLIGHT, preflight)
    preflight_sha = sha256_path(PREFLIGHT)

    print("PREFLIGHT_H1_H6 = PASS")
    print("CRITICAL_EVENTS =", critical_events, "/", critical_valid)
    print("H2 = NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS")
    print("H3_RELIABILITY = NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS")
    print("H4 = NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS")
    print("PREFLIGHT_SHA256 =", preflight_sha)

    # Point estimates before RNG.
    h1_pt = h1_point(h1_raw_tot, h1_cal_tot, h1_den_tot)

    h3_pt_k = np.nanmean(h3_k_diff, axis=0)
    h3_pt_o = np.nanmean(h3_overhead_diff, axis=0)

    h5_base_tot = h5_base.sum(axis=0, dtype=np.int64)
    h5_rec_tot = h5_recovery.sum(axis=0, dtype=np.int64)
    h5_pt = (h5_rec_tot / h5_total_n) - (h5_base_tot / h5_total_n)

    h6_pt = np.nanmean(h6_delta, axis=0)

    print("\n===== B. 10,000-REPLICATE PAIRED SCENARIO-CLUSTER BOOTSTRAP =====")
    print("RNG_ENGINE = numpy.random.PCG64")
    print("BOOTSTRAP_SEED =", BOOTSTRAP_SEED)

    # The RNG is intentionally instantiated only after PREFLIGHT is immutably written.
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
        idx = rng.integers(0, FORMAL_N, size=(bsz, FORMAL_N), dtype=np.int32)

        # H1: horizon-specific denominators are preserved inside each replicate.
        den = h1_den[idx].sum(axis=1, dtype=np.int64)
        raw = h1_raw[idx].sum(axis=1, dtype=np.int64)
        cal = h1_cal[idx].sum(axis=1, dtype=np.int64)
        req(np.all(den > 0), "H1 bootstrap replicate produced zero-support stratum")
        er = np.abs(raw / den - 0.95)
        ec = np.abs(cal / den - 0.95)
        ds = ec - er
        r_h1[rep0:rep0+bsz] = np.mean(ds, axis=1)
        r_h1_strata[rep0:rep0+bsz, :] = ds

        # H3 resource: exact paired scene differences, valid-label scenes only.
        # NaN scenes (actor-1s-invalid/no-receiver) are excluded identically for both arms.
        ksel = h3_k_diff[idx]
        osel = h3_overhead_diff[idx]
        with np.errstate(invalid="ignore"):
            r_h3_k[rep0:rep0+bsz, :] = np.nanmean(ksel, axis=1)
            r_h3_o[rep0:rep0+bsz, :] = np.nanmean(osel, axis=1)

        # H5: reaggregate raw paired common-support counts within each sampled scenario cluster.
        n = h5_n[idx].sum(axis=1, dtype=np.int64)
        bv = h5_base[idx].sum(axis=1, dtype=np.int64)
        rv = h5_recovery[idx].sum(axis=1, dtype=np.int64)
        req(np.all(n > 0), "H5 bootstrap replicate produced zero denominator")
        r_h5[rep0:rep0+bsz, :] = (rv / n) - (bv / n)

        # H6: paired scene-level finite deltas; metric-specific finite support.
        d = h6_delta[idx]
        with np.errstate(invalid="ignore"):
            x = np.nanmean(d, axis=1)
        # Every H6 comparison reported here must have finite replicate support.
        req(np.all(np.isfinite(x)), "H6 bootstrap replicate produced zero paired finite support")
        r_h6[rep0:rep0+bsz, :] = x

        rep0 += bsz
        if rep0 % 500 == 0 or rep0 == BOOTSTRAP_REPS:
            print(f"BOOTSTRAP {rep0}/{BOOTSTRAP_REPS}", flush=True)

    # Immutable replicate carrier.
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

    # -------------------------------------------------------------------------------------------------
    # Summaries / frozen decisions.
    # -------------------------------------------------------------------------------------------------
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
            "resource_reduction_supported": bool(khi < 0.0 and ohi < 0.0),
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
            "decision": decision,
        }

    h6_summary = {}
    for k, (baseline, metric) in enumerate(H6_COMPARISONS):
        lo, hi = quantile_ci(r_h6[:, k])
        h6_summary[f"recovery_minus_{baseline}__{metric}"] = {
            "paired_finite_scene_count": int(h6_finite_support[k]),
            "point_delta": float(h6_pt[k]),
            "CI95": [lo, hi],
        }

    # Frozen Stage6 core H6 gates are point-estimate gates, not newly invented CI thresholds.
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

    summary = {
        "schema": "stage9_block915b_v2_H1_H6_statistics_summary_v1",
        "status": "CURRENT_FORMAL_H1_H6_STATISTICS_COMPLETE",
        "authority": {
            "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
            "stage915a_v2R4_seal_sha256": EXPECTED[str(P915A_SEAL)],
            "prebootstrap_schema_gate_sha256": preflight_sha,
            "bootstrap_replicates_npz_sha256": reps_sha,
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
            "same_global_scenario_draw_used_for_all_evaluable_endpoints_per_replicate": True,
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
            "status": (
                "SUPPORTED_CALIBRATION_REDUCES_MACRO_ABS_COVERAGE_ERROR"
                if h1_ci[1] < 0.0
                else "NOT_SUPPORTED_BY_95PCT_BOOTSTRAP_CI"
            ),
            "point_estimates": h1_pt,
            "primary_delta_CI95": list(h1_ci),
            "stratum_delta_CI95": h1_stratum_summary,
        },
        "H2": {
            "status": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
            "confirmatory_PASS_allowed": False,
            "critical_event_numerator": 0,
            "critical_event_denominator": int(critical_valid),
            "metric_substitution": False,
        },
        "H3": {
            "status": "PARTIALLY_EVALUABLE_RESOURCE_COMPONENT_ONLY",
            "critical_reliability_component": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
            "full_confirmatory_PASS_allowed": False,
            "resource_component_by_codebook": h3_summary,
            "new_noninferiority_margin_introduced": False,
        },
        "H4": {
            "status": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
            "confirmatory_PASS_allowed": False,
            "matched_comparator_inference_executed": False,
            "reason": "zero frozen Ccrit_GT support makes critical-conditioned H4 endpoint undefined",
            "FORMAL_based_comparator_selection": False,
        },
        "H5": {
            "status": "SECONDARY_FORMAL_SENSITIVITY_ALL_SIX_REPORTED",
            "comparison_count": 6,
            "comparisons": h5_summary,
            "FAST_excluded": False,
            "profile_or_baseline_dropped": False,
            "all_six_reduction_supported": bool(
                all(v["decision"] == "REDUCTION_SUPPORTED" for v in h5_summary.values())
            ),
        },
        "H6": {
            "status": (
                "PASS_FROZEN_STAGE6_CORE_GATES_WITH_BOOTSTRAP_CONTEXT"
                if h6_core_pass
                else "FAIL_FROZEN_STAGE6_CORE_GATE"
            ),
            "paired_deltas": h6_summary,
            "frozen_core_gate_results": h6_core,
            "frozen_core_gate_PASS": h6_core_pass,
            "road_illumination_new_threshold_introduced": False,
            "glare_new_threshold_introduced": False,
        },
        "retuning": False,
        "policy_changed": False,
        "baseline_changed": False,
        "threshold_changed": False,
        "C3_reselected": False,
        "FORMAL_population_changed": False,
        "next_block": "9.15C_ROBUSTNESS",
    }
    atomic_json(SUMMARY, summary)
    summary_sha = sha256_path(SUMMARY)

    seal = {
        "schema": "stage9_block915b_v2_statistics_seal_v1",
        "status": "FROZEN_COMPLETE_STAGE9_BLOCK915B_V2_CURRENT_FORMAL_H1_H6_STATISTICS",
        "stage913_seal_sha256": EXPECTED[str(P913_SEAL)],
        "stage915a_v2R4_seal_sha256": EXPECTED[str(P915A_SEAL)],
        "prebootstrap_schema_gate_sha256": preflight_sha,
        "bootstrap_replicates_sha256": reps_sha,
        "summary_sha256": summary_sha,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "H1_inference_complete": True,
        "H2_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H3_resource_inference_complete": True,
        "H3_reliability_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H4_disposition": "NOT_EVALUABLE_ZERO_OBSERVED_CRITICAL_EVENTS",
        "H5_all_six_inference_complete": True,
        "H6_inference_complete": True,
        "zero_support_metric_substitution": False,
        "post_FORMAL_retuning": False,
        "next_block": "9.15C_ROBUSTNESS",
    }
    atomic_json(SEAL, seal)

    print("\n" + "=" * 124)
    print("STAGE 9.15B-v2 = COMPLETE")
    print("H1_STATUS =", summary["H1"]["status"])
    print("H2_STATUS =", summary["H2"]["status"])
    print("H3_STATUS =", summary["H3"]["status"])
    print("H4_STATUS =", summary["H4"]["status"])
    print("H5_STATUS =", summary["H5"]["status"])
    print("H6_STATUS =", summary["H6"]["status"])
    print("PREFLIGHT_SHA256 =", preflight_sha)
    print("BOOTSTRAP_REPLICATES_SHA256 =", reps_sha)
    print("SUMMARY_SHA256 =", summary_sha)
    print("SEAL_SHA256 =", sha256_path(SEAL))
    print("NEXT_BLOCK = 9.15C_ROBUSTNESS")
    print("=" * 124)


if __name__ == "__main__":
    main()

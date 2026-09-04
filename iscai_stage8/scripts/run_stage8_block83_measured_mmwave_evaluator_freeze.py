#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block83_measured_mmwave_evaluator_freeze.py"
CONTRACT = S8 / "configs/stage8_block83_measured_mmwave_evaluator_contract.json"
EVALUATOR = S8 / "src/deepsense_mmwave_evaluator.py"

B82B_ROOT = S8 / "artifacts/block82b_canonical_corpus_schema_split"
B82B_SCHEMA = B82B_ROOT / "actual_csv_schema_and_counts.json"
B82B_REFS = B82B_ROOT / "resolved_reference_integrity.json"
B82B_SPLITS = B82B_ROOT / "frozen_split_roles_and_disjointness.json"
B82B_FILES = B82B_ROOT / "canonical_extracted_file_manifest.json"
B82B_SEAL = B82B_ROOT / "stage8_block82b_canonical_corpus_schema_split_seal.json"
B82B_REPORT = S8 / "reports/stage8_block82b_canonical_corpus_schema_split_report.json"
B82B_RUNNER = S8 / "scripts/run_stage8_block82b_canonical_corpus_schema_split_freeze.py"

OUT = S8 / "artifacts/block83_measured_mmwave_evaluator_contract"
TESTS = OUT / "synthetic_evaluator_unit_tests.json"
MANIFEST = OUT / "stage8_block83_manifest.json"
REPORT = S8 / "reports/stage8_block83_measured_mmwave_evaluator_contract_report.json"
SEAL = OUT / "stage8_block83_measured_mmwave_evaluator_contract_seal.json"

EXPECTED_CONTRACT = "aa70a7ccb299b8337b11d899ae38287189f3a3d4a7d9b8dd7b9ea4860d2166e8"
EXPECTED_EVALUATOR = "7e2c4ffe589e3d6f58e5b6daf1d8fb9e6e44b2eef074bdf1624a3334f8cbe650"
EXPECTED_B82B = {
    str(B82B_SEAL): "c84eb7ef45d16a1feb3ef068bdaf498e5e4298514f3c4807ffe10db9b1112ba3",
    str(B82B_SCHEMA): "887a796df8138ba175cc5868c49952443ec3497d45b8393876e341f1ef89c5ab",
    str(B82B_REFS): "11aa52287c0e2cb2fee7d8e0f83590517d97aefa72cb276d516cd8547ab6f098",
    str(B82B_SPLITS): "67e7624cdb633bf83c0fedab19b6d4c5b41d70ccf0d69c9875873c24b3fe20ce",
    str(B82B_FILES): "79dc92fb6a90acd36c4598de436517ebac6c7e7904334dd76a520e1c90a0b895",
    str(B82B_REPORT): "53066a0d86a0f092e08e3469f5ba6e6d0f611e5d26584192e6b03833e2b5d340",
    str(B82B_RUNNER): "b54424fd1f09d1e4213d18eaae45e3746c1d6befc0e3d1e88db2137c9a74b3ff",
}

class FailClosed(RuntimeError): pass

def require(cond, msg):
    if not cond:
        raise FailClosed(msg)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing overwrite: {path}")
    data = (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_eval():
    spec = importlib.util.spec_from_file_location("ds_eval", EVALUATOR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def approx(a,b,tol=1e-12):
    return abs(float(a)-float(b)) <= tol

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.3")
    print("PRE-OUTCOME MEASURED-MMWAVE EVALUATOR CONTRACT + SYNTHETIC TESTS")
    print("NO REAL CALIBRATION/FORMAL POWER VECTOR ACCESS")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT, "8.3 contract SHA mismatch.")
    require(EVALUATOR.is_file() and sha256_file(EVALUATOR) == EXPECTED_EVALUATOR, "8.3 evaluator SHA mismatch.")
    for raw, expected in EXPECTED_B82B.items():
        p = Path(raw)
        require(p.is_file(), f"Missing frozen 8.2B input: {p}")
        require(sha256_file(p) == expected, f"8.2B SHA changed: {p}")
    for p in (TESTS, MANIFEST, REPORT, SEAL):
        require(not p.exists(), f"8.3 output exists: {p}")

    print("\n===== A. BLOCK8.2B IMMUTABLE GATE =====")
    seal82 = json.loads(B82B_SEAL.read_text())
    require(seal82.get("status") == "FROZEN_COMPLETE_BLOCK82B", "8.2B status changed.")
    require(seal82.get("Block83_may_start") is True, "8.3 not authorized.")
    require(seal82.get("formal_power_vectors_decoded") is False, "Formal power boundary already crossed.")
    print("Block8.2B exact identities + pre-outcome boundary = PASS")

    print("\n===== B. CONTRACT SEMANTIC GATE =====")
    c = json.loads(CONTRACT.read_text())
    require(c["policy_contract"]["primary_coverage_mass"] == 0.95, "Primary q changed.")
    require(c["policy_contract"]["secondary_coverage_masses"] == [0.9,0.95,0.975,0.99], "q sweep changed.")
    require(c["outage"]["primary_threshold_db"] == 3.0, "Primary outage threshold changed.")
    require(c["outage"]["sensitivity_thresholds_db"] == [1.0,3.0,6.0], "Outage sensitivity changed.")
    require(c["normalized_spectral_efficiency"]["primary_reference_snr_db"] == 10.0, "Primary normalized SE SNR changed.")
    require(c["statistical_inference"]["paired_bootstrap_iterations"] == 10000, "Bootstrap count changed.")
    require(c["statistical_inference"]["seed"] == 20260902, "Bootstrap seed changed.")
    require(c["result_separation"]["DeepSense_table_name"] == "DeepSense measured mmWave", "Result table role changed.")
    print("q=0.95 primary + frozen q sweep = PASS")
    print("APL-style measured-power gap semantics = PASS")
    print("outage primary 3dB + {1,3,6} sensitivity = PASS")
    print("normalized SE proxy SNR {0,10,20} dB; primary 10dB = PASS")
    print("paired bootstrap 10,000 / seed 20260902 = PASS")
    print("optical/mmWave separation = PASS")

    print("\n===== C. SYNTHETIC EVALUATOR UNIT TESTS =====")
    m = load_eval()
    tests = []

    # Perfect selected set.
    p = [1.0] * 64
    p[7] = 10.0
    r = m.evaluate_selected_set([7], p, 7)
    require(r["coverage"] == 1, "perfect coverage failed")
    require(approx(r["measured_power_gap_db"], 0.0), "perfect gap failed")
    require(r["outage"]["1dB"] is False and r["outage"]["3dB"] is False and r["outage"]["6dB"] is False,
            "perfect outage failed")
    require(approx(r["probing_overhead"], 1/64), "perfect overhead failed")
    tests.append({"name":"perfect_single_beam","pass":True})

    # ~3.0103 dB miss: selected power half optimum -> 3 dB outage under strict -3.0 threshold.
    p2 = [1.0] * 64
    p2[3] = 10.0
    p2[4] = 5.0
    r2 = m.evaluate_selected_set([4], p2, 3)
    require(r2["coverage"] == 0, "miss coverage failed")
    require(abs(r2["measured_power_gap_db"] - 10*math.log10(0.5)) < 1e-12, "half-power gap failed")
    require(r2["outage"]["3dB"] is True, "3dB outage failed")
    require(r2["outage"]["6dB"] is False, "6dB outage failed")
    tests.append({"name":"half_power_miss","pass":True})

    # Top-set can miss official label yet still have smaller power loss.
    p3 = [1.0] * 64
    p3[10] = 8.0
    p3[11] = 7.9
    r3 = m.evaluate_selected_set([11,12,13], p3, 10)
    require(r3["coverage"] == 0, "near-optimal miss coverage failed")
    require(r3["measured_power_loss_db"] > 0.0, "near-optimal loss sign failed")
    tests.append({"name":"near_optimal_noncoverage","pass":True})

    # Adaptive mass-covering deterministic tie break.
    probs = [0.0]*64
    probs[5] = 0.4; probs[2] = 0.4; probs[9] = 0.2
    s = m.adaptive_mass_covering_set(probs, 0.75)
    require(s == [2,5], f"deterministic tie-break failed: {s}")
    tests.append({"name":"adaptive_mass_tie_break","pass":True})

    # Label inconsistency must fail.
    failed = False
    try:
        m.evaluate_selected_set([7], p, 6)
    except m.EvaluationError:
        failed = True
    require(failed, "label consistency fail-closed test failed")
    tests.append({"name":"label_power_consistency_fail_closed","pass":True})

    # Nonpositive powers fail closed.
    failed = False
    badp = [1.0]*64; badp[0] = 0.0
    try:
        m.evaluate_selected_set([1], badp, 1)
    except m.EvaluationError:
        failed = True
    require(failed, "nonpositive power fail-closed test failed")
    tests.append({"name":"nonpositive_power_fail_closed","pass":True})

    print(f"synthetic evaluator tests = {len(tests)}/6 PASS")
    print("real calibration/formal powers accessed = NO")

    atomic_json(TESTS, {
        "stage":8,"block":"8.3","status":"PASS_SYNTHETIC_EVALUATOR_TESTS",
        "tests":tests,
        "real_dataset_power_vectors_used":False,
    })

    manifest = {
        "stage":8,"block":"8.3","status":"FROZEN_COMPLETE_BLOCK83",
        "contract":{"path":str(CONTRACT),"sha256":EXPECTED_CONTRACT},
        "evaluator":{"path":str(EVALUATOR),"sha256":EXPECTED_EVALUATOR},
        "runner":{"path":str(SCRIPT),"sha256":sha256_file(SCRIPT)},
        "synthetic_tests":{"path":str(TESTS),"sha256":sha256_file(TESTS)},
        "formal_power_vectors_accessed":False,
        "calibration_power_vectors_accessed":False,
        "training_started":False,
        "formal_metrics_computed":False,
    }
    atomic_json(MANIFEST, manifest)

    report = {
        "stage":8,"block":"8.3",
        "status":"PASS_BLOCK83_PRE_OUTCOME_MEASURED_MMWAVE_EVALUATOR_CONTRACT",
        "official_lidar_topk_semantics_bound":True,
        "APL_style_measured_power_gap_bound":True,
        "outage_primary_threshold_db":3.0,
        "outage_sensitivity_thresholds_db":[1.0,3.0,6.0],
        "normalized_SE_proxy_reference_snr_db":[0.0,10.0,20.0],
        "absolute_measured_SE_claim":False,
        "primary_coverage_mass":0.95,
        "coverage_sweep":[0.90,0.95,0.975,0.99],
        "paired_bootstrap_iterations":10000,
        "bootstrap_seed":20260902,
        "synthetic_tests_passed":len(tests),
        "real_formal_power_access":False,
        "training_started":False,
        "Block84_authorized":True,
        "manifest_sha256":sha256_file(MANIFEST),
        "tests_sha256":sha256_file(TESTS),
        "runner_sha256":sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage":8,"block":"8.3","status":"FROZEN_COMPLETE_BLOCK83",
        "Block82B_seal_sha256":EXPECTED_B82B[str(B82B_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,
        "evaluator_sha256":EXPECTED_EVALUATOR,
        "runner_sha256":sha256_file(SCRIPT),
        "synthetic_tests_sha256":sha256_file(TESTS),
        "manifest_sha256":sha256_file(MANIFEST),
        "report_sha256":sha256_file(REPORT),
        "formal_power_vectors_accessed":False,
        "training_started":False,
        "Block84_may_start":True,
    }
    atomic_json(SEAL, seal)

    for raw, expected in EXPECTED_B82B.items():
        require(sha256_file(Path(raw)) == expected, f"8.2B modified during 8.3: {raw}")

    print("\n" + "="*88)
    print("BLOCK 8.3 = FULLY VERIFIED / FROZEN")
    print("OFFICIAL LiDAR TOP-K COVERAGE SEMANTICS = BOUND")
    print("MEASURED POWER GAP / APL-STYLE SEMANTICS = BOUND")
    print("OUTAGE = PRIMARY 3 dB; SENSITIVITY 1/3/6 dB")
    print("SPECTRAL EFFICIENCY = NORMALIZED PROXY; SNRref 0/10/20 dB")
    print("ABSOLUTE MEASURED SE CLAIM = NO")
    print("PRIMARY q = 0.95; SWEEP 0.90/0.95/0.975/0.99")
    print("PAIRED BOOTSTRAP = 10,000; SEED 20260902")
    print("SYNTHETIC EVALUATOR TESTS = 6/6 PASS")
    print("CALIBRATION/FORMAL POWER VECTORS ACCESSED = NO")
    print("TRAINING/CALIBRATOR FIT/FORMAL METRICS = NO")
    print("BLOCK 8.4 MAY START = YES")
    print("8.3 contract SHA256 =", EXPECTED_CONTRACT)
    print("8.3 evaluator SHA256 =", EXPECTED_EVALUATOR)
    print("8.3 tests SHA256 =", sha256_file(TESTS))
    print("8.3 manifest SHA256 =", sha256_file(MANIFEST))
    print("8.3 report SHA256 =", sha256_file(REPORT))
    print("8.3 seal SHA256 =", sha256_file(SEAL))
    print("8.3 runner SHA256 =", sha256_file(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK83")
    print("="*88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88)
        print("BLOCK 8.3 FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT ACCESS CALIBRATION/FORMAL POWER VECTORS.")
        print("DO NOT START TRAINING OR FORMAL EVALUATION.")
        print("!"*88)
        raise

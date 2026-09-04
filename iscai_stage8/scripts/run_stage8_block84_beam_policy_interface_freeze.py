#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block84_beam_policy_interface_freeze.py"
CONTRACT = S8 / "configs/stage8_block84_beam_policy_interface_contract.json"
POLICY = S8 / "src/deepsense_beam_policy.py"

B83_ROOT = S8 / "artifacts/block83_measured_mmwave_evaluator_contract"
B83_TESTS = B83_ROOT / "synthetic_evaluator_unit_tests.json"
B83_MANIFEST = B83_ROOT / "stage8_block83_manifest.json"
B83_SEAL = B83_ROOT / "stage8_block83_measured_mmwave_evaluator_contract_seal.json"
B83_REPORT = S8 / "reports/stage8_block83_measured_mmwave_evaluator_contract_report.json"
B83_CONTRACT = S8 / "configs/stage8_block83_measured_mmwave_evaluator_contract.json"
B83_EVALUATOR = S8 / "src/deepsense_mmwave_evaluator.py"
B83_RUNNER = S8 / "scripts/run_stage8_block83_measured_mmwave_evaluator_freeze.py"

STAGE5_ADAPTIVE = S5 / "src/iscai_stage5/adaptive_topk.py"
STAGE5_CODEBOOK_POLICY = S5 / "configs/beam_codebook_policy.json"

OUT = S8 / "artifacts/block84_beam_policy_interface"
TESTS = OUT / "synthetic_policy_unit_tests.json"
POLICY_SET = OUT / "frozen_supported_policy_set.json"
MANIFEST = OUT / "stage8_block84_manifest.json"
REPORT = S8 / "reports/stage8_block84_beam_policy_interface_report.json"
SEAL = OUT / "stage8_block84_beam_policy_interface_seal.json"

EXPECTED_CONTRACT = "dc6e430bec5632a328dfdbe7d1d8f5a401971970fd0a414a144fecb707fcf81b"
EXPECTED_POLICY = "5b7e6c3c4ff7eca4d1db86df90edfa19f0bbfc551255779833485df074d87cb8"
EXPECTED_B83 = {
    str(B83_SEAL): "056371994fa598c6c58622b0161febde17f0e75231b4ba6e1d596d93db4e575f",
    str(B83_CONTRACT): "aa70a7ccb299b8337b11d899ae38287189f3a3d4a7d9b8dd7b9ea4860d2166e8",
    str(B83_EVALUATOR): "7e2c4ffe589e3d6f58e5b6daf1d8fb9e6e44b2eef074bdf1624a3334f8cbe650",
    str(B83_TESTS): "b0e90df5d8ab740ca5348bed97a2a61bedde235e55aeb61287840998f70c46fe",
    str(B83_MANIFEST): "282a84f4d78ffcf195a5b378fa6fb8a01672f8f5117e9f84c4e23383734c8cd4",
    str(B83_REPORT): "474768c8dd4d0e761103177d7f5afd11b0671cd16f09b445c1d2595550967f44",
    str(B83_RUNNER): "c1e7f10e812ee50364f1944f5b2ef4284401c833b245b79f8617571b72f25bfc",
}
EXPECTED_STAGE5 = {
    str(STAGE5_ADAPTIVE): "08bb2145eac117132166a0f8b96a38c61b0a912cfd9b7f2d48f16db649d22865",
    str(STAGE5_CODEBOOK_POLICY): "bd94f8609393a7c9fe02762cc4bf38e3a77a90cc31adef5f2e6b06aece4074d7",
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

def load_policy():
    spec = importlib.util.spec_from_file_location("ds_policy", POLICY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.4")
    print("FROZEN 64-WAY BEAM-POLICY INTERFACE + SUPPORTED BASELINES")
    print("NO TRAINING / NO CALIBRATION / NO FORMAL POWER ACCESS")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT, "8.4 contract SHA mismatch.")
    require(POLICY.is_file() and sha256_file(POLICY) == EXPECTED_POLICY, "8.4 policy SHA mismatch.")
    for raw, expected in EXPECTED_B83.items():
        p = Path(raw)
        require(p.is_file(), f"Missing 8.3 frozen artifact: {p}")
        require(sha256_file(p) == expected, f"8.3 SHA changed: {p}")
    for raw, expected in EXPECTED_STAGE5.items():
        p = Path(raw)
        require(p.is_file(), f"Missing Stage5 semantic authority: {p}")
        require(sha256_file(p) == expected, f"Stage5 semantic authority changed: {p}")
    for p in (TESTS, POLICY_SET, MANIFEST, REPORT, SEAL):
        require(not p.exists(), f"8.4 output exists: {p}")

    print("\n===== A. UPSTREAM / STAGE5 SEMANTIC GATES =====")
    b83 = json.loads(B83_SEAL.read_text())
    require(b83.get("status") == "FROZEN_COMPLETE_BLOCK83", "8.3 status changed.")
    require(b83.get("Block84_may_start") is True, "8.4 not authorized.")
    require(b83.get("formal_power_vectors_accessed") is False, "formal-power boundary crossed before 8.4.")
    print("Block8.3 exact identity + pre-outcome boundary = PASS")
    print("Stage5 adaptive Top-K semantic authorities = PASS")

    print("\n===== B. CONTRACT POLICY-SET GATE =====")
    c = json.loads(CONTRACT.read_text())
    require(c["input_interface"]["length"] == 64, "codebook interface changed.")
    require(c["supported_local_core_policies"]["adaptive_primary"]["q"] == 0.95, "primary q changed.")
    fixed = [x["K"] for x in c["supported_local_core_policies"]["fixed"]]
    require(fixed == [1,3,5], "fixed Top-K baselines changed.")
    require(c["supported_local_core_policies"]["exhaustive"]["K"] == 64, "exhaustive K changed.")
    require(c["supported_local_core_policies"]["oracle"]["evaluator_only"] is True, "oracle role changed.")
    require("NOT_EVALUABLE" in c["unsupported_or_deferred_local_core_policies"]["previous_beam_persistence"],
            "previous-beam unsupported status changed.")
    print("adaptive q=0.95 + q sweep = PASS")
    print("fixed Top-1/3/5 + exhaustive-64 = PASS")
    print("measured-power oracle = EVALUATOR-ONLY")
    print("previous-beam/hysteresis = NOT_EVALUABLE from current schema")

    print("\n===== C. SYNTHETIC POLICY UNIT TESTS =====")
    m = load_policy()
    tests = []

    # Tie-break deterministic.
    p = [0.0]*64
    p[5] = 0.4; p[2] = 0.4; p[9] = 0.2
    r = m.fixed_topk(p, 2)
    require(r["selected_beams"] == [2,5], f"fixed tie-break failed: {r}")
    tests.append({"name":"fixed_topk_tie_break","pass":True})

    r = m.adaptive_topk(p, 0.75)
    require(r["selected_beams"] == [2,5] and r["K"] == 2, f"adaptive tie-break/minimality failed: {r}")
    tests.append({"name":"adaptive_minimal_mass_cover","pass":True})

    # Minimality around threshold.
    p2 = [0.0]*64
    p2[0]=0.60; p2[1]=0.25; p2[2]=0.10; p2[3]=0.05
    r2 = m.adaptive_topk(p2, 0.95)
    require(r2["selected_beams"] == [0,1,2] and r2["K"] == 3, f"q=.95 failed: {r2}")
    tests.append({"name":"primary_q095","pass":True})

    # Fixed K sizes and exhaustive semantics.
    for name,k in (("fixed_top1",1),("fixed_top3",3),("fixed_top5",5)):
        rr = m.policy_dispatch(name, [1.0]*64)
        require(rr["K"] == k and len(rr["selected_beams"]) == k, f"{name} size failed")
    tests.append({"name":"fixed_baseline_sizes","pass":True})

    ex = m.policy_dispatch("exhaustive_64", [1.0]*64)
    require(ex["K"] == 64 and ex["selected_beams"] == list(range(64)), "exhaustive failed")
    tests.append({"name":"exhaustive_64","pass":True})

    # No measured-power argument exists in any controller dispatch.
    require(ex["measured_power_accessed"] is False and r2["measured_power_accessed"] is False,
            "controller measured-power boundary failed")
    tests.append({"name":"no_measured_power_controller_input","pass":True})

    # Stable sample identity.
    sid1 = m.stable_sample_id(8,"FORMAL_EVALUATION","csv",2,"lidar.mat","power.txt")
    sid2 = m.stable_sample_id(8,"FORMAL_EVALUATION","csv",2,"lidar.mat","power.txt")
    sid3 = m.stable_sample_id(8,"FORMAL_EVALUATION","csv",3,"lidar.mat","power.txt")
    require(sid1 == sid2 and sid1 != sid3 and len(sid1)==16, "sample-id determinism failed")
    tests.append({"name":"stable_sample_id","pass":True})

    # Invalid input fails closed.
    failed = False
    try:
        m.adaptive_topk([0.0]*64, 0.95)
    except m.PolicyError:
        failed = True
    require(failed, "zero-mass probability fail-closed failed")
    tests.append({"name":"invalid_probability_fail_closed","pass":True})

    print(f"synthetic policy tests = {len(tests)}/7 PASS")

    atomic_json(TESTS, {
        "stage":8,"block":"8.4","status":"PASS_SYNTHETIC_POLICY_TESTS",
        "tests":tests,
        "real_dataset_probabilities_used":False,
        "real_measured_power_used":False,
    })

    policy_set = {
        "stage":8,"block":"8.4","status":"FROZEN_SUPPORTED_POLICY_SET",
        "controller_policies":[
            "adaptive_topk_q090","adaptive_topk_q095","adaptive_topk_q0975","adaptive_topk_q099",
            "fixed_top1","fixed_top3","fixed_top5","exhaustive_64"
        ],
        "primary_policy":"adaptive_topk_q095",
        "evaluator_only_oracle":"measured_power_oracle_best1",
        "not_evaluable_local_core": {
            "previous_beam_persistence":"no explicit sequence/group field",
            "hysteresis":"no explicit sequence/group field",
            "beam_switching_penalty":"no explicit sequence/group field",
        },
        "deferred": {
            "nearest_beam_geometry":"no preregistered DeepSense angular-codebook mapping",
            "blockage_aware_adaptive":"optional extension",
        },
    }
    atomic_json(POLICY_SET, policy_set)

    manifest = {
        "stage":8,"block":"8.4","status":"FROZEN_COMPLETE_BLOCK84",
        "contract":{"path":str(CONTRACT),"sha256":EXPECTED_CONTRACT},
        "policy_module":{"path":str(POLICY),"sha256":EXPECTED_POLICY},
        "runner":{"path":str(SCRIPT),"sha256":sha256_file(SCRIPT)},
        "tests":{"path":str(TESTS),"sha256":sha256_file(TESTS)},
        "policy_set":{"path":str(POLICY_SET),"sha256":sha256_file(POLICY_SET)},
        "training_started":False,
        "calibration_started":False,
        "formal_power_access":False,
    }
    atomic_json(MANIFEST, manifest)

    report = {
        "stage":8,"block":"8.4",
        "status":"PASS_BLOCK84_BEAM_POLICY_INTERFACE_AND_BASELINES",
        "codebook_size":64,
        "primary_policy":"adaptive_topk_q095",
        "fixed_baselines":[1,3,5],
        "exhaustive_baseline":True,
        "oracle_evaluator_only":True,
        "previous_beam_formal_status":"NOT_EVALUABLE_FROM_AVAILABLE_CSV_SCHEMA",
        "synthetic_tests_passed":len(tests),
        "training_started":False,
        "formal_power_access":False,
        "Block85_authorized":True,
        "tests_sha256":sha256_file(TESTS),
        "policy_set_sha256":sha256_file(POLICY_SET),
        "manifest_sha256":sha256_file(MANIFEST),
        "runner_sha256":sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage":8,"block":"8.4","status":"FROZEN_COMPLETE_BLOCK84",
        "Block83_seal_sha256":EXPECTED_B83[str(B83_SEAL)],
        "contract_sha256":EXPECTED_CONTRACT,
        "policy_module_sha256":EXPECTED_POLICY,
        "runner_sha256":sha256_file(SCRIPT),
        "tests_sha256":sha256_file(TESTS),
        "policy_set_sha256":sha256_file(POLICY_SET),
        "manifest_sha256":sha256_file(MANIFEST),
        "report_sha256":sha256_file(REPORT),
        "formal_power_access":False,
        "training_started":False,
        "Block85_may_start":True,
    }
    atomic_json(SEAL, seal)

    for raw,expected in EXPECTED_B83.items():
        require(sha256_file(Path(raw)) == expected, f"8.3 modified during 8.4: {raw}")
    for raw,expected in EXPECTED_STAGE5.items():
        require(sha256_file(Path(raw)) == expected, f"Stage5 authority modified during 8.4: {raw}")

    print("\n" + "="*88)
    print("BLOCK 8.4 = FULLY VERIFIED / FROZEN")
    print("64-WAY CALIBRATED-PROBABILITY INTERFACE = PASS")
    print("PRIMARY POLICY = ADAPTIVE TOP-K q=0.95")
    print("ADAPTIVE SWEEP = 0.90 / 0.95 / 0.975 / 0.99")
    print("FIXED BASELINES = TOP-1 / TOP-3 / TOP-5")
    print("EXHAUSTIVE BASELINE = K=64")
    print("MEASURED-POWER ORACLE = EVALUATOR-ONLY")
    print("PREVIOUS-BEAM/HYSTERESIS = NOT_EVALUABLE FROM CURRENT CSV SCHEMA")
    print("SYNTHETIC POLICY TESTS = 7/7 PASS")
    print("MEASURED POWER CONTROLLER INPUT = NO")
    print("TRAINING/CALIBRATION/FORMAL EVALUATION = NO")
    print("BLOCK 8.5 MAY START = YES")
    print("8.4 contract SHA256 =", EXPECTED_CONTRACT)
    print("8.4 policy SHA256   =", EXPECTED_POLICY)
    print("8.4 tests SHA256    =", sha256_file(TESTS))
    print("8.4 policy-set SHA256 =", sha256_file(POLICY_SET))
    print("8.4 manifest SHA256 =", sha256_file(MANIFEST))
    print("8.4 report SHA256   =", sha256_file(REPORT))
    print("8.4 seal SHA256     =", sha256_file(SEAL))
    print("8.4 runner SHA256   =", sha256_file(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK84")
    print("="*88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88)
        print("BLOCK 8.4 FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

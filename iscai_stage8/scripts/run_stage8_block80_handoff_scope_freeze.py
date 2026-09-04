#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"
S7 = ROOT / "iscai_stage7"
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block80_handoff_scope_freeze.py"
CONTRACT = S8 / "configs/stage8_block80_stage7_handoff_scope_contract.json"

MANIFEST = S8 / "artifacts/block80_stage7_handoff/stage8_block80_handoff_manifest.json"
REPORT = S8 / "reports/stage8_block80_handoff_scope_report.json"
SEAL = S8 / "artifacts/block80_stage7_handoff/stage8_block80_scope_freeze_seal.json"

EXPECTED_CONTRACT_SHA = "0ff3008ba5746657ee1832b0c804cfa7e6b3d72c4f7c3fe56dd6c63c92036f9b"

EXPECTED = {
    str(S7 / "artifacts/stage7_pdf_compliance_complete_superseding_seal.json"):
        "9f483cc19546b93ed5a28e045965eba39494d7aca9b2ba949fc49584965b0b0a",
    str(S7 / "artifacts/stage7_block75c3_final_reproducibility_seal.json"):
        "e01ba2e735d4d6403a584f8f1a304e3b3acb926d36fb3adb3cb0f62c5cc44d8a",
    str(S7 / "configs/stage7_block75d_latency_resource_closure_contract.json"):
        "b4f3c212093ffa45924c72272f8d95d5dbf3e2244df9ab0cacbdb3360b81da5e",
    str(S7 / "scripts/run_block75d_latency_resource_closure_complete.py"):
        "f084510fb20f0e8a41062564ba95851565e3282a2958277ee6677908672e296c",
    str(S7 / "artifacts/block75d_latency_resource_closure_complete/five_system_latency_resource_table.json"):
        "d31b9ae686d7db592a2e84634446a3a3127c13e01457a1bfa12b6e1a90fbc3f0",
    str(S7 / "artifacts/block75d_latency_resource_closure_complete/measured_latency_sensitivity.json"):
        "b793be84cb401ab03bea540c49210354f46a4b98d0e996db878aec6caf71903b",
    str(S7 / "artifacts/block75d_latency_resource_closure_complete/block75d_manifest.json"):
        "fcca7e34a88b53ecdd2272e259b1199e82099c9170a97aadb8db49db03078019",
    str(S7 / "reports/stage7_block75d_latency_resource_closure_complete_report.json"):
        "8f15d0c8ccc29dfc0f35a275be36c24c62e7e7cf108957651ebb03c9741e76b8",

    str(S5 / "src/iscai_stage5/adaptive_topk.py"):
        "08bb2145eac117132166a0f8b96a38c61b0a912cfd9b7f2d48f16db649d22865",
    str(S5 / "src/iscai_stage5/adaptive_topk_temporal.py"):
        "433dad357bcba95d5e85d650ee0dcf7ffc5ef157aa8294df1fdba0ef3ab4627a",
    str(S5 / "src/iscai_stage5/beam_latency.py"):
        "81c57865f5b5994e0dbbf9fba90e3a9e6638b99da1c7d7abccbe34a09d8d2810",
    str(S5 / "configs/beam_codebook_policy.json"):
        "bd94f8609393a7c9fe02762cc4bf38e3a77a90cc31adef5f2e6b06aece4074d7",
    str(S5 / "configs/angular_posterior_policy.json"):
        "846f6bf3d3419a2ea99fabc6293514726388bf60a31bb2a889aaf18b487fd82e",
    str(S5 / "configs/beam_latency_policy.json"):
        "aa04f2e73da591b87f2ab2e962d1b69c1bad645c043c0d15970dbf0990080ed0",
}

class FailClosed(RuntimeError):
    pass

def require(cond: bool, msg: str) -> None:
    if not cond:
        raise FailClosed(msg)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing overwrite: {path}")
    data = (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> int:
    print("="*88)
    print("STAGE 8 — BLOCK 8.0")
    print("STAGE7 -> STAGE8 IMMUTABLE HANDOFF + DEEPSENSE SCOPE FREEZE")
    print("NO DEEPSENSE DATA ACCESS / NO TRAINING / NO CALIBRATION / NO FORMAL EVALUATION")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(),
            f"Runner path mismatch: actual={Path(__file__).resolve()} expected={SCRIPT}")
    require(CONTRACT.is_file(), f"Missing scope contract: {CONTRACT}")
    require(sha256_file(CONTRACT) == EXPECTED_CONTRACT_SHA, "Block8.0 contract SHA changed.")

    for p in (MANIFEST, REPORT, SEAL):
        require(not p.exists(), f"Block8.0 output already exists: {p}")

    print("\n===== A. FROZEN STAGE7 + STAGE5 IDENTITY GATE =====")
    for raw, expected in EXPECTED.items():
        p = Path(raw)
        require(p.is_file(), f"Missing frozen authority: {p}")
        actual = sha256_file(p)
        require(actual == expected, f"Frozen SHA mismatch: {p}: {actual} != {expected}")
    print("Stage7 PDF-compliance seal + C3 seal = PASS")
    print("Stage7 7.5D closure identities = PASS")
    print("Stage5 adaptive Top-K / temporal / latency policies = PASS")

    print("\n===== B. STAGE7 FINAL-SEAL SEMANTIC GATE =====")
    s7seal = load_json(S7 / "artifacts/stage7_pdf_compliance_complete_superseding_seal.json")
    require(s7seal.get("status") == "STAGE7_PDF_CORE_COMPLETE_SUPERSEDING_SEAL",
            f"Unexpected Stage7 final seal status: {s7seal.get('status')}")
    require(s7seal.get("stage7_pdf_compliance_100_percent_within_stage7_scope") is True,
            "Stage7 PDF compliance flag is not true.")
    require(s7seal.get("scientific_integrity", {}).get("C1_C2_C3_modified") is False,
            "Stage7 scientific-integrity boundary changed.")
    print("Stage7 PDF core completion semantic readback = PASS")

    print("\n===== C. STAGE8 SCOPE FREEZE =====")
    c = load_json(CONTRACT)
    require(c["pdf_alignment"]["deepsense_scientific_role"] ==
            "measured mmWave validation of the beam-selection policy, not optical-headlamp validation",
            "DeepSense scientific role changed.")
    require(c["immutable_upstream_boundary"]["optical_and_mmWave_result_pooling_allowed"] is False,
            "Optical/mmWave separation changed.")
    require(c["stage8_data_boundary_before_block81"]["DeepSense_measured_beam_powers_accessed"] is False,
            "Measured-power boundary changed.")
    require(c["frozen_policy_semantics_to_preserve"]["primary_coverage"] == 0.95,
            "Primary coverage changed.")
    require(c["pdf_alignment"]["zero_shot_transfer"] == "OPTIONAL_NON_BLOCKING",
            "Zero-shot completion role changed.")
    print("DeepSense = measured-mmWave policy validation only = PASS")
    print("Optical/mmWave result separation = PASS")
    print("Primary adaptive coverage q=0.95 = PASS")
    print("Zero-shot = OPTIONAL/NON-BLOCKING = PASS")

    print("\n===== D. PRE-DEEPSENSE CAUSAL / OUTCOME BOUNDARY =====")
    print("DeepSense dataset accessed = NO")
    print("DeepSense measured beam powers accessed = NO")
    print("DeepSense training started = NO")
    print("DeepSense calibration started = NO")
    print("DeepSense formal evaluation started = NO")
    print("Stage4-Stage7 modifications = NO")

    runner_sha = sha256_file(SCRIPT)

    manifest = {
        "project": "Agni",
        "stage": 8,
        "block": "8.0",
        "status": "FROZEN_COMPLETE_BLOCK80_HANDOFF_SCOPE",
        "contract": {"path": str(CONTRACT), "sha256": EXPECTED_CONTRACT_SHA},
        "runner": {"path": str(SCRIPT), "sha256": runner_sha},
        "frozen_authorities": [
            {"path": raw, "sha256": expected} for raw, expected in sorted(EXPECTED.items())
        ],
        "scope": {
            "DeepSense_role": "measured_mmWave_policy_validation_not_optical_validation",
            "optical_mmWave_separate_tables_required": True,
            "measured_power_evaluator_only": True,
            "primary_coverage": 0.95,
            "coverage_sweep": [0.90, 0.95, 0.975, 0.99],
            "lightweight_retraining_in_stage8": True,
            "recalibration_in_stage8": True,
            "zero_shot_optional_non_blocking": True,
        },
        "pre_data_boundary": {
            "DeepSense_dataset_accessed": False,
            "measured_beam_powers_accessed": False,
            "training_started": False,
            "calibration_started": False,
            "formal_evaluation_started": False,
        },
        "next_block": {
            "block": "8.1",
            "authorized": True,
            "scope": "read-only DeepSense authority/dataset/schema/license/split audit",
        },
    }
    atomic_json(MANIFEST, manifest)

    report = {
        "stage": 8,
        "block": "8.0",
        "status": "PASS_BLOCK80_STAGE7_HANDOFF_AND_SCOPE_FREEZE",
        "Stage7_exact_handoff": True,
        "Stage5_policy_exact_handoff": True,
        "DeepSense_role_frozen": True,
        "optical_mmWave_separation_frozen": True,
        "no_DeepSense_data_access": True,
        "no_training": True,
        "no_calibration": True,
        "no_formal_evaluation": True,
        "Block81_authorized": True,
        "runner_sha256": runner_sha,
        "manifest_sha256": sha256_file(MANIFEST),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage": 8,
        "block": "8.0",
        "status": "FROZEN_COMPLETE_BLOCK80",
        "contract_sha256": EXPECTED_CONTRACT_SHA,
        "runner_sha256": runner_sha,
        "manifest_sha256": sha256_file(MANIFEST),
        "report_sha256": sha256_file(REPORT),
        "Stage7_PDF_compliance_seal_sha256":
            EXPECTED[str(S7 / "artifacts/stage7_pdf_compliance_complete_superseding_seal.json")],
        "DeepSense_data_accessed": False,
        "DeepSense_measured_power_accessed": False,
        "training_started": False,
        "formal_evaluation_started": False,
        "Block81_may_start": True,
    }
    atomic_json(SEAL, seal)

    print("\n" + "="*88)
    print("BLOCK 8.0 = FULLY VERIFIED / FROZEN")
    print("STAGE7 -> STAGE8 IMMUTABLE HANDOFF = PASS")
    print("FROZEN STAGE5 ADAPTIVE POLICY BINDING = PASS")
    print("DEEPSENSE ROLE = MEASURED MMWAVE POLICY VALIDATION")
    print("OPTICAL/MMWAVE SEPARATION = PASS")
    print("DEEPSENSE DATA ACCESSED = NO")
    print("TRAINING/CALIBRATION/FORMAL EVALUATION = NO")
    print("BLOCK 8.1 MAY START = YES")
    print("Block8.0 manifest SHA256 =", sha256_file(MANIFEST))
    print("Block8.0 report SHA256   =", sha256_file(REPORT))
    print("Block8.0 seal SHA256     =", sha256_file(SEAL))
    print("Block8.0 runner SHA256   =", runner_sha)
    print("STATUS = FROZEN_COMPLETE_BLOCK80")
    print("="*88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88)
        print("BLOCK 8.0 FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT ACCESS DEEPSENSE DATA.")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

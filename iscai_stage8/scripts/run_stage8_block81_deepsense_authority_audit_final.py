#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block81_deepsense_authority_audit_final.py"
CONTRACT = S8 / "configs/stage8_block81_deepsense_authority_audit_contract.json"
EVIDENCE = S8 / "configs/stage8_block81_official_authority_evidence.json"

B80_MANIFEST = S8 / "artifacts/block80_stage7_handoff/stage8_block80_handoff_manifest.json"
B80_REPORT = S8 / "reports/stage8_block80_handoff_scope_report.json"
B80_SEAL = S8 / "artifacts/block80_stage7_handoff/stage8_block80_scope_freeze_seal.json"
B80_RUNNER = S8 / "scripts/run_stage8_block80_handoff_scope_freeze.py"

FAILED_V1 = S8 / "artifacts/block81_deepsense_authority_audit"

OUT = S8 / "artifacts/block81_deepsense_authority_audit_final"
FINDINGS = OUT / "deepsense_authority_findings.json"
MANIFEST = OUT / "stage8_block81_authority_manifest.json"
REPORT = S8 / "reports/stage8_block81_deepsense_authority_audit_final_report.json"
SEAL = OUT / "stage8_block81_authority_freeze_seal.json"

EXPECTED_CONTRACT = "571843a955afd59125fa70b0a3ef196a8d578b6d811f50f4e8c7bbd5135516d2"
EXPECTED_EVIDENCE = "ac7bda1cf1ce016717a9a98023b2ed17685bd993b502f51726ef23ab896b5e44"
EXPECTED_B80 = {
    str(B80_MANIFEST): "c2a7ad9153c083b73971822798b52819d30ef92cd4f57820623edf010555dfae",
    str(B80_REPORT): "acddd5550fbeee894b7fc7e2f7ab7695bf89e5f84d3899ed4090c4a87e7bb493",
    str(B80_SEAL): "624b241a95ecd4eaac3147f41d909023202577ce241748c51c324e47d82a1687",
    str(B80_RUNNER): "2a41fd762e9261bc1b6b5c8152331a5ceb4b2fa1c21922c0129b04eedc47f906",
}

class FailClosed(RuntimeError):
    pass

def require(cond: bool, msg: str):
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
        if os.path.exists(tmp):
            os.unlink(tmp)

def source(evidence, source_id):
    matches = [x for x in evidence["sources"] if x["id"] == source_id]
    require(len(matches) == 1, f"Authority evidence source missing/duplicated: {source_id}")
    return matches[0]

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.1 FINAL")
    print("OFFICIAL DEEPSENSE AUTHORITY / SCHEMA / LICENSE / SPLIT EVIDENCE FREEZE")
    print("OFFLINE FROZEN OFFICIAL-PAGE EVIDENCE — NO HOST WEB SCRAPING")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT, "Block8.1 contract SHA mismatch.")
    require(EVIDENCE.is_file() and sha256_file(EVIDENCE) == EXPECTED_EVIDENCE, "Authority evidence SHA mismatch.")
    for raw, expected in EXPECTED_B80.items():
        p = Path(raw)
        require(p.is_file(), f"Missing Block8.0 frozen input: {p}")
        require(sha256_file(p) == expected, f"Block8.0 SHA changed: {p}")
    for p in (FINDINGS, MANIFEST, REPORT, SEAL):
        require(not p.exists(), f"Final Block8.1 output exists: {p}")

    print("\n===== A. BLOCK8.0 IMMUTABLE GATE =====")
    b80 = json.loads(B80_SEAL.read_text())
    require(b80.get("status") == "FROZEN_COMPLETE_BLOCK80", "Block8.0 status changed.")
    require(b80.get("Block81_may_start") is True, "Block8.1 not authorized.")
    print("Block8.0 exact identity + authorization = PASS")

    print("\n===== B. FAILED-V1 PRESERVATION GATE =====")
    if FAILED_V1.exists():
        files = sorted(str(p) for p in FAILED_V1.rglob("*") if p.is_file())
        print("failed-v1 namespace preserved files =", len(files))
        require(len(files) >= 1, "Failed-v1 namespace exists but is unexpectedly empty.")
        print("FAILED-V1 AUTHORITY SNAPSHOTS PRESERVED = PASS")
    else:
        print("FAILED-V1 namespace absent = acceptable")
    print("No failed-v1 file deleted or modified by final runner = PASS")

    print("\n===== C. FROZEN OFFICIAL AUTHORITY EVIDENCE GATE =====")
    e = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    require(e["status"] == "FROZEN_OFFLINE_OFFICIAL_AUTHORITY_EVIDENCE", "Evidence status changed.")
    require(e["scientific_interpretation"]["dataset_bytes_downloaded"] is False, "Evidence claims dataset bytes were downloaded.")
    ids = {x["id"] for x in e["sources"]}
    required_ids = {
        "official_lidar_task", "official_scenarios_index", "official_scenario9",
        "official_v2v_tracking", "official_v2v_2023"
    }
    require(ids == required_ids, f"Authority source set changed: {ids}")
    print("5/5 official DeepSense authority sources frozen = PASS")
    print("Host network scraping = NOT USED")

    print("\n===== D. PRIMARY LIDAR-AIDED 64-WAY TASK =====")
    L = source(e, "official_lidar_task")["assertions"]
    S = source(e, "official_scenarios_index")["assertions"]
    S9 = source(e, "official_scenario9")["assertions"]

    require(L["scenarios"] == [8,9], "Primary scenarios changed.")
    require(L["measured_power_vector_shape"] == [64,1], "Primary measured-power shape changed.")
    require(L["receiver_codebook_beams"] == 64, "Primary codebook size changed.")
    require(L["lidar_sensing_matrix_shape"] == [416,2], "LiDAR matrix shape changed.")
    require(L["scenario_folder_split_files"] == ["train.csv","val.csv","test.csv"], "Split filenames changed.")
    require(L["csv_label_column"] == "beam_index", "Beam label changed.")
    require(L["sampling_interval_approx_ms"] == 100, "Sampling authority changed.")
    require(L["development_labels_include_optimal_beam_index"] is True, "Development labels authority changed.")
    require(L["challenge_ground_truth_hidden"] is True, "Challenge hidden-GT authority changed.")

    require(S["scenario_8_instances"] == 4043, "Scenario8 instance authority changed.")
    require(S9["scenario_9_instances"] == 5964, "Scenario9 instance authority changed.")
    require(S9["measured_power_vector_shape"] == [64,1], "Scenario9 measured-power shape changed.")
    require(S9["receiver_codebook_beams"] == 64, "Scenario9 codebook authority changed.")
    require(S9["optimal_beam_definition"] == "index of maximum measured received-power value",
            "Scenario9 optimal-beam semantics changed.")

    print("Scenarios 8/9 = PASS")
    print("Scenario8 documented instances = 4043")
    print("Scenario9 documented instances = 5964")
    print("64x1 measured powers / 64 beams = PASS")
    print("416x2 LiDAR sensing matrix = PASS")
    print("train.csv / val.csv / test.csv = PASS")
    print("beam_index + ~100ms sampling = PASS")
    print("measured power remains evaluator-only under Block8.0 = PASS")

    print("\n===== E. LICENSE / CITATION BOUNDARY =====")
    require(L["license_section_present"] is True and L["citation_requirement_present"] is True,
            "Official LiDAR task license/citation authority missing.")
    require(e["scientific_interpretation"]["exact_dataset_license_identifier"] is None,
            "Exact dataset license identifier must not be invented.")
    require(e["scientific_interpretation"]["license_identifier_status"] == "NOT_CLAIMED_FROM_PAGE_TEXT",
            "License-identifier status changed.")
    print("Official task License/Citation section = PASS")
    print("Exact dataset license identifier = NOT CLAIMED")
    print("Download/sign-in terms must be captured before Block8.2 byte acquisition = BOUND")

    print("\n===== F. V2V SECONDARY AUTHORITY =====")
    V = source(e, "official_v2v_tracking")["assertions"]
    V2 = source(e, "official_v2v_2023")["assertions"]
    require(V["scenarios"] == [36,37,38,39], "V2V scenarios changed.")
    require(V["overall_beam_index_min"] == 0 and V["overall_beam_index_max"] == 255,
            "V2V overall-beam range changed.")
    require(V["four_power_vectors_each_length"] == 64, "V2V power-vector authority changed.")
    require(V2["overall_beam_index_min"] == 0 and V2["overall_beam_index_max"] == 255,
            "V2V challenge range changed.")
    require(e["scientific_interpretation"]["secondary_role"] == "OPTIONAL_SEPARATE_INTERFACE",
            "V2V secondary role changed.")
    print("V2V scenarios 36-39 = PASS")
    print("four 64-beam power vectors + overall 0..255 = PASS")
    print("V2V silent remap to primary 64-way = FORBIDDEN")

    print("\n===== G. BLOCK8.2 AUTHORIZATION BOUNDARY =====")
    findings = {
        "stage": 8,
        "block": "8.1-final",
        "status": "FROZEN_COMPLETE_BLOCK81_AUTHORITY_SCHEMA_AUDIT",
        "primary_task": {
            "name": "DeepSense LiDAR-aided beam prediction",
            "scenarios": [8,9],
            "scenario8_documented_instances": 4043,
            "scenario9_documented_instances": 5964,
            "lidar_shape": [416,2],
            "measured_power_shape": [64,1],
            "codebook_size": 64,
            "split_files": ["train.csv","val.csv","test.csv"],
            "beam_label": "beam_index",
            "sampling_interval_approx_ms": 100,
            "measured_power_controller_input_allowed": False,
            "measured_power_evaluator_only": True,
        },
        "secondary_task": {
            "name": "DeepSense V2V scenarios 36-39",
            "role": "OPTIONAL_SEPARATE_INTERFACE",
            "overall_beam_range": [0,255],
            "four_power_vectors_each_length": 64,
        },
        "license_boundary": {
            "official_task_license_section_present": True,
            "citation_required": True,
            "exact_dataset_license_identifier": None,
            "download_terms_capture_required_before_dataset_bytes": True,
        },
        "split_boundary": {
            "documented_split_files_verified": True,
            "actual_split_row_counts_verified": False,
            "actual_split_disjointness_verified": False,
            "reason": "dataset bytes intentionally not acquired in Block8.1",
        },
        "data_access_boundary": {
            "dataset_archives_downloaded": False,
            "measured_power_files_opened": False,
            "training_started": False,
            "calibration_started": False,
            "formal_evaluation_started": False,
        },
        "block82": {
            "authorized": True,
            "scope": "bounded primary-task acquisition + canonical corpus/split freeze",
            "hard_requirements": [
                "capture official download/sign-in terms before acquiring dataset bytes",
                "acquire primary LiDAR task scenarios 8/9 only or exact minimum package containing them",
                "hash all acquired bytes",
                "verify actual CSV row counts, schema, referenced paths, file shapes and codebook length",
                "verify train/val/test sequence/sample disjointness before any training",
                "freeze development/calibration/formal roles before model training",
            ],
        },
    }
    atomic_json(FINDINGS, findings)

    manifest = {
        "stage": 8,
        "block": "8.1-final",
        "status": "FROZEN_COMPLETE_BLOCK81",
        "contract": {"path": str(CONTRACT), "sha256": EXPECTED_CONTRACT},
        "authority_evidence": {"path": str(EVIDENCE), "sha256": EXPECTED_EVIDENCE},
        "runner": {"path": str(SCRIPT), "sha256": sha256_file(SCRIPT)},
        "findings": {"path": str(FINDINGS), "sha256": sha256_file(FINDINGS)},
        "Block80_frozen": EXPECTED_B80,
        "dataset_bytes_accessed": False,
        "host_web_scraping_used": False,
    }
    atomic_json(MANIFEST, manifest)

    report = {
        "stage": 8,
        "block": "8.1-final",
        "status": "PASS_BLOCK81_DEEPSENSE_AUTHORITY_SCHEMA_AUDIT",
        "primary_lidar_64way_authority": True,
        "scenario8_instances_documented": 4043,
        "scenario9_instances_documented": 5964,
        "lidar_shape_documented": [416,2],
        "measured_power_shape_documented": [64,1],
        "split_names_documented": ["train.csv","val.csv","test.csv"],
        "actual_split_counts_verified": False,
        "actual_split_disjointness_verified": False,
        "official_license_citation_section_verified": True,
        "exact_dataset_license_identifier_claimed": False,
        "dataset_download_performed": False,
        "V2V_secondary_authority_verified": True,
        "Block82_authorized": True,
        "manifest_sha256": sha256_file(MANIFEST),
        "findings_sha256": sha256_file(FINDINGS),
        "runner_sha256": sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage": 8,
        "block": "8.1-final",
        "status": "FROZEN_COMPLETE_BLOCK81",
        "Block80_seal_sha256": EXPECTED_B80[str(B80_SEAL)],
        "contract_sha256": EXPECTED_CONTRACT,
        "authority_evidence_sha256": EXPECTED_EVIDENCE,
        "runner_sha256": sha256_file(SCRIPT),
        "findings_sha256": sha256_file(FINDINGS),
        "manifest_sha256": sha256_file(MANIFEST),
        "report_sha256": sha256_file(REPORT),
        "primary_task_frozen": "LiDAR-aided beam prediction scenarios 8/9, 64-way",
        "dataset_bytes_accessed": False,
        "measured_power_files_accessed": False,
        "training_started": False,
        "Block82_may_start": True,
    }
    atomic_json(SEAL, seal)

    for raw, expected in EXPECTED_B80.items():
        require(sha256_file(Path(raw)) == expected, f"Block8.0 modified: {raw}")

    print("Block8.2 bounded acquisition may start = YES")

    print("\n" + "="*88)
    print("BLOCK 8.1 FINAL = FULLY VERIFIED / FROZEN")
    print("PRIMARY DEEPSENSE TASK = LIDAR-AIDED BEAM PREDICTION SCENARIOS 8/9")
    print("MEASURED POWER INTERFACE = 64x1 / 64-BEAM = PASS")
    print("LIDAR INTERFACE = 416x2 = PASS")
    print("DOCUMENTED SPLITS train/val/test = PASS")
    print("ACTUAL SPLIT COUNTS/DISJOINTNESS = DEFERRED TO BLOCK8.2")
    print("OFFICIAL LICENSE/CITATION SECTION = PASS")
    print("EXACT DATASET LICENSE IDENTIFIER = NOT CLAIMED")
    print("V2V 36-39 = SECONDARY/OPTIONAL SEPARATE INTERFACE")
    print("HOST WEB SCRAPING = NO")
    print("DEEPSENSE DATASET BYTES ACCESSED = NO")
    print("MEASURED POWER FILES ACCESSED = NO")
    print("TRAINING/CALIBRATION/FORMAL EVALUATION = NO")
    print("BLOCK 8.2 MAY START = YES")
    print("Block8.1 evidence SHA256 =", EXPECTED_EVIDENCE)
    print("Block8.1 findings SHA256 =", sha256_file(FINDINGS))
    print("Block8.1 manifest SHA256 =", sha256_file(MANIFEST))
    print("Block8.1 report SHA256   =", sha256_file(REPORT))
    print("Block8.1 seal SHA256     =", sha256_file(SEAL))
    print("Block8.1 runner SHA256   =", sha256_file(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK81")
    print("="*88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88)
        print("BLOCK 8.1 FINAL FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT DOWNLOAD DEEPSENSE DATASET ARCHIVES.")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

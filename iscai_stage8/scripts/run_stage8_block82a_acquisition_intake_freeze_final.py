#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"
INCOMING = ROOT / "deepsense_incoming/block82_primary"

SCRIPT = S8 / "scripts/run_stage8_block82a_acquisition_intake_freeze_final.py"
CONTRACT = S8 / "configs/stage8_block82a_acquisition_intake_contract_final.json"

B81_ROOT = S8 / "artifacts/block81_deepsense_authority_audit_final"
B81_FINDINGS = B81_ROOT / "deepsense_authority_findings.json"
B81_MANIFEST = B81_ROOT / "stage8_block81_authority_manifest.json"
B81_SEAL = B81_ROOT / "stage8_block81_authority_freeze_seal.json"
B81_REPORT = S8 / "reports/stage8_block81_deepsense_authority_audit_final_report.json"
B81_EVIDENCE = S8 / "configs/stage8_block81_official_authority_evidence.json"
B81_RUNNER = S8 / "scripts/run_stage8_block81_deepsense_authority_audit_final.py"

FAILED_V1_ROOT = S8 / "artifacts/block82a_deepsense_acquisition_intake"

OUT = S8 / "artifacts/block82a_deepsense_acquisition_intake_final"
FILE_MANIFEST = OUT / "input_file_manifest.json"
ARCHIVE_INVENTORY = OUT / "archive_member_inventory.json"
REPORT = S8 / "reports/stage8_block82a_acquisition_intake_final_report.json"
SEAL = OUT / "stage8_block82a_acquisition_intake_seal.json"

EXPECTED_CONTRACT = "a4def4e27ade3d891903d8c4c5c81a861d25af3403659a86e194c126b0c86edc"
EXPECTED_B81 = {
    str(B81_SEAL): "cbde57e2ecb915e45d1560db6c010132151d03ef269346caad65023905ad97e5",
    str(B81_MANIFEST): "960ce5fa544d5636d0593835c379f5d0e2530b64d8d91dcee807c8a903747b51",
    str(B81_REPORT): "828efa4469a435255a2e2960bd52c6248c9e5bdd17ff6eb41fa2f9e94fefc57d",
    str(B81_FINDINGS): "d932aeb9a7371996edb145e9b4bb02842692e1e4a249c7ad710ac5eb692ad808",
    str(B81_EVIDENCE): "ac7bda1cf1ce016717a9a98023b2ed17685bd993b502f51726ef23ab896b5e44",
    str(B81_RUNNER): "7ceccb3be41856fee80196ad59690e4e75f34bd3583ce716f5ac9693db9ad64b",
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

def safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts

def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())

def classify_member_scenarios(name: str):
    n = normalize_name(name)
    hits = []
    # Intentionally require explicit scenario token in member path.
    if "scenario8" in n:
        hits.append(8)
    if "scenario9" in n:
        hits.append(9)
    return hits

def inventory_archive(p: Path):
    lower = p.name.lower()
    members = []
    kind = None

    if lower.endswith(".zip"):
        kind = "zip"
        with zipfile.ZipFile(p, "r") as z:
            bad = z.testzip()
            require(bad is None, f"ZIP CRC failure in {p} member {bad}")
            for info in z.infolist():
                require(safe_member(info.filename), f"Unsafe ZIP member path: {info.filename}")
                members.append({
                    "name": info.filename,
                    "is_dir": bool(info.is_dir()),
                    "compressed_size": int(info.compress_size),
                    "uncompressed_size": int(info.file_size),
                })
    elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        kind = "tar"
        with tarfile.open(p, "r:*") as t:
            for info in t.getmembers():
                require(safe_member(info.name), f"Unsafe TAR member path: {info.name}")
                members.append({
                    "name": info.name,
                    "is_dir": bool(info.isdir()),
                    "compressed_size": None,
                    "uncompressed_size": int(info.size),
                })
    else:
        raise FailClosed(f"Unsupported archive format: {p.name}")

    scenarios = sorted(set(
        sc for m in members for sc in classify_member_scenarios(m["name"])
    ))

    # Helpful structural probes only; no payloads opened.
    csv_names = sorted(m["name"] for m in members
                       if not m["is_dir"] and PurePosixPath(m["name"]).name.lower() in
                       ("train.csv", "val.csv", "test.csv"))
    unit1_dirs = sorted(m["name"] for m in members
                        if "unit1" in normalize_name(m["name"]))

    return {
        "kind": kind,
        "member_count": len(members),
        "members": members,
        "scenario_candidates_from_members": scenarios,
        "documented_split_csv_member_candidates": csv_names,
        "unit1_member_candidates_count": len(unit1_dirs),
    }

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.2A FINAL")
    print("DEEPSENSE ACQUISITION INTAKE — SINGLE CONTAINER COMPATIBLE")
    print("HASH + CRC + MEMBER INVENTORY ONLY — NO EXTRACTION / NO PAYLOAD OPEN")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT,
            "Block8.2A-final contract SHA mismatch.")

    for raw, expected in EXPECTED_B81.items():
        p = Path(raw)
        require(p.is_file(), f"Missing Block8.1 authority: {p}")
        require(sha256_file(p) == expected, f"Block8.1 SHA changed: {p}")

    for p in (FILE_MANIFEST, ARCHIVE_INVENTORY, REPORT, SEAL):
        require(not p.exists(), f"Block8.2A-final output exists: {p}")

    print("\n===== A. BLOCK8.1 IMMUTABLE AUTHORIZATION =====")
    b81 = json.loads(B81_SEAL.read_text())
    require(b81.get("status") == "FROZEN_COMPLETE_BLOCK81", "Block8.1 status changed.")
    require(b81.get("Block82_may_start") is True, "Block8.2 not authorized.")
    print("Block8.1 exact identity + authorization = PASS")

    print("\n===== B. FAILED 8.2A ATTEMPT PRESERVATION =====")
    if FAILED_V1_ROOT.exists():
        old_files = sorted(p for p in FAILED_V1_ROOT.rglob("*") if p.is_file())
        print("failed-v1 artifact files =", len(old_files))
        print("FAILED-V1 NAMESPACE PRESERVED = PASS")
    else:
        print("failed-v1 artifact namespace absent = acceptable")

    print("\n===== C. USER OFFICIAL-ACCESS ATTESTATION =====")
    require(os.environ.get("DEEPSENSE_ACCESS_CONFIRMED") == "YES",
            "DEEPSENSE_ACCESS_CONFIRMED=YES is required after official signed-in access.")
    print("Official signed-in access by user = ATTESTED")
    print("Credentials captured by runner = NO")

    print("\n===== D. INPUT BYTE FREEZE =====")
    require(INCOMING.is_dir(), f"Missing incoming directory: {INCOMING}")
    files = sorted(p for p in INCOMING.iterdir() if p.is_file())
    require(len(files) >= 1, f"Need >=1 primary-task archive in {INCOMING}; found {len(files)}")

    file_rows = []
    total_bytes = 0
    for p in files:
        st = p.stat()
        require(st.st_size > 0, f"Empty input file: {p}")
        digest = sha256_file(p)
        file_rows.append({
            "path": str(p), "name": p.name, "bytes": int(st.st_size), "sha256": digest
        })
        total_bytes += int(st.st_size)
        print(f"{p.name} bytes={st.st_size} sha256={digest}")
    print("top-level input files =", len(files))
    print("total input bytes =", total_bytes)

    print("\n===== E. ARCHIVE CRC / MEMBER-NAME INVENTORY =====")
    archive_rows = []
    scenarios_seen = set()
    all_split_csv_candidates = []

    for p in files:
        inv = inventory_archive(p)
        scenarios_seen.update(inv["scenario_candidates_from_members"])
        all_split_csv_candidates.extend(inv["documented_split_csv_member_candidates"])
        archive_rows.append({
            "path": str(p),
            "sha256": sha256_file(p),
            **inv,
        })
        print(
            f"{p.name} kind={inv['kind']} members={inv['member_count']} "
            f"scenarios={inv['scenario_candidates_from_members']} "
            f"split_csv_candidates={len(inv['documented_split_csv_member_candidates'])}"
        )

    require(8 in scenarios_seen, "Scenario 8 not identified from supported archive member names.")
    require(9 in scenarios_seen, "Scenario 9 not identified from supported archive member names.")

    print("Scenario 8 identified from archive members = PASS")
    print("Scenario 9 identified from archive members = PASS")
    print("Both scenarios may reside in same archive = PASS")
    print("Archive extraction performed = NO")
    print("Measured-power payload opened = NO")

    manifest = {
        "stage": 8,
        "block": "8.2A-final",
        "status": "FROZEN_COMPLETE_BLOCK82A_ACQUISITION_INTAKE",
        "contract": {"path": str(CONTRACT), "sha256": EXPECTED_CONTRACT},
        "runner": {"path": str(SCRIPT), "sha256": sha256_file(SCRIPT)},
        "incoming_root": str(INCOMING),
        "input_files": file_rows,
        "total_bytes": total_bytes,
        "scenario_candidates_from_archive_members": sorted(scenarios_seen),
        "split_csv_member_candidates": sorted(all_split_csv_candidates),
        "archive_extraction_performed": False,
        "measured_power_payloads_opened": False,
        "training_started": False,
        "calibration_started": False,
        "formal_scoring_started": False,
    }
    atomic_json(FILE_MANIFEST, manifest)

    atomic_json(ARCHIVE_INVENTORY, {
        "stage": 8,
        "block": "8.2A-final",
        "status": "FROZEN_ARCHIVE_MEMBER_INVENTORY",
        "archives": archive_rows,
        "scenario_candidates_from_members": sorted(scenarios_seen),
        "extraction_performed": False,
    })

    report = {
        "stage": 8,
        "block": "8.2A-final",
        "status": "PASS_BLOCK82A_ACQUISITION_INTAKE",
        "input_file_count": len(file_rows),
        "input_total_bytes": total_bytes,
        "all_input_hashes_recorded": True,
        "archive_crc_integrity_checked": True,
        "scenario8_identified_from_member_inventory": True,
        "scenario9_identified_from_member_inventory": True,
        "single_container_layout_supported": True,
        "archive_extraction_performed": False,
        "measured_power_payloads_opened": False,
        "training_calibration_formal_scoring": False,
        "Block82B_authorized": True,
        "input_manifest_sha256": sha256_file(FILE_MANIFEST),
        "archive_inventory_sha256": sha256_file(ARCHIVE_INVENTORY),
        "runner_sha256": sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage": 8,
        "block": "8.2A-final",
        "status": "FROZEN_COMPLETE_BLOCK82A",
        "Block81_seal_sha256": EXPECTED_B81[str(B81_SEAL)],
        "contract_sha256": EXPECTED_CONTRACT,
        "runner_sha256": sha256_file(SCRIPT),
        "input_manifest_sha256": sha256_file(FILE_MANIFEST),
        "archive_inventory_sha256": sha256_file(ARCHIVE_INVENTORY),
        "report_sha256": sha256_file(REPORT),
        "scenario8_identified": True,
        "scenario9_identified": True,
        "archive_extraction_performed": False,
        "training_started": False,
        "Block82B_may_start": True,
    }
    atomic_json(SEAL, seal)

    # Exact post-run input/upstream immutability.
    for raw, expected in EXPECTED_B81.items():
        require(sha256_file(Path(raw)) == expected, f"Block8.1 modified: {raw}")
    for row in file_rows:
        require(sha256_file(Path(row["path"])) == row["sha256"],
                f"Incoming file modified during intake: {row['path']}")

    print("\n" + "="*88)
    print("BLOCK 8.2A FINAL = FULLY VERIFIED / FROZEN")
    print("OFFICIAL USER ACCESS ATTESTATION = PASS")
    print("ALL INPUT BYTES HASHED = PASS")
    print("ARCHIVE CRC / MEMBER INVENTORY = PASS")
    print("SCENARIO 8 FROM MEMBER INVENTORY = PASS")
    print("SCENARIO 9 FROM MEMBER INVENTORY = PASS")
    print("SINGLE LiDAR.zip CONTAINING BOTH SCENARIOS = SUPPORTED")
    print("ARCHIVE EXTRACTION = NO")
    print("MEASURED POWER PAYLOADS OPENED = NO")
    print("TRAINING/CALIBRATION/FORMAL SCORING = NO")
    print("BLOCK 8.2B MAY START = YES")
    print("Block8.2A manifest SHA256 =", sha256_file(FILE_MANIFEST))
    print("Block8.2A inventory SHA256 =", sha256_file(ARCHIVE_INVENTORY))
    print("Block8.2A report SHA256 =", sha256_file(REPORT))
    print("Block8.2A seal SHA256 =", sha256_file(SEAL))
    print("Block8.2A runner SHA256 =", sha256_file(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK82A")
    print("="*88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88)
        print("BLOCK 8.2A FINAL FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT EXTRACT OR DELETE THE INPUT ARCHIVE.")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

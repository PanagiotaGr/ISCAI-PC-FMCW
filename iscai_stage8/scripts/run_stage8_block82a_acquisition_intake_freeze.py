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

SCRIPT = S8 / "scripts/run_stage8_block82a_acquisition_intake_freeze.py"
CONTRACT = S8 / "configs/stage8_block82a_acquisition_intake_contract.json"

B81_ROOT = S8 / "artifacts/block81_deepsense_authority_audit_final"
B81_FINDINGS = B81_ROOT / "deepsense_authority_findings.json"
B81_MANIFEST = B81_ROOT / "stage8_block81_authority_manifest.json"
B81_SEAL = B81_ROOT / "stage8_block81_authority_freeze_seal.json"
B81_REPORT = S8 / "reports/stage8_block81_deepsense_authority_audit_final_report.json"
B81_EVIDENCE = S8 / "configs/stage8_block81_official_authority_evidence.json"
B81_RUNNER = S8 / "scripts/run_stage8_block81_deepsense_authority_audit_final.py"

OUT = S8 / "artifacts/block82a_deepsense_acquisition_intake"
FILE_MANIFEST = OUT / "input_file_manifest.json"
ARCHIVE_INVENTORY = OUT / "archive_member_inventory.json"
REPORT = S8 / "reports/stage8_block82a_acquisition_intake_report.json"
SEAL = OUT / "stage8_block82a_acquisition_intake_seal.json"

EXPECTED_CONTRACT = "306d586438dd3db2ca9c9e3908150db685a3eeb422f249441837f9941381257c"
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
    # Archive names are interpreted as POSIX paths regardless of host OS.
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts

def classify_scenario(text: str):
    s = text.lower().replace("_", "").replace("-", "").replace(" ", "")
    hits = []
    if "scenario8" in s or "/8/" in text.lower():
        hits.append(8)
    if "scenario9" in s or "/9/" in text.lower():
        hits.append(9)
    return hits

def inventory_archive(p: Path):
    lower = p.name.lower()
    members = []
    kind = None

    if lower.endswith(".zip"):
        kind = "zip"
        with zipfile.ZipFile(p, "r") as z:
            for info in z.infolist():
                name = info.filename
                require(safe_member(name), f"Unsafe ZIP member path in {p}: {name}")
                members.append({
                    "name": name,
                    "is_dir": info.is_dir(),
                    "compressed_size": int(info.compress_size),
                    "uncompressed_size": int(info.file_size),
                })
    elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        kind = "tar"
        with tarfile.open(p, "r:*") as t:
            for info in t.getmembers():
                name = info.name
                require(safe_member(name), f"Unsafe TAR member path in {p}: {name}")
                members.append({
                    "name": name,
                    "is_dir": info.isdir(),
                    "compressed_size": None,
                    "uncompressed_size": int(info.size),
                })
    else:
        return {"supported": False, "kind": None, "members": [], "scenario_candidates": []}

    scenario_candidates = sorted(set(
        x for m in members for x in classify_scenario(m["name"])
    ))
    return {
        "supported": True,
        "kind": kind,
        "member_count": len(members),
        "members": members,
        "scenario_candidates": scenario_candidates,
    }

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.2A")
    print("DEEPSENSE PRIMARY-TASK ACQUISITION INTAKE FREEZE")
    print("HASH + ARCHIVE MEMBER INVENTORY ONLY — NO EXTRACTION / NO TRAINING")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT, "Block8.2A contract SHA mismatch.")
    for raw, expected in EXPECTED_B81.items():
        p = Path(raw)
        require(p.is_file(), f"Missing Block8.1 authority: {p}")
        require(sha256_file(p) == expected, f"Block8.1 SHA changed: {p}")

    for p in (FILE_MANIFEST, ARCHIVE_INVENTORY, REPORT, SEAL):
        require(not p.exists(), f"Block8.2A output exists: {p}")

    print("\n===== A. BLOCK8.1 IMMUTABLE AUTHORIZATION =====")
    b81 = json.loads(B81_SEAL.read_text())
    require(b81.get("status") == "FROZEN_COMPLETE_BLOCK81", "Block8.1 status changed.")
    require(b81.get("Block82_may_start") is True, "Block8.2 not authorized.")
    require(b81.get("dataset_bytes_accessed") is False, "Block8.1 data boundary changed.")
    print("Block8.1 exact identity + Block8.2 authorization = PASS")

    print("\n===== B. USER-PERFORMED OFFICIAL ACCESS ATTESTATION =====")
    require(os.environ.get("DEEPSENSE_ACCESS_CONFIRMED") == "YES",
            "Set DEEPSENSE_ACCESS_CONFIRMED=YES only after obtaining Scenario 8/9 files through the official signed-in DeepSense access flow.")
    print("Official signed-in access performed by user = ATTESTED")
    print("Credentials captured by runner = NO")

    print("\n===== C. BOUNDED INCOMING FILE INVENTORY =====")
    require(INCOMING.is_dir(), f"Missing incoming directory: {INCOMING}")
    files = sorted(p for p in INCOMING.iterdir() if p.is_file())
    require(len(files) >= 2, f"Need at least two downloaded primary-task files in {INCOMING}; found {len(files)}")

    file_rows = []
    total_bytes = 0
    for p in files:
        st = p.stat()
        require(st.st_size > 0, f"Empty input file: {p}")
        row = {
            "path": str(p),
            "name": p.name,
            "bytes": int(st.st_size),
            "sha256": sha256_file(p),
        }
        file_rows.append(row)
        total_bytes += int(st.st_size)
        print(f"{p.name} bytes={st.st_size} sha={row['sha256'][:12]}")
    print(f"input files = {len(files)}")
    print(f"total input bytes = {total_bytes}")

    print("\n===== D. ARCHIVE MEMBER-NAME AUDIT — NO EXTRACTION =====")
    archive_rows = []
    scenario_to_files = {8: [], 9: []}
    unsupported = []
    for p in files:
        inv = inventory_archive(p)
        row = {
            "path": str(p),
            "sha256": sha256_file(p),
            "supported_archive": inv["supported"],
            "archive_kind": inv.get("kind"),
            "member_count": inv.get("member_count", 0),
            "scenario_candidates": inv.get("scenario_candidates", []),
        }
        if inv["supported"]:
            # Retain full member metadata in immutable inventory, not stdout.
            row["members"] = inv["members"]
            for sc in inv["scenario_candidates"]:
                scenario_to_files[sc].append(str(p))
            print(f"{p.name}: {inv['kind']} members={inv.get('member_count',0)} scenarios={inv.get('scenario_candidates',[])}")
        else:
            unsupported.append(str(p))
            # Filename may still provide a scenario hint, but it is not sufficient for 8.2B authorization.
            for sc in classify_scenario(p.name):
                scenario_to_files[sc].append(str(p))
            print(f"{p.name}: unsupported archive format")
        archive_rows.append(row)

    require(scenario_to_files[8], "Could not identify any Scenario 8 candidate from supported archive member names.")
    require(scenario_to_files[9], "Could not identify any Scenario 9 candidate from supported archive member names.")
    require(not unsupported,
            f"Unsupported downloaded archive format(s) prevent safe 8.2B authorization: {unsupported}")

    print("Scenario 8 candidate archive = PASS")
    print("Scenario 9 candidate archive = PASS")
    print("Supported archive formats only = PASS")
    print("Archive extraction performed = NO")

    manifest = {
        "stage": 8,
        "block": "8.2A",
        "status": "FROZEN_COMPLETE_BLOCK82A_ACQUISITION_INTAKE",
        "contract": {"path": str(CONTRACT), "sha256": EXPECTED_CONTRACT},
        "runner": {"path": str(SCRIPT), "sha256": sha256_file(SCRIPT)},
        "incoming_root": str(INCOMING),
        "user_access_attestation": {
            "official_signed_in_access_confirmed": True,
            "credentials_captured": False,
        },
        "files": file_rows,
        "total_bytes": total_bytes,
        "scenario_candidates": {
            "8": scenario_to_files[8],
            "9": scenario_to_files[9],
        },
        "input_modified": False,
        "archive_extraction_performed": False,
        "measured_power_payloads_opened": False,
        "training_started": False,
        "calibration_started": False,
        "formal_scoring_started": False,
    }
    atomic_json(FILE_MANIFEST, manifest)

    atomic_json(ARCHIVE_INVENTORY, {
        "stage": 8,
        "block": "8.2A",
        "status": "FROZEN_ARCHIVE_MEMBER_INVENTORY",
        "archives": archive_rows,
        "scenario_candidates": {
            "8": scenario_to_files[8],
            "9": scenario_to_files[9],
        },
        "extraction_performed": False,
    })

    report = {
        "stage": 8,
        "block": "8.2A",
        "status": "PASS_BLOCK82A_ACQUISITION_INTAKE",
        "input_file_count": len(file_rows),
        "input_total_bytes": total_bytes,
        "all_input_hashes_recorded": True,
        "scenario8_candidate_identified": True,
        "scenario9_candidate_identified": True,
        "supported_archives_only": True,
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
        "block": "8.2A",
        "status": "FROZEN_COMPLETE_BLOCK82A",
        "Block81_seal_sha256": EXPECTED_B81[str(B81_SEAL)],
        "contract_sha256": EXPECTED_CONTRACT,
        "runner_sha256": sha256_file(SCRIPT),
        "input_manifest_sha256": sha256_file(FILE_MANIFEST),
        "archive_inventory_sha256": sha256_file(ARCHIVE_INVENTORY),
        "report_sha256": sha256_file(REPORT),
        "scenario8_candidate_identified": True,
        "scenario9_candidate_identified": True,
        "archive_extraction_performed": False,
        "training_started": False,
        "Block82B_may_start": True,
    }
    atomic_json(SEAL, seal)

    # Upstream + input byte immutability recheck.
    for raw, expected in EXPECTED_B81.items():
        require(sha256_file(Path(raw)) == expected, f"Block8.1 modified during 8.2A: {raw}")
    for row in file_rows:
        require(sha256_file(Path(row["path"])) == row["sha256"], f"Input file modified during intake: {row['path']}")

    print("\n" + "="*88)
    print("BLOCK 8.2A = FULLY VERIFIED / FROZEN")
    print("OFFICIAL USER ACCESS ATTESTATION = PASS")
    print("ALL DOWNLOADED INPUT BYTES HASHED = PASS")
    print("SCENARIO 8 CANDIDATE = PASS")
    print("SCENARIO 9 CANDIDATE = PASS")
    print("ARCHIVE MEMBER INVENTORY = PASS")
    print("ARCHIVE EXTRACTION = NO")
    print("MEASURED POWER PAYLOADS OPENED = NO")
    print("TRAINING/CALIBRATION/FORMAL SCORING = NO")
    print("BLOCK 8.2B MAY START = YES")
    print("Block8.2A input manifest SHA256 =", sha256_file(FILE_MANIFEST))
    print("Block8.2A archive inventory SHA256 =", sha256_file(ARCHIVE_INVENTORY))
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
        print("BLOCK 8.2A FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT EXTRACT OR DELETE THE DOWNLOADED INPUT FILES.")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

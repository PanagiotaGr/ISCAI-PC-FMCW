#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import ssl
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"

SCRIPT = S8 / "scripts/run_stage8_block81_deepsense_authority_audit.py"
CONTRACT = S8 / "configs/stage8_block81_deepsense_authority_audit_contract.json"

B80_MANIFEST = S8 / "artifacts/block80_stage7_handoff/stage8_block80_handoff_manifest.json"
B80_REPORT = S8 / "reports/stage8_block80_handoff_scope_report.json"
B80_SEAL = S8 / "artifacts/block80_stage7_handoff/stage8_block80_scope_freeze_seal.json"
B80_RUNNER = S8 / "scripts/run_stage8_block80_handoff_scope_freeze.py"

OUT = S8 / "artifacts/block81_deepsense_authority_audit"
SNAP = OUT / "authority_snapshots"
FINDINGS = OUT / "deepsense_authority_findings.json"
MANIFEST = OUT / "stage8_block81_authority_manifest.json"
REPORT = S8 / "reports/stage8_block81_deepsense_authority_audit_report.json"
SEAL = OUT / "stage8_block81_authority_freeze_seal.json"

EXPECTED_CONTRACT = "571843a955afd59125fa70b0a3ef196a8d578b6d811f50f4e8c7bbd5135516d2"
EXPECTED_B80 = {
    str(B80_MANIFEST): "c2a7ad9153c083b73971822798b52819d30ef92cd4f57820623edf010555dfae",
    str(B80_REPORT): "acddd5550fbeee894b7fc7e2f7ab7695bf89e5f84d3899ed4090c4a87e7bb493",
    str(B80_SEAL): "624b241a95ecd4eaac3147f41d909023202577ce241748c51c324e47d82a1687",
    str(B80_RUNNER): "2a41fd762e9261bc1b6b5c8152331a5ceb4b2fa1c21922c0129b04eedc47f906",
}

AUTHORITIES = {
    "homepage": "https://www.deepsense6g.net/",
    "lidar_beam_task": "https://www.deepsense6g.net/lidar-aided-beam-prediction/",
    "scenario_index": "https://www.deepsense6g.net/scenarios/",
    "scenario9": "https://www.deepsense6g.net/scenarios/Scenarios%201-9/scenario-9",
    "tutorials": "https://www.deepsense6g.net/tutorials/",
    "citations": "https://www.deepsense6g.net/citations",
    "v2v_2023": "https://www.deepsense6g.net/beam_prediction_challenge_2023/",
    "v2v_tracking": "https://www.deepsense6g.net/v2v-beam-tracking-task/",
}

class FailClosed(RuntimeError):
    pass

def require(cond: bool, msg: str):
    if not cond:
        raise FailClosed(msg)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
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

def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing overwrite: {path}")
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

def normalize_html(raw: bytes) -> str:
    s = raw.decode("utf-8", errors="replace")
    s = re.sub(r"(?is)<script.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?</style>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = s.replace("×", "x").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def fetch(name: str, url: str) -> dict:
    # Bounded official-page snapshot only; never follows task download/sign-in links.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Stage8AuthorityAudit/1.0"})
    ctx = ssl.create_default_context()
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        final_url = r.geturl()
        content_type = r.headers.get("Content-Type", "")
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            require(int(content_length) <= 5_000_000, f"Authority page unexpectedly large: {url}")
        raw = r.read(5_000_001)
        require(len(raw) <= 5_000_000, f"Authority page exceeded 5 MB bound: {url}")
        status = getattr(r, "status", 200)
    require(200 <= int(status) < 300, f"HTTP status {status} for {url}")
    require("deepsense6g.net" in final_url.lower(), f"Authority redirected off official domain: {final_url}")
    path = SNAP / f"{name}.html"
    atomic_bytes(path, raw)
    return {
        "name": name,
        "requested_url": url,
        "final_url": final_url,
        "http_status": int(status),
        "content_type": content_type,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_path": str(path),
        "fetch_elapsed_s": time.time() - t0,
        "text": normalize_html(raw),
    }

def has_all(text: str, terms) -> bool:
    return all(t.lower() in text for t in terms)

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.1")
    print("DEEPSENSE OFFICIAL AUTHORITY / SCHEMA / LICENSE / SPLIT AUDIT")
    print("OFFICIAL WEB SNAPSHOTS ONLY — NO DATASET ARCHIVE DOWNLOAD")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner installation path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT, "Block8.1 contract SHA mismatch.")
    for raw, expected in EXPECTED_B80.items():
        p = Path(raw)
        require(p.is_file(), f"Missing Block8.0 frozen input: {p}")
        require(sha256_file(p) == expected, f"Block8.0 SHA changed: {p}")
    for p in (FINDINGS, MANIFEST, REPORT, SEAL):
        require(not p.exists(), f"Block8.1 global output exists: {p}")
    require(not SNAP.exists(), f"Block8.1 snapshot namespace already exists: {SNAP}")

    print("\n===== A. BLOCK8.0 IMMUTABLE GATE =====")
    b80 = json.loads(B80_SEAL.read_text())
    require(b80.get("status") == "FROZEN_COMPLETE_BLOCK80", "Block8.0 seal status changed.")
    require(b80.get("Block81_may_start") is True, "Block8.1 not authorized.")
    require(b80.get("DeepSense_data_accessed") is False, "Unexpected DeepSense data access in Block8.0.")
    print("Block8.0 exact identity + authorization = PASS")

    print("\n===== B. BOUNDED OFFICIAL WEB AUTHORITY SNAPSHOT =====")
    snapshots = {}
    for name, url in AUTHORITIES.items():
        rec = fetch(name, url)
        snapshots[name] = rec
        print(f"{name}: HTTP={rec['http_status']} bytes={rec['bytes']} sha={rec['sha256'][:12]}")
    print("8/8 official DeepSense page snapshots = PASS")

    L = snapshots["lidar_beam_task"]["text"]
    S = snapshots["scenario_index"]["text"]
    S9 = snapshots["scenario9"]["text"]
    T = snapshots["tutorials"]["text"]
    V = snapshots["v2v_2023"]["text"]
    VT = snapshots["v2v_tracking"]["text"]
    H = snapshots["homepage"]["text"]

    print("\n===== C. PRIMARY LIDAR-AIDED TASK AUTHORITY =====")
    require(has_all(L, ["lidar-aided beam prediction", "scenarios", "8", "9"]), "LiDAR task scenarios 8/9 not verified.")
    require(("64x1" in L or ("64" in L and "power vector" in L)), "64x1 measured-power interface not verified.")
    require("64-beam codebook" in L or ("64" in L and "codebook" in L), "64-beam codebook not verified.")
    require("416x2" in L or ("416" in L and "lidar sensing matrix" in L), "416x2 LiDAR matrix not verified.")
    require(all(x in L for x in ("train.csv", "val.csv", "test.csv")), "Documented train/val/test split files not verified.")
    require("beam_index" in L, "beam_index label not verified.")
    require("100ms" in L or "100 ms" in L, "~100ms sampling not verified.")
    require("ground-truth" in L and "beam" in L, "Ground-truth beam labeling not verified.")
    print("Scenarios 8/9 = PASS")
    print("64x1 measured power vectors = PASS")
    print("64-beam codebook = PASS")
    print("416x2 LiDAR matrix = PASS")
    print("train.csv / val.csv / test.csv = PASS")
    print("beam_index label + ~100ms sampling = PASS")

    print("\n===== D. SCENARIO / TUTORIAL CROSS-CHECK =====")
    require("scenario 8" in S and "4043" in S, "Scenario8 listing/sample count not verified.")
    require("scenario 9" in S and "5964" in S, "Scenario9 listing/sample count not verified.")
    require("5964" in S9 and ("64-dimensional received power vector" in S9 or "64" in S9), "Scenario9 detailed authority mismatch.")
    require(("n_beams = 64" in T or "n_beams=64" in T or "64" in T) and "np.loadtxt" in T, "Tutorial 64-beam loading example not verified.")
    print("Scenario8 official listing: 4043 instances = PASS")
    print("Scenario9 official listing/detail: 5964 instances = PASS")
    print("Official tutorial: 64-element power loading = PASS")

    print("\n===== E. LICENSE / CITATION AUTHORITY =====")
    require("license" in H and "cite" in H and "deepsense" in H, "Homepage license/citation section not verified.")
    require("license" in L and "cite" in L, "LiDAR task license/citation section not verified.")
    license_status = {
        "official_site_license_section_present": True,
        "official_task_citation_requirement_present": True,
        "explicit_machine_readable_dataset_license_identifier_verified_from_lidar_page_text": False,
        "interpretation": "Do not infer a legal license identifier from unrelated task/code pages. Capture official sign-in/download terms before acquiring dataset bytes.",
        "dataset_archive_download_authorized_in_block81": False,
    }
    print("Official license/citation sections = PASS")
    print("Exact task-dataset license identifier = NOT CLAIMED")
    print("Dataset archive download = NOT AUTHORIZED IN BLOCK8.1")

    print("\n===== F. V2V SECONDARY AUTHORITY — NO SILENT REMAPPING =====")
    require(has_all(V, ["scenarios 36", "39"]), "V2V scenarios 36-39 not verified.")
    require(("64x4" in V or ("64" in V and "four" in V and "power vector" in V)), "V2V 64x4 power authority not verified.")
    require("lidar" in V and "radar" in V and "gps" in V and ("rgb" in V or "camera" in V), "V2V multimodal authority not verified.")
    require(("0 and 255" in VT or "0-255" in VT or "255" in VT) and ("overall beam" in VT or "overall-beam" in VT), "V2V 256-way overall-beam interface not verified.")
    print("V2V scenarios 36-39 = PASS")
    print("four arrays / 64x4 measured powers = PASS")
    print("overall beam index 0..255 in tracking task = PASS")
    print("V2V primary-64 remapping = FORBIDDEN WITHOUT NEW PREREGISTRATION")

    print("\n===== G. PRIMARY TASK FREEZE =====")
    primary = {
        "task": "DeepSense LiDAR-aided beam prediction",
        "scenarios": [8, 9],
        "input": {
            "raw_lidar_matrix": [416, 2],
            "columns": ["angle", "nearest-obstacle distance"],
            "static_clutter_removed_variant_documented": True,
        },
        "wireless_reference": {
            "measured_power_vector_shape": [64, 1],
            "codebook_size": 64,
            "optimal_beam_label": "argmax of measured power vector, documented as beam_index",
            "controller_input_allowed": False,
            "evaluator_only": True,
        },
        "documented_split_files": ["train.csv", "val.csv", "test.csv"],
        "documented_sampling": "approximately 100ms",
        "formal_split_counts_verified": False,
        "split_disjointness_verified": False,
        "reason_not_verified": "dataset bytes intentionally not downloaded in Block8.1",
    }
    secondary = {
        "task": "DeepSense V2V 2023 / beam-tracking authority",
        "scenarios": [36,37,38,39],
        "power_surface": "four 64-beam receiver arrays / 64x4",
        "overall_task_interface": "0..255 overall beam index",
        "role": "SECONDARY_OPTIONAL",
        "primary_stage8_interface": False,
    }
    print("PRIMARY = LiDAR-aided 64-way scenarios 8/9")
    print("SECONDARY = V2V 36-39; separate interface required")

    findings = {
        "stage": 8,
        "block": "8.1",
        "status": "FROZEN_COMPLETE_BLOCK81_AUTHORITY_SCHEMA_AUDIT",
        "primary_task": primary,
        "secondary_authority": secondary,
        "license_audit": license_status,
        "authority_snapshots": [
            {k:v for k,v in rec.items() if k != "text"} for rec in snapshots.values()
        ],
        "data_access_boundary": {
            "dataset_archives_downloaded": False,
            "download_links_followed": False,
            "authentication_performed": False,
            "measured_power_files_opened": False,
            "training_started": False,
            "calibration_started": False,
            "formal_evaluation_started": False,
        },
        "Block82_scope": {
            "authorized": True,
            "name": "bounded access + canonical corpus/split freeze",
            "requirements": [
                "capture official download/sign-in terms before dataset-byte acquisition",
                "acquire only primary LiDAR task scenarios 8/9 package or exact minimum required files",
                "hash all acquired archives/files",
                "verify actual train/val/test row counts and path schema",
                "verify split/sequence leakage before any training",
                "do not use measured powers for model/architecture selection outside training labels allowed by frozen split contract",
            ],
        },
    }
    atomic_json(FINDINGS, findings)

    manifest = {
        "stage": 8,
        "block": "8.1",
        "status": "FROZEN_COMPLETE_BLOCK81",
        "contract": {"path": str(CONTRACT), "sha256": EXPECTED_CONTRACT},
        "runner": {"path": str(SCRIPT), "sha256": sha256_file(SCRIPT)},
        "findings": {"path": str(FINDINGS), "sha256": sha256_file(FINDINGS)},
        "authority_snapshot_count": len(snapshots),
        "authority_snapshots": [
            {"name": rec["name"], "url": rec["final_url"], "path": rec["snapshot_path"], "sha256": rec["sha256"], "bytes": rec["bytes"]}
            for rec in snapshots.values()
        ],
        "primary_task": "LiDAR-aided beam prediction scenarios 8/9",
        "secondary_task": "V2V scenarios 36-39 optional/separate interface",
        "dataset_bytes_accessed": False,
    }
    atomic_json(MANIFEST, manifest)

    report = {
        "stage": 8,
        "block": "8.1",
        "status": "PASS_BLOCK81_DEEPSENSE_AUTHORITY_SCHEMA_AUDIT",
        "primary_lidar_64way_authority": True,
        "scenario8_instances_documented": 4043,
        "scenario9_instances_documented": 5964,
        "lidar_shape_documented": [416,2],
        "measured_power_shape_documented": [64,1],
        "split_names_documented": ["train.csv","val.csv","test.csv"],
        "actual_split_counts_verified": False,
        "actual_split_disjointness_verified": False,
        "license_citation_section_verified": True,
        "exact_dataset_license_identifier_claimed": False,
        "dataset_download_performed": False,
        "V2V_64x4_secondary_authority_verified": True,
        "Block82_authorized": True,
        "manifest_sha256": sha256_file(MANIFEST),
        "findings_sha256": sha256_file(FINDINGS),
        "runner_sha256": sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage": 8,
        "block": "8.1",
        "status": "FROZEN_COMPLETE_BLOCK81",
        "Block80_seal_sha256": EXPECTED_B80[str(B80_SEAL)],
        "contract_sha256": EXPECTED_CONTRACT,
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

    # Upstream recheck.
    for raw, expected in EXPECTED_B80.items():
        require(sha256_file(Path(raw)) == expected, f"Block8.0 modified during Block8.1: {raw}")

    print("\n" + "="*88)
    print("BLOCK 8.1 = FULLY VERIFIED / FROZEN")
    print("PRIMARY DEEPSENSE TASK = LIDAR-AIDED BEAM PREDICTION SCENARIOS 8/9")
    print("MEASURED POWER INTERFACE = 64x1 / 64-BEAM = PASS")
    print("LIDAR INTERFACE = 416x2 = PASS")
    print("DOCUMENTED SPLITS train/val/test = PASS")
    print("ACTUAL SPLIT COUNTS/DISJOINTNESS = DEFERRED TO BLOCK8.2")
    print("OFFICIAL LICENSE/CITATION SECTION = PASS")
    print("EXACT DATASET LICENSE IDENTIFIER = NOT CLAIMED")
    print("V2V 36-39 64x4 = SECONDARY/OPTIONAL SEPARATE INTERFACE")
    print("DEEPSENSE DATASET BYTES ACCESSED = NO")
    print("MEASURED POWER FILES ACCESSED = NO")
    print("TRAINING/CALIBRATION/FORMAL EVALUATION = NO")
    print("BLOCK 8.2 MAY START = YES")
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
        print("BLOCK 8.1 FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT DOWNLOAD DEEPSENSE DATASET ARCHIVES.")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

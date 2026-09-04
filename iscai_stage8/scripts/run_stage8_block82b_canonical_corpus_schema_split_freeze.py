#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path("/home/agni/waymo")
S8 = ROOT / "iscai_stage8"
ZIP_PATH = ROOT / "deepsense_incoming/block82_primary/LiDAR.zip"

SCRIPT = S8 / "scripts/run_stage8_block82b_canonical_corpus_schema_split_freeze.py"
CONTRACT = S8 / "configs/stage8_block82b_canonical_corpus_schema_split_contract.json"

B82A_CONTRACT = S8 / "configs/stage8_block82a_acquisition_intake_contract_final.json"
B82A_RUNNER = S8 / "scripts/run_stage8_block82a_acquisition_intake_freeze_final.py"
B82A_ROOT = S8 / "artifacts/block82a_deepsense_acquisition_intake_final"
B82A_MANIFEST = B82A_ROOT / "input_file_manifest.json"
B82A_INVENTORY = B82A_ROOT / "archive_member_inventory.json"
B82A_REPORT = S8 / "reports/stage8_block82a_acquisition_intake_final_report.json"
B82A_SEAL = B82A_ROOT / "stage8_block82a_acquisition_intake_seal.json"

DATA_ROOT = S8 / "data/block82b_deepsense_primary"
OUT = S8 / "artifacts/block82b_canonical_corpus_schema_split"
SCHEMA = OUT / "actual_csv_schema_and_counts.json"
REFS = OUT / "resolved_reference_integrity.json"
SPLITS = OUT / "frozen_split_roles_and_disjointness.json"
FILES = OUT / "canonical_extracted_file_manifest.json"
REPORT = S8 / "reports/stage8_block82b_canonical_corpus_schema_split_report.json"
SEAL = OUT / "stage8_block82b_canonical_corpus_schema_split_seal.json"

EXPECTED_CONTRACT = "41dcb27b652b86fa5a938b3e9e5c43a3511fec0d62b061236d7997dac272eb43"
EXPECTED_ZIP_SHA = "41e14ad39d368b73882f5a93ed4c29d0f86bb208fe628cd8e82248db66ec316d"
EXPECTED_ZIP_BYTES = 73692795
EXPECTED_B82A = {
    str(B82A_CONTRACT): "a4def4e27ade3d891903d8c4c5c81a861d25af3403659a86e194c126b0c86edc",
    str(B82A_RUNNER): "98ef3e132100e553deaef4306037c5abd7a3d6382d743ed47da8770a07c6dac2",
    str(B82A_MANIFEST): "901fa695ff4bc6e77012d542ade1bf4b5a7c04ee24f75624e129f5536cb14bec",
    str(B82A_INVENTORY): "202812900ad8d27009e7a8e563328cbb4a89a5151dd30070567d32a415c81359",
    str(B82A_REPORT): "f5808440ba6ffb497396c222f760b60cc847974719e093ce19c59b3c6d19fe9b",
    str(B82A_SEAL): "fcf2c3c5efb5c1626a301012d983c68e617a0088453d60c419a73dd404c7da2f",
}

CSV_SPECS = [
    (8, "TRAIN", "LiDAR/Scenario8/development_dataset/scenario8_dev_train.csv"),
    (8, "CALIBRATION", "LiDAR/Scenario8/development_dataset/scenario8_dev_val.csv"),
    (8, "FORMAL_EVALUATION", "LiDAR/Scenario8/development_dataset/scenario8_dev_test.csv"),
    (9, "TRAIN", "LiDAR/Scenario9/development_dataset/scenario9_dev_train.csv"),
    (9, "CALIBRATION", "LiDAR/Scenario9/development_dataset/scenario9_dev_val.csv"),
    (9, "FORMAL_EVALUATION", "LiDAR/Scenario9/development_dataset/scenario9_dev_test.csv"),
]
PREFIXES = (
    "LiDAR/Scenario8/development_dataset/",
    "LiDAR/Scenario9/development_dataset/",
)

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

def safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts

def extract_selected_atomic(z: zipfile.ZipFile, names):
    require(not DATA_ROOT.exists(), f"Canonical data root already exists: {DATA_ROOT}")
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=DATA_ROOT.name + ".tmp.", dir=str(DATA_ROOT.parent)))
    try:
        for name in names:
            require(safe_member(name), f"Unsafe member path: {name}")
            rel = PurePosixPath(name)
            target = tmp.joinpath(*rel.parts)
            info = z.getinfo(name)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
                dst.flush(); os.fsync(dst.fileno())
        os.replace(tmp, DATA_ROOT)
        dfd = os.open(DATA_ROOT.parent, os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

def normalize_ref(raw: str, sc: int) -> str | None:
    s = (raw or "").strip().strip('"').strip("'").replace("\\", "/")
    if not s:
        return None
    lo = s.lower()
    if not (lo.endswith(".mat") or lo.endswith(".txt")):
        return None
    while s.startswith("./"):
        s = s[2:]
    s = s.lstrip("/")
    if s.lower().startswith("lidar/"):
        return str(PurePosixPath(s))
    if "scenario8/" in s.lower() or "scenario9/" in s.lower():
        idx = s.lower().find(f"scenario{sc}/")
        if idx >= 0:
            return str(PurePosixPath("LiDAR") / PurePosixPath(s[idx:]))
    return str(PurePosixPath(f"LiDAR/Scenario{sc}/development_dataset") / PurePosixPath(s))

def choose_columns(headers, sample_rows):
    # Determine path columns from actual values, not guessed header spelling.
    lidar_cols, power_cols = [], []
    for h in headers:
        vals = [(r.get(h) or "").strip().lower().replace("\\", "/") for r in sample_rows]
        mat_hits = sum(v.endswith(".mat") and "lidar" in v for v in vals)
        pwr_hits = sum(v.endswith(".txt") and ("mmwave" in v or "power" in v) for v in vals)
        if mat_hits:
            lidar_cols.append(h)
        if pwr_hits:
            power_cols.append(h)
    beam_cols = [h for h in headers if "beam" in re.sub(r"[^a-z0-9]", "", h.lower())
                 and ("index" in re.sub(r"[^a-z0-9]", "", h.lower())
                      or re.sub(r"[^a-z0-9]", "", h.lower()).endswith("beam"))]
    # fallback exact-ish names
    if not beam_cols:
        beam_cols = [h for h in headers if "beam" in h.lower()]
    return lidar_cols, power_cols, beam_cols

def parse_csv_from_zip(z, member, sc, role, all_members):
    raw = z.read(member)
    text = raw.decode("utf-8-sig", errors="strict")
    rdr = csv.DictReader(io.StringIO(text))
    headers = list(rdr.fieldnames or [])
    require(headers, f"No CSV header: {member}")
    rows = list(rdr)
    require(rows, f"Empty CSV: {member}")

    probe_rows = rows[:min(200, len(rows))]
    lidar_cols, power_cols, beam_cols = choose_columns(headers, probe_rows)
    require(lidar_cols, f"No LiDAR path column detected in {member} headers={headers}")
    require(len(power_cols) == 1, f"Expected exactly one mmWave power path column in {member}, found {power_cols}")
    require(beam_cols, f"No beam label column detected in {member} headers={headers}")
    beam_col = beam_cols[0]

    lidar_refs, power_refs = [], []
    missing_lidar, missing_power = [], []
    beam_min = beam_max = None

    for i, row in enumerate(rows, start=2):
        lrefs = []
        for c in lidar_cols:
            nr = normalize_ref(row.get(c, ""), sc)
            if nr:
                lrefs.append(nr)
        require(lrefs, f"Row {i} has no LiDAR reference in {member}")
        pref = normalize_ref(row.get(power_cols[0], ""), sc)
        require(pref is not None, f"Row {i} has no power reference in {member}")

        # Prefer raw lidar_data over SCR/preprocessed when multiple are present, but retain all.
        lidar_refs.extend(lrefs)
        power_refs.append(pref)
        for r in lrefs:
            if r not in all_members:
                missing_lidar.append((i, r))
        if pref not in all_members:
            missing_power.append((i, pref))

        # Outcome boundary: validate labels only for train/calibration. Do not inspect/report formal values.
        if role != "FORMAL_EVALUATION":
            v = (row.get(beam_col) or "").strip()
            require(v != "", f"Missing beam label at {member} row {i}")
            try:
                x = int(float(v))
            except Exception:
                raise FailClosed(f"Non-integer beam label at {member} row {i}: {v!r}")
            require(0 <= x <= 63, f"Beam label out of 0..63 at {member} row {i}: {x}")
            beam_min = x if beam_min is None else min(beam_min, x)
            beam_max = x if beam_max is None else max(beam_max, x)

    require(not missing_lidar, f"Missing LiDAR refs in {member}; first={missing_lidar[:3]}")
    require(not missing_power, f"Missing power refs in {member}; first={missing_power[:3]}")

    # Explicit grouping/sequence columns, if present.
    group_cols = [h for h in headers if any(tok in h.lower() for tok in ("sequence", "seq_id", "sequence_id", "group_id", "scene_id"))]
    groups = {}
    for c in group_cols:
        vals = {(r.get(c) or "").strip() for r in rows if (r.get(c) or "").strip()}
        groups[c] = sorted(vals)

    return {
        "member": member,
        "scenario": sc,
        "role": role,
        "headers": headers,
        "row_count": len(rows),
        "lidar_path_columns": lidar_cols,
        "power_path_column": power_cols[0],
        "beam_label_column": beam_col,
        "beam_label_range_checked": role != "FORMAL_EVALUATION",
        "beam_label_min": beam_min,
        "beam_label_max": beam_max,
        "formal_beam_label_values_not_inspected_or_reported": role == "FORMAL_EVALUATION",
        "unique_lidar_refs": sorted(set(lidar_refs)),
        "unique_power_refs": sorted(set(power_refs)),
        "duplicate_lidar_ref_count_within_split": len(lidar_refs) - len(set(lidar_refs)),
        "duplicate_power_ref_count_within_split": len(power_refs) - len(set(power_refs)),
        "explicit_group_columns": group_cols,
        "explicit_groups": groups,
    }

def parse_power_vector_structural(path: Path):
    # TRAIN-only structural probe. Values are not retained.
    toks = path.read_text(encoding="utf-8", errors="strict").replace(",", " ").split()
    vals = [float(x) for x in toks]
    require(len(vals) == 64, f"Expected 64 power values in structural TRAIN probe {path}, got {len(vals)}")
    require(all(math.isfinite(x) for x in vals), f"Non-finite TRAIN power structural probe: {path}")
    return {"path": str(path), "numeric_length": 64, "all_finite": True, "values_recorded": False}

def file_row(p: Path):
    return {
        "relative_path": str(p.relative_to(DATA_ROOT)),
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }

def main():
    print("="*88)
    print("STAGE 8 — BLOCK 8.2B")
    print("CANONICAL DEEPSENSE CORPUS + ACTUAL CSV SCHEMA + SPLIT/DISJOINTNESS FREEZE")
    print("NO TRAINING / NO CALIBRATOR FIT / NO FORMAL POWER-VECTOR DECODING")
    print("="*88)

    require(Path(__file__).resolve() == SCRIPT.resolve(), "Runner path mismatch.")
    require(CONTRACT.is_file() and sha256_file(CONTRACT) == EXPECTED_CONTRACT, "8.2B contract SHA mismatch.")
    for raw, expected in EXPECTED_B82A.items():
        p = Path(raw)
        require(p.is_file(), f"Missing frozen 8.2A artifact: {p}")
        require(sha256_file(p) == expected, f"8.2A SHA changed: {p}")

    require(ZIP_PATH.is_file(), f"LiDAR.zip missing: {ZIP_PATH}")
    require(ZIP_PATH.stat().st_size == EXPECTED_ZIP_BYTES, "LiDAR.zip byte size changed.")
    require(sha256_file(ZIP_PATH) == EXPECTED_ZIP_SHA, "LiDAR.zip SHA changed.")

    for p in (DATA_ROOT, SCHEMA, REFS, SPLITS, FILES, REPORT, SEAL):
        require(not p.exists(), f"8.2B output already exists: {p}")

    print("\n===== A. FROZEN INPUT GATE =====")
    print("Block8.2A exact identities = PASS")
    print("LiDAR.zip SHA/bytes = PASS")

    print("\n===== B. REAL ARCHIVE STRUCTURE GATE =====")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        bad = z.testzip()
        require(bad is None, f"ZIP CRC failure at member {bad}")
        names = z.namelist()
        all_members = set(names)
        for _,_,member in CSV_SPECS:
            require(member in all_members, f"Required development CSV missing: {member}")
        selected = [n for n in names if any(n.startswith(p) for p in PREFIXES)]
        require(selected, "No development_dataset members selected.")
        require(not any("/challenge_dataset/" in n for n in selected), "Challenge member leaked into selected extraction.")
        print("6/6 real development CSV names = PASS")
        print("Scenario8/9 development_dataset selection = PASS")
        print("challenge_dataset selected = NO")

        print("\n===== C. CSV SCHEMA / REFERENCE / SPLIT AUDIT BEFORE EXTRACTION =====")
        parsed = []
        for sc, role, member in CSV_SPECS:
            rec = parse_csv_from_zip(z, member, sc, role, all_members)
            parsed.append(rec)
            print(
                f"scenario={sc} role={role} rows={rec['row_count']} "
                f"lidar_cols={rec['lidar_path_columns']} "
                f"power_col={rec['power_path_column']} beam_col={rec['beam_label_column']}"
            )

        # Exact path-level disjointness across roles, within and across scenarios.
        role_lidar = {"TRAIN":set(),"CALIBRATION":set(),"FORMAL_EVALUATION":set()}
        role_power = {"TRAIN":set(),"CALIBRATION":set(),"FORMAL_EVALUATION":set()}
        for rec in parsed:
            role_lidar[rec["role"]].update(rec["unique_lidar_refs"])
            role_power[rec["role"]].update(rec["unique_power_refs"])

        pairs = [("TRAIN","CALIBRATION"),("TRAIN","FORMAL_EVALUATION"),("CALIBRATION","FORMAL_EVALUATION")]
        overlaps = {}
        for a,b in pairs:
            kl = f"{a}__{b}"
            lo = sorted(role_lidar[a] & role_lidar[b])
            po = sorted(role_power[a] & role_power[b])
            overlaps[kl] = {
                "lidar_overlap_count": len(lo),
                "power_overlap_count": len(po),
                "lidar_overlap_examples": lo[:5],
                "power_overlap_examples": po[:5],
            }
            require(not lo, f"LiDAR reference leakage {a} vs {b}: {lo[:3]}")
            require(not po, f"Power reference leakage {a} vs {b}: {po[:3]}")

        # Explicit sequence/group leakage where schema supports it.
        explicit_group_cols = sorted(set(c for rec in parsed for c in rec["explicit_group_columns"]))
        group_audit = {"status": "NOT_EVALUABLE_FROM_AVAILABLE_CSV_SCHEMA", "columns": []}
        if explicit_group_cols:
            group_audit = {"status": "PASS_ZERO_EXPLICIT_GROUP_OVERLAP", "columns": []}
            for c in explicit_group_cols:
                role_sets = {}
                for role in role_lidar:
                    vals = set()
                    for rec in parsed:
                        if rec["role"] == role:
                            vals.update(rec["explicit_groups"].get(c, []))
                    role_sets[role] = vals
                colrec = {"column": c, "overlaps": {}}
                for a,b in pairs:
                    ov = sorted(role_sets[a] & role_sets[b])
                    colrec["overlaps"][f"{a}__{b}"] = len(ov)
                    require(not ov, f"Explicit group leakage column={c} {a} vs {b}: {ov[:3]}")
                group_audit["columns"].append(colrec)

        print("exact LiDAR reference overlap across split roles = 0 PASS")
        print("exact mmWave reference overlap across split roles = 0 PASS")
        print("explicit sequence/group leakage =", group_audit["status"])

        print("\n===== D. ATOMIC CANONICAL EXTRACTION =====")
        extract_selected_atomic(z, selected)

    # Full extracted file hash manifest (parallel read-only hashing).
    extracted = sorted(p for p in DATA_ROOT.rglob("*") if p.is_file())
    require(extracted, "Canonical extraction produced no files.")
    workers = max(1, min(os.cpu_count() or 1, 32))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        file_rows = list(ex.map(file_row, extracted))
    print("canonical extracted files =", len(file_rows))
    print("parallel hash workers =", workers)

    print("\n===== E. TRAIN-ONLY STRUCTURAL 64-WAY POWER PROBE =====")
    train_probes = []
    for sc in (8,9):
        rec = next(r for r in parsed if r["scenario"] == sc and r["role"] == "TRAIN")
        require(rec["unique_power_refs"], f"No TRAIN power refs scenario {sc}")
        member = rec["unique_power_refs"][0]
        local = DATA_ROOT.joinpath(*PurePosixPath(member).parts)
        require(local.is_file(), f"Extracted TRAIN power probe missing: {local}")
        probe = parse_power_vector_structural(local)
        probe["scenario"] = sc
        train_probes.append(probe)
        print(f"scenario{sc} TRAIN power vector numeric length = 64 PASS")
    print("CALIBRATION power vectors decoded = NO")
    print("FORMAL power vectors decoded = NO")

    # Freeze compact schema artifact; do not expose thousands of refs here.
    schema_doc = {
        "stage":8, "block":"8.2B", "status":"FROZEN_ACTUAL_CSV_SCHEMA",
        "splits":[{
            k:v for k,v in rec.items()
            if k not in ("unique_lidar_refs","unique_power_refs","explicit_groups")
        } for rec in parsed],
        "structural_train_power_probes": train_probes,
        "formal_power_numeric_vectors_decoded": False,
        "calibration_power_numeric_vectors_decoded": False,
    }
    atomic_json(SCHEMA, schema_doc)

    ref_doc = {
        "stage":8, "block":"8.2B", "status":"FROZEN_REFERENCE_INTEGRITY",
        "splits":[{
            "scenario":rec["scenario"],"role":rec["role"],"member":rec["member"],
            "row_count":rec["row_count"],
            "unique_lidar_ref_count":len(rec["unique_lidar_refs"]),
            "unique_power_ref_count":len(rec["unique_power_refs"]),
            "duplicate_lidar_ref_count_within_split":rec["duplicate_lidar_ref_count_within_split"],
            "duplicate_power_ref_count_within_split":rec["duplicate_power_ref_count_within_split"],
            "all_lidar_refs_exist":True,"all_power_refs_exist":True,
        } for rec in parsed],
        "cross_role_overlaps": overlaps,
    }
    atomic_json(REFS, ref_doc)

    split_doc = {
        "stage":8, "block":"8.2B", "status":"FROZEN_SPLIT_ROLES_AND_DISJOINTNESS",
        "role_mapping": {
            "official_dev_train":"TRAIN",
            "official_dev_val":"CALIBRATION",
            "official_dev_test":"FORMAL_EVALUATION",
            "challenge_dataset":"EXCLUDED_FROM_LOCAL_CORE_HIDDEN_GT_SUBMISSION_ONLY",
        },
        "exact_lidar_reference_disjointness":"PASS_ZERO_OVERLAP",
        "exact_mmwave_reference_disjointness":"PASS_ZERO_OVERLAP",
        "explicit_sequence_group_audit":group_audit,
        "formal_test_used_for_model_selection":False,
        "formal_test_power_vectors_decoded":False,
    }
    atomic_json(SPLITS, split_doc)

    atomic_json(FILES, {
        "stage":8, "block":"8.2B", "status":"FROZEN_CANONICAL_EXTRACTED_FILE_MANIFEST",
        "source_zip":{"path":str(ZIP_PATH),"bytes":EXPECTED_ZIP_BYTES,"sha256":EXPECTED_ZIP_SHA},
        "data_root":str(DATA_ROOT),
        "file_count":len(file_rows),
        "files":file_rows,
    })

    report = {
        "stage":8, "block":"8.2B",
        "status":"PASS_BLOCK82B_CANONICAL_CORPUS_SCHEMA_SPLIT_FREEZE",
        "six_development_csvs_verified":True,
        "actual_schema_frozen":True,
        "all_referenced_lidar_paths_exist":True,
        "all_referenced_mmwave_paths_exist":True,
        "exact_cross_role_lidar_overlap":0,
        "exact_cross_role_mmwave_overlap":0,
        "sequence_group_audit_status":group_audit["status"],
        "canonical_extraction_complete":True,
        "challenge_dataset_extracted":False,
        "train_power_structural_probe_count":2,
        "train_power_vector_length":64,
        "calibration_power_vectors_decoded":False,
        "formal_power_vectors_decoded":False,
        "training_started":False,
        "calibrator_fit_started":False,
        "formal_metrics_computed":False,
        "Block83_authorized":True,
        "schema_sha256":sha256_file(SCHEMA),
        "references_sha256":sha256_file(REFS),
        "splits_sha256":sha256_file(SPLITS),
        "files_sha256":sha256_file(FILES),
        "runner_sha256":sha256_file(SCRIPT),
    }
    atomic_json(REPORT, report)

    seal = {
        "stage":8, "block":"8.2B", "status":"FROZEN_COMPLETE_BLOCK82B",
        "Block82A_seal_sha256":EXPECTED_B82A[str(B82A_SEAL)],
        "LiDAR_zip_sha256":EXPECTED_ZIP_SHA,
        "contract_sha256":EXPECTED_CONTRACT,
        "runner_sha256":sha256_file(SCRIPT),
        "schema_sha256":sha256_file(SCHEMA),
        "references_sha256":sha256_file(REFS),
        "splits_sha256":sha256_file(SPLITS),
        "canonical_files_manifest_sha256":sha256_file(FILES),
        "report_sha256":sha256_file(REPORT),
        "formal_power_vectors_decoded":False,
        "training_started":False,
        "Block83_may_start":True,
    }
    atomic_json(SEAL, seal)

    # Post-publication immutability.
    require(sha256_file(ZIP_PATH) == EXPECTED_ZIP_SHA, "LiDAR.zip modified during 8.2B.")
    for raw,expected in EXPECTED_B82A.items():
        require(sha256_file(Path(raw)) == expected, f"8.2A modified during 8.2B: {raw}")

    print("\n" + "="*88)
    print("BLOCK 8.2B = FULLY VERIFIED / FROZEN")
    print("REAL Scenario8/9 development CSV SCHEMA = PASS")
    print("CANONICAL development_dataset EXTRACTION = PASS")
    print("TRAIN = official dev_train")
    print("CALIBRATION = official dev_val")
    print("FORMAL_EVALUATION = official dev_test")
    print("CHALLENGE DATASET = EXCLUDED FROM LOCAL CORE")
    print("ALL LiDAR/mmWave REFERENCES RESOLVE = PASS")
    print("EXACT TRAIN/CALIBRATION/FORMAL SAMPLE OVERLAP = 0 PASS")
    print("SEQUENCE/GROUP LEAKAGE =", group_audit["status"])
    print("TRAIN STRUCTURAL POWER PROBES = 2 x 64 PASS")
    print("CALIBRATION POWER VECTORS DECODED = NO")
    print("FORMAL POWER VECTORS DECODED = NO")
    print("TRAINING/CALIBRATOR FIT/FORMAL METRICS = NO")
    print("BLOCK 8.3 MAY START = YES")
    print("8.2B schema SHA256 =", sha256_file(SCHEMA))
    print("8.2B refs SHA256   =", sha256_file(REFS))
    print("8.2B splits SHA256 =", sha256_file(SPLITS))
    print("8.2B files SHA256  =", sha256_file(FILES))
    print("8.2B report SHA256 =", sha256_file(REPORT))
    print("8.2B seal SHA256   =", sha256_file(SEAL))
    print("8.2B runner SHA256 =", sha256_file(SCRIPT))
    print("STATUS = FROZEN_COMPLETE_BLOCK82B")
    print("="*88)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!"*88)
        print("BLOCK 8.2B FAIL-CLOSED")
        print(f"{type(exc).__name__}: {exc}")
        print("DO NOT DELETE/MODIFY LiDAR.zip OR ANY 8.2A ARTIFACT.")
        print("DO NOT START TRAINING/CALIBRATION/FORMAL EVALUATION.")
        print("!"*88)
        raise

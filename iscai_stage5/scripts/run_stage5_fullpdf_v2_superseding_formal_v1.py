from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

PYTHON = Path(
    "/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python"
)

CHECKER2 = (
    S5 / "scripts/check_stage5_fullpdf_v2.py"
)

FORMAL_CHECKER = (
    S5
    / "scripts/run_stage5_fullpdf_v2_independent_formal_checker.py"
)

RUNTIME = (
    S5 / "src/iscai_stage5/fullpdf_v2.py"
)

PROTOCOL = (
    S5 / "configs/stage5_fullpdf_v2_protocol.json"
)

ADDENDUM = (
    S5
    / "configs/stage5_fullpdf_v2_preformal_addendum.json"
)

LEDGER = (
    S4
    / "artifacts/fullpdf_v2/formal_prediction_ledger_run1.jsonl"
)

BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
    "primary_receiver_binding_fullpdf_v2.jsonl"
)

AMENDMENT = (
    S5
    / "configs/"
    "stage5_fullpdf_v2_receiver_binding_amendment_v1.json"
)

REPAIR_REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_receiver_binding_materialization_v1.json"
)

SUPERSEDING_SEAL = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "stage5_fullpdf_v2_receiver_binding_"
    "superseding_pretruth_seal_v1.json"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

MARKER = (
    ART
    / "superseding_formal_receiver_binding_v1_invocation.json"
)

LOG = (
    ART
    / "checker2_superseding_receiver_binding_v1.log"
)

EXPECTED = {
    RUNTIME:
        "e57708e9332bb40e84f18270e483d91165e26c149b80e82c8f10b4df44376249",
    PROTOCOL:
        "91dec3cc3d3a7b72385f012d1d2bfdb8981a497e1344f585d38f3e05f2e4fa5a",
    ADDENDUM:
        "570fbada05c8e8ed71775bbb6eb668f4016dd65c98b09b4f158abeb379eca6aa",
    LEDGER:
        "2dd3c396d5c365677c41b3a0df6344a7c28d423daaa368dbe466bd4b17b4c2a8",
    BINDING:
        "5a5516ee050d2fd1c0a571193007650e75f23836b2734ee3df1950cc04f2e1c8",
    AMENDMENT:
        "3654fd3334076eb1649b74db46ffeb8c2065ed1bc9a70d6e011939edfebe6355",
    SUPERSEDING_SEAL:
        "b2c2035ee7839ceef3d8bf6039d5d181c993b5fac32113eca3df2723ef6c7c65",
    REPAIR_REPORT:
        "a7f25e073be25f32c00f89934c3452f19d06b53ef934c9930b3410ed39791727",
}


def fail(msg: str) -> None:
    raise SystemExit(
        "FAIL-CLOSED BEFORE FORMAL: " + msg
    )


def sha(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_once_json(
    path: Path,
    obj: dict,
) -> None:
    data = (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    if path.exists():
        fail(
            "superseding formal V1 "
            "has already been invoked; "
            f"marker exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )
    os.chmod(
        path,
        0o444,
    )


print("=" * 79)
print(
    "STAGE5 FULLPDF V2 — "
    "SUPERSEDING FORMAL V1"
)
print(
    "ONE SHOT | NO RETRAINING | "
    "NO RECALIBRATION | NO TUNING"
)
print("=" * 79)

for path in (
    PYTHON,
    CHECKER2,
    FORMAL_CHECKER,
    *EXPECTED.keys(),
):
    if not path.is_file():
        fail(
            f"missing required file: {path}"
        )

for path, expected in EXPECTED.items():
    actual = sha(path)

    if actual != expected:
        fail(
            "SHA mismatch:\n"
            f"{path}\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )

    print(
        "PASS | sealed SHA |",
        path.name,
    )

free_bytes = shutil.disk_usage(
    ROOT
).free

minimum = 250 * 1024**3

print(
    "free_GiB =",
    free_bytes / 1024**3,
)

if free_bytes < minimum:
    fail(
        "free-space reserve below "
        "250 GiB"
    )

print(
    "PASS | >=250 GiB free-space reserve"
)

history_root = (
    ART
    / "receiver_binding_failed_attempt_history"
)

history_manifests = sorted(
    history_root.glob(
        "*/history_manifest.json"
    )
)

if not history_manifests:
    fail(
        "no preserved prior failed-attempt "
        "history manifest"
    )

print(
    "PASS | prior failed attempt history preserved"
)
print(
    "history_manifests =",
    len(history_manifests),
)

checker2_before = sha(
    CHECKER2
)
formal_checker_before = sha(
    FORMAL_CHECKER
)

marker = {
    "stage": 5,
    "version": "fullpdf_v2",
    "attempt":
        "superseding_receiver_binding_v1",
    "status":
        "FORMAL_INVOCATION_STARTED",
    "started_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
    "one_shot":
        True,
    "Stage6_started_by_wrapper":
        False,
    "binding": {
        "path":
            str(BINDING),
        "sha256":
            sha(BINDING),
        "represented_scenes":
            120,
    },
    "receiver_binding_amendment_sha256":
        sha(AMENDMENT),
    "superseding_pretruth_seal_sha256":
        sha(SUPERSEDING_SEAL),
    "repair_report_sha256":
        sha(REPAIR_REPORT),
    "frozen": {
        "runtime_sha256":
            sha(RUNTIME),
        "protocol_sha256":
            sha(PROTOCOL),
        "addendum_sha256":
            sha(ADDENDUM),
        "Stage4_ledger_sha256":
            sha(LEDGER),
        "checker2_sha256":
            checker2_before,
        "formal_checker_sha256":
            formal_checker_before,
    },
    "scientific_boundary": {
        "retraining":
            False,
        "recalibration":
            False,
        "formal_tuning":
            False,
        "post_outcome_tuning":
            False,
    },
}

write_once_json(
    MARKER,
    marker,
)

print(
    "PASS | immutable one-shot "
    "invocation marker written"
)
print(
    "marker_sha256 =",
    sha(MARKER),
)

if LOG.exists():
    fail(
        "superseding V1 log already exists; "
        "will not overwrite"
    )

print()
print("=" * 79)
print(
    "EXECUTING CHECKER2 EXACTLY ONCE"
)
print("=" * 79)
sys.stdout.flush()

proc = subprocess.run(
    [
        str(PYTHON),
        str(CHECKER2),
    ],
    cwd=str(ROOT),
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)

LOG.write_text(
    proc.stdout,
    encoding="utf-8",
)

os.chmod(
    LOG,
    0o444,
)

# Show the checker output exactly once in terminal.
print(
    proc.stdout,
    end="",
)

print()
print("=" * 79)
print(
    "SUPERSEDING FORMAL PROCESS FINISHED"
)
print(
    "checker2_returncode =",
    proc.returncode,
)
print(
    "checker2_log =",
    LOG,
)
print(
    "checker2_log_sha256 =",
    sha(LOG),
)

if sha(CHECKER2) != checker2_before:
    fail(
        "Checker2 changed during formal run"
    )

if (
    sha(FORMAL_CHECKER)
    != formal_checker_before
):
    fail(
        "formal checker changed during formal run"
    )

for path, expected in EXPECTED.items():
    if sha(path) != expected:
        fail(
            "sealed preformal/frozen artifact "
            f"changed during formal: {path}"
        )

print(
    "PASS | all sealed/frozen "
    "preformal artifacts unchanged"
)
print(
    "Stage6 was NOT started by this wrapper."
)
print("=" * 79)

# Preserve Checker2's scientific return code.
raise SystemExit(
    proc.returncode
)

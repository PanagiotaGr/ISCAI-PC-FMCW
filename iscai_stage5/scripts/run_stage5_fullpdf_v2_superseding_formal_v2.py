from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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

BINDING_AMENDMENT = (
    S5
    / "configs/"
    "stage5_fullpdf_v2_receiver_binding_amendment_v1.json"
)

BINDING_SEAL = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "stage5_fullpdf_v2_receiver_binding_"
    "superseding_pretruth_seal_v1.json"
)

CHECKER_REPAIR_AMENDMENT = (
    S5
    / "configs/"
    "stage5_fullpdf_v2_checker_probeplan_property_amendment_v1.json"
)

CHECKER_REPAIR_SEAL = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "probeplan_property_repair_v1/"
    "stage5_fullpdf_v2_checker_probeplan_property_pretruth_seal_v1.json"
)

CHECKER_REPAIR_REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_checker_probeplan_property_repair_v1.json"
)

V1_MARKER = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "superseding_formal_receiver_binding_v1_invocation.json"
)

V1_WRAPPER_LOG = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "checker2_superseding_receiver_binding_v1.log"
)

FORMAL_REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_independent_formal_checker.json"
)

CHECKER2_REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_checker2_gate.json"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

HISTORY = (
    ART
    / "superseding_formal_v2_pre_run_history"
)

MARKER = (
    ART
    / "superseding_formal_receiver_binding_checkerfix_v2_invocation.json"
)

LOG = (
    ART
    / "checker2_superseding_receiver_binding_checkerfix_v2.log"
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

    BINDING_AMENDMENT:
        "3654fd3334076eb1649b74db46ffeb8c2065ed1bc9a70d6e011939edfebe6355",

    BINDING_SEAL:
        "b2c2035ee7839ceef3d8bf6039d5d181c993b5fac32113eca3df2723ef6c7c65",

    FORMAL_CHECKER:
        "342344902e6b00cb841036dcdc0e54f1e1ba396344fa602864521890f5eb5da5",

    CHECKER_REPAIR_AMENDMENT:
        "d47ac985089af164ebca1499112e237784131db069b704ac31f5f84c13f52e77",

    CHECKER_REPAIR_SEAL:
        "1da2f7f5731c5574c3c4fcb0ec2a216296f338fd1cc5adcfbd2125bde85b3548",

    CHECKER_REPAIR_REPORT:
        "d79ea27a7716edc7d0eba5f13a56f4b16cdcfd78c77536825e9937ee5a2259f8",

    V1_MARKER:
        "768b68b2bafc7152b79e051d8fa08582877625b2d62482bb16e4c69bb5eabfe6",

    V1_WRAPPER_LOG:
        "a28e060a5b2f54dbabe8b3453fb5c1b65ba5c18fd5c3fd89e9c3a7eff1031b9c",
}


def fail(msg: str) -> None:
    raise SystemExit(
        "FAIL-CLOSED BEFORE V2 FORMAL: "
        + str(msg)
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


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    if not isinstance(
        obj,
        dict,
    ):
        fail(
            f"JSON root not object: {path}"
        )

    return obj


def canonical_json(obj: dict) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once_json(
    path: Path,
    obj: dict,
) -> str:
    if path.exists():
        fail(
            "V2 invocation already exists; "
            f"do not rerun: {path}"
        )

    data = canonical_json(
        obj
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
        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )

    os.chmod(
        path,
        0o444,
    )

    return sha(
        path
    )


print("=" * 79)
print(
    "STAGE5 FULLPDF V2 — "
    "SUPERSEDING FORMAL V2"
)
print(
    "ONE SHOT AFTER CHECKER API REPAIR | "
    "NO RETRAINING | NO RECALIBRATION | NO TUNING"
)
print("=" * 79)

for path in (
    PYTHON,
    CHECKER2,
    FORMAL_REPORT,
    CHECKER2_REPORT,
    *EXPECTED.keys(),
):
    if not path.is_file():
        fail(
            f"missing required file: {path}"
        )

# ------------------------------------------------------------
# A. Exact immutable preformal chain.
# ------------------------------------------------------------

for path, expected in (
    EXPECTED.items()
):
    actual = sha(
        path
    )

    if actual != expected:
        fail(
            "sealed SHA mismatch:\n"
            f"path     = {path}\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )

    print(
        "PASS | sealed SHA |",
        path.name,
    )

# Checker2 itself was not changed by the checker repair.
v1_marker = load_json(
    V1_MARKER
)

old_checker2_sha = (
    v1_marker
    .get(
        "frozen",
        {},
    )
    .get(
        "checker2_sha256"
    )
)

if not old_checker2_sha:
    fail(
        "V1 marker does not bind Checker2"
    )

current_checker2_sha = sha(
    CHECKER2
)

if (
    current_checker2_sha
    != old_checker2_sha
):
    fail(
        "Checker2 changed since V1:\n"
        f"V1      = {old_checker2_sha}\n"
        f"current = {current_checker2_sha}"
    )

print(
    "PASS | Checker2 unchanged since V1"
)
print(
    "checker2_sha256 =",
    current_checker2_sha,
)

# ------------------------------------------------------------
# B. Disk reserve.
# ------------------------------------------------------------

free_bytes = shutil.disk_usage(
    ROOT
).free

free_gib = (
    free_bytes
    / 1024**3
)

print(
    "free_GiB =",
    free_gib,
)

if (
    free_bytes
    < 250 * 1024**3
):
    fail(
        "free-space reserve below 250 GiB"
    )

print(
    "PASS | >=250 GiB free-space reserve"
)

# ------------------------------------------------------------
# C. Preserve live V1/checker-repair state before V2 overwrite.
# ------------------------------------------------------------

if HISTORY.exists():
    fail(
        "V2 pre-run history already exists; "
        "will not overwrite"
    )

HISTORY.mkdir(
    parents=True,
    exist_ok=False,
)

history_entries = []

sources = [
    V1_MARKER,
    V1_WRAPPER_LOG,
    FORMAL_REPORT,
    CHECKER2_REPORT,
    FORMAL_CHECKER,
    CHECKER_REPAIR_AMENDMENT,
    CHECKER_REPAIR_SEAL,
    CHECKER_REPAIR_REPORT,
]

# Also preserve every current top-level formal artifact that
# Checker2/formal checker may overwrite during V2.
for p in sorted(
    ART.iterdir()
):
    if (
        p.is_file()
        and p not in sources
        and p != MARKER
        and p != LOG
        and p.suffix.lower()
        in {
            ".json",
            ".jsonl",
            ".log",
            ".txt",
        }
        and p.stat().st_size
        <= 100 * 1024 * 1024
    ):
        sources.append(
            p
        )

seen = set()

for index, src in enumerate(
    sources
):
    resolved = (
        src.resolve()
    )

    if resolved in seen:
        continue

    seen.add(
        resolved
    )

    dst = (
        HISTORY
        / (
            f"{index:03d}_"
            + src.name
        )
    )

    shutil.copy2(
        src,
        dst,
    )

    history_entries.append(
        {
            "source":
                str(src),
            "source_sha256":
                sha(src),
            "snapshot":
                str(dst),
            "snapshot_sha256":
                sha(dst),
        }
    )

history_manifest = (
    HISTORY
    / "history_manifest.json"
)

history_payload = {
    "stage": 5,
    "version": "fullpdf_v2",
    "status":
        "PRESERVED_BEFORE_SUPERSEDING_FORMAL_V2",
    "previous_attempt":
        "superseding_receiver_binding_v1",
    "previous_failure":
        "pretruth formal-checker API exception",
    "previous_scientific_outcome_computed":
        False,
    "checker_repair":
        "ProbePlan.probing_beam_count "
        "call -> property access",
    "files":
        history_entries,
}

history_manifest.write_bytes(
    canonical_json(
        history_payload
    )
)

os.chmod(
    history_manifest,
    0o444,
)

history_manifest_sha = sha(
    history_manifest
)

print(
    "PASS | V1/checker-repair state preserved"
)
print(
    "history_manifest_sha256 =",
    history_manifest_sha,
)

# ------------------------------------------------------------
# D. One-shot V2 invocation seal.
# ------------------------------------------------------------

if LOG.exists():
    fail(
        "V2 formal log already exists; "
        "do not overwrite"
    )

before_hashes = {
    "checker2":
        sha(CHECKER2),
    "formal_checker":
        sha(FORMAL_CHECKER),
    "runtime":
        sha(RUNTIME),
    "protocol":
        sha(PROTOCOL),
    "addendum":
        sha(ADDENDUM),
    "Stage4_ledger":
        sha(LEDGER),
    "receiver_binding":
        sha(BINDING),
    "binding_amendment":
        sha(BINDING_AMENDMENT),
    "binding_seal":
        sha(BINDING_SEAL),
    "checker_repair_amendment":
        sha(CHECKER_REPAIR_AMENDMENT),
    "checker_repair_seal":
        sha(CHECKER_REPAIR_SEAL),
    "checker_repair_report":
        sha(CHECKER_REPAIR_REPORT),
}

marker_payload = {
    "stage": 5,
    "version": "fullpdf_v2",
    "attempt":
        "superseding_receiver_binding_checkerfix_v2",
    "status":
        "FORMAL_INVOCATION_STARTED",
    "started_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),
    "one_shot":
        True,
    "previous_V1_preserved":
        True,
    "previous_V1_scientific_outcome":
        "NOT_COMPUTED",
    "receiver_binding": {
        "sha256":
            before_hashes[
                "receiver_binding"
            ],
        "represented_scenes":
            120,
    },
    "checker_API_repair": {
        "formal_checker_sha256":
            before_hashes[
                "formal_checker"
            ],
        "amendment_sha256":
            before_hashes[
                "checker_repair_amendment"
            ],
        "pretruth_seal_sha256":
            before_hashes[
                "checker_repair_seal"
            ],
        "repair_report_sha256":
            before_hashes[
                "checker_repair_report"
            ],
        "semantic_change":
            False,
    },
    "frozen_chain":
        before_hashes,
    "history_manifest_sha256":
        history_manifest_sha,
    "scientific_boundary": {
        "retraining":
            False,
        "recalibration":
            False,
        "formal_parameter_tuning":
            False,
        "post_outcome_tuning":
            False,
        "Stage6_started_by_wrapper":
            False,
    },
}

marker_sha = write_once_json(
    MARKER,
    marker_payload,
)

print(
    "PASS | immutable V2 invocation marker written"
)
print(
    "marker_sha256 =",
    marker_sha,
)

# ------------------------------------------------------------
# E. Execute Checker2 exactly once.
# ------------------------------------------------------------

print()
print("=" * 79)
print(
    "EXECUTING CHECKER2 EXACTLY ONCE — V2"
)
print("=" * 79)

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

print(
    proc.stdout,
    end="",
)

# ------------------------------------------------------------
# F. Post-run immutability.
# ------------------------------------------------------------

print()
print("=" * 79)
print(
    "SUPERSEDING FORMAL V2 PROCESS FINISHED"
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

post_hashes = {
    "checker2":
        sha(CHECKER2),
    "formal_checker":
        sha(FORMAL_CHECKER),
    "runtime":
        sha(RUNTIME),
    "protocol":
        sha(PROTOCOL),
    "addendum":
        sha(ADDENDUM),
    "Stage4_ledger":
        sha(LEDGER),
    "receiver_binding":
        sha(BINDING),
    "binding_amendment":
        sha(BINDING_AMENDMENT),
    "binding_seal":
        sha(BINDING_SEAL),
    "checker_repair_amendment":
        sha(CHECKER_REPAIR_AMENDMENT),
    "checker_repair_seal":
        sha(CHECKER_REPAIR_SEAL),
    "checker_repair_report":
        sha(CHECKER_REPAIR_REPORT),
}

changed = {
    key: (
        before_hashes[key],
        post_hashes[key],
    )
    for key in before_hashes
    if (
        before_hashes[key]
        != post_hashes[key]
    )
}

if changed:
    fail(
        "sealed/frozen artifacts changed "
        "during V2 formal:\n"
        + json.dumps(
            changed,
            indent=2,
            sort_keys=True,
        )
    )

print(
    "PASS | all sealed/frozen "
    "preformal artifacts unchanged"
)
print(
    "Stage6 was NOT started by this wrapper."
)
print("=" * 79)

raise SystemExit(
    proc.returncode
)

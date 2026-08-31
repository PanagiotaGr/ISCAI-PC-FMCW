from __future__ import annotations

import ast
import hashlib
import json
import os
import re
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
    "stage5_fullpdf_v2_checker_probeplan_property_"
    "pretruth_seal_v1.json"
)

CHECKER_REPAIR_REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_checker_probeplan_property_repair_v1.json"
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
    S5
    / "artifacts/fullpdf_v2_independent_formal"
)

OLD_PRETRUTH = (
    ART
    / "checker2_pretruth_seal.json"
)

NEW_PRETRUTH = (
    ART
    / "checker2_pretruth_seal_superseding_v3.json"
)

V2_MARKER = (
    ART
    / "superseding_formal_receiver_binding_checkerfix_v2_invocation.json"
)

V2_LOG = (
    ART
    / "checker2_superseding_receiver_binding_checkerfix_v2.log"
)

REPAIR_DIR = (
    ART
    / "checker2_pretruth_authority_repair_v1"
)

HISTORY = (
    REPAIR_DIR
    / "pre_repair_history"
)

REGRESSION_LOG = (
    REPAIR_DIR
    / "stage5_regression_271.log"
)

AUTHORITY_AMENDMENT = (
    S5
    / "configs/"
    "stage5_fullpdf_v2_checker2_pretruth_authority_amendment_v1.json"
)

AUTHORITY_SEAL = (
    REPAIR_DIR
    / "stage5_fullpdf_v2_checker2_pretruth_authority_"
    "superseding_seal_v1.json"
)

AUTHORITY_REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_checker2_pretruth_authority_repair_v1.json"
)

V3_MARKER = (
    ART
    / "superseding_formal_receiver_binding_checkerfix_"
    "pretruthfix_v3_invocation.json"
)

V3_LOG = (
    ART
    / "checker2_superseding_receiver_binding_checkerfix_"
    "pretruthfix_v3.log"
)

POSTRUN = (
    ART
    / "superseding_formal_v3_postrun_audit.json"
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

    OLD_PRETRUTH:
        "44b1ba5bd702f8081f4e7e8c07a9139ccaa9e588ec938851b652d05c82693803",

    V2_MARKER:
        "e76a7b3172beb551997611a07b0cc644912faf29ce2da4c360f97af65924c6e3",

    V2_LOG:
        "d74617988bd92dd0711cae54d78d997c2c01a233e2eb2da1876c6827fe96676f",
}

EXPECTED_CHECKER2_BEFORE = (
    "bca973ed50e97510ec9c7e2022e39b0033643eed89c9599843519fb83898db04"
)

OLD_FORMAL_CHECKER_SHA = (
    "0e8bd2570453fdc4a92e3fab065f9726cb4d097b334e6b262959c54b0dd9d902"
)

OLD_SEAL_NAME = (
    "checker2_pretruth_seal.json"
)

NEW_SEAL_NAME = (
    "checker2_pretruth_seal_superseding_v3.json"
)


def fail(msg: str) -> None:
    raise RuntimeError(
        "FAIL-CLOSED: " + str(msg)
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
            f"JSON root is not object: {path}"
        )

    return obj


def canonical(obj) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(
    path: Path,
    data: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

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


def write_once_json(
    path: Path,
    obj: dict,
) -> str:
    data = canonical(
        obj
    )

    if path.exists():
        if path.read_bytes() != data:
            fail(
                f"write-once collision: {path}"
            )

        return sha(
            path
        )

    atomic_write(
        path,
        data,
    )

    os.chmod(
        path,
        0o444,
    )

    return sha(
        path
    )


def flatten(
    obj,
    prefix="",
):
    out = {}

    if isinstance(
        obj,
        dict,
    ):
        for key in sorted(
            obj
        ):
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            out.update(
                flatten(
                    obj[key],
                    path,
                )
            )

    else:
        out[prefix] = obj

    return out


def assert_sha(
    path: Path,
    expected: str,
) -> None:
    if not path.is_file():
        fail(
            f"missing required file: {path}"
        )

    actual = sha(
        path
    )

    if actual != expected:
        fail(
            "SHA mismatch:\n"
            f"{path}\n"
            f"actual   = {actual}\n"
            f"expected = {expected}"
        )


def run_regression() -> None:
    env = dict(
        os.environ
    )

    srcs = [
        S5 / "src",
        S4 / "src",
        ROOT / "iscai_stage3/src",
        ROOT / "iscai_stage2/src",
        ROOT / "iscai_stage1/src",
        ROOT / "iscai_stage0/src",
    ]

    env["PYTHONPATH"] = (
        os.pathsep.join(
            [
                str(x)
                for x in srcs
                if x.is_dir()
            ]
            + (
                [
                    env["PYTHONPATH"]
                ]
                if env.get(
                    "PYTHONPATH"
                )
                else []
            )
        )
    )

    proc = subprocess.run(
        [
            str(PYTHON),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S5 / "tests"
            ),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=str(S5),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    atomic_write(
        REGRESSION_LOG,
        proc.stdout.encode(
            "utf-8"
        ),
    )

    os.chmod(
        REGRESSION_LOG,
        0o444,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        proc.returncode != 0
        or count != 271
    ):
        fail(
            "Stage5 regression failed: "
            f"rc={proc.returncode}, "
            f"count={count}"
        )


def main() -> int:
    # --------------------------------------------------------
    # 1. Immutable chain before any repair.
    # --------------------------------------------------------

    for path, expected in (
        EXPECTED.items()
    ):
        assert_sha(
            path,
            expected,
        )

    assert_sha(
        CHECKER2,
        EXPECTED_CHECKER2_BEFORE,
    )

    if not FORMAL_REPORT.is_file():
        fail(
            "current prior formal report missing"
        )

    if not CHECKER2_REPORT.is_file():
        fail(
            "current Checker2 report missing"
        )

    v2_text = V2_LOG.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if (
        "existing pretruth seal does not match "
        "current scientific binding at hashes"
        not in v2_text
    ):
        fail(
            "V2 failure was not the expected "
            "pretruth-seal mismatch"
        )

    if (
        "===== EXECUTE FROZEN FORMAL CHECKER ====="
        in v2_text
    ):
        fail(
            "V2 unexpectedly reached formal checker"
        )

    # --------------------------------------------------------
    # 2. Exact old-seal/current-core diff audit.
    # --------------------------------------------------------

    old_seal = load_json(
        OLD_PRETRUTH
    )

    old_hashes = old_seal.get(
        "hashes",
        {},
    )

    if (
        old_hashes.get(
            "checker2"
        )
        != EXPECTED_CHECKER2_BEFORE
    ):
        fail(
            "old seal does not bind original Checker2"
        )

    if (
        old_hashes.get(
            "frozen_checker"
        )
        != OLD_FORMAL_CHECKER_SHA
    ):
        fail(
            "old seal does not bind expected "
            "pre-repair formal checker"
        )

    prior_report_sha = sha(
        FORMAL_REPORT
    )

    current_core = {
        "stage": 5,
        "version": "fullpdf_v2",
        "scientific_boundary":
            "before_current_checker2_formal_truth_access",
        "formal_outcomes_parsed_before_seal":
            False,
        "training":
            False,
        "retraining":
            False,
        "recalibration":
            False,
        "formal_tuning":
            False,
        "hashes": {
            "checker2":
                sha(CHECKER2),
            "frozen_checker":
                sha(FORMAL_CHECKER),
            "runtime":
                sha(RUNTIME),
            "protocol":
                sha(PROTOCOL),
            "preformal_addendum":
                sha(ADDENDUM),
            "Stage4_formal_prediction_ledger":
                sha(LEDGER),
        },
        "prior_formal_report_opaque_sha256":
            prior_report_sha,
    }

    old_flat = flatten(
        old_seal
    )

    new_flat = flatten(
        current_core
    )

    diff_keys = {
        key
        for key in (
            set(old_flat)
            | set(new_flat)
        )
        if (
            old_flat.get(
                key,
                "<MISSING>",
            )
            != new_flat.get(
                key,
                "<MISSING>",
            )
        )
    }

    allowed_diffs = {
        "hashes.frozen_checker",
        "prior_formal_report_opaque_sha256",
    }

    if (
        "hashes.frozen_checker"
        not in diff_keys
    ):
        fail(
            "expected formal-checker repair "
            "difference is absent"
        )

    if not diff_keys.issubset(
        allowed_diffs
    ):
        fail(
            "unexpected old-seal/current-core "
            f"differences: {sorted(diff_keys)}"
        )

    # --------------------------------------------------------
    # 3. Existing supersession chain must prove no science edit.
    # --------------------------------------------------------

    repair_seal = load_json(
        CHECKER_REPAIR_SEAL
    )

    if (
        repair_seal.get(
            "checker_sha256"
        )
        != sha(
            FORMAL_CHECKER
        )
    ):
        fail(
            "checker-repair seal does not bind "
            "current formal checker"
        )

    if (
        repair_seal.get(
            "checker2_sha256"
        )
        != sha(
            CHECKER2
        )
    ):
        fail(
            "checker-repair seal does not bind "
            "current pre-repair Checker2"
        )

    if (
        repair_seal.get(
            "previous_receiver_binding_seal_sha256"
        )
        != sha(
            BINDING_SEAL
        )
    ):
        fail(
            "checker-repair seal does not bind "
            "receiver-binding seal"
        )

    scientific_boundary = (
        repair_seal.get(
            "scientific_boundary",
            {},
        )
    )

    if (
        scientific_boundary.get(
            "training"
        )
        is not False
        or scientific_boundary.get(
            "recalibration"
        )
        is not False
        or scientific_boundary.get(
            "tuning"
        )
        is not False
        or scientific_boundary.get(
            "formal_outcome_used"
        )
        is not False
        or scientific_boundary.get(
            "evaluator_truth_used_for_repair"
        )
        is not False
    ):
        fail(
            "checker-repair scientific boundary "
            "is not clean"
        )

    # --------------------------------------------------------
    # 4. Preserve pre-repair/V2 history.
    # --------------------------------------------------------

    if HISTORY.exists():
        fail(
            "pretruth-authority repair V1 "
            "history already exists"
        )

    HISTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    snapshot_sources = [
        CHECKER2,
        OLD_PRETRUTH,
        FORMAL_REPORT,
        CHECKER2_REPORT,
        V2_MARKER,
        V2_LOG,
        FORMAL_CHECKER,
        CHECKER_REPAIR_AMENDMENT,
        CHECKER_REPAIR_SEAL,
        CHECKER_REPAIR_REPORT,
        BINDING_SEAL,
    ]

    entries = []

    for index, src in enumerate(
        snapshot_sources
    ):
        dst = (
            HISTORY
            / f"{index:02d}_{src.name}"
        )

        shutil.copy2(
            src,
            dst,
        )

        entries.append(
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

    history_sha = write_once_json(
        history_manifest,
        {
            "stage": 5,
            "version": "fullpdf_v2",
            "status":
                "PRESERVED_BEFORE_CHECKER2_"
                "PRETRUTH_AUTHORITY_REPAIR_V1",
            "V2_formal_checker_started":
                False,
            "V2_evaluator_truth_opened":
                False,
            "old_pretruth_seal_preserved":
                True,
            "old_pretruth_seal_sha256":
                sha(
                    OLD_PRETRUTH
                ),
            "allowed_old_vs_current_diff_keys":
                sorted(
                    diff_keys
                ),
            "files":
                entries,
        },
    )

    # --------------------------------------------------------
    # 5. Minimal Checker2 repair:
    #    only version the PRETRUTH_SEAL filename.
    # --------------------------------------------------------

    source_before = CHECKER2.read_text(
        encoding="utf-8"
    )

    if (
        source_before.count(
            OLD_SEAL_NAME
        )
        != 1
    ):
        fail(
            "expected exactly one old pretruth "
            "seal filename occurrence"
        )

    if (
        NEW_SEAL_NAME
        in source_before
    ):
        fail(
            "new pretruth seal filename already "
            "exists in Checker2"
        )

    tree_before = ast.parse(
        source_before
    )

    assignments = []

    for node in ast.walk(
        tree_before
    ):
        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:
            if (
                isinstance(
                    target,
                    ast.Name,
                )
                and target.id
                == "PRETRUTH_SEAL"
            ):
                assignments.append(
                    node
                )

    if len(
        assignments
    ) != 1:
        fail(
            "expected exactly one "
            "PRETRUTH_SEAL assignment"
        )

    assignment_segment = (
        ast.get_source_segment(
            source_before,
            assignments[0],
        )
        or ""
    )

    if (
        OLD_SEAL_NAME
        not in assignment_segment
    ):
        fail(
            "old seal filename is not owned by "
            "PRETRUTH_SEAL assignment"
        )

    source_after = (
        source_before.replace(
            OLD_SEAL_NAME,
            NEW_SEAL_NAME,
            1,
        )
    )

    ast.parse(
        source_after
    )

    before_lines = (
        source_before.splitlines()
    )

    after_lines = (
        source_after.splitlines()
    )

    if len(
        before_lines
    ) != len(
        after_lines
    ):
        fail(
            "Checker2 line count changed"
        )

    changed_lines = [
        i + 1
        for i, (
            before,
            after,
        )
        in enumerate(
            zip(
                before_lines,
                after_lines,
            )
        )
        if before != after
    ]

    if len(
        changed_lines
    ) != 1:
        fail(
            "Checker2 repair changed more than "
            f"one line: {changed_lines}"
        )

    if NEW_PRETRUTH.exists():
        fail(
            "new versioned pretruth seal already exists"
        )

    tmp = CHECKER2.with_name(
        CHECKER2.name
        + ".tmp_pretruth_authority_v1"
    )

    tmp.write_text(
        source_after,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        CHECKER2,
    )

    checker2_after_sha = sha(
        CHECKER2
    )

    if (
        checker2_after_sha
        == EXPECTED_CHECKER2_BEFORE
    ):
        fail(
            "Checker2 SHA did not change"
        )

    # --------------------------------------------------------
    # 6. Full Stage5 regression after one-line harness repair.
    # --------------------------------------------------------

    run_regression()

    # All scientific assets remain frozen.
    for path, expected in (
        EXPECTED.items()
    ):
        assert_sha(
            path,
            expected,
        )

    assert_sha(
        FORMAL_CHECKER,
        EXPECTED[
            FORMAL_CHECKER
        ],
    )

    assert_sha(
        OLD_PRETRUTH,
        EXPECTED[
            OLD_PRETRUTH
        ],
    )

    # --------------------------------------------------------
    # 7. Freeze the authority repair BEFORE formal V3.
    # --------------------------------------------------------

    authority_amendment = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status":
            "FROZEN_CHECKER2_PRETRUTH_AUTHORITY_"
            "SUPERSESSION_BEFORE_FORMAL_V3",
        "scope":
            "checker harness authority/path only",
        "old_checker2_sha256":
            EXPECTED_CHECKER2_BEFORE,
        "new_checker2_sha256":
            checker2_after_sha,
        "changed_lines":
            changed_lines,
        "source_change": {
            "old":
                OLD_SEAL_NAME,
            "new":
                NEW_SEAL_NAME,
        },
        "old_pretruth_seal": {
            "path":
                str(
                    OLD_PRETRUTH
                ),
            "sha256":
                sha(
                    OLD_PRETRUTH
                ),
            "preserved":
                True,
        },
        "new_pretruth_seal": {
            "path":
                str(
                    NEW_PRETRUTH
                ),
            "exists_before_formal":
                False,
            "created_only_by_Checker2":
                True,
        },
        "old_vs_current_seal_core_diff_keys":
            sorted(
                diff_keys
            ),
        "authority_chain": {
            "receiver_binding_seal_sha256":
                sha(
                    BINDING_SEAL
                ),
            "checker_property_repair_seal_sha256":
                sha(
                    CHECKER_REPAIR_SEAL
                ),
            "history_manifest_sha256":
                history_sha,
        },
        "scientific_change":
            False,
        "runtime_change":
            False,
        "protocol_change":
            False,
        "Stage4_change":
            False,
        "receiver_binding_change":
            False,
        "formal_checker_semantics_change":
            False,
        "training":
            False,
        "retraining":
            False,
        "recalibration":
            False,
        "tuning":
            False,
        "formal_outcome_used_for_repair":
            False,
        "evaluator_truth_used_for_repair":
            False,
        "regression": {
            "tests":
                271,
            "status":
                "PASS",
            "log":
                str(
                    REGRESSION_LOG
                ),
            "log_sha256":
                sha(
                    REGRESSION_LOG
                ),
        },
        "formal_V3_executed":
            False,
        "Stage6_allowed":
            False,
    }

    amendment_sha = write_once_json(
        AUTHORITY_AMENDMENT,
        authority_amendment,
    )

    authority_seal_sha = write_once_json(
        AUTHORITY_SEAL,
        {
            "stage": 5,
            "version": "fullpdf_v2",
            "status":
                "PREFORMAL_SEALED_CHECKER2_"
                "PRETRUTH_AUTHORITY_SUPERSESSION_V1",
            "old_checker2_sha256":
                EXPECTED_CHECKER2_BEFORE,
            "new_checker2_sha256":
                checker2_after_sha,
            "old_pretruth_seal_sha256":
                sha(
                    OLD_PRETRUTH
                ),
            "old_pretruth_seal_preserved":
                True,
            "new_pretruth_seal_path":
                str(
                    NEW_PRETRUTH
                ),
            "authority_amendment_sha256":
                amendment_sha,
            "checker_property_repair_seal_sha256":
                sha(
                    CHECKER_REPAIR_SEAL
                ),
            "receiver_binding_seal_sha256":
                sha(
                    BINDING_SEAL
                ),
            "Stage4_ledger_sha256":
                sha(
                    LEDGER
                ),
            "runtime_sha256":
                sha(
                    RUNTIME
                ),
            "protocol_sha256":
                sha(
                    PROTOCOL
                ),
            "preformal_addendum_sha256":
                sha(
                    ADDENDUM
                ),
            "formal_checker_sha256":
                sha(
                    FORMAL_CHECKER
                ),
            "regression_log_sha256":
                sha(
                    REGRESSION_LOG
                ),
            "scientific_boundary": {
                "training":
                    False,
                "retraining":
                    False,
                "recalibration":
                    False,
                "tuning":
                    False,
                "scientific_semantics_changed":
                    False,
                "evaluator_truth_used":
                    False,
                "formal_outcome_used":
                    False,
            },
        },
    )

    authority_report_sha = write_once_json(
        AUTHORITY_REPORT,
        {
            **authority_amendment,
            "authority_amendment_sha256":
                amendment_sha,
            "authority_seal_sha256":
                authority_seal_sha,
            "status":
                "PASS_READY_FOR_ONE_SHOT_FORMAL_V3",
        },
    )

    # --------------------------------------------------------
    # 8. Hard free-space gate.
    # --------------------------------------------------------

    free_bytes = shutil.disk_usage(
        ROOT
    ).free

    if (
        free_bytes
        < 250 * 1024**3
    ):
        fail(
            "free-space reserve below 250 GiB"
        )

    # --------------------------------------------------------
    # 9. Immutable one-shot Formal V3 invocation marker.
    # --------------------------------------------------------

    if V3_MARKER.exists():
        fail(
            "Formal V3 invocation marker already exists"
        )

    if V3_LOG.exists():
        fail(
            "Formal V3 log already exists"
        )

    if NEW_PRETRUTH.exists():
        fail(
            "new Checker2 pretruth seal exists "
            "before Formal V3"
        )

    formal_report_before_sha = sha(
        FORMAL_REPORT
    )

    protected_before = {
        "checker2":
            sha(
                CHECKER2
            ),
        "formal_checker":
            sha(
                FORMAL_CHECKER
            ),
        "runtime":
            sha(
                RUNTIME
            ),
        "protocol":
            sha(
                PROTOCOL
            ),
        "addendum":
            sha(
                ADDENDUM
            ),
        "Stage4_ledger":
            sha(
                LEDGER
            ),
        "receiver_binding":
            sha(
                BINDING
            ),
        "receiver_binding_amendment":
            sha(
                BINDING_AMENDMENT
            ),
        "receiver_binding_seal":
            sha(
                BINDING_SEAL
            ),
        "checker_repair_seal":
            sha(
                CHECKER_REPAIR_SEAL
            ),
        "authority_amendment":
            sha(
                AUTHORITY_AMENDMENT
            ),
        "authority_seal":
            sha(
                AUTHORITY_SEAL
            ),
    }

    marker_sha = write_once_json(
        V3_MARKER,
        {
            "stage": 5,
            "version": "fullpdf_v2",
            "attempt":
                "superseding_formal_v3_after_"
                "pretruth_authority_supersession",
            "status":
                "FORMAL_INVOCATION_STARTED",
            "started_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
            "one_shot":
                True,
            "protected_preformal_hashes":
                protected_before,
            "prior_formal_report_opaque_sha256":
                formal_report_before_sha,
            "old_pretruth_seal_sha256":
                sha(
                    OLD_PRETRUTH
                ),
            "new_pretruth_seal_path":
                str(
                    NEW_PRETRUTH
                ),
            "authority_seal_sha256":
                authority_seal_sha,
            "scientific_boundary": {
                "training":
                    False,
                "retraining":
                    False,
                "recalibration":
                    False,
                "tuning":
                    False,
                "post_outcome_tuning":
                    False,
                "Stage6_started_by_wrapper":
                    False,
            },
        },
    )

    # --------------------------------------------------------
    # 10. Execute Checker2 exactly once.
    #     Do NOT echo its huge output to terminal.
    # --------------------------------------------------------

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

    atomic_write(
        V3_LOG,
        proc.stdout.encode(
            "utf-8"
        ),
    )

    os.chmod(
        V3_LOG,
        0o444,
    )

    # --------------------------------------------------------
    # 11. Post-run immutability.
    # --------------------------------------------------------

    protected_after = {
        "checker2":
            sha(
                CHECKER2
            ),
        "formal_checker":
            sha(
                FORMAL_CHECKER
            ),
        "runtime":
            sha(
                RUNTIME
            ),
        "protocol":
            sha(
                PROTOCOL
            ),
        "addendum":
            sha(
                ADDENDUM
            ),
        "Stage4_ledger":
            sha(
                LEDGER
            ),
        "receiver_binding":
            sha(
                BINDING
            ),
        "receiver_binding_amendment":
            sha(
                BINDING_AMENDMENT
            ),
        "receiver_binding_seal":
            sha(
                BINDING_SEAL
            ),
        "checker_repair_seal":
            sha(
                CHECKER_REPAIR_SEAL
            ),
        "authority_amendment":
            sha(
                AUTHORITY_AMENDMENT
            ),
        "authority_seal":
            sha(
                AUTHORITY_SEAL
            ),
    }

    if (
        protected_after
        != protected_before
    ):
        fail(
            "protected scientific/preformal "
            "artifacts changed during Formal V3"
        )

    assert_sha(
        OLD_PRETRUTH,
        EXPECTED[
            OLD_PRETRUTH
        ],
    )

    # --------------------------------------------------------
    # 12. The NEW write-once Checker2 seal must bind V3.
    # --------------------------------------------------------

    new_seal_ok = False
    new_seal_sha = None

    if NEW_PRETRUTH.is_file():
        new_seal = load_json(
            NEW_PRETRUTH
        )

        expected_new_hashes = {
            "checker2":
                protected_before[
                    "checker2"
                ],
            "frozen_checker":
                protected_before[
                    "formal_checker"
                ],
            "runtime":
                protected_before[
                    "runtime"
                ],
            "protocol":
                protected_before[
                    "protocol"
                ],
            "preformal_addendum":
                protected_before[
                    "addendum"
                ],
            "Stage4_formal_prediction_ledger":
                protected_before[
                    "Stage4_ledger"
                ],
        }

        if (
            new_seal.get(
                "hashes"
            )
            != expected_new_hashes
        ):
            fail(
                "new Checker2 pretruth seal "
                "does not bind exact V3 hashes"
            )

        if (
            new_seal.get(
                "prior_formal_report_opaque_sha256"
            )
            != formal_report_before_sha
        ):
            fail(
                "new Checker2 pretruth seal "
                "does not bind exact prior report"
            )

        if (
            new_seal.get(
                "formal_outcomes_parsed_before_seal"
            )
            is not False
        ):
            fail(
                "new seal claims outcomes parsed "
                "before seal"
            )

        new_seal_ok = True
        new_seal_sha = sha(
            NEW_PRETRUTH
        )

    # --------------------------------------------------------
    # 13. Compact post-run audit only.
    # --------------------------------------------------------

    checker2_result = (
        load_json(
            CHECKER2_REPORT
        )
        if CHECKER2_REPORT.is_file()
        else {}
    )

    formal_result = (
        load_json(
            FORMAL_REPORT
        )
        if FORMAL_REPORT.is_file()
        else {}
    )

    failures = checker2_result.get(
        "failures",
        [],
    )

    if not isinstance(
        failures,
        list,
    ):
        failures = []

    postrun_sha = write_once_json(
        POSTRUN,
        {
            "stage": 5,
            "version": "fullpdf_v2",
            "attempt":
                "superseding_formal_v3",
            "checker2_returncode":
                proc.returncode,
            "checker2_log": {
                "path":
                    str(
                        V3_LOG
                    ),
                "sha256":
                    sha(
                        V3_LOG
                    ),
            },
            "invocation_marker_sha256":
                marker_sha,
            "old_pretruth_seal_preserved":
                (
                    sha(
                        OLD_PRETRUTH
                    )
                    == EXPECTED[
                        OLD_PRETRUTH
                    ]
                ),
            "new_pretruth_seal_created":
                NEW_PRETRUTH.is_file(),
            "new_pretruth_seal_valid":
                new_seal_ok,
            "new_pretruth_seal_sha256":
                new_seal_sha,
            "Checker2_status":
                checker2_result.get(
                    "status"
                ),
            "Stage6_allowed":
                checker2_result.get(
                    "Stage6_allowed",
                    False,
                ),
            "mandatory_failure_count":
                len(
                    failures
                ),
            "formal_status":
                formal_result.get(
                    "status"
                ),
            "formal_failure_reason":
                formal_result.get(
                    "failure_reason",
                    formal_result.get(
                        "reason"
                    ),
                ),
            "post_outcome_tuning":
                False,
            "Stage6_started":
                False,
        },
    )

    # --------------------------------------------------------
    # 14. SMALL terminal summary only.
    # --------------------------------------------------------

    print("=" * 72)
    print(
        "STAGE5 FULLPDF V2 — "
        "PREFORMAL SUPERSESSION + FORMAL V3 COMPLETE"
    )
    print("=" * 72)

    print(
        "pretruth_authority_repair = PASS"
    )

    print(
        "old_seal_preserved = true"
    )

    print(
        "checker2_new_sha256 =",
        checker2_after_sha,
    )

    print(
        "271/271 = PASS"
    )

    print(
        "authority_seal_sha256 =",
        authority_seal_sha,
    )

    print(
        "checker2_returncode =",
        proc.returncode,
    )

    print(
        "new_pretruth_seal_created =",
        NEW_PRETRUTH.is_file(),
    )

    print(
        "new_pretruth_seal_valid =",
        new_seal_ok,
    )

    if new_seal_sha is not None:
        print(
            "new_pretruth_seal_sha256 =",
            new_seal_sha,
        )

    print(
        "formal_status =",
        formal_result.get(
            "status"
        ),
    )

    print(
        "Checker2_status =",
        checker2_result.get(
            "status"
        ),
    )

    print(
        "Stage6_allowed =",
        checker2_result.get(
            "Stage6_allowed",
            False,
        ),
    )

    print(
        "mandatory_failure_count =",
        len(
            failures
        ),
    )

    reason = formal_result.get(
        "failure_reason",
        formal_result.get(
            "reason"
        ),
    )

    if reason:
        print(
            "formal_reason =",
            reason,
        )

    print(
        "formal_log =",
        V3_LOG,
    )

    print(
        "formal_log_sha256 =",
        sha(
            V3_LOG
        ),
    )

    print(
        "postrun_audit_sha256 =",
        postrun_sha,
    )

    print(
        "Stage6_started = false"
    )

    print("=" * 72)

    return int(
        proc.returncode
    )


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print("=" * 72)
        print(
            "STAGE5 V3 ORCHESTRATOR = FAIL-CLOSED"
        )
        print(
            "reason =",
            repr(
                exc
            ),
        )
        print(
            "Stage6_allowed = false"
        )
        print(
            "Stage6_started = false"
        )
        print("=" * 72)
        raise

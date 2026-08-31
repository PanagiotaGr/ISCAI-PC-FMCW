from __future__ import annotations

import ast
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

CHECKER = (
    S5
    / "scripts/run_stage5_fullpdf_v2_independent_formal_checker.py"
)

CHECKER2 = (
    S5 / "scripts/check_stage5_fullpdf_v2.py"
)

RUNTIME = (
    S5 / "src/iscai_stage5/fullpdf_v2.py"
)

PROTOCOL = (
    S5 / "configs/stage5_fullpdf_v2_protocol.json"
)

ADDENDUM = (
    S5 / "configs/stage5_fullpdf_v2_preformal_addendum.json"
)

LEDGER = (
    S4 / "artifacts/fullpdf_v2/formal_prediction_ledger_run1.jsonl"
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

INVOCATION_MARKER = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "superseding_formal_receiver_binding_v1_invocation.json"
)

FAILED_WRAPPER_LOG = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "checker2_superseding_receiver_binding_v1.log"
)

FROZEN_CHECKER_LOG = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "checker2_frozen_checker.log"
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

OUT_DIR = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
    "probeplan_property_repair_v1"
)

HISTORY = (
    OUT_DIR
    / "failed_superseding_v1_history"
)

REGRESSION_LOG = (
    OUT_DIR
    / "stage5_regression_271.log"
)

AMENDMENT = (
    S5
    / "configs/"
    "stage5_fullpdf_v2_checker_probeplan_property_amendment_v1.json"
)

SEAL = (
    OUT_DIR
    / "stage5_fullpdf_v2_checker_probeplan_property_pretruth_seal_v1.json"
)

REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_checker_probeplan_property_repair_v1.json"
)

EXPECTED_FROZEN = {
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
    INVOCATION_MARKER:
        "768b68b2bafc7152b79e051d8fa08582877625b2d62482bb16e4c69bb5eabfe6",
    FAILED_WRAPPER_LOG:
        "a28e060a5b2f54dbabe8b3453fb5c1b65ba5c18fd5c3fd89e9c3a7eff1031b9c",
    FROZEN_CHECKER_LOG:
        "38dea00263a74475a8903435a0d92deb33aceb1ce81055797302d9fcbcd584cb",
    CHECKER2_REPORT:
        "d71d9b4f94d4c1d615c8336c7511e1c6de61339c40c7c85128842955ee62a853",
}


def fail(msg):
    raise SystemExit(
        "FAIL-CLOSED: " + str(msg)
    )


def sha(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def canonical_json(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path, data):
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
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )


def write_once(path, data):
    digest = hashlib.sha256(
        data
    ).hexdigest()

    if path.exists():
        if (
            path.read_bytes() != data
            or sha(path) != digest
        ):
            fail(
                f"write-once collision: {path}"
            )

        return digest

    atomic_write(
        path,
        data,
    )

    os.chmod(
        path,
        0o444,
    )

    return digest


def load_json(path):
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


def main():
    print("=" * 79)
    print(
        "STAGE5 FULLPDF V2 — "
        "CHECKER PROBEPLAN PROPERTY REPAIR V1"
    )
    print(
        "HARNESS/API ONLY | "
        "NO FORMAL METRICS | "
        "NO EVALUATOR TRUTH"
    )
    print("=" * 79)

    for p in (
        PYTHON,
        CHECKER,
        CHECKER2,
        FORMAL_REPORT,
        *EXPECTED_FROZEN.keys(),
    ):
        if not p.is_file():
            fail(
                f"missing required file: {p}"
            )

    # --------------------------------------------------------
    # 1. Frozen/pretruth integrity.
    # --------------------------------------------------------

    for path, expected in (
        EXPECTED_FROZEN.items()
    ):
        actual = sha(path)

        if actual != expected:
            fail(
                "frozen/pretruth SHA mismatch:\n"
                f"{path}\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

        print(
            "PASS | preserved SHA |",
            path.name,
        )

    marker = load_json(
        INVOCATION_MARKER
    )

    current_checker_sha = sha(
        CHECKER
    )

    marker_checker_sha = (
        marker
        .get(
            "frozen",
            {},
        )
        .get(
            "formal_checker_sha256"
        )
    )

    if (
        marker_checker_sha
        != current_checker_sha
    ):
        fail(
            "formal checker changed after "
            "failed V1 invocation:\n"
            f"marker  = {marker_checker_sha}\n"
            f"current = {current_checker_sha}"
        )

    marker_checker2_sha = (
        marker
        .get(
            "frozen",
            {},
        )
        .get(
            "checker2_sha256"
        )
    )

    if (
        marker_checker2_sha
        != sha(CHECKER2)
    ):
        fail(
            "Checker2 changed after "
            "failed V1 invocation"
        )

    print(
        "PASS | failed V1 invocation "
        "binds current checker/checker2"
    )

    # --------------------------------------------------------
    # 2. Preserve complete failed attempt before checker edit.
    # --------------------------------------------------------

    if HISTORY.exists():
        fail(
            "repair V1 history already exists; "
            "will not overwrite"
        )

    HISTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    history_sources = [
        INVOCATION_MARKER,
        FAILED_WRAPPER_LOG,
        FROZEN_CHECKER_LOG,
        FORMAL_REPORT,
        CHECKER2_REPORT,
        CHECKER,
    ]

    history_entries = []

    for index, src in enumerate(
        history_sources
    ):
        dst = (
            HISTORY
            / f"{index:02d}_{src.name}"
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

    history_manifest = {
        "stage": 5,
        "version": "fullpdf_v2",
        "attempt":
            "superseding_receiver_binding_v1",
        "status":
            "PRESERVED_PREFORMAL_CHECKER_EXCEPTION",
        "failure":
            "TypeError: int object is not callable "
            "at plan.probing_beam_count()",
        "decision_pass_1_completed":
            False,
        "decision_ledger_sealed":
            False,
        "scientific_outcome_computed":
            False,
        "files":
            history_entries,
    }

    history_manifest_path = (
        HISTORY
        / "history_manifest.json"
    )

    atomic_write(
        history_manifest_path,
        canonical_json(
            history_manifest
        ),
    )

    os.chmod(
        history_manifest_path,
        0o444,
    )

    history_manifest_sha = sha(
        history_manifest_path
    )

    print(
        "PASS | failed superseding "
        "V1 attempt preserved"
    )

    # --------------------------------------------------------
    # 3. Static proof of runtime API semantics.
    # --------------------------------------------------------

    runtime_src = (
        RUNTIME.read_text(
            encoding="utf-8"
        )
    )

    runtime_tree = ast.parse(
        runtime_src
    )

    probeplan = None

    for node in runtime_tree.body:
        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and node.name
            == "ProbePlan"
        ):
            probeplan = node
            break

    if probeplan is None:
        fail(
            "runtime ProbePlan class not found"
        )

    method = None

    for node in probeplan.body:
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "probing_beam_count"
        ):
            method = node
            break

    if method is None:
        fail(
            "ProbePlan.probing_beam_count "
            "not found"
        )

    decorators = [
        ast.unparse(x)
        for x
        in method.decorator_list
    ]

    if decorators != [
        "property"
    ]:
        fail(
            "probing_beam_count is not "
            "exactly a @property"
        )

    # Require exact semantic body:
    # return len(self.probes)
    if (
        len(method.body) != 1
        or not isinstance(
            method.body[0],
            ast.Return,
        )
    ):
        fail(
            "unexpected probing_beam_count "
            "property body"
        )

    returned = ast.unparse(
        method.body[0].value
    )

    if returned != (
        "len(self.probes)"
    ):
        fail(
            "probing_beam_count semantics "
            "changed: "
            f"{returned}"
        )

    print(
        "PASS | runtime ProbePlan."
        "probing_beam_count is exact "
        "@property -> len(self.probes)"
    )

    # --------------------------------------------------------
    # 4. Static proof of exact checker defect.
    # --------------------------------------------------------

    checker_before = (
        CHECKER.read_text(
            encoding="utf-8"
        )
    )

    checker_tree = ast.parse(
        checker_before
    )

    bad_calls = []

    for node in ast.walk(
        checker_tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = node.func

        if (
            isinstance(
                func,
                ast.Attribute,
            )
            and func.attr
            == "probing_beam_count"
        ):
            bad_calls.append(
                node
            )

    if len(
        bad_calls
    ) != 1:
        fail(
            "expected exactly one checker "
            "probing_beam_count() call, found "
            f"{len(bad_calls)}"
        )

    bad = bad_calls[0]

    if bad.lineno != 1990:
        fail(
            "checker defect moved from "
            f"expected line 1990 to {bad.lineno}"
        )

    print(
        "PASS | exactly one invalid "
        "checker property call found"
    )

    # --------------------------------------------------------
    # 5. Minimal source repair.
    # --------------------------------------------------------

    old_text = (
        "plan.probing_beam_count()"
    )
    new_text = (
        "plan.probing_beam_count"
    )

    if checker_before.count(
        old_text
    ) != 1:
        fail(
            "exact repair target count != 1"
        )

    checker_after = (
        checker_before.replace(
            old_text,
            new_text,
            1,
        )
    )

    # Syntax gate.
    after_tree = ast.parse(
        checker_after
    )

    # Prove no callable property access remains.
    remaining_bad = []

    for node in ast.walk(
        after_tree
    ):
        if (
            isinstance(
                node,
                ast.Call,
            )
            and isinstance(
                node.func,
                ast.Attribute,
            )
            and node.func.attr
            == "probing_beam_count"
        ):
            remaining_bad.append(
                node.lineno
            )

    if remaining_bad:
        fail(
            "callable probing_beam_count "
            f"remains: {remaining_bad}"
        )

    # Prove property access still exists exactly once.
    accesses = []

    for node in ast.walk(
        after_tree
    ):
        if (
            isinstance(
                node,
                ast.Attribute,
            )
            and node.attr
            == "probing_beam_count"
        ):
            accesses.append(
                node.lineno
            )

    if accesses != [
        1990
    ]:
        fail(
            "unexpected property access "
            f"locations after repair: {accesses}"
        )

    backup = (
        OUT_DIR
        / (
            "run_stage5_fullpdf_v2_"
            "independent_formal_checker."
            "pre_probeplan_property_fix."
            + current_checker_sha[:16]
            + ".py"
        )
    )

    write_once(
        backup,
        checker_before.encode(
            "utf-8"
        ),
    )

    tmp = CHECKER.with_name(
        CHECKER.name
        + ".tmp_probeplan_property_fix"
    )

    tmp.write_text(
        checker_after,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        CHECKER,
    )

    repaired_checker_sha = sha(
        CHECKER
    )

    if (
        repaired_checker_sha
        == current_checker_sha
    ):
        fail(
            "checker SHA did not change"
        )

    print(
        "PASS | checker minimally repaired"
    )
    print(
        "old_checker_sha256 =",
        current_checker_sha,
    )
    print(
        "new_checker_sha256 =",
        repaired_checker_sha,
    )

    # --------------------------------------------------------
    # 6. Re-audit exact changed line only.
    # --------------------------------------------------------

    before_lines = (
        checker_before.splitlines()
    )
    after_lines = (
        checker_after.splitlines()
    )

    if len(
        before_lines
    ) != len(
        after_lines
    ):
        fail(
            "checker line count changed"
        )

    changed = [
        i + 1
        for i, (
            a,
            b,
        )
        in enumerate(
            zip(
                before_lines,
                after_lines,
            )
        )
        if a != b
    ]

    if changed != [
        1990
    ]:
        fail(
            "checker changed outside "
            f"line 1990: {changed}"
        )

    print(
        "PASS | only checker line 1990 changed"
    )
    print(
        "before:",
        before_lines[1989].strip(),
    )
    print(
        "after :",
        after_lines[1989].strip(),
    )

    # --------------------------------------------------------
    # 7. Full 271-test regression.
    # --------------------------------------------------------

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

    env[
        "PYTHONPATH"
    ] = os.pathsep.join(
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

    if REGRESSION_LOG.exists():
        fail(
            "repair regression log "
            "already exists"
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

    import re

    m = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    count = (
        int(m.group(1))
        if m
        else None
    )

    if (
        proc.returncode != 0
        or count != 271
    ):
        fail(
            "Stage5 regression failed: "
            f"rc={proc.returncode}, "
            f"tests={count}"
        )

    print(
        "PASS | full Stage5 "
        "regression 271/271"
    )

    # --------------------------------------------------------
    # 8. Frozen scientific assets must remain unchanged.
    # --------------------------------------------------------

    for path, expected in (
        EXPECTED_FROZEN.items()
    ):
        if sha(path) != expected:
            fail(
                "frozen/pretruth asset "
                f"changed during repair: {path}"
            )

    if sha(
        CHECKER2
    ) != marker_checker2_sha:
        fail(
            "Checker2 changed during repair"
        )

    print(
        "PASS | all frozen scientific/"
        "pretruth assets unchanged"
    )

    # --------------------------------------------------------
    # 9. New checker-repair amendment and seal.
    # --------------------------------------------------------

    amendment = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status":
            "FROZEN_CHECKER_API_REPAIR_BEFORE_SCIENTIFIC_OUTCOME",
        "repair_scope":
            "formal-checker API access only",
        "failure_observed":
            "TypeError: 'int' object is not callable "
            "at plan.probing_beam_count()",
        "root_cause": {
            "runtime_type":
                "iscai_stage5.fullpdf_v2.ProbePlan",
            "runtime_member":
                "probing_beam_count",
            "runtime_member_kind":
                "@property",
            "runtime_semantics":
                "len(self.probes)",
            "invalid_checker_expression":
                "plan.probing_beam_count()",
            "repaired_checker_expression":
                "plan.probing_beam_count",
        },
        "semantic_change": False,
        "scientific_parameter_change": False,
        "runtime_change": False,
        "protocol_change": False,
        "receiver_binding_change": False,
        "Stage4_change": False,
        "training": False,
        "recalibration": False,
        "tuning": False,
        "formal_scientific_outcome_used":
            False,
        "previous_attempt": {
            "invocation_marker_sha256":
                sha(
                    INVOCATION_MARKER
                ),
            "wrapper_log_sha256":
                sha(
                    FAILED_WRAPPER_LOG
                ),
            "frozen_checker_log_sha256":
                sha(
                    FROZEN_CHECKER_LOG
                ),
            "checker2_report_sha256":
                sha(
                    CHECKER2_REPORT
                ),
            "decision_pass_1_completed":
                False,
            "decision_ledger_sealed":
                False,
            "evaluator_truth_opened":
                False,
        },
        "checker": {
            "path":
                str(CHECKER),
            "old_sha256":
                current_checker_sha,
            "new_sha256":
                repaired_checker_sha,
            "changed_lines": [
                1990
            ],
        },
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
        "failed_attempt_history": {
            "manifest":
                str(
                    history_manifest_path
                ),
            "manifest_sha256":
                history_manifest_sha,
        },
        "formal_rerun_executed":
            False,
        "Stage6_allowed":
            False,
    }

    amendment_sha = write_once(
        AMENDMENT,
        canonical_json(
            amendment
        ),
    )

    seal_payload = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status":
            "PREFORMAL_SEALED_AFTER_CHECKER_API_REPAIR_V1",
        "formal_scientific_outcome_not_yet_computed":
            True,
        "checker_repair_amendment_sha256":
            amendment_sha,
        "checker_sha256":
            repaired_checker_sha,
        "checker2_sha256":
            sha(
                CHECKER2
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
        "Stage4_ledger_sha256":
            sha(
                LEDGER
            ),
        "receiver_binding_sha256":
            sha(
                BINDING
            ),
        "receiver_binding_amendment_sha256":
            sha(
                BINDING_AMENDMENT
            ),
        "previous_receiver_binding_seal_sha256":
            sha(
                BINDING_SEAL
            ),
        "failed_attempt_history_manifest_sha256":
            history_manifest_sha,
        "regression_log_sha256":
            sha(
                REGRESSION_LOG
            ),
        "scientific_boundary": {
            "training":
                False,
            "recalibration":
                False,
            "tuning":
                False,
            "runtime_semantics_changed":
                False,
            "formal_outcome_used":
                False,
            "evaluator_truth_used_for_repair":
                False,
        },
    }

    seal_sha = write_once(
        SEAL,
        canonical_json(
            seal_payload
        ),
    )

    report = {
        **amendment,
        "amendment": {
            "path":
                str(
                    AMENDMENT
                ),
            "sha256":
                amendment_sha,
        },
        "pretruth_seal": {
            "path":
                str(
                    SEAL
                ),
            "sha256":
                seal_sha,
        },
        "status":
            "PASS_READY_FOR_NEW_SUPERSEDING_FORMAL_ATTEMPT",
    }

    report_sha = write_once(
        REPORT,
        canonical_json(
            report
        ),
    )

    print()
    print("=" * 79)
    print(
        "CHECKER PROBEPLAN PROPERTY "
        "REPAIR = PASS"
    )
    print(
        "Stage5 = READY_FOR_NEW_"
        "SUPERSEDING_FORMAL_ATTEMPT"
    )
    print(
        "formal scientific outcome = "
        "NOT COMPUTED"
    )
    print(
        "Stage6_allowed = false"
    )
    print(
        "checker_new_sha256 =",
        repaired_checker_sha,
    )
    print(
        "amendment_sha256 =",
        amendment_sha,
    )
    print(
        "pretruth_seal_sha256 =",
        seal_sha,
    )
    print(
        "report_sha256 =",
        report_sha,
    )
    print("=" * 79)


if __name__ == "__main__":
    main()

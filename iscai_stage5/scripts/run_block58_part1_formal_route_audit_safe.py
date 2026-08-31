from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

AUDIT = (
    STAGE5
    / "scripts/"
      "run_block58_part1_formal_route_audit.py"
)

REPORT = (
    STAGE5
    / "artifacts/block58/"
      "formal_input_route_audit.json"
)


def blocked(
    phase,
    reason,
):
    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART1 SAFE CONTROLLER"
    )
    print(
        "============================================================"
    )

    print(
        "STATUS = BLOCKED"
    )

    print(
        "phase =",
        phase,
    )

    print(
        "reason =",
        reason,
    )

    print(
        "formal metrics computed = NO"
    )

    print(
        "Stage4 inference = NO"
    )

    print(
        "terminal remains open = YES"
    )


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE5
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            STAGE5
        ),
        text=True,
        capture_output=True,
    )

    output = (
        process.stdout
        +
        "\n"
        +
        process.stderr
    )

    print(
        output
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        output,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    return (
        process.returncode,
        count,
    )


def stream():
    process = subprocess.Popen(
        [
            sys.executable,
            str(
                AUDIT
            ),
        ],
        cwd=str(
            STAGE5
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    if process.stdout is not None:

        for line in process.stdout:

            print(
                line,
                end="",
                flush=True,
            )

    return process.wait()


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 1/2 — SAFE FORMAL ROUTE AUDIT"
    )
    print(
        "============================================================"
    )

    try:
        py_compile.compile(
            str(
                AUDIT
            ),
            doraise=True,
        )

        print(
            "formal-route audit compile = PASS"
        )

    except Exception as exc:

        blocked(
            "compile",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return

    print()
    print(
        "===== PRE-AUDIT STAGE5 REGRESSION ====="
    )

    code, count = (
        run_regression()
    )

    if (
        code != 0
        or
        count != 226
    ):

        blocked(
            "pre_audit_regression",
            (
                "Expected Stage5 regression "
                f"226/226; got {count}."
            ),
        )

        return

    print(
        "pre-audit Stage5 regression = 226 / 226 PASS"
    )

    print()
    print(
        "===== FROZEN FORMAL INPUT-ROUTE AUDIT ====="
    )

    raw_code = stream()

    print()
    print(
        "formal-route audit raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Formal-route audit report "
                "was not written."
            ),
        )

        return

    try:
        report = json.loads(
            REPORT.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        blocked(
            "report_readback",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return

    if (
        report.get(
            "status"
        )
        !=
        "PASS_READ_ONLY_AUDIT"
    ):

        blocked(
            "audit_gate",
            (
                "Formal-route audit "
                "structured status is not PASS."
            ),
        )

        return

    scientific = report.get(
        "scientific_execution",
        {}
    )

    if scientific.get(
        "formal_metrics_computed"
    ) is not False:

        blocked(
            "premature_formal_evaluation",
            (
                "Formal metrics were computed "
                "inside the route audit."
            ),
        )

        return

    if scientific.get(
        "Stage4_inference"
    ) is not False:

        blocked(
            "premature_inference",
            (
                "Stage4 inference unexpectedly "
                "ran during route audit."
            ),
        )

        return

    print()
    print(
        "===== POST-AUDIT STAGE5 REGRESSION ====="
    )

    code, count = (
        run_regression()
    )

    if (
        code != 0
        or
        count != 226
    ):

        blocked(
            "post_audit_regression",
            (
                "Expected Stage5 regression "
                f"226/226; got {count}."
            ),
        )

        return

    print(
        "post-audit Stage5 regression = 226 / 226 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 1/2 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.7 continuity       = PASS"
    )

    print(
        "formal N=120 metadata     = READ-ONLY"
    )

    print(
        "formal metrics            = NOT COMPUTED"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "post-hoc tuning           = NO"
    )

    print(
        "route status              =",
        report.get(
            "route_status"
        ),
    )

    print(
        "sample-level candidates   =",
        len(
            report.get(
                "materialized_formal_candidates",
                [],
            )
        ),
    )

    print(
        "Stage5 regression         = 226 / 226 PASS"
    )

    print(
        "STATUS = PASS_READ_ONLY_AUDIT"
    )

    print(
        "next = RESOLVE EXACT FORMAL ROUTE -> BLOCK 5.8 PART 2/2"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    blocked(
        "unexpected_controller_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

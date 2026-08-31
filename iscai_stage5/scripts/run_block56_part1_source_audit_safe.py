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
      "run_block56_part1_exact_source_audit.py"
)

REPORT = (
    STAGE5
    / "artifacts/block56/"
      "exact_parta_optical_headlamp_audit.json"
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
        "BLOCK 5.6 PART1 SAFE CONTROLLER"
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
        "formal N=120 used = NO"
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
        "BLOCK 5.6 PART 1 — SAFE READ-ONLY SOURCE AUDIT"
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

    except Exception as exc:

        blocked(
            "compile",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return

    print(
        "audit compile = PASS"
    )

    print()
    print(
        "===== PRE-AUDIT STAGE5 REGRESSION ====="
    )

    code, count = run_regression()

    if (
        code != 0
        or
        count != 186
    ):

        blocked(
            "pre_audit_regression",
            (
                "Expected Stage5 regression "
                f"186/186; got {count}."
            ),
        )

        return

    print(
        "pre-audit Stage5 regression = 186 / 186 PASS"
    )

    print()
    print(
        "===== EXACT SOURCE AUDIT ====="
    )

    raw_code = stream()

    print()
    print(
        "audit raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.6 Part1 audit "
                "report was not written."
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
                "Block5.6 source audit "
                "structured status is not PASS."
            ),
        )

        return

    scientific = report.get(
        "scientific_execution",
        {}
    )

    if scientific.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "formal_leakage",
            (
                "Audit does not prove "
                "formal_N120_read=False."
            ),
        )

        return

    print()
    print(
        "===== POST-AUDIT STAGE5 REGRESSION ====="
    )

    code, count = run_regression()

    if (
        code != 0
        or
        count != 186
    ):

        blocked(
            "post_audit_regression",
            (
                "Expected Stage5 regression "
                f"186/186; got {count}."
            ),
        )

        return

    print(
        "post-audit Stage5 regression = 186 / 186 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.6 PART 1 SOURCE AUDIT FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part-A frozen integrity      = PASS"
    )

    print(
        "optical/DPSK source evidence = FOUND"
    )

    print(
        "effective-rate formula       = SOURCE STATUS AUDITED"
    )

    print(
        "headlamp/H0 transform        = SOURCE STATUS AUDITED"
    )

    print(
        "formal N=120 used            = NO"
    )

    print(
        "Stage4 inference             = NO"
    )

    print(
        "optical implementation       = NOT STARTED"
    )

    print(
        "Stage5 regression            = 186 / 186 PASS"
    )

    print(
        "STATUS = PASS_READ_ONLY_AUDIT"
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

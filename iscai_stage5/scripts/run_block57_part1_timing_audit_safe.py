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
      "run_block57_part1_timing_audit.py"
)

REPORT = (
    STAGE5
    / "artifacts/block57/"
      "timing_source_audit.json"
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
        "BLOCK 5.7 PART1 SAFE CONTROLLER"
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
        "BLOCK 5.7 PART 1/2 — SAFE READ-ONLY TIMING AUDIT"
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
            "timing-audit compile = PASS"
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
        count != 206
    ):

        blocked(
            "pre_audit_regression",
            (
                "Expected Stage5 regression "
                f"206/206; got {count}."
            ),
        )

        return

    print(
        "pre-audit Stage5 regression = 206 / 206 PASS"
    )

    print()
    print(
        "===== EXACT TIMING SOURCE AUDIT ====="
    )

    raw_code = stream()

    print()
    print(
        "timing-audit raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Timing audit report "
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
                "Timing audit structured "
                "status is not PASS."
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
                "Timing audit does not prove "
                "formal_N120_read=False."
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
        count != 206
    ):

        blocked(
            "post_audit_regression",
            (
                "Expected Stage5 regression "
                f"206/206; got {count}."
            ),
        )

        return

    print(
        "post-audit Stage5 regression = 206 / 206 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.7 PART 1/2 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.6 continuity       = PASS"
    )

    print(
        "timing provenance         = AUDITED"
    )

    print(
        "Tbeam                     = PENDING EXACT FREEZE"
    )

    print(
        "Tframe                    = PENDING EXACT FREEZE"
    )

    print(
        "latency instrumentation   = AUDITED"
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "Stage5 regression         = 206 / 206 PASS"
    )

    print(
        "STATUS = PASS_READ_ONLY_AUDIT"
    )

    print(
        "next = BLOCK 5.7 PART 2/2"
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

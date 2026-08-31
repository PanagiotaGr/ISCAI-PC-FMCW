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

FILES = (
    STAGE5
    / "src/iscai_stage5/"
      "adaptive_topk.py",

    STAGE5
    / "tests/"
      "test_block55_adaptive_topk.py",

    STAGE5
    / "scripts/"
      "run_block55_part1_gate.py",

    STAGE5
    / "scripts/"
      "run_block55_part1_safe.py",
)

GATE = (
    STAGE5
    / "scripts/"
      "run_block55_part1_gate.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block55_part1_adaptive_topk_core.json"
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
        "BLOCK 5.5 PART 1/2 SAFE CONTROLLER"
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


def run_tests(
    pattern,
):
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
            pattern,
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


def stream(
    script,
):
    process = subprocess.Popen(
        [
            sys.executable,
            str(
                script
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
        "BLOCK 5.5 PART 1/2 — SAFE EXECUTION"
    )
    print(
        "============================================================"
    )

    missing = [
        str(
            path
        )
        for path in FILES
        if not path.is_file()
    ]

    if missing:

        blocked(
            "required_files",
            (
                "Missing Block5.5 Part1 "
                "file(s): "
                +
                ", ".join(
                    missing
                )
            ),
        )

        return

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in FILES:

        try:
            py_compile.compile(
                str(
                    path
                ),
                doraise=True,
            )

            print(
                path.name,
                "= PASS",
            )

        except Exception as exc:

            blocked(
                "compile",
                (
                    f"{path.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            return

    print()
    print(
        "===== BLOCK5.5 PART1 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block55_adaptive_topk.py"
    )

    if (
        code != 0
        or
        count != 20
    ):

        blocked(
            "Part1_tests",
            (
                "Expected Block5.5 Part1 "
                f"20/20; got {count}."
            ),
        )

        return

    print(
        "Block5.5 Part1 tests = 20 / 20 PASS"
    )

    print()
    print(
        "===== PRE-GATE FULL STAGE5 REGRESSION ====="
    )

    code, count = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 166
    ):

        blocked(
            "pre_gate_regression",
            (
                "Expected Stage5 regression "
                f"166/166; got {count}."
            ),
        )

        return

    print(
        "pre-gate Stage5 regression = 166 / 166 PASS"
    )

    print()
    print(
        "===== BLOCK5.5 PART1 SCIENTIFIC GATE ====="
    )

    raw_code = stream(
        GATE
    )

    print()
    print(
        "Block5.5 gate raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.5 Part1 report "
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
        "PASS"
    ):

        blocked(
            "scientific_gate",
            (
                "Block5.5 Part1 structured "
                "status is not PASS."
            ),
        )

        return

    scope = report.get(
        "scope",
        {}
    )

    if scope.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "formal_leakage",
            (
                "Block5.5 Part1 does not "
                "prove formal_N120_read=False."
            ),
        )

        return

    if scope.get(
        "hysteresis_started"
    ) is not False:

        blocked(
            "scope_drift",
            (
                "Temporal hysteresis started "
                "inside Part1."
            ),
        )

        return

    print()
    print(
        "===== FINAL FULL STAGE5 REGRESSION ====="
    )

    code, count = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 166
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 166/166; got {count}."
            ),
        )

        return

    print(
        "final Stage5 regression = 166 / 166 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.5 PART 1/2 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.4 continuity       = PASS"
    )

    print(
        "coverage targets          = 90 / 95 / 97.5 / 99 % FROZEN"
    )

    print(
        "nominal coverage          = 95 % FROZEN"
    )

    print(
        "adaptive selection        = SMALLEST MASS-COVERING SET"
    )

    print(
        "ranking                   = DESCENDING P_b"
    )

    print(
        "outside-support mass      = EXPLICIT"
    )

    print(
        "outside renormalization   = NO"
    )

    print(
        "Kmax API                  = SUPPORTED"
    )

    print(
        "numeric Kmax              = NOT YET FROZEN"
    )

    print(
        "hysteresis/fallback       = NOT STARTED"
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "Stage5 regression         = 166 / 166 PASS"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "next = BLOCK 5.5 PART 2/2"
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

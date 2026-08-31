from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

FILES = (
    STAGE5
    / "src/iscai_stage5/"
      "receiver_selection.py",

    STAGE5
    / "src/iscai_stage5/"
      "receiver_geometry.py",

    STAGE5
    / "tests/"
      "test_block51_receiver_kernels.py",

    STAGE5
    / "scripts/"
      "run_block51_part1_receiver_contract.py",

    STAGE5
    / "scripts/"
      "run_block51_part1_safe.py",
)

RUNNER = (
    STAGE5
    / "scripts/"
      "run_block51_part1_receiver_contract.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block51_part1_receiver_contract.json"
)

LOG = (
    STAGE5
    / "artifacts/block51/"
      "part1_safe.log"
)


def blocked(
    phase,
    reason,
    recovery,
):
    print()
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.1 PART 1/2 SAFE CONTROLLER"
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
        "recovery =",
        recovery,
    )

    print(
        "training/inference = NO"
    )

    print(
        "formal evaluation = NO"
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


def main():
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.1 PART 1/2 — SAFE EXECUTION"
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
                "Missing new Block5.1 file(s): "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only the missing "
                "Block5.1 Part1 file."
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
                (
                    "Repair only the new "
                    "Block5.1 Part1 code."
                ),
            )

            return

    print()
    print(
        "===== BLOCK5.1 PART1 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block51_*.py"
    )

    if (
        code != 0
        or
        count != 16
    ):
        blocked(
            "Block51_tests",
            (
                "Expected 16/16 Block5.1 "
                f"Part1 tests; detected {count}."
            ),
            (
                "Repair only receiver-selection/"
                "geometry kernels or their tests."
            ),
        )

        return

    print(
        "Block5.1 Part1 tests = 16 / 16 PASS"
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
        count != 28
    ):
        blocked(
            "pre_gate_regression",
            (
                "Expected Stage5 regression "
                f"28/28; detected {count}."
            ),
            (
                "Do not run the Block5.1 "
                "scientific gate."
            ),
        )

        return

    print(
        "pre-gate Stage5 regression = "
        "28 / 28 PASS"
    )

    print()
    print(
        "===== RECEIVER CONTRACT GATE ====="
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                RUNNER
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

    with LOG.open(
        "w",
        encoding="utf-8",
    ) as log:

        if process.stdout is not None:

            for line in process.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

    raw_code = process.wait()

    print()
    print(
        "receiver-contract raw code =",
        raw_code,
    )

    if not REPORT.is_file():
        blocked(
            "structured_report",
            (
                "Block5.1 Part1 report "
                "was not written."
            ),
            (
                "Inspect artifacts/block51/"
                "part1_safe.log."
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
            "report_read",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Repair only Block5.1 "
                "Part1 report generation."
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
                "Block5.1 Part1 structured "
                "status is not PASS."
            ),
            (
                "Inspect only the new receiver "
                "selection/geometry contract."
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
        count != 28
    ):
        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 28/28; got {count}."
            ),
            (
                "Do not proceed to "
                "Block5.1 Part2."
            ),
        )

        return

    print(
        "final Stage5 regression = "
        "28 / 28 PASS"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.1 PART 1/2 FINAL"
    )

    print(
        "============================================================"
    )

    print(
        "Block5.0 frozen continuity = PASS"
    )

    print(
        "receiver selector kernel    = PASS"
    )

    print(
        "receiver geometry kernel    = PASS"
    )

    print(
        "centroid/known/uncertain    = PASS"
    )

    print(
        "future/oracle leakage       = NONE"
    )

    print(
        "formal data used            = NO"
    )

    print(
        "training/inference          = NO"
    )

    print(
        "Stage5 regression           = 28 / 28 PASS"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "next = BLOCK 5.1 PART 2/2"
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
        (
            "Safe-controller failure only. "
            "No shell exit is issued."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

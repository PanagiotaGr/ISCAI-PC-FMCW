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
      "angular_posterior.py",

    STAGE5
    / "tests/"
      "test_block52_angular_kernel.py",

    STAGE5
    / "scripts/"
      "run_block52_part1_angular_contract.py",

    STAGE5
    / "scripts/"
      "run_block52_part1_safe.py",
)

RUNNER = (
    STAGE5
    / "scripts/"
      "run_block52_part1_angular_contract.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block52_part1_angular_contract.json"
)

LOG = (
    STAGE5
    / "artifacts/block52/"
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
        "BLOCK 5.2 PART 1/2 SAFE CONTROLLER"
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
        "Stage4 inference = NO"
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
        "BLOCK 5.2 PART 1/2 — SAFE EXECUTION"
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
                "Missing Block5.2 Part1 file(s): "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only missing "
                "Block5.2 Part1 files."
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
                    "Repair only new "
                    "Block5.2 Part1 code."
                ),
            )

            return

    print()
    print(
        "===== BLOCK5.2 PART1 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block52_angular_kernel.py"
    )

    if (
        code != 0
        or
        count != 14
    ):

        blocked(
            "Block52_tests",
            (
                "Expected 14/14 Block5.2 "
                f"Part1 tests; detected {count}."
            ),
            (
                "Repair only the angular "
                "posterior mathematical kernel."
            ),
        )

        return

    print(
        "Block5.2 Part1 tests = 14 / 14 PASS"
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
        count != 54
    ):

        blocked(
            "pre_gate_regression",
            (
                "Expected Stage5 regression "
                f"54/54; detected {count}."
            ),
            (
                "Do not run Block5.2 "
                "scientific gate."
            ),
        )

        return

    print(
        "pre-gate Stage5 regression = "
        "54 / 54 PASS"
    )

    print()
    print(
        "===== ANGULAR CONTRACT GATE ====="
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

    raw_code = (
        process.wait()
    )

    print()
    print(
        "angular-contract raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.2 Part1 report "
                "was not written."
            ),
            (
                "Inspect artifacts/block52/"
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
                "Repair only Block5.2 "
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
                "Block5.2 Part1 structured "
                "status is not PASS."
            ),
            (
                "Inspect only the new angular "
                "mapping/Jacobian implementation."
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
        count != 54
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 54/54; got {count}."
            ),
            (
                "Do not proceed to "
                "Block5.2 Part2."
            ),
        )

        return

    print(
        "final Stage5 regression = "
        "54 / 54 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 PART 1/2 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.1 frozen continuity = PASS"
    )

    print(
        "receiver→angular geometry   = PASS"
    )

    print(
        "analytic Jacobian           = PASS"
    )

    print(
        "analytic covariance path    = PASS"
    )

    print(
        "primary MC path             = NOT YET RUN"
    )

    print(
        "formal data used            = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "beam codebook               = NOT STARTED"
    )

    print(
        "Stage5 regression           = 54 / 54 PASS"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "next = BLOCK 5.2 PART 2/2"
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

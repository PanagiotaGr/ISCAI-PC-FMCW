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

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

FILES = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "reproducibility_probe_runtime.py",

    STAGE4
    / "tests/"
      "test_block49_fresh_probe.py",

    STAGE4
    / "scripts/"
      "run_block49_fresh_probe.py",

    STAGE4
    / "scripts/"
      "run_block49_reproducibility.py",
)

PART1 = (
    STAGE4
    / "reports/"
      "block49_part1_reproducibility_audit.json"
)

RUNNER = (
    STAGE4
    / "scripts/"
      "run_block49_reproducibility.py"
)

FINAL = (
    STAGE4
    / "reports/"
      "block49_reproducibility.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block49_reproducibility_failure.json"
)

LOG = (
    STAGE4
    / "artifacts/block49/"
      "part2_safe.log"
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
        "BLOCK 4.9 PART 2/2 SAFE CONTROLLER"
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
        "terminal remains open = YES"
    )


def main():
    print(
        "============================================================"
    )

    print(
        "BLOCK 4.9 PART 2/2 — SAFE CONTROLLER"
    )

    print(
        "============================================================"
    )

    required = (
        *FILES,
        PART1,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_files",
            (
                "Missing: "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only the missing "
                "Block4.9 Part2 file."
            ),
        )

        return

    part1 = json.loads(
        PART1.read_text(
            encoding="utf-8"
        )
    )

    if (
        part1.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            "Part1_status",
            "Block4.9 Part1 is not PASS.",
            (
                "Do not start fresh-process "
                "probes."
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
                    "Block4.9 Part2 file."
                ),
            )

            return

    print()
    print(
        "===== NEW PART2 UNIT TESTS ====="
    )

    new_tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4
                / "tests"
            ),
            "-p",
            "test_block49_fresh_probe.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    output = (
        new_tests.stdout
        +
        "\n"
        +
        new_tests.stderr
    )

    print(
        output
    )

    if (
        new_tests.returncode
        !=
        0
    ):
        blocked(
            "Part2_tests",
            (
                "New fresh-process "
                "reproducibility tests failed."
            ),
            (
                "Repair only new Block4.9 "
                "Part2 runtime/tests."
            ),
        )

        return

    print()
    print(
        "===== PRE-PROBE FULL REGRESSION ====="
    )

    full = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    full_output = (
        full.stdout
        +
        "\n"
        +
        full.stderr
    )

    print(
        full_output
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        full_output,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        full.returncode != 0
        or
        count != 124
    ):
        blocked(
            "pre_probe_regression",
            (
                "Expected 124/124 tests, "
                f"detected {count}."
            ),
            (
                "Do not run fresh-process "
                "probes until regression passes."
            ),
        )

        return

    print(
        "pre-probe regression = "
        "124 / 124 PASS"
    )

    print()
    print(
        "===== REAL FRESH-PROCESS AUDIT ====="
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                RUNNER
            ),
        ],
        cwd=str(
            STAGE4
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
        "reproducibility runner raw code =",
        raw_code,
    )

    if FINAL.is_file():
        try:
            report = json.loads(
                FINAL.read_text(
                    encoding="utf-8"
                )
            )

            if (
                report.get(
                    "status"
                )
                ==
                "PASS"
            ):
                print()
                print(
                    "============================================================"
                )

                print(
                    "BLOCK 4.9 PART 2/2 SAFE CONTROLLER"
                )

                print(
                    "============================================================"
                )

                print(
                    "scientific STATUS = PASS"
                )

                print(
                    "Block4.9 closure   = PASS"
                )

                print(
                    "terminal remains open = YES"
                )

                return

        except Exception:
            pass

    if FAILURE.is_file():
        try:
            failure = json.loads(
                FAILURE.read_text(
                    encoding="utf-8"
                )
            )

            blocked(
                "fresh_process_reproducibility",
                failure.get(
                    "exception"
                ),
                failure.get(
                    "recovery_hint"
                ),
            )

            return

        except Exception:
            pass

    blocked(
        "unclassified_result",
        (
            "Block4.9 runner ended "
            "without structured PASS."
        ),
        (
            "Inspect artifacts/block49/"
            "part2_safe.log and the individual "
            "fresh_probes/*.log files."
        ),
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

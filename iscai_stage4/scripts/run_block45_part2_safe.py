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

RUNTIME = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "calibration_runtime.py"
)

TEST = (
    STAGE4
    / "tests/"
      "test_block45_runtime.py"
)

RUNNER = (
    STAGE4
    / "scripts/"
      "run_block45_real_calibration.py"
)

PART1 = (
    STAGE4
    / "reports/"
      "block45_part1_preflight.json"
)

FINAL_REPORT = (
    STAGE4
    / "reports/"
      "block45_calibration.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block45_calibration_failure.json"
)

LOG = (
    STAGE4
    / "artifacts/block45/"
      "part2_real_calibration.log"
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
        "BLOCK 4.5 PART 2/2 "
        "SAFE CONTROLLER"
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
        "BLOCK 4.5 PART 2/2 — "
        "SAFE CONTROLLER"
    )
    print(
        "============================================================"
    )

    required = (
        RUNTIME,
        TEST,
        RUNNER,
        PART1,
    )

    missing = [
        str(path)
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
                "Block4.5 Part2 files."
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
            "part1_status",
            "Block4.5 Part1 is not PASS.",
            (
                "Do not begin full "
                "calibration yet."
            ),
        )
        return

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in (
        RUNTIME,
        TEST,
        RUNNER,
    ):
        try:
            py_compile.compile(
                str(path),
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
                    "Repair only this new "
                    "Block4.5 Part2 file. "
                    "No calibration scan "
                    "has started."
                ),
            )
            return

    print()
    print(
        "===== NEW PART2 UNIT TESTS ====="
    )

    test = subprocess.run(
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
            "test_block45_runtime.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    output = (
        test.stdout
        +
        "\n"
        +
        test.stderr
    )

    print(
        output
    )

    if test.returncode != 0:
        blocked(
            "runtime_unit_tests",
            "New Part2 tests failed.",
            (
                "Repair only "
                "calibration_runtime.py "
                "or the new Part2 test."
            ),
        )
        return

    print()
    print(
        "===== PRE-SCAN FULL REGRESSION ====="
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
        count != 76
    ):
        blocked(
            "pre_scan_regression",
            (
                "Expected 76/76 tests, "
                f"detected {count}."
            ),
            (
                "Do not start the 7209-scene "
                "scan until regression is PASS."
            ),
        )
        return

    print(
        "pre-scan regression = "
        "76 / 76 PASS"
    )

    print()
    print(
        "===== FULL REAL CALIBRATION ====="
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

    LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG.open(
        "w",
        encoding="utf-8",
    ) as log:
        if (
            process.stdout
            is not None
        ):
            for line in (
                process.stdout
            ):
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
        "real-calibration raw "
        "process code =",
        raw_code,
    )

    if FINAL_REPORT.is_file():
        try:
            report = json.loads(
                FINAL_REPORT
                .read_text(
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
                    "BLOCK 4.5 PART 2/2 "
                    "SAFE CONTROLLER"
                )
                print(
                    "============================================================"
                )
                print(
                    "scientific STATUS = PASS"
                )
                print(
                    "Block4.5 closure   = PASS"
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
                "real_calibration",
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
            "Real calibration ended "
            "without a structured "
            "PASS report."
        ),
        (
            "Inspect artifacts/block45/"
            "part2_real_calibration.log. "
            "Completed per-scene shards "
            "remain reusable."
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
            "Controller failure only. "
            "The terminal remains open "
            "and completed calibration "
            "shards remain reusable."
        ),
    )

    print()
    traceback.print_exc()

# No sys.exit().

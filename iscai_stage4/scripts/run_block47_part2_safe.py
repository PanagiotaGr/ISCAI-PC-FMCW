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

ANALYSIS = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "uncertainty_analysis.py"
)

TEST = (
    STAGE4
    / "tests/test_block47_uncertainty.py"
)

RUNNER = (
    STAGE4
    / "scripts/run_block47_ablation_training.py"
)

PART1 = (
    STAGE4
    / "reports/block47_part1_preflight.json"
)

FINAL_REPORT = (
    STAGE4
    / "reports/block47_ablation_analysis.json"
)

FAILURE = (
    STAGE4
    / "reports/block47_failure.json"
)

LOG = (
    STAGE4
    / "artifacts/block47/part2_run.log"
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
        "BLOCK 4.7 PART 2/2 SAFE CONTROLLER"
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
        "BLOCK 4.7 PART 2/2 — SAFE CONTROLLER"
    )
    print(
        "============================================================"
    )

    required = (
        ANALYSIS,
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
                "Block4.7 Part2 file."
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
            "Block4.7 Part1 is not PASS.",
            (
                "Do not begin ablation "
                "training."
            ),
        )
        return

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in (
        ANALYSIS,
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
                    "Block4.7 Part2 file."
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
            "test_block47_uncertainty.py",
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

    if new_tests.returncode != 0:
        blocked(
            "unit_tests",
            "New uncertainty tests failed.",
            (
                "Repair only the new "
                "Block4.7 analysis code/tests."
            ),
        )
        return

    print()
    print(
        "===== PRE-TRAIN FULL REGRESSION ====="
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
        count != 108
    ):
        blocked(
            "pre_training_regression",
            (
                "Expected 108/108 tests, "
                f"detected {count}."
            ),
            (
                "Do not start retraining "
                "until regression passes."
            ),
        )
        return

    print(
        "pre-training regression = 108 / 108 PASS"
    )

    print()
    print(
        "===== REAL BLOCK4.7 ANALYSIS/TRAINING ====="
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
        "Block4.7 runner raw process code =",
        raw_code,
    )

    if FINAL_REPORT.is_file():
        try:
            report = json.loads(
                FINAL_REPORT.read_text(
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
                    "BLOCK 4.7 PART 2/2 SAFE CONTROLLER"
                )
                print(
                    "============================================================"
                )
                print(
                    "scientific STATUS = PASS"
                )
                print(
                    "Block4.7 closure   = PASS"
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
                "Block47_analysis_training",
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
            "Block4.7 runner ended without "
            "a structured PASS report."
        ),
        (
            "Inspect artifacts/block47/"
            "part2_run.log. Completed ablation "
            "epochs remain resumable."
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

# No sys.exit().

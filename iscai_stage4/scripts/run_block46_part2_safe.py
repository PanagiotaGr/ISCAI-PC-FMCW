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
    / "src/iscai_stage4/ml/gmm_runtime.py"
)

TEST = (
    STAGE4
    / "tests/test_block46_runtime.py"
)

TRAINER = (
    STAGE4
    / "scripts/run_block46_gmm_training.py"
)

PART1 = (
    STAGE4
    / "reports/block46_part1_preflight.json"
)

FINAL_REPORT = (
    STAGE4
    / "reports/block46_gmm_gru.json"
)

FAILURE = (
    STAGE4
    / "reports/block46_training_failure.json"
)

TRAINING_LOG = (
    STAGE4
    / "artifacts/block46/part2_training.log"
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
        "BLOCK 4.6 PART 2/2 SAFE CONTROLLER"
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
        "BLOCK 4.6 PART 2/2 — SAFE CONTROLLER"
    )
    print(
        "============================================================"
    )

    required = (
        RUNTIME,
        TEST,
        TRAINER,
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
                "Block4.6 Part2 file."
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
            "Block4.6 Part1 is not PASS.",
            (
                "Do not begin GMM training "
                "until Part1 remains PASS."
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
        TRAINER,
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
                    "Repair only the new "
                    "Block4.6 Part2 file. "
                    "Training has not started."
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
            "test_block46_runtime.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    new_output = (
        new_tests.stdout
        +
        "\n"
        +
        new_tests.stderr
    )

    print(
        new_output
    )

    if (
        new_tests.returncode
        !=
        0
    ):
        blocked(
            "runtime_tests",
            "New GMM runtime tests failed.",
            (
                "Repair only gmm_runtime.py "
                "or its new tests."
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
        count != 92
    ):
        blocked(
            "pre_training_regression",
            (
                "Expected 92/92 tests, "
                f"detected {count}."
            ),
            (
                "Do not start GMM training "
                "until regression is PASS."
            ),
        )
        return

    print(
        "pre-training regression = "
        "92 / 92 PASS"
    )

    print()
    print(
        "===== REAL GMM TRAINING ====="
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                TRAINER
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

    TRAINING_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TRAINING_LOG.open(
        "w",
        encoding="utf-8",
    ) as log:
        if (
            process.stdout
            is not None
        ):
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
        "GMM trainer raw process code =",
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
                    "BLOCK 4.6 PART 2/2 SAFE CONTROLLER"
                )
                print(
                    "============================================================"
                )
                print(
                    "scientific STATUS = PASS"
                )
                print(
                    "Block4.6 closure   = PASS"
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
                "GMM_training",
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
            "GMM trainer ended without "
            "a structured PASS report."
        ),
        (
            "Inspect artifacts/block46/"
            "part2_training.log. "
            "Completed epochs remain resumable."
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
            "Safe-controller error only. "
            "No shell exit is issued."
        ),
    )

    print()
    traceback.print_exc()

# No sys.exit().

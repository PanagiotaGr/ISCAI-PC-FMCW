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
      "formal_runtime.py"
)

TEST = (
    STAGE4
    / "tests/"
      "test_block48_formal_runtime.py"
)

RUNNER = (
    STAGE4
    / "scripts/"
      "run_block48_formal_evaluation.py"
)

PART1 = (
    STAGE4
    / "reports/"
      "block48_part1_discovery.json"
)

FINAL = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation_failure.json"
)

LOG = (
    STAGE4
    / "artifacts/block48/"
      "part2_safe_run.log"
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
        "BLOCK 4.8 PART 2/2 SAFE CONTROLLER"
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
        "BLOCK 4.8 PART 2/2 — SAFE CONTROLLER"
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
                "Block4.8 Part2 file."
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
            "Block4.8 Part1 is not PASS.",
            (
                "Do not start formal inference."
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
                    "Block4.8 Part2 file. "
                    "Formal inference has "
                    "not started."
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
            "test_block48_formal_runtime.py",
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
            "formal_runtime_tests",
            "New formal-runtime tests failed.",
            (
                "Repair only formal_runtime.py "
                "or its new tests."
            ),
        )
        return

    print()
    print(
        "===== PRE-FORMAL FULL REGRESSION ====="
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

    output = (
        full.stdout
        +
        "\n"
        +
        full.stderr
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

    if (
        full.returncode != 0
        or
        count != 114
    ):
        blocked(
            "pre_formal_regression",
            (
                "Expected 114/114 tests, "
                f"detected {count}."
            ),
            (
                "Do not start formal inference "
                "until regression is PASS."
            ),
        )
        return

    print(
        "pre-formal regression = 114 / 114 PASS"
    )

    print()
    print(
        "===== FROZEN FORMAL N=120 RUN ====="
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
        "formal runner raw process code =",
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
                    "BLOCK 4.8 PART 2/2 SAFE CONTROLLER"
                )
                print(
                    "============================================================"
                )
                print(
                    "scientific STATUS = PASS"
                )
                print(
                    "Block4.8 evaluation = COMPLETE"
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
                "formal_evaluation",
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
            "Formal runner finished "
            "without structured PASS."
        ),
        (
            "Inspect artifacts/block48/"
            "part2_safe_run.log. Valid "
            "per-scenario formal shards "
            "remain resumable."
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

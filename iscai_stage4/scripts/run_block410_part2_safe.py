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

FINALIZER = (
    STAGE4
    / "scripts/"
      "run_block410_final_freeze.py"
)

PART1 = (
    STAGE4
    / "reports/"
      "block410_part1_acceptance_audit.json"
)

FINAL_CLOSURE = (
    STAGE4
    / "reports/"
      "stage4_final_closure.json"
)

HANDOFF = (
    STAGE4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

FREEZE_MANIFEST = (
    STAGE4
    / "artifacts/block410/"
      "stage4_final_freeze_manifest.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "stage4_final_closure_failure.json"
)

LOG = (
    STAGE4
    / "artifacts/block410/"
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
        "BLOCK 4.10 PART 2/2 SAFE CONTROLLER"
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
        "Stage5 started = NO"
    )

    print(
        "terminal remains open = YES"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.10 PART 2/2 — SAFE CONTROLLER"
    )
    print(
        "============================================================"
    )

    required = (
        FINALIZER,
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
                "Block4.10 Part2 file."
            ),
        )

        return

    try:
        part1 = json.loads(
            PART1.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:
        blocked(
            "Part1_read",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            "Do not freeze Stage4.",
        )

        return

    if (
        part1.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            "Part1_status",
            "Block4.10 Part1 is not PASS.",
            "Do not freeze Stage4.",
        )

        return

    matrix = part1.get(
        "acceptance_matrix",
        {}
    )

    if not (
        matrix.get(
            "closure_ready"
        )
        is True
        and
        matrix.get(
            "mandatory_pass_count"
        )
        ==
        14
        and
        matrix.get(
            "mandatory_count"
        )
        ==
        14
        and
        not matrix.get(
            "mandatory_failures"
        )
    ):
        blocked(
            "acceptance_matrix",
            (
                "Part1 mandatory acceptance "
                "is not exactly 14/14."
            ),
            "Do not weaken the closure gate.",
        )

        return

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in (
        FINALIZER,
        Path(
            __file__
        ),
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
                    "Repair only new "
                    "Block4.10 Part2 code."
                ),
            )

            return

    print()
    print(
        "===== PRE-FREEZE FULL REGRESSION ====="
    )

    process = subprocess.run(
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

    if (
        process.returncode != 0
        or
        count != 130
    ):
        blocked(
            "pre_freeze_regression",
            (
                "Expected 130/130 tests; "
                f"detected {count}."
            ),
            (
                "Do not write final "
                "Stage4 freeze artifacts."
            ),
        )

        return

    print(
        "pre-freeze regression = 130 / 130 PASS"
    )

    print()
    print(
        "===== CANONICAL FINAL FREEZE ====="
    )

    runner = subprocess.Popen(
        [
            sys.executable,
            str(
                FINALIZER
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

        if runner.stdout is not None:
            for line in runner.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

    raw_code = runner.wait()

    print()
    print(
        "final-freeze raw process code =",
        raw_code,
    )

    if (
        FINAL_CLOSURE.is_file()
        and
        HANDOFF.is_file()
        and
        FREEZE_MANIFEST.is_file()
    ):
        try:
            closure = json.loads(
                FINAL_CLOSURE.read_text(
                    encoding="utf-8"
                )
            )

            handoff = json.loads(
                HANDOFF.read_text(
                    encoding="utf-8"
                )
            )

            freeze = json.loads(
                FREEZE_MANIFEST.read_text(
                    encoding="utf-8"
                )
            )

            if (
                closure.get(
                    "status"
                )
                ==
                "COMPLETE_FROZEN"
                and
                handoff.get(
                    "status"
                )
                ==
                "FROZEN_HANDOFF"
                and
                freeze.get(
                    "status"
                )
                ==
                "FROZEN"
            ):
                print()
                print(
                    "============================================================"
                )
                print(
                    "BLOCK 4.10 PART 2/2 SAFE CONTROLLER"
                )
                print(
                    "============================================================"
                )

                print(
                    "scientific STATUS = COMPLETE_FROZEN"
                )

                print(
                    "Stage4 closure      = COMPLETE"
                )

                print(
                    "Stage4 implementation= FROZEN"
                )

                print(
                    "Stage5 handoff       = FROZEN"
                )

                print(
                    "Stage5 started       = NO"
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
                "final_freeze",
                failure.get(
                    "exception"
                ),
                failure.get(
                    "recovery"
                ),
            )

            return

        except Exception:
            pass

    blocked(
        "unclassified_result",
        (
            "Finalizer ended without "
            "three valid frozen artifacts."
        ),
        (
            "Inspect artifacts/block410/"
            "part2_safe.log. Do not retrain "
            "or rerun formal N=120."
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

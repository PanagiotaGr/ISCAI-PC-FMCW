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
      "__init__.py",

    STAGE5
    / "src/iscai_stage5/"
      "contracts.py",

    STAGE5
    / "tests/"
      "test_block50_contracts.py",

    STAGE5
    / "scripts/"
      "run_block50_contract_freeze.py",

    STAGE5
    / "scripts/"
      "run_block50_part2_safe.py",
)

PART1 = (
    STAGE5
    / "reports/"
      "block50_part1_discovery.json"
)

RUNNER = (
    STAGE5
    / "scripts/"
      "run_block50_contract_freeze.py"
)

FINAL = (
    STAGE5
    / "reports/"
      "block50_contract_freeze.json"
)

LOG = (
    STAGE5
    / "artifacts/block50/"
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
        "BLOCK 5.0 PART 2/2 SAFE CONTROLLER"
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
        output,
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.0 PART 2/2 — SAFE CONTROLLER"
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
                "new Stage5 Block5.0 file."
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
            (
                "Do not freeze Stage5 contract."
            ),
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
            "Block5.0 Part1 is not PASS.",
            (
                "Do not proceed until Part1 "
                "continuity is restored."
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
                    "Block5.0 Part2 code."
                ),
            )

            return

    print()
    print(
        "===== BLOCK5.0 CONTRACT TESTS ====="
    )

    (
        code,
        count,
        _,
    ) = run_tests(
        "test_block50_contracts.py"
    )

    if (
        code != 0
        or
        count != 12
    ):
        blocked(
            "contract_tests",
            (
                "Expected 12/12 Block5.0 "
                f"contract tests; detected {count}."
            ),
            (
                "Repair only the new "
                "Stage5 contract/tests."
            ),
        )

        return

    print(
        "Block5.0 tests = 12 / 12 PASS"
    )

    print()
    print(
        "===== PRE-FREEZE FULL STAGE5 REGRESSION ====="
    )

    (
        code,
        count,
        _,
    ) = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 12
    ):
        blocked(
            "pre_freeze_regression",
            (
                "Expected fresh Stage5 "
                f"regression 12/12; got {count}."
            ),
            (
                "Do not freeze contract until "
                "all Stage5 tests pass."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = "
        "12 / 12 PASS"
    )

    print()
    print(
        "===== EXACT CONTRACT FREEZE ====="
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
        "contract-freeze raw code =",
        raw_code,
    )

    if not FINAL.is_file():
        blocked(
            "structured_report",
            (
                "Block5.0 final report "
                "was not written."
            ),
            (
                "Inspect artifacts/block50/"
                "part2_safe.log."
            ),
        )

        return

    try:
        final = json.loads(
            FINAL.read_text(
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
                "Repair only Block5.0 "
                "report generation."
            ),
        )

        return

    if (
        final.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            "contract_freeze",
            (
                "Block5.0 final structured "
                "status is not PASS."
            ),
            (
                "Inspect the exact Part-A or "
                "contract invariant that failed."
            ),
        )

        return

    print()
    print(
        "===== FINAL FULL STAGE5 REGRESSION ====="
    )

    (
        code,
        count,
        _,
    ) = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 12
    ):
        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 12/12; got {count}."
            ),
            (
                "Do not proceed to Block5.1."
            ),
        )

        return

    print(
        "final Stage5 regression = "
        "12 / 12 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.0 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 discovery             = PASS"
    )

    print(
        "Part2 contract freeze       = PASS"
    )

    print(
        "Stage5 structural contract  = FROZEN"
    )

    print(
        "Part-A source contract      = FROZEN"
    )

    print(
        "frozen upstream continuity  = PASS"
    )

    print(
        "formal tuning               = NO"
    )

    print(
        "training/inference          = NO"
    )

    print(
        "Stage5 regression           = 12 / 12 PASS"
    )

    print(
        "scientific STATUS           = PASS"
    )

    print(
        "Block5.0                    = COMPLETE"
    )

    print(
        "next                        = BLOCK 5.1"
    )

    print(
        "terminal remains open       = YES"
    )


try:
    main()

except BaseException as exc:

    blocked(
        "unexpected_safe_controller_error",
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

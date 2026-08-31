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
      "receiver_policy.py",

    STAGE5
    / "src/iscai_stage5/"
      "stage3_receiver_bridge.py",

    STAGE5
    / "tests/"
      "test_block51_stage3_bridge.py",

    STAGE5
    / "scripts/"
      "run_block51_part2_freeze.py",

    STAGE5
    / "scripts/"
      "run_block51_part2_safe.py",
)

RUNNER = (
    STAGE5
    / "scripts/"
      "run_block51_part2_freeze.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block51_receiver_geometry.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "receiver_policy.json"
)

LOG = (
    STAGE5
    / "artifacts/block51/"
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
        "BLOCK 5.1 PART 2/2 SAFE CONTROLLER"
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
        "formal data used = NO"
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
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.1 PART 2/2 — SAFE EXECUTION"
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
                "Missing Block5.1 Part2 file(s): "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only the missing "
                "new Block5.1 Part2 file."
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
                    "Block5.1 Part2 code."
                ),
            )

            return

    print()
    print(
        "===== BLOCK5.1 PART2 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block51_stage3_bridge.py"
    )

    if (
        code != 0
        or
        count != 12
    ):
        blocked(
            "Part2_tests",
            (
                "Expected 12/12 Block5.1 "
                f"Part2 tests; detected {count}."
            ),
            (
                "Repair only receiver-policy/"
                "Stage3-bridge code or tests."
            ),
        )

        return

    print(
        "Block5.1 Part2 tests = 12 / 12 PASS"
    )

    print()
    print(
        "===== PRE-FREEZE FULL STAGE5 REGRESSION ====="
    )

    code, count = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 40
    ):
        blocked(
            "pre_freeze_regression",
            (
                "Expected Stage5 regression "
                f"40/40; detected {count}."
            ),
            (
                "Do not freeze Block5.1."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = "
        "40 / 40 PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.1 FREEZE ====="
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
        "Block5.1 freeze raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Final Block5.1 report "
                "was not written."
            ),
            (
                "Inspect artifacts/block51/"
                "part2_safe.log."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "policy_artifact",
            (
                "Frozen receiver policy "
                "was not written."
            ),
            (
                "Inspect only Block5.1 "
                "policy serialization."
            ),
        )

        return

    try:
        report = json.loads(
            REPORT.read_text(
                encoding="utf-8"
            )
        )

        policy = json.loads(
            POLICY.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        blocked(
            "artifact_readback",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Repair only Block5.1 "
                "serialization/readback."
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
                "Block5.1 report status "
                "is not PASS."
            ),
            (
                "Inspect the exact "
                "receiver-policy invariant."
            ),
        )

        return

    if (
        policy.get(
            "status"
        )
        !=
        "FROZEN"
        or
        policy.get(
            "max_planar_range_m"
        )
        is not None
    ):

        blocked(
            "policy_readback",
            (
                "Frozen receiver-policy "
                "readback mismatch."
            ),
            (
                "Do not proceed to Block5.2."
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
        count != 40
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 40/40; got {count}."
            ),
            (
                "Do not proceed to Block5.2."
            ),
        )

        return

    print(
        "final Stage5 regression = "
        "40 / 40 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.1 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 receiver kernels        = PASS"
    )

    print(
        "Part2 causal bridge           = PASS"
    )

    print(
        "Stage3 observation provenance = PASS"
    )

    print(
        "primary receiver policy       = FROZEN"
    )

    print(
        "receiver range cutoff         = NONE"
    )

    print(
        "range performance tuning      = NO"
    )

    print(
        "centroid/known/uncertain      = FROZEN CONTRACT"
    )

    print(
        "future/oracle leakage         = NONE"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "training/inference            = NO"
    )

    print(
        "Stage5 regression             = 40 / 40 PASS"
    )

    print(
        "scientific STATUS             = PASS"
    )

    print(
        "Block5.1                      = COMPLETE"
    )

    print(
        "next                          = BLOCK 5.2"
    )

    print(
        "terminal remains open         = YES"
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

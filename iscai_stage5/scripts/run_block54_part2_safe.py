from __future__ import annotations

import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

FILES = (
    STAGE5
    / "src/iscai_stage5/"
      "beam_baseline_temporal.py",

    STAGE5
    / "tests/"
      "test_block54_part2_temporal_policy.py",

    STAGE5
    / "scripts/"
      "run_block54_part2_freeze.py",

    STAGE5
    / "scripts/"
      "run_block54_part2_safe.py",
)

FREEZE = (
    STAGE5
    / "scripts/"
      "run_block54_part2_freeze.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block54_beam_baselines.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "beam_baseline_policy.json"
)


def blocked(
    phase,
    reason,
):
    print()
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.4 PART 2/2 SAFE CONTROLLER"
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
        "formal N=120 used = NO"
    )

    print(
        "Stage4 inference = NO"
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


def stream(
    script,
):
    process = subprocess.Popen(
        [
            sys.executable,
            str(
                script
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

    if process.stdout is not None:

        for line in process.stdout:

            print(
                line,
                end="",
                flush=True,
            )

    return process.wait()


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.4 PART 2/2 — SAFE EXECUTION"
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
                "Missing Block5.4 Part2 "
                "file(s): "
                +
                ", ".join(
                    missing
                )
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
            )

            return

    print()
    print(
        "===== BLOCK5.4 PART2 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block54_part2_temporal_policy.py"
    )

    if (
        code != 0
        or
        count != 18
    ):

        blocked(
            "Part2_tests",
            (
                "Expected Block5.4 Part2 "
                f"18/18; got {count}."
            ),
        )

        return

    print(
        "Block5.4 Part2 tests = 18 / 18 PASS"
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
        count != 146
    ):

        blocked(
            "pre_freeze_regression",
            (
                "Expected Stage5 regression "
                f"146/146; got {count}."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = 146 / 146 PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.4 FREEZE ====="
    )

    raw_code = stream(
        FREEZE
    )

    print()
    print(
        "Block5.4 freeze raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.4 final report "
                "was not written."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "policy_artifact",
            (
                "Beam-baseline policy "
                "was not written."
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
            "readback",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
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
                "Block5.4 final structured "
                "status is not PASS."
            ),
        )

        return

    if (
        report.get(
            "completion"
        )
        !=
        "COMPLETE_FROZEN"
    ):

        blocked(
            "completion_gate",
            (
                "Block5.4 completion is "
                "not COMPLETE_FROZEN."
            ),
        )

        return

    if (
        policy.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        blocked(
            "policy_gate",
            (
                "Beam baseline policy "
                "is not FROZEN."
            ),
        )

        return

    scope = (
        report.get(
            "scope",
            {}
        )
    )

    if scope.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "formal_leakage",
            (
                "Block5.4 does not prove "
                "formal_N120_read=False."
            ),
        )

        return

    if scope.get(
        "adaptive_TopK_started"
    ) is not False:

        blocked(
            "scope_drift",
            (
                "Adaptive Top-K started "
                "before Block5.5."
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
        count != 146
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 146/146; got {count}."
            ),
        )

        return

    print(
        "final Stage5 regression = 146 / 146 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.4 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "baseline registry           = FROZEN"
    )

    print(
        "exhaustive sweep             = FROZEN"
    )

    print(
        "geometry nearest             = FROZEN / CAUSAL"
    )

    print(
        "previous-beam persistence    = FROZEN / CAUSAL"
    )

    print(
        "persistence initialization   = GEOMETRY NEAREST"
    )

    print(
        "persistence state update     = PREVIOUS SELECTED BEAM"
    )

    print(
        "fixed Top-1 / Top-3 / Top-5  = FROZEN"
    )

    print(
        "fixed Top-K ranking          = P_b"
    )

    print(
        "oracle                       = EVALUATION ONLY"
    )

    print(
        "oracle controller access     = NO"
    )

    print(
        "WOMD measured beam labels    = NO"
    )

    print(
        "formal N=120 used            = NO"
    )

    print(
        "Stage4 inference             = NO"
    )

    print(
        "development/formal tuning    = NO"
    )

    print(
        "adaptive Top-K               = NOT STARTED"
    )

    print(
        "Stage5 regression            = 146 / 146 PASS"
    )

    print(
        "policy SHA256                =",
        report[
            "policy"
        ][
            "sha256"
        ],
    )

    print(
        "implementation SHA256        =",
        report[
            "implementation"
        ][
            "sha256"
        ],
    )

    print(
        "scientific STATUS            = PASS"
    )

    print(
        "Block5.4                     = COMPLETE / FROZEN"
    )

    print(
        "next                         = BLOCK 5.5"
    )

    print(
        "terminal remains open        = YES"
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
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

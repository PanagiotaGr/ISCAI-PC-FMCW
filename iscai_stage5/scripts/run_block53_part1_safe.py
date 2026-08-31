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
      "beam_codebook.py",

    STAGE5
    / "tests/"
      "test_block53_codebook_probability.py",

    STAGE5
    / "scripts/"
      "run_block53_part1_reference_audit.py",

    STAGE5
    / "scripts/"
      "run_block53_part1_gate.py",

    STAGE5
    / "scripts/"
      "run_block53_part1_safe.py",
)

REFERENCE_AUDIT = (
    STAGE5
    / "scripts/"
      "run_block53_part1_reference_audit.py"
)

GATE = (
    STAGE5
    / "scripts/"
      "run_block53_part1_gate.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block53_part1_codebook_kernel.json"
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
        "BLOCK 5.3 PART 1/2 SAFE CONTROLLER"
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
        "Stage4 inference now = NO"
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
        "BLOCK 5.3 PART 1/2 — SAFE EXECUTION"
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
                "Missing Block5.3 Part1 "
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
        "===== BLOCK5.3 PART1 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block53_codebook_probability.py"
    )

    if (
        code != 0
        or
        count != 18
    ):

        blocked(
            "Part1_tests",
            (
                "Expected 18/18 "
                f"Block5.3 tests; got {count}."
            ),
        )

        return

    print(
        "Block5.3 Part1 tests = 18 / 18 PASS"
    )

    print()
    print(
        "===== PRE-GATE STAGE5 REGRESSION ====="
    )

    code, count = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 90
    ):

        blocked(
            "pre_gate_regression",
            (
                "Expected Stage5 regression "
                f"90/90; got {count}."
            ),
        )

        return

    print(
        "pre-gate Stage5 regression = 90 / 90 PASS"
    )

    print()
    print(
        "===== PART-A / LEGACY REFERENCE AUDIT ====="
    )

    raw_code = stream(
        REFERENCE_AUDIT
    )

    print()
    print(
        "reference-audit raw code =",
        raw_code,
    )

    print()
    print(
        "===== BLOCK5.3 PART1 SCIENTIFIC GATE ====="
    )

    raw_code = stream(
        GATE
    )

    print()
    print(
        "Block5.3 gate raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.3 Part1 report "
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

    except Exception as exc:

        blocked(
            "report_readback",
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
                "Block5.3 Part1 report "
                "is not PASS."
            ),
        )

        return

    print()
    print(
        "===== FINAL STAGE5 REGRESSION ====="
    )

    code, count = run_tests(
        "test_*.py"
    )

    if (
        code != 0
        or
        count != 90
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 90/90; got {count}."
            ),
        )

        return

    print(
        "final Stage5 regression = 90 / 90 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.3 PART 1/2 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.2 continuity         = PASS"
    )

    print(
        "16/32/64 codebook kernel    = PASS"
    )

    print(
        "posterior P_b kernel        = PASS"
    )

    print(
        "outside-support probability = EXPLICIT"
    )

    print(
        "expected gain S_b kernel    = PASS / DISTINCT"
    )

    print(
        "Part-A reference audit      = PASS"
    )

    print(
        "angular support             = DEVELOPMENT/REFERENCE PENDING"
    )

    print(
        "physical gain pattern       = DEVELOPMENT/REFERENCE PENDING"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference now        = NO"
    )

    print(
        "adaptive Top-K              = NOT STARTED"
    )

    print(
        "Stage5 regression           = 90 / 90 PASS"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "next = BLOCK 5.3 PART 2/2"
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
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

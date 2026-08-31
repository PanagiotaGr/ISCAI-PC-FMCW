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
      "beam_directional_gain.py",

    STAGE5
    / "tests/"
      "test_block53_part2_frozen_codebook.py",

    STAGE5
    / "scripts/"
      "run_block53_part2_reference_decision.py",

    STAGE5
    / "scripts/"
      "run_block53_part2_freeze.py",

    STAGE5
    / "scripts/"
      "run_block53_part2_safe.py",
)

REFERENCE = (
    STAGE5
    / "scripts/"
      "run_block53_part2_reference_decision.py"
)

# BLOCK53_REFERENCE_STRUCTURED_GATE_V2
REFERENCE_REPORT = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_decision.json"
)

FREEZE = (
    STAGE5
    / "scripts/"
      "run_block53_part2_freeze.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block53_codebook_probability.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "beam_codebook_policy.json"
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
        "BLOCK 5.3 PART 2/2 SAFE CONTROLLER"
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
        "BLOCK 5.3 PART 2/2 — SAFE EXECUTION"
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
                "Missing Block5.3 Part2 "
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
        "===== BLOCK5.3 PART2 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block53_part2_frozen_codebook.py"
    )

    if (
        code != 0
        or
        count != 18
    ):

        blocked(
            "Part2_tests",
            (
                "Expected Block5.3 Part2 "
                f"18/18; got {count}."
            ),
        )

        return

    print(
        "Block5.3 Part2 tests = 18 / 18 PASS"
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
        count != 108
    ):

        blocked(
            "pre_freeze_regression",
            (
                "Expected Stage5 regression "
                f"108/108; got {count}."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = 108 / 108 PASS"
    )

    print()
    print(
        "===== PART-A REFERENCE DECISION ====="
    )

    raw_code = stream(
        REFERENCE
    )

    print()
    print(
        "reference-decision raw code =",
        raw_code,
    )

    #
    # The child script intentionally never terminates the
    # shell with a non-zero exit status. Therefore raw_code
    # alone is NOT a scientific success signal.
    #
    # Require the structured PASS artifact before allowing
    # the final freeze to run.
    #
    if not REFERENCE_REPORT.is_file():

        blocked(
            "reference_structured_report",
            (
                "Part-A reference decision "
                "report was not written."
            ),
        )

        return

    try:
        reference_report = json.loads(
            REFERENCE_REPORT.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        blocked(
            "reference_report_readback",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return

    if (
        reference_report.get(
            "status"
        )
        !=
        "PASS"
    ):

        blocked(
            "reference_scientific_gate",
            (
                "Part-A reference decision "
                "structured status is not PASS."
            ),
        )

        return

    scientific = (
        reference_report.get(
            "scientific_execution",
            {}
        )
    )

    if scientific.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "reference_formal_leakage",
            (
                "Reference decision does not "
                "prove formal_N120_read=False."
            ),
        )

        return

    if scientific.get(
        "Stage4_inference"
    ) is not False:

        blocked(
            "reference_Stage4_inference",
            (
                "Reference decision unexpectedly "
                "used Stage4 inference."
            ),
        )

        return

    print(
        "reference structured gate = PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.3 FREEZE ====="
    )

    raw_code = stream(
        FREEZE
    )

    print()
    print(
        "Block5.3 freeze raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.3 final report "
                "was not written."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "frozen_policy",
            (
                "Beam codebook policy "
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
                "Block5.3 final report "
                "is not PASS."
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
                "Block5.3 completion "
                "is not COMPLETE_FROZEN."
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
                "Beam-codebook policy "
                "is not FROZEN."
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
        count != 108
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 108/108; got {count}."
            ),
        )

        return

    print(
        "final Stage5 regression = 108 / 108 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.3 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "16/32/64 codebooks          = FROZEN"
    )

    print(
        "azimuth support             = [-12,+12] deg FROZEN"
    )

    print(
        "support provenance          = CONSTRUCTED STAGE5 ASSUMPTION"
    )

    print(
        "Part-A source status        = VISUALIZATION ASSUMPTION ONLY"
    )

    print(
        "decision widths             = 1.5 / 0.75 / 0.375 deg"
    )

    print(
        "P_b posterior mass          = FROZEN"
    )

    print(
        "outside-support probability = EXPLICIT"
    )

    print(
        "S_b expected gain           = FROZEN / DISTINCT"
    )

    print(
        "gain model                  = CONSTRUCTED GAUSSIAN"
    )

    print(
        "HPBW                        = DECISION CELL WIDTH"
    )

    print(
        "final optical Gopt          = SEPARATE / BLOCK5.6"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "development/formal tuning   = NO"
    )

    print(
        "adaptive Top-K              = NOT STARTED"
    )

    print(
        "Stage5 regression           = 108 / 108 PASS"
    )

    print(
        "policy SHA256               =",
        report[
            "policy"
        ][
            "sha256"
        ],
    )

    print(
        "implementation SHA256       =",
        report[
            "implementation"
        ][
            "sha256"
        ],
    )

    print(
        "scientific STATUS           = PASS"
    )

    print(
        "Block5.3                    = COMPLETE / FROZEN"
    )

    print(
        "next                        = BLOCK 5.4"
    )

    print(
        "terminal remains open       = YES"
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

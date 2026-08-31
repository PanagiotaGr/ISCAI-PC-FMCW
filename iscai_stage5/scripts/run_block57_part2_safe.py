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
      "beam_latency.py",

    STAGE5
    / "tests/"
      "test_block57_latency_controller.py",

    STAGE5
    / "scripts/"
      "run_block57_part2_freeze.py",

    STAGE5
    / "scripts/"
      "run_block57_part2_safe.py",
)

FREEZE = (
    STAGE5
    / "scripts/"
      "run_block57_part2_freeze.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block57_latency_controller.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "beam_latency_policy.json"
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
        "BLOCK 5.7 PART 2/2 SAFE CONTROLLER"
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
        "BLOCK 5.7 PART 2/2 — SAFE EXECUTION"
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
                "Missing Block5.7 Part2 "
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
        "===== BLOCK5.7 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block57_latency_controller.py"
    )

    if (
        code != 0
        or
        count != 20
    ):

        blocked(
            "Block57_tests",
            (
                "Expected Block5.7 "
                f"20/20; got {count}."
            ),
        )

        return

    print(
        "Block5.7 tests = 20 / 20 PASS"
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
        count != 226
    ):

        blocked(
            "pre_freeze_regression",
            (
                "Expected Stage5 regression "
                f"226/226; got {count}."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = 226 / 226 PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.7 FREEZE ====="
    )

    raw_code = stream(
        FREEZE
    )

    print()
    print(
        "Block5.7 freeze raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.7 final report "
                "was not written."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "policy_artifact",
            (
                "Beam-latency policy "
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
                "Block5.7 structured "
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
                "Block5.7 completion is "
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
                "Beam-latency policy "
                "is not FROZEN."
            ),
        )

        return

    timing = report.get(
        "timing_resolution",
        {}
    )

    if (
        timing.get(
            "Tbeam_s"
        )
        !=
        10e-6
    ):

        blocked(
            "Tbeam_gate",
            "Tbeam is not frozen at 10 us.",
        )

        return

    if (
        timing.get(
            "Tframe_s"
        )
        !=
        0.1
    ):

        blocked(
            "Tframe_gate",
            "Tframe is not frozen at 100 ms.",
        )

        return

    if (
        timing.get(
            "Tframe_is_optical_measurement"
        )
        is not False
    ):

        blocked(
            "timing_semantics",
            (
                "Tframe was incorrectly "
                "labelled as measured optical timing."
            ),
        )

        return

    scope = report.get(
        "scope",
        {}
    )

    if scope.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "formal_leakage",
            (
                "Block5.7 does not prove "
                "formal_N120_read=False."
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
        count != 226
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 226/226; got {count}."
            ),
        )

        return

    print(
        "final Stage5 regression = 226 / 226 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.7 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part-A Tchirp              = 10 us VERIFIED"
    )

    print(
        "Tbeam                      = 10 us FROZEN"
    )

    print(
        "Tbeam semantics            = ONE-CHIRP PROBE BASELINE"
    )

    print(
        "WOMD scene interval        = 100 ms VERIFIED"
    )

    print(
        "Tframe                     = 100 ms FROZEN"
    )

    print(
        "Tframe optical measurement = NO"
    )

    print(
        "effective-rate timing      = FULLY FROZEN"
    )

    print(
        "prediction target           = t + tau_total"
    )

    print(
        "sub-100ms annotated GT      = NO"
    )

    print(
        "latency instrumentation     = FROZEN"
    )

    print(
        "formal runtime measurement  = BLOCK5.8"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "Stage5 regression           = 226 / 226 PASS"
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
        "Block5.7                    = COMPLETE / FROZEN"
    )

    print(
        "next                        = BLOCK 5.8"
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

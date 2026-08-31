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
      "optical_link.py",

    STAGE5
    / "tests/"
      "test_block56_optical_link.py",

    STAGE5
    / "scripts/"
      "run_block56_part2_freeze.py",

    STAGE5
    / "scripts/"
      "run_block56_part2_safe.py",
)

FREEZE = (
    STAGE5
    / "scripts/"
      "run_block56_part2_freeze.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block56_optical_link.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "optical_link_policy.json"
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
        "BLOCK 5.6 PART 2/2 SAFE CONTROLLER"
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
        "BLOCK 5.6 PART 2/2 — SAFE EXECUTION"
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
                "Missing Block5.6 Part2 "
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
        "===== BLOCK5.6 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block56_optical_link.py"
    )

    if (
        code != 0
        or
        count != 20
    ):

        blocked(
            "Block56_tests",
            (
                "Expected Block5.6 "
                f"20/20; got {count}."
            ),
        )

        return

    print(
        "Block5.6 tests = 20 / 20 PASS"
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
        count != 206
    ):

        blocked(
            "pre_freeze_regression",
            (
                "Expected Stage5 regression "
                f"206/206; got {count}."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = 206 / 206 PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.6 FREEZE ====="
    )

    raw_code = stream(
        FREEZE
    )

    print()
    print(
        "Block5.6 freeze raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Block5.6 final report "
                "was not written."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "policy_artifact",
            (
                "Optical-link policy "
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
                "Block5.6 structured "
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
                "Block5.6 completion is "
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
                "Optical-link policy "
                "is not FROZEN."
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
                "Block5.6 does not prove "
                "formal_N120_read=False."
            ),
        )

        return

    if (
        report[
            "headlamp_frame"
        ][
            "second_extrinsic"
        ]
        is not False
    ):

        blocked(
            "frame_error",
            (
                "A second headlamp "
                "extrinsic was introduced."
            ),
        )

        return

    if (
        report[
            "effective_rate"
        ][
            "formula_frozen"
        ]
        is not True
    ):

        blocked(
            "effective_rate_gate",
            (
                "Effective-rate formula "
                "is not frozen."
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
        count != 206
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 206/206; got {count}."
            ),
        )

        return

    print(
        "final Stage5 regression = 206 / 206 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.6 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "H0 transmitter/headlamp frame = FROZEN / RESOLVED"
    )

    print(
        "second headlamp extrinsic      = NO"
    )

    print(
        "Part-A raw rate                = 1 Gbps"
    )

    print(
        "Part-A AWGN reference SNR      = 20 dB"
    )

    print(
        "received power                 = NORMALIZED PART-A UNITS"
    )

    print(
        "absolute Watt claim            = NO"
    )

    print(
        "pointing -> Gopt               = FROZEN"
    )

    print(
        "Gopt                           = CONSTRUCTED STAGE5 2-D GAUSSIAN"
    )

    print(
        "Gopt -> PRX -> SNR             = FROZEN"
    )

    print(
        "SNR -> DPSK BER                = FROZEN"
    )

    print(
        "effective-rate formula         = FROZEN"
    )

    print(
        "Tbeam/Tframe numeric values    = BLOCK5.7"
    )

    print(
        "formal N=120 used              = NO"
    )

    print(
        "Stage4 inference               = NO"
    )

    print(
        "Stage5 regression              = 206 / 206 PASS"
    )

    print(
        "policy SHA256                  =",
        report[
            "policy"
        ][
            "sha256"
        ],
    )

    print(
        "implementation SHA256          =",
        report[
            "implementation"
        ][
            "sha256"
        ],
    )

    print(
        "scientific STATUS              = PASS"
    )

    print(
        "Block5.6                       = COMPLETE / FROZEN"
    )

    print(
        "next                           = BLOCK 5.7"
    )

    print(
        "terminal remains open          = YES"
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

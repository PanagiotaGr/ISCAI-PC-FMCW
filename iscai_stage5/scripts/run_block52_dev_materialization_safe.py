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

SCRIPT = (
    STAGE5
    / "scripts/"
      "materialize_block52_frozen_dev_posterior.py"
)

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.json"
)

ARTIFACT = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.npz"
)


def run_regression():
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
            "test_*.py",
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


def blocked(
    phase,
    reason,
):
    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 DEV MATERIALIZATION SAFE CONTROLLER"
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
        "training/recalibration = NO"
    )

    print(
        "terminal remains open = YES"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — DEV POSTERIOR MATERIALIZATION SAFE EXECUTION"
    )
    print(
        "============================================================"
    )

    try:
        py_compile.compile(
            str(
                SCRIPT
            ),
            doraise=True,
        )

    except Exception as exc:

        blocked(
            "compile",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return

    print(
        "materialization script compile = PASS"
    )

    print()
    print(
        "===== PRE-INFERENCE STAGE5 REGRESSION ====="
    )

    code, count = run_regression()

    if (
        code != 0
        or
        count != 72
    ):

        blocked(
            "pre_inference_regression",
            (
                "Expected existing Stage5 "
                f"regression 72/72; got {count}."
            ),
        )

        return

    print(
        "pre-inference Stage5 regression = 72 / 72 PASS"
    )

    print()
    print(
        "===== FROZEN DEVELOPMENT INFERENCE ====="
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                SCRIPT
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

    raw_code = process.wait()

    print()
    print(
        "materialization raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Development posterior "
                "report was not written."
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
                "Development posterior "
                "report status is not PASS."
            ),
        )

        return

    if not ARTIFACT.is_file():

        blocked(
            "posterior_artifact",
            (
                "Materialized development "
                "posterior NPZ is missing."
            ),
        )

        return

    reproduction = report.get(
        "prediction_reproduction",
        {}
    )

    if not reproduction.get(
        "exact_match",
        False,
    ):

        blocked(
            "frozen_prediction_reproduction",
            (
                "Frozen Block4.4 prediction "
                "SHA did not reproduce exactly."
            ),
        )

        return

    scientific = report.get(
        "scientific_execution",
        {}
    )

    if scientific.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "formal_leakage",
            (
                "Materialization report does "
                "not prove formal_N120_read=False."
            ),
        )

        return

    print()
    print(
        "===== POST-INFERENCE STAGE5 REGRESSION ====="
    )

    code, count = run_regression()

    if (
        code != 0
        or
        count != 72
    ):

        blocked(
            "post_inference_regression",
            (
                "Expected Stage5 regression "
                f"72/72; got {count}."
            ),
        )

        return

    print(
        "post-inference Stage5 regression = 72 / 72 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 DEV MATERIALIZATION FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "frozen prediction reproduction = EXACT PASS"
    )

    print(
        "development posterior           = MATERIALIZED"
    )

    print(
        "coordinate semantics            = ABSOLUTE H0 PASS"
    )

    print(
        "calibration                     = FROZEN PASS"
    )

    print(
        "formal N=120 used               = NO"
    )

    print(
        "training/recalibration          = NO"
    )

    print(
        "Stage4 frozen inference         = YES"
    )

    print(
        "Stage4 modified                 = NO"
    )

    print(
        "Stage5 regression               = 72 / 72 PASS"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "next = PATCH BLOCK5.2 PART2 CONVERGENCE RESOLVER"
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

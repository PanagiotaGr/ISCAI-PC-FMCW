from __future__ import annotations

import json
from pathlib import Path
import py_compile
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

REPORT = (
    STAGE4
    / "reports/"
      "block44_part1_safe_gate.json"
)

ORIGINAL_PREFLIGHT = (
    STAGE4
    / "scripts/"
      "run_block44_preflight.py"
)

LOG = (
    STAGE4
    / "artifacts/block44/"
      "preflight_safe.log"
)


FILES_TO_COMPILE = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "gaussian_gru.py",

    STAGE4
    / "src/iscai_stage4/ml/"
      "gaussian_math.py",

    STAGE4
    / "src/iscai_stage4/ml/"
      "__init__.py",

    STAGE4
    / "tests/"
      "test_block44_gaussian_gru.py",

    ORIGINAL_PREFLIGHT,
)


def write_report(
    payload,
):
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )


def blocked(
    *,
    phase,
    exception=None,
    detail=None,
    recovery=None,
):
    if exception is not None:
        exception_type = (
            type(exception).__name__
        )

        exception_text = str(
            exception
        )

        trace = (
            traceback.format_exc()
        )

    else:
        exception_type = None
        exception_text = detail
        trace = None

    payload = {
        "stage": 4,
        "block": "4.4_part_1",
        "status": "BLOCKED",

        "phase":
            phase,

        "exception_type":
            exception_type,

        "exception":
            exception_text,

        "traceback":
            trace,

        "recovery":
            recovery,

        "block43_modified":
            False,

        "probabilistic_training_started":
            False,

        "shell_exit_policy":
            "ALWAYS_ZERO",
    }

    write_report(
        payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.4 PART 1/2"
    )
    print(
        "============================================================"
    )

    print(
        "STATUS = BLOCKED"
    )

    print(
        "phase  =",
        phase,
    )

    print(
        "reason =",
        exception_text,
    )

    print()
    print(
        "RECOVERY:"
    )

    print(
        recovery
    )

    print()
    print(
        "Block4.3 modified      = NO"
    )

    print(
        "Gaussian training      = NOT STARTED"
    )

    print(
        "shell process status   = SAFE / EXIT 0"
    )

    print(
        "report =",
        REPORT,
    )


def main():
    # --------------------------------------------------------
    # A. Required file presence
    # --------------------------------------------------------

    missing = [
        str(path)
        for path in FILES_TO_COMPILE
        if not path.is_file()
    ]

    if missing:
        blocked(
            phase="required_files",
            detail=(
                "Missing new Block4.4 files: "
                +
                ", ".join(
                    missing
                )
            ),
            recovery=(
                "The previous Block4.4 command "
                "did not finish writing all new "
                "files. Recreate only the missing "
                "Block4.4 files; do not modify "
                "Block4.3 or regenerate caches."
            ),
        )

        return

    print(
        "===== A. REQUIRED FILES ====="
    )

    print(
        "Block4.4 files present = PASS"
    )

    # --------------------------------------------------------
    # B. Compile every new file individually.
    # --------------------------------------------------------

    print()
    print(
        "===== B. CONTROLLED COMPILE ====="
    )

    for path in FILES_TO_COMPILE:
        try:
            py_compile.compile(
                str(path),
                doraise=True,
            )

            print(
                path.name,
                "= PASS",
            )

        except Exception as exc:
            blocked(
                phase=(
                    "compile:"
                    +
                    path.name
                ),
                exception=exc,
                recovery=(
                    "Repair only the syntax in "
                    f"{path}. No upstream model, "
                    "cache, normalization or "
                    "checkpoint needs to rerun."
                ),
            )

            return

    print(
        "controlled compile = PASS"
    )

    # --------------------------------------------------------
    # C. Import smoke.
    # --------------------------------------------------------

    print()
    print(
        "===== C. IMPORT SMOKE ====="
    )

    try:
        from iscai_stage4.ml import (
            GaussianTrajectoryGRU,
            covariance_from_scale_tril,
            denormalize_gaussian,
            empirical_coverage,
            initialize_from_deterministic,
            masked_gaussian_nll,
        )

        _ = (
            GaussianTrajectoryGRU,
            covariance_from_scale_tril,
            denormalize_gaussian,
            empirical_coverage,
            initialize_from_deterministic,
            masked_gaussian_nll,
        )

    except Exception as exc:
        blocked(
            phase="import",
            exception=exc,
            recovery=(
                "Repair only Block4.4 imports/"
                "exports. Do not install guessed "
                "packages and do not change the "
                "frozen Block4.3 environment."
            ),
        )

        return

    print(
        "Gaussian imports = PASS"
    )

    # --------------------------------------------------------
    # D. Run ONLY the new Block4.4 unit test first.
    #
    # This catches Gaussian math problems separately
    # before running the entire 60-test suite.
    # --------------------------------------------------------

    print()
    print(
        "===== D. BLOCK4.4 UNIT TEST ====="
    )

    test44 = subprocess.run(
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
            "test_block44_gaussian_gru.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    test44_output = (
        test44.stdout
        +
        "\n"
        +
        test44.stderr
    )

    print(
        test44_output
    )

    if (
        test44.returncode
        !=
        0
    ):
        blocked(
            phase="block44_unit_tests",
            detail=(
                "New Gaussian unit tests "
                "did not pass."
            ),
            recovery=(
                "Use the test traceback printed "
                "immediately above. Repair only "
                "gaussian_gru.py, gaussian_math.py "
                "or the new Block4.4 test itself. "
                "Block4.3 remains frozen."
            ),
        )

        return

    print(
        "Block4.4 unit tests = PASS"
    )

    # --------------------------------------------------------
    # E. Full regression.
    # --------------------------------------------------------

    print()
    print(
        "===== E. FULL STAGE4 REGRESSION ====="
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

    full_output = (
        full.stdout
        +
        "\n"
        +
        full.stderr
    )

    print(
        full_output
    )

    if full.returncode != 0:
        blocked(
            phase="full_regression",
            detail=(
                "Full Stage4 regression "
                "contains a failing test."
            ),
            recovery=(
                "The failing test output is "
                "printed above. Do not alter "
                "frozen 4.0–4.3 logic merely "
                "to make a new 4.4 test pass."
            ),
        )

        return

    import re

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        full_output,
    )

    test_count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if test_count != 60:
        blocked(
            phase="regression_count",
            detail=(
                "Expected exactly 60 tests, "
                f"but detected {test_count}."
            ),
            recovery=(
                "Check whether the previous "
                "Block4.4 attempt created an "
                "extra duplicate test file or "
                "failed to create one. Do not "
                "change scientific code until "
                "the test inventory is known."
            ),
        )

        return

    print(
        "full Stage4 regression = "
        "60 / 60 PASS"
    )

    # --------------------------------------------------------
    # F. Run the scientific GPU/math preflight.
    #
    # Its non-zero return code is captured here.
    # It cannot terminate the shell.
    # --------------------------------------------------------

    print()
    print(
        "===== F. SCIENTIFIC GPU PREFLIGHT ====="
    )

    preflight = subprocess.run(
        [
            sys.executable,
            str(
                ORIGINAL_PREFLIGHT
            ),
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    preflight_output = (
        preflight.stdout
        +
        "\n"
        +
        preflight.stderr
    )

    LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG.write_text(
        preflight_output,
        encoding="utf-8",
    )

    print(
        preflight_output
    )

    if (
        preflight.returncode
        !=
        0
    ):
        # Read the original structured failure report,
        # if that runner managed to produce one.
        original_failure = (
            STAGE4
            / "reports/"
              "block44_preflight_failure.json"
        )

        extra = None

        if original_failure.is_file():
            try:
                extra = json.loads(
                    original_failure
                    .read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                extra = None

        reason = (
            extra.get(
                "exception"
            )
            if isinstance(
                extra,
                dict
            )
            else
            (
                "Scientific Block4.4 "
                "preflight returned "
                f"{preflight.returncode}."
            )
        )

        recovery = (
            extra.get(
                "recovery_hint"
            )
            if isinstance(
                extra,
                dict
            )
            else
            (
                "Inspect the scientific "
                "preflight output printed above. "
                "No training or cache regeneration "
                "has occurred."
            )
        )

        blocked(
            phase="scientific_preflight",
            detail=reason,
            recovery=recovery,
        )

        return

    # --------------------------------------------------------
    # G. Confirm scientific preflight report says PASS.
    # --------------------------------------------------------

    scientific_report = (
        STAGE4
        / "reports/"
          "block44_preflight.json"
    )

    if not scientific_report.is_file():
        blocked(
            phase="scientific_report",
            detail=(
                "Preflight process returned zero "
                "but block44_preflight.json "
                "was not created."
            ),
            recovery=(
                "Inspect the saved preflight log. "
                "Do not start Part2 until a "
                "structured PASS report exists."
            ),
        )

        return

    scientific = json.loads(
        scientific_report.read_text(
            encoding="utf-8"
        )
    )

    if (
        scientific.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            phase="scientific_report_status",
            detail=(
                "Scientific report status "
                "is not PASS."
            ),
            recovery=(
                "Inspect block44_preflight.json. "
                "Part2 must not start yet."
            ),
        )

        return

    # --------------------------------------------------------
    # PASS
    # --------------------------------------------------------

    payload = {
        "stage": 4,
        "block": "4.4_part_1",
        "status": "PASS",

        "compile":
            True,

        "imports":
            True,

        "block44_unit_tests":
            True,

        "full_stage4_regression": {
            "passed": 60,
            "total": 60,
        },

        "scientific_preflight":
            True,

        "probabilistic_training_started":
            False,

        "block43_modified":
            False,

        "shell_exit_policy":
            "ALWAYS_ZERO",
    }

    write_report(
        payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.4 PART 1/2 SAFE GATE"
    )
    print(
        "============================================================"
    )

    print(
        "compile                   = PASS"
    )

    print(
        "Gaussian imports          = PASS"
    )

    print(
        "Block4.4 unit tests       = PASS"
    )

    print(
        "full Stage4 regression    = 60 / 60 PASS"
    )

    print(
        "GPU/math preflight        = PASS"
    )

    print(
        "Block4.3 modified         = NO"
    )

    print(
        "probabilistic training    = NOT STARTED"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "shell process status      = SAFE / EXIT 0"
    )

    print(
        "report =",
        REPORT,
    )


if __name__ == "__main__":
    try:
        main()

    except BaseException as exc:
        # Last-resort guard.
        #
        # Even an unexpected Python-side failure
        # becomes a controlled BLOCKED report
        # rather than a bare shell exit 1/2.
        blocked(
            phase="unexpected_top_level",
            exception=exc,
            recovery=(
                "Unexpected safe-runner failure. "
                "The traceback is preserved in "
                "block44_part1_safe_gate.json. "
                "No probabilistic training has "
                "started and Block4.3 is untouched."
            ),
        )

    # Deliberately no sys.exit(1/2).
    # Shell exit code remains zero.

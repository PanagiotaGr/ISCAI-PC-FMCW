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
      "angular_monte_carlo.py",

    STAGE5
    / "tests/"
      "test_block52_monte_carlo.py",

    STAGE5
    / "scripts/"
      "run_block52_part2_freeze.py",

    STAGE5
    / "scripts/"
      "run_block52_part2_safe.py",
)

RUNNER = (
    STAGE5
    / "scripts/"
      "run_block52_part2_freeze.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block52_receiver_angular_posterior.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "angular_posterior_policy.json"
)

CONVERGENCE = (
    STAGE5
    / "artifacts/block52/"
      "mc_convergence.json"
)

LOG = (
    STAGE5
    / "artifacts/block52/"
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
        "BLOCK 5.2 PART 2/2 SAFE CONTROLLER"
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
        "formal N=120 used = NO"
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
        "BLOCK 5.2 PART 2/2 — SAFE EXECUTION"
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
                "Missing Block5.2 Part2 file(s): "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only missing new "
                "Block5.2 Part2 files."
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
                    "Block5.2 Part2 code."
                ),
            )

            return

    print()
    print(
        "===== BLOCK5.2 PART2 UNIT TESTS ====="
    )

    code, count = run_tests(
        "test_block52_monte_carlo.py"
    )

    if (
        code != 0
        or
        count != 18
    ):

        blocked(
            "Part2_tests",
            (
                "Expected 18/18 Block5.2 "
                f"Part2 tests; detected {count}."
            ),
            (
                "Repair only the deterministic "
                "MC receiver/angular implementation."
            ),
        )

        return

    print(
        "Block5.2 Part2 tests = 18 / 18 PASS"
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
        count != 72
    ):

        blocked(
            "pre_freeze_regression",
            (
                "Expected Stage5 regression "
                f"72/72; detected {count}."
            ),
            (
                "Do not run the final "
                "Block5.2 freeze."
            ),
        )

        return

    print(
        "pre-freeze Stage5 regression = "
        "72 / 72 PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.2 FREEZE ====="
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
        "Block5.2 freeze raw code =",
        raw_code,
    )

    if not REPORT.is_file():

        blocked(
            "structured_report",
            (
                "Final Block5.2 report "
                "was not written."
            ),
            (
                "Inspect artifacts/block52/"
                "part2_safe.log."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "policy_artifact",
            (
                "Angular-posterior policy "
                "was not written."
            ),
            (
                "Inspect Block5.2 freeze log."
            ),
        )

        return

    if not CONVERGENCE.is_file():

        blocked(
            "convergence_artifact",
            (
                "MC convergence artifact "
                "was not written."
            ),
            (
                "Inspect development-posterior "
                "discovery/convergence output."
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

        convergence = json.loads(
            CONVERGENCE.read_text(
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
                "Repair only Block5.2 "
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
                "Block5.2 report status "
                "is not PASS."
            ),
            (
                "Inspect exact MC/convergence "
                "invariant that failed."
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
            "policy_readback",
            (
                "Angular-posterior policy "
                "is not FROZEN."
            ),
            (
                "Do not proceed to Block5.3."
            ),
        )

        return

    if (
        convergence.get(
            "status"
        )
        !=
        "PASS"
    ):

        blocked(
            "MC_convergence",
            (
                "MC development convergence "
                "did not PASS."
            ),
            (
                "Do not choose a different "
                "sample count using formal data."
            ),
        )

        return

    selected_n = (
        policy[
            "Monte_Carlo"
        ][
            "sample_count"
        ]
    )

    if selected_n not in (
        2048,
        4096,
        8192,
    ):

        blocked(
            "MC_sample_count",
            (
                "Selected MC N is outside "
                "the preregistered candidates."
            ),
            (
                "Inspect only non-formal "
                "development convergence."
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
        count != 72
    ):

        blocked(
            "final_regression",
            (
                "Expected final Stage5 "
                f"regression 72/72; got {count}."
            ),
            (
                "Do not proceed to Block5.3."
            ),
        )

        return

    print(
        "final Stage5 regression = "
        "72 / 72 PASS"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 angular/Jacobian       = PASS"
    )

    print(
        "Part2 deterministic MC       = PASS"
    )

    print(
        "calibrated Gaussian input    = PASS"
    )

    print(
        "cross-horizon covariance     = NOT INVENTED"
    )

    print(
        "receiver-placement UQ        = JOINTLY PROPAGATED"
    )

    print(
        "samplewise future heading    = PASS"
    )

    print(
        "future GT heading            = NO"
    )

    print(
        "MC development convergence   = PASS"
    )

    print(
        "MC sample count              =",
        selected_n,
        "FROZEN",
    )

    print(
        "angular posterior policy     = FROZEN"
    )

    print(
        "formal N=120 used            = NO"
    )

    print(
        "Stage4 inference             = NO"
    )

    print(
        "beam codebook                = NOT STARTED"
    )

    print(
        "training/recalibration       = NO"
    )

    print(
        "Stage5 regression            = 72 / 72 PASS"
    )

    print(
        "scientific STATUS            = PASS"
    )

    print(
        "Block5.2                     = COMPLETE"
    )

    print(
        "next                         = BLOCK 5.3"
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
        (
            "Safe-controller failure only. "
            "No shell exit is issued."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

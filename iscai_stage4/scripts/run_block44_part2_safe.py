from __future__ import annotations

import json
from pathlib import Path
import py_compile
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
STAGE4 = ROOT / "iscai_stage4"

TRAINER = (
    STAGE4
    / "scripts/run_block44_gaussian_training.py"
)

PART1 = (
    STAGE4
    / "reports/block44_part1_safe_gate.json"
)

PREFLIGHT = (
    STAGE4
    / "reports/block44_preflight.json"
)

FINAL_REPORT = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

FAILURE = (
    STAGE4
    / "reports/block44_training_failure.json"
)

LOG = (
    STAGE4
    / "artifacts/block44/part2_training.log"
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
        "BLOCK 4.4 PART 2/2 SAFE CONTROLLER"
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
        "terminal remains open = YES"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.4 PART 2/2 — SAFE CONTROLLER"
    )
    print(
        "============================================================"
    )

    required = (
        TRAINER,
        PART1,
        PREFLIGHT,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_files",
            (
                "Missing: "
                +
                ", ".join(missing)
            ),
            (
                "Do not start training. "
                "Restore only the missing "
                "Block4.4 file/report."
            ),
        )
        return

    part1 = json.loads(
        PART1.read_text(
            encoding="utf-8"
        )
    )

    preflight = json.loads(
        PREFLIGHT.read_text(
            encoding="utf-8"
        )
    )

    if (
        part1.get("status")
        !=
        "PASS"
    ):
        blocked(
            "part1_status",
            "Block4.4 Part1 is not PASS.",
            (
                "Part2 must not run until "
                "Part1 is PASS."
            ),
        )
        return

    if (
        preflight.get("status")
        !=
        "PASS"
    ):
        blocked(
            "preflight_status",
            (
                "Scientific Gaussian "
                "preflight is not PASS."
            ),
            (
                "Do not train until the "
                "GPU/math preflight passes."
            ),
        )
        return

    try:
        py_compile.compile(
            str(TRAINER),
            doraise=True,
        )

    except Exception as exc:
        blocked(
            "compile",
            repr(exc),
            (
                "Repair only the new "
                "Block4.4 Part2 trainer. "
                "No frozen cache/checkpoint "
                "needs to rerun."
            ),
        )
        return

    print(
        "trainer compile             = PASS"
    )
    print(
        "Block4.4 Part1              = PASS"
    )
    print(
        "Gaussian GPU/math preflight = PASS"
    )

    run = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
        ],
        cwd=str(STAGE4),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    LOG.write_text(
        run.stdout or "",
        encoding="utf-8",
    )

    if run.stdout:
        print(
            run.stdout,
            end="",
        )

    # Trainer catches runtime failures itself,
    # so scientific state is read from reports.
    if FINAL_REPORT.is_file():
        try:
            report = json.loads(
                FINAL_REPORT.read_text(
                    encoding="utf-8"
                )
            )

            if (
                report.get("status")
                ==
                "PASS"
            ):
                print()
                print(
                    "============================================================"
                )
                print(
                    "BLOCK 4.4 PART 2/2 SAFE CONTROLLER"
                )
                print(
                    "============================================================"
                )
                print(
                    "scientific STATUS = PASS"
                )
                print(
                    "Gaussian training  = PASS"
                )
                print(
                    "terminal remains open = YES"
                )
                return

        except Exception:
            pass

    if FAILURE.is_file():
        try:
            failure = json.loads(
                FAILURE.read_text(
                    encoding="utf-8"
                )
            )

            blocked(
                "gaussian_training",
                failure.get(
                    "exception"
                ),
                failure.get(
                    "recovery_hint"
                ),
            )
            return

        except Exception:
            pass

    blocked(
        "unclassified_training_result",
        (
            "Trainer finished without a "
            "structured PASS report."
        ),
        (
            "Inspect artifacts/block44/"
            "part2_training.log. "
            "No Stage2/3 cache regeneration "
            "is required."
        ),
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
            "The controller itself failed. "
            "The terminal remains open and "
            "Block4.3 is untouched."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

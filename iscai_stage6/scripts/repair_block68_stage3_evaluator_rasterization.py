from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


S6 = Path("/home/agni/waymo/iscai_stage6")

TARGET = (
    S6
    / "scripts/run_block68_stage3_fixed_cache_parity.py"
)

BACKUP = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "rasterization_repair/"
      "run_block68_stage3_fixed_cache_parity."
      "pre_half_cell_raster_fix.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_stage3_evaluator_rasterization_repair.json"
)

FINAL_CACHE_REPORT = (
    S6
    / "reports/"
      "block68_stage3_fixed_cache_parity.json"
)

FINAL_CACHE_FREEZE = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "stage3_fixed_cache_freeze.json"
)


EXPECTED_TARGET_SHA = (
    "5c4952829354d8f2dc090d514bc1144f"
    "1fb42b530214233add5601073791833f"
)

EXPECTED_TESTS = 224


OLD_FUNCTION = '''def raster_projected(
    projected,
):
    center = float(
        projected.theta_center_rad
    )

    span = float(
        projected.theta_span_rad
    )

    dtheta = np.arctan2(
        np.sin(
            THETA_CENTERS
            -
            center
        ),
        np.cos(
            THETA_CENTERS
            -
            center
        ),
    )

    theta_active = (
        np.abs(
            dtheta
        )
        <=
        (
            0.5
            *
            span
        )
    )

    range_active = (
        (
            RANGE_CENTERS
            >=
            float(
                projected.ground_range_min_m
            )
        )
        &
        (
            RANGE_CENTERS
            <=
            float(
                projected.ground_range_max_m
            )
        )
    )

    return (
        theta_active[
            :,
            None
        ]
        &
        range_active[
            None,
            :
        ]
    )
'''


NEW_FUNCTION = '''def raster_projected(
    projected,
):
    """
    Frozen Stage-1/Stage-2 evaluator rasterization.

    A grid cell belongs to the projected support when its
    cell extent overlaps the projected angular/radial interval.

    This is deliberately NOT cell-center-only rasterization.
    """

    center = float(
        projected.theta_center_rad
    )

    span = float(
        projected.theta_span_rad
    )

    theta_half_step = (
        0.5
        *
        float(
            THETA_CENTERS[
                1
            ]
            -
            THETA_CENTERS[
                0
            ]
        )
    )

    range_half_step = (
        0.5
        *
        float(
            RANGE_CENTERS[
                1
            ]
            -
            RANGE_CENTERS[
                0
            ]
        )
    )

    dtheta = np.arctan2(
        np.sin(
            THETA_CENTERS
            -
            center
        ),
        np.cos(
            THETA_CENTERS
            -
            center
        ),
    )

    theta_active = (
        np.abs(
            dtheta
        )
        <=
        (
            0.5
            *
            span
            +
            theta_half_step
        )
    )

    range_active = (
        (
            RANGE_CENTERS
            +
            range_half_step
        )
        >=
        float(
            projected.ground_range_min_m
        )
    ) & (
        (
            RANGE_CENTERS
            -
            range_half_step
        )
        <=
        float(
            projected.ground_range_max_m
        )
    )

    return (
        theta_active[
            :,
            None
        ]
        &
        range_active[
            None,
            :
        ]
    )
'''


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)

    os.replace(
        temporary,
        path,
    )


def run_command(command):
    process = subprocess.run(
        command,
        cwd=str(S6),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return (
        process.returncode,
        process.stdout,
    )


def regression():
    rc, output = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ]
    )

    count = None

    for line in output.splitlines():
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        rc == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                output.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


def main():

    print("=" * 76)
    print("BLOCK 6.8 STAGE-3 EVALUATOR RASTERIZATION REPAIR")
    print("RESTORE FROZEN HALF-CELL OVERLAP SEMANTICS")
    print("=" * 76)

    # --------------------------------------------------------
    # A. Exact current boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT CURRENT BOUNDARY =====")

    require(
        TARGET.is_file(),
        f"Missing target: {TARGET}",
    )

    pre_sha = file_sha(
        TARGET
    )

    require(
        pre_sha
        ==
        EXPECTED_TARGET_SHA,
        (
            "Target differs from exact post-loader-repair "
            "runner.\n"
            f"expected={EXPECTED_TARGET_SHA}\n"
            f"actual={pre_sha}"
        ),
    )

    require(
        not FINAL_CACHE_REPORT.exists(),
        (
            "Final fixed-cache PASS report already exists."
        ),
    )

    require(
        not FINAL_CACHE_FREEZE.exists(),
        (
            "Final fixed-cache freeze already exists."
        ),
    )

    print(
        "target SHA256           = EXACT PASS"
    )

    print(
        "posterior subset repair = PRESERVED"
    )

    print(
        "scenario-loader repair  = PRESERVED"
    )

    print(
        "Stage-1 gamma           = IMMUTABLE"
    )

    print(
        "Stage-2 margins         = IMMUTABLE"
    )

    print(
        "Stage-3 scoring         = IMMUTABLE"
    )

    print(
        "Stage-3 floor outcomes  = NOT COMPUTED"
    )

    print(
        "formal evaluation       = NO"
    )

    # --------------------------------------------------------
    # B. Exact mismatch diagnosis
    # --------------------------------------------------------

    print()
    print("===== B. RASTERIZATION MISMATCH DIAGNOSIS =====")

    original_bytes = TARGET.read_bytes()

    original_source = original_bytes.decode(
        "utf-8"
    )

    count = original_source.count(
        OLD_FUNCTION
    )

    require(
        count == 1,
        (
            "Expected exactly one cell-center-only "
            f"raster_projected function; found {count}."
        ),
    )

    require(
        "selected_posterior_keys"
        in
        original_source,
        (
            "Posterior subset repair not preserved."
        ),
    )

    require(
        "read_training_scenario("
        in
        original_source,
        (
            "Scenario-loader repair not preserved."
        ),
    )

    print(
        "current angular rule = CELL CENTER ONLY"
    )

    print(
        "current radial rule  = CELL CENTER ONLY"
    )

    print(
        "frozen angular rule  = HALF-CELL OVERLAP"
    )

    print(
        "frozen radial rule   = HALF-CELL OVERLAP"
    )

    print(
        "repair scope         = EVALUATOR RASTER SUPPORT ONLY"
    )

    # --------------------------------------------------------
    # C. Backup
    # --------------------------------------------------------

    print()
    print("===== C. VERIFIED BACKUP =====")

    BACKUP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BACKUP.exists():

        require(
            BACKUP.read_bytes()
            ==
            original_bytes,
            (
                "Existing backup differs from "
                "current pre-patch runner."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    require(
        file_sha(BACKUP)
        ==
        pre_sha,
        "Backup SHA mismatch.",
    )

    print(
        "backup SHA256 =",
        file_sha(BACKUP),
    )

    # --------------------------------------------------------
    # D. Minimal repair
    # --------------------------------------------------------

    print()
    print("===== D. BUILD MINIMAL RASTERIZATION PATCH =====")

    patched = original_source.replace(
        OLD_FUNCTION,
        NEW_FUNCTION,
        1,
    )

    require(
        "theta_half_step"
        in
        patched,
        (
            "Angular half-cell overlap not inserted."
        ),
    )

    require(
        "range_half_step"
        in
        patched,
        (
            "Radial half-cell overlap not inserted."
        ),
    )

    require(
        "selected_posterior_keys"
        in
        patched,
        (
            "Posterior repair accidentally removed."
        ),
    )

    require(
        "read_training_scenario("
        in
        patched,
        (
            "Scenario-loader repair accidentally removed."
        ),
    )

    compile(
        patched,
        str(TARGET),
        "exec",
    )

    print(
        "patch syntax              = PASS"
    )

    print(
        "Stage-1 k                = UNCHANGED"
    )

    print(
        "Stage-2 margins          = UNCHANGED"
    )

    print(
        "timestamp rule           = UNCHANGED"
    )

    print(
        "actor population         = UNCHANGED"
    )

    print(
        "future GT provenance     = UNCHANGED"
    )

    print(
        "metric formulas          = UNCHANGED"
    )

    print(
        "Stage-3 scoring contract = UNCHANGED"
    )

    # --------------------------------------------------------
    # E. Transactional patch + tests
    # --------------------------------------------------------

    print()
    print("===== E. TRANSACTIONAL PATCH =====")

    applied = False

    try:

        atomic_write(
            TARGET,
            patched.encode("utf-8"),
        )

        applied = True

        rc, output = run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(TARGET),
            ]
        )

        require(
            rc == 0,
            (
                "Patched runner compile failed:\n"
                +
                output
            ),
        )

        tests = regression()

    except BaseException:

        if applied:

            print()
            print(
                "PATCH FAILURE -> EXACT ROLLBACK"
            )

            atomic_write(
                TARGET,
                original_bytes,
            )

            require(
                TARGET.read_bytes()
                ==
                original_bytes,
                (
                    "Rollback failed."
                ),
            )

            print(
                "rollback = EXACT PASS"
            )

        raise

    post_sha = file_sha(
        TARGET
    )

    print(
        "patched runner compile = PASS"
    )

    print(
        "Stage6 regression      =",
        f"{tests} / {tests} PASS",
    )

    print(
        "post-patch SHA256      =",
        post_sha,
    )

    # --------------------------------------------------------
    # F. Repair report
    # --------------------------------------------------------

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_evaluator_rasterization_repair",

        "status":
            "PASS_STAGE3_EVALUATOR_RASTERIZATION_REPAIRED",

        "diagnosis": {
            "failure_type":
                "replay_semantics_mismatch",

            "observed_failure": {
                "metric":
                    "TYPE_VEHICLE Stage1 mean IoU",

                "actual_with_cell_center_raster":
                    0.43252373424510754,

                "frozen_target":
                    0.442779087781594,

                "absolute_delta":
                    0.010255353536486489,
            },

            "cause":
                (
                    "Part1 fixed-cache evaluator used "
                    "cell-center-only projected-box rasterization "
                    "instead of the frozen Stage1/Stage2 "
                    "cell-extent overlap rasterization"
                ),

            "repair":
                (
                    "restore theta half-cell expansion and "
                    "radial cell-interval overlap exactly"
                ),
        },

        "runner": {
            "path":
                str(TARGET),

            "pre_patch_sha256":
                pre_sha,

            "post_patch_sha256":
                post_sha,

            "backup_path":
                str(BACKUP),

            "backup_sha256":
                file_sha(BACKUP),

            "patch_scope":
                "raster_projected evaluator support only",
        },

        "preserved": {
            "posterior_selected_subset_gate":
                True,

            "authoritative_scenario_loader":
                True,

            "Stage1_gamma":
                True,

            "Stage2_margin":
                True,

            "timestamp_alignment":
                True,

            "future_GT_provenance":
                True,

            "Stage3_scoring_contract":
                True,
        },

        "scientific_boundary": {
            "Stage3_floor_outcomes_computed":
                False,

            "Stage3_floor_selected":
                False,

            "Stage4_temporal_rate_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "regression":
            tests,

        "next":
            (
                "rerun identical Stage3 execution Part1 "
                "and require Stage1 + Stage2 replay parity"
            ),
    }

    atomic_write(
        REPORT,
        canonical_bytes(report),
    )

    print()
    print("=" * 76)
    print("BLOCK 6.8 EVALUATOR RASTERIZATION REPAIR — FINAL")
    print("=" * 76)

    print(
        "cell-center-only raster = REMOVED"
    )

    print(
        "theta half-cell overlap = RESTORED"
    )

    print(
        "range half-cell overlap = RESTORED"
    )

    print(
        "posterior subset repair = PRESERVED"
    )

    print(
        "scenario-loader repair  = PRESERVED"
    )

    print(
        "Stage-1 gamma           = UNCHANGED"
    )

    print(
        "Stage-2 margins         = UNCHANGED"
    )

    print(
        "Stage-3 scoring         = UNCHANGED"
    )

    print(
        "Stage-3 floor outcomes  = NOT COMPUTED"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "Stage6 regression       =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_EVALUATOR_RASTERIZATION_REPAIRED"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:

    main()

except BaseException as exc:

    print()
    print("=" * 76)
    print(
        "BLOCK 6.8 EVALUATOR RASTERIZATION REPAIR = BLOCKED"
    )
    print("=" * 76)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma          = STILL FROZEN"
    )

    print(
        "Stage-2 margins        = STILL FROZEN"
    )

    print(
        "Stage-3 floor outcomes = NOT COMPUTED"
    )

    print(
        "formal evaluation      = NO"
    )

    print()
    print(
        "Do not change k, margins, timestamps, "
        "population or metric targets."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

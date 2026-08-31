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
    / "scripts"
    / "run_block68_stage3_floor_selection_part2of2.py"
)

BACKUP = (
    S6
    / "artifacts/block68/stage3_floor_selection/"
      "road_roi_shape_repair/"
      "run_block68_stage3_floor_selection_part2of2."
      "pre_road_roi_shape_fix.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_stage3_road_roi_schedule_shape_repair.json"
)

EXPECTED_PRE_SHA256 = (
    "0b351f8cd7f4d51ff259b5a9132f012f"
    "fe3b58ea3fdc420d912d2db76dd459e4"
)

EXPECTED_TESTS = 224


OLD = """    road_roi = np.asarray(
        frozen_road_roi(
            GRID_SHAPE
        ),
        dtype=bool,
    )
"""


NEW = """    # ----------------------------------------------------
    # Frozen road ROI is a SPATIAL [theta, range] object.
    #
    # Stage-3 candidate illumination is a four-horizon
    # [horizon, theta, range] schedule.  Therefore:
    #
    #   1. materialize the frozen spatial ROI using only
    #      (theta_cells, range_cells);
    #   2. replicate that exact same spatial support over
    #      every frozen horizon.
    #
    # This is representation lifting only.  No road cell,
    # metric formula, horizon weight, or scientific policy
    # is changed.
    # ----------------------------------------------------

    road_roi_spatial = np.asarray(
        frozen_road_roi(
            GRID_SHAPE[
                1:
            ]
        ),
        dtype=bool,
    )

    require(
        road_roi_spatial.shape
        ==
        GRID_SHAPE[
            1:
        ],
        (
            "Frozen spatial road ROI shape mismatch: "
            f"{road_roi_spatial.shape} "
            f"vs {GRID_SHAPE[1:]}"
        ),
    )

    road_roi_spatial_count = int(
        np.count_nonzero(
            road_roi_spatial
        )
    )

    require(
        road_roi_spatial_count
        >
        0,
        "Frozen spatial road ROI is empty.",
    )

    road_roi = np.broadcast_to(
        road_roi_spatial[
            None,
            ...,
        ],
        GRID_SHAPE,
    ).copy()

    require(
        road_roi.shape
        ==
        GRID_SHAPE,
        (
            "Schedule-domain road ROI shape mismatch: "
            f"{road_roi.shape} vs {GRID_SHAPE}"
        ),
    )

    for horizon_index in range(
        GRID_SHAPE[
            0
        ]
    ):
        require(
            np.array_equal(
                road_roi[
                    horizon_index
                ],
                road_roi_spatial,
            ),
            (
                "Frozen road ROI changed during horizon "
                f"lifting at index {horizon_index}."
            ),
        )

    require(
        int(
            np.count_nonzero(
                road_roi
            )
        )
        ==
        (
            int(
                GRID_SHAPE[
                    0
                ]
            )
            *
            road_roi_spatial_count
        ),
        (
            "Schedule-domain road ROI support count "
            "is not exact horizon replication."
        ),
    )
"""


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


def atomic_write(path: Path, payload: bytes):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)

    os.replace(
        temporary,
        path,
    )


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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


def run_regression():
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
            + "\n".join(
                output.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"observed {count}."
        ),
    )

    return count


def main():

    print("=" * 78)
    print("BLOCK 6.8 STAGE-3 ROAD-ROI SCHEDULE-SHAPE REPAIR")
    print("REPRESENTATION ADAPTER ONLY — SCIENTIFIC ROI UNCHANGED")
    print("=" * 78)

    # ========================================================
    # A. Exact pre-patch boundary
    # ========================================================

    print()
    print("===== A. EXACT PRE-PATCH BOUNDARY =====")

    require(
        TARGET.is_file(),
        f"Missing target runner: {TARGET}",
    )

    pre_bytes = TARGET.read_bytes()
    pre_sha = sha256(
        pre_bytes
    ).hexdigest()

    print(
        "runner pre-patch SHA256 =",
        pre_sha,
    )

    require(
        pre_sha
        ==
        EXPECTED_PRE_SHA256,
        (
            "Part2 runner SHA differs from the exact "
            "post-synthetic-repair state.\n"
            f"expected={EXPECTED_PRE_SHA256}\n"
            f"actual  ={pre_sha}"
        ),
    )

    source = pre_bytes.decode(
        "utf-8"
    )

    require(
        source.count(OLD)
        ==
        1,
        (
            "Expected exactly one frozen_road_roi("
            "GRID_SHAPE) assignment."
        ),
    )

    require(
        "road_roi_spatial"
        not in
        source,
        (
            "Road-ROI schedule adapter appears to "
            "already exist."
        ),
    )

    print(
        "synthetic parity repair = PRESERVED"
    )

    print(
        "Stage-1 gamma          = FROZEN / UNCHANGED"
    )

    print(
        "Stage-2 margins        = FROZEN / UNCHANGED"
    )

    print(
        "Stage-3 floor outcome  = NOT YET SELECTED"
    )

    print(
        "Stage-4 temporal/rate  = NOT SELECTED"
    )

    print(
        "primary acceptance     = NOT TESTED"
    )

    print(
        "formal evaluation      = NO"
    )

    # ========================================================
    # B. Exact diagnosis
    # ========================================================

    print()
    print("===== B. EXACT SHAPE DIAGNOSIS =====")

    print(
        "schedule GRID_SHAPE    = (4, 501, 301)"
    )

    print(
        "frozen_road_roi domain = (theta, range)"
    )

    print(
        "spatial ROI shape      = (501, 301)"
    )

    print(
        "metric schedule shape  = (4, 501, 301)"
    )

    print(
        "repair semantics       = replicate identical "
        "2-D frozen ROI over four horizons"
    )

    print(
        "road spatial support   = UNCHANGED"
    )

    print(
        "horizon weighting      = UNCHANGED / EQUAL"
    )

    # ========================================================
    # C. Backup
    # ========================================================

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
            pre_bytes,
            (
                "Existing backup differs from current "
                "exact pre-patch runner."
            ),
        )
    else:
        BACKUP.write_bytes(
            pre_bytes
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

    # ========================================================
    # D. Build exact patch
    # ========================================================

    print()
    print("===== D. BUILD REPRESENTATION-ONLY PATCH =====")

    patched = source.replace(
        OLD,
        NEW,
        1,
    )

    require(
        patched.count(
            "frozen_road_roi("
        )
        ==
        source.count(
            "frozen_road_roi("
        ),
        (
            "Number of frozen_road_roi calls changed."
        ),
    )

    require(
        "frozen_road_roi(\n"
        "            GRID_SHAPE[\n"
        "                1:\n"
        "            ]\n"
        "        )"
        in
        patched,
        (
            "2-D spatial road-ROI call was not inserted."
        ),
    )

    require(
        "np.broadcast_to("
        in
        patched,
        (
            "Schedule-domain ROI replication "
            "was not inserted."
        ),
    )

    compile(
        patched,
        str(TARGET),
        "exec",
    )

    print(
        "patch syntax             = PASS"
    )

    print(
        "evaluator_reference.py   = UNCHANGED"
    )

    print(
        "metric_semantics.py      = UNCHANGED"
    )

    print(
        "frozen spatial ROI       = UNCHANGED"
    )

    print(
        "Stage-3 comparator       = UNCHANGED"
    )

    print(
        "3270 candidate space     = UNCHANGED"
    )

    # ========================================================
    # E. Transactional patch
    # ========================================================

    print()
    print("===== E. TRANSACTIONAL PATCH + REGRESSION =====")

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
                "Patched Part2 runner compile failed:\n"
                + output
            ),
        )

        tests = run_regression()

    except BaseException:
        if applied:
            print()
            print(
                "PATCH FAILURE -> EXACT ROLLBACK"
            )

            atomic_write(
                TARGET,
                pre_bytes,
            )

            require(
                TARGET.read_bytes()
                ==
                pre_bytes,
                "Exact rollback failed.",
            )

            print(
                "rollback = EXACT PASS"
            )

        raise

    post_sha = file_sha(
        TARGET
    )

    require(
        post_sha != pre_sha,
        "Patched runner SHA did not change.",
    )

    print(
        "patched runner compile = PASS"
    )

    print(
        "Stage6 regression      =",
        f"{tests} / {tests} PASS",
    )

    print(
        "runner post-patch SHA  =",
        post_sha,
    )

    # ========================================================
    # F. Repair report
    # ========================================================

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_road_roi_schedule_shape_repair",

        "status":
            "PASS_STAGE3_ROAD_ROI_SCHEDULE_SHAPE_REPAIRED",

        "diagnosis": {
            "failure_type":
                "spatial_roi_vs_schedule_shape_mismatch",

            "scientific_failure":
                False,

            "frozen_road_roi_native_domain":
                [
                    "theta",
                    "range",
                ],

            "frozen_spatial_shape":
                [
                    501,
                    301,
                ],

            "schedule_shape":
                [
                    4,
                    501,
                    301,
                ],

            "repair":
                (
                    "materialize frozen 2-D spatial ROI "
                    "then replicate exactly across the "
                    "four frozen horizons"
                ),
        },

        "immutability": {
            "evaluator_reference_modified":
                False,

            "metric_semantics_modified":
                False,

            "spatial_road_roi_modified":
                False,

            "Stage1_gamma_modified":
                False,

            "Stage2_margins_modified":
                False,

            "Stage3_candidate_space_modified":
                False,

            "Stage3_scoring_comparator_modified":
                False,

            "Stage4_parameters_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "mathematical_equivalence": {
            "same_spatial_ROI_every_horizon":
                True,

            "horizon_count":
                4,

            "equal_horizon_weighting_preserved":
                True,

            "schedule_ROI_mean_equivalent_to":
                (
                    "equal mean of the four "
                    "per-horizon frozen spatial-ROI means"
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
                (
                    "road ROI materialization / "
                    "schedule-shape adapter only"
                ),
        },

        "regression":
            tests,

        "next":
            (
                "rerun the identical Stage3 Part2/2; "
                "fixed-cache readback must complete "
                "before the exact 3270 floor sweep"
            ),
    }

    atomic_write(
        REPORT,
        canonical_bytes(report),
    )

    print()
    print("=" * 78)
    print("BLOCK 6.8 ROAD-ROI SHAPE REPAIR — FINAL")
    print("=" * 78)

    print(
        "frozen road ROI native shape = (501, 301)"
    )

    print(
        "schedule ROI shape           = (4, 501, 301)"
    )

    print(
        "spatial cells changed        = 0"
    )

    print(
        "horizon replication          = EXACT"
    )

    print(
        "evaluator runtime            = UNCHANGED"
    )

    print(
        "metric runtime               = UNCHANGED"
    )

    print(
        "Stage-3 scoring comparator   = UNCHANGED"
    )

    print(
        "Stage-1 gamma                = UNCHANGED"
    )

    print(
        "Stage-2 margins              = UNCHANGED"
    )

    print(
        "Stage-3 floor selected       = NO"
    )

    print(
        "Stage-4 temporal/rate        = NOT SELECTED"
    )

    print(
        "primary acceptance           = NOT TESTED"
    )

    print(
        "formal evaluation            = NO"
    )

    print(
        "Stage6 regression            =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_ROAD_ROI_SCHEDULE_SHAPE_REPAIRED"
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
    print("=" * 78)
    print(
        "BLOCK 6.8 ROAD-ROI SHAPE REPAIR = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma         = STILL FROZEN"
    )

    print(
        "Stage-2 margins       = STILL FROZEN"
    )

    print(
        "Stage-3 floor freeze  = NOT COMPLETE"
    )

    print(
        "Stage-4 temporal/rate = NOT SELECTED"
    )

    print(
        "primary acceptance    = NOT TESTED"
    )

    print(
        "formal evaluation     = NO"
    )

    print()
    print(
        "Do NOT modify evaluator_reference.py, "
        "metric_semantics.py, Stage1/2 numerics, "
        "or the Stage3 scoring comparator."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

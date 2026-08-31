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
    / "scripts/"
      "run_block68_stage3_floor_selection_part2of2.py"
)

BACKUP = (
    S6
    / "artifacts/block68/stage3_floor_selection/"
      "synthetic_parity_repair/"
      "run_block68_stage3_floor_selection_part2of2."
      "pre_synthetic_fixture_fix.py"
)

REPAIR_REPORT = (
    S6
    / "reports/"
      "block68_stage3_synthetic_metric_fixture_repair.json"
)

FINAL_REPORT = (
    S6
    / "reports/"
      "block68_stage3_class_floor_selection.json"
)

FINAL_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage3_class_floor_freeze.json"
)

CANDIDATE_SCORES = (
    S6
    / "artifacts/block68/stage3_floor_selection/"
      "block68_stage3_floor_candidate_scores.jsonl"
)

WINNER_SCENARIOS = (
    S6
    / "artifacts/block68/stage3_floor_selection/"
      "block68_stage3_winner_authoritative_scenarios.jsonl"
)

EXPECTED_TESTS = 224


START_MARKER = (
    "def synthetic_metric_parity():\n"
)

END_MARKER = (
    "\n\n# ============================================================\n"
    "# CANDIDATE SCORE ACCUMULATION\n"
    "# ============================================================\n"
)


NEW_FUNCTION = r'''def synthetic_metric_parity():
    """
    Synthetic implementation-parity test on the ACTUAL
    frozen actuator-grid shape.

    Important:
    over_masking_area is normalized by the complete actuator
    grid. Therefore a tiny 2x3x4 synthetic tensor is not a
    valid fixture for the production compressed kernel whose
    frozen denominator is GRID_CELL_COUNT = 4*501*301.

    Keep production metric semantics unchanged; make the
    synthetic fixture conform to them.
    """

    shape = GRID_SHAPE

    require(
        int(
            np.prod(
                shape
            )
        )
        ==
        GRID_CELL_COUNT,
        (
            "Synthetic parity fixture must use the "
            "frozen production grid-cell denominator."
        ),
    )

    raw = np.ones(
        shape,
        dtype=np.float64,
    )

    # Sparse deterministic non-unity intensities.
    # Everything not listed here remains exactly 1.
    assignments = (
        ((0, 0, 0), 0.8),
        ((0, 0, 1), 0.6),
        ((0, 1, 0), 0.5),
        ((0, 1, 1), 0.2),
        ((0, 2, 2), 0.4),
        ((0, 2, 3), 0.7),
        ((1, 0, 0), 0.9),
        ((1, 0, 2), 0.3),
        ((1, 1, 1), 0.6),
        ((1, 1, 2), 0.5),
        ((1, 2, 0), 0.2),
        ((1, 2, 3), 0.8),
    )

    for index, value in assignments:
        raw[
            index
        ] = float(
            value
        )

    oracle_all = np.zeros(
        shape,
        dtype=bool,
    )

    oracle_all[
        0,
        0,
        0:2,
    ] = True

    oracle_all[
        1,
        0,
        0:3,
    ] = True

    oracle_vehicle = np.zeros(
        shape,
        dtype=bool,
    )

    oracle_vehicle[
        0,
        0,
        0,
    ] = True

    oracle_vehicle[
        1,
        0,
        0,
    ] = True

    ped = np.zeros(
        shape,
        dtype=bool,
    )

    ped[
        0,
        1,
        0:2,
    ] = True

    ped[
        1,
        1,
        1:3,
    ] = True

    cyc = np.zeros(
        shape,
        dtype=bool,
    )

    cyc[
        0,
        2,
        2:4,
    ] = True

    cyc[
        1,
        2,
        0,
    ] = True

    cyc[
        1,
        2,
        3,
    ] = True

    surrogate = np.zeros(
        shape,
        dtype=bool,
    )

    surrogate[
        0,
        0,
        0:2,
    ] = True

    surrogate[
        1,
        0,
        0:3,
    ] = True

    # Frozen road-illumination metric operates on a full-grid
    # boolean ROI. For the synthetic parity fixture, use all
    # actuator cells so both implementations aggregate the
    # same complete domain.
    road = np.ones(
        shape,
        dtype=bool,
    )

    authoritative = authoritative_metrics(
        raw,

        oracle_all=
            oracle_all,

        oracle_vehicle=
            oracle_vehicle,

        pedestrian_region=
            ped,

        cyclist_region=
            cyc,

        vehicle_surrogate=
            surrogate,

        road_roi=
            road,
    )

    # Include the entire production grid in the compressed
    # synthetic test. This intentionally makes the compressed
    # denominator identical to the authoritative metric's
    # frozen full-grid denominator.
    u = np.ones(
        shape,
        dtype=bool,
    )

    raw_u = raw.reshape(
        1,
        -1,
    )

    require(
        raw_u.shape
        ==
        (
            1,
            GRID_CELL_COUNT,
        ),
        (
            "Synthetic compressed raw shape "
            "does not equal frozen full grid."
        ),
    )

    fast = compressed_metrics(
        raw_u,

        oracle_all_u=
            oracle_all[
                u
            ],

        oracle_vehicle_u=
            oracle_vehicle[
                u
            ],

        pedestrian_u=
            ped[
                u
            ],

        cyclist_u=
            cyc[
                u
            ],

        vehicle_surrogate_u=
            surrogate[
                u
            ],

        road_u=
            road[
                u
            ],

        oracle_all_count=
            int(
                np.count_nonzero(
                    oracle_all
                )
            ),

        oracle_vehicle_count=
            int(
                np.count_nonzero(
                    oracle_vehicle
                )
            ),

        pedestrian_count=
            int(
                np.count_nonzero(
                    ped
                )
            ),

        cyclist_count=
            int(
                np.count_nonzero(
                    cyc
                )
            ),

        vehicle_surrogate_count=
            int(
                np.count_nonzero(
                    surrogate
                )
            ),

        road_count=
            int(
                np.count_nonzero(
                    road
                )
            ),
    )

    maximum_delta = 0.0

    for metric in METRIC_NAMES:

        authority_value = float(
            authoritative[
                metric
            ]
        )

        fast_value = float(
            fast[
                metric
            ][
                0
            ]
        )

        if (
            math.isnan(
                authority_value
            )
            and
            math.isnan(
                fast_value
            )
        ):
            delta = 0.0

        else:

            require(
                math.isfinite(
                    authority_value
                )
                ==
                math.isfinite(
                    fast_value
                ),
                (
                    "Synthetic metric-kernel "
                    "finiteness mismatch for "
                    f"{metric}: "
                    f"authority={authority_value}, "
                    f"fast={fast_value}"
                ),
            )

            delta = abs(
                authority_value
                -
                fast_value
            )

            require(
                delta
                <=
                PARITY_ABS_TOL,
                (
                    "Synthetic metric-kernel parity "
                    f"failed for {metric}: "
                    f"authority={authority_value}, "
                    f"fast={fast_value}, "
                    f"delta={delta}"
                ),
            )

        maximum_delta = max(
            maximum_delta,
            delta,
        )

    return maximum_delta
'''


def require(
    condition,
    message,
):
    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def file_sha(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024
                *
                1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_bytes(
    value,
):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_write(
    path: Path,
    payload: bytes,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_bytes(
        payload
    )

    os.replace(
        temporary,
        path,
    )


def run_command(
    command,
):
    process = subprocess.run(
        command,
        cwd=str(
            S6
        ),
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
                match.group(
                    1
                )
            )

    require(
        rc == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                output.splitlines()[
                    -100:
                ]
            )
        ),
    )

    require(
        count
        ==
        EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


def main():

    print("=" * 78)
    print("BLOCK 6.8 STAGE-3 SYNTHETIC METRIC-FIXTURE REPAIR")
    print("TEST FIXTURE ONLY — PRODUCTION METRIC KERNEL UNCHANGED")
    print("=" * 78)

    # --------------------------------------------------------
    # A. Scientific boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT FAILURE BOUNDARY =====")

    require(
        TARGET.is_file(),
        f"Missing target runner: {TARGET}",
    )

    require(
        not FINAL_REPORT.exists(),
        (
            "Final Stage-3 floor PASS report already exists."
        ),
    )

    require(
        not FINAL_FREEZE.exists(),
        (
            "Final Stage-3 floor freeze already exists."
        ),
    )

    require(
        not CANDIDATE_SCORES.exists(),
        (
            "Candidate-score output already exists; "
            "failed run should have stopped before sweep."
        ),
    )

    require(
        not WINNER_SCENARIOS.exists(),
        (
            "Winner scenario output already exists; "
            "failed run should have stopped before sweep."
        ),
    )

    source_bytes = TARGET.read_bytes()

    source = source_bytes.decode(
        "utf-8"
    )

    pre_sha = sha256(
        source_bytes
    ).hexdigest()

    print(
        "runner pre-patch SHA256 =",
        pre_sha,
    )

    print(
        "Stage-1 gamma           = FROZEN"
    )

    print(
        "Stage-2 margins         = FROZEN"
    )

    print(
        "Stage-3 floor outcomes  = NOT COMPUTED"
    )

    print(
        "candidate sweep         = NOT STARTED"
    )

    print(
        "Stage-4 temporal/rate   = NOT SELECTED"
    )

    print(
        "primary acceptance      = NOT TESTED"
    )

    print(
        "formal evaluation       = NO"
    )

    # --------------------------------------------------------
    # B. Exact diagnosis
    # --------------------------------------------------------

    print()
    print("===== B. SYNTHETIC DENOMINATOR DIAGNOSIS =====")

    require(
        source.count(
            START_MARKER
        )
        ==
        1,
        (
            "Expected exactly one "
            "synthetic_metric_parity function."
        ),
    )

    require(
        source.count(
            END_MARKER
        )
        ==
        1,
        (
            "Expected exactly one candidate-score "
            "section marker."
        ),
    )

    start = source.index(
        START_MARKER
    )

    end = source.index(
        END_MARKER,
        start,
    )

    old_function = source[
        start:end
    ]

    require(
        "shape = (\n        2,\n        3,\n        4,\n    )"
        in
        old_function,
        (
            "Expected 2x3x4 synthetic fixture "
            "not found."
        ),
    )

    # Production kernel MUST remain on frozen full-grid
    # overmask denominator.
    require(
        '''float(
            GRID_CELL_COUNT
        )'''
        in
        source,
        (
            "Frozen production full-grid denominator "
            "not found."
        ),
    )

    expected_fast = (
        10.0
        /
        float(
            4
            *
            501
            *
            301
        )
    )

    require(
        abs(
            expected_fast
            -
            1.6578139402258606e-05
        )
        <=
        1.0e-20,
        (
            "Observed fast value is not explained "
            "by 10/full-production-grid."
        ),
    )

    print(
        "synthetic grid cells       = 24"
    )

    print(
        "production grid cells      =",
        4 * 501 * 301,
    )

    print(
        "authoritative synthetic    = 10 / 24"
    )

    print(
        "compressed observed        = "
        "10 / 603204 EXACT"
    )

    print(
        "production overmask kernel = CORRECT / UNCHANGED"
    )

    print(
        "failure source             = SYNTHETIC FIXTURE SHAPE"
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
            source_bytes,
            (
                "Existing backup differs from "
                "current pre-patch runner."
            ),
        )

    else:

        BACKUP.write_bytes(
            source_bytes
        )

    require(
        file_sha(
            BACKUP
        )
        ==
        pre_sha,
        "Backup SHA mismatch.",
    )

    print(
        "backup SHA256 =",
        file_sha(
            BACKUP
        ),
    )

    # --------------------------------------------------------
    # D. Replace only synthetic function
    # --------------------------------------------------------

    print()
    print("===== D. BUILD FIXTURE-ONLY PATCH =====")

    patched = (
        source[
            :start
        ]
        +
        NEW_FUNCTION
        +
        source[
            end:
        ]
    )

    require(
        "shape = (\n        2,\n        3,\n        4,\n    )"
        not in
        patched[
            start:
            start
            +
            len(
                NEW_FUNCTION
            )
            +
            200
        ],
        (
            "Old tiny synthetic fixture remains."
        ),
    )

    require(
        "shape = GRID_SHAPE"
        in
        patched,
        (
            "Production-grid synthetic fixture "
            "was not inserted."
        ),
    )

    # Production compressed kernel must still use its
    # original frozen denominator.
    require(
        '''overmask = (
        unnecessary.astype(
            np.float64
        )
        /
        float(
            GRID_CELL_COUNT
        )
    )'''
        in
        patched,
        (
            "Production overmask implementation "
            "was changed unexpectedly."
        ),
    )

    compile(
        patched,
        str(
            TARGET
        ),
        "exec",
    )

    print(
        "patch syntax                = PASS"
    )

    print(
        "synthetic fixture           = PRODUCTION GRID SHAPE"
    )

    print(
        "compressed production code = UNCHANGED"
    )

    print(
        "metric formulas             = UNCHANGED"
    )

    print(
        "candidate comparator        = UNCHANGED"
    )

    print(
        "Stage-1 gamma              = UNCHANGED"
    )

    print(
        "Stage-2 margins            = UNCHANGED"
    )

    print(
        "fixed caches               = UNCHANGED"
    )

    # --------------------------------------------------------
    # E. Transactional patch + regression
    # --------------------------------------------------------

    print()
    print("===== E. TRANSACTIONAL PATCH =====")

    applied = False

    try:

        atomic_write(
            TARGET,
            patched.encode(
                "utf-8"
            ),
        )

        applied = True

        rc, output = run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(
                    TARGET
                ),
            ]
        )

        require(
            rc == 0,
            (
                "Patched Part2 runner compile failed:\n"
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
                source_bytes,
            )

            require(
                TARGET.read_bytes()
                ==
                source_bytes,
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
        "runner post-patch SHA  =",
        post_sha,
    )

    # --------------------------------------------------------
    # F. Repair report
    # --------------------------------------------------------

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_synthetic_metric_fixture_repair",

        "status":
            "PASS_STAGE3_SYNTHETIC_METRIC_FIXTURE_REPAIRED",

        "diagnosis": {
            "failure_type":
                "synthetic_test_fixture_domain_mismatch",

            "production_scientific_failure":
                False,

            "authoritative_overmask_semantics":
                (
                    "|D\\O| / |full actuator grid|"
                ),

            "production_grid_cells":
                603204,

            "failed_synthetic_grid_cells":
                24,

            "observed_authority":
                0.4166666666666667,

            "observed_fast":
                1.6578139402258606e-05,

            "exact_fast_explanation":
                "10 / 603204",

            "repair":
                (
                    "run synthetic parity fixture on "
                    "the exact frozen production grid shape"
                ),
        },

        "runner": {
            "path":
                str(
                    TARGET
                ),

            "pre_patch_sha256":
                pre_sha,

            "post_patch_sha256":
                post_sha,

            "backup_path":
                str(
                    BACKUP
                ),

            "backup_sha256":
                file_sha(
                    BACKUP
                ),

            "patch_scope":
                "synthetic_metric_parity function only",
        },

        "production_semantics_preserved": {
            "compressed_metrics":
                True,

            "GRID_CELL_COUNT_denominator":
                True,

            "authoritative_metric_runtime":
                True,

            "Stage3_scoring_contract":
                True,

            "candidate_space":
                True,

            "lexicographic_comparator":
                True,
        },

        "scientific_boundary": {
            "Stage3_floor_outcomes_computed":
                False,

            "Stage3_candidate_sweep_started":
                False,

            "Stage3_floor_selected":
                False,

            "Stage4_temporal_rate_selected":
                False,

            "primary_acceptance_tested":
                False,

            "NI_bounds_modified":
                False,

            "formal_evaluation":
                False,
        },

        "regression":
            tests,

        "next":
            (
                "rerun identical Stage3 Part2/2; "
                "synthetic parity must pass before "
                "the 3270 development sweep starts"
            ),
    }

    atomic_write(
        REPAIR_REPORT,
        canonical_bytes(
            report
        ),
    )

    print()
    print("=" * 78)
    print("BLOCK 6.8 SYNTHETIC METRIC-FIXTURE REPAIR — FINAL")
    print("=" * 78)

    print(
        "production overmask denominator = UNCHANGED"
    )

    print(
        "production grid cells           = 603204"
    )

    print(
        "synthetic fixture               = NOW 4x501x301"
    )

    print(
        "compressed metric kernel        = UNCHANGED"
    )

    print(
        "authoritative metric runtime    = UNCHANGED"
    )

    print(
        "Stage-3 scoring comparator      = UNCHANGED"
    )

    print(
        "Stage-3 floor outcomes          = NOT COMPUTED"
    )

    print(
        "candidate sweep                 = NOT STARTED"
    )

    print(
        "Stage-4 temporal/rate           = NOT SELECTED"
    )

    print(
        "primary acceptance              = NOT TESTED"
    )

    print(
        "formal evaluation               = NO"
    )

    print(
        "Stage6 regression               =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_SYNTHETIC_METRIC_FIXTURE_REPAIRED"
    )

    print(
        "report =",
        REPAIR_REPORT,
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
        "BLOCK 6.8 SYNTHETIC METRIC-FIXTURE REPAIR = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
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
        "Stage-4 temporal/rate  = NOT SELECTED"
    )

    print(
        "formal evaluation      = NO"
    )

    print()
    print(
        "Do NOT change compressed_metrics(), "
        "GRID_CELL_COUNT or metric_semantics.py."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

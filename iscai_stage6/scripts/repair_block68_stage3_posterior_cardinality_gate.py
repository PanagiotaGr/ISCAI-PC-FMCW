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
    / "artifacts/block68/stage3_fixed_cache/cardinality_repair/"
      "run_block68_stage3_fixed_cache_parity.pre_posterior_cardinality_fix.py"
)

REPORT = (
    S6
    / "reports/block68_stage3_posterior_cardinality_repair.json"
)

FINAL_CACHE_REPORT = (
    S6
    / "reports/block68_stage3_fixed_cache_parity.json"
)

FINAL_CACHE_FREEZE = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "stage3_fixed_cache_freeze.json"
)

EXPECTED_TESTS = 224


OLD_CARDINALITY = '''    require(
        len(
            posterior_rows
        )
        ==
        877,
        "Posterior rows != 877.",
    )
'''


NEW_CARDINALITY = '''    # The identity-safe posterior is an upstream
    # development posterior population and may be a strict
    # superset of the 877 Stage-3 selected predictive actors.
    #
    # Scientific invariant:
    # every selected predictive actor must resolve to an
    # existing unique posterior row.  Total posterior-file
    # cardinality is NOT the Stage-3 actor-selection cardinality.

    posterior_total_rows = len(
        posterior_rows
    )

    require(
        posterior_total_rows
        >
        0,
        "Posterior file is empty.",
    )
'''


OLD_AFTER_LOOKUP = '''    by_scenario = defaultdict(
        list
    )

    class_counts = defaultdict(
        int
    )
'''


NEW_AFTER_LOOKUP = '''    # Exact selected-subset posterior identity gate.
    #
    # Do not require equality between total posterior rows
    # and selected predictive actors.  Require exact coverage
    # of the frozen 877-actor selection instead.

    selected_posterior_keys = {
        (
            str(
                row[
                    "scenario_id"
                ]
            ),
            str(
                row[
                    "prediction_id"
                ]
            ),
            int(
                row[
                    "cohort_index"
                ]
            ),
        )
        for row in matches
    }

    require(
        len(
            selected_posterior_keys
        )
        ==
        877,
        (
            "Selected predictive posterior identity "
            "keys are not 877 unique keys: "
            f"{len(selected_posterior_keys)}"
        ),
    )

    missing_selected_posteriors = sorted(
        selected_posterior_keys
        -
        set(
            posterior_lookup
        )
    )

    require(
        not missing_selected_posteriors,
        (
            "Frozen selected predictive actors are "
            "missing posterior rows: "
            f"{missing_selected_posteriors[:10]}"
        ),
    )

    print(
        "posterior total rows     =",
        posterior_total_rows,
    )

    print(
        "selected posterior keys =",
        len(
            selected_posterior_keys
        ),
        "/ 877 EXACT PASS",
    )

    print(
        "posterior superset rows  =",
        posterior_total_rows
        -
        len(
            selected_posterior_keys
        ),
    )

    print(
        "posterior total == 877   = NOT REQUIRED"
    )

    by_scenario = defaultdict(
        list
    )

    class_counts = defaultdict(
        int
    )
'''


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


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
    print("BLOCK 6.8 STAGE-3 POSTERIOR CARDINALITY REPAIR")
    print("SELECTED-SUBSET IDENTITY GATE ONLY")
    print("=" * 76)

    # --------------------------------------------------------
    # A. Failure-state boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT FAILURE BOUNDARY =====")

    require(
        TARGET.is_file(),
        f"Missing target runner: {TARGET}",
    )

    require(
        not FINAL_CACHE_REPORT.exists(),
        (
            "Final Stage-3 cache PASS report already exists; "
            "do not patch a completed freeze."
        ),
    )

    require(
        not FINAL_CACHE_FREEZE.exists(),
        (
            "Final Stage-3 cache freeze already exists; "
            "do not patch a completed freeze."
        ),
    )

    original_bytes = TARGET.read_bytes()
    original_source = original_bytes.decode("utf-8")
    original_sha = sha256(original_bytes).hexdigest()

    print(
        "target pre-patch SHA256 =",
        original_sha,
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
        "P_occ payload arrays    = NOT OPENED BY FAILED RUN"
    )

    print(
        "formal evaluation       = NO"
    )

    # --------------------------------------------------------
    # B. Exact obsolete assumption
    # --------------------------------------------------------

    print()
    print("===== B. OBSOLETE CARDINALITY ASSUMPTION =====")

    cardinality_count = original_source.count(
        OLD_CARDINALITY
    )

    require(
        cardinality_count == 1,
        (
            "Expected exactly one obsolete "
            "posterior len==877 assertion; "
            f"found {cardinality_count}."
        ),
    )

    insertion_count = original_source.count(
        OLD_AFTER_LOOKUP
    )

    require(
        insertion_count == 1,
        (
            "Expected exactly one posterior-lookup "
            f"insertion point; found {insertion_count}."
        ),
    )

    print(
        "obsolete `len(posterior_rows)==877` gate = FOUND EXACTLY ONCE"
    )

    print(
        "replacement semantic = "
        "877 SELECTED ACTORS MUST BE COVERED"
    )

    print(
        "extra upstream posterior rows = PERMITTED"
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
        original_sha,
        "Backup SHA mismatch.",
    )

    print(
        "backup SHA256 =",
        file_sha(BACKUP),
    )

    # --------------------------------------------------------
    # D. Minimal patch
    # --------------------------------------------------------

    print()
    print("===== D. BUILD MINIMAL PATCH =====")

    patched = original_source.replace(
        OLD_CARDINALITY,
        NEW_CARDINALITY,
        1,
    )

    patched = patched.replace(
        OLD_AFTER_LOOKUP,
        NEW_AFTER_LOOKUP,
        1,
    )

    require(
        "Posterior rows != 877."
        not in
        patched,
        (
            "Obsolete total-posterior cardinality "
            "gate remains."
        ),
    )

    require(
        "selected_posterior_keys"
        in
        patched,
        (
            "Selected-subset posterior parity "
            "gate not inserted."
        ),
    )

    compile(
        patched,
        str(TARGET),
        "exec",
    )

    print(
        "patch syntax               = PASS"
    )

    print(
        "scientific actor population = UNCHANGED"
    )

    print(
        "posterior file             = UNCHANGED"
    )

    print(
        "Stage1/2 numerics          = UNCHANGED"
    )

    print(
        "evaluator semantics        = UNCHANGED"
    )

    print(
        "floor scoring semantics    = UNCHANGED"
    )

    # --------------------------------------------------------
    # E. Transactional write + regression
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
                "Rollback failed.",
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
        "target post-patch SHA256 =",
        post_sha,
    )

    # --------------------------------------------------------
    # F. Repair report
    # --------------------------------------------------------

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_posterior_cardinality_repair",

        "status":
            "PASS_POSTERIOR_SELECTED_SUBSET_CARDINALITY_REPAIRED",

        "diagnosis": {
            "failure_type":
                "non_scientific_cardinality_assumption",

            "obsolete_assumption":
                (
                    "total identity-safe development "
                    "posterior rows must equal the "
                    "877 selected Stage-3 predictive actors"
                ),

            "authoritative_invariant":
                (
                    "all 877 unique selected predictive "
                    "actor posterior keys must exist in "
                    "the unique upstream posterior lookup"
                ),

            "extra_posterior_rows_permitted":
                True,
        },

        "runner": {
            "path":
                str(TARGET),

            "pre_patch_sha256":
                original_sha,

            "post_patch_sha256":
                post_sha,

            "backup_path":
                str(BACKUP),

            "backup_sha256":
                file_sha(BACKUP),

            "patch_scope":
                (
                    "posterior total-row cardinality "
                    "gate only + exact selected-subset "
                    "coverage evidence"
                ),
        },

        "scientific_boundary": {
            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage3_scoring_modified":
                False,

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
                "rerun the identical Stage-3 execution "
                "Part 1/2 cache/parity runner"
            ),
    }

    atomic_write(
        REPORT,
        canonical_bytes(report),
    )

    print()
    print("=" * 76)
    print("BLOCK 6.8 POSTERIOR CARDINALITY REPAIR — FINAL")
    print("=" * 76)

    print(
        "total posterior == 877   = NOT REQUIRED"
    )

    print(
        "selected actor count     = STILL 877"
    )

    print(
        "selected posterior join  = REQUIRED EXACT"
    )

    print(
        "extra posterior rows     = ALLOWED"
    )

    print(
        "Stage-1 gamma            = UNCHANGED"
    )

    print(
        "Stage-2 margins          = UNCHANGED"
    )

    print(
        "Stage-3 scoring          = UNCHANGED"
    )

    print(
        "Stage-3 floor outcomes   = NOT COMPUTED"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "Stage6 regression        =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_POSTERIOR_SELECTED_SUBSET_CARDINALITY_REPAIRED"
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
        "BLOCK 6.8 POSTERIOR CARDINALITY REPAIR = BLOCKED"
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

    print(
        "Do not modify the posterior file "
        "or filter it to 877 rows."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

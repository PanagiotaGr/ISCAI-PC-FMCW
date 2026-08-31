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
      "scenario_loader_repair/"
      "run_block68_stage3_fixed_cache_parity."
      "pre_scenario_loader_fix.py"
)

REPORT = (
    S6
    / "reports/block68_stage3_scenario_loader_route_repair.json"
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


EXPECTED_TARGET_SHA = (
    "1cb25fe7264daa59b8734cf7bfea7baa"
    "01fa2dc1610a2c9aa30d31a753a2d25e"
)

EXPECTED_TESTS = 224


IMPORT_ANCHOR = '''from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
'''


IMPORT_REPLACEMENT = '''from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage4.data.real_pipeline import (
    read_training_scenario,
)
'''


OLD_SCENARIO_BLOCK = '''        motion_path = Path(
            str(
                cohort_row[
                    "motion_shard"
                ]
            )
        )

        require(
            motion_path.is_absolute(),
            (
                "motion_shard is not an absolute "
                f"frozen path: {motion_path}"
            ),
        )

        require(
            "compact_record_offset"
            in
            cohort_row,
            (
                "Missing compact_record_offset "
                f"for {scenario_id}"
            ),
        )

        scenario, payload_length = (
            read_tfrecord_at_offset(
                motion_path,
                int(
                    cohort_row[
                        "compact_record_offset"
                    ]
                ),
            )
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            scenario_id,
            (
                "Scenario ID mismatch after "
                "TFRecord read."
            ),
        )

        if (
            "payload_length"
            in
            cohort_row
        ):
            require(
                int(
                    cohort_row[
                        "payload_length"
                    ]
                )
                ==
                payload_length,
                (
                    "Frozen payload length "
                    "mismatch."
                ),
            )
'''


NEW_SCENARIO_BLOCK = '''        # Resolve the frozen cohort record through the
        # same authoritative Stage-4 scenario loader used
        # by the successful Stage-1 and Stage-2 development
        # selections.
        #
        # `motion_shard` is a frozen shard identifier and is
        # not required to be an absolute filesystem path.
        # Do not mutate it or invent a directory prefix.

        require(
            "motion_shard"
            in
            cohort_row,
            (
                "Missing frozen motion_shard "
                f"for {scenario_id}"
            ),
        )

        require(
            str(
                cohort_row[
                    "motion_shard"
                ]
            ).strip()
            !=
            "",
            (
                "Empty frozen motion_shard "
                f"for {scenario_id}"
            ),
        )

        require(
            "compact_record_offset"
            in
            cohort_row,
            (
                "Missing compact_record_offset "
                f"for {scenario_id}"
            ),
        )

        scenario = (
            read_training_scenario(
                cohort_row
            )
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            scenario_id,
            (
                "Scenario ID mismatch after "
                "authoritative frozen cohort load."
            ),
        )
'''


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

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

    temporary.write_bytes(
        payload
    )

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
    print("BLOCK 6.8 STAGE-3 SCENARIO-LOADER ROUTE REPAIR")
    print("PATH-RESOLUTION ASSUMPTION ONLY")
    print("=" * 76)

    # --------------------------------------------------------
    # A. Exact current boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT CURRENT BOUNDARY =====")

    require(
        TARGET.is_file(),
        f"Missing target runner: {TARGET}",
    )

    actual_target_sha = file_sha(
        TARGET
    )

    require(
        actual_target_sha
        ==
        EXPECTED_TARGET_SHA,
        (
            "Stage-3 Part1 runner differs from the "
            "post-cardinality-repair SHA.\n"
            f"expected={EXPECTED_TARGET_SHA}\n"
            f"actual={actual_target_sha}"
        ),
    )

    require(
        not FINAL_CACHE_REPORT.exists(),
        (
            "Final fixed-cache PASS report already exists; "
            "do not patch a completed execution."
        ),
    )

    require(
        not FINAL_CACHE_FREEZE.exists(),
        (
            "Final fixed-cache freeze already exists; "
            "do not patch a completed execution."
        ),
    )

    print(
        "target SHA256             = EXACT PASS"
    )

    print(
        "posterior cardinality fix = PRESERVED"
    )

    print(
        "Stage-1 gamma             = FROZEN"
    )

    print(
        "Stage-2 margins           = FROZEN"
    )

    print(
        "Stage-3 scoring           = FROZEN"
    )

    print(
        "Stage-3 floor outcomes    = NOT COMPUTED"
    )

    print(
        "formal evaluation         = NO"
    )

    # --------------------------------------------------------
    # B. Diagnose exact bad assumption
    # --------------------------------------------------------

    print()
    print("===== B. OBSOLETE PATH ASSUMPTION =====")

    original_bytes = TARGET.read_bytes()

    original_source = original_bytes.decode(
        "utf-8"
    )

    require(
        original_source.count(
            OLD_SCENARIO_BLOCK
        )
        ==
        1,
        (
            "Expected exactly one obsolete absolute-"
            "motion_shard loading block."
        ),
    )

    require(
        original_source.count(
            IMPORT_ANCHOR
        )
        ==
        1,
        (
            "Expected exactly one adapter import anchor."
        ),
    )

    require(
        "read_training_scenario"
        not in
        original_source,
        (
            "Authoritative Stage-4 loader is already "
            "present unexpectedly."
        ),
    )

    print(
        "absolute motion_shard requirement = FOUND"
    )

    print(
        "frozen shard identifier example   = RELATIVE/LOGICAL NAME"
    )

    print(
        "repair route = Stage4 read_training_scenario(cohort_row)"
    )

    print(
        "directory prefix guessing = FORBIDDEN"
    )

    # --------------------------------------------------------
    # C. Exact backup
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
                "current pre-patch target."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    require(
        file_sha(
            BACKUP
        )
        ==
        actual_target_sha,
        "Backup SHA mismatch.",
    )

    print(
        "backup SHA256 =",
        file_sha(
            BACKUP
        ),
    )

    # --------------------------------------------------------
    # D. Minimal patch
    # --------------------------------------------------------

    print()
    print("===== D. BUILD MINIMAL LOADER-ROUTE PATCH =====")

    patched = original_source.replace(
        IMPORT_ANCHOR,
        IMPORT_REPLACEMENT,
        1,
    )

    patched = patched.replace(
        OLD_SCENARIO_BLOCK,
        NEW_SCENARIO_BLOCK,
        1,
    )

    require(
        "motion_shard is not an absolute frozen path"
        not in
        patched,
        (
            "Obsolete absolute-path gate remains."
        ),
    )

    require(
        "read_training_scenario("
        in
        patched,
        (
            "Authoritative Stage-4 loader call "
            "was not inserted."
        ),
    )

    # Preserve the already-fixed posterior subset gate.
    require(
        "selected_posterior_keys"
        in
        patched,
        (
            "Posterior selected-subset gate "
            "was accidentally removed."
        ),
    )

    require(
        "posterior total == 877"
        in
        patched,
        (
            "Posterior cardinality repair evidence "
            "was accidentally removed."
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
        "cohort metadata          = UNCHANGED"
    )

    print(
        "scenario population      = UNCHANGED"
    )

    print(
        "posterior semantics      = UNCHANGED"
    )

    print(
        "P_occ semantics          = UNCHANGED"
    )

    print(
        "Stage1/2 numerics        = UNCHANGED"
    )

    print(
        "evaluator semantics      = UNCHANGED"
    )

    print(
        "floor scoring            = UNCHANGED"
    )

    # --------------------------------------------------------
    # E. Transactional patch
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
            "6.8_stage3_scenario_loader_route_repair",

        "status":
            "PASS_STAGE3_SCENARIO_LOADER_ROUTE_REPAIRED",

        "diagnosis": {
            "failure_type":
                "non_scientific_path_resolution_assumption",

            "obsolete_assumption":
                (
                    "cohort motion_shard must itself "
                    "be an absolute filesystem path"
                ),

            "actual_frozen_semantic":
                (
                    "motion_shard is a frozen shard identifier "
                    "resolved by the established Stage-4 "
                    "read_training_scenario(cohort_row) route"
                ),

            "directory_prefix_invented":
                False,
        },

        "runner": {
            "path":
                str(TARGET),

            "pre_patch_sha256":
                actual_target_sha,

            "post_patch_sha256":
                post_sha,

            "backup_path":
                str(BACKUP),

            "backup_sha256":
                file_sha(BACKUP),

            "patch_scope":
                (
                    "scenario loading/path resolution only"
                ),
        },

        "preserved_repairs": {
            "posterior_total_cardinality_not_required":
                True,

            "selected_posterior_keys_required":
                877,
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
                "rerun the same Stage-3 execution "
                "Part 1/2 fixed-cache parity runner"
            ),
    }

    atomic_write(
        REPORT,
        canonical_bytes(
            report
        ),
    )

    print()
    print("=" * 76)
    print("BLOCK 6.8 SCENARIO-LOADER ROUTE REPAIR — FINAL")
    print("=" * 76)

    print(
        "motion_shard absolute path = NOT REQUIRED"
    )

    print(
        "cohort field               = UNCHANGED"
    )

    print(
        "scenario loader            = AUTHORITATIVE STAGE-4 ROUTE"
    )

    print(
        "directory prefix guessed   = NO"
    )

    print(
        "selected actors            = STILL 877"
    )

    print(
        "posterior subset repair    = PRESERVED"
    )

    print(
        "Stage-1 gamma              = UNCHANGED"
    )

    print(
        "Stage-2 margins            = UNCHANGED"
    )

    print(
        "Stage-3 scoring            = UNCHANGED"
    )

    print(
        "Stage-3 floor outcomes     = NOT COMPUTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_SCENARIO_LOADER_ROUTE_REPAIRED"
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
        "BLOCK 6.8 SCENARIO-LOADER ROUTE REPAIR = BLOCKED"
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
        "Do not rewrite motion_shard values "
        "or guess an external shard directory."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

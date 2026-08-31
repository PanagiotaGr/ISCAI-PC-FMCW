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
      "run_block68_primary_development_acceptance_gate.py"
)

MATERIALIZATION = (
    S6
    / "reports/"
      "block68_reactive_primary_comparator_materialization.json"
)

AUGMENTED = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_reactive_primary_comparator_augmented.jsonl"
)

BACKUP = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "schema_repair/"
      "run_block68_primary_development_acceptance_gate."
      "pre_reactive_schema_fix.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_primary_gate_reactive_schema_repair.json"
)

FINAL_GATE_REPORT = (
    S6
    / "reports/"
      "block68_primary_development_acceptance_gate.json"
)

FINAL_GATE_DETAIL = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_primary_development_acceptance_detail.json"
)

EXPECTED_TESTS = 224


OLD_SECTION_C = '''    print()
    print("===== C. FROZEN REACTIVE COMPARATOR =====")

    reactive_path = locate_reactive_raw()

    require(
        file_sha(reactive_path)
        ==
        REACTIVE_RAW_SHA,
        "Reactive raw SHA changed.",
    )

    reactive_rows = read_jsonl(
        reactive_path
    )

    require(
        len(reactive_rows) == 120,
        (
            "Expected 120 reactive development rows; "
            f"observed {len(reactive_rows)}."
        ),
    )

    print(
        "reactive raw artifact =",
        reactive_path,
    )

    print(
        "reactive raw SHA      =",
        REACTIVE_RAW_SHA,
    )

    print(
        "reactive scenarios    = 120 / 120"
    )
'''


NEW_SECTION_C = '''    print()
    print("===== C. FROZEN REACTIVE COMPARATOR =====")

    # The original reactive-only NI artifact intentionally
    # materialized overmask/pedestrian/cyclist NI quantities
    # but did not contain vehicle_shadow_zone_violation.
    #
    # The missing vehicle comparator has now been materialized
    # from the already-frozen original_reactive_ADB t0 action
    # held across the four evaluator horizons and the already-
    # frozen oracle_vehicle supports.  No policy or gate rule
    # was changed.

    require(
        MATERIALIZATION.is_file(),
        (
            "Missing reactive primary-comparator "
            f"materialization report: {MATERIALIZATION}"
        ),
    )

    materialization = read_json(
        MATERIALIZATION
    )

    require(
        materialization.get(
            "status"
        )
        ==
        "PASS_REACTIVE_PRIMARY_COMPARATOR_MATERIALIZED_FROZEN",
        (
            "Reactive primary-comparator "
            "materialization is not PASS."
        ),
    )

    require(
        materialization[
            "scientific_boundary"
        ][
            "primary_acceptance_tested"
        ]
        is False,
        (
            "Comparator materialization claims "
            "the primary gate was already tested."
        ),
    )

    require(
        materialization[
            "scientific_boundary"
        ][
            "acceptance_rule_modified"
        ]
        is False,
        (
            "Comparator materialization changed "
            "the acceptance rule."
        ),
    )

    reactive_path = AUGMENTED

    require(
        reactive_path.is_file(),
        (
            "Missing augmented frozen reactive "
            f"comparator: {reactive_path}"
        ),
    )

    reactive_source_sha = file_sha(
        reactive_path
    )

    require(
        reactive_source_sha
        ==
        materialization[
            "outputs"
        ][
            "augmented_primary_comparator"
        ][
            "sha256"
        ],
        (
            "Augmented reactive comparator SHA "
            "differs from its write-once freeze."
        ),
    )

    require(
        materialization[
            "schema_gap"
        ][
            "original_reactive_NI_sha256"
        ]
        ==
        REACTIVE_RAW_SHA,
        (
            "Materialization is not based on the "
            "exact original frozen NI comparator."
        ),
    )

    reactive_rows = read_jsonl(
        reactive_path
    )

    require(
        len(reactive_rows) == 120,
        (
            "Expected 120 augmented reactive rows; "
            f"observed {len(reactive_rows)}."
        ),
    )

    print(
        "reactive comparator artifact =",
        reactive_path,
    )

    print(
        "reactive comparator SHA      =",
        reactive_source_sha,
    )

    print(
        "original reactive NI SHA     =",
        REACTIVE_RAW_SHA,
    )

    print(
        "vehicle metric source        = "
        "FROZEN t0 HOLD + FROZEN oracle_vehicle"
    )

    print(
        "reactive scenarios           = 120 / 120"
    )
'''


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

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
        + "\n"
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

    print("=" * 78)
    print("BLOCK 6.8 PRIMARY-GATE REACTIVE-SCHEMA REPAIR")
    print("COMPARATOR SOURCE ONLY — POLICY / RULE UNCHANGED")
    print("=" * 78)

    # ========================================================
    # A. Exact current boundary
    # ========================================================

    print()
    print("===== A. EXACT CURRENT BOUNDARY =====")

    require(
        TARGET.is_file(),
        f"Missing primary-gate runner: {TARGET}",
    )

    require(
        MATERIALIZATION.is_file(),
        (
            "Missing successful comparator "
            f"materialization: {MATERIALIZATION}"
        ),
    )

    require(
        AUGMENTED.is_file(),
        (
            "Missing augmented comparator: "
            f"{AUGMENTED}"
        ),
    )

    materialization = json.loads(
        MATERIALIZATION.read_text(
            encoding="utf-8"
        )
    )

    require(
        materialization.get("status")
        ==
        "PASS_REACTIVE_PRIMARY_COMPARATOR_MATERIALIZED_FROZEN",
        "Comparator materialization status is not PASS.",
    )

    require(
        not FINAL_GATE_REPORT.exists(),
        (
            "A final primary-gate report already exists; "
            "do not patch after gate completion."
        ),
    )

    require(
        not FINAL_GATE_DETAIL.exists(),
        (
            "A final primary-gate detail already exists; "
            "do not patch after gate completion."
        ),
    )

    original_bytes = TARGET.read_bytes()
    source = original_bytes.decode(
        "utf-8"
    )

    pre_sha = sha256(
        original_bytes
    ).hexdigest()

    print(
        "runner pre-patch SHA256 =",
        pre_sha,
    )

    print(
        "Stage1/2/3/4 policy = IMMUTABLE"
    )

    print(
        "acceptance rule      = IMMUTABLE"
    )

    print(
        "primary gate         = NOT COMPLETED"
    )

    print(
        "formal evaluation    = NO"
    )

    # ========================================================
    # B. Verify exact failed-source block
    # ========================================================

    print()
    print("===== B. EXACT SCHEMA-REPAIR TARGET =====")

    require(
        source.count(
            OLD_SECTION_C
        )
        ==
        1,
        (
            "Expected exactly one old reactive "
            "comparator Section C."
        ),
    )

    require(
        "No finite values found for {metric}."
        in
        source,
        (
            "Primary-gate macro gate structure "
            "changed unexpectedly."
        ),
    )

    require(
        "OVERMASK_DELTA_MAX = 0.02"
        in
        source,
        (
            "Frozen overmask gate numeric changed."
        ),
    )

    require(
        "PEDESTRIAN_VISIBILITY_DELTA_MIN = -0.05"
        in
        source,
        (
            "Frozen pedestrian NI numeric changed."
        ),
    )

    require(
        "CYCLIST_VISIBILITY_DELTA_MIN = -0.05"
        in
        source,
        (
            "Frozen cyclist NI numeric changed."
        ),
    )

    print(
        "failed source = NI-only reactive JSONL"
    )

    print(
        "repair source = augmented immutable comparator"
    )

    print(
        "gate equations changed = NO"
    )

    print(
        "gate numerics changed  = NO"
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

    # ========================================================
    # D. Patch constants + Section C + provenance output
    # ========================================================

    print()
    print("===== D. BUILD MINIMAL SOURCE-ROUTE PATCH =====")

    insertion_anchor = '''OUTPUT = (
    S6
    / "reports/"
      "block68_primary_development_acceptance_gate.json"
)
'''

    require(
        source.count(
            insertion_anchor
        )
        ==
        1,
        "Could not locate output-constant anchor.",
    )

    constant_insert = insertion_anchor + '''

# Frozen comparator extension produced solely to add the
# previously absent reactive vehicle_shadow_zone_violation.
MATERIALIZATION = (
    S6
    / "reports/"
      "block68_reactive_primary_comparator_materialization.json"
)

AUGMENTED = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_reactive_primary_comparator_augmented.jsonl"
)
'''

    patched = source.replace(
        insertion_anchor,
        constant_insert,
        1,
    )

    patched = patched.replace(
        OLD_SECTION_C,
        NEW_SECTION_C,
        1,
    )

    # In the frozen result provenance, use the actual new
    # comparator SHA rather than pretending it is the old
    # NI-only SHA.
    old_provenance = '''            "sha256":
                REACTIVE_RAW_SHA,

            "scenario_rows":
                120,
'''

    new_provenance = '''            "sha256":
                reactive_source_sha,

            "original_NI_sha256":
                REACTIVE_RAW_SHA,

            "augmentation_materialization":
                str(
                    MATERIALIZATION
                ),

            "augmentation_materialization_sha256":
                file_sha(
                    MATERIALIZATION
                ),

            "scenario_rows":
                120,
'''

    require(
        patched.count(
            old_provenance
        )
        ==
        1,
        (
            "Expected exactly one result-provenance "
            "reactive SHA block."
        ),
    )

    patched = patched.replace(
        old_provenance,
        new_provenance,
        1,
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
        "macro_metric()            = UNCHANGED"
    )

    print(
        "vehicle strict rule       = UNCHANGED"
    )

    print(
        "overmask +0.02 rule       = UNCHANGED"
    )

    print(
        "ped/cyclist -0.05 rules   = UNCHANGED"
    )

    print(
        "predictive winner source  = UNCHANGED"
    )

    print(
        "formal boundary           = UNCHANGED"
    )

    # ========================================================
    # E. Transactional write + regression
    # ========================================================

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
                "Patched primary-gate runner "
                "compile failed:\n"
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
            "6.8_primary_gate_reactive_schema_repair",

        "status":
            "PASS_PRIMARY_GATE_REACTIVE_SCHEMA_SOURCE_REPAIRED",

        "diagnosis": {
            "failure_type":
                "reactive_comparator_schema_gap",

            "scientific_gate_failure":
                False,

            "failed_artifact_role":
                (
                    "reactive-only NI delta materialization; "
                    "did not include vehicle shadow-zone metric"
                ),

            "repair":
                (
                    "use write-once augmented comparator whose "
                    "vehicle metric is derived from the already-"
                    "frozen causal t0 action and already-frozen "
                    "oracle_vehicle evaluator support"
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
        },

        "comparator_materialization": {
            "path":
                str(MATERIALIZATION),

            "sha256":
                file_sha(MATERIALIZATION),

            "augmented_comparator":
                str(AUGMENTED),

            "augmented_comparator_sha256":
                file_sha(AUGMENTED),
        },

        "immutability": {
            "Stage1":
                True,

            "Stage2":
                True,

            "Stage3":
                True,

            "Stage4":
                True,

            "acceptance_rule":
                True,

            "NI_bounds":
                True,

            "predictive_results":
                True,

            "formal_evaluation":
                False,
        },

        "primary_acceptance_tested":
            False,

        "regression":
            tests,

        "next":
            "RERUN_FROZEN_PRIMARY_DEVELOPMENT_ACCEPTANCE_GATE",
    }

    REPORT.write_bytes(
        canonical_bytes(
            report
        )
    )

    print()
    print("=" * 78)
    print("PRIMARY-GATE REACTIVE-SCHEMA REPAIR — FINAL")
    print("=" * 78)

    print(
        "reactive NI source       = PRESERVED"
    )

    print(
        "missing vehicle metric   = BOUND FROM FROZEN SOURCES"
    )

    print(
        "primary gate equations   = UNCHANGED"
    )

    print(
        "primary gate bounds      = UNCHANGED"
    )

    print(
        "Stage1/2/3/4 policy      = IMMUTABLE"
    )

    print(
        "primary acceptance       = NOT YET TESTED"
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
        "PASS_PRIMARY_GATE_REACTIVE_SCHEMA_SOURCE_REPAIRED"
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
        "PRIMARY-GATE REACTIVE-SCHEMA REPAIR = BLOCKED"
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
        "Stage1/2/3/4 policy = STILL IMMUTABLE"
    )

    print(
        "primary gate         = NOT COMPLETED"
    )

    print(
        "formal evaluation    = NO"
    )

    print(
        "Do not change gate bounds or predictive policy."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

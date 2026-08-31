from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys
import traceback

from iscai_stage3.validation import (
    read_motion_scenario,
    read_validation_manifest,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

RUNNER = (
    STAGE4
    / "scripts/"
      "run_block48_formal_evaluation.py"
)

SAFE_CONTROLLER = (
    STAGE4
    / "scripts/"
      "run_block48_part2_safe.py"
)

FORMAL_RUNTIME = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "formal_runtime.py"
)

FORMAL_TEST = (
    STAGE4
    / "tests/"
      "test_block48_formal_runtime.py"
)

FORMAL_MANIFEST = (
    ROOT
    / "iscai_stage3/artifacts/block38e/"
      "formal_validation_120.jsonl"
)

CANONICAL_VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

PAIRED_ROOT = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0"
)

BACKUP = (
    STAGE4
    / "artifacts/block48/"
      "run_block48_formal_evaluation."
      "pre_canonical_validation_repair.py"
)

REPORT = (
    STAGE4
    / "reports/"
      "block48_validation_resolver_repair.json"
)

RESUME_LOG = (
    STAGE4
    / "artifacts/block48/"
      "part2_after_validation_resolver_repair.log"
)

EXPECTED_VALIDATION_MANIFEST_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_VALIDATION_ROWS = 44097

EXPECTED_FORMAL_ROWS = 120

REPAIR_MARKER = (
    "# BLOCK48_CANONICAL_VALIDATION_MANIFEST_REPAIR"
)


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_json(
    payload,
):
    REPORT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


def blocked(
    phase,
    reason,
    recovery,
):
    write_json({
        "stage":
            4,

        "block":
            "4.8_part_2_validation_resolver_repair",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "formal_neural_inference_started":
            False,

        "training":
            False,

        "recalibration":
            False,

        "classical_rerun":
            False,

        "upstream_modified":
            False,
    })

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 VALIDATION RESOLVER REPAIR = BLOCKED"
    )
    print(
        "============================================================"
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
        "formal neural inference = NOT STARTED"
    )

    print(
        "upstream modified       = NO"
    )

    print(
        "terminal remains open   = YES"
    )


def load_formal_rows():
    rows = tuple(
        json.loads(
            line
        )
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    if (
        len(
            rows
        )
        !=
        EXPECTED_FORMAL_ROWS
    ):
        raise RuntimeError(
            "Frozen formal manifest "
            f"has {len(rows)} rows, "
            "expected 120."
        )

    return rows


def audit_canonical_validation_route():
    print()
    print(
        "============================================================"
    )
    print(
        "A. CANONICAL VALIDATION MANIFEST AUDIT"
    )
    print(
        "============================================================"
    )

    if not CANONICAL_VALIDATION_MANIFEST.is_file():
        raise RuntimeError(
            "Canonical selected_validation.jsonl "
            "is missing."
        )

    actual_sha = file_sha256(
        CANONICAL_VALIDATION_MANIFEST
    )

    print(
        "manifest path =",
        CANONICAL_VALIDATION_MANIFEST,
    )

    print(
        "manifest SHA  =",
        actual_sha,
    )

    if (
        actual_sha
        !=
        EXPECTED_VALIDATION_MANIFEST_SHA
    ):
        raise RuntimeError(
            "Canonical validation-manifest "
            "SHA mismatch."
        )

    rows = read_validation_manifest(
        CANONICAL_VALIDATION_MANIFEST
    )

    print(
        "validation rows =",
        len(
            rows
        ),
    )

    if (
        len(
            rows
        )
        !=
        EXPECTED_VALIDATION_ROWS
    ):
        raise RuntimeError(
            "Canonical validation manifest "
            f"has {len(rows)} rows; "
            "expected 44097."
        )

    by_id = {}

    duplicates = []

    for row in rows:
        scenario_id = str(
            row.scenario_id
        )

        if scenario_id in by_id:
            duplicates.append(
                scenario_id
            )

        else:
            by_id[
                scenario_id
            ] = row

    if duplicates:
        raise RuntimeError(
            "Canonical validation manifest "
            "contains duplicate scenario IDs."
        )

    formal_rows = (
        load_formal_rows()
    )

    missing = []

    source_shard_mismatch = []

    selection_hash_mismatch = []

    split_mismatch = []

    for formal in formal_rows:
        scenario_id = str(
            formal[
                "scenario_id"
            ]
        )

        row = by_id.get(
            scenario_id
        )

        if row is None:
            missing.append(
                scenario_id
            )

            continue

        if (
            str(
                row.source_shard
            )
            !=
            str(
                formal[
                    "source_shard"
                ]
            )
        ):
            source_shard_mismatch.append({
                "scenario_id":
                    scenario_id,

                "formal":
                    str(
                        formal[
                            "source_shard"
                        ]
                    ),

                "canonical":
                    str(
                        row.source_shard
                    ),
            })

        if (
            str(
                row.selection_hash
            )
            !=
            str(
                formal[
                    "selection_hash"
                ]
            )
        ):
            selection_hash_mismatch.append({
                "scenario_id":
                    scenario_id,

                "formal":
                    str(
                        formal[
                            "selection_hash"
                        ]
                    ),

                "canonical":
                    str(
                        row.selection_hash
                    ),
            })

        if (
            str(
                row.split
            ).lower()
            !=
            "validation"
        ):
            split_mismatch.append({
                "scenario_id":
                    scenario_id,

                "split":
                    str(
                        row.split
                    ),
            })

    print()
    print(
        "formal IDs resolved      =",
        (
            EXPECTED_FORMAL_ROWS
            -
            len(
                missing
            )
        ),
        "/",
        EXPECTED_FORMAL_ROWS,
    )

    print(
        "source-shard matches     =",
        (
            EXPECTED_FORMAL_ROWS
            -
            len(
                source_shard_mismatch
            )
        ),
        "/",
        EXPECTED_FORMAL_ROWS,
    )

    print(
        "selection-hash matches   =",
        (
            EXPECTED_FORMAL_ROWS
            -
            len(
                selection_hash_mismatch
            )
        ),
        "/",
        EXPECTED_FORMAL_ROWS,
    )

    print(
        "validation split matches =",
        (
            EXPECTED_FORMAL_ROWS
            -
            len(
                split_mismatch
            )
        ),
        "/",
        EXPECTED_FORMAL_ROWS,
    )

    if missing:
        raise RuntimeError(
            "Canonical selected_validation "
            f"manifest is missing {len(missing)} "
            "formal scenario IDs."
        )

    if source_shard_mismatch:
        raise RuntimeError(
            "Formal source_shard differs from "
            "canonical validation manifest for "
            f"{len(source_shard_mismatch)} scenarios."
        )

    if selection_hash_mismatch:
        raise RuntimeError(
            "Formal selection_hash differs from "
            "canonical validation manifest for "
            f"{len(selection_hash_mismatch)} scenarios."
        )

    if split_mismatch:
        raise RuntimeError(
            "One or more formal scenarios are "
            "not validation rows."
        )

    print()
    print(
        "canonical validation route = PASS"
    )

    return (
        formal_rows,
        rows,
        by_id,
        actual_sha,
    )


def one_scenario_reader_smoke(
    formal_rows,
    by_id,
):
    print()
    print(
        "============================================================"
    )
    print(
        "B. ONE-SCENARIO READER SMOKE"
    )
    print(
        "NO NEURAL INFERENCE"
    )
    print(
        "============================================================"
    )

    formal = formal_rows[
        0
    ]

    scenario_id = str(
        formal[
            "scenario_id"
        ]
    )

    row = by_id[
        scenario_id
    ]

    scenario = read_motion_scenario(
        row,
        paired_root=PAIRED_ROOT,
        compact_record_offset=int(
            formal[
                "compact_record_offset"
            ]
        ),
    )

    actual_id = str(
        scenario.scenario_id
    )

    print(
        "expected scenario ID =",
        scenario_id,
    )

    print(
        "loaded scenario ID   =",
        actual_id,
    )

    print(
        "source shard         =",
        row.source_shard,
    )

    print(
        "compact offset       =",
        formal[
            "compact_record_offset"
        ],
    )

    if (
        actual_id
        !=
        scenario_id
    ):
        raise RuntimeError(
            "Canonical validation reader "
            "loaded the wrong scenario."
        )

    if (
        int(
            scenario.current_time_index
        )
        !=
        10
    ):
        raise RuntimeError(
            "Formal reader smoke produced "
            "unexpected current_time_index."
        )

    if (
        len(
            scenario.timestamps_seconds
        )
        !=
        91
    ):
        raise RuntimeError(
            "Formal reader smoke produced "
            "unexpected WOMD timestamp count."
        )

    print(
        "WOMD 91 timestamps    = PASS"
    )

    print(
        "current index 10       = PASS"
    )

    print(
        "reader route           = PASS"
    )


def patch_runner():
    print()
    print(
        "============================================================"
    )
    print(
        "C. MINIMAL BLOCK 4.8 RUNNER PATCH"
    )
    print(
        "============================================================"
    )

    if not RUNNER.is_file():
        raise RuntimeError(
            "Block4.8 formal runner is missing."
        )

    source = RUNNER.read_text(
        encoding="utf-8"
    )

    if REPAIR_MARKER in source:
        print(
            "resolver patch = ALREADY APPLIED"
        )

        return

    if not BACKUP.is_file():
        shutil.copy2(
            RUNNER,
            BACKUP,
        )

    constants_anchor = '''PART1 = (
'''

    if (
        constants_anchor
        not in
        source
    ):
        raise RuntimeError(
            "Could not locate PART1 constant "
            "anchor in Block4.8 runner."
        )

    constants = '''# BLOCK48_CANONICAL_VALIDATION_MANIFEST_REPAIR
#
# Canonical frozen validation catalog from Stage0/data-prep.
# Do not discover validation manifests heuristically during
# formal evaluation.
CANONICAL_VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

EXPECTED_CANONICAL_VALIDATION_MANIFEST_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_CANONICAL_VALIDATION_ROWS = 44097


'''

    source = source.replace(
        constants_anchor,
        constants
        +
        constants_anchor,
        1,
    )

    pattern = re.compile(
        r'''def validation_manifest_candidates\(\):
.*?
def build_gaussian\(
''',
        flags=re.DOTALL,
    )

    replacement = '''def resolve_validation_manifest(
    formal_rows,
):
    """
    Resolve formal scenarios only through the
    canonical frozen Stage0/data-prep validation
    manifest.

    No filename heuristics, recursive searches,
    or alternative validation populations are
    permitted here.
    """

    if not CANONICAL_VALIDATION_MANIFEST.is_file():
        raise RuntimeError(
            "Canonical selected_validation.jsonl "
            "is missing."
        )

    actual_sha = file_sha256(
        CANONICAL_VALIDATION_MANIFEST
    )

    if (
        actual_sha
        !=
        EXPECTED_CANONICAL_VALIDATION_MANIFEST_SHA
    ):
        raise RuntimeError(
            "Canonical validation manifest "
            "SHA changed."
        )

    rows = read_validation_manifest(
        CANONICAL_VALIDATION_MANIFEST
    )

    if (
        len(
            rows
        )
        !=
        EXPECTED_CANONICAL_VALIDATION_ROWS
    ):
        raise RuntimeError(
            "Canonical validation-manifest "
            "row count changed."
        )

    by_id = {}

    for row in rows:
        scenario_id = str(
            row.scenario_id
        )

        if scenario_id in by_id:
            raise RuntimeError(
                "Duplicate scenario_id in "
                "canonical validation manifest."
            )

        by_id[
            scenario_id
        ] = row

    for formal in formal_rows:
        scenario_id = str(
            formal[
                "scenario_id"
            ]
        )

        row = by_id.get(
            scenario_id
        )

        if row is None:
            raise RuntimeError(
                "Frozen formal scenario is "
                "missing from canonical "
                "validation manifest: "
                f"{scenario_id}"
            )

        if (
            str(
                row.source_shard
            )
            !=
            str(
                formal[
                    "source_shard"
                ]
            )
        ):
            raise RuntimeError(
                "Formal/canonical source_shard "
                "mismatch for "
                f"{scenario_id}"
            )

        if (
            str(
                row.selection_hash
            )
            !=
            str(
                formal[
                    "selection_hash"
                ]
            )
        ):
            raise RuntimeError(
                "Formal/canonical selection_hash "
                "mismatch for "
                f"{scenario_id}"
            )

        if (
            str(
                row.split
            ).lower()
            !=
            "validation"
        ):
            raise RuntimeError(
                "Formal scenario does not map "
                "to validation split: "
                f"{scenario_id}"
            )

    return (
        CANONICAL_VALIDATION_MANIFEST,
        rows,
        by_id,
    )


def build_gaussian(
'''

    (
        source,
        replacements,
    ) = pattern.subn(
        replacement,
        source,
        count=1,
    )

    if replacements != 1:
        raise RuntimeError(
            "Expected exactly one heuristic "
            "validation-resolver block to patch; "
            f"found {replacements}."
        )

    RUNNER.write_text(
        source,
        encoding="utf-8",
    )

    print(
        "resolver patch         = APPLIED"
    )

    print(
        "backup                 =",
        BACKUP,
    )

    print(
        "heuristic discovery    = REMOVED"
    )

    print(
        "canonical manifest     =",
        CANONICAL_VALIDATION_MANIFEST,
    )


def compile_gate():
    print()
    print(
        "============================================================"
    )
    print(
        "D. CONTROLLED COMPILE"
    )
    print(
        "============================================================"
    )

    paths = (
        RUNNER,
        SAFE_CONTROLLER,
        FORMAL_RUNTIME,
        FORMAL_TEST,
    )

    for path in paths:
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


def regression_gate():
    print()
    print(
        "============================================================"
    )
    print(
        "E. FULL STAGE4 REGRESSION BEFORE RESUME"
    )
    print(
        "============================================================"
    )

    process = subprocess.run(
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if process.stdout:
        print(
            process.stdout,
            end="",
        )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout
        or
        "",
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        process.returncode != 0
        or
        count != 114
    ):
        raise RuntimeError(
            "Expected 114/114 Stage4 "
            f"tests before resume; got {count}."
        )

    print(
        "full Stage4 regression = 114 / 114 PASS"
    )


def resume():
    print()
    print(
        "============================================================"
    )
    print(
        "F. RESUMING BLOCK 4.8 PART 2/2"
    )
    print(
        "============================================================"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                SAFE_CONTROLLER
            ),
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    with RESUME_LOG.open(
        "w",
        encoding="utf-8",
    ) as log:

        if process.stdout is not None:
            for line in (
                process.stdout
            ):
                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

    raw_code = (
        process.wait()
    )

    print()
    print(
        "safe-controller raw code =",
        raw_code,
    )

    print(
        "terminal remains open    = YES"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 PART 2/2 — "
        "CANONICAL VALIDATION RESOLVER REPAIR"
    )
    print(
        "============================================================"
    )

    required = (
        RUNNER,
        SAFE_CONTROLLER,
        FORMAL_RUNTIME,
        FORMAL_TEST,
        FORMAL_MANIFEST,
        CANONICAL_VALIDATION_MANIFEST,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_files",
            (
                "Missing: "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only the missing "
                "file. Do not retrain or "
                "recalibrate."
            ),
        )

        return

    try:
        (
            formal_rows,
            validation_rows,
            by_id,
            manifest_sha,
        ) = audit_canonical_validation_route()

    except Exception as exc:
        blocked(
            "canonical_validation_audit",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Do not patch the formal runner "
                "until the canonical validation "
                "catalog agrees with all 120 "
                "frozen formal rows."
            ),
        )

        return

    try:
        one_scenario_reader_smoke(
            formal_rows,
            by_id,
        )

    except Exception as exc:
        blocked(
            "one_scenario_reader_smoke",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Inspect only the canonical "
                "validation reader/path. "
                "No neural inference has run."
            ),
        )

        return

    try:
        patch_runner()
        compile_gate()
        regression_gate()

    except Exception as exc:
        blocked(
            "patch_or_pre_resume_gate",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Repair only the new Block4.8 "
                "resolver patch. Frozen models "
                "and formal population remain "
                "untouched."
            ),
        )

        return

    write_json({
        "stage":
            4,

        "block":
            "4.8_part_2_validation_resolver_repair",

        "status":
            "PASS",

        "canonical_validation_manifest":
            str(
                CANONICAL_VALIDATION_MANIFEST
            ),

        "canonical_validation_manifest_sha256":
            manifest_sha,

        "validation_rows":
            len(
                validation_rows
            ),

        "formal_rows":
            len(
                formal_rows
            ),

        "formal_IDs_resolved":
            120,

        "source_shard_matches":
            120,

        "selection_hash_matches":
            120,

        "one_scenario_reader_smoke":
            True,

        "heuristic_manifest_search_removed":
            True,

        "regression":
            "114/114 PASS",

        "formal_neural_inference_started_before_repair":
            False,

        "training":
            False,

        "recalibration":
            False,

        "classical_rerun":
            False,

        "upstream_modified":
            False,
    })

    print()
    print(
        "============================================================"
    )
    print(
        "VALIDATION RESOLVER REPAIR = PASS"
    )
    print(
        "============================================================"
    )

    print(
        "canonical manifest SHA = PASS"
    )

    print(
        "validation rows        = 44097 PASS"
    )

    print(
        "formal IDs             = 120 / 120 PASS"
    )

    print(
        "source_shard match     = 120 / 120 PASS"
    )

    print(
        "selection_hash match   = 120 / 120 PASS"
    )

    print(
        "one-scenario reader    = PASS"
    )

    print(
        "full regression        = 114 / 114 PASS"
    )

    print(
        "training               = NO"
    )

    print(
        "recalibration          = NO"
    )

    print(
        "classical rerun        = NO"
    )

    print(
        "safe to resume N=120   = YES"
    )

    resume()


try:
    main()

except BaseException as exc:
    blocked(
        "unexpected_repair_controller_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
        (
            "Unexpected resolver-repair "
            "controller failure. No shell "
            "exit is issued."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

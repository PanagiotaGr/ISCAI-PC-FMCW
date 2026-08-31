from __future__ import annotations

import json
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys
import time
import traceback


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

DISCOVERY = (
    STAGE4
    / "reports/"
      "block48_class_semantics_discovery.json"
)

REPORT = (
    STAGE4
    / "reports/"
      "block48_class_semantics_repair.json"
)

CACHE_DIR = (
    STAGE4
    / "artifacts/block48/cache"
)

BACKUP = (
    STAGE4
    / "artifacts/block48/"
      "run_block48_formal_evaluation."
      "pre_class_semantics_repair.py"
)

RESUME_LOG = (
    STAGE4
    / "artifacts/block48/"
      "part2_after_class_semantics_repair.log"
)

PATCH_MARKER = (
    "# BLOCK48_SEMANTIC_CLASS_MAPPING_REPAIR"
)


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
            "4.8_part_2_class_semantics_repair",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "training":
            False,

        "model_selection":
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
        "BLOCK 4.8 CLASS SEMANTICS REPAIR = BLOCKED"
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
        "training             = NO"
    )

    print(
        "recalibration        = NO"
    )

    print(
        "classical rerun      = NO"
    )

    print(
        "upstream modified    = NO"
    )

    print(
        "terminal remains open= YES"
    )


def validate_discovery():
    print()
    print(
        "============================================================"
    )
    print(
        "A. CLASS-SEMANTICS DISCOVERY VERIFICATION"
    )
    print(
        "============================================================"
    )

    if not DISCOVERY.is_file():
        raise RuntimeError(
            "Class-semantics discovery report "
            "is missing."
        )

    payload = json.loads(
        DISCOVERY.read_text(
            encoding="utf-8"
        )
    )

    if (
        payload.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Class-semantics discovery "
            "is not PASS."
        )

    if (
        payload.get(
            "diagnosis"
        )
        !=
        "NUMERIC_ENCODING_DIFFERENCE_ONLY"
    ):
        raise RuntimeError(
            "Discovery does not support "
            "numeric-encoding-only repair."
        )

    if (
        int(
            payload.get(
                "semantic_mismatch_count",
                -1,
            )
        )
        !=
        0
    ):
        raise RuntimeError(
            "Semantic class mismatches exist."
        )

    if not bool(
        payload.get(
            "all_semantic_classes_match"
        )
    ):
        raise RuntimeError(
            "Semantic class identity "
            "was not proven."
        )

    samples = int(
        payload.get(
            "supervised_samples_examined",
            0,
        )
    )

    if samples <= 0:
        raise RuntimeError(
            "Discovery examined no "
            "supervised samples."
        )

    source_counts = (
        payload.get(
            "truth_index_sources",
            {}
        )
    )

    if (
        set(
            source_counts
        )
        !=
        {
            "sample.truth_track_index"
        }
    ):
        raise RuntimeError(
            "Unexpected truth-index metadata "
            f"source(s): {source_counts}"
        )

    print(
        "diagnosis                    = "
        "NUMERIC_ENCODING_DIFFERENCE_ONLY PASS"
    )

    print(
        "semantic mismatch count      = 0 PASS"
    )

    print(
        "supervised samples examined  =",
        samples,
    )

    print(
        "truth-index source           = "
        "sample.truth_track_index PASS"
    )

    return payload


def archive_pre_semantic_cache():
    """
    The failed schema probe should have produced no shard,
    but the pre-repair cache contract did not encode class
    semantics. Archive any unexpected shards rather than
    risk reusing them.
    """

    print()
    print(
        "============================================================"
    )
    print(
        "B. PRE-REPAIR FORMAL CACHE AUDIT"
    )
    print(
        "============================================================"
    )

    if not CACHE_DIR.is_dir():
        print(
            "existing formal shards = 0"
        )

        return {
            "existing_shards":
                0,

            "archived":
                False,

            "archive_path":
                None,
        }

    shards = sorted(
        CACHE_DIR.glob(
            "*.npz"
        )
    )

    print(
        "existing formal shards =",
        len(
            shards
        ),
    )

    if not shards:
        return {
            "existing_shards":
                0,

            "archived":
                False,

            "archive_path":
                None,
        }

    archive = (
        STAGE4
        / "artifacts/block48"
        /
        (
            "cache_pre_semantic_mapping_"
            f"{int(time.time())}"
        )
    )

    archive.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in shards:
        shutil.move(
            str(
                path
            ),
            str(
                archive
                /
                path.name
            ),
        )

    print(
        "pre-repair shards archived =",
        len(
            shards
        ),
    )

    print(
        "archive path =",
        archive,
    )

    print(
        "shards deleted = NO"
    )

    return {
        "existing_shards":
            len(
                shards
            ),

        "archived":
            True,

        "archive_path":
            str(
                archive
            ),
    }


def patch_runner():
    print()
    print(
        "============================================================"
    )
    print(
        "C. MINIMAL FORMAL-EVALUATOR SEMANTIC PATCH"
    )
    print(
        "============================================================"
    )

    if not RUNNER.is_file():
        raise RuntimeError(
            "Block4.8 formal runner "
            "is missing."
        )

    source = RUNNER.read_text(
        encoding="utf-8"
    )

    if PATCH_MARKER in source:
        print(
            "semantic patch = ALREADY APPLIED"
        )

        return

    if (
        "# BLOCK48_CANONICAL_VALIDATION_MANIFEST_REPAIR"
        not in
        source
    ):
        raise RuntimeError(
            "Canonical validation-manifest "
            "repair is not present."
        )

    if not BACKUP.is_file():
        shutil.copy2(
            RUNNER,
            BACKUP,
        )

    # ========================================================
    # 1. Add semantic mapping helpers immediately
    #    before infer_scene().
    # ========================================================

    infer_anchor = '''def infer_scene(
'''

    if (
        infer_anchor
        not in
        source
    ):
        raise RuntimeError(
            "Could not locate infer_scene()."
        )

    helpers = '''# BLOCK48_SEMANTIC_CLASS_MAPPING_REPAIR
#
# Stage4 internal class IDs and WOMD protobuf object_type
# integers are different numeric domains. Formal evaluation
# therefore compares semantic enum names and explicitly maps
# WOMD truth classes into the frozen Stage4 internal IDs.


def canonical_actor_class_name(
    value,
):
    text = (
        str(value)
        .strip()
        .upper()
    )

    if text.startswith(
        "TYPE_"
    ):
        text = text[
            len(
                "TYPE_"
            ):
        ]

    return text


def womd_object_type_name(
    track,
):
    code = int(
        track.object_type
    )

    try:
        field = (
            track.DESCRIPTOR
            .fields_by_name[
                "object_type"
            ]
        )

        enum_type = (
            field.enum_type
        )

        enum_value = (
            enum_type
            .values_by_number[
                code
            ]
        )

        return str(
            enum_value.name
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not resolve symbolic WOMD "
            "object_type enum name for code "
            f"{code}: "
            f"{type(exc).__name__}: {exc}"
        )


def stage4_class_id_for_womd_track(
    track,
):
    womd_name = canonical_actor_class_name(
        womd_object_type_name(
            track
        )
    )

    matches = []

    for class_id, class_name in (
        CLASS_ID_TO_NAME.items()
    ):
        if (
            canonical_actor_class_name(
                class_name
            )
            ==
            womd_name
        ):
            matches.append(
                int(
                    class_id
                )
            )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "WOMD actor class does not map "
            "uniquely into frozen Stage4 "
            "class IDs: "
            f"WOMD={womd_name}, "
            f"matches={matches}"
        )

    return matches[
        0
    ]


'''

    source = source.replace(
        infer_anchor,
        helpers
        +
        infer_anchor,
        1,
    )

    # ========================================================
    # 2. Convert eligible WOMD classes into the
    #    Stage4 internal class-ID domain immediately
    #    after tracks_to_predict is read.
    # ========================================================

    eligible_old = '''    eligible = eligible_formal_targets(
        scenario
    )

    target_set = set(
'''

    eligible_new = '''    eligible = eligible_formal_targets(
        scenario
    )

    # eligible_formal_targets() correctly reads the
    # evaluator-only WOMD tracks_to_predict metadata, but
    # its class_id values are raw WOMD object_type integers.
    # Convert them into the frozen Stage4 internal class-ID
    # domain before any class-specific aggregation.
    eligible[
        "class_id"
    ] = np.asarray(
        [
            stage4_class_id_for_womd_track(
                scenario.tracks[
                    int(
                        truth_index
                    )
                ]
            )
            for truth_index in (
                eligible[
                    "track_index"
                ].tolist()
            )
        ],
        dtype=np.int64,
    )

    target_set = set(
'''

    if (
        eligible_old
        not in
        source
    ):
        raise RuntimeError(
            "Expected eligible_formal_targets "
            "block was not found."
        )

    source = source.replace(
        eligible_old,
        eligible_new,
        1,
    )

    # ========================================================
    # 3. Replace incorrect integer equality with equality
    #    in the common Stage4 class-ID domain.
    # ========================================================

    class_pattern = re.compile(
        r'''    # Cross-check class identity against WOMD truth track\.
    for local_index, truth_index in enumerate\(
        selected_truth_index\.tolist\(\)
    \):
        expected_class = int\(
            scenario\.tracks\[
                truth_index
            \]\.object_type
        \)

        actual_class = int\(
            class_all\[
                selected_mask
            \]\[
                local_index
            \]
        \)

        if \(
            actual_class
            !=
            expected_class
        \):
            raise RuntimeError\(
                "Neural sample actor class does "
                "not match WOMD truth track\."
            \)
''',
        flags=re.MULTILINE,
    )

    class_replacement = '''    # Cross-check semantic class identity.
    #
    # Do NOT compare raw integer values here:
    # Stage4 class_id and WOMD object_type intentionally
    # use different numeric encodings.
    selected_classes = class_all[
        selected_mask
    ]

    for local_index, truth_index in enumerate(
        selected_truth_index.tolist()
    ):
        expected_stage4_class = (
            stage4_class_id_for_womd_track(
                scenario.tracks[
                    truth_index
                ]
            )
        )

        actual_stage4_class = int(
            selected_classes[
                local_index
            ]
        )

        if (
            actual_stage4_class
            !=
            expected_stage4_class
        ):
            raise RuntimeError(
                "Neural sample actor class has a "
                "true semantic mismatch with the "
                "resolved WOMD truth track: "
                f"Stage4={actual_stage4_class}, "
                f"WOMD-mapped={expected_stage4_class}, "
                f"truth_track_index={truth_index}"
            )
'''

    (
        source,
        class_replacements,
    ) = class_pattern.subn(
        class_replacement,
        source,
        count=1,
    )

    if (
        class_replacements
        !=
        1
    ):
        raise RuntimeError(
            "Expected exactly one raw integer "
            "class-comparison block; found "
            f"{class_replacements}."
        )

    # ========================================================
    # 4. Version the formal cache contract so any shard
    #    from a pre-semantic evaluator cannot be reused.
    # ========================================================

    cache_old = '''        "Stage3_formal":
            EXPECTED_STAGE3_FORMAL_RUN_SHA,
    }
'''

    cache_new = '''        "Stage3_formal":
            EXPECTED_STAGE3_FORMAL_RUN_SHA,

        "class_identity_mapping":
            (
                "WOMD_enum_semantics_to_"
                "Stage4_internal_class_id_v1"
            ),
    }
'''

    if (
        cache_old
        not in
        source
    ):
        raise RuntimeError(
            "Could not locate Block4.8 "
            "cache-contract payload."
        )

    source = source.replace(
        cache_old,
        cache_new,
        1,
    )

    RUNNER.write_text(
        source,
        encoding="utf-8",
    )

    print(
        "semantic helper             = ADDED"
    )

    print(
        "raw integer class compare   = REMOVED"
    )

    print(
        "eligible WOMD class mapping = ADDED"
    )

    print(
        "cache contract              = VERSIONED"
    )

    print(
        "backup                      =",
        BACKUP,
    )


def static_source_gate():
    print()
    print(
        "============================================================"
    )
    print(
        "D. STATIC SEMANTIC-MAPPING GATE"
    )
    print(
        "============================================================"
    )

    source = RUNNER.read_text(
        encoding="utf-8"
    )

    required_fragments = (
        PATCH_MARKER,
        "stage4_class_id_for_womd_track",
        "WOMD_enum_semantics_to_",
        "eligible[",
        '"class_id"',
    )

    missing = [
        fragment
        for fragment in (
            required_fragments
        )
        if fragment
        not in source
    ]

    if missing:
        raise RuntimeError(
            "Semantic patch is incomplete: "
            +
            repr(
                missing
            )
        )

    old_failure_text = (
        '"Neural sample actor class does "'
        '\n'
        '                "not match WOMD truth track."'
    )

    if old_failure_text in source:
        raise RuntimeError(
            "Old raw integer class-check "
            "code still exists."
        )

    print(
        "semantic mapping helper    = PASS"
    )

    print(
        "eligible-class conversion = PASS"
    )

    print(
        "old integer comparison    = ABSENT PASS"
    )

    print(
        "cache contract version    = PASS"
    )


def compile_gate():
    print()
    print(
        "============================================================"
    )
    print(
        "E. CONTROLLED COMPILE"
    )
    print(
        "============================================================"
    )

    for path in (
        RUNNER,
        SAFE_CONTROLLER,
        FORMAL_RUNTIME,
        FORMAL_TEST,
    ):
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
        "F. FULL STAGE4 REGRESSION BEFORE RESUME"
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
            "Expected 114/114 tests "
            f"before formal resume; got {count}."
        )

    print(
        "full Stage4 regression = "
        "114 / 114 PASS"
    )


def resume():
    print()
    print(
        "============================================================"
    )
    print(
        "G. RESUMING BLOCK 4.8 PART 2/2"
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

        if (
            process.stdout
            is not None
        ):
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
        "SEMANTIC CLASS REPAIR"
    )
    print(
        "============================================================"
    )

    required = (
        RUNNER,
        SAFE_CONTROLLER,
        FORMAL_RUNTIME,
        FORMAL_TEST,
        DISCOVERY,
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
                "Block4.8 artifact."
            ),
        )
        return

    try:
        discovery = (
            validate_discovery()
        )

    except Exception as exc:
        blocked(
            "semantic_discovery_validation",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Do not apply a semantic "
                "mapping unless discovery "
                "proves zero semantic mismatches."
            ),
        )
        return

    try:
        cache_audit = (
            archive_pre_semantic_cache()
        )

        patch_runner()
        static_source_gate()
        compile_gate()
        regression_gate()

    except Exception as exc:
        blocked(
            "repair_pre_resume_gate",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Repair only the new Block4.8 "
                "semantic mapping. Do not alter "
                "truth association or frozen "
                "upstream artifacts."
            ),
        )
        return

    write_json({
        "stage":
            4,

        "block":
            "4.8_part_2_class_semantics_repair",

        "status":
            "PASS",

        "root_cause":
            (
                "Stage4 internal class_id and "
                "WOMD protobuf object_type have "
                "different numeric encodings."
            ),

        "discovery_supervised_samples":
            int(
                discovery[
                    "supervised_samples_examined"
                ]
            ),

        "semantic_mismatch_count":
            0,

        "truth_index_source":
            "sample.truth_track_index",

        "repair": {
            "comparison":
                (
                    "WOMD symbolic enum semantics "
                    "mapped into frozen Stage4 "
                    "internal class IDs"
                ),

            "eligible_class_domain":
                "Stage4_internal_class_id",

            "raw_integer_equality_removed":
                True,

            "cache_contract_versioned":
                True,
        },

        "pre_repair_cache":
            cache_audit,

        "regression":
            "114/114 PASS",

        "training":
            False,

        "model_selection":
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
        "BLOCK 4.8 SEMANTIC CLASS REPAIR = PASS"
    )
    print(
        "============================================================"
    )

    print(
        "truth-track metadata       = VALIDATED"
    )

    print(
        "semantic mismatches        = 0"
    )

    print(
        "numeric domains identical  = NO"
    )

    print(
        "semantic domains identical = YES"
    )

    print(
        "eligible class domain      = STAGE4 INTERNAL"
    )

    print(
        "raw integer equality       = REMOVED"
    )

    print(
        "cache contract             = VERSIONED"
    )

    print(
        "full regression            = 114 / 114 PASS"
    )

    print(
        "training                   = NO"
    )

    print(
        "recalibration              = NO"
    )

    print(
        "classical rerun            = NO"
    )

    print(
        "safe to resume formal N=120= YES"
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
            "Unexpected semantic-repair "
            "controller failure. No shell "
            "exit is issued."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

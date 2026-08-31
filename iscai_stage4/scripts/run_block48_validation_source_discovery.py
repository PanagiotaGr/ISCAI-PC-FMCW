from __future__ import annotations

import inspect
import json
from pathlib import Path
import py_compile
import re
import traceback

from iscai_stage3.validation import (
    read_motion_scenario,
    read_validation_manifest,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

DATA_PREP = (
    ROOT
    / "iscai_data_prep"
)

FORMAL_REPORT = (
    STAGE3
    / "reports/"
      "block38f_formal_evaluation.json"
)

STAGE3_CLOSURE = (
    STAGE3
    / "reports/"
      "stage3_final_closure.json"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

REPORT = (
    STAGE4
    / "reports/"
      "block48_validation_source_discovery.json"
)


def write_json(
    payload,
):
    REPORT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


def walk_strings(
    value,
    prefix="",
):
    result = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():
            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            result.extend(
                walk_strings(
                    item,
                    child,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value
        ):
            result.extend(
                walk_strings(
                    item,
                    f"{prefix}[{index}]",
                )
            )

    elif isinstance(
        value,
        str,
    ):
        result.append(
            (
                prefix,
                value,
            )
        )

    return result


def print_source(
    title,
    function,
):
    print()
    print(
        "============================================================"
    )
    print(
        title
    )
    print(
        "============================================================"
    )

    try:
        print(
            "source file =",
            inspect.getsourcefile(
                function
            ),
        )

        print(
            "signature   =",
            inspect.signature(
                function
            ),
        )

        print()
        print(
            inspect.getsource(
                function
            )
        )

    except Exception as exc:
        print(
            "SOURCE INSPECTION FAILED:",
            type(exc).__name__,
            str(exc),
        )


def source_search():
    """
    Search Stage3 source/scripts only.

    This is small and safe; no TFRecord/data scan.
    """

    roots = (
        STAGE3 / "src",
        STAGE3 / "scripts",
    )

    tokens = (
        "formal_validation_120",
        "block38f",
        "read_motion_scenario",
        "read_validation_manifest",
        "compact_record_offset",
        "ValidationManifestRow",
    )

    matches = []

    for root in roots:
        if not root.is_dir():
            continue

        for path in root.rglob(
            "*.py"
        ):
            try:
                lines = path.read_text(
                    encoding="utf-8"
                ).splitlines()

            except Exception:
                continue

            matching_lines = []

            for number, line in enumerate(
                lines,
                start=1,
            ):
                if any(
                    token
                    in
                    line
                    for token in tokens
                ):
                    matching_lines.append(
                        number
                    )

            if not matching_lines:
                continue

            contexts = []

            already = set()

            for number in matching_lines:
                start = max(
                    1,
                    number - 5,
                )

                stop = min(
                    len(lines),
                    number + 5,
                )

                block_key = (
                    start,
                    stop,
                )

                if block_key in already:
                    continue

                already.add(
                    block_key
                )

                context = [
                    (
                        line_number,
                        lines[
                            line_number - 1
                        ],
                    )
                    for line_number in range(
                        start,
                        stop + 1,
                    )
                ]

                contexts.append(
                    context
                )

            matches.append({
                "path":
                    str(path),

                "contexts":
                    contexts,
            })

    return matches


def likely_path_strings(
    payload,
):
    candidates = []

    for key, value in walk_strings(
        payload
    ):
        lower = value.lower()

        if (
            "/"
            in value
            or
            lower.endswith(
                ".json"
            )
            or
            lower.endswith(
                ".jsonl"
            )
            or
            "validation"
            in lower
            or
            "manifest"
            in lower
        ):
            candidate = {
                "json_key":
                    key,

                "value":
                    value,
            }

            try:
                path = Path(
                    value
                )

                candidate[
                    "exists_as_written"
                ] = path.exists()

                if (
                    not path.is_absolute()
                ):
                    rooted = (
                        ROOT
                        /
                        path
                    )

                    candidate[
                        "exists_under_waymo_root"
                    ] = rooted.exists()

                    candidate[
                        "waymo_root_candidate"
                    ] = str(
                        rooted
                    )

            except Exception:
                pass

            candidates.append(
                candidate
            )

    return candidates


def filename_search():
    """
    Filename-only search over metadata/project trees.

    No contents of TFRecords or LiDAR files are read.
    """

    roots = (
        STAGE3,
        DATA_PREP,
    )

    patterns = (
        "*manifest*.json",
        "*manifest*.jsonl",
        "*validation*.json",
        "*validation*.jsonl",
        "*formal*.json",
        "*formal*.jsonl",
    )

    paths = set()

    for root in roots:
        if not root.is_dir():
            continue

        for pattern in patterns:
            try:
                for path in root.rglob(
                    pattern
                ):
                    if path.is_file():
                        paths.add(
                            path.resolve()
                        )

            except Exception:
                pass

    return tuple(
        sorted(
            paths,
            key=str,
        )
    )


def first_formal_id_search(
    scenario_id,
    candidate_files,
):
    """
    Search only metadata files and cap reads at
    256 MiB/file. No binary dataset scan.
    """

    matches = []

    maximum_size = (
        256
        *
        1024
        *
        1024
    )

    for path in candidate_files:
        try:
            size = path.stat().st_size

            if size > maximum_size:
                matches.append({
                    "path":
                        str(path),

                    "size_bytes":
                        size,

                    "status":
                        "SKIPPED_OVER_256_MIB",
                })

                continue

            found_lines = []

            with path.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as stream:
                for line_number, line in enumerate(
                    stream,
                    start=1,
                ):
                    if scenario_id in line:
                        found_lines.append(
                            (
                                line_number,
                                line[
                                    :1000
                                ].rstrip(),
                            )
                        )

                        if len(
                            found_lines
                        ) >= 3:
                            break

            if found_lines:
                matches.append({
                    "path":
                        str(path),

                    "size_bytes":
                        size,

                    "status":
                        "FOUND",

                    "matches":
                        found_lines,
                })

        except Exception as exc:
            matches.append({
                "path":
                    str(path),

                "status":
                    (
                        "ERROR: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
            })

    return matches


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 — VALIDATION DATA-SOURCE DISCOVERY"
    )
    print(
        "READ ONLY / NO FORMAL INFERENCE"
    )
    print(
        "============================================================"
    )

    required = (
        FORMAL_REPORT,
        STAGE3_CLOSURE,
        FORMAL_MANIFEST,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "reason = missing frozen artifact(s):",
            missing,
        )
        return

    rows = [
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
    ]

    first_id = str(
        rows[
            0
        ][
            "scenario_id"
        ]
    )

    print(
        "formal rows             =",
        len(
            rows
        ),
    )

    print(
        "first formal scenario   =",
        first_id,
    )

    # ========================================================
    # A. Exact reader implementation
    # ========================================================

    print_source(
        "A1. read_motion_scenario SOURCE",
        read_motion_scenario,
    )

    print_source(
        "A2. read_validation_manifest SOURCE",
        read_validation_manifest,
    )

    # ========================================================
    # B. Stage3 formal report/closure path evidence
    # ========================================================

    formal_payload = json.loads(
        FORMAL_REPORT.read_text(
            encoding="utf-8"
        )
    )

    closure_payload = json.loads(
        STAGE3_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    formal_paths = likely_path_strings(
        formal_payload
    )

    closure_paths = likely_path_strings(
        closure_payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "B. PATH-LIKE STRINGS IN FROZEN STAGE3 REPORTS"
    )
    print(
        "============================================================"
    )

    print()
    print(
        "--- block38f_formal_evaluation.json ---"
    )

    for item in formal_paths:
        print(
            item
        )

    print()
    print(
        "--- stage3_final_closure.json ---"
    )

    for item in closure_paths:
        print(
            item
        )

    # ========================================================
    # C. Exact Stage3 source call sites
    # ========================================================

    source_matches = source_search()

    print()
    print(
        "============================================================"
    )
    print(
        "C. STAGE3 SOURCE CALL-SITE EVIDENCE"
    )
    print(
        "============================================================"
    )

    if not source_matches:
        print(
            "<no matching Stage3 Python source>"
        )

    for item in source_matches:
        print()
        print(
            "FILE:",
            item[
                "path"
            ],
        )

        for context in item[
            "contexts"
        ]:
            print(
                "-----"
            )

            for number, line in context:
                print(
                    f"{number:5d}: {line}"
                )

    # ========================================================
    # D. Broader metadata filename inventory
    # ========================================================

    candidate_files = (
        filename_search()
    )

    print()
    print(
        "============================================================"
    )
    print(
        "D. METADATA FILE CANDIDATES"
    )
    print(
        "============================================================"
    )

    print(
        "candidate count =",
        len(
            candidate_files
        ),
    )

    for path in candidate_files:
        try:
            size = path.stat().st_size
        except Exception:
            size = None

        print(
            path,
            "| bytes =",
            size,
        )

    # ========================================================
    # E. Search first frozen formal ID only inside metadata
    # ========================================================

    id_matches = first_formal_id_search(
        first_id,
        candidate_files,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "E. FIRST FORMAL SCENARIO ID IN METADATA"
    )
    print(
        "============================================================"
    )

    if not id_matches:
        print(
            "<first formal scenario ID not found "
            "in candidate metadata files>"
        )

    for item in id_matches:
        print()
        print(
            json.dumps(
                item,
                indent=2,
                default=str,
            )
        )

    write_json({
        "status":
            "PASS",

        "purpose":
            "read_only_validation_source_discovery",

        "formal_scenario_count":
            len(
                rows
            ),

        "first_formal_scenario_id":
            first_id,

        "read_motion_scenario_source_file":
            inspect.getsourcefile(
                read_motion_scenario
            ),

        "read_motion_scenario_signature":
            str(
                inspect.signature(
                    read_motion_scenario
                )
            ),

        "read_motion_scenario_source":
            inspect.getsource(
                read_motion_scenario
            ),

        "read_validation_manifest_source_file":
            inspect.getsourcefile(
                read_validation_manifest
            ),

        "read_validation_manifest_signature":
            str(
                inspect.signature(
                    read_validation_manifest
                )
            ),

        "read_validation_manifest_source":
            inspect.getsource(
                read_validation_manifest
            ),

        "formal_report_path_strings":
            formal_paths,

        "closure_path_strings":
            closure_paths,

        "Stage3_source_matches":
            source_matches,

        "metadata_candidate_files":
            [
                str(path)
                for path in (
                    candidate_files
                )
            ],

        "first_formal_id_metadata_matches":
            id_matches,

        "formal_model_inference_started":
            False,

        "training":
            False,

        "recalibration":
            False,

        "upstream_modified":
            False,
    })

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 VALIDATION SOURCE DISCOVERY"
    )
    print(
        "============================================================"
    )
    print(
        "reader source captured     = PASS"
    )
    print(
        "Stage3 call sites captured = PASS"
    )
    print(
        "metadata inventory         = PASS"
    )
    print(
        "formal inference started   = NO"
    )
    print(
        "upstream modified          = NO"
    )
    print(
        "STATUS = PASS"
    )
    print(
        "report =",
        REPORT,
    )
    print(
        "terminal remains open      = YES"
    )


try:
    main()

except BaseException as exc:
    print()
    print(
        "============================================================"
    )
    print(
        "VALIDATION SOURCE DISCOVERY = BLOCKED"
    )
    print(
        "============================================================"
    )
    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )
    print()
    traceback.print_exc()
    print(
        "formal inference started = NO"
    )
    print(
        "terminal remains open    = YES"
    )

# Deliberately no sys.exit().

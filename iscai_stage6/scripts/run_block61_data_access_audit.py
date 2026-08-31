from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import struct
import sys
import traceback


ROOT = Path("/home/agni/waymo")

S1 = ROOT / "iscai_stage1"
S2 = ROOT / "iscai_stage2"
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PILOT_MANIFEST = (
    S1
    / "artifacts/stage1a/manifests/"
      "tiny_pilot_validation.jsonl"
)

STAGE1_CLOSURE = (
    S1
    / "reports/stage1a/"
      "stage1a_closure_report.json"
)

BLOCK61_PART1 = (
    S6
    / "reports/"
      "block61_part1_geometry_core.json"
)

BLOCK60_CLOSURE = (
    S6
    / "reports/"
      "block60_closure.json"
)

STAGE5_CLOSURE = (
    S5
    / "reports/"
      "stage5_final_closure.json"
)

CURRENT_GENERATOR = (
    S6
    / "scripts/"
      "run_block61_part2_stage1_integration.py"
)

REPORT = (
    S6
    / "reports/"
      "block61_data_access_audit.json"
)

CANONICAL_PAIRED_ROOT = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0"
)

EXPECTED = {
    "pilot_manifest":
        (
            "2b6fd3d7e5c411455708d325fc8595af"
            "11d99f3fcba4dde621928150695ca859"
        ),

    "block61_part1":
        (
            "d83d2f6baed18ea20dd6ba40ac085cb5"
            "dcf206e5c4624df23b3eaaf7073c3346"
        ),

    "block60_closure":
        (
            "525cf827b217cc2806bfff6547c5e241"
            "b3c9db04d188f7957acea117b309c303"
        ),

    "stage5_closure":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),

    "current_generator":
        (
            "7aa401982fc13e1f72c1cad610e81228"
            "b3790aff0555ec995dd80030d73c5a6c"
        ),
}


# ============================================================
# Helpers
# ============================================================

def require(
    condition,
    message,
):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
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


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line in stream:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def canonical_bytes(
    payload,
) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def recursive_strings(
    value,
):
    if isinstance(
        value,
        str,
    ):
        yield value

    elif isinstance(
        value,
        dict,
    ):
        for item in value.values():
            yield from recursive_strings(
                item
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            yield from recursive_strings(
                item
            )


def bootstrap_project_sources():
    roots = tuple(
        ROOT
        / f"iscai_stage{i}"
        / "src"
        for i in range(7)
    )

    missing = [
        str(path)
        for path in roots
        if not path.is_dir()
    ]

    require(
        not missing,
        (
            "Missing project source root(s): "
            + ", ".join(missing)
        ),
    )

    for path in reversed(
        roots
    ):
        value = str(path)

        while value in sys.path:
            sys.path.remove(value)

        sys.path.insert(
            0,
            value,
        )

    existing = [
        item
        for item in os.environ.get(
            "PYTHONPATH",
            "",
        ).split(
            os.pathsep
        )
        if item
    ]

    merged = []
    seen = set()

    for item in (
        [
            str(path)
            for path in roots
        ]
        +
        existing
    ):
        if item not in seen:
            seen.add(item)
            merged.append(item)

    os.environ[
        "PYTHONPATH"
    ] = os.pathsep.join(
        merged
    )

    return [
        str(path)
        for path in roots
    ]


def derive_local_motion_path(
    historical_source_shard: str,
) -> tuple[
    Path,
    str,
]:
    """
    Historical example:
      validation.tfrecord-00000-of-00150

    Canonical paired example:
      paired-from-validation.tfrecord-00000-of-00150
    """

    historical_name = Path(
        historical_source_shard
    ).name

    match = re.fullmatch(
        r"validation\.tfrecord-(\d{5}-of-\d{5})",
        historical_name,
    )

    require(
        match is not None,
        (
            "Historical pilot source shard "
            "does not match expected validation "
            f"naming: {historical_name}"
        ),
    )

    suffix = match.group(1)

    local_name = (
        "paired-from-validation.tfrecord-"
        + suffix
    )

    local_path = (
        CANONICAL_PAIRED_ROOT
        / "validation"
        / "motion"
        / local_name
    )

    return (
        local_path,
        suffix,
    )


def scan_single_tfrecord_for_scenario(
    *,
    path: Path,
    target_scenario_id: str,
):
    """
    Read exactly one local motion shard until target scenario_id
    is found.

    This is a bounded read-only diagnostic of one shard, not a
    corpus scan and not the final Stage6 runtime reader.
    """

    from waymo_open_dataset.protos import (
        scenario_pb2,
    )

    require(
        path.is_file(),
        f"Local motion file missing: {path}",
    )

    records_examined = 0

    with path.open(
        "rb"
    ) as stream:

        while True:
            offset = stream.tell()

            length_bytes = stream.read(8)

            if not length_bytes:
                break

            require(
                len(length_bytes) == 8,
                (
                    "Incomplete TFRecord length "
                    f"at local offset {offset}"
                ),
            )

            payload_length = struct.unpack(
                "<Q",
                length_bytes,
            )[0]

            length_crc = stream.read(4)

            require(
                len(length_crc) == 4,
                (
                    "Incomplete TFRecord "
                    "length CRC."
                ),
            )

            payload = stream.read(
                payload_length
            )

            require(
                len(payload)
                ==
                payload_length,
                (
                    "Incomplete TFRecord payload "
                    f"at offset {offset}"
                ),
            )

            data_crc = stream.read(4)

            require(
                len(data_crc) == 4,
                (
                    "Incomplete TFRecord "
                    "payload CRC."
                ),
            )

            records_examined += 1

            scenario = (
                scenario_pb2.Scenario()
            )

            scenario.ParseFromString(
                payload
            )

            scenario_id = str(
                scenario.scenario_id
            )

            if (
                scenario_id
                ==
                target_scenario_id
            ):
                return {
                    "found":
                        True,

                    "local_record_offset":
                        int(offset),

                    "payload_length":
                        int(payload_length),

                    "records_examined":
                        int(records_examined),

                    "scenario_id":
                        scenario_id,

                    "current_time_index":
                        int(
                            scenario.current_time_index
                        ),

                    "track_count":
                        int(
                            len(
                                scenario.tracks
                            )
                        ),

                    "payload_sha256":
                        sha256(
                            payload
                        ).hexdigest(),
                }

    return {
        "found":
            False,

        "records_examined":
            int(records_examined),
    }


def discover_functions(
    function_names,
):
    results = []

    search_roots = (
        S2 / "src",
        S3 / "src",
        S4 / "src",
    )

    for source_root in search_roots:

        if not source_root.is_dir():
            continue

        for path in sorted(
            source_root.rglob(
                "*.py"
            )
        ):
            try:
                text = path.read_text(
                    encoding="utf-8"
                )

                tree = ast.parse(
                    text,
                    filename=str(path),
                )

            except Exception:
                continue

            for node in tree.body:

                if not isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue

                if (
                    node.name
                    not in
                    function_names
                ):
                    continue

                relative = (
                    path
                    .relative_to(
                        source_root
                    )
                    .with_suffix("")
                )

                module_name = ".".join(
                    relative.parts
                )

                signature_text = None
                import_error = None

                try:
                    module = (
                        importlib.import_module(
                            module_name
                        )
                    )

                    function = getattr(
                        module,
                        node.name,
                    )

                    signature_text = str(
                        inspect.signature(
                            function
                        )
                    )

                except BaseException as exc:
                    import_error = (
                        type(exc).__name__
                        + ": "
                        + str(exc)
                    )

                source_segment = (
                    ast.get_source_segment(
                        text,
                        node,
                    )
                    or
                    ""
                )

                source_lines = (
                    source_segment.splitlines()
                )

                results.append({
                    "function":
                        node.name,

                    "stage_source_root":
                        str(source_root),

                    "source_file":
                        str(path),

                    "source_sha256":
                        sha256_file(path),

                    "module":
                        module_name,

                    "signature":
                        signature_text,

                    "import_error":
                        import_error,

                    "source_preview":
                        "\n".join(
                            source_lines[:35]
                        ),
                })

    return results


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.1 — READ-ONLY DATA-ACCESS AUDIT"
    )
    print(
        "PROVENANCE PATH vs CANONICAL RUNTIME MOTION PATH"
    )
    print(
        "============================================================"
    )

    sources = bootstrap_project_sources()

    print()
    print(
        "===== A. SOURCE BOOTSTRAP / FROZEN SEALS ====="
    )

    required = (
        PILOT_MANIFEST,
        STAGE1_CLOSURE,
        BLOCK61_PART1,
        BLOCK60_CLOSURE,
        STAGE5_CLOSURE,
        CURRENT_GENERATOR,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing frozen evidence: "
            + ", ".join(missing)
        ),
    )

    require(
        sha256_file(
            PILOT_MANIFEST
        )
        ==
        EXPECTED[
            "pilot_manifest"
        ],
        "Pilot manifest SHA changed.",
    )

    require(
        sha256_file(
            BLOCK61_PART1
        )
        ==
        EXPECTED[
            "block61_part1"
        ],
        "Block6.1 Part1 report SHA changed.",
    )

    require(
        sha256_file(
            BLOCK60_CLOSURE
        )
        ==
        EXPECTED[
            "block60_closure"
        ],
        "Block6.0 closure SHA changed.",
    )

    require(
        sha256_file(
            STAGE5_CLOSURE
        )
        ==
        EXPECTED[
            "stage5_closure"
        ],
        "Stage5 closure SHA changed.",
    )

    require(
        sha256_file(
            CURRENT_GENERATOR
        )
        ==
        EXPECTED[
            "current_generator"
        ],
        (
            "Current Block6.1 Part2 generator "
            "SHA differs from expected post-"
            "importlib repair state."
        ),
    )

    print(
        "project source bootstrap = PASS"
    )

    print(
        "Stage1 pilot manifest    = EXACT PASS"
    )

    print(
        "Block6.0 closure         = EXACT PASS"
    )

    print(
        "Block6.1 Part1           = EXACT PASS"
    )

    print(
        "Stage5 closure           = EXACT PASS"
    )

    print(
        "current Part2 generator  = EXACT PASS"
    )

    # ========================================================
    # B. Manifest semantics
    # ========================================================

    print()
    print(
        "===== B. FROZEN PILOT PROVENANCE ====="
    )

    rows = load_jsonl(
        PILOT_MANIFEST
    )

    require(
        rows,
        "Frozen pilot manifest is empty.",
    )

    pilot = rows[0]

    scenario_id = str(
        pilot[
            "scenario_id"
        ]
    )

    historical_source = str(
        pilot[
            "source_shard"
        ]
    )

    historical_offset = pilot.get(
        "motion_record_offset"
    )

    print(
        "scenario_id              =",
        scenario_id,
    )

    print(
        "historical source_shard  =",
        historical_source,
    )

    print(
        "historical record offset =",
        historical_offset,
    )

    print(
        "historical path exists   =",
        Path(
            historical_source
        ).is_file(),
    )

    print(
        "source_shard semantics   = PROVENANCE / ORIGINAL LOCATION"
    )

    # ========================================================
    # C. Canonical paired root evidence
    # ========================================================

    print()
    print(
        "===== C. CANONICAL PAIRED CORPUS ====="
    )

    require(
        CANONICAL_PAIRED_ROOT.is_dir(),
        (
            "Canonical paired corpus missing: "
            f"{CANONICAL_PAIRED_ROOT}"
        ),
    )

    stage1_closure = load_json(
        STAGE1_CLOSURE
    )

    closure_strings = list(
        recursive_strings(
            stage1_closure
        )
    )

    root_mentioned = any(
        str(
            CANONICAL_PAIRED_ROOT
        )
        in
        value
        for value in closure_strings
    )

    print(
        "paired root =",
        CANONICAL_PAIRED_ROOT,
    )

    print(
        "paired root exists = PASS"
    )

    print(
        "Stage1 closure mentions paired root =",
        root_mentioned,
    )

    local_motion_path, shard_suffix = (
        derive_local_motion_path(
            historical_source
        )
    )

    require(
        local_motion_path.is_file(),
        (
            "Derived canonical local motion file "
            "does not exist: "
            f"{local_motion_path}"
        ),
    )

    print(
        "historical shard suffix =",
        shard_suffix,
    )

    print(
        "resolved local motion   =",
        local_motion_path,
    )

    print(
        "resolved local exists   = PASS"
    )

    print(
        "mapping                 = "
        "validation -> paired-from-validation"
    )

    # ========================================================
    # D. Exact scenario identity in local file
    # ========================================================

    print()
    print(
        "===== D. EXACT LOCAL SCENARIO IDENTITY ====="
    )

    local_record = (
        scan_single_tfrecord_for_scenario(
            path=local_motion_path,
            target_scenario_id=scenario_id,
        )
    )

    require(
        local_record.get(
            "found"
        )
        is True,
        (
            "Target scenario_id was not found "
            "in resolved local paired motion file."
        ),
    )

    require(
        local_record[
            "scenario_id"
        ]
        ==
        scenario_id,
        (
            "Local scenario identity mismatch."
        ),
    )

    print(
        "scenario_id match      = EXACT PASS"
    )

    print(
        "local record offset    =",
        local_record[
            "local_record_offset"
        ],
    )

    print(
        "historical offset      =",
        historical_offset,
    )

    print(
        "offset equality needed = NO"
    )

    print(
        "records examined       =",
        local_record[
            "records_examined"
        ],
    )

    print(
        "local payload length   =",
        local_record[
            "payload_length"
        ],
    )

    print(
        "local payload SHA256   =",
        local_record[
            "payload_sha256"
        ],
    )

    manifest_payload_length = (
        pilot.get(
            "motion_payload_length"
        )
    )

    payload_length_match = None

    if (
        manifest_payload_length
        is not None
    ):
        payload_length_match = (
            int(
                manifest_payload_length
            )
            ==
            int(
                local_record[
                    "payload_length"
                ]
            )
        )

        print(
            "manifest payload length=",
            manifest_payload_length,
        )

        print(
            "payload length matches =",
            payload_length_match,
            "(informational audit evidence)",
        )

    # ========================================================
    # E. Discover upstream validated readers
    # ========================================================

    print()
    print(
        "===== E. UPSTREAM MOTION READER DISCOVERY ====="
    )

    discovered = discover_functions(
        {
            "read_motion_scenario",
            "build_compact_motion_offset_index",
        }
    )

    require(
        discovered,
        (
            "Could not discover expected upstream "
            "motion-reader/index functions."
        ),
    )

    by_name = {}

    for item in discovered:
        by_name.setdefault(
            item[
                "function"
            ],
            [],
        ).append(item)

        print()
        print(
            "function =",
            item[
                "function"
            ],
        )

        print(
            "module   =",
            item[
                "module"
            ],
        )

        print(
            "source   =",
            item[
                "source_file"
            ],
        )

        print(
            "signature=",
            item[
                "signature"
            ],
        )

        print(
            "import_error =",
            item[
                "import_error"
            ],
        )

    require(
        "read_motion_scenario"
        in
        by_name,
        (
            "read_motion_scenario was not found "
            "in Stage2–4 source trees."
        ),
    )

    require(
        "build_compact_motion_offset_index"
        in
        by_name,
        (
            "build_compact_motion_offset_index "
            "was not found in Stage2–4 sources."
        ),
    )

    print()
    print(
        "read_motion_scenario             = DISCOVERED PASS"
    )

    print(
        "build_compact_motion_offset_index= DISCOVERED PASS"
    )

    # ========================================================
    # F. Correct semantics for final repair
    # ========================================================

    print()
    print(
        "===== F. DATA-ACCESS CONTRACT VERDICT ====="
    )

    verdict = {
        "historical_source_shard":
            "PROVENANCE_ONLY",

        "historical_record_offset":
            "PROVENANCE_FOR_HISTORICAL_FILE",

        "canonical_runtime_root":
            str(
                CANONICAL_PAIRED_ROOT
            ),

        "canonical_runtime_motion_file":
            str(
                local_motion_path
            ),

        "runtime_record_offset":
            int(
                local_record[
                    "local_record_offset"
                ]
            ),

        "scenario_identity":
            "EXACT_MATCH",

        "historical_offset_must_equal_runtime_offset":
            False,

        "final_Block61_reader_policy":
            (
                "reuse discovered upstream validated "
                "motion reader/index API"
            ),

        "direct_WOMD_state_role":
            (
                "GEOMETRY_INTEGRATION_VALIDATION_ONLY"
            ),

        "primary_Stage6_experimental_input":
            (
                "remains frozen track-based "
                "PC-FMCW-like posterior pipeline"
            ),
    }

    print(
        "historical source path = PROVENANCE ONLY"
    )

    print(
        "runtime data source     = CANONICAL PAIRED CORPUS"
    )

    print(
        "scenario identity       = EXACT MATCH"
    )

    print(
        "historical/local offsets= FILE-SPECIFIC / MAY DIFFER"
    )

    print(
        "final repair reader     = REUSE UPSTREAM API"
    )

    print(
        "WOMD GT role in Block6.1= GEOMETRY VALIDATION ONLY"
    )

    # ========================================================
    # G. Save audit report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.1",

        "audit":
            "data_access_semantics",

        "status":
            "PASS_READY_FOR_DATA_ACCESS_REPAIR",

        "project_source_roots":
            sources,

        "frozen_pilot": {
            "scenario_id":
                scenario_id,

            "historical_source_shard":
                historical_source,

            "historical_source_exists":
                Path(
                    historical_source
                ).is_file(),

            "historical_record_offset":
                historical_offset,

            "manifest_motion_payload_length":
                manifest_payload_length,
        },

        "canonical_runtime": {
            "paired_root":
                str(
                    CANONICAL_PAIRED_ROOT
                ),

            "paired_root_exists":
                True,

            "Stage1_closure_mentions_root":
                root_mentioned,

            "resolved_motion_path":
                str(
                    local_motion_path
                ),

            "resolved_motion_exists":
                True,

            "shard_suffix":
                shard_suffix,

            "local_record":
                local_record,

            "payload_length_match_if_available":
                payload_length_match,
        },

        "upstream_readers":
            discovered,

        "binding_verdict":
            verdict,

        "scientific_contract": {
            "Stage1_manifest_modified":
                False,

            "Stage2_4_sources_modified":
                False,

            "Stage5_modified":
                False,

            "Block61_generator_modified":
                False,

            "corpus_modified":
                False,

            "training":
                False,

            "trajectory_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,
        },
    }

    tmp = REPORT.with_suffix(
        REPORT.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(
            report
        )
    )

    os.replace(
        tmp,
        REPORT,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.1 DATA-ACCESS AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "historical source_shard   = PROVENANCE ONLY"
    )

    print(
        "canonical paired root     = PRESENT PASS"
    )

    print(
        "local paired motion file  = PRESENT PASS"
    )

    print(
        "target scenario_id        = EXACT MATCH PASS"
    )

    print(
        "local runtime offset      = RESOLVED"
    )

    print(
        "offset equality required = NO"
    )

    print(
        "upstream motion reader    = DISCOVERED PASS"
    )

    print(
        "upstream offset-index API = DISCOVERED PASS"
    )

    print(
        "WOMD annotation role      = GEOMETRY VALIDATION ONLY"
    )

    print(
        "Stage6 primary mode       = UNCHANGED"
    )

    print(
        "upstream modified         = NO"
    )

    print(
        "generator modified        = NO"
    )

    print(
        "STATUS = PASS_READY_FOR_DATA_ACCESS_REPAIR"
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
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.1 DATA-ACCESS AUDIT = BLOCKED"
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
    traceback.print_exc(
        limit=12
    )

    print()
    print(
        "Stage1 modified      = NO"
    )

    print(
        "Stage2–4 modified    = NO"
    )

    print(
        "Stage5 modified      = NO"
    )

    print(
        "Block6.1 generator   = UNCHANGED"
    )

    print(
        "corpus modified      = NO"
    )

    print(
        "training/inference   = NO"
    )

    print(
        "formal evaluation    = NO"
    )

    print(
        "parameter tuning     = NO"
    )

    print(
        "terminal remains open= YES"
    )

# Deliberately no non-zero sys.exit().

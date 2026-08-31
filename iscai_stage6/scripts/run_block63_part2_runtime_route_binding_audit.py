from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import sys
import traceback


ROOT = Path("/home/agni/waymo")

S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PART1 = (
    S6
    / "reports/block63_part1_deterministic_future_box.json"
)

BATCH_AUDIT = (
    S6
    / "reports/block63_part2_batch_row_binding_audit.json"
)

DEV_JSON = (
    S5
    / "artifacts/block52/development_gaussian_posterior.json"
)

RESOLVER_JSON = (
    S5
    / "artifacts/block52/dev_posterior_resolver_summary.json"
)

ROUTE_JSON = (
    S5
    / "artifacts/block52/actual_stage4_gaussian_route_audit.json"
)

RUNTIME_JSON = (
    S5
    / "artifacts/block52/calibrated_runtime_api_audit.json"
)

COORD_JSON = (
    S5
    / "artifacts/block52/stage4_prediction_coordinate_semantics.json"
)

CAL_RUNTIME = (
    S4
    / "src/iscai_stage4/ml/calibration_runtime.py"
)

FORMAL_RUNTIME = (
    S4
    / "src/iscai_stage4/ml/formal_runtime.py"
)

NEURAL_INPUTS = (
    S4
    / "src/iscai_stage4/data/neural_inputs.py"
)

DECISION = (
    S6
    / "configs/block63_part2_runtime_route_decision.json"
)

REPORT = (
    S6
    / "reports/block63_part2_runtime_route_binding_audit.json"
)

SOURCE_EXCERPT = (
    S6
    / "artifacts/block63/"
      "block63_part2_runtime_route_exact_source.txt"
)


EXPECTED = {
    "part1":
        (
            "871c8a81e43048ff2ccf7ef1844ad543"
            "3a859ff15b944749e13be9160b08256a"
        ),

    "batch_audit":
        (
            "a57d848cda03f002e4b16de96314372b"
            "f3d985b15568a5b9c06528e035a9311b"
        ),
}


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

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


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
        +
        "\n"
    ).encode("utf-8")


def write_new_or_exact_json(
    path: Path,
    payload,
):

    desired = canonical_bytes(
        payload
    )

    if path.exists():

        require(
            path.read_bytes()
            ==
            desired,
            (
                "Existing artifact differs: "
                f"{path}"
            ),
        )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        desired
    )

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def bootstrap():

    roots = tuple(
        ROOT
        / f"iscai_stage{i}"
        / "src"
        for i in range(7)
    )

    for path in reversed(roots):

        require(
            path.is_dir(),
            f"Missing source root: {path}",
        )

        value = str(path)

        while value in sys.path:
            sys.path.remove(value)

        sys.path.insert(
            0,
            value,
        )


def exact_definition(
    path: Path,
    name: str,
):

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    matches = []

    for node in tree.body:

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        if node.name != name:
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        if segment is None:
            continue

        matches.append({
            "path":
                str(path),

            "line":
                int(node.lineno),

            "kind":
                type(node).__name__,

            "source":
                segment,

            "sha256":
                sha256(
                    segment.encode(
                        "utf-8"
                    )
                ).hexdigest(),
        })

    return matches


def source_contexts(
    roots,
    pattern: re.Pattern,
    *,
    radius=8,
    maximum=120,
):

    results = []

    seen = set()

    for root in roots:

        if not root.exists():
            continue

        iterator = (
            root.rglob("*.py")
            if root.is_dir()
            else (root,)
        )

        for path in sorted(
            iterator
        ):

            if path in seen:
                continue

            seen.add(path)

            try:

                lines = path.read_text(
                    encoding="utf-8"
                ).splitlines()

            except BaseException:
                continue

            for index, line in enumerate(lines):

                if not pattern.search(line):
                    continue

                lo = max(
                    0,
                    index - radius,
                )

                hi = min(
                    len(lines),
                    index + radius + 1,
                )

                results.append({
                    "path":
                        str(path),

                    "line":
                        index + 1,

                    "context":
                        "\n".join(
                            f"{j + 1:04d}: {lines[j]}"
                            for j in range(
                                lo,
                                hi,
                            )
                        ),
                })

                if len(results) >= maximum:
                    return results

    return results


def flatten(
    value,
    prefix="",
    depth=0,
):

    if depth > 9:
        return []

    rows = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(
                child,
                (dict, list),
            ):

                rows.extend(
                    flatten(
                        child,
                        path,
                        depth + 1,
                    )
                )

            else:

                rows.append(
                    (
                        path,
                        child,
                    )
                )

    elif isinstance(
        value,
        list,
    ) and len(value) <= 60:

        for index, child in enumerate(value):

            path = (
                f"{prefix}[{index}]"
            )

            if isinstance(
                child,
                (dict, list),
            ):

                rows.extend(
                    flatten(
                        child,
                        path,
                        depth + 1,
                    )
                )

            else:

                rows.append(
                    (
                        path,
                        child,
                    )
                )

    return rows


def runtime_metadata(
    named_payloads,
):

    pattern = re.compile(
        r"("
        r"checkpoint|model|state_dict|weight|"
        r"normaliz|calibrat|alpha|"
        r"split|development|validation|train|"
        r"scenario|shard|tfrecord|offset|"
        r"probe|resolver|runtime|"
        r"history|prediction"
        r")",
        re.IGNORECASE,
    )

    result = {}

    for name, payload in named_payloads:

        rows = []

        for path, value in flatten(
            payload
        ):

            if not pattern.search(path):
                continue

            rows.append({
                "path":
                    path,

                "value":
                    value,
            })

        result[
            name
        ] = rows

    return result


def main():

    bootstrap()

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 PART 2/2"
    )
    print(
        "IDENTITY-PRESERVING STAGE4 RUNTIME ROUTE AUDIT"
    )
    print(
        "============================================================"
    )

    protected = (
        PART1,
        BATCH_AUDIT,
        DEV_JSON,
        RESOLVER_JSON,
        ROUTE_JSON,
        RUNTIME_JSON,
        COORD_JSON,
        CAL_RUNTIME,
        FORMAL_RUNTIME,
        NEURAL_INPUTS,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing frozen prerequisite(s): "
            + ", ".join(missing)
        ),
    )

    before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Exact Stage6 seal
    # ========================================================

    print()
    print(
        "===== A. BLOCK6.3 FROZEN SEAL ====="
    )

    require(
        sha256_file(PART1)
        ==
        EXPECTED[
            "part1"
        ],
        (
            "Block6.3 Part1 SHA mismatch."
        ),
    )

    require(
        sha256_file(BATCH_AUDIT)
        ==
        EXPECTED[
            "batch_audit"
        ],
        (
            "Batch-row audit SHA mismatch."
        ),
    )

    batch = load_json(
        BATCH_AUDIT
    )

    require(
        batch.get(
            "status"
        )
        ==
        (
            "PASS_BATCH_SCHEMA_BOUND_"
            "ROW_PROVENANCE_EXTRACTED"
        ),
        (
            "Unexpected batch-row audit status."
        ),
    )

    require(
        batch[
            "selector"
        ][
            "state"
        ]
        ==
        "NO_EXPLICIT_SELECTOR_FOUND",
        (
            "Batch artifact unexpectedly acquired "
            "a controller-safe row selector."
        ),
    )

    print(
        "Part1 = EXACT PASS"
    )

    print(
        "batch audit = EXACT PASS"
    )

    print(
        "anonymous batch row selector = NONE"
    )

    # ========================================================
    # B. Freeze route decision
    # ========================================================

    print()
    print(
        "===== B. CONTROLLER ROUTE DECISION ====="
    )

    decision = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "part":
            "2/2",

        "decision":
            (
                "USE_IDENTITY_PRESERVING_"
                "FROZEN_STAGE4_RUNTIME"
            ),

        "anonymous_batch": {
            "row_count":
                int(
                    batch[
                        "npz"
                    ][
                        "row_count"
                    ]
                ),

            "row_selector":
                None,

            "controller_input_role":
                False,

            "allowed_roles": [
                "schema_evidence",
                "calibration_evidence",
                "coordinate_invariant_evidence",
                "regression_evidence",
            ],

            "arbitrary_row_selection":
                False,

            "truth_based_row_reconstruction":
                False,
        },

        "runtime_route": {
            "source":
                "frozen_Stage4_calibrated_Gaussian_runtime",

            "record_domain":
                "development_only",

            "identity_preservation":
                (
                    "retain causal prediction_id and "
                    "latest_position_H0_m in memory "
                    "before posterior serialization"
                ),

            "output_mean_semantics":
                "metric_H0_displacement",

            "deterministic_covariance_use":
                False,

            "formal_N120":
                False,

            "parameter_tuning":
                False,
        },

        "forbidden": {
            "anonymous_batch_row_zero":
                True,

            "truth_id_join":
                True,

            "truth_track_index_join":
                True,

            "WOMD_perfect_identity_join":
                True,

            "tracks_to_predict_selection":
                True,

            "future_GT_selection":
                True,
        },
    }

    decision_status = (
        write_new_or_exact_json(
            DECISION,
            decision,
        )
    )

    print(
        "decision =",
        decision[
            "decision"
        ],
    )

    print(
        "anonymous batch controller use = NO"
    )

    print(
        "arbitrary row use = NO"
    )

    print(
        "runtime identity preservation = REQUIRED"
    )

    print(
        "formal N120 = FORBIDDEN"
    )

    print(
        "decision artifact =",
        decision_status,
    )

    print(
        "decision SHA256 =",
        sha256_file(
            DECISION
        ),
    )

    # ========================================================
    # C. Exact Stage4 runtime definitions
    # ========================================================

    print()
    print(
        "===== C. EXACT STAGE4 CALIBRATED RUNTIME DEFINITIONS ====="
    )

    requested = (
        (
            CAL_RUNTIME,
            "build_gaussian_model",
        ),
        (
            CAL_RUNTIME,
            "probe_calibrated_output",
        ),
        (
            CAL_RUNTIME,
            "infer_scene_statistics",
        ),
        (
            CAL_RUNTIME,
            "prediction_sha256",
        ),
        (
            FORMAL_RUNTIME,
            "extract_deterministic_prediction",
        ),
        (
            NEURAL_INPUTS,
            "build_causal_track_history",
        ),
        (
            NEURAL_INPUTS,
            "build_causal_scene_inputs",
        ),
    )

    definitions = {}

    excerpt_parts = []

    for path, name in requested:

        matches = exact_definition(
            path,
            name,
        )

        definitions[
            name
        ] = matches

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            name,
            "| definitions =",
            len(matches),
        )

        print(
            "------------------------------------------------------------"
        )

        for item in matches:

            print(
                "path =",
                item[
                    "path"
                ],
            )

            print(
                "line =",
                item[
                    "line"
                ],
            )

            print(
                "sha256 =",
                item[
                    "sha256"
                ],
            )

            print(
                item[
                    "source"
                ]
            )

            excerpt_parts.append(
                "\n"
                "============================================================\n"
                f"{name}\n"
                f"{item['path']}:{item['line']}\n"
                f"sha256={item['sha256']}\n"
                "============================================================\n"
                f"{item['source']}\n"
            )

    # ========================================================
    # D. Runtime signatures from importable frozen code
    # ========================================================

    print()
    print(
        "===== D. IMPORTED FROZEN RUNTIME SIGNATURES ====="
    )

    import iscai_stage4.ml.calibration_runtime as cr
    import iscai_stage4.ml.formal_runtime as fr
    import iscai_stage4.data.neural_inputs as ni

    runtime_objects = {
        "build_gaussian_model":
            cr.build_gaussian_model,

        "probe_calibrated_output":
            cr.probe_calibrated_output,

        "infer_scene_statistics":
            cr.infer_scene_statistics,

        "prediction_sha256":
            cr.prediction_sha256,

        "extract_deterministic_prediction":
            fr.extract_deterministic_prediction,

        "build_causal_track_history":
            ni.build_causal_track_history,

        "build_causal_scene_inputs":
            ni.build_causal_scene_inputs,
    }

    signatures = {}

    for name, obj in runtime_objects.items():

        signature = str(
            inspect.signature(
                obj
            )
        )

        signatures[
            name
        ] = signature

        print(
            f"{name:38s} {signature}"
        )

    # ========================================================
    # E. Exact Block5.2 producer/call-site evidence
    # ========================================================

    print()
    print(
        "===== E. BLOCK5.2 PRODUCER / CALL-SITE EVIDENCE ====="
    )

    producer_pattern = re.compile(
        r"("
        r"development_gaussian_posterior|"
        r"probe_calibrated_output|"
        r"build_gaussian_model|"
        r"infer_scene_statistics|"
        r"build_causal_scene_inputs|"
        r"build_causal_track_history|"
        r"mean_displacement_H0_m|"
        r"anchor_position_H0_m|"
        r"calibrated_covariance_H0_m2|"
        r"checkpoint|normalization|calibration"
        r")",
        re.IGNORECASE,
    )

    producer_contexts = (
        source_contexts(
            (
                S5 / "scripts",
                S5 / "src",
                S5 / "artifacts/block52",
            ),
            producer_pattern,
            radius=9,
            maximum=120,
        )
    )

    print(
        "producer contexts =",
        len(
            producer_contexts
        ),
    )

    for index, item in enumerate(
        producer_contexts[:80],
        start=1,
    ):

        print()
        print(
            f"--- producer context {index} ---"
        )

        print(
            item[
                "path"
            ],
            ":",
            item[
                "line"
            ],
            sep="",
        )

        print(
            item[
                "context"
            ]
        )

    if len(
        producer_contexts
    ) > 80:

        print(
            "...",
            len(
                producer_contexts
            ) - 80,
            "additional producer contexts stored in report",
        )

    # ========================================================
    # F. Frozen runtime metadata
    # ========================================================

    print()
    print(
        "===== F. CHECKPOINT / CALIBRATION / DEVELOPMENT METADATA ====="
    )

    named_payloads = (
        (
            "development_gaussian_posterior.json",
            load_json(
                DEV_JSON
            ),
        ),
        (
            "dev_posterior_resolver_summary.json",
            load_json(
                RESOLVER_JSON
            ),
        ),
        (
            "actual_stage4_gaussian_route_audit.json",
            load_json(
                ROUTE_JSON
            ),
        ),
        (
            "calibrated_runtime_api_audit.json",
            load_json(
                RUNTIME_JSON
            ),
        ),
        (
            "stage4_prediction_coordinate_semantics.json",
            load_json(
                COORD_JSON
            ),
        ),
    )

    metadata = runtime_metadata(
        named_payloads
    )

    for name, rows in metadata.items():

        print()
        print(
            "---",
            name,
            "---"
        )

        if not rows:

            print(
                "NO MATCHING SCALAR METADATA"
            )

            continue

        for item in rows[:100]:

            print(
                item[
                    "path"
                ],
                "=",
                item[
                    "value"
                ],
            )

    # ========================================================
    # G. Route-readiness gates
    # ========================================================

    print()
    print(
        "===== G. IDENTITY-PRESERVING RUNTIME READINESS ====="
    )

    required_definitions = (
        "build_gaussian_model",
        "probe_calibrated_output",
        "build_causal_track_history",
    )

    definition_gate = all(
        len(
            definitions[
                name
            ]
        )
        ==
        1
        for name in required_definitions
    )

    producer_text = "\n".join(
        item[
            "context"
        ]
        for item
        in producer_contexts
    ).lower()

    producer_gaussian_gate = (
        "development_gaussian_posterior"
        in producer_text
        and
        (
            "mean_displacement_h0_m"
            in producer_text
            or
            "probe_calibrated_output"
            in producer_text
        )
    )

    causal_history_gate = (
        "prediction_id"
        in definitions[
            "build_causal_track_history"
        ][0][
            "source"
        ]
        and
        "latest_position_H0_m"
        in definitions[
            "build_causal_track_history"
        ][0][
            "source"
        ]
    ) if (
        len(
            definitions[
                "build_causal_track_history"
            ]
        )
        ==
        1
    ) else False

    print(
        "exact calibrated runtime definitions =",
        (
            "PASS"
            if definition_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "Block5.2 Gaussian producer route =",
        (
            "PASS"
            if producer_gaussian_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "causal prediction_id retained pre-runtime =",
        (
            "PASS"
            if causal_history_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "anonymous NPZ row needed = NO"
    )

    print(
        "truth ID needed = NO"
    )

    if (
        definition_gate
        and
        producer_gaussian_gate
        and
        causal_history_gate
    ):

        status = (
            "PASS_READY_FOR_SINGLE_RECORD_"
            "DEVELOPMENT_RUNTIME_BINDING"
        )

    else:

        status = (
            "BLOCKED_RUNTIME_ROUTE_EVIDENCE_INCOMPLETE"
        )

    # ========================================================
    # H. Save exact source excerpt
    # ========================================================

    SOURCE_EXCERPT.write_text(
        "".join(
            excerpt_parts
        ),
        encoding="utf-8",
    )

    # ========================================================
    # I. Immutability
    # ========================================================

    print()
    print(
        "===== H. IMMUTABILITY ====="
    )

    changed = [
        str(path)
        for path in protected
        if (
            before[
                str(path)
            ]
            !=
            sha256_file(
                path
            )
        )
    ]

    require(
        not changed,
        (
            "Frozen source/artifact changed: "
            + ", ".join(changed)
        ),
    )

    print(
        "Block6.3 Part1 = UNCHANGED"
    )

    print(
        "Stage4 runtime code = UNCHANGED"
    )

    print(
        "Stage5 Block5.2 evidence = UNCHANGED"
    )

    print(
        "dataset access = NO"
    )

    print(
        "model inference = NO"
    )

    print(
        "formal evaluation = NO"
    )

    # ========================================================
    # J. Report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "part":
            "2/2",

        "audit":
            "IDENTITY_PRESERVING_RUNTIME_ROUTE",

        "status":
            status,

        "decision_artifact": {
            "path":
                str(
                    DECISION
                ),

            "sha256":
                sha256_file(
                    DECISION
                ),
        },

        "signatures":
            signatures,

        "definitions": {
            name: [
                {
                    "path":
                        item[
                            "path"
                        ],

                    "line":
                        item[
                            "line"
                        ],

                    "sha256":
                        item[
                            "sha256"
                        ],
                }
                for item in items
            ]
            for name, items
            in definitions.items()
        },

        "producer_contexts":
            producer_contexts,

        "runtime_metadata":
            metadata,

        "readiness": {
            "exact_runtime_definitions":
                definition_gate,

            "Block52_Gaussian_producer_route":
                producer_gaussian_gate,

            "causal_prediction_identity_retained":
                causal_history_gate,

            "anonymous_batch_row_required":
                False,

            "truth_identity_required":
                False,
        },

        "scope": {
            "dataset_access":
                False,

            "model_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "upstream_modified":
                False,
        },

        "source_excerpt":
            str(
                SOURCE_EXCERPT
            ),
    }

    write_new_or_exact_json(
        REPORT,
        report,
    )

    print()
    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 PART 2/2 RUNTIME-ROUTE AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "anonymous batched posterior as controller input = NO"
    )

    print(
        "runtime route = FROZEN STAGE4 CALIBRATED GAUSSIAN"
    )

    print(
        "identity preservation = IN-MEMORY CAUSAL PREDICTION_ID"
    )

    print(
        "causal anchor preservation = latest_position_H0_m"
    )

    print(
        "deterministic output = GAUSSIAN MEAN ONLY"
    )

    print(
        "predictive covariance use in Block6.3 = NO"
    )

    print(
        "truth/perfect ID = NO"
    )

    print(
        "formal N120 = NO"
    )

    print(
        "dataset access = NO"
    )

    print(
        "model inference = NO"
    )

    print(
        "STATUS =",
        status,
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
        "BLOCK 6.3 PART2 RUNTIME-ROUTE AUDIT = BLOCKED"
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
        limit=16
    )

    print()
    print(
        "Block6.3 Part1 modified = NO"
    )

    print(
        "Stage4/5 modified = NO"
    )

    print(
        "dataset access = NO"
    )

    print(
        "model inference = NO"
    )

    print(
        "formal evaluation = NO"
    )

    print(
        "terminal remains open = YES"
    )

# No sys.exit().

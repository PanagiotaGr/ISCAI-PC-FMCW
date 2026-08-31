from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")

S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

INTERFACE_AUDIT = (
    S6
    / "reports/block63_predictor_interface_audit.json"
)

BLOCK62 = (
    S6
    / "reports/block62_closure.json"
)

STAGE4_HANDOFF = (
    S4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

STAGE5_HANDOFF = (
    S5
    / "artifacts/block510/"
      "stage5_to_stage6_handoff.json"
)

COORDINATE_SEMANTICS = (
    S5
    / "artifacts/block52/"
      "stage4_prediction_coordinate_semantics.json"
)

GAUSSIAN_ROUTE_AUDIT = (
    S5
    / "artifacts/block52/"
      "actual_stage4_gaussian_route_audit.json"
)

RUNTIME_API_AUDIT = (
    S5
    / "artifacts/block52/"
      "calibrated_runtime_api_audit.json"
)

EXACT_FORMAL_ROUTE = (
    S5
    / "artifacts/block58/"
      "exact_formal_route_semantics.json"
)

FORMAL_INPUT_ROUTE = (
    S5
    / "artifacts/block58/"
      "formal_input_route_audit.json"
)

FORMAL_RECORDS = (
    S5
    / "artifacts/block58/"
      "formal_clean_records.jsonl"
)

STAGE4_FORMAL_NPZ = (
    S4
    / "artifacts/block48/"
      "formal_neural_outputs.npz"
)

STAGE4_CONTRACTS = (
    S4
    / "src/iscai_stage4/contracts.py"
)

STAGE4_GAUSSIAN = (
    S4
    / "src/iscai_stage4/ml/gaussian_gru.py"
)

STAGE4_FORMAL_RUNTIME = (
    S4
    / "src/iscai_stage4/ml/formal_runtime.py"
)

STAGE5_MONTE_CARLO = (
    S5
    / "src/iscai_stage5/angular_monte_carlo.py"
)

REPORT = (
    S6
    / "reports/"
      "block63_exact_prediction_route_audit.json"
)

SOURCE_EXCERPT = (
    S6
    / "artifacts/block63/"
      "block63_exact_prediction_route_source.txt"
)


EXPECTED = {
    "block62":
        (
            "23b299b4dc6a123ce64a6ab1350b3ed9"
            "cb149c8fed176b37e9715d9935704d92"
        ),

    "interface_audit":
        (
            "14fb92076b40c73e378a2ca249b7ebb7"
            "f95b011a0d31ad5062ac75fde01e7e9d"
        ),

    "stage4_handoff":
        (
            "491bce010d35c2a394f879ecf35ed26ef"
            "1de92f7fff1465dcbf46b072a87c6fd"
        ),
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:

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


def load_json(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(payload) -> bytes:

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


def atomic_json(path: Path, payload):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(payload)
    )

    os.replace(
        tmp,
        path,
    )


def summarize(value):

    if isinstance(value, dict):

        return {
            "type":
                "dict",

            "keys":
                list(value)[:30],

            "key_count":
                len(value),
        }

    if isinstance(value, list):

        result = {
            "type":
                "list",

            "length":
                len(value),
        }

        if len(value) <= 12:

            result["value"] = value

        return result

    return value


def flatten_matching(
    value,
    pattern,
    prefix="",
):

    results = []

    if isinstance(value, dict):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if pattern.search(path):

                results.append(
                    (
                        path,
                        summarize(child),
                    )
                )

            results.extend(
                flatten_matching(
                    child,
                    pattern,
                    path,
                )
            )

    elif isinstance(value, list):

        # Do not recursively flood large numerical arrays.
        if (
            len(value) <= 20
            and
            any(
                isinstance(
                    item,
                    (dict, list),
                )
                for item in value
            )
        ):

            for index, child in enumerate(value):

                path = (
                    f"{prefix}[{index}]"
                )

                results.extend(
                    flatten_matching(
                        child,
                        pattern,
                        path,
                    )
                )

    return results


def read_jsonl_head(
    path: Path,
    count: int = 3,
):

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line in stream:

            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

            if len(rows) >= count:
                break

    return rows


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

    found = []

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

        if segment is not None:

            found.append({
                "path":
                    str(path),

                "line":
                    int(node.lineno),

                "source":
                    segment,

                "sha256":
                    sha256(
                        segment.encode("utf-8")
                    ).hexdigest(),
            })

    require(
        len(found) == 1,
        (
            f"Expected one definition of {name} "
            f"in {path}; found {len(found)}."
        ),
    )

    return found[0]


def contexts(
    path: Path,
    tokens,
    radius=5,
):

    source = path.read_text(
        encoding="utf-8"
    )

    lines = source.splitlines()

    results = []

    for index, line in enumerate(lines):

        lower = line.lower()

        if not any(
            token.lower() in lower
            for token in tokens
        ):
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
            "line":
                index + 1,

            "text":
                "\n".join(
                    f"{j + 1:04d}: {lines[j]}"
                    for j in range(lo, hi)
                ),
        })

    return results


def print_matches(
    title,
    payload,
    pattern,
    limit=120,
):

    print()
    print(
        f"===== {title} ====="
    )

    matches = flatten_matching(
        payload,
        pattern,
    )

    print(
        "matched fields =",
        len(matches),
    )

    for path, value in matches[:limit]:

        print(
            path,
            "=",
            value,
        )

    if len(matches) > limit:

        print(
            "...",
            len(matches) - limit,
            "additional matches omitted",
        )

    return matches


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 EXACT PREDICTION-ROUTE AUDIT"
    )
    print(
        "SHARED MEAN / RECORD ID / H0 FRAME / YAW POLICY"
    )
    print(
        "============================================================"
    )

    protected = (
        BLOCK62,
        INTERFACE_AUDIT,
        STAGE4_HANDOFF,
        STAGE5_HANDOFF,
        COORDINATE_SEMANTICS,
        GAUSSIAN_ROUTE_AUDIT,
        RUNTIME_API_AUDIT,
        EXACT_FORMAL_ROUTE,
        FORMAL_INPUT_ROUTE,
        FORMAL_RECORDS,
        STAGE4_FORMAL_NPZ,
        STAGE4_CONTRACTS,
        STAGE4_GAUSSIAN,
        STAGE4_FORMAL_RUNTIME,
        STAGE5_MONTE_CARLO,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required frozen evidence: "
            + ", ".join(missing)
        ),
    )

    before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Frozen identity
    # ========================================================

    print()
    print(
        "===== A. FROZEN IDENTITY ====="
    )

    require(
        sha256_file(BLOCK62)
        ==
        EXPECTED["block62"],
        "Block6.2 closure SHA mismatch.",
    )

    require(
        sha256_file(INTERFACE_AUDIT)
        ==
        EXPECTED["interface_audit"],
        "Block6.3 interface-audit SHA mismatch.",
    )

    require(
        sha256_file(STAGE4_HANDOFF)
        ==
        EXPECTED["stage4_handoff"],
        "Stage4 handoff SHA mismatch.",
    )

    print(
        "Block6.2 closure        = EXACT PASS"
    )

    print(
        "Block6.3 interface audit= EXACT PASS"
    )

    print(
        "Stage4 handoff          = EXACT PASS"
    )

    print(
        "Stage5 handoff SHA256   =",
        sha256_file(
            STAGE5_HANDOFF
        ),
    )

    # ========================================================
    # B. Coordinate semantics
    # ========================================================

    coordinate = load_json(
        COORDINATE_SEMANTICS
    )

    coordinate_pattern = re.compile(
        r"("
        r"coordinate|frame|h0|"
        r"mean|prediction|position|"
        r"relative|absolute|offset|"
        r"x|y|z|horizon"
        r")",
        re.IGNORECASE,
    )

    coordinate_matches = print_matches(
        "B. STAGE4 PREDICTION COORDINATE SEMANTICS",
        coordinate,
        coordinate_pattern,
        limit=160,
    )

    coordinate_dump = (
        json.dumps(
            coordinate,
            sort_keys=True,
        ).lower()
    )

    require(
        "h0"
        in
        coordinate_dump,
        (
            "Stage4 prediction-coordinate artifact "
            "does not prove H0 semantics."
        ),
    )

    # ========================================================
    # C. Actual calibrated Gaussian route
    # ========================================================

    gaussian_route = load_json(
        GAUSSIAN_ROUTE_AUDIT
    )

    runtime_api = load_json(
        RUNTIME_API_AUDIT
    )

    route_pattern = re.compile(
        r"("
        r"mean|gaussian|calibrat|"
        r"covariance|scale|"
        r"horizon|prediction|"
        r"scenario|track|actor|id|"
        r"h0|coordinate|"
        r"current|heading"
        r")",
        re.IGNORECASE,
    )

    gaussian_matches = print_matches(
        "C1. ACTUAL STAGE4 GAUSSIAN ROUTE",
        gaussian_route,
        route_pattern,
        limit=180,
    )

    runtime_matches = print_matches(
        "C2. CALIBRATED RUNTIME API",
        runtime_api,
        route_pattern,
        limit=180,
    )

    # ========================================================
    # D. Exact formal Stage5 input route
    # ========================================================

    exact_formal = load_json(
        EXACT_FORMAL_ROUTE
    )

    formal_input = load_json(
        FORMAL_INPUT_ROUTE
    )

    formal_matches = print_matches(
        "D1. EXACT FORMAL ROUTE SEMANTICS",
        exact_formal,
        route_pattern,
        limit=180,
    )

    formal_input_matches = print_matches(
        "D2. FORMAL INPUT ROUTE AUDIT",
        formal_input,
        route_pattern,
        limit=180,
    )

    # ========================================================
    # E. Formal clean-record schema
    # ========================================================

    print()
    print(
        "===== E. FORMAL CLEAN-RECORD SCHEMA ====="
    )

    records = read_jsonl_head(
        FORMAL_RECORDS,
        count=3,
    )

    require(
        records,
        "formal_clean_records.jsonl is empty.",
    )

    record_pattern = re.compile(
        r"("
        r"scenario|track|actor|prediction|id|class|"
        r"mean|trajectory|position|"
        r"covariance|scale|"
        r"horizon|"
        r"heading|yaw|orientation|"
        r"current|anchor|"
        r"h0|frame"
        r")",
        re.IGNORECASE,
    )

    record_match_paths = []

    for index, record in enumerate(
        records
    ):

        print()
        print(
            f"--- formal record {index} ---"
        )

        print(
            "top-level keys =",
            sorted(record),
        )

        matches = flatten_matching(
            record,
            record_pattern,
        )

        record_match_paths.extend(
            path
            for path, _value
            in matches
        )

        for path, value in matches[:140]:

            print(
                path,
                "=",
                value,
            )

        if len(matches) > 140:

            print(
                "...",
                len(matches) - 140,
                "record fields omitted",
            )

    # ========================================================
    # F. Formal Stage4 NPZ shape/schema only
    # ========================================================

    print()
    print(
        "===== F. STAGE4 FORMAL NPZ SCHEMA ====="
    )

    npz_schema = {}

    with np.load(
        STAGE4_FORMAL_NPZ,
        allow_pickle=False,
    ) as archive:

        for key in archive.files:

            value = archive[
                key
            ]

            npz_schema[
                key
            ] = {
                "shape":
                    list(
                        value.shape
                    ),

                "dtype":
                    str(
                        value.dtype
                    ),
            }

            print(
                key,
                "| shape=",
                value.shape,
                "| dtype=",
                value.dtype,
            )

    # ========================================================
    # G. Exact upstream contracts/source
    # ========================================================

    print()
    print(
        "===== G. EXACT STAGE4/STAGE5 SOURCE CONTRACTS ====="
    )

    definitions = []

    requested = (
        (
            STAGE4_CONTRACTS,
            "GaussianHorizonPosterior",
        ),
        (
            STAGE4_CONTRACTS,
            "TrajectoryGaussianPosterior",
        ),
        (
            STAGE4_GAUSSIAN,
            "GaussianTrajectoryOutput",
        ),
        (
            STAGE4_FORMAL_RUNTIME,
            "extract_deterministic_prediction",
        ),
        (
            STAGE5_MONTE_CARLO,
            "deterministic_receiver_angular_posterior",
        ),
    )

    excerpt_parts = []

    for path, name in requested:

        item = exact_definition(
            path,
            name,
        )

        definitions.append({
            "name":
                name,

            "path":
                item["path"],

            "line":
                item["line"],

            "sha256":
                item["sha256"],
        })

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            "DEFINITION:",
            name,
        )

        print(
            "PATH:",
            path,
        )

        print(
            "SHA256:",
            item["sha256"],
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            item["source"]
        )

        excerpt_parts.append(
            (
                "\n"
                "============================================================\n"
                f"{name}\n"
                f"{path}\n"
                f"sha256={item['sha256']}\n"
                "============================================================\n"
                f"{item['source']}\n"
            )
        )

    # ========================================================
    # H. Source contexts that can resolve join/yaw semantics
    # ========================================================

    print()
    print(
        "===== H. JOIN / MEAN / HEADING SOURCE CONTEXTS ====="
    )

    search_sources = (
        STAGE4_FORMAL_RUNTIME,
        STAGE5_MONTE_CARLO,
    )

    tokens = (
        "trajectory_mean_h0_m",
        "current_position_h0_m",
        "current_heading_h0_rad",
        "prediction_id",
        "track_index",
        "track_id",
        "scenario_id",
        "heading",
    )

    source_context_evidence = {}

    for path in search_sources:

        found = contexts(
            path,
            tokens,
            radius=7,
        )

        source_context_evidence[
            str(path)
        ] = found

        print()
        print(
            "---",
            path,
            "---"
        )

        print(
            "contexts =",
            len(found),
        )

        for item in found[:28]:

            print()
            print(
                f"line {item['line']}"
            )

            print(
                item["text"]
            )

        if len(found) > 28:

            print(
                "...",
                len(found) - 28,
                "additional contexts omitted",
            )

    SOURCE_EXCERPT.write_text(
        "".join(
            excerpt_parts
        ),
        encoding="utf-8",
    )

    # ========================================================
    # I. Evidence gates — no guessed binding
    # ========================================================

    print()
    print(
        "===== I. BINDING EVIDENCE GATES ====="
    )

    all_artifact_text = (
        json.dumps(
            coordinate,
            sort_keys=True,
        )
        +
        json.dumps(
            gaussian_route,
            sort_keys=True,
        )
        +
        json.dumps(
            runtime_api,
            sort_keys=True,
        )
        +
        json.dumps(
            exact_formal,
            sort_keys=True,
        )
        +
        json.dumps(
            formal_input,
            sort_keys=True,
        )
        +
        "\n".join(
            json.dumps(
                record,
                sort_keys=True,
            )
            for record in records
        )
    ).lower()

    record_paths_text = (
        "\n".join(
            record_match_paths
        ).lower()
    )

    evidence = {
        "H0_prediction_semantics":
            (
                "h0"
                in
                all_artifact_text
            ),

        "trajectory_mean_present":
            (
                "mean"
                in
                all_artifact_text
                and
                "trajectory"
                in
                all_artifact_text
            ),

        "predictive_covariance_present":
            (
                "covariance"
                in
                all_artifact_text
                or
                "scale_tril"
                in
                all_artifact_text
            ),

        "scenario_identity_present":
            (
                "scenario"
                in
                record_paths_text
            ),

        "actor_prediction_identity_present":
            any(
                token
                in
                record_paths_text
                for token in (
                    "track_id",
                    "track_index",
                    "prediction_id",
                    "actor_id",
                )
            ),

        "four_required_horizons_supported":
            all(
                str(value)
                in
                all_artifact_text
                for value in (
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                )
            ),

        "future_GT_not_required_for_controller":
            (
                "future_ground_truth_as_controller_input"
                in
                json.dumps(
                    load_json(
                        STAGE5_HANDOFF
                    ),
                    sort_keys=True,
                )
                and
                load_json(
                    STAGE5_HANDOFF
                ).get(
                    "shared_posterior",
                    {}
                ).get(
                    "future_ground_truth_as_controller_input"
                )
                is
                False
            ),
    }

    for key, passed in evidence.items():

        print(
            f"{key:42s} =",
            "PASS"
            if passed
            else
            "NOT PROVEN",
        )

    critical = (
        "H0_prediction_semantics",
        "trajectory_mean_present",
        "predictive_covariance_present",
        "scenario_identity_present",
        "actor_prediction_identity_present",
        "four_required_horizons_supported",
        "future_GT_not_required_for_controller",
    )

    incomplete = [
        key
        for key in critical
        if not evidence[key]
    ]

    # ========================================================
    # J. PDF-directed interpretation — still no implementation
    # ========================================================

    print()
    print(
        "===== J. PDF-DIRECTED DETERMINISTIC-ADB INTERPRETATION ====="
    )

    print(
        "motion deterministic GRU = "
        "RETAINED AS SEPARATE STAGE4 MOTION BASELINE"
    )

    print(
        "Stage6 deterministic shared trajectory candidate = "
        "CALIBRATED GAUSSIAN MEAN ONLY"
    )

    print(
        "Stage6 deterministic covariance use = NONE"
    )

    print(
        "reason = isolate uncertainty effect while preserving "
        "shared predictor family"
    )

    print(
        "binding frozen in this audit = NO"
    )

    print(
        "future yaw binding = WAIT FOR EXACT SOURCE EVIDENCE"
    )

    # ========================================================
    # K. Immutability
    # ========================================================

    print()
    print(
        "===== K. IMMUTABILITY ====="
    )

    changed = [
        path
        for path in protected
        if (
            before[str(path)]
            !=
            sha256_file(path)
        )
    ]

    require(
        not changed,
        (
            "Frozen evidence changed during audit: "
            + ", ".join(
                str(path)
                for path in changed
            )
        ),
    )

    print(
        "Stage4 = UNCHANGED"
    )

    print(
        "Stage5 = UNCHANGED"
    )

    print(
        "Block6.2 = UNCHANGED"
    )

    # ========================================================
    # L. Report
    # ========================================================

    status = (
        "PASS_READY_FOR_BLOCK63_BINDING_DECISION"
        if not incomplete
        else
        "BLOCKED_BINDING_EVIDENCE_INCOMPLETE"
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "audit":
            "EXACT_PREDICTION_ROUTE",

        "status":
            status,

        "evidence_gates":
            evidence,

        "incomplete_evidence":
            incomplete,

        "coordinate_semantics_matches":
            coordinate_matches,

        "gaussian_route_matches":
            gaussian_matches,

        "runtime_api_matches":
            runtime_matches,

        "exact_formal_route_matches":
            formal_matches,

        "formal_input_matches":
            formal_input_matches,

        "formal_record_top_level_keys":
            [
                sorted(record)
                for record in records
            ],

        "formal_npz_schema":
            npz_schema,

        "source_definitions":
            definitions,

        "source_context_evidence":
            source_context_evidence,

        "PDF_directed_candidate": {
            "deterministic_predictive_ADB_source":
                (
                    "CALIBRATED_SHARED_GAUSSIAN_MEAN"
                ),

            "covariance_used":
                False,

            "deterministic_GRU_role":
                (
                    "SEPARATE_MOTION_BASELINE"
                ),

            "binding_frozen":
                False,

            "future_yaw":
                "AWAIT_EXACT_SOURCE_BINDING",
        },

        "scope_guards": {
            "future_GT_controller_input":
                False,

            "tracks_to_predict_selector":
                False,

            "centroid_only_future_projection":
                False,

            "probabilistic_mask_implementation":
                False,

            "class_aware_policy_implementation":
                False,

            "communication_codebook_reuse":
                False,

            "formal_grid_freeze":
                False,
        },

        "execution": {
            "model_inference":
                False,

            "training":
                False,

            "dataset_access":
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

    atomic_json(
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
        "BLOCK 6.3 EXACT PREDICTION-ROUTE AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "H0 prediction semantics =",
        (
            "PASS"
            if evidence[
                "H0_prediction_semantics"
            ]
            else
            "NOT PROVEN"
        ),
    )

    print(
        "trajectory mean schema   =",
        (
            "PASS"
            if evidence[
                "trajectory_mean_present"
            ]
            else
            "NOT PROVEN"
        ),
    )

    print(
        "prediction identity      =",
        (
            "PASS"
            if evidence[
                "scenario_identity_present"
            ]
            and
            evidence[
                "actor_prediction_identity_present"
            ]
            else
            "NOT PROVEN"
        ),
    )

    print(
        "horizons                 =",
        (
            "PASS"
            if evidence[
                "four_required_horizons_supported"
            ]
            else
            "NOT PROVEN"
        ),
    )

    print(
        "future GT controller use = NO"
    )

    print(
        "deterministic ADB candidate = "
        "SHARED CALIBRATED GAUSSIAN MEAN"
    )

    print(
        "candidate frozen         = NO"
    )

    print(
        "future yaw policy        = SOURCE EVIDENCE EXTRACTED / "
        "DECISION NOT YET FROZEN"
    )

    print(
        "training/inference       = NO"
    )

    print(
        "formal evaluation/tuning = NO"
    )

    print(
        "upstream modified        = NO"
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
        "BLOCK 6.3 EXACT PREDICTION-ROUTE AUDIT = BLOCKED"
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
        "Stage4 modified       = NO"
    )

    print(
        "Stage5 modified       = NO"
    )

    print(
        "Block6.2 modified     = NO"
    )

    print(
        "training/inference    = NO"
    )

    print(
        "formal evaluation     = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

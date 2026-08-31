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

S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

EXACT_ROUTE_AUDIT = (
    S6
    / "reports/block63_exact_prediction_route_audit.json"
)

INTERFACE_AUDIT = (
    S6
    / "reports/block63_predictor_interface_audit.json"
)

BLOCK62 = (
    S6
    / "reports/block62_closure.json"
)

NEURAL_INPUTS = (
    S4
    / "src/iscai_stage4/data/neural_inputs.py"
)

FORMAL_RUNTIME = (
    S4
    / "src/iscai_stage4/ml/formal_runtime.py"
)

FORMAL_NPZ = (
    S4
    / "artifacts/block48/formal_neural_outputs.npz"
)

FORMAL_CLEAN_RECORDS = (
    S5
    / "artifacts/block58/formal_clean_records.jsonl"
)

DEV_POSTERIOR_JSON = (
    S5
    / "artifacts/block52/development_gaussian_posterior.json"
)

DEV_POSTERIOR_NPZ = (
    S5
    / "artifacts/block52/development_gaussian_posterior.npz"
)

FORMAL_ROUTE_SCRIPT = (
    S5
    / "artifacts/block58/resolve_exact_formal_route_semantics.py"
)

WOMD_GEOMETRY = (
    S6
    / "src/iscai_stage6/adb/womd_geometry.py"
)

REPORT = (
    S6
    / "reports/block63_actor_identity_boundary_audit.json"
)

EXCERPT = (
    S6
    / "artifacts/block63/"
      "block63_actor_identity_exact_source.txt"
)

EXPECTED = {
    "block62":
        (
            "23b299b4dc6a123ce64a6ab1350b3ed9"
            "cb149c8fed176b37e9715d9935704d92"
        ),

    "interface":
        (
            "14fb92076b40c73e378a2ca249b7ebb7"
            "f95b011a0d31ad5062ac75fde01e7e9d"
        ),

    "exact_route":
        (
            "7fa674e6915c4d3ef49252f7def688079"
            "43fd5c0be6ea4f4aa260294c92d23ce"
        ),
}


# ============================================================
# Generic helpers
# ============================================================

def require(condition, message):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:

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


def load_json(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def exact_definitions(
    root: Path,
    name: str,
):

    results = []

    for path in sorted(
        root.rglob("*.py")
    ):

        try:

            source = path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source,
                filename=str(path),
            )

        except BaseException:

            continue

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

            results.append({
                "path":
                    str(path),

                "line":
                    int(node.lineno),

                "source":
                    segment,

                "sha256":
                    sha256(
                        segment.encode(
                            "utf-8"
                        )
                    ).hexdigest(),
            })

    return results


def class_fields(
    root: Path,
    class_name: str,
):

    definitions = exact_definitions(
        root,
        class_name,
    )

    require(
        len(definitions) == 1,
        (
            f"Expected exactly one {class_name}; "
            f"found {len(definitions)}."
        ),
    )

    source = definitions[0][
        "source"
    ]

    tree = ast.parse(source)

    cls = tree.body[0]

    fields = []

    for node in cls.body:

        if isinstance(
            node,
            ast.AnnAssign,
        ) and isinstance(
            node.target,
            ast.Name,
        ):

            fields.append(
                node.target.id
            )

    return (
        definitions[0],
        tuple(fields),
    )


def source_contexts(
    root: Path,
    pattern: re.Pattern,
    *,
    radius: int = 5,
    max_results: int = 80,
):

    results = []

    for path in sorted(
        root.rglob("*.py")
    ):

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

            if len(results) >= max_results:
                return results

    return results


def read_jsonl_head(
    path: Path,
    limit: int = 5,
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

            if len(rows) >= limit:
                break

    return rows


def flatten_paths(
    value,
    prefix="",
):

    rows = []

    if isinstance(value, dict):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.append(
                (
                    path,
                    child,
                )
            )

            if isinstance(
                child,
                (dict, list),
            ):
                rows.extend(
                    flatten_paths(
                        child,
                        path,
                    )
                )

    elif isinstance(value, list):

        if len(value) <= 30:

            for index, child in enumerate(value):

                path = (
                    f"{prefix}[{index}]"
                )

                rows.append(
                    (
                        path,
                        child,
                    )
                )

                if isinstance(
                    child,
                    (dict, list),
                ):
                    rows.extend(
                        flatten_paths(
                            child,
                            path,
                        )
                    )

    return rows


def id_like_npz_schema(path: Path):

    rows = {}

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        for key in archive.files:

            lower = key.lower()

            if not any(
                token in lower
                for token in (
                    "id",
                    "scenario",
                    "track",
                    "actor",
                    "prediction",
                    "sample",
                    "index",
                    "class",
                )
            ):
                continue

            value = archive[key]

            entry = {
                "shape":
                    list(value.shape),

                "dtype":
                    str(value.dtype),
            }

            if (
                value.size <= 20
                and
                value.dtype.kind
                in
                "USib"
            ):

                entry[
                    "values"
                ] = (
                    value.tolist()
                )

            elif (
                value.size > 0
                and
                value.dtype.kind
                in
                "USi"
            ):

                flat = value.reshape(-1)

                entry[
                    "head"
                ] = [
                    item.item()
                    if hasattr(
                        item,
                        "item",
                    )
                    else item
                    for item
                    in flat[:8]
                ]

            rows[
                key
            ] = entry

    return rows


def print_definition(
    title,
    definitions,
    excerpt_parts,
):

    print()
    print(
        f"===== {title} ====="
    )

    if not definitions:

        print(
            "NOT FOUND"
        )

        return

    for item in definitions:

        print()
        print(
            "PATH =",
            item[
                "path"
            ],
        )

        print(
            "LINE =",
            item[
                "line"
            ],
        )

        print(
            "SHA256 =",
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
            f"{title}\n"
            f"{item['path']}:{item['line']}\n"
            f"sha256={item['sha256']}\n"
            "============================================================\n"
            f"{item['source']}\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 ACTOR-IDENTITY BOUNDARY AUDIT"
    )
    print(
        "DIRECT CAUSAL KEY vs ANCHOR-GEOMETRY ASSOCIATION"
    )
    print(
        "============================================================"
    )

    protected = (
        BLOCK62,
        INTERFACE_AUDIT,
        EXACT_ROUTE_AUDIT,
        NEURAL_INPUTS,
        FORMAL_RUNTIME,
        FORMAL_NPZ,
        FORMAL_CLEAN_RECORDS,
        DEV_POSTERIOR_JSON,
        DEV_POSTERIOR_NPZ,
        WOMD_GEOMETRY,
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
    # A. Frozen prerequisites
    # ========================================================

    print()
    print(
        "===== A. FROZEN PREREQUISITES ====="
    )

    require(
        sha256_file(BLOCK62)
        ==
        EXPECTED["block62"],
        "Block6.2 SHA mismatch.",
    )

    require(
        sha256_file(INTERFACE_AUDIT)
        ==
        EXPECTED["interface"],
        "Interface-audit SHA mismatch.",
    )

    require(
        sha256_file(EXACT_ROUTE_AUDIT)
        ==
        EXPECTED["exact_route"],
        "Exact-route-audit SHA mismatch.",
    )

    route_report = load_json(
        EXACT_ROUTE_AUDIT
    )

    require(
        route_report.get(
            "status"
        )
        ==
        "BLOCKED_BINDING_EVIDENCE_INCOMPLETE",
        (
            "Exact-route audit no longer has "
            "the expected single unresolved gate."
        ),
    )

    print(
        "Block6.2             = EXACT PASS"
    )

    print(
        "interface audit      = EXACT PASS"
    )

    print(
        "exact route audit    = EXACT PASS"
    )

    print(
        "unresolved gate      = ACTOR IDENTITY ONLY"
    )

    # ========================================================
    # B. Exact Stage4 prediction-ID source
    # ========================================================

    excerpt_parts = []

    extract_prediction_id = (
        exact_definitions(
            S4 / "src",
            "_extract_prediction_id",
        )
    )

    build_history = (
        exact_definitions(
            S4 / "src",
            "build_causal_track_history",
        )
    )

    build_scene = (
        exact_definitions(
            S4 / "src",
            "build_causal_scene_inputs",
        )
    )

    causal_history_def, causal_history_fields = (
        class_fields(
            S4 / "src",
            "CausalTrackHistory",
        )
    )

    print_definition(
        "B1. _extract_prediction_id",
        extract_prediction_id,
        excerpt_parts,
    )

    print_definition(
        "B2. build_causal_track_history",
        build_history,
        excerpt_parts,
    )

    print_definition(
        "B3. build_causal_scene_inputs",
        build_scene,
        excerpt_parts,
    )

    print_definition(
        "B4. CausalTrackHistory",
        [
            causal_history_def
        ],
        excerpt_parts,
    )

    print(
        "CausalTrackHistory fields =",
        causal_history_fields,
    )

    require(
        "prediction_id"
        in
        causal_history_fields,
        (
            "Stage4 causal history lost "
            "prediction_id."
        ),
    )

    require(
        "latest_position_H0_m"
        in
        causal_history_fields,
        (
            "Stage4 causal history lost "
            "latest_position_H0_m."
        ),
    )

    # ========================================================
    # C. Stage3 internal prediction-ID provenance
    # ========================================================

    print()
    print(
        "===== C. STAGE3 PREDICTION-ID PROVENANCE ====="
    )

    prediction_contexts = source_contexts(
        S3 / "src",
        re.compile(
            r"prediction_id|s3trk-|truth_track_index|truth_id",
            re.IGNORECASE,
        ),
        radius=6,
        max_results=70,
    )

    print(
        "relevant source contexts =",
        len(
            prediction_contexts
        ),
    )

    for index, item in enumerate(
        prediction_contexts[:40],
        start=1,
    ):

        print()
        print(
            f"--- context {index} ---"
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

    if len(prediction_contexts) > 40:

        print(
            "...",
            len(prediction_contexts) - 40,
            "additional contexts stored in report",
        )

    stage3_classes = []

    for path in sorted(
        (S3 / "src").rglob(
            "*.py"
        )
    ):

        try:

            source = path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source,
                filename=str(path),
            )

        except BaseException:

            continue

        for node in tree.body:

            if not isinstance(
                node,
                ast.ClassDef,
            ):
                continue

            fields = []

            for child in node.body:

                if isinstance(
                    child,
                    ast.AnnAssign,
                ) and isinstance(
                    child.target,
                    ast.Name,
                ):

                    fields.append(
                        child.target.id
                    )

            if not any(
                name in fields
                for name in (
                    "prediction_id",
                    "truth_track_index",
                    "track_id",
                    "latest_position_H0_m",
                    "actor_class",
                )
            ):
                continue

            stage3_classes.append({
                "path":
                    str(path),

                "class":
                    node.name,

                "fields":
                    fields,
            })

    print()
    print(
        "Stage3 identity-bearing classes ="
    )

    for item in stage3_classes:

        print(
            item[
                "class"
            ],
            "|",
            item[
                "path"
            ],
            "| fields =",
            item[
                "fields"
            ],
        )

    # ========================================================
    # D. Stage6 causal-box identity fields
    # ========================================================

    print()
    print(
        "===== D. STAGE6 CAUSAL BOX IDENTITY CONTRACT ====="
    )

    box_def, box_fields = (
        class_fields(
            S6 / "src",
            "CausalADBActorBox",
        )
    )

    print(
        box_def[
            "source"
        ]
    )

    print(
        "CausalADBActorBox fields =",
        box_fields,
    )

    require(
        "scenario_id"
        in
        box_fields,
        "Stage6 causal box lost scenario_id.",
    )

    require(
        "track_id"
        in
        box_fields,
        "Stage6 causal box lost track_id.",
    )

    require(
        "track_index"
        in
        box_fields,
        "Stage6 causal box lost track_index.",
    )

    require(
        "stage1_anchor_center_H0_m"
        in
        box_fields,
        (
            "Stage6 causal box lost "
            "anchor-center geometry."
        ),
    )

    direct_actor_field_overlap = sorted(
        (
            set(
                causal_history_fields
            )
            &
            set(
                box_fields
            )
        )
        -
        {
            "scenario_id",
        }
    )

    print(
        "direct actor-field overlap =",
        direct_actor_field_overlap,
    )

    # ========================================================
    # E. Frozen Stage4/5 artifact identity schemas
    # ========================================================

    print()
    print(
        "===== E1. STAGE4 FORMAL NPZ ID-LIKE SCHEMA ====="
    )

    formal_npz_ids = (
        id_like_npz_schema(
            FORMAL_NPZ
        )
    )

    if formal_npz_ids:

        for key, value in (
            formal_npz_ids.items()
        ):

            print(
                key,
                "=",
                value,
            )

    else:

        print(
            "NO ID-LIKE ARRAYS"
        )

    print()
    print(
        "===== E2. STAGE5 DEV POSTERIOR NPZ ID-LIKE SCHEMA ====="
    )

    dev_npz_ids = (
        id_like_npz_schema(
            DEV_POSTERIOR_NPZ
        )
    )

    if dev_npz_ids:

        for key, value in (
            dev_npz_ids.items()
        ):

            print(
                key,
                "=",
                value,
            )

    else:

        print(
            "NO ID-LIKE ARRAYS"
        )

    print()
    print(
        "===== E3. STAGE5 DEV POSTERIOR JSON IDENTITY FIELDS ====="
    )

    dev_json = load_json(
        DEV_POSTERIOR_JSON
    )

    identity_pattern = re.compile(
        r"("
        r"scenario|prediction|actor|track|id|"
        r"current_position|anchor|class"
        r")",
        re.IGNORECASE,
    )

    dev_identity = [
        (
            path,
            value,
        )
        for path, value
        in flatten_paths(
            dev_json
        )
        if identity_pattern.search(
            path
        )
    ]

    for path, value in (
        dev_identity[:120]
    ):

        if isinstance(
            value,
            (dict, list),
        ):

            continue

        print(
            path,
            "=",
            value,
        )

    # ========================================================
    # F. Formal clean-record identity
    # ========================================================

    print()
    print(
        "===== F. FORMAL CLEAN-RECORD IDENTITY ====="
    )

    formal_rows = read_jsonl_head(
        FORMAL_CLEAN_RECORDS,
        limit=5,
    )

    formal_identity_paths = set()

    for index, row in enumerate(
        formal_rows
    ):

        print()
        print(
            f"--- formal record {index} ---"
        )

        print(
            "top-level keys =",
            sorted(row),
        )

        matches = [
            (
                path,
                value,
            )
            for path, value
            in flatten_paths(row)
            if identity_pattern.search(
                path
            )
        ]

        for path, value in matches:

            formal_identity_paths.add(
                path
            )

            if isinstance(
                value,
                (dict, list),
            ):

                continue

            print(
                path,
                "=",
                value,
            )

    # ========================================================
    # G. Formal-route creation code
    # ========================================================

    print()
    print(
        "===== G. FORMAL-ROUTE IDENTITY CREATION CONTEXT ====="
    )

    route_contexts = []

    if FORMAL_ROUTE_SCRIPT.is_file():

        lines = FORMAL_ROUTE_SCRIPT.read_text(
            encoding="utf-8"
        ).splitlines()

        pattern = re.compile(
            r"prediction_id|track_id|track_index|truth|formal_clean_records",
            re.IGNORECASE,
        )

        for index, line in enumerate(lines):

            if not pattern.search(line):
                continue

            lo = max(
                0,
                index - 7,
            )

            hi = min(
                len(lines),
                index + 8,
            )

            context = (
                "\n".join(
                    f"{j + 1:04d}: {lines[j]}"
                    for j in range(
                        lo,
                        hi,
                    )
                )
            )

            route_contexts.append(
                context
            )

        for index, context in enumerate(
            route_contexts[:24],
            start=1,
        ):

            print()
            print(
                f"--- route context {index} ---"
            )

            print(context)

    else:

        print(
            "formal route script not present; "
            "artifact-only audit continues."
        )

    # ========================================================
    # H. Truth/oracle boundary
    # ========================================================

    print()
    print(
        "===== H. CONTROLLER IDENTITY SAFETY BOUNDARY ====="
    )

    combined_stage3_source = "\n".join(
        item[
            "context"
        ]
        for item in prediction_contexts
    ).lower()

    history_source = "\n".join(
        item[
            "source"
        ]
        for item in (
            extract_prediction_id
            +
            build_history
            +
            build_scene
        )
    ).lower()

    exact_route_text = json.dumps(
        route_report,
        sort_keys=True,
    ).lower()

    truth_metadata_exists = any(
        token in (
            combined_stage3_source
            +
            exact_route_text
        )
        for token in (
            "truth_track_index",
            "truth_id",
            "matched_truth",
        )
    )

    predictor_has_internal_id = (
        "prediction_id"
        in
        causal_history_fields
    )

    causal_anchor_position_available = (
        "latest_position_H0_m"
        in
        causal_history_fields
    )

    causal_box_anchor_position_available = (
        "stage1_anchor_center_H0_m"
        in
        box_fields
    )

    # A direct key is proven only if the predictor's actor-level
    # identity itself is demonstrably the same field/value
    # carried by CausalADBActorBox. Merely having truth metadata
    # elsewhere does NOT count.
    direct_same_key_tokens = {
        "track_id",
        "track_index",
    }

    predictor_source_mentions_box_key = any(
        re.search(
            rf"\b{re.escape(token)}\b",
            history_source,
        )
        is not None
        for token in direct_same_key_tokens
    )

    direct_controller_key_proven = (
        predictor_has_internal_id
        and
        predictor_source_mentions_box_key
        and
        bool(
            set(
                direct_actor_field_overlap
            )
            &
            direct_same_key_tokens
        )
    )

    anchor_association_inputs_available = (
        causal_anchor_position_available
        and
        causal_box_anchor_position_available
    )

    print(
        "predictor internal prediction_id =",
        predictor_has_internal_id,
    )

    print(
        "truth/oracle metadata exists      =",
        truth_metadata_exists,
    )

    print(
        "direct controller-safe box key    =",
        (
            "PROVEN"
            if direct_controller_key_proven
            else
            "NOT PROVEN"
        ),
    )

    print(
        "predictor causal anchor H0         =",
        causal_anchor_position_available,
    )

    print(
        "causal box anchor H0               =",
        causal_box_anchor_position_available,
    )

    print(
        "causal anchor association possible =",
        anchor_association_inputs_available,
    )

    print()
    print(
        "truth_track_index controller use = FORBIDDEN"
    )

    print(
        "truth_id controller use          = FORBIDDEN"
    )

    print(
        "future GT association            = FORBIDDEN"
    )

    print(
        "tracks_to_predict identity join   = FORBIDDEN"
    )

    # ========================================================
    # I. Resolve the identity-boundary outcome
    # ========================================================

    print()
    print(
        "===== I. IDENTITY BINDING OUTCOME ====="
    )

    if direct_controller_key_proven:

        outcome = (
            "DIRECT_CAUSAL_IDENTITY_KEY_PROVEN"
        )

        next_binding = (
            "USE_EXISTING_CAUSAL_IDENTITY_KEY"
        )

        status = (
            "PASS_IDENTITY_BOUNDARY_RESOLVED"
        )

    elif anchor_association_inputs_available:

        outcome = (
            "NO_DIRECT_SAFE_KEY_"
            "CAUSAL_ANCHOR_ASSOCIATION_REQUIRED"
        )

        next_binding = (
            "FREEZE_DEVELOPMENT_ONLY_CAUSAL_"
            "ANCHOR_ASSOCIATION_BEFORE_BLOCK63"
        )

        status = (
            "PASS_IDENTITY_BOUNDARY_RESOLVED_"
            "ASSOCIATION_REQUIRED"
        )

    else:

        outcome = (
            "NO_SAFE_IDENTITY_ROUTE_PROVEN"
        )

        next_binding = (
            "BLOCK_IMPLEMENTATION"
        )

        status = (
            "BLOCKED_NO_CONTROLLER_SAFE_"
            "ACTOR_BINDING"
        )

    print(
        "identity outcome =",
        outcome,
    )

    print(
        "next binding     =",
        next_binding,
    )

    # ========================================================
    # J. Yaw decision readiness
    # ========================================================

    print()
    print(
        "===== J. FUTURE-YAW BINDING READINESS ====="
    )

    # The previous exact-route audit already source-extracted
    # the frozen Stage5 rule:
    # trajectory tangent -> atan2(dy, dx), with previous
    # causal/predicted heading carried forward at low speed.
    yaw_ready = (
        "resolve_samplewise_headings"
        in
        json.dumps(
            route_report,
            sort_keys=True,
        )
        or
        (
            "future yaw policy"
            in
            json.dumps(
                route_report,
                sort_keys=True,
            ).lower()
        )
    )

    print(
        "frozen Stage5 tangent-heading source =",
        (
            "AVAILABLE"
            if yaw_ready
            else
            "RECHECK REQUIRED"
        ),
    )

    print(
        "future GT heading required = NO"
    )

    print(
        "deterministic mean treatment = "
        "ONE-SAMPLE TRAJECTORY THROUGH SAME HEADING RULE"
    )

    print(
        "yaw binding frozen here = NO"
    )

    # ========================================================
    # K. Save exact source excerpt
    # ========================================================

    EXCERPT.write_text(
        "".join(
            excerpt_parts
        ),
        encoding="utf-8",
    )

    # ========================================================
    # L. Immutability
    # ========================================================

    print()
    print(
        "===== K. IMMUTABILITY ====="
    )

    changed = [
        str(path)
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
            "Frozen prerequisite changed: "
            + ", ".join(changed)
        ),
    )

    print(
        "Stage3 = UNCHANGED"
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

    print(
        "scientific implementation = UNCHANGED"
    )

    # ========================================================
    # M. Report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "audit":
            "ACTOR_IDENTITY_BOUNDARY",

        "status":
            status,

        "identity": {
            "outcome":
                outcome,

            "next_binding":
                next_binding,

            "predictor_internal_prediction_id":
                predictor_has_internal_id,

            "direct_controller_safe_box_key_proven":
                direct_controller_key_proven,

            "causal_predictor_anchor_H0_available":
                causal_anchor_position_available,

            "causal_box_anchor_H0_available":
                causal_box_anchor_position_available,

            "causal_anchor_association_inputs_available":
                anchor_association_inputs_available,

            "truth_oracle_metadata_exists":
                truth_metadata_exists,
        },

        "forbidden_controller_identity": {
            "truth_track_index":
                True,

            "truth_id":
                True,

            "future_GT_assignment":
                True,

            "tracks_to_predict_join":
                True,

            "perfect_WOMD_actor_ID_if_not_"
            "causally_available":
                True,
        },

        "stage4": {
            "CausalTrackHistory_fields":
                list(
                    causal_history_fields
                ),

            "prediction_id_source":
                extract_prediction_id,

            "build_history_source":
                build_history,

            "build_scene_source":
                build_scene,
        },

        "stage3_identity_classes":
            stage3_classes,

        "stage3_prediction_contexts":
            prediction_contexts,

        "stage6": {
            "CausalADBActorBox_fields":
                list(
                    box_fields
                ),

            "direct_actor_field_overlap":
                direct_actor_field_overlap,
        },

        "artifacts": {
            "Stage4_formal_npz_id_schema":
                formal_npz_ids,

            "Stage5_dev_npz_id_schema":
                dev_npz_ids,

            "Stage5_dev_json_identity":
                [
                    [
                        path,
                        value,
                    ]
                    for path, value
                    in dev_identity
                    if not isinstance(
                        value,
                        (dict, list),
                    )
                ][:300],

            "formal_clean_record_identity_paths":
                sorted(
                    formal_identity_paths
                ),
        },

        "yaw": {
            "frozen_Stage5_rule":
                (
                    "trajectory_tangent_heading_"
                    "with_low_speed_previous_"
                    "heading_carry_forward"
                ),

            "future_GT_heading":
                False,

            "deterministic_mean_application":
                (
                    "mean_as_single_trajectory_sample"
                ),

            "binding_frozen_here":
                False,
        },

        "scope": {
            "model_inference":
                False,

            "dataset_access":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "scientific_implementation_changed":
                False,

            "upstream_modified":
                False,
        },

        "source_excerpt":
            str(EXCERPT),
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
        "BLOCK 6.3 ACTOR-IDENTITY BOUNDARY AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Stage4 internal prediction_id =",
        (
            "PASS"
            if predictor_has_internal_id
            else
            "NOT PROVEN"
        ),
    )

    print(
        "direct predictor→box ID       =",
        (
            "PASS"
            if direct_controller_key_proven
            else
            "NOT PROVEN"
        ),
    )

    print(
        "causal predictor anchor H0    =",
        (
            "PASS"
            if causal_anchor_position_available
            else
            "NOT PROVEN"
        ),
    )

    print(
        "causal box anchor H0          =",
        (
            "PASS"
            if causal_box_anchor_position_available
            else
            "NOT PROVEN"
        ),
    )

    print(
        "truth/oracle identity join    = FORBIDDEN"
    )

    print(
        "identity outcome              =",
        outcome,
    )

    print(
        "future yaw source rule        = "
        "STAGE5 TANGENT + LOW-SPEED CARRY-FORWARD"
    )

    print(
        "future GT heading             = NO"
    )

    print(
        "deterministic Gaussian mean   = "
        "STILL CANDIDATE / NOT YET FROZEN"
    )

    print(
        "predictive ADB implementation = NOT STARTED"
    )

    print(
        "upstream modified             = NO"
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
        "BLOCK 6.3 ACTOR-IDENTITY AUDIT = BLOCKED"
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
        limit=14
    )

    print()
    print(
        "Stage3 modified       = NO"
    )

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
        "model inference       = NO"
    )

    print(
        "dataset access        = NO"
    )

    print(
        "formal evaluation     = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

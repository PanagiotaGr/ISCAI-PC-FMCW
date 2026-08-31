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

BLOCK62 = (
    S6
    / "reports/block62_closure.json"
)

STAGE4_CLOSURE = (
    S4
    / "reports/stage4_final_closure.json"
)

STAGE4_HANDOFF = (
    S4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

STAGE5_CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

STAGE5_HANDOFF = (
    S5
    / "artifacts/block510/"
      "stage5_to_stage6_handoff.json"
)

GEOMETRY = (
    S6
    / "src/iscai_stage6/adb/geometry.py"
)

REPORT = (
    S6
    / "reports/"
      "block63_predictor_interface_audit.json"
)

EXPECTED_BLOCK62_SHA = (
    "23b299b4dc6a123ce64a6ab1350b3ed9"
    "cb149c8fed176b37e9715d9935704d92"
)


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


def load_json(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def flatten(
    value,
    prefix="",
):

    rows = []

    if isinstance(value, dict):

        for key in sorted(value):

            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten(
                    value[key],
                    child,
                )
            )

    elif isinstance(value, list):

        if len(value) <= 12:

            rows.append(
                (
                    prefix,
                    value,
                )
            )

        else:

            rows.append(
                (
                    prefix,
                    f"<list len={len(value)}>",
                )
            )

    else:

        rows.append(
            (
                prefix,
                value,
            )
        )

    return rows


INTEREST = re.compile(
    r"("
    r"gaussian|calibrat|"
    r"deterministic|mean|mu|"
    r"covariance|sigma|cholesky|"
    r"horizon|forecast|prediction|"
    r"posterior|trajectory|"
    r"actor|class|track|scenario|"
    r"heading|yaw|orientation|"
    r"width|height|length|dimension|"
    r"center|position|"
    r"handoff|downstream|"
    r"illumination|adb|communication"
    r")",
    re.IGNORECASE,
)


def print_relevant_json(
    title,
    payload,
):

    print()
    print(
        f"===== {title} ====="
    )

    rows = flatten(payload)

    selected = [
        (
            key,
            value,
        )
        for key, value in rows
        if INTEREST.search(key)
    ]

    print(
        "relevant flattened fields =",
        len(selected),
    )

    for key, value in selected[:220]:

        print(
            key,
            "=",
            value,
        )

    if len(selected) > 220:

        print(
            "...",
            len(selected) - 220,
            "additional relevant fields omitted",
        )

    return selected


def source_inventory(root: Path):

    records = []

    keywords = re.compile(
        r"("
        r"predict|forecast|"
        r"gaussian|calibrat|"
        r"posterior|covariance|"
        r"horizon|trajectory|"
        r"infer|record|handoff"
        r")",
        re.IGNORECASE,
    )

    for path in sorted(
        root.rglob("*.py")
    ):

        try:

            source = path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source
            )

        except BaseException:

            continue

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):

                if not keywords.search(
                    node.name
                ):
                    continue

                if isinstance(
                    node,
                    ast.ClassDef,
                ):

                    header = (
                        f"class {node.name}"
                    )

                else:

                    try:

                        arguments = ast.unparse(
                            node.args
                        )

                    except BaseException:

                        arguments = "?"

                    prefix = (
                        "async def"
                        if isinstance(
                            node,
                            ast.AsyncFunctionDef,
                        )
                        else
                        "def"
                    )

                    header = (
                        f"{prefix} "
                        f"{node.name}"
                        f"({arguments})"
                    )

                records.append({
                    "path":
                        str(path),

                    "line":
                        int(node.lineno),

                    "name":
                        node.name,

                    "header":
                        header,
                })

    return records


def candidate_artifacts(root: Path):

    pattern = re.compile(
        r"("
        r"handoff|formal|"
        r"gaussian|calibrat|"
        r"prediction|posterior|"
        r"record"
        r")",
        re.IGNORECASE,
    )

    results = []

    for base in (
        root / "reports",
        root / "artifacts",
        root / "configs",
    ):

        if not base.is_dir():
            continue

        for path in base.rglob("*"):

            if (
                path.is_file()
                and
                pattern.search(
                    path.name
                )
            ):

                results.append(
                    str(path)
                )

    return sorted(results)


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


def main():

    bootstrap()

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 PRE-IMPLEMENTATION AUDIT"
    )
    print(
        "FROZEN PREDICTOR → FUTURE BOX INTERFACE"
    )
    print(
        "============================================================"
    )

    protected = (
        BLOCK62,
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE5_CLOSURE,
        STAGE5_HANDOFF,
        GEOMETRY,
    )

    for path in protected:

        require(
            path.is_file(),
            f"Missing prerequisite: {path}",
        )

    before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Frozen seals
    # ========================================================

    print()
    print(
        "===== A. FROZEN SEALS ====="
    )

    require(
        sha256_file(BLOCK62)
        ==
        EXPECTED_BLOCK62_SHA,
        (
            "Block6.2 closure SHA mismatch."
        ),
    )

    block62 = load_json(
        BLOCK62
    )

    require(
        block62.get("status")
        ==
        "PASS_COMPLETE",
        (
            "Block6.2 is not PASS_COMPLETE."
        ),
    )

    print(
        "Block6.2 closure SHA = EXACT PASS"
    )

    print(
        "Stage4 closure SHA   =",
        sha256_file(
            STAGE4_CLOSURE
        ),
    )

    print(
        "Stage4 handoff SHA   =",
        sha256_file(
            STAGE4_HANDOFF
        ),
    )

    print(
        "Stage5 closure SHA   =",
        sha256_file(
            STAGE5_CLOSURE
        ),
    )

    print(
        "Stage5 handoff SHA   =",
        sha256_file(
            STAGE5_HANDOFF
        ),
    )

    # ========================================================
    # B. Handoff semantics
    # ========================================================

    s4_closure = load_json(
        STAGE4_CLOSURE
    )

    s4_handoff = load_json(
        STAGE4_HANDOFF
    )

    s5_closure = load_json(
        STAGE5_CLOSURE
    )

    s5_handoff = load_json(
        STAGE5_HANDOFF
    )

    s4_fields = print_relevant_json(
        "B1. STAGE4 CLOSURE RELEVANT FIELDS",
        s4_closure,
    )

    s4_handoff_fields = print_relevant_json(
        "B2. STAGE4→5 HANDOFF RELEVANT FIELDS",
        s4_handoff,
    )

    s5_fields = print_relevant_json(
        "B3. STAGE5 CLOSURE RELEVANT FIELDS",
        s5_closure,
    )

    s5_handoff_fields = print_relevant_json(
        "B4. STAGE5→6 HANDOFF RELEVANT FIELDS",
        s5_handoff,
    )

    # ========================================================
    # C. Source API inventory
    # ========================================================

    print()
    print(
        "===== C. STAGE4 PREDICTOR API INVENTORY ====="
    )

    stage4_api = source_inventory(
        S4 / "src"
    )

    print(
        "candidate APIs =",
        len(stage4_api),
    )

    for item in stage4_api[:160]:

        print(
            f"{item['path']}:{item['line']} | "
            f"{item['header']}"
        )

    if len(stage4_api) > 160:

        print(
            "...",
            len(stage4_api) - 160,
            "additional APIs omitted",
        )

    print()
    print(
        "===== D. STAGE5 POSTERIOR API INVENTORY ====="
    )

    stage5_api = source_inventory(
        S5 / "src"
    )

    print(
        "candidate APIs =",
        len(stage5_api),
    )

    for item in stage5_api[:160]:

        print(
            f"{item['path']}:{item['line']} | "
            f"{item['header']}"
        )

    if len(stage5_api) > 160:

        print(
            "...",
            len(stage5_api) - 160,
            "additional APIs omitted",
        )

    # ========================================================
    # E. Artifact inventory
    # ========================================================

    print()
    print(
        "===== E. STAGE4 CANDIDATE OUTPUT ARTIFACTS ====="
    )

    stage4_artifacts = (
        candidate_artifacts(
            S4
        )
    )

    for value in stage4_artifacts[:120]:

        print(value)

    print()
    print(
        "===== F. STAGE5 CANDIDATE OUTPUT ARTIFACTS ====="
    )

    stage5_artifacts = (
        candidate_artifacts(
            S5
        )
    )

    for value in stage5_artifacts[:120]:

        print(value)

    # ========================================================
    # G. Stage6 full-box geometry contract
    # ========================================================

    print()
    print(
        "===== G. STAGE6 FUTURE-BOX GEOMETRY API ====="
    )

    from iscai_stage6.adb.geometry import (
        Box3D,
        project_box_to_headlamp,
    )

    print(
        "Box3D signature =",
        inspect.signature(
            Box3D
        ),
    )

    print(
        "Box3D fields =",
        list(
            Box3D
            .__dataclass_fields__
        ),
    )

    print(
        "project_box_to_headlamp signature =",
        inspect.signature(
            project_box_to_headlamp
        ),
    )

    print()
    print(
        "project_box_to_headlamp source:"
    )

    print(
        inspect.getsource(
            project_box_to_headlamp
        )
    )

    # ========================================================
    # H. Explicit unresolved binding questions
    # ========================================================

    print()
    print(
        "===== H. BLOCK6.3 BINDING QUESTIONS ====="
    )

    questions = {
        "deterministic_ADB_point_source":
            (
                "UNRESOLVED: deterministic predictor output "
                "vs calibrated-Gaussian mean"
            ),

        "prediction_record_identity":
            (
                "UNRESOLVED: exact scenario/actor key used "
                "to join prediction to causal box"
            ),

        "future_center_dimensionality":
            (
                "UNRESOLVED: exact predicted center fields "
                "and coordinate frame"
            ),

        "future_yaw_policy":
            (
                "UNRESOLVED: predicted heading/yaw if present; "
                "otherwise documented causal-or-derived fallback"
            ),

        "box_dimensions_policy":
            (
                "EXPECTED: causal current l/w/h held explicit "
                "unless frozen predictor output supplies dimensions"
            ),

        "deterministic_covariance_use":
            (
                "MUST BE NONE for deterministic masking; "
                "covariance retained for later uncertainty-aware block"
            ),

        "future_GT_controller_use":
            False,

        "tracks_to_predict_controller_use":
            False,

        "centroid_only_predictive_geometry":
            False,

        "communication_codebook_reuse":
            False,

        "formal_grid_freeze":
            "BLOCK6.8_PRE_FORMAL",
    }

    for key, value in questions.items():

        print(
            key,
            "=",
            value,
        )

    # ========================================================
    # I. Immutability
    # ========================================================

    print()
    print(
        "===== I. IMMUTABILITY ====="
    )

    changed = []

    for path in protected:

        if (
            before[str(path)]
            !=
            sha256_file(path)
        ):
            changed.append(
                str(path)
            )

    require(
        not changed,
        (
            "Frozen prerequisite changed: "
            + ", ".join(changed)
        ),
    )

    print(
        "Stage4          = UNCHANGED"
    )

    print(
        "Stage5          = UNCHANGED"
    )

    print(
        "Block6.2        = UNCHANGED"
    )

    print(
        "scientific code = UNCHANGED"
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

        "phase":
            "PRE_IMPLEMENTATION_INTERFACE_AUDIT",

        "status":
            "PASS_INTERFACE_AUDIT_COMPLETE",

        "frozen_inputs": {
            "block62_closure_sha256":
                sha256_file(BLOCK62),

            "stage4_closure_sha256":
                sha256_file(STAGE4_CLOSURE),

            "stage4_handoff_sha256":
                sha256_file(STAGE4_HANDOFF),

            "stage5_closure_sha256":
                sha256_file(STAGE5_CLOSURE),

            "stage5_handoff_sha256":
                sha256_file(STAGE5_HANDOFF),
        },

        "stage4_relevant_fields":
            [
                [key, value]
                for key, value
                in s4_fields
            ],

        "stage4_handoff_relevant_fields":
            [
                [key, value]
                for key, value
                in s4_handoff_fields
            ],

        "stage5_relevant_fields":
            [
                [key, value]
                for key, value
                in s5_fields
            ],

        "stage5_handoff_relevant_fields":
            [
                [key, value]
                for key, value
                in s5_handoff_fields
            ],

        "stage4_api_inventory":
            stage4_api,

        "stage5_api_inventory":
            stage5_api,

        "stage4_candidate_artifacts":
            stage4_artifacts,

        "stage5_candidate_artifacts":
            stage5_artifacts,

        "binding_questions":
            questions,

        "PDF_scope": {
            "future_box_projection_required":
                True,

            "centroid_only_forbidden":
                True,

            "box_corners_required":
                True,

            "actor_width_height_required":
                True,

            "probabilistic_masks_this_block":
                False,

            "class_aware_policy_this_block":
                False,

            "raised_cosine_kernel_reused":
                True,

            "communication_illumination_spaces_separate":
                True,
        },

        "execution": {
            "training":
                False,

            "model_inference":
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
    }

    atomic_json(
        REPORT,
        report,
    )

    print()
    print(
        "audit report =",
        REPORT,
    )

    print(
        "audit SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 PRE-IMPLEMENTATION AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block6.2 prerequisite      = EXACT PASS"
    )

    print(
        "Stage4 handoff             = INSPECTED"
    )

    print(
        "Stage5 handoff             = INSPECTED"
    )

    print(
        "predictor API inventory    = CAPTURED"
    )

    print(
        "posterior API inventory    = CAPTURED"
    )

    print(
        "Stage6 full-box API        = CAPTURED"
    )

    print(
        "deterministic source choice= NOT GUESSED"
    )

    print(
        "future yaw policy          = NOT GUESSED"
    )

    print(
        "future GT controller use   = NO"
    )

    print(
        "centroid-only predictive   = FORBIDDEN"
    )

    print(
        "probabilistic masks        = NOT STARTED"
    )

    print(
        "class-aware policies       = NOT STARTED"
    )

    print(
        "formal grid freeze         = NO"
    )

    print(
        "upstream modified          = NO"
    )

    print(
        "STATUS = PASS_INTERFACE_AUDIT_COMPLETE"
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
        "BLOCK 6.3 PRE-IMPLEMENTATION AUDIT = BLOCKED"
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
        "dataset access        = NO"
    )

    print(
        "formal evaluation     = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

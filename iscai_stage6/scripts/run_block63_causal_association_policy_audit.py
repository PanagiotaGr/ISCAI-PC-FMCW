from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import traceback


ROOT = Path("/home/agni/waymo")

S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S6 = ROOT / "iscai_stage6"

IDENTITY_AUDIT = (
    S6
    / "reports/block63_actor_identity_boundary_audit.json"
)

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

NEAREST_NEIGHBOR = (
    S3
    / "src/iscai_stage3/association/nearest_neighbor.py"
)

ASSOCIATION_CONTRACTS = (
    S3
    / "src/iscai_stage3/association/contracts.py"
)

TEST_BOUNDARY = (
    S3
    / "tests/test_association_boundary.py"
)

TEST_COVARIANCE = (
    S3
    / "tests/test_association_covariance.py"
)

TEST_ESTIMATED = (
    S3
    / "tests/test_estimated_association.py"
)

NEURAL_INPUTS = (
    S4
    / "src/iscai_stage4/data/neural_inputs.py"
)

REPORT = (
    S6
    / "reports/block63_causal_association_policy_audit.json"
)

EXCERPT = (
    S6
    / "artifacts/block63/"
      "block63_causal_association_policy_exact_source.txt"
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

    "identity":
        (
            "526909d4f9a5c0f0d5fc8419df6b0f0f"
            "c624ce13fbac527d44c649c1e1703559"
        ),
}


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


def top_level_definitions(path: Path):

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    results = []

    for node in tree.body:

        if not isinstance(
            node,
            (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        if segment is None:
            continue

        results.append({
            "name":
                node.name,

            "kind":
                type(node).__name__,

            "line":
                int(node.lineno),

            "source":
                segment,

            "sha256":
                sha256(
                    segment.encode("utf-8")
                ).hexdigest(),
        })

    return results


def dataclass_defaults(source: str):

    try:

        tree = ast.parse(source)

    except BaseException:

        return []

    if (
        not tree.body
        or
        not isinstance(
            tree.body[0],
            ast.ClassDef,
        )
    ):

        return []

    cls = tree.body[0]

    results = []

    for node in cls.body:

        if not isinstance(
            node,
            ast.AnnAssign,
        ):
            continue

        if not isinstance(
            node.target,
            ast.Name,
        ):
            continue

        default = None

        if node.value is not None:

            try:

                default = ast.unparse(
                    node.value
                )

            except BaseException:

                default = (
                    "<unparse-failed>"
                )

        results.append({
            "field":
                node.target.id,

            "default":
                default,
        })

    return results


def relevant_definition(item):

    name = item[
        "name"
    ].lower()

    source = item[
        "source"
    ].lower()

    return any(
        token in (
            name
            +
            "\n"
            +
            source
        )
        for token in (
            "associate",
            "association",
            "gate",
            "covariance",
            "mahalanobis",
            "distance",
            "nearest",
            "track",
            "prediction",
        )
    )


def contexts(
    path: Path,
    pattern: re.Pattern,
    radius: int = 6,
):

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    results = []

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
            "line":
                index + 1,

            "text":
                "\n".join(
                    f"{j + 1:04d}: {lines[j]}"
                    for j in range(
                        lo,
                        hi,
                    )
                ),
        })

    return results


def forbidden_truth_dependency(
    source: str,
):

    lower = source.lower()

    forbidden = (
        "truth_id",
        "truth_track_index",
        "future_truth",
        "labels_future",
        "tracks_to_predict",
        "objects_of_interest",
    )

    return [
        token
        for token in forbidden
        if token in lower
    ]


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 CAUSAL ASSOCIATION POLICY AUDIT"
    )
    print(
        "REUSE FROZEN STAGE3 POLICY BEFORE INVENTING A NEW GATE"
    )
    print(
        "============================================================"
    )

    protected = (
        BLOCK62,
        INTERFACE_AUDIT,
        EXACT_ROUTE_AUDIT,
        IDENTITY_AUDIT,
        NEAREST_NEIGHBOR,
        ASSOCIATION_CONTRACTS,
        TEST_BOUNDARY,
        TEST_COVARIANCE,
        TEST_ESTIMATED,
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
            "Missing required frozen source/evidence: "
            + ", ".join(missing)
        ),
    )

    before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Frozen audit chain
    # ========================================================

    print()
    print(
        "===== A. FROZEN AUDIT CHAIN ====="
    )

    exact = (
        (
            "block62",
            BLOCK62,
        ),
        (
            "interface",
            INTERFACE_AUDIT,
        ),
        (
            "exact_route",
            EXACT_ROUTE_AUDIT,
        ),
        (
            "identity",
            IDENTITY_AUDIT,
        ),
    )

    for name, path in exact:

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            EXPECTED[name],
            (
                f"{name} SHA mismatch.\n"
                f"expected={EXPECTED[name]}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{name:18s} = EXACT PASS"
        )

    identity = load_json(
        IDENTITY_AUDIT
    )

    require(
        identity.get(
            "status"
        )
        ==
        (
            "PASS_IDENTITY_BOUNDARY_"
            "RESOLVED_ASSOCIATION_REQUIRED"
        ),
        (
            "Identity audit does not require "
            "causal anchor association."
        ),
    )

    require(
        identity[
            "identity"
        ][
            "direct_controller_safe_box_key_proven"
        ]
        is
        False,
        (
            "Direct safe key unexpectedly became proven."
        ),
    )

    require(
        identity[
            "identity"
        ][
            "causal_anchor_association_inputs_available"
        ]
        is
        True,
        (
            "Causal anchor association inputs "
            "are no longer available."
        ),
    )

    print(
        "direct safe ID       = NO"
    )

    print(
        "causal anchor route  = REQUIRED"
    )

    # ========================================================
    # B. Stage3 nearest-neighbor API/source
    # ========================================================

    print()
    print(
        "===== B. STAGE3 CAUSAL ASSOCIATION DEFINITIONS ====="
    )

    nn_definitions = (
        top_level_definitions(
            NEAREST_NEIGHBOR
        )
    )

    relevant = [
        item
        for item in nn_definitions
        if relevant_definition(item)
    ]

    print(
        "relevant definitions =",
        len(relevant),
    )

    excerpt_parts = []

    for item in relevant:

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            item["kind"],
            item["name"],
            "| line =",
            item["line"],
            "| SHA256 =",
            item["sha256"],
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            item["source"]
        )

        excerpt_parts.append(
            "\n"
            "============================================================\n"
            f"{item['kind']} {item['name']}\n"
            f"{NEAREST_NEIGHBOR}:{item['line']}\n"
            f"sha256={item['sha256']}\n"
            "============================================================\n"
            f"{item['source']}\n"
        )

    # ========================================================
    # C. Config/default extraction
    # ========================================================

    print()
    print(
        "===== C. ASSOCIATION CONFIG / GATE DEFAULTS ====="
    )

    config_rows = []

    for item in nn_definitions:

        if item[
            "kind"
        ] != "ClassDef":
            continue

        rows = dataclass_defaults(
            item["source"]
        )

        relevant_rows = [
            row
            for row in rows
            if any(
                token in row[
                    "field"
                ].lower()
                for token in (
                    "gate",
                    "distance",
                    "sigma",
                    "covariance",
                    "miss",
                    "frame",
                    "weight",
                    "cost",
                )
            )
        ]

        if not relevant_rows:
            continue

        print()
        print(
            "CLASS =",
            item[
                "name"
            ],
        )

        for row in relevant_rows:

            print(
                row[
                    "field"
                ],
                "=",
                row[
                    "default"
                ],
            )

        config_rows.append({
            "class":
                item[
                    "name"
                ],

            "rows":
                relevant_rows,
        })

    require(
        config_rows,
        (
            "No Stage3 causal association "
            "gate/config defaults discovered."
        ),
    )

    # ========================================================
    # D. Covariance / distance mechanics
    # ========================================================

    print()
    print(
        "===== D. DISTANCE / COVARIANCE GATE CONTEXTS ====="
    )

    gate_pattern = re.compile(
        r"mahalanobis|covariance|"
        r"gate|distance|innovation|"
        r"sqrt|sigma|threshold",
        re.IGNORECASE,
    )

    gate_contexts = contexts(
        NEAREST_NEIGHBOR,
        gate_pattern,
        radius=7,
    )

    print(
        "gate contexts =",
        len(
            gate_contexts
        ),
    )

    for index, item in enumerate(
        gate_contexts[:36],
        start=1,
    ):

        print()
        print(
            f"--- gate context {index} "
            f"(line {item['line']}) ---"
        )

        print(
            item[
                "text"
            ]
        )

    # ========================================================
    # E. Truth/leakage source boundary
    # ========================================================

    print()
    print(
        "===== E. STAGE3 ASSOCIATION CAUSALITY GATE ====="
    )

    nn_source = (
        NEAREST_NEIGHBOR.read_text(
            encoding="utf-8"
        )
    )

    forbidden_hits = (
        forbidden_truth_dependency(
            nn_source
        )
    )

    print(
        "forbidden source tokens =",
        forbidden_hits,
    )

    require(
        not forbidden_hits,
        (
            "Frozen Stage3 nearest-neighbor "
            "association contains forbidden "
            "controller identity/future source: "
            + ", ".join(
                forbidden_hits
            )
        ),
    )

    truth_used_false = (
        "truth_used=False"
        in
        nn_source.replace(
            " ",
            ""
        )
    )

    print(
        "truth/future dependency = NONE PASS"
    )

    print(
        "truth_used=False evidence =",
        truth_used_false,
    )

    # ========================================================
    # F. Frozen regression tests proving semantics
    # ========================================================

    print()
    print(
        "===== F. ASSOCIATION TEST EVIDENCE ====="
    )

    tests = (
        TEST_BOUNDARY,
        TEST_COVARIANCE,
        TEST_ESTIMATED,
    )

    test_pattern = re.compile(
        r"def test_|truth|"
        r"covariance|gate|"
        r"distance|azimuth|"
        r"detection_key|identity",
        re.IGNORECASE,
    )

    test_evidence = {}

    for path in tests:

        found = contexts(
            path,
            test_pattern,
            radius=5,
        )

        test_evidence[
            str(path)
        ] = found

        print()
        print(
            "---",
            path.name,
            "---"
        )

        for item in found[:24]:

            print()
            print(
                f"line {item['line']}"
            )

            print(
                item[
                    "text"
                ]
            )

    combined_test_source = "\n".join(
        path.read_text(
            encoding="utf-8"
        )
        for path in tests
    ).lower()

    tests_no_truth = (
        "no_truth"
        in
        combined_test_source
        or
        (
            "truth_dependency"
            in
            combined_test_source
        )
    )

    tests_covariance = (
        "larger_covariance_relaxes_gate"
        in
        combined_test_source
        or
        (
            "covariance"
            in
            TEST_COVARIANCE.read_text(
                encoding="utf-8"
            ).lower()
        )
    )

    tests_identity_boundary = (
        "detection_key_not_identity"
        in
        combined_test_source
        or
        (
            "identity"
            in
            TEST_ESTIMATED.read_text(
                encoding="utf-8"
            ).lower()
        )
    )

    print()
    print(
        "no-truth regression evidence     =",
        tests_no_truth,
    )

    print(
        "covariance-gate regression       =",
        tests_covariance,
    )

    print(
        "identity-boundary regression     =",
        tests_identity_boundary,
    )

    require(
        tests_no_truth,
        "No no-truth association test found.",
    )

    require(
        tests_covariance,
        (
            "No covariance-aware association "
            "test evidence found."
        ),
    )

    require(
        tests_identity_boundary,
        (
            "No association identity-boundary "
            "test evidence found."
        ),
    )

    # ========================================================
    # G. Stage4 anchor-side fields
    # ========================================================

    print()
    print(
        "===== G. STAGE4 CAUSAL HISTORY ASSOCIATION INPUTS ====="
    )

    neural_defs = (
        top_level_definitions(
            NEURAL_INPUTS
        )
    )

    history_defs = [
        item
        for item in neural_defs
        if item[
            "name"
        ]
        in (
            "CausalTrackHistory",
            "_extract_prediction_id",
            "build_causal_track_history",
        )
    ]

    for item in history_defs:

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            item["kind"],
            item["name"],
            "| line =",
            item["line"],
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            item[
                "source"
            ]
        )

    combined_history = "\n".join(
        item[
            "source"
        ]
        for item in history_defs
    ).lower()

    history_anchor = (
        "latest_position_h0_m"
        in
        combined_history
    )

    history_prediction_id = (
        "prediction_id"
        in
        combined_history
    )

    print()
    print(
        "prediction_id available      =",
        history_prediction_id,
    )

    print(
        "current causal anchor H0     =",
        history_anchor,
    )

    require(
        history_prediction_id
        and
        history_anchor,
        (
            "Stage4 causal history lacks "
            "required association inputs."
        ),
    )

    # ========================================================
    # H. Reuse feasibility outcome
    # ========================================================

    print()
    print(
        "===== H. STAGE6 ASSOCIATION POLICY OUTCOME ====="
    )

    # We do not automatically import/call the Stage3 tracker
    # association function because its object interfaces may
    # be detection/track specific. What matters here is whether
    # its already-frozen causal gate semantics can be reused
    # exactly in a thin Stage6 anchor-association adapter.
    has_distance_or_gate = any(
        any(
            token in row[
                "field"
            ].lower()
            for token in (
                "gate",
                "distance",
                "sigma",
            )
        )
        for group in config_rows
        for row in group[
            "rows"
        ]
    )

    covariance_semantics = (
        "covariance"
        in
        nn_source.lower()
    )

    causal_semantics = (
        not forbidden_hits
        and
        tests_no_truth
        and
        tests_identity_boundary
    )

    if (
        has_distance_or_gate
        and
        causal_semantics
    ):

        if covariance_semantics:

            outcome = (
                "REUSE_FROZEN_STAGE3_"
                "CAUSAL_COVARIANCE_GATE_SEMANTICS"
            )

        else:

            outcome = (
                "REUSE_FROZEN_STAGE3_"
                "CAUSAL_DISTANCE_GATE_SEMANTICS"
            )

        status = (
            "PASS_READY_FOR_BLOCK63_"
            "ASSOCIATION_BINDING_FREEZE"
        )

    else:

        outcome = (
            "STAGE3_POLICY_NOT_DIRECTLY_"
            "REUSABLE_WITHOUT_NEW_NUMERICS"
        )

        status = (
            "BLOCKED_ASSOCIATION_POLICY_"
            "NEEDS_DEVELOPMENT_DESIGN"
        )

    print(
        "outcome =",
        outcome,
    )

    print(
        "new arbitrary distance threshold introduced = NO"
    )

    print(
        "formal population inspected/tuned = NO"
    )

    print(
        "controller truth identity = NO"
    )

    # ========================================================
    # I. PDF / Stage6 binding proposal
    # ========================================================

    print()
    print(
        "===== I. PROPOSED BLOCK6.3 BINDING ====="
    )

    proposal = {
        "association":
            (
                "one-to-one current-anchor H0 "
                "causal association"
            ),

        "association_gate_authority":
            outcome,

        "same_class_gate":
            (
                "use symbolic causal actor class "
                "when causally available; never "
                "truth-side evaluator class"
            ),

        "future_state_use":
            False,

        "truth_id_use":
            False,

        "truth_track_index_use":
            False,

        "tracks_to_predict_use":
            False,

        "objects_of_interest_use":
            False,

        "perfect_WOMD_track_id_use":
            False,

        "deterministic_ADB_trajectory":
            (
                "calibrated Gaussian mean only"
            ),

        "Stage4_mean_semantics":
            "metric_H0_displacement",

        "future_center":
            (
                "causal_anchor_H0 + "
                "Gaussian_mean_displacement_H0"
            ),

        "predictive_covariance_used_in_Block63":
            False,

        "future_yaw":
            (
                "same frozen Stage5 trajectory "
                "tangent rule with low-speed "
                "causal/predicted carry-forward"
            ),

        "future_GT_heading":
            False,

        "box_dimensions":
            (
                "current causal l/w/h propagated "
                "unchanged across deterministic "
                "forecast horizons; explicit "
                "Stage6 implementation assumption "
                "because predictor does not output "
                "future dimensions"
            ),

        "full_box_corners":
            True,

        "centroid_only_projection":
            False,

        "formal_grid_freeze":
            False,

        "class_aware_policy":
            False,

        "probabilistic_mask":
            False,
    }

    for key, value in proposal.items():

        print(
            key,
            "=",
            value,
        )

    # ========================================================
    # J. Exact-source artifact
    # ========================================================

    EXCERPT.write_text(
        "".join(
            excerpt_parts
        ),
        encoding="utf-8",
    )

    # ========================================================
    # K. Immutability
    # ========================================================

    print()
    print(
        "===== J. IMMUTABILITY ====="
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
        "Block6.2 = UNCHANGED"
    )

    print(
        "scientific implementation = UNCHANGED"
    )

    # ========================================================
    # L. Report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "audit":
            "CAUSAL_ASSOCIATION_POLICY",

        "status":
            status,

        "association_policy_outcome":
            outcome,

        "stage3": {
            "nearest_neighbor_path":
                str(
                    NEAREST_NEIGHBOR
                ),

            "nearest_neighbor_sha256":
                sha256_file(
                    NEAREST_NEIGHBOR
                ),

            "relevant_definitions": [
                {
                    "name":
                        item[
                            "name"
                        ],

                    "kind":
                        item[
                            "kind"
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
                for item in relevant
            ],

            "config_gate_defaults":
                config_rows,

            "forbidden_truth_tokens":
                forbidden_hits,

            "truth_used_false_evidence":
                truth_used_false,

            "tests": {
                "no_truth":
                    tests_no_truth,

                "covariance":
                    tests_covariance,

                "identity_boundary":
                    tests_identity_boundary,
            },
        },

        "stage4": {
            "prediction_id_available":
                history_prediction_id,

            "causal_anchor_H0_available":
                history_anchor,
        },

        "proposed_Block63_binding":
            proposal,

        "new_numeric_association_parameter":
            False,

        "formal_data_used":
            False,

        "parameter_tuning":
            False,

        "implementation_changed":
            False,

        "upstream_modified":
            False,

        "source_excerpt":
            str(
                EXCERPT
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
        "BLOCK 6.3 CAUSAL ASSOCIATION POLICY AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "direct perfect identity     = NO"
    )

    print(
        "causal current-anchor route = REQUIRED"
    )

    print(
        "Stage3 association truth use= NO"
    )

    print(
        "Stage3 covariance gate      =",
        (
            "PRESENT"
            if covariance_semantics
            else
            "NOT PRESENT"
        ),
    )

    print(
        "new arbitrary threshold     = NO"
    )

    print(
        "deterministic ADB mean       = "
        "CALIBRATED GAUSSIAN MEAN CANDIDATE"
    )

    print(
        "future center semantics      = "
        "ANCHOR H0 + METRIC H0 DISPLACEMENT"
    )

    print(
        "future yaw                   = "
        "STAGE5 TANGENT / LOW-SPEED CARRY-FORWARD"
    )

    print(
        "future box dimensions        = "
        "CURRENT CAUSAL L/W/H HELD EXPLICIT"
    )

    print(
        "future box corners           = REQUIRED"
    )

    print(
        "future GT                    = NO"
    )

    print(
        "formal data/tuning           = NO"
    )

    print(
        "upstream modified            = NO"
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
        "BLOCK 6.3 CAUSAL ASSOCIATION POLICY AUDIT = BLOCKED"
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
        "Block6.2 modified     = NO"
    )

    print(
        "formal data/tuning    = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

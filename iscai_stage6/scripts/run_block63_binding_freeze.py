from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

BLOCK62 = (
    S6
    / "reports/block62_closure.json"
)

INTERFACE_AUDIT = (
    S6
    / "reports/block63_predictor_interface_audit.json"
)

EXACT_ROUTE_AUDIT = (
    S6
    / "reports/block63_exact_prediction_route_audit.json"
)

IDENTITY_AUDIT = (
    S6
    / "reports/block63_actor_identity_boundary_audit.json"
)

ASSOCIATION_AUDIT = (
    S6
    / "reports/block63_causal_association_policy_audit.json"
)

BINDING = (
    S6
    / "configs/block63_deterministic_predictive_binding.json"
)

REPORT = (
    S6
    / "reports/block63_binding_freeze.json"
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

    "association":
        (
            "ecc25a7d3280354c1365bf6021b823006"
            "4461571da262eefb80dd35048fa741a"
        ),
}


def require(
    condition,
    message,
):

    if not bool(condition):
        raise RuntimeError(
            message
        )


def sha256_file(
    path: Path,
) -> str:

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
    ).encode(
        "utf-8"
    )


def write_exact_json(
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
                "Existing frozen artifact differs: "
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


def load_json(
    path: Path,
):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 BINDING FREEZE"
    )
    print(
        "DETERMINISTIC SHARED-MEAN FUTURE FULL-BOX POLICY"
    )
    print(
        "============================================================"
    )

    protected = (
        BLOCK62,
        INTERFACE_AUDIT,
        EXACT_ROUTE_AUDIT,
        IDENTITY_AUDIT,
        ASSOCIATION_AUDIT,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing prerequisite(s): "
            + ", ".join(
                missing
            )
        ),
    )

    before = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    # ========================================================
    # A. Exact audit-chain seal
    # ========================================================

    print()
    print(
        "===== A. EXACT AUDIT-CHAIN SEAL ====="
    )

    checks = (
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
        (
            "association",
            ASSOCIATION_AUDIT,
        ),
    )

    for name, path in checks:

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            EXPECTED[
                name
            ],
            (
                f"{name} SHA mismatch.\n"
                f"expected={EXPECTED[name]}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{name:18s} = EXACT PASS"
        )

    block62 = load_json(
        BLOCK62
    )

    interface = load_json(
        INTERFACE_AUDIT
    )

    exact_route = load_json(
        EXACT_ROUTE_AUDIT
    )

    identity = load_json(
        IDENTITY_AUDIT
    )

    association = load_json(
        ASSOCIATION_AUDIT
    )

    require(
        block62.get(
            "status"
        )
        ==
        "PASS_COMPLETE",
        (
            "Block6.2 is not frozen complete."
        ),
    )

    require(
        interface.get(
            "status"
        )
        ==
        "PASS_INTERFACE_AUDIT_COMPLETE",
        (
            "Block6.3 interface audit changed."
        ),
    )

    require(
        exact_route.get(
            "status"
        )
        ==
        "BLOCKED_BINDING_EVIDENCE_INCOMPLETE",
        (
            "Unexpected exact-route status."
        ),
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
            "Identity-boundary result changed."
        ),
    )

    require(
        association.get(
            "status"
        )
        ==
        (
            "BLOCKED_ASSOCIATION_POLICY_"
            "NEEDS_DEVELOPMENT_DESIGN"
        ),
        (
            "Association-policy audit "
            "has unexpected status."
        ),
    )

    print(
        "Block6.2 prerequisite        = PASS"
    )

    print(
        "direct safe actor ID         = NOT AVAILABLE"
    )

    print(
        "new Stage6 association design= REQUIRED"
    )

    # ========================================================
    # B. Freeze deterministic association design
    # ========================================================

    print()
    print(
        "===== B. PARAMETER-FREE CAUSAL ASSOCIATION ====="
    )

    association_policy = {
        "name":
            (
                "reciprocal_unique_nearest_"
                "current_anchor_H0_3D"
            ),

        "scientific_role":
            (
                "controller-safe association between "
                "Stage4 causal predictor tracks and "
                "current causal Stage6 ADB boxes"
            ),

        "predictor_anchor":
            (
                "CausalTrackHistory."
                "latest_position_H0_m"
            ),

        "box_anchor":
            (
                "CausalADBActorBox."
                "stage1_anchor_center_H0_m"
            ),

        "coordinate_frame":
            "H0",

        "distance_metric":
            "Euclidean_3D_m",

        "algorithm": {
            "step_1":
                (
                    "compute all finite predictor-anchor "
                    "to box-anchor Euclidean distances"
                ),

            "step_2":
                (
                    "for every predictor require one "
                    "unique nearest box"
                ),

            "step_3":
                (
                    "for every box require one "
                    "unique nearest predictor"
                ),

            "step_4":
                (
                    "accept pair only when the two "
                    "unique nearest relations are reciprocal"
                ),

            "forced_assignment":
                False,

            "Hungarian_assignment":
                False,

            "hard_distance_threshold_m":
                None,

            "covariance_gate":
                False,

            "class_gate_before_association":
                False,

            "prediction_id_equality":
                False,

            "WOMD_track_id_equality":
                False,

            "exact_distance_tie":
                "UNMATCHED",

            "free_numeric_hyperparameters":
                0,
        },

        "unmatched_policy": {
            "causal_ADB_box":
                (
                    "ORIGINAL_REACTIVE_ADB_FALLBACK"
                ),

            "predictor_track":
                (
                    "DO_NOT_APPLY_TO_ADB_BOX"
                ),

            "silent_drop_of_ADB_actor":
                False,
        },

        "forbidden_controller_inputs": {
            "truth_id":
                True,

            "truth_track_index":
                True,

            "future_GT":
                True,

            "future_validity":
                True,

            "tracks_to_predict":
                True,

            "objects_of_interest":
                True,

            "perfect_WOMD_actor_identity":
                True,
        },

        "diagnostics_required": {
            "matched_pair_count":
                True,

            "unmatched_box_count":
                True,

            "unmatched_predictor_count":
                True,

            "match_fraction":
                True,

            "anchor_distance_m":
                True,

            "nearest_second_nearest_distances":
                True,

            "reactive_fallback_fraction":
                True,

            "evaluator_truth_association_accuracy":
                (
                    "DEVELOPMENT_AND_FORMAL_REPORTING_ONLY_"
                    "AFTER_CONTROLLER_DECISION"
                ),
        },
    }

    require(
        association_policy[
            "algorithm"
        ][
            "free_numeric_hyperparameters"
        ]
        ==
        0,
        (
            "Association design unexpectedly "
            "introduced a tunable numeric parameter."
        ),
    )

    require(
        association_policy[
            "algorithm"
        ][
            "hard_distance_threshold_m"
        ]
        is
        None,
        (
            "A hard distance gate must not be "
            "invented in this freeze."
        ),
    )

    require(
        association_policy[
            "unmatched_policy"
        ][
            "silent_drop_of_ADB_actor"
        ]
        is
        False,
        (
            "Unmatched ADB actors cannot "
            "be silently dropped."
        ),
    )

    print(
        "association = RECIPROCAL UNIQUE NEAREST 3D H0"
    )

    print(
        "hard distance threshold = NONE"
    )

    print(
        "covariance association gate = NO"
    )

    print(
        "pre-association class gate = NO"
    )

    print(
        "forced assignment = NO"
    )

    print(
        "exact ties = UNMATCHED"
    )

    print(
        "unmatched box = ORIGINAL REACTIVE FALLBACK"
    )

    print(
        "new numeric hyperparameter = NONE"
    )

    # ========================================================
    # C. Freeze deterministic predictive trajectory
    # ========================================================

    print()
    print(
        "===== C. DETERMINISTIC SHARED-TRAJECTORY BINDING ====="
    )

    trajectory_policy = {
        "source_family":
            "calibrated_Gaussian_GRU",

        "deterministic_ADB_signal":
            "Gaussian_mean_only",

        "deterministic_GRU_role":
            (
                "SEPARATE_STAGE4_MOTION_BASELINE"
            ),

        "Stage4_prediction_semantics":
            "metric_H0_displacement",

        "predictor_anchor_semantics":
            (
                "latest_causal_observed_"
                "target_position_H0"
            ),

        "future_center_equation":
            (
                "future_center_H0[h] = "
                "predictor_latest_position_H0 + "
                "Gaussian_mean_metric_H0_displacement[h]"
            ),

        "matched_box_center_used_to_recenter_prediction":
            False,

        "predictive_covariance_used_for_"
        "deterministic_mask":
            False,

        "horizons_s":
            [
                0.1,
                0.3,
                0.5,
                1.0,
            ],

        "future_ground_truth":
            False,
    }

    print(
        "trajectory family = calibrated Gaussian GRU"
    )

    print(
        "deterministic signal = MEAN ONLY"
    )

    print(
        "mean semantics = METRIC H0 DISPLACEMENT"
    )

    print(
        "future center = PREDICTOR ANCHOR + MEAN DISPLACEMENT"
    )

    print(
        "snap/recenter to WOMD box = NO"
    )

    print(
        "predictive covariance in deterministic mask = NO"
    )

    # ========================================================
    # D. Freeze future-box construction
    # ========================================================

    print()
    print(
        "===== D. FUTURE FULL-BOX BINDING ====="
    )

    future_box_policy = {
        "box_geometry_source":
            "matched_current_causal_ADB_box",

        "center_source":
            (
                "shared_Gaussian_mean_"
                "future_center_H0"
            ),

        "dimensions": {
            "length_m":
                "current_causal_box_length_m",

            "width_m":
                "current_causal_box_width_m",

            "height_m":
                "current_causal_box_height_m",

            "propagation":
                "HELD_CONSTANT_ACROSS_0_1_TO_1_0_S",

            "reason":
                (
                    "frozen trajectory predictor does not "
                    "output future physical dimensions"
                ),

            "implementation_assumption":
                True,
        },

        "orientation": {
            "future_rule":
                (
                    "Stage5 trajectory-tangent heading "
                    "with low-speed previous-heading "
                    "carry-forward"
                ),

            "trajectory_input":
                (
                    "deterministic Gaussian mean treated "
                    "as one trajectory sample"
                ),

            "current_heading_seed":
                "matched_current_causal_box_yaw_H0",

            "current_heading_role":
                (
                    "LOW_SPEED_CAUSAL_SEED_ONLY"
                ),

            "future_GT_heading":
                False,

            "annotated_future_heading":
                False,
        },

        "projection": {
            "api":
                (
                    "iscai_stage6.adb.geometry."
                    "project_box_to_headlamp"
                ),

            "Box3D":
                True,

            "eight_physical_corners":
                True,

            "centroid_only":
                False,

            "width_height_explicit":
                True,

            "elevation_extent":
                True,

            "ground_range_extent":
                True,
        },

        "current_geometry_provenance":
            (
                "CAUSAL_CURRENT_ANNOTATION_ASSISTED_"
                "GEOMETRY_NOT_FUTURE_ORACLE"
            ),
    }

    print(
        "future dimensions = CURRENT CAUSAL L/W/H HELD"
    )

    print(
        "future yaw = STAGE5 TANGENT RULE"
    )

    print(
        "low-speed seed = CURRENT CAUSAL BOX YAW"
    )

    print(
        "future GT heading = NO"
    )

    print(
        "full 8-corner projection = REQUIRED"
    )

    print(
        "centroid-only projection = FORBIDDEN"
    )

    # ========================================================
    # E. Explicit scope boundaries
    # ========================================================

    print()
    print(
        "===== E. BLOCK6.3 SCOPE BOUNDARIES ====="
    )

    scope = {
        "implemented_next_Block63_Part1": {
            "causal_anchor_association":
                True,

            "deterministic_future_box":
                True,

            "future_box_projection":
                True,
        },

        "not_in_Block63_deterministic_mask": {
            "predictive_covariance":
                True,

            "probabilistic_occupancy":
                True,

            "P_occ_threshold_gamma":
                True,

            "class_specific_margin":
                True,

            "class_specific_floor":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limit":
                True,
        },

        "communication_codebook_reused":
            False,

        "formal_Stage6_grid_frozen":
            False,

        "Stage6_completion_claim":
            False,
    }

    print(
        "probabilistic occupancy = DEFERRED"
    )

    print(
        "class-aware policy = DEFERRED"
    )

    print(
        "smoothing/rate limits = DEFERRED"
    )

    print(
        "communication codebook = NOT REUSED"
    )

    print(
        "formal grid = NOT FROZEN"
    )

    # ========================================================
    # F. Development-before-formal rule
    # ========================================================

    print()
    print(
        "===== F. DEVELOPMENT VALIDATION RULE ====="
    )

    development_policy = {
        "policy_frozen_before_development_result":
            True,

        "development_only_validation_required":
            True,

        "formal_N120_parameter_selection":
            False,

        "formal_results_may_change_association_policy":
            False,

        "association_truth_metadata_role":
            (
                "EVALUATOR_ONLY_AFTER_CONTROLLER_ASSOCIATION"
            ),

        "if_development_validation_reveals_failure":
            (
                "BLOCK_STAGE6_AND_CREATE_EXPLICIT_"
                "PRE_FORMAL_BINDING_AMENDMENT"
            ),

        "silent_posthoc_repair":
            False,
    }

    print(
        "policy selected using formal N120 = NO"
    )

    print(
        "development validation required = YES"
    )

    print(
        "truth metadata = EVALUATOR ONLY"
    )

    print(
        "posthoc formal tuning = FORBIDDEN"
    )

    # ========================================================
    # G. Frozen binding artifact
    # ========================================================

    binding = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "artifact":
            "deterministic_predictive_binding",

        "status":
            "FROZEN_READY_FOR_BLOCK63_PART1",

        "provenance": {
            "block62_closure_sha256":
                EXPECTED[
                    "block62"
                ],

            "interface_audit_sha256":
                EXPECTED[
                    "interface"
                ],

            "exact_route_audit_sha256":
                EXPECTED[
                    "exact_route"
                ],

            "identity_audit_sha256":
                EXPECTED[
                    "identity"
                ],

            "association_policy_audit_sha256":
                EXPECTED[
                    "association"
                ],
        },

        "association_policy":
            association_policy,

        "trajectory_policy":
            trajectory_policy,

        "future_box_policy":
            future_box_policy,

        "scope":
            scope,

        "development_policy":
            development_policy,

        "scientific_classification": {
            "PDF_literal_requirement":
                False,

            "implementation_decision":
                True,

            "reason":
                (
                    "No direct controller-safe actor key "
                    "and no directly reusable frozen "
                    "Stage3 association gate were proven."
                ),
        },
    }

    binding_status = write_exact_json(
        BINDING,
        binding,
    )

    print()
    print(
        "binding artifact =",
        binding_status,
    )

    print(
        "binding path =",
        BINDING,
    )

    print(
        "binding SHA256 =",
        sha256_file(
            BINDING
        ),
    )

    # ========================================================
    # H. Semantic self-check
    # ========================================================

    print()
    print(
        "===== G. FROZEN BINDING SEMANTIC CHECK ====="
    )

    frozen = load_json(
        BINDING
    )

    ap = frozen[
        "association_policy"
    ]

    tp = frozen[
        "trajectory_policy"
    ]

    fp = frozen[
        "future_box_policy"
    ]

    require(
        ap[
            "algorithm"
        ][
            "hard_distance_threshold_m"
        ]
        is
        None,
        "Unexpected association threshold.",
    )

    require(
        ap[
            "algorithm"
        ][
            "forced_assignment"
        ]
        is
        False,
        "Forced assignment is forbidden.",
    )

    require(
        ap[
            "algorithm"
        ][
            "class_gate_before_association"
        ]
        is
        False,
        "Unproven class gate entered association.",
    )

    require(
        ap[
            "unmatched_policy"
        ][
            "causal_ADB_box"
        ]
        ==
        "ORIGINAL_REACTIVE_ADB_FALLBACK",
        (
            "Unmatched actor safety fallback "
            "changed."
        ),
    )

    require(
        tp[
            "matched_box_center_used_to_recenter_prediction"
        ]
        is
        False,
        (
            "Prediction must not be annotation-"
            "recentered after association."
        ),
    )

    require(
        tp[
            "predictive_covariance_used_for_"
            "deterministic_mask"
        ]
        is
        False,
        (
            "Deterministic baseline cannot "
            "use predictive covariance."
        ),
    )

    require(
        fp[
            "projection"
        ][
            "eight_physical_corners"
        ]
        is
        True,
        (
            "Full future-box projection "
            "must retain 8 corners."
        ),
    )

    require(
        fp[
            "projection"
        ][
            "centroid_only"
        ]
        is
        False,
        (
            "Centroid-only future projection "
            "is forbidden."
        ),
    )

    print(
        "association has free numeric threshold = NO"
    )

    print(
        "forced identity = NO"
    )

    print(
        "annotation recentering = NO"
    )

    print(
        "deterministic covariance use = NO"
    )

    print(
        "full-box corners = PASS"
    )

    # ========================================================
    # I. Upstream immutability
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
            "Frozen prerequisite changed: "
            + ", ".join(
                changed
            )
        ),
    )

    print(
        "Block6.2 = UNCHANGED"
    )

    print(
        "Block6.3 audits = UNCHANGED"
    )

    print(
        "Stage3–5 scientific code = UNCHANGED"
    )

    # ========================================================
    # J. Freeze report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "phase":
            "BINDING_FREEZE",

        "status":
            "PASS_READY_FOR_BLOCK63_PART1",

        "binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                sha256_file(
                    BINDING
                ),
        },

        "resolved": {
            "actor_association":
                (
                    "RECIPROCAL_UNIQUE_"
                    "NEAREST_CURRENT_ANCHOR_H0_3D"
                ),

            "new_numeric_association_threshold":
                False,

            "deterministic_trajectory":
                (
                    "SHARED_CALIBRATED_"
                    "GAUSSIAN_MEAN_ONLY"
                ),

            "future_center":
                (
                    "PREDICTOR_ANCHOR_PLUS_"
                    "METRIC_H0_MEAN_DISPLACEMENT"
                ),

            "prediction_recentered_to_box":
                False,

            "future_yaw":
                (
                    "STAGE5_TANGENT_WITH_"
                    "LOW_SPEED_CAUSAL_CARRY_FORWARD"
                ),

            "future_dimensions":
                (
                    "CURRENT_CAUSAL_LWH_HELD"
                ),

            "future_box":
                "FULL_8_CORNER",

            "unmatched_ADB_box":
                "ORIGINAL_REACTIVE_FALLBACK",
        },

        "not_started": {
            "model_inference":
                True,

            "probabilistic_occupancy":
                True,

            "class_aware_ADB":
                True,

            "formal_evaluation":
                True,

            "formal_grid_freeze":
                True,
        },

        "formal_parameter_tuning":
            False,

        "upstream_modified":
            False,

        "next":
            (
                "Block6.3 Part1 deterministic "
                "future full-box projection"
            ),
    }

    report_status = write_exact_json(
        REPORT,
        report,
    )

    print()
    print(
        "freeze report =",
        report_status,
    )

    print(
        "report path =",
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
        "BLOCK 6.3 BINDING FREEZE — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "actor association          = "
        "RECIPROCAL UNIQUE NEAREST H0"
    )

    print(
        "association threshold      = NONE"
    )

    print(
        "forced assignment          = NO"
    )

    print(
        "perfect actor identity     = NO"
    )

    print(
        "unmatched ADB actor        = ORIGINAL REACTIVE FALLBACK"
    )

    print(
        "deterministic trajectory   = SHARED GAUSSIAN MEAN"
    )

    print(
        "future center              = PREDICTOR ANCHOR + MEAN"
    )

    print(
        "annotation recentering     = NO"
    )

    print(
        "future yaw                 = STAGE5 TANGENT RULE"
    )

    print(
        "future dimensions          = CURRENT CAUSAL L/W/H"
    )

    print(
        "future projection          = FULL 8-CORNER BOX"
    )

    print(
        "predictive covariance      = NOT USED IN BLOCK6.3"
    )

    print(
        "class-aware policy         = NOT STARTED"
    )

    print(
        "formal grid                = NOT FROZEN"
    )

    print(
        "development validation     = REQUIRED BEFORE FORMAL"
    )

    print(
        "formal tuning              = FORBIDDEN"
    )

    print(
        "STATUS = PASS_READY_FOR_BLOCK63_PART1"
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
        "BLOCK 6.3 BINDING FREEZE = BLOCKED"
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
        "scientific implementation changed = NO"
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
        "upstream modified     = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

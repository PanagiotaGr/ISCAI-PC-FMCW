from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import traceback

from iscai_stage5.receiver_policy import (
    CONNECTIVITY_SEMANTICS,
    FUTURE_TRUTH_ALLOWED,
    MAX_PLANAR_RANGE_M,
    MAX_RANGE_DEVELOPMENT_TUNING_PERFORMED,
    MAX_RANGE_FORMAL_TUNING_PERFORMED,
    MAX_RANGE_RESOLUTION_BASIS,
    MAX_RANGE_SEMANTICS,
    MINIMUM_FORWARD_H0_M,
    ORACLE_CONNECTIVITY_ALLOWED,
    PERFECT_TRACK_ID_ALLOWED,
    PRIMARY_POLICY,
    RECEIVER_POLICY_STATUS,
    TRACKS_TO_PREDICT_ALLOWED,
    receiver_policy_dict,
)

from iscai_stage5.receiver_selection import (
    ReceiverSelectionConfig,
    select_primary_receiver,
)

from iscai_stage5.stage3_receiver_bridge import (
    CurrentAssociatedActorEstimateH0,
    SOURCE_SEMANTICS,
    receiver_candidate_from_current_estimate,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

PART1 = (
    STAGE5
    / "reports/"
      "block51_part1_receiver_contract.json"
)

BLOCK50 = (
    STAGE5
    / "reports/"
      "block50_contract_freeze.json"
)

STRUCTURAL_CONTRACT = (
    STAGE5
    / "configs/"
      "stage5_contract.json"
)

POLICY_CONFIG = (
    STAGE5
    / "configs/"
      "receiver_policy.json"
)

STAGE3_COMMON = (
    STAGE3
    / "src/iscai_stage3/"
      "contracts/common.py"
)

STAGE3_BRIDGE = (
    STAGE3
    / "src/iscai_stage3/"
      "observations/stage2_bridge.py"
)

REPORT = (
    STAGE5
    / "reports/"
      "block51_receiver_geometry.json"
)

IMPLEMENTATION_LOG = (
    STAGE5
    / "docs/"
      "implementation_log.md"
)


EXPECTED_PART1_IMPLEMENTATION_SHA = (
    "3006d2dd1592d5fa317776635d0355d8"
    "595cc79888aa9973a7feb6d975501e43"
)

EXPECTED_BLOCK50_IMPLEMENTATION_SHA = (
    "f92a4112ef9c23b5db4c24260e18b2e1"
    "da2749c50ac125f623101e45d688ab02"
)

EXPECTED_STAGE5_CONTRACT_SHA = (
    "04fd32e4bc6e160f2627f465a7820c74"
    "e19ca2e1a1f47f23ed1884f842cb7db4"
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


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
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

    temporary.replace(
        path
    )


def require(
    condition,
    message,
):
    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def implementation_fingerprint():
    roots = (
        STAGE5 / "src",
        STAGE5 / "tests",
        STAGE5 / "configs",
        STAGE5 / "scripts",
    )

    files = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE5
                )
            )
    )

    digest = sha256()

    for path in files:

        relative = str(
            path.relative_to(
                STAGE5
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )

        digest.update(
            b"\0"
        )

    return (
        len(
            files
        ),
        digest.hexdigest(),
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.1 PART 2/2"
    )
    print(
        "CAUSAL RECEIVER BRIDGE + POLICY FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK50,
        STRUCTURAL_CONTRACT,
        STAGE3_COMMON,
        STAGE3_BRIDGE,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required artifact/source: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Historical continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK5.0 / BLOCK5.1 PART1 CONTINUITY ====="
    )

    block50 = load_json(
        BLOCK50
    )

    part1 = load_json(
        PART1
    )

    require(
        block50.get(
            "status"
        )
        ==
        "PASS",
        "Block5.0 is not PASS.",
    )

    require(
        block50[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK50_IMPLEMENTATION_SHA,
        (
            "Historical Block5.0 "
            "implementation SHA changed."
        ),
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS",
        "Block5.1 Part1 is not PASS.",
    )

    require(
        part1[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_PART1_IMPLEMENTATION_SHA,
        (
            "Historical Block5.1 Part1 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            STRUCTURAL_CONTRACT
        )
        ==
        EXPECTED_STAGE5_CONTRACT_SHA,
        (
            "Frozen Block5.0 structural "
            "contract changed."
        ),
    )

    print(
        "Block5.0                   = PASS"
    )

    print(
        "Block5.1 Part1             = PASS"
    )

    print(
        "historical implementation = PASS"
    )

    print(
        "Stage5 structural contract= PASS"
    )

    # ========================================================
    # B. Frozen Stage3 observation-interface provenance
    # ========================================================

    print()
    print(
        "===== B. FROZEN STAGE3 OBSERVATION-INTERFACE PROVENANCE ====="
    )

    common_source = (
        STAGE3_COMMON.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    bridge_source = (
        STAGE3_BRIDGE.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    require(
        "stage2_degraded_algorithm_facing"
        in
        common_source,
        (
            "Stage3 common observation contract "
            "no longer exposes the frozen "
            "Stage2-degraded algorithm-facing semantics."
        ),
    )

    require(
        "algorithm_sequence_from_degraded_scene"
        in
        bridge_source,
        (
            "Stage3 Stage2-degraded bridge "
            "was not found."
        ),
    )

    stage3_source_hashes = {
        "common_contract":
            file_sha256(
                STAGE3_COMMON
            ),

        "Stage2_degraded_bridge":
            file_sha256(
                STAGE3_BRIDGE
            ),
    }

    print(
        "Stage3 observation contract = PASS"
    )

    print(
        "Stage2 degraded bridge       = PASS"
    )

    print(
        "algorithm-facing semantics   = FROZEN"
    )

    print(
        "future-truth bridge          = NOT INTRODUCED"
    )

    # ========================================================
    # C. Narrow causal Stage3→Stage5 boundary
    # ========================================================

    print()
    print(
        "===== C. CAUSAL CURRENT-ACTOR BRIDGE ====="
    )

    bridge_signature = str(
        inspect.signature(
            CurrentAssociatedActorEstimateH0
        )
    ).lower()

    candidate_signature = str(
        inspect.signature(
            receiver_candidate_from_current_estimate
        )
    ).lower()

    for signature in (
        bridge_signature,
        candidate_signature,
    ):
        for forbidden in (
            "tracks_to_predict",
            "objects_of_interest",
            "future_truth",
            "ground_truth",
            "oracle",
            "track_id",
        ):
            require(
                forbidden
                not in
                signature,
                (
                    "Forbidden metadata surfaced "
                    f"in Stage3→Stage5 bridge: "
                    f"{forbidden}"
                ),
            )

    estimate = (
        CurrentAssociatedActorEstimateH0(
            semantic_class="vehicle",

            position_h0_m=(
                25.0,
                1.5,
                0.0,
            ),

            current_available=True,

            association_valid=True,
        )
    )

    candidate = (
        receiver_candidate_from_current_estimate(
            estimate
        )
    )

    require(
        candidate.semantic_class
        ==
        "vehicle",
        "Bridge semantic class mismatch.",
    )

    require(
        candidate.position_h0_m
        ==
        (
            25.0,
            1.5,
            0.0,
        ),
        "Bridge position mismatch.",
    )

    require(
        estimate.source_semantics
        ==
        SOURCE_SEMANTICS,
        "Bridge source semantics changed.",
    )

    print(
        "source = CURRENT CAUSAL ASSOCIATED ACTOR ESTIMATE"
    )

    print(
        "semantic class            = CURRENT METADATA ONLY"
    )

    print(
        "current H0 position       = PASS"
    )

    print(
        "association validity      = PASS"
    )

    print(
        "tracks_to_predict         = NOT AVAILABLE"
    )

    print(
        "future truth              = NOT AVAILABLE"
    )

    print(
        "perfect track ID          = NOT AVAILABLE"
    )

    # ========================================================
    # D. Resolve receiver range policy
    # ========================================================

    print()
    print(
        "===== D. FINAL RECEIVER-ELIGIBILITY RANGE POLICY ====="
    )

    require(
        RECEIVER_POLICY_STATUS
        ==
        "FROZEN",
        "Receiver policy is not frozen.",
    )

    require(
        PRIMARY_POLICY
        ==
        "nearest_causal_vehicle_ahead",
        "Primary receiver policy changed.",
    )

    require(
        CONNECTIVITY_SEMANTICS
        ==
        "constructed_hypothetical_connected_vehicle",
        "Connectivity semantics changed.",
    )

    require(
        MINIMUM_FORWARD_H0_M
        ==
        0.0,
        "Forward receiver threshold changed.",
    )

    require(
        MAX_PLANAR_RANGE_M
        is None,
        (
            "Unexpected finite receiver "
            "eligibility range cutoff."
        ),
    )

    require(
        MAX_RANGE_DEVELOPMENT_TUNING_PERFORMED
        is False,
        (
            "Receiver range cutoff was "
            "development-tuned unexpectedly."
        ),
    )

    require(
        MAX_RANGE_FORMAL_TUNING_PERFORMED
        is False,
        (
            "Receiver range cutoff was "
            "formal-tuned unexpectedly."
        ),
    )

    selector_config = (
        ReceiverSelectionConfig(
            minimum_forward_h0_m=(
                MINIMUM_FORWARD_H0_M
            ),

            max_planar_range_m=(
                MAX_PLANAR_RANGE_M
            ),
        )
    )

    require(
        selector_config.max_planar_range_m
        is None,
        "Resolved selector config mismatch.",
    )

    print(
        "minimum forward H0       = > 0 m"
    )

    print(
        "max receiver range       = NONE"
    )

    print(
        "range cutoff tuning      = NO"
    )

    print(
        "formal tuning            = NO"
    )

    print(
        "distance remains         = REPORTED / SLICED LATER"
    )

    print(
        "resolution basis         =",
        MAX_RANGE_RESOLUTION_BASIS,
    )

    # ========================================================
    # E. Final anti-leakage assertions
    # ========================================================

    print()
    print(
        "===== E. FINAL BLOCK5.1 ANTI-LEAKAGE ====="
    )

    require(
        TRACKS_TO_PREDICT_ALLOWED
        is False,
        (
            "tracks_to_predict became "
            "receiver-policy input."
        ),
    )

    require(
        FUTURE_TRUTH_ALLOWED
        is False,
        (
            "future truth became "
            "receiver-policy input."
        ),
    )

    require(
        PERFECT_TRACK_ID_ALLOWED
        is False,
        (
            "perfect track ID became "
            "receiver-policy input."
        ),
    )

    require(
        ORACLE_CONNECTIVITY_ALLOWED
        is False,
        (
            "oracle connectivity became "
            "receiver-policy input."
        ),
    )

    print(
        "tracks_to_predict        = NO"
    )

    print(
        "future truth             = NO"
    )

    print(
        "perfect track ID         = NO"
    )

    print(
        "oracle connectivity      = NO"
    )

    print(
        "formal N=120 read        = NO"
    )

    print(
        "Stage4 inference         = NO"
    )

    print(
        "training/recalibration   = NO"
    )

    # ========================================================
    # F. Freeze policy JSON
    # ========================================================

    print()
    print(
        "===== F. WRITING FROZEN RECEIVER POLICY ====="
    )

    policy = receiver_policy_dict()

    policy[
        "upstream_provenance"
    ] = {
        "Stage3_common_contract":
            {
                "path":
                    str(
                        STAGE3_COMMON
                    ),

                "sha256":
                    stage3_source_hashes[
                        "common_contract"
                    ],
            },

        "Stage3_degraded_bridge":
            {
                "path":
                    str(
                        STAGE3_BRIDGE
                    ),

                "sha256":
                    stage3_source_hashes[
                        "Stage2_degraded_bridge"
                    ],
            },

        "Block5.0_contract_sha256":
            EXPECTED_STAGE5_CONTRACT_SHA,

        "Block5.1_Part1_implementation_sha256":
            EXPECTED_PART1_IMPLEMENTATION_SHA,
    }

    write_json(
        POLICY_CONFIG,
        policy,
    )

    policy_sha = file_sha256(
        POLICY_CONFIG
    )

    policy_readback = load_json(
        POLICY_CONFIG
    )

    require(
        policy_readback[
            "status"
        ]
        ==
        "FROZEN",
        "Receiver-policy readback failed.",
    )

    require(
        policy_readback[
            "max_planar_range_m"
        ]
        is None,
        (
            "Frozen receiver policy "
            "unexpectedly contains a range cutoff."
        ),
    )

    require(
        policy_readback[
            "forbidden_inputs"
        ][
            "tracks_to_predict"
        ]
        is False,
        (
            "Frozen policy permits "
            "tracks_to_predict."
        ),
    )

    print(
        "receiver policy JSON     = FROZEN"
    )

    print(
        "receiver policy SHA256   =",
        policy_sha,
    )

    # ========================================================
    # G. Final report
    # ========================================================

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.1",

        "status":
            "PASS",

        "Part1": {
            "status":
                "PASS",

            "report":
                str(
                    PART1
                ),

            "historical_implementation_sha256":
                EXPECTED_PART1_IMPLEMENTATION_SHA,
        },

        "Stage3_provenance": {
            "observation_interface":
                "stage2_degraded_algorithm_facing",

            "current_actor_bridge":
                SOURCE_SEMANTICS,

            "source_hashes":
                stage3_source_hashes,

            "future_truth_added":
                False,
        },

        "receiver_policy": {
            "path":
                str(
                    POLICY_CONFIG
                ),

            "sha256":
                policy_sha,

            "status":
                "FROZEN",

            "primary_policy":
                PRIMARY_POLICY,

            "connectivity":
                CONNECTIVITY_SEMANTICS,

            "minimum_forward_H0_m":
                MINIMUM_FORWARD_H0_M,

            "max_planar_range_m":
                None,

            "max_range_semantics":
                MAX_RANGE_SEMANTICS,

            "max_range_performance_tuned":
                False,

            "distance_still_evaluated":
                True,
        },

        "receiver_geometry": {
            "modes":
                [
                    "centroid_baseline",
                    "known_receiver_offset",
                    "uncertain_receiver_offset",
                ],

            "placement_covariance":
                "receiver_placement_uncertainty_only",

            "trajectory_plus_placement_uncertainty":
                "Block5.2",
        },

        "causal_bridge": {
            "input":
                SOURCE_SEMANTICS,

            "semantic_class":
                "current_causal_metadata",

            "position":
                "current_estimated_H0",

            "association_validity":
                "current",

            "perfect_track_ID":
                False,

            "tracks_to_predict":
                False,

            "future_truth":
                False,

            "oracle_connectivity":
                False,
        },

        "scientific_execution": {
            "dataset_scan":
                False,

            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "receiver_posterior_inference":
                False,

            "beam_selection":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "resolved_development_fields": {
            "receiver_eligibility_max_range":
                {
                    "value_m":
                        None,

                    "semantics":
                        MAX_RANGE_SEMANTICS,

                    "performance_tuned":
                        False,

                    "formal_data_used":
                        False,
                },
        },

        "remaining_development_fields_for_later_blocks": [
            "low_speed_heading_threshold",
            "angular_monte_carlo_sample_count",
            "codebook_angular_support",
            "formal_empirical_coverage_tolerance",
            "K_max",
            "hysteresis_parameters",
            "beam_switch_penalty",
            "local_neighbour_sweep_width",
            "fallback_parameters",
            "reacquisition_parameters",
        ],

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block5.2 receiver-aware "
                "trajectory-to-angular posterior"
            ),
    }

    write_json(
        REPORT,
        payload,
    )

    # ========================================================
    # H. Implementation log
    # ========================================================

    marker = (
        "## Block 5.1 — "
        "Communication receiver and geometry"
    )

    existing = (
        IMPLEMENTATION_LOG.read_text(
            encoding="utf-8"
        )
        if IMPLEMENTATION_LOG.is_file()
        else ""
    )

    if marker not in existing:

        with IMPLEMENTATION_LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:

            stream.write(
                "\n"
                +
                marker
                +
                "\n\n"
                "Status: PASS / FROZEN\n\n"
                "- Primary communication receiver policy: "
                  "nearest causal vehicle ahead in H0.\n"
                "- Connectivity is a constructed hypothetical "
                  "experimental assumption because WOMD does "
                  "not provide a measured communication-"
                  "connectivity label.\n"
                "- Receiver selection consumes only current "
                  "causal associated actor estimates and "
                  "current semantic metadata.\n"
                "- tracks_to_predict, future truth, perfect "
                  "track IDs and oracle connectivity are not "
                  "receiver-selector inputs.\n"
                "- Receiver geometry modes remain centroid, "
                  "known offset and uncertain offset.\n"
                "- Receiver-placement covariance remains "
                  "distinct from Stage2 measurement R_t and "
                  "Stage4 predictive covariance.\n"
                "- No additional finite receiver-range cutoff "
                  "is used in the primary policy; distance is "
                  "reported/sliced rather than tuned as an "
                  "eligibility threshold.\n"
                "- No formal N=120 data, Stage4 inference, "
                  "training or recalibration was used in "
                  "Block5.1.\n"
                f"- Frozen receiver-policy SHA256: "
                  f"{policy_sha}.\n"
                f"- Block5.1 implementation SHA256: "
                  f"{implementation_sha}.\n"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.1 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.0 continuity           = PASS"
    )

    print(
        "Block5.1 Part1 continuity     = PASS"
    )

    print(
        "Stage3 degraded observation   = PASS"
    )

    print(
        "current associated actor bridge= PASS"
    )

    print(
        "primary receiver policy       = FROZEN"
    )

    print(
        "constructed connectivity      = FROZEN"
    )

    print(
        "max receiver range            = NONE"
    )

    print(
        "range-cutoff tuning           = NO"
    )

    print(
        "distance evaluation           = PRESERVED"
    )

    print(
        "tracks_to_predict             = NO"
    )

    print(
        "future truth                  = NO"
    )

    print(
        "perfect track ID              = NO"
    )

    print(
        "oracle connectivity           = NO"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "Stage4 inference              = NO"
    )

    print(
        "training/recalibration        = NO"
    )

    print(
        "receiver policy SHA256        =",
        policy_sha,
    )

    print(
        "implementation files          =",
        implementation_files,
    )

    print(
        "implementation SHA256         =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.2"
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
        "BLOCK 5.1 PART 2/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print(
        "dataset/formal data used = NO"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "training/recalibration    = NO"
    )

    print(
        "upstream modified         = NO"
    )

    print(
        "terminal remains open     = YES"
    )

# Deliberately no sys.exit().

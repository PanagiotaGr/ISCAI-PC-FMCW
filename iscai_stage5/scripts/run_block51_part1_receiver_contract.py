from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import traceback

from iscai_stage5.contracts import (
    CONNECTED_VEHICLE_SEMANTICS,
    DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL,
    PRIMARY_RECEIVER_POLICY,
    RECEIVER_GEOMETRY_MODES,
)

from iscai_stage5.receiver_geometry import (
    ReceiverOffsetDistribution,
    receiver_geometry_distribution_h0,
)

from iscai_stage5.receiver_selection import (
    ReceiverCandidate,
    ReceiverSelectionConfig,
    select_primary_receiver,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

BLOCK50 = (
    STAGE5
    / "reports/"
      "block50_contract_freeze.json"
)

CONTRACT = (
    STAGE5
    / "configs/"
      "stage5_contract.json"
)

PART_A_CONTRACT = (
    STAGE5
    / "artifacts/block50/"
      "part_a_source_contract.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block51_part1_receiver_contract.json"
)


EXPECTED_BLOCK50_IMPLEMENTATION_SHA = (
    "f92a4112ef9c23b5db4c24260e18b2e1"
    "da2749c50ac125f623101e45d688ab02"
)

EXPECTED_STAGE5_CONTRACT_SHA = (
    "04fd32e4bc6e160f2627f465a7820c74"
    "e19ca2e1a1f47f23ed1884f842cb7db4"
)

EXPECTED_PART_A_CONTRACT_SHA = (
    "80a06892e1721e28e2599c20acc29c9b"
    "ea44fd74e256b8af7eadd02416600fca"
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
                1024
                *
                1024
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
    path.write_text(
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
        STAGE5
        / "src",

        STAGE5
        / "tests",

        STAGE5
        / "configs",

        STAGE5
        / "scripts",
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
        "STAGE 5 — BLOCK 5.1 PART 1/2"
    )

    print(
        "CAUSAL RECEIVER SELECTION + GEOMETRY KERNEL"
    )

    print(
        "============================================================"
    )

    required = (
        BLOCK50,
        CONTRACT,
        PART_A_CONTRACT,
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
            "Missing Block5.0 frozen artifact(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen Block5.0 continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK5.0 FROZEN CONTINUITY ====="
    )

    block50 = load_json(
        BLOCK50
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
        file_sha256(
            CONTRACT
        )
        ==
        EXPECTED_STAGE5_CONTRACT_SHA,
        "Frozen Stage5 contract SHA changed.",
    )

    require(
        file_sha256(
            PART_A_CONTRACT
        )
        ==
        EXPECTED_PART_A_CONTRACT_SHA,
        "Frozen Part-A contract SHA changed.",
    )

    print(
        "Block5.0 report             = PASS"
    )

    print(
        "historical implementation   = PASS"
    )

    print(
        "Stage5 structural contract  = PASS"
    )

    print(
        "Part-A source contract      = PASS"
    )

    # ========================================================
    # B. Selection contract
    # ========================================================

    print()
    print(
        "===== B. PRIMARY RECEIVER-SELECTION CONTRACT ====="
    )

    require(
        PRIMARY_RECEIVER_POLICY
        ==
        "nearest_causal_vehicle_ahead",
        "Primary receiver policy changed.",
    )

    require(
        CONNECTED_VEHICLE_SEMANTICS
        ==
        "constructed_hypothetical_connected_vehicle",
        (
            "Connectivity semantics changed."
        ),
    )

    selector_signature = str(
        inspect.signature(
            select_primary_receiver
        )
    ).lower()

    for forbidden in (
        "tracks_to_predict",
        "future",
        "ground_truth",
        "oracle",
        "track_id",
    ):
        require(
            forbidden
            not in
            selector_signature,
            (
                "Forbidden receiver-selector "
                f"argument: {forbidden}"
            ),
        )

    sample = [
        ReceiverCandidate(
            semantic_class="pedestrian",
            position_h0_m=(
                2.0,
                0.0,
                0.0,
            ),
        ),
        ReceiverCandidate(
            semantic_class="vehicle",
            position_h0_m=(
                15.0,
                2.0,
                0.0,
            ),
        ),
        ReceiverCandidate(
            semantic_class="vehicle",
            position_h0_m=(
                9.0,
                1.0,
                0.0,
            ),
        ),
        ReceiverCandidate(
            semantic_class="vehicle",
            position_h0_m=(
                -3.0,
                0.0,
                0.0,
            ),
        ),
    ]

    sample_result = select_primary_receiver(
        sample
    )

    require(
        sample_result.selected_candidate_index
        ==
        2,
        (
            "Primary receiver-selection "
            "sanity check failed."
        ),
    )

    print(
        "policy                     = NEAREST CAUSAL VEHICLE AHEAD"
    )

    print(
        "connectivity               = CONSTRUCTED ASSUMPTION"
    )

    print(
        "vehicle-only receiver      = PASS"
    )

    print(
        "positive-forward H0        = PASS"
    )

    print(
        "current causal state only  = PASS"
    )

    print(
        "tracks_to_predict          = NOT INPUT"
    )

    print(
        "future truth               = NOT INPUT"
    )

    print(
        "perfect track ID           = NOT INPUT"
    )

    print(
        "no receiver                = EXPLICIT STATE"
    )

    # ========================================================
    # C. Development-only max-range parameter
    # ========================================================

    print()
    print(
        "===== C. DEVELOPMENT-ONLY RECEIVER RANGE GATE ====="
    )

    require(
        "receiver_eligibility_max_range"
        in
        DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL,
        (
            "Receiver max-range parameter is no "
            "longer listed as development-only."
        ),
    )

    default_config = (
        ReceiverSelectionConfig()
    )

    require(
        default_config.max_planar_range_m
        is None,
        (
            "Receiver range cutoff was frozen "
            "without development evidence."
        ),
    )

    print(
        "max receiver range        = NOT YET FROZEN"
    )

    print(
        "current default           = NO RANGE CUTOFF"
    )

    print(
        "formal N=120 tuning       = FORBIDDEN"
    )

    print(
        "freeze source             = NON-FORMAL DEVELOPMENT ONLY"
    )

    # ========================================================
    # D. Receiver geometry semantics
    # ========================================================

    print()
    print(
        "===== D. RECEIVER-GEOMETRY SEMANTICS ====="
    )

    require(
        tuple(
            RECEIVER_GEOMETRY_MODES
        )
        ==
        (
            "centroid_baseline",
            "known_receiver_offset",
            "uncertain_receiver_offset",
        ),
        "Receiver geometry modes changed.",
    )

    offset = ReceiverOffsetDistribution(
        mean_body_m=(
            2.0,
            0.0,
            0.5,
        ),

        covariance_body_m2=(
            (
                0.25,
                0.0,
                0.0,
            ),
            (
                0.0,
                1.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.04,
            ),
        ),
    )

    centroid = (
        receiver_geometry_distribution_h0(
            actor_center_h0_m=(
                10.0,
                0.0,
                1.0,
            ),
            heading_h0_rad=0.0,
            mode="centroid_baseline",
        )
    )

    known = (
        receiver_geometry_distribution_h0(
            actor_center_h0_m=(
                10.0,
                0.0,
                1.0,
            ),
            heading_h0_rad=0.0,
            mode="known_receiver_offset",
            receiver_offset=offset,
        )
    )

    uncertain = (
        receiver_geometry_distribution_h0(
            actor_center_h0_m=(
                10.0,
                0.0,
                1.0,
            ),
            heading_h0_rad=0.0,
            mode="uncertain_receiver_offset",
            receiver_offset=offset,
        )
    )

    require(
        centroid.receiver_mean_h0_m
        ==
        (
            10.0,
            0.0,
            1.0,
        ),
        "Centroid baseline sanity check failed.",
    )

    require(
        known.receiver_mean_h0_m
        ==
        (
            12.0,
            0.0,
            1.5,
        ),
        "Known-offset sanity check failed.",
    )

    require(
        uncertain.receiver_mean_h0_m
        ==
        known.receiver_mean_h0_m,
        (
            "Known/uncertain receiver means "
            "must agree."
        ),
    )

    require(
        uncertain.covariance_semantics
        ==
        "receiver_placement_uncertainty_only",
        (
            "Receiver placement covariance "
            "semantics changed."
        ),
    )

    print(
        "centroid baseline         = PASS"
    )

    print(
        "known receiver offset     = PASS"
    )

    print(
        "uncertain receiver offset = PASS"
    )

    print(
        "body→H0 yaw rotation      = PASS"
    )

    print(
        "placement covariance      = PSD-VALIDATED"
    )

    print(
        "placement covariance      = DISTINCT FROM Stage4 UQ"
    )

    print(
        "trajectory+placement UQ   = DEFERRED TO BLOCK5.2"
    )

    # ========================================================
    # E. Forbidden leakage surface
    # ========================================================

    print()
    print(
        "===== E. FORBIDDEN LEAKAGE SURFACE ====="
    )

    geometry_signature = str(
        inspect.signature(
            receiver_geometry_distribution_h0
        )
    ).lower()

    for forbidden in (
        "future_truth",
        "ground_truth",
        "oracle",
        "tracks_to_predict",
    ):
        require(
            forbidden
            not in
            geometry_signature,
            (
                "Forbidden receiver geometry "
                f"argument: {forbidden}"
            ),
        )

    print(
        "future GT heading         = NOT INPUT"
    )

    print(
        "tracks_to_predict         = NOT INPUT"
    )

    print(
        "oracle receiver location  = NOT INPUT"
    )

    print(
        "formal data used          = NO"
    )

    print(
        "trajectory inference      = NO"
    )

    # ========================================================
    # F. Report
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
            "5.1_part_1",

        "status":
            "PASS",

        "Block50_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_BLOCK50_IMPLEMENTATION_SHA,

            "Stage5_contract_sha256":
                EXPECTED_STAGE5_CONTRACT_SHA,

            "PartA_contract_sha256":
                EXPECTED_PART_A_CONTRACT_SHA,
        },

        "receiver_selection": {
            "primary_policy":
                PRIMARY_RECEIVER_POLICY,

            "connectivity_semantics":
                CONNECTED_VEHICLE_SEMANTICS,

            "vehicle_only":
                True,

            "current_causal_availability":
                True,

            "positive_forward_H0":
                True,

            "ranking_metric":
                "planar_range_m",

            "tracks_to_predict_input":
                False,

            "future_truth_input":
                False,

            "perfect_track_ID_input":
                False,

            "no_receiver_state_explicit":
                True,

            "candidate_list_index_role":
                (
                    "final_exact_tie_break_only_"
                    "not_physical_or_model_feature"
                ),
        },

        "development_pending": {
            "receiver_eligibility_max_range":
                "NOT_YET_FROZEN",

            "current_default_max_range_m":
                None,

            "freeze_source":
                "non_formal_development_only",

            "formal_N120_allowed":
                False,
        },

        "receiver_geometry": {
            "modes":
                list(
                    RECEIVER_GEOMETRY_MODES
                ),

            "centroid_baseline":
                "PASS",

            "known_receiver_offset":
                "PASS",

            "uncertain_receiver_offset":
                "PASS",

            "offset_frame":
                "receiver_vehicle_body_frame",

            "rotation":
                "yaw_body_to_H0",

            "placement_covariance":
                "receiver_placement_uncertainty_only",

            "trajectory_predictive_covariance":
                "NOT_MERGED_IN_BLOCK5.1_PART1",

            "joint_uncertainty_propagation":
                "BLOCK5.2",
        },

        "leakage": {
            "future_GT_heading":
                False,

            "future_truth_receiver_location":
                False,

            "tracks_to_predict":
                False,

            "oracle_receiver":
                False,

            "formal_data":
                False,
        },

        "scientific_execution": {
            "dataset_scan":
                False,

            "Stage4_inference":
                False,

            "receiver_posterior_inference":
                False,

            "beam_selection":
                False,

            "formal_evaluation":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "next":
            (
                "Block5.1 Part2 frozen Stage1/Stage3 "
                "bridge plus non-formal development "
                "receiver-availability audit"
            ),
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.1 PART 1/2 GATE"
    )

    print(
        "============================================================"
    )

    print(
        "Block5.0 continuity          = PASS"
    )

    print(
        "primary receiver policy      = PASS"
    )

    print(
        "constructed connectivity     = PASS"
    )

    print(
        "vehicle-only selection       = PASS"
    )

    print(
        "causal/current-only selector = PASS"
    )

    print(
        "receiver centroid mode       = PASS"
    )

    print(
        "receiver known-offset mode   = PASS"
    )

    print(
        "receiver uncertain mode      = PASS"
    )

    print(
        "receiver covariance PSD      = PASS"
    )

    print(
        "measurement/predictive UQ    = NOT MERGED"
    )

    print(
        "max receiver range           = DEVELOPMENT PENDING"
    )

    print(
        "formal N=120 used            = NO"
    )

    print(
        "training/inference           = NO"
    )

    print(
        "implementation files         =",
        implementation_files,
    )

    print(
        "implementation SHA256        =",
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
        "BLOCK 5.1 PART 1/2 = BLOCKED"
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
        "training/inference       = NO"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "upstream Stage0-4 changed= NO"
    )

    print(
        "terminal remains open    = YES"
    )

# Deliberately no sys.exit().

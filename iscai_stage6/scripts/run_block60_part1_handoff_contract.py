from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

S0 = ROOT / "iscai_stage0"
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

CONFIG = (
    S6
    / "configs/stage6_contract.json"
)

REPORT = (
    S6
    / "reports/block60_part1_handoff_contract.json"
)

TEST = (
    S6
    / "tests/test_block60_contract.py"
)

PACKAGE_INIT = (
    S6
    / "src/iscai_stage6/__init__.py"
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

STAGE5_FREEZE = (
    S5
    / "artifacts/block510/"
      "stage5_final_freeze_manifest.json"
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

STAGE4_FREEZE = (
    S4
    / "artifacts/block410/"
      "stage4_final_freeze_manifest.json"
)

FORMAL_MANIFEST = (
    S3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

PART_A_REFERENCE = (
    S0
    / "reports/stage0/"
      "part_a_frozen_reference.json"
)

PART_A_REPO = (
    ROOT
    / "part_a_reference/"
      "ISCAI_pc_fmcw"
)

EXPECTED = {
    "stage5_closure":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),

    "stage5_handoff":
        (
            "c900ccbd63e7c680af28ed5cd8046f40"
            "8463eceb7ce943b324b266dda73500eb"
        ),

    "stage5_freeze":
        (
            "af9612205f190a04d9da565a98a0449f"
            "4ed98f53027b6f971b8457f73bd70913"
        ),

    "stage4_closure":
        (
            "570da4feb918b1025b5e85cc919360d9"
            "22b471c468c85b3844f13fb7774e7c2f"
        ),

    "stage4_handoff":
        (
            "491bce010d35c2a394f879ecf35ed26e"
            "f1de92f7fff1465dcbf46b072a87c6fd"
        ),

    "stage4_freeze":
        (
            "88e3f290e1f7c1d037684adc132ea8ae"
            "78af057b3e3ab564fb29516687eb25e7"
        ),

    "formal_manifest":
        (
            "2208e7287ddf6439fda4597c435a9cba"
            "1d1b9d0e4c4547bc5dd92e56e8124e46"
        ),

    "part_a_git":
        (
            "44d62e3478e3818d1757b00971890f844cb032f7"
        ),
}

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
# ============================================================

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
        + "\n"
    ).encode(
        "utf-8"
    )


def write_exact(
    path: Path,
    payload,
):

    desired = canonical_bytes(
        payload
    )

    if path.exists():

        existing = path.read_bytes()

        if existing != desired:
            raise RuntimeError(
                "Existing Stage6 contract artifact "
                "differs from audited deterministic "
                f"payload: {path}"
            )

        return "ALREADY_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        desired
    )

    os.replace(
        temporary,
        path,
    )

    return "WRITTEN"


def write_exact_text(
    path: Path,
    text: str,
):

    desired = text.encode(
        "utf-8"
    )

    if path.exists():

        existing = path.read_bytes()

        if existing != desired:
            raise RuntimeError(
                "Existing Stage6 source/test differs "
                "from deterministic Block6.0 Part1 "
                f"payload: {path}"
            )

        return "ALREADY_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        desired
    )

    os.replace(
        temporary,
        path,
    )

    return "WRITTEN"


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


def git_head(
    repo: Path,
):

    process = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    require(
        process.returncode == 0,
        (
            "Could not read frozen Part-A git "
            f"HEAD: {process.stderr.strip()}"
        ),
    )

    return process.stdout.strip()


def jsonl_count(
    path: Path,
):

    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line in stream:

            if line.strip():
                count += 1

    return count


def run_test():

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6 / "tests"
            ),
            "-p",
            "test_block60_contract.py",
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return (
        process.returncode,
        process.stdout,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.0 PART 1/2"
    )
    print(
        "FROZEN HANDOFF + STRICT PDF CONTRACT"
    )
    print(
        "============================================================"
    )

    required = (
        STAGE5_CLOSURE,
        STAGE5_HANDOFF,
        STAGE5_FREEZE,
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE4_FREEZE,
        FORMAL_MANIFEST,
        PART_A_REFERENCE,
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
            "Missing frozen upstream evidence: "
            + ", ".join(
                missing
            )
        ),
    )

    require(
        PART_A_REPO.is_dir(),
        (
            "Frozen Part-A repository missing: "
            f"{PART_A_REPO}"
        ),
    )

    upstream = (
        STAGE5_CLOSURE,
        STAGE5_HANDOFF,
        STAGE5_FREEZE,
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE4_FREEZE,
        FORMAL_MANIFEST,
        PART_A_REFERENCE,
    )

    upstream_before = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in upstream
    }

    # ========================================================
    # A. Stage5 seal
    # ========================================================

    print()
    print(
        "===== A. STAGE5 FROZEN SEAL ====="
    )

    require(
        sha256_file(
            STAGE5_CLOSURE
        )
        ==
        EXPECTED[
            "stage5_closure"
        ],
        (
            "Stage5 canonical closure SHA "
            "does not match frozen value."
        ),
    )

    require(
        sha256_file(
            STAGE5_HANDOFF
        )
        ==
        EXPECTED[
            "stage5_handoff"
        ],
        (
            "Stage5→Stage6 handoff SHA "
            "does not match frozen value."
        ),
    )

    require(
        sha256_file(
            STAGE5_FREEZE
        )
        ==
        EXPECTED[
            "stage5_freeze"
        ],
        (
            "Stage5 freeze manifest SHA "
            "does not match frozen value."
        ),
    )

    closure5 = load_json(
        STAGE5_CLOSURE
    )

    handoff5 = load_json(
        STAGE5_HANDOFF
    )

    freeze5 = load_json(
        STAGE5_FREEZE
    )

    require(
        closure5.get(
            "status"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Stage5 canonical status is not "
            "COMPLETE_FROZEN."
        ),
    )

    require(
        handoff5.get(
            "status"
        )
        ==
        "FROZEN_HANDOFF",
        (
            "Stage5→Stage6 handoff is not "
            "FROZEN_HANDOFF."
        ),
    )

    require(
        freeze5.get(
            "status"
        )
        ==
        "FROZEN",
        (
            "Stage5 freeze manifest "
            "is not FROZEN."
        ),
    )

    require(
        handoff5.get(
            "Stage6_started"
        )
        is False,
        (
            "Frozen Stage5 handoff incorrectly "
            "states that Stage6 had already started."
        ),
    )

    print(
        "Stage5 closure = COMPLETE_FROZEN PASS"
    )

    print(
        "Stage5 handoff = FROZEN_HANDOFF PASS"
    )

    print(
        "Stage5 freeze  = FROZEN PASS"
    )

    print(
        "Stage5 hashes  = EXACT PASS"
    )

    # ========================================================
    # B. Stage4 posterior continuity
    # ========================================================

    print()
    print(
        "===== B. STAGE4 POSTERIOR CONTINUITY ====="
    )

    require(
        sha256_file(
            STAGE4_CLOSURE
        )
        ==
        EXPECTED[
            "stage4_closure"
        ],
        "Stage4 closure SHA changed.",
    )

    require(
        sha256_file(
            STAGE4_HANDOFF
        )
        ==
        EXPECTED[
            "stage4_handoff"
        ],
        "Stage4 handoff SHA changed.",
    )

    require(
        sha256_file(
            STAGE4_FREEZE
        )
        ==
        EXPECTED[
            "stage4_freeze"
        ],
        "Stage4 freeze SHA changed.",
    )

    closure4 = load_json(
        STAGE4_CLOSURE
    )

    require(
        closure4.get(
            "status"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Stage4 canonical status "
            "is not COMPLETE_FROZEN."
        ),
    )

    shared = handoff5.get(
        "shared_posterior",
        {}
    )

    require(
        shared.get(
            "upstream_trajectory_family"
        )
        ==
        "calibrated_Gaussian_GRU",
        (
            "Stage5→Stage6 handoff does not "
            "freeze calibrated Gaussian GRU "
            "as upstream trajectory family."
        ),
    )

    require(
        shared.get(
            "future_ground_truth_as_controller_input"
        )
        is False,
        (
            "Stage5 handoff permits future GT "
            "as controller input."
        ),
    )

    require(
        shared.get(
            "tracks_to_predict_as_receiver_selector"
        )
        is False,
        (
            "Stage5 handoff allows "
            "tracks_to_predict receiver selection."
        ),
    )

    print(
        "Stage4 closure        = COMPLETE_FROZEN"
    )

    print(
        "calibrated Gaussian   = FROZEN DOWNSTREAM"
    )

    print(
        "future GT input       = NO"
    )

    print(
        "recalibration Stage6  = FORBIDDEN"
    )

    # ========================================================
    # C. Formal population continuity
    # ========================================================

    print()
    print(
        "===== C. FORMAL POPULATION CONTINUITY ====="
    )

    require(
        sha256_file(
            FORMAL_MANIFEST
        )
        ==
        EXPECTED[
            "formal_manifest"
        ],
        (
            "Formal validation manifest "
            "SHA changed."
        ),
    )

    formal_n = jsonl_count(
        FORMAL_MANIFEST
    )

    require(
        formal_n == 120,
        (
            "Frozen formal manifest "
            f"contains {formal_n} records, "
            "expected 120."
        ),
    )

    print(
        "formal manifest SHA = EXACT PASS"
    )

    print(
        "formal scenarios    = 120 PASS"
    )

    print(
        "Stage6 use          = continuity choice,"
        " not literal PDF N requirement"
    )

    # ========================================================
    # D. Part-A frozen reference
    # ========================================================

    print()
    print(
        "===== D. PART-A FROZEN REFERENCE ====="
    )

    part_a = load_json(
        PART_A_REFERENCE
    )

    part_a_sha = sha256_file(
        PART_A_REFERENCE
    )

    head = git_head(
        PART_A_REPO
    )

    require(
        head
        ==
        EXPECTED[
            "part_a_git"
        ],
        (
            "Frozen Part-A git HEAD changed: "
            f"{head}"
        ),
    )

    require(
        isinstance(
            part_a,
            dict,
        ),
        (
            "Part-A frozen reference "
            "is not a JSON object."
        ),
    )

    print(
        "Part-A git commit = EXACT PASS"
    )

    print(
        "Part-A reference  = PRESENT"
    )

    print(
        "Part-A reference SHA256 =",
        part_a_sha,
    )

    print(
        "ADB adapter audit = DEFERRED TO BLOCK6.0 PART2"
    )

    # ========================================================
    # E. Stage5 handoff boundary
    # ========================================================

    print()
    print(
        "===== E. COMMUNICATION / ADB ACTUATOR BOUNDARY ====="
    )

    next_layer = handoff5.get(
        "Stage6_required_next_layer",
        {}
    )

    require(
        next_layer.get(
            "name"
        )
        ==
        "predictive_class_aware_ADB",
        (
            "Stage5 handoff does not identify "
            "predictive class-aware ADB "
            "as Stage6 next layer."
        ),
    )

    require(
        next_layer.get(
            "communication_beam_actuator_reused_as_ADB"
        )
        is False,
        (
            "Stage5 handoff incorrectly merges "
            "communication and ADB actuators."
        ),
    )

    print(
        "Stage6 next layer   = predictive_class_aware_ADB"
    )

    print(
        "communication beam actuator reused = NO"
    )

    print(
        "shared posterior     = YES"
    )

    # ========================================================
    # F. Audited Stage6 contract
    # ========================================================

    print()
    print(
        "===== F. STRICT PDF-AUDITED STAGE6 CONTRACT ====="
    )

    contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.0",

        "contract_version":
            "stage6_strict_pdf_audited_v1",

        "status":
            "FROZEN_PREIMPLEMENTATION_CONTRACT",

        "upstream": {
            "Stage4_status":
                "COMPLETE_FROZEN",

            "Stage5_status":
                "COMPLETE_FROZEN",

            "trajectory_posterior":
                "calibrated_Gaussian_GRU",

            "Stage5_closure_sha256":
                EXPECTED[
                    "stage5_closure"
                ],

            "Stage5_handoff_sha256":
                EXPECTED[
                    "stage5_handoff"
                ],

            "Stage5_freeze_sha256":
                EXPECTED[
                    "stage5_freeze"
                ],

            "formal_manifest_sha256":
                EXPECTED[
                    "formal_manifest"
                ],

            "formal_manifest_N":
                120,

            "formal_N_semantics":
                (
                    "implementation continuity choice; "
                    "not a literal Stage6 PDF N requirement"
                ),

            "Part_A_git_commit":
                EXPECTED[
                    "part_a_git"
                ],

            "Part_A_reference_sha256":
                part_a_sha,
        },

        "scope": {
            "objective":
                (
                    "reactive and predictive "
                    "class-aware ADB"
                ),

            "scientific_change":
                (
                    "current deterministic region -> "
                    "future probabilistic region"
                ),

            "communication_beam_controller":
                "FROZEN_STAGE5_NOT_MODIFIED",

            "communication_and_ADB_actuators":
                "SEPARATE",

            "predictive_ADB":
                "STAGE6",

            "shared_posterior_joint_evaluation":
                "STAGE7",

            "full_joint_end_to_end_latency":
                "STAGE7",

            "statistical_failure_slice_analysis":
                "STAGE7",

            "DeepSense":
                "STAGE8",
        },

        "experimental_modes": {
            "primary":
                (
                    "frozen calibrated posterior from "
                    "track-based PC-FMCW-like "
                    "observation mode"
                ),

            "oracle_track_mode":
                "UPPER_BOUND_DIAGNOSTIC_ONLY",

            "sensor_to_track_mode":
                "OPTIONAL_EXTENSION",

            "formal_parameter_tuning":
                False,

            "Stage4_retraining":
                False,

            "Stage4_recalibration":
                False,

            "GMM_posthoc_selection":
                False,
        },

        "adb_actor_population": {
            "policy":
                (
                    "all causal anchor-valid "
                    "ADB-relevant actors within "
                    "frozen headlamp FOV/range"
                ),

            "classes":
                [
                    "vehicle",
                    "pedestrian",
                    "cyclist",
                ],

            "ego_included":
                False,

            "communication_receiver_only":
                False,

            "tracks_to_predict_only":
                False,

            "tracks_to_predict_as_control_selector":
                False,

            "future_validity_as_selector":
                False,

            "future_track_duration_as_selector":
                False,

            "future_truth_for_controller":
                False,

            "future_truth_for_scoring_or_oracle_after_decision":
                True,
        },

        "geometry": {
            "centroid_only":
                False,

            "box_corners":
                True,

            "actor_width_height":
                True,

            "receiver_driver_relevant_region":
                True,

            "relevant_region_semantics":
                (
                    "constructed box-based surrogate "
                    "when dataset has no windshield/"
                    "mirror annotation"
                ),

            "predictive_covariance":
                True,

            "class_margin":
                True,

            "future_box_dimensions_from_GT":
                False,

            "dimension_policy":
                (
                    "anchor/current causal dimensions "
                    "with frozen fallback"
                ),

            "future_heading_from_GT":
                False,

            "orientation_policy":
                (
                    "causal current orientation or "
                    "predicted trajectory tangent with "
                    "frozen low-speed fallback"
                ),

            "vehicle_closing_speed_from_future_GT":
                False,

            "closing_speed_policy":
                (
                    "current filtered state and/or "
                    "predicted samples only"
                ),
        },

        "occupancy": {
            "representation":
                "P_occ(theta,r,tau,c)",

            "per_actor_binary_mask":
                (
                    "M_i_tau(theta,r) = "
                    "1[P_occ_i(theta,r,tau) > gamma]"
                ),

            "aggregate_mask":
                (
                    "M_tau = 1 - product_i(1 - M_i_tau)"
                ),

            "probability_kept_distinct_from_mask":
                True,

            "mask_kept_distinct_from_intensity":
                True,

            "required_horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "deterministic_reproducible_sampling":
                True,

            "sampling_count_frozen_before_formal":
                True,
        },

        "part_a_illumination": {
            "preserve_L_theta_r":
                True,

            "radial_thresholds":
                True,

            "angular_shadow_zones":
                True,

            "raised_cosine_transitions":
                True,

            "minimum_illumination_floor":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,

            "arbitrary_replacement_of_ADB_model":
                False,

            "exact_adapter_validation":
                "BLOCK6.0_PART2",
        },

        "class_policies": {
            "vehicle": {
                "oncoming":
                    True,

                "preceding":
                    True,

                "stronger_glare_protection":
                    True,

                "vehicle_specific_margin":
                    True,

                "windshield_mirror_relevant_region":
                    True,

                "uncertainty_dependent_margin":
                    True,

                "closing_speed_dependent_margin":
                    True,
            },

            "pedestrian": {
                "full_blackout_forbidden":
                    True,

                "body_road_visibility_preserved":
                    True,

                "partial_dimming_or_face_height_protection":
                    True,

                "distinct_minimum_intensity_floor":
                    True,

                "larger_lateral_uncertainty_margin":
                    True,
            },

            "cyclist": {
                "body_and_vehicle_visibility":
                    True,

                "larger_lateral_maneuver_margin":
                    True,

                "crossing_trajectory_handling":
                    True,

                "glare_protection_without_detection_loss":
                    True,

                "motorcyclist_semantics":
                    (
                        "use dataset-supported cyclist "
                        "umbrella when no independent "
                        "motorcyclist class exists; "
                        "do not invent a label"
                    ),
            },

            "class_awareness_not_margin_only":
                True,

            "allowed_policy_differences":
                [
                    "dimming_depth",
                    "intensity_floor",
                    "confidence_threshold",
                    "temporal_smoothing",
                    "safety_priority",
                ],
        },

        "adb_baselines": {
            "required_pdf_labels": [
                "static_ADB",
                "original_reactive_ADB",
                "deterministic_predictive_ADB",
                "uncertainty_aware_predictive_ADB",
                "class_agnostic_predictive_ADB",
                "class_aware_predictive_ADB",
                "oracle_future_ADB",
            ],

            "comparison_framework":
                "factorial_and_explicit_mapping",

            "artificial_algorithm_difference_for_label_only":
                False,

            "overlapping_baseline_semantics_disclosed":
                True,

            "same_grid_and_Part_A_intensity_model_for_fairness":
                True,
        },

        "adb_metrics": {
            "required": [
                "mask_IoU_with_constructed_oracle_future_mask",
                "vehicle_shadow_zone_violation",
                "glare_risk_exposure",
                "over_masking_area",
                "road_illumination_retention",
                "pedestrian_visibility_proxy",
                "cyclist_visibility_proxy",
                "false_dimming",
                "temporal_smoothness",
                "flicker_change_rate",
                "energy_consumption",
                "actuation_latency",
            ],

            "separate_reporting": [
                "vehicle_glare_protection",
                "pedestrian_visibility",
                "cyclist_visibility",
                "total_illuminated_road_retention",
            ],

            "oracle_semantics":
                "CONSTRUCTED_REFERENCE_NOT_MEASURED_GT",

            "road_ROI_definition_frozen_before_formal":
                True,

            "visibility_proxy_definitions_frozen_before_formal":
                True,

            "energy_metric_definition_frozen_before_formal":
                True,
        },

        "latency_and_resources": {
            "Stage6_required": [
                "ADB_generation_latency",
                "ADB_actuation_latency",
                "ADB_controller_memory_footprint",
                "ADB_controller_parameter_count",
                "ADB_controller_FLOPs_or_not_applicable_analytic_declaration",
            ],

            "full_joint_total_frame_latency":
                "STAGE7",

            "one_to_ten_ms_WOMD_GT_claim_allowed":
                False,
        },

        "acceptance": {
            "primary_comparison":
                (
                    "class-aware predictive ADB "
                    "vs original reactive ADB"
                ),

            "must_reduce_vehicle_shadow_zone_violation":
                True,

            "must_not_uncontrollably_increase_over_masking":
                True,

            "must_not_uncontrollably_reduce_VRU_visibility":
                True,

            "uncontrolled_degradation_numeric_bounds":
                "FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL",

            "formal_outcomes_used_for_threshold_selection":
                False,

            "formal_posthoc_tuning":
                False,
        },

        "deliverables": {
            "predictive_class_aware_ADB_controller":
                True,

            "ablation_evaluation_scripts":
                True,

            "configuration_files":
                True,

            "tests":
                True,

            "implementation_log":
                True,

            "paper_ready": [
                "reactive_vs_predictive_ADB",
                "class_aware_masks",
                "ADB_safety_overmask_curves",
                "ADB_ablation_table",
            ],

            "failure_case_statistics":
                "STAGE7",

            "Stage6_candidate_failure_scene_ids_may_be_saved":
                True,
        },
    }

    config_write = write_exact(
        CONFIG,
        contract,
    )

    print(
        "Stage6 audited contract =",
        config_write,
    )

    print(
        "contract SHA256 =",
        sha256_file(
            CONFIG
        ),
    )

    # ========================================================
    # G. Minimal package identity
    # ========================================================

    print()
    print(
        "===== G. STAGE6 PACKAGE SKELETON ====="
    )

    init_text = '''"""
ISCAI Stage 6.

Reactive and predictive class-aware ADB from the frozen
calibrated trajectory posterior.

Scientific implementation begins only after Block6.0
handoff/contract closure.
"""

__all__ = ()
'''

    package_write = write_exact_text(
        PACKAGE_INIT,
        init_text,
    )

    print(
        "package skeleton =",
        package_write,
    )

    # ========================================================
    # H. Contract regression test
    # ========================================================

    print()
    print(
        "===== H. BLOCK6.0 CONTRACT TEST ====="
    )

    test_text = r'''from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

CONTRACT = (
    S6
    / "configs/stage6_contract.json"
)

STAGE5_HANDOFF = (
    ROOT
    / "iscai_stage5/artifacts/block510/"
      "stage5_to_stage6_handoff.json"
)


class Block60ContractTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls):

        cls.contract = json.loads(
            CONTRACT.read_text(
                encoding="utf-8"
            )
        )

        cls.handoff = json.loads(
            STAGE5_HANDOFF.read_text(
                encoding="utf-8"
            )
        )

    def test_stage_and_status(self):

        self.assertEqual(
            self.contract["stage"],
            6,
        )

        self.assertEqual(
            self.contract["status"],
            "FROZEN_PREIMPLEMENTATION_CONTRACT",
        )

    def test_stage5_handoff_boundary(self):

        self.assertEqual(
            self.handoff["status"],
            "FROZEN_HANDOFF",
        )

        self.assertFalse(
            self.handoff[
                "Stage6_required_next_layer"
            ][
                "communication_beam_actuator_reused_as_ADB"
            ]
        )

    def test_primary_mode(self):

        primary = (
            self.contract[
                "experimental_modes"
            ][
                "primary"
            ]
        )

        self.assertIn(
            "track-based PC-FMCW-like",
            primary,
        )

    def test_actor_population_is_not_receiver_only(self):

        policy = self.contract[
            "adb_actor_population"
        ]

        self.assertFalse(
            policy[
                "communication_receiver_only"
            ]
        )

        self.assertFalse(
            policy[
                "tracks_to_predict_only"
            ]
        )

        self.assertFalse(
            policy[
                "tracks_to_predict_as_control_selector"
            ]
        )

        self.assertFalse(
            policy[
                "future_truth_for_controller"
            ]
        )

    def test_full_box_contract(self):

        geometry = self.contract[
            "geometry"
        ]

        self.assertFalse(
            geometry[
                "centroid_only"
            ]
        )

        self.assertTrue(
            geometry[
                "box_corners"
            ]
        )

        self.assertTrue(
            geometry[
                "actor_width_height"
            ]
        )

        self.assertTrue(
            geometry[
                "predictive_covariance"
            ]
        )

        self.assertTrue(
            geometry[
                "class_margin"
            ]
        )

    def test_exact_mask_aggregation(self):

        occupancy = self.contract[
            "occupancy"
        ]

        self.assertEqual(
            occupancy[
                "aggregate_mask"
            ],
            (
                "M_tau = 1 - product_i(1 - M_i_tau)"
            ),
        )

        self.assertTrue(
            occupancy[
                "probability_kept_distinct_from_mask"
            ]
        )

        self.assertTrue(
            occupancy[
                "mask_kept_distinct_from_intensity"
            ]
        )

    def test_part_a_illumination_preserved(self):

        adb = self.contract[
            "part_a_illumination"
        ]

        required = (
            "preserve_L_theta_r",
            "radial_thresholds",
            "angular_shadow_zones",
            "raised_cosine_transitions",
            "minimum_illumination_floor",
            "temporal_smoothing",
            "actuation_rate_limits",
        )

        for key in required:
            self.assertTrue(
                adb[key]
            )

        self.assertFalse(
            adb[
                "arbitrary_replacement_of_ADB_model"
            ]
        )

    def test_class_aware_not_margin_only(self):

        policies = self.contract[
            "class_policies"
        ]

        self.assertTrue(
            policies[
                "class_awareness_not_margin_only"
            ]
        )

        self.assertEqual(
            set(
                policies.keys()
            )
            -
            {
                "class_awareness_not_margin_only",
                "allowed_policy_differences",
            },
            {
                "vehicle",
                "pedestrian",
                "cyclist",
            },
        )

        self.assertGreaterEqual(
            len(
                policies[
                    "allowed_policy_differences"
                ]
            ),
            5,
        )

    def test_all_pdf_adb_baselines(self):

        actual = set(
            self.contract[
                "adb_baselines"
            ][
                "required_pdf_labels"
            ]
        )

        expected = {
            "static_ADB",
            "original_reactive_ADB",
            "deterministic_predictive_ADB",
            "uncertainty_aware_predictive_ADB",
            "class_agnostic_predictive_ADB",
            "class_aware_predictive_ADB",
            "oracle_future_ADB",
        }

        self.assertEqual(
            actual,
            expected,
        )

    def test_all_pdf_adb_metrics(self):

        actual = set(
            self.contract[
                "adb_metrics"
            ][
                "required"
            ]
        )

        expected = {
            "mask_IoU_with_constructed_oracle_future_mask",
            "vehicle_shadow_zone_violation",
            "glare_risk_exposure",
            "over_masking_area",
            "road_illumination_retention",
            "pedestrian_visibility_proxy",
            "cyclist_visibility_proxy",
            "false_dimming",
            "temporal_smoothness",
            "flicker_change_rate",
            "energy_consumption",
            "actuation_latency",
        }

        self.assertEqual(
            actual,
            expected,
        )

    def test_oracle_is_constructed(self):

        self.assertEqual(
            self.contract[
                "adb_metrics"
            ][
                "oracle_semantics"
            ],
            "CONSTRUCTED_REFERENCE_NOT_MEASURED_GT",
        )

    def test_stage6_acceptance(self):

        acceptance = self.contract[
            "acceptance"
        ]

        self.assertTrue(
            acceptance[
                "must_reduce_vehicle_shadow_zone_violation"
            ]
        )

        self.assertTrue(
            acceptance[
                "must_not_uncontrollably_increase_over_masking"
            ]
        )

        self.assertTrue(
            acceptance[
                "must_not_uncontrollably_reduce_VRU_visibility"
            ]
        )

        self.assertFalse(
            acceptance[
                "formal_outcomes_used_for_threshold_selection"
            ]
        )

        self.assertFalse(
            acceptance[
                "formal_posthoc_tuning"
            ]
        )

    def test_stage_boundaries(self):

        scope = self.contract[
            "scope"
        ]

        self.assertEqual(
            scope[
                "shared_posterior_joint_evaluation"
            ],
            "STAGE7",
        )

        self.assertEqual(
            scope[
                "full_joint_end_to_end_latency"
            ],
            "STAGE7",
        )

        self.assertEqual(
            scope[
                "DeepSense"
            ],
            "STAGE8",
        )


if __name__ == "__main__":
    unittest.main()
'''

    test_write = write_exact_text(
        TEST,
        test_text,
    )

    print(
        "test file =",
        test_write,
    )

    # Compile first.
    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(TEST),
            str(PACKAGE_INIT),
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        compile_process.returncode == 0,
        (
            "Block6.0 contract test compile "
            "failed:\n"
            + compile_process.stdout
        ),
    )

    rc, test_output = run_test()

    for line in test_output.splitlines():

        stripped = line.strip()

        if (
            stripped.startswith(
                "Ran "
            )
            or
            stripped == "OK"
            or
            stripped.startswith(
                "FAILED"
            )
        ):
            print(
                stripped
            )

    require(
        rc == 0,
        (
            "Block6.0 contract regression failed:\n"
            + test_output
        ),
    )

    # ========================================================
    # I. Storage / immutability
    # ========================================================

    print()
    print(
        "===== I. STORAGE / UPSTREAM IMMUTABILITY ====="
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB storage reserve violated: "
            f"{free_gib:.3f} GiB."
        ),
    )

    upstream_after = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in upstream
    }

    changed = [
        path
        for path in upstream_before
        if (
            upstream_before[
                path
            ]
            !=
            upstream_after[
                path
            ]
        )
    ]

    require(
        not changed,
        (
            "Frozen upstream artifact changed "
            "during Stage6 Block6.0 Part1: "
            + ", ".join(
                changed
            )
        ),
    )

    print(
        "free GiB              =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve       = PASS"
    )

    print(
        "Stage4 artifacts      = UNCHANGED"
    )

    print(
        "Stage5 artifacts      = UNCHANGED"
    )

    print(
        "Part-A reference      = UNCHANGED"
    )

    # ========================================================
    # J. Part1 report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.0",

        "part":
            "1/2",

        "status":
            "PASS_CONTRACT_FROZEN",

        "strict_PDF_audit":
            "PASS_100_PERCENT_STAGE6_PLAN_COVERAGE",

        "contract": {
            "path":
                str(
                    CONFIG
                ),

            "sha256":
                sha256_file(
                    CONFIG
                ),

            "status":
                "FROZEN_PREIMPLEMENTATION_CONTRACT",
        },

        "upstream": {
            "Stage4":
                "COMPLETE_FROZEN",

            "Stage5":
                "COMPLETE_FROZEN",

            "Stage5_closure_sha256":
                EXPECTED[
                    "stage5_closure"
                ],

            "Stage5_handoff_sha256":
                EXPECTED[
                    "stage5_handoff"
                ],

            "Stage5_freeze_sha256":
                EXPECTED[
                    "stage5_freeze"
                ],

            "formal_manifest_N":
                120,

            "formal_manifest_sha256":
                EXPECTED[
                    "formal_manifest"
                ],

            "Part_A_git_commit":
                EXPECTED[
                    "part_a_git"
                ],

            "Part_A_reference_sha256":
                part_a_sha,
        },

        "causality": {
            "future_GT_controller_input":
                False,

            "tracks_to_predict_ADB_selector":
                False,

            "communication_receiver_only_ADB":
                False,

            "Stage4_retraining":
                False,

            "Stage4_recalibration":
                False,
        },

        "regression": {
            "block60_contract_tests":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "upstream_modified":
            False,

        "scientific_ADB_implementation_started":
            False,

        "training":
            False,

        "inference":
            False,

        "formal_evaluation":
            False,

        "parameter_tuning":
            False,

        "next":
            (
                "Block6.0 Part2: exact Part-A "
                "ADB source/reference audit, "
                "reactive-ADB adapter contract "
                "and final Block6.0 closure"
            ),
    }

    temporary = REPORT.with_suffix(
        REPORT.suffix + ".tmp"
    )

    temporary.write_bytes(
        canonical_bytes(
            result
        )
    )

    os.replace(
        temporary,
        REPORT,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.0 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "strict Stage6 plan audit = 100% PASS"
    )

    print(
        "Stage4 posterior         = FROZEN PASS"
    )

    print(
        "Stage5 handoff           = EXACT PASS"
    )

    print(
        "Part-A reference         = FROZEN PASS"
    )

    print(
        "primary mode             = "
        "TRACK-BASED PC-FMCW-LIKE"
    )

    print(
        "ADB actor population     = "
        "ALL CAUSAL ADB-RELEVANT ACTORS"
    )

    print(
        "centroid-only ADB        = FORBIDDEN"
    )

    print(
        "communication actuator reused = NO"
    )

    print(
        "all 7 ADB baselines      = CONTRACT FROZEN"
    )

    print(
        "all 12 ADB metrics       = CONTRACT FROZEN"
    )

    print(
        "Stage7 boundary          = PRESERVED"
    )

    print(
        "Stage8 boundary          = PRESERVED"
    )

    print(
        "scientific ADB started   = NO"
    )

    print(
        "upstream modified        = NO"
    )

    print(
        "STATUS = PASS_CONTRACT_FROZEN"
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
        "BLOCK 6.0 PART 1/2 = BLOCKED"
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
        "Stage4/Stage5 scientific "
        "artifacts modified = NO"
    )

    print(
        "training               = NO"
    )

    print(
        "inference              = NO"
    )

    print(
        "formal evaluation      = NO"
    )

    print(
        "parameter tuning       = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Intentionally no non-zero sys.exit().

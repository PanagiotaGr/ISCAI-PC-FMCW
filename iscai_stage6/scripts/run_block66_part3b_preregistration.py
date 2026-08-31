from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

CONFIG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

REPORT = (
    S6
    / "reports/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part3b_preregistration_freeze_manifest.json"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3b_to_part3c_handoff.json"
)

PART3A3_REPORT = (
    S6
    / "reports/"
      "block66_part3a3_operator_capability_correction.json"
)

PART3A3_HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3a3_to_part3b_handoff.json"
)

PART_A_REACTIVE = (
    S6
    / "src/iscai_stage6/adb/"
      "part_a_reactive.py"
)

ILLUMINATION = (
    S6
    / "src/iscai_stage6/adb/"
      "illumination.py"
)

PREDICTIVE_MASK = (
    S6
    / "src/iscai_stage6/adb/"
      "predictive_mask.py"
)

GEOMETRY = (
    S6
    / "src/iscai_stage6/adb/"
      "geometry.py"
)

PROB_OCC = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_occupancy.py"
)

PART2C2A_REPORT = (
    S6
    / "reports/"
      "block66_part2c2a_probabilistic_kernel_binding_audit.json"
)

POCC_REPORT = (
    S6
    / "reports/"
      "block66_part2c2b_eligible_n8192_pocc.json"
)

POCC_FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_pocc_freeze_manifest.json"
)

ELIGIBILITY_FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligibility_freeze_manifest.json"
)

# IMPORTANT REPAIR:
# These are JSONL artifacts and therefore are bound directly.
POCC_MANIFEST = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

DEVELOPMENT_COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)


EXPECTED = {
    "part3a3_report":
        (
            "d76474b09f88b6809e30f4a59615feb3"
            "d0e5687b93e0b7083816028d40704fe3"
        ),

    "part3a3_handoff":
        (
            "0067239bf65d32d7bd40fdd33c02b74a"
            "ff509265e8a86070a3d2a0ed404cb1ee"
        ),

    "part_a_reactive":
        (
            "19f80f325aebc03b5257ca3c0408d1dd"
            "ea1ac3a8ff4ad875914224584e2ef173"
        ),

    "illumination":
        (
            "daeb96029ff6e1720d17cd20d3e8f001"
            "55c566eeecec3b3726f4adcc8b05d832"
        ),

    "predictive_mask":
        (
            "239611ddb1cc21b9ddd46962eb561e58e"
            "0d1474be58878f9c79c256ba1940575"
        ),

    "geometry":
        (
            "f6e6cc0dd362338eeeaae9d8e2594743"
            "4a3e7c64b350f0baf81a79a46159f94e"
        ),

    "probabilistic_occupancy":
        (
            "d4206761fc70495563d53a6ab58aefe6"
            "db5d8bd1b2d09736bfc5bbc6b6f50e3e"
        ),

    "part2c2a_report":
        (
            "4c79d9ba8084aa3cfbfed03945bf98970"
            "3769feb5303827844413c2ee71ac0dc"
        ),

    "pocc_report":
        (
            "41257fbc43bcd6e0a01925c0f96648f3"
            "cb67850785e8f39d9787547ce1511348"
        ),

    "pocc_freeze":
        (
            "b0df9df17c6008f9bdf3a993d08b103d"
            "c694607be90abe3307bc8d4877f9f6a7"
        ),

    "pocc_manifest":
        (
            "c361ed640b91850831d7e7177d89733f"
            "edd3e171f5f3111bcd93698c74a4a300"
        ),

    "eligibility_freeze":
        (
            "2d9dcd668b6b07824793a4773430edf9"
            "8ce4e96311f7cf756b2947520468efc0"
        ),

    "development_cohort":
        (
            "57440e3d976e7aeff44811b5eeb075dc"
            "56e0e0e070a3832ec72d354253b998c9"
        ),

    "mc_numeric_config":
        (
            "993c4248a902e7dff3a4383ac343e7222"
            "cc43ef9b9372ed2efd0ee08d6d20a73"
        ),
}


class ControlledBlock(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise ControlledBlock(str(message))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode("utf-8")


def write_once_exact(path: Path, payload) -> str:
    data = canonical_bytes(payload)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes() == data,
            (
                "Existing preregistration artifact differs "
                f"and will not be overwritten: {path}"
            ),
        )

    else:
        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_bytes(data)
        temporary.replace(path)

    return sha256_file(path)


def exact_seal(
    path: Path,
    expected: str,
    label: str,
):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = sha256_file(path)

    require(
        actual == expected,
        (
            f"{label} SHA changed: "
            f"{actual}"
        ),
    )

    print(
        f"{label:35s} = EXACT PASS"
    )

    return {
        "path":
            str(path),

        "sha256":
            actual,
    }


def locate_small_json_by_sha(
    expected_sha: str,
):
    """
    Read-only search ONLY for the one still-unbound
    small JSON numeric freeze.

    JSONL artifacts are NOT routed through this function.
    """

    roots = (
        S6 / "configs",
        S6 / "reports",
        S6 / "artifacts",
    )

    matches = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*.json"):
            try:
                if (
                    "cache"
                    in
                    str(path).lower()
                ):
                    continue

                if (
                    path.stat().st_size
                    >
                    20 * 1024 * 1024
                ):
                    continue

                if (
                    sha256_file(path)
                    ==
                    expected_sha
                ):
                    matches.append(path)

            except OSError:
                continue

    unique = []

    for path in matches:
        if path not in unique:
            unique.append(path)

    require(
        len(unique) >= 1,
        (
            "Could not locate frozen JSON "
            f"with SHA {expected_sha}."
        ),
    )

    # Exact-content duplicates are scientifically identical.
    return sorted(
        unique,
        key=lambda item: str(item),
    )[0]


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    return {
        "returncode":
            int(process.returncode),

        "tests":
            (
                int(match.group(1))
                if match
                else None
            ),

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[-35:]
            ),
    }


try:
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3B — "
        "CLASS-AWARE POLICY PREREGISTRATION"
    )
    print("=" * 78)

    # ========================================================
    # A. Authoritative upstream seal
    # ========================================================

    print()
    print(
        "===== A. AUTHORITATIVE UPSTREAM SEAL ====="
    )

    frozen_inputs = {}

    frozen_inputs[
        "Part3A3_report"
    ] = exact_seal(
        PART3A3_REPORT,
        EXPECTED["part3a3_report"],
        "Part3A3 capability report",
    )

    frozen_inputs[
        "Part3A3_handoff"
    ] = exact_seal(
        PART3A3_HANDOFF,
        EXPECTED["part3a3_handoff"],
        "Part3A3 -> Part3B handoff",
    )

    frozen_inputs[
        "part_a_reactive_source"
    ] = exact_seal(
        PART_A_REACTIVE,
        EXPECTED["part_a_reactive"],
        "part_a_reactive.py",
    )

    frozen_inputs[
        "illumination_source"
    ] = exact_seal(
        ILLUMINATION,
        EXPECTED["illumination"],
        "illumination.py",
    )

    frozen_inputs[
        "predictive_mask_source"
    ] = exact_seal(
        PREDICTIVE_MASK,
        EXPECTED["predictive_mask"],
        "predictive_mask.py",
    )

    frozen_inputs[
        "geometry_source"
    ] = exact_seal(
        GEOMETRY,
        EXPECTED["geometry"],
        "geometry.py",
    )

    frozen_inputs[
        "probabilistic_occupancy_source"
    ] = exact_seal(
        PROB_OCC,
        EXPECTED["probabilistic_occupancy"],
        "probabilistic_occupancy.py",
    )

    # ========================================================
    # B. Exact development P_occ / cohort seal
    # ========================================================

    print()
    print(
        "===== B. EXACT DEVELOPMENT P_OCC / COHORT SEAL ====="
    )

    frozen_inputs[
        "part2c2a_report"
    ] = exact_seal(
        PART2C2A_REPORT,
        EXPECTED["part2c2a_report"],
        "Part2C/2A kernel audit",
    )

    frozen_inputs[
        "pocc_report"
    ] = exact_seal(
        POCC_REPORT,
        EXPECTED["pocc_report"],
        "Part2C/2B P_occ report",
    )

    frozen_inputs[
        "pocc_freeze"
    ] = exact_seal(
        POCC_FREEZE,
        EXPECTED["pocc_freeze"],
        "Part2C/2B P_occ freeze",
    )

    frozen_inputs[
        "eligibility_freeze"
    ] = exact_seal(
        ELIGIBILITY_FREEZE,
        EXPECTED["eligibility_freeze"],
        "headlamp eligibility freeze",
    )

    print()
    print(
        "----- exact JSONL seals -----"
    )

    frozen_inputs[
        "pocc_manifest"
    ] = exact_seal(
        POCC_MANIFEST,
        EXPECTED["pocc_manifest"],
        "development P_occ manifest",
    )

    frozen_inputs[
        "development_cohort"
    ] = exact_seal(
        DEVELOPMENT_COHORT,
        EXPECTED["development_cohort"],
        "development cohort N120",
    )

    mc_numeric_path = (
        locate_small_json_by_sha(
            EXPECTED[
                "mc_numeric_config"
            ]
        )
    )

    frozen_inputs[
        "mc_numeric_config"
    ] = exact_seal(
        mc_numeric_path,
        EXPECTED["mc_numeric_config"],
        "N8192/grid numeric freeze",
    )

    print(
        "numeric freeze path =",
        mc_numeric_path,
    )

    # ========================================================
    # C. Runtime API boundary smoke
    # ========================================================

    print()
    print(
        "===== C. RUNTIME API BOUNDARY SMOKE ====="
    )

    from iscai_stage6.adb.geometry import (
        FractionalBoxRegion,
    )

    from iscai_stage6.adb.illumination import (
        ReactiveShadowRegion,
        part_a_radial_profile,
    )

    from iscai_stage6.adb.predictive_mask import (
        threshold_actor_occupancy,
    )

    print(
        "FractionalBoxRegion =",
        inspect.signature(
            FractionalBoxRegion
        ),
    )

    print(
        "ReactiveShadowRegion =",
        inspect.signature(
            ReactiveShadowRegion
        ),
    )

    print(
        "part_a_radial_profile =",
        inspect.signature(
            part_a_radial_profile
        ),
    )

    print(
        "threshold_actor_occupancy =",
        inspect.signature(
            threshold_actor_occupancy
        ),
    )

    oncoming_surrogate = FractionalBoxRegion(
        x_bounds=(
            0.0,
            0.5,
        ),
        y_bounds=(
            -0.5,
            0.5,
        ),
        z_bounds=(
            0.0,
            0.5,
        ),
    )

    preceding_surrogate = FractionalBoxRegion(
        x_bounds=(
            -0.5,
            0.0,
        ),
        y_bounds=(
            -0.5,
            0.5,
        ),
        z_bounds=(
            0.0,
            0.5,
        ),
    )

    require(
        oncoming_surrogate is not None,
        "Oncoming surrogate construction failed.",
    )

    require(
        preceding_surrogate is not None,
        "Preceding surrogate construction failed.",
    )

    print(
        "oncoming front-upper surrogate = VALID"
    )

    print(
        "preceding rear-upper surrogate = VALID"
    )

    print(
        "surrogate physical claim = NONE"
    )

    # ========================================================
    # D. Candidate-universe invariants
    # ========================================================

    print()
    print(
        "===== D. CANDIDATE-UNIVERSE INVARIANTS ====="
    )

    N_MC = 8192

    gamma_k_min = 0
    gamma_k_max = 8191

    base_margin_j = tuple(
        range(
            0,
            9,
        )
    )

    uncertainty_j = tuple(
        range(
            1,
            7,
        )
    )

    vehicle_closing_j = tuple(
        range(
            1,
            7,
        )
    )

    cyclist_lateral_j = tuple(
        range(
            1,
            9,
        )
    )

    floor_j = tuple(
        range(
            0,
            21,
        )
    )

    vru_floor_j = tuple(
        range(
            1,
            21,
        )
    )

    smoothing_ms = (
        50,
        100,
        200,
        400,
    )

    rate_per_s = (
        1,
        2,
        4,
        8,
        16,
    )

    gamma_candidate_count = (
        gamma_k_max
        -
        gamma_k_min
        +
        1
    )

    vehicle_margin_count = (
        len(
            base_margin_j
        )
        *
        len(
            uncertainty_j
        )
        *
        len(
            vehicle_closing_j
        )
    )

    pedestrian_margin_count = (
        len(
            base_margin_j
        )
        *
        len(
            uncertainty_j
        )
    )

    cyclist_margin_count = (
        len(
            base_margin_j
        )
        *
        len(
            uncertainty_j
        )
        *
        len(
            cyclist_lateral_j
        )
    )

    floor_triplet_count = 0

    for fv in floor_j:
        for fp in vru_floor_j:
            for fc in vru_floor_j:
                if (
                    fv <= fp
                    and
                    fv <= fc
                ):
                    floor_triplet_count += 1

    rate_pair_count = 0

    for rho_dim in rate_per_s:
        for rho_bright in rate_per_s:
            if (
                rho_dim
                >=
                rho_bright
            ):
                rate_pair_count += 1

    temporal_rate_count = (
        len(
            smoothing_ms
        )
        *
        rate_pair_count
    )

    require(
        gamma_candidate_count
        ==
        8192,
        "Gamma candidate count changed.",
    )

    require(
        vehicle_margin_count
        ==
        324,
        "Vehicle margin candidate count changed.",
    )

    require(
        pedestrian_margin_count
        ==
        54,
        "Pedestrian margin candidate count changed.",
    )

    require(
        cyclist_margin_count
        ==
        432,
        "Cyclist margin candidate count changed.",
    )

    require(
        floor_triplet_count > 0,
        "Floor candidate space is empty.",
    )

    require(
        temporal_rate_count > 0,
        "Temporal/rate candidate space is empty.",
    )

    print(
        "gamma candidates per class =",
        gamma_candidate_count,
    )

    print(
        "vehicle margin candidates  =",
        vehicle_margin_count,
    )

    print(
        "pedestrian margin candidates=",
        pedestrian_margin_count,
    )

    print(
        "cyclist margin candidates  =",
        cyclist_margin_count,
    )

    print(
        "floor triplets             =",
        floor_triplet_count,
    )

    print(
        "temporal/rate combinations =",
        temporal_rate_count,
    )

    print(
        "numeric policy selected    = NO"
    )

    # ========================================================
    # E. Full preregistration payload
    # ========================================================

    print()
    print(
        "===== E. WRITE CLASS-AWARE POLICY PREREGISTRATION ====="
    )

    prereg = {
        "stage":
            6,

        "block":
            "6.6-Part3B",

        "status":
            (
                "PREREGISTERED_PARAMETERIZATION_"
                "AND_DEVELOPMENT_SELECTION_RULE"
            ),

        "purpose":
            (
                "Freeze the complete class-aware predictive "
                "ADB policy family, candidate-generation "
                "rules, evaluator formulas, class overlap "
                "semantics, temporal smoothing semantics, "
                "actuation-rate-limit semantics and staged "
                "development-only selection procedure before "
                "any class-policy sweep."
            ),

        "scientific_boundary": {
            "source_modified":
                False,

            "new_model_inference":
                False,

            "new_MC_sampling":
                False,

            "policy_sweep_executed":
                False,

            "numeric_policy_selected":
                False,

            "formal_outcomes_read":
                False,

            "formal_tuning":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "source_of_numeric_candidate_grids":
            (
                "Stage6 implementation preregistration; "
                "not numerical values supplied by the PDF "
                "or SAE J3069."
            ),

        "frozen_inputs":
            frozen_inputs,

        "development_population": {
            "partition":
                "Stage4 DEVELOPMENT only",

            "cohort_scenes":
                120,

            "cohort_selection":
                (
                    "frozen outcome-independent "
                    "Block6.6 development cohort"
                ),

            "formal_population_used":
                False,

            "calibration_partition_used_for_policy_tuning":
                False,

            "eligible_predictive_actors":
                877,

            "class_counts": {
                "TYPE_VEHICLE":
                    719,

                "TYPE_PEDESTRIAN":
                    110,

                "TYPE_CYCLIST":
                    48,
            },

            "eligible_reactive_fallback_actors":
                247,
        },

        "frozen_probabilistic_runtime": {
            "N_MC":
                8192,

            "seed":
                20260821,

            "horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "grid": {
                "theta_count":
                    501,

                "range_count":
                    301,

                "theta_min_deg":
                    -25.0,

                "theta_max_deg":
                    25.0,

                "theta_step_deg":
                    0.1,

                "range_min_m":
                    0.0,

                "range_max_m":
                    150.0,

                "range_step_m":
                    0.5,
            },

            "P_occ_storage":
                (
                    "canonical uint32 occupancy counts; "
                    "P_occ = count / 8192 exactly"
                ),

            "threshold_semantics":
                "STRICT_GREATER_THAN",
        },

        "preserved_PartA_semantics": {
            "generic_reactive_safety_margin":
                "preserved, not replaced",

            "target_width_m":
                1.9,

            "generic_lateral_margin_m":
                1.0,

            "generic_radial_margin_m":
                4.0,

            "transition_length_m":
                20.0,

            "raised_cosine":
                "exact frozen Part-A runtime function",

            "multi_actor_original_composition":
                "minimum intensity",

            "class_aware_predictive_margin":
                (
                    "additional expansion above the "
                    "frozen Part-A generic safety semantics"
                ),

            "original_reactive_baseline_modified":
                False,
        },

        "controller_geometry": {
            "primary_predictive_occupancy":
                "full projected 3D actor box",

            "centroid_only":
                False,

            "box_corners":
                True,

            "actor_width_height":
                True,

            "predictive_covariance":
                True,

            "actuator_axes":
                [
                    "theta",
                    "ground_range",
                ],

            "vertical_actuator_DOF":
                False,

            "face_height_pixel_control_claim":
                False,
        },

        "vehicle_relevant_region_surrogates": {
            "physical_ground_truth_claim":
                False,

            "semantics":
                (
                    "geometry-based heading-dependent "
                    "surrogate only; WOMD does not provide "
                    "actual windshield, mirror or driver "
                    "locations"
                ),

            "local_box_fraction_convention":
                (
                    "FractionalBoxRegion bounds relative "
                    "to actor-local length/width/height axes"
                ),

            "oncoming_front_upper_surrogate": {
                "x_bounds":
                    [
                        0.0,
                        0.5,
                    ],

                "y_bounds":
                    [
                        -0.5,
                        0.5,
                    ],

                "z_bounds":
                    [
                        0.0,
                        0.5,
                    ],
            },

            "preceding_rear_upper_surrogate": {
                "x_bounds":
                    [
                        -0.5,
                        0.0,
                    ],

                "y_bounds":
                    [
                        -0.5,
                        0.5,
                    ],

                "z_bounds":
                    [
                        0.0,
                        0.5,
                    ],
            },

            "controller_side_heading_source":
                (
                    "predicted/samplewise heading only; "
                    "no future GT heading"
                ),

            "evaluator_side_heading_source":
                (
                    "future GT heading permitted only after "
                    "controller decision for constructed "
                    "oracle/safety-reference evaluation"
                ),

            "oncoming_rule":
                (
                    "cos(actor_heading_H0_rad) < 0"
                ),

            "preceding_rule":
                (
                    "cos(actor_heading_H0_rad) >= 0"
                ),

            "required_before_parameter_sweep":
                (
                    "materialize/project these surrogate "
                    "regions from the same frozen N8192 "
                    "trajectory samples for vehicle "
                    "compliance evidence; full-box P_occ "
                    "remains the primary controller occupancy"
                ),
        },

        "class_threshold_parameterization": {
            "equation":
                (
                    "M_i,tau(theta,r) = "
                    "1[count_i,tau(theta,r) > k_c]"
                ),

            "equivalent_gamma":
                "gamma_c = k_c / 8192",

            "strict_comparison":
                ">",

            "candidate_rule": {
                "representation":
                    "exact integer occupancy quantum",

                "k_min":
                    0,

                "k_max":
                    8191,

                "step":
                    1,

                "candidates_per_class":
                    8192,
            },

            "independent_classes":
                [
                    "TYPE_VEHICLE",
                    "TYPE_PEDESTRIAN",
                    "TYPE_CYCLIST",
                ],

            "no_float_grid_search":
                True,
        },

        "additional_class_margin_parameterization": {
            "coordinate_frame":
                "H0 theta/r actuator plane",

            "meaning":
                (
                    "additional predictive dilation; "
                    "does not replace Part-A generic margin"
                ),

            "predicted_absolute_center":
                (
                    "p_tau_xy = causal_anchor_xy "
                    "+ calibrated_Gaussian_mean_displacement_xy"
                ),

            "predicted_range":
                (
                    "r_tau = hypot(p_tau_x, p_tau_y)"
                ),

            "lateral_LOS_unit":
                (
                    "ell_tau = "
                    "[-p_tau_y/r_tau, p_tau_x/r_tau]"
                ),

            "lateral_predictive_sigma_m":
                (
                    "sigma_lat = "
                    "sqrt(ell_tau^T Sigma_xy,tau ell_tau)"
                ),

            "closing_speed_mps":
                (
                    "v_close = "
                    "max(0, (r_current-r_tau)/tau)"
                ),

            "cyclist_lateral_rate_mps":
                (
                    "v_lat = "
                    "abs(ell_current^T mean_displacement_xy_tau)"
                    "/tau"
                ),

            "vehicle_formula":
                (
                    "m_extra = b_vehicle "
                    "+ u_vehicle*sigma_lat "
                    "+ q_vehicle*tau*v_close"
                ),

            "pedestrian_formula":
                (
                    "m_extra = b_pedestrian "
                    "+ u_pedestrian*sigma_lat"
                ),

            "cyclist_formula":
                (
                    "m_extra = b_cyclist "
                    "+ u_cyclist*sigma_lat "
                    "+ q_cyclist*tau*v_lat"
                ),

            "angular_expansion":
                (
                    "delta_theta = atan2("
                    "m_extra, max(r_tau, PartA_EPS))"
                ),

            "grid_dilation":
                (
                    "ceil(delta_theta / frozen_theta_step); "
                    "clip at actuator theta boundaries"
                ),

            "candidate_rules": {
                "base_margin_m": {
                    "representation":
                        "j/4 metres",

                    "j_min":
                        0,

                    "j_max":
                        8,

                    "candidate_count":
                        9,
                },

                "uncertainty_multiplier": {
                    "representation":
                        "j/2",

                    "j_min":
                        1,

                    "j_max":
                        6,

                    "candidate_count":
                        6,

                    "zero_permitted":
                        False,
                },

                "vehicle_closing_multiplier": {
                    "representation":
                        "j/20",

                    "j_min":
                        1,

                    "j_max":
                        6,

                    "candidate_count":
                        6,

                    "zero_permitted":
                        False,
                },

                "cyclist_lateral_multiplier": {
                    "representation":
                        "j/4",

                    "j_min":
                        1,

                    "j_max":
                        8,

                    "candidate_count":
                        8,

                    "zero_permitted":
                        False,
                },
            },

            "candidate_counts": {
                "TYPE_VEHICLE":
                    vehicle_margin_count,

                "TYPE_PEDESTRIAN":
                    pedestrian_margin_count,

                "TYPE_CYCLIST":
                    cyclist_margin_count,
            },
        },

        "predictive_mask_aggregation": {
            "per_actor":
                (
                    "strict class threshold followed by "
                    "class-aware angular dilation"
                ),

            "multi_actor_binary_equation":
                (
                    "M_tau = 1 - product_i(1-M_i,tau)"
                ),

            "binary_OR_equivalence":
                True,
        },

        "continuous_illumination_conversion": {
            "purpose":
                (
                    "convert predictive occupancy/masks back "
                    "into the preserved Part-A L(theta,r) "
                    "rather than replacing ADB with a binary map"
                ),

            "per_active_theta_column": {
                "r_far":
                    (
                        "maximum active frozen range-cell "
                        "center after threshold/dilation"
                    ),

                "r_shadow_end_m":
                    "r_far + 4.0",

                "r_transition_end_m":
                    "r_shadow_end_m + 20.0",

                "radial_profile":
                    (
                        "exact part_a_radial_profile with "
                        "class-specific intensity_floor"
                    ),
            },

            "inactive_theta_column":
                1.0,

            "raised_cosine_preserved":
                True,

            "range_profile_reimplemented":
                False,
        },

        "class_intensity_floor_parameterization": {
            "representation":
                "f_c = j_c / 20",

            "vehicle": {
                "j_min":
                    0,

                "j_max":
                    20,
            },

            "pedestrian": {
                "j_min":
                    1,

                "j_max":
                    20,

                "zero_permitted":
                    False,
            },

            "cyclist": {
                "j_min":
                    1,

                "j_max":
                    20,

                "zero_permitted":
                    False,
            },

            "structural_constraints": [
                "f_vehicle <= f_pedestrian",
                "f_vehicle <= f_cyclist",
                "f_pedestrian > 0",
                "f_cyclist > 0",
            ],

            "pedestrian_full_blackout_permitted":
                False,

            "cyclist_full_blackout_permitted":
                False,

            "candidate_triplets":
                floor_triplet_count,
        },

        "reactive_fallback_inside_class_aware_controller": {
            "geometry":
                (
                    "exact frozen original-reactive "
                    "Part-A geometry"
                ),

            "radial_profile":
                (
                    "exact frozen raised-cosine profile"
                ),

            "class_floor_applied":
                True,

            "scope":
                (
                    "class-aware controller only; "
                    "does not modify the original-reactive "
                    "baseline"
                ),

            "future_information":
                False,
        },

        "class_overlap_policy": {
            "hard_priority_order": [
                (
                    "VRU minimum-visibility floor "
                    "constraint"
                ),
                (
                    "vehicle glare suppression"
                ),
                (
                    "cyclist glare/visibility objective"
                ),
                (
                    "pedestrian glare/visibility objective"
                ),
            ],

            "same_level_suppression":
                "minimum intensity",

            "pre_floor_guard_command":
                (
                    "I_min = minimum over all active "
                    "actor illumination maps"
                ),

            "VRU_floor_guard":
                (
                    "F_vru = max(active pedestrian floor, "
                    "active cyclist floor, 0)"
                ),

            "raw_class_aware_output":
                (
                    "I_raw = max(I_min, F_vru)"
                ),

            "reason":
                (
                    "vehicle glare protection remains strong "
                    "but may not force a pedestrian/cyclist "
                    "below its hard non-blackout floor"
                ),

            "deterministic":
                True,
        },

        "temporal_smoothing": {
            "domain":
                (
                    "planned actuation time sequence at "
                    "tau=[0.1,0.3,0.5,1.0] s"
                ),

            "initial_state":
                (
                    "current causal class-aware reactive "
                    "illumination generated from exact "
                    "Part-A geometry; no future information"
                ),

            "equation":
                (
                    "alpha_h = 1-exp(-delta_t_h/T); "
                    "I_smooth_h = alpha_h*I_raw_h "
                    "+ (1-alpha_h)*I_prev"
                ),

            "candidate_time_constants_ms":
                list(
                    smoothing_ms
                ),

            "zero_smoothing_candidate":
                False,

            "candidate_values_are_PDF_values":
                False,
        },

        "actuation_rate_limit": {
            "applied_after":
                "temporal smoothing",

            "normalized_intensity_units":
                "[0,1]",

            "equation":
                (
                    "I_rate = clip("
                    "I_smooth, "
                    "I_prev-rho_dim*delta_t, "
                    "I_prev+rho_bright*delta_t)"
                ),

            "candidate_rates_per_s":
                list(
                    rate_per_s
                ),

            "constraint":
                (
                    "rho_dim >= rho_bright; "
                    "dimming is never slower than brightening"
                ),

            "finite_rate_required":
                True,

            "rate_pair_count":
                rate_pair_count,

            "safety_projection_after_rate_limit":
                (
                    "re-apply active VRU hard floor; "
                    "record every rate-limit safety override"
                ),

            "safety_override_metric":
                (
                    "fraction of actuator cells/time steps "
                    "where hard VRU floor overrides I_rate"
                ),
        },

        "constructed_oracle_reference": {
            "ground_truth_type":
                (
                    "constructed evaluator-only reference; "
                    "NOT measured ADB ground truth"
                ),

            "controller_access":
                False,

            "evaluation_access_timing":
                (
                    "future GT may be read only after "
                    "controller output is fixed"
                ),

            "identity":
                (
                    "truth track identity permitted "
                    "evaluator-side only"
                ),

            "primary_actor_geometry":
                "future GT full 3D box",

            "vehicle_glare_reference":
                (
                    "heading-dependent geometry-based "
                    "front-upper/rear-upper surrogate"
                ),

            "oracle_shadow_semantics":
                (
                    "future GT projected box -> exact "
                    "Part-A +4 m shadow end and +20 m "
                    "raised-cosine transition"
                ),
        },

        "road_ROI": {
            "metric_name":
                "road_illumination_retention",

            "reference_type":
                (
                    "constructed map-based road "
                    "illumination proxy"
                ),

            "not_measured_photometry":
                True,

            "construction":
                (
                    "map each frozen theta/range ground-cell "
                    "center to H0 XY; include cell when "
                    "its XY distance to a causal/static WOMD "
                    "lane-center polyline is <= 1.75 m"
                ),

            "lane_corridor_half_width_m":
                1.75,

            "future_dynamic_map_used":
                False,

            "empty_ROI":
                "NA and excluded from macro average",
        },

        "ADB_metric_formulas": {
            "binary_dim_support":
                "D_tau = [I_final_tau < 1.0]",

            "oracle_dim_support":
                (
                    "O_tau = union of constructed "
                    "future actor shadow supports"
                ),

            "mask_IoU": {
                "formula":
                    "|D∩O| / |D∪O|",

                "empty_union":
                    1.0,
            },

            "vehicle_shadow_zone_violation": {
                "formula":
                    "|O_vehicle \\ D| / |O_vehicle|",

                "direction":
                    "lower_is_better",

                "no_vehicle_cells":
                    "NA",
            },

            "glare_risk_exposure": {
                "formula":
                    (
                        "mean final normalized intensity "
                        "over vehicle constructed "
                        "front-upper/rear-upper surrogate "
                        "reference cells"
                    ),

                "direction":
                    "lower_is_better",
            },

            "over_masking_area": {
                "formula":
                    "|D \\ O| / |full actuator grid|",

                "direction":
                    "lower_is_better",
            },

            "road_illumination_retention": {
                "formula":
                    "mean(I_final over frozen road ROI)",

                "direction":
                    "higher_is_better",
            },

            "pedestrian_visibility_proxy": {
                "formula":
                    (
                        "mean(I_final over future GT "
                        "pedestrian projected full-box "
                        "footprint cells)"
                    ),

                "direction":
                    "higher_is_better",
            },

            "cyclist_visibility_proxy": {
                "formula":
                    (
                        "mean(I_final over future GT "
                        "cyclist projected full-box "
                        "footprint cells)"
                    ),

                "direction":
                    "higher_is_better",
            },

            "false_dimming": {
                "formula":
                    (
                        "|D \\ O| / |D|; "
                        "0 when |D|=0"
                    ),

                "direction":
                    "lower_is_better",
            },

            "temporal_smoothness": {
                "reported_metric":
                    "mean_absolute_change_rate",

                "formula":
                    (
                        "mean_h("
                        "mean_grid(|I_h-I_prev|)"
                        "/delta_t_h)"
                    ),

                "direction":
                    "lower_is_better",
            },

            "flicker_change_rate": {
                "formula":
                    (
                        "mean_h("
                        "count(I_h != I_prev)"
                        "/grid_cell_count)"
                    ),

                "direction":
                    "lower_is_better",
            },

            "energy_consumption": {
                "reported_semantics":
                    (
                        "normalized emitted-light "
                        "energy proxy, not joules"
                    ),

                "formula":
                    (
                        "mean over horizon/grid of "
                        "I_final"
                    ),

                "direction":
                    "reported_not_primary_safety_target",
            },

            "actuation_latency": {
                "formula":
                    (
                        "wall-clock runtime from loaded "
                        "P_occ/current geometry to final "
                        "four-horizon illumination schedule"
                    ),

                "units":
                    "seconds",

                "used_for_policy_selection":
                    False,
            },

            "rate_limit_safety_override_fraction": {
                "formula":
                    (
                        "override cells / total "
                        "actuator-cell transitions"
                    ),

                "direction":
                    "lower_is_better",
            },
        },

        "development_selection_protocol": {
            "partition":
                "development only",

            "formal_outcomes_allowed":
                False,

            "staged_search":
                True,

            "full_cartesian_search":
                False,

            "parameter_revisiting_after_later_stage":
                False,

            "stage_1_class_gamma": {
                "vary":
                    "k_c only",

                "held_fixed":
                    (
                        "no additional class margin; "
                        "floor not yet selected; "
                        "no temporal/rate tuning"
                    ),

                "objective":
                    (
                        "maximize macro mean per-actor/"
                        "per-horizon full-box mask IoU "
                        "against constructed development "
                        "future reference"
                    ),

                "tie_break": {
                    "TYPE_VEHICLE":
                        (
                            "smaller k first "
                            "(more protective threshold)"
                        ),

                    "TYPE_PEDESTRIAN":
                        (
                            "larger k first "
                            "(less unnecessary dimming)"
                        ),

                    "TYPE_CYCLIST":
                        (
                            "larger k first "
                            "(less unnecessary dimming)"
                        ),
                },
            },

            "stage_2_class_margin": {
                "vary":
                    (
                        "class base / uncertainty / "
                        "vehicle-closing / cyclist-lateral "
                        "coefficients"
                    ),

                "gamma":
                    "frozen Stage-1 development selection",

                "classwise_primary_objective":
                    (
                        "maximize macro actor-horizon "
                        "IoU after dilation"
                    ),

                "tie_break_order": [
                    "lower over-masking area",
                    "smaller base margin",
                    "smaller uncertainty multiplier",
                    (
                        "smaller closing/lateral "
                        "multiplier"
                    ),
                    "canonical numeric tuple",
                ],

                "mandatory_nonzero_uncertainty_response":
                    True,

                "vehicle_closing_response_required":
                    True,

                "cyclist_lateral_response_required":
                    True,
            },

            "stage_3_class_floors": {
                "vary":
                    "class floor triplets only",

                "gamma_and_margin":
                    (
                        "fixed from stages 1 and 2"
                    ),

                "risk_vector_all_components_unit_interval":
                    True,

                "normalized_risk_vector": [
                    "vehicle_shadow_zone_violation",
                    "glare_risk_exposure",
                    "1-pedestrian_visibility_proxy",
                    "1-cyclist_visibility_proxy",
                    "1-road_illumination_retention",
                ],

                "lexicographic_selection": [
                    (
                        "minimize maximum element of "
                        "normalized risk vector"
                    ),
                    (
                        "minimize arithmetic mean of "
                        "normalized risk vector"
                    ),
                    "minimize over_masking_area",
                    "minimize false_dimming",
                    (
                        "canonical floor tuple "
                        "(f_vehicle,f_pedestrian,f_cyclist)"
                    ),
                ],

                "arbitrary_weighted_sum":
                    False,
            },

            "stage_4_temporal_and_rate": {
                "vary":
                    (
                        "smoothing time constant and "
                        "finite dim/bright rate pair"
                    ),

                "all_class_parameters":
                    (
                        "fixed from stages 1-3"
                    ),

                "lexicographic_selection": [
                    (
                        "minimize maximum normalized "
                        "safety-risk component"
                    ),
                    (
                        "minimize mean normalized "
                        "safety-risk component"
                    ),
                    (
                        "minimize temporal "
                        "mean_absolute_change_rate"
                    ),
                    "minimize flicker_change_rate",
                    (
                        "minimize rate-limit "
                        "safety-override fraction"
                    ),
                    "minimize over_masking_area",
                    "canonical temporal/rate tuple",
                ],
            },

            "exact_tie_handling":
                (
                    "no tolerance-based tie; compare "
                    "stored deterministic float64 metrics "
                    "then canonical parameter tuple"
                ),

            "acceptance_thresholds":
                (
                    "NOT selected in Part3B; after the "
                    "development-only winning policy is "
                    "frozen, a separate pre-formal gate "
                    "must freeze formal acceptance/"
                    "non-inferiority thresholds before "
                    "any formal outcome is read"
                ),
        },

        "seven_ADB_baseline_factorization": {
            "1_static_ADB": {
                "future_prediction":
                    False,

                "uncertainty":
                    False,

                "class_policy":
                    False,

                "semantics":
                    "normalized all-on static map",
            },

            "2_original_reactive_ADB": {
                "future_prediction":
                    False,

                "uncertainty":
                    False,

                "class_policy":
                    False,

                "semantics":
                    (
                        "exact frozen Block6.2 "
                        "Part-A reactive implementation"
                    ),

                "must_remain_unchanged":
                    True,
            },

            "3_deterministic_predictive_ADB": {
                "future_prediction":
                    "shared calibrated Gaussian mean",

                "uncertainty":
                    False,

                "class_policy":
                    False,

                "geometry":
                    "future full boxes",

                "purpose":
                    "isolate uncertainty contribution",
            },

            "4_uncertainty_aware_predictive_ADB": {
                "future_prediction":
                    "frozen calibrated P_occ",

                "uncertainty":
                    True,

                "class_policy":
                    False,

                "class_labels_collapsed":
                    True,

                "margin":
                    (
                        "single shared sigma_lat-responsive "
                        "margin family; no class terms"
                    ),

                "purpose":
                    (
                        "isolate uncertainty-responsive "
                        "control from class awareness"
                    ),
            },

            "5_class_agnostic_predictive_ADB": {
                "future_prediction":
                    "frozen calibrated P_occ",

                "uncertainty_in_P_occ":
                    True,

                "additional_uncertainty_margin":
                    False,

                "class_policy":
                    False,

                "gamma":
                    0.5,

                "class_specific_floor":
                    False,

                "purpose":
                    (
                        "preserve frozen Block6.5 "
                        "class-agnostic boundary"
                    ),
            },

            "6_class_aware_predictive_ADB": {
                "future_prediction":
                    "frozen calibrated P_occ",

                "uncertainty":
                    True,

                "class_policy":
                    True,

                "class_gamma":
                    True,

                "class_margin":
                    True,

                "class_floor":
                    True,

                "class_priority":
                    True,

                "temporal_smoothing":
                    True,

                "actuation_rate_limit":
                    True,
            },

            "7_oracle_future_ADB": {
                "future_prediction":
                    "future GT evaluator-only",

                "controller_input_in_nonoracle_methods":
                    False,

                "purpose":
                    (
                        "upper bound / constructed oracle "
                        "future reference"
                    ),
            },

            "shared_actuator_dynamics_after_freeze":
                (
                    "deterministic, uncertainty-aware, "
                    "class-agnostic, class-aware and "
                    "oracle predictive maps use the same "
                    "frozen raised-cosine and actuator "
                    "dynamics where applicable; original "
                    "reactive baseline itself remains exact "
                    "and unmodified"
                ),

            "fake_duplicate_baselines_permitted":
                False,
        },

        "causality": {
            "future_GT_controller_input":
                False,

            "future_GT_heading_controller_input":
                False,

            "truth_track_index_controller_input":
                False,

            "perfect_track_id_join":
                False,

            "tracks_to_predict_selection":
                False,

            "objects_of_interest_selection":
                False,

            "future_track_duration_selection":
                False,

            "formal_outcome_tuning":
                False,

            "future_GT_evaluator_after_decision":
                True,
        },

        "required_runtime_components_to_implement_next": [
            "class_aware_policy_composition",
            "class_aware_predictive_margin_operator",
            "class_overlap_priority_rule",
            "temporal_smoothing_operator",
            "actuation_rate_limit_operator",
        ],

        "next_implementation_gate": {
            "block":
                "6.6-Part3C",

            "requirements_before_any_sweep": [
                (
                    "implement the five preregistered "
                    "runtime operators"
                ),
                (
                    "unit-test exact strict-gamma "
                    "semantics"
                ),
                (
                    "unit-test margin expansion in "
                    "theta cells"
                ),
                (
                    "unit-test class floors and "
                    "no-VRU-blackout invariant"
                ),
                (
                    "unit-test deterministic overlap "
                    "priority"
                ),
                (
                    "unit-test temporal smoothing"
                ),
                (
                    "unit-test finite actuation-rate "
                    "limiter"
                ),
                (
                    "unit-test safety-floor override "
                    "accounting"
                ),
                (
                    "materialize vehicle geometry-based "
                    "relevant-region evidence from the "
                    "same frozen N8192 samples"
                ),
                (
                    "prove no vertical/pixel actuation "
                    "claim is introduced"
                ),
                (
                    "fresh Stage6 regression"
                ),
            ],

            "development_parameter_sweep":
                "BLOCKED_UNTIL_PART3C_RUNTIME_PASS",
        },
    }

    config_sha = write_once_exact(
        CONFIG,
        prereg,
    )

    print(
        "preregistration =",
        CONFIG,
    )

    print(
        "preregistration SHA256 =",
        config_sha,
    )

    # ========================================================
    # F. Fresh regression
    # ========================================================

    print()
    print(
        "===== F. FRESH STAGE6 REGRESSION ====="
    )

    regression = run_regression()

    print(
        regression[
            "tail"
        ]
    )

    require(
        regression[
            "returncode"
        ]
        ==
        0,
        "Fresh Stage6 regression failed.",
    )

    require(
        regression[
            "tests"
        ]
        ==
        145,
        (
            "Expected 145 Stage6 tests "
            f"before new runtime code, got "
            f"{regression['tests']}."
        ),
    )

    print(
        "fresh Stage6 regression = PASS | 145"
    )

    # ========================================================
    # G. Report
    # ========================================================

    print()
    print(
        "===== G. PART3B REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part3B",

        "status":
            (
                "PASS_CLASS_AWARE_POLICY_"
                "PREREGISTRATION_FROZEN"
            ),

        "preregistration": {
            "path":
                str(CONFIG),

            "sha256":
                config_sha,
        },

        "scope": {
            "parameterization_frozen":
                True,

            "candidate_generation_rules_frozen":
                True,

            "metric_formulas_frozen":
                True,

            "road_ROI_definition_frozen":
                True,

            "class_overlap_semantics_frozen":
                True,

            "temporal_smoothing_semantics_frozen":
                True,

            "actuation_rate_limit_semantics_frozen":
                True,

            "staged_development_selection_rule_frozen":
                True,

            "seven_baseline_factorization_frozen":
                True,

            "numeric_policy_selected":
                False,

            "policy_sweep_executed":
                False,

            "formal_outcomes_read":
                False,

            "scientific_source_modified":
                False,
        },

        "development_evidence": {
            "scenes":
                120,

            "eligible_predictive_actors":
                877,

            "TYPE_VEHICLE":
                719,

            "TYPE_PEDESTRIAN":
                110,

            "TYPE_CYCLIST":
                48,

            "N_MC":
                8192,

            "seed":
                20260821,
        },

        "authoritative_capability_boundary": {
            "existing": [
                "PartA raised cosine",
                "PartA generic safety margin",
                "runtime intensity floor",
                "predictive strict gamma threshold",
                "full 3D geometry",
                "theta/r actuator",
            ],

            "new_required": [
                "class-aware composition",
                "class-aware predictive margin",
                "class overlap priority",
                "temporal smoothing",
                "actuation rate limiter",
            ],

            "vertical_actuator_DOF":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                145,
        },

        "next":
            (
                "BLOCK6.6_PART3C_IMPLEMENT_"
                "PREREGISTERED_CLASS_AWARE_RUNTIME"
            ),
    }

    report_sha = write_once_exact(
        REPORT,
        report_payload,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    # ========================================================
    # H. Freeze manifest
    # ========================================================

    print()
    print(
        "===== H. PART3B FREEZE MANIFEST ====="
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.6-Part3B",

        "status":
            "FROZEN_BEFORE_POLICY_SWEEP",

        "preregistration": {
            "path":
                str(CONFIG),

            "sha256":
                config_sha,
        },

        "report": {
            "path":
                str(REPORT),

            "sha256":
                report_sha,
        },

        "authoritative_upstream": {
            key:
                value
            for key, value
            in sorted(
                frozen_inputs.items()
            )
        },

        "prohibited_until_next_freeze": [
            "formal_outcome_read",
            "formal_tuning",
            "candidate-grid modification",
            "selection-objective modification",
            (
                "metric-formula modification "
                "after seeing sweep outcomes"
            ),
            (
                "road-ROI modification "
                "after seeing sweep outcomes"
            ),
            (
                "class-overlap-rule modification "
                "after seeing sweep outcomes"
            ),
        ],

        "scientific_source_modified":
            False,

        "numeric_policy_selected":
            False,
    }

    freeze_sha = write_once_exact(
        FREEZE,
        freeze_payload,
    )

    print(
        "freeze manifest =",
        FREEZE,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
    )

    # ========================================================
    # I. Part3C handoff
    # ========================================================

    print()
    print(
        "===== I. PART3C HANDOFF ====="
    )

    handoff_payload = {
        "stage":
            6,

        "from":
            "6.6-Part3B",

        "to":
            "6.6-Part3C",

        "status":
            (
                "PASS_READY_TO_IMPLEMENT_"
                "PREREGISTERED_CLASS_AWARE_RUNTIME"
            ),

        "preregistration": {
            "path":
                str(CONFIG),

            "sha256":
                config_sha,
        },

        "freeze_manifest": {
            "path":
                str(FREEZE),

            "sha256":
                freeze_sha,
        },

        "runtime_components_to_add": [
            "class_aware_policy_composition",
            "class_aware_predictive_margin_operator",
            "class_overlap_priority_rule",
            "temporal_smoothing_operator",
            "actuation_rate_limit_operator",
        ],

        "Part3C_is_allowed_to": [
            (
                "add runtime implementation matching "
                "the frozen Part3B semantics"
            ),
            "add unit/regression tests",
            (
                "materialize vehicle relevant-region "
                "N8192 evidence using the same frozen samples"
            ),
            (
                "run synthetic/operator smoke tests"
            ),
        ],

        "Part3C_is_not_allowed_to": [
            "run class-policy parameter sweep",
            "select gamma",
            "select class margin numerics",
            "select floors",
            "select smoothing time",
            "select rate limits",
            "read formal outcomes",
            "use future GT in controller",
            (
                "claim actual windshield/"
                "mirror locations"
            ),
            (
                "claim vertical/pixel ADB "
                "actuation"
            ),
        ],

        "PartA_continuity": {
            "generic_margin_preserved":
                True,

            "raised_cosine_preserved":
                True,

            "original_reactive_baseline_preserved":
                True,
        },

        "formal_outcomes_read":
            False,
    }

    handoff_sha = write_once_exact(
        HANDOFF,
        handoff_payload,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # J. Final
    # ========================================================

    print()
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3B — FINAL"
    )
    print("=" * 78)

    print(
        "STATUS = "
        "PASS_CLASS_AWARE_POLICY_PREREGISTRATION_FROZEN"
    )

    print(
        "development cohort           = 120 | FROZEN"
    )

    print(
        "development predictive actors= 877 | FROZEN"
    )

    print(
        "N_MC / seed                  = "
        "8192 / 20260821 | FROZEN"
    )

    print(
        "class gamma family           = PREREGISTERED"
    )

    print(
        "class predictive margins     = PREREGISTERED"
    )

    print(
        "class intensity floors       = PREREGISTERED"
    )

    print(
        "vehicle surrogate regions    = "
        "PREREGISTERED / GEOMETRY-BASED"
    )

    print(
        "class overlap priority       = PREREGISTERED"
    )

    print(
        "temporal smoothing           = PREREGISTERED"
    )

    print(
        "actuation-rate limiter       = PREREGISTERED"
    )

    print(
        "ADB metric formulas          = FROZEN BEFORE SWEEP"
    )

    print(
        "road ROI proxy               = FROZEN BEFORE SWEEP"
    )

    print(
        "seven baseline factorization = FROZEN"
    )

    print(
        "Part-A raised cosine         = PRESERVED"
    )

    print(
        "Part-A generic margin        = PRESERVED"
    )

    print(
        "original reactive baseline   = UNCHANGED"
    )

    print(
        "vertical actuator DOF        = NO"
    )

    print(
        "numeric policy selected      = NO"
    )

    print(
        "policy sweep executed        = NO"
    )

    print(
        "formal outcomes read         = NO"
    )

    print(
        "scientific source modified   = NO"
    )

    print(
        "Stage6 regression            = PASS | 145"
    )

    print(
        "preregistration SHA256       =",
        config_sha,
    )

    print(
        "report SHA256                =",
        report_sha,
    )

    print(
        "freeze SHA256                =",
        freeze_sha,
    )

    print(
        "handoff SHA256               =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.6 PART3C "
        "IMPLEMENT PREREGISTERED CLASS-AWARE RUNTIME"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 78)


except BaseException as exc:
    print()
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3B — CONTROLLED BLOCK"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=25
    )

    print()
    print(
        "scientific source modified = NO"
    )

    print(
        "numeric policy selected    = NO"
    )

    print(
        "policy sweep executed      = NO"
    )

    print(
        "formal outcomes read       = NO"
    )

    print(
        "terminal remains open      = YES"
    )

# Intentionally no sys.exit().

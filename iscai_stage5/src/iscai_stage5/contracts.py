from __future__ import annotations


# ============================================================
# Frozen Stage4 → Stage5 input contract
# ============================================================

STAGE4_DEFAULT_POSTERIOR = (
    "calibrated_Gaussian_GRU"
)

STAGE4_GMM_ROLE = (
    "frozen_multimodal_diagnostic_only"
)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

CALIBRATION_VARIANCE_SCALE_ALPHA_H = (
    1.2347064500315355,
    1.3451250295202921,
    1.354822057728461,
    1.2829700217319266,
)

MEASUREMENT_R_T_IS_MODEL_INPUT = True

PREDICTIVE_COVARIANCE_IS_DISTINCT = True

CALIBRATION_MODIFIES_MEASUREMENT_R_T = False

STAGE4_RETRAINING_ALLOWED = False

STAGE4_NORMALIZATION_REFIT_ALLOWED = False

STAGE4_RECALIBRATION_ALLOWED = False

STAGE4_POSTHOC_GMM_SELECTION_ALLOWED = False


# ============================================================
# Communication receiver contract
# ============================================================

#
# WOMD has no measured V2X/optical-connectivity label.
# Therefore "connected vehicle" is a declared constructed
# experimental eligibility assumption, never a dataset claim.
#
CONNECTED_VEHICLE_SEMANTICS = (
    "constructed_hypothetical_connected_vehicle"
)

PRIMARY_RECEIVER_POLICY = (
    "nearest_causal_vehicle_ahead"
)

PRIMARY_RECEIVER_POLICY_SEMANTICS = (
    "select_the_nearest_causally_available_vehicle_"
    "ahead_in_the_headlamp_frame_from_the_current_"
    "associated_actor_set"
)

PRIMARY_RECEIVER_REQUIRES_VEHICLE_CLASS = True

PRIMARY_RECEIVER_REQUIRES_CURRENT_CAUSAL_AVAILABILITY = True

PRIMARY_RECEIVER_REQUIRES_POSITIVE_FORWARD_H0 = True

TRACKS_TO_PREDICT_IS_RECEIVER_SELECTOR = False

FUTURE_TRUTH_IS_RECEIVER_SELECTOR = False

PERFECT_TRACK_ID_IS_CONTROL_FEATURE = False

NO_ELIGIBLE_RECEIVER_POLICY = (
    "explicit_no_link_target"
)


# ============================================================
# Receiver geometry contract
# ============================================================

RECEIVER_GEOMETRY_MODES = (
    "centroid_baseline",
    "known_receiver_offset",
    "uncertain_receiver_offset",
)

PRIMARY_RECEIVER_GEOMETRY_FOR_DOWNSTREAM = (
    "uncertain_receiver_offset"
)

RECEIVER_OFFSET_FRAME = (
    "receiver_vehicle_body_frame"
)

UNCERTAIN_RECEIVER_OFFSET_PROPAGATION = (
    "joint_with_trajectory_posterior"
)

FUTURE_RECEIVER_HEADING_SOURCE = (
    "trajectory_sample_tangent"
)

LOW_SPEED_HEADING_FALLBACK = (
    "last_causal_heading"
)

FUTURE_GROUND_TRUTH_HEADING_ALLOWED_IN_CONTROLLER = False


# ============================================================
# Angular posterior contract
# ============================================================

ANGULAR_POSTERIOR_OUTPUT = (
    "joint_receiver_range_azimuth_elevation_posterior"
)

PRIMARY_ANGULAR_PROPAGATION = (
    "deterministic_monte_carlo"
)

ANALYTIC_CROSSCHECK = (
    "Jacobian_Gaussian_known_offset_case"
)

MONTE_CARLO_SAMPLE_COUNT_STATUS = (
    "MUST_FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
)


# ============================================================
# Directional codebook contract
# ============================================================

CODEBOOK_SIZES = (
    16,
    32,
    64,
)

CODEBOOK_DECISION_CELLS = (
    "non_overlapping_probability_cells"
)

PHYSICAL_GAIN_LOBES_MAY_OVERLAP = True

BEAM_PROBABILITY_SEMANTICS = (
    "posterior_probability_mass_in_unique_decision_cell"
)

EXPECTED_BEAM_GAIN_SEMANTICS = (
    "posterior_integral_of_physical_beam_gain"
)

BEAM_PROBABILITY_AND_EXPECTED_GAIN_ARE_DISTINCT = True

EXACT_CODEBOOK_ANGULAR_SUPPORT_STATUS = (
    "MUST_FREEZE_FROM_PARTA_STAGE5_DEVELOPMENT_BEFORE_FORMAL"
)


# ============================================================
# Beam baselines
# ============================================================

BEAM_BASELINES = (
    "exhaustive_sweep",
    "previous_beam_persistence",
    "geometry_nearest_beam",
    "fixed_Top1",
    "fixed_Top3",
    "fixed_Top5",
    "oracle_beam_set",
)

ORACLE_IS_EVALUATOR_ONLY = True

ORACLE_ALLOWED_IN_CONTROLLER = False


# ============================================================
# Adaptive Top-K
# ============================================================

PROBABILITY_MASS_TARGETS = (
    0.90,
    0.95,
    0.975,
    0.99,
)

DEFAULT_PROBABILITY_MASS_TARGET = (
    0.95
)

ADAPTIVE_TOPK_RULE = (
    "smallest_K_whose_ranked_beam_probability_mass_"
    "is_at_least_the_requested_target"
)

SELECTED_POSTERIOR_MASS_REPORTED = True

EMPIRICAL_FUTURE_CONTAINMENT_REPORTED = True

POSTERIOR_MASS_AND_EMPIRICAL_CONTAINMENT_ARE_DISTINCT = True

FORMAL_COVERAGE_TOLERANCE_STATUS = (
    "MUST_FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
)

K_MAX_STATUS = (
    "MUST_FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
)

HYSTERESIS_STATUS = (
    "MUST_FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
)

LOCAL_SWEEP_STATUS = (
    "MUST_FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
)

REACQUISITION_STATUS = (
    "MUST_FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
)

LOSS_OF_LOCK_FINAL_FALLBACK = (
    "exhaustive_reacquisition"
)

BLOCKAGE_AWARE_TOPK = (
    "OPTIONAL_NOT_STAGE5_CLOSURE_REQUIREMENT"
)


# ============================================================
# Optical/DPSK communication evaluator contract
# ============================================================

OPTICAL_LINK_CHAIN = (
    "pointing_error",
    "optical_beam_gain",
    "received_power",
    "SNR",
    "DPSK_BER",
    "effective_rate",
)

PART_A_LINK_MODEL_POLICY = (
    "reuse_or_numerically_reproduce_frozen_PartA_"
    "communication_link_assumptions"
)

NEW_ARBITRARY_OPTICAL_MODEL_ALLOWED = False

PART_A_NUMERICAL_CROSSCHECK_REQUIRED = True

#
# The frozen Part-A notebook contains the physical
# communication/link basis through received power,
# SNR/DPSK BER and the raw data-rate constant.
#
# It does NOT contain a source-level Stage5
# overhead-aware effective-rate implementation.
#
# Therefore Stage5 must derive effective rate from
# the frozen Part-A link quantities plus explicit
# Stage5 beam-probing overhead.  The exact formula
# must be frozen on non-formal development evidence
# before the formal N=120 evaluation.
#
PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED = False

EFFECTIVE_RATE_PROVENANCE = (
    "Stage5_derived_from_frozen_PartA_link_"
    "quantities_raw_data_rate_and_beam_probing_overhead"
)

EFFECTIVE_RATE_FORMULA_STATUS = (
    "MUST_FREEZE_IN_BLOCK5.6_BEFORE_FORMAL"
)

BEAM_PROBING_OVERHEAD_MUST_AFFECT_EFFECTIVE_RATE = True


# ============================================================
# Temporal control / fallback / latency
# ============================================================

BEAM_CONTROLLER_MUST_IMPLEMENT = (
    "previous_beam_persistence",
    "hysteresis",
    "local_neighbour_sweep",
    "beam_switch_penalty",
    "widened_fallback",
    "loss_of_lock_reacquisition",
)

STAGE5_LATENCY_SCOPE = (
    "beam_controller_and_communication_link_latency"
)

FULL_JOINT_BEAM_ADB_END_TO_END_LATENCY_STAGE = (
    7
)

MILLISECOND_WOMD_GROUND_TRUTH_CLAIM_ALLOWED = False


# ============================================================
# Frozen formal evaluation
# ============================================================

FORMAL_POPULATION_N = (
    120
)

FORMAL_POPULATION_ROLE = (
    "evaluation_only"
)

FORMAL_TUNING_ALLOWED = False

FORMAL_MODEL_SELECTION_ALLOWED = False

FORMAL_RECALIBRATION_ALLOWED = False

FORMAL_RECEIVER_POLICY_SELECTION_ALLOWED = False

FORMAL_CODEBOOK_SELECTION_ALLOWED = False

FORMAL_THRESHOLD_SELECTION_ALLOWED = False


# ============================================================
# Stage5 acceptance
# ============================================================

STAGE5_ACCEPTANCE_REQUIREMENTS = (
    "adaptive_TopK_approximately_respects_requested_coverage",
    "adaptive_TopK_reduces_overhead_vs_fixed_or_exhaustive_probing",
)

APPROXIMATE_COVERAGE_NUMERICAL_TOLERANCE = (
    "TO_BE_PREREGISTERED_ON_DEVELOPMENT_BEFORE_FORMAL"
)

OVERHEAD_COMPARATORS = (
    "fixed_TopK",
    "exhaustive_sweep",
)

FORMAL_ACCEPTANCE_CANNOT_BE_REDEFINED_POSTHOC = True


# ============================================================
# Required formal metrics
# ============================================================

REQUIRED_FORMAL_BEAM_METRICS = (
    "Top1_hit_rate",
    "Top3_or_TopK_hit_rate",
    "average_selected_K",
    "selected_probability_mass",
    "empirical_receiver_coverage",
    "probing_overhead",
    "overhead_reduction",
    "beam_gain_loss",
    "received_power_loss",
    "SNR_loss",
    "DPSK_BER",
    "effective_rate",
    "effective_rate_loss",
    "outage_probability",
    "beam_switching_rate",
    "reacquisition_latency",
)

REQUIRED_FORMAL_SWEEPS = (
    "prediction_horizon",
    "codebook_size",
    "requested_probability_mass",
    "receiver_geometry_mode",
    "distance",
    "uncertainty",
)

REQUIRED_STAGE5_PAPER_ARTIFACTS = (
    "receiver_angular_posterior_visualization",
    "adaptive_beam_set_visualization",
    "reliability_overhead_curve",
    "outage_overhead_curve",
    "BER_curve",
    "effective_rate_curve",
    "Pareto_frontier",
)


# ============================================================
# Explicit Stage5 scope boundary
# ============================================================

STAGE5_OUT_OF_SCOPE = (
    "predictive_class_aware_ADB",
    "illumination_pixel_controller",
    "joint_beam_ADB_tradeoff_evaluation",
    "full_joint_beam_ADB_end_to_end_latency",
    "Stage7_failure_slice_analysis",
    "Stage7_statistical_confidence_intervals",
    "DeepSense_external_mmWave_validation",
)

NEXT_STAGE = (
    6
)


# ============================================================
# Development-only freeze fields
# ============================================================

DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL = (
    "receiver_eligibility_max_range",
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
)

DEVELOPMENT_SOURCE = (
    "non_formal_development_partition_only"
)


def contract_dict():
    return {
        "stage":
            5,

        "status":
            "STRUCTURAL_CONTRACT_FROZEN",

        "trajectory_posterior": {
            "family":
                STAGE4_DEFAULT_POSTERIOR,

            "GMM_role":
                STAGE4_GMM_ROLE,

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "calibration_variance_scale_alpha_h":
                list(
                    CALIBRATION_VARIANCE_SCALE_ALPHA_H
                ),

            "measurement_R_t_is_model_input":
                MEASUREMENT_R_T_IS_MODEL_INPUT,

            "predictive_covariance_is_distinct":
                PREDICTIVE_COVARIANCE_IS_DISTINCT,

            "calibration_modifies_measurement_R_t":
                CALIBRATION_MODIFIES_MEASUREMENT_R_T,

            "Stage4_retraining_allowed":
                STAGE4_RETRAINING_ALLOWED,

            "Stage4_normalization_refit_allowed":
                STAGE4_NORMALIZATION_REFIT_ALLOWED,

            "Stage4_recalibration_allowed":
                STAGE4_RECALIBRATION_ALLOWED,

            "Stage4_posthoc_GMM_selection_allowed":
                STAGE4_POSTHOC_GMM_SELECTION_ALLOWED,
        },

        "receiver_selection": {
            "connectivity_semantics":
                CONNECTED_VEHICLE_SEMANTICS,

            "primary_policy":
                PRIMARY_RECEIVER_POLICY,

            "policy_semantics":
                PRIMARY_RECEIVER_POLICY_SEMANTICS,

            "vehicle_class_required":
                PRIMARY_RECEIVER_REQUIRES_VEHICLE_CLASS,

            "current_causal_availability_required":
                PRIMARY_RECEIVER_REQUIRES_CURRENT_CAUSAL_AVAILABILITY,

            "positive_forward_H0_required":
                PRIMARY_RECEIVER_REQUIRES_POSITIVE_FORWARD_H0,

            "tracks_to_predict_selector":
                TRACKS_TO_PREDICT_IS_RECEIVER_SELECTOR,

            "future_truth_selector":
                FUTURE_TRUTH_IS_RECEIVER_SELECTOR,

            "perfect_track_ID_control_feature":
                PERFECT_TRACK_ID_IS_CONTROL_FEATURE,

            "no_eligible_receiver":
                NO_ELIGIBLE_RECEIVER_POLICY,
        },

        "receiver_geometry": {
            "modes":
                list(
                    RECEIVER_GEOMETRY_MODES
                ),

            "primary_downstream_mode":
                PRIMARY_RECEIVER_GEOMETRY_FOR_DOWNSTREAM,

            "offset_frame":
                RECEIVER_OFFSET_FRAME,

            "uncertain_offset_propagation":
                UNCERTAIN_RECEIVER_OFFSET_PROPAGATION,

            "future_heading_source":
                FUTURE_RECEIVER_HEADING_SOURCE,

            "low_speed_heading_fallback":
                LOW_SPEED_HEADING_FALLBACK,

            "future_GT_heading_controller":
                FUTURE_GROUND_TRUTH_HEADING_ALLOWED_IN_CONTROLLER,
        },

        "angular_posterior": {
            "output":
                ANGULAR_POSTERIOR_OUTPUT,

            "primary_propagation":
                PRIMARY_ANGULAR_PROPAGATION,

            "analytic_crosscheck":
                ANALYTIC_CROSSCHECK,

            "Monte_Carlo_sample_count":
                MONTE_CARLO_SAMPLE_COUNT_STATUS,
        },

        "codebook": {
            "sizes":
                list(
                    CODEBOOK_SIZES
                ),

            "decision_cells":
                CODEBOOK_DECISION_CELLS,

            "physical_gain_lobes_may_overlap":
                PHYSICAL_GAIN_LOBES_MAY_OVERLAP,

            "beam_probability":
                BEAM_PROBABILITY_SEMANTICS,

            "expected_beam_gain":
                EXPECTED_BEAM_GAIN_SEMANTICS,

            "probability_and_expected_gain_distinct":
                BEAM_PROBABILITY_AND_EXPECTED_GAIN_ARE_DISTINCT,

            "angular_support":
                EXACT_CODEBOOK_ANGULAR_SUPPORT_STATUS,
        },

        "beam_baselines":
            list(
                BEAM_BASELINES
            ),

        "adaptive_TopK": {
            "requested_probability_masses":
                list(
                    PROBABILITY_MASS_TARGETS
                ),

            "default_probability_mass":
                DEFAULT_PROBABILITY_MASS_TARGET,

            "rule":
                ADAPTIVE_TOPK_RULE,

            "selected_posterior_mass_reported":
                SELECTED_POSTERIOR_MASS_REPORTED,

            "empirical_future_containment_reported":
                EMPIRICAL_FUTURE_CONTAINMENT_REPORTED,

            "mass_and_containment_distinct":
                POSTERIOR_MASS_AND_EMPIRICAL_CONTAINMENT_ARE_DISTINCT,

            "formal_coverage_tolerance":
                FORMAL_COVERAGE_TOLERANCE_STATUS,

            "K_max":
                K_MAX_STATUS,

            "hysteresis":
                HYSTERESIS_STATUS,

            "local_sweep":
                LOCAL_SWEEP_STATUS,

            "reacquisition":
                REACQUISITION_STATUS,

            "loss_of_lock_fallback":
                LOSS_OF_LOCK_FINAL_FALLBACK,

            "blockage_aware":
                BLOCKAGE_AWARE_TOPK,
        },

        "optical_link": {
            "chain":
                list(
                    OPTICAL_LINK_CHAIN
                ),

            "PartA_policy":
                PART_A_LINK_MODEL_POLICY,

            "new_arbitrary_model_allowed":
                NEW_ARBITRARY_OPTICAL_MODEL_ALLOWED,

            "PartA_numerical_crosscheck_required":
                PART_A_NUMERICAL_CROSSCHECK_REQUIRED,

            "PartA_source_effective_rate_required":
                PART_A_SOURCE_EFFECTIVE_RATE_REQUIRED,

            "effective_rate_provenance":
                EFFECTIVE_RATE_PROVENANCE,

            "effective_rate_formula_status":
                EFFECTIVE_RATE_FORMULA_STATUS,

            "probing_overhead_affects_effective_rate":
                BEAM_PROBING_OVERHEAD_MUST_AFFECT_EFFECTIVE_RATE,
        },

        "temporal_controller": {
            "required_components":
                list(
                    BEAM_CONTROLLER_MUST_IMPLEMENT
                ),

            "Stage5_latency_scope":
                STAGE5_LATENCY_SCOPE,

            "full_joint_beam_ADB_latency_stage":
                FULL_JOINT_BEAM_ADB_END_TO_END_LATENCY_STAGE,

            "millisecond_WOMD_GT_claim_allowed":
                MILLISECOND_WOMD_GROUND_TRUTH_CLAIM_ALLOWED,
        },

        "formal": {
            "scenario_count":
                FORMAL_POPULATION_N,

            "role":
                FORMAL_POPULATION_ROLE,

            "tuning_allowed":
                FORMAL_TUNING_ALLOWED,

            "model_selection_allowed":
                FORMAL_MODEL_SELECTION_ALLOWED,

            "recalibration_allowed":
                FORMAL_RECALIBRATION_ALLOWED,

            "receiver_policy_selection_allowed":
                FORMAL_RECEIVER_POLICY_SELECTION_ALLOWED,

            "codebook_selection_allowed":
                FORMAL_CODEBOOK_SELECTION_ALLOWED,

            "threshold_selection_allowed":
                FORMAL_THRESHOLD_SELECTION_ALLOWED,
        },

        "acceptance": {
            "requirements":
                list(
                    STAGE5_ACCEPTANCE_REQUIREMENTS
                ),

            "coverage_tolerance":
                APPROXIMATE_COVERAGE_NUMERICAL_TOLERANCE,

            "overhead_comparators":
                list(
                    OVERHEAD_COMPARATORS
                ),

            "posthoc_redefinition_allowed":
                not
                FORMAL_ACCEPTANCE_CANNOT_BE_REDEFINED_POSTHOC,
        },

        "required_formal_beam_metrics":
            list(
                REQUIRED_FORMAL_BEAM_METRICS
            ),

        "required_formal_sweeps":
            list(
                REQUIRED_FORMAL_SWEEPS
            ),

        "required_paper_artifacts":
            list(
                REQUIRED_STAGE5_PAPER_ARTIFACTS
            ),

        "development_freeze": {
            "source":
                DEVELOPMENT_SOURCE,

            "fields":
                list(
                    DEVELOPMENT_FIELDS_TO_FREEZE_BEFORE_FORMAL
                ),

            "must_complete_before_formal":
                True,
        },

        "out_of_scope":
            list(
                STAGE5_OUT_OF_SCOPE
            ),

        "next_stage":
            NEXT_STAGE,
    }

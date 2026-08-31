from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ============================================================
# PDF-required ADB baseline labels
# ============================================================

class ADBBaseline(str, Enum):
    STATIC_ADB = (
        "static_ADB"
    )

    ORIGINAL_REACTIVE_ADB = (
        "original_reactive_ADB"
    )

    DETERMINISTIC_PREDICTIVE_ADB = (
        "deterministic_predictive_ADB"
    )

    UNCERTAINTY_AWARE_PREDICTIVE_ADB = (
        "uncertainty_aware_predictive_ADB"
    )

    CLASS_AGNOSTIC_PREDICTIVE_ADB = (
        "class_agnostic_predictive_ADB"
    )

    CLASS_AWARE_PREDICTIVE_ADB = (
        "class_aware_predictive_ADB"
    )

    ORACLE_FUTURE_ADB = (
        "oracle_future_ADB"
    )


# ============================================================
# PDF-required ADB metric labels
#
# Numerical formulas are deliberately NOT frozen here.
# They are frozen in Block6.8 before formal evaluation.
# ============================================================

class ADBMetric(str, Enum):
    MASK_IOU = (
        "mask_IoU_with_constructed_oracle_future_mask"
    )

    VEHICLE_SHADOW_VIOLATION = (
        "vehicle_shadow_zone_violation"
    )

    GLARE_RISK_EXPOSURE = (
        "glare_risk_exposure"
    )

    OVER_MASKING_AREA = (
        "over_masking_area"
    )

    ROAD_ILLUMINATION_RETENTION = (
        "road_illumination_retention"
    )

    PEDESTRIAN_VISIBILITY_PROXY = (
        "pedestrian_visibility_proxy"
    )

    CYCLIST_VISIBILITY_PROXY = (
        "cyclist_visibility_proxy"
    )

    FALSE_DIMMING = (
        "false_dimming"
    )

    TEMPORAL_SMOOTHNESS = (
        "temporal_smoothness"
    )

    FLICKER_CHANGE_RATE = (
        "flicker_change_rate"
    )

    ENERGY_CONSUMPTION = (
        "energy_consumption"
    )

    ACTUATION_LATENCY = (
        "actuation_latency"
    )


@dataclass(frozen=True)
class ADBBaselineSpec:
    """
    Frozen semantic configuration of one comparison label.

    `runtime_configuration_id` may intentionally be shared by
    two PDF labels when their semantics overlap.

    We never fabricate an algorithmic distinction merely to
    make labels appear different.
    """

    label: ADBBaseline

    predictive: bool

    future_mean: bool

    predictive_uncertainty: bool

    class_aware_policy: bool

    current_reactive_geometry: bool

    static_controller: bool

    evaluator_future_truth: bool

    future_truth_controller_input: bool

    constructed_oracle_only: bool

    runtime_configuration_id: str

    interpretation: str


@dataclass(frozen=True)
class ConstructedOracleContract:
    """
    Oracle reference contract.

    WOMD has no measured ADB command ground truth.

    The oracle is therefore a CONSTRUCTED evaluator reference
    generated from future actor geometry after the controller
    decision boundary.

    It must never become a causal feature.
    """

    name: str = (
        "constructed_future_ADB_reference"
    )

    source: str = (
        "future_actor_truth_evaluator_only"
    )

    measured_ADB_ground_truth: bool = False

    available_to_controller: bool = False

    available_for_parameter_tuning: bool = False

    available_for_scoring: bool = True

    same_grid_as_predictive_ADB: bool = True

    same_Part_A_intensity_model: bool = True


@dataclass(frozen=True)
class ADBMetricInterface:
    """
    Metric interface only.

    Block6.7 freezes WHICH metrics are mandatory and their
    causal/evaluator data boundary.

    Block6.8 freezes the exact numerical formulas, ROIs,
    thresholds and acceptance tolerances before formal use.
    """

    name: ADBMetric

    requires_constructed_oracle: bool

    requires_vehicle_truth_region: bool

    requires_pedestrian_truth_region: bool

    requires_cyclist_truth_region: bool

    requires_road_ROI: bool

    requires_temporal_sequence: bool

    requires_intensity_map: bool

    controller_input_metric: bool = False


def required_baseline_specs(
) -> tuple[
    ADBBaselineSpec,
    ...
]:
    """
    Exact seven-label PDF comparison matrix.

    Important:
    `uncertainty_aware_predictive_ADB` and
    `class_agnostic_predictive_ADB` may map to the same runtime
    configuration if the implementation has no scientifically
    meaningful additional axis separating them.

    That overlap is explicit rather than hidden.
    """

    shared_uncertainty_configuration = (
        "predictive_uncertainty_common_policy_v1"
    )

    return (
        ADBBaselineSpec(
            label=(
                ADBBaseline.STATIC_ADB
            ),

            predictive=False,
            future_mean=False,
            predictive_uncertainty=False,
            class_aware_policy=False,
            current_reactive_geometry=False,
            static_controller=True,
            evaluator_future_truth=False,
            future_truth_controller_input=False,
            constructed_oracle_only=False,

            runtime_configuration_id=(
                "static_part_a_grid_v1"
            ),

            interpretation=(
                "Static ADB reference without "
                "reactive or predictive actor control."
            ),
        ),

        ADBBaselineSpec(
            label=(
                ADBBaseline.ORIGINAL_REACTIVE_ADB
            ),

            predictive=False,
            future_mean=False,
            predictive_uncertainty=False,
            class_aware_policy=False,
            current_reactive_geometry=True,
            static_controller=False,
            evaluator_future_truth=False,
            future_truth_controller_input=False,
            constructed_oracle_only=False,

            runtime_configuration_id=(
                "part_a_current_reactive_v1"
            ),

            interpretation=(
                "Original Part-A-style current-time "
                "deterministic reactive ADB."
            ),
        ),

        ADBBaselineSpec(
            label=(
                ADBBaseline.DETERMINISTIC_PREDICTIVE_ADB
            ),

            predictive=True,
            future_mean=True,
            predictive_uncertainty=False,
            class_aware_policy=False,
            current_reactive_geometry=False,
            static_controller=False,
            evaluator_future_truth=False,
            future_truth_controller_input=False,
            constructed_oracle_only=False,

            runtime_configuration_id=(
                "predictive_mean_common_policy_v1"
            ),

            interpretation=(
                "Future predictive mean only; "
                "no predictive covariance."
            ),
        ),

        ADBBaselineSpec(
            label=(
                ADBBaseline.UNCERTAINTY_AWARE_PREDICTIVE_ADB
            ),

            predictive=True,
            future_mean=True,
            predictive_uncertainty=True,
            class_aware_policy=False,
            current_reactive_geometry=False,
            static_controller=False,
            evaluator_future_truth=False,
            future_truth_controller_input=False,
            constructed_oracle_only=False,

            runtime_configuration_id=(
                shared_uncertainty_configuration
            ),

            interpretation=(
                "Predictive mean plus calibrated "
                "uncertainty under a common "
                "class-agnostic illumination policy."
            ),
        ),

        ADBBaselineSpec(
            label=(
                ADBBaseline.CLASS_AGNOSTIC_PREDICTIVE_ADB
            ),

            predictive=True,
            future_mean=True,
            predictive_uncertainty=True,
            class_aware_policy=False,
            current_reactive_geometry=False,
            static_controller=False,
            evaluator_future_truth=False,
            future_truth_controller_input=False,
            constructed_oracle_only=False,

            runtime_configuration_id=(
                shared_uncertainty_configuration
            ),

            interpretation=(
                "Explicit PDF label for the same "
                "predictive-uncertainty controller "
                "with one common policy across classes."
            ),
        ),

        ADBBaselineSpec(
            label=(
                ADBBaseline.CLASS_AWARE_PREDICTIVE_ADB
            ),

            predictive=True,
            future_mean=True,
            predictive_uncertainty=True,
            class_aware_policy=True,
            current_reactive_geometry=False,
            static_controller=False,
            evaluator_future_truth=False,
            future_truth_controller_input=False,
            constructed_oracle_only=False,

            runtime_configuration_id=(
                "predictive_uncertainty_class_aware_v1"
            ),

            interpretation=(
                "Primary Stage6 controller: "
                "predictive uncertainty with distinct "
                "vehicle/pedestrian/cyclist policies."
            ),
        ),

        ADBBaselineSpec(
            label=(
                ADBBaseline.ORACLE_FUTURE_ADB
            ),

            predictive=False,
            future_mean=False,
            predictive_uncertainty=False,
            class_aware_policy=True,
            current_reactive_geometry=False,
            static_controller=False,
            evaluator_future_truth=True,
            future_truth_controller_input=False,
            constructed_oracle_only=True,

            runtime_configuration_id=(
                "constructed_future_oracle_v1"
            ),

            interpretation=(
                "Evaluator-only constructed future "
                "reference; never a controller."
            ),
        ),
    )


def required_metric_interfaces(
) -> tuple[
    ADBMetricInterface,
    ...
]:

    return (
        ADBMetricInterface(
            name=ADBMetric.MASK_IOU,
            requires_constructed_oracle=True,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=False,
            requires_intensity_map=False,
        ),

        ADBMetricInterface(
            name=(
                ADBMetric.VEHICLE_SHADOW_VIOLATION
            ),
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=True,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=False,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.GLARE_RISK_EXPOSURE,
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=True,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=False,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.OVER_MASKING_AREA,
            requires_constructed_oracle=True,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=True,
            requires_temporal_sequence=False,
            requires_intensity_map=False,
        ),

        ADBMetricInterface(
            name=(
                ADBMetric.ROAD_ILLUMINATION_RETENTION
            ),
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=True,
            requires_temporal_sequence=False,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=(
                ADBMetric.PEDESTRIAN_VISIBILITY_PROXY
            ),
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=True,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=False,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=(
                ADBMetric.CYCLIST_VISIBILITY_PROXY
            ),
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=True,
            requires_road_ROI=False,
            requires_temporal_sequence=False,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.FALSE_DIMMING,
            requires_constructed_oracle=True,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=True,
            requires_temporal_sequence=False,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.TEMPORAL_SMOOTHNESS,
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=True,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.FLICKER_CHANGE_RATE,
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=True,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.ENERGY_CONSUMPTION,
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=True,
            requires_temporal_sequence=True,
            requires_intensity_map=True,
        ),

        ADBMetricInterface(
            name=ADBMetric.ACTUATION_LATENCY,
            requires_constructed_oracle=False,
            requires_vehicle_truth_region=False,
            requires_pedestrian_truth_region=False,
            requires_cyclist_truth_region=False,
            requires_road_ROI=False,
            requires_temporal_sequence=True,
            requires_intensity_map=False,
        ),
    )


def baseline_by_label(
    label: ADBBaseline | str,
) -> ADBBaselineSpec:

    resolved = ADBBaseline(
        label
    )

    for spec in required_baseline_specs():
        if spec.label == resolved:
            return spec

    raise RuntimeError(
        f"Missing baseline: {resolved}"
    )


def metric_by_name(
    name: ADBMetric | str,
) -> ADBMetricInterface:

    resolved = ADBMetric(
        name
    )

    for metric in required_metric_interfaces():
        if metric.name == resolved:
            return metric

    raise RuntimeError(
        f"Missing metric: {resolved}"
    )

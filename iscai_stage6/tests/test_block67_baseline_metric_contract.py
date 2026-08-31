from __future__ import annotations

import unittest

from iscai_stage6.adb.baseline_metric_contract import (
    ADBBaseline,
    ADBMetric,
    ConstructedOracleContract,
    baseline_by_label,
    metric_by_name,
    required_baseline_specs,
    required_metric_interfaces,
)


class Block67BaselineMetricContractTest(
    unittest.TestCase
):

    def test_exact_seven_baseline_labels(
        self,
    ):

        self.assertEqual(
            {
                item.value
                for item in ADBBaseline
            },
            {
                "static_ADB",
                "original_reactive_ADB",
                "deterministic_predictive_ADB",
                "uncertainty_aware_predictive_ADB",
                "class_agnostic_predictive_ADB",
                "class_aware_predictive_ADB",
                "oracle_future_ADB",
            },
        )

        self.assertEqual(
            len(
                required_baseline_specs()
            ),
            7,
        )

    def test_exact_twelve_metric_labels(
        self,
    ):

        self.assertEqual(
            len(
                required_metric_interfaces()
            ),
            12,
        )

        self.assertEqual(
            {
                item.value
                for item in ADBMetric
            },
            {
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
            },
        )

    def test_no_nonoracle_controller_uses_future_truth(
        self,
    ):

        for spec in required_baseline_specs():

            self.assertFalse(
                spec.future_truth_controller_input
            )

            if (
                spec.label
                !=
                ADBBaseline.ORACLE_FUTURE_ADB
            ):
                self.assertFalse(
                    spec.evaluator_future_truth
                )

                self.assertFalse(
                    spec.constructed_oracle_only
                )

    def test_oracle_is_evaluator_only_constructed_reference(
        self,
    ):

        spec = baseline_by_label(
            ADBBaseline.ORACLE_FUTURE_ADB
        )

        self.assertTrue(
            spec.evaluator_future_truth
        )

        self.assertFalse(
            spec.future_truth_controller_input
        )

        self.assertTrue(
            spec.constructed_oracle_only
        )

        oracle = (
            ConstructedOracleContract()
        )

        self.assertFalse(
            oracle.measured_ADB_ground_truth
        )

        self.assertFalse(
            oracle.available_to_controller
        )

        self.assertFalse(
            oracle.available_for_parameter_tuning
        )

        self.assertTrue(
            oracle.available_for_scoring
        )

    def test_primary_class_aware_controller_has_uncertainty(
        self,
    ):

        spec = baseline_by_label(
            ADBBaseline.CLASS_AWARE_PREDICTIVE_ADB
        )

        self.assertTrue(
            spec.predictive
        )

        self.assertTrue(
            spec.future_mean
        )

        self.assertTrue(
            spec.predictive_uncertainty
        )

        self.assertTrue(
            spec.class_aware_policy
        )

        self.assertFalse(
            spec.future_truth_controller_input
        )

    def test_deterministic_predictive_removes_uncertainty(
        self,
    ):

        spec = baseline_by_label(
            ADBBaseline.DETERMINISTIC_PREDICTIVE_ADB
        )

        self.assertTrue(
            spec.predictive
        )

        self.assertTrue(
            spec.future_mean
        )

        self.assertFalse(
            spec.predictive_uncertainty
        )

    def test_uncertainty_and_class_agnostic_overlap_is_explicit(
        self,
    ):

        uncertainty = baseline_by_label(
            ADBBaseline.UNCERTAINTY_AWARE_PREDICTIVE_ADB
        )

        agnostic = baseline_by_label(
            ADBBaseline.CLASS_AGNOSTIC_PREDICTIVE_ADB
        )

        self.assertEqual(
            uncertainty.runtime_configuration_id,
            agnostic.runtime_configuration_id,
        )

        self.assertFalse(
            uncertainty.class_aware_policy
        )

        self.assertFalse(
            agnostic.class_aware_policy
        )

    def test_metric_interfaces_are_evaluator_only(
        self,
    ):

        for metric in (
            required_metric_interfaces()
        ):
            self.assertFalse(
                metric.controller_input_metric
            )

    def test_constructed_oracle_metrics_are_marked(
        self,
    ):

        for name in (
            ADBMetric.MASK_IOU,
            ADBMetric.OVER_MASKING_AREA,
            ADBMetric.FALSE_DIMMING,
        ):
            metric = metric_by_name(
                name
            )

            self.assertTrue(
                metric.requires_constructed_oracle
            )

    def test_metric_formula_freeze_is_not_done_here(
        self,
    ):
        """
        Block6.7 freezes the metric set/interface.
        Block6.8 must freeze numerical formulas/ROIs/
        acceptance tolerances before formal evaluation.
        """

        metric = metric_by_name(
            ADBMetric.VEHICLE_SHADOW_VIOLATION
        )

        self.assertTrue(
            metric.requires_vehicle_truth_region
        )

        self.assertTrue(
            metric.requires_intensity_map
        )


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import unittest

from iscai_stage4.ml.scientific_repair_v1 import (
    PredictionBeforeTruthLedger,
    PredictionEvent,
    ScientificBoundaryError,
    assert_causal_formal_source,
    paired_common_support_metrics,
    reject_cross_population_scalar_claim,
)


class TestStage4ScientificRepairV1(
    unittest.TestCase
):

    def test_truth_before_prediction_seal_fails(
        self
    ):

        ledger = (
            PredictionBeforeTruthLedger()
        )

        with self.assertRaises(
            ScientificBoundaryError
        ):
            ledger.allow_evaluator_truth()


    def test_prediction_after_truth_fails(
        self
    ):

        ledger = (
            PredictionBeforeTruthLedger()
        )

        ledger.add_prediction(
            PredictionEvent(
                scenario_id="s",
                prediction_id="p",
                horizon_s=0.1,
                predicted_xy_m=(
                    1.0,
                    2.0,
                ),
            )
        )

        ledger.seal_predictions()
        ledger.allow_evaluator_truth()

        with self.assertRaises(
            ScientificBoundaryError
        ):
            ledger.add_prediction(
                PredictionEvent(
                    scenario_id="s",
                    prediction_id="q",
                    horizon_s=0.1,
                    predicted_xy_m=(
                        0.0,
                        0.0,
                    ),
                )
            )


    def test_common_support_only(
        self
    ):

        neural = [
            {
                "scenario_id": "s",
                "target_id": "a",
                "horizon_s": 0.1,
                "predicted_xy_m":
                    [1.0, 0.0],
                "truth_xy_m":
                    [0.0, 0.0],
            },
            {
                "scenario_id": "s",
                "target_id": "b",
                "horizon_s": 0.1,
                "predicted_xy_m":
                    [100.0, 0.0],
                "truth_xy_m":
                    [0.0, 0.0],
            },
        ]

        classical = [
            {
                "scenario_id": "s",
                "target_id": "a",
                "horizon_s": 0.1,
                "predicted_xy_m":
                    [2.0, 0.0],
                "truth_xy_m":
                    [0.0, 0.0],
            }
        ]

        result = (
            paired_common_support_metrics(
                neural,
                classical,
            )
        )

        self.assertEqual(
            result[
                "common_event_count"
            ],
            1,
        )

        self.assertEqual(
            result[
                "neural_ADE_m"
            ],
            1.0,
        )

        self.assertEqual(
            result[
                "classical_ADE_m"
            ],
            2.0,
        )


    def test_cross_population_scalar_claim_fails(
        self
    ):

        with self.assertRaises(
            ScientificBoundaryError
        ):
            reject_cross_population_scalar_claim(
                neural_support=494,
                classical_support=465,
            )


    def test_common_support_allows_comparison(
        self
    ):

        reject_cross_population_scalar_claim(
            neural_support=494,
            classical_support=465,
            common_support=400,
        )


    def test_attach_supervision_rejected_from_repaired_route(
        self
    ):

        with self.assertRaises(
            ScientificBoundaryError
        ):
            assert_causal_formal_source(
                """
                PREDICTIONS_SEALED_BEFORE_EVALUATOR_TRUTH
                attach_supervision(...)
                EVALUATOR_TRUTH_READ_AFTER_PREDICTIONS
                """
            )


    def test_phase_order_passes(
        self
    ):

        assert_causal_formal_source(
            """
            PREDICTIONS_SEALED_BEFORE_EVALUATOR_TRUTH
            EVALUATOR_TRUTH_READ_AFTER_PREDICTIONS
            """
        )


if __name__ == "__main__":
    unittest.main()

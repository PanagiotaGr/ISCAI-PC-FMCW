import unittest

from iscai_stage3.evaluation import (
    EvaluationConfig,
    EvaluationPredictionSet,
    EvaluationPredictionTrack,
    EvaluationTruth,
    EvaluationTruthTrack,
    PredictionTrajectoryPoint,
    TruthTrajectoryPoint,
    assign_predictions_to_truth,
    evaluate_prediction_set,
)


HORIZONS = (
    0.1,
    0.3,
    0.5,
    1.0,
)


def truth_track(
    truth_id,
    *,
    anchor_x,
    velocity_x,
    y=0.0,
    z=0.0,
    actor_class="TYPE_VEHICLE",
):
    points = tuple(
        TruthTrajectoryPoint(
            horizon_s=horizon,
            timestamp_s=(
                1.0
                +
                horizon
            ),
            position_H0_m=(
                anchor_x
                +
                velocity_x
                *
                horizon,
                y,
                z,
            ),
        )
        for horizon in HORIZONS
    )

    return EvaluationTruthTrack(
        truth_id=truth_id,
        actor_class=actor_class,
        anchor_position_H0_m=(
            anchor_x,
            y,
            z,
        ),
        points=points,
    )


def prediction_track(
    prediction_id,
    *,
    anchor_x,
    velocity_x,
    y=0.0,
    z=0.0,
    offset_x=0.0,
):
    points = tuple(
        PredictionTrajectoryPoint(
            horizon_s=horizon,
            timestamp_s=(
                1.0
                +
                horizon
            ),
            position_H0_m=(
                anchor_x
                +
                velocity_x
                *
                horizon
                +
                offset_x,
                y,
                z,
            ),
        )
        for horizon in HORIZONS
    )

    return EvaluationPredictionTrack(
        prediction_id=(
            prediction_id
        ),
        anchor_position_H0_m=(
            anchor_x,
            y,
            z,
        ),
        points=points,
    )


def truth_set(
    *tracks,
):
    return EvaluationTruth(
        scenario_id="scene",
        anchor_timestamp_s=1.0,
        tracks=tuple(tracks),
    )


def prediction_set(
    *tracks,
    method="TEST",
):
    return EvaluationPredictionSet(
        method=method,
        scenario_id="scene",
        anchor_timestamp_s=1.0,
        tracks=tuple(tracks),
    )


class TestCommonEvaluator(
    unittest.TestCase
):

    def test_exact_prediction_zero_ADE_FDE(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=5.0,
            )
        )

        predictions = prediction_set(
            prediction_track(
                "P1",
                anchor_x=10.0,
                velocity_x=5.0,
            )
        )

        report = evaluate_prediction_set(
            predictions,
            truth,
        )

        self.assertAlmostEqual(
            report.ade_m,
            0.0,
        )

        self.assertAlmostEqual(
            report.fde_m,
            0.0,
        )

        self.assertAlmostEqual(
            report.ade_3d_m,
            0.0,
        )

        self.assertAlmostEqual(
            report.fde_3d_m,
            0.0,
        )

    def test_assignment_is_one_to_one(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            ),
            truth_track(
                "T2",
                anchor_x=20.0,
                velocity_x=0.0,
            ),
        )

        predictions = prediction_set(
            prediction_track(
                "P2",
                anchor_x=20.1,
                velocity_x=0.0,
            ),
            prediction_track(
                "P1",
                anchor_x=10.1,
                velocity_x=0.0,
            ),
        )

        assignment = (
            assign_predictions_to_truth(
                predictions,
                truth,
                config=EvaluationConfig(),
            )
        )

        pairs = {
            (
                match.prediction_id,
                match.truth_id,
            )
            for match in (
                assignment.matches
            )
        }

        self.assertEqual(
            pairs,
            {
                ("P1", "T1"),
                ("P2", "T2"),
            },
        )

    def test_future_truth_does_not_change_assignment(
        self,
    ):
        original = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=1.0,
            ),
            truth_track(
                "T2",
                anchor_x=20.0,
                velocity_x=-1.0,
            ),
        )

        mutated = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=100.0,
            ),
            truth_track(
                "T2",
                anchor_x=20.0,
                velocity_x=-100.0,
            ),
        )

        predictions = prediction_set(
            prediction_track(
                "P1",
                anchor_x=10.1,
                velocity_x=1.0,
            ),
            prediction_track(
                "P2",
                anchor_x=19.9,
                velocity_x=-1.0,
            ),
        )

        first = assign_predictions_to_truth(
            predictions,
            original,
            config=EvaluationConfig(),
        )

        second = assign_predictions_to_truth(
            predictions,
            mutated,
            config=EvaluationConfig(),
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertFalse(
            first
            .future_truth_used_for_assignment
        )

    def test_far_prediction_is_unmatched(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            )
        )

        predictions = prediction_set(
            prediction_track(
                "P1",
                anchor_x=100.0,
                velocity_x=0.0,
            )
        )

        report = evaluate_prediction_set(
            predictions,
            truth,
        )

        self.assertEqual(
            report.matched_track_count,
            0,
        )

        self.assertEqual(
            report.unmatched_truth_count,
            1,
        )

        self.assertEqual(
            report.false_prediction_count,
            1,
        )

    def test_reconstruction_precision_recall(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            ),
            truth_track(
                "T2",
                anchor_x=20.0,
                velocity_x=0.0,
            ),
        )

        predictions = prediction_set(
            prediction_track(
                "P1",
                anchor_x=10.0,
                velocity_x=0.0,
            ),
            prediction_track(
                "FP",
                anchor_x=100.0,
                velocity_x=0.0,
            ),
        )

        report = evaluate_prediction_set(
            predictions,
            truth,
        )

        self.assertAlmostEqual(
            report
            .reconstruction_recall,
            0.5,
        )

        self.assertAlmostEqual(
            report
            .reconstruction_precision,
            0.5,
        )

        self.assertAlmostEqual(
            report
            .reconstruction_f1,
            0.5,
        )

    def test_endpoint_threshold_miss(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            )
        )

        predictions = prediction_set(
            prediction_track(
                "P1",
                anchor_x=10.0,
                velocity_x=0.0,
                offset_x=3.0,
            )
        )

        report = evaluate_prediction_set(
            predictions,
            truth,
            config=EvaluationConfig(
                endpoint_miss_threshold_m=(
                    2.0
                ),
            ),
        )

        for aggregate in (
            report.horizon_aggregates
        ):
            self.assertEqual(
                aggregate
                .endpoint_miss_count,
                1,
            )

            self.assertAlmostEqual(
                aggregate
                .endpoint_miss_rate,
                1.0,
            )

    def test_missing_prediction_point_is_miss(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            )
        )

        prediction = (
            EvaluationPredictionTrack(
                prediction_id="P1",
                anchor_position_H0_m=(
                    10.0,
                    0.0,
                    0.0,
                ),
                points=(
                    PredictionTrajectoryPoint(
                        horizon_s=0.1,
                        timestamp_s=1.1,
                        position_H0_m=(
                            10.0,
                            0.0,
                            0.0,
                        ),
                    ),
                ),
            )
        )

        report = evaluate_prediction_set(
            prediction_set(
                prediction
            ),
            truth,
        )

        final = (
            report
            .horizon_aggregates[-1]
        )

        self.assertEqual(
            final.endpoint_miss_count,
            1,
        )

        self.assertAlmostEqual(
            final.endpoint_miss_rate,
            1.0,
        )

    def test_order_does_not_change_result(
        self,
    ):
        truth_a = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=1.0,
            ),
            truth_track(
                "T2",
                anchor_x=20.0,
                velocity_x=-1.0,
            ),
        )

        truth_b = truth_set(
            truth_track(
                "T2",
                anchor_x=20.0,
                velocity_x=-1.0,
            ),
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=1.0,
            ),
        )

        pred_a = prediction_set(
            prediction_track(
                "P1",
                anchor_x=10.1,
                velocity_x=1.0,
            ),
            prediction_track(
                "P2",
                anchor_x=20.1,
                velocity_x=-1.0,
            ),
        )

        pred_b = prediction_set(
            prediction_track(
                "P2",
                anchor_x=20.1,
                velocity_x=-1.0,
            ),
            prediction_track(
                "P1",
                anchor_x=10.1,
                velocity_x=1.0,
            ),
        )

        first = evaluate_prediction_set(
            pred_a,
            truth_a,
        )

        second = evaluate_prediction_set(
            pred_b,
            truth_b,
        )

        self.assertEqual(
            first,
            second,
        )

    def test_planar_and_3d_metrics_are_separate(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
                z=0.0,
            )
        )

        prediction = (
            EvaluationPredictionTrack(
                prediction_id="P1",
                anchor_position_H0_m=(
                    10.0,
                    0.0,
                    0.0,
                ),
                points=tuple(
                    PredictionTrajectoryPoint(
                        horizon_s=h,
                        timestamp_s=(
                            1.0 + h
                        ),
                        position_H0_m=(
                            10.0,
                            0.0,
                            2.0,
                        ),
                    )
                    for h in HORIZONS
                ),
            )
        )

        report = evaluate_prediction_set(
            prediction_set(
                prediction
            ),
            truth,
        )

        self.assertAlmostEqual(
            report.ade_m,
            0.0,
        )

        self.assertAlmostEqual(
            report.ade_3d_m,
            2.0,
        )

    def test_runtime_is_reported_not_used_in_metric(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            )
        )

        prediction = (
            EvaluationPredictionSet(
                method="TEST",
                scenario_id="scene",
                anchor_timestamp_s=1.0,
                tracks=(
                    prediction_track(
                        "P1",
                        anchor_x=10.0,
                        velocity_x=0.0,
                    ),
                ),
                runtime_ms=12.5,
            )
        )

        report = evaluate_prediction_set(
            prediction,
            truth,
        )

        self.assertAlmostEqual(
            report.runtime_ms,
            12.5,
        )

        self.assertAlmostEqual(
            report.ade_m,
            0.0,
        )

    def test_scenario_mismatch_rejected(
        self,
    ):
        truth = truth_set(
            truth_track(
                "T1",
                anchor_x=10.0,
                velocity_x=0.0,
            )
        )

        predictions = (
            EvaluationPredictionSet(
                method="TEST",
                scenario_id="other",
                anchor_timestamp_s=1.0,
                tracks=(),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            evaluate_prediction_set(
                predictions,
                truth,
            )


if __name__ == "__main__":
    unittest.main()

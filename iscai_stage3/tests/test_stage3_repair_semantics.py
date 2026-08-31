from __future__ import annotations

from dataclasses import dataclass
import unittest

from iscai_stage3.association.contracts import (
    EstimatedAssociationResult,
)
from iscai_stage3.evaluation.anchor_alignment import (
    align_state_prediction_to_global_anchor,
)
from iscai_stage3.evaluation.contracts import (
    EvaluationConfig,
    EvaluationPredictionSet,
    EvaluationPredictionTrack,
    EvaluationTruth,
    EvaluationTruthTrack,
    PredictionTrajectoryPoint,
    TruthTrajectoryPoint,
)
from iscai_stage3.evaluation.metrics import (
    evaluate_prediction_set,
)


class TestStage3ScientificRepair(unittest.TestCase):
    def _truth_track(self, truth_id, x):
        return EvaluationTruthTrack(
            truth_id=truth_id,
            actor_class="TYPE_VEHICLE",
            anchor_position_H0_m=(x, 0.0, 0.0),
            points=(
                TruthTrajectoryPoint(
                    horizon_s=0.1,
                    timestamp_s=1.1,
                    position_H0_m=(x, 0.0, 0.0),
                ),
            ),
        )

    def _prediction_track(self, prediction_id, x):
        return EvaluationPredictionTrack(
            prediction_id=prediction_id,
            anchor_position_H0_m=(x, 0.0, 0.0),
            points=(
                PredictionTrajectoryPoint(
                    horizon_s=0.1,
                    timestamp_s=1.1,
                    position_H0_m=(x, 0.0, 0.0),
                ),
            ),
        )

    def test_prediction_set_rejects_stale_timestamp_relabel(self):
        track = EvaluationPredictionTrack(
            prediction_id="stale",
            anchor_position_H0_m=(0.0, 0.0, 0.0),
            points=(
                PredictionTrajectoryPoint(
                    horizon_s=0.1,
                    timestamp_s=0.9,
                    position_H0_m=(0.0, 0.0, 0.0),
                ),
            ),
        )

        with self.assertRaises(ValueError):
            EvaluationPredictionSet(
                method="CV",
                scenario_id="s",
                anchor_timestamp_s=1.0,
                tracks=(track,),
            )

    def test_real_non_target_prediction_is_ignored_not_false_positive(self):
        predictions = EvaluationPredictionSet(
            method="CV",
            scenario_id="s",
            anchor_timestamp_s=1.0,
            tracks=(
                self._prediction_track("target", 0.0),
                self._prediction_track("real-nontarget", 10.0),
                self._prediction_track("clutter", 100.0),
            ),
        )

        primary = EvaluationTruth(
            scenario_id="s",
            anchor_timestamp_s=1.0,
            tracks=(self._truth_track("truth-target", 0.0),),
        )

        ignored = EvaluationTruth(
            scenario_id="s",
            anchor_timestamp_s=1.0,
            tracks=(self._truth_track("truth-other", 10.0),),
        )

        report = evaluate_prediction_set(
            predictions,
            primary,
            config=EvaluationConfig(
                horizons_s=(0.1,),
                anchor_assignment_gate_m=5.0,
                endpoint_miss_threshold_m=2.0,
            ),
            ignored_truth=ignored,
        )

        self.assertEqual(report.matched_track_count, 1)
        self.assertEqual(report.ignored_prediction_count, 1)
        self.assertEqual(report.false_prediction_count, 1)
        self.assertEqual(report.ignored_prediction_ids, ("real-nontarget",))
        self.assertAlmostEqual(report.reconstruction_precision, 0.5)
        self.assertAlmostEqual(report.reconstruction_recall, 1.0)

    def test_stale_active_state_is_propagated_to_global_anchor(self):
        @dataclass(frozen=True)
        class State:
            track_id: str = "x"
            timestamp_s: float = 0.8
            position_H0_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

        @dataclass(frozen=True)
        class Point:
            horizon_s: float
            timestamp_s: float
            position_H0_m: tuple[float, float, float]

        @dataclass(frozen=True)
        class Prediction:
            track_id: str
            points: tuple[Point, ...]

        def predictor(state, *, horizons_s):
            return Prediction(
                track_id=state.track_id,
                points=tuple(
                    Point(
                        horizon_s=float(tau),
                        timestamp_s=state.timestamp_s + float(tau),
                        position_H0_m=(float(tau), 0.0, 0.0),
                    )
                    for tau in horizons_s
                ),
            )

        result = align_state_prediction_to_global_anchor(
            State(),
            predictor,
            horizons_s=(0.1, 0.3),
            anchor_timestamp_s=1.0,
        )

        self.assertAlmostEqual(result.anchor_position_H0_m[0], 0.2)
        self.assertEqual(tuple(p.horizon_s for p in result.points), (0.1, 0.3))
        self.assertEqual(tuple(p.timestamp_s for p in result.points), (1.1, 1.3))
        self.assertEqual(
            EstimatedAssociationResult(scenario_id="s", frames=(), tracks=()).method,
            "deterministic_gated_greedy_nearest_neighbor",
        )


if __name__ == "__main__":
    unittest.main()

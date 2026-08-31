from .contracts import (
    EvaluationConfig,
    EvaluationPredictionSet,
    EvaluationPredictionTrack,
    EvaluationTruth,
    EvaluationTruthTrack,
    PredictionTrajectoryPoint,
    TruthTrajectoryPoint,
)

from .adapters import (
    adapt_prediction_object,
)

from .assignment import (
    AssignmentResult,
    EvaluationMatch,
    assign_predictions_to_truth,
)

from .metrics import (
    EvaluationReport,
    HorizonAggregate,
    HorizonError,
    TrackEvaluation,
    evaluate_prediction_set,
)


__all__ = [
    "EvaluationConfig",
    "EvaluationPredictionSet",
    "EvaluationPredictionTrack",
    "EvaluationTruth",
    "EvaluationTruthTrack",
    "PredictionTrajectoryPoint",
    "TruthTrajectoryPoint",

    "adapt_prediction_object",

    "AssignmentResult",
    "EvaluationMatch",
    "assign_predictions_to_truth",

    "EvaluationReport",
    "HorizonAggregate",
    "HorizonError",
    "TrackEvaluation",
    "evaluate_prediction_set",
]

from .womd_truth import (
    anchor_class_inventory,
    build_womd_truth_sidecar,
)

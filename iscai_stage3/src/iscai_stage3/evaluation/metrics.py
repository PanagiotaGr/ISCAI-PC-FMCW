from __future__ import annotations

from dataclasses import dataclass
from math import (
    sqrt,
)

from iscai_stage3.evaluation.assignment import (
    AssignmentResult,
    assign_predictions_to_truth,
)

from iscai_stage3.evaluation.contracts import (
    EvaluationConfig,
    EvaluationPredictionSet,
    EvaluationTruth,
)


@dataclass(frozen=True)
class HorizonError:
    horizon_s: float

    planar_error_m: float

    error_3d_m: float


@dataclass(frozen=True)
class TrackEvaluation:
    prediction_id: str
    truth_id: str

    actor_class: str

    anchor_assignment_error_m: float

    horizon_errors: tuple[
        HorizonError,
        ...
    ]

    ade_m: float | None
    fde_m: float | None

    ade_3d_m: float | None
    fde_3d_m: float | None


@dataclass(frozen=True)
class HorizonAggregate:
    horizon_s: float

    eligible_truth_tracks: int

    available_predictions: int

    mean_planar_error_m: float | None
    rmse_planar_error_m: float | None

    mean_3d_error_m: float | None
    rmse_3d_error_m: float | None

    endpoint_miss_count: int

    endpoint_miss_rate: float


@dataclass(frozen=True)
class EvaluationReport:
    method: str

    scenario_id: str

    anchor_timestamp_s: float

    horizons_s: tuple[
        float,
        ...
    ]

    truth_track_count: int
    prediction_track_count: int

    matched_track_count: int

    unmatched_truth_count: int
    false_prediction_count: int

    reconstruction_recall: float
    reconstruction_precision: float
    reconstruction_f1: float

    ade_m: float | None
    fde_m: float | None

    ade_3d_m: float | None
    fde_3d_m: float | None

    track_evaluations: tuple[
        TrackEvaluation,
        ...
    ]

    horizon_aggregates: tuple[
        HorizonAggregate,
        ...
    ]

    assignment: AssignmentResult

    runtime_ms: float | None

    assignment_future_truth_used: bool = False

    # Predictions that match a real current-valid actor outside
    # the benchmark target subset are evaluator-ignored, not FP.
    ignored_prediction_count: int = 0
    ignored_prediction_ids: tuple[str, ...] = ()


def _planar_distance(
    prediction,
    truth,
) -> float:
    dx = (
        float(prediction[0])
        -
        float(truth[0])
    )

    dy = (
        float(prediction[1])
        -
        float(truth[1])
    )

    return sqrt(
        dx * dx
        +
        dy * dy
    )


def _distance_3d(
    prediction,
    truth,
) -> float:
    return sqrt(
        sum(
            (
                float(prediction[index])
                -
                float(truth[index])
            )
            ** 2
            for index in range(3)
        )
    )


def _find_point(
    points,
    horizon_s: float,
    *,
    tolerance_s: float,
):
    matches = [
        point
        for point in points
        if abs(
            point.horizon_s
            -
            horizon_s
        ) <= tolerance_s
    ]

    if len(matches) > 1:
        raise ValueError(
            "Multiple points map to the "
            "same evaluation horizon."
        )

    if not matches:
        return None

    return matches[0]


def _mean(
    values,
):
    if not values:
        return None

    return (
        sum(values)
        /
        len(values)
    )


def _rmse(
    values,
):
    if not values:
        return None

    return sqrt(
        sum(
            value * value
            for value in values
        )
        /
        len(values)
    )


def _f1(
    precision: float,
    recall: float,
) -> float:
    if (
        precision
        +
        recall
    ) <= 0.0:
        return 0.0

    return (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
    )


def evaluate_prediction_set(
    predictions: EvaluationPredictionSet,
    truth: EvaluationTruth,
    *,
    config: EvaluationConfig | None = None,
    ignored_truth: EvaluationTruth | None = None,
) -> EvaluationReport:
    if config is None:
        config = EvaluationConfig()

    if (
        predictions.scenario_id
        !=
        truth.scenario_id
    ):
        raise ValueError(
            "Prediction/truth scenario IDs "
            "do not match."
        )

    if abs(
        predictions.anchor_timestamp_s
        -
        truth.anchor_timestamp_s
    ) > (
        config
        .anchor_timestamp_tolerance_s
    ):
        raise ValueError(
            "Prediction/truth anchor "
            "timestamps do not match."
        )

    if ignored_truth is not None:
        if ignored_truth.scenario_id != truth.scenario_id:
            raise ValueError(
                "Ignored-truth scenario ID does not match."
            )

        if abs(
            ignored_truth.anchor_timestamp_s
            - truth.anchor_timestamp_s
        ) > config.anchor_timestamp_tolerance_s:
            raise ValueError(
                "Ignored-truth anchor timestamp does not match."
            )

        primary_truth_ids = {
            track.truth_id
            for track in truth.tracks
        }
        ignored_truth_ids = {
            track.truth_id
            for track in ignored_truth.tracks
        }

        if primary_truth_ids & ignored_truth_ids:
            raise ValueError(
                "Primary and ignored truth sets must be disjoint."
            )

    assignment = (
        assign_predictions_to_truth(
            predictions,
            truth,
            config=config,
        )
    )

    prediction_by_id = {
        track.prediction_id: track
        for track in predictions.tracks
    }

    truth_by_id = {
        track.truth_id: track
        for track in truth.tracks
    }

    track_evaluations = []

    match_by_truth_id = {
        match.truth_id: match
        for match in assignment.matches
    }

    for match in assignment.matches:
        prediction = prediction_by_id[
            match.prediction_id
        ]

        target = truth_by_id[
            match.truth_id
        ]

        horizon_errors = []

        for horizon in config.horizons_s:
            prediction_point = _find_point(
                prediction.points,
                horizon,
                tolerance_s=(
                    config
                    .horizon_tolerance_s
                ),
            )

            truth_point = _find_point(
                target.points,
                horizon,
                tolerance_s=(
                    config
                    .horizon_tolerance_s
                ),
            )

            if (
                prediction_point is None
                or
                truth_point is None
            ):
                continue

            horizon_errors.append(
                HorizonError(
                    horizon_s=horizon,
                    planar_error_m=(
                        _planar_distance(
                            prediction_point
                            .position_H0_m,
                            truth_point
                            .position_H0_m,
                        )
                    ),
                    error_3d_m=(
                        _distance_3d(
                            prediction_point
                            .position_H0_m,
                            truth_point
                            .position_H0_m,
                        )
                    ),
                )
            )

        planar_errors = [
            item.planar_error_m
            for item in horizon_errors
        ]

        errors_3d = [
            item.error_3d_m
            for item in horizon_errors
        ]

        final_horizon = (
            config.horizons_s[-1]
        )

        final_error = next(
            (
                item
                for item in horizon_errors
                if abs(
                    item.horizon_s
                    -
                    final_horizon
                )
                <=
                config
                .horizon_tolerance_s
            ),
            None,
        )

        track_evaluations.append(
            TrackEvaluation(
                prediction_id=(
                    match.prediction_id
                ),
                truth_id=(
                    match.truth_id
                ),
                actor_class=(
                    target.actor_class
                ),
                anchor_assignment_error_m=(
                    match.anchor_distance_m
                ),
                horizon_errors=tuple(
                    horizon_errors
                ),
                ade_m=_mean(
                    planar_errors
                ),
                fde_m=(
                    None
                    if final_error is None
                    else
                    final_error
                    .planar_error_m
                ),
                ade_3d_m=_mean(
                    errors_3d
                ),
                fde_3d_m=(
                    None
                    if final_error is None
                    else
                    final_error
                    .error_3d_m
                ),
            )
        )

    track_evaluations.sort(
        key=lambda item: (
            item.truth_id,
            item.prediction_id,
        )
    )

    horizon_aggregates = []

    for horizon in config.horizons_s:
        eligible_truth = []

        for target in truth.tracks:
            truth_point = _find_point(
                target.points,
                horizon,
                tolerance_s=(
                    config
                    .horizon_tolerance_s
                ),
            )

            if truth_point is not None:
                eligible_truth.append(
                    (
                        target,
                        truth_point,
                    )
                )

        planar_errors = []
        errors_3d = []
        misses = 0

        for (
            target,
            truth_point,
        ) in eligible_truth:
            match = (
                match_by_truth_id.get(
                    target.truth_id
                )
            )

            if match is None:
                misses += 1
                continue

            prediction = (
                prediction_by_id[
                    match.prediction_id
                ]
            )

            prediction_point = (
                _find_point(
                    prediction.points,
                    horizon,
                    tolerance_s=(
                        config
                        .horizon_tolerance_s
                    ),
                )
            )

            if prediction_point is None:
                misses += 1
                continue

            planar = _planar_distance(
                prediction_point
                .position_H0_m,
                truth_point
                .position_H0_m,
            )

            distance_3d = _distance_3d(
                prediction_point
                .position_H0_m,
                truth_point
                .position_H0_m,
            )

            planar_errors.append(
                planar
            )

            errors_3d.append(
                distance_3d
            )

            if (
                planar
                >
                config
                .endpoint_miss_threshold_m
            ):
                misses += 1

        denominator = len(
            eligible_truth
        )

        miss_rate = (
            0.0
            if denominator == 0
            else
            misses
            /
            denominator
        )

        horizon_aggregates.append(
            HorizonAggregate(
                horizon_s=horizon,
                eligible_truth_tracks=(
                    denominator
                ),
                available_predictions=(
                    len(
                        planar_errors
                    )
                ),
                mean_planar_error_m=(
                    _mean(
                        planar_errors
                    )
                ),
                rmse_planar_error_m=(
                    _rmse(
                        planar_errors
                    )
                ),
                mean_3d_error_m=(
                    _mean(
                        errors_3d
                    )
                ),
                rmse_3d_error_m=(
                    _rmse(
                        errors_3d
                    )
                ),
                endpoint_miss_count=(
                    misses
                ),
                endpoint_miss_rate=(
                    miss_rate
                ),
            )
        )

    # Primary assignment is intentionally performed first.
    # Only predictions still unmatched after primary-target
    # assignment may be absorbed by the ignored real-actor set.
    ignored_prediction_ids: tuple[str, ...] = ()

    if ignored_truth is not None and assignment.unmatched_prediction_ids:
        unmatched_ids = set(
            assignment.unmatched_prediction_ids
        )

        residual_predictions = EvaluationPredictionSet(
            method=predictions.method,
            scenario_id=predictions.scenario_id,
            anchor_timestamp_s=predictions.anchor_timestamp_s,
            tracks=tuple(
                track
                for track in predictions.tracks
                if track.prediction_id in unmatched_ids
            ),
            runtime_ms=predictions.runtime_ms,
        )

        ignored_assignment = assign_predictions_to_truth(
            residual_predictions,
            ignored_truth,
            config=config,
        )

        ignored_prediction_ids = tuple(
            sorted(
                match.prediction_id
                for match in ignored_assignment.matches
            )
        )

    ignored_id_set = set(ignored_prediction_ids)

    false_prediction_ids = tuple(
        prediction_id
        for prediction_id in assignment.unmatched_prediction_ids
        if prediction_id not in ignored_id_set
    )

    truth_count = len(
        truth.tracks
    )

    prediction_count = len(
        predictions.tracks
    )

    match_count = len(
        assignment.matches
    )

    recall = (
        1.0
        if truth_count == 0
        else
        match_count
        /
        truth_count
    )

    precision_denominator = (
        match_count
        + len(false_prediction_ids)
    )

    precision = (
        1.0
        if precision_denominator == 0
        and truth_count == 0
        else
        (
            0.0
            if precision_denominator == 0
            else
            match_count
            /
            precision_denominator
        )
    )

    track_ades = [
        item.ade_m
        for item in track_evaluations
        if item.ade_m is not None
    ]

    track_fdes = [
        item.fde_m
        for item in track_evaluations
        if item.fde_m is not None
    ]

    track_ades_3d = [
        item.ade_3d_m
        for item in track_evaluations
        if item.ade_3d_m is not None
    ]

    track_fdes_3d = [
        item.fde_3d_m
        for item in track_evaluations
        if item.fde_3d_m is not None
    ]

    return EvaluationReport(
        method=predictions.method,
        scenario_id=(
            predictions.scenario_id
        ),
        anchor_timestamp_s=(
            predictions
            .anchor_timestamp_s
        ),
        horizons_s=(
            config.horizons_s
        ),
        truth_track_count=(
            truth_count
        ),
        prediction_track_count=(
            prediction_count
        ),
        matched_track_count=(
            match_count
        ),
        unmatched_truth_count=len(
            assignment
            .unmatched_truth_ids
        ),
        false_prediction_count=len(
            false_prediction_ids
        ),
        reconstruction_recall=(
            recall
        ),
        reconstruction_precision=(
            precision
        ),
        reconstruction_f1=(
            _f1(
                precision,
                recall,
            )
        ),
        ade_m=_mean(
            track_ades
        ),
        fde_m=_mean(
            track_fdes
        ),
        ade_3d_m=_mean(
            track_ades_3d
        ),
        fde_3d_m=_mean(
            track_fdes_3d
        ),
        track_evaluations=tuple(
            track_evaluations
        ),
        horizon_aggregates=tuple(
            horizon_aggregates
        ),
        assignment=assignment,
        runtime_ms=(
            predictions.runtime_ms
        ),
        assignment_future_truth_used=False,
        ignored_prediction_count=len(
            ignored_prediction_ids
        ),
        ignored_prediction_ids=(
            ignored_prediction_ids
        ),
    )

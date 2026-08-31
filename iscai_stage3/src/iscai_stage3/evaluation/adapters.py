from __future__ import annotations

from iscai_stage3.evaluation.contracts import (
    EvaluationPredictionTrack,
    PredictionTrajectoryPoint,
    Vec3,
)


def adapt_prediction_object(
    prediction,
    *,
    anchor_position_H0_m: Vec3,
) -> EvaluationPredictionTrack:
    """
    Duck-typed adapter shared by:

      CV
      CA
      CTRV
      Kalman/EKF
      IMM
      Multidimensional Hough

    No evaluator truth is consumed here.
    """

    if not hasattr(
        prediction,
        "track_id",
    ):
        raise TypeError(
            "Prediction object must expose "
            "track_id."
        )

    if not hasattr(
        prediction,
        "points",
    ):
        raise TypeError(
            "Prediction object must expose "
            "points."
        )

    converted = []

    for point in prediction.points:
        required = (
            "horizon_s",
            "timestamp_s",
            "position_H0_m",
        )

        for field in required:
            if not hasattr(
                point,
                field,
            ):
                raise TypeError(
                    "Prediction point missing "
                    f"{field!r}."
                )

        converted.append(
            PredictionTrajectoryPoint(
                horizon_s=float(
                    point.horizon_s
                ),
                timestamp_s=float(
                    point.timestamp_s
                ),
                position_H0_m=tuple(
                    float(x)
                    for x in (
                        point.position_H0_m
                    )
                ),
            )
        )

    return EvaluationPredictionTrack(
        prediction_id=str(
            prediction.track_id
        ),
        anchor_position_H0_m=tuple(
            float(x)
            for x in anchor_position_H0_m
        ),
        points=tuple(converted),
    )

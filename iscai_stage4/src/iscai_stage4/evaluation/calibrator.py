from __future__ import annotations

from iscai_stage4.predictors import (
    euclidean_error,
    normalized_error,
)


def evaluate_prediction(
    *,
    predicted_positions,
    covariance,
    future_positions,
):
    """
    Evaluate probabilistic prediction.

    Future is used only here,
    not inside predictor.
    """

    if len(predicted_positions) != len(
        future_positions
    ):
        raise ValueError(
            "Prediction/future length mismatch."
        )

    errors = []
    normalized = []


    for pred, real, cov in zip(
        predicted_positions,
        future_positions,
        covariance,
    ):

        error = euclidean_error(
            pred,
            real,
        )

        trace = (
            cov[0][0]
            +
            cov[1][1]
            +
            cov[2][2]
        )

        errors.append(
            error
        )

        normalized.append(
            normalized_error(
                error=error,
                variance_trace=trace,
            )
        )


    return {
        "mean_error":
            sum(errors) / len(errors),

        "mean_normalized_error":
            sum(normalized) / len(normalized),

        "steps":
            len(errors),
    }

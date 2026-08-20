from __future__ import annotations

from iscai_stage4.predictors import (
    predict_gaussian_cv,
)

from iscai_stage4.uncertainty import (
    validate_covariance,
)


def run_stage4_prediction(
    *,
    initial_position,
    velocity,
    horizon_s: float,
    dt: float,
):
    """
    Stage4 probabilistic prediction pipeline.
    """

    prediction = predict_gaussian_cv(
        initial_position=initial_position,
        velocity=velocity,
        horizon_s=horizon_s,
        dt=dt,
    )

    for covariance in prediction.covariance_m2:
        validate_covariance(
            covariance
        )

    return prediction

from __future__ import annotations

from iscai_stage3.baselines.kalman_core import (
    KalmanState,
    spherical_to_cartesian,
)


def update_position_only(
    *,
    predicted: KalmanState,
    measurement_position: tuple[
        float,
        float,
        float,
    ],
    measurement_variance: float = 1.0,
) -> KalmanState:
    """
    Simplified Kalman correction.

    Measurement:

        z = [x,y,z]

    This is the first real update layer.
    Stage2 covariance will replace the default
    variance in the next step.
    """

    if measurement_variance <= 0:
        raise ValueError(
            "Invalid measurement variance."
        )


    x_pred = predicted.position_m


    gain = (
        1.0 /
        (
            1.0
            +
            measurement_variance
        )
    )


    corrected_position = (
        x_pred[0]
        +
        gain
        *
        (
            measurement_position[0]
            -
            x_pred[0]
        ),

        x_pred[1]
        +
        gain
        *
        (
            measurement_position[1]
            -
            x_pred[1]
        ),

        x_pred[2]
        +
        gain
        *
        (
            measurement_position[2]
            -
            x_pred[2]
        ),
    )


    return KalmanState(
        position_m=corrected_position,

        velocity_mps=predicted.velocity_mps,

        covariance=predicted.covariance,
    )

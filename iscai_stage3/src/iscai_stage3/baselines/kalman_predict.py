from __future__ import annotations

from iscai_stage3.baselines.kalman_core import (
    KalmanState,
)



def predict_kalman_cv(
    state: KalmanState,
    dt: float,
    process_noise: float = 0.1,
) -> KalmanState:
    """
    Constant velocity Kalman prediction.

    State:

        [x,y,z,vx,vy,vz]

    No measurement update here.
    """


    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )


    x, y, z = state.position_m

    vx, vy, vz = state.velocity_mps


    predicted_position = (
        x + vx * dt,
        y + vy * dt,
        z + vz * dt,
    )


    predicted_velocity = (
        vx,
        vy,
        vz,
    )


    old = state.covariance


    q = process_noise


    covariance = tuple(
        tuple(
            old[i][j]
            +
            (
                q
                if i == j
                else 0.0
            )
            for j in range(6)
        )
        for i in range(6)
    )


    return KalmanState(
        position_m=predicted_position,
        velocity_mps=predicted_velocity,
        covariance=covariance,
    )

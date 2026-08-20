from __future__ import annotations

from typing import Tuple


Vec3 = Tuple[
    float,
    float,
    float,
]


def predict_kalman_cv(
    position: Vec3,
    velocity: Vec3,
    horizon_s: float,
    dt: float,
    process_noise: float = 0.1,
) -> tuple[Vec3, ...]:
    """
    Lightweight Kalman CV prediction baseline.

    State:
        [x,y,z,vx,vy,vz]

    Prediction only.
    No future observations.
    """

    if horizon_s <= 0:
        raise ValueError(
            "Invalid horizon."
        )

    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )

    if process_noise < 0:
        raise ValueError(
            "Negative process noise."
        )

    x, y, z = position
    vx, vy, vz = velocity

    predictions = []

    t = dt

    while t <= horizon_s:

        # prediction step
        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt

        predictions.append(
            (
                x,
                y,
                z,
            )
        )

        t += dt

    return tuple(predictions)

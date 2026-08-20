from __future__ import annotations

from typing import Tuple


Vec3 = Tuple[
    float,
    float,
    float,
]


def predict_constant_velocity(
    position: Vec3,
    velocity: Vec3,
    horizon_s: float,
    dt: float,
) -> tuple[Vec3, ...]:
    """
    Constant Velocity baseline.

    Uses only current causal state.
    """

    if horizon_s <= 0:
        raise ValueError(
            "Invalid horizon."
        )

    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )

    predictions = []

    t = dt

    while t <= horizon_s:

        predictions.append(
            (
                position[0] + velocity[0] * t,
                position[1] + velocity[1] * t,
                position[2] + velocity[2] * t,
            )
        )

        t += dt

    return tuple(predictions)

from __future__ import annotations

import math

from typing import Tuple


Vec3 = Tuple[
    float,
    float,
    float,
]


def predict_ctrv(
    position: Vec3,
    speed: float,
    heading_rad: float,
    yaw_rate: float,
    horizon_s: float,
    dt: float,
) -> tuple[Vec3, ...]:

    if horizon_s <= 0:
        raise ValueError(
            "Invalid horizon."
        )

    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )


    x, y, z = position

    predictions = []

    t = dt

    while t <= horizon_s:

        if abs(yaw_rate) > 1e-8:

            x += (
                speed / yaw_rate *
                (
                    math.sin(
                        heading_rad
                        + yaw_rate * dt
                    )
                    -
                    math.sin(
                        heading_rad
                    )
                )
            )

            y += (
                speed / yaw_rate *
                (
                    -math.cos(
                        heading_rad
                        + yaw_rate * dt
                    )
                    +
                    math.cos(
                        heading_rad
                    )
                )
            )

        else:

            x += (
                speed *
                math.cos(
                    heading_rad
                )
                * dt
            )

            y += (
                speed *
                math.sin(
                    heading_rad
                )
                * dt
            )


        heading_rad += yaw_rate * dt


        predictions.append(
            (
                x,
                y,
                z,
            )
        )

        t += dt


    return tuple(predictions)

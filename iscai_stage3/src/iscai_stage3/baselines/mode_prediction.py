from __future__ import annotations

import math

from iscai_stage3.baselines.imm_measurement_adapter import (
    cartesian_to_measurement,
)


Vec3 = tuple[
    float,
    float,
    float,
]


def predict_cv_measurement(
    *,
    position: Vec3,
    velocity: Vec3,
    dt: float,
):
    """
    One-step constant-velocity prediction.

    Uses only the state available before the
    measurement being predicted.
    """

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    future_position = (
        position[0] + velocity[0] * dt,
        position[1] + velocity[1] * dt,
        position[2] + velocity[2] * dt,
    )

    return cartesian_to_measurement(
        position=future_position,
        velocity=velocity,
    )


def predict_ca_measurement(
    *,
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    dt: float,
):
    """
    One-step constant-acceleration prediction.
    """

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    future_position = (
        position[0]
        + velocity[0] * dt
        + 0.5 * acceleration[0] * dt * dt,

        position[1]
        + velocity[1] * dt
        + 0.5 * acceleration[1] * dt * dt,

        position[2]
        + velocity[2] * dt
        + 0.5 * acceleration[2] * dt * dt,
    )

    future_velocity = (
        velocity[0]
        + acceleration[0] * dt,

        velocity[1]
        + acceleration[1] * dt,

        velocity[2]
        + acceleration[2] * dt,
    )

    return cartesian_to_measurement(
        position=future_position,
        velocity=future_velocity,
    )


def predict_ctrv_measurement(
    *,
    position: Vec3,
    speed_mps: float,
    heading_rad: float,
    yaw_rate_radps: float,
    vertical_velocity_mps: float,
    dt: float,
):
    """
    One-step CTRV prediction.

    Horizontal motion:
        constant turn rate + constant speed.

    Vertical motion:
        constant vertical velocity.
    """

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    if not all(
        math.isfinite(value)
        for value in (
            *position,
            speed_mps,
            heading_rad,
            yaw_rate_radps,
            vertical_velocity_mps,
        )
    ):
        raise ValueError(
            "CTRV state must be finite."
        )

    x, y, z = position

    if abs(
        yaw_rate_radps
    ) > 1e-8:

        future_x = (
            x
            + speed_mps
            / yaw_rate_radps
            * (
                math.sin(
                    heading_rad
                    + yaw_rate_radps * dt
                )
                - math.sin(
                    heading_rad
                )
            )
        )

        future_y = (
            y
            + speed_mps
            / yaw_rate_radps
            * (
                -math.cos(
                    heading_rad
                    + yaw_rate_radps * dt
                )
                + math.cos(
                    heading_rad
                )
            )
        )

    else:

        future_x = (
            x
            + speed_mps
            * math.cos(
                heading_rad
            )
            * dt
        )

        future_y = (
            y
            + speed_mps
            * math.sin(
                heading_rad
            )
            * dt
        )

    future_z = (
        z
        + vertical_velocity_mps
        * dt
    )

    future_heading = (
        heading_rad
        + yaw_rate_radps * dt
    )

    future_velocity = (
        speed_mps
        * math.cos(
            future_heading
        ),

        speed_mps
        * math.sin(
            future_heading
        ),

        vertical_velocity_mps,
    )

    return cartesian_to_measurement(
        position=(
            future_x,
            future_y,
            future_z,
        ),
        velocity=future_velocity,
    )

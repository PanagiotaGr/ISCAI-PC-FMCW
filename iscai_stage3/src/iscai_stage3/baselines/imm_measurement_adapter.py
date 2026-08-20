from __future__ import annotations

import math


Measurement = tuple[
    float,
    float,
    float,
    float,
]


def cartesian_to_measurement(
    position,
    velocity,
) -> Measurement:
    """
    Convert causal Cartesian state
    into Stage3 IMM measurement space.

    [range, radial_velocity, azimuth, elevation]

    No future information.
    """

    x, y, z = position

    vx, vy, vz = velocity


    if not all(
        math.isfinite(v)
        for v in (
            x,y,z,
            vx,vy,vz,
        )
    ):
        raise ValueError(
            "Non finite state."
        )


    horizontal = math.hypot(
        x,
        y,
    )

    rng = math.sqrt(
        x*x +
        y*y +
        z*z
    )


    if rng <= 0:
        raise ValueError(
            "Zero range."
        )


    azimuth = math.atan2(
        y,
        x,
    )

    elevation = math.atan2(
        z,
        horizontal,
    )


    radial_velocity = (
        x*vx +
        y*vy +
        z*vz
    ) / rng


    return (
        rng,
        radial_velocity,
        azimuth,
        elevation,
    )



def measurement_residual(
    predicted: Measurement,
    measured: Measurement,
) -> Measurement:

    return tuple(
        measured[i]
        -
        predicted[i]
        for i in range(4)
    )

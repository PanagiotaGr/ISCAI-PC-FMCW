from __future__ import annotations

import math

Vec3 = tuple[float, float, float]


def spherical_to_cartesian(
    r: float,
    azimuth: float,
    elevation: float,
) -> Vec3:

    x = (
        r
        * math.cos(elevation)
        * math.cos(azimuth)
    )

    y = (
        r
        * math.cos(elevation)
        * math.sin(azimuth)
    )

    z = (
        r
        * math.sin(elevation)
    )

    return (
        x,
        y,
        z,
    )



def estimate_velocity(
    previous: Vec3,
    current: Vec3,
    dt: float,
) -> Vec3:

    if dt <= 0:
        raise ValueError(
            "Invalid dt"
        )

    return (
        (current[0]-previous[0])/dt,
        (current[1]-previous[1])/dt,
        (current[2]-previous[2])/dt,
    )



def estimate_acceleration(
    v_previous: Vec3,
    v_current: Vec3,
    dt: float,
) -> Vec3:

    if dt <= 0:
        raise ValueError(
            "Invalid dt"
        )

    return (
        (v_current[0]-v_previous[0])/dt,
        (v_current[1]-v_previous[1])/dt,
        (v_current[2]-v_previous[2])/dt,
    )



def build_causal_state(
    observations: tuple[
        tuple[float,float,float],
        ...
    ],
    dt: float,
):

    if len(observations) < 2:
        raise ValueError(
            "Need at least two observations"
        )


    cartesian = tuple(
        spherical_to_cartesian(
            *obs
        )
        for obs in observations
    )


    position = cartesian[-1]


    velocity = estimate_velocity(
        cartesian[-2],
        cartesian[-1],
        dt
    )


    acceleration = (
        0.0,
        0.0,
        0.0,
    )


    if len(cartesian) >= 3:

        v0 = estimate_velocity(
            cartesian[-3],
            cartesian[-2],
            dt
        )

        acceleration = estimate_acceleration(
            v0,
            velocity,
            dt
        )


    return {

        "position":
            position,

        "velocity":
            velocity,

        "acceleration":
            acceleration,

        "history":
            cartesian,

    }

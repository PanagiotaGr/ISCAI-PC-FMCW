from __future__ import annotations

from dataclasses import dataclass
import math


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class CausalMotionState:

    position: Vec3

    velocity: Vec3

    acceleration: Vec3

    heading_rad: float

    yaw_rate_radps: float



def _safe_dt(
    current,
    previous,
):

    dt = current - previous

    if dt <= 0:
        raise ValueError(
            "Invalid timestamp order."
        )

    return dt



def estimate_causal_motion_state(
    *,
    positions: tuple[Vec3, ...],
    timestamps: tuple[float, ...],
):

    if len(positions) < 3:
        raise ValueError(
            "Need at least 3 causal samples."
        )


    p0 = positions[-3]
    p1 = positions[-2]
    p2 = positions[-1]


    t0 = timestamps[-3]
    t1 = timestamps[-2]
    t2 = timestamps[-1]


    dt1 = _safe_dt(
        t1,
        t0,
    )

    dt2 = _safe_dt(
        t2,
        t1,
    )


    velocity = (
        (p2[0]-p1[0]) / dt2,
        (p2[1]-p1[1]) / dt2,
        (p2[2]-p1[2]) / dt2,
    )


    previous_velocity = (
        (p1[0]-p0[0]) / dt1,
        (p1[1]-p0[1]) / dt1,
        (p1[2]-p0[2]) / dt1,
    )


    acceleration = (
        (velocity[0]-previous_velocity[0])
        / dt2,

        (velocity[1]-previous_velocity[1])
        / dt2,

        (velocity[2]-previous_velocity[2])
        / dt2,
    )


    heading = math.atan2(
        velocity[1],
        velocity[0],
    )


    previous_heading = math.atan2(
        previous_velocity[1],
        previous_velocity[0],
    )


    yaw_rate = (
        heading
        -
        previous_heading
    ) / dt2


    return CausalMotionState(
        position=p2,
        velocity=velocity,
        acceleration=acceleration,
        heading_rad=heading,
        yaw_rate_radps=yaw_rate,
    )

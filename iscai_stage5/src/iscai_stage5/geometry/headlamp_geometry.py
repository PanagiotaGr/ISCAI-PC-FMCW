from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadlampState:
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    radial_velocity: float


def world_to_headlamp(
    actor_x: float,
    actor_y: float,
    actor_z: float,
    ego_x: float = 0.0,
    ego_y: float = 0.0,
    ego_z: float = 0.0,
    ego_heading: float = 0.0,
) -> tuple[float, float, float]:

    dx = actor_x - ego_x
    dy = actor_y - ego_y
    dz = actor_z - ego_z

    c = math.cos(ego_heading)
    s = math.sin(ego_heading)

    hx = c * dx + s * dy
    hy = -s * dx + c * dy
    hz = dz

    return hx, hy, hz


def compute_headlamp_state(
    actor_position,
    actor_velocity,
    ego_position=(0.0, 0.0, 0.0),
    ego_velocity=(0.0, 0.0, 0.0),
    ego_heading=0.0,
) -> HeadlampState:

    hx, hy, hz = world_to_headlamp(
        actor_position[0],
        actor_position[1],
        actor_position[2],
        ego_position[0],
        ego_position[1],
        ego_position[2],
        ego_heading,
    )

    range_m = math.sqrt(
        hx * hx
        + hy * hy
        + hz * hz
    )

    azimuth_rad = math.atan2(
        hy,
        hx,
    )

    horizontal_range = math.sqrt(
        hx * hx
        + hy * hy
    )

    elevation_rad = math.atan2(
        hz,
        horizontal_range,
    )

    # Radial velocity must use RELATIVE velocity:
    # actor velocity minus ego velocity.
    rvx = actor_velocity[0] - ego_velocity[0]
    rvy = actor_velocity[1] - ego_velocity[1]
    rvz = actor_velocity[2] - ego_velocity[2]

    # Rotate relative velocity into the same headlamp frame.
    c = math.cos(ego_heading)
    s = math.sin(ego_heading)

    hvx = c * rvx + s * rvy
    hvy = -s * rvx + c * rvy
    hvz = rvz

    if range_m > 1e-9:
        radial_velocity = (
            hvx * hx
            + hvy * hy
            + hvz * hz
        ) / range_m
    else:
        radial_velocity = 0.0

    return HeadlampState(
        range_m=range_m,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        radial_velocity=radial_velocity,
    )

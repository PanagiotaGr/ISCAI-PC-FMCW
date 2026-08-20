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
    actor_x,
    actor_y,
    actor_z,
    ego_x=0.0,
    ego_y=0.0,
    ego_z=0.0,
    ego_heading=0.0,
):
    """
    Transform world coordinates
    to headlamp-centric coordinates.
    """

    dx = actor_x - ego_x
    dy = actor_y - ego_y
    dz = actor_z - ego_z


    c = math.cos(-ego_heading)
    s = math.sin(-ego_heading)

    hx = c * dx - s * dy
    hy = s * dx + c * dy
    hz = dz

    return hx, hy, hz



def compute_headlamp_state(
    actor_position,
    actor_velocity,
    ego_position=(0.0,0.0,0.0),
    ego_heading=0.0,
):

    hx, hy, hz = world_to_headlamp(
        actor_position[0],
        actor_position[1],
        actor_position[2],
        ego_position[0],
        ego_position[1],
        ego_position[2],
        ego_heading,
    )


    r = math.sqrt(
        hx*hx +
        hy*hy +
        hz*hz
    )


    azimuth = math.atan2(
        hy,
        hx
    )


    elevation = math.atan2(
        hz,
        math.sqrt(
            hx*hx + hy*hy
        )
    )


    vr = 0.0

    if r > 1e-6:
        vr = (
            actor_velocity[0]*hx +
            actor_velocity[1]*hy +
            actor_velocity[2]*hz
        ) / r


    return HeadlampState(
        range_m=r,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        radial_velocity=vr,
    )

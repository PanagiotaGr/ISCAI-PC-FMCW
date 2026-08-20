from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class KalmanState:
    position_m: tuple[float,float,float]
    velocity_mps: tuple[float,float,float]
    covariance: tuple[
        tuple[float,...],
        ...
    ]


def spherical_to_cartesian(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
):
    """
    Ht frame:

    x forward
    y left
    z up
    """

    ce = math.cos(elevation_rad)

    return (
        range_m * ce * math.cos(azimuth_rad),
        range_m * ce * math.sin(azimuth_rad),
        range_m * math.sin(elevation_rad),
    )


def radial_velocity_to_vector(
    radial_velocity: float,
    azimuth_rad: float,
    elevation_rad: float,
):
    ce = math.cos(elevation_rad)

    direction = (
        ce * math.cos(azimuth_rad),
        ce * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    )

    return (
        radial_velocity * direction[0],
        radial_velocity * direction[1],
        radial_velocity * direction[2],
    )


def build_measurement_state(
    *,
    range_m: float,
    radial_velocity_mps: float,
    azimuth_rad: float,
    elevation_rad: float,
):
    position = spherical_to_cartesian(
        range_m,
        azimuth_rad,
        elevation_rad,
    )

    velocity = radial_velocity_to_vector(
        radial_velocity_mps,
        azimuth_rad,
        elevation_rad,
    )

    covariance = (
        (1.,0.,0.,0.,0.,0.),
        (0.,1.,0.,0.,0.,0.),
        (0.,0.,1.,0.,0.,0.),
        (0.,0.,0.,1.,0.,0.),
        (0.,0.,0.,0.,1.,0.),
        (0.,0.,0.,0.,0.,1.),
    )

    return KalmanState(
        position_m=position,
        velocity_mps=velocity,
        covariance=covariance,
    )

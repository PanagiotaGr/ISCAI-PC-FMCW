from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeamCoordinate:
    range_m: float
    azimuth_rad: float
    elevation_rad: float


@dataclass(frozen=True)
class BeamOrigin:
    position_H0_m: tuple[float, float, float]


def point_H0_to_beam(
    point_H0_m: tuple[float, float, float],
    origin: BeamOrigin,
) -> BeamCoordinate:
    """
    Convert H0 Cartesian point into beam spherical coordinates.

    H0 convention:
        x = forward
        y = left
        z = up
    """

    dx = (
        point_H0_m[0]
        - origin.position_H0_m[0]
    )

    dy = (
        point_H0_m[1]
        - origin.position_H0_m[1]
    )

    dz = (
        point_H0_m[2]
        - origin.position_H0_m[2]
    )

    r = math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )

    if r <= 0.0:
        raise ValueError(
            "Beam target range must be positive."
        )

    azimuth = math.atan2(
        dy,
        dx,
    )

    elevation = math.atan2(
        dz,
        math.sqrt(
            dx * dx +
            dy * dy
        ),
    )

    return BeamCoordinate(
        range_m=r,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
    )

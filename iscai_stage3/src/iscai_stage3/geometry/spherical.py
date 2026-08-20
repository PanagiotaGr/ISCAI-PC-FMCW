from __future__ import annotations

from math import cos, sin
from typing import Tuple


Vec3 = Tuple[float, float, float]


def spherical_to_cartesian(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
) -> Vec3:
    """
    Convert spherical radar coordinates to H0 Cartesian.

    Convention:
        x forward
        y left
        z up
    """

    ce = cos(elevation_rad)
    se = sin(elevation_rad)

    ca = cos(azimuth_rad)
    sa = sin(azimuth_rad)

    x = range_m * ce * ca
    y = range_m * ce * sa
    z = range_m * se

    return (
        x,
        y,
        z,
    )


def spherical_position_jacobian(
    range_m: float,
    azimuth_rad: float,
    elevation_rad: float,
):
    """
    Jacobian:

        d(x,y,z)/d(r,azimuth,elevation)
    """

    ce = cos(elevation_rad)
    se = sin(elevation_rad)

    ca = cos(azimuth_rad)
    sa = sin(azimuth_rad)


    return (
        (
            ce * ca,
            -range_m * ce * sa,
            -range_m * se * ca,
        ),
        (
            ce * sa,
            range_m * ce * ca,
            -range_m * se * sa,
        ),
        (
            se,
            0.0,
            range_m * ce,
        ),
    )

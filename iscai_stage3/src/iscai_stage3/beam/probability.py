from __future__ import annotations

from math import sqrt, exp
from typing import Tuple


Vec3 = Tuple[
    float,
    float,
    float,
]


Matrix3 = Tuple[
    Tuple[float,float,float],
    Tuple[float,float,float],
    Tuple[float,float,float],
]



def dot(
    a: Vec3,
    b: Vec3,
) -> float:

    return (
        a[0]*b[0] +
        a[1]*b[1] +
        a[2]*b[2]
    )



def subtract(
    a: Vec3,
    b: Vec3,
) -> Vec3:

    return (
        a[0]-b[0],
        a[1]-b[1],
        a[2]-b[2],
    )



def norm(
    v: Vec3,
) -> float:

    return sqrt(
        dot(v,v)
    )



def beam_cross_track_error(
    point_H0_m: Vec3,
    beam_origin_H0_m: Vec3,
    beam_direction_H0: Vec3,
) -> float:
    """
    Distance of target mean from beam axis.
    """

    relative = subtract(
        point_H0_m,
        beam_origin_H0_m,
    )

    longitudinal = dot(
        relative,
        beam_direction_H0,
    )

    closest = (
        beam_direction_H0[0] * longitudinal,
        beam_direction_H0[1] * longitudinal,
        beam_direction_H0[2] * longitudinal,
    )

    residual = subtract(
        relative,
        closest,
    )

    return norm(residual)



def gaussian_beam_confidence(
    error_m: float,
    sigma_m: float,
) -> float:
    """
    Simplified Gaussian beam overlap score.

    1.0 -> centered
    0.0 -> far outside

    """

    if sigma_m <= 0.0:
        raise ValueError(
            "Invalid beam sigma."
        )

    normalized = (
        error_m / sigma_m
    )

    return exp(
        -0.5 * normalized * normalized
    )

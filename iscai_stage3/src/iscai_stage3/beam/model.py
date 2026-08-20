from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Tuple


Vec3 = Tuple[
    float,
    float,
    float,
]


def normalize(
    vector: Vec3,
) -> Vec3:

    norm = sqrt(
        vector[0] ** 2 +
        vector[1] ** 2 +
        vector[2] ** 2
    )

    if norm <= 0.0:
        raise ValueError(
            "Zero beam direction."
        )

    return (
        vector[0] / norm,
        vector[1] / norm,
        vector[2] / norm,
    )



@dataclass(frozen=True)
class BeamGeometry:
    """
    Deterministic beam geometry in H0.

    origin:
        beam emitter position

    direction:
        normalized pointing vector

    divergence_rad:
        half angular spread
    """

    origin_H0_m: Vec3

    direction_H0: Vec3

    divergence_rad: float


    def validate(self):

        d = self.direction_H0

        norm = sqrt(
            d[0]**2 +
            d[1]**2 +
            d[2]**2
        )

        if abs(norm - 1.0) > 1e-9:
            raise ValueError(
                "Beam direction must be normalized."
            )


        if self.divergence_rad < 0.0:
            raise ValueError(
                "Negative divergence."
            )

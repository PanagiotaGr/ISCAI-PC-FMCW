from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Tuple


Vec3 = Tuple[float, float, float]


Matrix4 = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


Matrix6 = Tuple[
    Tuple[float, ...],
    Tuple[float, ...],
    Tuple[float, ...],
    Tuple[float, ...],
    Tuple[float, ...],
    Tuple[float, ...],
]


@dataclass(frozen=True)
class ObservationGaussian:
    """
    Stage-2 observation uncertainty.

    Order:
        [range,
         radial_velocity,
         azimuth,
         elevation]
    """

    range_m: float
    radial_velocity_mps: float
    azimuth_rad: float
    elevation_rad: float

    covariance_4x4: Matrix4

    def validate(self) -> None:
        values = (
            self.range_m,
            self.radial_velocity_mps,
            self.azimuth_rad,
            self.elevation_rad,
        )

        if not all(isfinite(x) for x in values):
            raise ValueError(
                "Observation contains non-finite values."
            )


@dataclass(frozen=True)
class GaussianTargetState:
    """
    Cartesian uncertainty-aware target state.

    Order:

        [px,py,pz,
         vx,vy,vz]
    """

    position_H0_m: Vec3
    velocity_H0_mps: Vec3

    covariance_6x6: Matrix6

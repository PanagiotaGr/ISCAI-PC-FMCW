from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class GaussianTrajectoryPoint:
    """
    Single uncertain trajectory state.

    Frame:
        canonical H0

    State:
        position only.
    """

    timestamp_s: float

    position_H0_m: Vec3

    covariance_H0_m2: Matrix3



@dataclass(frozen=True)
class GaussianTrajectory:

    points: Tuple[
        GaussianTrajectoryPoint,
        ...
    ]

    def validate(self):

        if len(self.points) == 0:
            raise ValueError(
                "Empty trajectory."
            )


        previous = None

        for p in self.points:

            if previous is not None:

                if p.timestamp_s <= previous:
                    raise ValueError(
                        "Trajectory timestamps not increasing."
                    )

            previous = p.timestamp_s



def propagate_constant_velocity(
    position: Vec3,
    velocity: Vec3,
    dt: float,
) -> Vec3:
    """
    Deterministic CV propagation.
    """

    return (
        position[0] + velocity[0]*dt,
        position[1] + velocity[1]*dt,
        position[2] + velocity[2]*dt,
    )

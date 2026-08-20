from __future__ import annotations

from dataclasses import dataclass
import math


Vec3 = tuple[
    float,
    float,
    float,
]


Cov3 = tuple[
    Vec3,
    Vec3,
    Vec3,
]


@dataclass(frozen=True)
class Stage4ActorInput:
    """
    Strictly causal input for Stage 4
    probabilistic trajectory predictor.

    No future information allowed.
    """

    time_index: int

    position: Vec3
    velocity: Vec3

    heading_rad: float

    range_m: float
    radial_velocity_mps: float

    measurement_covariance: Cov3

    length_m: float
    width_m: float
    height_m: float

    actor_class: str


    def __post_init__(self):

        if self.time_index < 0:
            raise ValueError(
                "Invalid time index."
            )


        values = (
            *self.position,
            *self.velocity,
            self.heading_rad,
            self.range_m,
            self.radial_velocity_mps,
            self.length_m,
            self.width_m,
            self.height_m,
        )


        if not all(
            math.isfinite(x)
            for x in values
        ):
            raise ValueError(
                "Non finite input."
            )


        if self.range_m < 0:
            raise ValueError(
                "Negative range."
            )


        if (
            self.length_m <= 0
            or self.width_m <= 0
            or self.height_m <= 0
        ):
            raise ValueError(
                "Invalid dimensions."
            )


        if not self.actor_class:
            raise ValueError(
                "Missing actor class."
            )


    def feature_vector(
        self,
    ) -> tuple[float, ...]:
        """
        Neural predictor feature vector.

        Heading represented by sin/cos.
        """

        return (

            self.position[0],
            self.position[1],
            self.position[2],

            self.velocity[0],
            self.velocity[1],
            self.velocity[2],

            math.sin(
                self.heading_rad
            ),

            math.cos(
                self.heading_rad
            ),

            self.range_m,

            self.radial_velocity_mps,

            self.length_m,
            self.width_m,
            self.height_m,
        )

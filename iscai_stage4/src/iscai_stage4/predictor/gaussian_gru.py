from __future__ import annotations

from dataclasses import dataclass
import math


Vec3 = tuple[
    float,
    float,
    float,
]


Matrix3 = tuple[
    Vec3,
    Vec3,
    Vec3,
]


@dataclass(frozen=True)
class GaussianTrajectory:

    mean: tuple[Vec3, ...]

    covariance: tuple[
        Matrix3,
        ...
    ]


    @property
    def horizon(self):

        return len(
            self.mean
        )


    def validate(self):

        if len(self.mean) != len(
            self.covariance
        ):

            raise ValueError(
                "Mean/covariance mismatch."
            )


        for cov in self.covariance:

            for i in range(3):

                if cov[i][i] <= 0:

                    raise ValueError(
                        "Covariance not positive definite."
                    )



class GaussianGRUPredictor:
    """
    Stage 4 Gaussian predictor.

    MVP implementation.

    Output:
        N(mu, Sigma)

    Later replaced by trained GRU heads.
    """


    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.1,
    ):

        if horizon <= 0:

            raise ValueError(
                "Invalid horizon."
            )

        self.horizon = horizon
        self.dt = dt



    def predict(
        self,
        actor,
    ) -> GaussianTrajectory:


        px, py, pz = actor.position

        vx, vy, vz = actor.velocity


        mean = []

        covariance = []


        for step in range(
            1,
            self.horizon + 1,
        ):

            t = (
                step
                *
                self.dt
            )


            mean.append(
                (
                    px + vx*t,
                    py + vy*t,
                    pz + vz*t,
                )
            )


            # diagonal Cholesky-derived covariance
            sigma = (
                0.5
                +
                0.05*step
            )


            covariance.append(

                (
                    (
                        sigma,
                        0.0,
                        0.0,
                    ),

                    (
                        0.0,
                        sigma,
                        0.0,
                    ),

                    (
                        0.0,
                        0.0,
                        sigma,
                    ),
                )

            )


        result = GaussianTrajectory(

            mean=tuple(mean),

            covariance=tuple(
                covariance
            ),
        )


        result.validate()


        return result

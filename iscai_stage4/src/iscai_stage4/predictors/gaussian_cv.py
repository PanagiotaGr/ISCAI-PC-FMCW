from __future__ import annotations

from iscai_stage4.contracts import (
    GaussianTrajectoryPrediction,
)


Vec3 = tuple[
    float,
    float,
    float,
]


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def predict_gaussian_cv(
    *,
    initial_position: Vec3,
    velocity: Vec3,
    horizon_s: float,
    dt: float,
    initial_variance: float = 0.1,
    variance_growth: float = 0.05,
) -> GaussianTrajectoryPrediction:
    """
    Gaussian Constant Velocity predictor.

    Mean:
        x(t+k)=x0+v*t

    Covariance:
        uncertainty grows with time.
    """

    if horizon_s <= 0:
        raise ValueError(
            "Invalid horizon."
        )

    if dt <= 0:
        raise ValueError(
            "Invalid timestep."
        )

    if initial_variance < 0:
        raise ValueError(
            "Negative variance."
        )

    if variance_growth < 0:
        raise ValueError(
            "Negative variance growth."
        )


    timestamps = []
    means = []
    covariances = []


    t = 0.0

    while t <= horizon_s:

        x0, y0, z0 = initial_position
        vx, vy, vz = velocity


        means.append(
            (
                x0 + vx*t,
                y0 + vy*t,
                z0 + vz*t,
            )
        )


        variance = (
            initial_variance
            +
            variance_growth * t
        )


        covariances.append(
            (
                (
                    variance,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    variance,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    variance,
                ),
            )
        )


        timestamps.append(t)

        t += dt


    return GaussianTrajectoryPrediction(
        timestamps_s=tuple(
            timestamps
        ),
        mean_positions_m=tuple(
            means
        ),
        covariance_m2=tuple(
            covariances
        ),
        model_name="GaussianCV",
    )

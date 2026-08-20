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


def predict_gaussian_ca(
    *,
    initial_position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    horizon_s: float,
    dt: float,
    initial_variance: float = 0.1,
    variance_growth: float = 0.1,
) -> GaussianTrajectoryPrediction:
    """
    Gaussian Constant Acceleration predictor.

    Mean:
        x(t)=x0+v0*t+0.5*a*t^2

    Covariance:
        uncertainty grows with prediction horizon.
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


    x0, y0, z0 = initial_position
    vx, vy, vz = velocity
    ax, ay, az = acceleration


    t = 0.0

    while t <= horizon_s:

        means.append(
            (
                x0 + vx*t + 0.5*ax*t*t,
                y0 + vy*t + 0.5*ay*t*t,
                z0 + vz*t + 0.5*az*t*t,
            )
        )


        variance = (
            initial_variance
            +
            variance_growth*t
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
        timestamps_s=tuple(timestamps),
        mean_positions_m=tuple(means),
        covariance_m2=tuple(covariances),
        model_name="GaussianCA",
    )

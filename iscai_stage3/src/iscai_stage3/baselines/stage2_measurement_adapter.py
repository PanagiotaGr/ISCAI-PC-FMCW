from __future__ import annotations

from dataclasses import dataclass

from iscai_stage3.baselines.kalman_core import (
    spherical_to_cartesian,
)


@dataclass(frozen=True)
class CartesianMeasurement:

    position_m: tuple[
        float,
        float,
        float,
    ]

    variance_m2: float



def observation_to_cartesian_measurement(
    observation,
) -> CartesianMeasurement:
    """
    Stage2 PcfmcwLikeObservation -> Kalman measurement.

    Uses only measurement fields.

    No truth.
    No future.
    """

    vector = observation.measurement_vector()

    range_m = vector[0]
    azimuth = vector[2]
    elevation = vector[3]


    position = spherical_to_cartesian(
        range_m,
        azimuth,
        elevation,
    )


    covariance = observation.covariance


    # First-order Cartesian approximation.
    # Full Jacobian propagation follows next.

    variance = float(
        covariance[0][0]
    )


    return CartesianMeasurement(
        position_m=position,
        variance_m2=variance,
    )

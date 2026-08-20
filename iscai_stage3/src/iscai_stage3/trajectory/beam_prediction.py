from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeamState:
    timestamp_s: float

    azimuth_rad: float
    elevation_rad: float

    azimuth_variance: float
    elevation_variance: float



@dataclass(frozen=True)
class PredictedBeam:
    timestamp_s: float

    azimuth_rad: float
    elevation_rad: float

    azimuth_std_rad: float
    elevation_std_rad: float



def constant_velocity_predict(
    *,
    current: BeamState,
    previous: BeamState,
    timestamp_s: float,
    process_noise_rad2: float = 0.0,
) -> PredictedBeam:
    """
    Constant angular velocity prediction.

    Uses only past states.

    """

    if timestamp_s <= current.timestamp_s:
        raise ValueError(
            "Prediction time must be future."
        )


    dt = (
        current.timestamp_s
        -
        previous.timestamp_s
    )

    if dt <= 0:
        raise ValueError(
            "Invalid timestamps."
        )


    future_dt = (
        timestamp_s
        -
        current.timestamp_s
    )


    az_velocity = (
        current.azimuth_rad
        -
        previous.azimuth_rad
    ) / dt


    el_velocity = (
        current.elevation_rad
        -
        previous.elevation_rad
    ) / dt


    predicted_az = (
        current.azimuth_rad
        +
        az_velocity * future_dt
    )


    predicted_el = (
        current.elevation_rad
        +
        el_velocity * future_dt
    )


    az_var = (
        current.azimuth_variance
        +
        process_noise_rad2
    )

    el_var = (
        current.elevation_variance
        +
        process_noise_rad2
    )


    return PredictedBeam(
        timestamp_s=timestamp_s,

        azimuth_rad=predicted_az,
        elevation_rad=predicted_el,

        azimuth_std_rad=math.sqrt(
            az_var
        ),

        elevation_std_rad=math.sqrt(
            el_var
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
import math


Measurement = tuple[
    float,
    float,
    float,
    float,
]


@dataclass(frozen=True)
class IMMInnovation:

    mode_name: str

    innovation: Measurement

    covariance_trace: float

    log_likelihood: float



def measurement_vector(
    *,
    range_m: float,
    radial_velocity_mps: float,
    azimuth_rad: float,
    elevation_rad: float,
) -> Measurement:

    values = (
        range_m,
        radial_velocity_mps,
        azimuth_rad,
        elevation_rad,
    )

    if not all(
        math.isfinite(v)
        for v in values
    ):
        raise ValueError(
            "Measurement must be finite."
        )

    return values



def innovation_from_prediction(
    *,
    mode_name: str,
    predicted_measurement: Measurement,
    measured: Measurement,
) -> tuple[float, float, float, float]:

    if len(
        predicted_measurement
    ) != 4:
        raise ValueError(
            "Prediction dimension mismatch."
        )

    return tuple(
        measured[i]
        -
        predicted_measurement[i]
        for i in range(4)
    )



def build_innovation_record(
    *,
    mode_name: str,
    predicted_measurement: Measurement,
    measured: Measurement,
    covariance_trace: float,
    log_likelihood: float,
) -> IMMInnovation:

    if covariance_trace <= 0.0:
        raise ValueError(
            "Covariance trace must be positive."
        )

    if not math.isfinite(
        log_likelihood
    ):
        raise ValueError(
            "Invalid log likelihood."
        )

    innovation = (
        innovation_from_prediction(
            mode_name=mode_name,
            predicted_measurement=(
                predicted_measurement
            ),
            measured=measured,
        )
    )

    return IMMInnovation(
        mode_name=mode_name,
        innovation=innovation,
        covariance_trace=(
            covariance_trace
        ),
        log_likelihood=(
            log_likelihood
        ),
    )

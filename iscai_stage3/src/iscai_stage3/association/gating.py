from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from iscai_stage3.association.contracts import (
    AssociationConfig,
)

from iscai_stage3.observations import (
    MeasurementSnapshot,
    wrap_angle_rad,
)


@dataclass(frozen=True)
class PredictedMeasurement:
    range_m: float
    radial_velocity_mps: float
    azimuth_rad: float
    elevation_rad: float


@dataclass(frozen=True)
class AssociationCost:
    feasible: bool
    cost: float

    normalized_range: float
    normalized_radial_velocity: float
    normalized_azimuth: float
    normalized_elevation: float


def predict_measurement(
    *,
    history: tuple[
        tuple[float, MeasurementSnapshot],
        ...
    ],
    target_timestamp_s: float,
) -> PredictedMeasurement:
    if not history:
        raise ValueError(
            "Cannot predict an empty track."
        )

    last_t, last = history[-1]

    dt = (
        float(target_timestamp_s)
        -
        float(last_t)
    )

    if dt <= 0.0:
        raise ValueError(
            "Target timestamp must be later "
            "than the last observation."
        )

    if len(history) == 1:
        return PredictedMeasurement(
            range_m=last.range_m,
            radial_velocity_mps=(
                last.radial_velocity_mps
            ),
            azimuth_rad=last.azimuth_rad,
            elevation_rad=last.elevation_rad,
        )

    prev_t, prev = history[-2]

    history_dt = (
        float(last_t)
        -
        float(prev_t)
    )

    if history_dt <= 0.0:
        raise ValueError(
            "Track history timestamps "
            "must increase."
        )

    range_rate = (
        last.range_m
        -
        prev.range_m
    ) / history_dt

    radial_acceleration = (
        last.radial_velocity_mps
        -
        prev.radial_velocity_mps
    ) / history_dt

    azimuth_rate = (
        wrap_angle_rad(
            last.azimuth_rad
            -
            prev.azimuth_rad
        )
        / history_dt
    )

    elevation_rate = (
        last.elevation_rad
        -
        prev.elevation_rad
    ) / history_dt

    return PredictedMeasurement(
        range_m=(
            last.range_m
            +
            range_rate * dt
        ),
        radial_velocity_mps=(
            last.radial_velocity_mps
            +
            radial_acceleration * dt
        ),
        azimuth_rad=wrap_angle_rad(
            last.azimuth_rad
            +
            azimuth_rate * dt
        ),
        elevation_rad=(
            last.elevation_rad
            +
            elevation_rate * dt
        ),
    )


def _std_sum(
    previous: MeasurementSnapshot,
    current: MeasurementSnapshot,
    index: int,
) -> float:
    variance = (
        previous.covariance_4x4[index][index]
        +
        current.covariance_4x4[index][index]
    )

    if variance < 0.0:
        raise ValueError(
            "Negative combined variance."
        )

    return sqrt(variance)


def association_cost(
    *,
    predicted: PredictedMeasurement,
    previous: MeasurementSnapshot,
    current: MeasurementSnapshot,
    dt_s: float,
    config: AssociationConfig,
) -> AssociationCost:
    if dt_s <= 0.0:
        raise ValueError(
            "Association dt must be positive."
        )

    dr = abs(
        current.range_m
        -
        predicted.range_m
    )

    dvr = abs(
        current.radial_velocity_mps
        -
        predicted.radial_velocity_mps
    )

    daz = abs(
        wrap_angle_rad(
            current.azimuth_rad
            -
            predicted.azimuth_rad
        )
    )

    delv = abs(
        current.elevation_rad
        -
        predicted.elevation_rad
    )

    sigma = (
        config.covariance_sigma_multiplier
    )

    range_gate = (
        config.base_range_gate_m
        +
        config.max_range_rate_mps
        * dt_s
        +
        sigma
        * _std_sum(
            previous,
            current,
            0,
        )
    )

    radial_velocity_gate = (
        config.base_radial_velocity_gate_mps
        +
        config.max_radial_acceleration_mps2
        * dt_s
        +
        sigma
        * _std_sum(
            previous,
            current,
            1,
        )
    )

    azimuth_gate = (
        config.base_azimuth_gate_rad
        +
        config.max_azimuth_rate_radps
        * dt_s
        +
        sigma
        * _std_sum(
            previous,
            current,
            2,
        )
    )

    elevation_gate = (
        config.base_elevation_gate_rad
        +
        config.max_elevation_rate_radps
        * dt_s
        +
        sigma
        * _std_sum(
            previous,
            current,
            3,
        )
    )

    gates = (
        range_gate,
        radial_velocity_gate,
        azimuth_gate,
        elevation_gate,
    )

    if any(g <= 0.0 for g in gates):
        raise ValueError(
            "Association gates must be positive."
        )

    normalized = (
        dr / range_gate,
        dvr / radial_velocity_gate,
        daz / azimuth_gate,
        delv / elevation_gate,
    )

    feasible = all(
        value <= 1.0
        for value in normalized
    )

    cost = sqrt(
        sum(
            value * value
            for value in normalized
        )
        / 4.0
    )

    return AssociationCost(
        feasible=feasible,
        cost=cost,
        normalized_range=normalized[0],
        normalized_radial_velocity=normalized[1],
        normalized_azimuth=normalized[2],
        normalized_elevation=normalized[3],
    )

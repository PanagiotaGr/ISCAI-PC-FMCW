"""Causal Stage-2 observation series -> Stage-3 beam geometry.

This module performs geometry conversion only.

It does not:
- inspect future WOMD states,
- perform trajectory forecasting,
- inject sensing noise,
- perform beam selection,
- perform ADB control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol


class CausalGeometryObservation(Protocol):
    timestamp_s: float
    position_Ht_m: tuple[float, float, float]
    geometry_valid: bool


@dataclass(frozen=True)
class CausalBeamSample:
    timestamp_s: float

    geometry_valid: bool

    range_m: float | None
    azimuth_rad: float | None
    elevation_rad: float | None


def _beam_geometry(
    position_Ht_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert x-forward/y-left/z-up Cartesian geometry to spherical.

    azimuth:
        atan2(y, x)

    elevation:
        atan2(z, sqrt(x^2 + y^2))
    """

    x, y, z = (
        float(position_Ht_m[0]),
        float(position_Ht_m[1]),
        float(position_Ht_m[2]),
    )

    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError(
            "Beam geometry requires finite Cartesian coordinates."
        )

    horizontal_range = math.hypot(x, y)
    range_m = math.sqrt(x * x + y * y + z * z)

    if range_m <= 0.0:
        raise ValueError(
            "Beam geometry is undefined at zero range."
        )

    azimuth_rad = math.atan2(y, x)

    elevation_rad = math.atan2(
        z,
        horizontal_range,
    )

    return (
        range_m,
        azimuth_rad,
        elevation_rad,
    )


def build_causal_beam_series(
    observations: Iterable[CausalGeometryObservation],
) -> tuple[CausalBeamSample, ...]:
    """Build a beam-coordinate series from an already-causal input.

    The function receives no future trajectory object. Therefore future
    actor states cannot influence the transformation performed here.
    """

    observations = tuple(observations)

    result: list[CausalBeamSample] = []

    previous_timestamp: float | None = None

    for observation in observations:
        timestamp_s = float(observation.timestamp_s)

        if not math.isfinite(timestamp_s):
            raise ValueError(
                "Observation timestamp must be finite."
            )

        if (
            previous_timestamp is not None
            and timestamp_s <= previous_timestamp
        ):
            raise ValueError(
                "Causal observation timestamps must be "
                "strictly increasing."
            )

        previous_timestamp = timestamp_s

        if not observation.geometry_valid:
            result.append(
                CausalBeamSample(
                    timestamp_s=timestamp_s,
                    geometry_valid=False,
                    range_m=None,
                    azimuth_rad=None,
                    elevation_rad=None,
                )
            )
            continue

        range_m, azimuth_rad, elevation_rad = _beam_geometry(
            observation.position_Ht_m
        )

        result.append(
            CausalBeamSample(
                timestamp_s=timestamp_s,
                geometry_valid=True,
                range_m=range_m,
                azimuth_rad=azimuth_rad,
                elevation_rad=elevation_rad,
            )
        )

    return tuple(result)
from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)


@dataclass(frozen=True)
class Stage3BeamObservation:
    time_index: int
    timestamp_s: float

    range_m: float | None
    azimuth_rad: float | None
    elevation_rad: float | None

    geometry_valid: bool

    radial_velocity_mps: float | None
    radial_velocity_valid: bool

    measured_fmcw: bool


def build_stage3_beam_series_from_stage2(
    observations: tuple[
        IdealCausalObservable, ...
    ],
) -> tuple[Stage3BeamObservation, ...]:
    """
    Frozen Stage2 -> Stage3 beam representation.

    No sensor model.
    No noise.
    No future access.
    """

    result = []

    previous_time = None

    for observation in observations:

        if previous_time is not None:
            if observation.time_index <= previous_time:
                raise ValueError(
                    "Non-causal time ordering."
                )

        previous_time = observation.time_index

        result.append(
            Stage3BeamObservation(
                time_index=(
                    observation.time_index
                ),
                timestamp_s=(
                    observation.timestamp_s
                ),

                range_m=(
                    observation.range_m
                    if observation.geometry_valid
                    else None
                ),

                azimuth_rad=(
                    observation.azimuth_rad
                    if observation.geometry_valid
                    else None
                ),

                elevation_rad=(
                    observation.elevation_rad
                    if observation.geometry_valid
                    else None
                ),

                geometry_valid=(
                    observation.geometry_valid
                ),

                radial_velocity_mps=(
                    observation.radial_velocity_mps
                ),

                radial_velocity_valid=(
                    observation.radial_velocity_valid
                ),

                measured_fmcw=(
                    observation.measured_fmcw
                ),
            )
        )

    return tuple(result)

from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage2.observations.contracts import (
    MeasurementCovariance,
)
from iscai_stage2.pc_fmcw.crlb import (
    PartACrlbBounds,
    part_a_eq7_crlb,
)


ANGLE_UNCERTAINTY_SOURCE = (
    "explicit_scene_perception_or_angular_sensor_model"
)


@dataclass(frozen=True)
class CrlbMeasurementCovariance:
    sensing_snr_db: float

    crlb: PartACrlbBounds

    azimuth_std_rad: float
    elevation_std_rad: float

    covariance: MeasurementCovariance

    range_velocity_source: str = (
        "part_a_eq7_crlb"
    )

    angle_uncertainty_source: str = (
        ANGLE_UNCERTAINTY_SOURCE
    )


def measurement_covariance_from_part_a_crlb(
    *,
    sensing_snr_db: float,
    azimuth_std_rad: float,
    elevation_std_rad: float,
) -> CrlbMeasurementCovariance:
    """
    Construct Stage-2 covariance in the frozen order:

        [range, radial_velocity, azimuth, elevation]

    No cross-covariance is claimed by this baseline.
    """
    for name, value in (
        (
            "azimuth_std_rad",
            azimuth_std_rad,
        ),
        (
            "elevation_std_rad",
            elevation_std_rad,
        ),
    ):
        if (
            not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(
                f"{name} must be finite "
                "and non-negative."
            )

    crlb = part_a_eq7_crlb(
        sensing_snr_db
    )

    covariance = MeasurementCovariance(
        matrix=(
            (
                crlb.range_variance_m2,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                crlb.radial_velocity_variance_m2ps2,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                azimuth_std_rad ** 2,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                elevation_std_rad ** 2,
            ),
        )
    )

    return CrlbMeasurementCovariance(
        sensing_snr_db=(
            float(sensing_snr_db)
        ),
        crlb=crlb,
        azimuth_std_rad=(
            azimuth_std_rad
        ),
        elevation_std_rad=(
            elevation_std_rad
        ),
        covariance=covariance,
    )

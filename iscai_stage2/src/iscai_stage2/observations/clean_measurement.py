from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage2.observations.contracts import (
    PcfmcwLikeObservation,
)
from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)
from iscai_stage2.pc_fmcw.covariance import (
    CrlbMeasurementCovariance,
    measurement_covariance_from_part_a_crlb,
)
from iscai_stage2.pc_fmcw.sensing_snr import (
    FixedSensingSnrConfig,
    SensingSnrInputs,
    SensingSnrResult,
    fixed_sensing_snr,
)


CLEAN_MODE = "clean_crlb_conditioned"

CLEAN_MODE_SEMANTICS = (
    "noise_free_mean_with_measurement_covariance"
)


@dataclass(frozen=True)
class CleanObservationConfig:
    """
    Explicit clean Stage-2 baseline.

    sensing_snr_db is an experimental baseline setting,
    not a claim that every WOMD actor physically has this SNR.

    Angular standard deviations are explicit assumptions
    because Part-A Range-Doppler does not estimate full
    azimuth/elevation.
    """

    sensing_snr_db: float

    azimuth_std_rad: float
    elevation_std_rad: float

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.sensing_snr_db
        ):
            raise ValueError(
                "sensing_snr_db must be finite."
            )

        for name, value in (
            (
                "azimuth_std_rad",
                self.azimuth_std_rad,
            ),
            (
                "elevation_std_rad",
                self.elevation_std_rad,
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


@dataclass(frozen=True)
class CleanObservationRecord:
    time_index: int
    timestamp_s: float

    ideal_geometry_valid: bool
    ideal_radial_velocity_valid: bool

    measurement_valid: bool

    sensing_snr: SensingSnrResult | None
    covariance_model: (
        CrlbMeasurementCovariance | None
    )

    measurement: (
        PcfmcwLikeObservation | None
    )

    invalid_reason: str | None

    mode: str = CLEAN_MODE
    mode_semantics: str = CLEAN_MODE_SEMANTICS

    sensor_noise_applied: bool = False
    missed_detection_applied: bool = False
    false_alarm_applied: bool = False


def clean_crlb_conditioned_observation(
    *,
    scenario_id: str,
    track_id: str,
    object_class: str,

    ideal: IdealCausalObservable,
    config: CleanObservationConfig,
) -> CleanObservationRecord:
    """
    Convert one ideal causal observation into the clean
    CRLB-conditioned Stage-2 observation.

    The observation mean remains equal to ideal geometry.
    Covariance describes measurement uncertainty.

    No random corruption is applied in this mode.
    """

    if not ideal.geometry_valid:
        return CleanObservationRecord(
            time_index=ideal.time_index,
            timestamp_s=ideal.timestamp_s,

            ideal_geometry_valid=False,
            ideal_radial_velocity_valid=(
                ideal.radial_velocity_valid
            ),

            measurement_valid=False,

            sensing_snr=None,
            covariance_model=None,
            measurement=None,

            invalid_reason=(
                "invalid_geometry"
            ),
        )

    if not ideal.radial_velocity_valid:
        return CleanObservationRecord(
            time_index=ideal.time_index,
            timestamp_s=ideal.timestamp_s,

            ideal_geometry_valid=True,
            ideal_radial_velocity_valid=False,

            measurement_valid=False,

            sensing_snr=None,
            covariance_model=None,
            measurement=None,

            invalid_reason=(
                "invalid_radial_velocity"
            ),
        )

    if (
        ideal.range_m is None
        or ideal.radial_velocity_mps is None
        or ideal.azimuth_rad is None
        or ideal.elevation_rad is None
    ):
        raise RuntimeError(
            "Ideal-valid observation contains missing fields."
        )

    snr = fixed_sensing_snr(
        inputs=SensingSnrInputs(
            range_m=ideal.range_m
        ),
        config=FixedSensingSnrConfig(
            snr_db=config.sensing_snr_db
        ),
    )

    covariance_model = (
        measurement_covariance_from_part_a_crlb(
            sensing_snr_db=snr.snr_db,
            azimuth_std_rad=(
                config.azimuth_std_rad
            ),
            elevation_std_rad=(
                config.elevation_std_rad
            ),
        )
    )

    measurement = PcfmcwLikeObservation(
        scenario_id=scenario_id,
        track_id=track_id,
        object_class=object_class,

        time_index=ideal.time_index,
        timestamp_s=ideal.timestamp_s,

        range_m=ideal.range_m,

        radial_velocity_mps=(
            ideal.radial_velocity_mps
        ),

        azimuth_rad=ideal.azimuth_rad,
        elevation_rad=(
            ideal.elevation_rad
        ),

        covariance=(
            covariance_model.covariance
        ),

        measurement_valid=True,
    )

    return CleanObservationRecord(
        time_index=ideal.time_index,
        timestamp_s=ideal.timestamp_s,

        ideal_geometry_valid=True,
        ideal_radial_velocity_valid=True,

        measurement_valid=True,

        sensing_snr=snr,
        covariance_model=(
            covariance_model
        ),
        measurement=measurement,

        invalid_reason=None,
    )

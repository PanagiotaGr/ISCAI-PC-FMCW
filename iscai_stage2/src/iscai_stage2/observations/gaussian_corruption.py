from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from iscai_stage2.observations.clean_measurement import (
    CleanObservationRecord,
)
from iscai_stage2.observations.contracts import (
    MeasurementCovariance,
    PcfmcwLikeObservation,
)


GAUSSIAN_MODE = "gaussian_crlb_conditioned"

NOISE_GENERATOR = "sha256_box_muller_v1"


@dataclass(frozen=True)
class GaussianCorruptionConfig:
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise TypeError(
                "Gaussian corruption seed must be int."
            )


@dataclass(frozen=True)
class GaussianCorruptedRecord:
    time_index: int
    timestamp_s: float

    measurement_valid: bool

    measurement: PcfmcwLikeObservation | None

    noise_vector: (
        tuple[float, float, float, float]
        | None
    )

    invalid_reason: str | None

    mode: str = GAUSSIAN_MODE
    noise_generator: str = NOISE_GENERATOR

    sensor_noise_applied: bool = True
    missed_detection_applied: bool = False
    false_alarm_applied: bool = False


def _uniform_open_01(
    key: str,
) -> float:
    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    # Strictly inside (0, 1).
    return (
        value + 0.5
    ) / float(1 << 64)


def deterministic_standard_normal(
    *,
    seed: int,
    scenario_id: str,
    track_id: str,
    time_index: int,
    component: str,
) -> float:
    """
    Deterministic Box-Muller standard normal.

    No process-global RNG state is used.
    """
    prefix = (
        f"{NOISE_GENERATOR}|"
        f"{seed}|"
        f"{scenario_id}|"
        f"{track_id}|"
        f"{time_index}|"
        f"{component}"
    )

    u1 = _uniform_open_01(
        prefix + "|u1"
    )
    u2 = _uniform_open_01(
        prefix + "|u2"
    )

    z = (
        math.sqrt(
            -2.0 * math.log(u1)
        )
        * math.cos(
            2.0 * math.pi * u2
        )
    )

    if not math.isfinite(z):
        raise RuntimeError(
            "Generated non-finite Gaussian sample."
        )

    return z


def _diagonal_stds(
    covariance: MeasurementCovariance,
    *,
    tolerance: float = 1e-15,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Current Stage-2 baseline uses diagonal R.

    Do not silently discard future cross-covariances.
    """
    matrix = covariance.matrix

    for i in range(4):
        for j in range(4):
            if (
                i != j
                and abs(
                    matrix[i][j]
                ) > tolerance
            ):
                raise ValueError(
                    "Gaussian baseline currently requires "
                    "diagonal measurement covariance."
                )

    variances = tuple(
        matrix[i][i]
        for i in range(4)
    )

    if any(
        value < 0.0
        or not math.isfinite(value)
        for value in variances
    ):
        raise ValueError(
            "Covariance diagonal must be finite "
            "and non-negative."
        )

    return tuple(
        math.sqrt(value)
        for value in variances
    )  # type: ignore[return-value]


def _wrap_pi(
    angle_rad: float,
) -> float:
    wrapped = (
        angle_rad + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi

    # Keep contract closed at +pi if numerical input
    # happens to represent +pi exactly.
    if (
        wrapped == -math.pi
        and angle_rad > 0.0
    ):
        return math.pi

    return wrapped


def _clip_elevation(
    angle_rad: float,
) -> float:
    return min(
        math.pi / 2.0,
        max(
            -math.pi / 2.0,
            angle_rad,
        ),
    )


def gaussian_corrupt_clean_record(
    *,
    scenario_id: str,
    track_id: str,
    object_class: str,

    clean: CleanObservationRecord,

    config: GaussianCorruptionConfig,
) -> GaussianCorruptedRecord:
    """
    Apply zero-mean Gaussian corruption using the same R_i,t
    carried by the clean CRLB-conditioned observation.

    No missed detection or false alarm is introduced here.
    """

    if not clean.measurement_valid:
        return GaussianCorruptedRecord(
            time_index=clean.time_index,
            timestamp_s=clean.timestamp_s,

            measurement_valid=False,
            measurement=None,
            noise_vector=None,

            invalid_reason=(
                clean.invalid_reason
            ),
        )

    base = clean.measurement

    if base is None:
        raise RuntimeError(
            "Clean-valid record has no measurement."
        )

    stds = _diagonal_stds(
        base.covariance
    )

    components = (
        "range",
        "radial_velocity",
        "azimuth",
        "elevation",
    )

    noise = tuple(
        std
        * deterministic_standard_normal(
            seed=config.seed,
            scenario_id=scenario_id,
            track_id=track_id,
            time_index=clean.time_index,
            component=component,
        )
        for std, component in zip(
            stds,
            components,
        )
    )

    noisy_range = (
        base.range_m + noise[0]
    )

    if noisy_range < 0.0:
        raise RuntimeError(
            "Gaussian range corruption became negative. "
            "Use an explicit bounded-range policy before "
            "operating in such a regime."
        )

    noisy_vr = (
        base.radial_velocity_mps
        + noise[1]
    )

    noisy_azimuth = _wrap_pi(
        base.azimuth_rad
        + noise[2]
    )

    noisy_elevation = _clip_elevation(
        base.elevation_rad
        + noise[3]
    )

    measurement = PcfmcwLikeObservation(
        scenario_id=scenario_id,
        track_id=track_id,
        object_class=object_class,

        time_index=clean.time_index,
        timestamp_s=clean.timestamp_s,

        range_m=noisy_range,
        radial_velocity_mps=noisy_vr,
        azimuth_rad=noisy_azimuth,
        elevation_rad=noisy_elevation,

        covariance=base.covariance,

        measurement_valid=True,
    )

    return GaussianCorruptedRecord(
        time_index=clean.time_index,
        timestamp_s=clean.timestamp_s,

        measurement_valid=True,

        measurement=measurement,
        noise_vector=noise,

        invalid_reason=None,
    )

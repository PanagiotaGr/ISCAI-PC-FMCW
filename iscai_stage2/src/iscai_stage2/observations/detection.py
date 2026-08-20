from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from iscai_stage2.observations.contracts import (
    PcfmcwLikeObservation,
)
from iscai_stage2.observations.gaussian_corruption import (
    GaussianCorruptedRecord,
)


DETECTION_MODEL = (
    "logistic_pd_from_sensing_snr"
)

DETECTION_MODEL_SEMANTICS = (
    "stage2_configurable_detection_assumption_not_part_a"
)

DETECTION_RNG = (
    "sha256_uniform_detection_v1"
)


@dataclass(frozen=True)
class DetectionProbabilityConfig:
    """
    Explicit Stage-2 P_D(SNR) model.

    snr_midpoint_db:
        SNR where P_D is halfway between floor and ceiling.

    transition_width_db:
        Controls sigmoid steepness. Must be > 0.

    No numerical defaults are frozen here because the PDF /
    Part-A baseline does not specify a calibrated P_D curve.
    """

    snr_midpoint_db: float
    transition_width_db: float

    pd_floor: float = 0.0
    pd_ceiling: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.snr_midpoint_db,
            self.transition_width_db,
            self.pd_floor,
            self.pd_ceiling,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Detection parameters must be finite."
            )

        if self.transition_width_db <= 0.0:
            raise ValueError(
                "transition_width_db must be positive."
            )

        if not (
            0.0
            <= self.pd_floor
            <= self.pd_ceiling
            <= 1.0
        ):
            raise ValueError(
                "Require 0 <= pd_floor <= "
                "pd_ceiling <= 1."
            )


@dataclass(frozen=True)
class DetectionFilteredRecord:
    time_index: int
    timestamp_s: float

    upstream_measurement_valid: bool
    measurement_valid: bool

    sensing_snr_db: float | None
    detection_probability: float | None
    detection_uniform: float | None

    measurement: (
        PcfmcwLikeObservation | None
    )

    invalid_reason: str | None

    detection_model: str = DETECTION_MODEL
    detection_model_semantics: str = (
        DETECTION_MODEL_SEMANTICS
    )

    sensor_noise_applied: bool = True

    # True only when a previously valid measurement was
    # actually dropped by the detection draw.
    missed_detection_applied: bool = False

    false_alarm_applied: bool = False


def detection_probability(
    *,
    sensing_snr_db: float,
    config: DetectionProbabilityConfig,
) -> float:
    if not math.isfinite(
        sensing_snr_db
    ):
        raise ValueError(
            "Sensing SNR must be finite."
        )

    x = (
        sensing_snr_db
        - config.snr_midpoint_db
    ) / config.transition_width_db

    # Numerically stable logistic.
    if x >= 0.0:
        exp_neg = math.exp(-x)
        logistic = 1.0 / (
            1.0 + exp_neg
        )
    else:
        exp_pos = math.exp(x)
        logistic = exp_pos / (
            1.0 + exp_pos
        )

    probability = (
        config.pd_floor
        + (
            config.pd_ceiling
            - config.pd_floor
        )
        * logistic
    )

    if not (
        0.0 <= probability <= 1.0
    ):
        raise RuntimeError(
            "Detection probability out of bounds."
        )

    return probability


def deterministic_detection_uniform(
    *,
    seed: int,
    scenario_id: str,
    track_id: str,
    time_index: int,
) -> float:
    """
    Deterministic U[0,1) sample.

    Independent of actor iteration order and process-global RNG.
    """
    key = (
        f"{DETECTION_RNG}|"
        f"{seed}|"
        f"{scenario_id}|"
        f"{track_id}|"
        f"{time_index}"
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    result = value / float(
        1 << 64
    )

    if not (
        0.0 <= result < 1.0
    ):
        raise RuntimeError(
            "Detection uniform out of bounds."
        )

    return result


def apply_missed_detection(
    *,
    scenario_id: str,
    track_id: str,

    gaussian: GaussianCorruptedRecord,

    sensing_snr_db: float,

    probability_config: (
        DetectionProbabilityConfig
    ),

    seed: int,
) -> DetectionFilteredRecord:
    """
    Apply an SNR-conditioned Bernoulli detection decision
    to an already-noisy Stage-2 measurement.

    False alarms are deliberately not handled here.
    """

    if not gaussian.measurement_valid:
        return DetectionFilteredRecord(
            time_index=(
                gaussian.time_index
            ),
            timestamp_s=(
                gaussian.timestamp_s
            ),

            upstream_measurement_valid=False,
            measurement_valid=False,

            sensing_snr_db=None,
            detection_probability=None,
            detection_uniform=None,

            measurement=None,

            invalid_reason=(
                gaussian.invalid_reason
            ),

            missed_detection_applied=False,
        )

    if gaussian.measurement is None:
        raise RuntimeError(
            "Valid Gaussian record has no measurement."
        )

    pd = detection_probability(
        sensing_snr_db=(
            sensing_snr_db
        ),
        config=probability_config,
    )

    uniform = (
        deterministic_detection_uniform(
            seed=seed,
            scenario_id=scenario_id,
            track_id=track_id,
            time_index=(
                gaussian.time_index
            ),
        )
    )

    detected = uniform < pd

    if not detected:
        return DetectionFilteredRecord(
            time_index=(
                gaussian.time_index
            ),
            timestamp_s=(
                gaussian.timestamp_s
            ),

            upstream_measurement_valid=True,
            measurement_valid=False,

            sensing_snr_db=(
                sensing_snr_db
            ),
            detection_probability=pd,
            detection_uniform=uniform,

            measurement=None,

            invalid_reason=(
                "missed_detection"
            ),

            missed_detection_applied=True,
        )

    return DetectionFilteredRecord(
        time_index=(
            gaussian.time_index
        ),
        timestamp_s=(
            gaussian.timestamp_s
        ),

        upstream_measurement_valid=True,
        measurement_valid=True,

        sensing_snr_db=(
            sensing_snr_db
        ),
        detection_probability=pd,
        detection_uniform=uniform,

        measurement=(
            gaussian.measurement
        ),

        invalid_reason=None,

        missed_detection_applied=False,
    )

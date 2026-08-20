from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

from iscai_stage2.observations.contracts import (
    MeasurementCovariance,
    PcfmcwLikeObservation,
)
from iscai_stage2.observations.detection import (
    DetectionFilteredRecord,
)


TRUE_DETECTION = "true_detection"
FALSE_ALARM = "false_alarm"

FALSE_ALARM_MODEL = (
    "poisson_uniform_measurement_volume_v1"
)

FALSE_ALARM_MODEL_SEMANTICS = (
    "stage2_configurable_false_alarm_assumption_not_part_a"
)

DETECTION_KEY_VERSION = (
    "frame_local_opaque_detection_key_v1"
)

FRAME_ORDER_VERSION = (
    "sha256_detection_order_v1"
)


@dataclass(frozen=True)
class UnlabeledDetection:
    """
    Algorithm-facing detection.

    Deliberately contains:
      - no WOMD track ID,
      - no actor class,
      - no true/false source label,
      - no persistent cross-frame actor identity.

    detection_key is only a frame-local opaque bookkeeping key.
    """

    detection_key: str

    range_m: float
    radial_velocity_mps: float
    azimuth_rad: float
    elevation_rad: float

    covariance: MeasurementCovariance

    def measurement_vector(
        self,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        return (
            self.range_m,
            self.radial_velocity_mps,
            self.azimuth_rad,
            self.elevation_rad,
        )


@dataclass(frozen=True)
class UnlabeledDetectionFrame:
    """
    This is the object intended for tracking/prediction algorithms.
    """

    scenario_id: str
    time_index: int
    timestamp_s: float

    detections: tuple[
        UnlabeledDetection, ...
    ]

    association_information_present: bool = False
    truth_labels_present: bool = False


@dataclass(frozen=True)
class DetectionTruthEntry:
    """
    Evaluator-only metadata.

    MUST NOT be passed to MHT/Hough, Kalman, IMM or predictor input.
    """

    detection_key: str
    source_type: str

    source_track_id: str | None
    source_object_class: str | None


@dataclass(frozen=True)
class DetectionTruthSidecar:
    scenario_id: str
    time_index: int
    timestamp_s: float

    entries: tuple[
        DetectionTruthEntry, ...
    ]

    evaluator_only: bool = True


@dataclass(frozen=True)
class DetectionFrameBundle:
    """
    Explicit separation between algorithm input and scoring truth.
    """

    frame: UnlabeledDetectionFrame
    truth: DetectionTruthSidecar


@dataclass(frozen=True)
class FalseAlarmConfig:
    """
    Explicit Stage-2 false-alarm model.

    Count:
        N_FA ~ Poisson(mean_false_alarms_per_frame)

    Measurement coordinates:
        independently uniform inside the configured measurement volume.

    This is a configurable degraded-mode assumption, not a claim that
    it reproduces a calibrated Part-A CFAR false-alarm process.
    """

    seed: int

    mean_false_alarms_per_frame: float

    range_min_m: float
    range_max_m: float

    radial_velocity_min_mps: float
    radial_velocity_max_mps: float

    azimuth_min_rad: float
    azimuth_max_rad: float

    elevation_min_rad: float
    elevation_max_rad: float

    range_std_m: float
    radial_velocity_std_mps: float
    azimuth_std_rad: float
    elevation_std_rad: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.seed,
            int,
        ):
            raise TypeError(
                "False-alarm seed must be int."
            )

        if (
            not math.isfinite(
                self.mean_false_alarms_per_frame
            )
            or
            self.mean_false_alarms_per_frame
            < 0.0
        ):
            raise ValueError(
                "mean_false_alarms_per_frame must "
                "be finite and non-negative."
            )

        bounds = (
            (
                "range",
                self.range_min_m,
                self.range_max_m,
            ),
            (
                "radial_velocity",
                self.radial_velocity_min_mps,
                self.radial_velocity_max_mps,
            ),
            (
                "azimuth",
                self.azimuth_min_rad,
                self.azimuth_max_rad,
            ),
            (
                "elevation",
                self.elevation_min_rad,
                self.elevation_max_rad,
            ),
        )

        for name, low, high in bounds:
            if not (
                math.isfinite(low)
                and math.isfinite(high)
                and low < high
            ):
                raise ValueError(
                    f"{name} bounds must be finite "
                    "and strictly increasing."
                )

        if self.range_min_m < 0.0:
            raise ValueError(
                "False-alarm range cannot be negative."
            )

        if not (
            -math.pi
            <= self.azimuth_min_rad
            < self.azimuth_max_rad
            <= math.pi
        ):
            raise ValueError(
                "False-alarm azimuth bounds must lie "
                "inside [-pi, pi]."
            )

        if not (
            -math.pi / 2.0
            <= self.elevation_min_rad
            < self.elevation_max_rad
            <= math.pi / 2.0
        ):
            raise ValueError(
                "False-alarm elevation bounds must lie "
                "inside [-pi/2, pi/2]."
            )

        stds = (
            self.range_std_m,
            self.radial_velocity_std_mps,
            self.azimuth_std_rad,
            self.elevation_std_rad,
        )

        if any(
            not math.isfinite(value)
            or value < 0.0
            for value in stds
        ):
            raise ValueError(
                "False-alarm covariance standard "
                "deviations must be finite and "
                "non-negative."
            )


@dataclass(frozen=True)
class AssociatedFilteredRecord:
    """
    Adapter-side object only.

    Track/class metadata is consumed to create evaluator truth,
    then stripped from the algorithm-facing detection.
    """

    track_id: str
    object_class: str
    record: DetectionFilteredRecord


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

    return (
        value + 0.5
    ) / float(1 << 64)


def _uniform_closed_open(
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

    return value / float(1 << 64)


def _uniform_interval(
    *,
    key: str,
    low: float,
    high: float,
) -> float:
    u = _uniform_closed_open(
        key
    )

    return low + (
        high - low
    ) * u


def deterministic_poisson(
    *,
    mean: float,
    key: str,
) -> int:
    """
    Knuth Poisson sampler driven by deterministic SHA256 uniforms.

    Appropriate for the low/moderate false-alarm rates used by
    the Stage-2 frame-level degraded-mode experiments.
    """

    if (
        not math.isfinite(mean)
        or mean < 0.0
    ):
        raise ValueError(
            "Poisson mean must be finite "
            "and non-negative."
        )

    if mean == 0.0:
        return 0

    threshold = math.exp(
        -mean
    )

    product = 1.0
    k = 0

    while product > threshold:
        k += 1

        product *= _uniform_open_01(
            f"{key}|poisson|{k}"
        )

        if k > 100_000:
            raise RuntimeError(
                "Poisson sampler exceeded safety limit."
            )

    return k - 1


def _opaque_detection_key(
    *,
    scenario_id: str,
    time_index: int,
    source_token: str,
) -> str:
    """
    Hash includes time_index, so the key cannot act as a
    persistent actor identity across frames.
    """

    raw = (
        f"{DETECTION_KEY_VERSION}|"
        f"{scenario_id}|"
        f"{time_index}|"
        f"{source_token}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def _false_alarm_covariance(
    config: FalseAlarmConfig,
) -> MeasurementCovariance:
    return MeasurementCovariance(
        matrix=(
            (
                config.range_std_m ** 2,
                0.0,
                0.0,
                0.0,
            ),
            (
                0.0,
                config.radial_velocity_std_mps ** 2,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.0,
                config.azimuth_std_rad ** 2,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.0,
                config.elevation_std_rad ** 2,
            ),
        )
    )


def _true_detection(
    *,
    scenario_id: str,
    track_id: str,
    object_class: str,
    record: DetectionFilteredRecord,
) -> tuple[
    UnlabeledDetection,
    DetectionTruthEntry,
] | None:
    if not record.measurement_valid:
        return None

    measurement = record.measurement

    if measurement is None:
        raise RuntimeError(
            "Detection-valid record has no measurement."
        )

    key = _opaque_detection_key(
        scenario_id=scenario_id,
        time_index=record.time_index,

        # Oracle ID is used only inside the one-way hash
        # that creates a frame-local opaque key.
        source_token=(
            f"true|{track_id}"
        ),
    )

    detection = UnlabeledDetection(
        detection_key=key,

        range_m=measurement.range_m,
        radial_velocity_mps=(
            measurement.radial_velocity_mps
        ),
        azimuth_rad=(
            measurement.azimuth_rad
        ),
        elevation_rad=(
            measurement.elevation_rad
        ),

        covariance=(
            measurement.covariance
        ),
    )

    truth = DetectionTruthEntry(
        detection_key=key,
        source_type=TRUE_DETECTION,
        source_track_id=track_id,
        source_object_class=(
            object_class
        ),
    )

    return detection, truth


def _false_alarm(
    *,
    scenario_id: str,
    time_index: int,
    timestamp_s: float,
    false_alarm_index: int,
    config: FalseAlarmConfig,
) -> tuple[
    UnlabeledDetection,
    DetectionTruthEntry,
]:
    prefix = (
        f"{FALSE_ALARM_MODEL}|"
        f"{config.seed}|"
        f"{scenario_id}|"
        f"{time_index}|"
        f"{timestamp_s:.17g}|"
        f"{false_alarm_index}"
    )

    key = _opaque_detection_key(
        scenario_id=scenario_id,
        time_index=time_index,
        source_token=(
            f"false_alarm|"
            f"{config.seed}|"
            f"{false_alarm_index}"
        ),
    )

    detection = UnlabeledDetection(
        detection_key=key,

        range_m=_uniform_interval(
            key=prefix + "|range",
            low=config.range_min_m,
            high=config.range_max_m,
        ),

        radial_velocity_mps=(
            _uniform_interval(
                key=prefix + "|vr",
                low=(
                    config.radial_velocity_min_mps
                ),
                high=(
                    config.radial_velocity_max_mps
                ),
            )
        ),

        azimuth_rad=_uniform_interval(
            key=prefix + "|azimuth",
            low=config.azimuth_min_rad,
            high=config.azimuth_max_rad,
        ),

        elevation_rad=(
            _uniform_interval(
                key=prefix + "|elevation",
                low=config.elevation_min_rad,
                high=config.elevation_max_rad,
            )
        ),

        covariance=(
            _false_alarm_covariance(
                config
            )
        ),
    )

    truth = DetectionTruthEntry(
        detection_key=key,
        source_type=FALSE_ALARM,

        source_track_id=None,
        source_object_class=None,
    )

    return detection, truth


def _frame_order_key(
    *,
    seed: int,
    scenario_id: str,
    time_index: int,
    detection_key: str,
) -> str:
    """
    Prevent true detections and false alarms from being exposed
    in source-generation order.
    """

    raw = (
        f"{FRAME_ORDER_VERSION}|"
        f"{seed}|"
        f"{scenario_id}|"
        f"{time_index}|"
        f"{detection_key}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def build_unlabeled_detection_frame(
    *,
    scenario_id: str,
    time_index: int,
    timestamp_s: float,

    associated_records: Sequence[
        AssociatedFilteredRecord
    ],

    false_alarm_config: (
        FalseAlarmConfig | None
    ) = None,
) -> DetectionFrameBundle:
    """
    Build the common frame-level observation interface.

    The returned `frame` is algorithm-facing.
    The returned `truth` is evaluator-only.
    """

    if time_index < 0:
        raise ValueError(
            "time_index must be non-negative."
        )

    if not math.isfinite(
        timestamp_s
    ):
        raise ValueError(
            "timestamp_s must be finite."
        )

    detections: list[
        UnlabeledDetection
    ] = []

    truth_entries: list[
        DetectionTruthEntry
    ] = []

    for item in associated_records:
        record = item.record

        if record.time_index != time_index:
            raise ValueError(
                "Associated record time_index "
                "does not match frame."
            )

        if not math.isclose(
            record.timestamp_s,
            timestamp_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Associated record timestamp "
                "does not match frame."
            )

        pair = _true_detection(
            scenario_id=scenario_id,
            track_id=item.track_id,
            object_class=item.object_class,
            record=record,
        )

        if pair is None:
            continue

        detection, truth = pair

        detections.append(
            detection
        )

        truth_entries.append(
            truth
        )

    order_seed = 0

    if false_alarm_config is not None:
        order_seed = (
            false_alarm_config.seed
        )

        count_key = (
            f"{FALSE_ALARM_MODEL}|"
            f"{false_alarm_config.seed}|"
            f"{scenario_id}|"
            f"{time_index}|"
            f"{timestamp_s:.17g}|count"
        )

        false_alarm_count = (
            deterministic_poisson(
                mean=(
                    false_alarm_config
                    .mean_false_alarms_per_frame
                ),
                key=count_key,
            )
        )

        for index in range(
            false_alarm_count
        ):
            detection, truth = (
                _false_alarm(
                    scenario_id=scenario_id,
                    time_index=time_index,
                    timestamp_s=timestamp_s,
                    false_alarm_index=index,
                    config=(
                        false_alarm_config
                    ),
                )
            )

            detections.append(
                detection
            )

            truth_entries.append(
                truth
            )

    # Algorithm-facing order is source-agnostic.
    detections.sort(
        key=lambda detection:
            _frame_order_key(
                seed=order_seed,
                scenario_id=scenario_id,
                time_index=time_index,
                detection_key=(
                    detection.detection_key
                ),
            )
    )

    # Evaluator truth is separately indexed by opaque key.
    truth_entries.sort(
        key=lambda entry:
            entry.detection_key
    )

    return DetectionFrameBundle(
        frame=UnlabeledDetectionFrame(
            scenario_id=scenario_id,
            time_index=time_index,
            timestamp_s=timestamp_s,
            detections=tuple(
                detections
            ),
        ),

        truth=DetectionTruthSidecar(
            scenario_id=scenario_id,
            time_index=time_index,
            timestamp_s=timestamp_s,
            entries=tuple(
                truth_entries
            ),
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


Vec3 = tuple[
    float,
    float,
    float,
]


def _validate_vec3(
    value: Vec3,
    *,
    name: str,
) -> None:
    if len(value) != 3:
        raise ValueError(
            f"{name} must contain 3 values."
        )

    if not all(
        isfinite(float(x))
        for x in value
    ):
        raise ValueError(
            f"{name} must be finite."
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Common Stage3 deterministic trajectory
    evaluation policy.

    Assignment uses current/anchor truth only.
    Future truth is scoring-only.
    """

    horizons_s: tuple[
        float,
        ...
    ] = (
        0.1,
        0.3,
        0.5,
        1.0,
    )

    anchor_assignment_gate_m: float = 5.0

    endpoint_miss_threshold_m: float = 2.0

    horizon_tolerance_s: float = 1e-6

    anchor_timestamp_tolerance_s: float = 2e-3

    def __post_init__(self) -> None:
        if not self.horizons_s:
            raise ValueError(
                "Evaluation horizons cannot "
                "be empty."
            )

        previous = None

        for horizon in self.horizons_s:
            value = float(horizon)

            if (
                not isfinite(value)
                or
                value <= 0.0
            ):
                raise ValueError(
                    "Evaluation horizons must "
                    "be finite and positive."
                )

            if (
                previous is not None
                and
                value <= previous
            ):
                raise ValueError(
                    "Evaluation horizons must "
                    "strictly increase."
                )

            previous = value

        positive = (
            self.anchor_assignment_gate_m,
            self.endpoint_miss_threshold_m,
            self.horizon_tolerance_s,
            self.anchor_timestamp_tolerance_s,
        )

        if any(
            (
                not isfinite(float(value))
                or
                float(value) <= 0.0
            )
            for value in positive
        ):
            raise ValueError(
                "Evaluation gates/tolerances "
                "must be finite and positive."
            )


@dataclass(frozen=True)
class TruthTrajectoryPoint:
    """
    Evaluator-only WOMD future truth.

    horizon_s is the nominal evaluation
    horizon. timestamp_s preserves the actual
    WOMD timestamp.
    """

    horizon_s: float
    timestamp_s: float
    position_H0_m: Vec3

    def __post_init__(self) -> None:
        if (
            not isfinite(self.horizon_s)
            or
            self.horizon_s <= 0.0
        ):
            raise ValueError(
                "Truth horizon must be "
                "positive and finite."
            )

        if not isfinite(
            self.timestamp_s
        ):
            raise ValueError(
                "Truth timestamp must "
                "be finite."
            )

        _validate_vec3(
            self.position_H0_m,
            name="truth position",
        )


@dataclass(frozen=True)
class EvaluationTruthTrack:
    """
    Truth identity exists only inside the
    evaluator sidecar.

    It must never be supplied to a Stage3
    algorithm.
    """

    truth_id: str

    actor_class: str

    anchor_position_H0_m: Vec3

    points: tuple[
        TruthTrajectoryPoint,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.truth_id:
            raise ValueError(
                "Truth ID cannot be empty."
            )

        if not self.actor_class:
            raise ValueError(
                "Actor class cannot be empty."
            )

        _validate_vec3(
            self.anchor_position_H0_m,
            name="truth anchor position",
        )

        horizons = [
            point.horizon_s
            for point in self.points
        ]

        if len(horizons) != len(
            set(horizons)
        ):
            raise ValueError(
                "Truth horizons must be "
                "unique per track."
            )


@dataclass(frozen=True)
class EvaluationTruth:
    """
    Separate evaluator-only future-truth
    sidecar.
    """

    scenario_id: str

    anchor_timestamp_s: float

    tracks: tuple[
        EvaluationTruthTrack,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "Truth scenario ID cannot "
                "be empty."
            )

        if not isfinite(
            self.anchor_timestamp_s
        ):
            raise ValueError(
                "Truth anchor timestamp must "
                "be finite."
            )

        ids = [
            track.truth_id
            for track in self.tracks
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Truth IDs must be unique."
            )


@dataclass(frozen=True)
class PredictionTrajectoryPoint:
    horizon_s: float

    timestamp_s: float

    position_H0_m: Vec3

    def __post_init__(self) -> None:
        if (
            not isfinite(self.horizon_s)
            or
            self.horizon_s <= 0.0
        ):
            raise ValueError(
                "Prediction horizon must be "
                "positive and finite."
            )

        if not isfinite(
            self.timestamp_s
        ):
            raise ValueError(
                "Prediction timestamp must "
                "be finite."
            )

        _validate_vec3(
            self.position_H0_m,
            name="prediction position",
        )


@dataclass(frozen=True)
class EvaluationPredictionTrack:
    """
    Algorithm output packaged for evaluation.

    prediction_id is an algorithm-generated
    identifier. It is not WOMD truth identity.
    """

    prediction_id: str

    anchor_position_H0_m: Vec3

    points: tuple[
        PredictionTrajectoryPoint,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError(
                "Prediction ID cannot "
                "be empty."
            )

        _validate_vec3(
            self.anchor_position_H0_m,
            name="prediction anchor position",
        )

        horizons = [
            point.horizon_s
            for point in self.points
        ]

        if len(horizons) != len(
            set(horizons)
        ):
            raise ValueError(
                "Prediction horizons must "
                "be unique per track."
            )


@dataclass(frozen=True)
class EvaluationPredictionSet:
    method: str

    scenario_id: str

    anchor_timestamp_s: float

    tracks: tuple[
        EvaluationPredictionTrack,
        ...
    ]

    runtime_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError(
                "Method name cannot be empty."
            )

        if not self.scenario_id:
            raise ValueError(
                "Prediction scenario ID "
                "cannot be empty."
            )

        if not isfinite(
            self.anchor_timestamp_s
        ):
            raise ValueError(
                "Prediction anchor timestamp "
                "must be finite."
            )

        ids = [
            track.prediction_id
            for track in self.tracks
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Prediction IDs must "
                "be unique."
            )

        # Every prediction point must refer to the same
        # global anchor carried by EvaluationPredictionSet.
        # This prevents stale-track forecasts from being
        # relabeled as if they started at the WOMD anchor.
        timestamp_tolerance_s = 2e-3

        for track in self.tracks:
            for point in track.points:
                expected_timestamp_s = (
                    float(self.anchor_timestamp_s)
                    + float(point.horizon_s)
                )

                if abs(
                    float(point.timestamp_s)
                    - expected_timestamp_s
                ) > timestamp_tolerance_s:
                    raise ValueError(
                        "Prediction point timestamp is not "
                        "aligned to the prediction-set "
                        "global anchor."
                    )

        if (
            self.runtime_ms is not None
            and
            (
                not isfinite(
                    self.runtime_ms
                )
                or
                self.runtime_ms < 0.0
            )
        ):
            raise ValueError(
                "Runtime must be finite "
                "and non-negative."
            )

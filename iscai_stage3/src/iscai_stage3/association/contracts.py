from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.observations import (
    MeasurementSnapshot,
)


@dataclass(frozen=True)
class AssociationConfig:
    """
    Deterministic gated greedy nearest-neighbor
    association in Stage2 measurement space.

    These are Stage3 baseline parameters, not
    measured sensor specifications.
    """

    base_range_gate_m: float = 1.0
    max_range_rate_mps: float = 60.0

    base_radial_velocity_gate_mps: float = 2.0
    max_radial_acceleration_mps2: float = 15.0

    base_azimuth_gate_rad: float = 0.03490658503988659
    max_azimuth_rate_radps: float = 3.0

    base_elevation_gate_rad: float = 0.03490658503988659
    max_elevation_rate_radps: float = 1.5

    covariance_sigma_multiplier: float = 4.0

    max_missed_frames: int = 2

    def __post_init__(self) -> None:
        numeric = (
            self.base_range_gate_m,
            self.max_range_rate_mps,
            self.base_radial_velocity_gate_mps,
            self.max_radial_acceleration_mps2,
            self.base_azimuth_gate_rad,
            self.max_azimuth_rate_radps,
            self.base_elevation_gate_rad,
            self.max_elevation_rate_radps,
            self.covariance_sigma_multiplier,
        )

        if not all(
            isfinite(float(x))
            for x in numeric
        ):
            raise ValueError(
                "Association configuration "
                "must be finite."
            )

        if any(float(x) < 0.0 for x in numeric):
            raise ValueError(
                "Association gates/rates "
                "cannot be negative."
            )

        if self.covariance_sigma_multiplier <= 0:
            raise ValueError(
                "Covariance multiplier "
                "must be positive."
            )

        if self.max_missed_frames < 0:
            raise ValueError(
                "max_missed_frames cannot "
                "be negative."
            )


@dataclass(frozen=True)
class AssociatedDetection:
    """
    Stage3-generated association identity.

    track_id is NOT a WOMD/Stage2 truth ID.
    """

    track_id: str
    frame_index: int
    timestamp_s: float

    detection: MeasurementSnapshot


@dataclass(frozen=True)
class AssociatedFrame:
    frame_index: int
    timestamp_s: float

    detections: tuple[
        AssociatedDetection,
        ...
    ]


@dataclass(frozen=True)
class AssociatedTrack:
    track_id: str

    detections: tuple[
        AssociatedDetection,
        ...
    ]

    terminated: bool


@dataclass(frozen=True)
class EstimatedAssociationResult:
    scenario_id: str

    frames: tuple[
        AssociatedFrame,
        ...
    ]

    tracks: tuple[
        AssociatedTrack,
        ...
    ]

    method: str = (
        "deterministic_gated_greedy_"
        "nearest_neighbor"
    )

    truth_used: bool = False

    def __post_init__(self) -> None:
        if self.truth_used:
            raise ValueError(
                "Main estimated association "
                "must never use evaluator truth."
            )

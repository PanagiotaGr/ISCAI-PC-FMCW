from __future__ import annotations

from dataclasses import dataclass

from iscai_stage1.contracts.stage1a import Vec3


@dataclass(frozen=True)
class ScalarStats:
    minimum: float
    maximum: float
    mean: float
    std: float
    median: float
    variance: float


@dataclass(frozen=True)
class DecodedActorLidarPoint:
    point_H0_m: Vec3
    range_m: float
    intensity: float
    elongation: float


@dataclass(frozen=True)
class ActorLidarPointSummary:
    point_count: int
    has_points: bool

    range_stats: ScalarStats | None
    intensity_stats: ScalarStats | None
    elongation_stats: ScalarStats | None

    spatial_std_H0_m: Vec3 | None


@dataclass(frozen=True)
class ActorLidarFrameFeatures:
    time_index: int
    timestamp_s: float

    actor_state_valid: bool

    point_count: int

    range_m: ScalarStats | None
    intensity: ScalarStats | None
    elongation: ScalarStats | None

    # Same-time actor-oriented WOMD-box coordinates.
    spatial_std_actor_m: Vec3 | None


@dataclass(frozen=True)
class CausalActorLidarArtifact:
    scenario_id: str
    track_index: int
    track_id: str

    frames: tuple[ActorLidarFrameFeatures, ...]

    association_mode: str = "oracle_womd_track_id"
    lidar_actor_assignment_mode: str = "oracle_causal_box"

    artifact_semantics: str = "causal_womd_annotation_upstream"
    sensor_realistic: bool = False

from __future__ import annotations

from dataclasses import dataclass

from iscai_stage1.contracts.stage1a import Vec3


@dataclass(frozen=True)
class StaticMapFeature:
    feature_id: int
    kind: str

    points_W_m: tuple[Vec3, ...]
    points_H0_m: tuple[Vec3, ...]

    type_name: str | None = None

    speed_limit_mph: float | None = None
    interpolating: bool | None = None

    entry_lane_ids: tuple[int, ...] = ()
    exit_lane_ids: tuple[int, ...] = ()

    stop_sign_lane_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class DynamicLaneState:
    lane_id: int
    state_name: str

    stop_point_W_m: Vec3 | None
    stop_point_H0_m: Vec3 | None


@dataclass(frozen=True)
class CausalDynamicMapFrame:
    time_index: int
    timestamp_s: float
    lane_states: tuple[DynamicLaneState, ...]


@dataclass(frozen=True)
class CausalMapArtifact:
    scenario_id: str
    anchor_index: int

    static_features: tuple[StaticMapFeature, ...]
    dynamic_frames: tuple[CausalDynamicMapFrame, ...]

    artifact_semantics: str = "causal_womd_annotation_upstream"
    sensor_realistic: bool = False

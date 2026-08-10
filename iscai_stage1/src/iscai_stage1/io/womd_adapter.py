"""Dependency-free WOMD -> Stage-1A causal scene adapter.

This module intentionally does NOT import waymo_open_dataset.

It consumes an already-parsed Scenario-like object. Actual TFRecord/protobuf
parsing remains the responsibility of the frozen Stage-0 reader.

Important:
- only t <= current_time_index is inspected,
- annotated WOMD velocity is never read,
- tracks_to_predict and objects_of_interest are never read,
- track IDs remain metadata only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from iscai_stage1.actors.artifact import (
    CausalActorArtifact,
    build_causal_actor_artifact,
)
from iscai_stage1.actors.history import RawObjectStateW
from iscai_stage1.artifacts.hashing import causal_actor_payload
from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
    ObjectClass,
    ReceiverGeometryConfig,
    ReceiverGeometryH0,
)
from iscai_stage1.geometry.frames import (
    AnchorFrames,
    SdcStateW,
    build_anchor_frames,
)
from iscai_stage1.geometry.receiver import (
    ActorAnchorStateW,
    receiver_geometry_in_H0,
)
from iscai_stage1.maps.static_map import (
    StaticMapFeature,
    canonicalize_static_map,
)
from iscai_stage1.maps.dynamic_map import (
    CausalDynamicMapFrame,
    canonicalize_causal_dynamic_map,
)


@dataclass(frozen=True)
class ReceiverRecord:
    """Receiver metadata + deterministic Stage-1 anchor geometry.

    track_id is bookkeeping metadata, not a predictor feature.
    """

    track_id: int
    geometry: ReceiverGeometryH0


@dataclass(frozen=True)
class Stage1aScene:
    scenario_id: str
    anchor_index: int
    sdc_track_index: int
    anchor_frames: AnchorFrames
    actors: tuple[CausalActorArtifact, ...]
    receivers: tuple[ReceiverRecord, ...]
    static_map: tuple[StaticMapFeature, ...]
    dynamic_map: tuple[CausalDynamicMapFrame, ...]


def _object_class_from_womd_track(track: Any) -> ObjectClass:
    """Resolve WOMD object type through the proto descriptor.

    No numeric enum mapping is hard-coded.
    """

    descriptor = getattr(track, "DESCRIPTOR", None)

    if descriptor is None:
        raise ValueError("Track has no protobuf DESCRIPTOR.")

    field = descriptor.fields_by_name.get("object_type")

    if field is None or field.enum_type is None:
        raise ValueError("Track.object_type enum descriptor unavailable.")

    enum_value = field.enum_type.values_by_number.get(
        int(track.object_type)
    )

    if enum_value is None:
        return "TYPE_OTHER"

    name = enum_value.name

    allowed = {
        "TYPE_UNSET",
        "TYPE_VEHICLE",
        "TYPE_PEDESTRIAN",
        "TYPE_CYCLIST",
        "TYPE_OTHER",
    }

    if name not in allowed:
        return "TYPE_OTHER"

    return name  # type: ignore[return-value]


def _raw_state_from_womd(state: Any) -> RawObjectStateW:
    """Extract only the frozen Stage-1 causal annotation fields.

    Deliberately does not access velocity_x or velocity_y.
    """

    return RawObjectStateW(
        center_W_m=(
            float(state.center_x),
            float(state.center_y),
            float(state.center_z),
        ),
        dimensions_lwh_m=(
            float(state.length),
            float(state.width),
            float(state.height),
        ),
        heading_rad=float(state.heading),
        valid=bool(state.valid),
    )


def build_stage1a_scene_from_womd(
    scenario: Any,
    *,
    headlamp_config: HeadlampSurrogateConfig | None = None,
    receiver_config: ReceiverGeometryConfig | None = None,
) -> Stage1aScene:
    """Convert one parsed WOMD Scenario into a causal Stage-1A scene."""

    if headlamp_config is None:
        headlamp_config = HeadlampSurrogateConfig()

    if receiver_config is None:
        receiver_config = ReceiverGeometryConfig()

    anchor = int(scenario.current_time_index)
    sdc_track_index = int(scenario.sdc_track_index)

    if anchor < 0:
        raise ValueError("Scenario current_time_index is negative.")

    if not 0 <= sdc_track_index < len(scenario.tracks):
        raise ValueError("Invalid scenario.sdc_track_index.")

    causal_count = anchor + 1

    if len(scenario.timestamps_seconds) < causal_count:
        raise ValueError("Scenario lacks complete causal timestamps.")

    # Freeze causal timestamps immediately.
    timestamps = tuple(
        float(value)
        for value in scenario.timestamps_seconds[:causal_count]
    )

    sdc_track = scenario.tracks[sdc_track_index]

    if len(sdc_track.states) < causal_count:
        raise ValueError("SDC lacks complete causal state prefix.")

    sdc_anchor_proto = sdc_track.states[anchor]

    if not bool(sdc_anchor_proto.valid):
        raise ValueError("SDC anchor state is invalid.")

    sdc_anchor = SdcStateW(
        center_w_m=(
            float(sdc_anchor_proto.center_x),
            float(sdc_anchor_proto.center_y),
            float(sdc_anchor_proto.center_z),
        ),
        heading_rad=float(sdc_anchor_proto.heading),
        length_m=float(sdc_anchor_proto.length),
        valid=True,
    )

    anchor_frames = build_anchor_frames(
        sdc_anchor,
        headlamp_config,
    )
    static_map = canonicalize_static_map(
        scenario,
        T_H0_from_W=anchor_frames.T_H0_from_W,
    )
    dynamic_map = canonicalize_causal_dynamic_map(
        scenario,
        T_H0_from_W=anchor_frames.T_H0_from_W,
    )

    actors: list[CausalActorArtifact] = []
    receivers: list[ReceiverRecord] = []

    for track_index, track in enumerate(scenario.tracks):
        if len(track.states) < causal_count:
            raise ValueError(
                f"Track index {track_index} lacks causal state prefix."
            )

        object_class = _object_class_from_womd_track(track)
        is_sdc = track_index == sdc_track_index

        # Critical causality boundary:
        # future track states are never converted or inspected.
        causal_states = tuple(
            _raw_state_from_womd(state)
            for state in track.states[:causal_count]
        )

        anchor_state = causal_states[anchor]

        receiver_geometry: ReceiverGeometryH0 | None = None

        if (
            object_class == "TYPE_VEHICLE"
            and not is_sdc
            and anchor_state.valid
        ):
            receiver_geometry = receiver_geometry_in_H0(
                ActorAnchorStateW(
                    center_w_m=anchor_state.center_W_m,
                    heading_rad=anchor_state.heading_rad,
                    valid=True,
                ),
                anchor_frames.T_H0_from_W,
                receiver_config,
            )

        actor = build_causal_actor_artifact(
            scenario_id=str(scenario.scenario_id),
            track_id=int(track.id),
            object_class=object_class,
            is_sdc=is_sdc,
            receiver_geometry_valid=(
                receiver_geometry is not None
                and receiver_geometry.receiver_geometry_valid
            ),
            timestamps_seconds=timestamps,
            states=causal_states,
            current_time_index=anchor,
        )

        actors.append(actor)

        if receiver_geometry is not None:
            receivers.append(
                ReceiverRecord(
                    track_id=int(track.id),
                    geometry=receiver_geometry,
                )
            )

    return Stage1aScene(
        scenario_id=str(scenario.scenario_id),
        anchor_index=anchor,
        sdc_track_index=sdc_track_index,
        anchor_frames=anchor_frames,
        actors=tuple(actors),
        receivers=tuple(receivers),
        static_map=static_map,
        dynamic_map=dynamic_map,
    )


def stage1a_scene_causal_payload(
    scene: Stage1aScene,
) -> dict[str, object]:
    """Canonical deterministic payload for scene-level causality tests."""

    return {
        "scenario_id": scene.scenario_id,
        "anchor_index": scene.anchor_index,
        "sdc_track_index": scene.sdc_track_index,
        "anchor_frames": asdict(scene.anchor_frames),
        "actors": [
            causal_actor_payload(actor)
            for actor in scene.actors
        ],
        "receivers": [
            asdict(receiver)
            for receiver in scene.receivers
        ],
        "static_map": [
            asdict(feature)
            for feature in scene.static_map
        ],
        "dynamic_map": [
            asdict(frame)
            for frame in scene.dynamic_map
        ],
    }


def stage1a_scene_causal_sha256(scene: Stage1aScene) -> str:
    payload = stage1a_scene_causal_payload(scene)

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()
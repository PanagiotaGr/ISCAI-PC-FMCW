from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iscai_stage1.actors.artifact import (
    ActorMetadata,
    CausalActorArtifact,
)
from iscai_stage1.actors.history import (
    RawObjectStateW,
    canonicalize_causal_actor_history,
)
from iscai_stage1.actors.roles import compute_actor_role_masks
from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
    ReceiverGeometryConfig,
    Stage1ArtifactSemantics,
)
from iscai_stage1.geometry.frames import (
    AnchorFrames,
    SdcStateW,
    build_anchor_frames,
    horizontal_bearing_rad,
)
from iscai_stage1.geometry.receiver import (
    ActorAnchorStateW,
    ReceiverGeometryH0,
    receiver_geometry_in_H0,
)


@dataclass(frozen=True)
class AdaptedActor:
    track_index: int
    artifact: CausalActorArtifact
    anchor_center_H0_m: tuple[float, float, float] | None
    anchor_bearing_H0_rad: float | None
    receiver_geometry_H0: ReceiverGeometryH0 | None


@dataclass(frozen=True)
class AdaptedScenario:
    scenario_id: str
    anchor_index: int
    sdc_track_index: int
    frames: AnchorFrames
    actors: tuple[AdaptedActor, ...]


def object_type_name(track: Any) -> str:
    field = track.DESCRIPTOR.fields_by_name.get("object_type")

    if field is None or field.enum_type is None:
        raise RuntimeError(
            "WOMD Track.object_type enum descriptor unavailable."
        )

    value = field.enum_type.values_by_number.get(
        int(track.object_type)
    )

    if value is None:
        return f"UNKNOWN_{int(track.object_type)}"

    return value.name


def _raw_state(state: Any) -> RawObjectStateW:
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


def adapt_causal_womd_scenario(
    scenario: Any,
    *,
    headlamp_config: HeadlampSurrogateConfig | None = None,
    receiver_config: ReceiverGeometryConfig | None = None,
) -> AdaptedScenario:
    """
    Convert one parsed WOMD Scenario into Stage-1A causal actor artifacts.

    Important:
    - only timestamps/states through current_time_index are accessed;
    - WOMD annotated velocity is never read;
    - tracks_to_predict / objects_of_interest are never read;
    - track_id remains metadata only.
    """
    anchor = int(scenario.current_time_index)

    if anchor < 0:
        raise RuntimeError(
            f"Invalid current_time_index={anchor}"
        )

    timestamps_s = tuple(
        float(value)
        for value in scenario.timestamps_seconds[: anchor + 1]
    )

    if len(timestamps_s) != anchor + 1:
        raise RuntimeError(
            "Scenario does not contain complete causal timestamps."
        )

    sdc_index = int(scenario.sdc_track_index)

    if not 0 <= sdc_index < len(scenario.tracks):
        raise RuntimeError(
            f"Invalid sdc_track_index={sdc_index}"
        )

    sdc_track = scenario.tracks[sdc_index]

    if len(sdc_track.states) <= anchor:
        raise RuntimeError(
            "SDC track does not contain anchor state."
        )

    sdc_anchor_proto = sdc_track.states[anchor]

    if not sdc_anchor_proto.valid:
        raise RuntimeError("SDC anchor state is invalid.")

    h_cfg = (
        headlamp_config
        if headlamp_config is not None
        else HeadlampSurrogateConfig()
    )

    r_cfg = (
        receiver_config
        if receiver_config is not None
        else ReceiverGeometryConfig()
    )

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

    frames = build_anchor_frames(
        sdc_anchor,
        h_cfg,
    )

    semantics = Stage1ArtifactSemantics()
    actors: list[AdaptedActor] = []

    for track_index, track in enumerate(scenario.tracks):
        causal_proto_states = track.states[: anchor + 1]

        if len(causal_proto_states) != anchor + 1:
            raise RuntimeError(
                "Track does not contain complete causal prefix: "
                f"track_index={track_index}"
            )

        raw_states = tuple(
            _raw_state(state)
            for state in causal_proto_states
        )

        object_class = object_type_name(track)
        is_sdc = track_index == sdc_index

        track_fields = {
            field.name
            for field in track.DESCRIPTOR.fields
        }

        if "id" not in track_fields:
            raise RuntimeError(
                "WOMD Track.id field unavailable."
            )

        track_id = str(track.id)

        history = canonicalize_causal_actor_history(
            timestamps_seconds=timestamps_s,
            states=raw_states,
            current_time_index=anchor,
        )

        anchor_raw = raw_states[anchor]

        receiver_geometry = None
        receiver_geometry_valid = False
        anchor_center_H0 = None
        anchor_bearing_H0 = None

        if anchor_raw.valid:
            anchor_center_H0 = (
                frames.T_H0_from_W.apply_point(
                    anchor_raw.center_W_m
                )
            )

            anchor_bearing_H0 = horizontal_bearing_rad(
                anchor_center_H0
            )

            receiver_geometry = receiver_geometry_in_H0(
                ActorAnchorStateW(
                    center_w_m=anchor_raw.center_W_m,
                    heading_rad=anchor_raw.heading_rad,
                    valid=True,
                ),
                frames.T_H0_from_W,
                r_cfg,
            )

            receiver_geometry_valid = bool(
                receiver_geometry.receiver_geometry_valid
            )

        roles = compute_actor_role_masks(
            validity=history.state_valid,
            anchor_index=anchor,
            object_class=object_class,
            is_sdc=is_sdc,
            receiver_geometry_valid=receiver_geometry_valid,
        )

        artifact = CausalActorArtifact(
            semantics=semantics,
            metadata=ActorMetadata(
                scenario_id=str(scenario.scenario_id),
                track_id=track_id,
                object_class=object_class,
                is_sdc=is_sdc,
            ),
            history=history,
            roles=roles,
        )

        actors.append(
            AdaptedActor(
                track_index=track_index,
                artifact=artifact,
                anchor_center_H0_m=anchor_center_H0,
                anchor_bearing_H0_rad=anchor_bearing_H0,
                receiver_geometry_H0=receiver_geometry,
            )
        )

    return AdaptedScenario(
        scenario_id=str(scenario.scenario_id),
        anchor_index=anchor,
        sdc_track_index=sdc_index,
        frames=frames,
        actors=tuple(actors),
    )

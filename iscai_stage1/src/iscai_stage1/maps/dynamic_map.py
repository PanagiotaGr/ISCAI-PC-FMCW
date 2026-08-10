"""Causal WOMD dynamic-map canonicalization.

Only DynamicMapState entries at t <= current_time_index are accessed.
Future traffic-light states are forbidden.

TrafficSignalLaneState schema is checked through the actual protobuf
descriptor at runtime rather than hard-coded numeric enum mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from iscai_stage1.contracts.stage1a import Vec3
from iscai_stage1.geometry.rigid import RigidTransform


@dataclass(frozen=True)
class CausalLaneSignalState:
    lane_id: int
    state_name: str
    stop_point_W_m: Vec3 | None
    stop_point_H0_m: Vec3 | None


@dataclass(frozen=True)
class CausalDynamicMapFrame:
    time_index: int
    timestamp_s: float
    lane_states: tuple[CausalLaneSignalState, ...]


def _field(message: Any, name: str):
    descriptor = getattr(message, "DESCRIPTOR", None)

    if descriptor is None:
        raise ValueError(
            "TrafficSignalLaneState has no protobuf DESCRIPTOR."
        )

    field = descriptor.fields_by_name.get(name)

    if field is None:
        raise ValueError(
            f"Required TrafficSignalLaneState field missing: {name}"
        )

    return field


def _state_enum_name(message: Any) -> str:
    field = _field(message, "state")

    if field.enum_type is None:
        raise ValueError(
            "TrafficSignalLaneState.state is not an enum."
        )

    numeric_value = int(message.state)
    enum_value = field.enum_type.values_by_number.get(numeric_value)

    if enum_value is None:
        return f"UNKNOWN_{numeric_value}"

    return str(enum_value.name)


def _map_point(point: Any) -> Vec3:
    result = (
        float(point.x),
        float(point.y),
        float(point.z),
    )

    if not all(isfinite(value) for value in result):
        raise ValueError("Traffic-signal stop point is non-finite.")

    return result


def _optional_stop_point(
    message: Any,
    T_H0_from_W: RigidTransform,
) -> tuple[Vec3 | None, Vec3 | None]:
    descriptor = message.DESCRIPTOR

    if "stop_point" not in descriptor.fields_by_name:
        return None, None

    # Message fields support HasField in protobuf.
    has_field = getattr(message, "HasField", None)

    if callable(has_field):
        try:
            if not message.HasField("stop_point"):
                return None, None
        except (ValueError, TypeError):
            pass

    point_W = _map_point(message.stop_point)

    return (
        point_W,
        T_H0_from_W.apply_point(point_W),
    )


def _canonicalize_lane_state(
    message: Any,
    T_H0_from_W: RigidTransform,
) -> CausalLaneSignalState:
    # Runtime schema guards: do not silently assume these fields.
    _field(message, "lane")
    _field(message, "state")

    stop_W, stop_H0 = _optional_stop_point(
        message,
        T_H0_from_W,
    )

    return CausalLaneSignalState(
        lane_id=int(message.lane),
        state_name=_state_enum_name(message),
        stop_point_W_m=stop_W,
        stop_point_H0_m=stop_H0,
    )


def canonicalize_causal_dynamic_map(
    scenario: Any,
    *,
    T_H0_from_W: RigidTransform,
) -> tuple[CausalDynamicMapFrame, ...]:
    """Canonicalize dynamic-map states only through the anchor.

    No length or contents of the future dynamic-map suffix are used
    to construct the artifact.
    """

    anchor = int(scenario.current_time_index)

    if anchor < 0:
        raise ValueError("current_time_index must be non-negative.")

    frames: list[CausalDynamicMapFrame] = []

    for time_index in range(anchor + 1):
        try:
            timestamp_s = float(
                scenario.timestamps_seconds[time_index]
            )
            dynamic_state = scenario.dynamic_map_states[time_index]
        except IndexError as exc:
            raise ValueError(
                "Scenario lacks complete causal dynamic-map prefix."
            ) from exc

        if not isfinite(timestamp_s):
            raise ValueError(
                f"Non-finite timestamp at dynamic-map index "
                f"{time_index}."
            )

        lane_states = tuple(
            _canonicalize_lane_state(
                lane_state,
                T_H0_from_W,
            )
            for lane_state in dynamic_state.lane_states
        )

        frames.append(
            CausalDynamicMapFrame(
                time_index=time_index,
                timestamp_s=timestamp_s,
                lane_states=lane_states,
            )
        )

    return tuple(frames)
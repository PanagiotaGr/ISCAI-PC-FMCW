from __future__ import annotations

from typing import Any

from iscai_stage1.geometry.rigid import RigidTransform
from iscai_stage1.maps.contracts import (
    CausalDynamicMapFrame,
    CausalMapArtifact,
    DynamicLaneState,
    StaticMapFeature,
)


SUPPORTED_STATIC_KINDS = frozenset(
    {
        "lane",
        "road_line",
        "road_edge",
        "stop_sign",
        "crosswalk",
        "speed_bump",
        "driveway",
    }
)


def _active_oneof(message: Any) -> str | None:
    for oneof in message.DESCRIPTOR.oneofs:
        value = message.WhichOneof(oneof.name)
        if value is not None:
            return value
    return None


def _enum_name(message: Any, field_name: str) -> str:
    field = message.DESCRIPTOR.fields_by_name.get(field_name)

    if field is None or field.enum_type is None:
        raise RuntimeError(
            f"Expected enum field {field_name!r} in "
            f"{message.DESCRIPTOR.full_name}"
        )

    number = int(getattr(message, field_name))
    value = field.enum_type.values_by_number.get(number)

    if value is None:
        return f"UNKNOWN_{number}"

    return value.name


def _point_W(point: Any) -> tuple[float, float, float]:
    return (
        float(point.x),
        float(point.y),
        float(point.z),
    )


def _points_both_frames(
    points: Any,
    T_H0_from_W: RigidTransform,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    points_W = tuple(
        _point_W(point)
        for point in points
    )

    points_H0 = tuple(
        T_H0_from_W.apply_point(point)
        for point in points_W
    )

    return points_W, points_H0


def _optional_message_present(
    message: Any,
    field_name: str,
) -> bool:
    try:
        return bool(message.HasField(field_name))
    except (ValueError, AttributeError):
        return True


def extract_static_features(
    scenario: Any,
    T_H0_from_W: RigidTransform,
) -> tuple[StaticMapFeature, ...]:
    result: list[StaticMapFeature] = []

    for feature in scenario.map_features:
        kind = _active_oneof(feature)

        if kind is None:
            raise RuntimeError(
                f"MapFeature id={feature.id} has no active feature_data."
            )

        if kind not in SUPPORTED_STATIC_KINDS:
            raise RuntimeError(
                f"Unsupported static map kind {kind!r}."
            )

        nested = getattr(feature, kind)

        type_name = None
        speed_limit_mph = None
        interpolating = None
        entry_lane_ids: tuple[int, ...] = ()
        exit_lane_ids: tuple[int, ...] = ()
        stop_sign_lane_ids: tuple[int, ...] = ()

        if kind in {"lane", "road_line", "road_edge"}:
            points_W, points_H0 = _points_both_frames(
                nested.polyline,
                T_H0_from_W,
            )
            type_name = _enum_name(nested, "type")

        elif kind in {
            "crosswalk",
            "speed_bump",
            "driveway",
        }:
            points_W, points_H0 = _points_both_frames(
                nested.polygon,
                T_H0_from_W,
            )

        elif kind == "stop_sign":
            point_W = _point_W(nested.position)

            points_W = (point_W,)
            points_H0 = (
                T_H0_from_W.apply_point(point_W),
            )

            stop_sign_lane_ids = tuple(
                int(value)
                for value in nested.lane
            )

        else:
            raise AssertionError(kind)

        if kind == "lane":
            speed_limit_mph = float(
                nested.speed_limit_mph
            )
            interpolating = bool(
                nested.interpolating
            )
            entry_lane_ids = tuple(
                int(value)
                for value in nested.entry_lanes
            )
            exit_lane_ids = tuple(
                int(value)
                for value in nested.exit_lanes
            )

        result.append(
            StaticMapFeature(
                feature_id=int(feature.id),
                kind=kind,
                points_W_m=points_W,
                points_H0_m=points_H0,
                type_name=type_name,
                speed_limit_mph=speed_limit_mph,
                interpolating=interpolating,
                entry_lane_ids=entry_lane_ids,
                exit_lane_ids=exit_lane_ids,
                stop_sign_lane_ids=stop_sign_lane_ids,
            )
        )

    return tuple(result)


def _extract_dynamic_lane_state(
    lane_state: Any,
    T_H0_from_W: RigidTransform,
) -> DynamicLaneState:
    fields = lane_state.DESCRIPTOR.fields_by_name

    required = {
        "lane",
        "state",
        "stop_point",
    }

    missing = required - set(fields)

    if missing:
        raise RuntimeError(
            "Unexpected TrafficSignalLaneState schema; "
            f"missing={sorted(missing)}"
        )

    stop_W = None
    stop_H0 = None

    if _optional_message_present(
        lane_state,
        "stop_point",
    ):
        stop_W = _point_W(
            lane_state.stop_point
        )
        stop_H0 = T_H0_from_W.apply_point(
            stop_W
        )

    return DynamicLaneState(
        lane_id=int(lane_state.lane),
        state_name=_enum_name(
            lane_state,
            "state",
        ),
        stop_point_W_m=stop_W,
        stop_point_H0_m=stop_H0,
    )


def extract_causal_dynamic_frames(
    scenario: Any,
    T_H0_from_W: RigidTransform,
) -> tuple[CausalDynamicMapFrame, ...]:
    anchor = int(scenario.current_time_index)

    if len(scenario.timestamps_seconds) <= anchor:
        raise RuntimeError(
            "Incomplete timestamp prefix."
        )

    if len(scenario.dynamic_map_states) <= anchor:
        raise RuntimeError(
            "Incomplete causal dynamic-map prefix."
        )

    result: list[CausalDynamicMapFrame] = []

    # Deliberately access only 0..anchor.
    for time_index in range(anchor + 1):
        proto_frame = scenario.dynamic_map_states[
            time_index
        ]

        lane_states = tuple(
            _extract_dynamic_lane_state(
                item,
                T_H0_from_W,
            )
            for item in proto_frame.lane_states
        )

        result.append(
            CausalDynamicMapFrame(
                time_index=time_index,
                timestamp_s=float(
                    scenario.timestamps_seconds[
                        time_index
                    ]
                ),
                lane_states=lane_states,
            )
        )

    return tuple(result)


def adapt_causal_womd_map(
    scenario: Any,
    T_H0_from_W: RigidTransform,
) -> CausalMapArtifact:
    return CausalMapArtifact(
        scenario_id=str(
            scenario.scenario_id
        ),
        anchor_index=int(
            scenario.current_time_index
        ),
        static_features=extract_static_features(
            scenario,
            T_H0_from_W,
        ),
        dynamic_frames=extract_causal_dynamic_frames(
            scenario,
            T_H0_from_W,
        ),
    )

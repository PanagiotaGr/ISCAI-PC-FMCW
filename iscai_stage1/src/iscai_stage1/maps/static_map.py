"""Causal Stage-1 static WOMD map canonicalization.

Static map geometry is annotation-derived upstream context.

It is NOT a Stage-2 sensor observation.

Supported geometry was confirmed by the frozen Stage-0 schema audits:
- lane, road_line, road_edge -> polyline
- crosswalk, speed_bump, driveway -> polygon
- stop_sign -> position
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from iscai_stage1.contracts.stage1a import Vec3
from iscai_stage1.geometry.rigid import RigidTransform


MapFeatureKind = Literal[
    "lane",
    "road_line",
    "road_edge",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
]


@dataclass(frozen=True)
class StaticMapFeature:
    feature_id: int
    kind: MapFeatureKind

    # Geometry is retained in both the original WOMD world frame
    # and the canonical anchor headlamp frame.
    points_W_m: tuple[Vec3, ...]
    points_H0_m: tuple[Vec3, ...]

    # Minimal semantic attributes confirmed by Stage-0 schema.
    semantic_type: str | None = None
    speed_limit_mph: float | None = None
    interpolating: bool | None = None

    # Lane topology fields confirmed in the frozen nested schema.
    entry_lane_ids: tuple[int, ...] = ()
    exit_lane_ids: tuple[int, ...] = ()


def _enum_name(message: Any, field_name: str) -> str | None:
    descriptor = getattr(message, "DESCRIPTOR", None)

    if descriptor is None:
        return None

    field = descriptor.fields_by_name.get(field_name)

    if field is None or field.enum_type is None:
        return None

    value = field.enum_type.values_by_number.get(
        int(getattr(message, field_name))
    )

    if value is None:
        return None

    return str(value.name)


def _active_map_feature_kind(feature: Any) -> str | None:
    descriptor = getattr(feature, "DESCRIPTOR", None)

    if descriptor is None:
        raise ValueError("MapFeature has no protobuf DESCRIPTOR.")

    for oneof in descriptor.oneofs:
        active = feature.WhichOneof(oneof.name)

        if active is not None:
            return str(active)

    return None


def _map_point_to_vec3(point: Any) -> Vec3:
    values = (
        float(point.x),
        float(point.y),
        float(point.z),
    )

    if not all(isfinite(value) for value in values):
        raise ValueError("MapPoint contains non-finite coordinate.")

    return values


def _points_from_repeated_field(
    message: Any,
    field_name: str,
) -> tuple[Vec3, ...]:
    return tuple(
        _map_point_to_vec3(point)
        for point in getattr(message, field_name)
    )


def _transform_points(
    points_W_m: tuple[Vec3, ...],
    T_H0_from_W: RigidTransform,
) -> tuple[Vec3, ...]:
    return tuple(
        T_H0_from_W.apply_point(point_W)
        for point_W in points_W_m
    )


def canonicalize_static_map(
    scenario: Any,
    *,
    T_H0_from_W: RigidTransform,
) -> tuple[StaticMapFeature, ...]:
    """Canonicalize all supported static map features.

    No actor future state, dynamic-map future state, tracks_to_predict,
    or objects_of_interest is accessed.
    """

    output: list[StaticMapFeature] = []

    supported = {
        "lane",
        "road_line",
        "road_edge",
        "stop_sign",
        "crosswalk",
        "speed_bump",
        "driveway",
    }

    for feature in scenario.map_features:
        kind = _active_map_feature_kind(feature)

        if kind is None:
            continue

        if kind not in supported:
            raise ValueError(
                f"Unsupported confirmed MapFeature kind: {kind}"
            )

        nested = getattr(feature, kind)

        semantic_type: str | None = None
        speed_limit_mph: float | None = None
        interpolating: bool | None = None
        entry_lane_ids: tuple[int, ...] = ()
        exit_lane_ids: tuple[int, ...] = ()

        if kind in {"lane", "road_line", "road_edge"}:
            points_W = _points_from_repeated_field(
                nested,
                "polyline",
            )

            semantic_type = _enum_name(nested, "type")

            if kind == "lane":
                speed_limit_mph = float(nested.speed_limit_mph)
                interpolating = bool(nested.interpolating)

                entry_lane_ids = tuple(
                    int(value)
                    for value in nested.entry_lanes
                )
                exit_lane_ids = tuple(
                    int(value)
                    for value in nested.exit_lanes
                )

        elif kind in {
            "crosswalk",
            "speed_bump",
            "driveway",
        }:
            points_W = _points_from_repeated_field(
                nested,
                "polygon",
            )

        elif kind == "stop_sign":
            points_W = (
                _map_point_to_vec3(nested.position),
            )

        else:
            raise AssertionError("Unreachable map-feature branch.")

        output.append(
            StaticMapFeature(
                feature_id=int(feature.id),
                kind=kind,  # type: ignore[arg-type]
                points_W_m=points_W,
                points_H0_m=_transform_points(
                    points_W,
                    T_H0_from_W,
                ),
                semantic_type=semantic_type,
                speed_limit_mph=speed_limit_mph,
                interpolating=interpolating,
                entry_lane_ids=entry_lane_ids,
                exit_lane_ids=exit_lane_ids,
            )
        )

    return tuple(output)
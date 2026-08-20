from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage1.contracts.stage1a import Vec3
from iscai_stage1.geometry.rigid import RigidTransform


RADIAL_VELOCITY_SOURCE = (
    "geometry_derived_from_causal_womd_trajectory"
)

ANGLE_SOURCE = "scene_perception_geometry"

FRAME_NAME = "Ht"


@dataclass(frozen=True)
class IdealCausalObservable:
    """
    Noise-free causal geometry upstream of the Stage-2
    PC-FMCW-like measurement model.

    This is NOT yet a sensor measurement.
    """

    time_index: int
    timestamp_s: float

    actor_position_Ht_m: Vec3 | None

    range_m: float | None
    azimuth_rad: float | None
    elevation_rad: float | None

    radial_velocity_mps: float | None

    geometry_valid: bool
    radial_velocity_valid: bool

    frame_name: str = FRAME_NAME

    radial_velocity_source: str = (
        RADIAL_VELOCITY_SOURCE
    )

    angle_source: str = ANGLE_SOURCE

    measured_fmcw: bool = False


def _finite_vec3(value: Vec3) -> bool:
    return all(
        math.isfinite(component)
        for component in value
    )


def _sub(
    a: Vec3,
    b: Vec3,
) -> Vec3:
    return (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    )


def _norm(value: Vec3) -> float:
    return math.sqrt(
        value[0] * value[0]
        + value[1] * value[1]
        + value[2] * value[2]
    )


def headlamp_origin_W(
    T_Ht_from_W: RigidTransform,
) -> Vec3:
    """
    World-frame position of the dynamic headlamp origin.
    """
    value = (
        T_Ht_from_W
        .inverse()
        .apply_point(
            (0.0, 0.0, 0.0)
        )
    )

    result = (
        float(value[0]),
        float(value[1]),
        float(value[2]),
    )

    if not _finite_vec3(result):
        raise ValueError(
            "Headlamp world origin must be finite."
        )

    return result


def adjacent_headlamp_velocity_W(
    *,
    previous_origin_W_m: Vec3,
    current_origin_W_m: Vec3,
    previous_timestamp_s: float,
    current_timestamp_s: float,
) -> Vec3 | None:
    """
    Strict causal backward difference.

    No search-back, interpolation or future sample.
    """
    if not (
        _finite_vec3(previous_origin_W_m)
        and _finite_vec3(current_origin_W_m)
        and math.isfinite(previous_timestamp_s)
        and math.isfinite(current_timestamp_s)
    ):
        return None

    dt = (
        current_timestamp_s
        - previous_timestamp_s
    )

    if dt <= 0.0:
        return None

    return (
        (
            current_origin_W_m[0]
            - previous_origin_W_m[0]
        ) / dt,
        (
            current_origin_W_m[1]
            - previous_origin_W_m[1]
        ) / dt,
        (
            current_origin_W_m[2]
            - previous_origin_W_m[2]
        ) / dt,
    )


def ideal_causal_observable(
    *,
    time_index: int,
    timestamp_s: float,

    actor_position_W_m: Vec3,
    actor_position_valid: bool,

    actor_velocity_W_mps: Vec3,
    actor_velocity_valid: bool,

    headlamp_velocity_W_mps: Vec3 | None,

    T_Ht_from_W: RigidTransform,
) -> IdealCausalObservable:

    if time_index < 0:
        raise ValueError(
            "time_index must be non-negative."
        )

    if not math.isfinite(timestamp_s):
        raise ValueError(
            "timestamp_s must be finite."
        )

    if (
        not actor_position_valid
        or not _finite_vec3(
            actor_position_W_m
        )
    ):
        return IdealCausalObservable(
            time_index=time_index,
            timestamp_s=timestamp_s,
            actor_position_Ht_m=None,
            range_m=None,
            azimuth_rad=None,
            elevation_rad=None,
            radial_velocity_mps=None,
            geometry_valid=False,
            radial_velocity_valid=False,
        )

    point_Ht = (
        T_Ht_from_W.apply_point(
            actor_position_W_m
        )
    )

    point_Ht = (
        float(point_Ht[0]),
        float(point_Ht[1]),
        float(point_Ht[2]),
    )

    if not _finite_vec3(point_Ht):
        raise ValueError(
            "Transformed actor position is non-finite."
        )

    range_m = _norm(point_Ht)

    # Range=0 makes line-of-sight direction undefined.
    if range_m <= 1e-12:
        return IdealCausalObservable(
            time_index=time_index,
            timestamp_s=timestamp_s,
            actor_position_Ht_m=point_Ht,
            range_m=0.0,
            azimuth_rad=None,
            elevation_rad=None,
            radial_velocity_mps=None,
            geometry_valid=False,
            radial_velocity_valid=False,
        )

    x, y, z = point_Ht

    azimuth_rad = math.atan2(
        y,
        x,
    )

    horizontal_range = math.hypot(
        x,
        y,
    )

    elevation_rad = math.atan2(
        z,
        horizontal_range,
    )

    radial_velocity_mps = None
    radial_velocity_valid = False

    if (
        actor_velocity_valid
        and _finite_vec3(
            actor_velocity_W_mps
        )
        and headlamp_velocity_W_mps
        is not None
        and _finite_vec3(
            headlamp_velocity_W_mps
        )
    ):
        headlamp_W = headlamp_origin_W(
            T_Ht_from_W
        )

        relative_position_W = _sub(
            actor_position_W_m,
            headlamp_W,
        )

        relative_range_W = _norm(
            relative_position_W
        )

        if relative_range_W <= 1e-12:
            raise RuntimeError(
                "Ht/world range consistency failure."
            )

        relative_velocity_W = _sub(
            actor_velocity_W_mps,
            headlamp_velocity_W_mps,
        )

        unit_los_W = (
            relative_position_W[0]
            / relative_range_W,
            relative_position_W[1]
            / relative_range_W,
            relative_position_W[2]
            / relative_range_W,
        )

        radial_velocity_mps = (
            unit_los_W[0]
            * relative_velocity_W[0]
            + unit_los_W[1]
            * relative_velocity_W[1]
            + unit_los_W[2]
            * relative_velocity_W[2]
        )

        if not math.isfinite(
            radial_velocity_mps
        ):
            raise ValueError(
                "Computed radial velocity is non-finite."
            )

        radial_velocity_valid = True

    return IdealCausalObservable(
        time_index=time_index,
        timestamp_s=timestamp_s,
        actor_position_Ht_m=point_Ht,
        range_m=range_m,
        azimuth_rad=azimuth_rad,
        elevation_rad=elevation_rad,
        radial_velocity_mps=(
            radial_velocity_mps
        ),
        geometry_valid=True,
        radial_velocity_valid=(
            radial_velocity_valid
        ),
    )

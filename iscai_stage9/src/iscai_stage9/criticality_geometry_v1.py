from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


PRIMARY_HORIZON_S = 1.0
PRIMARY_CLEARANCE_M = 1.0
CLEARANCE_SENSITIVITY_M = (0.0, 0.5, 1.0, 2.0)
PRIMARY_M = 4096
YAW_RATE_CV_THRESHOLD_RAD_S = 0.001
EGO_HISTORY_WINDOW_S = 1.0


def wrap_angle(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def point_to_oriented_rectangle_distance(
    point_xy,
    rectangle_center_xy,
    rectangle_length_m: float,
    rectangle_width_m: float,
    rectangle_yaw_rad: float,
) -> float:
    """
    Exact Euclidean distance from a 2D point to a closed oriented rectangle.
    Distance is zero for a point inside or on the rectangle.
    """
    p = np.asarray(point_xy, dtype=np.float64)
    c = np.asarray(rectangle_center_xy, dtype=np.float64)
    if p.shape != (2,) or c.shape != (2,):
        raise ValueError("point_xy and rectangle_center_xy must have shape (2,).")

    L = float(rectangle_length_m)
    W = float(rectangle_width_m)
    yaw = float(rectangle_yaw_rad)
    vals = np.array([p[0], p[1], c[0], c[1], L, W, yaw], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError("Geometry inputs must be finite.")
    if L <= 0.0 or W <= 0.0:
        raise ValueError("Rectangle dimensions must be positive.")

    delta = p - c
    co = math.cos(yaw)
    si = math.sin(yaw)

    # Rotate world/H0 delta into rectangle-local coordinates by -yaw.
    local_x = co * delta[0] + si * delta[1]
    local_y = -si * delta[0] + co * delta[1]

    dx = max(abs(float(local_x)) - L / 2.0, 0.0)
    dy = max(abs(float(local_y)) - W / 2.0, 0.0)
    return float(math.hypot(dx, dy))


def critical_proximity_event(
    actor_center_xy,
    ego_center_xy,
    ego_length_m: float,
    ego_width_m: float,
    ego_yaw_rad: float,
    clearance_m: float = PRIMARY_CLEARANCE_M,
):
    clearance = float(clearance_m)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance_m must be finite and nonnegative.")
    distance = point_to_oriented_rectangle_distance(
        actor_center_xy,
        ego_center_xy,
        ego_length_m,
        ego_width_m,
        ego_yaw_rad,
    )
    return bool(distance <= clearance + 1e-12), float(distance)


def transform_heading_with_rigid_transform(
    heading_world_rad: float,
    T_H0_from_W,
) -> float:
    """
    Transform a world-frame planar heading using the frozen rigid transform's
    vector action. Translation is never applied to heading.
    """
    h = float(heading_world_rad)
    if not math.isfinite(h):
        raise ValueError("heading_world_rad must be finite.")
    unit = T_H0_from_W.apply_vector(
        (math.cos(h), math.sin(h), 0.0)
    )
    v = np.asarray(unit, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("Transformed heading vector invalid.")
    n = float(np.linalg.norm(v[:2]))
    if n <= 1e-12:
        raise ValueError("Transformed heading vector has zero planar norm.")
    return float(math.atan2(v[1], v[0]))


def world_ego_row_to_h0(row: dict, T_H0_from_W) -> dict:
    """
    Convert one CURRENT/PAST ego row to H0. No future fields are required.
    """
    p = T_H0_from_W.apply_point(
        (
            float(row["center_x"]),
            float(row["center_y"]),
            float(row["center_z"]),
        )
    )
    v = T_H0_from_W.apply_vector(
        (
            float(row["velocity_x"]),
            float(row["velocity_y"]),
            0.0,
        )
    )
    p = np.asarray(p, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if p.shape != (3,) or v.shape != (3,):
        raise ValueError("Rigid-transform output shape changed.")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        raise ValueError("Non-finite transformed ego state.")

    return {
        "timestamp_s": float(row["timestamp_s"]),
        "center_x_H0_m": float(p[0]),
        "center_y_H0_m": float(p[1]),
        "heading_H0_rad": transform_heading_with_rigid_transform(
            float(row["heading"]),
            T_H0_from_W,
        ),
        "velocity_x_H0_mps": float(v[0]),
        "velocity_y_H0_mps": float(v[1]),
        "length_m": float(row["length"]),
        "width_m": float(row["width"]),
        "valid": bool(row.get("valid", True)),
    }


def estimate_yaw_rate_ols(history_h0: Sequence[dict]) -> float:
    """
    OLS slope of unwrapped causal heading versus timestamps.
    Uses valid rows only. Requires >=2 rows; otherwise returns 0.
    """
    rows = [r for r in history_h0 if bool(r.get("valid", True))]
    if len(rows) < 2:
        return 0.0

    rows = sorted(rows, key=lambda r: float(r["timestamp_s"]))
    t = np.asarray([float(r["timestamp_s"]) for r in rows], dtype=np.float64)
    heading = np.asarray([float(r["heading_H0_rad"]) for r in rows], dtype=np.float64)

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(heading)):
        raise ValueError("Non-finite causal ego history.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Causal ego timestamps must be strictly increasing.")

    h = np.unwrap(heading)
    tc = t - float(np.mean(t))
    denom = float(tc @ tc)
    if denom <= 1e-18:
        return 0.0
    slope = float((tc @ (h - float(np.mean(h)))) / denom)
    if not math.isfinite(slope):
        raise ValueError("Yaw-rate estimate is non-finite.")
    return slope


@dataclass(frozen=True)
class CTRVState:
    center_x_H0_m: float
    center_y_H0_m: float
    heading_H0_rad: float
    speed_mps: float
    yaw_rate_rad_s: float
    length_m: float
    width_m: float


def bind_current_ego_ctrv(history_h0: Sequence[dict]) -> CTRVState:
    rows = [r for r in history_h0 if bool(r.get("valid", True))]
    if not rows:
        raise ValueError("No valid causal ego states.")
    rows = sorted(rows, key=lambda r: float(r["timestamp_s"]))
    current = rows[-1]

    vx = float(current["velocity_x_H0_mps"])
    vy = float(current["velocity_y_H0_mps"])
    speed = float(math.hypot(vx, vy))
    yaw_rate = float(estimate_yaw_rate_ols(rows))

    values = (
        float(current["center_x_H0_m"]),
        float(current["center_y_H0_m"]),
        float(current["heading_H0_rad"]),
        speed,
        yaw_rate,
        float(current["length_m"]),
        float(current["width_m"]),
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite CTRV binding.")
    if values[5] <= 0.0 or values[6] <= 0.0:
        raise ValueError("Current ego dimensions must be positive.")

    return CTRVState(
        center_x_H0_m=values[0],
        center_y_H0_m=values[1],
        heading_H0_rad=wrap_angle(values[2]),
        speed_mps=values[3],
        yaw_rate_rad_s=values[4],
        length_m=values[5],
        width_m=values[6],
    )


def propagate_ctrv(
    state: CTRVState,
    horizon_s: float = PRIMARY_HORIZON_S,
    yaw_rate_cv_threshold_rad_s: float = YAW_RATE_CV_THRESHOLD_RAD_S,
):
    """
    Deterministic planar CTRV continuation from the current H0 ego state.
    """
    dt = float(horizon_s)
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("horizon_s must be finite and nonnegative.")

    x0 = float(state.center_x_H0_m)
    y0 = float(state.center_y_H0_m)
    psi0 = float(state.heading_H0_rad)
    v = float(state.speed_mps)
    omega = float(state.yaw_rate_rad_s)
    threshold = float(yaw_rate_cv_threshold_rad_s)

    if not all(math.isfinite(z) for z in (x0, y0, psi0, v, omega, threshold)):
        raise ValueError("CTRV inputs must be finite.")
    if v < 0.0 or threshold < 0.0:
        raise ValueError("Invalid speed/threshold.")

    if abs(omega) < threshold:
        x = x0 + v * dt * math.cos(psi0)
        y = y0 + v * dt * math.sin(psi0)
        psi = psi0
        mode = "CV"
    else:
        x = x0 + (v / omega) * (
            math.sin(psi0 + omega * dt) - math.sin(psi0)
        )
        y = y0 - (v / omega) * (
            math.cos(psi0 + omega * dt) - math.cos(psi0)
        )
        psi = wrap_angle(psi0 + omega * dt)
        mode = "CTRV"

    return {
        "center_xy_H0_m": (float(x), float(y)),
        "heading_H0_rad": float(psi),
        "length_m": float(state.length_m),
        "width_m": float(state.width_m),
        "mode": mode,
    }

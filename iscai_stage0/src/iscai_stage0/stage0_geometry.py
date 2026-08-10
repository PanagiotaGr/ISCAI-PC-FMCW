from __future__ import annotations

import math

import numpy as np


def global_to_ego_xy(
    points_xy: np.ndarray,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> np.ndarray:
    """
    Convert global XY coordinates to an ego-at-anchor frame.

    This is used only for Stage-0 coordinate sanity/visualization.
    The production headlamp transforms belong to Stage 1.
    """
    points = np.asarray(points_xy, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Expected shape (N, 2), got {points.shape}"
        )

    translated = points - np.array(
        [ego_x, ego_y],
        dtype=np.float64,
    )

    c = math.cos(ego_heading)
    s = math.sin(ego_heading)

    rotation = np.array(
        [
            [c, s],
            [-s, c],
        ],
        dtype=np.float64,
    )

    return translated @ rotation.T


def ego_to_global_xy(
    points_xy: np.ndarray,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> np.ndarray:
    """Inverse of global_to_ego_xy()."""
    points = np.asarray(points_xy, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Expected shape (N, 2), got {points.shape}"
        )

    c = math.cos(ego_heading)
    s = math.sin(ego_heading)

    rotation_inverse = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )

    return (
        points @ rotation_inverse.T
        + np.array([ego_x, ego_y], dtype=np.float64)
    )


def box_corners_xy(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    heading: float,
) -> np.ndarray:
    """Return the four XY corners of a WOMD oriented box."""
    if length <= 0 or width <= 0:
        raise ValueError(
            f"Invalid box dimensions: length={length}, width={width}"
        )

    half_l = length / 2.0
    half_w = width / 2.0

    local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )

    c = math.cos(heading)
    s = math.sin(heading)

    rotation = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )

    return (
        local @ rotation.T
        + np.array([center_x, center_y], dtype=np.float64)
    )
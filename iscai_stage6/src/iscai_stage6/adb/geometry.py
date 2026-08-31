from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1.0e-12

_ALLOWED_CONTROLLER_STATE_SOURCES = frozenset(
    {
        "current_causal",
        "predicted_sample",
        "predicted_mean",
    }
)

_FORBIDDEN_CONTROLLER_STATE_SOURCES = frozenset(
    {
        "future_gt",
        "oracle_future",
        "future_annotation",
    }
)

_ALLOWED_ORIENTATION_SOURCES = frozenset(
    {
        "current_causal",
        "predicted_tangent",
        "predicted_orientation",
        "frozen_low_speed_fallback",
        "oracle_evaluator_only",
    }
)


def _finite_array(
    values,
    *,
    shape,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, "
            f"got {array.shape}"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return array


def wrap_angle(
    angle_rad,
):
    """
    Wrap angle(s) into [-pi, pi).
    """

    array = np.asarray(
        angle_rad,
        dtype=np.float64,
    )

    wrapped = (
        array + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    if np.ndim(
        angle_rad
    ) == 0:
        return float(
            wrapped
        )

    return wrapped


def validate_controller_provenance(
    *,
    state_source: str,
    orientation_source: str,
    controller_path: bool,
) -> None:

    state_source = str(
        state_source
    )

    orientation_source = str(
        orientation_source
    )

    if (
        orientation_source
        not in
        _ALLOWED_ORIENTATION_SOURCES
    ):
        raise ValueError(
            "Unknown orientation source: "
            f"{orientation_source}"
        )

    if not controller_path:
        return

    if (
        state_source
        in
        _FORBIDDEN_CONTROLLER_STATE_SOURCES
    ):
        raise ValueError(
            "Future/oracle annotation is forbidden "
            "on the Stage6 controller path: "
            f"{state_source}"
        )

    if (
        state_source
        not in
        _ALLOWED_CONTROLLER_STATE_SOURCES
    ):
        raise ValueError(
            "Unapproved controller state source: "
            f"{state_source}"
        )

    if (
        orientation_source
        ==
        "oracle_evaluator_only"
    ):
        raise ValueError(
            "Oracle orientation is forbidden "
            "on the controller path"
        )


@dataclass(frozen=True)
class Box3D:
    """
    Oriented actor box represented in headlamp H0 coordinates.

    Convention:
      +x forward
      +y left
      +z up

    length is along local x,
    width  is along local y,
    height is along local z,
    yaw is rotation about +z.
    """

    center_xyz: tuple[float, float, float]
    length_m: float
    width_m: float
    height_m: float
    yaw_rad: float

    def __post_init__(
        self,
    ) -> None:

        center = _finite_array(
            self.center_xyz,
            shape=(3,),
            name="center_xyz",
        )

        object.__setattr__(
            self,
            "center_xyz",
            tuple(
                float(v)
                for v in center
            ),
        )

        for name in (
            "length_m",
            "width_m",
            "height_m",
        ):

            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if (
                not math.isfinite(
                    value
                )
                or
                value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and > 0"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        yaw = float(
            self.yaw_rad
        )

        if not math.isfinite(
            yaw
        ):
            raise ValueError(
                "yaw_rad must be finite"
            )

        object.__setattr__(
            self,
            "yaw_rad",
            wrap_angle(
                yaw
            ),
        )

    @property
    def center(
        self,
    ) -> np.ndarray:

        return np.asarray(
            self.center_xyz,
            dtype=np.float64,
        )

    @property
    def dimensions(
        self,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.length_m,
                self.width_m,
                self.height_m,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class FractionalBoxRegion:
    """
    A body-frame subregion of an actor box.

    Each bound is expressed as a fraction of the corresponding
    full box dimension and must lie in [-0.5, +0.5].

    No semantic values such as windshield/face fractions are
    hard-coded here; those belong to the later class-policy block.
    """

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]

    def __post_init__(
        self,
    ) -> None:

        for name in (
            "x_bounds",
            "y_bounds",
            "z_bounds",
        ):

            values = tuple(
                float(v)
                for v in getattr(
                    self,
                    name,
                )
            )

            if len(
                values
            ) != 2:
                raise ValueError(
                    f"{name} requires exactly 2 values"
                )

            lo, hi = values

            if (
                not math.isfinite(
                    lo
                )
                or
                not math.isfinite(
                    hi
                )
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

            if not (
                -0.5
                <=
                lo
                <
                hi
                <=
                0.5
            ):
                raise ValueError(
                    f"{name} must satisfy "
                    "-0.5 <= lo < hi <= 0.5"
                )

            object.__setattr__(
                self,
                name,
                (
                    lo,
                    hi,
                ),
            )


@dataclass(frozen=True)
class ProjectedPoints:
    xyz: np.ndarray
    ground_range_m: np.ndarray
    range_3d_m: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray

    def __post_init__(
        self,
    ) -> None:

        n = int(
            self.xyz.shape[0]
        )

        if (
            self.xyz.ndim != 2
            or
            self.xyz.shape[1] != 3
        ):
            raise ValueError(
                "xyz must be Nx3"
            )

        for name in (
            "ground_range_m",
            "range_3d_m",
            "theta_rad",
            "phi_rad",
        ):

            array = np.asarray(
                getattr(
                    self,
                    name,
                ),
                dtype=np.float64,
            )

            if array.shape != (
                n,
            ):
                raise ValueError(
                    f"{name} must have shape ({n},)"
                )


@dataclass(frozen=True)
class ProjectedBox:
    box: Box3D
    corners: ProjectedPoints
    centroid: ProjectedPoints

    theta_center_rad: float
    theta_span_rad: float
    theta_min_unwrapped_rad: float
    theta_max_unwrapped_rad: float

    ground_range_min_m: float
    ground_range_max_m: float

    range_3d_min_m: float
    range_3d_max_m: float

    phi_min_rad: float
    phi_max_rad: float

    state_source: str
    orientation_source: str
    controller_path: bool

    @property
    def uses_full_box(
        self,
    ) -> bool:

        return (
            self.corners.xyz.shape
            ==
            (8, 3)
        )


@dataclass(frozen=True)
class ProjectedRegion:
    region: FractionalBoxRegion
    points: ProjectedPoints

    theta_center_rad: float
    theta_span_rad: float

    ground_range_min_m: float
    ground_range_max_m: float

    phi_min_rad: float
    phi_max_rad: float


def _rotation_z(
    yaw_rad: float,
) -> np.ndarray:

    c = math.cos(
        yaw_rad
    )

    s = math.sin(
        yaw_rad
    )

    return np.asarray(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def body_points_to_headlamp(
    box: Box3D,
    local_points_xyz,
) -> np.ndarray:

    points = np.asarray(
        local_points_xyz,
        dtype=np.float64,
    )

    if (
        points.ndim != 2
        or
        points.shape[1] != 3
    ):
        raise ValueError(
            "local_points_xyz must be Nx3"
        )

    if not np.all(
        np.isfinite(
            points
        )
    ):
        raise ValueError(
            "local_points_xyz contains "
            "non-finite values"
        )

    rotation = _rotation_z(
        box.yaw_rad
    )

    return (
        points
        @
        rotation.T
        +
        box.center[
            None,
            :
        ]
    )


def box_corners_headlamp(
    box: Box3D,
) -> np.ndarray:
    """
    Return all 8 physical corners of the actor box.
    """

    hx = 0.5 * box.length_m
    hy = 0.5 * box.width_m
    hz = 0.5 * box.height_m

    local = np.asarray(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )

    return body_points_to_headlamp(
        box,
        local,
    )


def region_corners_headlamp(
    box: Box3D,
    region: FractionalBoxRegion,
) -> np.ndarray:
    """
    Return the 8 corners of an arbitrary body-frame subregion.
    """

    x_lo, x_hi = region.x_bounds
    y_lo, y_hi = region.y_bounds
    z_lo, z_hi = region.z_bounds

    local = np.asarray(
        [
            [
                x_frac * box.length_m,
                y_frac * box.width_m,
                z_frac * box.height_m,
            ]
            for x_frac in (
                x_lo,
                x_hi,
            )
            for y_frac in (
                y_lo,
                y_hi,
            )
            for z_frac in (
                z_lo,
                z_hi,
            )
        ],
        dtype=np.float64,
    )

    return body_points_to_headlamp(
        box,
        local,
    )


def project_points_to_headlamp(
    points_xyz,
) -> ProjectedPoints:
    """
    Map H0 Cartesian coordinates into
    (ground range, 3D range, azimuth theta, elevation phi).
    """

    xyz = np.asarray(
        points_xyz,
        dtype=np.float64,
    )

    if (
        xyz.ndim != 2
        or
        xyz.shape[1] != 3
    ):
        raise ValueError(
            "points_xyz must be Nx3"
        )

    if (
        xyz.shape[0]
        < 1
    ):
        raise ValueError(
            "at least one point is required"
        )

    if not np.all(
        np.isfinite(
            xyz
        )
    ):
        raise ValueError(
            "points_xyz contains "
            "non-finite values"
        )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    ground_range = np.hypot(
        x,
        y,
    )

    range_3d = np.sqrt(
        x * x
        +
        y * y
        +
        z * z
    )

    if np.any(
        range_3d
        <=
        _EPS
    ):
        raise ValueError(
            "point coincides with headlamp origin"
        )

    theta = np.arctan2(
        y,
        x,
    )

    phi = np.arctan2(
        z,
        ground_range,
    )

    return ProjectedPoints(
        xyz=xyz.copy(),
        ground_range_m=ground_range,
        range_3d_m=range_3d,
        theta_rad=theta,
        phi_rad=phi,
    )


def _minimal_circular_interval(
    angles_rad,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Return:
      center,
      span,
      min_unwrapped,
      max_unwrapped

    using the minimum circular interval containing all angles.
    """

    angles = np.asarray(
        angles_rad,
        dtype=np.float64,
    )

    if (
        angles.ndim != 1
        or
        angles.size < 1
    ):
        raise ValueError(
            "angles_rad must be a non-empty vector"
        )

    if not np.all(
        np.isfinite(
            angles
        )
    ):
        raise ValueError(
            "angles_rad contains non-finite values"
        )

    if angles.size == 1:
        angle = wrap_angle(
            float(
                angles[0]
            )
        )

        return (
            angle,
            0.0,
            angle,
            angle,
        )

    wrapped_0_2pi = np.mod(
        angles,
        2.0 * np.pi,
    )

    ordered = np.sort(
        wrapped_0_2pi
    )

    extended = np.concatenate(
        [
            ordered,
            ordered[:1]
            +
            2.0 * np.pi,
        ]
    )

    gaps = np.diff(
        extended
    )

    largest_gap_index = int(
        np.argmax(
            gaps
        )
    )

    start_index = (
        largest_gap_index + 1
    ) % (
        ordered.size
    )

    start = float(
        ordered[
            start_index
        ]
    )

    relative = np.mod(
        wrapped_0_2pi
        -
        start,
        2.0 * np.pi,
    )

    lo = 0.0
    hi = float(
        np.max(
            relative
        )
    )

    span = hi - lo

    center_unwrapped = (
        start
        +
        0.5 * span
    )

    center = wrap_angle(
        center_unwrapped
    )

    min_unwrapped = (
        center_unwrapped
        -
        0.5 * span
    )

    max_unwrapped = (
        center_unwrapped
        +
        0.5 * span
    )

    return (
        float(
            center
        ),
        float(
            span
        ),
        float(
            min_unwrapped
        ),
        float(
            max_unwrapped
        ),
    )


def project_box_to_headlamp(
    box: Box3D,
    *,
    state_source: str,
    orientation_source: str,
    controller_path: bool = True,
) -> ProjectedBox:
    """
    Full 3D actor-box projection.

    Centroid projection is retained only as diagnostic metadata;
    all extent fields are derived from all 8 physical corners.
    """

    validate_controller_provenance(
        state_source=state_source,
        orientation_source=orientation_source,
        controller_path=controller_path,
    )

    corners_xyz = box_corners_headlamp(
        box
    )

    corners = project_points_to_headlamp(
        corners_xyz
    )

    centroid = project_points_to_headlamp(
        box.center[
            None,
            :
        ]
    )

    (
        theta_center,
        theta_span,
        theta_min,
        theta_max,
    ) = _minimal_circular_interval(
        corners.theta_rad
    )

    return ProjectedBox(
        box=box,
        corners=corners,
        centroid=centroid,

        theta_center_rad=theta_center,
        theta_span_rad=theta_span,
        theta_min_unwrapped_rad=theta_min,
        theta_max_unwrapped_rad=theta_max,

        ground_range_min_m=float(
            np.min(
                corners.ground_range_m
            )
        ),

        ground_range_max_m=float(
            np.max(
                corners.ground_range_m
            )
        ),

        range_3d_min_m=float(
            np.min(
                corners.range_3d_m
            )
        ),

        range_3d_max_m=float(
            np.max(
                corners.range_3d_m
            )
        ),

        phi_min_rad=float(
            np.min(
                corners.phi_rad
            )
        ),

        phi_max_rad=float(
            np.max(
                corners.phi_rad
            )
        ),

        state_source=str(
            state_source
        ),

        orientation_source=str(
            orientation_source
        ),

        controller_path=bool(
            controller_path
        ),
    )


def project_region_to_headlamp(
    box: Box3D,
    region: FractionalBoxRegion,
) -> ProjectedRegion:
    """
    Project a policy-defined 3D actor subregion.

    This provides the geometry hook required later for:
      - vehicle windshield/mirror-relevant region,
      - pedestrian face/body protection region,
      - cyclist body/vehicle visibility region.

    No class-specific fractions are hard-coded here.
    """

    points_xyz = region_corners_headlamp(
        box,
        region,
    )

    points = project_points_to_headlamp(
        points_xyz
    )

    (
        theta_center,
        theta_span,
        _,
        _,
    ) = _minimal_circular_interval(
        points.theta_rad
    )

    return ProjectedRegion(
        region=region,
        points=points,

        theta_center_rad=theta_center,
        theta_span_rad=theta_span,

        ground_range_min_m=float(
            np.min(
                points.ground_range_m
            )
        ),

        ground_range_max_m=float(
            np.max(
                points.ground_range_m
            )
        ),

        phi_min_rad=float(
            np.min(
                points.phi_rad
            )
        ),

        phi_max_rad=float(
            np.max(
                points.phi_rad
            )
        ),
    )


def centroid_only_projection(
    box: Box3D,
) -> ProjectedPoints:
    """
    Diagnostic/baseline helper only.

    This function deliberately does not produce an ADB extent.
    """

    return project_points_to_headlamp(
        box.center[
            None,
            :
        ]
    )

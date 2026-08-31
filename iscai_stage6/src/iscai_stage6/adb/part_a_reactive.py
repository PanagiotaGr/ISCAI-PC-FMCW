from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .illumination import (
    ReactiveShadowRegion,
    reactive_adb_map,
    static_adb_map,
)


# ============================================================
# Frozen Part-A dynamic reactive parameters
# ============================================================

PART_A_EPS = 1.0e-12

PART_A_TARGET_WIDTH_M = 1.9
PART_A_LATERAL_SAFETY_MARGIN_M = 1.0

PART_A_DELTA_THETA_DEG = 0.0

# Actual adb_intensity_map() controller parameters:
PART_A_RADIAL_SAFETY_MARGIN_M = 4.0
PART_A_CONTROL_TRANSITION_LENGTH_M = 20.0

# Track-reconstruction visualization radial-zone parameters.
# These are NOT substituted for the control L(theta,r) thresholds.
PART_A_TRZ_FRONT_MARGIN_UNITS = 4.0
PART_A_TRZ_TRANSITION_LENGTH_UNITS = 8.0

# Frozen fixed demonstration only.
# The dynamic track-driven path does not reference this value.
PART_A_FIXED_DEMO_EPSILON_DEG = 0.75

# Frozen Module-3 reference raster only.
# It is NOT the final Stage-6 scientific/evaluation grid.
PART_A_REFERENCE_THETA_MIN_DEG = -25.0
PART_A_REFERENCE_THETA_MAX_DEG = 25.0
PART_A_REFERENCE_THETA_POINTS = 501

PART_A_REFERENCE_RANGE_MIN_M = 0.0
PART_A_REFERENCE_RANGE_MAX_M = 180.0
PART_A_REFERENCE_RANGE_POINTS = 361

CAUSAL_BOX_CENTER_FIELD = 'stage1_anchor_center_H0_m'
CAUSAL_BOX_CLASS_FIELD = 'object_type'

ORIGINAL_REACTIVE_SUPPORTED_CLASS = "TYPE_VEHICLE"

PART_A_COORDINATE_CONVENTION = (
    "+y forward / +x right"
)

STAGE6_H0_COORDINATE_CONVENTION = (
    "+x forward / +y left / +z up"
)


@dataclass(frozen=True)
class PartAReactiveGeometry:
    """
    Exact current-state Part-A dynamic angular geometry,
    expressed in both original Part-A coordinates and
    Stage-6 H0 angular coordinates.

    This is a legacy original-reactive comparator.

    It is deliberately:
      - current-state only,
      - fixed-width Part-A geometry,
      - not predictive,
      - not covariance aware,
      - not class-aware beyond the original vehicle use,
      - not a full-box Stage-6 predictive policy.
    """

    center_H0_m: tuple[
        float,
        float,
        float,
    ]

    x_partA_right_m: float
    y_partA_forward_m: float

    target_range_m: float

    theta_c_partA_deg: float
    shadow_center_partA_deg: float
    epsilon_dynamic_deg: float

    theta_min_partA_deg: float
    theta_max_partA_deg: float

    theta_min_H0_deg: float
    theta_max_H0_deg: float

    control_r_min_m: float
    control_r_max_m: float

    trz_visual_r_min_m: float
    trz_visual_r_max_m: float


class PartAReferenceGrid:
    """
    Frozen Part-A Module-3 reference sampling axes.

    Baseline-parity / smoke grid only.

    The final common Stage-6 ADB scientific grid remains
    a Block-6.8 pre-formal freeze decision.
    """

    @property
    def theta_centers_rad(
        self,
    ) -> np.ndarray:

        return np.deg2rad(
            np.linspace(
                PART_A_REFERENCE_THETA_MIN_DEG,
                PART_A_REFERENCE_THETA_MAX_DEG,
                PART_A_REFERENCE_THETA_POINTS,
            )
        )

    @property
    def range_centers_m(
        self,
    ) -> np.ndarray:

        return np.linspace(
            PART_A_REFERENCE_RANGE_MIN_M,
            PART_A_REFERENCE_RANGE_MAX_M,
            PART_A_REFERENCE_RANGE_POINTS,
        )

    @property
    def shape(
        self,
    ) -> tuple[int, int]:

        return (
            PART_A_REFERENCE_THETA_POINTS,
            PART_A_REFERENCE_RANGE_POINTS,
        )


def h0_center_to_part_a_xy(
    center_H0_m,
) -> tuple[
    float,
    float,
]:
    """
    Exact convention conversion.

    Stage6 H0:
        x_H = forward
        y_H = left

    Part-A:
        x_A = right
        y_A = forward

    Therefore:
        x_A = -y_H
        y_A =  x_H
    """

    values = tuple(
        float(value)
        for value in center_H0_m
    )

    if len(values) != 3:
        raise ValueError(
            "center_H0_m must contain x,y,z"
        )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "center_H0_m must be finite"
        )

    x_H0, y_H0, _z_H0 = values

    return (
        -y_H0,
        x_H0,
    )


def part_a_dynamic_geometry_from_h0_center(
    center_H0_m,
) -> PartAReactiveGeometry:
    """
    Exact binding of the frozen Part-A dynamic geometry:

      target_range = sqrt(x_A^2 + y_A^2)

      theta_c =
          deg(atan2(x_A, y_A))

      shadow_center =
          theta_c - delta_theta

      effective_half_width =
          target_width / 2 + lateral_safety_margin

      epsilon =
          deg(
              atan(
                  effective_half_width
                  /
                  max(target_range, EPS)
              )
          )

    Angular interval is then converted back to H0:
        [theta_min_A, theta_max_A]
            ->
        [-theta_max_A, -theta_min_A]

    IMPORTANT:
    The illumination-control radial thresholds come from
    adb_intensity_map():
        r_min = range + 4 m
        r_max = r_min + 20 m

    The separate +4/+8 TRZ values are preserved only as
    visualization-geometry provenance.
    """

    center = tuple(
        float(value)
        for value in center_H0_m
    )

    x_A, y_A = (
        h0_center_to_part_a_xy(
            center
        )
    )

    target_range = math.sqrt(
        x_A * x_A
        +
        y_A * y_A
    )

    theta_c_A = math.degrees(
        math.atan2(
            x_A,
            y_A,
        )
    )

    shadow_center_A = (
        theta_c_A
        -
        PART_A_DELTA_THETA_DEG
    )

    effective_half_width = (
        PART_A_TARGET_WIDTH_M
        /
        2.0
        +
        PART_A_LATERAL_SAFETY_MARGIN_M
    )

    epsilon_dynamic = math.degrees(
        math.atan(
            effective_half_width
            /
            max(
                target_range,
                PART_A_EPS,
            )
        )
    )

    theta_min_A = (
        shadow_center_A
        -
        epsilon_dynamic
    )

    theta_max_A = (
        shadow_center_A
        +
        epsilon_dynamic
    )

    # Exact interval sign/order conversion.
    theta_min_H0 = (
        -theta_max_A
    )

    theta_max_H0 = (
        -theta_min_A
    )

    control_r_min = (
        target_range
        +
        PART_A_RADIAL_SAFETY_MARGIN_M
    )

    control_r_max = (
        control_r_min
        +
        PART_A_CONTROL_TRANSITION_LENGTH_M
    )

    trz_r_min = (
        target_range
        +
        PART_A_TRZ_FRONT_MARGIN_UNITS
    )

    trz_r_max = (
        trz_r_min
        +
        PART_A_TRZ_TRANSITION_LENGTH_UNITS
    )

    return PartAReactiveGeometry(
        center_H0_m=center,

        x_partA_right_m=x_A,
        y_partA_forward_m=y_A,

        target_range_m=target_range,

        theta_c_partA_deg=theta_c_A,
        shadow_center_partA_deg=shadow_center_A,
        epsilon_dynamic_deg=epsilon_dynamic,

        theta_min_partA_deg=theta_min_A,
        theta_max_partA_deg=theta_max_A,

        theta_min_H0_deg=theta_min_H0,
        theta_max_H0_deg=theta_max_H0,

        control_r_min_m=control_r_min,
        control_r_max_m=control_r_max,

        trz_visual_r_min_m=trz_r_min,
        trz_visual_r_max_m=trz_r_max,
    )


def part_a_region_from_h0_center(
    center_H0_m,
) -> ReactiveShadowRegion:
    """
    Convert one current target center into the existing
    normalized L(theta,r) kernel representation.

    floor=0 exactly preserves the original Part-A
    reactive vehicle baseline.
    """

    geometry = (
        part_a_dynamic_geometry_from_h0_center(
            center_H0_m
        )
    )

    theta_center_H0_deg = (
        0.5
        *
        (
            geometry.theta_min_H0_deg
            +
            geometry.theta_max_H0_deg
        )
    )

    theta_span_deg = (
        geometry.theta_max_H0_deg
        -
        geometry.theta_min_H0_deg
    )

    return ReactiveShadowRegion(
        theta_center_rad=(
            math.radians(
                theta_center_H0_deg
            )
        ),

        theta_span_rad=(
            math.radians(
                theta_span_deg
            )
        ),

        r_shadow_end_m=(
            geometry.control_r_min_m
        ),

        r_transition_end_m=(
            geometry.control_r_max_m
        ),

        intensity_floor=0.0,

        source_semantics=(
            "current_deterministic_region"
        ),
    )


def part_a_current_vehicle_centers_from_causal_boxes(
    actor_boxes: Iterable,
) -> tuple[
    tuple[
        float,
        float,
        float,
    ],
    ...,
]:
    """
    Extract current vehicle centers only.

    This intentionally reproduces the frozen legacy
    original-reactive baseline, whose current MHT track
    position feeds a fixed-width angular shadow formula.

    It MUST NOT be reused as the Stage-6 predictive
    full-box occupancy implementation.
    """

    centers = []

    for actor_box in actor_boxes:

        actor_class = str(
            getattr(
                actor_box,
                CAUSAL_BOX_CLASS_FIELD,
            )
        )

        if (
            actor_class
            !=
            ORIGINAL_REACTIVE_SUPPORTED_CLASS
        ):
            continue

        center = getattr(
            actor_box,
            CAUSAL_BOX_CENTER_FIELD,
        )

        if center is None:
            raise ValueError(
                "Vehicle causal box has no "
                "verified Stage1 anchor center."
            )

        values = tuple(
            float(value)
            for value in center
        )

        if len(values) != 3:
            raise ValueError(
                "Stage1 anchor center must be 3D."
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Stage1 anchor center contains "
                "non-finite values."
            )

        centers.append(
            values
        )

    return tuple(
        centers
    )


def part_a_original_reactive_map_from_h0_centers(
    grid,
    centers_H0_m: Iterable,
) -> np.ndarray:
    """
    Original current-state reactive ADB comparator.

    No future trajectory, posterior, covariance, class policy
    or communication-codebook input is accepted.
    """

    centers = tuple(
        centers_H0_m
    )

    if not centers:
        return static_adb_map(
            grid
        )

    regions = tuple(
        part_a_region_from_h0_center(
            center
        )
        for center in centers
    )

    return reactive_adb_map(
        grid,
        regions,
    )


def part_a_direct_source_reference_map_from_h0_centers(
    grid,
    centers_H0_m: Iterable,
) -> np.ndarray:
    """
    Independent direct transcription of the frozen
    adb_intensity_map() + np.minimum combination rule.

    Used only as a parity oracle for the adapter implementation.
    """

    theta_H_deg = np.rad2deg(
        np.asarray(
            grid.theta_centers_rad,
            dtype=np.float64,
        )
    )

    range_m = np.asarray(
        grid.range_centers_m,
        dtype=np.float64,
    )

    expected_shape = (
        theta_H_deg.size,
        range_m.size,
    )

    combined = np.ones(
        expected_shape,
        dtype=np.float64,
    )

    for center in tuple(
        centers_H0_m
    ):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                center
            )
        )

        intensity = np.ones(
            expected_shape,
            dtype=np.float64,
        )

        angular = (
            theta_H_deg
            >=
            geometry.theta_min_H0_deg
        ) & (
            theta_H_deg
            <=
            geometry.theta_max_H0_deg
        )

        full_shadow = (
            angular[:, None]
            &
            (
                range_m[None, :]
                <=
                geometry.control_r_min_m
            )
        )

        transition = (
            angular[:, None]
            &
            (
                range_m[None, :]
                >
                geometry.control_r_min_m
            )
            &
            (
                range_m[None, :]
                <
                geometry.control_r_max_m
            )
        )

        intensity[
            full_shadow
        ] = 0.0

        if np.any(
            transition
        ):

            range_grid = np.broadcast_to(
                range_m[
                    None,
                    :
                ],
                expected_shape,
            )

            u = (
                range_grid[
                    transition
                ]
                -
                geometry.control_r_min_m
            ) / (
                geometry.control_r_max_m
                -
                geometry.control_r_min_m
            )

            intensity[
                transition
            ] = (
                0.5
                *
                (
                    1.0
                    -
                    np.cos(
                        np.pi
                        *
                        u
                    )
                )
            )

        combined = np.minimum(
            combined,
            intensity,
        )

    return combined

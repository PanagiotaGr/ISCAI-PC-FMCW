from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .geometry import (
    wrap_angle,
)

from .grid import (
    IlluminationGridSpec,
)


_EPS = 1.0e-12


@dataclass(frozen=True)
class ReactiveShadowRegion:
    """
    One current deterministic ADB shadow region.

    This class deliberately describes illumination geometry only.
    It has no prediction posterior, covariance, future ground-truth,
    communication-beam or class-policy inputs.

    theta_center_rad / theta_span_rad:
        Horizontal angular shadow interval.

    r_shadow_end_m:
        Part-A/paper r_min:
        strongest attenuation applies up to this range.

    r_transition_end_m:
        Part-A/paper r_max:
        raised-cosine transition reaches full illumination here.

    intensity_floor:
        Normalized minimum illumination.
        floor=0 reproduces Eq. (5) of the reference paper.
        Non-zero floors are supported for later Stage6 policies.
    """

    theta_center_rad: float
    theta_span_rad: float

    r_shadow_end_m: float
    r_transition_end_m: float

    intensity_floor: float = 0.0

    source_semantics: str = (
        "current_deterministic_region"
    )

    def __post_init__(
        self,
    ) -> None:

        center = float(
            self.theta_center_rad
        )

        span = float(
            self.theta_span_rad
        )

        r_shadow = float(
            self.r_shadow_end_m
        )

        r_transition = float(
            self.r_transition_end_m
        )

        floor = float(
            self.intensity_floor
        )

        for name, value in (
            (
                "theta_center_rad",
                center,
            ),
            (
                "theta_span_rad",
                span,
            ),
            (
                "r_shadow_end_m",
                r_shadow,
            ),
            (
                "r_transition_end_m",
                r_transition,
            ),
            (
                "intensity_floor",
                floor,
            ),
        ):
            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not (
            0.0
            <=
            span
            <=
            2.0 * math.pi
        ):
            raise ValueError(
                "theta_span_rad must lie "
                "within [0, 2*pi]"
            )

        if r_shadow < 0.0:
            raise ValueError(
                "r_shadow_end_m must be >= 0"
            )

        if not (
            r_transition
            >
            r_shadow
        ):
            raise ValueError(
                "r_transition_end_m must be "
                "strictly greater than "
                "r_shadow_end_m"
            )

        if not (
            0.0
            <=
            floor
            <=
            1.0
        ):
            raise ValueError(
                "intensity_floor must lie "
                "within [0,1]"
            )

        source = str(
            self.source_semantics
        )

        if source != (
            "current_deterministic_region"
        ):
            raise ValueError(
                "ReactiveShadowRegion Part-A "
                "kernel accepts only "
                "current_deterministic_region "
                "semantics"
            )

        object.__setattr__(
            self,
            "theta_center_rad",
            float(
                wrap_angle(
                    center
                )
            ),
        )

        object.__setattr__(
            self,
            "theta_span_rad",
            span,
        )

        object.__setattr__(
            self,
            "r_shadow_end_m",
            r_shadow,
        )

        object.__setattr__(
            self,
            "r_transition_end_m",
            r_transition,
        )

        object.__setattr__(
            self,
            "intensity_floor",
            floor,
        )

        object.__setattr__(
            self,
            "source_semantics",
            source,
        )


def part_a_radial_profile(
    range_m,
    *,
    r_shadow_end_m: float,
    r_transition_end_m: float,
    intensity_floor: float = 0.0,
):
    """
    Normalized Part-A/paper raised-cosine radial profile.

    For floor = 0 this is exactly:

      0,
        r <= r_min

      0.5 * (
          1 - cos(
              pi * (r-r_min)/(r_max-r_min)
          )
      ),
        r_min < r < r_max

      1,
        r >= r_max

    A non-zero floor preserves the same raised-cosine shape
    while lifting the minimum intensity:

      floor + (1-floor) * raised_cosine.

    This generalization is needed later for VRU visibility
    policies but does not alter the floor=0 Part-A baseline.
    """

    r_shadow = float(
        r_shadow_end_m
    )

    r_transition = float(
        r_transition_end_m
    )

    floor = float(
        intensity_floor
    )

    if (
        not math.isfinite(
            r_shadow
        )
        or
        r_shadow < 0.0
    ):
        raise ValueError(
            "r_shadow_end_m must be finite "
            "and >= 0"
        )

    if (
        not math.isfinite(
            r_transition
        )
        or
        r_transition
        <=
        r_shadow
    ):
        raise ValueError(
            "r_transition_end_m must be finite "
            "and > r_shadow_end_m"
        )

    if (
        not math.isfinite(
            floor
        )
        or
        not (
            0.0 <= floor <= 1.0
        )
    ):
        raise ValueError(
            "intensity_floor must lie "
            "within [0,1]"
        )

    raw = np.asarray(
        range_m,
        dtype=np.float64,
    )

    scalar_input = (
        raw.ndim == 0
    )

    values = np.atleast_1d(
        raw
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "range_m contains non-finite values"
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "range_m must be non-negative"
        )

    profile = np.ones_like(
        values,
        dtype=np.float64,
    )

    shadow = (
        values
        <=
        r_shadow
    )

    transition = (
        values
        >
        r_shadow
    ) & (
        values
        <
        r_transition
    )

    profile[
        shadow
    ] = floor

    if np.any(
        transition
    ):

        u = (
            values[
                transition
            ]
            -
            r_shadow
        ) / (
            r_transition
            -
            r_shadow
        )

        raised_cosine = (
            0.5
            *
            (
                1.0
                -
                np.cos(
                    np.pi * u
                )
            )
        )

        profile[
            transition
        ] = (
            floor
            +
            (
                1.0
                -
                floor
            )
            *
            raised_cosine
        )

    profile = np.clip(
        profile,
        floor,
        1.0,
    )

    if scalar_input:
        return float(
            profile[0]
        )

    return profile.reshape(
        raw.shape
    )


def angular_shadow_mask(
    theta_rad,
    *,
    theta_center_rad: float,
    theta_span_rad: float,
):
    """
    Circular angular interval membership.

    Handles shadow intervals that cross -pi/+pi.
    """

    center = float(
        theta_center_rad
    )

    span = float(
        theta_span_rad
    )

    if (
        not math.isfinite(
            center
        )
        or
        not math.isfinite(
            span
        )
    ):
        raise ValueError(
            "angular parameters must be finite"
        )

    if not (
        0.0
        <=
        span
        <=
        2.0 * math.pi
    ):
        raise ValueError(
            "theta_span_rad must lie "
            "within [0,2*pi]"
        )

    raw = np.asarray(
        theta_rad,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            raw
        )
    ):
        raise ValueError(
            "theta_rad contains non-finite values"
        )

    if (
        span
        >=
        2.0 * math.pi
        -
        _EPS
    ):
        return np.ones(
            raw.shape,
            dtype=bool,
        )

    delta = wrap_angle(
        raw
        -
        wrap_angle(
            center
        )
    )

    return (
        np.abs(
            delta
        )
        <=
        0.5 * span
        +
        _EPS
    )


def static_adb_map(
    grid: IlluminationGridSpec,
) -> np.ndarray:
    """
    Static/no-dimming ADB baseline.

    Normalized intensity = 1 everywhere.
    """

    return np.ones(
        grid.shape,
        dtype=np.float64,
    )


def single_region_illumination_map(
    grid: IlluminationGridSpec,
    region: ReactiveShadowRegion,
) -> np.ndarray:
    """
    Apply one Part-A-style angular/radial shadow region.
    """

    angular = angular_shadow_mask(
        grid.theta_centers_rad,
        theta_center_rad=(
            region.theta_center_rad
        ),
        theta_span_rad=(
            region.theta_span_rad
        ),
    )

    radial = part_a_radial_profile(
        grid.range_centers_m,
        r_shadow_end_m=(
            region.r_shadow_end_m
        ),
        r_transition_end_m=(
            region.r_transition_end_m
        ),
        intensity_floor=(
            region.intensity_floor
        ),
    )

    result = static_adb_map(
        grid
    )

    result[
        angular,
        :,
    ] = radial[
        None,
        :
    ]

    return result


def combine_illumination_maps_minimum(
    maps: Iterable[
        np.ndarray
    ],
) -> np.ndarray:
    """
    Part-A multi-target combination rule:
    the strongest required dimming wins,
    implemented as elementwise minimum.
    """

    arrays = tuple(
        np.asarray(
            item,
            dtype=np.float64,
        )
        for item in maps
    )

    if not arrays:
        raise ValueError(
            "At least one illumination map "
            "is required"
        )

    shape = arrays[0].shape

    for array in arrays:

        if array.shape != shape:
            raise ValueError(
                "All illumination maps must "
                "have identical shape"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "Illumination map contains "
                "non-finite values"
            )

        if (
            np.any(
                array < 0.0
            )
            or
            np.any(
                array > 1.0
            )
        ):
            raise ValueError(
                "Normalized illumination map "
                "must lie within [0,1]"
            )

    stacked = np.stack(
        arrays,
        axis=0,
    )

    return np.min(
        stacked,
        axis=0,
    )


def reactive_adb_map(
    grid: IlluminationGridSpec,
    regions: Iterable[
        ReactiveShadowRegion
    ],
) -> np.ndarray:
    """
    Current-state reactive ADB illumination map.

    This is intentionally NOT predictive:
      - no future trajectory,
      - no posterior,
      - no covariance,
      - no communication codebook,
      - no class-aware future policy.

    Empty regions reduce exactly to the static ADB baseline.
    """

    regions = tuple(
        regions
    )

    if not regions:
        return static_adb_map(
            grid
        )

    maps = tuple(
        single_region_illumination_map(
            grid,
            region,
        )
        for region in regions
    )

    return (
        combine_illumination_maps_minimum(
            maps
        )
    )

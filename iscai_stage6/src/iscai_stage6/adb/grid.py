from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IlluminationGridSpec:
    """
    Parameterized Stage6 ADB grid in (theta, ground-range).

    No scientific numeric defaults are deliberately provided.
    Those values must be frozen only after the Block6.1 Part2
    audit against the frozen Stage1 headlamp geometry.
    """

    theta_min_rad: float
    theta_max_rad: float

    range_min_m: float
    range_max_m: float

    n_theta: int
    n_range: int

    def __post_init__(
        self,
    ) -> None:

        theta_min = float(
            self.theta_min_rad
        )

        theta_max = float(
            self.theta_max_rad
        )

        range_min = float(
            self.range_min_m
        )

        range_max = float(
            self.range_max_m
        )

        for name, value in (
            (
                "theta_min_rad",
                theta_min,
            ),
            (
                "theta_max_rad",
                theta_max,
            ),
            (
                "range_min_m",
                range_min,
            ),
            (
                "range_max_m",
                range_max,
            ),
        ):

            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not (
            -math.pi
            <=
            theta_min
            <
            theta_max
            <=
            math.pi
        ):
            raise ValueError(
                "theta bounds must satisfy "
                "-pi <= min < max <= pi"
            )

        if not (
            0.0
            <=
            range_min
            <
            range_max
        ):
            raise ValueError(
                "range bounds must satisfy "
                "0 <= min < max"
            )

        n_theta = int(
            self.n_theta
        )

        n_range = int(
            self.n_range
        )

        if n_theta < 2:
            raise ValueError(
                "n_theta must be >= 2"
            )

        if n_range < 2:
            raise ValueError(
                "n_range must be >= 2"
            )

        object.__setattr__(
            self,
            "theta_min_rad",
            theta_min,
        )

        object.__setattr__(
            self,
            "theta_max_rad",
            theta_max,
        )

        object.__setattr__(
            self,
            "range_min_m",
            range_min,
        )

        object.__setattr__(
            self,
            "range_max_m",
            range_max,
        )

        object.__setattr__(
            self,
            "n_theta",
            n_theta,
        )

        object.__setattr__(
            self,
            "n_range",
            n_range,
        )

    @property
    def theta_edges_rad(
        self,
    ) -> np.ndarray:

        return np.linspace(
            self.theta_min_rad,
            self.theta_max_rad,
            self.n_theta + 1,
            dtype=np.float64,
        )

    @property
    def range_edges_m(
        self,
    ) -> np.ndarray:

        return np.linspace(
            self.range_min_m,
            self.range_max_m,
            self.n_range + 1,
            dtype=np.float64,
        )

    @property
    def theta_centers_rad(
        self,
    ) -> np.ndarray:

        edges = self.theta_edges_rad

        return 0.5 * (
            edges[:-1]
            +
            edges[1:]
        )

    @property
    def range_centers_m(
        self,
    ) -> np.ndarray:

        edges = self.range_edges_m

        return 0.5 * (
            edges[:-1]
            +
            edges[1:]
        )

    @property
    def theta_step_rad(
        self,
    ) -> float:

        return (
            self.theta_max_rad
            -
            self.theta_min_rad
        ) / self.n_theta

    @property
    def range_step_m(
        self,
    ) -> float:

        return (
            self.range_max_m
            -
            self.range_min_m
        ) / self.n_range

    @property
    def shape(
        self,
    ) -> tuple[int, int]:

        return (
            self.n_theta,
            self.n_range,
        )

    def contains(
        self,
        theta_rad: float,
        range_m: float,
    ) -> bool:

        theta = float(
            theta_rad
        )

        distance = float(
            range_m
        )

        return (
            self.theta_min_rad
            <=
            theta
            <=
            self.theta_max_rad
            and
            self.range_min_m
            <=
            distance
            <=
            self.range_max_m
        )

    def nearest_cell(
        self,
        theta_rad: float,
        range_m: float,
    ) -> tuple[int, int]:

        theta = float(
            theta_rad
        )

        distance = float(
            range_m
        )

        if not self.contains(
            theta,
            distance,
        ):
            raise ValueError(
                "point lies outside "
                "illumination grid"
            )

        theta_index = int(
            np.argmin(
                np.abs(
                    self.theta_centers_rad
                    -
                    theta
                )
            )
        )

        range_index = int(
            np.argmin(
                np.abs(
                    self.range_centers_m
                    -
                    distance
                )
            )
        )

        return (
            theta_index,
            range_index,
        )


def rectangular_extent_cell_indices(
    grid: IlluminationGridSpec,
    *,
    theta_center_rad: float,
    theta_span_rad: float,
    range_min_m: float,
    range_max_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Return all theta/range cell indices overlapping a continuous
    projected extent.

    This is only a geometry utility. It is NOT the final
    probabilistic occupancy mask implementation.
    """

    theta_center = float(
        theta_center_rad
    )

    theta_span = float(
        theta_span_rad
    )

    if theta_span < 0.0:
        raise ValueError(
            "theta_span_rad must be >= 0"
        )

    theta_lo = (
        theta_center
        -
        0.5 * theta_span
    )

    theta_hi = (
        theta_center
        +
        0.5 * theta_span
    )

    range_lo = float(
        range_min_m
    )

    range_hi = float(
        range_max_m
    )

    if range_hi < range_lo:
        raise ValueError(
            "range_max_m must be >= range_min_m"
        )

    theta_edges = (
        grid.theta_edges_rad
    )

    range_edges = (
        grid.range_edges_m
    )

    theta_overlap = (
        theta_edges[:-1]
        <=
        theta_hi
    ) & (
        theta_edges[1:]
        >=
        theta_lo
    )

    range_overlap = (
        range_edges[:-1]
        <=
        range_hi
    ) & (
        range_edges[1:]
        >=
        range_lo
    )

    return (
        np.flatnonzero(
            theta_overlap
        ),
        np.flatnonzero(
            range_overlap
        ),
    )

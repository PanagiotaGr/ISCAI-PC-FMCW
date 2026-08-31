from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iscai_stage6.adb.probabilistic_full_box import (
    HORIZONS_S,
    StochasticFullBoxForecast,
)


@dataclass(frozen=True)
class OccupancyGrid:
    """
    Parameterized theta/r occupancy grid.

    theta_centers_rad and range_centers_m are cell centers.

    Block6.4 Part2A deliberately does not assign a
    scientific Stage6 grid. The caller supplies it.
    """

    theta_centers_rad: tuple[float, ...]
    range_centers_m: tuple[float, ...]

    def __post_init__(
        self,
    ) -> None:
        theta = np.asarray(
            self.theta_centers_rad,
            dtype=np.float64,
        )

        radial = np.asarray(
            self.range_centers_m,
            dtype=np.float64,
        )

        if (
            theta.ndim != 1
            or
            theta.size == 0
        ):
            raise ValueError(
                "theta_centers_rad must be "
                "a non-empty 1-D sequence."
            )

        if (
            radial.ndim != 1
            or
            radial.size == 0
        ):
            raise ValueError(
                "range_centers_m must be "
                "a non-empty 1-D sequence."
            )

        if not np.all(
            np.isfinite(
                theta
            )
        ):
            raise ValueError(
                "theta centers must be finite."
            )

        if not np.all(
            np.isfinite(
                radial
            )
        ):
            raise ValueError(
                "range centers must be finite."
            )

        if not np.all(
            np.diff(
                theta
            )
            >
            0.0
        ):
            raise ValueError(
                "theta centers must be strictly increasing."
            )

        if not np.all(
            np.diff(
                radial
            )
            >
            0.0
        ):
            raise ValueError(
                "range centers must be strictly increasing."
            )

        if np.any(
            radial
            <
            0.0
        ):
            raise ValueError(
                "range centers must be non-negative."
            )

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        return (
            len(
                self.theta_centers_rad
            ),
            len(
                self.range_centers_m
            ),
        )


@dataclass(frozen=True)
class ActorOccupancyProbability:
    """
    Monte-Carlo estimate of

        P_occ(theta, r, tau)

    for one actor.

    Shape convention:
        counts/probability [4, N_theta, N_range]
    """

    prediction_id: str

    sample_count: int

    horizons_s: tuple[
        float,
        float,
        float,
        float,
    ]

    occupancy_counts: np.ndarray

    occupancy_probability: np.ndarray

    rasterization_semantics: str = (
        "convex_hull_of_all_eight_"
        "projected_physical_box_corners"
    )

    @property
    def shape(
        self,
    ) -> tuple[int, int, int]:
        return tuple(
            int(value)
            for value in
            self.occupancy_probability.shape
        )


def _readonly(
    value,
    *,
    dtype,
) -> np.ndarray:
    array = np.array(
        value,
        dtype=dtype,
        copy=True,
    )

    array.setflags(
        write=False
    )

    return array


def _projected_theta_range_points(
    future_item,
) -> np.ndarray:
    """
    Use all eight physical projected box corners.

    Convert H0/headlamp XYZ corner locations to
    (theta, ground-range) coordinates.

    No centroid surrogate is used.
    """

    xyz = np.asarray(
        future_item
        .projection
        .corners
        .xyz,
        dtype=np.float64,
    )

    if xyz.shape != (
        8,
        3,
    ):
        raise ValueError(
            "Every projected physical box must "
            "contain exactly eight XYZ corners."
        )

    if not np.all(
        np.isfinite(
            xyz
        )
    ):
        raise ValueError(
            "Projected corner XYZ must be finite."
        )

    theta = np.arctan2(
        xyz[
            :,
            1
        ],
        xyz[
            :,
            0
        ],
    )

    ground_range = np.hypot(
        xyz[
            :,
            0
        ],
        xyz[
            :,
            1
        ],
    )

    result = np.column_stack(
        (
            theta,
            ground_range,
        )
    )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise ValueError(
            "Projected theta/r points must be finite."
        )

    return result


def _cross(
    origin: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        (
            a[
                0
            ]
            -
            origin[
                0
            ]
        )
        *
        (
            b[
                1
            ]
            -
            origin[
                1
            ]
        )
        -
        (
            a[
                1
            ]
            -
            origin[
                1
            ]
        )
        *
        (
            b[
                0
            ]
            -
            origin[
                0
            ]
        )
    )


def convex_hull_theta_range(
    points,
) -> np.ndarray:
    """
    Deterministic Andrew monotone-chain convex hull.

    The hull is a conservative projected full-box envelope
    in (theta, ground-range) space.

    It is derived from all eight physical corners rather
    than a centroid or Gaussian ellipse.
    """

    array = np.asarray(
        points,
        dtype=np.float64,
    )

    if (
        array.ndim != 2
        or
        array.shape[
            1
        ] != 2
    ):
        raise ValueError(
            "points must have shape (N,2)."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "points must be finite."
        )

    unique = np.unique(
        array,
        axis=0,
    )

    if unique.shape[
        0
    ] < 3:
        return _readonly(
            unique,
            dtype=np.float64,
        )

    ordered = sorted(
        (
            float(row[0]),
            float(row[1]),
        )
        for row in unique
    )

    def build_half(
        sequence,
    ):
        half = []

        for point in sequence:
            point_array = np.asarray(
                point,
                dtype=np.float64,
            )

            while (
                len(
                    half
                )
                >=
                2
                and
                _cross(
                    half[-2],
                    half[-1],
                    point_array,
                )
                <=
                0.0
            ):
                half.pop()

            half.append(
                point_array
            )

        return half

    lower = build_half(
        ordered
    )

    upper = build_half(
        reversed(
            ordered
        )
    )

    hull = np.asarray(
        lower[
            :-1
        ]
        +
        upper[
            :-1
        ],
        dtype=np.float64,
    )

    return _readonly(
        hull,
        dtype=np.float64,
    )


def _points_inside_ccw_convex_polygon(
    points,
    hull,
) -> np.ndarray:
    """
    Cell-center inclusion in a CCW convex polygon.

    Boundary is included.

    No numerical tolerance is introduced by Part2A.
    """

    query = np.asarray(
        points,
        dtype=np.float64,
    )

    polygon = np.asarray(
        hull,
        dtype=np.float64,
    )

    if (
        query.ndim != 2
        or
        query.shape[
            1
        ] != 2
    ):
        raise ValueError(
            "query points must have shape (N,2)."
        )

    if polygon.shape[
        0
    ] < 3:
        return np.zeros(
            (
                query.shape[
                    0
                ],
            ),
            dtype=bool,
        )

    inside = np.ones(
        (
            query.shape[
                0
            ],
        ),
        dtype=bool,
    )

    for index in range(
        polygon.shape[
            0
        ]
    ):
        a = polygon[
            index
        ]

        b = polygon[
            (
                index + 1
            )
            %
            polygon.shape[
                0
            ]
        ]

        edge_x = (
            b[
                0
            ]
            -
            a[
                0
            ]
        )

        edge_y = (
            b[
                1
            ]
            -
            a[
                1
            ]
        )

        rel_x = (
            query[
                :,
                0
            ]
            -
            a[
                0
            ]
        )

        rel_y = (
            query[
                :,
                1
            ]
            -
            a[
                1
            ]
        )

        cross = (
            edge_x
            *
            rel_y
            -
            edge_y
            *
            rel_x
        )

        inside &= (
            cross
            >=
            0.0
        )

        if not np.any(
            inside
        ):
            break

    return inside


def rasterize_projected_full_box(
    future_item,
    grid: OccupancyGrid,
) -> np.ndarray:
    """
    Rasterize one sampled projected full box.

    Returns bool [N_theta, N_range].

    Only candidate cells inside the hull bounding box are
    tested, but inclusion is against the convex hull itself,
    not its rectangular bounding box.
    """

    theta_axis = np.asarray(
        grid.theta_centers_rad,
        dtype=np.float64,
    )

    range_axis = np.asarray(
        grid.range_centers_m,
        dtype=np.float64,
    )

    result = np.zeros(
        grid.shape,
        dtype=bool,
    )

    points = (
        _projected_theta_range_points(
            future_item
        )
    )

    hull = convex_hull_theta_range(
        points
    )

    if hull.shape[
        0
    ] < 3:
        return result

    theta_min = float(
        np.min(
            hull[
                :,
                0
            ]
        )
    )

    theta_max = float(
        np.max(
            hull[
                :,
                0
            ]
        )
    )

    range_min = float(
        np.min(
            hull[
                :,
                1
            ]
        )
    )

    range_max = float(
        np.max(
            hull[
                :,
                1
            ]
        )
    )

    theta_lo = int(
        np.searchsorted(
            theta_axis,
            theta_min,
            side="left",
        )
    )

    theta_hi = int(
        np.searchsorted(
            theta_axis,
            theta_max,
            side="right",
        )
    )

    range_lo = int(
        np.searchsorted(
            range_axis,
            range_min,
            side="left",
        )
    )

    range_hi = int(
        np.searchsorted(
            range_axis,
            range_max,
            side="right",
        )
    )

    if (
        theta_lo
        >=
        theta_hi
        or
        range_lo
        >=
        range_hi
    ):
        return result

    theta_candidate = theta_axis[
        theta_lo:
        theta_hi
    ]

    range_candidate = range_axis[
        range_lo:
        range_hi
    ]

    theta_mesh, range_mesh = np.meshgrid(
        theta_candidate,
        range_candidate,
        indexing="ij",
    )

    query = np.column_stack(
        (
            theta_mesh.ravel(),
            range_mesh.ravel(),
        )
    )

    inside = (
        _points_inside_ccw_convex_polygon(
            query,
            hull,
        )
    ).reshape(
        theta_mesh.shape
    )

    result[
        theta_lo:
        theta_hi,
        range_lo:
        range_hi,
    ] = inside

    return result


def estimate_actor_occupancy_probability(
    forecast: StochasticFullBoxForecast,
    grid: OccupancyGrid,
) -> ActorOccupancyProbability:
    """
    Monte-Carlo estimate of one actor's

        P_occ(theta, r, tau).

    For every sample and horizon, the indicator is obtained
    from the convex hull of all eight physical projected box
    corners.

    Probability is exactly:
        occupancy_count / forecast.sample_count

    No gamma threshold is applied here.
    """

    count = int(
        forecast.sample_count
    )

    if count <= 0:
        raise ValueError(
            "forecast.sample_count must be positive."
        )

    if len(
        forecast.future
    ) != count:
        raise ValueError(
            "forecast.future/sample_count mismatch."
        )

    theta_count, range_count = (
        grid.shape
    )

    occupancy_counts = np.zeros(
        (
            4,
            theta_count,
            range_count,
        ),
        dtype=np.uint32,
    )

    for sample_index, sequence in enumerate(
        forecast.future
    ):
        if len(
            sequence
        ) != 4:
            raise ValueError(
                "Every trajectory sample must "
                "contain four future horizons."
            )

        for horizon_index, item in enumerate(
            sequence
        ):
            if int(
                item.sample_index
            ) != sample_index:
                raise ValueError(
                    "Forecast sample index mismatch."
                )

            if int(
                item.horizon_index
            ) != horizon_index:
                raise ValueError(
                    "Forecast horizon index mismatch."
                )

            mask = rasterize_projected_full_box(
                item,
                grid,
            )

            occupancy_counts[
                horizon_index
            ] += mask.astype(
                np.uint32
            )

    probability = (
        occupancy_counts.astype(
            np.float64
        )
        /
        float(
            count
        )
    )

    if np.any(
        probability
        <
        0.0
    ) or np.any(
        probability
        >
        1.0
    ):
        raise RuntimeError(
            "Occupancy probability left [0,1]."
        )

    return ActorOccupancyProbability(
        prediction_id=str(
            forecast.prediction_id
        ),
        sample_count=count,
        horizons_s=tuple(
            float(value)
            for value in HORIZONS_S
        ),
        occupancy_counts=(
            _readonly(
                occupancy_counts,
                dtype=np.uint32,
            )
        ),
        occupancy_probability=(
            _readonly(
                probability,
                dtype=np.float64,
            )
        ),
    )

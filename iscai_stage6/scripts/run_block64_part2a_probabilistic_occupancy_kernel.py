from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PART1_REPORT = (
    S6
    / "reports"
    / "block64_part1_stochastic_full_box_kernel.json"
)

PART1_CONTRACT = (
    S6
    / "configs"
    / "block64_part1_stochastic_full_box_kernel_contract.json"
)

PART1_MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "probabilistic_full_box.py"
)

MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "probabilistic_occupancy.py"
)

TEST = (
    S6
    / "tests"
    / "test_block64_part2a_probabilistic_occupancy.py"
)

CONTRACT = (
    S6
    / "configs"
    / "block64_part2a_probabilistic_occupancy_contract.json"
)

REPORT = (
    S6
    / "reports"
    / "block64_part2a_probabilistic_occupancy_kernel.json"
)


EXPECTED = {
    "part1_report":
        (
            "8bcc13051c93f216d7365ee44d6079df"
            "6a14567bc1b6b879b33e748473b206a4"
        ),

    "part1_contract":
        (
            "030148d9ab49e8dff2b434fa0fb7b819"
            "da77ad3d5faf3d42744bb13ef8024808"
        ),

    "part1_module":
        (
            "51daeab95713d6e7d2738ced466be4844"
            "b73dce70a09138b2b769de880462995"
        ),
}


MODULE_SOURCE = r'''from __future__ import annotations

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
'''


TEST_SOURCE = r'''from __future__ import annotations

import inspect
import unittest

import numpy as np

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.probabilistic_full_box import (
    CalibratedGaussianFullBoxPrediction,
    build_stochastic_future_full_boxes,
)

from iscai_stage6.adb.probabilistic_occupancy import (
    OccupancyGrid,
    convex_hull_theta_range,
    estimate_actor_occupancy_probability,
    rasterize_projected_full_box,
)


def covariance(
    variance: float,
):
    matrix = np.diag(
        [
            variance,
            variance,
            variance,
        ]
    )

    return tuple(
        tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in matrix
        )
        for _ in range(
            4
        )
    )


def prediction(
    *,
    variance=0.0,
):
    return CalibratedGaussianFullBoxPrediction(
        prediction_id="synthetic",
        latest_position_H0_m=(
            20.0,
            0.0,
            1.0,
        ),
        mean_displacement_H0_m=(
            (1.0, 0.0, 0.0),
            (3.0, 0.1, 0.0),
            (5.0, 0.2, 0.0),
            (10.0, 0.4, 0.0),
        ),
        calibrated_predictive_covariance_H0_m2=(
            covariance(
                float(
                    variance
                )
            )
        ),
    )


def actor_box():
    class Actor:
        pass

    actor = Actor()

    actor.box = Box3D(
        center_xyz=(
            20.0,
            0.0,
            1.0,
        ),
        length_m=4.5,
        width_m=2.0,
        height_m=1.6,
        yaw_rad=0.0,
    )

    actor.future_state_used = False
    actor.tracks_to_predict_used = False
    actor.objects_of_interest_used = False

    return actor


def grid():
    return OccupancyGrid(
        theta_centers_rad=tuple(
            np.linspace(
                -0.20,
                0.20,
                81,
                dtype=np.float64,
            )
        ),
        range_centers_m=tuple(
            np.linspace(
                15.0,
                35.0,
                81,
                dtype=np.float64,
            )
        ),
    )


class TestBlock64Part2AOccupancy(
    unittest.TestCase
):
    def test_01_grid_is_parameterized(
        self,
    ):
        item = grid()

        self.assertEqual(
            item.shape,
            (
                81,
                81,
            ),
        )

    def test_02_grid_requires_increasing_theta(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            OccupancyGrid(
                theta_centers_rad=(
                    0.0,
                    -0.1,
                ),
                range_centers_m=(
                    1.0,
                    2.0,
                ),
            )

    def test_03_grid_requires_nonnegative_range(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            OccupancyGrid(
                theta_centers_rad=(
                    -0.1,
                    0.1,
                ),
                range_centers_m=(
                    -1.0,
                    1.0,
                ),
            )

    def test_04_convex_hull_uses_polygon_not_rectangle(
        self,
    ):
        points = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )

        hull = convex_hull_theta_range(
            points
        )

        self.assertEqual(
            hull.shape,
            (
                3,
                2,
            ),
        )

    def test_05_one_box_raster_is_boolean(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=1,
                seed=11,
            )
        )

        mask = rasterize_projected_full_box(
            forecast.future[
                0
            ][
                0
            ],
            grid(),
        )

        self.assertEqual(
            mask.dtype,
            np.bool_,
        )

        self.assertEqual(
            mask.shape,
            (
                81,
                81,
            ),
        )

        self.assertGreater(
            int(
                np.count_nonzero(
                    mask
                )
            ),
            1,
        )

    def test_06_zero_uncertainty_is_binary_probability(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.0
                ),
                actor_box(),
                sample_count=8,
                seed=11,
            )
        )

        result = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
        )

        unique = set(
            np.unique(
                result.occupancy_probability
            ).tolist()
        )

        self.assertTrue(
            unique.issubset(
                {
                    0.0,
                    1.0,
                }
            )
        )

    def test_07_probability_equals_counts_over_n(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=17,
                seed=11,
            )
        )

        result = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
        )

        np.testing.assert_array_equal(
            result.occupancy_probability,
            (
                result
                .occupancy_counts
                .astype(
                    np.float64
                )
                /
                17.0
            ),
        )

    def test_08_probability_bounds(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=19,
                seed=11,
            )
        )

        probability = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
            .occupancy_probability
        )

        self.assertGreaterEqual(
            float(
                np.min(
                    probability
                )
            ),
            0.0,
        )

        self.assertLessEqual(
            float(
                np.max(
                    probability
                )
            ),
            1.0,
        )

    def test_09_same_seed_exact_occupancy_repeat(
        self,
    ):
        first_forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=23,
                seed=11,
            )
        )

        second_forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.05
                ),
                actor_box(),
                sample_count=23,
                seed=11,
            )
        )

        first = (
            estimate_actor_occupancy_probability(
                first_forecast,
                grid(),
            )
        )

        second = (
            estimate_actor_occupancy_probability(
                second_forecast,
                grid(),
            )
        )

        np.testing.assert_array_equal(
            first.occupancy_counts,
            second.occupancy_counts,
        )

        np.testing.assert_array_equal(
            first.occupancy_probability,
            second.occupancy_probability,
        )

    def test_10_uncertainty_produces_nonbinary_cells(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.08
                ),
                actor_box(),
                sample_count=64,
                seed=11,
            )
        )

        probability = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
            .occupancy_probability
        )

        fractional = (
            probability
            [
                (
                    probability
                    >
                    0.0
                )
                &
                (
                    probability
                    <
                    1.0
                )
            ]
        )

        self.assertGreater(
            fractional.size,
            0,
        )

    def test_11_output_shape_is_horizon_theta_range(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    variance=0.04
                ),
                actor_box(),
                sample_count=5,
                seed=11,
            )
        )

        result = (
            estimate_actor_occupancy_probability(
                forecast,
                grid(),
            )
        )

        self.assertEqual(
            result.shape,
            (
                4,
                81,
                81,
            ),
        )

    def test_12_no_gamma_threshold_argument(
        self,
    ):
        signature = inspect.signature(
            estimate_actor_occupancy_probability
        )

        names = set(
            signature.parameters.keys()
        )

        self.assertNotIn(
            "gamma",
            names,
        )

        self.assertNotIn(
            "threshold",
            names,
        )

    def test_13_no_class_policy_argument(
        self,
    ):
        signature = inspect.signature(
            estimate_actor_occupancy_probability
        )

        names = set(
            signature.parameters.keys()
        )

        self.assertFalse(
            any(
                "class"
                in name.lower()
                or
                "margin"
                in name.lower()
                or
                "floor"
                in name.lower()
                for name in names
            )
        )

    def test_14_no_centroid_api_in_occupancy_kernel(
        self,
    ):
        source = inspect.getsource(
            estimate_actor_occupancy_probability
        )

        self.assertNotIn(
            "centroid",
            source.lower(),
        )

    def test_15_all_eight_corners_required(
        self,
    ):
        source = inspect.getsource(
            rasterize_projected_full_box
        )

        self.assertIn(
            "_projected_theta_range_points",
            source,
        )

        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=1,
                seed=11,
            )
        )

        self.assertEqual(
            forecast.future[
                0
            ][
                0
            ]
            .projection
            .corners
            .xyz
            .shape,
            (
                8,
                3,
            ),
        )


if __name__ == "__main__":
    unittest.main()
'''


class ControlledBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(
        condition
    ):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return sha256_file(
        path
    )


def run_tests(
    pattern: str,
) -> dict[str, Any]:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            pattern,
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    count = (
        int(
            match.group(
                1
            )
        )
        if match
        else None
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "tests":
            count,

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -40:
                ]
            ),
    }


def main() -> None:
    try:
        print(
            "=" * 72
        )

        print(
            "BLOCK 6.4 PART2A — PARAMETERIZED PROBABILISTIC OCCUPANCY"
        )

        print(
            "=" * 72
        )

        # ====================================================
        # A. Part1 frozen seal
        # ====================================================

        print()
        print(
            "===== A. BLOCK6.4 PART1 SEAL ====="
        )

        for path, expected, label in (
            (
                PART1_REPORT,
                EXPECTED[
                    "part1_report"
                ],
                "Part1 report",
            ),
            (
                PART1_CONTRACT,
                EXPECTED[
                    "part1_contract"
                ],
                "Part1 contract",
            ),
            (
                PART1_MODULE,
                EXPECTED[
                    "part1_module"
                ],
                "Part1 scientific module",
            ),
        ):
            require(
                path.is_file(),
                f"Missing {label}: {path}",
            )

            actual = sha256_file(
                path
            )

            require(
                actual
                ==
                expected,
                (
                    f"{label} SHA changed: "
                    f"{actual}"
                ),
            )

            print(
                f"{label:24s} = EXACT PASS"
            )

        part1 = json.loads(
            PART1_REPORT.read_text(
                encoding="utf-8"
            )
        )

        require(
            part1.get(
                "status"
            )
            ==
            (
                "PASS_BLOCK64_PART1_"
                "STOCHASTIC_FULL_BOX_KERNEL"
            ),
            (
                "Unexpected Part1 status."
            ),
        )

        # ====================================================
        # B. Materialize occupancy kernel
        # ====================================================

        print()
        print(
            "===== B. MATERIALIZE P_OCC KERNEL ====="
        )

        require(
            not MODULE.exists(),
            (
                "Occupancy module already exists; "
                "refusing silent overwrite."
            ),
        )

        require(
            not TEST.exists(),
            (
                "Occupancy test already exists; "
                "refusing silent overwrite."
            ),
        )

        MODULE.write_text(
            MODULE_SOURCE,
            encoding="utf-8",
        )

        TEST.write_text(
            TEST_SOURCE,
            encoding="utf-8",
        )

        compile_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(
                    MODULE
                ),
                str(
                    TEST
                ),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        require(
            compile_result.returncode
            ==
            0,
            (
                "Part2A compile failed:\n"
                +
                compile_result.stdout
            ),
        )

        module_sha = sha256_file(
            MODULE
        )

        test_sha = sha256_file(
            TEST
        )

        print(
            "module =",
            MODULE,
        )

        print(
            "module SHA256 =",
            module_sha,
        )

        print(
            "test =",
            TEST,
        )

        print(
            "test SHA256 =",
            test_sha,
        )

        print(
            "compile = PASS"
        )

        # ====================================================
        # C. Dedicated tests
        # ====================================================

        print()
        print(
            "===== C. PART2A DEDICATED TESTS ====="
        )

        dedicated = run_tests(
            (
                "test_block64_part2a_"
                "probabilistic_occupancy.py"
            )
        )

        print(
            dedicated[
                "tail"
            ]
        )

        require(
            dedicated[
                "returncode"
            ]
            ==
            0,
            (
                "Part2A dedicated tests failed."
            ),
        )

        require(
            dedicated[
                "tests"
            ]
            ==
            15,
            (
                "Expected 15 Part2A tests, got "
                f"{dedicated['tests']}."
            ),
        )

        print(
            "dedicated tests = PASS | 15"
        )

        # ====================================================
        # D. Full regression
        # ====================================================

        print()
        print(
            "===== D. FULL STAGE6 REGRESSION ====="
        )

        regression = run_tests(
            "test_*.py"
        )

        print(
            regression[
                "tail"
            ]
        )

        require(
            regression[
                "returncode"
            ]
            ==
            0,
            (
                "Full Stage6 regression failed."
            ),
        )

        require(
            regression[
                "tests"
            ]
            ==
            127,
            (
                "Expected 112 + 15 = 127 tests; got "
                f"{regression['tests']}."
            ),
        )

        print(
            "full regression = PASS | 127"
        )

        # ====================================================
        # E. Semantic contract
        # ====================================================

        print()
        print(
            "===== E. PART2A OCCUPANCY CONTRACT ====="
        )

        contract = {
            "stage":
                6,

            "block":
                "6.4-Part2A",

            "status":
                (
                    "PASS_PARAMETERIZED_PROBABILISTIC_"
                    "OCCUPANCY_CONTRACT"
                ),

            "upstream": {
                "part1_report_sha256":
                    EXPECTED[
                        "part1_report"
                    ],

                "part1_contract_sha256":
                    EXPECTED[
                        "part1_contract"
                    ],

                "part1_scientific_module_sha256":
                    EXPECTED[
                        "part1_module"
                    ],
            },

            "occupancy_definition": {
                "symbol":
                    "P_occ(theta,r,tau)",

                "per_actor":
                    True,

                "estimator":
                    (
                        "Monte_Carlo_fraction_of_sampled_"
                        "full_boxes_covering_cell_center"
                    ),

                "exact_formula":
                    (
                        "occupancy_count(theta,r,tau) / "
                        "sample_count"
                    ),

                "shape":
                    "[4,N_theta,N_range]",

                "horizons_s":
                    [
                        0.1,
                        0.3,
                        0.5,
                        1.0,
                    ],
            },

            "geometry": {
                "source":
                    "sampled_future_full_Box3D",

                "physical_corners_per_box":
                    8,

                "projection_coordinates":
                    "theta_and_ground_range",

                "raster_envelope":
                    (
                        "convex_hull_of_all_eight_"
                        "projected_physical_corners"
                    ),

                "rectangular_bounding_box_as_mask":
                    False,

                "centroid_only":
                    False,

                "Gaussian_ellipse_as_box_surrogate":
                    False,
            },

            "grid": {
                "type":
                    "caller_parameterized_cell_centers",

                "theta_axis":
                    "strictly_increasing_radians",

                "range_axis":
                    "strictly_increasing_nonnegative_meters",

                "scientific_grid_frozen":
                    False,

                "PartA_grid_reused_as_final_without_study":
                    False,
            },

            "not_applied_here": {
                "occupancy_gamma":
                    True,

                "binary_actor_mask":
                    True,

                "multi_actor_mask_aggregation":
                    True,

                "class_aware_policy":
                    True,

                "class_margin":
                    True,

                "dimming_floor":
                    True,

                "temporal_smoothing":
                    True,

                "actuation_rate_limit":
                    True,
            },

            "not_yet_frozen": {
                "MC_sample_count":
                    True,

                "MC_seed_policy":
                    True,

                "scientific_grid":
                    True,

                "occupancy_gamma":
                    True,
            },

            "scientific_execution": {
                "real_development_posterior_sampled":
                    False,

                "formal_evaluation":
                    False,

                "model_forward":
                    False,

                "training":
                    False,

                "recalibration":
                    False,

                "parameter_tuning":
                    False,
            },

            "implementation": {
                "module":
                    str(
                        MODULE
                    ),

                "module_sha256":
                    module_sha,

                "test":
                    str(
                        TEST
                    ),

                "test_sha256":
                    test_sha,
            },
        }

        contract_sha = write_json(
            CONTRACT,
            contract,
        )

        print(
            "contract =",
            CONTRACT,
        )

        print(
            "contract SHA256 =",
            contract_sha,
        )

        # ====================================================
        # F. Report
        # ====================================================

        print()
        print(
            "===== F. PART2A REPORT ====="
        )

        report = {
            "stage":
                6,

            "block":
                "6.4-Part2A",

            "status":
                (
                    "PASS_BLOCK64_PART2A_"
                    "PROBABILISTIC_OCCUPANCY_KERNEL"
                ),

            "contract": {
                "path":
                    str(
                        CONTRACT
                    ),

                "sha256":
                    contract_sha,
            },

            "implementation": {
                "module":
                    str(
                        MODULE
                    ),

                "module_sha256":
                    module_sha,

                "test":
                    str(
                        TEST
                    ),

                "test_sha256":
                    test_sha,
            },

            "tests": {
                "dedicated":
                    {
                        "status":
                            "PASS",

                        "count":
                            15,
                    },

                "full_stage6":
                    {
                        "status":
                            "PASS",

                        "count":
                            127,
                    },
            },

            "semantics": {
                "P_occ_parameterized":
                    True,

                "full_box":
                    True,

                "eight_corner_hull":
                    True,

                "centroid_only":
                    False,

                "gamma_applied":
                    False,

                "multi_actor_aggregation":
                    False,

                "class_aware":
                    False,
            },

            "scope": {
                "MC_N_frozen":
                    False,

                "seed_policy_frozen":
                    False,

                "grid_frozen":
                    False,

                "gamma_frozen":
                    False,

                "real_development_sampling":
                    False,

                "formal_evaluation":
                    False,
            },

            "next":
                (
                    "BLOCK6.4_PART2B_REAL_DEVELOPMENT_"
                    "MC_CONVERGENCE_AND_NUMERIC_FREEZE"
                ),
        }

        report_sha = write_json(
            REPORT,
            report,
        )

        print(
            "report =",
            REPORT,
        )

        print(
            "report SHA256 =",
            report_sha,
        )

        # ====================================================
        # G. Final
        # ====================================================

        print()
        print(
            "=" * 72
        )

        print(
            "BLOCK 6.4 PART2A — FINAL"
        )

        print(
            "=" * 72
        )

        print(
            "STATUS = "
            "PASS_BLOCK64_PART2A_PROBABILISTIC_OCCUPANCY_KERNEL"
        )

        print(
            "P_occ definition             = "
            "MC COVERAGE FRACTION"
        )

        print(
            "per-actor occupancy          = YES"
        )

        print(
            "full sampled boxes           = YES"
        )

        print(
            "physical corners per box     = 8"
        )

        print(
            "raster envelope              = "
            "CONVEX HULL OF 8 PROJECTED CORNERS"
        )

        print(
            "centroid-only                = NO"
        )

        print(
            "gamma threshold applied      = NO"
        )

        print(
            "multi-actor aggregation      = NO"
        )

        print(
            "class-aware policy           = NO"
        )

        print(
            "MC sample count frozen       = NO"
        )

        print(
            "seed policy frozen           = NO"
        )

        print(
            "scientific grid frozen       = NO"
        )

        print(
            "real development sampling    = NO"
        )

        print(
            "formal evaluation            = NO"
        )

        print(
            "dedicated tests              = PASS | 15"
        )

        print(
            "full Stage6 regression       = PASS | 127"
        )

        print(
            "module SHA256                =",
            module_sha,
        )

        print(
            "contract SHA256              =",
            contract_sha,
        )

        print(
            "report SHA256                =",
            report_sha,
        )

        print(
            "NEXT = BLOCK6.4 PART2B "
            "REAL DEVELOPMENT MC CONVERGENCE + NUMERIC FREEZE"
        )

        print(
            "terminal remains open = YES"
        )

        print(
            "=" * 72
        )

    except BaseException as exc:
        print()
        print(
            "=" * 72
        )

        print(
            "BLOCK6.4 PART2A — CONTROLLED BLOCK"
        )

        print(
            "=" * 72
        )

        print(
            "exception type =",
            type(
                exc
            ).__name__,
        )

        print(
            "exception      =",
            str(
                exc
            ),
        )

        print()

        traceback.print_exc(
            limit=18
        )

        print()

        print(
            "formal evaluation      = NO"
        )

        print(
            "model forward          = NO"
        )

        print(
            "training/recalibration = NO / NO"
        )

        print(
            "upstream modification  = NO"
        )

        print(
            "terminal remains open  = YES"
        )

        print(
            "=" * 72
        )

    # Deliberately no sys.exit().


if __name__ == "__main__":
    main()

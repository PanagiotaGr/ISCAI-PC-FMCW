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

MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "probabilistic_full_box.py"
)

TEST = (
    S6
    / "tests"
    / "test_block64_part1_stochastic_full_box.py"
)

CONTRACT = (
    S6
    / "configs"
    / "block64_part1_stochastic_full_box_kernel_contract.json"
)

REPORT = (
    S6
    / "reports"
    / "block64_part1_stochastic_full_box_kernel.json"
)

CLOSURE63 = (
    S6
    / "reports"
    / "block63_final_closure.json"
)

HANDOFF63 = (
    S6
    / "artifacts"
    / "block63"
    / "block63_to_block64_handoff.json"
)

FREEZE63 = (
    S6
    / "artifacts"
    / "block63"
    / "block63_final_freeze_manifest.json"
)

POSTERIOR = (
    S6
    / "artifacts"
    / "block63"
    / "identity_safe_development_gaussian_posterior.jsonl"
)


EXPECTED = {
    "closure63":
        "c451c965fa4b8a237f36358da9c1c36436a4317ee0b1501e170be2c804977d1c",

    "handoff63":
        "611f1679b83fb1273cf2d1836a03073e3c557704c9656741d46ea19f80ca18c6",

    "freeze63":
        "230b800e3889a412202306c450824152b7bff2aa6e35d12376a22d007597e8f2",

    "posterior":
        "318b22dfc3788195a1432169075d1c8654f0d183337711918db2303b5668f5ea",
}


MODULE_SOURCE = r'''from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    HORIZONS_S as STAGE5_HORIZONS_S,
    TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
    resolve_samplewise_headings,
    sample_product_of_horizon_gaussians,
)

from iscai_stage6.adb.geometry import (
    Box3D,
    project_box_to_headlamp,
    validate_controller_provenance,
)


HORIZONS_S = tuple(
    float(value)
    for value in STAGE5_HORIZONS_S
)

if HORIZONS_S != (
    0.1,
    0.3,
    0.5,
    1.0,
):
    raise RuntimeError(
        "Frozen Stage5 horizons changed."
    )


STATE_SOURCE = "predicted_sample"
ORIENTATION_SOURCE = "predicted_tangent"


Vec3 = tuple[
    float,
    float,
    float,
]


def _finite_vec3(
    value,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must have shape (3,), "
            f"got {array.shape}."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return array


def _finite_horizon_vec3(
    value,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape != (
        4,
        3,
    ):
        raise ValueError(
            f"{name} must have shape (4,3), "
            f"got {array.shape}."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return array


def _finite_horizon_covariance(
    value,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape != (
        4,
        3,
        3,
    ):
        raise ValueError(
            "calibrated_predictive_covariance_H0_m2 "
            "must have shape (4,3,3), "
            f"got {array.shape}."
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "calibrated predictive covariance "
            "must be finite."
        )

    # Do not silently symmetrize or otherwise modify the
    # calibrated covariance. Stage5 owns the exact Gaussian
    # sampling validity checks/factorization.
    return array


def _readonly_copy(
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


def _require_causal_actor_box(
    actor_box,
) -> None:
    required_flags = (
        "future_state_used",
        "tracks_to_predict_used",
        "objects_of_interest_used",
    )

    for name in required_flags:
        if not hasattr(
            actor_box,
            name,
        ):
            raise ValueError(
                "actor_box lacks required causal "
                f"provenance field {name!r}."
            )

        if bool(
            getattr(
                actor_box,
                name,
            )
        ):
            raise ValueError(
                "Non-causal actor-box provenance: "
                f"{name}=True."
            )

    if not hasattr(
        actor_box,
        "box",
    ):
        raise ValueError(
            "actor_box lacks current causal Box3D."
        )

    if not isinstance(
        actor_box.box,
        Box3D,
    ):
        raise TypeError(
            "actor_box.box must be Box3D."
        )


@dataclass(frozen=True)
class CalibratedGaussianFullBoxPrediction:
    """
    Identity-safe calibrated Gaussian trajectory posterior
    required by Block6.4.

    Mean is the frozen Stage4 metric-H0 displacement mean.

    Covariance is the frozen calibrated predictive covariance
    of that displacement. Adding the deterministic causal
    anchor changes the mean into absolute H0 positions but
    does not change covariance.
    """

    prediction_id: str

    latest_position_H0_m: Vec3

    mean_displacement_H0_m: tuple[
        Vec3,
        Vec3,
        Vec3,
        Vec3,
    ]

    calibrated_predictive_covariance_H0_m2: tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
    ]

    def __post_init__(
        self,
    ) -> None:
        if not str(
            self.prediction_id
        ):
            raise ValueError(
                "prediction_id must be non-empty."
            )

        _finite_vec3(
            self.latest_position_H0_m,
            name="latest_position_H0_m",
        )

        _finite_horizon_vec3(
            self.mean_displacement_H0_m,
            name="mean_displacement_H0_m",
        )

        _finite_horizon_covariance(
            self.calibrated_predictive_covariance_H0_m2
        )


@dataclass(frozen=True)
class StochasticFutureProjectedBox:
    sample_index: int

    horizon_index: int

    horizon_s: float

    box: Box3D

    projection: Any

    heading_fallback_used: bool

    state_source: str = STATE_SOURCE

    orientation_source: str = ORIENTATION_SOURCE


@dataclass(frozen=True)
class StochasticFullBoxForecast:
    prediction_id: str

    sample_count: int

    seed: int

    absolute_mean_H0_m: np.ndarray

    trajectory_samples_H0_m: np.ndarray

    heading_samples_rad: np.ndarray

    heading_fallback_mask: np.ndarray

    future: tuple[
        tuple[
            StochasticFutureProjectedBox,
            StochasticFutureProjectedBox,
            StochasticFutureProjectedBox,
            StochasticFutureProjectedBox,
        ],
        ...,
    ]

    joint_sampling_semantics: str = (
        TRAJECTORY_JOINT_SAMPLING_SEMANTICS
    )

    @property
    def projected_box_count(
        self,
    ) -> int:
        return int(
            self.sample_count
            *
            len(
                HORIZONS_S
            )
        )

    @property
    def projected_corner_count(
        self,
    ) -> int:
        return int(
            self.projected_box_count
            *
            8
        )


def absolute_mean_positions_H0_m(
    prediction: CalibratedGaussianFullBoxPrediction,
) -> np.ndarray:
    """
    Convert the frozen metric-H0 displacement mean to future
    absolute H0 positions.

    Crucially, this uses the predictor's retained causal
    anchor. It never recenters the prediction on the matched
    annotation box center.
    """

    anchor = _finite_vec3(
        prediction.latest_position_H0_m,
        name="latest_position_H0_m",
    )

    mean_displacement = _finite_horizon_vec3(
        prediction.mean_displacement_H0_m,
        name="mean_displacement_H0_m",
    )

    result = (
        anchor[
            None,
            :
        ]
        +
        mean_displacement
    )

    return _readonly_copy(
        result,
        dtype=np.float64,
    )


def sample_calibrated_future_positions_H0_m(
    prediction: CalibratedGaussianFullBoxPrediction,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """
    Sample future absolute H0 positions using the frozen
    Stage5 product-of-calibrated-horizon-Gaussians rule.

    sample_count and seed are runtime parameters here.
    Block6.4 Part1 deliberately does NOT freeze their
    scientific values.
    """

    count = int(
        sample_count
    )

    if count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    mean_absolute = (
        absolute_mean_positions_H0_m(
            prediction
        )
    )

    covariance = (
        _finite_horizon_covariance(
            prediction
            .calibrated_predictive_covariance_H0_m2
        )
    )

    samples = (
        sample_product_of_horizon_gaussians(
            trajectory_mean_h0_m=(
                mean_absolute
            ),
            calibrated_covariance_h0_m2=(
                covariance
            ),
            sample_count=count,
            seed=int(
                seed
            ),
        )
    )

    samples = np.asarray(
        samples,
        dtype=np.float64,
    )

    if samples.shape != (
        count,
        4,
        3,
    ):
        raise RuntimeError(
            "Frozen Stage5 Gaussian sampler "
            "returned unexpected shape "
            f"{samples.shape}."
        )

    if not np.all(
        np.isfinite(
            samples
        )
    ):
        raise RuntimeError(
            "Stage5 trajectory samples "
            "contain non-finite values."
        )

    return _readonly_copy(
        samples,
        dtype=np.float64,
    )


def build_stochastic_future_full_boxes(
    prediction: CalibratedGaussianFullBoxPrediction,
    actor_box,
    *,
    sample_count: int,
    seed: int,
) -> StochasticFullBoxForecast:
    """
    Propagate calibrated trajectory uncertainty through
    physical future full-box geometry.

    For every trajectory sample and every horizon:
      sampled position -> Box3D center
      sampled tangent  -> box yaw
      current causal L/W/H -> held geometry
      8 physical corners -> headlamp projection

    No centroid-only probability approximation is used.
    """

    _require_causal_actor_box(
        actor_box
    )

    count = int(
        sample_count
    )

    if count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    anchor = _finite_vec3(
        prediction.latest_position_H0_m,
        name="latest_position_H0_m",
    )

    samples = (
        sample_calibrated_future_positions_H0_m(
            prediction,
            sample_count=count,
            seed=int(
                seed
            ),
        )
    )

    headings, fallback = (
        resolve_samplewise_headings(
            trajectory_samples_h0_m=(
                samples
            ),
            current_position_h0_m=(
                anchor
            ),
            current_heading_h0_rad=float(
                actor_box.box.yaw_rad
            ),
        )
    )

    headings = np.asarray(
        headings,
        dtype=np.float64,
    )

    fallback = np.asarray(
        fallback,
        dtype=bool,
    )

    if headings.shape != (
        count,
        4,
    ):
        raise RuntimeError(
            "Stage5 heading resolver returned "
            f"unexpected shape {headings.shape}."
        )

    if fallback.shape != (
        count,
        4,
    ):
        raise RuntimeError(
            "Stage5 heading fallback mask returned "
            f"unexpected shape {fallback.shape}."
        )

    if not np.all(
        np.isfinite(
            headings
        )
    ):
        raise RuntimeError(
            "Resolved headings contain "
            "non-finite values."
        )

    validate_controller_provenance(
        state_source=STATE_SOURCE,
        orientation_source=ORIENTATION_SOURCE,
        controller_path=True,
    )

    sample_sequences = []

    for sample_index in range(
        count
    ):
        horizon_items = []

        for horizon_index, horizon_s in enumerate(
            HORIZONS_S
        ):
            center = tuple(
                float(value)
                for value in
                samples[
                    sample_index,
                    horizon_index,
                    :
                ]
            )

            box = Box3D(
                center_xyz=center,
                length_m=float(
                    actor_box.box.length_m
                ),
                width_m=float(
                    actor_box.box.width_m
                ),
                height_m=float(
                    actor_box.box.height_m
                ),
                yaw_rad=float(
                    headings[
                        sample_index,
                        horizon_index
                    ]
                ),
            )

            projection = (
                project_box_to_headlamp(
                    box,
                    state_source=STATE_SOURCE,
                    orientation_source=(
                        ORIENTATION_SOURCE
                    ),
                    controller_path=True,
                )
            )

            corners = np.asarray(
                projection.corners.xyz,
                dtype=np.float64,
            )

            if corners.shape != (
                8,
                3,
            ):
                raise RuntimeError(
                    "Full-box projection must contain "
                    "exactly eight physical corners."
                )

            horizon_items.append(
                StochasticFutureProjectedBox(
                    sample_index=int(
                        sample_index
                    ),
                    horizon_index=int(
                        horizon_index
                    ),
                    horizon_s=float(
                        horizon_s
                    ),
                    box=box,
                    projection=projection,
                    heading_fallback_used=bool(
                        fallback[
                            sample_index,
                            horizon_index
                        ]
                    ),
                )
            )

        sample_sequences.append(
            tuple(
                horizon_items
            )
        )

    future = tuple(
        sample_sequences
    )

    absolute_mean = (
        absolute_mean_positions_H0_m(
            prediction
        )
    )

    return StochasticFullBoxForecast(
        prediction_id=str(
            prediction.prediction_id
        ),
        sample_count=count,
        seed=int(
            seed
        ),
        absolute_mean_H0_m=(
            _readonly_copy(
                absolute_mean,
                dtype=np.float64,
            )
        ),
        trajectory_samples_H0_m=(
            _readonly_copy(
                samples,
                dtype=np.float64,
            )
        ),
        heading_samples_rad=(
            _readonly_copy(
                headings,
                dtype=np.float64,
            )
        ),
        heading_fallback_mask=(
            _readonly_copy(
                fallback,
                dtype=bool,
            )
        ),
        future=future,
    )
'''


TEST_SOURCE = r'''from __future__ import annotations

from types import SimpleNamespace
import inspect
import unittest

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
)

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.probabilistic_full_box import (
    HORIZONS_S,
    ORIENTATION_SOURCE,
    STATE_SOURCE,
    CalibratedGaussianFullBoxPrediction,
    absolute_mean_positions_H0_m,
    build_stochastic_future_full_boxes,
    sample_calibrated_future_positions_H0_m,
)


def covariance(
    variance=0.04,
):
    matrix = np.diag(
        [
            float(
                variance
            ),
            float(
                variance
            ),
            float(
                variance
            ),
        ]
    )

    return tuple(
        tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in matrix
        )
        for _ in range(
            4
        )
    )


def prediction(
    *,
    zero_motion=False,
    variance=0.04,
):
    if zero_motion:
        mean = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )

    else:
        mean = (
            (1.0, 0.0, 0.0),
            (3.0, 0.2, 0.0),
            (5.0, 0.4, 0.0),
            (10.0, 0.8, 0.0),
        )

    return CalibratedGaussianFullBoxPrediction(
        prediction_id="synthetic_prediction",
        latest_position_H0_m=(
            20.0,
            0.0,
            1.0,
        ),
        mean_displacement_H0_m=mean,
        calibrated_predictive_covariance_H0_m2=(
            covariance(
                variance
            )
        ),
    )


def actor_box(
    *,
    center=(
        20.0,
        0.0,
        1.0,
    ),
    yaw=0.3,
    future_state_used=False,
):
    return SimpleNamespace(
        box=Box3D(
            center_xyz=center,
            length_m=4.4,
            width_m=1.9,
            height_m=1.6,
            yaw_rad=float(
                yaw
            ),
        ),
        future_state_used=bool(
            future_state_used
        ),
        tracks_to_predict_used=False,
        objects_of_interest_used=False,
    )


class TestBlock64Part1StochasticFullBox(
    unittest.TestCase
):
    def test_01_frozen_horizons_and_joint_semantics(
        self,
    ):
        self.assertEqual(
            HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

        self.assertEqual(
            TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
            (
                "product_of_frozen_calibrated_"
                "per_horizon_Gaussian_marginals"
            ),
        )

    def test_02_absolute_mean_is_anchor_plus_displacement(
        self,
    ):
        item = prediction()

        absolute = (
            absolute_mean_positions_H0_m(
                item
            )
        )

        expected = (
            np.asarray(
                item.latest_position_H0_m,
                dtype=np.float64,
            )[
                None,
                :
            ]
            +
            np.asarray(
                item.mean_displacement_H0_m,
                dtype=np.float64,
            )
        )

        np.testing.assert_array_equal(
            absolute,
            expected,
        )

    def test_03_same_seed_exact_repeat(
        self,
    ):
        item = prediction()

        first = (
            sample_calibrated_future_positions_H0_m(
                item,
                sample_count=32,
                seed=123,
            )
        )

        second = (
            sample_calibrated_future_positions_H0_m(
                item,
                sample_count=32,
                seed=123,
            )
        )

        self.assertTrue(
            np.array_equal(
                first,
                second,
            )
        )

    def test_04_different_seed_changes_samples(
        self,
    ):
        item = prediction()

        first = (
            sample_calibrated_future_positions_H0_m(
                item,
                sample_count=32,
                seed=123,
            )
        )

        second = (
            sample_calibrated_future_positions_H0_m(
                item,
                sample_count=32,
                seed=124,
            )
        )

        self.assertFalse(
            np.array_equal(
                first,
                second,
            )
        )

    def test_05_forecast_shape_and_counts(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=5,
                seed=123,
            )
        )

        self.assertEqual(
            forecast.trajectory_samples_H0_m.shape,
            (
                5,
                4,
                3,
            ),
        )

        self.assertEqual(
            forecast.heading_samples_rad.shape,
            (
                5,
                4,
            ),
        )

        self.assertEqual(
            len(
                forecast.future
            ),
            5,
        )

        self.assertTrue(
            all(
                len(
                    sequence
                )
                ==
                4
                for sequence in
                forecast.future
            )
        )

        self.assertEqual(
            forecast.projected_box_count,
            20,
        )

        self.assertEqual(
            forecast.projected_corner_count,
            160,
        )

    def test_06_every_future_box_has_eight_corners(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=3,
                seed=123,
            )
        )

        for sequence in forecast.future:
            for item in sequence:
                self.assertEqual(
                    item.projection.corners.xyz.shape,
                    (
                        8,
                        3,
                    ),
                )

                self.assertGreater(
                    item.projection.theta_span_rad,
                    0.0,
                )

                self.assertGreater(
                    item.projection.ground_range_max_m,
                    item.projection.ground_range_min_m,
                )

    def test_07_current_dimensions_are_preserved(
        self,
    ):
        current = actor_box()

        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                current,
                sample_count=3,
                seed=123,
            )
        )

        for sequence in forecast.future:
            for item in sequence:
                self.assertEqual(
                    item.box.length_m,
                    current.box.length_m,
                )

                self.assertEqual(
                    item.box.width_m,
                    current.box.width_m,
                )

                self.assertEqual(
                    item.box.height_m,
                    current.box.height_m,
                )

    def test_08_future_centers_equal_sampled_positions(
        self,
    ):
        # Deliberately mismatch actor-box center from predictor
        # anchor to prove there is no recentering.
        current = actor_box(
            center=(
                24.0,
                2.0,
                1.0,
            )
        )

        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                current,
                sample_count=4,
                seed=123,
            )
        )

        for sample_index, sequence in enumerate(
            forecast.future
        ):
            for horizon_index, item in enumerate(
                sequence
            ):
                np.testing.assert_array_equal(
                    np.asarray(
                        item.box.center_xyz
                    ),
                    forecast.trajectory_samples_H0_m[
                        sample_index,
                        horizon_index,
                        :
                    ],
                )

    def test_09_controller_provenance_is_exact(
        self,
    ):
        forecast = (
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(),
                sample_count=2,
                seed=123,
            )
        )

        self.assertEqual(
            STATE_SOURCE,
            "predicted_sample",
        )

        self.assertEqual(
            ORIENTATION_SOURCE,
            "predicted_tangent",
        )

        for sequence in forecast.future:
            for item in sequence:
                self.assertEqual(
                    item.state_source,
                    "predicted_sample",
                )

                self.assertEqual(
                    item.orientation_source,
                    "predicted_tangent",
                )

    def test_10_low_speed_heading_carry_forward(
        self,
    ):
        current_heading = 0.47

        # The frozen heading resolver receives the heading
        # stored by the current causal Box3D. Box3D may
        # canonicalize/wrap the floating-point yaw, so the
        # exact semantic reference is actor.box.yaw_rad,
        # not the pre-construction Python literal.
        current = actor_box(
            yaw=current_heading
        )

        forecast = (
            build_stochastic_future_full_boxes(
                prediction(
                    zero_motion=True,
                    variance=0.0,
                ),
                current,
                sample_count=2,
                seed=123,
            )
        )

        np.testing.assert_array_equal(
            forecast.heading_fallback_mask,
            np.ones(
                (
                    2,
                    4,
                ),
                dtype=bool,
            ),
        )

        np.testing.assert_allclose(
            forecast.heading_samples_rad,
            float(
                current.box.yaw_rad
            ),
            rtol=0.0,
            atol=0.0,
        )

    def test_11_calibrated_covariance_changes_support(
        self,
    ):
        deterministic = (
            sample_calibrated_future_positions_H0_m(
                prediction(
                    variance=0.0
                ),
                sample_count=32,
                seed=123,
            )
        )

        uncertain = (
            sample_calibrated_future_positions_H0_m(
                prediction(
                    variance=0.04
                ),
                sample_count=32,
                seed=123,
            )
        )

        mean = (
            absolute_mean_positions_H0_m(
                prediction(
                    variance=0.0
                )
            )
        )

        np.testing.assert_array_equal(
            deterministic,
            np.repeat(
                mean[
                    None,
                    :,
                    :
                ],
                32,
                axis=0,
            ),
        )

        self.assertFalse(
            np.array_equal(
                uncertain,
                deterministic,
            )
        )

    def test_12_noncausal_actor_box_is_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            build_stochastic_future_full_boxes(
                prediction(),
                actor_box(
                    future_state_used=True
                ),
                sample_count=2,
                seed=123,
            )

    def test_13_invalid_covariance_shape_is_rejected(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            CalibratedGaussianFullBoxPrediction(
                prediction_id="bad",
                latest_position_H0_m=(
                    20.0,
                    0.0,
                    1.0,
                ),
                mean_displacement_H0_m=(
                    (1.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (3.0, 0.0, 0.0),
                    (4.0, 0.0, 0.0),
                ),
                calibrated_predictive_covariance_H0_m2=(
                    np.zeros(
                        (
                            3,
                            3,
                            3,
                        ),
                        dtype=np.float64,
                    )
                ),
            )

    def test_14_no_cross_horizon_covariance_argument(
        self,
    ):
        signature = inspect.signature(
            CalibratedGaussianFullBoxPrediction
        )

        names = set(
            signature.parameters.keys()
        )

        self.assertIn(
            "calibrated_predictive_covariance_H0_m2",
            names,
        )

        self.assertFalse(
            any(
                "cross"
                in
                name.lower()
                or
                "12x12"
                in
                name.lower()
                for name in names
            )
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
    *,
    start_dir: Path,
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
                start_dir
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
            "BLOCK 6.4 PART1 — STOCHASTIC FULL-BOX KERNEL"
        )
        print(
            "=" * 72
        )

        # ====================================================
        # A. Frozen Block6.3 seal
        # ====================================================

        print()
        print(
            "===== A. FROZEN BLOCK6.3 HANDOFF SEAL ====="
        )

        for path, expected, label in (
            (
                CLOSURE63,
                EXPECTED[
                    "closure63"
                ],
                "closure",
            ),
            (
                HANDOFF63,
                EXPECTED[
                    "handoff63"
                ],
                "handoff",
            ),
            (
                FREEZE63,
                EXPECTED[
                    "freeze63"
                ],
                "freeze",
            ),
            (
                POSTERIOR,
                EXPECTED[
                    "posterior"
                ],
                "posterior",
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
                    f"Frozen {label} SHA changed: "
                    f"{actual}"
                ),
            )

            print(
                f"{label:10s} = EXACT PASS"
            )

        handoff = json.loads(
            HANDOFF63.read_text(
                encoding="utf-8"
            )
        )

        require(
            handoff.get(
                "status"
            )
            ==
            (
                "PASS_READY_FOR_BLOCK64_"
                "PROBABILISTIC_FULL_BOX_OCCUPANCY"
            ),
            (
                "Unexpected Block6.3 handoff status."
            ),
        )

        # ====================================================
        # B. Frozen posterior uncertainty boundary
        # ====================================================

        print()
        print(
            "===== B. CALIBRATED POSTERIOR BOUNDARY ====="
        )

        records = [
            json.loads(
                line
            )
            for line in
            POSTERIOR.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        require(
            len(
                records
            )
            ==
            27,
            (
                "Posterior record count changed."
            ),
        )

        for index, record in enumerate(
            records
        ):
            require(
                record.get(
                    "future_GT_used"
                )
                is False,
                (
                    f"future GT entered record {index}."
                ),
            )

            require(
                record.get(
                    "truth_id_used"
                )
                is False,
                (
                    f"truth ID entered record {index}."
                ),
            )

            require(
                record.get(
                    "tracks_to_predict_used"
                )
                is False,
                (
                    "tracks_to_predict entered "
                    f"record {index}."
                ),
            )

            covariance = record.get(
                "calibrated_predictive_covariance_H0_m2"
            )

            require(
                covariance
                is not None,
                (
                    "Calibrated predictive covariance "
                    f"missing at record {index}."
                ),
            )

        print(
            "records                        = 27"
        )

        print(
            "calibrated covariance          = PRESENT"
        )

        print(
            "future GT / truth ID           = NO / NO"
        )

        print(
            "tracks_to_predict              = NO"
        )

        # ====================================================
        # C. Materialize new implementation only
        # ====================================================

        print()
        print(
            "===== C. MATERIALIZE BLOCK6.4 PART1 KERNEL ====="
        )

        require(
            not MODULE.exists(),
            (
                "Target Block6.4 module already exists; "
                "refusing silent overwrite."
            ),
        )

        require(
            not TEST.exists(),
            (
                "Target Block6.4 test already exists; "
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

        compile_module = subprocess.run(
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
            compile_module.returncode
            ==
            0,
            (
                "Block6.4 Part1 compile failed:\n"
                +
                compile_module.stdout
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
        # D. Dedicated tests
        # ====================================================

        print()
        print(
            "===== D. BLOCK6.4 PART1 DEDICATED TESTS ====="
        )

        dedicated = run_tests(
            start_dir=(
                S6
                / "tests"
            ),
            pattern=(
                "test_block64_part1_"
                "stochastic_full_box.py"
            ),
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
                "Block6.4 Part1 dedicated "
                "tests failed."
            ),
        )

        require(
            dedicated[
                "tests"
            ]
            ==
            14,
            (
                "Expected 14 dedicated tests, "
                f"got {dedicated['tests']}."
            ),
        )

        print(
            "dedicated tests = PASS | tests = 14"
        )

        # ====================================================
        # E. Full Stage6 regression
        # ====================================================

        print()
        print(
            "===== E. FULL STAGE6 REGRESSION ====="
        )

        regression = run_tests(
            start_dir=(
                S6
                / "tests"
            ),
            pattern="test_*.py",
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
            112,
            (
                "Expected frozen 98 + new 14 "
                "Stage6 tests = 112; got "
                f"{regression['tests']}."
            ),
        )

        print(
            "full regression = PASS | tests =",
            regression[
                "tests"
            ],
        )

        # ====================================================
        # F. Contract — semantics only, no MC/grid numerics
        # ====================================================

        print()
        print(
            "===== F. BLOCK6.4 PART1 CONTRACT ====="
        )

        contract_payload = {
            "stage":
                6,

            "block":
                "6.4-Part1",

            "status":
                (
                    "PASS_STOCHASTIC_FULL_BOX_"
                    "KERNEL_CONTRACT"
                ),

            "upstream":
                {
                    "block63_closure_sha256":
                        EXPECTED[
                            "closure63"
                        ],

                    "block63_handoff_sha256":
                        EXPECTED[
                            "handoff63"
                        ],

                    "block63_freeze_sha256":
                        EXPECTED[
                            "freeze63"
                        ],

                    "identity_safe_posterior_sha256":
                        EXPECTED[
                            "posterior"
                        ],
                },

            "trajectory_uncertainty":
                {
                    "source":
                        (
                            "calibrated_predictive_"
                            "covariance_H0_m2"
                        ),

                    "mean_source":
                        "mean_displacement_H0_m",

                    "anchor_source":
                        "latest_position_H0_m",

                    "absolute_mean":
                        (
                            "latest_position_H0_m + "
                            "mean_displacement_H0_m"
                        ),

                    "joint_sampling_semantics":
                        (
                            "product_of_frozen_calibrated_"
                            "per_horizon_Gaussian_marginals"
                        ),

                    "cross_horizon_covariance_invented":
                        False,

                    "Stage5_sampler":
                        (
                            "sample_product_of_"
                            "horizon_gaussians"
                        ),

                    "sample_count":
                        "RUNTIME_PARAMETER_NOT_FROZEN",

                    "seed":
                        "RUNTIME_PARAMETER_NOT_FROZEN",
                },

            "heading":
                {
                    "source":
                        (
                            "resolve_samplewise_headings"
                        ),

                    "state_provenance":
                        "predicted_sample",

                    "orientation_provenance":
                        "predicted_tangent",

                    "low_speed_behavior":
                        (
                            "frozen_Stage5_previous_"
                            "causal_or_predicted_heading_"
                            "carry_forward"
                        ),

                    "future_GT_heading":
                        False,
                },

            "full_box_geometry":
                {
                    "center":
                        "sampled_absolute_H0_position",

                    "length_width_height":
                        "current_causal_box_held",

                    "corners_per_box":
                        8,

                    "centroid_only":
                        False,

                    "projector":
                        "project_box_to_headlamp",

                    "communication_codebook_reused":
                        False,
                },

            "not_yet_frozen":
                {
                    "MC_sample_count":
                        True,

                    "MC_seed_set":
                        True,

                    "occupancy_gamma":
                        True,

                    "scientific_illumination_grid":
                        True,

                    "class_margins":
                        True,

                    "dimming_floors":
                        True,

                    "temporal_smoothing":
                        True,

                    "actuation_rate_limits":
                        True,
                },

            "not_yet_implemented":
                {
                    "occupancy_rasterization":
                        True,

                    "occupancy_thresholding":
                        True,

                    "multi_actor_mask_aggregation":
                        True,

                    "class_aware_policy":
                        True,

                    "temporal_smoothing":
                        True,

                    "actuation_rate_limits":
                        True,
                },

            "scientific_execution":
                {
                    "real_posterior_sampling":
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

            "implementation":
                {
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
            contract_payload,
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
        # G. Part1 report
        # ====================================================

        print()
        print(
            "===== G. BLOCK6.4 PART1 REPORT ====="
        )

        report_payload = {
            "stage":
                6,

            "block":
                "6.4-Part1",

            "status":
                (
                    "PASS_BLOCK64_PART1_"
                    "STOCHASTIC_FULL_BOX_KERNEL"
                ),

            "contract":
                {
                    "path":
                        str(
                            CONTRACT
                        ),

                    "sha256":
                        contract_sha,
                },

            "implementation":
                {
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

            "tests":
                {
                    "dedicated":
                        {
                            "status":
                                "PASS",

                            "count":
                                14,
                        },

                    "full_stage6":
                        {
                            "status":
                                "PASS",

                            "count":
                                112,
                        },
                },

            "scientific_semantics":
                {
                    "calibrated_covariance_used":
                        True,

                    "full_box_uncertainty_propagation":
                        True,

                    "corners_per_sampled_box":
                        8,

                    "centroid_only":
                        False,

                    "cross_horizon_covariance_invented":
                        False,

                    "samplewise_tangent_heading":
                        True,

                    "future_GT":
                        False,

                    "perfect_identity":
                        False,
                },

            "scope":
                {
                    "MC_N_frozen":
                        False,

                    "seed_set_frozen":
                        False,

                    "gamma_frozen":
                        False,

                    "scientific_grid_frozen":
                        False,

                    "real_posterior_sampled":
                        False,

                    "formal_evaluation":
                        False,

                    "training":
                        False,

                    "recalibration":
                        False,
                },

            "next":
                (
                    "BLOCK6.4_PART2_DEVELOPMENT_"
                    "MC_CONVERGENCE_AND_OCCUPANCY_KERNEL"
                ),
        }

        report_sha = write_json(
            REPORT,
            report_payload,
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
        # H. Final
        # ====================================================

        print()
        print(
            "=" * 72
        )

        print(
            "BLOCK 6.4 PART1 — FINAL"
        )

        print(
            "=" * 72
        )

        print(
            "STATUS = "
            "PASS_BLOCK64_PART1_STOCHASTIC_FULL_BOX_KERNEL"
        )

        print(
            "calibrated covariance        = USED"
        )

        print(
            "trajectory sampling semantics = "
            "FROZEN STAGE5 PRODUCT OF MARGINALS"
        )

        print(
            "cross-horizon covariance      = NOT INVENTED"
        )

        print(
            "samplewise heading            = PREDICTED TANGENT"
        )

        print(
            "sampled future geometry       = FULL BOX"
        )

        print(
            "corners per sampled box       = 8"
        )

        print(
            "centroid-only                 = NO"
        )

        print(
            "MC sample count frozen        = NO"
        )

        print(
            "MC seed set frozen            = NO"
        )

        print(
            "occupancy gamma frozen        = NO"
        )

        print(
            "scientific grid frozen        = NO"
        )

        print(
            "real posterior sampled        = NO"
        )

        print(
            "model forward                 = NO"
        )

        print(
            "training/recalibration        = NO / NO"
        )

        print(
            "formal evaluation             = NO"
        )

        print(
            "dedicated tests               = PASS | 14"
        )

        print(
            "full Stage6 regression        = PASS | 112"
        )

        print(
            "module SHA256                 =",
            module_sha,
        )

        print(
            "contract SHA256               =",
            contract_sha,
        )

        print(
            "report SHA256                 =",
            report_sha,
        )

        print(
            "NEXT = BLOCK6.4 PART2 "
            "DEVELOPMENT MC CONVERGENCE + "
            "PROBABILISTIC OCCUPANCY KERNEL"
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
            "BLOCK6.4 PART1 — CONTROLLED BLOCK"
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

from __future__ import annotations

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

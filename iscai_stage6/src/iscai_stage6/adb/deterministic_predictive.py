from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    resolve_samplewise_headings,
)

from .deterministic_association import (
    PredictorAnchor,
    ReciprocalAssociationResult,
    reciprocal_unique_nearest_anchor_association,
)

from .geometry import (
    Box3D,
    ProjectedBox,
    project_box_to_headlamp,
)


HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

PREDICTED_STATE_SOURCE = (
    'predicted_sample'
)

PREDICTED_ORIENTATION_SOURCE = (
    'predicted_tangent'
)


Vec3 = tuple[
    float,
    float,
    float,
]


def _finite_vec3(
    value,
    *,
    name: str,
) -> Vec3:

    result = tuple(
        float(item)
        for item in value
    )

    if len(result) != 3:
        raise ValueError(
            f"{name} must be 3D."
        )

    if not all(
        math.isfinite(item)
        for item in result
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


@dataclass(frozen=True)
class DeterministicSharedMeanPrediction:
    """
    Deterministic Stage6 ablation of the frozen shared
    calibrated Gaussian trajectory posterior.

    mean_displacement_H0_m contains the Gaussian mean only.
    No predictive covariance enters this contract.
    """

    prediction_id: str

    latest_position_H0_m: Vec3

    mean_displacement_H0_m: tuple[
            Vec3,
            Vec3,
            Vec3,
            Vec3,
        ]

    def __post_init__(
        self,
    ) -> None:

        if not str(
            self.prediction_id
        ):
            raise ValueError(
                "prediction_id cannot be empty."
            )

        _finite_vec3(
            self.latest_position_H0_m,
            name=(
                "latest_position_H0_m"
            ),
        )

        if (
            len(
                self.mean_displacement_H0_m
            )
            !=
            len(
                HORIZONS_S
            )
        ):
            raise ValueError(
                "Shared mean must contain exactly "
                "the four frozen prediction horizons."
            )

        for index, value in enumerate(
            self.mean_displacement_H0_m
        ):

            _finite_vec3(
                value,
                name=(
                    "mean_displacement_H0_m"
                    f"[{index}]"
                ),
            )

    def association_anchor(
        self,
    ) -> PredictorAnchor:

        return PredictorAnchor(
            prediction_id=(
                self.prediction_id
            ),

            latest_position_H0_m=(
                self.latest_position_H0_m
            ),
        )

    def future_centers_H0_m(
        self,
    ) -> tuple[
        Vec3,
        Vec3,
        Vec3,
        Vec3,
    ]:
        """
        Frozen coordinate semantics:

          future_H0[h]
            =
          predictor_latest_causal_anchor_H0
            +
          Gaussian_mean_metric_H0_displacement[h]

        The matched WOMD/Stage6 box center is deliberately
        NOT used to recenter this trajectory.
        """

        anchor = np.asarray(
            self.latest_position_H0_m,
            dtype=np.float64,
        )

        displacement = np.asarray(
            self.mean_displacement_H0_m,
            dtype=np.float64,
        )

        result = (
            anchor[
                None,
                :
            ]
            +
            displacement
        )

        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in result
        )


@dataclass(frozen=True)
class DeterministicFutureProjectedBox:
    horizon_s: float

    center_H0_m: Vec3

    yaw_H0_rad: float

    heading_fallback_used: bool

    box: Box3D

    projection: ProjectedBox


@dataclass(frozen=True)
class MatchedDeterministicActorForecast:
    predictor_index: int
    actor_box_index: int

    prediction_id: str
    actor_object_type: str

    association_anchor_distance_m: float

    future: tuple[
            DeterministicFutureProjectedBox,
            DeterministicFutureProjectedBox,
            DeterministicFutureProjectedBox,
            DeterministicFutureProjectedBox,
        ]


@dataclass(frozen=True)
class DeterministicPredictivePlan:
    association: ReciprocalAssociationResult

    matched_forecasts: tuple[
            MatchedDeterministicActorForecast,
            ...,
        ]

    reactive_fallback_box_indices: tuple[
            int,
            ...,
        ]

    unmatched_prediction_indices: tuple[
            int,
            ...,
        ]

    @property
    def predicted_box_count(
        self,
    ) -> int:

        return sum(
            len(
                item.future
            )
            for item
            in self.matched_forecasts
        )


def _validate_causal_actor_box(
    actor_box,
) -> None:

    for name in (
        "tracks_to_predict_used",
        "objects_of_interest_used",
        "future_state_used",
    ):

        if bool(
            getattr(
                actor_box,
                name,
                False,
            )
        ):

            raise RuntimeError(
                "Deterministic predictive controller "
                "received non-causal actor-box provenance: "
                f"{name}=True"
            )

    if getattr(
        actor_box,
        "stage1_anchor_center_H0_m",
        None,
    ) is None:

        raise RuntimeError(
            "Matched actor box has no causal Stage1 anchor."
        )

    box = getattr(
        actor_box,
        "box",
        None,
    )

    if box is None:
        raise RuntimeError(
            "Matched actor box has no Box3D geometry."
        )

    dimensions = (
        float(
            box.length_m
        ),
        float(
            box.width_m
        ),
        float(
            box.height_m
        ),
    )

    if not all(
        math.isfinite(value)
        and
        value > 0.0
        for value in dimensions
    ):
        raise ValueError(
            "Matched causal box has invalid dimensions."
        )

    if not math.isfinite(
        float(
            box.yaw_rad
        )
    ):
        raise ValueError(
            "Matched causal box has invalid current yaw."
        )


def build_deterministic_future_full_boxes(
    prediction:
        DeterministicSharedMeanPrediction,
    actor_box,
) -> tuple[
    DeterministicFutureProjectedBox,
    DeterministicFutureProjectedBox,
    DeterministicFutureProjectedBox,
    DeterministicFutureProjectedBox,
]:
    """
    Build the deterministic predictive ADB geometry for one
    associated actor.

    State:
      shared calibrated-Gaussian mean trajectory only.

    Center:
      predictor causal anchor + metric-H0 mean displacement.

    Dimensions:
      current causal l/w/h held explicit across horizons.

    Orientation:
      frozen Stage5 trajectory tangent rule, with current causal
      actor-box yaw used only as the low-speed carry-forward seed.

    Projection:
      full 3D Box3D -> all eight physical corners.

    Deliberately absent:
      predictive covariance,
      probabilistic occupancy,
      future GT,
      class-specific margin/floor,
      communication codebook.
    """

    _validate_causal_actor_box(
        actor_box
    )

    centers = np.asarray(
        prediction.future_centers_H0_m(),
        dtype=np.float64,
    )

    heading_samples = centers[
        None,
        :,
        :,
    ]

    headings, fallback = (
        resolve_samplewise_headings(
            trajectory_samples_h0_m=(
                heading_samples
            ),

            current_position_h0_m=(
                prediction.latest_position_H0_m
            ),

            current_heading_h0_rad=float(
                actor_box.box.yaw_rad
            ),
        )
    )

    if headings.shape != (
        1,
        4,
    ):
        raise RuntimeError(
            "Unexpected frozen Stage5 heading shape."
        )

    if fallback.shape != (
        1,
        4,
    ):
        raise RuntimeError(
            "Unexpected frozen Stage5 fallback shape."
        )

    result = []

    for horizon_index, horizon_s in enumerate(
        HORIZONS_S
    ):

        center = tuple(
            float(value)
            for value
            in centers[
                horizon_index
            ]
        )

        yaw = float(
            headings[
                0,
                horizon_index,
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

            yaw_rad=yaw,
        )

        projection = (
            project_box_to_headlamp(
                box,

                state_source=(
                    PREDICTED_STATE_SOURCE
                ),

                orientation_source=(
                    PREDICTED_ORIENTATION_SOURCE
                ),

                controller_path=True,
            )
        )

        if projection.corners.xyz.shape != (
            8,
            3,
        ):
            raise RuntimeError(
                "Future predictive box did not "
                "produce exactly eight physical corners."
            )

        result.append(
            DeterministicFutureProjectedBox(
                horizon_s=float(
                    horizon_s
                ),

                center_H0_m=center,

                yaw_H0_rad=yaw,

                heading_fallback_used=bool(
                    fallback[
                        0,
                        horizon_index,
                    ]
                ),

                box=box,

                projection=projection,
            )
        )

    return tuple(
        result
    )


def build_deterministic_predictive_plan(
    predictions:
        Iterable[
            DeterministicSharedMeanPrediction
        ],
    actor_boxes:
        Iterable,
) -> DeterministicPredictivePlan:
    """
    Associate causal predictor tracks to causal Stage6 boxes,
    then construct deterministic future full boxes.

    Unmatched causal ADB boxes are explicitly returned for the
    frozen original-reactive fallback. They are never silently
    removed from illumination control.
    """

    predictions = tuple(
        predictions
    )

    actor_boxes = tuple(
        actor_boxes
    )

    anchors = tuple(
        prediction.association_anchor()
        for prediction
        in predictions
    )

    association = (
        reciprocal_unique_nearest_anchor_association(
            anchors,
            actor_boxes,
        )
    )

    forecasts = []

    for match in association.matches:

        actor_box = actor_boxes[
            match.box_index
        ]

        prediction = predictions[
            match.predictor_index
        ]

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                actor_box,
            )
        )

        forecasts.append(
            MatchedDeterministicActorForecast(
                predictor_index=(
                    match.predictor_index
                ),

                actor_box_index=(
                    match.box_index
                ),

                prediction_id=(
                    prediction.prediction_id
                ),

                actor_object_type=str(
                    getattr(
                        actor_box,
                        "object_type",
                        ""
                    )
                ),

                association_anchor_distance_m=(
                    float(
                        match.anchor_distance_m
                    )
                ),

                future=future,
            )
        )

    return DeterministicPredictivePlan(
        association=(
            association
        ),

        matched_forecasts=tuple(
            forecasts
        ),

        reactive_fallback_box_indices=(
            association.unmatched_box_indices
        ),

        unmatched_prediction_indices=(
            association.unmatched_predictor_indices
        ),
    )

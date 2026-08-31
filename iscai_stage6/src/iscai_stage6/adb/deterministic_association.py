from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class PredictorAnchor:
    """
    Controller-side causal predictor identity.

    prediction_id is internal tracker/predictor metadata only.
    It is NOT compared with a WOMD actor ID during association.
    """

    prediction_id: str
    latest_position_H0_m: Vec3

    def __post_init__(
        self,
    ) -> None:

        if not str(
            self.prediction_id
        ):
            raise ValueError(
                "prediction_id cannot be empty."
            )

        values = tuple(
            float(value)
            for value
            in self.latest_position_H0_m
        )

        if len(values) != 3:
            raise ValueError(
                "latest_position_H0_m must be 3D."
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "latest_position_H0_m must be finite."
            )


@dataclass(frozen=True)
class ReciprocalAssociationMatch:
    predictor_index: int
    box_index: int

    prediction_id: str

    anchor_distance_m: float

    predictor_second_nearest_distance_m: float | None

    box_second_nearest_distance_m: float | None


@dataclass(frozen=True)
class ReciprocalAssociationResult:
    matches: tuple[
            ReciprocalAssociationMatch,
            ...,
        ]

    unmatched_predictor_indices: tuple[
            int,
            ...,
        ]

    unmatched_box_indices: tuple[
            int,
            ...,
        ]

    predictor_count: int
    box_count: int

    @property
    def matched_pair_count(
        self,
    ) -> int:

        return len(
            self.matches
        )

    @property
    def match_fraction_of_boxes(
        self,
    ) -> float:

        if self.box_count == 0:
            return 1.0

        return (
            self.matched_pair_count
            /
            self.box_count
        )

    @property
    def reactive_fallback_fraction(
        self,
    ) -> float:

        if self.box_count == 0:
            return 0.0

        return (
            len(
                self.unmatched_box_indices
            )
            /
            self.box_count
        )


def _finite_vec3(
    value,
    *,
    name: str,
) -> np.ndarray:

    result = np.asarray(
        value,
        dtype=np.float64,
    )

    if result.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _box_anchor(
    actor_box,
) -> np.ndarray | None:

    value = getattr(
        actor_box,
        "stage1_anchor_center_H0_m",
        None,
    )

    if value is None:
        return None

    return _finite_vec3(
        value,
        name=(
            "stage1_anchor_center_H0_m"
        ),
    )


def _unique_nearest(
    squared_distances: np.ndarray,
) -> tuple[
    int | None,
    float,
    float | None,
]:
    """
    Parameter-free unique nearest relation.

    Exact equal minima are considered ambiguous and therefore
    unmatched. No tolerance, hard distance gate, covariance
    gate or class gate is introduced.
    """

    values = np.asarray(
        squared_distances,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Nearest-neighbour vector must be 1D."
        )

    if values.size == 0:
        return (
            None,
            float("inf"),
            None,
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "Association distances must be finite."
        )

    minimum = float(
        np.min(
            values
        )
    )

    nearest = np.flatnonzero(
        values
        ==
        minimum
    )

    ordered = np.sort(
        values
    )

    second = (
        None
        if ordered.size < 2
        else math.sqrt(
            float(
                ordered[
                    1
                ]
            )
        )
    )

    if nearest.size != 1:

        return (
            None,
            math.sqrt(
                minimum
            ),
            second,
        )

    return (
        int(
            nearest[
                0
            ]
        ),
        math.sqrt(
            minimum
        ),
        second,
    )


def reciprocal_unique_nearest_anchor_association(
    predictor_anchors:
        Iterable[
            PredictorAnchor
        ],
    actor_boxes:
        Iterable,
) -> ReciprocalAssociationResult:
    """
    Frozen Block-6.3 association policy.

    Association basis:
      predictor latest causal H0 anchor
        <->
      Stage6 causal ADB-box Stage1 H0 anchor

    Pair acceptance:
      - predictor has one exact unique nearest box,
      - box has one exact unique nearest predictor,
      - relations are reciprocal.

    Deliberately absent:
      - hard distance threshold,
      - covariance gate,
      - class gate,
      - Hungarian/forced assignment,
      - prediction_id == WOMD track_id,
      - truth_id / truth_track_index,
      - tracks_to_predict,
      - future state information.

    A causal box with no finite Stage1 anchor remains unmatched
    and is therefore eligible for the frozen original-reactive
    fallback downstream.
    """

    predictors = tuple(
        predictor_anchors
    )

    boxes = tuple(
        actor_boxes
    )

    prediction_ids = [
        str(
            item.prediction_id
        )
        for item in predictors
    ]

    if (
        len(
            prediction_ids
        )
        !=
        len(
            set(
                prediction_ids
            )
        )
    ):
        raise ValueError(
            "Predictor IDs must be unique "
            "within one association call."
        )

    predictor_positions = tuple(
        _finite_vec3(
            item.latest_position_H0_m,
            name=(
                "predictor latest_position_H0_m"
            ),
        )
        for item in predictors
    )

    box_positions = tuple(
        _box_anchor(
            actor_box
        )
        for actor_box in boxes
    )

    valid_box_indices = tuple(
        index
        for index, position
        in enumerate(
            box_positions
        )
        if position is not None
    )

    if (
        not predictors
        or
        not valid_box_indices
    ):

        return ReciprocalAssociationResult(
            matches=(),

            unmatched_predictor_indices=(
                tuple(
                    range(
                        len(
                            predictors
                        )
                    )
                )
            ),

            unmatched_box_indices=(
                tuple(
                    range(
                        len(
                            boxes
                        )
                    )
                )
            ),

            predictor_count=len(
                predictors
            ),

            box_count=len(
                boxes
            ),
        )

    distance_squared = np.empty(
        (
            len(
                predictors
            ),
            len(
                valid_box_indices
            ),
        ),
        dtype=np.float64,
    )

    for predictor_index, predictor_position in enumerate(
        predictor_positions
    ):

        for local_box_index, box_index in enumerate(
            valid_box_indices
        ):

            difference = (
                predictor_position
                -
                box_positions[
                    box_index
                ]
            )

            distance_squared[
                predictor_index,
                local_box_index,
            ] = float(
                difference
                @
                difference
            )

    predictor_nearest = []
    predictor_distance = []
    predictor_second = []

    for predictor_index in range(
        len(
            predictors
        )
    ):

        (
            local_box,
            distance,
            second,
        ) = _unique_nearest(
            distance_squared[
                predictor_index,
                :,
            ]
        )

        predictor_nearest.append(
            local_box
        )

        predictor_distance.append(
            distance
        )

        predictor_second.append(
            second
        )

    box_nearest = []
    box_second = []

    for local_box_index in range(
        len(
            valid_box_indices
        )
    ):

        (
            predictor_index,
            _distance,
            second,
        ) = _unique_nearest(
            distance_squared[
                :,
                local_box_index,
            ]
        )

        box_nearest.append(
            predictor_index
        )

        box_second.append(
            second
        )

    matches = []

    matched_predictors = set()
    matched_boxes = set()

    for predictor_index, local_box_index in enumerate(
        predictor_nearest
    ):

        if local_box_index is None:
            continue

        if (
            box_nearest[
                local_box_index
            ]
            !=
            predictor_index
        ):
            continue

        box_index = (
            valid_box_indices[
                local_box_index
            ]
        )

        matches.append(
            ReciprocalAssociationMatch(
                predictor_index=(
                    predictor_index
                ),

                box_index=(
                    box_index
                ),

                prediction_id=(
                    predictors[
                        predictor_index
                    ].prediction_id
                ),

                anchor_distance_m=float(
                    predictor_distance[
                        predictor_index
                    ]
                ),

                predictor_second_nearest_distance_m=(
                    predictor_second[
                        predictor_index
                    ]
                ),

                box_second_nearest_distance_m=(
                    box_second[
                        local_box_index
                    ]
                ),
            )
        )

        matched_predictors.add(
            predictor_index
        )

        matched_boxes.add(
            box_index
        )

    return ReciprocalAssociationResult(
        matches=tuple(
            matches
        ),

        unmatched_predictor_indices=tuple(
            index
            for index in range(
                len(
                    predictors
                )
            )
            if index not in matched_predictors
        ),

        unmatched_box_indices=tuple(
            index
            for index in range(
                len(
                    boxes
                )
            )
            if index not in matched_boxes
        ),

        predictor_count=len(
            predictors
        ),

        box_count=len(
            boxes
        ),
    )

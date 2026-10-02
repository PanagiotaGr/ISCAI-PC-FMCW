from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import (
    hypot,
    isfinite,
)

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from .neural_inputs import (
    MAP_CONTEXT_DIM,
    CausalSceneInputs,
    ModelInputPayload,
    build_model_input_payload,
)


HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

HORIZON_FRAME_OFFSETS = (
    1,
    3,
    5,
    10,
)

SUPERVISION_ASSOCIATION_GATE_M = 5.0

MAP_CATEGORIES = (
    "lane",
    "road_line",
    "road_edge",
    "crosswalk",
    "stop_sign",
)

MAP_RADIUS_M = 50.0
MAP_EMPTY_DISTANCE_M = 1000.0


Vec3 = tuple[
    float,
    float,
    float,
]


def _actor_class_name(
    object_type: int,
) -> str:
    try:
        return (
            scenario_pb2.Track
            .ObjectType
            .Name(
                int(object_type)
            )
        )
    except Exception:
        return (
            f"TYPE_UNKNOWN_"
            f"{int(object_type)}"
        )


@dataclass(frozen=True)
class FutureTrajectoryLabel:
    positions_H0_m: tuple[
        Vec3,
        ...,
    ]

    valid_mask: tuple[
        bool,
        ...,
    ]

    def __post_init__(self) -> None:
        if len(
            self.positions_H0_m
        ) != len(
            HORIZONS_S
        ):
            raise ValueError(
                "Future-label horizon "
                "count changed."
            )

        if len(
            self.valid_mask
        ) != len(
            HORIZONS_S
        ):
            raise ValueError(
                "Future-validity mask "
                "count changed."
            )

        for position in (
            self.positions_H0_m
        ):
            if len(position) != 3:
                raise ValueError(
                    "Future position "
                    "must be 3-D."
                )

            if not all(
                isfinite(
                    float(x)
                )
                for x in position
            ):
                raise ValueError(
                    "Future label must "
                    "be finite."
                )

    def sha256(
        self,
    ) -> str:
        return sha256(
            json.dumps(
                {
                    "positions":
                        self.positions_H0_m,
                    "mask":
                        self.valid_mask,
                },
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class SupervisedTrajectorySample:
    sample_id: str

    prediction_id: str

    model_input: ModelInputPayload

    actor_class: str
    truth_track_index: int

    historical_match_distance_m: float
    historical_match_frame_index: int

    future_label: FutureTrajectoryLabel
    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError(
                "sample_id cannot be empty."
            )

        if not self.prediction_id:
            raise ValueError(
                "prediction_id cannot "
                "be empty."
            )

        if self.truth_track_index < 0:
            raise ValueError(
                "truth_track_index cannot "
                "be negative."
            )

        if (
            self.historical_match_distance_m
            >
            SUPERVISION_ASSOCIATION_GATE_M
            +
            1e-9
        ):
            raise ValueError(
                "Supervision pairing "
                "outside causal gate."
            )

    def model_payload_sha256(
        self,
    ) -> str:
        return (
            self.model_input
            .sha256()
        )


def _point_H0(
    state,
    *,
    T_H0_from_W,
) -> Vec3:
    point = (
        T_H0_from_W
        .apply_point(
            (
                float(
                    state.center_x
                ),
                float(
                    state.center_y
                ),
                float(
                    state.center_z
                ),
            )
        )
    )

    return tuple(
        float(x)
        for x in point
    )


def _map_message_points(
    message,
):
    for name in (
        "polyline",
        "polygon",
    ):
        if hasattr(
            message,
            name,
        ):
            values = getattr(
                message,
                name,
            )

            for point in values:
                if all(
                    hasattr(
                        point,
                        axis,
                    )
                    for axis in (
                        "x",
                        "y",
                        "z",
                    )
                ):
                    yield (
                        float(point.x),
                        float(point.y),
                        float(point.z),
                    )

    if hasattr(
        message,
        "position",
    ):
        point = getattr(
            message,
            "position",
        )

        if all(
            hasattr(
                point,
                axis,
            )
            for axis in (
                "x",
                "y",
                "z",
            )
        ):
            yield (
                float(point.x),
                float(point.y),
                float(point.z),
            )


def build_static_map_index_H0(
    scenario,
    *,
    T_H0_from_W,
):
    result = {
        category: []
        for category in (
            MAP_CATEGORIES
        )
    }

    for feature in (
        scenario.map_features
    ):
        try:
            kind = (
                feature.WhichOneof(
                    "feature_data"
                )
            )
        except Exception:
            kind = None

        if kind not in result:
            continue

        message = getattr(
            feature,
            kind,
        )

        for point_W in (
            _map_message_points(
                message
            )
        ):
            point_H0 = (
                T_H0_from_W
                .apply_point(
                    point_W
                )
            )

            result[kind].append(
                tuple(
                    float(x)
                    for x in point_H0
                )
            )

    return {
        key:
            tuple(value)
        for key, value
        in result.items()
    }


def map_context_summary(
    map_index_H0,
    *,
    target_position_H0_m: Vec3,
) -> tuple[float, ...]:
    tx, ty, _ = (
        target_position_H0_m
    )

    output = []

    for category in (
        MAP_CATEGORIES
    ):
        points = (
            map_index_H0
            .get(
                category,
                (),
            )
        )

        distances = [
            hypot(
                point[0] - tx,
                point[1] - ty,
            )
            for point in points
        ]

        if distances:
            nearest = min(
                distances
            )

            count = sum(
                distance
                <=
                MAP_RADIUS_M
                for distance
                in distances
            )

        else:
            nearest = (
                MAP_EMPTY_DISTANCE_M
            )

            count = 0

        output.extend(
            (
                float(nearest),
                float(count),
            )
        )

    result = tuple(
        output
    )

    if len(result) != (
        MAP_CONTEXT_DIM
    ):
        raise RuntimeError(
            "Map-context dimension changed."
        )

    return result


def _current_valid_truth_indices(
    scenario,
):
    anchor = int(
        scenario.current_time_index
    )

    sdc = int(
        scenario.sdc_track_index
    )

    return tuple(
        index
        for index, track
        in enumerate(
            scenario.tracks
        )
        if (
            index != sdc
            and
            anchor
            <
            len(track.states)
            and
            bool(
                track.states[
                    anchor
                ].valid
            )
        )
    )


def _causal_supervision_matches(
    scene_inputs:
        CausalSceneInputs,
    scenario,
    *,
    T_H0_from_W,
):
    """
    Deterministic one-to-one supervision
    matching using only historical/current
    states.

    Future states are never inspected here.
    """

    truth_indices = (
        _current_valid_truth_indices(
            scenario
        )
    )

    edges = []

    for history in (
        scene_inputs.histories
    ):
        frame_index = (
            history
            .latest_observed_frame_index
        )

        hx, hy, _ = (
            history
            .latest_position_H0_m
        )

        for truth_index in (
            truth_indices
        ):
            track = (
                scenario.tracks[
                    truth_index
                ]
            )

            if frame_index >= len(
                track.states
            ):
                continue

            state = (
                track.states[
                    frame_index
                ]
            )

            if not bool(
                state.valid
            ):
                continue

            truth_position = (
                _point_H0(
                    state,
                    T_H0_from_W=(
                        T_H0_from_W
                    ),
                )
            )

            distance = hypot(
                truth_position[0]
                -
                hx,

                truth_position[1]
                -
                hy,
            )

            if (
                distance
                <=
                SUPERVISION_ASSOCIATION_GATE_M
            ):
                edges.append(
                    (
                        float(distance),
                        history.prediction_id,
                        int(truth_index),
                        int(frame_index),
                    )
                )

    edges.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    used_predictions = set()
    used_truth = set()

    matches = []

    for (
        distance,
        prediction_id,
        truth_index,
        frame_index,
    ) in edges:
        if (
            prediction_id
            in
            used_predictions
        ):
            continue

        if truth_index in used_truth:
            continue

        used_predictions.add(
            prediction_id
        )

        used_truth.add(
            truth_index
        )

        matches.append(
            (
                prediction_id,
                truth_index,
                distance,
                frame_index,
            )
        )

    return tuple(
        matches
    )


def _build_future_label(
    scenario,
    *,
    truth_track_index: int,
    T_H0_from_W,
) -> FutureTrajectoryLabel:
    anchor = int(
        scenario.current_time_index
    )

    track = (
        scenario.tracks[
            truth_track_index
        ]
    )

    positions = []
    masks = []

    for offset in (
        HORIZON_FRAME_OFFSETS
    ):
        index = (
            anchor
            +
            offset
        )

        if (
            index
            <
            len(track.states)
            and
            bool(
                track.states[
                    index
                ].valid
            )
        ):
            position = (
                _point_H0(
                    track.states[
                        index
                    ],
                    T_H0_from_W=(
                        T_H0_from_W
                    ),
                )
            )

            mask = True

        else:
            position = (
                0.0,
                0.0,
                0.0,
            )

            mask = False

        positions.append(
            position
        )

        masks.append(
            mask
        )

    return FutureTrajectoryLabel(
        positions_H0_m=tuple(
            positions
        ),
        valid_mask=tuple(
            masks
        ),
    )


def attach_supervision(
    scene_inputs:
        CausalSceneInputs,
    scenario,
    *,
    T_H0_from_W,
) -> tuple[
    SupervisedTrajectorySample,
    ...,
]:
    """
    Boundary:

      causal scene input already exists
                 ↓
      historical/current-only pairing
                 ↓
      map context
                 ↓
      future label extraction

    Future validity affects only the label mask.
    It never determines whether a matched
    causal sample exists.
    """

    matches = (
        _causal_supervision_matches(
            scene_inputs,
            scenario,
            T_H0_from_W=(
                T_H0_from_W
            ),
        )
    )

    map_index = (
        build_static_map_index_H0(
            scenario,
            T_H0_from_W=(
                T_H0_from_W
            ),
        )
    )

    samples = []

    for (
        prediction_id,
        truth_track_index,
        distance,
        frame_index,
    ) in matches:
        history = (
            scene_inputs
            .history_by_id(
                prediction_id
            )
        )

        map_context = (
            map_context_summary(
                map_index,
                target_position_H0_m=(
                    history
                    .latest_position_H0_m
                ),
            )
        )

        model_input = (
            build_model_input_payload(
                scene_inputs,
                prediction_id=(
                    prediction_id
                ),
                map_context=(
                    map_context
                ),
            )
        )

        track = (
            scenario.tracks[
                truth_track_index
            ]
        )

        future_label = (
            _build_future_label(
                scenario,
                truth_track_index=(
                    truth_track_index
                ),
                T_H0_from_W=(
                    T_H0_from_W
                ),
            )
        )

        samples.append(
            SupervisedTrajectorySample(
                sample_id=(
                    f"{scenario.scenario_id}:"
                    f"{prediction_id}"
                ),
                prediction_id=(
                    prediction_id
                ),
                model_input=(
                    model_input
                ),
                actor_class=(
                    _actor_class_name(
                        track.object_type
                    )
                ),
                truth_track_index=(
                    truth_track_index
                ),
                historical_match_distance_m=(
                    distance
                ),
                historical_match_frame_index=(
                    frame_index
                ),
                future_label=(
                    future_label
                ),
            )
        )

    samples.sort(
        key=lambda sample:
            sample.prediction_id
    )

    return tuple(
        samples
    )

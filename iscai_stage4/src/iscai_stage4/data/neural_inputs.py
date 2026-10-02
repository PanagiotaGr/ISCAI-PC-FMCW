from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import (
    atan2,
    hypot,
    isfinite,
    sqrt,
)

from iscai_stage3.geometry import (
    cartesianize_associated_track,
)


HISTORY_STEPS = 11
MAX_NEIGHBORS = 8

FEATURE_NAMES = (
    "x_H0_m",
    "y_H0_m",
    "z_H0_m",
    "vx_H0_mps",
    "vy_H0_mps",
    "vz_H0_mps",
    "meas_cov_xx_H0_m2",
    "meas_cov_xy_H0_m2",
    "meas_cov_xz_H0_m2",
    "meas_cov_yy_H0_m2",
    "meas_cov_yz_H0_m2",
    "meas_cov_zz_H0_m2",
    "observed_mask",
    "velocity_valid_mask",
)

MAP_CONTEXT_FEATURE_NAMES = (
    "nearest_lane_m",
    "lane_points_within_50m",
    "nearest_road_line_m",
    "road_line_points_within_50m",
    "nearest_road_edge_m",
    "road_edge_points_within_50m",
    "nearest_crosswalk_m",
    "crosswalk_points_within_50m",
    "nearest_stop_sign_m",
    "stop_sign_points_within_50m",
)

FEATURE_DIM = len(
    FEATURE_NAMES
)

MAP_CONTEXT_DIM = len(
    MAP_CONTEXT_FEATURE_NAMES
)


Vec3 = tuple[
    float,
    float,
    float,
]

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _finite3(
    value,
) -> Vec3:
    if len(value) != 3:
        raise ValueError(
            "Expected a 3-vector."
        )

    result = tuple(
        float(x)
        for x in value
    )

    if not all(
        isfinite(x)
        for x in result
    ):
        raise ValueError(
            "3-vector must be finite."
        )

    return result


def _matrix3(
    value,
) -> Matrix3:
    if len(value) != 3:
        raise ValueError(
            "Expected 3x3 covariance."
        )

    rows = tuple(
        tuple(
            float(x)
            for x in row
        )
        for row in value
    )

    if any(
        len(row) != 3
        for row in rows
    ):
        raise ValueError(
            "Expected 3x3 covariance."
        )

    if not all(
        isfinite(x)
        for row in rows
        for x in row
    ):
        raise ValueError(
            "Covariance must be finite."
        )

    for i in range(3):
        for j in range(3):
            if abs(
                rows[i][j]
                -
                rows[j][i]
            ) > 1e-8:
                raise ValueError(
                    "Measurement covariance "
                    "must be symmetric."
                )

    if any(
        rows[i][i] < 0.0
        for i in range(3)
    ):
        raise ValueError(
            "Measurement covariance "
            "diagonal cannot be negative."
        )

    return rows


def _zero_covariance() -> Matrix3:
    return (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


@dataclass(frozen=True)
class HistoryStep:
    frame_index: int
    timestamp_s: float

    position_H0_m: Vec3
    velocity_H0_mps: Vec3

    measurement_covariance_H0_m2: Matrix3
    observed: bool
    velocity_valid: bool

    def __post_init__(self) -> None:
        if not (
            0
            <=
            int(self.frame_index)
            <
            HISTORY_STEPS
        ):
            raise ValueError(
                "frame_index outside "
                "causal history."
            )

        if not isfinite(
            float(self.timestamp_s)
        ):
            raise ValueError(
                "timestamp must be finite."
            )

        _finite3(
            self.position_H0_m
        )

        _finite3(
            self.velocity_H0_mps
        )

        _matrix3(
            self.measurement_covariance_H0_m2
        )

        if (
            self.velocity_valid
            and
            not self.observed
        ):
            raise ValueError(
                "Missing observation cannot "
                "have a valid velocity."
            )

    def feature_row(
        self,
    ) -> tuple[float, ...]:
        covariance = (
            self
            .measurement_covariance_H0_m2
        )

        row = (
            float(
                self.position_H0_m[0]
            ),
            float(
                self.position_H0_m[1]
            ),
            float(
                self.position_H0_m[2]
            ),

            float(
                self.velocity_H0_mps[0]
            ),
            float(
                self.velocity_H0_mps[1]
            ),
            float(
                self.velocity_H0_mps[2]
            ),

            float(
                covariance[0][0]
            ),
            float(
                covariance[0][1]
            ),
            float(
                covariance[0][2]
            ),
            float(
                covariance[1][1]
            ),
            float(
                covariance[1][2]
            ),
            float(
                covariance[2][2]
            ),

            1.0
            if self.observed
            else 0.0,

            1.0
            if self.velocity_valid
            else 0.0,
        )

        if len(row) != FEATURE_DIM:
            raise RuntimeError(
                "Feature dimension changed."
            )

        return row


@dataclass(frozen=True)
class CausalTrackHistory:
    prediction_id: str

    steps: tuple[
        HistoryStep,
        ...,
    ]

    latest_observed_frame_index: int
    latest_position_H0_m: Vec3

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError(
                "prediction_id cannot "
                "be empty."
            )

        if len(
            self.steps
        ) != HISTORY_STEPS:
            raise ValueError(
                "Stage4 history must contain "
                "exactly 11 causal frames."
            )

        expected_indices = tuple(
            range(
                HISTORY_STEPS
            )
        )

        actual_indices = tuple(
            int(step.frame_index)
            for step in self.steps
        )

        if (
            actual_indices
            !=
            expected_indices
        ):
            raise ValueError(
                "History frame grid changed."
            )

        observed_indices = [
            step.frame_index
            for step in self.steps
            if step.observed
        ]

        if not observed_indices:
            raise ValueError(
                "Causal history contains "
                "no observations."
            )

        if (
            max(observed_indices)
            !=
            self.latest_observed_frame_index
        ):
            raise ValueError(
                "latest observed frame mismatch."
            )

        _finite3(
            self.latest_position_H0_m
        )

    @property
    def observed_count(
        self,
    ) -> int:
        return sum(
            step.observed
            for step in self.steps
        )

    def feature_rows(
        self,
    ) -> tuple[
        tuple[float, ...],
        ...,
    ]:
        return tuple(
            step.feature_row()
            for step in self.steps
        )


@dataclass(frozen=True)
class ModelInputPayload:
    target_history: tuple[
        tuple[float, ...],
        ...,
    ]

    neighbor_histories: tuple[
        tuple[
            tuple[float, ...],
            ...,
        ],
        ...,
    ]

    neighbor_mask: tuple[
        float,
        ...,
    ]

    map_context: tuple[
        float,
        ...,
    ]

    def __post_init__(self) -> None:
        if len(
            self.target_history
        ) != HISTORY_STEPS:
            raise ValueError(
                "Target history length "
                "must be 11."
            )

        if any(
            len(row)
            !=
            FEATURE_DIM
            for row in self.target_history
        ):
            raise ValueError(
                "Target feature width changed."
            )

        if len(
            self.neighbor_histories
        ) != MAX_NEIGHBORS:
            raise ValueError(
                "Neighbor slot count changed."
            )

        for history in (
            self.neighbor_histories
        ):
            if len(history) != (
                HISTORY_STEPS
            ):
                raise ValueError(
                    "Neighbor history length "
                    "changed."
                )

            if any(
                len(row)
                !=
                FEATURE_DIM
                for row in history
            ):
                raise ValueError(
                    "Neighbor feature width "
                    "changed."
                )

        if len(
            self.neighbor_mask
        ) != MAX_NEIGHBORS:
            raise ValueError(
                "Neighbor mask length changed."
            )

        if len(
            self.map_context
        ) != MAP_CONTEXT_DIM:
            raise ValueError(
                "Map-context width changed."
            )

        if not all(
            isfinite(float(x))
            for x in self.map_context
        ):
            raise ValueError(
                "Map context must be finite."
            )

    def numeric_dict(
        self,
    ) -> dict:
        """
        Only numeric model input.

        No prediction IDs.
        No WOMD track IDs.
        No actor class.
        No truth track index.
        No future labels.
        """

        return {
            "target_history":
                self.target_history,

            "neighbor_histories":
                self.neighbor_histories,

            "neighbor_mask":
                self.neighbor_mask,

            "map_context":
                self.map_context,
        }

    def sha256(
        self,
    ) -> str:
        encoded = json.dumps(
            self.numeric_dict(),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode("utf-8")

        return sha256(
            encoded
        ).hexdigest()


@dataclass(frozen=True)
class CausalSceneInputs:
    scenario_id: str

    timestamps_s: tuple[
        float,
        ...,
    ]

    histories: tuple[
        CausalTrackHistory,
        ...,
    ]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "scenario_id cannot "
                "be empty."
            )

        if len(
            self.timestamps_s
        ) != HISTORY_STEPS:
            raise ValueError(
                "Expected 11 causal timestamps."
            )

        if any(
            b <= a
            for a, b in zip(
                self.timestamps_s[:-1],
                self.timestamps_s[1:],
            )
        ):
            raise ValueError(
                "Causal timestamps must "
                "strictly increase."
            )

        ids = [
            history.prediction_id
            for history in self.histories
        ]

        if (
            len(ids)
            !=
            len(set(ids))
        ):
            raise ValueError(
                "Duplicate associated "
                "prediction IDs."
            )

    def history_by_id(
        self,
        prediction_id: str,
    ) -> CausalTrackHistory:
        for history in self.histories:
            if (
                history.prediction_id
                ==
                prediction_id
            ):
                return history

        raise KeyError(
            prediction_id
        )

    def sha256(
        self,
    ) -> str:
        payload = {
            "scenario_id":
                self.scenario_id,

            "timestamps_s":
                self.timestamps_s,

            "histories": [
                {
                    "prediction_id":
                        history.prediction_id,

                    "features":
                        history.feature_rows(),

                    "latest_frame":
                        history
                        .latest_observed_frame_index,
                }
                for history in self.histories
            ],
        }

        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


def _extract_prediction_id(
    associated_track,
) -> str:
    for name in (
        "track_id",
        "prediction_id",
        "associated_track_id",
    ):
        if hasattr(
            associated_track,
            name,
        ):
            value = getattr(
                associated_track,
                name,
            )

            if value is not None:
                value = str(value)

                if value:
                    return value

    raise RuntimeError(
        "Cannot resolve associated "
        "track identifier."
    )


def _extract_position(
    observation,
) -> Vec3:
    for name in (
        "position_H0_m",
        "position_m",
    ):
        if hasattr(
            observation,
            name,
        ):
            return _finite3(
                getattr(
                    observation,
                    name,
                )
            )

    raise RuntimeError(
        "Stage3 Cartesian observation "
        "does not expose H0 position."
    )


def _extract_covariance(
    observation,
) -> Matrix3:
    for name in (
        "position_covariance_H0_m2",
        "covariance_H0_m2",
        "position_covariance_m2",
        "covariance_m2",
    ):
        if hasattr(
            observation,
            name,
        ):
            value = getattr(
                observation,
                name,
            )

            try:
                return _matrix3(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    raise RuntimeError(
        "Stage3 Cartesian observation "
        "does not expose propagated "
        "measurement position covariance."
    )


def _extract_frame_index(
    observation,
    *,
    timestamps_s: tuple[
        float,
        ...,
    ],
) -> int:
    if hasattr(
        observation,
        "frame_index",
    ):
        index = int(
            getattr(
                observation,
                "frame_index",
            )
        )

        if (
            0
            <=
            index
            <
            len(
                timestamps_s
            )
        ):
            return index

    timestamp = None

    for name in (
        "timestamp_s",
        "time_s",
    ):
        if hasattr(
            observation,
            name,
        ):
            timestamp = float(
                getattr(
                    observation,
                    name,
                )
            )

            break

    if timestamp is None:
        raise RuntimeError(
            "Cartesian observation exposes "
            "neither frame_index nor timestamp."
        )

    differences = [
        abs(
            timestamp
            -
            reference
        )
        for reference
        in timestamps_s
    ]

    index = min(
        range(
            len(
                differences
            )
        ),
        key=lambda i:
            differences[i],
    )

    if differences[index] > 2e-3:
        raise RuntimeError(
            "Observation timestamp does not "
            "align with causal WOMD grid."
        )

    return index


def build_causal_track_history(
    associated_track,
    *,
    context,
    timestamps_s: tuple[
        float,
        ...,
    ],
) -> CausalTrackHistory:
    if len(
        timestamps_s
    ) != HISTORY_STEPS:
        raise ValueError(
            "Stage4 requires 11 "
            "causal timestamps."
        )

    observations = tuple(
        cartesianize_associated_track(
            associated_track,
            context=context,
        )
    )

    if not observations:
        raise ValueError(
            "Associated track has no "
            "Cartesian observations."
        )

    by_frame = {}

    for observation in observations:
        frame_index = (
            _extract_frame_index(
                observation,
                timestamps_s=(
                    timestamps_s
                ),
            )
        )

        by_frame[
            frame_index
        ] = (
            _extract_position(
                observation
            ),
            _extract_covariance(
                observation
            ),
        )

    steps = []

    previous_position = None
    previous_timestamp = None

    for frame_index, timestamp in (
        enumerate(
            timestamps_s
        )
    ):
        if frame_index not in by_frame:
            steps.append(
                HistoryStep(
                    frame_index=frame_index,
                    timestamp_s=(
                        float(timestamp)
                    ),
                    position_H0_m=(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    velocity_H0_mps=(
                        0.0,
                        0.0,
                        0.0,
                    ),
                    measurement_covariance_H0_m2=(
                        _zero_covariance()
                    ),
                    observed=False,
                    velocity_valid=False,
                )
            )

            continue

        (
            position,
            covariance,
        ) = by_frame[
            frame_index
        ]

        velocity = (
            0.0,
            0.0,
            0.0,
        )

        velocity_valid = False

        if (
            previous_position
            is not None
            and
            previous_timestamp
            is not None
        ):
            dt = (
                float(timestamp)
                -
                previous_timestamp
            )

            if dt <= 0.0:
                raise RuntimeError(
                    "Non-positive causal "
                    "velocity interval."
                )

            velocity = tuple(
                (
                    position[axis]
                    -
                    previous_position[axis]
                )
                /
                dt
                for axis in range(3)
            )

            velocity_valid = True

        steps.append(
            HistoryStep(
                frame_index=frame_index,
                timestamp_s=(
                    float(timestamp)
                ),
                position_H0_m=(
                    position
                ),
                velocity_H0_mps=(
                    velocity
                ),
                measurement_covariance_H0_m2=(
                    covariance
                ),
                observed=True,
                velocity_valid=(
                    velocity_valid
                ),
            )
        )

        previous_position = position
        previous_timestamp = float(
            timestamp
        )

    observed = [
        step
        for step in steps
        if step.observed
    ]

    latest = observed[-1]

    return CausalTrackHistory(
        prediction_id=(
            _extract_prediction_id(
                associated_track
            )
        ),
        steps=tuple(
            steps
        ),
        latest_observed_frame_index=(
            latest.frame_index
        ),
        latest_position_H0_m=(
            latest.position_H0_m
        ),
    )


def build_causal_scene_inputs(
    associated_tracks,
    *,
    scenario_id: str,
    context,
    timestamps_s: tuple[
        float,
        ...,
    ],
) -> CausalSceneInputs:
    histories = []

    for associated_track in (
        associated_tracks
    ):
        try:
            history = (
                build_causal_track_history(
                    associated_track,
                    context=context,
                    timestamps_s=(
                        timestamps_s
                    ),
                )
            )

        except ValueError:
            continue

        histories.append(
            history
        )

    histories.sort(
        key=lambda history:
            history.prediction_id
    )

    return CausalSceneInputs(
        scenario_id=scenario_id,
        timestamps_s=(
            timestamps_s
        ),
        histories=tuple(
            histories
        ),
    )


def zero_history_rows():
    return tuple(
        tuple(
            0.0
            for _ in range(
                FEATURE_DIM
            )
        )
        for _ in range(
            HISTORY_STEPS
        )
    )


def build_model_input_payload(
    scene_inputs: CausalSceneInputs,
    *,
    prediction_id: str,
    map_context: tuple[
        float,
        ...,
    ],
) -> ModelInputPayload:
    target = (
        scene_inputs
        .history_by_id(
            prediction_id
        )
    )

    tx, ty, _ = (
        target
        .latest_position_H0_m
    )

    candidates = []

    for history in (
        scene_inputs.histories
    ):
        if (
            history.prediction_id
            ==
            prediction_id
        ):
            continue

        x, y, _ = (
            history
            .latest_position_H0_m
        )

        distance = hypot(
            x - tx,
            y - ty,
        )

        candidates.append(
            (
                distance,
                history.prediction_id,
                history,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    selected = [
        item[2]
        for item in candidates[
            :MAX_NEIGHBORS
        ]
    ]

    neighbor_histories = [
        history.feature_rows()
        for history in selected
    ]

    neighbor_mask = [
        1.0
        for _ in selected
    ]

    while len(
        neighbor_histories
    ) < MAX_NEIGHBORS:
        neighbor_histories.append(
            zero_history_rows()
        )

        neighbor_mask.append(
            0.0
        )

    return ModelInputPayload(
        target_history=(
            target.feature_rows()
        ),
        neighbor_histories=tuple(
            neighbor_histories
        ),
        neighbor_mask=tuple(
            neighbor_mask
        ),
        map_context=tuple(
            float(x)
            for x in map_context
        ),
    )

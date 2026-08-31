from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.filters.imm import (
    IMMCombinedState,
    IMMConfig,
    IMMModeState,
    combine_imm_states,
    interact_mode_states,
    predict_imm_mode_state,
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


@dataclass(frozen=True)
class IMMPredictionPoint:
    horizon_s: float
    timestamp_s: float

    position_H0_m: Vec3

    position_covariance_H0_m2: Matrix3

    mode_probabilities: tuple[
        float,
        float,
        float,
    ]


@dataclass(frozen=True)
class IMMPrediction:
    track_id: str

    anchor_timestamp_s: float

    points: tuple[
        IMMPredictionPoint,
        ...
    ]

    model: str = "IMM_CV_CA_CTRV"

    truth_used: bool = False
    annotated_velocity_used: bool = False
    annotated_heading_used: bool = False
    future_information_used: bool = False


def _prediction_point(
    combined: IMMCombinedState,
    *,
    anchor_timestamp_s: float,
) -> IMMPredictionPoint:
    P = (
        combined
        .covariance_10x10
    )

    return IMMPredictionPoint(
        horizon_s=(
            combined.timestamp_s
            -
            anchor_timestamp_s
        ),
        timestamp_s=(
            combined.timestamp_s
        ),
        position_H0_m=(
            combined.mean_10[0],
            combined.mean_10[1],
            combined.mean_10[2],
        ),
        position_covariance_H0_m2=(
            (
                P[0][0],
                P[0][1],
                P[0][2],
            ),
            (
                P[1][0],
                P[1][1],
                P[1][2],
            ),
            (
                P[2][0],
                P[2][1],
                P[2][2],
            ),
        ),
        mode_probabilities=(
            combined.mode_probabilities
        ),
    )


def predict_imm(
    *,
    track_id: str,
    states: tuple[
        IMMModeState,
        IMMModeState,
        IMMModeState,
    ],
    mode_probabilities: tuple[
        float,
        float,
        float,
    ],
    horizons_s: tuple[
        float,
        ...
    ],
    config: IMMConfig | None = None,
) -> IMMPrediction:
    if config is None:
        config = IMMConfig()

    if not horizons_s:
        raise ValueError(
            "At least one IMM prediction "
            "horizon is required."
        )

    anchor_timestamps = {
        state.timestamp_s
        for state in states
    }

    if len(anchor_timestamps) != 1:
        raise ValueError(
            "IMM prediction states must "
            "be time-aligned."
        )

    anchor_timestamp_s = (
        states[0].timestamp_s
    )

    step = (
        config.prediction_step_s
    )

    requested_steps = []

    previous = None

    for horizon in horizons_s:
        value = float(horizon)

        if (
            not isfinite(value)
            or
            value <= 0.0
        ):
            raise ValueError(
                "IMM horizons must be finite "
                "and positive."
            )

        if (
            previous is not None
            and value <= previous
        ):
            raise ValueError(
                "IMM horizons must be "
                "strictly increasing."
            )

        ratio = (
            value
            /
            step
        )

        integer_step = int(
            round(
                ratio
            )
        )

        if abs(
            ratio
            -
            integer_step
        ) > 1e-9:
            raise ValueError(
                "IMM horizons must be integer "
                "multiples of prediction_step_s."
            )

        requested_steps.append(
            integer_step
        )

        previous = value

    current_states = states
    current_probabilities = (
        mode_probabilities
    )

    requested = set(
        requested_steps
    )

    result_by_step = {}

    maximum_step = max(
        requested_steps
    )

    for index in range(
        1,
        maximum_step + 1,
    ):
        (
            mixed_states,
            predicted_probabilities,
            _mixing,
        ) = interact_mode_states(
            current_states,
            current_probabilities,
            config=config,
        )

        target_timestamp = (
            anchor_timestamp_s
            +
            index
            *
            step
        )

        predicted_states = tuple(
            predict_imm_mode_state(
                state,
                target_timestamp_s=(
                    target_timestamp
                ),
                config=config,
            )
            for state in mixed_states
        )

        current_states = (
            predicted_states
        )

        current_probabilities = (
            predicted_probabilities
        )

        if index in requested:
            combined = (
                combine_imm_states(
                    current_states,
                    current_probabilities,
                )
            )

            result_by_step[index] = (
                _prediction_point(
                    combined,
                    anchor_timestamp_s=(
                        anchor_timestamp_s
                    ),
                )
            )

    points = tuple(
        result_by_step[index]
        for index in requested_steps
    )

    return IMMPrediction(
        track_id=track_id,
        anchor_timestamp_s=(
            anchor_timestamp_s
        ),
        points=points,
        truth_used=False,
        annotated_velocity_used=False,
        annotated_heading_used=False,
        future_information_used=False,
    )

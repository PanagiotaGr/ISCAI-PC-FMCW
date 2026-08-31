from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.filters.kalman_ekf import (
    KalmanConfig,
    KalmanState,
    predict_kalman_state,
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
class KalmanPredictionPoint:
    horizon_s: float
    timestamp_s: float

    position_H0_m: Vec3
    position_covariance_H0_m2: Matrix3


@dataclass(frozen=True)
class KalmanPrediction:
    track_id: str
    anchor_timestamp_s: float

    points: tuple[
        KalmanPredictionPoint,
        ...
    ]

    model: str = "CV_EKF"

    truth_used: bool = False
    annotated_velocity_used: bool = False
    future_information_used: bool = False


def predict_kalman(
    *,
    track_id: str,
    state: KalmanState,
    horizons_s: tuple[
        float,
        ...
    ],
    config: KalmanConfig | None = None,
) -> KalmanPrediction:
    if config is None:
        config = KalmanConfig()

    if not horizons_s:
        raise ValueError(
            "At least one Kalman prediction "
            "horizon is required."
        )

    previous = None
    points = []

    for value in horizons_s:
        tau = float(value)

        if not isfinite(tau):
            raise ValueError(
                "Kalman horizons must "
                "be finite."
            )

        if tau <= 0.0:
            raise ValueError(
                "Kalman horizons must "
                "be positive."
            )

        if (
            previous is not None
            and tau <= previous
        ):
            raise ValueError(
                "Kalman horizons must be "
                "strictly increasing."
            )

        predicted = predict_kalman_state(
            state,
            target_timestamp_s=(
                state.timestamp_s
                +
                tau
            ),
            config=config,
        )

        P = predicted.covariance_6x6

        points.append(
            KalmanPredictionPoint(
                horizon_s=tau,
                timestamp_s=(
                    predicted.timestamp_s
                ),
                position_H0_m=(
                    predicted.mean_6[0],
                    predicted.mean_6[1],
                    predicted.mean_6[2],
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
            )
        )

        previous = tau

    return KalmanPrediction(
        track_id=track_id,
        anchor_timestamp_s=(
            state.timestamp_s
        ),
        points=tuple(points),
        truth_used=False,
        annotated_velocity_used=False,
        future_information_used=False,
    )

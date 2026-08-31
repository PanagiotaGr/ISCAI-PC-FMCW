from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.state import (
    CausalCVState,
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
class CVPredictionPoint:
    horizon_s: float
    timestamp_s: float

    position_H0_m: Vec3

    position_covariance_H0_m2: Matrix3


@dataclass(frozen=True)
class CVPrediction:
    track_id: str

    anchor_timestamp_s: float

    points: tuple[
        CVPredictionPoint,
        ...
    ]

    model: str = "constant_velocity"

    annotated_velocity_used: bool = False
    future_information_used: bool = False


def _position_covariance_at_horizon(
    state: CausalCVState,
    tau: float,
) -> Matrix3:
    P = state.covariance_6x6

    # p(t+tau) = p + tau*v
    #
    # P_tau =
    # P_pp +
    # tau(P_pv + P_vp) +
    # tau^2 P_vv

    return tuple(
        tuple(
            (
                P[i][j]
                +
                tau
                * (
                    P[i][j + 3]
                    +
                    P[i + 3][j]
                )
                +
                tau
                * tau
                * P[i + 3][j + 3]
            )
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def predict_cv(
    state: CausalCVState,
    *,
    horizons_s: tuple[
        float,
        ...
    ],
) -> CVPrediction:
    if not horizons_s:
        raise ValueError(
            "At least one horizon is required."
        )

    previous = None

    normalized = []

    for value in horizons_s:
        tau = float(value)

        if not isfinite(tau):
            raise ValueError(
                "Prediction horizons must "
                "be finite."
            )

        if tau < 0.0:
            raise ValueError(
                "Prediction horizons cannot "
                "be negative."
            )

        if (
            previous is not None
            and tau <= previous
        ):
            raise ValueError(
                "Prediction horizons must be "
                "strictly increasing."
            )

        normalized.append(tau)
        previous = tau

    points = []

    for tau in normalized:
        position = tuple(
            state.position_H0_m[i]
            +
            tau
            * state.velocity_H0_mps[i]
            for i in range(3)
        )

        covariance = (
            _position_covariance_at_horizon(
                state,
                tau,
            )
        )

        points.append(
            CVPredictionPoint(
                horizon_s=tau,
                timestamp_s=(
                    state.timestamp_s
                    +
                    tau
                ),
                position_H0_m=position,
                position_covariance_H0_m2=(
                    covariance
                ),
            )
        )

    return CVPrediction(
        track_id=state.track_id,
        anchor_timestamp_s=(
            state.timestamp_s
        ),
        points=tuple(points),
        annotated_velocity_used=False,
        future_information_used=False,
    )

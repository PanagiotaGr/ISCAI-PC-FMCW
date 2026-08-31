from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.state.ca import (
    CausalCAState,
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
class CAPredictionPoint:
    horizon_s: float
    timestamp_s: float

    position_H0_m: Vec3
    position_covariance_H0_m2: Matrix3


@dataclass(frozen=True)
class CAPrediction:
    track_id: str
    anchor_timestamp_s: float

    points: tuple[
        CAPredictionPoint,
        ...
    ]

    model: str = "constant_acceleration"

    annotated_velocity_used: bool = False
    annotated_acceleration_used: bool = False
    future_information_used: bool = False


def _position_covariance_at_horizon(
    state: CausalCAState,
    tau: float,
) -> Matrix3:
    """
    Propagate the full causal 9x9 covariance
    through

        p(t+tau) = p + tau*v + 0.5*tau^2*a
    """

    coefficients = (
        1.0,
        tau,
        0.5 * tau * tau,
    )

    P = state.covariance_9x9

    result = []

    for i in range(3):
        row = []

        for j in range(3):
            value = 0.0

            for block_i in range(3):
                for block_j in range(3):
                    value += (
                        coefficients[block_i]
                        *
                        coefficients[block_j]
                        *
                        P[
                            block_i * 3 + i
                        ][
                            block_j * 3 + j
                        ]
                    )

            row.append(value)

        result.append(tuple(row))

    matrix = tuple(result)

    return tuple(
        tuple(
            0.5
            * (
                matrix[i][j]
                +
                matrix[j][i]
            )
            for j in range(3)
        )
        for i in range(3)
    )  # type: ignore[return-value]


def predict_ca(
    state: CausalCAState,
    *,
    horizons_s: tuple[
        float,
        ...
    ],
) -> CAPrediction:
    if not horizons_s:
        raise ValueError(
            "At least one prediction horizon "
            "is required."
        )

    normalized = []
    previous = None

    for value in horizons_s:
        tau = float(value)

        if not isfinite(tau):
            raise ValueError(
                "CA horizons must be finite."
            )

        if tau < 0.0:
            raise ValueError(
                "CA horizons cannot be negative."
            )

        if (
            previous is not None
            and tau <= previous
        ):
            raise ValueError(
                "CA horizons must be strictly "
                "increasing."
            )

        normalized.append(tau)
        previous = tau

    points = []

    for tau in normalized:
        position = tuple(
            state.position_H0_m[i]
            +
            state.velocity_H0_mps[i] * tau
            +
            0.5
            * state.acceleration_H0_mps2[i]
            * tau
            * tau
            for i in range(3)
        )

        points.append(
            CAPredictionPoint(
                horizon_s=tau,
                timestamp_s=(
                    state.timestamp_s + tau
                ),
                position_H0_m=position,
                position_covariance_H0_m2=(
                    _position_covariance_at_horizon(
                        state,
                        tau,
                    )
                ),
            )
        )

    return CAPrediction(
        track_id=state.track_id,
        anchor_timestamp_s=(
            state.timestamp_s
        ),
        points=tuple(points),
        annotated_velocity_used=False,
        annotated_acceleration_used=False,
        future_information_used=False,
    )

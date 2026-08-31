from __future__ import annotations

from dataclasses import dataclass
from math import (
    cos,
    isfinite,
    sin,
)

from iscai_stage3.observations import (
    wrap_angle_rad,
)

from iscai_stage3.state.ctrv import (
    CausalCTRVState,
)


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class CTRVPredictionPoint:
    horizon_s: float
    timestamp_s: float

    position_H0_m: Vec3
    heading_rad: float


@dataclass(frozen=True)
class CTRVPrediction:
    track_id: str
    anchor_timestamp_s: float

    points: tuple[
        CTRVPredictionPoint,
        ...
    ]

    model: str = "ctrv"

    annotated_velocity_used: bool = False
    annotated_heading_used: bool = False
    future_information_used: bool = False


def predict_ctrv(
    state: CausalCTRVState,
    *,
    horizons_s: tuple[
        float,
        ...
    ],
    turn_rate_epsilon_radps: float = 1e-6,
) -> CTRVPrediction:
    if not horizons_s:
        raise ValueError(
            "At least one CTRV horizon "
            "is required."
        )

    if (
        not isfinite(
            turn_rate_epsilon_radps
        )
        or turn_rate_epsilon_radps <= 0.0
    ):
        raise ValueError(
            "turn-rate epsilon must be "
            "finite and positive."
        )

    normalized = []
    previous = None

    for value in horizons_s:
        tau = float(value)

        if not isfinite(tau):
            raise ValueError(
                "CTRV horizons must be finite."
            )

        if tau < 0.0:
            raise ValueError(
                "CTRV horizons cannot "
                "be negative."
            )

        if (
            previous is not None
            and tau <= previous
        ):
            raise ValueError(
                "CTRV horizons must be "
                "strictly increasing."
            )

        normalized.append(tau)
        previous = tau

    x0, y0, z0 = (
        state.position_H0_m
    )

    speed = (
        state.planar_speed_mps
    )

    heading = state.heading_rad
    omega = state.turn_rate_radps

    points = []

    for tau in normalized:
        future_heading = (
            wrap_angle_rad(
                heading
                +
                omega * tau
            )
        )

        if (
            abs(omega)
            <= turn_rate_epsilon_radps
        ):
            x = (
                x0
                +
                speed
                * cos(heading)
                * tau
            )

            y = (
                y0
                +
                speed
                * sin(heading)
                * tau
            )

        else:
            x = (
                x0
                +
                speed
                / omega
                * (
                    sin(
                        heading
                        +
                        omega
                        * tau
                    )
                    -
                    sin(heading)
                )
            )

            y = (
                y0
                +
                speed
                / omega
                * (
                    -cos(
                        heading
                        +
                        omega
                        * tau
                    )
                    +
                    cos(heading)
                )
            )

        z = (
            z0
            +
            state.vertical_velocity_mps
            * tau
        )

        points.append(
            CTRVPredictionPoint(
                horizon_s=tau,
                timestamp_s=(
                    state.timestamp_s
                    +
                    tau
                ),
                position_H0_m=(
                    x,
                    y,
                    z,
                ),
                heading_rad=(
                    future_heading
                ),
            )
        )

    return CTRVPrediction(
        track_id=state.track_id,
        anchor_timestamp_s=(
            state.timestamp_s
        ),
        points=tuple(points),
        annotated_velocity_used=False,
        annotated_heading_used=False,
        future_information_used=False,
    )

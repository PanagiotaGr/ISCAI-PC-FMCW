from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage3.hough import HoughTrack


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class HoughPredictionPoint:
    horizon_s: float
    timestamp_s: float
    position_H0_m: Vec3


@dataclass(frozen=True)
class HoughPrediction:
    track_id: str
    anchor_timestamp_s: float

    points: tuple[
        HoughPredictionPoint,
        ...
    ]

    model: str = "multidimensional_hough"

    truth_used: bool = False
    estimated_association_used: bool = False
    actor_identity_used: bool = False
    future_information_used: bool = False


def predict_hough(
    track: HoughTrack,
    *,
    horizons_s: tuple[
        float,
        ...
    ],
) -> HoughPrediction:
    if not horizons_s:
        raise ValueError(
            "At least one Hough prediction "
            "horizon is required."
        )

    previous = None
    points = []

    for horizon in horizons_s:
        tau = float(horizon)

        if (
            not isfinite(tau)
            or
            tau <= 0.0
        ):
            raise ValueError(
                "Hough horizons must be "
                "finite and positive."
            )

        if (
            previous is not None
            and tau <= previous
        ):
            raise ValueError(
                "Hough horizons must be "
                "strictly increasing."
            )

        position = tuple(
            track.anchor_position_H0_m[index]
            +
            tau
            *
            track.velocity_H0_mps[index]
            for index in range(3)
        )

        points.append(
            HoughPredictionPoint(
                horizon_s=tau,
                timestamp_s=(
                    track.anchor_timestamp_s
                    +
                    tau
                ),
                position_H0_m=position,
            )
        )

        previous = tau

    return HoughPrediction(
        track_id=track.track_id,
        anchor_timestamp_s=(
            track.anchor_timestamp_s
        ),
        points=tuple(points),
        truth_used=False,
        estimated_association_used=False,
        actor_identity_used=False,
        future_information_used=False,
    )

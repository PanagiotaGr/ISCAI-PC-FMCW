from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryState:
    """
    Common causal trajectory state.

    No future information.
    """

    timestamp_s: float

    position_m: tuple[
        float,
        float,
        float,
    ]

    velocity_mps: tuple[
        float,
        float,
        float,
    ]

    acceleration_mps2: tuple[
        float,
        float,
        float,
    ] | None = None


@dataclass(frozen=True)
class PredictionRequest:

    horizon_s: float

    dt: float


@dataclass(frozen=True)
class BaselinePrediction:

    model_name: str

    states: tuple[
        tuple[float, float, float],
        ...
    ]

    causal_only: bool = True

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryPrediction:
    """
    Common Stage3 baseline output.
    """

    model_name: str

    states: tuple[
        tuple[float, float, float],
        ...
    ]

    causal_only: bool = True

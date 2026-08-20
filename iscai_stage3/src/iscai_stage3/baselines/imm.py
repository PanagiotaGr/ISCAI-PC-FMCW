from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IMMState:

    selected_model: str

    model_probabilities: dict[str, float]


def select_imm_model(
    *,
    velocity_score: float,
    acceleration_score: float,
    turn_rate_score: float,
) -> IMMState:
    """
    Lightweight IMM model selector.

    Higher score = more likely motion model.
    """

    scores = {
        "CV": velocity_score,
        "CA": acceleration_score,
        "CTRV": turn_rate_score,
    }

    total = sum(scores.values())

    if total <= 0:
        raise ValueError(
            "Invalid IMM scores."
        )

    probabilities = {
        key: value / total
        for key, value in scores.items()
    }

    selected = max(
        probabilities,
        key=probabilities.get,
    )

    return IMMState(
        selected_model=selected,
        model_probabilities=probabilities,
    )

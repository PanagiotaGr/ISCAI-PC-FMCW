
from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class ADBTargetScore:
    """
    Intermediate ADB decision score.
    """

    target_id: int

    beam_confidence: float

    priority: float

    uncertainty_penalty: float

    utility: float



def compute_adb_utility(
    *,
    beam_confidence: float,
    priority: float,
    uncertainty_penalty: float,
    uncertainty_weight: float = 1.0,
) -> float:
    """
    Stage3 ADB utility contract.

    U =
        priority * beam_confidence
        -
        uncertainty_weight * uncertainty
    """

    if beam_confidence < 0.0:
        raise ValueError(
            "Negative beam confidence."
        )

    if priority < 0.0:
        raise ValueError(
            "Negative priority."
        )

    if uncertainty_penalty < 0.0:
        raise ValueError(
            "Negative uncertainty."
        )

    if uncertainty_weight < 0.0:
        raise ValueError(
            "Negative uncertainty weight."
        )

    return (
        priority * beam_confidence
        -
        uncertainty_weight *
        uncertainty_penalty
    )



def choose_best_target(
    targets,
):
    """
    Deterministic ADB target selection.

    Physical rule:
    1. maximize beam confidence
    2. use utility as tie breaker

    This prevents low-confidence targets
    from winning only because they have
    small uncertainty.
    """

    if len(targets) == 0:
        raise ValueError(
            "No ADB targets."
        )

    return max(
        targets,
        key=lambda x: (
            x.beam_confidence,
            x.utility,
        ),
    )

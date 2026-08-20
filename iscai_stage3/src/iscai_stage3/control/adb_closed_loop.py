from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class ADBDecision:

    target_id: int

    utility: float

    uncertainty: float



def select_next_action(
    *,
    candidates,
) -> ADBDecision:

    if len(candidates) == 0:
        raise ValueError(
            "No candidates."
        )


    ranked = sorted(
        candidates,
        key=lambda x: (
            x.utility -
            x.uncertainty_penalty
        ),
        reverse=True,
    )


    best = ranked[0]


    return ADBDecision(

        target_id=best.target_id,

        utility=best.utility,

        uncertainty=best.uncertainty_penalty,

    )

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectedPrediction:

    actor_id: int

    model_name: str

    trajectory: tuple

    utility: float



def select_best_prediction(
    candidates,
):
    """
    Deterministic best candidate selection.

    Highest utility wins.
    Tie broken by model name.
    """


    if len(candidates) == 0:

        raise ValueError(
            "No candidates."
        )


    for candidate in candidates:

        if not (
            0.0 <= candidate.utility <= 1.0
        ):

            raise ValueError(
                "Invalid utility."
            )


    ordered = sorted(

        candidates,

        key=lambda x:
            (
                -x.utility,
                x.model_name,
            ),

    )


    best = ordered[0]


    return SelectedPrediction(

        actor_id=
            best.actor_id,

        model_name=
            best.model_name,

        trajectory=
            best.trajectory,

        utility=
            best.utility,

    )

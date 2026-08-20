from __future__ import annotations

from dataclasses import dataclass


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class ControllerPrediction:

    actor_id: int

    model_name: str

    trajectory: tuple[Vec3, ...]

    confidence: float

    utility: float



def prediction_to_controller(
    prediction: dict,
) -> ControllerPrediction:
    """
    Convert Stage4 probabilistic prediction
    to controller candidate.
    """


    uncertainty = prediction[
        "uncertainty"
    ]


    confidence = (
        1.0
        -
        uncertainty
    )


    trajectory = tuple(

        tuple(point)

        for point
        in prediction["trajectory"]

    )


    if len(trajectory) == 0:

        raise ValueError(
            "Empty trajectory."
        )


    if not (
        0.0 <= confidence <= 1.0
    ):

        raise ValueError(
            "Invalid confidence."
        )


    utility = confidence


    return ControllerPrediction(

        actor_id=
            prediction["actor_id"],

        model_name=
            prediction["model_name"],

        trajectory=
            trajectory,

        confidence=
            confidence,

        utility=
            utility,
    )

from __future__ import annotations

from dataclasses import dataclass


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class RegisteredPrediction:

    model_name: str

    states: tuple[
        Vec3,
        ...
    ]

    causal_only: bool

    future_used: bool = False



def validate_prediction(
    prediction: RegisteredPrediction,
) -> None:

    if not prediction.causal_only:
        raise ValueError(
            "Prediction must be causal."
        )

    if prediction.future_used:
        raise ValueError(
            "Future information leaked."
        )

    for state in prediction.states:

        if len(state) != 3:
            raise ValueError(
                "Invalid state dimension."
            )



def register_prediction(
    *,
    model_name: str,
    states: tuple[Vec3, ...],
) -> RegisteredPrediction:


    prediction = RegisteredPrediction(

        model_name=model_name,

        states=states,

        causal_only=True,

        future_used=False,
    )


    validate_prediction(
        prediction
    )


    return prediction

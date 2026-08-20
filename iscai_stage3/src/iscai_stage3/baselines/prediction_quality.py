from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class PredictionQualityRecord:

    model_name: str

    trajectory_length: int

    finite: bool

    causal_only: bool



def validate_trajectory(
    trajectory: tuple[Vec3, ...],
) -> None:

    if len(trajectory) == 0:
        raise ValueError(
            "Empty trajectory."
        )

    for state in trajectory:

        if len(state) != 3:
            raise ValueError(
                "Invalid state dimension."
            )

        if not all(
            math.isfinite(v)
            for v in state
        ):
            raise ValueError(
                "Non finite prediction."
            )



def evaluate_prediction(
    *,
    model_name: str,
    trajectory: tuple[Vec3, ...],
) -> PredictionQualityRecord:

    validate_trajectory(
        trajectory
    )

    return PredictionQualityRecord(

        model_name=model_name,

        trajectory_length=len(
            trajectory
        ),

        finite=True,

        causal_only=True,
    )



def prediction_sha256(
    records: tuple[
        PredictionQualityRecord,
        ...
    ],
) -> str:

    payload = json.dumps(
        [
            r.__dict__
            for r in records
        ],
        sort_keys=True,
    )

    return hashlib.sha256(
        payload.encode()
    ).hexdigest()

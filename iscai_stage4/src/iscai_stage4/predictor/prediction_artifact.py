from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class PredictionArtifact:

    actor_id: int

    model_name: str

    trajectory: tuple[Vec3, ...]

    uncertainty: float

    future_used: bool = False



    def validate(self):

        if self.actor_id < 0:

            raise ValueError(
                "Invalid actor id."
            )


        if not self.model_name:

            raise ValueError(
                "Missing model name."
            )


        if len(self.trajectory) == 0:

            raise ValueError(
                "Empty trajectory."
            )


        if not (
            0.0 <= self.uncertainty
            <= 1.0
        ):

            raise ValueError(
                "Invalid uncertainty."
            )


        if self.future_used:

            raise ValueError(
                "Future leakage."
            )



    def to_dict(self):

        return {

            "actor_id":
                self.actor_id,

            "model_name":
                self.model_name,

            "trajectory":
                [
                    list(p)
                    for p in self.trajectory
                ],

            "uncertainty":
                self.uncertainty,

            "future_used":
                self.future_used,
        }



def save_prediction_artifacts(
    artifacts,
    path: str,
):

    payload = [
        x.to_dict()
        for x in artifacts
    ]


    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass



@dataclass(frozen=True)
class Stage4Prediction:

    actor_id: int

    model_name: str

    trajectory: tuple

    uncertainty: float



def load_stage4_predictions(
    path: str,
):

    data = json.loads(
        Path(path).read_text()
    )


    predictions = []


    for item in data:

        if item["future_used"]:

            raise RuntimeError(
                "Future leakage."
            )


        predictions.append(

            Stage4Prediction(

                actor_id=
                    item.get(
                        "actor_id",
                        item.get(
                            "track_index"
                        )
                    ),

                model_name=
                    item["model_name"],

                trajectory=tuple(
                    tuple(x)
                    for x in item["trajectory"]
                ),

                uncertainty=
                    item["uncertainty"],
            )

        )


    return tuple(
        predictions
    )

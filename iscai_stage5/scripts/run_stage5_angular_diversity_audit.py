from collections import defaultdict
from pathlib import Path
import json
import math


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

HORIZONS = (
    1,
    3,
    5,
    9,
)

CODEBOOKS = (
    16,
    32,
    64,
)


data = json.loads(
    SOURCE.read_text()
)


groups = defaultdict(dict)


for item in data:

    key = (
        item["scenario_id"],
        int(item["track_index"]),
        item["actor_type"],
    )

    model = item.get(
        "model_name",
        "UNKNOWN",
    )

    if model in MODELS:
        groups[key][model] = item


def beam_index(angle, beams):

    normalized = (
        angle + math.pi
    ) / (
        2.0 * math.pi
    )

    index = int(
        normalized * beams
    )

    return max(
        0,
        min(
            beams - 1,
            index,
        ),
    )


print(
    "===== Stage5 Angular Diversity Audit ====="
)


for horizon in HORIZONS:

    print(
        f"\nHORIZON INDEX = {horizon}"
    )

    valid = 0

    angular_spreads = []

    disagreement = {
        beams: 0
        for beams in CODEBOOKS
    }


    for key, records in groups.items():

        if not all(
            model in records
            for model in MODELS
        ):
            continue


        angles = []


        for model in MODELS:

            trajectory = records[
                model
            ].get(
                "trajectory",
                [],
            )

            if len(trajectory) <= horizon:
                break


            point = trajectory[horizon]

            x = float(
                point[0]
            )

            y = float(
                point[1]
            )

            angles.append(
                math.atan2(
                    y,
                    x,
                )
            )


        if len(angles) != len(MODELS):
            continue


        valid += 1

        spread = (
            max(angles)
            -
            min(angles)
        )

        angular_spreads.append(
            abs(spread)
        )


        for beams in CODEBOOKS:

            indices = {
                beam_index(
                    angle,
                    beams,
                )
                for angle in angles
            }

            if len(indices) > 1:

                disagreement[
                    beams
                ] += 1


    mean_spread = (
        sum(angular_spreads)
        /
        len(angular_spreads)
        if angular_spreads
        else 0.0
    )


    max_spread = (
        max(angular_spreads)
        if angular_spreads
        else 0.0
    )


    print(
        "actors =",
        valid,
    )

    print(
        "mean angular spread deg =",
        math.degrees(
            mean_spread
        ),
    )

    print(
        "max angular spread deg =",
        math.degrees(
            max_spread
        ),
    )


    for beams in CODEBOOKS:

        fraction = (
            disagreement[beams]
            /
            valid
            if valid
            else 0.0
        )

        print(
            f"{beams} beams: "
            f"model-disagreement actors = "
            f"{disagreement[beams]} "
            f"({fraction:.4%})"
        )

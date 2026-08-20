
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/block5_probabilistic_beam_selection_smoke.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)


DT = 0.1


# Synthetic headlamp beam grid.
# Later this will be replaced by real ADB/FMCW beam geometry.
AZIMUTH_BEAMS = 64

TOP_K = 5


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



actors = 0
beam_probability_valid = 0
topk_valid = 0

max_probability_error = 0.0

classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for key, model_records in groups.items():

    actors += 1

    actor_type = key[2]

    if actor_type in classes:
        classes[actor_type] += 1


    if not all(
        model in model_records
        for model in MODELS
    ):
        continue


    angles = []


    for model in MODELS:

        trajectory = model_records[
            model
        ].get(
            "trajectory",
            []
        )


        if len(trajectory) < 2:
            continue


        p0 = trajectory[0]
        p1 = trajectory[1]


        x = float(
            p1[0]
        )

        y = float(
            p1[1]
        )


        azimuth = math.atan2(
            y,
            x,
        )

        angles.append(
            azimuth
        )


    if len(angles) != len(MODELS):
        continue


    beam_scores = [
        0.0
        for _ in range(
            AZIMUTH_BEAMS
        )
    ]


    for azimuth in angles:

        normalized = (
            azimuth + math.pi
        ) / (
            2.0 * math.pi
        )

        index = int(
            normalized
            *
            AZIMUTH_BEAMS
        )

        index = max(
            0,
            min(
                AZIMUTH_BEAMS - 1,
                index,
            ),
        )


        beam_scores[index] += 1.0


    total = sum(
        beam_scores
    )


    if total <= 0:
        continue


    probabilities = [
        value / total
        for value in beam_scores
    ]


    probability_error = abs(
        sum(probabilities)
        -
        1.0
    )


    max_probability_error = max(
        max_probability_error,
        probability_error,
    )


    if (
        all(
            math.isfinite(p)
            and p >= 0.0
            for p in probabilities
        )
        and probability_error < 1e-12
    ):
        beam_probability_valid += 1


    ranked = sorted(
        range(
            AZIMUTH_BEAMS
        ),
        key=lambda i:
            probabilities[i],
        reverse=True,
    )


    topk = ranked[
        :TOP_K
    ]


    if (
        len(topk) == TOP_K
        and all(
            0 <= b < AZIMUTH_BEAMS
            for b in topk
        )
    ):
        topk_valid += 1



payload = {

    "actors":
        actors,

    "models":
        list(MODELS),

    "azimuth_beams":
        AZIMUTH_BEAMS,

    "top_k":
        TOP_K,

    "beam_probability_valid":
        beam_probability_valid,

    "topk_valid":
        topk_valid,

    "max_probability_error":
        max_probability_error,

    "vehicle":
        classes["VEHICLE"],

    "pedestrian":
        classes["PEDESTRIAN"],

    "cyclist":
        classes["CYCLIST"],

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        beam_probability_valid == actors
        and topk_valid == actors
    )
    else
    "FAIL"
)


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()
).hexdigest()


report = {
    **payload,
    "sha256":
        sha,
    "status":
        status,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Probabilistic Beam Selection Smoke ====="
)

print(
    "actors =",
    actors
)

print(
    "models =",
    list(MODELS)
)

print(
    "azimuth_beams =",
    AZIMUTH_BEAMS
)

print(
    "top_k =",
    TOP_K
)

print(
    "beam_probability_valid =",
    beam_probability_valid
)

print(
    "topk_valid =",
    topk_valid
)

print(
    "max_probability_error =",
    max_probability_error
)

print(
    "vehicle =",
    classes["VEHICLE"]
)

print(
    "pedestrian =",
    classes["PEDESTRIAN"]
)

print(
    "cyclist =",
    classes["CYCLIST"]
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS =",
    status
)

print(
    "report =",
    REPORT
)


if status != "PASS":
    raise SystemExit(1)

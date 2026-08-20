
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
    "reports/block5_ablation_no_imm.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)


data = json.loads(
    SOURCE.read_text()
)


groups = defaultdict(dict)


for item in data:

    key = (
        str(item["scenario_id"]),
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
complete_sets = 0

full_dispersion_sum = 0.0
cv_deviation_sum = 0.0

full_valid = 0
cv_valid = 0


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


    trajectories = {
        model:
            model_records[model].get(
                "trajectory",
                [],
            )
        for model in MODELS
    }


    lengths = [
        len(trajectory)
        for trajectory in trajectories.values()
    ]


    if min(lengths) <= 0:
        continue


    horizon = min(
        lengths
    )

    complete_sets += 1


    actor_full_dispersion = 0.0
    actor_cv_deviation = 0.0


    for step in range(
        horizon
    ):

        points = [
            trajectories[model][step]
            for model in MODELS
        ]


        cx = sum(
            float(point[0])
            for point in points
        ) / len(points)

        cy = sum(
            float(point[1])
            for point in points
        ) / len(points)


        # Multi-model dispersion around consensus.
        step_dispersion = sum(
            (
                float(point[0]) - cx
            ) ** 2
            +
            (
                float(point[1]) - cy
            ) ** 2
            for point in points
        ) / len(points)


        actor_full_dispersion += (
            math.sqrt(
                step_dispersion
            )
        )


        # Ablated system = CV only.
        cv_point = trajectories[
            "CV"
        ][step]


        cv_deviation = math.sqrt(
            (
                float(cv_point[0]) - cx
            ) ** 2
            +
            (
                float(cv_point[1]) - cy
            ) ** 2
        )


        actor_cv_deviation += (
            cv_deviation
        )


    actor_full_dispersion /= horizon
    actor_cv_deviation /= horizon


    if math.isfinite(
        actor_full_dispersion
    ):

        full_dispersion_sum += (
            actor_full_dispersion
        )

        full_valid += 1


    if math.isfinite(
        actor_cv_deviation
    ):

        cv_deviation_sum += (
            actor_cv_deviation
        )

        cv_valid += 1


mean_full_dispersion = (
    full_dispersion_sum
    /
    full_valid
    if full_valid > 0
    else 0.0
)


mean_cv_deviation = (
    cv_deviation_sum
    /
    cv_valid
    if cv_valid > 0
    else 0.0
)


payload = {

    "actors":
        actors,

    "complete_model_sets":
        complete_sets,

    "full_system_models":
        list(MODELS),

    "ablated_system":
        "CV_ONLY",

    "mean_multimodel_dispersion_m":
        mean_full_dispersion,

    "mean_cv_to_consensus_deviation_m":
        mean_cv_deviation,

    "full_valid":
        full_valid,

    "cv_valid":
        cv_valid,

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
        actors > 0
        and complete_sets == actors
        and full_valid == actors
        and cv_valid == actors
        and mean_full_dispersion >= 0.0
        and mean_cv_deviation >= 0.0
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
    "===== Stage5 Ablation: No IMM ====="
)

print(
    "actors =",
    actors
)

print(
    "complete_model_sets =",
    complete_sets
)

print(
    "full_system_models =",
    list(MODELS)
)

print(
    "ablated_system = CV_ONLY"
)

print(
    "mean_multimodel_dispersion_m =",
    mean_full_dispersion
)

print(
    "mean_cv_to_consensus_deviation_m =",
    mean_cv_deviation
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

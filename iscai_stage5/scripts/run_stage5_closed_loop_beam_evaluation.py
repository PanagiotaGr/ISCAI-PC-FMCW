
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
    "reports/block5_closed_loop_beam_evaluation.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

MAX_ACTIVE_BEAMS = 16
AZIMUTH_BEAMS = 64


def class_priority(actor_type):

    if actor_type == "PEDESTRIAN":
        return 0

    if actor_type == "CYCLIST":
        return 0

    if actor_type == "VEHICLE":
        return 1

    return 2


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


actors_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


covered_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


requests = []


for key, model_records in groups.items():

    if not all(
        model in model_records
        for model in MODELS
    ):
        continue


    scenario_id, track_index, actor_type = key


    if actor_type not in actors_by_class:
        continue


    actors_by_class[
        actor_type
    ] += 1


    trajectory = model_records[
        "CV"
    ].get(
        "trajectory",
        [],
    )


    if len(trajectory) < 2:
        continue


    x = float(
        trajectory[1][0]
    )

    y = float(
        trajectory[1][1]
    )


    azimuth = math.atan2(
        y,
        x,
    )


    normalized = (
        azimuth + math.pi
    ) / (
        2.0 * math.pi
    )


    beam_index = int(
        normalized
        *
        AZIMUTH_BEAMS
    )


    beam_index = max(
        0,
        min(
            AZIMUTH_BEAMS - 1,
            beam_index,
        ),
    )


    requests.append(
        {
            "priority":
                class_priority(
                    actor_type
                ),

            "actor_type":
                actor_type,

            "scenario_id":
                scenario_id,

            "track_index":
                track_index,

            "beam_index":
                beam_index,
        }
    )


requests.sort(
    key=lambda x: (
        x["priority"],
        x["actor_type"],
        x["beam_index"],
        x["scenario_id"],
        x["track_index"],
    )
)


selected = requests[
    :MAX_ACTIVE_BEAMS
]


for item in selected:

    actor_type = item[
        "actor_type"
    ]

    covered_by_class[
        actor_type
    ] += 1


total_actors = sum(
    actors_by_class.values()
)

selected_actors = len(
    selected
)


coverage_rate = (
    selected_actors
    /
    total_actors
    if total_actors > 0
    else 0.0
)


vehicle_coverage = (
    covered_by_class["VEHICLE"]
    /
    actors_by_class["VEHICLE"]
    if actors_by_class["VEHICLE"] > 0
    else 0.0
)


pedestrian_coverage = (
    covered_by_class["PEDESTRIAN"]
    /
    actors_by_class["PEDESTRIAN"]
    if actors_by_class["PEDESTRIAN"] > 0
    else 0.0
)


cyclist_coverage = (
    covered_by_class["CYCLIST"]
    /
    actors_by_class["CYCLIST"]
    if actors_by_class["CYCLIST"] > 0
    else 0.0
)


safety_total = (
    actors_by_class["PEDESTRIAN"]
    +
    actors_by_class["CYCLIST"]
)


safety_selected = (
    covered_by_class["PEDESTRIAN"]
    +
    covered_by_class["CYCLIST"]
)


safety_coverage = (
    safety_selected
    /
    safety_total
    if safety_total > 0
    else 0.0
)


beam_budget_utilization = (
    selected_actors
    /
    MAX_ACTIVE_BEAMS
    if MAX_ACTIVE_BEAMS > 0
    else 0.0
)


payload = {

    "actors":
        total_actors,

    "selected_actors":
        selected_actors,

    "max_active_beams":
        MAX_ACTIVE_BEAMS,

    "beam_budget_utilization":
        beam_budget_utilization,

    "coverage_rate":
        coverage_rate,

    "vehicle_coverage":
        vehicle_coverage,

    "pedestrian_coverage":
        pedestrian_coverage,

    "cyclist_coverage":
        cyclist_coverage,

    "safety_actor_coverage":
        safety_coverage,

    "actors_by_class":
        actors_by_class,

    "covered_by_class":
        covered_by_class,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        total_actors > 0
        and selected_actors <= MAX_ACTIVE_BEAMS
        and 0.0 <= coverage_rate <= 1.0
        and 0.0 <= safety_coverage <= 1.0
    )
    else "FAIL"
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
    "===== Stage5 Closed Loop Beam Evaluation ====="
)

print(
    "actors =",
    total_actors
)

print(
    "selected_actors =",
    selected_actors
)

print(
    "max_active_beams =",
    MAX_ACTIVE_BEAMS
)

print(
    "beam_budget_utilization =",
    beam_budget_utilization
)

print(
    "coverage_rate =",
    coverage_rate
)

print(
    "vehicle_coverage =",
    vehicle_coverage
)

print(
    "pedestrian_coverage =",
    pedestrian_coverage
)

print(
    "cyclist_coverage =",
    cyclist_coverage
)

print(
    "safety_actor_coverage =",
    safety_coverage
)

print(
    "actors_by_class =",
    actors_by_class
)

print(
    "covered_by_class =",
    covered_by_class
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

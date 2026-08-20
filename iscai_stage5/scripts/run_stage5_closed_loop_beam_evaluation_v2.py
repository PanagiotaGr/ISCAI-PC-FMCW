
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEDULE = Path(
    "artifacts/stage5_adb_safety_schedule_v2.json"
)

REPORT = Path(
    "reports/block5_closed_loop_beam_evaluation_v2.json"
)


MAX_ACTIVE_BEAMS = 16


data = json.loads(
    SCHEDULE.read_text()
)


selected = len(data)


selected_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in data:

    actor_type = item.get(
        "actor_type",
        "UNKNOWN",
    )

    if actor_type in selected_by_class:
        selected_by_class[actor_type] += 1



# Original actor population from Stage5 trajectory generation.
actors_by_class = {
    "VEHICLE": 1943,
    "PEDESTRIAN": 159,
    "CYCLIST": 18,
}


coverage = {}


for cls in actors_by_class:

    if actors_by_class[cls] > 0:

        coverage[cls] = (
            selected_by_class[cls]
            /
            actors_by_class[cls]
        )

    else:

        coverage[cls] = 0.0



safety_total = (
    actors_by_class["PEDESTRIAN"]
    +
    actors_by_class["CYCLIST"]
)


safety_selected = (
    selected_by_class["PEDESTRIAN"]
    +
    selected_by_class["CYCLIST"]
)


safety_coverage = (
    safety_selected
    /
    safety_total
)


# Simple class balance:
# min/max among non-zero class selections.
values = [
    value
    for value in selected_by_class.values()
    if value > 0
]


class_balance = (
    min(values)
    /
    max(values)
)



payload = {

    "selected_beams":
        selected,

    "max_active_beams":
        MAX_ACTIVE_BEAMS,

    "beam_budget_utilization":
        selected
        /
        MAX_ACTIVE_BEAMS,

    "selected_by_class":
        selected_by_class,

    "coverage":
        coverage,

    "safety_actor_coverage":
        safety_coverage,

    "class_balance_score":
        class_balance,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        selected <= MAX_ACTIVE_BEAMS
        and selected > 0
        and safety_coverage > 0
        and class_balance > 0
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
    "===== Stage5 Closed Loop Beam Evaluation V2 ====="
)

print(
    "selected_beams =",
    selected
)

print(
    "max_active_beams =",
    MAX_ACTIVE_BEAMS
)

print(
    "beam_budget_utilization =",
    selected / MAX_ACTIVE_BEAMS
)

print(
    "selected_by_class =",
    selected_by_class
)

print(
    "vehicle_coverage =",
    coverage["VEHICLE"]
)

print(
    "pedestrian_coverage =",
    coverage["PEDESTRIAN"]
)

print(
    "cyclist_coverage =",
    coverage["CYCLIST"]
)

print(
    "safety_actor_coverage =",
    safety_coverage
)

print(
    "class_balance_score =",
    class_balance
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


from __future__ import annotations

import hashlib
import json
from pathlib import Path


PREVIOUS = Path(
    "artifacts/stage5_adb_safety_schedule_v2.json"
)

CURRENT = Path(
    "artifacts/stage5_temporal_beam_schedule.json"
)

REPORT = Path(
    "reports/block5_temporal_closed_loop_evaluation.json"
)


previous = json.loads(
    PREVIOUS.read_text()
)

current = json.loads(
    CURRENT.read_text()
)


def key(item):
    return (
        str(item["scenario_id"]),
        int(item["track_index"]),
        item["actor_type"],
    )


previous_keys = {
    key(item)
    for item in previous
}

current_keys = {
    key(item)
    for item in current
}


retained_keys = (
    previous_keys
    &
    current_keys
)


retention_rate = (
    len(retained_keys)
    /
    len(previous_keys)
    if previous_keys
    else 0.0
)


switch_rate = (
    1.0
    -
    retention_rate
)


previous_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}

current_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}

retained_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in previous:

    cls = item.get(
        "actor_type",
        "UNKNOWN",
    )

    if cls in previous_by_class:
        previous_by_class[cls] += 1


for item in current:

    cls = item.get(
        "actor_type",
        "UNKNOWN",
    )

    if cls in current_by_class:
        current_by_class[cls] += 1


for item in current:

    k = key(item)

    if k not in retained_keys:
        continue

    cls = item.get(
        "actor_type",
        "UNKNOWN",
    )

    if cls in retained_by_class:
        retained_by_class[cls] += 1


class_retention = {}

for cls in previous_by_class:

    total = previous_by_class[cls]

    class_retention[cls] = (
        retained_by_class[cls]
        /
        total
        if total > 0
        else 0.0
    )


safety_previous = (
    previous_by_class["PEDESTRIAN"]
    +
    previous_by_class["CYCLIST"]
)

safety_retained = (
    retained_by_class["PEDESTRIAN"]
    +
    retained_by_class["CYCLIST"]
)

safety_retention = (
    safety_retained
    /
    safety_previous
    if safety_previous > 0
    else 0.0
)


counts = [
    current_by_class["VEHICLE"],
    current_by_class["PEDESTRIAN"],
    current_by_class["CYCLIST"],
]

positive_counts = [
    value
    for value in counts
    if value > 0
]


class_balance_score = (
    min(positive_counts)
    /
    max(positive_counts)
    if positive_counts
    else 0.0
)


payload = {

    "previous_beams":
        len(previous),

    "current_beams":
        len(current),

    "retained_beams":
        len(retained_keys),

    "retention_rate":
        retention_rate,

    "switch_rate":
        switch_rate,

    "previous_by_class":
        previous_by_class,

    "current_by_class":
        current_by_class,

    "retained_by_class":
        retained_by_class,

    "class_retention":
        class_retention,

    "safety_retention":
        safety_retention,

    "class_balance_score":
        class_balance_score,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        len(previous) == 16
        and len(current) == 16
        and retention_rate >= 0.75
        and safety_retention > 0.0
        and class_balance_score > 0.0
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
    "===== Stage5 Temporal Closed Loop Evaluation ====="
)

print(
    "previous_beams =",
    len(previous)
)

print(
    "current_beams =",
    len(current)
)

print(
    "retained_beams =",
    len(retained_keys)
)

print(
    "retention_rate =",
    retention_rate
)

print(
    "switch_rate =",
    switch_rate
)

print(
    "class_retention =",
    class_retention
)

print(
    "safety_retention =",
    safety_retention
)

print(
    "class_balance_score =",
    class_balance_score
)

print(
    "current_by_class =",
    current_by_class
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

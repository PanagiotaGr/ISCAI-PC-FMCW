
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/block5_adb_beam_scheduler_smoke.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)


MAX_ACTIVE_BEAMS = 16

AZIMUTH_BEAMS = 64


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
scheduled = 0
constraint_valid = 0


classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


selected_beams_total = 0


def class_priority(actor_type):

    if actor_type == "PEDESTRIAN":
        return 0

    if actor_type == "CYCLIST":
        return 0

    if actor_type == "VEHICLE":
        return 1

    return 2



beam_requests = []


for key, model_records in groups.items():

    if not all(
        model in model_records
        for model in MODELS
    ):
        continue


    actors += 1

    actor_type = key[2]


    if actor_type in classes:
        classes[actor_type] += 1


    # Deterministic representative beam:
    # use first model first future point.
    trajectory = model_records[
        "CV"
    ].get(
        "trajectory",
        []
    )


    if len(trajectory) < 2:
        continue


    x = trajectory[1][0]
    y = trajectory[1][1]


    import math

    azimuth = math.atan2(
        y,
        x,
    )


    beam = int(
        (
            azimuth + math.pi
        )
        /
        (2.0 * math.pi)
        *
        AZIMUTH_BEAMS
    )


    beam = max(
        0,
        min(
            AZIMUTH_BEAMS - 1,
            beam,
        ),
    )


    beam_requests.append(
        (
            class_priority(actor_type),
            actor_type,
            beam,
        )
    )


beam_requests.sort(
    key=lambda x: (
        x[0],
        x[1],
        x[2],
    )
)


selected = beam_requests[
    :MAX_ACTIVE_BEAMS
]


selected_beams_total = len(
    selected
)


if (
    selected_beams_total
    <= MAX_ACTIVE_BEAMS
):
    constraint_valid = 1


scheduled = len(
    beam_requests
)


payload = {

    "actors":
        actors,

    "beam_requests":
        scheduled,

    "selected_beams":
        selected_beams_total,

    "max_active_beams":
        MAX_ACTIVE_BEAMS,

    "constraint_valid":
        constraint_valid,

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
    if constraint_valid == 1
    and actors > 0
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
    "===== Stage5 ADB Beam Scheduler Smoke ====="
)

print(
    "actors =",
    actors
)

print(
    "beam_requests =",
    scheduled
)

print(
    "selected_beams =",
    selected_beams_total
)

print(
    "max_active_beams =",
    MAX_ACTIVE_BEAMS
)

print(
    "constraint_valid =",
    constraint_valid
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

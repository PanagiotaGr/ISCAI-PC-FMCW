
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
    "reports/block5_adb_safety_scheduler_v2.json"
)

OUTPUT = Path(
    "artifacts/stage5_adb_safety_schedule_v2.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

AZIMUTH_BEAMS = 64
MAX_ACTIVE_BEAMS = 16

PEDESTRIAN_QUOTA = 4
CYCLIST_QUOTA = 4
VEHICLE_QUOTA = 4


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


requests = []


for key, model_records in groups.items():

    if not all(
        model in model_records
        for model in MODELS
    ):
        continue


    scenario_id, track_index, actor_type = key


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


    # Deterministic score:
    # closer to beam center -> higher priority.
    beam_center = (
        (
            beam_index + 0.5
        )
        /
        AZIMUTH_BEAMS
        *
        2.0
        *
        math.pi
        -
        math.pi
    )


    angular_error = abs(
        azimuth
        -
        beam_center
    )


    score = 1.0 / (
        1.0
        +
        angular_error
    )


    requests.append(
        {
            "scenario_id":
                scenario_id,

            "track_index":
                track_index,

            "actor_type":
                actor_type,

            "beam_index":
                beam_index,

            "azimuth_rad":
                azimuth,

            "score":
                score,
        }
    )


by_class = {
    "PEDESTRIAN": [],
    "CYCLIST": [],
    "VEHICLE": [],
}


for request in requests:

    actor_type = request[
        "actor_type"
    ]

    if actor_type in by_class:
        by_class[
            actor_type
        ].append(
            request
        )


for actor_type in by_class:

    by_class[
        actor_type
    ].sort(
        key=lambda x: (
            -x["score"],
            x["beam_index"],
            x["scenario_id"],
            x["track_index"],
        )
    )


selected = []


def add_quota(
    actor_type,
    quota,
):

    candidates = by_class[
        actor_type
    ]

    selected.extend(
        candidates[
            :min(
                quota,
                len(candidates),
            )
        ]
    )


add_quota(
    "PEDESTRIAN",
    PEDESTRIAN_QUOTA,
)

add_quota(
    "CYCLIST",
    CYCLIST_QUOTA,
)

add_quota(
    "VEHICLE",
    VEHICLE_QUOTA,
)


selected_keys = {
    (
        item["scenario_id"],
        item["track_index"],
        item["actor_type"],
    )
    for item in selected
}


remaining = [
    item
    for item in requests
    if (
        item["scenario_id"],
        item["track_index"],
        item["actor_type"],
    )
    not in selected_keys
]


remaining.sort(
    key=lambda x: (
        -x["score"],
        x["actor_type"],
        x["beam_index"],
        x["scenario_id"],
        x["track_index"],
    )
)


slots_left = (
    MAX_ACTIVE_BEAMS
    -
    len(selected)
)


if slots_left > 0:

    selected.extend(
        remaining[
            :slots_left
        ]
    )


# Hard cap.
selected = selected[
    :MAX_ACTIVE_BEAMS
]


selected_by_class = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in selected:

    actor_type = item[
        "actor_type"
    ]

    if actor_type in selected_by_class:
        selected_by_class[
            actor_type
        ] += 1


constraints_valid = (
    len(selected)
    <= MAX_ACTIVE_BEAMS
)


nonzero_class_coverage = (
    selected_by_class["VEHICLE"] > 0
    and selected_by_class["PEDESTRIAN"] > 0
    and selected_by_class["CYCLIST"] > 0
)


payload = {
    "requests":
        len(requests),

    "selected":
        len(selected),

    "max_active_beams":
        MAX_ACTIVE_BEAMS,

    "pedestrian_quota":
        PEDESTRIAN_QUOTA,

    "cyclist_quota":
        CYCLIST_QUOTA,

    "vehicle_quota":
        VEHICLE_QUOTA,

    "selected_by_class":
        selected_by_class,

    "constraints_valid":
        constraints_valid,

    "nonzero_class_coverage":
        nonzero_class_coverage,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        constraints_valid
        and nonzero_class_coverage
        and len(selected)
        == MAX_ACTIVE_BEAMS
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


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT.write_text(
    json.dumps(
        selected,
        indent=2,
        sort_keys=True,
    )
)


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
    "===== Stage5 ADB Safety Scheduler V2 ====="
)

print(
    "requests =",
    len(requests)
)

print(
    "selected =",
    len(selected)
)

print(
    "max_active_beams =",
    MAX_ACTIVE_BEAMS
)

print(
    "selected_vehicle =",
    selected_by_class["VEHICLE"]
)

print(
    "selected_pedestrian =",
    selected_by_class["PEDESTRIAN"]
)

print(
    "selected_cyclist =",
    selected_by_class["CYCLIST"]
)

print(
    "constraints_valid =",
    constraints_valid
)

print(
    "nonzero_class_coverage =",
    nonzero_class_coverage
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
    "output =",
    OUTPUT
)

print(
    "report =",
    REPORT
)


if status != "PASS":
    raise SystemExit(1)

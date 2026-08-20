
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
    "reports/block5_mht_gating_smoke.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

DT = 0.1

# Smoke-level deterministic gate.
# This is a hypothesis-consistency gate,
# not yet a statistical chi-square MHT gate.
GATE_THRESHOLD = 25.0


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
hypotheses_total = 0
hypotheses_valid = 0
gated_valid = 0
best_hypothesis_valid = 0

classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for key, model_records in groups.items():

    actors += 1

    scenario_id, track_index, actor_type = key

    if actor_type in classes:
        classes[actor_type] += 1


    if not all(
        model in model_records
        for model in MODELS
    ):
        continue

    complete_sets += 1


    hypotheses = {}


    for model in MODELS:

        trajectory = model_records[
            model
        ].get(
            "trajectory",
            [],
        )

        if len(trajectory) < 2:
            continue


        p0 = trajectory[0]
        p1 = trajectory[1]


        vx = (
            float(p1[0])
            -
            float(p0[0])
        ) / DT

        vy = (
            float(p1[1])
            -
            float(p0[1])
        ) / DT


        state = (
            float(p1[0]),
            float(p1[1]),
            vx,
            vy,
        )


        if all(
            math.isfinite(v)
            for v in state
        ):
            hypotheses[
                model
            ] = state

            hypotheses_total += 1
            hypotheses_valid += 1


    if len(hypotheses) != len(MODELS):
        continue


    # Consensus reference for smoke-level gating.
    # It is NOT ground truth.
    reference = tuple(
        sum(
            hypotheses[m][i]
            for m in MODELS
        ) / len(MODELS)
        for i in range(4)
    )


    scored = []


    for model in MODELS:

        state = hypotheses[
            model
        ]

        distance2 = sum(
            (
                state[i]
                -
                reference[i]
            ) ** 2
            for i in range(4)
        )


        if (
            math.isfinite(distance2)
            and distance2 <= GATE_THRESHOLD
        ):

            gated_valid += 1

            scored.append(
                (
                    distance2,
                    model,
                )
            )


    if not scored:
        continue


    scored.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )


    best_score, best_model = scored[0]


    if (
        best_model in MODELS
        and math.isfinite(best_score)
    ):
        best_hypothesis_valid += 1



payload = {

    "trajectory_records":
        len(data),

    "actors":
        actors,

    "models":
        list(MODELS),

    "complete_model_sets":
        complete_sets,

    "hypotheses_total":
        hypotheses_total,

    "hypotheses_valid":
        hypotheses_valid,

    "gated_hypotheses":
        gated_valid,

    "best_hypothesis_valid":
        best_hypothesis_valid,

    "gate_threshold":
        GATE_THRESHOLD,

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
        and hypotheses_valid == actors * len(MODELS)
        and best_hypothesis_valid == actors
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
    "===== Stage5 MHT Gating Smoke ====="
)

print(
    "trajectory_records =",
    len(data)
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
    "complete_model_sets =",
    complete_sets
)

print(
    "hypotheses_total =",
    hypotheses_total
)

print(
    "hypotheses_valid =",
    hypotheses_valid
)

print(
    "gated_hypotheses =",
    gated_valid
)

print(
    "best_hypothesis_valid =",
    best_hypothesis_valid
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

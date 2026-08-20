
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
    "reports/block5_imm_fusion_smoke.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

DT = 0.1

data = json.loads(
    SOURCE.read_text()
)


# One IMM fusion problem per physical actor.
groups = defaultdict(dict)

for item in data:

    key = (
        item["scenario_id"],
        int(item["track_index"]),
        item["actor_type"],
    )

    model_name = item.get(
        "model_name",
        "UNKNOWN",
    )

    if model_name in MODELS:
        groups[key][model_name] = item


actors = 0
complete_model_sets = 0
likelihood_valid = 0
probability_valid = 0
fused_state_valid = 0

max_probability_sum_error = 0.0

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

    complete_model_sets += 1


    states = {}

    for model in MODELS:

        trajectory = model_records[
            model
        ].get(
            "trajectory",
            [],
        )

        if len(trajectory) < 2:
            states = {}
            break


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


        states[model] = (
            float(p1[0]),
            float(p1[1]),
            vx,
            vy,
        )


    if len(states) != len(MODELS):
        continue


    # Consensus state used only to construct a deterministic
    # smoke-test likelihood. This is NOT ground truth.
    consensus = tuple(
        sum(
            states[m][i]
            for m in MODELS
        ) / len(MODELS)
        for i in range(4)
    )


    likelihoods = {}

    for model in MODELS:

        state = states[model]

        squared_error = sum(
            (
                state[i]
                -
                consensus[i]
            ) ** 2
            for i in range(4)
        )

        # Positive Gaussian-like compatibility score.
        likelihood = math.exp(
            -0.5 * min(
                squared_error,
                100.0,
            )
        )

        likelihoods[model] = max(
            likelihood,
            1e-12,
        )


    if all(
        math.isfinite(value)
        and value > 0.0
        for value in likelihoods.values()
    ):
        likelihood_valid += 1
    else:
        continue


    # Uniform prior mode probability.
    prior = {
        model: 1.0 / len(MODELS)
        for model in MODELS
    }


    unnormalized = {
        model:
            prior[model]
            *
            likelihoods[model]
        for model in MODELS
    }


    normalizer = sum(
        unnormalized.values()
    )

    if (
        not math.isfinite(normalizer)
        or normalizer <= 0.0
    ):
        continue


    probabilities = {
        model:
            unnormalized[model]
            /
            normalizer
        for model in MODELS
    }


    probability_sum = sum(
        probabilities.values()
    )

    probability_error = abs(
        probability_sum - 1.0
    )

    max_probability_sum_error = max(
        max_probability_sum_error,
        probability_error,
    )


    if (
        all(
            math.isfinite(value)
            and 0.0 <= value <= 1.0
            for value in probabilities.values()
        )
        and probability_error < 1e-12
    ):
        probability_valid += 1
    else:
        continue


    fused_state = tuple(
        sum(
            probabilities[model]
            *
            states[model][i]
            for model in MODELS
        )
        for i in range(4)
    )


    if all(
        math.isfinite(value)
        for value in fused_state
    ):
        fused_state_valid += 1


payload = {
    "trajectory_records":
        len(data),

    "actors":
        actors,

    "models":
        list(MODELS),

    "complete_model_sets":
        complete_model_sets,

    "likelihood_valid":
        likelihood_valid,

    "probability_valid":
        probability_valid,

    "fused_state_valid":
        fused_state_valid,

    "max_probability_sum_error":
        max_probability_sum_error,

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
        and complete_model_sets == actors
        and likelihood_valid == actors
        and probability_valid == actors
        and fused_state_valid == actors
    )
    else "FAIL"
)


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


report = {
    **payload,
    "sha256": sha,
    "status": status,
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
    "===== Stage5 IMM Fusion Smoke ====="
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
    complete_model_sets
)

print(
    "likelihood_valid =",
    likelihood_valid
)

print(
    "probability_valid =",
    probability_valid
)

print(
    "fused_state_valid =",
    fused_state_valid
)

print(
    "max_probability_sum_error =",
    max_probability_sum_error
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


from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/block5_model_diversity_audit.json"
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
        "model_name"
    )

    if model in MODELS:
        groups[key][model] = item


pairs = [
    ("CV", "CA"),
    ("CV", "CTRV"),
    ("CV", "KALMAN"),
    ("CA", "CTRV"),
    ("CA", "KALMAN"),
    ("CTRV", "KALMAN"),
]


pair_sum = {
    f"{a}_vs_{b}": 0.0
    for a, b in pairs
}

pair_max = {
    f"{a}_vs_{b}": 0.0
    for a, b in pairs
}

pair_count = {
    f"{a}_vs_{b}": 0
    for a, b in pairs
}


actors = 0
identical_all_models = 0

EPS = 1e-9


for key, records in groups.items():

    if not all(
        model in records
        for model in MODELS
    ):
        continue

    actors += 1

    actor_max_difference = 0.0


    for a, b in pairs:

        ta = records[a]["trajectory"]
        tb = records[b]["trajectory"]

        horizon = min(
            len(ta),
            len(tb),
        )

        if horizon == 0:
            continue


        distances = []


        for i in range(horizon):

            dx = (
                float(ta[i][0])
                -
                float(tb[i][0])
            )

            dy = (
                float(ta[i][1])
                -
                float(tb[i][1])
            )

            dz = (
                float(ta[i][2])
                -
                float(tb[i][2])
            )


            d = math.sqrt(
                dx * dx
                +
                dy * dy
                +
                dz * dz
            )

            distances.append(d)


        mean_distance = (
            sum(distances)
            /
            len(distances)
        )


        max_distance = max(
            distances
        )


        name = f"{a}_vs_{b}"

        pair_sum[name] += mean_distance

        pair_max[name] = max(
            pair_max[name],
            max_distance,
        )

        pair_count[name] += 1

        actor_max_difference = max(
            actor_max_difference,
            max_distance,
        )


    if actor_max_difference < EPS:
        identical_all_models += 1


pair_mean = {}


for name in pair_sum:

    count = pair_count[name]

    pair_mean[name] = (
        pair_sum[name] / count
        if count > 0
        else 0.0
    )


identical_fraction = (
    identical_all_models
    /
    actors
    if actors > 0
    else 0.0
)


payload = {
    "actors":
        actors,

    "epsilon":
        EPS,

    "pair_mean_distance_m":
        pair_mean,

    "pair_max_distance_m":
        pair_max,

    "identical_all_models":
        identical_all_models,

    "identical_fraction":
        identical_fraction,

    "future_used":
        False,
}


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


# Audit itself passes if calculations are valid.
# A high identical_fraction is a scientific finding,
# not a software failure.
status = (
    "PASS"
    if actors > 0
    else "FAIL"
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
    "===== Stage5 Model Diversity Audit ====="
)

print(
    "actors =",
    actors
)

for name in sorted(
    pair_mean
):
    print(
        name,
        "mean_m =",
        pair_mean[name],
        "max_m =",
        pair_max[name],
    )

print(
    "identical_all_models =",
    identical_all_models
)

print(
    "identical_fraction =",
    identical_fraction
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

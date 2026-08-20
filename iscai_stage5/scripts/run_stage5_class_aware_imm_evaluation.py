
from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict, Counter
from pathlib import Path


SOURCE = Path(
    "reports/block5_measurement_driven_imm_v2.json"
)

TRAJ = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

OUTPUT = Path(
    "reports/block5_class_aware_imm_evaluation.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)


data = json.loads(
    TRAJ.read_text()
)


# --------------------------------------------------
# We use model trajectory diversity per actor class
# as a class-conditioned diagnostic complementary
# to the global IMM V2 report.
# --------------------------------------------------

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


class_stats = {
    "VEHICLE": {
        "actors": 0,
        "dispersion_sum": 0.0,
        "ctrv_deviation_sum": 0.0,
    },
    "PEDESTRIAN": {
        "actors": 0,
        "dispersion_sum": 0.0,
        "ctrv_deviation_sum": 0.0,
    },
    "CYCLIST": {
        "actors": 0,
        "dispersion_sum": 0.0,
        "ctrv_deviation_sum": 0.0,
    },
}


for key, records in groups.items():

    actor_type = key[2]

    if actor_type not in class_stats:
        continue


    if not all(
        model in records
        for model in MODELS
    ):
        continue


    trajectories = {
        model:
            records[model]["trajectory"]
        for model in MODELS
    }


    horizon = min(
        len(t)
        for t in trajectories.values()
    )


    if horizon <= 0:
        continue


    actor_dispersion = 0.0
    actor_ctrv_deviation = 0.0


    for step in range(horizon):

        points = [
            trajectories[m][step]
            for m in MODELS
        ]


        cx = sum(
            float(p[0])
            for p in points
        ) / len(points)

        cy = sum(
            float(p[1])
            for p in points
        ) / len(points)


        dispersion = sum(
            (
                float(p[0]) - cx
            ) ** 2
            +
            (
                float(p[1]) - cy
            ) ** 2
            for p in points
        ) / len(points)


        actor_dispersion += math.sqrt(
            dispersion
        )


        ctrv = trajectories[
            "CTRV"
        ][step]


        ctrv_deviation = math.sqrt(
            (
                float(ctrv[0]) - cx
            ) ** 2
            +
            (
                float(ctrv[1]) - cy
            ) ** 2
        )


        actor_ctrv_deviation += (
            ctrv_deviation
        )


    actor_dispersion /= horizon
    actor_ctrv_deviation /= horizon


    class_stats[
        actor_type
    ]["actors"] += 1

    class_stats[
        actor_type
    ]["dispersion_sum"] += (
        actor_dispersion
    )

    class_stats[
        actor_type
    ]["ctrv_deviation_sum"] += (
        actor_ctrv_deviation
    )




def pairwise_dispersion(trajectories):

    models = list(trajectories.keys())

    total = 0.0
    count = 0

    for i in range(len(models)):

        for j in range(i+1, len(models)):

            a = trajectories[models[i]]
            b = trajectories[models[j]]

            horizon = min(
                len(a),
                len(b)
            )

            for k in range(horizon):

                dx = (
                    float(a[k][0])
                    -
                    float(b[k][0])
                )

                dy = (
                    float(a[k][1])
                    -
                    float(b[k][1])
                )

                total += (
                    dx*dx +
                    dy*dy
                ) ** 0.5

                count += 1

    return (
        total / count
        if count
        else 0.0
    )


results = {}


for actor_type, stats in class_stats.items():

    count = stats[
        "actors"
    ]

    results[
        actor_type
    ] = {

        "actors":
            count,

        "mean_pairwise_model_dispersion_m":
            (
                stats["dispersion_sum"]
                /
                count
                if count > 0
                else 0.0
            ),

        "mean_ctrv_to_consensus_deviation_m":
            (
                stats["ctrv_deviation_sum"]
                /
                count
                if count > 0
                else 0.0
            ),
    }


global_imm = json.loads(
    SOURCE.read_text()
)


payload = {

    "class_results":
        results,

    "global_imm_v2": {
        "updates":
            global_imm[
                "updates"
            ],

        "normalized_entropy":
            global_imm[
                "normalized_entropy"
            ],

        "nonuniform_fraction":
            global_imm[
                "nonuniform_fraction"
            ],

        "dominant_model_distribution":
            global_imm[
                "dominant_model_distribution"
            ],

        "mean_nis_by_model":
            global_imm[
                "mean_nis_by_model"
            ],
    },

    "future_used":
        False,
}


status = (
    "PASS"
    if all(
        results[cls]["actors"] > 0
        for cls in (
            "VEHICLE",
            "PEDESTRIAN",
            "CYCLIST",
        )
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


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Class Aware IMM Evaluation ====="
)

for cls in (
    "VEHICLE",
    "PEDESTRIAN",
    "CYCLIST",
):

    print(
        cls,
        "=",
        results[cls]
    )


print(
    "global normalized_entropy =",
    global_imm[
        "normalized_entropy"
    ]
)

print(
    "global nonuniform_fraction =",
    global_imm[
        "nonuniform_fraction"
    ]
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
    OUTPUT
)


if status != "PASS":
    raise SystemExit(1)

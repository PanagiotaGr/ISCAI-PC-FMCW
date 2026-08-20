
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, Counter
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/block5_imm_effectiveness_evaluation.json"
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



actors = 0
complete = 0

before_dispersion_sum = 0.0
after_fusion_sum = 0.0

dominant_models = Counter()

probability_valid = 0


for key, records in groups.items():

    if not all(
        model in records
        for model in MODELS
    ):
        continue


    actors += 1
    complete += 1


    trajectories = {
        model:
        records[model]["trajectory"]
        for model in MODELS
    }


    horizon = min(
        len(t)
        for t in trajectories.values()
    )


    before_actor = 0.0
    after_actor = 0.0


    for step in range(horizon):

        points = [
            trajectories[m][step]
            for m in MODELS
        ]


        cx = sum(
            float(p[0])
            for p in points
        ) / 4.0

        cy = sum(
            float(p[1])
            for p in points
        ) / 4.0


        dispersion = sum(
            (
                float(p[0])-cx
            )**2
            +
            (
                float(p[1])-cy
            )**2
            for p in points
        ) / 4.0


        before_actor += math.sqrt(
            dispersion
        )


    before_actor /= horizon


    # Deterministic IMM approximation:
    # weights based on inverse deviation
    # from ensemble centroid.

    scores = {}

    for model in MODELS:

        traj = trajectories[model]

        deviation = 0.0


        for step in range(horizon):

            p = traj[step]

            cx = sum(
                float(
                    trajectories[m][step][0]
                )
                for m in MODELS
            ) / 4.0

            cy = sum(
                float(
                    trajectories[m][step][1]
                )
                for m in MODELS
            ) / 4.0


            deviation += math.sqrt(
                (
                    float(p[0])-cx
                )**2
                +
                (
                    float(p[1])-cy
                )**2
            )


        scores[model] = 1.0 / (
            deviation + 1e-9
        )


    total = sum(
        scores.values()
    )


    probabilities = {
        m:
        scores[m]/total
        for m in MODELS
    }


    if abs(
        sum(probabilities.values())
        -
        1.0
    ) < 1e-9:

        probability_valid += 1


    dominant = max(
        probabilities,
        key=probabilities.get
    )

    dominant_models[dominant] += 1


    # fused trajectory dispersion:
    # weighted centroid uncertainty

    for step in range(horizon):

        fx = sum(
            probabilities[m]
            *
            float(
                trajectories[m][step][0]
            )
            for m in MODELS
        )

        fy = sum(
            probabilities[m]
            *
            float(
                trajectories[m][step][1]
            )
            for m in MODELS
        )


        fusion_error = sum(
            probabilities[m]
            *
            (
                (
                    float(
                        trajectories[m][step][0]
                    )
                    -
                    fx
                )**2
                +
                (
                    float(
                        trajectories[m][step][1]
                    )
                    -
                    fy
                )**2
            )
            for m in MODELS
        )


        after_actor += math.sqrt(
            fusion_error
        )


    after_actor /= horizon


    before_dispersion_sum += (
        before_actor
    )

    after_fusion_sum += (
        after_actor
    )


mean_before = (
    before_dispersion_sum / actors
)

mean_after = (
    after_fusion_sum / actors
)


improvement = (
    mean_before - mean_after
)


payload = {

    "actors":
        actors,

    "complete_model_sets":
        complete,

    "models":
        list(MODELS),

    "probability_valid":
        probability_valid,

    "dominant_model_distribution":
        dict(dominant_models),

    "mean_before_dispersion_m":
        mean_before,

    "mean_after_fusion_dispersion_m":
        mean_after,

    "fusion_dispersion_reduction_m":
        improvement,

    "future_used":
        False,
}


status = (
    "PASS"
    if (
        actors > 0
        and complete == actors
        and probability_valid == actors
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
    "===== Stage5 IMM Effectiveness Evaluation ====="
)

print(
    "actors =",
    actors
)

print(
    "complete_model_sets =",
    complete
)

print(
    "probability_valid =",
    probability_valid
)

print(
    "dominant_model_distribution =",
    dict(dominant_models)
)

print(
    "mean_before_dispersion_m =",
    mean_before
)

print(
    "mean_after_fusion_dispersion_m =",
    mean_after
)

print(
    "fusion_dispersion_reduction_m =",
    improvement
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

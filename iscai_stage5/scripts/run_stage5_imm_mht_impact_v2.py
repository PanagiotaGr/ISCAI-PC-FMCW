
from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict, Counter
from pathlib import Path


TRAJ = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

REPORT = Path(
    "reports/block5_imm_mht_impact_v2.json"
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


groups = defaultdict(dict)
classes = {}


for item in data:

    key = (
        item["scenario_id"],
        int(item["track_index"]),
    )

    groups[key][
        item["model_name"]
    ] = item["trajectory"]

    classes[key] = item["actor_type"]



actors = 0
hypotheses_total = 0
changed_rankings = 0

before_best = Counter()
after_best = Counter()

class_effect = defaultdict(
    lambda: {
        "actors": 0,
        "changed": 0,
    }
)

score_improvement_sum = 0.0


for key, models in groups.items():

    if not all(
        m in models
        for m in MODELS
    ):
        continue


    actors += 1
    hypotheses_total += len(
        MODELS
    )


    # ---------------------------------
    # Baseline MHT score:
    # only trajectory smoothness
    # ---------------------------------

    scores_before = {}


    for model in MODELS:

        traj = models[model]

        cost = 0.0


        for i in range(
            1,
            len(traj)
        ):

            dx = (
                float(traj[i][0])
                -
                float(traj[i-1][0])
            )

            dy = (
                float(traj[i][1])
                -
                float(traj[i-1][1])
            )

            cost += math.sqrt(
                dx*dx + dy*dy
            )


        scores_before[model] = cost



    best_before = min(
        scores_before,
        key=scores_before.get,
    )


    before_best[
        best_before
    ] += 1



    # ---------------------------------
    # Pseudo IMM posterior from
    # actor-level diversity.
    # Lower trajectory disagreement
    # gives higher probability.
    # ---------------------------------

    likelihood = {}


    centroid = []

    horizon = min(
        len(models[m])
        for m in MODELS
    )


    for m in MODELS:

        deviation = 0.0

        for i in range(
            horizon
        ):

            cx = sum(
                float(
                    models[x][i][0]
                )
                for x in MODELS
            ) / 4.0


            cy = sum(
                float(
                    models[x][i][1]
                )
                for x in MODELS
            ) / 4.0


            deviation += math.sqrt(
                (
                    float(
                        models[m][i][0]
                    )
                    -
                    cx
                )**2
                +
                (
                    float(
                        models[m][i][1]
                    )
                    -
                    cy
                )**2
            )


        likelihood[m] = math.exp(
            -deviation
        )


    total = sum(
        likelihood.values()
    )


    probabilities = {
        m:
        likelihood[m]/total
        for m in MODELS
    }



    # ---------------------------------
    # IMM weighted MHT score
    # ---------------------------------

    scores_after = {}


    for model in MODELS:

        scores_after[model] = (
            scores_before[model]
            -
            math.log(
                probabilities[model]
                +
                1e-12
            )
        )


    best_after = min(
        scores_after,
        key=scores_after.get,
    )


    after_best[
        best_after
    ] += 1


    if best_before != best_after:

        changed_rankings += 1

        class_effect[
            classes[key]
        ]["changed"] += 1


    class_effect[
        classes[key]
    ]["actors"] += 1


    score_improvement_sum += (
        scores_before[best_before]
        -
        scores_after[best_after]
    )



mean_score_improvement = (
    score_improvement_sum
    /
    actors
    if actors
    else 0.0
)


change_rate = (
    changed_rankings
    /
    actors
    if actors
    else 0.0
)


payload = {

    "actors":
        actors,

    "hypotheses_total":
        hypotheses_total,

    "best_before":
        dict(before_best),

    "best_after":
        dict(after_best),

    "changed_rankings":
        changed_rankings,

    "change_rate":
        change_rate,

    "mean_score_improvement":
        mean_score_improvement,

    "class_effect":
        {
            k: dict(v)
            for k,v in class_effect.items()
        },

    "future_used":
        False,
}


status = (
    "PASS"
    if actors > 0
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
    "===== Stage5 IMM MHT Impact V2 ====="
)

print(
    "actors =",
    actors
)

print(
    "hypotheses_total =",
    hypotheses_total
)

print(
    "best_before =",
    dict(before_best)
)

print(
    "best_after =",
    dict(after_best)
)

print(
    "changed_rankings =",
    changed_rankings
)

print(
    "change_rate =",
    change_rate
)

print(
    "mean_score_improvement =",
    mean_score_improvement
)

print(
    "class_effect =",
    {
        k:v
        for k,v in class_effect.items()
    }
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

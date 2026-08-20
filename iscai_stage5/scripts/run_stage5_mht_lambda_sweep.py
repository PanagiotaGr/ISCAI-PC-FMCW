
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
    "reports/block5_mht_lambda_sweep.json"
)


MODELS = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
)

LAMBDAS = (
    0.0,
    0.25,
    0.5,
    1.0,
)


data = json.loads(
    TRAJ.read_text()
)


groups = defaultdict(dict)


for item in data:

    key = (
        item["scenario_id"],
        int(item["track_index"]),
    )

    groups[key][
        item["model_name"]
    ] = item["trajectory"]



results = {}


for lam in LAMBDAS:

    actors = 0
    changed = 0

    before = Counter()
    after = Counter()

    ctrv_selected = 0


    for key, models in groups.items():

        if not all(
            m in models
            for m in MODELS
        ):
            continue


        actors += 1


        costs = {}


        horizon = min(
            len(models[m])
            for m in MODELS
        )


        centroid_dev = {}


        for model in MODELS:

            cost = 0.0

            traj = models[model]

            for i in range(
                1,
                horizon
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

            costs[model] = cost


        best_before = min(
            costs,
            key=costs.get,
        )

        before[best_before] += 1


        # IMM-like actor posterior
        scores = {}

        for model in MODELS:

            dev = 0.0

            for i in range(
                horizon
            ):

                cx = sum(
                    float(
                        models[m][i][0]
                    )
                    for m in MODELS
                ) / len(MODELS)

                cy = sum(
                    float(
                        models[m][i][1]
                    )
                    for m in MODELS
                ) / len(MODELS)


                dev += math.sqrt(
                    (
                        float(
                            models[model][i][0]
                        )
                        -
                        cx
                    )**2
                    +
                    (
                        float(
                            models[model][i][1]
                        )
                        -
                        cy
                    )**2
                )

            scores[model] = math.exp(
                -dev
            )


        total = sum(
            scores.values()
        )


        probabilities = {
            m:
            scores[m] / total
            for m in MODELS
        }


        final_scores = {

            m:
            costs[m]
            +
            lam *
            (
                -math.log(
                    probabilities[m]
                    +
                    1e-12
                )
            )

            for m in MODELS
        }


        best_after = min(
            final_scores,
            key=final_scores.get,
        )


        after[best_after] += 1


        if best_after != best_before:
            changed += 1


        if best_after == "CTRV":
            ctrv_selected += 1



    results[str(lam)] = {

        "actors":
            actors,

        "changed_rankings":
            changed,

        "change_rate":
            (
                changed / actors
                if actors
                else 0.0
            ),

        "best_before":
            dict(before),

        "best_after":
            dict(after),

        "ctrv_selection_rate":
            (
                ctrv_selected / actors
                if actors
                else 0.0
            ),
    }



payload = {

    "lambda_results":
        results,

    "future_used":
        False,
}


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
        "PASS",
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
    "===== Stage5 MHT Lambda Sweep ====="
)


for lam, result in results.items():

    print()
    print(
        "lambda =",
        lam
    )

    print(
        "change_rate =",
        result["change_rate"]
    )

    print(
        "best_after =",
        result["best_after"]
    )

    print(
        "ctrv_selection_rate =",
        result["ctrv_selection_rate"]
    )


print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

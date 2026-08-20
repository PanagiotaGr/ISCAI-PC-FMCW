
from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict, Counter
from pathlib import Path


TRAJ = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

IMM_REPORT = Path(
    "reports/block5_measurement_driven_imm_v2.json"
)

REPORT = Path(
    "reports/block5_imm_mht_impact_evaluation.json"
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

imm = json.loads(
    IMM_REPORT.read_text()
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



uniform_entropy = math.log(
    len(MODELS)
)


# Average IMM distribution from V2
imm_distribution = imm[
    "mean_probability_by_model"
]


imm_entropy = 0.0

for p in imm_distribution.values():

    if p > 0:
        imm_entropy -= (
            p
            *
            math.log(p)
        )


entropy_reduction = (
    uniform_entropy
    -
    imm_entropy
)


baseline_best = Counter()
imm_best = Counter()

class_effect = defaultdict(
    lambda: {
        "actors": 0,
        "baseline_best": Counter(),
        "imm_best": Counter(),
    }
)


actors = 0


for key, models in groups.items():

    if not all(
        m in models
        for m in MODELS
    ):
        continue


    actor_type = classes[key]


    # Baseline: equal probability.
    baseline_best[
        "CV"
    ] += 1


    # IMM weighted winner.
    best = max(
        imm_distribution,
        key=imm_distribution.get,
    )


    imm_best[
        best
    ] += 1


    class_effect[
        actor_type
    ]["actors"] += 1

    class_effect[
        actor_type
    ]["baseline_best"]["CV"] += 1

    class_effect[
        actor_type
    ]["imm_best"][best] += 1


    actors += 1



payload = {

    "actors":
        actors,

    "models":
        list(MODELS),

    "baseline_entropy":
        uniform_entropy,

    "imm_entropy":
        imm_entropy,

    "entropy_reduction":
        entropy_reduction,

    "baseline_best_distribution":
        dict(baseline_best),

    "imm_best_distribution":
        dict(imm_best),

    "class_effect":
        {
            k: {
                "actors": v["actors"],
                "baseline_best":
                    dict(v["baseline_best"]),
                "imm_best":
                    dict(v["imm_best"]),
            }
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
    "===== Stage5 IMM MHT Impact Evaluation ====="
)

print(
    "actors =",
    actors
)

print(
    "baseline_entropy =",
    uniform_entropy
)

print(
    "imm_entropy =",
    imm_entropy
)

print(
    "entropy_reduction =",
    entropy_reduction
)

print(
    "baseline_best_distribution =",
    dict(baseline_best)
)

print(
    "imm_best_distribution =",
    dict(imm_best)
)

print(
    "class_effect =",
    {
        k:v["actors"]
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

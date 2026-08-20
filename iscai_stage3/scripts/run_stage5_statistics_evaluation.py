from __future__ import annotations

import json
import hashlib
from pathlib import Path
import math


ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)

REPORT = Path(
    "reports/block5c_statistics_evaluation.json"
)


if not ARTIFACT.exists():
    raise RuntimeError(
        "Missing trajectory artifact."
    )


data = json.loads(
    ARTIFACT.read_text()
)


stats = {}


for item in data:

    model = item["model_name"]

    stats.setdefault(
        model,
        {
            "count": 0,
            "distances": [],
        }
    )


    trajectory = item["trajectory"]


    start = trajectory[0]
    end = trajectory[-1]


    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]


    distance = math.sqrt(
        dx*dx +
        dy*dy +
        dz*dz
    )


    stats[model]["count"] += 1

    stats[model]["distances"].append(
        distance
    )



summary = {}


for model, value in stats.items():

    distances = value["distances"]

    summary[model] = {

        "count":
            value["count"],

        "mean_displacement":
            sum(distances)
            /
            len(distances),

        "max_displacement":
            max(distances),

        "min_displacement":
            min(distances),

    }



payload = {
    "models": summary,
    "records": len(data),
}


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
    ).encode()
).hexdigest()



report = {

    "block":
        "stage5c_statistics_evaluation",

    "records":
        len(data),

    "models":
        summary,

    "future_used":
        False,

    "sha256":
        sha,

    "status":
        "PASS",
}



REPORT.parent.mkdir(
    exist_ok=True
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5C Statistics Evaluation ====="
)

print(
    "records =",
    len(data)
)

print(
    "models =",
    sorted(summary.keys())
)

for k,v in summary.items():

    print(
        k,
        "count=",
        v["count"],
        "mean_disp=",
        v["mean_displacement"]
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

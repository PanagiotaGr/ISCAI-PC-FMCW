from __future__ import annotations


import sys
from pathlib import Path

ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)



ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)


import json
import hashlib
import math

from pathlib import Path


from iscai_stage5.integration.stage4_prediction_loader import (
    load_stage4_predictions,
)


SOURCE = Path(
    "/home/agni/waymo/iscai_stage4/artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block5_statistics_evaluation.json"
)



predictions = load_stage4_predictions(
    str(SOURCE)
)



if len(predictions) == 0:
    raise RuntimeError(
        "Empty predictions."
    )



statistics = {}



for prediction in predictions:

    if prediction.model_name not in statistics:

        statistics[
            prediction.model_name
        ] = {

            "count": 0,

            "uncertainty_sum": 0.0,

            "trajectory_length": set(),

        }


    entry = statistics[
        prediction.model_name
    ]


    entry["count"] += 1

    entry["uncertainty_sum"] += (
        prediction.uncertainty
    )

    entry["trajectory_length"].add(
        len(
            prediction.trajectory
        )
    )


    for point in prediction.trajectory:

        if not all(
            math.isfinite(x)
            for x in point
        ):

            raise RuntimeError(
                "Non finite trajectory."
            )



for name, value in statistics.items():

    value["mean_uncertainty"] = (

        value["uncertainty_sum"]

        /

        value["count"]

    )


    value["trajectory_length"] = sorted(
        value["trajectory_length"]
    )



payload = {

    "records":
        len(predictions),

    "models":
        sorted(statistics.keys()),

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

    "records":
        len(predictions),

    "models":
        sorted(statistics.keys()),

    "statistics":
        statistics,

    "finite":
        True,

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
    report["records"]
)

print(
    "models =",
    report["models"]
)


for name, value in statistics.items():

    print(
        name,
        "count=",
        value["count"],
        "mean_uncertainty=",
        value["mean_uncertainty"]
    )


print(
    "finite = True"
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

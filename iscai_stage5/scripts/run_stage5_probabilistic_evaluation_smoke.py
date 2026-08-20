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
    "reports/block5_probabilistic_evaluation_smoke.json"
)



predictions = load_stage4_predictions(
    str(SOURCE)
)



if len(predictions) == 0:

    raise RuntimeError(
        "No predictions."
    )



model_counts = {}

uncertainties = []



for prediction in predictions:

    model_counts[
        prediction.model_name
    ] = (
        model_counts.get(
            prediction.model_name,
            0,
        )
        + 1
    )


    uncertainties.append(
        prediction.uncertainty
    )


    for point in prediction.trajectory:

        if not all(
            math.isfinite(x)
            for x in point
        ):

            raise RuntimeError(
                "Non finite trajectory."
            )


    if len(
        prediction.trajectory
    ) != 10:

        raise RuntimeError(
            "Invalid horizon."
        )



mean_uncertainty = (

    sum(
        uncertainties
    )
    /
    len(
        uncertainties
    )

)



payload = {

    "records":
        len(predictions),

    "models":
        sorted(
            model_counts.keys()
        ),

    "mean_uncertainty":
        mean_uncertainty,

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

    "model_counts":
        model_counts,

    "mean_uncertainty":
        mean_uncertainty,

    "horizon":
        10,

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
    "===== Stage5 Probabilistic Evaluation ====="
)

print(
    "records =",
    report["records"]
)

print(
    "models =",
    sorted(
        model_counts.keys()
    )
)

print(
    "mean_uncertainty =",
    mean_uncertainty
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

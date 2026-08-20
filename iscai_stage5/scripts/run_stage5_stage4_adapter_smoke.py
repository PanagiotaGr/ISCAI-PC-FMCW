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

from pathlib import Path


from iscai_stage5.integration.stage4_prediction_loader import (
    load_stage4_predictions,
)



SOURCE = Path(
    "/home/agni/waymo/iscai_stage4/artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block5_stage4_adapter_smoke.json"
)



if not SOURCE.exists():

    raise RuntimeError(
        "Missing Stage4 artifact."
    )



predictions = load_stage4_predictions(
    str(SOURCE)
)



if len(predictions) == 0:

    raise RuntimeError(
        "No predictions loaded."
    )



models = sorted(
    {
        x.model_name
        for x in predictions
    }
)



for prediction in predictions:

    if len(
        prediction.trajectory
    ) != 10:

        raise RuntimeError(
            "Invalid horizon."
        )


    if not (
        0.0 <= prediction.uncertainty <= 1.0
    ):

        raise RuntimeError(
            "Invalid uncertainty."
        )



payload = {

    "records":
        len(predictions),

    "models":
        models,

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
        models,

    "horizon":
        10,

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
    "===== Stage5 Stage4 Adapter Smoke ====="
)

print(
    "records =",
    report["records"]
)

print(
    "models =",
    report["models"]
)

print(
    "horizon =",
    report["horizon"]
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

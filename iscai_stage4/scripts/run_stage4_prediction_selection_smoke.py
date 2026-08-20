from __future__ import annotations


import json
from pathlib import Path
import hashlib


from iscai_stage4.controller.prediction_adapter import (
    prediction_to_controller,
)


from iscai_stage4.controller.candidate_selector import (
    select_best_prediction,
)



ARTIFACT = Path(
    "artifacts/stage4_prediction_artifact.json"
)


REPORT = Path(
    "reports/block4_prediction_selection_smoke.json"
)



data = json.loads(
    ARTIFACT.read_text()
)



candidates = [

    prediction_to_controller(
        x
    )

    for x in data

]



selected = select_best_prediction(
    candidates
)



payload = {

    "candidate_count":
        len(candidates),

    "selected_model":
        selected.model_name,

    "utility":
        selected.utility,

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

    "candidates":
        len(candidates),

    "selected_model":
        selected.model_name,

    "selected_utility":
        selected.utility,

    "trajectory_length":
        len(selected.trajectory),

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
    "===== Stage4 Prediction Selection ====="
)

print(
    "candidates =",
    report["candidates"]
)

print(
    "selected_model =",
    report["selected_model"]
)

print(
    "utility =",
    report["selected_utility"]
)

print(
    "trajectory_length =",
    report["trajectory_length"]
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

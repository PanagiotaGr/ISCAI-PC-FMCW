from __future__ import annotations


import json
import hashlib

from pathlib import Path


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
    "reports/block4_closed_loop_replay_smoke.json"
)



data = json.loads(
    ARTIFACT.read_text()
)



steps = 5

actions = []



for time_index in range(
    steps
):


    candidates = []


    for item in data:


        if item["future_used"]:

            raise RuntimeError(
                "Future leakage."
            )


        candidates.append(

            prediction_to_controller(
                item
            )

        )


    selected = select_best_prediction(
        candidates
    )


    actions.append(

        {

            "time_index":
                time_index,


            "model":
                selected.model_name,


            "utility":
                selected.utility,


            "trajectory_length":
                len(
                    selected.trajectory
                ),

        }

    )



for action in actions:


    if action[
        "trajectory_length"
    ] != 10:

        raise RuntimeError(
            "Invalid replay trajectory."
        )



payload = {

    "steps":
        steps,

    "actions":
        actions,

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

    "steps":
        steps,

    "actions":
        len(actions),

    "first_action":
        actions[0],


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
    "===== Stage4 Closed Loop Replay ====="
)

print(
    "steps =",
    report["steps"]
)

print(
    "actions =",
    report["actions"]
)

print(
    "first_action =",
    report["first_action"]
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

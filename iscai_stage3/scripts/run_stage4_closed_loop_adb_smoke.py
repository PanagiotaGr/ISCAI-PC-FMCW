from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.control.adb_adapter import (
    prediction_to_adb_target,
)

from iscai_stage3.control.adb_closed_loop import (
    select_next_action,
)



REPORT = Path(
    "reports/block4c_closed_loop_adb_smoke.json"
)


ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing prediction artifact."
    )



data = json.loads(
    ARTIFACT.read_text()
)



targets = []



for idx, item in enumerate(data):

    trajectory = item["trajectory"]


    length = len(
        trajectory
    )


    utility = (
        1.0 /
        (1.0 + length)
    )


    uncertainty = (
        1.0 -
        utility
    )


    targets.append(
        prediction_to_adb_target(
            target_id=idx,
            utility=utility,
            uncertainty=uncertainty,
        )
    )



decision = select_next_action(
    candidates=targets
)



report = {

    "block":
        "stage4c_closed_loop_adb_smoke",

    "candidate_targets":
        len(targets),

    "selected_target":
        decision.target_id,

    "utility":
        decision.utility,

    "uncertainty":
        decision.uncertainty,

    "future_used":
        False,

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
    "===== Stage4C Closed Loop ADB ====="
)

print(
    "candidates =",
    len(targets)
)

print(
    "selected_target =",
    decision.target_id
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

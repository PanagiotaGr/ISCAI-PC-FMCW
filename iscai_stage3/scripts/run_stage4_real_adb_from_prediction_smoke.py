from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.control.adb_adapter import (
    prediction_to_adb_target,
    build_adb_schedule,
)

from iscai_stage3.control.adb_utility import (
    target_utility,
)



REPORT = Path(
    "reports/block4b_real_adb_from_prediction_smoke.json"
)



TRAJECTORY_ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)



if not TRAJECTORY_ARTIFACT.exists():
    raise RuntimeError(
        "Missing prediction report."
    )



trajectory_data = json.loads(
    TRAJECTORY_ARTIFACT.read_text()
)



targets = []



target_id = 0




for item in trajectory_data:

    if item["future_used"]:
        raise RuntimeError(
            "Future leakage."
        )


    trajectory = tuple(
        tuple(point)
        for point in item["trajectory"]
    )


    utility = target_utility(
        trajectory=trajectory,
        confidence=0.9,
    )


    uncertainty = (
        1.0 -
        utility
    )


    targets.append(
        prediction_to_adb_target(
            target_id=target_id,
            utility=utility,
            uncertainty=uncertainty,
        )
    )


    target_id += 1



schedule = build_adb_schedule(
    tuple(targets),
    horizon_s=1.0,
    dt=0.1,
)



if len(schedule.steps) == 0:
    raise RuntimeError(
        "Empty ADB schedule."
    )



report = {

    "block":
        "stage4b_real_adb_from_prediction_smoke",

    "targets":
        len(targets),

    "steps":
        len(schedule.steps),

    "best_target":
        schedule.steps[0].target_id,

    "future_used":
        False,

    "status":
        "PASS",
}



REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)



print(
    "===== Stage4B Real ADB from Prediction ====="
)

print(
    "targets =",
    len(targets)
)

print(
    "steps =",
    len(schedule.steps)
)

print(
    "best_target =",
    schedule.steps[0].target_id
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

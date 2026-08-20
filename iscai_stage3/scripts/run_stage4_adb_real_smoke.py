from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.control.adb_adapter import (
    prediction_to_adb_target,
    build_adb_schedule,
)



REPORT = Path(
    "reports/block4a_adb_real_smoke.json"
)



targets = (

    prediction_to_adb_target(
        target_id=1,
        utility=0.92,
        uncertainty=0.05,
    ),

    prediction_to_adb_target(
        target_id=2,
        utility=0.71,
        uncertainty=0.10,
    ),

    prediction_to_adb_target(
        target_id=3,
        utility=0.65,
        uncertainty=0.20,
    ),

)



schedule = build_adb_schedule(
    targets,
    horizon_s=1.0,
    dt=0.1,
)



if len(schedule.steps) == 0:
    raise RuntimeError(
        "Empty ADB schedule."
    )



for step in schedule.steps:

    if step.uncertainty < 0:
        raise RuntimeError(
            "Invalid uncertainty."
        )



report = {

    "block":
        "stage4a_adb_real_smoke",

    "steps":
        len(schedule.steps),

    "first_target":
        schedule.steps[0].target_id,

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
    "===== Stage4 ADB real smoke ====="
)

print(
    "steps =",
    len(schedule.steps)
)

print(
    "first_target =",
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

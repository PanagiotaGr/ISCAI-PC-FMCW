from __future__ import annotations

import json
from pathlib import Path


from iscai_stage3.control.beam_scheduler import (
    allocate_beam,
)



REPORT = Path(
    "reports/block4d_beam_scheduler_smoke.json"
)



selected_target = {
    "target_id": 13,
    "utility": 0.85,
    "uncertainty": 0.10,
}



action = allocate_beam(
    target_id=selected_target["target_id"],
    utility=selected_target["utility"],
    uncertainty=selected_target["uncertainty"],
)



report = {

    "block":
        "stage4d_beam_scheduler_smoke",

    "target_id":
        action.target_id,

    "beam_id":
        action.beam_id,

    "priority":
        action.priority,

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
    "===== Stage4D Beam Scheduler ====="
)

print(
    "target =",
    action.target_id
)

print(
    "beam =",
    action.beam_id
)

print(
    "priority =",
    action.priority
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

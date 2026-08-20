from __future__ import annotations

import json
import hashlib
from pathlib import Path


from iscai_stage3.control.adb_adapter import (
    prediction_to_adb_target,
)

from iscai_stage3.control.adb_closed_loop import (
    select_next_action,
)

from iscai_stage3.control.beam_scheduler import (
    allocate_beam,
)



REPORT = Path(
    "reports/block4e_full_closed_loop_smoke.json"
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

    if item["future_used"]:

        raise RuntimeError(
            "Future leakage."
        )


    trajectory = item["trajectory"]


    start = trajectory[0]

    end = trajectory[-1]


    dx = end[0] - start[0]

    dy = end[1] - start[1]

    dz = end[2] - start[2]


    distance = (
        dx * dx +
        dy * dy +
        dz * dz
    ) ** 0.5



    utility = (
        distance /
        (1.0 + distance)
    )


    uncertainty = (
        1.0 /
        (1.0 + distance)
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



beam = allocate_beam(
    target_id=decision.target_id,
    utility=decision.utility,
    uncertainty=decision.uncertainty,
)



payload = {

    "target_id":
        beam.target_id,

    "beam_id":
        beam.beam_id,

    "priority":
        beam.priority,

}



sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
    ).encode()
).hexdigest()



report = {

    "block":
        "stage4e_full_closed_loop_smoke",

    "prediction_count":
        len(data),

    "selected_target":
        beam.target_id,

    "beam_id":
        beam.beam_id,

    "priority":
        beam.priority,

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
    "===== Stage4E Full Closed Loop ====="
)

print(
    "predictions =",
    len(data)
)

print(
    "selected_target =",
    beam.target_id
)

print(
    "beam_id =",
    beam.beam_id
)

print(
    "priority =",
    beam.priority
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

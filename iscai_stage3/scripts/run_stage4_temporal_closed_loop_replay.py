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
    "reports/block4f_temporal_closed_loop_replay.json"
)


ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing trajectory artifact."
    )



data = json.loads(
    ARTIFACT.read_text()
)



# replay steps
REPLAY_STEPS = 5



history = []



for t in range(REPLAY_STEPS):


    targets = []


    for idx, item in enumerate(data):


        if item["future_used"]:

            raise RuntimeError(
                "Future leakage."
            )


        trajectory = item["trajectory"]


        # temporal slice
        horizon = min(
            len(trajectory),
            t + 2,
        )


        partial = trajectory[
            :horizon
        ]


        start = partial[0]

        end = partial[-1]


        dx = end[0] - start[0]

        dy = end[1] - start[1]

        dz = end[2] - start[2]


        distance = (
            dx*dx +
            dy*dy +
            dz*dz
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


    history.append(
        {
            "time_index": t,

            "target_id":
                beam.target_id,

            "beam_id":
                beam.beam_id,

            "priority":
                beam.priority,
        }
    )



digest = hashlib.sha256(
    json.dumps(
        history,
        sort_keys=True,
    ).encode()
).hexdigest()



report = {

    "block":
        "stage4f_temporal_closed_loop_replay",

    "replay_steps":
        REPLAY_STEPS,

    "actions":
        len(history),

    "history":
        history,

    "future_used":
        False,

    "sha256":
        digest,

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
    "===== Stage4F Temporal Closed Loop Replay ====="
)

print(
    "steps =",
    REPLAY_STEPS
)

print(
    "actions =",
    len(history)
)

print(
    "first_action =",
    history[0]
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    digest
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

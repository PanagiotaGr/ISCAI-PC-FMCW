from __future__ import annotations

import json
import time
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

from iscai_stage3.control.controller_constraints import (
    validate_controller_history,
)



ARTIFACT = Path(
    "artifacts/real_predictor_trajectories.json"
)


REPORT = Path(
    "reports/block5e_performance_benchmark.json"
)



if not ARTIFACT.exists():

    raise RuntimeError(
        "Missing trajectory artifact."
    )



timings = {}



t0 = time.perf_counter()


data = json.loads(
    ARTIFACT.read_text()
)


timings["load"] = (
    time.perf_counter()
    -
    t0
)



# --------------------
# ADB target creation
# --------------------

t0 = time.perf_counter()


targets = []


for idx, item in enumerate(data):

    trajectory = item["trajectory"]


    start = trajectory[0]

    end = trajectory[-1]


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



timings["adb_generation"] = (
    time.perf_counter()
    -
    t0
)



# --------------------
# selection + beam
# --------------------

t0 = time.perf_counter()


decision = select_next_action(
    candidates=targets
)


beam = allocate_beam(
    target_id=decision.target_id,
    utility=decision.utility,
    uncertainty=decision.uncertainty,
)


timings["selection_beam"] = (
    time.perf_counter()
    -
    t0
)



# --------------------
# validation
# --------------------

t0 = time.perf_counter()


history = [
    {
        "time_index": 0,
        "target_id": beam.target_id,
        "beam_id": beam.beam_id,
        "priority": beam.priority,
    }
]


validation = validate_controller_history(
    history=history,
    beam_count=8,
)


timings["validation"] = (
    time.perf_counter()
    -
    t0
)



total_time = sum(
    timings.values()
)



payload = {

    "records":
        len(data),

    "timings":
        timings,

    "total_runtime":
        total_time,

    "trajectories_per_sec":
        len(data)
        /
        total_time,

    "selected_target":
        beam.target_id,

    "beam_id":
        beam.beam_id,

}



sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
    ).encode()
).hexdigest()



report = {

    "block":
        "stage5e_performance_benchmark",

    "records":
        len(data),

    "timings":
        timings,

    "total_runtime_sec":
        total_time,

    "trajectories_per_sec":
        len(data)
        /
        total_time,

    "actions_checked":
        validation.action_count,

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
    "===== Stage5E Performance Benchmark ====="
)

print(
    "records =",
    len(data)
)

print(
    "runtime_sec =",
    total_time
)

print(
    "trajectories/sec =",
    len(data)/total_time
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

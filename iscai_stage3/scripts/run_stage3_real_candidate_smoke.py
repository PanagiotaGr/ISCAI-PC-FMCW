from __future__ import annotations

import json
from pathlib import Path


from waymo_open_dataset.protos import scenario_pb2


from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


from iscai_stage3.pipeline.candidate_set import (
    build_candidate_set,
)


MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


SID = "b85e1bd6cc8e74c0"


REPORT = Path(
    "reports/block3d_real_candidate_smoke.json"
)


scenario = read_first_scenario(
    MOTION
)


if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected scenario."
    )


adapted = adapt_causal_womd_scenario(
    scenario
)


scene = build_real_ideal_observation_scene(
    raw_scenario=scenario,
    adapted=adapted,
    include_sdc=False,
)


candidate_set = build_candidate_set(
    scenario_id=scene.scenario_id,
    actor_series=scene.actors,
    azimuth_std_rad=0.01745,
    elevation_std_rad=0.01745,
    confidence=0.9,
)


for candidate in candidate_set.candidates:

    if not (
        candidate.scenario_id
        == SID
    ):
        raise RuntimeError(
            "Scenario leakage."
        )


report = {

    "block":
        "stage3_block3d_real_candidate",

    "scenario_id":
        SID,

    "actors":
        len(scene.actors),

    "candidates":
        len(candidate_set.candidates),

    "future_used":
        False,

    "measured_fmcw":
        False,

    "status":
        "PASS",
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print(
    "===== Stage3 real candidate smoke ====="
)

print(
    "scenario =",
    SID
)

print(
    "actors =",
    len(scene.actors)
)

print(
    "candidates =",
    len(candidate_set.candidates)
)

print(
    "future_used = NO"
)

print(
    "measured_fmcw = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

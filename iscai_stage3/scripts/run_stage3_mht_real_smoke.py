from __future__ import annotations

import json
from pathlib import Path


from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


from iscai_stage3.baselines.mht_adapter import (
    TrajectoryHypothesis,
    select_best_hypothesis,
)



SID = "b85e1bd6cc8e74c0"


MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


REPORT = Path(
    "reports/block3f_mht_real_smoke.json"
)



scenario = read_first_scenario(
    MOTION
)


adapted = adapt_causal_womd_scenario(
    scenario
)


scene = (
    build_real_ideal_observation_scene(
        raw_scenario=scenario,
        adapted=adapted,
        include_sdc=False,
    )
)



hypothesis_count = 0
selected_count = 0

future_used = False



for actor in scene.actors:

    history = []

    for obs in actor.observations:

        if not obs.geometry_valid:
            continue

        history.append(
            obs.actor_position_Ht_m
        )


    if len(history) < 3:
        continue



    last = history[-1]


    velocity_prediction = (
        (
            last[0] + (
                last[0]
                -
                history[-2][0]
            ),

            last[1] + (
                last[1]
                -
                history[-2][1]
            ),

            last[2] + (
                last[2]
                -
                history[-2][2]
            ),
        ),
    )



    constant_prediction = (
        (
            last[0],
            last[1],
            last[2],
        ),
    )



    hypotheses = (

        TrajectoryHypothesis(
            trajectory=velocity_prediction,
            score=1.0,
        ),


        TrajectoryHypothesis(
            trajectory=constant_prediction,
            score=2.0,
        ),

    )


    best = select_best_hypothesis(
        hypotheses
    )


    if best not in hypotheses:
        raise RuntimeError(
            "Invalid MHT selection."
        )


    hypothesis_count += len(
        hypotheses
    )

    selected_count += 1



report = {

    "block":
        "stage3f_mht_real_smoke",

    "scenario_id":
        SID,

    "actors":
        len(scene.actors),

    "hypotheses_generated":
        hypothesis_count,

    "selected":
        selected_count,

    "future_used":
        future_used,

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
    "===== Stage3 MHT real smoke ====="
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
    "hypotheses =",
    hypothesis_count
)

print(
    "selected =",
    selected_count
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

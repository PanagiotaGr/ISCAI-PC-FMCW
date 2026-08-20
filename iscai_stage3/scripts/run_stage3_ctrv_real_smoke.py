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


from iscai_stage3.baselines.ctrv_state import (
    estimate_ctrv_state,
)


from iscai_stage3.baselines.ctrv import (
    predict_ctrv,
)



MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


SID = "b85e1bd6cc8e74c0"



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



states = []


for actor in scene.actors:

    for index, obs in enumerate(
        actor.observations
    ):

        if index <= 1:
            continue


        if not obs.geometry_valid:
            continue


        try:

            state = estimate_ctrv_state(
                actor.observations,
                index,
            )


            states.append(
                state
            )

            break


        except ValueError:
            continue



if not states:
    raise RuntimeError(
        "No CTRV states generated."
    )



state = states[0]



prediction = predict_ctrv(
    position=state.position_m,

    speed=state.speed_mps,

    heading_rad=state.heading_rad,

    yaw_rate=state.yaw_rate_radps,

    horizon_s=1.0,

    dt=0.1,
)



report = {

    "block":
        "stage3c_ctrv_real_smoke",

    "scenario_id":
        SID,

    "states":
        len(states),

    "prediction_steps":
        len(prediction),

    "future_used":
        False,

    "causal_only":
        True,

    "state_source":
        "causal_velocity_heading",

    "status":
        "PASS",
}



Path(
    "reports"
).mkdir(
    exist_ok=True
)


Path(
    "reports/block3c_ctrv_real_smoke.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
)



print(
    "===== Stage3 CTRV real smoke ====="
)

print(
    "scenario =",
    SID
)

print(
    "states =",
    len(states)
)

print(
    "prediction_steps =",
    len(prediction)
)

print(
    "state_source = causal_velocity_heading"
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

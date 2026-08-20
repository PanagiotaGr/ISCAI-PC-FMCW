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


from iscai_stage3.baselines.stage2_adapter import (
    ideal_observation_to_state,
)


from iscai_stage3.baselines.constant_acceleration import (
    predict_constant_acceleration,
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

            state = ideal_observation_to_state(
                actor.observations,
                index,
            )


            if (
                state.acceleration_mps2
                is None
            ):
                continue


            states.append(
                state
            )

            break


        except ValueError:
            continue



if not states:
    raise RuntimeError(
        "No CA states generated."
    )



state = states[0]



prediction = predict_constant_acceleration(
    position=state.position_m,
    velocity=state.velocity_mps,
    acceleration=state.acceleration_mps2,
    horizon_s=1.0,
    dt=0.1,
)



report = {

    "block":
        "stage3b_ca_real_smoke",

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

    "acceleration_source":
        "causal_second_difference",

    "status":
        "PASS",
}



Path(
    "reports"
).mkdir(
    exist_ok=True
)


Path(
    "reports/block3b_ca_real_smoke.json"
).write_text(
    json.dumps(
        report,
        indent=2,
    )
)



print(
    "===== Stage3 CA real smoke ====="
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
    "acceleration_source = causal_second_difference"
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

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


from iscai_stage3.baselines.registry import (
    register_prediction,
)



SID = "b85e1bd6cc8e74c0"


MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


REPORT = Path(
    "reports/block3h_real_prediction_registry_smoke.json"
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


scene = (
    build_real_ideal_observation_scene(
        raw_scenario=scenario,
        adapted=adapted,
        include_sdc=False,
    )
)



models = (
    "CV",
    "CA",
    "CTRV",
    "KALMAN",
    "IMM",
    "MHT",
)



registered = 0
actor_count = 0



for actor in scene.actors:

    states = []


    for obs in actor.observations:

        if not obs.geometry_valid:
            continue

        states.append(
            (
                obs.range_m,
                obs.azimuth_rad,
                obs.elevation_rad,
            )
        )


    if len(states) < 2:
        continue


    # -------------------------------------------------
    # Registry receives only causal output.
    #
    # Real predictors will replace this placeholder
    # state sequence in the next closure step.
    # -------------------------------------------------

    for model in models:

        prediction = register_prediction(
            model_name=model,

            states=(
                tuple(states[-2:])
            ),
        )


        if prediction.future_used:
            raise RuntimeError(
                "Future leakage."
            )


        registered += 1


    actor_count += 1



report = {

    "block":
        "stage3h_real_prediction_registry_smoke",

    "scenario_id":
        SID,

    "actors":
        actor_count,

    "registered_predictions":
        registered,

    "models":
        list(models),

    "future_used":
        False,

    "truth_sidecar_used":
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
    "===== Stage3 Real Prediction Registry smoke ====="
)

print(
    "scenario =",
    SID
)

print(
    "actors =",
    actor_count
)

print(
    "registered =",
    registered
)

print(
    "models =",
    list(models)
)

print(
    "future_used = NO"
)

print(
    "truth_sidecar_used = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

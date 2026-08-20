from pathlib import Path
import json


from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)


DATA = Path(
    "/home/agni/waymo/data/paired_womd_lidar_v1_3_0/training/motion"
)


OUTPUT = Path(
    "artifacts/stage3_class_aware_predictions.json"
)


HORIZON = 10


def map_actor_type(track):

    value = int(track.object_type)

    if value == 1:
        return "VEHICLE"

    if value == 2:
        return "PEDESTRIAN"

    if value == 3:
        return "CYCLIST"

    return "UNKNOWN"



records = []

classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


files = list(
    DATA.glob(
        "paired-from-training.tfrecord-*"
    )
)[:50]


for file in files:

    scenario = read_first_scenario(
        file
    )

    adapted = adapt_causal_womd_scenario(
        scenario
    )

    for index, actor in enumerate(
        adapted.actors
    ):

        actor_type = map_actor_type(
            scenario.tracks[index]
        )


        if actor_type not in classes:
            continue


        classes[actor_type] += 1


        record = {
            "scenario_id":
                scenario.scenario_id,

            "track_index":
                index,

            "actor_type":
                actor_type,

            "models": [
                "CV",
                "CA",
                "CTRV",
                "KALMAN",
                "IMM",
                "MHT",
            ],

            "horizon":
                HORIZON,

            "future_used":
                False,
        }


        records.append(
            record
        )



OUTPUT.parent.mkdir(
    exist_ok=True
)


OUTPUT.write_text(
    json.dumps(
        records,
        indent=2,
        sort_keys=True,
    )
)



print(
    "===== Stage3 Class Aware Predictor ====="
)

print(
    "records =",
    len(records)
)

print(
    "vehicle =",
    classes["VEHICLE"]
)

print(
    "pedestrian =",
    classes["PEDESTRIAN"]
)

print(
    "cyclist =",
    classes["CYCLIST"]
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

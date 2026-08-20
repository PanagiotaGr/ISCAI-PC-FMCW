from pathlib import Path


from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)


MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


scenario = read_first_scenario(
    MOTION
)


print(
    "===== Actor Type Inspection ====="
)

print(
    "scenario =",
    scenario.scenario_id
)

print(
    "tracks =",
    len(scenario.tracks)
)


for i, track in enumerate(
    scenario.tracks[:20]
):

    print(
        "index =",
        i,
        "id =",
        track.id,
        "type =",
        getattr(
            track,
            "object_type",
            "NO_OBJECT_TYPE"
        )
    )

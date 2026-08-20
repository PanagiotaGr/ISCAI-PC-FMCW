from pathlib import Path

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

DATA = Path("/home/agni/waymo/data")

file = next(DATA.rglob("*.tfrecord"))

print("FILE =", file)

scenario = read_first_scenario(file)

print("scenario =", scenario.scenario_id)
print("tracks =", len(scenario.tracks))

for i, track in enumerate(scenario.tracks[:10]):

    print(
        "track",
        i,
        "fields =",
        [
            f.name
            for f in track.DESCRIPTOR.fields
        ]
    )

    print(
        "object_type raw =",
        track.object_type
    )


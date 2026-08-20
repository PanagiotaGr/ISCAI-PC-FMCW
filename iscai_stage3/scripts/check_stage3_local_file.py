from pathlib import Path

from iscai_stage0.womd_proto_io import (
    iter_scenarios,
)


PATH = Path(
    "/home/agni/waymo/data/womd_lidar_stage0/validation/b85e1bd6cc8e74c0.tfrecord"
)


def main():

    print(
        "===== checking local tfrecord ====="
    )

    for i, scenario in enumerate(
        iter_scenarios(PATH, limit=3)
    ):

        print(
            "record",
            i,
            "scenario_id=",
            scenario.scenario_id,
            "tracks=",
            len(scenario.tracks),
        )


if __name__ == "__main__":
    main()

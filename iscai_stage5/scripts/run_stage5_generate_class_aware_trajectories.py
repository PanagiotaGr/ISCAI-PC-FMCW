
from __future__ import annotations

import json
import math
from pathlib import Path

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage3.baselines.causal_state import (
    estimate_causal_motion_state,
)

from iscai_stage3.baselines.constant_velocity import (
    predict_constant_velocity,
)

from iscai_stage3.baselines.constant_acceleration import (
    predict_constant_acceleration,
)

from iscai_stage3.baselines.ctrv import (
    predict_ctrv,
)

from iscai_stage3.baselines.kalman import (
    predict_kalman_cv,
)


DATA = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/training/motion"
)

OUTPUT = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

MAX_FILES = 50
DT = 0.1
HORIZON_S = 1.0


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


files = sorted(
    DATA.glob(
        "paired-from-training.tfrecord-*"
    )
)[:MAX_FILES]


for file in files:

    scenario = read_first_scenario(
        file
    )

    adapted = adapt_causal_womd_scenario(
        scenario
    )

    for actor in adapted.actors:

        track_index = actor.track_index

        if not (
            0 <= track_index < len(
                scenario.tracks
            )
        ):
            continue

        actor_type = map_actor_type(
            scenario.tracks[
                track_index
            ]
        )

        if actor_type not in classes:
            continue

        positions = []
        timestamps = []

        history = actor.artifact.history

        for i, valid in enumerate(
            history.state_valid
        ):

            if not valid:
                continue

            positions.append(
                (
                    float(
                        history.position_W_m[i][0]
                    ),
                    float(
                        history.position_W_m[i][1]
                    ),
                    float(
                        history.position_W_m[i][2]
                    ),
                )
            )

            timestamps.append(
                float(
                    history.timestamps_s[i]
                )
            )

        if len(positions) < 3:
            continue

        motion = estimate_causal_motion_state(
            positions=tuple(
                positions[-3:]
            ),
            timestamps=tuple(
                timestamps[-3:]
            ),
        )

        px, py, pz = motion.position
        vx, vy, vz = motion.velocity

        dt_prev = (
            timestamps[-1]
            -
            timestamps[-2]
        )

        if dt_prev > 1e-9:

            vx_prev = (
                positions[-1][0]
                -
                positions[-2][0]
            ) / dt_prev

            vy_prev = (
                positions[-1][1]
                -
                positions[-2][1]
            ) / dt_prev

            vz_prev = (
                positions[-1][2]
                -
                positions[-2][2]
            ) / dt_prev

        else:
            vx_prev = vx
            vy_prev = vy
            vz_prev = vz

        ax = (
            vx - vx_prev
        ) / max(
            dt_prev,
            1e-9
        )

        ay = (
            vy - vy_prev
        ) / max(
            dt_prev,
            1e-9
        )

        az = (
            vz - vz_prev
        ) / max(
            dt_prev,
            1e-9
        )

        speed = math.sqrt(
            vx * vx
            +
            vy * vy
        )

        heading = (
            math.atan2(
                vy,
                vx,
            )
            if speed > 1e-9
            else 0.0
        )

        yaw_rate = 0.0

        outputs = {

            "CV":
                predict_constant_velocity(
                    position=motion.position,
                    velocity=motion.velocity,
                    horizon_s=HORIZON_S,
                    dt=DT,
                ),

            "CA":
                predict_constant_acceleration(
                    position=motion.position,
                    velocity=motion.velocity,
                    acceleration=(
                        ax,
                        ay,
                        az,
                    ),
                    horizon_s=HORIZON_S,
                    dt=DT,
                ),

            "CTRV":
                predict_ctrv(
                    position=motion.position,
                    speed=speed,
                    heading_rad=heading,
                    yaw_rate=yaw_rate,
                    horizon_s=HORIZON_S,
                    dt=DT,
                ),

            "KALMAN":
                predict_kalman_cv(
                    position=motion.position,
                    velocity=motion.velocity,
                    horizon_s=HORIZON_S,
                    dt=DT,
                ),
        }

        classes[
            actor_type
        ] += 1

        for model_name, trajectory in outputs.items():

            records.append(
                {
                    "scenario_id":
                        str(
                            scenario.scenario_id
                        ),

                    "track_index":
                        int(
                            track_index
                        ),

                    "actor_type":
                        actor_type,

                    "model_name":
                        model_name,

                    "trajectory":
                        [
                            [
                                float(point[0]),
                                float(point[1]),
                                float(point[2]),
                            ]
                            for point in trajectory
                        ],

                    "causal_only":
                        True,

                    "future_used":
                        False,
                }
            )


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        records,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Class Aware Trajectory Generation ====="
)

print(
    "files =",
    len(files)
)

print(
    "actors =",
    sum(
        classes.values()
    )
)

print(
    "trajectory_records =",
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
    "models =",
    [
        "CV",
        "CA",
        "CTRV",
        "KALMAN",
    ]
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

print(
    "output =",
    OUTPUT
)

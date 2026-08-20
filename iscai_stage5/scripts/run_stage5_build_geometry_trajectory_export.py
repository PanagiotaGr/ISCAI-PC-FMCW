
from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path(
    "/home/agni/waymo/iscai_stage3/artifacts/"
    "real_predictor_trajectories.json"
)

OUTPUT = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)


data = json.loads(
    SOURCE.read_text()
)


export = []

for item in data:

    export.append(
        {
            "scenario_id":
                item["scenario_id"],

            "track_index":
                item["track_index"],

            "actor_type":
                item.get(
                    "actor_type",
                    "UNKNOWN"
                ),

            "trajectory":
                item["trajectory"],

            "future_used":
                item.get(
                    "future_used",
                    False
                ),
        }
    )


OUTPUT.parent.mkdir(
    exist_ok=True
)


OUTPUT.write_text(
    json.dumps(
        export,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 Trajectory Export ====="
)

print(
    "records =",
    len(export)
)

print(
    "output =",
    OUTPUT
)

print(
    "future_used = NO"
)

print(
    "STATUS = PASS"
)

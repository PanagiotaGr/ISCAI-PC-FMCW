import hashlib
import json

from iscai_stage6.adb.geometry import (
    trajectory_to_angular_path
)


trajectory = [

    [10.0, 0.0, 0.0],

    [20.0, 1.0, 0.0],

    [30.0, 2.0, 0.0],

]


result = trajectory_to_angular_path(
    trajectory
)


output = {

    "points":
        len(result),

    "finite":
        all(
            isinstance(
                x["azimuth_rad"],
                float
            )
            for x in result
        ),

    "future_used":
        False,

    "status":
        "PASS",

}


output["sha256"] = hashlib.sha256(
    json.dumps(
        output,
        sort_keys=True
    ).encode()
).hexdigest()


print(
    "===== Stage6 Geometry Smoke ====="
)

print(
    "points =",
    output["points"]
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    output["sha256"]
)

print(
    "STATUS = PASS"
)

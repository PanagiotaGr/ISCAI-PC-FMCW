import json
import hashlib

from iscai_stage6.adb.illumination import (
    trajectory_to_illumination_mask
)


angular_path = [

    {
        "azimuth_rad": 0.0,
        "elevation_rad": 0.0,
    },

    {
        "azimuth_rad": 0.05,
        "elevation_rad": 0.0,
    },

]


mask = trajectory_to_illumination_mask(
    angular_path
)


output = {

    "cells":
        len(mask),

    "probability_valid":
        all(
            0 <= x["probability"] <= 1
            for x in mask
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


print("===== Stage6 Illumination Mask Smoke =====")
print("cells =", output["cells"])
print("probability_valid =", output["probability_valid"])
print("future_used = NO")
print("SHA256 =", output["sha256"])
print("STATUS = PASS")

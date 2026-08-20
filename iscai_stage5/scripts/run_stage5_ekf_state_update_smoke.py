
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path


SOURCE = Path(
    "artifacts/stage5_pcfmcw_observations.json"
)

REPORT = Path(
    "reports/block5_ekf_state_update_smoke.json"
)


data = json.loads(
    SOURCE.read_text()
)


DT = 0.1


updated = 0
innovation_valid = 0
covariance_valid = 0


classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in data:

    r = float(
        item["range_m"]
    )

    vr = float(
        item["radial_velocity_mps"]
    )

    theta = float(
        item["azimuth_rad"]
    )


    # Initial Cartesian state from measurement.
    px = r * math.cos(theta)
    py = r * math.sin(theta)


    vx = vr * math.cos(theta)
    vy = vr * math.sin(theta)


    x = [
        px,
        py,
        vx,
        vy,
    ]


    # Simple constant-velocity prediction.
    x_pred = [
        x[0] + x[2] * DT,
        x[1] + x[3] * DT,
        x[2],
        x[3],
    ]


    # Reconstructed measurement.
    pred_range = math.sqrt(
        x_pred[0] ** 2
        +
        x_pred[1] ** 2
    )


    pred_azimuth = math.atan2(
        x_pred[1],
        x_pred[0],
    )


    if pred_range > 1e-9:

        pred_vr = (
            x_pred[2] * x_pred[0]
            +
            x_pred[3] * x_pred[1]
        ) / pred_range

    else:

        pred_vr = 0.0


    z = [
        r,
        vr,
        theta,
    ]

    z_pred = [
        pred_range,
        pred_vr,
        pred_azimuth,
    ]


    innovation = [
        z[i] - z_pred[i]
        for i in range(3)
    ]


    if all(
        math.isfinite(v)
        for v in innovation
    ):
        innovation_valid += 1


    R = item.get(
        "measurement_covariance",
        None
    )


    if (
        R is not None
        and len(R) == 4
        and all(
            len(row) == 4
            for row in R
        )
    ):
        covariance_valid += 1


    # Smoke update:
    # apply bounded innovation correction.
    gain = 0.5

    x_new = [
        x_pred[0] + gain * innovation[0] * math.cos(theta),
        x_pred[1] + gain * innovation[0] * math.sin(theta),
        x_pred[2] + gain * innovation[1] * math.cos(theta),
        x_pred[3] + gain * innovation[1] * math.sin(theta),
    ]


    if all(
        math.isfinite(v)
        for v in x_new
    ):
        updated += 1


    actor_type = item.get(
        "actor_type",
        "UNKNOWN"
    )

    if actor_type in classes:
        classes[actor_type] += 1



payload = {

    "records":
        len(data),

    "innovation_valid":
        innovation_valid,

    "state_update_valid":
        updated,

    "covariance_valid":
        covariance_valid,

    "vehicle":
        classes["VEHICLE"],

    "pedestrian":
        classes["PEDESTRIAN"],

    "cyclist":
        classes["CYCLIST"],

    "future_used":
        False,
}


sha = hashlib.sha256(
    json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode()
).hexdigest()


status = (
    "PASS"
    if (
        innovation_valid == len(data)
        and updated == len(data)
        and covariance_valid == len(data)
    )
    else
    "FAIL"
)


payload.update(
    {
        "sha256": sha,
        "status": status,
    }
)


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 EKF State Update Smoke ====="
)

print(
    "records =",
    len(data)
)

print(
    "innovation_valid =",
    innovation_valid
)

print(
    "state_update_valid =",
    updated
)

print(
    "covariance_valid =",
    covariance_valid
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
    "SHA256 =",
    sha
)

print(
    "STATUS =",
    status
)

if status != "PASS":
    raise SystemExit(1)

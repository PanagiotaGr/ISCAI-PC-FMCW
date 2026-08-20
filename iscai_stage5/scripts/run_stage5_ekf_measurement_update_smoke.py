from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)


ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
    ROOT / "iscai_stage5" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)



import json
import math
import hashlib
from pathlib import Path

from iscai_stage5.uncertainty.measurement_covariance import (
    compute_measurement_uncertainty,
)


SOURCE = Path(
    "artifacts/stage5_pcfmcw_observations.json"
)

REPORT = Path(
    "reports/block5_ekf_measurement_update_smoke.json"
)


data = json.loads(
    SOURCE.read_text()
)


valid = 0
positive_diagonal = 0
finite_measurements = 0


classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in data:

    z = [
        float(
            item["range_m"]
        ),
        float(
            item["radial_velocity_mps"]
        ),
        float(
            item["azimuth_rad"]
        ),
        float(
            item["elevation_rad"]
        ),
    ]


    if all(
        math.isfinite(value)
        for value in z
    ):
        finite_measurements += 1


    uncertainty = compute_measurement_uncertainty(
        range_m=
            float(
                item["range_m"]
            ),

        snr_db=
            float(
                item["snr_db"]
            ),
    )


    R = uncertainty[
        "covariance"
    ]


    matrix_valid = (
        len(R) == 4
        and all(
            len(row) == 4
            for row in R
        )
    )


    if not matrix_valid:
        continue


    diagonal = [
        R[i][i]
        for i in range(4)
    ]


    if all(
        value > 0.0
        and math.isfinite(value)
        for value in diagonal
    ):
        positive_diagonal += 1


    # Smoke-level EKF interface validation.
    #
    # We are not yet performing a full nonlinear
    # predict/update cycle. Here we verify that each
    # observation forms a valid measurement packet:
    #
    # z = [r, vr, theta, phi]
    # R = 4x4 positive measurement covariance.
    packet_valid = (
        len(z) == 4
        and all(
            math.isfinite(value)
            for value in z
        )
        and all(
            value > 0.0
            for value in diagonal
        )
    )


    if packet_valid:
        valid += 1


    actor_type = item.get(
        "actor_type",
        "UNKNOWN"
    )


    if actor_type in classes:
        classes[
            actor_type
        ] += 1



payload = {
    "records":
        len(data),

    "measurement_dimension":
        4,

    "finite_measurements":
        finite_measurements,

    "positive_covariance_diagonal":
        positive_diagonal,

    "valid_ekf_packets":
        valid,

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
        valid == len(data)
        and finite_measurements == len(data)
        and positive_diagonal == len(data)
    )
    else "FAIL"
)


report = {
    **payload,

    "sha256":
        sha,

    "status":
        status,
}


REPORT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)


print(
    "===== Stage5 EKF Measurement Update Smoke ====="
)

print(
    "records =",
    len(data)
)

print(
    "measurement_dimension = 4"
)

print(
    "finite_measurements =",
    finite_measurements
)

print(
    "positive_covariance_diagonal =",
    positive_diagonal
)

print(
    "valid_ekf_packets =",
    valid
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

print(
    "report =",
    REPORT
)


if status != "PASS":
    raise SystemExit(1)

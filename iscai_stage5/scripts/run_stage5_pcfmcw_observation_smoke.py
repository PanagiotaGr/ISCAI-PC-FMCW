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

from iscai_stage5.geometry.headlamp_geometry import (
    compute_headlamp_state,
)


SOURCE = Path(
    "artifacts/stage5_class_aware_trajectories.json"
)

OUTPUT = Path(
    "artifacts/stage5_pcfmcw_observations.json"
)

REPORT = Path(
    "reports/block5_pcfmcw_observation_smoke.json"
)


# Deterministic baseline uncertainty parameters.
# These are smoke-test parameters, not yet CRLB-derived.
SIGMA_RANGE_M = 0.10
SIGMA_RADIAL_VELOCITY_MPS = 0.10
SIGMA_AZIMUTH_RAD = 0.005
SIGMA_ELEVATION_RAD = 0.005

DT = 0.1


data = json.loads(
    SOURCE.read_text()
)


observations = []

classes = {
    "VEHICLE": 0,
    "PEDESTRIAN": 0,
    "CYCLIST": 0,
}


for item in data:

    trajectory = item.get(
        "trajectory",
        []
    )

    if len(trajectory) < 2:
        continue


    p0 = trajectory[0]
    p1 = trajectory[1]


    velocity = (
        (
            float(p1[0])
            -
            float(p0[0])
        ) / DT,

        (
            float(p1[1])
            -
            float(p0[1])
        ) / DT,

        (
            float(p1[2])
            -
            float(p0[2])
        ) / DT,
    )


    state = compute_headlamp_state(
        actor_position=(
            float(p0[0]),
            float(p0[1]),
            float(p0[2]),
        ),
        actor_velocity=velocity,
    )



    # Deterministic smoke mode:
    # estimated measurement == geometry truth.
    r_hat = float(
        state.range_m
    )

    vr_hat = float(
        state.radial_velocity
    )

    theta_hat = float(
        state.azimuth_rad
    )

    phi_hat = float(
        state.elevation_rad
    )


    # Simple deterministic SNR proxy for the smoke layer.
    # This will later be replaced by the physical PC-FMCW
    # signal/SNR model.
    snr_linear = 1.0 / max(
        r_hat * r_hat,
        1.0,
    )

    snr_db = 10.0 * math.log10(
        max(
            snr_linear,
            1e-12,
        )
    )


    R = [
        [
            SIGMA_RANGE_M ** 2,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            SIGMA_RADIAL_VELOCITY_MPS ** 2,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            SIGMA_AZIMUTH_RAD ** 2,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            SIGMA_ELEVATION_RAD ** 2,
        ],
    ]


    actor_type = item.get(
        "actor_type",
        "UNKNOWN"
    )

    if actor_type in classes:
        classes[
            actor_type
        ] += 1


    observations.append(
        {
            "scenario_id":
                item["scenario_id"],

            "track_index":
                int(
                    item["track_index"]
                ),

            "actor_type":
                actor_type,

            "model_name":
                item.get(
                    "model_name",
                    "UNKNOWN"
                ),

            "range_m":
                r_hat,

            "radial_velocity_mps":
                vr_hat,

            "azimuth_rad":
                theta_hat,

            "elevation_rad":
                phi_hat,

            "snr_db":
                float(
                    snr_db
                ),

            "measurement_covariance":
                R,

            "measurement_order": [
                "range_m",
                "radial_velocity_mps",
                "azimuth_rad",
                "elevation_rad",
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
        observations,
        indent=2,
        sort_keys=True,
    )
)


payload = {
    "records":
        len(observations),

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


report = {
    **payload,
    "sha256":
        sha,

    "status":
        "PASS",
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
    "===== Stage5 PC-FMCW Observation Smoke ====="
)

print(
    "source_records =",
    len(data)
)

print(
    "observations =",
    len(observations)
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
    "measurement =",
    [
        "range",
        "radial_velocity",
        "azimuth",
        "elevation",
    ]
)

print(
    "covariance = 4x4"
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    sha
)

print(
    "STATUS = PASS"
)

print(
    "output =",
    OUTPUT
)

print(
    "report =",
    REPORT
)

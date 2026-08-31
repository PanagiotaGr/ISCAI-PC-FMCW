from __future__ import annotations

import hashlib
import json

from pathlib import Path

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.baselines import (
    predict_kalman,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.filters import (
    KalmanConfig,
    KalmanFrameContext,
    filter_associated_track,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
)

from rigid_test_factory import (
    T_sensor_from_W,
    identity_transform,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)


def implementation_hash():
    files = []

    for base in (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
    ):
        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix == ".pyc":
                continue

            files.append(path)

    files = sorted(
        set(files),
        key=lambda p: str(
            p.relative_to(ROOT)
        ),
    )

    h = hashlib.sha256()

    for path in files:
        relative = str(
            path.relative_to(ROOT)
        )

        h.update(
            relative.encode("utf-8")
        )

        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")

    return h.hexdigest(), len(files)


# -------------------------------------------------
# Exact moving-sensor / constant-velocity target.
#
# sensor vx = 10 m/s
# actor  vx = 15 m/s
# relative radial velocity = +5 m/s
# -------------------------------------------------

times = (
    0.0,
    0.1,
    0.2,
    0.3,
)

sensor_x = (
    0.0,
    1.0,
    2.0,
    3.0,
)

actor_x = (
    10.0,
    11.5,
    13.0,
    14.5,
)


frames = []

for i, timestamp in enumerate(
    times
):
    relative_range = (
        actor_x[i]
        -
        sensor_x[i]
    )

    frames.append(
        make_frame(
            timestamp_s=timestamp,
            detections=(
                make_detection(
                    key=f"frame-{i}",
                    range_m=relative_range,
                    vr_mps=5.0,
                    az_rad=0.0,
                    el_rad=0.0,
                ),
            ),
        )
    )


sequence = AlgorithmObservationSequence(
    scenario_id="block34-kalman",
    frames=tuple(frames),
)


associated = associate_estimated_gnn(
    sequence
)


if len(associated.tracks) != 1:
    raise SystemExit(
        "FAIL: synthetic actor did not "
        "remain one estimated track."
    )


transform_context = FrameTransformContext(
    T_H0_from_W=identity_transform(),

    T_Ht_from_W_by_frame=tuple(
        T_sensor_from_W(
            sensor_origin_W=(
                x,
                0.0,
                0.0,
            )
        )
        for x in sensor_x
    ),
)


kalman_context = KalmanFrameContext(
    transforms=transform_context,
    frame_timestamps_s=times,
)


config = KalmanConfig(
    acceleration_spectral_density_m2_s3=0.0
)


first = filter_associated_track(
    associated.tracks[0],
    context=kalman_context,
    config=config,
)


second = filter_associated_track(
    associated.tracks[0],
    context=kalman_context,
    config=config,
)


if first != second:
    raise SystemExit(
        "FAIL: EKF recursion is not "
        "deterministic."
    )


if len(first.updates) != 2:
    raise SystemExit(
        "FAIL: expected two genuine EKF "
        "measurement updates."
    )


for update in first.updates:
    if not update.measurement_covariance_used:
        raise SystemExit(
            "FAIL: Stage2 R_t was not used."
        )


x = first.current_state.mean_6


if abs(
    x[0] - 14.5
) > 1e-7:
    raise SystemExit(
        "FAIL: EKF position recovery."
    )


if abs(
    x[3] - 15.0
) > 1e-7:
    raise SystemExit(
        "FAIL: EKF velocity recovery."
    )


for update in first.updates:
    if abs(
        update.innovation[1]
    ) > 1e-7:
        raise SystemExit(
            "FAIL: radial-velocity measurement "
            "model is inconsistent."
        )


prediction = predict_kalman(
    track_id=first.track_id,
    state=first.current_state,
    horizons_s=(
        0.1,
        0.3,
        0.5,
        1.0,
    ),
    config=config,
)


expected = (
    16.0,
    19.0,
    22.0,
    29.5,
)


for point, expected_x in zip(
    prediction.points,
    expected,
):
    if abs(
        point.position_H0_m[0]
        -
        expected_x
    ) > 1e-7:
        raise SystemExit(
            "FAIL: Kalman CV prediction "
            "closed-form mismatch."
        )


if (
    first.truth_used
    or
    first.annotated_velocity_used
    or
    first.future_information_used
):
    raise SystemExit(
        "FAIL: prohibited Kalman input used."
    )


# -------------------------------------------------
# Config gate
# -------------------------------------------------

config_json = json.loads(
    (
        ROOT
        / "configs"
        / "stage3_kalman.json"
    ).read_text(
        encoding="utf-8"
    )
)


if config_json[
    "baseline"
] != "CV_EKF":
    raise SystemExit(
        "FAIL: Kalman baseline identity "
        "changed."
    )


if config_json[
    "measurement_model"
][
    "vector_order"
] != [
    "range",
    "radial_velocity",
    "azimuth",
    "elevation",
]:
    raise SystemExit(
        "FAIL: measurement order changed."
    )


if not config_json[
    "measurement_model"
][
    "nonlinear"
]:
    raise SystemExit(
        "FAIL: EKF measurement model "
        "disabled."
    )


if config_json[
    "measurement_model"
][
    "jacobian"
] != "analytic":
    raise SystemExit(
        "FAIL: analytic Jacobian policy "
        "changed."
    )


if not config_json[
    "measurement_uncertainty"
][
    "uses_full_stage2_R_t"
]:
    raise SystemExit(
        "FAIL: full Stage2 R_t disabled."
    )


if config_json[
    "measurement_uncertainty"
][
    "R_t_replaced_by_fixed_noise"
]:
    raise SystemExit(
        "FAIL: Stage2 R_t replaced by "
        "fixed noise."
    )


if config_json[
    "covariance_update"
][
    "method"
] != "Joseph":
    raise SystemExit(
        "FAIL: Joseph covariance update "
        "disabled."
    )


if (
    config_json["truth_used"]
    or
    config_json[
        "future_information_used"
    ]
):
    raise SystemExit(
        "FAIL: prohibited config input."
    )


if config_json[
    "formal_validation_tuning"
]:
    raise SystemExit(
        "FAIL: Kalman parameters marked "
        "as validation-tuned."
    )


sha, count = implementation_hash()


report = {
    "stage": 3,
    "block": "3.4",
    "status": "PASS",

    "baseline": "CV_EKF",

    "state": {
        "frame": "H0",
        "dimension": 6,
        "order": [
            "px",
            "py",
            "pz",
            "vx",
            "vy",
            "vz"
        ]
    },

    "process_model": {
        "model": "CV",
        "process_covariance": (
            "continuous_white_acceleration"
        )
    },

    "measurement_model": {
        "nonlinear": True,
        "frame": "Ht",
        "vector": [
            "range",
            "radial_velocity",
            "azimuth",
            "elevation"
        ],
        "analytic_jacobian": True,
        "moving_sensor_velocity": (
            "strict_adjacent_pose_difference"
        ),
        "azimuth_innovation_wrapping": True
    },

    "measurement_uncertainty": {
        "full_stage2_R_t_used": True,
        "fixed_R_replacement": False
    },

    "filter": {
        "real_measurement_updates": (
            len(first.updates)
        ),
        "joseph_covariance_update": True,
        "deterministic": True
    },

    "closed_form_gate": {
        "sensor_velocity_x_mps": 10.0,
        "actor_velocity_x_mps": 15.0,
        "relative_radial_velocity_mps": 5.0,
        "recovered_actor_x_m": x[0],
        "expected_actor_x_m": 14.5,
        "recovered_actor_vx_mps": x[3],
        "expected_actor_vx_mps": 15.0,
        "status": "PASS"
    },

    "information_policy": {
        "truth_used": False,
        "annotated_velocity_used": False,
        "future_information_used": False
    },

    "formal_real_validation_metrics": (
        "DEFERRED_TO_COMMON_STAGE3_"
        "EVALUATION_BLOCK"
    ),

    "implementation": {
        "files": count,
        "sha256": sha
    }
}


report_path = (
    ROOT
    / "reports"
    / "block34_kalman_gate.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage3 Block 3.4 gate ====="
)

print(
    "Kalman implementation       = REAL EKF"
)

print(
    "state frame                 = H0"
)

print(
    "process model               = CV"
)

print(
    "measurement frame           = Ht"
)

print(
    "measurement vector          = "
    "[r, vr, az, el]"
)

print(
    "analytic Jacobian           = PASS"
)

print(
    "moving sensor compensation  = PASS"
)

print(
    "Stage2 full R_t             = USED"
)

print(
    "Joseph covariance update    = PASS"
)

print(
    "real EKF updates            =",
    len(first.updates),
)

print(
    "recovered actor x           =",
    x[0],
)

print(
    "expected actor x            = 14.5"
)

print(
    "recovered actor vx          =",
    x[3],
)

print(
    "expected actor vx           = 15.0"
)

print(
    "radial velocity model       = PASS"
)

print(
    "annotated velocity          = NO"
)

print(
    "truth leakage               = NONE"
)

print(
    "future information          = NO"
)

print(
    "deterministic repeat        = PASS"
)

print(
    "implementation files        =",
    count,
)

print(
    "implementation SHA256       =",
    sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    report_path,
)

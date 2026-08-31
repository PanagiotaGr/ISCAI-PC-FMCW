from __future__ import annotations

import hashlib
import json

from pathlib import Path

from iscai_stage1.geometry.frames import (
    AnchorFrames,
    DynamicHeadlampFrame,
)

from iscai_stage1.geometry.rigid import (
    RigidTransform,
)

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.baselines import (
    predict_cv,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
    cartesianize_associated_track,
)

from iscai_stage3.state import (
    estimate_causal_cv_state,
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
# Frozen Stage1 public-API contract check
# -------------------------------------------------

for method in (
    "inverse",
    "apply_point",
    "apply_vector",
):
    if not hasattr(
        RigidTransform,
        method,
    ):
        raise SystemExit(
            "FAIL: frozen Stage1 "
            f"RigidTransform lacks {method}."
        )


anchor_fields = getattr(
    AnchorFrames,
    "__dataclass_fields__",
    {},
)

dynamic_fields = getattr(
    DynamicHeadlampFrame,
    "__dataclass_fields__",
    {},
)


if "T_H0_from_W" not in anchor_fields:
    raise SystemExit(
        "FAIL: AnchorFrames no longer "
        "contains T_H0_from_W."
    )


if "T_Ht_from_W" not in dynamic_fields:
    raise SystemExit(
        "FAIL: DynamicHeadlampFrame no longer "
        "contains T_Ht_from_W."
    )


# -------------------------------------------------
# Synthetic closed-form moving-headlamp gate
#
# Headlamp:
#   x_W = 0 -> 1 m in 0.1 s
#
# Actor:
#   x_W = 10 -> 11 m in 0.1 s
#
# Relative measurement:
#   r = 10 -> 10 m
#
# Naive H_t differencing => 0 m/s (WRONG)
# H0 reconstruction       => 10 m/s (CORRECT)
# -------------------------------------------------

sequence = AlgorithmObservationSequence(
    scenario_id="block32-moving-headlamp",
    frames=(
        make_frame(
            timestamp_s=0.0,
            detections=(
                make_detection(
                    key="frame0",
                    range_m=10.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    el_rad=0.0,
                ),
            ),
        ),
        make_frame(
            timestamp_s=0.1,
            detections=(
                make_detection(
                    key="frame1",
                    range_m=10.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    el_rad=0.0,
                ),
            ),
        ),
    ),
)


associated = associate_estimated_gnn(
    sequence
)


if len(associated.tracks) != 1:
    raise SystemExit(
        "FAIL: synthetic actor did not "
        "remain one associated track."
    )


context = FrameTransformContext(
    T_H0_from_W=identity_transform(),

    T_Ht_from_W_by_frame=(
        T_sensor_from_W(
            sensor_origin_W=(
                0.0,
                0.0,
                0.0,
            )
        ),
        T_sensor_from_W(
            sensor_origin_W=(
                1.0,
                0.0,
                0.0,
            )
        ),
    ),
)


observations = (
    cartesianize_associated_track(
        associated.tracks[0],
        context=context,
    )
)


if len(observations) != 2:
    raise SystemExit(
        "FAIL: expected two Cartesian "
        "observations."
    )


x0 = observations[0].position_H0_m[0]
x1 = observations[1].position_H0_m[0]


if abs(x0 - 10.0) > 1e-10:
    raise SystemExit(
        "FAIL: incorrect H0 reconstruction "
        "at first frame."
    )


if abs(x1 - 11.0) > 1e-10:
    raise SystemExit(
        "FAIL: ego/headlamp motion was not "
        "properly compensated."
    )


state = estimate_causal_cv_state(
    observations
)


if abs(
    state.velocity_H0_mps[0]
    -
    10.0
) > 1e-10:
    raise SystemExit(
        "FAIL: causal velocity is incorrect."
    )


if state.annotated_velocity_used:
    raise SystemExit(
        "FAIL: annotated velocity used."
    )


if state.future_information_used:
    raise SystemExit(
        "FAIL: future information used."
    )


prediction = predict_cv(
    state,
    horizons_s=(
        0.1,
        0.3,
        0.5,
        1.0,
    ),
)


expected_x = (
    12.0,
    14.0,
    16.0,
    21.0,
)


for point, expected in zip(
    prediction.points,
    expected_x,
):
    if abs(
        point.position_H0_m[0]
        -
        expected
    ) > 1e-10:
        raise SystemExit(
            "FAIL: CV closed-form "
            "prediction mismatch."
        )


# Measurement uncertainty must survive into
# both Cartesian observations and CV state.

if not any(
    observations[-1]
    .position_covariance_H0_m2[i][i]
    > 0.0
    for i in range(3)
):
    raise SystemExit(
        "FAIL: Stage2 measurement "
        "uncertainty was lost."
    )


if not any(
    state.covariance_6x6[i][i] > 0.0
    for i in range(6)
):
    raise SystemExit(
        "FAIL: causal state covariance "
        "was lost."
    )


config = json.loads(
    (
        ROOT
        / "configs"
        / "stage3_cv.json"
    ).read_text(
        encoding="utf-8"
    )
)


if config[
    "input"
][
    "annotated_womd_velocity"
]:
    raise SystemExit(
        "FAIL: config permits annotated "
        "velocity."
    )


if config[
    "input"
][
    "future_information"
]:
    raise SystemExit(
        "FAIL: config permits future "
        "information."
    )


if config[
    "coordinates"
][
    "state_and_prediction_frame"
] != "H0":
    raise SystemExit(
        "FAIL: CV state is not frozen "
        "to H0."
    )


if not config[
    "measurement_uncertainty"
][
    "stage2_R_t_used"
]:
    raise SystemExit(
        "FAIL: Stage2 R_t disabled."
    )


if config[
    "prediction_horizons_s"
] != [
    0.1,
    0.3,
    0.5,
    1.0,
]:
    raise SystemExit(
        "FAIL: canonical short horizons "
        "changed."
    )


sha, count = implementation_hash()


report = {
    "stage": 3,
    "block": "3.2",
    "status": "PASS",

    "baseline": "constant_velocity",

    "input": {
        "source": (
            "Stage2 noisy algorithm-facing "
            "detections after estimated "
            "association"
        ),
        "annotated_velocity_used": False,
        "future_information_used": False
    },

    "coordinates": {
        "measurement_frame": "Ht",
        "canonical_state_frame": "H0",
        "conversion": "Ht -> W -> H0",
        "frozen_stage1_transform_api": (
            "PASS"
        )
    },

    "measurement_uncertainty": {
        "Stage2_R_t_preserved": True,
        "spherical_to_cartesian": (
            "Jacobian propagation"
        ),
        "covariance_rotated_to_H0": True,
        "state_covariance_generated": True
    },

    "closed_form_gate": {
        "relative_range_m": [
            10.0,
            10.0
        ],
        "headlamp_world_x_m": [
            0.0,
            1.0
        ],
        "actor_H0_x_m": [
            x0,
            x1
        ],
        "recovered_velocity_x_mps": (
            state.velocity_H0_mps[0]
        ),
        "expected_velocity_x_mps": 10.0,
        "status": "PASS"
    },

    "prediction_horizons_s": [
        0.1,
        0.3,
        0.5,
        1.0
    ],

    "sub_100ms_ground_truth_claim": False,

    "formal_real_validation_metrics": (
        "DEFERRED_TO_STAGE3_EVALUATION_BLOCK"
    ),

    "implementation": {
        "files": count,
        "sha256": sha
    }
}


report_path = (
    ROOT
    / "reports"
    / "block32_cv_gate.json"
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
    "===== Stage3 Block 3.2 gate ====="
)

print(
    "Stage1 transform API      = PASS"
)

print(
    "measurement frame         = Ht"
)

print(
    "canonical state frame     = H0"
)

print(
    "headlamp motion comp.     = PASS"
)

print(
    "Stage2 R_t propagation    = PASS"
)

print(
    "velocity source           = "
    "past-only position difference"
)

print(
    "annotated velocity        = NO"
)

print(
    "future information        = NO"
)

print(
    "synthetic recovered vx    =",
    state.velocity_H0_mps[0],
)

print(
    "expected vx               = 10.0"
)

print(
    "CV horizons               = "
    "0.1 / 0.3 / 0.5 / 1.0 s"
)

print(
    "CV closed-form prediction = PASS"
)

print(
    "implementation files      =",
    count,
)

print(
    "implementation SHA256     =",
    sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    report_path,
)

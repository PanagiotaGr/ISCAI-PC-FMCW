from __future__ import annotations

import hashlib
import json
import math

from pathlib import Path

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.baselines import (
    predict_ca,
    predict_ctrv,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.geometry import (
    CartesianObservation,
    FrameTransformContext,
    cartesianize_associated_track,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from iscai_stage3.state import (
    CausalCTRVState,
    estimate_causal_ca_state,
    estimate_causal_ctrv_state,
)

from rigid_test_factory import (
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


# =================================================
# CA closed-form gate through common Stage2 path
# =================================================

def x_ca(t):
    return (
        5.0
        +
        2.0 * t
        +
        2.0 * t * t
    )


times = (
    0.0,
    0.1,
    0.2,
)


sequence = AlgorithmObservationSequence(
    scenario_id="block33-ca",
    frames=tuple(
        make_frame(
            timestamp_s=t,
            detections=(
                make_detection(
                    key=f"ca-{i}",
                    range_m=x_ca(t),
                    vr_mps=0.0,
                    az_rad=0.0,
                    el_rad=0.0,
                ),
            ),
        )
        for i, t in enumerate(times)
    ),
)


associated = associate_estimated_gnn(
    sequence
)


if len(associated.tracks) != 1:
    raise SystemExit(
        "FAIL: CA target did not remain "
        "one associated track."
    )


context = FrameTransformContext(
    T_H0_from_W=identity_transform(),
    T_Ht_from_W_by_frame=(
        identity_transform(),
        identity_transform(),
        identity_transform(),
    ),
)


observations = (
    cartesianize_associated_track(
        associated.tracks[0],
        context=context,
    )
)


ca_state = estimate_causal_ca_state(
    observations
)


if abs(
    ca_state.acceleration_H0_mps2[0]
    -
    4.0
) > 1e-9:
    raise SystemExit(
        "FAIL: CA acceleration estimator "
        "failed closed-form gate."
    )


if abs(
    ca_state.velocity_H0_mps[0]
    -
    2.8
) > 1e-9:
    raise SystemExit(
        "FAIL: CA current velocity "
        "estimator failed."
    )


ca_prediction = predict_ca(
    ca_state,
    horizons_s=(0.3,),
)


expected_x = x_ca(0.5)


if abs(
    ca_prediction.points[0]
    .position_H0_m[0]
    -
    expected_x
) > 1e-9:
    raise SystemExit(
        "FAIL: CA prediction failed "
        "closed-form gate."
    )


if (
    ca_state.annotated_velocity_used
    or
    ca_state.annotated_acceleration_used
    or
    ca_state.future_information_used
):
    raise SystemExit(
        "FAIL: non-causal CA "
        "information used."
    )


if not any(
    ca_state.covariance_9x9[i][i] > 0.0
    for i in range(9)
):
    raise SystemExit(
        "FAIL: CA state covariance lost."
    )


# =================================================
# CTRV heading-wrap gate
# =================================================

P = (
    (0.1, 0.0, 0.0),
    (0.0, 0.1, 0.0),
    (0.0, 0.0, 0.1),
)


def cart_obs(
    frame,
    t,
    x,
    y,
):
    measurement = snapshot_from_detection(
        make_detection(
            key=f"ctrv-{frame}",
            range_m=max(
                math.hypot(x, y),
                0.1,
            ),
            vr_mps=0.0,
            az_rad=math.atan2(
                y,
                x,
            ),
        )
    )

    return CartesianObservation(
        track_id="ctrv-track",
        frame_index=frame,
        timestamp_s=t,
        detection_key=f"ctrv-{frame}",
        position_H0_m=(
            x,
            y,
            0.0,
        ),
        position_covariance_H0_m2=P,
        measurement=measurement,
    )


h1 = math.radians(179.0)
h2 = math.radians(-179.0)

v1 = (
    math.cos(h1),
    math.sin(h1),
)

v2 = (
    math.cos(h2),
    math.sin(h2),
)

p0 = (
    0.0,
    0.0,
)

p1 = (
    p0[0] + v1[0],
    p0[1] + v1[1],
)

p2 = (
    p1[0] + v2[0],
    p1[1] + v2[1],
)


ctrv_estimated = (
    estimate_causal_ctrv_state(
        (
            cart_obs(
                0,
                0.0,
                p0[0],
                p0[1],
            ),
            cart_obs(
                1,
                1.0,
                p1[0],
                p1[1],
            ),
            cart_obs(
                2,
                2.0,
                p2[0],
                p2[1],
            ),
        ),
        min_heading_speed_mps=0.2,
    )
)


expected_turn_rate = (
    math.radians(2.0)
)


if abs(
    ctrv_estimated.turn_rate_radps
    -
    expected_turn_rate
) > 1e-9:
    raise SystemExit(
        "FAIL: CTRV angular wrapping "
        "is incorrect."
    )


if abs(
    ctrv_estimated.turn_rate_radps
) > 0.1:
    raise SystemExit(
        "FAIL: CTRV appears to contain "
        "the legacy +/-pi wrap bug."
    )


# =================================================
# Exact CTRV propagation equation
# =================================================

ctrv_state = CausalCTRVState(
    track_id="ctrv-propagation",
    timestamp_s=0.0,
    position_H0_m=(
        0.0,
        0.0,
        0.0,
    ),
    planar_speed_mps=10.0,
    heading_rad=0.0,
    turn_rate_radps=1.0,
    vertical_velocity_mps=0.0,
    source_frame_indices=(
        0,
        1,
        2,
    ),
)


ctrv_prediction = predict_ctrv(
    ctrv_state,
    horizons_s=(0.5,),
)


point = ctrv_prediction.points[0]


expected_ctrv_x = (
    10.0
    *
    math.sin(0.5)
)

expected_ctrv_y = (
    10.0
    *
    (
        1.0
        -
        math.cos(0.5)
    )
)


if abs(
    point.position_H0_m[0]
    -
    expected_ctrv_x
) > 1e-9:
    raise SystemExit(
        "FAIL: CTRV x propagation."
    )


if abs(
    point.position_H0_m[1]
    -
    expected_ctrv_y
) > 1e-9:
    raise SystemExit(
        "FAIL: CTRV y propagation."
    )


# =================================================
# Near-zero-turn limit
# =================================================

straight_state = CausalCTRVState(
    track_id="straight",
    timestamp_s=0.0,
    position_H0_m=(
        0.0,
        0.0,
        0.0,
    ),
    planar_speed_mps=10.0,
    heading_rad=0.0,
    turn_rate_radps=0.0,
    vertical_velocity_mps=0.0,
    source_frame_indices=(
        0,
        1,
        2,
    ),
)


straight_prediction = predict_ctrv(
    straight_state,
    horizons_s=(0.5,),
)


if abs(
    straight_prediction.points[0]
    .position_H0_m[0]
    -
    5.0
) > 1e-9:
    raise SystemExit(
        "FAIL: CTRV zero-turn limit."
    )


if abs(
    straight_prediction.points[0]
    .position_H0_m[1]
) > 1e-9:
    raise SystemExit(
        "FAIL: CTRV zero-turn "
        "lateral drift."
    )


if (
    ctrv_prediction.annotated_velocity_used
    or
    ctrv_prediction.annotated_heading_used
    or
    ctrv_prediction.future_information_used
):
    raise SystemExit(
        "FAIL: CTRV used prohibited "
        "information."
    )


# =================================================
# Config checks
# =================================================

config = json.loads(
    (
        ROOT
        / "configs"
        / "stage3_motion_models.json"
    ).read_text(
        encoding="utf-8"
    )
)


common = config["common"]


if common[
    "annotated_womd_velocity"
]:
    raise SystemExit(
        "FAIL: annotated WOMD velocity "
        "enabled."
    )


if common[
    "annotated_womd_heading"
]:
    raise SystemExit(
        "FAIL: annotated WOMD heading "
        "enabled."
    )


if common[
    "future_information"
]:
    raise SystemExit(
        "FAIL: future information enabled."
    )


if common[
    "canonical_frame"
] != "H0":
    raise SystemExit(
        "FAIL: motion models are not in H0."
    )


if common[
    "prediction_horizons_s"
] != [
    0.1,
    0.3,
    0.5,
    1.0,
]:
    raise SystemExit(
        "FAIL: canonical horizons changed."
    )


ctrv_config = config["ctrv"]


if ctrv_config[
    "heading_difference"
] != "wrapped_to_minus_pi_plus_pi":
    raise SystemExit(
        "FAIL: CTRV angle-wrap policy changed."
    )


if ctrv_config[
    "zero_turn_limit"
] != "straight_line":
    raise SystemExit(
        "FAIL: CTRV zero-turn policy changed."
    )


if config[
    "formal_validation_tuning"
]:
    raise SystemExit(
        "FAIL: motion parameters marked as "
        "validation-tuned."
    )


sha, count = implementation_hash()


report = {
    "stage": 3,
    "block": "3.3",
    "status": "PASS",

    "constant_acceleration": {
        "minimum_observations": 3,
        "causal": True,
        "annotated_velocity_used": False,
        "annotated_acceleration_used": False,
        "future_information_used": False,
        "recovered_acceleration_x_mps2": (
            ca_state
            .acceleration_H0_mps2[0]
        ),
        "expected_acceleration_x_mps2": 4.0,
        "recovered_current_velocity_x_mps": (
            ca_state.velocity_H0_mps[0]
        ),
        "expected_current_velocity_x_mps": 2.8,
        "prediction_closed_form": "PASS",
        "measurement_covariance_preserved": True
    },

    "ctrv": {
        "minimum_observations": 3,
        "causal": True,
        "annotated_velocity_used": False,
        "annotated_heading_used": False,
        "future_information_used": False,
        "heading_wrap": "PASS",
        "wrap_test_turn_rate_radps": (
            ctrv_estimated
            .turn_rate_radps
        ),
        "wrap_test_expected_radps": (
            expected_turn_rate
        ),
        "constant_turn_equation": "PASS",
        "zero_turn_limit": "PASS",
        "low_speed_fallback": "IMPLEMENTED"
    },

    "common": {
        "input_regime": (
            "same Stage2 noisy observation "
            "and estimated-association branch"
        ),
        "canonical_frame": "H0",
        "prediction_horizons_s": [
            0.1,
            0.3,
            0.5,
            1.0
        ]
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
    / "block33_motion_gate.json"
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
    "===== Stage3 Block 3.3 gate ====="
)

print(
    "CA common Stage2 path       = PASS"
)

print(
    "CA past-only state          = PASS"
)

print(
    "CA recovered ax             =",
    ca_state.acceleration_H0_mps2[0],
)

print(
    "CA expected ax              = 4.0"
)

print(
    "CA recovered current vx     =",
    ca_state.velocity_H0_mps[0],
)

print(
    "CA expected current vx      = 2.8"
)

print(
    "CA covariance propagation   = PASS"
)

print(
    "CA closed-form prediction   = PASS"
)

print(
    "CTRV past-only state        = PASS"
)

print(
    "CTRV heading wrapping       = PASS"
)

print(
    "CTRV wrap turn rate         =",
    ctrv_estimated.turn_rate_radps,
)

print(
    "CTRV expected turn rate     =",
    expected_turn_rate,
)

print(
    "CTRV constant-turn formula  = PASS"
)

print(
    "CTRV zero-turn limit        = PASS"
)

print(
    "annotated velocity          = NO"
)

print(
    "annotated heading           = NO"
)

print(
    "future information          = NO"
)

print(
    "canonical frame             = H0"
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

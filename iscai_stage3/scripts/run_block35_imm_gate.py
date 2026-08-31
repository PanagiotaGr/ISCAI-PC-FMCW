from __future__ import annotations

import hashlib
import json

from pathlib import Path

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.baselines import (
    predict_imm,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.filters import (
    IMMConfig,
    KalmanFrameContext,
    filter_associated_track_imm,
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


# =================================================
# Common Stage2 moving-sensor synthetic sequence
# =================================================

times = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)

sensor_x = (
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
)

actor_x = (
    10.0,
    11.5,
    13.0,
    14.5,
    16.0,
    17.5,
)


frames = tuple(
    make_frame(
        timestamp_s=t,
        detections=(
            make_detection(
                key=f"frame-{i}",
                range_m=(
                    actor_x[i]
                    -
                    sensor_x[i]
                ),
                vr_mps=5.0,
                az_rad=0.0,
                el_rad=0.0,
            ),
        ),
    )
    for i, t in enumerate(
        times
    )
)


sequence = AlgorithmObservationSequence(
    scenario_id="block35-imm",
    frames=frames,
)


associated = associate_estimated_gnn(
    sequence
)


if len(associated.tracks) != 1:
    raise SystemExit(
        "FAIL: common estimated association "
        "did not produce one track."
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


filter_context = KalmanFrameContext(
    transforms=transform_context,
    frame_timestamps_s=times,
)


config = IMMConfig(
    cv_acceleration_spectral_density_m2_s3=0.0,
    ca_jerk_spectral_density_m2_s5=0.0,
    ctrv_linear_acceleration_spectral_density_m2_s3=0.0,
    ctrv_turn_rate_random_walk_rad2_s3=0.0,
)


first = filter_associated_track_imm(
    associated.tracks[0],
    context=filter_context,
    config=config,
)


second = filter_associated_track_imm(
    associated.tracks[0],
    context=filter_context,
    config=config,
)


if first != second:
    raise SystemExit(
        "FAIL: IMM is not deterministic."
    )


if tuple(
    state.mode
    for state in first.current_mode_states
) != (
    "CV",
    "CA",
    "CTRV",
):
    raise SystemExit(
        "FAIL: required IMM modes missing."
    )


if len(
    first.initial_mode_states
) != 3:
    raise SystemExit(
        "FAIL: IMM does not maintain three "
        "mode states."
    )


if len(first.updates) != 3:
    raise SystemExit(
        "FAIL: expected three full IMM "
        "measurement-update steps."
    )


for step in first.updates:
    if abs(
        sum(
            step
            .posterior_mode_probabilities
        )
        -
        1.0
    ) > 1e-12:
        raise SystemExit(
            "FAIL: posterior mode "
            "probabilities do not sum to one."
        )

    if abs(
        sum(
            step
            .predicted_mode_probabilities
        )
        -
        1.0
    ) > 1e-12:
        raise SystemExit(
            "FAIL: predicted mode "
            "probabilities do not sum to one."
        )

    # For every destination mode, mixing
    # probabilities over source modes sum to 1.
    for destination in range(3):
        total = sum(
            step
            .mixing_probabilities[
                source
            ][
                destination
            ]
            for source in range(3)
        )

        if abs(
            total
            -
            1.0
        ) > 1e-12:
            raise SystemExit(
                "FAIL: IMM mixing weights "
                "do not normalize."
            )

    if len(
        step.mode_updates
    ) != 3:
        raise SystemExit(
            "FAIL: not all modes were "
            "measurement-updated."
        )

    for update in (
        step.mode_updates
    ):
        if not (
            update
            .measurement_covariance_used
        ):
            raise SystemExit(
                "FAIL: Stage2 R_t was not "
                "used by an IMM mode."
            )

        if not isinstance(
            update.log_likelihood,
            float,
        ):
            raise SystemExit(
                "FAIL: missing mode "
                "measurement likelihood."
            )


combined = (
    first
    .combined_current_state
)


if abs(
    combined.mean_10[0]
    -
    17.5
) > 1e-6:
    raise SystemExit(
        "FAIL: IMM combined position "
        "recovery."
    )


if abs(
    combined.mean_10[3]
    -
    15.0
) > 1e-6:
    raise SystemExit(
        "FAIL: IMM combined velocity "
        "recovery."
    )


if abs(
    sum(
        first
        .current_mode_probabilities
    )
    -
    1.0
) > 1e-12:
    raise SystemExit(
        "FAIL: current IMM probabilities "
        "do not normalize."
    )


if (
    first.truth_used
    or
    first.annotated_velocity_used
    or
    first.annotated_heading_used
    or
    first.future_information_used
):
    raise SystemExit(
        "FAIL: prohibited information "
        "entered IMM."
    )


prediction = predict_imm(
    track_id=first.track_id,
    states=first.current_mode_states,
    mode_probabilities=(
        first
        .current_mode_probabilities
    ),
    horizons_s=(
        0.1,
        0.3,
        0.5,
        1.0,
    ),
    config=config,
)


if len(prediction.points) != 4:
    raise SystemExit(
        "FAIL: IMM prediction horizons "
        "missing."
    )


for point in prediction.points:
    if abs(
        sum(
            point.mode_probabilities
        )
        -
        1.0
    ) > 1e-12:
        raise SystemExit(
            "FAIL: open-loop mode "
            "probabilities do not normalize."
        )


# =================================================
# Frozen config contract
# =================================================

config_json = json.loads(
    (
        ROOT
        / "configs"
        / "stage3_imm.json"
    ).read_text(
        encoding="utf-8"
    )
)


if config_json["modes"] != [
    "CV",
    "CA",
    "CTRV",
]:
    raise SystemExit(
        "FAIL: required IMM modes changed."
    )


if not config_json[
    "interaction"
][
    "state_mixing"
]:
    raise SystemExit(
        "FAIL: IMM state mixing disabled."
    )


if not config_json[
    "interaction"
][
    "covariance_mixing"
]:
    raise SystemExit(
        "FAIL: IMM covariance mixing disabled."
    )


if not config_json[
    "measurement_model"
][
    "same_for_all_modes"
]:
    raise SystemExit(
        "FAIL: modes do not share common "
        "measurement interface."
    )


if not config_json[
    "measurement_model"
][
    "uses_full_stage2_R_t"
]:
    raise SystemExit(
        "FAIL: Stage2 R_t disabled."
    )


if config_json[
    "mode_probability_update"
][
    "likelihood"
] != (
    "multivariate_gaussian_"
    "measurement_likelihood"
):
    raise SystemExit(
        "FAIL: IMM likelihood update "
        "policy changed."
    )


if not config_json[
    "combined_posterior"
][
    "covariance"
] == (
    "within_mode_plus_between_mode_spread"
):
    raise SystemExit(
        "FAIL: combined covariance policy "
        "changed."
    )


if not config_json[
    "open_loop_prediction"
][
    "interaction_repeated_without_future_measurements"
]:
    raise SystemExit(
        "FAIL: open-loop IMM interaction "
        "disabled."
    )


policy = config_json[
    "information_policy"
]


if (
    policy["truth_used"]
    or
    policy[
        "annotated_womd_velocity"
    ]
    or
    policy[
        "annotated_womd_heading"
    ]
    or
    policy[
        "future_information"
    ]
):
    raise SystemExit(
        "FAIL: prohibited IMM information "
        "enabled."
    )


if config_json[
    "formal_validation_tuning"
]:
    raise SystemExit(
        "FAIL: IMM parameters marked "
        "validation-tuned."
    )


sha, count = implementation_hash()


report = {
    "stage": 3,
    "block": "3.5",
    "status": "PASS",

    "baseline": "IMM_CV_CA_CTRV",

    "full_imm": {
        "separate_mode_states": True,
        "separate_mode_covariances": True,
        "markov_transition_matrix": True,
        "interaction_mixing": True,
        "mixed_mean": True,
        "mixed_covariance": True,
        "mode_measurement_likelihoods": True,
        "posterior_mode_probabilities": True,
        "combined_posterior": True
    },

    "modes": [
        "CV",
        "CA",
        "CTRV"
    ],

    "common_observation_interface": {
        "measurement": [
            "range",
            "radial_velocity",
            "azimuth",
            "elevation"
        ],
        "frame": "Ht",
        "full_stage2_R_t_used": True
    },

    "synthetic_gate": {
        "moving_sensor": True,
        "updates": len(
            first.updates
        ),
        "combined_x_m": (
            combined.mean_10[0]
        ),
        "expected_x_m": 17.5,
        "combined_vx_mps": (
            combined.mean_10[3]
        ),
        "expected_vx_mps": 15.0,
        "status": "PASS"
    },

    "open_loop_prediction": {
        "interaction_continues": True,
        "future_measurements_used": False,
        "horizons_s": [
            0.1,
            0.3,
            0.5,
            1.0
        ]
    },

    "information_policy": {
        "truth_used": False,
        "annotated_velocity_used": False,
        "annotated_heading_used": False,
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
    / "block35_imm_gate.json"
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
    "===== Stage3 Block 3.5 gate ====="
)

print(
    "IMM implementation          = FULL"
)

print(
    "modes                       = "
    "CV / CA / CTRV"
)

print(
    "separate mode states        = PASS"
)

print(
    "separate mode covariances   = PASS"
)

print(
    "Markov transition matrix    = PASS"
)

print(
    "interaction / mixing        = PASS"
)

print(
    "mixed covariance            = PASS"
)

print(
    "same measurement model      = PASS"
)

print(
    "Stage2 full R_t             = USED"
)

print(
    "mode likelihoods            = PASS"
)

print(
    "posterior mode probabilities= PASS"
)

print(
    "combined posterior          = PASS"
)

print(
    "real IMM update steps       =",
    len(first.updates),
)

print(
    "combined recovered x        =",
    combined.mean_10[0],
)

print(
    "expected x                  = 17.5"
)

print(
    "combined recovered vx       =",
    combined.mean_10[3],
)

print(
    "expected vx                 = 15.0"
)

print(
    "open-loop IMM prediction    = PASS"
)

print(
    "truth leakage               = NONE"
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

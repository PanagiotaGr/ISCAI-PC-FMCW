from __future__ import annotations

import sys
import json
import math
import hashlib
from pathlib import Path


ROOT = Path("/home/agni/waymo")

for stage_src in [
    ROOT / "iscai_stage0" / "src",
    ROOT / "iscai_stage1" / "src",
    ROOT / "iscai_stage2" / "src",
    ROOT / "iscai_stage3" / "src",
    ROOT / "iscai_stage4" / "src",
]:
    value = str(stage_src)

    if value not in sys.path:
        sys.path.insert(0, value)


from iscai_stage3.io.womd_reader import (
    read_scenario_by_id,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


from iscai_stage3.baselines.causal_state import (
    estimate_causal_motion_state,
)

from iscai_stage3.geometry.spherical import (
    spherical_position_jacobian,
)

from iscai_stage3.uncertainty.propagation import (
    propagate_position_covariance,
)

from iscai_stage2.pc_fmcw.covariance import (
    measurement_covariance_from_part_a_crlb,
)

from iscai_stage4.predictor.stage4_input import (
    Stage4ActorInput,
)

from iscai_stage4.predictor.deterministic_gru import (
    DeterministicGRUPredictor,
)

from iscai_stage4.predictor.gaussian_gru import (
    GaussianGRUPredictor,
)

from iscai_stage4.predictor.gmm_predictor import (
    GMMPredictor,
)


SCENARIO_ID = "b85e1bd6cc8e74c0"

ARTIFACT = Path(
    "artifacts/stage4_prediction_artifact.json"
)

REPORT = Path(
    "reports/block4_real_prediction_artifact.json"
)

HORIZON = 10
DT = 0.1


def trajectory_to_list(trajectory):
    return [
        [
            float(p[0]),
            float(p[1]),
            float(p[2]),
        ]
        for p in trajectory
    ]


(
    scenario,
    manifest_record,
    shard,
    record_index,
) = read_scenario_by_id(
    SCENARIO_ID
)


adapted = adapt_causal_womd_scenario(
    scenario
)


scene = build_real_ideal_observation_scene(
    adapted=adapted,
    raw_scenario=scenario,
    include_sdc=False,
)


records = []

actor_count = 0

skipped_invalid = 0
skipped_history = 0
skipped_velocity = 0
skipped_dimensions = 0
skipped_radial_velocity = 0


for actor_series in scene.actors:

    valid = [
        obs
        for obs in actor_series.observations
        if (
            obs.geometry_valid
            and
            obs.actor_position_Ht_m is not None
            and
            obs.range_m is not None
            and
            obs.azimuth_rad is not None
            and
            obs.elevation_rad is not None
        )
    ]

    if len(valid) < 3:
        skipped_invalid += 1
        continue


    current = valid[-1]

    t = current.time_index


    try:
        adapted_actor = adapted.actors[
            actor_series.track_index
        ]
    except Exception:
        skipped_history += 1
        continue


    history = adapted_actor.artifact.history


    if (
        t < 0
        or
        t >= len(history.timestamps_s)
    ):
        skipped_history += 1
        continue


    if not history.state_valid[t]:
        skipped_history += 1
        continue


    if (
        t >= len(history.velocity_W_mps)
        or
        not history.velocity_valid[t]
    ):
        skipped_velocity += 1
        continue


    dimensions = history.dimensions_lwh_m[t]

    if (
        dimensions[0] <= 0.0
        or dimensions[1] <= 0.0
        or dimensions[2] <= 0.0
    ):
        skipped_dimensions += 1
        continue


    if (
        not current.radial_velocity_valid
        or
        current.radial_velocity_mps is None
    ):
        skipped_radial_velocity += 1
        continue


    # ========================================================
    # Causal Ht-frame motion state
    #
    # Position, velocity and heading are all estimated from
    # the same last three causal Ht observations.
    # No future sample is consumed.
    # ========================================================

    causal_positions = tuple(
        obs.actor_position_Ht_m
        for obs in valid[-3:]
    )

    causal_timestamps = tuple(
        obs.timestamp_s
        for obs in valid[-3:]
    )


    state = estimate_causal_motion_state(
        positions=causal_positions,
        timestamps=causal_timestamps,
    )


    # ========================================================
    # Frozen Stage2 clean CRLB baseline
    #
    # Measurement order:
    # [range, radial_velocity, azimuth, elevation]
    #
    # For Cartesian position covariance we select:
    # [range, azimuth, elevation]
    #
    # and propagate:
    #
    # Sigma_xyz = J Sigma_rae J^T
    # ========================================================

    crlb_model = (
        measurement_covariance_from_part_a_crlb(
            sensing_snr_db=20.0,
            azimuth_std_rad=math.radians(
                1.0
            ),
            elevation_std_rad=math.radians(
                1.0
            ),
        )
    )


    R4 = crlb_model.covariance.matrix


    covariance_rae = (
        (
            float(R4[0][0]),
            float(R4[0][2]),
            float(R4[0][3]),
        ),
        (
            float(R4[2][0]),
            float(R4[2][2]),
            float(R4[2][3]),
        ),
        (
            float(R4[3][0]),
            float(R4[3][2]),
            float(R4[3][3]),
        ),
    )


    J = spherical_position_jacobian(
        range_m=float(
            current.range_m
        ),
        azimuth_rad=float(
            current.azimuth_rad
        ),
        elevation_rad=float(
            current.elevation_rad
        ),
    )


    measurement_covariance_xyz = (
        propagate_position_covariance(
            J,
            covariance_rae,
        )
    )


    actor_input = Stage4ActorInput(

        time_index=int(t),

        position=(
            float(state.position[0]),
            float(state.position[1]),
            float(state.position[2]),
        ),

        velocity=(
            float(state.velocity[0]),
            float(state.velocity[1]),
            float(state.velocity[2]),
        ),

        heading_rad=float(
            state.heading_rad
        ),

        range_m=float(
            current.range_m
        ),

        radial_velocity_mps=float(
            current.radial_velocity_mps
        ),

        measurement_covariance=(
            measurement_covariance_xyz
        ),

        length_m=float(
            dimensions[0]
        ),

        width_m=float(
            dimensions[1]
        ),

        height_m=float(
            dimensions[2]
        ),

        actor_class=str(
            actor_series.object_class
        ),
    )


    det = DeterministicGRUPredictor(
        horizon=HORIZON,
        dt=DT,
    ).predict(
        actor_input
    )


    records.append(
        {
            "scenario_id":
                SCENARIO_ID,

            "track_index":
                actor_series.track_index,

            "actor_class":
                actor_series.object_class,

            "time_index":
                int(t),

            "model_name":
                "DETERMINISTIC_GRU",

            "trajectory":
                trajectory_to_list(
                    det.positions
                ),

            "uncertainty":
                0.0,

            "causal_only":
                True,

            "future_used":
                False,
        }
    )


    gaussian = GaussianGRUPredictor(
        horizon=HORIZON,
        dt=DT,
    ).predict(
        actor_input
    )


    records.append(
        {
            "scenario_id":
                SCENARIO_ID,

            "track_index":
                actor_series.track_index,

            "actor_class":
                actor_series.object_class,

            "time_index":
                int(t),

            "model_name":
                "GAUSSIAN_GRU",

            "trajectory":
                trajectory_to_list(
                    gaussian.mean
                ),

            "uncertainty":
                0.2,

            "causal_only":
                True,

            "future_used":
                False,
        }
    )


    gmm = GMMPredictor(
        horizon=HORIZON,
        dt=DT,
    ).predict(
        actor_input
    )


    for component in gmm.components:

        records.append(
            {
                "scenario_id":
                    SCENARIO_ID,

                "track_index":
                    actor_series.track_index,

                "actor_class":
                    actor_series.object_class,

                "time_index":
                    int(t),

                "model_name":
                    "GMM_" + component.mode,

                "trajectory":
                    trajectory_to_list(
                        component.trajectory
                    ),

                "uncertainty":
                    float(
                        1.0 -
                        component.probability
                    ),

                "mode_probability":
                    float(
                        component.probability
                    ),

                "causal_only":
                    True,

                "future_used":
                    False,
            }
        )


    actor_count += 1


if actor_count == 0:
    raise RuntimeError(
        "No real WOMD actors produced Stage4 predictions."
    )


if not records:
    raise RuntimeError(
        "No Stage4 predictions produced."
    )


for item in records:

    if item["future_used"]:
        raise RuntimeError(
            "Future leakage."
        )

    if not item["causal_only"]:
        raise RuntimeError(
            "Non-causal Stage4 prediction."
        )

    if len(item["trajectory"]) != HORIZON:
        raise RuntimeError(
            "Invalid prediction horizon."
        )


ARTIFACT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


ARTIFACT.write_text(
    json.dumps(
        records,
        indent=2,
        sort_keys=True,
    )
)


artifact_sha = hashlib.sha256(
    ARTIFACT.read_bytes()
).hexdigest()


models = {}

classes = {}

for item in records:

    model = item["model_name"]

    models[model] = (
        models.get(model, 0) + 1
    )

    actor_class = item["actor_class"]

    classes[actor_class] = (
        classes.get(actor_class, 0) + 1
    )


report = {

    "block":
        "stage4_real_prediction_artifact",

    "scenario_id":
        SCENARIO_ID,

    "actors":
        actor_count,

    "records":
        len(records),

    "models":
        models,

    "classes":
        classes,

    "horizon":
        HORIZON,

    "dt":
        DT,

    "input_source":
        "real_womd_causal",

    "measurement_covariance_bridge":
        "stage2_part_a_crlb_jacobian_propagated_to_cartesian",

    "skipped": {
        "invalid_geometry":
            skipped_invalid,

        "history":
            skipped_history,

        "velocity":
            skipped_velocity,

        "dimensions":
            skipped_dimensions,

        "radial_velocity":
            skipped_radial_velocity,
    },

    "artifact_sha256":
        artifact_sha,

    "causal_only":
        True,

    "future_used":
        False,

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
    "===== Stage4 Real WOMD Prediction Artifact ====="
)

print(
    "scenario =",
    SCENARIO_ID
)

print(
    "actors =",
    actor_count
)

print(
    "records =",
    len(records)
)

print(
    "models =",
    models
)

print(
    "classes =",
    classes
)

print(
    "covariance_bridge = STAGE2_CRLB_PROPAGATED"
)

print(
    "future_used = NO"
)

print(
    "SHA256 =",
    artifact_sha
)

print(
    "STATUS = PASS"
)

print(
    "artifact =",
    ARTIFACT
)

print(
    "report =",
    REPORT
)

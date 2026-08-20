from __future__ import annotations

import json
import math
from pathlib import Path


from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)

from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
)

from iscai_stage2.observations.clean_scene import (
    build_clean_observation_scene,
)

from iscai_stage3.baselines.kalman_ekf import (
    initialize_ekf_from_measurement,
    predict_ekf_cv,
    update_ekf_with_stage2_measurement,
)


MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

SID = "b85e1bd6cc8e74c0"

REPORT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "reports/block3d_kalman_real_smoke.json"
)


# ============================================================
# Frozen real WOMD pilot
# ============================================================

scenario = read_first_scenario(
    MOTION
)

if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected scenario."
    )


adapted = adapt_causal_womd_scenario(
    scenario
)


ideal_scene = (
    build_real_ideal_observation_scene(
        raw_scenario=scenario,
        adapted=adapted,
        include_sdc=False,
    )
)


# ============================================================
# Frozen Stage2 clean sensing baseline
# ============================================================

clean_config = CleanObservationConfig(
    sensing_snr_db=20.0,
    azimuth_std_rad=math.radians(1.0),
    elevation_std_rad=math.radians(1.0),
)


clean_scene = build_clean_observation_scene(
    ideal_scene=ideal_scene,
    config=clean_config,
)


if clean_scene.scenario_id != SID:
    raise RuntimeError(
        "Stage2 clean-scene scenario mismatch."
    )


if clean_scene.measured_fmcw:
    raise RuntimeError(
        "Stage3 must not claim measured FMCW."
    )


# ============================================================
# One independent EKF per actor
# ============================================================

actors_initialized = 0
actors_with_updates = 0

valid_measurements = 0
prediction_steps = 0
measurement_updates = 0

posterior_states = {}


for actor in clean_scene.actors:

    posterior = None
    actor_updates = 0

    previous_timestamp = None

    for record in actor.records:

        # ----------------------------------------------------
        # Preserve causal timestamp ordering.
        # ----------------------------------------------------

        if previous_timestamp is not None:
            if record.timestamp_s <= previous_timestamp:
                raise RuntimeError(
                    "Non-increasing causal timestamps."
                )

        previous_timestamp = record.timestamp_s


        # ----------------------------------------------------
        # Missing/invalid observation:
        # no measurement update.
        # ----------------------------------------------------

        if not record.measurement_valid:
            continue

        measurement = record.measurement

        if measurement is None:
            raise RuntimeError(
                "Valid record missing measurement."
            )

        if measurement.measured_fmcw:
            raise RuntimeError(
                "Measured FMCW leaked into Stage3."
            )

        valid_measurements += 1


        # ----------------------------------------------------
        # First valid observation initializes this actor.
        # ----------------------------------------------------

        if posterior is None:

            posterior = (
                initialize_ekf_from_measurement(
                    measurement
                )
            )

            actors_initialized += 1

            continue


        # ----------------------------------------------------
        # CV prediction to current measurement time.
        # ----------------------------------------------------

        posterior = predict_ekf_cv(
            posterior,
            timestamp_s=(
                measurement.timestamp_s
            ),
            acceleration_noise_variance=1.0,
        )

        prediction_steps += 1


        # ----------------------------------------------------
        # EKF correction using full Stage2:
        #
        # z = [r, vr, azimuth, elevation]
        # R = Stage2 CRLB-conditioned covariance
        # ----------------------------------------------------

        posterior = (
            update_ekf_with_stage2_measurement(
                predicted=posterior,
                observation=measurement,
            )
        )

        measurement_updates += 1
        actor_updates += 1


    if posterior is not None:

        posterior_states[
            actor.track_index
        ] = posterior


    if actor_updates > 0:
        actors_with_updates += 1


# ============================================================
# Gates
# ============================================================

if actors_initialized == 0:
    raise RuntimeError(
        "No actor EKF was initialized."
    )


if measurement_updates == 0:
    raise RuntimeError(
        "No Kalman measurement update executed."
    )


if prediction_steps != measurement_updates:
    raise RuntimeError(
        "Each update must have exactly one prediction."
    )


if len(posterior_states) != actors_initialized:
    raise RuntimeError(
        "Posterior actor-state accounting mismatch."
    )


# ============================================================
# Report
# ============================================================

report = {
    "block":
        "stage3d_kalman_real_smoke",

    "scenario_id":
        SID,

    "causal_frames":
        clean_scene.anchor_index + 1,

    "actors":
        len(clean_scene.actors),

    "actors_initialized":
        actors_initialized,

    "actors_with_updates":
        actors_with_updates,

    "valid_measurements":
        valid_measurements,

    "prediction_steps":
        prediction_steps,

    "measurement_updates":
        measurement_updates,

    "measurement_order": [
        "range",
        "radial_velocity",
        "azimuth",
        "elevation",
    ],

    "measurement_covariance":
        "stage2_full_crlb_conditioned_R_t",

    "sensing_snr_db":
        clean_config.sensing_snr_db,

    "angular_std_deg": {
        "azimuth": 1.0,
        "elevation": 1.0,
    },

    "association_mode":
        "clean_mode_oracle_track_partition_for_baseline_validation",

    "truth_sidecar_used":
        False,

    "future_used":
        False,

    "measured_fmcw":
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
    )
)


print(
    "===== Stage3 Kalman real smoke ====="
)

print(
    "scenario            =",
    SID
)

print(
    "actors              =",
    len(clean_scene.actors)
)

print(
    "actors initialized  =",
    actors_initialized
)

print(
    "actors with updates =",
    actors_with_updates
)

print(
    "valid measurements  =",
    valid_measurements
)

print(
    "prediction steps    =",
    prediction_steps
)

print(
    "measurement updates =",
    measurement_updates
)

print(
    "covariance source   = Stage2 full CRLB R_t"
)

print(
    "truth sidecar used  = NO"
)

print(
    "future used         = NO"
)

print(
    "measured FMCW       = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

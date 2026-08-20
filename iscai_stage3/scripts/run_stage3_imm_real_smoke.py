from __future__ import annotations

import json
import math
from collections import Counter
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

from iscai_stage3.baselines.causal_state import (
    estimate_causal_motion_state,
)

from iscai_stage3.baselines.imm_core import (
    IMMProbabilities,
)

from iscai_stage3.baselines.imm_real_step import (
    imm_real_step,
)

from iscai_stage3.baselines.imm_report import (
    IMMReportRecord,
    report_sha256,
    select_mode,
)

from iscai_stage3.baselines.kalman_core import (
    spherical_to_cartesian,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage3/"
    "reports/block3e_imm_real_smoke.json"
)


# ============================================================
# Real frozen WOMD pilot
# ============================================================

scenario = read_first_scenario(
    MOTION
)

if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected real pilot scenario."
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
# Stage2 clean measurement stream
#
# Mean:
#   clean/noise-free measurement
#
# R_t:
#   CRLB-conditioned Stage2 covariance
#
# No random corruption.
# ============================================================

clean_config = CleanObservationConfig(
    sensing_snr_db=20.0,

    azimuth_std_rad=math.radians(
        1.0
    ),

    elevation_std_rad=math.radians(
        1.0
    ),
)


clean_scene = build_clean_observation_scene(
    ideal_scene=ideal_scene,
    config=clean_config,
)


if clean_scene.measured_fmcw:
    raise RuntimeError(
        "Stage3 must not claim measured FMCW."
    )


# ============================================================
# IMM causal evaluation
#
# IMPORTANT:
#
# To evaluate measurement z_k:
#
#   state estimate uses measurements only through k-1
#
#   prediction:
#       x_(k-1) -> z_k^mode
#
#   comparison:
#       z_k^mode vs observed z_k
#
# Therefore current z_k never enters the state used to
# predict itself.
# ============================================================

records: list[
    IMMReportRecord
] = []

mode_counts = Counter()

actors_with_imm_updates = 0

valid_measurements = 0

imm_updates = 0


for actor in clean_scene.actors:

    probabilities = IMMProbabilities(
        cv=0.34,
        ca=0.33,
        ctrv=0.33,
    )


    # Only past measured Cartesian positions.
    #
    # At the beginning of each loop iteration these lists
    # contain samples strictly before the current record.
    past_positions = []

    past_timestamps = []

    actor_updates = 0


    for record in actor.records:

        if not record.measurement_valid:
            continue


        measurement_object = (
            record.measurement
        )

        if measurement_object is None:
            raise RuntimeError(
                "Valid Stage2 record has no measurement."
            )


        if measurement_object.measured_fmcw:
            raise RuntimeError(
                "Measured FMCW leaked into IMM."
            )


        valid_measurements += 1


        current_position = (
            spherical_to_cartesian(
                measurement_object.range_m,
                measurement_object.azimuth_rad,
                measurement_object.elevation_rad,
            )
        )


        # ----------------------------------------------------
        # Need three strictly past samples for:
        #
        # velocity(k-1)
        # acceleration(k-1)
        # heading(k-1)
        # yaw-rate(k-1)
        #
        # Current measurement is NOT included yet.
        # ----------------------------------------------------

        if len(
            past_positions
        ) < 3:

            past_positions.append(
                current_position
            )

            past_timestamps.append(
                measurement_object.timestamp_s
            )

            continue


        motion_state = (
            estimate_causal_motion_state(
                positions=tuple(
                    past_positions[-3:]
                ),

                timestamps=tuple(
                    past_timestamps[-3:]
                ),
            )
        )


        dt = (
            measurement_object.timestamp_s
            -
            past_timestamps[-1]
        )


        if dt <= 0.0:
            raise RuntimeError(
                "Non-positive causal IMM timestep."
            )


        measurement = (
            measurement_object
            .measurement_vector()
        )


        covariance = (
            measurement_object
            .covariance
            .matrix
        )


        # ----------------------------------------------------
        # One IMM probability step.
        #
        # All three models see:
        #   - same past-only motion state
        #   - same current measurement
        #   - same Stage2 R_t
        # ----------------------------------------------------

        result = imm_real_step(
            probabilities=probabilities,

            position=motion_state.position,

            velocity=motion_state.velocity,

            acceleration=(
                motion_state.acceleration
            ),

            heading_rad=(
                motion_state.heading_rad
            ),

            yaw_rate_radps=(
                motion_state.yaw_rate_radps
            ),

            measurement=measurement,

            covariance=covariance,

            dt=dt,
        )


        probabilities = (
            result.probabilities
        )


        selected = select_mode(
            cv=probabilities.cv,
            ca=probabilities.ca,
            ctrv=probabilities.ctrv,
        )


        records.append(
            IMMReportRecord(
                track_index=(
                    actor.track_index
                ),

                time_index=(
                    record.time_index
                ),

                cv_probability=(
                    probabilities.cv
                ),

                ca_probability=(
                    probabilities.ca
                ),

                ctrv_probability=(
                    probabilities.ctrv
                ),

                cv_log_likelihood=(
                    result.cv_log_likelihood
                ),

                ca_log_likelihood=(
                    result.ca_log_likelihood
                ),

                ctrv_log_likelihood=(
                    result.ctrv_log_likelihood
                ),

                selected_mode=selected,
            )
        )


        mode_counts[
            selected
        ] += 1

        imm_updates += 1
        actor_updates += 1


        # ----------------------------------------------------
        # Only AFTER evaluation does z_k become history.
        # ----------------------------------------------------

        past_positions.append(
            current_position
        )

        past_timestamps.append(
            measurement_object.timestamp_s
        )


    if actor_updates > 0:
        actors_with_imm_updates += 1


# ============================================================
# Gates
# ============================================================

if valid_measurements <= 0:
    raise RuntimeError(
        "No valid Stage2 measurements."
    )


if imm_updates <= 0:
    raise RuntimeError(
        "No IMM update executed."
    )


if len(records) != imm_updates:
    raise RuntimeError(
        "IMM audit accounting mismatch."
    )


for record in records:

    probability_sum = (
        record.cv_probability
        +
        record.ca_probability
        +
        record.ctrv_probability
    )

    if abs(
        probability_sum - 1.0
    ) > 1e-10:
        raise RuntimeError(
            "IMM probabilities do not sum to one."
        )


audit_sha256 = report_sha256(
    tuple(records)
)


# ============================================================
# Report
# ============================================================

report = {
    "block":
        "stage3e_imm_real_smoke",

    "scenario_id":
        SID,

    "actors":
        len(clean_scene.actors),

    "actors_with_imm_updates":
        actors_with_imm_updates,

    "causal_frames":
        clean_scene.anchor_index + 1,

    "valid_measurements":
        valid_measurements,

    "imm_updates":
        imm_updates,

    "selected_mode_counts":
        dict(
            sorted(
                mode_counts.items()
            )
        ),

    "initial_mode_probabilities": {
        "CV": 0.34,
        "CA": 0.33,
        "CTRV": 0.33,
    },

    "state_history_semantics":
        "strictly_past_stage2_clean_measurements",

    "measurement_source":
        "stage2_clean_crlb_conditioned",

    "measurement_covariance":
        "stage2_full_R_t",

    "sensing_snr_db":
        clean_config.sensing_snr_db,

    "truth_sidecar_used":
        False,

    "future_used":
        False,

    "measured_fmcw":
        False,

    "imm_audit_records":
        len(records),

    "imm_audit_sha256":
        audit_sha256,

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
    "===== Stage3 IMM real likelihood gate ====="
)

print(
    "scenario              =",
    SID
)

print(
    "actors                =",
    len(clean_scene.actors)
)

print(
    "actors with updates   =",
    actors_with_imm_updates
)

print(
    "valid measurements    =",
    valid_measurements
)

print(
    "IMM updates           =",
    imm_updates
)

print(
    "selected mode counts  =",
    dict(
        sorted(
            mode_counts.items()
        )
    )
)

print(
    "state history         = STRICTLY PAST"
)

print(
    "covariance source     = Stage2 full R_t"
)

print(
    "truth sidecar used    = NO"
)

print(
    "future used           = NO"
)

print(
    "measured FMCW         = NO"
)

print(
    "IMM audit SHA256      =",
    audit_sha256
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT
)

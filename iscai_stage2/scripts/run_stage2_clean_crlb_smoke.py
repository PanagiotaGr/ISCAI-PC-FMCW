from __future__ import annotations

import json
import math
from pathlib import Path

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage2.observations.clean_measurement import (
    CleanObservationConfig,
)
from iscai_stage2.observations.clean_scene import (
    build_clean_observation_scene,
    clean_observation_scene_sha256,
)
from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/"
    "reports/stage2/"
    "block2c_clean_crlb_smoke.json"
)


# Explicit clean-baseline settings.
#
# 20 dB is used here only as a controlled Part-A-compatible
# CRLB reference point, not as a physical per-actor SNR claim.
CONFIG = CleanObservationConfig(
    sensing_snr_db=20.0,

    # Explicit angular assumptions for this smoke.
    azimuth_std_rad=math.radians(
        1.0
    ),
    elevation_std_rad=math.radians(
        1.0
    ),
)


def build(raw):
    adapted = adapt_causal_womd_scenario(
        raw
    )

    ideal = (
        build_real_ideal_observation_scene(
            raw_scenario=raw,
            adapted=adapted,
            include_sdc=False,
        )
    )

    clean = build_clean_observation_scene(
        ideal_scene=ideal,
        config=CONFIG,
    )

    return ideal, clean


scenario = read_first_scenario(
    MOTION
)

if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected pilot scenario."
    )

ideal, clean = build(
    scenario
)

clean_hash_before = (
    clean_observation_scene_sha256(
        clean
    )
)


total_records = 0
valid_measurements = 0
invalid_geometry = 0
invalid_vr = 0

range_std_values = []
vr_std_values = []

for actor in clean.actors:
    for record in actor.records:
        total_records += 1

        if record.measurement_valid:
            valid_measurements += 1

            measurement = (
                record.measurement
            )

            covariance_model = (
                record.covariance_model
            )

            if (
                measurement is None
                or covariance_model is None
            ):
                raise RuntimeError(
                    "Valid record missing measurement "
                    "or covariance."
                )

            if (
                measurement.measured_fmcw
            ):
                raise RuntimeError(
                    "False measured-FMCW claim."
                )

            if (
                measurement.measurement_vector()
                != (
                    measurement.range_m,
                    measurement.radial_velocity_mps,
                    measurement.azimuth_rad,
                    measurement.elevation_rad,
                )
            ):
                raise RuntimeError(
                    "Measurement-vector order changed."
                )

            range_std_values.append(
                covariance_model
                .crlb.range_std_m
            )

            vr_std_values.append(
                covariance_model
                .crlb.radial_velocity_std_mps
            )

        else:
            if (
                record.invalid_reason
                == "invalid_geometry"
            ):
                invalid_geometry += 1

            elif (
                record.invalid_reason
                == "invalid_radial_velocity"
            ):
                invalid_vr += 1

            else:
                raise RuntimeError(
                    "Unexpected invalid reason."
                )


if total_records != 143:
    raise RuntimeError(
        f"Expected 143 records, got "
        f"{total_records}"
    )

# Block1C established 86 valid vr records.
if valid_measurements != 86:
    raise RuntimeError(
        "Clean measurement count does not match "
        "the frozen ideal-observation gate: "
        f"{valid_measurements}"
    )

if any(
    record.sensor_noise_applied
    for actor in clean.actors
    for record in actor.records
):
    raise RuntimeError(
        "Noise leaked into clean mode."
    )


# ============================================================
# Future mutation invariance
# ============================================================

mutated = scenario_pb2.Scenario()
mutated.CopyFrom(
    scenario
)

future_index = (
    int(mutated.current_time_index)
    + 1
)

future = (
    mutated.tracks[0]
    .states[future_index]
)

future.center_x += 9999.0
future.center_y -= 8888.0
future.velocity_x += 777.0
future.velocity_y -= 666.0
future.heading += 1.25
future.valid = not future.valid

_, clean_mutated = build(
    mutated
)

clean_hash_after = (
    clean_observation_scene_sha256(
        clean_mutated
    )
)

if (
    clean_hash_before
    != clean_hash_after
):
    raise RuntimeError(
        "Future information leaked into clean "
        "Stage-2 measurement pipeline."
    )


report = {
    "status": "PASS",

    "block": (
        "stage2_block2c_clean_crlb_conditioned"
    ),

    "scenario_id": SID,

    "records": {
        "total": total_records,
        "valid_measurements":
            valid_measurements,
        "invalid_geometry":
            invalid_geometry,
        "invalid_radial_velocity":
            invalid_vr,
    },

    "mode": (
        "clean_crlb_conditioned"
    ),

    "sensing_snr": {
        "model": "fixed_snr",
        "snr_db": 20.0,
        "semantics": (
            "explicit_experimental_baseline_"
            "not_physical_per_actor_snr_claim"
        ),
    },

    "crlb": {
        "source": (
            "frozen_part_a_eq7"
        ),
        "range_std_m":
            range_std_values[0],
        "radial_velocity_std_mps":
            vr_std_values[0],
    },

    "angular_uncertainty": {
        "source": (
            "explicit_stage2_assumption"
        ),
        "azimuth_std_deg": 1.0,
        "elevation_std_deg": 1.0,
    },

    "noise_applied": False,
    "missed_detections_applied": False,
    "false_alarms_applied": False,

    "measurement_covariance_attached":
        True,

    "measured_fmcw": False,

    "future_dependency": False,

    "future_mutation_hash_identical": (
        clean_hash_before
        == clean_hash_after
    ),

    "clean_scene_sha256": (
        clean_hash_before
    ),
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
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage2 clean CRLB-conditioned smoke ====="
)

print(
    "total records        =",
    total_records,
)

print(
    "valid measurements   =",
    valid_measurements,
)

print(
    "invalid geometry     =",
    invalid_geometry,
)

print(
    "invalid radial vel   =",
    invalid_vr,
)

print(
    "sensing SNR          = 20 dB "
    "(explicit baseline)"
)

print(
    "range CRLB std [m]   =",
    range_std_values[0],
)

print(
    "vr CRLB std [m/s]    =",
    vr_std_values[0],
)

print(
    "noise                = NO"
)

print(
    "missed detections    = NO"
)

print(
    "false alarms         = NO"
)

print(
    "future hash identical=",
    clean_hash_before
    == clean_hash_after,
)

print(
    "measured FMCW        = NO"
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)

from __future__ import annotations

import json
import math
from pathlib import Path

from iscai_stage0.womd_proto_io import read_first_scenario
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_real_ideal_observation_scene,
)

from iscai_stage3.pipeline.real_beam_series import (
    build_stage3_beam_series_from_stage2,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)


REPORT = Path(
    "/home/agni/waymo/iscai_stage3/reports/"
    "block3a_real_beam_series_smoke.json"
)


scenario = read_first_scenario(
    MOTION
)


if scenario.scenario_id != SID:
    raise RuntimeError(
        "Unexpected scenario id."
    )


adapted = adapt_causal_womd_scenario(
    scenario
)


scene = build_real_ideal_observation_scene(
    raw_scenario=scenario,
    adapted=adapted,
    include_sdc=False,
)


total_samples = 0
valid_geometry = 0
invalid_geometry = 0

ranges = []


for actor in scene.actors:

    beam_series = (
        build_stage3_beam_series_from_stage2(
            actor.observations
        )
    )


    if len(beam_series) != 11:
        raise RuntimeError(
            "Causal series length mismatch."
        )


    previous_time = -1


    for sample in beam_series:

        if sample.time_index <= previous_time:
            raise RuntimeError(
                "Non causal ordering."
            )

        previous_time = sample.time_index

        total_samples += 1


        if sample.geometry_valid:

            valid_geometry += 1

            if (
                sample.range_m is None
                or sample.azimuth_rad is None
                or sample.elevation_rad is None
            ):
                raise RuntimeError(
                    "Valid geometry missing beam values."
                )


            values = (
                sample.range_m,
                sample.azimuth_rad,
                sample.elevation_rad,
            )

            if not all(
                math.isfinite(v)
                for v in values
            ):
                raise RuntimeError(
                    "Non finite beam coordinate."
                )

            ranges.append(
                sample.range_m
            )


        else:

            invalid_geometry += 1

            if (
                sample.range_m is not None
                or sample.azimuth_rad is not None
                or sample.elevation_rad is not None
            ):
                raise RuntimeError(
                    "Invalid geometry leaked values."
                )


        if sample.measured_fmcw:
            raise RuntimeError(
                "Stage3 must remain upstream of FMCW."
            )


report = {
    "block": "stage3_block3a_real_beam_series",
    "scenario_id": SID,
    "actors": len(scene.actors),
    "samples": total_samples,
    "valid_geometry": valid_geometry,
    "invalid_geometry": invalid_geometry,
    "range_min": min(ranges),
    "range_max": max(ranges),
    "measured_fmcw": False,
    "future_used": False,
    "status": "PASS",
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
    "===== Stage3 real beam-series smoke ====="
)

for k, v in report.items():
    print(
        f"{k:20} = {v}"
    )

print(
    "STATUS = PASS"
)

print(
    "report =", REPORT
)

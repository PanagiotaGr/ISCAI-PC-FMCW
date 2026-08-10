from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from iscai_stage0.womd_proto_io import read_first_scenario
from iscai_stage1.actors.artifact import (
    CausalActorArtifact,
)
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage1.artifacts.hashing import causal_actor_sha256


SCENARIO_ID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/reports/stage1a/"
    "block4d_real_actor_adapter.json"
)


scenario = read_first_scenario(MOTION)

if scenario.scenario_id != SCENARIO_ID:
    raise RuntimeError(
        f"Unexpected first scenario: {scenario.scenario_id}"
    )

adapted = adapt_causal_womd_scenario(scenario)

if adapted.scenario_id != SCENARIO_ID:
    raise RuntimeError("Adapted scenario-id mismatch.")

if adapted.anchor_index != 10:
    raise RuntimeError(
        f"Unexpected anchor={adapted.anchor_index}"
    )

if len(adapted.actors) != 14:
    raise RuntimeError(
        f"Expected 14 actors, got {len(adapted.actors)}"
    )


def count_role(name: str) -> int:
    return sum(
        bool(getattr(actor.artifact.roles, name))
        for actor in adapted.actors
    )


# ------------------------------------------------------------
# Core causal contract
# ------------------------------------------------------------

for actor in adapted.actors:
    h = actor.artifact.history

    if len(h.timestamps_s) != 11:
        raise RuntimeError("Non-causal timestamp length.")

    if len(h.state_valid) != 11:
        raise RuntimeError("Non-causal validity length.")

    if len(h.velocity_valid) != 11:
        raise RuntimeError("Velocity length mismatch.")

    numeric = actor.artifact.realistic_numeric_features()

    forbidden = {
        "track_id",
        "scenario_id",
        "tracks_to_predict",
        "objects_of_interest",
        "velocity_x",
        "velocity_y",
    }

    overlap = forbidden & set(numeric)

    if overlap:
        raise RuntimeError(
            f"Forbidden numeric features present: {overlap}"
        )


# ------------------------------------------------------------
# SDC H0 geometry contract
# ------------------------------------------------------------

sdc = adapted.actors[adapted.sdc_track_index]

if sdc.anchor_center_H0_m is None:
    raise RuntimeError("SDC H0 anchor geometry missing.")

# H0 origin is the headlamp surrogate, therefore the SDC center
# should lie approximately L/2 behind it on H0 x.
x, y, z = sdc.anchor_center_H0_m

if not x < 0.0:
    raise RuntimeError(
        f"SDC center should be behind H0 origin; x={x}"
    )

if abs(y) > 1e-8 or abs(z) > 1e-8:
    raise RuntimeError(
        "Baseline H0 SDC center should have ~zero y/z: "
        f"{sdc.anchor_center_H0_m}"
    )


# ------------------------------------------------------------
# Hash every causal actor artifact
# ------------------------------------------------------------

actor_hashes = [
    causal_actor_sha256(actor.artifact)
    for actor in adapted.actors
]

if len(set(actor_hashes)) != len(actor_hashes):
    raise RuntimeError(
        "Unexpected duplicate causal actor artifact hash."
    )


# ------------------------------------------------------------
# Compact report
# ------------------------------------------------------------

actor_rows = []

for actor, digest in zip(
    adapted.actors,
    actor_hashes,
):
    artifact: CausalActorArtifact = actor.artifact

    actor_rows.append(
        {
            "track_index": actor.track_index,
            "track_id": artifact.metadata.track_id,
            "object_class": artifact.metadata.object_class,
            "is_sdc": artifact.metadata.is_sdc,
            "roles": asdict(artifact.roles),
            "anchor_velocity_valid": bool(
                artifact.history.velocity_valid[
                    adapted.anchor_index
                ]
            ),
            "anchor_center_H0_m": (
                list(actor.anchor_center_H0_m)
                if actor.anchor_center_H0_m is not None
                else None
            ),
            "anchor_bearing_H0_rad": (
                actor.anchor_bearing_H0_rad
            ),
            "receiver_geometry_valid": (
                actor.receiver_geometry_H0 is not None
                and bool(
                    actor.receiver_geometry_H0
                    .receiver_geometry_valid
                )
            ),
            "causal_actor_sha256": digest,
        }
    )


report = {
    "status": "PASS",
    "block": "stage1a_block4d_real_actor_adapter",
    "scenario_id": adapted.scenario_id,
    "anchor_index": adapted.anchor_index,
    "actor_count": len(adapted.actors),
    "role_counts": {
        "context": count_role("is_context_actor"),
        "anchor_valid": count_role("is_anchor_valid"),
        "forecast_target_candidate": count_role(
            "is_forecasting_target_candidate"
        ),
        "receiver_candidate": count_role(
            "is_receiver_candidate"
        ),
    },
    "artifact_semantics": (
        adapted.actors[0]
        .artifact.semantics.artifact_semantics
    ),
    "sensor_realistic": (
        adapted.actors[0]
        .artifact.semantics.sensor_realistic
    ),
    "sdc_anchor_center_H0_m": list(
        sdc.anchor_center_H0_m
    ),
    "actors": actor_rows,
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

print("===== Stage1A real actor adapter =====")
print("scenario_id =", adapted.scenario_id)
print("actors      =", len(adapted.actors))
print("role_counts =", report["role_counts"])
print(
    "SDC H0      =",
    report["sdc_anchor_center_H0_m"],
)
print(
    "semantics   =",
    report["artifact_semantics"],
)
print(
    "sensor_realistic =",
    report["sensor_realistic"],
)
print("STATUS = PASS")
print("report =", REPORT)

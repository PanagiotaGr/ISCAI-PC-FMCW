#!/usr/bin/env python3
"""Run Stage-1A canonicalization on exactly one WOMD scenario.

No corpus scan.
No LiDAR.
No maps.
No training.
No large materialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE0_SRC = PROJECT_ROOT / "iscai_stage0" / "src"
STAGE1_SRC = PROJECT_ROOT / "iscai_stage1" / "src"

sys.path.insert(0, str(STAGE0_SRC))
sys.path.insert(0, str(STAGE1_SRC))


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfrecord",
        required=True,
        type=Path,
        help="Explicit compact WOMD TFRecord path. No directory scan.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    try:
        from iscai_stage0.womd_proto_io import read_first_scenario
    except ModuleNotFoundError as exc:
        print(
            "STAGE1A_RUNTIME_IMPORT_BLOCKED: "
            f"{exc}. Use the interpreter/environment containing "
            "the frozen Stage-0 Waymo protobuf package.",
            file=sys.stderr,
        )
        return 2

    from iscai_stage1.io.womd_adapter import (
        build_stage1a_scene_from_womd,
        stage1a_scene_causal_sha256,
    )

    scenario = read_first_scenario(args.tfrecord)

    scene = build_stage1a_scene_from_womd(scenario)

    report = {
        "status": "pass",
        "artifact_semantics": "causal_womd_annotation_upstream",
        "sensor_realistic": False,
        "scenario_id": scene.scenario_id,
        "anchor_index": scene.anchor_index,
        "sdc_track_index": scene.sdc_track_index,
        "actor_count": len(scene.actors),
        "context_actor_count": sum(
            actor.roles.is_context_actor
            for actor in scene.actors
        ),
        "anchor_valid_count": sum(
            actor.roles.is_anchor_valid
            for actor in scene.actors
        ),
        "forecasting_target_candidate_count": sum(
            actor.roles.is_forecasting_target_candidate
            for actor in scene.actors
        ),
        "receiver_candidate_count": sum(
            actor.roles.is_receiver_candidate
            for actor in scene.actors
        ),
        "anchor_velocity_valid_count": sum(
            actor.history.velocity_valid[actor.history.anchor_index]
            for actor in scene.actors
        ),
        "receiver_geometry_count": len(scene.receivers),
        "T_W_from_H0": {
            "rotation": scene.anchor_frames.T_W_from_H0.rotation,
            "translation": scene.anchor_frames.T_W_from_H0.translation,
        },
        "causal_artifact_sha256": stage1a_scene_causal_sha256(
            scene
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
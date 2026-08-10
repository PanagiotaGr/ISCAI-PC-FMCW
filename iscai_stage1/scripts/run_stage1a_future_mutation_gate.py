from __future__ import annotations

import json
from pathlib import Path

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage0.womd_proto_io import (
    read_first_scenario,
)
from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage1.artifacts.hashing import (
    causal_actor_sha256,
)
from iscai_stage1.artifacts.scenario_hashing import (
    combined_causal_artifact_sha256,
    future_dynamic_map_sha256,
    future_labels_sha256,
    manifest_record_sha256,
)
from iscai_stage1.maps.hashing import (
    causal_map_sha256,
)
from iscai_stage1.maps.womd_adapter import (
    adapt_causal_womd_map,
)


SID = "b85e1bd6cc8e74c0"

MOTION = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0/"
    "validation/motion/"
    "paired-from-validation.tfrecord-00000-of-00150"
)

MANIFEST = Path(
    "/home/agni/waymo/iscai_data_prep/"
    "manifests/selected_validation.jsonl"
)

REPORT = Path(
    "/home/agni/waymo/iscai_stage1/"
    "reports/stage1a/"
    "block7a_future_mutation_gate.json"
)


def causal_hashes(scenario):
    actors = adapt_causal_womd_scenario(
        scenario
    )

    maps = adapt_causal_womd_map(
        scenario,
        actors.frames.T_H0_from_W,
    )

    actor_hashes = tuple(
        causal_actor_sha256(
            actor.artifact
        )
        for actor in actors.actors
    )

    map_hash = causal_map_sha256(
        maps
    )

    combined = (
        combined_causal_artifact_sha256(
            scenario_id=str(
                scenario.scenario_id
            ),
            anchor_index=int(
                scenario.current_time_index
            ),
            actor_hashes=actor_hashes,
            map_hash=map_hash,
        )
    )

    return (
        actor_hashes,
        map_hash,
        combined,
    )


# ------------------------------------------------------------
# Frozen deterministic selection
# ------------------------------------------------------------

with MANIFEST.open("rb") as f:
    raw_manifest_line = f.readline()

entry = json.loads(
    raw_manifest_line.decode("utf-8")
)

if entry["scenario_id"] != SID:
    raise RuntimeError(
        "Unexpected frozen pilot manifest record."
    )

selection_hash_before = (
    manifest_record_sha256(
        raw_manifest_line
    )
)


# ------------------------------------------------------------
# Original scenario
# ------------------------------------------------------------

original = read_first_scenario(
    MOTION
)

if original.scenario_id != SID:
    raise RuntimeError(
        "Original scenario-id mismatch."
    )

anchor = int(
    original.current_time_index
)

(
    actor_hashes_before,
    map_hash_before,
    causal_hash_before,
) = causal_hashes(original)

labels_hash_before = (
    future_labels_sha256(original)
)

future_map_hash_before = (
    future_dynamic_map_sha256(original)
)


# ------------------------------------------------------------
# Deep-copy and mutate ONLY non-causal content
# ------------------------------------------------------------

mutated = scenario_pb2.Scenario()
mutated.CopyFrom(original)

future_index = anchor + 1

if future_index >= len(
    mutated.timestamps_seconds
):
    raise RuntimeError(
        "Pilot has no future state to mutate."
    )

if not mutated.tracks:
    raise RuntimeError(
        "Pilot contains no tracks."
    )

if future_index >= len(
    mutated.tracks[0].states
):
    raise RuntimeError(
        "Track 0 lacks first future state."
    )


future_state = (
    mutated.tracks[0]
    .states[future_index]
)

original_future_x = float(
    future_state.center_x
)

original_future_valid = bool(
    future_state.valid
)

# Future trajectory mutation.
future_state.center_x = (
    original_future_x + 1234.5
)
future_state.valid = (
    not original_future_valid
)

# Future dynamic-map mutation:
# retain exactly the causal prefix 0..anchor.
del mutated.dynamic_map_states[
    anchor + 1:
]

# Benchmark/evaluation metadata mutation.
# These MUST NOT influence causal realistic artifacts.
mutated.ClearField(
    "tracks_to_predict"
)
mutated.ClearField(
    "objects_of_interest"
)


# ------------------------------------------------------------
# Recompute causal branch
# ------------------------------------------------------------

(
    actor_hashes_after,
    map_hash_after,
    causal_hash_after,
) = causal_hashes(mutated)

labels_hash_after = (
    future_labels_sha256(mutated)
)

future_map_hash_after = (
    future_dynamic_map_sha256(mutated)
)

# Selection artifact itself was not touched.
selection_hash_after = (
    manifest_record_sha256(
        raw_manifest_line
    )
)


# ------------------------------------------------------------
# Strict gates
# ------------------------------------------------------------

if selection_hash_before != selection_hash_after:
    raise RuntimeError(
        "Causal selection hash changed."
    )

if actor_hashes_before != actor_hashes_after:
    changed = [
        index
        for index, (before, after)
        in enumerate(
            zip(
                actor_hashes_before,
                actor_hashes_after,
            )
        )
        if before != after
    ]

    raise RuntimeError(
        "Future mutation leaked into causal actors: "
        f"changed_tracks={changed}"
    )

if map_hash_before != map_hash_after:
    raise RuntimeError(
        "Future mutation leaked into causal map."
    )

if causal_hash_before != causal_hash_after:
    raise RuntimeError(
        "Combined causal artifact hash changed."
    )

if labels_hash_before == labels_hash_after:
    raise RuntimeError(
        "Future-label hash failed to change."
    )

if future_map_hash_before == future_map_hash_after:
    raise RuntimeError(
        "Future dynamic-map hash failed to change."
    )


report = {
    "status": "PASS",
    "block": "stage1a_block7a_future_mutation_gate",
    "scenario_id": SID,
    "anchor_index": anchor,

    "mutations": {
        "future_track_index": 0,
        "future_time_index": future_index,
        "future_center_x_before": (
            original_future_x
        ),
        "future_center_x_after": (
            float(future_state.center_x)
        ),
        "future_valid_before": (
            original_future_valid
        ),
        "future_valid_after": (
            bool(future_state.valid)
        ),
        "future_dynamic_map_removed": True,
        "tracks_to_predict_cleared": True,
        "objects_of_interest_cleared": True,
    },

    "gates": {
        "causal_selection_hash_identical": (
            selection_hash_before
            == selection_hash_after
        ),
        "all_causal_actor_hashes_identical": (
            actor_hashes_before
            == actor_hashes_after
        ),
        "causal_map_hash_identical": (
            map_hash_before
            == map_hash_after
        ),
        "combined_causal_artifact_hash_identical": (
            causal_hash_before
            == causal_hash_after
        ),
        "future_labels_hash_changed": (
            labels_hash_before
            != labels_hash_after
        ),
        "future_dynamic_map_hash_changed": (
            future_map_hash_before
            != future_map_hash_after
        ),
    },

    "hashes": {
        "selection_before": (
            selection_hash_before
        ),
        "selection_after": (
            selection_hash_after
        ),

        "causal_artifact_before": (
            causal_hash_before
        ),
        "causal_artifact_after": (
            causal_hash_after
        ),

        "causal_map_before": (
            map_hash_before
        ),
        "causal_map_after": (
            map_hash_after
        ),

        "future_labels_before": (
            labels_hash_before
        ),
        "future_labels_after": (
            labels_hash_after
        ),

        "future_dynamic_map_before": (
            future_map_hash_before
        ),
        "future_dynamic_map_after": (
            future_map_hash_after
        ),
    },

    "actor_count": len(
        actor_hashes_before
    ),

    "artifact_semantics": (
        "causal_womd_annotation_upstream"
    ),
    "sensor_realistic": False,

    "forbidden_realistic_dependencies": {
        "future_actor_states": False,
        "future_dynamic_map": False,
        "tracks_to_predict": False,
        "objects_of_interest": False,
        "womd_annotated_velocity": False,
    },
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


print("===== Stage1A future-mutation gate =====")
print(
    "selection hash identical =",
    report["gates"][
        "causal_selection_hash_identical"
    ],
)
print(
    "actor hashes identical   =",
    report["gates"][
        "all_causal_actor_hashes_identical"
    ],
)
print(
    "map hash identical       =",
    report["gates"][
        "causal_map_hash_identical"
    ],
)
print(
    "causal artifact identical=",
    report["gates"][
        "combined_causal_artifact_hash_identical"
    ],
)
print(
    "labels hash changed      =",
    report["gates"][
        "future_labels_hash_changed"
    ],
)
print(
    "future map hash changed  =",
    report["gates"][
        "future_dynamic_map_hash_changed"
    ],
)
print("STATUS = PASS")
print("report =", REPORT)

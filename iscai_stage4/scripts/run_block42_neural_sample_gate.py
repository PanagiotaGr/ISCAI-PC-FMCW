from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
import importlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.clean_scene import (
    build_clean_observation_scene,
)

from iscai_stage2.observations.degraded_scene import (
    build_degraded_observation_scene,
    degraded_algorithm_sha256,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_causal_dynamic_headlamp_frames,
    build_real_ideal_observation_scene,
)

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
)

from iscai_stage3.observations import (
    algorithm_sequence_from_degraded_scene,
)

from iscai_stage4.data import (
    FEATURE_DIM,
    MAP_CONTEXT_DIM,
    attach_supervision,
    build_causal_scene_inputs,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE2 = (
    ROOT
    / "iscai_stage2"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

TRAIN_MOTION = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0/"
      "training/motion"
)

BLOCK40 = (
    STAGE4
    / "reports/block40_bootstrap_gate.json"
)

BLOCK41 = (
    STAGE4
    / "reports/block41_training_split_gate.json"
)

FIT = (
    STAGE4
    / "artifacts/block41/fit.jsonl"
)

DEVELOPMENT = (
    STAGE4
    / "artifacts/block41/development.jsonl"
)

CALIBRATION = (
    STAGE4
    / "artifacts/block41/calibration.jsonl"
)

STAGE2_PROFILE = (
    STAGE2
    / "scripts/"
      "run_stage2_degraded_scene_smoke.py"
)

REPORT = (
    STAGE4
    / "reports/"
      "block42_neural_sample_gate.json"
)

CONFIG = (
    STAGE4
    / "configs/"
      "stage4_neural_sample_contract.json"
)

LOG = (
    STAGE4
    / "docs/implementation_log.md"
)

EXPECTED_FIT_SHA = (
    "284a61c877d937deb07137345beaabf3165690138785f9cc41f81655066d5276"
)

EXPECTED_DEVELOPMENT_SHA = (
    "e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c"
)

EXPECTED_CALIBRATION_SHA = (
    "e003dd5c4d5a253729700b12c3754c47ffb99dadfa035b46627f01ec91eed4be"
)

PILOT_PER_PARTITION = 4
MIN_FREE_GIB = 250.0


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_sha(
    value,
):
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def read_manifest(
    path: Path,
):
    return tuple(
        json.loads(line)
        for line in (
            path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )


def read_training_scenario(
    record,
):
    path = (
        TRAIN_MOTION
        /
        record[
            "motion_shard"
        ]
    )

    offset = int(
        record[
            "compact_record_offset"
        ]
    )

    expected_payload = int(
        record[
            "payload_length"
        ]
    )

    with path.open(
        "rb"
    ) as stream:
        stream.seek(
            offset
        )

        length_bytes = (
            stream.read(8)
        )

        if len(
            length_bytes
        ) != 8:
            raise RuntimeError(
                "Invalid TFRecord length "
                "header."
            )

        payload_length = (
            struct.unpack(
                "<Q",
                length_bytes,
            )[0]
        )

        if (
            int(
                payload_length
            )
            !=
            expected_payload
        ):
            raise RuntimeError(
                "Stored payload length "
                "does not match shard."
            )

        if len(
            stream.read(4)
        ) != 4:
            raise RuntimeError(
                "Missing TFRecord "
                "length CRC."
            )

        payload = stream.read(
            payload_length
        )

        if len(
            payload
        ) != payload_length:
            raise RuntimeError(
                "Incomplete TFRecord payload."
            )

        if len(
            stream.read(4)
        ) != 4:
            raise RuntimeError(
                "Missing TFRecord "
                "data CRC."
            )

    scenario = (
        scenario_pb2.Scenario()
    )

    scenario.ParseFromString(
        payload
    )

    if (
        scenario.scenario_id
        !=
        record["scenario_id"]
    ):
        raise RuntimeError(
            "Scenario ID mismatch."
        )

    return scenario


def load_frozen_stage2_configs():
    """
    Resolve the already-frozen Stage2
    CLEAN_CONFIG and DEGRADED_CONFIG
    directly from its source assignments.

    The Stage2 smoke script itself is
    not executed.
    """

    source = (
        STAGE2_PROFILE
        .read_text(
            encoding="utf-8"
        )
    )

    tree = ast.parse(
        source,
        filename=str(
            STAGE2_PROFILE
        ),
    )

    namespace = {
        "__builtins__":
            __builtins__,
    }

    for node in tree.body:
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                module = (
                    importlib
                    .import_module(
                        alias.name
                    )
                )

                namespace[
                    alias.asname
                    or
                    alias.name
                    .split(".")[0]
                ] = module

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if (
                node.module
                ==
                "__future__"
            ):
                continue

            module = (
                importlib
                .import_module(
                    node.module
                )
            )

            for alias in node.names:
                if alias.name == "*":
                    continue

                namespace[
                    alias.asname
                    or
                    alias.name
                ] = getattr(
                    module,
                    alias.name,
                )

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            if isinstance(
                node,
                ast.Assign,
            ):
                if (
                    len(node.targets)
                    !=
                    1
                    or
                    not isinstance(
                        node.targets[0],
                        ast.Name,
                    )
                ):
                    continue

                name = (
                    node.targets[0]
                    .id
                )

                value_node = (
                    node.value
                )

            else:
                if not isinstance(
                    node.target,
                    ast.Name,
                ):
                    continue

                name = (
                    node.target.id
                )

                value_node = (
                    node.value
                )

            if value_node is None:
                continue

            try:
                value = eval(
                    compile(
                        ast.Expression(
                            value_node
                        ),
                        str(
                            STAGE2_PROFILE
                        ),
                        "eval",
                    ),
                    namespace,
                    namespace,
                )

            except Exception:
                continue

            namespace[name] = value

    if (
        "CLEAN_CONFIG"
        not in namespace
        or
        "DEGRADED_CONFIG"
        not in namespace
    ):
        raise RuntimeError(
            "Could not recover frozen "
            "Stage2 degraded profile."
        )

    return (
        namespace[
            "CLEAN_CONFIG"
        ],
        namespace[
            "DEGRADED_CONFIG"
        ],
    )


def build_real_causal_inputs(
    scenario,
    *,
    clean_config,
    degraded_config,
):
    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    ideal = (
        build_real_ideal_observation_scene(
            raw_scenario=scenario,
            adapted=adapted,
            include_sdc=False,
        )
    )

    clean = (
        build_clean_observation_scene(
            ideal_scene=ideal,
            config=clean_config,
        )
    )

    degraded = (
        build_degraded_observation_scene(
            clean_scene=clean,
            config=degraded_config,
        )
    )

    algorithm_hash = (
        degraded_algorithm_sha256(
            degraded
        )
    )

    sequence = (
        algorithm_sequence_from_degraded_scene(
            scenario_id=(
                scenario.scenario_id
            ),
            scene=degraded,
        )
    )

    associated = (
        associate_estimated_gnn(
            sequence
        )
    )

    dynamic_frames, _ = (
        build_causal_dynamic_headlamp_frames(
            adapted
        )
    )

    expected = (
        int(
            scenario.current_time_index
        )
        +
        1
    )

    if (
        len(dynamic_frames)
        !=
        expected
        or
        any(
            frame is None
            for frame
            in dynamic_frames
        )
    ):
        raise RuntimeError(
            "Invalid causal headlamp "
            "frame sequence."
        )

    timestamps = tuple(
        float(x)
        for x
        in scenario.timestamps_seconds[
            :expected
        ]
    )

    context = (
        FrameTransformContext(
            T_H0_from_W=(
                adapted.frames
                .T_H0_from_W
            ),
            T_Ht_from_W_by_frame=tuple(
                frame.T_Ht_from_W
                for frame
                in dynamic_frames
            ),
        )
    )

    scene_inputs = (
        build_causal_scene_inputs(
            associated.tracks,
            scenario_id=(
                scenario.scenario_id
            ),
            context=context,
            timestamps_s=(
                timestamps
            ),
        )
    )

    return {
        "adapted":
            adapted,

        "degraded":
            degraded,

        "algorithm_hash":
            algorithm_hash,

        "associated_track_count":
            len(
                associated.tracks
            ),

        "scene_inputs":
            scene_inputs,
    }


def mutate_future(
    scenario,
):
    clone = (
        scenario_pb2.Scenario()
    )

    clone.CopyFrom(
        scenario
    )

    anchor = int(
        clone.current_time_index
    )

    for track in (
        clone.tracks
    ):
        for index in range(
            anchor + 1,
            len(
                track.states
            ),
        ):
            state = (
                track.states[
                    index
                ]
            )

            state.center_x = (
                float(
                    state.center_x
                )
                +
                1234.5
            )

            state.center_y = (
                float(
                    state.center_y
                )
                -
                987.5
            )

            state.center_z = (
                float(
                    state.center_z
                )
                +
                13.0
            )

            state.valid = (
                not bool(
                    state.valid
                )
            )

    return clone


def sample_payload_hashes(
    samples,
):
    return tuple(
        (
            sample.prediction_id,
            sample
            .model_payload_sha256(),
        )
        for sample in samples
    )


def sample_label_hashes(
    samples,
):
    return tuple(
        (
            sample.prediction_id,
            sample
            .future_label
            .sha256(),
        )
        for sample in samples
    )


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    files = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in (
                path.parts
            ):
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE4
                )
            )
    )

    digest = sha256()

    for path in files:
        relative = str(
            path.relative_to(
                STAGE4
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )

        digest.update(
            b"\0"
        )

    return (
        len(files),
        digest.hexdigest(),
    )


print(
    "============================================================"
)
print(
    "STAGE 4 — BLOCK 4.2 "
    "REAL NEURAL SAMPLE CONTRACT"
)
print(
    "============================================================"
)


# ============================================================
# Frozen upstream evidence
# ============================================================

block40 = json.loads(
    BLOCK40.read_text(
        encoding="utf-8"
    )
)

block41 = json.loads(
    BLOCK41.read_text(
        encoding="utf-8"
    )
)

if block40[
    "status"
] != "PASS":
    raise SystemExit(
        "FAIL: Block4.0 not PASS."
    )

if block41[
    "status"
] != "PASS":
    raise SystemExit(
        "FAIL: Block4.1 not PASS."
    )

if (
    file_sha256(FIT)
    !=
    EXPECTED_FIT_SHA
):
    raise SystemExit(
        "FAIL: fit manifest changed."
    )

if (
    file_sha256(
        DEVELOPMENT
    )
    !=
    EXPECTED_DEVELOPMENT_SHA
):
    raise SystemExit(
        "FAIL: development "
        "manifest changed."
    )

if (
    file_sha256(
        CALIBRATION
    )
    !=
    EXPECTED_CALIBRATION_SHA
):
    raise SystemExit(
        "FAIL: calibration "
        "manifest changed."
    )

print(
    "Block4.0 upstream          = PASS"
)

print(
    "Block4.1 frozen manifests  = PASS"
)


# ============================================================
# Causal-module static boundary
# ============================================================

causal_source = (
    STAGE4
    / "src/iscai_stage4/"
      "data/neural_inputs.py"
).read_text(
    encoding="utf-8"
)

for token in (
    "tracks_to_predict",
    "objects_of_interest",
    "truth_sidecar",
    "scenario_pb2",
):
    if token in causal_source:
        raise SystemExit(
            "FAIL: forbidden token "
            f"in causal model-input "
            f"builder: {token}"
        )

print(
    "causal builder WOMD truth   = NONE"
)

print(
    "tracks_to_predict input     = NO"
)

print(
    "objects_of_interest input   = NO"
)


# ============================================================
# Frozen Stage2 config
# ============================================================

(
    clean_config,
    degraded_config,
) = load_frozen_stage2_configs()

print(
    "frozen Stage2 degraded mode = PASS"
)


# ============================================================
# Deterministic real 12-scenario pilot
# ============================================================

partition_paths = {
    "fit":
        FIT,

    "development":
        DEVELOPMENT,

    "calibration":
        CALIBRATION,
}

pilot_records = []

for partition, path in (
    partition_paths.items()
):
    records = (
        read_manifest(
            path
        )
    )

    selected = records[
        :PILOT_PER_PARTITION
    ]

    if len(
        selected
    ) != (
        PILOT_PER_PARTITION
    ):
        raise RuntimeError(
            "Pilot selection count "
            "changed."
        )

    for record in selected:
        pilot_records.append(
            (
                partition,
                record,
            )
        )


scenario_reports = []

partition_sample_counts = Counter()
sample_classes = Counter()

real_covariance_rows = 0
real_observed_rows = 0
real_neighbor_slots_used = 0
real_map_nonzero_samples = 0

future_probe = None

pilot_start = time.perf_counter()


for index, (
    partition,
    record,
) in enumerate(
    pilot_records,
    start=1,
):
    scenario = (
        read_training_scenario(
            record
        )
    )

    built = (
        build_real_causal_inputs(
            scenario,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
        )
    )

    scene_inputs = (
        built[
            "scene_inputs"
        ]
    )

    # Crucial boundary:
    # future labels are attached only here,
    # after the full truth-free causal
    # algorithm input has already been built.
    samples = (
        attach_supervision(
            scene_inputs,
            scenario,
            T_H0_from_W=(
                built[
                    "adapted"
                ].frames
                .T_H0_from_W
            ),
        )
    )

    partition_sample_counts[
        partition
    ] += len(
        samples
    )

    for sample in samples:
        sample_classes[
            sample.actor_class
        ] += 1

        real_neighbor_slots_used += int(
            sum(
                sample
                .model_input
                .neighbor_mask
            )
        )

        if any(
            abs(x) > 0.0
            for x in (
                sample
                .model_input
                .map_context
            )
        ):
            real_map_nonzero_samples += 1

        for row in (
            sample
            .model_input
            .target_history
        ):
            observed = (
                row[-2]
                >
                0.5
            )

            if not observed:
                continue

            real_observed_rows += 1

            covariance_values = (
                row[6:12]
            )

            if any(
                abs(value)
                >
                0.0
                for value
                in covariance_values
            ):
                real_covariance_rows += 1

    report = {
        "partition":
            partition,

        "scenario_id":
            scenario.scenario_id,

        "stage2_algorithm_sha256":
            built[
                "algorithm_hash"
            ],

        "associated_tracks":
            built[
                "associated_track_count"
            ],

        "causal_histories":
            len(
                scene_inputs
                .histories
            ),

        "supervised_samples":
            len(samples),

        "sample_classes":
            dict(
                Counter(
                    sample.actor_class
                    for sample in samples
                )
            ),

        "causal_scene_sha256":
            scene_inputs.sha256(),

        "model_payload_sha256":
            canonical_sha(
                sample_payload_hashes(
                    samples
                )
            ),

        "label_sha256":
            canonical_sha(
                sample_label_hashes(
                    samples
                )
            ),
    }

    scenario_reports.append(
        report
    )

    if (
        future_probe is None
        and
        samples
    ):
        future_probe = (
            scenario,
            built,
            samples,
        )

    print(
        f"[{index:02d}/12] "
        f"{partition:11s} "
        f"{scenario.scenario_id} "
        f"| assoc="
        f"{built['associated_track_count']} "
        f"| histories="
        f"{len(scene_inputs.histories)} "
        f"| samples="
        f"{len(samples)}",
        flush=True,
    )


pilot_runtime_s = (
    time.perf_counter()
    -
    pilot_start
)


for partition in (
    "fit",
    "development",
    "calibration",
):
    if (
        partition_sample_counts[
            partition
        ]
        <=
        0
    ):
        raise SystemExit(
            "FAIL: real 4.2 pilot has "
            f"zero supervised samples "
            f"for {partition}."
        )


if real_observed_rows <= 0:
    raise SystemExit(
        "FAIL: no real observed "
        "history rows."
    )

if (
    real_covariance_rows
    <=
    0
):
    raise SystemExit(
        "FAIL: Stage2 measurement "
        "covariance did not reach "
        "Stage4 target features."
    )

if (
    real_neighbor_slots_used
    <=
    0
):
    raise SystemExit(
        "FAIL: no real neighbour "
        "context reached samples."
    )

if (
    real_map_nonzero_samples
    <=
    0
):
    raise SystemExit(
        "FAIL: real map context "
        "not populated."
    )


# ============================================================
# Strict future-mutation boundary
# ============================================================

if future_probe is None:
    raise SystemExit(
        "FAIL: no sample available "
        "for future-mutation probe."
    )

(
    original_scenario,
    original_built,
    original_samples,
) = future_probe

mutated_scenario = (
    mutate_future(
        original_scenario
    )
)

mutated_built = (
    build_real_causal_inputs(
        mutated_scenario,
        clean_config=(
            clean_config
        ),
        degraded_config=(
            degraded_config
        ),
    )
)

mutated_samples = (
    attach_supervision(
        mutated_built[
            "scene_inputs"
        ],
        mutated_scenario,
        T_H0_from_W=(
            mutated_built[
                "adapted"
            ].frames
            .T_H0_from_W
        ),
    )
)


if (
    original_built[
        "algorithm_hash"
    ]
    !=
    mutated_built[
        "algorithm_hash"
    ]
):
    raise SystemExit(
        "FAIL: future mutation changed "
        "Stage2 algorithm input."
    )


if (
    original_built[
        "scene_inputs"
    ].sha256()
    !=
    mutated_built[
        "scene_inputs"
    ].sha256()
):
    raise SystemExit(
        "FAIL: future mutation changed "
        "Stage4 causal scene input."
    )


original_payloads = (
    sample_payload_hashes(
        original_samples
    )
)

mutated_payloads = (
    sample_payload_hashes(
        mutated_samples
    )
)

if (
    original_payloads
    !=
    mutated_payloads
):
    raise SystemExit(
        "FAIL: future mutation changed "
        "numeric neural model payload."
    )


original_labels = (
    sample_label_hashes(
        original_samples
    )
)

mutated_labels = (
    sample_label_hashes(
        mutated_samples
    )
)

if (
    original_labels
    ==
    mutated_labels
):
    raise SystemExit(
        "FAIL: future-mutation diagnostic "
        "did not alter supervision labels."
    )


print()
print(
    "future mutation Stage2 hash = PASS"
)

print(
    "future mutation causal input= PASS"
)

print(
    "future mutation model tensor= PASS"
)

print(
    "future mutation labels      = CHANGED AS EXPECTED"
)


# ============================================================
# Frozen neural-sample contract
# ============================================================

config = {
    "stage": 4,
    "block": "4.2",
    "status": "FROZEN",

    "model_input": {
        "history_frames":
            11,

        "history_seconds":
            1.0,

        "history_feature_dim":
            FEATURE_DIM,

        "history_feature_semantics": [
            "H0 noisy Cartesian position",
            "strict-backward causal velocity",
            "Stage2 measurement covariance propagated to H0",
            "observation mask",
            "velocity-validity mask"
        ],

        "measurement_covariance":
            "MANDATORY_INPUT",

        "measurement_covariance_source": (
            "Stage2 R_t propagated through "
            "frozen Stage3 Cartesian geometry"
        ),

        "predictive_covariance":
            "NOT_AN_INPUT_NOT_YET_CREATED",

        "max_neighbors":
            8,

        "neighbor_selection": (
            "causal nearest associated "
            "tracks in H0"
        ),

        "map_context_dim":
            MAP_CONTEXT_DIM,

        "map_context": (
            "static current-scene WOMD map "
            "geometry only"
        ),

        "actor_class_numeric_feature":
            False,

        "perfect_track_id_numeric_feature":
            False,

        "tracks_to_predict_input":
            False,

        "objects_of_interest_input":
            False,

        "future_validity_input":
            False,

        "future_duration_input":
            False,

        "future_lidar_input":
            False,
    },

    "supervision": {
        "pairing": (
            "deterministic one-to-one "
            "historical/current-only "
            "distance matching"
        ),

        "pairing_gate_m":
            5.0,

        "future_used_for_pairing":
            False,

        "horizons_s": [
            0.1,
            0.3,
            0.5,
            1.0
        ],

        "future_positions":
            "label_only",

        "future_validity":
            "label_mask_only",

        "sample_inclusion_depends_on_future_validity":
            False,
    },

    "pilot": {
        "selection": (
            "first four frozen-manifest "
            "records from each fit/dev/cal "
            "partition"
        ),

        "selection_future_based":
            False,

        "selection_performance_based":
            False,

        "scenario_count":
            12,
    },

    "not_yet_claimed": [
        "normalization statistics",
        "class-balanced fit sampler",
        "LiDAR actor features",
        "learned map encoder",
        "GRU architecture",
        "predictive Gaussian covariance",
        "NLL training",
        "calibration"
    ],
}


CONFIG.write_text(
    json.dumps(
        config,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# Full Stage4 regression
# ============================================================

test = subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(
            STAGE4
            / "tests"
        ),
        "-p",
        "test_*.py",
    ],
    cwd=str(
        STAGE4
    ),
    text=True,
    capture_output=True,
)

combined = (
    test.stdout
    +
    "\n"
    +
    test.stderr
)

import re

match = re.search(
    r"Ran\s+(\d+)\s+tests?",
    combined,
)

test_count = (
    int(
        match.group(1)
    )
    if match
    else None
)

if (
    test.returncode
    !=
    0
    or
    test_count
    !=
    34
):
    print(
        combined
    )

    raise SystemExit(
        "FAIL: expected Stage4 "
        "regression 34/34."
    )


# ============================================================
# Implementation fingerprint
# ============================================================

(
    implementation_files,
    implementation_sha,
) = implementation_fingerprint()


free_gib = (
    shutil.disk_usage(
        ROOT
    ).free
    /
    1024**3
)

if free_gib < MIN_FREE_GIB:
    raise SystemExit(
        "FAIL: free-space reserve "
        "< 250 GiB."
    )


report = {
    "stage": 4,
    "block": "4.2",
    "status": "PASS",

    "real_pilot": {
        "scenario_count":
            12,

        "runtime_s":
            pilot_runtime_s,

        "partition_sample_counts":
            dict(
                partition_sample_counts
            ),

        "sample_classes":
            dict(
                sorted(
                    sample_classes.items()
                )
            ),

        "real_observed_rows":
            real_observed_rows,

        "real_covariance_rows":
            real_covariance_rows,

        "neighbor_slots_used":
            real_neighbor_slots_used,

        "map_context_nonzero_samples":
            real_map_nonzero_samples,

        "scenarios":
            scenario_reports,
    },

    "causality": {
        "future_mutation_stage2_hash":
            True,

        "future_mutation_causal_scene":
            True,

        "future_mutation_model_payload":
            True,

        "future_mutation_labels_changed":
            True,

        "future_used_for_supervision_pairing":
            False,

        "future_validity_model_input":
            False,

        "tracks_to_predict_model_input":
            False,

        "objects_of_interest_model_input":
            False,

        "perfect_track_id_numeric_feature":
            False,
    },

    "measurement_uncertainty": {
        "reaches_neural_features":
            True,

        "source":
            (
                "Stage2 R_t propagated "
                "to H0 through frozen "
                "Stage3 geometry"
            ),

        "distinct_from_predictive_uncertainty":
            True,
    },

    "context": {
        "multi_agent_neighbors":
            True,

        "maximum_neighbors":
            8,

        "static_map_context":
            True,

        "map_context_dim":
            MAP_CONTEXT_DIM,

        "LiDAR_actor_features":
            False,
    },

    "regression": {
        "tests_passed":
            34,

        "tests_total":
            34,

        "pass":
            True,
    },

    "storage": {
        "free_gib":
            free_gib,

        "hard_reserve_gib":
            MIN_FREE_GIB,

        "pass":
            True,
    },

    "implementation": {
        "file_count":
            implementation_files,

        "sha256":
            implementation_sha,
    },
}


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    +
    "\n",
    encoding="utf-8",
)


print()
print(
    "============================================================"
)
print(
    "STAGE4 BLOCK 4.2 GATE"
)
print(
    "============================================================"
)

print(
    "real scenarios              = 12 / 12"
)

print(
    "fit/dev/cal sample counts   =",
    dict(
        partition_sample_counts
    ),
)

print(
    "sample actor classes        =",
    dict(
        sorted(
            sample_classes
            .items()
        )
    ),
)

print(
    "history tensor              = 11 x",
    FEATURE_DIM,
)

print(
    "measurement covariance      = REAL INPUT"
)

print(
    "real covariance rows        =",
    real_covariance_rows,
)

print(
    "neighbour context           = PASS"
)

print(
    "neighbour slots used        =",
    real_neighbor_slots_used,
)

print(
    "static map context          = PASS"
)

print(
    "map-context dim             =",
    MAP_CONTEXT_DIM,
)

print(
    "future model input          = NO"
)

print(
    "future used for pairing     = NO"
)

print(
    "future validity input       = NO"
)

print(
    "tracks_to_predict input     = NO"
)

print(
    "perfect track ID feature    = NO"
)

print(
    "future-mutation input gate  = PASS"
)

print(
    "future labels               = SUPERVISION ONLY"
)

print(
    "full Stage4 regression      = 34 / 34 PASS"
)

print(
    "free GiB                    =",
    round(
        free_gib,
        3,
    ),
)

print(
    "250 GiB reserve             = PASS"
)

print(
    "implementation files        =",
    implementation_files,
)

print(
    "implementation SHA256       =",
    implementation_sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)


# ============================================================
# Close only after all checks pass.
# ============================================================

marker = (
    "## Block 4.2 — "
    "Real causal neural sample contract"
)

existing = (
    LOG.read_text(
        encoding="utf-8"
    )
    if LOG.exists()
    else ""
)

if marker not in existing:
    with LOG.open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write(
            "\n"
            + marker
            + "\n\n"
            "Status: PASS / FROZEN\n\n"
            "- Real Stage4 samples consume the "
              "frozen full-degraded Stage2 stream "
              "through the frozen Stage3 estimated "
              "association and Cartesian geometry.\n"
            "- Each target history contains exactly "
              "11 causal WOMD frames.\n"
            "- The frozen per-step numeric feature "
              "dimension is 14.\n"
            "- Position is noisy H0 position produced "
              "from the degraded observation stream.\n"
            "- Velocity is strict-backward and derived "
              "only from causal estimated positions; "
              "annotated WOMD velocity is not a "
              "realistic input.\n"
            "- Stage2 measurement covariance R_t is "
              "propagated to H0 by the frozen Stage3 "
              "geometry layer and is a mandatory "
              "numeric model input.\n"
            "- Measurement covariance remains distinct "
              "from predictive uncertainty, which has "
              "not yet been created.\n"
            "- Missing observations remain explicit "
              "through observed/velocity-valid masks.\n"
            "- Each target receives up to 8 nearest "
              "causal associated neighbour histories.\n"
            "- A lightweight static WOMD map-context "
              "summary is included as causal model "
              "context; learned map encoding remains "
              "a later ablation.\n"
            "- Perfect WOMD track ID, actor class, "
              "truth track index and sample ID are "
              "metadata only and are not numeric "
              "model inputs.\n"
            "- tracks_to_predict and "
              "objects_of_interest are not model "
              "inputs.\n"
            "- Supervision pairing is one-to-one and "
              "uses historical/current information "
              "only with a 5-m gate.\n"
            "- Future trajectory is accessed only "
              "after causal model-input construction "
              "and only as supervision.\n"
            "- Future validity is a label mask only "
              "and does not control sample inclusion.\n"
            "- Strict future-mutation testing leaves "
              "Stage2 algorithm input, Stage4 causal "
              "scene input and numeric model payload "
              "unchanged while changing labels.\n"
            "- Real deterministic pilot: first 4 "
              "frozen-manifest scenarios from each "
              "fit/development/calibration partition.\n"
            "- No normalization, class sampler, GRU, "
              "predictive covariance, NLL or "
              "calibration is claimed by Block4.2.\n"
            "- Full Stage4 regression: 34/34 PASS.\n"
            f"- Implementation files: "
            f"{implementation_files}.\n"
            f"- Implementation SHA256: "
            f"{implementation_sha}.\n"
        )


print()
print(
    "===== BLOCK 4.2 FINAL ====="
)

print(
    "degraded→neural bridge = PASS"
)

print(
    "R_t conditioning       = PASS"
)

print(
    "causal history          = PASS"
)

print(
    "multi-agent context     = PASS"
)

print(
    "map context             = PASS"
)

print(
    "future leakage          = NONE"
)

print(
    "supervision boundary    = PASS"
)

print(
    "regression              = 34 / 34"
)

print(
    "implementation SHA      =",
    implementation_sha,
)

print(
    "log closure             = PASS"
)

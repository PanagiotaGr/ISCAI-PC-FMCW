from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from iscai_stage4.data.training_index import (
    PRIMARY_CLASSES,
    TrainingScenarioRecord,
    current_anchor_class_counts,
    deterministic_split,
    iter_compact_motion_records,
    manifest_bytes,
    scenario_selection_hash,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

PAIRED = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0"
)

TRAIN_MOTION = (
    PAIRED
    / "training/motion"
)

VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/"
      "manifests/"
      "selected_validation.jsonl"
)

FORMAL_VALIDATION = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

BLOCK40_REPORT = (
    STAGE4
    / "reports/"
      "block40_bootstrap_gate.json"
)

STAGE3_CLOSURE = (
    STAGE3
    / "reports/"
      "stage3_final_closure.json"
)

ARTIFACT_DIR = (
    STAGE4
    / "artifacts/block41"
)

TRAINING_INDEX = (
    ARTIFACT_DIR
    / "canonical_training_index.jsonl"
)

FIT_MANIFEST = (
    ARTIFACT_DIR
    / "fit.jsonl"
)

DEVELOPMENT_MANIFEST = (
    ARTIFACT_DIR
    / "development.jsonl"
)

CALIBRATION_MANIFEST = (
    ARTIFACT_DIR
    / "calibration.jsonl"
)

SPLIT_CONFIG = (
    STAGE4
    / "configs/"
      "stage4_training_split.json"
)

REPORT = (
    STAGE4
    / "reports/"
      "block41_training_split_gate.json"
)

LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
)

EXPECTED_TRAINING_SCENARIOS = 72085
EXPECTED_VALIDATION_SCENARIOS = 44097

EXPECTED_VALIDATION_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_STAGE3_SHA = (
    "12b7a236cf1ea8e8a02b1c74da44d7db"
    "79cdaef363542a2be564b697036e6bbf"
)

MIN_FREE_GIB = 250.0


def file_sha256(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def jsonl_ids(
    path: Path,
) -> tuple[str, ...]:
    ids = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        item = json.loads(line)

        ids.append(
            str(
                item["scenario_id"]
            )
        )

    return tuple(ids)


def freeze_bytes(
    path: Path,
    content: bytes,
) -> None:
    if path.exists():
        actual = path.read_bytes()

        if actual != content:
            raise RuntimeError(
                "Frozen artifact differs "
                f"on rerun: {path}"
            )

        return

    path.write_bytes(
        content
    )


def canonical_metadata_signature(
    scenario: scenario_pb2.Scenario,
) -> str:
    """
    Signature of only metadata allowed to
    influence Block4.1 indexing.

    Deliberately excludes future states,
    tracks_to_predict and objects_of_interest.
    """

    payload = {
        "scenario_id":
            scenario.scenario_id,

        "current_time_index":
            int(
                scenario.current_time_index
            ),

        "sdc_track_index":
            int(
                scenario.sdc_track_index
            ),

        "current_anchor_class_counts":
            dict(
                current_anchor_class_counts(
                    scenario
                )
            ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode("utf-8")

    return sha256(
        encoded
    ).hexdigest()


def future_mutation_invariant(
    scenario: scenario_pb2.Scenario,
) -> bool:
    before = (
        canonical_metadata_signature(
            scenario
        )
    )

    clone = scenario_pb2.Scenario()

    clone.CopyFrom(
        scenario
    )

    anchor = int(
        clone.current_time_index
    )

    for track_index in range(
        min(
            len(clone.tracks),
            8,
        )
    ):
        track = clone.tracks[
            track_index
        ]

        for time_index in range(
            anchor + 1,
            len(track.states),
        ):
            state = track.states[
                time_index
            ]

            state.center_x = (
                float(
                    state.center_x
                )
                +
                1000.0
            )

            state.center_y = (
                float(
                    state.center_y
                )
                -
                1000.0
            )

            state.valid = (
                not bool(
                    state.valid
                )
            )

    after = (
        canonical_metadata_signature(
            clone
        )
    )

    return before == after


def static_forbidden_metadata_scan():
    path = (
        STAGE4
        / "src/iscai_stage4/"
          "data/training_index.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    forbidden = {
        "tracks_to_predict",
        "objects_of_interest",
    }

    found = set()

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Attribute,
        ):
            if node.attr in forbidden:
                found.add(
                    node.attr
                )

    return sorted(found)


def split_class_summary(
    records,
):
    actor_counts = Counter()
    scene_counts = Counter()

    for record in records:
        counts = (
            record.class_counts_dict()
        )

        for name, count in (
            counts.items()
        ):
            actor_counts[
                name
            ] += int(count)

            if int(count) > 0:
                scene_counts[
                    name
                ] += 1

    return {
        "current_valid_actor_counts":
            dict(
                sorted(
                    actor_counts.items()
                )
            ),

        "scenarios_containing_class":
            dict(
                sorted(
                    scene_counts.items()
                )
            ),
    }


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

            if "__pycache__" in path.parts:
                continue

            if path.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue

            files.append(path)

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
            relative.encode("utf-8")
        )
        digest.update(b"\0")

        digest.update(
            path.read_bytes()
        )
        digest.update(b"\0")

    return (
        len(files),
        digest.hexdigest(),
    )


print(
    "============================================================"
)
print(
    "STAGE 4 — BLOCK 4.1 "
    "CANONICAL TRAINING SPLIT"
)
print(
    "============================================================"
)


# ============================================================
# 1. Upstream closure
# ============================================================

block40 = json.loads(
    BLOCK40_REPORT.read_text(
        encoding="utf-8"
    )
)

if block40["status"] != "PASS":
    raise SystemExit(
        "FAIL: Block4.0 is not PASS."
    )

stage3 = json.loads(
    STAGE3_CLOSURE.read_text(
        encoding="utf-8"
    )
)

if (
    stage3["status"]
    !=
    "COMPLETE_FROZEN"
):
    raise SystemExit(
        "FAIL: Stage3 not frozen."
    )

if (
    stage3[
        "implementation"
    ]["sha256"]
    !=
    EXPECTED_STAGE3_SHA
):
    raise SystemExit(
        "FAIL: Stage3 SHA changed."
    )

print(
    "Block4.0 frozen upstream   = PASS"
)
print(
    "Stage3 frozen upstream     = PASS"
)


# ============================================================
# 2. Validation exclusion evidence
# ============================================================

if (
    file_sha256(
        VALIDATION_MANIFEST
    )
    !=
    EXPECTED_VALIDATION_SHA
):
    raise SystemExit(
        "FAIL: validation manifest "
        "SHA changed."
    )

if (
    file_sha256(
        FORMAL_VALIDATION
    )
    !=
    EXPECTED_FORMAL_SHA
):
    raise SystemExit(
        "FAIL: formal validation "
        "manifest SHA changed."
    )

validation_ids = set(
    jsonl_ids(
        VALIDATION_MANIFEST
    )
)

formal_ids = set(
    jsonl_ids(
        FORMAL_VALIDATION
    )
)

if (
    len(validation_ids)
    !=
    EXPECTED_VALIDATION_SCENARIOS
):
    raise SystemExit(
        "FAIL: validation count changed."
    )

if len(
    formal_ids
) != 120:
    raise SystemExit(
        "FAIL: formal validation "
        "count changed."
    )

print(
    "validation manifest         = PASS"
)
print(
    "formal validation N=120     = PASS"
)


# ============================================================
# 3. Canonical training scan
# ============================================================

if not TRAIN_MOTION.is_dir():
    raise SystemExit(
        "FAIL: canonical training/motion "
        "directory missing."
    )

shards = tuple(
    sorted(
        path
        for path in TRAIN_MOTION.iterdir()
        if path.is_file()
    )
)

if not shards:
    raise SystemExit(
        "FAIL: no training motion shards."
    )

print(
    "training motion shards      =",
    len(shards),
)

scan_start = time.perf_counter()

records = []
scenario_ids = set()

global_actor_counts = Counter()
global_scene_counts = Counter()

timeline_failures = 0

future_mutation_probes = 0
future_mutation_passes = 0


for shard_index, shard in enumerate(
    shards,
    start=1,
):
    for (
        offset,
        payload_length,
        scenario,
    ) in iter_compact_motion_records(
        shard
    ):
        scenario_id = str(
            scenario.scenario_id
        )

        if not scenario_id:
            raise RuntimeError(
                "Empty training scenario_id."
            )

        if scenario_id in scenario_ids:
            raise RuntimeError(
                "Duplicate training "
                f"scenario_id: {scenario_id}"
            )

        scenario_ids.add(
            scenario_id
        )

        if (
            int(
                scenario.current_time_index
            )
            !=
            10
        ):
            timeline_failures += 1

        if (
            len(
                scenario.timestamps_seconds
            )
            !=
            91
        ):
            timeline_failures += 1

        counts_tuple = (
            current_anchor_class_counts(
                scenario
            )
        )

        counts = dict(
            counts_tuple
        )

        for name, count in (
            counts.items()
        ):
            global_actor_counts[
                name
            ] += int(count)

            if int(count) > 0:
                global_scene_counts[
                    name
                ] += 1

        record = TrainingScenarioRecord(
            scenario_id=scenario_id,
            motion_shard=shard.name,
            compact_record_offset=(
                int(offset)
            ),
            payload_length=(
                int(payload_length)
            ),
            selection_hash=(
                scenario_selection_hash(
                    scenario_id
                )
            ),
            anchor_class_counts=(
                counts_tuple
            ),
        )

        records.append(
            record
        )

        if future_mutation_probes < 3:
            future_mutation_probes += 1

            if future_mutation_invariant(
                scenario
            ):
                future_mutation_passes += 1

    if (
        shard_index % 25 == 0
        or
        shard_index
        ==
        len(shards)
    ):
        print(
            f"scanned shards "
            f"{shard_index}/{len(shards)} "
            f"| scenarios={len(records)}",
            flush=True,
        )


scan_runtime_s = (
    time.perf_counter()
    -
    scan_start
)


if (
    len(records)
    !=
    EXPECTED_TRAINING_SCENARIOS
):
    raise SystemExit(
        "FAIL: expected "
        f"{EXPECTED_TRAINING_SCENARIOS} "
        f"training scenarios, got "
        f"{len(records)}."
    )

if timeline_failures != 0:
    raise SystemExit(
        "FAIL: non-standard WOMD "
        f"timeline count = "
        f"{timeline_failures}."
    )

if (
    future_mutation_probes
    !=
    3
    or
    future_mutation_passes
    !=
    3
):
    raise SystemExit(
        "FAIL: future-mutation "
        "invariance."
    )


print(
    "training scenarios          =",
    len(records),
)

print(
    "unique training IDs         = PASS"
)

print(
    "91 timestamps / anchor 10   = PASS"
)

print(
    "future mutation probes      = 3 / 3 PASS"
)


# ============================================================
# 4. Official split separation
# ============================================================

training_ids = set(
    scenario_ids
)

validation_overlap = (
    training_ids
    &
    validation_ids
)

formal_overlap = (
    training_ids
    &
    formal_ids
)

if validation_overlap:
    raise SystemExit(
        "FAIL: training/validation "
        f"scenario overlap = "
        f"{len(validation_overlap)}."
    )

if formal_overlap:
    raise SystemExit(
        "FAIL: training/formal-validation "
        f"overlap = "
        f"{len(formal_overlap)}."
    )

print(
    "train/validation overlap    = 0 PASS"
)

print(
    "train/formal-N120 overlap   = 0 PASS"
)


# ============================================================
# 5. No forbidden benchmark metadata
# ============================================================

forbidden_metadata = (
    static_forbidden_metadata_scan()
)

if forbidden_metadata:
    raise SystemExit(
        "FAIL: training index accesses "
        f"forbidden metadata: "
        f"{forbidden_metadata}"
    )

print(
    "tracks_to_predict access    = NO"
)

print(
    "objects_of_interest access  = NO"
)

print(
    "future outcome selection    = NO"
)


# ============================================================
# 6. Deterministic 80/10/10 partition
# ============================================================

records_tuple = tuple(
    records
)

split_a = deterministic_split(
    records_tuple
)

split_b = deterministic_split(
    tuple(
        reversed(
            records_tuple
        )
    )
)

for name in (
    "fit",
    "development",
    "calibration",
):
    ids_a = tuple(
        item.scenario_id
        for item in split_a[name]
    )

    ids_b = tuple(
        item.scenario_id
        for item in split_b[name]
    )

    if ids_a != ids_b:
        raise SystemExit(
            "FAIL: deterministic split "
            f"order invariant for {name}."
        )


fit = split_a["fit"]
development = split_a[
    "development"
]
calibration = split_a[
    "calibration"
]


if (
    len(fit),
    len(development),
    len(calibration),
) != (
    57668,
    7208,
    7209,
):
    raise SystemExit(
        "FAIL: frozen 80/10/10 "
        "partition counts."
    )


fit_ids = {
    item.scenario_id
    for item in fit
}

development_ids = {
    item.scenario_id
    for item in development
}

calibration_ids = {
    item.scenario_id
    for item in calibration
}


if any((
    fit_ids
    &
    development_ids,

    fit_ids
    &
    calibration_ids,

    development_ids
    &
    calibration_ids,
)):
    raise SystemExit(
        "FAIL: internal partition overlap."
    )


if (
    len(
        fit_ids
        |
        development_ids
        |
        calibration_ids
    )
    !=
    EXPECTED_TRAINING_SCENARIOS
):
    raise SystemExit(
        "FAIL: internal split does "
        "not cover all training scenarios."
    )


print(
    "fit scenarios               =",
    len(fit),
)

print(
    "development scenarios       =",
    len(development),
)

print(
    "calibration scenarios       =",
    len(calibration),
)

print(
    "internal overlap            = 0 PASS"
)

print(
    "split input-order invariant = PASS"
)


# ============================================================
# 7. Current-time class support
# ============================================================

split_summaries = {
    "fit":
        split_class_summary(
            fit
        ),

    "development":
        split_class_summary(
            development
        ),

    "calibration":
        split_class_summary(
            calibration
        ),
}


for partition, summary in (
    split_summaries.items()
):
    scenes = summary[
        "scenarios_containing_class"
    ]

    for actor_class in (
        PRIMARY_CLASSES
    ):
        if (
            scenes.get(
                actor_class,
                0,
            )
            <=
            0
        ):
            raise SystemExit(
                "FAIL: "
                f"{partition} lacks "
                f"{actor_class} support."
            )


print(
    "vehicle support all splits  = PASS"
)

print(
    "pedestrian support all      = PASS"
)

print(
    "cyclist support all         = PASS"
)

print(
    "class used for split        = NO"
)


# ============================================================
# 8. Freeze manifests
# ============================================================

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


assignment = {}

for name in (
    "fit",
    "development",
    "calibration",
):
    for rank, item in enumerate(
        split_a[name],
        start=1,
    ):
        assignment[
            item.scenario_id
        ] = (
            name,
            rank,
        )


index_rows = []

for item in records:
    partition, rank = (
        assignment[
            item.scenario_id
        ]
    )

    row = item.to_json_dict(
        partition=partition,
        partition_rank=rank,
    )

    index_rows.append(
        json.dumps(
            row,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )


index_bytes = (
    "\n".join(
        index_rows
    )
    +
    "\n"
).encode("utf-8")


fit_bytes = manifest_bytes(
    fit,
    partition="fit",
)

development_bytes = (
    manifest_bytes(
        development,
        partition="development",
    )
)

calibration_bytes = (
    manifest_bytes(
        calibration,
        partition="calibration",
    )
)


freeze_bytes(
    TRAINING_INDEX,
    index_bytes,
)

freeze_bytes(
    FIT_MANIFEST,
    fit_bytes,
)

freeze_bytes(
    DEVELOPMENT_MANIFEST,
    development_bytes,
)

freeze_bytes(
    CALIBRATION_MANIFEST,
    calibration_bytes,
)


manifest_shas = {
    "canonical_training_index":
        file_sha256(
            TRAINING_INDEX
        ),

    "fit":
        file_sha256(
            FIT_MANIFEST
        ),

    "development":
        file_sha256(
            DEVELOPMENT_MANIFEST
        ),

    "calibration":
        file_sha256(
            CALIBRATION_MANIFEST
        ),
}


# ============================================================
# 9. Freeze split config
# ============================================================

split_config = {
    "stage": 4,
    "block": "4.1",
    "status": "FROZEN",

    "source": {
        "dataset":
            "WOMD/WOMD-LiDAR v1.3.0 "
            "canonical paired corpus",

        "official_split":
            "training",

        "training_scenarios":
            EXPECTED_TRAINING_SCENARIOS,

        "validation_excluded":
            True,

        "formal_validation_excluded":
            True,

        "canonical_motion_root":
            str(
                TRAIN_MOTION
            ),
    },

    "partition_policy": {
        "unit":
            "scenario",

        "selection_key":
            "SHA256(scenario_id)",

        "ordering":
            "ascending_hash_then_scenario_id",

        "fit_count":
            57668,

        "development_count":
            7208,

        "calibration_count":
            7209,

        "fit_fraction_nominal":
            0.8,

        "development_fraction_nominal":
            0.1,

        "calibration_fraction_nominal":
            0.1,

        "class_used_for_partition":
            False,

        "future_used_for_partition":
            False,

        "performance_used_for_partition":
            False,

        "tracks_to_predict_used_for_partition":
            False,

        "objects_of_interest_used_for_partition":
            False,
    },

    "statistics_policy": {
        "normalization_source":
            "fit_only",

        "early_stopping_source":
            "development_only",

        "calibration_source":
            "calibration_only",

        "formal_validation_source":
            "evaluation_only",

        "class_balancing":
            (
                "later fit-sample sampler; "
                "current/anchor class only; "
                "does not modify frozen "
                "scenario partitions"
            ),
    },

    "manifests": {
        "canonical_training_index": {
            "path":
                str(
                    TRAINING_INDEX
                ),

            "sha256":
                manifest_shas[
                    "canonical_training_index"
                ],

            "count":
                EXPECTED_TRAINING_SCENARIOS,
        },

        "fit": {
            "path":
                str(
                    FIT_MANIFEST
                ),

            "sha256":
                manifest_shas["fit"],

            "count":
                len(fit),
        },

        "development": {
            "path":
                str(
                    DEVELOPMENT_MANIFEST
                ),

            "sha256":
                manifest_shas[
                    "development"
                ],

            "count":
                len(development),
        },

        "calibration": {
            "path":
                str(
                    CALIBRATION_MANIFEST
                ),

            "sha256":
                manifest_shas[
                    "calibration"
                ],

            "count":
                len(calibration),
        },
    },

    "current_anchor_class_support":
        split_summaries,

    "future_mutation_gate": {
        "probes":
            future_mutation_probes,

        "passes":
            future_mutation_passes,

        "pass":
            True,
    },
}


split_config_bytes = (
    json.dumps(
        split_config,
        indent=2,
        sort_keys=True,
    )
    +
    "\n"
).encode("utf-8")


freeze_bytes(
    SPLIT_CONFIG,
    split_config_bytes,
)


# ============================================================
# 10. Storage reserve
# ============================================================

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


# ============================================================
# 11. Full Stage4 regression
# ============================================================

test = subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(
            STAGE4 / "tests"
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
    test.returncode != 0
    or
    test_count != 20
):
    print(
        combined
    )

    raise SystemExit(
        "FAIL: expected full Stage4 "
        "regression 20/20."
    )


# ============================================================
# 12. Implementation fingerprint
# ============================================================

(
    implementation_files,
    implementation_sha,
) = implementation_fingerprint()


report = {
    "stage": 4,
    "block": "4.1",
    "status": "PASS",

    "purpose": (
        "canonical training access and "
        "deterministic scenario-level "
        "fit/development/calibration freeze"
    ),

    "training_corpus": {
        "scenario_count":
            len(records),

        "motion_shards":
            len(shards),

        "scan_runtime_s":
            scan_runtime_s,

        "current_anchor_actor_counts":
            dict(
                sorted(
                    global_actor_counts
                    .items()
                )
            ),

        "scenarios_containing_class":
            dict(
                sorted(
                    global_scene_counts
                    .items()
                )
            ),
    },

    "partitions": {
        "fit":
            len(fit),

        "development":
            len(development),

        "calibration":
            len(calibration),

        "scenario_level":
            True,

        "overlap":
            0,

        "complete_training_coverage":
            True,

        "deterministic":
            True,

        "input_order_invariant":
            True,
    },

    "leakage": {
        "training_validation_overlap":
            0,

        "training_formal_validation_overlap":
            0,

        "future_used_for_partition":
            False,

        "performance_used_for_partition":
            False,

        "tracks_to_predict_accessed":
            False,

        "objects_of_interest_accessed":
            False,

        "normalization_from_validation":
            False,

        "calibration_from_validation":
            False,

        "future_mutation_probes":
            3,

        "future_mutation_passes":
            3,
    },

    "class_support":
        split_summaries,

    "class_balancing_policy": (
        "fit-sample-level later; "
        "does not alter frozen scenario "
        "membership"
    ),

    "manifests": {
        "training_index_sha256":
            manifest_shas[
                "canonical_training_index"
            ],

        "fit_sha256":
            manifest_shas["fit"],

        "development_sha256":
            manifest_shas[
                "development"
            ],

        "calibration_sha256":
            manifest_shas[
                "calibration"
            ],

        "split_config_sha256":
            file_sha256(
                SPLIT_CONFIG
            ),
    },

    "storage": {
        "free_gib":
            free_gib,

        "hard_reserve_gib":
            MIN_FREE_GIB,

        "pass":
            True,
    },

    "regression": {
        "tests_passed":
            20,

        "tests_total":
            20,

        "pass":
            True,
    },

    "implementation": {
        "file_count":
            implementation_files,

        "sha256":
            implementation_sha,
    },

    "not_yet_claimed": [
        "Stage2-to-neural sample construction",
        "future target extraction",
        "measurement covariance tensor conditioning",
        "multi-agent context tensor",
        "map-context tensor",
        "normalization statistics",
        "GRU training",
        "probabilistic covariance",
        "calibration"
    ],
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
    "STAGE4 BLOCK 4.1 GATE"
)
print(
    "============================================================"
)

print(
    "canonical training scenarios =",
    len(records),
)

print(
    "training motion shards        =",
    len(shards),
)

print(
    "fit / dev / calibration       =",
    len(fit),
    "/",
    len(development),
    "/",
    len(calibration),
)

print(
    "scenario-level split          = PASS"
)

print(
    "deterministic SHA split       = PASS"
)

print(
    "input-order invariant         = PASS"
)

print(
    "partition overlap             = 0"
)

print(
    "train/validation overlap      = 0"
)

print(
    "train/formal-N120 overlap     = 0"
)

print(
    "future selection              = NO"
)

print(
    "performance selection         = NO"
)

print(
    "tracks_to_predict access      = NO"
)

print(
    "future mutation               = 3 / 3 PASS"
)

print(
    "fit class scenes              =",
    split_summaries[
        "fit"
    ][
        "scenarios_containing_class"
    ],
)

print(
    "development class scenes      =",
    split_summaries[
        "development"
    ][
        "scenarios_containing_class"
    ],
)

print(
    "calibration class scenes      =",
    split_summaries[
        "calibration"
    ][
        "scenarios_containing_class"
    ],
)

print(
    "fit manifest SHA256           =",
    manifest_shas[
        "fit"
    ],
)

print(
    "development manifest SHA256   =",
    manifest_shas[
        "development"
    ],
)

print(
    "calibration manifest SHA256   =",
    manifest_shas[
        "calibration"
    ],
)

print(
    "free GiB                      =",
    round(
        free_gib,
        3,
    ),
)

print(
    "250 GiB reserve               = PASS"
)

print(
    "full Stage4 regression        = 20 / 20 PASS"
)

print(
    "implementation files          =",
    implementation_files,
)

print(
    "implementation SHA256         =",
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
# 13. Close Block4.1 only after all gates passed.
# ============================================================

marker = (
    "## Block 4.1 — "
    "Frozen training partitions"
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
            "- Canonical paired WOMD training "
              "corpus contains 72,085 scenarios.\n"
            "- Partitioning is scenario-level "
              "and deterministic by ascending "
              "SHA256(scenario_id).\n"
            "- Frozen counts: fit=57,668, "
              "development=7,208, "
              "calibration=7,209.\n"
            "- Internal partition overlap: 0.\n"
            "- Training/official-validation "
              "scenario overlap: 0.\n"
            "- Training/frozen-formal-N120 "
              "scenario overlap: 0.\n"
            "- Future trajectories, future "
              "validity, performance outcomes, "
              "tracks_to_predict and "
              "objects_of_interest do not "
              "influence partition membership.\n"
            "- Three real-scenario future-mutation "
              "probes leave the allowed causal "
              "index metadata unchanged.\n"
            "- Current-time vehicle, pedestrian "
              "and cyclist support exists in "
              "fit, development and calibration.\n"
            "- Actor class is audited but does "
              "not alter scenario partitioning; "
              "class balancing will occur later "
              "inside the fit sample sampler.\n"
            "- Normalization statistics are "
              "restricted to fit only.\n"
            "- Early stopping/model development "
              "is restricted to development only.\n"
            "- Calibration fitting is restricted "
              "to calibration only.\n"
            "- Official/frozen validation is "
              "evaluation-only.\n"
            "- No future labels have yet been "
              "materialized by Stage4.\n"
            f"- Fit manifest SHA256: "
            f"{manifest_shas['fit']}.\n"
            f"- Development manifest SHA256: "
            f"{manifest_shas['development']}.\n"
            f"- Calibration manifest SHA256: "
            f"{manifest_shas['calibration']}.\n"
            f"- Full Stage4 regression: "
            f"20/20 PASS.\n"
            f"- Implementation files: "
            f"{implementation_files}.\n"
            f"- Implementation SHA256: "
            f"{implementation_sha}.\n"
        )


print()
print(
    "===== BLOCK 4.1 FINAL ====="
)

print(
    "training corpus       = PASS"
)

print(
    "fit/dev/cal freeze    = PASS"
)

print(
    "official split leak   = NONE"
)

print(
    "future selection      = NONE"
)

print(
    "multiclass support    = PASS"
)

print(
    "regression            = 20 / 20"
)

print(
    "implementation SHA    =",
    implementation_sha,
)

print(
    "log closure           = PASS"
)

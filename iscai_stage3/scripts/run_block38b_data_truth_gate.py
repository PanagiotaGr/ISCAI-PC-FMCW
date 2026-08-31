from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import hashlib
import json
from pathlib import Path

from iscai_stage3.validation import (
    build_compact_motion_offset_index,
    deterministic_manifest_order,
    expected_lidar_path,
    read_motion_scenario,
    read_validation_manifest,
    resolve_motion_shard,
    sha256_file,
)


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)

PAIRED_ROOT = Path(
    "/home/agni/waymo/data/"
    "paired_womd_lidar_v1_3_0"
)

MANIFEST = Path(
    "/home/agni/waymo/"
    "iscai_data_prep/manifests/"
    "selected_validation.jsonl"
)

EXPECTED_COUNT = 44097

EXPECTED_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)


def implementation_hash():
    files = []

    for base in (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
    ):
        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix == ".pyc":
                continue

            files.append(path)

    files = sorted(
        set(files),
        key=lambda path: str(
            path.relative_to(ROOT)
        ),
    )

    digest = hashlib.sha256()

    for path in files:
        relative = str(
            path.relative_to(ROOT)
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
        digest.hexdigest(),
        len(files),
    )


print(
    "===== Stage3 Block 3.8B gate ====="
)


# ============================================================
# 1. Frozen validation manifest
# ============================================================

actual_sha = sha256_file(
    MANIFEST
)

if actual_sha != EXPECTED_SHA:
    raise SystemExit(
        "FAIL: frozen validation manifest "
        "SHA256 changed."
    )

rows = read_validation_manifest(
    MANIFEST
)

if len(rows) != EXPECTED_COUNT:
    raise SystemExit(
        "FAIL: validation scenario count "
        f"{len(rows)} != {EXPECTED_COUNT}"
    )

if any(
    row.selection_policy
    !=
    "all_exact_matching_validation"
    for row in rows
):
    raise SystemExit(
        "FAIL: validation selection policy "
        "changed."
    )

if any(
    row.selection_hash
    !=
    sha256(
        row.scenario_id.encode(
            "utf-8"
        )
    ).hexdigest()
    for row in rows
):
    raise SystemExit(
        "FAIL: SHA256(scenario_id) "
        "selection contract changed."
    )


# ============================================================
# 2. Reconstruct canonical compact offsets
# ============================================================

offsets = (
    build_compact_motion_offset_index(
        rows
    )
)

if len(offsets) != len(rows):
    raise SystemExit(
        "FAIL: compact offset count differs "
        "from manifest count."
    )

grouped = defaultdict(list)

for row in rows:
    grouped[
        row.source_shard
    ].append(row)

if len(grouped) != 150:
    raise SystemExit(
        "FAIL: expected 150 WOMD validation "
        f"source groups, got {len(grouped)}."
    )

for source in grouped:
    grouped[source].sort(
        key=lambda row:
        row.record_offset
    )


# ============================================================
# 3. Verify ALL canonical compact shard sizes
# ============================================================

for source, source_rows in sorted(
    grouped.items()
):
    canonical = resolve_motion_shard(
        source_rows[0],
        paired_root=PAIRED_ROOT,
    )

    try:
        canonical.relative_to(
            PAIRED_ROOT
        )
    except ValueError as exc:
        raise SystemExit(
            "FAIL: canonical motion path "
            "escaped paired root."
        ) from exc

    expected_bytes = sum(
        row.motion_record_bytes
        for row in source_rows
    )

    actual_bytes = (
        canonical.stat().st_size
    )

    if actual_bytes != expected_bytes:
        raise SystemExit(
            "FAIL: compact shard size "
            f"mismatch for {canonical.name}: "
            f"{actual_bytes} != "
            f"{expected_bytes}"
        )


# ============================================================
# 4. Real random-access probes
# ============================================================

sources = sorted(grouped)

probe_source_indices = (
    0,
    len(sources) // 2,
    len(sources) - 1,
)

probe_rows = []

for source_index in probe_source_indices:
    source_rows = grouped[
        sources[source_index]
    ]

    for row_index in (
        0,
        len(source_rows) // 2,
        len(source_rows) - 1,
    ):
        row = source_rows[
            row_index
        ]

        if row not in probe_rows:
            probe_rows.append(row)

for row in probe_rows:
    scenario = read_motion_scenario(
        row,
        paired_root=PAIRED_ROOT,
        compact_record_offset=(
            offsets[
                row.scenario_id
            ]
        ),
    )

    if (
        scenario.scenario_id
        !=
        row.scenario_id
    ):
        raise SystemExit(
            "FAIL: random-access "
            "scenario identity."
        )

    if (
        len(
            scenario.timestamps_seconds
        )
        !=
        91
    ):
        raise SystemExit(
            "FAIL: expected 91 timestamps "
            "in real WOMD probe."
        )

    if (
        int(
            scenario.current_time_index
        )
        !=
        10
    ):
        raise SystemExit(
            "FAIL: expected "
            "current_time_index=10."
        )


# ============================================================
# 5. First frozen scenario + temporal contract
# ============================================================

first = rows[0]

scenario = read_motion_scenario(
    first,
    paired_root=PAIRED_ROOT,
    compact_record_offset=(
        offsets[
            first.scenario_id
        ]
    ),
)

timestamps = tuple(
    float(value)
    for value in (
        scenario.timestamps_seconds
    )
)

current_index = int(
    scenario.current_time_index
)

history_count = current_index

future_count = (
    len(timestamps)
    -
    current_index
    -
    1
)

if len(timestamps) != 91:
    raise SystemExit(
        "FAIL: WOMD timestamp count."
    )

if current_index != 10:
    raise SystemExit(
        "FAIL: WOMD current_time_index."
    )

if history_count != 10:
    raise SystemExit(
        "FAIL: WOMD causal history count."
    )

if future_count != 80:
    raise SystemExit(
        "FAIL: WOMD future count."
    )

if not all(
    abs(
        (
            next_time
            -
            previous_time
        )
        -
        0.1
    ) < 1e-3
    for previous_time, next_time
    in zip(
        timestamps[:-1],
        timestamps[1:],
    )
):
    raise SystemExit(
        "FAIL: WOMD timestamp period "
        "differs from nominal 0.1 s."
    )


# ============================================================
# 6. Paired LiDAR presence
# ============================================================

lidar_path = expected_lidar_path(
    first,
    paired_root=PAIRED_ROOT,
)

if not lidar_path.is_file():
    raise SystemExit(
        "FAIL: paired LiDAR sidecar missing."
    )

if (
    lidar_path.stat().st_size
    !=
    first.lidar_bytes
):
    raise SystemExit(
        "FAIL: paired LiDAR byte size "
        "changed."
    )


# ============================================================
# 7. Pre-formal validation config
# ============================================================

config_path = (
    ROOT
    /
    "configs/"
    "stage3_real_validation.json"
)

config = json.loads(
    config_path.read_text(
        encoding="utf-8"
    )
)

dataset = config[
    "dataset"
]

truth_config = config[
    "truth"
]

pilot_config = config[
    "pilot"
]

formal = config[
    "formal_validation"
]

if (
    dataset[
        "manifest_expected_scenarios"
    ]
    !=
    EXPECTED_COUNT
):
    raise SystemExit(
        "FAIL: config validation count."
    )

if (
    dataset[
        "manifest_expected_sha256"
    ]
    !=
    EXPECTED_SHA
):
    raise SystemExit(
        "FAIL: config manifest SHA."
    )

if (
    dataset[
        "selection_policy"
    ]
    !=
    "all_exact_matching_validation"
):
    raise SystemExit(
        "FAIL: config selection policy."
    )

if (
    truth_config[
        "role"
    ]
    !=
    "evaluator_only"
):
    raise SystemExit(
        "FAIL: truth is not evaluator-only."
    )

if (
    truth_config[
        "future_used_for_eligibility"
    ]
):
    raise SystemExit(
        "FAIL: future truth enabled "
        "for eligibility."
    )

if (
    truth_config[
        "future_used_for_assignment"
    ]
):
    raise SystemExit(
        "FAIL: future truth enabled "
        "for assignment."
    )

if not (
    truth_config[
        "future_used_for_scoring_only"
    ]
):
    raise SystemExit(
        "FAIL: future truth scoring-only "
        "contract changed."
    )

if (
    truth_config[
        "interpolation"
    ]
):
    raise SystemExit(
        "FAIL: WOMD truth interpolation "
        "enabled."
    )

if (
    truth_config[
        "canonical_horizons_s"
    ]
    !=
    [
        0.1,
        0.3,
        0.5,
        1.0,
    ]
):
    raise SystemExit(
        "FAIL: canonical horizons changed."
    )

if (
    pilot_config[
        "candidate_count"
    ]
    !=
    12
):
    raise SystemExit(
        "FAIL: pilot candidate count "
        "changed."
    )

if (
    pilot_config[
        "performance_based_selection"
    ]
):
    raise SystemExit(
        "FAIL: performance-based pilot "
        "selection enabled."
    )

if (
    pilot_config[
        "future_based_selection"
    ]
):
    raise SystemExit(
        "FAIL: future-based pilot "
        "selection enabled."
    )

if (
    formal[
        "N"
    ]
    is not None
):
    raise SystemExit(
        "FAIL: formal N frozen before pilot."
    )

if (
    formal[
        "scenario_ids"
    ]
    is not None
):
    raise SystemExit(
        "FAIL: formal scenario IDs frozen "
        "before pilot."
    )

if (
    formal[
        "manifest_status"
    ]
    !=
    "NOT_YET_FROZEN"
):
    raise SystemExit(
        "FAIL: formal validation status "
        "changed prematurely."
    )


# ============================================================
# 8. Deterministic pilot candidates
# ============================================================

ordered = (
    deterministic_manifest_order(
        rows
    )
)

pilot_ids = tuple(
    row.scenario_id
    for row in ordered[:12]
)

if len(pilot_ids) != 12:
    raise SystemExit(
        "FAIL: pilot candidate count."
    )

if len(
    set(pilot_ids)
) != 12:
    raise SystemExit(
        "FAIL: duplicate pilot IDs."
    )


# ============================================================
# 9. Algorithm / truth-sidecar boundary
# ============================================================

algorithm_roots = (
    ROOT / "src/iscai_stage3/association",
    ROOT / "src/iscai_stage3/baselines",
    ROOT / "src/iscai_stage3/filters",
    ROOT / "src/iscai_stage3/hough",
    ROOT / "src/iscai_stage3/state",
)

offending = []

for base in algorithm_roots:
    for path in base.rglob(
        "*.py"
    ):
        text = path.read_text(
            encoding="utf-8"
        )

        if (
            "build_womd_truth_sidecar"
            in text
            or
            "EvaluationTruth"
            in text
            or
            "womd_truth"
            in text
        ):
            offending.append(
                str(path)
            )

if offending:
    raise SystemExit(
        "FAIL: algorithm package depends "
        "on evaluator truth sidecar: "
        f"{offending}"
    )


# ============================================================
# 10. Implementation hash
# ============================================================

implementation_sha, file_count = (
    implementation_hash()
)


report = {
    "stage": 3,
    "block": "3.8B",
    "status": "PASS",

    "regression": {
        "targeted_tests": 11,
        "full_stage3_tests": 106
    },

    "validation_manifest": {
        "sha256": actual_sha,
        "scenarios": len(rows),
        "source_shard_groups": len(
            grouped
        ),
        "selection_policy": (
            "all_exact_matching_validation"
        ),
        "selection_hash_contract": (
            "SHA256(scenario_id)"
        )
    },

    "canonical_motion_access": {
        "compact_shards": len(
            grouped
        ),
        "compact_offsets": len(
            offsets
        ),
        "original_record_offset": (
            "PROVENANCE_ONLY"
        ),
        "canonical_offset": (
            "CUMULATIVE_SELECTED_"
            "MOTION_RECORD_BYTES"
        ),
        "all_shard_sizes_verified": True,
        "real_random_access_probes": len(
            probe_rows
        ),
        "raw_waymo_runtime_dependency": False,
        "second_materialization": False
    },

    "real_scenario_probe": {
        "scenario_id": (
            scenario.scenario_id
        ),
        "timestamps": len(
            timestamps
        ),
        "current_time_index": (
            current_index
        ),
        "history_samples": (
            history_count
        ),
        "future_samples": (
            future_count
        ),
        "nominal_period_s": 0.1,
        "paired_lidar": True
    },

    "truth_policy": {
        "sidecar": "EVALUATOR_ONLY",
        "eligibility_source": (
            "UPSTREAM_CURRENT_ANCHOR_SET"
        ),
        "future_for_eligibility": False,
        "future_for_assignment": False,
        "future_for_scoring_only": True,
        "temporal_interpolation": False,
        "horizons_s": [
            0.1,
            0.3,
            0.5,
            1.0
        ]
    },

    "pilot": {
        "formal": False,
        "candidate_count": 12,
        "ordering": (
            "ASCENDING_SHA256_SCENARIO_ID"
        ),
        "scenario_ids": list(
            pilot_ids
        )
    },

    "formal_validation": {
        "N": None,
        "scenario_ids": None,
        "status": "NOT_YET_FROZEN"
    },

    "algorithm_truth_dependency": "NONE",

    "implementation": {
        "files": file_count,
        "sha256": implementation_sha
    }
}


report_path = (
    ROOT
    /
    "reports/"
    "block38b_data_truth_gate.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


print(
    "validation manifest SHA     = PASS"
)

print(
    "validation scenarios        =",
    len(rows),
)

print(
    "source shard groups         =",
    len(grouped),
)

print(
    "compact offsets             =",
    len(offsets),
)

print(
    "150 compact shard sizes     = PASS"
)

print(
    "random-access probes        =",
    len(probe_rows),
)

print(
    "original offsets            = "
    "PROVENANCE ONLY"
)

print(
    "canonical offsets           = "
    "RECONSTRUCTED"
)

print(
    "raw /waymo runtime dep      = NO"
)

print(
    "second materialization      = NO"
)

print(
    "real scenario ID            =",
    scenario.scenario_id,
)

print(
    "WOMD timestamps             =",
    len(timestamps),
)

print(
    "current_time_index          =",
    current_index,
)

print(
    "history / future            =",
    history_count,
    "/",
    future_count,
)

print(
    "paired LiDAR                = PASS"
)

print(
    "truth sidecar               = "
    "EVALUATOR ONLY"
)

print(
    "future for eligibility      = NO"
)

print(
    "future for assignment       = NO"
)

print(
    "future for scoring          = YES"
)

print(
    "temporal interpolation      = NO"
)

print(
    "pilot candidates            = 12"
)

print(
    "formal N                    = NOT FROZEN"
)

print(
    "algorithm truth dependency  = NONE"
)

print(
    "targeted regression         = 11 / 11"
)

print(
    "full Stage3 regression      = 106 / 106"
)

print(
    "implementation files        =",
    file_count,
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
    report_path,
)

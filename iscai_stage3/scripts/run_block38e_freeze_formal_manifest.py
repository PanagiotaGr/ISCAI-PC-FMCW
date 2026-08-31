from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from iscai_stage3.validation import (
    build_compact_motion_offset_index,
    read_motion_scenario,
    read_validation_manifest,
    sha256_file,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    /
    "iscai_stage3"
)

PAIRED = (
    ROOT
    /
    "data/paired_womd_lidar_v1_3_0"
)

MANIFEST = (
    ROOT
    /
    "iscai_data_prep/manifests/"
    "selected_validation.jsonl"
)

CONFIG = (
    STAGE3
    /
    "configs/stage3_real_validation.json"
)

OUTPUT = (
    STAGE3
    /
    "artifacts/block38e/"
    "formal_validation_120.jsonl"
)

INVENTORY = (
    STAGE3
    /
    "artifacts/block38e/"
    "causal_class_inventory.json"
)

REPORT = (
    STAGE3
    /
    "reports/block38e_formal_manifest_gate.json"
)

EXPECTED_MANIFEST_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

FORMAL_N = 120
PER_STRATUM = 40

STRATA = (
    "cyclist",
    "pedestrian_no_cyclist",
    "vehicle_only",
)


def class_name(
    object_type: int,
) -> str:
    try:
        return (
            scenario_pb2.Track
            .ObjectType
            .Name(
                int(object_type)
            )
        )
    except Exception:
        return (
            f"TYPE_UNKNOWN_{int(object_type)}"
        )


def current_class_counts(
    scenario,
) -> dict[str, int]:
    """
    Strictly current/anchor-only.

    No state after current_time_index
    is inspected.
    """

    anchor = int(
        scenario.current_time_index
    )

    if anchor != 10:
        raise RuntimeError(
            "Unexpected WOMD "
            "current_time_index."
        )

    counts = Counter()

    for index, track in enumerate(
        scenario.tracks
    ):
        if index == int(
            scenario.sdc_track_index
        ):
            continue

        if anchor >= len(
            track.states
        ):
            raise RuntimeError(
                "Track/state alignment "
                "failure."
            )

        state = track.states[
            anchor
        ]

        if not bool(
            state.valid
        ):
            continue

        counts[
            class_name(
                track.object_type
            )
        ] += 1

    return dict(
        sorted(
            counts.items()
        )
    )


def assign_stratum(
    counts: dict[str, int],
) -> str | None:
    """
    Mutually-exclusive causal strata.

    Priority is deliberate:
      cyclist
      pedestrian without cyclist
      vehicle without pedestrian/cyclist
    """

    cyclist = (
        counts.get(
            "TYPE_CYCLIST",
            0,
        )
        > 0
    )

    pedestrian = (
        counts.get(
            "TYPE_PEDESTRIAN",
            0,
        )
        > 0
    )

    vehicle = (
        counts.get(
            "TYPE_VEHICLE",
            0,
        )
        > 0
    )

    if cyclist:
        return "cyclist"

    if pedestrian:
        return (
            "pedestrian_no_cyclist"
        )

    if vehicle:
        return "vehicle_only"

    return None


print(
    "===== Stage3 Block 3.8E "
    "causal formal-manifest freeze ====="
)

if sha256_file(
    MANIFEST
) != EXPECTED_MANIFEST_SHA:
    raise SystemExit(
        "FAIL: frozen validation "
        "manifest SHA changed."
    )

rows = read_validation_manifest(
    MANIFEST
)

if len(rows) != 44097:
    raise SystemExit(
        "FAIL: validation scenario "
        "count changed."
    )

offsets = (
    build_compact_motion_offset_index(
        rows
    )
)

if len(offsets) != len(rows):
    raise SystemExit(
        "FAIL: compact offset index."
    )


# ============================================================
# Full causal/current-only class inventory
# ============================================================

start = time.perf_counter()

buckets = {
    name: []
    for name in STRATA
}

global_anchor_classes = Counter()

scanned = 0
unstratified = 0

for number, row in enumerate(
    rows,
    start=1,
):
    scenario = read_motion_scenario(
        row,
        paired_root=PAIRED,
        compact_record_offset=(
            offsets[
                row.scenario_id
            ]
        ),
    )

    counts = current_class_counts(
        scenario
    )

    global_anchor_classes.update(
        counts
    )

    stratum = assign_stratum(
        counts
    )

    record = {
        "scenario_id":
            row.scenario_id,

        "selection_hash":
            row.selection_hash,

        "source_shard":
            row.source_shard,

        "compact_record_offset":
            offsets[
                row.scenario_id
            ],

        "anchor_class_counts":
            counts,

        "stratum":
            stratum,
    }

    if stratum is None:
        unstratified += 1
    else:
        buckets[
            stratum
        ].append(record)

    scanned += 1

    if (
        number % 5000
        ==
        0
    ):
        print(
            f"scanned {number}/44097",
            flush=True,
        )


scan_runtime_s = (
    time.perf_counter()
    -
    start
)


# ============================================================
# Availability gate BEFORE selection
# ============================================================

availability = {
    name: len(
        buckets[name]
    )
    for name in STRATA
}

print(
    "stratum availability =",
    availability,
)

for name in STRATA:
    if (
        availability[name]
        <
        PER_STRATUM
    ):
        raise SystemExit(
            "FAIL: insufficient causal "
            f"{name} scenes: "
            f"{availability[name]} "
            f"< {PER_STRATUM}"
        )


# ============================================================
# Deterministic within-stratum SHA selection
# ============================================================

selected = []

for name in STRATA:
    ordered = sorted(
        buckets[name],
        key=lambda item: (
            item[
                "selection_hash"
            ],
            item[
                "scenario_id"
            ],
        ),
    )

    chosen = ordered[
        :PER_STRATUM
    ]

    if len(chosen) != (
        PER_STRATUM
    ):
        raise RuntimeError(
            "Formal stratum count "
            "changed unexpectedly."
        )

    for rank, item in enumerate(
        chosen,
        start=1,
    ):
        selected.append({
            **item,

            "stratum_rank":
                rank,

            "formal_selection_policy": (
                "current_anchor_class_"
                "stratum_then_ascending_"
                "sha256_scenario_id"
            ),

            "future_used":
                False,

            "performance_used":
                False,

            "degraded_outcome_used":
                False,
        })


# Fixed presentation/order:
# cyclist -> pedestrian -> vehicle,
# then SHA rank within each stratum.

if len(selected) != FORMAL_N:
    raise RuntimeError(
        "Formal selection count "
        "is not 120."
    )

ids = [
    item["scenario_id"]
    for item in selected
]

if len(ids) != len(
    set(ids)
):
    raise RuntimeError(
        "Duplicate formal scenario."
    )


# ============================================================
# Freeze manifest
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    encoding="utf-8",
) as stream:
    for formal_rank, item in enumerate(
        selected,
        start=1,
    ):
        item = {
            "formal_rank":
                formal_rank,
            **item,
        }

        stream.write(
            json.dumps(
                item,
                sort_keys=True,
            )
            +
            "\n"
        )


formal_sha = sha256_file(
    OUTPUT
)


# ============================================================
# Selection-only anchor coverage report
# ============================================================

selected_class_totals = Counter()
selected_strata = Counter()

for item in selected:
    selected_class_totals.update(
        item[
            "anchor_class_counts"
        ]
    )

    selected_strata[
        item["stratum"]
    ] += 1


inventory_report = {
    "scanned_validation_scenarios":
        scanned,

    "scan_semantics":
        "current_anchor_only",

    "future_state_access":
        False,

    "tracks_to_predict_used":
        False,

    "model_outputs_used":
        False,

    "degraded_measurements_used":
        False,

    "global_anchor_class_totals":
        dict(
            sorted(
                global_anchor_classes
                .items()
            )
        ),

    "stratum_availability":
        availability,

    "unstratified_scenarios":
        unstratified,

    "scan_runtime_s":
        scan_runtime_s,
}

INVENTORY.write_text(
    json.dumps(
        inventory_report,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# Update formal config only NOW, after availability gate
# ============================================================

config = json.loads(
    CONFIG.read_text(
        encoding="utf-8"
    )
)

formal = config[
    "formal_validation"
]

if formal["N"] is not None:
    raise RuntimeError(
        "Formal N was already frozen."
    )

formal[
    "N"
] = FORMAL_N

formal[
    "scenario_ids"
] = ids

formal[
    "manifest_status"
] = "FROZEN"

formal[
    "manifest"
] = str(
    OUTPUT
)

formal[
    "manifest_sha256"
] = formal_sha

formal[
    "selection_policy"
] = (
    "40 cyclist-containing + "
    "40 pedestrian-containing/no-cyclist + "
    "40 vehicle-only; current/anchor metadata "
    "only; ascending SHA256(scenario_id) "
    "within stratum"
)

formal[
    "future_based_selection"
] = False

formal[
    "performance_based_selection"
] = False

formal[
    "degraded_outcome_based_selection"
] = False


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
# Final self-check
# ============================================================

reloaded = [
    json.loads(line)
    for line in OUTPUT.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if len(reloaded) != 120:
    raise RuntimeError(
        "Frozen manifest reload count "
        "is not 120."
    )

reloaded_strata = Counter(
    item["stratum"]
    for item in reloaded
)

expected_strata = {
    "cyclist": 40,
    "pedestrian_no_cyclist": 40,
    "vehicle_only": 40,
}

if dict(
    reloaded_strata
) != expected_strata:
    raise RuntimeError(
        "Frozen formal stratum counts "
        f"changed: {reloaded_strata}"
    )

if any(
    item["future_used"]
    or
    item["performance_used"]
    or
    item[
        "degraded_outcome_used"
    ]
    for item in reloaded
):
    raise RuntimeError(
        "Forbidden formal-selection "
        "dependency detected."
    )


report = {
    "stage": 3,
    "block": "3.8E",
    "status": "PASS",

    "formal_N": 120,

    "runtime_basis": {
        "block38d_scenario_runtime_median_ms":
            5125.356565520633,

        "block38d_scenario_runtime_p95_ms":
            20723.7813801039,

        "estimated_120_scene_median_minutes":
            (
                120
                *
                5125.356565520633
                /
                1000.0
                /
                60.0
            ),

        "estimated_120_scene_p95_budget_minutes":
            (
                120
                *
                20723.7813801039
                /
                1000.0
                /
                60.0
            ),
    },

    "selection": {
        "policy": (
            "current-anchor class "
            "stratification + deterministic "
            "SHA ordering"
        ),

        "strata": expected_strata,

        "future_used":
            False,

        "tracks_to_predict_used":
            False,

        "performance_used":
            False,

        "degraded_outcome_used":
            False,
    },

    "full_validation_inventory": {
        "scenarios":
            scanned,

        "stratum_availability":
            availability,

        "anchor_class_totals":
            dict(
                sorted(
                    global_anchor_classes
                    .items()
                )
            ),

        "runtime_s":
            scan_runtime_s,
    },

    "formal_selection_anchor_coverage": {
        "strata":
            dict(
                selected_strata
            ),

        "class_totals":
            dict(
                sorted(
                    selected_class_totals
                    .items()
                )
            ),
    },

    "formal_manifest": {
        "path":
            str(OUTPUT),

        "sha256":
            formal_sha,

        "scenario_count":
            len(selected),
    },
}


REPORT.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)


print()
print(
    "===== Stage3 Block 3.8E gate ====="
)

print(
    "validation scenarios scanned =",
    scanned,
)

print(
    "future state access          = NO"
)

print(
    "tracks_to_predict used       = NO"
)

print(
    "performance used             = NO"
)

print(
    "degraded outcome used        = NO"
)

print(
    "stratum availability         =",
    availability,
)

print(
    "formal strata                =",
    dict(
        selected_strata
    ),
)

print(
    "formal anchor classes        =",
    dict(
        sorted(
            selected_class_totals
            .items()
        )
    ),
)

print(
    "formal N                     =",
    FORMAL_N,
)

print(
    "formal manifest SHA256       =",
    formal_sha,
)

print(
    "causal inventory runtime s   =",
    round(
        scan_runtime_s,
        3,
    ),
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)

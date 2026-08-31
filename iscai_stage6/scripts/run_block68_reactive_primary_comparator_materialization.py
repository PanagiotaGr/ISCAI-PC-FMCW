from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np

from iscai_stage6.adb.metric_semantics import (
    vehicle_shadow_zone_violation,
)


S6 = Path("/home/agni/waymo/iscai_stage6")

# ============================================================
# FROZEN INPUTS
# ============================================================

REACTIVE_RAW = (
    S6
    / "artifacts/block68/"
      "block68_reactive_only_ni_values.jsonl"
)

CURRENT_BINDING = (
    S6
    / "artifacts/block68/stage4_preoutcome/"
      "stage4_current_state_scenario_binding.jsonl"
)

T0_MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

CACHE_MANIFEST = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "stage3_fixed_cache_manifest.jsonl"
)

CACHE_FREEZE = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "stage3_fixed_cache_freeze.json"
)

REACTIVE_CONTRACT = (
    S6
    / "configs/"
      "stage6_reactive_development_scoring_contract.json"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

STAGE4_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage4_temporal_rate_selection_freeze.json"
)

ACCEPTANCE_POLICY = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)


# ============================================================
# WRITE-ONCE OUTPUTS
# ============================================================

AUGMENTED = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_reactive_primary_comparator_augmented.jsonl"
)

VEHICLE_ROWS = (
    S6
    / "artifacts/block68/primary_development_gate/"
      "block68_reactive_vehicle_violation_values.jsonl"
)

FREEZE = (
    S6
    / "reports/"
      "block68_reactive_primary_comparator_materialization.json"
)


EXPECTED_SHA = {
    "reactive_raw":
        "3a81d6d39b29a2d30247f30f06f0e02bedd47ecc754a0eb550611947f6e9b0d5",

    "current_binding":
        "3a1c689484fdbd8dbf763cfb0a52b3a987a5930a236fe9692f0b084cb178387d",

    "t0_maps":
        "7c5d63c76c3d9ef74adbbd7a68e3ce7483b1dd80fd262cba06f6ce1909994993",

    "cache_manifest":
        "9607f5bd6022da5471167650f652556dca8381b241adf0b7faaf06bd27edc041",

    "cache_freeze":
        "6733ba1641afc341910bd5cc822a034103585706f32db4758d606b7bf2bf786d",

    "reactive_contract":
        "454d22a2f067025fd8dde877546b7305c4357000126585c860a4a2d9ef3535d5",

    "metric_runtime":
        "bf8358b5a7edfe40ffac2718ca7cd5af6e7b5fdb3ff8a2823caad6d0f40f057a",

    "stage4_freeze":
        "780e134a66153c52783bc18d55f607c891b9a6983f8cfc8ac78e2e7367874885",

    "acceptance_policy":
        "e103466b1a6eb62be4e6746029147cbda9671b05130acba5210a3ab7328e6b81",
}

EXPECTED_TESTS = 224

GRID_2D = (
    501,
    301,
)

GRID_3D = (
    4,
    501,
    301,
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
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


def exact(path: Path, expected: str, label: str):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(path)

    require(
        actual == expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, raw in enumerate(
            stream,
            start=1,
        ):

            if not raw.strip():
                continue

            try:
                value = json.loads(raw)

            except Exception as exc:
                raise RuntimeError(
                    f"Bad JSONL {path}:{line_number}"
                ) from exc

            require(
                isinstance(value, dict),
                (
                    "Expected object JSONL row at "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(value)

    return rows


def canonical_jsonl_bytes(rows):
    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once(path: Path, payload: bytes):
    if path.exists():

        require(
            path.read_bytes() == payload,
            (
                "Existing write-once artifact differs: "
                f"{path}"
            ),
        )

        return "EXISTING_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)
    temporary.replace(path)

    return "WRITTEN"


def array_content_sha(arrays):
    digest = sha256()

    for name in sorted(arrays):

        array = np.ascontiguousarray(
            arrays[name]
        )

        digest.update(
            name.encode("utf-8")
        )

        digest.update(
            str(array.dtype).encode("utf-8")
        )

        digest.update(
            json.dumps(
                list(array.shape)
            ).encode("utf-8")
        )

        digest.update(
            array.tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def metric_key_exists(value, target):
    if isinstance(value, dict):

        for key, child in value.items():

            if str(key) == target:
                return True

            if metric_key_exists(
                child,
                target,
            ):
                return True

    elif isinstance(value, list):

        for child in value:

            if metric_key_exists(
                child,
                target,
            ):
                return True

    return False


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"observed {count}."
        ),
    )

    return count


def finite_or_none(value):
    value = float(value)

    if math.isfinite(value):
        return value

    return None


def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("REACTIVE PRIMARY-COMPARATOR MATERIALIZATION")
    print("MISSING VEHICLE METRIC ONLY / NO POLICY CHANGE / NO GATE YET")
    print("=" * 78)

    # ========================================================
    # A. Exact source boundary
    # ========================================================

    print()
    print("===== A. EXACT FROZEN SOURCE BOUNDARY =====")

    paths = {
        "reactive_raw":
            REACTIVE_RAW,

        "current_binding":
            CURRENT_BINDING,

        "t0_maps":
            T0_MAPS,

        "cache_manifest":
            CACHE_MANIFEST,

        "cache_freeze":
            CACHE_FREEZE,

        "reactive_contract":
            REACTIVE_CONTRACT,

        "metric_runtime":
            METRIC_RUNTIME,

        "stage4_freeze":
            STAGE4_FREEZE,

        "acceptance_policy":
            ACCEPTANCE_POLICY,
    }

    seals = {}

    for key, path in paths.items():

        seals[key] = exact(
            path,
            EXPECTED_SHA[key],
            key,
        )

        print(
            f"{key:22s} = EXACT PASS"
        )

    stage4 = read_json(
        STAGE4_FREEZE
    )

    require(
        stage4[
            "scientific_boundary"
        ][
            "primary_acceptance_tested"
        ]
        is False,
        (
            "Primary gate unexpectedly already "
            "marked tested."
        ),
    )

    require(
        stage4[
            "scientific_boundary"
        ][
            "formal_evaluation"
        ]
        is False,
        "Formal evaluation unexpectedly opened.",
    )

    print(
        "Stage1/2/3/4 policy = IMMUTABLE"
    )

    print(
        "primary gate         = NOT YET EVALUATED"
    )

    print(
        "formal evaluation    = NO"
    )

    # ========================================================
    # B. Confirm exact schema gap
    # ========================================================

    print()
    print("===== B. EXACT REACTIVE-RAW SCHEMA GAP =====")

    reactive_rows = read_jsonl(
        REACTIVE_RAW
    )

    require(
        len(reactive_rows) == 120,
        (
            "Frozen reactive raw rows !=120: "
            f"{len(reactive_rows)}"
        ),
    )

    missing_vehicle = sum(
        not metric_key_exists(
            row,
            "vehicle_shadow_zone_violation",
        )
        for row in reactive_rows
    )

    require(
        missing_vehicle == 120,
        (
            "Expected vehicle metric to be absent "
            "from all 120 NI-only rows; "
            f"missing={missing_vehicle}"
        ),
    )

    require(
        all(
            "scenario_id" in row
            for row in reactive_rows
        ),
        (
            "Reactive raw rows do not all contain "
            "scenario_id."
        ),
    )

    reactive_ids = [
        str(
            row["scenario_id"]
        )
        for row in reactive_rows
    ]

    require(
        len(
            set(
                reactive_ids
            )
        )
        ==
        120,
        "Reactive scenario IDs not unique.",
    )

    print(
        "reactive raw rows              = 120"
    )

    print(
        "vehicle metric present         = 0 / 120"
    )

    print(
        "overmask / VRU NI source       = PRESERVED"
    )

    print(
        "schema gap confirmed           = YES"
    )

    # ========================================================
    # C. Exact scenario join
    # ========================================================

    print()
    print("===== C. EXACT SCENARIO / ACTUATOR / ORACLE JOIN =====")

    binding = read_jsonl(
        CURRENT_BINDING
    )

    manifest = read_jsonl(
        CACHE_MANIFEST
    )

    require(
        len(binding) == 120,
        "Current-state binding !=120.",
    )

    require(
        len(manifest) == 120,
        "Cache manifest !=120.",
    )

    binding_by_id = {
        str(row["scenario_id"]):
            row
        for row in binding
    }

    manifest_by_id = {
        str(row["scenario_id"]):
            row
        for row in manifest
    }

    require(
        len(binding_by_id) == 120,
        "Binding IDs not unique.",
    )

    require(
        len(manifest_by_id) == 120,
        "Manifest IDs not unique.",
    )

    require(
        set(reactive_ids)
        ==
        set(binding_by_id)
        ==
        set(manifest_by_id),
        (
            "Reactive / t0 binding / evaluator-cache "
            "scenario populations differ."
        ),
    )

    maps = np.load(
        T0_MAPS,
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        maps.shape
        ==
        (
            120,
            501,
            301,
        ),
        (
            "Frozen t0 map stack shape mismatch: "
            f"{maps.shape}"
        ),
    )

    require(
        maps.dtype
        ==
        np.dtype(np.float64),
        (
            "Frozen t0 map dtype mismatch: "
            f"{maps.dtype}"
        ),
    )

    print(
        "scenario identity join = 120 / 120 EXACT"
    )

    print(
        "t0 action source       = FROZEN"
    )

    print(
        "oracle vehicle source  = FROZEN CACHE"
    )

    print(
        "WOMD reopened          = NO"
    )

    print(
        "P_occ reopened         = NO"
    )

    # ========================================================
    # D. Materialize missing reactive vehicle metric
    # ========================================================

    print()
    print("===== D. AUTHORITATIVE REACTIVE VEHICLE MATERIALIZATION =====")

    vehicle_rows = []
    augmented_rows = []

    finite_values = []

    for position, original in enumerate(
        reactive_rows,
        start=1,
    ):

        scenario_id = str(
            original[
                "scenario_id"
            ]
        )

        binding_row = binding_by_id[
            scenario_id
        ]

        manifest_row = manifest_by_id[
            scenario_id
        ]

        row_index = int(
            binding_row[
                "t0_map_row_index"
            ]
        )

        require(
            0 <= row_index < 120,
            (
                "Invalid t0 row index for "
                f"{scenario_id}: {row_index}"
            ),
        )

        current = np.asarray(
            maps[
                row_index
            ],
            dtype=np.float64,
        )

        require(
            current.shape
            ==
            GRID_2D,
            (
                "t0 map shape mismatch for "
                f"{scenario_id}: {current.shape}"
            ),
        )

        require(
            np.all(
                np.isfinite(current)
            ),
            (
                "Non-finite t0 map for "
                f"{scenario_id}."
            ),
        )

        # Frozen original_reactive_ADB scoring contract:
        # hold the causal t0 action unchanged at all four
        # evaluator horizons.
        schedule = np.broadcast_to(
            current[
                None,
                ...,
            ],
            GRID_3D,
        )

        dim = (
            schedule
            <
            1.0
        )

        cache_path = Path(
            manifest_row[
                "npz_path"
            ]
        )

        require(
            cache_path.is_file(),
            (
                "Missing frozen evaluator cache: "
                f"{cache_path}"
            ),
        )

        with np.load(
            cache_path,
            allow_pickle=False,
        ) as loaded:

            required = {
                "mask_vehicle",
                "mask_pedestrian",
                "mask_cyclist",
                "oracle_all",
                "oracle_vehicle",
                "pedestrian_region",
                "cyclist_region",
                "vehicle_surrogate",
            }

            require(
                set(loaded.files)
                ==
                required,
                (
                    "Frozen cache fields changed for "
                    f"{scenario_id}: {loaded.files}"
                ),
            )

            arrays = {
                name:
                    np.asarray(
                        loaded[name],
                        dtype=bool,
                    )
                for name in required
            }

        content_sha = array_content_sha(
            arrays
        )

        require(
            content_sha
            ==
            str(
                manifest_row[
                    "array_content_sha256"
                ]
            ),
            (
                "Frozen cache content hash mismatch "
                f"for {scenario_id}."
            ),
        )

        oracle_vehicle = arrays[
            "oracle_vehicle"
        ]

        require(
            oracle_vehicle.shape
            ==
            GRID_3D,
            (
                "oracle_vehicle shape mismatch for "
                f"{scenario_id}: "
                f"{oracle_vehicle.shape}"
            ),
        )

        value = float(
            vehicle_shadow_zone_violation(
                dim,
                oracle_vehicle,
            )
        )

        json_value = finite_or_none(
            value
        )

        if math.isfinite(value):
            finite_values.append(
                value
            )

        vehicle_row = {
            "scenario_id":
                scenario_id,

            "vehicle_shadow_zone_violation":
                json_value,

            "reactive_action":
                "frozen_original_reactive_t0",

            "score_schedule":
                "hold_t0_action_at_0.1_0.3_0.5_1.0s",

            "decision_uses_future":
                False,

            "oracle_vehicle_role":
                "constructed_future_evaluator_only",

            "metric_runtime_sha256":
                EXPECTED_SHA[
                    "metric_runtime"
                ],
        }

        vehicle_rows.append(
            vehicle_row
        )

        # Preserve the entire original frozen NI row and
        # add exactly one previously absent authoritative key.
        augmented = dict(
            original
        )

        require(
            "vehicle_shadow_zone_violation"
            not in
            augmented,
            (
                "Unexpected top-level vehicle metric "
                f"already exists for {scenario_id}."
            ),
        )

        augmented[
            "vehicle_shadow_zone_violation"
        ] = json_value

        augmented_rows.append(
            augmented
        )

        if (
            position == 1
            or
            position % 20 == 0
            or
            position == 120
        ):
            print(
                (
                    f"reactive vehicle "
                    f"{position:03d}/120 "
                    f"{scenario_id} PASS"
                ),
                flush=True,
            )

    require(
        finite_values,
        (
            "No finite reactive vehicle "
            "shadow-zone violations."
        ),
    )

    vehicle_mean = float(
        math.fsum(
            finite_values
        )
        /
        len(
            finite_values
        )
    )

    print()
    print(
        "finite vehicle support =",
        len(
            finite_values
        ),
    )

    print(
        "reactive vehicle mean  =",
        repr(
            vehicle_mean
        ),
    )

    print(
        "selection/tuning       = NONE"
    )

    # ========================================================
    # E. Write-once augmented comparator
    # ========================================================

    print()
    print("===== E. WRITE-ONCE REACTIVE PRIMARY COMPARATOR =====")

    vehicle_state = write_once(
        VEHICLE_ROWS,
        canonical_jsonl_bytes(
            vehicle_rows
        ),
    )

    augmented_state = write_once(
        AUGMENTED,
        canonical_jsonl_bytes(
            augmented_rows
        ),
    )

    vehicle_sha = file_sha(
        VEHICLE_ROWS
    )

    augmented_sha = file_sha(
        AUGMENTED
    )

    print(
        "vehicle rows state =",
        vehicle_state,
    )

    print(
        "vehicle rows SHA   =",
        vehicle_sha,
    )

    print(
        "augmented state    =",
        augmented_state,
    )

    print(
        "augmented SHA      =",
        augmented_sha,
    )

    # ========================================================
    # F. Regression
    # ========================================================

    print()
    print("===== F. REGRESSION =====")

    tests = run_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    # ========================================================
    # G. Freeze provenance
    # ========================================================

    freeze = {
        "stage":
            6,

        "block":
            "6.8_reactive_primary_comparator_materialization",

        "status":
            "PASS_REACTIVE_PRIMARY_COMPARATOR_MATERIALIZED_FROZEN",

        "purpose":
            (
                "materialize the missing original_reactive_ADB "
                "vehicle_shadow_zone_violation required by the "
                "already-frozen primary acceptance rule"
            ),

        "schema_gap": {
            "original_reactive_NI_artifact":
                str(
                    REACTIVE_RAW
                ),

            "original_reactive_NI_sha256":
                EXPECTED_SHA[
                    "reactive_raw"
                ],

            "vehicle_metric_present_in_original":
                False,

            "original_NI_values_modified":
                False,
        },

        "vehicle_metric": {
            "name":
                "vehicle_shadow_zone_violation",

            "reactive_action":
                (
                    "frozen causal t0 "
                    "original_reactive_ADB map"
                ),

            "schedule":
                (
                    "same t0 action held unchanged "
                    "at 0.1,0.3,0.5,1.0 seconds"
                ),

            "oracle":
                (
                    "frozen constructed future "
                    "oracle_vehicle evaluator support"
                ),

            "metric_runtime":
                str(
                    METRIC_RUNTIME
                ),

            "metric_runtime_sha256":
                EXPECTED_SHA[
                    "metric_runtime"
                ],

            "finite_support":
                len(
                    finite_values
                ),

            "scenario_macro_mean":
                vehicle_mean,
        },

        "outputs": {
            "vehicle_values": {
                "path":
                    str(
                        VEHICLE_ROWS
                    ),

                "sha256":
                    vehicle_sha,

                "rows":
                    120,
            },

            "augmented_primary_comparator": {
                "path":
                    str(
                        AUGMENTED
                    ),

                "sha256":
                    augmented_sha,

                "rows":
                    120,
            },
        },

        "scientific_boundary": {
            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage3_modified":
                False,

            "Stage4_modified":
                False,

            "acceptance_rule_modified":
                False,

            "NI_bounds_modified":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,

            "new_parameter_tuning":
                False,
        },

        "source_seals":
            seals,

        "Stage6_regression":
            tests,

        "next":
            (
                "repair primary-gate comparator source "
                "to use this augmented frozen artifact "
                "and execute the existing frozen gate"
            ),
    }

    write_once(
        FREEZE,
        canonical_json_bytes(
            freeze
        ),
    )

    freeze_sha = file_sha(
        FREEZE
    )

    print()
    print("=" * 78)
    print("REACTIVE PRIMARY COMPARATOR MATERIALIZATION — FINAL")
    print("=" * 78)

    print(
        "original NI artifact        = UNCHANGED"
    )

    print(
        "missing vehicle metric      = MATERIALIZED"
    )

    print(
        "vehicle comparator source   = FROZEN t0 HOLD"
    )

    print(
        "vehicle oracle source       = FROZEN EVALUATOR CACHE"
    )

    print(
        "reactive vehicle mean       =",
        repr(
            vehicle_mean
        ),
    )

    print(
        "finite vehicle support      =",
        len(
            finite_values
        ),
    )

    print(
        "Stage1/2/3/4 policy         = IMMUTABLE"
    )

    print(
        "acceptance rule             = UNCHANGED"
    )

    print(
        "primary acceptance tested   = NO"
    )

    print(
        "formal evaluation           = NO"
    )

    print(
        "Stage6 regression           =",
        f"{tests} / {tests} PASS",
    )

    print(
        "augmented comparator SHA    =",
        augmented_sha,
    )

    print(
        "materialization freeze SHA  =",
        freeze_sha,
    )

    print(
        "STATUS = "
        "PASS_REACTIVE_PRIMARY_COMPARATOR_MATERIALIZED_FROZEN"
    )

    print(
        "report =",
        FREEZE,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "REACTIVE PRIMARY COMPARATOR MATERIALIZATION = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage1/2/3/4 policy    = STILL IMMUTABLE"
    )

    print(
        "acceptance rule        = UNCHANGED"
    )

    print(
        "primary gate           = NOT EVALUATED"
    )

    print(
        "formal evaluation      = NO"
    )

    print()
    print(
        "Do not substitute a guessed reactive vehicle "
        "number and do not retune the predictive policy."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

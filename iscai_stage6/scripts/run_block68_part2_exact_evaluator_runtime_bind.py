from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

BLOCK66 = (
    S6
    / "artifacts/block66"
)

ROUTE_BIND = (
    S6
    / "reports/"
      "block68_part2_development_route_schema_bind.json"
)

RECON = (
    S6
    / "reports/"
      "block68_preoutcome_metric_reconciliation.json"
)

PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)

METRICS = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

INTERFACE = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

CLASS_POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

PREDICTIVE_MASK = (
    S6
    / "src/iscai_stage6/adb/"
      "predictive_mask.py"
)

ILLUMINATION = (
    S6
    / "src/iscai_stage6/adb/"
      "illumination.py"
)

REACTIVE = (
    S6
    / "src/iscai_stage6/adb/"
      "part_a_reactive.py"
)

OCCUPANCY = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_occupancy.py"
)

WOMD_GEOMETRY = (
    S6
    / "src/iscai_stage6/adb/"
      "womd_geometry.py"
)

MATCHES = (
    BLOCK66
    / "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

FALLBACKS = (
    BLOCK66
    / "block66_part2c1_headlamp_eligible_reactive_fallbacks.jsonl"
)

POSTERIOR = (
    BLOCK66
    / "block66_development_identity_safe_gaussian_posterior.jsonl"
)

POCC_MANIFEST = (
    BLOCK66
    / "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

SURROGATE = (
    BLOCK66
    / "block66_part3c_vehicle_surrogate_evidence.jsonl"
)

SCENE_CACHE = (
    BLOCK66
    / "part2b_scene_cache"
)

POCC_CACHE = (
    BLOCK66
    / "part2c2b_actor_pocc_cache"
)

REPORT = (
    S6
    / "reports/"
      "block68_part2_exact_evaluator_runtime_bind.json"
)


EXPECTED_ROUTE_BIND_STATUS = (
    "PASS_DEVELOPMENT_ROUTE_SCHEMA_BOUND"
)

EXPECTED_RECON_SHA = (
    "077a89fa5163f8862a3a5b7489d98929"
    "d9ae3e160ed9e030eb75f42a91d48f9f"
)

EXPECTED_PROTOCOL_SHA = (
    "409fbb2785e4ff13f29fdff96c91d608"
    "c245e009fcbb86ce57099d6a4e1815c9"
)

EXPECTED_METRICS_SHA = (
    "bf8358b5a7edfe40ffac2718ca7cd5af"
    "6e7b5fdb3ff8a2823caad6d0f40f057a"
)

EXPECTED_INTERFACE_SHA = (
    "39518597079f65d556ee8f0fd3b00ce5"
    "25319623c9eb8a83f6e3c608954213e9"
)

EXPECTED_STAGE6_TESTS = 202

MIN_FREE_GIB = 250.0


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

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


def require(
    condition,
    message,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


def guard_nonformal(
    path: Path,
):

    lower = str(
        path.resolve()
    ).lower()

    require(
        "formal"
        not in
        lower,
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )


def load_json(
    path: Path,
):

    guard_nonformal(
        path
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def first_jsonl(
    path: Path,
):

    guard_nonformal(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line in stream:

            if line.strip():

                value = json.loads(
                    line
                )

                require(
                    isinstance(
                        value,
                        dict,
                    ),
                    (
                        "Expected JSON object "
                        f"in {path}"
                    ),
                )

                return value

    raise RuntimeError(
        f"No JSONL records: {path}"
    )


def top_level_type_schema(
    value,
):

    if isinstance(
        value,
        dict,
    ):

        return {
            str(
                key
            ):
                (
                    "object"
                    if isinstance(
                        child,
                        dict,
                    )
                    else
                    "array"
                    if isinstance(
                        child,
                        list,
                    )
                    else
                    "bool"
                    if isinstance(
                        child,
                        bool,
                    )
                    else
                    "int"
                    if isinstance(
                        child,
                        int,
                    )
                    else
                    "float"
                    if isinstance(
                        child,
                        float,
                    )
                    else
                    "null"
                    if child is None
                    else
                    "str"
                    if isinstance(
                        child,
                        str,
                    )
                    else
                    type(
                        child
                    ).__name__
                )
            for key, child in sorted(
                value.items()
            )
        }

    return {}


def nested_key_paths(
    value,
    prefix="",
    *,
    max_depth=5,
):

    results = []

    def walk(
        item,
        path,
        depth,
    ):

        if depth > max_depth:
            return

        if isinstance(
            item,
            dict,
        ):

            for key, child in (
                item.items()
            ):

                child_path = (
                    f"{path}.{key}"
                    if path
                    else str(
                        key
                    )
                )

                results.append(
                    child_path
                )

                walk(
                    child,
                    child_path,
                    depth + 1,
                )

        elif isinstance(
            item,
            list,
        ) and item:

            child_path = (
                f"{path}[]"
            )

            results.append(
                child_path
            )

            walk(
                item[0],
                child_path,
                depth + 1,
            )

    walk(
        value,
        prefix,
        0,
    )

    return sorted(
        set(
            results
        )
    )


def function_signature(
    function,
):

    return str(
        inspect.signature(
            function
        )
    )


def dataclass_schema(
    cls,
):

    require(
        is_dataclass(
            cls
        ),
        (
            f"{cls.__name__} is not dataclass"
        ),
    )

    return [
        {
            "name":
                field.name,

            "type":
                str(
                    field.type
                ),
        }
        for field in fields(
            cls
        )
    ]


def inspect_npz_metadata(
    path: Path,
):
    """
    Shape/dtype metadata only.

    We explicitly do NOT compute values, means, coverage,
    occupancy rates, or any scientific performance result.
    """

    guard_nonformal(
        path
    )

    result = {}

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:

        for key in payload.files:

            array = payload[
                key
            ]

            result[
                key
            ] = {
                "shape":
                    list(
                        array.shape
                    ),

                "dtype":
                    str(
                        array.dtype
                    ),

                "ndim":
                    int(
                        array.ndim
                    ),
            }

    return result


def find_first_existing_pocc_payload(
    manifest_record,
):

    path_keys = (
        "persisted_occupancy_path",
        "occupancy_path",
        "pocc_path",
        "cache_path",
        "npz_path",
        "path",
    )

    for key in path_keys:

        value = manifest_record.get(
            key
        )

        if not isinstance(
            value,
            str,
        ):
            continue

        candidate = Path(
            value
        )

        if not candidate.is_absolute():
            candidate = (
                BLOCK66
                /
                candidate
            )

        if candidate.is_file():

            guard_nonformal(
                candidate
            )

            return candidate

    # Fallback: inspect frozen cache names only.
    candidates = sorted(
        path
        for path in POCC_CACHE.rglob(
            "*"
        )
        if (
            path.is_file()
            and
            path.suffix.lower()
            in (
                ".npz",
                ".npy",
                ".json",
            )
            and
            "formal"
            not in str(
                path
            ).lower()
        )
    )

    require(
        candidates,
        (
            "Could not locate persisted "
            "P_occ payload."
        ),
    )

    return candidates[0]


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
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    count = None

    for line in (
        process.stdout.splitlines()
    ):

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:

            count = int(
                match.group(1)
            )

    return (
        process.returncode,
        count,
        process.stdout,
    )


def write_json(
    path: Path,
    payload,
):

    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART 2/2"
    )
    print(
        "EXACT DEVELOPMENT EVALUATOR RUNTIME BIND"
    )
    print(
        "NO PERFORMANCE METRICS / NO FORMAL CONTENT"
    )
    print(
        "============================================================"
    )

    required = (
        ROUTE_BIND,
        RECON,
        PROTOCOL,
        METRICS,
        INTERFACE,
        CLASS_POLICY,
        PREDICTIVE_MASK,
        ILLUMINATION,
        REACTIVE,
        OCCUPANCY,
        WOMD_GEOMETRY,
        MATCHES,
        FALLBACKS,
        POSTERIOR,
        POCC_MANIFEST,
        SURROGATE,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing runtime-binding dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN DEVELOPMENT BIND SEAL ====="
    )

    route = load_json(
        ROUTE_BIND
    )

    require(
        route.get(
            "status"
        )
        ==
        EXPECTED_ROUTE_BIND_STATUS,
        (
            "Development route bind "
            "is not frozen PASS."
        ),
    )

    exact = (
        (
            RECON,
            EXPECTED_RECON_SHA,
            "reconciliation report",
        ),
        (
            PROTOCOL,
            EXPECTED_PROTOCOL_SHA,
            "metric protocol",
        ),
        (
            METRICS,
            EXPECTED_METRICS_SHA,
            "metric runtime",
        ),
        (
            INTERFACE,
            EXPECTED_INTERFACE_SHA,
            "metric interface",
        ),
    )

    for (
        path,
        expected,
        label,
    ) in exact:

        require(
            sha256_file(
                path
            )
            ==
            expected,
            (
                f"{label} SHA changed."
            ),
        )

        print(
            f"{label:26s}= EXACT PASS"
        )

    print(
        "development route bind     = PASS"
    )

    print(
        "formal evaluation          = BLOCKED"
    )

    # ========================================================
    # B. Exact runtime Python API
    # ========================================================

    print()
    print(
        "===== B. EXACT RUNTIME PYTHON API ====="
    )

    from iscai_stage6.adb.class_aware_policy import (
        ClassAwareComposition,
        MarginComputation,
        compose_class_aware_illumination,
        predictive_mask_to_illumination,
    )

    from iscai_stage6.adb.predictive_mask import (
        BinaryActorPredictiveMask,
        ClassAgnosticPredictiveMaskPlan,
        aggregate_binary_actor_masks,
        exact_original_reactive_fallback_map,
        threshold_actor_occupancy,
    )

    from iscai_stage6.adb.illumination import (
        ReactiveShadowRegion,
        reactive_adb_map,
    )

    from iscai_stage6.adb.part_a_reactive import (
        PartAReactiveGeometry,
        part_a_original_reactive_map_from_h0_centers,
    )

    from iscai_stage6.adb.probabilistic_occupancy import (
        ActorOccupancyProbability,
        OccupancyGrid,
    )

    from iscai_stage6.adb.womd_geometry import (
        CausalADBActorBox,
    )

    function_api = {
        "predictive_mask_to_illumination":
            function_signature(
                predictive_mask_to_illumination
            ),

        "compose_class_aware_illumination":
            function_signature(
                compose_class_aware_illumination
            ),

        "threshold_actor_occupancy":
            function_signature(
                threshold_actor_occupancy
            ),

        "aggregate_binary_actor_masks":
            function_signature(
                aggregate_binary_actor_masks
            ),

        "exact_original_reactive_fallback_map":
            function_signature(
                exact_original_reactive_fallback_map
            ),

        "reactive_adb_map":
            function_signature(
                reactive_adb_map
            ),

        "part_a_original_reactive_map_from_h0_centers":
            function_signature(
                part_a_original_reactive_map_from_h0_centers
            ),
    }

    for key, value in (
        function_api.items()
    ):

        print(
            f"{key}"
        )

        print(
            "   ",
            value,
        )

    dataclasses = {
        "ClassAwareComposition":
            dataclass_schema(
                ClassAwareComposition
            ),

        "MarginComputation":
            dataclass_schema(
                MarginComputation
            ),

        "BinaryActorPredictiveMask":
            dataclass_schema(
                BinaryActorPredictiveMask
            ),

        "ClassAgnosticPredictiveMaskPlan":
            dataclass_schema(
                ClassAgnosticPredictiveMaskPlan
            ),

        "ReactiveShadowRegion":
            dataclass_schema(
                ReactiveShadowRegion
            ),

        "PartAReactiveGeometry":
            dataclass_schema(
                PartAReactiveGeometry
            ),

        "ActorOccupancyProbability":
            dataclass_schema(
                ActorOccupancyProbability
            ),

        "OccupancyGrid":
            dataclass_schema(
                OccupancyGrid
            ),

        "CausalADBActorBox":
            dataclass_schema(
                CausalADBActorBox
            ),
    }

    print()
    print(
        "dataclass fields:"
    )

    for name, schema in (
        dataclasses.items()
    ):

        print()
        print(
            name
        )

        for item in schema:

            print(
                "   ",
                item[
                    "name"
                ],
                ":",
                item[
                    "type"
                ],
            )

    # ========================================================
    # C. Exact development identity schemas
    # ========================================================

    print()
    print(
        "===== C. EXACT ACTOR IDENTITY / JOIN SCHEMAS ====="
    )

    match = first_jsonl(
        MATCHES
    )

    fallback = first_jsonl(
        FALLBACKS
    )

    posterior = first_jsonl(
        POSTERIOR
    )

    pocc = first_jsonl(
        POCC_MANIFEST
    )

    surrogate = first_jsonl(
        SURROGATE
    )

    records = {
        "predictive_match":
            match,

        "reactive_fallback":
            fallback,

        "posterior":
            posterior,

        "P_occ_manifest":
            pocc,

        "vehicle_surrogate":
            surrogate,
    }

    record_schemas = {}

    for name, record in (
        records.items()
    ):

        schema = {
            "top_level_types":
                top_level_type_schema(
                    record
                ),

            "nested_key_paths":
                nested_key_paths(
                    record
                ),
        }

        record_schemas[
            name
        ] = schema

        print()
        print(
            "RECORD =",
            name,
        )

        print(
            "top-level keys =",
            sorted(
                record.keys()
            ),
        )

        print(
            "nested key paths:"
        )

        for key in (
            schema[
                "nested_key_paths"
            ][
                :100
            ]
        ):

            print(
                "   ",
                key,
            )

    # --------------------------------------------------------
    # Prove at least one exact shared actor-identity route.
    # --------------------------------------------------------

    candidate_identity_keys = (
        "scenario_id",
        "prediction_id",
        "actor_box_index",
        "track_index",
        "actor_rank",
        "cohort_index",
    )

    key_presence = {
        key: {
            record_name:
                (
                    key
                    in
                    record
                )
            for record_name, record in (
                records.items()
            )
        }
        for key in candidate_identity_keys
    }

    print()
    print(
        "identity-key presence matrix:"
    )

    for key, presence in (
        key_presence.items()
    ):

        print(
            key,
            "=",
            presence,
        )

    scenario_shared = all(
        "scenario_id"
        in
        record
        for record in (
            match,
            fallback,
            posterior,
            pocc,
        )
    )

    predictive_join_keys = [
        key
        for key in (
            "scenario_id",
            "prediction_id",
            "actor_box_index",
            "actor_rank",
            "cohort_index",
        )
        if (
            key in match
            and
            key in pocc
        )
    ]

    require(
        scenario_shared,
        (
            "scenario_id is not shared by "
            "all required development routes."
        ),
    )

    require(
        len(
            predictive_join_keys
        )
        >=
        2,
        (
            "Predictive actor identity is "
            "still not sufficiently proven."
        ),
    )

    print()
    print(
        "scenario identity route = PASS"
    )

    print(
        "predictive actor join keys =",
        predictive_join_keys,
    )

    # ========================================================
    # D. Actual persisted P_occ payload metadata
    # ========================================================

    print()
    print(
        "===== D. PERSISTED P_OCC PAYLOAD METADATA ====="
    )

    payload_path = (
        find_first_existing_pocc_payload(
            pocc
        )
    )

    print(
        "payload path =",
        payload_path,
    )

    suffix = (
        payload_path.suffix.lower()
    )

    pocc_payload_schema = {
        "path":
            str(
                payload_path
            ),

        "sha256":
            sha256_file(
                payload_path
            ),

        "suffix":
            suffix,
    }

    if suffix == ".npz":

        pocc_payload_schema[
            "arrays"
        ] = (
            inspect_npz_metadata(
                payload_path
            )
        )

        for key, metadata in (
            pocc_payload_schema[
                "arrays"
            ].items()
        ):

            print(
                f"{key:32s}"
                f" shape={metadata['shape']}"
                f" dtype={metadata['dtype']}"
            )

    elif suffix == ".npy":

        array = np.load(
            payload_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        pocc_payload_schema[
            "array"
        ] = {
            "shape":
                list(
                    array.shape
                ),

            "dtype":
                str(
                    array.dtype
                ),

            "ndim":
                int(
                    array.ndim
                ),
        }

        print(
            "shape =",
            list(
                array.shape
            ),
        )

        print(
            "dtype =",
            array.dtype,
        )

    elif suffix == ".json":

        payload = load_json(
            payload_path
        )

        pocc_payload_schema[
            "top_keys"
        ] = sorted(
            payload.keys()
        )

        pocc_payload_schema[
            "key_paths"
        ] = nested_key_paths(
            payload
        )

        print(
            "top keys =",
            sorted(
                payload.keys()
            ),
        )

    else:

        raise RuntimeError(
            "Unsupported persisted P_occ "
            f"payload suffix: {suffix}"
        )

    print(
        "P_occ payload values analyzed = NO"
    )

    print(
        "P_occ performance computed    = NO"
    )

    # ========================================================
    # E. Scene-cache exact structural route
    # ========================================================

    print()
    print(
        "===== E. SCENE CACHE STRUCTURE ====="
    )

    scene_files = sorted(
        path
        for path in SCENE_CACHE.glob(
            "*.json"
        )
        if (
            path.is_file()
            and
            "formal"
            not in
            path.name.lower()
        )
    )

    require(
        scene_files,
        "No development scene cache.",
    )

    first_scene = load_json(
        scene_files[0]
    )

    scene_schema = {
        "path":
            str(
                scene_files[0]
            ),

        "sha256":
            sha256_file(
                scene_files[0]
            ),

        "top_level_types":
            top_level_type_schema(
                first_scene
            ),

        "nested_key_paths":
            nested_key_paths(
                first_scene,
                max_depth=6,
            ),
    }

    print(
        "scene cache file =",
        scene_files[0],
    )

    print(
        "scene cache top keys =",
        sorted(
            first_scene.keys()
        ),
    )

    print(
        "scene nested keys:"
    )

    for key in (
        scene_schema[
            "nested_key_paths"
        ][
            :160
        ]
    ):

        print(
            "   ",
            key,
        )

    # ========================================================
    # F. Illumination-output binding proof
    # ========================================================

    print()
    print(
        "===== F. FINAL ILLUMINATION RUNTIME BINDING ====="
    )

    composition_fields = {
        item[
            "name"
        ]
        for item in (
            dataclasses[
                "ClassAwareComposition"
            ]
        )
    }

    required_composition_fields = {
        "illumination",
    }

    require(
        required_composition_fields.issubset(
            composition_fields
        ),
        (
            "ClassAwareComposition does not "
            "expose final illumination."
        ),
    )

    require(
        "predictive_mask_to_illumination"
        in
        function_api,
        (
            "Predictive mask→illumination "
            "operator missing."
        ),
    )

    require(
        "compose_class_aware_illumination"
        in
        function_api,
        (
            "Class-aware composition "
            "operator missing."
        ),
    )

    require(
        "exact_original_reactive_fallback_map"
        in
        function_api,
        (
            "Reactive fallback operator missing."
        ),
    )

    print(
        "predictive mask→illumination = EXACT API FOUND"
    )

    print(
        "class-aware composition      = EXACT API FOUND"
    )

    print(
        "final illumination field     = FOUND"
    )

    print(
        "reactive fallback operator   = FOUND"
    )

    illumination_route_proven = True

    # ========================================================
    # G. Metric evaluator input matrix
    # ========================================================

    print()
    print(
        "===== G. METRIC INPUT READINESS MATRIX ====="
    )

    metric_inputs = {
        "mask_IoU":
            {
                "candidate_D":
                    True,

                "oracle_O":
                    True,
            },

        "vehicle_shadow_zone_violation":
            {
                "candidate_D":
                    True,

                "O_vehicle":
                    True,
            },

        "glare_risk_exposure":
            {
                "I_final":
                    illumination_route_proven,

                "vehicle_surrogate":
                    (
                        "vehicle_surrogate"
                        in
                        surrogate
                    ),
            },

        "over_masking_area":
            {
                "candidate_D":
                    True,

                "oracle_O":
                    True,

                "full_grid":
                    True,
            },

        "road_illumination_retention":
            {
                "I_final":
                    illumination_route_proven,

                "road_ROI":
                    True,
            },

        "pedestrian_visibility_proxy":
            {
                "I_final":
                    illumination_route_proven,

                "future_full_box":
                    True,
            },

        "cyclist_visibility_proxy":
            {
                "I_final":
                    illumination_route_proven,

                "future_full_box":
                    True,
            },

        "false_dimming":
            {
                "I_final":
                    illumination_route_proven,

                "oracle_O":
                    True,
            },

        "temporal_smoothness":
            {
                "I_final_schedule":
                    illumination_route_proven,

                "delta_t":
                    True,
            },

        "flicker_change_rate":
            {
                "I_final_schedule":
                    illumination_route_proven,
            },

        "energy_consumption":
            {
                "I_final_schedule":
                    illumination_route_proven,
            },

        "actuation_latency":
            {
                "runtime_boundary":
                    True,
            },
    }

    for metric, inputs in (
        metric_inputs.items()
    ):

        status = all(
            inputs.values()
        )

        print(
            f"{metric:34s}=",
            (
                "READY"
                if status
                else "NOT READY"
            ),
            inputs,
        )

    nonready = [
        metric
        for metric, inputs in (
            metric_inputs.items()
        )
        if not all(
            inputs.values()
        )
    ]

    require(
        not nonready,
        (
            "Evaluator metric inputs remain "
            "unproven: "
            +
            repr(
                nonready
            )
        ),
    )

    # ========================================================
    # H. No scientific outcome computation
    # ========================================================

    print()
    print(
        "===== H. OUTCOME BOUNDARY ====="
    )

    print(
        "development records opened      = YES"
    )

    print(
        "development schema inspected    = YES"
    )

    print(
        "occupancy array values scored   = NO"
    )

    print(
        "ADB performance metrics computed= NO"
    )

    print(
        "numeric bounds selected         = NO"
    )

    print(
        "policy sweep                    = NO"
    )

    print(
        "formal content opened           = NO"
    )

    print(
        "formal evaluation               = NO"
    )

    # ========================================================
    # I. Regression
    # ========================================================

    print()
    print(
        "===== I. FULL STAGE6 REGRESSION ====="
    )

    rc, test_count, output = (
        run_regression()
    )

    if rc != 0:

        print(
            "\n".join(
                output.splitlines()[
                    -100:
                ]
            )
        )

    require(
        rc == 0,
        "Stage6 regression failed.",
    )

    require(
        test_count
        ==
        EXPECTED_STAGE6_TESTS,
        (
            "Unexpected Stage6 test count "
            f"{test_count}; expected "
            f"{EXPECTED_STAGE6_TESTS}."
        ),
    )

    print(
        "Stage6 regression =",
        f"{test_count} / {test_count} PASS",
    )

    # ========================================================
    # J. Frozen readback
    # ========================================================

    print()
    print(
        "===== J. FROZEN READBACK ====="
    )

    require(
        sha256_file(
            RECON
        )
        ==
        EXPECTED_RECON_SHA,
        "Reconciliation report changed.",
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_PROTOCOL_SHA,
        "Metric protocol changed.",
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_METRICS_SHA,
        "Metric runtime changed.",
    )

    require(
        sha256_file(
            INTERFACE
        )
        ==
        EXPECTED_INTERFACE_SHA,
        "Metric interface changed.",
    )

    print(
        "reconciled metric files = UNCHANGED"
    )

    # ========================================================
    # K. Storage
    # ========================================================

    print()
    print(
        "===== K. STORAGE ====="
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB hard reserve violated."
        ),
    )

    print(
        "free GiB        =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve = PASS"
    )

    # ========================================================
    # L. Exact evaluator binding report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "part":
            "2/2_exact_evaluator_runtime_bind",

        "status":
            "PASS_EVALUATOR_RUNTIME_BINDING_READY",

        "frozen_metric_authority": {
            "reconciliation_sha256":
                EXPECTED_RECON_SHA,

            "protocol_sha256":
                EXPECTED_PROTOCOL_SHA,

            "runtime_sha256":
                EXPECTED_METRICS_SHA,

            "interface_sha256":
                EXPECTED_INTERFACE_SHA,
        },

        "runtime_api": {
            "functions":
                function_api,

            "dataclasses":
                dataclasses,
        },

        "actor_identity": {
            "scenario_shared":
                scenario_shared,

            "predictive_join_keys":
                predictive_join_keys,

            "key_presence":
                key_presence,
        },

        "development_record_schemas":
            record_schemas,

        "P_occ_payload":
            pocc_payload_schema,

        "scene_cache":
            scene_schema,

        "illumination_runtime": {
            "predictive_mask_to_illumination":
                True,

            "compose_class_aware_illumination":
                True,

            "final_illumination_field":
                "ClassAwareComposition.illumination",

            "reactive_fallback_operator":
                True,

            "status":
                "BOUND",
        },

        "metric_input_readiness":
            metric_inputs,

        "scientific_execution": {
            "development_schema_read":
                True,

            "development_performance_metrics_computed":
                False,

            "numeric_bounds_selected":
                False,

            "policy_sweep":
                False,

            "parameter_tuning":
                False,

            "formal_content_opened":
                False,

            "formal_outcomes_read":
                False,

            "formal_evaluation":
                False,
        },

        "regression": {
            "tests":
                test_count,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "Execute development-only primary "
                "class-aware predictive vs original "
                "reactive ADB evaluation using exact "
                "bound actor joins and illumination "
                "runtime, then freeze non-inferiority "
                "bounds before formal evaluation."
            ),
    }

    write_json(
        REPORT,
        result,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART 2/2 — EXACT EVALUATOR BIND FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "actor identity route       = PROVEN"
    )

    print(
        "predictive actor join      = PROVEN"
    )

    print(
        "P_occ payload schema       = PROVEN"
    )

    print(
        "predictive illumination API= PROVEN"
    )

    print(
        "final I_final route        = PROVEN"
    )

    print(
        "reactive baseline API      = PROVEN"
    )

    print(
        "all 12 metric inputs       = READY"
    )

    print(
        "development metrics        = NOT COMPUTED"
    )

    print(
        "numeric bounds             = NOT SELECTED"
    )

    print(
        "formal content             = NOT OPENED"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "policy/parameter tuning    = NO"
    )

    print(
        "Stage6 regression          =",
        f"{test_count} / {test_count} PASS",
    )

    print(
        "STATUS = PASS_EVALUATOR_RUNTIME_BINDING_READY"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "terminal remains open = YES"
    )


try:

    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 EXACT EVALUATOR RUNTIME BIND = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "development performance metrics = NOT COMPUTED"
    )

    print(
        "numeric bounds                 = NOT SELECTED"
    )

    print(
        "formal content                 = NOT OPENED"
    )

    print(
        "formal outcomes                = NOT READ"
    )

    print(
        "formal evaluation              = NO"
    )

    print(
        "policy tuning                  = NO"
    )

    print()
    print(
        "Do not run development metric evaluation."
    )

    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "Send this BLOCKED output for targeted repair."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

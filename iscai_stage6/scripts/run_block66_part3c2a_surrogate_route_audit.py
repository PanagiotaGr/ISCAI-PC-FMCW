from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PART3B_PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

PART3C1_REPORT = (
    S6
    / "reports/"
      "block66_part3c1_class_aware_runtime.json"
)

PART3C1_HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3c1_to_part3c2_handoff.json"
)

CLASS_AWARE_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

POCC_MANIFEST = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

POCC_CACHE = (
    S6
    / "artifacts/block66/"
      "part2c2b_actor_pocc_cache"
)

ELIGIBLE_MATCHES = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block66_part3c2a_vehicle_surrogate_route_binding_audit.json"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3c2a_to_part3c2b_handoff.json"
)


EXPECTED = {
    "part3b_prereg":
        (
            "5704494a89b4b3c7a3c19b0df8176c55"
            "0c00795ed17cf2906c343f340461429e"
        ),

    "part3c1_report":
        (
            "ef4c18367dfe26ceac3a7f55f725f545"
            "0c7a3a0fd28811aa565472d9b06073a2"
        ),

    "part3c1_handoff":
        (
            "5dccff7f48403bffcb3eab2f787bb09f"
            "96a4cce41a5e8c286050026fea58f813"
        ),

    "class_aware_module":
        (
            "b998f468b98c2770c48c84a2c0aaaa21"
            "77c2d84b8fc838e80fef9b9da1ebc14b"
        ),

    "pocc_manifest":
        (
            "c361ed640b91850831d7e7177d89733f"
            "edd3e171f5f3111bcd93698c74a4a300"
        ),

    "eligible_matches":
        (
            "d7ce8769deee08286974b0327cdfd55c0"
            "8ad7ea326cf9d6450a48a82b3e251b6"
        ),
}


class ControlledBlock(RuntimeError):
    pass


def require(condition, message):
    if not bool(condition):
        raise ControlledBlock(str(message))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def exact_seal(path: Path, expected: str, label: str):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = sha256_file(path)

    require(
        actual == expected,
        f"{label} SHA changed: {actual}",
    )

    print(
        f"{label:35s} = EXACT PASS"
    )

    return {
        "path":
            str(path),

        "sha256":
            actual,
    }


def read_jsonl(path: Path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)

            except BaseException as exc:
                raise ControlledBlock(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                )

            require(
                isinstance(value, dict),
                (
                    f"JSONL record at {path}:{line_number} "
                    "is not an object."
                ),
            )

            records.append(value)

    return records


def summarize_value(value):
    if isinstance(value, dict):
        return {
            "type":
                "dict",

            "keys":
                sorted(value.keys()),
        }

    if isinstance(value, list):
        return {
            "type":
                "list",

            "length":
                len(value),

            "first_type":
                (
                    type(value[0]).__name__
                    if value
                    else None
                ),
        }

    return {
        "type":
            type(value).__name__,

        "repr":
            repr(value)[:240],
    }


def recursive_string_candidates(value, prefix=""):
    """
    Read-only extraction of path-like strings from manifest records.
    No guessed field names are required.
    """

    output = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            output.extend(
                recursive_string_candidates(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(
                recursive_string_candidates(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    elif isinstance(value, str):
        lower = value.lower()

        if (
            "/" in value
            or
            lower.endswith(
                (
                    ".npz",
                    ".npy",
                    ".json",
                    ".jsonl",
                )
            )
        ):
            output.append(
                {
                    "field":
                        prefix,

                    "value":
                        value,
                }
            )

    return output


def resolve_path_string(value: str):
    raw = Path(value)

    candidates = []

    if raw.is_absolute():
        candidates.append(raw)

    else:
        candidates.extend(
            [
                S6 / raw,
                ROOT / raw,
                POCC_CACHE / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def inspect_numpy_file(path: Path):
    if path.suffix.lower() == ".npz":
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            return {
                "kind":
                    "npz",

                "keys":
                    list(
                        data.files
                    ),

                "arrays": {
                    key: {
                        "shape":
                            list(
                                np.asarray(
                                    data[key]
                                ).shape
                            ),

                        "dtype":
                            str(
                                np.asarray(
                                    data[key]
                                ).dtype
                            ),
                    }
                    for key in data.files
                },
            }

    if path.suffix.lower() == ".npy":
        array = np.load(
            path,
            allow_pickle=False,
            mmap_mode="r",
        )

        return {
            "kind":
                "npy",

            "shape":
                list(array.shape),

            "dtype":
                str(array.dtype),
        }

    return None


def canonical_write(path: Path, payload):
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode("utf-8")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes() == data,
            (
                "Existing Part3C/2A report differs "
                "from deterministic rerun."
            ),
        )

    else:
        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_bytes(data)
        temporary.replace(path)

    return sha256_file(path)


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S6 / "tests"),
            "-p",
            "test_*.py",
        ],
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    return {
        "returncode":
            int(process.returncode),

        "tests":
            (
                int(match.group(1))
                if match
                else None
            ),

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[-35:]
            ),
    }


try:
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3C/2A — "
        "VEHICLE SURROGATE FROZEN-EVIDENCE ROUTE BINDING AUDIT"
    )
    print("=" * 78)

    # ========================================================
    # A. Immutable upstream seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE UPSTREAM SEAL ====="
    )

    upstream = {}

    upstream[
        "Part3B_prereg"
    ] = exact_seal(
        PART3B_PREREG,
        EXPECTED["part3b_prereg"],
        "Part3B preregistration",
    )

    upstream[
        "Part3C1_report"
    ] = exact_seal(
        PART3C1_REPORT,
        EXPECTED["part3c1_report"],
        "Part3C/1 report",
    )

    upstream[
        "Part3C1_handoff"
    ] = exact_seal(
        PART3C1_HANDOFF,
        EXPECTED["part3c1_handoff"],
        "Part3C/1 handoff",
    )

    upstream[
        "class_aware_module"
    ] = exact_seal(
        CLASS_AWARE_MODULE,
        EXPECTED["class_aware_module"],
        "class_aware_policy.py",
    )

    upstream[
        "P_occ_manifest"
    ] = exact_seal(
        POCC_MANIFEST,
        EXPECTED["pocc_manifest"],
        "P_occ manifest",
    )

    upstream[
        "eligible_matches"
    ] = exact_seal(
        ELIGIBLE_MATCHES,
        EXPECTED["eligible_matches"],
        "eligible predictive matches",
    )

    require(
        POCC_CACHE.is_dir(),
        f"Missing P_occ cache directory: {POCC_CACHE}",
    )

    print(
        "P_occ cache directory              = PRESENT"
    )

    # ========================================================
    # B. Exact runtime geometry API
    # ========================================================

    print()
    print(
        "===== B. EXACT SURROGATE GEOMETRY API ====="
    )

    from iscai_stage6.adb.geometry import (
        Box3D,
        FractionalBoxRegion,
        project_region_to_headlamp,
    )

    from iscai_stage6.adb.probabilistic_full_box import (
        CalibratedGaussianFullBoxPrediction,
        StochasticFullBoxForecast,
        build_stochastic_future_full_boxes,
    )

    print(
        "Box3D =",
        inspect.signature(Box3D),
    )

    print(
        "FractionalBoxRegion =",
        inspect.signature(FractionalBoxRegion),
    )

    print(
        "project_region_to_headlamp =",
        inspect.signature(project_region_to_headlamp),
    )

    print(
        "CalibratedGaussianFullBoxPrediction =",
        inspect.signature(
            CalibratedGaussianFullBoxPrediction
        ),
    )

    print(
        "StochasticFullBoxForecast =",
        inspect.signature(
            StochasticFullBoxForecast
        ),
    )

    print(
        "build_stochastic_future_full_boxes =",
        inspect.signature(
            build_stochastic_future_full_boxes
        ),
    )

    oncoming = FractionalBoxRegion(
        x_bounds=(0.0, 0.5),
        y_bounds=(-0.5, 0.5),
        z_bounds=(0.0, 0.5),
    )

    preceding = FractionalBoxRegion(
        x_bounds=(-0.5, 0.0),
        y_bounds=(-0.5, 0.5),
        z_bounds=(0.0, 0.5),
    )

    print(
        "oncoming front-upper surrogate = VALID"
    )

    print(
        "preceding rear-upper surrogate = VALID"
    )

    print(
        "actual windshield/mirror annotation claim = NO"
    )

    # ========================================================
    # C. Manifest schema
    # ========================================================

    print()
    print(
        "===== C. P_OCC MANIFEST SCHEMA ====="
    )

    manifest_records = read_jsonl(
        POCC_MANIFEST
    )

    require(
        len(manifest_records) == 877,
        (
            "Expected exactly 877 P_occ manifest records; "
            f"got {len(manifest_records)}."
        ),
    )

    print(
        "manifest records =",
        len(manifest_records),
    )

    manifest_key_sets = {}

    for record in manifest_records:
        key_tuple = tuple(
            sorted(record.keys())
        )

        manifest_key_sets[
            key_tuple
        ] = (
            manifest_key_sets.get(
                key_tuple,
                0,
            )
            +
            1
        )

    print(
        "distinct top-level schemas =",
        len(manifest_key_sets),
    )

    for index, (
        keys,
        count,
    ) in enumerate(
        sorted(
            manifest_key_sets.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    ):
        print()
        print(
            f"schema[{index}] count =",
            count,
        )

        print(
            "keys =",
            keys,
        )

    print()
    print(
        "first manifest record field summaries:"
    )

    first_manifest = manifest_records[0]

    for key in sorted(
        first_manifest
    ):
        print(
            key,
            "=",
            summarize_value(
                first_manifest[key]
            ),
        )

    # ========================================================
    # D. Eligible-match schema
    # ========================================================

    print()
    print(
        "===== D. ELIGIBLE PREDICTIVE MATCH SCHEMA ====="
    )

    match_records = read_jsonl(
        ELIGIBLE_MATCHES
    )

    require(
        len(match_records) == 877,
        (
            "Expected exactly 877 eligible predictive matches; "
            f"got {len(match_records)}."
        ),
    )

    print(
        "eligible predictive match records =",
        len(match_records),
    )

    match_key_sets = {}

    for record in match_records:
        key_tuple = tuple(
            sorted(record.keys())
        )

        match_key_sets[
            key_tuple
        ] = (
            match_key_sets.get(
                key_tuple,
                0,
            )
            +
            1
        )

    print(
        "distinct match schemas =",
        len(match_key_sets),
    )

    for index, (
        keys,
        count,
    ) in enumerate(
        sorted(
            match_key_sets.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    ):
        print()
        print(
            f"match schema[{index}] count =",
            count,
        )

        print(
            "keys =",
            keys,
        )

    print()
    print(
        "first eligible-match record field summaries:"
    )

    first_match = match_records[0]

    for key in sorted(
        first_match
    ):
        print(
            key,
            "=",
            summarize_value(
                first_match[key]
            ),
        )

    # ========================================================
    # E. Path binding audit
    # ========================================================

    print()
    print(
        "===== E. MANIFEST PATH / CACHE BINDING ====="
    )

    path_candidates = []

    for record_index, record in enumerate(
        manifest_records[:20]
    ):
        for item in recursive_string_candidates(
            record
        ):
            resolved = resolve_path_string(
                item["value"]
            )

            path_candidates.append(
                {
                    "record_index":
                        record_index,

                    "field":
                        item["field"],

                    "value":
                        item["value"],

                    "resolved":
                        (
                            str(resolved)
                            if resolved is not None
                            else None
                        ),
                }
            )

    resolvable = [
        item
        for item in path_candidates
        if item["resolved"] is not None
    ]

    print(
        "path-like fields inspected =",
        len(path_candidates),
    )

    print(
        "resolvable existing paths =",
        len(resolvable),
    )

    for item in resolvable[:20]:
        print(
            json.dumps(
                item,
                sort_keys=True,
            )
        )

    # Independently inventory the cache directory without
    # assuming manifest field names.
    cache_files = sorted(
        path
        for path in POCC_CACHE.rglob("*")
        if path.is_file()
    )

    print()
    print(
        "cache file count =",
        len(cache_files),
    )

    suffix_counts = {}

    for path in cache_files:
        suffix = path.suffix.lower()

        suffix_counts[
            suffix
        ] = (
            suffix_counts.get(
                suffix,
                0,
            )
            +
            1
        )

    print(
        "cache suffix counts =",
        suffix_counts,
    )

    require(
        cache_files,
        "P_occ cache directory is empty.",
    )

    # ========================================================
    # F. Read-only sample cache inspection
    # ========================================================

    print()
    print(
        "===== F. READ-ONLY SAMPLE CACHE INSPECTION ====="
    )

    numpy_files = [
        path
        for path in cache_files
        if path.suffix.lower()
        in (
            ".npz",
            ".npy",
        )
    ]

    print(
        "numpy cache files =",
        len(numpy_files),
    )

    inspected_numpy = []

    for path in numpy_files[:3]:
        info = inspect_numpy_file(
            path
        )

        inspected_numpy.append(
            {
                "path":
                    str(path),

                "sha256":
                    sha256_file(path),

                "info":
                    info,
            }
        )

        print()
        print(
            "cache sample =",
            path,
        )

        print(
            json.dumps(
                info,
                sort_keys=True,
                indent=2,
            )
        )

    # Also show first three non-numpy small files, if any.
    other_samples = []

    for path in cache_files:
        if path in numpy_files:
            continue

        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
        except OSError:
            continue

        other_samples.append(path)

        if len(other_samples) >= 3:
            break

    for path in other_samples:
        print()
        print(
            "non-numpy cache sample =",
            path,
        )

        print(
            "size bytes =",
            path.stat().st_size,
        )

        try:
            text = path.read_text(
                encoding="utf-8"
            )

            print(
                "first 1200 chars ="
            )

            print(
                text[:1200]
            )

        except BaseException:
            print(
                "text decode = NOT AVAILABLE"
            )

    # ========================================================
    # G. Join-key evidence
    # ========================================================

    print()
    print(
        "===== G. JOIN-KEY EVIDENCE ====="
    )

    manifest_common_keys = set(
        manifest_records[0].keys()
    )

    for record in manifest_records[1:]:
        manifest_common_keys.intersection_update(
            record.keys()
        )

    match_common_keys = set(
        match_records[0].keys()
    )

    for record in match_records[1:]:
        match_common_keys.intersection_update(
            record.keys()
        )

    common_between = sorted(
        manifest_common_keys
        &
        match_common_keys
    )

    print(
        "manifest common keys =",
        sorted(
            manifest_common_keys
        ),
    )

    print(
        "eligible-match common keys =",
        sorted(
            match_common_keys
        ),
    )

    print(
        "shared top-level candidate join keys =",
        common_between,
    )

    # No guessed join is executed here.
    require(
        common_between,
        (
            "No top-level shared candidate join key found; "
            "Part3C/2B would require explicit schema resolution."
        ),
    )

    # ========================================================
    # H. Fresh Stage6 regression
    # ========================================================

    print()
    print(
        "===== H. FRESH STAGE6 REGRESSION ====="
    )

    regression = run_regression()

    print(
        regression["tail"]
    )

    require(
        regression["returncode"] == 0,
        "Stage6 regression failed.",
    )

    require(
        regression["tests"] == 166,
        (
            "Expected 166 Stage6 tests; "
            f"got {regression['tests']}."
        ),
    )

    print(
        "Stage6 regression = PASS | 166"
    )

    # ========================================================
    # I. Report
    # ========================================================

    print()
    print(
        "===== I. PART3C/2A ROUTE-AUDIT REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part3C/2A",

        "status":
            (
                "PASS_FROZEN_VEHICLE_SURROGATE_"
                "EVIDENCE_ROUTE_AUDITED"
            ),

        "upstream":
            upstream,

        "runtime_geometry_API": {
            "Box3D":
                str(
                    inspect.signature(
                        Box3D
                    )
                ),

            "FractionalBoxRegion":
                str(
                    inspect.signature(
                        FractionalBoxRegion
                    )
                ),

            "project_region_to_headlamp":
                str(
                    inspect.signature(
                        project_region_to_headlamp
                    )
                ),

            "CalibratedGaussianFullBoxPrediction":
                str(
                    inspect.signature(
                        CalibratedGaussianFullBoxPrediction
                    )
                ),

            "StochasticFullBoxForecast":
                str(
                    inspect.signature(
                        StochasticFullBoxForecast
                    )
                ),

            "build_stochastic_future_full_boxes":
                str(
                    inspect.signature(
                        build_stochastic_future_full_boxes
                    )
                ),
        },

        "surrogate_semantics": {
            "oncoming_front_upper":
                {
                    "x_bounds":
                        [0.0, 0.5],

                    "y_bounds":
                        [-0.5, 0.5],

                    "z_bounds":
                        [0.0, 0.5],
                },

            "preceding_rear_upper":
                {
                    "x_bounds":
                        [-0.5, 0.0],

                    "y_bounds":
                        [-0.5, 0.5],

                    "z_bounds":
                        [0.0, 0.5],
                },

            "physical_annotation_claim":
                False,

            "vertical_actuator_claim":
                False,
        },

        "manifest": {
            "record_count":
                len(
                    manifest_records
                ),

            "top_level_schema_counts":
                [
                    {
                        "keys":
                            list(keys),

                        "count":
                            count,
                    }
                    for keys, count
                    in sorted(
                        manifest_key_sets.items(),
                        key=lambda item: (
                            -item[1],
                            item[0],
                        ),
                    )
                ],

            "first_record_summary":
                {
                    key:
                        summarize_value(
                            first_manifest[key]
                        )
                    for key in
                    sorted(
                        first_manifest
                    )
                },
        },

        "eligible_matches": {
            "record_count":
                len(
                    match_records
                ),

            "top_level_schema_counts":
                [
                    {
                        "keys":
                            list(keys),

                        "count":
                            count,
                    }
                    for keys, count
                    in sorted(
                        match_key_sets.items(),
                        key=lambda item: (
                            -item[1],
                            item[0],
                        ),
                    )
                ],

            "first_record_summary":
                {
                    key:
                        summarize_value(
                            first_match[key]
                        )
                    for key in
                    sorted(
                        first_match
                    )
                },
        },

        "cache": {
            "root":
                str(
                    POCC_CACHE
                ),

            "file_count":
                len(
                    cache_files
                ),

            "suffix_counts":
                suffix_counts,

            "path_like_fields_inspected":
                len(
                    path_candidates
                ),

            "resolvable_manifest_paths":
                resolvable[:50],

            "numpy_samples":
                inspected_numpy,
        },

        "candidate_join_keys":
            common_between,

        "scientific_execution": {
            "new_sampling":
                False,

            "new_model_forward":
                False,

            "policy_sweep":
                False,

            "numeric_policy_selection":
                False,

            "formal_outcomes_read":
                False,

            "source_modified":
                False,

            "cache_modified":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                166,
        },

        "next":
            (
                "BLOCK6.6_PART3C2B_BIND_EXACT_CACHE_"
                "SCHEMA_AND_MATERIALIZE_GEOMETRY_BASED_"
                "VEHICLE_SURROGATE_EVIDENCE_WITHOUT_"
                "RESAMPLING"
            ),
    }

    report_sha = canonical_write(
        REPORT,
        report_payload,
    )

    handoff_payload = {
        "stage":
            6,

        "from":
            "6.6-Part3C/2A",

        "to":
            "6.6-Part3C/2B",

        "status":
            (
                "PASS_READY_FOR_EXACT_CACHE_BOUND_"
                "VEHICLE_SURROGATE_MATERIALIZATION"
            ),

        "route_audit_report": {
            "path":
                str(
                    REPORT
                ),

            "sha256":
                report_sha,
        },

        "candidate_join_keys":
            common_between,

        "cache_root":
            str(
                POCC_CACHE
            ),

        "new_sampling_allowed":
            False,

        "formal_outcomes_allowed":
            False,

        "numeric_policy_selection_allowed":
            False,
    }

    handoff_sha = canonical_write(
        HANDOFF,
        handoff_payload,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # J. Final
    # ========================================================

    print()
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3C/2A — FINAL"
    )
    print("=" * 78)

    print(
        "STATUS = "
        "PASS_FROZEN_VEHICLE_SURROGATE_EVIDENCE_ROUTE_AUDITED"
    )

    print(
        "P_occ manifest records      =",
        len(
            manifest_records
        ),
    )

    print(
        "eligible predictive matches =",
        len(
            match_records
        ),
    )

    print(
        "cache files                 =",
        len(
            cache_files
        ),
    )

    print(
        "candidate join keys         =",
        common_between,
    )

    print(
        "geometry surrogate API      = BOUND"
    )

    print(
        "new sampling                = NO"
    )

    print(
        "new model forward           = NO"
    )

    print(
        "policy sweep                = NO"
    )

    print(
        "numeric policy selected     = NO"
    )

    print(
        "formal outcomes read        = NO"
    )

    print(
        "source modified             = NO"
    )

    print(
        "cache modified              = NO"
    )

    print(
        "Stage6 regression           = PASS | 166"
    )

    print(
        "report SHA256               =",
        report_sha,
    )

    print(
        "handoff SHA256              =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.6 PART3C/2B "
        "EXACT-CACHE VEHICLE SURROGATE MATERIALIZATION"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 78)


except BaseException as exc:
    print()
    print("=" * 78)
    print(
        "BLOCK6.6 PART3C/2A — CONTROLLED BLOCK"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=25
    )

    print()
    print(
        "new sampling             = NO"
    )

    print(
        "new model forward        = NO"
    )

    print(
        "policy sweep             = NO"
    )

    print(
        "numeric policy selected  = NO"
    )

    print(
        "formal outcomes read     = NO"
    )

    print(
        "source modified          = NO"
    )

    print(
        "cache modified           = NO"
    )

    print(
        "terminal remains open    = YES"
    )

# No sys.exit().

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PART3C2A_REPORT = (
    S6 / "reports/"
    "block66_part3c2a_vehicle_surrogate_route_binding_audit.json"
)

PART3C2A_HANDOFF = (
    S6 / "artifacts/block66/"
    "block66_part3c2a_to_part3c2b_handoff.json"
)

PART3C1_REPORT = (
    S6 / "reports/"
    "block66_part3c1_class_aware_runtime.json"
)

CLASS_AWARE_MODULE = (
    S6 / "src/iscai_stage6/adb/"
    "class_aware_policy.py"
)

POCC_MANIFEST = (
    S6 / "artifacts/block66/"
    "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

ELIGIBLE_MATCHES = (
    S6 / "artifacts/block66/"
    "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

POCC_CACHE = (
    S6 / "artifacts/block66/"
    "part2c2b_actor_pocc_cache"
)

PART2C2B_REPORT = (
    S6 / "reports/"
    "block66_part2c2b_eligible_n8192_pocc.json"
)

PART2B_REPORT = (
    S6 / "reports/"
    "block66_part2b_cohort_posterior_association_census.json"
)

PERSISTENCE_CONTRACT = (
    S6 / "configs/"
    "block66_part2c2b_pocc_persistence_contract.json"
)

PART2C2B_RUNNER = (
    S6 / "scripts/"
    "run_block66_part2c2b_resumable_pocc.py"
)

REPORT = (
    S6 / "reports/"
    "block66_part3c2b0_persistence_sufficiency_audit.json"
)

HANDOFF = (
    S6 / "artifacts/block66/"
    "block66_part3c2b0_to_part3c2b1_handoff.json"
)


EXPECTED = {
    "part3c2a_report":
        "e2372c70d353f7122a91e514853ac1e823245e5ca1cfbf6d9eac5673ad550026",

    "part3c2a_handoff":
        "620d8d13b4f0ee04fdde627a698efe24c42d43b362c19a227eea4d7e0e35cc5e",

    "part3c1_report":
        "ef4c18367dfe26ceac3a7f55f725f5450c7a3a0fd28811aa565472d9b06073a2",

    "class_aware_module":
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    "pocc_manifest":
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    "eligible_matches":
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    "part2c2b_report":
        "41257fbc43bcd6e0a01925c0f96648f3cb67850785e8f39d9787547ce1511348",

    "part2b_report":
        "96dfccd808fd06d77b6f9d4b1c100f6a59db7d090a405f41b88d4fa6854c9705",

    "persistence_contract":
        "2777e25003a2c76d506609130e7927501f15a868fa6ec37dfd192bbb58ea79e5",

    "posterior":
        "f022471fbf86c3121106011bfd7f035c5d895f2c5aeeaa2fa3a20c04b3439acd",
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
        "path": str(path),
        "sha256": actual,
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
                record = json.loads(stripped)

            except BaseException as exc:
                raise ControlledBlock(
                    f"Invalid JSONL {path}:{line_number}: {exc}"
                )

            require(
                isinstance(record, dict),
                f"Non-object JSONL record at {path}:{line_number}",
            )

            records.append(record)

    return records


def key_paths(value, prefix=""):
    result = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            result.append(path)
            result.extend(
                key_paths(
                    child,
                    path,
                )
            )

    elif isinstance(value, list):
        # Do not expand full numerical arrays.
        for index, child in enumerate(
            value[:3]
        ):
            result.extend(
                key_paths(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    return result


def recursive_strings(value):
    result = []

    if isinstance(value, dict):
        for child in value.values():
            result.extend(
                recursive_strings(child)
            )

    elif isinstance(value, list):
        for child in value:
            result.extend(
                recursive_strings(child)
            )

    elif isinstance(value, str):
        result.append(value)

    return result


def locate_posterior():
    """
    First use path evidence from the exact frozen Part2B report.
    Fall back only to JSONL files under block66.
    """

    report_data = json.loads(
        PART2B_REPORT.read_text(
            encoding="utf-8"
        )
    )

    candidate_paths = []

    for value in recursive_strings(
        report_data
    ):
        if not (
            "/" in value
            or value.lower().endswith(".jsonl")
        ):
            continue

        raw = Path(value)

        candidates = (
            [raw]
            if raw.is_absolute()
            else [
                S6 / raw,
                ROOT / raw,
            ]
        )

        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate not in candidate_paths
            ):
                candidate_paths.append(candidate)

    for path in candidate_paths:
        try:
            if (
                sha256_file(path)
                ==
                EXPECTED["posterior"]
            ):
                return path
        except OSError:
            pass

    fallback = []

    block66 = (
        S6 / "artifacts/block66"
    )

    for path in block66.rglob("*.jsonl"):
        if (
            "cache"
            in str(path).lower()
        ):
            continue

        try:
            if (
                sha256_file(path)
                ==
                EXPECTED["posterior"]
            ):
                fallback.append(path)

        except OSError:
            continue

    require(
        len(fallback) == 1,
        (
            "Could not uniquely locate frozen posterior "
            f"SHA {EXPECTED['posterior']}; matches={fallback}"
        ),
    )

    return fallback[0]


def ast_relevant_calls(path: Path):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    wanted = {
        "build_stochastic_future_full_boxes",
        "estimate_actor_occupancy_probability",
        "save",
        "savez",
        "write_text",
        "dump",
    }

    calls = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = None

        if isinstance(
            node.func,
            ast.Name,
        ):
            name = node.func.id

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            name = node.func.attr

        if name not in wanted:
            continue

        segment = ast.get_source_segment(
            source,
            node,
        )

        calls.append({
            "line":
                int(node.lineno),

            "call":
                name,

            "source":
                (
                    segment[:1000]
                    if segment
                    else None
                ),
        })

    return source, calls


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
                "Existing Part3C/2B0 artifact differs "
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
        "BLOCK 6.6 PART3C/2B0 — "
        "SURROGATE PERSISTENCE SUFFICIENCY AUDIT"
    )
    print("=" * 78)

    # ========================================================
    # A. Immutable seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE UPSTREAM SEAL ====="
    )

    upstream = {}

    for key, path, label in (
        (
            "Part3C2A_report",
            PART3C2A_REPORT,
            "Part3C/2A report",
        ),
        (
            "Part3C2A_handoff",
            PART3C2A_HANDOFF,
            "Part3C/2A handoff",
        ),
        (
            "Part3C1_report",
            PART3C1_REPORT,
            "Part3C/1 report",
        ),
        (
            "class_aware_module",
            CLASS_AWARE_MODULE,
            "class_aware_policy.py",
        ),
        (
            "P_occ_manifest",
            POCC_MANIFEST,
            "P_occ manifest",
        ),
        (
            "eligible_matches",
            ELIGIBLE_MATCHES,
            "eligible predictive matches",
        ),
        (
            "Part2C2B_report",
            PART2C2B_REPORT,
            "Part2C/2B P_occ report",
        ),
        (
            "Part2B_report",
            PART2B_REPORT,
            "Part2B posterior census",
        ),
        (
            "persistence_contract",
            PERSISTENCE_CONTRACT,
            "P_occ persistence contract",
        ),
    ):
        expected_key = {
            "Part3C2A_report": "part3c2a_report",
            "Part3C2A_handoff": "part3c2a_handoff",
            "Part3C1_report": "part3c1_report",
            "class_aware_module": "class_aware_module",
            "P_occ_manifest": "pocc_manifest",
            "eligible_matches": "eligible_matches",
            "Part2C2B_report": "part2c2b_report",
            "Part2B_report": "part2b_report",
            "persistence_contract": "persistence_contract",
        }[key]

        upstream[key] = exact_seal(
            path,
            EXPECTED[expected_key],
            label,
        )

    require(
        PART2C2B_RUNNER.is_file(),
        (
            "Missing exact Part2C/2B runner: "
            f"{PART2C2B_RUNNER}"
        ),
    )

    runner_sha = sha256_file(
        PART2C2B_RUNNER
    )

    print(
        "Part2C/2B runner current SHA256 =",
        runner_sha,
    )

    # ========================================================
    # B. Exhaustive cache persistence inventory
    # ========================================================

    print()
    print(
        "===== B. EXHAUSTIVE P_OCC CACHE INVENTORY ====="
    )

    require(
        POCC_CACHE.is_dir(),
        f"Missing cache: {POCC_CACHE}",
    )

    cache_files = sorted(
        path
        for path in POCC_CACHE.rglob("*")
        if path.is_file()
    )

    npy_files = [
        path
        for path in cache_files
        if path.suffix.lower() == ".npy"
    ]

    json_files = [
        path
        for path in cache_files
        if path.suffix.lower() == ".json"
    ]

    other_files = [
        path
        for path in cache_files
        if path.suffix.lower()
        not in {
            ".npy",
            ".json",
        }
    ]

    print(
        "total files =",
        len(cache_files),
    )

    print(
        ".npy files  =",
        len(npy_files),
    )

    print(
        ".json files =",
        len(json_files),
    )

    print(
        "other files =",
        len(other_files),
    )

    require(
        len(cache_files) == 1754,
        (
            "Expected exact frozen cache size "
            f"1754; got {len(cache_files)}."
        ),
    )

    require(
        len(npy_files) == 877,
        (
            "Expected 877 .npy files; "
            f"got {len(npy_files)}."
        ),
    )

    require(
        len(json_files) == 877,
        (
            "Expected 877 .json files; "
            f"got {len(json_files)}."
        ),
    )

    require(
        not other_files,
        (
            "Unexpected additional cache files exist: "
            f"{other_files[:20]}"
        ),
    )

    # ========================================================
    # C. Metadata proof: are samplewise arrays persisted?
    # ========================================================

    print()
    print(
        "===== C. SAMPLEWISE-PERSISTENCE METADATA PROOF ====="
    )

    metadata = []

    all_key_paths = set()

    maximum_json_bytes = 0

    for path in json_files:
        maximum_json_bytes = max(
            maximum_json_bytes,
            path.stat().st_size,
        )

        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        metadata.append(value)

        all_key_paths.update(
            key_paths(value)
        )

    suspicious_terms = (
        "trajectory_samples",
        "heading_samples",
        "sample_positions",
        "samplewise_positions",
        "forecast_samples",
        "future_boxes",
        "stochastic_future",
    )

    samplewise_key_paths = sorted(
        path
        for path in all_key_paths
        if any(
            term in path.lower()
            for term in suspicious_terms
        )
    )

    print(
        "distinct metadata key paths =",
        len(all_key_paths),
    )

    print(
        "maximum JSON size bytes =",
        maximum_json_bytes,
    )

    print(
        "samplewise trajectory/heading/box key paths =",
        samplewise_key_paths,
    )

    print()
    print(
        "metadata top-level keys =",
        sorted(
            metadata[0].keys()
        ),
    )

    samplewise_persisted_in_cache = bool(
        other_files
        or
        samplewise_key_paths
    )

    print(
        "samplewise forecast persisted in P_occ cache =",
        samplewise_persisted_in_cache,
    )

    # ========================================================
    # D. Exact Part2C2B persistence source audit
    # ========================================================

    print()
    print(
        "===== D. PART2C/2B RUNNER PERSISTENCE AUDIT ====="
    )

    runner_source, relevant_calls = (
        ast_relevant_calls(
            PART2C2B_RUNNER
        )
    )

    for item in relevant_calls:
        print()
        print(
            "line",
            item["line"],
            "|",
            item["call"],
        )

        print(
            item["source"]
        )

    source_markers = {
        "build_stochastic_future_full_boxes":
            (
                "build_stochastic_future_full_boxes"
                in runner_source
            ),

        "estimate_actor_occupancy_probability":
            (
                "estimate_actor_occupancy_probability"
                in runner_source
            ),

        "trajectory_samples_H0_m":
            (
                "trajectory_samples_H0_m"
                in runner_source
            ),

        "heading_samples_rad":
            (
                "heading_samples_rad"
                in runner_source
            ),

        "occupancy_counts":
            (
                "occupancy_counts"
                in runner_source
            ),
    }

    print()
    print(
        "runner source markers =",
        json.dumps(
            source_markers,
            sort_keys=True,
            indent=2,
        ),
    )

    # ========================================================
    # E. Persistence contract
    # ========================================================

    print()
    print(
        "===== E. PERSISTENCE CONTRACT ====="
    )

    persistence = json.loads(
        PERSISTENCE_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    print(
        json.dumps(
            persistence,
            sort_keys=True,
            indent=2,
        )[:12000]
    )

    # ========================================================
    # F. Frozen posterior route
    # ========================================================

    print()
    print(
        "===== F. FROZEN POSTERIOR REPLAY INPUT ROUTE ====="
    )

    posterior_path = locate_posterior()

    posterior_sha = sha256_file(
        posterior_path
    )

    require(
        posterior_sha
        ==
        EXPECTED["posterior"],
        "Posterior SHA mismatch.",
    )

    print(
        "posterior path =",
        posterior_path,
    )

    print(
        "posterior SHA256 =",
        posterior_sha,
    )

    posterior_records = read_jsonl(
        posterior_path
    )

    print(
        "posterior records =",
        len(posterior_records),
    )

    require(
        len(posterior_records) == 6925,
        (
            "Expected 6925 frozen posterior records; "
            f"got {len(posterior_records)}."
        ),
    )

    required_posterior_keys = {
        "scenario_id",
        "prediction_id",
        "latest_position_H0_m",
        "mean_displacement_H0_m",
        "calibrated_predictive_covariance_H0_m2",
    }

    common_posterior_keys = set(
        posterior_records[0].keys()
    )

    for record in posterior_records[1:]:
        common_posterior_keys.intersection_update(
            record.keys()
        )

    print(
        "posterior common keys =",
        sorted(
            common_posterior_keys
        ),
    )

    missing_posterior_keys = (
        required_posterior_keys
        -
        common_posterior_keys
    )

    require(
        not missing_posterior_keys,
        (
            "Frozen posterior lacks replay inputs: "
            f"{sorted(missing_posterior_keys)}"
        ),
    )

    # ========================================================
    # G. Exact join sufficiency
    # ========================================================

    print()
    print(
        "===== G. EXACT REPLAY JOIN SUFFICIENCY ====="
    )

    manifest_records = read_jsonl(
        POCC_MANIFEST
    )

    match_records = read_jsonl(
        ELIGIBLE_MATCHES
    )

    require(
        len(manifest_records) == 877,
        "Manifest count changed.",
    )

    require(
        len(match_records) == 877,
        "Eligible-match count changed.",
    )

    posterior_index = {
        (
            str(record["scenario_id"]),
            str(record["prediction_id"]),
        ):
            record
        for record in posterior_records
    }

    match_index = {
        (
            str(record["scenario_id"]),
            str(record["prediction_id"]),
            int(record["cohort_index"]),
            int(record["actor_box_index"]),
        ):
            record
        for record in match_records
    }

    manifest_join_failures = []

    vehicle_count = 0

    for record in manifest_records:
        posterior_key = (
            str(record["scenario_id"]),
            str(record["prediction_id"]),
        )

        match_key = (
            str(record["scenario_id"]),
            str(record["prediction_id"]),
            int(record["cohort_index"]),
            int(record["actor_box_index"]),
        )

        if posterior_key not in posterior_index:
            manifest_join_failures.append(
                {
                    "kind":
                        "posterior",

                    "key":
                        posterior_key,
                }
            )

        if match_key not in match_index:
            manifest_join_failures.append(
                {
                    "kind":
                        "current_box",

                    "key":
                        match_key,
                }
            )

        if (
            record["object_type"]
            ==
            "TYPE_VEHICLE"
        ):
            vehicle_count += 1

    require(
        not manifest_join_failures,
        (
            "Replay input join failures: "
            f"{manifest_join_failures[:10]}"
        ),
    )

    require(
        vehicle_count == 719,
        (
            "Expected 719 predictive vehicles; "
            f"got {vehicle_count}."
        ),
    )

    example_vehicle = next(
        record
        for record in manifest_records
        if record["object_type"]
        ==
        "TYPE_VEHICLE"
    )

    example_key = (
        str(example_vehicle["scenario_id"]),
        str(example_vehicle["prediction_id"]),
        int(example_vehicle["cohort_index"]),
        int(example_vehicle["actor_box_index"]),
    )

    example_match = match_index[
        example_key
    ]

    example_posterior = posterior_index[
        (
            example_key[0],
            example_key[1],
        )
    ]

    print(
        "vehicle predictive actors =",
        vehicle_count,
    )

    print(
        "877/877 manifest -> posterior join = PASS"
    )

    print(
        "877/877 manifest -> current_box join = PASS"
    )

    print()
    print(
        "example vehicle key =",
        example_key,
    )

    print(
        "example current_box_H0 keys =",
        sorted(
            example_match[
                "current_box_H0"
            ].keys()
        ),
    )

    print(
        "example posterior replay inputs =",
        {
            key:
                (
                    type(
                        example_posterior[key]
                    ).__name__
                )
            for key in sorted(
                required_posterior_keys
            )
        },
    )

    # ========================================================
    # H. Scientific route classification
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC ROUTE CLASSIFICATION ====="
    )

    exact_cache_only_possible = (
        samplewise_persisted_in_cache
    )

    deterministic_replay_inputs_complete = True

    print(
        "samplewise samples persisted =",
        samplewise_persisted_in_cache,
    )

    print(
        "exact-cache subregion reconstruction possible =",
        exact_cache_only_possible,
    )

    print(
        "frozen deterministic replay inputs complete =",
        deterministic_replay_inputs_complete,
    )

    print(
        "replay executed in this audit = NO"
    )

    print(
        "surrogate occupancy materialized = NO"
    )

    if (
        not exact_cache_only_possible
        and deterministic_replay_inputs_complete
    ):
        route_status = (
            "REQUIRES_PRE_SWEEP_DETERMINISTIC_REPLAY_ADDENDUM"
        )

    elif exact_cache_only_possible:
        route_status = (
            "EXACT_PERSISTED_SAMPLEWISE_ROUTE_AVAILABLE"
        )

    else:
        route_status = (
            "NO_VALID_SURROGATE_ROUTE_RESOLVED"
        )

    print(
        "authoritative next-route status =",
        route_status,
    )

    # ========================================================
    # I. Regression
    # ========================================================

    print()
    print(
        "===== I. FRESH STAGE6 REGRESSION ====="
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
            "Expected 166 tests; "
            f"got {regression['tests']}."
        ),
    )

    print(
        "Stage6 regression = PASS | 166"
    )

    # ========================================================
    # J. Report + handoff
    # ========================================================

    print()
    print(
        "===== J. PART3C/2B0 REPORT + HANDOFF ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part3C/2B0",

        "status":
            (
                "PASS_SURROGATE_PERSISTENCE_"
                "SUFFICIENCY_AUDITED"
            ),

        "upstream":
            upstream,

        "cache_inventory": {
            "root":
                str(POCC_CACHE),

            "total_files":
                len(cache_files),

            "npy_files":
                len(npy_files),

            "json_files":
                len(json_files),

            "other_files":
                [
                    str(path)
                    for path in other_files
                ],

            "maximum_metadata_json_bytes":
                maximum_json_bytes,

            "samplewise_key_paths":
                samplewise_key_paths,

            "samplewise_forecast_persisted":
                samplewise_persisted_in_cache,
        },

        "Part2C2B_runner": {
            "path":
                str(PART2C2B_RUNNER),

            "current_sha256":
                runner_sha,

            "source_markers":
                source_markers,

            "relevant_calls":
                relevant_calls,
        },

        "posterior_route": {
            "path":
                str(posterior_path),

            "sha256":
                posterior_sha,

            "record_count":
                len(posterior_records),

            "required_replay_inputs_present":
                True,
        },

        "join_sufficiency": {
            "manifest_records":
                len(manifest_records),

            "predictive_vehicle_records":
                vehicle_count,

            "posterior_join":
                "877/877 PASS",

            "current_box_join":
                "877/877 PASS",
        },

        "route_classification": {
            "exact_cache_only_subregion_materialization":
                bool(
                    exact_cache_only_possible
                ),

            "deterministic_replay_input_route_complete":
                bool(
                    deterministic_replay_inputs_complete
                ),

            "authoritative_next_route":
                route_status,
        },

        "scientific_execution": {
            "new_sampling":
                False,

            "deterministic_replay":
                False,

            "surrogate_materialization":
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

            "frozen_cache_modified":
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
                "BLOCK6.6_PART3C2B1_FREEZE_"
                "DETERMINISTIC_REPLAY_RECOVERY_CONTRACT_"
                "BEFORE_ANY_REPLAY"
                if route_status
                ==
                "REQUIRES_PRE_SWEEP_DETERMINISTIC_REPLAY_ADDENDUM"
                else
                "BLOCK6.6_PART3C2B_MATERIALIZE_USING_"
                "PERSISTED_SAMPLEWISE_ROUTE"
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
            "6.6-Part3C/2B0",

        "to":
            "6.6-Part3C/2B1",

        "status":
            route_status,

        "report": {
            "path":
                str(REPORT),

            "sha256":
                report_sha,
        },

        "samplewise_forecast_persisted":
            samplewise_persisted_in_cache,

        "deterministic_replay_inputs_complete":
            deterministic_replay_inputs_complete,

        "replay_allowed_yet":
            False,

        "surrogate_materialization_allowed_yet":
            False,

        "policy_sweep_allowed":
            False,

        "numeric_selection_allowed":
            False,

        "formal_outcomes_allowed":
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
    # K. Final
    # ========================================================

    print()
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3C/2B0 — FINAL"
    )
    print("=" * 78)

    print(
        "STATUS = "
        "PASS_SURROGATE_PERSISTENCE_SUFFICIENCY_AUDITED"
    )

    print(
        "cache .npy/.json           =",
        len(npy_files),
        "/",
        len(json_files),
    )

    print(
        "samplewise forecast cached =",
        samplewise_persisted_in_cache,
    )

    print(
        "posterior replay route      = COMPLETE"
    )

    print(
        "manifest -> posterior       = PASS | 877/877"
    )

    print(
        "manifest -> current box     = PASS | 877/877"
    )

    print(
        "predictive vehicles         =",
        vehicle_count,
    )

    print(
        "replay executed             = NO"
    )

    print(
        "surrogate materialized      = NO"
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
        "Stage6 regression           = PASS | 166"
    )

    print(
        "NEXT ROUTE =",
        route_status,
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
        "terminal remains open = YES"
    )

    print("=" * 78)


except BaseException as exc:
    print()
    print("=" * 78)
    print(
        "BLOCK6.6 PART3C/2B0 — CONTROLLED BLOCK"
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
        "new sampling            = NO"
    )

    print(
        "deterministic replay    = NO"
    )

    print(
        "surrogate materialized  = NO"
    )

    print(
        "policy sweep            = NO"
    )

    print(
        "numeric policy selected = NO"
    )

    print(
        "formal outcomes read    = NO"
    )

    print(
        "frozen cache modified   = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no sys.exit().

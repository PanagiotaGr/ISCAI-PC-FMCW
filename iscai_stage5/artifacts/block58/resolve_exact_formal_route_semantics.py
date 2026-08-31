from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")

STAGE3 = ROOT / "iscai_stage3"
STAGE4 = ROOT / "iscai_stage4"
STAGE5 = ROOT / "iscai_stage5"

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

MERGED = (
    STAGE4
    / "artifacts/block48/"
      "formal_neural_outputs.npz"
)

CACHE_DIR = (
    STAGE4
    / "artifacts/block48/cache"
)

CACHE_MANIFEST = (
    STAGE4
    / "artifacts/block48/"
      "formal_cache_manifest.json"
)

BLOCK48 = (
    STAGE4
    / "scripts/"
      "run_block48_formal_evaluation.py"
)

FORMAL_RUNTIME = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "formal_runtime.py"
)

BLOCK52_REPORT = (
    STAGE5
    / "reports/"
      "block52_receiver_angular_posterior.json"
)

REPORT = (
    STAGE5
    / "artifacts/block58/"
      "exact_formal_route_semantics.json"
)


EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_MERGED_SHA = (
    "a3e8af17962a28a533070db23e53cee1"
    "0571d13ab641c74b9db21e79c9a5c8c9"
)

EXPECTED_CACHE_MANIFEST_SHA = (
    "3f2cc58393fdf2ed2aace41bbf7ba4c9"
    "15f92cc7d16e5b6f55f063af344e0baa"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha256(path: Path):
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


def write_json(path: Path, payload):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def preview_value(value):
    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if isinstance(
        value,
        bytes,
    ):
        try:
            return value.decode(
                "utf-8"
            )
        except Exception:
            return repr(value)

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


def describe_npz(path: Path):
    result = {
        "path": str(path),
        "sha256": file_sha256(path),
        "keys": [],
        "first_dim_counts": {},
        "identity_previews": {},
    }

    first_dims = []

    #
    # Trusted local frozen artifact created by this project.
    # allow_pickle is used only so that object/string identity
    # arrays, if any, can be inspected.
    #
    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        for key in data.files:
            array = np.asarray(
                data[key]
            )

            entry = {
                "key": key,
                "shape": list(
                    array.shape
                ),
                "dtype": str(
                    array.dtype
                ),
            }

            if array.ndim >= 1:
                first_dims.append(
                    int(
                        array.shape[0]
                    )
                )

            result[
                "keys"
            ].append(entry)

            lower = key.lower()

            identity_like = any(
                token in lower
                for token in (
                    "scenario",
                    "track_index",
                    "track_id",
                    "sample_id",
                    "class_id",
                    "object_type",
                    "prediction_sha",
                )
            )

            truth_like = any(
                token in lower
                for token in (
                    "truth",
                    "future",
                    "label",
                )
            )

            if (
                identity_like
                and
                not truth_like
                and
                array.ndim >= 1
            ):
                flat = array.reshape(-1)

                result[
                    "identity_previews"
                ][
                    key
                ] = [
                    preview_value(
                        value
                    )
                    for value in flat[:5]
                ]

                #
                # Uniqueness is identity/schema evidence,
                # not a scientific metric.
                #
                try:
                    result[
                        "identity_previews"
                    ][
                        key
                    ].append(
                        {
                            "unique_count":
                                len(
                                    set(
                                        preview_value(v)
                                        for v in flat
                                    )
                                )
                        }
                    )
                except Exception:
                    pass

    result[
        "first_dim_counts"
    ] = dict(
        Counter(
            first_dims
        )
    )

    return result


def print_npz_schema(title, description):
    print()
    print(
        "============================================================"
    )
    print(title)
    print(
        "============================================================"
    )

    print(
        "path =",
        description["path"],
    )

    print(
        "SHA256 =",
        description["sha256"],
    )

    print(
        "first-dimension frequencies =",
        description[
            "first_dim_counts"
        ],
    )

    print()
    print("KEY / SHAPE / DTYPE")

    for item in description["keys"]:
        print(
            item["key"],
            "=",
            tuple(
                item["shape"]
            ),
            item["dtype"],
        )

    print()
    print("IDENTITY PREVIEWS")

    for key, value in (
        description[
            "identity_previews"
        ].items()
    ):
        print(
            key,
            "=",
            value,
        )


def read_formal_manifest_metadata():
    rows = tuple(
        json.loads(line)
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    require(
        len(rows) == 120,
        "Formal manifest is not exactly N=120.",
    )

    identity_fields = (
        "scenario_id",
        "source_shard",
        "compact_record_offset",
        "selection_hash",
        "split",
    )

    previews = []

    for row in rows[:3]:
        previews.append({
            key:
                row.get(key)
            for key in identity_fields
            if key in row
        })

    return {
        "row_count": len(rows),
        "keys": sorted(
            rows[0].keys()
        )
        if rows
        else [],
        "identity_preview": previews,
    }


def json_interesting(
    value,
    *,
    prefix="",
    output=None,
):
    if output is None:
        output = []

    if isinstance(value, dict):

        for key, child in (
            value.items()
        ):
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            json_interesting(
                child,
                prefix=child_prefix,
                output=output,
            )

        return output

    if isinstance(value, list):

        if len(value) > 20:
            output.append({
                "path": prefix,
                "value":
                    f"<list length={len(value)}>",
            })

            return output

        for index, child in enumerate(
            value
        ):
            json_interesting(
                child,
                prefix=(
                    f"{prefix}[{index}]"
                ),
                output=output,
            )

        return output

    lower = (
        prefix.lower()
        + " "
        + str(value).lower()
    )

    if any(
        token in lower
        for token in (
            "scenario",
            "shard",
            "cache",
            "formal",
            "merged",
            "row",
            "sample",
            "count",
            "sha",
            "prediction",
            "gaussian",
        )
    ):
        output.append({
            "path": prefix,
            "value": value,
        })

    return output


def source_windows(
    path: Path,
    *,
    terms,
    radius=7,
    min_line=None,
    max_windows=35,
):
    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    indices = []

    for index, line in enumerate(
        lines
    ):
        number = index + 1

        if (
            min_line is not None
            and
            number < min_line
        ):
            continue

        lower = line.lower()

        if any(
            term.lower() in lower
            for term in terms
        ):
            indices.append(index)

    ranges = []

    for index in indices:
        start = max(
            0,
            index - radius,
        )

        stop = min(
            len(lines),
            index + radius + 1,
        )

        if (
            ranges
            and
            start <= ranges[-1][1]
        ):
            ranges[-1] = (
                ranges[-1][0],
                max(
                    ranges[-1][1],
                    stop,
                ),
            )

        else:
            ranges.append(
                (
                    start,
                    stop,
                )
            )

    result = []

    for start, stop in (
        ranges[
            :max_windows
        ]
    ):
        result.append({
            "start_line":
                start + 1,

            "end_line":
                stop,

            "text":
                "\n".join(
                    f"{number + 1:04d}: "
                    f"{lines[number]}"
                    for number in range(
                        start,
                        stop,
                    )
                ),
        })

    return result


def extract_function(
    path: Path,
    function_name: str,
):
    source = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    tree = ast.parse(source)

    lines = source.splitlines()

    for node in ast.walk(tree):

        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and
            node.name == function_name
        ):
            start = int(
                node.lineno
            )

            stop = int(
                node.end_lineno
            )

            return {
                "name":
                    function_name,

                "start_line":
                    start,

                "end_line":
                    stop,

                "text":
                    "\n".join(
                        f"{number:04d}: "
                        f"{lines[number - 1]}"
                        for number in range(
                            start,
                            stop + 1,
                        )
                    ),
            }

    return None


def stage5_receiver_windows():
    hits = []

    terms = (
        "nearest_causal_vehicle_ahead",
        "primary_policy",
        "receiver_geometry_in_h0",
        "receiver_policy",
    )

    for path in sorted(
        (
            STAGE5
            / "src/iscai_stage5"
        ).rglob("*.py"),
        key=str,
    ):
        windows = source_windows(
            path,
            terms=terms,
            radius=8,
            max_windows=12,
        )

        if windows:
            hits.append({
                "path": str(path),
                "windows": windows,
            })

    return hits


def semantic_key_groups(
    description,
):
    keys = [
        item["key"]
        for item in description["keys"]
    ]

    def select(*tokens):
        return [
            key
            for key in keys
            if any(
                token
                in
                key.lower()
                for token in tokens
            )
        ]

    return {
        "scenario_identity":
            select(
                "scenario"
            ),

        "track_identity":
            select(
                "track_id",
                "track_index",
                "target_index",
            ),

        "Gaussian_mean":
            [
                key
                for key in keys
                if (
                    "gaussian"
                    in key.lower()
                    and
                    "mean"
                    in key.lower()
                )
            ],

        "Gaussian_scale_or_covariance":
            [
                key
                for key in keys
                if (
                    "gaussian"
                    in key.lower()
                    and
                    (
                        "scale"
                        in key.lower()
                        or
                        "cov"
                        in key.lower()
                    )
                )
            ],

        "calibration_explicit":
            select(
                "calibrat",
                "alpha"
            ),

        "anchor_or_current_H0":
            select(
                "anchor",
                "origin",
                "current",
                "h0",
            ),

        "class_identity":
            select(
                "class_id",
                "object_type",
                "class_name",
            ),
    }


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 2/2 — PRE-EXECUTION FORMAL ROUTE LOCK"
    )
    print(
        "EXACT SAMPLE-LEVEL SEMANTIC RESOLUTION"
    )
    print(
        "============================================================"
    )

    print(
        "formal N=120 artifact read = YES — SCHEMA/ROUTE ONLY"
    )
    print(
        "formal performance metrics = NO"
    )
    print(
        "Stage4 inference           = NO"
    )
    print(
        "Stage5 formal controller   = NOT RUN"
    )
    print(
        "training/recalibration     = NO"
    )
    print(
        "post-hoc tuning            = NO"
    )

    required = (
        FORMAL_MANIFEST,
        MERGED,
        CACHE_MANIFEST,
        BLOCK48,
        FORMAL_RUNTIME,
        BLOCK52_REPORT,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing route-lock prerequisite(s): "
            + ", ".join(missing)
        ),
    )

    # --------------------------------------------------------
    # A. Immutable hashes
    # --------------------------------------------------------

    formal_sha = file_sha256(
        FORMAL_MANIFEST
    )

    merged_sha = file_sha256(
        MERGED
    )

    cache_manifest_sha = (
        file_sha256(
            CACHE_MANIFEST
        )
    )

    require(
        formal_sha
        ==
        EXPECTED_FORMAL_MANIFEST_SHA,
        (
            "Frozen N120 manifest SHA changed."
        ),
    )

    require(
        merged_sha
        ==
        EXPECTED_MERGED_SHA,
        (
            "Frozen formal neural-output "
            "artifact SHA changed."
        ),
    )

    require(
        cache_manifest_sha
        ==
        EXPECTED_CACHE_MANIFEST_SHA,
        (
            "Frozen formal cache-manifest "
            "SHA changed."
        ),
    )

    print()
    print(
        "===== A. IMMUTABLE FORMAL ARTIFACTS ====="
    )

    print(
        "formal manifest SHA       = PASS"
    )
    print(
        "formal N                  = 120"
    )
    print(
        "merged neural output SHA  = PASS"
    )
    print(
        "cache manifest SHA        = PASS"
    )

    # --------------------------------------------------------
    # B. Formal manifest identity
    # --------------------------------------------------------

    manifest_metadata = (
        read_formal_manifest_metadata()
    )

    print()
    print(
        "===== B. FORMAL MANIFEST IDENTITY CONTRACT ====="
    )

    print(
        "row count =",
        manifest_metadata[
            "row_count"
        ],
    )

    print(
        "keys =",
        manifest_metadata[
            "keys"
        ],
    )

    print(
        "identity preview =",
        manifest_metadata[
            "identity_preview"
        ],
    )

    # --------------------------------------------------------
    # C. Merged formal NPZ
    # --------------------------------------------------------

    merged_description = (
        describe_npz(
            MERGED
        )
    )

    print_npz_schema(
        "C. FORMAL_NEURAL_OUTPUTS.NPZ EXACT SCHEMA",
        merged_description,
    )

    key_groups = (
        semantic_key_groups(
            merged_description
        )
    )

    print()
    print(
        "SEMANTIC KEY GROUPS"
    )

    for key, values in (
        key_groups.items()
    ):
        print(
            key,
            "=",
            values,
        )

    # --------------------------------------------------------
    # D. Per-scene cache schema
    # --------------------------------------------------------

    shards = sorted(
        CACHE_DIR.glob(
            "*.npz"
        ),
        key=lambda path:
            path.name,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "D. PER-SCENE FORMAL CACHE SCHEMA"
    )
    print(
        "============================================================"
    )

    print(
        "cache shard count =",
        len(shards),
    )

    shard_descriptions = []

    if shards:
        selected = [
            shards[0],
            shards[
                len(shards) // 2
            ],
            shards[-1],
        ]

        deduplicated = []

        for path in selected:
            if path not in deduplicated:
                deduplicated.append(
                    path
                )

        for path in deduplicated:
            description = (
                describe_npz(
                    path
                )
            )

            shard_descriptions.append(
                description
            )

            print_npz_schema(
                (
                    "CACHE SHARD "
                    + path.name
                ),
                description,
            )

    # --------------------------------------------------------
    # E. Cache manifest semantics
    # --------------------------------------------------------

    cache_payload = json.loads(
        CACHE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    cache_interesting = (
        json_interesting(
            cache_payload
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "E. FORMAL CACHE MANIFEST SEMANTICS"
    )
    print(
        "============================================================"
    )

    for item in cache_interesting[:120]:
        print(
            item["path"],
            "=",
            item["value"],
        )

    # --------------------------------------------------------
    # F. Exact Stage4 creation/calibration/save semantics
    # --------------------------------------------------------

    block48_windows = (
        source_windows(
            BLOCK48,
            terms=(
                "eligible_formal_targets",
                "denormalize_gaussian",
                "gaussian_mean",
                "gaussian_scale",
                "variance_scale",
                "calibrat",
                "tracks_to_predict",
                "np.savez",
                "save_npz",
                "MERGED",
                "concatenate",
                "cache_path",
            ),
            radius=8,
            min_line=1000,
            max_windows=35,
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "F. BLOCK4.8 CREATION / CALIBRATION / SAVE WINDOWS"
    )
    print(
        "============================================================"
    )

    for index, window in enumerate(
        block48_windows,
        start=1,
    ):
        print()
        print(
            f"[BLOCK48 {index}] "
            f"lines "
            f"{window['start_line']}-"
            f"{window['end_line']}"
        )

        print(
            window["text"]
        )

    # --------------------------------------------------------
    # G. Exact eligible-target semantics
    # --------------------------------------------------------

    eligible_function = (
        extract_function(
            FORMAL_RUNTIME,
            "eligible_formal_targets",
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "G. STAGE4 ELIGIBLE_FORMAL_TARGETS"
    )
    print(
        "============================================================"
    )

    if eligible_function is None:
        print(
            "eligible_formal_targets function = NOT FOUND"
        )

    else:
        print(
            eligible_function[
                "text"
            ]
        )

    # --------------------------------------------------------
    # H. Stage5 receiver mapping source
    # --------------------------------------------------------

    receiver_windows = (
        stage5_receiver_windows()
    )

    print()
    print(
        "============================================================"
    )
    print(
        "H. STAGE5 FROZEN RECEIVER-SELECTION SOURCE"
    )
    print(
        "============================================================"
    )

    for file_hit in receiver_windows:
        print()
        print(
            "FILE =",
            file_hit[
                "path"
            ],
        )

        for index, window in enumerate(
            file_hit[
                "windows"
            ],
            start=1,
        ):
            print()
            print(
                f"[RECEIVER {index}] "
                f"lines "
                f"{window['start_line']}-"
                f"{window['end_line']}"
            )

            print(
                window["text"]
            )

    # --------------------------------------------------------
    # I. Remaining formal-coverage field
    # --------------------------------------------------------

    block52 = json.loads(
        BLOCK52_REPORT.read_text(
            encoding="utf-8"
        )
    )

    remaining = (
        block52.get(
            "remaining_development_fields",
            []
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "I. PRE-FORMAL COVERAGE ACCEPTANCE DEBT"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.2 remaining fields =",
        remaining,
    )

    empirical_tolerance_pending = any(
        "formal_empirical_coverage_tolerance"
        in str(value)
        for value in remaining
    )

    print(
        "formal empirical coverage tolerance pending =",
        empirical_tolerance_pending,
    )

    print(
        "formal empirical coverage metrics computed = NO"
    )

    print(
        "post-hoc threshold selection = NO"
    )

    # --------------------------------------------------------
    # J. Evidence-only route classification
    # --------------------------------------------------------

    has_scenario = bool(
        key_groups[
            "scenario_identity"
        ]
    )

    has_track = bool(
        key_groups[
            "track_identity"
        ]
    )

    has_gaussian_mean = bool(
        key_groups[
            "Gaussian_mean"
        ]
    )

    has_gaussian_uncertainty = bool(
        key_groups[
            "Gaussian_scale_or_covariance"
        ]
    )

    schema_candidate = (
        has_scenario
        and
        has_track
        and
        has_gaussian_mean
        and
        has_gaussian_uncertainty
    )

    payload = {
        "status":
            "PASS_ROUTE_SEMANTICS_EXTRACTED",

        "scientific_execution": {
            "formal_artifact_schema_read":
                True,

            "formal_metrics_computed":
                False,

            "Stage4_inference":
                False,

            "Stage5_formal_controller_run":
                False,

            "training":
                False,

            "recalibration":
                False,

            "posthoc_tuning":
                False,

            "upstream_modified":
                False,
        },

        "formal_manifest": {
            "path":
                str(
                    FORMAL_MANIFEST
                ),

            "sha256":
                formal_sha,

            "metadata":
                manifest_metadata,
        },

        "merged_formal_prediction": {
            "description":
                merged_description,

            "semantic_key_groups":
                key_groups,

            "schema_supports_candidate_mapping":
                schema_candidate,
        },

        "cache": {
            "manifest_path":
                str(
                    CACHE_MANIFEST
                ),

            "manifest_sha256":
                cache_manifest_sha,

            "shard_count":
                len(
                    shards
                ),

            "sample_shards":
                shard_descriptions,
        },

        "Block48_source_windows":
            block48_windows,

        "eligible_formal_targets":
            eligible_function,

        "Stage5_receiver_source":
            receiver_windows,

        "coverage_acceptance_debt": {
            "Block52_remaining_fields":
                remaining,

            "formal_empirical_coverage_tolerance_pending":
                empirical_tolerance_pending,

            "formal_metrics_seen":
                False,
        },

        "route_decision":
            (
                "PENDING_SEMANTIC_REVIEW_"
                "NO_STAGE4_INFERENCE_YET"
            ),
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 EXACT ROUTE SEMANTICS FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "formal N=120 manifest SHA     = PASS"
    )

    print(
        "formal neural artifact SHA    = PASS"
    )

    print(
        "formal cache manifest SHA     = PASS"
    )

    print(
        "scenario identity key         =",
        has_scenario,
    )

    print(
        "track identity key            =",
        has_track,
    )

    print(
        "Gaussian mean key             =",
        has_gaussian_mean,
    )

    print(
        "Gaussian scale/covariance key =",
        has_gaussian_uncertainty,
    )

    print(
        "schema mapping candidate      =",
        schema_candidate,
    )

    print(
        "coverage tolerance debt       =",
        empirical_tolerance_pending,
    )

    print(
        "formal metrics computed       = NO"
    )

    print(
        "Stage4 inference              = NO"
    )

    print(
        "Stage5 formal controller      = NOT RUN"
    )

    print(
        "STATUS = PASS_ROUTE_SEMANTICS_EXTRACTED"
    )

    print(
        "route decision = PENDING SEMANTIC REVIEW"
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
        "BLOCK 5.8 EXACT ROUTE SEMANTICS = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print(
        "formal metrics computed = NO"
    )

    print(
        "Stage4 inference = NO"
    )

    print(
        "Stage5 formal controller = NOT RUN"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

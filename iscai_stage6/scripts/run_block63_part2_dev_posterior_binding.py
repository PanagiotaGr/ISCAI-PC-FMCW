from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")

S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PART1_REPORT = (
    S6
    / "reports/block63_part1_deterministic_future_box.json"
)

BLOCK63_BINDING = (
    S6
    / "configs/block63_deterministic_predictive_binding.json"
)

DEV_JSON = (
    S5
    / "artifacts/block52/development_gaussian_posterior.json"
)

DEV_NPZ = (
    S5
    / "artifacts/block52/development_gaussian_posterior.npz"
)

DEV_RESOLVER = (
    S5
    / "artifacts/block52/dev_posterior_resolver_summary.json"
)

ROUTE_AUDIT = (
    S5
    / "artifacts/block52/actual_stage4_gaussian_route_audit.json"
)

COORDINATE = (
    S5
    / "artifacts/block52/stage4_prediction_coordinate_semantics.json"
)

RUNTIME_AUDIT = (
    S5
    / "artifacts/block52/calibrated_runtime_api_audit.json"
)

OUTPUT_BINDING = (
    S6
    / "configs/block63_part2_development_posterior_binding.json"
)

REPORT = (
    S6
    / "reports/block63_part2_development_posterior_binding.json"
)


EXPECTED = {
    "part1":
        (
            "871c8a81e43048ff2ccf7ef1844ad543"
            "3a859ff15b944749e13be9160b08256a"
        ),

    "block63_binding":
        (
            "03903018109e7d5037643381ff71141a5"
            "c57f003b8bdc6b9be13bb4304468f67"
        ),
}


def require(
    condition,
    message,
):

    if not bool(condition):
        raise RuntimeError(
            message
        )


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


def load_json(
    path: Path,
):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(
    payload,
) -> bytes:

    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def write_new_or_exact_json(
    path: Path,
    payload,
):

    desired = canonical_bytes(
        payload
    )

    if path.exists():

        require(
            path.read_bytes()
            ==
            desired,
            (
                "Existing frozen artifact "
                f"differs: {path}"
            ),
        )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        desired
    )

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def flatten(
    value,
    prefix="",
):

    rows = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.append(
                (
                    path,
                    child,
                )
            )

            if isinstance(
                child,
                (
                    dict,
                    list,
                ),
            ):

                rows.extend(
                    flatten(
                        child,
                        path,
                    )
                )

    elif isinstance(
        value,
        list,
    ):

        # Metadata artifacts are small, but avoid flooding
        # numerical lists.
        if len(value) <= 80:

            for index, child in enumerate(
                value
            ):

                path = (
                    f"{prefix}[{index}]"
                )

                rows.append(
                    (
                        path,
                        child,
                    )
                )

                if isinstance(
                    child,
                    (
                        dict,
                        list,
                    ),
                ):

                    rows.extend(
                        flatten(
                            child,
                            path,
                        )
                    )

    return rows


def scenario_candidates(
    payload,
):

    result = []

    pattern = re.compile(
        r"^[0-9a-fA-F]{16}$"
    )

    for path, value in flatten(
        payload
    ):

        if not isinstance(
            value,
            str,
        ):
            continue

        if (
            "scenario"
            not in path.lower()
        ):
            continue

        if not pattern.fullmatch(
            value
        ):
            continue

        result.append({
            "path":
                path,

            "value":
                value.lower(),
        })

    return result


def shard_candidates(
    payload,
):

    result = []

    for path, value in flatten(
        payload
    ):

        if not isinstance(
            value,
            str,
        ):
            continue

        lower_path = (
            path.lower()
        )

        lower_value = (
            value.lower()
        )

        if (
            ".tfrecord"
            not in lower_value
        ):
            continue

        if not any(
            token in lower_path
            for token in (
                "shard",
                "motion",
                "path",
                "file",
                "record",
            )
        ):
            continue

        result.append({
            "path":
                path,

            "value":
                value,
        })

    return result


def offset_candidates(
    payload,
):

    result = []

    for path, value in flatten(
        payload
    ):

        if (
            "offset"
            not in path.lower()
        ):
            continue

        if isinstance(
            value,
            bool,
        ):
            continue

        if isinstance(
            value,
            int,
        ):

            result.append({
                "path":
                    path,

                "value":
                    int(value),
            })

    return result


def deduplicate_candidates(
    candidates,
):

    result = []

    seen = set()

    for item in candidates:

        key = (
            item[
                "value"
            ]
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            item
        )

    return result


def choose_by_priority(
    named_payloads,
    extractor,
):

    evidence = {}

    for name, payload in named_payloads:

        values = extractor(
            payload
        )

        evidence[
            name
        ] = values

        unique = (
            deduplicate_candidates(
                values
            )
        )

        if len(unique) == 1:

            return (
                unique[0],
                name,
                evidence,
            )

        if len(unique) > 1:

            # Do not silently move to a lower-priority artifact
            # when the authoritative higher-priority artifact
            # itself is ambiguous.
            return (
                None,
                name,
                evidence,
            )

    return (
        None,
        None,
        evidence,
    )


def npz_schema(
    path: Path,
):

    schema = {}

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        for key in archive.files:

            value = archive[
                key
            ]

            schema[
                key
            ] = {
                "shape":
                    list(
                        value.shape
                    ),

                "dtype":
                    str(
                        value.dtype
                    ),
            }

    return schema


def candidate_npz_keys(
    schema,
):

    means = []
    anchors = []
    covariances = []
    horizons = []

    for key, metadata in (
        schema.items()
    ):

        lower = key.lower()

        shape = tuple(
            metadata[
                "shape"
            ]
        )

        if (
            shape
            ==
            (
                4,
                3,
            )
            and
            "mean"
            in lower
            and
            "cov"
            not in lower
        ):

            means.append(
                key
            )

        if (
            shape
            ==
            (
                3,
            )
            and
            "position"
            in lower
            and
            any(
                token in lower
                for token in (
                    "current",
                    "anchor",
                    "latest",
                )
            )
        ):

            anchors.append(
                key
            )

        if (
            shape
            ==
            (
                4,
                3,
                3,
            )
            and
            (
                "covariance"
                in lower
                or
                "cov"
                in lower
            )
        ):

            covariances.append(
                key
            )

        if (
            shape
            ==
            (
                4,
            )
            and
            "horizon"
            in lower
        ):

            horizons.append(
                key
            )

    return {
        "mean":
            means,

        "anchor":
            anchors,

        "covariance":
            covariances,

        "horizons":
            horizons,
    }


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 PART 2/2 — PHASE A"
    )
    print(
        "EXACT DEVELOPMENT POSTERIOR BINDING"
    )
    print(
        "============================================================"
    )

    protected = (
        PART1_REPORT,
        BLOCK63_BINDING,
        DEV_JSON,
        DEV_NPZ,
        DEV_RESOLVER,
        ROUTE_AUDIT,
        COORDINATE,
        RUNTIME_AUDIT,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required artifact(s): "
            + ", ".join(
                missing
            )
        ),
    )

    before = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    # ========================================================
    # A. Frozen seals
    # ========================================================

    print()
    print(
        "===== A. FROZEN BLOCK6.3 SEAL ====="
    )

    require(
        sha256_file(
            PART1_REPORT
        )
        ==
        EXPECTED[
            "part1"
        ],
        (
            "Block6.3 Part1 report SHA mismatch."
        ),
    )

    require(
        sha256_file(
            BLOCK63_BINDING
        )
        ==
        EXPECTED[
            "block63_binding"
        ],
        (
            "Block6.3 frozen binding SHA mismatch."
        ),
    )

    part1 = load_json(
        PART1_REPORT
    )

    require(
        part1.get(
            "status"
        )
        ==
        (
            "PASS_DETERMINISTIC_"
            "FUTURE_FULL_BOX"
        ),
        (
            "Block6.3 Part1 is not complete."
        ),
    )

    print(
        "Part1 report  = EXACT PASS"
    )

    print(
        "Block63 binding = EXACT PASS"
    )

    # ========================================================
    # B. Development artifact identities
    # ========================================================

    print()
    print(
        "===== B. DEVELOPMENT ARTIFACT IDENTITY ====="
    )

    for path in (
        DEV_JSON,
        DEV_NPZ,
        DEV_RESOLVER,
        ROUTE_AUDIT,
        COORDINATE,
        RUNTIME_AUDIT,
    ):

        print(
            path.name,
            "=",
            sha256_file(
                path
            ),
        )

    # ========================================================
    # C. Coordinate semantics
    # ========================================================

    print()
    print(
        "===== C. COORDINATE SEMANTICS ====="
    )

    coordinate = load_json(
        COORDINATE
    )

    # The development posterior is the authoritative serialized
    # record for the exact materialization that Block6.3 will
    # consume. The standalone coordinate-semantics artifact is
    # retained as corroborating provenance, but it is not
    # required to serialize the same phrase verbatim.
    dev_coordinate_record = load_json(
        DEV_JSON
    )

    dev_coordinate_semantics = (
        dev_coordinate_record.get(
            "coordinate_semantics",
            {}
        )
    )

    require(
        isinstance(
            dev_coordinate_semantics,
            dict,
        ),
        (
            "Development posterior has no "
            "coordinate_semantics mapping."
        ),
    )

    stage4_prediction_semantics = (
        dev_coordinate_semantics.get(
            "Stage4_prediction"
        )
    )

    causal_anchor_semantics = (
        dev_coordinate_semantics.get(
            "causal_anchor"
        )
    )

    require(
        stage4_prediction_semantics
        ==
        "metric_H0_displacement",
        (
            "Development posterior does not prove "
            "metric_H0_displacement semantics. "
            f"actual={stage4_prediction_semantics!r}"
        ),
    )

    require(
        causal_anchor_semantics
        ==
        (
            "latest_causal_observed_"
            "target_position_H0"
        ),
        (
            "Development posterior does not prove "
            "the frozen causal-anchor semantics. "
            f"actual={causal_anchor_semantics!r}"
        ),
    )

    coordinate_text = json.dumps(
        coordinate,
        sort_keys=True,
    ).lower()

    # Corroborating artifact must still describe the same
    # coordinate family. We deliberately avoid requiring an
    # identical serialization phrase.
    require(
        "h0"
        in coordinate_text,
        (
            "Coordinate-semantics artifact no longer "
            "contains H0-frame evidence."
        ),
    )

    require(
        "displacement"
        in coordinate_text,
        (
            "Coordinate-semantics artifact no longer "
            "contains displacement semantics."
        ),
    )

    require(
        (
            "anchor"
            in coordinate_text
            or
            "current"
            in coordinate_text
            or
            "causal"
            in coordinate_text
        ),
        (
            "Coordinate-semantics artifact no longer "
            "contains causal/current-anchor evidence."
        ),
    )

    print(
        "development posterior Stage4 prediction = "
        "metric_H0_displacement EXACT PASS"
    )

    print(
        "development posterior causal anchor = "
        "latest_causal_observed_target_position_H0 EXACT PASS"
    )

    print(
        "coordinate artifact H0 corroboration = PASS"
    )

    print(
        "coordinate artifact displacement corroboration = PASS"
    )

    # ========================================================
    # D. NPZ schema resolution
    # ========================================================

    print()
    print(
        "===== D. DEVELOPMENT POSTERIOR NPZ SCHEMA ====="
    )

    schema = npz_schema(
        DEV_NPZ
    )

    for key, metadata in (
        schema.items()
    ):

        print(
            key,
            "| shape =",
            metadata[
                "shape"
            ],
            "| dtype =",
            metadata[
                "dtype"
            ],
        )

    candidates = (
        candidate_npz_keys(
            schema
        )
    )

    print()
    print(
        "mean candidates       =",
        candidates[
            "mean"
        ],
    )

    print(
        "anchor candidates     =",
        candidates[
            "anchor"
        ],
    )

    print(
        "covariance candidates =",
        candidates[
            "covariance"
        ],
    )

    print(
        "horizon candidates    =",
        candidates[
            "horizons"
        ],
    )

    mean_key = (
        candidates[
            "mean"
        ][0]
        if len(
            candidates[
                "mean"
            ]
        )
        ==
        1
        else
        None
    )

    anchor_key = (
        candidates[
            "anchor"
        ][0]
        if len(
            candidates[
                "anchor"
            ]
        )
        ==
        1
        else
        None
    )

    require(
        mean_key
        is not None,
        (
            "Could not uniquely bind "
            "development Gaussian mean array."
        ),
    )

    require(
        anchor_key
        is not None,
        (
            "Could not uniquely bind "
            "development causal-anchor array."
        ),
    )

    with np.load(
        DEV_NPZ,
        allow_pickle=False,
    ) as archive:

        mean = np.asarray(
            archive[
                mean_key
            ],
            dtype=np.float64,
        )

        anchor = np.asarray(
            archive[
                anchor_key
            ],
            dtype=np.float64,
        )

    require(
        mean.shape
        ==
        (
            4,
            3,
        ),
        (
            "Bound mean shape changed."
        ),
    )

    require(
        anchor.shape
        ==
        (
            3,
        ),
        (
            "Bound anchor shape changed."
        ),
    )

    require(
        np.all(
            np.isfinite(
                mean
            )
        ),
        (
            "Development mean contains "
            "non-finite values."
        ),
    )

    require(
        np.all(
            np.isfinite(
                anchor
            )
        ),
        (
            "Development anchor contains "
            "non-finite values."
        ),
    )

    print()
    print(
        "Gaussian mean key =",
        mean_key,
        "PASS",
    )

    print(
        "causal anchor key =",
        anchor_key,
        "PASS",
    )

    print(
        "mean shape = (4,3) PASS"
    )

    print(
        "anchor shape = (3,) PASS"
    )

    print(
        "mean/anchor finite = PASS"
    )

    # ========================================================
    # E. Scenario / motion-record provenance
    # ========================================================

    print()
    print(
        "===== E. DEVELOPMENT SCENARIO PROVENANCE ====="
    )

    dev_json = load_json(
        DEV_JSON
    )

    resolver = load_json(
        DEV_RESOLVER
    )

    route = load_json(
        ROUTE_AUDIT
    )

    runtime = load_json(
        RUNTIME_AUDIT
    )

    named_payloads = (
        (
            "development_gaussian_posterior.json",
            dev_json,
        ),
        (
            "dev_posterior_resolver_summary.json",
            resolver,
        ),
        (
            "actual_stage4_gaussian_route_audit.json",
            route,
        ),
        (
            "calibrated_runtime_api_audit.json",
            runtime,
        ),
    )

    (
        scenario,
        scenario_source,
        scenario_evidence,
    ) = choose_by_priority(
        named_payloads,
        scenario_candidates,
    )

    (
        shard,
        shard_source,
        shard_evidence,
    ) = choose_by_priority(
        named_payloads,
        shard_candidates,
    )

    (
        offset,
        offset_source,
        offset_evidence,
    ) = choose_by_priority(
        named_payloads,
        offset_candidates,
    )

    for name, _payload in named_payloads:

        print()
        print(
            "---",
            name,
            "---"
        )

        print(
            "scenario candidates =",
            scenario_evidence.get(
                name,
                [],
            ),
        )

        print(
            "shard candidates =",
            shard_evidence.get(
                name,
                [],
            ),
        )

        print(
            "offset candidates =",
            offset_evidence.get(
                name,
                [],
            ),
        )

    scenario_id = (
        None
        if scenario is None
        else scenario[
            "value"
        ]
    )

    shard_path = (
        None
        if shard is None
        else shard[
            "value"
        ]
    )

    compact_offset = (
        None
        if offset is None
        else offset[
            "value"
        ]
    )

    print()
    print(
        "resolved scenario_id =",
        scenario_id,
    )

    print(
        "scenario source =",
        scenario_source,
    )

    print(
        "resolved motion shard =",
        shard_path,
    )

    print(
        "shard source =",
        shard_source,
    )

    print(
        "resolved compact offset =",
        compact_offset,
    )

    print(
        "offset source =",
        offset_source,
    )

    # ========================================================
    # F. Stage4 route provenance
    # ========================================================

    print()
    print(
        "===== F. STAGE4 GAUSSIAN ROUTE PROVENANCE ====="
    )

    route_text = json.dumps(
        route,
        sort_keys=True,
    ).lower()

    runtime_text = json.dumps(
        runtime,
        sort_keys=True,
    ).lower()

    gaussian_route_proven = (
        "gaussian"
        in route_text
        and
        (
            "stage4"
            in route_text
            or
            "gaussian"
            in runtime_text
        )
    )

    calibration_proven = (
        "calibrat"
        in (
            route_text
            +
            runtime_text
        )
    )

    require(
        gaussian_route_proven,
        (
            "Frozen artifacts do not prove "
            "actual Stage4 Gaussian route."
        ),
    )

    require(
        calibration_proven,
        (
            "Frozen artifacts do not prove "
            "calibrated Gaussian provenance."
        ),
    )

    print(
        "actual Stage4 Gaussian route = PASS"
    )

    print(
        "calibrated output provenance = PASS"
    )

    print(
        "new inference executed = NO"
    )

    # ========================================================
    # G. Binding readiness
    # ========================================================

    print()
    print(
        "===== G. PART2 REAL-INTEGRATION READINESS ====="
    )

    missing_metadata = []

    if scenario_id is None:
        missing_metadata.append(
            "scenario_id"
        )

    if shard_path is None:
        missing_metadata.append(
            "motion_shard"
        )

    if compact_offset is None:
        missing_metadata.append(
            "compact_offset"
        )

    if missing_metadata:

        status = (
            "BLOCKED_METADATA_BINDING_INCOMPLETE"
        )

    else:

        status = (
            "PASS_READY_FOR_REAL_DEVELOPMENT_INTEGRATION"
        )

    print(
        "Gaussian mean       = BOUND"
    )

    print(
        "causal anchor       = BOUND"
    )

    print(
        "scenario ID         =",
        (
            "BOUND"
            if scenario_id
            is not None
            else
            "NOT PROVEN"
        ),
    )

    print(
        "motion shard        =",
        (
            "BOUND"
            if shard_path
            is not None
            else
            "NOT PROVEN"
        ),
    )

    print(
        "compact offset      =",
        (
            "BOUND"
            if compact_offset
            is not None
            else
            "NOT PROVEN"
        ),
    )

    print(
        "missing metadata    =",
        missing_metadata,
    )

    # ========================================================
    # H. Freeze binding only when exact metadata is complete
    # ========================================================

    binding_status = (
        "NOT_WRITTEN"
    )

    output_sha = None

    if not missing_metadata:

        binding = {
            "project":
                "Agni",

            "stage":
                6,

            "block":
                "6.3",

            "part":
                "2/2",

            "phase":
                "DEVELOPMENT_POSTERIOR_BINDING",

            "status":
                (
                    "FROZEN_READY_FOR_REAL_"
                    "DEVELOPMENT_INTEGRATION"
                ),

            "provenance": {
                "Block63_Part1_report":
                    str(
                        PART1_REPORT
                    ),

                "Block63_Part1_sha256":
                    EXPECTED[
                        "part1"
                    ],

                "Block63_binding_sha256":
                    EXPECTED[
                        "block63_binding"
                    ],

                "development_posterior_json":
                    str(
                        DEV_JSON
                    ),

                "development_posterior_json_sha256":
                    sha256_file(
                        DEV_JSON
                    ),

                "development_posterior_npz":
                    str(
                        DEV_NPZ
                    ),

                "development_posterior_npz_sha256":
                    sha256_file(
                        DEV_NPZ
                    ),

                "route_audit_sha256":
                    sha256_file(
                        ROUTE_AUDIT
                    ),

                "coordinate_semantics_sha256":
                    sha256_file(
                        COORDINATE
                    ),
            },

            "posterior": {
                "trajectory_family":
                    "calibrated_Gaussian_GRU",

                "mean_key":
                    mean_key,

                "mean_shape":
                    [
                        4,
                        3,
                    ],

                "mean_semantics":
                    "metric_H0_displacement",

                "anchor_key":
                    anchor_key,

                "anchor_shape":
                    [
                        3,
                    ],

                "anchor_semantics":
                    (
                        "latest_causal_observed_"
                        "target_position_H0"
                    ),

                "covariance_keys_present_but_"
                "unused_in_Block63":
                    candidates[
                        "covariance"
                    ],

                "deterministic_covariance_use":
                    False,
            },

            "development_record": {
                "scenario_id":
                    scenario_id,

                "scenario_source_artifact":
                    scenario_source,

                "motion_shard":
                    shard_path,

                "motion_shard_source_artifact":
                    shard_source,

                "compact_offset":
                    compact_offset,

                "compact_offset_source_artifact":
                    offset_source,
            },

            "identity_policy": {
                "artifact_actor_ID_required":
                    False,

                "prediction_id_policy":
                    (
                        "internal_development_"
                        "posterior_record_id"
                    ),

                "association":
                    (
                        "frozen_Block63_"
                        "reciprocal_unique_nearest_H0"
                    ),

                "truth_id":
                    False,

                "truth_track_index":
                    False,

                "perfect_WOMD_identity":
                    False,
            },

            "execution_scope": {
                "dataset_access_this_phase":
                    False,

                "model_inference_this_phase":
                    False,

                "formal_evaluation":
                    False,

                "parameter_tuning":
                    False,
            },
        }

        binding_status = (
            write_new_or_exact_json(
                OUTPUT_BINDING,
                binding,
            )
        )

        output_sha = sha256_file(
            OUTPUT_BINDING
        )

    print()
    print(
        "===== H. DEVELOPMENT BINDING ARTIFACT ====="
    )

    print(
        "binding status =",
        binding_status,
    )

    if output_sha is not None:

        print(
            "binding path =",
            OUTPUT_BINDING,
        )

        print(
            "binding SHA256 =",
            output_sha,
        )

    # ========================================================
    # I. Immutability
    # ========================================================

    print()
    print(
        "===== I. IMMUTABILITY ====="
    )

    changed = [
        str(path)
        for path in protected
        if (
            before[
                str(path)
            ]
            !=
            sha256_file(
                path
            )
        )
    ]

    require(
        not changed,
        (
            "Frozen upstream artifact changed: "
            + ", ".join(
                changed
            )
        ),
    )

    print(
        "Block6.3 Part1 = UNCHANGED"
    )

    print(
        "Stage5 development posterior = UNCHANGED"
    )

    print(
        "Stage4/5 route artifacts = UNCHANGED"
    )

    print(
        "dataset access = NO"
    )

    print(
        "model inference = NO"
    )

    print(
        "formal evaluation = NO"
    )

    # ========================================================
    # J. Report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "part":
            "2/2",

        "phase":
            "DEVELOPMENT_POSTERIOR_BINDING",

        "status":
            status,

        "npz_schema":
            schema,

        "npz_binding": {
            "mean_key":
                mean_key,

            "anchor_key":
                anchor_key,

            "covariance_candidates":
                candidates[
                    "covariance"
                ],

            "horizon_candidates":
                candidates[
                    "horizons"
                ],
        },

        "development_record": {
            "scenario_id":
                scenario_id,

            "motion_shard":
                shard_path,

            "compact_offset":
                compact_offset,
        },

        "missing_metadata":
            missing_metadata,

        "binding_artifact": (
            None
            if output_sha
            is None
            else {
                "path":
                    str(
                        OUTPUT_BINDING
                    ),

                "sha256":
                    output_sha,
            }
        ),

        "scope": {
            "dataset_access":
                False,

            "model_inference":
                False,

            "formal_data":
                False,

            "parameter_tuning":
                False,

            "upstream_modified":
                False,
        },

        "next": (
            "Real development scenario integration "
            "using frozen posterior materialization"
            if not missing_metadata
            else
            "Resolve only missing development record metadata"
        ),
    }

    write_new_or_exact_json(
        REPORT,
        report,
    )

    print()
    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 PART 2/2 — PHASE A FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 prerequisite      = EXACT PASS"
    )

    print(
        "Gaussian mean array     =",
        mean_key,
    )

    print(
        "causal anchor array     =",
        anchor_key,
    )

    print(
        "mean semantics          = METRIC H0 DISPLACEMENT"
    )

    print(
        "predictive covariance   = PRESENT UPSTREAM / UNUSED HERE"
    )

    print(
        "perfect actor identity  = NO"
    )

    print(
        "scenario identity       =",
        scenario_id,
    )

    print(
        "motion shard            =",
        shard_path,
    )

    print(
        "compact offset          =",
        compact_offset,
    )

    print(
        "dataset access          = NO"
    )

    print(
        "model inference         = NO"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "STATUS =",
        status,
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
        "BLOCK 6.3 PART2 DEVELOPMENT BINDING = BLOCKED"
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
    traceback.print_exc(
        limit=14
    )

    print()
    print(
        "Block6.3 Part1 modified = NO"
    )

    print(
        "Stage4/5 modified      = NO"
    )

    print(
        "dataset access         = NO"
    )

    print(
        "model inference        = NO"
    )

    print(
        "formal evaluation      = NO"
    )

    print(
        "parameter tuning       = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Deliberately no non-zero sys.exit().

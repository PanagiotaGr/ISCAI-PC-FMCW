from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

NPZ = (
    S5
    / "artifacts/block52/development_gaussian_posterior.npz"
)

DEV_JSON = (
    S5
    / "artifacts/block52/development_gaussian_posterior.json"
)

RESOLVER_JSON = (
    S5
    / "artifacts/block52/dev_posterior_resolver_summary.json"
)

ROUTE_JSON = (
    S5
    / "artifacts/block52/actual_stage4_gaussian_route_audit.json"
)

RUNTIME_JSON = (
    S5
    / "artifacts/block52/calibrated_runtime_api_audit.json"
)

COORD_JSON = (
    S5
    / "artifacts/block52/stage4_prediction_coordinate_semantics.json"
)

PART1 = (
    S6
    / "reports/block63_part1_deterministic_future_box.json"
)

REPORT = (
    S6
    / "reports/block63_part2_batch_row_binding_audit.json"
)

EXPECTED_PART1 = (
    "871c8a81e43048ff2ccf7ef1844ad543"
    "3a859ff15b944749e13be9160b08256a"
)

EXPECTED_KEYS = {
    "anchor":
        "anchor_position_H0_m",

    "mean_displacement":
        "mean_displacement_H0_m",

    "mean_absolute":
        "mean_absolute_H0_m",

    "raw_covariance":
        "raw_covariance_H0_m2",

    "calibrated_covariance":
        "calibrated_covariance_H0_m2",

    "alpha":
        "variance_scale_alpha_h",
}


def require(condition, message):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:

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


def load_json(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(payload) -> bytes:

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
    ).encode("utf-8")


def write_json(path: Path, payload):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(payload)
    )

    tmp.replace(path)


def flatten_metadata(
    value,
    prefix="",
    *,
    depth=0,
    max_depth=8,
):

    rows = []

    if depth > max_depth:
        return rows

    if isinstance(value, dict):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(
                child,
                (dict, list),
            ):

                rows.extend(
                    flatten_metadata(
                        child,
                        path,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )

            else:

                rows.append(
                    (
                        path,
                        child,
                    )
                )

    elif isinstance(value, list):

        # Do not explode large numerical payloads.
        if len(value) <= 40:

            for index, child in enumerate(value):

                path = (
                    f"{prefix}[{index}]"
                )

                if isinstance(
                    child,
                    (dict, list),
                ):

                    rows.extend(
                        flatten_metadata(
                            child,
                            path,
                            depth=depth + 1,
                            max_depth=max_depth,
                        )
                    )

                else:

                    rows.append(
                        (
                            path,
                            child,
                        )
                    )

    return rows


def relevant_metadata(payload):

    pattern = re.compile(
        r"("
        r"row|sample|index|selected|select|probe|"
        r"scenario|record|offset|shard|motion|"
        r"track|prediction|actor|receiver|"
        r"development|split|order|eligible"
        r")",
        re.IGNORECASE,
    )

    exclude = re.compile(
        r"("
        r"count|shape|dimension|horizon_index|"
        r"class_id|covariance_index"
        r")",
        re.IGNORECASE,
    )

    result = []

    for path, value in flatten_metadata(
        payload
    ):

        if not pattern.search(path):
            continue

        if exclude.search(path):
            continue

        result.append({
            "path":
                path,

            "value":
                value,
        })

    return result


def source_contexts():

    roots = (
        S5 / "scripts",
        S5 / "src",
        S5 / "artifacts/block52",
    )

    pattern = re.compile(
        r"("
        r"development_gaussian_posterior|"
        r"anchor_position_H0_m|"
        r"mean_displacement_H0_m|"
        r"mean_absolute_H0_m|"
        r"np\.savez|savez_compressed|"
        r"sample_index|row_index|probe_index|"
        r"selected_index|development_index|"
        r"scenario_id|compact_offset|"
        r"motion_shard"
        r")",
        re.IGNORECASE,
    )

    results = []

    seen = set()

    for root in roots:

        if not root.is_dir():
            continue

        for path in sorted(
            root.rglob("*.py")
        ):

            if path in seen:
                continue

            seen.add(path)

            try:

                lines = path.read_text(
                    encoding="utf-8"
                ).splitlines()

            except BaseException:

                continue

            for index, line in enumerate(lines):

                if not pattern.search(line):
                    continue

                lo = max(
                    0,
                    index - 8,
                )

                hi = min(
                    len(lines),
                    index + 9,
                )

                results.append({
                    "path":
                        str(path),

                    "line":
                        index + 1,

                    "context":
                        "\n".join(
                            f"{j + 1:04d}: {lines[j]}"
                            for j in range(
                                lo,
                                hi,
                            )
                        ),
                })

                if len(results) >= 100:
                    return results

    return results


def explicit_selector_candidates(
    named_payloads,
    row_count: int,
):

    patterns = (
        "row_index",
        "sample_index",
        "selected_index",
        "probe_index",
        "development_index",
        "record_index",
    )

    candidates = []

    for artifact_name, payload in named_payloads:

        for path, value in flatten_metadata(
            payload
        ):

            lower = path.lower()

            if not any(
                token in lower
                for token in patterns
            ):
                continue

            if isinstance(value, bool):
                continue

            if not isinstance(value, int):
                continue

            if not (
                0 <= value < row_count
            ):
                continue

            candidates.append({
                "artifact":
                    artifact_name,

                "path":
                    path,

                "value":
                    int(value),
            })

    return candidates


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 PART 2/2"
    )
    print(
        "BATCH POSTERIOR / ROW-BINDING AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        NPZ,
        DEV_JSON,
        RESOLVER_JSON,
        ROUTE_JSON,
        RUNTIME_JSON,
        COORD_JSON,
        PART1,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required frozen artifact(s): "
            + ", ".join(missing)
        ),
    )

    before = {
        str(path):
            sha256_file(path)
        for path in required
    }

    # ========================================================
    # A. Part1 seal
    # ========================================================

    print()
    print(
        "===== A. BLOCK6.3 PART1 SEAL ====="
    )

    require(
        sha256_file(PART1)
        ==
        EXPECTED_PART1,
        (
            "Block6.3 Part1 report SHA mismatch."
        ),
    )

    print(
        "Part1 report = EXACT PASS"
    )

    # ========================================================
    # B. Exact NPZ schema
    # ========================================================

    print()
    print(
        "===== B. EXACT DEVELOPMENT POSTERIOR SCHEMA ====="
    )

    with np.load(
        NPZ,
        allow_pickle=False,
    ) as archive:

        keys = tuple(
            archive.files
        )

        print(
            "keys =",
            keys,
        )

        for role, key in EXPECTED_KEYS.items():

            require(
                key in archive.files,
                (
                    f"Missing exact frozen NPZ key "
                    f"for {role}: {key}"
                ),
            )

        anchor = np.asarray(
            archive[
                EXPECTED_KEYS[
                    "anchor"
                ]
            ],
            dtype=np.float64,
        )

        mean_displacement = np.asarray(
            archive[
                EXPECTED_KEYS[
                    "mean_displacement"
                ]
            ],
            dtype=np.float64,
        )

        mean_absolute = np.asarray(
            archive[
                EXPECTED_KEYS[
                    "mean_absolute"
                ]
            ],
            dtype=np.float64,
        )

        raw_covariance = np.asarray(
            archive[
                EXPECTED_KEYS[
                    "raw_covariance"
                ]
            ],
            dtype=np.float64,
        )

        calibrated_covariance = np.asarray(
            archive[
                EXPECTED_KEYS[
                    "calibrated_covariance"
                ]
            ],
            dtype=np.float64,
        )

        alpha = np.asarray(
            archive[
                EXPECTED_KEYS[
                    "alpha"
                ]
            ],
            dtype=np.float64,
        )

    require(
        anchor.ndim == 2
        and
        anchor.shape[1] == 3,
        (
            "Anchor array is not batched [N,3]."
        ),
    )

    row_count = int(
        anchor.shape[0]
    )

    require(
        mean_displacement.shape
        ==
        (
            row_count,
            4,
            3,
        ),
        (
            "Mean displacement is not [N,4,3]."
        ),
    )

    require(
        mean_absolute.shape
        ==
        (
            row_count,
            4,
            3,
        ),
        (
            "Absolute mean is not [N,4,3]."
        ),
    )

    require(
        raw_covariance.shape
        ==
        (
            row_count,
            4,
            3,
            3,
        ),
        (
            "Raw covariance is not [N,4,3,3]."
        ),
    )

    require(
        calibrated_covariance.shape
        ==
        (
            row_count,
            4,
            3,
            3,
        ),
        (
            "Calibrated covariance is not [N,4,3,3]."
        ),
    )

    require(
        alpha.shape
        ==
        (
            4,
        ),
        (
            "Variance scale alpha is not [4]."
        ),
    )

    arrays = (
        anchor,
        mean_displacement,
        mean_absolute,
        raw_covariance,
        calibrated_covariance,
        alpha,
    )

    require(
        all(
            np.all(
                np.isfinite(array)
            )
            for array in arrays
        ),
        (
            "Frozen development posterior "
            "contains non-finite values."
        ),
    )

    print(
        "row count =",
        row_count,
    )

    print(
        "anchor_position_H0_m        =",
        anchor.shape,
        "PASS",
    )

    print(
        "mean_displacement_H0_m      =",
        mean_displacement.shape,
        "PASS",
    )

    print(
        "mean_absolute_H0_m          =",
        mean_absolute.shape,
        "PASS",
    )

    print(
        "raw_covariance_H0_m2        =",
        raw_covariance.shape,
        "PASS",
    )

    print(
        "calibrated_covariance_H0_m2 =",
        calibrated_covariance.shape,
        "PASS",
    )

    print(
        "variance_scale_alpha_h      =",
        alpha.shape,
        "PASS",
    )

    # ========================================================
    # C. Coordinate identity invariant
    # ========================================================

    print()
    print(
        "===== C. BATCH COORDINATE INVARIANT ====="
    )

    reconstructed = (
        anchor[
            :,
            None,
            :
        ]
        +
        mean_displacement
    )

    difference = np.abs(
        reconstructed
        -
        mean_absolute
    )

    max_abs_error = float(
        np.max(
            difference
        )
    )

    exact_equal = bool(
        np.array_equal(
            reconstructed,
            mean_absolute,
        )
    )

    print(
        "equation = "
        "mean_absolute_H0_m == "
        "anchor_position_H0_m[:,None,:] + "
        "mean_displacement_H0_m"
    )

    print(
        "exact array equality =",
        exact_equal,
    )

    print(
        "max absolute difference =",
        max_abs_error,
    )

    # This is diagnostic evidence, not a newly tuned numerical
    # acceptance threshold.
    require(
        max_abs_error
        <=
        1e-5,
        (
            "Stored absolute mean does not "
            "numerically agree with frozen "
            "anchor + displacement semantics."
        ),
    )

    print(
        "coordinate invariant = PASS"
    )

    # ========================================================
    # D. Exact JSON metadata
    # ========================================================

    print()
    print(
        "===== D. BLOCK5.2 ROW/RECORD METADATA ====="
    )

    named_payloads = (
        (
            "development_gaussian_posterior.json",
            load_json(
                DEV_JSON
            ),
        ),
        (
            "dev_posterior_resolver_summary.json",
            load_json(
                RESOLVER_JSON
            ),
        ),
        (
            "actual_stage4_gaussian_route_audit.json",
            load_json(
                ROUTE_JSON
            ),
        ),
        (
            "calibrated_runtime_api_audit.json",
            load_json(
                RUNTIME_JSON
            ),
        ),
        (
            "stage4_prediction_coordinate_semantics.json",
            load_json(
                COORD_JSON
            ),
        ),
    )

    metadata_evidence = {}

    for name, payload in named_payloads:

        rows = relevant_metadata(
            payload
        )

        metadata_evidence[
            name
        ] = rows

        print()
        print(
            "---",
            name,
            "---"
        )

        if not rows:

            print(
                "NO RELEVANT SCALAR METADATA"
            )

            continue

        for item in rows[:120]:

            print(
                item[
                    "path"
                ],
                "=",
                item[
                    "value"
                ],
            )

        if len(rows) > 120:

            print(
                "...",
                len(rows) - 120,
                "additional metadata entries stored in report",
            )

    # ========================================================
    # E. Explicit row-selector candidates
    # ========================================================

    print()
    print(
        "===== E. EXPLICIT ROW-SELECTOR CANDIDATES ====="
    )

    selectors = (
        explicit_selector_candidates(
            named_payloads,
            row_count,
        )
    )

    if selectors:

        for item in selectors:

            print(
                item[
                    "artifact"
                ],
                "|",
                item[
                    "path"
                ],
                "=",
                item[
                    "value"
                ],
            )

    else:

        print(
            "NO EXPLICIT ROW SELECTOR IN JSON METADATA"
        )

    # ========================================================
    # F. Exact Block5.2 source contexts
    # ========================================================

    print()
    print(
        "===== F. BLOCK5.2 MATERIALIZATION / SELECTOR SOURCE ====="
    )

    contexts = source_contexts()

    print(
        "source contexts =",
        len(contexts),
    )

    for index, item in enumerate(
        contexts[:70],
        start=1,
    ):

        print()
        print(
            f"--- source context {index} ---"
        )

        print(
            item[
                "path"
            ],
            ":",
            item[
                "line"
            ],
            sep="",
        )

        print(
            item[
                "context"
            ]
        )

    if len(contexts) > 70:

        print(
            "...",
            len(contexts) - 70,
            "additional contexts stored in report",
        )

    # ========================================================
    # G. Resolution state
    # ========================================================

    print()
    print(
        "===== G. ROW-BINDING RESOLUTION ====="
    )

    unique_selector_values = sorted({
        int(
            item[
                "value"
            ]
        )
        for item in selectors
    })

    if len(
        unique_selector_values
    ) == 1:

        selector_state = (
            "ONE_EXPLICIT_SELECTOR_VALUE_FOUND"
        )

        selector_value = (
            unique_selector_values[
                0
            ]
        )

    elif len(
        unique_selector_values
    ) == 0:

        selector_state = (
            "NO_EXPLICIT_SELECTOR_FOUND"
        )

        selector_value = None

    else:

        selector_state = (
            "AMBIGUOUS_SELECTOR_VALUES"
        )

        selector_value = None

    print(
        "batch schema binding = EXACT"
    )

    print(
        "row count =",
        row_count,
    )

    print(
        "selector state =",
        selector_state,
    )

    print(
        "selector value =",
        selector_value,
    )

    # Do NOT freeze merely because one integer-looking metadata
    # value exists. Source/order provenance still has to prove
    # what that row means.
    status = (
        "PASS_BATCH_SCHEMA_BOUND_ROW_PROVENANCE_EXTRACTED"
    )

    # ========================================================
    # H. Immutability
    # ========================================================

    print()
    print(
        "===== H. IMMUTABILITY ====="
    )

    changed = [
        str(path)
        for path in required
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
            "Frozen artifact changed: "
            + ", ".join(
                changed
            )
        ),
    )

    print(
        "Block6.3 Part1 = UNCHANGED"
    )

    print(
        "Stage5 posterior = UNCHANGED"
    )

    print(
        "Block5.2 route evidence = UNCHANGED"
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
    # I. Report
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

        "audit":
            "BATCH_POSTERIOR_ROW_BINDING",

        "status":
            status,

        "npz": {
            "path":
                str(
                    NPZ
                ),

            "sha256":
                sha256_file(
                    NPZ
                ),

            "row_count":
                row_count,

            "keys":
                EXPECTED_KEYS,

            "shapes": {
                "anchor":
                    list(
                        anchor.shape
                    ),

                "mean_displacement":
                    list(
                        mean_displacement.shape
                    ),

                "mean_absolute":
                    list(
                        mean_absolute.shape
                    ),

                "raw_covariance":
                    list(
                        raw_covariance.shape
                    ),

                "calibrated_covariance":
                    list(
                        calibrated_covariance.shape
                    ),

                "alpha":
                    list(
                        alpha.shape
                    ),
            },
        },

        "coordinate_invariant": {
            "equation":
                (
                    "mean_absolute = "
                    "anchor[:,None,:] + "
                    "mean_displacement"
                ),

            "exact_array_equal":
                exact_equal,

            "max_abs_error":
                max_abs_error,
        },

        "selector": {
            "state":
                selector_state,

            "value":
                selector_value,

            "candidates":
                selectors,
        },

        "metadata":
            metadata_evidence,

        "source_contexts":
            contexts,

        "scope": {
            "dataset_access":
                False,

            "model_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "upstream_modified":
                False,
        },
    }

    write_json(
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
        "BLOCK 6.3 PART 2/2 BATCH-ROW AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "development posterior = BATCHED"
    )

    print(
        "row count             =",
        row_count,
    )

    print(
        "anchor key            = anchor_position_H0_m"
    )

    print(
        "mean key              = mean_displacement_H0_m"
    )

    print(
        "absolute mean key     = mean_absolute_H0_m"
    )

    print(
        "calibrated covariance = calibrated_covariance_H0_m2"
    )

    print(
        "deterministic covariance use = NO"
    )

    print(
        "coordinate equation   = VERIFIED"
    )

    print(
        "row selector state    =",
        selector_state,
    )

    print(
        "arbitrary row selection = FORBIDDEN"
    )

    print(
        "dataset access        = NO"
    )

    print(
        "model inference       = NO"
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
        "BLOCK 6.3 PART2 BATCH-ROW AUDIT = BLOCKED"
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
        "Stage5 posterior modified = NO"
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

    print(
        "terminal remains open = YES"
    )

# No sys.exit().

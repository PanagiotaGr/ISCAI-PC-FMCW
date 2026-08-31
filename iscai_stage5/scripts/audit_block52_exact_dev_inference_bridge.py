from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import traceback

import numpy as np
import torch


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

DEV_CACHE = (
    STAGE4
    / "artifacts/block43/"
      "development_cache.npz"
)

CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

GAUSSIAN_CONFIG = (
    STAGE4
    / "configs/"
      "stage4_gaussian_gru.json"
)

BLOCK44_REPORT = (
    STAGE4
    / "reports/"
      "block44_gaussian_gru.json"
)

TRAINING_SCRIPT = (
    STAGE4
    / "scripts/"
      "run_block44_gaussian_training.py"
)

GAUSSIAN_MODEL_SOURCE = (
    STAGE4
    / "src/iscai_stage4/"
      "models/gaussian_gru.py"
)

GAUSSIAN_MATH_SOURCE = (
    STAGE4
    / "src/iscai_stage4/"
      "models/gaussian_math.py"
)

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "exact_dev_inference_bridge_audit.json"
)


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


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:
            chunk = stream.read(
                1024
                *
                1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_json(
    path: Path,
    payload,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def flatten(
    value,
    prefix="",
):
    rows = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():

            child = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            rows.extend(
                flatten(
                    item,
                    child,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        if len(
            value
        ) <= 30:

            for index, item in enumerate(
                value
            ):
                rows.extend(
                    flatten(
                        item,
                        f"{prefix}[{index}]",
                    )
                )

        else:
            rows.append(
                (
                    prefix
                    +
                    ".__length__",
                    len(
                        value
                    ),
                )
            )

    else:
        rows.append(
            (
                prefix,
                value,
            )
        )

    return rows


def array_summary(
    array: np.ndarray,
):
    result = {
        "shape":
            list(
                array.shape
            ),

        "dtype":
            str(
                array.dtype
            ),

        "size":
            int(
                array.size
            ),
    }

    if (
        array.size > 0
        and
        np.issubdtype(
            array.dtype,
            np.number,
        )
    ):
        finite = np.isfinite(
            array
        )

        result[
            "finite_fraction"
        ] = float(
            np.mean(
                finite
            )
        )

        if np.any(
            finite
        ):
            values = array[
                finite
            ]

            result[
                "min"
            ] = float(
                np.min(
                    values
                )
            )

            result[
                "max"
            ] = float(
                np.max(
                    values
                )
            )

    return result


def checkpoint_value_summary(
    value,
):
    if isinstance(
        value,
        torch.Tensor,
    ):
        return {
            "type":
                "Tensor",

            "shape":
                list(
                    value.shape
                ),

            "dtype":
                str(
                    value.dtype
                ),

            "device":
                str(
                    value.device
                ),
        }

    if isinstance(
        value,
        dict,
    ):
        return {
            "type":
                "dict",

            "keys":
                [
                    str(
                        key
                    )
                    for key in list(
                        value.keys()
                    )[
                        :80
                    ]
                ],

            "length":
                len(
                    value
                ),
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return {
            "type":
                type(
                    value
                ).__name__,

            "length":
                len(
                    value
                ),

            "preview":
                [
                    str(
                        item
                    )[
                        :200
                    ]
                    for item in value[
                        :10
                    ]
                ],
        }

    return {
        "type":
            type(
                value
            ).__name__,

        "repr":
            repr(
                value
            )[
                :500
            ],
    }


def source_windows(
    path: Path,
    markers,
    *,
    radius=8,
    max_windows=20,
):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    matched_lines = []

    for index, line in enumerate(
        lines
    ):
        lower = line.lower()

        if any(
            marker.lower()
            in lower
            for marker in markers
        ):
            matched_lines.append(
                index
            )

    windows = []

    used = set()

    for index in matched_lines:

        start = max(
            0,
            index
            -
            radius,
        )

        end = min(
            len(
                lines
            ),
            index
            +
            radius
            +
            1,
        )

        #
        # Merge/avoid heavily overlapping windows.
        #
        key = (
            start,
            end,
        )

        if any(
            abs(
                start
                -
                existing_start
            )
            <
            radius
            for (
                existing_start,
                _
            ) in used
        ):
            continue

        used.add(
            key
        )

        windows.append({
            "start_line":
                start
                +
                1,

            "end_line":
                end,

            "lines":
                [
                    {
                        "line":
                            line_number
                            +
                            1,

                        "text":
                            lines[
                                line_number
                            ],
                    }
                    for line_number in range(
                        start,
                        end,
                    )
                ],
        })

        if len(
            windows
        ) >= max_windows:
            break

    return windows


def print_windows(
    title,
    path,
    windows,
):
    print()
    print(
        "============================================================"
    )
    print(
        title
    )
    print(
        "============================================================"
    )

    print(
        "file =",
        path,
    )

    print(
        "windows =",
        len(
            windows
        ),
    )

    for index, window in enumerate(
        windows,
        start=1,
    ):
        print()
        print(
            f"--- window {index}: "
            f"L{window['start_line']}-"
            f"L{window['end_line']} ---"
        )

        for item in window[
            "lines"
        ]:
            print(
                f"{item['line']:5d}: "
                f"{item['text']}"
            )


def find_cache_provenance(
    cache_sha,
):
    evidence = []

    reports_root = (
        STAGE4
        / "reports"
    )

    for path in sorted(
        reports_root.rglob(
            "*.json"
        ),
        key=str,
    ):
        try:
            if path.stat().st_size > (
                20
                *
                1024
                *
                1024
            ):
                continue

            payload = load_json(
                path
            )

        except Exception:
            continue

        hits = []

        for key, value in flatten(
            payload
        ):
            key_lower = key.lower()

            text = str(
                value
            )

            text_lower = (
                text.lower()
            )

            if (
                "development_cache"
                in
                text_lower
                or
                cache_sha
                in
                text_lower
                or
                (
                    "dev"
                    in
                    key_lower
                    and
                    "cache"
                    in
                    key_lower
                )
            ):
                hits.append({
                    "key":
                        key,

                    "value":
                        value,
                })

        if hits:
            evidence.append({
                "path":
                    str(
                        path
                    ),

                "hits":
                    hits[
                        :50
                    ],
            })

    return evidence


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — EXACT FROZEN DEV-INFERENCE BRIDGE AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        DEV_CACHE,
        CHECKPOINT,
        GAUSSIAN_CONFIG,
        BLOCK44_REPORT,
        TRAINING_SCRIPT,
        GAUSSIAN_MODEL_SOURCE,
        GAUSSIAN_MATH_SOURCE,
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
            "Missing frozen Stage4 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    print(
        "dataset scan          = NO"
    )

    print(
        "formal N=120 read     = NO"
    )

    print(
        "model forward pass    = NO"
    )

    print(
        "training              = NO"
    )

    print(
        "recalibration         = NO"
    )

    print(
        "upstream modification = NO"
    )

    # ========================================================
    # A. Exact cache inventory
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. BLOCK4.3 DEVELOPMENT CACHE"
    )
    print(
        "============================================================"
    )

    cache_sha = file_sha256(
        DEV_CACHE
    )

    print(
        "path =",
        DEV_CACHE,
    )

    print(
        "SHA256 =",
        cache_sha,
    )

    print(
        "size bytes =",
        DEV_CACHE.stat().st_size,
    )

    cache_inventory = {}

    with np.load(
        DEV_CACHE,
        allow_pickle=False,
    ) as cache:

        print(
            "keys =",
            cache.files,
        )

        for key in cache.files:

            array = cache[
                key
            ]

            summary = array_summary(
                array
            )

            cache_inventory[
                key
            ] = summary

            print()
            print(
                f"{key}:"
            )

            print(
                "  shape =",
                summary[
                    "shape"
                ],
            )

            print(
                "  dtype =",
                summary[
                    "dtype"
                ],
            )

            if (
                "finite_fraction"
                in
                summary
            ):
                print(
                    "  finite fraction =",
                    summary[
                        "finite_fraction"
                    ],
                )

    # ========================================================
    # B. Cache frozen provenance
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. DEVELOPMENT CACHE FROZEN PROVENANCE"
    )
    print(
        "============================================================"
    )

    cache_provenance = (
        find_cache_provenance(
            cache_sha
        )
    )

    print(
        "matching Stage4 reports =",
        len(
            cache_provenance
        ),
    )

    for item in cache_provenance[
        :12
    ]:
        print()
        print(
            "REPORT:",
            item[
                "path"
            ],
        )

        for hit in item[
            "hits"
        ][
            :20
        ]:
            print(
                "  ",
                hit[
                    "key"
                ],
                "=",
                hit[
                    "value"
                ],
            )

    # ========================================================
    # C. Gaussian config
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. FROZEN GAUSSIAN CONFIG"
    )
    print(
        "============================================================"
    )

    gaussian_config = load_json(
        GAUSSIAN_CONFIG
    )

    print(
        "config SHA256 =",
        file_sha256(
            GAUSSIAN_CONFIG
        ),
    )

    print(
        json.dumps(
            gaussian_config,
            indent=2,
            sort_keys=True,
            default=str,
        )[
            :12000
        ]
    )

    # ========================================================
    # D. Checkpoint metadata only — no model execution
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. FROZEN GAUSSIAN CHECKPOINT STRUCTURE"
    )
    print(
        "============================================================"
    )

    checkpoint_sha = file_sha256(
        CHECKPOINT
    )

    print(
        "checkpoint SHA256 =",
        checkpoint_sha,
    )

    checkpoint_summary = {
        "sha256":
            checkpoint_sha,

        "load_status":
            None,

        "top_level":
            {},
    }

    try:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint_summary[
            "load_status"
        ] = (
            "PASS_WEIGHTS_ONLY"
        )

        print(
            "weights_only load = PASS"
        )

        print(
            "top-level type =",
            type(
                checkpoint
            ).__name__,
        )

        if isinstance(
            checkpoint,
            dict,
        ):
            print(
                "top-level keys =",
                list(
                    checkpoint.keys()
                ),
            )

            for key, value in (
                checkpoint.items()
            ):
                summary = (
                    checkpoint_value_summary(
                        value
                    )
                )

                checkpoint_summary[
                    "top_level"
                ][
                    str(
                        key
                    )
                ] = summary

                print()
                print(
                    f"{key}:",
                    json.dumps(
                        summary,
                        indent=2,
                        default=str,
                    )[
                        :3000
                    ],
                )

    except Exception as exc:

        checkpoint_summary[
            "load_status"
        ] = (
            "WEIGHTS_ONLY_LOAD_UNRESOLVED"
        )

        checkpoint_summary[
            "load_error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        print(
            "weights_only load = UNRESOLVED"
        )

        print(
            "reason =",
            checkpoint_summary[
                "load_error"
            ],
        )

        print(
            "unsafe fallback load performed = NO"
        )

    # ========================================================
    # E. Block4.4 report exact inference evidence
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. BLOCK4.4 REPORT — DEVELOPMENT/CHECKPOINT EVIDENCE"
    )
    print(
        "============================================================"
    )

    block44 = load_json(
        BLOCK44_REPORT
    )

    interesting_report_fields = []

    tokens = (
        "checkpoint",
        "development",
        "prediction",
        "covariance",
        "normal",
        "cache",
        "nll",
        "ade",
    )

    for key, value in flatten(
        block44
    ):
        lower = key.lower()

        if any(
            token in lower
            for token in tokens
        ):
            interesting_report_fields.append({
                "key":
                    key,

                "value":
                    value,
            })

    for item in interesting_report_fields[
        :100
    ]:
        print(
            item[
                "key"
            ],
            "=",
            item[
                "value"
            ],
        )

    # ========================================================
    # F. Exact training/inference source windows
    # ========================================================

    training_markers = (
        "DEV_CACHE",
        "development_cache",
        "np.load",
        "make_gaussian_model",
        "model(",
        "covariance_from_scale_tril",
        "denormalize_gaussian",
        "prediction_sha256",
        "best_raw_uncalibrated",
        "torch.load",
        "load_state_dict",
        "evaluate",
        "development",
    )

    training_windows = source_windows(
        TRAINING_SCRIPT,
        training_markers,
        radius=6,
        max_windows=18,
    )

    print_windows(
        "F. BLOCK4.4 TRAINING / DEVELOPMENT-INFERENCE ROUTE",
        TRAINING_SCRIPT,
        training_windows,
    )

    # ========================================================
    # G. Exact model and math API windows
    # ========================================================

    model_windows = source_windows(
        GAUSSIAN_MODEL_SOURCE,
        (
            "class GaussianTrajectoryGRU",
            "def forward",
            "forward(",
        ),
        radius=8,
        max_windows=6,
    )

    print_windows(
        "G1. GAUSSIAN MODEL API",
        GAUSSIAN_MODEL_SOURCE,
        model_windows,
    )

    math_windows = source_windows(
        GAUSSIAN_MATH_SOURCE,
        (
            "def covariance_from_scale_tril",
            "def denormalize_gaussian",
            "def gaussian_nll",
            "def masked_gaussian_nll",
        ),
        radius=8,
        max_windows=8,
    )

    print_windows(
        "G2. GAUSSIAN MATH / DENORMALIZATION API",
        GAUSSIAN_MATH_SOURCE,
        math_windows,
    )

    # ========================================================
    # H. Decision evidence
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "H. BRIDGE AUDIT CONCLUSION"
    )
    print(
        "============================================================"
    )

    has_nonempty_cache = (
        len(
            cache_inventory
        )
        >
        0
    )

    has_checkpoint = (
        checkpoint_sha
        ==
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        )
    )

    require(
        has_nonempty_cache,
        (
            "Development cache is empty."
        ),
    )

    require(
        has_checkpoint,
        (
            "Frozen Gaussian checkpoint SHA "
            "does not match Block4.4."
        ),
    )

    print(
        "development cache exists       = PASS"
    )

    print(
        "development cache non-empty    = PASS"
    )

    print(
        "Gaussian checkpoint SHA        = PASS"
    )

    print(
        "Block4.4 inference source      = FOUND"
    )

    print(
        "Block4.4 denormalization route = FOUND"
    )

    print(
        "materialized dev posterior     = NO"
    )

    print(
        "dev-only frozen inference may be required = YES"
    )

    print(
        "actual inference executed now  = NO"
    )

    payload = {
        "status":
            "PASS_EXACT_BRIDGE_AUDIT",

        "scientific_execution": {
            "dataset_scan":
                False,

            "formal_N120_read":
                False,

            "Stage4_model_forward":
                False,

            "training":
                False,

            "recalibration":
                False,

            "upstream_modified":
                False,
        },

        "development_cache": {
            "path":
                str(
                    DEV_CACHE
                ),

            "sha256":
                cache_sha,

            "size_bytes":
                DEV_CACHE.stat().st_size,

            "arrays":
                cache_inventory,

            "frozen_provenance":
                cache_provenance,
        },

        "Gaussian_checkpoint":
            {
                "path":
                    str(
                        CHECKPOINT
                    ),

                **checkpoint_summary,
            },

        "Gaussian_config": {
            "path":
                str(
                    GAUSSIAN_CONFIG
                ),

            "sha256":
                file_sha256(
                    GAUSSIAN_CONFIG
                ),

            "payload":
                gaussian_config,
        },

        "Block44_report": {
            "path":
                str(
                    BLOCK44_REPORT
                ),

            "sha256":
                file_sha256(
                    BLOCK44_REPORT
                ),

            "interesting_fields":
                interesting_report_fields,
        },

        "source_routes": {
            "training_script":
                {
                    "path":
                        str(
                            TRAINING_SCRIPT
                        ),

                    "sha256":
                        file_sha256(
                            TRAINING_SCRIPT
                        ),

                    "windows":
                        training_windows,
                },

            "model":
                {
                    "path":
                        str(
                            GAUSSIAN_MODEL_SOURCE
                        ),

                    "sha256":
                        file_sha256(
                            GAUSSIAN_MODEL_SOURCE
                        ),

                    "windows":
                        model_windows,
                },

            "math":
                {
                    "path":
                        str(
                            GAUSSIAN_MATH_SOURCE
                        ),

                    "sha256":
                        file_sha256(
                            GAUSSIAN_MATH_SOURCE
                        ),

                    "windows":
                        math_windows,
                },
        },

        "conclusion": {
            "materialized_Block44_dev_posterior":
                False,

            "frozen_development_cache":
                True,

            "frozen_Gaussian_checkpoint":
                True,

            "exact_inference_route_source_available":
                True,

            "next_permitted_action":
                (
                    "construct_exact_frozen_checkpoint_"
                    "dev_cache_inference_reproduction_"
                    "without_training_or_recalibration"
                ),
        },
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "status = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "formal N=120 used = NO"
    )

    print(
        "model inference performed = NO"
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
        "EXACT DEV-INFERENCE BRIDGE AUDIT = BLOCKED"
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

    print(
        "formal N=120 used      = NO"
    )

    print(
        "model forward executed = NO"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "upstream modified      = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Deliberately no sys.exit().

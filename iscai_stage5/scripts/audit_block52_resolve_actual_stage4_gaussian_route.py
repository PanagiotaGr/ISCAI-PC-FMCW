from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import inspect
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

TRAINING_SCRIPT = (
    STAGE4
    / "scripts/"
      "run_block44_gaussian_training.py"
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

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "actual_stage4_gaussian_route_audit.json"
)


TARGET_SYMBOLS = (
    "GaussianTrajectoryGRU",
    "covariance_from_scale_tril",
    "denormalize_gaussian",
    "gaussian_nll_per_horizon",
    "masked_gaussian_nll",
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
                1024 * 1024
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


def source_window(
    path: Path,
    *,
    center_line: int,
    radius: int = 8,
):
    lines = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    start = max(
        0,
        int(
            center_line
        )
        -
        radius
        -
        1,
    )

    end = min(
        len(
            lines
        ),
        int(
            center_line
        )
        +
        radius,
    )

    return [
        {
            "line":
                index
                +
                1,

            "text":
                lines[
                    index
                ],
        }
        for index in range(
            start,
            end,
        )
    ]


def parse_target_imports():
    text = TRAINING_SCRIPT.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        text
    )

    resolved = {}

    all_import_rows = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            ast.ImportFrom,
        ):
            module = node.module

            for alias in node.names:

                imported = alias.name

                local_name = (
                    alias.asname
                    or
                    alias.name
                )

                row = {
                    "line":
                        int(
                            node.lineno
                        ),

                    "module":
                        module,

                    "imported_symbol":
                        imported,

                    "local_name":
                        local_name,
                }

                all_import_rows.append(
                    row
                )

                if (
                    local_name
                    in
                    TARGET_SYMBOLS
                    or
                    imported
                    in
                    TARGET_SYMBOLS
                ):
                    symbol = (
                        local_name
                        if local_name
                        in
                        TARGET_SYMBOLS
                        else imported
                    )

                    resolved[
                        symbol
                    ] = row

        elif isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:

                all_import_rows.append({
                    "line":
                        int(
                            node.lineno
                        ),

                    "module":
                        alias.name,

                    "imported_symbol":
                        None,

                    "local_name":
                        (
                            alias.asname
                            or
                            alias.name
                        ),
                })

    return (
        resolved,
        all_import_rows,
    )


def runtime_resolve_symbol(
    *,
    module_name,
    imported_symbol,
):
    module = importlib.import_module(
        module_name
    )

    require(
        hasattr(
            module,
            imported_symbol,
        ),
        (
            f"{module_name} does not expose "
            f"{imported_symbol}."
        ),
    )

    obj = getattr(
        module,
        imported_symbol,
    )

    source_file = inspect.getsourcefile(
        obj
    )

    require(
        source_file is not None,
        (
            f"Could not obtain source file for "
            f"{module_name}.{imported_symbol}."
        ),
    )

    source_path = Path(
        source_file
    ).resolve()

    source_lines, starting_line = (
        inspect.getsourcelines(
            obj
        )
    )

    return {
        "module":
            module_name,

        "symbol":
            imported_symbol,

        "object_type":
            type(
                obj
            ).__name__,

        "source_path":
            str(
                source_path
            ),

        "source_sha256":
            file_sha256(
                source_path
            ),

        "definition_start_line":
            int(
                starting_line
            ),

        "signature":
            (
                str(
                    inspect.signature(
                        obj
                    )
                )
                if callable(
                    obj
                )
                else None
            ),

        "source_preview":
            "".join(
                source_lines[
                    :40
                ]
            )[
                :6000
            ],
    }


def inspect_dev_cache():
    inventory = {}

    with np.load(
        DEV_CACHE,
        allow_pickle=False,
    ) as payload:

        for key in payload.files:

            array = payload[
                key
            ]

            item = {
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

                item[
                    "finite_fraction"
                ] = float(
                    np.mean(
                        finite
                    )
                )

            inventory[
                key
            ] = item

    return inventory


def inspect_checkpoint():
    result = {
        "sha256":
            file_sha256(
                CHECKPOINT
            ),

        "load_status":
            None,

        "top_level_keys":
            [],

        "top_level_summary":
            {},
    }

    try:
        payload = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )

    except Exception as exc:

        result[
            "load_status"
        ] = (
            "WEIGHTS_ONLY_LOAD_UNRESOLVED"
        )

        result[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return result

    result[
        "load_status"
    ] = (
        "PASS_WEIGHTS_ONLY"
    )

    if not isinstance(
        payload,
        dict,
    ):
        result[
            "top_level_type"
        ] = type(
            payload
        ).__name__

        return result

    result[
        "top_level_keys"
    ] = [
        str(
            key
        )
        for key in payload.keys()
    ]

    for key, value in payload.items():

        if isinstance(
            value,
            torch.Tensor,
        ):
            summary = {
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
            }

        elif isinstance(
            value,
            dict,
        ):
            summary = {
                "type":
                    "dict",

                "length":
                    len(
                        value
                    ),

                "keys_preview":
                    [
                        str(
                            item
                        )
                        for item in list(
                            value.keys()
                        )[
                            :40
                        ]
                    ],
            }

        elif isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            summary = {
                "type":
                    type(
                        value
                    ).__name__,

                "length":
                    len(
                        value
                    ),
            }

        else:
            summary = {
                "type":
                    type(
                        value
                    ).__name__,

                "repr":
                    repr(
                        value
                    )[
                        :1000
                    ],
            }

        result[
            "top_level_summary"
        ][
            str(
                key
            )
        ] = summary

    return result


def find_training_route_lines():
    text = TRAINING_SCRIPT.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    lines = text.splitlines()

    tokens = (
        "DEV_CACHE",
        "development_cache",
        "np.load",
        "load_state_dict",
        "CHECKPOINT",
        "GaussianTrajectoryGRU",
        "covariance_from_scale_tril",
        "denormalize_gaussian",
        "model(",
        "prediction_sha256",
        "best_raw_uncalibrated",
        "development",
    )

    hits = []

    for index, line in enumerate(
        lines,
        start=1,
    ):
        lower = line.lower()

        if not any(
            token.lower()
            in lower
            for token in tokens
        ):
            continue

        hits.append({
            "line":
                index,

            "text":
                line,
        })

    return hits


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — ACTUAL STAGE4 GAUSSIAN SOURCE-ROUTE AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        TRAINING_SCRIPT,
        DEV_CACHE,
        CHECKPOINT,
        GAUSSIAN_CONFIG,
        BLOCK44_REPORT,
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
            "Missing frozen prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    print(
        "formal N=120 read     = NO"
    )

    print(
        "model forward pass    = NO"
    )

    print(
        "dataset scan          = NO"
    )

    print(
        "training/recalibration= NO"
    )

    print(
        "upstream modification = NO"
    )

    # ========================================================
    # A. Parse actual imports
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. ACTUAL BLOCK4.4 GAUSSIAN IMPORTS"
    )
    print(
        "============================================================"
    )

    (
        target_imports,
        all_import_rows,
    ) = parse_target_imports()

    for symbol in TARGET_SYMBOLS:

        row = target_imports.get(
            symbol
        )

        if row is None:
            print(
                symbol,
                "= NOT DIRECTLY RESOLVED FROM IMPORT AST",
            )

        else:
            print(
                symbol,
                "=",
                (
                    f"from {row['module']} "
                    f"import {row['imported_symbol']}"
                ),
            )

    require(
        "GaussianTrajectoryGRU"
        in
        target_imports,
        (
            "Could not resolve GaussianTrajectoryGRU "
            "import from frozen Block4.4 script."
        ),
    )

    require(
        "covariance_from_scale_tril"
        in
        target_imports,
        (
            "Could not resolve covariance_from_scale_tril "
            "import."
        ),
    )

    require(
        "denormalize_gaussian"
        in
        target_imports,
        (
            "Could not resolve denormalize_gaussian "
            "import."
        ),
    )

    # ========================================================
    # B. Runtime source resolution — imports only, no forward
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. RESOLVED ACTUAL SOURCE FILES"
    )
    print(
        "============================================================"
    )

    symbol_routes = {}

    for symbol in TARGET_SYMBOLS:

        row = target_imports.get(
            symbol
        )

        if row is None:
            continue

        route = runtime_resolve_symbol(
            module_name=(
                row[
                    "module"
                ]
            ),

            imported_symbol=(
                row[
                    "imported_symbol"
                ]
            ),
        )

        symbol_routes[
            symbol
        ] = route

        print()
        print(
            symbol
        )

        print(
            "  module =",
            route[
                "module"
            ],
        )

        print(
            "  source =",
            route[
                "source_path"
            ],
        )

        print(
            "  source SHA256 =",
            route[
                "source_sha256"
            ],
        )

        print(
            "  definition line =",
            route[
                "definition_start_line"
            ],
        )

        print(
            "  signature =",
            route[
                "signature"
            ],
        )

    # ========================================================
    # C. Development cache schema
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. FROZEN BLOCK4.3 DEVELOPMENT CACHE"
    )
    print(
        "============================================================"
    )

    cache_inventory = (
        inspect_dev_cache()
    )

    print(
        "path =",
        DEV_CACHE,
    )

    print(
        "SHA256 =",
        file_sha256(
            DEV_CACHE
        ),
    )

    print(
        "keys =",
        list(
            cache_inventory.keys()
        ),
    )

    for key, item in (
        cache_inventory.items()
    ):
        print(
            f"{key:30s}",
            "shape =",
            item[
                "shape"
            ],
            "| dtype =",
            item[
                "dtype"
            ],
        )

    # ========================================================
    # D. Checkpoint structure
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. FROZEN BLOCK4.4 CHECKPOINT"
    )
    print(
        "============================================================"
    )

    checkpoint = (
        inspect_checkpoint()
    )

    print(
        "SHA256 =",
        checkpoint[
            "sha256"
        ],
    )

    print(
        "weights_only load =",
        checkpoint[
            "load_status"
        ],
    )

    print(
        "top-level keys =",
        checkpoint.get(
            "top_level_keys"
        ),
    )

    for key, value in (
        checkpoint.get(
            "top_level_summary",
            {}
        ).items()
    ):
        print(
            key,
            "=",
            json.dumps(
                value,
                default=str,
            )[
                :2000
            ],
        )

    require(
        checkpoint[
            "sha256"
        ]
        ==
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        ),
        (
            "Frozen Gaussian checkpoint "
            "SHA mismatch."
        ),
    )

    # ========================================================
    # E. Exact training/development inference route
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. BLOCK4.4 TRAINING / DEV-INFERENCE ROUTE"
    )
    print(
        "============================================================"
    )

    route_hits = (
        find_training_route_lines()
    )

    for hit in route_hits[
        :120
    ]:
        print(
            f"L{hit['line']:4d}: "
            f"{hit['text']}"
        )

    # ========================================================
    # F. Definition previews
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "F. CRITICAL GAUSSIAN API DEFINITIONS"
    )
    print(
        "============================================================"
    )

    for symbol in (
        "GaussianTrajectoryGRU",
        "covariance_from_scale_tril",
        "denormalize_gaussian",
    ):

        route = symbol_routes[
            symbol
        ]

        print()
        print(
            f"--- {symbol} ---"
        )

        print(
            route[
                "source_preview"
            ]
        )

    # ========================================================
    # G. Conclusion
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "G. SOURCE-ROUTE CONCLUSION"
    )
    print(
        "============================================================"
    )

    require(
        cache_inventory,
        (
            "Frozen development cache "
            "is empty."
        ),
    )

    require(
        "GaussianTrajectoryGRU"
        in
        symbol_routes,
        (
            "Gaussian model source route "
            "not resolved."
        ),
    )

    require(
        "covariance_from_scale_tril"
        in
        symbol_routes,
        (
            "Covariance source route "
            "not resolved."
        ),
    )

    require(
        "denormalize_gaussian"
        in
        symbol_routes,
        (
            "Denormalization source route "
            "not resolved."
        ),
    )

    print(
        "actual Gaussian model source = RESOLVED PASS"
    )

    print(
        "actual covariance API source = RESOLVED PASS"
    )

    print(
        "actual denormalization source= RESOLVED PASS"
    )

    print(
        "development cache schema     = RESOLVED PASS"
    )

    print(
        "checkpoint structure         = RESOLVED PASS"
    )

    print(
        "materialized dev posterior   = NO"
    )

    print(
        "frozen dev inference needed  = YES"
    )

    print(
        "forward pass executed now    = NO"
    )

    payload = {
        "status":
            "PASS_ACTUAL_STAGE4_ROUTE_RESOLVED",

        "scientific_execution": {
            "formal_N120_read":
                False,

            "model_forward":
                False,

            "dataset_scan":
                False,

            "training":
                False,

            "recalibration":
                False,

            "upstream_modified":
                False,
        },

        "training_script": {
            "path":
                str(
                    TRAINING_SCRIPT
                ),

            "sha256":
                file_sha256(
                    TRAINING_SCRIPT
                ),

            "target_imports":
                target_imports,

            "all_imports":
                all_import_rows,

            "route_hits":
                route_hits,
        },

        "resolved_symbols":
            symbol_routes,

        "development_cache": {
            "path":
                str(
                    DEV_CACHE
                ),

            "sha256":
                file_sha256(
                    DEV_CACHE
                ),

            "arrays":
                cache_inventory,
        },

        "checkpoint":
            checkpoint,

        "Gaussian_config": {
            "path":
                str(
                    GAUSSIAN_CONFIG
                ),

            "sha256":
                file_sha256(
                    GAUSSIAN_CONFIG
                ),
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
        },

        "conclusion": {
            "materialized_dev_posterior":
                False,

            "actual_model_source_resolved":
                True,

            "actual_covariance_source_resolved":
                True,

            "actual_denormalization_source_resolved":
                True,

            "frozen_development_cache":
                True,

            "frozen_checkpoint":
                True,

            "next_action":
                (
                    "exact_dev_only_forward_reproduction_"
                    "using_frozen_Block4.4_route"
                ),
        },
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "formal N=120 used = NO"
    )

    print(
        "model forward executed = NO"
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
        "ACTUAL STAGE4 SOURCE-ROUTE AUDIT = BLOCKED"
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

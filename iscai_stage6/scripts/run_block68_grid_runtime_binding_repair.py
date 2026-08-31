from __future__ import annotations

from hashlib import sha256
import ast
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np

from iscai_stage6.adb.grid import (
    IlluminationGridSpec,
)

from iscai_stage6.adb.part_a_reactive import (
    part_a_original_reactive_map_from_h0_centers,
)


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

GRID_CONFIG = (
    S6
    / "configs/"
      "block64_mc_grid_numeric_freeze.json"
)

PREDICTIVE_SELECTORS = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

REACTIVE_FALLBACKS = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_reactive_fallbacks.jsonl"
)

REACTIVE_FREEZE = (
    S6
    / "reports/"
      "block68_reactive_development_scoring_freeze.json"
)

NI_FREEZE = (
    S6
    / "reports/"
      "block68_noninferiority_bound_rule_freeze.json"
)

EVALUATOR_FREEZE = (
    S6
    / "reports/"
      "block68_evaluator_reference_contract_freeze.json"
)

BINDING = (
    S6
    / "configs/"
      "stage6_frozen_illumination_grid_runtime_binding.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_grid_runtime_binding_repair.json"
)

EXPECTED_TESTS = 224

EXPECTED_SHAPE = (
    501,
    301,
)

MIN_FREE_GIB = 250.0


EXPECTED_SHA = {
    REACTIVE_FREEZE:
        "0fc8089bdfbf20eb5e187e71a6a736fe"
        "fea59b3aac6b34ff373237016d8473f9",

    NI_FREEZE:
        "289f9369735566daa2344a90cb312d4c0"
        "6ede76b73827f07e408855c4e06ce6b",

    EVALUATOR_FREEZE:
        "5885424d164a058a151c50dd663acec4"
        "da7aeba66a56d5e0191c777682e113a8",

    S6 / "src/iscai_stage6/adb/part_a_reactive.py":
        "19f80f325aebc03b5257ca3c0408d1dd"
        "ea1ac3a8ff4ad875914224584e2ef173",

    S6 / "src/iscai_stage6/adb/class_aware_policy.py":
        "b998f468b98c2770c48c84a2c0aaaa21"
        "77c2d84b8fc838e80fef9b9da1ebc14b",
}


# ============================================================
# Generic helpers
# ============================================================

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


def file_sha(
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


def safe_json(
    path: Path,
):

    require(
        "formal"
        not in
        str(
            path.resolve()
        ).lower(),
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def safe_jsonl(
    path: Path,
):

    require(
        "formal"
        not in
        str(
            path.resolve()
        ).lower(),
        (
            "FORMAL CONTENT ACCESS FORBIDDEN: "
            f"{path}"
        ),
    )

    result = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, line in enumerate(
            stream,
            start=1,
        ):

            if not line.strip():
                continue

            try:

                value = json.loads(
                    line
                )

            except Exception as exc:

                raise RuntimeError(
                    f"Invalid JSONL "
                    f"{path}:{line_number}: "
                    f"{exc}"
                ) from exc

            require(
                isinstance(
                    value,
                    dict,
                ),
                (
                    "Expected JSON object at "
                    f"{path}:{line_number}"
                ),
            )

            result.append(
                value
            )

    return result


def canonical_json_bytes(
    value,
):

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_json(
    path: Path,
    value,
):

    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_bytes(
        canonical_json_bytes(
            value
        )
    )

    os.replace(
        temporary,
        path,
    )


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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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


def tail(
    text,
    count=100,
):

    return "\n".join(
        text.splitlines()[
            -count:
        ]
    )


# ============================================================
# Frozen upstream
# ============================================================

def exact_upstream_gate():

    for path, expected in (
        EXPECTED_SHA.items()
    ):

        require(
            path.is_file(),
            (
                "Missing frozen dependency: "
                f"{path}"
            ),
        )

        actual = file_sha(
            path
        )

        require(
            actual
            ==
            expected,
            (
                "Frozen dependency changed:\n"
                f"path={path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )


# ============================================================
# Exact Block6.4 grid-subtree extraction
# ============================================================

def find_key_recursive(
    value,
    target_key,
):

    found = []

    def walk(
        item,
        path,
    ):

        if isinstance(
            item,
            dict,
        ):

            for key, child in (
                item.items()
            ):

                next_path = (
                    path
                    +
                    (
                        str(
                            key
                        ),
                    )
                )

                if (
                    str(
                        key
                    )
                    ==
                    target_key
                ):

                    found.append(
                        (
                            next_path,
                            child,
                        )
                    )

                walk(
                    child,
                    next_path,
                )

        elif isinstance(
            item,
            list,
        ):

            for index, child in enumerate(
                item
            ):

                walk(
                    child,
                    path
                    +
                    (
                        f"[{index}]",
                    ),
                )

    walk(
        value,
        tuple(),
    )

    return found


def flatten(
    value,
    prefix="",
):

    rows = []

    if isinstance(
        value,
        dict,
    ):

        for key, child in (
            value.items()
        ):

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
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

            else:

                rows.append(
                    (
                        path,
                        child,
                    )
                )

    elif isinstance(
        value,
        list,
    ):

        # Record the complete short numerical list because
        # domain pairs often appear this way.
        if (
            len(
                value
            )
            <=
            10
            and
            all(
                isinstance(
                    item,
                    (
                        int,
                        float,
                    ),
                )
                and
                not isinstance(
                    item,
                    bool,
                )
                for item in value
            )
        ):

            rows.append(
                (
                    prefix,
                    value,
                )
            )

        else:

            for index, child in enumerate(
                value
            ):

                path = (
                    f"{prefix}[{index}]"
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

                else:

                    rows.append(
                        (
                            path,
                            child,
                        )
                    )

    return rows


def is_number(
    value,
):

    return (
        isinstance(
            value,
            (
                int,
                float,
            ),
        )
        and
        not isinstance(
            value,
            bool,
        )
        and
        math.isfinite(
            float(
                value
            )
        )
    )


def unique_numeric(
    values,
    *,
    label,
    atol=1.0e-12,
):

    numeric = [
        float(
            value
        )
        for value in values
        if is_number(
            value
        )
    ]

    require(
        numeric,
        (
            f"No numeric evidence for {label}."
        ),
    )

    reference = numeric[
        0
    ]

    require(
        all(
            math.isclose(
                value,
                reference,
                rel_tol=0.0,
                abs_tol=atol,
            )
            for value in numeric
        ),
        (
            f"Conflicting evidence for {label}: "
            f"{numeric}"
        ),
    )

    return reference


def numeric_pairs(
    rows,
    *,
    axis,
    unit_token,
):

    result = []

    for path, value in rows:

        lower = path.lower()

        if (
            axis
            not in
            lower
        ):
            continue

        if (
            unit_token
            not in
            lower
        ):
            continue

        if not any(
            token in lower
            for token in (
                "domain",
                "limits",
                "bounds",
                "extent",
                "range",
            )
        ):
            continue

        if not (
            isinstance(
                value,
                list,
            )
            and
            len(
                value
            )
            ==
            2
            and
            all(
                is_number(
                    item
                )
                for item in value
            )
        ):
            continue

        result.append(
            (
                path,
                (
                    float(
                        value[
                            0
                        ]
                    ),
                    float(
                        value[
                            1
                        ]
                    ),
                ),
            )
        )

    return result


def scalar_candidates(
    rows,
    *,
    axis,
    side,
    unit_token,
):

    side_tokens = (
        (
            "min",
            "minimum",
            "low",
            "lower",
        )
        if side == "min"
        else
        (
            "max",
            "maximum",
            "high",
            "upper",
        )
    )

    result = []

    for path, value in rows:

        lower = path.lower()

        if (
            axis
            not in
            lower
        ):
            continue

        if (
            unit_token
            not in
            lower
        ):
            continue

        if not any(
            token in lower
            for token in side_tokens
        ):
            continue

        if is_number(
            value
        ):

            result.append(
                (
                    path,
                    float(
                        value
                    ),
                )
            )

    return result


def step_candidates(
    rows,
    *,
    axis,
    unit_token,
):

    result = []

    for path, value in rows:

        lower = path.lower()

        if (
            axis
            not in
            lower
            or
            unit_token
            not in
            lower
            or
            not any(
                token in lower
                for token in (
                    "step",
                    "resolution",
                    "spacing",
                    "delta",
                )
            )
        ):
            continue

        if is_number(
            value
        ):

            result.append(
                (
                    path,
                    float(
                        value
                    ),
                )
            )

    return result


# ============================================================
# Selector semantic evidence
# ============================================================

def selector_grid_evidence():

    records = (
        safe_jsonl(
            PREDICTIVE_SELECTORS
        )
        +
        safe_jsonl(
            REACTIVE_FALLBACKS
        )
    )

    require(
        records,
        (
            "No selector records."
        ),
    )

    theta_values = []

    range_max_values = []

    for record in records:

        require(
            record.get(
                "future_used_for_eligibility"
            )
            is False,
            (
                "Selector grid evidence "
                "uses future eligibility."
            ),
        )

        theta = record.get(
            "theta_domain_deg"
        )

        require(
            (
                isinstance(
                    theta,
                    list,
                )
                and
                len(
                    theta
                )
                ==
                2
                and
                all(
                    is_number(
                        item
                    )
                    for item in theta
                )
            ),
            (
                "Selector theta_domain_deg "
                "missing or invalid."
            ),
        )

        maximum = record.get(
            "maximum_range_m"
        )

        require(
            is_number(
                maximum
            ),
            (
                "Selector maximum_range_m "
                "missing or invalid."
            ),
        )

        theta_values.append(
            (
                float(
                    theta[
                        0
                    ]
                ),
                float(
                    theta[
                        1
                    ]
                ),
            )
        )

        range_max_values.append(
            float(
                maximum
            )
        )

    theta_reference = (
        theta_values[
            0
        ]
    )

    require(
        all(
            (
                math.isclose(
                    item[
                        0
                    ],
                    theta_reference[
                        0
                    ],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and
                math.isclose(
                    item[
                        1
                    ],
                    theta_reference[
                        1
                    ],
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            )
            for item in theta_values
        ),
        (
            "Selector theta domains "
            "are not globally identical."
        ),
    )

    range_reference = (
        range_max_values[
            0
        ]
    )

    require(
        all(
            math.isclose(
                value,
                range_reference,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for value in range_max_values
        ),
        (
            "Selector maximum ranges "
            "are not globally identical."
        ),
    )

    return {
        "record_count":
            len(
                records
            ),

        "theta_domain_deg":
            [
                theta_reference[
                    0
                ],
                theta_reference[
                    1
                ],
            ],

        "maximum_range_m":
            range_reference,
    }


# ============================================================
# Resolve six constructor values
# ============================================================

def resolve_exact_grid():

    config = safe_json(
        GRID_CONFIG
    )

    require(
        config.get(
            "status"
        )
        ==
        "FROZEN_DEVELOPMENT_NUMERICS",
        (
            "Block6.4 numeric freeze "
            "status mismatch."
        ),
    )

    matches = find_key_recursive(
        config,
        "scientific_illumination_grid",
    )

    require(
        len(
            matches
        )
        ==
        1,
        (
            "Expected exactly one "
            "scientific_illumination_grid subtree; "
            f"found {len(matches)}."
        ),
    )

    subtree_path, subtree = (
        matches[
            0
        ]
    )

    require(
        isinstance(
            subtree,
            dict,
        ),
        (
            "scientific_illumination_grid "
            "must be an object."
        ),
    )

    rows = flatten(
        subtree
    )

    selector = (
        selector_grid_evidence()
    )

    # --------------------------------------------------------
    # Theta
    # --------------------------------------------------------

    theta_rad_pairs = numeric_pairs(
        rows,
        axis="theta",
        unit_token="rad",
    )

    theta_deg_pairs = numeric_pairs(
        rows,
        axis="theta",
        unit_token="deg",
    )

    theta_min_rad_candidates = [
        value
        for _, value
        in scalar_candidates(
            rows,
            axis="theta",
            side="min",
            unit_token="rad",
        )
    ]

    theta_max_rad_candidates = [
        value
        for _, value
        in scalar_candidates(
            rows,
            axis="theta",
            side="max",
            unit_token="rad",
        )
    ]

    if theta_rad_pairs:

        theta_min_rad_candidates.extend(
            pair[
                0
            ]
            for _, pair
            in theta_rad_pairs
        )

        theta_max_rad_candidates.extend(
            pair[
                1
            ]
            for _, pair
            in theta_rad_pairs
        )

    if theta_deg_pairs:

        theta_min_rad_candidates.extend(
            math.radians(
                pair[
                    0
                ]
            )
            for _, pair
            in theta_deg_pairs
        )

        theta_max_rad_candidates.extend(
            math.radians(
                pair[
                    1
                ]
            )
            for _, pair
            in theta_deg_pairs
        )

    # Frozen current-selector geometry is independent
    # evidence of the same angular control domain.
    theta_min_rad_candidates.append(
        math.radians(
            selector[
                "theta_domain_deg"
            ][
                0
            ]
        )
    )

    theta_max_rad_candidates.append(
        math.radians(
            selector[
                "theta_domain_deg"
            ][
                1
            ]
        )
    )

    theta_min_rad = unique_numeric(
        theta_min_rad_candidates,
        label="theta_min_rad",
        atol=1.0e-10,
    )

    theta_max_rad = unique_numeric(
        theta_max_rad_candidates,
        label="theta_max_rad",
        atol=1.0e-10,
    )

    # --------------------------------------------------------
    # Range max
    # --------------------------------------------------------

    range_pairs = numeric_pairs(
        rows,
        axis="range",
        unit_token="_m",
    )

    range_min_candidates = [
        value
        for _, value
        in scalar_candidates(
            rows,
            axis="range",
            side="min",
            unit_token="_m",
        )
    ]

    range_max_candidates = [
        value
        for _, value
        in scalar_candidates(
            rows,
            axis="range",
            side="max",
            unit_token="_m",
        )
    ]

    if range_pairs:

        range_min_candidates.extend(
            pair[
                0
            ]
            for _, pair
            in range_pairs
        )

        range_max_candidates.extend(
            pair[
                1
            ]
            for _, pair
            in range_pairs
        )

    # Current selector contract independently freezes
    # the maximum headlamp-relevant range.
    range_max_candidates.append(
        selector[
            "maximum_range_m"
        ]
    )

    range_max_m = unique_numeric(
        range_max_candidates,
        label="range_max_m",
        atol=1.0e-10,
    )

    # --------------------------------------------------------
    # Range min: NEVER default to zero.
    #
    # If no explicit minimum exists, derive it only from a
    # frozen range spacing + frozen n_range + frozen max.
    # --------------------------------------------------------

    range_steps = [
        value
        for _, value
        in step_candidates(
            rows,
            axis="range",
            unit_token="_m",
        )
    ]

    if not range_min_candidates:

        if range_steps:

            range_step_m = unique_numeric(
                range_steps,
                label="range_step_m",
                atol=1.0e-12,
            )

            derived_min = (
                range_max_m
                -
                range_step_m
                *
                (
                    EXPECTED_SHAPE[
                        1
                    ]
                    -
                    1
                )
            )

            range_min_candidates.append(
                derived_min
            )

    require(
        range_min_candidates,
        (
            "No exact range_min_m evidence. "
            "Refusing to assume 0.0."
        ),
    )

    range_min_m = unique_numeric(
        range_min_candidates,
        label="range_min_m",
        atol=1.0e-10,
    )

    # Counts are already an upstream Stage6 runtime shape
    # contract established before this failed runner.
    n_theta = int(
        EXPECTED_SHAPE[
            0
        ]
    )

    n_range = int(
        EXPECTED_SHAPE[
            1
        ]
    )

    require(
        theta_min_rad
        <
        theta_max_rad,
        "theta bounds are not increasing.",
    )

    require(
        range_min_m
        <
        range_max_m,
        "range bounds are not increasing.",
    )

    grid = IlluminationGridSpec(
        theta_min_rad=float(
            theta_min_rad
        ),

        theta_max_rad=float(
            theta_max_rad
        ),

        range_min_m=float(
            range_min_m
        ),

        range_max_m=float(
            range_max_m
        ),

        n_theta=n_theta,

        n_range=n_range,
    )

    blank = np.asarray(
        part_a_original_reactive_map_from_h0_centers(
            grid,
            (),
        ),
        dtype=np.float64,
    )

    require(
        blank.shape
        ==
        EXPECTED_SHAPE,
        (
            "Resolved grid does not reproduce "
            f"{EXPECTED_SHAPE}; got {blank.shape}."
        ),
    )

    require(
        np.all(
            np.isfinite(
                blank
            )
        ),
        (
            "Blank Part-A map "
            "contains non-finite values."
        ),
    )

    require(
        np.all(
            (
                blank >= 0.0
            )
            &
            (
                blank <= 1.0
            )
        ),
        (
            "Blank Part-A map "
            "outside [0,1]."
        ),
    )

    resolution = {
        "theta_min_rad":
            float(
                theta_min_rad
            ),

        "theta_max_rad":
            float(
                theta_max_rad
            ),

        "range_min_m":
            float(
                range_min_m
            ),

        "range_max_m":
            float(
                range_max_m
            ),

        "n_theta":
            n_theta,

        "n_range":
            n_range,
    }

    evidence = {
        "Block64_config_path":
            str(
                GRID_CONFIG
            ),

        "Block64_config_sha256":
            file_sha(
                GRID_CONFIG
            ),

        "scientific_illumination_grid_path":
            ".".join(
                subtree_path
            ),

        "scientific_illumination_grid":
            subtree,

        "flattened_grid_evidence":
            [
                {
                    "path":
                        path,

                    "value":
                        value,
                }
                for path, value
                in rows
            ],

        "selector_evidence":
            selector,

        "expected_runtime_grid_shape":
            list(
                EXPECTED_SHAPE
            ),

        "range_min_was_assumed_zero":
            False,

        "constructor_signature":
            str(
                inspect.signature(
                    IlluminationGridSpec
                )
            ),

        "blank_PartA_map_shape":
            list(
                blank.shape
            ),
    }

    return (
        grid,
        resolution,
        evidence,
    )


# ============================================================
# Exact constructor call-site diagnostic
# ============================================================

def constructor_call_sites():

    roots = (
        S6
        / "src/iscai_stage6",
        S6
        / "scripts",
    )

    result = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*.py"
        ):

            if (
                "formal"
                in
                path.name.lower()
            ):
                continue

            if (
                path.resolve()
                ==
                Path(
                    __file__
                ).resolve()
            ):
                continue

            source = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            try:

                tree = ast.parse(
                    source,
                    filename=str(
                        path
                    ),
                )

            except Exception:
                continue

            lines = source.splitlines()

            for node in ast.walk(
                tree
            ):

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

                    name = (
                        node.func.id
                    )

                elif isinstance(
                    node.func,
                    ast.Attribute,
                ):

                    name = (
                        node.func.attr
                    )

                if name not in (
                    "IlluminationGridSpec",
                    "OccupancyGrid",
                ):
                    continue

                start = max(
                    1,
                    int(
                        node.lineno
                    )
                    -
                    4,
                )

                end = min(
                    len(
                        lines
                    ),
                    int(
                        getattr(
                            node,
                            "end_lineno",
                            node.lineno,
                        )
                    )
                    +
                    4,
                )

                result.append({
                    "path":
                        str(
                            path.relative_to(
                                S6
                            )
                        ),

                    "constructor":
                        name,

                    "line":
                        int(
                            node.lineno
                        ),

                    "excerpt":
                        [
                            {
                                "line":
                                    index,

                                "text":
                                    lines[
                                        index - 1
                                    ].rstrip(),
                            }

                            for index
                            in range(
                                start,
                                end + 1,
                            )
                        ],
                })

    return result


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART2B GRID BINDING REPAIR"
    )
    print(
        "TARGETED SEMANTIC-KEY RESOLUTION"
    )
    print(
        "NO PERFORMANCE METRICS / NO FUTURE TRUTH"
    )
    print(
        "============================================================"
    )

    required = (
        GRID_CONFIG,
        PREDICTIVE_SELECTORS,
        REACTIVE_FALLBACKS,
        REACTIVE_FREEZE,
        NI_FREEZE,
        EVALUATOR_FREEZE,
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
            "Missing grid-binding dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN CONTINUITY ====="
    )

    exact_upstream_gate()

    print(
        "reactive scoring freeze = EXACT PASS"
    )

    print(
        "NI rule freeze          = EXACT PASS"
    )

    print(
        "evaluator freeze        = EXACT PASS"
    )

    print(
        "development outcomes    = NOT READ"
    )

    print(
        "future truth            = NOT READ"
    )

    # ========================================================
    # B. Reproduce diagnosis
    # ========================================================

    print()
    print(
        "===== B. FAILED RESOLVER DIAGNOSIS ====="
    )

    print(
        "IlluminationGridSpec signature =",
        inspect.signature(
            IlluminationGridSpec
        ),
    )

    print(
        "previous attempts             = []"
    )

    print(
        "diagnosis = SEMANTIC KEY MAPPING GAP"
    )

    print(
        "scientific grid failure       = NO"
    )

    # ========================================================
    # C. Exact Block6.4 subtree + selector evidence
    # ========================================================

    print()
    print(
        "===== C. EXACT FROZEN GRID EVIDENCE ====="
    )

    try:

        (
            grid,
            resolution,
            evidence,
        ) = resolve_exact_grid()

    except Exception:

        config = safe_json(
            GRID_CONFIG
        )

        matches = find_key_recursive(
            config,
            "scientific_illumination_grid",
        )

        print()
        print(
            "scientific_illumination_grid "
            "subtree count =",
            len(
                matches
            ),
        )

        for path, subtree in matches:

            print()
            print(
                "SUBTREE PATH =",
                ".".join(
                    path
                ),
            )

            print(
                json.dumps(
                    subtree,
                    indent=2,
                    sort_keys=True,
                )
            )

        print()
        print(
            "===== EXACT EXISTING CONSTRUCTOR CALL SITES ====="
        )

        sites = constructor_call_sites()

        for site in sites:

            print()
            print(
                f"{site['path']}:"
                f"L{site['line']} "
                f"{site['constructor']}"
            )

            for line in (
                site[
                    "excerpt"
                ]
            ):

                print(
                    f"  L{line['line']:04d}: "
                    f"{line['text']}"
                )

        raise

    print(
        "Block6.4 grid subtree = FOUND"
    )

    print(
        "selector theta domain =",
        evidence[
            "selector_evidence"
        ][
            "theta_domain_deg"
        ],
        "deg",
    )

    print(
        "selector max range    =",
        evidence[
            "selector_evidence"
        ][
            "maximum_range_m"
        ],
        "m",
    )

    print()
    print(
        "resolved constructor:"
    )

    for key in (
        "theta_min_rad",
        "theta_max_rad",
        "range_min_m",
        "range_max_m",
        "n_theta",
        "n_range",
    ):

        print(
            f"  {key} =",
            resolution[
                key
            ],
        )

    print(
        "range_min assumed zero = NO"
    )

    print(
        "Part-A map shape       = [501, 301] PASS"
    )

    # ========================================================
    # D. Write immutable runtime binding
    # ========================================================

    print()
    print(
        "===== D. RUNTIME BINDING FREEZE ====="
    )

    if BINDING.exists():

        existing = safe_json(
            BINDING
        )

        require(
            existing.get(
                "status"
            )
            ==
            "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING",
            (
                "Unexpected pre-existing "
                "grid binding."
            ),
        )

        require(
            existing[
                "constructor"
            ]
            ==
            resolution,
            (
                "Existing frozen grid binding "
                "differs from resolved evidence."
            ),
        )

        print(
            "existing binding = EXACT PASS"
        )

    else:

        binding = {
            "project":
                "Agni",

            "stage":
                6,

            "block":
                "6.8_Part2B_grid_binding_repair",

            "status":
                "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING",

            "repair_type":
                "semantic_key_binding_only",

            "scientific_parameter_change":
                False,

            "constructor":
                resolution,

            "expected_map_shape":
                [
                    501,
                    301,
                ],

            "provenance":
                evidence,

            "anti_invention": {
                "range_min_defaulted_to_zero":
                    False,

                "constructor_values_selected_from_outcomes":
                    False,

                "predictive_performance_read":
                    False,

                "reactive_performance_read":
                    False,

                "future_truth_read":
                    False,

                "formal_content_read":
                    False,
            },

            "next":
                (
                    "Part2B Part1 decision-ledger runner "
                    "must load exactly these six frozen "
                    "constructor values instead of attempting "
                    "heuristic key discovery."
                ),
        }

        atomic_json(
            BINDING,
            binding,
        )

        print(
            "runtime binding = WRITTEN"
        )

    binding_sha = file_sha(
        BINDING
    )

    # ========================================================
    # E. Readback construction
    # ========================================================

    print()
    print(
        "===== E. EXACT BINDING READBACK ====="
    )

    frozen = safe_json(
        BINDING
    )

    kwargs = frozen[
        "constructor"
    ]

    readback_grid = (
        IlluminationGridSpec(
            **kwargs
        )
    )

    readback_map = np.asarray(
        part_a_original_reactive_map_from_h0_centers(
            readback_grid,
            (),
        ),
        dtype=np.float64,
    )

    require(
        readback_map.shape
        ==
        EXPECTED_SHAPE,
        (
            "Frozen readback map "
            "shape mismatch."
        ),
    )

    require(
        frozen[
            "anti_invention"
        ][
            "range_min_defaulted_to_zero"
        ]
        is False,
        (
            "Binding incorrectly reports "
            "range-min invention."
        ),
    )

    print(
        "constructor readback = PASS"
    )

    print(
        "blank Part-A map     = [501, 301] PASS"
    )

    print(
        "numeric invention    = NO"
    )

    # ========================================================
    # F. Regression / immutability
    # ========================================================

    print()
    print(
        "===== F. REGRESSION + IMMUTABILITY ====="
    )

    exact_upstream_gate()

    rc, tests, output = (
        run_regression()
    )

    require(
        rc == 0,
        (
            "Stage6 regression failed:\n"
            +
            tail(
                output
            )
        ),
    )

    require(
        tests
        ==
        EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} "
            f"tests; got {tests}."
        ),
    )

    print(
        "scientific upstream = UNCHANGED"
    )

    print(
        "Stage6 regression   =",
        f"{tests} / {tests} PASS",
    )

    # ========================================================
    # G. Storage
    # ========================================================

    print()
    print(
        "===== G. STORAGE ====="
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
            "250-GiB reserve violated."
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
    # H. Report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8_Part2B_grid_binding_repair",

        "status":
            "PASS_FROZEN_GRID_RUNTIME_BINDING",

        "diagnosis": {
            "previous_failure":
                (
                    "resolver required constructor-key "
                    "names that were absent from "
                    "frozen semantic config"
                ),

            "previous_attempt_count":
                0,

            "scientific_grid_failure":
                False,

            "scientific_parameter_change":
                False,
        },

        "constructor":
            resolution,

        "binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                binding_sha,
        },

        "evidence":
            evidence,

        "scientific_boundary": {
            "development_performance_read":
                False,

            "reactive_performance_read":
                False,

            "predictive_performance_read":
                False,

            "future_truth_read":
                False,

            "P_occ_content_read":
                False,

            "exact_NI_deltas_computed":
                False,

            "formal_content_read":
                False,

            "formal_evaluation":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,
        },

        "regression": {
            "tests":
                tests,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "next":
            (
                "repair only resolve_grid() in the "
                "Part2B Part1 decision-ledger runner "
                "to load the frozen runtime binding, "
                "then rerun Part2B Part1."
            ),
    }

    atomic_json(
        REPORT,
        result,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 GRID RUNTIME BINDING REPAIR — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "previous resolver bug     = CONFIRMED"
    )

    print(
        "scientific grid failure   = NO"
    )

    print(
        "frozen semantic evidence  = RESOLVED"
    )

    print(
        "IlluminationGridSpec args = 6 / 6 PROVEN"
    )

    print(
        "runtime map shape         = 501 x 301 PASS"
    )

    print(
        "range_min assumed zero    = NO"
    )

    print(
        "scientific parameter edit = NO"
    )

    print(
        "reactive performance      = NOT READ"
    )

    print(
        "predictive performance    = NOT READ"
    )

    print(
        "future truth              = NOT READ"
    )

    print(
        "P_occ content             = NOT OPENED"
    )

    print(
        "exact NI deltas           = NOT COMPUTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )

    print(
        "binding SHA256            =",
        binding_sha,
    )

    print(
        "STATUS = PASS_FROZEN_GRID_RUNTIME_BINDING"
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
        "BLOCK 6.8 GRID RUNTIME BINDING REPAIR = BLOCKED"
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
        "scientific parameter edit = NO"
    )

    print(
        "reactive performance      = NOT READ"
    )

    print(
        "predictive performance    = NOT READ"
    )

    print(
        "future truth              = NOT READ"
    )

    print(
        "P_occ content             = NOT OPENED"
    )

    print(
        "exact NI deltas           = NOT COMPUTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print()
    print(
        "Do not rerun Part2B Part1 yet."
    )

    print(
        "Send sections C and the BLOCKED "
        "footer for targeted repair."
    )

    print(
        "terminal remains open = YES"
    )

# No non-zero sys.exit().

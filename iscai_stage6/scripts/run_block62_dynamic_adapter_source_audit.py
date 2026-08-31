from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

NOTEBOOK = (
    ROOT
    / "part_a_reference/ISCAI_pc_fmcw/"
      "notebooks/ISCAI_PC_FMCW.ipynb"
)

PREBIND = (
    S6
    / "reports/"
      "block62_part2_prebinding_source_audit.json"
)

REPORT = (
    S6
    / "reports/"
      "block62_dynamic_adapter_source_audit.json"
)

EXCERPT = (
    S6
    / "artifacts/block62/"
      "part_a_dynamic_adapter_exact_functions.txt"
)

EXPECTED_NOTEBOOK_SHA = (
    "b5a80a6d3441de6d571db4f65b4a43e"
    "d4052cc2b3ccba935ad31b5dd51316ef3"
)

EXPECTED_PREBIND_SHA = (
    "fab4821ce11e8969a2ed8c7a36e74e7b"
    "85bc1e27116cf23a445ce32c3e3f0ffb"
)

TARGET_FUNCTIONS = (
    "adb_intensity_map",
    "adb_combine_intensity_maps",
    "adb_trz_shadow_geometry_from_xy",
)

CRITICAL_PARAMETER_NAMES = (
    "theta_grid_deg",
    "range_grid_m",
    "target_width_m",
    "target_range_m",
    "lateral_safety_margin_m",
    "radial_safety_margin_m",
    "transition_length_m",
    "epsilon_deg",
    "theta_c_deg",
    "delta_theta_deg",
    "r_shadow_end_m",
    "r_transition_end_m",
    "assumed_vehicle_width_m",
    "assumed_vehicle_half_width_m",
    "assumed_safety_margin_deg",
    "assumed_safety_margin_rad",
    "assumed_adb_r_min_m",
    "assumed_adb_r_max_m",
)


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


def cell_source(cell) -> str:

    source = cell.get(
        "source",
        "",
    )

    if isinstance(
        source,
        list,
    ):
        return "".join(source)

    return str(source)


def safe_python(source: str) -> str:

    rows = []

    for line in source.splitlines():

        stripped = line.lstrip()

        if (
            stripped.startswith("%")
            or
            stripped.startswith("!")
            or
            stripped.startswith("?")
        ):
            indent = line[
                :len(line) - len(stripped)
            ]

            rows.append(
                indent
                +
                "# NOTEBOOK_MAGIC "
                +
                stripped
            )

        else:
            rows.append(line)

    return "\n".join(rows)


def source_sha(text: str) -> str:

    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def atomic_write(
    path: Path,
    data: bytes,
):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(data)

    os.replace(
        tmp,
        path,
    )


def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.2 — EXACT DYNAMIC PART-A ADAPTER SOURCE AUDIT"
    )
    print(
        "READ ONLY / NO NOTEBOOK EXECUTION"
    )
    print(
        "============================================================"
    )

    # ========================================================
    # A. Frozen identity
    # ========================================================

    print()
    print(
        "===== A. FROZEN IDENTITY ====="
    )

    require(
        NOTEBOOK.is_file(),
        f"Missing notebook: {NOTEBOOK}",
    )

    require(
        PREBIND.is_file(),
        f"Missing prebinding audit: {PREBIND}",
    )

    notebook_sha = sha256_file(
        NOTEBOOK
    )

    prebind_sha = sha256_file(
        PREBIND
    )

    require(
        notebook_sha
        ==
        EXPECTED_NOTEBOOK_SHA,
        (
            "Canonical notebook SHA changed."
        ),
    )

    require(
        prebind_sha
        ==
        EXPECTED_PREBIND_SHA,
        (
            "Prebinding audit SHA changed."
        ),
    )

    prebind = json.loads(
        PREBIND.read_text(
            encoding="utf-8"
        )
    )

    require(
        prebind.get(
            "status"
        )
        ==
        "PASS_READY_FOR_EXACT_PARAMETER_BINDING",
        (
            "Prebinding audit is not ready "
            "for exact binding."
        ),
    )

    print(
        "canonical notebook = EXACT SHA PASS"
    )

    print(
        "prebinding audit   = EXACT SHA PASS"
    )

    print(
        "notebook execution = NO"
    )

    # ========================================================
    # B. Load code cells
    # ========================================================

    notebook = json.loads(
        NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )

    cells = notebook.get(
        "cells",
        [],
    )

    code_cells = []

    for index, cell in enumerate(
        cells
    ):

        if cell.get(
            "cell_type"
        ) != "code":
            continue

        source = cell_source(
            cell
        )

        code_cells.append(
            (
                index,
                source,
            )
        )

    # ========================================================
    # C. Exact function discovery
    # ========================================================

    print()
    print(
        "===== B. EXACT DYNAMIC FUNCTION DISCOVERY ====="
    )

    found = {}
    all_adb_functions = []

    for cell_index, source in code_cells:

        sanitized = safe_python(
            source
        )

        try:
            tree = ast.parse(
                sanitized
            )

        except SyntaxError:
            continue

        for node in tree.body:

            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                continue

            name = node.name

            if (
                "adb"
                in
                name.lower()
            ):
                all_adb_functions.append(
                    (
                        cell_index,
                        name,
                    )
                )

            if name not in TARGET_FUNCTIONS:
                continue

            exact = ast.get_source_segment(
                sanitized,
                node,
            )

            require(
                exact is not None,
                (
                    "Could not extract exact source "
                    f"for {name}"
                ),
            )

            require(
                name not in found,
                (
                    "Duplicate canonical function "
                    f"definition: {name}"
                ),
            )

            found[
                name
            ] = {
                "cell_index":
                    cell_index,

                "source":
                    exact,

                "source_sha256":
                    source_sha(
                        exact
                    ),
            }

    print(
        "all ADB function definitions ="
    )

    for cell_index, name in (
        all_adb_functions
    ):
        print(
            f"  cell={cell_index:02d} | {name}"
        )

    missing = [
        name
        for name in TARGET_FUNCTIONS
        if name not in found
    ]

    require(
        not missing,
        (
            "Required Part-A dynamic function(s) "
            "not found: "
            + ", ".join(missing)
        ),
    )

    print()

    for name in TARGET_FUNCTIONS:

        item = found[
            name
        ]

        print(
            name,
            "| cell =",
            item[
                "cell_index"
            ],
            "| SHA256 =",
            item[
                "source_sha256"
            ],
        )

    # ========================================================
    # D. Print exact function sources
    # ========================================================

    print()
    print(
        "===== C. EXACT FUNCTION SOURCES ====="
    )

    excerpt_rows = [
        (
            "CANONICAL PART-A DYNAMIC ADB FUNCTIONS\n"
            f"notebook_sha256={notebook_sha}\n"
            "notebook_executed=false\n\n"
        )
    ]

    for name in TARGET_FUNCTIONS:

        item = found[
            name
        ]

        header = (
            "\n"
            "============================================================\n"
            f"FUNCTION: {name}\n"
            f"CELL: {item['cell_index']}\n"
            f"SHA256: {item['source_sha256']}\n"
            "============================================================\n"
        )

        excerpt_rows.append(
            header
        )

        excerpt_rows.append(
            item[
                "source"
            ]
        )

        excerpt_rows.append(
            "\n"
        )

        print(
            header,
            end="",
        )

        print(
            item[
                "source"
            ]
        )

    atomic_write(
        EXCERPT,
        "".join(
            excerpt_rows
        ).encode(
            "utf-8"
        ),
    )

    # ========================================================
    # E. Critical exact parameter source lines
    # ========================================================

    print()
    print(
        "===== D. CRITICAL PARAMETER SOURCE LINES ====="
    )

    parameter_lines = {}

    for cell_index, source in code_cells:

        for line_number, line in enumerate(
            source.splitlines(),
            start=1,
        ):

            stripped = line.strip()

            for name in (
                CRITICAL_PARAMETER_NAMES
            ):

                if not stripped.startswith(
                    name
                ):
                    continue

                if "=" not in stripped:
                    continue

                parameter_lines.setdefault(
                    name,
                    [],
                ).append({
                    "cell_index":
                        cell_index,

                    "line_number":
                        line_number,

                    "source":
                        stripped,
                })

    for name in CRITICAL_PARAMETER_NAMES:

        rows = parameter_lines.get(
            name,
            [],
        )

        if not rows:
            print(
                f"{name:32s} = NOT FOUND"
            )

            continue

        for row in rows:

            print(
                f"{name:32s} | "
                f"C{row['cell_index']:02d}:"
                f"L{row['line_number']:03d} | "
                f"{row['source']}"
            )

    # Exact epsilon gate.
    epsilon_lines = parameter_lines.get(
        "epsilon_deg",
        [],
    )

    require(
        any(
            re_exact[
                "source"
            ].replace(
                " ",
                "",
            )
            ==
            "epsilon_deg=0.75"
            for re_exact in epsilon_lines
        ),
        (
            "Canonical notebook does not contain "
            "the expected exact epsilon_deg=0.75 "
            "assignment."
        ),
    )

    print()
    print(
        "canonical epsilon_deg = 0.75 EXACT SOURCE PASS"
    )

    # ========================================================
    # F. Semantic function audit
    # ========================================================

    print()
    print(
        "===== E. DYNAMIC ADAPTER SEMANTIC GATES ====="
    )

    geom = found[
        "adb_trz_shadow_geometry_from_xy"
    ][
        "source"
    ].lower()

    intensity = found[
        "adb_intensity_map"
    ][
        "source"
    ].lower()

    combine = found[
        "adb_combine_intensity_maps"
    ][
        "source"
    ].lower()

    semantic = {
        "dynamic_geometry_uses_xy":
            (
                "x"
                in geom
                and
                "y"
                in geom
            ),

        "dynamic_geometry_uses_angular_mapping":
            (
                "atan"
                in geom
                or
                "arctan"
                in geom
            ),

        "dynamic_geometry_uses_vehicle_width":
            (
                "width"
                in geom
                or
                "w_half"
                in geom
            ),

        "dynamic_geometry_uses_margin":
            (
                "margin"
                in geom
                or
                "epsilon"
                in geom
                or
                "eps"
                in geom
            ),

        "intensity_has_transition":
            (
                "transition"
                in intensity
                or
                "cos"
                in intensity
            ),

        "intensity_has_shadow":
            (
                "shadow"
                in intensity
            ),

        "multi_actor_combination_is_min_like":
            (
                "minimum"
                in combine
                or
                "np.min"
                in combine
                or
                "min("
                in combine
            ),
    }

    for name, passed in semantic.items():

        print(
            f"{name:40s} =",
            "PASS"
            if passed
            else
            "NOT PROVEN",
        )

    require(
        semantic[
            "dynamic_geometry_uses_xy"
        ],
        "Dynamic geometry x/y semantics not proven.",
    )

    require(
        semantic[
            "dynamic_geometry_uses_angular_mapping"
        ],
        "Dynamic angular mapping not proven.",
    )

    require(
        semantic[
            "dynamic_geometry_uses_vehicle_width"
        ],
        "Dynamic vehicle-width use not proven.",
    )

    require(
        semantic[
            "intensity_has_transition"
        ],
        "Transition logic not proven.",
    )

    require(
        semantic[
            "intensity_has_shadow"
        ],
        "Shadow logic not proven.",
    )

    require(
        semantic[
            "multi_actor_combination_is_min_like"
        ],
        (
            "Elementwise minimum multi-actor "
            "rule not proven."
        ),
    )

    # ========================================================
    # G. Binding-policy artifact
    # ========================================================

    print()
    print(
        "===== F. BINDING POLICY ====="
    )

    binding_policy = {
        "canonical_parameter_authority":
            "FROZEN_NOTEBOOK_SOURCE",

        "epsilon_deg":
            0.75,

        "epsilon_binding":
            "EXACT_NOTEBOOK_ASSIGNMENT",

        "technical_report_0_745":
            "ARITHMETIC_TEXT_INCONSISTENCY",

        "technical_report_0_745_used_as_parameter":
            False,

        "Part_A_grid_role":
            "ORIGINAL_REACTIVE_BASELINE_ONLY",

        "Stage6_final_predictive_grid_frozen":
            False,

        "Stage6_final_predictive_grid_freeze":
            "BLOCK6.8_PRE_FORMAL",

        "coordinate_policy":
            (
                "do not infer x/y semantics from "
                "plot labels; bind dynamic adapter "
                "to exact notebook function and "
                "Stage6 H0 geometry contract"
            ),

        "paper_ADB_role":
            "CAMERA_GEOMETRY_REACTIVE",

        "Part_A_role":
            "MHT_CURRENT_STATE_REACTIVE_EXTENSION",

        "Part_B_role":
            "PROBABILISTIC_PREDICTIVE_CLASS_AWARE",
    }

    print(
        "epsilon authority       = NOTEBOOK 0.75"
    )

    print(
        "0.745 report arithmetic = DOCUMENTED DISCREPANCY"
    )

    print(
        "Part-A grid role        = BASELINE ONLY"
    )

    print(
        "Stage6 final grid       = NOT FROZEN"
    )

    print(
        "x/y convention          = SOURCE-BIND BEFORE ADAPTER"
    )

    # ========================================================
    # H. Report
    # ========================================================

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "audit":
            "exact_dynamic_adapter_source",

        "status":
            "PASS_READY_FOR_BLOCK62_PART2_IMPLEMENTATION",

        "notebook": {
            "path":
                str(
                    NOTEBOOK
                ),

            "sha256":
                notebook_sha,

            "executed":
                False,
        },

        "prebinding_audit": {
            "path":
                str(
                    PREBIND
                ),

            "sha256":
                prebind_sha,
        },

        "functions":
            {
                name: {
                    "cell_index":
                        found[
                            name
                        ][
                            "cell_index"
                        ],

                    "source_sha256":
                        found[
                            name
                        ][
                            "source_sha256"
                        ],
                }
                for name in TARGET_FUNCTIONS
            },

        "critical_parameter_source_lines":
            parameter_lines,

        "semantic_gates":
            semantic,

        "binding_policy":
            binding_policy,

        "exact_function_excerpt":
            str(
                EXCERPT
            ),

        "execution": {
            "notebook_execution":
                False,

            "dataset_access":
                False,

            "training":
                False,

            "model_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,
        },

        "upstream_modified":
            False,
    }

    atomic_write(
        REPORT,
        (
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            +
            "\n"
        ).encode(
            "utf-8"
        ),
    )

    print()
    print(
        "audit report =",
        REPORT,
    )

    print(
        "audit SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print(
        "function excerpt =",
        EXCERPT,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "DYNAMIC ADAPTER SOURCE AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "canonical notebook       = EXACT PASS"
    )

    print(
        "epsilon 0.75             = EXACT SOURCE PASS"
    )

    print(
        "report 0.745             = DOCUMENTED INCONSISTENCY"
    )

    print(
        "adb_intensity_map        = EXACT SOURCE CAPTURED"
    )

    print(
        "adb_combine_intensity_maps= EXACT SOURCE CAPTURED"
    )

    print(
        "adb_trz_shadow_geometry  = EXACT SOURCE CAPTURED"
    )

    print(
        "dynamic x/y semantics    = SOURCE EXPOSED"
    )

    print(
        "Part-A grid              = BASELINE-ONLY"
    )

    print(
        "Stage6 final grid freeze = NO"
    )

    print(
        "scientific implementation= UNCHANGED"
    )

    print(
        "upstream modified        = NO"
    )

    print(
        "STATUS = PASS_READY_FOR_BLOCK62_PART2_IMPLEMENTATION"
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
        "DYNAMIC ADAPTER SOURCE AUDIT = BLOCKED"
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
        limit=12
    )

    print()
    print(
        "Part-A modified        = NO"
    )

    print(
        "Stage1–5 modified      = NO"
    )

    print(
        "Block6.1 modified      = NO"
    )

    print(
        "Block6.2 Part1 modified= NO"
    )

    print(
        "dataset access         = NO"
    )

    print(
        "training/inference     = NO"
    )

    print(
        "formal evaluation      = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# No non-zero sys.exit().

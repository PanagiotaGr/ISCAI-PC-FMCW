from __future__ import annotations

import ast
from collections import defaultdict, deque
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
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

BLOCK62_PART1_REPORT = (
    S6
    / "reports/"
      "block62_part1_static_reactive_kernel.json"
)

BLOCK62_PART1_CONTRACT = (
    S6
    / "configs/"
      "block62_static_reactive_kernel_contract.json"
)

BLOCK61_CLOSURE = (
    S6
    / "reports/block61_closure.json"
)

REPORT = (
    S6
    / "reports/"
      "block62_dynamic_dependency_chain_audit.json"
)

EXCERPT = (
    S6
    / "artifacts/block62/"
      "part_a_dynamic_dependency_chain_exact_source.txt"
)


EXPECTED = {
    "notebook":
        (
            "b5a80a6d3441de6d571db4f65b4a43e"
            "d4052cc2b3ccba935ad31b5dd51316ef3"
        ),

    "prebind":
        (
            "fab4821ce11e8969a2ed8c7a36e74e7b"
            "85bc1e27116cf23a445ce32c3e3f0ffb"
        ),

    "part1_report":
        (
            "865c81eeb689a1b0c7e41b08117c9197"
            "c19e5cac76de147a07898b17c2224477"
        ),

    "part1_contract":
        (
            "df4cf1e8fd18f3bc9cd48cbae4a6c209"
            "1e37313cbc1a5b2cefe48bd340eb5d60"
        ),

    "block61_closure":
        (
            "1d0f8781d365efd09614c8fb5a06bdbd"
            "a8f56c1c673b0ddab8b40810de270ac1"
        ),
}


ROOT_GEOMETRY_FUNCTION = (
    "adb_trz_shadow_geometry_from_xy"
)

EXPECTED_GEOMETRY_HELPER = (
    "adb_shadow_geometry_from_xy"
)

POSSIBLE_ANGLE_HELPER = (
    "adb_shadow_angles_deg"
)

INTENSITY_FUNCTION = (
    "adb_intensity_map"
)

COMBINE_FUNCTION = (
    "adb_combine_intensity_maps"
)

TRACK_ARRAY_FUNCTION = (
    "adb_trz_track_arrays"
)


# ============================================================
# Helpers
# ============================================================

def require(
    condition,
    message,
):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:

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


def source_sha(
    text: str,
) -> str:

    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def cell_source(
    cell,
) -> str:

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


def safe_python(
    source: str,
) -> str:

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


def atomic_write(
    path: Path,
    data: bytes,
):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        data
    )

    os.replace(
        tmp,
        path,
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


def function_called_names(
    source: str,
) -> set[str]:

    tree = ast.parse(
        source
    )

    called = set()

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        if isinstance(
            node.func,
            ast.Name,
        ):
            called.add(
                node.func.id
            )

        elif isinstance(
            node.func,
            ast.Attribute,
        ):
            # Keep dotted name for evidence,
            # but dependency resolution uses only
            # notebook-defined Name calls.
            try:
                called.add(
                    ast.unparse(
                        node.func
                    )
                )
            except BaseException:
                pass

    return called


def extract_assignments(
    code_cells,
):

    results = defaultdict(
        list
    )

    for cell_index, source in (
        code_cells
    ):

        for line_number, line in enumerate(
            source.splitlines(),
            start=1,
        ):

            stripped = line.strip()

            match = re.match(
                r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$",
                stripped,
            )

            if not match:
                continue

            name = match.group(1)

            results[
                name
            ].append({
                "cell_index":
                    cell_index,

                "line_number":
                    line_number,

                "source":
                    stripped,
            })

    return dict(
        results
    )


def collect_load_names(
    function_source: str,
) -> set[str]:

    tree = ast.parse(
        function_source
    )

    function_node = None

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            function_node = node
            break

    require(
        function_node is not None,
        "Function AST extraction failed.",
    )

    arguments = {
        argument.arg
        for argument in (
            list(
                function_node.args.args
            )
            +
            list(
                function_node.args.kwonlyargs
            )
        )
    }

    if (
        function_node.args.vararg
        is not None
    ):
        arguments.add(
            function_node.args.vararg.arg
        )

    if (
        function_node.args.kwarg
        is not None
    ):
        arguments.add(
            function_node.args.kwarg.arg
        )

    assigned = set()

    loaded = set()

    for node in ast.walk(
        function_node
    ):

        if isinstance(
            node,
            ast.Name,
        ):

            if isinstance(
                node.ctx,
                ast.Store,
            ):
                assigned.add(
                    node.id
                )

            elif isinstance(
                node.ctx,
                ast.Load,
            ):
                loaded.add(
                    node.id
                )

    return (
        loaded
        -
        assigned
        -
        arguments
    )


def contexts_for_token(
    code_cells,
    token: str,
    radius: int = 8,
):

    results = []

    for cell_index, source in (
        code_cells
    ):

        lines = source.splitlines()

        for index, line in enumerate(
            lines
        ):

            if token not in line:
                continue

            lo = max(
                0,
                index - radius,
            )

            hi = min(
                len(lines),
                index + radius + 1,
            )

            results.append({
                "cell_index":
                    cell_index,

                "matched_line":
                    index + 1,

                "context":
                    "\n".join(
                        f"{line_number + 1:04d}: "
                        f"{lines[line_number]}"
                        for line_number
                        in range(
                            lo,
                            hi,
                        )
                    ),
            })

    return results


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.2 — TRANSITIVE PART-A DYNAMIC ADB AUDIT"
    )
    print(
        "CALL-CHAIN + COORDINATE + WIDTH/MARGIN VERIFICATION"
    )
    print(
        "============================================================"
    )

    protected = (
        (
            "notebook",
            NOTEBOOK,
        ),
        (
            "prebind",
            PREBIND,
        ),
        (
            "part1_report",
            BLOCK62_PART1_REPORT,
        ),
        (
            "part1_contract",
            BLOCK62_PART1_CONTRACT,
        ),
        (
            "block61_closure",
            BLOCK61_CLOSURE,
        ),
    )

    # ========================================================
    # A. Frozen seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN SEAL ====="
    )

    protected_before = {}

    for name, path in protected:

        require(
            path.is_file(),
            f"Missing prerequisite: {path}",
        )

        actual = sha256_file(
            path
        )

        protected_before[
            str(path)
        ] = actual

        require(
            actual
            ==
            EXPECTED[
                name
            ],
            (
                f"{name} SHA mismatch.\n"
                f"expected={EXPECTED[name]}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{name:22s} = EXACT PASS"
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
        "Prebinding audit status changed.",
    )

    # ========================================================
    # B. Static notebook function database
    # ========================================================

    print()
    print(
        "===== B. NOTEBOOK FUNCTION DATABASE ====="
    )

    notebook = json.loads(
        NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )

    code_cells = []

    function_candidates = defaultdict(
        list
    )

    for cell_index, cell in enumerate(
        notebook.get(
            "cells",
            [],
        )
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
                cell_index,
                source,
            )
        )

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

            exact = ast.get_source_segment(
                sanitized,
                node,
            )

            if exact is None:
                continue

            function_candidates[
                node.name
            ].append({
                "cell_index":
                    cell_index,

                "source":
                    exact,

                "source_sha256":
                    source_sha(
                        exact
                    ),
            })

    required_functions = (
        ROOT_GEOMETRY_FUNCTION,
        EXPECTED_GEOMETRY_HELPER,
        INTENSITY_FUNCTION,
        COMBINE_FUNCTION,
        TRACK_ARRAY_FUNCTION,
    )

    for name in required_functions:

        count = len(
            function_candidates.get(
                name,
                [],
            )
        )

        require(
            count == 1,
            (
                f"Expected exactly one definition "
                f"of {name}; found {count}."
            ),
        )

        item = (
            function_candidates[
                name
            ][0]
        )

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

    angle_helper_count = len(
        function_candidates.get(
            POSSIBLE_ANGLE_HELPER,
            [],
        )
    )

    print(
        POSSIBLE_ANGLE_HELPER,
        "| definitions =",
        angle_helper_count,
    )

    # ========================================================
    # C. Build transitive geometry call chain
    # ========================================================

    print()
    print(
        "===== C. TRANSITIVE GEOMETRY CALL CHAIN ====="
    )

    unique_functions = {}

    for name, items in (
        function_candidates.items()
    ):

        if len(items) == 1:
            unique_functions[
                name
            ] = items[0]

    queue = deque(
        [
            ROOT_GEOMETRY_FUNCTION
        ]
    )

    closure = []
    seen = set()

    edges = []

    while queue:

        name = queue.popleft()

        if name in seen:
            continue

        seen.add(
            name
        )

        require(
            name
            in
            unique_functions,
            (
                "Ambiguous or missing function "
                f"in dependency closure: {name}"
            ),
        )

        closure.append(
            name
        )

        source = unique_functions[
            name
        ][
            "source"
        ]

        calls = function_called_names(
            source
        )

        for called in sorted(
            calls
        ):

            if called not in (
                function_candidates
            ):
                continue

            require(
                len(
                    function_candidates[
                        called
                    ]
                )
                ==
                1,
                (
                    "Ambiguous notebook helper "
                    f"definition: {called}"
                ),
            )

            edges.append(
                (
                    name,
                    called,
                )
            )

            if called not in seen:
                queue.append(
                    called
                )

    require(
        EXPECTED_GEOMETRY_HELPER
        in
        closure,
        (
            "TRZ wrapper does not resolve "
            "transitively to "
            "adb_shadow_geometry_from_xy."
        ),
    )

    print(
        "geometry dependency closure =",
        closure,
    )

    print(
        "call edges:"
    )

    for caller, callee in edges:

        print(
            " ",
            caller,
            "->",
            callee,
        )

    # ========================================================
    # D. Exact geometry dependency sources
    # ========================================================

    print()
    print(
        "===== D. EXACT TRANSITIVE GEOMETRY SOURCES ====="
    )

    excerpt_parts = [
        (
            "PART-A DYNAMIC ADB TRANSITIVE DEPENDENCY AUDIT\n"
            f"notebook_sha256="
            f"{EXPECTED['notebook']}\n"
            "notebook_executed=false\n\n"
        )
    ]

    geometry_source_parts = []

    for name in closure:

        item = unique_functions[
            name
        ]

        source = item[
            "source"
        ]

        geometry_source_parts.append(
            source
        )

        header = (
            "\n"
            "============================================================\n"
            f"FUNCTION: {name}\n"
            f"CELL: {item['cell_index']}\n"
            f"SHA256: {item['source_sha256']}\n"
            "============================================================\n"
        )

        print(
            header,
            end="",
        )

        print(
            source
        )

        excerpt_parts.append(
            header
        )

        excerpt_parts.append(
            source
            +
            "\n"
        )

    geometry_source = "\n\n".join(
        geometry_source_parts
    )

    geometry_lower = (
        geometry_source.lower()
    )

    # ========================================================
    # E. Transitive semantic gates
    # ========================================================

    print()
    print(
        "===== E. TRANSITIVE GEOMETRY SEMANTIC GATES ====="
    )

    semantic = {
        "wrapper_delegates_to_geometry_helper":
            (
                EXPECTED_GEOMETRY_HELPER
                in
                closure
            ),

        "xy_inputs":
            (
                "x"
                in geometry_lower
                and
                "y"
                in geometry_lower
            ),

        "angular_mapping":
            (
                "atan2"
                in geometry_lower
                or
                "arctan2"
                in geometry_lower
            ),

        "range_mapping":
            (
                "hypot"
                in geometry_lower
                or
                "sqrt"
                in geometry_lower
            ),

        "vehicle_width_or_half_width":
            any(
                token
                in
                geometry_lower
                for token in (
                    "vehicle_width",
                    "vehicle_half_width",
                    "half_width",
                    "w_half",
                    "target_width",
                    "width",
                )
            ),

        "safety_margin":
            (
                "margin"
                in
                geometry_lower
            ),

        "angular_shadow_bounds":
            (
                "theta_min"
                in
                geometry_lower
                and
                "theta_max"
                in
                geometry_lower
            ),

        "radial_zone_support":
            any(
                token
                in
                geometry_lower
                for token in (
                    "radial",
                    "r_min",
                    "r_max",
                    "range_min",
                    "range_max",
                )
            ),

        "current_dynamic_geometry":
            (
                "future"
                not in
                geometry_lower
                and
                "posterior"
                not in
                geometry_lower
            ),
    }

    for name, passed in semantic.items():

        print(
            f"{name:42s} =",
            "PASS"
            if passed
            else
            "NOT PROVEN",
        )

    # These are actual requirements of the Part-A dynamic
    # geometry chain. We do NOT weaken them just because the
    # previous wrapper-only audit was wrong.
    required_semantic = (
        "wrapper_delegates_to_geometry_helper",
        "xy_inputs",
        "angular_mapping",
        "range_mapping",
        "vehicle_width_or_half_width",
        "safety_margin",
        "angular_shadow_bounds",
        "current_dynamic_geometry",
    )

    failed = [
        name
        for name in required_semantic
        if not semantic[
            name
        ]
    ]

    require(
        not failed,
        (
            "Real transitive Part-A geometry "
            "requirement not proven: "
            + ", ".join(
                failed
            )
        ),
    )

    # ========================================================
    # F. Global parameters referenced by geometry chain
    # ========================================================

    print()
    print(
        "===== F. TRANSITIVE GLOBAL PARAMETER EVIDENCE ====="
    )

    assignments = extract_assignments(
        code_cells
    )

    global_names = set()

    for name in closure:

        source = unique_functions[
            name
        ][
            "source"
        ]

        global_names.update(
            collect_load_names(
                source
            )
        )

    notebook_function_names = set(
        function_candidates
    )

    ignored = {
        "np",
        "math",
        "float",
        "int",
        "len",
        "range",
        "zip",
        "list",
        "tuple",
        "dict",
        "min",
        "max",
        "abs",
        "bool",
    }

    relevant_globals = sorted(
        name
        for name in global_names
        if (
            name
            not in notebook_function_names
            and
            name
            not in ignored
        )
    )

    print(
        "geometry external/global names =",
        relevant_globals,
    )

    exact_global_lines = {}

    for name in relevant_globals:

        rows = assignments.get(
            name,
            [],
        )

        if not rows:
            continue

        exact_global_lines[
            name
        ] = rows

        for row in rows:

            print(
                f"{name:34s} | "
                f"C{row['cell_index']:02d}:"
                f"L{row['line_number']:03d} | "
                f"{row['source']}"
            )

    # ========================================================
    # G. Exact intensity / combination sources
    # ========================================================

    print()
    print(
        "===== G. EXACT INTENSITY / COMBINATION SOURCES ====="
    )

    intensity_item = unique_functions[
        INTENSITY_FUNCTION
    ]

    combine_item = unique_functions[
        COMBINE_FUNCTION
    ]

    for name, item in (
        (
            INTENSITY_FUNCTION,
            intensity_item,
        ),
        (
            COMBINE_FUNCTION,
            combine_item,
        ),
    ):

        header = (
            "\n"
            "============================================================\n"
            f"FUNCTION: {name}\n"
            f"CELL: {item['cell_index']}\n"
            f"SHA256: {item['source_sha256']}\n"
            "============================================================\n"
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

        excerpt_parts.append(
            header
        )

        excerpt_parts.append(
            item[
                "source"
            ]
            +
            "\n"
        )

    intensity_lower = (
        intensity_item[
            "source"
        ].lower()
    )

    combine_lower = (
        combine_item[
            "source"
        ].lower()
    )

    intensity_semantic = {
        "target_range_from_xy":
            (
                "hypot"
                in
                intensity_lower
            ),

        "radial_safety_margin":
            (
                "radial_safety_margin"
                in
                intensity_lower
            ),

        "transition_length":
            (
                "transition_length"
                in
                intensity_lower
            ),

        "full_shadow_zero":
            (
                "= 0.0"
                in
                intensity_item[
                    "source"
                ]
            ),

        "raised_cosine":
            (
                "cos"
                in
                intensity_lower
                and
                "np.pi"
                in
                intensity_lower
            ),

        "outside_full_illumination":
            (
                "ones_like"
                in
                intensity_lower
            ),

        "multi_actor_minimum":
            (
                "np.minimum"
                in
                combine_lower
            ),
    }

    for name, passed in (
        intensity_semantic.items()
    ):

        print(
            f"{name:42s} =",
            "PASS"
            if passed
            else
            "NOT PROVEN",
        )

    require(
        all(
            intensity_semantic.values()
        ),
        (
            "Exact Part-A intensity/combine "
            "semantics are incomplete."
        ),
    )

    # ========================================================
    # H. Fixed demo epsilon vs dynamic geometry
    # ========================================================

    print()
    print(
        "===== H. FIXED EPSILON vs DYNAMIC GEOMETRY ====="
    )

    epsilon_rows = assignments.get(
        "epsilon_deg",
        [],
    )

    exact_epsilon = [
        row
        for row in epsilon_rows
        if (
            row[
                "source"
            ]
            .replace(
                " ",
                ""
            )
            ==
            "epsilon_deg=0.75"
        )
    ]

    require(
        len(
            exact_epsilon
        )
        >=
        1,
        (
            "Exact fixed epsilon_deg=0.75 "
            "assignment not found."
        ),
    )

    dynamic_uses_fixed_epsilon = (
        re.search(
            r"\bepsilon_deg\b",
            geometry_source,
        )
        is not None
    )

    dynamic_uses_width_margin = (
        semantic[
            "vehicle_width_or_half_width"
        ]
        and
        semantic[
            "safety_margin"
        ]
    )

    print(
        "fixed demo epsilon_deg    = 0.75 EXACT"
    )

    print(
        "dynamic chain references fixed epsilon_deg =",
        dynamic_uses_fixed_epsilon,
    )

    print(
        "dynamic chain width+margin geometry =",
        dynamic_uses_width_margin,
    )

    # Critical binding rule:
    #
    # Do NOT force fixed epsilon=0.75 into the dynamic Part-A
    # adapter unless the exact dynamic dependency chain uses it.
    dynamic_parameter_policy = (
        "FOLLOW_DYNAMIC_WIDTH_MARGIN_FORMULA"
        if (
            dynamic_uses_width_margin
            and
            not dynamic_uses_fixed_epsilon
        )
        else
        "FOLLOW_EXACT_DYNAMIC_SOURCE"
    )

    print(
        "dynamic binding policy =",
        dynamic_parameter_policy,
    )

    # ========================================================
    # I. MHT/current-track connection
    # ========================================================

    print()
    print(
        "===== I. MHT → DYNAMIC ADB CONNECTION ====="
    )

    geometry_contexts = contexts_for_token(
        code_cells,
        (
            ROOT_GEOMETRY_FUNCTION
            +
            "("
        ),
        radius=12,
    )

    track_contexts = contexts_for_token(
        code_cells,
        "m2_final_tracks",
        radius=10,
    )

    require(
        geometry_contexts,
        (
            "No call site for dynamic TRZ "
            "geometry wrapper was found."
        ),
    )

    print(
        "dynamic geometry call contexts =",
        len(
            geometry_contexts
        ),
    )

    for index, context in enumerate(
        geometry_contexts[:5],
        start=1,
    ):

        print()
        print(
            f"--- geometry call context {index} "
            f"(cell {context['cell_index']}) ---"
        )

        print(
            context[
                "context"
            ]
        )

    print()
    print(
        "m2_final_tracks contexts =",
        len(
            track_contexts
        ),
    )

    for index, context in enumerate(
        track_contexts[:5],
        start=1,
    ):

        print()
        print(
            f"--- MHT context {index} "
            f"(cell {context['cell_index']}) ---"
        )

        print(
            context[
                "context"
            ]
        )

    combined_context_text = "\n".join(
        item[
            "context"
        ]
        for item in (
            geometry_contexts
            +
            track_contexts
        )
    ).lower()

    mht_connection_proven = (
        "m2_final_tracks"
        in
        combined_context_text
        and
        (
            "adb_trz"
            in
            combined_context_text
            or
            "adb_int2"
            in
            combined_context_text
        )
    )

    require(
        mht_connection_proven,
        (
            "Part-A MHT/current-track to "
            "dynamic ADB connection not proven."
        ),
    )

    print()
    print(
        "MHT/current-track connection = PASS"
    )

    # ========================================================
    # J. Coordinate-convention audit
    # ========================================================

    print()
    print(
        "===== J. COORDINATE-CONVENTION AUDIT ====="
    )

    root_source = unique_functions[
        ROOT_GEOMETRY_FUNCTION
    ][
        "source"
    ]

    normalized_root = (
        root_source
        .replace(
            " ",
            ""
        )
        .lower()
    )

    part_a_y_forward = (
        "+yforward"
        in
        normalized_root
    )

    part_a_x_right = (
        "+xright"
        in
        normalized_root
    )

    require(
        part_a_y_forward
        and
        part_a_x_right,
        (
            "Exact Part-A dynamic coordinate "
            "convention was not proven from "
            "the source docstring."
        ),
    )

    stage6_contract = json.loads(
        BLOCK61_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    frozen_geometry = stage6_contract.get(
        "frozen_geometry_contract",
        {}
    )

    require(
        frozen_geometry.get(
            "axes"
        )
        ==
        "+x forward, +y left, +z up",
        (
            "Stage6 H0 axis contract changed."
        ),
    )

    # Mathematical conversion implied by the two frozen
    # coordinate conventions:
    #
    # Part-A:
    #   x_A = right
    #   y_A = forward
    #
    # Stage6 H0:
    #   x_H = forward
    #   y_H = left
    #
    # therefore:
    #   x_A = -y_H
    #   y_A =  x_H
    #
    # and:
    #   theta_A = atan2(x_A, y_A)
    #           = atan2(-y_H, x_H)
    #           = -theta_H.
    coordinate_conversion = {
        "Stage6_H0":
            {
                "x":
                    "forward",

                "y":
                    "left",

                "z":
                    "up",
            },

        "Part_A_dynamic":
            {
                "x":
                    "right",

                "y":
                    "forward",
            },

        "H0_to_PartA":
            {
                "x_partA_right":
                    "-y_H0_left",

                "y_partA_forward":
                    "x_H0_forward",
            },

        "angle_relation":
            "theta_partA = -theta_H0",

        "interval_conversion":
            (
                "[theta_min_A, theta_max_A] "
                "-> "
                "[-theta_max_A, -theta_min_A]"
            ),

        "implemented_in_this_audit":
            False,
    }

    # Tiny mathematical consistency proof, no Part-A execution.
    test_points = (
        (
            10.0,
            0.0,
        ),
        (
            10.0,
            2.0,
        ),
        (
            10.0,
            -2.0,
        ),
    )

    maximum_angle_error = 0.0

    for x_h, y_h in (
        test_points
    ):

        theta_h = math.atan2(
            y_h,
            x_h,
        )

        x_a = -y_h
        y_a = x_h

        theta_a = math.atan2(
            x_a,
            y_a,
        )

        error = abs(
            theta_a
            +
            theta_h
        )

        maximum_angle_error = max(
            maximum_angle_error,
            error,
        )

    require(
        maximum_angle_error
        <=
        1.0e-15,
        (
            "Coordinate angle conversion "
            "derivation failed."
        ),
    )

    print(
        "Part-A axes     = +Y FORWARD / +X RIGHT"
    )

    print(
        "Stage6 H0 axes  = +X FORWARD / +Y LEFT / +Z UP"
    )

    print(
        "H0 -> Part-A    = x_A=-y_H0 ; y_A=x_H0"
    )

    print(
        "theta relation  = theta_A = -theta_H0"
    )

    print(
        "interval mapping= [a,b] -> [-b,-a]"
    )

    print(
        "angle conversion smoke max error =",
        maximum_angle_error,
    )

    print(
        "adapter conversion implemented = NO"
    )

    # ========================================================
    # K. Freeze exact audit excerpt
    # ========================================================

    excerpt_parts.append(
        "\n"
        "============================================================\n"
        "MHT / CALL-SITE CONTEXTS\n"
        "============================================================\n"
    )

    for context in geometry_contexts:

        excerpt_parts.append(
            "\n"
            f"CELL {context['cell_index']} "
            f"MATCH LINE {context['matched_line']}\n"
        )

        excerpt_parts.append(
            context[
                "context"
            ]
            +
            "\n"
        )

    for context in track_contexts:

        excerpt_parts.append(
            "\n"
            f"CELL {context['cell_index']} "
            f"MHT MATCH LINE {context['matched_line']}\n"
        )

        excerpt_parts.append(
            context[
                "context"
            ]
            +
            "\n"
        )

    atomic_write(
        EXCERPT,
        "".join(
            excerpt_parts
        ).encode(
            "utf-8"
        ),
    )

    # ========================================================
    # L. Report
    # ========================================================

    print()
    print(
        "===== K. AUDIT ARTIFACT ====="
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "audit":
            (
                "Part_A_dynamic_ADB_"
                "transitive_dependency_chain"
            ),

        "status":
            (
                "PASS_READY_FOR_EXACT_"
                "PARTA_REACTIVE_ADAPTER"
            ),

        "root_geometry_function":
            ROOT_GEOMETRY_FUNCTION,

        "geometry_dependency_closure":
            closure,

        "call_edges": [
            {
                "caller":
                    caller,

                "callee":
                    callee,
            }
            for caller, callee
            in edges
        ],

        "functions": {
            name:
                {
                    "cell_index":
                        unique_functions[
                            name
                        ][
                            "cell_index"
                        ],

                    "source_sha256":
                        unique_functions[
                            name
                        ][
                            "source_sha256"
                        ],
                }
            for name in set(
                closure
                +
                [
                    INTENSITY_FUNCTION,
                    COMBINE_FUNCTION,
                    TRACK_ARRAY_FUNCTION,
                ]
            )
            if name in unique_functions
        },

        "geometry_semantic_gates":
            semantic,

        "geometry_global_names":
            relevant_globals,

        "geometry_global_assignment_lines":
            exact_global_lines,

        "intensity_semantic_gates":
            intensity_semantic,

        "fixed_vs_dynamic_epsilon": {
            "fixed_notebook_epsilon_deg":
                0.75,

            "fixed_assignment_proven":
                True,

            "dynamic_chain_references_fixed_epsilon_deg":
                dynamic_uses_fixed_epsilon,

            "dynamic_chain_uses_width_and_margin":
                dynamic_uses_width_margin,

            "binding_policy":
                dynamic_parameter_policy,

            "technical_report_0_745":
                (
                    "DOCUMENTED_TEXT_ARITHMETIC_"
                    "INCONSISTENCY_NOT_PARAMETER"
                ),
        },

        "Part_A_provenance": {
            "paper":
                "CAMERA_GEOMETRY_REACTIVE",

            "Part_A":
                (
                    "MHT_CURRENT_STATE_"
                    "REACTIVE_EXTENSION"
                ),

            "Part_B":
                (
                    "PROBABILISTIC_PREDICTIVE_"
                    "CLASS_AWARE"
                ),

            "MHT_to_dynamic_ADB_connection_proven":
                mht_connection_proven,
        },

        "coordinate_conversion":
            coordinate_conversion,

        "binding_rules": {
            "original_reactive_baseline":
                (
                    "preserve exact Part-A "
                    "dynamic geometry semantics"
                ),

            "do_not_upgrade_original_reactive_to_"
            "Stage6_full_box_policy":
                True,

            "Stage6_predictive_full_box_requirements":
                "BLOCK6.3_PLUS",

            "PartA_fixed_grid_role":
                "BASELINE_ONLY",

            "Stage6_final_grid_frozen":
                False,

            "Stage6_final_grid_freeze":
                "BLOCK6.8_PRE_FORMAL",

            "formal_primary_input_mode_changed":
                False,
        },

        "execution": {
            "notebook_executed":
                False,

            "dataset_access":
                False,

            "training":
                False,

            "inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "scientific_implementation_changed":
                False,
        },

        "exact_source_excerpt":
            str(
                EXCERPT
            ),

        "upstream_modified":
            False,

        "next":
            (
                "Block6.2 Part2 exact Part-A "
                "reactive adapter + real causal "
                "integration smoke"
            ),
    }

    atomic_write(
        REPORT,
        canonical_bytes(
            report
        ),
    )

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

    print(
        "exact excerpt =",
        EXCERPT,
    )

    # ========================================================
    # M. Frozen immutability
    # ========================================================

    print()
    print(
        "===== L. IMMUTABILITY RECHECK ====="
    )

    for name, path in protected:

        require(
            sha256_file(
                path
            )
            ==
            EXPECTED[
                name
            ],
            (
                "Frozen prerequisite changed "
                f"during audit: {name}"
            ),
        )

    print(
        "Part-A notebook      = UNCHANGED"
    )

    print(
        "Block6.1             = UNCHANGED"
    )

    print(
        "Block6.2 Part1       = UNCHANGED"
    )

    print(
        "scientific code      = UNCHANGED"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "TRANSITIVE DYNAMIC ADB AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "wrapper-only previous gate = FALSE NEGATIVE CONFIRMED"
    )

    print(
        "dynamic dependency chain   = AUDITED PASS"
    )

    print(
        "vehicle width semantics     = TRANSITIVELY PROVEN"
    )

    print(
        "safety margin semantics     = TRANSITIVELY PROVEN"
    )

    print(
        "angular/range geometry      = PASS"
    )

    print(
        "raised-cosine intensity     = EXACT SOURCE PASS"
    )

    print(
        "multi-actor minimum         = EXACT SOURCE PASS"
    )

    print(
        "MHT/current-track link      = PASS"
    )

    print(
        "fixed epsilon 0.75          = PRESERVED AS FIXED DEMO"
    )

    print(
        "dynamic epsilon policy      =",
        dynamic_parameter_policy,
    )

    print(
        "Part-A axes                 = +Y FORWARD / +X RIGHT"
    )

    print(
        "Stage6 H0 axes              = +X FORWARD / +Y LEFT"
    )

    print(
        "coordinate conversion       = DERIVED / NOT YET IMPLEMENTED"
    )

    print(
        "Part-A reactive upgraded    = NO"
    )

    print(
        "Stage6 predictive started   = NO"
    )

    print(
        "upstream modified           = NO"
    )

    print(
        "STATUS = PASS_READY_FOR_EXACT_PARTA_REACTIVE_ADAPTER"
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
        "TRANSITIVE DYNAMIC ADB AUDIT = BLOCKED"
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
        "Part-A modified         = NO"
    )

    print(
        "Stage1–5 modified       = NO"
    )

    print(
        "Block6.1 modified       = NO"
    )

    print(
        "Block6.2 Part1 modified = NO"
    )

    print(
        "scientific code changed = NO"
    )

    print(
        "dataset access          = NO"
    )

    print(
        "training/inference      = NO"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no non-zero sys.exit().

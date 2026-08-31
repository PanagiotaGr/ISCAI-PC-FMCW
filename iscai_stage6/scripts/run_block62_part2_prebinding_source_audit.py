from __future__ import annotations

import ast
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

NOTEBOOK = (
    ROOT
    / "part_a_reference/ISCAI_pc_fmcw/"
      "notebooks/ISCAI_PC_FMCW.ipynb"
)

PART_A_REFERENCE = (
    ROOT
    / "iscai_stage0/reports/stage0/"
      "part_a_frozen_reference.json"
)

BLOCK60_SOURCE_AUDIT = (
    S6
    / "artifacts/block60/"
      "part_a_adb_source_audit.json"
)

BLOCK60_ADAPTER_CONTRACT = (
    S6
    / "configs/"
      "part_a_reactive_adb_adapter_contract.json"
)

BLOCK61_CLOSURE = (
    S6
    / "reports/block61_closure.json"
)

BLOCK62_PART1_CONTRACT = (
    S6
    / "configs/"
      "block62_static_reactive_kernel_contract.json"
)

BLOCK62_PART1_REPORT = (
    S6
    / "reports/"
      "block62_part1_static_reactive_kernel.json"
)

REPORT = (
    S6
    / "reports/"
      "block62_part2_prebinding_source_audit.json"
)

SOURCE_EXCERPT = (
    S6
    / "artifacts/block62/"
      "part_a_adb_exact_source_excerpt.txt"
)


EXPECTED_SHA = {
    "notebook":
        (
            "b5a80a6d3441de6d571db4f65b4a43e"
            "d4052cc2b3ccba935ad31b5dd51316ef3"
        ),

    "part_a_reference":
        (
            "fcdfc0a7b9c14fa9447ceb8563a6f20a"
            "c277706223388dc9aa8a1edf416c5e2b"
        ),

    "block60_source_audit":
        (
            "5b3b8aacaa6c8947c6d9d57264a90899"
            "4a6a8d3193179a75bcd4f6b4697b0e64"
        ),

    "block60_adapter_contract":
        (
            "d552e9d766ec356fbf57079a647a7b3c"
            "5e882a6e153c8b9e5f06c8b207539e4c"
        ),

    "block61_closure":
        (
            "1d0f8781d365efd09614c8fb5a06bdbd"
            "a8f56c1c673b0ddab8b40810de270ac1"
        ),

    "block62_part1_contract":
        (
            "df4cf1e8fd18f3bc9cd48cbae4a6c209"
            "1e37313cbc1a5b2cefe48bd340eb5d60"
        ),

    "block62_part1_report":
        (
            "865c81eeb689a1b0c7e41b08117c9197"
            "c19e5cac76de147a07898b17c2224477"
        ),
}

EXPECTED_GIT_COMMIT = (
    "44d62e3478e3818d1757b00971890f844cb032f7"
)

PART_A_REPO = (
    ROOT
    / "part_a_reference/ISCAI_pc_fmcw"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    tmp.write_bytes(data)
    os.replace(tmp, path)


def cell_source(cell) -> str:
    source = cell.get("source", "")

    if isinstance(source, list):
        return "".join(source)

    return str(source)


def safe_python_for_ast(source: str) -> str:
    """
    Preserve line numbers but neutralize common
    notebook-only syntax such as %, ! and ?.
    """
    output = []

    for line in source.splitlines():

        stripped = line.lstrip()

        if (
            stripped.startswith("%")
            or stripped.startswith("!")
            or stripped.startswith("?")
        ):
            indent = line[:len(line) - len(stripped)]
            output.append(
                indent + "# NOTEBOOK_MAGIC " + stripped
            )
        else:
            output.append(line)

    return "\n".join(output)


UNRESOLVED = object()


def safe_eval_expr(node, env):
    """
    Tiny static numeric evaluator.

    This does NOT execute notebook code. Only simple
    literals/arithmetic and a small whitelist of
    NumPy/math constructors is interpreted.
    """

    try:
        return ast.literal_eval(node)
    except BaseException:
        pass

    if isinstance(node, ast.Name):
        return env.get(
            node.id,
            UNRESOLVED,
        )

    if isinstance(node, ast.UnaryOp):
        value = safe_eval_expr(
            node.operand,
            env,
        )

        if value is UNRESOLVED:
            return UNRESOLVED

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return +value

        return UNRESOLVED

    if isinstance(node, ast.BinOp):
        left = safe_eval_expr(
            node.left,
            env,
        )
        right = safe_eval_expr(
            node.right,
            env,
        )

        if (
            left is UNRESOLVED
            or right is UNRESOLVED
        ):
            return UNRESOLVED

        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
        except BaseException:
            return UNRESOLVED

        return UNRESOLVED

    if isinstance(node, ast.Attribute):

        if (
            isinstance(node.value, ast.Name)
            and node.attr == "pi"
            and node.value.id in {
                "np",
                "numpy",
                "math",
            }
        ):
            return math.pi

        return UNRESOLVED

    if isinstance(node, ast.Call):

        func = ""

        try:
            func = ast.unparse(
                node.func
            )
        except BaseException:
            return UNRESOLVED

        args = [
            safe_eval_expr(
                item,
                env,
            )
            for item in node.args
        ]

        if any(
            item is UNRESOLVED
            for item in args
        ):
            return UNRESOLVED

        kwargs = {}

        for kw in node.keywords:

            if kw.arg is None:
                return UNRESOLVED

            value = safe_eval_expr(
                kw.value,
                env,
            )

            if value is UNRESOLVED:
                return UNRESOLVED

            kwargs[
                kw.arg
            ] = value

        try:
            if func in {
                "float",
                "np.float64",
                "numpy.float64",
            }:
                return float(args[0])

            if func in {
                "int",
                "np.int64",
                "numpy.int64",
            }:
                return int(args[0])

            if func in {
                "math.radians",
                "np.deg2rad",
                "numpy.deg2rad",
            }:
                return np.deg2rad(
                    args[0]
                )

            if func in {
                "math.degrees",
                "np.rad2deg",
                "numpy.rad2deg",
            }:
                return np.rad2deg(
                    args[0]
                )

            if func in {
                "np.arange",
                "numpy.arange",
            }:
                value = np.arange(
                    *args,
                    **kwargs,
                )

                if value.size <= 100000:
                    return value

            if func in {
                "np.linspace",
                "numpy.linspace",
            }:
                value = np.linspace(
                    *args,
                    **kwargs,
                )

                if value.size <= 100000:
                    return value

            if func in {
                "np.array",
                "numpy.array",
                "np.asarray",
                "numpy.asarray",
            }:
                value = np.asarray(
                    args[0]
                )

                if value.size <= 100000:
                    return value

        except BaseException:
            return UNRESOLVED

    return UNRESOLVED


def summarize_value(value):

    if isinstance(
        value,
        np.ndarray,
    ):

        result = {
            "kind": "ndarray",
            "shape": list(value.shape),
            "size": int(value.size),
        }

        if (
            value.size
            and np.issubdtype(
                value.dtype,
                np.number,
            )
        ):
            finite = np.asarray(
                value,
                dtype=float,
            )

            if np.all(
                np.isfinite(finite)
            ):
                result.update({
                    "minimum":
                        float(
                            finite.min()
                        ),

                    "maximum":
                        float(
                            finite.max()
                        ),

                    "first":
                        float(
                            finite.flat[0]
                        ),

                    "last":
                        float(
                            finite.flat[-1]
                        ),
                })

                if (
                    finite.ndim == 1
                    and finite.size >= 2
                ):
                    diff = np.diff(
                        finite
                    )

                    if np.allclose(
                        diff,
                        diff[0],
                        rtol=1e-11,
                        atol=1e-13,
                    ):
                        result[
                            "uniform_step"
                        ] = float(
                            diff[0]
                        )

        return result

    if isinstance(
        value,
        np.generic,
    ):
        value = value.item()

    if isinstance(
        value,
        (
            int,
            float,
            str,
            bool,
        ),
    ) or value is None:
        return {
            "kind":
                type(value).__name__,
            "value":
                value,
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return {
            "kind":
                type(value).__name__,
            "value":
                list(value),
        }

    return {
        "kind":
            type(value).__name__,
        "repr":
            repr(value)[:500],
    }


def source_hash(text: str) -> str:
    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.2 PART 2/2 — PRE-BINDING SOURCE AUDIT"
    )
    print(
        "READ-ONLY CANONICAL PART-A NOTEBOOK INSPECTION"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Frozen prerequisite seal
    # --------------------------------------------------------

    print()
    print(
        "===== A. FROZEN PRE-BINDING SEAL ====="
    )

    gates = (
        (
            "notebook",
            NOTEBOOK,
        ),
        (
            "part_a_reference",
            PART_A_REFERENCE,
        ),
        (
            "block60_source_audit",
            BLOCK60_SOURCE_AUDIT,
        ),
        (
            "block60_adapter_contract",
            BLOCK60_ADAPTER_CONTRACT,
        ),
        (
            "block61_closure",
            BLOCK61_CLOSURE,
        ),
        (
            "block62_part1_contract",
            BLOCK62_PART1_CONTRACT,
        ),
        (
            "block62_part1_report",
            BLOCK62_PART1_REPORT,
        ),
    )

    hashes = {}

    for name, path in gates:

        require(
            path.is_file(),
            f"Missing prerequisite: {path}",
        )

        actual = sha256_file(
            path
        )

        hashes[
            name
        ] = actual

        require(
            actual
            ==
            EXPECTED_SHA[
                name
            ],
            (
                f"{name} SHA mismatch\n"
                f"expected={EXPECTED_SHA[name]}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{name:28s} = EXACT PASS"
        )

    git = subprocess.run(
        [
            "git",
            "-C",
            str(PART_A_REPO),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        git.returncode == 0,
        (
            "Could not read Part-A git HEAD:\n"
            + git.stdout
        ),
    )

    git_head = git.stdout.strip()

    require(
        git_head
        ==
        EXPECTED_GIT_COMMIT,
        (
            "Part-A git commit changed.\n"
            f"expected={EXPECTED_GIT_COMMIT}\n"
            f"actual={git_head}"
        ),
    )

    print(
        "Part-A git HEAD              = EXACT PASS"
    )

    # --------------------------------------------------------
    # B. Load canonical notebook, never execute
    # --------------------------------------------------------

    print()
    print(
        "===== B. CANONICAL NOTEBOOK READ-ONLY LOAD ====="
    )

    notebook = json.loads(
        NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )

    cells = notebook.get(
        "cells",
        [],
    )

    code_cells = [
        (
            index,
            cell,
            cell_source(cell),
        )
        for index, cell
        in enumerate(cells)
        if cell.get(
            "cell_type"
        ) == "code"
    ]

    require(
        len(code_cells) > 0,
        "Canonical notebook has no code cells.",
    )

    print(
        "notebook SHA256       =",
        hashes["notebook"],
    )
    print(
        "total cells           =",
        len(cells),
    )
    print(
        "code cells            =",
        len(code_cells),
    )
    print(
        "notebook execution    = NO"
    )
    print(
        "source modification   = NO"
    )

    # --------------------------------------------------------
    # C. Semantic ADB cell discovery
    # --------------------------------------------------------

    print()
    print(
        "===== C. ADB SOURCE CELL DISCOVERY ====="
    )

    strong_tokens = (
        "shadow",
        "illumination",
        "r_shadow",
        "r_transition",
        "margin_radial",
        "l_transition",
        "np.minimum",
        "theta_shadow",
    )

    supporting_tokens = (
        "adb",
        "theta_grid",
        "r_grid",
        "atan2",
        "arctan2",
        "epsilon",
        "eps",
        "intensity",
        "fig. 3",
        "fig 3",
        "mht",
    )

    candidates = []

    for index, cell, source in code_cells:

        lower = source.lower()

        strong_hits = [
            token
            for token in strong_tokens
            if token in lower
        ]

        supporting_hits = [
            token
            for token in supporting_tokens
            if token in lower
        ]

        if (
            strong_hits
            or
            len(
                supporting_hits
            ) >= 2
        ):
            candidates.append({
                "cell_index":
                    index,

                "source_sha256":
                    source_hash(
                        source
                    ),

                "strong_hits":
                    strong_hits,

                "supporting_hits":
                    supporting_hits,

                "source":
                    source,
            })

    require(
        candidates,
        (
            "No ADB source cells discovered "
            "in exact frozen notebook."
        ),
    )

    print(
        "candidate ADB cells =",
        [
            item[
                "cell_index"
            ]
            for item in candidates
        ],
    )

    print(
        "candidate count     =",
        len(candidates),
    )

    # --------------------------------------------------------
    # D. Static assignment interpretation
    # --------------------------------------------------------

    print()
    print(
        "===== D. STATIC PARAMETER EXTRACTION ====="
    )

    env = {}
    assignment_origin = {}
    assignment_expression = {}
    parse_failures = []

    for index, cell, source in code_cells:

        python_source = (
            safe_python_for_ast(
                source
            )
        )

        try:
            tree = ast.parse(
                python_source
            )

        except SyntaxError as exc:

            parse_failures.append({
                "cell_index":
                    index,

                "message":
                    str(exc),
            })

            continue

        for node in tree.body:

            target_name = None
            value_node = None

            if isinstance(
                node,
                ast.Assign,
            ):
                if (
                    len(node.targets) == 1
                    and isinstance(
                        node.targets[0],
                        ast.Name,
                    )
                ):
                    target_name = (
                        node.targets[0].id
                    )
                    value_node = (
                        node.value
                    )

            elif isinstance(
                node,
                ast.AnnAssign,
            ):
                if isinstance(
                    node.target,
                    ast.Name,
                ):
                    target_name = (
                        node.target.id
                    )
                    value_node = (
                        node.value
                    )

            if (
                target_name is None
                or value_node is None
            ):
                continue

            try:
                expression = ast.unparse(
                    value_node
                )
            except BaseException:
                expression = (
                    "<unparse-failed>"
                )

            assignment_expression[
                target_name
            ] = expression

            assignment_origin[
                target_name
            ] = index

            value = safe_eval_expr(
                value_node,
                env,
            )

            if value is not UNRESOLVED:
                env[
                    target_name
                ] = value

    relevant_pattern = re.compile(
        r"("
        r"theta|angle|"
        r"epsilon|eps|"
        r"shadow|"
        r"transition|"
        r"margin|"
        r"r_grid|"
        r"range_grid|"
        r"width|w_half|"
        r"intensity|"
        r"adb"
        r")",
        flags=re.IGNORECASE,
    )

    candidate_indices = {
        item[
            "cell_index"
        ]
        for item in candidates
    }

    relevant_assignments = {}

    for name, expression in (
        assignment_expression.items()
    ):

        if not relevant_pattern.search(
            name
        ):
            continue

        origin = assignment_origin[
            name
        ]

        # Prefer assignments originating in semantic ADB
        # cells. Include a few grid-like variables even if
        # the declaration occurs immediately before them.
        if (
            origin not in candidate_indices
            and not re.search(
                r"theta_grid|r_grid|range_grid",
                name,
                re.IGNORECASE,
            )
        ):
            continue

        entry = {
            "cell_index":
                origin,

            "expression":
                expression,
        }

        if name in env:
            entry[
                "statically_resolved"
            ] = True

            entry[
                "value"
            ] = summarize_value(
                env[name]
            )

        else:
            entry[
                "statically_resolved"
            ] = False

        relevant_assignments[
            name
        ] = entry

    print(
        "relevant assignments =",
        len(
            relevant_assignments
        ),
    )

    # Print compact high-value values only.
    for name in sorted(
        relevant_assignments
    ):

        item = relevant_assignments[
            name
        ]

        if item[
            "statically_resolved"
        ]:
            print(
                f"{name} | cell={item['cell_index']} | "
                f"{item['value']}"
            )
        else:
            print(
                f"{name} | cell={item['cell_index']} | "
                f"EXPR={item['expression'][:160]}"
            )

    # --------------------------------------------------------
    # E. Exact relevant source lines
    # --------------------------------------------------------

    print()
    print(
        "===== E. EXACT SOURCE-LINE EVIDENCE ====="
    )

    line_pattern = re.compile(
        r"("
        r"theta_grid|r_grid|"
        r"shadow|transition|"
        r"epsilon|eps|"
        r"margin_radial|"
        r"l_transition|"
        r"w_half|width|"
        r"atan2|arctan2|"
        r"minimum|"
        r"illumination|adb|"
        r"mht"
        r")",
        flags=re.IGNORECASE,
    )

    exact_lines = []

    for item in candidates:

        index = item[
            "cell_index"
        ]

        for line_number, line in enumerate(
            item["source"].splitlines(),
            start=1,
        ):
            if line_pattern.search(
                line
            ):
                exact_lines.append({
                    "cell_index":
                        index,

                    "line_number":
                        line_number,

                    "text":
                        line.rstrip(),
                })

    require(
        exact_lines,
        (
            "ADB cells found but no exact "
            "parameter/formula source lines found."
        ),
    )

    excerpt_rows = [
        (
            "CANONICAL PART-A NOTEBOOK ADB SOURCE EXCERPT\n"
            f"notebook_sha256={hashes['notebook']}\n"
            f"git_commit={git_head}\n"
            "NOTE: source inspection only; notebook was not executed.\n"
            "\n"
        )
    ]

    last_cell = None

    for row in exact_lines:

        cell_index = row[
            "cell_index"
        ]

        if cell_index != last_cell:
            excerpt_rows.append(
                "\n"
                "------------------------------------------------------------\n"
                f"CELL {cell_index}\n"
                f"SOURCE_SHA256="
                f"{next(x['source_sha256'] for x in candidates if x['cell_index'] == cell_index)}\n"
                "------------------------------------------------------------\n"
            )
            last_cell = cell_index

        excerpt_rows.append(
            f"{row['line_number']:04d}: "
            f"{row['text']}\n"
        )

    atomic_write(
        SOURCE_EXCERPT,
        "".join(
            excerpt_rows
        ).encode("utf-8"),
    )

    print(
        "exact matched source lines =",
        len(exact_lines),
    )

    print(
        "full source excerpt        =",
        SOURCE_EXCERPT,
    )

    # Do not flood the terminal.
    print(
        "compact evidence preview:"
    )

    for row in exact_lines[:40]:
        print(
            f"  C{row['cell_index']:02d}:"
            f"L{row['line_number']:03d} | "
            f"{row['text'][:180]}"
        )

    if len(exact_lines) > 40:
        print(
            "  ...",
            len(exact_lines) - 40,
            "additional exact lines saved to artifact",
        )

    # --------------------------------------------------------
    # F. Semantic evidence / discrepancy discovery
    # --------------------------------------------------------

    print()
    print(
        "===== F. PARAMETER-BINDING READINESS ====="
    )

    combined_source = "\n".join(
        item["source"]
        for item in candidates
    )

    lower = combined_source.lower()

    evidence = {
        "angular_grid":
            (
                "theta_grid"
                in lower
            ),

        "range_grid":
            (
                "r_grid"
                in lower
                or
                "range_grid"
                in lower
            ),

        "angular_shadow":
            (
                "shadow"
                in lower
            ),

        "radial_shadow_or_threshold":
            (
                "r_shadow"
                in lower
                or
                "rmin"
                in lower
                or
                "r_min"
                in lower
            ),

        "transition":
            (
                "transition"
                in lower
            ),

        "full_width_or_edge_geometry":
            (
                (
                    "atan2"
                    in lower
                    or
                    "arctan2"
                    in lower
                )
                and
                (
                    "width"
                    in lower
                    or
                    "w_half"
                    in lower
                )
            ),

        "multi_actor_minimum":
            (
                "minimum"
                in lower
                or
                "np.min("
                in lower
            ),

        "MHT_reactive_connection":
            (
                "mht"
                in lower
            ),
    }

    for name, passed in (
        evidence.items()
    ):
        print(
            f"{name:32s} =",
            "FOUND"
            if passed
            else
            "NOT PROVEN BY TOKEN AUDIT",
        )

    # Search specifically for epsilon-like assignments.
    epsilon_candidates = {
        name:
            item
        for name, item
        in relevant_assignments.items()
        if re.search(
            r"epsilon|eps",
            name,
            re.IGNORECASE,
        )
    }

    margin_candidates = {
        name:
            item
        for name, item
        in relevant_assignments.items()
        if re.search(
            r"margin|transition",
            name,
            re.IGNORECASE,
        )
    }

    print()
    print(
        "epsilon-like assignments =",
        list(
            sorted(
                epsilon_candidates
            )
        ),
    )

    print(
        "margin/transition assignments =",
        list(
            sorted(
                margin_candidates
            )
        ),
    )

    # This audit intentionally does not choose 0.75 vs 0.745.
    # It only exposes the exact notebook source.
    parameter_binding_performed = False

    # --------------------------------------------------------
    # G. Write deterministic audit report
    # --------------------------------------------------------

    print()
    print(
        "===== G. AUDIT ARTIFACT ====="
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "phase":
            "PART2_PRE_BINDING_SOURCE_AUDIT",

        "status":
            "PASS_READY_FOR_EXACT_PARAMETER_BINDING",

        "execution": {
            "notebook_executed":
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

            "parameter_binding_performed":
                parameter_binding_performed,
        },

        "frozen_identity": {
            "Part_A_git_commit":
                git_head,

            "notebook_path":
                str(
                    NOTEBOOK
                ),

            "notebook_sha256":
                hashes[
                    "notebook"
                ],

            "Block6_2_Part1_report_sha256":
                hashes[
                    "block62_part1_report"
                ],

            "Block6_2_Part1_contract_sha256":
                hashes[
                    "block62_part1_contract"
                ],
        },

        "source_semantics": {
            "paper_ADB":
                (
                    "camera_geometry_informed_"
                    "reactive_masking"
                ),

            "Part_A_extension":
                (
                    "MHT_current_state_"
                    "reactive_masking"
                ),

            "Part_B_contribution":
                (
                    "probabilistic_predictive_"
                    "class_aware_masking"
                ),
        },

        "candidate_cells": [
            {
                "cell_index":
                    item[
                        "cell_index"
                    ],

                "source_sha256":
                    item[
                        "source_sha256"
                    ],

                "strong_hits":
                    item[
                        "strong_hits"
                    ],

                "supporting_hits":
                    item[
                        "supporting_hits"
                    ],
            }
            for item in candidates
        ],

        "relevant_assignments":
            relevant_assignments,

        "semantic_evidence":
            evidence,

        "epsilon_candidates":
            epsilon_candidates,

        "margin_transition_candidates":
            margin_candidates,

        "exact_source_line_count":
            len(
                exact_lines
            ),

        "exact_source_excerpt":
            str(
                SOURCE_EXCERPT
            ),

        "AST_parse_failures": {
            "count":
                len(
                    parse_failures
                ),

            "cells":
                parse_failures,
        },

        "binding_policy": {
            "authoritative_source":
                (
                    "exact frozen canonical "
                    "Part-A notebook source"
                ),

            "technical_report_role":
                (
                    "interpretation_and_"
                    "cross_check"
                ),

            "do_not_arbitrarily_choose_"
            "epsilon_0_75_vs_0_745":
                True,

            "Part_A_grid_is_baseline_only":
                True,

            "Stage6_final_grid_frozen_here":
                False,

            "Stage6_final_grid_freeze":
                "BLOCK6.8_PRE_FORMAL",
        },

        "next":
            (
                "Block6.2 Part2 exact binding "
                "+ real causal reactive ADB smoke"
            ),

        "upstream_modified":
            False,
    }

    atomic_write(
        REPORT,
        canonical_json_bytes(
            report
        ),
    )

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

    # --------------------------------------------------------
    # H. Immutability recheck
    # --------------------------------------------------------

    print()
    print(
        "===== H. IMMUTABILITY RECHECK ====="
    )

    for name, path in gates:

        require(
            sha256_file(path)
            ==
            EXPECTED_SHA[name],
            (
                "Frozen prerequisite changed "
                f"during audit: {name}"
            ),
        )

    git_after = subprocess.run(
        [
            "git",
            "-C",
            str(PART_A_REPO),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        git_after.returncode == 0
        and
        git_after.stdout.strip()
        ==
        EXPECTED_GIT_COMMIT,
        "Part-A repository changed during audit.",
    )

    print(
        "Part-A notebook/repo = UNCHANGED"
    )
    print(
        "Block6.0/6.1        = UNCHANGED"
    )
    print(
        "Block6.2 Part1      = UNCHANGED"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "PRE-BINDING SOURCE AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part-A identity           = EXACT PASS"
    )
    print(
        "canonical notebook        = EXACT SHA PASS"
    )
    print(
        "notebook execution        = NO"
    )
    print(
        "ADB source cells          = DISCOVERED PASS"
    )
    print(
        "static parameter scan     = COMPLETE"
    )
    print(
        "exact source excerpt      = WRITTEN"
    )
    print(
        "0.75 vs 0.745 arbitrated  = NO / SOURCE EXPOSED"
    )
    print(
        "Part-A parameter binding  = NOT YET"
    )
    print(
        "Stage6 final grid freeze  = NO"
    )
    print(
        "dataset access            = NO"
    )
    print(
        "training/inference        = NO"
    )
    print(
        "formal evaluation/tuning  = NO"
    )
    print(
        "upstream modified         = NO"
    )
    print(
        "STATUS = PASS_READY_FOR_EXACT_PARAMETER_BINDING"
    )
    print(
        "report =",
        REPORT,
    )
    print(
        "source excerpt =",
        SOURCE_EXCERPT,
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
        "PRE-BINDING SOURCE AUDIT = BLOCKED"
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
        "scientific implementation changed = NO"
    )
    print(
        "Part-A modified       = NO"
    )
    print(
        "Stage1–5 modified     = NO"
    )
    print(
        "Block6.1 modified     = NO"
    )
    print(
        "Block6.2 Part1 modified= NO"
    )
    print(
        "dataset access        = NO"
    )
    print(
        "training/inference    = NO"
    )
    print(
        "formal evaluation     = NO"
    )
    print(
        "terminal remains open = YES"
    )

# Intentionally no non-zero sys.exit().

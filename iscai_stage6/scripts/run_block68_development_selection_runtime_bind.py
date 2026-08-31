from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

FINAL_SCRIPT = (
    S6
    / "scripts/"
      "run_block66_part3c_part2of2_final.py"
)

REPLAY_CONTRACT = (
    S6
    / "configs/"
      "block66_part3c_part2of2_deterministic_replay_contract.json"
)

PART3C_CLOSURE = (
    S6
    / "reports/"
      "block66_part3c_final_closure.json"
)

PART3C_FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part3c_final_freeze_manifest.json"
)

POCC_MANIFEST = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

PREDICTIVE_MATCHES = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

NI_DELTA = (
    S6
    / "configs/"
      "stage6_exact_noninferiority_deltas.json"
)

ACCEPTANCE = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

TIMESTAMP = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
)

OUTPUT = (
    S6
    / "reports/"
      "block68_development_selection_runtime_bind.json"
)


EXPECTED = {
    "prereg":
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    "final_script":
        "c8b87a64e5c5f6d1aba3d534fd0090772e24688f29c0425fb036457a5fa5de93",

    "replay_contract":
        "ff1816222564b519514e1fad342128010bd19ac31fe0291cd4b28045e339b5c1",

    "part3c_closure":
        "8413fd646a1bf18e15c12bf0441d1ddb1ea72ed142703020ac0d2c77dba0bb29",

    "part3c_freeze":
        "880c2efb0895f463aaf3350eed7730389a5871651d7ad6091cef052d95434738",

    "pocc_manifest":
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    "predictive_matches":
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    "ni_delta":
        "a77e548e060c314dd98a7220b0f9ed3aebd5d1978eb70f456b82920e792a29d4",

    "acceptance":
        "e103466b1a6eb62be4e6746029147cbda9671b05130acba5210a3ab7328e6b81",

    "timestamp":
        "51a3b56a119467001a9f4617e1e19dc82c548c0a30c79fc9c4a9c1f5ef27bf56",
}

EXPECTED_TESTS = 224


OPERATORS = {
    "threshold_occupancy_counts_strict_k",
    "compute_class_aware_margin",
    "angular_margin_cells",
    "dilate_mask_theta",
    "predictive_mask_to_illumination",
    "compose_class_aware_illumination",
    "temporal_smooth_schedule",
    "apply_actuation_rate_limit",
}


RELEVANT_JSON_TOKENS = (
    "policy",
    "gamma",
    "threshold",
    "margin",
    "floor",
    "smoothing",
    "time_constant",
    "rho",
    "rate",
    "horizon",
    "sample",
    "pocc",
    "occupancy",
    "current_illumination",
    "current",
    "grid",
    "replay",
    "numeric",
    "selection",
    "selected",
    "formal",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
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


def exact(path: Path, expected, label):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(path)

    require(
        actual == expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl_schema(path: Path, limit=3):
    """
    Schema only:
    parse at most first N JSON metadata rows.
    Never dereference any P_occ array/file path.
    """

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line in stream:
            if not line.strip():
                continue

            value = json.loads(line)

            require(
                isinstance(value, dict),
                f"Non-object JSONL row: {path}",
            )

            rows.append(value)

            if len(rows) >= limit:
                break

    require(
        rows,
        f"No rows in {path}",
    )

    return rows


def schema_paths(value, prefix=""):
    result = set()

    if isinstance(value, dict):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            result.add(path)

            result.update(
                schema_paths(
                    child,
                    path,
                )
            )

    elif isinstance(value, list):

        marker = (
            f"{prefix}[]"
        )

        result.add(marker)

        for child in value[:1]:
            result.update(
                schema_paths(
                    child,
                    marker,
                )
            )

    return result


def flatten_relevant(value, prefix=""):
    rows = []

    if isinstance(value, dict):

        for key in sorted(value):

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten_relevant(
                    value[key],
                    path,
                )
            )

    elif isinstance(value, list):

        for index, child in enumerate(value):

            rows.extend(
                flatten_relevant(
                    child,
                    f"{prefix}[{index}]",
                )
            )

    else:

        lower = prefix.lower()

        if any(
            token in lower
            for token in RELEVANT_JSON_TOKENS
        ):
            rows.append(
                {
                    "path":
                        prefix,

                    "value":
                        value,
                }
            )

    return rows


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def find_operator_calls(source):
    tree = ast.parse(
        source,
        filename=str(
            FINAL_SCRIPT
        ),
    )

    calls = []

    parents = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):

        if not isinstance(node, ast.Call):
            continue

        name = call_name(
            node.func
        )

        if name not in OPERATORS:
            continue

        keywords = {}

        for keyword in node.keywords:

            if keyword.arg is None:
                continue

            keywords[
                keyword.arg
            ] = ast.unparse(
                keyword.value
            )

        positional = [
            ast.unparse(argument)
            for argument in node.args
        ]

        # Find containing function, if any.
        current = node

        containing_function = None

        while current in parents:

            current = parents[
                current
            ]

            if isinstance(
                current,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                containing_function = current.name
                break

        calls.append(
            {
                "function":
                    name,

                "line":
                    int(
                        node.lineno
                    ),

                "end_line":
                    int(
                        node.end_lineno
                        or
                        node.lineno
                    ),

                "containing_function":
                    containing_function,

                "positional":
                    positional,

                "keywords":
                    keywords,
            }
        )

    calls.sort(
        key=lambda row:
            row[
                "line"
            ]
    )

    return calls


def source_excerpt(
    source,
    *,
    start,
    end,
):
    lines = source.splitlines()

    start = max(
        1,
        start,
    )

    end = min(
        len(lines),
        end,
    )

    return "\n".join(
        f"L{line_number:05d}: "
        f"{lines[line_number - 1]}"
        for line_number
        in range(
            start,
            end + 1,
        )
    )


def assignment_inventory(
    source,
    variable_names,
):
    tree = ast.parse(
        source,
        filename=str(
            FINAL_SCRIPT
        ),
    )

    rows = []

    for node in ast.walk(tree):

        targets = []

        value = None

        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value

        elif isinstance(node, ast.AnnAssign):
            targets = [
                node.target
            ]
            value = node.value

        elif isinstance(node, ast.NamedExpr):
            targets = [
                node.target
            ]
            value = node.value

        else:
            continue

        if value is None:
            continue

        for target in targets:

            if not isinstance(
                target,
                ast.Name,
            ):
                continue

            name = target.id

            if name not in variable_names:
                continue

            rows.append(
                {
                    "variable":
                        name,

                    "line":
                        int(
                            node.lineno
                        ),

                    "expression":
                        ast.unparse(
                            value
                        ),
                }
            )

    rows.sort(
        key=lambda row:
            row[
                "line"
            ]
    )

    return rows


def scalar_argument_identifiers(calls):
    names = set()

    for call in calls:

        expressions = (
            list(
                call[
                    "positional"
                ]
            )
            +
            list(
                call[
                    "keywords"
                ].values()
            )
        )

        for expression in expressions:

            try:
                tree = ast.parse(
                    expression,
                    mode="eval",
                )

            except Exception:
                continue

            for node in ast.walk(tree):

                if isinstance(
                    node,
                    ast.Name,
                ):
                    names.add(
                        node.id
                    )

    return names


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
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-80:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} "
            f"tests, got {count}."
        ),
    )

    return count


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "DEVELOPMENT CLASS-AWARE POLICY-SELECTION RUNTIME BIND"
    )
    print(
        "EXACT INTEGRATION ROUTE / NO P_OCC ARRAY OPEN"
    )
    print(
        "============================================================"
    )

    # ========================================================
    # A. Exact frozen boundary
    # ========================================================

    print()
    print(
        "===== A. EXACT FROZEN BOUNDARY ====="
    )

    seals = {}

    for key, path in (
        ("prereg", PREREG),
        ("final_script", FINAL_SCRIPT),
        ("replay_contract", REPLAY_CONTRACT),
        ("part3c_closure", PART3C_CLOSURE),
        ("part3c_freeze", PART3C_FREEZE),
        ("pocc_manifest", POCC_MANIFEST),
        ("predictive_matches", PREDICTIVE_MATCHES),
        ("ni_delta", NI_DELTA),
        ("acceptance", ACCEPTANCE),
        ("timestamp", TIMESTAMP),
    ):

        seals[key] = exact(
            path,
            EXPECTED[key],
            key,
        )

        print(
            f"{key:22s} = EXACT PASS"
        )

    print(
        "P_occ arrays             = NOT OPENED"
    )
    print(
        "predictive performance   = NOT COMPUTED"
    )
    print(
        "numeric policy selected  = NO"
    )
    print(
        "formal evaluation        = NO"
    )

    # ========================================================
    # B. Prove there is no already-selected class policy
    # ========================================================

    print()
    print(
        "===== B. PART3C NUMERIC-SELECTION STATUS ====="
    )

    closure = read_json(
        PART3C_CLOSURE
    )

    freeze = read_json(
        PART3C_FREEZE
    )

    # Exact Part3C selection-status authority.
    #
    # The final closure does NOT expose
    # numeric_policy_selected as a top-level key.
    # Its authoritative scientific-boundary and
    # integration-smoke records explicitly state
    # that no numeric policy was selected.
    #
    # The final freeze independently exposes the
    # same status as a top-level boolean.

    require(
        isinstance(
            closure.get(
                "scientific_boundary"
            ),
            dict,
        ),
        (
            "Part3C closure scientific_boundary "
            "is missing."
        ),
    )

    require(
        closure[
            "scientific_boundary"
        ].get(
            "numeric_policy_selected"
        )
        is False,
        (
            "Part3C closure scientific boundary "
            "does not explicitly freeze "
            "numeric_policy_selected=False."
        ),
    )

    require(
        closure[
            "scientific_boundary"
        ].get(
            "policy_parameter_sweep_executed"
        )
        is False,
        (
            "Part3C closure says a policy "
            "parameter sweep was executed."
        ),
    )

    require(
        closure[
            "scientific_boundary"
        ].get(
            "formal_outcomes_read"
        )
        is False,
        (
            "Part3C closure says formal "
            "outcomes were read."
        ),
    )

    require(
        isinstance(
            closure.get(
                "integration_smoke"
            ),
            dict,
        ),
        (
            "Part3C closure integration_smoke "
            "is missing."
        ),
    )

    require(
        closure[
            "integration_smoke"
        ].get(
            "numeric_policy_selected"
        )
        is False,
        (
            "Part3C integration smoke does not "
            "explicitly freeze "
            "numeric_policy_selected=False."
        ),
    )

    require(
        closure[
            "integration_smoke"
        ].get(
            "development_objective_evaluated"
        )
        is False,
        (
            "Part3C integration smoke says the "
            "development objective was evaluated."
        ),
    )

    require(
        closure[
            "integration_smoke"
        ].get(
            "formal_outcomes_read"
        )
        is False,
        (
            "Part3C integration smoke says "
            "formal outcomes were read."
        ),
    )

    require(
        freeze.get(
            "numeric_policy_selected"
        )
        is False,
        (
            "Part3C final freeze does not "
            "explicitly freeze "
            "numeric_policy_selected=False."
        ),
    )

    require(
        freeze.get(
            "formal_outcomes_read"
        )
        is False,
        (
            "Part3C final freeze says "
            "formal outcomes were read."
        ),
    )

    print(
        "Part3C closure numeric_policy_selected = FALSE"
    )
    print(
        "Part3C freeze  numeric_policy_selected = FALSE"
    )
    print(
        "formal outcomes read                  = FALSE"
    )
    print(
        "interpretation = DEVELOPMENT SELECTION STILL PENDING"
    )

    # ========================================================
    # C. Exact integration operator bindings
    # ========================================================

    print()
    print(
        "===== C. PART3C FINAL INTEGRATION OPERATOR CALLS ====="
    )

    source = FINAL_SCRIPT.read_text(
        encoding="utf-8"
    )

    calls = find_operator_calls(
        source
    )

    found = {
        row[
            "function"
        ]
        for row in calls
    }

    require(
        OPERATORS.issubset(
            found
        ),
        (
            "Part3C final integration script "
            "does not expose all required "
            "runtime operators.\n"
            f"missing={sorted(OPERATORS - found)}"
        ),
    )

    for call in calls:

        print()
        print(
            f"{call['function']} "
            f"@ L{call['line']}"
        )

        print(
            " containing function =",
            call[
                "containing_function"
            ],
        )

        print(
            " positional =",
            call[
                "positional"
            ],
        )

        print(
            " keywords   =",
            call[
                "keywords"
            ],
        )

    # ========================================================
    # D. Exact source region of scientific integration
    # ========================================================

    print()
    print(
        "===== D. PART3C FINAL SCIENTIFIC INTEGRATION EXCERPT ====="
    )

    minimum_line = min(
        row[
            "line"
        ]
        for row in calls
    )

    maximum_line = max(
        row[
            "end_line"
        ]
        for row in calls
    )

    excerpt = source_excerpt(
        source,
        start=minimum_line - 100,
        end=maximum_line + 100,
    )

    print(
        excerpt
    )

    # ========================================================
    # E. Trace variables used by the operator calls
    # ========================================================

    print()
    print(
        "===== E. OPERATOR-ARGUMENT VARIABLE BINDINGS ====="
    )

    variable_names = (
        scalar_argument_identifiers(
            calls
        )
    )

    assignments = (
        assignment_inventory(
            source,
            variable_names,
        )
    )

    print(
        "argument identifiers =",
        sorted(
            variable_names
        ),
    )

    print()

    for row in assignments:

        print(
            f"L{row['line']:05d}",
            row[
                "variable"
            ],
            "=",
            row[
                "expression"
            ],
        )

    # ========================================================
    # F. Deterministic replay contract values
    # ========================================================

    print()
    print(
        "===== F. DETERMINISTIC REPLAY CONTRACT ====="
    )

    replay = read_json(
        REPLAY_CONTRACT
    )

    replay_relevant = (
        flatten_relevant(
            replay
        )
    )

    for row in replay_relevant:

        print(
            row[
                "path"
            ],
            "=",
            repr(
                row[
                    "value"
                ]
            ),
        )

    # ========================================================
    # G. Part3C closure relevant values
    # ========================================================

    print()
    print(
        "===== G. PART3C FINAL CLOSURE RELEVANT VALUES ====="
    )

    closure_relevant = (
        flatten_relevant(
            closure
        )
    )

    for row in closure_relevant:

        print(
            row[
                "path"
            ],
            "=",
            repr(
                row[
                    "value"
                ]
            ),
        )

    # ========================================================
    # H. P_occ + predictive-match schema only
    # ========================================================

    print()
    print(
        "===== H. P_OCC / PREDICTIVE MATCH SCHEMA ONLY ====="
    )

    pocc_rows = read_jsonl_schema(
        POCC_MANIFEST,
        limit=3,
    )

    predictive_rows = read_jsonl_schema(
        PREDICTIVE_MATCHES,
        limit=3,
    )

    pocc_schema = set()

    for row in pocc_rows:

        pocc_schema.update(
            schema_paths(
                row
            )
        )

    predictive_schema = set()

    for row in predictive_rows:

        predictive_schema.update(
            schema_paths(
                row
            )
        )

    print()
    print(
        "P_occ manifest key paths:"
    )

    for path in sorted(
        pocc_schema
    ):

        print(
            " ",
            path,
        )

    print()
    print(
        "predictive-match key paths:"
    )

    for path in sorted(
        predictive_schema
    ):

        print(
            " ",
            path,
        )

    # Important: only schemas were read.
    print()
    print(
        "P_occ ndarray/file payloads dereferenced = NO"
    )

    # ========================================================
    # I. Preregistered Stage-1 selection semantics
    # ========================================================

    print()
    print(
        "===== I. PREREGISTERED STAGE-1 GAMMA SELECTION ====="
    )

    prereg = read_json(
        PREREG
    )

    protocol = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_1_class_gamma"
        ]
    )

    print(
        json.dumps(
            protocol,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    threshold_parameterization = (
        prereg[
            "class_threshold_parameterization"
        ]
    )

    print()
    print(
        "threshold parameterization:"
    )

    print(
        json.dumps(
            threshold_parameterization,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )

    require(
        protocol[
            "vary"
        ]
        ==
        "k_c only",
        (
            "Stage-1 preregistration "
            "does not vary k_c only."
        ),
    )

    require(
        threshold_parameterization[
            "strict_comparison"
        ]
        ==
        ">",
        (
            "Strict threshold semantics changed."
        ),
    )

    # ========================================================
    # J. Scientific boundary
    # ========================================================

    print()
    print(
        "===== J. SCIENTIFIC BOUNDARY ====="
    )

    tests = run_regression()

    print(
        "P_occ JSON metadata rows = 3 SCHEMA ONLY"
    )
    print(
        "P_occ arrays             = NOT OPENED"
    )
    print(
        "future oracle metrics    = NOT COMPUTED"
    )
    print(
        "gamma sweep              = NOT EXECUTED"
    )
    print(
        "numeric gamma selected   = NO"
    )
    print(
        "margin/floor sweep       = NO"
    )
    print(
        "temporal/rate sweep      = NO"
    )
    print(
        "predictive performance   = NOT COMPUTED"
    )
    print(
        "formal evaluation        = NO"
    )
    print(
        "Stage6 regression        =",
        f"{tests} / {tests} PASS",
    )

    # ========================================================
    # K. Report
    # ========================================================

    result = {
        "stage":
            6,

        "block":
            "6.8_development_selection_runtime_bind",

        "status":
            "PASS_DEVELOPMENT_SELECTION_RUNTIME_BOUND",

        "conclusion": {
            "preexisting_selected_numeric_policy":
                False,

            "selection_is_still_required":
                True,

            "next_selection_stage":
                "stage_1_class_gamma",
        },

        "frozen_seals":
            seals,

        "part3c_final_operator_calls":
            calls,

        "operator_argument_assignments":
            assignments,

        "replay_contract_relevant":
            replay_relevant,

        "part3c_closure_relevant":
            closure_relevant,

        "pocc_manifest_schema":
            sorted(
                pocc_schema
            ),

        "predictive_match_schema":
            sorted(
                predictive_schema
            ),

        "stage1_gamma_protocol":
            protocol,

        "threshold_parameterization":
            threshold_parameterization,

        "scientific_boundary": {
            "P_occ_arrays_opened":
                False,

            "future_oracle_metrics_computed":
                False,

            "gamma_sweep_executed":
                False,

            "numeric_policy_selected":
                False,

            "predictive_performance_computed":
                False,

            "formal_evaluation":
                False,
        },

        "regression":
            tests,

        "next":
            (
                "execute preregistered development-only "
                "Stage-1 class-gamma selection using "
                "P_occ counts and constructed future "
                "full-box reference; freeze k_vehicle, "
                "k_pedestrian, k_cyclist before Stage-2"
            ),
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 DEVELOPMENT-SELECTION RUNTIME BIND — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "existing numeric policy    = NO"
    )
    print(
        "Part3C integration runtime = EXACT BOUND"
    )
    print(
        "threshold operator         = EXACT BOUND"
    )
    print(
        "margin/dilation route      = EXACT BOUND"
    )
    print(
        "class illumination route   = EXACT BOUND"
    )
    print(
        "temporal/rate route        = EXACT BOUND"
    )
    print(
        "P_occ schema               = BOUND"
    )
    print(
        "P_occ arrays               = NOT OPENED"
    )
    print(
        "Stage-1 gamma protocol     = EXACT PREREG BOUND"
    )
    print(
        "gamma sweep                = NOT YET EXECUTED"
    )
    print(
        "predictive outcomes        = NOT COMPUTED"
    )
    print(
        "formal evaluation          = NO"
    )
    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )
    print(
        "STATUS = "
        "PASS_DEVELOPMENT_SELECTION_RUNTIME_BOUND"
    )
    print(
        "report =",
        OUTPUT,
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
        "BLOCK 6.8 DEVELOPMENT-SELECTION RUNTIME BIND = BLOCKED"
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

    print()
    print(
        "P_occ arrays             = NOT OPENED BY THIS RUNNER"
    )
    print(
        "gamma sweep              = NOT EXECUTED"
    )
    print(
        "numeric gamma selected   = NO"
    )
    print(
        "predictive performance   = NOT COMPUTED"
    )
    print(
        "formal evaluation        = NO"
    )
    print(
        "Do not infer a winning policy "
        "from Part3C integration-smoke numerics."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

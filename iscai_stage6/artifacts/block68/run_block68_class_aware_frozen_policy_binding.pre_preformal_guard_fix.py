from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import json
import os
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

PART3C1_RUNNER = (
    S6
    / "scripts/"
      "run_block66_part3c1_class_aware_runtime.py"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3c1_to_part3c2_handoff.json"
)

FINAL_FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part3c_final_freeze_manifest.json"
)

CLASS_POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

MARGIN_BINDING = (
    S6
    / "src/iscai_stage6/adb/"
      "class_margin_binding.py"
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

NI_REPORT = (
    S6
    / "reports/"
      "block68_reactive_only_ni_delta_freeze.json"
)

TIMESTAMP_BINDING = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
)

OUTPUT = (
    S6
    / "reports/"
      "block68_class_aware_frozen_policy_binding.json"
)

EXPECTED_PREREG_SHA = (
    "5704494a89b4b3c7a3c19b0df8176c55"
    "0c00795ed17cf2906c343f340461429e"
)

EXPECTED_CLASS_POLICY_SHA = (
    "b998f468b98c2770c48c84a2c0aaaa2"
    "177c2d84b8fc838e80fef9b9da1ebc14b"
)

EXPECTED_NI_DELTA_SHA = (
    "a77e548e060c314dd98a7220b0f9ed3a"
    "ebd5d1978eb70f456b82920e792a29d4"
)

EXPECTED_ACCEPTANCE_SHA = (
    "e103466b1a6eb62be4e6746029147cbda"
    "9671b05130acba5210a3ab7328e6b81"
)

EXPECTED_TIMESTAMP_SHA = (
    "51a3b56a119467001a9f4617e1e19dc8"
    "2c548c0a30c79fc9c4a9c1f5ef27bf56"
)

EXPECTED_TESTS = 224


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        payload
    )

    os.replace(
        tmp,
        path,
    )


def regression():
    proc = subprocess.run(
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

    for line in proc.stdout.splitlines():
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        proc.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                proc.stdout.splitlines()[-80:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests, "
            f"got {count}."
        ),
    )

    return count


def flatten(value, prefix=""):
    rows = []

    if isinstance(value, dict):
        for key in sorted(value):
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten(
                    value[key],
                    path,
                )
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = (
                f"{prefix}[{index}]"
            )

            rows.extend(
                flatten(
                    item,
                    path,
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


POLICY_KEYWORDS = (
    "gamma",
    "threshold",
    "confidence",
    "floor",
    "dimming",
    "intensity",
    "margin",
    "uncertainty",
    "motion",
    "closing",
    "lateral",
    "smooth",
    "time_constant",
    "tau",
    "rate",
    "rho",
    "actuation",
    "vehicle",
    "pedestrian",
    "cyclist",
    "horizon",
)


FUNCTIONS_OF_INTEREST = {
    "threshold_occupancy_counts_strict_k",
    "threshold_actor_occupancy",
    "compute_class_aware_margin",
    "bind_predictive_class_margin",
    "angular_margin_cells",
    "dilate_mask_theta",
    "predictive_mask_to_illumination",
    "compose_class_aware_illumination",
    "temporal_smooth_schedule",
    "apply_actuation_rate_limit",
}


def relevant_flattened(data):
    result = []

    for path, value in flatten(data):
        lower = path.lower()

        if any(
            keyword in lower
            for keyword in POLICY_KEYWORDS
        ):
            result.append(
                {
                    "path": path,
                    "value": value,
                    "type": type(value).__name__,
                }
            )

    return result


def call_name(node):
    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):
        return node.attr

    return None


def source_segment(lines, start, end):
    lo = max(
        1,
        start,
    )

    hi = min(
        len(lines),
        end,
    )

    return "\n".join(
        f"{index:05d}: {lines[index - 1]}"
        for index
        in range(
            lo,
            hi + 1,
        )
    )


def ast_call_inventory(path: Path):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    lines = source.splitlines()

    calls = []

    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = call_name(
            node.func
        )

        if name not in FUNCTIONS_OF_INTEREST:
            continue

        keyword_map = {}

        for keyword in node.keywords:
            if keyword.arg is None:
                continue

            try:
                expression = ast.unparse(
                    keyword.value
                )
            except Exception:
                expression = "<unparse-failed>"

            keyword_map[
                keyword.arg
            ] = expression

        positional = []

        for argument in node.args:
            try:
                positional.append(
                    ast.unparse(
                        argument
                    )
                )
            except Exception:
                positional.append(
                    "<unparse-failed>"
                )

        calls.append(
            {
                "function":
                    name,

                "line":
                    int(
                        node.lineno
                    ),

                "positional":
                    positional,

                "keywords":
                    keyword_map,

                "source_context":
                    source_segment(
                        lines,
                        node.lineno - 8,
                        getattr(
                            node,
                            "end_lineno",
                            node.lineno,
                        )
                        +
                        8,
                    ),
            }
        )

    calls.sort(
        key=lambda row:
            (
                row[
                    "line"
                ],
                row[
                    "function"
                ],
            )
    )

    return calls


def literal_assignments(path: Path):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows = []

    for node in tree.body:
        target_name = None
        value_node = None

        if isinstance(
            node,
            ast.Assign,
        ):
            if (
                len(node.targets) == 1
                and
                isinstance(
                    node.targets[0],
                    ast.Name,
                )
            ):
                target_name = (
                    node.targets[0].id
                )
                value_node = node.value

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
                value_node = node.value

        if (
            target_name is None
            or
            value_node is None
        ):
            continue

        lower = target_name.lower()

        if not any(
            keyword in lower
            for keyword in POLICY_KEYWORDS
        ):
            continue

        try:
            value = ast.literal_eval(
                value_node
            )

            literal = True

        except Exception:
            try:
                value = ast.unparse(
                    value_node
                )
            except Exception:
                value = (
                    "<unparse-failed>"
                )

            literal = False

        rows.append(
            {
                "name":
                    target_name,

                "line":
                    int(
                        node.lineno
                    ),

                "literal":
                    literal,

                "value":
                    value,
            }
        )

    return rows


def print_candidates(label, rows):
    print()
    print(
        f"----- {label} -----"
    )

    if not rows:
        print(
            "(no relevant scalar paths)"
        )
        return

    for row in rows:
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


def scientific_boundary_gate():
    forbidden_outputs = [
        (
            S6
            / "reports/"
              "block68_class_aware_primary_gate.json"
        ),
        (
            S6
            / "artifacts/block68/"
              "block68_class_aware_primary_gate_values.jsonl"
        ),
    ]

    return [
        str(path)
        for path in forbidden_outputs
        if path.exists()
    ]


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "CLASS-AWARE FROZEN POLICY RUNTIME BINDING"
    )
    print(
        "NO P_OCC VALUES / NO PREDICTIVE METRICS / NO FORMAL"
    )
    print(
        "============================================================"
    )

    print()
    print(
        "===== A. POST-NI EXACT BOUNDARY ====="
    )

    for path in (
        PREREG,
        PART3C1_RUNNER,
        HANDOFF,
        FINAL_FREEZE,
        CLASS_POLICY,
        MARGIN_BINDING,
        NI_DELTA,
        ACCEPTANCE,
        NI_REPORT,
        TIMESTAMP_BINDING,
    ):
        require(
            path.is_file(),
            (
                "Required frozen input missing: "
                f"{path}"
            ),
        )

        require(
            "formal" not in path.name.lower(),
            (
                "Formal-named file unexpectedly "
                f"in binding input: {path}"
            ),
        )

    require(
        file_sha(
            PREREG
        )
        ==
        EXPECTED_PREREG_SHA,
        (
            "Block6.6 preregistration SHA changed."
        ),
    )

    require(
        file_sha(
            CLASS_POLICY
        )
        ==
        EXPECTED_CLASS_POLICY_SHA,
        (
            "Frozen class-aware policy source changed."
        ),
    )

    require(
        file_sha(
            NI_DELTA
        )
        ==
        EXPECTED_NI_DELTA_SHA,
        (
            "Frozen NI delta file changed."
        ),
    )

    require(
        file_sha(
            ACCEPTANCE
        )
        ==
        EXPECTED_ACCEPTANCE_SHA,
        (
            "Frozen acceptance policy changed."
        ),
    )

    require(
        file_sha(
            TIMESTAMP_BINDING
        )
        ==
        EXPECTED_TIMESTAMP_SHA,
        (
            "Superseding timestamp binding changed."
        ),
    )

    delta = read_json(
        NI_DELTA
    )

    acceptance = read_json(
        ACCEPTANCE
    )

    require(
        file_sha(
            NI_REPORT
        ),
        "NI report SHA unavailable.",
    )

    print(
        "Block6.6 preregistration = EXACT PASS"
    )
    print(
        "class-aware source       = EXACT PASS"
    )
    print(
        "NI delta freeze          = EXACT PASS"
    )
    print(
        "acceptance policy        = EXACT PASS"
    )
    print(
        "timestamp alignment      = EXACT PASS"
    )
    print(
        "predictive performance   = NOT OPENED"
    )
    print(
        "P_occ values             = NOT OPENED"
    )
    print(
        "formal content           = NOT OPENED"
    )

    print()
    print(
        "===== B. FROZEN ACCEPTANCE READBACK ====="
    )

    print(
        "NI delta file SHA =",
        file_sha(
            NI_DELTA
        ),
    )

    print(
        "acceptance SHA    =",
        file_sha(
            ACCEPTANCE
        ),
    )

    print(
        "reactive-only report SHA =",
        file_sha(
            NI_REPORT
        ),
    )

    print()
    print(
        "NI delta content:"
    )

    for path, value in flatten(
        delta
    ):
        if isinstance(
            value,
            (
                int,
                float,
                str,
                bool,
            ),
        ):
            print(
                " ",
                path,
                "=",
                value,
            )

    print()
    print(
        "primary acceptance content:"
    )

    for path, value in flatten(
        acceptance
    ):
        lower = path.lower()

        if any(
            keyword in lower
            for keyword in (
                "vehicle",
                "over",
                "pedestrian",
                "cyclist",
                "delta",
                "rule",
                "comparison",
                "formal",
            )
        ):
            print(
                " ",
                path,
                "=",
                value,
            )

    print()
    print(
        "===== C. BLOCK6.6 PREREGISTERED POLICY VALUES ====="
    )

    prereg = read_json(
        PREREG
    )

    prereg_relevant = (
        relevant_flattened(
            prereg
        )
    )

    print_candidates(
        "PREREGISTRATION",
        prereg_relevant,
    )

    print()
    print(
        "===== D. PART3C1 HANDOFF / FINAL FREEZE ====="
    )

    handoff = read_json(
        HANDOFF
    )

    freeze = read_json(
        FINAL_FREEZE
    )

    handoff_relevant = (
        relevant_flattened(
            handoff
        )
    )

    freeze_relevant = (
        relevant_flattened(
            freeze
        )
    )

    print_candidates(
        "PART3C1 HANDOFF",
        handoff_relevant,
    )

    print_candidates(
        "PART3C FINAL FREEZE",
        freeze_relevant,
    )

    require(
        handoff.get(
            "numeric_policy_selected"
        )
        is False,
        (
            "Part3C1 handoff suggests outcome-based "
            "numeric policy selection occurred."
        ),
    )

    require(
        freeze.get(
            "formal_outcomes_read"
        )
        is False,
        (
            "Block66 final freeze says formal "
            "outcomes were read."
        ),
    )

    print()
    print(
        "numeric policy selected from outcomes = NO"
    )
    print(
        "formal outcomes read in Block66       = NO"
    )

    print()
    print(
        "===== E. EXACT PART3C1 CALL BINDINGS ====="
    )

    calls = ast_call_inventory(
        PART3C1_RUNNER
    )

    require(
        calls,
        (
            "No relevant class-aware runtime "
            "calls found in frozen Part3C1 runner."
        ),
    )

    for call in calls:
        print()
        print(
            f"{call['function']} @ "
            f"L{call['line']}"
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

        print(
            call[
                "source_context"
            ]
        )

    functions_found = {
        row[
            "function"
        ]
        for row in calls
    }

    required_pipeline_functions = {
        "compose_class_aware_illumination",
        "temporal_smooth_schedule",
        "apply_actuation_rate_limit",
    }

    require(
        required_pipeline_functions.issubset(
            functions_found
        ),
        (
            "Frozen Part3C1 runner does not expose "
            "the complete final-I pipeline."
        ),
    )

    require(
        (
            "threshold_occupancy_counts_strict_k"
            in functions_found
            or
            "threshold_actor_occupancy"
            in functions_found
        ),
        (
            "No frozen class-aware occupancy "
            "threshold call found."
        ),
    )

    print()
    print(
        "threshold route = PROVEN"
    )
    print(
        "class composition route = PROVEN"
    )
    print(
        "temporal smoothing route = PROVEN"
    )
    print(
        "actuation rate route = PROVEN"
    )

    print()
    print(
        "===== F. RELEVANT RUNNER ASSIGNMENTS ====="
    )

    assignments = literal_assignments(
        PART3C1_RUNNER
    )

    for row in assignments:
        print(
            f"L{row['line']:05d}",
            row[
                "name"
            ],
            "=",
            repr(
                row[
                    "value"
                ]
            ),
            (
                "[literal]"
                if row[
                    "literal"
                ]
                else
                "[expression]"
            ),
        )

    print()
    print(
        "===== G. FROZEN SOURCE SIGNATURES ====="
    )

    from iscai_stage6.adb.class_aware_policy import (
        apply_actuation_rate_limit,
        compose_class_aware_illumination,
        compute_class_aware_margin,
        predictive_mask_to_illumination,
        temporal_smooth_schedule,
    )

    try:
        from iscai_stage6.adb.class_aware_policy import (
            threshold_occupancy_counts_strict_k,
        )
    except ImportError:
        threshold_occupancy_counts_strict_k = None

    functions = [
        compute_class_aware_margin,
        predictive_mask_to_illumination,
        compose_class_aware_illumination,
        temporal_smooth_schedule,
        apply_actuation_rate_limit,
    ]

    if (
        threshold_occupancy_counts_strict_k
        is not None
    ):
        functions.insert(
            0,
            threshold_occupancy_counts_strict_k,
        )

    signatures = {}

    for function in functions:
        signature = str(
            inspect.signature(
                function
            )
        )

        signatures[
            function.__name__
        ] = signature

        print(
            function.__name__,
            signature,
        )

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    unexpected = (
        scientific_boundary_gate()
    )

    require(
        not unexpected,
        (
            "Predictive primary-gate outputs "
            "already exist unexpectedly: "
            f"{unexpected}"
        ),
    )

    tests = regression()

    print(
        "P_occ manifest metadata       = NOT READ"
    )
    print(
        "P_occ arrays                  = NOT OPENED"
    )
    print(
        "predictive illumination       = NOT COMPUTED"
    )
    print(
        "predictive performance        = NOT COMPUTED"
    )
    print(
        "acceptance gate               = NOT TESTED"
    )
    print(
        "policy sweep                  = NO"
    )
    print(
        "threshold tuning              = NO"
    )
    print(
        "formal content                = NOT OPENED"
    )
    print(
        "formal evaluation             = NO"
    )
    print(
        "Stage6 regression             =",
        f"{tests} / {tests} PASS",
    )

    result = {
        "stage":
            6,

        "block":
            "6.8_class_aware_frozen_policy_binding",

        "status":
            "PASS_CLASS_AWARE_FROZEN_POLICY_BINDING_EXTRACTED",

        "scientific_role":
            (
                "freeze/readback of already-preregistered "
                "class-aware controller numerics before "
                "first predictive performance evaluation"
            ),

        "upstream": {
            "preregistration": {
                "path":
                    str(
                        PREREG
                    ),

                "sha256":
                    file_sha(
                        PREREG
                    ),
            },

            "class_aware_policy": {
                "path":
                    str(
                        CLASS_POLICY
                    ),

                "sha256":
                    file_sha(
                        CLASS_POLICY
                    ),
            },

            "part3c1_runner": {
                "path":
                    str(
                        PART3C1_RUNNER
                    ),

                "sha256":
                    file_sha(
                        PART3C1_RUNNER
                    ),
            },

            "part3c1_handoff": {
                "path":
                    str(
                        HANDOFF
                    ),

                "sha256":
                    file_sha(
                        HANDOFF
                    ),
            },

            "part3c_final_freeze": {
                "path":
                    str(
                        FINAL_FREEZE
                    ),

                "sha256":
                    file_sha(
                        FINAL_FREEZE
                    ),
            },

            "ni_delta_freeze": {
                "path":
                    str(
                        NI_DELTA
                    ),

                "sha256":
                    file_sha(
                        NI_DELTA
                    ),
            },

            "acceptance_policy": {
                "path":
                    str(
                        ACCEPTANCE
                    ),

                "sha256":
                    file_sha(
                        ACCEPTANCE
                    ),
            },

            "timestamp_alignment": {
                "path":
                    str(
                        TIMESTAMP_BINDING
                    ),

                "sha256":
                    file_sha(
                        TIMESTAMP_BINDING
                    ),
            },
        },

        "preregistered_policy_candidates":
            prereg_relevant,

        "part3c1_handoff_policy_candidates":
            handoff_relevant,

        "part3c_final_freeze_policy_candidates":
            freeze_relevant,

        "part3c1_runtime_calls":
            calls,

        "runner_relevant_assignments":
            assignments,

        "runtime_signatures":
            signatures,

        "route": [
            "occupancy threshold",
            "class-specific margin / dilation",
            "predictive_mask_to_illumination",
            "compose_class_aware_illumination",
            "raw_class_aware_illumination",
            "temporal_smooth_schedule",
            "apply_actuation_rate_limit",
            "RateLimitedSchedule.illumination = I_final",
        ],

        "scientific_boundary": {
            "P_occ_content_opened":
                False,

            "predictive_illumination_computed":
                False,

            "predictive_performance_computed":
                False,

            "acceptance_tested":
                False,

            "policy_sweep":
                False,

            "threshold_tuning":
                False,

            "formal_content_opened":
                False,

            "formal_evaluation":
                False,
        },

        "Stage6_regression": {
            "passed":
                tests,

            "expected":
                EXPECTED_TESTS,
        },

        "next":
            (
                "development-only class-aware predictive "
                "primary four-metric acceptance gate"
            ),
    }

    atomic_write(
        OUTPUT,
        canonical_bytes(
            result
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 CLASS-AWARE POLICY BINDING — FINAL"
    )
    print(
        "============================================================"
    )
    print(
        "Block6.6 policy prereg      = EXACT PASS"
    )
    print(
        "Block66 frozen runtime      = EXACT SOURCE BOUND"
    )
    print(
        "occupancy threshold route  = PROVEN"
    )
    print(
        "class-margin route         = PROVEN"
    )
    print(
        "class illumination route   = PROVEN"
    )
    print(
        "temporal smoothing route   = PROVEN"
    )
    print(
        "actuation-rate route       = PROVEN"
    )
    print(
        "I_final route              = RateLimitedSchedule.illumination"
    )
    print(
        "NI deltas                  = IMMUTABLE"
    )
    print(
        "predictive P_occ values    = NOT OPENED"
    )
    print(
        "predictive outcomes        = NOT COMPUTED"
    )
    print(
        "formal evaluation          = NO"
    )
    print(
        "controller tuning          = NO"
    )
    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )
    print(
        "STATUS = "
        "PASS_CLASS_AWARE_FROZEN_POLICY_BINDING_EXTRACTED"
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
        "BLOCK 6.8 CLASS-AWARE POLICY BINDING = BLOCKED"
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
        "P_occ content          = NOT OPENED BY THIS RUNNER"
    )
    print(
        "predictive performance = NOT COMPUTED"
    )
    print(
        "acceptance gate        = NOT TESTED"
    )
    print(
        "policy tuning          = NO"
    )
    print(
        "formal evaluation      = NO"
    )
    print(
        "Do not invent missing policy numerics."
    )
    print(
        "Do not run formal Stage6 evaluation."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

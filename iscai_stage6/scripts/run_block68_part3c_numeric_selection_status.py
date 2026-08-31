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

CLOSURE = (
    S6
    / "reports/"
      "block66_part3c_final_closure.json"
)

FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part3c_final_freeze_manifest.json"
)

REPLAY = (
    S6
    / "configs/"
      "block66_part3c_part2of2_deterministic_replay_contract.json"
)

FINAL_SCRIPT = (
    S6
    / "scripts/"
      "run_block66_part3c_part2of2_final.py"
)

PART3C1_REPORT = (
    S6
    / "reports/"
      "block66_part3c1_class_aware_runtime.json"
)

PART3C1_HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3c1_to_part3c2_handoff.json"
)

OUTPUT = (
    S6
    / "reports/"
      "block68_part3c_numeric_selection_status.json"
)


EXPECTED = {
    "closure":
        "8413fd646a1bf18e15c12bf0441d1ddb1ea72ed142703020ac0d2c77dba0bb29",

    "freeze":
        "880c2efb0895f463aaf3350eed7730389a5871651d7ad6091cef052d95434738",

    "replay":
        "ff1816222564b519514e1fad342128010bd19ac31fe0291cd4b28045e339b5c1",

    "final_script":
        "c8b87a64e5c5f6d1aba3d534fd0090772e24688f29c0425fb036457a5fa5de93",

    "part3c1_report":
        "ef4c18367dfe26ceac3a7f55f725f5450c7a3a0fd28811aa565472d9b06073a2",

    "part3c1_handoff":
        "5dccff7f48403bffcb3eab2f787bb09f96a4cce41a5e8c286050026fea58f813",
}

EXPECTED_TESTS = 224

TOKENS = (
    "numeric",
    "policy",
    "select",
    "selection",
    "selected",
    "sweep",
    "gamma",
    "margin",
    "floor",
    "smoothing",
    "rate",
    "rho",
    "formal",
    "candidate",
    "smoke",
    "integration",
    "replay",
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def exact(path, expected, label):
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

            rows.extend(
                flatten(
                    item,
                    f"{prefix}[{index}]",
                )
            )

    else:

        rows.append(
            {
                "path":
                    prefix,

                "value":
                    value,

                "python_type":
                    type(value).__name__,
            }
        )

    return rows


def relevant(rows):
    result = []

    for row in rows:

        lower = row[
            "path"
        ].lower()

        if any(
            token in lower
            for token in TOKENS
        ):
            result.append(row)

    return result


def print_rows(label, rows):
    print()
    print(
        f"----- {label} -----"
    )

    for row in rows:
        print(
            row["path"],
            "=",
            repr(
                row["value"]
            ),
            "| type =",
            row["python_type"],
        )


def source_context(path):
    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    selected = set()

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        lower = line.lower()

        if any(
            token in lower
            for token in (
                "numeric_policy_selected",
                "numeric_gamma_selected",
                "numeric_margin_selected",
                "numeric_floor_selected",
                "numeric_smoothing_selected",
                "numeric_rate_selected",
                "policy_sweep",
                "numeric selection",
                "numeric_policy_selection",
            )
        ):

            for item in range(
                max(
                    1,
                    line_number - 5,
                ),
                min(
                    len(lines),
                    line_number + 5,
                )
                +
                1,
            ):
                selected.add(item)

    return [
        {
            "line":
                line_number,

            "text":
                lines[
                    line_number - 1
                ],
        }
        for line_number
        in sorted(selected)
    ]


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
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
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
        "PART3C NUMERIC-SELECTION STATUS DIAGNOSTIC"
    )
    print(
        "READ ONLY / ZERO P_OCC / ZERO PERFORMANCE"
    )
    print(
        "============================================================"
    )

    # ========================================================
    # A. Exact inputs
    # ========================================================

    print()
    print(
        "===== A. EXACT FROZEN INPUTS ====="
    )

    seals = {}

    for key, path in (
        ("closure", CLOSURE),
        ("freeze", FREEZE),
        ("replay", REPLAY),
        ("final_script", FINAL_SCRIPT),
        ("part3c1_report", PART3C1_REPORT),
        ("part3c1_handoff", PART3C1_HANDOFF),
    ):

        seals[key] = exact(
            path,
            EXPECTED[key],
            key,
        )

        print(
            f"{key:18s} = EXACT PASS"
        )

    # ========================================================
    # B. Exact offending value
    # ========================================================

    print()
    print(
        "===== B. EXACT `numeric_policy_selected` VALUES ====="
    )

    closure = read_json(
        CLOSURE
    )

    freeze = read_json(
        FREEZE
    )

    part3c1_report = read_json(
        PART3C1_REPORT
    )

    part3c1_handoff = read_json(
        PART3C1_HANDOFF
    )

    for label, data in (
        ("Part3C1 report", part3c1_report),
        ("Part3C1 handoff", part3c1_handoff),
        ("Part3C final closure", closure),
        ("Part3C final freeze", freeze),
    ):

        present = (
            "numeric_policy_selected"
            in
            data
        )

        value = data.get(
            "numeric_policy_selected",
            "<MISSING>",
        )

        print()
        print(
            label
        )

        print(
            "  key present =",
            present,
        )

        print(
            "  value       =",
            repr(value),
        )

        print(
            "  type        =",
            type(value).__name__,
        )

        print(
            "  `is False`  =",
            value is False,
        )

        print(
            "  `is True`   =",
            value is True,
        )

    # ========================================================
    # C. Relevant closure fields
    # ========================================================

    print()
    print(
        "===== C. FINAL CLOSURE SELECTION-RELATED LEAVES ====="
    )

    closure_rows = relevant(
        flatten(
            closure
        )
    )

    print_rows(
        "PART3C FINAL CLOSURE",
        closure_rows,
    )

    # ========================================================
    # D. Relevant final-freeze fields
    # ========================================================

    print()
    print(
        "===== D. FINAL FREEZE SELECTION-RELATED LEAVES ====="
    )

    freeze_rows = relevant(
        flatten(
            freeze
        )
    )

    print_rows(
        "PART3C FINAL FREEZE",
        freeze_rows,
    )

    # ========================================================
    # E. Replay-contract selection-related metadata
    # ========================================================

    print()
    print(
        "===== E. DETERMINISTIC REPLAY CONTRACT STATUS ====="
    )

    replay = read_json(
        REPLAY
    )

    replay_rows = relevant(
        flatten(
            replay
        )
    )

    print_rows(
        "DETERMINISTIC REPLAY CONTRACT",
        replay_rows,
    )

    # ========================================================
    # F. Exact final-script source context
    # ========================================================

    print()
    print(
        "===== F. PART3C FINAL SOURCE SELECTION CONTEXT ====="
    )

    context = source_context(
        FINAL_SCRIPT
    )

    require(
        context,
        (
            "No numeric-selection-related "
            "source context found."
        ),
    )

    for row in context:
        print(
            f"L{row['line']:05d}: "
            f"{row['text']}"
        )

    # ========================================================
    # G. Interpretation — classification only
    # ========================================================

    print()
    print(
        "===== G. STATUS CLASSIFICATION ====="
    )

    closure_present = (
        "numeric_policy_selected"
        in
        closure
    )

    closure_value = closure.get(
        "numeric_policy_selected"
    )

    freeze_present = (
        "numeric_policy_selected"
        in
        freeze
    )

    freeze_value = freeze.get(
        "numeric_policy_selected"
    )

    if (
        closure_value is True
        or
        freeze_value is True
    ):

        classification = (
            "EXPLICIT_SELECTED_POLICY_FLAG_TRUE"
        )

        next_action = (
            "bind exact already-selected policy; "
            "DO NOT rerun development selection"
        )

    elif (
        closure_value is False
        and
        freeze_value is False
    ):

        classification = (
            "EXPLICIT_NO_SELECTED_POLICY"
        )

        next_action = (
            "development staged selection remains pending"
        )

    elif (
        closure_value in (
            0,
            "NO",
            "No",
            "no",
            "FALSE",
            "False",
            "false",
        )
        and
        freeze_value is False
    ):

        classification = (
            "SEMANTIC_FALSE_SCHEMA_REPRESENTATION"
        )

        next_action = (
            "repair boolean identity gate only; "
            "do not change scientific semantics"
        )

    elif not closure_present:

        classification = (
            "CLOSURE_KEY_MISSING"
        )

        next_action = (
            "bind selection status from explicit "
            "scientific_boundary/freeze authority"
        )

    else:

        classification = (
            "AMBIGUOUS_SELECTION_STATUS"
        )

        next_action = (
            "inspect printed relevant leaves; "
            "do not sweep and do not infer"
        )

    print(
        "classification =",
        classification,
    )

    print(
        "next action     =",
        next_action,
    )

    # ========================================================
    # H. Scientific boundary
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    tests = run_regression()

    print(
        "P_occ manifest values       = NOT OPENED"
    )

    print(
        "P_occ arrays                = NOT OPENED"
    )

    print(
        "gamma sweep                 = NOT EXECUTED"
    )

    print(
        "predictive illumination     = NOT COMPUTED"
    )

    print(
        "predictive performance      = NOT COMPUTED"
    )

    print(
        "acceptance gate             = NOT TESTED"
    )

    print(
        "policy numerics changed     = NO"
    )

    print(
        "formal evaluation           = NO"
    )

    print(
        "Stage6 regression           =",
        f"{tests} / {tests} PASS",
    )

    # ========================================================
    # I. Report
    # ========================================================

    result = {
        "stage":
            6,

        "block":
            "6.8_Part3C_numeric_selection_status",

        "status":
            "PASS_PART3C_NUMERIC_SELECTION_STATUS_DIAGNOSED",

        "classification":
            classification,

        "next_action":
            next_action,

        "exact_values": {
            "Part3C1_report": {
                "present":
                    (
                        "numeric_policy_selected"
                        in
                        part3c1_report
                    ),

                "value":
                    part3c1_report.get(
                        "numeric_policy_selected"
                    ),

                "type":
                    type(
                        part3c1_report.get(
                            "numeric_policy_selected"
                        )
                    ).__name__,
            },

            "Part3C1_handoff": {
                "present":
                    (
                        "numeric_policy_selected"
                        in
                        part3c1_handoff
                    ),

                "value":
                    part3c1_handoff.get(
                        "numeric_policy_selected"
                    ),

                "type":
                    type(
                        part3c1_handoff.get(
                            "numeric_policy_selected"
                        )
                    ).__name__,
            },

            "Part3C_final_closure": {
                "present":
                    closure_present,

                "value":
                    closure_value,

                "type":
                    type(
                        closure_value
                    ).__name__,
            },

            "Part3C_final_freeze": {
                "present":
                    freeze_present,

                "value":
                    freeze_value,

                "type":
                    type(
                        freeze_value
                    ).__name__,
            },
        },

        "closure_relevant_leaves":
            closure_rows,

        "freeze_relevant_leaves":
            freeze_rows,

        "replay_relevant_leaves":
            replay_rows,

        "final_script_selection_context":
            context,

        "seals":
            seals,

        "scientific_boundary": {
            "P_occ_opened":
                False,

            "gamma_sweep_executed":
                False,

            "predictive_performance_computed":
                False,

            "acceptance_tested":
                False,

            "policy_modified":
                False,

            "formal_evaluation":
                False,
        },

        "regression":
            tests,
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
        "BLOCK 6.8 PART3C NUMERIC-SELECTION STATUS — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "classification          =",
        classification,
    )

    print(
        "P_occ arrays            = NOT OPENED"
    )

    print(
        "gamma sweep             = NOT EXECUTED"
    )

    print(
        "predictive performance  = NOT COMPUTED"
    )

    print(
        "policy modification     = NO"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "Stage6 regression       =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_PART3C_NUMERIC_SELECTION_STATUS_DIAGNOSED"
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
        "BLOCK 6.8 PART3C NUMERIC-SELECTION STATUS = BLOCKED"
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
        "P_occ arrays            = NOT OPENED"
    )

    print(
        "gamma sweep             = NOT EXECUTED"
    )

    print(
        "predictive performance  = NOT COMPUTED"
    )

    print(
        "policy modification     = NO"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "Do not normalize or reinterpret "
        "numeric_policy_selected manually."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

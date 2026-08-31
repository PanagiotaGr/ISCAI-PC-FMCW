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

from iscai_stage6.adb import geometry


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

RUNNER = (
    S6
    / "scripts/"
      "run_block68_reactive_only_ni_delta_freeze.py"
)

PREOUTCOME_CONTRACT = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_scoring_contract.json"
)

GEOMETRY_BIND_REPORT = (
    S6
    / "reports/"
      "block68_part2b_part2_geometry_identity_bind.json"
)

LEDGER = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_decision_ledger.jsonl"
)

T0_MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

RAW_VALUES = (
    S6
    / "artifacts/block68/"
      "block68_reactive_only_ni_values.jsonl"
)

DELTA_FREEZE = (
    S6
    / "configs/"
      "stage6_exact_noninferiority_deltas.json"
)

ACCEPTANCE_POLICY = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

FINAL_REPORT = (
    S6
    / "reports/"
      "block68_reactive_only_ni_delta_freeze.json"
)

PROVENANCE_BINDING = (
    S6
    / "configs/"
      "stage6_future_gt_evaluator_provenance_binding.json"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_future_gt_evaluator_provenance_repair.json"
)

BACKUP = (
    S6
    / "artifacts/block68/provenance_repair/"
      "run_block68_reactive_only_ni_delta_freeze.pre_provenance_patch.py"
)


EXPECTED_PREOUTCOME_SHA = (
    "8364c7cdfa4b66d820823c804d45905a"
    "207f77f66b53f06a6383e7722b8a9970"
)

EXPECTED_LEDGER_SHA = (
    "ac2b954571cdd0166bbcfa5bb8b29567"
    "85d1e132288dbe4af7fe2d55adeafcaa"
)

EXPECTED_T0_MAPS_SHA = (
    "7c5d63c76c3d9ef74adbbd7a68e3ce74"
    "83b1dd80fd262cba06f6ce1909994993"
)

EXPECTED_TESTS = 224

BAD_ORIENTATION_SOURCE = (
    "WOMD_future_GT_evaluator_only"
)

CURRENT_STATE_SOURCE = (
    "WOMD_future_GT_evaluator_only"
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


def canonical_json_bytes(value):
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
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)

    os.replace(
        temporary,
        path,
    )


def write_once(path: Path, value):
    payload = canonical_json_bytes(value)

    if path.exists():

        require(
            path.read_bytes() == payload,
            (
                "Existing frozen provenance artifact "
                f"differs: {path}"
            ),
        )

        return

    atomic_write(
        path,
        payload,
    )


def run_command(command):
    process = subprocess.run(
        command,
        cwd=str(S6),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return (
        process.returncode,
        process.stdout,
    )


def parse_test_count(output):
    for line in output.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            return int(match.group(1))

    return None


def run_regression():
    rc, output = run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ]
    )

    count = parse_test_count(output)

    require(
        rc == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                output.splitlines()[-100:]
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


# ============================================================
# AST helpers
# ============================================================

def constant_value(node):
    if isinstance(node, ast.Constant):
        return node.value

    return None


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def project_call_inventory():
    """
    Search existing non-formal Stage6 code for literal
    project_box_to_headlamp provenance pairs.

    This does NOT open WOMD data or scientific outcomes.
    """

    inventory = []

    roots = (
        S6 / "src/iscai_stage6",
        S6 / "scripts",
    )

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*.py"):

            lower = str(path).lower()

            if "formal" in lower:
                continue

            if path.resolve() == RUNNER.resolve():
                continue

            source = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            try:
                tree = ast.parse(
                    source,
                    filename=str(path),
                )

            except Exception:
                continue

            for node in ast.walk(tree):

                if not isinstance(node, ast.Call):
                    continue

                if (
                    call_name(node.func)
                    !=
                    "project_box_to_headlamp"
                ):
                    continue

                keywords = {
                    item.arg:
                        item.value
                    for item in node.keywords
                    if item.arg is not None
                }

                state = constant_value(
                    keywords.get(
                        "state_source"
                    )
                )

                orientation = constant_value(
                    keywords.get(
                        "orientation_source"
                    )
                )

                controller = constant_value(
                    keywords.get(
                        "controller_path"
                    )
                )

                if (
                    isinstance(state, str)
                    and
                    isinstance(orientation, str)
                    and
                    controller is False
                ):

                    inventory.append(
                        {
                            "path":
                                str(
                                    path.relative_to(S6)
                                ),

                            "line":
                                int(node.lineno),

                            "state_source":
                                state,

                            "orientation_source":
                                orientation,

                            "controller_path":
                                False,
                        }
                    )

    return inventory


def flatten_string_values(value):
    result = []

    if isinstance(value, str):
        result.append(value)

    elif isinstance(
        value,
        (
            tuple,
            list,
            set,
            frozenset,
        ),
    ):
        for item in value:
            result.extend(
                flatten_string_values(item)
            )

    elif isinstance(value, dict):
        for key, item in value.items():
            result.extend(
                flatten_string_values(key)
            )

            result.extend(
                flatten_string_values(item)
            )

    return result


def collect_orientation_candidates(
    validation_function,
    call_sites,
):
    """
    Candidate labels come only from existing geometry runtime
    or existing non-formal Stage6 call sites.
    """

    candidates = set()

    # Module-level scientific provenance constants.
    for name in dir(geometry):

        upper = name.upper()

        if (
            "ORIENTATION" not in upper
            and
            "SOURCE" not in upper
            and
            "PROVENANCE" not in upper
        ):
            continue

        try:
            value = getattr(
                geometry,
                name,
            )

        except Exception:
            continue

        for item in flatten_string_values(value):

            if len(item) <= 160:
                candidates.add(item)

    # Literal constants inside validator source.
    source = inspect.getsource(
        validation_function
    )

    tree = ast.parse(source)

    for node in ast.walk(tree):

        if (
            isinstance(node, ast.Constant)
            and
            isinstance(node.value, str)
            and
            len(node.value) <= 160
        ):
            candidates.add(
                node.value
            )

    # Existing validated/non-formal projection call sites.
    for item in call_sites:
        candidates.add(
            item[
                "orientation_source"
            ]
        )

    # Remove obvious diagnostic sentences.
    candidates = {
        item
        for item in candidates
        if (
            item.strip()
            and
            "\n" not in item
            and
            len(item.split()) <= 8
        )
    }

    return sorted(candidates)


def accepted_orientation_candidates(
    validation_function,
    candidates,
):
    accepted = []

    rejected = []

    for candidate in candidates:

        try:

            validation_function(
                state_source=CURRENT_STATE_SOURCE,
                orientation_source=candidate,
                controller_path=False,
            )

        except Exception as exc:

            rejected.append(
                {
                    "candidate":
                        candidate,

                    "exception":
                        type(exc).__name__,

                    "message":
                        str(exc),
                }
            )

        else:

            accepted.append(
                candidate
            )

    return (
        accepted,
        rejected,
    )


def semantic_score(
    candidate,
    call_sites,
):
    """
    Score only evidence relevance.

    This does not make a scientific choice from outcomes.
    """

    lower = candidate.lower()

    score = 0

    reasons = []

    same_state_calls = [
        item
        for item in call_sites
        if (
            item[
                "state_source"
            ]
            ==
            CURRENT_STATE_SOURCE
            and
            item[
                "orientation_source"
            ]
            ==
            candidate
        )
    ]

    if same_state_calls:
        score += 1000
        reasons.append(
            "existing_controller_path_false_call_with_exact_state_source"
        )

    future_gt_calls = [
        item
        for item in call_sites
        if (
            item[
                "orientation_source"
            ]
            ==
            candidate
            and
            any(
                token
                in
                item[
                    "state_source"
                ].lower()
                for token in (
                    "future",
                    "gt",
                    "truth",
                )
            )
        )
    ]

    if future_gt_calls:
        score += 500
        reasons.append(
            "existing_future_truth_evaluator_call"
        )

    for token, weight in (
        ("future", 50),
        ("ground_truth", 50),
        ("truth", 40),
        ("gt", 30),
        ("evaluator", 30),
        ("heading", 20),
        ("yaw", 20),
        ("orientation", 10),
        ("waymo", 10),
        ("womd", 10),
    ):

        if token in lower:
            score += weight
            reasons.append(
                f"semantic_token:{token}"
            )

    return (
        score,
        reasons,
    )


def choose_unique_orientation(
    accepted,
    call_sites,
):
    require(
        accepted,
        (
            "No orientation_source accepted by "
            "frozen validate_controller_provenance()."
        ),
    )

    ranked = []

    for candidate in accepted:

        score, reasons = semantic_score(
            candidate,
            call_sites,
        )

        ranked.append(
            {
                "candidate":
                    candidate,

                "score":
                    score,

                "reasons":
                    reasons,
            }
        )

    ranked.sort(
        key=lambda item: (
            -item["score"],
            item["candidate"],
        )
    )

    require(
        ranked[0][
            "score"
        ]
        >
        0,
        (
            "Accepted orientation labels exist, "
            "but none has future/evaluator semantic "
            "evidence. Refusing to guess."
        ),
    )

    if len(ranked) > 1:

        require(
            ranked[0]["score"]
            >
            ranked[1]["score"],
            (
                "Orientation-source evidence is "
                "ambiguous. Refusing to guess.\n"
                +
                json.dumps(
                    ranked,
                    indent=2,
                )
            ),
        )

    return (
        ranked[0]["candidate"],
        ranked,
    )


# ============================================================
# Exact runner patch
# ============================================================

def absolute_offset(
    source,
    lineno,
    col_offset,
):
    lines = source.splitlines(
        keepends=True
    )

    return (
        sum(
            len(line)
            for line in lines[
                :lineno - 1
            ]
        )
        +
        col_offset
    )


def locate_target_orientation_literal(
    source,
):
    tree = ast.parse(
        source,
        filename=str(RUNNER),
    )

    matches = []

    for node in ast.walk(tree):

        if not isinstance(node, ast.Call):
            continue

        if (
            call_name(node.func)
            !=
            "project_box_to_headlamp"
        ):
            continue

        keywords = {
            item.arg:
                item
            for item in node.keywords
            if item.arg is not None
        }

        controller_keyword = keywords.get(
            "controller_path"
        )

        state_keyword = keywords.get(
            "state_source"
        )

        orientation_keyword = keywords.get(
            "orientation_source"
        )

        if (
            controller_keyword is None
            or
            state_keyword is None
            or
            orientation_keyword is None
        ):
            continue

        controller_value = constant_value(
            controller_keyword.value
        )

        state_value = constant_value(
            state_keyword.value
        )

        orientation_value = constant_value(
            orientation_keyword.value
        )

        if (
            controller_value is False
            and
            state_value
            ==
            CURRENT_STATE_SOURCE
            and
            orientation_value
            ==
            BAD_ORIENTATION_SOURCE
        ):

            matches.append(
                (
                    node,
                    orientation_keyword.value,
                )
            )

    require(
        len(matches) == 1,
        (
            "Expected exactly one failed "
            "future-GT evaluator projection call; "
            f"found {len(matches)}."
        ),
    )

    return matches[0]


def patch_orientation_literal(
    source,
    replacement,
):
    (
        call,
        value_node,
    ) = locate_target_orientation_literal(
        source
    )

    require(
        value_node.end_lineno
        is not None,
        (
            "AST orientation literal "
            "end position unavailable."
        ),
    )

    start = absolute_offset(
        source,
        value_node.lineno,
        value_node.col_offset,
    )

    end = absolute_offset(
        source,
        value_node.end_lineno,
        value_node.end_col_offset,
    )

    original_segment = source[
        start:end
    ]

    patched = (
        source[:start]
        +
        json.dumps(
            replacement
        )
        +
        source[end:]
    )

    compile(
        patched,
        str(RUNNER),
        "exec",
    )

    # Structural proof.
    tree = ast.parse(
        patched,
        filename=str(RUNNER),
    )

    successful_matches = 0

    for node in ast.walk(tree):

        if not isinstance(node, ast.Call):
            continue

        if (
            call_name(node.func)
            !=
            "project_box_to_headlamp"
        ):
            continue

        keywords = {
            item.arg:
                item.value
            for item in node.keywords
            if item.arg is not None
        }

        if (
            constant_value(
                keywords.get(
                    "state_source"
                )
            )
            ==
            CURRENT_STATE_SOURCE
            and
            constant_value(
                keywords.get(
                    "orientation_source"
                )
            )
            ==
            replacement
            and
            constant_value(
                keywords.get(
                    "controller_path"
                )
            )
            is False
        ):

            successful_matches += 1

    require(
        successful_matches == 1,
        (
            "Patched runner does not contain "
            "exactly one expected future-GT "
            "evaluator projection call."
        ),
    )

    return (
        patched,
        {
            "call_line":
                int(call.lineno),

            "old_orientation_source":
                BAD_ORIENTATION_SOURCE,

            "new_orientation_source":
                replacement,

            "old_source_segment":
                original_segment,
        },
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART2C"
    )
    print(
        "FUTURE-GT EVALUATOR PROVENANCE LABEL REPAIR"
    )
    print(
        "NO METRIC OUTCOMES / NO PREDICTIVE / NO FORMAL"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Failure boundary
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT FAILURE BOUNDARY ====="
    )

    require(
        PREOUTCOME_CONTRACT.is_file(),
        (
            "Frozen preoutcome contract missing."
        ),
    )

    require(
        file_sha(
            PREOUTCOME_CONTRACT
        )
        ==
        EXPECTED_PREOUTCOME_SHA,
        (
            "Frozen preoutcome contract SHA changed."
        ),
    )

    require(
        file_sha(
            LEDGER
        )
        ==
        EXPECTED_LEDGER_SHA,
        (
            "Reactive ledger SHA changed."
        ),
    )

    require(
        file_sha(
            T0_MAPS
        )
        ==
        EXPECTED_T0_MAPS_SHA,
        (
            "Frozen t0 maps SHA changed."
        ),
    )

    require(
        not RAW_VALUES.exists(),
        (
            "Reactive raw metric artifact already exists."
        ),
    )

    require(
        not DELTA_FREEZE.exists(),
        (
            "NI delta freeze already exists."
        ),
    )

    require(
        not ACCEPTANCE_POLICY.exists(),
        (
            "Acceptance-policy freeze already exists."
        ),
    )

    require(
        not FINAL_REPORT.exists(),
        (
            "Final reactive-only PASS report already exists."
        ),
    )

    print(
        "preoutcome scoring contract = EXACT PASS"
    )

    print(
        "reactive raw metrics        = ABSENT"
    )

    print(
        "NI delta freeze             = ABSENT"
    )

    print(
        "acceptance policy           = ABSENT"
    )

    print(
        "predictive performance      = NOT COMPUTED"
    )

    print(
        "P_occ content               = NOT OPENED"
    )

    print(
        "formal evaluation           = NO"
    )

    print(
        "future GT                   = OPENED, EVALUATOR ONLY"
    )

    print(
        "metric values before failure= ZERO"
    )

    # --------------------------------------------------------
    # B. Frozen geometry provenance authority
    # --------------------------------------------------------

    print()
    print(
        "===== B. FROZEN GEOMETRY PROVENANCE AUTHORITY ====="
    )

    geometry_bind = read_json(
        GEOMETRY_BIND_REPORT
    )

    require(
        geometry_bind.get(
            "status"
        )
        ==
        "PASS_FUTURE_EVALUATOR_GEOMETRY_IDENTITY_BOUND",
        (
            "Geometry/identity binding is not PASS."
        ),
    )

    geometry_path = Path(
        inspect.getfile(
            geometry
        )
    )

    expected_geometry_sha = (
        geometry_bind[
            "geometry"
        ][
            "module_sha256"
        ]
    )

    require(
        file_sha(
            geometry_path
        )
        ==
        expected_geometry_sha,
        (
            "geometry.py changed after "
            "geometry/identity bind."
        ),
    )

    validator = getattr(
        geometry,
        "validate_controller_provenance",
        None,
    )

    require(
        callable(
            validator
        ),
        (
            "validate_controller_provenance "
            "is unavailable."
        ),
    )

    validator_source = inspect.getsource(
        validator
    )

    print(
        "geometry.py SHA256 = EXACT PASS"
    )

    print()
    print(
        "validate_controller_provenance source:"
    )

    print(
        validator_source
    )

    # --------------------------------------------------------
    # C. Existing call-site evidence
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXISTING NON-FORMAL PROVENANCE CALLS ====="
    )

    call_sites = project_call_inventory()

    print(
        "controller_path=False literal call sites =",
        len(
            call_sites
        ),
    )

    for item in call_sites:

        print(
            "  ",
            item[
                "path"
            ],
            f"L{item['line']}",
            "state=",
            repr(
                item[
                    "state_source"
                ]
            ),
            "orientation=",
            repr(
                item[
                    "orientation_source"
                ]
            ),
        )

    # --------------------------------------------------------
    # D. Candidate extraction + validator probe
    # --------------------------------------------------------

    print()
    print(
        "===== D. EXACT ORIENTATION-SOURCE RESOLUTION ====="
    )

    candidates = collect_orientation_candidates(
        validator,
        call_sites,
    )

    (
        accepted,
        rejected,
    ) = accepted_orientation_candidates(
        validator,
        candidates,
    )

    print(
        "candidate strings =",
        len(
            candidates
        ),
    )

    print(
        "accepted with state_source="
        f"{CURRENT_STATE_SOURCE!r} "
        "and controller_path=False:"
    )

    for item in accepted:
        print(
            "  ",
            repr(item),
        )

    (
        chosen,
        ranking,
    ) = choose_unique_orientation(
        accepted,
        call_sites,
    )

    print()
    print(
        "semantic evidence ranking:"
    )

    for item in ranking:

        print(
            "  ",
            item[
                "score"
            ],
            repr(
                item[
                    "candidate"
                ]
            ),
            item[
                "reasons"
            ],
        )

    require(
        chosen
        !=
        BAD_ORIENTATION_SOURCE,
        (
            "Resolution returned the already "
            "rejected orientation label."
        ),
    )

    # Final synthetic validator gate.
    validator(
        state_source=CURRENT_STATE_SOURCE,
        orientation_source=chosen,
        controller_path=False,
    )

    print()
    print(
        "selected orientation_source =",
        repr(
            chosen
        ),
    )

    print(
        "validator synthetic gate    = PASS"
    )

    print(
        "outcome-driven choice       = NO"
    )

    # --------------------------------------------------------
    # E. Freeze provenance binding BEFORE metric retry
    # --------------------------------------------------------

    print()
    print(
        "===== E. PROVENANCE BINDING FREEZE ====="
    )

    provenance_payload = {
        "stage":
            6,

        "block":
            "6.8_Part2C_future_GT_evaluator_provenance_repair",

        "status":
            "FROZEN_EVALUATOR_ONLY_FUTURE_GT_PROVENANCE_BINDING",

        "failure_type":
            "orientation_source_label_not_in_geometry_whitelist",

        "scientific_geometry_change":
            False,

        "future_truth_status_at_repair":
            (
                "opened evaluator-only before "
                "first projection; no metric computed"
            ),

        "projection": {
            "state_source":
                CURRENT_STATE_SOURCE,

            "orientation_source":
                chosen,

            "controller_path":
                False,
        },

        "selection_evidence": {
            "geometry_module":
                str(
                    geometry_path
                ),

            "geometry_module_sha256":
                expected_geometry_sha,

            "validator_source_sha256":
                sha256(
                    validator_source.encode(
                        "utf-8"
                    )
                ).hexdigest(),

            "existing_nonformal_call_sites":
                call_sites,

            "accepted_orientation_candidates":
                accepted,

            "ranking":
                ranking,

            "unique_choice":
                True,
        },

        "scientific_boundary": {
            "reactive_metric_values_computed_before_freeze":
                False,

            "predictive_performance_computed":
                False,

            "P_occ_opened":
                False,

            "NI_deltas_computed":
                False,

            "formal_evaluation":
                False,

            "controller_or_policy_modified":
                False,
        },

        "preoutcome_scoring_contract": {
            "path":
                str(
                    PREOUTCOME_CONTRACT
                ),

            "sha256":
                EXPECTED_PREOUTCOME_SHA,

            "modified":
                False,
        },
    }

    write_once(
        PROVENANCE_BINDING,
        provenance_payload,
    )

    provenance_sha = file_sha(
        PROVENANCE_BINDING
    )

    print(
        "provenance binding SHA256 =",
        provenance_sha,
    )

    # --------------------------------------------------------
    # F. Exact runner patch
    # --------------------------------------------------------

    print()
    print(
        "===== F. PATCH ORIENTATION LABEL ONLY ====="
    )

    require(
        RUNNER.is_file(),
        (
            f"Reactive-only runner missing: {RUNNER}"
        ),
    )

    original_bytes = RUNNER.read_bytes()

    original_source = original_bytes.decode(
        "utf-8"
    )

    pre_runner_sha = file_sha(
        RUNNER
    )

    (
        patched_source,
        patch_detail,
    ) = patch_orientation_literal(
        original_source,
        chosen,
    )

    require(
        original_source
        !=
        patched_source,
        (
            "Runner source did not change."
        ),
    )

    BACKUP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BACKUP.exists():

        require(
            BACKUP.read_bytes()
            ==
            original_bytes,
            (
                "Existing provenance-repair "
                "backup differs from current runner."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    print(
        "runner pre-patch SHA256 =",
        pre_runner_sha,
    )

    print(
        "target call line         =",
        patch_detail[
            "call_line"
        ],
    )

    print(
        "old orientation source   =",
        repr(
            patch_detail[
                "old_orientation_source"
            ]
        ),
    )

    print(
        "new orientation source   =",
        repr(
            patch_detail[
                "new_orientation_source"
            ]
        ),
    )

    applied = False

    try:

        atomic_write(
            RUNNER,
            patched_source.encode(
                "utf-8"
            ),
        )

        applied = True

        rc_compile, compile_output = run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(
                    RUNNER
                ),
            ]
        )

        require(
            rc_compile == 0,
            (
                "Patched reactive-only runner "
                "compile failed:\n"
                +
                compile_output
            ),
        )

        tests = run_regression()

        print(
            "patched runner compile = PASS"
        )

        print(
            "Stage6 regression      =",
            f"{tests} / {tests} PASS",
        )

    except BaseException:

        if applied:

            print()
            print(
                "PATCH FAILURE -> EXACT RUNNER ROLLBACK"
            )

            atomic_write(
                RUNNER,
                original_bytes,
            )

            require(
                file_sha(
                    RUNNER
                )
                ==
                pre_runner_sha,
                (
                    "Runner rollback SHA mismatch."
                ),
            )

            print(
                "runner rollback = EXACT PASS"
            )

        raise

    post_runner_sha = file_sha(
        RUNNER
    )

    # --------------------------------------------------------
    # G. Report
    # --------------------------------------------------------

    result = {
        "stage":
            6,

        "block":
            "6.8_Part2C_future_GT_evaluator_provenance_repair",

        "status":
            "PASS_FUTURE_GT_EVALUATOR_PROVENANCE_REPAIRED",

        "diagnosis": {
            "geometry_failure":
                False,

            "future_box_transform_failure":
                False,

            "rasterization_failure":
                False,

            "metric_failure":
                False,

            "failure":
                (
                    "runner supplied an orientation_source "
                    "label not accepted by frozen geometry "
                    "provenance validator"
                ),
        },

        "provenance_binding": {
            "path":
                str(
                    PROVENANCE_BINDING
                ),

            "sha256":
                provenance_sha,

            "state_source":
                CURRENT_STATE_SOURCE,

            "orientation_source":
                chosen,

            "controller_path":
                False,
        },

        "runner_patch": {
            "path":
                str(
                    RUNNER
                ),

            "pre_sha256":
                pre_runner_sha,

            "post_sha256":
                post_runner_sha,

            "backup_path":
                str(
                    BACKUP
                ),

            "backup_sha256":
                file_sha(
                    BACKUP
                ),

            "scope":
                "single orientation_source AST literal only",

            "scientific_algorithm_change":
                False,
        },

        "scientific_boundary": {
            "future_truth_opened":
                True,

            "future_truth_role":
                "evaluator_only",

            "reactive_metric_values_computed_before_repair":
                False,

            "predictive_performance_computed":
                False,

            "P_occ_opened":
                False,

            "NI_deltas_computed":
                False,

            "formal_evaluation":
                False,

            "controller_policy_tuning":
                False,
        },

        "regression": {
            "tests":
                tests,

            "status":
                "PASS",
        },

        "next":
            (
                "rerun the same reactive-only NI delta "
                "freeze runner; preoutcome scoring contract "
                "remains byte-identical"
            ),
    }

    atomic_write(
        PATCH_REPORT,
        canonical_json_bytes(
            result
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 FUTURE-GT PROVENANCE REPAIR — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "geometry implementation     = UNCHANGED"
    )

    print(
        "preoutcome contract         = UNCHANGED"
    )

    print(
        "state_source               =",
        repr(
            CURRENT_STATE_SOURCE
        ),
    )

    print(
        "orientation_source         =",
        repr(
            chosen
        ),
    )

    print(
        "controller_path            = False"
    )

    print(
        "provenance validator       = PASS"
    )

    print(
        "runner patch scope         = ONE AST LITERAL"
    )

    print(
        "reactive metrics before repair = 0"
    )

    print(
        "future truth               = OPENED, EVALUATOR ONLY"
    )

    print(
        "predictive performance     = NOT COMPUTED"
    )

    print(
        "P_occ content              = NOT OPENED"
    )

    print(
        "NI deltas                  = NOT COMPUTED"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )

    print(
        "patched runner SHA256      =",
        post_runner_sha,
    )

    print(
        "STATUS = "
        "PASS_FUTURE_GT_EVALUATOR_PROVENANCE_REPAIRED"
    )

    print(
        "report =",
        PATCH_REPORT,
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
        "BLOCK 6.8 FUTURE-GT PROVENANCE REPAIR = BLOCKED"
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
        "future truth           = OPENED, EVALUATOR ONLY"
    )

    print(
        "reactive metric outputs = NOT FROZEN"
    )

    print(
        "predictive performance = NOT COMPUTED"
    )

    print(
        "P_occ content          = NOT OPENED"
    )

    print(
        "NI deltas              = NOT COMPUTED"
    )

    print(
        "formal evaluation      = NO"
    )

    print(
        "Do not run predictive/formal evaluation."
    )

    print(
        "If candidate resolution is ambiguous, "
        "send Sections B-D for targeted binding."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

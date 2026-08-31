from __future__ import annotations

import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

TARGET = (
    S6
    / "scripts/"
      "run_block68_part2b_part1_reactive_decision_ledger.py"
)

BINDING = (
    S6
    / "configs/"
      "stage6_frozen_illumination_grid_runtime_binding.json"
)

GRID_REPORT = (
    S6
    / "reports/"
      "block68_grid_runtime_binding_repair.json"
)

MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

LEDGER = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_decision_ledger.jsonl"
)

DECISION_REPORT = (
    S6
    / "reports/"
      "block68_reactive_decision_ledger_freeze.json"
)

BACKUP = (
    S6
    / "artifacts/block68/grid_binding_patch_v3/"
      "run_block68_part2b_part1_reactive_decision_ledger.pre_v3.py"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_part2b_part1_grid_binding_patch_v3.json"
)


EXPECTED_TARGET_PRE_SHA = (
    "fb9c62a46f2ad238fbbbf9ddb4ce791c"
    "02e74b65c66f316e2f586ddbd2bd13d6"
)

EXPECTED_BINDING_SHA = (
    "be5145fe60a941e7906639916cfa384d"
    "5674dc29f294f5a42e12c962bc4df4bc"
)

EXPECTED_TESTS = 224


NEW_FUNCTION = r'''def resolve_grid():
    """
    Load the already frozen Stage6 illumination-grid
    runtime binding.

    No grid discovery, outcome-driven selection, future
    truth, or numerical invention occurs here.
    """

    binding_path = (
        S6
        / "configs/"
          "stage6_frozen_illumination_grid_runtime_binding.json"
    )

    expected_sha = "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc"

    require(
        binding_path.is_file(),
        (
            "Frozen illumination-grid runtime "
            f"binding missing: {binding_path}"
        ),
    )

    actual_sha = file_sha(
        binding_path
    )

    require(
        actual_sha == expected_sha,
        (
            "Frozen illumination-grid binding "
            "SHA changed.\n"
            f"expected={expected_sha}\n"
            f"actual={actual_sha}"
        ),
    )

    binding = safe_json(
        binding_path
    )

    require(
        binding.get("status")
        ==
        "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING",
        (
            "Frozen illumination-grid "
            "binding status mismatch."
        ),
    )

    require(
        binding.get(
            "scientific_parameter_change"
        )
        is False,
        (
            "Frozen grid binding unexpectedly "
            "records scientific parameter change."
        ),
    )

    anti = binding.get(
        "anti_invention",
        {}
    )

    required_false = (
        "range_min_defaulted_to_zero",
        "constructor_values_selected_from_outcomes",
        "predictive_performance_read",
        "reactive_performance_read",
        "future_truth_read",
        "formal_content_read",
    )

    for key in required_false:

        require(
            anti.get(key) is False,
            (
                "Frozen grid anti-invention "
                f"gate failed: {key}"
            ),
        )

    provenance = binding.get(
        "provenance",
        {}
    )

    block64_sha = provenance.get(
        "Block64_config_sha256"
    )

    require(
        isinstance(
            block64_sha,
            str,
        )
        and
        len(block64_sha) == 64,
        (
            "Frozen Block6.4 provenance "
            "SHA missing."
        ),
    )

    require(
        file_sha(GRID_CONFIG)
        ==
        block64_sha,
        (
            "Block6.4 grid source changed "
            "after runtime binding freeze."
        ),
    )

    constructor = binding.get(
        "constructor"
    )

    require(
        isinstance(
            constructor,
            dict,
        ),
        (
            "Frozen grid constructor "
            "must be an object."
        ),
    )

    expected_keys = {
        "theta_min_rad",
        "theta_max_rad",
        "range_min_m",
        "range_max_m",
        "n_theta",
        "n_range",
    }

    require(
        set(constructor)
        ==
        expected_keys,
        (
            "Frozen grid constructor "
            f"keys changed: {sorted(constructor)}"
        ),
    )

    kwargs = {
        "theta_min_rad":
            float(
                constructor[
                    "theta_min_rad"
                ]
            ),

        "theta_max_rad":
            float(
                constructor[
                    "theta_max_rad"
                ]
            ),

        "range_min_m":
            float(
                constructor[
                    "range_min_m"
                ]
            ),

        "range_max_m":
            float(
                constructor[
                    "range_max_m"
                ]
            ),

        "n_theta":
            int(
                constructor[
                    "n_theta"
                ]
            ),

        "n_range":
            int(
                constructor[
                    "n_range"
                ]
            ),
    }

    require(
        (
            kwargs[
                "n_theta"
            ],
            kwargs[
                "n_range"
            ],
        )
        ==
        GRID_SHAPE,
        (
            "Frozen illumination-grid shape "
            f"differs from {GRID_SHAPE}."
        ),
    )

    require(
        all(
            math.isfinite(
                kwargs[key]
            )
            for key in (
                "theta_min_rad",
                "theta_max_rad",
                "range_min_m",
                "range_max_m",
            )
        ),
        (
            "Frozen illumination-grid "
            "contains non-finite bounds."
        ),
    )

    require(
        kwargs[
            "theta_min_rad"
        ]
        <
        kwargs[
            "theta_max_rad"
        ],
        (
            "Frozen theta bounds "
            "are not increasing."
        ),
    )

    require(
        kwargs[
            "range_min_m"
        ]
        <
        kwargs[
            "range_max_m"
        ],
        (
            "Frozen range bounds "
            "are not increasing."
        ),
    )

    exact_bounds = {
        "theta_min_rad":
            -0.4363323129985824,

        "theta_max_rad":
            0.4363323129985824,

        "range_min_m":
            0.0,

        "range_max_m":
            150.0,
    }

    for key, expected in (
        exact_bounds.items()
    ):

        require(
            math.isclose(
                kwargs[key],
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            (
                f"Frozen {key} differs "
                "from sealed runtime binding."
            ),
        )

    grid = IlluminationGridSpec(
        **kwargs
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
        GRID_SHAPE,
        (
            "Frozen grid failed Part-A "
            f"shape readback: {blank.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(
                blank
            )
        ),
        (
            "Blank Part-A map contains "
            "non-finite values."
        ),
    )

    require(
        np.all(
            (blank >= 0.0)
            &
            (blank <= 1.0)
        ),
        (
            "Blank Part-A map outside [0,1]."
        ),
    )

    attempts = [
        {
            "label":
                "frozen_stage6_runtime_grid_binding",

            "status":
                "ACCEPTED",

            "binding_path":
                str(
                    binding_path
                ),

            "binding_sha256":
                actual_sha,

            "constructor":
                kwargs,

            "heuristic_discovery":
                False,

            "numeric_values_invented":
                False,

            "outcomes_used":
                False,

            "future_truth_used":
                False,
        }
    ]

    return (
        grid,
        attempts,
        str(
            inspect.signature(
                IlluminationGridSpec
            )
        ),
    )

'''


# ============================================================
# Generic helpers
# ============================================================

def require(
    condition,
    message,
):

    if not bool(condition):
        raise RuntimeError(message)


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
            "Formal content access "
            f"forbidden: {path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_write(
    path: Path,
    data: bytes,
):

    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_bytes(
        data
    )

    os.replace(
        temporary,
        path,
    )


def run(
    command,
):

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


def parse_test_count(
    output,
):

    for line in (
        output.splitlines()
    ):

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:

            return int(
                match.group(1)
            )

    return None


def tail(
    output,
    lines=100,
):

    return "\n".join(
        output.splitlines()[
            -lines:
        ]
    )


# ============================================================
# AST helpers
# ============================================================

def unique_resolve_grid(
    source,
):

    tree = ast.parse(
        source,
        filename=str(
            TARGET
        ),
    )

    matches = [
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            "resolve_grid"
        )
    ]

    require(
        len(matches) == 1,
        (
            "Expected exactly one top-level "
            "resolve_grid(); found "
            f"{len(matches)}."
        ),
    )

    node = matches[0]

    require(
        node.end_lineno
        is not None,
        (
            "resolve_grid() AST "
            "end_lineno missing."
        ),
    )

    require(
        (
            len(
                node.args.posonlyargs
            )
            ==
            0
            and
            len(
                node.args.args
            )
            ==
            0
            and
            len(
                node.args.kwonlyargs
            )
            ==
            0
            and
            node.args.vararg
            is None
            and
            node.args.kwarg
            is None
        ),
        (
            "resolve_grid() signature "
            "unexpectedly changed."
        ),
    )

    return (
        tree,
        node,
    )


def function_constants(
    node,
):

    values = []

    for child in ast.walk(
        node
    ):

        if (
            isinstance(
                child,
                ast.Constant,
            )
            and
            isinstance(
                child.value,
                str,
            )
        ):

            values.append(
                child.value
            )

    return values


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART2B PART1 GRID PATCH V3"
    )
    print(
        "SEMANTIC AST SHA VALIDATION"
    )
    print(
        "RUNNER-ONLY / NO SCIENTIFIC CHANGE"
    )
    print(
        "============================================================"
    )

    require(
        TARGET.is_file(),
        (
            f"Missing target runner: {TARGET}"
        ),
    )

    require(
        BINDING.is_file(),
        (
            f"Missing frozen binding: {BINDING}"
        ),
    )

    require(
        GRID_REPORT.is_file(),
        (
            "Missing grid-binding PASS report."
        ),
    )

    # ========================================================
    # A. Exact pre-state
    # ========================================================

    print()
    print(
        "===== A. EXACT PRE-PATCH STATE ====="
    )

    target_sha = file_sha(
        TARGET
    )

    require(
        target_sha
        ==
        EXPECTED_TARGET_PRE_SHA,
        (
            "Target runner differs from "
            "known unchanged pre-patch state.\n"
            f"expected={EXPECTED_TARGET_PRE_SHA}\n"
            f"actual={target_sha}"
        ),
    )

    require(
        file_sha(
            BINDING
        )
        ==
        EXPECTED_BINDING_SHA,
        (
            "Frozen grid-binding SHA changed."
        ),
    )

    grid_report = safe_json(
        GRID_REPORT
    )

    require(
        grid_report.get(
            "status"
        )
        ==
        "PASS_FROZEN_GRID_RUNTIME_BINDING",
        (
            "Grid-binding report "
            "is not frozen PASS."
        ),
    )

    require(
        grid_report[
            "scientific_boundary"
        ][
            "reactive_performance_read"
        ]
        is False,
        (
            "Reactive performance was "
            "already read."
        ),
    )

    require(
        grid_report[
            "scientific_boundary"
        ][
            "predictive_performance_read"
        ]
        is False,
        (
            "Predictive performance was "
            "already read."
        ),
    )

    require(
        grid_report[
            "scientific_boundary"
        ][
            "future_truth_read"
        ]
        is False,
        (
            "Future truth was already read."
        ),
    )

    print(
        "whole-runner SHA   = EXACT PASS"
    )

    print(
        "grid-binding SHA   = EXACT PASS"
    )

    print(
        "scientific outputs = NOT READ"
    )

    print(
        "future truth       = NOT READ"
    )

    # ========================================================
    # B. Ensure failed attempts left no outputs
    # ========================================================

    print()
    print(
        "===== B. PRE-SCIENTIFIC-OUTPUT BOUNDARY ====="
    )

    existing = [
        str(path)
        for path in (
            MAPS,
            LEDGER,
            DECISION_REPORT,
        )
        if path.exists()
    ]

    require(
        not existing,
        (
            "Unexpected Part2B scientific "
            "outputs already exist: "
            +
            repr(existing)
        ),
    )

    print(
        "t0 maps        = ABSENT"
    )

    print(
        "decision ledger= ABSENT"
    )

    print(
        "freeze report  = ABSENT"
    )

    # ========================================================
    # C. Locate original function
    # ========================================================

    print()
    print(
        "===== C. UNIQUE AST PATCH TARGET ====="
    )

    original_bytes = TARGET.read_bytes()

    original = original_bytes.decode(
        "utf-8"
    )

    (
        _,
        original_node,
    ) = unique_resolve_grid(
        original
    )

    lines = original.splitlines(
        keepends=True
    )

    start = int(
        original_node.lineno
    )

    end = int(
        original_node.end_lineno
    )

    old_function = "".join(
        lines[
            start - 1:
            end
        ]
    )

    old_function_sha = sha256(
        old_function.encode(
            "utf-8"
        )
    ).hexdigest()

    require(
        (
            "stage6_frozen_illumination_grid_runtime_binding.json"
            not in
            old_function
        ),
        (
            "Target resolve_grid() already "
            "appears patched."
        ),
    )

    print(
        "resolve_grid count  = 1"
    )

    print(
        "resolve_grid lines  =",
        f"{start}..{end}",
    )

    print(
        "old function SHA256 =",
        old_function_sha,
    )

    # ========================================================
    # D. Backup
    # ========================================================

    print()
    print(
        "===== D. VERIFIED BACKUP ====="
    )

    BACKUP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BACKUP.exists():

        require(
            file_sha(
                BACKUP
            )
            ==
            EXPECTED_TARGET_PRE_SHA,
            (
                "Existing V3 backup "
                "does not match pre-patch runner."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    require(
        file_sha(
            BACKUP
        )
        ==
        EXPECTED_TARGET_PRE_SHA,
        (
            "Backup SHA verification failed."
        ),
    )

    print(
        "backup SHA = EXACT PASS"
    )

    # ========================================================
    # E. Build patched source
    # ========================================================

    print()
    print(
        "===== E. BUILD SEMANTICALLY VERIFIED PATCH ====="
    )

    replacement = (
        NEW_FUNCTION
        if NEW_FUNCTION.endswith(
            "\n"
        )
        else
        NEW_FUNCTION
        +
        "\n"
    )

    patched = "".join(
        lines[
            :start - 1
        ]
        +
        [
            replacement
        ]
        +
        lines[
            end:
        ]
    )

    # Whole-file syntax gate.
    compile(
        patched,
        str(
            TARGET
        ),
        "exec",
    )

    (
        _,
        patched_node,
    ) = unique_resolve_grid(
        patched
    )

    constants = function_constants(
        patched_node
    )

    # This is the V3 repair:
    # compare Python AST semantic constant value rather than
    # raw source-text contiguity.
    require(
        EXPECTED_BINDING_SHA
        in
        constants,
        (
            "Replacement function AST does not "
            "contain exact frozen binding SHA "
            "as a semantic string constant."
        ),
    )

    require(
        (
            "stage6_frozen_illumination_grid_runtime_binding.json"
            in
            constants
        ),
        (
            "Replacement function AST does not "
            "contain exact binding filename."
        ),
    )

    replacement_function = ast.get_source_segment(
        patched,
        patched_node,
    )

    require(
        replacement_function
        is not None,
        (
            "Could not recover replacement "
            "source from AST."
        ),
    )

    require(
        "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING"
        in
        constants,
        (
            "Replacement function does not "
            "seal binding status."
        ),
    )

    print(
        "whole-file syntax       = PASS"
    )

    print(
        "binding filename AST    = PASS"
    )

    print(
        "full binding SHA AST    = PASS"
    )

    print(
        "binding status AST      = PASS"
    )

    print(
        "changed function        = resolve_grid ONLY"
    )

    print(
        "numeric invention       = NO"
    )

    # ========================================================
    # F. Transactional patch
    # ========================================================

    print()
    print(
        "===== F. TRANSACTIONAL PATCH ====="
    )

    applied = False

    try:

        atomic_write(
            TARGET,
            patched.encode(
                "utf-8"
            ),
        )

        applied = True

        rc_compile, out_compile = run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(
                    TARGET
                ),
            ]
        )

        require(
            rc_compile == 0,
            (
                "Patched runner compile failed:\n"
                +
                out_compile
            ),
        )

        print(
            "patched runner compile = PASS"
        )

        rc_tests, out_tests = run(
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

        test_count = parse_test_count(
            out_tests
        )

        require(
            rc_tests == 0,
            (
                "Stage6 regression failed "
                "after runner patch:\n"
                +
                tail(
                    out_tests
                )
            ),
        )

        require(
            test_count
            ==
            EXPECTED_TESTS,
            (
                f"Expected {EXPECTED_TESTS} "
                f"tests; got {test_count}."
            ),
        )

        print(
            "Stage6 regression      =",
            f"{test_count} / {test_count} PASS",
        )

    except BaseException:

        if applied:

            print()
            print(
                "PATCH FAILURE -> EXACT ROLLBACK"
            )

            atomic_write(
                TARGET,
                original_bytes,
            )

            require(
                file_sha(
                    TARGET
                )
                ==
                EXPECTED_TARGET_PRE_SHA,
                (
                    "Rollback failed to restore "
                    "exact pre-patch runner."
                ),
            )

            rc_restore, out_restore = run(
                [
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(
                        TARGET
                    ),
                ]
            )

            require(
                rc_restore == 0,
                (
                    "Rollback compile failed:\n"
                    +
                    out_restore
                ),
            )

            print(
                "runner rollback = EXACT PASS"
            )

        raise

    # ========================================================
    # G. Post-patch AST readback
    # ========================================================

    print()
    print(
        "===== G. POST-PATCH AST READBACK ====="
    )

    post_bytes = TARGET.read_bytes()

    post_sha = file_sha(
        TARGET
    )

    require(
        post_sha
        !=
        EXPECTED_TARGET_PRE_SHA,
        (
            "Patched runner SHA "
            "did not change."
        ),
    )

    post_source = post_bytes.decode(
        "utf-8"
    )

    (
        _,
        post_node,
    ) = unique_resolve_grid(
        post_source
    )

    post_constants = function_constants(
        post_node
    )

    require(
        EXPECTED_BINDING_SHA
        in
        post_constants,
        (
            "Post-patch AST lost "
            "exact binding SHA."
        ),
    )

    require(
        "stage6_frozen_illumination_grid_runtime_binding.json"
        in
        post_constants,
        (
            "Post-patch AST lost "
            "binding filename."
        ),
    )

    print(
        "post-patch AST SHA seal = PASS"
    )

    print(
        "binding path seal        = PASS"
    )

    print(
        "patched runner SHA256    =",
        post_sha,
    )

    # ========================================================
    # H. Scientific boundary
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    require(
        not MAPS.exists(),
        (
            "Patch unexpectedly generated "
            "reactive maps."
        ),
    )

    require(
        not LEDGER.exists(),
        (
            "Patch unexpectedly generated "
            "decision ledger."
        ),
    )

    require(
        not DECISION_REPORT.exists(),
        (
            "Patch unexpectedly generated "
            "decision freeze report."
        ),
    )

    print(
        "reactive performance   = NOT READ"
    )

    print(
        "predictive performance = NOT READ"
    )

    print(
        "future truth           = NOT READ"
    )

    print(
        "P_occ content          = NOT OPENED"
    )

    print(
        "exact NI deltas        = NOT COMPUTED"
    )

    print(
        "scientific parameter   = UNCHANGED"
    )

    # ========================================================
    # I. Patch report
    # ========================================================

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8_Part2B_Part1_grid_binding_patch_v3",

        "status":
            "PASS_PART2B_PART1_GRID_BINDING_PATCHED_V3",

        "V2_failure": {
            "type":
                "source_text_validation_bug",

            "cause":
                (
                    "full frozen SHA was semantically "
                    "correct but source-text validation "
                    "expected contiguous characters"
                ),

            "scientific_failure":
                False,

            "target_modified_by_V2":
                False,
        },

        "repair": {
            "scope":
                "resolve_grid_only",

            "target_selection":
                "unique_top_level_AST_function",

            "whole_runner_pre_sha256":
                EXPECTED_TARGET_PRE_SHA,

            "old_function_sha256":
                old_function_sha,

            "old_function_lines":
                [
                    start,
                    end,
                ],

            "semantic_binding_SHA_validation":
                True,

            "heuristic_grid_discovery":
                False,

            "scientific_parameter_change":
                False,

            "numeric_invention":
                False,
        },

        "binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                EXPECTED_BINDING_SHA,
        },

        "runner": {
            "path":
                str(
                    TARGET
                ),

            "post_patch_sha256":
                post_sha,

            "backup_path":
                str(
                    BACKUP
                ),

            "backup_sha256":
                file_sha(
                    BACKUP
                ),
        },

        "scientific_boundary": {
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

            "formal_evaluation":
                False,
        },

        "regression": {
            "tests":
                test_count,

            "status":
                "PASS",
        },

        "next":
            (
                "rerun original Part2B Part1 "
                "reactive decision-ledger runner"
            ),
    }

    atomic_write(
        PATCH_REPORT,
        (
            json.dumps(
                result,
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
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART2B PART1 GRID PATCH V3 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "V2 scientific failure      = NO"
    )

    print(
        "V2 validation failure      = CONFIRMED"
    )

    print(
        "whole-runner pre SHA       = EXACT PASS"
    )

    print(
        "AST resolve_grid target    = UNIQUE"
    )

    print(
        "binding SHA validation     = SEMANTIC AST PASS"
    )

    print(
        "frozen grid binding        = EXACT PASS"
    )

    print(
        "resolve_grid repair        = APPLIED"
    )

    print(
        "heuristic grid discovery   = DISABLED"
    )

    print(
        "numeric invention          = NO"
    )

    print(
        "scientific parameter edit  = NO"
    )

    print(
        "controller/metric edit     = NO"
    )

    print(
        "reactive performance       = NOT READ"
    )

    print(
        "predictive performance     = NOT READ"
    )

    print(
        "future truth               = NOT READ"
    )

    print(
        "P_occ content              = NOT OPENED"
    )

    print(
        "exact NI deltas            = NOT COMPUTED"
    )

    print(
        "Stage6 regression          =",
        f"{test_count} / {test_count} PASS",
    )

    print(
        "patched runner SHA256      =",
        post_sha,
    )

    print(
        "STATUS = "
        "PASS_PART2B_PART1_GRID_BINDING_PATCHED_V3"
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH V3 = BLOCKED"
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

    print(
        "Do not manually edit scientific modules."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

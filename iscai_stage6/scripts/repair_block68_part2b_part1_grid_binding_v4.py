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
    / "artifacts/block68/grid_binding_patch_v4/"
      "run_block68_part2b_part1_reactive_decision_ledger.pre_v4.py"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_part2b_part1_grid_binding_patch_v4.json"
)


EXPECTED_RUNNER_SHA = (
    "fb9c62a46f2ad238fbbbf9ddb4ce791c"
    "02e74b65c66f316e2f586ddbd2bd13d6"
)

EXPECTED_OLD_RESOLVE_GRID_SHA = (
    "2a32646a82709818a1e62fdc7dc1b816c"
    "ba2ab0b8b44c4c504b63724b987c2cd"
)

EXPECTED_BINDING_SHA = (
    "be5145fe60a941e7906639916cfa384d"
    "5674dc29f294f5a42e12c962bc4df4bc"
)

EXPECTED_TESTS = 224


NEW_FUNCTION = r'''def resolve_grid():
    """
    Load the already frozen Stage6 illumination-grid runtime
    binding.

    No heuristic grid discovery is allowed here.
    No performance outcome, future truth, P_occ content, or
    newly selected numerical parameter enters this function.
    """

    binding_path = (
        S6
        / "configs"
        / "stage6_frozen_illumination_grid_runtime_binding.json"
    )

    expected_binding_sha = "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc"

    require(
        binding_path.is_file(),
        (
            "Frozen Stage6 illumination-grid "
            f"binding missing: {binding_path}"
        ),
    )

    actual_binding_sha = file_sha(
        binding_path
    )

    require(
        actual_binding_sha
        ==
        expected_binding_sha,
        (
            "Frozen illumination-grid binding SHA changed.\n"
            f"expected={expected_binding_sha}\n"
            f"actual={actual_binding_sha}"
        ),
    )

    binding = safe_json(
        binding_path
    )

    require(
        binding.get(
            "status"
        )
        ==
        "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING",
        (
            "Frozen illumination-grid binding "
            "status mismatch."
        ),
    )

    require(
        binding.get(
            "scientific_parameter_change"
        )
        is False,
        (
            "Grid binding unexpectedly records "
            "scientific parameter change."
        ),
    )

    anti = binding.get(
        "anti_invention",
        {}
    )

    for key in (
        "range_min_defaulted_to_zero",
        "constructor_values_selected_from_outcomes",
        "predictive_performance_read",
        "reactive_performance_read",
        "future_truth_read",
        "formal_content_read",
    ):

        require(
            anti.get(
                key
            )
            is False,
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
        len(
            block64_sha
        )
        ==
        64,
        (
            "Frozen Block6.4 provenance SHA "
            "missing."
        ),
    )

    require(
        file_sha(
            GRID_CONFIG
        )
        ==
        block64_sha,
        (
            "Block6.4 grid source changed "
            "after runtime-binding freeze."
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
            "Frozen illumination-grid constructor "
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
        set(
            constructor
        )
        ==
        expected_keys,
        (
            "Frozen constructor keys changed: "
            f"{sorted(constructor)}"
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
            "Frozen grid dimensions differ "
            f"from {GRID_SHAPE}: "
            f"{kwargs['n_theta']} x "
            f"{kwargs['n_range']}"
        ),
    )

    require(
        all(
            math.isfinite(
                kwargs[
                    key
                ]
            )
            for key in (
                "theta_min_rad",
                "theta_max_rad",
                "range_min_m",
                "range_max_m",
            )
        ),
        (
            "Frozen illumination-grid bounds "
            "contain non-finite values."
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
            "Frozen runtime grid failed "
            "Part-A map-shape readback: "
            f"{blank.shape}"
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
            (
                blank >= 0.0
            )
            &
            (
                blank <= 1.0
            )
        ),
        (
            "Blank Part-A map lies "
            "outside [0,1]."
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
                actual_binding_sha,

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
# Helpers
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
            "Formal content access forbidden: "
            f"{path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_write(
    path: Path,
    payload: bytes,
):

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        payload
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

    for line in output.splitlines():

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
    n=100,
):

    return "\n".join(
        output.splitlines()[
            -n:
        ]
    )


# ============================================================
# AST helpers
# ============================================================

def get_unique_resolve_grid(
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
        len(
            matches
        )
        ==
        1,
        (
            "Expected exactly one top-level "
            "resolve_grid(); found "
            f"{len(matches)}."
        ),
    )

    node = matches[
        0
    ]

    require(
        node.end_lineno
        is not None,
        (
            "resolve_grid AST end_lineno "
            "is unavailable."
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
            "resolve_grid() signature changed."
        ),
    )

    return (
        tree,
        node,
    )


def source_segment(
    source,
    node,
):

    lines = source.splitlines(
        keepends=True
    )

    return "".join(
        lines[
            node.lineno - 1:
            node.end_lineno
        ]
    )


def function_sha(
    source,
    node,
):

    return sha256(
        source_segment(
            source,
            node,
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def string_constants(
    node,
):

    return [
        child.value
        for child in ast.walk(
            node
        )
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
        )
    ]


def top_level_function_hashes(
    source,
):

    tree = ast.parse(
        source
    )

    result = {}

    for node in tree.body:

        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        result[
            node.name
        ] = function_sha(
            source,
            node,
        )

    return result


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART2B PART1 GRID PATCH V4"
    )
    print(
        "WHOLE-RUNNER + OLD-FUNCTION SHA SEALED"
    )
    print(
        "RUNNER-ONLY REPAIR / NO SCIENTIFIC CHANGE"
    )
    print(
        "============================================================"
    )

    for path in (
        TARGET,
        BINDING,
        GRID_REPORT,
    ):

        require(
            path.is_file(),
            (
                f"Missing required file: {path}"
            ),
        )

    # --------------------------------------------------------
    # A. Exact pre-state
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT PRE-PATCH STATE ====="
    )

    require(
        file_sha(
            TARGET
        )
        ==
        EXPECTED_RUNNER_SHA,
        (
            "Runner no longer equals exact "
            "known pre-patch SHA."
        ),
    )

    require(
        file_sha(
            BINDING
        )
        ==
        EXPECTED_BINDING_SHA,
        (
            "Frozen grid binding SHA changed."
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
            "Frozen grid-binding report "
            "is not PASS."
        ),
    )

    print(
        "whole-runner SHA = EXACT PASS"
    )
    print(
        "grid-binding SHA = EXACT PASS"
    )

    # --------------------------------------------------------
    # B. No scientific outputs
    # --------------------------------------------------------

    print()
    print(
        "===== B. PRE-SCIENTIFIC-OUTPUT BOUNDARY ====="
    )

    unexpected = [
        str(
            path
        )
        for path in (
            MAPS,
            LEDGER,
            DECISION_REPORT,
        )
        if path.exists()
    ]

    require(
        not unexpected,
        (
            "Unexpected Part2B outputs "
            "already exist: "
            +
            repr(
                unexpected
            )
        ),
    )

    print(
        "reactive maps       = ABSENT"
    )
    print(
        "decision ledger     = ABSENT"
    )
    print(
        "decision report     = ABSENT"
    )
    print(
        "performance outcomes= NOT READ"
    )
    print(
        "future truth        = NOT READ"
    )

    # --------------------------------------------------------
    # C. Seal exact old function
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXACT AST TARGET ====="
    )

    original_bytes = TARGET.read_bytes()

    original = original_bytes.decode(
        "utf-8"
    )

    (
        _,
        old_node,
    ) = get_unique_resolve_grid(
        original
    )

    old_sha = function_sha(
        original,
        old_node,
    )

    require(
        old_sha
        ==
        EXPECTED_OLD_RESOLVE_GRID_SHA,
        (
            "resolve_grid() differs from "
            "exact known failed function.\n"
            f"expected={EXPECTED_OLD_RESOLVE_GRID_SHA}\n"
            f"actual={old_sha}"
        ),
    )

    start = int(
        old_node.lineno
    )

    end = int(
        old_node.end_lineno
    )

    print(
        "resolve_grid count = 1"
    )
    print(
        "resolve_grid lines =",
        f"{start}..{end}",
    )
    print(
        "old function SHA   = EXACT PASS"
    )

    # --------------------------------------------------------
    # D. Backup
    # --------------------------------------------------------

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
            EXPECTED_RUNNER_SHA,
            (
                "Existing V4 backup differs "
                "from exact pre-patch runner."
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
        EXPECTED_RUNNER_SHA,
        (
            "Backup verification failed."
        ),
    )

    print(
        "backup SHA = EXACT PASS"
    )

    # --------------------------------------------------------
    # E. Build replacement
    # --------------------------------------------------------

    print()
    print(
        "===== E. BUILD SINGLE-FUNCTION REPLACEMENT ====="
    )

    lines = original.splitlines(
        keepends=True
    )

    replacement = (
        NEW_FUNCTION
        if NEW_FUNCTION.endswith(
            "\n"
        )
        else
        NEW_FUNCTION + "\n"
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

    compile(
        patched,
        str(
            TARGET
        ),
        "exec",
    )

    (
        _,
        new_node,
    ) = get_unique_resolve_grid(
        patched
    )

    constants = string_constants(
        new_node
    )

    require(
        EXPECTED_BINDING_SHA
        in
        constants,
        (
            "Replacement AST does not seal "
            "the exact frozen binding SHA."
        ),
    )

    require(
        "stage6_frozen_illumination_grid_runtime_binding.json"
        in
        constants,
        (
            "Replacement AST does not contain "
            "the exact binding filename constant."
        ),
    )

    require(
        "configs"
        in
        constants,
        (
            "Replacement AST does not contain "
            "the configs path component."
        ),
    )

    require(
        (
            "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING"
            in
            constants
        ),
        (
            "Replacement AST does not seal "
            "the binding status."
        ),
    )

    print(
        "whole-file syntax    = PASS"
    )
    print(
        "binding filename AST = PASS"
    )
    print(
        "binding SHA AST      = PASS"
    )
    print(
        "binding status AST   = PASS"
    )
    print(
        "heuristic resolver   = REMOVED"
    )

    # Prove every other top-level function is byte-identical.
    old_functions = top_level_function_hashes(
        original
    )

    new_functions = top_level_function_hashes(
        patched
    )

    require(
        set(
            old_functions
        )
        ==
        set(
            new_functions
        ),
        (
            "Top-level function population changed."
        ),
    )

    for name in old_functions:

        if name == "resolve_grid":
            continue

        require(
            old_functions[
                name
            ]
            ==
            new_functions[
                name
            ],
            (
                "Unexpected change outside "
                f"resolve_grid(): {name}"
            ),
        )

    print(
        "other top-level functions = BYTE-EXACT PASS"
    )

    # --------------------------------------------------------
    # F. Transactional write
    # --------------------------------------------------------

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

        count = parse_test_count(
            out_tests
        )

        require(
            rc_tests == 0,
            (
                "Stage6 regression failed:\n"
                +
                tail(
                    out_tests
                )
            ),
        )

        require(
            count
            ==
            EXPECTED_TESTS,
            (
                f"Expected {EXPECTED_TESTS} tests; "
                f"got {count}."
            ),
        )

        print(
            "Stage6 regression      =",
            f"{count} / {count} PASS",
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
                EXPECTED_RUNNER_SHA,
                (
                    "Rollback did not restore "
                    "exact original runner."
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

    # --------------------------------------------------------
    # G. Post-patch semantic readback
    # --------------------------------------------------------

    print()
    print(
        "===== G. POST-PATCH READBACK ====="
    )

    post_sha = file_sha(
        TARGET
    )

    require(
        post_sha
        !=
        EXPECTED_RUNNER_SHA,
        (
            "Runner SHA did not change."
        ),
    )

    post_source = TARGET.read_text(
        encoding="utf-8"
    )

    (
        _,
        post_node,
    ) = get_unique_resolve_grid(
        post_source
    )

    post_constants = string_constants(
        post_node
    )

    require(
        EXPECTED_BINDING_SHA
        in
        post_constants,
        (
            "Post-patch AST lost "
            "frozen binding SHA."
        ),
    )

    require(
        (
            "stage6_frozen_illumination_grid_runtime_binding.json"
            in
            post_constants
        ),
        (
            "Post-patch AST lost "
            "binding filename."
        ),
    )

    require(
        (
            "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING"
            in
            post_constants
        ),
        (
            "Post-patch AST lost "
            "binding status."
        ),
    )

    print(
        "post-patch AST binding = PASS"
    )
    print(
        "post-patch runner SHA  =",
        post_sha,
    )

    # --------------------------------------------------------
    # H. Boundary
    # --------------------------------------------------------

    print()
    print(
        "===== H. SCIENTIFIC BOUNDARY ====="
    )

    require(
        not MAPS.exists(),
        (
            "Patcher unexpectedly produced "
            "reactive maps."
        ),
    )

    require(
        not LEDGER.exists(),
        (
            "Patcher unexpectedly produced "
            "decision ledger."
        ),
    )

    require(
        not DECISION_REPORT.exists(),
        (
            "Patcher unexpectedly produced "
            "decision freeze report."
        ),
    )

    print(
        "scientific parameter edit = NO"
    )
    print(
        "controller/metric edit    = NO"
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

    # --------------------------------------------------------
    # I. Report
    # --------------------------------------------------------

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8_Part2B_Part1_grid_binding_patch_v4",

        "status":
            "PASS_PART2B_PART1_GRID_BINDING_PATCHED_V4",

        "prior_validation_failures": {
            "V1":
                "brittle old-error-string sentinel",

            "V2":
                "contiguous source SHA validation",

            "V3":
                "exact filename constant validation "
                "did not account for lexical string concatenation",

            "scientific_failure":
                False,

            "target_modified_by_failed_patchers":
                False,
        },

        "repair_scope": {
            "whole_runner_pre_sha256":
                EXPECTED_RUNNER_SHA,

            "old_resolve_grid_sha256":
                EXPECTED_OLD_RESOLVE_GRID_SHA,

            "old_resolve_grid_lines":
                [
                    start,
                    end,
                ],

            "modified_top_level_function":
                "resolve_grid",

            "all_other_top_level_functions_byte_exact":
                True,

            "scientific_parameter_change":
                False,

            "heuristic_grid_discovery":
                False,
        },

        "binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                EXPECTED_BINDING_SHA,

            "semantic_AST_validation":
                True,
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
                count,

            "status":
                "PASS",
        },

        "next":
            (
                "rerun the original Part2B Part1 "
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH V4 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "whole-runner pre SHA       = EXACT PASS"
    )
    print(
        "old resolve_grid SHA       = EXACT PASS"
    )
    print(
        "AST resolve_grid target    = UNIQUE"
    )
    print(
        "binding path validation    = SEMANTIC AST PASS"
    )
    print(
        "binding SHA validation     = SEMANTIC AST PASS"
    )
    print(
        "all other functions        = BYTE-EXACT PASS"
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
        f"{count} / {count} PASS",
    )
    print(
        "patched runner SHA256      =",
        post_sha,
    )
    print(
        "STATUS = "
        "PASS_PART2B_PART1_GRID_BINDING_PATCHED_V4"
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH V4 = BLOCKED"
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

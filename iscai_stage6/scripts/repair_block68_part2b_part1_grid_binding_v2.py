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

BACKUP = (
    S6
    / "artifacts/block68/grid_binding_patch_v2/"
      "run_block68_part2b_part1_reactive_decision_ledger.pre_v2.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_part2b_part1_grid_binding_patch_v2.json"
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
    Block6.8 Part2B frozen runtime-grid binding.

    This runner performs NO grid-parameter discovery.

    All six IlluminationGridSpec constructor values come
    exclusively from the already frozen pre-outcome Stage6
    runtime-binding artifact.
    """

    binding_path = (
        S6
        / "configs/"
          "stage6_frozen_illumination_grid_runtime_binding.json"
    )

    expected_sha = (
        "be5145fe60a941e7906639916cfa384d"
        "5674dc29f294f5a42e12c962bc4df4bc"
    )

    require(
        binding_path.is_file(),
        (
            "Frozen grid binding missing: "
            f"{binding_path}"
        ),
    )

    actual_sha = file_sha(
        binding_path
    )

    require(
        actual_sha == expected_sha,
        (
            "Frozen grid binding SHA changed.\n"
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
            "Frozen grid-binding status mismatch."
        ),
    )

    require(
        binding.get(
            "scientific_parameter_change"
        )
        is False,
        (
            "Frozen binding unexpectedly records "
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
            "Block6.4 provenance SHA "
            "missing from frozen binding."
        ),
    )

    require(
        file_sha(GRID_CONFIG)
        ==
        block64_sha,
        (
            "Frozen Block6.4 grid source changed."
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
            "Frozen grid-constructor keys changed: "
            f"{sorted(constructor)}"
        ),
    )

    kwargs = {
        "theta_min_rad":
            float(
                constructor["theta_min_rad"]
            ),

        "theta_max_rad":
            float(
                constructor["theta_max_rad"]
            ),

        "range_min_m":
            float(
                constructor["range_min_m"]
            ),

        "range_max_m":
            float(
                constructor["range_max_m"]
            ),

        "n_theta":
            int(
                constructor["n_theta"]
            ),

        "n_range":
            int(
                constructor["n_range"]
            ),
    }

    require(
        (
            kwargs["n_theta"],
            kwargs["n_range"],
        )
        ==
        GRID_SHAPE,
        (
            "Frozen grid shape differs from "
            f"{GRID_SHAPE}."
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
            "Non-finite frozen grid bounds."
        ),
    )

    require(
        kwargs["theta_min_rad"]
        <
        kwargs["theta_max_rad"],
        (
            "Theta bounds are not increasing."
        ),
    )

    require(
        kwargs["range_min_m"]
        <
        kwargs["range_max_m"],
        (
            "Range bounds are not increasing."
        ),
    )

    # Exact readback of the already frozen binding.
    exact_values = {
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
        exact_values.items()
    ):

        require(
            math.isclose(
                kwargs[key],
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            (
                f"Frozen {key} changed: "
                f"{kwargs[key]}"
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
            np.isfinite(blank)
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
                str(binding_path),

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

    with path.open("rb") as stream:

        while True:

            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_json(
    path: Path,
):

    require(
        "formal"
        not in
        str(path.resolve()).lower(),
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
    data: bytes,
):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_bytes(data)

    os.replace(
        temp,
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
        output.splitlines()[-n:]
    )


def locate_resolve_grid(
    source: str,
):

    tree = ast.parse(
        source,
        filename=str(TARGET),
    )

    nodes = [
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name == "resolve_grid"
        )
    ]

    require(
        len(nodes) == 1,
        (
            "Expected exactly one top-level "
            f"resolve_grid(); found {len(nodes)}."
        ),
    )

    node = nodes[0]

    require(
        node.end_lineno is not None,
        (
            "resolve_grid() AST end line missing."
        ),
    )

    require(
        len(node.args.args) == 0
        and
        len(node.args.kwonlyargs) == 0
        and
        node.args.vararg is None
        and
        node.args.kwarg is None,
        (
            "resolve_grid() signature changed; "
            "refusing patch."
        ),
    )

    return node


def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART2B PART1 GRID PATCH V2"
    )
    print(
        "WHOLE-RUNNER SHA SEALED / AST FUNCTION REPLACEMENT"
    )
    print(
        "============================================================"
    )

    require(
        TARGET.is_file(),
        f"Missing runner: {TARGET}",
    )

    require(
        BINDING.is_file(),
        f"Missing binding: {BINDING}",
    )

    require(
        GRID_REPORT.is_file(),
        (
            "Missing frozen grid-binding report."
        ),
    )

    # ========================================================
    # A. Exact state
    # ========================================================

    print()
    print(
        "===== A. EXACT PRE-PATCH STATE ====="
    )

    actual_target_sha = file_sha(
        TARGET
    )

    require(
        actual_target_sha
        ==
        EXPECTED_TARGET_PRE_SHA,
        (
            "Runner no longer equals the exact "
            "known pre-patch state.\n"
            f"expected={EXPECTED_TARGET_PRE_SHA}\n"
            f"actual={actual_target_sha}"
        ),
    )

    require(
        file_sha(BINDING)
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
        grid_report.get("status")
        ==
        "PASS_FROZEN_GRID_RUNTIME_BINDING",
        (
            "Frozen grid-binding report is not PASS."
        ),
    )

    print(
        "runner pre-patch SHA = EXACT PASS"
    )

    print(
        "grid binding SHA     = EXACT PASS"
    )

    print(
        "scientific grid      = ALREADY FROZEN"
    )

    # ========================================================
    # B. Confirm no prior outputs
    # ========================================================

    print()
    print(
        "===== B. SCIENTIFIC OUTPUT BOUNDARY ====="
    )

    existing_scientific_outputs = [
        str(path)
        for path in (
            MAPS,
            LEDGER,
            DECISION_REPORT,
        )
        if path.exists()
    ]

    require(
        not existing_scientific_outputs,
        (
            "Unexpected scientific outputs "
            "already exist: "
            +
            repr(existing_scientific_outputs)
        ),
    )

    print(
        "reactive maps       = ABSENT"
    )

    print(
        "decision ledger     = ABSENT"
    )

    print(
        "freeze report       = ABSENT"
    )

    print(
        "performance outcomes= NOT READ"
    )

    print(
        "future truth        = NOT READ"
    )

    # ========================================================
    # C. Exact AST target
    # ========================================================

    print()
    print(
        "===== C. AST PATCH TARGET ====="
    )

    original_bytes = TARGET.read_bytes()

    original = original_bytes.decode(
        "utf-8"
    )

    node = locate_resolve_grid(
        original
    )

    lines = original.splitlines(
        keepends=True
    )

    start = int(
        node.lineno
    )

    end = int(
        node.end_lineno
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
            "resolve_grid() already references "
            "the frozen runtime binding."
        ),
    )

    print(
        "resolve_grid count    = 1"
    )

    print(
        "resolve_grid lines    =",
        f"{start}..{end}",
    )

    print(
        "old function SHA256   =",
        old_function_sha,
    )

    print(
        "brittle string gate   = REMOVED"
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
            file_sha(BACKUP)
            ==
            EXPECTED_TARGET_PRE_SHA,
            (
                "Existing V2 backup is not the "
                "known pre-patch runner."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    require(
        file_sha(BACKUP)
        ==
        EXPECTED_TARGET_PRE_SHA,
        (
            "Backup SHA verification failed."
        ),
    )

    print(
        "backup SHA256 = EXACT PASS"
    )

    # ========================================================
    # E. Construct patched file
    # ========================================================

    print()
    print(
        "===== E. BUILD SINGLE-FUNCTION PATCH ====="
    )

    replacement = (
        NEW_FUNCTION
        if NEW_FUNCTION.endswith("\n")
        else
        NEW_FUNCTION + "\n"
    )

    patched = "".join(
        lines[:start - 1]
        +
        [replacement]
        +
        lines[end:]
    )

    compile(
        patched,
        str(TARGET),
        "exec",
    )

    # Structural proof after replacement.
    patched_node = locate_resolve_grid(
        patched
    )

    patched_lines = patched.splitlines(
        keepends=True
    )

    patched_function = "".join(
        patched_lines[
            patched_node.lineno - 1:
            patched_node.end_lineno
        ]
    )

    require(
        (
            "stage6_frozen_illumination_grid_runtime_binding.json"
            in
            patched_function
        ),
        (
            "Replacement function does not "
            "load frozen grid binding."
        ),
    )

    require(
        EXPECTED_BINDING_SHA
        in
        patched_function,
        (
            "Replacement function does not "
            "seal frozen binding SHA."
        ),
    )

    print(
        "changed top-level function = resolve_grid ONLY"
    )

    print(
        "replacement syntax         = PASS"
    )

    print(
        "heuristic grid discovery   = REMOVED"
    )

    print(
        "numeric invention          = NO"
    )

    # ========================================================
    # F. Transactional write and gates
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
                str(TARGET),
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
                tail(out_tests)
            ),
        )

        require(
            count == EXPECTED_TESTS,
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
                "PATCH FAILURE -> AUTOMATIC ROLLBACK"
            )

            atomic_write(
                TARGET,
                original_bytes,
            )

            require(
                file_sha(TARGET)
                ==
                EXPECTED_TARGET_PRE_SHA,
                (
                    "Rollback did not restore "
                    "exact runner SHA."
                ),
            )

            rc_restore, out_restore = run(
                [
                    sys.executable,
                    "-m",
                    "py_compile",
                    str(TARGET),
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
    # G. Exact post-patch readback
    # ========================================================

    print()
    print(
        "===== G. POST-PATCH READBACK ====="
    )

    new_sha = file_sha(
        TARGET
    )

    require(
        new_sha
        !=
        EXPECTED_TARGET_PRE_SHA,
        (
            "Runner SHA did not change."
        ),
    )

    post_source = TARGET.read_text(
        encoding="utf-8"
    )

    post_node = locate_resolve_grid(
        post_source
    )

    post_lines = post_source.splitlines(
        keepends=True
    )

    post_function = "".join(
        post_lines[
            post_node.lineno - 1:
            post_node.end_lineno
        ]
    )

    require(
        (
            "stage6_frozen_illumination_grid_runtime_binding.json"
            in
            post_function
        ),
        (
            "Post-patch resolve_grid binding "
            "reference missing."
        ),
    )

    require(
        (
            "heuristic semantic-key"
            not in
            post_function.lower()
        ),
        (
            "Unexpected heuristic resolver "
            "language remains."
        ),
    )

    print(
        "post-patch runner SHA =",
        new_sha,
    )

    print(
        "binding reference      = PASS"
    )

    print(
        "binding SHA seal       = PASS"
    )

    print(
        "runner-only repair     = PASS"
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
            "6.8_Part2B_Part1_grid_binding_patch_v2",

        "status":
            "PASS_PART2B_PART1_GRID_BINDING_PATCHED_V2",

        "reason_for_v2":
            (
                "prior patcher used brittle literal-string "
                "sentinel despite successful AST location"
            ),

        "repair_scope":
            "exact top-level resolve_grid function only",

        "scientific_change":
            False,

        "grid_binding": {
            "path":
                str(BINDING),

            "sha256":
                EXPECTED_BINDING_SHA,
        },

        "runner": {
            "path":
                str(TARGET),

            "pre_patch_sha256":
                EXPECTED_TARGET_PRE_SHA,

            "post_patch_sha256":
                new_sha,

            "backup_path":
                str(BACKUP),

            "backup_sha256":
                file_sha(BACKUP),

            "old_resolve_grid_sha256":
                old_function_sha,

            "old_resolve_grid_lines":
                [
                    start,
                    end,
                ],
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
                "rerun unchanged Part2B Part1 "
                "reactive decision-ledger runner"
            ),
    }

    atomic_write(
        REPORT,
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH V2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "pre-patch whole-runner SHA = EXACT PASS"
    )

    print(
        "AST resolve_grid target     = UNIQUE"
    )

    print(
        "brittle literal gate        = REMOVED"
    )

    print(
        "frozen grid binding         = EXACT PASS"
    )

    print(
        "resolve_grid repair         = APPLIED"
    )

    print(
        "heuristic grid discovery    = DISABLED"
    )

    print(
        "numeric invention           = NO"
    )

    print(
        "scientific parameter edit   = NO"
    )

    print(
        "controller/metric edit      = NO"
    )

    print(
        "reactive performance        = NOT READ"
    )

    print(
        "predictive performance      = NOT READ"
    )

    print(
        "future truth                = NOT READ"
    )

    print(
        "P_occ content               = NOT OPENED"
    )

    print(
        "exact NI deltas             = NOT COMPUTED"
    )

    print(
        "Stage6 regression           =",
        f"{count} / {count} PASS",
    )

    print(
        "patched runner SHA256       =",
        new_sha,
    )

    print(
        "STATUS = "
        "PASS_PART2B_PART1_GRID_BINDING_PATCHED_V2"
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH V2 = BLOCKED"
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

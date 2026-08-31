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


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

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

BINDING_REPORT = (
    S6
    / "reports/"
      "block68_grid_runtime_binding_repair.json"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_part2b_part1_grid_binding_patch.json"
)

BACKUP = (
    S6
    / "artifacts/block68/"
      "grid_binding_patch/"
      "run_block68_part2b_part1_reactive_decision_ledger.pre_patch.py"
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


EXPECTED_BINDING_SHA = (
    "be5145fe60a941e7906639916cfa384d"
    "5674dc29f294f5a42e12c962bc4df4bc"
)

EXPECTED_TESTS = 224


NEW_RESOLVE_GRID = r'''def resolve_grid():
    """
    Block6.8 Part2B runtime-binding repair.

    Load the already frozen Stage6 illumination-grid
    constructor instead of performing heuristic semantic-key
    discovery.

    Scientific parameters are NOT selected here.

    Binding authority:
      configs/
      stage6_frozen_illumination_grid_runtime_binding.json
    """

    binding_path = (
        S6
        / "configs/"
          "stage6_frozen_illumination_grid_runtime_binding.json"
    )

    expected_binding_sha = (
        "be5145fe60a941e7906639916cfa384d"
        "5674dc29f294f5a42e12c962bc4df4bc"
    )

    require(
        binding_path.is_file(),
        (
            "Frozen illumination-grid runtime "
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
            "Frozen illumination-grid binding "
            "SHA changed.\n"
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
            "a scientific parameter change."
        ),
    )

    anti = binding.get(
        "anti_invention",
        {}
    )

    require(
        anti.get(
            "range_min_defaulted_to_zero"
        )
        is False,
        (
            "Frozen binding indicates an invented "
            "range_min_m."
        ),
    )

    require(
        anti.get(
            "constructor_values_selected_from_outcomes"
        )
        is False,
        (
            "Grid constructor was selected "
            "from outcomes."
        ),
    )

    require(
        anti.get(
            "predictive_performance_read"
        )
        is False,
        (
            "Predictive performance entered "
            "grid binding."
        ),
    )

    require(
        anti.get(
            "reactive_performance_read"
        )
        is False,
        (
            "Reactive performance entered "
            "grid binding."
        ),
    )

    require(
        anti.get(
            "future_truth_read"
        )
        is False,
        (
            "Future truth entered grid binding."
        ),
    )

    require(
        anti.get(
            "formal_content_read"
        )
        is False,
        (
            "Formal content entered grid binding."
        ),
    )

    provenance = binding.get(
        "provenance",
        {}
    )

    frozen_block64_sha = (
        provenance.get(
            "Block64_config_sha256"
        )
    )

    require(
        isinstance(
            frozen_block64_sha,
            str,
        )
        and
        len(
            frozen_block64_sha
        )
        ==
        64,
        (
            "Frozen Block6.4 provenance SHA "
            "missing from grid binding."
        ),
    )

    require(
        file_sha(
            GRID_CONFIG
        )
        ==
        frozen_block64_sha,
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
            "must be a dict."
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
            "Frozen constructor contains "
            "non-finite bounds."
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
            "Frozen grid counts differ "
            f"from {GRID_SHAPE}: "
            f"{kwargs['n_theta']} x "
            f"{kwargs['n_range']}"
        ),
    )

    # Exact frozen-value readback.
    require(
        math.isclose(
            kwargs[
                "theta_min_rad"
            ],
            -0.4363323129985824,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        (
            "Frozen theta_min_rad "
            "does not match sealed binding."
        ),
    )

    require(
        math.isclose(
            kwargs[
                "theta_max_rad"
            ],
            0.4363323129985824,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        (
            "Frozen theta_max_rad "
            "does not match sealed binding."
        ),
    )

    require(
        math.isclose(
            kwargs[
                "range_min_m"
            ],
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        (
            "Frozen range_min_m "
            "does not match sealed binding."
        ),
    )

    require(
        math.isclose(
            kwargs[
                "range_max_m"
            ],
            150.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        (
            "Frozen range_max_m "
            "does not match sealed binding."
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
            "Frozen runtime grid fails "
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
            "Frozen blank Part-A map "
            "contains non-finite values."
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
            "Frozen blank Part-A map "
            "is outside [0,1]."
        ),
    )

    attempts = [
        {
            "label":
                (
                    "frozen Stage6 illumination-grid "
                    "runtime binding"
                ),

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

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


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
    data: bytes,
):

    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_bytes(
        data
    )

    os.replace(
        temporary,
        path,
    )


def run_command(
    command,
):

    process = subprocess.run(
        command,
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return (
        process.returncode,
        process.stdout,
    )


def test_count(
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
    count=100,
):

    return "\n".join(
        output.splitlines()[
            -count:
        ]
    )


# ============================================================
# Patch
# ============================================================

def build_patched_source(
    source: str,
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
            "AST does not expose "
            "resolve_grid end line."
        ),
    )

    lines = source.splitlines(
        keepends=True
    )

    start = int(
        node.lineno
    )

    end = int(
        node.end_lineno
    )

    original_function = "".join(
        lines[
            start - 1:
            end
        ]
    )

    require(
        (
            "Could not prove frozen illumination-grid "
            "constructor without inventing numerics."
        )
        in
        original_function,
        (
            "Target resolve_grid() no longer matches "
            "the known failed heuristic resolver."
        ),
    )

    require(
        "stage6_frozen_illumination_grid_runtime_binding.json"
        not in
        original_function,
        (
            "resolve_grid() appears to have "
            "already been repaired."
        ),
    )

    replacement = (
        NEW_RESOLVE_GRID
        if NEW_RESOLVE_GRID.endswith(
            "\n"
        )
        else
        NEW_RESOLVE_GRID
        +
        "\n"
    )

    new_lines = (
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

    patched = "".join(
        new_lines
    )

    compile(
        patched,
        str(
            TARGET
        ),
        "exec",
    )

    return (
        patched,
        start,
        end,
        original_function,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART2B PART1 — GRID-BINDING PATCH"
    )
    print(
        "RUNNER-ONLY REPAIR / NO SCIENTIFIC CHANGE"
    )
    print(
        "============================================================"
    )

    require(
        TARGET.is_file(),
        (
            "Decision-ledger runner missing: "
            f"{TARGET}"
        ),
    )

    require(
        BINDING.is_file(),
        (
            "Frozen grid binding missing: "
            f"{BINDING}"
        ),
    )

    require(
        BINDING_REPORT.is_file(),
        (
            "Grid-binding repair report missing: "
            f"{BINDING_REPORT}"
        ),
    )

    # ========================================================
    # A. Grid-binding authority
    # ========================================================

    print()
    print(
        "===== A. FROZEN GRID-BINDING AUTHORITY ====="
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

    binding = safe_json(
        BINDING
    )

    grid_report = safe_json(
        BINDING_REPORT
    )

    require(
        binding.get(
            "status"
        )
        ==
        "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING",
        (
            "Grid binding is not frozen."
        ),
    )

    require(
        grid_report.get(
            "status"
        )
        ==
        "PASS_FROZEN_GRID_RUNTIME_BINDING",
        (
            "Grid-binding repair report "
            "is not PASS."
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
            "Reactive outcomes were read "
            "before runner patch."
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
            "Predictive outcomes were read "
            "before runner patch."
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
            "Future truth was read "
            "before runner patch."
        ),
    )

    print(
        "binding SHA256         = EXACT PASS"
    )

    print(
        "binding status         = FROZEN"
    )

    print(
        "scientific grid change = NO"
    )

    print(
        "performance outcomes   = NOT READ"
    )

    print(
        "future truth           = NOT READ"
    )

    # ========================================================
    # B. No scientific outputs from failed prior run
    # ========================================================

    print()
    print(
        "===== B. PRIOR FAILED-RUN OUTPUT BOUNDARY ====="
    )

    unexpected_outputs = [
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
        not unexpected_outputs,
        (
            "Decision-ledger outputs already exist "
            "despite previous pre-output BLOCKED run: "
            +
            repr(
                unexpected_outputs
            )
        ),
    )

    print(
        "t0 map artifact       = ABSENT"
    )

    print(
        "decision ledger       = ABSENT"
    )

    print(
        "decision freeze report= ABSENT"
    )

    print(
        "previous failure point= PRE-SCIENTIFIC-OUTPUT PASS"
    )

    # ========================================================
    # C. Idempotent patch readback
    # ========================================================

    if PATCH_REPORT.is_file():

        existing = safe_json(
            PATCH_REPORT
        )

        if (
            existing.get(
                "status"
            )
            ==
            "PASS_PART2B_PART1_GRID_BINDING_PATCHED"
        ):

            require(
                file_sha(
                    TARGET
                )
                ==
                existing[
                    "patched_runner"
                ][
                    "sha256"
                ],
                (
                    "Existing patched runner "
                    "SHA changed."
                ),
            )

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

            count = test_count(
                output
            )

            require(
                (
                    rc == 0
                    and
                    count == EXPECTED_TESTS
                ),
                (
                    "Existing patch regression failed:\n"
                    +
                    tail(
                        output
                    )
                ),
            )

            print()
            print(
                "===== EXISTING PATCH DETECTED ====="
            )

            print(
                "patched runner     = EXACT PASS"
            )

            print(
                "Stage6 regression  = 224 / 224 PASS"
            )

            print(
                "STATUS = "
                "PASS_PART2B_PART1_GRID_BINDING_PATCHED"
            )

            print(
                "no rewrite performed = YES"
            )

            print(
                "terminal remains open = YES"
            )

            return

    # ========================================================
    # D. Backup
    # ========================================================

    print()
    print(
        "===== C. VERIFIED RUNNER BACKUP ====="
    )

    original = TARGET.read_bytes()

    old_sha = file_sha(
        TARGET
    )

    BACKUP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if BACKUP.exists():

        require(
            BACKUP.read_bytes()
            ==
            original,
            (
                "Existing backup differs "
                "from current pre-patch runner."
            ),
        )

    else:

        BACKUP.write_bytes(
            original
        )

    require(
        BACKUP.read_bytes()
        ==
        original,
        (
            "Runner backup verification failed."
        ),
    )

    print(
        "pre-patch runner SHA256 =",
        old_sha,
    )

    print(
        "backup verification       = PASS"
    )

    # ========================================================
    # E. Build exact minimal patch
    # ========================================================

    print()
    print(
        "===== D. BUILD MINIMAL resolve_grid() PATCH ====="
    )

    (
        patched_source,
        start_line,
        end_line,
        old_function,
    ) = build_patched_source(
        original.decode(
            "utf-8"
        )
    )

    patched_bytes = patched_source.encode(
        "utf-8"
    )

    print(
        "patched function = resolve_grid() ONLY"
    )

    print(
        "old source lines =",
        f"{start_line}..{end_line}",
    )

    print(
        "frozen binding   =",
        BINDING,
    )

    print(
        "numeric invention= NO"
    )

    # ========================================================
    # F. Transactional patch + compile + regression
    # ========================================================

    print()
    print(
        "===== E. TRANSACTIONAL PATCH ====="
    )

    applied = False

    try:

        atomic_write(
            TARGET,
            patched_bytes,
        )

        applied = True

        rc_compile, compile_output = run_command(
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
                compile_output
            ),
        )

        print(
            "patched runner compile = PASS"
        )

        rc_tests, test_output = run_command(
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

        count = test_count(
            test_output
        )

        require(
            rc_tests == 0,
            (
                "Stage6 regression failed "
                "after runner patch:\n"
                +
                tail(
                    test_output
                )
            ),
        )

        require(
            count
            ==
            EXPECTED_TESTS,
            (
                "Unexpected regression count "
                f"after patch: {count}; "
                f"expected {EXPECTED_TESTS}."
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
                "PATCH FAILURE -> "
                "AUTOMATIC RUNNER ROLLBACK"
            )

            atomic_write(
                TARGET,
                original,
            )

            rc_restore, restore_output = run_command(
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
                    "Runner rollback compile failed:\n"
                    +
                    restore_output
                ),
            )

            require(
                file_sha(
                    TARGET
                )
                ==
                old_sha,
                (
                    "Runner rollback SHA mismatch."
                ),
            )

            print(
                "runner rollback = EXACT PASS"
            )

        raise

    # ========================================================
    # G. Patch-content readback
    # ========================================================

    print()
    print(
        "===== F. PATCH READBACK ====="
    )

    new_sha = file_sha(
        TARGET
    )

    patched_text = TARGET.read_text(
        encoding="utf-8"
    )

    require(
        (
            "stage6_frozen_illumination_grid_runtime_binding.json"
            in
            patched_text
        ),
        (
            "Patched runner does not reference "
            "frozen grid binding."
        ),
    )

    require(
        EXPECTED_BINDING_SHA
        in
        patched_text,
        (
            "Patched runner does not seal "
            "binding SHA."
        ),
    )

    require(
        (
            "range_min_defaulted_to_zero"
            in
            patched_text
        ),
        (
            "Anti-invention readback missing."
        ),
    )

    require(
        (
            "Could not prove frozen illumination-grid "
            "constructor without inventing numerics."
        )
        not in
        patched_text[
            patched_text.find(
                "def resolve_grid"
            ):
            patched_text.find(
                "# ============================================================",
                patched_text.find(
                    "def resolve_grid"
                )
            )
        ],
        (
            "Old heuristic resolver appears "
            "to remain active."
        ),
    )

    print(
        "frozen binding reference = PASS"
    )

    print(
        "binding SHA seal          = PASS"
    )

    print(
        "old heuristic resolver    = REMOVED"
    )

    print(
        "scientific code edited     = NO"
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
            "6.8_Part2B_Part1_grid_binding_patch",

        "status":
            "PASS_PART2B_PART1_GRID_BINDING_PATCHED",

        "repair_scope":
            "runner_resolve_grid_only",

        "scientific_change":
            False,

        "frozen_binding": {
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

            "pre_patch_sha256":
                old_sha,

            "backup_path":
                str(
                    BACKUP
                ),

            "backup_sha256":
                file_sha(
                    BACKUP
                ),

            "resolve_grid_original_lines":
                [
                    start_line,
                    end_line,
                ],
        },

        "patched_runner": {
            "sha256":
                new_sha,

            "compile":
                "PASS",

            "binding_load":
                "EXACT_FROZEN_CONFIG",

            "heuristic_grid_discovery":
                False,

            "numeric_values_invented":
                False,
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
                "rerun the same Part2B Part1 "
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "frozen grid binding       = EXACT PASS"
    )

    print(
        "resolve_grid repair       = APPLIED"
    )

    print(
        "heuristic grid discovery  = DISABLED"
    )

    print(
        "numeric invention         = NO"
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

    print(
        "Stage6 regression         =",
        f"{count} / {count} PASS",
    )

    print(
        "patched runner SHA256     =",
        new_sha,
    )

    print(
        "STATUS = "
        "PASS_PART2B_PART1_GRID_BINDING_PATCHED"
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
        "BLOCK 6.8 PART2B PART1 GRID PATCH = BLOCKED"
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

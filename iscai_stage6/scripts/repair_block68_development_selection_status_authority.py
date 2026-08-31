from __future__ import annotations

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
      "run_block68_development_selection_runtime_bind.py"
)

DIAGNOSTIC = (
    S6
    / "reports/"
      "block68_part3c_numeric_selection_status.json"
)

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

FINAL_SCRIPT = (
    S6
    / "scripts/"
      "run_block66_part3c_part2of2_final.py"
)

BACKUP = (
    S6
    / "artifacts/block68/"
      "selection_status_authority_patch/"
      "run_block68_development_selection_runtime_bind.pre_status_patch.py"
)

PATCH_REPORT = (
    S6
    / "reports/"
      "block68_development_selection_status_authority_patch.json"
)


EXPECTED_CLOSURE_SHA = (
    "8413fd646a1bf18e15c12bf0441d1ddb"
    "1ea72ed142703020ac0d2c77dba0bb29"
)

EXPECTED_FREEZE_SHA = (
    "880c2efb0895f463aaf3350eed773038"
    "9a5871651d7ad6091cef052d95434738"
)

EXPECTED_FINAL_SCRIPT_SHA = (
    "c8b87a64e5c5f6d1aba3d534fd009077"
    "2e24688f29c0425fb036457a5fa5de93"
)

EXPECTED_TESTS = 224


OLD_BLOCK = '''    require(
        closure.get(
            "numeric_policy_selected"
        )
        is False,
        (
            "Part3C closure unexpectedly "
            "contains a selected numeric policy."
        ),
    )

    require(
        freeze.get(
            "numeric_policy_selected"
        )
        is False,
        (
            "Part3C freeze unexpectedly "
            "contains a selected numeric policy."
        ),
    )

    require(
        freeze.get(
            "formal_outcomes_read"
        )
        is False,
        (
            "Part3C freeze says formal "
            "outcomes were read."
        ),
    )
'''


NEW_BLOCK = '''    # Exact Part3C selection-status authority.
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
'''


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


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_write(path: Path, payload: bytes):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)

    os.replace(
        temporary,
        path,
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
            return int(
                match.group(1)
            )

    return None


def regression():
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

    count = parse_test_count(
        output
    )

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


def prove_final_source_no_selection():
    source = FINAL_SCRIPT.read_text(
        encoding="utf-8"
    )

    required_fragments = (
        '"policy_sweep":',
        '"numeric_policy_selection":',
        '"numeric_policy_selected":',
        '"development_objective_evaluated":',
        '"formal_outcomes_read":',
    )

    for fragment in required_fragments:
        require(
            fragment in source,
            (
                "Expected Part3C final-source "
                f"selection marker missing: {fragment}"
            ),
        )

    # Exact semantic snippets that were independently
    # surfaced by the read-only diagnostic.
    require(
        re.search(
            r'"numeric_policy_selection"\s*:\s*False',
            source,
        )
        is not None,
        (
            "Part3C source does not explicitly "
            "record numeric_policy_selection=False."
        ),
    )

    require(
        re.search(
            r'"numeric_policy_selected"\s*:\s*False',
            source,
        )
        is not None,
        (
            "Part3C source does not explicitly "
            "record numeric_policy_selected=False."
        ),
    )

    require(
        re.search(
            r'"formal_outcomes_read"\s*:\s*False',
            source,
        )
        is not None,
        (
            "Part3C source does not explicitly "
            "record formal_outcomes_read=False."
        ),
    )

    return True


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "DEVELOPMENT-SELECTION STATUS AUTHORITY REPAIR"
    )
    print(
        "SCHEMA BINDING ONLY / NO SCIENTIFIC CHANGE"
    )
    print(
        "============================================================"
    )

    # ========================================================
    # A. Exact authoritative evidence
    # ========================================================

    print()
    print(
        "===== A. EXACT AUTHORITATIVE EVIDENCE ====="
    )

    require(
        TARGET.is_file(),
        (
            f"Missing target runtime-bind script: {TARGET}"
        ),
    )

    require(
        file_sha(
            CLOSURE
        )
        ==
        EXPECTED_CLOSURE_SHA,
        (
            "Part3C final closure SHA changed."
        ),
    )

    require(
        file_sha(
            FREEZE
        )
        ==
        EXPECTED_FREEZE_SHA,
        (
            "Part3C final freeze SHA changed."
        ),
    )

    require(
        file_sha(
            FINAL_SCRIPT
        )
        ==
        EXPECTED_FINAL_SCRIPT_SHA,
        (
            "Part3C final source SHA changed."
        ),
    )

    diagnostic = read_json(
        DIAGNOSTIC
    )

    require(
        diagnostic.get(
            "status"
        )
        ==
        "PASS_PART3C_NUMERIC_SELECTION_STATUS_DIAGNOSED",
        (
            "Selection-status diagnostic is not PASS."
        ),
    )

    require(
        diagnostic.get(
            "classification"
        )
        ==
        "CLOSURE_KEY_MISSING",
        (
            "Unexpected status-diagnostic "
            "classification."
        ),
    )

    closure = read_json(
        CLOSURE
    )

    freeze = read_json(
        FREEZE
    )

    # Exact closure evidence.
    require(
        "numeric_policy_selected"
        not in
        closure,
        (
            "Closure top-level schema changed; "
            "numeric_policy_selected now exists."
        ),
    )

    require(
        closure[
            "scientific_boundary"
        ][
            "numeric_policy_selected"
        ]
        is False,
        (
            "Closure scientific-boundary "
            "selection status changed."
        ),
    )

    require(
        closure[
            "scientific_boundary"
        ][
            "policy_parameter_sweep_executed"
        ]
        is False,
        (
            "Closure policy-sweep status changed."
        ),
    )

    require(
        closure[
            "scientific_boundary"
        ][
            "formal_outcomes_read"
        ]
        is False,
        (
            "Closure formal-outcome status changed."
        ),
    )

    require(
        closure[
            "integration_smoke"
        ][
            "numeric_policy_selected"
        ]
        is False,
        (
            "Integration-smoke selection "
            "status changed."
        ),
    )

    require(
        closure[
            "integration_smoke"
        ][
            "development_objective_evaluated"
        ]
        is False,
        (
            "Integration-smoke development "
            "objective status changed."
        ),
    )

    require(
        freeze[
            "numeric_policy_selected"
        ]
        is False,
        (
            "Final-freeze numeric selection "
            "status changed."
        ),
    )

    require(
        freeze[
            "formal_outcomes_read"
        ]
        is False,
        (
            "Final-freeze formal-outcome "
            "status changed."
        ),
    )

    prove_final_source_no_selection()

    print(
        "closure top-level key          = MISSING, EXPECTED"
    )

    print(
        "closure scientific boundary    = NUMERIC POLICY FALSE"
    )

    print(
        "integration smoke selection    = FALSE"
    )

    print(
        "development objective evaluated= FALSE"
    )

    print(
        "policy sweep executed          = FALSE"
    )

    print(
        "final freeze numeric selected  = FALSE"
    )

    print(
        "formal outcomes read           = FALSE"
    )

    print(
        "final source semantics         = EXACT PASS"
    )

    print(
        "scientific conclusion          = SELECTION PENDING"
    )

    # ========================================================
    # B. Exact target pattern
    # ========================================================

    print()
    print(
        "===== B. TARGET SCHEMA-BUG PATTERN ====="
    )

    original_bytes = TARGET.read_bytes()

    original_source = original_bytes.decode(
        "utf-8"
    )

    count = original_source.count(
        OLD_BLOCK
    )

    require(
        count == 1,
        (
            "Expected exactly one obsolete "
            "top-level closure gate; "
            f"found {count}."
        ),
    )

    print(
        "obsolete closure gate count = 1 PASS"
    )

    # ========================================================
    # C. Backup
    # ========================================================

    print()
    print(
        "===== C. VERIFIED BACKUP ====="
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
                "Existing backup differs from "
                "current pre-patch target."
            ),
        )

    else:

        BACKUP.write_bytes(
            original_bytes
        )

    print(
        "backup SHA256 =",
        file_sha(
            BACKUP
        ),
    )

    # ========================================================
    # D. Build exact patch
    # ========================================================

    print()
    print(
        "===== D. BUILD AUTHORITY-BINDING PATCH ====="
    )

    patched_source = original_source.replace(
        OLD_BLOCK,
        NEW_BLOCK,
        1,
    )

    compile(
        patched_source,
        str(
            TARGET
        ),
        "exec",
    )

    require(
        "scientific_boundary"
        in
        patched_source,
        (
            "Patched authority gate missing "
            "scientific_boundary binding."
        ),
    )

    require(
        "development_objective_evaluated"
        in
        patched_source,
        (
            "Patched authority gate missing "
            "development-objective binding."
        ),
    )

    print(
        "patch syntax            = PASS"
    )

    print(
        "scientific numerics     = UNCHANGED"
    )

    print(
        "runtime operators       = UNCHANGED"
    )

    print(
        "P_occ access            = UNCHANGED"
    )

    # ========================================================
    # E. Transactional patch + regression
    # ========================================================

    print()
    print(
        "===== E. TRANSACTIONAL PATCH ====="
    )

    applied = False

    try:

        atomic_write(
            TARGET,
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
                    TARGET
                ),
            ]
        )

        require(
            rc_compile == 0,
            (
                "Patched runtime-bind compile failed:\n"
                +
                compile_output
            ),
        )

        tests = regression()

        print(
            "patched runtime-bind compile = PASS"
        )

        print(
            "Stage6 regression            =",
            f"{tests} / {tests} PASS",
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
                TARGET.read_bytes()
                ==
                original_bytes,
                (
                    "Rollback failed to restore "
                    "exact runtime-bind runner."
                ),
            )

            print(
                "rollback = EXACT PASS"
            )

        raise

    post_sha = file_sha(
        TARGET
    )

    # ========================================================
    # F. Report
    # ========================================================

    result = {
        "stage":
            6,

        "block":
            "6.8_development_selection_status_authority_patch",

        "status":
            "PASS_SELECTION_STATUS_AUTHORITY_REBOUND",

        "diagnosis": {
            "scientific_failure":
                False,

            "schema_failure":
                True,

            "obsolete_assumption":
                (
                    "Part3C final closure must expose "
                    "top-level numeric_policy_selected"
                ),

            "actual_authority": [
                "closure.scientific_boundary.numeric_policy_selected",
                "closure.scientific_boundary.policy_parameter_sweep_executed",
                "closure.scientific_boundary.formal_outcomes_read",
                "closure.integration_smoke.numeric_policy_selected",
                "closure.integration_smoke.development_objective_evaluated",
                "freeze.numeric_policy_selected",
                "freeze.formal_outcomes_read",
            ],
        },

        "semantic_conclusion": {
            "numeric_policy_selected":
                False,

            "policy_parameter_sweep_executed":
                False,

            "development_objective_evaluated":
                False,

            "formal_outcomes_read":
                False,

            "development_numeric_selection_pending":
                True,
        },

        "runner": {
            "path":
                str(
                    TARGET
                ),

            "pre_patch_sha256":
                sha256(
                    original_bytes
                ).hexdigest(),

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

            "patch_scope":
                "selection-status schema authority gate only",
        },

        "scientific_boundary": {
            "P_occ_opened":
                False,

            "gamma_sweep_executed":
                False,

            "predictive_performance_computed":
                False,

            "acceptance_tested":
                False,

            "policy_numeric_change":
                False,

            "formal_evaluation":
                False,
        },

        "regression":
            tests,

        "next":
            (
                "rerun development-selection runtime bind "
                "against corrected selection-status authority"
            ),
    }

    atomic_write(
        PATCH_REPORT,
        canonical_bytes(
            result
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 SELECTION-STATUS AUTHORITY REPAIR — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "closure top-level key      = NOT REQUIRED"
    )

    print(
        "scientific-boundary status = EXACT FALSE"
    )

    print(
        "integration-smoke status   = EXACT FALSE"
    )

    print(
        "final-freeze status        = EXACT FALSE"
    )

    print(
        "development objective      = NOT EVALUATED"
    )

    print(
        "policy sweep               = NOT EXECUTED"
    )

    print(
        "numeric policy selected    = NO"
    )

    print(
        "selection pending          = YES"
    )

    print(
        "P_occ arrays               = NOT OPENED"
    )

    print(
        "predictive performance     = NOT COMPUTED"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "Stage6 regression          =",
        f"{tests} / {tests} PASS",
    )

    print(
        "patched runtime-bind SHA   =",
        post_sha,
    )

    print(
        "STATUS = "
        "PASS_SELECTION_STATUS_AUTHORITY_REBOUND"
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
        "BLOCK 6.8 SELECTION-STATUS AUTHORITY REPAIR = BLOCKED"
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
        "P_occ arrays             = NOT OPENED"
    )

    print(
        "gamma sweep              = NOT EXECUTED"
    )

    print(
        "predictive performance   = NOT COMPUTED"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "Do not alter scientific policy numerics."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

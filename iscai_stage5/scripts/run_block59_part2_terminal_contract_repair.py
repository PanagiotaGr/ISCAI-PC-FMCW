from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"

TEST59 = (
    S5
    / "tests/test_block59_reproducibility.py"
)

REPORT59 = (
    S5
    / "reports/block59_reproducibility.json"
)

PART1 = (
    S5
    / "reports/block59_part1_resume_audit.json"
)

BLOCK58_FORMAL = (
    S5
    / "reports/block58_formal_evaluation.json"
)

BLOCK58_RECONCILED = (
    S5
    / "reports/block58_part2_reconciled_gate.json"
)

BLOCK58_RECONCILIATION = (
    S5
    / "artifacts/block58/"
      "block58_part2_reconciliation.json"
)

BLOCK58_FINAL = (
    S5
    / "artifacts/block58/"
      "block58_final_closure.json"
)

FORMAL_MANIFEST = (
    ROOT
    / "iscai_stage3/artifacts/block38e/"
      "formal_validation_120.jsonl"
)

BACKUP = (
    S5
    / "artifacts/block59/"
      "test_block59_reproducibility."
      "pre_terminal_status_repair.py"
)

OUT = (
    S5
    / "reports/"
      "block59_part2_terminal_contract_repair.json"
)

EXPECTED_REPORT59_SHA = (
    "980bfbd2007ed798742bd7c36b5e8b6b"
    "cd0e808590c0dc5d9608db1ec6211845"
)

OLD_STATUS = (
    "PASS_REPRODUCIBILITY_PRE_FREEZE"
)

NEW_STATUS = (
    "PASS_FROZEN"
)


def sha256_file(path: Path) -> str:
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


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def concise_process_output(
    text: str,
    *,
    on_failure: bool,
):
    lines = text.splitlines()

    if on_failure:
        return lines[-45:]

    selected = []

    for line in lines:
        stripped = line.strip()

        if (
            stripped.startswith("Ran ")
            or stripped == "OK"
            or stripped.startswith("FAILED")
        ):
            selected.append(stripped)

    return selected or lines[-10:]


def run_tests(pattern: str):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S5 / "tests"),
            "-p",
            pattern,
        ],
        cwd=str(S5),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return (
        process.returncode,
        process.stdout,
    )


def restore_original(
    original_bytes: bytes,
):
    TEST59.write_bytes(
        original_bytes
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.9 PART 2/2"
    )
    print(
        "TERMINAL REPRODUCIBILITY CONTRACT REPAIR"
    )
    print(
        "============================================================"
    )

    required = (
        TEST59,
        REPORT59,
        PART1,
        BLOCK58_FORMAL,
        BLOCK58_RECONCILED,
        BLOCK58_RECONCILIATION,
        BLOCK58_FINAL,
        FORMAL_MANIFEST,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required evidence: "
            + ", ".join(missing)
        ),
    )

    # --------------------------------------------------------
    # Protect every scientific/frozen artifact.
    # --------------------------------------------------------

    protected = (
        REPORT59,
        BLOCK58_FORMAL,
        BLOCK58_RECONCILED,
        BLOCK58_RECONCILIATION,
        BLOCK58_FINAL,
        FORMAL_MANIFEST,
    )

    before_protected_sha = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    require(
        before_protected_sha[
            str(REPORT59)
        ]
        ==
        EXPECTED_REPORT59_SHA,
        (
            "Block5.9 frozen report SHA changed "
            "since Part1."
        ),
    )

    # --------------------------------------------------------
    # A. Confirm Part1 diagnosis.
    # --------------------------------------------------------

    print()
    print(
        "===== A. PART1 DIAGNOSIS CONTINUITY ====="
    )

    part1 = load_json(
        PART1
    )

    require(
        part1.get("status")
        ==
        "PASS_TARGETED_REPAIR_READY",
        (
            "Block5.9 Part1 is not "
            "PASS_TARGETED_REPAIR_READY."
        ),
    )

    diagnosis = part1.get(
        "diagnosis",
        {}
    )

    require(
        diagnosis.get(
            "scientific_reproducibility_failure"
        )
        is False,
        (
            "Part1 indicates a scientific "
            "reproducibility failure."
        ),
    )

    require(
        diagnosis.get(
            "stale_test_status_expectation"
        )
        is True,
        (
            "Part1 did not identify the stale "
            "status expectation."
        ),
    )

    require(
        diagnosis.get(
            "sole_current_Block59_test_defect"
        )
        is True,
        (
            "Part1 found more than one "
            "Block5.9 test defect."
        ),
    )

    print(
        "Part1 diagnosis          = PASS"
    )

    print(
        "scientific failure       = NO"
    )

    print(
        "sole defect              = STALE TEST STATUS"
    )

    # --------------------------------------------------------
    # B. Re-check terminal frozen report.
    # --------------------------------------------------------

    print()
    print(
        "===== B. TERMINAL BLOCK5.9 REPORT ====="
    )

    report59 = load_json(
        REPORT59
    )

    require(
        report59.get("status")
        ==
        NEW_STATUS,
        (
            "Canonical Block5.9 report is not "
            "PASS_FROZEN."
        ),
    )

    require(
        report59.get(
            "exact_match_to_frozen_block58"
        )
        is True,
        (
            "Exact match to frozen Block5.8 "
            "is not true."
        ),
    )

    require(
        report59.get(
            "fresh_process_repeat_exact"
        )
        is True,
        (
            "Fresh-process exact repeat "
            "is not true."
        ),
    )

    require(
        int(
            report59.get(
                "fresh_process_count"
            )
        )
        >=
        2,
        (
            "Expected at least two fresh "
            "reproducibility processes."
        ),
    )

    frozen58 = report59.get(
        "frozen_block58",
        {}
    )

    require(
        int(
            frozen58.get(
                "formal_cache_count"
            )
        )
        ==
        120,
        (
            "Frozen Block5.8 cache count "
            "is not 120."
        ),
    )

    require(
        report59.get(
            "beam_probability_vectors_exact"
        )
        is True,
        (
            "Beam probability vectors are "
            "not exact across repeats."
        ),
    )

    require(
        report59.get(
            "block58_mutated"
        )
        is False,
        (
            "Block5.8 mutation was recorded."
        ),
    )

    print(
        "status                   = PASS_FROZEN"
    )

    print(
        "exact Block5.8 match     = PASS"
    )

    print(
        "fresh-process repeats    =",
        report59.get(
            "fresh_process_count"
        ),
        "PASS",
    )

    print(
        "beam probability vectors = EXACT PASS"
    )

    print(
        "Block5.8 mutated         = NO"
    )

    # --------------------------------------------------------
    # C. Exact, narrow test-contract repair.
    # --------------------------------------------------------

    print()
    print(
        "===== C. TARGETED TEST CONTRACT REPAIR ====="
    )

    original_bytes = TEST59.read_bytes()

    original_text = original_bytes.decode(
        "utf-8"
    )

    original_test_sha = sha256(
        original_bytes
    ).hexdigest()

    old_count = original_text.count(
        OLD_STATUS
    )

    new_count_before = original_text.count(
        NEW_STATUS
    )

    print(
        "old PRE_FREEZE occurrences =",
        old_count,
    )

    print(
        "PASS_FROZEN occurrences before =",
        new_count_before,
    )

    require(
        old_count == 1,
        (
            "Expected exactly one stale "
            "PRE_FREEZE status occurrence; "
            f"found {old_count}."
        ),
    )

    # Preserve exact pre-repair test.
    if BACKUP.exists():
        require(
            BACKUP.read_bytes()
            ==
            original_bytes,
            (
                "Existing Block5.9 test backup "
                "does not match current pre-repair "
                "test. Refusing overwrite."
            ),
        )

    else:
        BACKUP.write_bytes(
            original_bytes
        )

    repaired_text = original_text.replace(
        OLD_STATUS,
        NEW_STATUS,
        1,
    )

    require(
        repaired_text.count(
            OLD_STATUS
        )
        ==
        0,
        (
            "Stale PRE_FREEZE expectation "
            "remains after replacement."
        ),
    )

    require(
        repaired_text.count(
            NEW_STATUS
        )
        ==
        new_count_before + 1,
        (
            "PASS_FROZEN replacement count "
            "is inconsistent."
        ),
    )

    TEST59.write_text(
        repaired_text,
        encoding="utf-8",
    )

    repaired_test_sha = sha256_file(
        TEST59
    )

    print(
        "test backup              = PASS"
    )

    print(
        "old test SHA256          =",
        original_test_sha,
    )

    print(
        "new test SHA256          =",
        repaired_test_sha,
    )

    print(
        "semantic change          = "
        "PRE_FREEZE → PASS_FROZEN ONLY"
    )

    # --------------------------------------------------------
    # D. Controlled compile.
    # --------------------------------------------------------

    print()
    print(
        "===== D. CONTROLLED COMPILE ====="
    )

    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(TEST59),
        ],
        cwd=str(S5),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if compile_process.returncode != 0:
        restore_original(
            original_bytes
        )

        raise RuntimeError(
            "Repaired Block5.9 test failed "
            "compile; original automatically "
            "restored. "
            + compile_process.stdout
        )

    print(
        "test_block59_reproducibility.py = PASS"
    )

    # --------------------------------------------------------
    # E. Run Block5.9 test in fresh Python process.
    # --------------------------------------------------------

    print()
    print(
        "===== E. BLOCK5.9 TARGETED REGRESSION ====="
    )

    rc59, out59 = run_tests(
        "test_block59_reproducibility.py"
    )

    for line in concise_process_output(
        out59,
        on_failure=(
            rc59 != 0
        ),
    ):
        print(line)

    if rc59 != 0:
        restore_original(
            original_bytes
        )

        raise RuntimeError(
            "Block5.9 targeted regression "
            "failed; original test automatically "
            "restored."
        )

    print(
        "Block5.9 regression      = 3 / 3 PASS"
    )

    # --------------------------------------------------------
    # F. Full Stage5 regression.
    # --------------------------------------------------------

    print()
    print(
        "===== F. FULL STAGE5 REGRESSION ====="
    )

    rc_all, out_all = run_tests(
        "test_*.py"
    )

    for line in concise_process_output(
        out_all,
        on_failure=(
            rc_all != 0
        ),
    ):
        print(line)

    if rc_all != 0:
        restore_original(
            original_bytes
        )

        # Prove rollback compiles before reporting.
        rollback_compile = subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                str(TEST59),
            ],
            cwd=str(S5),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        raise RuntimeError(
            "Full Stage5 regression failed. "
            "The test repair was rolled back. "
            "Rollback compile="
            f"{rollback_compile.returncode}."
        )

    print(
        "full Stage5 regression   = PASS"
    )

    # --------------------------------------------------------
    # G. Verify exact semantic edit only.
    # --------------------------------------------------------

    print()
    print(
        "===== G. TEST DIFF CONTRACT ====="
    )

    final_text = TEST59.read_text(
        encoding="utf-8"
    )

    reverse_text = final_text.replace(
        NEW_STATUS,
        OLD_STATUS,
        1,
    )

    require(
        reverse_text
        ==
        original_text,
        (
            "Test file contains modifications "
            "beyond the single approved status "
            "transition."
        ),
    )

    print(
        "approved changed tokens  = 1"
    )

    print(
        "only change              = "
        "expected terminal status"
    )

    print(
        "test semantics weakened  = NO"
    )

    # --------------------------------------------------------
    # H. Scientific/frozen artifact immutability.
    # --------------------------------------------------------

    print()
    print(
        "===== H. SCIENTIFIC ARTIFACT IMMUTABILITY ====="
    )

    after_protected_sha = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    changed_protected = [
        path
        for path in before_protected_sha
        if (
            before_protected_sha[path]
            !=
            after_protected_sha[path]
        )
    ]

    if changed_protected:
        restore_original(
            original_bytes
        )

        raise RuntimeError(
            "Protected scientific/frozen "
            "artifact changed during Part2: "
            + ", ".join(
                changed_protected
            )
        )

    require(
        sha256_file(
            REPORT59
        )
        ==
        EXPECTED_REPORT59_SHA,
        (
            "Block5.9 frozen reproducibility "
            "report changed."
        ),
    )

    print(
        "Block5.8 formal evidence = UNCHANGED"
    )

    print(
        "Block5.8 closure         = UNCHANGED"
    )

    print(
        "formal manifest          = UNCHANGED"
    )

    print(
        "Block5.9 frozen report   = UNCHANGED"
    )

    print(
        "scientific outputs       = UNCHANGED"
    )

    # --------------------------------------------------------
    # I. Final Block5.9 repair record.
    # --------------------------------------------------------

    payload = {
        "stage":
            5,

        "block":
            "5.9",

        "part":
            "2/2",

        "status":
            "PASS_COMPLETE",

        "canonical_reproducibility_status":
            "PASS_FROZEN",

        "canonical_reproducibility_report": {
            "path":
                str(REPORT59),

            "sha256":
                EXPECTED_REPORT59_SHA,

            "exact_match_to_frozen_block58":
                True,

            "fresh_process_repeat_exact":
                True,

            "fresh_process_count":
                int(
                    report59[
                        "fresh_process_count"
                    ]
                ),

            "beam_probability_vectors_exact":
                True,

            "formal_cache_count":
                120,
        },

        "repair": {
            "type":
                (
                    "stale_test_contract_"
                    "status_transition"
                ),

            "scientific_change":
                False,

            "test_before_sha256":
                original_test_sha,

            "test_after_sha256":
                repaired_test_sha,

            "backup":
                str(BACKUP),

            "old_expected_status":
                OLD_STATUS,

            "new_expected_status":
                NEW_STATUS,

            "replacement_count":
                1,

            "other_test_changes":
                False,

            "test_semantics_weakened":
                False,
        },

        "regression": {
            "Block5_9":
                "3/3_PASS",

            "full_Stage5":
                "PASS",
        },

        "immutability": {
            "Block5_8_formal":
                True,

            "Block5_8_closure":
                True,

            "formal_manifest":
                True,

            "Block5_9_frozen_report":
                True,

            "scientific_outputs":
                True,
        },

        "scientific_execution": {
            "training":
                False,

            "formal_inference":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "reproducibility_rerun":
                False,
        },

        "next":
            (
                "Block5.10 final Stage5 "
                "acceptance/closure and "
                "Stage5→Stage6 handoff"
            ),
    }

    atomic_json(
        OUT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.9 PART 2/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "canonical reproducibility = PASS_FROZEN"
    )

    print(
        "exact Block5.8 match      = PASS"
    )

    print(
        "fresh-process repeat      = EXACT PASS"
    )

    print(
        "Block5.9 targeted tests   = 3 / 3 PASS"
    )

    print(
        "full Stage5 regression    = PASS"
    )

    print(
        "stale test expectation    = REPAIRED"
    )

    print(
        "scientific outputs changed= NO"
    )

    print(
        "formal inference rerun    = NO"
    )

    print(
        "parameter tuning          = NO"
    )

    print(
        "Block5.9 report changed   = NO"
    )

    print(
        "STATUS = PASS_COMPLETE"
    )

    print(
        "report =",
        OUT,
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
        "BLOCK 5.9 PART 2/2 = BLOCKED"
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

    print(
        "scientific outputs modified = NO"
    )

    print(
        "training/formal inference    = NO"
    )

    print(
        "parameter tuning             = NO"
    )

    print(
        "terminal remains open        = YES"
    )

# Deliberately no sys.exit().

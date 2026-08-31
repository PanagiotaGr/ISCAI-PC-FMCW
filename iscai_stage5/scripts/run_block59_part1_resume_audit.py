from __future__ import annotations

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import py_compile
import traceback
import unittest


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"

REPORT59 = (
    S5
    / "reports/block59_reproducibility.json"
)

TEST59 = (
    S5
    / "tests/test_block59_reproducibility.py"
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

OUT = (
    S5
    / "reports/"
      "block59_part1_resume_audit.json"
)


def sha256_file(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
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
        tmp,
        path,
    )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def flatten(value, prefix=""):
    rows = []

    if isinstance(value, dict):
        for key, item in value.items():
            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten(
                    item,
                    child,
                )
            )

    elif isinstance(value, list):
        for i, item in enumerate(value):
            rows.extend(
                flatten(
                    item,
                    f"{prefix}[{i}]",
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


def concise_lines(text):
    result = []

    for line in text.splitlines():
        stripped = line.strip()

        if (
            stripped.startswith("Ran ")
            or
            stripped == "OK"
            or
            stripped.startswith("FAILED")
            or
            "AssertionError:" in stripped
        ):
            result.append(stripped)

    return result[-15:]


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.9 PART 1/2"
    )
    print(
        "EXISTING REPRODUCIBILITY FREEZE AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        REPORT59,
        TEST59,
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
            "Missing required reproducibility "
            "evidence: "
            + ", ".join(missing)
        ),
    )

    protected = (
        REPORT59,
        BLOCK58_FORMAL,
        BLOCK58_RECONCILED,
        BLOCK58_RECONCILIATION,
        BLOCK58_FINAL,
        FORMAL_MANIFEST,
    )

    before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Existing Block5.9 report
    # ========================================================

    print()
    print(
        "===== A. EXISTING BLOCK5.9 REPORT ====="
    )

    r59 = load_json(
        REPORT59
    )

    print(
        "status =",
        r59.get("status"),
    )

    print(
        "stage  =",
        r59.get("stage"),
    )

    print(
        "block  =",
        r59.get("block"),
    )

    require(
        r59.get("stage") == 5,
        "Block5.9 report stage != 5.",
    )

    require(
        str(
            r59.get("block")
        )
        ==
        "5.9",
        "Block5.9 report block != 5.9.",
    )

    require(
        r59.get("status")
        ==
        "PASS_FROZEN",
        (
            "Existing Block5.9 report is not "
            "the expected terminal PASS_FROZEN."
        ),
    )

    print(
        "terminal reproducibility status = PASS_FROZEN"
    )

    print(
        "report SHA256 =",
        sha256_file(
            REPORT59
        ),
    )

    # ========================================================
    # B. Core reproducibility invariants
    # ========================================================

    print()
    print(
        "===== B. REPRODUCIBILITY INVARIANTS ====="
    )

    require(
        r59.get(
            "exact_match_to_frozen_block58"
        )
        is True,
        (
            "Block5.9 does not prove exact "
            "match to frozen Block5.8."
        ),
    )

    require(
        r59.get(
            "fresh_process_repeat_exact"
        )
        is True,
        (
            "Block5.9 does not prove exact "
            "fresh-process repeat."
        ),
    )

    print(
        "exact match to Block5.8 = PASS"
    )

    print(
        "fresh-process repeat    = EXACT PASS"
    )

    interesting = []

    tokens = (
        "sha",
        "hash",
        "repeat",
        "fresh",
        "exact",
        "scenario",
        "formal",
        "environment",
        "block58",
        "frozen",
    )

    for key, value in flatten(
        r59
    ):
        lower = key.lower()

        if any(
            token in lower
            for token in tokens
        ):
            interesting.append(
                (
                    key,
                    value,
                )
            )

    print(
        "reproducibility evidence fields =",
        len(
            interesting
        ),
    )

    # Only concise selected evidence.
    for key, value in interesting[:30]:
        text = str(value)

        if len(text) > 160:
            text = (
                text[:157]
                + "..."
            )

        print(
            f"  {key} = {text}"
        )

    # ========================================================
    # C. Frozen Block5.8 continuity
    # ========================================================

    print()
    print(
        "===== C. BLOCK5.8 FROZEN CONTINUITY ====="
    )

    formal = load_json(
        BLOCK58_FORMAL
    )

    reconciled = load_json(
        BLOCK58_RECONCILED
    )

    reconciliation = load_json(
        BLOCK58_RECONCILIATION
    )

    require(
        formal.get("status")
        ==
        "PASS",
        "Block5.8 formal report changed.",
    )

    require(
        reconciled.get("status")
        ==
        "PASS_RECONCILED",
        (
            "Block5.8 reconciled gate "
            "is not PASS_RECONCILED."
        ),
    )

    require(
        reconciliation.get("status")
        ==
        "PASS_RECONCILED_COMPLETE",
        (
            "Block5.8 reconciliation "
            "is not complete."
        ),
    )

    require(
        reconciliation.get(
            "formal_rerun"
        )
        is False,
        (
            "Block5.8 reconciliation unexpectedly "
            "performed formal rerun."
        ),
    )

    require(
        reconciliation.get(
            "parameter_tuning"
        )
        is False,
        (
            "Block5.8 reconciliation unexpectedly "
            "performed parameter tuning."
        ),
    )

    print(
        "Block5.8 formal          = PASS"
    )

    print(
        "Block5.8 reconciliation  = PASS"
    )

    print(
        "formal rerun in repair   = NO"
    )

    print(
        "parameter tuning         = NO"
    )

    # ========================================================
    # D. Inspect stale test expectation
    # ========================================================

    print()
    print(
        "===== D. BLOCK5.9 TEST EXPECTATION ====="
    )

    source = TEST59.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    expects_pre_freeze = (
        '"PASS_REPRODUCIBILITY_PRE_FREEZE"'
        in source
        or
        "'PASS_REPRODUCIBILITY_PRE_FREEZE'"
        in source
    )

    expects_frozen = (
        '"PASS_FROZEN"'
        in source
        or
        "'PASS_FROZEN'"
        in source
    )

    print(
        "test expects PRE_FREEZE =",
        expects_pre_freeze,
    )

    print(
        "test knows PASS_FROZEN   =",
        expects_frozen,
    )

    require(
        expects_pre_freeze,
        (
            "Known stale PRE_FREEZE expectation "
            "is no longer present; inspect before "
            "performing Part2 repair."
        ),
    )

    # ========================================================
    # E. Run only Block5.9 test
    # ========================================================

    print()
    print(
        "===== E. CURRENT BLOCK5.9 TEST ====="
    )

    py_compile.compile(
        str(TEST59),
        doraise=True,
    )

    loader = unittest.TestLoader()

    suite = loader.discover(
        str(S5 / "tests"),
        pattern="test_block59_reproducibility.py",
    )

    buffer = io.StringIO()

    result = unittest.TextTestRunner(
        stream=buffer,
        verbosity=1,
    ).run(
        suite
    )

    output = buffer.getvalue()

    for line in concise_lines(
        output
    ):
        print(line)

    failure_text = "\n".join(
        text
        for _, text in (
            result.failures
            +
            result.errors
        )
    )

    status_mismatch_only = (
        len(result.failures) == 1
        and
        len(result.errors) == 0
        and
        "PASS_FROZEN"
        in failure_text
        and
        "PASS_REPRODUCIBILITY_PRE_FREEZE"
        in failure_text
    )

    print(
        "tests run               =",
        result.testsRun,
    )

    print(
        "failures                =",
        len(result.failures),
    )

    print(
        "errors                  =",
        len(result.errors),
    )

    print(
        "sole defect = stale status expectation =",
        status_mismatch_only,
    )

    require(
        status_mismatch_only,
        (
            "Block5.9 has a defect beyond the "
            "known stale status expectation."
        ),
    )

    # ========================================================
    # F. Immutability readback
    # ========================================================

    print()
    print(
        "===== F. READ-ONLY IMMUTABILITY ====="
    )

    after = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    changed = [
        path
        for path in before
        if before[path] != after[path]
    ]

    require(
        not changed,
        (
            "Protected evidence changed during "
            "read-only audit: "
            + ", ".join(changed)
        ),
    )

    print(
        "Block5.8 evidence        = UNCHANGED"
    )

    print(
        "Block5.9 frozen report   = UNCHANGED"
    )

    print(
        "formal manifest          = UNCHANGED"
    )

    # ========================================================
    # G. Part1 result
    # ========================================================

    payload = {
        "stage":
            5,

        "block":
            "5.9_part_1",

        "status":
            "PASS_TARGETED_REPAIR_READY",

        "existing_reproducibility_report": {
            "path":
                str(REPORT59),

            "sha256":
                sha256_file(REPORT59),

            "status":
                r59.get("status"),

            "exact_match_to_frozen_block58":
                True,

            "fresh_process_repeat_exact":
                True,
        },

        "diagnosis": {
            "scientific_reproducibility_failure":
                False,

            "frozen_report_failure":
                False,

            "stale_test_status_expectation":
                True,

            "current_report_status":
                "PASS_FROZEN",

            "stale_expected_status":
                (
                    "PASS_REPRODUCIBILITY_"
                    "PRE_FREEZE"
                ),

            "sole_current_Block59_test_defect":
                True,
        },

        "scientific_execution": {
            "formal_rerun":
                False,

            "training":
                False,

            "inference":
                False,

            "parameter_tuning":
                False,
        },

        "protected_artifacts_modified":
            False,

        "next":
            (
                "Block5.9 Part2: update stale "
                "test contract to terminal "
                "PASS_FROZEN and rerun full "
                "Stage5 regression."
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
        "BLOCK 5.9 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "existing Block5.9 status = PASS_FROZEN"
    )

    print(
        "exact Block5.8 match     = PASS"
    )

    print(
        "fresh-process repeat     = EXACT PASS"
    )

    print(
        "scientific failure       = NO"
    )

    print(
        "sole regression defect   = STALE STATUS EXPECTATION"
    )

    print(
        "scientific artifacts changed = NO"
    )

    print(
        "safe targeted Part2 repair = YES"
    )

    print(
        "STATUS = PASS_TARGETED_REPAIR_READY"
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
        "BLOCK 5.9 PART 1/2 = BLOCKED"
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
        "formal rerun             = NO"
    )

    print(
        "training/inference       = NO"
    )

    print(
        "scientific artifacts changed = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

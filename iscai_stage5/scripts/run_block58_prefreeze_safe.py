from __future__ import annotations

import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    /
    "iscai_stage5"
)

EXPECTED_PREVIOUS_TESTS = 226
EXPECTED_NEW_TESTS = 12
EXPECTED_TOTAL_TESTS = (
    EXPECTED_PREVIOUS_TESTS
    +
    EXPECTED_NEW_TESTS
)

FILES = (
    STAGE5
    /
    "src/iscai_stage5/"
    "formal_acceptance.py",

    STAGE5
    /
    "tests/"
    "test_block58_formal_acceptance.py",

    STAGE5
    /
    "scripts/"
    "run_block58_acceptance_freeze.py",

    STAGE5
    /
    "scripts/"
    "run_block58_routeb_schema_probe.py",

    STAGE5
    /
    "scripts/"
    "run_block58_prefreeze_safe.py",
)

TEST_FILE = (
    STAGE5
    /
    "tests/"
    "test_block58_formal_acceptance.py"
)

ACCEPTANCE = (
    STAGE5
    /
    "scripts/"
    "run_block58_acceptance_freeze.py"
)

SCHEMA_PROBE = (
    STAGE5
    /
    "scripts/"
    "run_block58_routeb_schema_probe.py"
)

SOURCE_ROOTS = (
    ROOT / "iscai_stage5/src",
    ROOT / "iscai_stage4/src",
    ROOT / "iscai_stage3/src",
    ROOT / "iscai_stage2/src",
    ROOT / "iscai_stage1/src",
    ROOT / "iscai_stage0/src",
)


def environment():
    result = dict(
        os.environ
    )

    existing = result.get(
        "PYTHONPATH",
        "",
    )

    pieces = [
        str(
            path
        )
        for path in SOURCE_ROOTS
    ]

    if existing:
        pieces.append(
            existing
        )

    result[
        "PYTHONPATH"
    ] = os.pathsep.join(
        pieces
    )

    return result


def run_process(
    label,
    command,
):
    result = subprocess.run(
        command,
        cwd=STAGE5,
        env=environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    combined = (
        result.stdout
        +
        result.stderr
    )

    if result.returncode != 0:
        print()
        print(
            f"===== {label} — BLOCKED ====="
        )
        print(
            f"raw code = {result.returncode}"
        )
        print(
            "\n".join(
                combined
                .splitlines()[
                    -120:
                ]
            )
        )

        return (
            False,
            combined,
        )

    return (
        True,
        combined,
    )


def test_count(
    output,
):
    matches = re.findall(
        r"Ran\s+(\d+)\s+tests?",
        output,
    )

    if not matches:
        return None

    return int(
        matches[
            -1
        ]
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 2/2 — SAFE PRE-FORMAL CONTROLLER"
    )
    print(
        "============================================================"
    )

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    try:
        for path in FILES:
            py_compile.compile(
                str(
                    path
                ),
                doraise=True,
            )

            print(
                path.name,
                "= PASS",
            )
    except Exception as error:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = CONTROLLED COMPILE"
        )
        print(
            "reason =",
            repr(
                error
            ),
        )
        print(
            "terminal remains open = YES"
        )
        return

    ok, output = (
        run_process(
            "BLOCK5.8 ACCEPTANCE UNIT TESTS",
            [
                sys.executable,
                str(
                    TEST_FILE
                ),
            ],
        )
    )

    if not ok:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "recovery = repair narrow tests only; "
            "formal data untouched"
        )
        print(
            "terminal remains open = YES"
        )
        return

    narrow_count = (
        test_count(
            output
        )
    )

    if (
        narrow_count
        !=
        EXPECTED_NEW_TESTS
    ):
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = NARROW TEST COUNT"
        )
        print(
            "expected =",
            EXPECTED_NEW_TESTS,
        )
        print(
            "actual   =",
            narrow_count,
        )
        print(
            "terminal remains open = YES"
        )
        return

    print()
    print(
        "narrow acceptance tests =",
        narrow_count,
        "/",
        EXPECTED_NEW_TESTS,
        "PASS",
    )

    ok, output = (
        run_process(
            "PRE-FREEZE FULL STAGE5 REGRESSION",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(
                    STAGE5
                    /
                    "tests"
                ),
                "-p",
                "test_*.py",
            ],
        )
    )

    if not ok:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = PRE-FREEZE REGRESSION"
        )
        print(
            "formal metrics = NOT COMPUTED"
        )
        print(
            "terminal remains open = YES"
        )
        return

    count = (
        test_count(
            output
        )
    )

    if (
        count
        !=
        EXPECTED_TOTAL_TESTS
    ):
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = REGRESSION CONTINUITY"
        )
        print(
            "expected =",
            EXPECTED_TOTAL_TESTS,
        )
        print(
            "actual   =",
            count,
        )
        print(
            "Do not loosen the count automatically."
        )
        print(
            "terminal remains open = YES"
        )
        return

    print(
        "pre-freeze Stage5 regression =",
        count,
        "/",
        EXPECTED_TOTAL_TESTS,
        "PASS",
    )

    ok, output = (
        run_process(
            "PRE-FORMAL ACCEPTANCE FREEZE",
            [
                sys.executable,
                str(
                    ACCEPTANCE
                ),
            ],
        )
    )

    if not ok:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = ACCEPTANCE FREEZE"
        )
        print(
            "formal metrics = NOT COMPUTED"
        )
        print(
            "terminal remains open = YES"
        )
        return

    print()
    print(
        output.strip()
    )

    ok, output = (
        run_process(
            "ROUTE-B ONE-SCENE SCHEMA PROBE",
            [
                sys.executable,
                str(
                    SCHEMA_PROBE
                ),
            ],
        )
    )

    if not ok:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = ROUTE-B SCHEMA PROBE"
        )
        print(
            "model forward = NO"
        )
        print(
            "formal metrics = NOT COMPUTED"
        )
        print(
            "recovery = inspect schema report/error; "
            "do not run formal materialization yet"
        )
        print(
            "terminal remains open = YES"
        )
        return

    print()
    print(
        output.strip()
    )

    ok, output = (
        run_process(
            "FINAL FULL STAGE5 REGRESSION",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(
                    STAGE5
                    /
                    "tests"
                ),
                "-p",
                "test_*.py",
            ],
        )
    )

    if not ok:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = FINAL REGRESSION"
        )
        print(
            "formal metrics = NOT COMPUTED"
        )
        print(
            "terminal remains open = YES"
        )
        return

    final_count = (
        test_count(
            output
        )
    )

    if (
        final_count
        !=
        EXPECTED_TOTAL_TESTS
    ):
        print(
            "STATUS = BLOCKED"
        )
        print(
            "phase  = FINAL TEST COUNT"
        )
        print(
            "expected =",
            EXPECTED_TOTAL_TESTS,
        )
        print(
            "actual   =",
            final_count,
        )
        print(
            "terminal remains open = YES"
        )
        return

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 2/2 — PRE-FORMAL LOCK FINAL"
    )
    print(
        "============================================================"
    )
    print(
        "coverage acceptance      = FROZEN BEFORE FORMAL RESULTS"
    )
    print(
        "nominal q                = 0.95"
    )
    print(
        "alpha                    = 0.05"
    )
    print(
        "fixed %-point tolerance  = NONE"
    )
    print(
        "Route B identity bridge  = PROBED"
    )
    print(
        "Stage4 model forward     = NO"
    )
    print(
        "Stage5 formal metrics    = NOT COMPUTED"
    )
    print(
        "post-hoc tuning          = NO"
    )
    print(
        "Stage5 regression        =",
        final_count,
        "/",
        EXPECTED_TOTAL_TESTS,
        "PASS",
    )
    print(
        "STATUS                   = PASS_PREFORMAL_LOCK"
    )
    print(
        "next                     = ROUTE-B PRE-TTP "
        "FORMAL MATERIALIZATION"
    )
    print(
        "terminal remains open    = YES"
    )


if __name__ == "__main__":
    main()

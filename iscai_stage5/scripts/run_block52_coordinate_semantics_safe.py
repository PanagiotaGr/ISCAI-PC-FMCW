from __future__ import annotations

from pathlib import Path
import py_compile
import subprocess
import sys
import traceback


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

SCRIPT = (
    STAGE5
    / "scripts/"
      "audit_block52_stage4_coordinate_semantics.py"
)


def main():
    print(
        "============================================================"
    )

    print(
        "BLOCK 5.2 — COORDINATE SEMANTICS SAFE CONTROLLER"
    )

    print(
        "============================================================"
    )

    try:
        py_compile.compile(
            str(
                SCRIPT
            ),
            doraise=True,
        )

    except Exception as exc:

        print(
            "compile = BLOCKED"
        )

        print(
            "reason =",
            f"{type(exc).__name__}: {exc}",
        )

        print(
            "model forward executed = NO"
        )

        print(
            "formal N=120 used = NO"
        )

        print(
            "terminal remains open = YES"
        )

        return

    print(
        "compile = PASS"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                SCRIPT
            ),
        ],
        cwd=str(
            STAGE5
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    if process.stdout is not None:

        for line in process.stdout:
            print(
                line,
                end="",
                flush=True,
            )

    raw_code = process.wait()

    print()
    print(
        "semantic-audit raw code =",
        raw_code,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print(
        "safe controller = BLOCKED"
    )

    print(
        "reason =",
        f"{type(exc).__name__}: {exc}",
    )

    traceback.print_exc()

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

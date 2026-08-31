from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import py_compile
import shutil
import traceback


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

REFERENCE = (
    STAGE5
    / "scripts/"
      "run_block53_part2_reference_decision.py"
)

SAFE = (
    STAGE5
    / "scripts/"
      "run_block53_part2_safe.py"
)

REPORT = (
    STAGE5
    / "artifacts/block53/"
      "reference_gate_repair.json"
)


REFERENCE_MARKER = (
    "BLOCK53_REFERENCE_ASSUMPTION_LINK_V2"
)

SAFE_MARKER = (
    "BLOCK53_REFERENCE_STRUCTURED_GATE_V2"
)


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


def file_sha256(
    path: Path,
):
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


def write_json(
    path: Path,
    payload,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def backup(
    path: Path,
    sha: str,
):
    destination = (
        STAGE5
        / "artifacts/block53/"
          f"{path.stem}."
          f"pre_reference_gate_repair."
          f"{sha[:16]}"
          f"{path.suffix}"
    )

    if not destination.is_file():

        shutil.copy2(
            path,
            destination,
        )

    return destination


def patch_reference():
    source = REFERENCE.read_text(
        encoding="utf-8"
    )

    old_sha = file_sha256(
        REFERENCE
    )

    if REFERENCE_MARKER in source:

        return {
            "status":
                "ALREADY_PATCHED",

            "old_sha256":
                old_sha,

            "new_sha256":
                old_sha,

            "backup":
                None,
        }

    old_block = '''        if (
            "Headlamp half-FOV assumption"
            in
            text
            and
            "12"
            in
            text
        ):
            fov_assumption_hits.append(
                hit
            )
'''

    new_block = '''        # BLOCK53_REFERENCE_ASSUMPTION_LINK_V2
        #
        # The Part-A source stores the numeric value in:
        #
        #   assumed_headlamp_fov_deg = 12.0
        #
        # while the separate annotation line is an f-string:
        #
        #   Headlamp half-FOV assumption :
        #       ±{assumed_headlamp_fov_deg:.1f} deg
        #
        # Therefore the annotation source does not itself
        # contain the literal text "12". Link the semantic
        # label to the exact variable name instead.
        #
        if (
            "Headlamp half-FOV assumption"
            in
            text
            and
            "assumed_headlamp_fov_deg"
            in
            text
        ):
            fov_assumption_hits.append(
                hit
            )
'''

    require(
        old_block in source,
        (
            "Could not locate the known "
            "Block5.3 FOV-assumption parser "
            "condition."
        ),
    )

    source = source.replace(
        old_block,
        new_block,
        1,
    )

    old_require = '''    require(
        fov_assumption_hits,
        (
            "Could not verify Part-A labels "
            "the +/-12 deg headlamp FOV as "
            "an assumption."
        ),
    )
'''

    new_require = '''    require(
        fov_assumption_hits,
        (
            "Could not verify Part-A labels "
            "the +/-12 deg headlamp FOV as "
            "an assumption."
        ),
    )

    #
    # The numeric assignment and the semantic assumption
    # annotation must originate from the same frozen Part-A
    # source file. This links:
    #
    #   assumed_headlamp_fov_deg = 12.0
    #
    # to:
    #
    #   Headlamp half-FOV assumption :
    #       ±{assumed_headlamp_fov_deg:.1f} deg
    #
    assignment_paths = {
        str(
            hit.get(
                "path",
                ""
            )
        )
        for hit in fov_assignment_hits
    }

    assumption_paths = {
        str(
            hit.get(
                "path",
                ""
            )
        )
        for hit in fov_assumption_hits
    }

    shared_paths = (
        assignment_paths
        &
        assumption_paths
    )

    require(
        shared_paths,
        (
            "Numeric +/-12 deg assignment "
            "and FOV-assumption annotation "
            "do not share a frozen Part-A "
            "source file."
        ),
    )
'''

    require(
        old_require in source,
        (
            "Could not locate the known "
            "FOV-assumption evidence gate."
        ),
    )

    source = source.replace(
        old_require,
        new_require,
        1,
    )

    backup_path = backup(
        REFERENCE,
        old_sha,
    )

    temporary = REFERENCE.with_suffix(
        ".py.tmp"
    )

    temporary.write_text(
        source,
        encoding="utf-8",
    )

    temporary.replace(
        REFERENCE
    )

    new_sha = file_sha256(
        REFERENCE
    )

    return {
        "status":
            "PATCHED",

        "old_sha256":
            old_sha,

        "new_sha256":
            new_sha,

        "backup":
            str(
                backup_path
            ),
    }


def patch_safe_controller():
    source = SAFE.read_text(
        encoding="utf-8"
    )

    old_sha = file_sha256(
        SAFE
    )

    if SAFE_MARKER in source:

        return {
            "status":
                "ALREADY_PATCHED",

            "old_sha256":
                old_sha,

            "new_sha256":
                old_sha,

            "backup":
                None,
        }

    old_constant = '''REFERENCE = (
    STAGE5
    / "scripts/"
      "run_block53_part2_reference_decision.py"
)

FREEZE = (
'''

    new_constant = '''REFERENCE = (
    STAGE5
    / "scripts/"
      "run_block53_part2_reference_decision.py"
)

# BLOCK53_REFERENCE_STRUCTURED_GATE_V2
REFERENCE_REPORT = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_decision.json"
)

FREEZE = (
'''

    require(
        old_constant in source,
        (
            "Could not locate REFERENCE "
            "constant in Block5.3 Part2 "
            "safe controller."
        ),
    )

    source = source.replace(
        old_constant,
        new_constant,
        1,
    )

    old_execution = '''    print()
    print(
        "reference-decision raw code =",
        raw_code,
    )

    print()
    print(
        "===== FINAL BLOCK5.3 FREEZE ====="
    )
'''

    new_execution = '''    print()
    print(
        "reference-decision raw code =",
        raw_code,
    )

    #
    # The child script intentionally never terminates the
    # shell with a non-zero exit status. Therefore raw_code
    # alone is NOT a scientific success signal.
    #
    # Require the structured PASS artifact before allowing
    # the final freeze to run.
    #
    if not REFERENCE_REPORT.is_file():

        blocked(
            "reference_structured_report",
            (
                "Part-A reference decision "
                "report was not written."
            ),
        )

        return

    try:
        reference_report = json.loads(
            REFERENCE_REPORT.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        blocked(
            "reference_report_readback",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

        return

    if (
        reference_report.get(
            "status"
        )
        !=
        "PASS"
    ):

        blocked(
            "reference_scientific_gate",
            (
                "Part-A reference decision "
                "structured status is not PASS."
            ),
        )

        return

    scientific = (
        reference_report.get(
            "scientific_execution",
            {}
        )
    )

    if scientific.get(
        "formal_N120_read"
    ) is not False:

        blocked(
            "reference_formal_leakage",
            (
                "Reference decision does not "
                "prove formal_N120_read=False."
            ),
        )

        return

    if scientific.get(
        "Stage4_inference"
    ) is not False:

        blocked(
            "reference_Stage4_inference",
            (
                "Reference decision unexpectedly "
                "used Stage4 inference."
            ),
        )

        return

    print(
        "reference structured gate = PASS"
    )

    print()
    print(
        "===== FINAL BLOCK5.3 FREEZE ====="
    )
'''

    require(
        old_execution in source,
        (
            "Could not locate reference-to-freeze "
            "transition in safe controller."
        ),
    )

    source = source.replace(
        old_execution,
        new_execution,
        1,
    )

    backup_path = backup(
        SAFE,
        old_sha,
    )

    temporary = SAFE.with_suffix(
        ".py.tmp"
    )

    temporary.write_text(
        source,
        encoding="utf-8",
    )

    temporary.replace(
        SAFE
    )

    new_sha = file_sha256(
        SAFE
    )

    return {
        "status":
            "PATCHED",

        "old_sha256":
            old_sha,

        "new_sha256":
            new_sha,

        "backup":
            str(
                backup_path
            ),
    }


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.3 — REFERENCE GATE REPAIR"
    )
    print(
        "============================================================"
    )

    missing = [
        str(
            path
        )
        for path in (
            REFERENCE,
            SAFE,
        )
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing repair target(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    print(
        "scientific parameters changed = NO"
    )

    print(
        "beam support changed          = NO"
    )

    print(
        "gain model changed            = NO"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "Stage4 inference              = NO"
    )

    reference_result = (
        patch_reference()
    )

    safe_result = (
        patch_safe_controller()
    )

    print()
    print(
        "reference parser patch =",
        reference_result[
            "status"
        ],
    )

    print(
        "reference old SHA256 =",
        reference_result[
            "old_sha256"
        ],
    )

    print(
        "reference new SHA256 =",
        reference_result[
            "new_sha256"
        ],
    )

    print()
    print(
        "safe-controller patch =",
        safe_result[
            "status"
        ],
    )

    print(
        "safe old SHA256 =",
        safe_result[
            "old_sha256"
        ],
    )

    print(
        "safe new SHA256 =",
        safe_result[
            "new_sha256"
        ],
    )

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in (
        REFERENCE,
        SAFE,
    ):

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

    payload = {
        "status":
            "PASS",

        "repair_scope": {
            "reference_evidence_link":
                (
                    "numeric assignment 12.0 "
                    "linked to f-string assumption "
                    "annotation by variable name"
                ),

            "safe_controller":
                (
                    "structured reference PASS "
                    "required before final freeze"
                ),
        },

        "scientific_changes": {
            "beam_support":
                False,

            "gain_model":
                False,

            "HPBW":
                False,

            "P_b":
                False,

            "S_b":
                False,

            "formal_tuning":
                False,
        },

        "scientific_execution": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "training":
                False,

            "recalibration":
                False,

            "upstream_modified":
                False,
        },

        "reference_script":
            reference_result,

        "safe_controller":
            safe_result,
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "REFERENCE GATE REPAIR FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "FOV evidence parser       = REPAIRED"
    )

    print(
        "structured PASS gate      = ADDED"
    )

    print(
        "scientific contract       = UNCHANGED"
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "STATUS = PASS"
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
        "REFERENCE GATE REPAIR = BLOCKED"
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

    print(
        "formal N=120 used = NO"
    )

    print(
        "Stage4 inference = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

from __future__ import annotations

import json
from pathlib import Path
import re
import traceback


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

INPUT = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_audit.json"
)

OUTPUT = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_decision.json"
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


def write_json(
    path,
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


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.3 PART 2 — PART-A REFERENCE DECISION"
    )
    print(
        "============================================================"
    )

    require(
        INPUT.is_file(),
        (
            "Part-A reference audit "
            "artifact missing."
        ),
    )

    payload = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    require(
        payload.get(
            "status"
        )
        ==
        "PASS_READ_ONLY_REFERENCE_AUDIT",
        (
            "Part-A reference audit "
            "is not PASS."
        ),
    )

    hits = (
        payload[
            "PartA"
        ][
            "hits"
        ]
    )

    fov_assignment_hits = []

    fov_assumption_hits = []

    strong_physical_gain_hits = []

    physical_pattern = re.compile(
        r"("
        r"beamwidth|"
        r"beam\s+width|"
        r"divergence|"
        r"half[-\s]?power|"
        r"optical\s+gain|"
        r"gain\s+pattern|"
        r"\bgopt\b|"
        r"\blpoint\b"
        r")",
        flags=re.IGNORECASE,
    )

    for hit in hits:

        text = str(
            hit.get(
                "text",
                ""
            )
        )

        if (
            "assumed_headlamp_fov_deg = 12.0"
            in
            text
        ):
            fov_assignment_hits.append(
                hit
            )

        if (
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

        if physical_pattern.search(
            text
        ):
            strong_physical_gain_hits.append(
                hit
            )

    require(
        fov_assignment_hits,
        (
            "Could not verify Part-A explicit "
            "assumed_headlamp_fov_deg=12.0."
        ),
    )

    require(
        fov_assumption_hits,
        (
            "Could not verify Part-A labels "
            "the +/-12 deg headlamp FOV as "
            "an assumption."
        ),
    )

    #
    # If this unexpectedly finds genuine physical gain
    # evidence, stop rather than silently overriding it.
    #
    require(
        not strong_physical_gain_hits,
        (
            "Unexpected Part-A physical "
            "beamwidth/gain evidence found. "
            "Inspect before freezing the "
            "constructed Stage5 gain contract."
        ),
    )

    decision = {
        "status":
            "PASS",

        "PartA_support_evidence": {
            "value_half_FOV_deg":
                12.0,

            "classification":
                "explicit_visualization_assumption",

            "paper_given":
                False,

            "measured":
                False,

            "communication_beamwidth":
                False,

            "reuse_in_Stage5":
                (
                    "constructed_support_"
                    "for_consistency_only"
                ),

            "assignment_hits":
                fov_assignment_hits,

            "assumption_label_hits":
                fov_assumption_hits,
        },

        "PartA_physical_communication_gain": {
            "usable_numeric_pattern_found":
                False,

            "strong_pattern_hits":
                [],

            "Stage5_action":
                (
                    "freeze_explicit_constructed_"
                    "normalized_directional_"
                    "gain_surrogate"
                ),
        },

        "false_positive_note":
            (
                "generic substring 'gain' may occur "
                "inside words such as 'against'; "
                "such hits are not optical-gain evidence"
            ),

        "scientific_execution": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "training":
                False,

            "filesystem_rescan":
                False,

            "upstream_modified":
                False,
        },
    }

    write_json(
        OUTPUT,
        decision,
    )

    print(
        "Part-A +/-12 deg FOV value = FOUND"
    )

    print(
        "classification              = VISUALIZATION ASSUMPTION"
    )

    print(
        "paper-given communication FOV = NO"
    )

    print(
        "measured communication FOV    = NO"
    )

    print(
        "usable physical gain pattern  = NO"
    )

    print(
        "Stage5 support route          = CONSTRUCTED REUSE"
    )

    print(
        "Stage5 gain route             = CONSTRUCTED SURROGATE"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "Stage4 inference              = NO"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        OUTPUT,
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
        "PART-A REFERENCE DECISION = BLOCKED"
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

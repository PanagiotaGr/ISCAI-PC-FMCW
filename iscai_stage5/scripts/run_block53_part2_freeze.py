from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.beam_directional_gain import (
    FROZEN_GAIN_IS_MEASURED,
    FROZEN_GAIN_IS_PARTA_OPTICAL_LINK_GAIN,
    FROZEN_GAIN_MODEL,
    FROZEN_GAIN_ROLE,
    FROZEN_HPBW_RULE,
    FROZEN_SUPPORT_HALF_ANGLE_DEG,
    FROZEN_SUPPORT_IS_MEASURED,
    FROZEN_SUPPORT_IS_PAPER_GIVEN,
    FROZEN_SUPPORT_MAX_DEG,
    FROZEN_SUPPORT_MIN_DEG,
    FROZEN_SUPPORT_PROVENANCE,
    build_frozen_directional_codebook,
    frozen_beam_scores_from_samples,
    normalized_gaussian_directional_gain,
)


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

PART1 = (
    STAGE5
    / "reports/"
      "block53_part1_codebook_kernel.json"
)

REFERENCE_DECISION = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_decision.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "beam_codebook_policy.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block53_codebook_probability.json"
)


EXPECTED_PART1_IMPLEMENTATION_SHA = (
    "1a705734d1de3a36e10ccd900633726af"
    "a6ca3a2fe0ff9ccfbf917cfd1867b75"
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


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


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


def implementation_fingerprint():
    roots = (
        STAGE5 / "src",
        STAGE5 / "tests",
        STAGE5 / "configs",
        STAGE5 / "scripts",
    )

    files = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE5
                )
            )
    )

    digest = sha256()

    for path in files:

        relative = str(
            path.relative_to(
                STAGE5
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )

        digest.update(
            b"\0"
        )

    return (
        len(
            files
        ),
        digest.hexdigest(),
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.3 PART 2/2"
    )
    print(
        "FROZEN DIRECTIONAL CODEBOOK + GAIN CONTRACT"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        REFERENCE_DECISION,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing Block5.3 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # --------------------------------------------------------
    # A. Part1 continuity
    # --------------------------------------------------------

    print()
    print(
        "===== A. BLOCK5.3 PART1 CONTINUITY ====="
    )

    part1 = load_json(
        PART1
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS",
        "Block5.3 Part1 is not PASS.",
    )

    require(
        part1[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_PART1_IMPLEMENTATION_SHA,
        (
            "Historical Block5.3 Part1 "
            "implementation SHA changed."
        ),
    )

    require(
        part1[
            "codebook_kernel"
        ][
            "beam_counts"
        ]
        ==
        [
            16,
            32,
            64,
        ],
        (
            "Frozen beam-count contract changed."
        ),
    )

    print(
        "Part1 status              = PASS"
    )

    print(
        "historical Part1 SHA      = PASS"
    )

    print(
        "beam counts               = 16 / 32 / 64"
    )

    # --------------------------------------------------------
    # B. Reference decision
    # --------------------------------------------------------

    print()
    print(
        "===== B. PART-A REFERENCE CLASSIFICATION ====="
    )

    reference = load_json(
        REFERENCE_DECISION
    )

    require(
        reference.get(
            "status"
        )
        ==
        "PASS",
        (
            "Part-A reference decision "
            "is not PASS."
        ),
    )

    support_evidence = (
        reference[
            "PartA_support_evidence"
        ]
    )

    require(
        support_evidence[
            "value_half_FOV_deg"
        ]
        ==
        12.0,
        (
            "Part-A visualization "
            "half-FOV value changed."
        ),
    )

    require(
        support_evidence[
            "classification"
        ]
        ==
        "explicit_visualization_assumption",
        (
            "Part-A FOV assumption "
            "classification changed."
        ),
    )

    require(
        support_evidence[
            "paper_given"
        ]
        is False,
        (
            "Part-A +/-12 deg support "
            "must not be labelled paper-given."
        ),
    )

    require(
        support_evidence[
            "measured"
        ]
        is False,
        (
            "Part-A +/-12 deg support "
            "must not be labelled measured."
        ),
    )

    require(
        reference[
            "PartA_physical_communication_gain"
        ][
            "usable_numeric_pattern_found"
        ]
        is False,
        (
            "Unexpected physical communication "
            "gain evidence appeared."
        ),
    )

    print(
        "Part-A half-FOV value     = +/-12 deg"
    )

    print(
        "classification            = VISUALIZATION ASSUMPTION"
    )

    print(
        "paper-given               = NO"
    )

    print(
        "measured                  = NO"
    )

    print(
        "physical comm gain found  = NO"
    )

    # --------------------------------------------------------
    # C. Freeze support/codebooks
    # --------------------------------------------------------

    print()
    print(
        "===== C. FROZEN AZIMUTH SUPPORT / CODEBOOKS ====="
    )

    require(
        FROZEN_SUPPORT_HALF_ANGLE_DEG
        ==
        12.0,
        (
            "Frozen Stage5 half-angle changed."
        ),
    )

    require(
        FROZEN_SUPPORT_IS_MEASURED
        is False,
        (
            "Constructed support cannot "
            "be labelled measured."
        ),
    )

    require(
        FROZEN_SUPPORT_IS_PAPER_GIVEN
        is False,
        (
            "Constructed support cannot "
            "be labelled paper-given."
        ),
    )

    codebook_rows = []

    expected_widths = {
        16:
            1.5,

        32:
            0.75,

        64:
            0.375,
    }

    for beam_count in (
        16,
        32,
        64,
    ):

        codebook = (
            build_frozen_directional_codebook(
                beam_count
            )
        )

        width_deg = math.degrees(
            codebook.decision_width_rad
        )

        require(
            abs(
                width_deg
                -
                expected_widths[
                    beam_count
                ]
            )
            <
            1e-12,
            (
                f"{beam_count}-beam "
                "decision width changed."
            ),
        )

        centers_deg = [
            math.degrees(
                cell.center_azimuth_rad
            )
            for cell
            in codebook.cells
        ]

        codebook_rows.append({
            "beam_count":
                beam_count,

            "support_min_deg":
                FROZEN_SUPPORT_MIN_DEG,

            "support_max_deg":
                FROZEN_SUPPORT_MAX_DEG,

            "decision_width_deg":
                width_deg,

            "centers_deg":
                centers_deg,
        })

        print(
            f"{beam_count:2d} beams:"
            f" width = {width_deg:.6f} deg PASS"
        )

    print(
        "support                   = [-12,+12] deg FROZEN"
    )

    print(
        "support provenance        = CONSTRUCTED / REFERENCE-CONSISTENT"
    )

    print(
        "decision cells            = NON-OVERLAPPING"
    )

    # --------------------------------------------------------
    # D. Gain contract
    # --------------------------------------------------------

    print()
    print(
        "===== D. FROZEN CODEBOOK GAIN-SCORE CONTRACT ====="
    )

    require(
        FROZEN_GAIN_IS_MEASURED
        is False,
        (
            "Constructed gain cannot "
            "be labelled measured."
        ),
    )

    require(
        FROZEN_GAIN_IS_PARTA_OPTICAL_LINK_GAIN
        is False,
        (
            "Codebook gain surrogate cannot "
            "be conflated with Part-A Gopt."
        ),
    )

    codebook = (
        build_frozen_directional_codebook(
            16
        )
    )

    left = (
        codebook.cells[
            7
        ]
    )

    right = (
        codebook.cells[
            8
        ]
    )

    boundary = (
        left.upper_azimuth_rad
    )

    left_gain = float(
        normalized_gaussian_directional_gain(
            left,
            np.asarray(
                [
                    boundary
                ]
            ),
        )[
            0
        ]
    )

    right_gain = float(
        normalized_gaussian_directional_gain(
            right,
            np.asarray(
                [
                    boundary
                ]
            ),
        )[
            0
        ]
    )

    require(
        abs(
            left_gain
            -
            0.5
        )
        <
        1e-13,
        (
            "Left beam does not have "
            "half-power boundary gain."
        ),
    )

    require(
        abs(
            right_gain
            -
            0.5
        )
        <
        1e-13,
        (
            "Right beam does not have "
            "half-power boundary gain."
        ),
    )

    print(
        "gain model                = NORMALIZED GAUSSIAN SURROGATE"
    )

    print(
        "HPBW rule                 = DECISION CELL WIDTH"
    )

    print(
        "adjacent boundary gain    = 0.5 / 0.5 PASS"
    )

    print(
        "physical overlap          = YES"
    )

    print(
        "role                      = CODEBOOK S_b SCORING ONLY"
    )

    print(
        "final optical Gopt        = NOT THIS MODEL"
    )

    # --------------------------------------------------------
    # E. P_b / S_b sanity
    # --------------------------------------------------------

    print()
    print(
        "===== E. P_b / S_b FROZEN SEMANTICS ====="
    )

    samples = np.radians(
        np.linspace(
            -15.0,
            15.0,
            2048,
        )
    )

    scores = (
        frozen_beam_scores_from_samples(
            azimuth_samples_rad=(
                samples
            ),

            beam_count=(
                32
            ),
        )
    )

    probability_total = (
        sum(
            scores
            .probability
            .masses
        )
        +
        scores
        .probability
        .outside_support_mass
    )

    require(
        abs(
            probability_total
            -
            1.0
        )
        <
        1e-12,
        (
            "Frozen beam probability "
            "accounting failed."
        ),
    )

    require(
        scores
        .probability
        .outside_support_mass
        >
        0.0,
        (
            "Outside-support posterior "
            "mass was not retained."
        ),
    )

    require(
        not np.allclose(
            np.asarray(
                scores
                .probability
                .masses
            ),

            np.asarray(
                scores
                .expected_gain
                .scores
            ),
        ),
        (
            "P_b and S_b were conflated."
        ),
    )

    print(
        "P_b                       = DECISION-CELL POSTERIOR MASS"
    )

    print(
        "S_b                       = E_p[G_b(theta)]"
    )

    print(
        "P_b != S_b                = PASS"
    )

    print(
        "outside-support mass      = EXPLICIT"
    )

    # --------------------------------------------------------
    # F. Freeze policy
    # --------------------------------------------------------

    policy_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.3",

        "status":
            "FROZEN",

        "azimuth_support": {
            "min_deg":
                FROZEN_SUPPORT_MIN_DEG,

            "max_deg":
                FROZEN_SUPPORT_MAX_DEG,

            "half_angle_deg":
                FROZEN_SUPPORT_HALF_ANGLE_DEG,

            "provenance":
                FROZEN_SUPPORT_PROVENANCE,

            "source_context":
                (
                    "Part-A explicit visualization "
                    "headlamp half-FOV assumption"
                ),

            "paper_given":
                False,

            "measured":
                False,

            "WOMD_quantity":
                False,

            "formal_tuning":
                False,
        },

        "codebooks":
            codebook_rows,

        "probability_accounting": {
            "P_b":
                (
                    "posterior_probability_mass_"
                    "inside_nonoverlapping_"
                    "decision_cell"
                ),

            "outside_support_mass":
                "reported_explicitly",

            "cells_overlap":
                False,
        },

        "directional_gain_score": {
            "S_b":
                "posterior_expectation_of_G_b",

            "model":
                FROZEN_GAIN_MODEL,

            "formula":
                (
                    "exp(-4*ln(2)*"
                    "((theta-theta_b)/HPBW_b)^2)"
                ),

            "HPBW_rule":
                FROZEN_HPBW_RULE,

            "gain_at_center":
                1.0,

            "gain_at_cell_boundary":
                0.5,

            "adjacent_patterns_overlap":
                True,

            "measured":
                False,

            "paper_given":
                False,

            "role":
                FROZEN_GAIN_ROLE,

            "is_final_PartA_optical_link_gain":
                False,

            "formal_tuning":
                False,
        },

        "elevation": {
            "decision_codebook":
                "NOT_DISCRETIZED_IN_BLOCK5.3",

            "posterior_dimension_preserved":
                True,

            "future_optical_pointing_use":
                "Block5.6",
        },

        "scope": {
            "adaptive_TopK_started":
                False,

            "beam_policy_baselines_started":
                False,

            "optical_link_budget_started":
                False,

            "ADB_started":
                False,
        },
    }

    write_json(
        POLICY,
        policy_payload,
    )

    policy_sha = file_sha256(
        POLICY
    )

    # --------------------------------------------------------
    # G. Implementation fingerprint + report
    # --------------------------------------------------------

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    report_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.3",

        "status":
            "PASS",

        "completion":
            "COMPLETE_FROZEN",

        "Part1_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_PART1_IMPLEMENTATION_SHA,
        },

        "reference_resolution": {
            "PartA_half_FOV_deg":
                12.0,

            "classification":
                "explicit_visualization_assumption",

            "usable_PartA_communication_beamwidth":
                False,

            "usable_PartA_physical_gain_pattern":
                False,
        },

        "frozen_codebook": {
            "beam_counts":
                [
                    16,
                    32,
                    64,
                ],

            "support_deg":
                [
                    -12.0,
                    12.0,
                ],

            "decision_width_deg": {
                "16":
                    1.5,

                "32":
                    0.75,

                "64":
                    0.375,
            },

            "support_is_constructed":
                True,

            "formal_tuning":
                False,
        },

        "frozen_gain_contract": {
            "model":
                FROZEN_GAIN_MODEL,

            "HPBW":
                "decision_cell_width",

            "adjacent_boundary_gain":
                [
                    left_gain,
                    right_gain,
                ],

            "overlap":
                True,

            "constructed":
                True,

            "is_final_optical_link_gain":
                False,

            "formal_tuning":
                False,
        },

        "beam_metrics": {
            "P_b":
                "posterior_mass",

            "S_b":
                "expected_directional_gain",

            "P_b_and_S_b_distinct":
                True,

            "outside_support_mass_explicit":
                True,
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

            "development_tuning":
                False,

            "upstream_modified":
                False,
        },

        "policy": {
            "path":
                str(
                    POLICY
                ),

            "sha256":
                policy_sha,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "next":
            "Block5.4 beam-policy baselines",
    }

    write_json(
        REPORT,
        report_payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.3 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 continuity              = PASS"
    )

    print(
        "beam counts                   = 16 / 32 / 64 FROZEN"
    )

    print(
        "azimuth support               = [-12,+12] deg FROZEN"
    )

    print(
        "support classification        = CONSTRUCTED ASSUMPTION"
    )

    print(
        "Part-A paper-given support    = NO"
    )

    print(
        "Part-A measured support       = NO"
    )

    print(
        "decision widths               = 1.5 / 0.75 / 0.375 deg"
    )

    print(
        "P_b                           = POSTERIOR MASS PASS"
    )

    print(
        "outside-support probability   = EXPLICIT"
    )

    print(
        "S_b                           = EXPECTED GAIN PASS"
    )

    print(
        "gain model                    = CONSTRUCTED GAUSSIAN"
    )

    print(
        "HPBW                           = DECISION CELL WIDTH"
    )

    print(
        "adjacent gain overlap          = PASS"
    )

    print(
        "final optical Gopt conflated   = NO"
    )

    print(
        "formal N=120 used              = NO"
    )

    print(
        "Stage4 inference               = NO"
    )

    print(
        "development/formal tuning      = NO"
    )

    print(
        "adaptive Top-K                 = NOT STARTED"
    )

    print(
        "beam policy baselines          = NOT STARTED"
    )

    print(
        "policy SHA256                  =",
        policy_sha,
    )

    print(
        "implementation files           =",
        implementation_files,
    )

    print(
        "implementation SHA256          =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "Block5.3 = COMPLETE / FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.4"
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
        "BLOCK 5.3 PART 2/2 = BLOCKED"
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
        "training/recalibration = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

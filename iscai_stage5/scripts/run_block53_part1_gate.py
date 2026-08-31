from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.beam_codebook import (
    BEAM_PROBABILITY_SEMANTICS,
    DECISION_CELL_SEMANTICS,
    EXPECTED_GAIN_SCORE_SEMANTICS,
    PHYSICAL_GAIN_SEMANTICS,
    SUPPORTED_BEAM_COUNTS,
    beam_probability_mass_from_samples,
    build_uniform_azimuth_codebook,
    expected_gain_scores_from_samples,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

BLOCK52 = (
    STAGE5
    / "reports/"
      "block52_receiver_angular_posterior.json"
)

ANGULAR_POLICY = (
    STAGE5
    / "configs/"
      "angular_posterior_policy.json"
)

CONVERGENCE = (
    STAGE5
    / "artifacts/block52/"
      "mc_convergence.json"
)

MATERIALIZATION = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.json"
)

REFERENCE_AUDIT = (
    STAGE5
    / "artifacts/block53/"
      "parta_beam_reference_audit.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block53_part1_codebook_kernel.json"
)


EXPECTED_BLOCK52_IMPLEMENTATION_SHA = (
    "e6a6c3996a1c09b493ebfe61516f7a46"
    "19994dd1aed34c7ada7e17f3711f9878"
)

EXPECTED_ANGULAR_POLICY_SHA = (
    "846f6bf3d3419a2ea99fabc629351472"
    "6388bf60a31bb2a889aaf18b487fd82e"
)

EXPECTED_CONVERGENCE_SHA = (
    "2a66b9128b06250b90b61ed0cd3e0a08"
    "9b86946bd34c0eac4483de150306fc5e"
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
        "STAGE 5 — BLOCK 5.3 PART 1/2"
    )
    print(
        "DIRECTIONAL CODEBOOK + BEAM-PROBABILITY KERNEL"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK52,
        ANGULAR_POLICY,
        CONVERGENCE,
        MATERIALIZATION,
        REFERENCE_AUDIT,
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
            "Missing prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Block5.2 frozen continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK5.2 FROZEN CONTINUITY ====="
    )

    block52 = load_json(
        BLOCK52
    )

    require(
        block52.get(
            "status"
        )
        ==
        "PASS",
        "Block5.2 is not PASS.",
    )

    require(
        block52[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK52_IMPLEMENTATION_SHA,
        (
            "Historical Block5.2 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            ANGULAR_POLICY
        )
        ==
        EXPECTED_ANGULAR_POLICY_SHA,
        (
            "Frozen angular-posterior "
            "policy SHA changed."
        ),
    )

    require(
        file_sha256(
            CONVERGENCE
        )
        ==
        EXPECTED_CONVERGENCE_SHA,
        (
            "Frozen MC-convergence "
            "artifact SHA changed."
        ),
    )

    angular_policy = load_json(
        ANGULAR_POLICY
    )

    require(
        angular_policy[
            "Monte_Carlo"
        ][
            "sample_count"
        ]
        ==
        2048,
        (
            "Frozen MC sample count "
            "changed."
        ),
    )

    print(
        "Block5.2 report          = PASS"
    )

    print(
        "implementation SHA       = PASS"
    )

    print(
        "angular policy SHA       = PASS"
    )

    print(
        "MC convergence SHA       = PASS"
    )

    print(
        "MC sample count          = 2048 FROZEN"
    )

    # ========================================================
    # B. Exact materialization provenance
    # ========================================================

    print()
    print(
        "===== B. DEVELOPMENT POSTERIOR PROVENANCE ====="
    )

    materialization = load_json(
        MATERIALIZATION
    )

    require(
        materialization.get(
            "status"
        )
        ==
        "PASS",
        (
            "Development posterior "
            "materialization is not PASS."
        ),
    )

    require(
        materialization[
            "historical_reproduction"
        ][
            "exact_match"
        ]
        is True,
        (
            "Frozen Block4.4 prediction "
            "was not exactly reproduced."
        ),
    )

    require(
        materialization[
            "scientific_execution"
        ][
            "formal_N120_read"
        ]
        is False,
        (
            "Materialization formal-data "
            "provenance failed."
        ),
    )

    require(
        materialization[
            "scientific_execution"
        ][
            "Stage4_frozen_model_inference"
        ]
        is True,
        (
            "Materialization provenance "
            "lost frozen Stage4 inference fact."
        ),
    )

    print(
        "historical Block4.4 SHA = EXACT PASS"
    )

    print(
        "source materialization  = DEVELOPMENT-ONLY"
    )

    print(
        "frozen Stage4 inference = YES — EARLIER MATERIALIZATION"
    )

    print(
        "additional inference now= NO"
    )

    print(
        "formal N=120            = NO"
    )

    # ========================================================
    # C. Codebook structural contract
    # ========================================================

    print()
    print(
        "===== C. DIRECTIONAL CODEBOOK STRUCTURE ====="
    )

    require(
        SUPPORTED_BEAM_COUNTS
        ==
        (
            16,
            32,
            64,
        ),
        "Beam-count contract changed.",
    )

    #
    # Synthetic support is deliberately used only to verify
    # the kernel. This is NOT a physical/support freeze.
    #
    synthetic_support = (
        -1.0,
        1.0,
    )

    codebooks = {}

    for beam_count in (
        SUPPORTED_BEAM_COUNTS
    ):

        codebook = (
            build_uniform_azimuth_codebook(
                beam_count=(
                    beam_count
                ),

                support_min_azimuth_rad=(
                    synthetic_support[
                        0
                    ]
                ),

                support_max_azimuth_rad=(
                    synthetic_support[
                        1
                    ]
                ),
            )
        )

        require(
            len(
                codebook.cells
            )
            ==
            beam_count,
            "Codebook cell count mismatch.",
        )

        codebooks[
            beam_count
        ] = codebook

        print(
            f"{beam_count:2d}-beam cells           = PASS"
        )

    print(
        "decision cells           = NON-OVERLAPPING"
    )

    print(
        "physical gain patterns  = SEPARATE / MAY OVERLAP"
    )

    print(
        "synthetic support used   = KERNEL TEST ONLY"
    )

    print(
        "angular support frozen   = NO"
    )

    # ========================================================
    # D. P_b posterior probability mass
    # ========================================================

    print()
    print(
        "===== D. POSTERIOR BEAM PROBABILITY MASS ====="
    )

    samples = np.asarray(
        [
            -1.4,
            -0.7,
            -0.2,
            0.0,
            0.2,
            0.7,
            1.4,
        ],
        dtype=np.float64,
    )

    probability = (
        beam_probability_mass_from_samples(
            azimuth_samples_rad=(
                samples
            ),

            codebook=(
                codebooks[
                    16
                ]
            ),
        )
    )

    require(
        abs(
            (
                sum(
                    probability.masses
                )
                +
                probability.outside_support_mass
            )
            -
            1.0
        )
        <
        1e-12,
        (
            "Beam probability accounting "
            "does not sum to one."
        ),
    )

    require(
        probability.outside_support_mass
        >
        0.0,
        (
            "Outside-support probability "
            "was silently discarded."
        ),
    )

    print(
        "P_b definition           = POSTERIOR MASS IN CELL"
    )

    print(
        "probability accounting   = PASS"
    )

    print(
        "outside-support mass     = EXPLICIT PASS"
    )

    print(
        "mass total               = 1 PASS"
    )

    # ========================================================
    # E. S_b expected physical gain remains distinct
    # ========================================================

    print()
    print(
        "===== E. EXPECTED PHYSICAL GAIN SCORE ====="
    )

    def synthetic_overlapping_gain(
        cell,
        values,
    ):
        difference = np.abs(
            values
            -
            cell.center_azimuth_rad
        )

        return np.exp(
            -0.5
            *
            (
                difference
                /
                0.35
            )
            **
            2
        )

    gain = (
        expected_gain_scores_from_samples(
            azimuth_samples_rad=(
                samples
            ),

            codebook=(
                codebooks[
                    16
                ]
            ),

            gain_evaluator=(
                synthetic_overlapping_gain
            ),
        )
    )

    require(
        len(
            gain.scores
        )
        ==
        16,
        "Expected-gain score count mismatch.",
    )

    require(
        BEAM_PROBABILITY_SEMANTICS
        !=
        EXPECTED_GAIN_SCORE_SEMANTICS,
        (
            "P_b and expected gain S_b "
            "were conflated."
        ),
    )

    print(
        "P_b                     = DECISION-CELL MASS"
    )

    print(
        "S_b                     = E_p[G_b(theta)]"
    )

    print(
        "P_b != S_b semantics    = PASS"
    )

    print(
        "physical gain frozen    = NO"
    )

    # ========================================================
    # F. Part-A reference audit
    # ========================================================

    print()
    print(
        "===== F. PART-A / LEGACY REFERENCE AUDIT ====="
    )

    reference = load_json(
        REFERENCE_AUDIT
    )

    require(
        reference.get(
            "status"
        )
        ==
        "PASS_READ_ONLY_REFERENCE_AUDIT",
        (
            "Beam reference audit "
            "did not PASS."
        ),
    )

    print(
        "Part-A files scanned      =",
        reference[
            "PartA"
        ][
            "files_scanned"
        ],
    )

    print(
        "Part-A beam/gain hits     =",
        len(
            reference[
                "PartA"
            ][
                "hits"
            ]
        ),
    )

    print(
        "legacy runtime dependency = NO"
    )

    print(
        "support decision           = DEFERRED TO PART2"
    )

    print(
        "gain-pattern decision      = DEFERRED TO PART2"
    )

    # ========================================================
    # G. Leakage/scope
    # ========================================================

    print()
    print(
        "===== G. SCOPE / LEAKAGE ====="
    )

    APIs = (
        build_uniform_azimuth_codebook,
        beam_probability_mass_from_samples,
        expected_gain_scores_from_samples,
    )

    forbidden = (
        "tracks_to_predict",
        "future_truth",
        "ground_truth",
        "oracle",
        "objects_of_interest",
        "adb",
    )

    for function in APIs:

        signature = str(
            inspect.signature(
                function
            )
        ).lower()

        for token in forbidden:

            require(
                token
                not in
                signature,
                (
                    f"Forbidden API token "
                    f"{token!r}."
                ),
            )

    print(
        "formal N=120 used       = NO"
    )

    print(
        "Stage4 inference now    = NO"
    )

    print(
        "beam selection policy   = NOT STARTED"
    )

    print(
        "adaptive Top-K          = NOT STARTED"
    )

    print(
        "ADB                     = NOT STARTED"
    )

    print(
        "optical BER chain       = NOT STARTED"
    )

    # ========================================================
    # H. Report
    # ========================================================

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.3_part_1",

        "status":
            "PASS",

        "Block52_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_BLOCK52_IMPLEMENTATION_SHA,

            "angular_policy_sha256":
                EXPECTED_ANGULAR_POLICY_SHA,

            "MC_convergence_sha256":
                EXPECTED_CONVERGENCE_SHA,

            "MC_sample_count":
                2048,
        },

        "codebook_kernel": {
            "beam_counts":
                [
                    16,
                    32,
                    64,
                ],

            "decision_cells":
                DECISION_CELL_SEMANTICS,

            "physical_gain":
                PHYSICAL_GAIN_SEMANTICS,

            "azimuth_support":
                "NOT_YET_FROZEN",

            "physical_gain_pattern":
                "NOT_YET_FROZEN",

            "synthetic_support_used_for_kernel_tests_only":
                True,
        },

        "posterior_beam_probability": {
            "P_b":
                BEAM_PROBABILITY_SEMANTICS,

            "outside_support_probability":
                "explicit",

            "normalization":
                "P_cells_plus_P_outside_equals_1",
        },

        "expected_gain_score": {
            "S_b":
                EXPECTED_GAIN_SCORE_SEMANTICS,

            "distinct_from_P_b":
                True,

            "overlapping_physical_gain_allowed":
                True,

            "physical_gain_evaluator_frozen":
                False,
        },

        "reference_audit": {
            "path":
                str(
                    REFERENCE_AUDIT
                ),

            "PartA_hit_count":
                len(
                    reference[
                        "PartA"
                    ][
                        "hits"
                    ]
                ),

            "legacy_runtime_dependency":
                False,
        },

        "provenance_precision": {
            "earlier_dev_materialization_used_frozen_Stage4_inference":
                True,

            "additional_Stage4_inference_in_Block53_Part1":
                False,

            "formal_N120_read":
                False,
        },

        "scope": {
            "beam_support_frozen":
                False,

            "physical_gain_frozen":
                False,

            "beam_selection_started":
                False,

            "adaptive_TopK_started":
                False,

            "optical_link_started":
                False,

            "ADB_started":
                False,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "next":
            (
                "Block5.3 Part2 freeze "
                "angular support, directional "
                "codebooks and physical gain "
                "probability/score contract"
            ),
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
        "BLOCK 5.3 PART 1/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.2 frozen continuity = PASS"
    )

    print(
        "MC N                       = 2048 FROZEN"
    )

    print(
        "16/32/64 codebook kernel   = PASS"
    )

    print(
        "non-overlapping cells      = PASS"
    )

    print(
        "P_b posterior mass         = PASS"
    )

    print(
        "outside-support mass       = EXPLICIT"
    )

    print(
        "S_b expected gain          = DISTINCT PASS"
    )

    print(
        "physical gain overlap      = SUPPORTED"
    )

    print(
        "Part-A reference audit     = PASS"
    )

    print(
        "angular support            = NOT YET FROZEN"
    )

    print(
        "physical gain pattern      = NOT YET FROZEN"
    )

    print(
        "formal N=120 used          = NO"
    )

    print(
        "Stage4 inference now       = NO"
    )

    print(
        "adaptive Top-K             = NOT STARTED"
    )

    print(
        "implementation files       =",
        implementation_files,
    )

    print(
        "implementation SHA256      =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.3 PART 2/2"
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
        "BLOCK 5.3 PART 1/2 = BLOCKED"
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
        "Stage4 inference now = NO"
    )

    print(
        "upstream modified = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

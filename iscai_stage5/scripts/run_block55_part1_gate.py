from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.adaptive_topk import (
    ADAPTIVE_SELECTION_RULE,
    ADAPTIVE_TIE_BREAK,
    FROZEN_COVERAGE_TARGETS,
    FROZEN_NOMINAL_COVERAGE,
    KMAX_POLICY_STATUS,
    OUTSIDE_SUPPORT_POLICY,
    adaptive_topk_probability_mass,
    all_frozen_coverage_targets,
    nominal_adaptive_topk,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)


STAGE5 = Path(
    "/home/agni/waymo/iscai_stage5"
)

BLOCK54 = (
    STAGE5
    / "reports/"
      "block54_beam_baselines.json"
)

BASELINE_POLICY = (
    STAGE5
    / "configs/"
      "beam_baseline_policy.json"
)

CODEBOOK_POLICY = (
    STAGE5
    / "configs/"
      "beam_codebook_policy.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block55_part1_adaptive_topk_core.json"
)


EXPECTED_BLOCK54_IMPLEMENTATION_SHA = (
    "1fd8e9417ed64014e176c2536ebbce14"
    "db074eff94dea6e270d4d412e4a8daef"
)

EXPECTED_BLOCK54_POLICY_SHA = (
    "0eef4bb57955ea6cba011b5e9b3ff0a6"
    "506d7696d0691652a3256f03d2a4ca14"
)

EXPECTED_CODEBOOK_POLICY_SHA = (
    "bd94f8609393a7c9fe02762cc4bf38e3"
    "a77a90cc31adef5f2e6b06aece4074d7"
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
        "STAGE 5 — BLOCK 5.5 PART 1/2"
    )
    print(
        "ADAPTIVE TOP-K PROBABILITY-MASS CORE"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK54,
        BASELINE_POLICY,
        CODEBOOK_POLICY,
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
            "Missing Block5.5 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # --------------------------------------------------------
    # A. Block5.4 continuity
    # --------------------------------------------------------

    print()
    print(
        "===== A. BLOCK5.4 FROZEN CONTINUITY ====="
    )

    block54 = json.loads(
        BLOCK54.read_text(
            encoding="utf-8"
        )
    )

    require(
        block54.get(
            "status"
        )
        ==
        "PASS",
        "Block5.4 is not PASS.",
    )

    require(
        block54.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        "Block5.4 is not COMPLETE_FROZEN.",
    )

    require(
        block54[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK54_IMPLEMENTATION_SHA,
        (
            "Historical Block5.4 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            BASELINE_POLICY
        )
        ==
        EXPECTED_BLOCK54_POLICY_SHA,
        (
            "Frozen beam-baseline "
            "policy SHA changed."
        ),
    )

    require(
        file_sha256(
            CODEBOOK_POLICY
        )
        ==
        EXPECTED_CODEBOOK_POLICY_SHA,
        (
            "Frozen codebook policy "
            "SHA changed."
        ),
    )

    print(
        "Block5.4                = COMPLETE / FROZEN"
    )

    print(
        "implementation SHA      = PASS"
    )

    print(
        "baseline policy SHA      = PASS"
    )

    print(
        "codebook policy SHA      = PASS"
    )

    # --------------------------------------------------------
    # B. Frozen coverage targets
    # --------------------------------------------------------

    print()
    print(
        "===== B. COVERAGE TARGET CONTRACT ====="
    )

    require(
        FROZEN_COVERAGE_TARGETS
        ==
        (
            0.90,
            0.95,
            0.975,
            0.99,
        ),
        (
            "Coverage targets changed."
        ),
    )

    require(
        FROZEN_NOMINAL_COVERAGE
        ==
        0.95,
        (
            "Nominal coverage changed."
        ),
    )

    print(
        "coverage targets         = 90 / 95 / 97.5 / 99 %"
    )

    print(
        "nominal target           = 95 %"
    )

    print(
        "selection rule           = SMALLEST MASS-COVERING SET"
    )

    print(
        "ranking                  = DESCENDING P_b"
    )

    print(
        "tie-break                = LOWER BEAM INDEX"
    )

    # --------------------------------------------------------
    # C. Minimality sanity
    # --------------------------------------------------------

    print()
    print(
        "===== C. ADAPTIVE MINIMALITY ====="
    )

    codebook = (
        build_frozen_directional_codebook(
            32
        )
    )

    masses = np.zeros(
        32,
        dtype=np.float64,
    )

    masses[
        16
    ] = 0.50

    masses[
        15
    ] = 0.30

    masses[
        17
    ] = 0.15

    masses[
        14
    ] = 0.05

    probability = BeamProbabilityMass(
        masses=tuple(
            float(
                value
            )
            for value in masses
        ),

        inside_support_mass=1.0,

        outside_support_mass=0.0,

        sample_count=2048,
    )

    nominal = (
        nominal_adaptive_topk(
            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )
    )

    require(
        nominal.beam_indices
        ==
        (
            16,
            15,
            17,
        ),
        (
            "Nominal adaptive Top-K "
            "minimal prefix changed."
        ),
    )

    require(
        nominal.k
        ==
        3,
        (
            "Nominal adaptive K changed."
        ),
    )

    require(
        nominal.achieved_requested_coverage
        is True,
        (
            "Nominal 95% coverage "
            "was not achieved."
        ),
    )

    print(
        "synthetic nominal K      = 3 PASS"
    )

    print(
        "selected mass            =",
        nominal.selected_in_support_mass,
    )

    print(
        "minimal-prefix rule      = PASS"
    )

    # --------------------------------------------------------
    # D. Coverage monotonicity
    # --------------------------------------------------------

    print()
    print(
        "===== D. COVERAGE-TARGET MONOTONICITY ====="
    )

    results = (
        all_frozen_coverage_targets(
            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )
    )

    ks = [
        results[
            q
        ].k
        for q in (
            FROZEN_COVERAGE_TARGETS
        )
    ]

    require(
        ks
        ==
        sorted(
            ks
        ),
        (
            "Adaptive K is not monotonic "
            "with requested coverage."
        ),
    )

    for q in (
        FROZEN_COVERAGE_TARGETS
    ):

        result = (
            results[
                q
            ]
        )

        print(
            f"q={100*q:5.1f}%"
            f" -> K={result.k}"
            f" mass={result.selected_in_support_mass:.6f}"
        )

    print(
        "K monotonicity           = PASS"
    )

    # --------------------------------------------------------
    # E. Outside-support honesty
    # --------------------------------------------------------

    print()
    print(
        "===== E. OUTSIDE-SUPPORT COVERAGE CONTRACT ====="
    )

    outside_masses = np.zeros(
        32,
        dtype=np.float64,
    )

    outside_masses[
        16
    ] = 0.55

    outside_masses[
        15
    ] = 0.30

    outside_masses[
        17
    ] = 0.05

    outside_probability = BeamProbabilityMass(
        masses=tuple(
            float(
                value
            )
            for value in (
                outside_masses
            )
        ),

        inside_support_mass=0.90,

        outside_support_mass=0.10,

        sample_count=2048,
    )

    impossible = (
        adaptive_topk_probability_mass(
            probability=(
                outside_probability
            ),

            codebook=(
                codebook
            ),

            requested_coverage=0.95,
        )
    )

    require(
        impossible
        .achieved_requested_coverage
        is False,
        (
            "Controller falsely claimed "
            "95% coverage with only "
            "90% in-support mass."
        ),
    )

    require(
        impossible
        .unattainable_due_to_outside_support
        is True,
        (
            "Outside-support unattainability "
            "was not explicit."
        ),
    )

    print(
        "outside-support mass     = EXPLICIT"
    )

    print(
        "renormalization away     = NO"
    )

    print(
        "false coverage claim      = BLOCKED PASS"
    )

    # --------------------------------------------------------
    # F. Kmax structural support
    # --------------------------------------------------------

    print()
    print(
        "===== F. KMAX STRUCTURAL CONTRACT ====="
    )

    uniform = np.ones(
        32,
        dtype=np.float64,
    ) / 32.0

    uniform_probability = BeamProbabilityMass(
        masses=tuple(
            float(
                value
            )
            for value in uniform
        ),

        inside_support_mass=1.0,

        outside_support_mass=0.0,

        sample_count=2048,
    )

    capped = (
        adaptive_topk_probability_mass(
            probability=(
                uniform_probability
            ),

            codebook=(
                codebook
            ),

            requested_coverage=0.95,

            kmax=5,
        )
    )

    require(
        capped.k
        ==
        5,
        "Kmax cap failed.",
    )

    require(
        capped.kmax_limited
        is True,
        (
            "Kmax-limited condition "
            "was not explicit."
        ),
    )

    require(
        capped
        .achieved_requested_coverage
        is False,
        (
            "Kmax-limited selection "
            "falsely claimed target coverage."
        ),
    )

    print(
        "Kmax API                 = SUPPORTED"
    )

    print(
        "numeric Kmax freeze      = DEFERRED TO PART2"
    )

    print(
        "Kmax failure reporting   = EXPLICIT PASS"
    )

    # --------------------------------------------------------
    # G. Leakage / scope
    # --------------------------------------------------------

    print()
    print(
        "===== G. SCOPE / LEAKAGE ====="
    )

    signature = str(
        inspect.signature(
            adaptive_topk_probability_mass
        )
    ).lower()

    for token in (
        "future",
        "ground_truth",
        "realized",
        "oracle",
        "tracks_to_predict",
        "objects_of_interest",
    ):

        require(
            token
            not in
            signature,
            (
                f"Forbidden adaptive API "
                f"token {token!r}."
            ),
        )

    print(
        "formal N=120 used        = NO"
    )

    print(
        "Stage4 inference         = NO"
    )

    print(
        "development tuning       = NO"
    )

    print(
        "training/recalibration   = NO"
    )

    print(
        "hysteresis               = NOT STARTED"
    )

    print(
        "neighbor sweep           = NOT STARTED"
    )

    print(
        "fallback/reacquisition   = NOT STARTED"
    )

    print(
        "optical link             = NOT STARTED"
    )

    # --------------------------------------------------------
    # H. Part1 report
    # --------------------------------------------------------

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
            "5.5_part_1",

        "status":
            "PASS",

        "Block54_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_BLOCK54_IMPLEMENTATION_SHA,

            "beam_baseline_policy_sha256":
                EXPECTED_BLOCK54_POLICY_SHA,

            "beam_codebook_policy_sha256":
                EXPECTED_CODEBOOK_POLICY_SHA,
        },

        "coverage_targets": {
            "values":
                list(
                    FROZEN_COVERAGE_TARGETS
                ),

            "nominal":
                FROZEN_NOMINAL_COVERAGE,

            "status":
                "FROZEN",
        },

        "adaptive_core": {
            "rule":
                ADAPTIVE_SELECTION_RULE,

            "ranking":
                "descending_P_b",

            "tie_break":
                ADAPTIVE_TIE_BREAK,

            "smallest_set":
                True,

            "outside_support_policy":
                OUTSIDE_SUPPORT_POLICY,

            "outside_support_renormalized":
                False,
        },

        "Kmax": {
            "API_supported":
                True,

            "numeric_value_frozen":
                False,

            "status":
                KMAX_POLICY_STATUS,
        },

        "scope": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "development_tuning":
                False,

            "training":
                False,

            "recalibration":
                False,

            "hysteresis_started":
                False,

            "neighbor_sweep_started":
                False,

            "fallback_started":
                False,

            "optical_link_started":
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
                "Block5.5 Part2 freeze Kmax, "
                "temporal hysteresis/persistence, "
                "neighbor/fallback/reacquisition "
                "controller contract using "
                "non-formal development only"
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
        "BLOCK 5.5 PART 1/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.4 continuity       = PASS"
    )

    print(
        "coverage q                = 90 / 95 / 97.5 / 99 % FROZEN"
    )

    print(
        "nominal q                 = 95 % FROZEN"
    )

    print(
        "adaptive rule             = SMALLEST MASS-COVERING SET"
    )

    print(
        "ranking                   = P_b"
    )

    print(
        "tie-break                 = LOWER BEAM INDEX"
    )

    print(
        "outside-support mass      = EXPLICIT"
    )

    print(
        "outside renormalization   = NO"
    )

    print(
        "Kmax support              = PASS"
    )

    print(
        "numeric Kmax              = NOT YET FROZEN"
    )

    print(
        "hysteresis/fallback       = NOT STARTED"
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "implementation files      =",
        implementation_files,
    )

    print(
        "implementation SHA256     =",
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
        "next = BLOCK 5.5 PART 2/2"
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
        "BLOCK 5.5 PART 1/2 = BLOCKED"
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

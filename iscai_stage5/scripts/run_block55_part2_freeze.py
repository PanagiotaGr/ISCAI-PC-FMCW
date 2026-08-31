from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.adaptive_topk import (
    FROZEN_COVERAGE_TARGETS,
    FROZEN_NOMINAL_COVERAGE,
)

from iscai_stage5.adaptive_topk_temporal import (
    COARSER_CODEBOOK,
    FROZEN_MC_SAMPLE_COUNT,
    FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE,
    HYSTERESIS_RULE,
    KMAX_RULE,
    LOSS_OF_LOCK_RECOVERY,
    LOSS_OF_LOCK_RULE,
    NEIGHBOR_SWEEP_RULE,
    SWITCH_PENALTY_RULE,
    WIDENED_FALLBACK_RULE,
    adaptive_temporal_step,
    all_coverage_temporal_decisions,
    frozen_kmax_for_codebook,
    initial_adaptive_temporal_state,
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

PART1 = (
    STAGE5
    / "reports/"
      "block55_part1_adaptive_topk_core.json"
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

ANGULAR_POLICY = (
    STAGE5
    / "configs/"
      "angular_posterior_policy.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "adaptive_topk_policy.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block55_adaptive_topk.json"
)


EXPECTED_PART1_IMPLEMENTATION_SHA = (
    "feb6353ff320ac9c8f7839d546fe9df3"
    "a7a3a6baacab7fa978d5875caadb4997"
)

EXPECTED_BLOCK54_POLICY_SHA = (
    "0eef4bb57955ea6cba011b5e9b3ff0a6"
    "506d7696d0691652a3256f03d2a4ca14"
)

EXPECTED_CODEBOOK_POLICY_SHA = (
    "bd94f8609393a7c9fe02762cc4bf38e3"
    "a77a90cc31adef5f2e6b06aece4074d7"
)

EXPECTED_ANGULAR_POLICY_SHA = (
    "846f6bf3d3419a2ea99fabc629351472"
    "6388bf60a31bb2a889aaf18b487fd82e"
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


def synthetic_probability(
    beam_count,
):
    masses = np.zeros(
        beam_count,
        dtype=np.float64,
    )

    center = (
        beam_count
        //
        2
    )

    masses[
        center
    ] = 0.55

    masses[
        center - 1
    ] = 0.25

    masses[
        center + 1
    ] = 0.15

    masses[
        center - 2
    ] = 0.05

    return BeamProbabilityMass(
        masses=tuple(
            float(
                value
            )
            for value in masses
        ),

        inside_support_mass=1.0,

        outside_support_mass=0.0,

        sample_count=(
            FROZEN_MC_SAMPLE_COUNT
        ),
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.5 PART 2/2"
    )
    print(
        "ADAPTIVE TOP-K TEMPORAL / RECOVERY CONTRACT FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK54,
        BASELINE_POLICY,
        CODEBOOK_POLICY,
        ANGULAR_POLICY,
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

    # ========================================================
    # A. Frozen continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN CONTINUITY ====="
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
        (
            "Block5.5 Part1 is not PASS."
        ),
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
            "Historical Block5.5 Part1 "
            "implementation SHA changed."
        ),
    )

    block54 = load_json(
        BLOCK54
    )

    require(
        block54.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.4 is not COMPLETE_FROZEN."
        ),
    )

    require(
        file_sha256(
            BASELINE_POLICY
        )
        ==
        EXPECTED_BLOCK54_POLICY_SHA,
        (
            "Frozen Block5.4 policy SHA changed."
        ),
    )

    require(
        file_sha256(
            CODEBOOK_POLICY
        )
        ==
        EXPECTED_CODEBOOK_POLICY_SHA,
        (
            "Frozen codebook policy SHA changed."
        ),
    )

    require(
        file_sha256(
            ANGULAR_POLICY
        )
        ==
        EXPECTED_ANGULAR_POLICY_SHA,
        (
            "Frozen angular-posterior policy "
            "SHA changed."
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
        FROZEN_MC_SAMPLE_COUNT,
        (
            "Frozen MC sample count changed."
        ),
    )

    print(
        "Block5.5 Part1          = PASS"
    )

    print(
        "historical Part1 SHA    = PASS"
    )

    print(
        "Block5.4               = COMPLETE / FROZEN"
    )

    print(
        "baseline policy SHA     = PASS"
    )

    print(
        "codebook policy SHA     = PASS"
    )

    print(
        "angular policy SHA      = PASS"
    )

    print(
        "MC sample count         = 2048 FROZEN"
    )

    # ========================================================
    # B. Coverage / numerical tolerance
    # ========================================================

    print()
    print(
        "===== B. COVERAGE ACCEPTANCE CONTRACT ====="
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

    require(
        FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE
        ==
        1.0
        /
        2048.0,
        (
            "Posterior-mass numerical "
            "tolerance changed."
        ),
    )

    print(
        "coverage targets             = 90 / 95 / 97.5 / 99 %"
    )

    print(
        "nominal coverage             = 95 %"
    )

    print(
        "posterior mass tolerance     =",
        FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE,
    )

    print(
        "tolerance provenance         = 1 / FROZEN MC N"
    )

    print(
        "empirical GT containment     = REPORT SEPARATELY"
    )

    # ========================================================
    # C. Kmax freeze
    # ========================================================

    print()
    print(
        "===== C. KMAX FREEZE ====="
    )

    kmax_rows = {}

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

        kmax = (
            frozen_kmax_for_codebook(
                codebook
            )
        )

        require(
            kmax
            ==
            beam_count,
            (
                "Kmax is not the full "
                "frozen codebook size."
            ),
        )

        kmax_rows[
            str(
                beam_count
            )
        ] = (
            kmax
        )

        print(
            f"{beam_count:2d}-beam Kmax = {kmax} PASS"
        )

    print(
        "Kmax rule                   = FULL CODEBOOK SIZE"
    )

    print(
        "Kmax tuned on development    = NO"
    )

    print(
        "Kmax can cause false failure = NO"
    )

    # ========================================================
    # D. Temporal hysteresis / persistence
    # ========================================================

    print()
    print(
        "===== D. TEMPORAL HYSTERESIS / PERSISTENCE ====="
    )

    codebook = (
        build_frozen_directional_codebook(
            16
        )
    )

    first_mass = np.zeros(
        16,
        dtype=np.float64,
    )

    first_mass[
        8
    ] = 0.60

    first_mass[
        7
    ] = 0.35

    first_mass[
        9
    ] = 0.05

    first_probability = (
        BeamProbabilityMass(
            masses=tuple(
                first_mass
            ),

            inside_support_mass=1.0,

            outside_support_mass=0.0,

            sample_count=2048,
        )
    )

    state = (
        initial_adaptive_temporal_state()
    )

    first, state = (
        adaptive_temporal_step(
            state=(
                state
            ),

            probability=(
                first_probability
            ),

            codebook=(
                codebook
            ),
        )
    )

    require(
        first.primary_beam_index
        ==
        8,
        (
            "Initial adaptive primary changed."
        ),
    )

    second_mass = np.zeros(
        16,
        dtype=np.float64,
    )

    second_mass[
        7
    ] = 0.55

    second_mass[
        8
    ] = 0.40

    second_mass[
        9
    ] = 0.05

    second_probability = (
        BeamProbabilityMass(
            masses=tuple(
                second_mass
            ),

            inside_support_mass=1.0,

            outside_support_mass=0.0,

            sample_count=2048,
        )
    )

    second, state = (
        adaptive_temporal_step(
            state=(
                state
            ),

            probability=(
                second_probability
            ),

            codebook=(
                codebook
            ),
        )
    )

    require(
        second.primary_beam_index
        ==
        8,
        (
            "Previous primary was not "
            "retained inside current "
            "mass-covering set."
        ),
    )

    require(
        second
        .previous_primary_retained
        is True,
        (
            "Hysteresis retention flag failed."
        ),
    )

    require(
        second
        .switch_penalty_units
        ==
        0,
        (
            "No-switch step accumulated "
            "switch penalty."
        ),
    )

    require(
        set(
            second
            .ordered_beam_indices
        )
        ==
        set(
            second
            .adaptive_selection
            .beam_indices
        ),
        (
            "Hysteresis altered the "
            "mass-covering beam set."
        ),
    )

    print(
        "hysteresis rule              = PREVIOUS PRIMARY IF STILL IN SET"
    )

    print(
        "coverage set modified         = NO"
    )

    print(
        "numeric hysteresis margin     = NONE"
    )

    print(
        "switch penalty                = UNIT EVENT INDICATOR"
    )

    print(
        "tuned switch weight           = NO"
    )

    # ========================================================
    # E. Neighbor / widened recovery
    # ========================================================

    print()
    print(
        "===== E. LOCAL / WIDENED RECOVERY ====="
    )

    require(
        COARSER_CODEBOOK
        ==
        {
            64:
                32,

            32:
                16,

            16:
                None,
        },
        (
            "Coarser-codebook chain changed."
        ),
    )

    for beam_count in (
        64,
        32,
        16,
    ):

        codebook = (
            build_frozen_directional_codebook(
                beam_count
            )
        )

        probability = (
            synthetic_probability(
                beam_count
            )
        )

        decision, _ = (
            adaptive_temporal_step(
                state=(
                    initial_adaptive_temporal_state()
                ),

                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),
            )
        )

        require(
            len(
                decision
                .local_neighbor_probe_indices
            )
            in
            (
                1,
                2,
            ),
            (
                "Local graph-neighbor "
                "sweep changed."
            ),
        )

        expected_coarser = (
            COARSER_CODEBOOK[
                beam_count
            ]
        )

        if expected_coarser is None:

            require(
                decision
                .widened_fallback
                .available
                is False,
                (
                    "Unexpected coarser fallback "
                    "for 16-beam codebook."
                ),
            )

        else:

            require(
                decision
                .widened_fallback
                .available
                is True,
                (
                    "Expected widened fallback "
                    "is unavailable."
                ),
            )

            require(
                decision
                .widened_fallback
                .fallback_beam_count
                ==
                expected_coarser,
                (
                    "Widened fallback codebook "
                    "changed."
                ),
            )

        print(
            f"{beam_count:2d}-beam recovery contract = PASS"
        )

    print(
        "local neighbor radius        = GRAPH ADJACENCY ONLY"
    )

    print(
        "widened fallback             = 64->32->16"
    )

    print(
        "new beamwidth parameter      = NO"
    )

    # ========================================================
    # F. Loss-of-lock exhaustive recovery
    # ========================================================

    print()
    print(
        "===== F. LOSS-OF-LOCK / EXHAUSTIVE RECOVERY ====="
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
    ] = 0.55

    masses[
        15
    ] = 0.30

    masses[
        17
    ] = 0.05

    probability = BeamProbabilityMass(
        masses=tuple(
            masses
        ),

        inside_support_mass=0.90,

        outside_support_mass=0.10,

        sample_count=2048,
    )

    failed, _ = (
        adaptive_temporal_step(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                probability
            ),

            codebook=(
                codebook
            ),

            requested_coverage=0.95,
        )
    )

    require(
        failed.loss_of_lock
        is True,
        (
            "Unattainable coverage did "
            "not trigger loss-of-lock."
        ),
    )

    require(
        failed
        .exhaustive_fallback_active
        is True,
        (
            "Loss-of-lock did not activate "
            "exhaustive sweep."
        ),
    )

    require(
        failed
        .exhaustive_fallback_indices
        ==
        tuple(
            range(
                32
            )
        ),
        (
            "Exhaustive recovery beam set changed."
        ),
    )

    require(
        failed.loss_of_lock_reason
        ==
        (
            "outside_support_mass_prevents_"
            "requested_coverage"
        ),
        (
            "Loss-of-lock reason changed."
        ),
    )

    print(
        "loss-of-lock trigger         = UNATTAINABLE REQUESTED MASS"
    )

    print(
        "outside-support honesty      = PASS"
    )

    print(
        "recovery                     = WIDENED THEN EXHAUSTIVE"
    )

    print(
        "exhaustive beam set          = ALL CURRENT CODEBOOK BEAMS"
    )

    # ========================================================
    # G. All coverage targets / leakage
    # ========================================================

    print()
    print(
        "===== G. ALL q TARGETS / LEAKAGE ====="
    )

    probability = (
        synthetic_probability(
            32
        )
    )

    decisions = (
        all_coverage_temporal_decisions(
            state=(
                initial_adaptive_temporal_state()
            ),

            probability=(
                probability
            ),

            codebook=(
                codebook
            ),
        )
    )

    require(
        tuple(
            decisions.keys()
        )
        ==
        FROZEN_COVERAGE_TARGETS,
        (
            "Frozen coverage-target "
            "suite changed."
        ),
    )

    for q, decision in (
        decisions.items()
    ):

        require(
            decision
            .adaptive_selection
            .selected_in_support_mass
            +
            FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE
            >=
            q,
            (
                "Synthetic in-support "
                "coverage gate failed."
            ),
        )

        print(
            f"q={100*q:5.1f}%"
            f" K={decision.adaptive_selection.k}"
            f" mass="
            f"{decision.adaptive_selection.selected_in_support_mass:.6f}"
        )

    signature = str(
        inspect.signature(
            adaptive_temporal_step
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
            token not in signature,
            (
                f"Forbidden adaptive controller "
                f"API token {token!r}."
            ),
        )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "development tuning          = NO"
    )

    print(
        "training/recalibration      = NO"
    )

    print(
        "future receiver truth       = NO"
    )

    print(
        "oracle controller access    = NO"
    )

    print(
        "optical link                = NOT STARTED"
    )

    # ========================================================
    # H. Freeze policy
    # ========================================================

    policy_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.5",

        "status":
            "FROZEN",

        "coverage": {
            "targets":
                list(
                    FROZEN_COVERAGE_TARGETS
                ),

            "nominal":
                FROZEN_NOMINAL_COVERAGE,

            "posterior_mass_numerical_tolerance":
                FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE,

            "tolerance_provenance":
                "one_MC_probability_quantum_1_over_2048",

            "empirical_future_receiver_containment":
                "REPORT_SEPARATELY_NOT_CONTROLLER_INPUT",
        },

        "adaptive_selection": {
            "rule":
                (
                    "smallest_descending_P_b_prefix_"
                    "reaching_requested_coverage"
                ),

            "tie_break":
                "lower_beam_index",

            "outside_support_renormalized":
                False,
        },

        "Kmax": {
            "rule":
                KMAX_RULE,

            "values_by_codebook":
                kmax_rows,

            "development_tuned":
                False,

            "formal_tuned":
                False,
        },

        "hysteresis_persistence": {
            "rule":
                HYSTERESIS_RULE,

            "changes_selected_mass_covering_set":
                False,

            "numeric_margin":
                None,

            "switch_penalty":
                SWITCH_PENALTY_RULE,

            "switch_weight_tuned":
                False,
        },

        "local_neighbor_sweep": {
            "rule":
                NEIGHBOR_SWEEP_RULE,

            "radius_parameter":
                None,

            "graph_adjacency_only":
                True,
        },

        "widened_fallback": {
            "rule":
                WIDENED_FALLBACK_RULE,

            "codebook_chain":
                {
                    "64":
                        32,

                    "32":
                        16,

                    "16":
                        None,
                },

            "new_beamwidth_parameter":
                False,
        },

        "loss_of_lock": {
            "rule":
                LOSS_OF_LOCK_RULE,

            "recovery":
                LOSS_OF_LOCK_RECOVERY,

            "exhaustive_current_codebook":
                True,

            "false_coverage_claim":
                False,
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

            "future_truth_controller_input":
                False,

            "oracle_controller_input":
                False,

            "optical_link_started":
                False,

            "ADB_started":
                False,
        },
    }

    write_json(
        POLICY,
        policy_payload,
    )

    policy_sha = (
        file_sha256(
            POLICY
        )
    )

    # ========================================================
    # I. Final report
    # ========================================================

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
            "5.5",

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

        "controller": {
            "coverage_targets":
                list(
                    FROZEN_COVERAGE_TARGETS
                ),

            "nominal_coverage":
                FROZEN_NOMINAL_COVERAGE,

            "Kmax":
                kmax_rows,

            "hysteresis":
                HYSTERESIS_RULE,

            "neighbor_sweep":
                NEIGHBOR_SWEEP_RULE,

            "widened_fallback":
                WIDENED_FALLBACK_RULE,

            "loss_of_lock":
                LOSS_OF_LOCK_RULE,

            "recovery":
                LOSS_OF_LOCK_RECOVERY,

            "switch_penalty":
                SWITCH_PENALTY_RULE,
        },

        "acceptance_preregistration": {
            "posterior_probability_coverage":
                (
                    "selected_mass + 1/2048 "
                    "must be >= requested q "
                    "when q is attainable "
                    "inside frozen support"
                ),

            "outside_support":
                (
                    "unattainable coverage "
                    "reported explicitly"
                ),

            "empirical_future_containment":
                (
                    "reported separately; "
                    "not controller input and "
                    "not used for post-hoc tuning"
                ),
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

            "optical_link_started":
                False,

            "ADB_started":
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
            (
                "Block5.6 optical pointing "
                "gain -> received power -> "
                "SNR -> DPSK BER -> effective rate"
            ),
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
        "BLOCK 5.5 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 continuity            = PASS"
    )

    print(
        "coverage q                  = 90 / 95 / 97.5 / 99 % FROZEN"
    )

    print(
        "nominal q                   = 95 % FROZEN"
    )

    print(
        "posterior mass tolerance    = 1 / 2048 FROZEN"
    )

    print(
        "adaptive rule               = SMALLEST MASS-COVERING SET"
    )

    print(
        "Kmax                        = FULL CODEBOOK SIZE FROZEN"
    )

    print(
        "Kmax 16/32/64               = 16 / 32 / 64"
    )

    print(
        "hysteresis                  = PREVIOUS PRIMARY IF STILL IN SET"
    )

    print(
        "numeric hysteresis margin   = NONE"
    )

    print(
        "local neighbor sweep        = IMMEDIATE GRAPH NEIGHBORS"
    )

    print(
        "widened fallback            = 64 -> 32 -> 16"
    )

    print(
        "loss-of-lock recovery       = WIDENED + EXHAUSTIVE"
    )

    print(
        "switch penalty              = UNIT SWITCH EVENT"
    )

    print(
        "outside-support mass        = EXPLICIT / NOT RENORMALIZED"
    )

    print(
        "false coverage claim        = FORBIDDEN"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "development/formal tuning   = NO"
    )

    print(
        "optical link                = NOT STARTED"
    )

    print(
        "policy SHA256               =",
        policy_sha,
    )

    print(
        "implementation files        =",
        implementation_files,
    )

    print(
        "implementation SHA256       =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "Block5.5 = COMPLETE / FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.6"
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
        "BLOCK 5.5 PART 2/2 = BLOCKED"
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

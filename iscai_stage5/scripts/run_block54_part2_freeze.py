from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.beam_baselines import (
    CONTROLLER_BASELINE_NAMES,
    EVALUATOR_ONLY_BASELINE_NAMES,
    FIXED_TOP_K_RANKING,
    FIXED_TOP_K_VALUES,
    FROZEN_BASELINE_NAMES,
    exhaustive_sweep,
    frozen_baseline_registry,
    oracle_best_gain_beam,
)

from iscai_stage5.beam_baseline_temporal import (
    ORACLE_ACCESS_POLICY,
    PERSISTENCE_INITIALIZATION,
    PERSISTENCE_RESET,
    PERSISTENCE_STATE_SEMANTICS,
    PERSISTENCE_UPDATE,
    REALIZED_RECEIVER_AZIMUTH_POLICY,
    fixed_probability_baseline_suite,
    frozen_temporal_baseline_contract,
    initial_persistence_state,
    persistence_step,
    validate_controller_selection,
    validate_evaluator_oracle,
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
      "block54_part1_beam_baselines.json"
)

BLOCK53 = (
    STAGE5
    / "reports/"
      "block53_codebook_probability.json"
)

CODEBOOK_POLICY = (
    STAGE5
    / "configs/"
      "beam_codebook_policy.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "beam_baseline_policy.json"
)

REPORT = (
    STAGE5
    / "reports/"
      "block54_beam_baselines.json"
)


EXPECTED_PART1_IMPLEMENTATION_SHA = (
    "1716f961014bf7e1069f1114112f0d79"
    "a0aea5bae7868c164928d0587f3b062f"
)

EXPECTED_BLOCK53_POLICY_SHA = (
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
        "STAGE 5 — BLOCK 5.4 PART 2/2"
    )
    print(
        "BASELINE POLICY + TEMPORAL CONTRACT FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK53,
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
            "Missing Block5.4 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Part1 / Block5.3 continuity
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
            "Block5.4 Part1 is not PASS."
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
            "Historical Block5.4 Part1 "
            "implementation SHA changed."
        ),
    )

    block53 = load_json(
        BLOCK53
    )

    require(
        block53.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.3 is no longer "
            "COMPLETE_FROZEN."
        ),
    )

    require(
        file_sha256(
            CODEBOOK_POLICY
        )
        ==
        EXPECTED_BLOCK53_POLICY_SHA,
        (
            "Frozen Block5.3 codebook "
            "policy SHA changed."
        ),
    )

    print(
        "Block5.4 Part1           = PASS"
    )

    print(
        "historical Part1 SHA     = PASS"
    )

    print(
        "Block5.3                 = COMPLETE / FROZEN"
    )

    print(
        "codebook policy SHA      = PASS"
    )

    # ========================================================
    # B. Registry freeze
    # ========================================================

    print()
    print(
        "===== B. BASELINE REGISTRY FREEZE ====="
    )

    require(
        FROZEN_BASELINE_NAMES
        ==
        (
            "exhaustive_sweep",
            "previous_beam_persistence",
            "geometry_nearest",
            "fixed_top1_probability",
            "fixed_top3_probability",
            "fixed_top5_probability",
            "oracle_best_gain_eval_only",
        ),
        (
            "Baseline registry changed."
        ),
    )

    require(
        FIXED_TOP_K_VALUES
        ==
        (
            1,
            3,
            5,
        ),
        (
            "Fixed K registry changed."
        ),
    )

    registry = (
        frozen_baseline_registry()
    )

    temporal = (
        frozen_temporal_baseline_contract()
    )

    require(
        registry[
            "adaptive_TopK_included"
        ]
        is False,
        (
            "Adaptive Top-K leaked into Block5.4."
        ),
    )

    require(
        temporal[
            "adaptive_TopK"
        ]
        is False,
        (
            "Temporal contract started "
            "adaptive Top-K."
        ),
    )

    print(
        "controller baselines      =",
        list(
            CONTROLLER_BASELINE_NAMES
        ),
    )

    print(
        "evaluator-only baselines  =",
        list(
            EVALUATOR_ONLY_BASELINE_NAMES
        ),
    )

    print(
        "fixed K                   = 1 / 3 / 5"
    )

    print(
        "fixed Top-K ranking       = P_b"
    )

    print(
        "adaptive Top-K            = NOT STARTED"
    )

    # ========================================================
    # C. Temporal persistence freeze
    # ========================================================

    print()
    print(
        "===== C. PERSISTENCE TEMPORAL CONTRACT ====="
    )

    codebook = (
        build_frozen_directional_codebook(
            32
        )
    )

    first_target = (
        codebook.cells[
            20
        ].center_azimuth_rad
    )

    state = (
        initial_persistence_state()
    )

    first, state = persistence_step(
        state=(
            state
        ),

        causal_predicted_azimuth_rad=(
            first_target
        ),

        codebook=(
            codebook
        ),
    )

    require(
        first.beam_indices
        ==
        (
            20,
        ),
        (
            "Persistence initialization changed."
        ),
    )

    require(
        first.fallback_used
        is True,
        (
            "First persistence step must "
            "use causal geometry fallback."
        ),
    )

    second_target = (
        codebook.cells[
            4
        ].center_azimuth_rad
    )

    second, state = persistence_step(
        state=(
            state
        ),

        causal_predicted_azimuth_rad=(
            second_target
        ),

        codebook=(
            codebook
        ),
    )

    require(
        second.beam_indices
        ==
        (
            20,
        ),
        (
            "Persistence did not reuse "
            "previous selected beam."
        ),
    )

    require(
        second.fallback_used
        is False,
        (
            "Later persistence step "
            "unexpectedly used fallback."
        ),
    )

    require(
        state.step_count
        ==
        2,
        (
            "Persistence step count changed."
        ),
    )

    print(
        "state                     = PREVIOUS SELECTED BEAM"
    )

    print(
        "first-step initialization = CAUSAL GEOMETRY NEAREST"
    )

    print(
        "later-step policy         = REUSE PREVIOUS BEAM"
    )

    print(
        "state update              = SELECTED BEAM -> NEXT STATE"
    )

    print(
        "reset semantics           = CLEAR PREVIOUS STATE"
    )

    print(
        "future receiver truth     = NO"
    )

    # ========================================================
    # D. Fixed Top-K comparison suite
    # ========================================================

    print()
    print(
        "===== D. FIXED TOP-K COMPARISON CONTRACT ====="
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
    ] = 0.25

    masses[
        17
    ] = 0.15

    masses[
        14
    ] = 0.05

    masses[
        18
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

    suite = (
        fixed_probability_baseline_suite(
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
            suite.keys()
        )
        ==
        (
            1,
            3,
            5,
        ),
        (
            "Fixed comparison suite changed."
        ),
    )

    for k in (
        1,
        3,
        5,
    ):

        require(
            suite[
                k
            ].probing_count
            ==
            k,
            (
                f"Fixed Top-{k} probing "
                "count changed."
            ),
        )

    print(
        "Top-1                    = FROZEN"
    )

    print(
        "Top-3                    = FROZEN"
    )

    print(
        "Top-5                    = FROZEN"
    )

    print(
        "ranking                  = DESCENDING P_b"
    )

    print(
        "tie-break                = LOWER BEAM INDEX"
    )

    print(
        "adaptive K               = NO"
    )

    # ========================================================
    # E. Oracle isolation freeze
    # ========================================================

    print()
    print(
        "===== E. EVALUATION-ONLY ORACLE CONTRACT ====="
    )

    oracle = (
        oracle_best_gain_beam(
            realized_azimuth_rad=(
                codebook.cells[
                    9
                ].center_azimuth_rad
            ),

            codebook=(
                codebook
            ),
        )
    )

    validate_evaluator_oracle(
        selection=(
            oracle
        ),

        codebook=(
            codebook
        ),
    )

    blocked_controller = False

    try:

        validate_controller_selection(
            selection=(
                oracle
            ),

            codebook=(
                codebook
            ),
        )

    except ValueError:

        blocked_controller = True

    require(
        blocked_controller,
        (
            "Oracle was not blocked from "
            "controller-visible state."
        ),
    )

    require(
        ORACLE_ACCESS_POLICY
        ==
        (
            "evaluator_only_after_"
            "controller_decision"
        ),
        (
            "Oracle access policy changed."
        ),
    )

    require(
        REALIZED_RECEIVER_AZIMUTH_POLICY
        ==
        (
            "evaluation_only_never_"
            "controller_input"
        ),
        (
            "Realized receiver azimuth "
            "policy changed."
        ),
    )

    print(
        "oracle input              = REALIZED RECEIVER AZIMUTH"
    )

    print(
        "oracle timing             = AFTER CONTROLLER DECISION"
    )

    print(
        "oracle controller access  = BLOCKED PASS"
    )

    print(
        "WOMD measured beam label  = NO"
    )

    print(
        "oracle label type         = GEOMETRY/GAIN-DERIVED"
    )

    # ========================================================
    # F. Leakage and scope
    # ========================================================

    print()
    print(
        "===== F. SCOPE / LEAKAGE ====="
    )

    controller_api = (
        persistence_step
    )

    signature = str(
        inspect.signature(
            controller_api
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
                f"Forbidden persistence API "
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
        "adaptive Top-K           = NOT STARTED"
    )

    print(
        "blockage-aware baseline  = OPTIONAL / NOT STARTED"
    )

    print(
        "optical link evaluation  = NOT STARTED"
    )

    print(
        "ADB                      = NOT STARTED"
    )

    # ========================================================
    # G. Freeze policy
    # ========================================================

    policy_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.4",

        "status":
            "FROZEN",

        "controller_baselines":
            list(
                CONTROLLER_BASELINE_NAMES
            ),

        "evaluation_only_baselines":
            list(
                EVALUATOR_ONLY_BASELINE_NAMES
            ),

        "exhaustive_sweep": {
            "beam_set":
                "all_codebook_beams",

            "probing_count":
                "beam_count",
        },

        "geometry_nearest": {
            "input":
                "causal_predicted_receiver_azimuth",

            "outside_support":
                (
                    "nearest_edge_beam_plus_"
                    "explicit_outside_support_flag"
                ),

            "future_truth":
                False,
        },

        "fixed_probability_baselines": {
            "K":
                [
                    1,
                    3,
                    5,
                ],

            "ranking":
                FIXED_TOP_K_RANKING,

            "tie_break":
                "lower_beam_index",

            "adaptive":
                False,
        },

        "previous_beam_persistence": {
            "state":
                PERSISTENCE_STATE_SEMANTICS,

            "initialization":
                PERSISTENCE_INITIALIZATION,

            "update":
                PERSISTENCE_UPDATE,

            "reset":
                PERSISTENCE_RESET,

            "future_truth":
                False,
        },

        "oracle_best_gain": {
            "access":
                ORACLE_ACCESS_POLICY,

            "realized_receiver_azimuth":
                REALIZED_RECEIVER_AZIMUTH_POLICY,

            "controller_access":
                False,

            "evaluation_only":
                True,

            "beam_label_type":
                "geometry_gain_derived",

            "WOMD_measured_beam_label":
                False,

            "real_communication_measurement":
                False,
        },

        "blockage_aware": {
            "status":
                "OPTIONAL_NOT_IMPLEMENTED_IN_BLOCK5.4",
        },

        "scope": {
            "formal_N120_read":
                False,

            "Stage4_inference":
                False,

            "development_tuning":
                False,

            "adaptive_TopK_started":
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
    # H. Final report
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
            "5.4",

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

        "baseline_registry": {
            "controller":
                list(
                    CONTROLLER_BASELINE_NAMES
                ),

            "evaluation_only":
                list(
                    EVALUATOR_ONLY_BASELINE_NAMES
                ),

            "fixed_K":
                [
                    1,
                    3,
                    5,
                ],

            "adaptive_TopK":
                False,
        },

        "temporal_contract": {
            "persistence_state":
                PERSISTENCE_STATE_SEMANTICS,

            "initialization":
                PERSISTENCE_INITIALIZATION,

            "update":
                PERSISTENCE_UPDATE,

            "reset":
                PERSISTENCE_RESET,

            "causal":
                True,
        },

        "oracle_contract": {
            "evaluation_only":
                True,

            "controller_access":
                False,

            "realized_receiver_azimuth":
                "evaluator_only",

            "WOMD_measured_beam_label":
                False,

            "real_communication_measurement":
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

            "adaptive_TopK_started":
                False,

            "blockage_aware_started":
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
            "Block5.5 adaptive Top-K probabilistic controller",
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
        "BLOCK 5.4 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 continuity             = PASS"
    )

    print(
        "baseline registry            = FROZEN"
    )

    print(
        "exhaustive sweep             = FROZEN"
    )

    print(
        "geometry nearest             = FROZEN / CAUSAL"
    )

    print(
        "previous-beam persistence    = FROZEN / CAUSAL"
    )

    print(
        "persistence initialization   = GEOMETRY NEAREST"
    )

    print(
        "persistence state update     = SELECTED BEAM"
    )

    print(
        "fixed Top-1 / 3 / 5          = FROZEN"
    )

    print(
        "fixed Top-K ranking          = P_b"
    )

    print(
        "oracle                      = EVALUATION ONLY"
    )

    print(
        "oracle controller access    = NO"
    )

    print(
        "WOMD measured beam labels   = NO"
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
        "adaptive Top-K              = NOT STARTED"
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
        "Block5.4 = COMPLETE / FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.5"
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
        "BLOCK 5.4 PART 2/2 = BLOCKED"
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

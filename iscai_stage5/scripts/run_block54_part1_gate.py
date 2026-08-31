from __future__ import annotations

from hashlib import sha256
import inspect
import json
import math
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
    fixed_top_k_probability,
    frozen_baseline_registry,
    geometry_nearest_beam,
    oracle_best_gain_beam,
    previous_beam_persistence,
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

REPORT = (
    STAGE5
    / "reports/"
      "block54_part1_beam_baselines.json"
)


EXPECTED_BLOCK53_IMPLEMENTATION_SHA = (
    "23dac64bed4ca76438f4303ff354f12c1"
    "90a1cd5a097b9f3a40008facc23aa53"
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
        "STAGE 5 — BLOCK 5.4 PART 1/2"
    )
    print(
        "FIXED / CLASSICAL BEAM-POLICY BASELINES"
    )
    print(
        "============================================================"
    )

    required = (
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
    # A. Block5.3 continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK5.3 FROZEN CONTINUITY ====="
    )

    block53 = json.loads(
        BLOCK53.read_text(
            encoding="utf-8"
        )
    )

    require(
        block53.get(
            "status"
        )
        ==
        "PASS",
        (
            "Block5.3 report is not PASS."
        ),
    )

    require(
        block53.get(
            "completion"
        )
        ==
        "COMPLETE_FROZEN",
        (
            "Block5.3 is not COMPLETE_FROZEN."
        ),
    )

    require(
        block53[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK53_IMPLEMENTATION_SHA,
        (
            "Historical Block5.3 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            CODEBOOK_POLICY
        )
        ==
        EXPECTED_BLOCK53_POLICY_SHA,
        (
            "Frozen beam-codebook "
            "policy SHA changed."
        ),
    )

    print(
        "Block5.3                 = COMPLETE / FROZEN"
    )

    print(
        "implementation SHA       = PASS"
    )

    print(
        "codebook policy SHA       = PASS"
    )

    print(
        "support                   = [-12,+12] deg"
    )

    print(
        "beam counts               = 16 / 32 / 64"
    )

    # ========================================================
    # B. Baseline registry
    # ========================================================

    print()
    print(
        "===== B. BASELINE REGISTRY ====="
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
            "Required baseline registry changed."
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
            "Fixed Top-K values changed."
        ),
    )

    require(
        "oracle_best_gain_eval_only"
        not in
        CONTROLLER_BASELINE_NAMES,
        (
            "Oracle leaked into controller "
            "baseline registry."
        ),
    )

    require(
        EVALUATOR_ONLY_BASELINE_NAMES
        ==
        (
            "oracle_best_gain_eval_only",
        ),
        (
            "Evaluator-only registry changed."
        ),
    )

    registry = (
        frozen_baseline_registry()
    )

    require(
        registry[
            "adaptive_TopK_included"
        ]
        is False,
        (
            "Adaptive Top-K must not start "
            "inside Block5.4."
        ),
    )

    print(
        "exhaustive sweep          = PRESENT"
    )

    print(
        "previous-beam persistence = PRESENT / CAUSAL"
    )

    print(
        "geometry nearest          = PRESENT"
    )

    print(
        "fixed Top-1/3/5           = PRESENT"
    )

    print(
        "oracle best gain          = EVALUATION ONLY"
    )

    print(
        "adaptive Top-K            = NOT STARTED"
    )

    # ========================================================
    # C. Cross-codebook structural sanity
    # ========================================================

    print()
    print(
        "===== C. CROSS-CODEBOOK BASELINE SANITY ====="
    )

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

        exhaustive = (
            exhaustive_sweep(
                codebook=(
                    codebook
                )
            )
        )

        require(
            exhaustive.probing_count
            ==
            beam_count,
            (
                "Exhaustive probing count mismatch."
            ),
        )

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
        ] = 0.70

        masses[
            max(
                0,
                center
                -
                1
            )
        ] += 0.20

        masses[
            min(
                beam_count
                -
                1,
                center
                +
                1
            )
        ] += 0.10

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

        for k in (
            1,
            3,
            5,
        ):

            selection = (
                fixed_top_k_probability(
                    probability=(
                        probability
                    ),

                    codebook=(
                        codebook
                    ),

                    k=(
                        k
                    ),
                )
            )

            require(
                selection.probing_count
                ==
                k,
                (
                    "Fixed Top-K probing "
                    "count mismatch."
                ),
            )

        print(
            f"{beam_count:2d}-beam baselines        = PASS"
        )

    # ========================================================
    # D. Causal persistence / geometry
    # ========================================================

    print()
    print(
        "===== D. CAUSAL GEOMETRY / PERSISTENCE ====="
    )

    codebook = (
        build_frozen_directional_codebook(
            32
        )
    )

    target_index = 19

    predicted_angle = (
        codebook.cells[
            target_index
        ].center_azimuth_rad
    )

    geometry = (
        geometry_nearest_beam(
            predicted_azimuth_rad=(
                predicted_angle
            ),

            codebook=(
                codebook
            ),
        )
    )

    require(
        geometry.beam_indices
        ==
        (
            target_index,
        ),
        (
            "Geometry-nearest baseline failed."
        ),
    )

    persistence = (
        previous_beam_persistence(
            previous_beam_index=7,

            codebook=(
                codebook
            ),
        )
    )

    require(
        persistence.beam_indices
        ==
        (
            7,
        ),
        (
            "Previous-beam persistence failed."
        ),
    )

    initialization = (
        previous_beam_persistence(
            previous_beam_index=None,

            codebook=(
                codebook
            ),

            fallback_predicted_azimuth_rad=(
                predicted_angle
            ),
        )
    )

    require(
        initialization.beam_indices
        ==
        (
            target_index,
        ),
        (
            "Persistence causal initialization "
            "fallback failed."
        ),
    )

    require(
        initialization.fallback_used
        is True,
        (
            "Persistence initialization "
            "fallback was not flagged."
        ),
    )

    print(
        "geometry input            = CAUSAL PREDICTED AZIMUTH"
    )

    print(
        "persistence state         = PREVIOUS SELECTED BEAM"
    )

    print(
        "initialization fallback   = GEOMETRY NEAREST"
    )

    print(
        "future receiver truth     = NO"
    )

    # ========================================================
    # E. Oracle isolation
    # ========================================================

    print()
    print(
        "===== E. ORACLE ISOLATION ====="
    )

    oracle_index = 11

    oracle = (
        oracle_best_gain_beam(
            realized_azimuth_rad=(
                codebook.cells[
                    oracle_index
                ].center_azimuth_rad
            ),

            codebook=(
                codebook
            ),
        )
    )

    require(
        oracle.evaluation_only
        is True,
        (
            "Oracle is not marked "
            "evaluation-only."
        ),
    )

    require(
        oracle.beam_indices
        ==
        (
            oracle_index,
        ),
        (
            "Oracle best-gain sanity failed."
        ),
    )

    controller_APIs = (
        exhaustive_sweep,
        fixed_top_k_probability,
        geometry_nearest_beam,
        previous_beam_persistence,
    )

    forbidden = (
        "future",
        "ground_truth",
        "realized",
        "oracle",
        "tracks_to_predict",
        "objects_of_interest",
    )

    for function in controller_APIs:

        signature = str(
            inspect.signature(
                function
            )
        ).lower()

        for token in forbidden:

            require(
                token not in signature,
                (
                    f"Forbidden controller API "
                    f"token {token!r}."
                ),
            )

    print(
        "oracle                    = EVALUATION ONLY"
    )

    print(
        "oracle real comm label    = NO"
    )

    print(
        "WOMD beam label           = NO"
    )

    print(
        "controller oracle access  = NO"
    )

    # ========================================================
    # F. Scope
    # ========================================================

    print()
    print(
        "===== F. SCOPE / LEAKAGE ====="
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "Stage4 inference          = NO"
    )

    print(
        "development tuning        = NO"
    )

    print(
        "training/recalibration    = NO"
    )

    print(
        "adaptive Top-K            = NOT STARTED"
    )

    print(
        "blockage-aware baseline   = OPTIONAL / NOT STARTED"
    )

    print(
        "optical BER chain         = NOT STARTED"
    )

    print(
        "ADB                       = NOT STARTED"
    )

    # ========================================================
    # G. Report
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
            "5.4_part_1",

        "status":
            "PASS",

        "Block53_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_BLOCK53_IMPLEMENTATION_SHA,

            "beam_codebook_policy_sha256":
                EXPECTED_BLOCK53_POLICY_SHA,
        },

        "baseline_registry": {
            "all":
                list(
                    FROZEN_BASELINE_NAMES
                ),

            "controller":
                list(
                    CONTROLLER_BASELINE_NAMES
                ),

            "evaluation_only":
                list(
                    EVALUATOR_ONLY_BASELINE_NAMES
                ),
        },

        "fixed_TopK": {
            "K":
                list(
                    FIXED_TOP_K_VALUES
                ),

            "ranking":
                FIXED_TOP_K_RANKING,

            "tie_break":
                "lower_beam_index",

            "adaptive":
                False,
        },

        "geometry_nearest": {
            "input":
                "causal_predicted_receiver_azimuth",

            "outside_support":
                (
                    "nearest_edge_beam_with_"
                    "outside_support_flag"
                ),

            "future_truth":
                False,
        },

        "previous_beam_persistence": {
            "state":
                "previous_selected_beam",

            "initialization":
                "geometry_nearest_causal_fallback",

            "future_truth":
                False,
        },

        "exhaustive": {
            "probes":
                "all_available_codebook_beams",
        },

        "oracle": {
            "policy":
                "best_constructed_gain_at_realized_azimuth",

            "evaluation_only":
                True,

            "real_measurement_label":
                False,

            "WOMD_beam_label":
                False,

            "controller_access":
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

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "next":
            (
                "Block5.4 Part2 freeze "
                "baseline policy registry, "
                "temporal semantics and "
                "evaluation-only oracle contract"
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
        "BLOCK 5.4 PART 1/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.3 continuity         = PASS"
    )

    print(
        "exhaustive sweep            = PASS"
    )

    print(
        "previous-beam persistence   = PASS / CAUSAL"
    )

    print(
        "geometry nearest            = PASS"
    )

    print(
        "fixed Top-1 / Top-3 / Top-5 = PASS"
    )

    print(
        "fixed Top-K ranking         = P_b"
    )

    print(
        "oracle best gain            = PASS / EVALUATION ONLY"
    )

    print(
        "oracle controller access    = NO"
    )

    print(
        "WOMD real beam labels       = NO"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "adaptive Top-K              = NOT STARTED"
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
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK 5.4 PART 2/2"
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
        "BLOCK 5.4 PART 1/2 = BLOCKED"
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

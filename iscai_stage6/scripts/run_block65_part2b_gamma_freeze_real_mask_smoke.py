from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

SCENARIO_ID = "b85e1bd6cc8e74c0"

N_MC = 8192
GAMMA = 0.5
THRESHOLD_K = 4096

GRID_SHAPE = (
    501,
    301,
)

EXPECTED_UNION_BY_HORIZON = (
    933,
    440,
    0,
    0,
)

EXPECTED_UNION_TOTAL = 1373

EXPECTED_FALLBACK_BOX_INDICES = (
    1,
    3,
    8,
)


# ============================================================
# Frozen evidence
# ============================================================

PART1_MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "predictive_mask.py"
)

PART1_CONTRACT = (
    S6
    / "configs"
    / "block65_part1_parameterized_mask_contract.json"
)

PART1_REPORT = (
    S6
    / "reports"
    / "block65_part1_parameterized_mask_kernel.json"
)

PREREG = (
    S6
    / "configs"
    / "block65_class_agnostic_gamma_preregistration.json"
)

CURVE = (
    S6
    / "artifacts"
    / "block65"
    / "block65_class_agnostic_gamma_exhaustive_curve.npz"
)

PART2A_REPORT = (
    S6
    / "reports"
    / "block65_part2a_gamma_curve.json"
)

BLOCK64_NUMERIC = (
    S6
    / "configs"
    / "block64_mc_grid_numeric_freeze.json"
)

BLOCK64_HANDOFF = (
    S6
    / "artifacts"
    / "block64"
    / "block64_to_block65_handoff.json"
)

COUNTS = (
    S6
    / "artifacts"
    / "block64"
    / "block64_part2b1_real_mc_convergence_counts.npz"
)

CONVERGENCE_REPORT = (
    S6
    / "reports"
    / "block64_part2b1_real_mc_convergence.json"
)

POSTERIOR = (
    S6
    / "artifacts"
    / "block63"
    / "identity_safe_development_gaussian_posterior.jsonl"
)

MOTION = (
    ROOT
    / "data"
    / "paired_womd_lidar_v1_3_0"
    / "validation"
    / "motion"
    / "paired-from-validation.tfrecord-00000-of-00150"
)


# ============================================================
# New Block6.5 Part2B outputs
# ============================================================

GAMMA_FREEZE = (
    S6
    / "configs"
    / "block65_class_agnostic_gamma_freeze.json"
)

ACTOR_MASK_ARTIFACT = (
    S6
    / "artifacts"
    / "block65"
    / "block65_gamma0p5_real_actor_masks.npy"
)

UNION_MASK_ARTIFACT = (
    S6
    / "artifacts"
    / "block65"
    / "block65_gamma0p5_real_union_mask.npy"
)

REPORT = (
    S6
    / "reports"
    / "block65_part2b_gamma_freeze_real_mask_smoke.json"
)

FREEZE_MANIFEST = (
    S6
    / "artifacts"
    / "block65"
    / "block65_class_agnostic_gamma_freeze_manifest.json"
)

HANDOFF = (
    S6
    / "artifacts"
    / "block65"
    / "block65_to_block66_handoff.json"
)


EXPECTED = {
    "part1_module":
        (
            "239611ddb1cc21b9ddd46962eb561e58e"
            "0d1474be58878f9c79c256ba1940575"
        ),

    "part1_contract":
        (
            "8b447f2ea141064a940eb098de3c0632"
            "ad4e960e89180bb9b0deab9547e13870"
        ),

    "part1_report":
        (
            "a8bed7b90b25846cc3e3cbd72bc275c1"
            "b201f8e5e7ef0f32d456abe4314a0c8d"
        ),

    "prereg":
        (
            "bec7749511d37b01a39967864cfb3259"
            "f0c506d16f3b825d3d33d06b8819f02d"
        ),

    "curve":
        (
            "b235b2fc6ef0cc261d57fb4cf57e899c"
            "3108868b52f15bdde511de0e2ce13e3c"
        ),

    "part2a_report":
        (
            "93948ca4c0c14f2503b503426432d3bd"
            "df9a201de2e5a51b9db2f441c2e5ba3f"
        ),

    "block64_numeric":
        (
            "993c4248a902e7dff3a4383ac343e722"
            "2cc43ef9b9372ed2efd0ee08d6d20a73"
        ),

    "block64_handoff":
        (
            "50c914e017d8d2aaa487c75077b49098"
            "c3ae5f2165b3048834b3a4bcc17719ae"
        ),

    "counts":
        (
            "ee536151df1b49294ebdeda598d747340"
            "83dc90769fd3d70dc8a058142701b7a"
        ),

    "convergence_report":
        (
            "57f23aaa55e31967653528157f2d7703f"
            "c10b632d353e2b8e25c53c379bc4d1f"
        ),

    "posterior":
        (
            "318b22dfc3788195a1432169075d1c865"
            "4f0d183337711918db2303b5668f5ea"
        ),
}


class ControlledBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(condition):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
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


def canonical_bytes(
    payload: Any,
) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def write_json_exact_or_create(
    path: Path,
    payload: Any,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = canonical_bytes(
        payload
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            expected,
            (
                "Existing JSON is not an "
                f"exact rerun match: {path}"
            ),
        )

    else:
        path.write_bytes(
            expected
        )

    return sha256_file(
        path
    )


def save_npy_exact_or_create(
    path: Path,
    array: np.ndarray,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    value = np.asarray(
        array
    )

    if path.exists():
        existing = np.load(
            path,
            allow_pickle=False,
        )

        require(
            existing.dtype
            ==
            value.dtype,
            (
                f"Existing NPY dtype mismatch: {path}"
            ),
        )

        require(
            existing.shape
            ==
            value.shape,
            (
                f"Existing NPY shape mismatch: {path}"
            ),
        )

        require(
            np.array_equal(
                existing,
                value,
            ),
            (
                f"Existing NPY data mismatch: {path}"
            ),
        )

    else:
        np.save(
            path,
            value,
            allow_pickle=False,
        )

    return sha256_file(
        path
    )


def run_regression() -> dict[str, Any]:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "tests":
            (
                int(
                    match.group(1)
                )
                if match
                else None
            ),

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -30:
                ]
            ),
    }


def tuple3(
    value,
):
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        array.shape
        ==
        (
            3,
        ),
        (
            f"Expected Vec3; got {array.shape}."
        ),
    )

    return tuple(
        float(x)
        for x in array
    )


def tuple4x3(
    value,
):
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        array.shape
        ==
        (
            4,
            3,
        ),
        (
            f"Expected [4,3]; got {array.shape}."
        ),
    )

    return tuple(
        tuple(
            float(x)
            for x in row
        )
        for row in array
    )


try:
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.5 PART2B — CLASS-AGNOSTIC GAMMA "
        "FREEZE + REAL MASK SMOKE"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # A. Immutable upstream seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE UPSTREAM SEAL ====="
    )

    for path, expected, label in (
        (
            PART1_MODULE,
            EXPECTED[
                "part1_module"
            ],
            "Part1 predictive-mask module",
        ),
        (
            PART1_CONTRACT,
            EXPECTED[
                "part1_contract"
            ],
            "Part1 contract",
        ),
        (
            PART1_REPORT,
            EXPECTED[
                "part1_report"
            ],
            "Part1 report",
        ),
        (
            PREREG,
            EXPECTED[
                "prereg"
            ],
            "gamma preregistration",
        ),
        (
            CURVE,
            EXPECTED[
                "curve"
            ],
            "exhaustive gamma curve",
        ),
        (
            PART2A_REPORT,
            EXPECTED[
                "part2a_report"
            ],
            "Part2A report",
        ),
        (
            BLOCK64_NUMERIC,
            EXPECTED[
                "block64_numeric"
            ],
            "Block6.4 numeric freeze",
        ),
        (
            BLOCK64_HANDOFF,
            EXPECTED[
                "block64_handoff"
            ],
            "Block6.4 handoff",
        ),
        (
            COUNTS,
            EXPECTED[
                "counts"
            ],
            "frozen N=8192 counts",
        ),
        (
            CONVERGENCE_REPORT,
            EXPECTED[
                "convergence_report"
            ],
            "convergence report",
        ),
        (
            POSTERIOR,
            EXPECTED[
                "posterior"
            ],
            "identity-safe posterior",
        ),
    ):
        require(
            path.is_file(),
            f"Missing {label}: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:30s} = EXACT PASS"
        )

    # ========================================================
    # B. Preregistration / non-posthoc gate
    # ========================================================

    print()
    print(
        "===== B. GAMMA PREREGISTRATION GATE ====="
    )

    prereg = json.loads(
        PREREG.read_text(
            encoding="utf-8"
        )
    )

    gamma_report = json.loads(
        PART2A_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        prereg[
            "status"
        ]
        ==
        "PREREGISTERED_BEFORE_GAMMA_CURVE_EXTRACTION",
        (
            "Gamma preregistration status changed."
        ),
    )

    require(
        float(
            prereg[
                "selected_gamma"
            ]
        )
        ==
        GAMMA,
        (
            "Preregistered gamma changed."
        ),
    )

    require(
        prereg[
            "selection_rule_details"
        ][
            "development_metric_optimization"
        ]
        is False,
        (
            "Gamma was marked development-optimized."
        ),
    )

    require(
        prereg[
            "selection_rule_details"
        ][
            "posthoc_curve_selection"
        ]
        is False,
        (
            "Gamma was marked posthoc-selected."
        ),
    )

    require(
        gamma_report[
            "status"
        ]
        ==
        (
            "PASS_CLASS_AGNOSTIC_GAMMA_"
            "PREREGISTRATION_AND_CURVE"
        ),
        (
            "Part2A gamma report status changed."
        ),
    )

    require(
        gamma_report[
            "preregistration"
        ][
            "written_before_gamma_curve_counts_loaded"
        ]
        is True,
        (
            "Preregistration ordering not proven."
        ),
    )

    require(
        gamma_report[
            "optimality_validation"
        ][
            "selected_is_global_minimizer"
        ]
        is True,
        (
            "Preregistered gamma failed "
            "symmetric-loss validation."
        ),
    )

    require(
        int(
            gamma_report[
                "optimality_validation"
            ][
                "number_of_tied_minimizing_breakpoints"
            ]
        )
        ==
        2,
        (
            "Expected two tied minimizing breakpoints."
        ),
    )

    print(
        "gamma scope          = CLASS-AGNOSTIC BASELINE"
    )

    print(
        "gamma                = 0.5"
    )

    print(
        "threshold            = STRICT >"
    )

    print(
        "P_occ == gamma       = INACTIVE"
    )

    print(
        "selection            = PREREGISTERED / NOT POSTHOC"
    )

    print(
        "curve validation     = GLOBAL MINIMIZER PASS"
    )

    print(
        "tied minimizers      = 2 | prereg resolves choice"
    )

    # ========================================================
    # C. Freeze gamma BEFORE real-mask smoke in this runner
    # ========================================================

    print()
    print(
        "===== C. CLASS-AGNOSTIC GAMMA FREEZE ====="
    )

    gamma_freeze_payload = {
        "stage":
            6,

        "block":
            "6.5-Part2B",

        "status":
            "FROZEN_CLASS_AGNOSTIC_GAMMA",

        "scope":
            "class_agnostic_predictive_ADB_baseline_only",

        "gamma":
            GAMMA,

        "exact_MC_breakpoint": {
            "k":
                THRESHOLD_K,

            "N":
                N_MC,

            "fraction":
                "4096/8192",
        },

        "threshold_semantics":
            (
                "M_i_tau = "
                "1[P_occ_i(theta,r,tau) > 0.5]"
            ),

        "equal_to_gamma_is_active":
            False,

        "selection_basis":
            (
                "preregistered_symmetric_posterior_"
                "Bayes_MAP_rule"
            ),

        "selection_timing":
            "before_gamma_curve_extraction",

        "curve_role":
            "validation_and_sensitivity_only",

        "validation": {
            "global_symmetric_loss_minimizer":
                True,

            "tied_global_minimizers":
                2,

            "posthoc_selection":
                False,
        },

        "frozen_upstream_numerics": {
            "MC_N":
                N_MC,

            "MC_seed":
                20260821,

            "grid_shape":
                [
                    501,
                    301,
                ],

            "theta_deg":
                [
                    -25.0,
                    25.0,
                    0.1,
                ],

            "range_m":
                [
                    0.0,
                    150.0,
                    0.5,
                ],
        },

        "important_scope_boundary": {
            "class_aware_gamma_frozen":
                False,

            "vehicle_policy_frozen":
                False,

            "pedestrian_policy_frozen":
                False,

            "cyclist_policy_frozen":
                False,

            "class_margins_frozen":
                False,

            "dimming_floors_frozen":
                False,

            "temporal_smoothing_frozen":
                False,

            "actuation_rate_limits_frozen":
                False,
        },

        "formal_boundary": {
            "formal_population_used":
                False,

            "formal_outcomes_read":
                False,

            "formal_tuning":
                False,
        },
    }

    gamma_freeze_sha = (
        write_json_exact_or_create(
            GAMMA_FREEZE,
            gamma_freeze_payload,
        )
    )

    print(
        "gamma freeze =",
        GAMMA_FREEZE,
    )

    print(
        "gamma freeze SHA256 =",
        gamma_freeze_sha,
    )

    print(
        "class-aware gamma frozen = NO"
    )

    # ========================================================
    # D. Real causal association / fallback preservation
    # ========================================================

    print()
    print(
        "===== D. REAL CAUSAL PLAN / FALLBACK PRESERVATION ====="
    )

    from iscai_stage0.womd_proto_io import (
        read_first_scenario,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    from iscai_stage6.adb.deterministic_predictive import (
        DeterministicSharedMeanPrediction,
        build_deterministic_predictive_plan,
    )

    posterior_records = [
        json.loads(
            line
        )
        for line in
        POSTERIOR.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(
            posterior_records
        )
        ==
        27,
        (
            "Posterior record count changed."
        ),
    )

    deterministic_predictions = tuple(
        DeterministicSharedMeanPrediction(
            prediction_id=str(
                record[
                    "prediction_id"
                ]
            ),

            latest_position_H0_m=tuple3(
                record[
                    "latest_position_H0_m"
                ]
            ),

            mean_displacement_H0_m=tuple4x3(
                record[
                    "mean_displacement_H0_m"
                ]
            ),
        )
        for record in posterior_records
    )

    scenario = read_first_scenario(
        MOTION
    )

    require(
        str(
            scenario.scenario_id
        )
        ==
        SCENARIO_ID,
        (
            "Development scenario changed."
        ),
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    actor_boxes = (
        build_causal_adb_actor_boxes(
            scenario=scenario,
            adapted=adapted,
        )
    )

    real_plan = (
        build_deterministic_predictive_plan(
            deterministic_predictions,
            actor_boxes,
        )
    )

    require(
        len(
            real_plan.matched_forecasts
        )
        ==
        6,
        (
            "Frozen matched count changed."
        ),
    )

    require(
        tuple(
            int(x)
            for x in
            real_plan.reactive_fallback_box_indices
        )
        ==
        EXPECTED_FALLBACK_BOX_INDICES,
        (
            "Frozen reactive fallback "
            "box indices changed."
        ),
    )

    require(
        len(
            real_plan.unmatched_prediction_indices
        )
        ==
        21,
        (
            "Unmatched predictor count changed."
        ),
    )

    real_prediction_ids = tuple(
        str(
            item.prediction_id
        )
        for item in
        real_plan.matched_forecasts
    )

    real_classes = tuple(
        str(
            item.actor_object_type
        )
        for item in
        real_plan.matched_forecasts
    )

    print(
        "scenario              =",
        SCENARIO_ID,
    )

    print(
        "causal ADB boxes      =",
        len(
            actor_boxes
        ),
    )

    print(
        "matched actors        = 6"
    )

    print(
        "reactive fallback box indices =",
        EXPECTED_FALLBACK_BOX_INDICES,
    )

    print(
        "unmatched predictors  = 21"
    )

    print(
        "matched actor classes =",
        real_classes,
    )

    print(
        "future GT             = NO"
    )

    print(
        "perfect identity      = NO"
    )

    # ========================================================
    # E. Load frozen real N=8192 occupancy counts
    # ========================================================

    print()
    print(
        "===== E. REAL N=8192 OCCUPANCY BINDING ====="
    )

    convergence = json.loads(
        CONVERGENCE_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        convergence[
            "development_population"
        ][
            "formal_population_used"
        ]
        is False,
        (
            "Formal population entered smoke."
        ),
    )

    require(
        int(
            convergence[
                "MC_study"
            ][
                "minimum_sample_count"
            ]
        )
        ==
        2048,
        (
            "Unexpected Stage5 MC minimum."
        ),
    )

    convergence_ids = tuple(
        str(
            actor[
                "prediction_id"
            ]
        )
        for actor in
        convergence[
            "actors"
        ]
    )

    require(
        convergence_ids
        ==
        real_prediction_ids,
        (
            "Frozen count actor ordering no "
            "longer matches real causal plan."
        ),
    )

    data = np.load(
        COUNTS,
        allow_pickle=False,
    )

    frozen_counts = []

    for actor_index in range(
        6
    ):
        key = (
            f"actor_{actor_index:02d}_"
            f"counts_N{N_MC:05d}"
        )

        require(
            key in data.files,
            (
                f"Missing frozen occupancy "
                f"array {key}."
            ),
        )

        counts = np.asarray(
            data[
                key
            ],
            dtype=np.uint32,
        )

        require(
            counts.shape
            ==
            (
                4,
                *GRID_SHAPE,
            ),
            (
                f"Unexpected {key} shape: "
                f"{counts.shape}"
            ),
        )

        require(
            np.all(
                counts
                <=
                N_MC
            ),
            (
                f"{key} contains count > N_MC."
            ),
        )

        frozen_counts.append(
            counts
        )

        print(
            f"actor {actor_index} "
            f"| prediction={real_prediction_ids[actor_index]} "
            f"| positive P_occ cells="
            f"{int(np.count_nonzero(counts))}"
        )

    frozen_counts = np.stack(
        frozen_counts,
        axis=0,
    )

    require(
        frozen_counts.shape
        ==
        (
            6,
            4,
            *GRID_SHAPE,
        ),
        (
            "Stacked real occupancy shape changed."
        ),
    )

    # ========================================================
    # F. Public real predictive-mask smoke
    # ========================================================

    print()
    print(
        "===== F. PUBLIC CLASS-AGNOSTIC MASK SMOKE ====="
    )

    from iscai_stage6.adb.probabilistic_occupancy import (
        ActorOccupancyProbability,
    )

    from iscai_stage6.adb.predictive_mask import (
        build_class_agnostic_predictive_mask_plan,
    )

    occupancies = []

    for actor_index in range(
        6
    ):
        counts = frozen_counts[
            actor_index
        ]

        probability = (
            counts.astype(
                np.float64
            )
            /
            float(
                N_MC
            )
        )

        # 8192 is a power of two, therefore every
        # count/8192 value is exactly representable
        # in binary float64.
        require(
            np.array_equal(
                (
                    probability
                    *
                    float(
                        N_MC
                    )
                ).astype(
                    np.uint32
                ),
                counts,
            ),
            (
                "Count/probability exact binding "
                f"failed for actor {actor_index}."
            ),
        )

        occupancies.append(
            ActorOccupancyProbability(
                prediction_id=(
                    real_prediction_ids[
                        actor_index
                    ]
                ),

                sample_count=N_MC,

                horizons_s=(
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ),

                occupancy_counts=counts,

                occupancy_probability=(
                    probability
                ),
            )
        )

    mask_plan = (
        build_class_agnostic_predictive_mask_plan(
            occupancies,
            gamma=GAMMA,
            grid_shape=GRID_SHAPE,

            # Reactive fallback is deliberately
            # represented separately. Part1 already
            # froze its exact Block6.2 call-through.
            reactive_grid=None,
            reactive_fallback_regions=(),
        )
    )

    require(
        mask_plan.gamma
        ==
        GAMMA,
        (
            "Public plan gamma changed."
        ),
    )

    require(
        mask_plan.predictive_mask.shape
        ==
        (
            4,
            *GRID_SHAPE,
        ),
        (
            "Public predictive mask shape changed."
        ),
    )

    require(
        mask_plan.predictive_mask.dtype
        ==
        np.bool_,
        (
            "Public predictive mask is not bool."
        ),
    )

    require(
        mask_plan
        .reactive_fallback_illumination
        is None,
        (
            "Part2B unexpectedly composed "
            "reactive illumination."
        ),
    )

    require(
        mask_plan
        .final_illumination_composition_applied
        is False,
        (
            "Final intensity composition occurred "
            "before class-policy freeze."
        ),
    )

    public_actor_masks = np.stack(
        [
            np.asarray(
                item.mask,
                dtype=bool,
            )
            for item in
            mask_plan.actor_masks
        ],
        axis=0,
    )

    direct_actor_masks = (
        frozen_counts
        >
        THRESHOLD_K
    )

    require(
        np.array_equal(
            public_actor_masks,
            direct_actor_masks,
        ),
        (
            "Public per-actor mask differs "
            "from exact count >4096 threshold."
        ),
    )

    direct_union = np.any(
        direct_actor_masks,
        axis=0,
    )

    require(
        np.array_equal(
            mask_plan.predictive_mask,
            direct_union,
        ),
        (
            "Public multi-actor mask differs "
            "from exact binary union."
        ),
    )

    equality_cells = int(
        np.count_nonzero(
            frozen_counts
            ==
            THRESHOLD_K
        )
    )

    if equality_cells:
        require(
            not np.any(
                public_actor_masks[
                    frozen_counts
                    ==
                    THRESHOLD_K
                ]
            ),
            (
                "P_occ == 0.5 became active."
            ),
        )

    union_by_horizon = tuple(
        int(
            np.count_nonzero(
                mask_plan.predictive_mask[
                    horizon_index
                ]
            )
        )
        for horizon_index in range(
            4
        )
    )

    union_total = int(
        np.count_nonzero(
            mask_plan.predictive_mask
        )
    )

    require(
        union_by_horizon
        ==
        EXPECTED_UNION_BY_HORIZON,
        (
            "Real mask horizon counts changed: "
            f"{union_by_horizon}"
        ),
    )

    require(
        union_total
        ==
        EXPECTED_UNION_TOTAL,
        (
            "Real total predictive-mask "
            f"count changed: {union_total}"
        ),
    )

    actor_mask_cells = tuple(
        int(
            np.count_nonzero(
                public_actor_masks[
                    actor_index
                ]
            )
        )
        for actor_index in range(
            6
        )
    )

    print(
        "public per-actor parity      = EXACT PASS"
    )

    print(
        "public multi-actor parity    = EXACT PASS"
    )

    print(
        "P_occ == 0.5 cell count      =",
        equality_cells,
    )

    print(
        "strict equality handling     = PASS"
    )

    print(
        "actor mask cells             =",
        actor_mask_cells,
    )

    print(
        "union cells by horizon       =",
        union_by_horizon,
    )

    print(
        "union cells total            =",
        union_total,
    )

    print(
        "horizon 0.5s mask active     =",
        bool(
            union_by_horizon[
                2
            ]
        ),
    )

    print(
        "horizon 1.0s mask active     =",
        bool(
            union_by_horizon[
                3
            ]
        ),
    )

    print(
        "final illumination composition = NO"
    )

    # ========================================================
    # G. Persist deterministic real-mask artifacts
    # ========================================================

    print()
    print(
        "===== G. REAL MASK ARTIFACTS ====="
    )

    actor_mask_sha = (
        save_npy_exact_or_create(
            ACTOR_MASK_ARTIFACT,
            public_actor_masks,
        )
    )

    union_mask_sha = (
        save_npy_exact_or_create(
            UNION_MASK_ARTIFACT,
            np.asarray(
                mask_plan.predictive_mask,
                dtype=bool,
            ),
        )
    )

    print(
        "actor masks =",
        ACTOR_MASK_ARTIFACT,
    )

    print(
        "actor masks SHA256 =",
        actor_mask_sha,
    )

    print(
        "union mask =",
        UNION_MASK_ARTIFACT,
    )

    print(
        "union mask SHA256 =",
        union_mask_sha,
    )

    # ========================================================
    # H. Fresh regression
    # ========================================================

    print()
    print(
        "===== H. FRESH STAGE6 REGRESSION ====="
    )

    regression = run_regression()

    print(
        regression[
            "tail"
        ]
    )

    require(
        regression[
            "returncode"
        ]
        ==
        0,
        (
            "Stage6 regression failed."
        ),
    )

    require(
        regression[
            "tests"
        ]
        ==
        145,
        (
            "Expected 145 Stage6 tests, got "
            f"{regression['tests']}."
        ),
    )

    print(
        "fresh regression = PASS | 145"
    )

    # ========================================================
    # I. Part2B report
    # ========================================================

    print()
    print(
        "===== I. BLOCK6.5 PART2B REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.5-Part2B",

        "status":
            (
                "PASS_CLASS_AGNOSTIC_GAMMA_"
                "FROZEN_REAL_MASK_SMOKE"
            ),

        "gamma_freeze": {
            "path":
                str(
                    GAMMA_FREEZE
                ),

            "sha256":
                gamma_freeze_sha,

            "scope":
                "class_agnostic_baseline_only",

            "gamma":
                GAMMA,

            "strict_greater_than":
                True,

            "equal_to_gamma_inactive":
                True,

            "posthoc_selection":
                False,
        },

        "real_smoke": {
            "scenario_id":
                SCENARIO_ID,

            "formal_population":
                False,

            "matched_predictive_actors":
                6,

            "causal_ADB_boxes":
                int(
                    len(
                        actor_boxes
                    )
                ),

            "reactive_fallback_box_indices":
                list(
                    EXPECTED_FALLBACK_BOX_INDICES
                ),

            "unmatched_predictors":
                21,

            "actor_prediction_ids":
                list(
                    real_prediction_ids
                ),

            "actor_object_types":
                list(
                    real_classes
                ),

            "MC_N":
                N_MC,

            "new_MC_sampling":
                False,

            "grid_shape":
                list(
                    GRID_SHAPE
                ),

            "actor_mask_cells":
                list(
                    actor_mask_cells
                ),

            "union_mask_cells_by_horizon":
                list(
                    union_by_horizon
                ),

            "union_mask_cells_total":
                union_total,

            "count_equal_to_gamma_breakpoint_cells":
                equality_cells,
        },

        "public_API_parity": {
            "per_actor_threshold":
                "EXACT_PASS",

            "multi_actor_union":
                "EXACT_PASS",

            "direct_rule":
                "count > 4096",

            "predictive_formula":
                (
                    "1_minus_product_i_of_"
                    "1_minus_M_i_tau"
                ),
        },

        "reactive_fallback_boundary": {
            "fallback_actor_indices_preserved":
                True,

            "exact_Block6.2_callthrough_implementation":
                True,

            "real_fallback_map_synthesized_in_this_smoke":
                False,

            "reason":
                (
                    "predictive_binary_mask_and_"
                    "continuous_reactive_L_theta_r_"
                    "remain_separate_until_policy_"
                    "intensity_composition"
                ),

            "raised_cosine_modified":
                False,
        },

        "artifacts": {
            "actor_masks": {
                "path":
                    str(
                        ACTOR_MASK_ARTIFACT
                    ),

                "sha256":
                    actor_mask_sha,
            },

            "union_mask": {
                "path":
                    str(
                        UNION_MASK_ARTIFACT
                    ),

                "sha256":
                    union_mask_sha,
            },
        },

        "important_interpretation": {
            "zero_predictive_cells_at_0p5s":
                (
                    union_by_horizon[
                        2
                    ]
                    ==
                    0
                ),

            "zero_predictive_cells_at_1p0s":
                (
                    union_by_horizon[
                        3
                    ]
                    ==
                    0
                ),

            "treated_as_software_failure":
                False,

            "class_aware_controller_may_copy_gamma_0p5_without_new_preregistration":
                False,
        },

        "formal_boundary": {
            "formal_population_used":
                False,

            "formal_outcomes_read":
                False,

            "formal_tuning":
                False,
        },

        "scientific_execution": {
            "model_forward":
                False,

            "new_MC_sampling":
                False,

            "training":
                False,

            "recalibration":
                False,

            "source_modification":
                False,

            "final_intensity_composition":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                145,
        },

        "next":
            (
                "BLOCK6.6_CLASS_AWARE_DEVELOPMENT_"
                "COHORT_AND_POLICY_PREREGISTRATION"
            ),
    }

    report_sha = (
        write_json_exact_or_create(
            REPORT,
            report_payload,
        )
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    # ========================================================
    # J. Freeze manifest
    # ========================================================

    print()
    print(
        "===== J. BLOCK6.5 FREEZE MANIFEST ====="
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.5",

        "status":
            (
                "BLOCK65_CLASS_AGNOSTIC_"
                "MASK_BASELINE_FROZEN"
            ),

        "immutable_inputs": {
            str(
                PART1_MODULE
            ):
                EXPECTED[
                    "part1_module"
                ],

            str(
                PART1_CONTRACT
            ):
                EXPECTED[
                    "part1_contract"
                ],

            str(
                PART1_REPORT
            ):
                EXPECTED[
                    "part1_report"
                ],

            str(
                PREREG
            ):
                EXPECTED[
                    "prereg"
                ],

            str(
                CURVE
            ):
                EXPECTED[
                    "curve"
                ],

            str(
                PART2A_REPORT
            ):
                EXPECTED[
                    "part2a_report"
                ],

            str(
                BLOCK64_NUMERIC
            ):
                EXPECTED[
                    "block64_numeric"
                ],

            str(
                COUNTS
            ):
                EXPECTED[
                    "counts"
                ],
        },

        "gamma_freeze": {
            "path":
                str(
                    GAMMA_FREEZE
                ),

            "sha256":
                gamma_freeze_sha,
        },

        "real_mask_artifacts": {
            "actor_masks_sha256":
                actor_mask_sha,

            "union_mask_sha256":
                union_mask_sha,
        },

        "freeze_report": {
            "path":
                str(
                    REPORT
                ),

            "sha256":
                report_sha,
        },

        "scientific_lock": {
            "class_agnostic_gamma":
                GAMMA,

            "MC_N":
                N_MC,

            "MC_seed":
                20260821,

            "grid_shape":
                [
                    501,
                    301,
                ],

            "class_aware_policy":
                "NOT_FROZEN",

            "final_intensity_composition":
                "NOT_FROZEN",

            "formal_tuning":
                "FORBIDDEN",
        },
    }

    freeze_sha = (
        write_json_exact_or_create(
            FREEZE_MANIFEST,
            freeze_payload,
        )
    )

    print(
        "freeze manifest =",
        FREEZE_MANIFEST,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
    )

    # ========================================================
    # K. Block6.5 -> Block6.6 handoff
    # ========================================================

    print()
    print(
        "===== K. BLOCK6.5 → BLOCK6.6 HANDOFF ====="
    )

    handoff_payload = {
        "stage":
            6,

        "from_block":
            "6.5",

        "to_block":
            "6.6",

        "status":
            (
                "PASS_READY_FOR_BLOCK66_"
                "CLASS_AWARE_DEVELOPMENT"
            ),

        "frozen_class_agnostic_baseline": {
            "gamma":
                GAMMA,

            "threshold_semantics":
                "strict_P_occ_greater_than_gamma",

            "MC_N":
                N_MC,

            "MC_seed":
                20260821,

            "grid_shape":
                [
                    501,
                    301,
                ],

            "multi_actor_rule":
                (
                    "1_minus_product_i_of_"
                    "1_minus_M_i_tau"
                ),

            "reactive_unmatched_fallback":
                "frozen_Block6.2_continuous_L_theta_r",
        },

        "development_smoke": {
            "scenario_id":
                SCENARIO_ID,

            "matched_actor_count":
                6,

            "fallback_box_indices":
                list(
                    EXPECTED_FALLBACK_BOX_INDICES
                ),

            "matched_actor_object_types":
                list(
                    real_classes
                ),

            "union_mask_cells_by_horizon":
                list(
                    union_by_horizon
                ),

            "union_mask_cells_total":
                union_total,
        },

        "critical_next_boundary": {
            "class_aware_gamma_inherits_0p5_automatically":
                False,

            "class_specific_parameters_must_be_preregistered_before_formal":
                True,

            "development_cohort_must_cover": [
                "TYPE_VEHICLE",
                "TYPE_PEDESTRIAN",
                "TYPE_CYCLIST",
            ],

            "formal_population_may_be_used_for_parameter_selection":
                False,

            "formal_outcomes_may_be_read_before_class_policy_freeze":
                False,
        },

        "required_Block66_work": [
            (
                "construct_or_verify_nonformal_"
                "development_class_coverage"
            ),
            (
                "preregister_vehicle_policy_"
                "beyond_margin_only"
            ),
            (
                "preregister_pedestrian_visibility_"
                "preserving_policy"
            ),
            (
                "preregister_cyclist_visibility_"
                "preserving_policy"
            ),
            (
                "define_class_specific_confidence_"
                "thresholds_or_cost_rules"
            ),
            (
                "define_class_specific_dimming_"
                "floors_and_spatial_margins"
            ),
            (
                "preserve_reactive_fallback_and_"
                "PartA_raised_cosine"
            ),
        ],

        "upstream_freeze": {
            "gamma_freeze_sha256":
                gamma_freeze_sha,

            "report_sha256":
                report_sha,

            "freeze_manifest_sha256":
                freeze_sha,
        },
    }

    handoff_sha = (
        write_json_exact_or_create(
            HANDOFF,
            handoff_payload,
        )
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # L. Final
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.5 PART2B — FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "STATUS = "
        "PASS_BLOCK65_CLASS_AGNOSTIC_MASK_BASELINE_FROZEN"
    )

    print(
        "gamma scope              = CLASS-AGNOSTIC BASELINE ONLY"
    )

    print(
        "gamma                    = 0.5 | FROZEN"
    )

    print(
        "threshold                = STRICT P_occ > 0.5"
    )

    print(
        "posthoc selection        = NO"
    )

    print(
        "MC N / seed              = 8192 / 20260821"
    )

    print(
        "real matched actors      = 6"
    )

    print(
        "reactive fallback boxes  =",
        EXPECTED_FALLBACK_BOX_INDICES,
    )

    print(
        "real predictive union    =",
        union_by_horizon,
    )

    print(
        "real predictive cells    =",
        union_total,
    )

    print(
        "horizon 0.5s / 1.0s      = 0 / 0 MASK CELLS"
    )

    print(
        "treated as software error= NO"
    )

    print(
        "reactive L(theta,r) changed = NO"
    )

    print(
        "final intensity composition = NO"
    )

    print(
        "class-aware gamma frozen = NO"
    )

    print(
        "class-aware floors/margins= NO"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "formal outcomes read     = NO"
    )

    print(
        "new MC sampling          = NO"
    )

    print(
        "model forward            = NO"
    )

    print(
        "training/recalibration   = NO / NO"
    )

    print(
        "fresh Stage6 regression  = PASS | 145"
    )

    print(
        "gamma freeze SHA256      =",
        gamma_freeze_sha,
    )

    print(
        "actor mask SHA256        =",
        actor_mask_sha,
    )

    print(
        "union mask SHA256        =",
        union_mask_sha,
    )

    print(
        "report SHA256            =",
        report_sha,
    )

    print(
        "freeze SHA256            =",
        freeze_sha,
    )

    print(
        "handoff SHA256           =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.6 "
        "CLASS-AWARE DEVELOPMENT COHORT + "
        "POLICY PREREGISTRATION"
    )

    print(
        "terminal remains open = YES"
    )

    print(
        "=" * 72
    )


except BaseException as exc:
    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK6.5 PART2B — CONTROLLED BLOCK"
    )

    print(
        "=" * 72
    )

    print(
        "exception type =",
        type(
            exc
        ).__name__,
    )

    print(
        "exception      =",
        str(
            exc
        ),
    )

    print()
    traceback.print_exc(
        limit=18
    )

    print()
    print(
        "gamma freeze complete    = NO/INCOMPLETE"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "formal tuning            = NO"
    )

    print(
        "new MC sampling          = NO"
    )

    print(
        "source modification      = NO"
    )

    print(
        "terminal remains open    = YES"
    )

# Deliberately no sys.exit().

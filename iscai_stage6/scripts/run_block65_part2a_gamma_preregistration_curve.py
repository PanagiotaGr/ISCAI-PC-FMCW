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

CONVERGENCE_REPORT = (
    S6
    / "reports"
    / "block64_part2b1_real_mc_convergence.json"
)

COUNTS = (
    S6
    / "artifacts"
    / "block64"
    / "block64_part2b1_real_mc_convergence_counts.npz"
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

REPORT = (
    S6
    / "reports"
    / "block65_part2a_gamma_curve.json"
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

    "convergence_report":
        (
            "57f23aaa55e31967653528157f2d7703f"
            "c10b632d353e2b8e25c53c379bc4d1f"
        ),

    "counts":
        (
            "ee536151df1b49294ebdeda598d747340"
            "83dc90769fd3d70dc8a058142701b7a"
        ),
}


N_MC = 8192
SELECTED_K = N_MC // 2
SELECTED_GAMMA = 0.5

ACTOR_COUNT = 6
HORIZON_COUNT = 4
GRID_SHAPE = (
    501,
    301,
)


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

    expected_bytes = canonical_bytes(
        payload
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            expected_bytes,
            (
                "Existing preregistration/report "
                f"is not exact: {path}"
            ),
        )

    else:
        path.write_bytes(
            expected_bytes
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


def histogram_for_counts(
    counts: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(
        counts,
        dtype=np.int64,
    ).ravel()

    require(
        np.all(
            flat >= 0
        ),
        (
            "Negative occupancy count."
        ),
    )

    require(
        np.all(
            flat <= N_MC
        ),
        (
            "Occupancy count exceeds frozen N_MC."
        ),
    )

    histogram = np.bincount(
        flat,
        minlength=(
            N_MC + 1
        ),
    )

    require(
        histogram.shape
        ==
        (
            N_MC + 1,
        ),
        (
            "Unexpected occupancy histogram shape."
        ),
    )

    return histogram.astype(
        np.int64,
        copy=False,
    )


def exact_loss_curve_from_histogram(
    histogram: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    For threshold gamma = k/N:

        inactive iff count <= k
        active   iff count >  k

    Symmetric posterior 0-1 loss, expressed exactly in
    integer numerator units:

        inactive cell loss numerator = count
        active cell loss numerator   = N - count

    Divide by N only for expected-mass interpretation.
    """

    h = np.asarray(
        histogram,
        dtype=np.int64,
    )

    require(
        h.shape
        ==
        (
            N_MC + 1,
        ),
        (
            "Histogram shape mismatch."
        ),
    )

    values = np.arange(
        N_MC + 1,
        dtype=np.int64,
    )

    false_negative_weight = (
        values
        *
        h
    )

    false_dim_weight = (
        (
            N_MC
            -
            values
        )
        *
        h
    )

    cumulative_fn = np.cumsum(
        false_negative_weight,
        dtype=np.int64,
    )

    total_fd = int(
        np.sum(
            false_dim_weight,
            dtype=np.int64,
        )
    )

    cumulative_fd = np.cumsum(
        false_dim_weight,
        dtype=np.int64,
    )

    # For threshold k:
    # active values are c > k.
    false_dim = (
        total_fd
        -
        cumulative_fd
    )

    risk = (
        cumulative_fn
        +
        false_dim
    )

    total_cells = int(
        np.sum(
            h,
            dtype=np.int64,
        )
    )

    cumulative_cells = np.cumsum(
        h,
        dtype=np.int64,
    )

    masked_cells = (
        total_cells
        -
        cumulative_cells
    )

    return {
        "false_negative_numerator":
            cumulative_fn,

        "false_dim_numerator":
            false_dim,

        "risk_numerator":
            risk,

        "masked_cells":
            masked_cells,
    }


def report_point(
    *,
    k: int,
    gamma: float,
    aggregate_risk: np.ndarray,
    aggregate_fn: np.ndarray,
    aggregate_fd: np.ndarray,
    union_mask_cells: np.ndarray,
) -> dict[str, Any]:
    return {
        "k":
            int(
                k
            ),

        "gamma":
            float(
                gamma
            ),

        "symmetric_loss_numerator":
            int(
                aggregate_risk[
                    k
                ]
            ),

        "expected_missed_occupancy_mass":
            float(
                aggregate_fn[
                    k
                ]
                /
                N_MC
            ),

        "expected_false_dim_mass":
            float(
                aggregate_fd[
                    k
                ]
                /
                N_MC
            ),

        "union_predictive_mask_cells_total":
            int(
                np.sum(
                    union_mask_cells[
                        k
                    ],
                    dtype=np.int64,
                )
            ),

        "union_predictive_mask_cells_by_horizon":
            [
                int(
                    value
                )
                for value in
                union_mask_cells[
                    k
                ]
            ],
    }


try:
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.5 PART2A — CLASS-AGNOSTIC GAMMA "
        "PREREGISTRATION + EXHAUSTIVE CURVE"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # A. Frozen upstream seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN UPSTREAM SEAL ====="
    )

    for path, expected, label in (
        (
            PART1_MODULE,
            EXPECTED[
                "part1_module"
            ],
            "Block6.5 Part1 module",
        ),
        (
            PART1_CONTRACT,
            EXPECTED[
                "part1_contract"
            ],
            "Block6.5 Part1 contract",
        ),
        (
            PART1_REPORT,
            EXPECTED[
                "part1_report"
            ],
            "Block6.5 Part1 report",
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
            CONVERGENCE_REPORT,
            EXPECTED[
                "convergence_report"
            ],
            "Block6.4 convergence report",
        ),
        (
            COUNTS,
            EXPECTED[
                "counts"
            ],
            "Block6.4 convergence counts",
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
            actual == expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:28s} = EXACT PASS"
        )

    numeric = json.loads(
        BLOCK64_NUMERIC.read_text(
            encoding="utf-8"
        )
    )

    require(
        numeric[
            "MC"
        ][
            "runtime_sample_count"
        ]
        ==
        N_MC,
        (
            "Frozen N_MC changed."
        ),
    )

    require(
        numeric[
            "scientific_illumination_grid"
        ][
            "shape"
        ]
        ==
        list(
            GRID_SHAPE
        ),
        (
            "Frozen grid changed."
        ),
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
            "Formal population entered "
            "gamma-development evidence."
        ),
    )

    # ========================================================
    # B. PREREGISTRATION — BEFORE LOADING COUNTS
    # ========================================================

    print()
    print(
        "===== B. CLASS-AGNOSTIC GAMMA PREREGISTRATION ====="
    )

    prereg_payload = {
        "stage":
            6,

        "block":
            "6.5-Part2A",

        "status":
            "PREREGISTERED_BEFORE_GAMMA_CURVE_EXTRACTION",

        "scope":
            (
                "class_agnostic_predictive_ADB_baseline"
            ),

        "selected_gamma":
            SELECTED_GAMMA,

        "selected_gamma_exact_MC_breakpoint": {
            "k":
                SELECTED_K,

            "N":
                N_MC,

            "value":
                (
                    "4096/8192 = 0.5"
                ),
        },

        "threshold_semantics":
            (
                "M_i_tau = "
                "1[P_occ_i(theta,r,tau) > gamma]"
            ),

        "equal_probability_behavior":
            (
                "P_occ == gamma is inactive"
            ),

        "selection_rule":
            (
                "symmetric_posterior_0_1_Bayes_MAP_"
                "occupancy_decision_threshold"
            ),

        "selection_rule_details": {
            "false_dim_cost":
                "1_unit",

            "missed_occupancy_cost":
                "1_unit",

            "posterior_action":
                (
                    "mask iff P(occupied) > "
                    "P(not_occupied)"
                ),

            "development_metric_optimization":
                False,

            "posthoc_curve_selection":
                False,
        },

        "gamma_curve_plan": {
            "evaluation":
                "exhaustive",

            "threshold_breakpoints":
                (
                    "all_k_over_8192_for_"
                    "k_equal_0_to_8192"
                ),

            "threshold_count":
                8193,

            "purpose":
                (
                    "sensitivity_and_validation_"
                    "not_threshold_tuning"
                ),
        },

        "important_boundary": {
            "this_gamma_applies_to":
                (
                    "class_agnostic_predictive_baseline"
                ),

            "future_class_aware_thresholds_forbidden":
                False,

            "class_aware_thresholds":
                (
                    "must_be_separately_preregistered_"
                    "on_development_before_formal"
                ),

            "vehicle_pedestrian_cyclist_costs":
                "NOT_DEFINED_HERE",
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

    prereg_sha = write_json_exact_or_create(
        PREREG,
        prereg_payload,
    )

    print(
        "preregistration =",
        PREREG,
    )

    print(
        "preregistration SHA256 =",
        prereg_sha,
    )

    print(
        "selected gamma = 0.5"
    )

    print(
        "selection basis = "
        "SYMMETRIC POSTERIOR BAYES/MAP RULE"
    )

    print(
        "gamma curve outcomes read before prereg = NO"
    )

    # ========================================================
    # C. Load real development N=8192 occupancy evidence
    # ========================================================

    print()
    print(
        "===== C. REAL DEVELOPMENT P_OCC EVIDENCE ====="
    )

    data = np.load(
        COUNTS,
        allow_pickle=False,
    )

    actor_counts = []

    actor_ids = []

    require(
        len(
            convergence[
                "actors"
            ]
        )
        ==
        ACTOR_COUNT,
        (
            "Expected six matched development actors."
        ),
    )

    for actor_index in range(
        ACTOR_COUNT
    ):
        key = (
            f"actor_{actor_index:02d}_"
            f"counts_N{N_MC:05d}"
        )

        require(
            key in data.files,
            (
                f"Missing frozen count array {key}."
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
                HORIZON_COUNT,
                *GRID_SHAPE,
            ),
            (
                f"Unexpected shape for {key}: "
                f"{counts.shape}"
            ),
        )

        require(
            np.all(
                counts <= N_MC
            ),
            (
                "Occupancy count exceeds N_MC."
            ),
        )

        actor_counts.append(
            counts
        )

        actor_ids.append(
            str(
                convergence[
                    "actors"
                ][
                    actor_index
                ][
                    "prediction_id"
                ]
            )
        )

        print(
            f"actor {actor_index} "
            f"| prediction={actor_ids[-1]} "
            f"| positive cells="
            f"{int(np.count_nonzero(counts))}"
        )

    actor_counts_array = np.stack(
        actor_counts,
        axis=0,
    )

    require(
        actor_counts_array.shape
        ==
        (
            ACTOR_COUNT,
            HORIZON_COUNT,
            *GRID_SHAPE,
        ),
        (
            "Stacked actor count shape mismatch."
        ),
    )

    print(
        "matched actors =",
        ACTOR_COUNT,
    )

    print(
        "N_MC =",
        N_MC,
    )

    print(
        "grid =",
        GRID_SHAPE,
    )

    # ========================================================
    # D. Exhaustive gamma breakpoints
    # ========================================================

    print()
    print(
        "===== D. EXHAUSTIVE 8193-POINT GAMMA CURVE ====="
    )

    k_values = np.arange(
        N_MC + 1,
        dtype=np.int64,
    )

    gamma_values = (
        k_values.astype(
            np.float64
        )
        /
        float(
            N_MC
        )
    )

    per_actor_risk = np.zeros(
        (
            ACTOR_COUNT,
            N_MC + 1,
        ),
        dtype=np.int64,
    )

    per_actor_fn = np.zeros_like(
        per_actor_risk
    )

    per_actor_fd = np.zeros_like(
        per_actor_risk
    )

    per_actor_mask_cells = np.zeros_like(
        per_actor_risk
    )

    per_actor_horizon_mask_cells = np.zeros(
        (
            ACTOR_COUNT,
            N_MC + 1,
            HORIZON_COUNT,
        ),
        dtype=np.int64,
    )

    for actor_index in range(
        ACTOR_COUNT
    ):
        total_hist = histogram_for_counts(
            actor_counts_array[
                actor_index
            ]
        )

        curve = (
            exact_loss_curve_from_histogram(
                total_hist
            )
        )

        per_actor_risk[
            actor_index
        ] = curve[
            "risk_numerator"
        ]

        per_actor_fn[
            actor_index
        ] = curve[
            "false_negative_numerator"
        ]

        per_actor_fd[
            actor_index
        ] = curve[
            "false_dim_numerator"
        ]

        per_actor_mask_cells[
            actor_index
        ] = curve[
            "masked_cells"
        ]

        for horizon_index in range(
            HORIZON_COUNT
        ):
            h_hist = histogram_for_counts(
                actor_counts_array[
                    actor_index,
                    horizon_index,
                ]
            )

            h_curve = (
                exact_loss_curve_from_histogram(
                    h_hist
                )
            )

            per_actor_horizon_mask_cells[
                actor_index,
                :,
                horizon_index,
            ] = h_curve[
                "masked_cells"
            ]

    aggregate_risk = np.sum(
        per_actor_risk,
        axis=0,
        dtype=np.int64,
    )

    aggregate_fn = np.sum(
        per_actor_fn,
        axis=0,
        dtype=np.int64,
    )

    aggregate_fd = np.sum(
        per_actor_fd,
        axis=0,
        dtype=np.int64,
    )

    # Exact predictive multi-actor union for common gamma:
    #
    # union active iff any actor count > k
    #              iff max_i(count_i) > k.
    max_actor_counts = np.max(
        actor_counts_array,
        axis=0,
    )

    union_mask_cells = np.zeros(
        (
            N_MC + 1,
            HORIZON_COUNT,
        ),
        dtype=np.int64,
    )

    for horizon_index in range(
        HORIZON_COUNT
    ):
        histogram = histogram_for_counts(
            max_actor_counts[
                horizon_index
            ]
        )

        curve = (
            exact_loss_curve_from_histogram(
                histogram
            )
        )

        union_mask_cells[
            :,
            horizon_index
        ] = curve[
            "masked_cells"
        ]

    print(
        "gamma breakpoints =",
        gamma_values.size,
    )

    print(
        "minimum gamma =",
        gamma_values[
            0
        ],
    )

    print(
        "maximum gamma =",
        gamma_values[
            -1
        ],
    )

    print(
        "selected gamma breakpoint index =",
        SELECTED_K,
    )

    # ========================================================
    # E. Exact MAP/Bayes optimality gate
    # ========================================================

    print()
    print(
        "===== E. PREREGISTERED GAMMA OPTIMALITY GATE ====="
    )

    minimum_risk = int(
        np.min(
            aggregate_risk
        )
    )

    selected_risk = int(
        aggregate_risk[
            SELECTED_K
        ]
    )

    minimizer_indices = np.flatnonzero(
        aggregate_risk
        ==
        minimum_risk
    )

    require(
        selected_risk
        ==
        minimum_risk,
        (
            "Preregistered gamma=0.5 is not "
            "an exact minimizer of the symmetric "
            "posterior 0-1 loss."
        ),
    )

    require(
        SELECTED_K
        in
        set(
            int(x)
            for x in minimizer_indices
        ),
        (
            "Selected gamma breakpoint missing "
            "from exact risk minimizers."
        ),
    )

    print(
        "selected gamma =",
        SELECTED_GAMMA,
    )

    print(
        "selected k =",
        SELECTED_K,
    )

    print(
        "exact selected risk numerator =",
        selected_risk,
    )

    print(
        "exact global minimum risk numerator =",
        minimum_risk,
    )

    print(
        "number of tied minimizing breakpoints =",
        int(
            minimizer_indices.size
        ),
    )

    print(
        "preregistered gamma is exact minimizer = PASS"
    )

    # ========================================================
    # F. Public Block6.5 API parity at gamma=0.5
    # ========================================================

    print()
    print(
        "===== F. PUBLIC MASK API PARITY ====="
    )

    from iscai_stage6.adb.probabilistic_occupancy import (
        ActorOccupancyProbability,
    )

    from iscai_stage6.adb.predictive_mask import (
        aggregate_binary_actor_masks,
        threshold_actor_occupancy,
    )

    actor_masks = []

    for actor_index in range(
        ACTOR_COUNT
    ):
        counts = actor_counts_array[
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

        occupancy = (
            ActorOccupancyProbability(
                prediction_id=(
                    actor_ids[
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

        actor_masks.append(
            threshold_actor_occupancy(
                occupancy,
                gamma=SELECTED_GAMMA,
            )
        )

    public_union = (
        aggregate_binary_actor_masks(
            actor_masks,
            grid_shape=GRID_SHAPE,
        )
    )

    exact_union = (
        max_actor_counts
        >
        SELECTED_K
    )

    require(
        np.array_equal(
            public_union,
            exact_union,
        ),
        (
            "Public Block6.5 mask API does "
            "not match exact frozen count "
            "threshold at gamma=0.5."
        ),
    )

    print(
        "strict threshold parity = EXACT PASS"
    )

    print(
        "multi-actor union parity = EXACT PASS"
    )

    print(
        "P_occ == 0.5 inactive = YES"
    )

    # ========================================================
    # G. Selected-point development diagnostics
    # ========================================================

    print()
    print(
        "===== G. GAMMA=0.5 DEVELOPMENT DIAGNOSTICS ====="
    )

    selected_point = report_point(
        k=SELECTED_K,
        gamma=SELECTED_GAMMA,
        aggregate_risk=aggregate_risk,
        aggregate_fn=aggregate_fn,
        aggregate_fd=aggregate_fd,
        union_mask_cells=union_mask_cells,
    )

    for key, value in selected_point.items():
        print(
            f"{key} = {value}"
        )

    total_grid_cells_per_horizon = (
        GRID_SHAPE[
            0
        ]
        *
        GRID_SHAPE[
            1
        ]
    )

    selected_mask_fraction = [
        float(
            value
            /
            total_grid_cells_per_horizon
        )
        for value in
        union_mask_cells[
            SELECTED_K
        ]
    ]

    print(
        "union mask fraction by horizon =",
        selected_mask_fraction,
    )

    # ========================================================
    # H. Persist exhaustive curve
    # ========================================================

    print()
    print(
        "===== H. EXHAUSTIVE CURVE ARTIFACT ====="
    )

    CURVE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        CURVE,

        gamma=gamma_values,

        threshold_k=k_values,

        aggregate_symmetric_loss_numerator=(
            aggregate_risk
        ),

        aggregate_expected_missed_occupancy_numerator=(
            aggregate_fn
        ),

        aggregate_expected_false_dim_numerator=(
            aggregate_fd
        ),

        per_actor_symmetric_loss_numerator=(
            per_actor_risk
        ),

        per_actor_expected_missed_occupancy_numerator=(
            per_actor_fn
        ),

        per_actor_expected_false_dim_numerator=(
            per_actor_fd
        ),

        per_actor_mask_cells=(
            per_actor_mask_cells
        ),

        per_actor_horizon_mask_cells=(
            per_actor_horizon_mask_cells
        ),

        union_predictive_mask_cells=(
            union_mask_cells
        ),

        actor_prediction_ids=np.asarray(
            actor_ids,
            dtype="U64",
        ),

        N_MC=np.asarray(
            [
                N_MC
            ],
            dtype=np.int64,
        ),

        selected_gamma=np.asarray(
            [
                SELECTED_GAMMA
            ],
            dtype=np.float64,
        ),

        selected_k=np.asarray(
            [
                SELECTED_K
            ],
            dtype=np.int64,
        ),
    )

    curve_sha = sha256_file(
        CURVE
    )

    print(
        "curve artifact =",
        CURVE,
    )

    print(
        "curve SHA256 =",
        curve_sha,
    )

    # ========================================================
    # I. Report
    # ========================================================

    print()
    print(
        "===== I. BLOCK6.5 PART2A REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.5-Part2A",

        "status":
            (
                "PASS_CLASS_AGNOSTIC_GAMMA_"
                "PREREGISTRATION_AND_CURVE"
            ),

        "preregistration": {
            "path":
                str(
                    PREREG
                ),

            "sha256":
                prereg_sha,

            "written_before_gamma_curve_counts_loaded":
                True,
        },

        "gamma": {
            "scope":
                "class_agnostic_predictive_baseline",

            "preregistered_value":
                SELECTED_GAMMA,

            "exact_breakpoint":
                {
                    "k":
                        SELECTED_K,

                    "N":
                        N_MC,
                },

            "threshold_semantics":
                "strict_greater_than",

            "equal_to_gamma_inactive":
                True,

            "selection_rule":
                (
                    "symmetric_posterior_0_1_"
                    "Bayes_MAP_rule"
                ),

            "selected_from_development_curve":
                False,

            "curve_used_for":
                "sensitivity_and_validation_only",
        },

        "curve": {
            "exhaustive":
                True,

            "breakpoint_count":
                int(
                    gamma_values.size
                ),

            "minimum_gamma":
                0.0,

            "maximum_gamma":
                1.0,

            "artifact_path":
                str(
                    CURVE
                ),

            "artifact_sha256":
                curve_sha,
        },

        "optimality_validation": {
            "selected_risk_numerator":
                selected_risk,

            "global_minimum_risk_numerator":
                minimum_risk,

            "selected_is_global_minimizer":
                True,

            "number_of_tied_minimizing_breakpoints":
                int(
                    minimizer_indices.size
                ),

            "minimum_breakpoint_k":
                int(
                    minimizer_indices[
                        0
                    ]
                ),

            "maximum_breakpoint_k":
                int(
                    minimizer_indices[
                        -1
                    ]
                ),
        },

        "selected_point_diagnostics":
            {
                **selected_point,

                "union_mask_fraction_by_horizon":
                    selected_mask_fraction,
            },

        "public_API_validation": {
            "threshold_actor_occupancy":
                "EXACT_PASS",

            "aggregate_binary_actor_masks":
                "EXACT_PASS",

            "direct_count_threshold_equivalence":
                True,
        },

        "reactive_fallback": {
            "gamma_dependent":
                False,

            "modified":
                False,

            "semantics":
                (
                    "exact_frozen_Block6.2_"
                    "continuous_reactive_fallback"
                ),
        },

        "future_class_aware_controller": {
            "gamma_0p5_forced":
                False,

            "class_specific_thresholds_may_differ":
                True,

            "require_separate_development_preregistration":
                True,
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
            "new_MC_sampling":
                False,

            "model_forward":
                False,

            "training":
                False,

            "recalibration":
                False,

            "source_modification":
                False,
        },

        "next":
            (
                "BLOCK6.5_PART2B_FREEZE_"
                "CLASS_AGNOSTIC_GAMMA_AND_REAL_MASK_SMOKE"
            ),
    }

    report_sha = write_json_exact_or_create(
        REPORT,
        report_payload,
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
    # J. Regression
    # ========================================================

    print()
    print(
        "===== J. FRESH STAGE6 REGRESSION ====="
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
            "Fresh Stage6 regression failed."
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
    # K. Final
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.5 PART2A — FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "STATUS = "
        "PASS_CLASS_AGNOSTIC_GAMMA_PREREGISTRATION_AND_CURVE"
    )

    print(
        "gamma scope                = CLASS-AGNOSTIC BASELINE"
    )

    print(
        "preregistered gamma        =",
        SELECTED_GAMMA,
    )

    print(
        "exact MC breakpoint        = "
        "4096 / 8192"
    )

    print(
        "selection rule             = "
        "SYMMETRIC POSTERIOR BAYES/MAP"
    )

    print(
        "posthoc gamma selection    = NO"
    )

    print(
        "gamma curve                = "
        "EXHAUSTIVE | 8193 BREAKPOINTS"
    )

    print(
        "selected gamma global risk minimizer = PASS"
    )

    print(
        "strict public API parity   = PASS"
    )

    print(
        "multi-actor public parity  = PASS"
    )

    print(
        "reactive fallback changed  = NO"
    )

    print(
        "new MC sampling            = NO"
    )

    print(
        "class-aware gamma frozen   = NO"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "formal outcomes read       = NO"
    )

    print(
        "source modification        = NO"
    )

    print(
        "fresh Stage6 regression    = PASS | 145"
    )

    print(
        "prereg SHA256              =",
        prereg_sha,
    )

    print(
        "curve SHA256               =",
        curve_sha,
    )

    print(
        "report SHA256              =",
        report_sha,
    )

    print(
        "NEXT = BLOCK6.5 PART2B "
        "FREEZE CLASS-AGNOSTIC GAMMA + REAL MASK SMOKE"
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
        "BLOCK6.5 PART2A — CONTROLLED BLOCK"
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
        "gamma freeze         = NO"
    )

    print(
        "formal evaluation    = NO"
    )

    print(
        "formal tuning        = NO"
    )

    print(
        "source modification  = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

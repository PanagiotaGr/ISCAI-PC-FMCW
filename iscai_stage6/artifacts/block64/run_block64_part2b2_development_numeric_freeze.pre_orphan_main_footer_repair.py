from __future__ import annotations

from hashlib import sha256
import json
import math
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

PART2B1_REPORT = (
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

PART2A_REPORT = (
    S6
    / "reports"
    / "block64_part2a_probabilistic_occupancy_kernel.json"
)

PART2A_CONTRACT = (
    S6
    / "configs"
    / "block64_part2a_probabilistic_occupancy_contract.json"
)

OCCUPANCY_MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "probabilistic_occupancy.py"
)

FULL_BOX_MODULE = (
    S6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "probabilistic_full_box.py"
)

POSTERIOR = (
    S6
    / "artifacts"
    / "block63"
    / "identity_safe_development_gaussian_posterior.jsonl"
)

CONFIG = (
    S6
    / "configs"
    / "block64_mc_grid_numeric_freeze.json"
)

REPORT = (
    S6
    / "reports"
    / "block64_part2b2_development_numeric_freeze.json"
)

FREEZE = (
    S6
    / "artifacts"
    / "block64"
    / "block64_part2b2_numeric_freeze_manifest.json"
)

HANDOFF = (
    S6
    / "artifacts"
    / "block64"
    / "block64_to_block65_handoff.json"
)


EXPECTED = {
    "part2b1_report":
        (
            "57f23aaa55e31967653528157f2d7703f"
            "c10b632d353e2b8e25c53c379bc4d1f"
        ),

    "counts":
        (
            "ee536151df1b49294ebdeda598d747340"
            "83dc90769fd3d70dc8a058142701b7a"
        ),

    "part2a_report":
        (
            "42ba4b5bfd54a4e5b74773f71a5ce2d"
            "040021a188dffb336beb2b32fb852b0a3"
        ),

    "part2a_contract":
        (
            "3c61222ca829849ee77b276ff21a199b"
            "016b0fe8455b6f4045d4e980f4874ae8"
        ),

    "occupancy_module":
        (
            "d4206761fc70495563d53a6ab58aefe6"
            "db5d8bd1b2d09736bfc5bbc6b6f50e3e"
        ),

    "full_box_module":
        (
            "51daeab95713d6e7d2738ced466be4844"
            "b73dce70a09138b2b769de880462995"
        ),

    "posterior":
        (
            "318b22dfc3788195a1432169075d1c865"
            "4f0d183337711918db2303b5668f5ea"
        ),
}


CANDIDATES = (
    2048,
    4096,
    8192,
)

REFERENCE_N = 32768
SELECTED_N = 8192
FROZEN_SEED = 20260821

ACTIVE_ACTORS_EXPECTED = (
    2,
    4,
)

INACTIVE_ACTORS_EXPECTED = (
    0,
    1,
    3,
    5,
)


class ControlledBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(
        condition
    ):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
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
    payload: dict[str, Any],
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
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
                    match.group(
                        1
                    )
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


def active_metrics(
    candidate_counts: np.ndarray,
    reference_counts: np.ndarray,
    *,
    candidate_n: int,
) -> dict[str, float]:
    candidate = (
        candidate_counts.astype(
            np.float64
        )
        /
        float(
            candidate_n
        )
    )

    reference = (
        reference_counts.astype(
            np.float64
        )
        /
        float(
            REFERENCE_N
        )
    )

    error = np.abs(
        candidate
        -
        reference
    )

    support = (
        (
            candidate_counts
            >
            0
        )
        |
        (
            reference_counts
            >
            0
        )
    )

    intersection = (
        (
            candidate_counts
            >
            0
        )
        &
        (
            reference_counts
            >
            0
        )
    )

    reference_mass = float(
        np.sum(
            reference
        )
    )

    candidate_mass = float(
        np.sum(
            candidate
        )
    )

    require(
        reference_mass
        >
        0.0,
        (
            "active_metrics called on "
            "zero-reference-support actor."
        ),
    )

    return {
        "max_abs":
            float(
                np.max(
                    error
                )
            ),

        "support_mae":
            float(
                np.mean(
                    error[
                        support
                    ]
                )
            ),

        "normalized_l1":
            float(
                np.sum(
                    error
                )
                /
                reference_mass
            ),

        "mass_relative_error":
            float(
                abs(
                    candidate_mass
                    -
                    reference_mass
                )
                /
                reference_mass
            ),

        "support_iou":
            float(
                np.count_nonzero(
                    intersection
                )
                /
                np.count_nonzero(
                    support
                )
            ),
    }


try:
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.4 PART2B/2 — DEVELOPMENT NUMERIC FREEZE"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # A. Immutable evidence seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE DEVELOPMENT EVIDENCE SEAL ====="
    )

    for path, expected, label in (
        (
            PART2B1_REPORT,
            EXPECTED[
                "part2b1_report"
            ],
            "Part2B/1 report",
        ),
        (
            COUNTS,
            EXPECTED[
                "counts"
            ],
            "convergence counts",
        ),
        (
            PART2A_REPORT,
            EXPECTED[
                "part2a_report"
            ],
            "Part2A report",
        ),
        (
            PART2A_CONTRACT,
            EXPECTED[
                "part2a_contract"
            ],
            "Part2A contract",
        ),
        (
            OCCUPANCY_MODULE,
            EXPECTED[
                "occupancy_module"
            ],
            "occupancy module",
        ),
        (
            FULL_BOX_MODULE,
            EXPECTED[
                "full_box_module"
            ],
            "full-box module",
        ),
        (
            POSTERIOR,
            EXPECTED[
                "posterior"
            ],
            "posterior",
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
            f"{label:20s} = EXACT PASS"
        )

    convergence_report = json.loads(
        PART2B1_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        convergence_report.get(
            "status"
        )
        ==
        (
            "PASS_REAL_DEVELOPMENT_MC_"
            "CONVERGENCE_CURVES_GENERATED"
        ),
        (
            "Unexpected Part2B/1 status."
        ),
    )

    require(
        convergence_report[
            "development_population"
        ][
            "formal_population_used"
        ]
        is False,
        (
            "Formal population entered "
            "development convergence."
        ),
    )

    print(
        "formal population used = NO"
    )

    # ========================================================
    # B. Reproduce reference-active population
    # ========================================================

    print()
    print(
        "===== B. REFERENCE-ACTIVE SUPPORT REPRODUCTION ====="
    )

    data = np.load(
        COUNTS,
        allow_pickle=False,
    )

    active = []
    inactive = []

    reference_counts = {}

    for actor_index in range(
        6
    ):
        key = (
            f"actor_{actor_index:02d}_"
            f"counts_N{REFERENCE_N:05d}"
        )

        require(
            key in data.files,
            (
                f"Missing counts key {key}."
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
                501,
                301,
            ),
            (
                "Unexpected convergence-count "
                f"shape for actor {actor_index}: "
                f"{counts.shape}"
            ),
        )

        reference_counts[
            actor_index
        ] = counts

        if np.count_nonzero(
            counts
        ):
            active.append(
                actor_index
            )

        else:
            inactive.append(
                actor_index
            )

    require(
        tuple(
            active
        )
        ==
        ACTIVE_ACTORS_EXPECTED,
        (
            "Reference-active actor set changed: "
            f"{active}"
        ),
    )

    require(
        tuple(
            inactive
        )
        ==
        INACTIVE_ACTORS_EXPECTED,
        (
            "Reference-inactive actor set changed: "
            f"{inactive}"
        ),
    )

    print(
        "reference-active actors   =",
        tuple(
            active
        ),
    )

    print(
        "reference-inactive actors =",
        tuple(
            inactive
        ),
    )

    print(
        "active horizons actor 2   =",
        [
            int(
                np.count_nonzero(
                    reference_counts[
                        2
                    ][
                        h
                    ]
                )
            )
            for h in range(
                4
            )
        ],
    )

    print(
        "active horizons actor 4   =",
        [
            int(
                np.count_nonzero(
                    reference_counts[
                        4
                    ][
                        h
                    ]
                )
            )
            for h in range(
                4
            )
        ],
    )

    # ========================================================
    # C. Recompute active-only convergence
    # ========================================================

    print()
    print(
        "===== C. ACTIVE-ONLY CONVERGENCE REPRODUCTION ====="
    )

    summaries = {}

    for candidate_n in CANDIDATES:
        actor_metrics = []

        for actor_index in active:
            key = (
                f"actor_{actor_index:02d}_"
                f"counts_N{candidate_n:05d}"
            )

            require(
                key in data.files,
                (
                    f"Missing candidate counts {key}."
                ),
            )

            candidate_counts = np.asarray(
                data[
                    key
                ],
                dtype=np.uint32,
            )

            actor_metrics.append(
                active_metrics(
                    candidate_counts,
                    reference_counts[
                        actor_index
                    ],
                    candidate_n=(
                        candidate_n
                    ),
                )
            )

        summary = {
            "sample_count":
                int(
                    candidate_n
                ),

            "worst_max_abs":
                float(
                    max(
                        item[
                            "max_abs"
                        ]
                        for item in actor_metrics
                    )
                ),

            "mean_support_mae":
                float(
                    np.mean(
                        [
                            item[
                                "support_mae"
                            ]
                            for item in actor_metrics
                        ]
                    )
                ),

            "worst_support_mae":
                float(
                    max(
                        item[
                            "support_mae"
                        ]
                        for item in actor_metrics
                    )
                ),

            "mean_normalized_l1":
                float(
                    np.mean(
                        [
                            item[
                                "normalized_l1"
                            ]
                            for item in actor_metrics
                        ]
                    )
                ),

            "worst_normalized_l1":
                float(
                    max(
                        item[
                            "normalized_l1"
                        ]
                        for item in actor_metrics
                    )
                ),

            "mean_mass_relative_error":
                float(
                    np.mean(
                        [
                            item[
                                "mass_relative_error"
                            ]
                            for item in actor_metrics
                        ]
                    )
                ),

            "mean_support_iou":
                float(
                    np.mean(
                        [
                            item[
                                "support_iou"
                            ]
                            for item in actor_metrics
                        ]
                    )
                ),
        }

        summaries[
            candidate_n
        ] = summary

        print()
        print(
            "N =",
            candidate_n,
        )

        for key, value in summary.items():
            if key == "sample_count":
                continue

            print(
                f"  {key} = {value}"
            )

    # ========================================================
    # D. Development selection rule
    # ========================================================

    print()
    print(
        "===== D. DEVELOPMENT MC SELECTION RULE ====="
    )

    n2048 = summaries[
        2048
    ]

    n4096 = summaries[
        4096
    ]

    n8192 = summaries[
        8192
    ]

    decreasing_metrics = (
        "worst_max_abs",
        "mean_support_mae",
        "worst_support_mae",
        "mean_normalized_l1",
        "worst_normalized_l1",
        "mean_mass_relative_error",
    )

    for metric in decreasing_metrics:
        require(
            n8192[
                metric
            ]
            <
            n4096[
                metric
            ]
            <
            n2048[
                metric
            ],
            (
                "Required development convergence "
                f"ordering failed for {metric}: "
                f"{n2048[metric]}, "
                f"{n4096[metric]}, "
                f"{n8192[metric]}"
            ),
        )

    require(
        n8192[
            "mean_support_iou"
        ]
        >
        n4096[
            "mean_support_iou"
        ]
        >
        n2048[
            "mean_support_iou"
        ],
        (
            "Support-IoU convergence ordering "
            "failed."
        ),
    )

    require(
        SELECTED_N
        ==
        max(
            CANDIDATES
        ),
        (
            "Selected N is not the largest "
            "frozen Stage5 candidate."
        ),
    )

    print(
        "eligible candidates =",
        CANDIDATES,
    )

    print(
        "reference N         =",
        REFERENCE_N,
        "(development diagnostic only)",
    )

    print(
        "selection evidence  = "
        "8192 strictly improves every "
        "precomputed active-support metric "
        "relative to 4096 and 2048"
    )

    print(
        "selected runtime N  =",
        SELECTED_N,
    )

    print(
        "new arbitrary MC N  = NO"
    )

    print(
        "reference N selected as runtime = NO"
    )

    # ========================================================
    # E. Freeze seed and scientific illumination grid
    # ========================================================

    print()
    print(
        "===== E. SEED + SCIENTIFIC GRID FREEZE ====="
    )

    from iscai_stage5.angular_monte_carlo import (
        MONTE_CARLO_BASE_SEED,
        MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS,
        MONTE_CARLO_REFERENCE_SAMPLE_COUNT,
    )

    require(
        int(
            MONTE_CARLO_BASE_SEED
        )
        ==
        FROZEN_SEED,
        (
            "Frozen Stage5 base seed changed."
        ),
    )

    require(
        tuple(
            int(x)
            for x in
            MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS
        )
        ==
        CANDIDATES,
        (
            "Frozen Stage5 candidate set changed."
        ),
    )

    require(
        int(
            MONTE_CARLO_REFERENCE_SAMPLE_COUNT
        )
        ==
        REFERENCE_N,
        (
            "Frozen Stage5 reference N changed."
        ),
    )

    # Exact Part-A controller domain, expressed in Stage6
    # H0/headlamp theta convention. The Part-A↔H0 angle
    # mapping is a sign reversal; because the domain is
    # symmetric, its numerical support remains [-25,25] deg.
    theta_deg = np.linspace(
        -25.0,
        25.0,
        501,
        dtype=np.float64,
    )

    range_m = np.linspace(
        0.0,
        150.0,
        301,
        dtype=np.float64,
    )

    require(
        np.allclose(
            np.diff(
                theta_deg
            ),
            0.1,
            rtol=0.0,
            atol=1.0e-12,
        ),
        (
            "Theta grid construction failed."
        ),
    )

    require(
        np.allclose(
            np.diff(
                range_m
            ),
            0.5,
            rtol=0.0,
            atol=1.0e-12,
        ),
        (
            "Range grid construction failed."
        ),
    )

    print(
        "MC runtime sample count =",
        SELECTED_N,
    )

    print(
        "MC runtime seed         =",
        FROZEN_SEED,
    )

    print(
        "seed source             = "
        "frozen Stage5 MONTE_CARLO_BASE_SEED"
    )

    print(
        "seed policy             = "
        "same frozen deterministic seed per "
        "actor posterior evaluation"
    )

    print(
        "theta grid              = "
        "-25..25 deg | 501 centers | 0.1 deg"
    )

    print(
        "range grid              = "
        "0..150 m | 301 centers | 0.5 m"
    )

    print(
        "grid provenance         = "
        "frozen Part-A ADB actuator domain"
    )

    print(
        "grid tuning on formal   = NO"
    )

    # ========================================================
    # F. Numeric freeze config
    # ========================================================

    print()
    print(
        "===== F. BLOCK6.4 NUMERIC FREEZE CONFIG ====="
    )

    config_payload = {
        "stage":
            6,

        "block":
            "6.4-Part2B/2",

        "status":
            "FROZEN_DEVELOPMENT_NUMERICS",

        "freeze_scope":
            [
                "MC_runtime_sample_count",
                "MC_runtime_seed",
                "scientific_illumination_grid",
            ],

        "MC": {
            "runtime_sample_count":
                SELECTED_N,

            "runtime_seed":
                FROZEN_SEED,

            "seed_policy":
                (
                    "same_frozen_Stage5_base_seed_"
                    "for_each_actor_posterior_evaluation"
                ),

            "candidate_counts_considered":
                list(
                    CANDIDATES
                ),

            "reference_sample_count":
                REFERENCE_N,

            "reference_role":
                "development_convergence_reference_only",

            "joint_sampling_semantics":
                (
                    "product_of_frozen_calibrated_"
                    "per_horizon_Gaussian_marginals"
                ),

            "selection_population":
                {
                    "formal":
                        False,

                    "scenario_id":
                        SCENARIO_ID,

                    "reference_active_actor_indices":
                        list(
                            ACTIVE_ACTORS_EXPECTED
                        ),

                    "reference_inactive_actor_indices":
                        list(
                            INACTIVE_ACTORS_EXPECTED
                        ),

                    "inactive_reason":
                        (
                            "deterministic_mean_full_boxes_"
                            "outside_forward_headlamp_"
                            "angular_domain"
                        ),
                },

            "selection_rule":
                (
                    "select_largest_preexisting_Stage5_"
                    "candidate_after_confirming_strict_"
                    "development_active_support_"
                    "improvement_over_all_lower_candidates"
                ),

            "active_support_convergence":
                {
                    str(n):
                        summaries[
                            n
                        ]
                    for n in CANDIDATES
                },
        },

        "scientific_illumination_grid": {
            "coordinate_frame":
                "Stage6_H0_headlamp",

            "theta_centers_deg": {
                "minimum":
                    -25.0,

                "maximum":
                    25.0,

                "step":
                    0.1,

                "count":
                    501,
            },

            "range_centers_m": {
                "minimum":
                    0.0,

                "maximum":
                    150.0,

                "step":
                    0.5,

                "count":
                    301,
            },

            "shape":
                [
                    501,
                    301,
                ],

            "provenance":
                (
                    "frozen_PartA_ADB_actuator_domain_"
                    "mapped_to_Stage6_H0_headlamp_theta"
                ),

            "formal_tuning":
                False,
        },

        "explicitly_not_frozen_yet": {
            "occupancy_gamma":
                True,

            "class_specific_gamma":
                True,

            "class_margins":
                True,

            "dimming_floors":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,

            "safety_metric_acceptance_thresholds":
                True,
        },

        "formal_boundary": {
            "formal_population_read":
                False,

            "formal_outcomes_read":
                False,

            "formal_parameter_tuning":
                False,
        },
    }

    config_sha = write_json(
        CONFIG,
        config_payload,
    )

    print(
        "config =",
        CONFIG,
    )

    print(
        "config SHA256 =",
        config_sha,
    )

    # ========================================================
    # G. Fresh Stage6 regression
    # ========================================================

    print()
    print(
        "===== G. FRESH STAGE6 REGRESSION ====="
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
        127,
        (
            "Expected 127 Stage6 tests, got "
            f"{regression['tests']}."
        ),
    )

    print(
        "fresh regression = PASS | 127"
    )

    # ========================================================
    # H. Freeze report
    # ========================================================

    print()
    print(
        "===== H. PART2B/2 DEVELOPMENT FREEZE REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.4-Part2B/2",

        "status":
            (
                "PASS_BLOCK64_MC_AND_GRID_"
                "DEVELOPMENT_NUMERIC_FREEZE"
            ),

        "numeric_freeze": {
            "MC_N":
                SELECTED_N,

            "MC_seed":
                FROZEN_SEED,

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

        "selection_evidence": {
            "reference_active_actors":
                list(
                    ACTIVE_ACTORS_EXPECTED
                ),

            "reference_inactive_actors":
                list(
                    INACTIVE_ACTORS_EXPECTED
                ),

            "candidate_metrics":
                {
                    str(n):
                        summaries[
                            n
                        ]
                    for n in CANDIDATES
                },

            "selected_candidate":
                SELECTED_N,

            "reference_N":
                REFERENCE_N,

            "all_selected_quality_metrics_improve_over_lower_candidates":
                True,

            "zero_support_actors_counted_as_convergence_success":
                False,
        },

        "upstream": {
            "part2b1_report_sha256":
                EXPECTED[
                    "part2b1_report"
                ],

            "counts_sha256":
                EXPECTED[
                    "counts"
                ],

            "part2a_report_sha256":
                EXPECTED[
                    "part2a_report"
                ],

            "part2a_contract_sha256":
                EXPECTED[
                    "part2a_contract"
                ],

            "occupancy_module_sha256":
                EXPECTED[
                    "occupancy_module"
                ],

            "full_box_module_sha256":
                EXPECTED[
                    "full_box_module"
                ],

            "posterior_sha256":
                EXPECTED[
                    "posterior"
                ],
        },

        "config": {
            "path":
                str(
                    CONFIG
                ),

            "sha256":
                config_sha,
        },

        "still_deferred": {
            "occupancy_gamma":
                True,

            "binary_actor_masks":
                True,

            "multi_actor_mask_aggregation":
                True,

            "class_aware_policy":
                True,

            "dimming_floors":
                True,

            "temporal_smoothing":
                True,

            "actuation_rate_limits":
                True,
        },

        "scientific_execution": {
            "new_model_forward":
                False,

            "new_MC_sampling":
                False,

            "formal_evaluation":
                False,

            "training":
                False,

            "recalibration":
                False,

            "formal_parameter_tuning":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                127,
        },

        "next":
            (
                "BLOCK6.5_DEVELOPMENT_GAMMA_"
                "AND_MULTI_ACTOR_MASK_AGGREGATION"
            ),
    }

    report_sha = write_json(
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
    # I. Freeze manifest + handoff
    # ========================================================

    print()
    print(
        "===== I. FREEZE MANIFEST + BLOCK6.5 HANDOFF ====="
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.4-Part2B/2",

        "status":
            "BLOCK64_MC_GRID_NUMERICS_FROZEN",

        "immutable_evidence": {
            str(
                PART2B1_REPORT
            ):
                EXPECTED[
                    "part2b1_report"
                ],

            str(
                COUNTS
            ):
                EXPECTED[
                    "counts"
                ],

            str(
                PART2A_REPORT
            ):
                EXPECTED[
                    "part2a_report"
                ],

            str(
                PART2A_CONTRACT
            ):
                EXPECTED[
                    "part2a_contract"
                ],

            str(
                OCCUPANCY_MODULE
            ):
                EXPECTED[
                    "occupancy_module"
                ],

            str(
                FULL_BOX_MODULE
            ):
                EXPECTED[
                    "full_box_module"
                ],

            str(
                POSTERIOR
            ):
                EXPECTED[
                    "posterior"
                ],
        },

        "numeric_config": {
            "path":
                str(
                    CONFIG
                ),

            "sha256":
                config_sha,
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
            "MC_runtime_N":
                SELECTED_N,

            "MC_seed":
                FROZEN_SEED,

            "grid_shape":
                [
                    501,
                    301,
                ],

            "gamma":
                "NOT_YET_FROZEN",

            "formal_tuning":
                "FORBIDDEN",
        },
    }

    freeze_sha = write_json(
        FREEZE,
        freeze_payload,
    )

    handoff_payload = {
        "stage":
            6,

        "from_block":
            "6.4",

        "to_block":
            "6.5",

        "status":
            (
                "PASS_READY_FOR_BLOCK65_"
                "DEVELOPMENT_MASK_POLICY"
            ),

        "frozen_MC": {
            "sample_count":
                SELECTED_N,

            "seed":
                FROZEN_SEED,

            "reference_N_not_runtime":
                REFERENCE_N,
        },

        "frozen_grid": {
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

            "shape":
                [
                    501,
                    301,
                ],
        },

        "occupancy_kernel": {
            "P_occ":
                "MC_full_box_coverage_fraction",

            "corners_per_box":
                8,

            "gamma":
                "NOT_YET_FROZEN",

            "multi_actor_aggregation":
                "NOT_YET_IMPLEMENTED",
        },

        "required_next_development_work": [
            "occupancy_gamma_preregistration",
            "binary_per_actor_mask_M_i_tau",
            "exact_multi_actor_aggregation",
            "class_agnostic_predictive_mask_baseline",
            "preserve_original_reactive_fallback_for_unmatched_ADB_actors",
        ],

        "formal_boundary": {
            "formal_population_used":
                False,

            "formal_outcomes_read":
                False,

            "formal_tuning_allowed":
                False,
        },

        "upstream_numeric_freeze": {
            "config_sha256":
                config_sha,

            "report_sha256":
                report_sha,

            "freeze_manifest_sha256":
                freeze_sha,
        },
    }

    handoff_sha = write_json(
        HANDOFF,
        handoff_payload,
    )

    print(
        "freeze manifest =",
        FREEZE,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
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
    # J. Final
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.4 PART2B/2 — FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "STATUS = "
        "PASS_BLOCK64_MC_GRID_DEVELOPMENT_NUMERICS_FROZEN"
    )

    print(
        "MC runtime N              =",
        SELECTED_N,
    )

    print(
        "MC seed                   =",
        FROZEN_SEED,
    )

    print(
        "MC reference N            =",
        REFERENCE_N,
        "| DEVELOPMENT ONLY"
    )

    print(
        "active actors for N study =",
        ACTIVE_ACTORS_EXPECTED,
    )

    print(
        "zero-support actors        =",
        INACTIVE_ACTORS_EXPECTED,
        "| OUTSIDE HEADLAMP DOMAIN"
    )

    print(
        "scientific grid            = "
        "501 x 301"
    )

    print(
        "theta                      = "
        "-25..25 deg @ 0.1 deg"
    )

    print(
        "range                      = "
        "0..150 m @ 0.5 m"
    )

    print(
        "grid provenance            = "
        "FROZEN PART-A ADB DOMAIN"
    )

    print(
        "occupancy gamma            = NOT YET FROZEN"
    )

    print(
        "class policy               = NOT YET FROZEN"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "formal outcomes read       = NO"
    )

    print(
        "training/recalibration     = NO / NO"
    )

    print(
        "new MC sampling            = NO"
    )

    print(
        "fresh Stage6 regression    = PASS | 127"
    )

    print(
        "config SHA256              =",
        config_sha,
    )

    print(
        "report SHA256              =",
        report_sha,
    )

    print(
        "freeze SHA256              =",
        freeze_sha,
    )

    print(
        "handoff SHA256             =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.5 "
        "DEVELOPMENT GAMMA + MULTI-ACTOR MASK AGGREGATION"
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
        "BLOCK6.4 PART2B/2 — CONTROLLED BLOCK"
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
        "numeric freeze created = NO/INCOMPLETE"
    )

    print(
        "formal evaluation      = NO"
    )

    print(
        "formal tuning          = NO"
    )

    print(
        "training/recalibration = NO / NO"
    )

    print(
        "source modification    = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Deliberately no sys.exit().


if __name__ == "__main__":
    main()

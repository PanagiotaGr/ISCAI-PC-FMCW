from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

import numpy as np
import torch


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

PART1 = (
    STAGE4
    / "reports/"
      "block410_part1_acceptance_audit.json"
)

BLOCK48 = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation.json"
)

BLOCK49 = (
    STAGE4
    / "reports/"
      "block49_reproducibility.json"
)

REPRO_MANIFEST = (
    STAGE4
    / "artifacts/block49/"
      "stage4_reproducibility_manifest.json"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

DET_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_gru.pt"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/"
      "covariance_scaler.json"
)

GMM_CHECKPOINT = (
    STAGE4
    / "artifacts/block46/"
      "gmm_gru.pt"
)

FINAL_CLOSURE = (
    STAGE4
    / "reports/"
      "stage4_final_closure.json"
)

HANDOFF = (
    STAGE4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

FREEZE_MANIFEST = (
    STAGE4
    / "artifacts/block410/"
      "stage4_final_freeze_manifest.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "stage4_final_closure_failure.json"
)

IMPLEMENTATION_LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
)


EXPECTED = {
    "part1_implementation":
        (
            "f2b85dcf8d533a9525ecff8ed823bbf0"
            "6783785045fcff6adf626a1c1593ce41"
        ),

    "block49_implementation":
        (
            "e1c779c05155dd86794f405a8567c994"
            "f0acac617015f717944e5ff6868491cd"
        ),

    "reproducibility_manifest":
        (
            "135a3c60e5a2c4419125fa0daa6d2d554"
            "52f923255a04698f0bc1aaca5f474b8"
        ),

    "formal_manifest":
        (
            "2208e7287ddf6439fda4597c435a9cba"
            "1d1b9d0e4c4547bc5dd92e56e8124e46"
        ),

    "validation_manifest":
        (
            "dc10609ef18a2ba881657eb3da3a3df7"
            "a81bdcc8345ecbc2227102ab16b8833c"
        ),

    "normalization":
        (
            "3d7fc0a66d4a4f566f6569befa9c3766"
            "ecae21830bb4e46256df2f326a82a5f6"
        ),

    "deterministic_checkpoint":
        (
            "5456a76b84d558e9983a59b9f1d3060b"
            "a245e0d36d60883519654f809996dbc5"
        ),

    "Gaussian_checkpoint":
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        ),

    "calibrator":
        (
            "508ff2e3fbcfafe8e001155340c25baaf"
            "3772fe2561a8022a9ed1cf780e66087"
        ),

    "GMM_checkpoint":
        (
            "5aebeaf40d522d58345d424ae2557d6f"
            "87e2c02bdc26c23635cc6e3a28cbe3ee"
        ),

    "formal_content":
        (
            "fe5cf7e413b8142c0670d05727affd4f"
            "a573cf4607665e4a9bda951f1d9bd0d5"
        ),

    "formal_scenarios":
        120,

    "formal_matched":
        494,

    "formal_eligible":
        547,

    "regression":
        130,
}


def file_sha256(
    path,
):
    digest = sha256()

    with Path(
        path
    ).open(
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


def atomic_json(
    path,
    payload,
):
    path = Path(
        path
    )

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
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    files = []

    for root in roots:
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
                    STAGE4
                )
            )
    )

    digest = sha256()

    for path in files:
        relative = str(
            path.relative_to(
                STAGE4
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


def environment_snapshot():
    return {
        "python":
            platform.python_version(),

        "numpy":
            np.__version__,

        "torch":
            torch.__version__,

        "torch_cuda":
            torch.version.cuda,

        "CUDA_available":
            bool(
                torch.cuda.is_available()
            ),

        "GPU":
            (
                torch.cuda.get_device_name(
                    0
                )
                if
                torch.cuda.is_available()
                else
                None
            ),

        "CUBLAS_WORKSPACE_CONFIG":
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
    }


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


def verify_hash(
    path,
    expected,
    name,
):
    actual = file_sha256(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} SHA changed: {actual}"
        )

    return actual


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    output = (
        process.stdout
        +
        "\n"
        +
        process.stderr
    )

    print(
        output
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        output,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        process.returncode != 0
        or
        count
        !=
        EXPECTED[
            "regression"
        ]
    ):
        raise RuntimeError(
            "Final Stage4 regression "
            f"expected 130/130; got {count}."
        )

    return count


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.10 FINAL FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK48,
        BLOCK49,
        REPRO_MANIFEST,
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
        NORMALIZATION,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
        GMM_CHECKPOINT,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing required frozen artifact(s): "
            +
            ", ".join(
                missing
            )
        )

    # ========================================================
    # A. Upstream closure evidence
    # ========================================================

    print()
    print(
        "===== A. FINAL UPSTREAM CLOSURE VERIFICATION ====="
    )

    part1 = json.loads(
        PART1.read_text(
            encoding="utf-8"
        )
    )

    block48 = json.loads(
        BLOCK48.read_text(
            encoding="utf-8"
        )
    )

    block49 = json.loads(
        BLOCK49.read_text(
            encoding="utf-8"
        )
    )

    reproducibility = json.loads(
        REPRO_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS",
        "Block4.10 Part1 is not PASS.",
    )

    require(
        part1[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED[
            "part1_implementation"
        ],
        (
            "Block4.10 Part1 historical "
            "implementation SHA changed."
        ),
    )

    matrix = part1[
        "acceptance_matrix"
    ]

    require(
        bool(
            matrix[
                "closure_ready"
            ]
        ),
        "Part1 closure_ready is false.",
    )

    require(
        int(
            matrix[
                "mandatory_count"
            ]
        )
        ==
        14,
        "Mandatory criterion count changed.",
    )

    require(
        int(
            matrix[
                "mandatory_pass_count"
            ]
        )
        ==
        14,
        "Not all 14 mandatory criteria pass.",
    )

    require(
        not matrix[
            "mandatory_failures"
        ],
        "Part1 has mandatory closure failures.",
    )

    require(
        block48.get(
            "status"
        )
        ==
        "PASS",
        "Block4.8 is not PASS.",
    )

    require(
        block49.get(
            "status"
        )
        ==
        "PASS",
        "Block4.9 is not PASS.",
    )

    require(
        block49[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED[
            "block49_implementation"
        ],
        "Frozen Block4.9 implementation changed.",
    )

    require(
        reproducibility.get(
            "status"
        )
        ==
        "FROZEN_REPRODUCIBLE",
        (
            "Stage4 reproducibility manifest "
            "is not FROZEN_REPRODUCIBLE."
        ),
    )

    print(
        "Block4.10 Part1          = PASS"
    )

    print(
        "mandatory acceptance     = 14 / 14 PASS"
    )

    print(
        "Block4.8 formal eval     = PASS"
    )

    print(
        "Block4.9 reproducibility = PASS"
    )

    # ========================================================
    # B. Final immutable artifact hashes
    # ========================================================

    print()
    print(
        "===== B. FINAL IMMUTABLE ARTIFACT HASHES ====="
    )

    artifacts = {
        "formal_manifest":
            verify_hash(
                FORMAL_MANIFEST,
                EXPECTED[
                    "formal_manifest"
                ],
                "formal manifest",
            ),

        "validation_manifest":
            verify_hash(
                VALIDATION_MANIFEST,
                EXPECTED[
                    "validation_manifest"
                ],
                "validation manifest",
            ),

        "normalization":
            verify_hash(
                NORMALIZATION,
                EXPECTED[
                    "normalization"
                ],
                "fit-only normalization",
            ),

        "deterministic_checkpoint":
            verify_hash(
                DET_CHECKPOINT,
                EXPECTED[
                    "deterministic_checkpoint"
                ],
                "deterministic checkpoint",
            ),

        "Gaussian_checkpoint":
            verify_hash(
                GAUSSIAN_CHECKPOINT,
                EXPECTED[
                    "Gaussian_checkpoint"
                ],
                "Gaussian checkpoint",
            ),

        "calibrator":
            verify_hash(
                CALIBRATOR,
                EXPECTED[
                    "calibrator"
                ],
                "covariance calibrator",
            ),

        "GMM_checkpoint":
            verify_hash(
                GMM_CHECKPOINT,
                EXPECTED[
                    "GMM_checkpoint"
                ],
                "GMM checkpoint",
            ),

        "reproducibility_manifest":
            verify_hash(
                REPRO_MANIFEST,
                EXPECTED[
                    "reproducibility_manifest"
                ],
                "reproducibility manifest",
            ),
    }

    for name in artifacts:
        print(
            f"{name:29s}= PASS"
        )

    # ========================================================
    # C. Formal scientific invariants
    # ========================================================

    print()
    print(
        "===== C. FINAL SCIENTIFIC INVARIANTS ====="
    )

    formal_population = (
        block48[
            "formal_population"
        ]
    )

    availability = (
        block48[
            "availability"
        ]
    )

    comparison = (
        block48[
            "comparison"
        ]
    )

    calibration = (
        block48[
            "formal_calibration_generalization"
        ]
    )

    acceptance = (
        block48[
            "Stage4_acceptance_evidence"
        ]
    )

    require(
        int(
            formal_population[
                "scenario_count"
            ]
        )
        ==
        EXPECTED[
            "formal_scenarios"
        ],
        "Formal scenario count changed.",
    )

    require(
        int(
            availability[
                "matched_tracks"
            ]
        )
        ==
        EXPECTED[
            "formal_matched"
        ],
        "Formal matched-target count changed.",
    )

    require(
        int(
            availability[
                "eligible_truth_tracks"
            ]
        )
        ==
        EXPECTED[
            "formal_eligible"
        ],
        "Formal eligible-target count changed.",
    )

    require(
        block48[
            "reproducibility"
        ][
            "merged_formal_content_sha256"
        ]
        ==
        EXPECTED[
            "formal_content"
        ],
        "Formal output content SHA changed.",
    )

    require(
        bool(
            acceptance[
                "PDF_probabilistic_beats_at_least_one_classical"
            ]
        ),
        "PDF performance gate failed.",
    )

    require(
        bool(
            acceptance[
                "PDF_calibration_measured"
            ]
        ),
        "PDF calibration evidence missing.",
    )

    require(
        bool(
            acceptance[
                "internal_CV_gate"
            ]
        ),
        "Internal Gaussian<CV gate failed.",
    )

    require(
        float(
            calibration[
                "calibrated_macro_ECE"
            ]
        )
        <
        float(
            calibration[
                "raw_macro_ECE"
            ]
        ),
        "Formal calibration ECE did not improve.",
    )

    print(
        "formal scenarios         = 120 PASS"
    )

    print(
        "matched / eligible       = 494 / 547 PASS"
    )

    print(
        "calibrated Gaussian ADE  =",
        comparison[
            "calibrated_Gaussian_ADE_m"
        ],
    )

    print(
        "frozen CV ADE            =",
        comparison[
            "frozen_CV_ADE_m"
        ],
    )

    print(
        "classical baselines beaten =",
        comparison[
            "classical_baselines_beaten"
        ],
    )

    print(
        "raw → calibrated ECE     =",
        calibration[
            "raw_macro_ECE"
        ],
        "→",
        calibration[
            "calibrated_macro_ECE"
        ],
    )

    print(
        "formal scientific gates  = PASS"
    )

    # ========================================================
    # D. Leakage invariants
    # ========================================================

    print()
    print(
        "===== D. FINAL LEAKAGE INVARIANTS ====="
    )

    leakage = (
        block48[
            "leakage_contract"
        ]
    )

    false_keys = (
        "training_on_formal",
        "model_selection_on_formal",
        "normalization_fit_on_formal",
        "calibrator_fit_on_formal",
        "tracks_to_predict_model_input",
        "future_model_input",
    )

    for key in false_keys:
        require(
            not bool(
                leakage[
                    key
                ]
            ),
            f"Leakage flag became true: {key}",
        )

    require(
        bool(
            leakage[
                "tracks_to_predict_accessed_after_prediction"
            ]
        ),
        (
            "tracks_to_predict evaluator-only "
            "ordering contract failed."
        ),
    )

    print(
        "formal training            = NO"
    )

    print(
        "formal model selection     = NO"
    )

    print(
        "formal recalibration       = NO"
    )

    print(
        "future model input         = NO"
    )

    print(
        "tracks_to_predict input    = NO"
    )

    print(
        "tracks_to_predict ordering = AFTER PREDICTION PASS"
    )

    print(
        "leakage contract           = PASS"
    )

    # ========================================================
    # E. Read final calibrator contract
    # ========================================================

    print()
    print(
        "===== E. CALIBRATED GAUSSIAN HANDOFF CONTRACT ====="
    )

    calibrator = json.loads(
        CALIBRATOR.read_text(
            encoding="utf-8"
        )
    )

    variance_scale = [
        float(
            value
        )
        for value in (
            calibrator[
                "variance_scale"
            ]
        )
    ]

    std_scale = [
        float(
            value
        )
        for value in (
            calibrator[
                "standard_deviation_scale"
            ]
        )
    ]

    require(
        len(
            variance_scale
        )
        ==
        4,
        "Expected four covariance scales.",
    )

    require(
        len(
            std_scale
        )
        ==
        4,
        "Expected four standard-deviation scales.",
    )

    require(
        all(
            value > 0.0
            for value in variance_scale
        ),
        "Calibration variance scale is non-positive.",
    )

    print(
        "default downstream family = CALIBRATED GAUSSIAN"
    )

    print(
        "horizons s                = [0.1, 0.3, 0.5, 1.0]"
    )

    print(
        "variance scales alpha_h   =",
        variance_scale,
    )

    print(
        "std scales sqrt(alpha_h)  =",
        std_scale,
    )

    print(
        "predictive mean modified  = NO"
    )

    print(
        "measurement R_t modified  = NO"
    )

    print(
        "predictive covariance     = CALIBRATED"
    )

    # ========================================================
    # F. Final regression BEFORE writing freeze artifacts
    # ========================================================

    print()
    print(
        "===== F. FINAL PRE-FREEZE REGRESSION ====="
    )

    regression_count = (
        run_regression()
    )

    print(
        "full Stage4 regression = "
        "130 / 130 PASS"
    )

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    environment = (
        environment_snapshot()
    )

    # ========================================================
    # G. Stage4 -> Stage5 frozen handoff
    # ========================================================

    print()
    print(
        "===== G. WRITING STAGE4 → STAGE5 HANDOFF ====="
    )

    handoff = {
        "project":
            "Agni",

        "from_stage":
            4,

        "to_stage":
            5,

        "status":
            "FROZEN_HANDOFF",

        "Stage4_role":
            (
                "causal uncertainty-aware "
                "trajectory posterior"
            ),

        "default_downstream_posterior": {
            "family":
                "calibrated_Gaussian_GRU",

            "mandatory_default":
                True,

            "checkpoint_path":
                str(
                    GAUSSIAN_CHECKPOINT
                ),

            "checkpoint_sha256":
                EXPECTED[
                    "Gaussian_checkpoint"
                ],

            "normalization_path":
                str(
                    NORMALIZATION
                ),

            "normalization_sha256":
                EXPECTED[
                    "normalization"
                ],

            "calibrator_path":
                str(
                    CALIBRATOR
                ),

            "calibrator_sha256":
                EXPECTED[
                    "calibrator"
                ],

            "horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "predictive_mean":
                "unchanged_by_calibration",

            "predictive_covariance":
                (
                    "Sigma_cal,h = "
                    "alpha_h * Sigma_raw,h"
                ),

            "variance_scale_alpha_h":
                variance_scale,

            "standard_deviation_scale":
                std_scale,

            "covariance_family":
                "full_3D_SPD",

            "covariance_parameterization":
                "Cholesky",

            "formal_covariance_SPD":
                True,
        },

        "measurement_predictive_uncertainty_separation": {
            "measurement_uncertainty":
                "Stage2_R_t_model_input",

            "predictive_uncertainty":
                (
                    "Stage4_learned_Gaussian_"
                    "covariance_output"
                ),

            "calibration_applies_to":
                "predictive_covariance_only",

            "measurement_R_t_modified_by_calibration":
                False,

            "separate":
                True,
        },

        "causal_input_contract": {
            "history_frames":
                11,

            "history_feature_dimension":
                14,

            "measurement_covariance_feature_indices":
                [
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                ],

            "maximum_neighbors":
                8,

            "map_context_dimension":
                10,

            "future_as_model_input":
                False,

            "tracks_to_predict_as_model_input":
                False,

            "perfect_track_ID_as_numeric_feature":
                False,

            "observation_mode":
                "full_frozen_Stage2_degraded",

            "association":
                "frozen_Stage3_estimated_GNN",
        },

        "alternative_frozen_models": {
            "deterministic_GRU": {
                "role":
                    "deterministic_baseline",

                "checkpoint_path":
                    str(
                        DET_CHECKPOINT
                    ),

                "checkpoint_sha256":
                    EXPECTED[
                        "deterministic_checkpoint"
                    ],
            },

            "GMM_GRU": {
                "role":
                    (
                        "multimodal_ablation_"
                        "diagnostic"
                    ),

                "checkpoint_path":
                    str(
                        GMM_CHECKPOINT
                    ),

                "checkpoint_sha256":
                    EXPECTED[
                        "GMM_checkpoint"
                    ],

                "components":
                    3,

                "trajectory_level_modes":
                    True,

                "calibrated":
                    False,

                "selected_as_default_downstream":
                    False,

                "selection_policy":
                    (
                        "do_not_select_post_hoc_"
                        "without_new_preregistered_"
                        "scientific_reason"
                    ),
            },
        },

        "formal_evidence": {
            "population":
                "immutable_Stage3_N120",

            "scenario_count":
                120,

            "eligible_targets":
                547,

            "matched_targets":
                494,

            "matched_target_recall":
                availability[
                    "matched_target_recall"
                ],

            "calibrated_Gaussian_ADE_m":
                comparison[
                    "calibrated_Gaussian_ADE_m"
                ],

            "Gaussian_FDE_1s_m":
                block48[
                    "neural_models"
                ][
                    "calibrated_Gaussian_GRU"
                ][
                    "point_metrics"
                ][
                    "fde_1.0s_m"
                ],

            "raw_macro_ECE":
                calibration[
                    "raw_macro_ECE"
                ],

            "calibrated_macro_ECE":
                calibration[
                    "calibrated_macro_ECE"
                ],

            "raw_Gaussian_NLL":
                calibration[
                    "raw_metric_Gaussian_NLL"
                ],

            "calibrated_Gaussian_NLL":
                calibration[
                    "calibrated_metric_Gaussian_NLL"
                ],

            "classical_baselines_beaten":
                comparison[
                    "classical_baselines_beaten"
                ],

            "PDF_performance_gate":
                True,

            "PDF_calibration_measured":
                True,

            "internal_Gaussian_less_than_CV":
                True,

            "formal_output_content_sha256":
                EXPECTED[
                    "formal_content"
                ],
        },

        "reproducibility": {
            "manifest_path":
                str(
                    REPRO_MANIFEST
                ),

            "manifest_sha256":
                EXPECTED[
                    "reproducibility_manifest"
                ],

            "fresh_process_exact_repeats":
                "4/4_PASS",

            "full_Stage4_regression":
                "130/130_PASS",
        },

        "Stage5_scope_start": {
            "next_required_layer":
                "receiver_angular_posterior",

            "trajectory_posterior_source":
                "calibrated_Gaussian_GRU",

            "adaptive_TopK_beam_policy":
                "Stage5_scope",

            "communication_beam_and_ADB":
                "distinct_actuators",

            "optical_link_evaluation":
                "later_downstream_scope",

            "DeepSense":
                (
                    "separate_later_mmWave_"
                    "external_validation"
                ),
        },

        "do_not_reopen_without_structural_defect": [
            "Stage4_fit_dev_calibration_split",
            "Stage4_normalization",
            "Stage4_deterministic_checkpoint",
            "Stage4_Gaussian_checkpoint",
            "Stage4_covariance_calibrator",
            "Stage4_GMM_checkpoint",
            "Stage4_formal_N120_results",
            "Stage4_association_contract",
        ],

        "Stage5_must_not": [
            (
                "fit_or_recalibrate_the_Stage4_"
                "trajectory_posterior_on_formal_N120"
            ),
            (
                "reinterpret_tracks_to_predict_"
                "as_a_model_input"
            ),
            (
                "merge_measurement_R_t_and_"
                "predictive_covariance_semantics"
            ),
            (
                "silently_switch_to_GMM_based_"
                "on_formal_performance"
            ),
        ],

        "Stage4_final_implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },
    }

    atomic_json(
        HANDOFF,
        handoff,
    )

    handoff_sha = file_sha256(
        HANDOFF
    )

    print(
        "Stage5 handoff             = WRITTEN"
    )

    print(
        "handoff SHA256             =",
        handoff_sha,
    )

    # ========================================================
    # H. Canonical final closure report
    # ========================================================

    print()
    print(
        "===== H. WRITING CANONICAL STAGE4 CLOSURE ====="
    )

    final_closure = {
        "project":
            "Agni",

        "stage":
            4,

        "status":
            "COMPLETE_FROZEN",

        "closure_block":
            "4.10",

        "scientific_scope_completed": [
            (
                "causal_real_trajectory_PC_FMCW_"
                "observation_interface"
            ),
            "measurement_covariance_R_t_conditioning",
            "deterministic_multi_agent_GRU",
            "full_3D_Gaussian_predictive_posterior",
            "trajectory_uncertainty_calibration",
            "trajectory_level_GMM_extension",
            "measurement_to_predictive_uncertainty_analysis",
            "multi_agent_and_map_ablations",
            "frozen_formal_N120_evaluation",
            "exact_reproducibility_freeze",
        ],

        "mandatory_acceptance": {
            "criteria":
                14,

            "passed":
                14,

            "failures":
                [],

            "closure_ready":
                True,
        },

        "primary_downstream_posterior": {
            "family":
                "calibrated_Gaussian_GRU",

            "horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "variance_scale_alpha_h":
                variance_scale,

            "mean_preserved":
                True,

            "full_3D_SPD_covariance":
                True,

            "measurement_R_t_remains_separate":
                True,
        },

        "formal_N120": {
            "scenarios":
                120,

            "eligible_targets":
                547,

            "matched_targets":
                494,

            "matched_target_recall":
                availability[
                    "matched_target_recall"
                ],

            "calibrated_Gaussian_ADE_m":
                comparison[
                    "calibrated_Gaussian_ADE_m"
                ],

            "frozen_CV_ADE_m":
                comparison[
                    "frozen_CV_ADE_m"
                ],

            "classical_baselines_beaten":
                comparison[
                    "classical_baselines_beaten"
                ],

            "raw_macro_ECE":
                calibration[
                    "raw_macro_ECE"
                ],

            "calibrated_macro_ECE":
                calibration[
                    "calibrated_macro_ECE"
                ],

            "raw_coverage_event_Brier":
                calibration[
                    "raw_coverage_event_Brier"
                ],

            "calibrated_coverage_event_Brier":
                calibration[
                    "calibrated_coverage_event_Brier"
                ],

            "raw_metric_Gaussian_NLL":
                calibration[
                    "raw_metric_Gaussian_NLL"
                ],

            "calibrated_metric_Gaussian_NLL":
                calibration[
                    "calibrated_metric_Gaussian_NLL"
                ],

            "formal_content_sha256":
                EXPECTED[
                    "formal_content"
                ],
        },

        "scientific_acceptance": {
            "probabilistic_beats_at_least_one_classical":
                True,

            "internal_Gaussian_less_than_frozen_CV":
                True,

            "formal_calibration_measured":
                True,

            "formal_calibration_ECE_improved":
                True,

            "measurement_predictive_uncertainty_separated":
                True,

            "GMM_extension_evaluated":
                True,

            "GMM_selected_downstream":
                False,

            "causal_leakage_free":
                True,

            "exact_reproducibility":
                True,
        },

        "formal_leakage": {
            "training_on_formal":
                False,

            "normalization_fit_on_formal":
                False,

            "model_selection_on_formal":
                False,

            "calibrator_fit_on_formal":
                False,

            "future_model_input":
                False,

            "tracks_to_predict_model_input":
                False,

            "tracks_to_predict_accessed_after_prediction":
                True,
        },

        "critical_artifacts": {
            "formal_manifest": {
                "path":
                    str(
                        FORMAL_MANIFEST
                    ),

                "sha256":
                    EXPECTED[
                        "formal_manifest"
                    ],
            },

            "validation_manifest": {
                "path":
                    str(
                        VALIDATION_MANIFEST
                    ),

                "sha256":
                    EXPECTED[
                        "validation_manifest"
                    ],
            },

            "fit_only_normalization": {
                "path":
                    str(
                        NORMALIZATION
                    ),

                "sha256":
                    EXPECTED[
                        "normalization"
                    ],
            },

            "deterministic_checkpoint": {
                "path":
                    str(
                        DET_CHECKPOINT
                    ),

                "sha256":
                    EXPECTED[
                        "deterministic_checkpoint"
                    ],
            },

            "Gaussian_checkpoint": {
                "path":
                    str(
                        GAUSSIAN_CHECKPOINT
                    ),

                "sha256":
                    EXPECTED[
                        "Gaussian_checkpoint"
                    ],
            },

            "covariance_calibrator": {
                "path":
                    str(
                        CALIBRATOR
                    ),

                "sha256":
                    EXPECTED[
                        "calibrator"
                    ],
            },

            "GMM_checkpoint": {
                "path":
                    str(
                        GMM_CHECKPOINT
                    ),

                "sha256":
                    EXPECTED[
                        "GMM_checkpoint"
                    ],
            },

            "reproducibility_manifest": {
                "path":
                    str(
                        REPRO_MANIFEST
                    ),

                "sha256":
                    EXPECTED[
                        "reproducibility_manifest"
                    ],
            },

            "Stage5_handoff": {
                "path":
                    str(
                        HANDOFF
                    ),

                "sha256":
                    handoff_sha,
            },
        },

        "scope_boundary": {
            "Stage4_ends_at":
                (
                    "calibrated_probabilistic_"
                    "trajectory_posterior"
                ),

            "Stage5_begins_with":
                "receiver_angular_posterior",

            "receiver_angular_posterior_done":
                False,

            "adaptive_TopK_beam_done":
                False,

            "optical_ADB_done":
                False,

            "DeepSense_done":
                False,

            "deferred_items_are_missing_Stage4_requirements":
                False,
        },

        "reproducibility": {
            "fresh_process_exact_repeats":
                4,

            "formal_cache_content_verified":
                True,

            "full_regression_tests":
                regression_count,

            "environment":
                environment,
        },

        "final_implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "Stage5_handoff_sha256":
            handoff_sha,

        "training_performed_in_Block410":
            False,

        "formal_inference_performed_in_Block410":
            False,

        "recalibration_performed_in_Block410":
            False,

        "Stage5_started":
            False,
    }

    atomic_json(
        FINAL_CLOSURE,
        final_closure,
    )

    closure_sha = file_sha256(
        FINAL_CLOSURE
    )

    print(
        "canonical closure          = WRITTEN"
    )

    print(
        "closure SHA256             =",
        closure_sha,
    )

    # ========================================================
    # I. Final freeze manifest
    # ========================================================

    print()
    print(
        "===== I. WRITING FINAL FREEZE MANIFEST ====="
    )

    freeze_manifest = {
        "project":
            "Agni",

        "stage":
            4,

        "status":
            "FROZEN",

        "canonical_closure": {
            "path":
                str(
                    FINAL_CLOSURE
                ),

            "sha256":
                closure_sha,
        },

        "Stage5_handoff": {
            "path":
                str(
                    HANDOFF
                ),

            "sha256":
                handoff_sha,
        },

        "reproducibility_manifest": {
            "path":
                str(
                    REPRO_MANIFEST
                ),

            "sha256":
                EXPECTED[
                    "reproducibility_manifest"
                ],
        },

        "formal_output_content_sha256":
            EXPECTED[
                "formal_content"
            ],

        "final_implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "regression":
            "130/130_PASS",

        "scientific_acceptance":
            "14/14_PASS",

        "formal_leakage":
            "NONE",

        "next_stage":
            5,

        "Stage5_started":
            False,
    }

    atomic_json(
        FREEZE_MANIFEST,
        freeze_manifest,
    )

    freeze_sha = file_sha256(
        FREEZE_MANIFEST
    )

    print(
        "freeze manifest            = WRITTEN"
    )

    print(
        "freeze manifest SHA256     =",
        freeze_sha,
    )

    # ========================================================
    # J. Final readback verification
    # ========================================================

    print()
    print(
        "===== J. FINAL READBACK VERIFICATION ====="
    )

    closure_readback = json.loads(
        FINAL_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    handoff_readback = json.loads(
        HANDOFF.read_text(
            encoding="utf-8"
        )
    )

    freeze_readback = json.loads(
        FREEZE_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        closure_readback[
            "status"
        ]
        ==
        "COMPLETE_FROZEN",
        "Final closure readback failed.",
    )

    require(
        handoff_readback[
            "status"
        ]
        ==
        "FROZEN_HANDOFF",
        "Stage5 handoff readback failed.",
    )

    require(
        handoff_readback[
            "default_downstream_posterior"
        ][
            "family"
        ]
        ==
        "calibrated_Gaussian_GRU",
        "Wrong default Stage5 posterior.",
    )

    require(
        freeze_readback[
            "status"
        ]
        ==
        "FROZEN",
        "Freeze manifest readback failed.",
    )

    require(
        freeze_readback[
            "canonical_closure"
        ][
            "sha256"
        ]
        ==
        closure_sha,
        "Closure SHA readback mismatch.",
    )

    require(
        freeze_readback[
            "Stage5_handoff"
        ][
            "sha256"
        ]
        ==
        handoff_sha,
        "Handoff SHA readback mismatch.",
    )

    require(
        file_sha256(
            FINAL_CLOSURE
        )
        ==
        closure_sha,
        "Closure changed after write.",
    )

    require(
        file_sha256(
            HANDOFF
        )
        ==
        handoff_sha,
        "Handoff changed after write.",
    )

    print(
        "canonical closure readback = PASS"
    )

    print(
        "Stage5 handoff readback    = PASS"
    )

    print(
        "freeze-manifest readback   = PASS"
    )

    print(
        "default posterior          = CALIBRATED GAUSSIAN PASS"
    )

    # ========================================================
    # K. Implementation log
    # ========================================================

    marker = (
        "## Stage 4 Final Closure"
    )

    existing = (
        IMPLEMENTATION_LOG.read_text(
            encoding="utf-8"
        )
        if
        IMPLEMENTATION_LOG.exists()
        else
        ""
    )

    if marker not in existing:
        with IMPLEMENTATION_LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:

            stream.write(
                "\n"
                + marker
                + "\n\n"
                "Status: COMPLETE / FROZEN\n\n"
                "- Blocks 4.0 through 4.10 "
                  "completed with all mandatory "
                  "Stage4 acceptance criteria PASS.\n"
                "- Mandatory acceptance matrix: "
                  "14/14 PASS.\n"
                "- Frozen formal evaluation: "
                  "120 scenarios, 494/547 matched "
                  "targets.\n"
                "- Calibrated Gaussian formal ADE: "
                f"{comparison['calibrated_Gaussian_ADE_m']:.9f} m.\n"
                "- Formal macro coverage ECE improved "
                f"from {calibration['raw_macro_ECE']:.9f} "
                f"to {calibration['calibrated_macro_ECE']:.9f}.\n"
                "- Calibrated Gaussian is the frozen "
                  "default Stage5 trajectory posterior.\n"
                "- GMM is preserved as a multimodal "
                  "diagnostic/ablation and is not "
                  "selected downstream post hoc.\n"
                "- Measurement covariance R_t remains "
                  "a model input and is distinct from "
                  "learned predictive covariance.\n"
                "- Fresh-process exact reproducibility: "
                  "4/4 PASS.\n"
                "- Final Stage4 regression: "
                  "130/130 PASS.\n"
                f"- Final Stage4 implementation SHA256: "
                f"{implementation_sha}.\n"
                f"- Canonical closure SHA256: "
                f"{closure_sha}.\n"
                f"- Stage4→Stage5 handoff SHA256: "
                f"{handoff_sha}.\n"
                f"- Final freeze-manifest SHA256: "
                f"{freeze_sha}.\n"
                "- Stage5 has not been started by "
                  "this closure block.\n"
            )

    if FAILURE.exists():
        FAILURE.unlink()

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE 4 FINAL CLOSURE GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Blocks 4.0 → 4.10          = COMPLETE"
    )

    print(
        "mandatory acceptance       = 14 / 14 PASS"
    )

    print(
        "formal N=120               = PASS"
    )

    print(
        "probabilistic > classical  = PASS"
    )

    print(
        "Gaussian < frozen CV       = PASS"
    )

    print(
        "formal calibration         = PASS"
    )

    print(
        "measurement/predictive UQ  = SEPARATE PASS"
    )

    print(
        "GMM extension              = FROZEN DIAGNOSTIC"
    )

    print(
        "causal/leakage contract    = PASS"
    )

    print(
        "exact reproducibility      = PASS"
    )

    print(
        "full Stage4 regression     = 130 / 130 PASS"
    )

    print(
        "default Stage5 posterior   = CALIBRATED GAUSSIAN"
    )

    print(
        "Stage5 handoff             = FROZEN"
    )

    print(
        "training in Block4.10      = NO"
    )

    print(
        "formal inference Block4.10 = NO"
    )

    print(
        "recalibration Block4.10    = NO"
    )

    print(
        "Stage5 started             = NO"
    )

    print(
        "final implementation files =",
        implementation_files,
    )

    print(
        "final implementation SHA256=",
        implementation_sha,
    )

    print(
        "canonical closure SHA256   =",
        closure_sha,
    )

    print(
        "Stage5 handoff SHA256      =",
        handoff_sha,
    )

    print(
        "freeze manifest SHA256     =",
        freeze_sha,
    )

    print(
        "STATUS = COMPLETE_FROZEN"
    )

    print(
        "closure report =",
        FINAL_CLOSURE,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "freeze manifest =",
        FREEZE_MANIFEST,
    )

    print()
    print(
        "===== STAGE 4 FINAL ====="
    )

    print(
        "Stage 4 scientific scope   = COMPLETE"
    )

    print(
        "Stage 4 implementation     = FROZEN"
    )

    print(
        "Stage 4 formal evidence    = FROZEN"
    )

    print(
        "Stage 4 reproducibility    = FROZEN"
    )

    print(
        "Stage 5 interface          = FROZEN"
    )

    print(
        "next stage                 = STAGE 5"
    )

    print(
        "terminal remains open      = YES"
    )


try:
    main()

except BaseException as exc:
    failure = {
        "project":
            "Agni",

        "stage":
            4,

        "block":
            "4.10_part_2",

        "status":
            "BLOCKED",

        "exception_type":
            type(
                exc
            ).__name__,

        "exception":
            str(
                exc
            ),

        "traceback":
            traceback.format_exc(),

        "training":
            False,

        "formal_inference":
            False,

        "recalibration":
            False,

        "Stage5_started":
            False,

        "upstream_modified":
            False,

        "recovery":
            (
                "Inspect only the final Block4.10 "
                "freeze/handoff code or the specific "
                "frozen hash/invariant that failed. "
                "Do not retrain, recalibrate or rerun "
                "the full formal N=120 evaluation."
            ),
    }

    atomic_json(
        FAILURE,
        failure,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE 4 FINAL FREEZE = BLOCKED"
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
    print(
        "RECOVERY:"
    )

    print(
        failure[
            "recovery"
        ]
    )

    print()
    print(
        "training performed      = NO"
    )

    print(
        "formal inference        = NO"
    )

    print(
        "recalibration           = NO"
    )

    print(
        "Stage5 started          = NO"
    )

    print(
        "terminal remains open   = YES"
    )

    print(
        "failure report =",
        FAILURE,
    )

# Deliberately no sys.exit().

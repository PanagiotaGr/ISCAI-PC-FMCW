from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback

from iscai_stage4.ml.closure_runtime import (
    acceptance_matrix,
    discover_frozen_block_report,
    file_sha256,
    require_false,
    require_true,
)


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

REPORTS = (
    STAGE4
    / "reports"
)

OUTPUT = (
    REPORTS
    / "block410_part1_acceptance_audit.json"
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

REPRO_MANIFEST = (
    STAGE4
    / "artifacts/block49/"
      "stage4_reproducibility_manifest.json"
)

NEW_FILES = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "closure_runtime.py",

    STAGE4
    / "tests/"
      "test_block410_closure.py",

    STAGE4
    / "scripts/"
      "run_block410_part1_safe.py",
)


EXPECTED_BLOCK_IMPLEMENTATION_SHA = {
    "4.0":
        (
            "22f27101f5c6e57a1bc66a1299c05be9"
            "11ea6af26a18a0fe36807cfa4e944a4f"
        ),

    "4.1":
        (
            "2021ef79a5f875239e132e404805f4c51"
            "2e45926d0fdac9d2d6c7a5b51eab7eb"
        ),

    "4.2":
        (
            "4f6ef90bd3455bf55f100561d66ef935"
            "49d8472112b85fce45930ec169f89fd0"
        ),

    "4.3":
        (
            "7f42e0cf534ed3ffb8d69e7230f5e04e"
            "ecde8639428911017af771c7b2993def"
        ),

    "4.4":
        (
            "b9ea474443d3a54235f09cd4e2b61637"
            "4b37456f2b9e080f5334f76f4dd7c432"
        ),

    "4.5":
        (
            "482fa179b958f05526657d9970346b3ca"
            "699c649b77783f63d4c8de2fd7be7cf"
        ),

    "4.6":
        (
            "bf7ba0ef8c83d0b6a8d5ef2acd624190"
            "35f4367b17d4d6844f7a99c288f1c961"
        ),

    "4.7":
        (
            "4b32edff0c08f23de4ec5109b46b64a8"
            "eccc5357da0deb4fc516b72fe3cad006"
        ),

    "4.8":
        (
            "f21abaed100d1337ad95e6005298727ab"
            "5e103d10d3a173b7d2e561d886dc88d"
        ),

    "4.9":
        (
            "e1c779c05155dd86794f405a8567c994"
            "f0acac617015f717944e5ff6868491cd"
        ),
}


EXPECTED_ARTIFACT_SHA = {
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

    "reproducibility_manifest":
        (
            "135a3c60e5a2c4419125fa0daa6d2d554"
            "52f923255a04698f0bc1aaca5f474b8"
        ),

    "formal_content":
        (
            "fe5cf7e413b8142c0670d05727affd4f"
            "a573cf4607665e4a9bda951f1d9bd0d5"
        ),
}


def write_json(
    payload,
):
    OUTPUT.write_text(
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


def blocked(
    phase,
    reason,
    recovery,
):
    write_json({
        "stage":
            4,

        "block":
            "4.10_part_1",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

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
    })

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.10 PART 1/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "phase =",
        phase,
    )

    print(
        "reason =",
        reason,
    )

    print(
        "recovery =",
        recovery,
    )

    print(
        "training            = NO"
    )

    print(
        "formal inference    = NO"
    )

    print(
        "recalibration       = NO"
    )

    print(
        "Stage5 started      = NO"
    )

    print(
        "terminal remains open = YES"
    )


def current_implementation_fingerprint():
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


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.10 PART 1/2 "
        "FINAL ACCEPTANCE AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
        NORMALIZATION,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
        GMM_CHECKPOINT,
        REPRO_MANIFEST,
        *NEW_FILES,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_files",
            (
                "Missing: "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore/audit only the "
                "missing frozen artifact."
            ),
        )
        return

    # ========================================================
    # A. Compile before reading scientific evidence.
    # ========================================================

    print()
    print(
        "===== A. CONTROLLED COMPILE ====="
    )

    for path in NEW_FILES:
        try:
            py_compile.compile(
                str(
                    path
                ),
                doraise=True,
            )

            print(
                path.name,
                "= PASS",
            )

        except Exception as exc:
            blocked(
                "compile",
                (
                    f"{path.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                (
                    "Repair only new "
                    "Block4.10 Part1 code."
                ),
            )
            return

    # ========================================================
    # B. Discover and verify frozen Block4.0–4.9 chain.
    # ========================================================

    print()
    print(
        "===== B. FROZEN BLOCK4.0 → BLOCK4.9 CHAIN ====="
    )

    frozen_reports = {}

    try:
        for block in (
            "4.0",
            "4.1",
            "4.2",
            "4.3",
            "4.4",
            "4.5",
            "4.6",
            "4.7",
            "4.8",
            "4.9",
        ):
            discovered = (
                discover_frozen_block_report(
                    REPORTS,
                    block=block,
                    expected_implementation_sha=(
                        EXPECTED_BLOCK_IMPLEMENTATION_SHA[
                            block
                        ]
                    ),
                )
            )

            frozen_reports[
                block
            ] = discovered

            print(
                f"Block {block:3s} = PASS |",
                discovered[
                    "path"
                ].name,
            )

    except Exception as exc:
        blocked(
            "frozen_report_chain",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Do not infer a missing "
                "closure report. Audit only "
                "the affected frozen Block."
            ),
        )
        return

    block48 = frozen_reports[
        "4.8"
    ][
        "payload"
    ]

    block49 = frozen_reports[
        "4.9"
    ][
        "payload"
    ]

    # ========================================================
    # C. Independent critical artifact hashes.
    # ========================================================

    print()
    print(
        "===== C. CRITICAL FROZEN ARTIFACTS ====="
    )

    artifact_checks = (
        (
            "formal manifest",
            FORMAL_MANIFEST,
            EXPECTED_ARTIFACT_SHA[
                "formal_manifest"
            ],
        ),
        (
            "validation manifest",
            VALIDATION_MANIFEST,
            EXPECTED_ARTIFACT_SHA[
                "validation_manifest"
            ],
        ),
        (
            "fit-only normalization",
            NORMALIZATION,
            EXPECTED_ARTIFACT_SHA[
                "normalization"
            ],
        ),
        (
            "deterministic checkpoint",
            DET_CHECKPOINT,
            EXPECTED_ARTIFACT_SHA[
                "deterministic_checkpoint"
            ],
        ),
        (
            "Gaussian checkpoint",
            GAUSSIAN_CHECKPOINT,
            EXPECTED_ARTIFACT_SHA[
                "Gaussian_checkpoint"
            ],
        ),
        (
            "covariance calibrator",
            CALIBRATOR,
            EXPECTED_ARTIFACT_SHA[
                "calibrator"
            ],
        ),
        (
            "GMM checkpoint",
            GMM_CHECKPOINT,
            EXPECTED_ARTIFACT_SHA[
                "GMM_checkpoint"
            ],
        ),
        (
            "Stage4 reproducibility manifest",
            REPRO_MANIFEST,
            EXPECTED_ARTIFACT_SHA[
                "reproducibility_manifest"
            ],
        ),
    )

    artifact_hashes = {}

    for name, path, expected in (
        artifact_checks
    ):
        actual = file_sha256(
            path
        )

        artifact_hashes[
            name
        ] = actual

        if actual != expected:
            blocked(
                "artifact_hash",
                (
                    f"{name} SHA changed: "
                    f"{actual}"
                ),
                (
                    "Do not regenerate or "
                    "replace frozen evidence."
                ),
            )
            return

        print(
            f"{name:31s}= PASS"
        )

    # ========================================================
    # D. Reproducibility-manifest invariants.
    # ========================================================

    print()
    print(
        "===== D. REPRODUCIBILITY MANIFEST ====="
    )

    reproducibility = json.loads(
        REPRO_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    if (
        reproducibility.get(
            "status"
        )
        !=
        "FROZEN_REPRODUCIBLE"
    ):
        blocked(
            "reproducibility_manifest",
            (
                "Stage4 reproducibility "
                "manifest is not frozen."
            ),
            (
                "Do not proceed to "
                "Stage4 final closure."
            ),
        )
        return

    formal_population = (
        reproducibility[
            "formal_population"
        ]
    )

    if (
        int(
            formal_population[
                "N"
            ]
        )
        !=
        120
        or
        int(
            formal_population[
                "matched_targets"
            ]
        )
        !=
        494
        or
        int(
            formal_population[
                "eligible_targets"
            ]
        )
        !=
        547
        or
        formal_population[
            "formal_output_content_sha256"
        ]
        !=
        EXPECTED_ARTIFACT_SHA[
            "formal_content"
        ]
    ):
        blocked(
            "formal_reproducibility",
            (
                "Frozen formal population/content "
                "does not match Block4.8/4.9."
            ),
            (
                "Do not reinterpret formal "
                "evaluation."
            ),
        )
        return

    fresh = (
        reproducibility[
            "fresh_process_reproducibility"
        ]
    )

    if (
        int(
            fresh[
                "probe_count"
            ]
        )
        !=
        4
        or
        int(
            fresh[
                "independent_python_processes"
            ]
        )
        !=
        4
        or
        not bool(
            fresh[
                "strict_state_loads"
            ]
        )
        or
        not bool(
            fresh[
                "exact_prediction_SHA_matches"
            ]
        )
        or
        bool(
            fresh[
                "tracks_to_predict_accessed"
            ]
        )
    ):
        blocked(
            "fresh_process_reproducibility",
            (
                "Fresh-process exact "
                "reproducibility contract failed."
            ),
            (
                "Do not proceed to final closure."
            ),
        )
        return

    print(
        "formal N                  = 120 PASS"
    )

    print(
        "formal matched/eligible   = 494 / 547 PASS"
    )

    print(
        "formal output content SHA = PASS"
    )

    print(
        "fresh-process probes      = 4 / 4 PASS"
    )

    print(
        "exact prediction SHA      = 4 / 4 PASS"
    )

    # ========================================================
    # E. Formal scientific evidence.
    # ========================================================

    print()
    print(
        "===== E. FORMAL SCIENTIFIC EVIDENCE ====="
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

    availability = (
        block48[
            "availability"
        ]
    )

    Gaussian_ADE = float(
        comparison[
            "calibrated_Gaussian_ADE_m"
        ]
    )

    CV_ADE = float(
        comparison[
            "frozen_CV_ADE_m"
        ]
    )

    matched_recall = float(
        availability[
            "matched_target_recall"
        ]
    )

    if abs(
        matched_recall
        -
        (
            494.0
            /
            547.0
        )
    ) > 1e-12:
        blocked(
            "formal_recall",
            (
                "Formal matched-target recall "
                "does not equal 494/547."
            ),
            "Audit Block4.8 report.",
        )
        return

    require_true(
        acceptance[
            "PDF_probabilistic_beats_at_least_one_classical"
        ],
        name=(
            "probabilistic predictor beats "
            "at least one classical baseline"
        ),
    )

    require_true(
        acceptance[
            "PDF_calibration_measured"
        ],
        name="formal calibration measured",
    )

    require_true(
        acceptance[
            "internal_CV_gate"
        ],
        name="calibrated Gaussian ADE < frozen CV ADE",
    )

    require_true(
        Gaussian_ADE
        <
        CV_ADE,
        name="numeric Gaussian < CV cross-check",
    )

    beaten = tuple(
        comparison[
            "classical_baselines_beaten"
        ]
    )

    if not beaten:
        blocked(
            "classical_comparison",
            (
                "No frozen classical baseline "
                "is beaten."
            ),
            "Do not change gate post hoc.",
        )
        return

    raw_ECE = float(
        calibration[
            "raw_macro_ECE"
        ]
    )

    calibrated_ECE = float(
        calibration[
            "calibrated_macro_ECE"
        ]
    )

    require_true(
        calibration[
            "ECE_nonworsening"
        ],
        name="formal calibration ECE non-worsening",
    )

    require_true(
        calibrated_ECE
        <
        raw_ECE,
        name="formal calibration ECE improvement",
    )

    print(
        "calibrated Gaussian ADE m =",
        Gaussian_ADE,
    )

    print(
        "frozen CV ADE m           =",
        CV_ADE,
    )

    print(
        "classical baselines beaten=",
        list(
            beaten
        ),
    )

    print(
        "matched-target recall     =",
        matched_recall,
    )

    print(
        "raw formal ECE            =",
        raw_ECE,
    )

    print(
        "calibrated formal ECE     =",
        calibrated_ECE,
    )

    print(
        "PDF performance gate      = PASS"
    )

    print(
        "PDF calibration gate      = PASS"
    )

    print(
        "internal CV gate          = PASS"
    )

    # ========================================================
    # F. Leakage contract.
    # ========================================================

    print()
    print(
        "===== F. CAUSAL / LEAKAGE CONTRACT ====="
    )

    leakage48 = (
        block48[
            "leakage_contract"
        ]
    )

    forbidden_true = (
        "training_on_formal",
        "model_selection_on_formal",
        "normalization_fit_on_formal",
        "calibrator_fit_on_formal",
        "tracks_to_predict_model_input",
        "future_model_input",
    )

    for key in forbidden_true:
        require_false(
            leakage48[
                key
            ],
            name=key,
        )

    require_true(
        leakage48[
            "tracks_to_predict_accessed_after_prediction"
        ],
        name=(
            "tracks_to_predict evaluator-only "
            "after predictions"
        ),
    )

    leakage49 = (
        reproducibility[
            "leakage"
        ]
    )

    for key, value in leakage49.items():
        require_false(
            value,
            name=(
                "reproducibility leakage."
                f"{key}"
            ),
        )

    print(
        "future model input          = NO"
    )

    print(
        "tracks_to_predict input     = NO"
    )

    print(
        "tracks_to_predict ordering  = AFTER PREDICTION PASS"
    )

    print(
        "formal training             = NO"
    )

    print(
        "formal normalization fit    = NO"
    )

    print(
        "formal model selection      = NO"
    )

    print(
        "formal calibrator fit       = NO"
    )

    print(
        "causal/leakage contract     = PASS"
    )

    # ========================================================
    # G. Scientific/scope acceptance matrix.
    # ========================================================

    print()
    print(
        "===== G. STAGE4 ACCEPTANCE MATRIX ====="
    )

    GMM_report = (
        block48[
            "neural_models"
        ][
            "GMM_ablation"
        ]
    )

    calibrated_Gaussian_report = (
        block48[
            "neural_models"
        ][
            "calibrated_Gaussian_GRU"
        ]
    )

    criteria = [
        {
            "id":
                "causal_real_PC_FMCW_observation_interface",

            "mandatory":
                True,

            "pass":
                (
                    block48[
                        "observation_interface"
                    ][
                        "Stage2"
                    ]
                    ==
                    "frozen_full_degraded"
                    and
                    block48[
                        "observation_interface"
                    ][
                        "association"
                    ]
                    ==
                    "frozen_Stage3_estimated_GNN"
                ),

            "evidence":
                (
                    "Frozen Stage2 degraded "
                    "observations + Stage3 GNN"
                ),
        },

        {
            "id":
                "measurement_covariance_R_t_is_input",

            "mandatory":
                True,

            "pass":
                (
                    block48[
                        "observation_interface"
                    ][
                        "measurement_covariance_R_t"
                    ]
                    ==
                    "real_input"
                ),

            "evidence":
                "Block4.2 / Block4.7",
        },

        {
            "id":
                "deterministic_GRU_baseline",

            "mandatory":
                True,

            "pass":
                (
                    "deterministic_GRU"
                    in
                    block48[
                        "neural_models"
                    ]
                ),

            "evidence":
                "Block4.3 + formal Block4.8",
        },

        {
            "id":
                "Gaussian_probabilistic_predictor",

            "mandatory":
                True,

            "pass":
                (
                    "raw_Gaussian_GRU"
                    in
                    block48[
                        "neural_models"
                    ]
                    and
                    bool(
                        calibrated_Gaussian_report[
                            "covariance_SPD"
                        ]
                    )
                ),

            "evidence":
                "Full 3D Cholesky Gaussian / SPD",
        },

        {
            "id":
                "trajectory_uncertainty_calibration",

            "mandatory":
                True,

            "pass":
                (
                    bool(
                        acceptance[
                            "PDF_calibration_measured"
                        ]
                    )
                    and
                    bool(
                        calibration[
                            "ECE_nonworsening"
                        ]
                    )
                ),

            "evidence":
                (
                    "Calibration partition fit; "
                    "formal generalization measured"
                ),
        },

        {
            "id":
                "GMM_multimodal_extension",

            "mandatory":
                True,

            "pass":
                (
                    GMM_report[
                        "minADE_m"
                    ]
                    is not None
                    and
                    GMM_report[
                        "minFDE_1.0s_m"
                    ]
                    is not None
                ),

            "evidence":
                (
                    "3-component trajectory-level "
                    "GMM evaluated formally"
                ),
        },

        {
            "id":
                "GMM_not_selected_post_hoc",

            "mandatory":
                True,

            "pass":
                (
                    not bool(
                        GMM_report[
                            "downstream_selected"
                        ]
                    )
                ),

            "evidence":
                (
                    "Calibrated Gaussian remains "
                    "mandatory/default posterior"
                ),
        },

        {
            "id":
                "measurement_to_predictive_uncertainty_analysis",

            "mandatory":
                True,

            "pass":
                (
                    frozen_reports[
                        "4.7"
                    ][
                        "payload"
                    ].get(
                        "status"
                    )
                    ==
                    "PASS"
                ),

            "evidence":
                (
                    "R_t analysis + physical "
                    "sensitivity + retrained ablations"
                ),
        },

        {
            "id":
                "multi_agent_and_map_ablation",

            "mandatory":
                True,

            "pass":
                (
                    frozen_reports[
                        "4.7"
                    ][
                        "payload"
                    ].get(
                        "status"
                    )
                    ==
                    "PASS"
                ),

            "evidence":
                "actor-only and no-map retraining",
        },

        {
            "id":
                "frozen_formal_N120_evaluation",

            "mandatory":
                True,

            "pass":
                (
                    int(
                        block48[
                            "formal_population"
                        ][
                            "scenario_count"
                        ]
                    )
                    ==
                    120
                    and
                    bool(
                        block48[
                            "formal_population"
                        ][
                            "same_frozen_population"
                        ]
                    )
                ),

            "evidence":
                "Immutable Stage3 N=120",
        },

        {
            "id":
                "probabilistic_beats_classical",

            "mandatory":
                True,

            "pass":
                bool(
                    acceptance[
                        "PDF_probabilistic_beats_at_least_one_classical"
                    ]
                ),

            "evidence":
                list(
                    beaten
                ),
        },

        {
            "id":
                "internal_Gaussian_beats_CV",

            "mandatory":
                True,

            "pass":
                bool(
                    acceptance[
                        "internal_CV_gate"
                    ]
                ),

            "evidence":
                {
                    "Gaussian_ADE_m":
                        Gaussian_ADE,

                    "CV_ADE_m":
                        CV_ADE,
                },
        },

        {
            "id":
                "reproducibility_exact",

            "mandatory":
                True,

            "pass":
                (
                    bool(
                        fresh[
                            "strict_state_loads"
                        ]
                    )
                    and
                    bool(
                        fresh[
                            "exact_prediction_SHA_matches"
                        ]
                    )
                ),

            "evidence":
                "4/4 independent fresh processes",
        },

        {
            "id":
                "causal_leakage_free",

            "mandatory":
                True,

            "pass":
                (
                    not any(
                        bool(
                            leakage48[
                                key
                            ]
                        )
                        for key in forbidden_true
                    )
                    and
                    bool(
                        leakage48[
                            "tracks_to_predict_accessed_after_prediction"
                        ]
                    )
                ),

            "evidence":
                "Block4.8 + Block4.9 leakage contracts",
        },

        # Explicitly NOT Stage4 requirements.
        {
            "id":
                "receiver_angular_posterior",

            "mandatory":
                False,

            "pass":
                False,

            "evidence":
                "Deferred to Stage5 by frozen scope",
        },

        {
            "id":
                "adaptive_TopK_beam_control",

            "mandatory":
                False,

            "pass":
                False,

            "evidence":
                "Deferred to Stage5 by frozen scope",
        },

        {
            "id":
                "optical_link_and_ADB",

            "mandatory":
                False,

            "pass":
                False,

            "evidence":
                "Deferred beyond Stage4",
        },

        {
            "id":
                "DeepSense_external_validation",

            "mandatory":
                False,

            "pass":
                False,

            "evidence":
                "Later separate mmWave validation",
        },
    ]

    matrix = acceptance_matrix(
        criteria
    )

    for row in (
        matrix[
            "rows"
        ]
    ):
        if row[
            "mandatory"
        ]:
            status = (
                "PASS"
                if row[
                    "pass"
                ]
                else
                "BLOCKED"
            )

            print(
                f"{row['id']:44s}= {status}"
            )

    print()
    print(
        "mandatory criteria      =",
        matrix[
            "mandatory_count"
        ],
    )

    print(
        "mandatory criteria PASS =",
        matrix[
            "mandatory_pass_count"
        ],
    )

    print(
        "mandatory failures      =",
        matrix[
            "mandatory_failures"
        ],
    )

    if not matrix[
        "closure_ready"
    ]:
        blocked(
            "acceptance_matrix",
            (
                "Mandatory scientific "
                "closure criterion failed: "
                +
                repr(
                    matrix[
                        "mandatory_failures"
                    ]
                )
            ),
            (
                "Do not weaken or redefine "
                "the frozen acceptance criteria."
            ),
        )
        return

    print(
        "Stage4 closure readiness = PASS"
    )

    # ========================================================
    # H. New tests + complete regression.
    # ========================================================

    print()
    print(
        "===== H. BLOCK4.10 UNIT TESTS ====="
    )

    tests = subprocess.run(
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
            "test_block410_closure.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    output = (
        tests.stdout
        +
        "\n"
        +
        tests.stderr
    )

    print(
        output
    )

    if tests.returncode != 0:
        blocked(
            "Block410_tests",
            "New closure tests failed.",
            (
                "Repair only Block4.10 "
                "Part1 code/tests."
            ),
        )
        return

    print()
    print(
        "===== I. FULL STAGE4 REGRESSION ====="
    )

    full = subprocess.run(
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

    full_output = (
        full.stdout
        +
        "\n"
        +
        full.stderr
    )

    print(
        full_output
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        full_output,
    )

    test_count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        full.returncode != 0
        or
        test_count != 130
    ):
        blocked(
            "regression",
            (
                "Expected 130/130 Stage4 "
                f"tests, detected {test_count}."
            ),
            (
                "Do not freeze final Stage4 "
                "closure until regression passes."
            ),
        )
        return

    (
        implementation_files,
        implementation_sha,
    ) = current_implementation_fingerprint()

    payload = {
        "stage":
            4,

        "block":
            "4.10_part_1",

        "status":
            "PASS",

        "purpose":
            (
                "final_scientific_acceptance_"
                "and_scope_audit"
            ),

        "frozen_blocks": {
            block: {
                "implementation_sha256":
                    EXPECTED_BLOCK_IMPLEMENTATION_SHA[
                        block
                    ],

                "report_path":
                    str(
                        item[
                            "path"
                        ]
                    ),

                "report_file_sha256":
                    item[
                        "file_sha256"
                    ],
            }
            for block, item in (
                frozen_reports.items()
            )
        },

        "critical_artifact_hashes":
            artifact_hashes,

        "formal_summary": {
            "scenarios":
                120,

            "eligible_targets":
                547,

            "matched_targets":
                494,

            "matched_target_recall":
                matched_recall,

            "calibrated_Gaussian_ADE_m":
                Gaussian_ADE,

            "frozen_CV_ADE_m":
                CV_ADE,

            "classical_baselines_beaten":
                list(
                    beaten
                ),

            "raw_macro_ECE":
                raw_ECE,

            "calibrated_macro_ECE":
                calibrated_ECE,

            "formal_content_sha256":
                EXPECTED_ARTIFACT_SHA[
                    "formal_content"
                ],
        },

        "acceptance_matrix":
            matrix,

        "scope_boundary": {
            "Stage4_complete_through":
                (
                    "probabilistic trajectory "
                    "posterior and calibration"
                ),

            "Stage5_next":
                (
                    "receiver/angular posterior "
                    "and beam-policy interface"
                ),

            "receiver_angular_posterior_in_Stage4":
                False,

            "adaptive_TopK_beam_in_Stage4":
                False,

            "optical_ADB_in_Stage4":
                False,

            "DeepSense_in_Stage4":
                False,
        },

        "causal_leakage_contract":
            "PASS",

        "reproducibility":
            "PASS",

        "regression": {
            "tests_passed":
                130,

            "tests_total":
                130,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "training":
            False,

        "formal_inference":
            False,

        "recalibration":
            False,

        "final_Stage4_freeze_written":
            False,

        "Stage5_started":
            False,

        "upstream_modified":
            False,
    }

    write_json(
        payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.10 PART 1/2 SAFE GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block4.0→4.9 frozen chain = PASS"
    )

    print(
        "critical artifact hashes  = PASS"
    )

    print(
        "formal N=120 evidence      = PASS"
    )

    print(
        "probabilistic > classical  = PASS"
    )

    print(
        "Gaussian < frozen CV       = PASS"
    )

    print(
        "formal calibration measured= PASS"
    )

    print(
        "formal ECE improved        = PASS"
    )

    print(
        "measurement/predictive UQ  = PASS"
    )

    print(
        "multi-agent/map ablations  = PASS"
    )

    print(
        "GMM extension              = PASS"
    )

    print(
        "causal/leakage contract    = PASS"
    )

    print(
        "exact reproducibility      = PASS"
    )

    print(
        "mandatory acceptance       =",
        (
            f"{matrix['mandatory_pass_count']} / "
            f"{matrix['mandatory_count']} PASS"
        ),
    )

    print(
        "Stage4 scope boundary      = PASS"
    )

    print(
        "receiver/beam/ADB          = DEFERRED AS PLANNED"
    )

    print(
        "training performed         = NO"
    )

    print(
        "formal inference performed = NO"
    )

    print(
        "recalibration performed    = NO"
    )

    print(
        "final Stage4 freeze written= NO"
    )

    print(
        "Stage5 started             = NO"
    )

    print(
        "full Stage4 regression     = 130 / 130 PASS"
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
        OUTPUT,
    )

    print(
        "terminal remains open      = YES"
    )


try:
    main()

except BaseException as exc:
    blocked(
        "unexpected_controller_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
        (
            "Unexpected final-acceptance "
            "audit failure. No Stage4 freeze "
            "or Stage5 execution occurred."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

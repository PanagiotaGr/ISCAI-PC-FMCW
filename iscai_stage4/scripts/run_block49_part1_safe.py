from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback

import torch

from iscai_stage4.ml.reproducibility_runtime import (
    checkpoint_state_sha256,
    file_sha256,
    formal_cache_content_sha256,
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

REPORTS = {
    "Block4.3":
        STAGE4
        /
        "reports/block43_deterministic_gru.json",

    "Block4.4":
        STAGE4
        /
        "reports/block44_gaussian_gru.json",

    "Block4.5":
        STAGE4
        /
        "reports/block45_calibration.json",

    "Block4.6":
        STAGE4
        /
        "reports/block46_gmm_gru.json",

    "Block4.7":
        STAGE4
        /
        "reports/block47_ablation_analysis.json",

    "Block4.8":
        STAGE4
        /
        "reports/block48_formal_evaluation.json",
}

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

FORMAL_CACHE = (
    STAGE4
    / "artifacts/block48/cache"
)

FORMAL_MERGED = (
    STAGE4
    / "artifacts/block48/"
      "formal_neural_outputs.npz"
)

FORMAL_CACHE_REPORT = (
    STAGE4
    / "artifacts/block48/"
      "formal_cache_manifest.json"
)

OUTPUT = (
    STAGE4
    / "reports/"
      "block49_part1_reproducibility_audit.json"
)

NEW_FILES = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "reproducibility_runtime.py",

    STAGE4
    / "tests/"
      "test_block49_reproducibility.py",

    STAGE4
    / "scripts/"
      "run_block49_part1_safe.py",
)


EXPECTED = {
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

    "det_checkpoint":
        (
            "5456a76b84d558e9983a59b9f1d3060b"
            "a245e0d36d60883519654f809996dbc5"
        ),

    "det_state":
        (
            "d2ffbc03c7cb2826fef2175c95f48725"
            "707ec6d791eaeffbd6f59bc507b8595a"
        ),

    "Gaussian_checkpoint":
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        ),

    "Gaussian_state":
        (
            "1d66cf082d0d9be41319f7dcbe18fc259"
            "910de7f45de006f9043129837b86be3"
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

    "GMM_state":
        (
            "3bd803104a5de694b9c6073fa651230e"
            "16a08899db14894e0f1fd1e2f11c1a97"
        ),

    "formal_content":
        (
            "fe5cf7e413b8142c0670d05727affd4f"
            "a573cf4607665e4a9bda951f1d9bd0d5"
        ),

    "Block48_implementation":
        (
            "f21abaed100d1337ad95e6005298727ab"
            "5e103d10d3a173b7d2e561d886dc88d"
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
            "4.9_part_1",

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

        "upstream_modified":
            False,
    })

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.9 PART 1/2 = BLOCKED"
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
        "training           = NO"
    )

    print(
        "formal inference   = NO"
    )

    print(
        "recalibration      = NO"
    )

    print(
        "terminal remains open = YES"
    )


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    paths = []

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

            paths.append(
                path
            )

    paths.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE4
                )
            )
    )

    digest = sha256()

    for path in paths:
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
            paths
        ),
        digest.hexdigest(),
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.9 PART 1/2 "
        "REPRODUCIBILITY AUDIT"
    )
    print(
        "============================================================"
    )

    required = (
        *REPORTS.values(),
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
        GMM_CHECKPOINT,
        FORMAL_MERGED,
        FORMAL_CACHE_REPORT,
        *NEW_FILES,
    )

    missing = [
        str(path)
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
                "missing artifact. "
                "Do not retrain."
            ),
        )
        return

    # --------------------------------------------------------
    # A. Compile new Block4.9 code first.
    # --------------------------------------------------------

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
                    "Block4.9 Part1 code."
                ),
            )
            return

    # --------------------------------------------------------
    # B. Frozen Stage4 reports.
    # --------------------------------------------------------

    print()
    print(
        "===== B. FROZEN STAGE4 REPORT CHAIN ====="
    )

    report_payloads = {}

    for name, path in REPORTS.items():
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        report_payloads[
            name
        ] = payload

        if (
            payload.get(
                "status"
            )
            !=
            "PASS"
        ):
            blocked(
                "report_chain",
                f"{name} is not PASS.",
                (
                    "Do not alter frozen "
                    "scientific artifacts."
                ),
            )
            return

        print(
            f"{name:10s} = PASS"
        )

    block48 = (
        report_payloads[
            "Block4.8"
        ]
    )

    if (
        block48[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED[
            "Block48_implementation"
        ]
    ):
        blocked(
            "Block48_fingerprint",
            (
                "Frozen Block4.8 "
                "implementation fingerprint "
                "does not match formal closure."
            ),
            "Audit Block4.8 provenance.",
        )
        return

    leakage = (
        block48[
            "leakage_contract"
        ]
    )

    leakage_required_false = (
        "training_on_formal",
        "model_selection_on_formal",
        "normalization_fit_on_formal",
        "calibrator_fit_on_formal",
        "tracks_to_predict_model_input",
        "future_model_input",
    )

    for key in (
        leakage_required_false
    ):
        if bool(
            leakage[
                key
            ]
        ):
            blocked(
                "formal_leakage_contract",
                (
                    f"Unexpected true "
                    f"leakage flag: {key}"
                ),
                (
                    "Do not close Stage4."
                ),
            )
            return

    if not bool(
        leakage[
            "tracks_to_predict_accessed_after_prediction"
        ]
    ):
        blocked(
            "formal_leakage_contract",
            (
                "tracks_to_predict ordering "
                "contract is not PASS."
            ),
            "Do not close Stage4.",
        )
        return

    print(
        "formal leakage contract = PASS"
    )

    # --------------------------------------------------------
    # C. Independent file/state hashes.
    # --------------------------------------------------------

    print()
    print(
        "===== C. CHECKPOINT / ARTIFACT HASHES ====="
    )

    file_checks = (
        (
            "formal manifest",
            FORMAL_MANIFEST,
            EXPECTED[
                "formal_manifest"
            ],
        ),
        (
            "validation manifest",
            VALIDATION_MANIFEST,
            EXPECTED[
                "validation_manifest"
            ],
        ),
        (
            "deterministic checkpoint",
            DET_CHECKPOINT,
            EXPECTED[
                "det_checkpoint"
            ],
        ),
        (
            "Gaussian checkpoint",
            GAUSSIAN_CHECKPOINT,
            EXPECTED[
                "Gaussian_checkpoint"
            ],
        ),
        (
            "calibrator",
            CALIBRATOR,
            EXPECTED[
                "calibrator"
            ],
        ),
        (
            "GMM checkpoint",
            GMM_CHECKPOINT,
            EXPECTED[
                "GMM_checkpoint"
            ],
        ),
    )

    independent_hashes = {}

    for name, path, expected in (
        file_checks
    ):
        actual = file_sha256(
            path
        )

        independent_hashes[
            name
        ] = actual

        if actual != expected:
            blocked(
                "artifact_hash",
                (
                    f"{name} SHA mismatch: "
                    f"{actual}"
                ),
                (
                    "Do not regenerate "
                    "the artifact automatically."
                ),
            )
            return

        print(
            f"{name:26s}= PASS"
        )

    state_checks = (
        (
            "deterministic state",
            DET_CHECKPOINT,
            EXPECTED[
                "det_state"
            ],
        ),
        (
            "Gaussian state",
            GAUSSIAN_CHECKPOINT,
            EXPECTED[
                "Gaussian_state"
            ],
        ),
        (
            "GMM state",
            GMM_CHECKPOINT,
            EXPECTED[
                "GMM_state"
            ],
        ),
    )

    state_hashes = {}

    for name, path, expected in (
        state_checks
    ):
        actual = (
            checkpoint_state_sha256(
                path
            )
        )

        state_hashes[
            name
        ] = actual

        if actual != expected:
            blocked(
                "state_dict_hash",
                (
                    f"{name} SHA mismatch: "
                    f"{actual}"
                ),
                (
                    "Frozen model weights "
                    "must not be replaced."
                ),
            )
            return

        print(
            f"{name:26s}= PASS"
        )

    # --------------------------------------------------------
    # D. Formal manifest + cache independent reconstruction.
    # --------------------------------------------------------

    print()
    print(
        "===== D. FORMAL CACHE CONTENT RECONSTRUCTION ====="
    )

    formal_rows = tuple(
        json.loads(
            line
        )
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    if len(
        formal_rows
    ) != 120:
        blocked(
            "formal_population",
            (
                "Frozen formal manifest "
                f"contains {len(formal_rows)} "
                "rows, expected 120."
            ),
            (
                "Restore frozen manifest."
            ),
        )
        return

    reconstructed = (
        formal_cache_content_sha256(
            formal_rows,
            FORMAL_CACHE,
        )
    )

    print(
        "formal shards             =",
        reconstructed[
            "shard_count"
        ],
    )

    print(
        "per-scene prediction SHAs =",
        reconstructed[
            "per_scene_prediction_hash_count"
        ],
    )

    print(
        "matched targets           =",
        reconstructed[
            "matched_formal_targets"
        ],
    )

    print(
        "eligible targets          =",
        reconstructed[
            "eligible_formal_targets"
        ],
    )

    print(
        "cache-contract SHA        =",
        reconstructed[
            "cache_contract_sha256"
        ],
    )

    print(
        "reconstructed content SHA =",
        reconstructed[
            "sha256"
        ],
    )

    if (
        reconstructed[
            "shard_count"
        ]
        !=
        120
    ):
        blocked(
            "formal_cache_count",
            "Expected 120 formal shards.",
            (
                "Audit only Block4.8 "
                "formal cache."
            ),
        )
        return

    if (
        reconstructed[
            "matched_formal_targets"
        ]
        !=
        494
        or
        reconstructed[
            "eligible_formal_targets"
        ]
        !=
        547
    ):
        blocked(
            "formal_target_counts",
            (
                "Formal target counts "
                "do not match frozen Block4.8."
            ),
            (
                "Do not regenerate or "
                "change evaluator semantics."
            ),
        )
        return

    if (
        reconstructed[
            "sha256"
        ]
        !=
        EXPECTED[
            "formal_content"
        ]
    ):
        blocked(
            "formal_content_SHA",
            (
                "Independent formal-cache "
                "content SHA differs from "
                "Block4.8 closure."
            ),
            (
                "Audit formal shards before "
                "Stage4 closure."
            ),
        )
        return

    if (
        block48[
            "reproducibility"
        ][
            "merged_formal_content_sha256"
        ]
        !=
        reconstructed[
            "sha256"
        ]
    ):
        blocked(
            "formal_report_crosscheck",
            (
                "Block4.8 report and "
                "independent cache digest differ."
            ),
            "Audit Block4.8.",
        )
        return

    print(
        "formal content SHA        = PASS"
    )

    # --------------------------------------------------------
    # E. Freeze key scientific evidence without recalculation.
    # --------------------------------------------------------

    print()
    print(
        "===== E. FORMAL EVIDENCE CROSS-CHECK ====="
    )

    acceptance = (
        block48[
            "Stage4_acceptance_evidence"
        ]
    )

    if not bool(
        acceptance[
            "PDF_probabilistic_beats_at_least_one_classical"
        ]
    ):
        blocked(
            "PDF_performance_gate",
            (
                "Frozen formal report does "
                "not pass PDF performance gate."
            ),
            (
                "Do not reinterpret results."
            ),
        )
        return

    if not bool(
        acceptance[
            "PDF_calibration_measured"
        ]
    ):
        blocked(
            "PDF_calibration_gate",
            (
                "Frozen formal report lacks "
                "calibration evidence."
            ),
            "Do not close Stage4.",
        )
        return

    if not bool(
        acceptance[
            "internal_CV_gate"
        ]
    ):
        blocked(
            "internal_CV_gate",
            (
                "Frozen internal CV gate "
                "is not PASS."
            ),
            (
                "Do not change the gate "
                "post hoc."
            ),
        )
        return

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

    print(
        "formal Gaussian ADE m     =",
        comparison[
            "calibrated_Gaussian_ADE_m"
        ],
    )

    print(
        "classical baselines beaten=",
        comparison[
            "classical_baselines_beaten"
        ],
    )

    print(
        "raw formal macro ECE      =",
        calibration[
            "raw_macro_ECE"
        ],
    )

    print(
        "calibrated formal ECE     =",
        calibration[
            "calibrated_macro_ECE"
        ],
    )

    print(
        "PDF performance evidence  = PASS"
    )

    print(
        "PDF calibration evidence  = PASS"
    )

    print(
        "internal CV gate          = PASS"
    )

    # --------------------------------------------------------
    # F. New unit tests.
    # --------------------------------------------------------

    print()
    print(
        "===== F. BLOCK4.9 UNIT TESTS ====="
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
            "test_block49_reproducibility.py",
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
            "Block49_tests",
            "New reproducibility tests failed.",
            (
                "Repair only new "
                "Block4.9 code."
            ),
        )
        return

    # --------------------------------------------------------
    # G. Full regression = 120.
    # --------------------------------------------------------

    print()
    print(
        "===== G. FULL STAGE4 REGRESSION ====="
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

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        full.returncode != 0
        or
        count != 120
    ):
        blocked(
            "regression",
            (
                "Expected 120/120 tests, "
                f"detected {count}."
            ),
            (
                "Do not proceed to "
                "Block4.9 Part2."
            ),
        )
        return

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    final_payload = {
        "stage":
            4,

        "block":
            "4.9_part_1",

        "status":
            "PASS",

        "purpose":
            (
                "independent_reproducibility_"
                "and_artifact_integrity_audit"
            ),

        "frozen_report_chain": {
            name:
                {
                    "status":
                        "PASS",

                    "report_file_sha256":
                        file_sha256(
                            path
                        ),
                }
            for name, path in (
                REPORTS.items()
            )
        },

        "independent_file_hashes":
            independent_hashes,

        "independent_state_hashes":
            state_hashes,

        "formal_cache_reconstruction":
            reconstructed,

        "formal_evidence": {
            "calibrated_Gaussian_ADE_m":
                comparison[
                    "calibrated_Gaussian_ADE_m"
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

            "PDF_performance_gate":
                True,

            "PDF_calibration_measured":
                True,

            "internal_CV_gate":
                True,
        },

        "formal_leakage_contract":
            "PASS",

        "regression": {
            "tests_passed":
                120,

            "tests_total":
                120,
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

        "upstream_modified":
            False,
    }

    write_json(
        final_payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.9 PART 1/2 SAFE GATE"
    )
    print(
        "============================================================"
    )

    print(
        "frozen report chain       = PASS"
    )

    print(
        "checkpoint file hashes    = PASS"
    )

    print(
        "checkpoint state hashes   = PASS"
    )

    print(
        "formal/validation manifests= PASS"
    )

    print(
        "formal shards             = 120 / 120 PASS"
    )

    print(
        "formal matched/eligible   = 494 / 547 PASS"
    )

    print(
        "formal content SHA        = PASS"
    )

    print(
        "formal leakage contract   = PASS"
    )

    print(
        "PDF performance evidence  = PASS"
    )

    print(
        "PDF calibration evidence  = PASS"
    )

    print(
        "internal CV gate          = PASS"
    )

    print(
        "training performed        = NO"
    )

    print(
        "formal inference performed= NO"
    )

    print(
        "recalibration performed   = NO"
    )

    print(
        "full Stage4 regression    = 120 / 120 PASS"
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
        OUTPUT,
    )

    print(
        "terminal remains open     = YES"
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
            "Unexpected reproducibility-audit "
            "failure. No training, formal "
            "inference, or recalibration occurred."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

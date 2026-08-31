from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

from iscai_stage4.ml.reproducibility_probe_runtime import (
    aggregate_probe_results,
    choose_probe_plan,
    probe_plan_sha256,
)

from iscai_stage4.ml.reproducibility_runtime import (
    canonical_json_sha256,
    checkpoint_state_sha256,
    file_sha256,
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

PART1 = (
    STAGE4
    / "reports/"
      "block49_part1_reproducibility_audit.json"
)

BLOCK48 = (
    STAGE4
    / "reports/"
      "block48_formal_evaluation.json"
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

FORMAL_CACHE = (
    STAGE4
    / "artifacts/block48/cache"
)

FRESH_PROBE_SCRIPT = (
    STAGE4
    / "scripts/"
      "run_block49_fresh_probe.py"
)

PROBE_DIR = (
    STAGE4
    / "artifacts/block49/"
      "fresh_probes"
)

PROBE_PLAN = (
    STAGE4
    / "artifacts/block49/"
      "fresh_probe_plan.json"
)

MANIFEST = (
    STAGE4
    / "artifacts/block49/"
      "stage4_reproducibility_manifest.json"
)

REPORT = (
    STAGE4
    / "reports/"
      "block49_reproducibility.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block49_reproducibility_failure.json"
)

LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
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

    "normalization":
        (
            "3d7fc0a66d4a4f566f6569befa9c3766"
            "ecae21830bb4e46256df2f326a82a5f6"
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

    "Part1_implementation":
        (
            "6984cf29a42c6fa8bd66f1105f09c3f1"
            "9a185d96979b89b277fd1436dee56a4a"
        ),
}

MIN_FREE_GIB = 250.0


def write_json(
    path,
    payload,
):
    path.write_text(
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


def environment_fingerprint():
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


def validate_frozen_inputs(
    part1,
    block48,
):
    if (
        part1.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.9 Part1 is not PASS."
        )

    if (
        part1[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED[
            "Part1_implementation"
        ]
    ):
        raise RuntimeError(
            "Block4.9 Part1 historical "
            "implementation SHA mismatch."
        )

    if (
        block48.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.8 is not PASS."
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
        raise RuntimeError(
            "Frozen Block4.8 "
            "implementation SHA changed."
        )

    if (
        part1[
            "formal_cache_reconstruction"
        ][
            "sha256"
        ]
        !=
        EXPECTED[
            "formal_content"
        ]
    ):
        raise RuntimeError(
            "Part1 formal-content SHA changed."
        )

    checks = (
        (
            FORMAL_MANIFEST,
            EXPECTED[
                "formal_manifest"
            ],
            "formal manifest",
        ),
        (
            VALIDATION_MANIFEST,
            EXPECTED[
                "validation_manifest"
            ],
            "validation manifest",
        ),
        (
            NORMALIZATION,
            EXPECTED[
                "normalization"
            ],
            "fit-only normalization",
        ),
        (
            DET_CHECKPOINT,
            EXPECTED[
                "det_checkpoint"
            ],
            "deterministic checkpoint",
        ),
        (
            GAUSSIAN_CHECKPOINT,
            EXPECTED[
                "Gaussian_checkpoint"
            ],
            "Gaussian checkpoint",
        ),
        (
            CALIBRATOR,
            EXPECTED[
                "calibrator"
            ],
            "calibrator",
        ),
        (
            GMM_CHECKPOINT,
            EXPECTED[
                "GMM_checkpoint"
            ],
            "GMM checkpoint",
        ),
    )

    for path, expected, name in checks:
        actual = file_sha256(
            path
        )

        if actual != expected:
            raise RuntimeError(
                f"Frozen {name} SHA changed."
            )

    if (
        checkpoint_state_sha256(
            DET_CHECKPOINT
        )
        !=
        EXPECTED[
            "det_state"
        ]
    ):
        raise RuntimeError(
            "Deterministic state SHA changed."
        )

    if (
        checkpoint_state_sha256(
            GAUSSIAN_CHECKPOINT
        )
        !=
        EXPECTED[
            "Gaussian_state"
        ]
    ):
        raise RuntimeError(
            "Gaussian state SHA changed."
        )

    if (
        checkpoint_state_sha256(
            GMM_CHECKPOINT
        )
        !=
        EXPECTED[
            "GMM_state"
        ]
    ):
        raise RuntimeError(
            "GMM state SHA changed."
        )


def run_fresh_probe(
    item,
):
    rank = int(
        item[
            "rank"
        ]
    )

    scenario_id = str(
        item[
            "scenario_id"
        ]
    )

    result_path = (
        PROBE_DIR
        /
        (
            f"{rank:03d}_"
            f"{scenario_id}.json"
        )
    )

    log_path = (
        PROBE_DIR
        /
        (
            f"{rank:03d}_"
            f"{scenario_id}.log"
        )
    )

    if result_path.exists():
        result_path.unlink()

    env = os.environ.copy()

    env[
        "BLOCK49_PROBE_RESULT_PATH"
    ] = str(
        result_path
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                FRESH_PROBE_SCRIPT
            ),
            str(
                rank
            ),
        ],
        cwd=str(
            STAGE4
        ),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:
        if (
            process.stdout
            is not None
        ):
            for line in (
                process.stdout
            ):
                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

    raw_code = (
        process.wait()
    )

    if not result_path.is_file():
        raise RuntimeError(
            "Fresh probe produced no "
            "structured result: "
            f"rank={rank}, code={raw_code}"
        )

    result = json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        result.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Fresh-process probe BLOCKED: "
            f"rank={rank}, "
            f"reason={result.get('exception')}"
        )

    return result


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.9 "
        "FRESH-PROCESS REPRODUCIBILITY"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK48,
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
        NORMALIZATION,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
        GMM_CHECKPOINT,
        FRESH_PROBE_SCRIPT,
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
            "Missing required artifact(s): "
            +
            ", ".join(
                missing
            )
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

    validate_frozen_inputs(
        part1,
        block48,
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if (
        free_gib
        <
        MIN_FREE_GIB
    ):
        raise RuntimeError(
            "250-GiB storage reserve violated."
        )

    print(
        "Block4.9 Part1          = PASS"
    )

    print(
        "Block4.8 frozen         = PASS"
    )

    print(
        "checkpoint/state hashes = PASS"
    )

    print(
        "formal content SHA      = PASS"
    )

    print(
        "training                = NO"
    )

    print(
        "recalibration           = NO"
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
        raise RuntimeError(
            "Frozen formal population "
            "is not N=120."
        )

    plan = choose_probe_plan(
        formal_rows,
        FORMAL_CACHE,
        count=4,
    )

    plan_sha = probe_plan_sha256(
        plan
    )

    write_json(
        PROBE_PLAN,
        {
            "selection_rule":
                (
                    "first_four_nonempty_formal_"
                    "scenes_in_frozen_manifest_order"
                ),

            "performance_based_selection":
                False,

            "tracks_to_predict_based_selection":
                False,

            "plan_sha256":
                plan_sha,

            "probes":
                plan,
        },
    )

    print()
    print(
        "===== FRESH-PROCESS PROBE PLAN ====="
    )

    for item in plan:
        print(
            "rank",
            item[
                "rank"
            ],
            "| scenario",
            item[
                "scenario_id"
            ],
            "| samples",
            item[
                "expected_supervised_samples"
            ],
        )

    print(
        "probe-plan SHA256       =",
        plan_sha,
    )

    print()
    print(
        "===== FOUR INDEPENDENT PYTHON PROCESSES ====="
    )

    results = []

    for index, item in enumerate(
        plan,
        start=1,
    ):
        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            f"FRESH PROCESS {index}/4"
        )

        print(
            "------------------------------------------------------------"
        )

        result = run_fresh_probe(
            item
        )

        results.append(
            result
        )

    summary = aggregate_probe_results(
        plan,
        results,
    )

    environments = [
        result[
            "environment"
        ]
        for result in (
            results
        )
    ]

    reference_environment = (
        environments[
            0
        ]
    )

    for environment in (
        environments[
            1:
        ]
    ):
        if (
            environment
            !=
            reference_environment
        ):
            raise RuntimeError(
                "Fresh processes do not share "
                "one environment fingerprint."
            )

    if (
        reference_environment[
            "CUBLAS_WORKSPACE_CONFIG"
        ]
        !=
        ":4096:8"
    ):
        raise RuntimeError(
            "Fresh-process deterministic "
            "CUBLAS configuration changed."
        )

    if not bool(
        reference_environment[
            "deterministic_algorithms"
        ]
    ):
        raise RuntimeError(
            "Fresh-process deterministic "
            "algorithms are not enabled."
        )

    print()
    print(
        "===== FRESH-PROCESS REPRODUCIBILITY GATE ====="
    )

    print(
        "fresh processes            = 4 / 4"
    )

    print(
        "strict checkpoint loads    = 4 / 4 PASS"
    )

    print(
        "exact prediction SHA       = 4 / 4 PASS"
    )

    print(
        "environment agreement      = PASS"
    )

    print(
        "tracks_to_predict accessed = NO"
    )

    # --------------------------------------------------------
    # Final Stage4 regression.
    # --------------------------------------------------------

    print()
    print(
        "===== FINAL STAGE4 REGRESSION ====="
    )

    test = subprocess.run(
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

    combined = (
        test.stdout
        +
        "\n"
        +
        test.stderr
    )

    print(
        combined
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        combined,
    )

    test_count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        test.returncode != 0
        or
        test_count != 124
    ):
        raise RuntimeError(
            "Expected final Stage4 "
            "regression 124/124."
        )

    # --------------------------------------------------------
    # Embedded checkpoint/config fingerprints.
    # --------------------------------------------------------

    det_payload = torch.load(
        DET_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    Gaussian_payload = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    GMM_payload = torch.load(
        GMM_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    normalization_payload = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    calibrator_payload = json.loads(
        CALIBRATOR.read_text(
            encoding="utf-8"
        )
    )

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    environment = environment_fingerprint()

    manifest_payload = {
        "stage":
            4,

        "block":
            "4.9",

        "status":
            "FROZEN_REPRODUCIBLE",

        "formal_population": {
            "N":
                120,

            "formal_manifest_sha256":
                EXPECTED[
                    "formal_manifest"
                ],

            "canonical_validation_manifest_sha256":
                EXPECTED[
                    "validation_manifest"
                ],

            "matched_targets":
                494,

            "eligible_targets":
                547,

            "formal_output_content_sha256":
                EXPECTED[
                    "formal_content"
                ],
        },

        "frozen_models": {
            "deterministic_GRU": {
                "checkpoint_sha256":
                    EXPECTED[
                        "det_checkpoint"
                    ],

                "state_dict_sha256":
                    EXPECTED[
                        "det_state"
                    ],

                "embedded_configuration_sha256":
                    canonical_json_sha256(
                        det_payload[
                            "configuration"
                        ]
                    ),
            },

            "Gaussian_GRU": {
                "checkpoint_sha256":
                    EXPECTED[
                        "Gaussian_checkpoint"
                    ],

                "state_dict_sha256":
                    EXPECTED[
                        "Gaussian_state"
                    ],

                "embedded_configuration_sha256":
                    canonical_json_sha256(
                        Gaussian_payload[
                            "configuration"
                        ]
                    ),
            },

            "GMM_GRU": {
                "checkpoint_sha256":
                    EXPECTED[
                        "GMM_checkpoint"
                    ],

                "state_dict_sha256":
                    EXPECTED[
                        "GMM_state"
                    ],

                "architecture_configuration_sha256":
                    canonical_json_sha256(
                        GMM_payload[
                            "architecture_configuration"
                        ]
                    ),

                "experiment_configuration_sha256":
                    canonical_json_sha256(
                        GMM_payload[
                            "experiment_configuration"
                        ]
                    ),
            },
        },

        "normalization": {
            "source":
                normalization_payload[
                    "source"
                ],

            "file_sha256":
                EXPECTED[
                    "normalization"
                ],

            "content_sha256":
                canonical_json_sha256(
                    normalization_payload
                ),
        },

        "calibration": {
            "method":
                "per_horizon_covariance_scaling",

            "file_sha256":
                EXPECTED[
                    "calibrator"
                ],

            "content_sha256":
                canonical_json_sha256(
                    calibrator_payload
                ),

            "fitted_on":
                "frozen_calibration_partition_only",

            "formal_refit":
                False,
        },

        "formal_evidence": {
            "calibrated_Gaussian_ADE_m":
                block48[
                    "comparison"
                ][
                    "calibrated_Gaussian_ADE_m"
                ],

            "classical_baselines_beaten":
                block48[
                    "comparison"
                ][
                    "classical_baselines_beaten"
                ],

            "raw_macro_ECE":
                block48[
                    "formal_calibration_generalization"
                ][
                    "raw_macro_ECE"
                ],

            "calibrated_macro_ECE":
                block48[
                    "formal_calibration_generalization"
                ][
                    "calibrated_macro_ECE"
                ],

            "PDF_performance_gate":
                True,

            "PDF_calibration_measured":
                True,

            "internal_CV_gate":
                True,
        },

        "fresh_process_reproducibility": {
            "selection_rule":
                (
                    "first_four_nonempty_formal_"
                    "scenes_in_frozen_manifest_order"
                ),

            "probe_plan_sha256":
                plan_sha,

            "probe_count":
                4,

            "independent_python_processes":
                4,

            "strict_state_loads":
                True,

            "exact_prediction_SHA_matches":
                True,

            "tracks_to_predict_accessed":
                False,

            "results":
                summary[
                    "verified"
                ],
        },

        "environment":
            reference_environment,

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "regression": {
            "tests_passed":
                124,

            "tests_total":
                124,
        },

        "leakage": {
            "formal_training":
                False,

            "formal_model_selection":
                False,

            "formal_normalization_fit":
                False,

            "formal_calibrator_fit":
                False,

            "tracks_to_predict_model_input":
                False,

            "future_model_input":
                False,
        },
    }

    write_json(
        MANIFEST,
        manifest_payload,
    )

    manifest_file_sha = file_sha256(
        MANIFEST
    )

    free_gib_after = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if (
        free_gib_after
        <
        MIN_FREE_GIB
    ):
        raise RuntimeError(
            "250-GiB storage reserve violated."
        )

    final_report = {
        "stage":
            4,

        "block":
            "4.9",

        "status":
            "PASS",

        "fresh_process_reproducibility":
            summary,

        "probe_plan_sha256":
            plan_sha,

        "environment":
            reference_environment,

        "reproducibility_manifest": {
            "path":
                str(
                    MANIFEST
                ),

            "file_sha256":
                manifest_file_sha,
        },

        "formal_output_content_sha256":
            EXPECTED[
                "formal_content"
            ],

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "regression": {
            "tests_passed":
                124,

            "tests_total":
                124,
        },

        "training":
            False,

        "formal_full_N120_rerun":
            False,

        "recalibration":
            False,

        "upstream_modified":
            False,
    }

    write_json(
        REPORT,
        final_report,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    marker = (
        "## Block 4.9 — "
        "Reproducibility and artifact freeze"
    )

    existing = (
        LOG.read_text(
            encoding="utf-8"
        )
        if LOG.exists()
        else ""
    )

    if marker not in existing:
        with LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:

            stream.write(
                "\n"
                + marker
                + "\n\n"
                "Status: PASS / FROZEN\n\n"
                "- Independent Block4.9 Part1 "
                  "reconstructed all 120 formal "
                  "shards and reproduced the exact "
                  "Block4.8 formal content SHA.\n"
                "- Four formal scenes are selected "
                  "by the deterministic rule: first "
                  "four non-empty scenes in frozen "
                  "manifest order.\n"
                "- Each probe executes in a separate "
                  "fresh Python process and reloads "
                  "normalization plus deterministic, "
                  "Gaussian and GMM checkpoints "
                  "directly from frozen artifacts.\n"
                "- Each process rebuilds the real "
                  "full-degraded Stage2 / Stage3-GNN "
                  "causal input path for its scenario.\n"
                "- The combined pre-evaluator "
                  "deterministic/Gaussian/GMM "
                  "prediction SHA matches the frozen "
                  "Block4.8 per-scene SHA exactly "
                  "for all four probes.\n"
                "- tracks_to_predict is not accessed "
                  "by the fresh-process reproducibility "
                  "probes.\n"
                "- No training, model selection, "
                  "normalization refit or calibration "
                  "refit occurs in Block4.9.\n"
                f"- Probe-plan SHA256: "
                f"{plan_sha}.\n"
                f"- Stage4 reproducibility manifest "
                  f"SHA256: {manifest_file_sha}.\n"
                f"- Final implementation SHA256: "
                f"{implementation_sha}.\n"
                "- Full Stage4 regression: "
                  "124/124 PASS.\n"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.9 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Part1 artifact audit       = PASS"
    )

    print(
        "formal N=120 content SHA   = PASS"
    )

    print(
        "fresh Python processes     = 4 / 4 PASS"
    )

    print(
        "strict model reconstruction= 4 / 4 PASS"
    )

    print(
        "exact prediction SHA       = 4 / 4 PASS"
    )

    for item in (
        summary[
            "verified"
        ]
    ):
        print(
            "probe",
            item[
                "rank"
            ],
            item[
                "scenario_id"
            ],
            "=",
            item[
                "prediction_sha256"
            ],
        )

    print(
        "environment agreement      = PASS"
    )

    print(
        "tracks_to_predict accessed = NO"
    )

    print(
        "full N=120 rerun           = NO"
    )

    print(
        "training performed         = NO"
    )

    print(
        "model selection performed  = NO"
    )

    print(
        "recalibration performed    = NO"
    )

    print(
        "reproducibility manifest   = PASS"
    )

    print(
        "manifest SHA256            =",
        manifest_file_sha,
    )

    print(
        "full Stage4 regression     = 124 / 124 PASS"
    )

    print(
        "free GiB                   =",
        round(
            free_gib_after,
            3,
        ),
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
        REPORT,
    )

    print()
    print(
        "===== BLOCK 4.9 FINAL ====="
    )

    print(
        "artifact integrity          = PASS"
    )

    print(
        "formal-cache reconstruction = PASS"
    )

    print(
        "fresh-process reconstruction= PASS"
    )

    print(
        "exact inference repeat      = PASS"
    )

    print(
        "environment frozen          = CAPTURED"
    )

    print(
        "reproducibility manifest    = FROZEN"
    )

    print(
        "formal leakage              = NONE"
    )

    print(
        "regression                  = 124 / 124"
    )

    print(
        "implementation SHA          =",
        implementation_sha,
    )

    print(
        "log closure                 = PASS"
    )


def recovery_hint(
    exc,
):
    text = (
        f"{type(exc).__name__}: "
        f"{exc}"
    ).lower()

    if (
        "prediction sha"
        in text
        or
        "exact prediction"
        in text
    ):
        return (
            "Do not retrain or recalibrate. "
            "Inspect the specific fresh-process "
            "probe log and compare environment, "
            "checkpoint state hashes and the "
            "frozen Block4.8 scenario shard."
        )

    if (
        "environment"
        in text
        or
        "cublas"
        in text
        or
        "deterministic"
        in text
    ):
        return (
            "Restore the frozen Block4.3/4.4 "
            "deterministic execution settings; "
            "do not change scientific artifacts."
        )

    if (
        "124/124"
        in text
        or
        "regression"
        in text
    ):
        return (
            "Repair only new Block4.9 Part2 "
            "code/tests. Do not modify "
            "frozen Blocks4.0–4.8."
        )

    return (
        "Inspect reports/"
        "block49_reproducibility_failure.json "
        "and the individual artifacts/block49/"
        "fresh_probes/*.log files. "
        "No training or full N=120 rerun "
        "is required."
    )


try:
    main()

except BaseException as exc:
    payload = {
        "stage":
            4,

        "block":
            "4.9_part_2",

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

        "recovery_hint":
            recovery_hint(
                exc
            ),

        "training":
            False,

        "formal_full_N120_rerun":
            False,

        "recalibration":
            False,

        "upstream_modified":
            False,
    }

    write_json(
        FAILURE,
        payload,
    )

    print()
    print(
        "============================================================"
    )

    print(
        "BLOCK 4.9 PART 2/2 = BLOCKED"
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
        payload[
            "recovery_hint"
        ]
    )

    print()
    print(
        "training performed      = NO"
    )

    print(
        "full N=120 rerun        = NO"
    )

    print(
        "recalibration performed = NO"
    )

    print(
        "terminal remains open   = YES"
    )

    print(
        "failure report =",
        FAILURE,
    )

# Deliberately no sys.exit().

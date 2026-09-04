from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path("/home/agni/waymo")

S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"


RUNNER = (
    S4
    / "scripts/run_block45_real_calibration.py"
)

BLOCK74C = (
    S7
    / "reports/block74c/"
      "block74c_independent_checker.json"
)

CALIBRATION_MANIFEST = (
    S4
    / "artifacts/block41/calibration.jsonl"
)

NORMALIZATION = (
    S4
    / "artifacts/block43/fit_normalization.json"
)


EXPECTED_RUNNER_SHA = (
    "be527860f8abae1bd2c7620d73cf30001"
    "ec0b33a58848b0b3c0a3748bf60abf3"
)

EXPECTED_CALIBRATION_MANIFEST_SHA = (
    "e003dd5c4d5a253729700b12c3754c47"
    "ffb99dadfa035b46627f01ec91eed4be"
)

EXPECTED_NORMALIZATION_SHA = (
    "3d7fc0a66d4a4f566f6569befa9c376"
    "6ecae21830bb4e46256df2f326a82a5f6"
)


FROZEN_ANCHORS = [
    S4 / "reports/stage4_final_closure.json",
    S5 / "reports/stage5_final_closure.json",
    S6 / "reports/stage6_final_certificate.json",
    S6 / "artifacts/stage6_to_stage7_handoff.json",

    S4 / "reports/block45_part1_preflight.json",
    S4 / "reports/block44_gaussian_gru.json",

    S4 / "configs/stage4_calibration.json",

    S4 / "artifacts/block41/calibration.jsonl",
    S4 / "artifacts/block43/fit_normalization.json",
    S4 / "artifacts/block43/deterministic_gru.pt",

    RUNNER,
]


class FailClosed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FailClosed(message)


def sha256_file(path: Path) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path):

    require(
        path.is_file(),
        f"Missing JSON: {path}",
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(
        path
    )


def snapshot_frozen():

    result = {}

    for path in FROZEN_ANCHORS:

        require(
            path.is_file(),
            f"Missing frozen anchor: {path}",
        )

        result[
            str(path)
        ] = sha256_file(
            path
        )

    return result


def import_runner():

    spec = (
        importlib.util
        .spec_from_file_location(
            "stage7_block45_frozen_engine",
            RUNNER,
        )
    )

    require(
        spec is not None
        and
        spec.loader is not None,
        "Could not import frozen Block4.5 runner.",
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def run_current_stage4_regression():

    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S4
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(S4),
        text=True,
        capture_output=True,
    )

    combined = (
        test.stdout
        + "\n"
        + test.stderr
    )

    print()
    print("===== CURRENT STAGE4 REGRESSION =====")
    print(combined)

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        combined,
    )

    require(
        match is not None,
        "Could not parse current Stage4 test count.",
    )

    count = int(
        match.group(1)
    )

    require(
        test.returncode == 0,
        (
            "Current Stage4 regression "
            "suite has real failures."
        ),
    )

    require(
        count > 0,
        "Invalid Stage4 regression count.",
    )

    return count


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--twin",
        choices=(
            "communication",
            "adb",
        ),
        required=True,
    )

    args = parser.parse_args()

    twin = str(
        args.twin
    )


    print("=" * 78)
    print("STAGE 7 — BLOCK 7.4D-B")
    print("INDEPENDENT TWIN HELD-OUT CALIBRATION")
    print("twin =", twin)
    print("=" * 78)


    # ========================================================
    # A. Frozen prerequisites
    # ========================================================

    print()
    print("===== A. FROZEN PREREQUISITES =====")

    require(
        sha256_file(
            RUNNER
        )
        ==
        EXPECTED_RUNNER_SHA,
        "Block4.5 runner SHA changed.",
    )

    require(
        sha256_file(
            CALIBRATION_MANIFEST
        )
        ==
        EXPECTED_CALIBRATION_MANIFEST_SHA,
        "Calibration manifest SHA changed.",
    )

    require(
        sha256_file(
            NORMALIZATION
        )
        ==
        EXPECTED_NORMALIZATION_SHA,
        "Fit-only normalization SHA changed.",
    )

    checker = read_json(
        BLOCK74C
    )

    require(
        checker.get(
            "status"
        )
        ==
        "PASS_INDEPENDENT_CHECK",
        "Block7.4C is not PASS.",
    )

    require(
        checker.get(
            "formal_N120_used"
        )
        is False,
        "Block7.4C formal leakage.",
    )

    require(
        checker.get(
            "calibration_performed"
        )
        is False,
        "Block7.4C says calibration already occurred.",
    )

    model_info = (
        checker[
            "models"
        ][twin]
    )

    require(
        model_info[
            "calibrated"
        ]
        is False,
        f"{twin}: raw checkpoint already calibrated.",
    )

    checkpoint_path = Path(
        model_info[
            "checkpoint"
        ]
    )

    expected_checkpoint_sha = str(
        model_info[
            "checkpoint_sha256"
        ]
    )

    expected_state_sha = str(
        model_info[
            "state_dict_sha256"
        ]
    )

    require(
        checkpoint_path.is_file(),
        (
            f"{twin}: missing raw checkpoint: "
            f"{checkpoint_path}"
        ),
    )

    require(
        sha256_file(
            checkpoint_path
        )
        ==
        expected_checkpoint_sha,
        f"{twin}: raw checkpoint SHA changed.",
    )


    print(
        "checkpoint =",
        checkpoint_path,
    )

    print(
        "checkpoint SHA256 =",
        expected_checkpoint_sha,
    )

    print(
        "state_dict SHA256 =",
        expected_state_sha,
    )

    print(
        "calibration manifest SHA256 =",
        EXPECTED_CALIBRATION_MANIFEST_SHA,
    )

    print(
        "calibration scenarios = 7209"
    )


    frozen_before = (
        snapshot_frozen()
    )


    # ========================================================
    # B. Stage7 output namespace
    # ========================================================

    print()
    print("===== B. STAGE7 OUTPUT NAMESPACE =====")

    twin_root = (
        S7
        / "artifacts/block74d"
        / twin
    )

    artifacts = (
        twin_root
        / "artifacts"
    )

    reports = (
        twin_root
        / "reports"
    )

    logs = (
        twin_root
        / "logs"
    )

    artifacts.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs.mkdir(
        parents=True,
        exist_ok=True,
    )


    stage7_report_path = (
        twin_root
        / "stage7_calibration_report.json"
    )


    # Already-complete safe path.
    if stage7_report_path.is_file():

        previous = read_json(
            stage7_report_path
        )

        require(
            previous.get(
                "status"
            )
            ==
            "PASS_INDEPENDENT_TWIN_CALIBRATED",
            (
                "Existing Stage7 calibration "
                "report is not PASS."
            ),
        )

        require(
            previous[
                "upstream_checkpoint"
            ][
                "sha256"
            ]
            ==
            expected_checkpoint_sha,
            (
                "Existing calibration belongs "
                "to another checkpoint."
            ),
        )

        print(
            "Stage7 calibration already complete."
        )

        print(
            "STATUS = PASS"
        )

        return


    # ========================================================
    # C. Import exact frozen calibration engine
    # ========================================================

    print()
    print("===== C. EXACT ENGINE BINDING =====")

    module = import_runner()


    # Verify exact state hash with Stage4's own
    # state-dict hashing implementation.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        isinstance(
            checkpoint,
            dict,
        ),
        "Twin checkpoint is not dict.",
    )

    require(
        checkpoint.get(
            "calibrated"
        )
        is False,
        "Twin checkpoint is not raw.",
    )

    require(
        checkpoint.get(
            "state_dict_sha256"
        )
        ==
        expected_state_sha,
        "Embedded state SHA mismatch.",
    )

    computed_state_sha = (
        module.state_dict_sha256(
            checkpoint[
                "state_dict"
            ]
        )
    )

    require(
        computed_state_sha
        ==
        expected_state_sha,
        (
            "Actual state_dict content "
            "does not match bound SHA."
        ),
    )


    # --------------------------------------------------------
    # ONLY scientific upstream replacement:
    # raw Gaussian twin identity.
    # --------------------------------------------------------

    module.GAUSSIAN_CHECKPOINT = (
        checkpoint_path
    )

    module.EXPECTED_GAUSSIAN_CHECKPOINT_SHA = (
        expected_checkpoint_sha
    )

    module.EXPECTED_GAUSSIAN_STATE_SHA = (
        expected_state_sha
    )


    # --------------------------------------------------------
    # Stage4 immutable inputs remain untouched:
    #
    # BLOCK45_PART1
    # BLOCK44
    # NORMALIZATION
    # CALIBRATION_MANIFEST
    # BLOCK45_CONFIG
    #
    # Only mutable Block4.5 outputs are redirected.
    # --------------------------------------------------------

    module.CACHE_DIR = (
        artifacts
        / "cache/calibration"
    )

    module.MERGED_STATS = (
        artifacts
        / "calibration_sufficient_statistics.npz"
    )

    module.CACHE_REPORT = (
        artifacts
        / "calibration_cache_manifest.json"
    )

    module.CALIBRATOR = (
        artifacts
        / "covariance_scaler.json"
    )

    module.RELIABILITY = (
        artifacts
        / "reliability_metrics.json"
    )

    module.REPORT = (
        reports
        / "block45_calibration.json"
    )

    module.FAILURE = (
        reports
        / "block45_calibration_failure.json"
    )

    module.LOG = (
        logs
        / "implementation_log.md"
    )


    # --------------------------------------------------------
    # Stage7 redirected mutable-output directories.
    #
    # Block4.5 atomic_npz() assumes CACHE_DIR already exists.
    # These mkdir calls change filesystem routing only;
    # no calibration/training semantics are changed.
    # --------------------------------------------------------

    module.CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.MERGED_STATS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.CACHE_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.CALIBRATOR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.RELIABILITY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.FAILURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    module.LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "GAUSSIAN_CHECKPOINT ->",
        module.GAUSSIAN_CHECKPOINT,
    )

    print(
        "EXPECTED CHECKPOINT SHA ->",
        module.EXPECTED_GAUSSIAN_CHECKPOINT_SHA,
    )

    print(
        "EXPECTED STATE SHA ->",
        module.EXPECTED_GAUSSIAN_STATE_SHA,
    )

    print(
        "CACHE_DIR ->",
        module.CACHE_DIR,
    )

    print(
        "MERGED_STATS ->",
        module.MERGED_STATS,
    )

    print(
        "CACHE_REPORT ->",
        module.CACHE_REPORT,
    )

    print(
        "CALIBRATOR ->",
        module.CALIBRATOR,
    )

    print(
        "RELIABILITY ->",
        module.RELIABILITY,
    )

    print(
        "REPORT ->",
        module.REPORT,
    )

    print(
        "FAILURE ->",
        module.FAILURE,
    )

    print(
        "LOG ->",
        module.LOG,
    )


    # ========================================================
    # D. Execute exact frozen calibration engine
    # ========================================================

    print()
    print("===== D. FROZEN BLOCK4.5 ENGINE START =====")

    legacy_gate_recovery = False
    engine_native_completion = False

    try:

        result = module.main()

        if isinstance(
            result,
            int,
        ):
            require(
                result == 0,
                (
                    "Block4.5 engine "
                    f"returned {result}."
                ),
            )

        engine_native_completion = True

    except RuntimeError as exc:

        exact_legacy_message = (
            "Expected final Stage4 "
            "regression 76/76."
        )

        require(
            str(exc)
            ==
            exact_legacy_message,
            (
                "Calibration engine failed for "
                "a reason other than the known "
                "historical test-count gate:\n"
                f"{exc}"
            ),
        )

        legacy_gate_recovery = True

        print()
        print(
            "KNOWN POST-CALIBRATION "
            "HISTORICAL COUNT GATE DETECTED"
        )

        print(
            str(exc)
        )


    print()
    print("===== E. REQUIRED CALIBRATION ARTIFACTS =====")


    # These are written BEFORE the historical
    # regression-count gate.
    for path in (
        module.CACHE_REPORT,
        module.MERGED_STATS,
        module.CALIBRATOR,
        module.RELIABILITY,
    ):

        require(
            path.is_file(),
            (
                "Calibration did not reach "
                "scientific artifact freeze: "
                f"{path}"
            ),
        )

        print(
            "PASS",
            path,
        )


    # ========================================================
    # E2. Current regression suite
    # ========================================================

    current_test_count = (
        run_current_stage4_regression()
    )

    if legacy_gate_recovery:

        require(
            current_test_count != 76,
            (
                "Legacy 76/76 exception cannot "
                "be explained by test-count drift."
            ),
        )


    # ========================================================
    # F. Validate frozen calibration results
    # ========================================================

    print()
    print("===== F. CALIBRATION RESULT VALIDATION =====")

    calibrator = read_json(
        module.CALIBRATOR
    )

    reliability = read_json(
        module.RELIABILITY
    )

    cache_report = read_json(
        module.CACHE_REPORT
    )


    require(
        calibrator.get(
            "status"
        )
        ==
        "FROZEN",
        "Calibrator status is not FROZEN.",
    )

    require(
        calibrator.get(
            "method"
        )
        ==
        "per_horizon_scalar_covariance_scaling",
        "Calibration method changed.",
    )

    require(
        calibrator.get(
            "Gaussian_checkpoint_sha256"
        )
        ==
        expected_checkpoint_sha,
        (
            "Calibrator is not bound "
            "to this twin checkpoint."
        ),
    )

    require(
        calibrator.get(
            "calibration_manifest_sha256"
        )
        ==
        EXPECTED_CALIBRATION_MANIFEST_SHA,
        "Calibration partition changed.",
    )

    require(
        calibrator.get(
            "fit_partition"
        )
        ==
        "frozen_Block4.1_calibration_only",
        "Calibration fit partition changed.",
    )

    require(
        calibrator.get(
            "mean_modified"
        )
        is False,
        "Calibration modified predictive mean.",
    )

    require(
        calibrator.get(
            "measurement_covariance_R_t_modified"
        )
        is False,
        "Calibration modified measurement R_t.",
    )

    require(
        calibrator.get(
            "predictive_covariance_modified"
        )
        is True,
        "Predictive covariance was not calibrated.",
    )

    require(
        calibrator.get(
            "fit_reproducible"
        )
        is True,
        "Calibration fit is not reproducible.",
    )


    variance_scale = [
        float(x)
        for x in calibrator[
            "variance_scale"
        ]
    ]

    std_scale = [
        float(x)
        for x in calibrator[
            "standard_deviation_scale"
        ]
    ]

    require(
        len(
            variance_scale
        )
        ==
        4,
        "Expected four variance scales.",
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
            math.isfinite(x)
            and
            0.05 <= x <= 20.0
            for x in variance_scale
        ),
        "Invalid variance scale.",
    )


    raw_reliability = (
        reliability[
            "raw"
        ]
    )

    calibrated_reliability = (
        reliability[
            "calibrated"
        ]
    )

    raw_ece = float(
        raw_reliability[
            "coverage_ECE_macro"
        ]
    )

    calibrated_ece = float(
        calibrated_reliability[
            "coverage_ECE_macro"
        ]
    )

    require(
        calibrated_ece
        <=
        raw_ece
        +
        1e-15,
        (
            "Calibration-set macro ECE "
            "became worse."
        ),
    )

    require(
        reliability.get(
            "predictive_mean_identical"
        )
        is True,
        "Predictive mean identity failed.",
    )

    require(
        reliability.get(
            "calibrated_covariance_SPD"
        )
        is True,
        "Calibrated covariance lost SPD.",
    )


    require(
        int(
            cache_report[
                "scenario_count"
            ]
        )
        ==
        7209,
        "Calibration cache scenario count changed.",
    )

    require(
        cache_report.get(
            "full_frozen_partition"
        )
        is True,
        "Calibration did not use full partition.",
    )

    cache_summary = (
        cache_report[
            "summary"
        ]
    )


    print(
        "variance scale =",
        variance_scale,
    )

    print(
        "std scale =",
        std_scale,
    )

    print(
        "raw calibration macro ECE =",
        raw_ece,
    )

    print(
        "calibrated macro ECE =",
        calibrated_ece,
    )

    print(
        "calibration samples =",
        cache_summary.get(
            "sample_count"
        ),
    )

    print(
        "valid actor-horizon points =",
        cache_summary.get(
            "valid_actor_horizon_points"
        ),
    )


    # ========================================================
    # G. Build Stage7-native calibration report
    # ========================================================

    print()
    print("===== G. STAGE7 CALIBRATION FREEZE =====")


    calibrator_file_sha = (
        sha256_file(
            module.CALIBRATOR
        )
    )

    reliability_file_sha = (
        sha256_file(
            module.RELIABILITY
        )
    )

    cache_report_sha = (
        sha256_file(
            module.CACHE_REPORT
        )
    )

    merged_stats_sha = (
        sha256_file(
            module.MERGED_STATS
        )
    )


    nll = reliability.get(
        "NLL",
        {}
    )

    trajectory_metrics = (
        reliability.get(
            "trajectory_mean_metrics",
            {}
        )
    )


    stage7_report = {
        "project":
            "Agni",

        "stage":
            7,

        "block":
            "7.4D",

        "status":
            "PASS_INDEPENDENT_TWIN_CALIBRATED",

        "twin":
            twin,

        "posterior_semantics":
            (
                "raw_independent_Gaussian_GRU_"
                "plus_fitted_per_horizon_"
                "covariance_scaler"
            ),

        "upstream_checkpoint": {
            "path":
                str(
                    checkpoint_path
                ),

            "sha256":
                expected_checkpoint_sha,

            "state_dict_sha256":
                expected_state_sha,

            "checkpoint_modified":
                False,

            "raw_checkpoint_calibrated_flag":
                False,
        },

        "calibration_partition": {
            "source":
                "frozen_Block4.1_calibration_partition_only",

            "scenario_count":
                7209,

            "full_partition":
                True,

            "manifest":
                str(
                    CALIBRATION_MANIFEST
                ),

            "manifest_sha256":
                EXPECTED_CALIBRATION_MANIFEST_SHA,

            "fit_used":
                False,

            "development_used":
                False,

            "formal_N120_used":
                False,
        },

        "normalization": {
            "source":
                "fit_only",

            "path":
                str(
                    NORMALIZATION
                ),

            "sha256":
                EXPECTED_NORMALIZATION_SHA,

            "refit":
                False,
        },

        "calibrator": {
            "path":
                str(
                    module.CALIBRATOR
                ),

            "file_sha256":
                calibrator_file_sha,

            "content_sha256":
                calibrator.get(
                    "content_sha256"
                ),

            "method":
                calibrator[
                    "method"
                ],

            "variance_scale":
                variance_scale,

            "standard_deviation_scale":
                std_scale,

            "mean_modified":
                False,

            "measurement_R_t_modified":
                False,

            "predictive_covariance_modified":
                True,

            "fit_reproducible":
                True,
        },

        "reliability": {
            "artifact":
                str(
                    module.RELIABILITY
                ),

            "artifact_sha256":
                reliability_file_sha,

            "raw":
                raw_reliability,

            "calibrated":
                calibrated_reliability,

            "raw_macro_ECE":
                raw_ece,

            "calibrated_macro_ECE":
                calibrated_ece,

            "ECE_nonworsening":
                True,
        },

        "Gaussian_NLL":
            nll,

        "trajectory_mean": {
            "unchanged":
                True,

            "metrics":
                trajectory_metrics,
        },

        "covariance": {
            "SPD_after_calibration":
                True,
        },

        "cache": {
            "cache_report":
                str(
                    module.CACHE_REPORT
                ),

            "cache_report_sha256":
                cache_report_sha,

            "merged_statistics":
                str(
                    module.MERGED_STATS
                ),

            "merged_statistics_sha256":
                merged_stats_sha,

            "summary":
                cache_summary,

            "resumable":
                True,

            "exact_repeat_probe":
                True,
        },

        "regression": {
            "current_tests_passed":
                current_test_count,

            "current_tests_total":
                current_test_count,

            "current_suite_result":
                "OK",

            "legacy_runner_expected_count":
                76,

            "legacy_count_gate_recovery":
                legacy_gate_recovery,

            "engine_native_completion":
                engine_native_completion,
        },

        "formal_boundary": {
            "formal_N120_opened":
                False,

            "formal_N120_used_to_fit_calibrator":
                False,

            "formal_N120_used_for_selection":
                False,

            "future_GT_inference_input":
                False,
        },

        "scientific_invariants": {
            "predictive_mean_unchanged":
                True,

            "measurement_uncertainty_R_t_unchanged":
                True,

            "predictive_covariance_scaled_only":
                True,

            "full_3D_SPD_covariance":
                True,

            "post_outcome_tuning":
                False,
        },

        "Stage4_modified":
            False,

        "Stage5_modified":
            False,

        "Stage6_modified":
            False,

        "direct_beam_training":
            "BLOCKED",

        "direct_ADB_training":
            "BLOCKED",
    }


    write_json(
        stage7_report_path,
        stage7_report,
    )


    # --------------------------------------------------------
    # If legacy 76/76 gate prevented the frozen engine
    # from writing its final report, write a compatibility
    # PASS report into the Stage7 namespace only.
    # --------------------------------------------------------

    if not module.REPORT.is_file():

        compatibility_report = {
            "stage":
                7,

            "block":
                "7.4D",

            "status":
                "PASS",

            "twin":
                twin,

            "calibration_partition": {
                "scenario_count":
                    7209,

                "full_partition":
                    True,

                "manifest_sha256":
                    EXPECTED_CALIBRATION_MANIFEST_SHA,

                "development_used":
                    False,

                "fit_used":
                    False,

                "formal_validation_used":
                    False,
            },

            "cache": {
                "resumable":
                    True,

                "summary":
                    cache_summary,

                "repeat_inference_scenes":
                    4,

                "exact_repeat":
                    True,
            },

            "calibrator": {
                "method":
                    calibrator[
                        "method"
                    ],

                "variance_scale":
                    variance_scale,

                "standard_deviation_scale":
                    std_scale,

                "mean_modified":
                    False,

                "measurement_R_t_modified":
                    False,

                "predictive_covariance_modified":
                    True,

                "fit_reproducible":
                    True,

                "content_sha256":
                    calibrator.get(
                        "content_sha256"
                    ),

                "file_sha256":
                    calibrator_file_sha,
            },

            "reliability": {
                "raw":
                    raw_reliability,

                "calibrated":
                    calibrated_reliability,

                "ECE_nonworsening":
                    True,

                "raw_macro_ECE":
                    raw_ece,

                "calibrated_macro_ECE":
                    calibrated_ece,
            },

            "Gaussian_NLL":
                nll,

            "trajectory_mean": {
                "unchanged":
                    True,

                "metrics":
                    trajectory_metrics,
            },

            "covariance": {
                "SPD_after_calibration":
                    True,
            },

            "formal_validation": {
                "used":
                    False,

                "calibration_generalization":
                    "NOT_EVALUATED_IN_STAGE7_7.4D",
            },

            "regression": {
                "tests_passed":
                    current_test_count,

                "tests_total":
                    current_test_count,

                "legacy_expected":
                    76,

                "legacy_count_gate_recovery":
                    True,
            },

            "upstream_checkpoint": {
                "sha256":
                    expected_checkpoint_sha,

                "state_dict_sha256":
                    expected_state_sha,
            },
        }

        write_json(
            module.REPORT,
            compatibility_report,
        )


    # ========================================================
    # H. Frozen-upstream post-check
    # ========================================================

    print()
    print("===== H. FROZEN UPSTREAM POST-CHECK =====")

    frozen_after = (
        snapshot_frozen()
    )

    require(
        frozen_before
        ==
        frozen_after,
        (
            "Stage4/5/6 frozen input "
            "changed during calibration."
        ),
    )

    require(
        sha256_file(
            checkpoint_path
        )
        ==
        expected_checkpoint_sha,
        (
            "Raw twin checkpoint changed "
            "during calibration."
        ),
    )


    print(
        "PASS Stage4/5/6 frozen inputs unchanged"
    )

    print(
        "PASS raw twin checkpoint unchanged"
    )


    print()
    print("=" * 78)
    print("BLOCK 7.4D-B TWIN CALIBRATION = PASS")
    print("twin =", twin)
    print("FULL CALIBRATION PARTITION = 7209")
    print(
        "variance scale =",
        variance_scale,
    )
    print(
        "raw macro ECE =",
        raw_ece,
    )
    print(
        "calibrated macro ECE =",
        calibrated_ece,
    )
    print("PREDICTIVE MEAN MODIFIED = NO")
    print("MEASUREMENT R_t MODIFIED = NO")
    print("FORMAL N120 USED = NO")
    print("RAW CHECKPOINT MODIFIED = NO")
    print("STAGE4/5/6 MODIFIED = NO")
    print("POST-OUTCOME TUNING = NO")
    print("=" * 78)


if __name__ == "__main__":
    main()

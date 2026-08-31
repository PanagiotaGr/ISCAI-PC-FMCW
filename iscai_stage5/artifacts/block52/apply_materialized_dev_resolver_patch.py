from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

RUNNER = (
    STAGE5
    / "scripts/"
      "run_block52_part2_freeze.py"
)

SAFE = (
    STAGE5
    / "scripts/"
      "run_block52_part2_safe.py"
)

DEV_POSTERIOR = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.npz"
)

DEV_POSTERIOR_REPORT = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.json"
)

FINAL_REPORT = (
    STAGE5
    / "reports/"
      "block52_receiver_angular_posterior.json"
)

POLICY = (
    STAGE5
    / "configs/"
      "angular_posterior_policy.json"
)

CONVERGENCE = (
    STAGE5
    / "artifacts/block52/"
      "mc_convergence.json"
)

PATCH_REPORT = (
    STAGE5
    / "artifacts/block52/"
      "materialized_dev_resolver_patch.json"
)

FINAL_LOG = (
    STAGE5
    / "artifacts/block52/"
      "final_freeze_after_materialized_resolver.log"
)


EXPECTED_POSTERIOR_FILE_SHA = (
    "39dacd4a1dd83ac995124432840338442"
    "1731dcffc1cf01617bcc6c6395460b4"
)

EXPECTED_POSTERIOR_CONTENT_SHA = (
    "e63e7f14be6e119587716a4e6b25d492"
    "c154ec1881aa75225a454fd05964e4b9"
)

EXPECTED_BLOCK44_PREDICTION_SHA = (
    "a0e96b4d9ba051bb3ceb0e1270d592fc"
    "13cfa0b3b4ddeb3dc16ed7a29a82e66a"
)

EXPECTED_DEVELOPMENT_N = (
    13682
)

PATCH_MARKER = (
    "STAGE5_EXACT_MATERIALIZED_"
    "DEV_POSTERIOR_RESOLVER_V1"
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


def blocked(
    phase,
    reason,
):
    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 MATERIALIZED-RESOLVER REPAIR"
    )
    print(
        "============================================================"
    )

    print(
        "STATUS = BLOCKED"
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
        "formal N=120 used = NO"
    )

    print(
        "additional Stage4 inference = NO"
    )

    print(
        "training/recalibration = NO"
    )

    print(
        "Stage4 modified = NO"
    )

    print(
        "terminal remains open = YES"
    )


def validate_materialization():
    require(
        DEV_POSTERIOR.is_file(),
        (
            "Materialized development "
            "posterior NPZ missing."
        ),
    )

    require(
        DEV_POSTERIOR_REPORT.is_file(),
        (
            "Materialized development "
            "posterior report missing."
        ),
    )

    file_sha = file_sha256(
        DEV_POSTERIOR
    )

    require(
        file_sha
        ==
        EXPECTED_POSTERIOR_FILE_SHA,
        (
            "Materialized posterior file "
            "SHA changed."
        ),
    )

    report = load_json(
        DEV_POSTERIOR_REPORT
    )

    require(
        report.get(
            "status"
        )
        ==
        "PASS",
        (
            "Materialization report "
            "is not PASS."
        ),
    )

    reproduction = report.get(
        "historical_reproduction",
        {}
    )

    require(
        reproduction.get(
            "exact_match"
        )
        is True,
        (
            "Historical Block4.4 exact "
            "prediction reproduction "
            "is not PASS."
        ),
    )

    require(
        reproduction.get(
            "sample_limit"
        )
        ==
        2048,
        (
            "Historical hash sample "
            "limit changed."
        ),
    )

    require(
        reproduction.get(
            "batch_size"
        )
        ==
        512,
        (
            "Historical hash batch "
            "size changed."
        ),
    )

    require(
        reproduction.get(
            "expected_sha256"
        )
        ==
        EXPECTED_BLOCK44_PREDICTION_SHA,
        (
            "Historical expected "
            "prediction SHA changed."
        ),
    )

    require(
        reproduction.get(
            "actual_sha256"
        )
        ==
        EXPECTED_BLOCK44_PREDICTION_SHA,
        (
            "Historical actual "
            "prediction SHA changed."
        ),
    )

    scientific = report.get(
        "scientific_execution",
        {}
    )

    require(
        scientific.get(
            "development_only"
        )
        is True,
        (
            "Materialization was not "
            "explicitly development-only."
        ),
    )

    require(
        scientific.get(
            "formal_N120_read"
        )
        is False,
        (
            "Materialization does not "
            "prove formal_N120_read=False."
        ),
    )

    require(
        scientific.get(
            "training"
        )
        is False,
        (
            "Unexpected training during "
            "materialization."
        ),
    )

    require(
        scientific.get(
            "recalibration"
        )
        is False,
        (
            "Unexpected recalibration during "
            "materialization."
        ),
    )

    require(
        scientific.get(
            "Stage4_modified"
        )
        is False,
        (
            "Materialization report does "
            "not prove Stage4_modified=False."
        ),
    )

    coordinate = report.get(
        "coordinate_semantics",
        {}
    )

    require(
        coordinate.get(
            "Stage4_prediction"
        )
        ==
        "metric_H0_displacement",
        (
            "Stage4 prediction-coordinate "
            "semantics changed."
        ),
    )

    require(
        coordinate.get(
            "future_truth_used"
        )
        is False,
        (
            "Materialization coordinate "
            "conversion used future truth."
        ),
    )

    posterior = report.get(
        "posterior",
        {}
    )

    require(
        posterior.get(
            "development_N"
        )
        ==
        EXPECTED_DEVELOPMENT_N,
        (
            "Development posterior "
            "sample count changed."
        ),
    )

    require(
        posterior.get(
            "artifact_file_sha256"
        )
        ==
        EXPECTED_POSTERIOR_FILE_SHA,
        (
            "Materialization report "
            "artifact SHA mismatch."
        ),
    )

    require(
        posterior.get(
            "content_sha256"
        )
        ==
        EXPECTED_POSTERIOR_CONTENT_SHA,
        (
            "Materialized posterior "
            "content SHA changed."
        ),
    )

    return report


DISCOVER_REPLACEMENT = r'''
def discover_dev_gaussian_npz():
    """
    STAGE5_EXACT_MATERIALIZED_DEV_POSTERIOR_RESOLVER_V1

    Resolve the Stage5-only materialization that was produced
    by exact reproduction of the frozen Block4.4 development
    inference route.

    This function performs no model inference and never
    accesses the formal N=120 population.
    """

    path = (
        STAGE5
        / "artifacts/block52/"
          "development_gaussian_posterior.npz"
    )

    report_path = (
        STAGE5
        / "artifacts/block52/"
          "development_gaussian_posterior.json"
    )

    require(
        path.is_file(),
        (
            "Exact materialized development "
            "posterior is missing."
        ),
    )

    require(
        report_path.is_file(),
        (
            "Exact materialization provenance "
            "report is missing."
        ),
    )

    expected_file_sha = (
        "39dacd4a1dd83ac995124432840338442"
        "1731dcffc1cf01617bcc6c6395460b4"
    )

    expected_content_sha = (
        "e63e7f14be6e119587716a4e6b25d492"
        "c154ec1881aa75225a454fd05964e4b9"
    )

    expected_prediction_sha = (
        "a0e96b4d9ba051bb3ceb0e1270d592fc"
        "13cfa0b3b4ddeb3dc16ed7a29a82e66a"
    )

    actual_file_sha = file_sha256(
        path
    )

    require(
        actual_file_sha
        ==
        expected_file_sha,
        (
            "Materialized development "
            "posterior SHA changed."
        ),
    )

    materialization = load_json(
        report_path
    )

    require(
        materialization.get(
            "status"
        )
        ==
        "PASS",
        (
            "Materialization report "
            "is not PASS."
        ),
    )

    reproduction = materialization[
        "historical_reproduction"
    ]

    require(
        reproduction[
            "exact_match"
        ]
        is True,
        (
            "Frozen Block4.4 historical "
            "prediction was not exactly "
            "reproduced."
        ),
    )

    require(
        reproduction[
            "expected_sha256"
        ]
        ==
        expected_prediction_sha,
        (
            "Historical expected prediction "
            "SHA changed."
        ),
    )

    require(
        reproduction[
            "actual_sha256"
        ]
        ==
        expected_prediction_sha,
        (
            "Historical actual prediction "
            "SHA changed."
        ),
    )

    require(
        reproduction[
            "sample_limit"
        ]
        ==
        2048,
        (
            "Historical Block4.4 hash "
            "scope changed."
        ),
    )

    require(
        reproduction[
            "batch_size"
        ]
        ==
        512,
        (
            "Historical Block4.4 hash "
            "batch size changed."
        ),
    )

    scientific = materialization[
        "scientific_execution"
    ]

    require(
        scientific[
            "development_only"
        ]
        is True,
        (
            "Development evidence is not "
            "explicitly development-only."
        ),
    )

    require(
        scientific[
            "formal_N120_read"
        ]
        is False,
        (
            "Formal N=120 leakage detected "
            "in materialization provenance."
        ),
    )

    require(
        scientific[
            "training"
        ]
        is False,
        "Unexpected training provenance.",
    )

    require(
        scientific[
            "recalibration"
        ]
        is False,
        "Unexpected recalibration provenance.",
    )

    require(
        scientific[
            "Stage4_modified"
        ]
        is False,
        (
            "Materialization does not prove "
            "Stage4 remained unchanged."
        ),
    )

    require(
        materialization[
            "coordinate_semantics"
        ][
            "Stage4_prediction"
        ]
        ==
        "metric_H0_displacement",
        (
            "Stage4 prediction-coordinate "
            "semantics changed."
        ),
    )

    require(
        materialization[
            "coordinate_semantics"
        ][
            "future_truth_used"
        ]
        is False,
        (
            "Future truth entered absolute "
            "H0 conversion."
        ),
    )

    require(
        materialization[
            "posterior"
        ][
            "development_N"
        ]
        ==
        13682,
        (
            "Development posterior "
            "sample count changed."
        ),
    )

    require(
        materialization[
            "posterior"
        ][
            "content_sha256"
        ]
        ==
        expected_content_sha,
        (
            "Development posterior "
            "content SHA changed."
        ),
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:

        required_keys = (
            "anchor_position_H0_m",
            "mean_displacement_H0_m",
            "mean_absolute_H0_m",
            "raw_covariance_H0_m2",
            "calibrated_covariance_H0_m2",
            "variance_scale_alpha_h",
        )

        missing = [
            key
            for key in required_keys
            if key not in payload
        ]

        require(
            not missing,
            (
                "Materialized posterior "
                "missing key(s): "
                +
                ", ".join(
                    missing
                )
            ),
        )

        mean = np.asarray(
            payload[
                "mean_absolute_H0_m"
            ]
        )

        covariance = np.asarray(
            payload[
                "calibrated_covariance_H0_m2"
            ]
        )

        alpha = np.asarray(
            payload[
                "variance_scale_alpha_h"
            ],
            dtype=np.float64,
        )

        #
        # Formal/evaluator labels must not be present in this
        # numerical-convergence posterior artifact.
        #
        forbidden_keys = (
            "future",
            "future_mask",
            "tracks_to_predict",
            "objects_of_interest",
            "ground_truth",
        )

        for key in forbidden_keys:

            require(
                key not in payload,
                (
                    "Forbidden evaluator/formal "
                    f"field present: {key}"
                ),
            )

    require(
        mean.shape
        ==
        (
            13682,
            4,
            3,
        ),
        (
            "Absolute development posterior "
            "mean shape changed."
        ),
    )

    require(
        covariance.shape
        ==
        (
            13682,
            4,
            3,
            3,
        ),
        (
            "Calibrated development "
            "covariance shape changed."
        ),
    )

    require(
        np.all(
            np.isfinite(
                mean
            )
        ),
        (
            "Absolute development posterior "
            "mean contains non-finite values."
        ),
    )

    require(
        np.all(
            np.isfinite(
                covariance
            )
        ),
        (
            "Calibrated development "
            "covariance contains non-finite values."
        ),
    )

    expected_alpha = np.asarray(
        (
            1.2347064500315355,
            1.3451250295202921,
            1.354822057728461,
            1.2829700217319266,
        ),
        dtype=np.float64,
    )

    require(
        np.allclose(
            alpha,
            expected_alpha,
            rtol=0.0,
            atol=1e-7,
        ),
        (
            "Frozen Stage4 variance "
            "calibration changed."
        ),
    )

    return {
        "path":
            path,

        "sha256":
            actual_file_sha,

        "mean_key":
            "mean_absolute_H0_m",

        "covariance_key":
            "calibrated_covariance_H0_m2",

        "sample_count":
            13682,

        "discovery_score":
            "exact_frozen_Block4.4_materialization",

        "diagnostics": {
            "materialization_report":
                str(
                    report_path
                ),

            "historical_prediction_sha256":
                expected_prediction_sha,

            "historical_hash_sample_limit":
                2048,

            "historical_hash_batch_size":
                512,

            "posterior_content_sha256":
                expected_content_sha,

            "coordinate_semantics":
                "absolute_H0_after_causal_anchor_translation",

            "covariance_semantics":
                "already_calibrated_Stage4_predictive_covariance",

            "additional_Stage4_inference":
                False,

            "formal_N120_read":
                False,
        },
    }
'''.strip()


LOAD_REPLACEMENT = r'''
def load_dev_arrays(
    discovery,
):
    """
    Load the exact Stage5-only development posterior.

    Important:
      covariance is ALREADY calibrated by the frozen
      Block4.5 alpha_h values.

    Therefore this loader MUST NOT call
    calibrate_stage4_horizon_covariances again.
    """

    path = discovery[
        "path"
    ]

    require(
        file_sha256(
            path
        )
        ==
        discovery[
            "sha256"
        ],
        (
            "Development posterior changed "
            "between discovery and load."
        ),
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as payload:

        means = np.asarray(
            payload[
                discovery[
                    "mean_key"
                ]
            ],
            dtype=np.float64,
        )

        calibrated_covariance = np.asarray(
            payload[
                discovery[
                    "covariance_key"
                ]
            ],
            dtype=np.float64,
        )

    require(
        means.shape
        ==
        (
            13682,
            4,
            3,
        ),
        (
            "Materialized absolute H0 "
            "mean shape mismatch."
        ),
    )

    require(
        calibrated_covariance.shape
        ==
        (
            13682,
            4,
            3,
            3,
        ),
        (
            "Materialized calibrated "
            "covariance shape mismatch."
        ),
    )

    require(
        np.all(
            np.isfinite(
                means
            )
        ),
        (
            "Materialized absolute H0 "
            "means contain non-finite values."
        ),
    )

    require(
        np.all(
            np.isfinite(
                calibrated_covariance
            )
        ),
        (
            "Materialized calibrated "
            "covariance contains non-finite values."
        ),
    )

    symmetric = np.allclose(
        calibrated_covariance,
        np.swapaxes(
            calibrated_covariance,
            -1,
            -2,
        ),
        atol=2e-6,
        rtol=0.0,
    )

    require(
        symmetric,
        (
            "Materialized calibrated "
            "covariance is not symmetric."
        ),
    )

    #
    # No second covariance calibration here.
    #
    return (
        means,
        calibrated_covariance,
    )
'''.strip()


def replace_functions(
    source,
):
    tree = ast.parse(
        source
    )

    targets = {}

    for node in tree.body:

        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        if node.name in (
            "discover_dev_gaussian_npz",
            "load_dev_arrays",
        ):
            targets[
                node.name
            ] = node

    require(
        set(
            targets
        )
        ==
        {
            "discover_dev_gaussian_npz",
            "load_dev_arrays",
        },
        (
            "Could not resolve exact "
            "resolver functions in current "
            "Block5.2 freeze runner."
        ),
    )

    lines = source.splitlines(
        keepends=True
    )

    replacements = {
        "discover_dev_gaussian_npz":
            DISCOVER_REPLACEMENT,

        "load_dev_arrays":
            LOAD_REPLACEMENT,
    }

    nodes = sorted(
        targets.items(),
        key=lambda item:
            item[
                1
            ].lineno,
        reverse=True,
    )

    for name, node in nodes:

        start = (
            int(
                node.lineno
            )
            -
            1
        )

        end = int(
            node.end_lineno
        )

        replacement = (
            replacements[
                name
            ]
            +
            "\n"
        )

        lines[
            start:end
        ] = [
            replacement
        ]

    return "".join(
        lines
    )


def semantic_report_repair(
    source,
):
    #
    # The final freeze itself performs no Stage4 inference,
    # but its source development posterior was materialized
    # earlier through exact frozen Stage4 inference.
    #
    replacements = (
        (
            '"frozen_non_formal_Block4.4_development_predictions"',
            (
                '"Stage5_materialized_from_exact_frozen_'
                'Block4.4_development_inference"'
            ),
        ),

        (
            '        "Stage4_inference_performed":\n'
            '            False,',
            (
                '        "Stage4_inference_performed_in_this_freeze_run":\n'
                '            False,\n\n'
                '        "source_materialization_used_frozen_Stage4_inference":\n'
                '            True,'
            ),
        ),

        (
            '            "Stage4_inference_performed":\n'
            '                False,',
            (
                '            "Stage4_inference_performed_in_this_freeze_run":\n'
                '                False,\n\n'
                '            "source_materialization_used_frozen_Stage4_inference":\n'
                '                True,'
            ),
        ),

        (
            '            "model_inference_performed":\n'
            '                False,',
            (
                '            "model_inference_performed_in_this_freeze_run":\n'
                '                False,\n\n'
                '            "source_materialization_used_frozen_Stage4_inference":\n'
                '                True,'
            ),
        ),

        (
            '            "Stage4_model_inference":\n'
            '                False,',
            (
                '            "Stage4_model_inference_in_this_freeze_run":\n'
                '                False,\n\n'
                '            "development_source_used_frozen_Stage4_inference":\n'
                '                True,'
            ),
        ),

        (
            '"Stage4 model inference    = NO"',
            '"Stage4 inference this run = NO"',
        ),

        (
            '"Stage4 model inference        = NO"',
            '"Stage4 inference this run     = NO"',
        ),
    )

    counts = {}

    for old, new in replacements:

        count = source.count(
            old
        )

        counts[
            old[
                :80
            ]
        ] = count

        if count:
            source = source.replace(
                old,
                new,
            )

    return (
        source,
        counts,
    )


def run_tests(
    pattern,
):
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE5
                / "tests"
            ),
            "-p",
            pattern,
        ],
        cwd=str(
            STAGE5
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

    return (
        process.returncode,
        count,
    )


def stream_final_safe():
    process = subprocess.Popen(
        [
            sys.executable,
            str(
                SAFE
            ),
        ],
        cwd=str(
            STAGE5
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    with FINAL_LOG.open(
        "w",
        encoding="utf-8",
    ) as log:

        if process.stdout is not None:

            for line in process.stdout:

                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

    return process.wait()


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — MATERIALIZED DEV RESOLVER REPAIR"
    )
    print(
        "============================================================"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "additional Stage4 inference   = NO"
    )

    print(
        "training/recalibration        = NO"
    )

    print(
        "Stage4 modification           = NO"
    )

    # ========================================================
    # A. Materialization integrity
    # ========================================================

    print()
    print(
        "===== A. MATERIALIZED DEVELOPMENT POSTERIOR ====="
    )

    materialization = (
        validate_materialization()
    )

    print(
        "historical Block4.4 SHA = EXACT PASS"
    )

    print(
        "historical hash scope   = FIRST 2048 / BATCH 512"
    )

    print(
        "posterior file SHA      = PASS"
    )

    print(
        "posterior content SHA   = PASS"
    )

    print(
        "development N           = 13682"
    )

    print(
        "absolute H0 mean        = PASS"
    )

    print(
        "calibrated covariance   = PASS"
    )

    print(
        "formal N=120 used       = NO"
    )

    # ========================================================
    # B. Targeted runner patch
    # ========================================================

    print()
    print(
        "===== B. TARGETED CONVERGENCE RESOLVER PATCH ====="
    )

    require(
        RUNNER.is_file(),
        (
            "Block5.2 final freeze runner "
            "is missing."
        ),
    )

    original_source = RUNNER.read_text(
        encoding="utf-8"
    )

    original_sha = file_sha256(
        RUNNER
    )

    if PATCH_MARKER in original_source:

        print(
            "resolver patch          = ALREADY APPLIED"
        )

        patched_source = (
            original_source
        )

        semantic_counts = {}

        backup_path = None

    else:

        backup_path = (
            STAGE5
            / "artifacts/block52/"
              f"run_block52_part2_freeze."
              f"pre_materialized_resolver."
              f"{original_sha[:16]}.py"
        )

        if not backup_path.is_file():

            shutil.copy2(
                RUNNER,
                backup_path,
            )

        patched_source = replace_functions(
            original_source
        )

        (
            patched_source,
            semantic_counts,
        ) = semantic_report_repair(
            patched_source
        )

        require(
            PATCH_MARKER
            in
            patched_source,
            (
                "Patched resolver marker "
                "was not inserted."
            ),
        )

        temporary = RUNNER.with_suffix(
            ".py.tmp"
        )

        temporary.write_text(
            patched_source,
            encoding="utf-8",
        )

        temporary.replace(
            RUNNER
        )

        print(
            "resolver patch          = APPLIED"
        )

        print(
            "backup                  =",
            backup_path,
        )

    patched_sha = file_sha256(
        RUNNER
    )

    print(
        "runner old SHA256       =",
        original_sha,
    )

    print(
        "runner new SHA256       =",
        patched_sha,
    )

    print(
        "double calibration      = FORBIDDEN / REMOVED"
    )

    print(
        "resolver mean key       = mean_absolute_H0_m"
    )

    print(
        "resolver covariance key = calibrated_covariance_H0_m2"
    )

    # ========================================================
    # C. Controlled compile
    # ========================================================

    print()
    print(
        "===== C. CONTROLLED COMPILE ====="
    )

    for path in (
        RUNNER,
        SAFE,
    ):

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

    # ========================================================
    # D. Regression before final freeze
    # ========================================================

    print()
    print(
        "===== D. BLOCK5.2 PART2 TESTS ====="
    )

    code, count = run_tests(
        "test_block52_monte_carlo.py"
    )

    require(
        code == 0
        and
        count == 18,
        (
            "Expected Block5.2 Part2 "
            f"18/18 tests; got {count}."
        ),
    )

    print(
        "Block5.2 Part2 tests = 18 / 18 PASS"
    )

    print()
    print(
        "===== E. PRE-FREEZE STAGE5 REGRESSION ====="
    )

    code, count = run_tests(
        "test_*.py"
    )

    require(
        code == 0
        and
        count == 72,
        (
            "Expected Stage5 regression "
            f"72/72; got {count}."
        ),
    )

    print(
        "pre-freeze Stage5 regression = 72 / 72 PASS"
    )

    # ========================================================
    # F. Patch provenance
    # ========================================================

    patch_payload = {
        "status":
            "PASS",

        "purpose":
            (
                "replace_nonexistent_Block44_"
                "dev_prediction_NPZ_resolver_with_"
                "exact_Stage5_materialized_"
                "frozen_Block44_dev_posterior"
            ),

        "scientific_execution": {
            "formal_N120_read":
                False,

            "additional_Stage4_inference":
                False,

            "training":
                False,

            "recalibration":
                False,

            "Stage4_modified":
                False,
        },

        "materialized_posterior": {
            "path":
                str(
                    DEV_POSTERIOR
                ),

            "file_sha256":
                EXPECTED_POSTERIOR_FILE_SHA,

            "content_sha256":
                EXPECTED_POSTERIOR_CONTENT_SHA,

            "development_N":
                EXPECTED_DEVELOPMENT_N,

            "mean_key":
                "mean_absolute_H0_m",

            "covariance_key":
                "calibrated_covariance_H0_m2",

            "covariance_already_calibrated":
                True,

            "second_calibration":
                False,

            "historical_Block44_prediction_SHA256":
                EXPECTED_BLOCK44_PREDICTION_SHA,
        },

        "runner": {
            "path":
                str(
                    RUNNER
                ),

            "pre_patch_sha256":
                original_sha,

            "post_patch_sha256":
                patched_sha,

            "backup":
                (
                    str(
                        backup_path
                    )
                    if backup_path
                    is not None
                    else
                    None
                ),
        },

        "semantic_repair_counts":
            semantic_counts,

        "regression_before_final_freeze":
            "72/72_PASS",
    }

    write_json(
        PATCH_REPORT,
        patch_payload,
    )

    print(
        "patch report =",
        PATCH_REPORT,
    )

    # ========================================================
    # G. Existing final Block5.2 safe freeze
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "G. FINAL BLOCK5.2 FREEZE"
    )
    print(
        "============================================================"
    )

    raw_code = stream_final_safe()

    print()
    print(
        "final safe raw code =",
        raw_code,
    )

    # ========================================================
    # H. Structured final readback
    # ========================================================

    if not FINAL_REPORT.is_file():

        blocked(
            "final_structured_report",
            (
                "Final Block5.2 report "
                "was not written."
            ),
        )

        return

    if not POLICY.is_file():

        blocked(
            "final_policy",
            (
                "Angular-posterior policy "
                "was not frozen."
            ),
        )

        return

    if not CONVERGENCE.is_file():

        blocked(
            "MC_convergence",
            (
                "MC convergence artifact "
                "was not written."
            ),
        )

        return

    final_report = load_json(
        FINAL_REPORT
    )

    policy = load_json(
        POLICY
    )

    convergence = load_json(
        CONVERGENCE
    )

    if (
        final_report.get(
            "status"
        )
        !=
        "PASS"
    ):

        blocked(
            "final_scientific_gate",
            (
                "Final Block5.2 report "
                "status is not PASS."
            ),
        )

        return

    if (
        policy.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        blocked(
            "angular_policy",
            (
                "Angular-posterior policy "
                "is not FROZEN."
            ),
        )

        return

    if (
        convergence.get(
            "status"
        )
        !=
        "PASS"
    ):

        blocked(
            "MC_convergence",
            (
                "Development MC convergence "
                "is not PASS."
            ),
        )

        return

    selected_n = (
        policy[
            "Monte_Carlo"
        ][
            "sample_count"
        ]
    )

    require(
        selected_n
        in
        (
            2048,
            4096,
            8192,
        ),
        (
            "Frozen MC sample count "
            "is outside preregistered "
            "candidate set."
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 RESOLVER-REPAIRED FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "historical Block4.4 SHA      = EXACT PASS"
    )

    print(
        "materialized dev posterior   = PASS"
    )

    print(
        "absolute H0 semantics        = PASS"
    )

    print(
        "double covariance calibration= NO"
    )

    print(
        "development MC convergence   = PASS"
    )

    print(
        "MC sample count              =",
        selected_n,
        "FROZEN",
    )

    print(
        "formal N=120 used            = NO"
    )

    print(
        "additional Stage4 inference  = NO"
    )

    print(
        "training/recalibration       = NO"
    )

    print(
        "Stage4 modified              = NO"
    )

    print(
        "final implementation SHA256  =",
        final_report[
            "implementation"
        ][
            "sha256"
        ],
    )

    print(
        "angular policy SHA256        =",
        final_report[
            "angular_policy"
        ][
            "sha256"
        ],
    )

    print(
        "convergence SHA256           =",
        final_report[
            "development_convergence"
        ][
            "artifact_sha256"
        ],
    )

    print(
        "Stage5 regression            = 72 / 72 PASS"
    )

    print(
        "scientific STATUS            = PASS"
    )

    print(
        "Block5.2                     = COMPLETE / FROZEN"
    )

    print(
        "next                         = BLOCK 5.3"
    )

    print(
        "terminal remains open        = YES"
    )


try:
    main()

except BaseException as exc:

    blocked(
        "unexpected_repair_controller_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

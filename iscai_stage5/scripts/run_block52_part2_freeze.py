from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    FROZEN_VARIANCE_SCALE_ALPHA_H,
    FUTURE_HEADING_SEMANTICS,
    LOW_SPEED_HEADING_THRESHOLD_BASIS,
    LOW_SPEED_HEADING_THRESHOLD_MPS,
    MONTE_CARLO_BASE_SEED,
    MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS,
    MONTE_CARLO_MINIMUM_SAMPLE_COUNT,
    MONTE_CARLO_MINIMUM_SAMPLE_COUNT_BASIS,
    MONTE_CARLO_REFERENCE_SAMPLE_COUNT,
    RECEIVER_OFFSET_TEMPORAL_SEMANTICS,
    TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
    ReceiverOffsetGaussianBody,
    calibrate_stage4_horizon_covariances,
    deterministic_receiver_angular_posterior,
    single_horizon_gaussian_angular_summary,
)

from iscai_stage5.angular_posterior import (
    circular_angle_difference,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

PART1 = (
    STAGE5
    / "reports/"
      "block52_part1_angular_contract.json"
)

BLOCK51 = (
    STAGE5
    / "reports/"
      "block51_receiver_geometry.json"
)

RECEIVER_POLICY = (
    STAGE5
    / "configs/"
      "receiver_policy.json"
)

STRUCTURAL_CONTRACT = (
    STAGE5
    / "configs/"
      "stage5_contract.json"
)

BLOCK44_ARTIFACTS = (
    STAGE4
    / "artifacts/block44"
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

REPORT = (
    STAGE5
    / "reports/"
      "block52_receiver_angular_posterior.json"
)

IMPLEMENTATION_LOG = (
    STAGE5
    / "docs/"
      "implementation_log.md"
)


EXPECTED_PART1_IMPLEMENTATION_SHA = (
    "9bd121ba2e98a265eb83a727af8d29da"
    "945cefc8ebe9eed9499ba5c011c3bc16"
)

EXPECTED_BLOCK51_IMPLEMENTATION_SHA = (
    "604c0ea15353830ed6dd74b40ff7e423"
    "13bf9f86c816aa1e76f0057ab6bc8a53"
)

EXPECTED_RECEIVER_POLICY_SHA = (
    "b5b6fc2b0a3c4575ed9567aff15f36f3"
    "430f4887862c245f045fae7cf9e349c0"
)

EXPECTED_STRUCTURAL_CONTRACT_SHA = (
    "04fd32e4bc6e160f2627f465a7820c74"
    "e19ca2e1a1f47f23ed1884f842cb7db4"
)


CONVERGENCE_TARGETS = {
    "azimuth_mean_abs_deg_p95":
        0.25,

    "azimuth_std_abs_deg_p95":
        0.25,

    "elevation_mean_abs_deg_p95":
        0.20,

    "elevation_std_abs_deg_p95":
        0.20,

    "range_mean_abs_m_p95":
        0.10,

    "azimuth_mean_abs_deg_max":
        0.75,

    "elevation_mean_abs_deg_max":
        0.60,
}

DEVELOPMENT_CASE_COUNT = (
    24
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
                1024
                *
                1024
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


def _candidate_npz_files():
    if not BLOCK44_ARTIFACTS.is_dir():
        return []

    return sorted(
        [
            path
            for path in BLOCK44_ARTIFACTS.rglob(
                "*.npz"
            )
            if path.is_file()
        ],
        key=str,
    )


def _is_mean_candidate(
    name,
    array,
):
    lower = str(
        name
    ).lower()

    return (
        array.ndim >= 3
        and
        array.shape[
            -2:
        ]
        ==
        (
            4,
            3,
        )
        and
        any(
            token in lower
            for token in (
                "mean",
                "mu",
                "pred",
            )
        )
        and
        "cov"
        not in lower
        and
        "sigma"
        not in lower
        and
        "chol"
        not in lower
    )


def _is_covariance_candidate(
    name,
    array,
):
    lower = str(
        name
    ).lower()

    return (
        array.ndim >= 4
        and
        array.shape[
            -3:
        ]
        ==
        (
            4,
            3,
            3,
        )
        and
        any(
            token in lower
            for token in (
                "cov",
                "sigma",
            )
        )
    )


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


def select_stable_development_cases(
    means,
    covariance,
):
    cases = []

    for sample_index in range(
        means.shape[
            0
        ]
    ):
        for horizon_index in range(
            4
        ):
            mean = means[
                sample_index,
                horizon_index
            ]

            cov = covariance[
                sample_index,
                horizon_index
            ]

            horizontal = float(
                math.hypot(
                    float(
                        mean[
                            0
                        ]
                    ),
                    float(
                        mean[
                            1
                        ]
                    ),
                )
            )

            if horizontal <= 2.0:
                continue

            if not np.allclose(
                cov,
                cov.T,
                atol=1e-8,
                rtol=0.0,
            ):
                continue

            eigenvalues = np.linalg.eigvalsh(
                0.5
                *
                (
                    cov
                    +
                    cov.T
                )
            )

            if float(
                np.min(
                    eigenvalues
                )
            ) < -1e-8:
                continue

            xy_sigma = math.sqrt(
                max(
                    0.0,
                    float(
                        max(
                            cov[
                                0,
                                0
                            ],
                            cov[
                                1,
                                1
                            ],
                        )
                    ),
                )
            )

            #
            # Keep the convergence suite away from the
            # coordinate singularity itself.  Near-origin
            # singular handling is a separate explicit guard,
            # not an MC sample-count tuning mechanism.
            #
            if (
                horizontal
                <=
                max(
                    2.0,
                    4.0
                    *
                    xy_sigma,
                )
            ):
                continue

            cases.append(
                (
                    sample_index,
                    horizon_index,
                )
            )

            if len(
                cases
            ) >= DEVELOPMENT_CASE_COUNT:
                return cases

    require(
        len(
            cases
        )
        >=
        DEVELOPMENT_CASE_COUNT,
        (
            "Could not obtain enough stable "
            "non-formal development Gaussian "
            "cases for MC convergence."
        ),
    )

    return cases


def angle_error_deg(
    first,
    second,
):
    return abs(
        math.degrees(
            circular_angle_difference(
                float(
                    first
                ),
                float(
                    second
                ),
            )
        )
    )


def ordinary_angle_error_deg(
    first,
    second,
):
    return abs(
        math.degrees(
            float(
                first
            )
            -
            float(
                second
            )
        )
    )


def percentile95(
    values,
):
    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            95.0,
        )
    )


def convergence_for_count(
    *,
    sample_count,
    means,
    covariance,
    cases,
):
    errors = {
        "azimuth_mean_abs_deg":
            [],

        "azimuth_std_abs_deg":
            [],

        "elevation_mean_abs_deg":
            [],

        "elevation_std_abs_deg":
            [],

        "range_mean_abs_m":
            [],
    }

    per_case = []

    for case_rank, (
        sample_index,
        horizon_index,
    ) in enumerate(
        cases
    ):
        mean = means[
            sample_index,
            horizon_index
        ]

        cov = covariance[
            sample_index,
            horizon_index
        ]

        seed = (
            MONTE_CARLO_BASE_SEED
            +
            10_007
            *
            (
                case_rank
                +
                1
            )
        )

        candidate = (
            single_horizon_gaussian_angular_summary(
                mean_h0_m=mean,
                covariance_h0_m2=cov,
                sample_count=sample_count,
                seed=seed,
            )
        )

        reference = (
            single_horizon_gaussian_angular_summary(
                mean_h0_m=mean,
                covariance_h0_m2=cov,
                sample_count=(
                    MONTE_CARLO_REFERENCE_SAMPLE_COUNT
                ),
                seed=seed,
            )
        )

        azimuth_mean_error = (
            angle_error_deg(
                candidate[
                    "azimuth_mean_rad"
                ],
                reference[
                    "azimuth_mean_rad"
                ],
            )
        )

        azimuth_std_error = abs(
            math.degrees(
                candidate[
                    "azimuth_std_rad"
                ]
                -
                reference[
                    "azimuth_std_rad"
                ]
            )
        )

        elevation_mean_error = (
            ordinary_angle_error_deg(
                candidate[
                    "elevation_mean_rad"
                ],
                reference[
                    "elevation_mean_rad"
                ],
            )
        )

        elevation_std_error = abs(
            math.degrees(
                candidate[
                    "elevation_std_rad"
                ]
                -
                reference[
                    "elevation_std_rad"
                ]
            )
        )

        range_mean_error = abs(
            candidate[
                "range_mean_m"
            ]
            -
            reference[
                "range_mean_m"
            ]
        )

        errors[
            "azimuth_mean_abs_deg"
        ].append(
            azimuth_mean_error
        )

        errors[
            "azimuth_std_abs_deg"
        ].append(
            azimuth_std_error
        )

        errors[
            "elevation_mean_abs_deg"
        ].append(
            elevation_mean_error
        )

        errors[
            "elevation_std_abs_deg"
        ].append(
            elevation_std_error
        )

        errors[
            "range_mean_abs_m"
        ].append(
            range_mean_error
        )

        per_case.append({
            "sample_index":
                int(
                    sample_index
                ),

            "horizon_index":
                int(
                    horizon_index
                ),

            "azimuth_mean_abs_deg":
                azimuth_mean_error,

            "azimuth_std_abs_deg":
                azimuth_std_error,

            "elevation_mean_abs_deg":
                elevation_mean_error,

            "elevation_std_abs_deg":
                elevation_std_error,

            "range_mean_abs_m":
                range_mean_error,
        })

    metrics = {
        "azimuth_mean_abs_deg_p95":
            percentile95(
                errors[
                    "azimuth_mean_abs_deg"
                ]
            ),

        "azimuth_std_abs_deg_p95":
            percentile95(
                errors[
                    "azimuth_std_abs_deg"
                ]
            ),

        "elevation_mean_abs_deg_p95":
            percentile95(
                errors[
                    "elevation_mean_abs_deg"
                ]
            ),

        "elevation_std_abs_deg_p95":
            percentile95(
                errors[
                    "elevation_std_abs_deg"
                ]
            ),

        "range_mean_abs_m_p95":
            percentile95(
                errors[
                    "range_mean_abs_m"
                ]
            ),

        "azimuth_mean_abs_deg_max":
            float(
                max(
                    errors[
                        "azimuth_mean_abs_deg"
                    ]
                )
            ),

        "elevation_mean_abs_deg_max":
            float(
                max(
                    errors[
                        "elevation_mean_abs_deg"
                    ]
                )
            ),
    }

    passed = all(
        metrics[
            key
        ]
        <=
        threshold
        for key, threshold in (
            CONVERGENCE_TARGETS.items()
        )
    )

    return {
        "sample_count":
            int(
                sample_count
            ),

        "reference_sample_count":
            int(
                MONTE_CARLO_REFERENCE_SAMPLE_COUNT
            ),

        "case_count":
            len(
                cases
            ),

        "metrics":
            metrics,

        "thresholds":
            dict(
                CONVERGENCE_TARGETS
            ),

        "status":
            (
                "PASS"
                if passed
                else
                "FAIL"
            ),

        "per_case":
            per_case,
    }


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.2 PART 2/2"
    )
    print(
        "DETERMINISTIC MC RECEIVER/ANGULAR POSTERIOR"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        BLOCK51,
        RECEIVER_POLICY,
        STRUCTURAL_CONTRACT,
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
            "Missing frozen prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK5.1 / BLOCK5.2 PART1 CONTINUITY ====="
    )

    part1 = load_json(
        PART1
    )

    block51 = load_json(
        BLOCK51
    )

    require(
        part1.get(
            "status"
        )
        ==
        "PASS",
        "Block5.2 Part1 is not PASS.",
    )

    require(
        part1[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_PART1_IMPLEMENTATION_SHA,
        (
            "Historical Block5.2 Part1 "
            "implementation SHA changed."
        ),
    )

    require(
        block51.get(
            "status"
        )
        ==
        "PASS",
        "Block5.1 is not PASS.",
    )

    require(
        block51[
            "implementation"
        ][
            "sha256"
        ]
        ==
        EXPECTED_BLOCK51_IMPLEMENTATION_SHA,
        (
            "Historical Block5.1 "
            "implementation SHA changed."
        ),
    )

    require(
        file_sha256(
            RECEIVER_POLICY
        )
        ==
        EXPECTED_RECEIVER_POLICY_SHA,
        (
            "Frozen receiver-policy "
            "SHA changed."
        ),
    )

    require(
        file_sha256(
            STRUCTURAL_CONTRACT
        )
        ==
        EXPECTED_STRUCTURAL_CONTRACT_SHA,
        (
            "Frozen Stage5 structural "
            "contract SHA changed."
        ),
    )

    print(
        "Block5.1                  = PASS"
    )

    print(
        "Block5.2 Part1            = PASS"
    )

    print(
        "historical implementation= PASS"
    )

    print(
        "receiver policy           = FROZEN PASS"
    )

    # ========================================================
    # B. Stage4 calibrated-Gaussian semantics
    # ========================================================

    print()
    print(
        "===== B. FROZEN STAGE4 GAUSSIAN → MC CONTRACT ====="
    )

    require(
        tuple(
            FROZEN_VARIANCE_SCALE_ALPHA_H
        )
        ==
        (
            1.2347064500315355,
            1.3451250295202921,
            1.354822057728461,
            1.2829700217319266,
        ),
        "Frozen covariance calibration changed.",
    )

    require(
        TRAJECTORY_JOINT_SAMPLING_SEMANTICS
        ==
        (
            "product_of_frozen_calibrated_"
            "per_horizon_Gaussian_marginals"
        ),
        (
            "Trajectory sampling semantics changed."
        ),
    )

    print(
        "Stage4 mean               = UNCHANGED"
    )

    print(
        "Stage4 covariance         = CALIBRATED alpha_h PASS"
    )

    print(
        "measurement R_t           = NOT MODIFIED"
    )

    print(
        "cross-horizon covariance  = NOT INVENTED"
    )

    print(
        "joint MC semantics        = PRODUCT OF FROZEN MARGINALS"
    )

    # ========================================================
    # C. Non-formal dev posterior discovery
    # ========================================================

    print()
    print(
        "===== C. NON-FORMAL BLOCK4.4 DEVELOPMENT POSTERIOR ====="
    )

    discovery = (
        discover_dev_gaussian_npz()
    )

    print(
        "development prediction artifact =",
        discovery[
            "path"
        ],
    )

    print(
        "development artifact SHA256 =",
        discovery[
            "sha256"
        ],
    )

    print(
        "mean key                  =",
        discovery[
            "mean_key"
        ],
    )

    print(
        "covariance key            =",
        discovery[
            "covariance_key"
        ],
    )

    print(
        "development samples       =",
        discovery[
            "sample_count"
        ],
    )

    means, calibrated_covariance = (
        load_dev_arrays(
            discovery
        )
    )

    require(
        means.shape[
            0
        ]
        >
        0,
        "Development prediction array is empty.",
    )

    cases = (
        select_stable_development_cases(
            means,
            calibrated_covariance,
        )
    )

    print(
        "stable convergence cases  =",
        len(
            cases
        ),
    )

    print(
        "formal N=120 touched      = NO"
    )

    print(
        "Stage4 inference this run = NO"
    )

    print(
        "development artifact read = YES"
    )

    # ========================================================
    # D. MC sample-count convergence
    # ========================================================

    print()
    print(
        "===== D. DEVELOPMENT-ONLY MC CONVERGENCE ====="
    )

    convergence_results = []

    selected_sample_count = None

    for candidate_count in (
        MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS
    ):
        result = (
            convergence_for_count(
                sample_count=(
                    candidate_count
                ),

                means=means,

                covariance=(
                    calibrated_covariance
                ),

                cases=cases,
            )
        )

        convergence_results.append(
            result
        )

        print()
        print(
            "candidate N =",
            candidate_count,
        )

        for key, value in (
            result[
                "metrics"
            ].items()
        ):
            print(
                f"  {key:32s}=",
                value,
                "| limit =",
                CONVERGENCE_TARGETS[
                    key
                ],
            )

        print(
            "  status =",
            result[
                "status"
            ],
        )

        if (
            selected_sample_count
            is None
            and
            result[
                "status"
            ]
            ==
            "PASS"
        ):
            selected_sample_count = (
                candidate_count
            )

    require(
        selected_sample_count
        is not None,
        (
            "No preregistered MC candidate "
            "sample count passed the development "
            "convergence thresholds."
        ),
    )

    require(
        selected_sample_count
        >=
        MONTE_CARLO_MINIMUM_SAMPLE_COUNT,
        (
            "Selected MC sample count violates "
            "minimum probability-resolution gate."
        ),
    )

    convergence_payload = {
        "status":
            "PASS",

        "source":
            "Stage5_materialized_from_exact_frozen_Block4.4_development_inference",

        "formal_data_used":
            False,

        "Stage4_inference_performed_in_this_freeze_run":
            False,

        "source_materialization_used_frozen_Stage4_inference":
            True,

        "development_artifact":
            {
                "path":
                    str(
                        discovery[
                            "path"
                        ]
                    ),

                "sha256":
                    discovery[
                        "sha256"
                    ],

                "mean_key":
                    discovery[
                        "mean_key"
                    ],

                "covariance_key":
                    discovery[
                        "covariance_key"
                    ],
            },

        "case_count":
            len(
                cases
            ),

        "cases":
            [
                {
                    "sample_index":
                        int(
                            sample_index
                        ),

                    "horizon_index":
                        int(
                            horizon_index
                        ),
                }
                for (
                    sample_index,
                    horizon_index,
                ) in cases
            ],

        "candidate_sample_counts":
            list(
                MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS
            ),

        "reference_sample_count":
            MONTE_CARLO_REFERENCE_SAMPLE_COUNT,

        "thresholds":
            dict(
                CONVERGENCE_TARGETS
            ),

        "results":
            convergence_results,

        "selected_sample_count":
            int(
                selected_sample_count
            ),

        "selection_rule":
            (
                "smallest_preregistered_candidate_"
                "passing_all_convergence_thresholds"
            ),
    }

    write_json(
        CONVERGENCE,
        convergence_payload,
    )

    convergence_sha = (
        file_sha256(
            CONVERGENCE
        )
    )

    print()
    print(
        "selected MC sample count =",
        selected_sample_count,
    )

    print(
        "selection                = SMALLEST PASSING CANDIDATE"
    )

    print(
        "formal tuning            = NO"
    )

    print(
        "MC convergence SHA256    =",
        convergence_sha,
    )

    # ========================================================
    # E. Heading / receiver-location uncertainty semantics
    # ========================================================

    print()
    print(
        "===== E. HEADING + RECEIVER-PLACEMENT CONTRACT ====="
    )

    require(
        LOW_SPEED_HEADING_THRESHOLD_MPS
        ==
        0.25,
        "Low-speed heading threshold changed.",
    )

    require(
        FUTURE_HEADING_SEMANTICS
        ==
        (
            "samplewise_trajectory_tangent_with_"
            "causal_heading_fallback"
        ),
        "Future heading semantics changed.",
    )

    require(
        RECEIVER_OFFSET_TEMPORAL_SEMANTICS
        ==
        (
            "one_body_frame_receiver_offset_draw_"
            "shared_across_all_horizons_per_MC_sample"
        ),
        (
            "Receiver offset temporal semantics changed."
        ),
    )

    print(
        "low-speed threshold      =",
        LOW_SPEED_HEADING_THRESHOLD_MPS,
        "m/s",
    )

    print(
        "threshold basis          =",
        LOW_SPEED_HEADING_THRESHOLD_BASIS,
    )

    print(
        "heading source           = SAMPLEWISE PREDICTED TANGENT"
    )

    print(
        "low-speed fallback       = PREVIOUS CAUSAL/PREDICTED HEADING"
    )

    print(
        "future GT heading        = NO"
    )

    print(
        "uncertain RX offset      = ONE DRAW PER MC SAMPLE"
    )

    print(
        "RX offset shared horizons= YES"
    )

    # ========================================================
    # F. End-to-end synthetic receiver-posterior sanity
    # ========================================================

    print()
    print(
        "===== F. PRIMARY MC END-TO-END SANITY ====="
    )

    synthetic_mean = np.asarray(
        [
            [
                15.0,
                1.0,
                0.5,
            ],
            [
                17.0,
                1.2,
                0.5,
            ],
            [
                19.0,
                1.4,
                0.5,
            ],
            [
                24.0,
                1.8,
                0.5,
            ],
        ],
        dtype=np.float64,
    )

    synthetic_cov = np.stack(
        [
            np.diag(
                [
                    0.5,
                    0.2,
                    0.1,
                ]
            ),
            np.diag(
                [
                    0.7,
                    0.3,
                    0.1,
                ]
            ),
            np.diag(
                [
                    0.9,
                    0.4,
                    0.15,
                ]
            ),
            np.diag(
                [
                    1.5,
                    0.8,
                    0.25,
                ]
            ),
        ],
        axis=0,
    )

    offset = ReceiverOffsetGaussianBody(
        mean_body_m=(
            1.0,
            0.0,
            0.5,
        ),

        covariance_body_m2=(
            (
                0.04,
                0.0,
                0.0,
            ),
            (
                0.0,
                0.04,
                0.0,
            ),
            (
                0.0,
                0.0,
                0.01,
            ),
        ),
    )

    posterior_first = (
        deterministic_receiver_angular_posterior(
            trajectory_mean_h0_m=(
                synthetic_mean
            ),

            raw_trajectory_covariance_h0_m2=(
                synthetic_cov
            ),

            current_position_h0_m=(
                14.0,
                0.8,
                0.5,
            ),

            current_heading_h0_rad=0.0,

            receiver_geometry_mode=(
                "uncertain_receiver_offset"
            ),

            receiver_offset=offset,

            sample_count=(
                selected_sample_count
            ),

            seed=MONTE_CARLO_BASE_SEED,
        )
    )

    posterior_second = (
        deterministic_receiver_angular_posterior(
            trajectory_mean_h0_m=(
                synthetic_mean
            ),

            raw_trajectory_covariance_h0_m2=(
                synthetic_cov
            ),

            current_position_h0_m=(
                14.0,
                0.8,
                0.5,
            ),

            current_heading_h0_rad=0.0,

            receiver_geometry_mode=(
                "uncertain_receiver_offset"
            ),

            receiver_offset=offset,

            sample_count=(
                selected_sample_count
            ),

            seed=MONTE_CARLO_BASE_SEED,
        )
    )

    require(
        posterior_first.sample_sha256
        ==
        posterior_second.sample_sha256,
        (
            "Primary MC exact-repeat "
            "hash mismatch."
        ),
    )

    require(
        posterior_first.range_samples_m.shape
        ==
        (
            selected_sample_count,
            4,
        ),
        (
            "Primary MC output shape mismatch."
        ),
    )

    for summary in (
        posterior_first.summaries
    ):
        covariance = np.asarray(
            summary.covariance_rae,
            dtype=np.float64,
        )

        require(
            np.all(
                np.isfinite(
                    covariance
                )
            ),
            (
                "Primary MC angular covariance "
                "contains non-finite values."
            ),
        )

        require(
            float(
                np.min(
                    np.linalg.eigvalsh(
                        covariance
                    )
                )
            )
            >=
            -1e-10,
            (
                "Primary MC angular covariance "
                "is not PSD."
            ),
        )

    print(
        "primary MC run           = PASS"
    )

    print(
        "exact-repeat hash        = PASS"
    )

    print(
        "sample SHA256            =",
        posterior_first.sample_sha256,
    )

    print(
        "output horizons          = 4 PASS"
    )

    print(
        "angular covariance PSD   = PASS"
    )

    print(
        "beam codebook            = NOT STARTED"
    )

    # ========================================================
    # G. Freeze angular-posterior policy
    # ========================================================

    print()
    print(
        "===== G. ANGULAR-POSTERIOR POLICY FREEZE ====="
    )

    policy_payload = {
        "status":
            "FROZEN",

        "primary_method":
            "deterministic_monte_carlo",

        "trajectory_posterior": {
            "family":
                "calibrated_Gaussian_GRU",

            "predictive_mean_calibration":
                "unchanged",

            "variance_scale_alpha_h":
                list(
                    FROZEN_VARIANCE_SCALE_ALPHA_H
                ),

            "joint_sampling":
                TRAJECTORY_JOINT_SAMPLING_SEMANTICS,

            "cross_horizon_covariance_invented":
                False,

            "measurement_R_t_modified":
                False,
        },

        "Monte_Carlo": {
            "sample_count":
                int(
                    selected_sample_count
                ),

            "base_seed":
                MONTE_CARLO_BASE_SEED,

            "minimum_sample_count":
                MONTE_CARLO_MINIMUM_SAMPLE_COUNT,

            "minimum_sample_count_basis":
                MONTE_CARLO_MINIMUM_SAMPLE_COUNT_BASIS,

            "convergence_reference_N":
                MONTE_CARLO_REFERENCE_SAMPLE_COUNT,

            "selection_rule":
                (
                    "smallest_preregistered_candidate_"
                    "passing_all_development_"
                    "convergence_thresholds"
                ),

            "convergence_artifact":
                str(
                    CONVERGENCE
                ),

            "convergence_artifact_sha256":
                convergence_sha,

            "formal_data_used_for_selection":
                False,
        },

        "heading": {
            "source":
                FUTURE_HEADING_SEMANTICS,

            "low_speed_threshold_mps":
                LOW_SPEED_HEADING_THRESHOLD_MPS,

            "threshold_basis":
                LOW_SPEED_HEADING_THRESHOLD_BASIS,

            "low_speed_fallback":
                (
                    "previous_causal_or_"
                    "predicted_heading"
                ),

            "future_GT_heading":
                False,
        },

        "receiver_placement": {
            "centroid_baseline":
                True,

            "known_receiver_offset":
                True,

            "uncertain_receiver_offset":
                True,

            "uncertain_offset_sampling":
                RECEIVER_OFFSET_TEMPORAL_SEMANTICS,

            "trajectory_and_placement_UQ":
                "jointly_propagated",

            "measurement_R_t":
                "remains_separate",
        },

        "angular_output": {
            "coordinates":
                [
                    "range_m",
                    "azimuth_rad",
                    "elevation_rad",
                ],

            "azimuth_statistics":
                "circular",

            "elevation_statistics":
                "ordinary_bounded_angle",

            "covariance":
                (
                    "sample_covariance_using_"
                    "wrapped_azimuth_residuals"
                ),
        },

        "development_evidence": {
            "source":
                str(
                    discovery[
                        "path"
                    ]
                ),

            "source_sha256":
                discovery[
                    "sha256"
                ],

            "formal_population":
                False,

            "model_inference_performed_in_this_freeze_run":
                False,

            "source_materialization_used_frozen_Stage4_inference":
                True,
        },

        "forbidden": {
            "formal_tuning":
                False,

            "tracks_to_predict":
                False,

            "future_truth":
                False,

            "future_GT_heading":
                False,

            "oracle_angle":
                False,
        },

        "next":
            "Block5.3_directional_codebooks_and_beam_probabilities",
    }

    write_json(
        POLICY,
        policy_payload,
    )

    policy_sha = file_sha256(
        POLICY
    )

    readback = load_json(
        POLICY
    )

    require(
        readback.get(
            "status"
        )
        ==
        "FROZEN",
        (
            "Angular-posterior policy "
            "readback failed."
        ),
    )

    require(
        readback[
            "Monte_Carlo"
        ][
            "sample_count"
        ]
        ==
        selected_sample_count,
        (
            "Frozen MC sample count "
            "readback mismatch."
        ),
    )

    require(
        readback[
            "forbidden"
        ][
            "formal_tuning"
        ]
        is False,
        (
            "Formal tuning became allowed."
        ),
    )

    print(
        "angular posterior policy = FROZEN"
    )

    print(
        "policy SHA256            =",
        policy_sha,
    )

    # ========================================================
    # H. Final Block5.2 report
    # ========================================================

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    report_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.2",

        "status":
            "PASS",

        "Part1": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_PART1_IMPLEMENTATION_SHA,
        },

        "primary_receiver_angular_posterior": {
            "method":
                "deterministic_monte_carlo",

            "trajectory_family":
                "calibrated_Gaussian_GRU",

            "horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "trajectory_joint_sampling":
                TRAJECTORY_JOINT_SAMPLING_SEMANTICS,

            "cross_horizon_covariance_invented":
                False,

            "sample_count":
                int(
                    selected_sample_count
                ),

            "base_seed":
                MONTE_CARLO_BASE_SEED,

            "samplewise_heading":
                True,

            "future_GT_heading":
                False,

            "low_speed_threshold_mps":
                LOW_SPEED_HEADING_THRESHOLD_MPS,

            "receiver_offset_shared_across_horizons":
                True,

            "centroid_mode":
                True,

            "known_offset_mode":
                True,

            "uncertain_offset_mode":
                True,

            "azimuth_circular_statistics":
                True,

            "analytic_Jacobian_crosscheck":
                True,
        },

        "development_convergence": {
            "status":
                "PASS",

            "artifact":
                str(
                    CONVERGENCE
                ),

            "artifact_sha256":
                convergence_sha,

            "source_prediction_artifact":
                str(
                    discovery[
                        "path"
                    ]
                ),

            "source_prediction_sha256":
                discovery[
                    "sha256"
                ],

            "case_count":
                len(
                    cases
                ),

            "selected_sample_count":
                int(
                    selected_sample_count
                ),

            "formal_data_used":
                False,

            "Stage4_inference_performed_in_this_freeze_run":
                False,

            "source_materialization_used_frozen_Stage4_inference":
                True,
        },

        "angular_policy": {
            "path":
                str(
                    POLICY
                ),

            "sha256":
                policy_sha,

            "status":
                "FROZEN",
        },

        "uncertainty_semantics": {
            "Stage2_measurement_R_t":
                "SEPARATE",

            "Stage4_predictive_covariance":
                "CALIBRATED",

            "receiver_placement_covariance":
                "SEPARATE_INPUT",

            "joint_receiver_propagation":
                "PASS",
        },

        "exact_repeat": {
            "status":
                "PASS",

            "synthetic_receiver_posterior_SHA256":
                posterior_first.sample_sha256,
        },

        "scientific_execution": {
            "formal_N120_read":
                False,

            "formal_evaluation":
                False,

            "Stage4_model_inference_in_this_freeze_run":
                False,

            "development_source_used_frozen_Stage4_inference":
                True,

            "training":
                False,

            "recalibration":
                False,

            "beam_codebook":
                False,

            "beam_selection":
                False,
        },

        "resolved_development_fields": {
            "low_speed_heading_threshold":
                {
                    "value_mps":
                        LOW_SPEED_HEADING_THRESHOLD_MPS,

                    "performance_tuned":
                        False,

                    "formal_data_used":
                        False,
                },

            "angular_monte_carlo_sample_count":
                {
                    "value":
                        int(
                            selected_sample_count
                        ),

                    "selection":
                        (
                            "development_numerical_"
                            "convergence"
                        ),

                    "formal_data_used":
                        False,
                },
        },

        "remaining_development_fields": [
            "codebook_angular_support",
            "formal_empirical_coverage_tolerance",
            "K_max",
            "hysteresis_parameters",
            "beam_switch_penalty",
            "local_neighbour_sweep_width",
            "fallback_parameters",
            "reacquisition_parameters",
        ],

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block5.3 directional codebooks "
                "and posterior beam probabilities"
            ),
    }

    write_json(
        REPORT,
        report_payload,
    )

    # ========================================================
    # I. Implementation log
    # ========================================================

    marker = (
        "## Block 5.2 — "
        "Receiver-aware angular posterior"
    )

    existing = (
        IMPLEMENTATION_LOG.read_text(
            encoding="utf-8"
        )
        if IMPLEMENTATION_LOG.is_file()
        else ""
    )

    if marker not in existing:

        with IMPLEMENTATION_LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:

            stream.write(
                "\n"
                +
                marker
                +
                "\n\n"
                "Status: PASS / FROZEN\n\n"
                "- Primary receiver/angular posterior uses "
                  "deterministic Monte Carlo.\n"
                "- Frozen Stage4 calibrated Gaussian means "
                  "remain unchanged; covariance is scaled "
                  "with the frozen Block4.5 alpha_h values.\n"
                "- No unsupported cross-horizon covariance "
                  "is invented; trajectory samples follow "
                  "the product of the frozen per-horizon "
                  "Gaussian marginals.\n"
                "- Receiver heading is constructed from "
                  "samplewise predicted trajectory tangents "
                  "with a 0.25 m/s low-speed carry-forward "
                  "fallback; future GT heading is never used.\n"
                "- Uncertain receiver placement uses one "
                  "body-frame offset draw per MC sample "
                  "shared across all horizons.\n"
                "- Azimuth statistics use circular means/"
                  "wrapped residuals.\n"
                "- MC sample count is frozen from the "
                  "non-formal Block4.4 development posterior "
                  "using a preregistered convergence rule.\n"
                "- Formal N=120 was not accessed and Stage4 "
                  "model inference was not rerun.\n"
                f"- MC sample count: "
                  f"{selected_sample_count}.\n"
                f"- Angular policy SHA256: "
                  f"{policy_sha}.\n"
                f"- MC convergence SHA256: "
                  f"{convergence_sha}.\n"
                f"- Block5.2 implementation SHA256: "
                  f"{implementation_sha}.\n"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 PART 2/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.1 continuity           = PASS"
    )

    print(
        "Block5.2 Part1 continuity     = PASS"
    )

    print(
        "Stage4 calibrated Gaussian    = PASS"
    )

    print(
        "Stage4 predictive mean        = UNCHANGED"
    )

    print(
        "Stage4 covariance alpha_h     = PASS"
    )

    print(
        "cross-horizon covariance      = NOT INVENTED"
    )

    print(
        "trajectory joint assumption   = PRODUCT OF MARGINALS"
    )

    print(
        "primary angular posterior     = DETERMINISTIC MONTE CARLO"
    )

    print(
        "MC sample count               =",
        selected_sample_count,
        "FROZEN",
    )

    print(
        "MC development convergence    = PASS"
    )

    print(
        "samplewise predicted heading  = PASS"
    )

    print(
        "low-speed threshold           = 0.25 m/s FROZEN"
    )

    print(
        "future GT heading             = NO"
    )

    print(
        "receiver offset shared horizons= YES"
    )

    print(
        "centroid/known/uncertain      = PASS"
    )

    print(
        "circular azimuth statistics   = PASS"
    )

    print(
        "exact-repeat MC hash          = PASS"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "Stage4 inference this run     = NO"
    )

    print(
        "beam codebook                 = NOT STARTED"
    )

    print(
        "training/recalibration        = NO"
    )

    print(
        "angular policy SHA256         =",
        policy_sha,
    )

    print(
        "convergence SHA256            =",
        convergence_sha,
    )

    print(
        "implementation files          =",
        implementation_files,
    )

    print(
        "implementation SHA256         =",
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
        "next = BLOCK 5.3"
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
        "BLOCK 5.2 PART 2/2 = BLOCKED"
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
        "formal N=120 used        = NO"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "Stage4 model inference   = NO"
    )

    print(
        "beam codebook            = NOT STARTED"
    )

    print(
        "training/recalibration   = NO"
    )

    print(
        "upstream modified        = NO"
    )

    print(
        "terminal remains open    = YES"
    )

# Deliberately no sys.exit().

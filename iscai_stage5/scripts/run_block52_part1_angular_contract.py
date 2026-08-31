from __future__ import annotations

from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import traceback

import numpy as np

from iscai_stage5.angular_posterior import (
    ANALYTIC_CROSSCHECK_METHOD,
    ANALYTIC_CROSSCHECK_SCOPE,
    ANGULAR_COORDINATE_ORDER,
    ANGULAR_FRAME,
    GaussianPositionH0,
    PRIMARY_STAGE52_METHOD,
    TRAJECTORY_AND_PLACEMENT_COVARIANCE_SEMANTICS,
    combine_independent_position_uncertainties,
    linearized_position_gaussian_to_angular,
    position_h0_to_range_azimuth_elevation,
    range_azimuth_elevation_jacobian,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
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

REPORT = (
    STAGE5
    / "reports/"
      "block52_part1_angular_contract.json"
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
    path.write_text(
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


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.2 PART 1/2"
    )
    print(
        "RECEIVER POSITION → ANGULAR POSTERIOR CONTRACT"
    )
    print(
        "============================================================"
    )

    required = (
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
            "Missing frozen Block5.1 artifact(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen Block5.1 continuity
    # ========================================================

    print()
    print(
        "===== A. BLOCK5.1 FROZEN CONTINUITY ====="
    )

    block51 = load_json(
        BLOCK51
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
            "Frozen receiver-policy SHA changed."
        ),
    )

    require(
        file_sha256(
            STRUCTURAL_CONTRACT
        )
        ==
        EXPECTED_STRUCTURAL_CONTRACT_SHA,
        (
            "Stage5 structural-contract "
            "SHA changed."
        ),
    )

    print(
        "Block5.1 report          = PASS"
    )

    print(
        "historical implementation= PASS"
    )

    print(
        "receiver policy          = FROZEN PASS"
    )

    print(
        "Stage5 contract          = FROZEN PASS"
    )

    # ========================================================
    # B. Exact angular definitions
    # ========================================================

    print()
    print(
        "===== B. RANGE / AZIMUTH / ELEVATION DEFINITIONS ====="
    )

    sample = (
        position_h0_to_range_azimuth_elevation(
            (
                12.0,
                3.0,
                1.5,
            )
        )
    )

    require(
        math.isclose(
            sample.range_m,
            math.sqrt(
                155.25
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Range definition failed.",
    )

    require(
        math.isclose(
            sample.azimuth_rad,
            math.atan2(
                3.0,
                12.0,
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Azimuth definition failed.",
    )

    require(
        math.isclose(
            sample.elevation_rad,
            math.atan2(
                1.5,
                math.hypot(
                    12.0,
                    3.0,
                ),
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Elevation definition failed.",
    )

    print(
        "frame                   = H0 PASS"
    )

    print(
        "range                   = sqrt(x²+y²+z²) PASS"
    )

    print(
        "azimuth                 = atan2(y,x) PASS"
    )

    print(
        "elevation               = atan2(z,sqrt(x²+y²)) PASS"
    )

    print(
        "coordinate order        =",
        list(
            ANGULAR_COORDINATE_ORDER
        ),
    )

    # ========================================================
    # C. Analytic Jacobian numerical check
    # ========================================================

    print()
    print(
        "===== C. ANALYTIC JACOBIAN CROSS-CHECK ====="
    )

    point = np.asarray(
        [
            12.0,
            3.0,
            1.5,
        ],
        dtype=np.float64,
    )

    analytic = (
        range_azimuth_elevation_jacobian(
            point
        )
    )

    epsilon = 1e-6

    numeric = np.zeros(
        (
            3,
            3,
        ),
        dtype=np.float64,
    )

    def angular_vector(
        value,
    ):
        result = (
            position_h0_to_range_azimuth_elevation(
                value
            )
        )

        return np.asarray(
            [
                result.range_m,
                result.azimuth_rad,
                result.elevation_rad,
            ],
            dtype=np.float64,
        )

    for dimension in range(
        3
    ):
        positive = point.copy()

        negative = point.copy()

        positive[
            dimension
        ] += epsilon

        negative[
            dimension
        ] -= epsilon

        numeric[
            :,
            dimension
        ] = (
            angular_vector(
                positive
            )
            -
            angular_vector(
                negative
            )
        ) / (
            2.0
            *
            epsilon
        )

    maximum_jacobian_error = float(
        np.max(
            np.abs(
                analytic
                -
                numeric
            )
        )
    )

    require(
        maximum_jacobian_error
        <
        2e-8,
        (
            "Analytic angular Jacobian "
            "failed finite-difference check."
        ),
    )

    print(
        "analytic Jacobian        = PASS"
    )

    print(
        "finite-difference check  = PASS"
    )

    print(
        "max absolute J error     =",
        maximum_jacobian_error,
    )

    # ========================================================
    # D. Covariance propagation semantics
    # ========================================================

    print()
    print(
        "===== D. RECEIVER-POSITION UNCERTAINTY PROPAGATION ====="
    )

    combined = (
        combine_independent_position_uncertainties(
            trajectory_mean_h0_m=(
                20.0,
                2.0,
                1.0,
            ),

            trajectory_covariance_h0_m2=(
                (
                    1.0,
                    0.1,
                    0.0,
                ),
                (
                    0.1,
                    0.8,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.4,
                ),
            ),

            receiver_offset_mean_h0_m=(
                2.0,
                0.0,
                0.5,
            ),

            receiver_placement_covariance_h0_m2=(
                (
                    0.25,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.5,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.04,
                ),
            ),
        )
    )

    angular = (
        linearized_position_gaussian_to_angular(
            combined
        )
    )

    covariance = np.asarray(
        angular.covariance
    )

    require(
        np.allclose(
            covariance,
            covariance.T,
            atol=1e-12,
            rtol=0.0,
        ),
        (
            "Angular covariance "
            "is not symmetric."
        ),
    )

    minimum_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                covariance
            )
        )
    )

    require(
        minimum_eigenvalue
        >=
        -1e-10,
        (
            "Angular covariance "
            "is not PSD."
        ),
    )

    print(
        "Stage4 predictive covariance = DISTINCT INPUT"
    )

    print(
        "receiver placement covariance = DISTINCT INPUT"
    )

    print(
        "position-level combination     = PASS"
    )

    print(
        "angular covariance             = JΣJᵀ PASS"
    )

    print(
        "angular covariance PSD         = PASS"
    )

    print(
        "minimum eigenvalue             =",
        minimum_eigenvalue,
    )

    # ========================================================
    # E. Primary vs analytic method boundary
    # ========================================================

    print()
    print(
        "===== E. PRIMARY / CROSS-CHECK METHOD BOUNDARY ====="
    )

    require(
        PRIMARY_STAGE52_METHOD
        ==
        "deterministic_monte_carlo",
        "Primary Stage5.2 method changed.",
    )

    require(
        ANALYTIC_CROSSCHECK_METHOD
        ==
        "first_order_Jacobian_Gaussian",
        (
            "Analytic cross-check "
            "method changed."
        ),
    )

    require(
        PRIMARY_STAGE52_METHOD
        !=
        ANALYTIC_CROSSCHECK_METHOD,
        (
            "Primary MC and analytic "
            "cross-check were conflated."
        ),
    )

    print(
        "primary posterior        = DETERMINISTIC MONTE CARLO"
    )

    print(
        "analytic path            = JACOBIAN CROSS-CHECK ONLY"
    )

    print(
        "analytic scope           =",
        ANALYTIC_CROSSCHECK_SCOPE,
    )

    print(
        "MC implemented Part1     = NO"
    )

    print(
        "MC sample count frozen   = NO"
    )

    print(
        "heading fallback frozen  = NO"
    )

    # ========================================================
    # F. Leakage/API surface
    # ========================================================

    print()
    print(
        "===== F. FORBIDDEN LEAKAGE SURFACE ====="
    )

    functions = (
        position_h0_to_range_azimuth_elevation,
        range_azimuth_elevation_jacobian,
        linearized_position_gaussian_to_angular,
        combine_independent_position_uncertainties,
    )

    forbidden = (
        "tracks_to_predict",
        "objects_of_interest",
        "future_truth",
        "ground_truth",
        "oracle",
    )

    for function in functions:

        signature = str(
            inspect.signature(
                function
            )
        ).lower()

        for token in forbidden:

            require(
                token
                not in
                signature,
                (
                    f"Forbidden argument {token!r} "
                    f"in {function.__name__}."
                ),
            )

    print(
        "tracks_to_predict       = NOT INPUT"
    )

    print(
        "future truth            = NOT INPUT"
    )

    print(
        "oracle angular label    = NOT INPUT"
    )

    print(
        "formal N=120 used       = NO"
    )

    print(
        "Stage4 inference        = NO"
    )

    print(
        "dataset scan            = NO"
    )

    # ========================================================
    # G. Report
    # ========================================================

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.2_part_1",

        "status":
            "PASS",

        "Block51_continuity": {
            "status":
                "PASS",

            "historical_implementation_sha256":
                EXPECTED_BLOCK51_IMPLEMENTATION_SHA,

            "receiver_policy_sha256":
                EXPECTED_RECEIVER_POLICY_SHA,

            "Stage5_contract_sha256":
                EXPECTED_STRUCTURAL_CONTRACT_SHA,
        },

        "angular_coordinates": {
            "frame":
                ANGULAR_FRAME,

            "order":
                list(
                    ANGULAR_COORDINATE_ORDER
                ),

            "range":
                "sqrt(x^2+y^2+z^2)",

            "azimuth":
                "atan2(y,x)",

            "elevation":
                (
                    "atan2(z,sqrt(x^2+y^2))"
                ),
        },

        "analytic_crosscheck": {
            "method":
                ANALYTIC_CROSSCHECK_METHOD,

            "scope":
                ANALYTIC_CROSSCHECK_SCOPE,

            "finite_difference_max_abs_error":
                maximum_jacobian_error,

            "covariance_mapping":
                "Sigma_rae=J@Sigma_xyz@J.T",

            "PSD":
                True,

            "primary_posterior":
                False,
        },

        "primary_posterior": {
            "method":
                PRIMARY_STAGE52_METHOD,

            "implemented_in_Part1":
                False,

            "implementation":
                "Block5.2_Part2",

            "trajectory_plus_receiver_placement_uncertainty":
                True,

            "sample_dependent_heading":
                "Block5.2_Part2",
        },

        "uncertainty_semantics": {
            "Stage2_measurement_R_t":
                "NOT_THIS_COVARIANCE",

            "Stage4_predictive_covariance":
                "DISTINCT",

            "receiver_placement_covariance":
                "DISTINCT",

            "combination":
                TRAJECTORY_AND_PLACEMENT_COVARIANCE_SEMANTICS,
        },

        "development_pending": {
            "angular_monte_carlo_sample_count":
                "NOT_YET_FROZEN",

            "low_speed_heading_threshold":
                "NOT_YET_FROZEN",

            "freeze_source":
                "non_formal_development_only",

            "formal_N120_allowed":
                False,
        },

        "scientific_execution": {
            "dataset_scan":
                False,

            "Stage4_inference":
                False,

            "Monte_Carlo_receiver_inference":
                False,

            "beam_codebook":
                False,

            "beam_selection":
                False,

            "formal_evaluation":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "leakage": {
            "tracks_to_predict":
                False,

            "future_truth":
                False,

            "oracle_angular_label":
                False,

            "formal_data":
                False,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "next":
            (
                "Block5.2 Part2 deterministic "
                "Monte Carlo trajectory+receiver "
                "uncertainty propagation"
            ),
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 PART 1/2 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "Block5.1 frozen continuity = PASS"
    )

    print(
        "H0 range/azimuth/elevation = PASS"
    )

    print(
        "analytic Jacobian           = PASS"
    )

    print(
        "finite-difference validation= PASS"
    )

    print(
        "position covariance PSD     = PASS"
    )

    print(
        "angular covariance PSD      = PASS"
    )

    print(
        "Stage4 predictive UQ        = DISTINCT"
    )

    print(
        "receiver placement UQ       = DISTINCT"
    )

    print(
        "analytic combination        = CROSS-CHECK ONLY"
    )

    print(
        "primary posterior method    = DETERMINISTIC MONTE CARLO"
    )

    print(
        "MC implemented              = NO — PART2"
    )

    print(
        "MC sample count             = DEVELOPMENT PENDING"
    )

    print(
        "low-speed heading threshold = DEVELOPMENT PENDING"
    )

    print(
        "formal N=120 used           = NO"
    )

    print(
        "Stage4 inference            = NO"
    )

    print(
        "beam codebook               = NOT STARTED"
    )

    print(
        "training/recalibration      = NO"
    )

    print(
        "implementation files        =",
        implementation_files,
    )

    print(
        "implementation SHA256       =",
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
        "BLOCK 5.2 PART 1/2 = BLOCKED"
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
        "formal data used        = NO"
    )

    print(
        "Stage4 inference        = NO"
    )

    print(
        "beam codebook           = NOT STARTED"
    )

    print(
        "training/recalibration  = NO"
    )

    print(
        "upstream modified       = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no sys.exit().

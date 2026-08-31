from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

SCENARIO_ID = "b85e1bd6cc8e74c0"

MOTION = (
    ROOT
    / "data"
    / "paired_womd_lidar_v1_3_0"
    / "validation"
    / "motion"
    / "paired-from-validation.tfrecord-00000-of-00150"
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

REPORT = (
    S6
    / "reports"
    / "block64_part2b1_real_mc_convergence.json"
)

COUNTS_NPZ = (
    S6
    / "artifacts"
    / "block64"
    / "block64_part2b1_real_mc_convergence_counts.npz"
)


EXPECTED = {
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
    value: dict[str, Any],
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
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


def canonical_digest(
    value: Any,
) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return sha256(
        payload
    ).hexdigest()


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


def tuple3(
    value,
):
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        array.shape == (
            3,
        ),
        (
            "Expected Vec3, got "
            f"{array.shape}."
        ),
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        (
            "Vec3 contains non-finite values."
        ),
    )

    return tuple(
        float(x)
        for x in array
    )


def tuple4x3(
    value,
):
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        array.shape
        ==
        (
            4,
            3,
        ),
        (
            "Expected [4,3], got "
            f"{array.shape}."
        ),
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        (
            "Mean contains non-finite values."
        ),
    )

    return tuple(
        tuple(
            float(x)
            for x in row
        )
        for row in array
    )


def tuple4x3x3(
    value,
):
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        array.shape
        ==
        (
            4,
            3,
            3,
        ),
        (
            "Expected [4,3,3], got "
            f"{array.shape}."
        ),
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        (
            "Covariance contains "
            "non-finite values."
        ),
    )

    return tuple(
        tuple(
            tuple(
                float(x)
                for x in row
            )
            for row in matrix
        )
        for matrix in array
    )


def convergence_metrics(
    counts,
    *,
    sample_count: int,
    reference_counts,
    reference_sample_count: int,
) -> dict[str, Any]:
    candidate = (
        np.asarray(
            counts,
            dtype=np.float64,
        )
        /
        float(
            sample_count
        )
    )

    reference = (
        np.asarray(
            reference_counts,
            dtype=np.float64,
        )
        /
        float(
            reference_sample_count
        )
    )

    error = np.abs(
        candidate
        -
        reference
    )

    union_support = (
        (
            np.asarray(
                counts
            )
            >
            0
        )
        |
        (
            np.asarray(
                reference_counts
            )
            >
            0
        )
    )

    intersection_support = (
        (
            np.asarray(
                counts
            )
            >
            0
        )
        &
        (
            np.asarray(
                reference_counts
            )
            >
            0
        )
    )

    union_count = int(
        np.count_nonzero(
            union_support
        )
    )

    intersection_count = int(
        np.count_nonzero(
            intersection_support
        )
    )

    if union_count:
        support_mae = float(
            np.mean(
                error[
                    union_support
                ]
            )
        )

        support_rmse = float(
            math.sqrt(
                float(
                    np.mean(
                        np.square(
                            error[
                                union_support
                            ]
                        )
                    )
                )
            )
        )

        support_iou = float(
            intersection_count
            /
            union_count
        )

    else:
        support_mae = 0.0
        support_rmse = 0.0
        support_iou = 1.0

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

    l1 = float(
        np.sum(
            error
        )
    )

    normalized_l1 = (
        float(
            l1
            /
            reference_mass
        )
        if reference_mass
        >
        0.0
        else 0.0
    )

    mass_relative_error = (
        float(
            abs(
                candidate_mass
                -
                reference_mass
            )
            /
            reference_mass
        )
        if reference_mass
        >
        0.0
        else 0.0
    )

    horizon_metrics = []

    for horizon_index in range(
        4
    ):
        h_error = error[
            horizon_index
        ]

        h_union = union_support[
            horizon_index
        ]

        h_reference = reference[
            horizon_index
        ]

        h_candidate = candidate[
            horizon_index
        ]

        h_reference_mass = float(
            np.sum(
                h_reference
            )
        )

        if np.any(
            h_union
        ):
            h_support_mae = float(
                np.mean(
                    h_error[
                        h_union
                    ]
                )
            )
        else:
            h_support_mae = 0.0

        horizon_metrics.append(
            {
                "horizon_index":
                    int(
                        horizon_index
                    ),

                "max_abs_probability_error":
                    float(
                        np.max(
                            h_error
                        )
                    ),

                "support_MAE":
                    h_support_mae,

                "normalized_L1":
                    (
                        float(
                            np.sum(
                                h_error
                            )
                            /
                            h_reference_mass
                        )
                        if h_reference_mass
                        >
                        0.0
                        else 0.0
                    ),

                "probability_mass_relative_error":
                    (
                        float(
                            abs(
                                np.sum(
                                    h_candidate
                                )
                                -
                                h_reference_mass
                            )
                            /
                            h_reference_mass
                        )
                        if h_reference_mass
                        >
                        0.0
                        else 0.0
                    ),
            }
        )

    return {
        "sample_count":
            int(
                sample_count
            ),

        "reference_sample_count":
            int(
                reference_sample_count
            ),

        "max_abs_probability_error":
            float(
                np.max(
                    error
                )
            ),

        "mean_abs_probability_error_all_cells":
            float(
                np.mean(
                    error
                )
            ),

        "support_MAE":
            support_mae,

        "support_RMSE":
            support_rmse,

        "positive_MC_support_IoU":
            support_iou,

        "normalized_L1":
            normalized_l1,

        "probability_mass_relative_error":
            mass_relative_error,

        "candidate_probability_mass":
            candidate_mass,

        "reference_probability_mass":
            reference_mass,

        "union_support_cells":
            union_count,

        "horizons":
            horizon_metrics,
    }


try:
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.4 PART2B/1 — REAL DEVELOPMENT MC CONVERGENCE"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # A. Immutable upstream seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN UPSTREAM SEAL ====="
    )

    for path, expected, label in (
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
            actual == expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:20s} = EXACT PASS"
        )

    # ========================================================
    # B. Exact frozen Stage5 MC study points
    # ========================================================

    print()
    print(
        "===== B. FROZEN STAGE5 MC STUDY POINTS ====="
    )

    from iscai_stage5.angular_monte_carlo import (
        MONTE_CARLO_BASE_SEED,
        MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS,
        MONTE_CARLO_MINIMUM_SAMPLE_COUNT,
        MONTE_CARLO_REFERENCE_SAMPLE_COUNT,
        TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
        resolve_samplewise_headings,
    )

    require(
        int(
            MONTE_CARLO_BASE_SEED
        )
        ==
        20260821,
        (
            "Stage5 base seed changed."
        ),
    )

    require(
        tuple(
            int(x)
            for x in
            MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS
        )
        ==
        (
            2048,
            4096,
            8192,
        ),
        (
            "Stage5 candidate counts changed."
        ),
    )

    require(
        int(
            MONTE_CARLO_MINIMUM_SAMPLE_COUNT
        )
        ==
        2048,
        (
            "Stage5 minimum count changed."
        ),
    )

    require(
        int(
            MONTE_CARLO_REFERENCE_SAMPLE_COUNT
        )
        ==
        32768,
        (
            "Stage5 reference count changed."
        ),
    )

    CANDIDATES = tuple(
        int(x)
        for x in
        MONTE_CARLO_CANDIDATE_SAMPLE_COUNTS
    )

    REFERENCE_N = int(
        MONTE_CARLO_REFERENCE_SAMPLE_COUNT
    )

    BASE_SEED = int(
        MONTE_CARLO_BASE_SEED
    )

    SNAPSHOT_COUNTS = (
        *CANDIDATES,
        REFERENCE_N,
    )

    print(
        "candidate N =",
        CANDIDATES,
    )

    print(
        "reference N =",
        REFERENCE_N,
    )

    print(
        "minimum N   =",
        MONTE_CARLO_MINIMUM_SAMPLE_COUNT,
    )

    print(
        "base seed   =",
        BASE_SEED,
    )

    print(
        "joint semantics =",
        TRAJECTORY_JOINT_SAMPLING_SEMANTICS,
    )

    print(
        "Stage6 seed policy frozen = NO"
    )

    print(
        "diagnostic common-random-number seed = "
        "FROZEN STAGE5 BASE SEED"
    )

    # ========================================================
    # C. Real causal binding
    # ========================================================

    print()
    print(
        "===== C. REAL DEVELOPMENT MATCHED ACTORS ====="
    )

    from iscai_stage0.womd_proto_io import (
        read_first_scenario,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    from iscai_stage6.adb.deterministic_predictive import (
        DeterministicSharedMeanPrediction,
        build_deterministic_predictive_plan,
    )

    from iscai_stage6.adb.probabilistic_full_box import (
        CalibratedGaussianFullBoxPrediction,
        absolute_mean_positions_H0_m,
        sample_calibrated_future_positions_H0_m,
    )

    from iscai_stage6.adb.geometry import (
        Box3D,
        project_box_to_headlamp,
    )

    from iscai_stage6.adb.probabilistic_occupancy import (
        OccupancyGrid,
        _points_inside_ccw_convex_polygon,
        _projected_theta_range_points,
        convex_hull_theta_range,
        rasterize_projected_full_box,
    )

    records = [
        json.loads(
            line
        )
        for line in
        POSTERIOR.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    require(
        len(
            records
        )
        ==
        27,
        (
            "Posterior record count changed."
        ),
    )

    deterministic_predictions = tuple(
        DeterministicSharedMeanPrediction(
            prediction_id=str(
                record[
                    "prediction_id"
                ]
            ),

            latest_position_H0_m=tuple3(
                record[
                    "latest_position_H0_m"
                ]
            ),

            mean_displacement_H0_m=tuple4x3(
                record[
                    "mean_displacement_H0_m"
                ]
            ),
        )
        for record in records
    )

    scenario = read_first_scenario(
        MOTION
    )

    require(
        str(
            scenario.scenario_id
        )
        ==
        SCENARIO_ID,
        (
            "Development pilot scenario changed."
        ),
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    actor_boxes = (
        build_causal_adb_actor_boxes(
            scenario=scenario,
            adapted=adapted,
        )
    )

    plan = (
        build_deterministic_predictive_plan(
            deterministic_predictions,
            actor_boxes,
        )
    )

    require(
        len(
            plan.matched_forecasts
        )
        ==
        6,
        (
            "Frozen matched count changed."
        ),
    )

    require(
        tuple(
            int(x)
            for x in
            plan.reactive_fallback_box_indices
        )
        ==
        (
            1,
            3,
            8,
        ),
        (
            "Frozen fallback indices changed."
        ),
    )

    require(
        len(
            plan.unmatched_prediction_indices
        )
        ==
        21,
        (
            "Frozen unmatched predictor "
            "count changed."
        ),
    )

    print(
        "scenario            =",
        SCENARIO_ID,
    )

    print(
        "posterior records   = 27"
    )

    print(
        "causal ADB boxes    =",
        len(
            actor_boxes
        ),
    )

    print(
        "matched actors      =",
        len(
            plan.matched_forecasts
        ),
    )

    print(
        "fallback boxes      =",
        tuple(
            int(x)
            for x in
            plan.reactive_fallback_box_indices
        ),
    )

    print(
        "unmatched predictors=",
        len(
            plan.unmatched_prediction_indices
        ),
    )

    # ========================================================
    # D. Diagnostic grid — not final Stage6 grid
    # ========================================================

    print()
    print(
        "===== D. DEVELOPMENT DIAGNOSTIC GRID ====="
    )

    theta_deg = np.linspace(
        -25.0,
        25.0,
        501,
        dtype=np.float64,
    )

    theta_rad = np.deg2rad(
        theta_deg
    )

    range_m = np.linspace(
        0.0,
        150.0,
        301,
        dtype=np.float64,
    )

    grid = OccupancyGrid(
        theta_centers_rad=tuple(
            float(x)
            for x in
            theta_rad
        ),

        range_centers_m=tuple(
            float(x)
            for x in
            range_m
        ),
    )

    require(
        grid.shape
        ==
        (
            501,
            301,
        ),
        (
            "Diagnostic grid shape changed."
        ),
    )

    print(
        "theta extent/resolution = "
        "-25..25 deg / 0.1 deg"
    )

    print(
        "range extent/resolution = "
        "0..150 m / 0.5 m"
    )

    print(
        "grid frame semantics    = Stage6 H0/headlamp theta"
    )

    print(
        "grid purpose            = DEVELOPMENT CONVERGENCE ONLY"
    )

    print(
        "final scientific grid frozen = NO"
    )

    theta_axis = np.asarray(
        grid.theta_centers_rad,
        dtype=np.float64,
    )

    range_axis = np.asarray(
        grid.range_centers_m,
        dtype=np.float64,
    )

    # ========================================================
    # E. Exact sparse equivalent of Part2A rasterizer
    # ========================================================

    print()
    print(
        "===== E. SPARSE RASTER ACCUMULATOR PARITY ====="
    )

    def sparse_region(
        projection,
    ):
        item = SimpleNamespace(
            projection=projection
        )

        points = (
            _projected_theta_range_points(
                item
            )
        )

        hull = convex_hull_theta_range(
            points
        )

        if hull.shape[
            0
        ] < 3:
            return None

        theta_min = float(
            np.min(
                hull[
                    :,
                    0
                ]
            )
        )

        theta_max = float(
            np.max(
                hull[
                    :,
                    0
                ]
            )
        )

        range_min = float(
            np.min(
                hull[
                    :,
                    1
                ]
            )
        )

        range_max = float(
            np.max(
                hull[
                    :,
                    1
                ]
            )
        )

        theta_lo = int(
            np.searchsorted(
                theta_axis,
                theta_min,
                side="left",
            )
        )

        theta_hi = int(
            np.searchsorted(
                theta_axis,
                theta_max,
                side="right",
            )
        )

        range_lo = int(
            np.searchsorted(
                range_axis,
                range_min,
                side="left",
            )
        )

        range_hi = int(
            np.searchsorted(
                range_axis,
                range_max,
                side="right",
            )
        )

        if (
            theta_lo
            >=
            theta_hi
            or
            range_lo
            >=
            range_hi
        ):
            return None

        theta_candidate = theta_axis[
            theta_lo:
            theta_hi
        ]

        range_candidate = range_axis[
            range_lo:
            range_hi
        ]

        theta_mesh, range_mesh = np.meshgrid(
            theta_candidate,
            range_candidate,
            indexing="ij",
        )

        query = np.column_stack(
            (
                theta_mesh.ravel(),
                range_mesh.ravel(),
            )
        )

        inside = (
            _points_inside_ccw_convex_polygon(
                query,
                hull,
            )
        ).reshape(
            theta_mesh.shape
        )

        return (
            theta_lo,
            theta_hi,
            range_lo,
            range_hi,
            inside,
        )


    def accumulate_projection(
        counts,
        *,
        horizon_index: int,
        projection,
    ):
        region = sparse_region(
            projection
        )

        if region is None:
            return

        (
            theta_lo,
            theta_hi,
            range_lo,
            range_hi,
            inside,
        ) = region

        counts[
            horizon_index,
            theta_lo:
            theta_hi,
            range_lo:
            range_hi,
        ] += inside


    parity_checked = False

    # ========================================================
    # F. Real MC convergence execution
    # ========================================================

    print()
    print(
        "===== F. REAL CALIBRATED P_OCC CONVERGENCE ====="
    )

    npz_payload = {
        "theta_centers_rad":
            theta_axis,

        "range_centers_m":
            range_axis,

        "sample_counts":
            np.asarray(
                SNAPSHOT_COUNTS,
                dtype=np.int64,
            ),

        "base_seed":
            np.asarray(
                [
                    BASE_SEED
                ],
                dtype=np.int64,
            ),
    }

    actor_results = []

    study_start = time.perf_counter()

    for matched_index, match in enumerate(
        plan.matched_forecasts
    ):
        actor_start = time.perf_counter()

        predictor_index = int(
            match.predictor_index
        )

        actor_box_index = int(
            match.actor_box_index
        )

        record = records[
            predictor_index
        ]

        actor_box = actor_boxes[
            actor_box_index
        ]

        require(
            str(
                record[
                    "prediction_id"
                ]
            )
            ==
            str(
                match.prediction_id
            ),
            (
                "Matched forecast/persisted "
                "prediction_id mismatch."
            ),
        )

        require(
            actor_box.future_state_used
            is False,
            (
                "Future state entered real "
                "stochastic actor box."
            ),
        )

        probabilistic_prediction = (
            CalibratedGaussianFullBoxPrediction(
                prediction_id=str(
                    record[
                        "prediction_id"
                    ]
                ),

                latest_position_H0_m=tuple3(
                    record[
                        "latest_position_H0_m"
                    ]
                ),

                mean_displacement_H0_m=tuple4x3(
                    record[
                        "mean_displacement_H0_m"
                    ]
                ),

                calibrated_predictive_covariance_H0_m2=(
                    tuple4x3x3(
                        record[
                            "calibrated_predictive_covariance_H0_m2"
                        ]
                    )
                ),
            )
        )

        absolute_mean = (
            absolute_mean_positions_H0_m(
                probabilistic_prediction
            )
        )

        reference_samples = (
            sample_calibrated_future_positions_H0_m(
                probabilistic_prediction,
                sample_count=REFERENCE_N,
                seed=BASE_SEED,
            )
        )

        require(
            reference_samples.shape
            ==
            (
                REFERENCE_N,
                4,
                3,
            ),
            (
                "Reference trajectory sample "
                "shape changed."
            ),
        )

        # Exact common-random-number prefix gate.
        prefix_sample_checks = {}

        for candidate_n in CANDIDATES:
            candidate_samples = (
                sample_calibrated_future_positions_H0_m(
                    probabilistic_prediction,
                    sample_count=candidate_n,
                    seed=BASE_SEED,
                )
            )

            equal = bool(
                np.array_equal(
                    candidate_samples,
                    reference_samples[
                        :candidate_n
                    ],
                )
            )

            require(
                equal,
                (
                    "Stage5 same-seed sampling "
                    f"is not an exact prefix at N={candidate_n}."
                ),
            )

            prefix_sample_checks[
                str(
                    candidate_n
                )
            ] = True

        (
            reference_headings,
            reference_fallback,
        ) = resolve_samplewise_headings(
            trajectory_samples_h0_m=(
                reference_samples
            ),

            current_position_h0_m=(
                probabilistic_prediction
                .latest_position_H0_m
            ),

            current_heading_h0_rad=float(
                actor_box.box.yaw_rad
            ),
        )

        require(
            reference_headings.shape
            ==
            (
                REFERENCE_N,
                4,
            ),
            (
                "Heading shape changed."
            ),
        )

        running_counts = np.zeros(
            (
                4,
                grid.shape[
                    0
                ],
                grid.shape[
                    1
                ],
            ),
            dtype=np.uint32,
        )

        snapshots = {}

        for sample_index in range(
            REFERENCE_N
        ):
            for horizon_index in range(
                4
            ):
                center = tuple(
                    float(x)
                    for x in
                    reference_samples[
                        sample_index,
                        horizon_index,
                        :
                    ]
                )

                box = Box3D(
                    center_xyz=center,

                    length_m=float(
                        actor_box.box.length_m
                    ),

                    width_m=float(
                        actor_box.box.width_m
                    ),

                    height_m=float(
                        actor_box.box.height_m
                    ),

                    yaw_rad=float(
                        reference_headings[
                            sample_index,
                            horizon_index
                        ]
                    ),
                )

                projection = (
                    project_box_to_headlamp(
                        box,

                        state_source=(
                            "predicted_sample"
                        ),

                        orientation_source=(
                            "predicted_tangent"
                        ),

                        controller_path=True,
                    )
                )

                if not parity_checked:
                    public = (
                        rasterize_projected_full_box(
                            SimpleNamespace(
                                projection=projection
                            ),
                            grid,
                        )
                    )

                    sparse = np.zeros(
                        grid.shape,
                        dtype=bool,
                    )

                    region = sparse_region(
                        projection
                    )

                    if region is not None:
                        (
                            theta_lo,
                            theta_hi,
                            range_lo,
                            range_hi,
                            inside,
                        ) = region

                        sparse[
                            theta_lo:
                            theta_hi,
                            range_lo:
                            range_hi,
                        ] = inside

                    require(
                        np.array_equal(
                            public,
                            sparse,
                        ),
                        (
                            "Sparse diagnostic accumulator "
                            "does not exactly match frozen "
                            "Part2A public rasterizer."
                        ),
                    )

                    parity_checked = True

                    print(
                        "public rasterizer parity = EXACT PASS"
                    )

                accumulate_projection(
                    running_counts,
                    horizon_index=horizon_index,
                    projection=projection,
                )

            completed = (
                sample_index
                +
                1
            )

            if completed in SNAPSHOT_COUNTS:
                snapshots[
                    completed
                ] = running_counts.copy()

                print(
                    f"actor {matched_index + 1}/6 "
                    f"| prediction={match.prediction_id} "
                    f"| snapshot N={completed} "
                    f"| occupied-count cells="
                    f"{int(np.count_nonzero(running_counts))}"
                )

        require(
            set(
                snapshots.keys()
            )
            ==
            set(
                SNAPSHOT_COUNTS
            ),
            (
                "Not all convergence snapshots "
                "were captured."
            ),
        )

        reference_counts = snapshots[
            REFERENCE_N
        ]

        candidate_metrics = []

        for candidate_n in CANDIDATES:
            candidate_metrics.append(
                convergence_metrics(
                    snapshots[
                        candidate_n
                    ],
                    sample_count=(
                        candidate_n
                    ),
                    reference_counts=(
                        reference_counts
                    ),
                    reference_sample_count=(
                        REFERENCE_N
                    ),
                )
            )

        actor_key = (
            f"actor_{matched_index:02d}"
        )

        for sample_count in SNAPSHOT_COUNTS:
            npz_payload[
                (
                    f"{actor_key}_"
                    f"counts_N{sample_count:05d}"
                )
            ] = snapshots[
                sample_count
            ]

        npz_payload[
            f"{actor_key}_predictor_index"
        ] = np.asarray(
            [
                predictor_index
            ],
            dtype=np.int64,
        )

        npz_payload[
            f"{actor_key}_actor_box_index"
        ] = np.asarray(
            [
                actor_box_index
            ],
            dtype=np.int64,
        )

        actor_elapsed = (
            time.perf_counter()
            -
            actor_start
        )

        actor_result = {
            "matched_index":
                int(
                    matched_index
                ),

            "prediction_id":
                str(
                    match.prediction_id
                ),

            "actor_object_type":
                str(
                    match.actor_object_type
                ),

            "predictor_index":
                predictor_index,

            "actor_box_index":
                actor_box_index,

            "association_anchor_distance_m":
                float(
                    match.association_anchor_distance_m
                ),

            "base_seed":
                BASE_SEED,

            "reference_N":
                REFERENCE_N,

            "candidate_N":
                list(
                    CANDIDATES
                ),

            "exact_sample_prefix_checks":
                prefix_sample_checks,

            "heading_fallback_fraction":
                float(
                    np.mean(
                        reference_fallback
                    )
                ),

            "absolute_mean_H0_m":
                absolute_mean.tolist(),

            "convergence":
                candidate_metrics,

            "runtime_s":
                float(
                    actor_elapsed
                ),
        }

        actor_results.append(
            actor_result
        )

        print(
            f"actor {matched_index + 1}/6 complete "
            f"| runtime_s={actor_elapsed:.3f}"
        )

    require(
        parity_checked,
        (
            "Public/sparse rasterizer parity "
            "was never checked."
        ),
    )

    study_elapsed = (
        time.perf_counter()
        -
        study_start
    )

    # ========================================================
    # G. Aggregate convergence curves
    # ========================================================

    print()
    print(
        "===== G. CONVERGENCE SUMMARY — NO ACCEPTANCE THRESHOLD ====="
    )

    aggregate = []

    for candidate_n in CANDIDATES:
        metrics = [
            next(
                item
                for item in
                actor[
                    "convergence"
                ]
                if item[
                    "sample_count"
                ]
                ==
                candidate_n
            )
            for actor in
            actor_results
        ]

        summary = {
            "sample_count":
                int(
                    candidate_n
                ),

            "worst_actor_max_abs_probability_error":
                float(
                    max(
                        item[
                            "max_abs_probability_error"
                        ]
                        for item in metrics
                    )
                ),

            "mean_actor_max_abs_probability_error":
                float(
                    np.mean(
                        [
                            item[
                                "max_abs_probability_error"
                            ]
                            for item in metrics
                        ]
                    )
                ),

            "mean_actor_support_MAE":
                float(
                    np.mean(
                        [
                            item[
                                "support_MAE"
                            ]
                            for item in metrics
                        ]
                    )
                ),

            "worst_actor_support_MAE":
                float(
                    max(
                        item[
                            "support_MAE"
                        ]
                        for item in metrics
                    )
                ),

            "mean_actor_normalized_L1":
                float(
                    np.mean(
                        [
                            item[
                                "normalized_L1"
                            ]
                            for item in metrics
                        ]
                    )
                ),

            "worst_actor_normalized_L1":
                float(
                    max(
                        item[
                            "normalized_L1"
                        ]
                        for item in metrics
                    )
                ),

            "mean_actor_probability_mass_relative_error":
                float(
                    np.mean(
                        [
                            item[
                                "probability_mass_relative_error"
                            ]
                            for item in metrics
                        ]
                    )
                ),

            "mean_positive_MC_support_IoU":
                float(
                    np.mean(
                        [
                            item[
                                "positive_MC_support_IoU"
                            ]
                            for item in metrics
                        ]
                    )
                ),
        }

        aggregate.append(
            summary
        )

        print()
        print(
            "N =",
            candidate_n,
        )

        print(
            "  worst max |ΔP| =",
            summary[
                "worst_actor_max_abs_probability_error"
            ],
        )

        print(
            "  mean support MAE =",
            summary[
                "mean_actor_support_MAE"
            ],
        )

        print(
            "  worst support MAE =",
            summary[
                "worst_actor_support_MAE"
            ],
        )

        print(
            "  mean normalized L1 =",
            summary[
                "mean_actor_normalized_L1"
            ],
        )

        print(
            "  mean probability-mass rel error =",
            summary[
                "mean_actor_probability_mass_relative_error"
            ],
        )

        print(
            "  mean positive-support IoU =",
            summary[
                "mean_positive_MC_support_IoU"
            ],
        )

    print()
    print(
        "monotonic convergence required = NO"
    )

    print(
        "automatic acceptance threshold = NONE"
    )

    print(
        "automatic N selection          = NO"
    )

    # ========================================================
    # H. Persist development evidence
    # ========================================================

    print()
    print(
        "===== H. DEVELOPMENT ARTIFACT MATERIALIZATION ====="
    )

    COUNTS_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        COUNTS_NPZ,
        **npz_payload,
    )

    counts_sha = sha256_file(
        COUNTS_NPZ
    )

    scientific_summary = {
        "scenario_id":
            SCENARIO_ID,

        "matched_actor_count":
            6,

        "fallback_actor_indices":
            [
                1,
                3,
                8,
            ],

        "candidate_sample_counts":
            list(
                CANDIDATES
            ),

        "reference_sample_count":
            REFERENCE_N,

        "diagnostic_seed":
            BASE_SEED,

        "common_random_number_prefix_design":
            True,

        "grid": {
            "theta_deg_min":
                -25.0,

            "theta_deg_max":
                25.0,

            "theta_step_deg":
                0.1,

            "range_m_min":
                0.0,

            "range_m_max":
                150.0,

            "range_step_m":
                0.5,

            "shape":
                [
                    501,
                    301,
                ],

            "development_diagnostic_only":
                True,

            "final_scientific_grid_frozen":
                False,
        },

        "aggregate_convergence":
            aggregate,

        "actors":
            actor_results,
    }

    scientific_digest = canonical_digest(
        scientific_summary
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.4-Part2B/1",

        "status":
            (
                "PASS_REAL_DEVELOPMENT_MC_"
                "CONVERGENCE_CURVES_GENERATED"
            ),

        "upstream": {
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

        "development_population": {
            "scenario_id":
                SCENARIO_ID,

            "formal_population_used":
                False,

            "posterior_records":
                27,

            "causal_ADB_boxes":
                9,

            "matched_actors":
                6,

            "reactive_fallback_actors":
                3,

            "unmatched_predictors":
                21,
        },

        "MC_study": {
            "candidate_sample_counts":
                list(
                    CANDIDATES
                ),

            "minimum_sample_count":
                2048,

            "reference_sample_count":
                REFERENCE_N,

            "diagnostic_base_seed":
                BASE_SEED,

            "candidate_streams_are_exact_reference_prefixes":
                True,

            "seed_source":
                "frozen_Stage5_MONTE_CARLO_BASE_SEED",

            "Stage6_final_seed_policy_frozen":
                False,

            "automatic_N_selection":
                False,

            "convergence_acceptance_threshold":
                None,
        },

        "occupancy": {
            "definition":
                "P_occ(theta,r,tau)",

            "full_sampled_boxes":
                True,

            "corners_per_box":
                8,

            "raster_semantics":
                (
                    "convex_hull_of_all_eight_"
                    "projected_physical_corners"
                ),

            "public_sparse_rasterizer_exact_parity":
                True,

            "gamma_applied":
                False,

            "multi_actor_binary_aggregation":
                False,

            "class_aware_policy":
                False,
        },

        "diagnostic_grid": {
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

            "purpose":
                "development_MC_convergence_only",

            "final_scientific_grid_frozen":
                False,
        },

        "aggregate_convergence":
            aggregate,

        "actors":
            actor_results,

        "counts_artifact": {
            "path":
                str(
                    COUNTS_NPZ
                ),

            "sha256":
                counts_sha,
        },

        "scientific_digest":
            scientific_digest,

        "runtime_s":
            float(
                study_elapsed
            ),

        "scientific_execution": {
            "real_development_posterior_sampled":
                True,

            "formal_evaluation":
                False,

            "model_forward":
                False,

            "training":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "MC_N_frozen":
                False,

            "seed_policy_frozen":
                False,

            "gamma_frozen":
                False,

            "final_grid_frozen":
                False,
        },

        "next":
            (
                "BLOCK6.4_PART2B2_DEVELOPMENT_"
                "PREREGISTRATION_AND_NUMERIC_FREEZE"
            ),
    }

    report_sha = write_json(
        REPORT,
        report_payload,
    )

    print(
        "counts NPZ =",
        COUNTS_NPZ,
    )

    print(
        "counts SHA256 =",
        counts_sha,
    )

    print(
        "scientific digest =",
        scientific_digest,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    print(
        "study runtime_s =",
        study_elapsed,
    )

    # ========================================================
    # I. Regression
    # ========================================================

    print()
    print(
        "===== I. FRESH STAGE6 REGRESSION ====="
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
    # J. Final
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "BLOCK 6.4 PART2B/1 — FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "STATUS = "
        "PASS_REAL_DEVELOPMENT_MC_CONVERGENCE_CURVES_GENERATED"
    )

    print(
        "real matched actors          = 6"
    )

    print(
        "candidate N                  =",
        CANDIDATES,
    )

    print(
        "reference N                  =",
        REFERENCE_N,
    )

    print(
        "diagnostic seed              =",
        BASE_SEED,
    )

    print(
        "candidate/reference streams  = EXACT PREFIXES"
    )

    print(
        "P_occ geometry               = FULL BOX / 8 CORNERS"
    )

    print(
        "sparse/public raster parity  = PASS"
    )

    print(
        "gamma applied                = NO"
    )

    print(
        "automatic N selection        = NO"
    )

    print(
        "convergence threshold frozen = NO"
    )

    print(
        "Stage6 seed policy frozen    = NO"
    )

    print(
        "final scientific grid frozen = NO"
    )

    print(
        "formal evaluation            = NO"
    )

    print(
        "model forward                = NO"
    )

    print(
        "training/recalibration       = NO / NO"
    )

    print(
        "fresh Stage6 regression      = PASS | 127"
    )

    print(
        "counts SHA256                =",
        counts_sha,
    )

    print(
        "report SHA256                =",
        report_sha,
    )

    print(
        "NEXT = BLOCK6.4 PART2B/2 "
        "DEVELOPMENT PREREGISTRATION + NUMERIC FREEZE"
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
        "BLOCK6.4 PART2B/1 — CONTROLLED BLOCK"
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
        "formal evaluation      = NO"
    )

    print(
        "model forward          = NO"
    )

    print(
        "training/recalibration = NO / NO"
    )

    print(
        "numeric freeze         = NO"
    )

    print(
        "upstream modification  = NO"
    )

    print(
        "terminal remains open  = YES"
    )

    print(
        "=" * 72
    )

# Deliberately no sys.exit().

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
    object_type_name,
)

from iscai_stage4.data.real_pipeline import (
    read_training_scenario,
)

from iscai_stage6.adb.class_aware_policy import (
    TYPE_CYCLIST,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
    angular_margin_cells,
    compute_class_aware_margin,
    dilate_mask_theta,
    threshold_occupancy_counts_strict_k,
)

from iscai_stage6.adb.geometry import (
    Box3D,
    project_box_to_headlamp,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PREREG = (
    S6 / "configs/"
    "block66_part3b_class_aware_policy_preregistration.json"
)

STAGE1_FREEZE = (
    S6 / "configs/"
    "stage6_development_stage1_class_gamma_freeze.json"
)

POCC_MANIFEST = (
    S6 / "artifacts/block66/"
    "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

PREDICTIVE_MATCHES = (
    S6 / "artifacts/block66/"
    "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

LEDGER = (
    S6 / "artifacts/block68/"
    "block68_original_reactive_decision_ledger.jsonl"
)

COHORT = (
    S6 / "artifacts/block66/"
    "block66_class_aware_development_cohort_120.jsonl"
)

REPLAY_CONTRACT = (
    S6 / "configs/"
    "block66_part3c_part2of2_deterministic_replay_contract.json"
)

MARGIN_BIND_REPORT = (
    S6 / "reports/"
    "block67_part1_margin_binding_gate.json"
)

CLASS_POLICY = (
    S6 / "src/iscai_stage6/adb/"
    "class_aware_policy.py"
)

TIMESTAMP_BINDING = (
    S6 / "configs/"
    "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
)

GEOMETRY_BIND = (
    S6 / "reports/"
    "block68_part2b_part2_geometry_identity_bind.json"
)

INPUT_BINDING = (
    S6 / "configs/"
    "stage6_development_stage2_margin_input_binding.json"
)

SCORES = (
    S6 / "artifacts/block68/stage2_margin/"
    "block68_stage2_class_margin_candidate_scores.jsonl"
)

FREEZE = (
    S6 / "configs/"
    "stage6_development_stage2_class_margin_freeze.json"
)

REPORT = (
    S6 / "reports/"
    "block68_stage2_class_margin_selection.json"
)


EXPECTED = {
    "prereg":
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    "stage1_freeze":
        "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87",

    "pocc_manifest":
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    "predictive_matches":
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    "ledger":
        "ac2b954571cdd0166bbcfa5bb8b2956785d1e132288dbe4af7fe2d55adeafcaa",

    "replay_contract":
        "ff1816222564b519514e1fad342128010bd19ac31fe0291cd4b28045e339b5c1",

    "margin_bind":
        "80c35d2aa5cd7fea1e54f7a6cb52b8bb0f28f36632ca66e6500f992dcff6d5cf",

    "class_policy":
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    "timestamp":
        "51a3b56a119467001a9f4617e1e19dc82c548c0a30c79fc9c4a9c1f5ef27bf56",
}

EXPECTED_TESTS = 224

N_MC = 8192

CLASS_ORDER = (
    TYPE_VEHICLE,
    TYPE_PEDESTRIAN,
    TYPE_CYCLIST,
)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

THETA_MIN_RAD = -0.4363323129985824
THETA_MAX_RAD = +0.4363323129985824
RANGE_MIN_M = 0.0
RANGE_MAX_M = 150.0
N_THETA = 501
N_RANGE = 301

GRID_SHAPE = (
    N_THETA,
    N_RANGE,
)

COUNTS_SHAPE = (
    4,
    N_THETA,
    N_RANGE,
)

GRID_CELL_COUNT = (
    N_THETA
    *
    N_RANGE
)

HARD_RESERVE_GIB = 250.0


# ============================================================
# Generic utilities
# ============================================================

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def exact(path, expected, label):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(path)

    require(
        actual == expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


def read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            if not line.strip():
                continue

            value = json.loads(line)

            require(
                isinstance(value, dict),
                (
                    f"Non-object JSONL row "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(value)

    return rows


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode("utf-8")


def write_once_json(path, value):
    payload = canonical_bytes(value)

    if path.exists():
        require(
            path.read_bytes() == payload,
            (
                "Existing frozen JSON differs: "
                f"{path}"
            ),
        )
        return

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)
    temporary.replace(path)


def write_once_jsonl(path, rows):
    payload = (
        "\n".join(
            json.dumps(
                row,
                sort_keys=True,
                allow_nan=False,
            )
            for row in rows
        )
        +
        "\n"
    ).encode("utf-8")

    if path.exists():
        require(
            path.read_bytes() == payload,
            (
                "Existing frozen JSONL differs: "
                f"{path}"
            ),
        )
        return

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(payload)
    temporary.replace(path)


def free_gib():
    return (
        shutil.disk_usage(S6).free
        /
        (1024 ** 3)
    )


def regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    import re

    for line in process.stdout.splitlines():
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-80:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


# ============================================================
# Frozen future-evaluator geometry
# ============================================================

def circular_abs_difference(theta, center):
    difference = (
        np.asarray(
            theta,
            dtype=np.float64,
        )
        -
        float(center)
    )

    return np.abs(
        np.arctan2(
            np.sin(difference),
            np.cos(difference),
        )
    )


def rasterize_projected_full_box(
    projected,
    *,
    theta_centers,
    range_centers,
    theta_half_step,
    range_half_step,
):
    angular = (
        circular_abs_difference(
            theta_centers,
            projected.theta_center_rad,
        )
        <=
        (
            0.5
            *
            float(
                projected.theta_span_rad
            )
            +
            float(
                theta_half_step
            )
        )
    )

    radial = (
        (
            range_centers
            +
            range_half_step
        )
        >=
        float(
            projected.ground_range_min_m
        )
    ) & (
        (
            range_centers
            -
            range_half_step
        )
        <=
        float(
            projected.ground_range_max_m
        )
    )

    return (
        angular[:, None]
        &
        radial[None, :]
    )


def resolve_future_index(
    timestamps,
    *,
    anchor_index,
    horizon_s,
):
    timestamps = np.asarray(
        timestamps,
        dtype=np.float64,
    )

    anchor_time = float(
        timestamps[
            anchor_index
        ]
    )

    candidates = np.arange(
        anchor_index + 1,
        len(timestamps),
        dtype=np.int64,
    )

    require(
        len(candidates) > 0,
        "No future timestamp candidates.",
    )

    target = (
        anchor_time
        +
        float(horizon_s)
    )

    errors = np.abs(
        timestamps[
            candidates
        ]
        -
        target
    )

    position = int(
        np.argmin(errors)
    )

    selected = int(
        candidates[
            position
        ]
    )

    minimum = float(
        np.min(errors)
    )

    tied = np.flatnonzero(
        np.isclose(
            errors,
            minimum,
            rtol=0.0,
            atol=1.0e-12,
        )
    )

    require(
        selected
        ==
        int(
            candidates[
                tied[0]
            ]
        ),
        "Timestamp tie-break changed.",
    )

    return selected


def future_box_H0(
    *,
    state,
    T_H0_from_W,
):
    center = (
        T_H0_from_W.apply_point(
            (
                float(state.center_x),
                float(state.center_y),
                float(state.center_z),
            )
        )
    )

    rotation = np.asarray(
        T_H0_from_W.rotation,
        dtype=np.float64,
    )

    require(
        rotation.shape == (3, 3),
        "Unexpected H0 rotation shape.",
    )

    forward_W = np.asarray(
        [
            math.cos(
                float(state.heading)
            ),
            math.sin(
                float(state.heading)
            ),
            0.0,
        ],
        dtype=np.float64,
    )

    forward_H0 = (
        rotation
        @
        forward_W
    )

    yaw_H0 = math.atan2(
        float(
            forward_H0[1]
        ),
        float(
            forward_H0[0]
        ),
    )

    return Box3D(
        center_xyz=(
            float(center[0]),
            float(center[1]),
            float(center[2]),
        ),
        length_m=float(
            state.length
        ),
        width_m=float(
            state.width
        ),
        height_m=float(
            state.height
        ),
        yaw_rad=float(
            yaw_H0
        ),
    )


# ============================================================
# Exact theta-dilation sufficient statistics
# ============================================================

def theta_distance_to_active(base_mask):
    """
    Exact symmetric theta-axis distance in integer cells.

    A cell is active after dilate_mask_theta(..., c)
    iff its distance to an already-active theta cell
    in the same range column is <= c.
    """

    base = np.asarray(
        base_mask,
        dtype=bool,
    )

    require(
        base.shape == GRID_SHAPE,
        (
            "Unexpected base-mask shape: "
            f"{base.shape}"
        ),
    )

    sentinel = (
        N_THETA
        +
        1
    )

    distance = np.full(
        GRID_SHAPE,
        sentinel,
        dtype=np.int16,
    )

    distance[
        base
    ] = 0

    # Forward pass.
    for theta_index in range(
        1,
        N_THETA,
    ):
        distance[
            theta_index
        ] = np.minimum(
            distance[
                theta_index
            ],
            distance[
                theta_index - 1
            ]
            +
            1,
        )

    # Backward pass.
    for theta_index in range(
        N_THETA - 2,
        -1,
        -1,
    ):
        distance[
            theta_index
        ] = np.minimum(
            distance[
                theta_index
            ],
            distance[
                theta_index + 1
            ]
            +
            1,
        )

    return distance


def metric_curves_by_dilation(
    base_mask,
    oracle_mask,
):
    distance = theta_distance_to_active(
        base_mask
    )

    oracle = np.asarray(
        oracle_mask,
        dtype=bool,
    )

    require(
        oracle.shape == GRID_SHAPE,
        "Oracle mask shape mismatch.",
    )

    reachable = (
        distance
        <=
        N_THETA
    )

    histogram_all = np.bincount(
        np.asarray(
            distance[
                reachable
            ],
            dtype=np.int64,
        ),
        minlength=N_THETA + 1,
    )

    histogram_oracle = np.bincount(
        np.asarray(
            distance[
                reachable
                &
                oracle
            ],
            dtype=np.int64,
        ),
        minlength=N_THETA + 1,
    )

    predicted = np.cumsum(
        histogram_all,
        dtype=np.int64,
    ).astype(
        np.float64
    )

    intersection = np.cumsum(
        histogram_oracle,
        dtype=np.int64,
    ).astype(
        np.float64
    )

    oracle_count = float(
        np.count_nonzero(
            oracle
        )
    )

    union = (
        predicted
        +
        oracle_count
        -
        intersection
    )

    iou = np.ones(
        N_THETA + 1,
        dtype=np.float64,
    )

    nonempty = (
        union > 0.0
    )

    iou[
        nonempty
    ] = (
        intersection[
            nonempty
        ]
        /
        union[
            nonempty
        ]
    )

    overmask = (
        predicted
        -
        intersection
    ) / float(
        GRID_CELL_COUNT
    )

    return (
        distance,
        iou,
        overmask,
    )


# ============================================================
# Candidate-space binding
# ============================================================

def candidate_spaces(prereg):
    parameterization = (
        prereg[
            "additional_class_margin_parameterization"
        ]
    )

    rules = (
        parameterization[
            "candidate_rules"
        ]
    )

    base_rule = rules[
        "base_margin_m"
    ]

    uncertainty_rule = rules[
        "uncertainty_multiplier"
    ]

    vehicle_rule = rules[
        "vehicle_closing_multiplier"
    ]

    cyclist_rule = rules[
        "cyclist_lateral_multiplier"
    ]

    require(
        base_rule[
            "representation"
        ]
        ==
        "j/4 metres",
        "Base-margin representation changed.",
    )

    require(
        uncertainty_rule[
            "representation"
        ]
        ==
        "j/2",
        "Uncertainty representation changed.",
    )

    require(
        vehicle_rule[
            "representation"
        ]
        ==
        "j/20",
        "Vehicle-closing representation changed.",
    )

    require(
        cyclist_rule[
            "representation"
        ]
        ==
        "j/4",
        "Cyclist-lateral representation changed.",
    )

    require(
        uncertainty_rule[
            "zero_permitted"
        ]
        is False,
        "Zero uncertainty response became permitted.",
    )

    require(
        vehicle_rule[
            "zero_permitted"
        ]
        is False,
        "Zero vehicle closing response became permitted.",
    )

    require(
        cyclist_rule[
            "zero_permitted"
        ]
        is False,
        "Zero cyclist lateral response became permitted.",
    )

    bases = [
        j / 4.0
        for j in range(
            int(
                base_rule[
                    "j_min"
                ]
            ),
            int(
                base_rule[
                    "j_max"
                ]
            )
            +
            1,
        )
    ]

    uncertainties = [
        j / 2.0
        for j in range(
            int(
                uncertainty_rule[
                    "j_min"
                ]
            ),
            int(
                uncertainty_rule[
                    "j_max"
                ]
            )
            +
            1,
        )
    ]

    vehicle_motion = [
        j / 20.0
        for j in range(
            int(
                vehicle_rule[
                    "j_min"
                ]
            ),
            int(
                vehicle_rule[
                    "j_max"
                ]
            )
            +
            1,
        )
    ]

    cyclist_motion = [
        j / 4.0
        for j in range(
            int(
                cyclist_rule[
                    "j_min"
                ]
            ),
            int(
                cyclist_rule[
                    "j_max"
                ]
            )
            +
            1,
        )
    ]

    candidates = {
        TYPE_VEHICLE: [
            {
                "base_margin_m":
                    float(base),

                "uncertainty_multiplier":
                    float(uncertainty),

                "motion_multiplier":
                    float(motion),
            }
            for base in bases
            for uncertainty in uncertainties
            for motion in vehicle_motion
        ],

        TYPE_PEDESTRIAN: [
            {
                "base_margin_m":
                    float(base),

                "uncertainty_multiplier":
                    float(uncertainty),

                "motion_multiplier":
                    0.0,
            }
            for base in bases
            for uncertainty in uncertainties
        ],

        TYPE_CYCLIST: [
            {
                "base_margin_m":
                    float(base),

                "uncertainty_multiplier":
                    float(uncertainty),

                "motion_multiplier":
                    float(motion),
            }
            for base in bases
            for uncertainty in uncertainties
            for motion in cyclist_motion
        ],
    }

    expected_counts = (
        parameterization[
            "candidate_counts"
        ]
    )

    for actor_class in CLASS_ORDER:
        require(
            len(
                candidates[
                    actor_class
                ]
            )
            ==
            int(
                expected_counts[
                    actor_class
                ]
            ),
            (
                "Candidate-count mismatch for "
                f"{actor_class}."
            ),
        )

    return candidates


# ============================================================
# Identity / posterior helpers
# ============================================================

def build_ledger_identity(rows):
    index = {}

    for scenario_row in rows:
        sid = str(
            scenario_row[
                "scenario_id"
            ]
        )

        cohort_index = int(
            scenario_row[
                "cohort_index"
            ]
        )

        for actor in scenario_row[
            "selected_adb_actors"
        ]:
            prediction_id = actor.get(
                "prediction_id"
            )

            if prediction_id is None:
                continue

            key = (
                sid,
                str(prediction_id),
                cohort_index,
                int(
                    actor[
                        "actor_box_index"
                    ]
                ),
            )

            index[
                key
            ] = actor

    return index


def build_posterior_index(rows):
    index = {}

    for row in rows:
        sid = str(
            row[
                "scenario_id"
            ]
        )

        prediction_id = str(
            row[
                "prediction_id"
            ]
        )

        key = (
            sid,
            prediction_id,
        )

        require(
            key not in index,
            (
                "Duplicate posterior key: "
                f"{key}"
            ),
        )

        for required_key in (
            "latest_position_H0_m",
            "mean_displacement_H0_m",
            "calibrated_predictive_covariance_H0_m2",
        ):
            require(
                required_key in row,
                (
                    "Posterior missing "
                    f"{required_key}: {key}"
                ),
            )

        index[
            key
        ] = row

    return index


def selected_stage1_k(stage1):
    selected = (
        stage1[
            "selected"
        ]
    )

    result = {
        actor_class:
            int(
                selected[
                    actor_class
                ][
                    "k"
                ]
            )
        for actor_class
        in CLASS_ORDER
    }

    require(
        result
        ==
        {
            TYPE_VEHICLE:
                980,

            TYPE_PEDESTRIAN:
                499,

            TYPE_CYCLIST:
                557,
        },
        (
            "Stage-1 selected k values "
            "do not match frozen result."
        ),
    )

    return result


# ============================================================
# Margin component + exact candidate mapping
# ============================================================

def margin_components(
    actor_class,
    *,
    posterior,
    horizon_index,
    horizon_s,
):
    motion = (
        0.0
        if actor_class
        ==
        TYPE_PEDESTRIAN
        else
        1.0
    )

    result = compute_class_aware_margin(
        actor_class,
        current_position_H0_m=
            posterior[
                "latest_position_H0_m"
            ],
        mean_displacement_H0_m=
            posterior[
                "mean_displacement_H0_m"
            ][
                horizon_index
            ],
        covariance_H0_m2=
            posterior[
                "calibrated_predictive_covariance_H0_m2"
            ][
                horizon_index
            ],
        horizon_s=
            float(
                horizon_s
            ),
        base_margin_m=
            0.0,
        uncertainty_multiplier=
            1.0,
        motion_multiplier=
            motion,
    )

    return result


def analytic_extra_margin(
    actor_class,
    candidate,
    *,
    component,
    horizon_s,
):
    base = float(
        candidate[
            "base_margin_m"
        ]
    )

    uncertainty = float(
        candidate[
            "uncertainty_multiplier"
        ]
    )

    motion = float(
        candidate[
            "motion_multiplier"
        ]
    )

    value = (
        base
        +
        uncertainty
        *
        float(
            component.lateral_predictive_sigma_m
        )
    )

    if actor_class == TYPE_VEHICLE:
        value += (
            motion
            *
            float(
                horizon_s
            )
            *
            float(
                component.closing_speed_mps
            )
        )

    elif actor_class == TYPE_CYCLIST:
        value += (
            motion
            *
            float(
                horizon_s
            )
            *
            float(
                component.cyclist_lateral_rate_mps
            )
        )

    return float(value)


def candidate_dilation_cells(
    actor_class,
    candidate,
    *,
    component,
    horizon_s,
    theta_step_rad,
):
    m_extra = analytic_extra_margin(
        actor_class,
        candidate,
        component=component,
        horizon_s=horizon_s,
    )

    cells = angular_margin_cells(
        # IMPORTANT:
        # frozen Block6.7 binding requires m_extra only.
        # The argument name is legacy/misleading.
        total_margin_m=
            m_extra,

        predicted_range_m=
            float(
                component.predicted_range_m
            ),

        theta_step_rad=
            float(
                theta_step_rad
            ),
    )

    require(
        int(cells) >= 0,
        "Negative dilation-cell count.",
    )

    return (
        m_extra,
        int(cells),
        min(
            int(cells),
            N_THETA,
        ),
    )


def prove_component_factorization(
    actor_class,
    candidates,
    *,
    posterior,
    horizon_index,
    horizon_s,
    component,
    theta_step_rad,
):
    """
    First scored pair per class:
    verify every preregistered numeric candidate against
    the exact frozen compute_class_aware_margin operator.
    """

    effective_cells = set()

    for candidate in candidates:

        (
            analytic_extra,
            analytic_cells,
            effective,
        ) = candidate_dilation_cells(
            actor_class,
            candidate,
            component=component,
            horizon_s=horizon_s,
            theta_step_rad=theta_step_rad,
        )

        exact_margin = compute_class_aware_margin(
            actor_class,
            current_position_H0_m=
                posterior[
                    "latest_position_H0_m"
                ],
            mean_displacement_H0_m=
                posterior[
                    "mean_displacement_H0_m"
                ][
                    horizon_index
                ],
            covariance_H0_m2=
                posterior[
                    "calibrated_predictive_covariance_H0_m2"
                ][
                    horizon_index
                ],
            horizon_s=
                float(
                    horizon_s
                ),
            base_margin_m=
                float(
                    candidate[
                        "base_margin_m"
                    ]
                ),
            uncertainty_multiplier=
                float(
                    candidate[
                        "uncertainty_multiplier"
                    ]
                ),
            motion_multiplier=
                float(
                    candidate[
                        "motion_multiplier"
                    ]
                ),
        )

        require(
            math.isclose(
                analytic_extra,
                float(
                    exact_margin.additional_class_margin_m
                ),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            (
                "Analytic m_extra factorization "
                f"differs for {actor_class}."
            ),
        )

        exact_cells = angular_margin_cells(
            total_margin_m=
                float(
                    exact_margin.additional_class_margin_m
                ),
            predicted_range_m=
                float(
                    exact_margin.predicted_range_m
                ),
            theta_step_rad=
                float(
                    theta_step_rad
                ),
        )

        require(
            int(exact_cells)
            ==
            int(analytic_cells),
            (
                "Analytic margin-cell route differs "
                f"for {actor_class}."
            ),
        )

        effective_cells.add(
            effective
        )

    return sorted(
        effective_cells
    )


# ============================================================
# Selection
# ============================================================

def select_candidate(
    actor_class,
    candidates,
    iou_means,
    overmask_means,
):
    require(
        len(candidates)
        ==
        len(iou_means)
        ==
        len(overmask_means),
        "Candidate-score length mismatch.",
    )

    primary_max = float(
        np.max(
            iou_means
        )
    )

    primary_winners = [
        index
        for index, value
        in enumerate(
            iou_means
        )
        if float(value)
        ==
        primary_max
    ]

    minimum_overmask = min(
        float(
            overmask_means[
                index
            ]
        )
        for index
        in primary_winners
    )

    secondary_winners = [
        index
        for index
        in primary_winners
        if float(
            overmask_means[
                index
            ]
        )
        ==
        minimum_overmask
    ]

    def canonical_key(index):
        candidate = candidates[
            index
        ]

        return (
            float(
                candidate[
                    "base_margin_m"
                ]
            ),
            float(
                candidate[
                    "uncertainty_multiplier"
                ]
            ),
            float(
                candidate[
                    "motion_multiplier"
                ]
            ),
            tuple(
                sorted(
                    (
                        key,
                        float(value),
                    )
                    for key, value
                    in candidate.items()
                )
            ),
        )

    chosen_index = min(
        secondary_winners,
        key=canonical_key,
    )

    candidate = dict(
        candidates[
            chosen_index
        ]
    )

    return {
        **candidate,

        "candidate_index":
            int(
                chosen_index
            ),

        "macro_actor_horizon_IoU":
            float(
                iou_means[
                    chosen_index
                ]
            ),

        "macro_actor_horizon_over_masking_area":
            float(
                overmask_means[
                    chosen_index
                ]
            ),

        "primary_IoU_maximizer_count":
            int(
                len(
                    primary_winners
                )
            ),

        "secondary_overmask_winner_count":
            int(
                len(
                    secondary_winners
                )
            ),

        "tie_break_order": [
            "lower over-masking area",
            "smaller base margin",
            "smaller uncertainty multiplier",
            (
                "smaller closing/lateral multiplier"
                if actor_class
                !=
                TYPE_PEDESTRIAN
                else
                "motion multiplier not applicable"
            ),
            "canonical numeric tuple",
        ],
    }


# ============================================================
# Idempotence
# ============================================================

def idempotent_readback():
    if not REPORT.is_file():
        return False

    report = read_json(
        REPORT
    )

    if (
        report.get(
            "status"
        )
        !=
        "PASS_STAGE2_CLASS_MARGIN_SELECTED_FROZEN"
    ):
        return False

    for key, path in (
        ("input_binding", INPUT_BINDING),
        ("scores", SCORES),
        ("freeze", FREEZE),
    ):
        require(
            path.is_file(),
            f"Missing frozen Stage-2 output: {path}",
        )

        require(
            file_sha(path)
            ==
            report[
                "outputs"
            ][
                key
            ][
                "sha256"
            ],
            (
                "Frozen Stage-2 artifact changed: "
                f"{key}"
            ),
        )

    print(
        "existing Stage-2 margin freeze = EXACT PASS"
    )

    print(
        "STATUS = PASS_STAGE2_CLASS_MARGIN_SELECTED_FROZEN"
    )

    return True


# ============================================================
# Main
# ============================================================

def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "DEVELOPMENT STAGE-2 CLASS-MARGIN SELECTION"
    )
    print(
        "STAGE-1 k/GAMMA IMMUTABLE"
    )
    print(
        "M_EXTRA DILATION ONLY"
    )
    print(
        "NO FLOORS / NO TEMPORAL / NO RATE / NO FORMAL"
    )
    print(
        "============================================================"
    )

    if idempotent_readback():
        return

    # --------------------------------------------------------
    # A. Exact frozen prerequisites
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT PRE-STAGE2 BOUNDARY ====="
    )

    seals = {}

    for key, path in (
        ("prereg", PREREG),
        ("stage1_freeze", STAGE1_FREEZE),
        ("pocc_manifest", POCC_MANIFEST),
        ("predictive_matches", PREDICTIVE_MATCHES),
        ("ledger", LEDGER),
        ("replay_contract", REPLAY_CONTRACT),
        ("margin_bind", MARGIN_BIND_REPORT),
        ("class_policy", CLASS_POLICY),
        ("timestamp", TIMESTAMP_BINDING),
    ):

        seals[key] = exact(
            path,
            EXPECTED[key],
            key,
        )

        print(
            f"{key:20s} = EXACT PASS"
        )

    stage1 = read_json(
        STAGE1_FREEZE
    )

    require(
        stage1.get(
            "status"
        )
        ==
        "FROZEN_STAGE1_CLASS_GAMMA_BEFORE_STAGE2_MARGIN_SELECTION",
        (
            "Stage-1 freeze status mismatch."
        ),
    )

    k_by_class = selected_stage1_k(
        stage1
    )

    margin_bind = read_json(
        MARGIN_BIND_REPORT
    )

    require(
        margin_bind.get(
            "status"
        )
        ==
        "PASS_RUNTIME_BINDING_FROZEN",
        (
            "Margin binding is not authoritative PASS."
        ),
    )

    prereg = read_json(
        PREREG
    )

    protocol = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_2_class_margin"
        ]
    )

    require(
        protocol[
            "gamma"
        ]
        ==
        "frozen Stage-1 development selection",
        (
            "Stage-2 gamma freeze contract changed."
        ),
    )

    require(
        protocol[
            "classwise_primary_objective"
        ]
        ==
        "maximize macro actor-horizon IoU after dilation",
        (
            "Stage-2 primary objective changed."
        ),
    )

    require(
        protocol[
            "mandatory_nonzero_uncertainty_response"
        ]
        is True,
        (
            "Mandatory uncertainty response changed."
        ),
    )

    require(
        protocol[
            "vehicle_closing_response_required"
        ]
        is True,
        (
            "Vehicle closing response requirement changed."
        ),
    )

    require(
        protocol[
            "cyclist_lateral_response_required"
        ]
        is True,
        (
            "Cyclist lateral response requirement changed."
        ),
    )

    require(
        protocol[
            "tie_break_order"
        ]
        ==
        [
            "lower over-masking area",
            "smaller base margin",
            "smaller uncertainty multiplier",
            "smaller closing/lateral multiplier",
            "canonical numeric tuple",
        ],
        (
            "Stage-2 tie-break order changed."
        ),
    )

    candidates = candidate_spaces(
        prereg
    )

    print(
        "vehicle k               =",
        k_by_class[
            TYPE_VEHICLE
        ],
        "(FROZEN)"
    )

    print(
        "pedestrian k            =",
        k_by_class[
            TYPE_PEDESTRIAN
        ],
        "(FROZEN)"
    )

    print(
        "cyclist k               =",
        k_by_class[
            TYPE_CYCLIST
        ],
        "(FROZEN)"
    )

    print(
        "vehicle candidates      =",
        len(
            candidates[
                TYPE_VEHICLE
            ]
        ),
    )

    print(
        "pedestrian candidates   =",
        len(
            candidates[
                TYPE_PEDESTRIAN
            ]
        ),
    )

    print(
        "cyclist candidates      =",
        len(
            candidates[
                TYPE_CYCLIST
            ]
        ),
    )

    print(
        "Stage2 dilation semantic= M_EXTRA ONLY"
    )

    print(
        "Part-A generic 1m       = NOT RE-APPLIED"
    )

    # --------------------------------------------------------
    # B. Posterior source bind BEFORE Stage-2 outcomes
    # --------------------------------------------------------

    print()
    print(
        "===== B. STAGE2 POSTERIOR INPUT BINDING ====="
    )

    replay = read_json(
        REPLAY_CONTRACT
    )

    posterior_path = Path(
        replay[
            "upstream"
        ][
            "posterior"
        ][
            "path"
        ]
    )

    expected_posterior_path = (
        S6
        / "artifacts/block66/"
          "block66_development_identity_safe_gaussian_posterior.jsonl"
    )

    require(
        posterior_path
        ==
        expected_posterior_path,
        (
            "Unexpected development posterior path: "
            f"{posterior_path}"
        ),
    )

    require(
        posterior_path.is_file(),
        "Development posterior file missing.",
    )

    posterior_sha = file_sha(
        posterior_path
    )

    input_binding = {
        "stage":
            6,

        "block":
            "6.8_development_stage2_margin_input_binding",

        "status":
            "FROZEN_BEFORE_STAGE2_MARGIN_OBJECTIVE",

        "posterior": {
            "path":
                str(
                    posterior_path
                ),

            "sha256":
                posterior_sha,

            "source":
                (
                    "exact path bound by frozen "
                    "Part3C deterministic replay contract"
                ),
        },

        "stage1_gamma_freeze": {
            "path":
                str(
                    STAGE1_FREEZE
                ),

            "sha256":
                EXPECTED[
                    "stage1_freeze"
                ],

            "selected_k":
                k_by_class,
        },

        "candidate_counts": {
            actor_class:
                len(
                    candidates[
                        actor_class
                    ]
                )
            for actor_class
            in CLASS_ORDER
        },

        "dilation_semantics":
            "additional_class_margin_m / m_extra ONLY",

        "PartA_generic_margin_reapplied":
            False,

        "scientific_boundary": {
            "stage2_outcomes_seen_before_binding":
                False,

            "floor_selection":
                False,

            "temporal_rate_selection":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },
    }

    write_once_json(
        INPUT_BINDING,
        input_binding,
    )

    input_binding_sha = file_sha(
        INPUT_BINDING
    )

    print(
        "posterior SHA256        =",
        posterior_sha,
    )

    print(
        "Stage2 input binding SHA=",
        input_binding_sha,
    )

    tests_pre = regression()

    print(
        "Stage6 pre regression   =",
        f"{tests_pre} / {tests_pre} PASS",
    )

    require(
        free_gib()
        >=
        HARD_RESERVE_GIB,
        "Disk reserve gate failed.",
    )

    # --------------------------------------------------------
    # C. Load frozen development inputs
    # --------------------------------------------------------

    print()
    print(
        "===== C. DEVELOPMENT INPUT JOIN ====="
    )

    manifest_rows = read_jsonl(
        POCC_MANIFEST
    )

    match_rows = read_jsonl(
        PREDICTIVE_MATCHES
    )

    ledger_rows = read_jsonl(
        LEDGER
    )

    cohort_rows = read_jsonl(
        COHORT
    )

    posterior_rows = read_jsonl(
        posterior_path
    )

    posterior_index = build_posterior_index(
        posterior_rows
    )

    match_index = {}

    for row in match_rows:
        key = (
            str(
                row[
                    "scenario_id"
                ]
            ),
            str(
                row[
                    "prediction_id"
                ]
            ),
            int(
                row[
                    "cohort_index"
                ]
            ),
            int(
                row[
                    "actor_box_index"
                ]
            ),
        )

        require(
            key not in match_index,
            (
                "Duplicate predictive match: "
                f"{key}"
            ),
        )

        match_index[
            key
        ] = row

    identity_index = build_ledger_identity(
        ledger_rows
    )

    cohort_by_sid = {
        str(
            row[
                "scenario_id"
            ]
        ):
        row
        for row in cohort_rows
    }

    require(
        len(
            cohort_by_sid
        )
        ==
        120,
        "Development cohort != 120.",
    )

    by_scenario = defaultdict(
        list
    )

    for row in manifest_rows:
        by_scenario[
            str(
                row[
                    "scenario_id"
                ]
            )
        ].append(row)

    class_counts = Counter(
        str(
            row[
                "object_type"
            ]
        )
        for row in manifest_rows
    )

    require(
        dict(
            class_counts
        )
        ==
        {
            TYPE_VEHICLE:
                719,

            TYPE_PEDESTRIAN:
                110,

            TYPE_CYCLIST:
                48,
        },
        (
            "Development class counts changed: "
            f"{dict(class_counts)}"
        ),
    )

    for row in manifest_rows:
        key = (
            str(
                row[
                    "scenario_id"
                ]
            ),
            str(
                row[
                    "prediction_id"
                ]
            ),
            int(
                row[
                    "cohort_index"
                ]
            ),
            int(
                row[
                    "actor_box_index"
                ]
            ),
        )

        require(
            key in match_index,
            f"Missing predictive match: {key}",
        )

        require(
            key in identity_index,
            f"Missing identity join: {key}",
        )

        posterior_key = (
            key[0],
            key[1],
        )

        require(
            posterior_key
            in
            posterior_index,
            (
                "Missing posterior: "
                f"{posterior_key}"
            ),
        )

    print(
        "P_occ actors            =",
        len(
            manifest_rows
        ),
    )

    print(
        "posterior join          = EXACT PASS"
    )

    print(
        "predictive-match join   = EXACT PASS"
    )

    print(
        "track-identity join     = EXACT PASS"
    )

    print(
        "future used as selector = NO"
    )

    # --------------------------------------------------------
    # D. Stage-2 scoring
    # --------------------------------------------------------

    print()
    print(
        "===== D. STAGE-2 CLASS-MARGIN SWEEP ====="
    )

    theta_centers = np.linspace(
        THETA_MIN_RAD,
        THETA_MAX_RAD,
        N_THETA,
        dtype=np.float64,
    )

    range_centers = np.linspace(
        RANGE_MIN_M,
        RANGE_MAX_M,
        N_RANGE,
        dtype=np.float64,
    )

    theta_step_rad = float(
        theta_centers[1]
        -
        theta_centers[0]
    )

    theta_half_step = (
        0.5
        *
        theta_step_rad
    )

    range_half_step = (
        0.5
        *
        float(
            range_centers[1]
            -
            range_centers[0]
        )
    )

    iou_sums = {
        actor_class:
            np.zeros(
                len(
                    candidates[
                        actor_class
                    ]
                ),
                dtype=np.float64,
            )
        for actor_class
        in CLASS_ORDER
    }

    overmask_sums = {
        actor_class:
            np.zeros(
                len(
                    candidates[
                        actor_class
                    ]
                ),
                dtype=np.float64,
            )
        for actor_class
        in CLASS_ORDER
    }

    scored_pairs = Counter()
    invalid_future_pairs = Counter()

    parity_done = {
        actor_class:
            False
        for actor_class
        in CLASS_ORDER
    }

    processed_actors = 0

    ordered_sids = sorted(
        by_scenario,
        key=lambda sid:
            int(
                by_scenario[
                    sid
                ][
                    0
                ][
                    "cohort_index"
                ]
            ),
    )

    for scenario_rank, sid in enumerate(
        ordered_sids,
        start=1,
    ):
        scenario = read_training_scenario(
            cohort_by_sid[
                sid
            ]
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            sid,
            (
                "Loaded scenario ID mismatch: "
                f"{sid}"
            ),
        )

        anchor_index = int(
            scenario.current_time_index
        )

        adapted = adapt_causal_womd_scenario(
            scenario
        )

        T_H0_from_W = (
            adapted
            .frames
            .T_H0_from_W
        )

        future_indices = {
            horizon:
                resolve_future_index(
                    scenario.timestamps_seconds,
                    anchor_index=anchor_index,
                    horizon_s=horizon,
                )
            for horizon in HORIZONS_S
        }

        for manifest in by_scenario[
            sid
        ]:
            key = (
                sid,
                str(
                    manifest[
                        "prediction_id"
                    ]
                ),
                int(
                    manifest[
                        "cohort_index"
                    ]
                ),
                int(
                    manifest[
                        "actor_box_index"
                    ]
                ),
            )

            actor_class = str(
                manifest[
                    "object_type"
                ]
            )

            identity = identity_index[
                key
            ]

            track_index = int(
                identity[
                    "track_index"
                ]
            )

            track = scenario.tracks[
                track_index
            ]

            require(
                str(
                    track.id
                )
                ==
                str(
                    identity[
                        "track_id"
                    ]
                ),
                (
                    "Track identity mismatch: "
                    f"{key}"
                ),
            )

            require(
                object_type_name(
                    track
                )
                ==
                actor_class,
                (
                    "Track class mismatch: "
                    f"{key}"
                ),
            )

            posterior = posterior_index[
                (
                    sid,
                    key[1],
                )
            ]

            counts_path = Path(
                manifest[
                    "occupancy_counts_path"
                ]
            )

            require(
                file_sha(
                    counts_path
                )
                ==
                str(
                    manifest[
                        "occupancy_counts_sha256"
                    ]
                ),
                (
                    "P_occ count SHA mismatch: "
                    f"{counts_path}"
                ),
            )

            counts = np.load(
                counts_path,
                allow_pickle=False,
                mmap_mode="r",
            )

            require(
                counts.shape
                ==
                COUNTS_SHAPE,
                (
                    "P_occ shape mismatch: "
                    f"{counts.shape}"
                ),
            )

            fixed_binary = (
                threshold_occupancy_counts_strict_k(
                    counts,
                    k=
                        k_by_class[
                            actor_class
                        ],
                    sample_count=
                        N_MC,
                )
            )

            require(
                fixed_binary.shape
                ==
                COUNTS_SHAPE,
                (
                    "Thresholded P_occ shape mismatch."
                ),
            )

            for h, horizon in enumerate(
                HORIZONS_S
            ):
                future_index = future_indices[
                    horizon
                ]

                state = track.states[
                    future_index
                ]

                if not bool(
                    state.valid
                ):
                    invalid_future_pairs[
                        actor_class
                    ] += 1
                    continue

                box_H0 = future_box_H0(
                    state=state,
                    T_H0_from_W=T_H0_from_W,
                )

                projected = project_box_to_headlamp(
                    box_H0,
                    state_source=
                        "WOMD_future_GT_evaluator_only",
                    orientation_source=
                        "oracle_evaluator_only",
                    controller_path=False,
                )

                oracle = rasterize_projected_full_box(
                    projected,
                    theta_centers=theta_centers,
                    range_centers=range_centers,
                    theta_half_step=theta_half_step,
                    range_half_step=range_half_step,
                )

                base_mask = np.asarray(
                    fixed_binary[
                        h
                    ],
                    dtype=bool,
                )

                (
                    distance,
                    iou_by_cells,
                    overmask_by_cells,
                ) = metric_curves_by_dilation(
                    base_mask,
                    oracle,
                )

                component = margin_components(
                    actor_class,
                    posterior=posterior,
                    horizon_index=h,
                    horizon_s=horizon,
                )

                # One complete exact factorization +
                # dilation-operator parity gate per class.
                if not parity_done[
                    actor_class
                ]:
                    effective_cells = (
                        prove_component_factorization(
                            actor_class,
                            candidates[
                                actor_class
                            ],
                            posterior=posterior,
                            horizon_index=h,
                            horizon_s=horizon,
                            component=component,
                            theta_step_rad=theta_step_rad,
                        )
                    )

                    for cells in effective_cells:
                        actual = (
                            dilate_mask_theta(
                                base_mask[
                                    None,
                                    :,
                                    :,
                                ],
                                dilation_cells=
                                    int(
                                        cells
                                    ),
                            )[0]
                        )

                        accelerated = (
                            distance
                            <=
                            int(
                                cells
                            )
                        )

                        require(
                            np.array_equal(
                                actual,
                                accelerated,
                            ),
                            (
                                "Distance-transform dilation "
                                "does not equal frozen operator "
                                f"class={actor_class} "
                                f"cells={cells}"
                            ),
                        )

                    parity_done[
                        actor_class
                    ] = True

                    print(
                        actor_class,
                        "factorization/operator parity = PASS",
                    )

                class_candidates = (
                    candidates[
                        actor_class
                    ]
                )

                pair_iou = iou_sums[
                    actor_class
                ]

                pair_over = overmask_sums[
                    actor_class
                ]

                for candidate_index, candidate in enumerate(
                    class_candidates
                ):
                    (
                        _m_extra,
                        _raw_cells,
                        effective_cells,
                    ) = candidate_dilation_cells(
                        actor_class,
                        candidate,
                        component=component,
                        horizon_s=horizon,
                        theta_step_rad=theta_step_rad,
                    )

                    pair_iou[
                        candidate_index
                    ] += float(
                        iou_by_cells[
                            effective_cells
                        ]
                    )

                    pair_over[
                        candidate_index
                    ] += float(
                        overmask_by_cells[
                            effective_cells
                        ]
                    )

                scored_pairs[
                    actor_class
                ] += 1

            processed_actors += 1

            if (
                processed_actors
                %
                100
                ==
                0
            ):
                print(
                    "processed actors =",
                    f"{processed_actors} / "
                    f"{len(manifest_rows)}",
                )

        if (
            scenario_rank
            %
            20
            ==
            0
        ):
            print(
                "development scenarios =",
                f"{scenario_rank} / 120",
            )

    require(
        all(
            parity_done.values()
        ),
        (
            "Not all class dilation routes "
            "passed exact parity."
        ),
    )

    # --------------------------------------------------------
    # E. Frozen classwise selection
    # --------------------------------------------------------

    print()
    print(
        "===== E. EXACT PREREGISTERED STAGE-2 SELECTION ====="
    )

    selected = {}
    score_rows = []

    for actor_class in CLASS_ORDER:
        n_pairs = int(
            scored_pairs[
                actor_class
            ]
        )

        require(
            n_pairs > 0,
            (
                "No Stage-2 objective support "
                f"for {actor_class}."
            ),
        )

        mean_iou = (
            iou_sums[
                actor_class
            ]
            /
            float(
                n_pairs
            )
        )

        mean_overmask = (
            overmask_sums[
                actor_class
            ]
            /
            float(
                n_pairs
            )
        )

        winner = select_candidate(
            actor_class,
            candidates[
                actor_class
            ],
            mean_iou,
            mean_overmask,
        )

        winner[
            "stage1_k"
        ] = int(
            k_by_class[
                actor_class
            ]
        )

        winner[
            "stage1_gamma"
        ] = float(
            k_by_class[
                actor_class
            ]
            /
            N_MC
        )

        winner[
            "scored_actor_horizon_pairs"
        ] = n_pairs

        winner[
            "invalid_future_actor_horizon_pairs_NA"
        ] = int(
            invalid_future_pairs[
                actor_class
            ]
        )

        selected[
            actor_class
        ] = winner

        for index, candidate in enumerate(
            candidates[
                actor_class
            ]
        ):
            score_rows.append(
                {
                    "object_type":
                        actor_class,

                    "candidate_index":
                        index,

                    "base_margin_m":
                        float(
                            candidate[
                                "base_margin_m"
                            ]
                        ),

                    "uncertainty_multiplier":
                        float(
                            candidate[
                                "uncertainty_multiplier"
                            ]
                        ),

                    "motion_multiplier":
                        float(
                            candidate[
                                "motion_multiplier"
                            ]
                        ),

                    "macro_actor_horizon_IoU":
                        float(
                            mean_iou[
                                index
                            ]
                        ),

                    "macro_actor_horizon_over_masking_area":
                        float(
                            mean_overmask[
                                index
                            ]
                        ),
                }
            )

        print()
        print(
            actor_class
        )

        print(
            "  scored pairs       =",
            n_pairs,
        )

        print(
            "  invalid future / NA=",
            invalid_future_pairs[
                actor_class
            ],
        )

        print(
            "  base margin m      =",
            winner[
                "base_margin_m"
            ],
        )

        print(
            "  uncertainty mult   =",
            winner[
                "uncertainty_multiplier"
            ],
        )

        print(
            "  motion mult        =",
            winner[
                "motion_multiplier"
            ],
        )

        print(
            "  macro IoU          =",
            winner[
                "macro_actor_horizon_IoU"
            ],
        )

        print(
            "  macro overmask     =",
            winner[
                "macro_actor_horizon_over_masking_area"
            ],
        )

        print(
            "  primary ties       =",
            winner[
                "primary_IoU_maximizer_count"
            ],
        )

        print(
            "  after overmask ties=",
            winner[
                "secondary_overmask_winner_count"
            ],
        )

    # --------------------------------------------------------
    # F. Post-selection regression
    # --------------------------------------------------------

    print()
    print(
        "===== F. POST-SELECTION REGRESSION ====="
    )

    tests_post = regression()

    print(
        "Stage6 regression =",
        f"{tests_post} / {tests_post} PASS",
    )

    # --------------------------------------------------------
    # G. Write-once freeze
    # --------------------------------------------------------

    print()
    print(
        "===== G. WRITE-ONCE STAGE-2 MARGIN FREEZE ====="
    )

    score_rows.sort(
        key=lambda row: (
            CLASS_ORDER.index(
                row[
                    "object_type"
                ]
            ),
            row[
                "candidate_index"
            ],
        )
    )

    write_once_jsonl(
        SCORES,
        score_rows,
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.8_development_stage2_class_margin",

        "status":
            "FROZEN_STAGE2_CLASS_MARGIN_BEFORE_STAGE3_FLOOR_SELECTION",

        "development_only":
            True,

        "formal_outcomes_read":
            False,

        "stage1_gamma": {
            actor_class: {
                "k":
                    int(
                        k_by_class[
                            actor_class
                        ]
                    ),

                "gamma":
                    float(
                        k_by_class[
                            actor_class
                        ]
                        /
                        N_MC
                    ),
            }
            for actor_class
            in CLASS_ORDER
        },

        "selected":
            selected,

        "objective":
            (
                "maximize macro actor-horizon full-box "
                "mask IoU after additional class-margin "
                "theta dilation"
            ),

        "secondary_tie_break":
            (
                "lower macro actor-horizon "
                "over-masking area"
            ),

        "overmask_semantics":
            (
                "|candidate_dilated_mask \\ "
                "actor_oracle_mask| / "
                "|full actuator grid|"
            ),

        "margin_semantics": {
            "dilation_input":
                "additional_class_margin_m / m_extra",

            "PartA_generic_lateral_margin_m":
                1.0,

            "PartA_generic_margin_reapplied_in_stage2_dilation":
                False,

            "vehicle":
                (
                    "b + u*sigma_lat + "
                    "q*tau*v_close"
                ),

            "pedestrian":
                "b + u*sigma_lat",

            "cyclist":
                (
                    "b + u*sigma_lat + "
                    "q*tau*v_lat"
                ),
        },

        "stage2_only": {
            "gamma_selected":
                True,

            "class_margin_selected":
                True,

            "class_floor_selected":
                False,

            "temporal_smoothing_selected":
                False,

            "actuation_rate_selected":
                False,
        },

        "immutable_after_freeze": [
            "Stage-1 class k/gamma",
            "vehicle base/uncertainty/closing coefficients",
            "pedestrian base/uncertainty coefficients",
            "cyclist base/uncertainty/lateral coefficients",
        ],

        "upstream": {
            "input_binding": {
                "path":
                    str(
                        INPUT_BINDING
                    ),

                "sha256":
                    input_binding_sha,
            },

            "stage1_freeze": {
                "path":
                    str(
                        STAGE1_FREEZE
                    ),

                "sha256":
                    EXPECTED[
                        "stage1_freeze"
                    ],
            },

            "posterior_sha256":
                posterior_sha,

            **seals,
        },

        "scientific_boundary": {
            "floor_selection":
                False,

            "temporal_rate_selection":
                False,

            "primary_acceptance_tested":
                False,

            "NI_bounds_modified":
                False,

            "formal_evaluation":
                False,
        },
    }

    write_once_json(
        FREEZE,
        freeze_payload,
    )

    scores_sha = file_sha(
        SCORES
    )

    freeze_sha = file_sha(
        FREEZE
    )

    print(
        "candidate scores SHA =",
        scores_sha,
    )

    print(
        "Stage-2 freeze SHA    =",
        freeze_sha,
    )

    free_after = free_gib()

    require(
        free_after
        >=
        HARD_RESERVE_GIB,
        (
            "Post-run disk reserve failed: "
            f"{free_after:.3f} GiB"
        ),
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.8_development_stage2_class_margin",

        "status":
            "PASS_STAGE2_CLASS_MARGIN_SELECTED_FROZEN",

        "selected":
            selected,

        "support": {
            actor_class: {
                "scored_actor_horizon_pairs":
                    int(
                        scored_pairs[
                            actor_class
                        ]
                    ),

                "invalid_future_actor_horizon_pairs_NA":
                    int(
                        invalid_future_pairs[
                            actor_class
                        ]
                    ),
            }
            for actor_class
            in CLASS_ORDER
        },

        "scientific_boundary": {
            "P_occ_opened":
                True,

            "posterior_opened":
                True,

            "future_GT_role":
                "development evaluator only",

            "Stage1_gamma_modified":
                False,

            "floor_selection":
                False,

            "temporal_rate_selection":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "regression": {
            "pre":
                tests_pre,

            "post":
                tests_post,
        },

        "storage": {
            "free_GiB":
                free_after,

            "reserve_GiB":
                HARD_RESERVE_GIB,

            "reserve_pass":
                True,
        },

        "outputs": {
            "input_binding": {
                "path":
                    str(
                        INPUT_BINDING
                    ),

                "sha256":
                    input_binding_sha,
            },

            "scores": {
                "path":
                    str(
                        SCORES
                    ),

                "sha256":
                    scores_sha,
            },

            "freeze": {
                "path":
                    str(
                        FREEZE
                    ),

                "sha256":
                    freeze_sha,
            },
        },

        "next":
            (
                "Stage-3 class-floor development selection "
                "with Stage-1 gamma and Stage-2 margins "
                "immutable"
            ),
    }

    write_once_json(
        REPORT,
        report_payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 STAGE-2 CLASS-MARGIN SELECTION — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Stage-1 gamma              = IMMUTABLE"
    )

    print(
        "vehicle base/u/q           =",
        selected[
            TYPE_VEHICLE
        ][
            "base_margin_m"
        ],
        selected[
            TYPE_VEHICLE
        ][
            "uncertainty_multiplier"
        ],
        selected[
            TYPE_VEHICLE
        ][
            "motion_multiplier"
        ],
    )

    print(
        "pedestrian base/u          =",
        selected[
            TYPE_PEDESTRIAN
        ][
            "base_margin_m"
        ],
        selected[
            TYPE_PEDESTRIAN
        ][
            "uncertainty_multiplier"
        ],
    )

    print(
        "cyclist base/u/q           =",
        selected[
            TYPE_CYCLIST
        ][
            "base_margin_m"
        ],
        selected[
            TYPE_CYCLIST
        ][
            "uncertainty_multiplier"
        ],
        selected[
            TYPE_CYCLIST
        ][
            "motion_multiplier"
        ],
    )

    print(
        "margin dilation            = M_EXTRA ONLY"
    )

    print(
        "Part-A generic margin      = NOT DOUBLE-APPLIED"
    )

    print(
        "floor selection            = NOT EXECUTED"
    )

    print(
        "temporal/rate selection    = NOT EXECUTED"
    )

    print(
        "primary acceptance gate    = NOT TESTED"
    )

    print(
        "NI bounds                  = UNCHANGED"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "Stage6 regression          =",
        f"{tests_post} / {tests_post} PASS",
    )

    print(
        "free GiB                   =",
        f"{free_after:.3f}",
    )

    print(
        "STATUS = "
        "PASS_STAGE2_CLASS_MARGIN_SELECTED_FROZEN"
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
        "BLOCK 6.8 STAGE-2 CLASS-MARGIN SELECTION = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma remains FROZEN."
    )

    print(
        "Do not rerun or alter Stage-1 based "
        "on Stage-2 outcomes."
    )

    print(
        "Do not alter NI bounds, timestamp alignment, "
        "metric semantics or m_extra binding."
    )

    print(
        "floor selection         = NOT EXECUTED BY THIS RUNNER"
    )

    print(
        "temporal/rate selection = NOT EXECUTED BY THIS RUNNER"
    )

    print(
        "primary acceptance      = NOT TESTED BY THIS RUNNER"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

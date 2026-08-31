from __future__ import annotations

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
from iscai_stage6.adb.geometry import (
    Box3D,
    project_box_to_headlamp,
)
from iscai_stage6.adb.metric_semantics import (
    cyclist_visibility_proxy,
    over_masking_area,
    pedestrian_visibility_proxy,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

LEDGER = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_decision_ledger.jsonl"
)

T0_MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

REACTIVE_FREEZE_REPORT = (
    S6
    / "reports/"
      "block68_reactive_decision_ledger_freeze.json"
)

GRID_BINDING = (
    S6
    / "configs/"
      "stage6_frozen_illumination_grid_runtime_binding.json"
)

EVALUATOR_CONTRACT = (
    S6
    / "configs/"
      "stage6_evaluator_reference_contract.json"
)

GEOMETRY_BIND_REPORT = (
    S6
    / "reports/"
      "block68_part2b_part2_geometry_identity_bind.json"
)

NI_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "noninferiority_bounds.py"
)

PREOUTCOME_CONTRACT = (
    S6
    / "configs/"
      "stage6_reactive_future_truth_scoring_contract.json"
)

RAW_VALUES = (
    S6
    / "artifacts/block68/"
      "block68_reactive_only_ni_values.jsonl"
)

DELTA_FREEZE = (
    S6
    / "configs/"
      "stage6_exact_noninferiority_deltas.json"
)

ACCEPTANCE_POLICY = (
    S6
    / "configs/"
      "stage6_preformal_primary_acceptance_policy.json"
)

FINAL_REPORT = (
    S6
    / "reports/"
      "block68_reactive_only_ni_delta_freeze.json"
)


EXPECTED_SHA = {
    "ledger":
        "ac2b954571cdd0166bbcfa5bb8b2956785d1e132288dbe4af7fe2d55adeafcaa",

    "t0_maps":
        "7c5d63c76c3d9ef74adbbd7a68e3ce7483b1dd80fd262cba06f6ce1909994993",

    "reactive_freeze_report":
        "6ca868dbdf13e8a675813d4af2d3c814ce5297155126e5124ac452131670903b",

    "grid_binding":
        "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc",

    "evaluator_contract":
        "c9ff0390820a3c37702fb391ba603244fffcf7c03c476045b590aa483e89bd7e",

    "ni_runtime":
        "3c67a7f152359710cb24db1a6e01c4b580728e6480b7c5d8f9c40a41349cf1cd",
}

EXPECTED_TESTS = 224

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

# WOMD is frozen at 100-ms annotated intervals.
FUTURE_INDEX_OFFSETS = (
    1,
    3,
    5,
    10,
)

TIMESTAMP_TOLERANCE_S = 1.0e-3

THETA_MIN_RAD = -0.4363323129985824
THETA_MAX_RAD = +0.4363323129985824
RANGE_MIN_M = 0.0
RANGE_MAX_M = 150.0

N_THETA = 501
N_RANGE = 301

GRID_2D_SHAPE = (
    N_THETA,
    N_RANGE,
)

SCHEDULE_SHAPE = (
    len(HORIZONS_S),
    N_THETA,
    N_RANGE,
)

TYPE_VEHICLE = "TYPE_VEHICLE"
TYPE_PEDESTRIAN = "TYPE_PEDESTRIAN"
TYPE_CYCLIST = "TYPE_CYCLIST"

ALLOWED_CLASSES = {
    TYPE_VEHICLE,
    TYPE_PEDESTRIAN,
    TYPE_CYCLIST,
}

NI_Z_SCALE = 1.96
NI_DELTA_MIN = 0.02
NI_DELTA_MAX = 0.05

MIN_VRU_SUPPORTED_SCENARIOS = 20

HARD_RESERVE_GIB = 250.0


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def exact_file(
    path: Path,
    expected_sha: str,
    label: str,
):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(
        path
    )

    require(
        actual == expected_sha,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected_sha}\n"
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
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            value = json.loads(
                text
            )

            require(
                isinstance(
                    value,
                    dict,
                ),
                (
                    f"{path} line "
                    f"{line_number} is not an object."
                ),
            )

            rows.append(
                value
            )

    return rows


def canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def write_once_exact(
    path: Path,
    payload,
):
    data = canonical_json_bytes(
        payload
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing write-once artifact differs: "
                f"{path}"
            ),
        )

        return

    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_bytes(
        data
    )

    temporary.replace(
        path
    )


def write_jsonl_once(
    path: Path,
    rows,
):
    lines = []

    for row in rows:
        lines.append(
            json.dumps(
                row,
                sort_keys=True,
                allow_nan=False,
            )
        )

    data = (
        "\n".join(lines)
        +
        "\n"
    ).encode(
        "utf-8"
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing write-once JSONL differs: "
                f"{path}"
            ),
        )

        return

    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_bytes(
        data
    )

    temporary.replace(
        path
    )


def run_regression():
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

    output = process.stdout

    count = None

    for line in output.splitlines():
        text = line.strip()

        if (
            text.startswith("Ran ")
            and
            " test" in text
        ):
            try:
                count = int(
                    text.split()[1]
                )
            except Exception:
                pass

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                output.splitlines()[-100:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} "
            f"Stage6 tests; got {count}."
        ),
    )

    return count


def free_gib() -> float:
    return (
        shutil.disk_usage(
            S6
        ).free
        /
        (1024 ** 3)
    )


def circular_abs_difference(
    theta,
    center,
):
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
            np.sin(
                difference
            ),
            np.cos(
                difference
            ),
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
    """
    Frozen pre-outcome evaluator rasterization.

    A grid cell belongs to the constructed future full-box
    support iff its theta/range cell rectangle intersects the
    corner-derived angular/radial bounding support.

    Angular support:
      circular distance(theta_cell_center, box_theta_center)
      <= box_theta_span/2 + theta_half_cell

    Radial support:
      [r_cell-half, r_cell+half] intersects
      [projected ground_range_min, ground_range_max]

    This is evaluator-only and uses all eight physical box
    corners through project_box_to_headlamp().
    """

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


def future_box_H0(
    *,
    state,
    T_H0_from_W,
):
    values = (
        float(state.center_x),
        float(state.center_y),
        float(state.center_z),
        float(state.length),
        float(state.width),
        float(state.height),
        float(state.heading),
    )

    require(
        all(
            math.isfinite(value)
            for value in values
        ),
        "Valid future GT state has non-finite box geometry.",
    )

    require(
        float(state.length) > 0.0
        and
        float(state.width) > 0.0
        and
        float(state.height) > 0.0,
        "Valid future GT state has non-positive dimensions.",
    )

    center_H0 = (
        T_H0_from_W.apply_point(
            (
                float(
                    state.center_x
                ),
                float(
                    state.center_y
                ),
                float(
                    state.center_z
                ),
            )
        )
    )

    rotation = np.asarray(
        T_H0_from_W.rotation,
        dtype=np.float64,
    )

    require(
        rotation.shape == (3, 3),
        (
            "Unexpected T_H0_from_W rotation "
            f"shape: {rotation.shape}"
        ),
    )

    forward_W = np.asarray(
        [
            math.cos(
                float(
                    state.heading
                )
            ),
            math.sin(
                float(
                    state.heading
                )
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

    require(
        np.all(
            np.isfinite(
                forward_H0
            )
        ),
        "Future heading transform produced non-finite values.",
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
            float(
                center_H0[0]
            ),
            float(
                center_H0[1]
            ),
            float(
                center_H0[2]
            ),
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


def union_supports(
    supports,
):
    result = np.zeros(
        SCHEDULE_SHAPE,
        dtype=bool,
    )

    for support in supports:
        value = np.asarray(
            support,
            dtype=bool,
        )

        require(
            value.shape
            ==
            SCHEDULE_SHAPE,
            (
                "Actor support shape mismatch: "
                f"{value.shape}"
            ),
        )

        result |= value

    return result


def nullable_metric(value):
    numeric = float(
        value
    )

    if math.isnan(
        numeric
    ):
        return None

    require(
        math.isfinite(
            numeric
        ),
        "Metric is non-finite.",
    )

    return numeric


def summarize_and_derive_delta(
    values,
    *,
    required_n,
    label,
):
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    require(
        array.ndim == 1,
        f"{label}: vector is not 1-D.",
    )

    require(
        len(array) >= required_n,
        (
            f"{label}: insufficient development "
            f"support n={len(array)} < {required_n}."
        ),
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        f"{label}: non-finite reactive values.",
    )

    require(
        len(array) >= 2,
        f"{label}: sample SD requires n>=2.",
    )

    mean_value = float(
        np.mean(
            array
        )
    )

    sample_std = float(
        np.std(
            array,
            ddof=1,
        )
    )

    standard_error = float(
        sample_std
        /
        math.sqrt(
            len(array)
        )
    )

    raw_delta = float(
        NI_Z_SCALE
        *
        standard_error
    )

    frozen_delta = float(
        np.clip(
            raw_delta,
            NI_DELTA_MIN,
            NI_DELTA_MAX,
        )
    )

    return {
        "n":
            int(
                len(array)
            ),

        "reactive_mean":
            mean_value,

        "reactive_sample_std_ddof1":
            sample_std,

        "reactive_standard_error":
            standard_error,

        "fixed_variability_scale":
            NI_Z_SCALE,

        "raw_delta":
            raw_delta,

        "clip_lower":
            NI_DELTA_MIN,

        "clip_upper":
            NI_DELTA_MAX,

        "frozen_delta":
            frozen_delta,

        "delta_uses_predictive_values":
            False,
    }


def verify_idempotent_final():
    if not FINAL_REPORT.is_file():
        return False

    report = read_json(
        FINAL_REPORT
    )

    if (
        report.get(
            "status"
        )
        !=
        "PASS_REACTIVE_ONLY_NI_DELTAS_FROZEN"
    ):
        return False

    outputs = report.get(
        "outputs",
        {},
    )

    for key, path in (
        ("raw_values", RAW_VALUES),
        ("delta_freeze", DELTA_FREEZE),
        ("acceptance_policy", ACCEPTANCE_POLICY),
        ("preoutcome_contract", PREOUTCOME_CONTRACT),
    ):
        require(
            path.is_file(),
            f"Idempotent output missing: {path}",
        )

        expected = (
            outputs
            .get(
                key,
                {},
            )
            .get(
                "sha256"
            )
        )

        require(
            isinstance(
                expected,
                str,
            ),
            (
                "Final report lacks output SHA: "
                f"{key}"
            ),
        )

        require(
            file_sha(
                path
            )
            ==
            expected,
            (
                "Idempotent output hash mismatch: "
                f"{key}"
            ),
        )

    print(
        "existing final PASS = EXACT READBACK PASS"
    )

    print(
        "STATUS = PASS_REACTIVE_ONLY_NI_DELTAS_FROZEN"
    )

    print(
        "No scientific recomputation performed."
    )

    return True


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART2C"
    )
    print(
        "FIRST EVALUATOR-ONLY FUTURE-TRUTH RUN"
    )
    print(
        "ORIGINAL-REACTIVE ONLY → NI DELTA FREEZE"
    )
    print(
        "NO PREDICTIVE PERFORMANCE / NO P_OCC / NO FORMAL"
    )
    print(
        "============================================================"
    )

    if verify_idempotent_final():
        return

    # --------------------------------------------------------
    # A. Exact frozen boundary
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT FROZEN BOUNDARY ====="
    )

    seals = {}

    for key, path in (
        (
            "ledger",
            LEDGER,
        ),
        (
            "t0_maps",
            T0_MAPS,
        ),
        (
            "reactive_freeze_report",
            REACTIVE_FREEZE_REPORT,
        ),
        (
            "grid_binding",
            GRID_BINDING,
        ),
        (
            "evaluator_contract",
            EVALUATOR_CONTRACT,
        ),
        (
            "ni_runtime",
            NI_RUNTIME,
        ),
    ):
        seals[key] = exact_file(
            path,
            EXPECTED_SHA[key],
            key,
        )

        print(
            f"{key:24s} = EXACT PASS"
        )

    geometry_bind = read_json(
        GEOMETRY_BIND_REPORT
    )

    require(
        geometry_bind.get(
            "status"
        )
        ==
        "PASS_FUTURE_EVALUATOR_GEOMETRY_IDENTITY_BOUND",
        (
            "Geometry/identity bind report "
            "is not authoritative PASS."
        ),
    )

    geometry_module_path = Path(
        geometry_bind[
            "geometry"
        ][
            "module"
        ]
    )

    geometry_module_expected_sha = (
        geometry_bind[
            "geometry"
        ][
            "module_sha256"
        ]
    )

    require(
        file_sha(
            geometry_module_path
        )
        ==
        geometry_module_expected_sha,
        "Geometry module changed after binding.",
    )

    print(
        "geometry/identity bind    = EXACT RUNTIME PASS"
    )

    # --------------------------------------------------------
    # B. Schema-only population checks
    # --------------------------------------------------------

    print()
    print(
        "===== B. DEVELOPMENT POPULATION / IDENTITY ====="
    )

    ledger_rows = read_jsonl(
        LEDGER
    )

    cohort_rows = read_jsonl(
        COHORT
    )

    require(
        len(
            ledger_rows
        )
        ==
        120,
        "Reactive ledger must contain 120 rows.",
    )

    require(
        len(
            cohort_rows
        )
        ==
        120,
        "Development cohort must contain 120 rows.",
    )

    ledger_ids = [
        row[
            "scenario_id"
        ]
        for row in ledger_rows
    ]

    cohort_ids = [
        row[
            "scenario_id"
        ]
        for row in cohort_rows
    ]

    require(
        len(
            set(
                ledger_ids
            )
        )
        ==
        120,
        "Duplicate scenario in reactive ledger.",
    )

    require(
        len(
            set(
                cohort_ids
            )
        )
        ==
        120,
        "Duplicate scenario in development cohort.",
    )

    require(
        set(
            ledger_ids
        )
        ==
        set(
            cohort_ids
        ),
        (
            "Reactive ledger population differs "
            "from development cohort."
        ),
    )

    cohort_by_sid = {
        row[
            "scenario_id"
        ]:
        row
        for row in cohort_rows
    }

    print(
        "development cohort       = 120 / 120 PASS"
    )
    print(
        "reactive ledger          = 120 / 120 PASS"
    )
    print(
        "scenario set parity      = EXACT PASS"
    )
    print(
        "future validity selector = NO"
    )
    print(
        "future duration selector = NO"
    )

    # --------------------------------------------------------
    # C. PRE-OUTCOME evaluator/scoring contract
    # Absolutely before np.load(t0 maps) or future state reads.
    # --------------------------------------------------------

    print()
    print(
        "===== C. PRE-OUTCOME SCORING CONTRACT FREEZE ====="
    )

    theta_step = (
        THETA_MAX_RAD
        -
        THETA_MIN_RAD
    ) / (
        N_THETA
        -
        1
    )

    range_step = (
        RANGE_MAX_M
        -
        RANGE_MIN_M
    ) / (
        N_RANGE
        -
        1
    )

    preoutcome_contract = {
        "stage":
            6,

        "block":
            "6.8_reactive_only_future_truth_scoring",

        "status":
            "FROZEN_BEFORE_FUTURE_TRUTH_READ",

        "purpose":
            (
                "development-only original-reactive "
                "comparator statistics for pre-formal "
                "non-inferiority delta freeze"
            ),

        "population": {
            "scenario_count":
                120,

            "actor_population_source":
                str(
                    LEDGER
                ),

            "population_fixed_at_t0":
                True,

            "future_validity_used_as_selector":
                False,

            "future_track_duration_used_as_selector":
                False,

            "tracks_to_predict_used_as_selector":
                False,

            "objects_of_interest_used_as_selector":
                False,

            "invalid_future_state_semantics":
                (
                    "actor remains in frozen t0 population; "
                    "that actor contributes empty evaluator "
                    "support at that horizon"
                ),
        },

        "future_truth": {
            "role":
                "EVALUATOR_ONLY_AFTER_FROZEN_T0_DECISION",

            "controller_input":
                False,

            "policy_selection_input":
                False,

            "prediction_input":
                False,

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "WOMD_index_offsets":
                list(
                    FUTURE_INDEX_OFFSETS
                ),

            "timestamp_tolerance_s":
                TIMESTAMP_TOLERANCE_S,

            "interpolation":
                False,
        },

        "future_box_geometry": {
            "center":
                "future WOMD GT center transformed W->frozen H0",

            "dimensions":
                "future WOMD GT length/width/height; evaluator only",

            "heading":
                (
                    "future WOMD GT world forward vector "
                    "rotated by frozen T_H0_from_W; "
                    "yaw=atan2(H0_y,H0_x)"
                ),

            "projection":
                "iscai_stage6.adb.geometry.project_box_to_headlamp",

            "controller_path":
                False,

            "centroid_only":
                False,

            "physical_box_corners":
                8,
        },

        "grid": {
            "theta_min_rad":
                THETA_MIN_RAD,

            "theta_max_rad":
                THETA_MAX_RAD,

            "range_min_m":
                RANGE_MIN_M,

            "range_max_m":
                RANGE_MAX_M,

            "n_theta":
                N_THETA,

            "n_range":
                N_RANGE,

            "theta_step_rad":
                theta_step,

            "range_step_m":
                range_step,

            "source_binding_sha256":
                EXPECTED_SHA[
                    "grid_binding"
                ],
        },

        "evaluator_rasterization": {
            "freeze_timing":
                "BEFORE_ANY_FUTURE_TRUTH_VALUE_READ",

            "semantics":
                (
                    "cell-rectangle intersection with "
                    "corner-derived projected full-box "
                    "theta/range bounding support"
                ),

            "theta_rule":
                (
                    "circular_abs(theta_cell_center-theta_box_center) "
                    "<= theta_span/2 + theta_cell_half_step"
                ),

            "range_rule":
                (
                    "[r_center-half_step,r_center+half_step] "
                    "intersects "
                    "[ground_range_min,ground_range_max]"
                ),

            "conservative_subcell_intersection":
                True,

            "centroid_rasterization":
                False,
        },

        "original_reactive_candidate": {
            "action_source":
                str(
                    T0_MAPS
                ),

            "decision_time":
                "t0",

            "four_horizon_schedule":
                "HOLD_T0_ACTION_UNCHANGED",

            "binary_dim_support":
                "D_tau = [I_final_tau < 1.0]",
        },

        "oracle_support": {
            "semantics":
                (
                    "O_tau = union of constructed future "
                    "full-box supports of frozen t0-selected "
                    "ADB actors"
                ),

            "measured_ADB_ground_truth_claim":
                False,
        },

        "metrics_allowed_in_this_run": [
            "over_masking_area",
            "pedestrian_visibility_proxy",
            "cyclist_visibility_proxy",
        ],

        "metrics_forbidden_in_this_run": [
            "vehicle_shadow_zone_violation",
            "glare_risk_exposure",
            "mask_IoU_with_constructed_oracle_future_mask",
            "road_illumination_retention",
            "false_dimming",
            "temporal_smoothness",
            "flicker_change_rate",
            "energy_consumption",
            "actuation_latency",
        ],

        "NI_delta_rule": {
            "comparator_values":
                "ORIGINAL_REACTIVE_ONLY",

            "sample_std":
                "ddof=1",

            "SE":
                "sample_std/sqrt(n)",

            "raw_delta":
                "1.96*SE",

            "fixed_variability_scale":
                NI_Z_SCALE,

            "confidence_interval_claim":
                False,

            "clip":
                [
                    NI_DELTA_MIN,
                    NI_DELTA_MAX,
                ],

            "over_masking_required_n":
                120,

            "pedestrian_min_supported_n":
                MIN_VRU_SUPPORTED_SCENARIOS,

            "cyclist_min_supported_n":
                MIN_VRU_SUPPORTED_SCENARIOS,

            "predictive_values_allowed":
                False,
        },

        "forbidden": {
            "P_occ_open":
                True,

            "predictive_controller_execution":
                True,

            "predictive_metric_read":
                True,

            "formal_file_content_open":
                True,

            "formal_evaluation":
                True,

            "controller_or_policy_tuning":
                True,
        },

        "frozen_input_hashes": {
            **seals,

            "cohort":
                file_sha(
                    COHORT
                ),

            "geometry_bind_report":
                file_sha(
                    GEOMETRY_BIND_REPORT
                ),

            "geometry_module":
                geometry_module_expected_sha,
        },
    }

    write_once_exact(
        PREOUTCOME_CONTRACT,
        preoutcome_contract,
    )

    preoutcome_sha = file_sha(
        PREOUTCOME_CONTRACT
    )

    print(
        "rasterization semantics = FROZEN PRE-OUTCOME"
    )
    print(
        "nominal offset labels    = [1, 3, 5, 10] (diagnostic only)"
    )
    print(
        "future validity selector = NO"
    )
    print(
        "predictive values        = FORBIDDEN"
    )
    print(
        "contract SHA256          =",
        preoutcome_sha,
    )

    # --------------------------------------------------------
    # D. Pre-run regression and storage
    # --------------------------------------------------------

    print()
    print(
        "===== D. PRE-SCIENTIFIC-RUN GATES ====="
    )

    tests_pre = run_regression()

    print(
        "Stage6 regression =",
        f"{tests_pre} / {tests_pre} PASS",
    )

    free_before = free_gib()

    require(
        free_before
        >=
        HARD_RESERVE_GIB,
        (
            "Disk reserve gate failed: "
            f"{free_before:.3f} GiB"
        ),
    )

    print(
        "free GiB          =",
        f"{free_before:.3f}",
    )
    print(
        "250-GiB reserve   = PASS"
    )

    # --------------------------------------------------------
    # Superseding timestamp-alignment authority
    # --------------------------------------------------------

    _alignment_path = (
        S6
        / "configs/"
          "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
    )

    require(
        _alignment_path.is_file(),
        (
            "Superseding timestamp alignment "
            f"missing: {_alignment_path}"
        ),
    )

    require(
        file_sha(
            _alignment_path
        )
        ==
        "51a3b56a119467001a9f4617e1e19dc82c548c0a30c79fc9c4a9c1f5ef27bf56",
        (
            "Superseding timestamp-alignment "
            "SHA changed."
        ),
    )

    _alignment_contract = read_json(
        _alignment_path
    )

    require(
        _alignment_contract.get(
            "status"
        )
        ==
        "FROZEN_SUPERSEDING_TIMESTAMP_ALIGNMENT_RULE",
        (
            "Superseding timestamp-alignment "
            "status mismatch."
        ),
    )

    print(
        "timestamp alignment       = "
        "NEAREST OBSERVED FUTURE SAMPLE"
    )

    print(
        "interpolation             = NO"
    )

    print(
        "timestamp tolerance gate  = NONE"
    )

    # --------------------------------------------------------
    # E. FIRST scientific outcome access
    # From this line onward future GT is evaluator-only.
    # --------------------------------------------------------

    print()
    print(
        "===== E. FIRST LEGAL FUTURE-TRUTH REACTIVE SCORING ====="
    )

    t0_maps = np.load(
        T0_MAPS,
        allow_pickle=False,
    )

    require(
        t0_maps.shape
        ==
        (
            120,
            N_THETA,
            N_RANGE,
        ),
        (
            "Frozen t0 map shape mismatch: "
            f"{t0_maps.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(
                t0_maps
            )
        ),
        "Frozen t0 maps contain non-finite values.",
    )

    require(
        np.all(
            t0_maps >= 0.0
        )
        and
        np.all(
            t0_maps <= 1.0
        ),
        "Frozen t0 illumination lies outside [0,1].",
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

    theta_half_step = (
        0.5
        *
        float(
            theta_centers[1]
            -
            theta_centers[0]
        )
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

    raw_rows = []

    overmask_values = []
    pedestrian_values = []
    cyclist_values = []

    aggregate_selected_actor_count = 0
    aggregate_future_valid_state_count = 0
    aggregate_future_invalid_state_count = 0
    aggregate_future_support_nonempty = 0

    for scenario_rank, ledger_row in enumerate(
        ledger_rows,
        start=1,
    ):
        sid = ledger_row[
            "scenario_id"
        ]

        cohort_row = cohort_by_sid[
            sid
        ]

        scenario = read_training_scenario(
            cohort_row
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            str(
                sid
            ),
            (
                "Loaded scenario identity mismatch: "
                f"{sid}"
            ),
        )

        anchor_index = int(
            scenario.current_time_index
        )

        require(
            anchor_index
            ==
            int(
                ledger_row[
                    "current_time_index"
                ]
            ),
            (
                "Anchor index mismatch for "
                f"{sid}"
            ),
        )

        if "current_time_index" in cohort_row:
            require(
                anchor_index
                ==
                int(
                    cohort_row[
                        "current_time_index"
                    ]
                ),
                (
                    "Cohort anchor mismatch for "
                    f"{sid}"
                ),
            )

        adapted = adapt_causal_womd_scenario(
            scenario
        )

        require(
            adapted.anchor_index
            ==
            anchor_index,
            (
                "Adapted anchor mismatch for "
                f"{sid}"
            ),
        )

        T_H0_from_W = (
            adapted
            .frames
            .T_H0_from_W
        )

        selected = ledger_row[
            "selected_adb_actors"
        ]

        require(
            len(
                selected
            )
            ==
            int(
                ledger_row[
                    "selected_adb_actor_count"
                ]
            ),
            (
                "Ledger selected actor count mismatch "
                f"for {sid}"
            ),
        )

        aggregate_selected_actor_count += len(
            selected
        )

        actor_supports = []
        pedestrian_actor_supports = []
        cyclist_actor_supports = []

        scenario_future_valid = 0
        scenario_future_invalid = 0
        scenario_nonempty_actor_horizon_supports = 0

        anchor_timestamp = float(
            scenario.timestamps_seconds[
                anchor_index
            ]
        )

        for actor in selected:
            track_index = int(
                actor[
                    "track_index"
                ]
            )

            require(
                0
                <=
                track_index
                <
                len(
                    scenario.tracks
                ),
                (
                    "Invalid track_index "
                    f"{track_index} in {sid}"
                ),
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
                    actor[
                        "track_id"
                    ]
                ),
                (
                    "track_id mismatch for "
                    f"{sid} track_index={track_index}"
                ),
            )

            object_type = object_type_name(
                track
            )

            require(
                object_type
                ==
                actor[
                    "object_type"
                ],
                (
                    "object_type mismatch for "
                    f"{sid} track_index={track_index}"
                ),
            )

            require(
                object_type
                in
                ALLOWED_CLASSES,
                (
                    "Unexpected ADB actor class: "
                    f"{object_type}"
                ),
            )

            actor_support = np.zeros(
                SCHEDULE_SHAPE,
                dtype=bool,
            )

            for horizon_index, (
                horizon_s,
                offset,
            ) in enumerate(
                zip(
                    HORIZONS_S,
                    FUTURE_INDEX_OFFSETS,
                    strict=True,
                )
            ):
                # SUPERSEDING_TIMESTAMP_ALIGNMENT
                # Nominal offset is diagnostic only.
                # Actual future state is selected from
                # observed timestamps nearest to the
                # requested physical horizon.

                _timestamps = np.asarray(
                    scenario.timestamps_seconds,
                    dtype=np.float64,
                )

                _target_timestamp = (
                    anchor_timestamp
                    +
                    float(
                        horizon_s
                    )
                )

                _candidate_indices = np.arange(
                    anchor_index + 1,
                    len(
                        _timestamps
                    ),
                    dtype=np.int64,
                )

                require(
                    len(
                        _candidate_indices
                    )
                    >
                    0,
                    (
                        "No future timestamp candidates "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _timestamp_errors = np.abs(
                    _timestamps[
                        _candidate_indices
                    ]
                    -
                    _target_timestamp
                )

                require(
                    np.all(
                        np.isfinite(
                            _timestamp_errors
                        )
                    ),
                    (
                        "Non-finite timestamp resolution "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _nearest_position = int(
                    np.argmin(
                        _timestamp_errors
                    )
                )

                future_index = int(
                    _candidate_indices[
                        _nearest_position
                    ]
                )

                _minimum_error = float(
                    np.min(
                        _timestamp_errors
                    )
                )

                _tie_positions = np.flatnonzero(
                    np.isclose(
                        _timestamp_errors,
                        _minimum_error,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )

                require(
                    len(
                        _tie_positions
                    )
                    >=
                    1,
                    (
                        "Nearest timestamp resolution "
                        f"failed sid={sid} h={horizon_s}"
                    ),
                )

                _earliest_tied_index = int(
                    _candidate_indices[
                        _tie_positions[
                            0
                        ]
                    ]
                )

                require(
                    future_index
                    ==
                    _earliest_tied_index,
                    (
                        "Timestamp tie-break changed "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _lower = np.flatnonzero(
                    _timestamps
                    <=
                    _target_timestamp
                )

                _upper = np.flatnonzero(
                    _timestamps
                    >=
                    _target_timestamp
                )

                require(
                    (
                        len(
                            _lower
                        )
                        >
                        0
                        and
                        len(
                            _upper
                        )
                        >
                        0
                    ),
                    (
                        "Nominal horizon not bracketed "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                _resolved_offset = (
                    future_index
                    -
                    anchor_index
                )

                require(
                    future_index
                    <
                    len(
                        scenario.timestamps_seconds
                    ),
                    (
                        "Future timestamp unavailable "
                        f"for {sid} horizon={horizon_s}"
                    ),
                )

                require(
                    future_index
                    <
                    len(
                        track.states
                    ),
                    (
                        "Future actor state unavailable "
                        f"for {sid} track={track_index}"
                    ),
                )

                actual_dt = (
                    float(
                        scenario.timestamps_seconds[
                            future_index
                        ]
                    )
                    -
                    anchor_timestamp
                )

                require(
                    math.isfinite(
                        actual_dt
                    ),
                    (
                        "Resolved future elapsed time "
                        "is non-finite "
                        f"sid={sid} h={horizon_s}"
                    ),
                )

                # Diagnostic only. There is deliberately no
                # post-hoc numerical tolerance gate.
                _resolved_timestamp_error_s = abs(
                    actual_dt
                    -
                    float(
                        horizon_s
                    )
                )

                state = track.states[
                    future_index
                ]

                # CRITICAL:
                # validity is used only to determine evaluator
                # support at this already-selected actor/horizon.
                # It does NOT change the frozen t0 population.
                if not bool(
                    state.valid
                ):
                    scenario_future_invalid += 1
                    aggregate_future_invalid_state_count += 1
                    continue

                scenario_future_valid += 1
                aggregate_future_valid_state_count += 1

                box_H0 = future_box_H0(
                    state=state,
                    T_H0_from_W=T_H0_from_W,
                )

                projected = project_box_to_headlamp(
                    box_H0,
                    state_source=(
                        "WOMD_future_GT_"
                        "evaluator_only"
                    ),
                    orientation_source=(
                        "oracle_evaluator_only"
                    ),
                    controller_path=False,
                )

                support = rasterize_projected_full_box(
                    projected,
                    theta_centers=theta_centers,
                    range_centers=range_centers,
                    theta_half_step=theta_half_step,
                    range_half_step=range_half_step,
                )

                require(
                    support.shape
                    ==
                    GRID_2D_SHAPE,
                    "Rasterized support shape mismatch.",
                )

                actor_support[
                    horizon_index
                ] = support

                if np.any(
                    support
                ):
                    scenario_nonempty_actor_horizon_supports += 1
                    aggregate_future_support_nonempty += 1

            actor_supports.append(
                actor_support
            )

            if object_type == TYPE_PEDESTRIAN:
                pedestrian_actor_supports.append(
                    actor_support
                )

            elif object_type == TYPE_CYCLIST:
                cyclist_actor_supports.append(
                    actor_support
                )

        oracle_support = union_supports(
            actor_supports
        )

        pedestrian_region = union_supports(
            pedestrian_actor_supports
        )

        cyclist_region = union_supports(
            cyclist_actor_supports
        )

        # Frozen original-reactive semantics:
        # hold t0 action unchanged across all four horizons.
        t0_illumination = np.asarray(
            t0_maps[
                scenario_rank
                -
                1
            ],
            dtype=np.float64,
        )

        reactive_schedule = np.repeat(
            t0_illumination[
                None,
                :,
                :,
            ],
            repeats=len(
                HORIZONS_S
            ),
            axis=0,
        )

        require(
            reactive_schedule.shape
            ==
            SCHEDULE_SHAPE,
            "Reactive hold schedule shape mismatch.",
        )

        reactive_dim_support = (
            reactive_schedule
            <
            1.0
        )

        overmask = float(
            over_masking_area(
                reactive_dim_support,
                oracle_support,
            )
        )

        pedestrian = nullable_metric(
            pedestrian_visibility_proxy(
                reactive_schedule,
                pedestrian_region,
            )
        )

        cyclist = nullable_metric(
            cyclist_visibility_proxy(
                reactive_schedule,
                cyclist_region,
            )
        )

        overmask_values.append(
            overmask
        )

        if pedestrian is not None:
            pedestrian_values.append(
                pedestrian
            )

        if cyclist is not None:
            cyclist_values.append(
                cyclist
            )

        raw_rows.append(
            {
                "scenario_rank":
                    scenario_rank,

                "scenario_id":
                    sid,

                "cohort_index":
                    ledger_row.get(
                        "cohort_index"
                    ),

                "selected_adb_actor_count":
                    len(
                        selected
                    ),

                "future_valid_actor_horizon_states":
                    scenario_future_valid,

                "future_invalid_actor_horizon_states":
                    scenario_future_invalid,

                "nonempty_actor_horizon_supports":
                    scenario_nonempty_actor_horizon_supports,

                "oracle_support_cell_count":
                    int(
                        np.count_nonzero(
                            oracle_support
                        )
                    ),

                "pedestrian_support_cell_count":
                    int(
                        np.count_nonzero(
                            pedestrian_region
                        )
                    ),

                "cyclist_support_cell_count":
                    int(
                        np.count_nonzero(
                            cyclist_region
                        )
                    ),

                "metrics": {
                    "over_masking_area":
                        overmask,

                    "pedestrian_visibility_proxy":
                        pedestrian,

                    "cyclist_visibility_proxy":
                        cyclist,
                },

                "causality": {
                    "decision_source":
                        "frozen_t0_reactive_ledger",

                    "future_GT_role":
                        "evaluator_only_after_decision",

                    "future_validity_used_for_population_selection":
                        False,

                    "future_track_duration_used_for_population_selection":
                        False,

                    "predictive_controller_executed":
                        False,

                    "P_occ_opened":
                        False,
                },
            }
        )

        if (
            scenario_rank % 10
            ==
            0
        ):
            print(
                "reactive evaluator scenarios =",
                f"{scenario_rank} / 120",
            )

    # --------------------------------------------------------
    # F. Reactive-only statistics and NI deltas
    # --------------------------------------------------------

    print()
    print(
        "===== F. REACTIVE-ONLY NI DELTA DERIVATION ====="
    )

    require(
        len(
            overmask_values
        )
        ==
        120,
        "Overmask vector must contain exactly 120 values.",
    )

    overmask_summary = summarize_and_derive_delta(
        overmask_values,
        required_n=120,
        label="over_masking_area",
    )

    pedestrian_summary = summarize_and_derive_delta(
        pedestrian_values,
        required_n=MIN_VRU_SUPPORTED_SCENARIOS,
        label="pedestrian_visibility_proxy",
    )

    cyclist_summary = summarize_and_derive_delta(
        cyclist_values,
        required_n=MIN_VRU_SUPPORTED_SCENARIOS,
        label="cyclist_visibility_proxy",
    )

    print(
        "overmask n/mean/std/delta =",
        overmask_summary["n"],
        f'{overmask_summary["reactive_mean"]:.9f}',
        f'{overmask_summary["reactive_sample_std_ddof1"]:.9f}',
        f'{overmask_summary["frozen_delta"]:.9f}',
    )

    print(
        "ped n/mean/std/delta      =",
        pedestrian_summary["n"],
        f'{pedestrian_summary["reactive_mean"]:.9f}',
        f'{pedestrian_summary["reactive_sample_std_ddof1"]:.9f}',
        f'{pedestrian_summary["frozen_delta"]:.9f}',
    )

    print(
        "cyclist n/mean/std/delta  =",
        cyclist_summary["n"],
        f'{cyclist_summary["reactive_mean"]:.9f}',
        f'{cyclist_summary["reactive_sample_std_ddof1"]:.9f}',
        f'{cyclist_summary["frozen_delta"]:.9f}',
    )

    # --------------------------------------------------------
    # G. Post-run regression BEFORE committing result files
    # --------------------------------------------------------

    print()
    print(
        "===== G. POST-SCIENTIFIC-RUN REGRESSION ====="
    )

    tests_post = run_regression()

    print(
        "Stage6 regression =",
        f"{tests_post} / {tests_post} PASS",
    )

    # --------------------------------------------------------
    # H. Freeze outputs
    # --------------------------------------------------------

    print()
    print(
        "===== H. WRITE-ONCE NI DELTA / ACCEPTANCE FREEZE ====="
    )

    delta_payload = {
        "stage":
            6,

        "block":
            "6.8_reactive_only_NI_delta_freeze",

        "status":
            "FROZEN_BEFORE_PREDICTIVE_DEVELOPMENT_PERFORMANCE",

        "development_population":
            120,

        "derivation":
            (
                "reactive comparator variability only; "
                "SE=sample_std(ddof=1)/sqrt(n); "
                "raw_delta=1.96*SE; "
                "delta=clip(raw_delta,0.02,0.05)"
            ),

        "fixed_variability_scale":
            NI_Z_SCALE,

        "confidence_interval_claim":
            False,

        "predictive_values_seen":
            False,

        "metrics": {
            "over_masking_area":
                overmask_summary,

            "pedestrian_visibility_proxy":
                pedestrian_summary,

            "cyclist_visibility_proxy":
                cyclist_summary,
        },

        "vehicle_shadow_zone_violation": {
            "numeric_delta":
                None,

            "formal_rule":
                "predictive_mean < reactive_mean",
        },

        "preoutcome_scoring_contract_sha256":
            preoutcome_sha,

        "NI_runtime_sha256":
            EXPECTED_SHA[
                "ni_runtime"
            ],
    }

    policy_payload = {
        "stage":
            6,

        "block":
            "6.8_preformal_primary_acceptance_policy",

        "status":
            "FROZEN_BEFORE_PREDICTIVE_DEVELOPMENT_PERFORMANCE",

        "primary_comparison":
            (
                "class_aware_predictive_ADB "
                "vs original_reactive_ADB"
            ),

        "predictive_outcomes_seen_at_freeze":
            False,

        "formal_rules": {
            "vehicle_shadow_zone_violation": {
                "direction":
                    "lower_is_better",

                "rule":
                    "predictive_mean < reactive_mean",

                "tolerance":
                    None,
            },

            "over_masking_area": {
                "direction":
                    "lower_is_better",

                "rule":
                    (
                        "predictive_mean - reactive_mean "
                        "<= frozen_delta"
                    ),

                "frozen_delta":
                    overmask_summary[
                        "frozen_delta"
                    ],
            },

            "pedestrian_visibility_proxy": {
                "direction":
                    "higher_is_better",

                "rule":
                    (
                        "predictive_mean - reactive_mean "
                        ">= -frozen_delta"
                    ),

                "frozen_delta":
                    pedestrian_summary[
                        "frozen_delta"
                    ],
            },

            "cyclist_visibility_proxy": {
                "direction":
                    "higher_is_better",

                "rule":
                    (
                        "predictive_mean - reactive_mean "
                        ">= -frozen_delta"
                    ),

                "frozen_delta":
                    cyclist_summary[
                        "frozen_delta"
                    ],
            },
        },

        "policy_tuning_after_freeze_allowed":
            False,

        "formal_parameter_tuning_allowed":
            False,

        "next_gate":
            (
                "development-only class-aware predictive "
                "runtime evaluation against these already "
                "frozen acceptance bounds"
            ),
    }

    write_jsonl_once(
        RAW_VALUES,
        raw_rows,
    )

    write_once_exact(
        DELTA_FREEZE,
        delta_payload,
    )

    write_once_exact(
        ACCEPTANCE_POLICY,
        policy_payload,
    )

    raw_sha = file_sha(
        RAW_VALUES
    )

    delta_sha = file_sha(
        DELTA_FREEZE
    )

    policy_sha = file_sha(
        ACCEPTANCE_POLICY
    )

    print(
        "reactive raw values SHA  =",
        raw_sha,
    )
    print(
        "NI delta freeze SHA      =",
        delta_sha,
    )
    print(
        "acceptance policy SHA    =",
        policy_sha,
    )

    # --------------------------------------------------------
    # I. Storage / final report
    # --------------------------------------------------------

    print()
    print(
        "===== I. FINAL SCIENTIFIC BOUNDARY ====="
    )

    free_after = free_gib()

    require(
        free_after
        >=
        HARD_RESERVE_GIB,
        (
            "Post-run disk reserve gate failed: "
            f"{free_after:.3f} GiB"
        ),
    )

    final_payload = {
        "stage":
            6,

        "block":
            "6.8_reactive_only_NI_delta_freeze",

        "status":
            "PASS_REACTIVE_ONLY_NI_DELTAS_FROZEN",

        "development_scenarios":
            120,

        "selected_t0_actor_count":
            aggregate_selected_actor_count,

        "future_evaluator": {
            "valid_actor_horizon_states":
                aggregate_future_valid_state_count,

            "invalid_actor_horizon_states":
                aggregate_future_invalid_state_count,

            "nonempty_actor_horizon_supports":
                aggregate_future_support_nonempty,

            "future_validity_used_as_selector":
                False,

            "future_track_duration_used_as_selector":
                False,
        },

        "NI_summaries": {
            "over_masking_area":
                overmask_summary,

            "pedestrian_visibility_proxy":
                pedestrian_summary,

            "cyclist_visibility_proxy":
                cyclist_summary,
        },

        "scientific_boundary": {
            "future_truth_opened":
                True,

            "future_truth_role":
                "REACTIVE_EVALUATOR_ONLY_AFTER_T0_DECISION",

            "reactive_performance_computed":
                True,

            "computed_metrics":
                [
                    "over_masking_area",
                    "pedestrian_visibility_proxy",
                    "cyclist_visibility_proxy",
                ],

            "vehicle_shadow_zone_violation_computed":
                False,

            "predictive_performance_computed":
                False,

            "P_occ_opened":
                False,

            "formal_content_opened":
                False,

            "formal_evaluation":
                False,

            "controller_policy_tuning":
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

            "hard_reserve_GiB":
                HARD_RESERVE_GIB,

            "reserve_pass":
                True,
        },

        "frozen_inputs": {
            **preoutcome_contract[
                "frozen_input_hashes"
            ],

            "superseding_timestamp_alignment":
                "51a3b56a119467001a9f4617e1e19dc82c548c0a30c79fc9c4a9c1f5ef27bf56",
        },

        "outputs": {
            "preoutcome_contract": {
                "path":
                    str(
                        PREOUTCOME_CONTRACT
                    ),

                "sha256":
                    preoutcome_sha,
            },

            "raw_values": {
                "path":
                    str(
                        RAW_VALUES
                    ),

                "sha256":
                    raw_sha,
            },

            "delta_freeze": {
                "path":
                    str(
                        DELTA_FREEZE
                    ),

                "sha256":
                    delta_sha,
            },

            "acceptance_policy": {
                "path":
                    str(
                        ACCEPTANCE_POLICY
                    ),

                "sha256":
                    policy_sha,
            },
        },

        "next":
            (
                "development-only class-aware predictive ADB "
                "evaluation; frozen NI deltas MUST NOT change"
            ),
    }

    write_once_exact(
        FINAL_REPORT,
        final_payload,
    )

    print(
        "future truth opened       = YES, EVALUATOR ONLY"
    )
    print(
        "future population selector= NO"
    )
    print(
        "reactive metrics computed = EXACTLY 3"
    )
    print(
        "predictive performance    = NOT COMPUTED"
    )
    print(
        "P_occ content             = NOT OPENED"
    )
    print(
        "formal content            = NOT OPENED"
    )
    print(
        "formal evaluation         = NO"
    )
    print(
        "policy/controller tuning  = NO"
    )
    print(
        "NI deltas                 = FROZEN"
    )
    print(
        "Stage6 regression         =",
        f"{tests_post} / {tests_post} PASS",
    )
    print(
        "free GiB                  =",
        f"{free_after:.3f}",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 REACTIVE-ONLY NI DELTA FREEZE — FINAL"
    )
    print(
        "============================================================"
    )
    print(
        "development population     = 120 / 120"
    )
    print(
        "frozen t0 decisions         = USED"
    )
    print(
        "future GT role              = EVALUATOR ONLY"
    )
    print(
        "future validity selector    = NO"
    )
    print(
        "full future box corners     = YES"
    )
    print(
        "reactive schedule           = HOLD t0 ACTION"
    )
    print(
        "overmask observations       =",
        overmask_summary[
            "n"
        ],
    )
    print(
        "ped visibility observations =",
        pedestrian_summary[
            "n"
        ],
    )
    print(
        "cyclist visibility obs      =",
        cyclist_summary[
            "n"
        ],
    )
    print(
        "delta overmask              =",
        overmask_summary[
            "frozen_delta"
        ],
    )
    print(
        "delta pedestrian            =",
        pedestrian_summary[
            "frozen_delta"
        ],
    )
    print(
        "delta cyclist               =",
        cyclist_summary[
            "frozen_delta"
        ],
    )
    print(
        "predictive outcomes seen     = NO"
    )
    print(
        "formal evaluation allowed    = NO"
    )
    print(
        "STATUS = PASS_REACTIVE_ONLY_NI_DELTAS_FROZEN"
    )
    print(
        "report =",
        FINAL_REPORT,
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
        "BLOCK 6.8 REACTIVE-ONLY NI DELTA FREEZE = BLOCKED"
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

    print()
    print(
        "IMPORTANT BOUNDARY:"
    )
    print(
        "predictive performance = NOT COMPUTED BY THIS RUNNER"
    )
    print(
        "P_occ content          = NOT OPENED BY THIS RUNNER"
    )
    print(
        "formal evaluation      = NO"
    )
    print(
        "controller/policy edit = NO"
    )
    print(
        "Do not run predictive/formal evaluation "
        "until this block is repaired and passes."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

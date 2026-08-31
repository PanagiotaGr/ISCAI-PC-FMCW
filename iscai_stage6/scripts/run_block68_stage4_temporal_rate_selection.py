from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np

from iscai_stage6.adb.class_aware_policy import (
    TYPE_CYCLIST,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
    apply_actuation_rate_limit,
    compose_class_aware_illumination,
    predictive_mask_to_illumination,
    temporal_smooth_schedule,
)

from iscai_stage6.adb.evaluator_reference import (
    frozen_road_roi,
)

from iscai_stage6.adb.metric_semantics import (
    cyclist_visibility_proxy,
    false_dimming,
    glare_risk_exposure,
    over_masking_area,
    pedestrian_visibility_proxy,
    road_illumination_retention,
    vehicle_shadow_zone_violation,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"


# ============================================================
# FROZEN INPUTS
# ============================================================

STAGE4_PREFREEZE = (
    S6
    / "configs/"
      "stage6_development_stage4_temporal_rate_scoring_freeze.json"
)

STAGE4_PREFREEZE_REPORT = (
    S6
    / "reports/"
      "block68_stage4_initial_state_scoring_freeze.json"
)

CURRENT_BINDING = (
    S6
    / "artifacts/block68/stage4_preoutcome/"
      "stage4_current_state_scenario_binding.jsonl"
)

T0_MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

STAGE3_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage3_class_floor_freeze.json"
)

CACHE_MANIFEST = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "stage3_fixed_cache_manifest.jsonl"
)

CACHE_FREEZE = (
    S6
    / "artifacts/block68/stage3_fixed_cache/"
      "stage3_fixed_cache_freeze.json"
)

CLASS_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

EVALUATOR_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "evaluator_reference.py"
)

ROAD_ROI_ARTIFACT = (
    S6
    / "artifacts/block68/"
      "frozen_road_roi_definition.json"
)


# ============================================================
# WRITE-ONCE OUTPUTS
# ============================================================

SCORES = (
    S6
    / "artifacts/block68/stage4_temporal_rate_selection/"
      "block68_stage4_candidate_scores.jsonl"
)

WINNER_SCENARIOS = (
    S6
    / "artifacts/block68/stage4_temporal_rate_selection/"
      "block68_stage4_winner_scenario_metrics.jsonl"
)

FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage4_temporal_rate_selection_freeze.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_stage4_temporal_rate_selection.json"
)


EXPECTED_SHA = {
    "stage4_prefreeze":
        "f8a0850610e4cee4984b668d5f7046c4ec6a5f873b28f645776d5f3a5ced2717",

    "stage4_prefreeze_report":
        "0e9462d9a811f4df9283a90fa74f303e6b41c817fce44e0fbcf39658ed69aa8e",

    "current_binding":
        "3a1c689484fdbd8dbf763cfb0a52b3a987a5930a236fe9692f0b084cb178387d",

    "t0_maps":
        "7c5d63c76c3d9ef74adbbd7a68e3ce7483b1dd80fd262cba06f6ce1909994993",

    "stage3_freeze":
        "c8536a1a2f888ebc3310cb900626cf0c8c0c2ca3ce34c2d321120a269c548d00",

    "cache_manifest":
        "9607f5bd6022da5471167650f652556dca8381b241adf0b7faaf06bd27edc041",

    "cache_freeze":
        "6733ba1641afc341910bd5cc822a034103585706f32db4758d606b7bf2bf786d",

    "class_runtime":
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    "metric_runtime":
        "bf8358b5a7edfe40ffac2718ca7cd5af6e7b5fdb3ff8a2823caad6d0f40f057a",

    "evaluator_runtime":
        "2015136fdcbfe3f837bcba5a2ee05578a4d8a01c4ecfd52e4dfac9f864349891",

    "road_roi":
        "832f4bc3215fe585fee781080c98b2928157f26e8a66d22f428f15604afb75e0",
}


EXPECTED_TESTS = 224
HARD_RESERVE_GIB = 250.0

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

GRID_2D = (
    501,
    301,
)

GRID_3D = (
    4,
    501,
    301,
)

GRID_CELLS_2D = (
    501
    *
    301
)

GRID_CELLS_3D = (
    4
    *
    GRID_CELLS_2D
)

RANGE_CENTERS = np.linspace(
    0.0,
    150.0,
    301,
    dtype=np.float64,
)

PARITY_ABS_TOL = 1.0e-12


SAFETY_METRICS = (
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "road_illumination_retention",
)

ALL_SWEEP_METRICS = (
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "road_illumination_retention",
    "over_masking_area",
    "false_dimming",
    "temporal_mean_absolute_change_rate",
    "flicker_change_rate",
    "rate_limit_safety_override_fraction",
)


# ============================================================
# HELPERS
# ============================================================

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


def file_sha(
    path: Path,
):
    digest = sha256()

    with path.open("rb") as stream:

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


def exact(
    path: Path,
    expected: str,
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
        actual
        ==
        expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


def read_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, raw in enumerate(
            stream,
            start=1,
        ):

            line = raw.strip()

            if not line:
                continue

            try:
                value = json.loads(
                    line
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL {path}:{line_number}"
                ) from exc

            require(
                isinstance(
                    value,
                    dict,
                ),
                (
                    "JSONL row is not object: "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(
                value
            )

    return rows


def canonical_json_bytes(
    value,
):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def canonical_jsonl_bytes(
    rows,
):
    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
        +
        "\n"
        for row in rows
    ).encode(
        "utf-8"
    )


def write_once(
    path: Path,
    payload: bytes,
):
    if path.exists():

        require(
            path.read_bytes()
            ==
            payload,
            (
                "WRITE-ONCE output differs: "
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
        payload
    )

    temporary.replace(
        path
    )


def finite_json(
    value,
):
    value = float(
        value
    )

    if math.isfinite(
        value
    ):
        return value

    return None


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

    count = None

    for line in process.stdout.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(
                    1
                )
            )

    require(
        process.returncode
        ==
        0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                process.stdout.splitlines()[-100:]
            )
        ),
    )

    require(
        count
        ==
        EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"observed {count}."
        ),
    )

    return count


def free_gib():
    return (
        shutil.disk_usage(
            S6
        ).free
        /
        (
            1024
            **
            3
        )
    )


# ============================================================
# EXACT RAW STAGE-3 CONTROLLER
# ============================================================

def build_stage3_raw(
    arrays,
):
    mask_vehicle = arrays[
        "mask_vehicle"
    ]

    mask_pedestrian = arrays[
        "mask_pedestrian"
    ]

    mask_cyclist = arrays[
        "mask_cyclist"
    ]

    vehicle = (
        predictive_mask_to_illumination(
            mask_vehicle,
            range_centers_m=
                RANGE_CENTERS,
            intensity_floor=
                0.15,
        )
    )

    pedestrian = (
        predictive_mask_to_illumination(
            mask_pedestrian,
            range_centers_m=
                RANGE_CENTERS,
            intensity_floor=
                1.0,
        )
    )

    cyclist = (
        predictive_mask_to_illumination(
            mask_cyclist,
            range_centers_m=
                RANGE_CENTERS,
            intensity_floor=
                1.0,
        )
    )

    composition = (
        compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    vehicle,

                TYPE_PEDESTRIAN:
                    pedestrian,

                TYPE_CYCLIST:
                    cyclist,
            },

            class_active_masks={
                TYPE_PEDESTRIAN:
                    mask_pedestrian,

                TYPE_CYCLIST:
                    mask_cyclist,
            },

            class_floors={
                TYPE_PEDESTRIAN:
                    1.0,

                TYPE_CYCLIST:
                    1.0,
            },
        )
    )

    raw = np.asarray(
        composition
        .raw_class_aware_illumination,
        dtype=np.float64,
    )

    require(
        raw.shape
        ==
        GRID_3D,
        (
            "Raw Stage-3 illumination shape "
            f"mismatch: {raw.shape}"
        ),
    )

    require(
        hasattr(
            composition,
            "vru_floor_guard",
        ),
        (
            "ClassAwareComposition lost "
            "vru_floor_guard."
        ),
    )

    guard = np.asarray(
        composition
        .vru_floor_guard,
        dtype=np.float64,
    )

    require(
        guard.shape
        ==
        GRID_3D,
        (
            "VRU floor-guard shape mismatch: "
            f"{guard.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(
                raw
            )
        ),
        "Raw Stage-3 illumination non-finite.",
    )

    require(
        np.all(
            np.isfinite(
                guard
            )
        ),
        "VRU floor guard non-finite.",
    )

    return (
        raw,
        guard,
    )


# ============================================================
# AUTHORITATIVE SAFETY / ROAD METRICS
# ============================================================

def output_metrics(
    final,
    arrays,
    road_roi,
):
    final = np.asarray(
        final,
        dtype=np.float64,
    )

    require(
        final.shape
        ==
        GRID_3D,
        (
            "I_final shape mismatch: "
            f"{final.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(
                final
            )
        ),
        "I_final non-finite.",
    )

    require(
        np.all(
            final
            >=
            0.0
        )
        and
        np.all(
            final
            <=
            1.0
        ),
        "I_final outside [0,1].",
    )

    dim = (
        final
        <
        1.0
    )

    return {
        "vehicle_shadow_zone_violation":
            float(
                vehicle_shadow_zone_violation(
                    dim,
                    arrays[
                        "oracle_vehicle"
                    ],
                )
            ),

        "glare_risk_exposure":
            float(
                glare_risk_exposure(
                    final,
                    arrays[
                        "vehicle_surrogate"
                    ],
                )
            ),

        "pedestrian_visibility_proxy":
            float(
                pedestrian_visibility_proxy(
                    final,
                    arrays[
                        "pedestrian_region"
                    ],
                )
            ),

        "cyclist_visibility_proxy":
            float(
                cyclist_visibility_proxy(
                    final,
                    arrays[
                        "cyclist_region"
                    ],
                )
            ),

        "road_illumination_retention":
            float(
                road_illumination_retention(
                    final,
                    road_roi,
                )
            ),

        "over_masking_area":
            float(
                over_masking_area(
                    dim,
                    arrays[
                        "oracle_all"
                    ],
                )
            ),

        "false_dimming":
            float(
                false_dimming(
                    final,
                    arrays[
                        "oracle_all"
                    ],
                )
            ),
    }


# ============================================================
# EXACT FROZEN TEMPORAL METRICS
# ============================================================

def temporal_change_metrics(
    final,
    current,
):
    previous = np.asarray(
        current,
        dtype=np.float64,
    )

    previous_t = 0.0

    smooth_terms = []
    flicker_terms = []

    for horizon_index, horizon_s in enumerate(
        HORIZONS_S
    ):
        dt = (
            float(
                horizon_s
            )
            -
            float(
                previous_t
            )
        )

        require(
            dt
            >
            0.0,
            "Non-positive frozen horizon delta.",
        )

        current_output = final[
            horizon_index
        ]

        smooth_terms.append(
            float(
                np.mean(
                    np.abs(
                        current_output
                        -
                        previous
                    ),
                    dtype=np.float64,
                )
            )
            /
            dt
        )

        flicker_terms.append(
            float(
                np.count_nonzero(
                    current_output
                    !=
                    previous
                )
            )
            /
            float(
                GRID_CELLS_2D
            )
        )

        previous = current_output
        previous_t = float(
            horizon_s
        )

    return {
        "temporal_mean_absolute_change_rate":
            float(
                math.fsum(
                    smooth_terms
                )
                /
                len(
                    smooth_terms
                )
            ),

        "flicker_change_rate":
            float(
                math.fsum(
                    flicker_terms
                )
                /
                len(
                    flicker_terms
                )
            ),
    }


# ============================================================
# ACCUMULATION
# ============================================================

def new_accumulator(
    n_candidates,
):
    return {
        metric: {
            "sum":
                np.zeros(
                    n_candidates,
                    dtype=np.float64,
                ),

            "count":
                np.zeros(
                    n_candidates,
                    dtype=np.int64,
                ),
        }
        for metric in
        ALL_SWEEP_METRICS
    }


def add_value(
    accumulator,
    metric,
    candidate_index,
    value,
):
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        return

    accumulator[
        metric
    ][
        "sum"
    ][
        candidate_index
    ] += value

    accumulator[
        metric
    ][
        "count"
    ][
        candidate_index
    ] += 1


def candidate_macro(
    accumulator,
    metric,
    index,
):
    count = int(
        accumulator[
            metric
        ][
            "count"
        ][
            index
        ]
    )

    require(
        count
        >
        0,
        (
            f"Candidate {index} has zero finite "
            f"support for {metric}."
        ),
    )

    return float(
        accumulator[
            metric
        ][
            "sum"
        ][
            index
        ]
        /
        float(
            count
        )
    )


# ============================================================
# SELECTION KEY
# ============================================================

def selection_key(
    metrics,
    candidate,
):
    risk = (
        float(
            metrics[
                "vehicle_shadow_zone_violation"
            ]
        ),

        float(
            metrics[
                "glare_risk_exposure"
            ]
        ),

        1.0
        -
        float(
            metrics[
                "pedestrian_visibility_proxy"
            ]
        ),

        1.0
        -
        float(
            metrics[
                "cyclist_visibility_proxy"
            ]
        ),

        1.0
        -
        float(
            metrics[
                "road_illumination_retention"
            ]
        ),
    )

    require(
        all(
            math.isfinite(
                value
            )
            and
            0.0
            <=
            value
            <=
            1.0
            for value in
            risk
        ),
        (
            "Safety risk vector outside [0,1]: "
            f"{risk}"
        ),
    )

    key = (
        max(
            risk
        ),

        math.fsum(
            risk
        )
        /
        5.0,

        float(
            metrics[
                "temporal_mean_absolute_change_rate"
            ]
        ),

        float(
            metrics[
                "flicker_change_rate"
            ]
        ),

        float(
            metrics[
                "rate_limit_safety_override_fraction"
            ]
        ),

        float(
            metrics[
                "over_masking_area"
            ]
        ),

        int(
            candidate[
                "time_constant_ms"
            ]
        ),

        int(
            candidate[
                "rho_dim_per_s"
            ]
        ),

        int(
            candidate[
                "rho_bright_per_s"
            ]
        ),
    )

    return (
        key,
        risk,
    )


# ============================================================
# ONE AUTHORITATIVE CANDIDATE / SCENARIO
# ============================================================

def evaluate_candidate(
    *,
    raw,
    guard,
    current,
    arrays,
    road_roi,
    candidate,
    smoothed=None,
):
    if smoothed is None:

        smoothed = (
            temporal_smooth_schedule(
                raw,

                current_illumination=
                    current,

                horizons_s=
                    HORIZONS_S,

                time_constant_s=
                    float(
                        candidate[
                            "time_constant_s"
                        ]
                    ),
            )
        )

    limited = (
        apply_actuation_rate_limit(
            smoothed,

            current_illumination=
                current,

            horizons_s=
                HORIZONS_S,

            rho_dim_per_s=
                float(
                    candidate[
                        "rho_dim_per_s"
                    ]
                ),

            rho_bright_per_s=
                float(
                    candidate[
                        "rho_bright_per_s"
                    ]
                ),

            vru_floor_guard=
                guard,
        )
    )

    final = np.asarray(
        limited.illumination,
        dtype=np.float64,
    )

    metrics = output_metrics(
        final,
        arrays,
        road_roi,
    )

    metrics.update(
        temporal_change_metrics(
            final,
            current,
        )
    )

    override_fraction = float(
        limited
        .safety_override_fraction
    )

    require(
        math.isfinite(
            override_fraction
        )
        and
        0.0
        <=
        override_fraction
        <=
        1.0,
        (
            "Invalid safety override fraction: "
            f"{override_fraction}"
        ),
    )

    if hasattr(
        limited,
        "safety_override_count",
    ):

        override_count = int(
            limited
            .safety_override_count
        )

        expected_fraction = (
            override_count
            /
            float(
                GRID_CELLS_3D
            )
        )

        require(
            abs(
                expected_fraction
                -
                override_fraction
            )
            <=
            1.0e-15,
            (
                "Rate-limit safety override fraction "
                "does not equal override_count / "
                "four-horizon actuator transitions."
            ),
        )

    metrics[
        "rate_limit_safety_override_fraction"
    ] = override_fraction

    return (
        metrics,
        final,
        limited,
    )


# ============================================================
# IDEMPOTENT FINAL READBACK
# ============================================================

def existing_final_pass():
    if not REPORT.is_file():
        return False

    value = read_json(
        REPORT
    )

    if (
        value.get(
            "status"
        )
        !=
        "PASS_STAGE4_TEMPORAL_RATE_SELECTED_FROZEN"
    ):
        return False

    for name, path in (
        (
            "scores",
            SCORES,
        ),
        (
            "winner_scenarios",
            WINNER_SCENARIOS,
        ),
        (
            "freeze",
            FREEZE,
        ),
    ):

        require(
            path.is_file(),
            (
                "Existing PASS report references "
                f"missing {name} output."
            ),
        )

        require(
            file_sha(
                path
            )
            ==
            value[
                "outputs"
            ][
                name
            ][
                "sha256"
            ],
            (
                "Existing Stage-4 frozen output "
                f"changed: {name}"
            ),
        )

    print(
        "existing Stage-4 temporal/rate freeze = EXACT PASS"
    )

    print(
        "STATUS = "
        "PASS_STAGE4_TEMPORAL_RATE_SELECTED_FROZEN"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("DEVELOPMENT STAGE-4 TEMPORAL/RATE SELECTION")
    print("EXACT 60-CANDIDATE AUTHORITATIVE SWEEP")
    print("STAGE1+2+3 IMMUTABLE / NO PRIMARY ACCEPTANCE / NO FORMAL")
    print("=" * 78)

    if existing_final_pass():
        return

    # --------------------------------------------------------
    # A. Exact boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT FROZEN SELECTION BOUNDARY =====")

    paths = {
        "stage4_prefreeze":
            STAGE4_PREFREEZE,

        "stage4_prefreeze_report":
            STAGE4_PREFREEZE_REPORT,

        "current_binding":
            CURRENT_BINDING,

        "t0_maps":
            T0_MAPS,

        "stage3_freeze":
            STAGE3_FREEZE,

        "cache_manifest":
            CACHE_MANIFEST,

        "cache_freeze":
            CACHE_FREEZE,

        "class_runtime":
            CLASS_RUNTIME,

        "metric_runtime":
            METRIC_RUNTIME,

        "evaluator_runtime":
            EVALUATOR_RUNTIME,

        "road_roi":
            ROAD_ROI_ARTIFACT,
    }

    seals = {}

    for key, path in paths.items():

        seals[
            key
        ] = exact(
            path,
            EXPECTED_SHA[
                key
            ],
            key,
        )

        print(
            f"{key:24s} = EXACT PASS"
        )

    prefreeze = read_json(
        STAGE4_PREFREEZE
    )

    require(
        prefreeze.get(
            "status"
        )
        ==
        "FROZEN_STAGE4_PREOUTCOME_INITIAL_STATE_AND_SCORING",
        (
            "Stage-4 pre-outcome scoring freeze "
            "status mismatch."
        ),
    )

    boundary = prefreeze[
        "scientific_boundary"
    ]

    require(
        boundary[
            "Stage4_outcomes_computed"
        ]
        is False,
        (
            "Pre-outcome freeze claims "
            "Stage-4 outcomes already computed."
        ),
    )

    require(
        boundary[
            "Stage4_winner_selected"
        ]
        is False,
        (
            "Pre-outcome freeze claims "
            "Stage-4 winner already selected."
        ),
    )

    require(
        boundary[
            "primary_acceptance_tested"
        ]
        is False,
        "Primary gate unexpectedly tested.",
    )

    require(
        boundary[
            "formal_evaluation"
        ]
        is False,
        "Formal evaluation unexpectedly opened.",
    )

    print(
        "Stage-1 gamma         = IMMUTABLE"
    )

    print(
        "Stage-2 margins       = IMMUTABLE"
    )

    print(
        "Stage-3 floors        = IMMUTABLE"
    )

    print(
        "primary acceptance    = NOT TESTED"
    )

    print(
        "formal evaluation     = NO"
    )

    tests_pre = run_regression()

    print(
        "Stage6 pre regression =",
        f"{tests_pre} / {tests_pre} PASS",
    )

    # --------------------------------------------------------
    # B. Exact candidate space from frozen pre-outcome file
    # --------------------------------------------------------

    print()
    print("===== B. EXACT FROZEN 60-CANDIDATE SPACE =====")

    candidates = (
        prefreeze[
            "candidate_space"
        ][
            "candidates"
        ]
    )

    require(
        len(
            candidates
        )
        ==
        60,
        (
            "Frozen Stage-4 candidate count "
            f"!=60: {len(candidates)}"
        ),
    )

    tuples = [
        (
            int(
                row[
                    "time_constant_ms"
                ]
            ),

            int(
                row[
                    "rho_dim_per_s"
                ]
            ),

            int(
                row[
                    "rho_bright_per_s"
                ]
            ),
        )
        for row in
        candidates
    ]

    require(
        len(
            set(
                tuples
            )
        )
        ==
        60,
        "Frozen Stage-4 tuples are not unique.",
    )

    require(
        all(
            rho_dim
            >=
            rho_bright
            for _, rho_dim, rho_bright in
            tuples
        ),
        "Frozen rate constraint violated.",
    )

    print(
        "candidate count = 60 EXACT"
    )

    print(
        "time constants  = [50, 100, 200, 400] ms"
    )

    print(
        "rate pairs      = 15 EXACT"
    )

    print(
        "selection key   = FROZEN / UNCHANGED"
    )

    # --------------------------------------------------------
    # C. Scenario identity and current state
    # --------------------------------------------------------

    print()
    print("===== C. EXACT 120-SCENARIO JOIN =====")

    manifest = read_jsonl(
        CACHE_MANIFEST
    )

    binding = read_jsonl(
        CURRENT_BINDING
    )

    require(
        len(
            manifest
        )
        ==
        120,
        "Stage-3 cache manifest !=120.",
    )

    require(
        len(
            binding
        )
        ==
        120,
        "Current-state binding !=120.",
    )

    binding_by_scenario = {
        str(
            row[
                "scenario_id"
            ]
        ):
            row
        for row in
        binding
    }

    require(
        len(
            binding_by_scenario
        )
        ==
        120,
        "Current-state binding scenario IDs duplicate.",
    )

    manifest_ids = {
        str(
            row[
                "scenario_id"
            ]
        )
        for row in
        manifest
    }

    require(
        manifest_ids
        ==
        set(
            binding_by_scenario
        ),
        (
            "Stage-3 cache/current-state "
            "scenario sets differ."
        ),
    )

    t0_maps = np.load(
        T0_MAPS,
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        t0_maps.shape
        ==
        (
            120,
            501,
            301,
        ),
        (
            "Frozen t0 map stack shape "
            f"changed: {t0_maps.shape}"
        ),
    )

    require(
        t0_maps.dtype
        ==
        np.dtype(
            np.float64
        ),
        (
            "Frozen t0 map dtype changed: "
            f"{t0_maps.dtype}"
        ),
    )

    road_spatial = np.asarray(
        frozen_road_roi(
            GRID_2D
        ),
        dtype=bool,
    )

    require(
        road_spatial.shape
        ==
        GRID_2D,
        "Frozen road ROI spatial shape mismatch.",
    )

    road_roi = np.broadcast_to(
        road_spatial[
            None,
            ...,
        ],
        GRID_3D,
    )

    print(
        "scenario join     = 120 / 120 EXACT"
    )

    print(
        "I_prev maps       = 120 / 120"
    )

    print(
        "road ROI schedule = EXACT 4-horizon replication"
    )

    print(
        "P_occ reopened    = NO"
    )

    print(
        "WOMD reopened     = NO"
    )

    # --------------------------------------------------------
    # D. Replay Stage-3 winner before first Stage-4 outcome
    # --------------------------------------------------------

    print()
    print("===== D. STAGE-3 RAW HANDOFF REPLAY =====")

    stage3_freeze = read_json(
        STAGE3_FREEZE
    )

    stage3_expected = (
        stage3_freeze[
            "selected"
        ][
            "metrics"
        ]
    )

    stage3_values = {
        metric:
            []
        for metric in (
            "vehicle_shadow_zone_violation",
            "glare_risk_exposure",
            "pedestrian_visibility_proxy",
            "cyclist_visibility_proxy",
            "road_illumination_retention",
            "over_masking_area",
            "false_dimming",
        )
    }

    for scenario_number, item in enumerate(
        manifest,
        start=1,
    ):

        scenario_id = str(
            item[
                "scenario_id"
            ]
        )

        with np.load(
            Path(
                item[
                    "npz_path"
                ]
            ),
            allow_pickle=False,
        ) as loaded:

            arrays = {
                name:
                    np.asarray(
                        loaded[
                            name
                        ],
                        dtype=bool,
                    )
                for name in (
                    "mask_vehicle",
                    "mask_pedestrian",
                    "mask_cyclist",
                    "oracle_all",
                    "oracle_vehicle",
                    "pedestrian_region",
                    "cyclist_region",
                    "vehicle_surrogate",
                )
            }

        raw, guard = build_stage3_raw(
            arrays
        )

        metrics = output_metrics(
            raw,
            arrays,
            road_roi,
        )

        for metric in stage3_values:

            value = float(
                metrics[
                    metric
                ]
            )

            if math.isfinite(
                value
            ):
                stage3_values[
                    metric
                ].append(
                    value
                )

    max_stage3_delta = 0.0

    for metric, values in stage3_values.items():

        require(
            values,
            (
                "No finite Stage-3 replay values "
                f"for {metric}."
            ),
        )

        actual = float(
            math.fsum(
                values
            )
            /
            len(
                values
            )
        )

        expected = float(
            stage3_expected[
                metric
            ]
        )

        delta = abs(
            actual
            -
            expected
        )

        require(
            delta
            <=
            PARITY_ABS_TOL,
            (
                "Stage-3 handoff replay mismatch\n"
                f"metric={metric}\n"
                f"actual={actual!r}\n"
                f"expected={expected!r}\n"
                f"delta={delta!r}"
            ),
        )

        max_stage3_delta = max(
            max_stage3_delta,
            delta,
        )

        print(
            metric,
            "| delta=",
            repr(
                delta
            ),
            "| PASS",
        )

    print(
        "Stage-3 max replay delta =",
        repr(
            max_stage3_delta
        ),
    )

    print(
        "Stage-3 raw handoff      = EXACT PASS"
    )

    # --------------------------------------------------------
    # E. FIRST/ONLY frozen Stage-4 development sweep
    # --------------------------------------------------------

    print()
    print("===== E. EXACT 60-CANDIDATE DEVELOPMENT SWEEP =====")

    n_candidates = 60

    accumulator = new_accumulator(
        n_candidates
    )

    for scenario_number, item in enumerate(
        manifest,
        start=1,
    ):

        scenario_id = str(
            item[
                "scenario_id"
            ]
        )

        current_binding = (
            binding_by_scenario[
                scenario_id
            ]
        )

        row_index = int(
            current_binding[
                "t0_map_row_index"
            ]
        )

        current = np.asarray(
            t0_maps[
                row_index
            ],
            dtype=np.float64,
        )

        require(
            current.shape
            ==
            GRID_2D,
            (
                "I_prev shape mismatch for "
                f"{scenario_id}."
            ),
        )

        require(
            np.all(
                np.isfinite(
                    current
                )
            ),
            (
                "I_prev non-finite for "
                f"{scenario_id}."
            ),
        )

        with np.load(
            Path(
                item[
                    "npz_path"
                ]
            ),
            allow_pickle=False,
        ) as loaded:

            arrays = {
                name:
                    np.asarray(
                        loaded[
                            name
                        ],
                        dtype=bool,
                    )
                for name in (
                    "mask_vehicle",
                    "mask_pedestrian",
                    "mask_cyclist",
                    "oracle_all",
                    "oracle_vehicle",
                    "pedestrian_region",
                    "cyclist_region",
                    "vehicle_surrogate",
                )
            }

        raw, guard = build_stage3_raw(
            arrays
        )

        # Only four authoritative smoothing executions per scenario.
        smoothed_by_ms = {}

        for time_ms in (
            50,
            100,
            200,
            400,
        ):

            smoothed_by_ms[
                time_ms
            ] = (
                temporal_smooth_schedule(
                    raw,

                    current_illumination=
                        current,

                    horizons_s=
                        HORIZONS_S,

                    time_constant_s=
                        time_ms
                        /
                        1000.0,
                )
            )

        for candidate_index, candidate in enumerate(
            candidates
        ):

            time_ms = int(
                candidate[
                    "time_constant_ms"
                ]
            )

            metrics, final, limited = (
                evaluate_candidate(
                    raw=
                        raw,

                    guard=
                        guard,

                    current=
                        current,

                    arrays=
                        arrays,

                    road_roi=
                        road_roi,

                    candidate=
                        candidate,

                    smoothed=
                        smoothed_by_ms[
                            time_ms
                        ],
                )
            )

            for metric in ALL_SWEEP_METRICS:

                add_value(
                    accumulator,
                    metric,
                    candidate_index,
                    metrics[
                        metric
                    ],
                )

        if (
            scenario_number
            ==
            1
            or
            scenario_number
            %
            10
            ==
            0
            or
            scenario_number
            ==
            120
        ):

            print(
                (
                    f"scenario "
                    f"{scenario_number:03d}/120 "
                    f"{scenario_id} "
                    f"60/60=PASS"
                ),
                flush=True,
            )

    # --------------------------------------------------------
    # F. Macro aggregate / exact selection
    # --------------------------------------------------------

    print()
    print("===== F. EXACT MACRO AGGREGATION / SELECTION =====")

    score_rows = []
    ranking = []

    for candidate_index, candidate in enumerate(
        candidates
    ):

        metrics = {
            metric:
                candidate_macro(
                    accumulator,
                    metric,
                    candidate_index,
                )
            for metric in
            ALL_SWEEP_METRICS
        }

        key, risk = selection_key(
            metrics,
            candidate,
        )

        row = {
            "candidate_index":
                candidate_index,

            "time_constant_ms":
                int(
                    candidate[
                        "time_constant_ms"
                    ]
                ),

            "time_constant_s":
                float(
                    candidate[
                        "time_constant_s"
                    ]
                ),

            "rho_dim_per_s":
                int(
                    candidate[
                        "rho_dim_per_s"
                    ]
                ),

            "rho_bright_per_s":
                int(
                    candidate[
                        "rho_bright_per_s"
                    ]
                ),

            **metrics,

            "safety_risk_vector": {
                "vehicle_shadow_zone_violation":
                    float(
                        risk[
                            0
                        ]
                    ),

                "glare_risk_exposure":
                    float(
                        risk[
                            1
                        ]
                    ),

                "1-pedestrian_visibility_proxy":
                    float(
                        risk[
                            2
                        ]
                    ),

                "1-cyclist_visibility_proxy":
                    float(
                        risk[
                            3
                        ]
                    ),

                "1-road_illumination_retention":
                    float(
                        risk[
                            4
                        ]
                    ),
            },

            "risk_max":
                float(
                    key[
                        0
                    ]
                ),

            "risk_mean":
                float(
                    key[
                        1
                    ]
                ),

            "selection_key": [
                float(
                    key[
                        0
                    ]
                ),
                float(
                    key[
                        1
                    ]
                ),
                float(
                    key[
                        2
                    ]
                ),
                float(
                    key[
                        3
                    ]
                ),
                float(
                    key[
                        4
                    ]
                ),
                float(
                    key[
                        5
                    ]
                ),
                int(
                    key[
                        6
                    ]
                ),
                int(
                    key[
                        7
                    ]
                ),
                int(
                    key[
                        8
                    ]
                ),
            ],

            "finite_scenario_support": {
                metric:
                    int(
                        accumulator[
                            metric
                        ][
                            "count"
                        ][
                            candidate_index
                        ]
                    )
                for metric in
                ALL_SWEEP_METRICS
            },
        }

        score_rows.append(
            row
        )

        ranking.append(
            (
                key,
                candidate_index,
            )
        )

    ranking.sort(
        key=lambda item:
            item[
                0
            ]
    )

    winner_index = int(
        ranking[
            0
        ][
            1
        ]
    )

    runner_up_index = int(
        ranking[
            1
        ][
            1
        ]
    )

    winner = score_rows[
        winner_index
    ]

    runner_up = score_rows[
        runner_up_index
    ]

    print(
        "candidate scores = 60 / 60"
    )

    print(
        "winner provisional =",
        (
            winner[
                "time_constant_ms"
            ],
            winner[
                "rho_dim_per_s"
            ],
            winner[
                "rho_bright_per_s"
            ],
        ),
    )

    print(
        "winner risk max =",
        repr(
            winner[
                "risk_max"
            ]
        ),
    )

    print(
        "winner risk mean =",
        repr(
            winner[
                "risk_mean"
            ]
        ),
    )

    print(
        "runner-up =",
        (
            runner_up[
                "time_constant_ms"
            ],
            runner_up[
                "rho_dim_per_s"
            ],
            runner_up[
                "rho_bright_per_s"
            ],
        ),
    )

    # --------------------------------------------------------
    # G. Authoritative winner replay
    # --------------------------------------------------------

    print()
    print("===== G. AUTHORITATIVE WINNER REPLAY — 120/120 =====")

    winner_candidate = candidates[
        winner_index
    ]

    replay_values = {
        metric:
            []
        for metric in
        ALL_SWEEP_METRICS
    }

    replay_rows = []

    for scenario_number, item in enumerate(
        manifest,
        start=1,
    ):

        scenario_id = str(
            item[
                "scenario_id"
            ]
        )

        row_index = int(
            binding_by_scenario[
                scenario_id
            ][
                "t0_map_row_index"
            ]
        )

        current = np.asarray(
            t0_maps[
                row_index
            ],
            dtype=np.float64,
        )

        with np.load(
            Path(
                item[
                    "npz_path"
                ]
            ),
            allow_pickle=False,
        ) as loaded:

            arrays = {
                name:
                    np.asarray(
                        loaded[
                            name
                        ],
                        dtype=bool,
                    )
                for name in (
                    "mask_vehicle",
                    "mask_pedestrian",
                    "mask_cyclist",
                    "oracle_all",
                    "oracle_vehicle",
                    "pedestrian_region",
                    "cyclist_region",
                    "vehicle_surrogate",
                )
            }

        raw, guard = build_stage3_raw(
            arrays
        )

        metrics, final, limited = (
            evaluate_candidate(
                raw=
                    raw,

                guard=
                    guard,

                current=
                    current,

                arrays=
                    arrays,

                road_roi=
                    road_roi,

                candidate=
                    winner_candidate,
            )
        )

        replay_row = {
            "scenario_number":
                scenario_number,

            "scenario_id":
                scenario_id,

            "time_constant_ms":
                int(
                    winner_candidate[
                        "time_constant_ms"
                    ]
                ),

            "rho_dim_per_s":
                int(
                    winner_candidate[
                        "rho_dim_per_s"
                    ]
                ),

            "rho_bright_per_s":
                int(
                    winner_candidate[
                        "rho_bright_per_s"
                    ]
                ),
        }

        for metric in ALL_SWEEP_METRICS:

            value = float(
                metrics[
                    metric
                ]
            )

            replay_row[
                metric
            ] = finite_json(
                value
            )

            if math.isfinite(
                value
            ):
                replay_values[
                    metric
                ].append(
                    value
                )

        replay_rows.append(
            replay_row
        )

        if (
            scenario_number
            ==
            1
            or
            scenario_number
            %
            20
            ==
            0
            or
            scenario_number
            ==
            120
        ):

            print(
                (
                    f"winner replay "
                    f"{scenario_number:03d}/120 "
                    f"{scenario_id} PASS"
                ),
                flush=True,
            )

    replay_macro = {}

    maximum_replay_delta = 0.0

    for metric in ALL_SWEEP_METRICS:

        require(
            replay_values[
                metric
            ],
            (
                "Winner replay has no finite "
                f"values for {metric}."
            ),
        )

        actual = float(
            math.fsum(
                replay_values[
                    metric
                ]
            )
            /
            len(
                replay_values[
                    metric
                ]
            )
        )

        expected = float(
            winner[
                metric
            ]
        )

        delta = abs(
            actual
            -
            expected
        )

        require(
            delta
            <=
            PARITY_ABS_TOL,
            (
                "Winner replay aggregate mismatch\n"
                f"metric={metric}\n"
                f"sweep={expected!r}\n"
                f"replay={actual!r}\n"
                f"delta={delta!r}"
            ),
        )

        maximum_replay_delta = max(
            maximum_replay_delta,
            delta,
        )

        replay_macro[
            metric
        ] = actual

    replay_key, replay_risk = selection_key(
        replay_macro,
        winner_candidate,
    )

    require(
        (
            int(
                replay_key[
                    6
                ]
            ),
            int(
                replay_key[
                    7
                ]
            ),
            int(
                replay_key[
                    8
                ]
            ),
        )
        ==
        (
            int(
                winner[
                    "time_constant_ms"
                ]
            ),
            int(
                winner[
                    "rho_dim_per_s"
                ]
            ),
            int(
                winner[
                    "rho_bright_per_s"
                ]
            ),
        ),
        "Winner tuple changed during replay.",
    )

    print(
        "winner authoritative replay = 120 / 120 PASS"
    )

    print(
        "winner max aggregate delta  =",
        repr(
            maximum_replay_delta
        ),
    )

    print(
        "winner authoritative key    =",
        replay_key,
    )

    # --------------------------------------------------------
    # H. Regression
    # --------------------------------------------------------

    print()
    print("===== H. POST-SELECTION REGRESSION =====")

    tests_post = run_regression()

    print(
        "Stage6 regression =",
        f"{tests_post} / {tests_post} PASS",
    )

    # --------------------------------------------------------
    # I. Write-once outputs
    # --------------------------------------------------------

    print()
    print("===== I. WRITE-ONCE STAGE-4 NUMERIC FREEZE =====")

    write_once(
        SCORES,
        canonical_jsonl_bytes(
            score_rows
        ),
    )

    write_once(
        WINNER_SCENARIOS,
        canonical_jsonl_bytes(
            replay_rows
        ),
    )

    scores_sha = file_sha(
        SCORES
    )

    replay_sha = file_sha(
        WINNER_SCENARIOS
    )

    selected = {
        "time_constant_ms":
            int(
                winner[
                    "time_constant_ms"
                ]
            ),

        "time_constant_s":
            float(
                winner[
                    "time_constant_s"
                ]
            ),

        "rho_dim_per_s":
            int(
                winner[
                    "rho_dim_per_s"
                ]
            ),

        "rho_bright_per_s":
            int(
                winner[
                    "rho_bright_per_s"
                ]
            ),

        "metrics":
            replay_macro,

        "safety_risk_vector": {
            "vehicle_shadow_zone_violation":
                float(
                    replay_risk[
                        0
                    ]
                ),

            "glare_risk_exposure":
                float(
                    replay_risk[
                        1
                    ]
                ),

            "1-pedestrian_visibility_proxy":
                float(
                    replay_risk[
                        2
                    ]
                ),

            "1-cyclist_visibility_proxy":
                float(
                    replay_risk[
                        3
                    ]
                ),

            "1-road_illumination_retention":
                float(
                    replay_risk[
                        4
                    ]
                ),
        },

        "selection_key": [
            float(
                replay_key[
                    0
                ]
            ),
            float(
                replay_key[
                    1
                ]
            ),
            float(
                replay_key[
                    2
                ]
            ),
            float(
                replay_key[
                    3
                ]
            ),
            float(
                replay_key[
                    4
                ]
            ),
            float(
                replay_key[
                    5
                ]
            ),
            int(
                replay_key[
                    6
                ]
            ),
            int(
                replay_key[
                    7
                ]
            ),
            int(
                replay_key[
                    8
                ]
            ),
        ],

        "finite_scenario_support": {
            metric:
                len(
                    replay_values[
                        metric
                    ]
                )
            for metric in
            ALL_SWEEP_METRICS
        },
    }

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.8_development_stage4_temporal_rate_selection",

        "status":
            (
                "FROZEN_STAGE4_TEMPORAL_RATE_"
                "BEFORE_PRIMARY_ACCEPTANCE_GATE"
            ),

        "development_only":
            True,

        "formal_outcomes_read":
            False,

        "upstream_policy": {
            "Stage1_gamma":
                "IMMUTABLE",

            "Stage2_margins":
                "IMMUTABLE",

            "Stage3_floors":
                "IMMUTABLE",

            "Stage3_vehicle_floor":
                0.15,

            "Stage3_pedestrian_floor":
                1.0,

            "Stage3_cyclist_floor":
                1.0,
        },

        "I_prev": {
            "semantic":
                (
                    "already-issued current causal "
                    "frozen Part-A t0 actuator map"
                ),

            "maps_path":
                str(
                    T0_MAPS
                ),

            "maps_sha256":
                EXPECTED_SHA[
                    "t0_maps"
                ],

            "scenario_binding_path":
                str(
                    CURRENT_BINDING
                ),

            "scenario_binding_sha256":
                EXPECTED_SHA[
                    "current_binding"
                ],

            "future_information":
                False,

            "all_on_smoke":
                False,
        },

        "candidate_space": {
            "candidate_count":
                60,

            "time_constants_ms":
                [
                    50,
                    100,
                    200,
                    400,
                ],

            "rate_pair_count":
                15,

            "rate_values_per_s":
                [
                    1,
                    2,
                    4,
                    8,
                    16,
                ],

            "rate_constraint":
                "rho_dim_per_s >= rho_bright_per_s",
        },

        "selection_semantics": {
            "scenario_aggregation":
                (
                    "equal development-scenario macro "
                    "over finite authoritative metric values"
                ),

            "safety_risk_vector": [
                "vehicle_shadow_zone_violation",
                "glare_risk_exposure",
                "1-pedestrian_visibility_proxy",
                "1-cyclist_visibility_proxy",
                "1-road_illumination_retention",
            ],

            "selection_key":
                (
                    "(max risk, mean risk, "
                    "temporal mean_absolute_change_rate, "
                    "flicker_change_rate, "
                    "rate_limit_safety_override_fraction, "
                    "over_masking_area, "
                    "time_constant_ms, "
                    "rho_dim_per_s, "
                    "rho_bright_per_s)"
                ),

            "weighted_sum":
                False,

            "epsilon_tolerance":
                None,

            "round_before_comparison":
                False,
        },

        "runtime": {
            "raw":
                (
                    "ClassAwareComposition."
                    "raw_class_aware_illumination"
                ),

            "smoothing":
                "temporal_smooth_schedule",

            "rate_limit":
                "apply_actuation_rate_limit",

            "VRU_guard":
                "ClassAwareComposition.vru_floor_guard",

            "final":
                "RateLimitedSchedule.illumination",
        },

        "selected":
            selected,

        "runner_up": {
            "time_constant_ms":
                int(
                    runner_up[
                        "time_constant_ms"
                ]),

            "rho_dim_per_s":
                int(
                    runner_up[
                        "rho_dim_per_s"
                ]),

            "rho_bright_per_s":
                int(
                    runner_up[
                        "rho_bright_per_s"
                ]),

            "selection_key":
                runner_up[
                    "selection_key"
                ],
        },

        "execution_integrity": {
            "Stage3_raw_handoff_replay":
                "PASS",

            "Stage3_raw_handoff_max_abs_delta":
                max_stage3_delta,

            "winner_authoritative_replay":
                "PASS_120_OF_120",

            "winner_authoritative_max_abs_delta":
                maximum_replay_delta,
        },

        "outputs": {
            "candidate_scores": {
                "path":
                    str(
                        SCORES
                    ),

                "sha256":
                    scores_sha,

                "records":
                    60,
            },

            "winner_scenarios": {
                "path":
                    str(
                        WINNER_SCENARIOS
                    ),

                "sha256":
                    replay_sha,

                "records":
                    120,
            },
        },

        "scientific_boundary": {
            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage3_modified":
                False,

            "Stage4_winner_selected":
                True,

            "post_outcome_retuning_allowed":
                False,

            "primary_acceptance_tested":
                False,

            "NI_bounds_modified":
                False,

            "formal_evaluation":
                False,
        },

        "upstream_seals":
            seals,

        "next":
            (
                "Run frozen primary development acceptance "
                "gate exactly once against original_reactive_ADB; "
                "no parameter modification permitted."
            ),
    }

    write_once(
        FREEZE,
        canonical_json_bytes(
            freeze_payload
        ),
    )

    freeze_sha = file_sha(
        FREEZE
    )

    print(
        "candidate scores SHA =",
        scores_sha,
    )

    print(
        "winner replay SHA    =",
        replay_sha,
    )

    print(
        "Stage-4 freeze SHA   =",
        freeze_sha,
    )

    # --------------------------------------------------------
    # J. Final report
    # --------------------------------------------------------

    available_gib = free_gib()

    require(
        available_gib
        >=
        HARD_RESERVE_GIB,
        (
            "Storage reserve violated: "
            f"{available_gib:.3f} GiB"
        ),
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.8_development_stage4_temporal_rate_selection",

        "status":
            "PASS_STAGE4_TEMPORAL_RATE_SELECTED_FROZEN",

        "candidate_count":
            60,

        "selected":
            selected,

        "runner_up": {
            "time_constant_ms":
                int(
                    runner_up[
                        "time_constant_ms"
                ]),

            "rho_dim_per_s":
                int(
                    runner_up[
                        "rho_dim_per_s"
                ]),

            "rho_bright_per_s":
                int(
                    runner_up[
                        "rho_bright_per_s"
                ]),

            "selection_key":
                runner_up[
                    "selection_key"
                ],
        },

        "execution_integrity": {
            "Stage3_raw_max_abs_delta":
                max_stage3_delta,

            "winner_replay_scenarios":
                120,

            "winner_max_abs_delta":
                maximum_replay_delta,
        },

        "regression": {
            "pre":
                tests_pre,

            "post":
                tests_post,
        },

        "storage": {
            "free_GiB":
                available_gib,

            "reserve_GiB":
                HARD_RESERVE_GIB,

            "reserve_pass":
                True,
        },

        "outputs": {
            "scores": {
                "path":
                    str(
                        SCORES
                    ),

                "sha256":
                    scores_sha,
            },

            "winner_scenarios": {
                "path":
                    str(
                        WINNER_SCENARIOS
                    ),

                "sha256":
                    replay_sha,
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

        "scientific_boundary": {
            "Stage4_winner_selected":
                True,

            "Stage4_frozen":
                True,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,

            "further_development_parameter_tuning":
                False,
        },

        "next":
            "RUN_FROZEN_PRIMARY_DEVELOPMENT_ACCEPTANCE_GATE",
    }

    write_once(
        REPORT,
        canonical_json_bytes(
            report_payload
        ),
    )

    report_sha = file_sha(
        REPORT
    )

    print()
    print("=" * 78)
    print("BLOCK 6.8 STAGE-4 TEMPORAL/RATE SELECTION — FINAL")
    print("=" * 78)

    print(
        "development scenarios        = 120 / 120"
    )

    print(
        "Stage-4 candidates           = 60 / 60"
    )

    print(
        "Stage-1 gamma                = IMMUTABLE"
    )

    print(
        "Stage-2 margins              = IMMUTABLE"
    )

    print(
        "Stage-3 floors               = IMMUTABLE"
    )

    print(
        "selected time constant ms    =",
        selected[
            "time_constant_ms"
        ],
    )

    print(
        "selected rho_dim / s         =",
        selected[
            "rho_dim_per_s"
        ],
    )

    print(
        "selected rho_bright / s      =",
        selected[
            "rho_bright_per_s"
        ],
    )

    print(
        "risk max                     =",
        repr(
            selected[
                "selection_key"
            ][
                0
            ]
        ),
    )

    print(
        "risk mean                    =",
        repr(
            selected[
                "selection_key"
            ][
                1
            ]
        ),
    )

    print(
        "temporal change rate         =",
        repr(
            selected[
                "metrics"
            ][
                "temporal_mean_absolute_change_rate"
            ]
        ),
    )

    print(
        "flicker change rate          =",
        repr(
            selected[
                "metrics"
            ][
                "flicker_change_rate"
            ]
        ),
    )

    print(
        "safety override fraction     =",
        repr(
            selected[
                "metrics"
            ][
                "rate_limit_safety_override_fraction"
            ]
        ),
    )

    print(
        "over-masking area            =",
        repr(
            selected[
                "metrics"
            ][
                "over_masking_area"
            ]
        ),
    )

    print(
        "vehicle violation            =",
        repr(
            selected[
                "metrics"
            ][
                "vehicle_shadow_zone_violation"
            ]
        ),
    )

    print(
        "pedestrian visibility        =",
        repr(
            selected[
                "metrics"
            ][
                "pedestrian_visibility_proxy"
            ]
        ),
    )

    print(
        "cyclist visibility           =",
        repr(
            selected[
                "metrics"
            ][
                "cyclist_visibility_proxy"
            ]
        ),
    )

    print(
        "road illumination retention  =",
        repr(
            selected[
                "metrics"
            ][
                "road_illumination_retention"
            ]
        ),
    )

    print(
        "Stage-3 raw replay           = PASS"
    )

    print(
        "winner authoritative replay  = 120 / 120 PASS"
    )

    print(
        "Stage-4 temporal/rate        = FROZEN"
    )

    print(
        "further parameter tuning     = FORBIDDEN"
    )

    print(
        "primary acceptance gate      = NOT TESTED"
    )

    print(
        "NI bounds                    = IMMUTABLE / UNCHANGED"
    )

    print(
        "formal evaluation            = NO"
    )

    print(
        "Stage6 regression            =",
        f"{tests_post} / {tests_post} PASS",
    )

    print(
        "candidate scores SHA         =",
        scores_sha,
    )

    print(
        "winner replay SHA            =",
        replay_sha,
    )

    print(
        "Stage-4 freeze SHA           =",
        freeze_sha,
    )

    print(
        "report SHA                   =",
        report_sha,
    )

    print(
        "free GiB                     =",
        f"{available_gib:.3f}",
    )

    print(
        "STATUS = "
        "PASS_STAGE4_TEMPORAL_RATE_SELECTED_FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "NEXT = FROZEN PRIMARY DEVELOPMENT ACCEPTANCE GATE"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "BLOCK 6.8 STAGE-4 TEMPORAL/RATE SELECTION = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma          = STILL IMMUTABLE"
    )

    print(
        "Stage-2 margins        = STILL IMMUTABLE"
    )

    print(
        "Stage-3 floors         = STILL IMMUTABLE"
    )

    print(
        "Stage-4 prefreeze      = STILL IMMUTABLE"
    )

    print(
        "primary acceptance     = NOT TESTED"
    )

    print(
        "NI bounds              = UNCHANGED"
    )

    print(
        "formal evaluation      = NO"
    )

    print()
    print(
        "IMPORTANT: if Section E had started, "
        "Stage-4 development outcomes have been observed. "
        "Do NOT change any candidate, I_prev semantic, "
        "metric aggregation, selection key, or Stage1/2/3 value."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

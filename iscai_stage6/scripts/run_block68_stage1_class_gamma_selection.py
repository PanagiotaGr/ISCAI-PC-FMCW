from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import io
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


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

PREREG = (
    S6 / "configs/"
    "block66_part3b_class_aware_policy_preregistration.json"
)

RUNTIME_BIND = (
    S6 / "reports/"
    "block68_development_selection_runtime_bind.json"
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

GRID_BINDING = (
    S6 / "configs/"
    "stage6_frozen_illumination_grid_runtime_binding.json"
)

TIMESTAMP_BINDING = (
    S6 / "configs/"
    "stage6_reactive_future_truth_timestamp_alignment_superseding.json"
)

PREOUTCOME = (
    S6 / "configs/"
    "stage6_reactive_future_truth_scoring_contract.json"
)

NI_DELTA = (
    S6 / "configs/"
    "stage6_exact_noninferiority_deltas.json"
)

ACCEPTANCE = (
    S6 / "configs/"
    "stage6_preformal_primary_acceptance_policy.json"
)

GEOMETRY_BIND = (
    S6 / "reports/"
    "block68_part2b_part2_geometry_identity_bind.json"
)

CURVES = (
    S6 / "artifacts/block68/stage1_gamma/"
    "block68_stage1_class_gamma_objective_curves.npy"
)

PAIR_SUMMARY = (
    S6 / "artifacts/block68/stage1_gamma/"
    "block68_stage1_class_gamma_pair_support.jsonl"
)

FREEZE = (
    S6 / "configs/"
    "stage6_development_stage1_class_gamma_freeze.json"
)

REPORT = (
    S6 / "reports/"
    "block68_stage1_class_gamma_selection.json"
)


EXPECTED = {
    "prereg":
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    "pocc_manifest":
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    "predictive_matches":
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    "ledger":
        "ac2b954571cdd0166bbcfa5bb8b2956785d1e132288dbe4af7fe2d55adeafcaa",

    "grid_binding":
        "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc",

    "timestamp_binding":
        "51a3b56a119467001a9f4617e1e19dc82c548c0a30c79fc9c4a9c1f5ef27bf56",

    "preoutcome":
        "8364c7cdfa4b66d820823c804d45905a207f77f66b53f06a6383e7722b8a9970",

    "ni_delta":
        "a77e548e060c314dd98a7220b0f9ed3aebd5d1978eb70f456b82920e792a29d4",

    "acceptance":
        "e103466b1a6eb62be4e6746029147cbda9671b05130acba5210a3ab7328e6b81",
}

EXPECTED_TESTS = 224
N_MC = 8192

TYPE_VEHICLE = "TYPE_VEHICLE"
TYPE_PEDESTRIAN = "TYPE_PEDESTRIAN"
TYPE_CYCLIST = "TYPE_CYCLIST"

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
    len(HORIZONS_S),
    N_THETA,
    N_RANGE,
)

HARD_RESERVE_GIB = 250.0


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def exact(path: Path, expected: str, label: str):
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
    ) as f:
        for line_number, line in enumerate(
            f,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            value = json.loads(text)

            require(
                isinstance(value, dict),
                (
                    f"Non-object JSONL row "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(value)

    return rows


def canonical_json_bytes(value):
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


def write_once_json(path: Path, value):
    payload = canonical_json_bytes(value)

    if path.exists():
        require(
            path.read_bytes() == payload,
            f"Existing frozen JSON differs: {path}",
        )
        return

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(payload)
    tmp.replace(path)


def write_once_jsonl(path: Path, rows):
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
            f"Existing JSONL differs: {path}",
        )
        return

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(payload)
    tmp.replace(path)


def write_once_npy(path: Path, array):
    buffer = io.BytesIO()

    np.save(
        buffer,
        np.asarray(array),
        allow_pickle=False,
    )

    payload = buffer.getvalue()

    if path.exists():
        require(
            path.read_bytes() == payload,
            f"Existing NPY differs: {path}",
        )
        return

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(payload)
    tmp.replace(path)


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

    for line in process.stdout.splitlines():
        match = __import__("re").search(
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
            theta_half_step
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

    chosen = int(
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
        chosen
        ==
        int(
            candidates[
                tied[0]
            ]
        ),
        "Nearest timestamp tie-break changed.",
    )

    lower = np.flatnonzero(
        timestamps <= target
    )

    upper = np.flatnonzero(
        timestamps >= target
    )

    require(
        len(lower) > 0
        and
        len(upper) > 0,
        "Target horizon is not timestamp-bracketed.",
    )

    return chosen


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
        "Future GT box has non-finite values.",
    )

    require(
        state.length > 0
        and
        state.width > 0
        and
        state.height > 0,
        "Future GT box has non-positive dimensions.",
    )

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
        (
            "Unexpected H0 rotation shape: "
            f"{rotation.shape}"
        ),
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


def iou_curve_from_integer_counts(
    counts_2d,
    oracle_mask,
):
    counts = np.asarray(
        counts_2d,
    )

    oracle = np.asarray(
        oracle_mask,
        dtype=bool,
    )

    require(
        counts.shape == GRID_SHAPE,
        (
            "P_occ count grid shape mismatch: "
            f"{counts.shape}"
        ),
    )

    require(
        oracle.shape == GRID_SHAPE,
        (
            "Oracle grid shape mismatch: "
            f"{oracle.shape}"
        ),
    )

    require(
        np.issubdtype(
            counts.dtype,
            np.integer,
        ),
        (
            "Occupancy counts are not integer-valued: "
            f"{counts.dtype}"
        ),
    )

    minimum = int(
        np.min(counts)
    )

    maximum = int(
        np.max(counts)
    )

    require(
        minimum >= 0
        and
        maximum <= N_MC,
        (
            "Occupancy counts outside [0,8192]: "
            f"min={minimum}, max={maximum}"
        ),
    )

    flat = np.asarray(
        counts,
        dtype=np.int64,
    ).ravel()

    histogram = np.bincount(
        flat,
        minlength=N_MC + 1,
    )

    inside_histogram = np.bincount(
        flat[
            oracle.ravel()
        ],
        minlength=N_MC + 1,
    )

    require(
        len(histogram)
        ==
        N_MC + 1,
        "Unexpected occupancy histogram size.",
    )

    predicted_ge = np.cumsum(
        histogram[::-1],
        dtype=np.int64,
    )[::-1]

    intersection_ge = np.cumsum(
        inside_histogram[::-1],
        dtype=np.int64,
    )[::-1]

    # Candidate k=0..8191:
    # strict mask is count > k == count >= k+1.
    predicted = predicted_ge[
        1:
    ].astype(
        np.float64
    )

    intersection = intersection_ge[
        1:
    ].astype(
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

    curve = np.ones(
        N_MC,
        dtype=np.float64,
    )

    nonempty = (
        union > 0.0
    )

    curve[
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

    require(
        np.all(
            np.isfinite(curve)
        ),
        "Non-finite IoU curve.",
    )

    require(
        np.all(
            curve >= 0.0
        )
        and
        np.all(
            curve <= 1.0
        ),
        "IoU curve outside [0,1].",
    )

    return curve


def ledger_index(rows):
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

            require(
                key not in index,
                (
                    "Duplicate predictive actor in ledger: "
                    f"{key}"
                ),
            )

            index[
                key
            ] = actor

    return index


def choose_k(
    curve,
    *,
    actor_class,
):
    maximum = float(
        np.max(curve)
    )

    winners = np.flatnonzero(
        curve == maximum
    )

    require(
        len(winners) >= 1,
        (
            f"No objective winner for "
            f"{actor_class}."
        ),
    )

    if actor_class == TYPE_VEHICLE:
        # Preregistered protective tie-break:
        # smaller k first.
        chosen = int(
            winners[0]
        )

        tie_break = (
            "smaller_k_first"
        )

    else:
        # Preregistered VRU tie-break:
        # larger k first / less unnecessary dimming.
        chosen = int(
            winners[-1]
        )

        tie_break = (
            "larger_k_first"
        )

    return {
        "k":
            chosen,

        "gamma_equivalent":
            float(
                chosen
                /
                N_MC
            ),

        "objective":
            maximum,

        "exact_maximizer_count":
            int(
                len(winners)
            ),

        "smallest_maximizer_k":
            int(
                winners[0]
            ),

        "largest_maximizer_k":
            int(
                winners[-1]
            ),

        "tie_break":
            tie_break,
    }


def verify_idempotent():
    if not REPORT.is_file():
        return False

    report = read_json(
        REPORT
    )

    if (
        report.get("status")
        !=
        "PASS_STAGE1_CLASS_GAMMA_SELECTED_FROZEN"
    ):
        return False

    for name, path in (
        ("curves", CURVES),
        ("pair_summary", PAIR_SUMMARY),
        ("freeze", FREEZE),
    ):
        require(
            path.is_file(),
            (
                "Frozen Stage-1 output missing: "
                f"{path}"
            ),
        )

        require(
            file_sha(path)
            ==
            report[
                "outputs"
            ][
                name
            ][
                "sha256"
            ],
            (
                "Frozen Stage-1 output changed: "
                f"{name}"
            ),
        )

    print(
        "existing Stage-1 gamma freeze = EXACT PASS"
    )

    print(
        "No P_occ recomputation performed."
    )

    print(
        "STATUS = "
        "PASS_STAGE1_CLASS_GAMMA_SELECTED_FROZEN"
    )

    return True


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "DEVELOPMENT STAGE-1 CLASS-GAMMA SELECTION"
    )
    print(
        "FIRST FROZEN P_OCC ARRAY ACCESS"
    )
    print(
        "VARY k_c ONLY — NO MARGIN/FLOOR/TEMPORAL/RATE TUNING"
    )
    print(
        "============================================================"
    )

    if verify_idempotent():
        return

    # --------------------------------------------------------
    # A. Exact frozen boundary
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT PRE-SELECTION BOUNDARY ====="
    )

    seals = {}

    for key, path in (
        ("prereg", PREREG),
        ("pocc_manifest", POCC_MANIFEST),
        ("predictive_matches", PREDICTIVE_MATCHES),
        ("ledger", LEDGER),
        ("grid_binding", GRID_BINDING),
        ("timestamp_binding", TIMESTAMP_BINDING),
        ("preoutcome", PREOUTCOME),
        ("ni_delta", NI_DELTA),
        ("acceptance", ACCEPTANCE),
    ):
        seals[key] = exact(
            path,
            EXPECTED[key],
            key,
        )

        print(
            f"{key:21s} = EXACT PASS"
        )

    runtime_bind = read_json(
        RUNTIME_BIND
    )

    require(
        runtime_bind.get(
            "status"
        )
        ==
        "PASS_DEVELOPMENT_SELECTION_RUNTIME_BOUND",
        (
            "Development-selection runtime bind "
            "is not authoritative PASS."
        ),
    )

    geometry_bind = read_json(
        GEOMETRY_BIND
    )

    require(
        geometry_bind.get(
            "status"
        )
        ==
        "PASS_FUTURE_EVALUATOR_GEOMETRY_IDENTITY_BOUND",
        (
            "Geometry/identity bind is not PASS."
        ),
    )

    geometry_path = Path(
        geometry_bind[
            "geometry"
        ][
            "module"
        ]
    )

    require(
        file_sha(
            geometry_path
        )
        ==
        geometry_bind[
            "geometry"
        ][
            "module_sha256"
        ],
        "Geometry module changed.",
    )

    prereg = read_json(
        PREREG
    )

    protocol = (
        prereg[
            "development_selection_protocol"
        ][
            "stage_1_class_gamma"
        ]
    )

    threshold_contract = (
        prereg[
            "class_threshold_parameterization"
        ]
    )

    require(
        protocol[
            "vary"
        ]
        ==
        "k_c only",
        "Stage-1 protocol no longer varies k_c only.",
    )

    require(
        threshold_contract[
            "candidate_rule"
        ][
            "k_min"
        ]
        ==
        0,
        "k_min changed.",
    )

    require(
        threshold_contract[
            "candidate_rule"
        ][
            "k_max"
        ]
        ==
        8191,
        "k_max changed.",
    )

    require(
        threshold_contract[
            "candidate_rule"
        ][
            "step"
        ]
        ==
        1,
        "k step changed.",
    )

    require(
        threshold_contract[
            "strict_comparison"
        ]
        ==
        ">",
        "Threshold comparison changed.",
    )

    require(
        threshold_contract[
            "equivalent_gamma"
        ]
        ==
        "gamma_c = k_c / 8192",
        "Equivalent-gamma rule changed.",
    )

    print(
        "runtime bind             = PASS"
    )
    print(
        "geometry bind            = PASS"
    )
    print(
        "candidate k              = 0..8191 EXACT"
    )
    print(
        "threshold semantics      = count > k"
    )
    print(
        "equivalent gamma         = k / 8192"
    )
    print(
        "NI deltas                = IMMUTABLE"
    )
    print(
        "formal evaluation        = NO"
    )

    tests_pre = regression()

    print(
        "Stage6 pre regression    =",
        f"{tests_pre} / {tests_pre} PASS",
    )

    free_before = free_gib()

    require(
        free_before >= HARD_RESERVE_GIB,
        (
            "Disk reserve failed: "
            f"{free_before:.3f} GiB"
        ),
    )

    # --------------------------------------------------------
    # B. Population joins before opening count payloads
    # --------------------------------------------------------

    print()
    print(
        "===== B. FROZEN DEVELOPMENT POPULATION JOIN ====="
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

    manifest_rows.sort(
        key=lambda row: (
            int(
                row[
                    "cohort_index"
                ]
            ),
            int(
                row.get(
                    "actor_rank",
                    0,
                )
            ),
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
        )
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
            f"Duplicate predictive match: {key}",
        )

        match_index[
            key
        ] = row

    actor_identity = ledger_index(
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
        "Development cohort must contain 120 scenarios.",
    )

    class_counts = Counter(
        str(
            row[
                "object_type"
            ]
        )
        for row in manifest_rows
    )

    expected_class_counts = {
        key:
            int(value)
        for key, value in
        prereg[
            "development_population"
        ][
            "class_counts"
        ].items()
    }

    require(
        dict(class_counts)
        ==
        expected_class_counts,
        (
            "Predictive P_occ class population "
            "does not equal preregistration.\n"
            f"actual={dict(class_counts)}\n"
            f"expected={expected_class_counts}"
        ),
    )

    for row in manifest_rows:
        require(
            int(
                row[
                    "sample_count"
                ]
            )
            ==
            N_MC,
            "P_occ sample_count != 8192.",
        )

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
            (
                "P_occ actor missing predictive match: "
                f"{key}"
            ),
        )

        require(
            key in actor_identity,
            (
                "P_occ actor missing frozen track identity: "
                f"{key}"
            ),
        )

    print(
        "P_occ actors             =",
        len(
            manifest_rows
        ),
    )
    print(
        "vehicle actors           =",
        class_counts[
            TYPE_VEHICLE
        ],
    )
    print(
        "pedestrian actors        =",
        class_counts[
            TYPE_PEDESTRIAN
        ],
    )
    print(
        "cyclist actors           =",
        class_counts[
            TYPE_CYCLIST
        ],
    )
    print(
        "predictive-match join    = EXACT PASS"
    )
    print(
        "track-identity join      = EXACT PASS"
    )
    print(
        "future validity selector = NO"
    )

    # --------------------------------------------------------
    # C. FIRST P_occ access + exact objective curves
    # --------------------------------------------------------

    print()
    print(
        "===== C. FIRST FROZEN P_OCC ARRAY ACCESS / IoU SWEEP ====="
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

    objective_sums = {
        actor_class:
            np.zeros(
                N_MC,
                dtype=np.float64,
            )
        for actor_class in CLASS_ORDER
    }

    scored_pairs = Counter()
    invalid_future_pairs = Counter()
    valid_empty_oracle_pairs = Counter()

    pair_rows = []

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

    processed_actors = 0
    verified_count_files = 0

    for scenario_rank, sid in enumerate(
        sorted(
            by_scenario,
            key=lambda value:
                int(
                    by_scenario[value][0][
                        "cohort_index"
                    ]
                ),
        ),
        start=1,
    ):
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
            sid,
            (
                "Scenario ID mismatch after WOMD load: "
                f"{sid}"
            ),
        )

        anchor_index = int(
            scenario.current_time_index
        )

        adapted = adapt_causal_womd_scenario(
            scenario
        )

        require(
            adapted.anchor_index
            ==
            anchor_index,
            (
                "Adapted anchor mismatch: "
                f"{sid}"
            ),
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

            match = match_index[
                key
            ]

            identity = actor_identity[
                key
            ]

            actor_class = str(
                manifest[
                    "object_type"
                ]
            )

            require(
                actor_class
                in
                CLASS_ORDER,
                (
                    "Unexpected actor class: "
                    f"{actor_class}"
                ),
            )

            require(
                str(
                    match[
                        "object_type"
                    ]
                )
                ==
                actor_class,
                "Manifest/match class mismatch.",
            )

            require(
                str(
                    identity[
                        "object_type"
                    ]
                )
                ==
                actor_class,
                "Manifest/ledger class mismatch.",
            )

            track_index = int(
                identity[
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
                    "Invalid frozen track_index: "
                    f"{key}"
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
                    identity[
                        "track_id"
                    ]
                ),
                (
                    "WOMD track_id mismatch: "
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
                    "WOMD object class mismatch: "
                    f"{key}"
                ),
            )

            counts_path = Path(
                manifest[
                    "occupancy_counts_path"
                ]
            )

            require(
                counts_path.is_file(),
                (
                    "Frozen occupancy-count file missing: "
                    f"{counts_path}"
                ),
            )

            expected_counts_sha = str(
                manifest[
                    "occupancy_counts_sha256"
                ]
            )

            actual_counts_sha = file_sha(
                counts_path
            )

            require(
                actual_counts_sha
                ==
                expected_counts_sha,
                (
                    "Occupancy-count SHA mismatch\n"
                    f"path={counts_path}\n"
                    f"expected={expected_counts_sha}\n"
                    f"actual={actual_counts_sha}"
                ),
            )

            verified_count_files += 1

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
                    "P_occ count tensor shape mismatch: "
                    f"{counts.shape}"
                ),
            )

            for h, horizon in enumerate(
                HORIZONS_S
            ):
                future_index = future_indices[
                    horizon
                ]

                require(
                    future_index
                    <
                    len(
                        track.states
                    ),
                    (
                        "Track state array does not cover "
                        f"horizon={horizon}: {key}"
                    ),
                )

                state = track.states[
                    future_index
                ]

                if not bool(
                    state.valid
                ):
                    # Actor population remains frozen.
                    # This actor/horizon has no evaluator
                    # reference and is NA for the objective.
                    invalid_future_pairs[
                        actor_class
                    ] += 1

                    pair_rows.append(
                        {
                            "scenario_id":
                                sid,

                            "prediction_id":
                                str(
                                    manifest[
                                        "prediction_id"
                                    ]
                                ),

                            "cohort_index":
                                int(
                                    manifest[
                                        "cohort_index"
                                    ]
                                ),

                            "actor_box_index":
                                int(
                                    manifest[
                                        "actor_box_index"
                                    ]
                                ),

                            "track_index":
                                track_index,

                            "object_type":
                                actor_class,

                            "horizon_s":
                                horizon,

                            "future_state_valid":
                                False,

                            "objective_support":
                                "NA",

                            "oracle_cell_count":
                                None,
                        }
                    )

                    continue

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

                oracle = rasterize_projected_full_box(
                    projected,
                    theta_centers=theta_centers,
                    range_centers=range_centers,
                    theta_half_step=theta_half_step,
                    range_half_step=range_half_step,
                )

                oracle_cells = int(
                    np.count_nonzero(
                        oracle
                    )
                )

                if oracle_cells == 0:
                    valid_empty_oracle_pairs[
                        actor_class
                    ] += 1

                curve = (
                    iou_curve_from_integer_counts(
                        counts[h],
                        oracle,
                    )
                )

                objective_sums[
                    actor_class
                ] += curve

                scored_pairs[
                    actor_class
                ] += 1

                pair_rows.append(
                    {
                        "scenario_id":
                            sid,

                        "prediction_id":
                            str(
                                manifest[
                                    "prediction_id"
                                ]
                            ),

                        "cohort_index":
                            int(
                                manifest[
                                    "cohort_index"
                                ]
                            ),

                        "actor_box_index":
                            int(
                                manifest[
                                    "actor_box_index"
                                ]
                            ),

                        "track_index":
                            track_index,

                        "object_type":
                            actor_class,

                        "horizon_s":
                            horizon,

                        "future_state_valid":
                            True,

                        "objective_support":
                            "SCORED",

                        "oracle_cell_count":
                            oracle_cells,
                    }
                )

            processed_actors += 1

            if (
                processed_actors % 100
                ==
                0
            ):
                print(
                    "processed predictive actors =",
                    f"{processed_actors} / "
                    f"{len(manifest_rows)}",
                )

        if scenario_rank % 20 == 0:
            print(
                "development scenarios       =",
                f"{scenario_rank} / 120",
            )

    require(
        processed_actors
        ==
        len(
            manifest_rows
        ),
        "Not all predictive actors were processed.",
    )

    require(
        verified_count_files
        ==
        len(
            manifest_rows
        ),
        "Not all occupancy-count SHAs were verified.",
    )

    # --------------------------------------------------------
    # D. Exact classwise objective / tie-break
    # --------------------------------------------------------

    print()
    print(
        "===== D. EXACT PREREGISTERED CLASSWISE SELECTION ====="
    )

    objective_curves = np.zeros(
        (
            len(CLASS_ORDER),
            N_MC,
        ),
        dtype=np.float64,
    )

    selected = {}

    for class_index, actor_class in enumerate(
        CLASS_ORDER
    ):
        n_pairs = int(
            scored_pairs[
                actor_class
            ]
        )

        require(
            n_pairs > 0,
            (
                "No valid actor-horizon objective "
                f"support for {actor_class}."
            ),
        )

        curve = (
            objective_sums[
                actor_class
            ]
            /
            float(
                n_pairs
            )
        )

        require(
            np.all(
                np.isfinite(curve)
            ),
            (
                "Non-finite class objective curve: "
                f"{actor_class}"
            ),
        )

        objective_curves[
            class_index
        ] = curve

        chosen = choose_k(
            curve,
            actor_class=actor_class,
        )

        chosen[
            "scored_actor_horizon_pairs"
        ] = n_pairs

        chosen[
            "invalid_future_actor_horizon_pairs"
        ] = int(
            invalid_future_pairs[
                actor_class
            ]
        )

        chosen[
            "valid_empty_oracle_pairs"
        ] = int(
            valid_empty_oracle_pairs[
                actor_class
            ]
        )

        selected[
            actor_class
        ] = chosen

        print()
        print(
            actor_class
        )

        print(
            "  scored pairs          =",
            n_pairs,
        )

        print(
            "  invalid future / NA   =",
            invalid_future_pairs[
                actor_class
            ],
        )

        print(
            "  empty valid oracle    =",
            valid_empty_oracle_pairs[
                actor_class
            ],
        )

        print(
            "  selected k            =",
            chosen[
                "k"
            ],
        )

        print(
            "  equivalent gamma      =",
            chosen[
                "gamma_equivalent"
            ],
        )

        print(
            "  macro actor×h IoU     =",
            chosen[
                "objective"
            ],
        )

        print(
            "  exact maximizer count =",
            chosen[
                "exact_maximizer_count"
            ],
        )

        print(
            "  tie break             =",
            chosen[
                "tie_break"
            ],
        )

    # --------------------------------------------------------
    # E. Post-computation regression
    # --------------------------------------------------------

    print()
    print(
        "===== E. POST-SELECTION REGRESSION ====="
    )

    tests_post = regression()

    print(
        "Stage6 regression =",
        f"{tests_post} / {tests_post} PASS",
    )

    # --------------------------------------------------------
    # F. Freeze selected Stage-1 policy
    # --------------------------------------------------------

    print()
    print(
        "===== F. WRITE-ONCE STAGE-1 GAMMA FREEZE ====="
    )

    pair_rows.sort(
        key=lambda row: (
            row[
                "cohort_index"
            ],
            row[
                "actor_box_index"
            ],
            row[
                "prediction_id"
            ],
            row[
                "horizon_s"
            ],
        )
    )

    write_once_npy(
        CURVES,
        objective_curves,
    )

    write_once_jsonl(
        PAIR_SUMMARY,
        pair_rows,
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.8_development_stage1_class_gamma",

        "status":
            "FROZEN_STAGE1_CLASS_GAMMA_BEFORE_STAGE2_MARGIN_SELECTION",

        "development_only":
            True,

        "formal_outcomes_read":
            False,

        "candidate_space": {
            "k_min":
                0,

            "k_max":
                8191,

            "step":
                1,

            "sample_count":
                N_MC,

            "strict_threshold":
                "count > k",

            "equivalent_gamma":
                "k / 8192",
        },

        "objective": (
            "macro mean per valid actor-horizon "
            "full-box mask IoU against constructed "
            "future evaluator reference"
        ),

        "missing_future_semantics": (
            "actor remains in frozen population; "
            "invalid future state at a horizon is "
            "NA for that actor-horizon objective"
        ),

        "valid_empty_oracle_semantics": (
            "valid future GT outside actuator support "
            "remains a scored empty oracle; "
            "empty/empty IoU = 1"
        ),

        "selected":
            selected,

        "class_order_in_curve_file":
            list(
                CLASS_ORDER
            ),

        "stage1_only": {
            "gamma_selected":
                True,

            "additional_class_margin_selected":
                False,

            "class_floor_selected":
                False,

            "temporal_smoothing_selected":
                False,

            "actuation_rate_selected":
                False,
        },

        "immutable_after_this_freeze": [
            "TYPE_VEHICLE k/gamma",
            "TYPE_PEDESTRIAN k/gamma",
            "TYPE_CYCLIST k/gamma",
        ],

        "upstream_hashes": {
            **seals,

            "runtime_bind":
                file_sha(
                    RUNTIME_BIND
                ),

            "geometry_bind":
                file_sha(
                    GEOMETRY_BIND
                ),

            "geometry_module":
                file_sha(
                    geometry_path
                ),

            "cohort":
                file_sha(
                    COHORT
                ),
        },

        "acceptance_boundary": {
            "NI_deltas_modified":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },
    }

    write_once_json(
        FREEZE,
        freeze_payload,
    )

    curves_sha = file_sha(
        CURVES
    )

    pair_sha = file_sha(
        PAIR_SUMMARY
    )

    freeze_sha = file_sha(
        FREEZE
    )

    print(
        "objective curves SHA =",
        curves_sha,
    )

    print(
        "pair support SHA      =",
        pair_sha,
    )

    print(
        "Stage-1 freeze SHA    =",
        freeze_sha,
    )

    # --------------------------------------------------------
    # G. Final report
    # --------------------------------------------------------

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
            "6.8_development_stage1_class_gamma",

        "status":
            "PASS_STAGE1_CLASS_GAMMA_SELECTED_FROZEN",

        "development_population": {
            "scenarios":
                120,

            "predictive_actors":
                len(
                    manifest_rows
                ),

            "class_counts":
                dict(
                    class_counts
                ),

            "actor_population_selected_using_future":
                False,
        },

        "selected":
            selected,

        "scientific_boundary": {
            "P_occ_arrays_opened":
                True,

            "predictive_development_objective_computed":
                True,

            "predictive_primary_acceptance_metrics_computed":
                False,

            "NI_deltas_modified":
                False,

            "additional_margin_selection":
                False,

            "floor_selection":
                False,

            "temporal_rate_selection":
                False,

            "formal_content_opened":
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
            "curves": {
                "path":
                    str(
                        CURVES
                    ),

                "sha256":
                    curves_sha,
            },

            "pair_summary": {
                "path":
                    str(
                        PAIR_SUMMARY
                    ),

                "sha256":
                    pair_sha,
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

        "next": (
            "Stage-2 development-only class-margin "
            "selection with Stage-1 k/gamma immutable"
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
        "BLOCK 6.8 STAGE-1 CLASS-GAMMA SELECTION — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "development scenarios        = 120 / 120"
    )

    print(
        "predictive actors            =",
        len(
            manifest_rows
        ),
    )

    print(
        "P_occ arrays                 = OPENED, FROZEN INPUT"
    )

    print(
        "objective                    = MACRO ACTOR×HORIZON FULL-BOX IoU"
    )

    print(
        "vehicle selected k           =",
        selected[
            TYPE_VEHICLE
        ][
            "k"
        ],
    )

    print(
        "vehicle gamma                =",
        selected[
            TYPE_VEHICLE
        ][
            "gamma_equivalent"
        ],
    )

    print(
        "pedestrian selected k        =",
        selected[
            TYPE_PEDESTRIAN
        ][
            "k"
        ],
    )

    print(
        "pedestrian gamma             =",
        selected[
            TYPE_PEDESTRIAN
        ][
            "gamma_equivalent"
        ],
    )

    print(
        "cyclist selected k           =",
        selected[
            TYPE_CYCLIST
        ][
            "k"
        ],
    )

    print(
        "cyclist gamma                =",
        selected[
            TYPE_CYCLIST
        ][
            "gamma_equivalent"
        ],
    )

    print(
        "Stage-1 gamma                = FROZEN"
    )

    print(
        "margin selection             = NOT EXECUTED"
    )

    print(
        "floor selection              = NOT EXECUTED"
    )

    print(
        "temporal/rate selection      = NOT EXECUTED"
    )

    print(
        "primary acceptance gate      = NOT TESTED"
    )

    print(
        "NI deltas                    = IMMUTABLE / UNCHANGED"
    )

    print(
        "formal evaluation            = NO"
    )

    print(
        "Stage6 regression            =",
        f"{tests_post} / {tests_post} PASS",
    )

    print(
        "free GiB                     =",
        f"{free_after:.3f}",
    )

    print(
        "STATUS = "
        "PASS_STAGE1_CLASS_GAMMA_SELECTED_FROZEN"
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
        "BLOCK 6.8 STAGE-1 CLASS-GAMMA SELECTION = BLOCKED"
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
        "IMPORTANT:"
    )

    print(
        "P_occ arrays may have been opened "
        "for development scoring."
    )

    print(
        "Do NOT change NI bounds, timestamp alignment, "
        "metric semantics or preregistration."
    )

    print(
        "Stage-1 gamma freeze = NOT COMPLETE "
        "unless final PASS report exists."
    )

    print(
        "margin/floor/temporal/rate tuning = NOT EXECUTED "
        "by this runner"
    )

    print(
        "primary acceptance gate = NOT TESTED "
        "by this runner"
    )

    print(
        "formal evaluation = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

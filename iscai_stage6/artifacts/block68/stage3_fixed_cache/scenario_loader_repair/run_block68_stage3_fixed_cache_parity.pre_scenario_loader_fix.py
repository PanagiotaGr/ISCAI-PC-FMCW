from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import traceback

import numpy as np

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
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
    FractionalBoxRegion,
    project_box_to_headlamp,
    project_region_to_headlamp,
)

import iscai_stage6.adb.geometry as geometry_module

from iscai_stage6.adb.metric_semantics import (
    mask_iou,
    over_masking_area,
)

from iscai_stage6.adb.womd_geometry import (
    build_causal_adb_actor_boxes,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

ART = (
    S6
    / "artifacts/block68/stage3_fixed_cache"
)

CACHE = (
    ART
    / "scenarios"
)

REPORT = (
    S6
    / "reports/block68_stage3_fixed_cache_parity.json"
)

MANIFEST = (
    ART
    / "stage3_fixed_cache_manifest.jsonl"
)

FREEZE = (
    ART
    / "stage3_fixed_cache_freeze.json"
)


# ============================================================
# FROZEN INPUTS
# ============================================================

PREREG = (
    S6
    / "configs/block66_part3b_class_aware_policy_preregistration.json"
)

STAGE1 = (
    S6
    / "configs/stage6_development_stage1_class_gamma_freeze.json"
)

STAGE2 = (
    S6
    / "configs/stage6_development_stage2_class_margin_freeze.json"
)

STAGE3_EXEC = (
    S6
    / "configs/stage6_development_stage3_floor_execution_binding.json"
)

STAGE3_SCORING = (
    S6
    / "configs/stage6_development_stage3_floor_scoring_semantics.json"
)

GRID_BINDING = (
    S6
    / "configs/stage6_frozen_illumination_grid_runtime_binding.json"
)

PROVENANCE_BINDING = (
    S6
    / "configs/stage6_future_gt_evaluator_provenance_binding.json"
)

REACTIVE_ONLY_RUNNER = (
    S6
    / "scripts/run_block68_reactive_only_ni_delta_freeze.py"
)

POCC_MANIFEST = (
    S6
    / "artifacts/block66/"
    "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

PREDICTIVE_MATCHES = (
    S6
    / "artifacts/block66/"
    "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

POSTERIOR = (
    S6
    / "artifacts/block66/"
    "block66_development_identity_safe_gaussian_posterior.jsonl"
)

COHORT = (
    S6
    / "artifacts/block66/"
    "block66_class_aware_development_cohort_120.jsonl"
)

CLASS_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/class_aware_policy.py"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/metric_semantics.py"
)


EXPECTED_SHA = {
    "prereg":
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    "stage1":
        "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87",

    "stage2":
        "b061541f0563181cd07a91107a211b3519c6fdcc2e08243171db5b3bd9393ae8",

    "stage3_execution":
        "28cc6e986d89ac449cce4b5180f3879567e0f6a8bd77a2bd6054fcb262cba091",

    "stage3_scoring":
        "a8bf30b29056562eb303f287acfcca6d8aee011f26a353897f8d18b47325e1b3",

    "grid_binding":
        "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc",

    "pocc_manifest":
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    "predictive_matches":
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    "posterior":
        "f022471fbf86c3121106011bfd7f035c5d895f2c5aeeaa2fa3a20c04b3439acd",

    "class_runtime":
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    "metric_runtime":
        "bf8358b5a7edfe40ffac2718ca7cd5af6e7b5fdb3ff8a2823caad6d0f40f057a",
}


# ============================================================
# EXACT FROZEN SCIENTIFIC NUMERICS
# ============================================================

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

N_MC = 8192

THETA_CENTERS = np.linspace(
    math.radians(-25.0),
    math.radians(25.0),
    501,
    dtype=np.float64,
)

RANGE_CENTERS = np.linspace(
    0.0,
    150.0,
    301,
    dtype=np.float64,
)

THETA_STEP = float(
    THETA_CENTERS[1]
    -
    THETA_CENTERS[0]
)

GRID_SHAPE = (
    4,
    501,
    301,
)


K_BY_CLASS = {
    TYPE_VEHICLE:
        980,

    TYPE_PEDESTRIAN:
        499,

    TYPE_CYCLIST:
        557,
}


STAGE2_PARAMS = {
    TYPE_VEHICLE: {
        "base_margin_m":
            0.0,

        "uncertainty_multiplier":
            0.5,

        "motion_multiplier":
            0.05,
    },

    TYPE_PEDESTRIAN: {
        "base_margin_m":
            0.0,

        "uncertainty_multiplier":
            0.5,

        "motion_multiplier":
            0.0,
    },

    TYPE_CYCLIST: {
        "base_margin_m":
            0.0,

        "uncertainty_multiplier":
            0.5,

        "motion_multiplier":
            0.25,
    },
}


# These are not new thresholds.
# They are the already-frozen Stage-1/Stage-2 development outcomes
# and are used solely as deterministic replay parity targets.

STAGE1_TARGET = {
    TYPE_VEHICLE: {
        "mean_iou":
            0.442779087781594,

        "scored":
            2819,

        "invalid_future":
            57,

        "empty_valid_oracle":
            40,
    },

    TYPE_PEDESTRIAN: {
        "mean_iou":
            0.25486530932489326,

        "scored":
            412,

        "invalid_future":
            28,

        "empty_valid_oracle":
            6,
    },

    TYPE_CYCLIST: {
        "mean_iou":
            0.2451696448623915,

        "scored":
            182,

        "invalid_future":
            10,

        "empty_valid_oracle":
            0,
    },
}


STAGE2_TARGET = {
    TYPE_VEHICLE: {
        "mean_iou":
            0.41918095515651327,

        "mean_overmask":
            0.003226307053031618,
    },

    TYPE_PEDESTRIAN: {
        "mean_iou":
            0.22030627084752338,

        "mean_overmask":
            0.0009210846442456835,
    },

    TYPE_CYCLIST: {
        "mean_iou":
            0.2229309135285922,

        "mean_overmask":
            0.0016067678450553899,
    },
}


EXPECTED_ACTORS = {
    TYPE_VEHICLE:
        719,

    TYPE_PEDESTRIAN:
        110,

    TYPE_CYCLIST:
        48,
}


REPLAY_ABS_TOL = 1.0e-12

EXPECTED_TESTS = 224

CONTRACT_ID = (
    "block68_stage3_fixed_mask_and_evaluator_cache_v1"
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
) -> str:
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


def exact(
    path: Path,
    expected: str,
    label: str,
) -> str:
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
        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Bad JSONL {path}:{line_number}"
                ) from exc

    return rows


def canonical_bytes(
    value,
) -> bytes:
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


def write_json_atomic(
    path: Path,
    value,
):
    payload = canonical_bytes(
        value
    )

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


def write_text_atomic(
    path: Path,
    text: str,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
        text,
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def flatten_strings(
    value,
):
    result = []

    if isinstance(
        value,
        dict,
    ):
        for item in value.values():
            result.extend(
                flatten_strings(
                    item
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for item in value:
            result.extend(
                flatten_strings(
                    item
                )
            )

    elif isinstance(
        value,
        str,
    ):
        result.append(
            value
        )

    return result


def read_tfrecord_at_offset(
    path: Path,
    offset: int,
):
    require(
        path.is_file(),
        f"Motion TFRecord missing: {path}",
    )

    with path.open(
        "rb"
    ) as stream:
        stream.seek(
            int(
                offset
            )
        )

        length_bytes = stream.read(
            8
        )

        require(
            len(
                length_bytes
            )
            ==
            8,
            (
                "Invalid TFRecord header at "
                f"{path}:{offset}"
            ),
        )

        length = struct.unpack(
            "<Q",
            length_bytes,
        )[0]

        require(
            len(
                stream.read(
                    4
                )
            )
            ==
            4,
            "Missing TFRecord length CRC.",
        )

        payload = stream.read(
            length
        )

        require(
            len(
                payload
            )
            ==
            length,
            "Incomplete TFRecord payload.",
        )

        require(
            len(
                stream.read(
                    4
                )
            )
            ==
            4,
            "Missing TFRecord payload CRC.",
        )

    scenario = (
        scenario_pb2
        .Scenario()
    )

    scenario.ParseFromString(
        payload
    )

    return (
        scenario,
        int(
            length
        ),
    )


def angle_diff(
    a,
    b,
):
    return math.atan2(
        math.sin(
            float(a)
            -
            float(b)
        ),
        math.cos(
            float(a)
            -
            float(b)
        ),
    )


def box_center(
    box,
):
    if hasattr(
        box,
        "center_xyz",
    ):
        return tuple(
            float(x)
            for x in
            box.center_xyz
        )

    value = np.asarray(
        box.center,
        dtype=np.float64,
    ).reshape(
        -1
    )

    return tuple(
        float(x)
        for x in
        value[:3]
    )


def verify_current_box_match(
    causal,
    record,
):
    source = (
        record[
            "current_box_H0"
        ]
    )

    target_center = np.asarray(
        source[
            "center_xyz_H0_m"
        ],
        dtype=np.float64,
    )

    actual_center = np.asarray(
        box_center(
            causal.box
        ),
        dtype=np.float64,
    )

    require(
        np.max(
            np.abs(
                target_center
                -
                actual_center
            )
        )
        <=
        1.0e-8,
        (
            "actor_box_index -> causal box "
            "center mismatch"
        ),
    )

    for field_record, field_box in (
        (
            "length_m",
            "length_m",
        ),
        (
            "width_m",
            "width_m",
        ),
        (
            "height_m",
            "height_m",
        ),
    ):
        require(
            abs(
                float(
                    source[
                        field_record
                    ]
                )
                -
                float(
                    getattr(
                        causal.box,
                        field_box,
                    )
                )
            )
            <=
            1.0e-8,
            (
                "actor_box_index -> causal box "
                f"{field_record} mismatch"
            ),
        )

    require(
        abs(
            angle_diff(
                source[
                    "yaw_rad"
                ],
                causal.box.yaw_rad,
            )
        )
        <=
        1.0e-8,
        (
            "actor_box_index -> causal box "
            "yaw mismatch"
        ),
    )


def future_indices(
    scenario,
):
    anchor = int(
        scenario.current_time_index
    )

    timestamps = np.asarray(
        scenario.timestamps_seconds,
        dtype=np.float64,
    )

    anchor_t = float(
        timestamps[
            anchor
        ]
    )

    result = []

    for horizon in HORIZONS_S:
        target = (
            anchor_t
            +
            float(
                horizon
            )
        )

        candidates = range(
            anchor
            +
            1,
            len(
                timestamps
            ),
        )

        selected = min(
            candidates,
            key=lambda index: (
                abs(
                    float(
                        timestamps[
                            index
                        ]
                    )
                    -
                    target
                ),
                int(
                    index
                ),
            ),
        )

        result.append(
            int(
                selected
            )
        )

    return tuple(
        result
    )


def resolve_future_orientation_source():
    validator = getattr(
        geometry_module,
        "validate_controller_provenance",
        None,
    )

    require(
        callable(
            validator
        ),
        (
            "geometry.validate_controller_provenance "
            "not available"
        ),
    )

    candidates = []

    if PROVENANCE_BINDING.is_file():
        binding = read_json(
            PROVENANCE_BINDING
        )

        candidates.extend(
            flatten_strings(
                binding
            )
        )

    if REACTIVE_ONLY_RUNNER.is_file():
        text = REACTIVE_ONLY_RUNNER.read_text(
            encoding="utf-8"
        )

        # Literal strings only.
        candidates.extend(
            re.findall(
                r"""["']([^"']+)["']""",
                text,
            )
        )

    filtered = []

    for value in candidates:
        text = str(
            value
        )

        lower = text.lower()

        if not any(
            token in lower
            for token in (
                "future",
                "gt",
                "evaluator",
                "heading",
                "yaw",
                "truth",
            )
        ):
            continue

        if text not in filtered:
            filtered.append(
                text
            )

    passing = []

    for candidate in filtered:
        try:
            validator(
                state_source=
                    "WOMD_future_GT_evaluator_only",

                orientation_source=
                    candidate,

                controller_path=
                    False,
            )

        except Exception:
            continue

        passing.append(
            candidate
        )

    require(
        len(
            passing
        )
        ==
        1,
        (
            "Could not uniquely recover frozen "
            "future-GT evaluator orientation source.\n"
            f"passing={passing}"
        ),
    )

    return passing[0]


def raster_projected(
    projected,
):
    center = float(
        projected.theta_center_rad
    )

    span = float(
        projected.theta_span_rad
    )

    dtheta = np.arctan2(
        np.sin(
            THETA_CENTERS
            -
            center
        ),
        np.cos(
            THETA_CENTERS
            -
            center
        ),
    )

    theta_active = (
        np.abs(
            dtheta
        )
        <=
        (
            0.5
            *
            span
        )
    )

    range_active = (
        (
            RANGE_CENTERS
            >=
            float(
                projected.ground_range_min_m
            )
        )
        &
        (
            RANGE_CENTERS
            <=
            float(
                projected.ground_range_max_m
            )
        )
    )

    return (
        theta_active[
            :,
            None
        ]
        &
        range_active[
            None,
            :
        ]
    )


def future_box_and_yaw(
    *,
    scenario,
    adapted,
    causal_actor,
    time_index,
):
    track_index = int(
        causal_actor.track_index
    )

    require(
        0
        <=
        track_index
        <
        len(
            scenario.tracks
        ),
        "Invalid causal actor track_index.",
    )

    track = (
        scenario.tracks[
            track_index
        ]
    )

    require(
        0
        <=
        int(
            time_index
        )
        <
        len(
            track.states
        ),
        "Future time index outside track.",
    )

    state = track.states[
        int(
            time_index
        )
    ]

    if not bool(
        state.valid
    ):
        return None

    center_W = (
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

    transform = (
        adapted
        .frames
        .T_H0_from_W
    )

    center_H0 = np.asarray(
        transform.apply_point(
            center_W
        ),
        dtype=np.float64,
    )

    heading_W = float(
        state.heading
    )

    forward_W = np.asarray(
        [
            math.cos(
                heading_W
            ),
            math.sin(
                heading_W
            ),
            0.0,
        ],
        dtype=np.float64,
    )

    rotation = np.asarray(
        transform.rotation,
        dtype=np.float64,
    )

    require(
        rotation.shape
        ==
        (
            3,
            3,
        ),
        "Unexpected H0 rotation shape.",
    )

    forward_H0 = (
        rotation
        @
        forward_W
    )

    yaw_H0 = math.atan2(
        float(
            forward_H0[
                1
            ]
        ),
        float(
            forward_H0[
                0
            ]
        ),
    )

    values = (
        float(
            state.length
        ),
        float(
            state.width
        ),
        float(
            state.height
        ),
    )

    require(
        all(
            np.isfinite(
                value
            )
            and
            value
            >
            0.0
            for value in values
        ),
        (
            "Valid future WOMD state has "
            "invalid dimensions."
        ),
    )

    box = Box3D(
        center_xyz=(
            float(
                center_H0[
                    0
                ]
            ),
            float(
                center_H0[
                    1
                ]
            ),
            float(
                center_H0[
                    2
                ]
            ),
        ),

        length_m=
            values[
                0
            ],

        width_m=
            values[
                1
            ],

        height_m=
            values[
                2
            ],

        yaw_rad=
            float(
                yaw_H0
            ),
    )

    return (
        box,
        float(
            yaw_H0
        ),
    )


def full_box_support(
    box,
    *,
    orientation_source,
):
    projected = (
        project_box_to_headlamp(
            box,

            state_source=
                "WOMD_future_GT_evaluator_only",

            orientation_source=
                orientation_source,

            controller_path=
                False,
        )
    )

    return raster_projected(
        projected
    )


def vehicle_surrogate_support(
    box,
    yaw_H0,
):
    if math.cos(
        float(
            yaw_H0
        )
    ) < 0.0:

        # Oncoming: front-upper surrogate.
        region = FractionalBoxRegion(
            x_bounds=(
                0.0,
                0.5,
            ),
            y_bounds=(
                -0.5,
                0.5,
            ),
            z_bounds=(
                0.0,
                0.5,
            ),
        )

    else:
        # Preceding: rear-upper surrogate.
        region = FractionalBoxRegion(
            x_bounds=(
                -0.5,
                0.0,
            ),
            y_bounds=(
                -0.5,
                0.5,
            ),
            z_bounds=(
                0.0,
                0.5,
            ),
        )

    projected = (
        project_region_to_headlamp(
            box,
            region,
        )
    )

    return raster_projected(
        projected
    )


def array_content_sha(
    arrays,
):
    digest = sha256()

    for name in sorted(
        arrays
    ):
        array = np.ascontiguousarray(
            arrays[
                name
            ]
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                array.dtype
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            json.dumps(
                list(
                    array.shape
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            array.tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def save_npz_atomic(
    path: Path,
    arrays,
):
    temporary = Path(
        str(
            path
        )
        +
        ".tmp.npz"
    )

    np.savez_compressed(
        temporary,
        **arrays,
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
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in (
        process.stdout
        .splitlines()
    ):
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
                process.stdout
                .splitlines()[
                    -120:
                ]
            )
        ),
    )

    require(
        count
        ==
        EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} "
            f"tests; got {count}"
        ),
    )

    return count


def numeric_parity(
    actual,
    expected,
    *,
    label,
):
    delta = abs(
        float(
            actual
        )
        -
        float(
            expected
        )
    )

    require(
        delta
        <=
        REPLAY_ABS_TOL,
        (
            f"{label} replay mismatch\n"
            f"actual={actual!r}\n"
            f"expected={expected!r}\n"
            f"abs_delta={delta!r}"
        ),
    )

    return delta


def main():

    print("=" * 76)
    print("STAGE 6 — BLOCK 6.8")
    print("STAGE-3 EXECUTION PART 1/2")
    print("FIXED STAGE1+2 MASK CACHE + EVALUATOR SUPPORT + REPLAY PARITY")
    print("NO FLOOR OUTCOMES / NO STAGE4 / NO FORMAL")
    print("=" * 76)

    ART.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # A. Exact frozen upstream
    # --------------------------------------------------------

    print()
    print("===== A. EXACT FROZEN UPSTREAM =====")

    seals = {}

    for key, path in (
        (
            "prereg",
            PREREG,
        ),
        (
            "stage1",
            STAGE1,
        ),
        (
            "stage2",
            STAGE2,
        ),
        (
            "stage3_execution",
            STAGE3_EXEC,
        ),
        (
            "stage3_scoring",
            STAGE3_SCORING,
        ),
        (
            "grid_binding",
            GRID_BINDING,
        ),
        (
            "pocc_manifest",
            POCC_MANIFEST,
        ),
        (
            "predictive_matches",
            PREDICTIVE_MATCHES,
        ),
        (
            "posterior",
            POSTERIOR,
        ),
        (
            "class_runtime",
            CLASS_RUNTIME,
        ),
        (
            "metric_runtime",
            METRIC_RUNTIME,
        ),
    ):
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
            f"{key:22s} = EXACT PASS"
        )

    print(
        "Stage-1 gamma           = IMMUTABLE"
    )
    print(
        "Stage-2 margin          = IMMUTABLE"
    )
    print(
        "Stage-3 scoring         = IMMUTABLE"
    )
    print(
        "Stage-3 floor outcomes  = NOT COMPUTED"
    )
    print(
        "Stage-4 temporal/rate   = NOT SELECTED"
    )
    print(
        "formal evaluation       = NO"
    )

    # --------------------------------------------------------
    # B. Grid / policy exactness
    # --------------------------------------------------------

    print()
    print("===== B. FROZEN POLICY READBACK =====")

    print(
        "grid shape =",
        GRID_SHAPE,
    )

    print(
        "horizons =",
        HORIZONS_S,
    )

    for actor_class in (
        TYPE_VEHICLE,
        TYPE_PEDESTRIAN,
        TYPE_CYCLIST,
    ):
        print(
            actor_class,
            "k=",
            K_BY_CLASS[
                actor_class
            ],
            "margin=",
            STAGE2_PARAMS[
                actor_class
            ],
        )

    print(
        "predictive dilation input = "
        "additional_class_margin_m ONLY"
    )

    print(
        "Part-A generic margin     = "
        "NOT re-applied by Stage-2 dilation"
    )

    # --------------------------------------------------------
    # C. Load exact development metadata
    # --------------------------------------------------------

    print()
    print("===== C. DEVELOPMENT POPULATION / JOINS =====")

    cohort = read_jsonl(
        COHORT
    )

    matches = read_jsonl(
        PREDICTIVE_MATCHES
    )

    pocc_rows = read_jsonl(
        POCC_MANIFEST
    )

    posterior_rows = read_jsonl(
        POSTERIOR
    )

    require(
        len(
            cohort
        )
        ==
        120,
        "Development cohort != 120.",
    )

    require(
        len(
            matches
        )
        ==
        877,
        "Predictive matches != 877.",
    )

    require(
        len(
            pocc_rows
        )
        ==
        877,
        "P_occ rows != 877.",
    )

    # The identity-safe posterior is an upstream
    # development posterior population and may be a strict
    # superset of the 877 Stage-3 selected predictive actors.
    #
    # Scientific invariant:
    # every selected predictive actor must resolve to an
    # existing unique posterior row.  Total posterior-file
    # cardinality is NOT the Stage-3 actor-selection cardinality.

    posterior_total_rows = len(
        posterior_rows
    )

    require(
        posterior_total_rows
        >
        0,
        "Posterior file is empty.",
    )

    scenario_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in cohort
    ]

    require(
        len(
            set(
                scenario_ids
            )
        )
        ==
        120,
        "Duplicate development scenario ID.",
    )

    match_key = lambda row: (
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
                "actor_box_index"
            ]
        ),
        int(
            row[
                "cohort_index"
            ]
        ),
    )

    pocc_lookup = {
        match_key(
            row
        ):
            row
        for row in pocc_rows
    }

    require(
        len(
            pocc_lookup
        )
        ==
        877,
        "Duplicate P_occ identity key.",
    )

    posterior_lookup = {}

    for row in posterior_rows:
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
        )

        require(
            key
            not in
            posterior_lookup,
            (
                "Duplicate posterior "
                f"identity key: {key}"
            ),
        )

        posterior_lookup[
            key
        ] = row

    # Exact selected-subset posterior identity gate.
    #
    # Do not require equality between total posterior rows
    # and selected predictive actors.  Require exact coverage
    # of the frozen 877-actor selection instead.

    selected_posterior_keys = {
        (
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
        )
        for row in matches
    }

    require(
        len(
            selected_posterior_keys
        )
        ==
        877,
        (
            "Selected predictive posterior identity "
            "keys are not 877 unique keys: "
            f"{len(selected_posterior_keys)}"
        ),
    )

    missing_selected_posteriors = sorted(
        selected_posterior_keys
        -
        set(
            posterior_lookup
        )
    )

    require(
        not missing_selected_posteriors,
        (
            "Frozen selected predictive actors are "
            "missing posterior rows: "
            f"{missing_selected_posteriors[:10]}"
        ),
    )

    print(
        "posterior total rows     =",
        posterior_total_rows,
    )

    print(
        "selected posterior keys =",
        len(
            selected_posterior_keys
        ),
        "/ 877 EXACT PASS",
    )

    print(
        "posterior superset rows  =",
        posterior_total_rows
        -
        len(
            selected_posterior_keys
        ),
    )

    print(
        "posterior total == 877   = NOT REQUIRED"
    )

    by_scenario = defaultdict(
        list
    )

    class_counts = defaultdict(
        int
    )

    for record in matches:
        key = match_key(
            record
        )

        require(
            key
            in
            pocc_lookup,
            (
                "Predictive match missing "
                f"P_occ row: {key}"
            ),
        )

        pocc = pocc_lookup[
            key
        ]

        require(
            str(
                record[
                    "object_type"
                ]
            )
            ==
            str(
                pocc[
                    "object_type"
                ]
            ),
            "P_occ/match class mismatch.",
        )

        require(
            record.get(
                "future_used_for_eligibility"
            )
            is False,
            (
                "Future used for predictive "
                "actor eligibility."
            ),
        )

        require(
            record.get(
                "tracks_to_predict_used"
            )
            is False,
            "tracks_to_predict used.",
        )

        require(
            record.get(
                "objects_of_interest_used"
            )
            is False,
            "objects_of_interest used.",
        )

        posterior_key = (
            str(
                record[
                    "scenario_id"
                ]
            ),
            str(
                record[
                    "prediction_id"
                ]
            ),
            int(
                record[
                    "cohort_index"
                ]
            ),
        )

        require(
            posterior_key
            in
            posterior_lookup,
            (
                "Predictive actor missing "
                f"posterior: {posterior_key}"
            ),
        )

        by_scenario[
            str(
                record[
                    "scenario_id"
                ]
            )
        ].append(
            record
        )

        class_counts[
            str(
                record[
                    "object_type"
                ]
            )
        ] += 1

    require(
        dict(
            class_counts
        )
        ==
        EXPECTED_ACTORS,
        (
            "Predictive class population mismatch: "
            f"{dict(class_counts)}"
        ),
    )

    print(
        "development scenarios = 120 EXACT"
    )
    print(
        "predictive actors      = 877 EXACT"
    )

    for key, value in (
        EXPECTED_ACTORS.items()
    ):
        print(
            f"  {key:17s} = {value}"
        )

    print(
        "identity join          = EXACT"
    )

    # --------------------------------------------------------
    # D. Resolve already-frozen evaluator provenance
    # --------------------------------------------------------

    print()
    print("===== D. FUTURE-GT EVALUATOR PROVENANCE =====")

    orientation_source = (
        resolve_future_orientation_source()
    )

    print(
        "state source       = "
        "WOMD_future_GT_evaluator_only"
    )

    print(
        "orientation source =",
        orientation_source,
    )

    print(
        "controller_path    = False"
    )

    print(
        "future truth role  = EVALUATOR ONLY"
    )

    # --------------------------------------------------------
    # E. Prepare aggregate parity
    # --------------------------------------------------------

    stage1_values = {
        actor_class:
            []
        for actor_class in
        EXPECTED_ACTORS
    }

    stage2_iou_values = {
        actor_class:
            []
        for actor_class in
        EXPECTED_ACTORS
    }

    stage2_overmask_values = {
        actor_class:
            []
        for actor_class in
        EXPECTED_ACTORS
    }

    invalid_future = defaultdict(
        int
    )

    empty_valid_oracle = defaultdict(
        int
    )

    cache_records = []

    timestamp_offsets = {
        horizon:
            defaultdict(
                int
            )
        for horizon in
        HORIZONS_S
    }

    # --------------------------------------------------------
    # F. Scenario processing
    # --------------------------------------------------------

    print()
    print("===== E. MATERIALIZE FIXED MASK / EVALUATOR CACHE =====")

    for cohort_rank, cohort_row in enumerate(
        cohort
    ):
        scenario_id = str(
            cohort_row[
                "scenario_id"
            ]
        )

        records = sorted(
            by_scenario.get(
                scenario_id,
                [],
            ),
            key=lambda row: (
                int(
                    row[
                        "actor_box_index"
                    ]
                ),
                str(
                    row[
                        "prediction_id"
                    ]
                ),
            ),
        )

        cache_npz = (
            CACHE
            /
            (
                f"{cohort_rank:03d}_"
                f"{scenario_id}.npz"
            )
        )

        cache_meta = (
            CACHE
            /
            (
                f"{cohort_rank:03d}_"
                f"{scenario_id}.json"
            )
        )

        # Current runner always reconstructs scenario parity.
        # Existing cache is replaced atomically only after
        # successful scenario completion.

        motion_path = Path(
            str(
                cohort_row[
                    "motion_shard"
                ]
            )
        )

        require(
            motion_path.is_absolute(),
            (
                "motion_shard is not an absolute "
                f"frozen path: {motion_path}"
            ),
        )

        require(
            "compact_record_offset"
            in
            cohort_row,
            (
                "Missing compact_record_offset "
                f"for {scenario_id}"
            ),
        )

        scenario, payload_length = (
            read_tfrecord_at_offset(
                motion_path,
                int(
                    cohort_row[
                        "compact_record_offset"
                    ]
                ),
            )
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            scenario_id,
            (
                "Scenario ID mismatch after "
                "TFRecord read."
            ),
        )

        if (
            "payload_length"
            in
            cohort_row
        ):
            require(
                int(
                    cohort_row[
                        "payload_length"
                    ]
                )
                ==
                payload_length,
                (
                    "Frozen payload length "
                    "mismatch."
                ),
            )

        adapted = (
            adapt_causal_womd_scenario(
                scenario
            )
        )

        causal_boxes = (
            build_causal_adb_actor_boxes(
                scenario=scenario,
                adapted=adapted,
            )
        )

        selected_indices = {
            int(
                row[
                    "actor_box_index"
                ]
            )
            for row in records
        }

        require(
            all(
                0
                <=
                index
                <
                len(
                    causal_boxes
                )
                for index in
                selected_indices
            ),
            (
                "actor_box_index outside "
                "causal box tuple."
            ),
        )

        horizon_indices = (
            future_indices(
                scenario
            )
        )

        anchor = int(
            scenario.current_time_index
        )

        for horizon, index in zip(
            HORIZONS_S,
            horizon_indices,
        ):
            timestamp_offsets[
                horizon
            ][
                int(
                    index
                    -
                    anchor
                )
            ] += 1

        class_masks = {
            actor_class:
                np.zeros(
                    GRID_SHAPE,
                    dtype=bool,
                )
            for actor_class in
            EXPECTED_ACTORS
        }

        oracle_all = np.zeros(
            GRID_SHAPE,
            dtype=bool,
        )

        oracle_vehicle = np.zeros(
            GRID_SHAPE,
            dtype=bool,
        )

        pedestrian_region = np.zeros(
            GRID_SHAPE,
            dtype=bool,
        )

        cyclist_region = np.zeros(
            GRID_SHAPE,
            dtype=bool,
        )

        vehicle_surrogate = np.zeros(
            GRID_SHAPE,
            dtype=bool,
        )

        scenario_actor_counts = defaultdict(
            int
        )

        for record in records:
            actor_class = str(
                record[
                    "object_type"
                ]
            )

            require(
                actor_class
                in
                EXPECTED_ACTORS,
                (
                    "Unexpected Stage3 actor "
                    f"class: {actor_class}"
                ),
            )

            actor_index = int(
                record[
                    "actor_box_index"
                ]
            )

            causal = causal_boxes[
                actor_index
            ]

            require(
                str(
                    causal.object_type
                )
                ==
                actor_class,
                (
                    "Causal actor class mismatch."
                ),
            )

            verify_current_box_match(
                causal,
                record,
            )

            key = match_key(
                record
            )

            pocc = pocc_lookup[
                key
            ]

            pocc_path = Path(
                str(
                    pocc[
                        "occupancy_counts_path"
                    ]
                )
            )

            require(
                pocc_path.is_file(),
                (
                    "P_occ payload missing: "
                    f"{pocc_path}"
                ),
            )

            if (
                "occupancy_counts_sha256"
                in
                pocc
            ):
                require(
                    file_sha(
                        pocc_path
                    )
                    ==
                    str(
                        pocc[
                            "occupancy_counts_sha256"
                        ]
                    ),
                    (
                        "P_occ payload SHA mismatch: "
                        f"{pocc_path}"
                    ),
                )

            counts = np.load(
                pocc_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            require(
                counts.shape
                ==
                GRID_SHAPE,
                (
                    "P_occ payload shape mismatch: "
                    f"{counts.shape}"
                ),
            )

            require(
                counts.dtype
                ==
                np.dtype(
                    np.uint32
                ),
                (
                    "P_occ payload dtype mismatch: "
                    f"{counts.dtype}"
                ),
            )

            require(
                int(
                    pocc[
                        "sample_count"
                    ]
                )
                ==
                N_MC,
                "P_occ sample count != 8192.",
            )

            binary = (
                threshold_occupancy_counts_strict_k(
                    counts,
                    k=
                        K_BY_CLASS[
                            actor_class
                        ],
                    sample_count=
                        N_MC,
                )
            )

            binary = np.asarray(
                binary,
                dtype=bool,
            )

            require(
                binary.shape
                ==
                GRID_SHAPE,
                (
                    "Threshold output shape "
                    "mismatch."
                ),
            )

            posterior_key = (
                scenario_id,
                str(
                    record[
                        "prediction_id"
                    ]
                ),
                int(
                    record[
                        "cohort_index"
                    ]
                ),
            )

            posterior = posterior_lookup[
                posterior_key
            ]

            require(
                posterior.get(
                    "future_GT_used"
                )
                is False,
                (
                    "Controller posterior claims "
                    "future GT usage."
                ),
            )

            require(
                tuple(
                    float(
                        value
                    )
                    for value in
                    posterior[
                        "horizons_s"
                    ]
                )
                ==
                HORIZONS_S,
                (
                    "Posterior horizons mismatch."
                ),
            )

            dilated = np.empty_like(
                binary,
                dtype=bool,
            )

            params = STAGE2_PARAMS[
                actor_class
            ]

            for h, horizon in enumerate(
                HORIZONS_S
            ):
                margin = (
                    compute_class_aware_margin(
                        actor_class,

                        current_position_H0_m=
                            posterior[
                                "latest_position_H0_m"
                            ],

                        mean_displacement_H0_m=
                            posterior[
                                "mean_displacement_H0_m"
                            ][
                                h
                            ],

                        covariance_H0_m2=
                            posterior[
                                "calibrated_predictive_covariance_H0_m2"
                            ][
                                h
                            ],

                        horizon_s=
                            float(
                                horizon
                            ),

                        base_margin_m=
                            params[
                                "base_margin_m"
                            ],

                        uncertainty_multiplier=
                            params[
                                "uncertainty_multiplier"
                            ],

                        motion_multiplier=
                            params[
                                "motion_multiplier"
                            ],
                    )
                )

                # CRITICAL frozen Block6.7 semantic:
                # predictive dilation receives m_extra only.
                cells = (
                    angular_margin_cells(
                        total_margin_m=
                            float(
                                margin
                                .additional_class_margin_m
                            ),

                        predicted_range_m=
                            float(
                                margin
                                .predicted_range_m
                            ),

                        theta_step_rad=
                            THETA_STEP,
                    )
                )

                dilated[
                    h
                ] = (
                    dilate_mask_theta(
                        binary[
                            h:h+1
                        ],

                        dilation_cells=
                            int(
                                cells
                            ),
                    )[
                        0
                    ]
                )

            class_masks[
                actor_class
            ] |= dilated

            scenario_actor_counts[
                actor_class
            ] += 1

            # ------------------------------------------------
            # Evaluator-only future full-box reference
            # ------------------------------------------------

            for h, time_index in enumerate(
                horizon_indices
            ):
                future = (
                    future_box_and_yaw(
                        scenario=scenario,
                        adapted=adapted,
                        causal_actor=causal,
                        time_index=time_index,
                    )
                )

                if future is None:
                    invalid_future[
                        actor_class
                    ] += 1

                    continue

                (
                    future_box,
                    future_yaw,
                ) = future

                oracle = (
                    full_box_support(
                        future_box,

                        orientation_source=
                            orientation_source,
                    )
                )

                require(
                    oracle.shape
                    ==
                    (
                        501,
                        301,
                    ),
                    (
                        "Future oracle support "
                        "shape mismatch."
                    ),
                )

                if not np.any(
                    oracle
                ):
                    empty_valid_oracle[
                        actor_class
                    ] += 1

                # Stage1 replay metric.
                stage1_values[
                    actor_class
                ].append(
                    mask_iou(
                        binary[
                            h
                        ],
                        oracle,
                    )
                )

                # Stage2 replay metrics.
                stage2_iou_values[
                    actor_class
                ].append(
                    mask_iou(
                        dilated[
                            h
                        ],
                        oracle,
                    )
                )

                stage2_overmask_values[
                    actor_class
                ].append(
                    over_masking_area(
                        dilated[
                            h
                        ],
                        oracle,
                    )
                )

                oracle_all[
                    h
                ] |= oracle

                if actor_class == TYPE_VEHICLE:
                    oracle_vehicle[
                        h
                    ] |= oracle

                    vehicle_surrogate[
                        h
                    ] |= (
                        vehicle_surrogate_support(
                            future_box,
                            future_yaw,
                        )
                    )

                elif (
                    actor_class
                    ==
                    TYPE_PEDESTRIAN
                ):
                    pedestrian_region[
                        h
                    ] |= oracle

                elif (
                    actor_class
                    ==
                    TYPE_CYCLIST
                ):
                    cyclist_region[
                        h
                    ] |= oracle

        arrays = {
            "mask_vehicle":
                class_masks[
                    TYPE_VEHICLE
                ],

            "mask_pedestrian":
                class_masks[
                    TYPE_PEDESTRIAN
                ],

            "mask_cyclist":
                class_masks[
                    TYPE_CYCLIST
                ],

            "oracle_all":
                oracle_all,

            "oracle_vehicle":
                oracle_vehicle,

            "pedestrian_region":
                pedestrian_region,

            "cyclist_region":
                cyclist_region,

            "vehicle_surrogate":
                vehicle_surrogate,
        }

        for name, array in arrays.items():
            require(
                array.shape
                ==
                GRID_SHAPE,
                (
                    f"{name} shape mismatch "
                    f"in {scenario_id}: "
                    f"{array.shape}"
                ),
            )

            require(
                array.dtype
                ==
                np.dtype(
                    bool
                ),
                (
                    f"{name} dtype mismatch "
                    f"in {scenario_id}"
                ),
            )

        content_sha = (
            array_content_sha(
                arrays
            )
        )

        save_npz_atomic(
            cache_npz,
            arrays,
        )

        meta = {
            "stage":
                6,

            "block":
                "6.8_stage3_fixed_cache",

            "status":
                "PASS_SCENARIO_FIXED_CACHE",

            "contract_id":
                CONTRACT_ID,

            "cohort_rank":
                int(
                    cohort_rank
                ),

            "cohort_index":
                int(
                    cohort_row[
                        "cohort_rank"
                    ]
                    if
                    "cohort_rank"
                    in
                    cohort_row
                    else
                    cohort_rank
                ),

            "scenario_id":
                scenario_id,

            "predictive_actor_count":
                len(
                    records
                ),

            "predictive_actor_counts_by_class":
                {
                    key:
                        int(
                            scenario_actor_counts[
                                key
                            ]
                        )
                    for key in
                    EXPECTED_ACTORS
                },

            "future_time_indices":
                [
                    int(
                        x
                    )
                    for x in
                    horizon_indices
                ],

            "future_offsets_from_anchor":
                [
                    int(
                        x
                        -
                        int(
                            scenario.current_time_index
                        )
                    )
                    for x in
                    horizon_indices
                ],

            "future_truth_role":
                "EVALUATOR_ONLY_AFTER_CONTROLLER_DECISION",

            "controller_future_truth_access":
                False,

            "orientation_source":
                orientation_source,

            "predictive_margin_dilation_input":
                "additional_class_margin_m_only",

            "array_content_sha256":
                content_sha,

            "npz_path":
                str(
                    cache_npz
                ),

            "scientific_boundary": {
                "floor_values_evaluated":
                    False,

                "Stage3_floor_metrics_computed":
                    False,

                "Stage3_floor_selected":
                    False,

                "Stage4_temporal_or_rate_selected":
                    False,

                "primary_acceptance_tested":
                    False,

                "formal_evaluation":
                    False,
            },

            "upstream_seals":
                seals,
        }

        write_json_atomic(
            cache_meta,
            meta,
        )

        cache_records.append(
            {
                "cohort_rank":
                    int(
                        cohort_rank
                    ),

                "scenario_id":
                    scenario_id,

                "npz_path":
                    str(
                        cache_npz
                    ),

                "meta_path":
                    str(
                        cache_meta
                    ),

                "array_content_sha256":
                    content_sha,

                "predictive_actor_count":
                    len(
                        records
                    ),
            }
        )

        if (
            cohort_rank
            ==
            0
            or
            (
                cohort_rank
                +
                1
            )
            %
            10
            ==
            0
            or
            cohort_rank
            ==
            119
        ):
            print(
                (
                    f"scenario "
                    f"{cohort_rank + 1:03d}/120 "
                    f"{scenario_id} "
                    f"actors={len(records)} "
                    f"cache=PASS"
                ),
                flush=True,
            )

    # --------------------------------------------------------
    # F. Timestamp replay
    # --------------------------------------------------------

    print()
    print("===== F. FUTURE-TIMESTAMP REPLAY =====")

    expected_offsets = {
        0.1:
            {
                1:
                    120,
            },

        0.3:
            {
                3:
                    120,
            },

        0.5:
            {
                5:
                    120,
            },

        1.0:
            {
                9:
                    2,
                10:
                    118,
            },
    }

    for horizon in HORIZONS_S:
        actual = {
            int(
                key
            ):
                int(
                    value
                )
            for key, value in
            timestamp_offsets[
                horizon
            ].items()
        }

        require(
            actual
            ==
            expected_offsets[
                horizon
            ],
            (
                "Frozen timestamp replay mismatch "
                f"@ {horizon}s: "
                f"{actual}"
            ),
        )

        print(
            f"{horizon:0.1f}s offsets =",
            actual,
            "PASS",
        )

    # --------------------------------------------------------
    # G. Stage1 parity
    # --------------------------------------------------------

    print()
    print("===== G. FROZEN STAGE-1 GAMMA REPLAY PARITY =====")

    stage1_summary = {}

    for actor_class in (
        TYPE_VEHICLE,
        TYPE_PEDESTRIAN,
        TYPE_CYCLIST,
    ):
        values = stage1_values[
            actor_class
        ]

        target = STAGE1_TARGET[
            actor_class
        ]

        require(
            len(
                values
            )
            ==
            int(
                target[
                    "scored"
                ]
            ),
            (
                f"{actor_class} Stage1 scored "
                f"count mismatch: {len(values)}"
            ),
        )

        require(
            int(
                invalid_future[
                    actor_class
                ]
            )
            ==
            int(
                target[
                    "invalid_future"
                ]
            ),
            (
                f"{actor_class} invalid-future "
                "count mismatch."
            ),
        )

        require(
            int(
                empty_valid_oracle[
                    actor_class
                ]
            )
            ==
            int(
                target[
                    "empty_valid_oracle"
                ]
            ),
            (
                f"{actor_class} empty-valid-oracle "
                "count mismatch."
            ),
        )

        mean_iou = float(
            math.fsum(
                float(
                    value
                )
                for value in
                values
            )
            /
            len(
                values
            )
        )

        delta = numeric_parity(
            mean_iou,
            target[
                "mean_iou"
            ],
            label=(
                f"{actor_class} Stage1 mean IoU"
            ),
        )

        stage1_summary[
            actor_class
        ] = {
            "mean_iou":
                mean_iou,

            "target_mean_iou":
                target[
                    "mean_iou"
                ],

            "absolute_delta":
                delta,

            "scored_actor_horizons":
                len(
                    values
                ),

            "invalid_future":
                int(
                    invalid_future[
                        actor_class
                    ]
                ),

            "empty_valid_oracle":
                int(
                    empty_valid_oracle[
                        actor_class
                    ]
                ),
        }

        print(
            actor_class,
            "| meanIoU=",
            repr(
                mean_iou
            ),
            "| target=",
            repr(
                target[
                    "mean_iou"
                ]
            ),
            "| delta=",
            repr(
                delta
            ),
            "| PASS",
        )

    # --------------------------------------------------------
    # H. Stage2 parity
    # --------------------------------------------------------

    print()
    print("===== H. FROZEN STAGE-2 MARGIN REPLAY PARITY =====")

    stage2_summary = {}

    for actor_class in (
        TYPE_VEHICLE,
        TYPE_PEDESTRIAN,
        TYPE_CYCLIST,
    ):
        iou_values = (
            stage2_iou_values[
                actor_class
            ]
        )

        over_values = (
            stage2_overmask_values[
                actor_class
            ]
        )

        require(
            len(
                iou_values
            )
            ==
            len(
                stage1_values[
                    actor_class
                ]
            ),
            (
                f"{actor_class} Stage2 IoU "
                "support mismatch."
            ),
        )

        require(
            len(
                over_values
            )
            ==
            len(
                iou_values
            ),
            (
                f"{actor_class} Stage2 overmask "
                "support mismatch."
            ),
        )

        mean_iou = float(
            math.fsum(
                float(
                    value
                )
                for value in
                iou_values
            )
            /
            len(
                iou_values
            )
        )

        mean_overmask = float(
            math.fsum(
                float(
                    value
                )
                for value in
                over_values
            )
            /
            len(
                over_values
            )
        )

        target = STAGE2_TARGET[
            actor_class
        ]

        iou_delta = numeric_parity(
            mean_iou,
            target[
                "mean_iou"
            ],
            label=(
                f"{actor_class} Stage2 mean IoU"
            ),
        )

        over_delta = numeric_parity(
            mean_overmask,
            target[
                "mean_overmask"
            ],
            label=(
                f"{actor_class} Stage2 mean overmask"
            ),
        )

        stage2_summary[
            actor_class
        ] = {
            "mean_iou":
                mean_iou,

            "target_mean_iou":
                target[
                    "mean_iou"
                ],

            "iou_absolute_delta":
                iou_delta,

            "mean_overmask":
                mean_overmask,

            "target_mean_overmask":
                target[
                    "mean_overmask"
                ],

            "overmask_absolute_delta":
                over_delta,

            "scored_actor_horizons":
                len(
                    iou_values
                ),

            "dilation_input":
                "additional_class_margin_m_only",
        }

        print(
            actor_class,
            "| IoU=",
            repr(
                mean_iou
            ),
            "| overmask=",
            repr(
                mean_overmask
            ),
            "| PASS",
        )

    # --------------------------------------------------------
    # I. Manifest
    # --------------------------------------------------------

    print()
    print("===== I. FIXED CACHE MANIFEST FREEZE =====")

    require(
        len(
            cache_records
        )
        ==
        120,
        "Fixed cache manifest count != 120.",
    )

    manifest_text = "".join(
        json.dumps(
            record,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
        +
        "\n"
        for record in
        cache_records
    )

    write_text_atomic(
        MANIFEST,
        manifest_text,
    )

    manifest_sha = file_sha(
        MANIFEST
    )

    freeze = {
        "stage":
            6,

        "block":
            "6.8_stage3_execution_part1",

        "status":
            "FROZEN_STAGE3_FIXED_CACHE_PARITY_PASS",

        "contract_id":
            CONTRACT_ID,

        "development_population": {
            "scenarios":
                120,

            "predictive_actors":
                877,

            "class_counts":
                EXPECTED_ACTORS,
        },

        "policy": {
            "Stage1_k":
                K_BY_CLASS,

            "Stage2_margin":
                STAGE2_PARAMS,

            "predictive_dilation_input":
                "additional_class_margin_m_only",
        },

        "evaluator": {
            "future_truth_role":
                "EVALUATOR_ONLY",

            "orientation_source":
                orientation_source,

            "full_box":
                True,

            "centroid_only":
                False,

            "future_validity_is_selector":
                False,
        },

        "Stage1_replay":
            stage1_summary,

        "Stage2_replay":
            stage2_summary,

        "manifest": {
            "path":
                str(
                    MANIFEST
                ),

            "sha256":
                manifest_sha,

            "records":
                120,
        },

        "scientific_boundary": {
            "P_occ_values_opened":
                True,

            "future_GT_opened":
                True,

            "future_GT_role":
                "EVALUATOR_ONLY",

            "Stage3_floor_candidate_evaluated":
                False,

            "Stage3_floor_metrics_computed":
                False,

            "Stage3_floor_selected":
                False,

            "Stage4_temporal_or_rate_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "upstream_seals":
            seals,
    }

    write_json_atomic(
        FREEZE,
        freeze,
    )

    freeze_sha = file_sha(
        FREEZE
    )

    print(
        "manifest SHA256 =",
        manifest_sha,
    )

    print(
        "freeze SHA256   =",
        freeze_sha,
    )

    # --------------------------------------------------------
    # J. Regression / storage
    # --------------------------------------------------------

    print()
    print("===== J. FULL STAGE6 REGRESSION =====")

    tests = run_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    free_gib = (
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

    require(
        free_gib
        >=
        250.0,
        (
            "Storage reserve violated: "
            f"{free_gib:.3f} GiB"
        ),
    )

    print()
    print("===== K. STORAGE =====")
    print(
        "free GiB        =",
        f"{free_gib:.3f}",
    )
    print(
        "250-GiB reserve = PASS"
    )

    report = {
        "stage":
            6,

        "block":
            "6.8_stage3_execution_part1",

        "status":
            "PASS_STAGE3_FIXED_CACHE_PARITY_FROZEN",

        "fixed_cache_freeze": {
            "path":
                str(
                    FREEZE
                ),

            "sha256":
                freeze_sha,
        },

        "manifest": {
            "path":
                str(
                    MANIFEST
                ),

            "sha256":
                manifest_sha,
        },

        "development_scenarios":
            120,

        "predictive_actors":
            877,

        "Stage1_replay":
            stage1_summary,

        "Stage2_replay":
            stage2_summary,

        "Stage6_regression":
            tests,

        "formal_evaluation":
            False,

        "next":
            (
                "execute exactly 3270 frozen Stage-3 "
                "class-floor candidates using only this "
                "fixed cache and the already-frozen "
                "Stage-3 scoring contract"
            ),
    }

    write_json_atomic(
        REPORT,
        report,
    )

    print()
    print("=" * 76)
    print("BLOCK 6.8 STAGE-3 EXECUTION PART 1/2 — FINAL")
    print("=" * 76)

    print(
        "Stage-1 gamma             = IMMUTABLE"
    )

    print(
        "Stage-2 margins           = IMMUTABLE"
    )

    print(
        "predictive actors         = 877 EXACT"
    )

    print(
        "fixed scenario caches     = 120 / 120 PASS"
    )

    print(
        "P_occ values              = OPENED DEVELOPMENT ONLY"
    )

    print(
        "future GT                 = EVALUATOR ONLY"
    )

    print(
        "future validity selector  = NO"
    )

    print(
        "full-box evaluator        = YES"
    )

    print(
        "centroid-only evaluator   = NO"
    )

    print(
        "Stage-1 replay parity     = PASS"
    )

    print(
        "Stage-2 replay parity     = PASS"
    )

    print(
        "Stage-2 dilation input    = m_extra ONLY"
    )

    print(
        "Stage-3 floor outcomes    = NOT COMPUTED"
    )

    print(
        "selected floor tuple      = NONE"
    )

    print(
        "Stage-4 temporal/rate     = NOT SELECTED"
    )

    print(
        "primary acceptance        = NOT TESTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_FIXED_CACHE_PARITY_FROZEN"
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
    print("=" * 76)
    print(
        "BLOCK 6.8 STAGE-3 EXECUTION PART 1/2 = BLOCKED"
    )
    print("=" * 76)

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
        "Stage-1 gamma          = STILL FROZEN"
    )
    print(
        "Stage-2 margins        = STILL FROZEN"
    )
    print(
        "Stage-3 floor outcomes = NOT COMPUTED"
    )
    print(
        "selected floor tuple   = NONE"
    )
    print(
        "Stage-4 temporal/rate  = NOT SELECTED"
    )
    print(
        "primary acceptance     = NOT TESTED"
    )
    print(
        "formal evaluation      = NO"
    )

    print()
    print(
        "Do NOT execute the 3270-triplet "
        "floor sweep until this parity gate passes."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

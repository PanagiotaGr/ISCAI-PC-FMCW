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
    compose_class_aware_illumination,
    predictive_mask_to_illumination,
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


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

CACHE_ROOT = (
    S6
    / "artifacts/block68/stage3_fixed_cache"
)

CACHE_DIR = (
    CACHE_ROOT
    / "scenarios"
)

CACHE_MANIFEST = (
    CACHE_ROOT
    / "stage3_fixed_cache_manifest.jsonl"
)

CACHE_FREEZE = (
    CACHE_ROOT
    / "stage3_fixed_cache_freeze.json"
)

STAGE1 = (
    S6
    / "configs/"
      "stage6_development_stage1_class_gamma_freeze.json"
)

STAGE2 = (
    S6
    / "configs/"
      "stage6_development_stage2_class_margin_freeze.json"
)

STAGE3_EXECUTION = (
    S6
    / "configs/"
      "stage6_development_stage3_floor_execution_binding.json"
)

STAGE3_SCORING = (
    S6
    / "configs/"
      "stage6_development_stage3_floor_scoring_semantics.json"
)

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

CLASS_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/class_aware_policy.py"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/metric_semantics.py"
)

EVALUATOR_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/evaluator_reference.py"
)

ROAD_ROI_ARTIFACT = (
    S6
    / "artifacts/block68/frozen_road_roi_definition.json"
)

SCORES = (
    S6
    / "artifacts/block68/stage3_floor_selection/"
      "block68_stage3_floor_candidate_scores.jsonl"
)

WINNER_SCENARIOS = (
    S6
    / "artifacts/block68/stage3_floor_selection/"
      "block68_stage3_winner_authoritative_scenarios.jsonl"
)

FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage3_class_floor_freeze.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_stage3_class_floor_selection.json"
)


# ============================================================
# EXACT FROZEN HASHES
# ============================================================

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


# ============================================================
# FROZEN NUMERICS / SEMANTICS
# ============================================================

EXPECTED_TESTS = 224

N_HORIZONS = 4
N_THETA = 501
N_RANGE = 301

GRID_SHAPE = (
    N_HORIZONS,
    N_THETA,
    N_RANGE,
)

GRID_CELL_COUNT = int(
    np.prod(
        GRID_SHAPE
    )
)

RANGE_CENTERS = np.linspace(
    0.0,
    150.0,
    N_RANGE,
    dtype=np.float64,
)

CLASS_ORDER = (
    TYPE_VEHICLE,
    TYPE_PEDESTRIAN,
    TYPE_CYCLIST,
)

EXPECTED_CLASS_COUNTS = {
    TYPE_VEHICLE:
        719,

    TYPE_PEDESTRIAN:
        110,

    TYPE_CYCLIST:
        48,
}

RISK_NAMES = (
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "1-pedestrian_visibility_proxy",
    "1-cyclist_visibility_proxy",
    "1-road_illumination_retention",
)

METRIC_NAMES = (
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "road_illumination_retention",
    "over_masking_area",
    "false_dimming",
)

ANCHOR_TUPLES = (
    (0, 1, 1),
    (10, 10, 10),
    (20, 20, 20),
)

PARITY_ABS_TOL = 1.0e-12

HARD_RESERVE_GIB = 250.0


# ============================================================
# BASIC HELPERS
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
                1024 * 1024
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
        actual == expected,
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
                    f"Non-object JSONL row "
                    f"{path}:{line_number}"
                ),
            )

            rows.append(
                value
            )

    return rows


def canonical_bytes(
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


def atomic_write(
    path: Path,
    payload: bytes,
):
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


def write_once_json(
    path: Path,
    value,
):
    payload = canonical_bytes(
        value
    )

    if path.exists():

        require(
            path.read_bytes()
            ==
            payload,
            (
                "Existing frozen JSON differs: "
                f"{path}"
            ),
        )

        return

    atomic_write(
        path,
        payload,
    )


def write_once_jsonl(
    path: Path,
    rows,
):
    payload = (
        "".join(
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
        )
    ).encode(
        "utf-8"
    )

    if path.exists():

        require(
            path.read_bytes()
            ==
            payload,
            (
                "Existing frozen JSONL differs: "
                f"{path}"
            ),
        )

        return

    atomic_write(
        path,
        payload,
    )


def finite_or_none(
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
                    -100:
                ]
            )
        ),
    )

    require(
        count
        ==
        EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}"
        ),
    )

    return count


# ============================================================
# CACHE HASH
# ============================================================

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


# ============================================================
# FLOOR SPACE
# ============================================================

def enumerate_candidates():
    result = []

    candidate_index = 0

    for j_vehicle in range(
        0,
        21,
    ):
        for j_pedestrian in range(
            1,
            21,
        ):
            for j_cyclist in range(
                1,
                21,
            ):

                if (
                    j_vehicle
                    >
                    j_pedestrian
                ):
                    continue

                if (
                    j_vehicle
                    >
                    j_cyclist
                ):
                    continue

                result.append(
                    {
                        "candidate_index":
                            candidate_index,

                        "j_vehicle":
                            j_vehicle,

                        "j_pedestrian":
                            j_pedestrian,

                        "j_cyclist":
                            j_cyclist,

                        "f_vehicle":
                            j_vehicle
                            /
                            20.0,

                        "f_pedestrian":
                            j_pedestrian
                            /
                            20.0,

                        "f_cyclist":
                            j_cyclist
                            /
                            20.0,
                    }
                )

                candidate_index += 1

    require(
        len(
            result
        )
        ==
        3270,
        (
            "Floor candidate enumeration "
            f"!=3270: {len(result)}"
        ),
    )

    return result


# ============================================================
# AUTHORITATIVE PIPELINE
# ============================================================

def authoritative_raw(
    *,
    mask_vehicle,
    mask_pedestrian,
    mask_cyclist,
    f_vehicle,
    f_pedestrian,
    f_cyclist,
):
    vehicle = (
        predictive_mask_to_illumination(
            mask_vehicle,
            range_centers_m=
                RANGE_CENTERS,
            intensity_floor=
                float(
                    f_vehicle
                ),
        )
    )

    pedestrian = (
        predictive_mask_to_illumination(
            mask_pedestrian,
            range_centers_m=
                RANGE_CENTERS,
            intensity_floor=
                float(
                    f_pedestrian
                ),
        )
    )

    cyclist = (
        predictive_mask_to_illumination(
            mask_cyclist,
            range_centers_m=
                RANGE_CENTERS,
            intensity_floor=
                float(
                    f_cyclist
                ),
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
                    float(
                        f_pedestrian
                    ),

                TYPE_CYCLIST:
                    float(
                        f_cyclist
                    ),
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
        GRID_SHAPE,
        (
            "Authoritative raw map shape "
            f"mismatch: {raw.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(
                raw
            )
        ),
        "Authoritative raw map non-finite.",
    )

    require(
        np.all(
            raw >= 0.0
        )
        and
        np.all(
            raw <= 1.0
        ),
        "Authoritative raw outside [0,1].",
    )

    return raw


def authoritative_metrics(
    raw,
    *,
    oracle_all,
    oracle_vehicle,
    pedestrian_region,
    cyclist_region,
    vehicle_surrogate,
    road_roi,
):
    candidate_mask = (
        np.asarray(
            raw,
            dtype=np.float64,
        )
        <
        1.0
    )

    return {
        "vehicle_shadow_zone_violation":
            float(
                vehicle_shadow_zone_violation(
                    candidate_mask,
                    oracle_vehicle,
                )
            ),

        "glare_risk_exposure":
            float(
                glare_risk_exposure(
                    raw,
                    vehicle_surrogate,
                )
            ),

        "pedestrian_visibility_proxy":
            float(
                pedestrian_visibility_proxy(
                    raw,
                    pedestrian_region,
                )
            ),

        "cyclist_visibility_proxy":
            float(
                cyclist_visibility_proxy(
                    raw,
                    cyclist_region,
                )
            ),

        "road_illumination_retention":
            float(
                road_illumination_retention(
                    raw,
                    road_roi,
                )
            ),

        "over_masking_area":
            float(
                over_masking_area(
                    candidate_mask,
                    oracle_all,
                )
            ),

        "false_dimming":
            float(
                false_dimming(
                    raw,
                    oracle_all,
                )
            ),
    }


# ============================================================
# FAST METRIC KERNEL
# ============================================================

def metric_empty_values():
    empty = np.zeros(
        (
            1,
            2,
            3,
        ),
        dtype=bool,
    )

    ones = np.ones(
        (
            1,
            2,
            3,
        ),
        dtype=np.float64,
    )

    return {
        "vehicle_shadow_zone_violation":
            float(
                vehicle_shadow_zone_violation(
                    np.zeros_like(
                        empty
                    ),
                    empty,
                )
            ),

        "glare_risk_exposure":
            float(
                glare_risk_exposure(
                    ones,
                    empty,
                )
            ),

        "pedestrian_visibility_proxy":
            float(
                pedestrian_visibility_proxy(
                    ones,
                    empty,
                )
            ),

        "cyclist_visibility_proxy":
            float(
                cyclist_visibility_proxy(
                    ones,
                    empty,
                )
            ),
    }


EMPTY_METRIC = metric_empty_values()


def compressed_region_mean(
    raw,
    *,
    region_on_u,
    total_region_count,
):
    batch_size = int(
        raw.shape[
            0
        ]
    )

    if (
        total_region_count
        ==
        0
    ):
        return np.full(
            batch_size,
            np.nan,
            dtype=np.float64,
        )

    on_u_count = int(
        np.count_nonzero(
            region_on_u
        )
    )

    outside_u_count = (
        int(
            total_region_count
        )
        -
        on_u_count
    )

    if on_u_count == 0:

        return np.ones(
            batch_size,
            dtype=np.float64,
        )

    inside_sum = np.sum(
        raw[
            :,
            region_on_u,
        ],
        axis=1,
        dtype=np.float64,
    )

    return (
        inside_sum
        +
        float(
            outside_u_count
        )
    ) / float(
        total_region_count
    )


def compressed_metrics(
    raw,
    *,
    oracle_all_u,
    oracle_vehicle_u,
    pedestrian_u,
    cyclist_u,
    vehicle_surrogate_u,
    road_u,
    oracle_all_count,
    oracle_vehicle_count,
    pedestrian_count,
    cyclist_count,
    vehicle_surrogate_count,
    road_count,
):
    batch_size = int(
        raw.shape[
            0
        ]
    )

    dim = (
        raw
        <
        1.0
    )

    dim_count = np.sum(
        dim,
        axis=1,
        dtype=np.int64,
    )

    if np.any(
        oracle_all_u
    ):
        dim_oracle = np.sum(
            dim[
                :,
                oracle_all_u,
            ],
            axis=1,
            dtype=np.int64,
        )

    else:
        dim_oracle = np.zeros(
            batch_size,
            dtype=np.int64,
        )

    unnecessary = (
        dim_count
        -
        dim_oracle
    )

    overmask = (
        unnecessary.astype(
            np.float64
        )
        /
        float(
            GRID_CELL_COUNT
        )
    )

    false = np.zeros(
        batch_size,
        dtype=np.float64,
    )

    nonempty_dim = (
        dim_count
        >
        0
    )

    false[
        nonempty_dim
    ] = (
        unnecessary[
            nonempty_dim
        ].astype(
            np.float64
        )
        /
        dim_count[
            nonempty_dim
        ].astype(
            np.float64
        )
    )

    if oracle_vehicle_count == 0:

        vehicle_violation = np.full(
            batch_size,
            EMPTY_METRIC[
                "vehicle_shadow_zone_violation"
            ],
            dtype=np.float64,
        )

    else:

        if np.any(
            oracle_vehicle_u
        ):
            protected = np.sum(
                dim[
                    :,
                    oracle_vehicle_u,
                ],
                axis=1,
                dtype=np.int64,
            )

        else:
            protected = np.zeros(
                batch_size,
                dtype=np.int64,
            )

        vehicle_violation = (
            float(
                oracle_vehicle_count
            )
            -
            protected.astype(
                np.float64
            )
        ) / float(
            oracle_vehicle_count
        )

    if vehicle_surrogate_count == 0:

        glare = np.full(
            batch_size,
            EMPTY_METRIC[
                "glare_risk_exposure"
            ],
            dtype=np.float64,
        )

    else:

        glare = compressed_region_mean(
            raw,
            region_on_u=
                vehicle_surrogate_u,
            total_region_count=
                vehicle_surrogate_count,
        )

    if pedestrian_count == 0:

        pedestrian = np.full(
            batch_size,
            EMPTY_METRIC[
                "pedestrian_visibility_proxy"
            ],
            dtype=np.float64,
        )

    else:

        pedestrian = compressed_region_mean(
            raw,
            region_on_u=
                pedestrian_u,
            total_region_count=
                pedestrian_count,
        )

    if cyclist_count == 0:

        cyclist = np.full(
            batch_size,
            EMPTY_METRIC[
                "cyclist_visibility_proxy"
            ],
            dtype=np.float64,
        )

    else:

        cyclist = compressed_region_mean(
            raw,
            region_on_u=
                cyclist_u,
            total_region_count=
                cyclist_count,
        )

    road = compressed_region_mean(
        raw,
        region_on_u=
            road_u,
        total_region_count=
            road_count,
    )

    return {
        "vehicle_shadow_zone_violation":
            vehicle_violation,

        "glare_risk_exposure":
            glare,

        "pedestrian_visibility_proxy":
            pedestrian,

        "cyclist_visibility_proxy":
            cyclist,

        "road_illumination_retention":
            road,

        "over_masking_area":
            overmask,

        "false_dimming":
            false,
    }


# ============================================================
# SYNTHETIC METRIC PARITY — NO DEVELOPMENT OUTCOME
# ============================================================

def synthetic_metric_parity():
    """
    Synthetic implementation-parity test on the ACTUAL
    frozen actuator-grid shape.

    Important:
    over_masking_area is normalized by the complete actuator
    grid. Therefore a tiny 2x3x4 synthetic tensor is not a
    valid fixture for the production compressed kernel whose
    frozen denominator is GRID_CELL_COUNT = 4*501*301.

    Keep production metric semantics unchanged; make the
    synthetic fixture conform to them.
    """

    shape = GRID_SHAPE

    require(
        int(
            np.prod(
                shape
            )
        )
        ==
        GRID_CELL_COUNT,
        (
            "Synthetic parity fixture must use the "
            "frozen production grid-cell denominator."
        ),
    )

    raw = np.ones(
        shape,
        dtype=np.float64,
    )

    # Sparse deterministic non-unity intensities.
    # Everything not listed here remains exactly 1.
    assignments = (
        ((0, 0, 0), 0.8),
        ((0, 0, 1), 0.6),
        ((0, 1, 0), 0.5),
        ((0, 1, 1), 0.2),
        ((0, 2, 2), 0.4),
        ((0, 2, 3), 0.7),
        ((1, 0, 0), 0.9),
        ((1, 0, 2), 0.3),
        ((1, 1, 1), 0.6),
        ((1, 1, 2), 0.5),
        ((1, 2, 0), 0.2),
        ((1, 2, 3), 0.8),
    )

    for index, value in assignments:
        raw[
            index
        ] = float(
            value
        )

    oracle_all = np.zeros(
        shape,
        dtype=bool,
    )

    oracle_all[
        0,
        0,
        0:2,
    ] = True

    oracle_all[
        1,
        0,
        0:3,
    ] = True

    oracle_vehicle = np.zeros(
        shape,
        dtype=bool,
    )

    oracle_vehicle[
        0,
        0,
        0,
    ] = True

    oracle_vehicle[
        1,
        0,
        0,
    ] = True

    ped = np.zeros(
        shape,
        dtype=bool,
    )

    ped[
        0,
        1,
        0:2,
    ] = True

    ped[
        1,
        1,
        1:3,
    ] = True

    cyc = np.zeros(
        shape,
        dtype=bool,
    )

    cyc[
        0,
        2,
        2:4,
    ] = True

    cyc[
        1,
        2,
        0,
    ] = True

    cyc[
        1,
        2,
        3,
    ] = True

    surrogate = np.zeros(
        shape,
        dtype=bool,
    )

    surrogate[
        0,
        0,
        0:2,
    ] = True

    surrogate[
        1,
        0,
        0:3,
    ] = True

    # Frozen road-illumination metric operates on a full-grid
    # boolean ROI. For the synthetic parity fixture, use all
    # actuator cells so both implementations aggregate the
    # same complete domain.
    road = np.ones(
        shape,
        dtype=bool,
    )

    authoritative = authoritative_metrics(
        raw,

        oracle_all=
            oracle_all,

        oracle_vehicle=
            oracle_vehicle,

        pedestrian_region=
            ped,

        cyclist_region=
            cyc,

        vehicle_surrogate=
            surrogate,

        road_roi=
            road,
    )

    # Include the entire production grid in the compressed
    # synthetic test. This intentionally makes the compressed
    # denominator identical to the authoritative metric's
    # frozen full-grid denominator.
    u = np.ones(
        shape,
        dtype=bool,
    )

    raw_u = raw.reshape(
        1,
        -1,
    )

    require(
        raw_u.shape
        ==
        (
            1,
            GRID_CELL_COUNT,
        ),
        (
            "Synthetic compressed raw shape "
            "does not equal frozen full grid."
        ),
    )

    fast = compressed_metrics(
        raw_u,

        oracle_all_u=
            oracle_all[
                u
            ],

        oracle_vehicle_u=
            oracle_vehicle[
                u
            ],

        pedestrian_u=
            ped[
                u
            ],

        cyclist_u=
            cyc[
                u
            ],

        vehicle_surrogate_u=
            surrogate[
                u
            ],

        road_u=
            road[
                u
            ],

        oracle_all_count=
            int(
                np.count_nonzero(
                    oracle_all
                )
            ),

        oracle_vehicle_count=
            int(
                np.count_nonzero(
                    oracle_vehicle
                )
            ),

        pedestrian_count=
            int(
                np.count_nonzero(
                    ped
                )
            ),

        cyclist_count=
            int(
                np.count_nonzero(
                    cyc
                )
            ),

        vehicle_surrogate_count=
            int(
                np.count_nonzero(
                    surrogate
                )
            ),

        road_count=
            int(
                np.count_nonzero(
                    road
                )
            ),
    )

    maximum_delta = 0.0

    for metric in METRIC_NAMES:

        authority_value = float(
            authoritative[
                metric
            ]
        )

        fast_value = float(
            fast[
                metric
            ][
                0
            ]
        )

        if (
            math.isnan(
                authority_value
            )
            and
            math.isnan(
                fast_value
            )
        ):
            delta = 0.0

        else:

            require(
                math.isfinite(
                    authority_value
                )
                ==
                math.isfinite(
                    fast_value
                ),
                (
                    "Synthetic metric-kernel "
                    "finiteness mismatch for "
                    f"{metric}: "
                    f"authority={authority_value}, "
                    f"fast={fast_value}"
                ),
            )

            delta = abs(
                authority_value
                -
                fast_value
            )

            require(
                delta
                <=
                PARITY_ABS_TOL,
                (
                    "Synthetic metric-kernel parity "
                    f"failed for {metric}: "
                    f"authority={authority_value}, "
                    f"fast={fast_value}, "
                    f"delta={delta}"
                ),
            )

        maximum_delta = max(
            maximum_delta,
            delta,
        )

    return maximum_delta


# ============================================================
# CANDIDATE SCORE ACCUMULATION
# ============================================================

def add_metric_values(
    sums,
    counts,
    metric,
    start,
    end,
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = np.isfinite(
        values
    )

    if not np.any(
        finite
    ):
        return

    target_sums = sums[
        metric
    ][
        start:end
    ]

    target_counts = counts[
        metric
    ][
        start:end
    ]

    target_sums[
        finite
    ] += values[
        finite
    ]

    target_counts[
        finite
    ] += 1


def aggregate_metrics(
    sums,
    counts,
):
    result = {}

    for metric in METRIC_NAMES:

        require(
            np.all(
                counts[
                    metric
                ]
                >
                0
            ),
            (
                f"Metric {metric} has candidate "
                "with zero finite scenario support."
            ),
        )

        result[
            metric
        ] = (
            sums[
                metric
            ]
            /
            counts[
                metric
            ].astype(
                np.float64
            )
        )

    return result


def candidate_key(
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
            "Stage-3 risk vector outside "
            f"frozen [0,1] domain: {risk}"
        ),
    )

    return (
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
                "over_masking_area"
            ]
        ),

        float(
            metrics[
                "false_dimming"
            ]
        ),

        int(
            candidate[
                "j_vehicle"
            ]
        ),

        int(
            candidate[
                "j_pedestrian"
            ]
        ),

        int(
            candidate[
                "j_cyclist"
            ]
        ),
    ), risk


# ============================================================
# SCENARIO FLOOR TABLE
# ============================================================

def build_scenario_tables(
    *,
    mask_vehicle,
    mask_pedestrian,
    mask_cyclist,
):
    masks = {
        TYPE_VEHICLE:
            mask_vehicle,

        TYPE_PEDESTRIAN:
            mask_pedestrian,

        TYPE_CYCLIST:
            mask_cyclist,
    }

    floor_js = {
        TYPE_VEHICLE:
            tuple(
                range(
                    0,
                    21,
                )
            ),

        TYPE_PEDESTRIAN:
            tuple(
                range(
                    1,
                    21,
                )
            ),

        TYPE_CYCLIST:
            tuple(
                range(
                    1,
                    21,
                )
            ),
    }

    minimum_j = {
        TYPE_VEHICLE:
            0,

        TYPE_PEDESTRIAN:
            1,

        TYPE_CYCLIST:
            1,
    }

    base_full = {}

    class_support = {}

    for actor_class in CLASS_ORDER:

        floor = (
            minimum_j[
                actor_class
            ]
            /
            20.0
        )

        full = (
            predictive_mask_to_illumination(
                masks[
                    actor_class
                ],

                range_centers_m=
                    RANGE_CENTERS,

                intensity_floor=
                    floor,
            )
        )

        full = np.asarray(
            full,
            dtype=np.float64,
        )

        require(
            full.shape
            ==
            GRID_SHAPE,
            (
                "Class illumination base shape "
                f"mismatch for {actor_class}."
            ),
        )

        base_full[
            actor_class
        ] = full

        class_support[
            actor_class
        ] = (
            full
            <
            1.0
        )

    interesting = (
        class_support[
            TYPE_VEHICLE
        ]
        |
        class_support[
            TYPE_PEDESTRIAN
        ]
        |
        class_support[
            TYPE_CYCLIST
        ]
        |
        mask_vehicle
        |
        mask_pedestrian
        |
        mask_cyclist
    )

    u_index = np.flatnonzero(
        interesting.reshape(
            -1
        )
    )

    u_size = int(
        len(
            u_index
        )
    )

    tables = {}

    anchor_full_maps = {
        actor_class:
            {}
        for actor_class
        in CLASS_ORDER
    }

    anchor_js = {
        TYPE_VEHICLE:
            {
                0,
                10,
                20,
            },

        TYPE_PEDESTRIAN:
            {
                1,
                10,
                20,
            },

        TYPE_CYCLIST:
            {
                1,
                10,
                20,
            },
    }

    for actor_class in CLASS_ORDER:

        js = floor_js[
            actor_class
        ]

        table = np.empty(
            (
                len(
                    js
                ),
                u_size,
            ),
            dtype=np.float64,
        )

        for table_index, j in enumerate(
            js
        ):

            if (
                j
                ==
                minimum_j[
                    actor_class
                ]
            ):
                full = base_full[
                    actor_class
                ]

            else:

                full = (
                    predictive_mask_to_illumination(
                        masks[
                            actor_class
                        ],

                        range_centers_m=
                            RANGE_CENTERS,

                        intensity_floor=
                            (
                                j
                                /
                                20.0
                            ),
                    )
                )

                full = np.asarray(
                    full,
                    dtype=np.float64,
                )

            require(
                full.shape
                ==
                GRID_SHAPE,
                (
                    "Class illumination shape "
                    f"changed for {actor_class}, j={j}"
                ),
            )

            outside_support = (
                ~class_support[
                    actor_class
                ]
            )

            # Higher floors may remove dimming, but must
            # never create new support outside the minimum-
            # floor authoritative class illumination support.
            require(
                np.all(
                    full[
                        outside_support
                    ]
                    ==
                    1.0
                ),
                (
                    "Floor changed class illumination "
                    "support outside the minimum-floor "
                    f"support: class={actor_class}, j={j}"
                ),
            )

            if u_size > 0:

                table[
                    table_index
                ] = (
                    full.reshape(
                        -1
                    )[
                        u_index
                    ]
                )

            if (
                j
                in
                anchor_js[
                    actor_class
                ]
            ):
                anchor_full_maps[
                    actor_class
                ][
                    j
                ] = (
                    full.copy()
                )

        tables[
            actor_class
        ] = table

    return (
        interesting,
        u_index,
        tables,
        anchor_full_maps,
    )


# ============================================================
# COMPRESSED RAW
# ============================================================

def compressed_raw(
    *,
    tables,
    mask_pedestrian_u,
    mask_cyclist_u,
    j_vehicle,
    j_pedestrian,
    j_cyclist,
):
    j_vehicle = np.asarray(
        j_vehicle,
        dtype=np.int64,
    )

    j_pedestrian = np.asarray(
        j_pedestrian,
        dtype=np.int64,
    )

    j_cyclist = np.asarray(
        j_cyclist,
        dtype=np.int64,
    )

    vehicle = (
        tables[
            TYPE_VEHICLE
        ][
            j_vehicle
        ]
    )

    pedestrian = (
        tables[
            TYPE_PEDESTRIAN
        ][
            j_pedestrian
            -
            1
        ]
    )

    cyclist = (
        tables[
            TYPE_CYCLIST
        ][
            j_cyclist
            -
            1
        ]
    )

    raw = np.minimum(
        vehicle,
        pedestrian,
    )

    raw = np.minimum(
        raw,
        cyclist,
    )

    batch_size = int(
        raw.shape[
            0
        ]
    )

    u_size = int(
        raw.shape[
            1
        ]
    )

    guard = np.zeros(
        (
            batch_size,
            u_size,
        ),
        dtype=np.float64,
    )

    if np.any(
        mask_pedestrian_u
    ):

        guard[
            :,
            mask_pedestrian_u,
        ] = (
            j_pedestrian[
                :,
                None,
            ]
            /
            20.0
        )

    if np.any(
        mask_cyclist_u
    ):

        cyclist_floor = (
            j_cyclist[
                :,
                None,
            ]
            /
            20.0
        )

        guard[
            :,
            mask_cyclist_u,
        ] = np.maximum(
            guard[
                :,
                mask_cyclist_u,
            ],
            cyclist_floor,
        )

    return np.maximum(
        raw,
        guard,
    )


# ============================================================
# PARITY
# ============================================================

def compare_metric_dict(
    authority,
    fast,
    *,
    label,
):
    maximum_delta = 0.0

    for metric in METRIC_NAMES:

        a = float(
            authority[
                metric
            ]
        )

        b = float(
            fast[
                metric
            ]
        )

        if (
            math.isnan(
                a
            )
            and
            math.isnan(
                b
            )
        ):
            delta = 0.0

        else:

            require(
                math.isfinite(
                    a
                )
                ==
                math.isfinite(
                    b
                ),
                (
                    f"{label} metric finiteness "
                    f"mismatch: {metric}, "
                    f"authority={a}, fast={b}"
                ),
            )

            delta = abs(
                a
                -
                b
            )

            require(
                delta
                <=
                PARITY_ABS_TOL,
                (
                    f"{label} metric parity failed: "
                    f"{metric}\n"
                    f"authority={a!r}\n"
                    f"fast={b!r}\n"
                    f"delta={delta!r}"
                ),
            )

        maximum_delta = max(
            maximum_delta,
            delta,
        )

    return maximum_delta


# ============================================================
# IDENTITY / IDEMPOTENCE
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
        "PASS_STAGE3_CLASS_FLOORS_SELECTED_FROZEN"
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
                "Frozen Stage-3 output "
                f"missing: {path}"
            ),
        )

        require(
            file_sha(
                path
            )
            ==
            report[
                "outputs"
            ][
                name
            ][
                "sha256"
            ],
            (
                "Frozen Stage-3 output "
                f"changed: {name}"
            ),
        )

    print(
        "existing Stage-3 floor freeze = EXACT PASS"
    )

    print(
        "STATUS = "
        "PASS_STAGE3_CLASS_FLOORS_SELECTED_FROZEN"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("STAGE-3 EXECUTION PART 2/2")
    print("EXACT 3270 CLASS-FLOOR DEVELOPMENT SWEEP")
    print("RAW CLASS-AWARE ILLUMINATION ONLY")
    print("NO STAGE4 TEMPORAL/RATE / NO PRIMARY ACCEPTANCE / NO FORMAL")
    print("=" * 78)

    if idempotent_readback():
        return

    # --------------------------------------------------------
    # A. Exact frozen boundary
    # --------------------------------------------------------

    print()
    print("===== A. EXACT PART-2/2 FROZEN BOUNDARY =====")

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
            STAGE3_EXECUTION,
        ),
        (
            "stage3_scoring",
            STAGE3_SCORING,
        ),
        (
            "cache_manifest",
            CACHE_MANIFEST,
        ),
        (
            "cache_freeze",
            CACHE_FREEZE,
        ),
        (
            "class_runtime",
            CLASS_RUNTIME,
        ),
        (
            "metric_runtime",
            METRIC_RUNTIME,
        ),
        (
            "evaluator_runtime",
            EVALUATOR_RUNTIME,
        ),
        (
            "road_roi",
            ROAD_ROI_ARTIFACT,
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

    cache_freeze = read_json(
        CACHE_FREEZE
    )

    require(
        cache_freeze.get(
            "status"
        )
        ==
        "FROZEN_STAGE3_FIXED_CACHE_PARITY_PASS",
        (
            "Stage-3 fixed cache freeze "
            "status mismatch."
        ),
    )

    scoring = read_json(
        STAGE3_SCORING
    )

    require(
        scoring.get(
            "status"
        )
        ==
        (
            "FROZEN_STAGE3_SCORING_SEMANTICS_"
            "BEFORE_FLOOR_OUTCOMES"
        ),
        (
            "Stage-3 scoring semantics "
            "status mismatch."
        ),
    )

    print(
        "Stage-1 gamma             = IMMUTABLE"
    )

    print(
        "Stage-2 margins           = IMMUTABLE"
    )

    print(
        "fixed scenario cache      = FROZEN"
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

    tests_pre = run_regression()

    print(
        "Stage6 pre regression     =",
        f"{tests_pre} / {tests_pre} PASS",
    )

    # --------------------------------------------------------
    # B. Metric kernel parity before development sweep
    # --------------------------------------------------------

    print()
    print("===== B. AUTHORITATIVE METRIC-KERNEL PARITY =====")

    synthetic_delta = (
        synthetic_metric_parity()
    )

    print(
        "synthetic max abs delta =",
        repr(
            synthetic_delta
        ),
    )

    print(
        "metric-kernel parity     = PASS"
    )

    # --------------------------------------------------------
    # C. Candidate space
    # --------------------------------------------------------

    print()
    print("===== C. EXACT 3270 FLOOR CANDIDATES =====")

    candidates = (
        enumerate_candidates()
    )

    n_candidates = len(
        candidates
    )

    candidate_jv = np.asarray(
        [
            row[
                "j_vehicle"
            ]
            for row in
            candidates
        ],
        dtype=np.int64,
    )

    candidate_jp = np.asarray(
        [
            row[
                "j_pedestrian"
            ]
            for row in
            candidates
        ],
        dtype=np.int64,
    )

    candidate_jc = np.asarray(
        [
            row[
                "j_cyclist"
            ]
            for row in
            candidates
        ],
        dtype=np.int64,
    )

    print(
        "candidate triplets =",
        n_candidates,
        "EXACT",
    )

    print(
        "vehicle j          = 0..20"
    )

    print(
        "pedestrian j       = 1..20"
    )

    print(
        "cyclist j          = 1..20"
    )

    print(
        "constraints        = "
        "j_vehicle <= j_pedestrian, "
        "j_vehicle <= j_cyclist"
    )

    print(
        "selection key      = "
        "(max risk, mean risk, overmask, false dimming, "
        "j_vehicle, j_pedestrian, j_cyclist)"
    )

    # --------------------------------------------------------
    # D. Cache manifest / road ROI
    # --------------------------------------------------------

    print()
    print("===== D. FIXED CACHE READBACK =====")

    manifest = read_jsonl(
        CACHE_MANIFEST
    )

    require(
        len(
            manifest
        )
        ==
        120,
        (
            "Fixed cache manifest "
            f"!=120: {len(manifest)}"
        ),
    )

    road_roi = np.asarray(
        frozen_road_roi(
            GRID_SHAPE
        ),
        dtype=bool,
    )

    require(
        road_roi.shape
        ==
        GRID_SHAPE,
        (
            "Frozen road ROI shape "
            f"mismatch: {road_roi.shape}"
        ),
    )

    road_count = int(
        np.count_nonzero(
            road_roi
        )
    )

    require(
        road_count > 0,
        "Frozen road ROI is empty.",
    )

    print(
        "scenario caches = 120 EXACT"
    )

    print(
        "road ROI cells  =",
        road_count,
    )

    print(
        "P_occ reopened  = NO"
    )

    print(
        "WOMD reopened   = NO"
    )

    # --------------------------------------------------------
    # E. Accumulators
    # --------------------------------------------------------

    metric_sums = {
        metric:
            np.zeros(
                n_candidates,
                dtype=np.float64,
            )
        for metric in
        METRIC_NAMES
    }

    metric_counts = {
        metric:
            np.zeros(
                n_candidates,
                dtype=np.int64,
            )
        for metric in
        METRIC_NAMES
    }

    anchor_max_delta = 0.0

    total_interesting_cells = 0

    # --------------------------------------------------------
    # F. Scenario-major 3270 sweep
    # --------------------------------------------------------

    print()
    print("===== E. EXACT DEVELOPMENT FLOOR SWEEP =====")

    for scenario_number, manifest_row in enumerate(
        manifest,
        start=1,
    ):

        scenario_id = str(
            manifest_row[
                "scenario_id"
            ]
        )

        cache_path = Path(
            manifest_row[
                "npz_path"
            ]
        )

        meta_path = Path(
            manifest_row[
                "meta_path"
            ]
        )

        require(
            cache_path.is_file(),
            (
                "Missing scenario cache: "
                f"{cache_path}"
            ),
        )

        require(
            meta_path.is_file(),
            (
                "Missing scenario cache metadata: "
                f"{meta_path}"
            ),
        )

        metadata = read_json(
            meta_path
        )

        require(
            str(
                metadata[
                    "scenario_id"
                ]
            )
            ==
            scenario_id,
            (
                "Scenario cache metadata "
                "identity mismatch."
            ),
        )

        with np.load(
            cache_path,
            allow_pickle=False,
        ) as loaded:

            required_arrays = {
                "mask_vehicle",
                "mask_pedestrian",
                "mask_cyclist",
                "oracle_all",
                "oracle_vehicle",
                "pedestrian_region",
                "cyclist_region",
                "vehicle_surrogate",
            }

            require(
                set(
                    loaded.files
                )
                ==
                required_arrays,
                (
                    "Scenario cache fields mismatch: "
                    f"{scenario_id}, "
                    f"{loaded.files}"
                ),
            )

            arrays = {
                name:
                    np.asarray(
                        loaded[
                            name
                        ],
                        dtype=bool,
                    )
                for name in
                required_arrays
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

        content_sha = (
            array_content_sha(
                arrays
            )
        )

        require(
            content_sha
            ==
            str(
                manifest_row[
                    "array_content_sha256"
                ]
            ),
            (
                "Scenario cache content SHA "
                f"mismatch: {scenario_id}"
            ),
        )

        require(
            content_sha
            ==
            str(
                metadata[
                    "array_content_sha256"
                ]
            ),
            (
                "Scenario cache metadata SHA "
                f"mismatch: {scenario_id}"
            ),
        )

        mask_vehicle = arrays[
            "mask_vehicle"
        ]

        mask_pedestrian = arrays[
            "mask_pedestrian"
        ]

        mask_cyclist = arrays[
            "mask_cyclist"
        ]

        oracle_all = arrays[
            "oracle_all"
        ]

        oracle_vehicle = arrays[
            "oracle_vehicle"
        ]

        pedestrian_region = arrays[
            "pedestrian_region"
        ]

        cyclist_region = arrays[
            "cyclist_region"
        ]

        vehicle_surrogate = arrays[
            "vehicle_surrogate"
        ]

        (
            interesting,
            u_index,
            tables,
            anchor_full_maps,
        ) = build_scenario_tables(
            mask_vehicle=
                mask_vehicle,

            mask_pedestrian=
                mask_pedestrian,

            mask_cyclist=
                mask_cyclist,
        )

        u_size = int(
            len(
                u_index
            )
        )

        total_interesting_cells += (
            u_size
        )

        flat_interesting = (
            interesting.reshape(
                -1
            )
        )

        mask_pedestrian_u = (
            mask_pedestrian.reshape(
                -1
            )[
                u_index
            ]
        )

        mask_cyclist_u = (
            mask_cyclist.reshape(
                -1
            )[
                u_index
            ]
        )

        oracle_all_u = (
            oracle_all.reshape(
                -1
            )[
                u_index
            ]
        )

        oracle_vehicle_u = (
            oracle_vehicle.reshape(
                -1
            )[
                u_index
            ]
        )

        pedestrian_u = (
            pedestrian_region.reshape(
                -1
            )[
                u_index
            ]
        )

        cyclist_u = (
            cyclist_region.reshape(
                -1
            )[
                u_index
            ]
        )

        vehicle_surrogate_u = (
            vehicle_surrogate.reshape(
                -1
            )[
                u_index
            ]
        )

        road_u = (
            road_roi.reshape(
                -1
            )[
                u_index
            ]
        )

        oracle_all_count = int(
            np.count_nonzero(
                oracle_all
            )
        )

        oracle_vehicle_count = int(
            np.count_nonzero(
                oracle_vehicle
            )
        )

        pedestrian_count = int(
            np.count_nonzero(
                pedestrian_region
            )
        )

        cyclist_count = int(
            np.count_nonzero(
                cyclist_region
            )
        )

        vehicle_surrogate_count = int(
            np.count_nonzero(
                vehicle_surrogate
            )
        )

        # ----------------------------------------------------
        # E1. Low/mid/high authoritative composition parity
        # ----------------------------------------------------

        for (
            j_vehicle,
            j_pedestrian,
            j_cyclist,
        ) in ANCHOR_TUPLES:

            authority_raw = np.maximum(
                np.minimum.reduce(
                    [
                        anchor_full_maps[
                            TYPE_VEHICLE
                        ][
                            j_vehicle
                        ],

                        anchor_full_maps[
                            TYPE_PEDESTRIAN
                        ][
                            j_pedestrian
                        ],

                        anchor_full_maps[
                            TYPE_CYCLIST
                        ][
                            j_cyclist
                        ],
                    ]
                ),

                np.maximum(
                    np.where(
                        mask_pedestrian,
                        j_pedestrian
                        /
                        20.0,
                        0.0,
                    ),

                    np.where(
                        mask_cyclist,
                        j_cyclist
                        /
                        20.0,
                        0.0,
                    ),
                ),
            )

            # Cross-check against the actual composition
            # operator, not only the algebraic expression.
            composition = (
                compose_class_aware_illumination(
                    class_illumination={
                        TYPE_VEHICLE:
                            anchor_full_maps[
                                TYPE_VEHICLE
                            ][
                                j_vehicle
                            ],

                        TYPE_PEDESTRIAN:
                            anchor_full_maps[
                                TYPE_PEDESTRIAN
                            ][
                                j_pedestrian
                            ],

                        TYPE_CYCLIST:
                            anchor_full_maps[
                                TYPE_CYCLIST
                            ][
                                j_cyclist
                            ],
                    },

                    class_active_masks={
                        TYPE_PEDESTRIAN:
                            mask_pedestrian,

                        TYPE_CYCLIST:
                            mask_cyclist,
                    },

                    class_floors={
                        TYPE_PEDESTRIAN:
                            (
                                j_pedestrian
                                /
                                20.0
                            ),

                        TYPE_CYCLIST:
                            (
                                j_cyclist
                                /
                                20.0
                            ),
                    },
                )
            )

            authoritative_operator_raw = (
                np.asarray(
                    composition
                    .raw_class_aware_illumination,
                    dtype=np.float64,
                )
            )

            require(
                np.array_equal(
                    authority_raw,
                    authoritative_operator_raw,
                ),
                (
                    "Frozen composition algebra "
                    "differs from authoritative operator "
                    f"scenario={scenario_id}, "
                    f"tuple={(j_vehicle, j_pedestrian, j_cyclist)}"
                ),
            )

            compressed = compressed_raw(
                tables=
                    tables,

                mask_pedestrian_u=
                    mask_pedestrian_u,

                mask_cyclist_u=
                    mask_cyclist_u,

                j_vehicle=
                    np.asarray(
                        [
                            j_vehicle
                        ],
                        dtype=np.int64,
                    ),

                j_pedestrian=
                    np.asarray(
                        [
                            j_pedestrian
                        ],
                        dtype=np.int64,
                    ),

                j_cyclist=
                    np.asarray(
                        [
                            j_cyclist
                        ],
                        dtype=np.int64,
                    ),
            )

            if u_size > 0:

                require(
                    np.array_equal(
                        compressed[
                            0
                        ],
                        authority_raw.reshape(
                            -1
                        )[
                            u_index
                        ],
                    ),
                    (
                        "Compressed raw illumination "
                        "differs from authoritative raw "
                        f"scenario={scenario_id}, "
                        f"tuple={(j_vehicle, j_pedestrian, j_cyclist)}"
                    ),
                )

            require(
                np.all(
                    authority_raw.reshape(
                        -1
                    )[
                        ~flat_interesting
                    ]
                    ==
                    1.0
                ),
                (
                    "Authoritative raw illumination "
                    "changes outside compressed support "
                    f"for {scenario_id}."
                ),
            )

            authority_metrics = (
                authoritative_metrics(
                    authority_raw,

                    oracle_all=
                        oracle_all,

                    oracle_vehicle=
                        oracle_vehicle,

                    pedestrian_region=
                        pedestrian_region,

                    cyclist_region=
                        cyclist_region,

                    vehicle_surrogate=
                        vehicle_surrogate,

                    road_roi=
                        road_roi,
                )
            )

            fast_metrics = (
                compressed_metrics(
                    compressed,

                    oracle_all_u=
                        oracle_all_u,

                    oracle_vehicle_u=
                        oracle_vehicle_u,

                    pedestrian_u=
                        pedestrian_u,

                    cyclist_u=
                        cyclist_u,

                    vehicle_surrogate_u=
                        vehicle_surrogate_u,

                    road_u=
                        road_u,

                    oracle_all_count=
                        oracle_all_count,

                    oracle_vehicle_count=
                        oracle_vehicle_count,

                    pedestrian_count=
                        pedestrian_count,

                    cyclist_count=
                        cyclist_count,

                    vehicle_surrogate_count=
                        vehicle_surrogate_count,

                    road_count=
                        road_count,
                )
            )

            fast_scalar = {
                metric:
                    float(
                        fast_metrics[
                            metric
                        ][
                            0
                        ]
                    )
                for metric in
                METRIC_NAMES
            }

            delta = compare_metric_dict(
                authority_metrics,
                fast_scalar,
                label=(
                    f"{scenario_id} "
                    f"{(j_vehicle, j_pedestrian, j_cyclist)}"
                ),
            )

            anchor_max_delta = max(
                anchor_max_delta,
                delta,
            )

        # ----------------------------------------------------
        # E2. Full 3270 candidate scoring
        # ----------------------------------------------------

        # Keep transient raw+guard+mask memory bounded.
        if u_size == 0:

            batch_size = 64

        else:

            target_bytes = (
                96
                *
                1024
                *
                1024
            )

            estimated_per_candidate = max(
                1,
                u_size
                *
                (
                    8
                    +
                    8
                    +
                    1
                )
            )

            batch_size = max(
                1,
                min(
                    64,
                    int(
                        target_bytes
                        /
                        estimated_per_candidate
                    ),
                ),
            )

        for start in range(
            0,
            n_candidates,
            batch_size,
        ):

            end = min(
                n_candidates,
                start
                +
                batch_size,
            )

            raw = compressed_raw(
                tables=
                    tables,

                mask_pedestrian_u=
                    mask_pedestrian_u,

                mask_cyclist_u=
                    mask_cyclist_u,

                j_vehicle=
                    candidate_jv[
                        start:end
                    ],

                j_pedestrian=
                    candidate_jp[
                        start:end
                    ],

                j_cyclist=
                    candidate_jc[
                        start:end
                    ],
            )

            scenario_metrics = (
                compressed_metrics(
                    raw,

                    oracle_all_u=
                        oracle_all_u,

                    oracle_vehicle_u=
                        oracle_vehicle_u,

                    pedestrian_u=
                        pedestrian_u,

                    cyclist_u=
                        cyclist_u,

                    vehicle_surrogate_u=
                        vehicle_surrogate_u,

                    road_u=
                        road_u,

                    oracle_all_count=
                        oracle_all_count,

                    oracle_vehicle_count=
                        oracle_vehicle_count,

                    pedestrian_count=
                        pedestrian_count,

                    cyclist_count=
                        cyclist_count,

                    vehicle_surrogate_count=
                        vehicle_surrogate_count,

                    road_count=
                        road_count,
                )
            )

            for metric in METRIC_NAMES:

                add_metric_values(
                    metric_sums,
                    metric_counts,
                    metric,
                    start,
                    end,
                    scenario_metrics[
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
                    f"compressed_cells={u_size} "
                    f"3270=PASS"
                ),
                flush=True,
            )

    # --------------------------------------------------------
    # F. Aggregate all 3270
    # --------------------------------------------------------

    print()
    print("===== F. SCENARIO-MACRO AGGREGATION =====")

    aggregates = aggregate_metrics(
        metric_sums,
        metric_counts,
    )

    score_rows = []

    ranking = []

    for index, candidate in enumerate(
        candidates
    ):

        metrics = {
            metric:
                float(
                    aggregates[
                        metric
                    ][
                        index
                    ]
                )
            for metric in
            METRIC_NAMES
        }

        key, risk = candidate_key(
            metrics,
            candidate,
        )

        row = {
            **candidate,

            **metrics,

            "risk_vector": {
                RISK_NAMES[
                    risk_index
                ]:
                    float(
                        risk[
                            risk_index
                        ]
                    )
                for risk_index in
                range(
                    5
                )
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
                int(
                    key[
                        4
                    ]
                ),
                int(
                    key[
                        5
                    ]
                ),
                int(
                    key[
                        6
                    ]
                ),
            ],

            "finite_scenario_support": {
                metric:
                    int(
                        metric_counts[
                            metric
                        ][
                            index
                        ]
                    )
                for metric in
                METRIC_NAMES
            },
        }

        score_rows.append(
            row
        )

        ranking.append(
            (
                key,
                index,
            )
        )

    ranking.sort(
        key=lambda item:
            item[
                0
            ]
    )

    require(
        len(
            ranking
        )
        ==
        3270,
        "Ranking does not contain all candidates.",
    )

    winner_index = int(
        ranking[
            0
        ][
            1
        ]
    )

    winner = score_rows[
        winner_index
    ]

    runner_up = score_rows[
        int(
            ranking[
                1
            ][
                1
            ]
        )
    ]

    print(
        "candidate scores        = 3270 / 3270"
    )

    print(
        "anchor max parity delta =",
        repr(
            anchor_max_delta
        ),
    )

    print(
        "winner provisional j    =",
        (
            winner[
                "j_vehicle"
            ],
            winner[
                "j_pedestrian"
            ],
            winner[
                "j_cyclist"
            ],
        ),
    )

    print(
        "winner provisional floor=",
        (
            winner[
                "f_vehicle"
            ],
            winner[
                "f_pedestrian"
            ],
            winner[
                "f_cyclist"
            ],
        ),
    )

    print(
        "winner risk max         =",
        repr(
            winner[
                "risk_max"
            ]
        ),
    )

    print(
        "winner risk mean        =",
        repr(
            winner[
                "risk_mean"
            ]
        ),
    )

    print(
        "runner-up j             =",
        (
            runner_up[
                "j_vehicle"
            ],
            runner_up[
                "j_pedestrian"
            ],
            runner_up[
                "j_cyclist"
            ],
        ),
    )

    # --------------------------------------------------------
    # G. AUTHORITATIVE WINNER REPLAY
    # --------------------------------------------------------

    print()
    print("===== G. AUTHORITATIVE WINNER REPLAY — 120/120 =====")

    winner_scenario_rows = []

    winner_sums = {
        metric:
            0.0
        for metric in
        METRIC_NAMES
    }

    winner_counts = {
        metric:
            0
        for metric in
        METRIC_NAMES
    }

    maximum_winner_metric_delta = 0.0

    winner_j = (
        int(
            winner[
                "j_vehicle"
            ]
        ),
        int(
            winner[
                "j_pedestrian"
            ]
        ),
        int(
            winner[
                "j_cyclist"
            ]
        ),
    )

    for scenario_number, manifest_row in enumerate(
        manifest,
        start=1,
    ):

        scenario_id = str(
            manifest_row[
                "scenario_id"
            ]
        )

        with np.load(
            Path(
                manifest_row[
                    "npz_path"
                ]
            ),
            allow_pickle=False,
        ) as loaded:

            mask_vehicle = np.asarray(
                loaded[
                    "mask_vehicle"
                ],
                dtype=bool,
            )

            mask_pedestrian = np.asarray(
                loaded[
                    "mask_pedestrian"
                ],
                dtype=bool,
            )

            mask_cyclist = np.asarray(
                loaded[
                    "mask_cyclist"
                ],
                dtype=bool,
            )

            oracle_all = np.asarray(
                loaded[
                    "oracle_all"
                ],
                dtype=bool,
            )

            oracle_vehicle = np.asarray(
                loaded[
                    "oracle_vehicle"
                ],
                dtype=bool,
            )

            pedestrian_region = np.asarray(
                loaded[
                    "pedestrian_region"
                ],
                dtype=bool,
            )

            cyclist_region = np.asarray(
                loaded[
                    "cyclist_region"
                ],
                dtype=bool,
            )

            vehicle_surrogate = np.asarray(
                loaded[
                    "vehicle_surrogate"
                ],
                dtype=bool,
            )

        raw = authoritative_raw(
            mask_vehicle=
                mask_vehicle,

            mask_pedestrian=
                mask_pedestrian,

            mask_cyclist=
                mask_cyclist,

            f_vehicle=
                winner_j[
                    0
                ]
                /
                20.0,

            f_pedestrian=
                winner_j[
                    1
                ]
                /
                20.0,

            f_cyclist=
                winner_j[
                    2
                ]
                /
                20.0,
        )

        metrics = authoritative_metrics(
            raw,

            oracle_all=
                oracle_all,

            oracle_vehicle=
                oracle_vehicle,

            pedestrian_region=
                pedestrian_region,

            cyclist_region=
                cyclist_region,

            vehicle_surrogate=
                vehicle_surrogate,

            road_roi=
                road_roi,
        )

        row = {
            "scenario_number":
                scenario_number,

            "scenario_id":
                scenario_id,

            "j_vehicle":
                winner_j[
                    0
                ],

            "j_pedestrian":
                winner_j[
                    1
                ],

            "j_cyclist":
                winner_j[
                    2
                ],
        }

        for metric in METRIC_NAMES:

            value = float(
                metrics[
                    metric
                ]
            )

            row[
                metric
            ] = finite_or_none(
                value
            )

            if math.isfinite(
                value
            ):
                winner_sums[
                    metric
                ] += value

                winner_counts[
                    metric
                ] += 1

        winner_scenario_rows.append(
            row
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

    authoritative_winner_metrics = {}

    for metric in METRIC_NAMES:

        require(
            winner_counts[
                metric
            ]
            >
            0,
            (
                "Winner metric has zero finite "
                f"support: {metric}"
            ),
        )

        value = (
            winner_sums[
                metric
            ]
            /
            float(
                winner_counts[
                    metric
                ]
            )
        )

        authoritative_winner_metrics[
            metric
        ] = float(
            value
        )

        delta = abs(
            float(
                winner[
                    metric
                ]
            )
            -
            float(
                value
            )
        )

        require(
            delta
            <=
            PARITY_ABS_TOL,
            (
                "Winner authoritative aggregate "
                f"differs from sweep for {metric}\n"
                f"sweep={winner[metric]!r}\n"
                f"authority={value!r}\n"
                f"delta={delta!r}"
            ),
        )

        maximum_winner_metric_delta = max(
            maximum_winner_metric_delta,
            delta,
        )

    authority_key, authority_risk = candidate_key(
        authoritative_winner_metrics,
        winner,
    )

    require(
        tuple(
            winner_j
        )
        ==
        (
            int(
                authority_key[
                    4
                ]
            ),
            int(
                authority_key[
                    5
                ]
            ),
            int(
                authority_key[
                    6
                ]
            ),
        ),
        (
            "Winner identity changed during "
            "authoritative replay."
        ),
    )

    print(
        "winner authoritative replay = 120 / 120 PASS"
    )

    print(
        "winner max metric delta      =",
        repr(
            maximum_winner_metric_delta
        ),
    )

    print(
        "winner authoritative key     =",
        authority_key,
    )

    # --------------------------------------------------------
    # H. Post-selection regression
    # --------------------------------------------------------

    print()
    print("===== H. POST-SELECTION REGRESSION =====")

    tests_post = run_regression()

    print(
        "Stage6 regression =",
        f"{tests_post} / {tests_post} PASS",
    )

    # --------------------------------------------------------
    # I. Write frozen scores / winner
    # --------------------------------------------------------

    print()
    print("===== I. WRITE-ONCE STAGE-3 FLOOR FREEZE =====")

    write_once_jsonl(
        SCORES,
        score_rows,
    )

    write_once_jsonl(
        WINNER_SCENARIOS,
        winner_scenario_rows,
    )

    scores_sha = file_sha(
        SCORES
    )

    winner_scenarios_sha = file_sha(
        WINNER_SCENARIOS
    )

    selected_payload = {
        "j_vehicle":
            winner_j[
                0
            ],

        "j_pedestrian":
            winner_j[
                1
            ],

        "j_cyclist":
            winner_j[
                2
            ],

        "f_vehicle":
            winner_j[
                0
            ]
            /
            20.0,

        "f_pedestrian":
            winner_j[
                1
            ]
            /
            20.0,

        "f_cyclist":
            winner_j[
                2
            ]
            /
            20.0,

        "metrics":
            authoritative_winner_metrics,

        "risk_vector": {
            RISK_NAMES[
                index
            ]:
                float(
                    authority_risk[
                        index
                    ]
                )
            for index in
            range(
                5
            )
        },

        "selection_key": [
            float(
                authority_key[
                    0
                ]
            ),
            float(
                authority_key[
                    1
                ]
            ),
            float(
                authority_key[
                    2
                ]
            ),
            float(
                authority_key[
                    3
                ]
            ),
            int(
                authority_key[
                    4
                ]
            ),
            int(
                authority_key[
                    5
                ]
            ),
            int(
                authority_key[
                    6
                ]
            ),
        ],

        "finite_scenario_support": {
            metric:
                int(
                    winner_counts[
                        metric
                    ]
                )
            for metric in
            METRIC_NAMES
        },
    }

    freeze = {
        "stage":
            6,

        "block":
            "6.8_development_stage3_class_floors",

        "status":
            (
                "FROZEN_STAGE3_CLASS_FLOORS_"
                "BEFORE_STAGE4_TEMPORAL_RATE_SELECTION"
            ),

        "development_only":
            True,

        "formal_outcomes_read":
            False,

        "candidate_space": {
            "candidate_triplets":
                3270,

            "representation":
                "f_c = j_c / 20",

            "vehicle_j":
                [
                    0,
                    20,
                ],

            "pedestrian_j":
                [
                    1,
                    20,
                ],

            "cyclist_j":
                [
                    1,
                    20,
                ],

            "constraints": [
                "j_vehicle <= j_pedestrian",
                "j_vehicle <= j_cyclist",
                "j_pedestrian > 0",
                "j_cyclist > 0",
            ],
        },

        "selection_population": {
            "development_scenarios":
                120,

            "predictive_actors":
                877,

            "class_counts":
                EXPECTED_CLASS_COUNTS,

            "reactive_fallback_in_selection":
                False,

            "future_validity_as_selector":
                False,
        },

        "selection_semantics": {
            "terminal_illumination":
                (
                    "ClassAwareComposition."
                    "raw_class_aware_illumination"
                ),

            "temporal_smoothing":
                False,

            "actuation_rate_limit":
                False,

            "scenario_aggregation":
                (
                    "equal-weight arithmetic macro mean "
                    "over finite scenario-level metrics"
                ),

            "risk_vector":
                list(
                    RISK_NAMES
                ),

            "selection_key":
                (
                    "(max(risk_vector), mean(risk_vector), "
                    "over_masking_area, false_dimming, "
                    "j_vehicle, j_pedestrian, j_cyclist)"
                ),

            "epsilon_tolerance":
                None,

            "rounding":
                False,

            "weighted_sum":
                False,
        },

        "selected":
            selected_payload,

        "immutable_after_freeze": [
            "Stage-1 class k/gamma",
            "Stage-2 class margin coefficients",
            "Stage-3 vehicle intensity floor",
            "Stage-3 pedestrian intensity floor",
            "Stage-3 cyclist intensity floor",
        ],

        "execution_integrity": {
            "fixed_cache_parity":
                "PASS",

            "synthetic_metric_kernel_parity_max_abs_delta":
                synthetic_delta,

            "development_anchor_max_abs_delta":
                anchor_max_delta,

            "winner_authoritative_replay_scenarios":
                120,

            "winner_authoritative_max_abs_delta":
                maximum_winner_metric_delta,
        },

        "upstream_seals":
            seals,

        "outputs": {
            "candidate_scores": {
                "path":
                    str(
                        SCORES
                    ),

                "sha256":
                    scores_sha,

                "records":
                    3270,
            },

            "winner_scenarios": {
                "path":
                    str(
                        WINNER_SCENARIOS
                    ),

                "sha256":
                    winner_scenarios_sha,

                "records":
                    120,
            },
        },

        "scientific_boundary": {
            "P_occ_reopened":
                False,

            "WOMD_reopened":
                False,

            "Stage1_modified":
                False,

            "Stage2_modified":
                False,

            "Stage3_floor_selected":
                True,

            "Stage4_temporal_rate_selected":
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
        freeze,
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
        winner_scenarios_sha,
    )

    print(
        "Stage-3 freeze SHA   =",
        freeze_sha,
    )

    # --------------------------------------------------------
    # J. Final report
    # --------------------------------------------------------

    free_after = free_gib()

    require(
        free_after
        >=
        HARD_RESERVE_GIB,
        (
            "250-GiB storage reserve "
            f"violated: {free_after:.3f} GiB"
        ),
    )

    report = {
        "stage":
            6,

        "block":
            "6.8_development_stage3_class_floors",

        "status":
            "PASS_STAGE3_CLASS_FLOORS_SELECTED_FROZEN",

        "candidate_triplets":
            3270,

        "selected":
            selected_payload,

        "runner_up": {
            "j_vehicle":
                int(
                    runner_up[
                        "j_vehicle"
                    ]
                ),

            "j_pedestrian":
                int(
                    runner_up[
                        "j_pedestrian"
                    ]
                ),

            "j_cyclist":
                int(
                    runner_up[
                        "j_cyclist"
                    ]
                ),

            "selection_key":
                runner_up[
                    "selection_key"
                ],
        },

        "execution_integrity": {
            "anchor_max_abs_delta":
                anchor_max_delta,

            "winner_authoritative_max_abs_delta":
                maximum_winner_metric_delta,

            "winner_replay_scenarios":
                120,
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
                    winner_scenarios_sha,
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
            "Stage1_gamma_modified":
                False,

            "Stage2_margin_modified":
                False,

            "Stage3_floor_selected":
                True,

            "Stage4_temporal_rate_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,
        },

        "next":
            (
                "execute preregistered development-only "
                "Stage-4 temporal smoothing / actuation-rate "
                "selection with Stage1+2+3 immutable"
            ),
    }

    write_once_json(
        REPORT,
        report,
    )

    print()
    print("=" * 78)
    print("BLOCK 6.8 STAGE-3 EXECUTION PART 2/2 — FINAL")
    print("=" * 78)

    print(
        "fixed caches                = 120 / 120"
    )

    print(
        "candidate triplets          = 3270 / 3270"
    )

    print(
        "Stage-1 gamma               = IMMUTABLE"
    )

    print(
        "Stage-2 margins             = IMMUTABLE"
    )

    print(
        "vehicle floor j/f           =",
        winner_j[
            0
        ],
        winner_j[
            0
        ]
        /
        20.0,
    )

    print(
        "pedestrian floor j/f        =",
        winner_j[
            1
        ],
        winner_j[
            1
        ]
        /
        20.0,
    )

    print(
        "cyclist floor j/f           =",
        winner_j[
            2
        ],
        winner_j[
            2
        ]
        /
        20.0,
    )

    print(
        "risk max                    =",
        repr(
            authority_key[
                0
            ]
        ),
    )

    print(
        "risk mean                   =",
        repr(
            authority_key[
                1
            ]
        ),
    )

    print(
        "vehicle violation           =",
        repr(
            authoritative_winner_metrics[
                "vehicle_shadow_zone_violation"
            ]
        ),
    )

    print(
        "glare exposure              =",
        repr(
            authoritative_winner_metrics[
                "glare_risk_exposure"
            ]
        ),
    )

    print(
        "pedestrian visibility       =",
        repr(
            authoritative_winner_metrics[
                "pedestrian_visibility_proxy"
            ]
        ),
    )

    print(
        "cyclist visibility          =",
        repr(
            authoritative_winner_metrics[
                "cyclist_visibility_proxy"
            ]
        ),
    )

    print(
        "road illumination retention =",
        repr(
            authoritative_winner_metrics[
                "road_illumination_retention"
            ]
        ),
    )

    print(
        "over-masking area           =",
        repr(
            authoritative_winner_metrics[
                "over_masking_area"
            ]
        ),
    )

    print(
        "false dimming               =",
        repr(
            authoritative_winner_metrics[
                "false_dimming"
            ]
        ),
    )

    print(
        "anchor execution parity     = PASS"
    )

    print(
        "winner authoritative replay = 120 / 120 PASS"
    )

    print(
        "Stage-3 floors              = FROZEN"
    )

    print(
        "Stage-4 temporal/rate       = NOT SELECTED"
    )

    print(
        "primary acceptance gate     = NOT TESTED"
    )

    print(
        "NI bounds                   = IMMUTABLE / UNCHANGED"
    )

    print(
        "formal evaluation           = NO"
    )

    print(
        "Stage6 regression           =",
        f"{tests_post} / {tests_post} PASS",
    )

    print(
        "free GiB                    =",
        f"{free_after:.3f}",
    )

    print(
        "STATUS = "
        "PASS_STAGE3_CLASS_FLOORS_SELECTED_FROZEN"
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
    print("=" * 78)
    print(
        "BLOCK 6.8 STAGE-3 EXECUTION PART 2/2 = BLOCKED"
    )
    print("=" * 78)

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
        "Stage-1 gamma           = STILL FROZEN"
    )

    print(
        "Stage-2 margins         = STILL FROZEN"
    )

    print(
        "Stage-3 floor freeze    = NOT COMPLETE "
        "unless final PASS report exists"
    )

    print(
        "Stage-4 temporal/rate   = NOT SELECTED"
    )

    print(
        "primary acceptance      = NOT TESTED"
    )

    print(
        "NI bounds               = UNCHANGED"
    )

    print(
        "formal evaluation       = NO"
    )

    print()
    print(
        "Do not retune Stage1/2 or alter "
        "the frozen Stage3 scoring comparator."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

from __future__ import annotations

from collections import Counter
import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

# ============================================================
# Frozen inputs
# ============================================================

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

ELIGIBLE = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

POSTERIOR = (
    S6
    / "artifacts/block66/"
      "block66_development_identity_safe_gaussian_posterior.jsonl"
)

PART2C1_HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_to_part2c2_handoff.json"
)

PART2C2A_REPORT = (
    S6
    / "reports/"
      "block66_part2c2a_probabilistic_kernel_binding_audit.json"
)

NUMERIC_FREEZE = (
    S6
    / "configs/"
      "block64_mc_grid_numeric_freeze.json"
)

FULLBOX_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_full_box.py"
)

OCCUPANCY_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "probabilistic_occupancy.py"
)

WOMD_GEOMETRY_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "womd_geometry.py"
)

GEOMETRY_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "geometry.py"
)


# ============================================================
# New outputs
# ============================================================

PERSISTENCE_FREEZE = (
    S6
    / "configs/"
      "block66_part2c2b_pocc_persistence_contract.json"
)

CACHE_DIR = (
    S6
    / "artifacts/block66/"
      "part2c2b_actor_pocc_cache"
)

MANIFEST = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block66_part2c2b_eligible_n8192_pocc.json"
)

FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_pocc_freeze_manifest.json"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part2c2b_to_part3_handoff.json"
)


EXPECTED = {
    "cohort":
        (
            "57440e3d976e7aeff44811b5eeb075dc"
            "56e0e0e070a3832ec72d354253b998c9"
        ),

    "eligible":
        (
            "d7ce8769deee08286974b0327cdfd55c"
            "08ad7ea326cf9d6450a48a82b3e251b6"
        ),

    "posterior":
        (
            "f022471fbf86c3121106011bfd7f035c5"
            "d895f2c5aeeaa2fa3a20c04b3439acd"
        ),

    "part2c1_handoff":
        (
            "9610565cadb8a21622ffc82557e5d0e9"
            "ce0279b4796ecfba158eb5e808d3f175"
        ),

    "part2c2a":
        (
            "4c79d9ba8084aa3cfbfed03945bf9897"
            "03769feb5303827844413c2ee71ac0dc"
        ),

    "numeric_freeze":
        (
            "993c4248a902e7dff3a4383ac343e722"
            "2cc43ef9b9372ed2efd0ee08d6d20a73"
        ),

    "fullbox_module":
        (
            "51daeab95713d6e7d2738ced466be4844"
            "b73dce70a09138b2b769de880462995"
        ),

    "occupancy_module":
        (
            "d4206761fc70495563d53a6ab58aefe6"
            "db5d8bd1b2d09736bfc5bbc6b6f50e3e"
        ),

    "womd_geometry_module":
        (
            "b90f47289bd091474658e3864c4b1ccc"
            "8c2fc841aaa38160f38f0b95ece7187c"
        ),

    "geometry_module":
        (
            "f6e6cc0dd362338eeeaae9d8e2594743"
            "4a3e7c64b350f0baf81a79a46159f94e"
        ),
}


N_MC = 8192
MC_SEED = 20260821

GRID_SHAPE = (
    501,
    301,
)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

EXPECTED_CLASS_COUNTS = {
    "TYPE_VEHICLE":
        719,

    "TYPE_PEDESTRIAN":
        110,

    "TYPE_CYCLIST":
        48,
}

CORE_CLASSES = tuple(
    EXPECTED_CLASS_COUNTS
)

EXPECTED_TOTAL = 877

MIN_FREE_GIB = 250.0


class ControlledBlock(RuntimeError):
    pass


# ============================================================
# Generic helpers
# ============================================================

def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(condition):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as stream:
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


def free_gib() -> float:
    return (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )


def canonical_json_bytes(
    payload: Any,
) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def pretty_json_bytes(
    payload: Any,
) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def write_json_exact_or_create(
    path: Path,
    payload: Any,
) -> str:
    data = pretty_json_bytes(
        payload
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing JSON differs from "
                f"deterministic rerun: {path}"
            ),
        )

    else:
        path.write_bytes(
            data
        )

    return sha256_file(
        path
    )


def write_jsonl_exact_or_create(
    path: Path,
    rows,
) -> str:
    data = b"".join(
        canonical_json_bytes(
            row
        )
        for row in rows
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing JSONL differs from "
                f"deterministic rerun: {path}"
            ),
        )

    else:
        path.write_bytes(
            data
        )

    return sha256_file(
        path
    )


def atomic_json_write(
    path: Path,
    payload: Any,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pretty_json_bytes(
        payload
    )

    tmp = path.with_name(
        path.name
        +
        ".tmp"
    )

    tmp.write_bytes(
        data
    )

    os.replace(
        tmp,
        path,
    )

    return sha256_file(
        path
    )


def atomic_npy_write(
    path: Path,
    array: np.ndarray,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        +
        ".tmp"
    )

    with tmp.open(
        "wb"
    ) as handle:
        np.save(
            handle,
            array,
            allow_pickle=False,
        )

    os.replace(
        tmp,
        path,
    )

    return sha256_file(
        path
    )


def read_jsonl(
    path: Path,
):
    return [
        json.loads(
            line
        )
        for line in
        path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def safe_token(
    value: str,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(
            value
        ),
    )


def cache_paths(
    actor_rank: int,
    scenario_id: str,
    prediction_id: str,
):
    stem = (
        f"actor_{actor_rank:04d}__"
        f"{safe_token(scenario_id)}__"
        f"{safe_token(prediction_id)}"
    )

    return (
        CACHE_DIR
        /
        (
            stem
            +
            ".counts.npy"
        ),

        CACHE_DIR
        /
        (
            stem
            +
            ".json"
        ),
    )


def tuple3(
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
            3,
        ),
        (
            "Expected [3], got "
            f"{array.shape}."
        ),
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        "Non-finite Vec3.",
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
        "Non-finite [4,3] value.",
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
        "Non-finite covariance.",
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


def fresh_box_payload(
    actor_box,
):
    box = actor_box.box

    return {
        "center_xyz_H0_m":
            [
                float(value)
                for value in
                box.center_xyz
            ],

        "length_m":
            float(
                box.length_m
            ),

        "width_m":
            float(
                box.width_m
            ),

        "height_m":
            float(
                box.height_m
            ),

        "yaw_rad":
            float(
                box.yaw_rad
            ),
    }


def summary_from_counts(
    counts: np.ndarray,
):
    counts = np.asarray(
        counts
    )

    require(
        counts.dtype
        ==
        np.uint32,
        (
            "Canonical persisted counts "
            f"must be uint32, got {counts.dtype}."
        ),
    )

    require(
        counts.shape
        ==
        (
            4,
            *GRID_SHAPE,
        ),
        (
            "Occupancy count shape changed: "
            f"{counts.shape}"
        ),
    )

    require(
        int(
            np.max(
                counts
            )
        )
        <=
        N_MC,
        (
            "Occupancy count exceeds N_MC."
        ),
    )

    nonzero_cells = []

    maximum_count = []

    probability_mass = []

    count_mass = []

    for horizon_index in range(
        4
    ):
        horizon_counts = counts[
            horizon_index
        ]

        nonzero_cells.append(
            int(
                np.count_nonzero(
                    horizon_counts
                )
            )
        )

        maximum_count.append(
            int(
                np.max(
                    horizon_counts
                )
            )
        )

        mass = int(
            np.sum(
                horizon_counts,
                dtype=np.uint64,
            )
        )

        count_mass.append(
            mass
        )

        probability_mass.append(
            float(
                mass
                /
                N_MC
            )
        )

    return {
        "nonzero_cells_by_horizon":
            nonzero_cells,

        "maximum_count_by_horizon":
            maximum_count,

        "maximum_probability_by_horizon":
            [
                float(
                    value
                    /
                    N_MC
                )
                for value in maximum_count
            ],

        "occupancy_count_mass_by_horizon":
            count_mass,

        "occupancy_probability_mass_by_horizon":
            probability_mass,

        "has_any_support":
            bool(
                any(
                    value > 0
                    for value in
                    nonzero_cells
                )
            ),
    }


def validate_cached_actor(
    *,
    metadata_path: Path,
    counts_path: Path,
    expected_identity: dict,
    persistence_sha: str,
):
    require(
        metadata_path.is_file(),
        (
            "Metadata exists incompletely or "
            f"is missing: {metadata_path}"
        ),
    )

    require(
        counts_path.is_file(),
        (
            "Cached counts missing: "
            f"{counts_path}"
        ),
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        metadata.get(
            "status"
        )
        ==
        "PASS_ACTOR_N8192_P_OCC",
        (
            "Cached actor does not have "
            "PASS status."
        ),
    )

    for key, value in (
        expected_identity.items()
    ):
        require(
            metadata.get(
                key
            )
            ==
            value,
            (
                "Cached actor identity "
                f"mismatch for {key}."
            ),
        )

    require(
        metadata[
            "frozen_seals"
        ][
            "persistence_contract_sha256"
        ]
        ==
        persistence_sha,
        (
            "Cached persistence-contract "
            "SHA mismatch."
        ),
    )

    require(
        metadata[
            "frozen_seals"
        ][
            "eligible_predictive_sha256"
        ]
        ==
        EXPECTED[
            "eligible"
        ],
        (
            "Cached eligible-manifest "
            "SHA mismatch."
        ),
    )

    require(
        metadata[
            "frozen_seals"
        ][
            "posterior_sha256"
        ]
        ==
        EXPECTED[
            "posterior"
        ],
        (
            "Cached posterior SHA mismatch."
        ),
    )

    require(
        sha256_file(
            counts_path
        )
        ==
        metadata[
            "occupancy_counts"
        ][
            "sha256"
        ],
        (
            "Cached occupancy-count "
            "file hash mismatch."
        ),
    )

    counts = np.load(
        counts_path,
        allow_pickle=False,
    )

    summary = summary_from_counts(
        counts
    )

    require(
        summary
        ==
        metadata[
            "summary"
        ],
        (
            "Cached occupancy summary "
            "does not reproduce."
        ),
    )

    return metadata


def run_regression():
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
                    match.group(1)
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


# ============================================================
# Main
# ============================================================

def main():
    start = time.perf_counter()

    print("=" * 78)
    print(
        "BLOCK 6.6 PART2C/2B — RESUMABLE "
        "HEADLAMP-ELIGIBLE N8192 P_OCC MATERIALIZATION"
    )
    print("=" * 78)

    # ========================================================
    # A. Immutable scientific seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE SCIENTIFIC SEAL ====="
    )

    for path, expected, label in (
        (
            COHORT,
            EXPECTED[
                "cohort"
            ],
            "development cohort",
        ),
        (
            ELIGIBLE,
            EXPECTED[
                "eligible"
            ],
            "eligible predictive manifest",
        ),
        (
            POSTERIOR,
            EXPECTED[
                "posterior"
            ],
            "identity-safe posterior",
        ),
        (
            PART2C1_HANDOFF,
            EXPECTED[
                "part2c1_handoff"
            ],
            "Part2C/1 handoff",
        ),
        (
            PART2C2A_REPORT,
            EXPECTED[
                "part2c2a"
            ],
            "Part2C/2A API audit",
        ),
        (
            NUMERIC_FREEZE,
            EXPECTED[
                "numeric_freeze"
            ],
            "Block6.4 numeric freeze",
        ),
        (
            FULLBOX_MODULE,
            EXPECTED[
                "fullbox_module"
            ],
            "probabilistic full-box module",
        ),
        (
            OCCUPANCY_MODULE,
            EXPECTED[
                "occupancy_module"
            ],
            "probabilistic occupancy module",
        ),
        (
            WOMD_GEOMETRY_MODULE,
            EXPECTED[
                "womd_geometry_module"
            ],
            "WOMD geometry module",
        ),
        (
            GEOMETRY_MODULE,
            EXPECTED[
                "geometry_module"
            ],
            "Stage6 geometry module",
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
            actual
            ==
            expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:32s} = EXACT PASS"
        )

    require(
        free_gib()
        >=
        MIN_FREE_GIB,
        (
            "250-GiB storage reserve "
            "violated before sampling."
        ),
    )

    print(
        "storage reserve = PASS"
    )

    # ========================================================
    # B. Freeze persistence BEFORE first actor sampling
    # ========================================================

    print()
    print(
        "===== B. P_OCC PERSISTENCE CONTRACT ====="
    )

    persistence_payload = {
        "stage":
            6,

        "block":
            "6.6-Part2C/2B",

        "status":
            (
                "FROZEN_BEFORE_FIRST_"
                "PART2C2B_ACTOR_SAMPLE"
            ),

        "canonical_persisted_statistic":
            "occupancy_counts",

        "dtype":
            "uint32",

        "shape_per_actor":
            [
                4,
                501,
                301,
            ],

        "probability_definition":
            (
                "P_occ = occupancy_counts / 8192"
            ),

        "probability_persisted_separately":
            False,

        "probability_reconstruction_lossless":
            True,

        "lossless_reason":
            (
                "public_kernel_returns_uint32_counts_"
                "and_8192_equals_2_pow_13"
            ),

        "compression":
            "NONE_NPY",

        "scientific_kernel_modified":
            False,

        "quantization":
            False,

        "uint16_downcast":
            False,

        "sample_count":
            N_MC,

        "seed":
            MC_SEED,

        "same_seed_per_actor":
            True,

        "grid_shape":
            [
                501,
                301,
            ],

        "gamma_applied":
            False,

        "class_policy_applied":
            False,

        "formal_outcomes_read":
            False,
    }

    persistence_sha = (
        write_json_exact_or_create(
            PERSISTENCE_FREEZE,
            persistence_payload,
        )
    )

    print(
        "persistence contract =",
        PERSISTENCE_FREEZE,
    )

    print(
        "persistence SHA256 =",
        persistence_sha,
    )

    print(
        "persist dtype = uint32"
    )

    print(
        "P_occ reconstruction = counts / 8192"
    )

    print(
        "float64 probability file = NOT STORED"
    )

    print(
        "gamma/class policy = NOT APPLIED"
    )

    # ========================================================
    # C. Frozen actor/posterior binding
    # ========================================================

    print()
    print(
        "===== C. ELIGIBLE ACTOR / POSTERIOR BINDING ====="
    )

    cohort = read_jsonl(
        COHORT
    )

    eligible = read_jsonl(
        ELIGIBLE
    )

    posterior = read_jsonl(
        POSTERIOR
    )

    require(
        len(
            cohort
        )
        ==
        120,
        "Cohort N changed.",
    )

    require(
        len(
            eligible
        )
        ==
        EXPECTED_TOTAL,
        (
            "Eligible actor count "
            f"changed: {len(eligible)}."
        ),
    )

    posterior_index = {}

    for row in posterior:
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
        )

        require(
            key
            not in
            posterior_index,
            (
                "Duplicate posterior key: "
                f"{key}"
            ),
        )

        posterior_index[
            key
        ] = row

    class_counts = Counter(
        str(
            row[
                "object_type"
            ]
        )
        for row in eligible
    )

    actual_class_counts = {
        name:
            int(
                class_counts[
                    name
                ]
            )
        for name in CORE_CLASSES
    }

    require(
        actual_class_counts
        ==
        EXPECTED_CLASS_COUNTS,
        (
            "Eligible class census "
            f"changed: {actual_class_counts}"
        ),
    )

    for rank, row in enumerate(
        eligible
    ):
        require(
            int(
                row[
                    "cohort_index"
                ]
            )
            in
            range(
                120
            ),
            (
                "Invalid cohort index."
            ),
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
        )

        require(
            key
            in
            posterior_index,
            (
                "Eligible actor lacks "
                f"posterior: {key}"
            ),
        )

    print(
        "eligible actors = 877"
    )

    print(
        "eligible by class =",
        actual_class_counts,
    )

    print(
        "posterior join = PASS | 877/877"
    )

    # ========================================================
    # D. Exact frozen public kernels / grid
    # ========================================================

    print()
    print(
        "===== D. FROZEN PUBLIC PROBABILISTIC KERNELS ====="
    )

    from iscai_stage4.data.real_pipeline import (
        read_training_scenario,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    from iscai_stage6.adb.probabilistic_full_box import (
        CalibratedGaussianFullBoxPrediction,
        build_stochastic_future_full_boxes,
    )

    from iscai_stage6.adb.probabilistic_occupancy import (
        OccupancyGrid,
        estimate_actor_occupancy_probability,
    )

    numeric = json.loads(
        NUMERIC_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    require(
        int(
            numeric[
                "MC"
            ][
                "runtime_sample_count"
            ]
        )
        ==
        N_MC,
        "Frozen N_MC changed.",
    )

    require(
        int(
            numeric[
                "MC"
            ][
                "runtime_seed"
            ]
        )
        ==
        MC_SEED,
        "Frozen MC seed changed.",
    )

    require(
        numeric[
            "scientific_illumination_grid"
        ][
            "shape"
        ]
        ==
        [
            501,
            301,
        ],
        "Frozen grid shape changed.",
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
            for x in theta_rad
        ),

        range_centers_m=tuple(
            float(x)
            for x in range_m
        ),
    )

    require(
        tuple(
            grid.shape
        )
        ==
        GRID_SHAPE,
        (
            "Constructed occupancy grid "
            "shape changed."
        ),
    )

    print(
        "full-box kernel = "
        "build_stochastic_future_full_boxes"
    )

    print(
        "occupancy kernel = "
        "estimate_actor_occupancy_probability"
    )

    print(
        "N_MC / seed = 8192 / 20260821"
    )

    print(
        "grid = 501 x 301"
    )

    print(
        "same seed per actor = YES"
    )

    # ========================================================
    # E. Resumable N8192 actor materialization
    # ========================================================

    print()
    print(
        "===== E. RESUMABLE N8192 ACTOR MATERIALIZATION ====="
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    executed = 0
    reused = 0

    current_scene_index = None
    current_boxes = None
    current_scenario_id = None

    actor_metadata = []

    for actor_rank, eligible_row in enumerate(
        eligible
    ):
        scenario_id = str(
            eligible_row[
                "scenario_id"
            ]
        )

        prediction_id = str(
            eligible_row[
                "prediction_id"
            ]
        )

        object_type = str(
            eligible_row[
                "object_type"
            ]
        )

        cohort_index = int(
            eligible_row[
                "cohort_index"
            ]
        )

        actor_box_index = int(
            eligible_row[
                "actor_box_index"
            ]
        )

        counts_path, metadata_path = (
            cache_paths(
                actor_rank,
                scenario_id,
                prediction_id,
            )
        )

        expected_identity = {
            "actor_rank":
                int(
                    actor_rank
                ),

            "cohort_index":
                cohort_index,

            "scenario_id":
                scenario_id,

            "prediction_id":
                prediction_id,

            "actor_box_index":
                actor_box_index,

            "object_type":
                object_type,
        }

        if (
            metadata_path.is_file()
            or
            counts_path.is_file()
        ):
            metadata = (
                validate_cached_actor(
                    metadata_path=metadata_path,
                    counts_path=counts_path,
                    expected_identity=(
                        expected_identity
                    ),
                    persistence_sha=(
                        persistence_sha
                    ),
                )
            )

            actor_metadata.append(
                metadata
            )

            reused += 1

            if (
                (actor_rank + 1)
                %
                10
                ==
                0
                or
                actor_rank == 0
                or
                actor_rank + 1
                ==
                EXPECTED_TOTAL
            ):
                print(
                    f"[{actor_rank+1:03d}/877] "
                    f"{object_type:15s} "
                    f"{scenario_id} "
                    f"{prediction_id} "
                    "= CACHE PASS"
                )

            continue

        actor_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # E1. Rebuild exact causal actor box once per scene
        # ----------------------------------------------------

        if (
            current_scene_index
            !=
            cohort_index
        ):
            source_row = cohort[
                cohort_index
            ]

            require(
                str(
                    source_row[
                        "scenario_id"
                    ]
                )
                ==
                scenario_id,
                (
                    "Eligible cohort index "
                    "does not resolve scenario."
                ),
            )

            scenario = (
                read_training_scenario(
                    source_row
                )
            )

            require(
                str(
                    scenario.scenario_id
                )
                ==
                scenario_id,
                (
                    "Random-access scenario "
                    "ID mismatch."
                ),
            )

            adapted = (
                adapt_causal_womd_scenario(
                    scenario
                )
            )

            current_boxes = tuple(
                build_causal_adb_actor_boxes(
                    scenario=scenario,
                    adapted=adapted,
                )
            )

            current_scene_index = (
                cohort_index
            )

            current_scenario_id = (
                scenario_id
            )

        require(
            current_scenario_id
            ==
            scenario_id,
            (
                "Internal scenario cache "
                "identity mismatch."
            ),
        )

        require(
            0
            <=
            actor_box_index
            <
            len(
                current_boxes
            ),
            (
                "actor_box_index out of range."
            ),
        )

        actor_box = current_boxes[
            actor_box_index
        ]

        require(
            str(
                actor_box.object_type
            )
            ==
            object_type,
            (
                "Fresh actor-box class "
                "does not match eligible row."
            ),
        )

        require(
            fresh_box_payload(
                actor_box
            )
            ==
            eligible_row[
                "current_box_H0"
            ],
            (
                "Fresh causal full box does "
                "not exactly reproduce frozen "
                "Part2C/1 geometry."
            ),
        )

        require(
            actor_box.state_source
            ==
            "current_causal",
            (
                "Actor box is not "
                "current causal."
            ),
        )

        require(
            actor_box.orientation_source
            ==
            "current_causal",
            (
                "Actor orientation is not "
                "current causal."
            ),
        )

        require(
            not bool(
                actor_box.future_state_used
            ),
            (
                "Future state entered "
                "actor box."
            ),
        )

        require(
            not bool(
                actor_box.tracks_to_predict_used
            ),
            (
                "tracks_to_predict entered "
                "actor box."
            ),
        )

        require(
            not bool(
                actor_box.objects_of_interest_used
            ),
            (
                "objects_of_interest entered "
                "actor box."
            ),
        )

        # ----------------------------------------------------
        # E2. Exact posterior object
        # ----------------------------------------------------

        posterior_row = posterior_index[
            (
                scenario_id,
                prediction_id,
            )
        ]

        require(
            posterior_row[
                "future_GT_used"
            ]
            is False,
            (
                "Posterior future_GT flag "
                "changed."
            ),
        )

        require(
            posterior_row[
                "truth_id_used"
            ]
            is False,
            (
                "Posterior truth ID flag "
                "changed."
            ),
        )

        require(
            posterior_row[
                "tracks_to_predict_used"
            ]
            is False,
            (
                "Posterior tracks_to_predict "
                "flag changed."
            ),
        )

        prediction = (
            CalibratedGaussianFullBoxPrediction(
                prediction_id=(
                    prediction_id
                ),

                latest_position_H0_m=(
                    tuple3(
                        posterior_row[
                            "latest_position_H0_m"
                        ]
                    )
                ),

                mean_displacement_H0_m=(
                    tuple4x3(
                        posterior_row[
                            "mean_displacement_H0_m"
                        ]
                    )
                ),

                calibrated_predictive_covariance_H0_m2=(
                    tuple4x3x3(
                        posterior_row[
                            "calibrated_predictive_covariance_H0_m2"
                        ]
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # E3. Frozen stochastic full-box + public P_occ kernel
        # ----------------------------------------------------

        forecast = (
            build_stochastic_future_full_boxes(
                prediction,
                actor_box,
                sample_count=N_MC,
                seed=MC_SEED,
            )
        )

        require(
            str(
                forecast.prediction_id
            )
            ==
            prediction_id,
            (
                "Forecast prediction ID "
                "changed."
            ),
        )

        require(
            int(
                forecast.sample_count
            )
            ==
            N_MC,
            (
                "Forecast sample count "
                "changed."
            ),
        )

        occupancy = (
            estimate_actor_occupancy_probability(
                forecast,
                grid,
            )
        )

        require(
            str(
                occupancy.prediction_id
            )
            ==
            prediction_id,
            (
                "Occupancy prediction ID "
                "changed."
            ),
        )

        require(
            int(
                occupancy.sample_count
            )
            ==
            N_MC,
            (
                "Occupancy sample count "
                "changed."
            ),
        )

        counts = np.asarray(
            occupancy.occupancy_counts
        )

        require(
            counts.dtype
            ==
            np.uint32,
            (
                "Public occupancy-count dtype "
                f"changed: {counts.dtype}"
            ),
        )

        require(
            counts.shape
            ==
            (
                4,
                501,
                301,
            ),
            (
                "Public occupancy-count "
                f"shape changed: {counts.shape}"
            ),
        )

        probability = np.asarray(
            occupancy.occupancy_probability,
            dtype=np.float64,
        )

        require(
            probability.shape
            ==
            counts.shape,
            (
                "Public P_occ shape "
                "changed."
            ),
        )

        # Exact public probability definition.
        reconstructed_probability = (
            counts.astype(
                np.float64
            )
            /
            float(
                N_MC
            )
        )

        require(
            np.array_equal(
                probability,
                reconstructed_probability,
            ),
            (
                "Public P_occ is not exactly "
                "occupancy_counts / 8192."
            ),
        )

        # ----------------------------------------------------
        # E4. Persist only canonical uint32 counts
        # ----------------------------------------------------

        counts_sha = (
            atomic_npy_write(
                counts_path,
                counts,
            )
        )

        summary = summary_from_counts(
            counts
        )

        metadata = {
            "status":
                "PASS_ACTOR_N8192_P_OCC",

            **expected_identity,

            "sample_count":
                N_MC,

            "seed":
                MC_SEED,

            "same_seed_per_actor":
                True,

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "grid_shape":
                [
                    501,
                    301,
                ],

            "posterior_family":
                "calibrated_Gaussian_GRU",

            "full_box_semantics":
                (
                    "all_eight_physical_box_corners_"
                    "samplewise_predicted_tangent_heading"
                ),

            "rasterization_semantics":
                str(
                    occupancy.rasterization_semantics
                ),

            "occupancy_counts": {
                "path":
                    str(
                        counts_path
                    ),

                "sha256":
                    counts_sha,

                "dtype":
                    "uint32",

                "shape":
                    [
                        4,
                        501,
                        301,
                    ],
            },

            "probability": {
                "persisted_separately":
                    False,

                "exact_reconstruction":
                    "occupancy_counts / 8192",

                "public_probability_parity":
                    "EXACT_PASS",
            },

            "summary":
                summary,

            "frozen_seals": {
                "eligible_predictive_sha256":
                    EXPECTED[
                        "eligible"
                    ],

                "posterior_sha256":
                    EXPECTED[
                        "posterior"
                    ],

                "Part2C2A_report_sha256":
                    EXPECTED[
                        "part2c2a"
                    ],

                "fullbox_module_sha256":
                    EXPECTED[
                        "fullbox_module"
                    ],

                "occupancy_module_sha256":
                    EXPECTED[
                        "occupancy_module"
                    ],

                "numeric_freeze_sha256":
                    EXPECTED[
                        "numeric_freeze"
                    ],

                "persistence_contract_sha256":
                    persistence_sha,
            },

            "causality": {
                "future_GT":
                    False,

                "truth_ID":
                    False,

                "tracks_to_predict":
                    False,

                "objects_of_interest":
                    False,

                "centroid_only":
                    False,
            },

            "policy_boundary": {
                "gamma_applied":
                    False,

                "class_policy_applied":
                    False,

                "reactive_fallback_changed":
                    False,

                "formal_evaluation":
                    False,
            },
        }

        atomic_json_write(
            metadata_path,
            metadata,
        )

        # Verify fresh persistence immediately.
        metadata = validate_cached_actor(
            metadata_path=metadata_path,
            counts_path=counts_path,
            expected_identity=(
                expected_identity
            ),
            persistence_sha=(
                persistence_sha
            ),
        )

        actor_metadata.append(
            metadata
        )

        executed += 1

        elapsed = (
            time.perf_counter()
            -
            actor_start
        )

        print(
            f"[{actor_rank+1:03d}/877] "
            f"{object_type:15s} "
            f"{scenario_id} "
            f"{prediction_id} | "
            f"support="
            f"{summary['nonzero_cells_by_horizon']} | "
            f"{elapsed:.2f}s"
        )

        del (
            forecast,
            occupancy,
            counts,
            probability,
            reconstructed_probability,
            prediction,
        )

        gc.collect()

        require(
            free_gib()
            >=
            MIN_FREE_GIB,
            (
                "250-GiB storage reserve "
                "violated during sampling."
            ),
        )

    # ========================================================
    # F. Complete-cache verification
    # ========================================================

    print()
    print(
        "===== F. COMPLETE 877-ACTOR CACHE VERIFICATION ====="
    )

    require(
        len(
            actor_metadata
        )
        ==
        EXPECTED_TOTAL,
        (
            "Expected metadata for "
            "877 actors."
        ),
    )

    expected_json_paths = set()
    expected_count_paths = set()

    fully_verified = []

    for actor_rank, eligible_row in enumerate(
        eligible
    ):
        scenario_id = str(
            eligible_row[
                "scenario_id"
            ]
        )

        prediction_id = str(
            eligible_row[
                "prediction_id"
            ]
        )

        counts_path, metadata_path = (
            cache_paths(
                actor_rank,
                scenario_id,
                prediction_id,
            )
        )

        expected_json_paths.add(
            metadata_path.resolve()
        )

        expected_count_paths.add(
            counts_path.resolve()
        )

        expected_identity = {
            "actor_rank":
                int(
                    actor_rank
                ),

            "cohort_index":
                int(
                    eligible_row[
                        "cohort_index"
                    ]
                ),

            "scenario_id":
                scenario_id,

            "prediction_id":
                prediction_id,

            "actor_box_index":
                int(
                    eligible_row[
                        "actor_box_index"
                    ]
                ),

            "object_type":
                str(
                    eligible_row[
                        "object_type"
                    ]
                ),
        }

        metadata = validate_cached_actor(
            metadata_path=metadata_path,
            counts_path=counts_path,
            expected_identity=(
                expected_identity
            ),
            persistence_sha=(
                persistence_sha
            ),
        )

        fully_verified.append(
            metadata
        )

    actual_json_paths = {
        path.resolve()
        for path in
        CACHE_DIR.glob(
            "actor_*.json"
        )
    }

    actual_count_paths = {
        path.resolve()
        for path in
        CACHE_DIR.glob(
            "actor_*.counts.npy"
        )
    }

    require(
        actual_json_paths
        ==
        expected_json_paths,
        (
            "Actor-cache metadata file set "
            "differs from expected 877."
        ),
    )

    require(
        actual_count_paths
        ==
        expected_count_paths,
        (
            "Actor-cache count-file set "
            "differs from expected 877."
        ),
    )

    print(
        "actor caches = PASS | 877/877"
    )

    print(
        "executed this run =",
        executed,
    )

    print(
        "reused this run =",
        reused,
    )

    # ========================================================
    # G. Deterministic consolidation
    # ========================================================

    print()
    print(
        "===== G. DETERMINISTIC P_OCC CONSOLIDATION ====="
    )

    manifest_rows = []

    aggregate = {}

    for object_type in CORE_CLASSES:
        aggregate[
            object_type
        ] = {
            "actor_count":
                0,

            "actors_with_any_support":
                0,

            "actors_with_zero_support":
                0,

            "actors_with_support_by_horizon":
                [
                    0,
                    0,
                    0,
                    0,
                ],

            "support_cells_sum_by_horizon":
                [
                    0,
                    0,
                    0,
                    0,
                ],

            "occupancy_count_mass_sum_by_horizon":
                [
                    0,
                    0,
                    0,
                    0,
                ],

            "maximum_count_observed_by_horizon":
                [
                    0,
                    0,
                    0,
                    0,
                ],
        }

    global_manifest_digest = sha256()

    for metadata in fully_verified:
        object_type = str(
            metadata[
                "object_type"
            ]
        )

        summary = metadata[
            "summary"
        ]

        target = aggregate[
            object_type
        ]

        target[
            "actor_count"
        ] += 1

        if summary[
            "has_any_support"
        ]:
            target[
                "actors_with_any_support"
            ] += 1
        else:
            target[
                "actors_with_zero_support"
            ] += 1

        for h in range(
            4
        ):
            if (
                int(
                    summary[
                        "nonzero_cells_by_horizon"
                    ][
                        h
                    ]
                )
                >
                0
            ):
                target[
                    "actors_with_support_by_horizon"
                ][
                    h
                ] += 1

            target[
                "support_cells_sum_by_horizon"
            ][
                h
            ] += int(
                summary[
                    "nonzero_cells_by_horizon"
                ][
                    h
                ]
            )

            target[
                "occupancy_count_mass_sum_by_horizon"
            ][
                h
            ] += int(
                summary[
                    "occupancy_count_mass_by_horizon"
                ][
                    h
                ]
            )

            target[
                "maximum_count_observed_by_horizon"
            ][
                h
            ] = max(
                int(
                    target[
                        "maximum_count_observed_by_horizon"
                    ][
                        h
                    ]
                ),
                int(
                    summary[
                        "maximum_count_by_horizon"
                    ][
                        h
                    ]
                ),
            )

        manifest_row = {
            "actor_rank":
                int(
                    metadata[
                        "actor_rank"
                    ]
                ),

            "cohort_index":
                int(
                    metadata[
                        "cohort_index"
                    ]
                ),

            "scenario_id":
                str(
                    metadata[
                        "scenario_id"
                    ]
                ),

            "prediction_id":
                str(
                    metadata[
                        "prediction_id"
                    ]
                ),

            "actor_box_index":
                int(
                    metadata[
                        "actor_box_index"
                    ]
                ),

            "object_type":
                object_type,

            "occupancy_counts_path":
                metadata[
                    "occupancy_counts"
                ][
                    "path"
                ],

            "occupancy_counts_sha256":
                metadata[
                    "occupancy_counts"
                ][
                    "sha256"
                ],

            "sample_count":
                N_MC,

            "seed":
                MC_SEED,

            "summary":
                summary,
        }

        manifest_rows.append(
            manifest_row
        )

        global_manifest_digest.update(
            canonical_json_bytes(
                manifest_row
            )
        )

    actual_aggregate_counts = {
        object_type:
            int(
                aggregate[
                    object_type
                ][
                    "actor_count"
                ]
            )
        for object_type in CORE_CLASSES
    }

    require(
        actual_aggregate_counts
        ==
        EXPECTED_CLASS_COUNTS,
        (
            "Consolidated class count "
            f"changed: {actual_aggregate_counts}"
        ),
    )

    manifest_sha = (
        write_jsonl_exact_or_create(
            MANIFEST,
            manifest_rows,
        )
    )

    print(
        "actors by class =",
        actual_aggregate_counts,
    )

    print()
    print(
        "classwise P_occ support:"
    )

    for object_type in CORE_CLASSES:
        item = aggregate[
            object_type
        ]

        probability_mass = [
            float(
                value
                /
                N_MC
            )
            for value in
            item[
                "occupancy_count_mass_sum_by_horizon"
            ]
        ]

        item[
            "occupancy_probability_mass_sum_by_horizon"
        ] = probability_mass

        item[
            "maximum_probability_observed_by_horizon"
        ] = [
            float(
                value
                /
                N_MC
            )
            for value in
            item[
                "maximum_count_observed_by_horizon"
            ]
        ]

        print()
        print(
            object_type,
            "=",
            json.dumps(
                item,
                sort_keys=True,
            ),
        )

    all_classes_have_nonzero_pocc = all(
        aggregate[
            object_type
        ][
            "actors_with_any_support"
        ]
        >
        0
        for object_type in CORE_CLASSES
    )

    require(
        all_classes_have_nonzero_pocc,
        (
            "At least one core class has "
            "zero actors with nonzero P_occ."
        ),
    )

    print()
    print(
        "all core classes have nonzero P_occ evidence = PASS"
    )

    print(
        "manifest =",
        MANIFEST,
    )

    print(
        "manifest SHA256 =",
        manifest_sha,
    )

    print(
        "scientific manifest digest =",
        global_manifest_digest.hexdigest(),
    )

    # ========================================================
    # H. Fresh Stage6 regression
    # ========================================================

    print()
    print(
        "===== H. FRESH STAGE6 REGRESSION ====="
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
            "Fresh Stage6 regression failed."
        ),
    )

    require(
        regression[
            "tests"
        ]
        ==
        145,
        (
            "Expected 145 Stage6 tests, "
            f"got {regression['tests']}."
        ),
    )

    print(
        "fresh Stage6 regression = PASS | 145"
    )

    # ========================================================
    # I. Final Part2C/2B report
    # ========================================================

    print()
    print(
        "===== I. BLOCK6.6 PART2C/2B REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part2C/2B",

        "status":
            (
                "PASS_ELIGIBLE_N8192_"
                "P_OCC_MATERIALIZED"
            ),

        "eligible_predictive": {
            "actor_count":
                EXPECTED_TOTAL,

            "by_class":
                EXPECTED_CLASS_COUNTS,

            "source_manifest":
                str(
                    ELIGIBLE
                ),

            "source_manifest_sha256":
                EXPECTED[
                    "eligible"
                ],
        },

        "probabilistic_kernel": {
            "trajectory_family":
                "calibrated_Gaussian_GRU",

            "joint_sampling_semantics":
                (
                    "product_of_frozen_calibrated_"
                    "per_horizon_Gaussian_marginals"
                ),

            "sample_count_per_actor":
                N_MC,

            "seed_per_actor":
                MC_SEED,

            "same_seed_for_each_actor":
                True,

            "future_geometry":
                (
                    "full_box_all_eight_physical_"
                    "corners"
                ),

            "heading":
                (
                    "samplewise_predicted_trajectory_"
                    "tangent_with_frozen_low_speed_"
                    "carry_forward"
                ),

            "rasterization":
                (
                    "convex_hull_of_all_eight_"
                    "projected_physical_box_corners_"
                    "cell_center_inclusion"
                ),

            "grid_shape":
                [
                    501,
                    301,
                ],

            "theta_deg":
                [
                    -25.0,
                    25.0,
                ],

            "range_m":
                [
                    0.0,
                    150.0,
                ],
        },

        "persistence": {
            "contract_path":
                str(
                    PERSISTENCE_FREEZE
                ),

            "contract_sha256":
                persistence_sha,

            "canonical_statistic":
                "uint32 occupancy_counts",

            "probability_definition":
                "occupancy_counts / 8192",

            "probability_file_persisted":
                False,

            "lossy_conversion":
                False,
        },

        "classwise_evidence":
            aggregate,

        "all_core_classes_have_nonzero_P_occ":
            True,

        "artifacts": {
            "actor_cache_directory":
                str(
                    CACHE_DIR
                ),

            "manifest_path":
                str(
                    MANIFEST
                ),

            "manifest_sha256":
                manifest_sha,

            "manifest_scientific_digest_sha256":
                global_manifest_digest.hexdigest(),
        },

        "causality": {
            "future_GT_controller_input":
                False,

            "truth_ID":
                False,

            "tracks_to_predict":
                False,

            "objects_of_interest":
                False,

            "perfect_ID_association":
                False,

            "centroid_only":
                False,
        },

        "policy_boundary": {
            "gamma_applied":
                False,

            "gamma_reoptimized":
                False,

            "class_specific_gamma_frozen":
                False,

            "class_margins_frozen":
                False,

            "class_dimming_floors_frozen":
                False,

            "class_priority_frozen":
                False,

            "temporal_smoothing_frozen":
                False,

            "actuation_rate_limit_frozen":
                False,

            "reactive_fallback_changed":
                False,
        },

        "scientific_execution": {
            "new_model_forward":
                False,

            "Monte_Carlo_sampling":
                True,

            "training":
                False,

            "recalibration":
                False,

            "formal_evaluation":
                False,

            "formal_outcomes_read":
                False,

            "formal_tuning":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                145,
        },

        "next":
            (
                "BLOCK6.6_PART3_CLASS_AWARE_"
                "POLICY_DEVELOPMENT_PREREGISTRATION_"
                "FROM_FROZEN_P_OCC_EVIDENCE"
            ),
    }

    report_sha = (
        write_json_exact_or_create(
            REPORT,
            report_payload,
        )
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    # ========================================================
    # J. Freeze + handoff
    # ========================================================

    print()
    print(
        "===== J. FREEZE MANIFEST + PART3 HANDOFF ====="
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.6-Part2C/2B",

        "status":
            "BLOCK66_DEVELOPMENT_P_OCC_FROZEN",

        "immutable_inputs": {
            str(
                ELIGIBLE
            ):
                EXPECTED[
                    "eligible"
                ],

            str(
                POSTERIOR
            ):
                EXPECTED[
                    "posterior"
                ],

            str(
                PART2C2A_REPORT
            ):
                EXPECTED[
                    "part2c2a"
                ],

            str(
                FULLBOX_MODULE
            ):
                EXPECTED[
                    "fullbox_module"
                ],

            str(
                OCCUPANCY_MODULE
            ):
                EXPECTED[
                    "occupancy_module"
                ],

            str(
                NUMERIC_FREEZE
            ):
                EXPECTED[
                    "numeric_freeze"
                ],
        },

        "persistence_contract": {
            "path":
                str(
                    PERSISTENCE_FREEZE
                ),

            "sha256":
                persistence_sha,
        },

        "P_occ_manifest": {
            "path":
                str(
                    MANIFEST
                ),

            "sha256":
                manifest_sha,

            "actor_count":
                877,

            "by_class":
                EXPECTED_CLASS_COUNTS,
        },

        "report": {
            "path":
                str(
                    REPORT
                ),

            "sha256":
                report_sha,
        },

        "scientific_lock": {
            "N_MC":
                N_MC,

            "seed":
                MC_SEED,

            "grid_shape":
                [
                    501,
                    301,
                ],

            "P_occ":
                "FROZEN_DEVELOPMENT_EVIDENCE",

            "class_policy":
                "NOT_FROZEN",

            "formal_tuning":
                "FORBIDDEN",
        },
    }

    freeze_sha = (
        write_json_exact_or_create(
            FREEZE,
            freeze_payload,
        )
    )

    handoff_payload = {
        "stage":
            6,

        "from":
            "6.6-Part2C/2B",

        "to":
            "6.6-Part3",

        "status":
            (
                "PASS_READY_FOR_CLASS_AWARE_"
                "POLICY_DEVELOPMENT"
            ),

        "development_P_occ": {
            "actor_count":
                877,

            "by_class":
                EXPECTED_CLASS_COUNTS,

            "manifest_path":
                str(
                    MANIFEST
                ),

            "manifest_sha256":
                manifest_sha,

            "sample_count":
                N_MC,

            "seed":
                MC_SEED,

            "grid_shape":
                [
                    501,
                    301,
                ],
        },

        "reactive_fallback": {
            "eligible_fallback_actors":
                247,

            "semantics":
                "original_frozen_Block6.2_L_theta_r",

            "modified":
                False,
        },

        "class_policy_status": {
            "class_agnostic_gamma_0p5":
                (
                    "already_frozen_as_separate_"
                    "Block6.5_baseline"
                ),

            "class_aware_gamma":
                "NOT_FROZEN",

            "vehicle_policy":
                "NOT_FROZEN",

            "pedestrian_policy":
                "NOT_FROZEN",

            "cyclist_policy":
                "NOT_FROZEN",

            "dimming_floors":
                "NOT_FROZEN",

            "margins":
                "NOT_FROZEN",

            "temporal_smoothing":
                "NOT_FROZEN",

            "actuation_rate":
                "NOT_FROZEN",
        },

        "required_next": [
            (
                "define_development_only_class_"
                "policy_objectives_and_metrics_before_"
                "numeric_parameter_selection"
            ),
            (
                "vehicle_policy_must_go_beyond_"
                "margin_only_and_prioritize_glare_protection"
            ),
            (
                "pedestrian_policy_must_preserve_"
                "VRU_visibility_and_avoid_full_blackout"
            ),
            (
                "cyclist_policy_must_preserve_"
                "body_vehicle_visibility_and_lateral_margin"
            ),
            (
                "freeze_class_specific_policy_"
                "numerics_before_formal_evaluation"
            ),
            (
                "preserve_PartA_raised_cosine_"
                "reactive_L_theta_r_and_reactive_fallback"
            ),
        ],

        "forbidden_next": [
            "formal_outcome_read_before_policy_freeze",
            "future_GT_controller_input",
            "perfect_ID_association",
            "posthoc_formal_tuning",
            "reuse_communication_beam_actuator_as_ADB",
        ],

        "upstream_freeze": {
            "report_sha256":
                report_sha,

            "freeze_sha256":
                freeze_sha,

            "persistence_contract_sha256":
                persistence_sha,
        },
    }

    handoff_sha = (
        write_json_exact_or_create(
            HANDOFF,
            handoff_payload,
        )
    )

    print(
        "freeze manifest =",
        FREEZE,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # K. Final
    # ========================================================

    elapsed_total = (
        time.perf_counter()
        -
        start
    )

    print()
    print("=" * 78)
    print(
        "BLOCK 6.6 PART2C/2B — FINAL"
    )
    print("=" * 78)

    print(
        "STATUS = "
        "PASS_BLOCK66_ELIGIBLE_N8192_P_OCC_MATERIALIZED"
    )

    print(
        "eligible actors             = 877"
    )

    print(
        "vehicle / pedestrian / cyclist =",
        EXPECTED_CLASS_COUNTS,
    )

    print(
        "N_MC / seed                = 8192 / 20260821"
    )

    print(
        "same seed per actor         = YES"
    )

    print(
        "grid                        = 501 x 301"
    )

    print(
        "public full-box kernel      = FROZEN"
    )

    print(
        "public occupancy kernel     = FROZEN"
    )

    print(
        "all 8 physical corners      = YES"
    )

    print(
        "P_occ persisted             = uint32 counts"
    )

    print(
        "P_occ reconstruction        = counts / 8192"
    )

    print(
        "actors sampled this run     =",
        executed,
    )

    print(
        "actors reused from cache    =",
        reused,
    )

    print(
        "all classes nonzero P_occ   = PASS"
    )

    print(
        "gamma applied               = NO"
    )

    print(
        "class policy frozen         = NO"
    )

    print(
        "reactive fallback changed   = NO"
    )

    print(
        "new model forward           = NO"
    )

    print(
        "training/recalibration      = NO / NO"
    )

    print(
        "formal evaluation           = NO"
    )

    print(
        "formal outcomes read        = NO"
    )

    print(
        "Stage6 regression           = PASS | 145"
    )

    print(
        "manifest SHA256             =",
        manifest_sha,
    )

    print(
        "report SHA256               =",
        report_sha,
    )

    print(
        "freeze SHA256               =",
        freeze_sha,
    )

    print(
        "handoff SHA256              =",
        handoff_sha,
    )

    print(
        "runtime this invocation s   =",
        elapsed_total,
    )

    print(
        "NEXT = BLOCK6.6 PART3 "
        "CLASS-AWARE POLICY DEVELOPMENT PREREGISTRATION"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 78)


try:
    main()

except BaseException as exc:
    print()
    print("=" * 78)
    print(
        "BLOCK6.6 PART2C/2B — CONTROLLED BLOCK"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=24
    )

    print()
    print(
        "completed per-actor caches = PRESERVED"
    )

    print(
        "gamma applied            = NO"
    )

    print(
        "class policy frozen      = NO"
    )

    print(
        "formal outcomes read     = NO"
    )

    print(
        "training/recalibration   = NO / NO"
    )

    print(
        "terminal remains open    = YES"
    )

# Deliberately no sys.exit().

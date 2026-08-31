from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# Frozen roots
# ============================================================

ROOT = Path("/home/agni/waymo")
STAGE5 = ROOT / "iscai_stage5"
STAGE6 = ROOT / "iscai_stage6"

STARTING_EXECUTOR_BACKUP = (
    ROOT
    / "audits/stage6_experiment5/"
      "run_stage6_experiment5_final_executor.pre_explicit_eval.py"
)

EXPECTED_STARTING_EXECUTOR_SHA256 = (
    "6c2b3b73773929f8384cfa63839dd0c49"
    "b9246f5a6429dc3e3b4c45a9fa47f4f"
)

EXPECTED_STAGE5_HANDOFF_SHA256 = (
    "ac6564dbfe537f39fe70e7f2e92a7b2a"
    "5509aa3285d1037983924714bca36124"
)

EXPECTED_METRIC_SOURCE_SHA256 = (
    "bf8358b5a7edfe40ffac2718ca7cd5af"
    "6e7b5fdb3ff8a2823caad6d0f40f057a"
)

EXPECTED_REACTIVE_SHA256 = (
    "7c5d63c76c3d9ef74adbbd7a68e3ce7"
    "483b1dd80fd262cba06f6ce1909994993"
)

EXPECTED_DETERMINISTIC_SHA256 = (
    "6fe5f297da7a20282dcc1cd9cb283468"
    "1be5bdcc7f84f1fad6fb59f57f64cfa4"
)

EXPECTED_COMMON_SHA256 = (
    "814623c55c368e23fbb2b7fd6a901632"
    "3d602f532578f7f8816f008cbdc957a5"
)

EXPECTED_CLASS_AWARE_SHA256 = (
    "723f29e1973e8267ee2c499b91355f77"
    "489d00523d43e9d8cf0726409ec9c53a"
)

EXPECTED_ORACLE_SHA256 = (
    "687ebab2cdb10565235d04af841577db8"
    "3ff54739df2019e8d7d8ea5a7aa36ca"
)

EXPECTED_RESOURCE_EVIDENCE_SHA256 = (
    "b224194c38191b298d1f786fe4abb9b7"
    "1beed7c8f96fcf6c3fe35e4f04bbcf80"
)


# Frozen V3 evidence identities already present in the
# starting implementation authority.
EXPECTED_V3_HASHES = {
    "protocol":
        "d748e6bae34de723b95181b29d3af6759"
        "aa8df48e0dea6229b39d7bfa55b984f",

    "result":
        "e76ec51e2ca875c16c2e7a0e168be3b2"
        "cc54ac37882bf5ef7e2eea0c33ffef44",

    "closure":
        "5f5ccb4e35f08565a9bedf69c9735634c"
        "b70ee83d93776440d62c9b764e69f00",

    "freeze":
        "e28d1378f4599edea41c80de72a864759"
        "5db80009d0f85a60148db5a3a40b008",
}


# ============================================================
# Exact frozen paths
# ============================================================

METRIC_SOURCE = (
    STAGE6
    / "src/iscai_stage6/adb/metric_semantics.py"
)

REACTIVE_PATH = (
    STAGE6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

MATERIALIZED = (
    STAGE6
    / "artifacts/experiment5_final_materialized"
)

DETERMINISTIC_PATH = (
    MATERIALIZED
    / "deterministic_predictive_ADB.npy"
)

COMMON_PATH = (
    MATERIALIZED
    / "predictive_uncertainty_common_policy_v1.npy"
)

CLASS_AWARE_PATH = (
    MATERIALIZED
    / "class_aware_predictive_ADB.npy"
)

ORACLE_PATH = (
    MATERIALIZED
    / "oracle_future_ADB.npy"
)

EVALUATOR_DIR = (
    STAGE6
    / "artifacts/block68/"
      "stage3_fixed_cache/scenarios"
)

FROZEN_RESOURCE_EVIDENCE = (
    STAGE6
    / "reports/stage6_experiment5_latency_resources.json"
)


# ============================================================
# Final outputs
# ============================================================

REPORTS = STAGE6 / "reports"
ARTIFACTS = STAGE6 / "artifacts"

FINAL_MATRIX = (
    REPORTS
    / "stage6_experiment5_numeric_matrix.json"
)

FINAL_MATRIX_CSV = (
    REPORTS
    / "stage6_experiment5_numeric_matrix.csv"
)

FINAL_SAFETY = (
    REPORTS
    / "stage6_experiment5_safety_overmask_ablation.json"
)

FINAL_RESOURCE_SEAL = (
    REPORTS
    / "stage6_experiment5_latency_resource_seal.json"
)

FINAL_REPRO = (
    REPORTS
    / "stage6_experiment5_reproducibility_seal.json"
)

FINAL_FULLPDF_AUDIT = (
    REPORTS
    / "stage6_fullpdf_final_evidence_audit.json"
)

FINAL_PER_SCENARIO = (
    ARTIFACTS
    / "stage6_experiment5_per_scenario_metrics.jsonl"
)

FINAL_FREEZE = (
    ARTIFACTS
    / "stage6_experiment5_freeze_manifest.json"
)

FINAL_CERTIFICATE = (
    REPORTS
    / "stage6_final_certificate.json"
)

FINAL_HANDOFF = (
    ARTIFACTS
    / "stage6_to_stage7_handoff.json"
)


# ============================================================
# Frozen Experiment-5 contract
# ============================================================

BASELINES = (
    "static_ADB",
    "original_reactive_ADB",
    "deterministic_predictive_ADB",
    "uncertainty_aware_predictive_ADB",
    "class_agnostic_predictive_ADB",
    "class_aware_predictive_ADB",
    "oracle_future_ADB",
)

METRICS = (
    "mask_IoU_with_oracle_future_mask",
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "over_masking_area",
    "road_illumination_retention",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "false_dimming",
    "temporal_smoothness",
    "flicker_change_rate",
    "energy_consumption",
    "actuation_latency",
)

SCENARIO_METRICS = METRICS[:-1]

EVALUATOR_FIELDS = (
    "mask_vehicle",
    "mask_pedestrian",
    "mask_cyclist",
    "oracle_all",
    "oracle_vehicle",
    "pedestrian_region",
    "cyclist_region",
    "vehicle_surrogate",
)

N_SCENARIOS = 120
GRID_SHAPE = (501, 301)
SCHEDULE_SHAPE = (4, 501, 301)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

# Current causal t0 authority is prepended:
#
# t0 -> 0.1 -> 0.3 -> 0.5 -> 1.0
#
# Therefore the exact four transitions are:
TEMPORAL_DELTA_T_S = (
    0.1,
    0.2,
    0.2,
    0.5,
)

V3_PRIMARY = {
    "reactive_vehicle_violation":
        0.09481248124008006,

    "v3_vehicle_violation":
        0.07308317791689012,

    "reactive_overmask":
        0.18862460571658454,

    "v3_overmask":
        0.201618399413797,

    "overmask_delta":
        0.012993793697212458,

    "overmask_delta_bound":
        0.02,

    "v3_road_illumination_retention":
        0.8171252155614439,

    "predictive_extra_budget":
        0.01,
}


# ============================================================
# Environment before Stage6 import
# ============================================================

for stage in (
    "iscai_stage0",
    "iscai_stage1",
    "iscai_stage2",
    "iscai_stage3",
    "iscai_stage4",
    "iscai_stage5",
    "iscai_stage6",
):
    path = str(
        ROOT / stage / "src"
    )

    if path not in sys.path:
        sys.path.insert(
            0,
            path,
        )


# ============================================================
# Explicit frozen metric imports
#
# No importlib discovery.
# No inspect.signature.
# No aliases.
# No parameter-name inference.
# ============================================================

from iscai_stage6.adb.metric_semantics import (  # noqa: E402
    binary_dim_support,
    cyclist_visibility_proxy,
    false_dimming,
    flicker_change_rate,
    glare_risk_exposure,
    mask_iou,
    normalized_energy_consumption,
    over_masking_area,
    pedestrian_visibility_proxy,
    road_illumination_retention,
    temporal_smoothness,
    vehicle_shadow_zone_violation,
)


class FailClosed(RuntimeError):
    pass


# ============================================================
# Generic deterministic helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sha256_bytes(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def canonical_bytes(
    payload: Any,
) -> bytes:

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def verify_sha(
    path: Path,
    expected: str,
    label: str,
) -> str:

    if not path.is_file():
        raise FailClosed(
            f"{label}: missing {path}"
        )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise FailClosed(
            f"{label}: SHA mismatch "
            f"{actual} != {expected}"
        )

    return actual


def find_unique_small_file_by_sha(
    roots: tuple[Path, ...],
    expected: str,
    label: str,
) -> Path:

    matches = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            try:
                if (
                    path.stat().st_size
                    >
                    64 * 1024 * 1024
                ):
                    continue
            except OSError:
                continue

            try:
                digest = sha256_file(
                    path
                )
            except OSError:
                continue

            if digest == expected:
                matches.append(
                    path.resolve()
                )

    matches = sorted(
        set(matches)
    )

    if len(matches) != 1:
        raise FailClosed(
            f"{label}: expected exactly one "
            f"SHA-bound file, found {matches}"
        )

    return matches[0]


def atomic_preflight(
    payloads: dict[Path, bytes],
) -> None:

    for path, data in payloads.items():

        if not path.exists():
            continue

        existing = path.read_bytes()

        if existing != data:
            raise FailClosed(
                "Refusing to overwrite existing "
                "non-identical final artifact: "
                f"{path}"
            )


def atomic_publish(
    payloads: dict[Path, bytes],
) -> None:

    atomic_preflight(
        payloads
    )

    for path, data in payloads.items():

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if path.exists():
            continue

        temporary = path.with_name(
            path.name
            + f".tmp.{os.getpid()}"
        )

        with temporary.open("xb") as handle:

            handle.write(data)

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary,
            path,
        )


def finite_mean(
    values: list[float],
    *,
    metric: str,
    baseline: str,
) -> tuple[float, int]:

    finite = [
        float(value)
        for value in values
        if math.isfinite(
            float(value)
        )
    ]

    if not finite:
        raise FailClosed(
            f"{baseline}/{metric}: "
            "zero finite scenario support"
        )

    return (
        float(
            sum(finite)
            /
            len(finite)
        ),
        len(finite),
    )


def safe_json_number(
    value: float,
) -> float | None:

    value = float(value)

    if math.isfinite(value):
        return value

    return None


# ============================================================
# Frozen-source gates
# ============================================================

def verify_starting_executor_backup(
) -> None:

    verify_sha(
        STARTING_EXECUTOR_BACKUP,
        EXPECTED_STARTING_EXECUTOR_SHA256,
        "starting Experiment-5 executor backup",
    )


def verify_frozen_sources(
) -> dict[str, Any]:

    verify_starting_executor_backup()

    verify_sha(
        METRIC_SOURCE,
        EXPECTED_METRIC_SOURCE_SHA256,
        "frozen metric_semantics.py",
    )

    stage5_handoff = (
        find_unique_small_file_by_sha(
            (
                STAGE5 / "artifacts",
                STAGE5 / "reports",
            ),
            EXPECTED_STAGE5_HANDOFF_SHA256,
            "Stage5 immutable handoff",
        )
    )

    v3_files = {}

    for label, digest in (
        EXPECTED_V3_HASHES.items()
    ):

        v3_files[label] = (
            find_unique_small_file_by_sha(
                (
                    STAGE6 / "configs",
                    STAGE6 / "reports",
                    STAGE6 / "artifacts",
                ),
                digest,
                f"Stage6-V3 {label}",
            )
        )

    verify_sha(
        REACTIVE_PATH,
        EXPECTED_REACTIVE_SHA256,
        "reactive t0 maps",
    )

    verify_sha(
        DETERMINISTIC_PATH,
        EXPECTED_DETERMINISTIC_SHA256,
        "deterministic predictive ADB",
    )

    verify_sha(
        COMMON_PATH,
        EXPECTED_COMMON_SHA256,
        "uncertainty/common ADB",
    )

    verify_sha(
        CLASS_AWARE_PATH,
        EXPECTED_CLASS_AWARE_SHA256,
        "class-aware V3 ADB",
    )

    verify_sha(
        FROZEN_RESOURCE_EVIDENCE,
        EXPECTED_RESOURCE_EVIDENCE_SHA256,
        "frozen latency/resource evidence",
    )

    return {
        "starting_executor_sha256":
            EXPECTED_STARTING_EXECUTOR_SHA256,

        "metric_source_sha256":
            EXPECTED_METRIC_SOURCE_SHA256,

        "stage5_handoff": {
            "path":
                str(stage5_handoff),

            "sha256":
                EXPECTED_STAGE5_HANDOFF_SHA256,
        },

        "stage6_v3": {
            label: {
                "path":
                    str(path),

                "sha256":
                    EXPECTED_V3_HASHES[
                        label
                    ],
            }
            for label, path
            in v3_files.items()
        },
    }


# ============================================================
# Regression gate
# ============================================================

def run_stage6_regression(
) -> dict[str, Any]:

    environment = (
        os.environ.copy()
    )

    source_paths = [
        str(
            ROOT / f"iscai_stage{i}" / "src"
        )
        for i in range(0, 7)
    ]

    environment[
        "PYTHONPATH"
    ] = ":".join(
        source_paths
    )

    command = [
        sys.executable,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
        "-v",
    ]

    result = subprocess.run(
        command,
        cwd=STAGE6,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print(
        result.stdout,
        end="",
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        result.stdout,
    )

    count = (
        int(match.group(1))
        if match is not None
        else None
    )

    if (
        result.returncode != 0
        or count != 229
    ):
        raise FailClosed(
            "Stage6 regression gate failed: "
            f"returncode={result.returncode}, "
            f"tests={count}"
        )

    return {
        "status":
            "PASS",

        "tests":
            count,

        "returncode":
            result.returncode,
    }


# ============================================================
# Exact controller-output locking
#
# Future evaluator/oracle data is NOT opened before this point.
# ============================================================

def validate_intensity_array(
    array: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:

    x = np.asarray(
        array
    )

    if x.shape != expected_shape:
        raise FailClosed(
            f"{label}: shape {x.shape} "
            f"!= {expected_shape}"
        )

    if not np.issubdtype(
        x.dtype,
        np.number,
    ):
        raise FailClosed(
            f"{label}: non-numeric dtype {x.dtype}"
        )

    if not np.all(
        np.isfinite(x)
    ):
        raise FailClosed(
            f"{label}: non-finite values"
        )

    if (
        np.any(x < 0.0)
        or
        np.any(x > 1.0)
    ):
        raise FailClosed(
            f"{label}: values outside [0,1]"
        )

    return x


def lock_nonoracle_controller_outputs(
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
]:

    reactive = np.load(
        REACTIVE_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    validate_intensity_array(
        reactive,
        label="original_reactive_t0",
        expected_shape=(
            N_SCENARIOS,
            *GRID_SHAPE,
        ),
    )

    deterministic = np.load(
        DETERMINISTIC_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    common = np.load(
        COMMON_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    class_aware = np.load(
        CLASS_AWARE_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    for label, array in (
        (
            "deterministic_predictive_ADB",
            deterministic,
        ),
        (
            "predictive_uncertainty_common_policy_v1",
            common,
        ),
        (
            "class_aware_predictive_ADB",
            class_aware,
        ),
    ):

        validate_intensity_array(
            array,
            label=label,
            expected_shape=(
                N_SCENARIOS,
                *SCHEDULE_SHAPE,
            ),
        )

    static_semantic = (
        b"static_ADB;"
        b"I=1;"
        b"N=120;"
        b"schedule_shape=(4,501,301)"
    )

    locks = {
        "static_ADB": {
            "semantic":
                "exact full illumination I=1",

            "semantic_sha256":
                sha256_bytes(
                    static_semantic
                ),
        },

        "original_reactive_ADB": {
            "file":
                str(REACTIVE_PATH),

            "sha256":
                EXPECTED_REACTIVE_SHA256,

            "future_representation":
                "hold exact causal t0 action unchanged",
        },

        "deterministic_predictive_ADB": {
            "file":
                str(DETERMINISTIC_PATH),

            "sha256":
                EXPECTED_DETERMINISTIC_SHA256,
        },

        "uncertainty_aware_predictive_ADB": {
            "file":
                str(COMMON_PATH),

            "sha256":
                EXPECTED_COMMON_SHA256,
        },

        "class_agnostic_predictive_ADB": {
            "file":
                str(COMMON_PATH),

            "sha256":
                EXPECTED_COMMON_SHA256,

            "shared_runtime_with_uncertainty_aware":
                True,
        },

        "class_aware_predictive_ADB": {
            "file":
                str(CLASS_AWARE_PATH),

            "sha256":
                EXPECTED_CLASS_AWARE_SHA256,
        },
    }

    maps = {
        "deterministic_predictive_ADB":
            deterministic,

        "uncertainty_aware_predictive_ADB":
            common,

        "class_agnostic_predictive_ADB":
            common,

        "class_aware_predictive_ADB":
            class_aware,
    }

    return (
        reactive,
        maps,
        locks,
    )


# ============================================================
# Only after non-oracle locks: oracle + evaluator
# ============================================================

def load_oracle_after_lock(
) -> np.ndarray:

    verify_sha(
        ORACLE_PATH,
        EXPECTED_ORACLE_SHA256,
        "oracle future ADB",
    )

    oracle = np.load(
        ORACLE_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    validate_intensity_array(
        oracle,
        label="oracle_future_ADB",
        expected_shape=(
            N_SCENARIOS,
            *SCHEDULE_SHAPE,
        ),
    )

    return oracle


def evaluator_index_after_lock(
) -> list[
    tuple[
        str,
        Path,
        Path,
    ]
]:

    if not EVALUATOR_DIR.is_dir():
        raise FailClosed(
            "Frozen evaluator directory missing."
        )

    npz_files = sorted(
        EVALUATOR_DIR.glob(
            "*.npz"
        )
    )

    json_files = sorted(
        EVALUATOR_DIR.glob(
            "*.json"
        )
    )

    if (
        len(npz_files)
        != N_SCENARIOS
        or
        len(json_files)
        != N_SCENARIOS
    ):
        raise FailClosed(
            "Expected exactly 120 NPZ + "
            "120 JSON evaluator sidecars."
        )

    result = []

    for index, npz_path in enumerate(
        npz_files
    ):

        prefix = (
            f"{index:03d}_"
        )

        if not npz_path.stem.startswith(
            prefix
        ):
            raise FailClosed(
                f"Evaluator ordering failure at "
                f"{npz_path.name}"
            )

        scenario_id = (
            npz_path.stem[
                len(prefix):
            ]
        )

        json_path = (
            npz_path.with_suffix(
                ".json"
            )
        )

        if not json_path.is_file():
            raise FailClosed(
                f"Missing evaluator sidecar "
                f"{json_path}"
            )

        sidecar = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            int(
                sidecar[
                    "cohort_index"
                ]
            )
            != index
        ):
            raise FailClosed(
                f"{json_path}: cohort_index mismatch"
            )

        if (
            str(
                sidecar[
                    "scenario_id"
                ]
            )
            != scenario_id
        ):
            raise FailClosed(
                f"{json_path}: scenario_id mismatch"
            )

        result.append(
            (
                scenario_id,
                npz_path,
                json_path,
            )
        )

    return result


def read_evaluator_scenario(
    path: Path,
) -> dict[str, np.ndarray]:

    result = {}

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        if set(
            archive.files
        ) != set(
            EVALUATOR_FIELDS
        ):
            raise FailClosed(
                f"{path}: evaluator field-set mismatch"
            )

        for field in EVALUATOR_FIELDS:

            array = np.asarray(
                archive[field]
            )

            if array.shape != SCHEDULE_SHAPE:
                raise FailClosed(
                    f"{path}/{field}: "
                    f"shape {array.shape}"
                )

            if array.dtype != np.bool_:
                raise FailClosed(
                    f"{path}/{field}: "
                    "must be bool"
                )

            result[field] = (
                array
            )

    return result


# ============================================================
# Exact frozen resource evidence
# ============================================================

def load_resource_evidence(
) -> dict[str, Any]:

    verify_sha(
        FROZEN_RESOURCE_EVIDENCE,
        EXPECTED_RESOURCE_EVIDENCE_SHA256,
        "latency/resource evidence",
    )

    document = json.loads(
        FROZEN_RESOURCE_EVIDENCE.read_text(
            encoding="utf-8"
        )
    )

    if (
        document.get("status")
        !=
        "PASS_REAL_ADB_LATENCY_RESOURCE_MEASUREMENT"
    ):
        raise FailClosed(
            "Unexpected frozen resource status."
        )

    actuation = document[
        "adb_actuation_latency"
    ]

    generation = document[
        "adb_generation_latency"
    ]

    if (
        int(
            actuation[
                "calls"
            ]
        )
        != 120
    ):
        raise FailClosed(
            "Actuation-latency evidence must "
            "contain exactly 120 calls."
        )

    for section_name, section in (
        (
            "adb_actuation_latency",
            actuation,
        ),
        (
            "adb_generation_latency",
            generation,
        ),
    ):

        for key in (
            "mean_s",
            "median_s",
            "p95_s",
            "max_s",
        ):

            value = float(
                section[key]
            )

            if (
                not math.isfinite(
                    value
                )
                or value < 0.0
            ):
                raise FailClosed(
                    f"{section_name}/{key}: "
                    "invalid latency evidence"
                )

    boundary = document[
        "measurement_boundary"
    ]

    if (
        boundary.get(
            "future_GT_controller_input"
        )
        is not False
        or
        boundary.get(
            "new_MC_sampling"
        )
        is not False
        or
        boundary.get(
            "retuning"
        )
        is not False
        or
        boundary.get(
            "training"
        )
        is not False
    ):
        raise FailClosed(
            "Frozen latency measurement boundary "
            "violates Experiment-5 contract."
        )

    return {
        "status":
            "PASS",

        "source":
            str(
                FROZEN_RESOURCE_EVIDENCE
            ),

        "source_sha256":
            EXPECTED_RESOURCE_EVIDENCE_SHA256,

        "scope":
            (
                "COMMON_MEASURED_STAGE6_ADB_RUNTIME_REFERENCE;"
                " NOT A PER-BASELINE TIMING BENCHMARK"
            ),

        "matrix_actuation_latency_semantics":
            (
                "The same measured common Stage6 ADB-runtime "
                "mean is carried into each baseline row solely "
                "to keep the required 7x12 PDF table rectangular. "
                "No claim of seven separately measured baseline "
                "latencies is made."
            ),

        "adb_actuation_latency":
            {
                key:
                    actuation[key]
                for key in (
                    "calls",
                    "mean_s",
                    "median_s",
                    "p95_s",
                    "max_s",
                )
            },

        "adb_generation_latency":
            {
                key:
                    generation[key]
                for key in (
                    "calls",
                    "mean_s",
                    "median_s",
                    "p95_s",
                    "max_s",
                )
            },

        "controller_complexity":
            document[
                "controller_complexity"
            ],

        "memory":
            document[
                "memory"
            ],

        "measurement_boundary":
            boundary,

        "provenance":
            document[
                "provenance"
            ],
    }


# ============================================================
# Canonical per-scenario schedules
# ============================================================

def schedule_for(
    baseline: str,
    scenario_index: int,
    *,
    reactive: np.ndarray,
    maps: dict[str, np.ndarray],
    oracle: np.ndarray,
) -> np.ndarray:

    if baseline == "static_ADB":

        schedule = np.ones(
            SCHEDULE_SHAPE,
            dtype=np.float64,
        )

    elif (
        baseline
        ==
        "original_reactive_ADB"
    ):

        t0 = np.asarray(
            reactive[
                scenario_index
            ],
            dtype=np.float64,
        )

        schedule = np.repeat(
            t0[
                None,
                :,
                :,
            ],
            4,
            axis=0,
        )

        for horizon in range(
            1,
            4,
        ):

            if not np.array_equal(
                schedule[0],
                schedule[horizon],
            ):
                raise FailClosed(
                    "Reactive hold semantics failed."
                )

    elif baseline == "oracle_future_ADB":

        schedule = np.asarray(
            oracle[
                scenario_index
            ],
            dtype=np.float64,
        )

    else:

        schedule = np.asarray(
            maps[
                baseline
            ][
                scenario_index
            ],
            dtype=np.float64,
        )

    return validate_intensity_array(
        schedule,
        label=(
            f"{baseline}[{scenario_index}]"
        ),
        expected_shape=SCHEDULE_SHAPE,
    )


# ============================================================
# Explicit 11 scenario-level metric calls
# ============================================================

def evaluate_scenario_explicit(
    schedule: np.ndarray,
    current_reactive_t0: np.ndarray,
    evaluator: dict[str, np.ndarray],
) -> dict[str, float]:

    candidate_mask = (
        binary_dim_support(
            schedule
        )
    )

    # Frozen pre-outcome road-ROI authority:
    #
    # all valid cells of the theta-range headlamp
    # actuator grid.
    road_roi = np.ones(
        SCHEDULE_SHAPE,
        dtype=np.bool_,
    )

    current = np.asarray(
        current_reactive_t0,
        dtype=np.float64,
    )

    if current.shape != GRID_SHAPE:
        raise FailClosed(
            "Current t0 illumination shape failure."
        )

    temporal_sequence = (
        np.concatenate(
            (
                current[
                    None,
                    :,
                    :,
                ],
                schedule,
            ),
            axis=0,
        )
    )

    values = {
        "mask_IoU_with_oracle_future_mask":
            mask_iou(
                candidate_mask,
                evaluator[
                    "oracle_all"
                ],
            ),

        "vehicle_shadow_zone_violation":
            vehicle_shadow_zone_violation(
                candidate_mask,
                evaluator[
                    "oracle_vehicle"
                ],
            ),

        "glare_risk_exposure":
            glare_risk_exposure(
                schedule,
                evaluator[
                    "vehicle_surrogate"
                ],
            ),

        "over_masking_area":
            over_masking_area(
                candidate_mask,
                evaluator[
                    "oracle_all"
                ],
            ),

        "road_illumination_retention":
            road_illumination_retention(
                schedule,
                road_roi,
            ),

        "pedestrian_visibility_proxy":
            pedestrian_visibility_proxy(
                schedule,
                evaluator[
                    "pedestrian_region"
                ],
            ),

        "cyclist_visibility_proxy":
            cyclist_visibility_proxy(
                schedule,
                evaluator[
                    "cyclist_region"
                ],
            ),

        "false_dimming":
            false_dimming(
                schedule,
                evaluator[
                    "oracle_all"
                ],
            ),

        "temporal_smoothness":
            temporal_smoothness(
                temporal_sequence,
                delta_t_s=(
                    TEMPORAL_DELTA_T_S
                ),
            ),

        "flicker_change_rate":
            flicker_change_rate(
                temporal_sequence
            ),

        "energy_consumption":
            normalized_energy_consumption(
                schedule
            ),
    }

    if set(values) != set(
        SCENARIO_METRICS
    ):
        raise FailClosed(
            "Explicit metric-set mismatch."
        )

    # NaN is authoritative NA only for metrics whose
    # frozen evaluator support may be empty.
    allowed_na = {
        "vehicle_shadow_zone_violation",
        "glare_risk_exposure",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    }

    for metric, value in values.items():

        numeric = float(
            value
        )

        if (
            not math.isfinite(
                numeric
            )
            and metric not in allowed_na
        ):
            raise FailClosed(
                f"{metric}: unexpected non-finite value"
            )

    return {
        key:
            float(value)
        for key, value
        in values.items()
    }


# ============================================================
# One-real-scenario smoke before full matrix
# ============================================================

def one_scenario_smoke(
    evaluator_index,
    reactive,
    maps,
    oracle,
) -> dict[str, Any]:

    (
        scenario_id,
        npz_path,
        _,
    ) = evaluator_index[0]

    evaluator = (
        read_evaluator_scenario(
            npz_path
        )
    )

    outputs = {}

    for baseline in BASELINES:

        schedule = schedule_for(
            baseline,
            0,
            reactive=reactive,
            maps=maps,
            oracle=oracle,
        )

        values = (
            evaluate_scenario_explicit(
                schedule,
                np.asarray(
                    reactive[0],
                    dtype=np.float64,
                ),
                evaluator,
            )
        )

        outputs[
            baseline
        ] = {
            key:
                safe_json_number(
                    value
                )
            for key, value
            in values.items()
        }

    reactive_schedule = schedule_for(
        "original_reactive_ADB",
        0,
        reactive=reactive,
        maps=maps,
        oracle=oracle,
    )

    if not all(
        np.array_equal(
            reactive_schedule[h],
            reactive[0],
        )
        for h in range(4)
    ):
        raise FailClosed(
            "Reactive smoke hold failed."
        )

    static_schedule = schedule_for(
        "static_ADB",
        0,
        reactive=reactive,
        maps=maps,
        oracle=oracle,
    )

    if not np.array_equal(
        static_schedule,
        np.ones(
            SCHEDULE_SHAPE,
            dtype=np.float64,
        ),
    ):
        raise FailClosed(
            "Static ADB smoke failed."
        )

    return {
        "status":
            "PASS",

        "scenario_index":
            0,

        "scenario_id":
            scenario_id,

        "explicit_metric_calls":
            True,

        "reactive_hold":
            "PASS",

        "static_I_equals_1":
            "PASS",

        "metrics":
            outputs,
    }


# ============================================================
# Full evaluation:
#
# one scenario schedule -> 11 metrics -> finite macro mean.
#
# No global N=120 flattening.
# ============================================================

def evaluate_once(
    evaluator_index,
    reactive,
    maps,
    oracle,
    *,
    common_actuation_latency_s: float,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, int]],
    list[dict[str, Any]],
]:

    accumulated = {
        baseline: {
            metric: []
            for metric in SCENARIO_METRICS
        }
        for baseline in BASELINES
    }

    rows = []

    for (
        scenario_index,
        (
            scenario_id,
            npz_path,
            _,
        ),
    ) in enumerate(
        evaluator_index
    ):

        evaluator = (
            read_evaluator_scenario(
                npz_path
            )
        )

        current_t0 = np.asarray(
            reactive[
                scenario_index
            ],
            dtype=np.float64,
        )

        for baseline in BASELINES:

            schedule = schedule_for(
                baseline,
                scenario_index,
                reactive=reactive,
                maps=maps,
                oracle=oracle,
            )

            metrics = (
                evaluate_scenario_explicit(
                    schedule,
                    current_t0,
                    evaluator,
                )
            )

            for (
                metric,
                value,
            ) in metrics.items():

                accumulated[
                    baseline
                ][
                    metric
                ].append(
                    float(value)
                )

            rows.append(
                {
                    "scenario_index":
                        scenario_index,

                    "scenario_id":
                        scenario_id,

                    "baseline":
                        baseline,

                    "metrics": {
                        metric:
                            safe_json_number(
                                value
                            )
                        for metric, value
                        in metrics.items()
                    },

                    # No fabricated per-scenario latency:
                    "actuation_latency":
                        None,

                    "actuation_latency_semantics":
                        (
                            "shared common frozen Stage6 "
                            "ADB-runtime evidence; not a "
                            "scenario/baseline-specific timing"
                        ),
                }
            )

    matrix = {}
    support = {}

    for baseline in BASELINES:

        row = {}
        counts = {}

        for metric in SCENARIO_METRICS:

            (
                aggregate,
                finite_count,
            ) = finite_mean(
                accumulated[
                    baseline
                ][
                    metric
                ],
                metric=metric,
                baseline=baseline,
            )

            row[
                metric
            ] = aggregate

            counts[
                metric
            ] = finite_count

        # Required PDF table cell #12.
        #
        # This is explicitly the SAME common measured
        # Stage6 ADB-runtime reference for all rows.
        row[
            "actuation_latency"
        ] = float(
            common_actuation_latency_s
        )

        counts[
            "actuation_latency"
        ] = 120

        matrix[
            baseline
        ] = row

        support[
            baseline
        ] = counts

    return (
        matrix,
        support,
        rows,
    )


def validate_matrix(
    matrix,
) -> None:

    if set(matrix) != set(
        BASELINES
    ):
        raise FailClosed(
            "Expected exactly seven baselines."
        )

    cells = 0

    for baseline in BASELINES:

        row = matrix[
            baseline
        ]

        if set(row) != set(
            METRICS
        ):
            raise FailClosed(
                f"{baseline}: 12-metric set failure"
            )

        for metric in METRICS:

            value = float(
                row[
                    metric
                ]
            )

            if not math.isfinite(
                value
            ):
                raise FailClosed(
                    f"{baseline}/{metric}: "
                    "aggregate is not finite"
                )

            cells += 1

    if cells != 84:
        raise FailClosed(
            f"Expected 84 numeric cells, got {cells}"
        )


# ============================================================
# Frozen V3 numerical authority
# ============================================================

def validate_v3_identity(
    matrix,
) -> dict[str, Any]:
    """
    Validate the immutable V3 controller and the final
    Experiment-5 safety gate without mixing evaluator
    populations.

    Important distinction
    ---------------------
    The historical reactive over-mask value used during
    development/NI freezing was computed on the frozen
    t0-selected ADB actor population.

    Final Experiment-5 uses the frozen final evaluator
    reference supplied by the 120 scenario shards.

    Therefore the historical absolute reactive over-mask
    value is provenance, NOT an exact final-evaluator
    identity constraint.

    No controller parameter, metric formula, tolerance,
    threshold, posterior calibration, or V3 tensor is
    changed here.
    """

    reactive = matrix[
        "original_reactive_ADB"
    ]

    v3 = matrix[
        "class_aware_predictive_ADB"
    ]

    # --------------------------------------------------------
    # Frozen acceptance policy.
    #
    # This is read, not modified.
    # --------------------------------------------------------

    acceptance_path = (
        STAGE6
        / "configs/"
          "stage6_preformal_primary_acceptance_policy.json"
    )

    expected_acceptance_sha = (
        "e103466b1a6eb62be4e6746029147cbda"
        "9671b05130acba5210a3ab7328e6b81"
    )

    if not acceptance_path.is_file():
        raise FailClosed(
            "Frozen Stage6 acceptance policy missing."
        )

    actual_acceptance_sha = sha256_file(
        acceptance_path
    )

    if (
        actual_acceptance_sha
        != expected_acceptance_sha
    ):
        raise FailClosed(
            "Frozen Stage6 acceptance-policy SHA changed."
        )

    acceptance = json.loads(
        acceptance_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        acceptance.get(
            "formal_parameter_tuning_allowed"
        )
        is not False
    ):
        raise FailClosed(
            "Acceptance policy unexpectedly permits tuning."
        )

    rules = acceptance[
        "formal_rules"
    ]

    overmask_bound = float(
        rules[
            "over_masking_area"
        ][
            "frozen_delta"
        ]
    )

    pedestrian_bound = float(
        rules[
            "pedestrian_visibility_proxy"
        ][
            "frozen_delta"
        ]
    )

    cyclist_bound = float(
        rules[
            "cyclist_visibility_proxy"
        ][
            "frozen_delta"
        ]
    )

    if not math.isclose(
        overmask_bound,
        0.02,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise FailClosed(
            "Frozen over-mask bound changed."
        )

    if not math.isclose(
        pedestrian_bound,
        0.05,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise FailClosed(
            "Frozen pedestrian bound changed."
        )

    if not math.isclose(
        cyclist_bound,
        0.05,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise FailClosed(
            "Frozen cyclist bound changed."
        )


    # --------------------------------------------------------
    # Exact numerical identities ONLY where the aggregation /
    # evaluator semantics are directly comparable.
    #
    # Controller immutability itself is already guaranteed
    # separately by the exact V3 .npy SHA gate.
    # --------------------------------------------------------

    exact_checks = {
        "reactive_vehicle_violation": (
            reactive[
                "vehicle_shadow_zone_violation"
            ],
            V3_PRIMARY[
                "reactive_vehicle_violation"
            ],
        ),

        "v3_vehicle_violation": (
            v3[
                "vehicle_shadow_zone_violation"
            ],
            V3_PRIMARY[
                "v3_vehicle_violation"
            ],
        ),

        # Road-retention depends only on final illumination
        # and the frozen full-grid road ROI. It does not use
        # the historical t0-selected oracle population.
        "v3_road_illumination_retention": (
            v3[
                "road_illumination_retention"
            ],
            V3_PRIMARY[
                "v3_road_illumination_retention"
            ],
        ),
    }

    exact_report = {}

    for label, (
        actual,
        expected,
    ) in exact_checks.items():

        passed = math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        exact_report[
            label
        ] = {
            "actual":
                float(actual),

            "expected":
                float(expected),

            "pass":
                passed,
        }

        if not passed:
            raise FailClosed(
                "Comparable frozen V3 identity failed: "
                f"{label}"
            )


    # --------------------------------------------------------
    # FINAL Experiment-5 safety comparison.
    #
    # Both sides are evaluated using EXACTLY the same:
    #   - 120 scenarios,
    #   - final frozen evaluator shards,
    #   - oracle_all / class reference regions,
    #   - authoritative metric functions,
    #   - equal-scenario macro aggregation.
    #
    # The tolerances themselves remain the already-frozen
    # pre-outcome values.
    # --------------------------------------------------------

    vehicle_reactive = float(
        reactive[
            "vehicle_shadow_zone_violation"
        ]
    )

    vehicle_v3 = float(
        v3[
            "vehicle_shadow_zone_violation"
        ]
    )

    overmask_reactive = float(
        reactive[
            "over_masking_area"
        ]
    )

    overmask_v3 = float(
        v3[
            "over_masking_area"
        ]
    )

    pedestrian_reactive = float(
        reactive[
            "pedestrian_visibility_proxy"
        ]
    )

    pedestrian_v3 = float(
        v3[
            "pedestrian_visibility_proxy"
        ]
    )

    cyclist_reactive = float(
        reactive[
            "cyclist_visibility_proxy"
        ]
    )

    cyclist_v3 = float(
        v3[
            "cyclist_visibility_proxy"
        ]
    )


    vehicle_pass = (
        vehicle_v3
        <
        vehicle_reactive
    )

    final_overmask_delta = (
        overmask_v3
        -
        overmask_reactive
    )

    overmask_pass = (
        final_overmask_delta
        <=
        overmask_bound
        + 1e-15
    )

    final_pedestrian_delta = (
        pedestrian_v3
        -
        pedestrian_reactive
    )

    pedestrian_pass = (
        final_pedestrian_delta
        >=
        -pedestrian_bound
        - 1e-15
    )

    final_cyclist_delta = (
        cyclist_v3
        -
        cyclist_reactive
    )

    cyclist_pass = (
        final_cyclist_delta
        >=
        -cyclist_bound
        - 1e-15
    )


    if not vehicle_pass:
        raise FailClosed(
            "Final same-evaluator V3 vehicle "
            "strict-improvement gate failed."
        )

    if not overmask_pass:
        raise FailClosed(
            "Final same-evaluator V3 over-mask "
            "non-inferiority gate failed."
        )

    if not pedestrian_pass:
        raise FailClosed(
            "Final same-evaluator V3 pedestrian "
            "visibility non-inferiority gate failed."
        )

    if not cyclist_pass:
        raise FailClosed(
            "Final same-evaluator V3 cyclist "
            "visibility non-inferiority gate failed."
        )


    # --------------------------------------------------------
    # Historical V3 development-gate values are retained as
    # provenance, but are NOT silently reinterpreted as
    # final-evaluator absolute identities.
    # --------------------------------------------------------

    historical = {
        "role":
            (
                "FROZEN_DEVELOPMENT_GATE_PROVENANCE;"
                " NOT FINAL_EXPERIMENT5_ABSOLUTE_IDENTITY"
            ),

        "historical_reactive_overmask":
            V3_PRIMARY[
                "reactive_overmask"
            ],

        "historical_v3_overmask":
            V3_PRIMARY[
                "v3_overmask"
            ],

        "historical_reported_delta":
            V3_PRIMARY[
                "overmask_delta"
            ],

        "historical_allowed_delta":
            V3_PRIMARY[
                "overmask_delta_bound"
            ],

        "reason_not_used_as_final_absolute_identity":
            (
                "historical reactive NI comparator used "
                "future full-box supports of the frozen "
                "t0-selected ADB actor population; final "
                "Experiment-5 uses the frozen final "
                "evaluator reference for both baselines"
            ),
    }


    return {
        "status":
            "PASS",

        "controller_identity": {
            "class_aware_tensor_sha_locked":
                True,

            "V3_regenerated":
                False,

            "V3_retuned":
                False,
        },

        "comparable_exact_numeric_checks":
            exact_report,

        "final_same_evaluator_acceptance": {
            "vehicle_shadow_zone_violation": {
                "reactive":
                    vehicle_reactive,

                "class_aware":
                    vehicle_v3,

                "rule":
                    "class_aware < reactive",

                "pass":
                    vehicle_pass,
            },

            "over_masking_area": {
                "reactive":
                    overmask_reactive,

                "class_aware":
                    overmask_v3,

                "delta":
                    final_overmask_delta,

                "frozen_allowed_delta":
                    overmask_bound,

                "rule":
                    (
                        "class_aware - reactive "
                        "<= frozen_delta"
                    ),

                "pass":
                    overmask_pass,
            },

            "pedestrian_visibility_proxy": {
                "reactive":
                    pedestrian_reactive,

                "class_aware":
                    pedestrian_v3,

                "delta":
                    final_pedestrian_delta,

                "frozen_allowed_degradation":
                    pedestrian_bound,

                "rule":
                    (
                        "class_aware - reactive "
                        ">= -frozen_delta"
                    ),

                "pass":
                    pedestrian_pass,
            },

            "cyclist_visibility_proxy": {
                "reactive":
                    cyclist_reactive,

                "class_aware":
                    cyclist_v3,

                "delta":
                    final_cyclist_delta,

                "frozen_allowed_degradation":
                    cyclist_bound,

                "rule":
                    (
                        "class_aware - reactive "
                        ">= -frozen_delta"
                    ),

                "pass":
                    cyclist_pass,
            },
        },

        "historical_development_gate":
            historical,

        "acceptance_policy": {
            "path":
                str(
                    acceptance_path
                ),

            "sha256":
                actual_acceptance_sha,

            "overmask_delta":
                overmask_bound,

            "pedestrian_delta":
                pedestrian_bound,

            "cyclist_delta":
                cyclist_bound,
        },

        "evaluator_comparison":
            "SAME_FINAL_EVALUATOR_BOTH_SIDES",

        "predictive_extra_spatial_budget":
            V3_PRIMARY[
                "predictive_extra_budget"
            ],

        "retuning":
            False,

        "threshold_change":
            False,

        "acceptance_bound_change":
            False,

        "Stage4_recalibration":
            False,

        "future_GT_controller_input":
            False,
    }


# ============================================================
# Safety / ablation evidence
# ============================================================

def build_safety_ablation(
    matrix,
) -> dict[str, Any]:

    selected = (
        "vehicle_shadow_zone_violation",
        "over_masking_area",
        "road_illumination_retention",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    )

    points = {
        baseline: {
            metric:
                float(
                    matrix[
                        baseline
                    ][
                        metric
                    ]
                )
            for metric in selected
        }
        for baseline in BASELINES
    }

    pairs = (
        (
            "original_reactive_ADB",
            "deterministic_predictive_ADB",
        ),
        (
            "deterministic_predictive_ADB",
            "uncertainty_aware_predictive_ADB",
        ),
        (
            "uncertainty_aware_predictive_ADB",
            "class_agnostic_predictive_ADB",
        ),
        (
            "class_agnostic_predictive_ADB",
            "class_aware_predictive_ADB",
        ),
        (
            "original_reactive_ADB",
            "class_aware_predictive_ADB",
        ),
    )

    deltas = {}

    for left, right in pairs:

        deltas[
            f"{right}_minus_{left}"
        ] = {
            metric:
                (
                    points[
                        right
                    ][
                        metric
                    ]
                    -
                    points[
                        left
                    ][
                        metric
                    ]
                )
            for metric in selected
        }

    curve = sorted(
        (
            {
                "baseline":
                    baseline,

                "over_masking_area":
                    points[
                        baseline
                    ][
                        "over_masking_area"
                    ],

                "vehicle_shadow_zone_violation":
                    points[
                        baseline
                    ][
                        "vehicle_shadow_zone_violation"
                    ],

                "road_illumination_retention":
                    points[
                        baseline
                    ][
                        "road_illumination_retention"
                    ],
            }
            for baseline in BASELINES
        ),
        key=lambda item: (
            item[
                "over_masking_area"
            ],
            item[
                "vehicle_shadow_zone_violation"
            ],
            item[
                "baseline"
            ],
        ),
    )

    common_equal = (
        matrix[
            "uncertainty_aware_predictive_ADB"
        ]
        ==
        matrix[
            "class_agnostic_predictive_ADB"
        ]
    )

    if not common_equal:
        raise FailClosed(
            "Two labels bound to the same frozen "
            "runtime produced different metrics."
        )

    return {
        "status":
            "PASS_EVIDENCE_COMPLETE",

        "points":
            points,

        "deltas":
            deltas,

        "safety_overmask_curve":
            curve,

        "uncertainty_aware_and_class_agnostic": {
            "shared_frozen_runtime":
                True,

            "same_source_sha256":
                EXPECTED_COMMON_SHA256,

            "numeric_rows_identical":
                True,

            "artificial_difference_introduced":
                False,
        },

        "post_outcome_tuning":
            False,
    }


# ============================================================
# Serialization
# ============================================================

def matrix_csv_bytes(
    matrix,
) -> bytes:

    output = io.StringIO(
        newline=""
    )

    writer = csv.writer(
        output,
        lineterminator="\n",
    )

    writer.writerow(
        (
            "baseline",
            *METRICS,
        )
    )

    for baseline in BASELINES:

        writer.writerow(
            (
                baseline,
                *(
                    format(
                        float(
                            matrix[
                                baseline
                            ][
                                metric
                            ]
                        ),
                        ".17g",
                    )
                    for metric in METRICS
                ),
            )
        )

    return output.getvalue().encode(
        "utf-8"
    )


def jsonl_bytes(
    rows,
) -> bytes:

    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode(
        "utf-8"
    )


# ============================================================
# Main
# ============================================================

def main() -> int:

    try:

        print(
            "============================================================"
        )
        print(
            "STAGE 6 — EXPERIMENT 5 EXPLICIT FINAL EVALUATOR"
        )
        print(
            "============================================================"
        )

        # ------------------------------------------------------
        # A. Immutable provenance
        # ------------------------------------------------------

        frozen = (
            verify_frozen_sources()
        )

        print(
            "frozen provenance      = PASS"
        )

        # ------------------------------------------------------
        # B. Regression
        # ------------------------------------------------------

        regression = (
            run_stage6_regression()
        )

        print(
            "Stage6 regression      = 229/229 PASS"
        )

        # ------------------------------------------------------
        # C. Lock ALL non-oracle controller outputs
        # BEFORE any future evaluator/oracle data is opened.
        # ------------------------------------------------------

        (
            reactive,
            maps,
            causal_locks,
        ) = (
            lock_nonoracle_controller_outputs()
        )

        print(
            "non-oracle locks        = PASS"
        )

        print(
            "future evaluator read   = NOT YET"
        )

        # ------------------------------------------------------
        # D. Only now future evaluator/oracle side is opened.
        # ------------------------------------------------------

        oracle = (
            load_oracle_after_lock()
        )

        evaluator_index = (
            evaluator_index_after_lock()
        )

        if len(
            evaluator_index
        ) != 120:
            raise FailClosed(
                "Evaluator scenario count != 120"
            )

        print(
            "oracle/evaluator        = OPENED AFTER LOCK"
        )

        # ------------------------------------------------------
        # E. Frozen resource evidence, explicit exact file
        # ------------------------------------------------------

        resources = (
            load_resource_evidence()
        )

        common_latency = float(
            resources[
                "adb_actuation_latency"
            ][
                "mean_s"
            ]
        )

        print(
            "latency/resources       = PASS"
        )

        # ------------------------------------------------------
        # F. One real scenario smoke
        # ------------------------------------------------------

        smoke = (
            one_scenario_smoke(
                evaluator_index,
                reactive,
                maps,
                oracle,
            )
        )

        print(
            "one-scenario smoke      = PASS"
        )

        # ------------------------------------------------------
        # G. Full Experiment-5 pass 1
        # ------------------------------------------------------

        (
            matrix_1,
            support_1,
            rows_1,
        ) = (
            evaluate_once(
                evaluator_index,
                reactive,
                maps,
                oracle,
                common_actuation_latency_s=(
                    common_latency
                ),
            )
        )

        validate_matrix(
            matrix_1
        )

        print(
            "Experiment-5 pass1      = 84/84 numeric"
        )

        # ------------------------------------------------------
        # H. Frozen V3 identity
        # ------------------------------------------------------

        v3_identity = (
            validate_v3_identity(
                matrix_1
            )
        )

        print(
            "frozen V3 identity      = PASS"
        )

        # ------------------------------------------------------
        # I. Full deterministic pass 2
        # ------------------------------------------------------

        (
            matrix_2,
            support_2,
            rows_2,
        ) = (
            evaluate_once(
                evaluator_index,
                reactive,
                maps,
                oracle,
                common_actuation_latency_s=(
                    common_latency
                ),
            )
        )

        validate_matrix(
            matrix_2
        )

        reproducibility_payload_1 = (
            canonical_bytes(
                {
                    "matrix":
                        matrix_1,

                    "finite_support":
                        support_1,

                    "per_scenario":
                        rows_1,
                }
            )
        )

        reproducibility_payload_2 = (
            canonical_bytes(
                {
                    "matrix":
                        matrix_2,

                    "finite_support":
                        support_2,

                    "per_scenario":
                        rows_2,
                }
            )
        )

        if (
            reproducibility_payload_1
            !=
            reproducibility_payload_2
        ):
            raise FailClosed(
                "Experiment-5 reproducibility failed."
            )

        reproducibility = {
            "status":
                "PASS",

            "exact_matrix_and_per_scenario_bytes_identical":
                True,

            "pass1_sha256":
                sha256_bytes(
                    reproducibility_payload_1
                ),

            "pass2_sha256":
                sha256_bytes(
                    reproducibility_payload_2
                ),

            "scenario_count":
                120,

            "baseline_count":
                7,

            "metric_count":
                12,

            "post_outcome_tuning":
                False,
        }

        print(
            "reproducibility         = PASS"
        )

        # ------------------------------------------------------
        # J. Ablation / safety evidence
        # ------------------------------------------------------

        safety = (
            build_safety_ablation(
                matrix_1
            )
        )

        print(
            "safety/ablation         = PASS"
        )

        # ------------------------------------------------------
        # K. Complete Experiment-5 record
        # ------------------------------------------------------

        experiment5 = {
            "status":
                "PASS",

            "stage":
                6,

            "experiment":
                "Experiment 5 — ADB",

            "evaluation_unit":
                (
                    "one scenario with complete "
                    "[4,501,301] horizon schedule"
                ),

            "scenario_count":
                120,

            "baseline_count":
                7,

            "metric_count":
                12,

            "numeric_cells":
                84,

            "baselines":
                list(
                    BASELINES
                ),

            "metrics":
                list(
                    METRICS
                ),

            "matrix":
                matrix_1,

            "finite_scenario_support":
                support_1,

            "aggregation": {
                "scenario_weighting":
                    "equal",

                "actor_count_weighting":
                    False,

                "horizon_reweighting":
                    False,

                "NA_rule":
                    (
                        "authoritative NaN excluded only "
                        "from that metric aggregate"
                    ),

                "global_N120_flattening":
                    False,
            },

            "road_ROI": {
                "rule":
                    (
                        "all_valid_cells_of_frozen_"
                        "theta_range_headlamp_actuator_grid"
                    ),

                "mask_value_on_valid_grid":
                    True,

                "future_truth_dependent":
                    False,

                "outcome_dependent":
                    False,
            },

            "temporal_metrics": {
                "current_initial_authority":
                    (
                        "frozen original-reactive t0 map"
                    ),

                "evaluation_horizons_s":
                    list(
                        HORIZONS_S
                    ),

                "transition_delta_t_s":
                    list(
                        TEMPORAL_DELTA_T_S
                    ),

                "flicker_extra_threshold":
                    False,
            },

            "actuation_latency_cell_semantics":
                resources[
                    "matrix_actuation_latency_semantics"
                ],

            "oracle_future_ADB": {
                "role":
                    "constructed evaluator reference",

                "measured_ADB_ground_truth":
                    False,

                "future_GT_controller_input":
                    False,
            },

            "causality": {
                "non_oracle_controller_outputs_locked_before_future_scoring":
                    True,

                "causal_locks":
                    causal_locks,

                "future_GT_nonoracle_controller_input":
                    False,

                "post_outcome_tuning":
                    False,

                "V3_modified":
                    False,
            },

            "one_scenario_smoke":
                smoke,

            "v3_primary_identity":
                v3_identity,
        }

        # ------------------------------------------------------
        # L. Final full-PDF Stage6 gate
        # ------------------------------------------------------

        fullpdf = {
            "status":
                "PASS_100_PERCENT",

            "stage":
                "Stage 6",

            "stage6_V3_primary_gate":
                "PASS_FROZEN_IMMUTABLE",

            "regression":
                "229/229 PASS",

            "Experiment5": {
                "baselines":
                    "7/7",

                "metrics":
                    "12/12",

                "numeric_cells":
                    84,

                "scenario_macro_aggregation":
                    "PASS",

                "NA_semantics":
                    "PASS",

                "explicit_metric_binding":
                    "PASS",

                "heuristic_metric_discovery_used":
                    False,
            },

            "vehicle_shadow_reduction":
                "PASS",

            "overmask_noninferiority":
                "PASS",

            "pedestrian_visibility_gate":
                "PASS",

            "cyclist_visibility_gate":
                "PASS",

            "V3_road_illumination_identity":
                "PASS",

            "resource_evidence":
                "PASS",

            "reproducibility":
                "PASS",

            "V3_modified":
                False,

            "post_outcome_tuning":
                False,

            "Stage4_recalibration":
                False,

            "future_GT_nonoracle_controller_input":
                False,

            "oracle_future_ADB_measured_GT":
                False,

            "stage7_allowed":
                True,
        }

        # ------------------------------------------------------
        # M. Prepare ALL output bytes before writing anything.
        # ------------------------------------------------------

        matrix_bytes = (
            canonical_bytes(
                experiment5
            )
        )

        csv_bytes = (
            matrix_csv_bytes(
                matrix_1
            )
        )

        safety_bytes = (
            canonical_bytes(
                safety
            )
        )

        resource_bytes = (
            canonical_bytes(
                resources
            )
        )

        repro_bytes = (
            canonical_bytes(
                reproducibility
            )
        )

        fullpdf_bytes = (
            canonical_bytes(
                fullpdf
            )
        )

        per_scenario_bytes = (
            jsonl_bytes(
                rows_1
            )
        )

        evidence_hashes = {
            str(
                FINAL_MATRIX
            ):
                sha256_bytes(
                    matrix_bytes
                ),

            str(
                FINAL_MATRIX_CSV
            ):
                sha256_bytes(
                    csv_bytes
                ),

            str(
                FINAL_SAFETY
            ):
                sha256_bytes(
                    safety_bytes
                ),

            str(
                FINAL_RESOURCE_SEAL
            ):
                sha256_bytes(
                    resource_bytes
                ),

            str(
                FINAL_REPRO
            ):
                sha256_bytes(
                    repro_bytes
                ),

            str(
                FINAL_FULLPDF_AUDIT
            ):
                sha256_bytes(
                    fullpdf_bytes
                ),

            str(
                FINAL_PER_SCENARIO
            ):
                sha256_bytes(
                    per_scenario_bytes
                ),
        }

        freeze = {
            "status":
                "PASS_100_PERCENT",

            "stage":
                "Stage 6",

            "experiment5":
                "COMPLETE",

            "numeric_cells":
                84,

            "stage5_final_handoff":
                frozen[
                    "stage5_handoff"
                ],

            "stage6_v3":
                frozen[
                    "stage6_v3"
                ],

            "metric_source_sha256":
                EXPECTED_METRIC_SOURCE_SHA256,

            "frozen_input_sha256": {
                "reactive_t0":
                    EXPECTED_REACTIVE_SHA256,

                "deterministic_predictive":
                    EXPECTED_DETERMINISTIC_SHA256,

                "uncertainty_common":
                    EXPECTED_COMMON_SHA256,

                "class_aware_V3":
                    EXPECTED_CLASS_AWARE_SHA256,

                "oracle_future":
                    EXPECTED_ORACLE_SHA256,

                "resource_evidence":
                    EXPECTED_RESOURCE_EVIDENCE_SHA256,
            },

            "controller_locks":
                causal_locks,

            "evidence_output_sha256":
                evidence_hashes,

            "regression":
                regression,

            "V3_immutable":
                True,

            "V3_recomputed":
                False,

            "V3_retuned":
                False,

            "threshold_change":
                False,

            "Stage4_recalibration":
                False,

            "future_GT_nonoracle_controller_input":
                False,

            "post_outcome_tuning":
                False,

            "oracle_future_ADB_measured_GT":
                False,

            "stage7_allowed":
                True,
        }

        freeze_bytes = (
            canonical_bytes(
                freeze
            )
        )

        freeze_sha = (
            sha256_bytes(
                freeze_bytes
            )
        )

        certificate = {
            "status":
                "PASS_100_PERCENT",

            "stage":
                "Stage 6",

            "pdf_fidelity":
                "PASS_100_PERCENT",

            "experiment5_numeric_matrix": {
                "path":
                    str(
                        FINAL_MATRIX
                    ),

                "sha256":
                    sha256_bytes(
                        matrix_bytes
                    ),

                "numeric_cells":
                    84,
            },

            "per_scenario_evidence": {
                "path":
                    str(
                        FINAL_PER_SCENARIO
                    ),

                "sha256":
                    sha256_bytes(
                        per_scenario_bytes
                    ),

                "rows":
                    120 * 7,
            },

            "resource_seal": {
                "path":
                    str(
                        FINAL_RESOURCE_SEAL
                    ),

                "sha256":
                    sha256_bytes(
                        resource_bytes
                    ),
            },

            "reproducibility_seal": {
                "path":
                    str(
                        FINAL_REPRO
                    ),

                "sha256":
                    sha256_bytes(
                        repro_bytes
                    ),
            },

            "fullpdf_final_evidence_audit": {
                "path":
                    str(
                        FINAL_FULLPDF_AUDIT
                    ),

                "sha256":
                    sha256_bytes(
                        fullpdf_bytes
                    ),
            },

            "freeze_manifest": {
                "path":
                    str(
                        FINAL_FREEZE
                    ),

                "sha256":
                    freeze_sha,
            },

            "stage5_final_handoff_sha256":
                EXPECTED_STAGE5_HANDOFF_SHA256,

            "frozen_V3_closure_sha256":
                EXPECTED_V3_HASHES[
                    "closure"
                ],

            "oracle_future_ADB":
                "constructed evaluator-only",

            "measured_ADB_ground_truth":
                False,

            "stage7_allowed":
                True,
        }

        certificate_bytes = (
            canonical_bytes(
                certificate
            )
        )

        certificate_sha = (
            sha256_bytes(
                certificate_bytes
            )
        )

        handoff = {
            "status":
                "READY_FOR_STAGE7",

            "from_stage":
                "Stage 6",

            "to_stage":
                "Stage 7",

            "stage6_certificate":
                str(
                    FINAL_CERTIFICATE
                ),

            "stage6_certificate_sha256":
                certificate_sha,

            "stage6_freeze_manifest":
                str(
                    FINAL_FREEZE
                ),

            "stage6_freeze_manifest_sha256":
                freeze_sha,

            "stage5_final_handoff_sha256":
                EXPECTED_STAGE5_HANDOFF_SHA256,

            "shared_posterior_required":
                True,

            "V3_frozen":
                True,

            "experiment5_complete":
                True,

            "experiment5_numeric_cells":
                84,

            "stage7_allowed":
                True,
        }

        handoff_bytes = (
            canonical_bytes(
                handoff
            )
        )

        payloads = {
            FINAL_MATRIX:
                matrix_bytes,

            FINAL_MATRIX_CSV:
                csv_bytes,

            FINAL_SAFETY:
                safety_bytes,

            FINAL_RESOURCE_SEAL:
                resource_bytes,

            FINAL_REPRO:
                repro_bytes,

            FINAL_FULLPDF_AUDIT:
                fullpdf_bytes,

            FINAL_PER_SCENARIO:
                per_scenario_bytes,

            FINAL_FREEZE:
                freeze_bytes,

            FINAL_CERTIFICATE:
                certificate_bytes,

            FINAL_HANDOFF:
                handoff_bytes,
        }

        # ------------------------------------------------------
        # N. Fail-closed all-or-nothing preflight
        # ------------------------------------------------------

        atomic_preflight(
            payloads
        )

        atomic_publish(
            payloads
        )

        # ------------------------------------------------------
        # O. Final readback SHA checks
        # ------------------------------------------------------

        for path, expected_bytes in (
            payloads.items()
        ):

            actual = path.read_bytes()

            if actual != expected_bytes:
                raise FailClosed(
                    f"Published artifact changed: {path}"
                )

        print(
            "publish                = PASS"
        )
        print(
            "Experiment-5 matrix   = 84/84 numeric"
        )
        print(
            "scenario rows         =",
            len(
                rows_1
            ),
        )
        print(
            "frozen V3 identity    = PASS"
        )
        print(
            "full-PDF evidence     = PASS_100_PERCENT"
        )
        print(
            "certificate           =",
            FINAL_CERTIFICATE,
        )
        print(
            "certificate SHA256    =",
            certificate_sha,
        )
        print(
            "handoff               =",
            FINAL_HANDOFF,
        )
        print(
            "handoff SHA256        =",
            sha256_bytes(
                handoff_bytes
            ),
        )
        print(
            "Stage7_allowed        = true"
        )
        print(
            "STATUS                = PASS_100_PERCENT"
        )

        return 0

    except Exception as exc:

        print()
        print(
            "============================================================"
        )
        print(
            "FAIL_CLOSED"
        )
        print(
            "============================================================"
        )
        print(
            type(exc).__name__
            + ": "
            + str(exc)
        )
        print(
            "No tuning performed."
        )
        print(
            "No V3 controller regenerated."
        )
        print(
            "Stage7_allowed = false"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

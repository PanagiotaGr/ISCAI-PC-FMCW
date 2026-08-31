from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScalarMinimizeResult:
    def __init__(self, x, fun, success=True):
        self.x = float(x)
        self.fun = float(fun)
        self.success = bool(success)


def minimize_scalar(
    fun,
    *,
    bounds,
    method="bounded",
):
    """
    Deterministic dependency-free bounded scalar minimizer.

    Golden-section search over a closed finite interval.
    This replaces scipy.optimize.minimize_scalar solely for
    the Stage-4 calibration scalar search.
    """
    if method != "bounded":
        raise ValueError(
            "Only method='bounded' is supported."
        )

    a = float(bounds[0])
    b = float(bounds[1])

    if not (
        math.isfinite(a)
        and math.isfinite(b)
        and a < b
    ):
        raise ValueError(
            f"Invalid bounded interval: {bounds}"
        )

    inv_phi = (
        math.sqrt(5.0) - 1.0
    ) / 2.0

    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)

    fc = float(fun(c))
    fd = float(fun(d))

    # Fixed iteration count => deterministic across runs.
    for _ in range(96):
        if fc <= fd:
            b = d
            d = c
            fd = fc

            c = (
                b
                - inv_phi
                * (b - a)
            )

            fc = float(
                fun(c)
            )

        else:
            a = c
            c = d
            fc = fd

            d = (
                a
                + inv_phi
                * (b - a)
            )

            fd = float(
                fun(d)
            )

    candidates = [
        (
            a,
            float(fun(a)),
        ),
        (
            b,
            float(fun(b)),
        ),
        (
            c,
            fc,
        ),
        (
            d,
            fd,
        ),
        (
            0.5 * (a + b),
            float(
                fun(
                    0.5 * (a + b)
                )
            ),
        ),
    ]

    x, value = min(
        candidates,
        key=lambda item:
            (
                item[1],
                item[0],
            ),
    )

    return _ScalarMinimizeResult(
        x=x,
        fun=value,
        success=True,
    )


class _ChiSquareDistribution:
    """
    Dependency-free chi-square quantile implementation.

    Stage-4 calibration uses df=3 because the predictive
    Gaussian is full 3-D.

    For df=3:

        F(x) = erf(sqrt(x/2))
               - sqrt(2*x/pi) * exp(-x/2)

    PPF is obtained by deterministic bisection.
    """

    @staticmethod
    def _cdf_df3(x):
        x = float(x)

        if x <= 0.0:
            return 0.0

        value = (
            math.erf(
                math.sqrt(
                    x / 2.0
                )
            )
            -
            math.sqrt(
                2.0 * x / math.pi
            )
            * math.exp(
                -x / 2.0
            )
        )

        return min(
            1.0,
            max(
                0.0,
                float(value),
            ),
        )

    @classmethod
    def ppf(
        cls,
        probability,
        *,
        df,
    ):
        probability = float(
            probability
        )

        df = int(df)

        if df != 3:
            raise ValueError(
                "Stage4 dependency-free chi2.ppf "
                "supports exactly df=3."
            )

        if not (
            0.0
            <= probability
            <= 1.0
        ):
            raise ValueError(
                "Probability must lie in [0,1]."
            )

        if probability == 0.0:
            return 0.0

        if probability == 1.0:
            return math.inf

        lower = 0.0
        upper = 1.0

        while (
            cls._cdf_df3(
                upper
            )
            < probability
        ):
            upper *= 2.0

            if upper > 1.0e6:
                raise RuntimeError(
                    "Could not bracket chi-square quantile."
                )

        # Fixed iterations => deterministic.
        for _ in range(160):
            middle = (
                lower + upper
            ) / 2.0

            if (
                cls._cdf_df3(
                    middle
                )
                < probability
            ):
                lower = middle
            else:
                upper = middle

        return (
            lower + upper
        ) / 2.0


chi2 = _ChiSquareDistribution()
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
)


# =====================================================================
# PATHS / FROZEN PROVENANCE
# =====================================================================

ROOT = Path("/home/agni/waymo")
S0 = ROOT / "iscai_stage0"
S1 = ROOT / "iscai_stage1"
S2 = ROOT / "iscai_stage2"
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"

PAIRED = (
    ROOT
    / "data"
    / "paired_womd_lidar_v1_3_0"
)

AUDIT = (
    ROOT
    / "audits"
    / "stage4_fullpdf_v2"
)

PROTOCOL = (
    S4
    / "configs"
    / "stage4_fullpdf_v2_protocol.json"
)

TRAIN_FREEZE = (
    S4
    / "configs"
    / "stage4_fullpdf_v2_training_freeze.json"
)

FIT_MANIFEST = (
    S4
    / "artifacts"
    / "block41"
    / "fit.jsonl"
)

DEV_MANIFEST = (
    S4
    / "artifacts"
    / "block41"
    / "development.jsonl"
)

CAL_MANIFEST = (
    S4
    / "artifacts"
    / "block41"
    / "calibration.jsonl"
)

OUT = (
    S4
    / "artifacts"
    / "fullpdf_v2"
)

CACHE = OUT / "cache"
MODELS = OUT / "models"

REPORTS = S4 / "reports"

REPAIR_REPORT = (
    REPORTS
    / "stage4_fullpdf_v2_repair.json"
)

STATUS_REPORT = (
    REPORTS
    / "stage4_fullpdf_v2_status.json"
)

FAIL_REPORT = (
    AUDIT
    / "stage4_fullpdf_v2_failure.json"
)

LEGACY_FORMAL = (
    S4
    / "reports"
    / "block48_formal_evaluation.json"
)

LEGACY_FORMAL_NPZ = (
    S4
    / "artifacts"
    / "block48"
    / "formal_neural_outputs.npz"
)

LEGACY_CLOSURE = (
    S4
    / "reports"
    / "stage4_final_closure.json"
)


EXPECTED_SHA = {
    "fit":
        "284a61c877d937deb07137345beaabf3"
        "165690138785f9cc41f81655066d5276",

    "development":
        "e4689698bddd80e58add7267f791aea9"
        "0ef4bef0309ca18d0d7a9e7e94fe7e5c",

    "calibration":
        "e003dd5c4d5a253729700b12c3754c4"
        "7ffb99dadfa035b46627f01ec91eed4be",

    "normalization_post_fix":
        "971c45c0fc5d8f8778bf926effee19e5"
        "b022772a5f1f5ec518062eadcfe833c0",
}


REQUIRED_PDF_ABLATIONS = [
    "position_only",
    "position_plus_annotated_velocity",
    "position_plus_past_only_estimated_velocity",
    "position_plus_geometry_derived_radial_velocity",
    "position_plus_PC_FMCW_noisy_radial_velocity",
    "position_plus_measurement_covariance",
    "position_plus_actor_level_LiDAR_features",
    "actor_only_vs_multi_agent",
    "no_map_vs_map",
    "no_LiDAR_vs_LiDAR_context",
]


# =====================================================================
# PRE-OUTCOME TRAINING FREEZE
# =====================================================================

TRAINING_FREEZE = {
    "stage": 4,

    "status":
        "FROZEN_BEFORE_V2_TRAINING_OUTCOMES",

    "seed":
        20260826,

    "partition_policy": {
        "fit_scenarios":
            4096,

        "development_scenarios":
            512,

        "calibration_scenarios":
            512,

        "selection":
            "first_N_of_existing_"
            "SHA_ordered_frozen_partitions",

        "future_based":
            False,

        "performance_based":
            False,
    },

    "sampling": {
        "sample_cap_per_scene":
            32,

        "class_balanced":
            True,

        "method":
            "inverse_class_frequency_"
            "WeightedRandomSampler",
    },

    "LiDAR_ablation": {
        "fit_scenarios":
            1000,

        "development_scenarios":
            256,

        "selection":
            "first_N_of_same_frozen_partitions",

        "semantics":
            "causal_annotation_assisted_"
            "actor_box_ablation_only",

        "primary_sensor_to_track_claim":
            False,
    },

    "primary": {
        "model":
            "full_covariance_Gaussian_GRU",

        "multi_agent":
            True,

        "map_context":
            True,

        "actor_LiDAR":
            False,

        "annotated_velocity":
            False,

        "actual_Stage2_noisy_vr":
            True,

        "measurement_covariance":
            True,

        "future_input":
            False,

        "perfect_track_ID_numeric_input":
            False,
    },

    "architecture": {
        "target_hidden":
            64,

        "neighbor_hidden":
            32,

        "map_hidden":
            32,

        "fusion_hidden":
            128,

        "GMM_components":
            3,

        "GMM_semantics":
            "trajectory_level_shared_mode",

        "Gaussian_covariance":
            "full_3D_SPD_per_horizon",
    },

    "optimizer": {
        "name":
            "AdamW",

        "learning_rate":
            1.0e-3,

        "weight_decay":
            1.0e-4,

        "batch_size":
            512,

        "gradient_clip":
            5.0,
    },

    "epochs": {
        "ablation_max":
            12,

        "primary_deterministic_max":
            20,

        "Gaussian_max":
            25,

        "GMM_max":
            15,

        "early_stopping_patience":
            5,

        "early_stopping_source":
            "development_only",
    },

    "normalization": {
        "source":
            "fit_only",

        "featurewise_validity_aware":
            True,

        "label_source":
            "fit_only",
    },

    "calibration": {
        "source":
            "calibration_partition_only",

        "method":
            "per_horizon_scalar_"
            "predictive_covariance_scaling",

        "confidence_levels": [
            0.50,
            0.80,
            0.90,
            0.95,
            0.99,
        ],

        "measurement_R_modified":
            False,

        "predictive_mean_modified":
            False,
    },

    "formal": {
        "run_in_repair":
            False,

        "allowed_only_after_freeze":
            True,

        "comparison":
            "exact_common_actor_horizon_"
            "support_vs_repaired_Stage3",
    },

    "required_PDF_input_ablations":
        REQUIRED_PDF_ABLATIONS,
}


# =====================================================================
# FEATURE SCHEMA
# =====================================================================

HISTORY_STEPS = 11
HORIZONS = (0.1, 0.3, 0.5, 1.0)

MAP_DIM = 10
MAX_NEIGHBORS = 8

FEATURE_NAMES = (
    # estimated Cartesian state
    "pos_x",
    "pos_y",
    "pos_z",

    "past_est_vx",
    "past_est_vy",
    "past_est_vz",

    # actual Stage2 degraded measurement
    "pcfmcw_noisy_vr",

    # causal derived reference
    "geometry_derived_vr",

    # WOMD historical annotation upper bound
    "annotated_vx",
    "annotated_vy",
    "annotated_vz",

    # Cartesian position covariance
    "pos_cov_xx",
    "pos_cov_xy",
    "pos_cov_xz",
    "pos_cov_yy",
    "pos_cov_yz",
    "pos_cov_zz",

    # raw Stage2 measurement R:
    # [r, vr, az, el] upper triangle
    "R_rr",
    "R_rvr",
    "R_raz",
    "R_rel",
    "R_vrvr",
    "R_vraz",
    "R_vrel",
    "R_azaz",
    "R_azel",
    "R_elel",

    # masks
    "observed",
    "past_est_velocity_valid",
    "pcfmcw_noisy_vr_valid",
    "geometry_vr_valid",
    "annotated_velocity_valid",

    # actor-level LiDAR
    "lidar_log1p_point_count",
    "lidar_range_mean",
    "lidar_range_std",
    "lidar_intensity_mean",
    "lidar_intensity_median",
    "lidar_intensity_variance",
    "lidar_elongation_mean",
    "lidar_elongation_variance",
    "lidar_spread_x",
    "lidar_spread_y",
    "lidar_spread_z",
    "lidar_valid",
)

FI = {
    name: index
    for index, name
    in enumerate(FEATURE_NAMES)
}

FEATURE_DIM = len(FEATURE_NAMES)

if FEATURE_DIM != 44:
    raise RuntimeError(
        f"Unexpected feature dimension {FEATURE_DIM}"
    )


POS = list(range(0, 3))
EST_VEL = list(range(3, 6))
NOISY_VR = [6]
GEOMETRY_VR = [7]
ANNOTATED_VEL = list(range(8, 11))
POS_COV = list(range(11, 17))
RAW_R4 = list(range(17, 27))
LIDAR = list(range(32, 44))

MASK_INDICES = {
    FI["observed"],
    FI["past_est_velocity_valid"],
    FI["pcfmcw_noisy_vr_valid"],
    FI["geometry_vr_valid"],
    FI["annotated_velocity_valid"],
    FI["lidar_valid"],
}


PRIMARY_FEATURES = sorted(
    set(
        POS
        + EST_VEL
        + NOISY_VR
        + POS_COV
        + RAW_R4
        + [
            FI["observed"],
            FI["past_est_velocity_valid"],
            FI["pcfmcw_noisy_vr_valid"],
        ]
    )
)


VARIANTS = {
    "position_only": {
        "features":
            POS
            + [
                FI["observed"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,
    },

    # Auxiliary exact-support counterpart for LiDAR pair.
    "position_only_LiDARsubset": {
        "features":
            POS
            + [
                FI["observed"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            True,
    },

    "position_plus_annotated_velocity": {
        "features":
            POS
            + ANNOTATED_VEL
            + [
                FI["observed"],
                FI["annotated_velocity_valid"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,

        "semantics":
            "historical_WOMD_annotation_"
            "upper_bound_only",
    },

    "position_plus_past_only_estimated_velocity": {
        "features":
            POS
            + EST_VEL
            + [
                FI["observed"],
                FI["past_est_velocity_valid"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,
    },

    "position_plus_geometry_derived_radial_velocity": {
        "features":
            POS
            + GEOMETRY_VR
            + [
                FI["observed"],
                FI["geometry_vr_valid"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,
    },

    "position_plus_PC_FMCW_noisy_radial_velocity": {
        "features":
            POS
            + NOISY_VR
            + [
                FI["observed"],
                FI["pcfmcw_noisy_vr_valid"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,

        "semantics":
            "actual_Stage2_degraded_"
            "radial_velocity_mps",
    },

    "position_plus_measurement_covariance": {
        "features":
            POS
            + POS_COV
            + RAW_R4
            + [
                FI["observed"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,
    },

    "position_plus_actor_level_LiDAR_features": {
        "features":
            POS
            + LIDAR
            + [
                FI["observed"],
            ],

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            True,

        "semantics":
            "causal_annotation_assisted_"
            "actor_box_LiDAR_ablation",
    },

    "actor_only": {
        "features":
            PRIMARY_FEATURES,

        "neighbors":
            False,

        "map":
            False,

        "lidar_subset":
            False,
    },

    "multi_agent": {
        "features":
            PRIMARY_FEATURES,

        "neighbors":
            True,

        "map":
            False,

        "lidar_subset":
            False,
    },

    "no_map": {
        "features":
            PRIMARY_FEATURES,

        "neighbors":
            True,

        "map":
            False,

        "lidar_subset":
            False,
    },

    # deterministic primary
    "map": {
        "features":
            PRIMARY_FEATURES,

        "neighbors":
            True,

        "map":
            True,

        "lidar_subset":
            False,
    },

    "no_LiDAR": {
        "features":
            PRIMARY_FEATURES,

        "neighbors":
            True,

        "map":
            True,

        "lidar_subset":
            True,
    },

    "LiDAR_context": {
        "features":
            sorted(
                set(
                    PRIMARY_FEATURES
                    + LIDAR
                )
            ),

        "neighbors":
            True,

        "map":
            True,

        "lidar_subset":
            True,

        "semantics":
            "target_actor_causal_annotation_"
            "assisted_LiDAR_context",
    },
}


ABLATION_PAIRS = {
    "position_only_vs_annotated_velocity":
        (
            "position_only",
            "position_plus_annotated_velocity",
        ),

    "position_only_vs_past_only_estimated_velocity":
        (
            "position_only",
            "position_plus_past_only_estimated_velocity",
        ),

    "position_only_vs_geometry_derived_radial_velocity":
        (
            "position_only",
            "position_plus_geometry_derived_radial_velocity",
        ),

    "position_only_vs_PC_FMCW_noisy_radial_velocity":
        (
            "position_only",
            "position_plus_PC_FMCW_noisy_radial_velocity",
        ),

    "position_only_vs_measurement_covariance":
        (
            "position_only",
            "position_plus_measurement_covariance",
        ),

    "position_only_vs_actor_level_LiDAR_features":
        (
            "position_only_LiDARsubset",
            "position_plus_actor_level_LiDAR_features",
        ),

    "actor_only_vs_multi_agent":
        (
            "actor_only",
            "multi_agent",
        ),

    "no_map_vs_map":
        (
            "no_map",
            "map",
        ),

    "no_LiDAR_vs_LiDAR_context":
        (
            "no_LiDAR",
            "LiDAR_context",
        ),
}


# =====================================================================
# BASIC UTILITIES
# =====================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def atomic_npz(path, **arrays):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp.npz"
    )

    np.savez_compressed(
        tmp,
        **arrays,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_torch_save(path, value):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        value,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def read_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def read_jsonl(path, limit=None):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

            if (
                limit is not None
                and len(rows) >= limit
            ):
                break

    return rows


def freeze_json(path, value):
    path = Path(path)

    if path.exists():
        existing = read_json(
            path
        )

        if existing != value:
            raise RuntimeError(
                "Frozen V2 configuration "
                "differs; refusing post-outcome "
                f"rewrite: {path}"
            )

    else:
        atomic_json(
            path,
            value,
        )

    return sha256_file(
        path
    )


def free_gib():
    return (
        shutil.disk_usage(
            ROOT
        ).free
        / (1024 ** 3)
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def actor_class_id(value):
    value = str(value).upper()

    if "VEHICLE" in value:
        return 0

    if "PEDESTRIAN" in value:
        return 1

    if "CYCLIST" in value:
        return 2

    raise RuntimeError(
        f"Unsupported actor class {value!r}"
    )


def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)
    sys.stdout.flush()


def formal_seal():
    result = {}

    for path in (
        LEGACY_FORMAL,
        LEGACY_FORMAL_NPZ,
        LEGACY_CLOSURE,
    ):
        result[str(path)] = (
            sha256_file(path)
            if path.is_file()
            else None
        )

    return result


# =====================================================================
# STAGE2 CONFIG RESOLUTION
# =====================================================================

def resolve_stage2_configs(loader):
    value = loader()

    objects = []

    def collect(x):
        if x is None:
            return

        if isinstance(
            x,
            dict,
        ):
            for item in x.values():
                collect(item)
            return

        if isinstance(
            x,
            (tuple, list),
        ):
            for item in x:
                collect(item)
            return

        objects.append(x)

        for name in (
            "clean_config",
            "degraded_config",
            "clean",
            "degraded",
        ):
            if hasattr(
                x,
                name,
            ):
                objects.append(
                    getattr(
                        x,
                        name,
                    )
                )

    collect(value)

    clean = None
    degraded = None

    for obj in objects:
        name = (
            obj.__class__
            .__name__
            .lower()
        )

        if (
            "cleanobservationconfig"
            in name
        ):
            clean = obj

        if (
            "degradedobservationconfig"
            in name
        ):
            degraded = obj

    if (
        clean is None
        or degraded is None
    ):
        if (
            isinstance(
                value,
                (tuple, list),
            )
            and len(value) >= 2
        ):
            clean = (
                clean
                or value[0]
            )

            degraded = (
                degraded
                or value[1]
            )

    if (
        clean is None
        or degraded is None
    ):
        raise RuntimeError(
            "Could not resolve frozen "
            "Stage2 clean/degraded configs."
        )

    return clean, degraded


# =====================================================================
# REAL CAUSAL PIPELINE COMPONENT RESOLUTION
# =====================================================================

def walk_runtime_objects(root):
    queue = [
        (
            root,
            0,
        )
    ]

    seen = set()

    while queue:
        obj, depth = queue.pop(0)

        if obj is None:
            continue

        key = id(obj)

        if key in seen:
            continue

        seen.add(key)

        yield obj

        if depth >= 4:
            continue

        if isinstance(
            obj,
            dict,
        ):
            queue.extend(
                (
                    value,
                    depth + 1,
                )
                for value
                in obj.values()
            )

        elif isinstance(
            obj,
            (tuple, list),
        ):
            queue.extend(
                (
                    value,
                    depth + 1,
                )
                for value
                in obj
            )

        elif dataclasses.is_dataclass(
            obj
        ):
            for field in dataclasses.fields(
                obj
            ):
                value = getattr(
                    obj,
                    field.name,
                )

                module = (
                    value.__class__
                    .__module__
                )

                if (
                    module.startswith(
                        "iscai_"
                    )
                    or isinstance(
                        value,
                        (
                            dict,
                            tuple,
                            list,
                        ),
                    )
                ):
                    queue.append(
                        (
                            value,
                            depth + 1,
                        )
                    )


def is_transform(value):
    if value is None:
        return False

    return any(
        callable(
            getattr(
                value,
                name,
                None,
            )
        )
        for name in (
            "apply",
            "apply_point",
            "transform_point",
            "apply_vector",
            "rotate_vector",
            "transform_vector",
        )
    )


def transform_from_dynamic_frame(
    frame,
):
    if frame is None:
        return None

    for name in (
        "T_H_from_W",
        "T_Ht_from_W",
        "T_H0_from_W",
        "T_from_W",
    ):
        value = getattr(
            frame,
            name,
            None,
        )

        if is_transform(
            value
        ):
            return value

    try:
        values = vars(
            frame
        ).items()

    except Exception:
        values = []

    for name, value in values:
        if (
            "from_W" in name
            and is_transform(
                value
            )
        ):
            return value

    return None


def transform_vector(
    transform,
    vector,
):
    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    for name in (
        "apply_vector",
        "rotate_vector",
        "transform_vector",
    ):
        fn = getattr(
            transform,
            name,
            None,
        )

        if callable(fn):
            result = np.asarray(
                fn(
                    tuple(
                        float(x)
                        for x in vector
                    )
                ),
                dtype=np.float64,
            )

            if result.shape == (3,):
                return result

    # translation-cancelled point fallback
    for name in (
        "apply",
        "apply_point",
        "transform_point",
    ):
        fn = getattr(
            transform,
            name,
            None,
        )

        if callable(fn):
            origin = np.asarray(
                fn(
                    (
                        0.0,
                        0.0,
                        0.0,
                    )
                ),
                dtype=np.float64,
            )

            endpoint = np.asarray(
                fn(
                    tuple(
                        float(x)
                        for x in vector
                    )
                ),
                dtype=np.float64,
            )

            if (
                origin.shape
                == endpoint.shape
                == (3,)
            ):
                return (
                    endpoint
                    - origin
                )

    for name in (
        "rotation",
        "R",
        "rotation_matrix",
    ):
        if hasattr(
            transform,
            name,
        ):
            matrix = np.asarray(
                getattr(
                    transform,
                    name,
                ),
                dtype=np.float64,
            )

            if matrix.shape == (
                3,
                3,
            ):
                return (
                    matrix
                    @ vector
                )

    raise RuntimeError(
        "Cannot transform vector "
        f"with {type(transform)}"
    )


def causal_components(
    scenario,
    clean_config,
    degraded_config,
):
    from iscai_stage4.data.real_pipeline import (
        FrameTransformContext,
        adapt_causal_womd_scenario,
        algorithm_sequence_from_degraded_scene,
        associate_estimated_gnn,
        build_causal_dynamic_headlamp_frames,
        build_causal_scene_inputs,
        build_clean_observation_scene,
        build_degraded_observation_scene,
        build_real_causal_inputs,
        build_real_ideal_observation_scene,
    )

    result = (
        build_real_causal_inputs(
            scenario,
            clean_config=clean_config,
            degraded_config=(
                degraded_config
            ),
        )
    )

    scene_inputs = None
    association = None
    context = None
    adapted = None
    degraded = None

    for obj in walk_runtime_objects(
        result
    ):
        name = (
            obj.__class__
            .__name__
        )

        if (
            scene_inputs is None
            and
            "CausalSceneInputs"
            in name
            and hasattr(
                obj,
                "histories",
            )
        ):
            scene_inputs = obj

        if (
            association is None
            and
            "Association"
            in name
            and hasattr(
                obj,
                "tracks",
            )
            and hasattr(
                obj,
                "truth_used",
            )
        ):
            association = obj

        if (
            context is None
            and
            "TransformContext"
            in name
            and hasattr(
                obj,
                "T_H0_from_W",
            )
        ):
            context = obj

        if (
            adapted is None
            and name
            == "AdaptedScenario"
        ):
            adapted = obj

        if (
            degraded is None
            and
            "DegradedObservationScene"
            in name
        ):
            degraded = obj

    if adapted is None:
        adapted = (
            adapt_causal_womd_scenario(
                scenario
            )
        )

    if (
        degraded is None
        or association is None
    ):
        ideal = (
            build_real_ideal_observation_scene(
                raw_scenario=scenario,
                adapted=adapted,
                include_sdc=False,
            )
        )

        clean = (
            build_clean_observation_scene(
                ideal_scene=ideal,
                config=clean_config,
            )
        )

        degraded = (
            build_degraded_observation_scene(
                clean_scene=clean,
                config=degraded_config,
            )
        )

        sequence = (
            algorithm_sequence_from_degraded_scene(
                scenario_id=str(
                    scenario.scenario_id
                ),
                scene=degraded,
            )
        )

        association = (
            associate_estimated_gnn(
                sequence
            )
        )

    if bool(
        getattr(
            association,
            "truth_used",
            False,
        )
    ):
        raise RuntimeError(
            "Estimated association "
            "reported truth_used=True."
        )

    if context is None:
        dynamic_frames, _ = (
            build_causal_dynamic_headlamp_frames(
                adapted
            )
        )

        transforms = tuple(
            transform_from_dynamic_frame(
                frame
            )
            for frame
            in dynamic_frames
        )

        anchor = int(
            scenario.current_time_index
        )

        if (
            anchor >= len(
                transforms
            )
            or transforms[
                anchor
            ]
            is None
        ):
            raise RuntimeError(
                "Could not reconstruct "
                "FrameTransformContext."
            )

        context = (
            FrameTransformContext(
                T_H0_from_W=(
                    transforms[
                        anchor
                    ]
                ),
                T_Ht_from_W_by_frame=(
                    transforms
                ),
            )
        )

    if scene_inputs is None:
        scene_inputs = (
            build_causal_scene_inputs(
                association.tracks,
                scenario_id=str(
                    scenario.scenario_id
                ),
                context=context,
                timestamps_s=tuple(
                    float(x)
                    for x
                    in scenario
                    .timestamps_seconds[
                        :
                        int(
                            scenario
                            .current_time_index
                        )
                        + 1
                    ]
                ),
            )
        )

    return (
        scene_inputs,
        association,
        context,
        adapted,
    )


def association_map(
    association,
):
    return {
        str(
            track.track_id
        ):
            track

        for track
        in association.tracks
    }


def detection_map(
    associated_track,
):
    if associated_track is None:
        return {}

    return {
        int(
            item.frame_index
        ):
            item.detection

        for item
        in associated_track.detections
    }


# =====================================================================
# MEASUREMENT COVARIANCES
# =====================================================================

def position_covariance_upper(
    value,
):
    P = np.asarray(
        value,
        dtype=np.float64,
    )

    if P.shape != (
        3,
        3,
    ):
        raise RuntimeError(
            f"Bad P shape {P.shape}"
        )

    P = (
        P
        + P.T
    ) / 2.0

    return np.asarray(
        [
            P[0, 0],
            P[0, 1],
            P[0, 2],
            P[1, 1],
            P[1, 2],
            P[2, 2],
        ],
        dtype=np.float32,
    )


def measurement_R_upper(
    value,
):
    R = np.asarray(
        value,
        dtype=np.float64,
    )

    if R.shape != (
        4,
        4,
    ):
        raise RuntimeError(
            f"Bad R shape {R.shape}"
        )

    if not np.all(
        np.isfinite(
            R
        )
    ):
        raise RuntimeError(
            "Non-finite Stage2 R."
        )

    R = (
        R
        + R.T
    ) / 2.0

    return np.asarray(
        [
            R[0, 0],
            R[0, 1],
            R[0, 2],
            R[0, 3],

            R[1, 1],
            R[1, 2],
            R[1, 3],

            R[2, 2],
            R[2, 3],

            R[3, 3],
        ],
        dtype=np.float32,
    )


# =====================================================================
# HISTORICAL WOMD VELOCITY — UPPER-BOUND ABLATION ONLY
# =====================================================================

def annotated_velocity_H0(
    scenario,
    truth_track_index,
    frame_index,
    T_H0_from_W,
):
    track = (
        scenario.tracks[
            int(
                truth_track_index
            )
        ]
    )

    if not (
        0
        <= frame_index
        < len(
            track.states
        )
    ):
        return (
            np.zeros(
                3,
                dtype=np.float32,
            ),
            False,
        )

    state = (
        track.states[
            frame_index
        ]
    )

    if not bool(
        state.valid
    ):
        return (
            np.zeros(
                3,
                dtype=np.float32,
            ),
            False,
        )

    if (
        hasattr(
            state,
            "velocity_x",
        )
        and
        hasattr(
            state,
            "velocity_y",
        )
    ):
        # WOMD ObjectState velocity is principally planar.
        world_velocity = (
            float(
                state.velocity_x
            ),
            float(
                state.velocity_y
            ),
            float(
                getattr(
                    state,
                    "velocity_z",
                    0.0,
                )
            ),
        )

    elif (
        hasattr(
            state,
            "vel_x",
        )
        and
        hasattr(
            state,
            "vel_y",
        )
    ):
        world_velocity = (
            float(
                state.vel_x
            ),
            float(
                state.vel_y
            ),
            float(
                getattr(
                    state,
                    "vel_z",
                    0.0,
                )
            ),
        )

    else:
        return (
            np.zeros(
                3,
                dtype=np.float32,
            ),
            False,
        )

    velocity = transform_vector(
        T_H0_from_W,
        world_velocity,
    ).astype(
        np.float32
    )

    return (
        velocity,
        bool(
            np.all(
                np.isfinite(
                    velocity
                )
            )
        ),
    )


# =====================================================================
# ACTOR-LEVEL LIDAR ADAPTER
# =====================================================================

def lidar_sidecar(
    scenario_id,
):
    return (
        PAIRED
        / "training"
        / "lidar"
        / scenario_id[:2]
        / (
            scenario_id
            + ".tfrecord"
        )
    )


def scalar_stat(
    value,
    name,
):
    if value is None:
        return 0.0

    if isinstance(
        value,
        (
            float,
            int,
            np.floating,
            np.integer,
        ),
    ):
        if name == "mean":
            return float(value)

        return 0.0

    if isinstance(
        value,
        dict,
    ):
        candidate = value.get(
            name
        )

    else:
        candidate = getattr(
            value,
            name,
            None,
        )

    try:
        candidate = float(
            candidate
        )

    except Exception:
        return 0.0

    return (
        candidate
        if math.isfinite(
            candidate
        )
        else 0.0
    )


def actor_lidar_items(
    result,
):
    if isinstance(
        result,
        dict,
    ):
        return list(
            result.items()
        )

    for name in (
        "actors",
        "actor_histories",
        "histories",
        "features",
    ):
        value = getattr(
            result,
            name,
            None,
        )

        if isinstance(
            value,
            dict,
        ):
            return list(
                value.items()
            )

        if isinstance(
            value,
            (
                tuple,
                list,
            ),
        ):
            return [
                (
                    None,
                    x,
                )
                for x in value
            ]

    if isinstance(
        result,
        (
            tuple,
            list,
        ),
    ):
        return [
            (
                None,
                x,
            )
            for x in result
        ]

    return [
        (
            None,
            result,
        )
    ]


def actor_lidar_map(
    result,
    scenario,
):
    id_to_index = {
        str(
            track.id
        ):
            index

        for index, track
        in enumerate(
            scenario.tracks
        )
    }

    output = {}

    for raw_key, actor in actor_lidar_items(
        result
    ):
        index = getattr(
            actor,
            "track_index",
            None,
        )

        if index is None:
            try:
                index = int(
                    raw_key
                )

            except Exception:
                index = None

        if index is None:
            track_id = getattr(
                actor,
                "track_id",
                raw_key,
            )

            if track_id is not None:
                index = id_to_index.get(
                    str(
                        track_id
                    )
                )

        if (
            index is not None
            and
            0
            <= int(
                index
            )
            < len(
                scenario.tracks
            )
        ):
            output[
                int(
                    index
                )
            ] = actor

    return output


def actor_lidar_frames(
    actor,
):
    if actor is None:
        return {}

    if isinstance(
        actor,
        dict,
    ):
        frames = actor.get(
            "frames"
        )

    else:
        frames = getattr(
            actor,
            "frames",
            None,
        )

    if frames is None:
        return {}

    result = {}

    for default_index, frame in enumerate(
        frames
    ):
        if isinstance(
            frame,
            dict,
        ):
            index = frame.get(
                "time_index",
                frame.get(
                    "frame_index",
                    default_index,
                ),
            )

        else:
            index = getattr(
                frame,
                "time_index",
                getattr(
                    frame,
                    "frame_index",
                    default_index,
                ),
            )

        result[
            int(
                index
            )
        ] = frame

    return result


def lidar_feature_vector(
    frame,
):
    result = np.zeros(
        12,
        dtype=np.float32,
    )

    if frame is None:
        return result

    def get(name):
        if isinstance(
            frame,
            dict,
        ):
            return frame.get(
                name
            )

        return getattr(
            frame,
            name,
            None,
        )

    try:
        count = max(
            0,
            int(
                get(
                    "point_count"
                )
                or 0
            ),
        )

    except Exception:
        count = 0

    range_stats = (
        get(
            "range_m"
        )
        or get(
            "range_stats"
        )
    )

    intensity = (
        get(
            "intensity"
        )
        or get(
            "intensity_stats"
        )
    )

    elongation = (
        get(
            "elongation"
        )
        or get(
            "elongation_stats"
        )
    )

    spread = (
        get(
            "spatial_std_actor_m"
        )
        or get(
            "spatial_std_H0_m"
        )
    )

    try:
        spread = np.asarray(
            spread,
            dtype=np.float64,
        )

        if (
            spread.shape
            != (3,)
            or not np.all(
                np.isfinite(
                    spread
                )
            )
        ):
            spread = np.zeros(
                3
            )

    except Exception:
        spread = np.zeros(
            3
        )

    result[:] = [
        math.log1p(
            count
        ),

        scalar_stat(
            range_stats,
            "mean",
        ),

        scalar_stat(
            range_stats,
            "std",
        ),

        scalar_stat(
            intensity,
            "mean",
        ),

        scalar_stat(
            intensity,
            "median",
        ),

        scalar_stat(
            intensity,
            "variance",
        ),

        scalar_stat(
            elongation,
            "mean",
        ),

        scalar_stat(
            elongation,
            "variance",
        ),

        float(
            spread[0]
        ),

        float(
            spread[1]
        ),

        float(
            spread[2]
        ),

        float(
            count > 0
        ),
    ]

    return result


def extract_actor_lidar(
    scenario,
    context,
    adapted,
):
    from iscai_stage0.womd_proto_io import (
        read_first_scenario,
    )

    from iscai_stage1.lidar.actor_features import (
        extract_causal_actor_lidar,
    )

    from iscai_stage4.data.real_pipeline import (
        build_causal_dynamic_headlamp_frames,
    )

    sid = str(
        scenario.scenario_id
    )

    path = lidar_sidecar(
        sid
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    lidar_scenario = (
        read_first_scenario(
            path
        )
    )

    signature = inspect.signature(
        extract_causal_actor_lidar
    )

    positional = []
    kwargs = {}

    dynamic_frames = None

    aliases = {
        "scenario":
            scenario,

        "motion_scenario":
            scenario,

        "raw_scenario":
            scenario,

        "womd_scenario":
            scenario,

        "lidar_scenario":
            lidar_scenario,

        "lidar_sidecar_scenario":
            lidar_scenario,

        "lidar":
            lidar_scenario,

        "lidar_proto":
            lidar_scenario,

        "lidar_path":
            path,

        "lidar_file":
            path,

        "lidar_tfrecord":
            path,

        "lidar_tfrecord_path":
            path,

        "T_H0_from_W":
            context.T_H0_from_W,

        "transform_H0_from_W":
            context.T_H0_from_W,

        "adapted":
            adapted,

        "adapted_scenario":
            adapted,

        "current_time_index":
            int(
                scenario
                .current_time_index
            ),

        "anchor_time_index":
            int(
                scenario
                .current_time_index
            ),

        "anchor_index":
            int(
                scenario
                .current_time_index
            ),
    }

    for name, parameter in (
        signature
        .parameters
        .items()
    ):
        if name in aliases:
            value = aliases[
                name
            ]

        elif name in (
            "dynamic_headlamp_frames",
            "headlamp_frames",
        ):
            if dynamic_frames is None:
                dynamic_frames, _ = (
                    build_causal_dynamic_headlamp_frames(
                        adapted
                    )
                )

            value = dynamic_frames

        elif (
            parameter.default
            is not inspect._empty
        ):
            continue

        else:
            raise RuntimeError(
                "Unsupported required "
                "extract_causal_actor_lidar "
                f"parameter {name!r}; "
                f"signature={signature}"
            )

        if (
            parameter.kind
            ==
            inspect.Parameter
            .POSITIONAL_ONLY
        ):
            positional.append(
                value
            )

        else:
            kwargs[
                name
            ] = value

    result = (
        extract_causal_actor_lidar(
            *positional,
            **kwargs,
        )
    )

    return (
        actor_lidar_map(
            result,
            scenario,
        ),
        str(
            signature
        ),
    )


# =====================================================================
# V2 FEATURE CONSTRUCTION
# =====================================================================

def build_feature_history(
    history,
    associated_track,
    origin_H0,
    *,
    scenario=None,
    truth_track_index=None,
    T_H0_from_W=None,
    lidar_actor=None,
):
    output = np.zeros(
        (
            HISTORY_STEPS,
            FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    detections = detection_map(
        associated_track
    )

    lidar_frames = (
        actor_lidar_frames(
            lidar_actor
        )
    )

    for step in history.steps:
        frame_index = int(
            step.frame_index
        )

        if not (
            0
            <= frame_index
            < HISTORY_STEPS
        ):
            continue

        row = output[
            frame_index
        ]

        position = np.asarray(
            step.position_H0_m,
            dtype=np.float64,
        )

        velocity = np.asarray(
            step.velocity_H0_mps,
            dtype=np.float64,
        )

        row[0:3] = (
            position
            - origin_H0
        ).astype(
            np.float32
        )

        row[3:6] = (
            velocity.astype(
                np.float32
            )
        )

        row[11:17] = (
            position_covariance_upper(
                step
                .measurement_covariance_H0_m2
            )
        )

        observed = bool(
            step.observed
        )

        velocity_valid = bool(
            step.velocity_valid
        )

        row[
            FI["observed"]
        ] = float(
            observed
        )

        row[
            FI[
                "past_est_velocity_valid"
            ]
        ] = float(
            velocity_valid
        )

        detection = detections.get(
            frame_index
        )

        if detection is not None:
            radial_velocity = float(
                detection
                .radial_velocity_mps
            )

            if math.isfinite(
                radial_velocity
            ):
                row[
                    FI[
                        "pcfmcw_noisy_vr"
                    ]
                ] = radial_velocity

                row[
                    FI[
                        "pcfmcw_noisy_vr_valid"
                    ]
                ] = 1.0

            row[
                17:27
            ] = (
                measurement_R_upper(
                    detection
                    .covariance_4x4
                )
            )

        # Causal geometry-derived radial velocity:
        # radial component of past-only estimated Cartesian velocity.
        norm = float(
            np.linalg.norm(
                position
            )
        )

        if (
            observed
            and velocity_valid
            and norm > 1.0e-6
            and np.all(
                np.isfinite(
                    velocity
                )
            )
        ):
            geometry_vr = float(
                np.dot(
                    position,
                    velocity,
                )
                / norm
            )

            if math.isfinite(
                geometry_vr
            ):
                row[
                    FI[
                        "geometry_derived_vr"
                    ]
                ] = geometry_vr

                row[
                    FI[
                        "geometry_vr_valid"
                    ]
                ] = 1.0

        # Historical WOMD velocity is never in primary.
        if (
            scenario is not None
            and truth_track_index
            is not None
            and T_H0_from_W
            is not None
        ):
            ann_velocity, ann_valid = (
                annotated_velocity_H0(
                    scenario,
                    int(
                        truth_track_index
                    ),
                    frame_index,
                    T_H0_from_W,
                )
            )

            if ann_valid:
                row[
                    8:11
                ] = ann_velocity

                row[
                    FI[
                        "annotated_velocity_valid"
                    ]
                ] = 1.0

        row[
            32:44
        ] = (
            lidar_feature_vector(
                lidar_frames.get(
                    frame_index
                )
            )
        )

    return output


def nearest_histories(
    scene_inputs,
    target_history,
):
    target_position = np.asarray(
        target_history
        .latest_position_H0_m,
        dtype=np.float64,
    )

    values = []

    for history in (
        scene_inputs.histories
    ):
        if (
            str(
                history.prediction_id
            )
            ==
            str(
                target_history
                .prediction_id
            )
        ):
            continue

        position = np.asarray(
            history
            .latest_position_H0_m,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(
                position
            )
        ):
            continue

        distance = float(
            np.linalg.norm(
                position
                - target_position
            )
        )

        values.append(
            (
                distance,
                str(
                    history.prediction_id
                ),
                history,
            )
        )

    values.sort(
        key=lambda x:
            (
                x[0],
                x[1],
            )
    )

    return [
        x[2]
        for x in values[
            :MAX_NEIGHBORS
        ]
    ]


def pooled_neighbor_history(
    scene_inputs,
    target_history,
    association_by_id,
    origin_H0,
):
    histories = nearest_histories(
        scene_inputs,
        target_history,
    )

    if not histories:
        return np.zeros(
            (
                HISTORY_STEPS,
                FEATURE_DIM,
            ),
            dtype=np.float32,
        )

    arrays = []

    for history in histories:
        arrays.append(
            build_feature_history(
                history,
                association_by_id.get(
                    str(
                        history
                        .prediction_id
                    )
                ),
                origin_H0,
            )
        )

    arrays = np.stack(
        arrays,
        axis=0,
    )

    observed = (
        arrays[
            :,
            :,
            FI["observed"]
        ]
        > 0.5
    )

    pooled = np.zeros(
        (
            HISTORY_STEPS,
            FEATURE_DIM,
        ),
        dtype=np.float32,
    )

    for time_index in range(
        HISTORY_STEPS
    ):
        active = observed[
            :,
            time_index,
        ]

        if np.any(
            active
        ):
            pooled[
                time_index
            ] = np.mean(
                arrays[
                    active,
                    time_index,
                    :
                ],
                axis=0,
            )

    return pooled


# =====================================================================
# SCENE CACHE
# =====================================================================

def build_scene_cache(
    record,
    partition,
    rank,
    *,
    lidar_enabled,
    clean_config,
    degraded_config,
):
    from iscai_stage4.data.real_pipeline import (
        read_training_scenario,
    )

    from iscai_stage4.data.supervision import (
        attach_supervision,
    )

    scenario = (
        read_training_scenario(
            record
        )
    )

    (
        scene_inputs,
        association,
        context,
        adapted,
    ) = causal_components(
        scenario,
        clean_config,
        degraded_config,
    )

    # Training labels only.
    # This happens AFTER causal Stage2-degraded tracks/features exist.
    supervised = (
        attach_supervision(
            scene_inputs,
            scenario,
            T_H0_from_W=(
                context
                .T_H0_from_W
            ),
        )
    )

    # Outcome-independent deterministic cap.
    samples = sorted(
        supervised,
        key=lambda sample:
            str(
                sample.sample_id
            ),
    )[
        :
        TRAINING_FREEZE[
            "sampling"
        ][
            "sample_cap_per_scene"
        ]
    ]

    associations = (
        association_map(
            association
        )
    )

    lidar_map = {}
    lidar_signature = (
        "NOT_RUN"
    )

    if lidar_enabled:
        (
            lidar_map,
            lidar_signature,
        ) = extract_actor_lidar(
            scenario,
            context,
            adapted,
        )

    target_rows = []
    neighbor_rows = []
    map_rows = []
    future_rows = []
    future_masks = []
    class_rows = []
    truth_indices = []
    anchor_rows = []
    prediction_ids = []
    sample_ids = []

    for sample in samples:
        history = (
            scene_inputs
            .history_by_id(
                sample.prediction_id
            )
        )

        origin = np.asarray(
            history
            .latest_position_H0_m,
            dtype=np.float64,
        )

        lidar_actor = (
            lidar_map.get(
                int(
                    sample
                    .truth_track_index
                )
            )
            if lidar_enabled
            else None
        )

        target = (
            build_feature_history(
                history,
                associations.get(
                    str(
                        sample
                        .prediction_id
                    )
                ),
                origin,

                scenario=scenario,

                truth_track_index=(
                    int(
                        sample
                        .truth_track_index
                    )
                ),

                T_H0_from_W=(
                    context
                    .T_H0_from_W
                ),

                lidar_actor=(
                    lidar_actor
                ),
            )
        )

        neighbors = (
            pooled_neighbor_history(
                scene_inputs,
                history,
                associations,
                origin,
            )
        )

        future_absolute = np.asarray(
            sample
            .future_label
            .positions_H0_m,
            dtype=np.float32,
        )

        if future_absolute.shape != (
            4,
            3,
        ):
            raise RuntimeError(
                "Unexpected future shape "
                f"{future_absolute.shape}"
            )

        future_displacement = (
            future_absolute
            - origin.astype(
                np.float32
            )[
                None,
                :
            ]
        )

        future_mask = np.asarray(
            sample
            .future_label
            .valid_mask,
            dtype=np.float32,
        )

        map_context = np.asarray(
            sample
            .model_input
            .map_context,
            dtype=np.float32,
        )

        if map_context.shape != (
            MAP_DIM,
        ):
            raise RuntimeError(
                "Unexpected map-context "
                f"shape {map_context.shape}"
            )

        target_rows.append(
            target
        )

        neighbor_rows.append(
            neighbors
        )

        map_rows.append(
            map_context
        )

        future_rows.append(
            future_displacement
        )

        future_masks.append(
            future_mask
        )

        class_rows.append(
            actor_class_id(
                sample.actor_class
            )
        )

        truth_indices.append(
            int(
                sample
                .truth_track_index
            )
        )

        anchor_rows.append(
            origin.astype(
                np.float32
            )
        )

        prediction_ids.append(
            str(
                sample.prediction_id
            )
        )

        sample_ids.append(
            str(
                sample.sample_id
            )
        )

    n = len(
        samples
    )

    return {
        "scenario_id":
            np.asarray(
                str(
                    scenario.scenario_id
                )
            ),

        "partition":
            np.asarray(
                partition
            ),

        "partition_rank":
            np.asarray(
                rank,
                dtype=np.int64,
            ),

        "schema_version":
            np.asarray(
                "stage4_fullpdf_v2_44d"
            ),

        "feature_names":
            np.asarray(
                FEATURE_NAMES
            ),

        "lidar_subset_scene":
            np.asarray(
                bool(
                    lidar_enabled
                )
            ),

        "lidar_extractor_signature":
            np.asarray(
                lidar_signature
            ),

        "target":
            (
                np.stack(
                    target_rows
                )
                if n
                else
                np.zeros(
                    (
                        0,
                        HISTORY_STEPS,
                        FEATURE_DIM,
                    ),
                    dtype=np.float32,
                )
            ),

        "neighbor_pool":
            (
                np.stack(
                    neighbor_rows
                )
                if n
                else
                np.zeros(
                    (
                        0,
                        HISTORY_STEPS,
                        FEATURE_DIM,
                    ),
                    dtype=np.float32,
                )
            ),

        "map_context":
            (
                np.stack(
                    map_rows
                )
                if n
                else
                np.zeros(
                    (
                        0,
                        MAP_DIM,
                    ),
                    dtype=np.float32,
                )
            ),

        "future_displacement":
            (
                np.stack(
                    future_rows
                )
                if n
                else
                np.zeros(
                    (
                        0,
                        4,
                        3,
                    ),
                    dtype=np.float32,
                )
            ),

        "future_mask":
            (
                np.stack(
                    future_masks
                )
                if n
                else
                np.zeros(
                    (
                        0,
                        4,
                    ),
                    dtype=np.float32,
                )
            ),

        "class_id":
            np.asarray(
                class_rows,
                dtype=np.int64,
            ),

        "truth_track_index":
            np.asarray(
                truth_indices,
                dtype=np.int64,
            ),

        "anchor_position_H0":
            (
                np.stack(
                    anchor_rows
                )
                if n
                else
                np.zeros(
                    (
                        0,
                        3,
                    ),
                    dtype=np.float32,
                )
            ),

        "prediction_id":
            np.asarray(
                prediction_ids
            ),

        "sample_id":
            np.asarray(
                sample_ids
            ),
    }


def cache_path(
    partition,
    rank,
    scenario_id,
):
    return (
        CACHE
        / partition
        / (
            f"{rank:05d}_"
            f"{scenario_id}.npz"
        )
    )


def build_partition(
    records,
    partition,
    *,
    lidar_scenes,
    clean_config,
    degraded_config,
):
    header(
        "CACHE "
        + partition.upper()
    )

    manifest = []
    start = time.time()

    for rank, record in enumerate(
        records,
        start=1,
    ):
        scenario_id = str(
            record[
                "scenario_id"
            ]
        )

        path = cache_path(
            partition,
            rank,
            scenario_id,
        )

        lidar_enabled = (
            rank
            <= lidar_scenes
        )

        if path.exists():
            with np.load(
                path,
                allow_pickle=False,
            ) as z:
                if (
                    str(
                        z[
                            "schema_version"
                        ].item()
                    )
                    !=
                    "stage4_fullpdf_v2_44d"
                ):
                    raise RuntimeError(
                        "Existing incompatible "
                        f"V2 cache: {path}"
                    )

        else:
            arrays = (
                build_scene_cache(
                    record,
                    partition,
                    rank,

                    lidar_enabled=(
                        lidar_enabled
                    ),

                    clean_config=(
                        clean_config
                    ),

                    degraded_config=(
                        degraded_config
                    ),
                )
            )

            atomic_npz(
                path,
                **arrays,
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as z:
            count = int(
                z[
                    "class_id"
                ].shape[0]
            )

            class_counts = {
                str(index):
                    int(
                        np.sum(
                            z[
                                "class_id"
                            ]
                            == index
                        )
                    )
                for index
                in (
                    0,
                    1,
                    2,
                )
            }

            lidar_signature = str(
                z[
                    "lidar_extractor_signature"
                ].item()
            )

        manifest.append(
            {
                "rank":
                    rank,

                "scenario_id":
                    scenario_id,

                "path":
                    str(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),

                "samples":
                    count,

                "class_counts":
                    class_counts,

                "lidar_subset_scene":
                    lidar_enabled,

                "lidar_extractor_signature":
                    lidar_signature,
            }
        )

        if (
            rank == 1
            or rank % 50 == 0
            or rank == len(
                records
            )
        ):
            print(
                partition,
                f"{rank}/{len(records)}",
                "elapsed_min=",
                round(
                    (
                        time.time()
                        - start
                    )
                    / 60.0,
                    2,
                ),
                "free_GiB=",
                round(
                    free_gib(),
                    2,
                ),
                flush=True,
            )

        if free_gib() < 250.0:
            raise RuntimeError(
                "250-GiB reserve violated."
            )

    report = {
        "partition":
            partition,

        "scenario_count":
            len(
                manifest
            ),

        "sample_count":
            sum(
                row[
                    "samples"
                ]
                for row
                in manifest
            ),

        "lidar_scene_count":
            sum(
                int(
                    row[
                        "lidar_subset_scene"
                    ]
                )
                for row
                in manifest
            ),

        "feature_dim":
            FEATURE_DIM,

        "feature_names":
            FEATURE_NAMES,

        "rows":
            manifest,
    }

    atomic_json(
        OUT
        / (
            partition
            + "_cache_manifest.json"
        ),
        report,
    )

    return manifest


# =====================================================================
# FIT-ONLY NORMALIZATION
# =====================================================================

def feature_validity(
    values,
):
    valid = np.ones(
        values.shape,
        dtype=bool,
    )

    observed = (
        values[
            ...,
            FI["observed"]
        ]
        > 0.5
    )

    est_valid = (
        values[
            ...,
            FI[
                "past_est_velocity_valid"
            ]
        ]
        > 0.5
    )

    noisy_valid = (
        values[
            ...,
            FI[
                "pcfmcw_noisy_vr_valid"
            ]
        ]
        > 0.5
    )

    geometry_valid = (
        values[
            ...,
            FI[
                "geometry_vr_valid"
            ]
        ]
        > 0.5
    )

    annotated_valid = (
        values[
            ...,
            FI[
                "annotated_velocity_valid"
            ]
        ]
        > 0.5
    )

    lidar_valid = (
        values[
            ...,
            FI[
                "lidar_valid"
            ]
        ]
        > 0.5
    )

    valid[
        ...,
        POS
    ] = observed[
        ...,
        None
    ]

    valid[
        ...,
        EST_VEL
    ] = est_valid[
        ...,
        None
    ]

    valid[
        ...,
        NOISY_VR
    ] = noisy_valid[
        ...,
        None
    ]

    valid[
        ...,
        GEOMETRY_VR
    ] = geometry_valid[
        ...,
        None
    ]

    valid[
        ...,
        ANNOTATED_VEL
    ] = annotated_valid[
        ...,
        None
    ]

    valid[
        ...,
        POS_COV
    ] = observed[
        ...,
        None
    ]

    valid[
        ...,
        RAW_R4
    ] = noisy_valid[
        ...,
        None
    ]

    valid[
        ...,
        LIDAR
    ] = lidar_valid[
        ...,
        None
    ]

    return valid


def compute_fit_normalization(
    cache_rows,
):
    header(
        "FIT-ONLY NORMALIZATION"
    )

    sum_x = np.zeros(
        FEATURE_DIM,
        dtype=np.float64,
    )

    sum_x2 = np.zeros(
        FEATURE_DIM,
        dtype=np.float64,
    )

    count_x = np.zeros(
        FEATURE_DIM,
        dtype=np.int64,
    )

    map_sum = np.zeros(
        MAP_DIM,
        dtype=np.float64,
    )

    map_sum2 = np.zeros(
        MAP_DIM,
        dtype=np.float64,
    )

    map_count = 0

    label_sum = np.zeros(
        3,
        dtype=np.float64,
    )

    label_sum2 = np.zeros(
        3,
        dtype=np.float64,
    )

    label_count = np.zeros(
        3,
        dtype=np.int64,
    )

    for rank, row in enumerate(
        cache_rows,
        start=1,
    ):
        with np.load(
            row[
                "path"
            ],
            allow_pickle=False,
        ) as z:
            for key in (
                "target",
                "neighbor_pool",
            ):
                values = np.asarray(
                    z[key],
                    dtype=np.float64,
                )

                if values.size == 0:
                    continue

                validity = (
                    feature_validity(
                        values
                    )
                )

                for feature in range(
                    FEATURE_DIM
                ):
                    if (
                        feature
                        in MASK_INDICES
                    ):
                        continue

                    x = (
                        values[
                            ...,
                            feature
                        ][
                            validity[
                                ...,
                                feature
                            ]
                        ]
                    )

                    if x.size:
                        sum_x[
                            feature
                        ] += x.sum()

                        sum_x2[
                            feature
                        ] += np.square(
                            x
                        ).sum()

                        count_x[
                            feature
                        ] += x.size

            map_value = np.asarray(
                z[
                    "map_context"
                ],
                dtype=np.float64,
            )

            if map_value.size:
                map_sum += (
                    map_value.sum(
                        axis=0
                    )
                )

                map_sum2 += (
                    np.square(
                        map_value
                    ).sum(
                        axis=0
                    )
                )

                map_count += (
                    map_value.shape[
                        0
                    ]
                )

            labels = np.asarray(
                z[
                    "future_displacement"
                ],
                dtype=np.float64,
            )

            label_mask = np.asarray(
                z[
                    "future_mask"
                ],
                dtype=bool,
            )

            for coordinate in range(
                3
            ):
                x = (
                    labels[
                        ...,
                        coordinate
                    ][
                        label_mask
                    ]
                )

                if x.size:
                    label_sum[
                        coordinate
                    ] += x.sum()

                    label_sum2[
                        coordinate
                    ] += np.square(
                        x
                    ).sum()

                    label_count[
                        coordinate
                    ] += x.size

        if rank % 500 == 0:
            print(
                "normalization:",
                rank,
                "/",
                len(
                    cache_rows
                ),
                flush=True,
            )

    feature_mean = np.zeros(
        FEATURE_DIM,
        dtype=np.float64,
    )

    feature_std = np.ones(
        FEATURE_DIM,
        dtype=np.float64,
    )

    for feature in range(
        FEATURE_DIM
    ):
        if feature in MASK_INDICES:
            continue

        if count_x[
            feature
        ]:
            feature_mean[
                feature
            ] = (
                sum_x[
                    feature
                ]
                /
                count_x[
                    feature
                ]
            )

            variance = max(
                (
                    sum_x2[
                        feature
                    ]
                    /
                    count_x[
                        feature
                    ]
                )
                -
                feature_mean[
                    feature
                ] ** 2,
                1.0e-8,
            )

            feature_std[
                feature
            ] = math.sqrt(
                variance
            )

    map_mean = (
        map_sum
        / max(
            map_count,
            1,
        )
    )

    map_variance = np.maximum(
        (
            map_sum2
            / max(
                map_count,
                1,
            )
        )
        - map_mean ** 2,
        1.0e-8,
    )

    map_std = np.sqrt(
        map_variance
    )

    label_mean = (
        label_sum
        /
        np.maximum(
            label_count,
            1,
        )
    )

    label_variance = np.maximum(
        (
            label_sum2
            /
            np.maximum(
                label_count,
                1,
            )
        )
        - label_mean ** 2,
        1.0e-8,
    )

    label_std = np.sqrt(
        label_variance
    )

    result = {
        "source":
            "fit_only",

        "feature_names":
            FEATURE_NAMES,

        "feature_mean":
            feature_mean.tolist(),

        "feature_std":
            feature_std.tolist(),

        "feature_count":
            count_x.tolist(),

        "structural_mask_indices":
            sorted(
                MASK_INDICES
            ),

        "map_mean":
            map_mean.tolist(),

        "map_std":
            map_std.tolist(),

        "map_count":
            map_count,

        "label_mean":
            label_mean.tolist(),

        "label_std":
            label_std.tolist(),

        "label_count":
            label_count.tolist(),
    }

    atomic_json(
        OUT
        / "fit_normalization_v2.json",
        result,
    )

    return result


def normalize_feature_array(
    raw,
    normalization,
):
    raw = np.asarray(
        raw,
        dtype=np.float32,
    )

    output = raw.copy()

    validity = (
        feature_validity(
            raw
        )
    )

    mean = np.asarray(
        normalization[
            "feature_mean"
        ],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization[
            "feature_std"
        ],
        dtype=np.float32,
    )

    for feature in range(
        FEATURE_DIM
    ):
        if feature in MASK_INDICES:
            continue

        output[
            ...,
            feature
        ] = (
            output[
                ...,
                feature
            ]
            - mean[
                feature
            ]
        ) / std[
            feature
        ]

        output[
            ...,
            feature
        ][
            ~validity[
                ...,
                feature
            ]
        ] = 0.0

    return output


# =====================================================================
# PARTITION DATA
# =====================================================================

class PartitionData:
    def __init__(
        self,
        cache_rows,
        normalization,
    ):
        arrays = defaultdict(
            list
        )

        for row in cache_rows:
            with np.load(
                row[
                    "path"
                ],
                allow_pickle=False,
            ) as z:
                n = (
                    z[
                        "class_id"
                    ].shape[
                        0
                    ]
                )

                if n == 0:
                    continue

                for key in (
                    "target",
                    "neighbor_pool",
                    "map_context",
                    "future_displacement",
                    "future_mask",
                    "class_id",
                ):
                    arrays[
                        key
                    ].append(
                        np.asarray(
                            z[key]
                        )
                    )

                arrays[
                    "lidar_subset"
                ].append(
                    np.full(
                        n,
                        bool(
                            z[
                                "lidar_subset_scene"
                            ].item()
                        ),
                        dtype=bool,
                    )
                )

        if not arrays[
            "class_id"
        ]:
            raise RuntimeError(
                "Partition contains "
                "no training samples."
            )

        self.target = (
            normalize_feature_array(
                np.concatenate(
                    arrays[
                        "target"
                    ],
                    axis=0,
                ),
                normalization,
            )
        )

        self.neighbor = (
            normalize_feature_array(
                np.concatenate(
                    arrays[
                        "neighbor_pool"
                    ],
                    axis=0,
                ),
                normalization,
            )
        )

        raw_map = np.concatenate(
            arrays[
                "map_context"
            ],
            axis=0,
        ).astype(
            np.float32
        )

        map_mean = np.asarray(
            normalization[
                "map_mean"
            ],
            dtype=np.float32,
        )

        map_std = np.asarray(
            normalization[
                "map_std"
            ],
            dtype=np.float32,
        )

        self.map = (
            raw_map
            - map_mean
        ) / map_std

        self.y_metric = (
            np.concatenate(
                arrays[
                    "future_displacement"
                ],
                axis=0,
            )
            .astype(
                np.float32
            )
        )

        label_mean = np.asarray(
            normalization[
                "label_mean"
            ],
            dtype=np.float32,
        )

        label_std = np.asarray(
            normalization[
                "label_std"
            ],
            dtype=np.float32,
        )

        self.y = (
            self.y_metric
            -
            label_mean[
                None,
                None,
                :
            ]
        ) / label_std[
            None,
            None,
            :
        ]

        self.mask = (
            np.concatenate(
                arrays[
                    "future_mask"
                ],
                axis=0,
            )
            .astype(
                np.float32
            )
        )

        self.class_id = (
            np.concatenate(
                arrays[
                    "class_id"
                ],
                axis=0,
            )
            .astype(
                np.int64
            )
        )

        self.lidar_subset = (
            np.concatenate(
                arrays[
                    "lidar_subset"
                ],
                axis=0,
            )
        )

    def indices(
        self,
        lidar_only=False,
    ):
        if lidar_only:
            return np.flatnonzero(
                self.lidar_subset
            )

        return np.arange(
            len(
                self.class_id
            ),
            dtype=np.int64,
        )


# =====================================================================
# MODELS
# =====================================================================

class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.target_gru = nn.GRU(
            FEATURE_DIM,
            64,
            batch_first=True,
        )

        self.neighbor_gru = nn.GRU(
            FEATURE_DIM,
            32,
            batch_first=True,
        )

        self.map_encoder = nn.Sequential(
            nn.Linear(
                MAP_DIM,
                32,
            ),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                64
                + 32
                + 32,
                128,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        target,
        neighbor,
        map_context,
        feature_mask,
        *,
        use_neighbors,
        use_map,
    ):
        feature_mask = (
            feature_mask
            .view(
                1,
                1,
                -1,
            )
        )

        target = (
            target
            * feature_mask
        )

        neighbor = (
            neighbor
            * feature_mask
        )

        _, target_hidden = (
            self.target_gru(
                target
            )
        )

        _, neighbor_hidden = (
            self.neighbor_gru(
                neighbor
            )
        )

        target_hidden = (
            target_hidden[
                -1
            ]
        )

        neighbor_hidden = (
            neighbor_hidden[
                -1
            ]
        )

        if not use_neighbors:
            neighbor_hidden = (
                torch.zeros_like(
                    neighbor_hidden
                )
            )

        map_hidden = (
            self.map_encoder(
                map_context
            )
        )

        if not use_map:
            map_hidden = (
                torch.zeros_like(
                    map_hidden
                )
            )

        return self.fusion(
            torch.cat(
                [
                    target_hidden,
                    neighbor_hidden,
                    map_hidden,
                ],
                dim=-1,
            )
        )


class DeterministicV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = (
            ContextEncoder()
        )

        self.head = nn.Linear(
            128,
            12,
        )

    def forward(
        self,
        target,
        neighbor,
        map_context,
        feature_mask,
        *,
        use_neighbors,
        use_map,
    ):
        hidden = self.encoder(
            target,
            neighbor,
            map_context,
            feature_mask,

            use_neighbors=(
                use_neighbors
            ),

            use_map=(
                use_map
            ),
        )

        return (
            self.head(
                hidden
            )
            .reshape(
                -1,
                4,
                3,
            )
        )


def scale_tril_from_raw(
    raw,
):
    raw = raw.reshape(
        -1,
        4,
        6,
    )

    batch = raw.shape[
        0
    ]

    L = torch.zeros(
        (
            batch,
            4,
            3,
            3,
        ),
        dtype=raw.dtype,
        device=raw.device,
    )

    L[
        ...,
        0,
        0
    ] = (
        F.softplus(
            raw[
                ...,
                0
            ]
        )
        + 1.0e-3
    )

    L[
        ...,
        1,
        0
    ] = raw[
        ...,
        1
    ]

    L[
        ...,
        1,
        1
    ] = (
        F.softplus(
            raw[
                ...,
                2
            ]
        )
        + 1.0e-3
    )

    L[
        ...,
        2,
        0
    ] = raw[
        ...,
        3
    ]

    L[
        ...,
        2,
        1
    ] = raw[
        ...,
        4
    ]

    L[
        ...,
        2,
        2
    ] = (
        F.softplus(
            raw[
                ...,
                5
            ]
        )
        + 1.0e-3
    )

    return L


class GaussianV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = (
            ContextEncoder()
        )

        self.mean_head = nn.Linear(
            128,
            12,
        )

        self.scale_head = nn.Linear(
            128,
            24,
        )

    def forward(
        self,
        target,
        neighbor,
        map_context,
        feature_mask,
    ):
        hidden = self.encoder(
            target,
            neighbor,
            map_context,
            feature_mask,

            use_neighbors=True,
            use_map=True,
        )

        mean = (
            self.mean_head(
                hidden
            )
            .reshape(
                -1,
                4,
                3,
            )
        )

        L = scale_tril_from_raw(
            self.scale_head(
                hidden
            )
        )

        return (
            mean,
            L,
        )


class GMMV2(nn.Module):
    def __init__(
        self,
        components=3,
    ):
        super().__init__()

        self.components = (
            int(
                components
            )
        )

        self.encoder = (
            ContextEncoder()
        )

        self.mean_head = nn.Linear(
            128,
            self.components
            * 12,
        )

        self.scale_head = nn.Linear(
            128,
            self.components
            * 24,
        )

        self.logit_head = nn.Linear(
            128,
            self.components,
        )

    def forward(
        self,
        target,
        neighbor,
        map_context,
        feature_mask,
    ):
        hidden = self.encoder(
            target,
            neighbor,
            map_context,
            feature_mask,

            use_neighbors=True,
            use_map=True,
        )

        batch = hidden.shape[
            0
        ]

        means = (
            self.mean_head(
                hidden
            )
            .reshape(
                batch,
                self.components,
                4,
                3,
            )
        )

        raw_scale = (
            self.scale_head(
                hidden
            )
            .reshape(
                batch
                * self.components,
                24,
            )
        )

        L = (
            scale_tril_from_raw(
                raw_scale
            )
            .reshape(
                batch,
                self.components,
                4,
                3,
                3,
            )
        )

        logits = self.logit_head(
            hidden
        )

        return (
            means,
            L,
            logits,
        )


# =====================================================================
# LOSSES / NORMALIZATION
# =====================================================================

def gaussian_nll(
    mean,
    L,
    target,
):
    difference = (
        target
        - mean
    )

    solved = (
        torch.linalg
        .solve_triangular(
            L,
            difference.unsqueeze(
                -1
            ),
            upper=False,
        )
        .squeeze(
            -1
        )
    )

    d2 = torch.sum(
        solved ** 2,
        dim=-1,
    )

    diagonal = (
        torch.diagonal(
            L,
            dim1=-2,
            dim2=-1,
        )
    )

    logdet = (
        2.0
        * torch.sum(
            torch.log(
                diagonal
            ),
            dim=-1,
        )
    )

    return (
        0.5
        * (
            d2
            + logdet
            + 3.0
            * math.log(
                2.0
                * math.pi
            )
        )
    )


def masked_mean(
    values,
    mask,
):
    mask = mask.to(
        values.dtype
    )

    return (
        (
            values
            * mask
        ).sum()
        /
        torch.clamp(
            mask.sum(),
            min=1.0,
        )
    )


def deterministic_loss(
    prediction,
    target,
    mask,
):
    return masked_mean(
        torch.sum(
            (
                prediction
                - target
            ) ** 2,
            dim=-1,
        ),
        mask,
    )


def gaussian_loss(
    mean,
    L,
    target,
    mask,
):
    return masked_mean(
        gaussian_nll(
            mean,
            L,
            target,
        ),
        mask,
    )


def gmm_loss(
    means,
    L,
    logits,
    target,
    mask,
):
    expanded_target = (
        target[
            :,
            None,
            :,
            :
        ]
        .expand_as(
            means
        )
    )

    component_nll = (
        gaussian_nll(
            means,
            L,
            expanded_target,
        )
    )

    component_nll = (
        component_nll
        * mask[
            :,
            None,
            :
        ]
    )

    trajectory_nll = (
        component_nll.sum(
            dim=-1
        )
    )

    log_probability = (
        torch.log_softmax(
            logits,
            dim=-1,
        )
        - trajectory_nll
    )

    loss = (
        -torch.logsumexp(
            log_probability,
            dim=-1,
        )
    )

    valid = (
        mask.sum(
            dim=-1
        )
        > 0
    )

    if not torch.any(
        valid
    ):
        return (
            loss.mean()
            * 0.0
        )

    return loss[
        valid
    ].mean()


def feature_mask(
    indices,
    device,
):
    mask = torch.zeros(
        FEATURE_DIM,
        dtype=torch.float32,
        device=device,
    )

    mask[
        torch.asarray(
            sorted(
                set(
                    indices
                )
            ),
            dtype=torch.long,
            device=device,
        )
    ] = 1.0

    return mask


def denormalize_mean(
    mean,
    normalization,
):
    label_mean = torch.asarray(
        normalization[
            "label_mean"
        ],
        dtype=mean.dtype,
        device=mean.device,
    )

    label_std = torch.asarray(
        normalization[
            "label_std"
        ],
        dtype=mean.dtype,
        device=mean.device,
    )

    return (
        mean
        * label_std[
            None,
            None,
            :
        ]
        + label_mean[
            None,
            None,
            :
        ]
    )


def denormalize_L(
    L,
    normalization,
):
    label_std = torch.asarray(
        normalization[
            "label_std"
        ],
        dtype=L.dtype,
        device=L.device,
    )

    matrix = torch.diag(
        label_std
    )[
        None,
        None,
        :,
        :
    ]

    return (
        matrix
        @ L
    )


# =====================================================================
# DATA LOADERS
# =====================================================================

def make_loader(
    data,
    indices,
    *,
    balanced,
    seed,
    batch_size,
):
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    dataset = TensorDataset(
        torch.from_numpy(
            data.target[
                indices
            ]
        ),

        torch.from_numpy(
            data.neighbor[
                indices
            ]
        ),

        torch.from_numpy(
            data.map[
                indices
            ]
        ),

        torch.from_numpy(
            data.y[
                indices
            ]
        ),

        torch.from_numpy(
            data.mask[
                indices
            ]
        ),

        torch.from_numpy(
            data.class_id[
                indices
            ]
        ),

        torch.from_numpy(
            data.y_metric[
                indices
            ]
        ),
    )

    if not balanced:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    classes = (
        data.class_id[
            indices
        ]
    )

    counts = np.bincount(
        classes,
        minlength=3,
    )

    if np.any(
        counts == 0
    ):
        raise RuntimeError(
            "Class-balanced sampling "
            f"impossible: {counts.tolist()}"
        )

    weights = np.asarray(
        [
            1.0
            / counts[
                actor_class
            ]
            for actor_class
            in classes
        ],
        dtype=np.float64,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )

    sampler = (
        WeightedRandomSampler(
            torch.from_numpy(
                weights
            ),
            num_samples=len(
                weights
            ),
            replacement=True,
            generator=generator,
        )
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )


# =====================================================================
# EVALUATION — DEVELOPMENT/CALIBRATION ONLY IN REPAIR
# =====================================================================

@torch.no_grad()
def evaluate_deterministic(
    model,
    data,
    indices,
    variant,
    normalization,
    device,
):
    spec = VARIANTS[
        variant
    ]

    loader = make_loader(
        data,
        indices,
        balanced=False,
        seed=0,
        batch_size=1024,
    )

    fmask = feature_mask(
        spec[
            "features"
        ],
        device,
    )

    total_error = 0.0
    total_count = 0

    model.eval()

    for (
        target,
        neighbor,
        map_context,
        y,
        mask,
        actor_class,
        y_metric,
    ) in loader:
        target = target.to(
            device
        )

        neighbor = neighbor.to(
            device
        )

        map_context = (
            map_context.to(
                device
            )
        )

        mask = mask.to(
            device
        )

        prediction = model(
            target,
            neighbor,
            map_context,
            fmask,

            use_neighbors=(
                spec[
                    "neighbors"
                ]
            ),

            use_map=(
                spec[
                    "map"
                ]
            ),
        )

        prediction = (
            denormalize_mean(
                prediction,
                normalization,
            )
        )

        truth = y_metric.to(
            device
        )

        error = (
            torch.linalg
            .vector_norm(
                prediction[
                    ...,
                    :2
                ]
                -
                truth[
                    ...,
                    :2
                ],
                dim=-1,
            )
        )

        total_error += float(
            (
                error
                * mask
            )
            .sum()
            .cpu()
        )

        total_count += int(
            mask.sum()
            .cpu()
        )

    return {
        "ADE_m":
            (
                total_error
                /
                max(
                    total_count,
                    1,
                )
            ),

        "events":
            total_count,
    }


@torch.no_grad()
def evaluate_gaussian(
    model,
    data,
    indices,
    normalization,
    device,
    covariance_scale=None,
):
    loader = make_loader(
        data,
        indices,
        balanced=False,
        seed=0,
        batch_size=1024,
    )

    fmask = feature_mask(
        PRIMARY_FEATURES,
        device,
    )

    ade_sum = 0.0
    nll_sum = 0.0
    event_count = 0

    d2_rows = []
    mask_rows = []

    model.eval()

    for (
        target,
        neighbor,
        map_context,
        y,
        mask,
        actor_class,
        y_metric,
    ) in loader:
        target = target.to(
            device
        )

        neighbor = neighbor.to(
            device
        )

        map_context = (
            map_context.to(
                device
            )
        )

        mask = mask.to(
            device
        )

        mean, L = model(
            target,
            neighbor,
            map_context,
            fmask,
        )

        mean = (
            denormalize_mean(
                mean,
                normalization,
            )
        )

        L = denormalize_L(
            L,
            normalization,
        )

        if covariance_scale is not None:
            scale = torch.asarray(
                covariance_scale,
                dtype=L.dtype,
                device=L.device,
            )[
                None,
                :,
                None,
                None,
            ]

            L = (
                L
                * torch.sqrt(
                    scale
                )
            )

        truth = y_metric.to(
            device
        )

        planar_error = (
            torch.linalg
            .vector_norm(
                mean[
                    ...,
                    :2
                ]
                -
                truth[
                    ...,
                    :2
                ],
                dim=-1,
            )
        )

        nll = gaussian_nll(
            mean,
            L,
            truth,
        )

        ade_sum += float(
            (
                planar_error
                * mask
            )
            .sum()
            .cpu()
        )

        nll_sum += float(
            (
                nll
                * mask
            )
            .sum()
            .cpu()
        )

        event_count += int(
            mask.sum()
            .cpu()
        )

        difference = (
            truth
            - mean
        )

        solved = (
            torch.linalg
            .solve_triangular(
                L,
                difference.unsqueeze(
                    -1
                ),
                upper=False,
            )
            .squeeze(
                -1
            )
        )

        d2 = torch.sum(
            solved ** 2,
            dim=-1,
        )

        d2_rows.append(
            d2.cpu()
            .numpy()
        )

        mask_rows.append(
            mask.cpu()
            .numpy()
            .astype(
                bool
            )
        )

    return {
        "ADE_m":
            (
                ade_sum
                /
                max(
                    event_count,
                    1,
                )
            ),

        "NLL":
            (
                nll_sum
                /
                max(
                    event_count,
                    1,
                )
            ),

        "events":
            event_count,

        "d2":
            np.concatenate(
                d2_rows,
                axis=0,
            ),

        "mask":
            np.concatenate(
                mask_rows,
                axis=0,
            ),
    }


# =====================================================================
# TRAIN DETERMINISTIC / ABLATIONS
# =====================================================================

def train_deterministic(
    variant,
    fit_data,
    dev_data,
    normalization,
    device,
):
    spec = VARIANTS[
        variant
    ]

    checkpoint = (
        MODELS
        / (
            "deterministic_"
            + variant
            + ".pt"
        )
    )

    metadata_path = (
        MODELS
        / (
            "deterministic_"
            + variant
            + ".json"
        )
    )

    if (
        checkpoint.is_file()
        and metadata_path.is_file()
    ):
        metadata = read_json(
            metadata_path
        )

        if (
            metadata.get(
                "training_freeze_sha256"
            )
            ==
            sha256_file(
                TRAIN_FREEZE
            )
        ):
            model = (
                DeterministicV2()
                .to(
                    device
                )
            )

            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            model.load_state_dict(
                state[
                    "state_dict"
                ]
            )

            print(
                "RESUME MODEL:",
                variant,
            )

            return (
                model,
                metadata,
            )

    fit_indices = (
        fit_data.indices(
            spec[
                "lidar_subset"
            ]
        )
    )

    dev_indices = (
        dev_data.indices(
            spec[
                "lidar_subset"
            ]
        )
    )

    seed_everything(
        TRAINING_FREEZE[
            "seed"
        ]
    )

    model = (
        DeterministicV2()
        .to(
            device
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-4,
        )
    )

    fmask = feature_mask(
        spec[
            "features"
        ],
        device,
    )

    max_epochs = (
        TRAINING_FREEZE[
            "epochs"
        ][
            "primary_deterministic_max"
        ]
        if variant == "map"
        else
        TRAINING_FREEZE[
            "epochs"
        ][
            "ablation_max"
        ]
    )

    best_score = None
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        loader = make_loader(
            fit_data,
            fit_indices,

            balanced=True,

            seed=(
                TRAINING_FREEZE[
                    "seed"
                ]
                + epoch
            ),

            batch_size=512,
        )

        model.train()

        loss_sum = 0.0
        batches = 0

        for (
            target,
            neighbor,
            map_context,
            y,
            mask,
            actor_class,
            y_metric,
        ) in loader:
            target = target.to(
                device
            )

            neighbor = neighbor.to(
                device
            )

            map_context = (
                map_context.to(
                    device
                )
            )

            y = y.to(
                device
            )

            mask = mask.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                target,
                neighbor,
                map_context,
                fmask,

                use_neighbors=(
                    spec[
                        "neighbors"
                    ]
                ),

                use_map=(
                    spec[
                        "map"
                    ]
                ),
            )

            loss = (
                deterministic_loss(
                    prediction,
                    y,
                    mask,
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            loss_sum += float(
                loss.detach()
                .cpu()
            )

            batches += 1

        metrics = (
            evaluate_deterministic(
                model,
                dev_data,
                dev_indices,
                variant,
                normalization,
                device,
            )
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                loss_sum
                / max(
                    batches,
                    1,
                ),

            **metrics,
        }

        history.append(
            row
        )

        print(
            "DET",
            variant,
            "epoch=",
            epoch,
            "dev_ADE=",
            round(
                metrics[
                    "ADE_m"
                ],
                6,
            ),
            flush=True,
        )

        score = metrics[
            "ADE_m"
        ]

        if (
            best_score is None
            or score
            < best_score
            - 1.0e-9
        ):
            best_score = score

            best_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in model
                .state_dict()
                .items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

        if (
            bad_epochs
            >=
            TRAINING_FREEZE[
                "epochs"
            ][
                "early_stopping_patience"
            ]
        ):
            break

    model.load_state_dict(
        best_state
    )

    final_metrics = (
        evaluate_deterministic(
            model,
            dev_data,
            dev_indices,
            variant,
            normalization,
            device,
        )
    )

    state = {
        "model_type":
            "DeterministicV2",

        "variant":
            variant,

        "state_dict":
            best_state,

        "feature_names":
            FEATURE_NAMES,

        "feature_indices":
            spec[
                "features"
            ],

        "use_neighbors":
            spec[
                "neighbors"
            ],

        "use_map":
            spec[
                "map"
            ],

        "training_freeze_sha256":
            sha256_file(
                TRAIN_FREEZE
            ),
    }

    atomic_torch_save(
        checkpoint,
        state,
    )

    metadata = {
        "variant":
            variant,

        "checkpoint":
            str(
                checkpoint
            ),

        "sha256":
            sha256_file(
                checkpoint
            ),

        "training_freeze_sha256":
            sha256_file(
                TRAIN_FREEZE
            ),

        "fit_samples":
            len(
                fit_indices
            ),

        "development_samples":
            len(
                dev_indices
            ),

        "history":
            history,

        "development":
            final_metrics,
    }

    atomic_json(
        metadata_path,
        metadata,
    )

    return (
        model,
        metadata,
    )


# =====================================================================
# GAUSSIAN TRAINING
# =====================================================================

def initialize_gaussian(
    gaussian,
    deterministic,
):
    gaussian.encoder.load_state_dict(
        deterministic
        .encoder
        .state_dict()
    )

    gaussian.mean_head.load_state_dict(
        deterministic
        .head
        .state_dict()
    )

    with torch.no_grad():
        gaussian.scale_head.weight.zero_()
        gaussian.scale_head.bias.zero_()


def train_gaussian(
    deterministic,
    fit_data,
    dev_data,
    normalization,
    device,
):
    checkpoint = (
        MODELS
        / "gaussian_primary.pt"
    )

    metadata_path = (
        MODELS
        / "gaussian_primary.json"
    )

    if (
        checkpoint.is_file()
        and metadata_path.is_file()
    ):
        metadata = read_json(
            metadata_path
        )

        if (
            metadata.get(
                "training_freeze_sha256"
            )
            ==
            sha256_file(
                TRAIN_FREEZE
            )
        ):
            model = (
                GaussianV2()
                .to(
                    device
                )
            )

            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

            model.load_state_dict(
                state[
                    "state_dict"
                ]
            )

            print(
                "RESUME MODEL: Gaussian"
            )

            return (
                model,
                metadata,
            )

    seed_everything(
        TRAINING_FREEZE[
            "seed"
        ]
    )

    model = (
        GaussianV2()
        .to(
            device
        )
    )

    initialize_gaussian(
        model,
        deterministic,
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-4,
        )
    )

    fit_indices = (
        fit_data.indices()
    )

    dev_indices = (
        dev_data.indices()
    )

    fmask = feature_mask(
        PRIMARY_FEATURES,
        device,
    )

    best_score = None
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(
        1,
        TRAINING_FREEZE[
            "epochs"
        ][
            "Gaussian_max"
        ]
        + 1,
    ):
        loader = make_loader(
            fit_data,
            fit_indices,

            balanced=True,

            seed=(
                TRAINING_FREEZE[
                    "seed"
                ]
                + 1000
                + epoch
            ),

            batch_size=512,
        )

        model.train()

        loss_sum = 0.0
        batches = 0

        for (
            target,
            neighbor,
            map_context,
            y,
            mask,
            actor_class,
            y_metric,
        ) in loader:
            target = target.to(
                device
            )

            neighbor = neighbor.to(
                device
            )

            map_context = (
                map_context.to(
                    device
                )
            )

            y = y.to(
                device
            )

            mask = mask.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            mean, L = model(
                target,
                neighbor,
                map_context,
                fmask,
            )

            loss = gaussian_loss(
                mean,
                L,
                y,
                mask,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            loss_sum += float(
                loss.detach()
                .cpu()
            )

            batches += 1

        metrics = (
            evaluate_gaussian(
                model,
                dev_data,
                dev_indices,
                normalization,
                device,
            )
        )

        row = {
            "epoch":
                epoch,

            "train_NLL":
                loss_sum
                / max(
                    batches,
                    1,
                ),

            "development_ADE_m":
                metrics[
                    "ADE_m"
                ],

            "development_NLL":
                metrics[
                    "NLL"
                ],
        }

        history.append(
            row
        )

        print(
            "GAUSSIAN epoch=",
            epoch,
            "dev_NLL=",
            round(
                metrics[
                    "NLL"
                ],
                6,
            ),
            "dev_ADE=",
            round(
                metrics[
                    "ADE_m"
                ],
                6,
            ),
            flush=True,
        )

        score = metrics[
            "NLL"
        ]

        if (
            best_score is None
            or score
            < best_score
            - 1.0e-9
        ):
            best_score = score

            best_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in model
                .state_dict()
                .items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

        if (
            bad_epochs
            >=
            TRAINING_FREEZE[
                "epochs"
            ][
                "early_stopping_patience"
            ]
        ):
            break

    model.load_state_dict(
        best_state
    )

    metrics = evaluate_gaussian(
        model,
        dev_data,
        dev_indices,
        normalization,
        device,
    )

    metrics.pop(
        "d2"
    )

    metrics.pop(
        "mask"
    )

    atomic_torch_save(
        checkpoint,
        {
            "model_type":
                "GaussianV2",

            "state_dict":
                best_state,

            "feature_names":
                FEATURE_NAMES,

            "feature_indices":
                PRIMARY_FEATURES,

            "use_neighbors":
                True,

            "use_map":
                True,

            "full_3D_covariance":
                True,

            "measurement_R_is_input":
                True,

            "measurement_R_is_predictive_covariance":
                False,

            "training_freeze_sha256":
                sha256_file(
                    TRAIN_FREEZE
                ),
        },
    )

    metadata = {
        "checkpoint":
            str(
                checkpoint
            ),

        "sha256":
            sha256_file(
                checkpoint
            ),

        "training_freeze_sha256":
            sha256_file(
                TRAIN_FREEZE
            ),

        "history":
            history,

        "development":
            metrics,
    }

    atomic_json(
        metadata_path,
        metadata,
    )

    return (
        model,
        metadata,
    )


# =====================================================================
# GMM
# =====================================================================

def initialize_gmm(
    gmm,
    gaussian,
):
    gmm.encoder.load_state_dict(
        gaussian
        .encoder
        .state_dict()
    )

    with torch.no_grad():
        for component in range(
            3
        ):
            gmm.mean_head.weight[
                component
                * 12
                :
                (
                    component
                    + 1
                )
                * 12
            ].copy_(
                gaussian
                .mean_head
                .weight
            )

            gmm.mean_head.bias[
                component
                * 12
                :
                (
                    component
                    + 1
                )
                * 12
            ].copy_(
                gaussian
                .mean_head
                .bias
            )

            lateral_offset = (
                component
                - 1
            ) * 0.05

            gmm.mean_head.bias[
                component
                * 12
                + 1
                :
                (
                    component
                    + 1
                )
                * 12
                :
                3
            ] += lateral_offset

            gmm.scale_head.weight[
                component
                * 24
                :
                (
                    component
                    + 1
                )
                * 24
            ].copy_(
                gaussian
                .scale_head
                .weight
            )

            gmm.scale_head.bias[
                component
                * 24
                :
                (
                    component
                    + 1
                )
                * 24
            ].copy_(
                gaussian
                .scale_head
                .bias
            )

        gmm.logit_head.weight.zero_()
        gmm.logit_head.bias.zero_()


@torch.no_grad()
def evaluate_gmm(
    model,
    data,
    indices,
    normalization,
    device,
):
    loader = make_loader(
        data,
        indices,

        balanced=False,

        seed=0,

        batch_size=1024,
    )

    fmask = feature_mask(
        PRIMARY_FEATURES,
        device,
    )

    nll_sum = 0.0
    sample_count = 0

    ade_sum = 0.0
    event_count = 0

    model.eval()

    for (
        target,
        neighbor,
        map_context,
        y,
        mask,
        actor_class,
        y_metric,
    ) in loader:
        target = target.to(
            device
        )

        neighbor = neighbor.to(
            device
        )

        map_context = (
            map_context.to(
                device
            )
        )

        y = y.to(
            device
        )

        mask = mask.to(
            device
        )

        means, L, logits = model(
            target,
            neighbor,
            map_context,
            fmask,
        )

        expanded_y = (
            y[
                :,
                None,
                :,
                :
            ]
            .expand_as(
                means
            )
        )

        component_nll = (
            gaussian_nll(
                means,
                L,
                expanded_y,
            )
            * mask[
                :,
                None,
                :
            ]
        )

        trajectory_nll = (
            component_nll.sum(
                dim=-1
            )
        )

        sample_nll = (
            -torch.logsumexp(
                torch.log_softmax(
                    logits,
                    dim=-1,
                )
                - trajectory_nll,
                dim=-1,
            )
        )

        valid_sample = (
            mask.sum(
                dim=-1
            )
            > 0
        )

        nll_sum += float(
            sample_nll[
                valid_sample
            ]
            .sum()
            .cpu()
        )

        sample_count += int(
            valid_sample.sum()
            .cpu()
        )

        weights = (
            torch.softmax(
                logits,
                dim=-1,
            )[
                :,
                :,
                None,
                None,
            ]
        )

        expected_mean = (
            weights
            * means
        ).sum(
            dim=1
        )

        expected_mean = (
            denormalize_mean(
                expected_mean,
                normalization,
            )
        )

        truth = y_metric.to(
            device
        )

        error = (
            torch.linalg
            .vector_norm(
                expected_mean[
                    ...,
                    :2
                ]
                -
                truth[
                    ...,
                    :2
                ],
                dim=-1,
            )
        )

        ade_sum += float(
            (
                error
                * mask
            )
            .sum()
            .cpu()
        )

        event_count += int(
            mask.sum()
            .cpu()
        )

    return {
        "trajectory_NLL":
            (
                nll_sum
                /
                max(
                    sample_count,
                    1,
                )
            ),

        "ADE_m":
            (
                ade_sum
                /
                max(
                    event_count,
                    1,
                )
            ),

        "samples":
            sample_count,

        "events":
            event_count,
    }


def train_gmm(
    gaussian,
    fit_data,
    dev_data,
    normalization,
    device,
):
    checkpoint = (
        MODELS
        / "gmm_primary_3component.pt"
    )

    metadata_path = (
        MODELS
        / "gmm_primary_3component.json"
    )

    if (
        checkpoint.is_file()
        and metadata_path.is_file()
    ):
        metadata = read_json(
            metadata_path
        )

        model = (
            GMMV2(
                3
            )
            .to(
                device
            )
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            state[
                "state_dict"
            ]
        )

        return (
            model,
            metadata,
        )

    seed_everything(
        TRAINING_FREEZE[
            "seed"
        ]
    )

    model = (
        GMMV2(
            3
        )
        .to(
            device
        )
    )

    initialize_gmm(
        model,
        gaussian,
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-4,
        )
    )

    fit_indices = (
        fit_data.indices()
    )

    dev_indices = (
        dev_data.indices()
    )

    fmask = feature_mask(
        PRIMARY_FEATURES,
        device,
    )

    best_score = None
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(
        1,
        TRAINING_FREEZE[
            "epochs"
        ][
            "GMM_max"
        ]
        + 1,
    ):
        loader = make_loader(
            fit_data,
            fit_indices,

            balanced=True,

            seed=(
                TRAINING_FREEZE[
                    "seed"
                ]
                + 2000
                + epoch
            ),

            batch_size=512,
        )

        model.train()

        total = 0.0
        batches = 0

        for (
            target,
            neighbor,
            map_context,
            y,
            mask,
            actor_class,
            y_metric,
        ) in loader:
            target = target.to(
                device
            )

            neighbor = neighbor.to(
                device
            )

            map_context = (
                map_context.to(
                    device
                )
            )

            y = y.to(
                device
            )

            mask = mask.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                means,
                L,
                logits,
            ) = model(
                target,
                neighbor,
                map_context,
                fmask,
            )

            loss = gmm_loss(
                means,
                L,
                logits,
                y,
                mask,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total += float(
                loss.detach()
                .cpu()
            )

            batches += 1

        metrics = evaluate_gmm(
            model,
            dev_data,
            dev_indices,
            normalization,
            device,
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_NLL":
                    total
                    / max(
                        batches,
                        1,
                    ),

                **metrics,
            }
        )

        print(
            "GMM epoch=",
            epoch,
            "dev_NLL=",
            round(
                metrics[
                    "trajectory_NLL"
                ],
                6,
            ),
            "dev_ADE=",
            round(
                metrics[
                    "ADE_m"
                ],
                6,
            ),
            flush=True,
        )

        score = metrics[
            "trajectory_NLL"
        ]

        if (
            best_score is None
            or score
            < best_score
            - 1.0e-9
        ):
            best_score = score

            best_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in model
                .state_dict()
                .items()
            }

            bad_epochs = 0

        else:
            bad_epochs += 1

        if (
            bad_epochs
            >=
            TRAINING_FREEZE[
                "epochs"
            ][
                "early_stopping_patience"
            ]
        ):
            break

    model.load_state_dict(
        best_state
    )

    final_metrics = (
        evaluate_gmm(
            model,
            dev_data,
            dev_indices,
            normalization,
            device,
        )
    )

    atomic_torch_save(
        checkpoint,
        {
            "model_type":
                "GMMV2",

            "components":
                3,

            "trajectory_level_mixture":
                True,

            "state_dict":
                best_state,

            "feature_indices":
                PRIMARY_FEATURES,

            "training_freeze_sha256":
                sha256_file(
                    TRAIN_FREEZE
                ),
        },
    )

    metadata = {
        "checkpoint":
            str(
                checkpoint
            ),

        "sha256":
            sha256_file(
                checkpoint
            ),

        "training_freeze_sha256":
            sha256_file(
                TRAIN_FREEZE
            ),

        "history":
            history,

        "development":
            final_metrics,
    }

    atomic_json(
        metadata_path,
        metadata,
    )

    return (
        model,
        metadata,
    )


# =====================================================================
# HELD-OUT CALIBRATION
# =====================================================================

def coverage_metrics(
    d2,
    mask,
    scales,
):
    confidence_levels = (
        TRAINING_FREEZE[
            "calibration"
        ][
            "confidence_levels"
        ]
    )

    errors = []
    briers = []
    per_horizon = []

    for horizon_index in range(
        4
    ):
        values = (
            d2[
                :,
                horizon_index
            ][
                mask[
                    :,
                    horizon_index
                ]
            ]
            /
            float(
                scales[
                    horizon_index
                ]
            )
        )

        horizon_result = {
            "horizon_s":
                HORIZONS[
                    horizon_index
                ],

            "n":
                int(
                    values.size
                ),

            "coverage":
                {},
        }

        for confidence in (
            confidence_levels
        ):
            threshold = float(
                chi2.ppf(
                    confidence,
                    df=3,
                )
            )

            hit = (
                values
                <= threshold
            ).astype(
                np.float64
            )

            coverage = float(
                hit.mean()
            )

            horizon_result[
                "coverage"
            ][
                str(
                    confidence
                )
            ] = coverage

            errors.append(
                abs(
                    coverage
                    - confidence
                )
            )

            briers.append(
                float(
                    np.mean(
                        (
                            hit
                            - confidence
                        ) ** 2
                    )
                )
            )

        per_horizon.append(
            horizon_result
        )

    return {
        "macro_ECE":
            float(
                np.mean(
                    errors
                )
            ),

        "macro_Brier":
            float(
                np.mean(
                    briers
                )
            ),

        "per_horizon":
            per_horizon,
    }


def fit_calibrator(
    gaussian,
    calibration_data,
    normalization,
    device,
):
    calibrator_path = (
        OUT
        / "covariance_calibrator_v2.json"
    )

    if calibrator_path.is_file():
        return read_json(
            calibrator_path
        )

    indices = (
        calibration_data.indices()
    )

    raw = evaluate_gaussian(
        gaussian,
        calibration_data,
        indices,
        normalization,
        device,
    )

    d2 = raw[
        "d2"
    ]

    mask = raw[
        "mask"
    ]

    confidence_levels = (
        TRAINING_FREEZE[
            "calibration"
        ][
            "confidence_levels"
        ]
    )

    scales = []

    for horizon_index in range(
        4
    ):
        values = (
            d2[
                :,
                horizon_index
            ][
                mask[
                    :,
                    horizon_index
                ]
            ]
        )

        if values.size < 20:
            raise RuntimeError(
                "Too few calibration "
                "events at horizon "
                f"{HORIZONS[horizon_index]}"
            )

        def objective(
            log_scale,
        ):
            scale = math.exp(
                float(
                    log_scale
                )
            )

            errors = []

            for confidence in (
                confidence_levels
            ):
                threshold = float(
                    chi2.ppf(
                        confidence,
                        df=3,
                    )
                )

                empirical = float(
                    np.mean(
                        values
                        / scale
                        <= threshold
                    )
                )

                errors.append(
                    abs(
                        empirical
                        - confidence
                    )
                )

            return float(
                np.mean(
                    errors
                )
            )

        candidate = (
            minimize_scalar(
                objective,
                bounds=(
                    math.log(
                        0.05
                    ),
                    math.log(
                        20.0
                    ),
                ),
                method="bounded",
            )
        )

        # Scale=1 is always an admissible candidate.
        # Calibration can therefore never be selected
        # merely because the optimizer found a worse solution.
        if (
            objective(
                0.0
            )
            <=
            objective(
                candidate.x
            )
        ):
            scale = 1.0

        else:
            scale = math.exp(
                float(
                    candidate.x
                )
            )

        scales.append(
            float(
                scale
            )
        )

    before = coverage_metrics(
        d2,
        mask,
        [
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    after = coverage_metrics(
        d2,
        mask,
        scales,
    )

    calibrated = (
        evaluate_gaussian(
            gaussian,
            calibration_data,
            indices,
            normalization,
            device,

            covariance_scale=(
                scales
            ),
        )
    )

    result = {
        "source":
            "calibration_partition_only",

        "method":
            "per_horizon_scalar_"
            "predictive_covariance_scaling",

        "variance_scale":
            scales,

        "measurement_R_modified":
            False,

        "predictive_mean_modified":
            False,

        "raw": {
            "ADE_m":
                raw[
                    "ADE_m"
                ],

            "NLL":
                raw[
                    "NLL"
                ],

            **before,
        },

        "calibrated": {
            "ADE_m":
                calibrated[
                    "ADE_m"
                ],

            "NLL":
                calibrated[
                    "NLL"
                ],

            **after,
        },
    }

    atomic_json(
        calibrator_path,
        result,
    )

    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    start_time = time.time()

    AUDIT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    formal_before = (
        formal_seal()
    )

    header(
        "STAGE 4 FULL-PDF V2 "
        "REPAIR 1/2"
    )

    print(
        "python =",
        sys.executable,
    )

    print(
        "torch  =",
        torch.__version__,
    )

    print(
        "CUDA   =",
        torch.cuda.is_available(),
    )

    print(
        "free GiB =",
        free_gib(),
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    if free_gib() < 250.0:
        raise RuntimeError(
            "Storage reserve failed."
        )

    # ---------------------------------------------------------
    # Preconditions
    # ---------------------------------------------------------

    header(
        "A. FROZEN PRECONDITIONS"
    )

    manifest_checks = (
        (
            "fit",
            FIT_MANIFEST,
        ),
        (
            "development",
            DEV_MANIFEST,
        ),
        (
            "calibration",
            CAL_MANIFEST,
        ),
    )

    for name, path in (
        manifest_checks
    ):
        actual = sha256_file(
            path
        )

        print(
            name,
            actual,
        )

        if (
            actual
            != EXPECTED_SHA[
                name
            ]
        ):
            raise RuntimeError(
                f"{name} manifest "
                "SHA changed."
            )

    normalization_source = (
        S4
        / "src"
        / "iscai_stage4"
        / "training"
        / "normalization.py"
    )

    normalization_sha = (
        sha256_file(
            normalization_source
        )
    )

    print(
        "normalization SHA =",
        normalization_sha,
    )

    if (
        normalization_sha
        !=
        EXPECTED_SHA[
            "normalization_post_fix"
        ]
    ):
        raise RuntimeError(
            "Unexpected normalization.py "
            "post-fix SHA."
        )

    __import__(
        "iscai_stage4.training.normalization"
    )

    protocol = read_json(
        PROTOCOL
    )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN_BEFORE_V2_TRAINING_"
        "OR_V2_EVALUATION"
    ):
        raise RuntimeError(
            "Full-PDF V2 protocol "
            "is not frozen."
        )

    if (
        protocol.get(
            "required_input_ablations"
        )
        !=
        REQUIRED_PDF_ABLATIONS
    ):
        raise RuntimeError(
            "Frozen PDF ablation "
            "list changed."
        )

    training_freeze_sha = (
        freeze_json(
            TRAIN_FREEZE,
            TRAINING_FREEZE,
        )
    )

    print(
        "training freeze SHA =",
        training_freeze_sha,
    )

    # Preserve old evidence, never overwrite it.
    historical = (
        AUDIT
        / "historical_before_fullpdf_v2"
    )

    historical.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        LEGACY_FORMAL,
        LEGACY_FORMAL_NPZ,
        LEGACY_CLOSURE,
    ):
        if path.is_file():
            destination = (
                historical
                / path.name
            )

            if not destination.exists():
                shutil.copy2(
                    path,
                    destination,
                )

    atomic_json(
        STATUS_REPORT,
        {
            "stage":
                4,

            "status":
                "FULLPDF_V2_"
                "TRAINING_IN_PROGRESS",

            "formal_evaluation_run":
                False,

            "downstream_stage5_allowed":
                False,
        },
    )

    # ---------------------------------------------------------
    # Frozen Stage2 configs
    # ---------------------------------------------------------

    from iscai_stage4.data.real_pipeline import (
        load_frozen_stage2_configs,
    )

    (
        clean_config,
        degraded_config,
    ) = resolve_stage2_configs(
        load_frozen_stage2_configs
    )

    fit_records = read_jsonl(
        FIT_MANIFEST,
        4096,
    )

    dev_records = read_jsonl(
        DEV_MANIFEST,
        512,
    )

    calibration_records = (
        read_jsonl(
            CAL_MANIFEST,
            512,
        )
    )

    if (
        len(
            fit_records
        )
        != 4096
        or len(
            dev_records
        )
        != 512
        or len(
            calibration_records
        )
        != 512
    ):
        raise RuntimeError(
            "Frozen first-N partition "
            "selection incomplete."
        )

    # ---------------------------------------------------------
    # Build V2 causal caches
    # ---------------------------------------------------------

    fit_cache = build_partition(
        fit_records,
        "fit",

        lidar_scenes=1000,

        clean_config=(
            clean_config
        ),

        degraded_config=(
            degraded_config
        ),
    )

    dev_cache = build_partition(
        dev_records,
        "development",

        lidar_scenes=256,

        clean_config=(
            clean_config
        ),

        degraded_config=(
            degraded_config
        ),
    )

    calibration_cache = (
        build_partition(
            calibration_records,
            "calibration",

            lidar_scenes=0,

            clean_config=(
                clean_config
            ),

            degraded_config=(
                degraded_config
            ),
        )
    )

    # ---------------------------------------------------------
    # Fit-only normalization
    # ---------------------------------------------------------

    normalization = (
        compute_fit_normalization(
            fit_cache
        )
    )

    header(
        "B. LOAD V2 PARTITIONS"
    )

    fit_data = PartitionData(
        fit_cache,
        normalization,
    )

    dev_data = PartitionData(
        dev_cache,
        normalization,
    )

    calibration_data = (
        PartitionData(
            calibration_cache,
            normalization,
        )
    )

    print(
        "fit/dev/cal samples =",
        len(
            fit_data.class_id
        ),
        len(
            dev_data.class_id
        ),
        len(
            calibration_data.class_id
        ),
    )

    fit_class_counts = (
        np.bincount(
            fit_data.class_id,
            minlength=3,
        )
    )

    print(
        "fit classes =",
        fit_class_counts.tolist(),
    )

    if np.any(
        fit_class_counts == 0
    ):
        raise RuntimeError(
            "All three classes "
            "are required."
        )

    device = torch.device(
        "cuda"
    )

    # ---------------------------------------------------------
    # Deterministic + all PDF ablations
    # ---------------------------------------------------------

    header(
        "C. PDF INPUT ABLATIONS"
    )

    ablation_results = {}

    deterministic_primary = None

    for variant in VARIANTS:
        (
            model,
            result,
        ) = train_deterministic(
            variant,
            fit_data,
            dev_data,
            normalization,
            device,
        )

        ablation_results[
            variant
        ] = result

        if variant == "map":
            deterministic_primary = (
                model
            )

        else:
            del model

        torch.cuda.empty_cache()

    if deterministic_primary is None:
        raise RuntimeError(
            "Primary deterministic "
            "model unavailable."
        )

    # ---------------------------------------------------------
    # Gaussian primary
    # ---------------------------------------------------------

    header(
        "D. FULL-COVARIANCE "
        "GAUSSIAN GRU"
    )

    (
        gaussian,
        gaussian_result,
    ) = train_gaussian(
        deterministic_primary,
        fit_data,
        dev_data,
        normalization,
        device,
    )

    # ---------------------------------------------------------
    # GMM
    # ---------------------------------------------------------

    header(
        "E. TRAJECTORY-LEVEL GMM"
    )

    (
        gmm,
        gmm_result,
    ) = train_gmm(
        gaussian,
        fit_data,
        dev_data,
        normalization,
        device,
    )

    del gmm
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------

    header(
        "F. HELD-OUT CALIBRATION"
    )

    calibrator = fit_calibrator(
        gaussian,
        calibration_data,
        normalization,
        device,
    )

    print(
        "variance scale =",
        calibrator[
            "variance_scale"
        ],
    )

    print(
        "calibration ECE =",
        calibrator[
            "raw"
        ][
            "macro_ECE"
        ],
        "->",
        calibrator[
            "calibrated"
        ][
            "macro_ECE"
        ],
    )

    # ---------------------------------------------------------
    # Pairwise development evidence
    # ---------------------------------------------------------

    pairwise = {}

    for name, (
        left,
        right,
    ) in ABLATION_PAIRS.items():
        left_ade = (
            ablation_results[
                left
            ][
                "development"
            ][
                "ADE_m"
            ]
        )

        right_ade = (
            ablation_results[
                right
            ][
                "development"
            ][
                "ADE_m"
            ]
        )

        pairwise[
            name
        ] = {
            "left":
                left,

            "right":
                right,

            "left_ADE_m":
                left_ade,

            "right_ADE_m":
                right_ade,

            "right_minus_left_m":
                (
                    right_ade
                    - left_ade
                ),

            "support":
                (
                    "same_frozen_"
                    "LiDAR_development_subset"
                    if (
                        VARIANTS[
                            left
                        ][
                            "lidar_subset"
                        ]
                        or
                        VARIANTS[
                            right
                        ][
                            "lidar_subset"
                        ]
                    )
                    else
                    "same_frozen_"
                    "development_support"
                ),
        }

    ablation_report = {
        "stage":
            4,

        "status":
            "MATERIALIZED_PRE_FORMAL",

        "all_required_PDF_"
        "ablations_materialized":
            True,

        "required_PDF_ablations":
            REQUIRED_PDF_ABLATIONS,

        "variants":
            ablation_results,

        "pairwise":
            pairwise,

        "semantics": {
            "annotated_velocity":
                "historical WOMD annotation "
                "upper-bound ablation only",

            "PC_FMCW_noisy_vr":
                "actual Stage2 degraded "
                "measurement radial_velocity_mps",

            "geometry_vr":
                "causal radial projection "
                "of estimated position/velocity",

            "LiDAR":
                "causal annotation-assisted "
                "actor-box ablation; intensity "
                "is relative proxy only",

            "LiDAR_primary_sensor_to_track":
                False,
        },
    }

    ablation_path = (
        REPORTS
        / "stage4_fullpdf_v2_"
        "ablation_development.json"
    )

    atomic_json(
        ablation_path,
        ablation_report,
    )

    # ---------------------------------------------------------
    # Freeze everything BEFORE formal
    # ---------------------------------------------------------

    model_freeze = {
        "stage":
            4,

        "status":
            "FROZEN_BEFORE_"
            "FORMAL_EVALUATION",

        "primary_model":
            gaussian_result[
                "checkpoint"
            ],

        "primary_model_sha256":
            gaussian_result[
                "sha256"
            ],

        "deterministic_primary":
            ablation_results[
                "map"
            ][
                "checkpoint"
            ],

        "deterministic_primary_sha256":
            ablation_results[
                "map"
            ][
                "sha256"
            ],

        "GMM":
            gmm_result[
                "checkpoint"
            ],

        "GMM_sha256":
            gmm_result[
                "sha256"
            ],

        "calibrator":
            str(
                OUT
                / "covariance_calibrator_v2.json"
            ),

        "calibrator_sha256":
            sha256_file(
                OUT
                / "covariance_calibrator_v2.json"
            ),

        "normalization":
            str(
                OUT
                / "fit_normalization_v2.json"
            ),

        "normalization_sha256":
            sha256_file(
                OUT
                / "fit_normalization_v2.json"
            ),

        "feature_names":
            FEATURE_NAMES,

        "primary_feature_indices":
            PRIMARY_FEATURES,

        "measurement_R_input":
            True,

        "measurement_R_separate_from_"
        "predictive_covariance":
            True,

        "direct_actual_Stage2_noisy_vr":
            True,

        "formal_outcomes_used":
            False,

        "formal_evaluation_run":
            False,

        "Stage5_allowed":
            False,

        "training_freeze_sha256":
            training_freeze_sha,
    }

    model_freeze_path = (
        OUT
        / "preformal_model_freeze.json"
    )

    atomic_json(
        model_freeze_path,
        model_freeze,
    )

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------

    header(
        "G. FULL STAGE4 REGRESSION"
    )

    env = os.environ.copy()

    env[
        "PYTHONPATH"
    ] = os.pathsep.join(
        [
            str(
                S0
                / "src"
            ),
            str(
                S1
                / "src"
            ),
            str(
                S2
                / "src"
            ),
            str(
                S3
                / "src"
            ),
            str(
                S4
                / "src"
            ),
            str(
                S4
                / "tests"
            ),
            env.get(
                "PYTHONPATH",
                "",
            ),
        ]
    )

    regression = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=S4,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(
        regression.stdout
    )

    if regression.returncode != 0:
        raise RuntimeError(
            "Stage4 regression failed."
        )

    # ---------------------------------------------------------
    # Verify formal artifacts untouched
    # ---------------------------------------------------------

    formal_after = formal_seal()

    if formal_after != formal_before:
        raise RuntimeError(
            "Legacy formal evidence "
            "changed during V2 repair."
        )

    # ---------------------------------------------------------
    # Final PRE-FORMAL repair report
    # ---------------------------------------------------------

    repair_report = {
        "stage":
            4,

        "repair":
            "fullpdf_v2_"
            "retraining_repair_1_of_2",

        "status":
            "TRAINED_CALIBRATED_"
            "AND_FROZEN_PRE_FORMAL",

        "PDF_full_compliance":
            False,

        "closure_blocker":
            "independent causal formal "
            "checker still required",

        "formal_validation_used":
            False,

        "formal_evaluation_run":
            False,

        "training_freeze": {
            "path":
                str(
                    TRAIN_FREEZE
                ),

            "sha256":
                training_freeze_sha,
        },

        "feature_schema": {
            "dimension":
                FEATURE_DIM,

            "names":
                FEATURE_NAMES,

            "primary_indices":
                PRIMARY_FEATURES,
        },

        "partitions": {
            "fit_scenarios":
                4096,

            "development_scenarios":
                512,

            "calibration_scenarios":
                512,

            "LiDAR_fit_scenarios":
                1000,

            "LiDAR_development_scenarios":
                256,
        },

        "models": {
            "deterministic":
                ablation_results[
                    "map"
                ],

            "Gaussian":
                gaussian_result,

            "GMM":
                gmm_result,
        },

        "calibration":
            calibrator,

        "ablations": {
            "report":
                str(
                    ablation_path
                ),

            "sha256":
                sha256_file(
                    ablation_path
                ),

            "all_PDF_required_"
            "materialized":
                True,
        },

        "formal_seal_unchanged":
            True,

        "formal_seal":
            formal_before,

        "regression_return_code":
            regression.returncode,

        "elapsed_hours":
            (
                time.time()
                - start_time
            )
            / 3600.0,

        "free_GiB":
            free_gib(),

        "next":
            "Stage4 Full-PDF V2 "
            "independent checker 2/2",
    }

    atomic_json(
        REPAIR_REPORT,
        repair_report,
    )

    atomic_json(
        STATUS_REPORT,
        {
            "stage":
                4,

            "status":
                "FULLPDF_V2_"
                "PRE_FORMAL_FROZEN",

            "PDF_full_compliance":
                False,

            "downstream_stage5_allowed":
                False,

            "repair_report":
                str(
                    REPAIR_REPORT
                ),
        },
    )

    header(
        "STAGE4 FULL-PDF V2 "
        "REPAIR 1/2 = PASS PRE-FORMAL"
    )

    print(
        "repair report =",
        REPAIR_REPORT,
    )

    print(
        "model freeze  =",
        model_freeze_path,
    )

    print(
        "all PDF input ablations = MATERIALIZED"
    )

    print(
        "formal evaluation = NO"
    )

    print(
        "Stage5 = BLOCKED UNTIL CHECKER 2/2 PASS"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        AUDIT.mkdir(
            parents=True,
            exist_ok=True,
        )

        atomic_json(
            FAIL_REPORT,
            {
                "stage":
                    4,

                "repair":
                    "fullpdf_v2_"
                    "retraining_repair_1_of_2",

                "status":
                    "FAILED_CLOSED",

                "exception":
                    repr(
                        exc
                    ),

                "traceback":
                    traceback.format_exc(),

                "formal_evaluation_run":
                    False,

                "downstream_stage5_allowed":
                    False,
            },
        )

        try:
            atomic_json(
                STATUS_REPORT,
                {
                    "stage":
                        4,

                    "status":
                        "FULLPDF_V2_"
                        "REPAIR_FAILED_CLOSED",

                    "PDF_full_compliance":
                        False,

                    "downstream_stage5_allowed":
                        False,

                    "failure_report":
                        str(
                            FAIL_REPORT
                        ),
                },
            )

        except Exception:
            pass

        print(
            "\nSTAGE4 FULL-PDF V2 "
            "REPAIR = FAILED_CLOSED",
            file=sys.stderr,
        )

        print(
            "failure report =",
            FAIL_REPORT,
            file=sys.stderr,
        )

        raise

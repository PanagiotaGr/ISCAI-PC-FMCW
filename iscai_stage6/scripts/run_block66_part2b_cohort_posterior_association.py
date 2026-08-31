from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(
    "/home/agni/waymo"
)

S4 = ROOT / "iscai_stage4"
S6 = ROOT / "iscai_stage6"


# ============================================================
# Frozen input artifacts
# ============================================================

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

PART2A_REPORT = (
    S6
    / "reports/"
      "block66_part2a_exact_runtime_route_resolution.json"
)

GAUSSIAN_CHECKPOINT = (
    S4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

ARCHITECTURE_CHECKPOINT = (
    S4
    / "artifacts/block43/"
      "deterministic_gru.pt"
)

NORMALIZATION = (
    S4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CALIBRATOR = (
    S4
    / "artifacts/block45/"
      "covariance_scaler.json"
)


# ============================================================
# Resumable / final outputs
# ============================================================

SCENE_DIR = (
    S6
    / "artifacts/block66/"
      "part2b_scene_cache"
)

POSTERIOR_JSONL = (
    S6
    / "artifacts/block66/"
      "block66_development_identity_safe_gaussian_posterior.jsonl"
)

MATCH_JSONL = (
    S6
    / "artifacts/block66/"
      "block66_development_association_matches.jsonl"
)

FALLBACK_JSONL = (
    S6
    / "artifacts/block66/"
      "block66_development_reactive_fallback_boxes.jsonl"
)

SCENE_SUMMARY_JSONL = (
    S6
    / "artifacts/block66/"
      "block66_development_scene_association_summary.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block66_part2b_cohort_posterior_association_census.json"
)


EXPECTED_SHA = {
    "cohort":
        (
            "57440e3d976e7aeff44811b5eeb075dc"
            "56e0e0e070a3832ec72d354253b998c9"
        ),

    "part2a":
        (
            "96154071016f10e05f95f47039773160"
            "1e0b9a3f4f856f80b48fcaf1fabaa5bc"
        ),

    "gaussian_checkpoint":
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        ),

    "architecture_checkpoint":
        (
            "5456a76b84d558e9983a59b9f1d3060b"
            "a245e0d36d60883519654f809996dbc5"
        ),

    "normalization":
        (
            "3d7fc0a66d4a4f566f6569befa9c3766"
            "ecae21830bb4e46256df2f326a82a5f6"
        ),

    "calibrator":
        (
            "508ff2e3fbcfafe8e001155340c25baaf"
            "3772fe2561a8022a9ed1cf780e66087"
        ),

    "association_module":
        (
            "5a4cc1803f9e45a8af23a57d89f47d9d"
            "f6cc06af57823b2ce00063afa0a110a8"
        ),

    "womd_geometry_module":
        (
            "b90f47289bd091474658e3864c4b1ccc"
            "8c2fc841aaa38160f38f0b95ece7187c"
        ),
}


EXPECTED_ALPHA = np.asarray(
    [
        1.2347064500315355,
        1.3451250295202921,
        1.354822057728461,
        1.2829700217319266,
    ],
    dtype=np.float64,
)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

CORE_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

MIN_FREE_GIB = 250.0


# ============================================================
# Generic safe helpers
# ============================================================

def require(
    condition: bool,
    message: str,
) -> None:

    if not bool(condition):
        raise RuntimeError(
            message
        )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:
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


def canonical_json_bytes(
    payload: Any,
) -> bytes:

    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_json_write(
    path: Path,
    payload: Any,
) -> str:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = canonical_json_bytes(
        payload
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
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


def atomic_jsonl_write(
    path: Path,
    rows,
) -> str:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                    allow_nan=False,
                )
            )

            handle.write(
                "\n"
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
) -> list[dict]:

    return [
        json.loads(
            line
        )
        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def free_gib() -> float:

    return (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )


def finite3(
    values,
) -> tuple[
    float,
    float,
    float,
]:

    result = tuple(
        float(x)
        for x in values
    )

    require(
        len(result) == 3,
        (
            "Expected 3-vector, got "
            f"{result}"
        ),
    )

    require(
        all(
            math.isfinite(x)
            for x in result
        ),
        (
            "Non-finite 3-vector: "
            f"{result}"
        ),
    )

    return result


def finite_optional(
    value,
):

    if value is None:
        return None

    result = float(
        value
    )

    require(
        math.isfinite(
            result
        ),
        (
            "Expected finite optional "
            f"value, got {result}"
        ),
    )

    return result


def class_counter():
    return {
        name: 0
        for name
        in CORE_CLASSES
    }


def increment_class(
    counter: dict[str, int],
    name: str,
    amount: int = 1,
) -> None:

    require(
        name in CORE_CLASSES,
        (
            "Unexpected ADB class: "
            f"{name}"
        ),
    )

    counter[
        name
    ] += int(
        amount
    )


def safe_scene_filename(
    index: int,
    scenario_id: str,
) -> Path:

    clean = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        scenario_id,
    )

    return (
        SCENE_DIR
        /
        (
            f"{index:03d}_"
            f"{clean}.json"
        )
    )


# ============================================================
# Find CausalSceneInputs without inspecting future WOMD data
# ============================================================

def find_instances(
    root: Any,
    cls: type,
) -> list[Any]:

    found = []
    seen = set()

    def visit(
        value,
        depth=0,
    ):
        if depth > 8:
            return

        object_id = id(
            value
        )

        if object_id in seen:
            return

        seen.add(
            object_id
        )

        if isinstance(
            value,
            cls,
        ):
            found.append(
                value
            )
            return

        if isinstance(
            value,
            dict,
        ):
            for child in (
                value.values()
            ):
                visit(
                    child,
                    depth + 1,
                )
            return

        if isinstance(
            value,
            (
                tuple,
                list,
            ),
        ):
            for child in value:
                visit(
                    child,
                    depth + 1,
                )
            return

        module_name = (
            value.__class__
            .__module__
        )

        if (
            module_name.startswith(
                "iscai_"
            )
            and
            hasattr(
                value,
                "__dict__",
            )
        ):
            for child in (
                vars(
                    value
                ).values()
            ):
                visit(
                    child,
                    depth + 1,
                )

    visit(
        root
    )

    unique = []

    unique_ids = set()

    for item in found:
        if id(item) in unique_ids:
            continue

        unique_ids.add(
            id(item)
        )

        unique.append(
            item
        )

    return unique


# ============================================================
# Exact frozen Stage4 prepare() controller-input mapping
# ============================================================

def build_prepare_arrays(
    payloads: tuple[Any, ...],
) -> dict[
    str,
    np.ndarray,
]:

    count = len(
        payloads
    )

    require(
        count > 0,
        "No causal model payloads.",
    )

    target = np.asarray(
        [
            payload.target_history
            for payload
            in payloads
        ],
        dtype=np.float32,
    )

    neighbors = np.asarray(
        [
            payload.neighbor_histories
            for payload
            in payloads
        ],
        dtype=np.float32,
    )

    neighbor_mask = np.asarray(
        [
            payload.neighbor_mask
            for payload
            in payloads
        ],
        dtype=np.float32,
    )

    map_context = np.asarray(
        [
            payload.map_context
            for payload
            in payloads
        ],
        dtype=np.float32,
    )

    require(
        target.shape[0]
        ==
        count,
        "Target batch mismatch.",
    )

    require(
        neighbors.shape[0]
        ==
        count,
        "Neighbor batch mismatch.",
    )

    require(
        neighbor_mask.shape[0]
        ==
        count,
        "Neighbor-mask batch mismatch.",
    )

    require(
        map_context.shape
        ==
        (
            count,
            10,
        ),
        (
            "Map context must be [N,10], "
            f"got {map_context.shape}."
        ),
    )

    for name, array in (
        (
            "target",
            target,
        ),
        (
            "neighbors",
            neighbors,
        ),
        (
            "neighbor_mask",
            neighbor_mask,
        ),
        (
            "map_context",
            map_context,
        ),
    ):
        require(
            bool(
                np.all(
                    np.isfinite(
                        array
                    )
                )
            ),
            (
                f"{name} contains "
                "non-finite values."
            ),
        )

    # These three fields are required only by the frozen
    # normalizer.prepare API. They are not model.forward inputs.
    # They contain no future GT and no truth class.
    future_placeholder = np.zeros(
        (
            count,
            4,
            3,
        ),
        dtype=np.float32,
    )

    future_mask_placeholder = np.zeros(
        (
            count,
            4,
        ),
        dtype=np.float32,
    )

    class_id_placeholder = np.zeros(
        (
            count,
        ),
        dtype=np.int64,
    )

    arrays = {
        "target":
            target,

        "neighbors":
            neighbors,

        "neighbor_mask":
            neighbor_mask,

        "map_context":
            map_context,

        "future":
            future_placeholder,

        "future_mask":
            future_mask_placeholder,

        "class_id":
            class_id_placeholder,
    }

    require(
        set(
            arrays
        )
        ==
        {
            "target",
            "neighbors",
            "neighbor_mask",
            "map_context",
            "future",
            "future_mask",
            "class_id",
        },
        (
            "Frozen prepare-array "
            "contract changed."
        ),
    )

    return arrays


# ============================================================
# Numeric identity-safe posterior helpers
# ============================================================

def verify_covariances(
    raw_covariance: np.ndarray,
    calibrated_covariance: np.ndarray,
) -> None:

    require(
        raw_covariance.ndim == 4,
        "Raw covariance rank changed.",
    )

    require(
        calibrated_covariance.shape
        ==
        raw_covariance.shape,
        (
            "Calibrated covariance "
            "shape mismatch."
        ),
    )

    require(
        raw_covariance.shape[
            1:
        ]
        ==
        (
            4,
            3,
            3,
        ),
        (
            "Covariance must be "
            "[N,4,3,3]."
        ),
    )

    for array_name, array in (
        (
            "raw",
            raw_covariance,
        ),
        (
            "calibrated",
            calibrated_covariance,
        ),
    ):

        require(
            bool(
                np.all(
                    np.isfinite(
                        array
                    )
                )
            ),
            (
                f"{array_name} covariance "
                "contains non-finite values."
            ),
        )

        for matrix in (
            array.reshape(
                -1,
                3,
                3,
            )
        ):
            require(
                np.array_equal(
                    matrix,
                    matrix.T,
                )
                or
                np.allclose(
                    matrix,
                    matrix.T,
                    rtol=0.0,
                    atol=1.0e-12,
                ),
                (
                    f"{array_name} covariance "
                    "lost symmetry."
                ),
            )

            # No scientific threshold:
            # Cholesky itself is the SPD gate.
            np.linalg.cholesky(
                matrix
            )


def posterior_numeric_digest(
    records: list[dict],
) -> str:

    digest = hashlib.sha256()

    for record in records:

        digest.update(
            str(
                record[
                    "scenario_id"
                ]
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                record[
                    "prediction_id"
                ]
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        for key in (
            "latest_position_H0_m",
            "mean_displacement_H0_m",
            "raw_predictive_covariance_H0_m2",
            "calibrated_predictive_covariance_H0_m2",
        ):
            array = np.asarray(
                record[
                    key
                ],
                dtype=np.float32,
            )

            digest.update(
                array.tobytes(
                    order="C"
                )
            )

    return digest.hexdigest()


# ============================================================
# Main
# ============================================================

def main():

    overall_start = (
        time.perf_counter()
    )

    print(
        "=" * 76
    )

    print(
        "BLOCK 6.6 PART2B — RESUMABLE DEVELOPMENT "
        "POSTERIOR + ASSOCIATION CENSUS"
    )

    print(
        "=" * 76
    )

    # --------------------------------------------------------
    # A. Immutable seals
    # --------------------------------------------------------

    print()
    print(
        "===== A. IMMUTABLE FROZEN SEAL ====="
    )

    association_module = (
        S6
        / "src/iscai_stage6/adb/"
          "deterministic_association.py"
    )

    womd_geometry_module = (
        S6
        / "src/iscai_stage6/adb/"
          "womd_geometry.py"
    )

    sealed = {
        "cohort":
            COHORT,

        "part2a":
            PART2A_REPORT,

        "gaussian_checkpoint":
            GAUSSIAN_CHECKPOINT,

        "architecture_checkpoint":
            ARCHITECTURE_CHECKPOINT,

        "normalization":
            NORMALIZATION,

        "calibrator":
            CALIBRATOR,

        "association_module":
            association_module,

        "womd_geometry_module":
            womd_geometry_module,
    }

    seal_report = {}

    for name, path in (
        sealed.items()
    ):

        require(
            path.is_file(),
            (
                "Missing frozen "
                f"artifact: {path}"
            ),
        )

        actual = sha256_file(
            path
        )

        expected = (
            EXPECTED_SHA[
                name
            ]
        )

        require(
            actual
            ==
            expected,
            (
                f"Frozen SHA changed for "
                f"{name}: {actual}"
            ),
        )

        seal_report[
            name
        ] = {
            "path":
                str(
                    path
                ),

            "sha256":
                actual,
        }

        print(
            f"{name:28s} = EXACT PASS"
        )

    require(
        free_gib()
        >=
        MIN_FREE_GIB,
        (
            "250-GiB storage "
            "reserve violated."
        ),
    )

    print(
        "storage reserve             = PASS"
    )

    # --------------------------------------------------------
    # B. Cohort
    # --------------------------------------------------------

    print()
    print(
        "===== B. FROZEN DEVELOPMENT COHORT ====="
    )

    cohort_rows = (
        read_jsonl(
            COHORT
        )
    )

    require(
        len(
            cohort_rows
        )
        ==
        120,
        (
            "Frozen development cohort "
            "must contain 120 scenarios."
        ),
    )

    scenario_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row
        in cohort_rows
    ]

    require(
        len(
            set(
                scenario_ids
            )
        )
        ==
        120,
        (
            "Development cohort "
            "scenario IDs are not unique."
        ),
    )

    print(
        "cohort scenarios            = 120"
    )

    print(
        "unique scenario IDs         = PASS"
    )

    print(
        "partition                   = DEVELOPMENT"
    )

    print(
        "formal outcomes             = NOT READ"
    )

    # --------------------------------------------------------
    # C. Exact imports
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXACT FROZEN RUNTIME IMPORTS ====="
    )

    import torch

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage4.data import (
        CausalSceneInputs,
        build_model_input_payload,
    )

    from iscai_stage4.data.real_pipeline import (
        build_real_causal_inputs,
        load_frozen_stage2_configs,
        read_training_scenario,
    )

    from iscai_stage4.data.supervision import (
        build_static_map_index_H0,
        map_context_summary,
    )

    from iscai_stage4.ml.calibration_runtime import (
        CalibrationNormalizer,
        build_gaussian_model,
        prediction_sha256,
    )

    from iscai_stage4.ml.gaussian_math import (
        covariance_from_scale_tril,
        denormalize_gaussian,
    )

    from iscai_stage5.angular_monte_carlo import (
        calibrate_stage4_horizon_covariances,
    )

    from iscai_stage6.adb.deterministic_association import (
        PredictorAnchor,
        reciprocal_unique_nearest_anchor_association,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    print(
        "CalibrationNormalizer       = calibration_runtime"
    )

    print(
        "Gaussian model builder      = calibration_runtime"
    )

    print(
        "Gaussian metric math        = gaussian_math"
    )

    print(
        "covariance calibration      = angular_monte_carlo"
    )

    print(
        "association                 = reciprocal unique nearest anchor"
    )

    print(
        "association threshold       = NONE"
    )

    print(
        "association class gate      = NONE"
    )

    # --------------------------------------------------------
    # D. Frozen model / normalizer
    # --------------------------------------------------------

    print()
    print(
        "===== D. FROZEN GAUSSIAN RUNTIME ====="
    )

    require(
        torch.cuda.is_available(),
        (
            "CUDA is unavailable; "
            "do not silently change "
            "the frozen execution device."
        ),
    )

    device = torch.device(
        "cuda:0"
    )

    print(
        "device                      =",
        device,
    )

    print(
        "GPU                         =",
        torch.cuda.get_device_name(
            0
        ),
    )

    torch.manual_seed(
        0
    )

    torch.cuda.manual_seed_all(
        0
    )

    torch.use_deterministic_algorithms(
        True
    )

    normalization_json = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    normalization_payload = (
        normalization_json[
            "payload"
        ]
        if (
            isinstance(
                normalization_json,
                dict,
            )
            and
            isinstance(
                normalization_json.get(
                    "payload"
                ),
                dict,
            )
        )
        else
        normalization_json
    )

    required_normalization_keys = {
        "continuous_feature_mean",
        "continuous_feature_std",
        "map_mean",
        "map_std",
        "label_displacement_mean",
        "label_displacement_std",
    }

    require(
        required_normalization_keys
        .issubset(
            normalization_payload
        ),
        (
            "Frozen fit-only normalization "
            "payload schema changed."
        ),
    )

    normalizer = CalibrationNormalizer(
        normalization_payload,
        device=device,
    )

    gaussian_checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    architecture_checkpoint = torch.load(
        ARCHITECTURE_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    require(
        isinstance(
            architecture_checkpoint.get(
                "configuration"
            ),
            dict,
        ),
        (
            "Architecture checkpoint "
            "configuration missing."
        ),
    )

    architecture_configuration = (
        architecture_checkpoint[
            "configuration"
        ]
    )

    require(
        "model"
        in
        architecture_configuration,
        (
            "Frozen deterministic architecture "
            "configuration lacks model section."
        ),
    )

    require(
        isinstance(
            gaussian_checkpoint.get(
                "state_dict"
            ),
            dict,
        ),
        (
            "Frozen Gaussian checkpoint "
            "state_dict missing."
        ),
    )

    model = build_gaussian_model(
        architecture_configuration,
        device=device,
    )

    load_result = model.load_state_dict(
        gaussian_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    require(
        not getattr(
            load_result,
            "missing_keys",
            (),
        ),
        "Gaussian model missing keys.",
    )

    require(
        not getattr(
            load_result,
            "unexpected_keys",
            (),
        ),
        "Gaussian model unexpected keys.",
    )

    model.eval()

    model_parameter_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
    )

    require(
        model_parameter_count
        ==
        48484,
        (
            "Frozen Gaussian parameter "
            f"count changed: {model_parameter_count}"
        ),
    )

    calibrator_json = json.loads(
        CALIBRATOR.read_text(
            encoding="utf-8"
        )
    )

    calibrator_text = json.dumps(
        calibrator_json,
        sort_keys=True,
    )

    for alpha in (
        EXPECTED_ALPHA
    ):
        require(
            str(
                float(
                    alpha
                )
            )
            in
            calibrator_text,
            (
                "Expected frozen calibration "
                f"alpha not found: {alpha}"
            ),
        )

    label_mean = torch.as_tensor(
        normalization_payload[
            "label_displacement_mean"
        ],
        dtype=torch.float32,
        device=device,
    )

    label_std = torch.as_tensor(
        normalization_payload[
            "label_displacement_std"
        ],
        dtype=torch.float32,
        device=device,
    )

    print(
        "Gaussian parameters         =",
        model_parameter_count,
    )

    print(
        "Gaussian predictive weights = Block4.4 FROZEN"
    )

    print(
        "architecture metadata       = Block4.3 FROZEN"
    )

    print(
        "normalization               = FIT-ONLY FROZEN"
    )

    print(
        "mean calibration            = UNCHANGED"
    )

    print(
        "covariance alpha            =",
        EXPECTED_ALPHA.tolist(),
    )

    # --------------------------------------------------------
    # E. Stage2 frozen causal configs
    # --------------------------------------------------------

    print()
    print(
        "===== E. FROZEN CAUSAL INPUT CONFIG ====="
    )

    (
        clean_config,
        degraded_config,
    ) = load_frozen_stage2_configs()

    print(
        "Stage2 configs              = LOADED FROZEN"
    )

    print(
        "future labels               = NOT REQUESTED"
    )

    print(
        "truth/perfect ID            = NOT USED"
    )

    print(
        "tracks_to_predict           = NOT USED"
    )

    # --------------------------------------------------------
    # F. Resumable scene execution
    # --------------------------------------------------------

    print()
    print(
        "===== F. RESUMABLE 120-SCENARIO EXECUTION ====="
    )

    SCENE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reused_count = 0
    executed_count = 0

    for cohort_index, row in enumerate(
        cohort_rows
    ):

        scenario_id = str(
            row[
                "scenario_id"
            ]
        )

        scene_path = (
            safe_scene_filename(
                cohort_index,
                scenario_id,
            )
        )

        if scene_path.is_file():

            existing = json.loads(
                scene_path.read_text(
                    encoding="utf-8"
                )
            )

            require(
                existing.get(
                    "status"
                )
                ==
                "PASS_SCENE_POSTERIOR_ASSOCIATION",
                (
                    "Existing resumable scene artifact "
                    "does not have PASS status: "
                    f"{scene_path}"
                ),
            )

            require(
                int(
                    existing[
                        "cohort_index"
                    ]
                )
                ==
                cohort_index,
                (
                    "Cached cohort index mismatch."
                ),
            )

            require(
                str(
                    existing[
                        "scenario_id"
                    ]
                )
                ==
                scenario_id,
                (
                    "Cached scenario ID mismatch."
                ),
            )

            require(
                existing[
                    "frozen_seals"
                ][
                    "cohort_sha256"
                ]
                ==
                EXPECTED_SHA[
                    "cohort"
                ],
                (
                    "Cached cohort SHA mismatch."
                ),
            )

            require(
                existing[
                    "frozen_seals"
                ][
                    "part2a_route_sha256"
                ]
                ==
                EXPECTED_SHA[
                    "part2a"
                ],
                (
                    "Cached Part2A route SHA mismatch."
                ),
            )

            require(
                existing[
                    "frozen_seals"
                ][
                    "gaussian_checkpoint_sha256"
                ]
                ==
                EXPECTED_SHA[
                    "gaussian_checkpoint"
                ],
                (
                    "Cached Gaussian SHA mismatch."
                ),
            )

            reused_count += 1

            if (
                (cohort_index + 1)
                % 10
                ==
                0
                or
                cohort_index == 0
            ):
                print(
                    f"[{cohort_index+1:03d}/120] "
                    f"{scenario_id} = RESUME CACHE PASS"
                )

            continue

        scene_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # F1. Exact frozen random access
        # ----------------------------------------------------

        scenario = read_training_scenario(
            row
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            scenario_id,
            (
                "Random-access scenario ID "
                "does not match cohort."
            ),
        )

        require(
            int(
                scenario.current_time_index
            )
            ==
            10,
            (
                "Unexpected WOMD current "
                "time index."
            ),
        )

        # ----------------------------------------------------
        # F2. Current causal ADB boxes
        # ----------------------------------------------------

        adapted = (
            adapt_causal_womd_scenario(
                scenario
            )
        )

        actor_boxes = tuple(
            build_causal_adb_actor_boxes(
                scenario=scenario,
                adapted=adapted,
            )
        )

        require(
            len(
                actor_boxes
            )
            >
            0,
            (
                "Frozen class-covered cohort "
                "scenario has zero causal "
                "ADB core-class boxes."
            ),
        )

        box_counts_by_class = (
            class_counter()
        )

        for box in actor_boxes:

            object_type = str(
                box.object_type
            )

            increment_class(
                box_counts_by_class,
                object_type,
            )

            require(
                box.state_source
                ==
                "current_causal",
                (
                    "ADB box state source "
                    "is not current causal."
                ),
            )

            require(
                box.orientation_source
                ==
                "current_causal",
                (
                    "ADB box orientation source "
                    "is not current causal."
                ),
            )

            require(
                not bool(
                    box.future_state_used
                ),
                (
                    "Future state leaked into "
                    "current ADB box."
                ),
            )

            require(
                not bool(
                    box.tracks_to_predict_used
                ),
                (
                    "tracks_to_predict leaked "
                    "into ADB box."
                ),
            )

            require(
                not bool(
                    box.objects_of_interest_used
                ),
                (
                    "objects_of_interest leaked "
                    "into ADB box."
                ),
            )

        # ----------------------------------------------------
        # F3. Exact frozen causal predictor inputs
        # ----------------------------------------------------

        causal_root = build_real_causal_inputs(
            scenario,
            clean_config=clean_config,
            degraded_config=degraded_config,
        )

        scene_candidates = find_instances(
            causal_root,
            CausalSceneInputs,
        )

        require(
            len(
                scene_candidates
            )
            ==
            1,
            (
                "Expected exactly one "
                "CausalSceneInputs instance; "
                f"found {len(scene_candidates)} "
                f"for {scenario_id}."
            ),
        )

        scene_inputs = (
            scene_candidates[
                0
            ]
        )

        histories = tuple(
            scene_inputs.histories
        )

        prediction_ids = tuple(
            str(
                history.prediction_id
            )
            for history
            in histories
        )

        require(
            len(
                prediction_ids
            )
            ==
            len(
                set(
                    prediction_ids
                )
            ),
            (
                "Causal prediction IDs "
                "are not unique within scene."
            ),
        )

        # ----------------------------------------------------
        # F4. Identity-safe Gaussian posterior
        # ----------------------------------------------------

        posterior_records = []

        raw_prediction_sha = None

        if histories:

            map_index_H0 = (
                build_static_map_index_H0(
                    scenario,
                    T_H0_from_W=(
                        adapted.frames
                        .T_H0_from_W
                    ),
                )
            )

            require(
                isinstance(
                    map_index_H0,
                    dict,
                ),
                (
                    "Static map index "
                    "contract changed."
                ),
            )

            map_contexts = tuple(
                map_context_summary(
                    map_index_H0,
                    target_position_H0_m=(
                        history
                        .latest_position_H0_m
                    ),
                )
                for history
                in histories
            )

            require(
                all(
                    len(
                        value
                    )
                    ==
                    10
                    for value
                    in map_contexts
                ),
                (
                    "Map context dimension "
                    "changed."
                ),
            )

            model_payloads = tuple(
                build_model_input_payload(
                    scene_inputs,
                    prediction_id=prediction_id,
                    map_context=map_context,
                )
                for (
                    prediction_id,
                    map_context,
                )
                in zip(
                    prediction_ids,
                    map_contexts,
                    strict=True,
                )
            )

            require(
                len(
                    model_payloads
                )
                ==
                len(
                    histories
                ),
                (
                    "Model payload/history "
                    "count mismatch."
                ),
            )

            arrays = (
                build_prepare_arrays(
                    model_payloads
                )
            )

            batch = normalizer.prepare(
                arrays,
                device=device,
            )

            require(
                {
                    "target",
                    "neighbors",
                    "neighbor_mask",
                    "map_context",
                }
                .issubset(
                    batch
                ),
                (
                    "Frozen prepared batch "
                    "model-input keys changed."
                ),
            )

            model_kwargs = {
                "target":
                    batch[
                        "target"
                    ],

                "neighbors":
                    batch[
                        "neighbors"
                    ],

                "neighbor_mask":
                    batch[
                        "neighbor_mask"
                    ],

                "map_context":
                    batch[
                        "map_context"
                    ],
            }

            with torch.inference_mode():

                output = model(
                    batch["target"],
                    batch["neighbors"],
                    batch["neighbor_mask"],
                    batch["map_context"],
                )

            require(
                hasattr(
                    output,
                    "mean",
                ),
                "Gaussian output has no mean.",
            )

            require(
                hasattr(
                    output,
                    "scale_tril",
                ),
                (
                    "Gaussian output has "
                    "no scale_tril."
                ),
            )

            raw_prediction_sha = (
                prediction_sha256(
                    output.mean,
                    output.scale_tril,
                )
            )

            (
                mean_metric,
                scale_metric,
            ) = denormalize_gaussian(
                output.mean,
                output.scale_tril,
                label_mean,
                label_std,
            )

            require(
                tuple(
                    mean_metric.shape
                )
                ==
                (
                    len(
                        histories
                    ),
                    4,
                    3,
                ),
                (
                    "Metric Gaussian mean "
                    "shape mismatch."
                ),
            )

            require(
                tuple(
                    scale_metric.shape
                )
                ==
                (
                    len(
                        histories
                    ),
                    4,
                    3,
                    3,
                ),
                (
                    "Metric Gaussian scale "
                    "shape mismatch."
                ),
            )

            raw_covariance = (
                covariance_from_scale_tril(
                    scale_metric
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            mean_metric_np = (
                mean_metric
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            calibrated_covariance = np.stack(
                [
                    np.asarray(
                        calibrate_stage4_horizon_covariances(
                            raw_covariance[
                                index
                            ]
                        ),
                        dtype=np.float64,
                    )
                    for index
                    in range(
                        len(
                            histories
                        )
                    )
                ],
                axis=0,
            )

            verify_covariances(
                raw_covariance,
                calibrated_covariance,
            )

            require(
                bool(
                    np.all(
                        np.isfinite(
                            mean_metric_np
                        )
                    )
                ),
                (
                    "Gaussian metric mean "
                    "contains non-finite values."
                ),
            )

            for index, history in enumerate(
                histories
            ):

                latest_position = finite3(
                    history
                    .latest_position_H0_m
                )

                posterior_records.append(
                    {
                        "cohort_index":
                            cohort_index,

                        "scenario_id":
                            scenario_id,

                        "prediction_id":
                            prediction_ids[
                                index
                            ],

                        "latest_position_H0_m":
                            list(
                                latest_position
                            ),

                        "horizons_s":
                            list(
                                HORIZONS_S
                            ),

                        "mean_displacement_H0_m":
                            mean_metric_np[
                                index
                            ].tolist(),

                        "raw_predictive_covariance_H0_m2":
                            raw_covariance[
                                index
                            ].tolist(),

                        "calibrated_predictive_covariance_H0_m2":
                            calibrated_covariance[
                                index
                            ].tolist(),

                        "variance_scale_alpha_h":
                            EXPECTED_ALPHA.tolist(),

                        "mean_calibration":
                            "UNCHANGED",

                        "posterior_family":
                            "calibrated_Gaussian_GRU",

                        "future_GT_used":
                            False,

                        "truth_id_used":
                            False,

                        "tracks_to_predict_used":
                            False,

                        "objects_of_interest_used":
                            False,
                    }
                )

        # ----------------------------------------------------
        # F5. Frozen association — ALL causal core ADB boxes
        # ----------------------------------------------------

        predictor_anchors = tuple(
            PredictorAnchor(
                prediction_id=(
                    history.prediction_id
                ),
                latest_position_H0_m=(
                    history
                    .latest_position_H0_m
                ),
            )
            for history
            in histories
        )

        if predictor_anchors:

            association = (
                reciprocal_unique_nearest_anchor_association(
                    predictor_anchors,
                    actor_boxes,
                )
            )

            matches = tuple(
                association.matches
            )

            unmatched_predictor_indices = (
                tuple(
                    association
                    .unmatched_predictor_indices
                )
            )

            unmatched_box_indices = (
                tuple(
                    association
                    .unmatched_box_indices
                )
            )

            require(
                int(
                    association
                    .predictor_count
                )
                ==
                len(
                    predictor_anchors
                ),
                (
                    "Association predictor "
                    "count mismatch."
                ),
            )

            require(
                int(
                    association
                    .box_count
                )
                ==
                len(
                    actor_boxes
                ),
                (
                    "Association box "
                    "count mismatch."
                ),
            )

        else:

            # Exact semantic limit of the frozen policy:
            # with no causal predictor anchors there can be no
            # reciprocal match and every current causal ADB box
            # remains on original-reactive fallback.
            matches = tuple()

            unmatched_predictor_indices = (
                tuple()
            )

            unmatched_box_indices = tuple(
                range(
                    len(
                        actor_boxes
                    )
                )
            )

        require(
            len(
                matches
            )
            +
            len(
                unmatched_box_indices
            )
            ==
            len(
                actor_boxes
            ),
            (
                "Matched + reactive-fallback "
                "boxes do not partition ADB boxes."
            ),
        )

        require(
            len(
                matches
            )
            +
            len(
                unmatched_predictor_indices
            )
            ==
            len(
                histories
            ),
            (
                "Matched + unmatched "
                "predictors do not partition "
                "causal predictor histories."
            ),
        )

        matched_box_indices = [
            int(
                item.box_index
            )
            for item
            in matches
        ]

        matched_predictor_indices = [
            int(
                item.predictor_index
            )
            for item
            in matches
        ]

        require(
            len(
                matched_box_indices
            )
            ==
            len(
                set(
                    matched_box_indices
                )
            ),
            (
                "Association reused an "
                "ADB box."
            ),
        )

        require(
            len(
                matched_predictor_indices
            )
            ==
            len(
                set(
                    matched_predictor_indices
                )
            ),
            (
                "Association reused a "
                "predictor."
            ),
        )

        match_records = []

        match_counts_by_class = (
            class_counter()
        )

        association_distances = []

        for match in matches:

            predictor_index = int(
                match.predictor_index
            )

            box_index = int(
                match.box_index
            )

            require(
                0
                <=
                predictor_index
                <
                len(
                    histories
                ),
                (
                    "Association predictor "
                    "index out of range."
                ),
            )

            require(
                0
                <=
                box_index
                <
                len(
                    actor_boxes
                ),
                (
                    "Association box "
                    "index out of range."
                ),
            )

            actor_box = (
                actor_boxes[
                    box_index
                ]
            )

            object_type = str(
                actor_box.object_type
            )

            increment_class(
                match_counts_by_class,
                object_type,
            )

            distance = float(
                match.anchor_distance_m
            )

            require(
                math.isfinite(
                    distance
                )
                and
                distance >= 0.0,
                (
                    "Association anchor "
                    "distance invalid."
                ),
            )

            association_distances.append(
                distance
            )

            require(
                str(
                    match.prediction_id
                )
                ==
                prediction_ids[
                    predictor_index
                ],
                (
                    "Association prediction_id "
                    "does not match causal "
                    "history index."
                ),
            )

            match_records.append(
                {
                    "cohort_index":
                        cohort_index,

                    "scenario_id":
                        scenario_id,

                    "prediction_id":
                        prediction_ids[
                            predictor_index
                        ],

                    "predictor_index":
                        predictor_index,

                    # Local causal box-array index only.
                    # No WOMD track identity is used to match.
                    "actor_box_index":
                        box_index,

                    "object_type":
                        object_type,

                    "association_anchor_distance_m":
                        distance,

                    "predictor_second_nearest_distance_m":
                        finite_optional(
                            match
                            .predictor_second_nearest_distance_m
                        ),

                    "box_second_nearest_distance_m":
                        finite_optional(
                            match
                            .box_second_nearest_distance_m
                        ),

                    "association_policy":
                        (
                            "reciprocal_unique_nearest_"
                            "current_anchor_H0"
                        ),

                    "distance_threshold_m":
                        None,

                    "class_gate":
                        False,

                    "perfect_ID_used":
                        False,

                    "truth_id_used":
                        False,

                    "future_GT_used":
                        False,
                }
            )

        fallback_records = []

        fallback_counts_by_class = (
            class_counter()
        )

        for box_index in (
            unmatched_box_indices
        ):

            box_index = int(
                box_index
            )

            actor_box = (
                actor_boxes[
                    box_index
                ]
            )

            object_type = str(
                actor_box.object_type
            )

            increment_class(
                fallback_counts_by_class,
                object_type,
            )

            fallback_records.append(
                {
                    "cohort_index":
                        cohort_index,

                    "scenario_id":
                        scenario_id,

                    "actor_box_index":
                        box_index,

                    "object_type":
                        object_type,

                    "fallback":
                        "ORIGINAL_REACTIVE_ADB",

                    "association_threshold_m":
                        None,

                    "class_gate":
                        False,

                    "perfect_ID_used":
                        False,

                    "future_GT_used":
                        False,
                }
            )

        for object_type in (
            CORE_CLASSES
        ):

            require(
                match_counts_by_class[
                    object_type
                ]
                +
                fallback_counts_by_class[
                    object_type
                ]
                ==
                box_counts_by_class[
                    object_type
                ],
                (
                    "Class-wise association "
                    "partition mismatch for "
                    f"{object_type}."
                ),
            )

        posterior_digest = (
            posterior_numeric_digest(
                posterior_records
            )
        )

        scene_runtime_s = (
            time.perf_counter()
            -
            scene_start
        )

        scene_payload = {
            "status":
                "PASS_SCENE_POSTERIOR_ASSOCIATION",

            "stage":
                6,

            "block":
                "6.6-Part2B",

            "cohort_index":
                cohort_index,

            "scenario_id":
                scenario_id,

            "cohort_metadata": {
                "stratum":
                    (
                        row.get(
                            "stratum"
                        )
                        or
                        row.get(
                            "cohort_stratum"
                        )
                    ),

                # Preserved only if already present in the
                # frozen cohort. Not recomputed or tuned here.
                "frozen_headlamp_relevant_counts":
                    row.get(
                        "headlamp_relevant_counts"
                    ),
            },

            "frozen_seals": {
                "cohort_sha256":
                    EXPECTED_SHA[
                        "cohort"
                    ],

                "part2a_route_sha256":
                    EXPECTED_SHA[
                        "part2a"
                    ],

                "gaussian_checkpoint_sha256":
                    EXPECTED_SHA[
                        "gaussian_checkpoint"
                    ],

                "architecture_checkpoint_sha256":
                    EXPECTED_SHA[
                        "architecture_checkpoint"
                    ],

                "normalization_sha256":
                    EXPECTED_SHA[
                        "normalization"
                    ],

                "calibrator_sha256":
                    EXPECTED_SHA[
                        "calibrator"
                    ],

                "association_module_sha256":
                    EXPECTED_SHA[
                        "association_module"
                    ],

                "womd_geometry_module_sha256":
                    EXPECTED_SHA[
                        "womd_geometry_module"
                    ],
            },

            "causal_predictor": {
                "history_count":
                    len(
                        histories
                    ),

                "prediction_id_unique":
                    True,

                "raw_prediction_sha256":
                    raw_prediction_sha,

                "posterior_numeric_digest_sha256":
                    posterior_digest,

                "posterior_record_count":
                    len(
                        posterior_records
                    ),

                "future_GT_used":
                    False,

                "truth_id_used":
                    False,

                "tracks_to_predict_used":
                    False,

                "objects_of_interest_used":
                    False,
            },

            "current_causal_ADB_boxes": {
                "count":
                    len(
                        actor_boxes
                    ),

                "by_class":
                    box_counts_by_class,

                "source":
                    (
                        "current_anchor_valid_"
                        "non_SDC_core_classes"
                    ),

                "future_state_used":
                    False,
            },

            "association": {
                "policy":
                    (
                        "reciprocal_unique_nearest_"
                        "current_anchor_H0"
                    ),

                "hard_distance_threshold":
                    None,

                "covariance_gate":
                    False,

                "class_gate":
                    False,

                "perfect_ID":
                    False,

                "truth_identity":
                    False,

                "match_count":
                    len(
                        matches
                    ),

                "unmatched_predictor_count":
                    len(
                        unmatched_predictor_indices
                    ),

                "reactive_fallback_box_count":
                    len(
                        unmatched_box_indices
                    ),

                "matched_by_class":
                    match_counts_by_class,

                "reactive_fallback_by_class":
                    fallback_counts_by_class,

                "anchor_distance_m": {
                    "minimum":
                        (
                            min(
                                association_distances
                            )
                            if association_distances
                            else None
                        ),

                    "mean":
                        (
                            float(
                                np.mean(
                                    association_distances
                                )
                            )
                            if association_distances
                            else None
                        ),

                    "maximum":
                        (
                            max(
                                association_distances
                            )
                            if association_distances
                            else None
                        ),
                },
            },

            "posterior_records":
                posterior_records,

            "association_matches":
                match_records,

            "reactive_fallback_boxes":
                fallback_records,

            "scientific_execution": {
                "frozen_model_forward":
                    bool(
                        histories
                    ),

                "training":
                    False,

                "recalibration":
                    False,

                "Monte_Carlo_sampling":
                    False,

                "P_occ_materialization":
                    False,

                "gamma_tuning":
                    False,

                "class_policy_freeze":
                    False,

                "formal_evaluation":
                    False,
            },

            "scope_boundary": {
                "association_runs_on_all_current_causal_core_ADB_boxes":
                    True,

                "new_headlamp_filter_inserted_into_association":
                    False,

                "headlamp_eligibility_rejoin_deferred":
                    True,

                "original_reactive_fallback_preserved":
                    True,
            },

            "runtime_s":
                scene_runtime_s,
        }

        atomic_json_write(
            scene_path,
            scene_payload,
        )

        executed_count += 1

        print(
            f"[{cohort_index+1:03d}/120] "
            f"{scenario_id} | "
            f"hist={len(histories):3d} | "
            f"boxes={len(actor_boxes):3d} | "
            f"match={len(matches):3d} | "
            f"fallback={len(unmatched_box_indices):3d} | "
            f"{scene_runtime_s:.2f}s"
        )

        del (
            scenario,
            adapted,
            actor_boxes,
            causal_root,
            scene_candidates,
            scene_inputs,
            histories,
            predictor_anchors,
            posterior_records,
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        require(
            free_gib()
            >=
            MIN_FREE_GIB,
            (
                "Storage reserve violated "
                "during resumable execution."
            ),
        )

    # --------------------------------------------------------
    # G. Full cache completeness
    # --------------------------------------------------------

    print()
    print(
        "===== G. CACHE COMPLETENESS ====="
    )

    scene_payloads = []

    for cohort_index, row in enumerate(
        cohort_rows
    ):

        scenario_id = str(
            row[
                "scenario_id"
            ]
        )

        path = safe_scene_filename(
            cohort_index,
            scenario_id,
        )

        require(
            path.is_file(),
            (
                "Missing resumable scene "
                f"artifact: {path}"
            ),
        )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        require(
            payload[
                "status"
            ]
            ==
            "PASS_SCENE_POSTERIOR_ASSOCIATION",
            (
                "Non-PASS scene artifact "
                f"at {path}"
            ),
        )

        require(
            payload[
                "scenario_id"
            ]
            ==
            scenario_id,
            (
                "Scene cache ordering "
                "mismatch."
            ),
        )

        scene_payloads.append(
            payload
        )

    require(
        len(
            scene_payloads
        )
        ==
        120,
        (
            "Expected 120 complete "
            "scene artifacts."
        ),
    )

    print(
        "complete scene artifacts    = 120/120 PASS"
    )

    print(
        "executed this run           =",
        executed_count,
    )

    print(
        "reused from cache           =",
        reused_count,
    )

    # --------------------------------------------------------
    # H. Deterministic consolidation
    # --------------------------------------------------------

    print()
    print(
        "===== H. DETERMINISTIC CONSOLIDATION ====="
    )

    all_posterior = []

    all_matches = []

    all_fallbacks = []

    scene_summaries = []

    total_histories = 0
    total_boxes = 0
    total_matches = 0
    total_fallback = 0
    total_unmatched_predictors = 0

    boxes_by_class = (
        class_counter()
    )

    matches_by_class = (
        class_counter()
    )

    fallback_by_class = (
        class_counter()
    )

    distances = []

    for payload in (
        scene_payloads
    ):

        all_posterior.extend(
            payload[
                "posterior_records"
            ]
        )

        all_matches.extend(
            payload[
                "association_matches"
            ]
        )

        all_fallbacks.extend(
            payload[
                "reactive_fallback_boxes"
            ]
        )

        causal = (
            payload[
                "causal_predictor"
            ]
        )

        adb = (
            payload[
                "current_causal_ADB_boxes"
            ]
        )

        association = (
            payload[
                "association"
            ]
        )

        total_histories += int(
            causal[
                "history_count"
            ]
        )

        total_boxes += int(
            adb[
                "count"
            ]
        )

        total_matches += int(
            association[
                "match_count"
            ]
        )

        total_fallback += int(
            association[
                "reactive_fallback_box_count"
            ]
        )

        total_unmatched_predictors += int(
            association[
                "unmatched_predictor_count"
            ]
        )

        for object_type in (
            CORE_CLASSES
        ):

            boxes_by_class[
                object_type
            ] += int(
                adb[
                    "by_class"
                ][
                    object_type
                ]
            )

            matches_by_class[
                object_type
            ] += int(
                association[
                    "matched_by_class"
                ][
                    object_type
                ]
            )

            fallback_by_class[
                object_type
            ] += int(
                association[
                    "reactive_fallback_by_class"
                ][
                    object_type
                ]
            )

        for match in (
            payload[
                "association_matches"
            ]
        ):
            distances.append(
                float(
                    match[
                        "association_anchor_distance_m"
                    ]
                )
            )

        scene_summaries.append(
            {
                "cohort_index":
                    payload[
                        "cohort_index"
                    ],

                "scenario_id":
                    payload[
                        "scenario_id"
                    ],

                "causal_history_count":
                    causal[
                        "history_count"
                    ],

                "ADB_box_count":
                    adb[
                        "count"
                    ],

                "ADB_boxes_by_class":
                    adb[
                        "by_class"
                    ],

                "match_count":
                    association[
                        "match_count"
                    ],

                "matched_by_class":
                    association[
                        "matched_by_class"
                    ],

                "reactive_fallback_box_count":
                    association[
                        "reactive_fallback_box_count"
                    ],

                "reactive_fallback_by_class":
                    association[
                        "reactive_fallback_by_class"
                    ],

                "unmatched_predictor_count":
                    association[
                        "unmatched_predictor_count"
                    ],
            }
        )

    require(
        len(
            all_posterior
        )
        ==
        total_histories,
        (
            "Consolidated posterior count "
            "does not equal causal "
            "history count."
        ),
    )

    require(
        len(
            all_matches
        )
        ==
        total_matches,
        (
            "Consolidated match count "
            "mismatch."
        ),
    )

    require(
        len(
            all_fallbacks
        )
        ==
        total_fallback,
        (
            "Consolidated fallback count "
            "mismatch."
        ),
    )

    require(
        total_matches
        +
        total_fallback
        ==
        total_boxes,
        (
            "Global matched/fallback "
            "partition mismatch."
        ),
    )

    require(
        total_matches
        +
        total_unmatched_predictors
        ==
        total_histories,
        (
            "Global matched/unmatched "
            "predictor partition mismatch."
        ),
    )

    for object_type in (
        CORE_CLASSES
    ):

        require(
            matches_by_class[
                object_type
            ]
            +
            fallback_by_class[
                object_type
            ]
            ==
            boxes_by_class[
                object_type
            ],
            (
                "Global class partition "
                f"mismatch for {object_type}."
            ),
        )

    posterior_sha = (
        atomic_jsonl_write(
            POSTERIOR_JSONL,
            all_posterior,
        )
    )

    match_sha = (
        atomic_jsonl_write(
            MATCH_JSONL,
            all_matches,
        )
    )

    fallback_sha = (
        atomic_jsonl_write(
            FALLBACK_JSONL,
            all_fallbacks,
        )
    )

    scene_summary_sha = (
        atomic_jsonl_write(
            SCENE_SUMMARY_JSONL,
            scene_summaries,
        )
    )

    global_posterior_digest = (
        posterior_numeric_digest(
            all_posterior
        )
    )

    print(
        "posterior records           =",
        len(
            all_posterior
        ),
    )

    print(
        "causal ADB boxes            =",
        total_boxes,
    )

    print(
        "association matches         =",
        total_matches,
    )

    print(
        "reactive fallback boxes     =",
        total_fallback,
    )

    print(
        "unmatched predictors        =",
        total_unmatched_predictors,
    )

    print(
        "boxes by class              =",
        boxes_by_class,
    )

    print(
        "matches by class            =",
        matches_by_class,
    )

    print(
        "fallback by class           =",
        fallback_by_class,
    )

    # This is only an association-support observation.
    # It is NOT yet the current-headlamp eligible predictive
    # support used for class-aware policy tuning.
    all_classes_have_association_support = all(
        matches_by_class[
            object_type
        ]
        >
        0
        for object_type
        in CORE_CLASSES
    )

    print(
        "all classes matched somewhere =",
        all_classes_have_association_support,
    )

    # --------------------------------------------------------
    # I. Fresh Stage6 regression
    # --------------------------------------------------------

    print()
    print(
        "===== I. FRESH STAGE6 REGRESSION ====="
    )

    env = dict(
        os.environ
    )

    regression = subprocess.run(
        [
            str(
                ROOT
                / "iscai_stage1/"
                  ".venv_lidar/bin/python"
            ),
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
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    regression_tail = (
        regression.stdout[
            -8000:
        ]
    )

    print(
        regression_tail
    )

    require(
        regression.returncode
        ==
        0,
        (
            "Fresh Stage6 regression "
            "failed."
        ),
    )

    passed_match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        regression.stdout,
    )

    regression_passed = (
        int(
            passed_match.group(
                1
            )
        )
        if passed_match
        else None
    )

    print(
        "Stage6 regression          = PASS"
    )

    if regression_passed is not None:
        print(
            "tests passed               =",
            regression_passed,
        )

    # --------------------------------------------------------
    # J. Final report
    # --------------------------------------------------------

    print()
    print(
        "===== J. BLOCK6.6 PART2B REPORT ====="
    )

    total_runtime_s = (
        time.perf_counter()
        -
        overall_start
    )

    association_distance_summary = {
        "count":
            len(
                distances
            ),

        "minimum_m":
            (
                min(
                    distances
                )
                if distances
                else None
            ),

        "mean_m":
            (
                float(
                    np.mean(
                        distances
                    )
                )
                if distances
                else None
            ),

        "maximum_m":
            (
                max(
                    distances
                )
                if distances
                else None
            ),
    }

    report = {
        "status":
            (
                "PASS_BLOCK66_PART2B_"
                "COHORT_POSTERIOR_ASSOCIATION_MATERIALIZED"
            ),

        "stage":
            6,

        "block":
            "6.6",

        "part":
            "2B",

        "purpose":
            (
                "Identity-safe calibrated Gaussian posterior "
                "materialization and frozen causal association "
                "census on the preregistered development cohort."
            ),

        "frozen_inputs":
            seal_report,

        "development_cohort": {
            "scenario_count":
                120,

            "partition":
                "development",

            "cohort_sha256":
                EXPECTED_SHA[
                    "cohort"
                ],

            "membership_changed":
                False,

            "selected_using_model_outcomes":
                False,

            "selected_using_future":
                False,

            "formal_outcomes_read":
                False,
        },

        "runtime": {
            "device":
                str(
                    device
                ),

            "GPU":
                torch.cuda.get_device_name(
                    0
                ),

            "model_parameter_count":
                model_parameter_count,

            "scene_cache_directory":
                str(
                    SCENE_DIR
                ),

            "executed_scenes_this_run":
                executed_count,

            "reused_scenes_this_run":
                reused_count,

            "total_runtime_s":
                total_runtime_s,
        },

        "posterior": {
            "family":
                "calibrated_Gaussian_GRU",

            "horizons_s":
                list(
                    HORIZONS_S
                ),

            "record_count":
                len(
                    all_posterior
                ),

            "mean_calibration":
                "UNCHANGED",

            "variance_scale_alpha_h":
                EXPECTED_ALPHA.tolist(),

            "measurement_R_t_modified_by_calibration":
                False,

            "identity_safe":
                True,

            "prediction_id_retained":
                True,

            "causal_anchor_retained":
                True,

            "anonymous_row_selection":
                False,

            "future_GT_used":
                False,

            "truth_id_used":
                False,

            "tracks_to_predict_used":
                False,

            "objects_of_interest_used":
                False,

            "numeric_digest_sha256":
                global_posterior_digest,

            "jsonl_path":
                str(
                    POSTERIOR_JSONL
                ),

            "jsonl_sha256":
                posterior_sha,
        },

        "association": {
            "policy":
                (
                    "reciprocal_unique_nearest_"
                    "current_anchor_H0"
                ),

            "runs_on":
                (
                    "all current causal anchor-valid "
                    "non-SDC vehicle/pedestrian/cyclist "
                    "ADB boxes"
                ),

            "hard_distance_threshold":
                None,

            "covariance_gate":
                False,

            "class_gate":
                False,

            "perfect_ID":
                False,

            "truth_identity":
                False,

            "predictor_count":
                total_histories,

            "ADB_box_count":
                total_boxes,

            "match_count":
                total_matches,

            "unmatched_predictor_count":
                total_unmatched_predictors,

            "reactive_fallback_box_count":
                total_fallback,

            "ADB_boxes_by_class":
                boxes_by_class,

            "matches_by_class":
                matches_by_class,

            "reactive_fallback_by_class":
                fallback_by_class,

            "anchor_distance":
                association_distance_summary,

            "all_core_classes_have_at_least_one_match":
                all_classes_have_association_support,

            "match_manifest_path":
                str(
                    MATCH_JSONL
                ),

            "match_manifest_sha256":
                match_sha,

            "fallback_manifest_path":
                str(
                    FALLBACK_JSONL
                ),

            "fallback_manifest_sha256":
                fallback_sha,
        },

        "scene_summary": {
            "path":
                str(
                    SCENE_SUMMARY_JSONL
                ),

            "sha256":
                scene_summary_sha,

            "record_count":
                120,
        },

        "scope_boundary": {
            "headlamp_cohort_membership":
                (
                    "already frozen by Block6.6 Part1A/Part1B"
                ),

            "new_headlamp_filter_inserted_into_frozen_association":
                False,

            "current_headlamp_actor_level_eligibility_join":
                "DEFERRED_TO_NEXT_GATE",

            "N_MC_8192_sampling":
                False,

            "P_occ_materialized":
                False,

            "gamma_reoptimized":
                False,

            "class_aware_gamma_frozen":
                False,

            "class_margins_frozen":
                False,

            "class_floors_frozen":
                False,

            "temporal_smoothing_frozen":
                False,

            "actuation_rate_frozen":
                False,

            "formal_evaluation":
                False,
        },

        "scientific_execution": {
            "frozen_Gaussian_model_forward":
                True,

            "training":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "formal_inference":
                False,
        },

        "regression": {
            "return_code":
                regression.returncode,

            "passed":
                regression_passed,

            "status":
                "PASS",
        },

        "next": (
            "Block6.6 Part2C: bind the already-frozen "
            "Part1A current-headlamp actor eligibility at "
            "box level to the Part2B association records; "
            "only after that gate materialize N_MC=8192 "
            "P_occ for the eligible matched predictive actors."
        ),
    }

    report_sha = (
        atomic_json_write(
            REPORT,
            report,
        )
    )

    require(
        free_gib()
        >=
        MIN_FREE_GIB,
        (
            "Storage reserve violated "
            "after Part2B."
        ),
    )

    print(
        "report                      =",
        REPORT,
    )

    print(
        "report SHA256               =",
        report_sha,
    )

    print(
        "posterior JSONL SHA256      =",
        posterior_sha,
    )

    print(
        "match JSONL SHA256          =",
        match_sha,
    )

    print(
        "fallback JSONL SHA256       =",
        fallback_sha,
    )

    print(
        "scene summary SHA256        =",
        scene_summary_sha,
    )

    print()
    print(
        "=" * 76
    )

    print(
        "BLOCK 6.6 PART2B — FINAL"
    )

    print(
        "=" * 76
    )

    print(
        "STATUS = "
        "PASS_BLOCK66_PART2B_"
        "COHORT_POSTERIOR_ASSOCIATION_MATERIALIZED"
    )

    print(
        "development scenarios       = 120/120"
    )

    print(
        "posterior records           =",
        len(
            all_posterior
        ),
    )

    print(
        "causal ADB boxes            =",
        total_boxes,
    )

    print(
        "association matches         =",
        total_matches,
    )

    print(
        "reactive fallback boxes     =",
        total_fallback,
    )

    print(
        "unmatched predictors        =",
        total_unmatched_predictors,
    )

    print(
        "ADB boxes by class          =",
        boxes_by_class,
    )

    print(
        "matches by class            =",
        matches_by_class,
    )

    print(
        "fallback by class           =",
        fallback_by_class,
    )

    print(
        "all classes matched         =",
        all_classes_have_association_support,
    )

    print(
        "association threshold       = NONE"
    )

    print(
        "association class gate      = NO"
    )

    print(
        "perfect/truth ID            = NO / NO"
    )

    print(
        "future GT                   = NO"
    )

    print(
        "training/recalibration      = NO / NO"
    )

    print(
        "N_MC=8192 executed          = NO"
    )

    print(
        "P_occ materialized          = NO"
    )

    print(
        "class policy frozen         = NO"
    )

    print(
        "formal evaluation           = NO"
    )

    print(
        "Stage6 regression           = PASS"
    )

    print(
        "NEXT = BLOCK6.6 PART2C "
        "EXACT HEADLAMP-ELIGIBILITY JOIN "
        "THEN ELIGIBLE N8192 P_OCC"
    )

    print(
        "terminal remains open       = YES"
    )

    print(
        "=" * 76
    )


try:
    main()

except BaseException as exc:

    print()
    print(
        "=" * 76
    )

    print(
        "BLOCK 6.6 PART2B — CONTROLLED BLOCK"
    )

    print(
        "=" * 76
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
    traceback.print_exc(
        limit=24
    )

    print()
    print(
        "resumable per-scene cache = PRESERVED"
    )

    print(
        "cohort membership modified = NO"
    )

    print(
        "training/recalibration     = NO / NO"
    )

    print(
        "N_MC=8192 executed         = NO"
    )

    print(
        "class policy frozen        = NO"
    )

    print(
        "formal outcomes read       = NO"
    )

    print(
        "terminal remains open      = YES"
    )

    print(
        "=" * 76
    )

# Deliberately no sys.exit().

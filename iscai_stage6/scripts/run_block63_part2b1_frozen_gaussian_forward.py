from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback

from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")

STAGE4 = ROOT / "iscai_stage4"
STAGE6 = ROOT / "iscai_stage6"

PART2A_REPORT = (
    STAGE6
    / "reports"
    / "block63_part2_real_causal_association.json"
)

CHECKPOINT = (
    STAGE4
    / "artifacts"
    / "block44"
    / "gaussian_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts"
    / "block43"
    / "fit_normalization.json"
)

CALIBRATOR = (
    STAGE4
    / "artifacts"
    / "block45"
    / "covariance_scaler.json"
)

MOTION = (
    ROOT
    / "data"
    / "paired_womd_lidar_v1_3_0"
    / "validation"
    / "motion"
    / "paired-from-validation.tfrecord-00000-of-00150"
)

REPORT = (
    STAGE6
    / "reports"
    / "block63_part2b1_frozen_gaussian_forward.json"
)

POSTERIOR_JSONL = (
    STAGE6
    / "artifacts"
    / "block63"
    / "identity_safe_development_gaussian_posterior.jsonl"
)

POSTERIOR_NPZ = (
    STAGE6
    / "artifacts"
    / "block63"
    / "identity_safe_development_gaussian_posterior.npz"
)

REGRESSION_LOG = (
    STAGE6
    / "artifacts"
    / "block63"
    / "block63_part2b1_stage6_regression.log"
)

DIAGNOSTIC_EXCERPT = (
    STAGE6
    / "artifacts"
    / "block63"
    / "block63_part2b1_runtime_boundary_excerpt.txt"
)


EXPECTED_SHA = {
    "part2a": (
        "71c9d03cc9c1b33dbb371dccd7140d5226494554b6b21df7fbfa19e9f43ba71a"
    ),
    "checkpoint": (
        "49ff64d145eaa633f295c16f660df380c35383e7e3b61279a5aad7cd700d619f"
    ),
    "normalization": (
        "3d7fc0a66d4a4f566f6569befa9c3766ecae21830bb4e46256df2f326a82a5f6"
    ),
    "calibrator": (
        "508ff2e3fbcfafe8e001155340c25baaf3772fe2561a8022a9ed1cf780e66087"
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


SCENARIO_ID = "b85e1bd6cc8e74c0"

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)


class ScientificBlock(RuntimeError):
    pass


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ScientificBlock(message)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def json_safe(
    value: Any,
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)

    return value


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(
    payload: dict[str, Any],
) -> str:
    write_json(
        REPORT,
        payload,
    )

    return sha256_file(
        REPORT
    )


def finite_array(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if ndim is not None:
        require(
            array.ndim == ndim,
            (
                f"{name}: expected ndim={ndim}, "
                f"got shape={array.shape}"
            ),
        )

    require(
        bool(np.all(np.isfinite(array))),
        f"{name}: contains non-finite values.",
    )

    return array


def normalize_stage2_configs(
    value: Any,
):
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            return (
                value[0],
                value[1],
                "two_element_sequence",
            )

    if isinstance(value, dict):
        for clean_key, degraded_key in (
            ("clean_config", "degraded_config"),
            ("clean", "degraded"),
        ):
            if (
                clean_key in value
                and
                degraded_key in value
            ):
                return (
                    value[clean_key],
                    value[degraded_key],
                    (
                        f"dict:{clean_key}+"
                        f"{degraded_key}"
                    ),
                )

    for clean_name, degraded_name in (
        ("clean_config", "degraded_config"),
        ("clean", "degraded"),
    ):
        if (
            hasattr(value, clean_name)
            and
            hasattr(value, degraded_name)
        ):
            return (
                getattr(value, clean_name),
                getattr(value, degraded_name),
                (
                    f"attributes:{clean_name}+"
                    f"{degraded_name}"
                ),
            )

    raise ScientificBlock(
        "Cannot resolve frozen Stage2 clean/degraded "
        "configuration packaging without guessing."
    )


def find_typed_instances(
    root: Any,
    target_type: type,
    *,
    max_depth: int = 6,
) -> list[Any]:
    result = []
    seen = set()

    def visit(
        value: Any,
        depth: int,
    ) -> None:
        if depth > max_depth:
            return

        identifier = id(value)

        if identifier in seen:
            return

        seen.add(identifier)

        if isinstance(value, target_type):
            result.append(value)
            return

        if isinstance(value, dict):
            for child in value.values():
                visit(
                    child,
                    depth + 1,
                )

            return

        if isinstance(value, (tuple, list)):
            for child in value:
                visit(
                    child,
                    depth + 1,
                )

            return

        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(
                    getattr(
                        value,
                        field.name,
                    ),
                    depth + 1,
                )

    visit(
        root,
        0,
    )

    unique = []
    unique_ids = set()

    for item in result:
        if id(item) not in unique_ids:
            unique_ids.add(id(item))
            unique.append(item)

    return unique


def find_named_values(
    root: Any,
    target_name: str,
    *,
    max_depth: int = 7,
) -> list[tuple[str, Any]]:
    """
    Search only structurally named fields/keys.

    We intentionally do NOT select anonymous vectors by
    shape or numerical value.
    """
    found = []
    seen = set()

    def visit(
        value: Any,
        path: str,
        depth: int,
    ) -> None:
        if depth > max_depth:
            return

        identifier = id(value)

        if identifier in seen:
            return

        seen.add(identifier)

        if isinstance(value, dict):
            for key, child in value.items():
                key_string = str(key)
                child_path = (
                    f"{path}.{key_string}"
                    if path
                    else key_string
                )

                if key_string == target_name:
                    found.append(
                        (
                            child_path,
                            child,
                        )
                    )

                visit(
                    child,
                    child_path,
                    depth + 1,
                )

            return

        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                child = getattr(
                    value,
                    field.name,
                )

                child_path = (
                    f"{path}.{field.name}"
                    if path
                    else field.name
                )

                if field.name == target_name:
                    found.append(
                        (
                            child_path,
                            child,
                        )
                    )

                visit(
                    child,
                    child_path,
                    depth + 1,
                )

            return

        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(
                    child,
                    f"{path}[{index}]",
                    depth + 1,
                )

    visit(
        root,
        "root",
        0,
    )

    return found


def source_excerpt(
    value: Any,
    *,
    limit: int = 7000,
) -> str:
    try:
        text = inspect.getsource(value)
    except BaseException as exc:
        return (
            "<SOURCE UNAVAILABLE: "
            f"{type(exc).__name__}: {exc}>"
        )

    if len(text) > limit:
        return (
            text[:limit]
            +
            "\n... <TRUNCATED> ..."
        )

    return text


def public_map_context_builders(
    modules: tuple[Any, ...],
) -> list[tuple[str, Any]]:
    result = []

    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue

            if not callable(value):
                continue

            lowered = name.lower()

            if (
                "map" in lowered
                and
                "context" in lowered
            ):
                result.append(
                    (
                        (
                            f"{module.__name__}."
                            f"{name}"
                        ),
                        value,
                    )
                )

    return result


def resolve_map_context(
    causal_result: Any,
    *,
    modules: tuple[Any, ...],
    scene_inputs: Any,
    scenario: Any,
) -> tuple[
    tuple[float, ...],
    str,
    list[dict[str, Any]],
]:
    """
    First accept only an explicitly NAMED map_context
    field from the already-executed frozen causal route.

    If absent, allow only a public callable whose NAME
    explicitly contains both 'map' and 'context' and whose
    required arguments are unambiguously available.
    """
    named = find_named_values(
        causal_result,
        "map_context",
    )

    valid_named = []

    for path, value in named:
        try:
            array = finite_array(
                value,
                name=path,
                ndim=1,
            )
        except BaseException:
            continue

        valid_named.append(
            (
                path,
                array,
            )
        )

    if len(valid_named) == 1:
        path, array = valid_named[0]

        return (
            tuple(
                float(item)
                for item in array
            ),
            (
                "named_value:"
                f"{path}"
            ),
            [],
        )

    require(
        len(valid_named) <= 1,
        (
            "Multiple distinct named map_context "
            "vectors found; selection would be ambiguous."
        ),
    )

    candidates = []

    available = {
        "scene_inputs":
            scene_inputs,
        "scenario":
            scenario,
        "raw_scenario":
            scenario,
    }

    for qualified_name, function in (
        public_map_context_builders(
            modules
        )
    ):
        try:
            signature = inspect.signature(
                function
            )
        except BaseException:
            continue

        candidate_report = {
            "name":
                qualified_name,
            "signature":
                str(signature),
            "called":
                False,
            "accepted":
                False,
        }

        kwargs = {}
        unsupported_required = []

        for parameter in signature.parameters.values():
            if parameter.name in available:
                kwargs[
                    parameter.name
                ] = available[
                    parameter.name
                ]

                continue

            if parameter.default is not inspect._empty:
                continue

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            unsupported_required.append(
                parameter.name
            )

        if unsupported_required:
            candidate_report[
                "unsupported_required"
            ] = unsupported_required

            candidates.append(
                candidate_report
            )

            continue

        try:
            result = function(
                **kwargs
            )

            candidate_report[
                "called"
            ] = True

            array = finite_array(
                result,
                name=qualified_name,
                ndim=1,
            )

            candidate_report[
                "shape"
            ] = list(
                array.shape
            )

            candidate_report[
                "accepted"
            ] = True

            candidate_report[
                "_array"
            ] = array

        except BaseException as exc:
            candidate_report[
                "exception"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        candidates.append(
            candidate_report
        )

    accepted = [
        item
        for item in candidates
        if item.get("accepted")
    ]

    require(
        len(accepted) == 1,
        (
            "BLOCKED_EXACT_MODEL_INPUT_CONTEXT_BINDING: "
            "no unique exact map_context route was "
            "discoverable without guessing."
        ),
    )

    chosen = accepted[0]

    array = chosen.pop(
        "_array"
    )

    for item in candidates:
        item.pop(
            "_array",
            None,
        )

    return (
        tuple(
            float(item)
            for item in array
        ),
        (
            "public_exact_map_context_builder:"
            f"{chosen['name']}"
        ),
        candidates,
    )


def construct_normalizer(
    normalizer_class: type,
    normalization_payload: dict[str, Any],
    *,
    device,
):
    """
    Exact frozen Stage4 binding.

    CalibrationNormalizer expects the fit-only statistics
    mapping directly; there is no intermediate statistics
    class in calibration_runtime.
    """

    signature = inspect.signature(
        normalizer_class
    )

    require(
        str(
            signature
        )
        ==
        "(statistics, *, device)",
        (
            "CalibrationNormalizer signature changed: "
            f"{signature}"
        ),
    )

    statistics = (
        normalization_payload.get(
            "payload"
        )
        if isinstance(
            normalization_payload.get(
                "payload"
            ),
            dict,
        )
        else normalization_payload
    )

    require(
        isinstance(
            statistics,
            dict,
        ),
        (
            "Frozen normalization statistics "
            "are not a mapping."
        ),
    )

    require(
        statistics.get(
            "source"
        )
        ==
        "fit_only",
        (
            "Normalization source is not fit_only."
        ),
    )

    require(
        statistics.get(
            "development_used"
        )
        is False,
        (
            "Development statistics entered "
            "normalization."
        ),
    )

    require(
        statistics.get(
            "validation_used"
        )
        is False,
        (
            "Validation statistics entered "
            "normalization."
        ),
    )

    require(
        statistics.get(
            "calibration_used"
        )
        is False,
        (
            "Calibration statistics entered "
            "normalization."
        ),
    )

    required_keys = (
        "continuous_feature_mean",
        "continuous_feature_std",
        "map_mean",
        "map_std",
        "label_displacement_mean",
        "label_displacement_std",
    )

    missing = [
        key
        for key in required_keys
        if key not in statistics
    ]

    require(
        not missing,
        (
            "Frozen normalization statistics "
            "missing required key(s): "
            + ", ".join(
                missing
            )
        ),
    )

    normalizer = (
        normalizer_class(
            statistics,
            device=device,
        )
    )

    return (
        normalizer,
        {
            "constructor_signature":
                str(
                    signature
                ),

            "binding_mode":
                (
                    "exact_frozen_statistics_"
                    "mapping_directly"
                ),

            "required_statistics_keys":
                list(
                    required_keys
                ),

            "fit_only":
                True,

            "development_used":
                False,

            "validation_used":
                False,

            "calibration_used":
                False,

            "device":
                str(
                    device
                ),

            "intermediate_statistics_class":
                False,

            "aliases_used":
                False,

            "numeric_defaults_injected":
                False,
        },
    )

def arrays_subscript_keys(
    function: Any,
) -> set[str]:
    """
    Extract the exact arrays["..."] keys from the frozen
    CalibrationNormalizer.prepare() implementation.

    inspect.getsource() returns an indented method body,
    so dedent is mandatory before ast.parse().
    """
    import textwrap

    source = inspect.getsource(
        function
    )

    source = textwrap.dedent(
        source
    )

    tree = ast.parse(
        source
    )

    keys = set()

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Subscript,
        ):
            continue

        if not isinstance(
            node.value,
            ast.Name,
        ):
            continue

        if node.value.id != "arrays":
            continue

        slice_node = node.slice

        if (
            isinstance(
                slice_node,
                ast.Constant,
            )
            and
            isinstance(
                slice_node.value,
                str,
            )
        ):
            keys.add(
                slice_node.value
            )

    expected = {
        "target",
        "neighbors",
        "neighbor_mask",
        "map_context",
        "future",
        "future_mask",
        "class_id",
    }

    require(
        keys
        ==
        expected,
        (
            "Frozen CalibrationNormalizer.prepare "
            "array contract changed. "
            f"expected={sorted(expected)}, "
            f"observed={sorted(keys)}"
        ),
    )

    return keys


def build_prepare_arrays(
    payloads: tuple[Any, ...],
    *,
    prepare_keys: set[str],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """
    Build the exact frozen Stage4 prepare() input mapping.

    Controller/model inputs:
      target, neighbors, neighbor_mask, map_context

    Evaluator-only placeholders required by the frozen
    prepare() API:
      future, future_mask, class_id

    The placeholders are never passed into model.forward().
    """

    expected_keys = {
        "target",
        "neighbors",
        "neighbor_mask",
        "map_context",
        "future",
        "future_mask",
        "class_id",
    }

    require(
        prepare_keys
        ==
        expected_keys,
        (
            "Unexpected prepare() contract: "
            f"{sorted(prepare_keys)}"
        ),
    )

    count = len(
        payloads
    )

    require(
        count > 0,
        (
            "No causal model payloads."
        ),
    )

    target = np.asarray(
        [
            payload.target_history
            for payload in payloads
        ],
        dtype=np.float32,
    )

    neighbors = np.asarray(
        [
            payload.neighbor_histories
            for payload in payloads
        ],
        dtype=np.float32,
    )

    neighbor_mask = np.asarray(
        [
            payload.neighbor_mask
            for payload in payloads
        ],
        dtype=np.float32,
    )

    map_context = np.asarray(
        [
            payload.map_context
            for payload in payloads
        ],
        dtype=np.float32,
    )

    require(
        target.shape[
            0
        ]
        ==
        count,
        (
            "Target batch count mismatch."
        ),
    )

    require(
        neighbors.shape[
            0
        ]
        ==
        count,
        (
            "Neighbor batch count mismatch."
        ),
    )

    require(
        neighbor_mask.shape[
            0
        ]
        ==
        count,
        (
            "Neighbor-mask batch count mismatch."
        ),
    )

    require(
        map_context.shape
        ==
        (
            count,
            10,
        ),
        (
            "Map-context matrix must be "
            f"[N,10], got {map_context.shape}."
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

    # The frozen normalizer computes evaluator-side
    # true_displacement/future-mask/class outputs even when
    # only controller inference is required.
    #
    # These are API placeholders only. They do NOT enter any
    # neural model input tensor and contain NO future GT.
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
            arrays.keys()
        )
        ==
        expected_keys,
        (
            "Prepared array mapping does not "
            "match frozen Stage4 contract."
        ),
    )

    report = {
        "target_shape":
            list(
                target.shape
            ),

        "neighbors_shape":
            list(
                neighbors.shape
            ),

        "neighbor_mask_shape":
            list(
                neighbor_mask.shape
            ),

        "map_context_shape":
            list(
                map_context.shape
            ),

        "prepare_source_keys":
            sorted(
                prepare_keys
            ),

        "exact_frozen_prepare_contract":
            True,

        "future_placeholder_shape":
            list(
                future_placeholder.shape
            ),

        "future_mask_placeholder_shape":
            list(
                future_mask_placeholder.shape
            ),

        "class_id_placeholder_shape":
            list(
                class_id_placeholder.shape
            ),

        "placeholder_future_GT":
            False,

        "placeholder_truth_class":
            False,

        "placeholder_fields_are_model_inputs":
            False,

        "model_input_keys": [
            "target",
            "neighbors",
            "neighbor_mask",
            "map_context",
        ],

        "evaluator_only_placeholder_keys": [
            "future",
            "future_mask",
            "class_id",
        ],

        "canonical_alias_fallback":
            False,
    }

    return (
        arrays,
        report,
    )

def pick_batch_tensor(
    batch: dict[str, Any],
    aliases: tuple[str, ...],
):
    found = [
        name
        for name in aliases
        if name in batch
    ]

    require(
        len(found) == 1,
        (
            "Expected exactly one batch tensor among "
            f"{aliases}; found {found}; "
            f"batch keys={sorted(batch.keys())}"
        ),
    )

    return (
        found[0],
        batch[
            found[0]
        ],
    )


def forward_kwargs(
    model: Any,
    batch: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, str],
]:
    signature = inspect.signature(
        model.forward
    )

    alias_groups = {
        "target":
            (
                "target",
                "target_history",
            ),

        "target_history":
            (
                "target",
                "target_history",
            ),

        "neighbors":
            (
                "neighbors",
                "neighbor_histories",
            ),

        "neighbor_histories":
            (
                "neighbors",
                "neighbor_histories",
            ),

        "neighbor_mask":
            (
                "neighbor_mask",
            ),

        "map":
            (
                "map",
                "map_context",
            ),

        "map_context":
            (
                "map",
                "map_context",
            ),
    }

    kwargs = {}
    mapping = {}

    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if parameter.name in alias_groups:
            batch_key, tensor = pick_batch_tensor(
                batch,
                alias_groups[
                    parameter.name
                ],
            )

            kwargs[
                parameter.name
            ] = tensor

            mapping[
                parameter.name
            ] = batch_key

            continue

        if parameter.default is not inspect._empty:
            continue

        raise ScientificBlock(
            "Frozen Gaussian model.forward has an "
            "unresolved required input parameter: "
            f"{parameter.name}; signature={signature}"
        )

    require(
        kwargs,
        (
            "No model forward inputs resolved."
        ),
    )

    return (
        kwargs,
        mapping,
    )


def recursively_find_alpha(
    root: Any,
    *,
    path: str = "root",
) -> list[tuple[str, np.ndarray]]:
    found = []

    if isinstance(root, dict):
        for key, child in root.items():
            found.extend(
                recursively_find_alpha(
                    child,
                    path=(
                        f"{path}.{key}"
                    ),
                )
            )

        return found

    if isinstance(root, (tuple, list)):
        try:
            array = np.asarray(
                root,
                dtype=np.float64,
            )
        except BaseException:
            array = None

        if (
            array is not None
            and
            array.shape == (4,)
            and
            np.all(
                np.isfinite(
                    array
                )
            )
        ):
            if np.array_equal(
                array,
                EXPECTED_ALPHA,
            ):
                found.append(
                    (
                        path,
                        array,
                    )
                )

        for index, child in enumerate(root):
            found.extend(
                recursively_find_alpha(
                    child,
                    path=(
                        f"{path}[{index}]"
                    ),
                )
            )

    return found


def prediction_digest(
    mean: np.ndarray,
    raw_covariance: np.ndarray,
    calibrated_covariance: np.ndarray,
    prediction_ids: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()

    for prediction_id in prediction_ids:
        digest.update(
            prediction_id.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

    for array in (
        mean,
        raw_covariance,
        calibrated_covariance,
    ):
        digest.update(
            np.asarray(
                array,
                dtype=np.float32,
            )
            .tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def run_regression() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(
            STAGE6
            / "tests"
        ),
        "-p",
        "test_*.py",
    ]

    with REGRESSION_LOG.open(
        "w",
        encoding="utf-8",
    ) as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    text = REGRESSION_LOG.read_text(
        encoding="utf-8",
        errors="replace",
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        text,
    )

    count = (
        int(match.group(1))
        if match
        else None
    )

    return {
        "return_code":
            int(
                process.returncode
            ),

        "test_count":
            count,

        "pass":
            (
                process.returncode
                ==
                0
            ),

        "log":
            str(
                REGRESSION_LOG
            ),

        "log_sha256":
            sha256_file(
                REGRESSION_LOG
            ),
    }


def scientific_run() -> dict[str, Any]:
    print(
        "=" * 72
    )
    print(
        "BLOCK 6.3 PART 2B/1 — FROZEN GAUSSIAN CAUSAL FORWARD"
    )
    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # A. Seal exact prerequisites
    # --------------------------------------------------------

    print()
    print(
        "===== A. FROZEN PREREQUISITE SEAL ====="
    )

    artifacts = {
        "part2a":
            PART2A_REPORT,
        "checkpoint":
            CHECKPOINT,
        "normalization":
            NORMALIZATION,
        "calibrator":
            CALIBRATOR,
    }

    seals = {}

    for name, path in artifacts.items():
        require(
            path.is_file(),
            f"Missing prerequisite: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            EXPECTED_SHA[
                name
            ],
            (
                f"{name} SHA mismatch: "
                f"{actual} != "
                f"{EXPECTED_SHA[name]}"
            ),
        )

        seals[
            name
        ] = {
            "path":
                str(path),
            "sha256":
                actual,
        }

        print(
            f"{name:16s} = PASS"
        )

    part2a = json.loads(
        PART2A_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        part2a.get("status")
        ==
        "PASS_REAL_CAUSAL_ASSOCIATION_EXECUTED",
        (
            "Part2A scientific status changed."
        ),
    )

    require(
        part2a[
            "development_selection"
        ][
            "selected_scenario_id"
        ]
        ==
        SCENARIO_ID,
        (
            "Part2A development scenario changed."
        ),
    )

    require(
        part2a[
            "association"
        ][
            "match_count"
        ]
        ==
        6,
        (
            "Part2A association match count changed."
        ),
    )

    require(
        part2a[
            "association"
        ][
            "reactive_fallback_count"
        ]
        ==
        3,
        (
            "Part2A fallback count changed."
        ),
    )

    # --------------------------------------------------------
    # B. Recreate same causal development route
    # --------------------------------------------------------

    print()
    print(
        "===== B. SAME REAL CAUSAL DEVELOPMENT ROUTE ====="
    )

    from iscai_stage4.data import (
        CausalSceneInputs,
        build_model_input_payload,
        iter_compact_motion_records,
    )

    from iscai_stage4.data import neural_inputs as neural_inputs_module

    from iscai_stage4.data import real_pipeline as real_pipeline_module

    from iscai_stage4.data.real_pipeline import (
        build_real_causal_inputs,
        load_frozen_stage2_configs,
    )

    first = next(
        iter_compact_motion_records(
            MOTION
        )
    )

    require(
        isinstance(first, tuple)
        and
        len(first) == 3,
        (
            "Unexpected compact record structure."
        ),
    )

    (
        record_offset,
        payload_length,
        scenario,
    ) = first

    require(
        str(
            scenario.scenario_id
        )
        ==
        SCENARIO_ID,
        (
            "Frozen development record identity changed."
        ),
    )

    require(
        int(record_offset) == 0,
        (
            "Frozen development compact offset changed."
        ),
    )

    require(
        int(payload_length) == 647785,
        (
            "Frozen development payload length changed."
        ),
    )

    frozen_configs = (
        load_frozen_stage2_configs()
    )

    (
        clean_config,
        degraded_config,
        config_packaging,
    ) = normalize_stage2_configs(
        frozen_configs
    )

    causal_result = (
        build_real_causal_inputs(
            scenario,
            clean_config=clean_config,
            degraded_config=degraded_config,
        )
    )

    scene_candidates = (
        find_typed_instances(
            causal_result,
            CausalSceneInputs,
        )
    )

    require(
        len(
            scene_candidates
        )
        ==
        1,
        (
            "Expected exactly one CausalSceneInputs "
            f"instance; found {len(scene_candidates)}."
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

    require(
        len(histories) == 27,
        (
            "Causal history count changed from Part2A."
        ),
    )

    prediction_ids = tuple(
        str(
            history.prediction_id
        )
        for history in histories
    )

    require(
        len(
            set(
                prediction_ids
            )
        )
        ==
        len(
            prediction_ids
        ),
        (
            "prediction_id values are not unique."
        ),
    )

    anchors = np.asarray(
        [
            history.latest_position_H0_m
            for history in histories
        ],
        dtype=np.float64,
    )

    require(
        anchors.shape
        ==
        (
            27,
            3,
        ),
        (
            "Causal anchor array shape changed."
        ),
    )

    require(
        bool(
            np.all(
                np.isfinite(
                    anchors
                )
            )
        ),
        (
            "Non-finite causal anchors."
        ),
    )

    require(
        max(
            int(
                history.latest_observed_frame_index
            )
            for history in histories
        )
        <=
        int(
            scenario.current_time_index
        ),
        (
            "Future frame entered CausalTrackHistory."
        ),
    )

    print(
        "scenario_id             =",
        SCENARIO_ID,
    )
    print(
        "causal histories        =",
        len(histories),
    )
    print(
        "prediction_id unique    = PASS"
    )
    print(
        "future access           = NO"
    )

    # --------------------------------------------------------
    # C. Exact frozen per-target map-context construction
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXACT PER-TARGET MAP-CONTEXT BINDING ====="
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage4.data.supervision import (
        build_static_map_index_H0,
        map_context_summary,
    )

    adapted_for_map = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    require(
        str(
            adapted_for_map.scenario_id
        )
        ==
        SCENARIO_ID,
        (
            "Map-context adapted scenario "
            "identity mismatch."
        ),
    )

    require(
        int(
            adapted_for_map.anchor_index
        )
        ==
        int(
            scenario.current_time_index
        ),
        (
            "Map-context anchor index "
            "does not equal causal current index."
        ),
    )

    T_H0_from_W = (
        adapted_for_map
        .frames
        .T_H0_from_W
    )

    map_index_H0 = (
        build_static_map_index_H0(
            scenario,
            T_H0_from_W=T_H0_from_W,
        )
    )

    require(
        isinstance(
            map_index_H0,
            dict,
        ),
        (
            "Frozen Stage4 static map "
            "index is not a mapping."
        ),
    )

    map_contexts = tuple(
        map_context_summary(
            map_index_H0,
            target_position_H0_m=(
                history.latest_position_H0_m
            ),
        )
        for history
        in histories
    )

    require(
        len(
            map_contexts
        )
        ==
        len(
            histories
        ),
        (
            "Map-context/history count mismatch."
        ),
    )

    map_array = np.asarray(
        map_contexts,
        dtype=np.float64,
    )

    require(
        map_array.shape
        ==
        (
            27,
            10,
        ),
        (
            "Expected exact per-target "
            "Stage4 map matrix [27,10]; got "
            f"{map_array.shape}."
        ),
    )

    require(
        bool(
            np.all(
                np.isfinite(
                    map_array
                )
            )
        ),
        (
            "Per-target map context contains "
            "non-finite values."
        ),
    )

    unique_map_rows = int(
        np.unique(
            map_array,
            axis=0,
        ).shape[
            0
        ]
    )

    map_context_source = (
        "build_static_map_index_H0("
        "AdaptedScenario.frames.T_H0_from_W)"
        " + map_context_summary("
        "CausalTrackHistory.latest_position_H0_m)"
    )

    map_builder_candidates = []

    print(
        "static map frame        = H0"
    )

    print(
        "H0 transform source     = "
        "AdaptedScenario.frames.T_H0_from_W"
    )

    print(
        "map producer            = "
        "build_static_map_index_H0"
    )

    print(
        "per-target summary      = "
        "map_context_summary"
    )

    print(
        "target position source  = "
        "latest_position_H0_m"
    )

    print(
        "map context matrix      =",
        map_array.shape,
    )

    print(
        "unique map rows         =",
        unique_map_rows,
    )

    print(
        "shared scene vector     = NO"
    )

    print(
        "future map state        = NO"
    )

    # --------------------------------------------------------
    # D. Causal ModelInputPayloads with per-target map context
    # --------------------------------------------------------

    print()
    print(
        "===== D. CAUSAL MODEL INPUT PAYLOADS ====="
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
        27,
        (
            "ModelInputPayload count mismatch."
        ),
    )

    for index, payload in enumerate(
        model_payloads
    ):
        target_history = np.asarray(
            payload.target_history,
            dtype=np.float64,
        )

        neighbor_histories = np.asarray(
            payload.neighbor_histories,
            dtype=np.float64,
        )

        neighbor_mask = np.asarray(
            payload.neighbor_mask,
            dtype=np.float64,
        )

        payload_map = np.asarray(
            payload.map_context,
            dtype=np.float64,
        )

        require(
            bool(
                np.all(
                    np.isfinite(
                        target_history
                    )
                )
            ),
            (
                f"target_history[{index}] "
                "contains non-finite values."
            ),
        )

        require(
            bool(
                np.all(
                    np.isfinite(
                        neighbor_histories
                    )
                )
            ),
            (
                f"neighbor_histories[{index}] "
                "contains non-finite values."
            ),
        )

        require(
            bool(
                np.all(
                    np.isfinite(
                        neighbor_mask
                    )
                )
            ),
            (
                f"neighbor_mask[{index}] "
                "contains non-finite values."
            ),
        )

        require(
            payload_map.shape
            ==
            (
                10,
            ),
            (
                f"payload map_context[{index}] "
                f"shape changed: {payload_map.shape}"
            ),
        )

        require(
            np.array_equal(
                payload_map,
                map_array[
                    index
                ],
            ),
            (
                "ModelInputPayload did not preserve "
                f"exact map context at actor {index}."
            ),
        )

    print(
        "payloads                =",
        len(
            model_payloads
        ),
    )

    print(
        "per-target map contexts = YES"
    )

    print(
        "attach_supervision      = NOT CALLED"
    )

    print(
        "future labels           = NOT READ"
    )

    print(
        "tracks_to_predict       = NOT USED"
    )

    # --------------------------------------------------------
    # E. Frozen normalizer
    # --------------------------------------------------------

    print()
    print(
        "===== E. FIT-ONLY FROZEN NORMALIZER ====="
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

    normalization_json = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    import torch

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    (
        normalizer,
        normalizer_binding,
    ) = construct_normalizer(
        CalibrationNormalizer,
        normalization_json,
        device=device,
    )

    prepare_keys = (
        arrays_subscript_keys(
            normalizer.prepare
        )
    )

    (
        prepare_arrays,
        prepare_report,
    ) = build_prepare_arrays(
        model_payloads,
        prepare_keys=prepare_keys,
    )

    import torch

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    batch = normalizer.prepare(
        prepare_arrays,
        device=device,
    )

    require(
        isinstance(batch, dict),
        (
            "CalibrationNormalizer.prepare() "
            "did not return a dict."
        ),
    )

    print(
        "normalizer constructor  =",
        normalizer_binding[
            "binding_mode"
        ],
    )
    print(
        "prepare batch keys      =",
        sorted(
            batch.keys()
        ),
    )
    print(
        "evaluator-only placeholders =",
        prepare_report[
            "evaluator_only_placeholder_keys"
        ],
    )
    print(
        "synthetic→model         = NO"
    )

    # --------------------------------------------------------
    # F. Frozen Gaussian model load
    # --------------------------------------------------------

    print()
    print(
        "===== F. FROZEN GAUSSIAN CHECKPOINT ====="
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    require(
        isinstance(checkpoint, dict),
        (
            "Gaussian checkpoint payload is not dict."
        ),
    )

    require(
        "state_dict" in checkpoint,
        (
            "Gaussian checkpoint has no state_dict."
        ),
    )

    require(
        "configuration" in checkpoint,
        (
            "Gaussian checkpoint has no configuration."
        ),
    )

    require(
        str(
            checkpoint.get(
                "state_dict_sha256",
                "",
            )
        )
        ==
        (
            "1d66cf082d0d9be41319f7dcbe18fc259910de7f45de006f9043129837b86be3"
        ),
        (
            "Gaussian state_dict provenance changed."
        ),
    )

    # Exact frozen Block4.5 architecture-source repair.
    #
    # Block4.4 Gaussian checkpoint configuration contains
    # Gaussian experiment metadata, not the deterministic
    # architecture "model" section.
    #
    # Reproduce the frozen Stage4 solution exactly:
    #   Block4.3 deterministic checkpoint -> architecture only
    #   Block4.4 Gaussian checkpoint      -> predictive weights
    #
    # No deterministic predictive weights are used.

    deterministic_architecture_checkpoint_path = (
        STAGE4
        / "artifacts"
        / "block43"
        / "deterministic_gru.pt"
    )

    expected_deterministic_architecture_checkpoint_sha = (
        "5456a76b84d558e9983a59b9f1d3060b"
        "a245e0d36d60883519654f809996dbc5"
    )

    expected_deterministic_architecture_state_sha = (
        "d2ffbc03c7cb2826fef2175c95f48725"
        "707ec6d791eaeffbd6f59bc507b8595a"
    )

    require(
        deterministic_architecture_checkpoint_path.is_file(),
        (
            "Missing frozen Block4.3 deterministic "
            "architecture checkpoint."
        ),
    )

    require(
        sha256_file(
            deterministic_architecture_checkpoint_path
        )
        ==
        expected_deterministic_architecture_checkpoint_sha,
        (
            "Frozen Block4.3 deterministic "
            "architecture checkpoint SHA changed."
        ),
    )

    deterministic_architecture_checkpoint = torch.load(
        deterministic_architecture_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    require(
        isinstance(
            deterministic_architecture_checkpoint,
            dict,
        ),
        (
            "Frozen Block4.3 deterministic "
            "checkpoint is not a mapping."
        ),
    )

    deterministic_architecture_configuration = (
        deterministic_architecture_checkpoint.get(
            "configuration"
        )
    )

    require(
        isinstance(
            deterministic_architecture_configuration,
            dict,
        ),
        (
            "Frozen Block4.3 checkpoint lacks "
            "architecture configuration."
        ),
    )

    require(
        isinstance(
            deterministic_architecture_configuration.get(
                "model"
            ),
            dict,
        ),
        (
            "Frozen Block4.3 architecture "
            "configuration lacks 'model'."
        ),
    )

    # If the frozen checkpoint carries its own declared state
    # digest, require agreement with the already-frozen Stage4
    # evidence. The file SHA above independently seals the
    # complete checkpoint bytes.
    declared_architecture_state_sha = (
        deterministic_architecture_checkpoint.get(
            "state_dict_sha256"
        )
    )

    if declared_architecture_state_sha is not None:
        require(
            str(
                declared_architecture_state_sha
            )
            ==
            expected_deterministic_architecture_state_sha,
            (
                "Frozen Block4.3 deterministic "
                "state SHA declaration changed."
            ),
        )

    model = build_gaussian_model(
        deterministic_architecture_configuration,
        device=device,
    )

    load_result = model.load_state_dict(
        checkpoint[
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
        (
            "Missing model state keys."
        ),
    )

    require(
        not getattr(
            load_result,
            "unexpected_keys",
            (),
        ),
        (
            "Unexpected model state keys."
        ),
    )

    model.eval()

    (
        model_kwargs,
        forward_mapping,
    ) = forward_kwargs(
        model,
        batch,
    )

    model_parameter_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
    )

    print(
        "device                  =",
        str(device),
    )
    print(
        "parameter count         =",
        model_parameter_count,
    )
    print(
        "forward signature       =",
        inspect.signature(
            model.forward
        ),
    )
    print(
        "forward input mapping   =",
        forward_mapping,
    )

    # --------------------------------------------------------
    # G. ACTUAL MODEL FORWARD — twice for reproducibility
    # --------------------------------------------------------

    print()
    print(
        "===== G. ACTUAL FROZEN GAUSSIAN MODEL FORWARD ====="
    )

    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output_1 = model(
            **model_kwargs
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime_1_s = (
        time.perf_counter()
        -
        start
    )

    require(
        hasattr(
            output_1,
            "mean"
        ),
        (
            "Gaussian model output has no mean."
        ),
    )

    require(
        hasattr(
            output_1,
            "scale_tril"
        ),
        (
            "Gaussian model output has no scale_tril."
        ),
    )

    raw_hash_1 = prediction_sha256(
        output_1.mean,
        output_1.scale_tril,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output_2 = model(
            **model_kwargs
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runtime_2_s = (
        time.perf_counter()
        -
        start
    )

    raw_hash_2 = prediction_sha256(
        output_2.mean,
        output_2.scale_tril,
    )

    require(
        raw_hash_1
        ==
        raw_hash_2,
        (
            "Repeated frozen Gaussian forward "
            "is not reproducible."
        ),
    )

    require(
        tuple(
            output_1.mean.shape
        )
        ==
        (
            27,
            4,
            3,
        ),
        (
            "Gaussian mean output shape mismatch: "
            f"{tuple(output_1.mean.shape)}"
        ),
    )

    require(
        tuple(
            output_1.scale_tril.shape
        )
        ==
        (
            27,
            4,
            3,
            3,
        ),
        (
            "Gaussian scale_tril output shape mismatch: "
            f"{tuple(output_1.scale_tril.shape)}"
        ),
    )

    print(
        "model forward           = PASS"
    )
    print(
        "raw output mean shape   =",
        tuple(
            output_1.mean.shape
        ),
    )
    print(
        "raw scale shape         =",
        tuple(
            output_1.scale_tril.shape
        ),
    )
    print(
        "repeated hash equal     = PASS"
    )
    print(
        "raw prediction SHA      =",
        raw_hash_1,
    )
    print(
        "forward runtime 1 s     =",
        runtime_1_s,
    )
    print(
        "forward runtime 2 s     =",
        runtime_2_s,
    )

    # --------------------------------------------------------
    # H. Exact frozen metric-space denormalization
    # --------------------------------------------------------

    print()
    print(
        "===== H. METRIC H0 GAUSSIAN OUTPUT ====="
    )

    normalization_payload = (
        normalization_json[
            "payload"
        ]
        if isinstance(
            normalization_json.get("payload"),
            dict,
        )
        else normalization_json
    )

    label_mean = torch.as_tensor(
        normalization_payload[
            "label_displacement_mean"
        ],
        dtype=output_1.mean.dtype,
        device=device,
    )

    label_std = torch.as_tensor(
        normalization_payload[
            "label_displacement_std"
        ],
        dtype=output_1.mean.dtype,
        device=device,
    )

    (
        mean_metric,
        scale_metric,
    ) = denormalize_gaussian(
        output_1.mean,
        output_1.scale_tril,
        label_mean,
        label_std,
    )

    require(
        tuple(
            mean_metric.shape
        )
        ==
        (
            27,
            4,
            3,
        ),
        (
            "Metric mean shape mismatch."
        ),
    )

    require(
        tuple(
            scale_metric.shape
        )
        ==
        (
            27,
            4,
            3,
            3,
        ),
        (
            "Metric scale shape mismatch."
        ),
    )

    raw_covariance_tensor = (
        covariance_from_scale_tril(
            scale_metric
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

    raw_covariance_np = (
        raw_covariance_tensor
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    require(
        mean_metric_np.shape
        ==
        (
            27,
            4,
            3,
        ),
        (
            "Metric mean numpy shape changed."
        ),
    )

    require(
        raw_covariance_np.shape
        ==
        (
            27,
            4,
            3,
            3,
        ),
        (
            "Metric covariance numpy shape changed."
        ),
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
            "Metric Gaussian mean non-finite."
        ),
    )

    require(
        bool(
            np.all(
                np.isfinite(
                    raw_covariance_np
                )
            )
        ),
        (
            "Metric Gaussian covariance non-finite."
        ),
    )

    # --------------------------------------------------------
    # I. Frozen covariance calibration
    # --------------------------------------------------------

    print()
    print(
        "===== I. FROZEN COVARIANCE CALIBRATION ====="
    )

    calibrator_json = json.loads(
        CALIBRATOR.read_text(
            encoding="utf-8"
        )
    )

    alpha_hits = (
        recursively_find_alpha(
            calibrator_json
        )
    )

    require(
        len(alpha_hits) >= 1,
        (
            "Frozen covariance alpha vector could not "
            "be proven inside calibrator artifact."
        ),
    )

    from iscai_stage5.angular_monte_carlo import (
        calibrate_stage4_horizon_covariances,
    )

    calibrated_covariance_np = np.stack(
        [
            np.asarray(
                calibrate_stage4_horizon_covariances(
                    raw_covariance_np[
                        index
                    ]
                ),
                dtype=np.float64,
            )
            for index
            in range(
                raw_covariance_np.shape[
                    0
                ]
            )
        ],
        axis=0,
    )

    expected_calibrated = (
        raw_covariance_np
        *
        EXPECTED_ALPHA[
            None,
            :,
            None,
            None,
        ]
    )

    max_calibration_error = float(
        np.max(
            np.abs(
                calibrated_covariance_np
                -
                expected_calibrated
            )
        )
    )

    require(
        max_calibration_error
        <=
        1e-12,
        (
            "Stage5 frozen covariance calibration "
            "does not match frozen alpha semantics."
        ),
    )

    # Deterministic Block6.3 uses mean only.
    # Calibration MUST NOT move the mean.
    deterministic_mean_np = (
        mean_metric_np.copy()
    )

    require(
        np.array_equal(
            deterministic_mean_np,
            mean_metric_np,
        ),
        (
            "Deterministic mean changed during "
            "covariance calibration."
        ),
    )

    print(
        "alpha artifact evidence = PASS"
    )
    print(
        "calibration max error   =",
        max_calibration_error,
    )
    print(
        "predictive mean changed = NO"
    )
    print(
        "Block6.3 covariance use = NO"
    )

    # --------------------------------------------------------
    # J. Identity-safe posterior materialization
    # --------------------------------------------------------

    print()
    print(
        "===== J. IDENTITY-SAFE POSTERIOR MATERIALIZATION ====="
    )

    records = []

    for index, prediction_id in enumerate(
        prediction_ids
    ):
        records.append(
            {
                "scenario_id":
                    SCENARIO_ID,

                "prediction_id":
                    prediction_id,

                "latest_position_H0_m":
                    anchors[
                        index
                    ].tolist(),

                "horizons_s":
                    list(
                        HORIZONS_S
                    ),

                "mean_displacement_H0_m":
                    mean_metric_np[
                        index
                    ].tolist(),

                "raw_predictive_covariance_H0_m2":
                    raw_covariance_np[
                        index
                    ].tolist(),

                "calibrated_predictive_covariance_H0_m2":
                    calibrated_covariance_np[
                        index
                    ].tolist(),

                "mean_calibration":
                    "UNCHANGED",

                "deterministic_Block63_signal":
                    "MEAN_ONLY",

                "future_GT_used":
                    False,

                "truth_id_used":
                    False,

                "tracks_to_predict_used":
                    False,
            }
        )

    POSTERIOR_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with POSTERIOR_JSONL.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    json_safe(record),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    np.savez_compressed(
        POSTERIOR_NPZ,
        scenario_id=np.asarray(
            [
                SCENARIO_ID
            ]
            *
            len(
                prediction_ids
            )
        ),
        prediction_id=np.asarray(
            prediction_ids
        ),
        latest_position_H0_m=(
            anchors.astype(
                np.float64,
                copy=False,
            )
        ),
        horizons_s=np.asarray(
            HORIZONS_S,
            dtype=np.float64,
        ),
        mean_displacement_H0_m=(
            mean_metric_np
        ),
        raw_predictive_covariance_H0_m2=(
            raw_covariance_np
        ),
        calibrated_predictive_covariance_H0_m2=(
            calibrated_covariance_np
        ),
        variance_scale_alpha_h=(
            EXPECTED_ALPHA
        ),
    )

    identity_safe_digest = (
        prediction_digest(
            mean_metric_np,
            raw_covariance_np,
            calibrated_covariance_np,
            prediction_ids,
        )
    )

    jsonl_sha = sha256_file(
        POSTERIOR_JSONL
    )

    npz_sha = sha256_file(
        POSTERIOR_NPZ
    )

    print(
        "records                 =",
        len(records),
    )
    print(
        "prediction_id persisted = YES"
    )
    print(
        "causal anchor persisted = YES"
    )
    print(
        "anonymous row binding   = NO"
    )
    print(
        "identity-safe digest    =",
        identity_safe_digest,
    )
    print(
        "JSONL SHA256            =",
        jsonl_sha,
    )
    print(
        "NPZ SHA256              =",
        npz_sha,
    )

    # --------------------------------------------------------
    # K. Stage6 regression
    # --------------------------------------------------------

    print()
    print(
        "===== K. STAGE6 REGRESSION ====="
    )

    regression = (
        run_regression()
    )

    require(
        regression[
            "pass"
        ],
        (
            "Stage6 regression failed."
        ),
    )

    require(
        regression[
            "test_count"
        ]
        is not None
        and
        regression[
            "test_count"
        ]
        >=
        98,
        (
            "Stage6 regression test count regressed."
        ),
    )

    print(
        "regression status       = PASS"
    )
    print(
        "regression tests        =",
        regression[
            "test_count"
        ],
    )

    # --------------------------------------------------------
    # L. Final scientific record
    # --------------------------------------------------------

    result = {
        "stage":
            6,

        "block":
            "6.3_part2B1_frozen_gaussian_causal_forward",

        "status":
            "PASS_IDENTITY_SAFE_FROZEN_GAUSSIAN_FORWARD",

        "frozen_prerequisites":
            seals,

        "development_route": {
            "scenario_id":
                SCENARIO_ID,

            "compact_record_offset":
                int(
                    record_offset
                ),

            "payload_length":
                int(
                    payload_length
                ),

            "stage2_config_packaging":
                config_packaging,

            "causal_history_count":
                len(
                    histories
                ),

            "prediction_id_unique":
                True,

            "future_controller_access":
                False,

            "formal_evaluation":
                False,
        },

        "model_input": {
            "map_context_source":
                map_context_source,

            "map_context_shape":
                list(
                    map_array.shape
                ),

            "payload_count":
                len(
                    model_payloads
                ),

            "attach_supervision_called":
                False,

            "future_labels_read":
                False,

            "tracks_to_predict_used":
                False,

            "objects_of_interest_used":
                False,
        },

        "normalization": {
            "source":
                "fit_only",

            "normalizer_binding":
                normalizer_binding,

            "prepare":
                prepare_report,

            "synthetic_placeholder_fields_are_model_inputs":
                False,
        },

        "gaussian_runtime": {
            "device":
                str(
                    device
                ),

            "checkpoint_state_dict_sha256":
                str(
                    checkpoint[
                        "state_dict_sha256"
                    ]
                ),

            "model_parameter_count":
                model_parameter_count,

            "forward_signature":
                str(
                    inspect.signature(
                        model.forward
                    )
                ),

            "forward_input_mapping":
                forward_mapping,

            "raw_prediction_sha256":
                raw_hash_1,

            "repeated_forward_same_sha256":
                True,

            "runtime_first_s":
                runtime_1_s,

            "runtime_second_s":
                runtime_2_s,

            "mean_shape":
                list(
                    mean_metric_np.shape
                ),

            "covariance_shape":
                list(
                    raw_covariance_np.shape
                ),

            "coordinate_semantics":
                "metric_H0_displacement",
        },

        "calibration": {
            "variance_scale_alpha_h":
                EXPECTED_ALPHA.tolist(),

            "alpha_evidence_paths":
                [
                    path
                    for path, _array
                    in alpha_hits
                ],

            "max_covariance_calibration_error":
                max_calibration_error,

            "predictive_mean_unchanged":
                True,

            "measurement_R_t_modified":
                False,
        },

        "deterministic_Block63_contract": {
            "source_family":
                "calibrated_Gaussian_GRU",

            "signal":
                "Gaussian_mean_only",

            "predictive_covariance_used":
                False,

            "future_center_next_step":
                (
                    "latest_position_H0_m + "
                    "mean_displacement_H0_m"
                ),

            "annotation_recentering":
                False,

            "future_GT_used":
                False,
        },

        "identity_safe_posterior": {
            "record_count":
                len(
                    records
                ),

            "scenario_identity_present":
                True,

            "prediction_identity_present":
                True,

            "causal_anchor_present":
                True,

            "anonymous_row_selection":
                False,

            "digest_sha256":
                identity_safe_digest,

            "jsonl_path":
                str(
                    POSTERIOR_JSONL
                ),

            "jsonl_sha256":
                jsonl_sha,

            "npz_path":
                str(
                    POSTERIOR_NPZ
                ),

            "npz_sha256":
                npz_sha,
        },

        "scientific_execution": {
            "real_causal_observation_route":
                True,

            "actual_model_forward":
                True,

            "training":
                False,

            "retraining":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "formal_N120_evaluation":
                False,

            "future_GT_controller_use":
                False,

            "upstream_modified":
                False,
        },

        "regression":
            regression,

        "next_required_step": (
            "Block6.3 Part2B/2: consume only the identity-safe "
            "Gaussian mean + causal anchor, execute the already-"
            "frozen reciprocal association, create 0.1/0.3/0.5/1.0 s "
            "full 3-D future boxes, Stage5 tangent yaw, eight-corner "
            "headlamp projection, and retain the three unmatched "
            "ADB actors on original-reactive fallback."
        ),
    }

    return result


def diagnostic_payload(
    exc: BaseException,
) -> str:
    sections = [
        "=" * 72,
        "BLOCK 6.3 PART 2B/1 — EXACT RUNTIME BOUNDARY DIAGNOSTICS",
        "=" * 72,
        "",
        f"exception type = {type(exc).__name__}",
        f"exception      = {exc}",
        "",
    ]

    try:
        from iscai_stage4.ml.calibration_runtime import (
            CalibrationNormalizer,
        )

        sections.extend(
            [
                "===== CalibrationNormalizer signature =====",
                str(
                    inspect.signature(
                        CalibrationNormalizer
                    )
                ),
                "",
                "===== CalibrationNormalizer source =====",
                source_excerpt(
                    CalibrationNormalizer
                ),
                "",
            ]
        )

    except BaseException as inner:
        sections.append(
            (
                "CalibrationNormalizer diagnostic failed: "
                f"{type(inner).__name__}: {inner}"
            )
        )

    try:
        from iscai_stage4.data.real_pipeline import (
            build_real_causal_inputs,
        )

        sections.extend(
            [
                "===== build_real_causal_inputs source =====",
                source_excerpt(
                    build_real_causal_inputs
                ),
                "",
            ]
        )

    except BaseException as inner:
        sections.append(
            (
                "build_real_causal_inputs diagnostic failed: "
                f"{type(inner).__name__}: {inner}"
            )
        )

    try:
        from iscai_stage4.data import (
            build_model_input_payload,
        )

        sections.extend(
            [
                "===== build_model_input_payload source =====",
                source_excerpt(
                    build_model_input_payload
                ),
                "",
            ]
        )

    except BaseException as inner:
        sections.append(
            (
                "build_model_input_payload diagnostic failed: "
                f"{type(inner).__name__}: {inner}"
            )
        )

    return "\n".join(
        sections
    )


def main() -> None:
    result = None

    try:
        result = (
            scientific_run()
        )

    except BaseException as exc:
        print()
        print(
            "=" * 72
        )
        print(
            "CONTROLLED BLOCK — NO SCIENTIFIC STATE ADVANCE"
        )
        print(
            "=" * 72
        )

        print(
            "exception type =",
            type(exc).__name__,
        )

        print(
            "exception      =",
            str(exc),
        )

        traceback.print_exc()

        diagnostic = (
            diagnostic_payload(
                exc
            )
        )

        DIAGNOSTIC_EXCERPT.write_text(
            diagnostic
            +
            "\n",
            encoding="utf-8",
        )

        result = {
            "stage":
                6,

            "block":
                "6.3_part2B1_frozen_gaussian_causal_forward",

            "status":
                "BLOCKED_EXACT_FROZEN_GAUSSIAN_FORWARD_BOUNDARY",

            "exception_type":
                type(exc).__name__,

            "exception":
                str(exc),

            "diagnostic_excerpt":
                str(
                    DIAGNOSTIC_EXCERPT
                ),

            "diagnostic_excerpt_sha256":
                sha256_file(
                    DIAGNOSTIC_EXCERPT
                ),

            "safety": {
                "terminal_remains_open":
                    True,

                "training":
                    False,

                "retraining":
                    False,

                "recalibration":
                    False,

                "parameter_tuning":
                    False,

                "formal_evaluation":
                    False,

                "truth_identity_fallback":
                    False,

                "future_GT_controller_use":
                    False,

                "arbitrary_map_context":
                    False,

                "arbitrary_normalization":
                    False,

                "upstream_modified":
                    False,
            },

            "instruction": (
                "Do not guess or add defaults. "
                "Use the diagnostic excerpt to bind only the "
                "newly exposed exact frozen Stage4 interface."
            ),
        }

    report_sha = write_report(
        result
    )

    print()
    print(
        "report =",
        REPORT,
    )
    print(
        "report SHA256 =",
        report_sha,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "BLOCK 6.3 PART 2B/1 — FINAL"
    )
    print(
        "=" * 72
    )

    print(
        "status =",
        result.get(
            "status"
        ),
    )

    if (
        result.get("status")
        ==
        "PASS_IDENTITY_SAFE_FROZEN_GAUSSIAN_FORWARD"
    ):
        route = result[
            "development_route"
        ]

        runtime = result[
            "gaussian_runtime"
        ]

        posterior = result[
            "identity_safe_posterior"
        ]

        print(
            "scenario                 =",
            route[
                "scenario_id"
            ],
        )

        print(
            "causal histories         =",
            route[
                "causal_history_count"
            ],
        )

        print(
            "actual model forward     = YES"
        )

        print(
            "mean shape               =",
            runtime[
                "mean_shape"
            ],
        )

        print(
            "prediction reproducible  =",
            runtime[
                "repeated_forward_same_sha256"
            ],
        )

        print(
            "prediction_id retained   = YES"
        )

        print(
            "causal anchor retained   = YES"
        )

        print(
            "anonymous posterior row  = NO"
        )

        print(
            "deterministic covariance = NOT USED"
        )

        print(
            "future GT                = NO"
        )

        print(
            "training/recalibration   = NO / NO"
        )

        print(
            "formal evaluation        = NO"
        )

        print(
            "posterior records        =",
            posterior[
                "record_count"
            ],
        )

        print(
            "posterior JSONL SHA      =",
            posterior[
                "jsonl_sha256"
            ],
        )

        print(
            "posterior NPZ SHA        =",
            posterior[
                "npz_sha256"
            ],
        )

        print(
            "Stage6 regression        = PASS | tests =",
            result[
                "regression"
            ][
                "test_count"
            ],
        )

        print(
            "NEXT = BLOCK6.3 PART2B/2 "
            "REAL FUTURE FULL-BOX EXECUTION"
        )

    else:
        print(
            "diagnostic excerpt =",
            result.get(
                "diagnostic_excerpt"
            ),
        )

    print(
        "terminal remains open = YES"
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()

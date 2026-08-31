from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import traceback

from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(
    "/home/agni/waymo"
)

STAGE1 = ROOT / "iscai_stage1"
STAGE3 = ROOT / "iscai_stage3"
STAGE4 = ROOT / "iscai_stage4"
STAGE6 = ROOT / "iscai_stage6"

REPORT = (
    STAGE6
    / "reports"
    / "block63_part2_real_causal_association.json"
)

REGRESSION_LOG = (
    STAGE6
    / "artifacts"
    / "block63"
    / "block63_part2_stage6_regression.log"
)

BINDING = (
    STAGE6
    / "configs"
    / "block63_deterministic_predictive_binding.json"
)

PART1_REPORT = (
    STAGE6
    / "reports"
    / "block63_part1_deterministic_future_box.json"
)

BLOCK62_CLOSURE = (
    STAGE6
    / "reports"
    / "block62_closure.json"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts"
    / "block38e"
    / "formal_validation_120.jsonl"
)

PILOT_MANIFEST = (
    STAGE1
    / "artifacts"
    / "stage1a"
    / "manifests"
    / "tiny_pilot_validation.jsonl"
)

MOTION = (
    ROOT
    / "data"
    / "paired_womd_lidar_v1_3_0"
    / "validation"
    / "motion"
    / "paired-from-validation.tfrecord-00000-of-00150"
)


EXPECTED = {
    "binding": (
        "03903018109e7d5037643381ff71141a5"
        "c57f003b8bdc6b9be13bb4304468f67"
    ),

    "part1_report": (
        "871c8a81e43048ff2ccf7ef1844ad543"
        "3a859ff15b944749e13be9160b08256a"
    ),

    "block62_closure": (
        "23b299b4dc6a123ce64a6ab1350b3ed9"
        "cb149c8fed176b37e9715d9935704d92"
    ),

    "formal_manifest": (
        "2208e7287ddf6439fda4597c435a9cba"
        "1d1b9d0e4c4547bc5dd92e56e8124e46"
    ),

    "pilot_manifest": (
        "2b6fd3d7e5c411455708d325fc8595af"
        "11d99f3fcba4dde621928150695ca859"
    ),
}


FROZEN_PILOT_IDS = (
    "b85e1bd6cc8e74c0",
    "4d82fec943ddaa44",
    "bbc29ed5e271f29b",
)

ALLOWED_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(
            message
        )


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            return str(
                value
            )

    return value


def write_report(
    payload: dict[str, Any],
) -> str:
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = json.dumps(
        json_safe(
            payload
        ),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )

    REPORT.write_text(
        text + "\n",
        encoding="utf-8",
    )

    return sha256_file(
        REPORT
    )


def describe_object(
    value: Any,
) -> dict[str, Any]:
    result = {
        "type": (
            f"{type(value).__module__}."
            f"{type(value).__qualname__}"
        ),
    }

    if isinstance(
        value,
        dict,
    ):
        result[
            "keys"
        ] = sorted(
            str(key)
            for key in value.keys()
        )[:50]

    elif isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        result[
            "length"
        ] = len(
            value
        )

        result[
            "item_types"
        ] = [
            (
                f"{type(item).__module__}."
                f"{type(item).__qualname__}"
            )
            for item in value[:20]
        ]

    elif dataclasses.is_dataclass(
        value
    ):
        result[
            "dataclass_fields"
        ] = [
            field.name
            for field in dataclasses.fields(
                value
            )
        ]

    return result


def load_formal_ids_only() -> set[str]:
    """
    Read ONLY scenario_id from the frozen formal manifest.

    No formal prediction, metric, score, error, outcome, label
    or performance field is inspected.
    """
    result: set[str] = set()

    with FORMAL_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            row = json.loads(
                text
            )

            require(
                "scenario_id" in row,
                (
                    "Formal manifest row "
                    f"{line_number} has no scenario_id."
                ),
            )

            sid = str(
                row[
                    "scenario_id"
                ]
            )

            require(
                bool(
                    sid
                ),
                (
                    "Empty formal scenario_id "
                    f"at row {line_number}."
                ),
            )

            result.add(
                sid
            )

    return result


def normalize_stage2_configs(
    value: Any,
):
    """
    Resolve only structural packaging of the already-frozen
    Stage-2 configs. No scientific default is invented.
    """

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        if len(
            value
        ) == 2:
            return (
                value[0],
                value[1],
                "two_element_sequence",
            )

    if isinstance(
        value,
        dict,
    ):
        exact_pairs = (
            (
                "clean_config",
                "degraded_config",
            ),
            (
                "clean",
                "degraded",
            ),
        )

        for clean_key, degraded_key in exact_pairs:
            if (
                clean_key in value
                and
                degraded_key in value
            ):
                return (
                    value[
                        clean_key
                    ],
                    value[
                        degraded_key
                    ],
                    (
                        "dict:"
                        f"{clean_key}+"
                        f"{degraded_key}"
                    ),
                )

    attribute_pairs = (
        (
            "clean_config",
            "degraded_config",
        ),
        (
            "clean",
            "degraded",
        ),
    )

    for clean_name, degraded_name in attribute_pairs:
        if (
            hasattr(
                value,
                clean_name,
            )
            and
            hasattr(
                value,
                degraded_name,
            )
        ):
            return (
                getattr(
                    value,
                    clean_name,
                ),
                getattr(
                    value,
                    degraded_name,
                ),
                (
                    "attributes:"
                    f"{clean_name}+"
                    f"{degraded_name}"
                ),
            )

    raise RuntimeError(
        "Frozen Stage2 config packaging was not one "
        "of the structurally proven forms. "
        f"Observed: {describe_object(value)}"
    )


def find_typed_instances(
    root: Any,
    target_type: type,
    *,
    max_depth: int = 5,
) -> list[Any]:
    """
    Structural traversal only.

    This avoids guessing a return-field name for
    build_real_causal_inputs().
    """
    found: list[Any] = []
    seen: set[int] = set()

    def visit(
        value: Any,
        depth: int,
    ) -> None:
        if depth > max_depth:
            return

        identifier = id(
            value
        )

        if identifier in seen:
            return

        seen.add(
            identifier
        )

        if isinstance(
            value,
            target_type,
        ):
            found.append(
                value
            )
            return

        if isinstance(
            value,
            dict,
        ):
            for child in value.values():
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

        if dataclasses.is_dataclass(
            value
        ):
            for field in dataclasses.fields(
                value
            ):
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

    for item in found:
        key = id(
            item
        )

        if key not in unique_ids:
            unique_ids.add(
                key
            )
            unique.append(
                item
            )

    return unique


def locate_exact_box_builder():
    """
    Find the source definition by AST rather than guessing
    the Stage6 module name.
    """
    import iscai_stage6.adb as adb_package

    package_root = Path(
        adb_package.__file__
    ).resolve().parent

    definitions = []

    for path in sorted(
        package_root.rglob(
            "*.py"
        )
    ):
        try:
            source = path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(
                source,
                filename=str(
                    path
                ),
            )

        except BaseException:
            continue

        found_here = False

        for node in tree.body:
            if (
                isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                )
                and
                node.name
                ==
                "build_causal_adb_actor_boxes"
            ):
                found_here = True
                break

        if not found_here:
            continue

        relative = path.relative_to(
            package_root
        )

        if relative.name == "__init__.py":
            module_name = (
                "iscai_stage6.adb"
            )
        else:
            module_name = (
                "iscai_stage6.adb."
                +
                ".".join(
                    relative.with_suffix(
                        ""
                    ).parts
                )
            )

        definitions.append(
            (
                module_name,
                path,
            )
        )

    require(
        len(
            definitions
        ) == 1,
        (
            "Expected exactly one source definition of "
            "build_causal_adb_actor_boxes, got "
            f"{definitions}"
        ),
    )

    module_name, path = (
        definitions[0]
    )

    module = importlib.import_module(
        module_name
    )

    function = getattr(
        module,
        "build_causal_adb_actor_boxes",
        None,
    )

    require(
        callable(
            function
        ),
        (
            "Discovered builder definition is "
            "not importable/callable."
        ),
    )

    signature = inspect.signature(
        function
    )

    required_parameters = {
        "scenario",
        "adapted",
        "allowed_classes",
    }

    require(
        required_parameters.issubset(
            signature.parameters.keys()
        ),
        (
            "ADB box builder signature changed. "
            f"Observed {signature}"
        ),
    )

    return (
        function,
        module_name,
        path,
        str(
            signature
        ),
    )


def finite3(
    value: Any,
    *,
    name: str,
) -> np.ndarray:
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
            f"{name} must be shape [3], "
            f"got {array.shape}."
        ),
    )

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

    return array


def unique_argmin_exact(
    values: np.ndarray,
):
    """
    Parameter-free exact unique nearest-neighbor decision.

    No epsilon, threshold, covariance gate or tunable
    acceptance radius is introduced.
    """
    minimum = np.min(
        values
    )

    indices = np.flatnonzero(
        values
        ==
        minimum
    )

    if indices.size != 1:
        return None

    return int(
        indices[
            0
        ]
    )


def reciprocal_unique_nearest_reference(
    predictor_positions: np.ndarray,
    box_positions: np.ndarray,
):
    """
    Independent reference execution of the already-frozen
    Block6.3 causal-association semantics.

    Used here only to validate the real causal boundary;
    it does not replace the scientific controller module.
    """
    require(
        predictor_positions.ndim == 2
        and
        predictor_positions.shape[1] == 3,
        "Predictor anchor matrix must be [N,3].",
    )

    require(
        box_positions.ndim == 2
        and
        box_positions.shape[1] == 3,
        "Box anchor matrix must be [M,3].",
    )

    require(
        predictor_positions.shape[0] > 0,
        "No predictor anchors available.",
    )

    require(
        box_positions.shape[0] > 0,
        "No valid box anchors available.",
    )

    delta = (
        predictor_positions[
            :,
            None,
            :,
        ]
        -
        box_positions[
            None,
            :,
            :,
        ]
    )

    distance_sq = np.sum(
        delta
        *
        delta,
        axis=2,
    )

    require(
        bool(
            np.all(
                np.isfinite(
                    distance_sq
                )
            )
        ),
        "Association distance matrix is non-finite.",
    )

    predictor_to_box = []

    for predictor_index in range(
        distance_sq.shape[
            0
        ]
    ):
        predictor_to_box.append(
            unique_argmin_exact(
                distance_sq[
                    predictor_index,
                    :,
                ]
            )
        )

    box_to_predictor = []

    for box_index in range(
        distance_sq.shape[
            1
        ]
    ):
        box_to_predictor.append(
            unique_argmin_exact(
                distance_sq[
                    :,
                    box_index,
                ]
            )
        )

    pairs = []

    for predictor_index, box_index in enumerate(
        predictor_to_box
    ):
        if box_index is None:
            continue

        reverse = box_to_predictor[
            box_index
        ]

        if reverse != predictor_index:
            continue

        pairs.append(
            (
                predictor_index,
                box_index,
                float(
                    math.sqrt(
                        float(
                            distance_sq[
                                predictor_index,
                                box_index,
                            ]
                        )
                    )
                ),
            )
        )

    matched_predictors = {
        item[0]
        for item in pairs
    }

    matched_boxes = {
        item[1]
        for item in pairs
    }

    unmatched_predictors = [
        index
        for index in range(
            predictor_positions.shape[
                0
            ]
        )
        if index not in matched_predictors
    ]

    unmatched_boxes = [
        index
        for index in range(
            box_positions.shape[
                0
            ]
        )
        if index not in matched_boxes
    ]

    return {
        "pairs":
            pairs,

        "unmatched_predictor_indices":
            unmatched_predictors,

        "unmatched_valid_box_indices":
            unmatched_boxes,

        "distance_sq":
            distance_sq,

        "predictor_to_box":
            predictor_to_box,

        "box_to_predictor":
            box_to_predictor,
    }


def run_stage6_regression():
    REGRESSION_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            env=os.environ.copy(),
            check=False,
        )

    text = REGRESSION_LOG.read_text(
        encoding="utf-8",
        errors="replace",
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        text,
    )

    test_count = (
        int(
            match.group(
                1
            )
        )
        if match
        else None
    )

    return {
        "return_code":
            int(
                process.returncode
            ),

        "test_count":
            test_count,

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
        "=" * 68
    )
    print(
        "BLOCK 6.3 PART 2A — REAL CAUSAL ROUTE + ASSOCIATION"
    )
    print(
        "=" * 68
    )

    # ========================================================
    # A. Frozen prerequisite integrity
    # ========================================================

    print()
    print(
        "===== A. FROZEN PREREQUISITES ====="
    )

    prerequisites = {
        "binding":
            BINDING,

        "part1_report":
            PART1_REPORT,

        "block62_closure":
            BLOCK62_CLOSURE,

        "formal_manifest":
            FORMAL_MANIFEST,

        "pilot_manifest":
            PILOT_MANIFEST,
    }

    prerequisite_report = {}

    for name, path in prerequisites.items():
        require(
            path.is_file(),
            (
                "Missing frozen prerequisite: "
                f"{path}"
            ),
        )

        actual = sha256_file(
            path
        )

        expected = EXPECTED[
            name
        ]

        require(
            actual
            ==
            expected,
            (
                f"{name} SHA mismatch: "
                f"{actual} != {expected}"
            ),
        )

        prerequisite_report[
            name
        ] = {
            "path":
                str(
                    path
                ),
            "sha256":
                actual,
            "pass":
                True,
        }

        print(
            f"{name:20s} = PASS"
        )

    # ========================================================
    # B. Development scenario selection
    # ========================================================

    print()
    print(
        "===== B. DEVELOPMENT-ONLY SCENARIO SELECTION ====="
    )

    formal_ids = (
        load_formal_ids_only()
    )

    require(
        len(
            formal_ids
        ) == 120,
        (
            "Frozen formal ID population "
            f"changed: {len(formal_ids)}"
        ),
    )

    from iscai_stage4.data import (
        iter_compact_motion_records,
    )

    compact_first_three = []

    iterator = (
        iter_compact_motion_records(
            MOTION
        )
    )

    for index, item in enumerate(
        iterator
    ):
        if index >= 3:
            break

        require(
            isinstance(
                item,
                tuple,
            )
            and
            len(
                item
            ) == 3,
            (
                "Unexpected compact record "
                f"shape: {type(item)}"
            ),
        )

        (
            record_offset,
            payload_length,
            scenario,
        ) = item

        compact_first_three.append(
            {
                "record_offset":
                    int(
                        record_offset
                    ),
                "payload_length":
                    int(
                        payload_length
                    ),
                "scenario":
                    scenario,
                "scenario_id":
                    str(
                        scenario.scenario_id
                    ),
            }
        )

    observed_ids = tuple(
        item[
            "scenario_id"
        ]
        for item in compact_first_three
    )

    require(
        observed_ids
        ==
        FROZEN_PILOT_IDS,
        (
            "First-three compact pilot identity "
            "changed. "
            f"Observed={observed_ids}"
        ),
    )

    selected = None

    for item in compact_first_three:
        if (
            item[
                "scenario_id"
            ]
            not in formal_ids
        ):
            selected = item
            break

    require(
        selected is not None,
        (
            "All three frozen tiny-pilot "
            "scenarios are in formal N120. "
            "No development scenario may be "
            "selected from them."
        ),
    )

    scenario = selected[
        "scenario"
    ]

    scenario_id = selected[
        "scenario_id"
    ]

    require(
        scenario_id
        not in formal_ids,
        (
            "Development scenario leaked into "
            "formal N120."
        ),
    )

    print(
        "selection policy       = "
        "FIRST_FROZEN_TINY_PILOT_NOT_IN_FORMAL_N120"
    )
    print(
        "scenario_id            =",
        scenario_id,
    )
    print(
        "compact record offset  =",
        selected[
            "record_offset"
        ],
    )
    print(
        "payload length         =",
        selected[
            "payload_length"
        ],
    )
    print(
        "formal overlap         = NO"
    )
    print(
        "formal outcomes read   = NO"
    )

    # ========================================================
    # C. Exact frozen Stage2 -> Stage4 causal route
    # ========================================================

    print()
    print(
        "===== C. FROZEN REAL CAUSAL INPUT ROUTE ====="
    )

    from iscai_stage4.data import (
        CausalSceneInputs,
    )

    from iscai_stage4.data.real_pipeline import (
        build_real_causal_inputs,
        load_frozen_stage2_configs,
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

    print(
        "Stage2 config packaging =",
        config_packaging,
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
            max_depth=5,
        )
    )

    if len(
        scene_candidates
    ) != 1:
        raise RuntimeError(
            "Expected exactly one CausalSceneInputs "
            "inside build_real_causal_inputs() result; "
            f"found {len(scene_candidates)}. "
            "Top-level result = "
            f"{describe_object(causal_result)}"
        )

    scene_inputs = (
        scene_candidates[0]
    )

    require(
        str(
            scene_inputs.scenario_id
        )
        ==
        scenario_id,
        (
            "CausalSceneInputs scenario_id "
            "does not match selected scenario."
        ),
    )

    require(
        len(
            scene_inputs.timestamps_s
        )
        == 11,
        (
            "Stage4 causal route must expose "
            "exactly 11 causal timestamps."
        ),
    )

    histories = tuple(
        scene_inputs.histories
    )

    require(
        len(
            histories
        ) > 0,
        (
            "Real causal route produced "
            "zero histories."
        ),
    )

    prediction_ids = [
        str(
            history.prediction_id
        )
        for history in histories
    ]

    require(
        all(
            prediction_ids
        ),
        (
            "At least one causal history has "
            "empty prediction_id."
        ),
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
            "Causal prediction_id values "
            "are not unique."
        ),
    )

    predictor_positions = []

    latest_indices = []

    for history_index, history in enumerate(
        histories
    ):
        predictor_positions.append(
            finite3(
                history.latest_position_H0_m,
                name=(
                    "history["
                    f"{history_index}"
                    "].latest_position_H0_m"
                ),
            )
        )

        latest_index = int(
            history.latest_observed_frame_index
        )

        require(
            latest_index
            <=
            int(
                scenario.current_time_index
            ),
            (
                "Future history index entered "
                "controller input."
            ),
        )

        latest_indices.append(
            latest_index
        )

    predictor_positions = np.stack(
        predictor_positions,
        axis=0,
    )

    print(
        "CausalSceneInputs        = PASS"
    )
    print(
        "causal timestamps        =",
        len(
            scene_inputs.timestamps_s
        ),
    )
    print(
        "causal histories         =",
        len(
            histories
        ),
    )
    print(
        "prediction_id unique     = PASS"
    )
    print(
        "latest frame max         =",
        max(
            latest_indices
        ),
    )
    print(
        "scenario current index   =",
        int(
            scenario.current_time_index
        ),
    )
    print(
        "future controller access = NO"
    )

    # ========================================================
    # D. Build current causal ADB boxes
    # ========================================================

    print()
    print(
        "===== D. CURRENT CAUSAL ADB BOX GEOMETRY ====="
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    (
        build_boxes,
        builder_module,
        builder_path,
        builder_signature,
    ) = locate_exact_box_builder()

    boxes_result = (
        build_boxes(
            scenario=scenario,
            adapted=adapted,
            allowed_classes=(
                ALLOWED_CLASSES
            ),
        )
    )

    boxes = tuple(
        boxes_result
    )

    require(
        len(
            boxes
        ) > 0,
        (
            "No current causal ADB-relevant "
            "boxes were produced."
        ),
    )

    valid_box_original_indices = []
    valid_box_positions = []
    invalid_anchor_box_indices = []

    for box_index, actor_box in enumerate(
        boxes
    ):
        object_type = str(
            actor_box.object_type
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

        require(
            str(
                actor_box.scenario_id
            )
            ==
            scenario_id,
            (
                "ADB box scenario identity "
                "mismatch."
            ),
        )

        require(
            not bool(
                actor_box.future_state_used
            ),
            (
                "Current ADB box declares "
                "future-state use."
            ),
        )

        require(
            not bool(
                actor_box.tracks_to_predict_used
            ),
            (
                "Current ADB box declares "
                "tracks_to_predict use."
            ),
        )

        require(
            not bool(
                actor_box.objects_of_interest_used
            ),
            (
                "Current ADB box declares "
                "objects_of_interest use."
            ),
        )

        anchor = (
            actor_box
            .stage1_anchor_center_H0_m
        )

        if anchor is None:
            invalid_anchor_box_indices.append(
                box_index
            )
            continue

        try:
            anchor_array = finite3(
                anchor,
                name=(
                    "ADB box "
                    f"{box_index} "
                    "stage1_anchor_center_H0_m"
                ),
            )

        except BaseException:
            invalid_anchor_box_indices.append(
                box_index
            )
            continue

        valid_box_original_indices.append(
            box_index
        )

        valid_box_positions.append(
            anchor_array
        )

    require(
        len(
            valid_box_positions
        ) > 0,
        (
            "No ADB boxes have a finite "
            "current causal H0 anchor."
        ),
    )

    box_positions = np.stack(
        valid_box_positions,
        axis=0,
    )

    print(
        "builder module           =",
        builder_module,
    )
    print(
        "builder source           =",
        builder_path,
    )
    print(
        "builder signature        =",
        builder_signature,
    )
    print(
        "ADB-relevant boxes       =",
        len(
            boxes
        ),
    )
    print(
        "finite current anchors   =",
        len(
            valid_box_positions
        ),
    )
    print(
        "invalid anchor boxes     =",
        len(
            invalid_anchor_box_indices
        ),
    )
    print(
        "future box state used    = NO"
    )

    # ========================================================
    # E. Frozen causal-association semantics
    # ========================================================

    print()
    print(
        "===== E. PARAMETER-FREE CAUSAL ASSOCIATION ====="
    )

    association = (
        reciprocal_unique_nearest_reference(
            predictor_positions,
            box_positions,
        )
    )

    pairs_local = association[
        "pairs"
    ]

    require(
        len(
            pairs_local
        ) > 0,
        (
            "Zero reciprocal causal-anchor "
            "matches on the frozen development "
            "scenario. Stage6 must block rather "
            "than invent/tune a distance gate."
        ),
    )

    pair_records = []

    matched_original_box_indices = set()

    for (
        predictor_index,
        valid_box_index,
        distance_m,
    ) in pairs_local:
        original_box_index = (
            valid_box_original_indices[
                valid_box_index
            ]
        )

        matched_original_box_indices.add(
            original_box_index
        )

        pair_records.append(
            {
                "predictor_index":
                    int(
                        predictor_index
                    ),

                "adb_box_index":
                    int(
                        original_box_index
                    ),

                "anchor_distance_m":
                    float(
                        distance_m
                    ),
            }
        )

    unmatched_original_box_indices = [
        box_index
        for box_index in range(
            len(
                boxes
            )
        )
        if box_index
        not in matched_original_box_indices
    ]

    unmatched_predictor_indices = (
        association[
            "unmatched_predictor_indices"
        ]
    )

    distances = np.asarray(
        [
            item[
                "anchor_distance_m"
            ]
            for item in pair_records
        ],
        dtype=np.float64,
    )

    fallback_count = len(
        unmatched_original_box_indices
    )

    fallback_fraction = (
        float(
            fallback_count
            /
            len(
                boxes
            )
        )
        if boxes
        else 0.0
    )

    require(
        fallback_count
        ==
        (
            len(
                boxes
            )
            -
            len(
                matched_original_box_indices
            )
        ),
        (
            "Reactive fallback accounting "
            "is inconsistent."
        ),
    )

    print(
        "predictor histories      =",
        predictor_positions.shape[
            0
        ],
    )
    print(
        "ADB boxes                =",
        len(
            boxes
        ),
    )
    print(
        "reciprocal matches       =",
        len(
            pair_records
        ),
    )
    print(
        "unmatched predictors     =",
        len(
            unmatched_predictor_indices
        ),
    )
    print(
        "reactive fallback boxes  =",
        fallback_count,
    )
    print(
        "reactive fallback frac   =",
        fallback_fraction,
    )
    print(
        "matched distance min m   =",
        float(
            np.min(
                distances
            )
        ),
    )
    print(
        "matched distance mean m  =",
        float(
            np.mean(
                distances
            )
        ),
    )
    print(
        "matched distance max m   =",
        float(
            np.max(
                distances
            )
        ),
    )
    print(
        "distance threshold       = NONE"
    )
    print(
        "covariance gate          = NONE"
    )
    print(
        "class gate               = NONE"
    )
    print(
        "forced assignment        = NO"
    )
    print(
        "perfect-ID comparison    = NO"
    )
    print(
        "truth association        = NO"
    )
    print(
        "unmatched box policy     = ORIGINAL_REACTIVE_ADB_FALLBACK"
    )

    # ========================================================
    # F. Stage6 non-regression
    # ========================================================

    print()
    print(
        "===== F. STAGE6 REGRESSION ====="
    )

    regression = (
        run_stage6_regression()
    )

    print(
        "regression return code   =",
        regression[
            "return_code"
        ],
    )
    print(
        "regression test count    =",
        regression[
            "test_count"
        ],
    )
    print(
        "regression status        =",
        (
            "PASS"
            if regression[
                "pass"
            ]
            else
            "FAIL"
        ),
    )

    require(
        regression[
            "pass"
        ],
        (
            "Stage6 regression failed. "
            "Scientific state is not advanced."
        ),
    )

    # ========================================================
    # G. Final report
    # ========================================================

    result = {
        "stage":
            6,

        "block":
            "6.3_part2A_real_causal_association",

        "status":
            "PASS_REAL_CAUSAL_ASSOCIATION_EXECUTED",

        "purpose": (
            "Execute the frozen real causal observation-to-history "
            "boundary on a development-only scenario and validate "
            "the frozen parameter-free current-anchor association "
            "before Gaussian model forward."
        ),

        "pdf_alignment": {
            "stage6_sequence":
                (
                    "deterministic predictive full-box baseline "
                    "before probabilistic occupancy and "
                    "class-aware predictive ADB"
                ),

            "deterministic_predictive_adb_required":
                True,

            "probabilistic_masks_not_claimed_complete":
                True,

            "class_aware_policies_not_claimed_complete":
                True,

            "stage6_completion_not_claimed":
                True,
        },

        "frozen_prerequisites":
            prerequisite_report,

        "development_selection": {
            "policy":
                (
                    "first frozen Stage1 tiny-pilot scenario "
                    "not present in frozen formal N120"
                ),

            "frozen_pilot_ids":
                list(
                    FROZEN_PILOT_IDS
                ),

            "selected_scenario_id":
                scenario_id,

            "compact_record_offset":
                selected[
                    "record_offset"
                ],

            "payload_length":
                selected[
                    "payload_length"
                ],

            "formal_population_size":
                len(
                    formal_ids
                ),

            "selected_in_formal_population":
                False,

            "formal_ids_read_only_for_exclusion":
                True,

            "formal_outcome_metrics_read":
                False,

            "performance_based_selection":
                False,

            "class_based_selection":
                False,
        },

        "real_causal_route": {
            "source":
                (
                    "frozen Stage2 PC-FMCW-like clean/degraded "
                    "observation configuration through "
                    "Stage4 build_real_causal_inputs"
                ),

            "config_packaging":
                config_packaging,

            "scenario_id":
                str(
                    scene_inputs.scenario_id
                ),

            "causal_timestamp_count":
                len(
                    scene_inputs.timestamps_s
                ),

            "causal_history_count":
                len(
                    histories
                ),

            "prediction_ids_nonempty":
                True,

            "prediction_ids_unique":
                True,

            "causal_anchor_field":
                "latest_position_H0_m",

            "latest_observed_frame_max":
                max(
                    latest_indices
                ),

            "scenario_current_time_index":
                int(
                    scenario.current_time_index
                ),

            "future_state_access":
                False,

            "tracks_to_predict_controller_use":
                False,

            "objects_of_interest_controller_use":
                False,

            "truth_id_controller_use":
                False,
        },

        "adb_current_geometry": {
            "mode":
                (
                    "current_causal_annotation_assisted_box_"
                    "shape_orientation"
                ),

            "sensor_to_track_claim":
                False,

            "box_builder_module":
                builder_module,

            "box_builder_source":
                str(
                    builder_path
                ),

            "box_builder_signature":
                builder_signature,

            "allowed_classes":
                list(
                    ALLOWED_CLASSES
                ),

            "total_boxes":
                len(
                    boxes
                ),

            "finite_anchor_boxes":
                len(
                    valid_box_positions
                ),

            "invalid_anchor_boxes":
                len(
                    invalid_anchor_box_indices
                ),

            "future_box_state_used":
                False,
        },

        "association": {
            "policy":
                (
                    "reciprocal_unique_nearest_neighbor_"
                    "current_anchor_H0_3D"
                ),

            "reference_execution_only":
                True,

            "controller_binding_source":
                str(
                    BINDING
                ),

            "input_predictor_field":
                "CausalTrackHistory.latest_position_H0_m",

            "input_box_field":
                "CausalADBActorBox.stage1_anchor_center_H0_m",

            "distance":
                "Euclidean_3D_H0",

            "hard_distance_threshold":
                None,

            "covariance_gate":
                False,

            "pre_association_class_gate":
                False,

            "forced_global_assignment":
                False,

            "exact_distance_ties":
                "unmatched",

            "free_numeric_hyperparameters":
                0,

            "prediction_id_to_box_id_equality_used":
                False,

            "box_track_id_compared":
                False,

            "truth_track_index_used":
                False,

            "truth_id_used":
                False,

            "future_GT_used":
                False,

            "match_count":
                len(
                    pair_records
                ),

            "unmatched_predictor_count":
                len(
                    unmatched_predictor_indices
                ),

            "unmatched_box_count":
                fallback_count,

            "reactive_fallback_count":
                fallback_count,

            "reactive_fallback_fraction":
                fallback_fraction,

            "unmatched_box_policy":
                "ORIGINAL_REACTIVE_ADB_FALLBACK",

            "unmatched_predictor_policy":
                "IGNORE_FOR_ADB_BOX_CONTROL",

            "matched_anchor_distance_m": {
                "minimum":
                    float(
                        np.min(
                            distances
                        )
                    ),

                "mean":
                    float(
                        np.mean(
                            distances
                        )
                    ),

                "maximum":
                    float(
                        np.max(
                            distances
                        )
                    ),
            },

            "pairs_index_only":
                pair_records,
        },

        "deterministic_predictive_binding": {
            "trajectory_family":
                "calibrated_Gaussian_GRU",

            "future_signal":
                "Gaussian_mean_only",

            "predictive_covariance_used":
                False,

            "future_center_semantics":
                (
                    "predictor_latest_position_H0_m + "
                    "metric_H0_mean_displacement"
                ),

            "annotation_box_recentering":
                False,

            "future_yaw":
                (
                    "frozen_Stage5_predicted_tangent_"
                    "with_low_speed_carry_forward"
                ),

            "future_dimensions":
                (
                    "current_causal_L_W_H_held_explicit_"
                    "across_forecast_horizons"
                ),

            "full_box_projection":
                True,

            "centroid_only_projection":
                False,

            "horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],
        },

        "scientific_execution": {
            "dataset_access":
                True,

            "dataset_scope":
                "one_frozen_development_pilot_record",

            "stage2_observation_execution":
                True,

            "stage4_causal_history_materialization":
                True,

            "model_forward":
                False,

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

            "formal_results_used_for_parameter_selection":
                False,

            "upstream_modified":
                False,
        },

        "regression":
            regression,

        "next_required_step":
            (
                "Block6.3 Part2B: execute the frozen Stage4 "
                "calibrated Gaussian model on these identity-"
                "preserving causal histories; retain prediction_id "
                "and latest_position_H0_m in memory; use Gaussian "
                "mean only for deterministic future full boxes."
            ),
    }

    return result


def main() -> None:
    report = None

    try:
        report = (
            scientific_run()
        )

    except BaseException as exc:
        print()
        print(
            "=" * 68
        )
        print(
            "CONTROLLED BLOCK — NO SCIENTIFIC STATE ADVANCE"
        )
        print(
            "=" * 68
        )

        print(
            "exception type =",
            type(
                exc
            ).__name__,
        )

        print(
            "exception      =",
            str(
                exc
            ),
        )

        traceback.print_exc()

        report = {
            "stage":
                6,

            "block":
                "6.3_part2A_real_causal_association",

            "status":
                "BLOCKED_REAL_CAUSAL_ROUTE_OR_ASSOCIATION",

            "exception_type":
                type(
                    exc
                ).__name__,

            "exception":
                str(
                    exc
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

                "upstream_modified":
                    False,

                "no_arbitrary_association_threshold_added":
                    True,

                "no_truth_identity_fallback_added":
                    True,
            },

            "instruction": (
                "Do not repair by guessing. "
                "Inspect only the newly exposed exact runtime "
                "boundary and continue from there."
            ),
        }

    report_sha = write_report(
        report
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
        "=" * 68
    )
    print(
        "BLOCK 6.3 PART 2A — FINAL"
    )
    print(
        "=" * 68
    )

    print(
        "status =",
        report.get(
            "status"
        ),
    )

    if (
        report.get(
            "status"
        )
        ==
        "PASS_REAL_CAUSAL_ASSOCIATION_EXECUTED"
    ):
        development = report[
            "development_selection"
        ]

        route = report[
            "real_causal_route"
        ]

        association = report[
            "association"
        ]

        execution = report[
            "scientific_execution"
        ]

        regression = report[
            "regression"
        ]

        print(
            "development scenario     =",
            development[
                "selected_scenario_id"
            ],
        )

        print(
            "formal overlap           =",
            development[
                "selected_in_formal_population"
            ],
        )

        print(
            "causal histories         =",
            route[
                "causal_history_count"
            ],
        )

        print(
            "reciprocal matches       =",
            association[
                "match_count"
            ],
        )

        print(
            "reactive fallback boxes  =",
            association[
                "reactive_fallback_count"
            ],
        )

        print(
            "distance threshold       =",
            association[
                "hard_distance_threshold"
            ],
        )

        print(
            "truth/perfect ID         = NO"
        )

        print(
            "future GT                = NO"
        )

        print(
            "model forward            =",
            execution[
                "model_forward"
            ],
        )

        print(
            "training/recalibration   = NO / NO"
        )

        print(
            "formal evaluation        =",
            execution[
                "formal_evaluation"
            ],
        )

        print(
            "Stage6 regression        =",
            (
                "PASS"
                if regression[
                    "pass"
                ]
                else
                "FAIL"
            ),
            "| tests =",
            regression[
                "test_count"
            ],
        )

        print(
            "NEXT = BLOCK6.3 PART2B "
            "FROZEN GAUSSIAN MODEL FORWARD"
        )

    print(
        "terminal remains open     = YES"
    )
    print(
        "=" * 68
    )


if __name__ == "__main__":
    main()

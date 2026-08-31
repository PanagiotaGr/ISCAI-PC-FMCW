from __future__ import annotations

from dataclasses import (
    fields,
    is_dataclass,
)
from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S0 = ROOT / "iscai_stage0"
S6 = ROOT / "iscai_stage6"

SCENARIO_ID = "b85e1bd6cc8e74c0"

MOTION = (
    ROOT
    / "data"
    / "paired_womd_lidar_v1_3_0"
    / "validation"
    / "motion"
    / "paired-from-validation.tfrecord-00000-of-00150"
)

PART2A_REPORT = (
    S6
    / "reports"
    / "block63_part2_real_causal_association.json"
)

PART2B1_REPORT = (
    S6
    / "reports"
    / "block63_part2b1_frozen_gaussian_forward.json"
)

BINDING = (
    S6
    / "configs"
    / "block63_deterministic_predictive_binding.json"
)

PART1_REPORT = (
    S6
    / "reports"
    / "block63_part1_deterministic_future_box.json"
)

REPORT = (
    S6
    / "reports"
    / "block63_part2b2_real_future_full_box.json"
)

ARTIFACT = (
    S6
    / "artifacts"
    / "block63"
    / "block63_part2b2_real_future_full_box.json"
)


EXPECTED = {
    "part2a":
        (
            "71c9d03cc9c1b33dbb371dccd7140d5226494554"
            "b6b21df7fbfa19e9f43ba71a"
        ),

    "part2b1":
        (
            "54a4c99394d0fe29e4e7e480deacaefd07682b4f"
            "a1f1f8d8555a051a5efdabaf"
        ),

    "binding":
        (
            "03903018109e7d5037643381ff71141a5c57f003"
            "b8bdc6b9be13bb4304468f67"
        ),

    "part1":
        (
            "871c8a81e43048ff2ccf7ef1844ad5433a859ff1"
            "5b944749e13be9160b08256a"
        ),

    "posterior_jsonl":
        (
            "318b22dfc3788195a1432169075d1c8654f0d183"
            "337711918db2303b5668f5ea"
        ),

    "posterior_npz":
        (
            "2d35613bfb0bdc7133c5b3e96b4ab28e9cca643"
            "40442f9a2ab1bcf9db3aad914"
        ),
}


class ScientificBlock(
    RuntimeError
):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(condition):
        raise ScientificBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as handle:
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


def canonical_sha256(
    value: Any,
) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )

    return sha256(
        payload
    ).hexdigest()


def json_safe(
    value: Any,
) -> Any:
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
        dict,
    ):
        return {
            str(key):
                json_safe(
                    child
                )
            for key, child
            in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            json_safe(
                child
            )
            for child
            in value
        ]

    return value


def find_artifact_by_sha(
    root: Path,
    *,
    suffix: str,
    expected_sha: str,
) -> Path:
    matches = []

    for path in root.rglob(
        f"*{suffix}"
    ):
        if not path.is_file():
            continue

        try:
            actual = sha256_file(
                path
            )

        except BaseException:
            continue

        if actual == expected_sha:
            matches.append(
                path
            )

    require(
        len(
            matches
        )
        ==
        1,
        (
            "Exact posterior artifact discovery "
            f"for SHA {expected_sha} returned "
            f"{len(matches)} matches: "
            f"{matches}"
        ),
    )

    return matches[
        0
    ]


def finite_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        array.shape
        ==
        shape,
        (
            f"{name} shape mismatch: "
            f"expected={shape}, "
            f"actual={array.shape}"
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


def tuple3(
    value: Any,
) -> tuple[
    float,
    float,
    float,
]:
    array = finite_array(
        value,
        shape=(
            3,
        ),
        name="anchor_position_H0_m",
    )

    return tuple(
        float(x)
        for x in array
    )


def tuple4x3(
    value: Any,
) -> tuple[
    tuple[
        float,
        float,
        float,
    ],
    ...,
]:
    array = finite_array(
        value,
        shape=(
            4,
            3,
        ),
        name="mean_displacement_H0_m",
    )

    return tuple(
        tuple(
            float(x)
            for x in row
        )
        for row in array
    )


def construct_prediction(
    cls,
    *,
    prediction_id: str,
    anchor_position_H0_m,
    mean_displacement_H0_m,
    horizons_s,
):
    """
    Bind only deterministic mean information.

    No covariance field is permitted to be supplied.
    Unknown required constructor fields block execution.
    """

    signature = inspect.signature(
        cls
    )

    kwargs = {}
    semantic_binding = {}

    for parameter in (
        signature
        .parameters
        .values()
    ):
        name = parameter.name
        lower = name.lower()

        if name == "prediction_id":
            kwargs[
                name
            ] = prediction_id

            semantic_binding[
                name
            ] = "prediction_id"

            continue

        if (
            "covariance"
            in lower
            or
            lower.startswith(
                "cov"
            )
        ):
            require(
                parameter.default
                is not
                inspect._empty,
                (
                    "DeterministicSharedMeanPrediction "
                    "unexpectedly requires covariance: "
                    f"{name}"
                ),
            )

            continue

        if (
            "anchor"
            in lower
            and
            (
                "position"
                in lower
                or
                "h0"
                in lower
            )
        ):
            kwargs[
                name
            ] = anchor_position_H0_m

            semantic_binding[
                name
            ] = "anchor_position_H0_m"

            continue

        if (
            "mean"
            in lower
            and
            (
                "displacement"
                in lower
                or
                "trajectory"
                in lower
            )
        ):
            kwargs[
                name
            ] = mean_displacement_H0_m

            semantic_binding[
                name
            ] = "mean_displacement_H0_m"

            continue

        if (
            "horizon"
            in lower
            and
            lower.endswith(
                "_s"
            )
        ):
            kwargs[
                name
            ] = tuple(
                float(x)
                for x in horizons_s
            )

            semantic_binding[
                name
            ] = "HORIZONS_S"

            continue

        if (
            parameter.default
            is
            inspect._empty
        ):
            raise ScientificBlock(
                "BLOCKED_EXACT_PREDICTION_CONSTRUCTOR_BINDING: "
                "unresolved required field "
                f"{name!r}; signature={signature}"
            )

    require(
        any(
            value
            ==
            "prediction_id"
            for value in
            semantic_binding.values()
        ),
        (
            "prediction_id was not bound "
            "into deterministic prediction."
        ),
    )

    require(
        any(
            value
            ==
            "anchor_position_H0_m"
            for value in
            semantic_binding.values()
        ),
        (
            "causal anchor was not bound "
            "into deterministic prediction."
        ),
    )

    require(
        any(
            value
            ==
            "mean_displacement_H0_m"
            for value in
            semantic_binding.values()
        ),
        (
            "Gaussian mean displacement was "
            "not bound into deterministic prediction."
        ),
    )

    require(
        all(
            "cov"
            not in key.lower()
            for key in kwargs
        ),
        (
            "Predictive covariance entered "
            "Block6.3 prediction kwargs."
        ),
    )

    prediction = cls(
        **kwargs
    )

    return (
        prediction,
        {
            "signature":
                str(
                    signature
                ),

            "semantic_binding":
                semantic_binding,

            "covariance_kwargs":
                [],
        },
    )


def future_sequence(
    matched_forecast: Any,
) -> tuple[
    str,
    tuple,
]:
    """
    Find the unique four-item future-box sequence from the
    frozen matched-forecast dataclass/object.
    """

    candidates = []

    names = []

    if is_dataclass(
        matched_forecast
    ):
        names = [
            field.name
            for field
            in fields(
                matched_forecast
            )
        ]

    elif hasattr(
        matched_forecast,
        "__dict__",
    ):
        names = list(
            vars(
                matched_forecast
            ).keys()
        )

    for name in names:
        try:
            value = getattr(
                matched_forecast,
                name,
            )

        except BaseException:
            continue

        if not isinstance(
            value,
            (
                tuple,
                list,
            ),
        ):
            continue

        if len(
            value
        ) != 4:
            continue

        if all(
            hasattr(
                item,
                "box",
            )
            and
            hasattr(
                item,
                "projection",
            )
            for item
            in value
        ):
            candidates.append(
                (
                    name,
                    tuple(
                        value
                    ),
                )
            )

    require(
        len(
            candidates
        )
        ==
        1,
        (
            "Could not uniquely bind matched "
            "future-box sequence. candidates="
            f"{[name for name, _ in candidates]}"
        ),
    )

    return candidates[
        0
    ]


def collect_strings(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> set[str]:
    if seen is None:
        seen = set()

    if depth > 5:
        return set()

    if isinstance(
        value,
        str,
    ):
        return {
            value
        }

    object_id = id(
        value
    )

    if object_id in seen:
        return set()

    seen.add(
        object_id
    )

    result = set()

    if isinstance(
        value,
        dict,
    ):
        for child in value.values():
            result.update(
                collect_strings(
                    child,
                    depth=(
                        depth + 1
                    ),
                    seen=seen,
                )
            )

        return result

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for child in value:
            result.update(
                collect_strings(
                    child,
                    depth=(
                        depth + 1
                    ),
                    seen=seen,
                )
            )

        return result

    if is_dataclass(
        value
    ):
        for field in fields(
            value
        ):
            try:
                child = getattr(
                    value,
                    field.name,
                )

            except BaseException:
                continue

            result.update(
                collect_strings(
                    child,
                    depth=(
                        depth + 1
                    ),
                    seen=seen,
                )
            )

        return result

    if hasattr(
        value,
        "__dict__",
    ):
        for child in vars(
            value
        ).values():
            result.update(
                collect_strings(
                    child,
                    depth=(
                        depth + 1
                    ),
                    seen=seen,
                )
            )

    return result


def run_regression():
    command = [
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
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(
            S6
        ),
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    count = (
        int(
            match.group(
                1
            )
        )
        if match
        else None
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "test_count":
            count,

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -40:
                ]
            ),
    }


def scientific_run():
    print(
        "=" * 72
    )
    print(
        "BLOCK 6.3 PART 2B/2 — REAL FUTURE FULL-BOX EXECUTION"
    )
    print(
        "=" * 72
    )

    # ========================================================
    # A. Frozen prerequisite seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN PREREQUISITE SEAL ====="
    )

    for path, expected, name in (
        (
            PART2A_REPORT,
            EXPECTED[
                "part2a"
            ],
            "Part2A",
        ),
        (
            PART2B1_REPORT,
            EXPECTED[
                "part2b1"
            ],
            "Part2B/1",
        ),
        (
            BINDING,
            EXPECTED[
                "binding"
            ],
            "binding",
        ),
        (
            PART1_REPORT,
            EXPECTED[
                "part1"
            ],
            "Part1",
        ),
    ):
        require(
            path.is_file(),
            f"Missing {name}: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            expected,
            (
                f"{name} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{name:12s} = EXACT PASS"
        )

    part2a = json.loads(
        PART2A_REPORT.read_text(
            encoding="utf-8"
        )
    )

    part2b1 = json.loads(
        PART2B1_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        part2a.get(
            "status"
        )
        ==
        "PASS_REAL_CAUSAL_ASSOCIATION_EXECUTED",
        (
            "Part2A status changed."
        ),
    )

    require(
        part2b1.get(
            "status"
        )
        ==
        "PASS_IDENTITY_SAFE_FROZEN_GAUSSIAN_FORWARD",
        (
            "Part2B/1 status changed."
        ),
    )

    print(
        "Part2A status  = PASS"
    )

    print(
        "Part2B/1 status= PASS"
    )

    # ========================================================
    # B. Exact identity-safe posterior
    # ========================================================

    print()
    print(
        "===== B. IDENTITY-SAFE POSTERIOR LOAD ====="
    )

    jsonl_path = find_artifact_by_sha(
        S6
        / "artifacts"
        / "block63",
        suffix=".jsonl",
        expected_sha=(
            EXPECTED[
                "posterior_jsonl"
            ]
        ),
    )

    npz_path = find_artifact_by_sha(
        S6
        / "artifacts"
        / "block63",
        suffix=".npz",
        expected_sha=(
            EXPECTED[
                "posterior_npz"
            ]
        ),
    )

    lines = [
        line
        for line in
        jsonl_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    records = tuple(
        json.loads(
            line
        )
        for line in lines
    )

    require(
        len(
            records
        )
        ==
        27,
        (
            "Identity-safe posterior "
            "record count changed."
        ),
    )

    required_record_keys = {
        "scenario_id",
        "prediction_id",
        "anchor_position_H0_m",
        "mean_displacement_H0_m",
    }

    for index, record in enumerate(
        records
    ):
        require(
            required_record_keys
            .issubset(
                record.keys()
            ),
            (
                "Identity-safe posterior schema "
                f"changed at row {index}; "
                f"required={sorted(required_record_keys)}, "
                f"actual={sorted(record.keys())}"
            ),
        )

        require(
            str(
                record[
                    "scenario_id"
                ]
            )
            ==
            SCENARIO_ID,
            (
                "Posterior scenario identity "
                f"changed at row {index}."
            ),
        )

        require(
            str(
                record[
                    "prediction_id"
                ]
            ),
            (
                "Empty prediction_id."
            ),
        )

        finite_array(
            record[
                "anchor_position_H0_m"
            ],
            shape=(
                3,
            ),
            name=(
                "anchor_position_H0_m"
            ),
        )

        finite_array(
            record[
                "mean_displacement_H0_m"
            ],
            shape=(
                4,
                3,
            ),
            name=(
                "mean_displacement_H0_m"
            ),
        )

    prediction_ids = tuple(
        str(
            record[
                "prediction_id"
            ]
        )
        for record
        in records
    )

    require(
        len(
            set(
                prediction_ids
            )
        )
        ==
        27,
        (
            "prediction_id uniqueness changed."
        ),
    )

    print(
        "JSONL =",
        jsonl_path,
    )

    print(
        "JSONL SHA =",
        sha256_file(
            jsonl_path
        ),
    )

    print(
        "NPZ SHA   =",
        sha256_file(
            npz_path
        ),
    )

    print(
        "records   =",
        len(
            records
        ),
    )

    print(
        "prediction_id unique = PASS"
    )

    print(
        "anonymous row binding = NO"
    )

    # ========================================================
    # C. Same real causal scenario + ADB boxes
    # ========================================================

    print()
    print(
        "===== C. REAL CAUSAL ADB BOXES ====="
    )

    from iscai_stage0.womd_proto_io import (
        read_first_scenario,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    require(
        MOTION.is_file(),
        (
            f"Missing motion shard: {MOTION}"
        ),
    )

    scenario = read_first_scenario(
        MOTION
    )

    require(
        str(
            scenario.scenario_id
        )
        ==
        SCENARIO_ID,
        (
            "First compact scenario changed."
        ),
    )

    require(
        int(
            scenario.current_time_index
        )
        ==
        10,
        (
            "Scenario current_time_index "
            "changed."
        ),
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    boxes = (
        build_causal_adb_actor_boxes(
            scenario=scenario,
            adapted=adapted,
        )
    )

    require(
        len(
            boxes
        )
        ==
        9,
        (
            "Real causal ADB box count "
            f"changed: {len(boxes)}"
        ),
    )

    for index, actor in enumerate(
        boxes
    ):
        require(
            actor.future_state_used
            is False,
            (
                f"Box {index} used future state."
            ),
        )

        require(
            actor.tracks_to_predict_used
            is False,
            (
                f"Box {index} used tracks_to_predict."
            ),
        )

        require(
            actor.objects_of_interest_used
            is False,
            (
                f"Box {index} used objects_of_interest."
            ),
        )

        require(
            actor.stage1_anchor_center_H0_m
            is not None,
            (
                f"Box {index} has no causal H0 anchor."
            ),
        )

    print(
        "scenario_id       =",
        scenario.scenario_id,
    )

    print(
        "current index     =",
        scenario.current_time_index,
    )

    print(
        "causal ADB boxes  =",
        len(
            boxes
        ),
    )

    print(
        "future box input  = NO"
    )

    print(
        "tracks_to_predict = NO"
    )

    # ========================================================
    # D. Exact deterministic mean prediction objects
    # ========================================================

    print()
    print(
        "===== D. GAUSSIAN MEAN → DETERMINISTIC PREDICTIONS ====="
    )

    from iscai_stage6.adb.deterministic_predictive import (
        HORIZONS_S,
        DeterministicSharedMeanPrediction,
        build_deterministic_predictive_plan,
    )

    require(
        tuple(
            float(x)
            for x in HORIZONS_S
        )
        ==
        (
            0.1,
            0.3,
            0.5,
            1.0,
        ),
        (
            f"Frozen horizons changed: "
            f"{HORIZONS_S}"
        ),
    )

    predictions = []

    constructor_bindings = []

    for record in records:
        prediction, binding = (
            construct_prediction(
                DeterministicSharedMeanPrediction,
                prediction_id=str(
                    record[
                        "prediction_id"
                    ]
                ),
                anchor_position_H0_m=tuple3(
                    record[
                        "anchor_position_H0_m"
                    ]
                ),
                mean_displacement_H0_m=tuple4x3(
                    record[
                        "mean_displacement_H0_m"
                    ]
                ),
                horizons_s=HORIZONS_S,
            )
        )

        predictions.append(
            prediction
        )

        constructor_bindings.append(
            binding
        )

    predictions = tuple(
        predictions
    )

    require(
        len(
            predictions
        )
        ==
        27,
        (
            "Deterministic prediction "
            "count mismatch."
        ),
    )

    signatures = {
        item[
            "signature"
        ]
        for item
        in constructor_bindings
    }

    require(
        len(
            signatures
        )
        ==
        1,
        (
            "Prediction constructor signature "
            "changed across rows."
        ),
    )

    require(
        all(
            not item[
                "covariance_kwargs"
            ]
            for item
            in constructor_bindings
        ),
        (
            "Covariance entered deterministic "
            "prediction construction."
        ),
    )

    print(
        "prediction class =",
        DeterministicSharedMeanPrediction.__name__,
    )

    print(
        "constructor      =",
        next(
            iter(
                signatures
            )
        ),
    )

    print(
        "predictions      =",
        len(
            predictions
        ),
    )

    print(
        "trajectory input = CALIBRATED GAUSSIAN MEAN ONLY"
    )

    print(
        "covariance use   = NO"
    )

    # ========================================================
    # E. Actual frozen association + predictive plan
    # ========================================================

    print()
    print(
        "===== E. REAL DETERMINISTIC PREDICTIVE PLAN ====="
    )

    plan = (
        build_deterministic_predictive_plan(
            predictions,
            boxes,
        )
    )

    matched_count = len(
        plan.matched_forecasts
    )

    fallback_indices = tuple(
        int(x)
        for x
        in plan.reactive_fallback_box_indices
    )

    unmatched_predictions = tuple(
        int(x)
        for x
        in plan.unmatched_prediction_indices
    )

    require(
        matched_count
        ==
        6,
        (
            "Frozen reciprocal association "
            f"changed: matched={matched_count}, "
            "expected=6."
        ),
    )

    require(
        len(
            fallback_indices
        )
        ==
        3,
        (
            "Reactive fallback count changed: "
            f"{fallback_indices}"
        ),
    )

    require(
        len(
            unmatched_predictions
        )
        ==
        21,
        (
            "Unmatched predictor count changed: "
            f"{len(unmatched_predictions)}"
        ),
    )

    require(
        int(
            plan.predicted_box_count
        )
        ==
        24,
        (
            "Predicted future-box count "
            f"changed: {plan.predicted_box_count}"
        ),
    )

    print(
        "matched forecasts       =",
        matched_count,
    )

    print(
        "reactive fallback boxes =",
        len(
            fallback_indices
        ),
        fallback_indices,
    )

    print(
        "unmatched predictors    =",
        len(
            unmatched_predictions
        ),
    )

    print(
        "predicted future boxes  =",
        plan.predicted_box_count,
    )

    print(
        "association threshold   = NONE"
    )

    print(
        "covariance gate         = NONE"
    )

    print(
        "class gate              = NONE"
    )

    print(
        "perfect actor ID        = NO"
    )

    # ========================================================
    # F. Full 3D future-box + projection validation
    # ========================================================

    print()
    print(
        "===== F. FULL-BOX / CORNER EXECUTION ====="
    )

    matched_summaries = []

    total_future_boxes = 0
    total_corners = 0

    all_strings = set()

    sequence_names = set()

    for forecast_index, forecast in enumerate(
        plan.matched_forecasts
    ):
        sequence_name, future_items = (
            future_sequence(
                forecast
            )
        )

        sequence_names.add(
            sequence_name
        )

        require(
            len(
                future_items
            )
            ==
            4,
            (
                "Matched forecast does not "
                "contain four horizons."
            ),
        )

        forecast_summary = {
            "matched_forecast_index":
                int(
                    forecast_index
                ),

            "future_sequence_field":
                sequence_name,

            "horizons":
                [],
        }

        for horizon_index, item in enumerate(
            future_items
        ):
            require(
                hasattr(
                    item,
                    "box",
                ),
                (
                    "Future item lacks Box3D."
                ),
            )

            require(
                hasattr(
                    item,
                    "projection",
                ),
                (
                    "Future item lacks projection."
                ),
            )

            box = item.box
            projection = item.projection

            corners = np.asarray(
                projection
                .corners
                .xyz,
                dtype=np.float64,
            )

            require(
                corners.shape
                ==
                (
                    8,
                    3,
                ),
                (
                    "Future box does not project "
                    f"8 physical corners: "
                    f"{corners.shape}"
                ),
            )

            require(
                bool(
                    np.all(
                        np.isfinite(
                            corners
                        )
                    )
                ),
                (
                    "Projected corners contain "
                    "non-finite values."
                ),
            )

            require(
                float(
                    box.length_m
                )
                >
                0.0
                and
                float(
                    box.width_m
                )
                >
                0.0
                and
                float(
                    box.height_m
                )
                >
                0.0,
                (
                    "Future box dimensions "
                    "are non-positive."
                ),
            )

            require(
                float(
                    projection.theta_span_rad
                )
                >
                0.0,
                (
                    "Future box angular span "
                    "collapsed to centroid."
                ),
            )

            require(
                float(
                    projection.ground_range_max_m
                )
                >
                float(
                    projection.ground_range_min_m
                ),
                (
                    "Future full-box ground-range "
                    "extent collapsed."
                ),
            )

            total_future_boxes += 1

            total_corners += int(
                corners.shape[
                    0
                ]
            )

            all_strings.update(
                collect_strings(
                    item
                )
            )

            forecast_summary[
                "horizons"
            ].append(
                {
                    "horizon_s":
                        float(
                            HORIZONS_S[
                                horizon_index
                            ]
                        ),

                    "center_xyz":
                        [
                            float(x)
                            for x in
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

                    "corner_count":
                        8,

                    "theta_span_rad":
                        float(
                            projection.theta_span_rad
                        ),

                    "ground_range_min_m":
                        float(
                            projection.ground_range_min_m
                        ),

                    "ground_range_max_m":
                        float(
                            projection.ground_range_max_m
                        ),

                    "corners_xyz":
                        corners.tolist(),
                }
            )

        matched_summaries.append(
            forecast_summary
        )

    require(
        total_future_boxes
        ==
        24,
        (
            "Actual full-box total changed: "
            f"{total_future_boxes}"
        ),
    )

    require(
        total_corners
        ==
        192,
        (
            "Actual projected corner total "
            f"changed: {total_corners}"
        ),
    )

    print(
        "future sequence field(s) =",
        sorted(
            sequence_names
        ),
    )

    print(
        "future full boxes        =",
        total_future_boxes,
    )

    print(
        "corners per box          = 8"
    )

    print(
        "total projected corners  =",
        total_corners,
    )

    print(
        "centroid-only projection = NO"
    )

    # Provenance is already structurally enforced by the
    # frozen deterministic_predictive implementation.
    # If the returned dataclasses persist those labels,
    # verify them explicitly as an additional runtime gate.
    persisted_predicted_sample = (
        "predicted_sample"
        in
        all_strings
    )

    persisted_predicted_tangent = (
        "predicted_tangent"
        in
        all_strings
    )

    print(
        "persisted predicted_sample =",
        persisted_predicted_sample,
    )

    print(
        "persisted predicted_tangent=",
        persisted_predicted_tangent,
    )

    print(
        "controller provenance route = "
        "FROZEN BLOCK6.3 IMPLEMENTATION"
    )

    # ========================================================
    # G. Deterministic-scope guards
    # ========================================================

    print()
    print(
        "===== G. BLOCK6.3 SCOPE GUARDS ====="
    )

    predictive_source = inspect.getsource(
        build_deterministic_predictive_plan
    )

    require(
        "future_ground_truth"
        not in
        predictive_source,
        (
            "Unexpected future_ground_truth "
            "token in deterministic plan builder."
        ),
    )

    print(
        "predictive covariance used = NO"
    )

    print(
        "probabilistic occupancy    = NOT IMPLEMENTED HERE"
    )

    print(
        "class-aware policy         = NOT IMPLEMENTED HERE"
    )

    print(
        "temporal smoothing         = NOT IMPLEMENTED HERE"
    )

    print(
        "actuation-rate limits      = NOT IMPLEMENTED HERE"
    )

    print(
        "communication codebook     = NOT REUSED"
    )

    print(
        "formal grid                = NOT FROZEN"
    )

    print(
        "future GT                  = NO"
    )

    print(
        "training/recalibration     = NO / NO"
    )

    print(
        "model forward              = NO"
    )

    print(
        "formal evaluation          = NO"
    )

    # ========================================================
    # H. Persist deterministic full-box evidence
    # ========================================================

    print()
    print(
        "===== H. ARTIFACT MATERIALIZATION ====="
    )

    ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_payload = {
        "stage":
            6,

        "block":
            "6.3-Part2B/2",

        "status":
            "PASS_REAL_FUTURE_FULL_BOX_EXECUTED",

        "scenario_id":
            SCENARIO_ID,

        "posterior": {
            "records":
                27,

            "jsonl_path":
                str(
                    jsonl_path
                ),

            "jsonl_sha256":
                EXPECTED[
                    "posterior_jsonl"
                ],

            "npz_path":
                str(
                    npz_path
                ),

            "npz_sha256":
                EXPECTED[
                    "posterior_npz"
                ],

            "trajectory_signal":
                "calibrated_Gaussian_mean_only",

            "predictive_covariance_used":
                False,
        },

        "association": {
            "policy":
                (
                    "reciprocal_unique_nearest_"
                    "current_anchor_H0"
                ),

            "matched":
                matched_count,

            "ADB_boxes":
                len(
                    boxes
                ),

            "unmatched_predictions":
                len(
                    unmatched_predictions
                ),

            "reactive_fallback_box_indices":
                list(
                    fallback_indices
                ),

            "reactive_fallback_count":
                len(
                    fallback_indices
                ),

            "hard_distance_threshold":
                None,

            "covariance_gate":
                None,

            "class_gate":
                None,

            "perfect_identity":
                False,
        },

        "future_geometry": {
            "horizons_s":
                [
                    float(x)
                    for x in
                    HORIZONS_S
                ],

            "future_box_count":
                total_future_boxes,

            "corners_per_box":
                8,

            "projected_corner_count":
                total_corners,

            "centroid_only":
                False,

            "dimensions":
                (
                    "current_causal_LWH_held_"
                    "across_deterministic_horizons"
                ),

            "heading":
                (
                    "frozen_Stage5_trajectory_"
                    "tangent_low_speed_carry_forward"
                ),

            "matched_forecasts":
                matched_summaries,
        },

        "scope": {
            "future_GT":
                False,

            "tracks_to_predict":
                False,

            "objects_of_interest":
                False,

            "predictive_covariance":
                False,

            "probabilistic_occupancy":
                False,

            "class_aware_policy":
                False,

            "communication_codebook_reused":
                False,

            "formal_evaluation":
                False,

            "training":
                False,

            "recalibration":
                False,

            "model_forward":
                False,
        },
    }

    artifact_payload[
        "scientific_digest"
    ] = canonical_sha256(
        artifact_payload
    )

    ARTIFACT.write_text(
        json.dumps(
            json_safe(
                artifact_payload
            ),
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    artifact_sha = sha256_file(
        ARTIFACT
    )

    print(
        "artifact =",
        ARTIFACT,
    )

    print(
        "artifact SHA256 =",
        artifact_sha,
    )

    print(
        "scientific digest =",
        artifact_payload[
            "scientific_digest"
        ],
    )

    # ========================================================
    # I. Stage6 full regression
    # ========================================================

    print()
    print(
        "===== I. STAGE6 REGRESSION ====="
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
            "Stage6 regression failed."
        ),
    )

    require(
        regression[
            "test_count"
        ]
        ==
        98,
        (
            "Stage6 regression test count "
            "changed: "
            f"{regression['test_count']}"
        ),
    )

    print(
        "regression status = PASS"
    )

    print(
        "regression tests  =",
        regression[
            "test_count"
        ],
    )

    # ========================================================
    # J. Closure report
    # ========================================================

    print()
    print(
        "===== J. PART2B/2 CLOSURE ====="
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.3-Part2B/2",

        "status":
            "PASS_REAL_FUTURE_FULL_BOX_EXECUTED",

        "scenario_id":
            SCENARIO_ID,

        "frozen_prerequisites": {
            "Part2A_sha256":
                EXPECTED[
                    "part2a"
                ],

            "Part2B1_sha256":
                EXPECTED[
                    "part2b1"
                ],

            "binding_sha256":
                EXPECTED[
                    "binding"
                ],

            "Part1_sha256":
                EXPECTED[
                    "part1"
                ],
        },

        "posterior": {
            "records":
                27,

            "identity_safe":
                True,

            "prediction_id_retained":
                True,

            "causal_anchor_retained":
                True,

            "anonymous_row_binding":
                False,

            "mean_only":
                True,

            "covariance_used":
                False,
        },

        "association": {
            "matched":
                matched_count,

            "ADB_boxes":
                9,

            "reactive_fallback":
                len(
                    fallback_indices
                ),

            "unmatched_predictions":
                len(
                    unmatched_predictions
                ),
        },

        "geometry": {
            "future_boxes":
                total_future_boxes,

            "corners_per_box":
                8,

            "total_projected_corners":
                total_corners,

            "full_box":
                True,

            "centroid_only":
                False,

            "horizons_s":
                [
                    float(x)
                    for x in
                    HORIZONS_S
                ],
        },

        "provenance": {
            "state_source_expected":
                "predicted_sample",

            "orientation_source_expected":
                "predicted_tangent",

            "persisted_predicted_sample":
                persisted_predicted_sample,

            "persisted_predicted_tangent":
                persisted_predicted_tangent,

            "future_GT":
                False,

            "perfect_actor_identity":
                False,
        },

        "scientific_execution": {
            "new_model_forward":
                False,

            "training":
                False,

            "recalibration":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,
        },

        "regression": {
            "status":
                "PASS",

            "tests":
                regression[
                    "test_count"
                ],
        },

        "artifact": {
            "path":
                str(
                    ARTIFACT
                ),

            "sha256":
                artifact_sha,

            "scientific_digest":
                artifact_payload[
                    "scientific_digest"
                ],
        },

        "next":
            (
                "BLOCK6.3 FINAL DETERMINISTIC "
                "PREDICTIVE BASELINE CLOSURE"
            ),
    }

    REPORT.write_text(
        json.dumps(
            json_safe(
                report_payload
            ),
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    report_sha = sha256_file(
        REPORT
    )

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
        "BLOCK 6.3 PART 2B/2 — FINAL"
    )

    print(
        "=" * 72
    )

    print(
        "status                   = "
        "PASS_REAL_FUTURE_FULL_BOX_EXECUTED"
    )

    print(
        "scenario                 =",
        SCENARIO_ID,
    )

    print(
        "posterior records        = 27"
    )

    print(
        "Gaussian signal          = MEAN ONLY"
    )

    print(
        "matched actors           =",
        matched_count,
    )

    print(
        "reactive fallback actors =",
        len(
            fallback_indices
        ),
    )

    print(
        "unmatched predictors     =",
        len(
            unmatched_predictions
        ),
    )

    print(
        "future boxes             =",
        total_future_boxes,
    )

    print(
        "corners per box          = 8"
    )

    print(
        "projected corners total  =",
        total_corners,
    )

    print(
        "centroid-only            = NO"
    )

    print(
        "predictive covariance    = NOT USED"
    )

    print(
        "future GT                = NO"
    )

    print(
        "training/recalibration   = NO / NO"
    )

    print(
        "new model forward        = NO"
    )

    print(
        "formal evaluation        = NO"
    )

    print(
        "Stage6 regression        = PASS | tests =",
        regression[
            "test_count"
        ],
    )

    print(
        "report SHA256            =",
        report_sha,
    )

    print(
        "NEXT = BLOCK6.3 FINAL DETERMINISTIC "
        "PREDICTIVE BASELINE CLOSURE"
    )

    print(
        "terminal remains open = YES"
    )

    print(
        "=" * 72
    )


def main():
    try:
        scientific_run()

    except BaseException as exc:
        print()
        print(
            "=" * 72
        )

        print(
            "CONTROLLED BLOCK — "
            "NO SCIENTIFIC STATE ADVANCE"
        )

        print(
            "=" * 72
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

        print()

        traceback.print_exc(
            limit=18
        )

        print()

        print(
            "training/recalibration = NO / NO"
        )

        print(
            "formal evaluation      = NO"
        )

        print(
            "upstream modification  = NO"
        )

        print(
            "terminal remains open  = YES"
        )

        print(
            "=" * 72
        )

    # Deliberately no sys.exit().


if __name__ == "__main__":
    main()

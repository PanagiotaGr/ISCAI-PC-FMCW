from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import numpy as np

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)
from iscai_stage4.data.real_pipeline import (
    read_training_scenario,
)
from iscai_stage6.adb.grid import (
    IlluminationGridSpec,
)
from iscai_stage6.adb.part_a_reactive import (
    part_a_original_reactive_map_from_h0_centers,
)
from iscai_stage6.adb.reactive_development_scoring import (
    hold_current_reactive_schedule,
)
from iscai_stage6.adb.womd_geometry import (
    build_causal_adb_actor_boxes,
)


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

PRED = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

FALLBACK = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_reactive_fallbacks.jsonl"
)

GRID_CONFIG = (
    S6
    / "configs/block64_mc_grid_numeric_freeze.json"
)

MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

LEDGER = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_decision_ledger.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block68_reactive_decision_ledger_freeze.json"
)

N = 120
GRID_SHAPE = (501, 301)
TESTS = 224
MIN_FREE_GIB = 250.0

CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

VEHICLE = "TYPE_VEHICLE"


EXPECTED = {
    S6 / "reports/block67_final_closure.json":
        "67d168ed7678164b8d85885e6e81f41f"
        "7206bd8f754e13ec5cc28c3a65aa20e7",

    S6 / "reports/block68_preoutcome_metric_reconciliation.json":
        "077a89fa5163f8862a3a5b7489d98929"
        "d9ae3e160ed9e030eb75f42a91d48f9f",

    S6 / "configs/stage6_metric_freeze_protocol.json":
        "409fbb2785e4ff13f29fdff96c91d608"
        "c245e009fcbb86ce57099d6a4e1815c9",

    S6 / "src/iscai_stage6/adb/metric_semantics.py":
        "bf8358b5a7edfe40ffac2718ca7cd5af"
        "6e7b5fdb3ff8a2823caad6d0f40f057a",

    S6 / "src/iscai_stage6/adb/baseline_metric_contract.py":
        "39518597079f65d556ee8f0fd3b00ce5"
        "25319623c9eb8a83f6e3c608954213e9",

    S6 / "reports/block68_exact_evaluator_binding_repair.json":
        "1505c780423efbe27735c3729b95a18c"
        "3da2093f49826bd96cbfad8fdce5d5f1",

    S6 / "reports/block68_evaluator_reference_contract_freeze.json":
        "5885424d164a058a151c50dd663acec4"
        "da7aeba66a56d5e0191c777682e113a8",

    S6 / "configs/stage6_evaluator_reference_contract.json":
        "c9ff0390820a3c37702fb391ba603244f"
        "ffcf7c03c476045b590aa483e89bd7e",

    S6 / "artifacts/block68/frozen_road_roi_definition.json":
        "832f4bc3215fe585fee781080c98b2928"
        "157f26e8a66d22f428f15604afb75e0",

    S6 / "src/iscai_stage6/adb/evaluator_reference.py":
        "2015136fdcbfe3f837bcba5a2ee05578a"
        "4d8a01c4ecfd52e4dfac9f864349891",

    S6 / "reports/block68_noninferiority_bound_rule_freeze.json":
        "289f9369735566daa2344a90cb312d4c0"
        "6ede76b73827f07e408855c4e06ce6b",

    S6 / "configs/stage6_noninferiority_bound_selection_rule.json":
        "b5c0f40725e26578a4739f0582e4467e"
        "c48b72db8bb077d13862cc83dee98d24",

    S6 / "src/iscai_stage6/adb/noninferiority_bounds.py":
        "3c67a7f152359710cb24db1a6e01c4b5"
        "80728e6480b7c5d8f9c40a41349cf1cd",

    S6 / "configs/stage6_reactive_development_scoring_contract.json":
        "454d22a2f067025fd8dde877546b7305c"
        "4357000126585c860a4a2d9ef3535d5",

    S6 / "src/iscai_stage6/adb/reactive_development_scoring.py":
        "e3114093406da9e4f8088a70cfcf9884"
        "3055634a6e45ef30e1f3f0420f586d7b",

    S6 / "reports/block68_reactive_development_scoring_freeze.json":
        "0fc8089bdfbf20eb5e187e71a6a736fe"
        "fea59b3aac6b34ff373237016d8473f9",

    S6 / "src/iscai_stage6/adb/class_aware_policy.py":
        "b998f468b98c2770c48c84a2c0aaaa21"
        "77c2d84b8fc838e80fef9b9da1ebc14b",

    S6 / "src/iscai_stage6/adb/part_a_reactive.py":
        "19f80f325aebc03b5257ca3c0408d1dd"
        "ea1ac3a8ff4ad875914224584e2ef173",

    PRED:
        "d7ce8769deee08286974b0327cdfd55c"
        "08ad7ea326cf9d6450a48a82b3e251b6",
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def array_sha(array) -> str:
    value = np.ascontiguousarray(
        np.asarray(
            array,
            dtype=np.float64,
        )
    )

    digest = sha256()

    digest.update(
        json.dumps(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    digest.update(
        value.tobytes(order="C")
    )

    return digest.hexdigest()


def exact_gate():
    for path, expected in EXPECTED.items():

        require(
            path.is_file(),
            f"Missing frozen dependency: {path}",
        )

        actual = file_sha(path)

        require(
            actual == expected,
            (
                "Frozen dependency changed:\n"
                f"path={path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )


def safe_json(path: Path):
    require(
        "formal"
        not in
        str(path.resolve()).lower(),
        (
            "Formal-named content forbidden "
            f"in this runner: {path}"
        ),
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def safe_jsonl(path: Path):
    require(
        "formal"
        not in
        str(path.resolve()).lower(),
        (
            "Formal-named content forbidden "
            f"in this runner: {path}"
        ),
    )

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, line in enumerate(
            stream,
            1,
        ):
            if not line.strip():
                continue

            try:
                value = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL {path}:{line_number}: {exc}"
                ) from exc

            require(
                isinstance(value, dict),
                (
                    "Non-object JSONL record "
                    f"{path}:{line_number}"
                ),
            )

            records.append(value)

    return records


def write_jsonl(path: Path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        for row in rows:

            stream.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )

            stream.write("\n")

        stream.flush()
        os.fsync(
            stream.fileno()
        )


def atomic_json(path: Path, value):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
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
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    return (
        process.returncode,
        count,
        process.stdout,
    )


def output_tail(text, line_count=100):
    return "\n".join(
        text.splitlines()[
            -line_count:
        ]
    )


# ============================================================
# Frozen grid binding
# ============================================================

def walk_dicts(value):

    if isinstance(value, dict):

        yield value

        for child in value.values():
            yield from walk_dicts(child)

    elif isinstance(value, list):

        for child in value:
            yield from walk_dicts(child)


def flatten_leaves(
    value,
    output=None,
):

    if output is None:
        output = defaultdict(list)

    if isinstance(value, dict):

        for key, child in value.items():

            if isinstance(
                child,
                (
                    dict,
                    list,
                ),
            ):
                flatten_leaves(
                    child,
                    output,
                )

            else:
                output[
                    str(key)
                ].append(child)

    elif isinstance(value, list):

        for child in value:
            flatten_leaves(
                child,
                output,
            )

    return output


def resolve_grid():
    """
    Load the already frozen Stage6 illumination-grid runtime
    binding.

    No heuristic grid discovery is allowed here.
    No performance outcome, future truth, P_occ content, or
    newly selected numerical parameter enters this function.
    """

    binding_path = (
        S6
        / "configs"
        / "stage6_frozen_illumination_grid_runtime_binding.json"
    )

    expected_binding_sha = "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc"

    require(
        binding_path.is_file(),
        (
            "Frozen Stage6 illumination-grid "
            f"binding missing: {binding_path}"
        ),
    )

    actual_binding_sha = file_sha(
        binding_path
    )

    require(
        actual_binding_sha
        ==
        expected_binding_sha,
        (
            "Frozen illumination-grid binding SHA changed.\n"
            f"expected={expected_binding_sha}\n"
            f"actual={actual_binding_sha}"
        ),
    )

    binding = safe_json(
        binding_path
    )

    require(
        binding.get(
            "status"
        )
        ==
        "FROZEN_STAGE6_ILLUMINATION_GRID_RUNTIME_BINDING",
        (
            "Frozen illumination-grid binding "
            "status mismatch."
        ),
    )

    require(
        binding.get(
            "scientific_parameter_change"
        )
        is False,
        (
            "Grid binding unexpectedly records "
            "scientific parameter change."
        ),
    )

    anti = binding.get(
        "anti_invention",
        {}
    )

    for key in (
        "range_min_defaulted_to_zero",
        "constructor_values_selected_from_outcomes",
        "predictive_performance_read",
        "reactive_performance_read",
        "future_truth_read",
        "formal_content_read",
    ):

        require(
            anti.get(
                key
            )
            is False,
            (
                "Frozen grid anti-invention "
                f"gate failed: {key}"
            ),
        )

    provenance = binding.get(
        "provenance",
        {}
    )

    block64_sha = provenance.get(
        "Block64_config_sha256"
    )

    require(
        isinstance(
            block64_sha,
            str,
        )
        and
        len(
            block64_sha
        )
        ==
        64,
        (
            "Frozen Block6.4 provenance SHA "
            "missing."
        ),
    )

    require(
        file_sha(
            GRID_CONFIG
        )
        ==
        block64_sha,
        (
            "Block6.4 grid source changed "
            "after runtime-binding freeze."
        ),
    )

    constructor = binding.get(
        "constructor"
    )

    require(
        isinstance(
            constructor,
            dict,
        ),
        (
            "Frozen illumination-grid constructor "
            "must be an object."
        ),
    )

    expected_keys = {
        "theta_min_rad",
        "theta_max_rad",
        "range_min_m",
        "range_max_m",
        "n_theta",
        "n_range",
    }

    require(
        set(
            constructor
        )
        ==
        expected_keys,
        (
            "Frozen constructor keys changed: "
            f"{sorted(constructor)}"
        ),
    )

    kwargs = {
        "theta_min_rad":
            float(
                constructor[
                    "theta_min_rad"
                ]
            ),

        "theta_max_rad":
            float(
                constructor[
                    "theta_max_rad"
                ]
            ),

        "range_min_m":
            float(
                constructor[
                    "range_min_m"
                ]
            ),

        "range_max_m":
            float(
                constructor[
                    "range_max_m"
                ]
            ),

        "n_theta":
            int(
                constructor[
                    "n_theta"
                ]
            ),

        "n_range":
            int(
                constructor[
                    "n_range"
                ]
            ),
    }

    require(
        (
            kwargs[
                "n_theta"
            ],
            kwargs[
                "n_range"
            ],
        )
        ==
        GRID_SHAPE,
        (
            "Frozen grid dimensions differ "
            f"from {GRID_SHAPE}: "
            f"{kwargs['n_theta']} x "
            f"{kwargs['n_range']}"
        ),
    )

    require(
        all(
            math.isfinite(
                kwargs[
                    key
                ]
            )
            for key in (
                "theta_min_rad",
                "theta_max_rad",
                "range_min_m",
                "range_max_m",
            )
        ),
        (
            "Frozen illumination-grid bounds "
            "contain non-finite values."
        ),
    )

    require(
        kwargs[
            "theta_min_rad"
        ]
        <
        kwargs[
            "theta_max_rad"
        ],
        (
            "Frozen theta bounds "
            "are not increasing."
        ),
    )

    require(
        kwargs[
            "range_min_m"
        ]
        <
        kwargs[
            "range_max_m"
        ],
        (
            "Frozen range bounds "
            "are not increasing."
        ),
    )

    grid = IlluminationGridSpec(
        **kwargs
    )

    blank = np.asarray(
        part_a_original_reactive_map_from_h0_centers(
            grid,
            (),
        ),
        dtype=np.float64,
    )

    require(
        blank.shape
        ==
        GRID_SHAPE,
        (
            "Frozen runtime grid failed "
            "Part-A map-shape readback: "
            f"{blank.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(
                blank
            )
        ),
        (
            "Blank Part-A map contains "
            "non-finite values."
        ),
    )

    require(
        np.all(
            (
                blank >= 0.0
            )
            &
            (
                blank <= 1.0
            )
        ),
        (
            "Blank Part-A map lies "
            "outside [0,1]."
        ),
    )

    attempts = [
        {
            "label":
                "frozen_stage6_runtime_grid_binding",

            "status":
                "ACCEPTED",

            "binding_path":
                str(
                    binding_path
                ),

            "binding_sha256":
                actual_binding_sha,

            "constructor":
                kwargs,

            "heuristic_discovery":
                False,

            "numeric_values_invented":
                False,

            "outcomes_used":
                False,

            "future_truth_used":
                False,
        }
    ]

    return (
        grid,
        attempts,
        str(
            inspect.signature(
                IlluminationGridSpec
            )
        ),
    )



# ============================================================
# Current-box / selector integrity
# ============================================================

def one_attr(
    value,
    names,
    label,
):

    present = [
        name
        for name in names
        if hasattr(
            value,
            name,
        )
    ]

    require(
        len(present) == 1,
        (
            f"{label}: expected exactly one "
            f"of {names}; found {present}"
        ),
    )

    return getattr(
        value,
        present[0],
    )


def box_geometry(
    actor_box,
):

    box = actor_box.box

    center = np.asarray(
        one_attr(
            box,
            (
                "center_H0_m",
                "center_xyz_H0_m",
                "center_m",
                "center",
            ),
            "Box center",
        ),
        dtype=np.float64,
    )

    require(
        center.shape == (3,),
        "Causal Box3D center is not 3-D.",
    )

    result = {
        "center_xyz_H0_m":
            [
                float(value)
                for value in center
            ],

        "length_m":
            float(
                one_attr(
                    box,
                    ("length_m",),
                    "Box length",
                )
            ),

        "width_m":
            float(
                one_attr(
                    box,
                    ("width_m",),
                    "Box width",
                )
            ),

        "height_m":
            float(
                one_attr(
                    box,
                    ("height_m",),
                    "Box height",
                )
            ),

        "yaw_rad":
            float(
                one_attr(
                    box,
                    (
                        "yaw_rad",
                        "heading_rad",
                    ),
                    "Box yaw",
                )
            ),
    }

    require(
        all(
            math.isfinite(value)
            for value in (
                *result[
                    "center_xyz_H0_m"
                ],
                result["length_m"],
                result["width_m"],
                result["height_m"],
                result["yaw_rad"],
            )
        ),
        "Non-finite causal box geometry.",
    )

    return result


def wrapped_yaw_error(
    first,
    second,
):

    return abs(
        math.atan2(
            math.sin(
                float(first)
                -
                float(second)
            ),
            math.cos(
                float(first)
                -
                float(second)
            ),
        )
    )


def geometry_parity(
    selector,
    actor_box,
    tolerance=1.0e-8,
):

    persisted = selector[
        "current_box_H0"
    ]

    current = box_geometry(
        actor_box
    )

    persisted_center = np.asarray(
        persisted[
            "center_xyz_H0_m"
        ],
        dtype=np.float64,
    )

    current_center = np.asarray(
        current[
            "center_xyz_H0_m"
        ],
        dtype=np.float64,
    )

    require(
        persisted_center.shape
        ==
        (3,),
        "Persisted selector center is not 3-D.",
    )

    center_error = float(
        np.max(
            np.abs(
                persisted_center
                -
                current_center
            )
        )
    )

    dimension_error = max(
        abs(
            float(persisted[key])
            -
            float(current[key])
        )
        for key in (
            "length_m",
            "width_m",
            "height_m",
        )
    )

    yaw_error = wrapped_yaw_error(
        persisted["yaw_rad"],
        current["yaw_rad"],
    )

    require(
        center_error
        <=
        tolerance,
        (
            "Selector/current center mismatch: "
            f"{center_error}"
        ),
    )

    require(
        dimension_error
        <=
        tolerance,
        (
            "Selector/current dimensions mismatch: "
            f"{dimension_error}"
        ),
    )

    require(
        yaw_error
        <=
        tolerance,
        (
            "Selector/current yaw mismatch: "
            f"{yaw_error}"
        ),
    )

    return {
        "center_m":
            center_error,

        "dimension_m":
            dimension_error,

        "yaw_rad":
            yaw_error,
    }


def selector_gate(
    record,
    route,
):

    require(
        str(
            record.get(
                "object_type"
            )
        )
        in
        CLASSES,
        f"{route}: unsupported actor class.",
    )

    require(
        record.get(
            "current_headlamp_eligible"
        )
        is True,
        f"{route}: not current-headlamp eligible.",
    )

    require(
        record.get(
            "future_used_for_eligibility"
        )
        is False,
        f"{route}: future used for eligibility.",
    )

    require(
        record.get(
            "tracks_to_predict_used"
        )
        is False,
        f"{route}: tracks_to_predict used.",
    )

    require(
        record.get(
            "objects_of_interest_used"
        )
        is False,
        f"{route}: objects_of_interest used.",
    )

    if route == "predictive_match":

        require(
            record.get(
                "centroid_only"
            )
            is False,
            (
                "Predictive selector "
                "is unexpectedly centroid-only."
            ),
        )


def frozen_class_counts(
    value,
):

    require(
        isinstance(
            value,
            dict,
        ),
        (
            "headlamp_relevant_counts "
            "must be a dict."
        ),
    )

    return {
        class_name:
            int(
                value.get(
                    class_name,
                    0,
                )
            )
        for class_name
        in CLASSES
    }


def actor_record(
    selector,
    route,
    actor_box,
    parity,
):

    persisted = selector[
        "current_box_H0"
    ]

    result = {
        "selector_route":
            route,

        "actor_box_index":
            int(
                selector[
                    "actor_box_index"
                ]
            ),

        "object_type":
            str(
                selector[
                    "object_type"
                ]
            ),

        "track_index":
            int(
                actor_box.track_index
            ),

        "track_id":
            str(
                actor_box.track_id
            ),

        "current_box_H0": {
            "center_xyz_H0_m":
                [
                    float(value)
                    for value in persisted[
                        "center_xyz_H0_m"
                    ]
                ],

            "length_m":
                float(
                    persisted[
                        "length_m"
                    ]
                ),

            "width_m":
                float(
                    persisted[
                        "width_m"
                    ]
                ),

            "height_m":
                float(
                    persisted[
                        "height_m"
                    ]
                ),

            "yaw_rad":
                float(
                    persisted[
                        "yaw_rad"
                    ]
                ),
        },

        "reactive_control_input":
            (
                str(
                    selector[
                        "object_type"
                    ]
                )
                ==
                VEHICLE
            ),

        "geometry_parity":
            parity,

        "future_used_for_eligibility":
            False,

        "tracks_to_predict_used":
            False,

        "objects_of_interest_used":
            False,

        "causal_box_future_state_used":
            bool(
                actor_box.future_state_used
            ),
    }

    if route == "predictive_match":

        result[
            "prediction_id"
        ] = str(
            selector[
                "prediction_id"
            ]
        )

    return result


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART 2B — PART 1/2"
    )
    print(
        "ORIGINAL-REACTIVE CAUSAL DECISION LEDGER FREEZE"
    )
    print(
        "NO FUTURE-TRUTH SCORING / NO PERFORMANCE METRICS"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Idempotent frozen readback
    # --------------------------------------------------------

    if REPORT.is_file():

        existing = safe_json(
            REPORT
        )

        if (
            existing.get(
                "status"
            )
            ==
            "PASS_REACTIVE_DECISION_LEDGER_FROZEN"
        ):

            exact_gate()

            for key, path in (
                (
                    "maps",
                    MAPS,
                ),
                (
                    "ledger",
                    LEDGER,
                ),
            ):

                require(
                    path.is_file(),
                    (
                        "Frozen artifact missing: "
                        f"{path}"
                    ),
                )

                require(
                    file_sha(path)
                    ==
                    existing[
                        "frozen_outputs"
                    ][
                        key
                    ][
                        "sha256"
                    ],
                    (
                        "Frozen artifact changed: "
                        f"{path}"
                    ),
                )

            rc, count, output = (
                regression()
            )

            require(
                (
                    rc == 0
                    and
                    count == TESTS
                ),
                (
                    "Readback regression failed:\n"
                    +
                    output_tail(
                        output
                    )
                ),
            )

            print(
                "existing freeze       = EXACT PASS"
            )
            print(
                "Stage6 regression     = 224 / 224 PASS"
            )
            print(
                "future truth scoring  = NO"
            )
            print(
                "performance metrics   = NO"
            )
            print(
                "STATUS = "
                "PASS_REACTIVE_DECISION_LEDGER_FROZEN"
            )
            print(
                "no rewrite performed  = YES"
            )
            print(
                "terminal remains open = YES"
            )

            return

    # --------------------------------------------------------
    # A. Exact upstream
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT FROZEN BOUNDARY ====="
    )

    exact_gate()

    reactive_freeze = safe_json(
        S6
        / "reports/"
          "block68_reactive_development_scoring_freeze.json"
    )

    require(
        reactive_freeze.get(
            "status"
        )
        ==
        "PASS_REACTIVE_SCORING_SEMANTICS_FROZEN",
        (
            "Reactive scoring semantics "
            "are not frozen PASS."
        ),
    )

    require(
        reactive_freeze[
            "scientific_boundary"
        ][
            "development_metric_values_read"
        ]
        is False,
        "Development metrics were already read.",
    )

    require(
        reactive_freeze[
            "scientific_boundary"
        ][
            "formal_content_opened"
        ]
        is False,
        "Formal content was already opened.",
    )

    print(
        "metric/evaluator/NI/reactive freezes = EXACT PASS"
    )

    # --------------------------------------------------------
    # B. Pre regression
    # --------------------------------------------------------

    print()
    print(
        "===== B. PRE-RUN REGRESSION ====="
    )

    (
        rc_pre,
        tests_pre,
        output_pre,
    ) = regression()

    require(
        (
            rc_pre == 0
            and
            tests_pre == TESTS
        ),
        (
            "Pre-run Stage6 regression failed:\n"
            +
            output_tail(
                output_pre
            )
        ),
    )

    print(
        "pre-run regression = 224 / 224 PASS"
    )

    # --------------------------------------------------------
    # C. Current-only population
    # --------------------------------------------------------

    print()
    print(
        "===== C. CURRENT-ONLY DEVELOPMENT POPULATION ====="
    )

    cohort = safe_jsonl(
        COHORT
    )

    predictive = safe_jsonl(
        PRED
    )

    fallback = safe_jsonl(
        FALLBACK
    )

    require(
        len(cohort) == N,
        (
            f"Expected {N} cohort rows; "
            f"got {len(cohort)}."
        ),
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
        N,
        "Development cohort IDs are not unique.",
    )

    for row in cohort:

        require(
            row.get(
                "selection_uses_future"
            )
            is False,
            "Cohort selection used future.",
        )

        require(
            row.get(
                "tracks_to_predict_used"
            )
            is False,
            "Cohort selection used tracks_to_predict.",
        )

        require(
            row.get(
                "objects_of_interest_used"
            )
            is False,
            "Cohort selection used objects_of_interest.",
        )

        require(
            row.get(
                "selection_uses_full_box"
            )
            is True,
            "Cohort is not frozen full-box selected.",
        )

    by_key = {}
    by_scenario = defaultdict(
        list
    )

    for route, records in (
        (
            "predictive_match",
            predictive,
        ),
        (
            "reactive_fallback",
            fallback,
        ),
    ):

        for record in records:

            selector_gate(
                record,
                route,
            )

            key = (
                str(
                    record[
                        "scenario_id"
                    ]
                ),
                int(
                    record[
                        "cohort_index"
                    ]
                ),
                int(
                    record[
                        "actor_box_index"
                    ]
                ),
            )

            require(
                key not in by_key,
                (
                    "Duplicate selector actor "
                    f"identity: {key}"
                ),
            )

            item = (
                route,
                record,
            )

            by_key[
                key
            ] = item

            by_scenario[
                str(
                    record[
                        "scenario_id"
                    ]
                )
            ].append(
                item
            )

    require(
        set(
            by_scenario
        )
        ==
        set(
            scenario_ids
        ),
        (
            "Selector scenario set differs "
            "from development cohort."
        ),
    )

    for index, row in enumerate(
        cohort
    ):

        scenario_id = str(
            row[
                "scenario_id"
            ]
        )

        selected = by_scenario[
            scenario_id
        ]

        require(
            all(
                int(
                    record[
                        "cohort_index"
                    ]
                )
                ==
                index
                for _, record
                in selected
            ),
            (
                "Selector cohort_index differs "
                f"for {scenario_id}."
            ),
        )

        observed = Counter(
            str(
                record[
                    "object_type"
                ]
            )
            for _, record
            in selected
        )

        require(
            {
                class_name:
                    int(
                        observed.get(
                            class_name,
                            0,
                        )
                    )
                for class_name
                in CLASSES
            }
            ==
            frozen_class_counts(
                row[
                    "headlamp_relevant_counts"
                ]
            ),
            (
                "Selector union differs from "
                "frozen headlamp-relevant population "
                f"for {scenario_id}."
            ),
        )

    print(
        "development cohort   = 120 / 120 PASS"
    )
    print(
        "selector records     =",
        len(
            by_key
        ),
    )
    print(
        "selector union       = COMPLETE + DISJOINT PASS"
    )
    print(
        "future eligibility   = NOT USED"
    )

    # --------------------------------------------------------
    # D. Exact already-frozen grid
    # --------------------------------------------------------

    print()
    print(
        "===== D. FROZEN GRID RUNTIME BIND ====="
    )

    grid_config_sha = file_sha(
        GRID_CONFIG
    )

    (
        grid,
        grid_attempts,
        grid_signature,
    ) = resolve_grid()

    blank = np.asarray(
        part_a_original_reactive_map_from_h0_centers(
            grid,
            (),
        ),
        dtype=np.float64,
    )

    require(
        blank.shape
        ==
        GRID_SHAPE,
        "Frozen grid shape mismatch.",
    )

    print(
        "grid map shape       = [501, 301] PASS"
    )
    print(
        "numeric invention    = NO"
    )

    # --------------------------------------------------------
    # E. 120 causal decisions
    # --------------------------------------------------------

    print()
    print(
        "===== E. MATERIALIZE 120 CAUSAL t0 DECISIONS ====="
    )

    for path in (
        MAPS,
        LEDGER,
        REPORT,
    ):

        require(
            not path.exists(),
            (
                "Unexpected output exists "
                "without PASS report: "
                f"{path}"
            ),
        )

    MAPS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    maps_tmp = MAPS.with_suffix(
        ".npy.tmp"
    )

    ledger_tmp = LEDGER.with_suffix(
        ".jsonl.tmp"
    )

    for path in (
        maps_tmp,
        ledger_tmp,
    ):

        if path.exists():
            path.unlink()

    maps = np.lib.format.open_memmap(
        maps_tmp,
        mode="w+",
        dtype=np.float64,
        shape=(
            N,
            *GRID_SHAPE,
        ),
    )

    ledger_rows = []
    class_totals = Counter()

    max_parity_error = {
        "center_m":
            0.0,

        "dimension_m":
            0.0,

        "yaw_rad":
            0.0,
    }

    vehicle_input_total = 0
    committed = False

    try:

        for index, row in enumerate(
            cohort
        ):

            scenario_id = str(
                row[
                    "scenario_id"
                ]
            )

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
                    "Loaded scenario ID mismatch: "
                    f"{scenario_id}"
                ),
            )

            require(
                int(
                    scenario.current_time_index
                )
                ==
                int(
                    row[
                        "current_time_index"
                    ]
                ),
                (
                    "current_time_index mismatch: "
                    f"{scenario_id}"
                ),
            )

            # Important:
            # this runner never directly accesses
            # scenario.tracks[*].states.
            adapted = adapt_causal_womd_scenario(
                scenario
            )

            causal_boxes = tuple(
                build_causal_adb_actor_boxes(
                    scenario=scenario,
                    adapted=adapted,
                )
            )

            require(
                all(
                    box.future_state_used
                    is False
                    for box in causal_boxes
                ),
                (
                    "Causal box builder reports "
                    f"future-state use in {scenario_id}."
                ),
            )

            require(
                all(
                    box.tracks_to_predict_used
                    is False
                    for box in causal_boxes
                ),
                (
                    "Causal box builder reports "
                    f"tracks_to_predict use in {scenario_id}."
                ),
            )

            require(
                all(
                    box.objects_of_interest_used
                    is False
                    for box in causal_boxes
                ),
                (
                    "Causal box builder reports "
                    f"objects_of_interest use in {scenario_id}."
                ),
            )

            actor_rows = []
            vehicle_centers = []

            selected = sorted(
                by_scenario[
                    scenario_id
                ],
                key=lambda item: (
                    int(
                        item[
                            1
                        ][
                            "actor_box_index"
                        ]
                    ),
                    item[0],
                ),
            )

            for route, selector in selected:

                actor_box_index = int(
                    selector[
                        "actor_box_index"
                    ]
                )

                require(
                    0
                    <=
                    actor_box_index
                    <
                    len(
                        causal_boxes
                    ),
                    (
                        "actor_box_index out of range: "
                        f"{scenario_id}/"
                        f"{actor_box_index}"
                    ),
                )

                actor_box = causal_boxes[
                    actor_box_index
                ]

                require(
                    str(
                        actor_box.scenario_id
                    )
                    ==
                    scenario_id,
                    "Causal actor scenario mismatch.",
                )

                require(
                    str(
                        actor_box.object_type
                    )
                    ==
                    str(
                        selector[
                            "object_type"
                        ]
                    ),
                    (
                        "Selector/current class mismatch: "
                        f"{scenario_id}/"
                        f"{actor_box_index}"
                    ),
                )

                parity = geometry_parity(
                    selector,
                    actor_box,
                )

                for key in max_parity_error:

                    max_parity_error[
                        key
                    ] = max(
                        max_parity_error[
                            key
                        ],
                        parity[
                            key
                        ],
                    )

                center = np.asarray(
                    actor_box
                    .stage1_anchor_center_H0_m,
                    dtype=np.float64,
                )

                require(
                    (
                        center.shape
                        ==
                        (3,)
                        and
                        np.all(
                            np.isfinite(
                                center
                            )
                        )
                    ),
                    (
                        "Invalid causal H0 center: "
                        f"{scenario_id}/"
                        f"{actor_box_index}"
                    ),
                )

                actor_class = str(
                    actor_box.object_type
                )

                class_totals[
                    actor_class
                ] += 1

                # Frozen original-reactive Part-A baseline:
                # current vehicle centers only.
                if actor_class == VEHICLE:

                    vehicle_centers.append(
                        tuple(
                            float(value)
                            for value in center
                        )
                    )

                actor_rows.append(
                    actor_record(
                        selector,
                        route,
                        actor_box,
                        parity,
                    )
                )

            current_map = np.asarray(
                part_a_original_reactive_map_from_h0_centers(
                    grid,
                    tuple(
                        vehicle_centers
                    ),
                ),
                dtype=np.float64,
            )

            require(
                current_map.shape
                ==
                GRID_SHAPE,
                (
                    "Reactive map shape mismatch: "
                    f"{scenario_id}"
                ),
            )

            require(
                (
                    np.all(
                        np.isfinite(
                            current_map
                        )
                    )
                    and
                    np.all(
                        (
                            current_map >= 0.0
                        )
                        &
                        (
                            current_map <= 1.0
                        )
                    )
                ),
                (
                    "Reactive map invalid: "
                    f"{scenario_id}"
                ),
            )

            held = (
                hold_current_reactive_schedule(
                    current_map
                )
            )

            require(
                held.shape
                ==
                (
                    4,
                    *GRID_SHAPE,
                ),
                "Held schedule shape mismatch.",
            )

            require(
                all(
                    np.array_equal(
                        held[
                            horizon
                        ],
                        current_map,
                    )
                    for horizon
                    in range(4)
                ),
                (
                    "Reactive scoring helper "
                    "did not hold t0 action exactly."
                ),
            )

            maps[
                index
            ] = current_map

            vehicle_input_total += len(
                vehicle_centers
            )

            ledger_rows.append(
                {
                    "stage":
                        6,

                    "block":
                        "6.8_Part2B_Part1of2",

                    "cohort_index":
                        index,

                    "cohort_rank":
                        int(
                            row[
                                "cohort_rank"
                            ]
                        ),

                    "scenario_id":
                        scenario_id,

                    "current_time_index":
                        int(
                            row[
                                "current_time_index"
                            ]
                        ),

                    "selected_adb_actor_count":
                        len(
                            actor_rows
                        ),

                    "selected_adb_actor_counts":
                        {
                            class_name:
                                sum(
                                    actor[
                                        "object_type"
                                    ]
                                    ==
                                    class_name

                                    for actor
                                    in actor_rows
                                )

                            for class_name
                            in CLASSES
                        },

                    "selected_adb_actors":
                        actor_rows,

                    "original_reactive_ADB": {
                        "decision_time":
                            "t0_current_causal_anchor",

                        "controller_input_class":
                            "TYPE_VEHICLE_ONLY",

                        "vehicle_center_count":
                            len(
                                vehicle_centers
                            ),

                        "vehicle_centers_H0_m":
                            [
                                list(
                                    center
                                )
                                for center
                                in vehicle_centers
                            ],

                        "future_input":
                            False,

                        "posterior_input":
                            False,

                        "future_reobservation":
                            False,

                        "future_controller_refresh":
                            False,

                        "constructed_oracle_input":
                            False,

                        "score_schedule":
                            (
                                "hold_t0_action_at_"
                                "0.1_0.3_0.5_1.0s"
                            ),
                    },

                    "t0_map_row_index":
                        index,

                    "t0_map_shape":
                        list(
                            GRID_SHAPE
                        ),

                    "t0_map_dtype":
                        "float64",

                    "t0_map_sha256":
                        array_sha(
                            current_map
                        ),

                    "causality": {
                        "future_state_field_access_by_runner":
                            False,

                        "future_validity_selector":
                            False,

                        "future_track_duration_selector":
                            False,

                        "tracks_to_predict_selector":
                            False,

                        "objects_of_interest_selector":
                            False,

                        "P_occ_opened":
                            False,

                        "future_truth_scoring":
                            False,

                        "performance_metric_computed":
                            False,
                    },
                }
            )

            if (
                (
                    index + 1
                )
                %
                10
                ==
                0
                or
                (
                    index + 1
                )
                ==
                N
            ):

                print(
                    "t0 decisions          = "
                    f"{index + 1} / {N}"
                )

        maps.flush()

        del maps
        maps = None

        write_jsonl(
            ledger_tmp,
            ledger_rows,
        )

        # Commit only after all 120 decisions exist.
        os.replace(
            maps_tmp,
            MAPS,
        )

        os.replace(
            ledger_tmp,
            LEDGER,
        )

        committed = True

    finally:

        try:

            if maps is not None:

                maps.flush()
                del maps

        except Exception:
            pass

        if not committed:

            for path in (
                maps_tmp,
                ledger_tmp,
            ):

                try:

                    if path.exists():
                        path.unlink()

                except Exception:
                    pass

    print(
        "future truth scoring = NO"
    )
    print(
        "P_occ content         = NOT OPENED"
    )
    print(
        "performance metrics  = NOT COMPUTED"
    )

    # --------------------------------------------------------
    # F. Readback
    # --------------------------------------------------------

    print()
    print(
        "===== F. EXACT LEDGER READBACK ====="
    )

    frozen_ledger = safe_jsonl(
        LEDGER
    )

    require(
        len(
            frozen_ledger
        )
        ==
        N,
        "Frozen ledger row count != 120.",
    )

    frozen_maps = np.load(
        MAPS,
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        frozen_maps.shape
        ==
        (
            N,
            *GRID_SHAPE,
        ),
        "Frozen map-array shape changed.",
    )

    require(
        frozen_maps.dtype
        ==
        np.dtype(
            np.float64
        ),
        "Frozen map-array dtype changed.",
    )

    for index, row in enumerate(
        frozen_ledger
    ):

        require(
            int(
                row[
                    "t0_map_row_index"
                ]
            )
            ==
            index,
            "Ledger/map row index mismatch.",
        )

        require(
            array_sha(
                np.asarray(
                    frozen_maps[
                        index
                    ]
                )
            )
            ==
            row[
                "t0_map_sha256"
            ],
            (
                "Map content SHA mismatch "
                f"at row {index}."
            ),
        )

        require(
            row[
                "causality"
            ][
                "future_truth_scoring"
            ]
            is False,
            "Future truth scoring flag is true.",
        )

        require(
            row[
                "causality"
            ][
                "performance_metric_computed"
            ]
            is False,
            "Performance metric flag is true.",
        )

    del frozen_maps

    maps_sha = file_sha(
        MAPS
    )

    ledger_sha = file_sha(
        LEDGER
    )

    print(
        "map rows/hash         = 120 / 120 EXACT PASS"
    )
    print(
        "ledger/map binding    = EXACT PASS"
    )

    # --------------------------------------------------------
    # G. Post gate
    # --------------------------------------------------------

    print()
    print(
        "===== G. IMMUTABILITY + POST-RUN REGRESSION ====="
    )

    exact_gate()

    require(
        file_sha(
            GRID_CONFIG
        )
        ==
        grid_config_sha,
        "Frozen grid config changed.",
    )

    (
        rc_post,
        tests_post,
        output_post,
    ) = regression()

    require(
        (
            rc_post == 0
            and
            tests_post == TESTS
        ),
        (
            "Post-run regression failed:\n"
            +
            output_tail(
                output_post
            )
        ),
    )

    print(
        "scientific sources    = UNCHANGED"
    )
    print(
        "Stage6 regression     = 224 / 224 PASS"
    )

    # --------------------------------------------------------
    # H. Storage
    # --------------------------------------------------------

    print()
    print(
        "===== H. STORAGE ====="
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        "250-GiB storage reserve violated.",
    )

    print(
        "free GiB              =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve       = PASS"
    )

    # --------------------------------------------------------
    # I. Report
    # --------------------------------------------------------

    result = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8_Part2B_Part1of2",

        "status":
            "PASS_REACTIVE_DECISION_LEDGER_FROZEN",

        "scientific_boundary": {
            "development_current_causal_data_read":
                True,

            "development_performance_metric_values_read":
                False,

            "reactive_performance_outcomes_read":
                False,

            "predictive_performance_outcomes_read":
                False,

            "future_state_fields_accessed_by_runner":
                False,

            "future_truth_scoring":
                False,

            "P_occ_content_opened":
                False,

            "exact_NI_deltas_computed":
                False,

            "formal_content_opened_by_runner":
                False,

            "formal_evaluation":
                False,

            "controller_modified":
                False,

            "policy_tuning":
                False,

            "parameter_tuning":
                False,
        },

        "population": {
            "scenario_count":
                N,

            "cohort_path":
                str(
                    COHORT
                ),

            "cohort_sha256":
                file_sha(
                    COHORT
                ),

            "predictive_selector_path":
                str(
                    PRED
                ),

            "predictive_selector_sha256":
                file_sha(
                    PRED
                ),

            "reactive_fallback_selector_path":
                str(
                    FALLBACK
                ),

            "reactive_fallback_selector_sha256":
                file_sha(
                    FALLBACK
                ),

            "selector_union_count":
                len(
                    by_key
                ),

            "selector_union_complete_disjoint":
                True,

            "actor_class_totals":
                {
                    class_name:
                        int(
                            class_totals[
                                class_name
                            ]
                        )
                    for class_name
                    in CLASSES
                },

            "vehicle_centers_used_total":
                int(
                    vehicle_input_total
                ),
        },

        "reactive_decision_semantics": {
            "baseline":
                "original_reactive_ADB",

            "decision_time":
                "current_causal_t0",

            "controller_actor_input":
                (
                    "selected_headlamp_relevant_"
                    "TYPE_VEHICLE_centers_only"
                ),

            "future_input":
                False,

            "posterior_input":
                False,

            "future_reobservation":
                False,

            "future_controller_refresh":
                False,

            "constructed_oracle_input":
                False,

            "future_scoring_schedule":
                (
                    "hold_t0_action_at_"
                    "0.1_0.3_0.5_1.0s"
                ),
        },

        "grid": {
            "shape":
                list(
                    GRID_SHAPE
                ),

            "grid_config_path":
                str(
                    GRID_CONFIG
                ),

            "grid_config_sha256":
                grid_config_sha,

            "constructor_signature":
                grid_signature,

            "resolution_attempts":
                grid_attempts,

            "numeric_values_invented":
                False,
        },

        "geometry_parity_max":
            {
                key:
                    float(
                        value
                    )
                for key, value
                in max_parity_error.items()
            },

        "frozen_outputs": {
            "maps": {
                "path":
                    str(
                        MAPS
                    ),

                "sha256":
                    maps_sha,

                "shape":
                    [
                        N,
                        *GRID_SHAPE,
                    ],

                "dtype":
                    "float64",
            },

            "ledger": {
                "path":
                    str(
                        LEDGER
                    ),

                "sha256":
                    ledger_sha,

                "rows":
                    N,
            },
        },

        "regression": {
            "pre":
                tests_pre,

            "post":
                tests_post,

            "status":
                "PASS",
        },

        "storage": {
            "free_GiB":
                free_gib,

            "reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "formal_evaluation_allowed":
            False,

        "next":
            (
                "Part2B Part2of2 may now access "
                "evaluator-only future truth, construct "
                "exact future oracle/full-box supports, "
                "compute original-reactive overmask and "
                "pedestrian/cyclist visibility values, "
                "then apply the already frozen NI rule. "
                "Predictive performance must remain unused "
                "until all three exact NI deltas freeze."
            ),
    }

    atomic_json(
        REPORT,
        result,
    )

    print()
    print(
        "===== I. FREEZE REPORT ====="
    )

    print(
        "t0 maps SHA256        =",
        maps_sha,
    )

    print(
        "ledger SHA256         =",
        ledger_sha,
    )

    print(
        "report SHA256         =",
        file_sha(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 REACTIVE DECISION LEDGER — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "development cohort        = 120 / 120"
    )
    print(
        "selector union             = COMPLETE + DISJOINT PASS"
    )
    print(
        "current geometry parity    = PASS"
    )
    print(
        "original reactive input    = CURRENT VEHICLE CENTERS ONLY"
    )
    print(
        "reactive t0 maps           = 120 FROZEN"
    )
    print(
        "4-horizon scoring          = HOLD t0 ACTION"
    )
    print(
        "future state field access  = NO"
    )
    print(
        "future truth scoring       = NO"
    )
    print(
        "P_occ content              = NOT OPENED"
    )
    print(
        "reactive performance       = NOT COMPUTED"
    )
    print(
        "predictive performance     = NOT COMPUTED"
    )
    print(
        "exact NI deltas            = NOT COMPUTED"
    )
    print(
        "formal content             = NOT OPENED BY RUNNER"
    )
    print(
        "controller/policy tuning   = NO"
    )
    print(
        "Stage6 regression          = 224 / 224 PASS"
    )
    print(
        "formal evaluation allowed  = NO"
    )
    print(
        "STATUS = "
        "PASS_REACTIVE_DECISION_LEDGER_FROZEN"
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
        "BLOCK 6.8 REACTIVE DECISION LEDGER = BLOCKED"
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
        "future truth scoring       = NO"
    )
    print(
        "P_occ content              = NOT OPENED"
    )
    print(
        "reactive performance       = NOT COMPUTED"
    )
    print(
        "predictive performance     = NOT COMPUTED"
    )
    print(
        "exact NI deltas            = NOT COMPUTED"
    )
    print(
        "formal evaluation          = NO"
    )
    print(
        "Do not run Part2B Part2/2."
    )
    print(
        "Send this BLOCKED output for targeted repair."
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

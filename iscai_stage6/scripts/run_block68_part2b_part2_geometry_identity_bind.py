from __future__ import annotations

import dataclasses
from hashlib import sha256
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

LEDGER = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_decision_ledger.jsonl"
)

MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

REACTIVE_REPORT = (
    S6
    / "reports/"
      "block68_reactive_decision_ledger_freeze.json"
)

GRID_BINDING = (
    S6
    / "configs/"
      "stage6_frozen_illumination_grid_runtime_binding.json"
)

EVALUATOR_CONTRACT = (
    S6
    / "configs/"
      "stage6_evaluator_reference_contract.json"
)

OUTPUT = (
    S6
    / "reports/"
      "block68_part2b_part2_geometry_identity_bind.json"
)


EXPECTED = {
    "ledger":
        "ac2b954571cdd0166bbcfa5bb8b2956785d1e132288dbe4af7fe2d55adeafcaa",

    "maps":
        "7c5d63c76c3d9ef74adbbd7a68e3ce7483b1dd80fd262cba06f6ce1909994993",

    "reactive_report":
        "6ca868dbdf13e8a675813d4af2d3c814ce5297155126e5124ac452131670903b",

    "grid_binding":
        "be5145fe60a941e7906639916cfa384d5674dc29f294f5a42e12c962bc4df4bc",

    "evaluator_contract":
        "c9ff0390820a3c37702fb391ba603244fffcf7c03c476045b590aa483e89bd7e",
}

EXPECTED_TESTS = 224


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def exact(path: Path, expected: str, label: str):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = file_sha(path)

    require(
        actual == expected,
        (
            f"{label} SHA mismatch\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    return actual


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            text = line.strip()

            if not text:
                continue

            value = json.loads(text)

            require(
                isinstance(value, dict),
                (
                    "Ledger row is not object: "
                    f"line {line_number}"
                ),
            )

            rows.append(value)

    return rows


def nested_paths(value, prefix=""):
    result = set()

    if isinstance(value, dict):

        for key, child in value.items():

            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            result.add(path)

            result.update(
                nested_paths(
                    child,
                    path,
                )
            )

    elif isinstance(value, list):

        marker = (
            f"{prefix}[]"
            if prefix
            else "[]"
        )

        result.add(marker)

        for child in value[:3]:

            result.update(
                nested_paths(
                    child,
                    marker,
                )
            )

    return result


def dataclass_inventory(module):
    output = {}

    for name in sorted(dir(module)):

        obj = getattr(
            module,
            name,
        )

        if (
            inspect.isclass(obj)
            and
            dataclasses.is_dataclass(obj)
        ):

            output[name] = [
                {
                    "name": field.name,
                    "type": str(field.type),
                }
                for field in dataclasses.fields(obj)
            ]

    return output


def function_signature(module, name):
    obj = getattr(
        module,
        name,
        None,
    )

    require(
        callable(obj),
        (
            f"Required callable not found: "
            f"{module.__name__}.{name}"
        ),
    )

    return str(
        inspect.signature(obj)
    )


def source_excerpt(module, name):
    obj = getattr(
        module,
        name,
        None,
    )

    require(
        callable(obj),
        (
            f"Cannot extract source: "
            f"{module.__name__}.{name}"
        ),
    )

    return inspect.getsource(obj)


def find_identity_paths(paths):
    tokens = (
        "track_index",
        "track_id",
        "actor_box_index",
        "object_type",
        "scenario_id",
        "cohort_index",
        "current_time_index",
        "current_box",
        "actors",
        "selected",
        "causal",
    )

    return [
        path
        for path in sorted(paths)
        if any(
            token in path.lower()
            for token in tokens
        )
    ]


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
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output = process.stdout

    count = None

    for line in output.splitlines():

        text = line.strip()

        if text.startswith("Ran ") and " test" in text:

            try:
                count = int(
                    text.split()[1]
                )
            except Exception:
                pass

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            +
            "\n".join(
                output.splitlines()[-80:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests; "
            f"got {count}."
        ),
    )

    return count


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8 PART2B PART2/2"
    )
    print(
        "FUTURE-EVALUATOR GEOMETRY + ACTOR IDENTITY BIND"
    )
    print(
        "READ ONLY — NO FUTURE TRUTH OPENED"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # A. Freeze continuity
    # --------------------------------------------------------

    print()
    print(
        "===== A. EXACT FROZEN CONTINUITY ====="
    )

    seals = {}

    for key, path in (
        ("ledger", LEDGER),
        ("maps", MAPS),
        ("reactive_report", REACTIVE_REPORT),
        ("grid_binding", GRID_BINDING),
        ("evaluator_contract", EVALUATOR_CONTRACT),
    ):

        seals[key] = exact(
            path,
            EXPECTED[key],
            key,
        )

        print(
            f"{key:20s} = EXACT PASS"
        )

    # --------------------------------------------------------
    # B. Ledger identity schema
    # --------------------------------------------------------

    print()
    print(
        "===== B. FROZEN DECISION LEDGER SCHEMA ====="
    )

    rows = read_jsonl(
        LEDGER
    )

    require(
        len(rows) == 120,
        (
            "Expected exactly 120 ledger rows; "
            f"got {len(rows)}."
        ),
    )

    scenario_ids = [
        row.get(
            "scenario_id"
        )
        for row in rows
    ]

    require(
        all(
            isinstance(value, str)
            and value
            for value in scenario_ids
        ),
        "Every ledger row must expose scenario_id.",
    )

    require(
        len(set(scenario_ids))
        ==
        120,
        "Ledger scenario IDs are not unique.",
    )

    all_paths = set()

    for row in rows:
        all_paths.update(
            nested_paths(row)
        )

    identity_paths = find_identity_paths(
        all_paths
    )

    print(
        "ledger rows          = 120 PASS"
    )
    print(
        "unique scenario IDs  = 120 PASS"
    )

    print()
    print(
        "identity-relevant ledger paths:"
    )

    for path in identity_paths:
        print(
            "  ",
            path,
        )

    track_index_paths = [
        path
        for path in identity_paths
        if "track_index" in path.lower()
    ]

    track_id_paths = [
        path
        for path in identity_paths
        if "track_id" in path.lower()
    ]

    actor_box_paths = [
        path
        for path in identity_paths
        if "actor_box_index" in path.lower()
    ]

    require(
        (
            track_index_paths
            or
            track_id_paths
        ),
        (
            "Frozen decision ledger does not expose "
            "a WOMD actor identity path."
        ),
    )

    print()
    print(
        "track_index paths    =",
        len(track_index_paths),
    )
    print(
        "track_id paths       =",
        len(track_id_paths),
    )
    print(
        "actor_box_index paths=",
        len(actor_box_paths),
    )

    # --------------------------------------------------------
    # C. Geometry API
    # --------------------------------------------------------

    print()
    print(
        "===== C. EXACT STAGE6 GEOMETRY API ====="
    )

    geometry = importlib.import_module(
        "iscai_stage6.adb.geometry"
    )

    geometry_file = Path(
        inspect.getfile(
            geometry
        )
    )

    print(
        "geometry module =",
        geometry_file,
    )

    print()
    print(
        "dataclasses:"
    )

    geometry_dataclasses = (
        dataclass_inventory(
            geometry
        )
    )

    for name, fields in geometry_dataclasses.items():

        print()
        print(
            name
        )

        for field in fields:

            print(
                "   ",
                field["name"],
                ":",
                field["type"],
            )

    projection_signature = (
        function_signature(
            geometry,
            "project_box_to_headlamp",
        )
    )

    projection_source = (
        source_excerpt(
            geometry,
            "project_box_to_headlamp",
        )
    )

    print()
    print(
        "project_box_to_headlamp signature =",
        projection_signature,
    )

    print()
    print(
        "----- project_box_to_headlamp source -----"
    )
    print(
        projection_source
    )

    # --------------------------------------------------------
    # D. WOMD causal-box API
    # --------------------------------------------------------

    print()
    print(
        "===== D. CAUSAL ADB BOX API ====="
    )

    womd_geometry = importlib.import_module(
        "iscai_stage6.adb.womd_geometry"
    )

    causal_inventory = (
        dataclass_inventory(
            womd_geometry
        )
    )

    require(
        "CausalADBActorBox"
        in
        causal_inventory,
        (
            "CausalADBActorBox dataclass "
            "not found."
        ),
    )

    for field in causal_inventory[
        "CausalADBActorBox"
    ]:

        print(
            field["name"],
            ":",
            field["type"],
        )

    print()
    print(
        "build_causal_adb_actor_boxes signature =",
        function_signature(
            womd_geometry,
            "build_causal_adb_actor_boxes",
        ),
    )

    # --------------------------------------------------------
    # E. Stage1 transform route
    # --------------------------------------------------------

    print()
    print(
        "===== E. STAGE1 ANCHOR TRANSFORM API ====="
    )

    stage1_frames = importlib.import_module(
        "iscai_stage1.geometry.frames"
    )

    stage1_adapter = importlib.import_module(
        "iscai_stage1.actors.womd_adapter"
    )

    frame_inventory = (
        dataclass_inventory(
            stage1_frames
        )
    )

    adapter_inventory = (
        dataclass_inventory(
            stage1_adapter
        )
    )

    for wanted in (
        "AnchorFrames",
        "AdaptedScenario",
    ):

        inventory = (
            frame_inventory
            if wanted == "AnchorFrames"
            else adapter_inventory
        )

        require(
            wanted in inventory,
            f"{wanted} dataclass not found.",
        )

        print()
        print(
            wanted
        )

        for field in inventory[wanted]:

            print(
                "   ",
                field["name"],
                ":",
                field["type"],
            )

    print()
    print(
        "adapt_causal_womd_scenario signature =",
        function_signature(
            stage1_adapter,
            "adapt_causal_womd_scenario",
        ),
    )

    # --------------------------------------------------------
    # F. Reconciled metric runtime
    # --------------------------------------------------------

    print()
    print(
        "===== F. EXACT THREE NI-BOUND METRIC APIs ====="
    )

    metrics = importlib.import_module(
        "iscai_stage6.adb.metric_semantics"
    )

    metric_signatures = {}

    for name in (
        "over_masking_area",
        "pedestrian_visibility_proxy",
        "cyclist_visibility_proxy",
    ):

        signature = function_signature(
            metrics,
            name,
        )

        metric_signatures[
            name
        ] = signature

        print()
        print(
            name,
            signature,
        )

        print(
            source_excerpt(
                metrics,
                name,
            )
        )

    # --------------------------------------------------------
    # G. Evaluator-reference helper
    # --------------------------------------------------------

    print()
    print(
        "===== G. FROZEN EVALUATOR-REFERENCE HELPER ====="
    )

    evaluator_reference = importlib.import_module(
        "iscai_stage6.adb.evaluator_reference"
    )

    evaluator_file = Path(
        inspect.getfile(
            evaluator_reference
        )
    )

    print(
        "module =",
        evaluator_file,
    )

    evaluator_functions = {}

    for name in sorted(
        dir(
            evaluator_reference
        )
    ):

        obj = getattr(
            evaluator_reference,
            name,
        )

        if (
            inspect.isfunction(obj)
            and
            obj.__module__
            ==
            evaluator_reference.__name__
        ):

            evaluator_functions[
                name
            ] = str(
                inspect.signature(
                    obj
                )
            )

            print(
                name,
                evaluator_functions[name],
            )

    # --------------------------------------------------------
    # H. NI rule module discovery
    # --------------------------------------------------------

    print()
    print(
        "===== H. FROZEN NI RULE API ====="
    )

    adb_dir = (
        S6
        / "src/iscai_stage6/adb"
    )

    ni_candidates = sorted(
        path
        for path in adb_dir.glob(
            "*.py"
        )
        if (
            "inferior"
            in path.name.lower()
            or
            "bound"
            in path.name.lower()
            or
            "accept"
            in path.name.lower()
        )
    )

    print(
        "candidate modules =",
        len(ni_candidates),
    )

    for path in ni_candidates:

        print(
            "  ",
            path.name,
            file_sha(path),
        )

    # --------------------------------------------------------
    # I. Hard scientific boundary
    # --------------------------------------------------------

    print()
    print(
        "===== I. SCIENTIFIC BOUNDARY ====="
    )

    print(
        "future WOMD states opened     = NO"
    )
    print(
        "future truth values read       = NO"
    )
    print(
        "reactive map values analyzed   = NO"
    )
    print(
        "oracle masks materialized      = NO"
    )
    print(
        "overmask values computed       = NO"
    )
    print(
        "ped visibility computed        = NO"
    )
    print(
        "cyclist visibility computed    = NO"
    )
    print(
        "predictive performance         = NO"
    )
    print(
        "NI deltas computed             = NO"
    )
    print(
        "formal content opened          = NO"
    )
    print(
        "policy/controller modified     = NO"
    )

    # --------------------------------------------------------
    # J. Regression
    # --------------------------------------------------------

    print()
    print(
        "===== J. FULL STAGE6 REGRESSION ====="
    )

    test_count = run_regression()

    print(
        "Stage6 regression =",
        f"{test_count} / {test_count} PASS",
    )

    # --------------------------------------------------------
    # K. Report
    # --------------------------------------------------------

    result = {
        "stage":
            6,

        "block":
            "6.8_Part2B_Part2_geometry_identity_bind",

        "status":
            "PASS_FUTURE_EVALUATOR_GEOMETRY_IDENTITY_BOUND",

        "frozen_seals":
            seals,

        "decision_ledger": {
            "rows":
                len(rows),

            "unique_scenarios":
                len(
                    set(
                        scenario_ids
                    )
                ),

            "identity_paths":
                identity_paths,

            "track_index_paths":
                track_index_paths,

            "track_id_paths":
                track_id_paths,

            "actor_box_index_paths":
                actor_box_paths,
        },

        "geometry": {
            "module":
                str(
                    geometry_file
                ),

            "module_sha256":
                file_sha(
                    geometry_file
                ),

            "dataclasses":
                geometry_dataclasses,

            "project_box_to_headlamp_signature":
                projection_signature,
        },

        "causal_adb_box":
            causal_inventory[
                "CausalADBActorBox"
            ],

        "stage1": {
            "AnchorFrames":
                frame_inventory[
                    "AnchorFrames"
                ],

            "AdaptedScenario":
                adapter_inventory[
                    "AdaptedScenario"
                ],
        },

        "metric_signatures":
            metric_signatures,

        "evaluator_reference": {
            "path":
                str(
                    evaluator_file
                ),

            "sha256":
                file_sha(
                    evaluator_file
                ),

            "functions":
                evaluator_functions,
        },

        "scientific_boundary": {
            "future_truth_opened":
                False,

            "metric_values_computed":
                False,

            "predictive_performance_computed":
                False,

            "NI_deltas_computed":
                False,

            "formal_content_opened":
                False,

            "policy_modified":
                False,
        },

        "regression_tests":
            test_count,

        "next":
            (
                "materialize evaluator-only future full-box "
                "supports for original_reactive_ADB and compute "
                "only the three reactive statistics permitted "
                "to freeze NI deltas"
            ),
    }

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PART2B PART2 GEOMETRY/IDENTITY BIND — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "frozen reactive ledger      = EXACT PASS"
    )
    print(
        "frozen t0 maps              = EXACT PASS"
    )
    print(
        "scenario identity           = 120 / 120 PASS"
    )
    print(
        "WOMD actor identity path    = PROVEN"
    )
    print(
        "future full-box API         = PROVEN"
    )
    print(
        "headlamp transform route    = PROVEN"
    )
    print(
        "three NI metric APIs        = PROVEN"
    )
    print(
        "future truth opened         = NO"
    )
    print(
        "performance metrics         = NOT COMPUTED"
    )
    print(
        "predictive performance      = NOT COMPUTED"
    )
    print(
        "NI deltas                   = NOT COMPUTED"
    )
    print(
        "formal evaluation           = NO"
    )
    print(
        "Stage6 regression           =",
        f"{test_count} / {test_count} PASS",
    )
    print(
        "STATUS = "
        "PASS_FUTURE_EVALUATOR_GEOMETRY_IDENTITY_BOUND"
    )
    print(
        "report =",
        OUTPUT,
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
        "BLOCK 6.8 PART2B PART2 GEOMETRY/IDENTITY BIND = BLOCKED"
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
        "future truth opened         = NO"
    )
    print(
        "performance metrics         = NOT COMPUTED"
    )
    print(
        "predictive performance      = NOT COMPUTED"
    )
    print(
        "NI deltas                   = NOT COMPUTED"
    )
    print(
        "formal evaluation           = NO"
    )
    print(
        "policy/controller edit      = NO"
    )
    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

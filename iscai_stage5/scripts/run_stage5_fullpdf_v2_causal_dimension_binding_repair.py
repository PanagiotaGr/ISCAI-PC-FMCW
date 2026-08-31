from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S1 = ROOT / "iscai_stage1"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

OLD_FORMAL = (
    S5 / "scripts/run_block58_part2_formal_evaluation.py"
)

CHECKER = (
    S5
    / "scripts/"
    "run_stage5_fullpdf_v2_independent_formal_checker.py"
)

RUNTIME = (
    S5 / "src/iscai_stage5/fullpdf_v2.py"
)

PROTOCOL = (
    S5 / "configs/stage5_fullpdf_v2_protocol.json"
)

RECEIVER_SELECTION = (
    S5 / "src/iscai_stage5/receiver_selection.py"
)

STAGE1_ADAPTER = (
    S1 / "src/iscai_stage1/actors/womd_adapter.py"
)

STAGE4_LEDGER = (
    S4
    / "artifacts/fullpdf_v2/"
    "formal_prediction_ledger_run1.jsonl"
)

OUTDIR = (
    S5 / "artifacts/fullpdf_v2_repair"
)

BINDING = (
    OUTDIR
    / "selected_receiver_causal_geometry_binding_fullpdf_v2.jsonl"
)

REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_causal_dimension_binding_repair.json"
)

REGRESSION_LOG = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_causal_dimension_binding_regression.log"
)

FORMAL_OUT = (
    S5
    / "artifacts/fullpdf_v2_independent_formal"
)

EXPECTED_TESTS = 271

EXPECTED_HASHES = {
    OLD_FORMAL:
        "16495f3c46ed5f95145918b25b169f8aa33408b2ef84ffc19cb66af1db3c189a",

    CHECKER:
        "0e8bd2570453fdc4a92e3fab065f9726cb4d097b334e6b262959c54b0dd9d902",

    RUNTIME:
        "e57708e9332bb40e84f18270e483d91165e26c149b80e82c8f10b4df44376249",

    PROTOCOL:
        "91dec3cc3d3a7b72385f012d1d2bfdb8981a497e1344f585d38f3e05f2e4fa5a",

    RECEIVER_SELECTION:
        "c9a8e1af88caa32bd4dccbe244c6e453af2ff5d184828d5b8b5f5f49fdb1cc7e",

    STAGE1_ADAPTER:
        "6a6eec7b8642b22304fb5dfe51201b1ef9411efc6164bcba1ed60c90b7cf4c09",

    STAGE4_LEDGER:
        "2dd3c396d5c365677c41b3a0df6344a7c28d423daaa368dbe466bd4b17b4c2a8",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("FAIL_CLOSED: " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not load module spec: {path}",
    )

    module = importlib.util.module_from_spec(spec)

    # Required by Python 3.13 dataclasses.
    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


def read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            require(
                isinstance(row, dict),
                f"Non-object JSONL row in {path}",
            )

            rows.append(row)

    return rows


def write_jsonl_atomic(
    path: Path,
    rows: list[dict],
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    tmp.replace(path)


def write_json_atomic(
    path: Path,
    value: dict,
) -> None:
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(path)


def finite_position(
    value: Any,
) -> np.ndarray:
    arr = np.asarray(
        value,
        dtype=np.float64,
    )

    require(
        arr.shape == (3,),
        "H0 position is not shape (3,).",
    )

    require(
        bool(np.all(np.isfinite(arr))),
        "H0 position contains non-finite values.",
    )

    return arr


def is_vehicle(value: Any) -> bool:
    return (
        str(value)
        .strip()
        .upper()
        == "TYPE_VEHICLE"
    )


def verify_frozen_hashes() -> None:
    for path, expected in EXPECTED_HASHES.items():
        require(
            path.is_file(),
            f"Missing frozen dependency: {path}",
        )

        actual = sha256_file(path)

        require(
            actual == expected,
            (
                f"Frozen hash mismatch: {path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )


def verify_preformal_boundary() -> None:
    protected = (
        FORMAL_OUT / "decision_pass_run1.jsonl",
        FORMAL_OUT / "decision_pass_run2.jsonl",
        FORMAL_OUT / "decision_ledger_sealed.jsonl",
        FORMAL_OUT / "decision_ledger_seal.json",
        FORMAL_OUT / "formal_evaluator_rows.jsonl",
        FORMAL_OUT
        / "stage5_to_stage6_handoff_fullpdf_v2.json",
    )

    existing = [
        str(path)
        for path in protected
        if path.exists()
    ]

    require(
        not existing,
        (
            "Formal artifacts already exist; "
            "repair must remain pre-truth: "
            + repr(existing)
        ),
    )

    prior_report = (
        S5
        / "reports/"
        "stage5_fullpdf_v2_independent_formal_checker.json"
    )

    if prior_report.is_file():
        prior = read_json(prior_report)

        require(
            prior.get(
                "formal_evaluator_truth_opened"
            )
            is False,
            "Prior checker says formal truth was opened.",
        )

        require(
            prior.get(
                "formal_outcomes_read"
            )
            is False,
            "Prior checker says formal outcomes were read.",
        )

        require(
            prior.get(
                "decision_ledger_sealed"
            )
            is False,
            "Prior checker says decision ledger was sealed.",
        )


def stage1_current_vehicle_boxes(
    *,
    scenario,
    adapted,
) -> list[dict]:
    from iscai_stage1.actors.womd_adapter import (
        object_type_name,
    )

    anchor = int(
        scenario.current_time_index
    )

    require(
        int(adapted.anchor_index) == anchor,
        (
            "Stage1 adapted anchor differs from "
            "scenario.current_time_index."
        ),
    )

    T_H0_from_W = (
        adapted.frames.T_H0_from_W
    )

    result: list[dict] = []

    for adapted_actor in adapted.actors:
        track_index = int(
            adapted_actor.track_index
        )

        if (
            track_index
            ==
            int(scenario.sdc_track_index)
        ):
            continue

        require(
            0 <= track_index < len(scenario.tracks),
            "Invalid Stage1 causal track index.",
        )

        track = scenario.tracks[
            track_index
        ]

        actor_class = str(
            object_type_name(track)
        )

        if not is_vehicle(actor_class):
            continue

        require(
            anchor < len(track.states),
            "Anchor exceeds track state sequence.",
        )

        state = track.states[
            anchor
        ]

        if not bool(state.valid):
            continue

        length = float(state.length)
        width = float(state.width)
        height = float(state.height)

        require(
            all(
                math.isfinite(v)
                and v > 0.0
                for v in (
                    length,
                    width,
                    height,
                )
            ),
            "Anchor-valid vehicle has invalid L/W/H.",
        )

        center_h0 = finite_position(
            T_H0_from_W.apply_point(
                (
                    float(state.center_x),
                    float(state.center_y),
                    float(state.center_z),
                )
            )
        )

        result.append(
            {
                "track_index":
                    track_index,

                "actor_class":
                    actor_class,

                "center_h0":
                    center_h0,

                "length_m":
                    length,

                "width_m":
                    width,

                "height_m":
                    height,
            }
        )

    return result


def bind_selected_receiver_box(
    *,
    projection,
    selected,
    vehicle_boxes: list[dict],
) -> tuple[dict, float]:
    require(
        vehicle_boxes,
        "No anchor-valid causal vehicle boxes.",
    )

    selected_pid = str(
        selected.prediction_id
    )

    selected_position = finite_position(
        selected.history.latest_position_H0_m
    )

    # Selected receiver -> nearest current causal Stage1 vehicle.
    forward = []

    for box in vehicle_boxes:
        distance = float(
            np.linalg.norm(
                selected_position
                -
                box["center_h0"]
            )
        )

        forward.append(
            (
                distance,
                int(box["track_index"]),
                box,
            )
        )

    forward.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    require(
        len(forward) >= 1,
        "No current vehicle association candidate.",
    )

    if len(forward) > 1:
        require(
            forward[0][0]
            != forward[1][0],
            (
                "Exact selected-receiver -> "
                "Stage1-box distance tie."
            ),
        )

    chosen = forward[0][2]
    chosen_distance = float(
        forward[0][0]
    )

    # Reverse check:
    # chosen Stage1 current vehicle -> selected causal history.
    vehicle_rows = [
        row
        for row in projection["rows"]
        if is_vehicle(
            row.semantic_class
        )
    ]

    require(
        vehicle_rows,
        "No causal Stage5 vehicle histories.",
    )

    reverse = []

    for row in vehicle_rows:
        position = finite_position(
            row.history.latest_position_H0_m
        )

        distance = float(
            np.linalg.norm(
                position
                -
                chosen["center_h0"]
            )
        )

        reverse.append(
            (
                distance,
                str(row.prediction_id),
            )
        )

    reverse.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    if len(reverse) > 1:
        require(
            reverse[0][0]
            != reverse[1][0],
            (
                "Exact Stage1-box -> "
                "causal-history distance tie."
            ),
        )

    require(
        reverse[0][1] == selected_pid,
        (
            "Receiver/Stage1 geometry association is "
            "not reciprocal unique nearest."
        ),
    )

    return chosen, chosen_distance


def run_regression() -> dict:
    print()
    print(
        "===== FULL STAGE5 REGRESSION =====",
        flush=True,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S5 / "tests"),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=str(S5),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    REGRESSION_LOG.write_text(
        proc.stdout,
        encoding="utf-8",
    )

    # Keep terminal readable.
    tail = proc.stdout.splitlines()[-12:]

    for line in tail:
        print(line)

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    count = (
        int(match.group(1))
        if match
        else None
    )

    require(
        proc.returncode == 0,
        (
            "Stage5 regression failed. "
            f"See {REGRESSION_LOG}"
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Regression count {count} "
            f"!= {EXPECTED_TESTS}."
        ),
    )

    return {
        "returncode":
            proc.returncode,

        "tests_passed":
            count,

        "tests_expected":
            EXPECTED_TESTS,

        "log":
            str(REGRESSION_LOG),

        "log_sha256":
            sha256_file(
                REGRESSION_LOG
            ),
    }


def main() -> None:
    print(
        "STAGE5 FULLPDF-V2 CLEAN PREFORMAL REPAIR",
        flush=True,
    )

    print(
        "NO TRAINING / NO RECALIBRATION / "
        "NO FUTURE TRUTH",
        flush=True,
    )

    verify_preformal_boundary()
    verify_frozen_hashes()

    formal = load_module(
        OLD_FORMAL,
        "_stage5_frozen_formal_for_causal_binding",
    )

    parameters = (
        formal.frozen_stage5_parameters()
    )

    receiver_config = (
        parameters["receiver_config"]
    )

    low_speed_threshold = float(
        parameters[
            "low_speed_threshold_mps"
        ]
    )

    clean_config, degraded_config = (
        formal.load_frozen_stage2_configs()
    )

    formal_rows = list(
        formal.read_formal_rows()
    )

    require(
        len(formal_rows) == 120,
        (
            "Frozen formal manifest is not "
            "N=120."
        ),
    )

    validation_rows = list(
        formal.read_validation_manifest(
            formal.VALIDATION_MANIFEST
        )
    )

    validation_by_id = {
        str(row.scenario_id):
            row
        for row in validation_rows
    }

    stage4_rows = read_jsonl(
        STAGE4_LEDGER
    )

    ledger_keys = {
        (
            str(row["scenario_id"]),
            str(row["prediction_id"]),
        )
        for row in stage4_rows
    }

    ledger_scene_ids = {
        sid
        for sid, _ in ledger_keys
    }

    formal_scene_ids = {
        str(row["scenario_id"])
        for row in formal_rows
    }

    require(
        ledger_scene_ids == formal_scene_ids,
        (
            "Stage4 ledger scene cohort does not "
            "equal frozen Stage5 formal cohort."
        ),
    )

    binding_rows: list[dict] = []
    association_distances: list[float] = []

    no_eligible_receiver = 0
    selected_prediction_unavailable = 0

    for rank, formal_row in enumerate(
        formal_rows,
        start=1,
    ):
        scenario_id = str(
            formal_row["scenario_id"]
        )

        validation_row = (
            validation_by_id.get(
                scenario_id
            )
        )

        require(
            validation_row is not None,
            (
                "Formal scenario absent from "
                f"validation manifest: {scenario_id}"
            ),
        )

        require(
            str(validation_row.source_shard)
            ==
            str(formal_row["source_shard"]),
            (
                "Source-shard mismatch for "
                f"{scenario_id}"
            ),
        )

        require(
            str(validation_row.selection_hash)
            ==
            str(formal_row["selection_hash"]),
            (
                "Selection-hash mismatch for "
                f"{scenario_id}"
            ),
        )

        scenario = (
            formal.read_motion_scenario(
                validation_row,
                paired_root=(
                    formal.PAIRED_ROOT
                ),
                compact_record_offset=int(
                    formal_row[
                        "compact_record_offset"
                    ]
                ),
            )
        )

        require(
            str(scenario.scenario_id)
            ==
            scenario_id,
            "Physical scenario ID mismatch.",
        )

        current_index = int(
            scenario.current_time_index
        )

        require(
            current_index == 10,
            (
                "Frozen WOMD current_time_index "
                "is not 10."
            ),
        )

        built = (
            formal.build_real_causal_inputs(
                scenario,
                clean_config=clean_config,
                degraded_config=(
                    degraded_config
                ),
            )
        )

        require(
            bool(
                getattr(
                    built["degraded"],
                    "algorithm_input_truth_free",
                    False,
                )
            ),
            (
                "Current degraded observation "
                "is not truth-free."
            ),
        )

        # IMPORTANT:
        # samples=() intentionally.
        #
        # Receiver selection uses the complete current causal
        # history set + current truth-free degraded classes.
        # We do NOT call attach_supervision here, so this repair
        # never requests future labels/states.
        projection = (
            formal.project_causal_controller_samples(
                built["scene_inputs"],
                (),
                current_time_index=current_index,
                degraded=built["degraded"],
            )
        )

        require(
            int(
                projection.get(
                    "supervised_class_bridge_checks",
                    0,
                )
            )
            == 0,
            (
                "Unexpected supervised bridge use "
                "during preformal geometry repair."
            ),
        )

        receiver_result, selected = (
            formal.choose_primary_receiver(
                projection,
                receiver_config,
            )
        )

        if selected is None:
            no_eligible_receiver += 1
            continue

        prediction_id = str(
            selected.prediction_id
        )

        # No substitution. If frozen Stage4 has no prediction
        # for the causally selected receiver, it remains unavailable.
        if (
            scenario_id,
            prediction_id,
        ) not in ledger_keys:
            selected_prediction_unavailable += 1
            continue

        vehicle_boxes = (
            stage1_current_vehicle_boxes(
                scenario=scenario,
                adapted=built["adapted"],
            )
        )

        box, association_distance = (
            bind_selected_receiver_box(
                projection=projection,
                selected=selected,
                vehicle_boxes=vehicle_boxes,
            )
        )

        heading_h0_rad = float(
            formal.latest_history_heading(
                selected.history,
                low_speed_threshold,
            )
        )

        require(
            math.isfinite(
                heading_h0_rad
            ),
            "Frozen causal heading is non-finite.",
        )

        # Minimal schema only.
        # No truth/future/evaluator fields are placed in this
        # causal binding artifact.
        binding_rows.append(
            {
                "scenario_id":
                    scenario_id,

                "prediction_id":
                    prediction_id,

                "heading_h0_rad":
                    heading_h0_rad,

                "length_m":
                    float(box["length_m"]),

                "width_m":
                    float(box["width_m"]),

                "height_m":
                    float(box["height_m"]),

                "actor_class":
                    "TYPE_VEHICLE",

                "selected_receiver":
                    True,
            }
        )

        association_distances.append(
            association_distance
        )

        if (
            rank % 20 == 0
            or rank == len(formal_rows)
        ):
            print(
                f"causal binding progress: "
                f"{rank}/{len(formal_rows)}",
                flush=True,
            )

    require(
        binding_rows,
        (
            "No selected causal receiver has a "
            "frozen Stage4 prediction."
        ),
    )

    by_scene: dict[str, list[dict]] = (
        defaultdict(list)
    )

    for row in binding_rows:
        by_scene[
            row["scenario_id"]
        ].append(row)

    require(
        all(
            len(rows) == 1
            for rows in by_scene.values()
        ),
        (
            "More than one selected receiver row "
            "exists for a represented scenario."
        ),
    )

    keys = {
        (
            row["scenario_id"],
            row["prediction_id"],
        )
        for row in binding_rows
    }

    require(
        len(keys) == len(binding_rows),
        "Duplicate causal binding key.",
    )

    regression = run_regression()

    OUTDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Since no formal truth has been opened, replacing a previous
    # PRE-FORMAL repair artifact is safe. It is never silently
    # combined with an old binding.
    if BINDING.exists():
        BINDING.unlink()

    write_jsonl_atomic(
        BINDING,
        binding_rows,
    )

    binding_sha = sha256_file(
        BINDING
    )

    print()
    print(
        "===== INDEPENDENT CHECKER PREFLIGHT =====",
        flush=True,
    )

    checker = load_module(
        CHECKER,
        "_stage5_fullpdf_v2_checker_preflight",
    )

    parsed = [
        checker.parse_causal_row(row)
        for row in binding_rows
    ]

    require(
        all(
            row is not None
            for row in parsed
        ),
        (
            "Checker rejected generated causal "
            "binding schema."
        ),
    )

    resolved_path, resolved_rows = (
        checker.resolve_causal_binding(
            ledger_keys
        )
    )

    resolved_sha = sha256_file(
        resolved_path
    )

    require(
        resolved_sha == binding_sha,
        (
            "Checker resolved a non-equivalent "
            "causal binding artifact."
        ),
    )

    require(
        len(resolved_rows)
        ==
        len(binding_rows),
        (
            "Checker resolved different causal "
            "binding row count."
        ),
    )

    report = {
        "stage":
            5,

        "version":
            "fullpdf_v2",

        "status":
            (
                "PASS_PRE_FORMAL_CAUSAL_RECEIVER_"
                "DIMENSION_BINDING_REPAIR"
            ),

        "Stage5":
            "REPAIRED_AND_FROZEN_PRE_FORMAL",

        "Stage6_allowed":
            False,

        "formal_run_permitted":
            True,

        "formal_evaluator_truth_opened":
            False,

        "formal_outcomes_read":
            False,

        "decision_ledger_sealed":
            False,

        "binding": {
            "path":
                str(BINDING),

            "sha256":
                binding_sha,

            "rows":
                len(binding_rows),

            "association_rule":
                (
                    "reciprocal unique nearest "
                    "current H0 geometry"
                ),

            "acceptance_distance_threshold_m":
                None,

            "maximum_observed_association_distance_m":
                (
                    max(association_distances)
                    if association_distances
                    else None
                ),
        },

        "availability": {
            "formal_scenarios":
                len(formal_rows),

            "scenarios_without_eligible_receiver":
                no_eligible_receiver,

            "selected_receiver_prediction_unavailable":
                selected_prediction_unavailable,

            "binding_rows":
                len(binding_rows),
        },

        "causal_contract": {
            "dimension_source":
                "WOMD current_time_index anchor state",

            "receiver_selection":
                (
                    "unchanged frozen Stage5 "
                    "nearest causal vehicle ahead"
                ),

            "heading_source":
                (
                    "unchanged frozen Stage5 "
                    "latest_history_heading"
                ),

            "supervision_used_for_repair":
                False,

            "default_dimensions_used":
                False,

            "association_threshold_tuned":
                False,
        },

        "regression":
            regression,

        "checker_preflight": {
            "pass":
                True,

            "resolved_path":
                str(resolved_path),

            "resolved_sha256":
                resolved_sha,

            "rows":
                len(resolved_rows),
        },

        "frozen_source_hashes": {
            str(path):
                sha256_file(path)
            for path in EXPECTED_HASHES
        },

        "no_training":
            True,

        "no_retraining":
            True,

        "no_recalibration":
            True,

        "no_parameter_tuning":
            True,

        "no_post_outcome_tuning":
            True,

        "next_action":
            (
                "Independent FullPDF-V2 formal checker "
                "is now eligible for one new formal run."
            ),
    }

    write_json_atomic(
        REPORT,
        report,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "CLEAN PREFORMAL REPAIR = PASS"
    )
    print(
        "============================================================"
    )
    print(
        "formal_scenarios =",
        len(formal_rows),
    )
    print(
        "binding_rows =",
        len(binding_rows),
    )
    print(
        "no_eligible_receiver =",
        no_eligible_receiver,
    )
    print(
        "prediction_unavailable =",
        selected_prediction_unavailable,
    )
    print(
        "regression = 271/271 PASS"
    )
    print(
        "checker_binding_preflight = PASS"
    )
    print(
        "formal_evaluator_truth_opened = False"
    )
    print(
        "formal_run_permitted = True"
    )
    print(
        "READY_FOR_FORMAL=YES"
    )
    print(
        "binding_sha256 =",
        binding_sha,
    )
    print(
        "repair_report_sha256 =",
        sha256_file(REPORT),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys


ROOT = Path(
    "/home/agni/waymo"
)

STAGE5 = (
    ROOT
    /
    "iscai_stage5"
)

for source_root in (
    ROOT / "iscai_stage5/src",
    ROOT / "iscai_stage4/src",
    ROOT / "iscai_stage3/src",
    ROOT / "iscai_stage2/src",
    ROOT / "iscai_stage1/src",
    ROOT / "iscai_stage0/src",
):
    sys.path.insert(
        0,
        str(
            source_root
        ),
    )


from iscai_stage3.validation import (
    read_motion_scenario,
    read_validation_manifest,
)

from iscai_stage4.data import (
    attach_supervision,
)

from iscai_stage4.data.real_pipeline import (
    build_real_causal_inputs,
    load_frozen_stage2_configs,
)

from iscai_stage4.ml.formal_runtime import (
    resolve_sample_truth_track_index,
)


FORMAL_MANIFEST = (
    ROOT
    /
    "iscai_stage3/artifacts/block38e/"
    "formal_validation_120.jsonl"
)

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

VALIDATION_MANIFEST = (
    ROOT
    /
    "iscai_data_prep/manifests/"
    "selected_validation.jsonl"
)

PAIRED_ROOT = (
    ROOT
    /
    "data/"
    "paired_womd_lidar_v1_3_0"
)

REPORT = (
    STAGE5
    /
    "artifacts/block58/"
    "route_b_one_scene_schema_probe.json"
)


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def read_formal_rows():
    rows = tuple(
        json.loads(
            line
        )
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    if len(
        rows
    ) != 120:
        raise RuntimeError(
            "Formal manifest is not N=120."
        )

    return rows


def public_fields(
    value,
):
    if value is None:
        return []

    if is_dataclass(
        value
    ):
        return [
            field.name
            for field in fields(
                value
            )
        ]

    descriptor = getattr(
        value,
        "DESCRIPTOR",
        None,
    )

    if (
        descriptor is not None
        and
        hasattr(
            descriptor,
            "fields",
        )
    ):
        return [
            field.name
            for field in descriptor.fields
        ]

    dictionary = getattr(
        value,
        "__dict__",
        None,
    )

    if isinstance(
        dictionary,
        dict,
    ):
        return sorted(
            str(key)
            for key in dictionary
            if not str(
                key
            ).startswith(
                "_"
            )
        )

    slots = getattr(
        type(value),
        "__slots__",
        (),
    )

    if isinstance(
        slots,
        str,
    ):
        slots = (
            slots,
        )

    return sorted(
        str(name)
        for name in slots
        if not str(
            name
        ).startswith(
            "_"
        )
    )


def type_name(
    value,
):
    return (
        f"{type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def atomic_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        +
        "\n"
    )

    temporary = (
        path
        .with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_text(
        data,
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def main():
    if (
        file_sha256(
            FORMAL_MANIFEST
        )
        !=
        EXPECTED_FORMAL_MANIFEST_SHA
    ):
        raise RuntimeError(
            "Formal manifest SHA changed."
        )

    formal_rows = (
        read_formal_rows()
    )

    formal = (
        formal_rows[
            0
        ]
    )

    validation_rows = (
        read_validation_manifest(
            VALIDATION_MANIFEST
        )
    )

    validation_by_id = {
        str(
            row.scenario_id
        ):
            row
        for row in validation_rows
    }

    scenario_id = str(
        formal[
            "scenario_id"
        ]
    )

    validation_row = (
        validation_by_id.get(
            scenario_id
        )
    )

    if validation_row is None:
        raise RuntimeError(
            "Formal scene missing from canonical "
            "validation manifest."
        )

    if (
        str(
            validation_row.source_shard
        )
        !=
        str(
            formal[
                "source_shard"
            ]
        )
    ):
        raise RuntimeError(
            "Formal/canonical source_shard mismatch."
        )

    if (
        str(
            validation_row.selection_hash
        )
        !=
        str(
            formal[
                "selection_hash"
            ]
        )
    ):
        raise RuntimeError(
            "Formal/canonical selection_hash mismatch."
        )

    if (
        str(
            validation_row.split
        )
        !=
        "validation"
    ):
        raise RuntimeError(
            "Formal scene is not validation split."
        )

    scenario = (
        read_motion_scenario(
            validation_row,
            paired_root=(
                PAIRED_ROOT
            ),
            compact_record_offset=int(
                formal[
                    "compact_record_offset"
                ]
            ),
        )
    )

    if (
        str(
            scenario.scenario_id
        )
        !=
        scenario_id
    ):
        raise RuntimeError(
            "Loaded scenario ID mismatch."
        )

    (
        clean_config,
        degraded_config,
    ) = (
        load_frozen_stage2_configs()
    )

    built = (
        build_real_causal_inputs(
            scenario,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
        )
    )

    scene_inputs = (
        built[
            "scene_inputs"
        ]
    )

    samples = (
        attach_supervision(
            scene_inputs,
            scenario,
            T_H0_from_W=(
                built[
                    "adapted"
                ]
                .frames
                .T_H0_from_W
            ),
        )
    )

    histories = tuple(
        scene_inputs.histories
    )

    history_ids = [
        str(
            history.prediction_id
        )
        for history in histories
    ]

    sample_ids = [
        str(
            sample.prediction_id
        )
        for sample in samples
    ]

    if (
        len(
            history_ids
        )
        !=
        len(
            set(
                history_ids
            )
        )
    ):
        raise RuntimeError(
            "Causal history prediction IDs "
            "are not unique."
        )

    if (
        len(
            sample_ids
        )
        !=
        len(
            set(
                sample_ids
            )
        )
    ):
        raise RuntimeError(
            "Supervised sample prediction IDs "
            "are not unique."
        )

    history_id_set = set(
        history_ids
    )

    unmatched_sample_ids = [
        value
        for value in sample_ids
        if value not in history_id_set
    ]

    if unmatched_sample_ids:
        raise RuntimeError(
            "Supervised samples cannot be bridged "
            "to causal histories by prediction_id: "
            f"{unmatched_sample_ids[:5]}"
        )

    history_fields = (
        public_fields(
            histories[
                0
            ]
        )
        if histories
        else
        []
    )

    first_history = (
        histories[
            0
        ]
        if histories
        else
        None
    )

    first_step = None

    if (
        first_history is not None
        and
        first_history.steps
    ):
        index = int(
            first_history
            .latest_observed_frame_index
        )

        if (
            0
            <=
            index
            <
            len(
                first_history.steps
            )
        ):
            first_step = (
                first_history
                .steps[
                    index
                ]
            )
        else:
            first_step = (
                first_history
                .steps[
                    -1
                ]
            )

    sample_fields = (
        public_fields(
            samples[
                0
            ]
        )
        if samples
        else
        []
    )

    future_label = (
        getattr(
            samples[
                0
            ],
            "future_label",
            None,
        )
        if samples
        else
        None
    )

    step_fields = (
        public_fields(
            first_step
        )
    )

    class_field_hints = [
        name
        for name in step_fields
        if any(
            token
            in
            name.lower()
            for token in (
                "class",
                "type",
                "semantic",
                "category",
            )
        )
    ]

    geometry_field_hints = [
        name
        for name in step_fields
        if any(
            token
            in
            name.lower()
            for token in (
                "position",
                "observed",
                "valid",
                "frame",
            )
        )
    ]

    sample_truth_preview = []

    for sample in samples[
        : min(
            5,
            len(
                samples
            ),
        )
    ]:
        resolved = (
            resolve_sample_truth_track_index(
                sample
            )
        )

        sample_truth_preview.append(
            {
                "prediction_id":
                    str(
                        sample.prediction_id
                    ),

                "truth_track_index":
                    int(
                        resolved[
                            "index"
                        ]
                    ),

                "source":
                    str(
                        resolved[
                            "source"
                        ]
                    ),
            }
        )

    position_preview = []

    for history in histories[
        : min(
            5,
            len(
                histories
            ),
        )
    ]:
        position = tuple(
            float(
                value
            )
            for value in (
                history
                .latest_position_H0_m
            )
        )

        if (
            len(
                position
            )
            != 3
            or
            not all(
                isfinite(
                    value
                )
                for value in position
            )
        ):
            raise RuntimeError(
                "Invalid causal H0 position."
            )

        position_preview.append(
            {
                "prediction_id":
                    str(
                        history.prediction_id
                    ),

                "latest_position_H0_m":
                    list(
                        position
                    ),
            }
        )

    report = {
        "stage":
            5,

        "block":
            "5.8",

        "phase":
            "route_B_one_scene_schema_probe",

        "status":
            "PASS_SCHEMA_PROBE",

        "formal_manifest": {
            "path":
                str(
                    FORMAL_MANIFEST
                ),

            "sha256":
                EXPECTED_FORMAL_MANIFEST_SHA,

            "N":
                120,

            "scenes_physically_loaded":
                1,
        },

        "scenario": {
            "scenario_id":
                scenario_id,

            "current_time_index":
                int(
                    scenario.current_time_index
                ),

            "causal_history_count":
                len(
                    histories
                ),

            "supervised_sample_count":
                len(
                    samples
                ),
        },

        "identity_bridge": {
            "history_key":
                "prediction_id",

            "sample_key":
                "prediction_id",

            "history_ids_unique":
                True,

            "sample_ids_unique":
                True,

            "every_sample_prediction_id_in_histories":
                True,

            "unmatched_sample_ids":
                [],

            "overlap_count":
                len(
                    set(
                        sample_ids
                    )
                    &
                    history_id_set
                ),

            "truth_mapping_preview":
                sample_truth_preview,
        },

        "schema": {
            "history_type":
                (
                    type_name(
                        first_history
                    )
                    if first_history
                    is not None
                    else
                    None
                ),

            "history_fields":
                history_fields,

            "history_step_type":
                (
                    type_name(
                        first_step
                    )
                    if first_step
                    is not None
                    else
                    None
                ),

            "history_step_fields":
                step_fields,

            "history_step_class_field_hints":
                class_field_hints,

            "history_step_geometry_field_hints":
                geometry_field_hints,

            "sample_type":
                (
                    type_name(
                        samples[
                            0
                        ]
                    )
                    if samples
                    else
                    None
                ),

            "sample_fields":
                sample_fields,

            "future_label_type":
                (
                    type_name(
                        future_label
                    )
                    if future_label
                    is not None
                    else
                    None
                ),

            "future_label_fields":
                public_fields(
                    future_label
                ),
        },

        "causal_H0_position_preview":
            position_preview,

        "scientific_execution": {
            "model_forward":
                False,

            "Stage4_inference":
                False,

            "Stage5_formal_metrics":
                False,

            "tracks_to_predict_accessed":
                False,

            "eligible_formal_targets_called":
                False,

            "receiver_selection_performed":
                False,

            "future_truth_control_input":
                False,

            "future_supervision_attached_for_schema_mapping_only":
                True,

            "training":
                False,

            "recalibration":
                False,

            "model_selection":
                False,

            "threshold_tuning":
                False,
        },

        "next":
            (
                "write deterministic Route-B pre-TTP "
                "formal Gaussian materializer using "
                "the proven prediction_id bridge"
            ),
    }

    atomic_json(
        REPORT,
        report,
    )

    print(
        "============================================================"
    )
    print(
        "BLOCK 5.8 PART 2/2 — ROUTE-B ONE-SCENE SCHEMA PROBE"
    )
    print(
        "============================================================"
    )
    print(
        "formal manifest            = N=120 / SHA PASS"
    )
    print(
        "formal scenes loaded        = 1"
    )
    print(
        "scenario                    =",
        scenario_id,
    )
    print(
        "causal histories            =",
        len(
            histories
        ),
    )
    print(
        "supervised samples          =",
        len(
            samples
        ),
    )
    print(
        "prediction_id bridge        = PASS"
    )
    print(
        "history fields              =",
        history_fields,
    )
    print(
        "latest-step fields          =",
        step_fields,
    )
    print(
        "class-field hints           =",
        class_field_hints,
    )
    print(
        "sample fields               =",
        sample_fields,
    )
    print(
        "future-label fields         =",
        public_fields(
            future_label
        ),
    )
    print(
        "model forward               = NO"
    )
    print(
        "tracks_to_predict access    = NO"
    )
    print(
        "formal Stage5 metrics       = NOT COMPUTED"
    )
    print(
        "STATUS                      = PASS_SCHEMA_PROBE"
    )
    print(
        "report                      =",
        REPORT,
    )


if __name__ == "__main__":
    main()

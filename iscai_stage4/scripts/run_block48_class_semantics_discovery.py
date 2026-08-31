from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import traceback

import numpy as np

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

from iscai_stage4.ml.calibration_runtime import (
    samples_to_arrays,
)

from iscai_stage4.ml.formal_runtime import (
    resolve_sample_truth_track_index,
)

from iscai_stage4.ml.training_utils import (
    CLASS_ID_TO_NAME,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

PAIRED_ROOT = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

REPORT = (
    STAGE4
    / "reports/"
      "block48_class_semantics_discovery.json"
)


def write_json(
    payload,
):
    REPORT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


def canonical_name(
    value,
):
    text = (
        str(value)
        .strip()
        .upper()
    )

    if text.startswith(
        "TYPE_"
    ):
        text = text[
            len(
                "TYPE_"
            ):
        ]

    return text


def womd_enum_name(
    track,
):
    """
    Resolve symbolic protobuf enum name such as
    TYPE_VEHICLE without hard-coding numeric values.
    """

    code = int(
        track.object_type
    )

    try:
        field = (
            track.DESCRIPTOR
            .fields_by_name[
                "object_type"
            ]
        )

        enum_type = (
            field.enum_type
        )

        value = (
            enum_type
            .values_by_number[
                code
            ]
        )

        return value.name

    except Exception:
        pass

    # Descriptor fallback for protobuf variants.
    try:
        for field in (
            track.DESCRIPTOR.fields
        ):
            if (
                field.name
                !=
                "object_type"
            ):
                continue

            enum_type = (
                field.enum_type
            )

            for value in (
                enum_type.values
            ):
                if (
                    int(
                        value.number
                    )
                    ==
                    code
                ):
                    return value.name

    except Exception:
        pass

    return (
        f"<UNKNOWN_ENUM_{code}>"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 — CLASS SEMANTICS DISCOVERY"
    )
    print(
        "READ ONLY / NO NEURAL INFERENCE"
    )
    print(
        "============================================================"
    )

    required = (
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        print(
            "STATUS = BLOCKED"
        )
        print(
            "reason = missing required file(s):",
            missing,
        )
        return

    formal_rows = tuple(
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

    print()
    print(
        "===== A. INTERNAL STAGE4 CLASS MAPPING ====="
    )

    normalized_mapping = {}

    for key, value in sorted(
        CLASS_ID_TO_NAME.items(),
        key=lambda item:
            int(
                item[
                    0
                ]
            ),
    ):
        normalized_mapping[
            int(
                key
            )
        ] = str(
            value
        )

        print(
            "Stage4 class_id",
            int(
                key
            ),
            "=>",
            repr(
                value
            ),
        )

    (
        clean_config,
        degraded_config,
    ) = load_frozen_stage2_configs()

    records = []

    numeric_pair_counts = Counter()
    semantic_pair_counts = Counter()

    semantic_mismatches = []

    scenarios_examined = 0
    supervised_samples_examined = 0

    truth_index_sources = Counter()

    # A few scenes are enough to establish whether the two
    # class representations use different integer domains.
    # No tracks_to_predict access and no neural model call.
    for formal in formal_rows[
        :10
    ]:
        scenario_id = str(
            formal[
                "scenario_id"
            ]
        )

        row = validation_by_id.get(
            scenario_id
        )

        if row is None:
            raise RuntimeError(
                "Formal scenario missing from "
                "canonical validation catalog: "
                f"{scenario_id}"
            )

        scenario = read_motion_scenario(
            row,
            paired_root=PAIRED_ROOT,
            compact_record_offset=int(
                formal[
                    "compact_record_offset"
                ]
            ),
        )

        built = build_real_causal_inputs(
            scenario,
            clean_config=clean_config,
            degraded_config=degraded_config,
        )

        samples = attach_supervision(
            built[
                "scene_inputs"
            ],
            scenario,
            T_H0_from_W=(
                built[
                    "adapted"
                ]
                .frames
                .T_H0_from_W
            ),
        )

        scenarios_examined += 1

        if not samples:
            continue

        arrays = samples_to_arrays(
            samples
        )

        class_ids = np.asarray(
            arrays[
                "class_id"
            ],
            dtype=np.int64,
        )

        if len(
            class_ids
        ) != len(
            samples
        ):
            raise RuntimeError(
                "samples_to_arrays class_id length "
                "does not match sample count."
            )

        for local_index, sample in enumerate(
            samples
        ):
            resolved = (
                resolve_sample_truth_track_index(
                    sample
                )
            )

            truth_index = int(
                resolved[
                    "index"
                ]
            )

            truth_index_sources[
                resolved[
                    "source"
                ]
            ] += 1

            if (
                truth_index < 0
                or
                truth_index
                >=
                len(
                    scenario.tracks
                )
            ):
                raise RuntimeError(
                    "Resolved truth index is "
                    "outside WOMD track range."
                )

            internal_id = int(
                class_ids[
                    local_index
                ]
            )

            internal_name = (
                normalized_mapping.get(
                    internal_id,
                    f"<UNKNOWN_INTERNAL_{internal_id}>",
                )
            )

            track = scenario.tracks[
                truth_index
            ]

            womd_id = int(
                track.object_type
            )

            womd_name = (
                womd_enum_name(
                    track
                )
            )

            internal_semantic = (
                canonical_name(
                    internal_name
                )
            )

            womd_semantic = (
                canonical_name(
                    womd_name
                )
            )

            semantic_match = (
                internal_semantic
                ==
                womd_semantic
            )

            numeric_match = (
                internal_id
                ==
                womd_id
            )

            numeric_pair_counts[
                (
                    internal_id,
                    womd_id,
                )
            ] += 1

            semantic_pair_counts[
                (
                    internal_semantic,
                    womd_semantic,
                )
            ] += 1

            record = {
                "scenario_id":
                    scenario_id,

                "local_sample_index":
                    int(
                        local_index
                    ),

                "truth_track_index":
                    truth_index,

                "truth_index_source":
                    resolved[
                        "source"
                    ],

                "Stage4_class_id":
                    internal_id,

                "Stage4_class_name":
                    internal_name,

                "WOMD_object_type":
                    womd_id,

                "WOMD_object_type_name":
                    womd_name,

                "numeric_match":
                    bool(
                        numeric_match
                    ),

                "semantic_match":
                    bool(
                        semantic_match
                    ),
            }

            records.append(
                record
            )

            if not semantic_match:
                semantic_mismatches.append(
                    record
                )

            supervised_samples_examined += 1

        if (
            supervised_samples_examined
            >=
            100
        ):
            break

    if (
        supervised_samples_examined
        ==
        0
    ):
        raise RuntimeError(
            "No supervised samples found "
            "in discovery scenes."
        )

    print()
    print(
        "===== B. NUMERIC ENCODING PAIRS ====="
    )

    for (
        internal_id,
        womd_id,
    ), count in sorted(
        numeric_pair_counts.items()
    ):
        print(
            "Stage4",
            internal_id,
            "=> WOMD",
            womd_id,
            "| count =",
            count,
        )

    print()
    print(
        "===== C. SEMANTIC CLASS PAIRS ====="
    )

    for (
        internal_name,
        womd_name,
    ), count in sorted(
        semantic_pair_counts.items()
    ):
        print(
            internal_name,
            "=>",
            womd_name,
            "| count =",
            count,
        )

    print()
    print(
        "===== D. FIRST 25 SAMPLE COMPARISONS ====="
    )

    for record in records[
        :25
    ]:
        print(
            "scenario=",
            record[
                "scenario_id"
            ],
            "| truth=",
            record[
                "truth_track_index"
            ],
            "| source=",
            record[
                "truth_index_source"
            ],
            "| Stage4=",
            (
                f"{record['Stage4_class_id']}:"
                f"{record['Stage4_class_name']}"
            ),
            "| WOMD=",
            (
                f"{record['WOMD_object_type']}:"
                f"{record['WOMD_object_type_name']}"
            ),
            "| numeric=",
            record[
                "numeric_match"
            ],
            "| semantic=",
            record[
                "semantic_match"
            ],
        )

    numeric_mismatch_count = sum(
        1
        for item in records
        if not item[
            "numeric_match"
        ]
    )

    semantic_mismatch_count = len(
        semantic_mismatches
    )

    all_semantics_match = (
        semantic_mismatch_count
        ==
        0
    )

    has_numeric_encoding_difference = (
        numeric_mismatch_count
        >
        0
    )

    if (
        all_semantics_match
        and
        has_numeric_encoding_difference
    ):
        diagnosis = (
            "NUMERIC_ENCODING_DIFFERENCE_ONLY"
        )

    elif (
        all_semantics_match
        and
        not
        has_numeric_encoding_difference
    ):
        diagnosis = (
            "NUMERIC_AND_SEMANTIC_ENCODINGS_MATCH"
        )

    else:
        diagnosis = (
            "TRUE_SEMANTIC_CLASS_MISMATCH_PRESENT"
        )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 CLASS SEMANTICS DIAGNOSIS"
    )
    print(
        "============================================================"
    )

    print(
        "scenarios examined        =",
        scenarios_examined,
    )

    print(
        "supervised samples        =",
        supervised_samples_examined,
    )

    print(
        "truth-index sources       =",
        dict(
            truth_index_sources
        ),
    )

    print(
        "numeric mismatch count    =",
        numeric_mismatch_count,
    )

    print(
        "semantic mismatch count   =",
        semantic_mismatch_count,
    )

    print(
        "all semantic classes match=",
        (
            "PASS"
            if all_semantics_match
            else
            "NO"
        ),
    )

    print(
        "diagnosis                 =",
        diagnosis,
    )

    if semantic_mismatches:
        print()
        print(
            "FIRST SEMANTIC MISMATCHES:"
        )

        for item in (
            semantic_mismatches[
                :20
            ]
        ):
            print(
                json.dumps(
                    item,
                    sort_keys=True,
                )
            )

    write_json({
        "stage":
            4,

        "block":
            "4.8_class_semantics_discovery",

        "status":
            "PASS",

        "purpose":
            (
                "determine_whether_Stage4_class_id_"
                "and_WOMD_object_type_share_the_"
                "same_numeric_encoding"
            ),

        "Stage4_CLASS_ID_TO_NAME":
            normalized_mapping,

        "scenarios_examined":
            scenarios_examined,

        "supervised_samples_examined":
            supervised_samples_examined,

        "truth_index_sources":
            dict(
                truth_index_sources
            ),

        "numeric_pair_counts": [
            {
                "Stage4_class_id":
                    int(
                        internal_id
                    ),

                "WOMD_object_type":
                    int(
                        womd_id
                    ),

                "count":
                    int(
                        count
                    ),
            }
            for (
                internal_id,
                womd_id,
            ), count in sorted(
                numeric_pair_counts.items()
            )
        ],

        "semantic_pair_counts": [
            {
                "Stage4":
                    internal_name,

                "WOMD":
                    womd_name,

                "count":
                    int(
                        count
                    ),
            }
            for (
                internal_name,
                womd_name,
            ), count in sorted(
                semantic_pair_counts.items()
            )
        ],

        "numeric_mismatch_count":
            numeric_mismatch_count,

        "semantic_mismatch_count":
            semantic_mismatch_count,

        "all_semantic_classes_match":
            all_semantics_match,

        "diagnosis":
            diagnosis,

        "first_semantic_mismatches":
            semantic_mismatches[
                :20
            ],

        "tracks_to_predict_accessed":
            False,

        "neural_inference_started":
            False,

        "training":
            False,

        "recalibration":
            False,

        "upstream_modified":
            False,
    })

    print()
    print(
        "discovery report          =",
        REPORT,
    )

    print(
        "tracks_to_predict accessed= NO"
    )

    print(
        "neural inference started  = NO"
    )

    print(
        "upstream modified         = NO"
    )

    print(
        "DISCOVERY STATUS = PASS"
    )

    print(
        "terminal remains open     = YES"
    )


try:
    main()

except BaseException as exc:
    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 CLASS SEMANTICS DISCOVERY = BLOCKED"
    )
    print(
        "============================================================"
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
    traceback.print_exc()

    print(
        "tracks_to_predict accessed = NO"
    )

    print(
        "neural inference started   = NO"
    )

    print(
        "upstream modified          = NO"
    )

    print(
        "terminal remains open      = YES"
    )

# Deliberately no sys.exit().

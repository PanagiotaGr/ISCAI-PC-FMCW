from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import torch

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
    CalibrationNormalizer,
    samples_to_arrays,
)

from iscai_stage4.ml.formal_runtime import (
    construct_deterministic_model,
    extract_deterministic_prediction,
    resolve_sample_truth_track_index,
    tensor_bundle_sha256,
)

from iscai_stage4.ml.gaussian_gru import (
    GaussianTrajectoryGRU,
)

from iscai_stage4.ml.gmm_gru import (
    GMMTrajectoryGRU,
)

from iscai_stage4.ml.reproducibility_runtime import (
    file_sha256,
    state_dict_sha256,
)

from iscai_stage4.ml.training_utils import (
    set_global_determinism,
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

DET_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_gru.pt"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

GMM_CHECKPOINT = (
    STAGE4
    / "artifacts/block46/"
      "gmm_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CACHE_DIR = (
    STAGE4
    / "artifacts/block48/cache"
)

EXPECTED = {
    "formal_manifest":
        (
            "2208e7287ddf6439fda4597c435a9cba"
            "1d1b9d0e4c4547bc5dd92e56e8124e46"
        ),

    "validation_manifest":
        (
            "dc10609ef18a2ba881657eb3da3a3df7"
            "a81bdcc8345ecbc2227102ab16b8833c"
        ),

    "det_checkpoint":
        (
            "5456a76b84d558e9983a59b9f1d3060b"
            "a245e0d36d60883519654f809996dbc5"
        ),

    "Gaussian_checkpoint":
        (
            "49ff64d145eaa633f295c16f660df380"
            "c35383e7e3b61279a5aad7cd700d619f"
        ),

    "GMM_checkpoint":
        (
            "5aebeaf40d522d58345d424ae2557d6f"
            "87e2c02bdc26c23635cc6e3a28cbe3ee"
        ),

    "det_state":
        (
            "d2ffbc03c7cb2826fef2175c95f48725"
            "707ec6d791eaeffbd6f59bc507b8595a"
        ),

    "Gaussian_state":
        (
            "1d66cf082d0d9be41319f7dcbe18fc259"
            "910de7f45de006f9043129837b86be3"
        ),

    "GMM_state":
        (
            "3bd803104a5de694b9c6073fa651230e"
            "16a08899db14894e0f1fd1e2f11c1a97"
        ),
}


def write_result(
    payload,
):
    result_path = os.environ.get(
        "BLOCK49_PROBE_RESULT_PATH"
    )

    if not result_path:
        raise RuntimeError(
            "BLOCK49_PROBE_RESULT_PATH "
            "is not set."
        )

    path = Path(
        result_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )


def build_gaussian(
    architecture,
    *,
    device,
):
    section = architecture[
        "model"
    ]

    return GaussianTrajectoryGRU(
        target_hidden_dim=int(
            section[
                "target_hidden_dim"
            ]
        ),

        neighbor_hidden_dim=int(
            section[
                "neighbor_hidden_dim"
            ]
        ),

        map_hidden_dim=int(
            section[
                "map_hidden_dim"
            ]
        ),

        fusion_hidden_dim=int(
            section[
                "fusion_hidden_dim"
            ]
        ),

        use_neighbors=True,
        use_map=True,
    ).to(
        device
    )


def build_gmm(
    architecture,
    *,
    device,
):
    section = architecture[
        "model"
    ]

    return GMMTrajectoryGRU(
        target_hidden_dim=int(
            section[
                "target_hidden_dim"
            ]
        ),

        neighbor_hidden_dim=int(
            section[
                "neighbor_hidden_dim"
            ]
        ),

        map_hidden_dim=int(
            section[
                "map_hidden_dim"
            ]
        ),

        fusion_hidden_dim=int(
            section[
                "fusion_hidden_dim"
            ]
        ),

        use_neighbors=True,
        use_map=True,
    ).to(
        device
    )


def parse_rank():
    if len(
        sys.argv
    ) != 2:
        raise RuntimeError(
            "Expected exactly one formal-rank "
            "argument."
        )

    try:
        rank = int(
            sys.argv[
                1
            ]
        )

    except Exception as exc:
        raise RuntimeError(
            "Formal rank is not an integer."
        ) from exc

    if (
        rank < 1
        or
        rank > 120
    ):
        raise RuntimeError(
            "Formal rank must be 1..120."
        )

    return rank


def main():
    rank = parse_rank()

    print(
        "============================================================"
    )
    print(
        "BLOCK 4.9 FRESH-PROCESS "
        f"PROBE — FORMAL RANK {rank}"
    )
    print(
        "============================================================"
    )

    required = (
        FORMAL_MANIFEST,
        VALIDATION_MANIFEST,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        GMM_CHECKPOINT,
        NORMALIZATION,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing frozen artifact(s): "
            +
            ", ".join(
                missing
            )
        )

    file_checks = (
        (
            FORMAL_MANIFEST,
            EXPECTED[
                "formal_manifest"
            ],
        ),
        (
            VALIDATION_MANIFEST,
            EXPECTED[
                "validation_manifest"
            ],
        ),
        (
            DET_CHECKPOINT,
            EXPECTED[
                "det_checkpoint"
            ],
        ),
        (
            GAUSSIAN_CHECKPOINT,
            EXPECTED[
                "Gaussian_checkpoint"
            ],
        ),
        (
            GMM_CHECKPOINT,
            EXPECTED[
                "GMM_checkpoint"
            ],
        ),
    )

    for path, expected in (
        file_checks
    ):
        if (
            file_sha256(
                path
            )
            !=
            expected
        ):
            raise RuntimeError(
                "Frozen artifact SHA changed: "
                f"{path}"
            )

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

    if len(
        formal_rows
    ) != 120:
        raise RuntimeError(
            "Formal manifest is not N=120."
        )

    formal = formal_rows[
        rank - 1
    ]

    scenario_id = str(
        formal[
            "scenario_id"
        ]
    )

    cached_path = (
        CACHE_DIR
        /
        (
            f"{rank:03d}_"
            f"{scenario_id}.npz"
        )
    )

    if not cached_path.is_file():
        raise RuntimeError(
            "Frozen Block4.8 shard "
            "is missing."
        )

    with np.load(
        cached_path,
        allow_pickle=False,
    ) as data:
        expected_prediction_sha = str(
            data[
                "all_prediction_sha256"
            ].item()
        )

        expected_samples = int(
            data[
                "all_supervised_prediction_count"
            ].item()
        )

        expected_truth_source = str(
            data[
                "truth_index_source"
            ].item()
        )

    if expected_samples <= 0:
        raise RuntimeError(
            "Fresh-process probe was assigned "
            "an empty formal scene."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    gpu_name = torch.cuda.get_device_name(
        0
    )

    if (
        "A6000"
        not in
        gpu_name
    ):
        raise RuntimeError(
            "Expected frozen RTX A6000."
        )

    if (
        os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        )
        !=
        ":4096:8"
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG "
            "changed."
        )

    set_global_determinism(
        20260824
    )

    validation_rows = (
        read_validation_manifest(
            VALIDATION_MANIFEST
        )
    )

    by_id = {
        str(
            row.scenario_id
        ):
            row
        for row in (
            validation_rows
        )
    }

    validation_row = by_id.get(
        scenario_id
    )

    if validation_row is None:
        raise RuntimeError(
            "Formal scenario is absent "
            "from canonical validation manifest."
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
            "Fresh-process source-shard "
            "provenance mismatch."
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
            "Fresh-process selection-hash "
            "provenance mismatch."
        )

    scenario = read_motion_scenario(
        validation_row,
        paired_root=PAIRED_ROOT,
        compact_record_offset=int(
            formal[
                "compact_record_offset"
            ]
        ),
    )

    if (
        str(
            scenario.scenario_id
        )
        !=
        scenario_id
    ):
        raise RuntimeError(
            "Fresh process loaded wrong scenario."
        )

    if (
        int(
            scenario.current_time_index
        )
        !=
        10
    ):
        raise RuntimeError(
            "WOMD current_time_index changed."
        )

    if (
        len(
            scenario.timestamps_seconds
        )
        !=
        91
    ):
        raise RuntimeError(
            "WOMD timestamp count changed."
        )

    (
        clean_config,
        degraded_config,
    ) = load_frozen_stage2_configs()

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

    if (
        len(
            samples
        )
        !=
        expected_samples
    ):
        raise RuntimeError(
            "Fresh-process supervised sample "
            "count differs from frozen shard."
        )

    truth_sources = set()

    for sample in samples:
        resolved = (
            resolve_sample_truth_track_index(
                sample
            )
        )

        truth_sources.add(
            str(
                resolved[
                    "source"
                ]
            )
        )

    if (
        truth_sources
        !=
        {
            "sample.truth_track_index"
        }
    ):
        raise RuntimeError(
            "Fresh-process truth-index "
            f"metadata changed: {truth_sources}"
        )

    if (
        expected_truth_source
        and
        expected_truth_source
        not in truth_sources
    ):
        raise RuntimeError(
            "Fresh-process truth-index source "
            "differs from frozen shard."
        )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    if (
        normalization.get(
            "source"
        )
        !=
        "fit_only"
    ):
        raise RuntimeError(
            "Normalization source changed."
        )

    deterministic_checkpoint = torch.load(
        DET_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    Gaussian_checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    GMM_checkpoint = torch.load(
        GMM_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        state_dict_sha256(
            deterministic_checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED[
            "det_state"
        ]
    ):
        raise RuntimeError(
            "Deterministic state SHA changed."
        )

    if (
        state_dict_sha256(
            Gaussian_checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED[
            "Gaussian_state"
        ]
    ):
        raise RuntimeError(
            "Gaussian state SHA changed."
        )

    if (
        state_dict_sha256(
            GMM_checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED[
            "GMM_state"
        ]
    ):
        raise RuntimeError(
            "GMM state SHA changed."
        )

    architecture = (
        deterministic_checkpoint[
            "configuration"
        ]
    )

    (
        deterministic_model,
        deterministic_class,
    ) = construct_deterministic_model(
        architecture,
        deterministic_checkpoint[
            "state_dict"
        ],
        device=device,
    )

    Gaussian_model = build_gaussian(
        architecture,
        device=device,
    )

    Gaussian_model.load_state_dict(
        Gaussian_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    GMM_model = build_gmm(
        GMM_checkpoint[
            "architecture_configuration"
        ],
        device=device,
    )

    GMM_model.load_state_dict(
        GMM_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    deterministic_model.eval()
    Gaussian_model.eval()
    GMM_model.eval()

    normalizer = CalibrationNormalizer(
        normalization,
        device=device,
    )

    arrays = samples_to_arrays(
        samples
    )

    batch = normalizer.prepare(
        arrays,
        device=device,
    )

    with torch.inference_mode():
        deterministic_output = (
            deterministic_model(
                batch[
                    "target"
                ],
                batch[
                    "neighbors"
                ],
                batch[
                    "neighbor_mask"
                ],
                batch[
                    "map_context"
                ],
            )
        )

        deterministic_prediction = (
            extract_deterministic_prediction(
                deterministic_output
            )
        )

        Gaussian_output = Gaussian_model(
            batch[
                "target"
            ],
            batch[
                "neighbors"
            ],
            batch[
                "neighbor_mask"
            ],
            batch[
                "map_context"
            ],
        )

        GMM_output = GMM_model(
            batch[
                "target"
            ],
            batch[
                "neighbors"
            ],
            batch[
                "neighbor_mask"
            ],
            batch[
                "map_context"
            ],
        )

        actual_prediction_sha = (
            tensor_bundle_sha256(
                (
                    (
                        "det",
                        deterministic_prediction,
                    ),
                    (
                        "gaussian_mean",
                        Gaussian_output.mean,
                    ),
                    (
                        "gaussian_scale",
                        Gaussian_output.scale_tril,
                    ),
                    (
                        "gmm_logits",
                        GMM_output.mixture_logits,
                    ),
                    (
                        "gmm_means",
                        GMM_output.means,
                    ),
                    (
                        "gmm_scale",
                        GMM_output.scale_tril,
                    ),
                )
            )
        )

    exact = (
        actual_prediction_sha
        ==
        expected_prediction_sha
    )

    if not exact:
        raise RuntimeError(
            "Fresh-process prediction SHA "
            "does not match frozen Block4.8 "
            f"for rank {rank}."
        )

    environment = {
        "python":
            platform.python_version(),

        "numpy":
            np.__version__,

        "torch":
            torch.__version__,

        "torch_cuda":
            torch.version.cuda,

        "GPU":
            gpu_name,

        "CUBLAS_WORKSPACE_CONFIG":
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),

        "deterministic_algorithms":
            bool(
                torch
                .are_deterministic_algorithms_enabled()
            ),
    }

    result = {
        "stage":
            4,

        "block":
            "4.9_fresh_process_probe",

        "status":
            "PASS",

        "rank":
            int(
                rank
            ),

        "scenario_id":
            scenario_id,

        "supervised_samples":
            int(
                len(
                    samples
                )
            ),

        "truth_index_source":
            (
                "sample.truth_track_index"
            ),

        "deterministic_class":
            deterministic_class,

        "strict_state_load":
            True,

        "expected_prediction_sha256":
            expected_prediction_sha,

        "actual_prediction_sha256":
            actual_prediction_sha,

        "exact_prediction_match":
            True,

        "tracks_to_predict_accessed":
            False,

        "future_used_as_model_input":
            False,

        "training":
            False,

        "model_selection":
            False,

        "recalibration":
            False,

        "environment":
            environment,
    }

    write_result(
        result
    )

    print(
        "scenario ID              =",
        scenario_id,
    )

    print(
        "supervised samples       =",
        len(
            samples
        ),
    )

    print(
        "deterministic class      =",
        deterministic_class,
    )

    print(
        "strict state load        = PASS"
    )

    print(
        "expected prediction SHA  =",
        expected_prediction_sha,
    )

    print(
        "actual prediction SHA    =",
        actual_prediction_sha,
    )

    print(
        "exact prediction match   = PASS"
    )

    print(
        "tracks_to_predict access = NO"
    )

    print(
        "STATUS = PASS"
    )


try:
    main()

except BaseException as exc:
    failure = {
        "stage":
            4,

        "block":
            "4.9_fresh_process_probe",

        "status":
            "BLOCKED",

        "exception_type":
            type(
                exc
            ).__name__,

        "exception":
            str(
                exc
            ),

        "traceback":
            traceback.format_exc(),

        "strict_state_load":
            False,

        "exact_prediction_match":
            False,

        "tracks_to_predict_accessed":
            False,

        "training":
            False,

        "model_selection":
            False,

        "recalibration":
            False,
    }

    try:
        write_result(
            failure
        )

    except Exception:
        pass

    print()
    print(
        "============================================================"
    )

    print(
        "FRESH-PROCESS PROBE = BLOCKED"
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

    print(
        "training             = NO"
    )

    print(
        "recalibration        = NO"
    )

    print(
        "tracks_to_predict    = NOT ACCESSED"
    )

    print(
        "terminal remains open= YES"
    )

# Deliberately no sys.exit().

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from iscai_stage4.data import (
    attach_supervision,
)

from iscai_stage4.data.real_pipeline import (
    build_real_causal_inputs,
    load_frozen_stage2_configs,
    read_training_scenario,
)

from iscai_stage4.ml import (
    CLASS_CYCLIST,
    CLASS_ID_TO_NAME,
    CLASS_OTHER,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
    DeterministicTrajectoryGRU,
    actor_class_id,
    balanced_class_weights,
    compute_fit_normalization,
    masked_mse_loss,
    set_global_determinism,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

CONFIG_PATH = (
    STAGE4
    / "configs/"
      "stage4_deterministic_gru.json"
)

PREFLIGHT = (
    STAGE4
    / "reports/"
      "block43_preflight.json"
)

BLOCK42 = (
    STAGE4
    / "reports/"
      "block42_neural_sample_gate.json"
)

FIT_MANIFEST = (
    STAGE4
    / "artifacts/block41/"
      "fit.jsonl"
)

DEV_MANIFEST = (
    STAGE4
    / "artifacts/block41/"
      "development.jsonl"
)

SELECTION = (
    STAGE4
    / "artifacts/block43/"
      "model_development_selection.json"
)

CACHE_ROOT = (
    STAGE4
    / "artifacts/block43/"
      "cache"
)

FIT_CACHE = (
    STAGE4
    / "artifacts/block43/"
      "fit_cache.npz"
)

DEV_CACHE = (
    STAGE4
    / "artifacts/block43/"
      "development_cache.npz"
)

CACHE_REPORT = (
    STAGE4
    / "artifacts/block43/"
      "cache_manifest.json"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CHECKPOINT = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_gru.pt"
)

REPORT = (
    STAGE4
    / "reports/"
      "block43_deterministic_gru.json"
)

FAILURE_REPORT = (
    STAGE4
    / "reports/"
      "block43_failure.json"
)

LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
)

EXPECTED_BLOCK42_SHA = (
    "4f6ef90bd3455bf55f100561d66ef935"
    "49d8472112b85fce45930ec169f89fd0"
)

EXPECTED_FIT_SHA = (
    "284a61c877d937deb07137345beaabf3165690138785f9cc41f81655066d5276"
)

EXPECTED_DEV_SHA = (
    "e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c"
)

MIN_FREE_GIB = 250.0

ARRAY_KEYS = (
    "target",
    "neighbors",
    "neighbor_mask",
    "map_context",
    "future",
    "future_mask",
    "class_id",
)


def file_sha256(
    path: Path,
) -> str:
    digest = sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_json_sha(
    value,
) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def array_bundle_sha(
    arrays,
) -> str:
    digest = sha256()

    for key in ARRAY_KEYS:
        value = np.ascontiguousarray(
            arrays[key]
        )

        digest.update(
            key.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                value.dtype
            ).encode(
                "ascii"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            json.dumps(
                list(
                    value.shape
                )
            ).encode(
                "ascii"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            value.tobytes(
                order="C"
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def freeze_json(
    path: Path,
    value,
) -> None:
    content = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    )

    if path.exists():
        existing = (
            path.read_text(
                encoding="utf-8"
            )
        )

        if existing != content:
            raise RuntimeError(
                "Frozen artifact differs "
                f"on rerun: {path}"
            )

        return

    path.write_text(
        content,
        encoding="utf-8",
    )


def read_jsonl(
    path: Path,
):
    return tuple(
        json.loads(line)
        for line in (
            path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )


def atomic_npz(
    path: Path,
    **arrays,
):
    tmp = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    with tmp.open(
        "wb"
    ) as stream:
        np.savez(
            stream,
            **arrays,
        )

    os.replace(
        tmp,
        path,
    )


def empty_arrays():
    return {
        "target":
            np.zeros(
                (
                    0,
                    11,
                    14,
                ),
                dtype=np.float32,
            ),

        "neighbors":
            np.zeros(
                (
                    0,
                    8,
                    11,
                    14,
                ),
                dtype=np.float32,
            ),

        "neighbor_mask":
            np.zeros(
                (
                    0,
                    8,
                ),
                dtype=np.float32,
            ),

        "map_context":
            np.zeros(
                (
                    0,
                    10,
                ),
                dtype=np.float32,
            ),

        "future":
            np.zeros(
                (
                    0,
                    4,
                    3,
                ),
                dtype=np.float32,
            ),

        "future_mask":
            np.zeros(
                (
                    0,
                    4,
                ),
                dtype=np.float32,
            ),

        "class_id":
            np.zeros(
                (
                    0,
                ),
                dtype=np.int64,
            ),
    }


def samples_to_arrays(
    samples,
):
    if not samples:
        return empty_arrays()

    target = np.asarray(
        [
            sample
            .model_input
            .target_history
            for sample in samples
        ],
        dtype=np.float32,
    )

    neighbors = np.asarray(
        [
            sample
            .model_input
            .neighbor_histories
            for sample in samples
        ],
        dtype=np.float32,
    )

    neighbor_mask = np.asarray(
        [
            sample
            .model_input
            .neighbor_mask
            for sample in samples
        ],
        dtype=np.float32,
    )

    map_context = np.asarray(
        [
            sample
            .model_input
            .map_context
            for sample in samples
        ],
        dtype=np.float32,
    )

    future = np.asarray(
        [
            sample
            .future_label
            .positions_H0_m
            for sample in samples
        ],
        dtype=np.float32,
    )

    future_mask = np.asarray(
        [
            sample
            .future_label
            .valid_mask
            for sample in samples
        ],
        dtype=np.float32,
    )

    class_id = np.asarray(
        [
            actor_class_id(
                sample.actor_class
            )
            for sample in samples
        ],
        dtype=np.int64,
    )

    result = {
        "target":
            target,

        "neighbors":
            neighbors,

        "neighbor_mask":
            neighbor_mask,

        "map_context":
            map_context,

        "future":
            future,

        "future_mask":
            future_mask,

        "class_id":
            class_id,
    }

    validate_arrays(
        result
    )

    return result


def validate_arrays(
    arrays,
):
    required = set(
        ARRAY_KEYS
    )

    if set(
        arrays
    ) != required:
        raise ValueError(
            "Cache array keys changed."
        )

    n = int(
        arrays[
            "target"
        ].shape[0]
    )

    expected_shapes = {
        "target":
            (
                n,
                11,
                14,
            ),

        "neighbors":
            (
                n,
                8,
                11,
                14,
            ),

        "neighbor_mask":
            (
                n,
                8,
            ),

        "map_context":
            (
                n,
                10,
            ),

        "future":
            (
                n,
                4,
                3,
            ),

        "future_mask":
            (
                n,
                4,
            ),

        "class_id":
            (
                n,
            ),
    }

    for key, expected in (
        expected_shapes.items()
    ):
        if tuple(
            arrays[key].shape
        ) != expected:
            raise ValueError(
                f"{key} shape changed: "
                f"{arrays[key].shape} "
                f"!= {expected}"
            )

    for key in (
        "target",
        "neighbors",
        "neighbor_mask",
        "map_context",
        "future",
        "future_mask",
    ):
        if not np.isfinite(
            arrays[key]
        ).all():
            raise ValueError(
                f"Non-finite cache "
                f"values in {key}."
            )


def load_npz_arrays(
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        arrays = {
            key:
                np.asarray(
                    data[key]
                )
            for key in ARRAY_KEYS
        }

    validate_arrays(
        arrays
    )

    return arrays


def valid_scene_shard(
    path: Path,
    *,
    scenario_id: str,
):
    if not path.is_file():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            stored_id = str(
                data[
                    "scenario_id"
                ].item()
            )

            if (
                stored_id
                !=
                scenario_id
            ):
                return False

            arrays = {
                key:
                    np.asarray(
                        data[key]
                    )
                for key in ARRAY_KEYS
            }

        validate_arrays(
            arrays
        )

        return True

    except Exception:
        return False


def archive_corrupt_shard(
    path: Path,
):
    if not path.exists():
        return

    stamp = int(
        time.time()
    )

    destination = (
        path.with_name(
            path.name
            +
            f".corrupt.{stamp}"
        )
    )

    os.replace(
        path,
        destination,
    )


def scene_shard_path(
    partition: str,
    rank: int,
    scenario_id: str,
):
    return (
        CACHE_ROOT
        /
        partition
        /
        (
            f"{rank:05d}_"
            f"{scenario_id}.npz"
        )
    )


def build_scene_shard(
    *,
    partition: str,
    rank: int,
    record: dict,
    clean_config,
    degraded_config,
    max_samples: int,
):
    scenario_id = (
        record[
            "scenario_id"
        ]
    )

    path = (
        scene_shard_path(
            partition,
            rank,
            scenario_id,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if valid_scene_shard(
        path,
        scenario_id=(
            scenario_id
        ),
    ):
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            return {
                "path":
                    str(path),

                "scenario_id":
                    scenario_id,

                "sample_count":
                    int(
                        data[
                            "target"
                        ].shape[0]
                    ),

                "reused":
                    True,
            }

    if path.exists():
        archive_corrupt_shard(
            path
        )

    scenario = (
        read_training_scenario(
            record
        )
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

    samples = (
        attach_supervision(
            built[
                "scene_inputs"
            ],
            scenario,
            T_H0_from_W=(
                built[
                    "adapted"
                ].frames
                .T_H0_from_W
            ),
        )
    )

    # attach_supervision already returns
    # deterministic prediction_id ordering.
    # The cap is therefore causal and does
    # not inspect future-label quality.
    samples = samples[
        :max_samples
    ]

    arrays = (
        samples_to_arrays(
            samples
        )
    )

    atomic_npz(
        path,
        scenario_id=np.asarray(
            scenario_id
        ),
        **arrays,
    )

    if not valid_scene_shard(
        path,
        scenario_id=(
            scenario_id
        ),
    ):
        raise RuntimeError(
            "New scene cache shard "
            "failed validation."
        )

    return {
        "path":
            str(path),

        "scenario_id":
            scenario_id,

        "sample_count":
            int(
                arrays[
                    "target"
                ].shape[0]
            ),

        "reused":
            False,

        "stage2_algorithm_sha256":
            built[
                "stage2_algorithm_sha256"
            ],

        "associated_tracks":
            built[
                "association_track_count"
            ],
    }


def merge_partition_cache(
    partition: str,
    selected_records,
    destination: Path,
):
    parts = {
        key: []
        for key in ARRAY_KEYS
    }

    scene_counts = []

    for rank, record in enumerate(
        selected_records,
        start=1,
    ):
        path = (
            scene_shard_path(
                partition,
                rank,
                record[
                    "scenario_id"
                ],
            )
        )

        if not valid_scene_shard(
            path,
            scenario_id=(
                record[
                    "scenario_id"
                ]
            ),
        ):
            raise RuntimeError(
                "Cannot merge invalid "
                f"scene shard: {path}"
            )

        arrays = load_npz_arrays(
            path
        )

        scene_counts.append(
            int(
                arrays[
                    "target"
                ].shape[0]
            )
        )

        for key in ARRAY_KEYS:
            parts[
                key
            ].append(
                arrays[key]
            )

    merged = {}

    for key in ARRAY_KEYS:
        if parts[key]:
            merged[
                key
            ] = np.concatenate(
                parts[key],
                axis=0,
            )
        else:
            merged[
                key
            ] = (
                empty_arrays()[
                    key
                ]
            )

    validate_arrays(
        merged
    )

    if (
        merged[
            "target"
        ].shape[0]
        <=
        0
    ):
        raise RuntimeError(
            f"{partition} merged "
            "cache is empty."
        )

    atomic_npz(
        destination,
        **merged,
    )

    reloaded = (
        load_npz_arrays(
            destination
        )
    )

    content_sha = (
        array_bundle_sha(
            reloaded
        )
    )

    class_counts = Counter(
        int(x)
        for x in (
            reloaded[
                "class_id"
            ]
            .tolist()
        )
    )

    valid_per_horizon = (
        reloaded[
            "future_mask"
        ].sum(
            axis=0
        )
    )

    return {
        "sample_count":
            int(
                reloaded[
                    "target"
                ].shape[0]
            ),

        "content_sha256":
            content_sha,

        "file_sha256":
            file_sha256(
                destination
            ),

        "class_counts":
            {
                CLASS_ID_TO_NAME[
                    class_id
                ]:
                    int(count)
                for class_id, count
                in sorted(
                    class_counts.items()
                )
            },

        "valid_labels_per_horizon":
            [
                int(x)
                for x in (
                    valid_per_horizon
                    .tolist()
                )
            ],

        "zero_sample_scenes":
            int(
                sum(
                    count == 0
                    for count
                    in scene_counts
                )
            ),

        "max_samples_in_scene":
            int(
                max(
                    scene_counts
                )
            ),
    }


def latest_position_torch(
    target,
):
    observed = (
        target[:, :, 12]
        >
        0.5
    )

    if not bool(
        observed.any(
            dim=1
        ).all()
    ):
        raise RuntimeError(
            "Target without causal "
            "observation in cache."
        )

    grid = torch.arange(
        target.shape[1],
        device=target.device,
        dtype=torch.long,
    )[None, :]

    minus_one = torch.full_like(
        grid.expand_as(
            observed
        ),
        -1,
    )

    indices = torch.where(
        observed,
        grid.expand_as(
            observed
        ),
        minus_one,
    ).max(
        dim=1
    ).values

    batch_index = torch.arange(
        target.shape[0],
        device=target.device,
    )

    return target[
        batch_index,
        indices,
        :3,
    ]


class TorchNormalizer:

    def __init__(
        self,
        statistics,
        *,
        device,
    ):
        self.feature_mean = (
            torch.tensor(
                statistics[
                    "continuous_feature_mean"
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        self.feature_std = (
            torch.tensor(
                statistics[
                    "continuous_feature_std"
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        self.map_mean = (
            torch.tensor(
                statistics[
                    "map_mean"
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        self.map_std = (
            torch.tensor(
                statistics[
                    "map_std"
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        self.label_mean = (
            torch.tensor(
                statistics[
                    "label_displacement_mean"
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        self.label_std = (
            torch.tensor(
                statistics[
                    "label_displacement_std"
                ],
                dtype=torch.float32,
                device=device,
            )
        )

    def prepare(
        self,
        arrays,
        indices,
        *,
        device,
    ):
        index = np.asarray(
            indices,
            dtype=np.int64,
        )

        target = torch.from_numpy(
            np.asarray(
                arrays[
                    "target"
                ][index],
                dtype=np.float32,
            )
        ).to(
            device=device,
            non_blocking=False,
        )

        neighbors = torch.from_numpy(
            np.asarray(
                arrays[
                    "neighbors"
                ][index],
                dtype=np.float32,
            )
        ).to(
            device=device,
            non_blocking=False,
        )

        neighbor_mask = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "neighbor_mask"
                    ][index],
                    dtype=np.float32,
                )
            )
            .to(
                device=device
            )
        )

        map_context = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "map_context"
                    ][index],
                    dtype=np.float32,
                )
            )
            .to(
                device=device
            )
        )

        future = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "future"
                    ][index],
                    dtype=np.float32,
                )
            )
            .to(
                device=device
            )
        )

        future_mask = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "future_mask"
                    ][index],
                    dtype=np.float32,
                )
            )
            .to(
                device=device
            )
        )

        class_id = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "class_id"
                    ][index],
                    dtype=np.int64,
                )
            )
            .to(
                device=device
            )
        )

        # ----------------------------------------------------
        # Normalize continuous history features.
        # Missing rows remain numerically zero.
        # ----------------------------------------------------

        target_observed = (
            target[
                :,
                :,
                12:13
            ]
        )

        target_cont = (
            (
                target[
                    :,
                    :,
                    :12
                ]
                -
                self.feature_mean
            )
            /
            self.feature_std
        )

        target_cont = (
            target_cont
            *
            target_observed
        )

        target_normalized = (
            torch.cat(
                (
                    target_cont,
                    target[
                        :,
                        :,
                        12:
                    ],
                ),
                dim=-1,
            )
        )

        neighbor_observed = (
            neighbors[
                :,
                :,
                :,
                12:13
            ]
            *
            neighbor_mask[
                :,
                :,
                None,
                None
            ]
        )

        neighbor_cont = (
            (
                neighbors[
                    :,
                    :,
                    :,
                    :12
                ]
                -
                self.feature_mean
            )
            /
            self.feature_std
        )

        neighbor_cont = (
            neighbor_cont
            *
            neighbor_observed
        )

        neighbors_normalized = (
            torch.cat(
                (
                    neighbor_cont,
                    neighbors[
                        :,
                        :,
                        :,
                        12:
                    ]
                    *
                    neighbor_mask[
                        :,
                        :,
                        None,
                        None
                    ],
                ),
                dim=-1,
            )
        )

        map_normalized = (
            (
                map_context
                -
                self.map_mean
            )
            /
            self.map_std
        )

        # ----------------------------------------------------
        # Output target is future H0 displacement
        # relative to latest causal observed position.
        # ----------------------------------------------------

        origin = (
            latest_position_torch(
                target
            )
        )

        displacement = (
            future
            -
            origin[:, None, :]
        )

        label_normalized = (
            (
                displacement
                -
                self.label_mean
            )
            /
            self.label_std
        )

        label_normalized = (
            label_normalized
            *
            future_mask[
                :,
                :,
                None
            ]
        )

        return {
            "target":
                target_normalized,

            "neighbors":
                neighbors_normalized,

            "neighbor_mask":
                neighbor_mask,

            "map_context":
                map_normalized,

            "label_normalized":
                label_normalized,

            "future_mask":
                future_mask,

            "class_id":
                class_id,

            "true_displacement":
                displacement,
        }

    def denormalize_prediction(
        self,
        prediction,
    ):
        return (
            prediction
            *
            self.label_std
            +
            self.label_mean
        )


def state_dict_sha256(
    state_dict,
):
    digest = sha256()

    for key in sorted(
        state_dict
    ):
        tensor = (
            state_dict[
                key
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            key.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "ascii"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            tensor.numpy()
            .tobytes()
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def save_checkpoint_atomic(
    payload,
):
    tmp = CHECKPOINT.with_suffix(
        ".pt.tmp"
    )

    torch.save(
        payload,
        tmp,
    )

    os.replace(
        tmp,
        CHECKPOINT,
    )


def model_from_config(
    config,
    *,
    device,
):
    section = (
        config[
            "model"
        ]
    )

    return (
        DeterministicTrajectoryGRU(
            target_hidden_dim=(
                int(
                    section[
                        "target_hidden_dim"
                    ]
                )
            ),
            neighbor_hidden_dim=(
                int(
                    section[
                        "neighbor_hidden_dim"
                    ]
                )
            ),
            map_hidden_dim=(
                int(
                    section[
                        "map_hidden_dim"
                    ]
                )
            ),
            fusion_hidden_dim=(
                int(
                    section[
                        "fusion_hidden_dim"
                    ]
                )
            ),
            use_neighbors=True,
            use_map=True,
        )
        .to(
            device
        )
    )


def evaluate_model(
    model,
    arrays,
    normalizer,
    *,
    device,
    batch_size=1024,
):
    model.eval()

    total_error_sum = 0.0
    total_valid = 0

    horizon_error_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_valid = np.zeros(
        4,
        dtype=np.int64,
    )

    class_error_sum = Counter()
    class_valid = Counter()

    with torch.inference_mode():
        for start in range(
            0,
            arrays[
                "target"
            ].shape[0],
            batch_size,
        ):
            stop = min(
                arrays[
                    "target"
                ].shape[0],
                start
                +
                batch_size,
            )

            index = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            batch = (
                normalizer.prepare(
                    arrays,
                    index,
                    device=device,
                )
            )

            prediction_normalized = (
                model(
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

            prediction = (
                normalizer
                .denormalize_prediction(
                    prediction_normalized
                )
            )

            truth = (
                batch[
                    "true_displacement"
                ]
            )

            valid = (
                batch[
                    "future_mask"
                ]
                >
                0.5
            )

            planar = torch.sqrt(
                (
                    prediction[
                        :,
                        :,
                        0
                    ]
                    -
                    truth[
                        :,
                        :,
                        0
                    ]
                ).square()
                +
                (
                    prediction[
                        :,
                        :,
                        1
                    ]
                    -
                    truth[
                        :,
                        :,
                        1
                    ]
                ).square()
            )

            total_error_sum += float(
                planar[
                    valid
                ].sum().item()
            )

            total_valid += int(
                valid.sum().item()
            )

            for horizon in range(4):
                hmask = (
                    valid[
                        :,
                        horizon
                    ]
                )

                count = int(
                    hmask.sum().item()
                )

                if count > 0:
                    horizon_error_sum[
                        horizon
                    ] += float(
                        planar[
                            :,
                            horizon
                        ][
                            hmask
                        ].sum().item()
                    )

                    horizon_valid[
                        horizon
                    ] += count

            classes = (
                batch[
                    "class_id"
                ]
            )

            for class_id in (
                CLASS_VEHICLE,
                CLASS_PEDESTRIAN,
                CLASS_CYCLIST,
                CLASS_OTHER,
            ):
                cmask = (
                    classes
                    ==
                    class_id
                )[:, None] & valid

                count = int(
                    cmask.sum().item()
                )

                if count > 0:
                    name = (
                        CLASS_ID_TO_NAME[
                            class_id
                        ]
                    )

                    class_error_sum[
                        name
                    ] += float(
                        planar[
                            cmask
                        ].sum().item()
                    )

                    class_valid[
                        name
                    ] += count

    if total_valid <= 0:
        raise RuntimeError(
            "Development evaluation "
            "has zero valid labels."
        )

    horizon_mean = []

    for horizon in range(4):
        if (
            horizon_valid[
                horizon
            ]
            <=
            0
        ):
            raise RuntimeError(
                "Development horizon "
                f"{horizon} has "
                "zero valid labels."
            )

        horizon_mean.append(
            float(
                horizon_error_sum[
                    horizon
                ]
                /
                horizon_valid[
                    horizon
                ]
            )
        )

    class_ade = {}

    for name in sorted(
        class_valid
    ):
        class_ade[
            name
        ] = float(
            class_error_sum[
                name
            ]
            /
            class_valid[
                name
            ]
        )

    return {
        "planar_ADE_m":
            float(
                total_error_sum
                /
                total_valid
            ),

        "horizon_mean_planar_error_m":
            {
                "0.1":
                    horizon_mean[0],

                "0.3":
                    horizon_mean[1],

                "0.5":
                    horizon_mean[2],

                "1.0":
                    horizon_mean[3],
            },

        "one_second_error_m":
            horizon_mean[3],

        "valid_actor_horizon_points":
            int(
                total_valid
            ),

        "class_planar_ADE_m":
            class_ade,
    }


def prediction_sha256(
    model,
    arrays,
    normalizer,
    *,
    device,
    sample_limit=2048,
):
    model.eval()

    digest = sha256()

    count = min(
        int(
            arrays[
                "target"
            ].shape[0]
        ),
        int(
            sample_limit
        ),
    )

    with torch.inference_mode():
        for start in range(
            0,
            count,
            512,
        ):
            stop = min(
                count,
                start
                +
                512,
            )

            index = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            batch = (
                normalizer.prepare(
                    arrays,
                    index,
                    device=device,
                )
            )

            output = (
                model(
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

            output = (
                output.detach()
                .cpu()
                .contiguous()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            digest.update(
                output.tobytes(
                    order="C"
                )
            )

    return digest.hexdigest()


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    files = []

    for root in roots:
        for path in root.rglob(
            "*"
        ):
            if not path.is_file():
                continue

            if "__pycache__" in (
                path.parts
            ):
                continue

            if path.suffix in (
                ".pyc",
                ".pyo",
            ):
                continue

            files.append(
                path
            )

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE4
                )
            )
    )

    digest = sha256()

    for path in files:
        relative = str(
            path.relative_to(
                STAGE4
            )
        )

        digest.update(
            relative.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            path.read_bytes()
        )

        digest.update(
            b"\0"
        )

    return (
        len(files),
        digest.hexdigest(),
    )


def failure_hint(
    exc,
):
    text = (
        f"{type(exc).__name__}: "
        f"{exc}"
    ).lower()

    if isinstance(
        exc,
        KeyboardInterrupt,
    ):
        return (
            "Rerun the identical Block4.3 "
            "training command. Completed "
            "per-scenario cache shards are "
            "validated and reused automatically; "
            "only missing work is regenerated."
        )

    if (
        "out of memory"
        in text
        or
        "cuda oom"
        in text
    ):
        return (
            "Do not change the frozen scientific "
            "configuration automatically. Check "
            "nvidia-smi for another GPU process. "
            "The A6000 48-GB card is far above "
            "the expected memory requirement for "
            "batch_size=512."
        )

    if (
        "deterministic"
        in text
        or
        "cublas_workspace"
        in text
    ):
        return (
            "Verify that the command exported "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before Python started. Do not disable "
            "deterministic algorithms."
        )

    if isinstance(
        exc,
        ModuleNotFoundError,
    ):
        return (
            "The runner requires the frozen "
            "PYTHONPATH containing Stage0–4 src "
            "directories. Rerun through the provided "
            "shell wrapper; do not install a guessed "
            "replacement package."
        )

    if (
        "no space"
        in text
        or
        "disk"
        in text
    ):
        return (
            "Check free space before retrying. "
            "Do not remove frozen Stage0–3 or "
            "legacy evidence. Generated Block4.3 "
            "cache shards are resumable."
        )

    if (
        "cache"
        in text
        or
        "npz"
        in text
    ):
        return (
            "Rerun the same command. Existing "
            "scene shards are validated first; "
            "a corrupt Block4.3-generated shard "
            "is archived with a .corrupt timestamp "
            "and regenerated automatically."
        )

    return (
        "Rerun only after inspecting "
        "reports/block43_failure.json. "
        "Already completed causal cache shards "
        "are resumable, so the expensive Stage2/"
        "association work does not need to restart."
    )


def main():
    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    preflight = json.loads(
        PREFLIGHT.read_text(
            encoding="utf-8"
        )
    )

    block42 = json.loads(
        BLOCK42.read_text(
            encoding="utf-8"
        )
    )

    if (
        preflight[
            "status"
        ]
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.3 preflight "
            "is not PASS."
        )

    if (
        block42[
            "status"
        ]
        !=
        "PASS"
        or
        block42[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK42_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.2 "
            "evidence changed."
        )

    if (
        file_sha256(
            FIT_MANIFEST
        )
        !=
        EXPECTED_FIT_SHA
    ):
        raise RuntimeError(
            "Frozen fit manifest changed."
        )

    if (
        file_sha256(
            DEV_MANIFEST
        )
        !=
        EXPECTED_DEV_SHA
    ):
        raise RuntimeError(
            "Frozen development "
            "manifest changed."
        )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if free_gib < (
        MIN_FREE_GIB
    ):
        raise RuntimeError(
            "250-GiB free-space "
            "reserve violated."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Frozen CUDA device "
            "is unavailable."
        )

    device = torch.device(
        "cuda:0"
    )

    if (
        "A6000"
        not in
        torch.cuda
        .get_device_name(0)
    ):
        raise RuntimeError(
            "Expected frozen RTX A6000 "
            "training device."
        )

    set_global_determinism(
        int(
            config[
                "training"
            ]["seed"]
        )
    )

    fit_records = read_jsonl(
        FIT_MANIFEST
    )

    dev_records = read_jsonl(
        DEV_MANIFEST
    )

    fit_count = int(
        config[
            "data"
        ][
            "fit_scene_count"
        ]
    )

    dev_count = int(
        config[
            "data"
        ][
            "development_scene_count"
        ]
    )

    max_samples = int(
        config[
            "data"
        ][
            "max_supervised_samples_per_scene"
        ]
    )

    selected_fit = (
        fit_records[
            :fit_count
        ]
    )

    selected_dev = (
        dev_records[
            :dev_count
        ]
    )

    if (
        len(selected_fit)
        !=
        fit_count
        or
        len(selected_dev)
        !=
        dev_count
    ):
        raise RuntimeError(
            "Frozen model-development "
            "scene selection count changed."
        )

    selection = {
        "stage": 4,
        "block": "4.3",
        "status":
            "FROZEN_BEFORE_TRAINING",

        "policy":
            (
                "first_N_records_of_"
                "frozen_SHA_ordered_"
                "partition_manifest"
            ),

        "future_used":
            False,

        "performance_used":
            False,

        "fit": {
            "count":
                fit_count,

            "scenario_ids": [
                record[
                    "scenario_id"
                ]
                for record
                in selected_fit
            ],
        },

        "development": {
            "count":
                dev_count,

            "scenario_ids": [
                record[
                    "scenario_id"
                ]
                for record
                in selected_dev
            ],
        },
    }

    freeze_json(
        SELECTION,
        selection,
    )

    selection_sha = (
        file_sha256(
            SELECTION
        )
    )

    (
        clean_config,
        degraded_config,
    ) = load_frozen_stage2_configs()

    print(
        "============================================================"
    )
    print(
        "BLOCK 4.3 — REAL CACHE GENERATION"
    )
    print(
        "============================================================"
    )

    cache_start = (
        time.perf_counter()
    )

    build_reports = {
        "fit": [],
        "development": [],
    }

    reused = Counter()

    for partition, records in (
        (
            "fit",
            selected_fit,
        ),
        (
            "development",
            selected_dev,
        ),
    ):
        total = len(
            records
        )

        print()
        print(
            f"===== {partition.upper()} "
            f"{total} SCENES ====="
        )

        for rank, record in enumerate(
            records,
            start=1,
        ):
            result = (
                build_scene_shard(
                    partition=partition,
                    rank=rank,
                    record=record,
                    clean_config=(
                        clean_config
                    ),
                    degraded_config=(
                        degraded_config
                    ),
                    max_samples=(
                        max_samples
                    ),
                )
            )

            build_reports[
                partition
            ].append(
                result
            )

            if result[
                "reused"
            ]:
                reused[
                    partition
                ] += 1

            if (
                rank % 100 == 0
                or
                rank == total
            ):
                samples_so_far = sum(
                    item[
                        "sample_count"
                    ]
                    for item in (
                        build_reports[
                            partition
                        ]
                    )
                )

                print(
                    f"{partition}: "
                    f"{rank}/{total} "
                    f"| samples="
                    f"{samples_so_far} "
                    f"| reused="
                    f"{reused[partition]}",
                    flush=True,
                )

    cache_generation_s = (
        time.perf_counter()
        -
        cache_start
    )

    print()
    print(
        "===== MERGING CACHE ====="
    )

    fit_cache_info = (
        merge_partition_cache(
            "fit",
            selected_fit,
            FIT_CACHE,
        )
    )

    dev_cache_info = (
        merge_partition_cache(
            "development",
            selected_dev,
            DEV_CACHE,
        )
    )

    for info_name, info in (
        (
            "fit",
            fit_cache_info,
        ),
        (
            "development",
            dev_cache_info,
        ),
    ):
        classes = (
            info[
                "class_counts"
            ]
        )

        for required in (
            "TYPE_VEHICLE",
            "TYPE_PEDESTRIAN",
            "TYPE_CYCLIST",
        ):
            if (
                classes.get(
                    required,
                    0,
                )
                <=
                0
            ):
                raise RuntimeError(
                    f"{info_name} cache "
                    f"has no {required}."
                )

    cache_report = {
        "selection_sha256":
            selection_sha,

        "selection_future_based":
            False,

        "selection_performance_based":
            False,

        "max_samples_per_scene":
            max_samples,

        "cache_generation_runtime_s":
            cache_generation_s,

        "fit_scene_shards_reused":
            int(
                reused[
                    "fit"
                ]
            ),

        "development_scene_shards_reused":
            int(
                reused[
                    "development"
                ]
            ),

        "fit":
            fit_cache_info,

        "development":
            dev_cache_info,
    }

    CACHE_REPORT.write_text(
        json.dumps(
            cache_report,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print(
        "fit cache samples         =",
        fit_cache_info[
            "sample_count"
        ],
    )

    print(
        "development cache samples =",
        dev_cache_info[
            "sample_count"
        ],
    )

    print(
        "fit classes               =",
        fit_cache_info[
            "class_counts"
        ],
    )

    print(
        "development classes       =",
        dev_cache_info[
            "class_counts"
        ],
    )

    print(
        "fit cache content SHA     =",
        fit_cache_info[
            "content_sha256"
        ],
    )

    print(
        "development content SHA   =",
        dev_cache_info[
            "content_sha256"
        ],
    )

    # ========================================================
    # Fit-only normalization
    # ========================================================

    print()
    print(
        "===== FIT-ONLY NORMALIZATION ====="
    )

    fit_arrays = (
        load_npz_arrays(
            FIT_CACHE
        )
    )

    dev_arrays = (
        load_npz_arrays(
            DEV_CACHE
        )
    )

    normalization = (
        compute_fit_normalization(
            fit_arrays
        )
    )

    freeze_json(
        NORMALIZATION,
        normalization,
    )

    normalization_sha = (
        file_sha256(
            NORMALIZATION
        )
    )

    if (
        normalization[
            "source"
        ]
        !=
        "fit_only"
    ):
        raise RuntimeError(
            "Normalization source "
            "is not fit-only."
        )

    if any((
        normalization[
            "development_used"
        ],
        normalization[
            "calibration_used"
        ],
        normalization[
            "validation_used"
        ],
    )):
        raise RuntimeError(
            "Normalization leakage."
        )

    print(
        "normalization source      = FIT ONLY"
    )

    print(
        "normalization SHA256      =",
        normalization_sha,
    )

    # ========================================================
    # Class-aware fit sampler
    # ========================================================

    class_weights = (
        balanced_class_weights(
            fit_arrays[
                "class_id"
            ]
        )
    )

    class_ids_cpu = torch.as_tensor(
        fit_arrays[
            "class_id"
        ],
        dtype=torch.long,
    )

    primary_mass = {}

    for class_id in (
        CLASS_VEHICLE,
        CLASS_PEDESTRIAN,
        CLASS_CYCLIST,
    ):
        primary_mass[
            CLASS_ID_TO_NAME[
                class_id
            ]
        ] = float(
            class_weights[
                class_ids_cpu
                ==
                class_id
            ].sum().item()
        )

    if (
        max(
            primary_mass.values()
        )
        -
        min(
            primary_mass.values()
        )
        >
        1e-12
    ):
        raise RuntimeError(
            "Primary class sampler "
            "mass is not balanced."
        )

    print(
        "class-balanced fit sampler= PASS"
    )

    print(
        "primary sampling mass     =",
        primary_mass,
    )

    # ========================================================
    # Deterministic GPU training
    # ========================================================

    print()
    print(
        "===== DETERMINISTIC GRU TRAINING ====="
    )

    seed = int(
        config[
            "training"
        ]["seed"]
    )

    batch_size = int(
        config[
            "training"
        ][
            "batch_size"
        ]
    )

    max_epochs = int(
        config[
            "training"
        ][
            "max_epochs"
        ]
    )

    patience_limit = int(
        config[
            "training"
        ][
            "early_stopping_patience"
        ]
    )

    min_delta = float(
        config[
            "training"
        ][
            "early_stopping_min_delta_m"
        ]
    )

    learning_rate = float(
        config[
            "training"
        ][
            "learning_rate"
        ]
    )

    weight_decay = float(
        config[
            "training"
        ][
            "weight_decay"
        ]
    )

    gradient_clip = float(
        config[
            "training"
        ][
            "gradient_clip_norm"
        ]
    )

    set_global_determinism(
        seed
    )

    model = model_from_config(
        config,
        device=device,
    )

    normalizer = TorchNormalizer(
        normalization,
        device=device,
    )

    initial_dev = evaluate_model(
        model,
        dev_arrays,
        normalizer,
        device=device,
    )

    print(
        "untrained dev ADE m       =",
        initial_dev[
            "planar_ADE_m"
        ],
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=(
                learning_rate
            ),
            weight_decay=(
                weight_decay
            ),
        )
    )

    best_ade = math.inf
    best_epoch = None
    patience = 0

    history = []

    training_start = (
        time.perf_counter()
    )

    fit_n = int(
        fit_arrays[
            "target"
        ].shape[0]
    )

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        model.train()

        generator = (
            torch.Generator(
                device="cpu"
            )
        )

        generator.manual_seed(
            seed
            +
            epoch
        )

        sampled = (
            torch.multinomial(
                class_weights,
                num_samples=(
                    fit_n
                ),
                replacement=True,
                generator=(
                    generator
                ),
            )
            .numpy()
        )

        epoch_loss_sum = 0.0
        epoch_valid_coordinates = 0

        for start in range(
            0,
            fit_n,
            batch_size,
        ):
            stop = min(
                fit_n,
                start
                +
                batch_size,
            )

            index = sampled[
                start:stop
            ]

            batch = (
                normalizer.prepare(
                    fit_arrays,
                    index,
                    device=device,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
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

            loss = masked_mse_loss(
                prediction,
                batch[
                    "label_normalized"
                ],
                batch[
                    "future_mask"
                ],
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite deterministic "
                    "GRU training loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    gradient_clip
                ),
            )

            optimizer.step()

            valid_coordinates = int(
                (
                    batch[
                        "future_mask"
                    ].sum()
                    *
                    3
                ).item()
            )

            epoch_loss_sum += (
                float(
                    loss
                    .detach()
                    .item()
                )
                *
                max(
                    valid_coordinates,
                    1,
                )
            )

            epoch_valid_coordinates += (
                valid_coordinates
            )

        torch.cuda.synchronize()

        if (
            epoch_valid_coordinates
            <=
            0
        ):
            raise RuntimeError(
                "Training epoch has zero "
                "valid target coordinates."
            )

        training_loss = (
            epoch_loss_sum
            /
            epoch_valid_coordinates
        )

        dev_metrics = (
            evaluate_model(
                model,
                dev_arrays,
                normalizer,
                device=device,
            )
        )

        dev_ade = float(
            dev_metrics[
                "planar_ADE_m"
            ]
        )

        improved = (
            dev_ade
            <
            best_ade
            -
            min_delta
        )

        if improved:
            best_ade = dev_ade
            best_epoch = epoch
            patience = 0

            state_sha = (
                state_dict_sha256(
                    model.state_dict()
                )
            )

            save_checkpoint_atomic({
                "stage": 4,
                "block": "4.3",

                "model_type":
                    "DeterministicTrajectoryGRU",

                "state_dict":
                    model.state_dict(),

                "state_dict_sha256":
                    state_sha,

                "best_epoch":
                    best_epoch,

                "best_development_ADE_m":
                    best_ade,

                "normalization_sha256":
                    normalization_sha,

                "fit_cache_content_sha256":
                    fit_cache_info[
                        "content_sha256"
                    ],

                "development_cache_content_sha256":
                    dev_cache_info[
                        "content_sha256"
                    ],

                "selection_sha256":
                    selection_sha,

                "configuration":
                    config,
            })

        else:
            patience += 1

        history.append({
            "epoch":
                epoch,

            "training_normalized_MSE":
                float(
                    training_loss
                ),

            "development_planar_ADE_m":
                dev_ade,

            "development_one_second_error_m":
                float(
                    dev_metrics[
                        "one_second_error_m"
                    ]
                ),

            "improved":
                bool(
                    improved
                ),

            "patience":
                int(
                    patience
                ),
        })

        print(
            f"epoch {epoch:02d} "
            f"| trainMSE="
            f"{training_loss:.6f} "
            f"| devADE="
            f"{dev_ade:.6f} m "
            f"| dev1s="
            f"{dev_metrics['one_second_error_m']:.6f} m "
            f"| best="
            f"{best_ade:.6f} m "
            f"| patience="
            f"{patience}/{patience_limit}",
            flush=True,
        )

        if (
            patience
            >=
            patience_limit
        ):
            print(
                "early stopping = TRIGGERED"
            )

            break

    training_runtime_s = (
        time.perf_counter()
        -
        training_start
    )

    if not CHECKPOINT.is_file():
        raise RuntimeError(
            "No deterministic GRU "
            "checkpoint was produced."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint[
            "normalization_sha256"
        ]
        !=
        normalization_sha
    ):
        raise RuntimeError(
            "Checkpoint normalization "
            "hash mismatch."
        )

    best_model = model_from_config(
        config,
        device=device,
    )

    best_model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    best_model.eval()

    actual_state_sha = (
        state_dict_sha256(
            best_model.state_dict()
        )
    )

    if (
        actual_state_sha
        !=
        checkpoint[
            "state_dict_sha256"
        ]
    ):
        raise RuntimeError(
            "Checkpoint state_dict "
            "fingerprint mismatch."
        )

    final_dev = evaluate_model(
        best_model,
        dev_arrays,
        normalizer,
        device=device,
    )

    if (
        final_dev[
            "planar_ADE_m"
        ]
        >=
        initial_dev[
            "planar_ADE_m"
        ]
    ):
        raise RuntimeError(
            "Deterministic training did "
            "not improve development ADE "
            "over the untrained model."
        )

    for required_class in (
        "TYPE_VEHICLE",
        "TYPE_PEDESTRIAN",
        "TYPE_CYCLIST",
    ):
        if required_class not in (
            final_dev[
                "class_planar_ADE_m"
            ]
        ):
            raise RuntimeError(
                "Development evaluation "
                f"lacks {required_class}."
            )

    # ========================================================
    # Exact inference repeat from frozen checkpoint
    # ========================================================

    hash_a = prediction_sha256(
        best_model,
        dev_arrays,
        normalizer,
        device=device,
    )

    repeat_model = model_from_config(
        config,
        device=device,
    )

    repeat_checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    repeat_model.load_state_dict(
        repeat_checkpoint[
            "state_dict"
        ]
    )

    repeat_model.eval()

    hash_b = prediction_sha256(
        repeat_model,
        dev_arrays,
        normalizer,
        device=device,
    )

    if hash_a != hash_b:
        raise RuntimeError(
            "Frozen checkpoint inference "
            "is not exactly repeatable."
        )

    print()
    print(
        "exact checkpoint inference = PASS"
    )

    print(
        "prediction SHA256          =",
        hash_a,
    )

    # ========================================================
    # Full Stage4 regression
    # ========================================================

    import re
    import subprocess

    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    combined = (
        test.stdout
        +
        "\n"
        +
        test.stderr
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        combined,
    )

    test_count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        test.returncode
        !=
        0
        or
        test_count
        !=
        44
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected final Block4.3 "
            "regression 44/44."
        )

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

    checkpoint_sha = (
        file_sha256(
            CHECKPOINT
        )
    )

    free_gib_after = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if (
        free_gib_after
        <
        MIN_FREE_GIB
    ):
        raise RuntimeError(
            "250-GiB reserve violated "
            "after Block4.3."
        )

    report = {
        "stage": 4,
        "block": "4.3",
        "status": "PASS",

        "scope":
            "deterministic_multi_agent_GRU",

        "data": {
            "fit_scenes":
                fit_count,

            "development_scenes":
                dev_count,

            "selection_sha256":
                selection_sha,

            "selection_future_based":
                False,

            "selection_performance_based":
                False,

            "fit_cache":
                fit_cache_info,

            "development_cache":
                dev_cache_info,

            "sample_cap_per_scene":
                max_samples,
        },

        "normalization": {
            "source":
                "fit_only",

            "sha256":
                normalization_sha,

            "development_used":
                False,

            "calibration_used":
                False,

            "formal_validation_used":
                False,
        },

        "sampler": {
            "class_aware":
                True,

            "actor_class_is_model_feature":
                False,

            "primary_class_total_mass":
                primary_mass,
        },

        "model": {
            "architecture":
                "GRU",

            "multi_agent_context":
                True,

            "map_context":
                True,

            "measurement_covariance_input":
                True,

            "predictive_covariance":
                False,

            "perfect_track_id_feature":
                False,

            "actor_class_feature":
                False,

            "future_input":
                False,
        },

        "training": {
            "device":
                str(device),

            "gpu":
                torch.cuda
                .get_device_name(0),

            "seed":
                seed,

            "optimizer":
                "AdamW",

            "batch_size":
                batch_size,

            "max_epochs":
                max_epochs,

            "executed_epochs":
                len(history),

            "best_epoch":
                int(
                    checkpoint[
                        "best_epoch"
                    ]
                ),

            "runtime_s":
                training_runtime_s,

            "history":
                history,
        },

        "development": {
            "untrained":
                initial_dev,

            "best":
                final_dev,

            "improved_over_untrained":
                True,
        },

        "checkpoint": {
            "path":
                str(
                    CHECKPOINT
                ),

            "file_sha256":
                checkpoint_sha,

            "state_dict_sha256":
                actual_state_sha,

            "exact_inference_repeat":
                True,

            "prediction_sha256":
                hash_a,
        },

        "formal_validation": {
            "used_for_training":
                False,

            "used_for_early_stopping":
                False,

            "used_for_normalization":
                False,

            "used_for_model_selection":
                False,

            "formal_comparison":
                "DEFERRED_TO_BLOCK_4.8",
        },

        "regression": {
            "tests_passed":
                44,

            "tests_total":
                44,
        },

        "storage": {
            "free_gib":
                free_gib_after,

            "hard_reserve_gib":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "implementation": {
            "file_count":
                implementation_files,

            "sha256":
                implementation_sha,
        },

        "not_yet_claimed": [
            "predictive Gaussian covariance",
            "Gaussian NLL",
            "trajectory calibration",
            "coverage/ECE/Brier",
            "GMM",
            "formal validation performance",
            "probabilistic superiority over classical baseline"
        ],
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    if FAILURE_REPORT.exists():
        FAILURE_REPORT.unlink()

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.3 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "fit scenes                 =",
        fit_count,
    )

    print(
        "development scenes         =",
        dev_count,
    )

    print(
        "fit samples                =",
        fit_cache_info[
            "sample_count"
        ],
    )

    print(
        "development samples        =",
        dev_cache_info[
            "sample_count"
        ],
    )

    print(
        "fit classes                =",
        fit_cache_info[
            "class_counts"
        ],
    )

    print(
        "development classes        =",
        dev_cache_info[
            "class_counts"
        ],
    )

    print(
        "normalization source       = FIT ONLY"
    )

    print(
        "class-balanced sampler     = PASS"
    )

    print(
        "measurement covariance     = INPUT"
    )

    print(
        "multi-agent context        = PASS"
    )

    print(
        "map context                = PASS"
    )

    print(
        "future model input         = NO"
    )

    print(
        "actor class model feature  = NO"
    )

    print(
        "untrained dev ADE m        =",
        round(
            initial_dev[
                "planar_ADE_m"
            ],
            6,
        ),
    )

    print(
        "best dev ADE m             =",
        round(
            final_dev[
                "planar_ADE_m"
            ],
            6,
        ),
    )

    print(
        "best dev 1.0s error m      =",
        round(
            final_dev[
                "one_second_error_m"
            ],
            6,
        ),
    )

    print(
        "best epoch                 =",
        checkpoint[
            "best_epoch"
        ],
    )

    print(
        "class dev ADE              =",
        final_dev[
            "class_planar_ADE_m"
        ],
    )

    print(
        "checkpoint SHA256          =",
        checkpoint_sha,
    )

    print(
        "state_dict SHA256          =",
        actual_state_sha,
    )

    print(
        "inference repeat SHA       =",
        hash_a,
    )

    print(
        "exact inference repeat     = PASS"
    )

    print(
        "formal validation used     = NO"
    )

    print(
        "full Stage4 regression     = 44 / 44 PASS"
    )

    print(
        "free GiB                   =",
        round(
            free_gib_after,
            3,
        ),
    )

    print(
        "implementation files       =",
        implementation_files,
    )

    print(
        "implementation SHA256      =",
        implementation_sha,
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    # ========================================================
    # Closure log only after full PASS
    # ========================================================

    marker = (
        "## Block 4.3 — "
        "Deterministic multi-agent GRU"
    )

    existing = (
        LOG.read_text(
            encoding="utf-8"
        )
        if LOG.exists()
        else ""
    )

    if marker not in existing:
        with LOG.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                "\n"
                + marker
                + "\n\n"
                "Status: PASS / FROZEN\n\n"
                "- A deterministic lightweight "
                  "multi-agent GRU baseline was "
                  "trained on real frozen Stage4 "
                  "causal samples.\n"
                f"- Frozen model-development "
                  f"subset: {fit_count} fit scenes "
                  f"and {dev_count} development "
                  f"scenes, selected only by "
                  f"pre-frozen SHA order.\n"
                "- Per-scene supervised-sample cap "
                  "is deterministic by causal "
                  "prediction ID and does not use "
                  "future-label quality.\n"
                "- Cache generation is resumable; "
                  "valid completed scene shards are "
                  "reused and corrupt generated "
                  "shards are archived before "
                  "regeneration.\n"
                "- Input uses the frozen Stage2 "
                  "degraded observations through "
                  "Stage3 estimated association.\n"
                "- Stage2 measurement covariance "
                  "remains a mandatory numeric "
                  "predictor input.\n"
                "- Main deterministic architecture "
                  "contains target-history GRU, "
                  "neighbour-history GRU aggregation "
                  "and lightweight map encoder.\n"
                "- Actor class is used only for the "
                  "fit sampler and is not a neural "
                  "model feature.\n"
                "- Primary vehicle, pedestrian and "
                  "cyclist samples receive equal "
                  "total fit-sampling mass.\n"
                "- All normalization statistics are "
                  "fit-only; missing rows and padded "
                  "neighbours are excluded.\n"
                "- Early stopping uses development "
                  "planar ADE only.\n"
                "- Calibration and formal validation "
                  "are not used by Block4.3.\n"
                "- Future trajectory is supervision "
                  "only; future validity is a loss "
                  "mask only.\n"
                "- Deterministic checkpoint inference "
                  "repeats exactly from the same "
                  "frozen checkpoint.\n"
                f"- Best development planar ADE: "
                f"{final_dev['planar_ADE_m']:.9f} m.\n"
                f"- Best epoch: "
                f"{checkpoint['best_epoch']}.\n"
                f"- Checkpoint SHA256: "
                f"{checkpoint_sha}.\n"
                f"- State-dict SHA256: "
                f"{actual_state_sha}.\n"
                f"- Prediction repeat SHA256: "
                f"{hash_a}.\n"
                "- No predictive covariance/NLL/"
                  "calibration/GMM or formal "
                  "superiority is claimed yet.\n"
                "- Full Stage4 regression: "
                  "44/44 PASS.\n"
                f"- Implementation files: "
                f"{implementation_files}.\n"
                f"- Implementation SHA256: "
                f"{implementation_sha}.\n"
            )

    print()
    print(
        "===== BLOCK 4.3 FINAL ====="
    )

    print(
        "real deterministic GRU = PASS"
    )

    print(
        "fit-only normalization  = PASS"
    )

    print(
        "class-aware sampler     = PASS"
    )

    print(
        "multi-agent/map context = PASS"
    )

    print(
        "future leakage          = NONE"
    )

    print(
        "checkpoint freeze       = PASS"
    )

    print(
        "exact inference repeat  = PASS"
    )

    print(
        "regression              = 44 / 44"
    )

    print(
        "implementation SHA      =",
        implementation_sha,
    )

    print(
        "log closure             = PASS"
    )


if __name__ == "__main__":
    try:
        main()

    except (
        Exception,
        KeyboardInterrupt,
    ) as exc:
        FAILURE_REPORT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        failure = {
            "stage": 4,
            "block": "4.3",
            "status": "BLOCKED",

            "exception_type":
                type(exc).__name__,

            "exception":
                str(exc),

            "traceback":
                traceback.format_exc(),

            "recovery_hint":
                failure_hint(
                    exc
                ),

            "cache_is_resumable":
                True,

            "scientific_configuration_changed":
                False,
        }

        FAILURE_REPORT.write_text(
            json.dumps(
                failure,
                indent=2,
                sort_keys=True,
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
            "BLOCK 4.3 = BLOCKED"
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
        print(
            "RECOVERY HINT:"
        )

        print(
            failure[
                "recovery_hint"
            ]
        )

        print()
        print(
            "completed cache shards "
            "will be reused = YES"
        )

        print(
            "scientific config "
            "auto-modified = NO"
        )

        print(
            "failure report =",
            FAILURE_REPORT,
        )

        traceback.print_exc()

        raise SystemExit(2)

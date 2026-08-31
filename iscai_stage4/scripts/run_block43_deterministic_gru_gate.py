from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from iscai_stage4.data.real_pipeline import (
    build_real_causal_samples,
    load_frozen_stage2_configs,
    read_jsonl,
    read_training_scenario,
)

from iscai_stage4.models import (
    DeterministicGRUConfig,
    DeterministicTrajectoryGRU,
)

from iscai_stage4.training import (
    FitNormalizationStats,
    FitStatsAccumulator,
    denormalize_displacement,
    normalized_sample_arrays,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

BLOCK42_REPORT = (
    STAGE4
    / "reports/"
      "block42_neural_sample_gate.json"
)

FIT_MANIFEST = (
    STAGE4
    / "artifacts/block41/fit.jsonl"
)

DEV_MANIFEST = (
    STAGE4
    / "artifacts/block41/"
      "development.jsonl"
)

CAL_MANIFEST = (
    STAGE4
    / "artifacts/block41/"
      "calibration.jsonl"
)

TRAIN_SUBSET = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_train_scenes.jsonl"
)

DEV_SUBSET = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_dev_scenes.jsonl"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

CHECKPOINT = (
    STAGE4
    / "checkpoints/"
      "block43_deterministic_gru.pt"
)

TRAINING_HISTORY = (
    STAGE4
    / "artifacts/block43/"
      "training_history.json"
)

CONFIG = (
    STAGE4
    / "configs/"
      "block43_deterministic_gru.json"
)

REPORT = (
    STAGE4
    / "reports/"
      "block43_deterministic_gru_gate.json"
)

LOG = (
    STAGE4
    / "docs/"
      "implementation_log.md"
)


EXPECTED_BLOCK42_SHA = (
    "4f6ef90bd3455bf55f100561d66ef93549d8472112b85fce45930ec169f89fd0"
)

EXPECTED_FIT_SHA = (
    "284a61c877d937deb07137345beaabf3165690138785f9cc41f81655066d5276"
)

EXPECTED_DEV_SHA = (
    "e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c"
)

EXPECTED_CAL_SHA = (
    "e003dd5c4d5a253729700b12c3754c47ffb99dadfa035b46627f01ec91eed4be"
)


SEED = 20260820

TRAIN_QUOTA = 512
DEV_QUOTA = 128

TRAIN_SCENES = (
    TRAIN_QUOTA
    *
    3
)

DEV_SCENES = (
    DEV_QUOTA
    *
    3
)

BATCH_SIZE = 256
EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

PRIMARY_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

MIN_FREE_GIB = 250.0


def file_sha256(
    path: Path,
):
    digest = sha256()

    with path.open("rb") as stream:
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


def canonical_sha(
    value,
):
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


def freeze_bytes(
    path: Path,
    content: bytes,
):
    if path.exists():
        if (
            path.read_bytes()
            !=
            content
        ):
            raise RuntimeError(
                "Frozen Block4.3 artifact "
                f"changed on rerun: {path}"
            )

        return

    path.write_bytes(
        content
    )


def stratum(
    record,
):
    counts = (
        record[
            "anchor_class_counts"
        ]
    )

    cyclist = (
        int(
            counts.get(
                "TYPE_CYCLIST",
                0,
            )
        )
        >
        0
    )

    pedestrian = (
        int(
            counts.get(
                "TYPE_PEDESTRIAN",
                0,
            )
        )
        >
        0
    )

    vehicle = (
        int(
            counts.get(
                "TYPE_VEHICLE",
                0,
            )
        )
        >
        0
    )

    if cyclist:
        return "cyclist"

    if pedestrian:
        return (
            "pedestrian_no_cyclist"
        )

    if vehicle:
        return "vehicle_only"

    return None


def deterministic_subset(
    records,
    *,
    quota,
):
    buckets = {
        "cyclist": [],
        "pedestrian_no_cyclist": [],
        "vehicle_only": [],
    }

    for record in records:
        group = stratum(
            record
        )

        if group is not None:
            buckets[
                group
            ].append(
                record
            )

    selected = []

    for group in (
        "cyclist",
        "pedestrian_no_cyclist",
        "vehicle_only",
    ):
        ordered = sorted(
            buckets[group],
            key=lambda item: (
                item[
                    "selection_hash"
                ],
                item[
                    "scenario_id"
                ],
            ),
        )

        if len(
            ordered
        ) < quota:
            raise RuntimeError(
                f"Insufficient {group} "
                f"scenarios: "
                f"{len(ordered)} < {quota}"
            )

        for rank, record in enumerate(
            ordered[:quota],
            start=1,
        ):
            selected.append({
                **record,

                "block43_stratum":
                    group,

                "block43_stratum_rank":
                    rank,

                "selection_future_based":
                    False,

                "selection_performance_based":
                    False,
            })

    return tuple(
        selected
    )


def manifest_bytes(
    records,
):
    return (
        "\n".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
            for record in records
        )
        +
        "\n"
    ).encode("utf-8")


def model_payload_hash(
    samples,
):
    return canonical_sha(
        tuple(
            (
                sample.prediction_id,
                sample
                .model_payload_sha256(),
            )
            for sample in samples
        )
    )


def label_hash(
    samples,
):
    return canonical_sha(
        tuple(
            (
                sample.prediction_id,
                sample
                .future_label
                .sha256(),
            )
            for sample in samples
        )
    )


def configure_determinism():
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = ":4096:8"

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    torch.use_deterministic_algorithms(
        True
    )

    # Avoid CuDNN RNN nondeterminism.
    torch.backends.cudnn.enabled = False

    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.allow_tf32 = False


def class_weights_from_counts(
    counts,
):
    for actor_class in (
        PRIMARY_CLASSES
    ):
        if (
            counts.get(
                actor_class,
                0,
            )
            <=
            0
        ):
            raise RuntimeError(
                "Training subset lacks "
                f"{actor_class} samples."
            )

    maximum = max(
        counts[
            actor_class
        ]
        for actor_class
        in PRIMARY_CLASSES
    )

    result = {}

    for actor_class in (
        PRIMARY_CLASSES
    ):
        weight = math.sqrt(
            maximum
            /
            counts[
                actor_class
            ]
        )

        result[
            actor_class
        ] = min(
            4.0,
            float(weight),
        )

    result[
        "DEFAULT"
    ] = 1.0

    return result


def build_batch(
    samples,
    *,
    stats,
    class_weights,
    device,
):
    arrays = [
        normalized_sample_arrays(
            sample,
            stats,
        )
        for sample in samples
    ]

    target = torch.from_numpy(
        np.stack(
            [
                item["target"]
                for item in arrays
            ],
            axis=0,
        )
    ).to(
        device
    )

    neighbors = torch.from_numpy(
        np.stack(
            [
                item["neighbors"]
                for item in arrays
            ],
            axis=0,
        )
    ).to(
        device
    )

    neighbor_mask = torch.from_numpy(
        np.stack(
            [
                item[
                    "neighbor_mask"
                ]
                for item in arrays
            ],
            axis=0,
        )
    ).to(
        device
    )

    map_context = torch.from_numpy(
        np.stack(
            [
                item["map"]
                for item in arrays
            ],
            axis=0,
        )
    ).to(
        device
    )

    label = torch.from_numpy(
        np.stack(
            [
                item["label"]
                for item in arrays
            ],
            axis=0,
        )
    ).to(
        device
    )

    valid = torch.from_numpy(
        np.stack(
            [
                item["valid"]
                for item in arrays
            ],
            axis=0,
        )
    ).to(
        device
    )

    weights = torch.tensor(
        [
            float(
                class_weights.get(
                    item[
                        "actor_class"
                    ],
                    class_weights[
                        "DEFAULT"
                    ],
                )
            )
            for item in arrays
        ],
        dtype=torch.float32,
        device=device,
    )

    return (
        target,
        neighbors,
        neighbor_mask,
        map_context,
        label,
        valid,
        weights,
        arrays,
    )


def masked_weighted_loss(
    prediction,
    label,
    valid,
    weights,
):
    element = F.smooth_l1_loss(
        prediction,
        label,
        reduction="none",
    )

    mask = (
        valid[
            :,
            :,
            None
        ]
    )

    coordinate_count = (
        mask.sum(dim=(1, 2))
        *
        3.0
    )

    per_sample = (
        (
            element
            *
            mask
        )
        .sum(dim=(1, 2))
        /
        coordinate_count
        .clamp_min(1.0)
    )

    usable = (
        coordinate_count
        >
        0.0
    ).float()

    effective_weights = (
        weights
        *
        usable
    )

    denominator = (
        effective_weights
        .sum()
        .clamp_min(1e-6)
    )

    return (
        per_sample
        *
        effective_weights
    ).sum() / denominator


def iter_scene_samples(
    records,
    *,
    clean_config,
    degraded_config,
    shuffle_seed=None,
    progress_prefix=None,
):
    order = list(
        records
    )

    if shuffle_seed is not None:
        rng = random.Random(
            shuffle_seed
        )

        rng.shuffle(
            order
        )

    for index, record in enumerate(
        order,
        start=1,
    ):
        scenario = (
            read_training_scenario(
                record
            )
        )

        built = (
            build_real_causal_samples(
                scenario,
                clean_config=(
                    clean_config
                ),
                degraded_config=(
                    degraded_config
                ),
            )
        )

        if (
            progress_prefix
            and
            (
                index % 128 == 0
                or
                index
                ==
                len(order)
            )
        ):
            print(
                f"{progress_prefix}: "
                f"{index}/{len(order)}",
                flush=True,
            )

        yield (
            record,
            built[
                "samples"
            ],
        )


def train_epoch(
    model,
    optimizer,
    records,
    *,
    stats,
    class_weights,
    clean_config,
    degraded_config,
    device,
    epoch_index,
):
    model.train()

    buffer = []

    loss_sum = 0.0
    batch_count = 0
    sample_count = 0

    rng = random.Random(
        SEED
        +
        1000
        +
        epoch_index
    )

    def consume(
        batch_samples,
    ):
        nonlocal (
            loss_sum,
            batch_count,
            sample_count
        )

        (
            target,
            neighbors,
            neighbor_mask,
            map_context,
            label,
            valid,
            weights,
            _,
        ) = build_batch(
            batch_samples,
            stats=stats,
            class_weights=(
                class_weights
            ),
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

        loss = (
            masked_weighted_loss(
                prediction,
                label,
                valid,
                weights,
            )
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite training loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        loss_sum += float(
            loss.detach()
            .cpu()
            .item()
        )

        batch_count += 1

        sample_count += len(
            batch_samples
        )

    for _, samples in (
        iter_scene_samples(
            records,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
            shuffle_seed=(
                SEED
                +
                epoch_index
            ),
            progress_prefix=(
                f"train epoch "
                f"{epoch_index}"
            ),
        )
    ):
        samples = list(
            samples
        )

        rng.shuffle(
            samples
        )

        buffer.extend(
            samples
        )

        while len(
            buffer
        ) >= BATCH_SIZE:
            batch = buffer[
                :BATCH_SIZE
            ]

            del buffer[
                :BATCH_SIZE
            ]

            consume(
                batch
            )

    if buffer:
        consume(
            buffer
        )

    return {
        "mean_loss":
            (
                loss_sum
                /
                max(
                    batch_count,
                    1,
                )
            ),

        "batch_count":
            batch_count,

        "sample_count":
            sample_count,
    }


def evaluate_development(
    model,
    records,
    *,
    stats,
    class_weights,
    clean_config,
    degraded_config,
    device,
):
    model.eval()

    buffer = []

    planar_errors = []
    fde_errors = []

    class_errors = (
        {}
    )

    total_samples = 0
    valid_samples = 0

    label_mean = np.asarray(
        stats.label_mean,
        dtype=np.float32,
    )

    label_std = np.asarray(
        stats.label_std,
        dtype=np.float32,
    )

    def consume(
        batch_samples,
    ):
        nonlocal (
            total_samples,
            valid_samples
        )

        (
            target,
            neighbors,
            neighbor_mask,
            map_context,
            label,
            valid,
            _,
            arrays,
        ) = build_batch(
            batch_samples,
            stats=stats,
            class_weights=(
                class_weights
            ),
            device=device,
        )

        with torch.no_grad():
            prediction = model(
                target,
                neighbors,
                neighbor_mask,
                map_context,
            )

        prediction = (
            prediction
            .detach()
            .cpu()
            .numpy()
        )

        label_np = (
            label
            .detach()
            .cpu()
            .numpy()
        )

        valid_np = (
            valid
            .detach()
            .cpu()
            .numpy()
        )

        pred_disp = (
            prediction
            *
            label_std[
                None,
                :,
                :,
            ]
            +
            label_mean[
                None,
                :,
                :,
            ]
        )

        true_disp = (
            label_np
            *
            label_std[
                None,
                :,
                :,
            ]
            +
            label_mean[
                None,
                :,
                :,
            ]
        )

        for sample_index, (
            sample,
            array_data,
        ) in enumerate(
            zip(
                batch_samples,
                arrays,
            )
        ):
            total_samples += 1

            any_valid = False

            actor_class = (
                array_data[
                    "actor_class"
                ]
            )

            class_errors.setdefault(
                actor_class,
                [],
            )

            for horizon in range(4):
                if (
                    valid_np[
                        sample_index,
                        horizon,
                    ]
                    <=
                    0.5
                ):
                    continue

                any_valid = True

                error = float(
                    np.linalg.norm(
                        pred_disp[
                            sample_index,
                            horizon,
                            :2,
                        ]
                        -
                        true_disp[
                            sample_index,
                            horizon,
                            :2,
                        ]
                    )
                )

                planar_errors.append(
                    error
                )

                class_errors[
                    actor_class
                ].append(
                    error
                )

                if horizon == 3:
                    fde_errors.append(
                        error
                    )

            if any_valid:
                valid_samples += 1

    for _, samples in (
        iter_scene_samples(
            records,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
            shuffle_seed=None,
            progress_prefix=(
                "development"
            ),
        )
    ):
        buffer.extend(
            samples
        )

        while len(
            buffer
        ) >= BATCH_SIZE:
            batch = buffer[
                :BATCH_SIZE
            ]

            del buffer[
                :BATCH_SIZE
            ]

            consume(
                batch
            )

    if buffer:
        consume(
            buffer
        )

    if not planar_errors:
        raise RuntimeError(
            "Development set produced "
            "no valid forecast errors."
        )

    ade = float(
        np.mean(
            planar_errors
        )
    )

    fde = (
        float(
            np.mean(
                fde_errors
            )
        )
        if fde_errors
        else None
    )

    class_ade = {
        actor_class:
            float(
                np.mean(
                    values
                )
            )
        for actor_class, values
        in sorted(
            class_errors.items()
        )
        if values
    }

    if not math.isfinite(
        ade
    ):
        raise RuntimeError(
            "Development ADE "
            "is non-finite."
        )

    return {
        "ADE_m":
            ade,

        "FDE_1s_m":
            fde,

        "valid_error_points":
            len(
                planar_errors
            ),

        "total_samples":
            total_samples,

        "samples_with_any_valid_future":
            valid_samples,

        "class_ADE_m":
            class_ade,
    }


def implementation_fingerprint():
    roots = (
        STAGE4 / "src",
        STAGE4 / "tests",
        STAGE4 / "configs",
        STAGE4 / "scripts",
    )

    files = []

    for root in roots:
        for path in root.rglob("*"):
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


print(
    "============================================================"
)
print(
    "STAGE 4 — BLOCK 4.3 "
    "DETERMINISTIC MULTI-AGENT GRU"
)
print(
    "============================================================"
)


# ============================================================
# Frozen upstream gates
# ============================================================

block42 = json.loads(
    BLOCK42_REPORT.read_text(
        encoding="utf-8"
    )
)

if (
    block42["status"]
    !=
    "PASS"
):
    raise RuntimeError(
        "Block4.2 is not PASS."
    )

if (
    block42[
        "implementation"
    ]["sha256"]
    !=
    EXPECTED_BLOCK42_SHA
):
    raise RuntimeError(
        "Block4.2 implementation "
        "SHA changed."
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

if (
    file_sha256(
        CAL_MANIFEST
    )
    !=
    EXPECTED_CAL_SHA
):
    raise RuntimeError(
        "Frozen calibration "
        "manifest changed."
    )

print(
    "Block4.2 frozen upstream    = PASS"
)

print(
    "fit/dev/cal manifests       = PASS"
)


# ============================================================
# CUDA / deterministic environment
# ============================================================

configure_determinism()

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable."
    )

device = torch.device(
    "cuda:0"
)

gpu_name = (
    torch.cuda
    .get_device_name(0)
)

print(
    "torch                       =",
    torch.__version__,
)

print(
    "CUDA runtime                =",
    torch.version.cuda,
)

print(
    "GPU                         =",
    gpu_name,
)

print(
    "deterministic algorithms    = ON"
)

print(
    "CuDNN GRU path              = DISABLED"
)


# ============================================================
# Verify reusable pipeline is EXACTLY equivalent to frozen
# Block4.2 real pilot before any model training.
# ============================================================

(
    clean_config,
    degraded_config,
) = load_frozen_stage2_configs()

frozen_pilot = (
    block42[
        "real_pilot"
    ][
        "scenarios"
    ]
)

for index, expected in enumerate(
    frozen_pilot,
    start=1,
):
    partition = expected[
        "partition"
    ]

    path = {
        "fit":
            FIT_MANIFEST,

        "development":
            DEV_MANIFEST,

        "calibration":
            CAL_MANIFEST,
    }[
        partition
    ]

    records = read_jsonl(
        path
    )

    matches = [
        record
        for record in records
        if (
            record[
                "scenario_id"
            ]
            ==
            expected[
                "scenario_id"
            ]
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Block4.2 equivalence "
            "scenario resolution failed."
        )

    scenario = (
        read_training_scenario(
            matches[0]
        )
    )

    built = (
        build_real_causal_samples(
            scenario,
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
        )
    )

    if (
        built[
            "algorithm_hash"
        ]
        !=
        expected[
            "stage2_algorithm_sha256"
        ]
    ):
        raise RuntimeError(
            "Stage2 algorithm hash "
            "changed after factoring "
            "real pipeline."
        )

    if (
        built[
            "scene_inputs"
        ].sha256()
        !=
        expected[
            "causal_scene_sha256"
        ]
    ):
        raise RuntimeError(
            "Causal scene hash "
            "changed after factoring "
            "real pipeline."
        )

    if (
        model_payload_hash(
            built["samples"]
        )
        !=
        expected[
            "model_payload_sha256"
        ]
    ):
        raise RuntimeError(
            "Model payload hash "
            "changed after factoring "
            "real pipeline."
        )

    if (
        label_hash(
            built["samples"]
        )
        !=
        expected[
            "label_sha256"
        ]
    ):
        raise RuntimeError(
            "Supervision label hash "
            "changed after factoring "
            "real pipeline."
        )

print(
    "Block4.2 12-scene equivalence= PASS"
)


# ============================================================
# Freeze computationally bounded, class-aware MODEL-DEV
# subsets inside already frozen fit/dev memberships.
#
# This does NOT alter Block4.1 partition membership.
# It does NOT use future or performance.
# ============================================================

fit_records = read_jsonl(
    FIT_MANIFEST
)

dev_records = read_jsonl(
    DEV_MANIFEST
)

train_subset = (
    deterministic_subset(
        fit_records,
        quota=TRAIN_QUOTA,
    )
)

dev_subset = (
    deterministic_subset(
        dev_records,
        quota=DEV_QUOTA,
    )
)

if len(
    train_subset
) != TRAIN_SCENES:
    raise RuntimeError(
        "Training subset count changed."
    )

if len(
    dev_subset
) != DEV_SCENES:
    raise RuntimeError(
        "Development subset count changed."
    )

train_ids = {
    row["scenario_id"]
    for row in train_subset
}

dev_ids = {
    row["scenario_id"]
    for row in dev_subset
}

if train_ids & dev_ids:
    raise RuntimeError(
        "Block4.3 train/dev subset "
        "overlap."
    )

freeze_bytes(
    TRAIN_SUBSET,
    manifest_bytes(
        train_subset
    ),
)

freeze_bytes(
    DEV_SUBSET,
    manifest_bytes(
        dev_subset
    ),
)

train_subset_sha = (
    file_sha256(
        TRAIN_SUBSET
    )
)

dev_subset_sha = (
    file_sha256(
        DEV_SUBSET
    )
)

print(
    "Block4.3 train scenes       =",
    len(train_subset),
)

print(
    "Block4.3 dev scenes         =",
    len(dev_subset),
)

print(
    "train strata                = 512 / 512 / 512"
)

print(
    "dev strata                  = 128 / 128 / 128"
)

print(
    "subset future selection     = NO"
)

print(
    "subset performance selection= NO"
)


# ============================================================
# PASS 1 — fit-only normalization + class support
# ============================================================

stats_accumulator = (
    FitStatsAccumulator()
)

fit_sample_classes = Counter()
fit_sample_count = 0
fit_zero_sample_scenes = 0

normalization_start = (
    time.perf_counter()
)

for index, (
    _,
    samples,
) in enumerate(
    iter_scene_samples(
        train_subset,
        clean_config=(
            clean_config
        ),
        degraded_config=(
            degraded_config
        ),
        shuffle_seed=None,
        progress_prefix=(
            "fit normalization"
        ),
    ),
    start=1,
):
    if not samples:
        fit_zero_sample_scenes += 1

    for sample in samples:
        stats_accumulator.update_sample(
            sample
        )

        fit_sample_classes[
            sample.actor_class
        ] += 1

        fit_sample_count += 1


stats = (
    stats_accumulator
    .finalize()
)

normalization_runtime_s = (
    time.perf_counter()
    -
    normalization_start
)

class_weights = (
    class_weights_from_counts(
        fit_sample_classes
    )
)

stats.write_json(
    NORMALIZATION
)

normalization_sha = (
    file_sha256(
        NORMALIZATION
    )
)

print()
print(
    "fit normalization source    = FIT ONLY"
)

print(
    "fit samples                 =",
    fit_sample_count,
)

print(
    "fit sample classes          =",
    dict(
        sorted(
            fit_sample_classes
            .items()
        )
    ),
)

print(
    "class weights               =",
    class_weights,
)

print(
    "normalization SHA256        =",
    normalization_sha,
)

print(
    "calibration data accessed   = NO"
)

print(
    "formal validation accessed  = NO"
)


# ============================================================
# Deterministic GRU training
# ============================================================

model_config = (
    DeterministicGRUConfig()
)

model = (
    DeterministicTrajectoryGRU(
        model_config
    )
    .to(device)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

training_history = []

best_dev_ade = float(
    "inf"
)

best_epoch = None

training_start = (
    time.perf_counter()
)


for epoch in range(
    1,
    EPOCHS + 1,
):
    epoch_start = (
        time.perf_counter()
    )

    training = train_epoch(
        model,
        optimizer,
        train_subset,
        stats=stats,
        class_weights=(
            class_weights
        ),
        clean_config=(
            clean_config
        ),
        degraded_config=(
            degraded_config
        ),
        device=device,
        epoch_index=epoch,
    )

    development = (
        evaluate_development(
            model,
            dev_subset,
            stats=stats,
            class_weights=(
                class_weights
            ),
            clean_config=(
                clean_config
            ),
            degraded_config=(
                degraded_config
            ),
            device=device,
        )
    )

    epoch_runtime_s = (
        time.perf_counter()
        -
        epoch_start
    )

    row = {
        "epoch":
            epoch,

        "training":
            training,

        "development":
            development,

        "runtime_s":
            epoch_runtime_s,
    }

    training_history.append(
        row
    )

    print()
    print(
        f"epoch {epoch}/{EPOCHS}"
    )

    print(
        "  train loss =",
        training[
            "mean_loss"
        ],
    )

    print(
        "  train samples =",
        training[
            "sample_count"
        ],
    )

    print(
        "  dev ADE m =",
        development[
            "ADE_m"
        ],
    )

    print(
        "  dev FDE@1s m =",
        development[
            "FDE_1s_m"
        ],
    )

    print(
        "  dev class ADE =",
        development[
            "class_ADE_m"
        ],
    )

    print(
        "  runtime s =",
        round(
            epoch_runtime_s,
            3,
        ),
    )

    if (
        development[
            "ADE_m"
        ]
        <
        best_dev_ade
    ):
        best_dev_ade = (
            development[
                "ADE_m"
            ]
        )

        best_epoch = epoch

        checkpoint_payload = {
            "stage":
                4,

            "block":
                "4.3",

            "model_type":
                "deterministic_multi_agent_GRU",

            "model_config":
                model_config.to_dict(),

            "seed":
                SEED,

            "best_epoch":
                best_epoch,

            "best_development_ADE_m":
                best_dev_ade,

            "normalization_sha256":
                normalization_sha,

            "train_subset_sha256":
                train_subset_sha,

            "development_subset_sha256":
                dev_subset_sha,

            "state_dict": {
                key:
                    value
                    .detach()
                    .cpu()
                for key, value
                in (
                    model
                    .state_dict()
                    .items()
                )
            },
        }

        torch.save(
            checkpoint_payload,
            CHECKPOINT,
        )


training_runtime_s = (
    time.perf_counter()
    -
    training_start
)


if best_epoch is None:
    raise RuntimeError(
        "No deterministic checkpoint "
        "was selected."
    )

if not math.isfinite(
    best_dev_ade
):
    raise RuntimeError(
        "Best development ADE "
        "is non-finite."
    )


TRAINING_HISTORY.write_text(
    json.dumps(
        training_history,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# Reload frozen checkpoint and prove exact deterministic
# inference on a fixed development probe.
# ============================================================

checkpoint_sha = (
    file_sha256(
        CHECKPOINT
    )
)

saved = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

if (
    saved[
        "normalization_sha256"
    ]
    !=
    normalization_sha
):
    raise RuntimeError(
        "Checkpoint normalization "
        "SHA mismatch."
    )

reloaded = (
    DeterministicTrajectoryGRU(
        DeterministicGRUConfig(
            **saved[
                "model_config"
            ]
        )
    )
)

reloaded.load_state_dict(
    saved[
        "state_dict"
    ]
)

reloaded = reloaded.to(
    device
)

reloaded.eval()


probe_samples = []

for _, samples in (
    iter_scene_samples(
        dev_subset[:2],
        clean_config=(
            clean_config
        ),
        degraded_config=(
            degraded_config
        ),
        shuffle_seed=None,
        progress_prefix=None,
    )
):
    probe_samples.extend(
        samples
    )

    if len(
        probe_samples
    ) >= 64:
        break

probe_samples = (
    probe_samples[
        :64
    ]
)

if not probe_samples:
    raise RuntimeError(
        "No deterministic inference "
        "probe samples."
    )

(
    probe_target,
    probe_neighbors,
    probe_neighbor_mask,
    probe_map,
    _,
    _,
    _,
    _,
) = build_batch(
    probe_samples,
    stats=stats,
    class_weights=(
        class_weights
    ),
    device=device,
)

with torch.no_grad():
    first = reloaded(
        probe_target,
        probe_neighbors,
        probe_neighbor_mask,
        probe_map,
    )

    second = reloaded(
        probe_target,
        probe_neighbors,
        probe_neighbor_mask,
        probe_map,
    )

torch.cuda.synchronize()

if not torch.equal(
    first,
    second,
):
    raise RuntimeError(
        "Deterministic inference "
        "repeat is not bit-exact."
    )

inference_sha = sha256(
    first
    .detach()
    .cpu()
    .numpy()
    .tobytes()
).hexdigest()


# ============================================================
# Training config freeze
# ============================================================

config = {
    "stage": 4,
    "block": "4.3",
    "status": "FROZEN",

    "purpose":
        "deterministic learned trajectory baseline",

    "training_source": {
        "partition":
            "Block4.1 fit only",

        "scene_count":
            TRAIN_SCENES,

        "selection": (
            "512 cyclist-containing + "
            "512 pedestrian/no-cyclist + "
            "512 vehicle-only"
        ),

        "selection_current_metadata_only":
            True,

        "future_based":
            False,

        "performance_based":
            False,

        "manifest_sha256":
            train_subset_sha,
    },

    "development_source": {
        "partition":
            "Block4.1 development only",

        "scene_count":
            DEV_SCENES,

        "selection": (
            "128 cyclist-containing + "
            "128 pedestrian/no-cyclist + "
            "128 vehicle-only"
        ),

        "manifest_sha256":
            dev_subset_sha,
    },

    "calibration_partition_used":
        False,

    "formal_validation_used":
        False,

    "normalization": {
        "source":
            "fit training subset only",

        "sha256":
            normalization_sha,

        "measurement_covariance_normalized":
            True,

        "observation_masks_not_standardized":
            True,

        "future_label_normalization":
            "fit supervision only",
    },

    "class_policy": {
        "actor_class_numeric_input":
            False,

        "actor_class_used_for_loss_weight":
            True,

        "weight_formula":
            "sqrt(max_primary_count / class_count), capped at 4",

        "weights":
            class_weights,
    },

    "model": (
        model_config
        .to_dict()
    ),

    "training": {
        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "loss":
            "class-weighted masked SmoothL1 on normalized displacement",

        "gradient_clip":
            GRAD_CLIP,

        "device":
            "cuda:0",

        "GPU":
            gpu_name,

        "torch":
            torch.__version__,

        "torch_cuda_runtime":
            torch.version.cuda,

        "deterministic_algorithms":
            True,

        "cudnn_GRU":
            False,

        "TF32":
            False,
    },

    "model_inputs": {
        "measurement_covariance_R_t":
            True,

        "multi_agent_neighbors":
            True,

        "static_map_context":
            True,

        "actor_class":
            False,

        "perfect_track_id":
            False,

        "future":
            False,
    },

    "model_output": (
        "4x3 normalized future displacement; "
        "converted to H0 position by adding "
        "the causal latest observed target position"
    ),

    "not_yet_claimed": [
        "Gaussian predictive covariance",
        "NLL",
        "probabilistic calibration",
        "GMM",
        "formal N=120 evaluation"
    ],
}


CONFIG.write_text(
    json.dumps(
        config,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    +
    "\n",
    encoding="utf-8",
)


# ============================================================
# Full Stage4 regression
# ============================================================

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

import re

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
    46
):
    print(
        combined
    )

    raise RuntimeError(
        "Expected full Stage4 "
        "regression 46/46."
    )


# ============================================================
# Storage + fingerprint
# ============================================================

free_gib = (
    shutil.disk_usage(
        ROOT
    ).free
    /
    1024**3
)

if free_gib < MIN_FREE_GIB:
    raise RuntimeError(
        "Free-space reserve "
        "< 250 GiB."
    )

(
    implementation_files,
    implementation_sha,
) = implementation_fingerprint()


report = {
    "stage": 4,
    "block": "4.3",
    "status": "PASS",

    "pipeline_equivalence": {
        "Block4.2_real_scenarios":
            12,

        "Stage2_algorithm_hashes_exact":
            True,

        "causal_scene_hashes_exact":
            True,

        "model_payload_hashes_exact":
            True,

        "supervision_label_hashes_exact":
            True,
    },

    "training_subset": {
        "scene_count":
            TRAIN_SCENES,

        "manifest_sha256":
            train_subset_sha,

        "fit_sample_count":
            fit_sample_count,

        "sample_classes":
            dict(
                sorted(
                    fit_sample_classes
                    .items()
                )
            ),

        "zero_sample_scenes":
            fit_zero_sample_scenes,
    },

    "development_subset": {
        "scene_count":
            DEV_SCENES,

        "manifest_sha256":
            dev_subset_sha,
    },

    "normalization": {
        "source":
            "fit_only",

        "sha256":
            normalization_sha,

        "runtime_s":
            normalization_runtime_s,

        "calibration_partition_used":
            False,

        "formal_validation_used":
            False,
    },

    "class_weights":
        class_weights,

    "training": {
        "epochs":
            EPOCHS,

        "runtime_s":
            training_runtime_s,

        "history":
            training_history,

        "best_epoch":
            best_epoch,

        "best_development_ADE_m":
            best_dev_ade,
    },

    "checkpoint": {
        "path":
            str(
                CHECKPOINT
            ),

        "sha256":
            checkpoint_sha,

        "normalization_sha256":
            normalization_sha,
    },

    "inference_reproducibility": {
        "probe_samples":
            len(
                probe_samples
            ),

        "bit_exact_repeat":
            True,

        "prediction_sha256":
            inference_sha,
    },

    "architecture": {
        "deterministic_GRU":
            True,

        "measurement_covariance_input":
            True,

        "multi_agent_context":
            True,

        "map_context":
            True,

        "perfect_track_id_feature":
            False,

        "actor_class_numeric_feature":
            False,

        "future_input":
            False,
    },

    "regression": {
        "tests_passed":
            46,

        "tests_total":
            46,

        "pass":
            True,
    },

    "storage": {
        "free_gib":
            free_gib,

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
        "formal Stage3-baseline improvement",
        "Gaussian predictive uncertainty",
        "NLL",
        "coverage calibration",
        "GMM"
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
    "Block4.2 pipeline equivalence = PASS"
)

print(
    "training scenes               =",
    TRAIN_SCENES,
)

print(
    "development scenes            =",
    DEV_SCENES,
)

print(
    "training samples              =",
    fit_sample_count,
)

print(
    "training sample classes       =",
    dict(
        sorted(
            fit_sample_classes
            .items()
        )
    ),
)

print(
    "normalization source          = FIT ONLY"
)

print(
    "calibration data used         = NO"
)

print(
    "formal validation used        = NO"
)

print(
    "measurement covariance R_t    = MODEL INPUT"
)

print(
    "multi-agent context           = MODEL INPUT"
)

print(
    "map context                   = MODEL INPUT"
)

print(
    "actor class numeric feature   = NO"
)

print(
    "perfect track ID feature      = NO"
)

print(
    "future model input            = NO"
)

print(
    "best epoch                    =",
    best_epoch,
)

print(
    "best development ADE m        =",
    best_dev_ade,
)

print(
    "checkpoint SHA256             =",
    checkpoint_sha,
)

print(
    "bit-exact inference repeat    = PASS"
)

print(
    "inference prediction SHA256   =",
    inference_sha,
)

print(
    "full Stage4 regression        = 46 / 46 PASS"
)

print(
    "free GiB                      =",
    round(
        free_gib,
        3,
    ),
)

print(
    "250 GiB reserve               = PASS"
)

print(
    "implementation files          =",
    implementation_files,
)

print(
    "implementation SHA256         =",
    implementation_sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    REPORT,
)


# ============================================================
# Close Block4.3 only after every gate above passed.
# ============================================================

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
            "- The reusable real-data runtime path "
              "was verified against all 12 frozen "
              "Block4.2 real-pilot scenario hashes "
              "before training.\n"
            "- The deterministic learned baseline "
              "is a lightweight GRU operating on "
              "the same frozen Stage2 degraded "
              "observation family used by Stage3.\n"
            "- Measurement covariance R_t remains "
              "an explicit numeric model input.\n"
            "- Target history, up to 8 causal "
              "neighbor histories and lightweight "
              "static map context are encoded.\n"
            "- Actor class is not a numeric model "
              "input; it is used only for fit-loss "
              "weighting to address imbalance.\n"
            "- Perfect track IDs, tracks_to_predict "
              "and future information are not model "
              "inputs.\n"
            "- Training subset membership is chosen "
              "from the frozen fit partition using "
              "current/anchor class metadata only; "
              "future and performance do not affect "
              "selection.\n"
            "- Development subset membership is "
              "chosen analogously from the frozen "
              "development partition.\n"
            "- Normalization statistics are fitted "
              "only on the Block4.3 fit training "
              "subset.\n"
            "- Calibration partition is untouched.\n"
            "- Formal validation is untouched.\n"
            "- Deterministic output is 4x3 future "
              "displacement at 0.1/0.3/0.5/1.0 s, "
              "converted back to H0 by the inference "
              "adapter in later evaluation.\n"
            "- Development ADE, not formal "
              "validation performance, selects the "
              "best deterministic checkpoint.\n"
            "- Frozen checkpoint SHA256: "
            f"{checkpoint_sha}.\n"
            "- Fit normalization SHA256: "
            f"{normalization_sha}.\n"
            "- Exact repeated inference from the "
              "frozen checkpoint is bit-identical.\n"
            "- Gaussian covariance/NLL/calibration "
              "are explicitly not claimed by "
              "Block4.3.\n"
            "- Full Stage4 regression: "
              "46/46 PASS.\n"
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
    "real pipeline         = PASS"
)

print(
    "fit-only normalization= PASS"
)

print(
    "deterministic GRU     = PASS"
)

print(
    "R_t conditioning      = PASS"
)

print(
    "multi-agent context   = PASS"
)

print(
    "map context           = PASS"
)

print(
    "development-only select= PASS"
)

print(
    "formal validation leak= NONE"
)

print(
    "inference reproducible= PASS"
)

print(
    "regression            = 46 / 46"
)

print(
    "implementation SHA    =",
    implementation_sha,
)

print(
    "log closure           = PASS"
)

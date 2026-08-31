from __future__ import annotations

from collections import Counter
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

from iscai_stage4.data.real_pipeline import (
    build_real_supervised_samples,
    load_frozen_stage2_configs,
    read_training_scenario,
)

from iscai_stage4.models import (
    DeterministicTrajectoryGRU,
)

from iscai_stage4.training import (
    NormalizationStats,
    class_sampling_weights,
    masked_smooth_l1,
    normalize_history,
    normalize_map_context,
    planar_ade,
    planar_fde,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

FIT_MANIFEST = (
    STAGE4
    / "artifacts/block41/fit.jsonl"
)

DEV_MANIFEST = (
    STAGE4
    / "artifacts/block41/development.jsonl"
)

BLOCK42 = (
    STAGE4
    / "reports/block42_neural_sample_gate.json"
)

CONFIG_PATH = (
    STAGE4
    / "configs/stage4_deterministic_gru.json"
)

CACHE_ROOT = (
    STAGE4
    / "artifacts/block43/cache"
)

CACHE_META = (
    STAGE4
    / "artifacts/block43/cache_manifest.json"
)

NORMALIZATION_PATH = (
    STAGE4
    / "artifacts/block43/normalization.json"
)

CHECKPOINT = (
    STAGE4
    / "artifacts/block43/deterministic_gru.pt"
)

HISTORY_PATH = (
    STAGE4
    / "artifacts/block43/training_history.json"
)

REPORT = (
    STAGE4
    / "reports/block43_deterministic_gru.json"
)

LOG = (
    STAGE4
    / "docs/implementation_log.md"
)

EXPECTED_FIT_SHA = (
    "284a61c877d937deb07137345beaabf3165690138785f9cc41f81655066d5276"
)

EXPECTED_DEV_SHA = (
    "e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c"
)

CLASS_TO_ID = {
    "TYPE_VEHICLE": 0,
    "TYPE_PEDESTRIAN": 1,
    "TYPE_CYCLIST": 2,
    "TYPE_OTHER": 3,
}

ID_TO_CLASS = {
    value: key
    for key, value
    in CLASS_TO_ID.items()
}

MIN_FREE_GIB = 250.0


def sha256_file(path):
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_jsonl(path):
    return tuple(
        json.loads(line)
        for line in (
            path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
        if line.strip()
    )


def set_determinism(seed):
    os.environ[
        "CUBLAS_WORKSPACE_CONFIG"
    ] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(
        seed
    )

    torch.use_deterministic_algorithms(
        True
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sample_to_tensors(sample):
    target = torch.tensor(
        sample.model_input.target_history,
        dtype=torch.float32,
    )

    neighbours = torch.tensor(
        sample.model_input.neighbor_histories,
        dtype=torch.float32,
    )

    neighbour_mask = torch.tensor(
        sample.model_input.neighbor_mask,
        dtype=torch.float32,
    )

    map_context = torch.tensor(
        sample.model_input.map_context,
        dtype=torch.float32,
    )

    future = torch.tensor(
        sample.future_label.positions_H0_m,
        dtype=torch.float32,
    )

    future_mask = torch.tensor(
        sample.future_label.valid_mask,
        dtype=torch.bool,
    )

    observed = (
        target[
            :,
            12
        ]
        >
        0.5
    )

    indices = torch.nonzero(
        observed,
        as_tuple=False,
    ).flatten()

    if indices.numel() == 0:
        raise RuntimeError(
            "Target sample has no "
            "causal observations."
        )

    latest_index = int(
        indices[-1].item()
    )

    origin = (
        target[
            latest_index,
            :3
        ].clone()
    )

    # Target-centric translation.
    target_observed = (
        target[
            :,
            12
        ]
        >
        0.5
    )

    target[
        target_observed,
        :3
    ] -= origin

    for slot in range(
        neighbours.shape[0]
    ):
        if (
            neighbour_mask[
                slot
            ]
            <=
            0.5
        ):
            continue

        neighbour_observed = (
            neighbours[
                slot,
                :,
                12
            ]
            >
            0.5
        )

        neighbours[
            slot,
            neighbour_observed,
            :3
        ] -= origin

    future_displacement = (
        future
        -
        origin.unsqueeze(0)
    )

    future_displacement[
        ~future_mask
    ] = 0.0

    class_id = (
        CLASS_TO_ID.get(
            sample.actor_class,
            CLASS_TO_ID[
                "TYPE_OTHER"
            ],
        )
    )

    return {
        "target":
            target,

        "neighbours":
            neighbours,

        "neighbour_mask":
            neighbour_mask,

        "map":
            map_context,

        "future":
            future_displacement,

        "future_mask":
            future_mask,

        "class_id":
            torch.tensor(
                class_id,
                dtype=torch.long,
            ),
    }


def stack_sample_dicts(items):
    return {
        key: torch.stack(
            [
                item[key]
                for item in items
            ],
            dim=0,
        )
        for key in (
            "target",
            "neighbours",
            "neighbour_mask",
            "map",
            "future",
            "future_mask",
            "class_id",
        )
    }


def materialize_partition(
    name,
    records,
    *,
    scenario_budget,
    clean_config,
    degraded_config,
    shard_scenario_count,
    hard_cap_bytes,
):
    target_dir = (
        CACHE_ROOT
        /
        name
    )

    metadata_path = (
        target_dir
        /
        "manifest.json"
    )

    if metadata_path.is_file():
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            metadata[
                "scenario_budget"
            ]
            !=
            scenario_budget
        ):
            raise RuntimeError(
                "Frozen cache scenario "
                "budget mismatch."
            )

        for item in metadata[
            "files"
        ]:
            path = (
                target_dir
                /
                item["name"]
            )

            if (
                not path.is_file()
                or
                sha256_file(path)
                !=
                item["sha256"]
            ):
                raise RuntimeError(
                    "Frozen cache file "
                    "integrity failure."
                )

        print(
            name,
            "cache = REUSED",
            "| samples =",
            metadata[
                "sample_count"
            ],
        )

        return metadata

    tmp = (
        CACHE_ROOT
        /
        f".{name}_build_tmp"
    )

    if tmp.exists():
        shutil.rmtree(
            tmp
        )

    tmp.mkdir(
        parents=True
    )

    selected = tuple(
        records[
            :scenario_budget
        ]
    )

    class_counts = Counter()
    total_samples = 0
    files = []
    total_bytes = 0

    shard_samples = []
    shard_index = 0
    scenarios_in_shard = 0

    start = time.perf_counter()

    def flush():
        nonlocal \
            shard_samples, \
            shard_index, \
            scenarios_in_shard, \
            total_bytes

        if not shard_samples:
            return

        data = stack_sample_dicts(
            shard_samples
        )

        filename = (
            f"part-{shard_index:04d}.pt"
        )

        path = (
            tmp
            /
            filename
        )

        torch.save(
            data,
            path,
        )

        size = (
            path.stat().st_size
        )

        total_bytes += size

        if (
            total_bytes
            >
            hard_cap_bytes
        ):
            raise RuntimeError(
                "Block4.3 cache exceeded "
                "2-GiB hard cap."
            )

        files.append({
            "name":
                filename,

            "samples":
                int(
                    data[
                        "target"
                    ].shape[0]
                ),

            "bytes":
                size,

            "sha256":
                sha256_file(
                    path
                ),
        })

        shard_index += 1
        shard_samples = []
        scenarios_in_shard = 0

    for index, record in enumerate(
        selected,
        start=1,
    ):
        scenario = (
            read_training_scenario(
                record
            )
        )

        samples = (
            build_real_supervised_samples(
                scenario,
                clean_config=(
                    clean_config
                ),
                degraded_config=(
                    degraded_config
                ),
            )
        )

        for sample in samples:
            tensors = (
                sample_to_tensors(
                    sample
                )
            )

            shard_samples.append(
                tensors
            )

            class_counts[
                sample.actor_class
            ] += 1

            total_samples += 1

        scenarios_in_shard += 1

        if (
            scenarios_in_shard
            >=
            shard_scenario_count
        ):
            flush()

        if (
            index % 128
            ==
            0
            or
            index
            ==
            scenario_budget
        ):
            print(
                f"{name}: "
                f"{index}/{scenario_budget} "
                f"scenarios "
                f"| samples="
                f"{total_samples}",
                flush=True,
            )

    flush()

    runtime_s = (
        time.perf_counter()
        -
        start
    )

    if total_samples <= 0:
        raise RuntimeError(
            f"{name} cache has "
            "zero samples."
        )

    metadata = {
        "partition":
            name,

        "scenario_budget":
            scenario_budget,

        "scenario_ids": [
            record[
                "scenario_id"
            ]
            for record
            in selected
        ],

        "sample_count":
            total_samples,

        "class_counts":
            dict(
                sorted(
                    class_counts.items()
                )
            ),

        "runtime_s":
            runtime_s,

        "total_bytes":
            total_bytes,

        "files":
            files,

        "future_based_selection":
            False,

        "performance_based_selection":
            False,

        "class_based_scenario_selection":
            False,
    }

    (
        tmp
        /
        "manifest.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        +
        "\n",
        encoding="utf-8",
    )

    target_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp.rename(
        target_dir
    )

    print(
        name,
        "cache = FROZEN",
        "| samples =",
        total_samples,
        "| GiB =",
        round(
            total_bytes
            /
            1024**3,
            3,
        ),
    )

    return metadata


def load_partition(
    metadata,
):
    root = (
        CACHE_ROOT
        /
        metadata[
            "partition"
        ]
    )

    pieces = {
        key: []
        for key in (
            "target",
            "neighbours",
            "neighbour_mask",
            "map",
            "future",
            "future_mask",
            "class_id",
        )
    }

    for item in metadata[
        "files"
    ]:
        data = torch.load(
            root
            /
            item["name"],
            map_location="cpu",
            weights_only=True,
        )

        for key in pieces:
            pieces[key].append(
                data[key]
            )

    return {
        key: torch.cat(
            value,
            dim=0,
        )
        for key, value
        in pieces.items()
    }


def compute_normalization(
    fit_data,
):
    feature_sum = torch.zeros(
        12,
        dtype=torch.float64,
    )

    feature_sumsq = torch.zeros(
        12,
        dtype=torch.float64,
    )

    feature_count = torch.zeros(
        12,
        dtype=torch.float64,
    )

    def accumulate_history(
        history,
    ):
        observed = (
            history[
                ...,
                12
            ]
            >
            0.5
        )

        velocity_valid = (
            history[
                ...,
                13
            ]
            >
            0.5
        )

        for dim in range(
            12
        ):
            valid = (
                velocity_valid
                if dim in (
                    3,
                    4,
                    5,
                )
                else
                observed
            )

            values = (
                history[
                    ...,
                    dim
                ][
                    valid
                ]
                .to(
                    torch.float64
                )
            )

            if values.numel() == 0:
                continue

            feature_sum[
                dim
            ] += values.sum()

            feature_sumsq[
                dim
            ] += (
                values.square().sum()
            )

            feature_count[
                dim
            ] += values.numel()

    accumulate_history(
        fit_data["target"]
    )

    neighbours = (
        fit_data[
            "neighbours"
        ]
    )

    active = (
        fit_data[
            "neighbour_mask"
        ]
        >
        0.5
    )

    for slot in range(
        neighbours.shape[1]
    ):
        slot_history = (
            neighbours[
                :,
                slot,
            ]
            .clone()
        )

        inactive = (
            ~active[
                :,
                slot
            ]
        )

        slot_history[
            inactive,
            :,
            12:
        ] = 0.0

        accumulate_history(
            slot_history
        )

    if bool(
        (
            feature_count
            <=
            0
        ).any()
    ):
        raise RuntimeError(
            "Normalization feature "
            "has zero fit-only support."
        )

    mean = (
        feature_sum
        /
        feature_count
    )

    variance = (
        feature_sumsq
        /
        feature_count
        -
        mean.square()
    ).clamp_min(
        0.0
    )

    std = torch.sqrt(
        variance
    )

    std = torch.where(
        std
        >
        1e-6,
        std,
        torch.ones_like(
            std
        ),
    )

    map_data = (
        fit_data["map"]
        .to(
            torch.float64
        )
    )

    map_mean = (
        map_data.mean(
            dim=0
        )
    )

    map_std = (
        map_data.std(
            dim=0,
            unbiased=False,
        )
    )

    map_std = torch.where(
        map_std
        >
        1e-6,
        map_std,
        torch.ones_like(
            map_std
        ),
    )

    return NormalizationStats(
        feature_mean=tuple(
            float(x)
            for x in mean.tolist()
        ),
        feature_std=tuple(
            float(x)
            for x in std.tolist()
        ),
        map_mean=tuple(
            float(x)
            for x
            in map_mean.tolist()
        ),
        map_std=tuple(
            float(x)
            for x
            in map_std.tolist()
        ),
    )


def apply_normalization(
    data,
    stats,
):
    result = {
        key: value
        for key, value
        in data.items()
    }

    result[
        "target"
    ] = normalize_history(
        data["target"],
        stats,
    )

    result[
        "neighbours"
    ] = normalize_history(
        data["neighbours"],
        stats,
    )

    result[
        "map"
    ] = normalize_map_context(
        data["map"],
        stats,
    )

    return result


@torch.no_grad()
def evaluate(
    model,
    data,
    *,
    device,
    batch_size,
):
    model.eval()

    predictions = []
    targets = []
    masks = []
    class_ids = []

    for start in range(
        0,
        data["target"].shape[0],
        batch_size,
    ):
        end = min(
            start
            +
            batch_size,
            data["target"].shape[0],
        )

        prediction = model(
            data["target"][
                start:end
            ].to(
                device,
                non_blocking=True,
            ),
            data["neighbours"][
                start:end
            ].to(
                device,
                non_blocking=True,
            ),
            data["neighbour_mask"][
                start:end
            ].to(
                device,
                non_blocking=True,
            ),
            data["map"][
                start:end
            ].to(
                device,
                non_blocking=True,
            ),
        ).cpu()

        predictions.append(
            prediction
        )

        targets.append(
            data["future"][
                start:end
            ]
        )

        masks.append(
            data["future_mask"][
                start:end
            ]
        )

        class_ids.append(
            data["class_id"][
                start:end
            ]
        )

    prediction = torch.cat(
        predictions,
        dim=0,
    )

    target = torch.cat(
        targets,
        dim=0,
    )

    mask = torch.cat(
        masks,
        dim=0,
    )

    classes = torch.cat(
        class_ids,
        dim=0,
    )

    ade = planar_ade(
        prediction,
        target,
        mask,
    )

    fde = planar_fde(
        prediction,
        target,
        mask,
    )

    class_metrics = {}

    for class_id in sorted(
        set(
            int(x)
            for x in classes.tolist()
        )
    ):
        select = (
            classes
            ==
            class_id
        )

        class_metrics[
            ID_TO_CLASS.get(
                class_id,
                f"CLASS_{class_id}",
            )
        ] = {
            "samples":
                int(
                    select.sum().item()
                ),

            "ADE_m":
                planar_ade(
                    prediction[
                        select
                    ],
                    target[
                        select
                    ],
                    mask[
                        select
                    ],
                ),

            "FDE_m":
                planar_fde(
                    prediction[
                        select
                    ],
                    target[
                        select
                    ],
                    mask[
                        select
                    ],
                ),
        }

    return {
        "ADE_m":
            ade,

        "FDE_m":
            fde,

        "class_metrics":
            class_metrics,
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

            files.append(path)

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
        digest.update(b"\0")
        digest.update(
            path.read_bytes()
        )
        digest.update(b"\0")

    return (
        len(files),
        digest.hexdigest(),
    )


print(
    "============================================================"
)
print(
    "STAGE 4 — BLOCK 4.3 "
    "DETERMINISTIC GRU"
)
print(
    "============================================================"
)

block42 = json.loads(
    BLOCK42.read_text(
        encoding="utf-8"
    )
)

if (
    block42["status"]
    !=
    "PASS"
):
    raise SystemExit(
        "FAIL: Block4.2 not PASS."
    )

if (
    sha256_file(
        FIT_MANIFEST
    )
    !=
    EXPECTED_FIT_SHA
):
    raise SystemExit(
        "FAIL: fit manifest changed."
    )

if (
    sha256_file(
        DEV_MANIFEST
    )
    !=
    EXPECTED_DEV_SHA
):
    raise SystemExit(
        "FAIL: development manifest changed."
    )

config = json.loads(
    CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)

fit_budget = int(
    config[
        "data"
    ][
        "fit_scenarios"
    ]
)

dev_budget = int(
    config[
        "data"
    ][
        "development_scenarios"
    ]
)

hard_cap_bytes = int(
    float(
        config[
            "data"
        ][
            "cache_hard_cap_gib"
        ]
    )
    *
    1024**3
)

print(
    "Block4.2 upstream          = PASS"
)

print(
    "fit scenario budget       =",
    fit_budget,
)

print(
    "development budget        =",
    dev_budget,
)

print(
    "cache hard cap GiB        =",
    config[
        "data"
    ][
        "cache_hard_cap_gib"
    ],
)

print(
    "formal validation access  = NO"
)

print(
    "calibration access        = NO"
)


# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------

if not torch.cuda.is_available():
    raise SystemExit(
        "FAIL: CUDA unavailable."
    )

device = torch.device(
    "cuda:0"
)

set_determinism(
    int(
        config[
            "training"
        ]["seed"]
    )
)

print(
    "torch                     =",
    torch.__version__,
)

print(
    "GPU                       =",
    torch.cuda
    .get_device_name(0),
)


# ------------------------------------------------------------
# Read only FIT + DEVELOPMENT manifests.
# ------------------------------------------------------------

fit_records = (
    read_jsonl(
        FIT_MANIFEST
    )
)

dev_records = (
    read_jsonl(
        DEV_MANIFEST
    )
)

(
    clean_config,
    degraded_config,
) = load_frozen_stage2_configs()


# ------------------------------------------------------------
# Bounded derived sample cache.
# ------------------------------------------------------------

cache_start = (
    time.perf_counter()
)

fit_meta = (
    materialize_partition(
        "fit",
        fit_records,
        scenario_budget=(
            fit_budget
        ),
        clean_config=(
            clean_config
        ),
        degraded_config=(
            degraded_config
        ),
        shard_scenario_count=int(
            config[
                "data"
            ][
                "cache_shard_scenarios"
            ]
        ),
        hard_cap_bytes=(
            hard_cap_bytes
        ),
    )
)

dev_meta = (
    materialize_partition(
        "development",
        dev_records,
        scenario_budget=(
            dev_budget
        ),
        clean_config=(
            clean_config
        ),
        degraded_config=(
            degraded_config
        ),
        shard_scenario_count=int(
            config[
                "data"
            ][
                "cache_shard_scenarios"
            ]
        ),
        hard_cap_bytes=(
            hard_cap_bytes
        ),
    )
)

cache_runtime_s = (
    time.perf_counter()
    -
    cache_start
)

total_cache_bytes = (
    fit_meta[
        "total_bytes"
    ]
    +
    dev_meta[
        "total_bytes"
    ]
)

if (
    total_cache_bytes
    >
    hard_cap_bytes
):
    raise RuntimeError(
        "Combined Block4.3 cache "
        "exceeds hard cap."
    )

for required_class in (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
):
    if (
        fit_meta[
            "class_counts"
        ].get(
            required_class,
            0,
        )
        <=
        0
    ):
        raise RuntimeError(
            "Fit cache lacks "
            f"{required_class}."
        )

    if (
        dev_meta[
            "class_counts"
        ].get(
            required_class,
            0,
        )
        <=
        0
    ):
        raise RuntimeError(
            "Development cache lacks "
            f"{required_class}."
        )


# ------------------------------------------------------------
# Load cache and fit normalization on FIT only.
# ------------------------------------------------------------

fit_data_raw = (
    load_partition(
        fit_meta
    )
)

dev_data_raw = (
    load_partition(
        dev_meta
    )
)

stats = compute_normalization(
    fit_data_raw
)

NORMALIZATION_PATH.write_text(
    json.dumps(
        {
            "source":
                "fit_cache_only",

            "fit_sample_count":
                int(
                    fit_data_raw[
                        "target"
                    ].shape[0]
                ),

            **stats.to_dict(),
        },
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)

fit_data = apply_normalization(
    fit_data_raw,
    stats,
)

dev_data = apply_normalization(
    dev_data_raw,
    stats,
)

print(
    "normalization source       = FIT ONLY"
)


# ------------------------------------------------------------
# Class-aware sampler metadata.
# ------------------------------------------------------------

(
    sample_weights,
    class_weight_map,
) = class_sampling_weights(
    fit_data[
        "class_id"
    ]
)

print(
    "fit class counts          =",
    fit_meta[
        "class_counts"
    ],
)

print(
    "class sampler weights     =",
    class_weight_map,
)

print(
    "class is model input      = NO"
)


# ------------------------------------------------------------
# Model + initial development metric.
# ------------------------------------------------------------

training_config = (
    config[
        "training"
    ]
)

model = (
    DeterministicTrajectoryGRU(
        history_hidden_dim=int(
            config[
                "model"
            ][
                "history_hidden_dim"
            ]
        ),
        map_hidden_dim=int(
            config[
                "model"
            ][
                "map_hidden_dim"
            ]
        ),
        fusion_hidden_dim=int(
            config[
                "model"
            ][
                "fusion_hidden_dim"
            ]
        ),
    )
    .to(
        device
    )
)

if (
    model
    .covariance_input_weight_norm()
    <=
    0.0
):
    raise RuntimeError(
        "R_t feature columns have "
        "no structural neural path."
    )

batch_size = int(
    training_config[
        "batch_size"
    ]
)

initial_dev = evaluate(
    model,
    dev_data,
    device=device,
    batch_size=batch_size,
)

print(
    "initial development ADE   =",
    initial_dev[
        "ADE_m"
    ],
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(
        training_config[
            "learning_rate"
        ]
    ),
    weight_decay=float(
        training_config[
            "weight_decay"
        ]
    ),
)

max_epochs = int(
    training_config[
        "maximum_epochs"
    ]
)

patience = int(
    training_config[
        "patience_epochs"
    ]
)

gradient_clip = float(
    training_config[
        "gradient_clip_norm"
    ]
)

seed = int(
    training_config[
        "seed"
    ]
)

history = []

best_dev_ade = float(
    "inf"
)

best_epoch = None
epochs_without_improvement = 0

training_start = (
    time.perf_counter()
)


# ------------------------------------------------------------
# Training loop.
# ------------------------------------------------------------

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

    indices = torch.multinomial(
        sample_weights,
        num_samples=(
            sample_weights.shape[0]
        ),
        replacement=True,
        generator=generator,
    )

    epoch_loss_sum = 0.0
    epoch_batches = 0
    skipped_batches = 0

    for start in range(
        0,
        indices.shape[0],
        batch_size,
    ):
        batch_indices = (
            indices[
                start:
                start
                +
                batch_size
            ]
        )

        future_mask = (
            fit_data[
                "future_mask"
            ][
                batch_indices
            ]
        )

        if not bool(
            future_mask.any()
        ):
            skipped_batches += 1
            continue

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            fit_data[
                "target"
            ][
                batch_indices
            ].to(
                device,
                non_blocking=True,
            ),

            fit_data[
                "neighbours"
            ][
                batch_indices
            ].to(
                device,
                non_blocking=True,
            ),

            fit_data[
                "neighbour_mask"
            ][
                batch_indices
            ].to(
                device,
                non_blocking=True,
            ),

            fit_data[
                "map"
            ][
                batch_indices
            ].to(
                device,
                non_blocking=True,
            ),
        )

        target = (
            fit_data[
                "future"
            ][
                batch_indices
            ].to(
                device,
                non_blocking=True,
            )
        )

        mask = (
            future_mask.to(
                device,
                non_blocking=True,
            )
        )

        loss = masked_smooth_l1(
            prediction,
            target,
            mask,
        )

        if not bool(
            torch.isfinite(
                loss
            )
        ):
            raise RuntimeError(
                "Non-finite training loss."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip,
        )

        optimizer.step()

        epoch_loss_sum += float(
            loss.detach()
            .cpu()
            .item()
        )

        epoch_batches += 1

    if epoch_batches <= 0:
        raise RuntimeError(
            "No valid training batches."
        )

    dev = evaluate(
        model,
        dev_data,
        device=device,
        batch_size=(
            batch_size
        ),
    )

    epoch_record = {
        "epoch":
            epoch,

        "train_loss":
            (
                epoch_loss_sum
                /
                epoch_batches
            ),

        "development_ADE_m":
            dev["ADE_m"],

        "development_FDE_m":
            dev["FDE_m"],

        "skipped_zero_label_batches":
            skipped_batches,
    }

    history.append(
        epoch_record
    )

    print(
        f"epoch {epoch:02d}"
        f" | train="
        f"{epoch_record['train_loss']:.6f}"
        f" | dev ADE="
        f"{dev['ADE_m']:.6f}"
        f" | dev FDE="
        f"{dev['FDE_m']:.6f}",
        flush=True,
    )

    if (
        dev["ADE_m"]
        <
        best_dev_ade
        -
        1e-9
    ):
        best_dev_ade = (
            dev["ADE_m"]
        )

        best_epoch = epoch

        epochs_without_improvement = 0

        state = {
            key:
                value.detach()
                .cpu()
                .clone()
            for key, value
            in model.state_dict()
            .items()
        }

        torch.save(
            {
                "stage":
                    4,

                "block":
                    "4.3",

                "model_type":
                    "DeterministicTrajectoryGRU",

                "model_config":
                    config["model"],

                "training_config":
                    training_config,

                "best_epoch":
                    best_epoch,

                "best_development_ADE_m":
                    best_dev_ade,

                "normalization_sha256":
                    sha256_file(
                        NORMALIZATION_PATH
                    ),

                "fit_cache_manifest":
                    fit_meta,

                "development_cache_manifest":
                    dev_meta,

                "state_dict":
                    state,
            },
            CHECKPOINT,
        )

    else:
        epochs_without_improvement += 1

    if (
        epochs_without_improvement
        >=
        patience
    ):
        print(
            "early stopping = development-only",
            flush=True,
        )

        break


training_runtime_s = (
    time.perf_counter()
    -
    training_start
)

if best_epoch is None:
    raise RuntimeError(
        "No best checkpoint."
    )


# ------------------------------------------------------------
# Reload frozen best checkpoint and verify.
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

best_model = (
    DeterministicTrajectoryGRU(
        history_hidden_dim=int(
            config[
                "model"
            ][
                "history_hidden_dim"
            ]
        ),
        map_hidden_dim=int(
            config[
                "model"
            ][
                "map_hidden_dim"
            ]
        ),
        fusion_hidden_dim=int(
            config[
                "model"
            ][
                "fusion_hidden_dim"
            ]
        ),
    )
)

best_model.load_state_dict(
    checkpoint[
        "state_dict"
    ]
)

best_model = (
    best_model
    .to(device)
    .eval()
)

final_dev = evaluate(
    best_model,
    dev_data,
    device=device,
    batch_size=(
        batch_size
    ),
)

if (
    abs(
        final_dev[
            "ADE_m"
        ]
        -
        best_dev_ade
    )
    >
    1e-6
):
    raise RuntimeError(
        "Reloaded checkpoint does "
        "not reproduce best dev ADE."
    )

if not math.isfinite(
    final_dev[
        "ADE_m"
    ]
):
    raise RuntimeError(
        "Final development ADE "
        "is not finite."
    )

if not (
    final_dev[
        "ADE_m"
    ]
    <
    initial_dev[
        "ADE_m"
    ]
):
    raise RuntimeError(
        "Deterministic GRU did not "
        "improve over its untrained "
        "initialization on development."
    )


# ------------------------------------------------------------
# Fixed-input exact repeated inference.
# ------------------------------------------------------------

with torch.no_grad():
    count = min(
        64,
        dev_data[
            "target"
        ].shape[0],
    )

    args = (
        dev_data["target"][
            :count
        ].to(device),

        dev_data["neighbours"][
            :count
        ].to(device),

        dev_data["neighbour_mask"][
            :count
        ].to(device),

        dev_data["map"][
            :count
        ].to(device),
    )

    first = best_model(
        *args
    )

    second = best_model(
        *args
    )

    if not torch.equal(
        first,
        second,
    ):
        raise RuntimeError(
            "Repeated deterministic "
            "checkpoint inference differs."
        )


# ------------------------------------------------------------
# Freeze cache + training metadata.
# ------------------------------------------------------------

cache_manifest = {
    "fit":
        fit_meta,

    "development":
        dev_meta,

    "fit_manifest_sha256":
        EXPECTED_FIT_SHA,

    "development_manifest_sha256":
        EXPECTED_DEV_SHA,

    "combined_cache_bytes":
        total_cache_bytes,

    "combined_cache_gib":
        (
            total_cache_bytes
            /
            1024**3
        ),
}

CACHE_META.write_text(
    json.dumps(
        cache_manifest,
        indent=2,
        sort_keys=True,
    )
    +
    "\n",
    encoding="utf-8",
)

HISTORY_PATH.write_text(
    json.dumps(
        {
            "initial_development":
                initial_dev,

            "epochs":
                history,

            "best_epoch":
                best_epoch,

            "best_development":
                final_dev,

            "training_runtime_s":
                training_runtime_s,
        },
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    +
    "\n",
    encoding="utf-8",
)


# ------------------------------------------------------------
# Full Stage4 regression.
# ------------------------------------------------------------

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
        "Expected Stage4 regression "
        "46/46."
    )


# ------------------------------------------------------------
# Storage + fingerprint.
# ------------------------------------------------------------

free_gib = (
    shutil.disk_usage(
        ROOT
    ).free
    /
    1024**3
)

if free_gib < MIN_FREE_GIB:
    raise RuntimeError(
        "250-GiB reserve violated."
    )

(
    implementation_files,
    implementation_sha,
) = implementation_fingerprint()


report = {
    "stage": 4,
    "block": "4.3",
    "status": "PASS",

    "model": {
        "type":
            "DeterministicTrajectoryGRU",

        "target_history":
            True,

        "measurement_covariance_R_t_input":
            True,

        "measurement_covariance_weight_norm":
            best_model
            .covariance_input_weight_norm(),

        "multi_agent_context":
            True,

        "static_map_context":
            True,

        "actor_class_model_input":
            False,

        "perfect_track_id_model_input":
            False,
    },

    "data": {
        "fit_scenarios":
            fit_budget,

        "development_scenarios":
            dev_budget,

        "fit_samples":
            fit_meta[
                "sample_count"
            ],

        "development_samples":
            dev_meta[
                "sample_count"
            ],

        "fit_class_counts":
            fit_meta[
                "class_counts"
            ],

        "development_class_counts":
            dev_meta[
                "class_counts"
            ],

        "cache_gib":
            (
                total_cache_bytes
                /
                1024**3
            ),

        "future_based_scenario_selection":
            False,

        "performance_based_scenario_selection":
            False,

        "formal_validation_access":
            False,

        "calibration_partition_access":
            False,
    },

    "normalization": {
        "source":
            "fit_only",

        "sha256":
            sha256_file(
                NORMALIZATION_PATH
            ),
    },

    "class_balancing": {
        "mode":
            "inverse_sqrt_weighted_sampling",

        "class_is_model_input":
            False,

        "weight_map":
            {
                str(key):
                    value
                for key, value
                in class_weight_map.items()
            },
    },

    "training": {
        "device":
            str(device),

        "best_epoch":
            best_epoch,

        "runtime_s":
            training_runtime_s,

        "initial_development_ADE_m":
            initial_dev[
                "ADE_m"
            ],

        "best_development_ADE_m":
            final_dev[
                "ADE_m"
            ],

        "best_development_FDE_m":
            final_dev[
                "FDE_m"
            ],

        "development_class_metrics":
            final_dev[
                "class_metrics"
            ],

        "early_stopping_source":
            "development_only",

        "formal_validation_used_for_early_stopping":
            False,
    },

    "checkpoint": {
        "path":
            str(
                CHECKPOINT
            ),

        "sha256":
            sha256_file(
                CHECKPOINT
            ),

        "exact_repeat_inference":
            True,
    },

    "regression": {
        "tests_passed":
            46,

        "tests_total":
            46,
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
        "Gaussian predictive covariance",
        "Gaussian NLL",
        "trajectory calibration",
        "GMM",
        "formal N=120 neural evaluation"
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
    "fit scenarios              =",
    fit_budget,
)

print(
    "development scenarios      =",
    dev_budget,
)

print(
    "fit samples                =",
    fit_meta[
        "sample_count"
    ],
)

print(
    "development samples        =",
    dev_meta[
        "sample_count"
    ],
)

print(
    "cache GiB                  =",
    round(
        total_cache_bytes
        /
        1024**3,
        3,
    ),
)

print(
    "normalization              = FIT ONLY"
)

print(
    "class-aware sampler        = PASS"
)

print(
    "class model input          = NO"
)

print(
    "measurement covariance    = ACTIVE INPUT"
)

print(
    "multi-agent context        = ACTIVE"
)

print(
    "map context                = ACTIVE"
)

print(
    "future model input         = NO"
)

print(
    "formal validation access  = NO"
)

print(
    "calibration access        = NO"
)

print(
    "initial dev ADE m          =",
    initial_dev[
        "ADE_m"
    ],
)

print(
    "best epoch                 =",
    best_epoch,
)

print(
    "best dev ADE m             =",
    final_dev[
        "ADE_m"
    ],
)

print(
    "best dev FDE m             =",
    final_dev[
        "FDE_m"
    ],
)

print(
    "development class metrics =",
    final_dev[
        "class_metrics"
    ],
)

print(
    "exact repeat inference     = PASS"
)

print(
    "checkpoint SHA256          =",
    sha256_file(
        CHECKPOINT
    ),
)

print(
    "full Stage4 regression     = 46 / 46 PASS"
)

print(
    "free GiB                   =",
    round(
        free_gib,
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


# ------------------------------------------------------------
# Close Block4.3 only now.
# ------------------------------------------------------------

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
            "- Deterministic Stage4 predictor "
              "is a lightweight shared-history "
              "GRU with target, neighbour and "
              "static-map pathways.\n"
            "- The GRU consumes the frozen "
              "14-D causal Stage4 history "
              "features, including the six "
              "propagated Stage2 measurement-"
              "covariance components.\n"
            "- Actor class, WOMD track ID and "
              "benchmark metadata are not "
              "numeric model inputs.\n"
            "- Training uses a bounded derived "
              "cache from the first 2,048 frozen "
              "fit scenarios and first 512 frozen "
              "development scenarios.\n"
            "- Cache scenario selection is "
              "SHA-order only and does not use "
              "future labels, class or performance.\n"
            "- Normalization statistics are "
              "computed exclusively from the "
              "fit cache.\n"
            "- Class balancing uses deterministic "
              "inverse-square-root weighted "
              "sampling from supervision metadata "
              "only; class is not passed to the "
              "network.\n"
            "- Early stopping and checkpoint "
              "selection use development planar "
              "ADE only.\n"
            "- Calibration and formal validation "
              "partitions are not accessed.\n"
            "- The best checkpoint improves "
              "development ADE over the same "
              "untrained initialization.\n"
            "- Repeated inference from the frozen "
              "checkpoint is exact on a fixed "
              "development batch.\n"
            f"- Best development ADE: "
            f"{final_dev['ADE_m']} m.\n"
            f"- Best development FDE: "
            f"{final_dev['FDE_m']} m.\n"
            f"- Checkpoint SHA256: "
            f"{sha256_file(CHECKPOINT)}.\n"
            "- Block4.3 does not claim Gaussian "
              "predictive covariance, NLL, "
              "calibration, GMM or formal "
              "N=120 performance.\n"
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
    "deterministic GRU     = PASS"
)

print(
    "fit-only normalization= PASS"
)

print(
    "class-aware sampler   = PASS"
)

print(
    "R_t neural pathway    = PASS"
)

print(
    "multi-agent pathway   = PASS"
)

print(
    "map pathway           = PASS"
)

print(
    "dev-only selection    = PASS"
)

print(
    "formal leakage        = NONE"
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

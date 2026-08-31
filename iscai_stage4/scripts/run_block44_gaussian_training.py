from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from iscai_stage4.ml import (
    CHI_SQUARE_3_THRESHOLDS,
    CLASS_CYCLIST,
    CLASS_ID_TO_NAME,
    CLASS_OTHER,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
    GaussianTrajectoryGRU,
    balanced_class_weights,
    covariance_from_scale_tril,
    denormalize_gaussian,
    gaussian_nll_per_horizon,
    initialize_from_deterministic,
    mahalanobis_squared,
    masked_gaussian_nll,
    set_global_determinism,
)


ROOT = Path("/home/agni/waymo")
STAGE4 = ROOT / "iscai_stage4"

PART1 = (
    STAGE4
    / "reports/block44_part1_safe_gate.json"
)

PREFLIGHT = (
    STAGE4
    / "reports/block44_preflight.json"
)

BLOCK43 = (
    STAGE4
    / "reports/block43_deterministic_gru.json"
)

CONFIG = (
    STAGE4
    / "configs/stage4_gaussian_gru.json"
)

DET_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/deterministic_gru.pt"
)

FIT_CACHE = (
    STAGE4
    / "artifacts/block43/fit_cache.npz"
)

DEV_CACHE = (
    STAGE4
    / "artifacts/block43/development_cache.npz"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/fit_normalization.json"
)

CHECKPOINT = (
    STAGE4
    / "artifacts/block44/gaussian_gru.pt"
)

PROGRESS = (
    STAGE4
    / "artifacts/block44/training_progress.pt"
)

REPORT = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

FAILURE = (
    STAGE4
    / "reports/block44_training_failure.json"
)

LOG = (
    STAGE4
    / "docs/implementation_log.md"
)

EXPECTED_BLOCK43_IMPL_SHA = (
    "7f42e0cf534ed3ffb8d69e7230f5e04e"
    "ecde8639428911017af771c7b2993def"
)

EXPECTED_DET_CHECKPOINT_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_DET_STATE_SHA = (
    "d2ffbc03c7cb2826fef2175c95f48725"
    "707ec6d791eaeffbd6f59bc507b8595a"
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

CONFIDENCE_LEVELS = (
    0.50,
    0.80,
    0.90,
    0.95,
    0.99,
)


# ============================================================
# Generic fingerprints / safe writes
# ============================================================

def file_sha256(
    path: Path,
) -> str:
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


def state_dict_sha256(
    state_dict,
) -> str:
    digest = sha256()

    for key in sorted(state_dict):
        tensor = (
            state_dict[key]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            key.encode("utf-8")
        )
        digest.update(b"\0")

        digest.update(
            str(tensor.dtype)
            .encode("ascii")
        )
        digest.update(b"\0")

        digest.update(
            tensor.numpy()
            .tobytes()
        )
        digest.update(b"\0")

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
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
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
            relative.encode("utf-8")
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


def atomic_torch_save(
    payload,
    path: Path,
):
    temporary = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def write_json(
    path: Path,
    payload,
):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


# ============================================================
# Frozen Block4.3 cache
# ============================================================

def load_cache(
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

    n = int(
        arrays["target"].shape[0]
    )

    expected = {
        "target":
            (n, 11, 14),

        "neighbors":
            (n, 8, 11, 14),

        "neighbor_mask":
            (n, 8),

        "map_context":
            (n, 10),

        "future":
            (n, 4, 3),

        "future_mask":
            (n, 4),

        "class_id":
            (n,),
    }

    for key, shape in expected.items():
        if tuple(
            arrays[key].shape
        ) != shape:
            raise RuntimeError(
                f"{path.name}: "
                f"{key} shape changed: "
                f"{arrays[key].shape}"
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
            raise RuntimeError(
                f"Non-finite values "
                f"in {path.name}:{key}"
            )

    return arrays


# ============================================================
# Same fit-only normalization semantics as Block4.3
# ============================================================

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
            "Target without any "
            "causal observation."
        )

    grid = torch.arange(
        target.shape[1],
        device=target.device,
        dtype=torch.long,
    )

    grid = grid[
        None,
        :
    ].expand_as(
        observed
    )

    minus_one = torch.full_like(
        grid,
        -1,
    )

    indices = torch.where(
        observed,
        grid,
        minus_one,
    ).max(
        dim=1
    ).values

    batch = torch.arange(
        target.shape[0],
        device=target.device,
    )

    return target[
        batch,
        indices,
        :3,
    ]


class GaussianNormalizer:

    def __init__(
        self,
        statistics,
        *,
        device,
    ):
        self.feature_mean = torch.tensor(
            statistics[
                "continuous_feature_mean"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.feature_std = torch.tensor(
            statistics[
                "continuous_feature_std"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.map_mean = torch.tensor(
            statistics["map_mean"],
            dtype=torch.float32,
            device=device,
        )

        self.map_std = torch.tensor(
            statistics["map_std"],
            dtype=torch.float32,
            device=device,
        )

        self.label_mean = torch.tensor(
            statistics[
                "label_displacement_mean"
            ],
            dtype=torch.float32,
            device=device,
        )

        self.label_std = torch.tensor(
            statistics[
                "label_displacement_std"
            ],
            dtype=torch.float32,
            device=device,
        )

        if bool(
            torch.any(
                self.feature_std
                <=
                0.0
            )
        ):
            raise RuntimeError(
                "Invalid frozen "
                "feature std."
            )

        if bool(
            torch.any(
                self.map_std
                <=
                0.0
            )
        ):
            raise RuntimeError(
                "Invalid frozen map std."
            )

        if bool(
            torch.any(
                self.label_std
                <=
                0.0
            )
        ):
            raise RuntimeError(
                "Invalid frozen label std."
            )

    def prepare(
        self,
        arrays,
        indices,
        *,
        device,
    ):
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        target = torch.from_numpy(
            np.asarray(
                arrays["target"][
                    indices
                ],
                dtype=np.float32,
            )
        ).to(device)

        neighbors = torch.from_numpy(
            np.asarray(
                arrays["neighbors"][
                    indices
                ],
                dtype=np.float32,
            )
        ).to(device)

        neighbor_mask = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "neighbor_mask"
                    ][indices],
                    dtype=np.float32,
                )
            )
            .to(device)
        )

        map_context = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "map_context"
                    ][indices],
                    dtype=np.float32,
                )
            )
            .to(device)
        )

        future = torch.from_numpy(
            np.asarray(
                arrays["future"][
                    indices
                ],
                dtype=np.float32,
            )
        ).to(device)

        future_mask = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "future_mask"
                    ][indices],
                    dtype=np.float32,
                )
            )
            .to(device)
        )

        class_id = (
            torch.from_numpy(
                np.asarray(
                    arrays[
                        "class_id"
                    ][indices],
                    dtype=np.int64,
                )
            )
            .to(device)
        )

        target_observed = (
            target[
                :,
                :,
                12:13
            ]
        )

        target_continuous = (
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

        target_continuous = (
            target_continuous
            *
            target_observed
        )

        target_normalized = (
            torch.cat(
                (
                    target_continuous,
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

        neighbor_continuous = (
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

        neighbor_continuous = (
            neighbor_continuous
            *
            neighbor_observed
        )

        neighbors_normalized = (
            torch.cat(
                (
                    neighbor_continuous,

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

        origin = (
            latest_position_torch(
                target
            )
        )

        displacement = (
            future
            -
            origin[
                :,
                None,
                :
            ]
        )

        normalized_label = (
            (
                displacement
                -
                self.label_mean[
                    None,
                    :,
                    :
                ]
            )
            /
            self.label_std[
                None,
                :,
                :
            ]
        )

        normalized_label = (
            normalized_label
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
                normalized_label,

            "true_displacement":
                displacement,

            "future_mask":
                future_mask,

            "class_id":
                class_id,
        }


# ============================================================
# Model construction
# ============================================================

def build_gaussian_model(
    deterministic_configuration,
    *,
    device,
):
    section = (
        deterministic_configuration[
            "model"
        ]
    )

    return (
        GaussianTrajectoryGRU(
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
        )
        .to(device)
    )


# ============================================================
# Development evaluation
# ============================================================

def evaluate_gaussian(
    model,
    arrays,
    normalizer,
    *,
    device,
    batch_size=1024,
):
    model.eval()

    nll_sum = 0.0
    valid_total = 0

    horizon_nll_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_error_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_valid = np.zeros(
        4,
        dtype=np.int64,
    )

    planar_error_sum = 0.0

    class_error_sum = Counter()
    class_valid = Counter()

    coverage_inside = {
        level: 0
        for level
        in CONFIDENCE_LEVELS
    }

    coverage_inside_by_horizon = {
        level:
            np.zeros(
                4,
                dtype=np.int64,
            )
        for level
        in CONFIDENCE_LEVELS
    }

    predictive_std_sum = np.zeros(
        (
            4,
            3,
        ),
        dtype=np.float64,
    )

    predictive_std_count = np.zeros(
        4,
        dtype=np.int64,
    )

    spd_failure_count = 0

    sample_count = int(
        arrays["target"].shape[0]
    )

    with torch.inference_mode():
        for start in range(
            0,
            sample_count,
            batch_size,
        ):
            stop = min(
                sample_count,
                start
                +
                batch_size,
            )

            indices = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            batch = normalizer.prepare(
                arrays,
                indices,
                device=device,
            )

            output = model(
                batch["target"],
                batch["neighbors"],
                batch[
                    "neighbor_mask"
                ],
                batch["map_context"],
            )

            (
                mean_metric,
                scale_metric,
            ) = denormalize_gaussian(
                output.mean,
                output.scale_tril,
                normalizer.label_mean,
                normalizer.label_std,
            )

            target_metric = (
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

            nll = (
                gaussian_nll_per_horizon(
                    mean_metric,
                    scale_metric,
                    target_metric,
                )
            )

            if not bool(
                torch.isfinite(
                    nll[
                        valid
                    ]
                ).all()
            ):
                raise RuntimeError(
                    "Non-finite metric "
                    "development NLL."
                )

            nll_sum += float(
                nll[
                    valid
                ].sum().item()
            )

            batch_valid = int(
                valid.sum().item()
            )

            valid_total += (
                batch_valid
            )

            planar = torch.sqrt(
                (
                    mean_metric[
                        :,
                        :,
                        0
                    ]
                    -
                    target_metric[
                        :,
                        :,
                        0
                    ]
                ).square()
                +
                (
                    mean_metric[
                        :,
                        :,
                        1
                    ]
                    -
                    target_metric[
                        :,
                        :,
                        1
                    ]
                ).square()
            )

            planar_error_sum += float(
                planar[
                    valid
                ].sum().item()
            )

            mahalanobis = (
                mahalanobis_squared(
                    mean_metric,
                    scale_metric,
                    target_metric,
                )
            )

            covariance = (
                covariance_from_scale_tril(
                    scale_metric
                )
            )

            (
                _,
                info,
            ) = torch.linalg.cholesky_ex(
                covariance
            )

            spd_failure_count += int(
                (
                    info
                    !=
                    0
                ).sum().item()
            )

            predictive_std = torch.sqrt(
                torch.diagonal(
                    covariance,
                    dim1=-2,
                    dim2=-1,
                )
                .clamp_min(0.0)
            )

            for horizon in range(4):
                hmask = (
                    valid[
                        :,
                        horizon
                    ]
                )

                hcount = int(
                    hmask.sum().item()
                )

                if hcount <= 0:
                    continue

                horizon_valid[
                    horizon
                ] += (
                    hcount
                )

                horizon_nll_sum[
                    horizon
                ] += float(
                    nll[
                        :,
                        horizon
                    ][
                        hmask
                    ].sum().item()
                )

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

                predictive_std_sum[
                    horizon
                ] += (
                    predictive_std[
                        :,
                        horizon,
                        :
                    ][
                        hmask
                    ]
                    .sum(
                        dim=0
                    )
                    .cpu()
                    .numpy()
                )

                predictive_std_count[
                    horizon
                ] += (
                    hcount
                )

            for level in (
                CONFIDENCE_LEVELS
            ):
                threshold = (
                    CHI_SQUARE_3_THRESHOLDS[
                        level
                    ]
                )

                inside = (
                    mahalanobis
                    <=
                    threshold
                )

                coverage_inside[
                    level
                ] += int(
                    (
                        inside
                        &
                        valid
                    ).sum().item()
                )

                for horizon in range(4):
                    coverage_inside_by_horizon[
                        level
                    ][
                        horizon
                    ] += int(
                        (
                            inside[
                                :,
                                horizon
                            ]
                            &
                            valid[
                                :,
                                horizon
                            ]
                        )
                        .sum()
                        .item()
                    )

            classes = batch[
                "class_id"
            ]

            for class_id in (
                CLASS_VEHICLE,
                CLASS_PEDESTRIAN,
                CLASS_CYCLIST,
                CLASS_OTHER,
            ):
                class_mask = (
                    (
                        classes
                        ==
                        class_id
                    )[
                        :,
                        None
                    ]
                    &
                    valid
                )

                count = int(
                    class_mask
                    .sum()
                    .item()
                )

                if count <= 0:
                    continue

                name = (
                    CLASS_ID_TO_NAME[
                        class_id
                    ]
                )

                class_error_sum[
                    name
                ] += float(
                    planar[
                        class_mask
                    ].sum().item()
                )

                class_valid[
                    name
                ] += (
                    count
                )

    if valid_total <= 0:
        raise RuntimeError(
            "Development set has "
            "zero valid future labels."
        )

    if spd_failure_count != 0:
        raise RuntimeError(
            "Predictive metric covariance "
            f"SPD failures = "
            f"{spd_failure_count}"
        )

    horizon_nll = {}
    horizon_error = {}
    std_by_horizon = {}

    horizon_names = (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    )

    for horizon, name in enumerate(
        horizon_names
    ):
        count = int(
            horizon_valid[
                horizon
            ]
        )

        if count <= 0:
            raise RuntimeError(
                "Zero development labels "
                f"at horizon {name}s."
            )

        horizon_nll[
            name
        ] = float(
            horizon_nll_sum[
                horizon
            ]
            /
            count
        )

        horizon_error[
            name
        ] = float(
            horizon_error_sum[
                horizon
            ]
            /
            count
        )

        std_by_horizon[
            name
        ] = [
            float(x)
            for x in (
                predictive_std_sum[
                    horizon
                ]
                /
                predictive_std_count[
                    horizon
                ]
            ).tolist()
        ]

    coverage = {}

    for level in CONFIDENCE_LEVELS:
        key = f"{level:.2f}"

        coverage[key] = {
            "nominal":
                float(level),

            "empirical":
                float(
                    coverage_inside[
                        level
                    ]
                    /
                    valid_total
                ),

            "inside":
                int(
                    coverage_inside[
                        level
                    ]
                ),

            "valid":
                int(
                    valid_total
                ),

            "by_horizon": {
                horizon_names[
                    horizon
                ]:
                    float(
                        coverage_inside_by_horizon[
                            level
                        ][
                            horizon
                        ]
                        /
                        horizon_valid[
                            horizon
                        ]
                    )
                for horizon
                in range(4)
            },
        }

    class_ade = {
        name:
            float(
                class_error_sum[
                    name
                ]
                /
                class_valid[
                    name
                ]
            )
        for name in sorted(
            class_valid
        )
    }

    return {
        "metric_Gaussian_NLL":
            float(
                nll_sum
                /
                valid_total
            ),

        "planar_ADE_m":
            float(
                planar_error_sum
                /
                valid_total
            ),

        "horizon_metric_NLL":
            horizon_nll,

        "horizon_planar_error_m":
            horizon_error,

        "one_second_planar_error_m":
            horizon_error[
                "1.0"
            ],

        "class_planar_ADE_m":
            class_ade,

        "raw_empirical_coverage":
            coverage,

        "mean_predictive_std_m":
            std_by_horizon,

        "valid_actor_horizon_points":
            int(
                valid_total
            ),

        "predictive_covariance_SPD_failures":
            0,
    }


# ============================================================
# Exact repeated probabilistic inference hash
# ============================================================

def probabilistic_prediction_sha256(
    model,
    arrays,
    normalizer,
    *,
    device,
    sample_limit=2048,
):
    digest = sha256()

    model.eval()

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
                start + 512,
            )

            indices = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            batch = (
                normalizer.prepare(
                    arrays,
                    indices,
                    device=device,
                )
            )

            output = model(
                batch["target"],
                batch["neighbors"],
                batch[
                    "neighbor_mask"
                ],
                batch["map_context"],
            )

            for tensor in (
                output.mean,
                output.scale_tril,
            ):
                value = (
                    tensor
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                digest.update(
                    value.tobytes(
                        order="C"
                    )
                )

    return digest.hexdigest()


# ============================================================
# Checkpoint/progress
# ============================================================

def save_best_checkpoint(
    model,
    *,
    epoch,
    metrics,
    fingerprints,
    configuration,
):
    state_sha = (
        state_dict_sha256(
            model.state_dict()
        )
    )

    atomic_torch_save(
        {
            "stage":
                4,

            "block":
                "4.4",

            "model_type":
                "GaussianTrajectoryGRU",

            "state_dict":
                model.state_dict(),

            "state_dict_sha256":
                state_sha,

            "best_epoch":
                int(epoch),

            "development_metrics":
                metrics,

            "fingerprints":
                fingerprints,

            "configuration":
                configuration,

            "calibrated":
                False,
        },
        CHECKPOINT,
    )


def save_progress(
    model,
    optimizer,
    *,
    last_epoch,
    best_epoch,
    best_nll,
    best_ade,
    patience,
    history,
    initial_metrics,
    fingerprints,
):
    atomic_torch_save(
        {
            "last_completed_epoch":
                int(
                    last_epoch
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "best_nll":
                float(
                    best_nll
                ),

            "best_ade":
                float(
                    best_ade
                ),

            "patience":
                int(
                    patience
                ),

            "history":
                history,

            "initial_metrics":
                initial_metrics,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "fingerprints":
                fingerprints,
        },
        PROGRESS,
    )


def load_valid_progress(
    model,
    optimizer,
    *,
    device,
    fingerprints,
):
    if not PROGRESS.is_file():
        return None

    try:
        payload = torch.load(
            PROGRESS,
            map_location=device,
            weights_only=False,
        )

        if (
            payload.get(
                "fingerprints"
            )
            !=
            fingerprints
        ):
            raise RuntimeError(
                "Progress fingerprint "
                "does not match current "
                "frozen artifacts."
            )

        model.load_state_dict(
            payload[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            payload[
                "optimizer_state_dict"
            ]
        )

        return payload

    except Exception as exc:
        destination = (
            PROGRESS.with_name(
                PROGRESS.name
                +
                f".corrupt."
                f"{int(time.time())}"
            )
        )

        os.replace(
            PROGRESS,
            destination,
        )

        print()
        print(
            "WARNING: unusable progress "
            "checkpoint archived:"
        )
        print(
            destination
        )
        print(
            "reason =",
            type(exc).__name__,
            str(exc),
        )
        print(
            "training will restart from "
            "the frozen deterministic "
            "initialization; caches are reused."
        )

        return None


# ============================================================
# Upstream validation
# ============================================================

def validate_upstream():
    required = (
        PART1,
        PREFLIGHT,
        BLOCK43,
        CONFIG,
        DET_CHECKPOINT,
        FIT_CACHE,
        DEV_CACHE,
        NORMALIZATION,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            "Missing required frozen "
            "artifacts: "
            +
            ", ".join(missing)
        )

    part1 = json.loads(
        PART1.read_text(
            encoding="utf-8"
        )
    )

    preflight = json.loads(
        PREFLIGHT.read_text(
            encoding="utf-8"
        )
    )

    block43 = json.loads(
        BLOCK43.read_text(
            encoding="utf-8"
        )
    )

    configuration = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    if part1.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Block4.4 Part1 is "
            "not PASS."
        )

    if preflight.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Block4.4 scientific "
            "preflight is not PASS."
        )

    if block43.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Block4.3 is not PASS."
        )

    if (
        block43[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK43_IMPL_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.3 "
            "implementation fingerprint "
            "changed."
        )

    if (
        file_sha256(
            DET_CHECKPOINT
        )
        !=
        EXPECTED_DET_CHECKPOINT_SHA
    ):
        raise RuntimeError(
            "Frozen deterministic "
            "checkpoint SHA changed."
        )

    if (
        file_sha256(
            FIT_CACHE
        )
        !=
        block43[
            "data"
        ][
            "fit_cache"
        ][
            "file_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen fit cache "
            "SHA changed."
        )

    if (
        file_sha256(
            DEV_CACHE
        )
        !=
        block43[
            "data"
        ][
            "development_cache"
        ][
            "file_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen development "
            "cache SHA changed."
        )

    if (
        file_sha256(
            NORMALIZATION
        )
        !=
        block43[
            "normalization"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen fit-only "
            "normalization changed."
        )

    if (
        normalization.get(
            "source"
        )
        !=
        "fit_only"
    ):
        raise RuntimeError(
            "Normalization is "
            "not fit-only."
        )

    if (
        configuration[
            "input"
        ][
            "measurement_covariance_R_t_input"
        ]
        is not True
    ):
        raise RuntimeError(
            "Measurement covariance "
            "input contract changed."
        )

    if (
        configuration[
            "predictive_distribution"
        ][
            "predictive_covariance_is_distinct_from_measurement_covariance"
        ]
        is not True
    ):
        raise RuntimeError(
            "Measurement/predictive "
            "uncertainty distinction "
            "changed."
        )

    return (
        block43,
        configuration,
        normalization,
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.4 "
        "GAUSSIAN PROBABILISTIC TRAINING"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Fast path for an already frozen successful rerun.
    # --------------------------------------------------------

    if (
        REPORT.is_file()
        and
        CHECKPOINT.is_file()
    ):
        previous = json.loads(
            REPORT.read_text(
                encoding="utf-8"
            )
        )

        if (
            previous.get(
                "status"
            )
            ==
            "PASS"
        ):
            expected = (
                previous[
                    "checkpoint"
                ][
                    "file_sha256"
                ]
            )

            if (
                file_sha256(
                    CHECKPOINT
                )
                ==
                expected
            ):
                print(
                    "Block4.4 already "
                    "COMPLETE/FROZEN."
                )
                print(
                    "checkpoint SHA256 =",
                    expected,
                )
                print(
                    "STATUS = PASS"
                )
                return

    (
        block43,
        configuration,
        normalization,
    ) = validate_upstream()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    if (
        torch.version.cuda
        !=
        "13.2"
    ):
        raise RuntimeError(
            "Frozen CUDA runtime "
            "changed."
        )

    device = torch.device(
        "cuda:0"
    )

    if (
        "A6000"
        not in
        torch.cuda.get_device_name(
            0
        )
    ):
        raise RuntimeError(
            "Expected frozen "
            "RTX A6000."
        )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            "250-GiB storage "
            "reserve violated."
        )

    seed = int(
        configuration[
            "training"
        ][
            "seed"
        ]
    )

    batch_size = int(
        configuration[
            "training"
        ][
            "batch_size"
        ]
    )

    max_epochs = int(
        configuration[
            "training"
        ][
            "max_epochs"
        ]
    )

    patience_limit = int(
        configuration[
            "training"
        ][
            "early_stopping_patience"
        ]
    )

    learning_rate = float(
        configuration[
            "training"
        ][
            "learning_rate"
        ]
    )

    weight_decay = float(
        configuration[
            "training"
        ][
            "weight_decay"
        ]
    )

    gradient_clip = float(
        configuration[
            "training"
        ][
            "gradient_clip_norm"
        ]
    )

    set_global_determinism(
        seed
    )

    deterministic_checkpoint = (
        torch.load(
            DET_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
    )

    if (
        deterministic_checkpoint[
            "state_dict_sha256"
        ]
        !=
        EXPECTED_DET_STATE_SHA
    ):
        raise RuntimeError(
            "Frozen deterministic "
            "state-dict declaration "
            "changed."
        )

    if (
        state_dict_sha256(
            deterministic_checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED_DET_STATE_SHA
    ):
        raise RuntimeError(
            "Frozen deterministic "
            "state-dict content changed."
        )

    fit_arrays = load_cache(
        FIT_CACHE
    )

    dev_arrays = load_cache(
        DEV_CACHE
    )

    print(
        "Block4.4 Part1          = PASS"
    )
    print(
        "Block4.3 frozen         = PASS"
    )
    print(
        "fit samples             =",
        fit_arrays[
            "target"
        ].shape[0],
    )
    print(
        "development samples     =",
        dev_arrays[
            "target"
        ].shape[0],
    )
    print(
        "cache regeneration      = NO"
    )
    print(
        "normalization           = FIT ONLY"
    )
    print(
        "GPU                     =",
        torch.cuda.get_device_name(0),
    )

    normalizer = (
        GaussianNormalizer(
            normalization,
            device=device,
        )
    )

    fingerprints = {
        "configuration_sha256":
            file_sha256(CONFIG),

        "deterministic_checkpoint_sha256":
            EXPECTED_DET_CHECKPOINT_SHA,

        "deterministic_state_dict_sha256":
            EXPECTED_DET_STATE_SHA,

        "fit_cache_sha256":
            file_sha256(
                FIT_CACHE
            ),

        "development_cache_sha256":
            file_sha256(
                DEV_CACHE
            ),

        "normalization_sha256":
            file_sha256(
                NORMALIZATION
            ),
    }

    model = build_gaussian_model(
        deterministic_checkpoint[
            "configuration"
        ],
        device=device,
    )

    initialize_from_deterministic(
        model,
        deterministic_checkpoint[
            "state_dict"
        ],
        initial_std_normalized=float(
            configuration[
                "predictive_distribution"
            ][
                "initial_predictive_std_normalized"
            ]
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

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

    class_mass = {}

    for class_id in (
        CLASS_VEHICLE,
        CLASS_PEDESTRIAN,
        CLASS_CYCLIST,
    ):
        name = (
            CLASS_ID_TO_NAME[
                class_id
            ]
        )

        class_mass[
            name
        ] = float(
            class_weights[
                class_ids_cpu
                ==
                class_id
            ].sum().item()
        )

    if (
        max(
            class_mass.values()
        )
        -
        min(
            class_mass.values()
        )
        >
        1e-12
    ):
        raise RuntimeError(
            "Primary class sampling "
            "mass is not balanced."
        )

    # --------------------------------------------------------
    # Resume completed epochs if available.
    # --------------------------------------------------------

    progress = load_valid_progress(
        model,
        optimizer,
        device=device,
        fingerprints=fingerprints,
    )

    if progress is None:
        print()
        print(
            "===== INITIAL GAUSSIAN EVALUATION ====="
        )

        initial_metrics = (
            evaluate_gaussian(
                model,
                dev_arrays,
                normalizer,
                device=device,
            )
        )

        best_nll = float(
            initial_metrics[
                "metric_Gaussian_NLL"
            ]
        )

        best_ade = float(
            initial_metrics[
                "planar_ADE_m"
            ]
        )

        best_epoch = 0
        patience = 0
        history = []

        save_best_checkpoint(
            model,
            epoch=0,
            metrics=initial_metrics,
            fingerprints=fingerprints,
            configuration=configuration,
        )

        save_progress(
            model,
            optimizer,
            last_epoch=0,
            best_epoch=0,
            best_nll=best_nll,
            best_ade=best_ade,
            patience=0,
            history=history,
            initial_metrics=(
                initial_metrics
            ),
            fingerprints=fingerprints,
        )

        start_epoch = 1

        print(
            "initialized dev NLL      =",
            best_nll,
        )
        print(
            "initialized dev ADE m    =",
            best_ade,
        )

    else:
        initial_metrics = (
            progress[
                "initial_metrics"
            ]
        )

        best_nll = float(
            progress[
                "best_nll"
            ]
        )

        best_ade = float(
            progress[
                "best_ade"
            ]
        )

        best_epoch = int(
            progress[
                "best_epoch"
            ]
        )

        patience = int(
            progress[
                "patience"
            ]
        )

        history = list(
            progress[
                "history"
            ]
        )

        start_epoch = (
            int(
                progress[
                    "last_completed_epoch"
                ]
            )
            +
            1
        )

        print()
        print(
            "===== RESUMING GAUSSIAN TRAINING ====="
        )
        print(
            "last completed epoch     =",
            start_epoch - 1,
        )
        print(
            "best epoch so far        =",
            best_epoch,
        )
        print(
            "best dev NLL so far      =",
            best_nll,
        )
        print(
            "cached Stage2/3 rebuilt  = NO"
        )

    fit_n = int(
        fit_arrays[
            "target"
        ].shape[0]
    )

    training_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # NLL training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        max_epochs + 1,
    ):
        if patience >= (
            patience_limit
        ):
            print(
                "early stopping was "
                "already satisfied "
                "in saved progress."
            )
            break

        model.train()

        generator = (
            torch.Generator(
                device="cpu"
            )
        )

        generator.manual_seed(
            seed + epoch
        )

        sampled = (
            torch.multinomial(
                class_weights,
                num_samples=fit_n,
                replacement=True,
                generator=generator,
            )
            .numpy()
        )

        epoch_nll_sum = 0.0
        epoch_valid = 0

        for start in range(
            0,
            fit_n,
            batch_size,
        ):
            stop = min(
                fit_n,
                start + batch_size,
            )

            indices = sampled[
                start:stop
            ]

            batch = (
                normalizer.prepare(
                    fit_arrays,
                    indices,
                    device=device,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                batch["target"],
                batch["neighbors"],
                batch[
                    "neighbor_mask"
                ],
                batch[
                    "map_context"
                ],
            )

            loss = masked_gaussian_nll(
                output.mean,
                output.scale_tril,
                batch[
                    "label_normalized"
                ],
                batch[
                    "future_mask"
                ],
            )

            if not bool(
                torch.isfinite(
                    loss
                )
            ):
                raise RuntimeError(
                    "Non-finite Gaussian "
                    "training NLL."
                )

            loss.backward()

            gradients = [
                parameter.grad
                for parameter
                in model.parameters()
                if parameter.grad
                is not None
            ]

            if not gradients:
                raise RuntimeError(
                    "Gaussian backward "
                    "produced no gradients."
                )

            if not all(
                bool(
                    torch.isfinite(
                        gradient
                    ).all()
                )
                for gradient
                in gradients
            ):
                raise RuntimeError(
                    "Gaussian backward "
                    "produced non-finite "
                    "gradients."
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    gradient_clip
                ),
            )

            optimizer.step()

            valid_count = int(
                batch[
                    "future_mask"
                ].sum().item()
            )

            epoch_nll_sum += (
                float(
                    loss
                    .detach()
                    .item()
                )
                *
                valid_count
            )

            epoch_valid += (
                valid_count
            )

        torch.cuda.synchronize()

        if epoch_valid <= 0:
            raise RuntimeError(
                "Training epoch has "
                "zero valid labels."
            )

        training_nll = (
            epoch_nll_sum
            /
            epoch_valid
        )

        development = (
            evaluate_gaussian(
                model,
                dev_arrays,
                normalizer,
                device=device,
            )
        )

        dev_nll = float(
            development[
                "metric_Gaussian_NLL"
            ]
        )

        dev_ade = float(
            development[
                "planar_ADE_m"
            ]
        )

        # Primary = development metric NLL.
        # ADE is only the frozen tiebreaker.
        nll_improved = (
            dev_nll
            <
            best_nll
            -
            1e-4
        )

        nll_tie = (
            abs(
                dev_nll
                -
                best_nll
            )
            <=
            1e-4
        )

        ade_tiebreak = (
            nll_tie
            and
            dev_ade
            <
            best_ade
            -
            1e-6
        )

        improved = (
            nll_improved
            or
            ade_tiebreak
        )

        if improved:
            best_nll = (
                dev_nll
            )

            best_ade = (
                dev_ade
            )

            best_epoch = (
                epoch
            )

            patience = 0

            save_best_checkpoint(
                model,
                epoch=epoch,
                metrics=(
                    development
                ),
                fingerprints=(
                    fingerprints
                ),
                configuration=(
                    configuration
                ),
            )

        else:
            patience += 1

        history.append({
            "epoch":
                int(epoch),

            "training_normalized_Gaussian_NLL":
                float(
                    training_nll
                ),

            "development_metric_Gaussian_NLL":
                dev_nll,

            "development_planar_ADE_m":
                dev_ade,

            "development_one_second_planar_error_m":
                float(
                    development[
                        "one_second_planar_error_m"
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

        save_progress(
            model,
            optimizer,
            last_epoch=epoch,
            best_epoch=best_epoch,
            best_nll=best_nll,
            best_ade=best_ade,
            patience=patience,
            history=history,
            initial_metrics=(
                initial_metrics
            ),
            fingerprints=(
                fingerprints
            ),
        )

        print(
            f"epoch {epoch:02d} "
            f"| trainNLL="
            f"{training_nll:.6f} "
            f"| devNLL="
            f"{dev_nll:.6f} "
            f"| devADE="
            f"{dev_ade:.6f} m "
            f"| bestNLL="
            f"{best_nll:.6f} "
            f"| bestEpoch="
            f"{best_epoch} "
            f"| patience="
            f"{patience}/"
            f"{patience_limit}",
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

    # --------------------------------------------------------
    # Freeze/evaluate best Gaussian checkpoint.
    # --------------------------------------------------------

    if not CHECKPOINT.is_file():
        raise RuntimeError(
            "Gaussian best checkpoint "
            "was not created."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint[
            "fingerprints"
        ]
        !=
        fingerprints
    ):
        raise RuntimeError(
            "Gaussian checkpoint "
            "fingerprints mismatch."
        )

    best_model = build_gaussian_model(
        deterministic_checkpoint[
            "configuration"
        ],
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
            "Gaussian checkpoint "
            "state-dict SHA mismatch."
        )

    final_development = (
        evaluate_gaussian(
            best_model,
            dev_arrays,
            normalizer,
            device=device,
        )
    )

    initial_nll = float(
        initial_metrics[
            "metric_Gaussian_NLL"
        ]
    )

    final_nll = float(
        final_development[
            "metric_Gaussian_NLL"
        ]
    )

    if not (
        final_nll
        <
        initial_nll
        -
        1e-6
    ):
        raise RuntimeError(
            "Gaussian training did "
            "not improve development "
            "metric NLL over the "
            "initialized Gaussian."
        )

    if (
        final_development[
            "predictive_covariance_SPD_failures"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Final predictive "
            "covariance is not SPD."
        )

    for required_class in (
        "TYPE_VEHICLE",
        "TYPE_PEDESTRIAN",
        "TYPE_CYCLIST",
    ):
        if required_class not in (
            final_development[
                "class_planar_ADE_m"
            ]
        ):
            raise RuntimeError(
                "Final development "
                f"metrics lack "
                f"{required_class}."
            )

    # --------------------------------------------------------
    # Exact probabilistic inference repeat.
    # --------------------------------------------------------

    inference_sha_a = (
        probabilistic_prediction_sha256(
            best_model,
            dev_arrays,
            normalizer,
            device=device,
        )
    )

    repeat_checkpoint = (
        torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
    )

    repeat_model = (
        build_gaussian_model(
            deterministic_checkpoint[
                "configuration"
            ],
            device=device,
        )
    )

    repeat_model.load_state_dict(
        repeat_checkpoint[
            "state_dict"
        ]
    )

    repeat_model.eval()

    inference_sha_b = (
        probabilistic_prediction_sha256(
            repeat_model,
            dev_arrays,
            normalizer,
            device=device,
        )
    )

    if (
        inference_sha_a
        !=
        inference_sha_b
    ):
        raise RuntimeError(
            "Frozen Gaussian checkpoint "
            "inference is not exactly "
            "repeatable."
        )

    print()
    print(
        "exact probabilistic "
        "inference repeat = PASS"
    )
    print(
        "probabilistic prediction "
        "SHA256 =",
        inference_sha_a,
    )

    # --------------------------------------------------------
    # Full 60-test regression.
    # --------------------------------------------------------

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
        cwd=str(STAGE4),
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
        60
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected final "
            "Stage4 regression "
            "60/60."
        )

    checkpoint_sha = (
        file_sha256(
            CHECKPOINT
        )
    )

    (
        implementation_files,
        implementation_sha,
    ) = implementation_fingerprint()

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
            "250-GiB reserve "
            "violated after 4.4."
        )

    raw_coverage = (
        final_development[
            "raw_empirical_coverage"
        ]
    )

    report = {
        "stage": 4,
        "block": "4.4",
        "status": "PASS",

        "scope":
            "single_full_covariance_Gaussian_probabilistic_GRU",

        "upstream": {
            "block43":
                "FROZEN",

            "cache_regenerated":
                False,

            "fit_normalization":
                "FROZEN_FIT_ONLY",

            "deterministic_checkpoint_sha256":
                EXPECTED_DET_CHECKPOINT_SHA,
        },

        "uncertainty_semantics": {
            "measurement_uncertainty":
                "Stage2_R_t_model_input",

            "predictive_uncertainty":
                "learned_Gaussian_output_covariance",

            "separate":
                True,
        },

        "training": {
            "seed":
                seed,

            "device":
                str(device),

            "gpu":
                torch.cuda
                .get_device_name(0),

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

            "class_balanced_sampler":
                True,

            "primary_class_sampling_mass":
                class_mass,

            "objective":
                "full_3D_Gaussian_NLL",

            "model_selection":
                "development_metric_Gaussian_NLL",

            "history":
                history,

            "runtime_this_invocation_s":
                training_runtime_s,
        },

        "development": {
            "initialized_Gaussian":
                initial_metrics,

            "best_raw_uncalibrated":
                final_development,

            "NLL_improved_over_initialization":
                True,

            "calibration_applied":
                False,
        },

        "raw_coverage": {
            "confidence_levels":
                [
                    0.50,
                    0.80,
                    0.90,
                    0.95,
                    0.99,
                ],

            "space":
                "3D_predictive_confidence_ellipsoid",

            "distribution":
                "chi_square_df_3",

            "results":
                raw_coverage,

            "used_for_training":
                False,

            "used_for_model_selection":
                False,
        },

        "checkpoint": {
            "path":
                str(CHECKPOINT),

            "file_sha256":
                checkpoint_sha,

            "state_dict_sha256":
                actual_state_sha,

            "best_epoch":
                int(
                    checkpoint[
                        "best_epoch"
                    ]
                ),

            "exact_inference_repeat":
                True,

            "prediction_sha256":
                inference_sha_a,
        },

        "formal_validation": {
            "used":
                False,

            "comparison_with_Stage3":
                "DEFERRED_TO_BLOCK_4.8",
        },

        "calibration": {
            "performed":
                False,

            "reserved_for":
                "Block4.5",
        },

        "regression": {
            "tests_passed":
                60,

            "tests_total":
                60,
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
            "calibrated predictive uncertainty",
            "post-calibration coverage",
            "ECE",
            "coverage-event Brier score",
            "GMM multimodality",
            "formal validation superiority over classical baseline"
        ],
    }

    write_json(
        REPORT,
        report,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    # --------------------------------------------------------
    # Closure log
    # --------------------------------------------------------

    marker = (
        "## Block 4.4 — "
        "Gaussian probabilistic GRU"
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
                "- The frozen Block4.3 "
                  "fit/development caches and "
                  "fit-only normalization are "
                  "reused without regeneration.\n"
                "- The probabilistic predictor "
                  "uses the same causal target, "
                  "measurement-covariance, "
                  "multi-agent and map inputs as "
                  "the deterministic GRU.\n"
                "- Stage2 R_t remains measurement "
                  "uncertainty input; predictive "
                  "uncertainty is a separate "
                  "learned Gaussian output.\n"
                "- The learned distribution is a "
                  "single full 3-D Gaussian at "
                  "0.1/0.3/0.5/1.0 s.\n"
                "- Predictive covariance is "
                  "parameterized by a six-parameter "
                  "Cholesky factor per horizon with "
                  "strictly positive diagonal.\n"
                "- The Gaussian backbone and mean "
                  "head initialize exactly from the "
                  "frozen deterministic Block4.3 "
                  "checkpoint.\n"
                "- Training uses full Gaussian NLL "
                  "and the frozen class-balanced "
                  "fit sampler.\n"
                "- Model selection uses development "
                  "metric Gaussian NLL only, with "
                  "development ADE as tiebreaker.\n"
                "- Raw 50/80/90/95/99 empirical "
                  "coverage is measured but is not "
                  "used for training/model selection.\n"
                "- No calibration is applied in "
                  "Block4.4; calibration is reserved "
                  "for Block4.5.\n"
                "- Formal N=120 validation remains "
                  "untouched.\n"
                f"- Initial development Gaussian "
                  f"NLL: {initial_nll:.9f}.\n"
                f"- Best development Gaussian "
                  f"NLL: {final_nll:.9f}.\n"
                f"- Best development planar ADE: "
                f"{final_development['planar_ADE_m']:.9f} m.\n"
                f"- Best epoch: "
                f"{checkpoint['best_epoch']}.\n"
                f"- Checkpoint SHA256: "
                f"{checkpoint_sha}.\n"
                f"- State-dict SHA256: "
                f"{actual_state_sha}.\n"
                f"- Probabilistic inference SHA256: "
                f"{inference_sha_a}.\n"
                "- Exact checkpoint inference repeat: "
                  "PASS.\n"
                "- Full Stage4 regression: "
                  "60/60 PASS.\n"
                f"- Implementation files: "
                f"{implementation_files}.\n"
                f"- Implementation SHA256: "
                f"{implementation_sha}.\n"
            )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.4 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "fit/dev cache reuse        = PASS"
    )
    print(
        "fit-only normalization     = PASS"
    )
    print(
        "measurement R_t            = INPUT"
    )
    print(
        "predictive covariance      = LEARNED OUTPUT"
    )
    print(
        "uncertainty separation     = PASS"
    )
    print(
        "Gaussian family            = FULL 3D"
    )
    print(
        "covariance SPD failures    =",
        final_development[
            "predictive_covariance_SPD_failures"
        ],
    )
    print(
        "initialized dev NLL        =",
        round(
            initial_nll,
            6,
        ),
    )
    print(
        "best raw dev NLL           =",
        round(
            final_nll,
            6,
        ),
    )
    print(
        "NLL improved               = PASS"
    )
    print(
        "best raw dev ADE m         =",
        round(
            final_development[
                "planar_ADE_m"
            ],
            6,
        ),
    )
    print(
        "best raw dev 1.0s error m  =",
        round(
            final_development[
                "one_second_planar_error_m"
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
        final_development[
            "class_planar_ADE_m"
        ],
    )

    print()
    print(
        "RAW COVERAGE BEFORE CALIBRATION"
    )

    for level in CONFIDENCE_LEVELS:
        key = f"{level:.2f}"

        print(
            f"coverage {int(level*100):02d}%"
            f"                  = "
            f"{raw_coverage[key]['empirical']:.6f}"
        )

    print()
    print(
        "checkpoint SHA256          =",
        checkpoint_sha,
    )
    print(
        "state_dict SHA256          =",
        actual_state_sha,
    )
    print(
        "probabilistic inference SHA=",
        inference_sha_a,
    )
    print(
        "exact inference repeat     = PASS"
    )
    print(
        "calibration applied        = NO"
    )
    print(
        "formal validation used     = NO"
    )
    print(
        "full Stage4 regression     = 60 / 60 PASS"
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

    print()
    print(
        "===== BLOCK 4.4 FINAL ====="
    )
    print(
        "Gaussian probabilistic GRU = PASS"
    )
    print(
        "full SPD covariance        = PASS"
    )
    print(
        "Gaussian NLL training      = PASS"
    )
    print(
        "measurement/predictive UQ  = SEPARATE"
    )
    print(
        "raw coverage measured      = PASS"
    )
    print(
        "calibration                = NOT YET"
    )
    print(
        "future leakage             = NONE"
    )
    print(
        "checkpoint freeze          = PASS"
    )
    print(
        "exact inference repeat     = PASS"
    )
    print(
        "regression                 = 60 / 60"
    )
    print(
        "implementation SHA         =",
        implementation_sha,
    )
    print(
        "log closure                = PASS"
    )


def recovery_hint(
    exc,
):
    text = (
        f"{type(exc).__name__}: "
        f"{exc}"
    ).lower()

    if (
        "out of memory"
        in text
        or
        "cuda oom"
        in text
    ):
        return (
            "Check nvidia-smi for another "
            "GPU process. Do not change the "
            "frozen batch/model configuration "
            "before verifying external GPU use."
        )

    if (
        "non-finite"
        in text
        or
        "nan"
        in text
        or
        "inf"
        in text
    ):
        return (
            "The runner has preserved the "
            "last completed epoch progress. "
            "Do not regenerate Stage2/3 caches. "
            "Inspect the last training epoch and "
            "Gaussian covariance/NLL state before "
            "changing optimization parameters."
        )

    if (
        "progress"
        in text
        or
        "checkpoint"
        in text
        or
        "fingerprint"
        in text
    ):
        return (
            "Do not modify frozen Block4.3. "
            "The generated Block4.4 epoch-progress "
            "file is separately resumable and any "
            "invalid progress file is automatically "
            "archived before a clean Gaussian "
            "restart."
        )

    if (
        "triton"
        in text
        or
        "python.h"
        in text
    ):
        return (
            "The Python 3.13 development-header "
            "repair from Block4.4 Part1 must remain "
            "available. Recheck Python.h and the "
            "Triton native-driver smoke; do not "
            "reinstall PyTorch."
        )

    if (
        "60/60"
        in text
        or
        "regression"
        in text
    ):
        return (
            "Inspect the failing unittest output. "
            "Do not alter frozen Block4.0–4.3 "
            "scientific behavior merely to satisfy "
            "a new Block4.4 test."
        )

    return (
        "Inspect reports/"
        "block44_training_failure.json. "
        "Completed Block4.4 epochs are resumable; "
        "the expensive Stage2/Stage3 caches are "
        "never regenerated by this runner."
    )


try:
    main()

except BaseException as exc:
    FAILURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "stage": 4,
        "block": "4.4_part_2",
        "status": "BLOCKED",

        "exception_type":
            type(exc).__name__,

        "exception":
            str(exc),

        "traceback":
            traceback.format_exc(),

        "recovery_hint":
            recovery_hint(exc),

        "block43_modified":
            False,

        "stage2_stage3_cache_regenerated":
            False,

        "epoch_progress_resumable":
            True,

        "calibration_started":
            False,

        "formal_validation_used":
            False,
    }

    write_json(
        FAILURE,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.4 PART 2/2 = BLOCKED"
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
        "RECOVERY:"
    )
    print(
        payload[
            "recovery_hint"
        ]
    )
    print()
    print(
        "Block4.3 modified       = NO"
    )
    print(
        "Stage2/3 cache rebuilt  = NO"
    )
    print(
        "epoch progress resumable= YES"
    )
    print(
        "calibration started     = NO"
    )
    print(
        "formal validation used  = NO"
    )
    print(
        "terminal remains open   = YES"
    )
    print(
        "failure report =",
        FAILURE,
    )

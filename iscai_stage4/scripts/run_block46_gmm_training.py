from __future__ import annotations

from hashlib import sha256
import json
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

from iscai_stage4.ml.gmm_gru import (
    GMMTrajectoryGRU,
    initialize_gmm_from_gaussian,
)

from iscai_stage4.ml.gmm_math import (
    masked_gmm_joint_nll,
)

from iscai_stage4.ml.gmm_runtime import (
    GMMNormalizer,
    evaluate_gmm,
    prediction_sha256,
)

from iscai_stage4.ml.training_utils import (
    CLASS_CYCLIST,
    CLASS_ID_TO_NAME,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
    balanced_class_weights,
    set_global_determinism,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

PART1 = (
    STAGE4
    / "reports/block46_part1_preflight.json"
)

BLOCK45 = (
    STAGE4
    / "reports/block45_calibration.json"
)

BLOCK44 = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

GMM_CONFIG = (
    STAGE4
    / "configs/stage4_gmm_gru.json"
)

DETERMINISTIC_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/deterministic_gru.pt"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/gaussian_gru.pt"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/covariance_scaler.json"
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
    / "artifacts/block46/gmm_gru.pt"
)

PROGRESS = (
    STAGE4
    / "artifacts/block46/training_progress.pt"
)

REPORT = (
    STAGE4
    / "reports/block46_gmm_gru.json"
)

FAILURE = (
    STAGE4
    / "reports/block46_training_failure.json"
)

LOG = (
    STAGE4
    / "docs/implementation_log.md"
)

EXPECTED_BLOCK45_IMPL_SHA = (
    "482fa179b958f05526657d9970346b3ca"
    "699c649b77783f63d4c8de2fd7be7cf"
)

EXPECTED_DETERMINISTIC_CHECKPOINT_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_DETERMINISTIC_STATE_SHA = (
    "d2ffbc03c7cb2826fef2175c95f48725"
    "707ec6d791eaeffbd6f59bc507b8595a"
)

EXPECTED_GAUSSIAN_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_GAUSSIAN_STATE_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
)

EXPECTED_CALIBRATOR_SHA = (
    "508ff2e3fbcfafe8e001155340c25baaf"
    "3772fe2561a8022a9ed1cf780e66087"
)

MIN_FREE_GIB = 250.0

CACHE_KEYS = (
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
):
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

            if "__pycache__" in path.parts:
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


def atomic_torch_save(
    payload,
    path,
):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
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
    path,
    payload,
):
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


def load_cache(
    path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        arrays = {
            key:
                np.asarray(
                    data[
                        key
                    ]
                )
            for key in CACHE_KEYS
        }

    n = int(
        arrays[
            "target"
        ].shape[0]
    )

    expected = {
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

    for key, shape in (
        expected.items()
    ):
        if tuple(
            arrays[
                key
            ].shape
        ) != shape:
            raise RuntimeError(
                f"{path.name}:{key} "
                f"shape changed."
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
            arrays[
                key
            ]
        ).all():
            raise RuntimeError(
                f"Non-finite cache values: "
                f"{path.name}:{key}"
            )

    return arrays


def build_gmm(
    architecture_configuration,
    *,
    device,
):
    section = (
        architecture_configuration[
            "model"
        ]
    )

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
    ).to(device)


def save_best(
    model,
    *,
    epoch,
    metrics,
    architecture_configuration,
    experiment_configuration,
    fingerprints,
):
    state_sha = state_dict_sha256(
        model.state_dict()
    )

    atomic_torch_save(
        {
            "stage":
                4,

            "block":
                "4.6",

            "model_type":
                "GMMTrajectoryGRU",

            # Keep architecture and experiment
            # metadata separate by construction.
            "architecture_configuration":
                architecture_configuration,

            "experiment_configuration":
                experiment_configuration,

            "state_dict":
                model.state_dict(),

            "state_dict_sha256":
                state_sha,

            "best_epoch":
                int(
                    epoch
                ),

            "development_metrics":
                metrics,

            "fingerprints":
                fingerprints,

            "calibrated":
                False,

            "downstream_selected":
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


def load_progress(
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
                "Saved progress fingerprints "
                "do not match frozen inputs."
            )

        model.load_state_dict(
            payload[
                "model_state_dict"
            ],
            strict=True,
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
                f".corrupt.{int(time.time())}"
            )
        )

        os.replace(
            PROGRESS,
            destination,
        )

        print()
        print(
            "WARNING: unusable GMM progress "
            "archived:"
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
            "GMM training will restart from "
            "the frozen Gaussian initialization."
        )

        return None


def inference_sha256(
    model,
    arrays,
    normalizer,
    *,
    device,
    limit=2048,
):
    digest = sha256()

    count = min(
        int(
            arrays[
                "target"
            ].shape[0]
        ),
        int(
            limit
        ),
    )

    model.eval()

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
                batch[
                    "map_context"
                ],
            )

            for tensor in (
                output.mixture_logits,
                output.means,
                output.scale_tril,
            ):
                value = (
                    tensor.detach()
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


def validate_upstream():
    required = (
        PART1,
        BLOCK45,
        BLOCK44,
        GMM_CONFIG,
        DETERMINISTIC_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
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
            "Missing frozen artifact(s): "
            +
            ", ".join(
                missing
            )
        )

    part1 = json.loads(
        PART1.read_text(
            encoding="utf-8"
        )
    )

    block45 = json.loads(
        BLOCK45.read_text(
            encoding="utf-8"
        )
    )

    block44 = json.loads(
        BLOCK44.read_text(
            encoding="utf-8"
        )
    )

    config = json.loads(
        GMM_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    if (
        part1.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.6 Part1 is not PASS."
        )

    if (
        block45.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.5 is not PASS."
        )

    if (
        block45[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK45_IMPL_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.5 "
            "implementation SHA changed."
        )

    if (
        block44.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.4 is not PASS."
        )

    if (
        file_sha256(
            DETERMINISTIC_CHECKPOINT
        )
        !=
        EXPECTED_DETERMINISTIC_CHECKPOINT_SHA
    ):
        raise RuntimeError(
            "Frozen deterministic "
            "checkpoint SHA changed."
        )

    if (
        file_sha256(
            GAUSSIAN_CHECKPOINT
        )
        !=
        EXPECTED_GAUSSIAN_CHECKPOINT_SHA
    ):
        raise RuntimeError(
            "Frozen Gaussian "
            "checkpoint SHA changed."
        )

    if (
        file_sha256(
            CALIBRATOR
        )
        !=
        EXPECTED_CALIBRATOR_SHA
    ):
        raise RuntimeError(
            "Frozen calibrated-Gaussian "
            "artifact SHA changed."
        )

    if (
        file_sha256(
            FIT_CACHE
        )
        !=
        block44[
            "upstream"
        ][
            "fit_cache_sha256"
        ]
    ) if (
        isinstance(
            block44.get(
                "upstream"
            ),
            dict,
        )
        and
        "fit_cache_sha256"
        in
        block44[
            "upstream"
        ]
    ) else False:
        raise RuntimeError(
            "Frozen fit cache SHA changed."
        )

    if (
        normalization.get(
            "source"
        )
        !=
        "fit_only"
    ):
        raise RuntimeError(
            "Normalization source "
            "is not fit-only."
        )

    return (
        config,
        normalization,
        block44,
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.6 GMM TRAINING"
    )
    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Already-frozen fast path.
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
                    "Block4.6 already "
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
        experiment_config,
        normalization,
        block44,
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
            "Frozen CUDA runtime changed."
        )

    if (
        "A6000"
        not in
        torch.cuda.get_device_name(0)
    ):
        raise RuntimeError(
            "Expected frozen RTX A6000."
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
            "250-GiB reserve violated."
        )

    device = torch.device(
        "cuda:0"
    )

    seed = int(
        experiment_config[
            "training"
        ][
            "seed"
        ]
    )

    batch_size = int(
        experiment_config[
            "training"
        ][
            "batch_size"
        ]
    )

    max_epochs = int(
        experiment_config[
            "training"
        ][
            "max_epochs"
        ]
    )

    patience_limit = int(
        experiment_config[
            "training"
        ][
            "early_stopping_patience"
        ]
    )

    learning_rate = float(
        experiment_config[
            "training"
        ][
            "learning_rate"
        ]
    )

    weight_decay = float(
        experiment_config[
            "training"
        ][
            "weight_decay"
        ]
    )

    gradient_clip = float(
        experiment_config[
            "training"
        ][
            "gradient_clip_norm"
        ]
    )

    set_global_determinism(
        seed
    )

    deterministic_checkpoint = torch.load(
        DETERMINISTIC_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    gaussian_checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
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
        EXPECTED_DETERMINISTIC_STATE_SHA
    ):
        raise RuntimeError(
            "Frozen deterministic "
            "state SHA changed."
        )

    if (
        state_dict_sha256(
            gaussian_checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED_GAUSSIAN_STATE_SHA
    ):
        raise RuntimeError(
            "Frozen Gaussian "
            "state SHA changed."
        )

    architecture_configuration = (
        deterministic_checkpoint[
            "configuration"
        ]
    )

    if (
        "model"
        not in
        architecture_configuration
    ):
        raise RuntimeError(
            "Frozen architecture configuration "
            "lacks model section."
        )

    fit_arrays = load_cache(
        FIT_CACHE
    )

    dev_arrays = load_cache(
        DEV_CACHE
    )

    print(
        "Block4.6 Part1          = PASS"
    )
    print(
        "Block4.5 frozen         = PASS"
    )
    print(
        "Gaussian baseline       = FROZEN"
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
        "Stage2/3 rescan         = NO"
    )
    print(
        "normalization           = FIT ONLY"
    )
    print(
        "GPU                     =",
        torch.cuda.get_device_name(0),
    )

    normalizer = GMMNormalizer(
        normalization,
        device=device,
    )

    fingerprints = {
        "experiment_config_sha256":
            file_sha256(
                GMM_CONFIG
            ),

        "Part1_report_sha256":
            file_sha256(
                PART1
            ),

        "deterministic_checkpoint_sha256":
            EXPECTED_DETERMINISTIC_CHECKPOINT_SHA,

        "Gaussian_checkpoint_sha256":
            EXPECTED_GAUSSIAN_CHECKPOINT_SHA,

        "Gaussian_state_sha256":
            EXPECTED_GAUSSIAN_STATE_SHA,

        "calibrated_Gaussian_sha256":
            EXPECTED_CALIBRATOR_SHA,

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

    model = build_gmm(
        architecture_configuration,
        device=device,
    )

    initialize_gmm_from_gaussian(
        model,
        gaussian_checkpoint[
            "state_dict"
        ],
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
        name = CLASS_ID_TO_NAME[
            class_id
        ]

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

    progress = load_progress(
        model,
        optimizer,
        device=device,
        fingerprints=fingerprints,
    )

    if progress is None:
        print()
        print(
            "===== INITIAL GMM EVALUATION ====="
        )

        initial_metrics = evaluate_gmm(
            model,
            dev_arrays,
            normalizer,
            device=device,
            class_id_to_name=(
                CLASS_ID_TO_NAME
            ),
        )

        best_nll = float(
            initial_metrics[
                "joint_GMM_NLL_per_sample"
            ]
        )

        best_ade = float(
            initial_metrics[
                "mixture_mean_planar_ADE_m"
            ]
        )

        best_epoch = 0
        patience = 0
        history = []

        save_best(
            model,
            epoch=0,
            metrics=initial_metrics,
            architecture_configuration=(
                architecture_configuration
            ),
            experiment_configuration=(
                experiment_config
            ),
            fingerprints=fingerprints,
        )

        save_progress(
            model,
            optimizer,
            last_epoch=0,
            best_epoch=0,
            best_nll=best_nll,
            best_ade=best_ade,
            patience=patience,
            history=history,
            initial_metrics=initial_metrics,
            fingerprints=fingerprints,
        )

        start_epoch = 1

        print(
            "initialized dev joint NLL =",
            best_nll,
        )
        print(
            "initialized mixture ADE m =",
            best_ade,
        )
        print(
            "initialized minADE m      =",
            initial_metrics[
                "min_component_ADE_m"
            ],
        )
        print(
            "initialized effective K   =",
            initial_metrics[
                "effective_mode_count"
            ],
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
            "===== RESUMING GMM TRAINING ====="
        )
        print(
            "last completed epoch =",
            start_epoch - 1,
        )
        print(
            "best epoch so far    =",
            best_epoch,
        )
        print(
            "best dev joint NLL   =",
            best_nll,
        )
        print(
            "Stage2/3 rescan      = NO"
        )

    fit_n = int(
        fit_arrays[
            "target"
        ].shape[0]
    )

    training_start = (
        time.perf_counter()
    )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        max_epochs + 1,
    ):
        if patience >= (
            patience_limit
        ):
            print(
                "early stopping was already "
                "satisfied in saved progress."
            )
            break

        model.train()

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            seed + epoch
        )

        sampled = torch.multinomial(
            class_weights,
            num_samples=fit_n,
            replacement=True,
            generator=generator,
        ).numpy()

        train_loss_sum = 0.0
        train_active = 0

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

            batch = normalizer.prepare(
                fit_arrays,
                indices,
                device=device,
            )

            active = (
                batch[
                    "future_mask"
                ]
                >
                0.5
            ).any(
                dim=-1
            )

            active_count = int(
                active.sum().item()
            )

            if active_count <= 0:
                continue

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

            loss = masked_gmm_joint_nll(
                output.mixture_logits,
                output.means,
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
                    "Non-finite GMM training NLL."
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
                    "GMM backward produced "
                    "no gradients."
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
                    "GMM backward produced "
                    "non-finite gradients."
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )

            optimizer.step()

            train_loss_sum += (
                float(
                    loss.detach().item()
                )
                *
                active_count
            )

            train_active += (
                active_count
            )

        torch.cuda.synchronize()

        if train_active <= 0:
            raise RuntimeError(
                "Training epoch has "
                "zero active samples."
            )

        train_nll = (
            train_loss_sum
            /
            train_active
        )

        development = evaluate_gmm(
            model,
            dev_arrays,
            normalizer,
            device=device,
            class_id_to_name=(
                CLASS_ID_TO_NAME
            ),
        )

        dev_nll = float(
            development[
                "joint_GMM_NLL_per_sample"
            ]
        )

        dev_ade = float(
            development[
                "mixture_mean_planar_ADE_m"
            ]
        )

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
            best_nll = dev_nll
            best_ade = dev_ade
            best_epoch = epoch
            patience = 0

            save_best(
                model,
                epoch=epoch,
                metrics=development,
                architecture_configuration=(
                    architecture_configuration
                ),
                experiment_configuration=(
                    experiment_config
                ),
                fingerprints=(
                    fingerprints
                ),
            )

        else:
            patience += 1

        history.append({
            "epoch":
                int(
                    epoch
                ),

            "training_joint_GMM_NLL_per_sample":
                float(
                    train_nll
                ),

            "development_joint_GMM_NLL_per_sample":
                dev_nll,

            "development_mixture_mean_ADE_m":
                dev_ade,

            "development_MAP_ADE_m":
                float(
                    development[
                        "MAP_component_planar_ADE_m"
                    ]
                ),

            "development_minADE_m":
                float(
                    development[
                        "min_component_ADE_m"
                    ]
                ),

            "development_minFDE_1.0s_m":
                float(
                    development[
                        "min_component_FDE_1.0s_m"
                    ]
                ),

            "development_effective_mode_count":
                float(
                    development[
                        "effective_mode_count"
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
            initial_metrics=initial_metrics,
            fingerprints=fingerprints,
        )

        print(
            f"epoch {epoch:02d} "
            f"| trainNLL="
            f"{train_nll:.6f} "
            f"| devNLL="
            f"{dev_nll:.6f} "
            f"| mixADE="
            f"{dev_ade:.6f} "
            f"| minADE="
            f"{development['min_component_ADE_m']:.6f} "
            f"| effK="
            f"{development['effective_mode_count']:.4f} "
            f"| bestEpoch="
            f"{best_epoch} "
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

    # ========================================================
    # Load and evaluate frozen best checkpoint.
    # ========================================================

    if not CHECKPOINT.is_file():
        raise RuntimeError(
            "Best GMM checkpoint "
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
            "Frozen GMM checkpoint "
            "fingerprints mismatch."
        )

    if (
        "architecture_configuration"
        not in checkpoint
    ):
        raise RuntimeError(
            "GMM checkpoint lacks explicit "
            "architecture metadata."
        )

    if (
        "experiment_configuration"
        not in checkpoint
    ):
        raise RuntimeError(
            "GMM checkpoint lacks explicit "
            "experiment metadata."
        )

    best_model = build_gmm(
        checkpoint[
            "architecture_configuration"
        ],
        device=device,
    )

    best_model.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    best_model.eval()

    actual_state_sha = state_dict_sha256(
        best_model.state_dict()
    )

    if (
        actual_state_sha
        !=
        checkpoint[
            "state_dict_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen GMM state-dict "
            "SHA mismatch."
        )

    final_development = evaluate_gmm(
        best_model,
        dev_arrays,
        normalizer,
        device=device,
        class_id_to_name=(
            CLASS_ID_TO_NAME
        ),
    )

    initial_nll = float(
        initial_metrics[
            "joint_GMM_NLL_per_sample"
        ]
    )

    final_nll = float(
        final_development[
            "joint_GMM_NLL_per_sample"
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
            "GMM training did not improve "
            "development joint NLL over "
            "initialized GMM."
        )

    if (
        final_development[
            "component_covariance_SPD_failures"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Final GMM contains "
            "non-SPD covariance."
        )

    for required_class in (
        "TYPE_VEHICLE",
        "TYPE_PEDESTRIAN",
        "TYPE_CYCLIST",
    ):
        if required_class not in (
            final_development[
                "class_mixture_mean_ADE_m"
            ]
        ):
            raise RuntimeError(
                "Final GMM metrics lack "
                f"{required_class}."
            )

    # ========================================================
    # Exact checkpoint inference repeat.
    # ========================================================

    inference_sha_a = inference_sha256(
        best_model,
        dev_arrays,
        normalizer,
        device=device,
    )

    repeated_checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    repeat_model = build_gmm(
        repeated_checkpoint[
            "architecture_configuration"
        ],
        device=device,
    )

    repeat_model.load_state_dict(
        repeated_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    repeat_model.eval()

    inference_sha_b = inference_sha256(
        repeat_model,
        dev_arrays,
        normalizer,
        device=device,
    )

    if (
        inference_sha_a
        !=
        inference_sha_b
    ):
        raise RuntimeError(
            "Frozen GMM checkpoint "
            "inference is not exactly repeatable."
        )

    print()
    print(
        "exact GMM inference repeat = PASS"
    )
    print(
        "GMM prediction SHA256      =",
        inference_sha_a,
    )

    # ========================================================
    # Final 92-test regression.
    # ========================================================

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
        test.returncode != 0
        or
        test_count != 92
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected final Stage4 "
            "regression 92/92."
        )

    checkpoint_sha = file_sha256(
        CHECKPOINT
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
            "250-GiB reserve violated."
        )

    gaussian_dev_ade = float(
        block44[
            "development"
        ][
            "best_raw_uncalibrated"
        ][
            "planar_ADE_m"
        ]
    )

    gmm_dev_ade = float(
        final_development[
            "mixture_mean_planar_ADE_m"
        ]
    )

    # This is diagnostic only.
    # Block4.6 does not automatically replace
    # the mandatory calibrated Gaussian.
    ade_delta_vs_gaussian = (
        gmm_dev_ade
        -
        gaussian_dev_ade
    )

    final_report = {
        "stage":
            4,

        "block":
            "4.6",

        "status":
            "PASS",

        "role": {
            "mandatory_probabilistic_baseline":
                "frozen_calibrated_Gaussian_Block4.5",

            "GMM":
                "multimodal_ablation_extension",

            "GMM_automatically_replaces_Gaussian":
                False,

            "GMM_downstream_selected":
                False,

            "GMM_calibration_required_if_selected":
                True,

            "Gaussian_adequacy_decision":
                "DEFERRED_TO_BLOCK4.8_FORMAL_AND_SLICES",
        },

        "upstream": {
            "fit_cache_reused":
                True,

            "development_cache_reused":
                True,

            "Stage2_Stage3_rescan":
                False,

            "normalization":
                "FROZEN_FIT_ONLY",

            "Gaussian_checkpoint_sha256":
                EXPECTED_GAUSSIAN_CHECKPOINT_SHA,

            "calibrated_Gaussian_artifact_sha256":
                EXPECTED_CALIBRATOR_SHA,
        },

        "distribution": {
            "components":
                3,

            "mode_scope":
                "trajectory_level",

            "independent_horizon_mode_switching":
                False,

            "component_covariance":
                "full_3D_SPD_Cholesky",

            "semantic_maneuver_labels":
                False,
        },

        "training": {
            "seed":
                seed,

            "device":
                str(
                    device
                ),

            "GPU":
                torch.cuda
                .get_device_name(0),

            "objective":
                "joint_trajectory_GMM_NLL",

            "model_selection":
                "development_joint_GMM_NLL_per_sample",

            "tiebreaker":
                "development_mixture_mean_planar_ADE",

            "class_balanced_sampler":
                True,

            "primary_class_sampling_mass":
                class_mass,

            "best_epoch":
                int(
                    checkpoint[
                        "best_epoch"
                    ]
                ),

            "history":
                history,

            "runtime_this_invocation_s":
                training_runtime_s,
        },

        "development": {
            "initialized_GMM":
                initial_metrics,

            "best_GMM":
                final_development,

            "joint_NLL_improved":
                True,

            "Gaussian_mixture_mean_ADE_reference_m":
                gaussian_dev_ade,

            "GMM_minus_Gaussian_ADE_m":
                ade_delta_vs_gaussian,

            "ADE_comparison_is_diagnostic_only":
                True,
        },

        "multimodality": {
            "minADE_m":
                final_development[
                    "min_component_ADE_m"
                ],

            "minFDE_1.0s_m":
                final_development[
                    "min_component_FDE_1.0s_m"
                ],

            "mixture_entropy_nats":
                final_development[
                    "mixture_entropy_nats"
                ],

            "effective_mode_count":
                final_development[
                    "effective_mode_count"
                ],

            "mean_mixture_probability":
                final_development[
                    "mean_mixture_probability"
                ],

            "MAP_mode_counts":
                final_development[
                    "MAP_mode_counts"
                ],

            "mean_pairwise_mode_separation_1.0s_m":
                final_development[
                    "mean_pairwise_mode_separation_1.0s_m"
                ],

            "mode_diversity_is_closure_gate":
                False,
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

            "architecture_metadata_separate":
                True,

            "experiment_metadata_separate":
                True,

            "exact_inference_repeat":
                True,

            "prediction_sha256":
                inference_sha_a,
        },

        "calibration": {
            "GMM_calibrated":
                False,

            "reason":
                "GMM is not selected for downstream use in Block4.6",

            "required_before_downstream_use_if_selected":
                True,
        },

        "formal_validation": {
            "used":
                False,

            "evaluation":
                "DEFERRED_TO_BLOCK4.8",
        },

        "regression": {
            "tests_passed":
                92,

            "tests_total":
                92,
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
            "GMM superiority on frozen formal N=120 validation",
            "GMM calibration",
            "Gaussian inadequacy",
            "GMM downstream selection",
            "measurement-covariance ablation",
            "multi-agent/map ablation",
            "receiver/angular posterior"
        ],
    }

    write_json(
        REPORT,
        final_report,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    # ========================================================
    # Closure log.
    # ========================================================

    marker = (
        "## Block 4.6 — "
        "Trajectory-level GMM ablation"
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
                "- The mandatory calibrated Gaussian "
                  "from Block4.5 remains frozen and "
                  "is not automatically replaced.\n"
                "- A K=3 trajectory-level GMM is "
                  "implemented and trained using the "
                  "same frozen causal input/cache and "
                  "fit-only normalization.\n"
                "- One mixture weight applies to each "
                  "complete future trajectory mode; "
                  "independent horizon mode switching "
                  "is not allowed.\n"
                "- Every component has a full 3-D "
                  "SPD Cholesky covariance at every "
                  "prediction horizon.\n"
                "- Initialization copies the frozen "
                  "Block4.4 Gaussian encoder/fusion/"
                  "covariance heads and uses only a "
                  "small deterministic symmetry-break "
                  "offset between component means.\n"
                "- No maneuver labels, perfect track "
                  "IDs, future states, tracks_to_predict "
                  "or objects_of_interest are model "
                  "inputs.\n"
                "- Training objective is joint "
                  "trajectory GMM NLL.\n"
                "- Development model selection uses "
                  "joint GMM NLL; mixture-mean ADE is "
                  "only a tiebreaker.\n"
                "- minADE, 1.0-s minFDE, mixture "
                  "entropy, effective mode count and "
                  "mode separation are measured.\n"
                "- Mode diversity is diagnostic and "
                  "not a post-hoc closure threshold.\n"
                "- The GMM is not calibrated or "
                  "selected for downstream use in "
                  "Block4.6. If later selected, "
                  "calibration is required first.\n"
                "- Formal validation remains untouched "
                  "until Block4.8.\n"
                f"- Initial joint GMM NLL: "
                f"{initial_nll:.9f}.\n"
                f"- Best joint GMM NLL: "
                f"{final_nll:.9f}.\n"
                f"- Mixture-mean development ADE: "
                f"{final_development['mixture_mean_planar_ADE_m']:.9f} m.\n"
                f"- MAP-component development ADE: "
                f"{final_development['MAP_component_planar_ADE_m']:.9f} m.\n"
                f"- minADE: "
                f"{final_development['min_component_ADE_m']:.9f} m.\n"
                f"- 1.0-s minFDE: "
                f"{final_development['min_component_FDE_1.0s_m']:.9f} m.\n"
                f"- Effective mode count: "
                f"{final_development['effective_mode_count']:.9f}.\n"
                f"- Mean 1.0-s pairwise mode separation: "
                f"{final_development['mean_pairwise_mode_separation_1.0s_m']:.9f} m.\n"
                f"- Best epoch: "
                f"{checkpoint['best_epoch']}.\n"
                f"- Checkpoint SHA256: "
                f"{checkpoint_sha}.\n"
                f"- State-dict SHA256: "
                f"{actual_state_sha}.\n"
                f"- Prediction SHA256: "
                f"{inference_sha_a}.\n"
                "- Exact inference repeat: PASS.\n"
                "- Full Stage4 regression: "
                  "92/92 PASS.\n"
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
        "STAGE4 BLOCK 4.6 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "fit/dev cache reuse        = PASS"
    )
    print(
        "Stage2/3 rescan            = NO"
    )
    print(
        "fit-only normalization     = PASS"
    )
    print(
        "mandatory Gaussian         = FROZEN / UNCHANGED"
    )
    print(
        "GMM components             = 3"
    )
    print(
        "trajectory-level modes     = PASS"
    )
    print(
        "component SPD failures     =",
        final_development[
            "component_covariance_SPD_failures"
        ],
    )

    print()
    print(
        "initialized joint GMM NLL  =",
        round(
            initial_nll,
            6,
        ),
    )
    print(
        "best joint GMM NLL         =",
        round(
            final_nll,
            6,
        ),
    )
    print(
        "joint NLL improved         = PASS"
    )
    print(
        "best epoch                 =",
        checkpoint[
            "best_epoch"
        ],
    )

    print()
    print(
        "mixture-mean ADE m         =",
        round(
            final_development[
                "mixture_mean_planar_ADE_m"
            ],
            6,
        ),
    )
    print(
        "MAP-component ADE m        =",
        round(
            final_development[
                "MAP_component_planar_ADE_m"
            ],
            6,
        ),
    )
    print(
        "minADE m                   =",
        round(
            final_development[
                "min_component_ADE_m"
            ],
            6,
        ),
    )
    print(
        "minFDE @1.0s m             =",
        round(
            final_development[
                "min_component_FDE_1.0s_m"
            ],
            6,
        ),
    )
    print(
        "mixture-mean 1.0s error m  =",
        round(
            final_development[
                "mixture_mean_one_second_error_m"
            ],
            6,
        ),
    )

    print()
    print(
        "mixture entropy nats       =",
        round(
            final_development[
                "mixture_entropy_nats"
            ],
            6,
        ),
    )
    print(
        "effective mode count       =",
        round(
            final_development[
                "effective_mode_count"
            ],
            6,
        ),
    )
    print(
        "mean mixture probability   =",
        final_development[
            "mean_mixture_probability"
        ],
    )
    print(
        "MAP mode counts            =",
        final_development[
            "MAP_mode_counts"
        ],
    )
    print(
        "pairwise mode sep @1.0s m  =",
        round(
            final_development[
                "mean_pairwise_mode_separation_1.0s_m"
            ],
            6,
        ),
    )

    print()
    print(
        "Gaussian dev ADE reference =",
        round(
            gaussian_dev_ade,
            6,
        ),
    )
    print(
        "GMM-Gaussian ADE delta m   =",
        round(
            ade_delta_vs_gaussian,
            6,
        ),
    )
    print(
        "comparison role            = DIAGNOSTIC"
    )

    print()
    print(
        "class mixture ADE          =",
        final_development[
            "class_mixture_mean_ADE_m"
        ],
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
        "GMM prediction SHA256      =",
        inference_sha_a,
    )
    print(
        "exact inference repeat     = PASS"
    )
    print(
        "GMM calibrated             = NO"
    )
    print(
        "GMM downstream selected    = NO"
    )
    print(
        "formal validation used     = NO"
    )
    print(
        "full Stage4 regression     = 92 / 92 PASS"
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
        "===== BLOCK 4.6 FINAL ====="
    )
    print(
        "trajectory-level GMM       = PASS"
    )
    print(
        "joint GMM NLL training     = PASS"
    )
    print(
        "full SPD components        = PASS"
    )
    print(
        "minADE/minFDE              = MEASURED"
    )
    print(
        "mode diagnostics           = MEASURED"
    )
    print(
        "mandatory Gaussian         = PRESERVED"
    )
    print(
        "GMM downstream selection   = NOT YET"
    )
    print(
        "formal validation leak     = NONE"
    )
    print(
        "checkpoint freeze          = PASS"
    )
    print(
        "exact inference repeat     = PASS"
    )
    print(
        "regression                 = 92 / 92"
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
            "Check nvidia-smi for another GPU "
            "process first. Do not reduce the "
            "frozen K=3 model automatically. "
            "Completed epoch progress is resumable."
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
            "Do not regenerate Stage2/3 caches. "
            "The last completed GMM epoch is "
            "preserved. Inspect the GMM NLL/"
            "component covariance state before "
            "changing optimization."
        )

    if (
        "checkpoint"
        in text
        or
        "state"
        in text
        or
        "fingerprint"
        in text
    ):
        return (
            "Do not retrain Blocks4.3–4.5. "
            "Audit the explicit architecture/"
            "experiment metadata and frozen "
            "checkpoint hashes first."
        )

    if (
        "python.h"
        in text
        or
        "triton"
        in text
    ):
        return (
            "Reuse the existing python3.13-dev "
            "repair from Block4.4. Do not "
            "reinstall PyTorch."
        )

    if (
        "92/92"
        in text
        or
        "regression"
        in text
    ):
        return (
            "Inspect the failing regression. "
            "Do not alter frozen Blocks4.0–4.5 "
            "to satisfy new GMM code."
        )

    return (
        "Inspect reports/"
        "block46_training_failure.json. "
        "GMM epoch progress is resumable and "
        "the frozen fit/development caches "
        "do not need regeneration."
    )


try:
    main()

except BaseException as exc:
    payload = {
        "stage":
            4,

        "block":
            "4.6_part_2",

        "status":
            "BLOCKED",

        "exception_type":
            type(exc).__name__,

        "exception":
            str(exc),

        "traceback":
            traceback.format_exc(),

        "recovery_hint":
            recovery_hint(
                exc
            ),

        "GMM_epoch_progress_resumable":
            True,

        "Stage2_Stage3_cache_regenerated":
            False,

        "Block45_modified":
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
        "BLOCK 4.6 PART 2/2 = BLOCKED"
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
        "GMM epoch progress        = RESUMABLE"
    )
    print(
        "Stage2/3 cache rebuilt    = NO"
    )
    print(
        "Block4.5 modified         = NO"
    )
    print(
        "formal validation used    = NO"
    )
    print(
        "terminal remains open     = YES"
    )
    print(
        "failure report =",
        FAILURE,
    )

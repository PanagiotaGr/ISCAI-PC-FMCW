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

from iscai_stage4.ml import (
    CLASS_CYCLIST,
    CLASS_ID_TO_NAME,
    CLASS_PEDESTRIAN,
    CLASS_VEHICLE,
    GaussianTrajectoryGRU,
    balanced_class_weights,
    denormalize_gaussian,
    gaussian_nll_per_horizon,
    initialize_from_deterministic,
    masked_gaussian_nll,
    set_global_determinism,
)

from iscai_stage4.ml.ablation_runtime import (
    ABLATION_ACTOR_ONLY,
    ABLATION_FULL,
    ABLATION_NO_MAP,
    ABLATION_NO_R,
    apply_normalized_ablation,
)

from iscai_stage4.ml.gmm_runtime import (
    GMMNormalizer,
)

from iscai_stage4.ml.uncertainty_analysis import (
    EXPECTED_R_INDICES,
    evaluate_gaussian_condition,
    latest_measurement_sigma_rms_m,
    uncertainty_relation_report,
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
    / "reports/block47_part1_preflight.json"
)

FEATURE_CONTRACT = (
    STAGE4
    / "artifacts/block47/"
      "measurement_covariance_feature_contract.json"
)

BLOCK46 = (
    STAGE4
    / "reports/block46_gmm_gru.json"
)

BLOCK45 = (
    STAGE4
    / "reports/block45_calibration.json"
)

BLOCK44 = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
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

ARTIFACT_DIR = (
    STAGE4
    / "artifacts/block47"
)

REPORT = (
    STAGE4
    / "reports/block47_ablation_analysis.json"
)

FAILURE = (
    STAGE4
    / "reports/block47_failure.json"
)

LOG = (
    STAGE4
    / "docs/implementation_log.md"
)

EXPECTED_BLOCK46_IMPL_SHA = (
    "bf7ba0ef8c83d0b6a8d5ef2acd624190"
    "35f4367b17d4d6844f7a99c288f1c961"
)

EXPECTED_DETERMINISTIC_CHECKPOINT_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
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

VARIANTS = (
    ABLATION_NO_R,
    ABLATION_ACTOR_ONLY,
    ABLATION_NO_MAP,
)

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
    path,
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
                    data[key]
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
                f"{path.name}:{key} shape changed."
            )

    return arrays


def build_model(
    deterministic_configuration,
    *,
    device,
):
    section = (
        deterministic_configuration[
            "model"
        ]
    )

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
    ).to(device)


def checkpoint_path(
    variant,
):
    return (
        ARTIFACT_DIR
        /
        f"{variant}_gaussian.pt"
    )


def progress_path(
    variant,
):
    return (
        ARTIFACT_DIR
        /
        f"{variant}_progress.pt"
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


def save_best(
    variant,
    model,
    *,
    epoch,
    metrics,
    architecture_configuration,
    experiment_configuration,
    fingerprints,
):
    path = checkpoint_path(
        variant
    )

    state_sha = state_dict_sha256(
        model.state_dict()
    )

    atomic_torch_save(
        {
            "stage":
                4,

            "block":
                "4.7",

            "variant":
                variant,

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
        },
        path,
    )


def save_progress(
    variant,
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
        progress_path(
            variant
        ),
    )


def load_progress(
    variant,
    model,
    optimizer,
    *,
    device,
    fingerprints,
):
    path = progress_path(
        variant
    )

    if not path.is_file():
        return None

    try:
        payload = torch.load(
            path,
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
                "Progress fingerprints changed."
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
        archived = (
            path.with_name(
                path.name
                +
                f".corrupt.{int(time.time())}"
            )
        )

        os.replace(
            path,
            archived,
        )

        print(
            "WARNING: invalid progress archived:",
            archived,
        )

        print(
            "reason =",
            type(exc).__name__,
            str(exc),
        )

        return None


def evaluate_variant(
    model,
    arrays,
    normalizer,
    *,
    variant,
    device,
    batch_size=1024,
):
    model.eval()

    total_nll = 0.0
    total_planar = 0.0
    total_valid = 0

    horizon_planar_sum = np.zeros(
        4,
        dtype=np.float64,
    )

    horizon_count = np.zeros(
        4,
        dtype=np.int64,
    )

    class_error = {}
    class_valid = {}

    SPD_failures = 0

    n = int(
        arrays[
            "target"
        ].shape[0]
    )

    with torch.inference_mode():
        for start in range(
            0,
            n,
            batch_size,
        ):
            stop = min(
                n,
                start + batch_size,
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

            (
                target,
                neighbors,
                neighbor_mask,
                map_context,
            ) = apply_normalized_ablation(
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
                mode=variant,
                covariance_indices=(
                    EXPECTED_R_INDICES
                ),
            )

            output = model(
                target,
                neighbors,
                neighbor_mask,
                map_context,
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

            truth = batch[
                "true_displacement"
            ]

            valid = (
                batch[
                    "future_mask"
                ]
                >
                0.5
            )

            nll = gaussian_nll_per_horizon(
                mean_metric,
                scale_metric,
                truth,
            )

            planar = torch.sqrt(
                (
                    mean_metric[
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
                    mean_metric[
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

            if not bool(
                torch.isfinite(
                    nll[
                        valid
                    ]
                ).all()
            ):
                raise RuntimeError(
                    f"{variant}: non-finite NLL."
                )

            covariance = (
                scale_metric
                @
                scale_metric.transpose(
                    -1,
                    -2,
                )
            )

            (
                _,
                info,
            ) = torch.linalg.cholesky_ex(
                covariance
            )

            SPD_failures += int(
                (
                    info != 0
                ).sum().item()
            )

            total_nll += float(
                nll[
                    valid
                ].sum().item()
            )

            total_planar += float(
                planar[
                    valid
                ].sum().item()
            )

            total_valid += int(
                valid.sum().item()
            )

            for horizon in range(4):
                mask = valid[
                    :,
                    horizon
                ]

                count = int(
                    mask.sum().item()
                )

                if count <= 0:
                    continue

                horizon_count[
                    horizon
                ] += count

                horizon_planar_sum[
                    horizon
                ] += float(
                    planar[
                        :,
                        horizon
                    ][
                        mask
                    ].sum().item()
                )

            classes = batch[
                "class_id"
            ]

            for class_id in torch.unique(
                classes
            ).detach().cpu().tolist():
                class_id = int(
                    class_id
                )

                mask = (
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
                    mask.sum().item()
                )

                if count <= 0:
                    continue

                name = CLASS_ID_TO_NAME.get(
                    class_id,
                    str(
                        class_id
                    ),
                )

                class_error[
                    name
                ] = (
                    class_error.get(
                        name,
                        0.0,
                    )
                    +
                    float(
                        planar[
                            mask
                        ].sum().item()
                    )
                )

                class_valid[
                    name
                ] = (
                    class_valid.get(
                        name,
                        0,
                    )
                    +
                    count
                )

    if total_valid <= 0:
        raise RuntimeError(
            f"{variant}: no valid labels."
        )

    if SPD_failures != 0:
        raise RuntimeError(
            f"{variant}: SPD failures={SPD_failures}"
        )

    names = (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    )

    horizon_error = {
        names[h]:
            float(
                horizon_planar_sum[h]
                /
                horizon_count[h]
            )
        for h in range(4)
    }

    return {
        "metric_Gaussian_NLL":
            float(
                total_nll
                /
                total_valid
            ),

        "planar_ADE_m":
            float(
                total_planar
                /
                total_valid
            ),

        "horizon_planar_error_m":
            horizon_error,

        "one_second_planar_error_m":
            horizon_error[
                "1.0"
            ],

        "class_planar_ADE_m": {
            name:
                float(
                    class_error[
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
        },

        "predictive_covariance_SPD_failures":
            0,
    }


def train_variant(
    variant,
    *,
    fit_arrays,
    dev_arrays,
    normalizer,
    deterministic_checkpoint,
    Gaussian_experiment_configuration,
    fingerprints,
    device,
):
    training = (
        Gaussian_experiment_configuration[
            "training"
        ]
    )

    seed = int(
        training[
            "seed"
        ]
    )

    batch_size = int(
        training[
            "batch_size"
        ]
    )

    max_epochs = int(
        training[
            "max_epochs"
        ]
    )

    patience_limit = int(
        training[
            "early_stopping_patience"
        ]
    )

    learning_rate = float(
        training[
            "learning_rate"
        ]
    )

    weight_decay = float(
        training[
            "weight_decay"
        ]
    )

    gradient_clip = float(
        training[
            "gradient_clip_norm"
        ]
    )

    set_global_determinism(
        seed
    )

    architecture = (
        deterministic_checkpoint[
            "configuration"
        ]
    )

    model = build_model(
        architecture,
        device=device,
    )

    initial_std = float(
        Gaussian_experiment_configuration[
            "predictive_distribution"
        ][
            "initial_predictive_std_normalized"
        ]
    )

    initialize_from_deterministic(
        model,
        deterministic_checkpoint[
            "state_dict"
        ],
        initial_std_normalized=(
            initial_std
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    progress = load_progress(
        variant,
        model,
        optimizer,
        device=device,
        fingerprints=fingerprints,
    )

    if progress is None:
        initial_metrics = evaluate_variant(
            model,
            dev_arrays,
            normalizer,
            variant=variant,
            device=device,
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

        save_best(
            variant,
            model,
            epoch=0,
            metrics=initial_metrics,
            architecture_configuration=(
                architecture
            ),
            experiment_configuration=(
                Gaussian_experiment_configuration
            ),
            fingerprints=fingerprints,
        )

        save_progress(
            variant,
            model,
            optimizer,
            last_epoch=0,
            best_epoch=0,
            best_nll=best_nll,
            best_ade=best_ade,
            patience=0,
            history=history,
            initial_metrics=initial_metrics,
            fingerprints=fingerprints,
        )

        start_epoch = 1

        print()
        print(
            f"===== {variant} INITIAL ====="
        )
        print(
            "initial dev NLL =",
            best_nll,
        )
        print(
            "initial dev ADE =",
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
            f"===== {variant} RESUME ====="
        )
        print(
            "last completed epoch =",
            start_epoch - 1,
        )

    class_weights = (
        balanced_class_weights(
            fit_arrays[
                "class_id"
            ]
        )
    )

    fit_n = int(
        fit_arrays[
            "target"
        ].shape[0]
    )

    for epoch in range(
        start_epoch,
        max_epochs + 1,
    ):
        if patience >= (
            patience_limit
        ):
            break

        model.train()

        generator = torch.Generator(
            device="cpu"
        )

        # SAME sampler sequence as the full Gaussian.
        generator.manual_seed(
            seed + epoch
        )

        sampled = torch.multinomial(
            class_weights,
            num_samples=fit_n,
            replacement=True,
            generator=generator,
        ).numpy()

        training_loss_sum = 0.0
        training_valid = 0

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

            (
                target,
                neighbors,
                neighbor_mask,
                map_context,
            ) = apply_normalized_ablation(
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
                mode=variant,
                covariance_indices=(
                    EXPECTED_R_INDICES
                ),
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                target,
                neighbors,
                neighbor_mask,
                map_context,
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
                    f"{variant}: non-finite training loss."
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
                    f"{variant}: no gradients."
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
                    f"{variant}: non-finite gradients."
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip,
            )

            optimizer.step()

            valid_count = int(
                batch[
                    "future_mask"
                ].sum().item()
            )

            training_loss_sum += (
                float(
                    loss.detach().item()
                )
                *
                valid_count
            )

            training_valid += (
                valid_count
            )

        torch.cuda.synchronize()

        train_nll = (
            training_loss_sum
            /
            training_valid
        )

        development = evaluate_variant(
            model,
            dev_arrays,
            normalizer,
            variant=variant,
            device=device,
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
                variant,
                model,
                epoch=epoch,
                metrics=development,
                architecture_configuration=(
                    architecture
                ),
                experiment_configuration=(
                    Gaussian_experiment_configuration
                ),
                fingerprints=fingerprints,
            )

        else:
            patience += 1

        history.append({
            "epoch":
                int(
                    epoch
                ),

            "train_normalized_NLL":
                float(
                    train_nll
                ),

            "dev_metric_NLL":
                dev_nll,

            "dev_ADE_m":
                dev_ade,

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
            variant,
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
            f"{variant} "
            f"epoch {epoch:02d} "
            f"| trainNLL={train_nll:.6f} "
            f"| devNLL={dev_nll:.6f} "
            f"| devADE={dev_ade:.6f} "
            f"| bestEpoch={best_epoch} "
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
                f"{variant} early stopping = TRIGGERED"
            )
            break

    best_payload = torch.load(
        checkpoint_path(
            variant
        ),
        map_location=device,
        weights_only=False,
    )

    best_model = build_model(
        best_payload[
            "architecture_configuration"
        ],
        device=device,
    )

    best_model.load_state_dict(
        best_payload[
            "state_dict"
        ],
        strict=True,
    )

    final_metrics = evaluate_variant(
        best_model,
        dev_arrays,
        normalizer,
        variant=variant,
        device=device,
    )

    if not (
        final_metrics[
            "metric_Gaussian_NLL"
        ]
        <
        initial_metrics[
            "metric_Gaussian_NLL"
        ]
        -
        1e-6
    ):
        raise RuntimeError(
            f"{variant}: training did not "
            "improve initialized NLL."
        )

    return {
        "initial":
            initial_metrics,

        "best":
            final_metrics,

        "best_epoch":
            int(
                best_payload[
                    "best_epoch"
                ]
            ),

        "history":
            history,

        "checkpoint_file_sha256":
            file_sha256(
                checkpoint_path(
                    variant
                )
            ),

        "state_dict_sha256":
            state_dict_sha256(
                best_payload[
                    "state_dict"
                ]
            ),
    }


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 4 — BLOCK 4.7 "
        "ABLATIONS + UNCERTAINTY ANALYSIS"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        FEATURE_CONTRACT,
        BLOCK46,
        BLOCK45,
        BLOCK44,
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
            "Missing required artifact(s): "
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

    feature_contract = json.loads(
        FEATURE_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    block46 = json.loads(
        BLOCK46.read_text(
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

    if (
        part1.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.7 Part1 is not PASS."
        )

    expected_names = (
        "meas_cov_xx_H0_m2",
        "meas_cov_xy_H0_m2",
        "meas_cov_xz_H0_m2",
        "meas_cov_yy_H0_m2",
        "meas_cov_yz_H0_m2",
        "meas_cov_zz_H0_m2",
    )

    if tuple(
        feature_contract[
            "measurement_covariance_indices"
        ]
    ) != EXPECTED_R_INDICES:
        raise RuntimeError(
            "Frozen R_t indices changed."
        )

    if tuple(
        feature_contract[
            "measurement_covariance_names"
        ]
    ) != expected_names:
        raise RuntimeError(
            "Frozen R_t names changed."
        )

    if (
        block46.get(
            "status"
        )
        !=
        "PASS"
    ):
        raise RuntimeError(
            "Block4.6 is not PASS."
        )

    if (
        block46[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK46_IMPL_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.6 closure "
            "fingerprint changed."
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
            "checkpoint changed."
        )

    if (
        file_sha256(
            GAUSSIAN_CHECKPOINT
        )
        !=
        EXPECTED_GAUSSIAN_CHECKPOINT_SHA
    ):
        raise RuntimeError(
            "Frozen Gaussian checkpoint changed."
        )

    if (
        file_sha256(
            CALIBRATOR
        )
        !=
        EXPECTED_CALIBRATOR_SHA
    ):
        raise RuntimeError(
            "Frozen Block4.5 calibrator changed."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
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
            "250-GiB storage reserve violated."
        )

    deterministic_checkpoint = torch.load(
        DETERMINISTIC_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    Gaussian_checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        state_dict_sha256(
            Gaussian_checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED_GAUSSIAN_STATE_SHA
    ):
        raise RuntimeError(
            "Frozen Gaussian state changed."
        )

    Gaussian_experiment_configuration = (
        Gaussian_checkpoint[
            "configuration"
        ]
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
            "Normalization is not fit-only."
        )

    calibrator = json.loads(
        CALIBRATOR.read_text(
            encoding="utf-8"
        )
    )

    calibrated_std_scale = np.asarray(
        calibrator[
            "standard_deviation_scale"
        ],
        dtype=np.float64,
    )

    fit_arrays = load_cache(
        FIT_CACHE
    )

    dev_arrays = load_cache(
        DEV_CACHE
    )

    normalizer = GMMNormalizer(
        normalization,
        device=device,
    )

    print(
        "Block4.7 Part1          = PASS"
    )
    print(
        "R_t mapping             = 6..11 FROZEN"
    )
    print(
        "fit/dev cache reuse     = PASS"
    )
    print(
        "Stage2/3 rescan         = NO"
    )
    print(
        "normalization           = FIT ONLY"
    )
    print(
        "formal validation used  = NO"
    )

    # ========================================================
    # A. Cross-check the new analysis runtime against
    #    the frozen Block4.4 Gaussian metrics.
    # ========================================================

    full_model = build_model(
        deterministic_checkpoint[
            "configuration"
        ],
        device=device,
    )

    full_model.load_state_dict(
        Gaussian_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    full_model.eval()

    full_dev = evaluate_gaussian_condition(
        full_model,
        dev_arrays,
        normalizer,
        device=device,
        ablation_mode=(
            ABLATION_FULL
        ),
        R_variance_scale=1.0,
        return_sample_arrays=True,
    )

    frozen_ade = float(
        block44[
            "development"
        ][
            "best_raw_uncalibrated"
        ][
            "planar_ADE_m"
        ]
    )

    frozen_nll = float(
        block44[
            "development"
        ][
            "best_raw_uncalibrated"
        ][
            "metric_Gaussian_NLL"
        ]
    )

    if (
        abs(
            full_dev[
                "planar_ADE_m"
            ]
            -
            frozen_ade
        )
        >
        1e-5
    ):
        raise RuntimeError(
            "New ablation runtime does not "
            "reproduce frozen Gaussian ADE."
        )

    if (
        abs(
            full_dev[
                "metric_Gaussian_NLL"
            ]
            -
            frozen_nll
        )
        >
        1e-5
    ):
        raise RuntimeError(
            "New ablation runtime does not "
            "reproduce frozen Gaussian NLL."
        )

    print()
    print(
        "===== FROZEN FULL-GAUSSIAN CROSS-CHECK ====="
    )
    print(
        "ADE m =",
        full_dev[
            "planar_ADE_m"
        ],
        "PASS",
    )
    print(
        "NLL   =",
        full_dev[
            "metric_Gaussian_NLL"
        ],
        "PASS",
    )

    # ========================================================
    # B. Physical measurement→predictive uncertainty analysis
    # ========================================================

    measurement = (
        latest_measurement_sigma_rms_m(
            dev_arrays[
                "target"
            ]
        )
    )

    if (
        measurement[
            "PSD_violation_count"
        ]
        !=
        0
    ):
        raise RuntimeError(
            "Latest target R_t contains "
            "PSD violations."
        )

    samples = full_dev[
        "sample_arrays"
    ]

    relation = uncertainty_relation_report(
        measurement[
            "sigma_rms_m"
        ],
        samples[
            "predictive_sigma_RMS_m"
        ],
        samples[
            "planar_error_m"
        ],
        samples[
            "validity"
        ],
        calibrated_std_scale=(
            calibrated_std_scale
        ),
    )

    print()
    print(
        "===== MEASUREMENT → PREDICTIVE UNCERTAINTY ====="
    )
    print(
        "measurement sigma = sqrt(trace(R_t)/3)"
    )
    print(
        "R_t PSD violations = 0 PASS"
    )
    print(
        "minimum R_t eigenvalue =",
        measurement[
            "minimum_eigenvalue"
        ],
    )

    for horizon in (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    ):
        value = relation[
            "correlations"
        ][
            horizon
        ]

        print(
            horizon,
            "s | Spearman meas→pred sigma =",
            round(
                value[
                    "Spearman_measurement_vs_predictive_sigma"
                ],
                6,
            ),
            "| meas→error =",
            round(
                value[
                    "Spearman_measurement_vs_planar_error"
                ],
                6,
            ),
        )

    # ========================================================
    # C. Controlled physical R_t sensitivity.
    # ========================================================

    sensitivity = {}

    print()
    print(
        "===== PHYSICAL R_t SCALE SENSITIVITY ====="
    )

    for scale in (
        0.5,
        1.0,
        2.0,
    ):
        metrics = evaluate_gaussian_condition(
            full_model,
            dev_arrays,
            normalizer,
            device=device,
            ablation_mode=(
                ABLATION_FULL
            ),
            R_variance_scale=scale,
            return_sample_arrays=False,
        )

        sensitivity[
            str(scale)
        ] = metrics

        print(
            f"R_t x {scale:.1f}",
            "| ADE=",
            round(
                metrics[
                    "planar_ADE_m"
                ],
                6,
            ),
            "| NLL=",
            round(
                metrics[
                    "metric_Gaussian_NLL"
                ],
                6,
            ),
            "| pred sigma@1s=",
            round(
                metrics[
                    "mean_predictive_sigma_RMS_m"
                ][
                    "1.0"
                ],
                6,
            ),
        )

    # ========================================================
    # D. Fair retrained ablations
    # ========================================================

    fingerprints_base = {
        "Part1_report_sha256":
            file_sha256(
                PART1
            ),

        "feature_contract_sha256":
            file_sha256(
                FEATURE_CONTRACT
            ),

        "deterministic_checkpoint_sha256":
            EXPECTED_DETERMINISTIC_CHECKPOINT_SHA,

        "Gaussian_checkpoint_sha256":
            EXPECTED_GAUSSIAN_CHECKPOINT_SHA,

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

    ablations = {}

    for variant in VARIANTS:
        print()
        print(
            "============================================================"
        )
        print(
            "TRAINING ABLATION:",
            variant,
        )
        print(
            "============================================================"
        )

        fingerprints = dict(
            fingerprints_base
        )

        fingerprints[
            "variant"
        ] = variant

        result = train_variant(
            variant,
            fit_arrays=fit_arrays,
            dev_arrays=dev_arrays,
            normalizer=normalizer,
            deterministic_checkpoint=(
                deterministic_checkpoint
            ),
            Gaussian_experiment_configuration=(
                Gaussian_experiment_configuration
            ),
            fingerprints=fingerprints,
            device=device,
        )

        ablations[
            variant
        ] = result

        print()
        print(
            variant,
            "BEST NLL =",
            result[
                "best"
            ][
                "metric_Gaussian_NLL"
            ],
        )

        print(
            variant,
            "BEST ADE =",
            result[
                "best"
            ][
                "planar_ADE_m"
            ],
        )

        del result
        torch.cuda.empty_cache()

    # Reload concise results after deletion.
    concise = {}

    for variant in VARIANTS:
        checkpoint = torch.load(
            checkpoint_path(
                variant
            ),
            map_location=device,
            weights_only=False,
        )

        model = build_model(
            checkpoint[
                "architecture_configuration"
            ],
            device=device,
        )

        model.load_state_dict(
            checkpoint[
                "state_dict"
            ],
            strict=True,
        )

        metrics = evaluate_variant(
            model,
            dev_arrays,
            normalizer,
            variant=variant,
            device=device,
        )

        concise[
            variant
        ] = {
            "best_epoch":
                int(
                    checkpoint[
                        "best_epoch"
                    ]
                ),

            "metrics":
                metrics,

            "checkpoint_file_sha256":
                file_sha256(
                    checkpoint_path(
                        variant
                    )
                ),

            "state_dict_sha256":
                state_dict_sha256(
                    checkpoint[
                        "state_dict"
                    ]
                ),
        }

        del model
        torch.cuda.empty_cache()

    # ========================================================
    # E. Comparison against frozen full Gaussian
    # ========================================================

    comparison = {
        "full":
            {
                "NLL":
                    full_dev[
                        "metric_Gaussian_NLL"
                    ],

                "ADE_m":
                    full_dev[
                        "planar_ADE_m"
                    ],
            }
    }

    for variant in VARIANTS:
        metrics = concise[
            variant
        ][
            "metrics"
        ]

        comparison[
            variant
        ] = {
            "NLL":
                metrics[
                    "metric_Gaussian_NLL"
                ],

            "ADE_m":
                metrics[
                    "planar_ADE_m"
                ],

            "delta_NLL_vs_full":
                (
                    metrics[
                        "metric_Gaussian_NLL"
                    ]
                    -
                    full_dev[
                        "metric_Gaussian_NLL"
                    ]
                ),

            "delta_ADE_m_vs_full":
                (
                    metrics[
                        "planar_ADE_m"
                    ]
                    -
                    full_dev[
                        "planar_ADE_m"
                    ]
                ),
        }

    # ========================================================
    # F. Regression
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
        test_count != 108
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected final Stage4 regression "
            "108/108."
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

    if free_gib_after < MIN_FREE_GIB:
        raise RuntimeError(
            "250-GiB reserve violated."
        )

    report = {
        "stage":
            4,

        "block":
            "4.7",

        "status":
            "PASS",

        "feature_contract": {
            "R_t_indices":
                list(
                    EXPECTED_R_INDICES
                ),

            "R_t_names":
                list(
                    expected_names
                ),

            "R_t_used_by_full_model":
                True,
        },

        "measurement_uncertainty": {
            "scalar_definition":
                "sqrt(trace(latest_causal_target_R_t)/3)",

            "units":
                "m",

            "PSD_violation_count":
                0,

            "minimum_eigenvalue":
                measurement[
                    "minimum_eigenvalue"
                ],
        },

        "measurement_to_predictive_uncertainty":
            relation,

        "physical_R_t_sensitivity": {
            "variance_scaling":
                [
                    0.5,
                    1.0,
                    2.0,
                ],

            "scaling_applied_to":
                "all_observed_target_and_neighbor_history_R_t",

            "results":
                sensitivity,

            "model_retrained_for_sensitivity":
                False,
        },

        "retrained_input_context_ablations": {
            "protocol":
                "same_Gaussian_training_seed_sampler_optimizer_and_model_selection",

            "full_reference":
                comparison[
                    "full"
                ],

            "results":
                concise,

            "comparison":
                comparison,

            "performance_direction_is_closure_gate":
                False,
        },

        "uncertainty_separation": {
            "measurement_uncertainty":
                "Stage2_R_t_input",

            "predictive_uncertainty":
                "learned_Gaussian_Sigma_output",

            "Block45_calibration":
                "predictive_covariance_only",

            "separate":
                True,
        },

        "formal_validation": {
            "used":
                False,

            "formal_ablation_comparison":
                "not_required_before_Block4.8",
        },

        "regression": {
            "tests_passed":
                108,

            "tests_total":
                108,
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
    }

    write_json(
        REPORT,
        report,
    )

    if FAILURE.exists():
        FAILURE.unlink()

    marker = (
        "## Block 4.7 — "
        "Measurement/context uncertainty ablations"
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
                "- The exact six Cartesian H0 "
                  "measurement-covariance features "
                  "R_t are frozen at indices 6–11.\n"
                "- Measurement uncertainty is "
                  "separate from learned predictive "
                  "uncertainty.\n"
                "- Measurement uncertainty scalar "
                  "for analysis is the latest causal "
                  "target sqrt(trace(R_t)/3), in m.\n"
                "- Measurement-to-predictive "
                  "uncertainty correlations and "
                  "equal-count uncertainty bins are "
                  "measured on development only.\n"
                "- Physical R_t sensitivity evaluates "
                  "positive covariance variance scales "
                  "0.5/1/2 over the complete observed "
                  "target and neighbour histories.\n"
                "- no-R_t, actor-only and no-map "
                  "Gaussian predictors are retrained "
                  "using the same fit cache, fit-only "
                  "normalization, class-balanced "
                  "sampler, seed, optimizer and "
                  "development model-selection rule "
                  "as the full Gaussian.\n"
                "- no-R_t uses zero standardized "
                  "covariance features, equivalent "
                  "to fit-mean imputation and removal "
                  "of sample-specific R_t information.\n"
                "- actor-only removes all neighbour "
                  "information; no-map removes all "
                  "scene-varying map information.\n"
                "- Ablation performance direction is "
                  "reported, not post-hoc gated.\n"
                "- Formal N=120 validation remains "
                  "untouched until Block4.8.\n"
                f"- Full Gaussian ADE: "
                f"{comparison['full']['ADE_m']:.9f} m.\n"
                f"- no-R_t ADE: "
                f"{comparison[ABLATION_NO_R]['ADE_m']:.9f} m.\n"
                f"- actor-only ADE: "
                f"{comparison[ABLATION_ACTOR_ONLY]['ADE_m']:.9f} m.\n"
                f"- no-map ADE: "
                f"{comparison[ABLATION_NO_MAP]['ADE_m']:.9f} m.\n"
                "- Full Stage4 regression: "
                  "108/108 PASS.\n"
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
        "STAGE4 BLOCK 4.7 GATE"
    )
    print(
        "============================================================"
    )

    print(
        "R_t indices                 =",
        EXPECTED_R_INDICES,
    )
    print(
        "measurement scalar          = sqrt(trace(R_t)/3)"
    )
    print(
        "R_t PSD violations          = 0 PASS"
    )
    print(
        "measurement/predictive UQ   = SEPARATE"
    )

    print()
    print(
        "FULL GAUSSIAN REFERENCE"
    )
    print(
        "NLL                         =",
        round(
            comparison[
                "full"
            ][
                "NLL"
            ],
            6,
        ),
    )
    print(
        "ADE m                       =",
        round(
            comparison[
                "full"
            ][
                "ADE_m"
            ],
            6,
        ),
    )

    print()
    print(
        "RETRAINED ABLATIONS"
    )

    for variant in VARIANTS:
        item = comparison[
            variant
        ]

        print(
            variant,
            "| NLL=",
            round(
                item[
                    "NLL"
                ],
                6,
            ),
            "| ADE=",
            round(
                item[
                    "ADE_m"
                ],
                6,
            ),
            "| deltaADE=",
            round(
                item[
                    "delta_ADE_m_vs_full"
                ],
                6,
            ),
        )

    print()
    print(
        "R_t PHYSICAL SENSITIVITY @1.0s"
    )

    for scale in (
        "0.5",
        "1.0",
        "2.0",
    ):
        item = sensitivity[
            scale
        ]

        print(
            "R_t x",
            scale,
            "| pred sigma=",
            round(
                item[
                    "mean_predictive_sigma_RMS_m"
                ][
                    "1.0"
                ],
                6,
            ),
            "| ADE=",
            round(
                item[
                    "planar_ADE_m"
                ],
                6,
            ),
        )

    print()
    print(
        "MEASUREMENT→PREDICTIVE SPEARMAN"
    )

    for horizon in (
        "0.1",
        "0.3",
        "0.5",
        "1.0",
    ):
        value = relation[
            "correlations"
        ][
            horizon
        ][
            "Spearman_measurement_vs_predictive_sigma"
        ]

        print(
            horizon,
            "s =",
            round(
                value,
                6,
            ),
        )

    print()
    print(
        "fit/dev cache reuse         = PASS"
    )
    print(
        "Stage2/3 rescan             = NO"
    )
    print(
        "formal validation used      = NO"
    )
    print(
        "full Stage4 regression      = 108 / 108 PASS"
    )
    print(
        "free GiB                    =",
        round(
            free_gib_after,
            3,
        ),
    )
    print(
        "implementation files        =",
        implementation_files,
    )
    print(
        "implementation SHA256       =",
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
        "===== BLOCK 4.7 FINAL ====="
    )
    print(
        "measurement covariance R_t  = ANALYZED"
    )
    print(
        "measurement→predictive UQ   = MEASURED"
    )
    print(
        "physical R_t sensitivity    = MEASURED"
    )
    print(
        "no-R_t retraining           = PASS"
    )
    print(
        "actor-only retraining       = PASS"
    )
    print(
        "no-map retraining           = PASS"
    )
    print(
        "measurement/predictive UQ   = SEPARATE"
    )
    print(
        "formal validation leak      = NONE"
    )
    print(
        "regression                  = 108 / 108"
    )
    print(
        "implementation SHA          =",
        implementation_sha,
    )
    print(
        "log closure                 = PASS"
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
            "Check nvidia-smi first. "
            "Each ablation has its own "
            "resumable epoch progress; "
            "do not regenerate caches."
        )

    if (
        "progress"
        in text
        or
        "fingerprint"
        in text
    ):
        return (
            "Rerun the same command. "
            "Invalid Block4.7 progress is "
            "archived automatically; frozen "
            "fit/dev caches are reused."
        )

    if (
        "r_t"
        in text
        or
        "covariance"
        in text
        or
        "psd"
        in text
    ):
        return (
            "Do not guess or remap R_t. "
            "Inspect only the frozen six-feature "
            "Block4.7 covariance contract and "
            "new uncertainty-analysis code."
        )

    if (
        "108/108"
        in text
        or
        "regression"
        in text
    ):
        return (
            "Inspect the failing test. "
            "Do not change frozen Blocks4.0–4.6."
        )

    return (
        "Inspect reports/block47_failure.json. "
        "Completed ablation epochs are resumable "
        "and no WOMD/Stage2/Stage3 rescan is needed."
    )


try:
    main()

except BaseException as exc:
    payload = {
        "stage":
            4,

        "block":
            "4.7_part_2",

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

        "ablation_progress_resumable":
            True,

        "Stage2_Stage3_rescan":
            False,

        "Blocks43_to_46_modified":
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
        "BLOCK 4.7 PART 2/2 = BLOCKED"
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
        "ablation progress         = RESUMABLE"
    )
    print(
        "Stage2/3 rescan           = NO"
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

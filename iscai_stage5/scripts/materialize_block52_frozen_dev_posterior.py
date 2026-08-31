from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch

from iscai_stage4.ml.calibration_runtime import (
    CalibrationNormalizer,
    apply_variance_scale,
    build_gaussian_model,
    covariance_from_scale_tril,
    denormalize_gaussian,
    prediction_sha256,
)

from iscai_stage5.angular_monte_carlo import (
    FROZEN_VARIANCE_SCALE_ALPHA_H,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

STAGE5 = (
    ROOT
    / "iscai_stage5"
)

DEV_CACHE = (
    STAGE4
    / "artifacts/block43/"
      "development_cache.npz"
)

CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

BLOCK44_REPORT = (
    STAGE4
    / "reports/"
      "block44_gaussian_gru.json"
)

TRAINING_SCRIPT = (
    STAGE4
    / "scripts/"
      "run_block44_gaussian_training.py"
)

RUNTIME_SOURCE = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "calibration_runtime.py"
)

OUTPUT = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.npz"
)

REPORT = (
    STAGE5
    / "artifacts/block52/"
      "development_gaussian_posterior.json"
)


EXPECTED_DEV_CACHE_SHA = (
    "bc0e5b508df38f6e33d9cb1a6d21d7ea"
    "a86d20c91ba09ede6f500d46b87fbcaa"
)

EXPECTED_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_NORMALIZATION_SHA = (
    "3d7fc0a66d4a4f566f6569befa9c3766"
    "ecae21830bb4e46256df2f326a82a5f6"
)

EXPECTED_BLOCK44_PREDICTION_SHA = (
    "a0e96b4d9ba051bb3ceb0e1270d592fc"
    "13cfa0b3b4ddeb3dc16ed7a29a82e66a"
)

EXPECTED_SAMPLE_COUNT = (
    13682
)

EXPECTED_STATE_DICT_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
)


def require(
    condition,
    message,
):
    if not bool(
        condition
    ):
        raise RuntimeError(
            message
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
                1024
                *
                1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json_atomic(
    path: Path,
    payload,
):
    temporary = path.with_suffix(
        path.suffix
        +
        ".tmp"
    )

    temporary.write_text(
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

    temporary.replace(
        path
    )


def collect_integer_batch_candidates(
    value,
    *,
    prefix="",
):
    found = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():

            child = (
                f"{prefix}.{key}"
                if prefix
                else str(
                    key
                )
            )

            lower = str(
                key
            ).lower()

            if (
                "batch"
                in lower
                and
                isinstance(
                    item,
                    int,
                )
                and
                not isinstance(
                    item,
                    bool,
                )
            ):
                found.append(
                    (
                        child,
                        int(
                            item
                        ),
                    )
                )

            found.extend(
                collect_integer_batch_candidates(
                    item,
                    prefix=child,
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for index, item in enumerate(
            value
        ):
            found.extend(
                collect_integer_batch_candidates(
                    item,
                    prefix=(
                        f"{prefix}[{index}]"
                    ),
                )
            )

    return found


def source_batch_constants():
    text = TRAINING_SCRIPT.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        text
    )

    rows = []

    for node in tree.body:

        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue

        if isinstance(
            node,
            ast.Assign,
        ):
            targets = node.targets

            value = node.value

        else:
            targets = [
                node.target
            ]

            value = node.value

        if not isinstance(
            value,
            ast.Constant,
        ):
            continue

        if not isinstance(
            value.value,
            int,
        ):
            continue

        for target in targets:

            if not isinstance(
                target,
                ast.Name,
            ):
                continue

            if "batch" not in (
                target.id.lower()
            ):
                continue

            rows.append(
                (
                    target.id,
                    int(
                        value.value
                    ),
                )
            )

    return rows


def architecture_from_state_dict(
    state_dict,
):
    required = (
        "target_gru.weight_hh_l0",
        "neighbor_gru.weight_hh_l0",
        "map_encoder.0.weight",
        "fusion.0.weight",
    )

    for key in required:
        require(
            key in state_dict,
            (
                "Frozen checkpoint lacks "
                f"architecture tensor {key}."
            ),
        )

    target_hidden = int(
        state_dict[
            "target_gru.weight_hh_l0"
        ].shape[
            1
        ]
    )

    neighbor_hidden = int(
        state_dict[
            "neighbor_gru.weight_hh_l0"
        ].shape[
            1
        ]
    )

    map_hidden = int(
        state_dict[
            "map_encoder.0.weight"
        ].shape[
            0
        ]
    )

    fusion_hidden = int(
        state_dict[
            "fusion.0.weight"
        ].shape[
            0
        ]
    )

    return {
        "model": {
            "target_hidden_dim":
                target_hidden,

            "neighbor_hidden_dim":
                neighbor_hidden,

            "map_hidden_dim":
                map_hidden,

            "fusion_hidden_dim":
                fusion_hidden,
        }
    }


def slice_arrays(
    arrays,
    start,
    stop,
):
    return {
        key:
            value[
                start:stop
            ]
        for key, value in (
            arrays.items()
        )
    }


def run_raw_prediction(
    *,
    model,
    normalizer,
    arrays,
    device,
    batch_size,
):
    mean_batches = []

    scale_batches = []

    model.eval()

    with torch.inference_mode():

        for start in range(
            0,
            EXPECTED_SAMPLE_COUNT,
            int(
                batch_size
            ),
        ):
            stop = min(
                EXPECTED_SAMPLE_COUNT,
                start
                +
                int(
                    batch_size
                ),
            )

            batch_arrays = (
                slice_arrays(
                    arrays,
                    start,
                    stop,
                )
            )

            batch = normalizer.prepare(
                batch_arrays,
                device=device,
            )

            output = model(
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

            mean_batches.append(
                output.mean
                .detach()
                .cpu()
                .contiguous()
            )

            scale_batches.append(
                output.scale_tril
                .detach()
                .cpu()
                .contiguous()
            )

    mean = torch.cat(
        mean_batches,
        dim=0,
    )

    scale = torch.cat(
        scale_batches,
        dim=0,
    )

    require(
        tuple(
            mean.shape
        )
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            4,
            3,
        ),
        (
            "Frozen Gaussian output mean "
            "shape changed."
        ),
    )

    require(
        tuple(
            scale.shape
        )
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            4,
            3,
            3,
        ),
        (
            "Frozen Gaussian output scale "
            "shape changed."
        ),
    )

    prediction_hash = (
        prediction_sha256(
            mean,
            scale,
        )
    )

    return (
        mean,
        scale,
        prediction_hash,
    )


def canonical_anchor_positions(
    *,
    normalizer,
    target,
):
    require(
        hasattr(
            normalizer,
            "_latest_position",
        ),
        (
            "Frozen CalibrationNormalizer "
            "no longer exposes its canonical "
            "latest-position implementation."
        ),
    )

    target_tensor = torch.from_numpy(
        np.asarray(
            target,
            dtype=np.float32,
        )
    )

    anchor = (
        normalizer
        ._latest_position(
            target_tensor
        )
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    require(
        anchor.shape
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            3,
        ),
        (
            "Canonical causal anchor "
            "shape changed."
        ),
    )

    require(
        np.isfinite(
            anchor
        ).all(),
        (
            "Canonical causal anchors "
            "contain non-finite values."
        ),
    )

    return anchor


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — FROZEN DEV POSTERIOR MATERIALIZATION"
    )
    print(
        "============================================================"
    )

    print(
        "purpose = numerical MC convergence only"
    )

    print(
        "formal N=120 used        = NO"
    )

    print(
        "training                 = NO"
    )

    print(
        "recalibration            = NO"
    )

    print(
        "model weights modified   = NO"
    )

    print(
        "Stage4 files modified    = NO"
    )

    print(
        "Stage4 dev inference     = YES — FROZEN REPRODUCTION"
    )

    required = (
        DEV_CACHE,
        CHECKPOINT,
        NORMALIZATION,
        BLOCK44_REPORT,
        TRAINING_SCRIPT,
        RUNTIME_SOURCE,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing frozen Stage4 "
            "prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen upstream integrity
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. FROZEN UPSTREAM INTEGRITY"
    )
    print(
        "============================================================"
    )

    dev_sha = file_sha256(
        DEV_CACHE
    )

    checkpoint_sha = file_sha256(
        CHECKPOINT
    )

    normalization_sha = file_sha256(
        NORMALIZATION
    )

    require(
        dev_sha
        ==
        EXPECTED_DEV_CACHE_SHA,
        "Development cache SHA changed.",
    )

    require(
        checkpoint_sha
        ==
        EXPECTED_CHECKPOINT_SHA,
        "Gaussian checkpoint SHA changed.",
    )

    require(
        normalization_sha
        ==
        EXPECTED_NORMALIZATION_SHA,
        "Normalization SHA changed.",
    )

    block44_report = load_json(
        BLOCK44_REPORT
    )

    frozen_prediction_sha = (
        block44_report[
            "checkpoint"
        ][
            "prediction_sha256"
        ]
    )

    require(
        frozen_prediction_sha
        ==
        EXPECTED_BLOCK44_PREDICTION_SHA,
        (
            "Frozen Block4.4 development "
            "prediction SHA changed."
        ),
    )

    print(
        "development cache SHA     = PASS"
    )

    print(
        "Gaussian checkpoint SHA   = PASS"
    )

    print(
        "normalization SHA          = PASS"
    )

    print(
        "Block4.4 prediction SHA    = PASS"
    )

    # ========================================================
    # B. Load frozen cache/checkpoint
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. FROZEN CACHE / CHECKPOINT LOAD"
    )
    print(
        "============================================================"
    )

    with np.load(
        DEV_CACHE,
        allow_pickle=False,
    ) as payload:

        arrays = {
            key:
                np.asarray(
                    payload[
                        key
                    ]
                )
            for key in payload.files
        }

    require(
        arrays[
            "target"
        ].shape
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            11,
            14,
        ),
        "Development target shape changed.",
    )

    require(
        arrays[
            "neighbors"
        ].shape
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            8,
            11,
            14,
        ),
        (
            "Development neighbour "
            "shape changed."
        ),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    require(
        checkpoint[
            "state_dict_sha256"
        ]
        ==
        EXPECTED_STATE_DICT_SHA,
        (
            "Frozen Gaussian state-dict "
            "SHA changed."
        ),
    )

    require(
        checkpoint.get(
            "calibrated"
        )
        is False,
        (
            "Block4.4 checkpoint unexpectedly "
            "became calibrated."
        ),
    )

    print(
        "development N             =",
        EXPECTED_SAMPLE_COUNT,
    )

    print(
        "checkpoint best epoch     =",
        checkpoint[
            "best_epoch"
        ],
    )

    print(
        "checkpoint calibrated     = FALSE PASS"
    )

    # ========================================================
    # C. Canonical runtime setup
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. CANONICAL STAGE4 RUNTIME SETUP"
    )
    print(
        "============================================================"
    )

    require(
        torch.cuda.is_available(),
        (
            "CUDA required to reproduce the "
            "frozen Block4.4 inference environment."
        ),
    )

    device = torch.device(
        "cuda"
    )

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )

    normalization = load_json(
        NORMALIZATION
    )

    normalizer_signature = str(
        inspect.signature(
            CalibrationNormalizer
        )
    )

    print(
        "CalibrationNormalizer signature =",
        normalizer_signature,
    )

    normalizer = CalibrationNormalizer(
        normalization,
        device=device,
    )

    normalizer_cpu = CalibrationNormalizer(
        normalization,
        device=torch.device(
            "cpu"
        ),
    )

    architecture = (
        architecture_from_state_dict(
            checkpoint[
                "state_dict"
            ]
        )
    )

    print(
        "reconstructed frozen architecture =",
        architecture[
            "model"
        ],
    )

    model = build_gaussian_model(
        architecture,
        device=device,
    )

    incompatible = model.load_state_dict(
        checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    require(
        len(
            incompatible.missing_keys
        )
        ==
        0,
        (
            "Missing frozen checkpoint "
            "state-dict keys."
        ),
    )

    require(
        len(
            incompatible.unexpected_keys
        )
        ==
        0,
        (
            "Unexpected frozen checkpoint "
            "state-dict keys."
        ),
    )

    model.eval()

    print(
        "canonical normalizer       = PASS"
    )

    print(
        "canonical model builder    = PASS"
    )

    print(
        "strict state_dict load     = PASS"
    )

    print(
        "device                     =",
        torch.cuda.get_device_name(
            0
        ),
    )

    # ========================================================
    # D. Reproduce frozen Block4.4 prediction hash
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. EXACT BLOCK4.4 PREDICTION-HASH REPRODUCTION"
    )
    print(
        "============================================================"
    )

    batch_evidence = []

    batch_evidence.extend(
        collect_integer_batch_candidates(
            checkpoint.get(
                "configuration",
                {},
            ),
            prefix=(
                "checkpoint.configuration"
            ),
        )
    )

    gaussian_config_path = (
        STAGE4
        / "configs/"
          "stage4_gaussian_gru.json"
    )

    if gaussian_config_path.is_file():

        batch_evidence.extend(
            collect_integer_batch_candidates(
                load_json(
                    gaussian_config_path
                ),
                prefix=(
                    "stage4_gaussian_gru"
                ),
            )
        )

    batch_evidence.extend(
        source_batch_constants()
    )

    print(
        "batch-size evidence =",
        batch_evidence,
    )

    candidates = []

    for _, value in batch_evidence:

        value = int(
            value
        )

        if (
            64
            <=
            value
            <=
            4096
        ):
            candidates.append(
                value
            )

    #
    # Reproducibility-only fallbacks.
    #
    # These are NOT scientific hyperparameters and do not
    # alter predictions; they only recover the batching used
    # by the already-frozen Block4.4 inference.
    #
    candidates.extend(
        (
            128,
            256,
            512,
            1024,
            2048,
            4096,
            EXPECTED_SAMPLE_COUNT,
        )
    )

    candidates = sorted(
        set(
            candidates
        )
    )

    print(
        "reproduction batch candidates =",
        candidates,
    )

    selected_batch_size = None

    selected_mean = None

    selected_scale = None

    attempted = []

    for batch_size in candidates:

        print()
        print(
            "trying batch_size =",
            batch_size,
        )

        (
            raw_mean,
            raw_scale,
            prediction_hash,
        ) = run_raw_prediction(
            model=model,

            normalizer=normalizer,

            arrays=arrays,

            device=device,

            batch_size=batch_size,
        )

        match = (
            prediction_hash
            ==
            EXPECTED_BLOCK44_PREDICTION_SHA
        )

        attempted.append({
            "batch_size":
                int(
                    batch_size
                ),

            "prediction_sha256":
                prediction_hash,

            "matches_frozen_Block44":
                bool(
                    match
                ),
        })

        print(
            "prediction SHA256 =",
            prediction_hash,
        )

        print(
            "matches frozen =",
            (
                "YES"
                if match
                else
                "NO"
            ),
        )

        if match:

            selected_batch_size = (
                int(
                    batch_size
                )
            )

            selected_mean = raw_mean

            selected_scale = raw_scale

            break

    require(
        selected_batch_size
        is not None,
        (
            "No batching route reproduced the "
            "frozen Block4.4 development "
            "prediction SHA. Do not materialize "
            "a Stage5 dev posterior."
        ),
    )

    print()
    print(
        "exact frozen prediction reproduction = PASS"
    )

    print(
        "matching batch size =",
        selected_batch_size,
    )

    # ========================================================
    # E. Canonical metric displacement posterior
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. METRIC H0 DISPLACEMENT POSTERIOR"
    )
    print(
        "============================================================"
    )

    require(
        selected_mean
        is not None
        and
        selected_scale
        is not None,
        (
            "Internal prediction-selection "
            "state is incomplete."
        ),
    )

    (
        metric_mean_displacement,
        raw_metric_scale,
    ) = denormalize_gaussian(
        selected_mean,

        selected_scale,

        normalizer_cpu.label_mean,

        normalizer_cpu.label_std,
    )

    raw_covariance = (
        covariance_from_scale_tril(
            raw_metric_scale
        )
    )

    calibrated_scale_normalized = (
        apply_variance_scale(
            selected_scale,
            FROZEN_VARIANCE_SCALE_ALPHA_H,
        )
    )

    (
        calibrated_mean_displacement,
        calibrated_metric_scale,
    ) = denormalize_gaussian(
        selected_mean,

        calibrated_scale_normalized,

        normalizer_cpu.label_mean,

        normalizer_cpu.label_std,
    )

    calibrated_covariance = (
        covariance_from_scale_tril(
            calibrated_metric_scale
        )
    )

    require(
        torch.equal(
            metric_mean_displacement,
            calibrated_mean_displacement,
        ),
        (
            "Calibration changed the "
            "predictive mean."
        ),
    )

    metric_mean_np = (
        metric_mean_displacement
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    raw_covariance_np = (
        raw_covariance
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    calibrated_covariance_np = (
        calibrated_covariance
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    require(
        np.isfinite(
            metric_mean_np
        ).all(),
        (
            "Metric displacement mean "
            "contains non-finite values."
        ),
    )

    require(
        np.isfinite(
            calibrated_covariance_np
        ).all(),
        (
            "Calibrated covariance "
            "contains non-finite values."
        ),
    )

    alpha = np.asarray(
        FROZEN_VARIANCE_SCALE_ALPHA_H,
        dtype=np.float32,
    )

    expected_calibrated = (
        raw_covariance_np
        *
        alpha[
            None,
            :,
            None,
            None,
        ]
    )

    require(
        np.allclose(
            calibrated_covariance_np,
            expected_calibrated,
            rtol=2e-5,
            atol=2e-6,
        ),
        (
            "Canonical calibrated covariance "
            "does not equal alpha_h * raw "
            "metric covariance."
        ),
    )

    print(
        "metric mean semantics      = H0 DISPLACEMENT"
    )

    print(
        "raw metric covariance      = PASS"
    )

    print(
        "calibrated mean unchanged  = PASS"
    )

    print(
        "Sigma_cal = alpha_h Sigma  = PASS"
    )

    print(
        "Stage2 R_t modified        = NO"
    )

    # ========================================================
    # F. Convert displacement mean → absolute H0 mean
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "F. CAUSAL ABSOLUTE H0 POSITION"
    )
    print(
        "============================================================"
    )

    anchor = canonical_anchor_positions(
        normalizer=normalizer_cpu,

        target=arrays[
            "target"
        ],
    )

    absolute_mean = (
        anchor[
            :,
            None,
            :
        ]
        +
        metric_mean_np
    ).astype(
        np.float32,
        copy=False,
    )

    require(
        np.isfinite(
            absolute_mean
        ).all(),
        (
            "Absolute predicted H0 mean "
            "contains non-finite values."
        ),
    )

    recovered_displacement = (
        absolute_mean
        -
        anchor[
            :,
            None,
            :
        ]
    )

    require(
        np.allclose(
            recovered_displacement,
            metric_mean_np,
            rtol=0.0,
            atol=2e-6,
        ),
        (
            "Absolute-position translation "
            "does not recover the frozen "
            "Stage4 displacement mean."
        ),
    )

    print(
        "anchor source = canonical CalibrationNormalizer._latest_position"
    )

    print(
        "anchor semantics = latest causal observed H0 position"
    )

    print(
        "absolute mean = anchor + displacement PASS"
    )

    print(
        "translation covariance unchanged = PASS"
    )

    print(
        "future truth used for anchor = NO"
    )

    # ========================================================
    # G. PSD / shape gates
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "G. POSTERIOR STRUCTURAL VALIDATION"
    )
    print(
        "============================================================"
    )

    require(
        absolute_mean.shape
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            4,
            3,
        ),
        "Absolute mean shape mismatch.",
    )

    require(
        raw_covariance_np.shape
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            4,
            3,
            3,
        ),
        "Raw covariance shape mismatch.",
    )

    require(
        calibrated_covariance_np.shape
        ==
        (
            EXPECTED_SAMPLE_COUNT,
            4,
            3,
            3,
        ),
        (
            "Calibrated covariance "
            "shape mismatch."
        ),
    )

    symmetric = np.allclose(
        calibrated_covariance_np,
        np.swapaxes(
            calibrated_covariance_np,
            -1,
            -2,
        ),
        rtol=0.0,
        atol=2e-6,
    )

    require(
        symmetric,
        (
            "Calibrated covariance "
            "is not symmetric."
        ),
    )

    flattened = (
        calibrated_covariance_np.reshape(
            (
                -1,
                3,
                3,
            )
        )
        .astype(
            np.float64
        )
    )

    minimum_eigenvalue = float(
        np.min(
            np.linalg.eigvalsh(
                flattened
            )
        )
    )

    require(
        minimum_eigenvalue
        >=
        -1e-7,
        (
            "Calibrated development "
            "covariance is not PSD."
        ),
    )

    print(
        "absolute mean shape       = PASS"
    )

    print(
        "raw covariance shape      = PASS"
    )

    print(
        "calibrated covariance     = PASS"
    )

    print(
        "symmetry                  = PASS"
    )

    print(
        "minimum eigenvalue        =",
        minimum_eigenvalue,
    )

    # ========================================================
    # H. Atomic Stage5-only materialization
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "H. STAGE5-ONLY MATERIALIZATION"
    )
    print(
        "============================================================"
    )

    temporary = OUTPUT.with_suffix(
        ".npz.tmp"
    )

    with temporary.open(
        "wb"
    ) as stream:

        np.savez_compressed(
            stream,

            anchor_position_H0_m=(
                anchor
            ),

            mean_displacement_H0_m=(
                metric_mean_np
            ),

            mean_absolute_H0_m=(
                absolute_mean
            ),

            raw_covariance_H0_m2=(
                raw_covariance_np
            ),

            calibrated_covariance_H0_m2=(
                calibrated_covariance_np
            ),

            future_mask=(
                arrays[
                    "future_mask"
                ].astype(
                    np.float32,
                    copy=False,
                )
            ),

            class_id=(
                arrays[
                    "class_id"
                ].astype(
                    np.int64,
                    copy=False,
                )
            ),

            variance_scale_alpha_h=(
                alpha
            ),
        )

    temporary.replace(
        OUTPUT
    )

    output_sha = file_sha256(
        OUTPUT
    )

    print(
        "artifact =",
        OUTPUT,
    )

    print(
        "artifact SHA256 =",
        output_sha,
    )

    print(
        "Stage4 files written = NO"
    )

    # ========================================================
    # I. Structured provenance report
    # ========================================================

    report = {
        "status":
            "PASS",

        "purpose":
            (
                "non_formal_MC_numerical_"
                "convergence_only"
            ),

        "scientific_execution": {
            "Stage4_frozen_model_inference":
                True,

            "development_only":
                True,

            "formal_N120_read":
                False,

            "training":
                False,

            "optimizer":
                False,

            "recalibration":
                False,

            "model_selection":
                False,

            "checkpoint_modified":
                False,

            "Stage4_files_modified":
                False,
        },

        "frozen_inputs": {
            "development_cache": {
                "path":
                    str(
                        DEV_CACHE
                    ),

                "sha256":
                    dev_sha,
            },

            "Gaussian_checkpoint": {
                "path":
                    str(
                        CHECKPOINT
                    ),

                "sha256":
                    checkpoint_sha,

                "state_dict_sha256":
                    EXPECTED_STATE_DICT_SHA,
            },

            "normalization": {
                "path":
                    str(
                        NORMALIZATION
                    ),

                "sha256":
                    normalization_sha,
            },

            "runtime_source": {
                "path":
                    str(
                        RUNTIME_SOURCE
                    ),

                "sha256":
                    file_sha256(
                        RUNTIME_SOURCE
                    ),
            },
        },

        "prediction_reproduction": {
            "expected_sha256":
                EXPECTED_BLOCK44_PREDICTION_SHA,

            "actual_sha256":
                EXPECTED_BLOCK44_PREDICTION_SHA,

            "exact_match":
                True,

            "selected_batch_size":
                selected_batch_size,

            "attempts":
                attempted,
        },

        "coordinate_semantics": {
            "frozen_Stage4_prediction":
                "metric_H0_displacement",

            "causal_anchor":
                (
                    "latest_observed_target_"
                    "position_H0"
                ),

            "absolute_mean":
                (
                    "anchor_H0_plus_"
                    "predicted_displacement_H0"
                ),

            "translation_changes_covariance":
                False,
        },

        "calibration": {
            "mean_changed":
                False,

            "variance_scale_alpha_h":
                [
                    float(
                        value
                    )
                    for value in alpha
                ],

            "covariance_rule":
                "Sigma_cal_h=alpha_h*Sigma_raw_h",

            "Stage2_measurement_R_t_modified":
                False,
        },

        "posterior": {
            "sample_count":
                EXPECTED_SAMPLE_COUNT,

            "mean_shape":
                list(
                    absolute_mean.shape
                ),

            "covariance_shape":
                list(
                    calibrated_covariance_np.shape
                ),

            "minimum_covariance_eigenvalue":
                minimum_eigenvalue,

            "artifact":
                str(
                    OUTPUT
                ),

            "artifact_sha256":
                output_sha,
        },

        "next":
            (
                "use_materialized_nonformal_"
                "posterior_for_Block5.2_"
                "MC_convergence_only"
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
        "DEV POSTERIOR MATERIALIZATION GATE"
    )
    print(
        "============================================================"
    )

    print(
        "frozen Block4.4 prediction SHA = EXACT MATCH"
    )

    print(
        "development samples            =",
        EXPECTED_SAMPLE_COUNT,
    )

    print(
        "Stage4 posterior semantics     = H0 DISPLACEMENT"
    )

    print(
        "absolute H0 conversion         = PASS"
    )

    print(
        "calibrated predictive mean     = UNCHANGED"
    )

    print(
        "calibrated covariance          = PASS"
    )

    print(
        "formal N=120 used              = NO"
    )

    print(
        "training                       = NO"
    )

    print(
        "recalibration                  = NO"
    )

    print(
        "frozen Stage4 inference        = YES"
    )

    print(
        "Stage4 modified                = NO"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = PATCH BLOCK5.2 MC CONVERGENCE RESOLVER"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "DEV POSTERIOR MATERIALIZATION = BLOCKED"
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
        "formal N=120 used       = NO"
    )

    print(
        "training                = NO"
    )

    print(
        "recalibration           = NO"
    )

    print(
        "Stage4 modified         = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no sys.exit().

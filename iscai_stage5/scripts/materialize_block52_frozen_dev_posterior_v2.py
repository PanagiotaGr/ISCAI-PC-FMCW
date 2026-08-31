from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from iscai_stage4.ml import (
    covariance_from_scale_tril,
    denormalize_gaussian,
)

from iscai_stage4.ml.calibration_runtime import (
    apply_variance_scale,
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

TRAINING_SCRIPT = (
    STAGE4
    / "scripts/"
      "run_block44_gaussian_training.py"
)

DEV_CACHE = (
    STAGE4
    / "artifacts/block43/"
      "development_cache.npz"
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

GAUSSIAN_CONFIG = (
    STAGE4
    / "configs/"
      "stage4_gaussian_gru.json"
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


EXPECTED_BLOCK44_PREDICTION_SHA = (
    "a0e96b4d9ba051bb3ceb0e1270d592fc"
    "13cfa0b3b4ddeb3dc16ed7a29a82e66a"
)

EXPECTED_GAUSSIAN_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_GAUSSIAN_STATE_DICT_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
)

EXPECTED_DEVELOPMENT_N = (
    13682
)

EXACT_HASH_SAMPLE_LIMIT = (
    2048
)

EXACT_HASH_BATCH_SIZE = (
    512
)

FULL_MATERIALIZATION_BATCH_SIZE = (
    512
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
                1024 * 1024
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


def load_frozen_block44_module():
    """
    Import the frozen Block4.4 script under a non-main
    module name.

    Its __main__ execution path therefore does not run.
    We reuse its exact:
        GaussianNormalizer
        build_gaussian_model
        probabilistic_prediction_sha256
        latest_position_torch
        set_global_determinism
    """

    spec = (
        importlib.util
        .spec_from_file_location(
            "iscai_stage4_frozen_block44_route",
            TRAINING_SCRIPT,
        )
    )

    require(
        spec is not None
        and
        spec.loader is not None,
        (
            "Could not create import spec "
            "for frozen Block4.4 script."
        ),
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    required_symbols = (
        "GaussianNormalizer",
        "build_gaussian_model",
        "probabilistic_prediction_sha256",
        "latest_position_torch",
        "set_global_determinism",
    )

    missing = [
        symbol
        for symbol in required_symbols
        if not hasattr(
            module,
            symbol,
        )
    ]

    require(
        not missing,
        (
            "Frozen Block4.4 module lacks "
            "required symbol(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    return module


def full_prediction_sha256(
    mean_normalized,
    scale_tril_normalized,
):
    """
    Stage5 provenance hash for the full development posterior.

    This is NOT the historical Block4.4 prediction hash.
    The historical hash intentionally covers only its
    frozen 2048-sample reproducibility probe.
    """

    digest = sha256()

    for array in (
        mean_normalized,
        scale_tril_normalized,
    ):
        value = np.ascontiguousarray(
            array,
            dtype=np.float32,
        )

        digest.update(
            value.tobytes(
                order="C"
            )
        )

    return digest.hexdigest()


def posterior_artifact_content_sha256(
    *,
    anchor,
    mean_displacement,
    mean_absolute,
    raw_covariance,
    calibrated_covariance,
):
    digest = sha256()

    for name, array in (
        (
            "anchor_position_H0_m",
            anchor,
        ),
        (
            "mean_displacement_H0_m",
            mean_displacement,
        ),
        (
            "mean_absolute_H0_m",
            mean_absolute,
        ),
        (
            "raw_covariance_H0_m2",
            raw_covariance,
        ),
        (
            "calibrated_covariance_H0_m2",
            calibrated_covariance,
        ),
    ):
        value = np.ascontiguousarray(
            array
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                value.shape
            ).encode(
                "utf-8"
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


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.2 — CORRECTED FROZEN DEV POSTERIOR MATERIALIZATION"
    )
    print(
        "============================================================"
    )

    print(
        "formal N=120 used         = NO"
    )

    print(
        "development data only     = YES"
    )

    print(
        "frozen Stage4 inference   = YES"
    )

    print(
        "training                  = NO"
    )

    print(
        "optimizer                  = NO"
    )

    print(
        "recalibration             = NO"
    )

    print(
        "Stage4 modification       = NO"
    )

    required = (
        TRAINING_SCRIPT,
        DEV_CACHE,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        NORMALIZATION,
        BLOCK44_REPORT,
        GAUSSIAN_CONFIG,
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
            "Missing frozen Stage4 prerequisite(s): "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # A. Frozen integrity
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "A. FROZEN STAGE4 INTEGRITY"
    )
    print(
        "============================================================"
    )

    gaussian_checkpoint_sha = (
        file_sha256(
            GAUSSIAN_CHECKPOINT
        )
    )

    require(
        gaussian_checkpoint_sha
        ==
        EXPECTED_GAUSSIAN_CHECKPOINT_SHA,
        (
            "Frozen Gaussian checkpoint "
            "SHA mismatch."
        ),
    )

    gaussian_checkpoint = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    require(
        gaussian_checkpoint[
            "state_dict_sha256"
        ]
        ==
        EXPECTED_GAUSSIAN_STATE_DICT_SHA,
        (
            "Frozen Gaussian state_dict "
            "SHA mismatch."
        ),
    )

    fingerprints = (
        gaussian_checkpoint[
            "fingerprints"
        ]
    )

    development_cache_sha = (
        file_sha256(
            DEV_CACHE
        )
    )

    normalization_sha = (
        file_sha256(
            NORMALIZATION
        )
    )

    deterministic_checkpoint_sha = (
        file_sha256(
            DET_CHECKPOINT
        )
    )

    require(
        development_cache_sha
        ==
        fingerprints[
            "development_cache_sha256"
        ],
        (
            "Frozen development-cache "
            "fingerprint mismatch."
        ),
    )

    require(
        normalization_sha
        ==
        fingerprints[
            "normalization_sha256"
        ],
        (
            "Frozen normalization "
            "fingerprint mismatch."
        ),
    )

    require(
        deterministic_checkpoint_sha
        ==
        fingerprints[
            "deterministic_checkpoint_sha256"
        ],
        (
            "Frozen deterministic-checkpoint "
            "fingerprint mismatch."
        ),
    )

    frozen_report = load_json(
        BLOCK44_REPORT
    )

    require(
        frozen_report[
            "checkpoint"
        ][
            "prediction_sha256"
        ]
        ==
        EXPECTED_BLOCK44_PREDICTION_SHA,
        (
            "Frozen Block4.4 prediction SHA "
            "in report changed."
        ),
    )

    require(
        gaussian_checkpoint.get(
            "calibrated"
        )
        is False,
        (
            "Block4.4 checkpoint unexpectedly "
            "contains calibrated=True."
        ),
    )

    print(
        "Gaussian checkpoint       = PASS"
    )

    print(
        "Gaussian state_dict       = PASS"
    )

    print(
        "development cache         = PASS"
    )

    print(
        "fit normalization         = PASS"
    )

    print(
        "deterministic checkpoint  = PASS"
    )

    print(
        "frozen prediction SHA     = PASS"
    )

    # ========================================================
    # B. Exact frozen Block4.4 route
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "B. EXACT FROZEN BLOCK4.4 ROUTE"
    )
    print(
        "============================================================"
    )

    frozen = (
        load_frozen_block44_module()
    )

    gaussian_config = load_json(
        GAUSSIAN_CONFIG
    )

    seed = int(
        gaussian_config[
            "training"
        ][
            "seed"
        ]
    )

    require(
        seed
        ==
        20260821,
        (
            "Frozen Gaussian seed changed."
        ),
    )

    require(
        int(
            gaussian_config[
                "training"
            ][
                "batch_size"
            ]
        )
        ==
        EXACT_HASH_BATCH_SIZE,
        (
            "Frozen Gaussian training "
            "batch size changed."
        ),
    )

    frozen.set_global_determinism(
        seed
    )

    require(
        torch.cuda.is_available(),
        (
            "CUDA unavailable; exact frozen "
            "Block4.4 route requires cuda:0."
        ),
    )

    device = torch.device(
        "cuda:0"
    )

    deterministic_checkpoint = torch.load(
        DET_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    gaussian_checkpoint_device = torch.load(
        GAUSSIAN_CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    normalization = load_json(
        NORMALIZATION
    )

    normalizer = frozen.GaussianNormalizer(
        normalization,
        device=device,
    )

    model = frozen.build_gaussian_model(
        deterministic_checkpoint[
            "configuration"
        ],
        device=device,
    )

    incompatible = model.load_state_dict(
        gaussian_checkpoint_device[
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
            "Missing frozen Gaussian "
            "checkpoint keys."
        ),
    )

    require(
        len(
            incompatible.unexpected_keys
        )
        ==
        0,
        (
            "Unexpected frozen Gaussian "
            "checkpoint keys."
        ),
    )

    model.eval()

    print(
        "frozen GaussianNormalizer = PASS"
    )

    print(
        "frozen model builder      = PASS"
    )

    print(
        "strict state_dict load    = PASS"
    )

    print(
        "seed                      =",
        seed,
    )

    print(
        "device                    =",
        torch.cuda.get_device_name(
            0
        ),
    )

    # ========================================================
    # C. Frozen development arrays
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "C. FROZEN DEVELOPMENT CACHE"
    )
    print(
        "============================================================"
    )

    with np.load(
        DEV_CACHE,
        allow_pickle=False,
    ) as payload:

        dev_arrays = {
            key:
                np.asarray(
                    payload[
                        key
                    ]
                )
            for key in payload.files
        }

    development_n = int(
        dev_arrays[
            "target"
        ].shape[
            0
        ]
    )

    require(
        development_n
        ==
        EXPECTED_DEVELOPMENT_N,
        (
            "Frozen development sample "
            "count changed."
        ),
    )

    print(
        "development N           =",
        development_n,
    )

    print(
        "target shape            =",
        dev_arrays[
            "target"
        ].shape,
    )

    print(
        "neighbors shape         =",
        dev_arrays[
            "neighbors"
        ].shape,
    )

    print(
        "future truth used below = NO"
    )

    # ========================================================
    # D. Exact historical Block4.4 hash reproduction
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "D. EXACT HISTORICAL BLOCK4.4 PREDICTION HASH"
    )
    print(
        "============================================================"
    )

    reproduced_prediction_sha = (
        frozen
        .probabilistic_prediction_sha256(
            model,
            dev_arrays,
            normalizer,
            device=device,
        )
    )

    print(
        "frozen hash sample limit =",
        EXACT_HASH_SAMPLE_LIMIT,
    )

    print(
        "frozen hash batch size   =",
        EXACT_HASH_BATCH_SIZE,
    )

    print(
        "expected SHA256          =",
        EXPECTED_BLOCK44_PREDICTION_SHA,
    )

    print(
        "actual SHA256            =",
        reproduced_prediction_sha,
    )

    require(
        reproduced_prediction_sha
        ==
        EXPECTED_BLOCK44_PREDICTION_SHA,
        (
            "Exact frozen Block4.4 "
            "prediction SHA did not reproduce. "
            "Do not materialize posterior."
        ),
    )

    print(
        "exact frozen prediction reproduction = PASS"
    )

    # ========================================================
    # E. Full dev posterior using exact same preprocessing
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "E. FULL DEVELOPMENT POSTERIOR"
    )
    print(
        "============================================================"
    )

    mean_normalized_batches = []

    scale_normalized_batches = []

    metric_mean_batches = []

    raw_covariance_batches = []

    calibrated_covariance_batches = []

    anchor_batches = []

    alpha = torch.tensor(
        FROZEN_VARIANCE_SCALE_ALPHA_H,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    with torch.inference_mode():

        for start in range(
            0,
            development_n,
            FULL_MATERIALIZATION_BATCH_SIZE,
        ):
            stop = min(
                development_n,
                start
                +
                FULL_MATERIALIZATION_BATCH_SIZE,
            )

            indices = np.arange(
                start,
                stop,
                dtype=np.int64,
            )

            batch = normalizer.prepare(
                dev_arrays,
                indices,
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

            mean_normalized = (
                output.mean
            )

            scale_normalized = (
                output.scale_tril
            )

            (
                metric_mean_displacement,
                raw_metric_scale,
            ) = denormalize_gaussian(
                mean_normalized,
                scale_normalized,
                normalizer.label_mean,
                normalizer.label_std,
            )

            raw_covariance = (
                covariance_from_scale_tril(
                    raw_metric_scale
                )
            )

            calibrated_scale_normalized = (
                apply_variance_scale(
                    scale_normalized,
                    alpha,
                )
            )

            (
                calibrated_metric_mean,
                calibrated_metric_scale,
            ) = denormalize_gaussian(
                mean_normalized,
                calibrated_scale_normalized,
                normalizer.label_mean,
                normalizer.label_std,
            )

            require(
                torch.equal(
                    metric_mean_displacement,
                    calibrated_metric_mean,
                ),
                (
                    "Frozen calibration changed "
                    "predictive mean."
                ),
            )

            calibrated_covariance = (
                covariance_from_scale_tril(
                    calibrated_metric_scale
                )
            )

            #
            # Causal anchor only:
            # raw current/past target history.
            #
            raw_target = torch.from_numpy(
                np.asarray(
                    dev_arrays[
                        "target"
                    ][
                        indices
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            anchor = (
                frozen
                .latest_position_torch(
                    raw_target
                )
            )

            require(
                tuple(
                    anchor.shape
                )
                ==
                (
                    stop
                    -
                    start,
                    3,
                ),
                (
                    "Frozen latest causal "
                    "position shape changed."
                ),
            )

            mean_normalized_batches.append(
                mean_normalized
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            scale_normalized_batches.append(
                scale_normalized
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            metric_mean_batches.append(
                metric_mean_displacement
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            raw_covariance_batches.append(
                raw_covariance
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            calibrated_covariance_batches.append(
                calibrated_covariance
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

            anchor_batches.append(
                anchor
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

    mean_normalized_np = np.concatenate(
        mean_normalized_batches,
        axis=0,
    )

    scale_normalized_np = np.concatenate(
        scale_normalized_batches,
        axis=0,
    )

    mean_displacement_np = np.concatenate(
        metric_mean_batches,
        axis=0,
    )

    raw_covariance_np = np.concatenate(
        raw_covariance_batches,
        axis=0,
    )

    calibrated_covariance_np = np.concatenate(
        calibrated_covariance_batches,
        axis=0,
    )

    anchor_np = np.concatenate(
        anchor_batches,
        axis=0,
    )

    require(
        mean_displacement_np.shape
        ==
        (
            development_n,
            4,
            3,
        ),
        (
            "Metric displacement "
            "mean shape mismatch."
        ),
    )

    require(
        calibrated_covariance_np.shape
        ==
        (
            development_n,
            4,
            3,
            3,
        ),
        (
            "Calibrated covariance "
            "shape mismatch."
        ),
    )

    require(
        anchor_np.shape
        ==
        (
            development_n,
            3,
        ),
        (
            "Causal anchor shape mismatch."
        ),
    )

    #
    # Important: no future labels participate here.
    #
    mean_absolute_np = (
        anchor_np[
            :,
            None,
            :
        ]
        +
        mean_displacement_np
    ).astype(
        np.float32,
        copy=False,
    )

    require(
        np.isfinite(
            mean_absolute_np
        ).all(),
        (
            "Absolute H0 predictive mean "
            "contains non-finite values."
        ),
    )

    require(
        np.isfinite(
            calibrated_covariance_np
        ).all(),
        (
            "Calibrated predictive covariance "
            "contains non-finite values."
        ),
    )

    print(
        "full mean shape            =",
        mean_absolute_np.shape,
    )

    print(
        "full covariance shape      =",
        calibrated_covariance_np.shape,
    )

    print(
        "Stage4 output semantics    = H0 DISPLACEMENT"
    )

    print(
        "causal anchor              = LATEST OBSERVED H0"
    )

    print(
        "absolute H0 conversion     = PASS"
    )

    print(
        "future truth used          = NO"
    )

    # ========================================================
    # F. Calibration equivalence / PSD
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "F. FROZEN CALIBRATION / PSD VALIDATION"
    )
    print(
        "============================================================"
    )

    alpha_np = np.asarray(
        FROZEN_VARIANCE_SCALE_ALPHA_H,
        dtype=np.float32,
    )

    expected_calibrated_covariance = (
        raw_covariance_np
        *
        alpha_np[
            None,
            :,
            None,
            None,
        ]
    )

    maximum_calibration_error = float(
        np.max(
            np.abs(
                calibrated_covariance_np
                -
                expected_calibrated_covariance
            )
        )
    )

    require(
        np.allclose(
            calibrated_covariance_np,
            expected_calibrated_covariance,
            rtol=2e-5,
            atol=2e-6,
        ),
        (
            "Canonical Stage4 calibrated "
            "covariance is not alpha_h "
            "times raw metric covariance."
        ),
    )

    symmetry_error = float(
        np.max(
            np.abs(
                calibrated_covariance_np
                -
                np.swapaxes(
                    calibrated_covariance_np,
                    -1,
                    -2,
                )
            )
        )
    )

    require(
        symmetry_error
        <=
        2e-6,
        (
            "Calibrated covariance "
            "lost symmetry."
        ),
    )

    flattened = (
        calibrated_covariance_np
        .reshape(
            (
                -1,
                3,
                3,
            )
        )
        .astype(
            np.float64,
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
        "predictive mean unchanged = PASS"
    )

    print(
        "Sigma_cal=alpha_h*Sigma   = PASS"
    )

    print(
        "max calibration error     =",
        maximum_calibration_error,
    )

    print(
        "max symmetry error        =",
        symmetry_error,
    )

    print(
        "minimum eigenvalue        =",
        minimum_eigenvalue,
    )

    print(
        "Stage2 measurement R_t    = UNCHANGED / SEPARATE"
    )

    # ========================================================
    # G. Full-development provenance hashes
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "G. FULL DEVELOPMENT POSTERIOR HASHES"
    )
    print(
        "============================================================"
    )

    full_normalized_prediction_sha = (
        full_prediction_sha256(
            mean_normalized_np,
            scale_normalized_np,
        )
    )

    content_sha = (
        posterior_artifact_content_sha256(
            anchor=anchor_np,

            mean_displacement=(
                mean_displacement_np
            ),

            mean_absolute=(
                mean_absolute_np
            ),

            raw_covariance=(
                raw_covariance_np
            ),

            calibrated_covariance=(
                calibrated_covariance_np
            ),
        )
    )

    print(
        "historical first-2048 SHA =",
        reproduced_prediction_sha,
    )

    print(
        "full normalized posterior SHA =",
        full_normalized_prediction_sha,
    )

    print(
        "Stage5 posterior content SHA  =",
        content_sha,
    )

    # ========================================================
    # H. Stage5-only atomic artifact
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
                anchor_np
            ),

            mean_displacement_H0_m=(
                mean_displacement_np
            ),

            mean_absolute_H0_m=(
                mean_absolute_np
            ),

            raw_covariance_H0_m2=(
                raw_covariance_np
            ),

            calibrated_covariance_H0_m2=(
                calibrated_covariance_np
            ),

            variance_scale_alpha_h=(
                alpha_np
            ),
        )

    temporary.replace(
        OUTPUT
    )

    output_file_sha = file_sha256(
        OUTPUT
    )

    print(
        "artifact =",
        OUTPUT,
    )

    print(
        "artifact file SHA256 =",
        output_file_sha,
    )

    print(
        "Stage4 files written = NO"
    )

    # ========================================================
    # I. Structured report
    # ========================================================

    report_payload = {
        "status":
            "PASS",

        "purpose":
            (
                "Block5.2_nonformal_"
                "MC_convergence_only"
            ),

        "scientific_execution": {
            "development_only":
                True,

            "Stage4_frozen_model_inference":
                True,

            "formal_N120_read":
                False,

            "training":
                False,

            "optimizer":
                False,

            "model_selection":
                False,

            "recalibration":
                False,

            "Stage4_modified":
                False,
        },

        "historical_reproduction": {
            "function":
                (
                    "Block4.4."
                    "probabilistic_prediction_sha256"
                ),

            "sample_limit":
                EXACT_HASH_SAMPLE_LIMIT,

            "batch_size":
                EXACT_HASH_BATCH_SIZE,

            "expected_sha256":
                EXPECTED_BLOCK44_PREDICTION_SHA,

            "actual_sha256":
                reproduced_prediction_sha,

            "exact_match":
                True,
        },

        "frozen_inputs": {
            "training_script": {
                "path":
                    str(
                        TRAINING_SCRIPT
                    ),

                "sha256":
                    file_sha256(
                        TRAINING_SCRIPT
                    ),
            },

            "development_cache": {
                "path":
                    str(
                        DEV_CACHE
                    ),

                "sha256":
                    development_cache_sha,
            },

            "deterministic_checkpoint": {
                "path":
                    str(
                        DET_CHECKPOINT
                    ),

                "sha256":
                    deterministic_checkpoint_sha,
            },

            "Gaussian_checkpoint": {
                "path":
                    str(
                        GAUSSIAN_CHECKPOINT
                    ),

                "sha256":
                    gaussian_checkpoint_sha,

                "state_dict_sha256":
                    EXPECTED_GAUSSIAN_STATE_DICT_SHA,
            },

            "normalization": {
                "path":
                    str(
                        NORMALIZATION
                    ),

                "sha256":
                    normalization_sha,
            },
        },

        "coordinate_semantics": {
            "Stage4_prediction":
                "metric_H0_displacement",

            "causal_anchor":
                (
                    "latest_causal_observed_"
                    "target_position_H0"
                ),

            "absolute_mean_rule":
                (
                    "anchor_position_H0_m"
                    "+"
                    "mean_displacement_H0_m"
                ),

            "translation_changes_covariance":
                False,

            "future_truth_used":
                False,
        },

        "calibration": {
            "predictive_mean_changed":
                False,

            "variance_scale_alpha_h":
                [
                    float(
                        value
                    )
                    for value in (
                        FROZEN_VARIANCE_SCALE_ALPHA_H
                    )
                ],

            "rule":
                (
                    "Sigma_cal_h="
                    "alpha_h*Sigma_raw_h"
                ),

            "Stage2_measurement_R_t":
                "SEPARATE_UNCHANGED",

            "max_numeric_equivalence_error":
                maximum_calibration_error,
        },

        "posterior": {
            "development_N":
                development_n,

            "full_materialization_batch_size":
                FULL_MATERIALIZATION_BATCH_SIZE,

            "mean_shape":
                list(
                    mean_absolute_np.shape
                ),

            "covariance_shape":
                list(
                    calibrated_covariance_np.shape
                ),

            "minimum_covariance_eigenvalue":
                minimum_eigenvalue,

            "full_normalized_prediction_sha256":
                full_normalized_prediction_sha,

            "content_sha256":
                content_sha,

            "artifact":
                str(
                    OUTPUT
                ),

            "artifact_file_sha256":
                output_file_sha,
        },

        "next":
            (
                "patch_Block5.2_Part2_"
                "convergence_resolver_to_"
                "this_Stage5_only_artifact"
            ),
    }

    write_json_atomic(
        REPORT,
        report_payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "CORRECTED DEV POSTERIOR MATERIALIZATION GATE"
    )
    print(
        "============================================================"
    )

    print(
        "exact frozen Block4.4 SHA     = PASS"
    )

    print(
        "historical hash sample limit  = 2048"
    )

    print(
        "historical hash batch size    = 512"
    )

    print(
        "development posterior N       =",
        development_n,
    )

    print(
        "Stage4 posterior              = H0 DISPLACEMENT"
    )

    print(
        "causal absolute H0 conversion = PASS"
    )

    print(
        "calibrated predictive mean    = UNCHANGED"
    )

    print(
        "calibrated covariance         = PASS"
    )

    print(
        "future truth used             = NO"
    )

    print(
        "formal N=120 used             = NO"
    )

    print(
        "training/recalibration        = NO"
    )

    print(
        "Stage4 frozen inference       = YES"
    )

    print(
        "Stage4 modified               = NO"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "next = BLOCK5.2 PART2 CONVERGENCE RESOLVER PATCH"
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
        "CORRECTED DEV POSTERIOR MATERIALIZATION = BLOCKED"
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
        "training/recalibration  = NO"
    )

    print(
        "Stage4 modified         = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no sys.exit().

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

from iscai_stage4.ml import (
    DeterministicTrajectoryGRU,
    GaussianTrajectoryGRU,
    covariance_from_scale_tril,
    denormalize_gaussian,
    empirical_coverage,
    initialize_from_deterministic,
    masked_gaussian_nll,
    set_global_determinism,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

BLOCK43 = (
    STAGE4
    / "reports/"
      "block43_deterministic_gru.json"
)

DET_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_gru.pt"
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

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/"
      "fit_normalization.json"
)

GAUSSIAN_CONFIG = (
    STAGE4
    / "configs/"
      "stage4_gaussian_gru.json"
)

REPORT = (
    STAGE4
    / "reports/"
      "block44_preflight.json"
)

FAILURE = (
    STAGE4
    / "reports/"
      "block44_preflight_failure.json"
)

EXPECTED_BLOCK43_IMPL_SHA = (
    "7f42e0cf534ed3ffb8d69e7230f5e04e"
    "ecde8639428911017af771c7b2993def"
)

EXPECTED_CHECKPOINT_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_STATE_SHA = (
    "d2ffbc03c7cb2826fef2175c95f48725"
    "707ec6d791eaeffbd6f59bc507b8595a"
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


def state_dict_sha256(
    state_dict,
) -> str:
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


def load_cache_shapes(
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        required = (
            "target",
            "neighbors",
            "neighbor_mask",
            "map_context",
            "future",
            "future_mask",
            "class_id",
        )

        for key in required:
            if key not in data:
                raise RuntimeError(
                    f"Cache lacks {key}: "
                    f"{path}"
                )

        return {
            key:
                list(
                    data[
                        key
                    ].shape
                )
            for key in required
        }


def make_deterministic_model(
    configuration,
    *,
    device,
):
    model_config = (
        configuration[
            "model"
        ]
    )

    return (
        DeterministicTrajectoryGRU(
            target_hidden_dim=int(
                model_config[
                    "target_hidden_dim"
                ]
            ),
            neighbor_hidden_dim=int(
                model_config[
                    "neighbor_hidden_dim"
                ]
            ),
            map_hidden_dim=int(
                model_config[
                    "map_hidden_dim"
                ]
            ),
            fusion_hidden_dim=int(
                model_config[
                    "fusion_hidden_dim"
                ]
            ),
            use_neighbors=True,
            use_map=True,
        )
        .to(
            device
        )
    )


def make_gaussian_model(
    configuration,
    *,
    device,
):
    model_config = (
        configuration[
            "model"
        ]
    )

    return (
        GaussianTrajectoryGRU(
            target_hidden_dim=int(
                model_config[
                    "target_hidden_dim"
                ]
            ),
            neighbor_hidden_dim=int(
                model_config[
                    "neighbor_hidden_dim"
                ]
            ),
            map_hidden_dim=int(
                model_config[
                    "map_hidden_dim"
                ]
            ),
            fusion_hidden_dim=int(
                model_config[
                    "fusion_hidden_dim"
                ]
            ),
            use_neighbors=True,
            use_map=True,
        )
        .to(
            device
        )
    )


def main():
    if not BLOCK43.is_file():
        raise RuntimeError(
            "Block4.3 report missing."
        )

    block43 = json.loads(
        BLOCK43.read_text(
            encoding="utf-8"
        )
    )

    if block43[
        "status"
    ] != "PASS":
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
            "implementation SHA changed."
        )

    if not DET_CHECKPOINT.is_file():
        raise RuntimeError(
            "Frozen deterministic "
            "checkpoint missing."
        )

    if (
        file_sha256(
            DET_CHECKPOINT
        )
        !=
        EXPECTED_CHECKPOINT_SHA
    ):
        raise RuntimeError(
            "Frozen deterministic "
            "checkpoint SHA changed."
        )

    if not FIT_CACHE.is_file():
        raise RuntimeError(
            "Frozen fit cache missing."
        )

    if not DEV_CACHE.is_file():
        raise RuntimeError(
            "Frozen development "
            "cache missing."
        )

    if not NORMALIZATION.is_file():
        raise RuntimeError(
            "Frozen normalization "
            "artifact missing."
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
            "file SHA changed."
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
            "Frozen development cache "
            "file SHA changed."
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

    gaussian_config = json.loads(
        GAUSSIAN_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    if (
        gaussian_config[
            "predictive_distribution"
        ][
            "predictive_covariance_is_distinct_from_measurement_covariance"
        ]
        is not True
    ):
        raise RuntimeError(
            "Measurement/predictive "
            "uncertainty distinction "
            "not frozen."
        )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    if torch.version.cuda != (
        "13.2"
    ):
        raise RuntimeError(
            "Expected CUDA runtime 13.2."
        )

    device = torch.device(
        "cuda:0"
    )

    if "A6000" not in (
        torch.cuda
        .get_device_name(0)
    ):
        raise RuntimeError(
            "Expected frozen RTX A6000."
        )

    set_global_determinism(
        20260821
    )

    checkpoint = torch.load(
        DET_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint[
            "state_dict_sha256"
        ]
        !=
        EXPECTED_STATE_SHA
    ):
        raise RuntimeError(
            "Deterministic checkpoint "
            "declared state SHA changed."
        )

    if (
        state_dict_sha256(
            checkpoint[
                "state_dict"
            ]
        )
        !=
        EXPECTED_STATE_SHA
    ):
        raise RuntimeError(
            "Deterministic state_dict "
            "content SHA changed."
        )

    deterministic = (
        make_deterministic_model(
            checkpoint[
                "configuration"
            ],
            device=device,
        )
    )

    deterministic.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    deterministic.eval()

    gaussian = (
        make_gaussian_model(
            checkpoint[
                "configuration"
            ],
            device=device,
        )
    )

    initialize_from_deterministic(
        gaussian,
        checkpoint[
            "state_dict"
        ],
        initial_std_normalized=0.5,
    )

    gaussian.eval()

    batch = 32

    target = torch.randn(
        batch,
        11,
        14,
        device=device,
    )

    neighbors = torch.randn(
        batch,
        8,
        11,
        14,
        device=device,
    )

    neighbor_mask = torch.ones(
        batch,
        8,
        device=device,
    )

    map_context = torch.randn(
        batch,
        10,
        device=device,
    )

    with torch.no_grad():
        deterministic_mean = (
            deterministic(
                target,
                neighbors,
                neighbor_mask,
                map_context,
            )
        )

        gaussian_output = (
            gaussian(
                target,
                neighbors,
                neighbor_mask,
                map_context,
            )
        )

    if not torch.equal(
        deterministic_mean,
        gaussian_output.mean,
    ):
        maximum_difference = float(
            (
                deterministic_mean
                -
                gaussian_output.mean
            )
            .abs()
            .max()
            .item()
        )

        raise RuntimeError(
            "Gaussian initial mean does "
            "not exactly reproduce the "
            "frozen deterministic GRU. "
            f"max difference = "
            f"{maximum_difference}"
        )

    diagonal = torch.diagonal(
        gaussian_output.scale_tril,
        dim1=-2,
        dim2=-1,
    )

    if not torch.allclose(
        diagonal,
        torch.full_like(
            diagonal,
            0.5,
        ),
        atol=1e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Initial predictive "
            "standard deviation changed."
        )

    covariance = (
        covariance_from_scale_tril(
            gaussian_output
            .scale_tril
        )
    )

    torch.linalg.cholesky(
        covariance
    )

    truth = torch.randn(
        batch,
        4,
        3,
        device=device,
    )

    validity = torch.ones(
        batch,
        4,
        device=device,
    )

    gaussian.train()

    output = gaussian(
        target,
        neighbors,
        neighbor_mask,
        map_context,
    )

    loss = masked_gaussian_nll(
        output.mean,
        output.scale_tril,
        truth,
        validity,
    )

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            "Gaussian CUDA NLL "
            "is non-finite."
        )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in gaussian.parameters()
        if parameter.grad
        is not None
    ]

    if not gradients:
        raise RuntimeError(
            "Gaussian CUDA backward "
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
            "Gaussian CUDA backward "
            "produced non-finite gradients."
        )

    torch.cuda.synchronize()

    gaussian.eval()

    with torch.no_grad():
        output = gaussian(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

        label_mean = torch.tensor(
            normalization[
                "label_displacement_mean"
            ],
            dtype=torch.float32,
            device=device,
        )

        label_std = torch.tensor(
            normalization[
                "label_displacement_std"
            ],
            dtype=torch.float32,
            device=device,
        )

        (
            metric_mean,
            metric_scale,
        ) = denormalize_gaussian(
            output.mean,
            output.scale_tril,
            label_mean,
            label_std,
        )

        metric_target = (
            truth
            *
            label_std[
                None,
                :,
                :
            ]
            +
            label_mean[
                None,
                :,
                :
            ]
        )

        coverage = empirical_coverage(
            metric_mean,
            metric_scale,
            metric_target,
            validity,
        )

    if tuple(
        coverage.keys()
    ) != (
        "0.50",
        "0.80",
        "0.90",
        "0.95",
        "0.99",
    ):
        raise RuntimeError(
            "Coverage confidence-level "
            "contract changed."
        )

    fit_shapes = (
        load_cache_shapes(
            FIT_CACHE
        )
    )

    dev_shapes = (
        load_cache_shapes(
            DEV_CACHE
        )
    )

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
        60
    ):
        print(
            combined
        )

        raise RuntimeError(
            "Expected full Stage4 "
            "regression 60/60."
        )

    report = {
        "stage": 4,
        "block": "4.4_preflight",
        "status": "PASS",

        "upstream": {
            "block43":
                "PASS",

            "block43_implementation_sha256":
                EXPECTED_BLOCK43_IMPL_SHA,

            "deterministic_checkpoint_sha256":
                EXPECTED_CHECKPOINT_SHA,

            "deterministic_state_dict_sha256":
                EXPECTED_STATE_SHA,

            "fit_cache_unchanged":
                True,

            "development_cache_unchanged":
                True,

            "fit_normalization_unchanged":
                True
        },

        "environment": {
            "torch":
                torch.__version__,

            "cuda_runtime":
                torch.version.cuda,

            "gpu":
                torch.cuda
                .get_device_name(0),
        },

        "probabilistic_contract": {
            "distribution":
                "full_covariance_Gaussian",

            "position_dimension":
                3,

            "horizons":
                4,

            "cholesky_parameters_per_horizon":
                6,

            "SPD_by_construction":
                True,

            "initial_predictive_std_normalized":
                0.5,

            "initial_mean_exactly_matches_deterministic":
                True,

            "measurement_covariance_is_input":
                True,

            "predictive_covariance_is_output":
                True,

            "measurement_predictive_uncertainty_separate":
                True,
        },

        "cuda_smoke": {
            "forward":
                True,

            "SPD_cholesky":
                True,

            "Gaussian_NLL":
                True,

            "backward":
                True,

            "finite":
                True,

            "metric_covariance_transform":
                True,

            "raw_coverage_metrics":
                True,
        },

        "cache_shapes": {
            "fit":
                fit_shapes,

            "development":
                dev_shapes,
        },

        "regression": {
            "tests_passed":
                60,

            "tests_total":
                60,
        },

        "not_yet_claimed": [
            "probabilistic model training",
            "trained Gaussian NLL",
            "calibration",
            "ECE",
            "Brier score",
            "GMM",
            "formal validation superiority"
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

    if FAILURE.exists():
        FAILURE.unlink()

    print(
        "============================================================"
    )

    print(
        "STAGE4 BLOCK 4.4 PART 1/2 PREFLIGHT"
    )

    print(
        "============================================================"
    )

    print(
        "Block4.3 frozen upstream    = PASS"
    )

    print(
        "deterministic checkpoint    = PASS"
    )

    print(
        "fit/dev cache reuse         = PASS"
    )

    print(
        "fit-only normalization      = PASS"
    )

    print(
        "Gaussian family             = FULL 3D"
    )

    print(
        "covariance parameterization = CHOLESKY"
    )

    print(
        "predictive covariance SPD   = PASS"
    )

    print(
        "initial deterministic mean  = EXACT MATCH"
    )

    print(
        "initial predictive std      = 0.5 normalized"
    )

    print(
        "measurement R_t             = INPUT"
    )

    print(
        "predictive covariance       = MODEL OUTPUT"
    )

    print(
        "uncertainty separation      = PASS"
    )

    print(
        "CUDA Gaussian forward       = PASS"
    )

    print(
        "CUDA Gaussian NLL           = PASS"
    )

    print(
        "CUDA Gaussian backward      = PASS"
    )

    print(
        "metric covariance transform = PASS"
    )

    print(
        "raw coverage 50/80/90/95/99= PASS"
    )

    print(
        "full Stage4 regression      = 60 / 60 PASS"
    )

    print(
        "STATUS = PASS"
    )

    print(
        "report =",
        REPORT
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        FAILURE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        text = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        lowered = (
            text.lower()
        )

        if (
            "checkpoint"
            in lowered
            or
            "state_dict"
            in lowered
        ):
            hint = (
                "Do not retrain Block4.3. "
                "Check whether a frozen "
                "Block4.3 artifact/path/hash "
                "was altered or whether this "
                "preflight assumed the wrong "
                "checkpoint schema."
            )

        elif (
            "60/60"
            in lowered
            or
            "regression"
            in lowered
        ):
            hint = (
                "Inspect the unittest output "
                "in artifacts/block44/"
                "preflight.log. Repair only "
                "the new Block4.4 code; do not "
                "modify frozen Block4.3 modules."
            )

        elif (
            "cuda"
            in lowered
            or
            "out of memory"
            in lowered
        ):
            hint = (
                "Check nvidia-smi first. "
                "This preflight uses only a "
                "batch of 32 and should not "
                "stress the RTX A6000. Do not "
                "reduce the scientific model "
                "configuration automatically."
            )

        elif (
            "cholesky"
            in lowered
            or
            "positive"
            in lowered
            or
            "covariance"
            in lowered
        ):
            hint = (
                "Repair the Block4.4 Cholesky "
                "parameterization/math only. "
                "Do not add covariance clipping "
                "that changes the frozen "
                "probabilistic semantics."
            )

        else:
            hint = (
                "Inspect this failure report "
                "and the final log lines. "
                "No expensive Stage2/Stage3 "
                "cache regeneration has occurred; "
                "Block4.3 remains untouched."
            )

        payload = {
            "stage": 4,
            "block": "4.4_part_1",
            "status": "BLOCKED",

            "exception_type":
                type(exc).__name__,

            "exception":
                str(exc),

            "traceback":
                traceback.format_exc(),

            "recovery_hint":
                hint,

            "Block43_modified":
                False,

            "expensive_cache_regenerated":
                False,

            "probabilistic_training_started":
                False,
        }

        FAILURE.write_text(
            json.dumps(
                payload,
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
            "BLOCK 4.4 PART 1/2 = BLOCKED"
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
            hint
        )

        print()
        print(
            "Block4.3 modified          = NO"
        )

        print(
            "cache regeneration         = NO"
        )

        print(
            "probabilistic training     = NOT STARTED"
        )

        print(
            "failure report =",
            FAILURE
        )

        traceback.print_exc()

        raise SystemExit(2)

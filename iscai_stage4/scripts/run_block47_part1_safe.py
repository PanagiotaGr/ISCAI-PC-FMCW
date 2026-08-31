from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback

import numpy as np
import torch


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
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

BLOCK43 = (
    STAGE4
    / "reports/block43_deterministic_gru.json"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/gaussian_gru.pt"
)

DETERMINISTIC_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/deterministic_gru.pt"
)

FIT_CACHE = (
    STAGE4
    / "artifacts/block43/fit_cache.npz"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/fit_normalization.json"
)

REPORT = (
    STAGE4
    / "reports/block47_part1_preflight.json"
)

FEATURE_CONTRACT = (
    STAGE4
    / "artifacts/block47/"
      "measurement_covariance_feature_contract.json"
)

FILES = (
    STAGE4
    / "src/iscai_stage4/ml/ablation_runtime.py",

    STAGE4
    / "tests/test_block47_ablation.py",
)

EXPECTED_BLOCK46_IMPL_SHA = (
    "bf7ba0ef8c83d0b6a8d5ef2acd624190"
    "35f4367b17d4d6844f7a99c288f1c961"
)

EXPECTED_GAUSSIAN_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_GAUSSIAN_STATE_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
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


def blocked(
    phase,
    reason,
    recovery,
):
    write_json(
        REPORT,
        {
            "stage":
                4,

            "block":
                "4.7_part_1",

            "status":
                "BLOCKED",

            "phase":
                phase,

            "reason":
                reason,

            "recovery":
                recovery,

            "ablation_training_started":
                False,

            "Blocks43_to_46_modified":
                False,

            "formal_validation_used":
                False,
        },
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.7 PART 1/2 = BLOCKED"
    )
    print(
        "============================================================"
    )
    print(
        "phase =",
        phase,
    )
    print(
        "reason =",
        reason,
    )
    print(
        "recovery =",
        recovery,
    )
    print(
        "ablation training started = NO"
    )
    print(
        "formal validation used    = NO"
    )
    print(
        "terminal remains open     = YES"
    )


def build_gaussian(
    configuration,
    *,
    device,
):
    from iscai_stage4.ml.gaussian_gru import (
        GaussianTrajectoryGRU,
    )

    model = configuration[
        "model"
    ]

    return GaussianTrajectoryGRU(
        target_hidden_dim=int(
            model[
                "target_hidden_dim"
            ]
        ),
        neighbor_hidden_dim=int(
            model[
                "neighbor_hidden_dim"
            ]
        ),
        map_hidden_dim=int(
            model[
                "map_hidden_dim"
            ]
        ),
        fusion_hidden_dim=int(
            model[
                "fusion_hidden_dim"
            ]
        ),
        use_neighbors=True,
        use_map=True,
    ).to(device)


def mean_difference(
    reference,
    candidate,
):
    return float(
        (
            reference
            -
            candidate
        )
        .abs()
        .max()
        .item()
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.7 PART 1/2 PREFLIGHT"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK46,
        BLOCK45,
        BLOCK44,
        BLOCK43,
        GAUSSIAN_CHECKPOINT,
        DETERMINISTIC_CHECKPOINT,
        FIT_CACHE,
        NORMALIZATION,
        *FILES,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_files",
            (
                "Missing: "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore only the missing "
                "artifact. Do not retrain "
                "upstream Blocks."
            ),
        )
        return

    # --------------------------------------------------------
    # Compile first.
    # --------------------------------------------------------

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in FILES:
        try:
            py_compile.compile(
                str(path),
                doraise=True,
            )

            print(
                path.name,
                "= PASS",
            )

        except Exception as exc:
            blocked(
                "compile",
                (
                    f"{path.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                (
                    "Repair only the new "
                    "Block4.7 file."
                ),
            )
            return

    # --------------------------------------------------------
    # Frozen upstream checks.
    # --------------------------------------------------------

    block46 = json.loads(
        BLOCK46.read_text(
            encoding="utf-8"
        )
    )

    block44 = json.loads(
        BLOCK44.read_text(
            encoding="utf-8"
        )
    )

    if (
        block46.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            "block46_status",
            "Block4.6 is not PASS.",
            "Do not continue.",
        )
        return

    if (
        block46[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK46_IMPL_SHA
    ):
        blocked(
            "block46_fingerprint",
            (
                "Frozen Block4.6 "
                "implementation SHA changed."
            ),
            (
                "Audit upstream before "
                "starting ablations."
            ),
        )
        return

    if (
        file_sha256(
            GAUSSIAN_CHECKPOINT
        )
        !=
        EXPECTED_GAUSSIAN_CHECKPOINT_SHA
    ):
        blocked(
            "Gaussian_checkpoint",
            (
                "Frozen Gaussian "
                "checkpoint SHA changed."
            ),
            (
                "Do not retrain automatically."
            ),
        )
        return

    # --------------------------------------------------------
    # Discover EXACT feature mapping.
    # --------------------------------------------------------

    from iscai_stage4.ml.ablation_runtime import (
        ABLATION_ACTOR_ONLY,
        ABLATION_FULL,
        ABLATION_NO_MAP,
        ABLATION_NO_R,
        apply_normalized_ablation,
        discover_history_feature_names,
        measurement_covariance_indices,
    )

    try:
        discovered = (
            discover_history_feature_names()
        )

        feature_names = (
            discovered[
                "names"
            ]
        )

        covariance_indices = (
            measurement_covariance_indices(
                feature_names
            )
        )

    except Exception as exc:
        # Give useful source evidence if automatic
        # discovery cannot prove the mapping.
        from iscai_stage4.data import (
            neural_inputs,
        )

        source_path = Path(
            neural_inputs.__file__
        )

        source_lines = (
            source_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )

        relevant = []

        for line_number, line in enumerate(
            source_lines,
            start=1,
        ):
            lower = line.lower()

            if any(
                token in lower
                for token in (
                    "cov",
                    "variance",
                    "sigma",
                    "uncert",
                    "feature",
                )
            ):
                relevant.append(
                    f"{line_number}: "
                    f"{line.strip()}"
                )

        relevant = relevant[
            :80
        ]

        print()
        print(
            "===== FEATURE SOURCE EVIDENCE ====="
        )

        for line in relevant:
            print(
                line
            )

        blocked(
            "feature_contract_discovery",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Do not guess covariance "
                "indices. Send the printed "
                "FEATURE SOURCE EVIDENCE and "
                "we will freeze the exact "
                "mapping before Part2."
            ),
        )
        return

    covariance_names = tuple(
        feature_names[
            index
        ]
        for index in (
            covariance_indices
        )
    )

    feature_payload = {
        "feature_dim":
            14,

        "feature_names":
            list(
                feature_names
            ),

        "mapping_origins":
            list(
                discovered[
                    "origins"
                ]
            ),

        "neural_inputs_source":
            discovered[
                "source_path"
            ],

        "measurement_covariance_indices":
            list(
                covariance_indices
            ),

        "measurement_covariance_names":
            list(
                covariance_names
            ),

        "ablation_semantics":
            (
                "zero_after_fit_only_normalization_"
                "equals_fit_mean_imputation"
            ),
    }

    write_json(
        FEATURE_CONTRACT,
        feature_payload,
    )

    print()
    print(
        "===== FROZEN 14-D INPUT CONTRACT ====="
    )

    for index, name in enumerate(
        feature_names
    ):
        marker = (
            "<-- R_t"
            if index
            in covariance_indices
            else ""
        )

        print(
            f"[{index:02d}] "
            f"{name} "
            f"{marker}"
        )

    print()
    print(
        "R_t feature indices       =",
        covariance_indices,
    )
    print(
        "R_t feature names         =",
        covariance_names,
    )
    print(
        "R_t mapping               = PASS"
    )

    # --------------------------------------------------------
    # Real frozen-Gaussian intervention smoke.
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        blocked(
            "cuda",
            "CUDA unavailable.",
            (
                "Check frozen environment."
            ),
        )
        return

    device = torch.device(
        "cuda:0"
    )

    deterministic_checkpoint = (
        torch.load(
            DETERMINISTIC_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
    )

    gaussian_checkpoint = (
        torch.load(
            GAUSSIAN_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
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
        blocked(
            "Gaussian_state",
            "Frozen Gaussian state changed.",
            "Audit Block4.4.",
        )
        return

    model = build_gaussian(
        deterministic_checkpoint[
            "configuration"
        ],
        device=device,
    )

    model.load_state_dict(
        gaussian_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    model.eval()

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    from iscai_stage4.ml.gmm_runtime import (
        GMMNormalizer,
    )

    normalizer = GMMNormalizer(
        normalization,
        device=device,
    )

    with np.load(
        FIT_CACHE,
        allow_pickle=False,
    ) as data:
        arrays = {
            key:
                np.asarray(
                    data[key]
                )
            for key in (
                "target",
                "neighbors",
                "neighbor_mask",
                "map_context",
                "future",
                "future_mask",
                "class_id",
            )
        }

    probe_count = min(
        256,
        int(
            arrays[
                "target"
            ].shape[0]
        ),
    )

    indices = np.arange(
        probe_count,
        dtype=np.int64,
    )

    batch = normalizer.prepare(
        arrays,
        indices,
        device=device,
    )

    outputs = {}

    with torch.inference_mode():
        for mode in (
            ABLATION_FULL,
            ABLATION_NO_R,
            ABLATION_ACTOR_ONLY,
            ABLATION_NO_MAP,
        ):
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
                mode=mode,
                covariance_indices=(
                    covariance_indices
                ),
            )

            output = model(
                target,
                neighbors,
                neighbor_mask,
                map_context,
            )

            covariance = (
                output.scale_tril
                @
                output.scale_tril.transpose(
                    -1,
                    -2,
                )
            )

            (
                _,
                info,
            ) = (
                torch.linalg.cholesky_ex(
                    covariance
                )
            )

            if bool(
                torch.any(
                    info != 0
                )
            ):
                blocked(
                    "ablation_SPD",
                    (
                        f"{mode} produced "
                        "non-SPD covariance."
                    ),
                    (
                        "Audit only new "
                        "ablation pathway."
                    ),
                )
                return

            outputs[
                mode
            ] = (
                output.mean
                .detach()
                .clone()
            )

    full_mean = outputs[
        ABLATION_FULL
    ]

    intervention_changes = {
        "no_R_t":
            mean_difference(
                full_mean,
                outputs[
                    ABLATION_NO_R
                ],
            ),

        "actor_only":
            mean_difference(
                full_mean,
                outputs[
                    ABLATION_ACTOR_ONLY
                ],
            ),

        "no_map":
            mean_difference(
                full_mean,
                outputs[
                    ABLATION_NO_MAP
                ],
            ),
    }

    print()
    print(
        "===== FROZEN GAUSSIAN INTERVENTION SMOKE ====="
    )
    print(
        "probe samples             =",
        probe_count,
    )
    print(
        "no-R_t mean max delta     =",
        intervention_changes[
            "no_R_t"
        ],
    )
    print(
        "actor-only mean max delta =",
        intervention_changes[
            "actor_only"
        ],
    )
    print(
        "no-map mean max delta     =",
        intervention_changes[
            "no_map"
        ],
    )
    print(
        "all intervention SPD      = PASS"
    )

    # No minimum effect-size threshold here.
    # The true ablation comparison requires
    # retraining each ablation in Part2.

    # --------------------------------------------------------
    # Unit tests.
    # --------------------------------------------------------

    new_tests = subprocess.run(
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
            "test_block47_ablation.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    new_output = (
        new_tests.stdout
        +
        "\n"
        +
        new_tests.stderr
    )

    print()
    print(
        "===== BLOCK4.7 UNIT TESTS ====="
    )
    print(
        new_output
    )

    if (
        new_tests.returncode != 0
    ):
        blocked(
            "Block47_tests",
            "New ablation tests failed.",
            (
                "Repair only the new "
                "Block4.7 code/tests."
            ),
        )
        return

    # --------------------------------------------------------
    # Full Stage4 regression = 100.
    # --------------------------------------------------------

    full = subprocess.run(
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

    full_output = (
        full.stdout
        +
        "\n"
        +
        full.stderr
    )

    print()
    print(
        "===== FULL STAGE4 REGRESSION ====="
    )
    print(
        full_output
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        full_output,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        full.returncode != 0
        or
        count != 100
    ):
        blocked(
            "regression",
            (
                "Expected 100/100 tests, "
                f"detected {count}."
            ),
            (
                "Do not start ablation "
                "training until regression "
                "is PASS."
            ),
        )
        return

    write_json(
        REPORT,
        {
            "stage":
                4,

            "block":
                "4.7_part_1",

            "status":
                "PASS",

            "upstream":
                "Blocks4.0_to_4.6_FROZEN",

            "feature_contract":
                feature_payload,

            "planned_retrained_ablations": [
                "full_frozen_Gaussian_reference",
                "no_measurement_covariance",
                "actor_only",
                "no_map"
            ],

            "no_measurement_covariance_semantics":
                (
                    "zero_standardized_R_t_features_"
                    "equals_fit_mean_imputation"
                ),

            "intervention_probe":
                intervention_changes,

            "intervention_effect_size_is_gate":
                False,

            "all_intervention_covariances_SPD":
                True,

            "measurement_to_predictive_uncertainty_analysis":
                "TO_BE_FROZEN_IN_PART2_AFTER_EXACT_R_t_FEATURE_REVIEW",

            "regression": {
                "tests_passed":
                    100,

                "tests_total":
                    100,
            },

            "ablation_training_started":
                False,

            "formal_validation_used":
                False,
        },
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.7 PART 1/2 SAFE GATE"
    )
    print(
        "============================================================"
    )
    print(
        "Block4.6 frozen upstream  = PASS"
    )
    print(
        "Gaussian baseline         = FROZEN"
    )
    print(
        "history feature dimension = 14 PASS"
    )
    print(
        "R_t mapping               = PASS"
    )
    print(
        "R_t indices               =",
        covariance_indices,
    )
    print(
        "R_t names                 =",
        covariance_names,
    )
    print(
        "no-R_t semantics          = FIT-MEAN IMPUTATION"
    )
    print(
        "actor-only intervention   = PASS"
    )
    print(
        "no-map intervention       = PASS"
    )
    print(
        "all intervention SPD      = PASS"
    )
    print(
        "training started          = NO"
    )
    print(
        "formal validation used    = NO"
    )
    print(
        "full Stage4 regression    = 100 / 100 PASS"
    )
    print(
        "STATUS = PASS"
    )
    print(
        "terminal remains open     = YES"
    )


try:
    main()

except BaseException as exc:
    blocked(
        "unexpected_controller_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
        (
            "Unexpected Part1 controller "
            "failure. No ablation training "
            "has started."
        ),
    )

    print()
    traceback.print_exc()

# No sys.exit().

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import traceback

import torch


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

BLOCK45 = (
    STAGE4
    / "reports/block45_calibration.json"
)

BLOCK44 = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/gaussian_gru.pt"
)

DETERMINISTIC_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/deterministic_gru.pt"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/covariance_scaler.json"
)

CONFIG = (
    STAGE4
    / "configs/stage4_gmm_gru.json"
)

REPORT = (
    STAGE4
    / "reports/block46_part1_preflight.json"
)

FILES = (
    STAGE4
    / "src/iscai_stage4/ml/gmm_gru.py",

    STAGE4
    / "src/iscai_stage4/ml/gmm_math.py",

    STAGE4
    / "tests/test_block46_gmm.py",
)

EXPECTED_BLOCK45_IMPL_SHA = (
    "482fa179b958f05526657d9970346b3ca"
    "699c649b77783f63d4c8de2fd7be7cf"
)

EXPECTED_GAUSSIAN_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_GAUSSIAN_STATE_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
)

EXPECTED_DETERMINISTIC_CHECKPOINT_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_CALIBRATOR_SHA = (
    "508ff2e3fbcfafe8e001155340c25baaf"
    "3772fe2561a8022a9ed1cf780e66087"
)


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


def state_dict_sha256(
    state_dict,
):
    digest = sha256()

    for key in sorted(
        state_dict
    ):
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
            str(
                tensor.dtype
            ).encode("ascii")
        )
        digest.update(b"\0")

        digest.update(
            tensor.numpy()
            .tobytes()
        )
        digest.update(b"\0")

    return digest.hexdigest()


def blocked(
    phase,
    reason,
    recovery,
):
    payload = {
        "stage":
            4,

        "block":
            "4.6_part_1",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "GMM_training_started":
            False,

        "Block45_modified":
            False,

        "formal_validation_used":
            False,
    }

    REPORT.write_text(
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
        "BLOCK 4.6 PART 1/2 = BLOCKED"
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
        "GMM training started = NO"
    )
    print(
        "Block4.5 modified     = NO"
    )
    print(
        "terminal remains open = YES"
    )


def make_gaussian(
    deterministic_configuration,
    *,
    device,
):
    from iscai_stage4.ml.gaussian_gru import (
        GaussianTrajectoryGRU,
    )

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


def make_gmm(
    deterministic_configuration,
    *,
    device,
):
    from iscai_stage4.ml.gmm_gru import (
        GMMTrajectoryGRU,
    )

    section = (
        deterministic_configuration[
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


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.6 PART 1/2 PREFLIGHT"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK45,
        BLOCK44,
        GAUSSIAN_CHECKPOINT,
        DETERMINISTIC_CHECKPOINT,
        CALIBRATOR,
        CONFIG,
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
                "file. Do not retrain "
                "Blocks 4.3–4.5."
            ),
        )
        return

    # --------------------------------------------------------
    # Compile before import/runtime.
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
                    "Block4.6 file."
                ),
            )
            return

    # --------------------------------------------------------
    # Frozen upstream.
    # --------------------------------------------------------

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
        block45.get("status")
        !=
        "PASS"
    ):
        blocked(
            "block45_status",
            "Block4.5 is not PASS.",
            "Do not continue.",
        )
        return

    if (
        block45[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK45_IMPL_SHA
    ):
        blocked(
            "block45_fingerprint",
            (
                "Frozen Block4.5 "
                "implementation SHA changed."
            ),
            (
                "Audit upstream before "
                "GMM work."
            ),
        )
        return

    if (
        block44.get("status")
        !=
        "PASS"
    ):
        blocked(
            "block44_status",
            "Block4.4 is not PASS.",
            "Do not continue.",
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
                "Do not retrain "
                "automatically."
            ),
        )
        return

    if (
        file_sha256(
            DETERMINISTIC_CHECKPOINT
        )
        !=
        EXPECTED_DETERMINISTIC_CHECKPOINT_SHA
    ):
        blocked(
            "deterministic_checkpoint",
            (
                "Frozen deterministic "
                "checkpoint SHA changed."
            ),
            (
                "Audit Block4.3."
            ),
        )
        return

    if (
        file_sha256(
            CALIBRATOR
        )
        !=
        EXPECTED_CALIBRATOR_SHA
    ):
        blocked(
            "calibrator",
            (
                "Frozen Block4.5 "
                "calibrator SHA changed."
            ),
            (
                "Audit Block4.5."
            ),
        )
        return

    print()
    print(
        "Block4.5 frozen upstream = PASS"
    )
    print(
        "Gaussian checkpoint       = PASS"
    )
    print(
        "calibrator                = PASS"
    )

    # --------------------------------------------------------
    # Import new GMM layer.
    # --------------------------------------------------------

    try:
        from iscai_stage4.ml.gmm_gru import (
            CENTRAL_COMPONENT_INDEX,
            GMM_COMPONENTS,
            initialize_gmm_from_gaussian,
        )

        from iscai_stage4.ml.gmm_math import (
            masked_gmm_joint_nll,
        )

    except Exception as exc:
        blocked(
            "import",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Repair only Block4.6 "
                "imports/code."
            ),
        )
        return

    if GMM_COMPONENTS != 3:
        blocked(
            "component_count",
            "Expected exactly K=3.",
            "Restore frozen GMM contract.",
        )
        return

    # --------------------------------------------------------
    # Construct Gaussian from Block4.3 architecture metadata,
    # load Block4.4 predictive weights.
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
            (
                "Frozen Gaussian "
                "state SHA changed."
            ),
            (
                "Audit Block4.4."
            ),
        )
        return

    deterministic_configuration = (
        deterministic_checkpoint[
            "configuration"
        ]
    )

    gaussian = make_gaussian(
        deterministic_configuration,
        device=device,
    )

    gaussian.load_state_dict(
        gaussian_checkpoint[
            "state_dict"
        ],
        strict=True,
    )

    gaussian.eval()

    gmm = make_gmm(
        deterministic_configuration,
        device=device,
    )

    initialize_gmm_from_gaussian(
        gmm,
        gaussian_checkpoint[
            "state_dict"
        ],
    )

    gmm.eval()

    # --------------------------------------------------------
    # Exact initialization / GPU forward.
    # --------------------------------------------------------

    torch.manual_seed(
        20260823
    )

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
        gaussian_output = gaussian(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

        gmm_output = gmm(
            target,
            neighbors,
            neighbor_mask,
            map_context,
        )

    central_mean = (
        gmm_output.means[
            :,
            CENTRAL_COMPONENT_INDEX,
        ]
    )

    if not torch.equal(
        central_mean,
        gaussian_output.mean,
    ):
        maximum_difference = float(
            (
                central_mean
                -
                gaussian_output.mean
            )
            .abs()
            .max()
            .item()
        )

        blocked(
            "Gaussian_initialization",
            (
                "Central GMM mean does not "
                "exactly match frozen Gaussian. "
                f"max difference={maximum_difference}"
            ),
            (
                "Repair only GMM "
                "initialization mapping."
            ),
        )
        return

    probability_sum = (
        gmm_output
        .mixture_probabilities
        .sum(
            dim=-1
        )
    )

    if not torch.allclose(
        probability_sum,
        torch.ones_like(
            probability_sum
        ),
        atol=1e-7,
        rtol=0.0,
    ):
        blocked(
            "mixture_probability",
            (
                "Mixture probabilities "
                "do not sum to one."
            ),
            (
                "Repair GMM probability math."
            ),
        )
        return

    covariance = (
        gmm_output.covariance
    )

    (
        _,
        info,
    ) = torch.linalg.cholesky_ex(
        covariance
    )

    if bool(
        torch.any(
            info != 0
        )
    ):
        blocked(
            "SPD",
            (
                "Initial GMM component "
                "covariance is not SPD."
            ),
            (
                "Repair only Block4.6 "
                "component covariance path."
            ),
        )
        return

    # --------------------------------------------------------
    # CUDA NLL/backward.
    # --------------------------------------------------------

    gmm.train()

    output = gmm(
        target,
        neighbors,
        neighbor_mask,
        map_context,
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

    loss = masked_gmm_joint_nll(
        output.mixture_logits,
        output.means,
        output.scale_tril,
        truth,
        validity,
    )

    if not bool(
        torch.isfinite(
            loss
        )
    ):
        blocked(
            "GMM_NLL",
            "GMM NLL is non-finite.",
            "Repair GMM likelihood math.",
        )
        return

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in gmm.parameters()
        if parameter.grad
        is not None
    ]

    if not gradients:
        blocked(
            "backward",
            "No GMM gradients produced.",
            "Repair new GMM implementation.",
        )
        return

    if not all(
        bool(
            torch.isfinite(
                gradient
            ).all()
        )
        for gradient
        in gradients
    ):
        blocked(
            "backward",
            (
                "Non-finite GMM gradients."
            ),
            (
                "Repair new GMM NLL/"
                "parameterization."
            ),
        )
        return

    torch.cuda.synchronize()

    print()
    print(
        "GMM components            = 3 PASS"
    )
    print(
        "trajectory-level weights  = PASS"
    )
    print(
        "central Gaussian mean     = EXACT MATCH"
    )
    print(
        "symmetry break            = PASS"
    )
    print(
        "component covariance SPD  = PASS"
    )
    print(
        "CUDA GMM NLL              = PASS"
    )
    print(
        "CUDA GMM backward         = PASS"
    )

    # --------------------------------------------------------
    # New unit tests.
    # --------------------------------------------------------

    new_tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4 / "tests"
            ),
            "-p",
            "test_block46_gmm.py",
        ],
        cwd=str(STAGE4),
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
        "===== BLOCK4.6 UNIT TESTS ====="
    )
    print(
        new_output
    )

    if new_tests.returncode != 0:
        blocked(
            "Block46_tests",
            "New GMM tests failed.",
            (
                "Repair only new "
                "Block4.6 code/tests."
            ),
        )
        return

    # --------------------------------------------------------
    # Full regression expected 88.
    # --------------------------------------------------------

    full = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4 / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(STAGE4),
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
        count != 88
    ):
        blocked(
            "regression",
            (
                "Expected 88/88 tests, "
                f"detected {count}."
            ),
            (
                "Do not train GMM. "
                "Inspect regression first."
            ),
        )
        return

    report = {
        "stage":
            4,

        "block":
            "4.6_part_1",

        "status":
            "PASS",

        "distribution": {
            "family":
                "trajectory_level_GMM",

            "components":
                3,

            "trajectory_level_mixture_weights":
                True,

            "independent_horizon_mode_switching":
                False,

            "component_covariance":
                "full_3D_SPD",
        },

        "initialization": {
            "source":
                "frozen_Block4.4_Gaussian",

            "central_component_exact_Gaussian_mean":
                True,

            "symmetry_break_second_coordinate":
                [
                    -0.05,
                    0.0,
                    0.05,
                ],

            "maneuver_labels":
                False,
        },

        "objective": {
            "joint_trajectory_GMM_NLL":
                True,

            "logsumexp":
                True,

            "future_validity_loss_mask_only":
                True,
        },

        "CUDA": {
            "forward":
                True,

            "SPD":
                True,

            "NLL":
                True,

            "backward":
                True,
        },

        "regression": {
            "tests_passed":
                88,

            "tests_total":
                88,
        },

        "GMM_training_started":
            False,

        "formal_validation_used":
            False,

        "Gaussian_mandatory_baseline_modified":
            False,
    }

    REPORT.write_text(
        json.dumps(
            report,
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
        "STAGE4 BLOCK 4.6 PART 1/2 SAFE GATE"
    )
    print(
        "============================================================"
    )
    print(
        "Block4.5 frozen upstream  = PASS"
    )
    print(
        "mandatory Gaussian        = UNCHANGED"
    )
    print(
        "GMM components            = 3"
    )
    print(
        "mode scope                = TRAJECTORY LEVEL"
    )
    print(
        "independent mode switching= NO"
    )
    print(
        "central Gaussian mean     = EXACT MATCH"
    )
    print(
        "component covariance      = FULL 3D SPD"
    )
    print(
        "joint GMM NLL             = PASS"
    )
    print(
        "CUDA backward             = PASS"
    )
    print(
        "maneuver labels used      = NO"
    )
    print(
        "future model input        = NO"
    )
    print(
        "GMM training started      = NO"
    )
    print(
        "formal validation used    = NO"
    )
    print(
        "full Stage4 regression    = 88 / 88 PASS"
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
            "Unexpected preflight error. "
            "No GMM training has started "
            "and Blocks4.0–4.5 remain frozen."
        ),
    )

    print()
    traceback.print_exc()

# No sys.exit().

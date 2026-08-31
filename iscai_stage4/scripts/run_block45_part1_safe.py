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

BLOCK44 = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

CHECKPOINT = (
    STAGE4
    / "artifacts/block44/gaussian_gru.pt"
)

CALIBRATION_MANIFEST = (
    STAGE4
    / "artifacts/block41/calibration.jsonl"
)

CONFIG = (
    STAGE4
    / "configs/stage4_calibration.json"
)

REPORT = (
    STAGE4
    / "reports/block45_part1_preflight.json"
)

FILES = (
    STAGE4
    / "src/iscai_stage4/ml/calibration.py",

    STAGE4
    / "src/iscai_stage4/ml/__init__.py",

    STAGE4
    / "tests/test_block45_calibration.py",
)

EXPECTED_BLOCK44_IMPL_SHA = (
    "b9ea474443d3a54235f09cd4e2b616374"
    "b37456f2b9e080f5334f76f4dd7c432"
)

EXPECTED_CHECKPOINT_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_STATE_SHA = (
    "1d66cf082d0d9be41319f7dcbe18fc259"
    "910de7f45de006f9043129837b86be3"
)

EXPECTED_CALIBRATION_MANIFEST_SHA = (
    "e003dd5c4d5a253729700b12c3754c47"
    "ffb99dadfa035b46627f01ec91eed4be"
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


def write_report(
    payload,
):
    REPORT.write_text(
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
    payload = {
        "stage": 4,
        "block": "4.5_part_1",
        "status": "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "block44_modified":
            False,

        "calibration_cache_built":
            False,

        "calibrator_fitted":
            False,

        "formal_validation_used":
            False,
    }

    write_report(
        payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.5 PART 1/2 = BLOCKED"
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
        "Block4.4 modified       = NO"
    )
    print(
        "calibration cache built = NO"
    )
    print(
        "calibrator fitted       = NO"
    )
    print(
        "formal validation used  = NO"
    )
    print(
        "terminal remains open   = YES"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.5 PART 1/2 PREFLIGHT"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK44,
        CHECKPOINT,
        CALIBRATION_MANIFEST,
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
                "Restore only missing "
                "Block4.5 files/artifacts. "
                "Do not modify Block4.4."
            ),
        )
        return

    # --------------------------------------------------------
    # Compile all new files before imports/tests.
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
                    f"{path}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                (
                    "Repair only the new "
                    "Block4.5 source/test file."
                ),
            )
            return

    # --------------------------------------------------------
    # Frozen upstream checks.
    # --------------------------------------------------------

    block44 = json.loads(
        BLOCK44.read_text(
            encoding="utf-8"
        )
    )

    if (
        block44.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            "block44_status",
            "Block4.4 is not PASS.",
            (
                "Do not continue until "
                "Block4.4 remains frozen."
            ),
        )
        return

    if (
        block44[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK44_IMPL_SHA
    ):
        blocked(
            "block44_fingerprint",
            (
                "Frozen Block4.4 "
                "implementation SHA changed."
            ),
            (
                "Do not recalibrate. "
                "Identify why Block4.4 "
                "source changed first."
            ),
        )
        return

    if (
        file_sha256(
            CHECKPOINT
        )
        !=
        EXPECTED_CHECKPOINT_SHA
    ):
        blocked(
            "gaussian_checkpoint",
            (
                "Frozen Gaussian "
                "checkpoint SHA changed."
            ),
            (
                "Do not retrain automatically. "
                "Restore/audit the frozen "
                "Block4.4 checkpoint."
            ),
        )
        return

    if (
        block44[
            "checkpoint"
        ][
            "state_dict_sha256"
        ]
        !=
        EXPECTED_STATE_SHA
    ):
        blocked(
            "gaussian_state",
            (
                "Frozen Gaussian state "
                "SHA declaration changed."
            ),
            (
                "Audit Block4.4 before "
                "starting calibration."
            ),
        )
        return

    if (
        file_sha256(
            CALIBRATION_MANIFEST
        )
        !=
        EXPECTED_CALIBRATION_MANIFEST_SHA
    ):
        blocked(
            "calibration_manifest",
            (
                "Frozen calibration "
                "manifest SHA changed."
            ),
            (
                "Do not create calibration "
                "cache until Block4.1 "
                "partition evidence is restored."
            ),
        )
        return

    lines = [
        line
        for line in (
            CALIBRATION_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    ]

    if len(lines) != 7209:
        blocked(
            "calibration_manifest_count",
            (
                "Expected 7209 frozen "
                f"calibration scenarios, "
                f"got {len(lines)}."
            ),
            (
                "Audit the frozen "
                "Block4.1 manifest."
            ),
        )
        return

    print()
    print(
        "Block4.4 frozen upstream = PASS"
    )
    print(
        "Gaussian checkpoint       = PASS"
    )
    print(
        "calibration manifest      = 7209 PASS"
    )

    # --------------------------------------------------------
    # Import/calibration math smoke.
    # --------------------------------------------------------

    try:
        from iscai_stage4.ml import (
            CovarianceScaleCalibration,
            apply_variance_scale,
            covariance_from_scale_tril,
            fit_per_horizon_covariance_scale,
            reliability_metrics,
        )

    except Exception as exc:
        blocked(
            "import",
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            (
                "Repair only Block4.5 "
                "imports/exports."
            ),
        )
        return

    print(
        "calibration imports       = PASS"
    )

    # --------------------------------------------------------
    # Synthetic GPU mathematical smoke.
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        blocked(
            "cuda",
            "CUDA unavailable.",
            (
                "Check the already frozen "
                "PyTorch/CUDA environment."
            ),
        )
        return

    device = torch.device(
        "cuda:0"
    )

    torch.manual_seed(
        20260822
    )

    residual = torch.randn(
        20000,
        4,
        3,
        device=device,
    )

    # Deliberately underdispersed raw model:
    # q is inflated relative to chi-square df=3.
    q = (
        2.0
        *
        residual.square()
        .sum(
            dim=-1
        )
    )

    mask = torch.ones(
        20000,
        4,
        device=device,
    )

    fitted = (
        fit_per_horizon_covariance_scale(
            q,
            mask,
        )
    )

    calibrator = fitted[
        "calibrator"
    ]

    raw_ece = fitted[
        "raw_reliability"
    ][
        "coverage_ECE_macro"
    ]

    calibrated_ece = fitted[
        "calibrated_reliability"
    ][
        "coverage_ECE_macro"
    ]

    if (
        calibrated_ece
        >
        raw_ece
        +
        1e-15
    ):
        blocked(
            "synthetic_calibration",
            (
                "Synthetic calibration "
                "made ECE worse."
            ),
            (
                "Repair the new "
                "Block4.5 calibration "
                "candidate/objective logic."
            ),
        )
        return

    if not all(
        value > 1.0
        for value in (
            calibrator
            .variance_scale
        )
    ):
        blocked(
            "synthetic_inflation",
            (
                "Underdispersed synthetic "
                "posterior did not select "
                "covariance inflation."
            ),
            (
                "Audit the scale-fitting "
                "semantics before real "
                "calibration."
            ),
        )
        return

    identity_scale = (
        torch.eye(
            3,
            device=device,
        )
        .reshape(
            1,
            1,
            3,
            3,
        )
        .repeat(
            32,
            4,
            1,
            1,
        )
    )

    calibrated_scale = (
        apply_variance_scale(
            identity_scale,
            calibrator
            .variance_scale,
        )
    )

    covariance = (
        covariance_from_scale_tril(
            calibrated_scale
        )
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
            "synthetic_SPD",
            (
                "Calibrated synthetic "
                "covariance lost SPD."
            ),
            (
                "Repair only covariance "
                "scaling math."
            ),
        )
        return

    print()
    print(
        "synthetic raw ECE          =",
        raw_ece,
    )
    print(
        "synthetic calibrated ECE   =",
        calibrated_ece,
    )
    print(
        "synthetic variance scales  =",
        calibrator
        .variance_scale,
    )
    print(
        "ECE non-worsening          = PASS"
    )
    print(
        "calibrated covariance SPD  = PASS"
    )

    # --------------------------------------------------------
    # New Block4.5 tests.
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
            "test_block45_calibration.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    output = (
        new_tests.stdout
        +
        "\n"
        +
        new_tests.stderr
    )

    print()
    print(
        "===== BLOCK4.5 UNIT TESTS ====="
    )
    print(
        output
    )

    if new_tests.returncode != 0:
        blocked(
            "block45_tests",
            "New Block4.5 tests failed.",
            (
                "Use the traceback above "
                "and repair only new "
                "calibration code/tests."
            ),
        )
        return

    # --------------------------------------------------------
    # Full Stage4 regression.
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
        count != 72
    ):
        blocked(
            "full_regression",
            (
                "Expected 72/72 tests, "
                f"detected {count}."
            ),
            (
                "Do not modify frozen "
                "Block4.0–4.4 behavior. "
                "Inspect failing test output."
            ),
        )
        return

    payload = {
        "stage": 4,
        "block": "4.5_part_1",
        "status": "PASS",

        "upstream": {
            "block44":
                "FROZEN",

            "checkpoint_sha256":
                EXPECTED_CHECKPOINT_SHA,

            "state_dict_sha256":
                EXPECTED_STATE_SHA,

            "calibration_manifest_sha256":
                EXPECTED_CALIBRATION_MANIFEST_SHA,

            "calibration_scenarios":
                7209,
        },

        "method": {
            "family":
                "per_horizon_scalar_covariance_scaling",

            "mean_modified":
                False,

            "measurement_R_t_modified":
                False,

            "predictive_covariance_modified":
                True,

            "identity_candidate":
                True,
        },

        "metrics": {
            "coverage_levels":
                [
                    0.50,
                    0.80,
                    0.90,
                    0.95,
                    0.99,
                ],

            "ECE":
                "macro_absolute_coverage_error",

            "Brier":
                "coverage_event_Brier",

            "reliability":
                "nominal_vs_empirical_coverage",
        },

        "synthetic_preflight": {
            "raw_ECE":
                raw_ece,

            "calibrated_ECE":
                calibrated_ece,

            "variance_scales":
                [
                    float(x)
                    for x in (
                        calibrator
                        .variance_scale
                    )
                ],

            "ECE_nonworsening":
                True,

            "SPD":
                True,
        },

        "regression": {
            "tests_passed":
                72,

            "tests_total":
                72,
        },

        "calibration_cache_built":
            False,

        "real_calibrator_fitted":
            False,

        "formal_validation_used":
            False,
    }

    write_report(
        payload
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.5 PART 1/2 SAFE GATE"
    )
    print(
        "============================================================"
    )
    print(
        "Block4.4 frozen upstream = PASS"
    )
    print(
        "calibration partition     = 7209 FROZEN"
    )
    print(
        "calibration family        = COVARIANCE SCALING"
    )
    print(
        "per-horizon alpha         = PASS"
    )
    print(
        "predictive mean modified  = NO"
    )
    print(
        "measurement R_t modified  = NO"
    )
    print(
        "predictive covariance     = CALIBRATED"
    )
    print(
        "coverage levels           = 50/80/90/95/99"
    )
    print(
        "coverage ECE semantics    = FROZEN"
    )
    print(
        "coverage Brier semantics  = FROZEN"
    )
    print(
        "synthetic ECE gate        = PASS"
    )
    print(
        "SPD after scaling         = PASS"
    )
    print(
        "full Stage4 regression    = 72 / 72 PASS"
    )
    print(
        "real calibration started  = NO"
    )
    print(
        "formal validation used    = NO"
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
            "error. Block4.4 is untouched "
            "and real calibration has not "
            "started."
        ),
    )

    print()
    traceback.print_exc()

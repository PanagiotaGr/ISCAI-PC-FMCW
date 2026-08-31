from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import py_compile
import re
import shutil
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

RUNNER = (
    STAGE4
    / "scripts/"
      "run_block45_real_calibration.py"
)

SAFE_CONTROLLER = (
    STAGE4
    / "scripts/"
      "run_block45_part2_safe.py"
)

RUNTIME = (
    STAGE4
    / "src/iscai_stage4/ml/"
      "calibration_runtime.py"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/"
      "gaussian_gru.pt"
)

DETERMINISTIC_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/"
      "deterministic_gru.pt"
)

BACKUP = (
    STAGE4
    / "artifacts/block45/"
      "run_block45_real_calibration."
      "pre_model_config_repair.py"
)

REPAIR_REPORT = (
    STAGE4
    / "reports/"
      "block45_part2_model_config_repair.json"
)

RESUME_LOG = (
    STAGE4
    / "artifacts/block45/"
      "part2_after_model_config_repair.log"
)

EXPECTED_DET_CHECKPOINT_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_DET_STATE_SHA = (
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

REPAIR_MARKER = (
    "# BLOCK45_ARCHITECTURE_SOURCE_REPAIR"
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
            state_dict[key]
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


def write_report(
    payload,
):
    REPAIR_REPORT.write_text(
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
    write_report({
        "stage":
            4,

        "block":
            "4.5_part_2_repair",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "block43_modified":
            False,

        "block44_modified":
            False,

        "calibration_scientific_contract_modified":
            False,

        "full_calibration_scan_started_by_repair":
            False,
    })

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.5 PART 2/2 REPAIR = BLOCKED"
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
        "Block4.3 modified       = NO"
    )

    print(
        "Block4.4 modified       = NO"
    )

    print(
        "calibration semantics   = UNCHANGED"
    )

    print(
        "terminal remains open   = YES"
    )


def main():
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.5 PART 2/2 — MODEL CONFIG REPAIR"
    )
    print(
        "============================================================"
    )

    # ========================================================
    # A. Required files
    # ========================================================

    required = (
        RUNNER,
        SAFE_CONTROLLER,
        RUNTIME,
        GAUSSIAN_CHECKPOINT,
        DETERMINISTIC_CHECKPOINT,
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
                "Block4.3 or Block4.4."
            ),
        )
        return

    # ========================================================
    # B. Verify BOTH frozen checkpoints before patching
    # ========================================================

    print()
    print(
        "===== FROZEN CHECKPOINT VERIFICATION ====="
    )

    actual_det_file_sha = (
        file_sha256(
            DETERMINISTIC_CHECKPOINT
        )
    )

    actual_gaussian_file_sha = (
        file_sha256(
            GAUSSIAN_CHECKPOINT
        )
    )

    if (
        actual_det_file_sha
        !=
        EXPECTED_DET_CHECKPOINT_SHA
    ):
        blocked(
            "deterministic_checkpoint_file",
            (
                "Frozen Block4.3 checkpoint "
                "file SHA changed."
            ),
            (
                "Do not proceed with "
                "calibration until the "
                "frozen artifact is audited."
            ),
        )
        return

    if (
        actual_gaussian_file_sha
        !=
        EXPECTED_GAUSSIAN_CHECKPOINT_SHA
    ):
        blocked(
            "gaussian_checkpoint_file",
            (
                "Frozen Block4.4 checkpoint "
                "file SHA changed."
            ),
            (
                "Do not retrain automatically. "
                "Audit the frozen Gaussian "
                "artifact first."
            ),
        )
        return

    deterministic_checkpoint = (
        torch.load(
            DETERMINISTIC_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
    )

    gaussian_checkpoint = (
        torch.load(
            GAUSSIAN_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )
    )

    actual_det_state_sha = (
        state_dict_sha256(
            deterministic_checkpoint[
                "state_dict"
            ]
        )
    )

    actual_gaussian_state_sha = (
        state_dict_sha256(
            gaussian_checkpoint[
                "state_dict"
            ]
        )
    )

    if (
        actual_det_state_sha
        !=
        EXPECTED_DET_STATE_SHA
    ):
        blocked(
            "deterministic_state",
            (
                "Frozen Block4.3 "
                "state-dict SHA changed."
            ),
            (
                "Do not continue until "
                "the deterministic artifact "
                "is restored/audited."
            ),
        )
        return

    if (
        actual_gaussian_state_sha
        !=
        EXPECTED_GAUSSIAN_STATE_SHA
    ):
        blocked(
            "gaussian_state",
            (
                "Frozen Block4.4 "
                "state-dict SHA changed."
            ),
            (
                "Do not continue until "
                "the Gaussian artifact "
                "is restored/audited."
            ),
        )
        return

    deterministic_configuration = (
        deterministic_checkpoint.get(
            "configuration"
        )
    )

    gaussian_configuration = (
        gaussian_checkpoint.get(
            "configuration"
        )
    )

    if not isinstance(
        deterministic_configuration,
        dict,
    ):
        blocked(
            "deterministic_configuration",
            (
                "Block4.3 checkpoint lacks "
                "a configuration dictionary."
            ),
            (
                "Inspect the frozen checkpoint "
                "schema; do not guess model "
                "dimensions."
            ),
        )
        return

    if (
        "model"
        not in
        deterministic_configuration
    ):
        blocked(
            "deterministic_model_section",
            (
                "Block4.3 configuration "
                "unexpectedly lacks 'model'."
            ),
            (
                "Inspect the frozen Block4.3 "
                "checkpoint schema."
            ),
        )
        return

    gaussian_has_model = (
        isinstance(
            gaussian_configuration,
            dict,
        )
        and
        "model"
        in
        gaussian_configuration
    )

    print(
        "Block4.3 checkpoint SHA   = PASS"
    )

    print(
        "Block4.3 state SHA        = PASS"
    )

    print(
        "Block4.3 model config     = PASS"
    )

    print(
        "Block4.4 checkpoint SHA   = PASS"
    )

    print(
        "Block4.4 state SHA        = PASS"
    )

    print(
        "Block4.4 config has model =",
        gaussian_has_model,
    )

    if gaussian_has_model:
        print(
            "NOTE: Gaussian config already "
            "contains model architecture; "
            "repair remains harmless."
        )

    else:
        print(
            "ROOT CAUSE CONFIRMED:"
        )

        print(
            "Block4.4 checkpoint configuration "
            "does not contain 'model'."
        )

        print(
            "Architecture must therefore be "
            "read from frozen Block4.3 config, "
            "exactly as Block4.4 training did."
        )

    # ========================================================
    # C. Minimal idempotent source repair
    # ========================================================

    source = RUNNER.read_text(
        encoding="utf-8"
    )

    old = '''    model = build_gaussian_model(
        checkpoint[
            "configuration"
        ],
        device=device,
    )

    model.load_state_dict(
'''

    new = '''    # BLOCK45_ARCHITECTURE_SOURCE_REPAIR
    #
    # The frozen Block4.4 Gaussian checkpoint stores the
    # Gaussian experiment configuration under "configuration".
    # That configuration intentionally does not contain the
    # deterministic architecture "model" section.
    #
    # Block4.4 itself constructed GaussianTrajectoryGRU from
    # the frozen Block4.3 deterministic checkpoint
    # configuration and then trained/loaded Gaussian weights.
    # Reproduce exactly that architecture source here.
    #
    # Block4.3 is used ONLY for architecture metadata.
    # Predictive weights remain exclusively the frozen
    # Block4.4 Gaussian state_dict.

    deterministic_architecture_checkpoint_path = (
        STAGE4
        / "artifacts/block43/"
          "deterministic_gru.pt"
    )

    expected_deterministic_checkpoint_sha = (
        "5456a76b84d558e9983a59b9f1d3060b"
        "a245e0d36d60883519654f809996dbc5"
    )

    expected_deterministic_state_sha = (
        "d2ffbc03c7cb2826fef2175c95f48725"
        "707ec6d791eaeffbd6f59bc507b8595a"
    )

    if (
        file_sha256(
            deterministic_architecture_checkpoint_path
        )
        !=
        expected_deterministic_checkpoint_sha
    ):
        raise RuntimeError(
            "Frozen Block4.3 deterministic "
            "checkpoint SHA changed."
        )

    deterministic_architecture_checkpoint = (
        torch.load(
            deterministic_architecture_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    )

    if (
        state_dict_sha256(
            deterministic_architecture_checkpoint[
                "state_dict"
            ]
        )
        !=
        expected_deterministic_state_sha
    ):
        raise RuntimeError(
            "Frozen Block4.3 deterministic "
            "state-dict SHA changed."
        )

    deterministic_architecture_configuration = (
        deterministic_architecture_checkpoint.get(
            "configuration"
        )
    )

    if not isinstance(
        deterministic_architecture_configuration,
        dict,
    ):
        raise RuntimeError(
            "Frozen Block4.3 checkpoint "
            "lacks architecture configuration."
        )

    if (
        "model"
        not in
        deterministic_architecture_configuration
    ):
        raise RuntimeError(
            "Frozen Block4.3 architecture "
            "configuration lacks 'model'."
        )

    model = build_gaussian_model(
        deterministic_architecture_configuration,
        device=device,
    )

    # Strict load proves that the architecture reconstructed
    # from Block4.3 metadata exactly matches the frozen
    # Block4.4 Gaussian state_dict.
    model.load_state_dict(
'''

    if REPAIR_MARKER in source:
        print()
        print(
            "source repair            = "
            "ALREADY APPLIED"
        )

    elif old in source:
        if not BACKUP.is_file():
            shutil.copy2(
                RUNNER,
                BACKUP,
            )

        source = source.replace(
            old,
            new,
            1,
        )

        RUNNER.write_text(
            source,
            encoding="utf-8",
        )

        print()
        print(
            "source repair            = APPLIED"
        )

        print(
            "backup                    =",
            BACKUP,
        )

    else:
        blocked(
            "source_repair_pattern",
            (
                "Expected pre-repair model "
                "construction block was not "
                "found, and repair marker "
                "is absent."
            ),
            (
                "Do not perform a broad "
                "automatic rewrite. Inspect "
                "only the model-construction "
                "section of the current "
                "Block4.5 runner."
            ),
        )
        return

    # ========================================================
    # D. Controlled compile
    # ========================================================

    print()
    print(
        "===== CONTROLLED COMPILE ====="
    )

    for path in (
        RUNTIME,
        RUNNER,
        SAFE_CONTROLLER,
    ):
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
                    "Repair only the patched "
                    "Block4.5 source. "
                    "No scenario scan has begun."
                ),
            )
            return

    # ========================================================
    # E. Constructor-only smoke BEFORE 7209-scene scan
    # ========================================================

    print()
    print(
        "===== GAUSSIAN CONSTRUCTOR SMOKE ====="
    )

    from iscai_stage4.ml.calibration_runtime import (
        build_gaussian_model,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    smoke_model = (
        build_gaussian_model(
            deterministic_configuration,
            device=device,
        )
    )

    incompatible = (
        smoke_model.load_state_dict(
            gaussian_checkpoint[
                "state_dict"
            ],
            strict=True,
        )
    )

    if (
        incompatible.missing_keys
        or
        incompatible.unexpected_keys
    ):
        blocked(
            "strict_state_load",
            (
                "Frozen Gaussian weights "
                "do not strictly match "
                "Block4.3 architecture."
            ),
            (
                "Do not scan calibration "
                "scenarios. Audit architecture "
                "metadata first."
            ),
        )
        return

    smoke_model.eval()

    with torch.inference_mode():
        target = torch.randn(
            2,
            11,
            14,
            device=device,
        )

        neighbours = torch.randn(
            2,
            8,
            11,
            14,
            device=device,
        )

        neighbour_mask = torch.ones(
            2,
            8,
            device=device,
        )

        map_context = torch.randn(
            2,
            10,
            device=device,
        )

        output = smoke_model(
            target,
            neighbours,
            neighbour_mask,
            map_context,
        )

    if tuple(
        output.mean.shape
    ) != (
        2,
        4,
        3,
    ):
        blocked(
            "constructor_output",
            (
                "Gaussian mean output "
                "shape is unexpected."
            ),
            (
                "Audit architecture before "
                "real calibration."
            ),
        )
        return

    if tuple(
        output.scale_tril.shape
    ) != (
        2,
        4,
        3,
        3,
    ):
        blocked(
            "constructor_output",
            (
                "Gaussian Cholesky output "
                "shape is unexpected."
            ),
            (
                "Audit architecture before "
                "real calibration."
            ),
        )
        return

    print(
        "architecture source        = "
        "FROZEN BLOCK4.3 CONFIG"
    )

    print(
        "predictive weights source  = "
        "FROZEN BLOCK4.4 STATE"
    )

    print(
        "strict state_dict load     = PASS"
    )

    print(
        "Gaussian forward           = PASS"
    )

    print(
        "mean shape                 = "
        "[2, 4, 3] PASS"
    )

    print(
        "Cholesky shape             = "
        "[2, 4, 3, 3] PASS"
    )

    del smoke_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # F. Full 76-test regression BEFORE scan
    # ========================================================

    print()
    print(
        "===== PRE-RESUME FULL REGRESSION ====="
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if test.stdout:
        print(
            test.stdout,
            end="",
        )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        test.stdout or "",
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
        test_count != 76
    ):
        blocked(
            "regression",
            (
                "Expected 76/76 tests; "
                f"detected {test_count}."
            ),
            (
                "Do not begin the "
                "7209-scenario scan. "
                "Inspect the test output above."
            ),
        )
        return

    print(
        "full Stage4 regression    = "
        "76 / 76 PASS"
    )

    # ========================================================
    # G. Repair PASS
    # ========================================================

    write_report({
        "stage":
            4,

        "block":
            "4.5_part_2_repair",

        "status":
            "PASS",

        "root_cause":
            (
                "Block4.4 Gaussian checkpoint "
                "experiment configuration has no "
                "'model' section; Block4.5 had "
                "incorrectly treated it as an "
                "architecture configuration."
            ),

        "repair":
            (
                "Construct GaussianTrajectoryGRU "
                "from frozen Block4.3 deterministic "
                "architecture configuration, then "
                "strictly load frozen Block4.4 "
                "Gaussian state_dict."
            ),

        "deterministic_checkpoint_sha256":
            EXPECTED_DET_CHECKPOINT_SHA,

        "deterministic_state_sha256":
            EXPECTED_DET_STATE_SHA,

        "gaussian_checkpoint_sha256":
            EXPECTED_GAUSSIAN_CHECKPOINT_SHA,

        "gaussian_state_sha256":
            EXPECTED_GAUSSIAN_STATE_SHA,

        "strict_state_dict_load":
            True,

        "constructor_smoke":
            True,

        "regression":
            "76/76 PASS",

        "block43_modified":
            False,

        "block44_modified":
            False,

        "calibration_scientific_contract_modified":
            False,
    })

    print()
    print(
        "============================================================"
    )
    print(
        "MODEL CONFIG REPAIR = PASS"
    )
    print(
        "============================================================"
    )

    print(
        "Block4.3 modified         = NO"
    )

    print(
        "Block4.4 modified         = NO"
    )

    print(
        "Gaussian weights changed  = NO"
    )

    print(
        "calibration method changed= NO"
    )

    print(
        "7209 scan safe to resume  = YES"
    )

    # ========================================================
    # H. Automatically resume existing SAFE controller
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "RESUMING BLOCK 4.5 PART 2/2"
    )
    print(
        "============================================================"
    )

    process = subprocess.Popen(
        [
            sys.executable,
            str(
                SAFE_CONTROLLER
            ),
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    with RESUME_LOG.open(
        "w",
        encoding="utf-8",
    ) as log:

        if process.stdout is not None:
            for line in process.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )

                log.write(
                    line
                )

                log.flush()

    raw_code = (
        process.wait()
    )

    print()
    print(
        "safe controller raw code =",
        raw_code,
    )

    print(
        "terminal remains open    = YES"
    )


try:
    main()

except BaseException as exc:
    blocked(
        "unexpected_repair_controller_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
        (
            "Unexpected repair-controller "
            "failure. No shell exit is issued. "
            "Block4.3/4.4 remain untouched."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

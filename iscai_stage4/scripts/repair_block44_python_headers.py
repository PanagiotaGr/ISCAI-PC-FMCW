from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import traceback


ROOT = Path("/home/agni/waymo")
STAGE4 = ROOT / "iscai_stage4"

REPORT = (
    STAGE4
    / "reports/block44_environment_repair.json"
)

SAFE_REPORT = (
    STAGE4
    / "reports/block44_part1_safe_gate.json"
)

SAFE_RUNNER = (
    STAGE4
    / "scripts/run_block44_part1_safe.py"
)


def run(
    command,
    *,
    env=None,
):
    print()
    print(
        "$",
        " ".join(
            str(x)
            for x in command
        ),
        flush=True,
    )

    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    if completed.stdout:
        print(
            completed.stdout,
            end="",
            flush=True,
        )

    print(
        f"[return code = "
        f"{completed.returncode}]",
        flush=True,
    )

    return completed


def write_report(
    *,
    status,
    phase,
    detail,
    package=None,
):
    payload = {
        "stage": 4,
        "block": "4.4_part_1",
        "purpose":
            "Python development headers "
            "required by PyTorch/Triton",

        "status":
            status,

        "phase":
            phase,

        "detail":
            detail,

        "installed_package":
            package,

        "python_executable":
            sys.executable,

        "python_version":
            sys.version,

        "torch_environment_modified":
            False,

        "block43_modified":
            False,

        "block43_cache_regenerated":
            False,

        "probabilistic_training_started":
            False,
    }

    REPORT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def finish_blocked(
    phase,
    detail,
):
    write_report(
        status="BLOCKED",
        phase=phase,
        detail=detail,
    )

    print()
    print(
        "========================================"
    )
    print(
        "BLOCK 4.4 PART 1/2 = BLOCKED"
    )
    print(
        "========================================"
    )
    print(
        "phase  =",
        phase,
    )
    print(
        "reason =",
        detail,
    )
    print(
        "Block4.3 modified      = NO"
    )
    print(
        "cache regeneration     = NO"
    )
    print(
        "Gaussian training      = NOT STARTED"
    )
    print(
        "terminal remains open  = YES"
    )


def main():
    print(
        "========================================"
    )
    print(
        "BLOCK 4.4 — PYTHON HEADER REPAIR"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # 1. Freeze/check current Python + PyTorch environment.
    # --------------------------------------------------------

    import torch

    before = {
        "python":
            sys.version,

        "python_executable":
            sys.executable,

        "torch":
            torch.__version__,

        "torch_cuda":
            torch.version.cuda,

        "cuda_available":
            torch.cuda.is_available(),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
    }

    print()
    print(
        "Python =",
        sys.version.split()[0],
    )
    print(
        "torch  =",
        torch.__version__,
    )
    print(
        "CUDA   =",
        torch.version.cuda,
    )
    print(
        "GPU    =",
        before["gpu"],
    )

    if (
        torch.__version__
        !=
        "2.13.0+cu132"
    ):
        finish_blocked(
            "environment_before_repair",
            "Frozen PyTorch version changed.",
        )
        return

    if (
        torch.version.cuda
        !=
        "13.2"
    ):
        finish_blocked(
            "environment_before_repair",
            "Frozen CUDA runtime changed.",
        )
        return

    if not (
        torch.cuda.is_available()
    ):
        finish_blocked(
            "environment_before_repair",
            "CUDA is unexpectedly unavailable.",
        )
        return

    # --------------------------------------------------------
    # 2. Resolve exactly where Triton expects Python.h.
    # --------------------------------------------------------

    include_dir = Path(
        sysconfig.get_paths()[
            "include"
        ]
    )

    python_h = (
        include_dir
        /
        "Python.h"
    )

    print()
    print(
        "Python include dir =",
        include_dir,
    )
    print(
        "Python.h expected  =",
        python_h,
    )
    print(
        "Python.h exists    =",
        python_h.is_file(),
    )

    installed_package = None

    # --------------------------------------------------------
    # 3. Install ONLY development headers if needed.
    #
    # No TensorFlow/JAX.
    # No NumPy downgrade.
    # No PyTorch reinstall.
    # --------------------------------------------------------

    if not python_h.is_file():

        apt_get = shutil.which(
            "apt-get"
        )

        if apt_get is None:
            finish_blocked(
                "python_headers",
                (
                    "Python.h is absent and "
                    "apt-get is unavailable."
                ),
            )
            return

        if os.geteuid() != 0:
            finish_blocked(
                "python_headers",
                (
                    "Python.h is absent and this "
                    "repair needs root privileges "
                    "for the development-header "
                    "package."
                ),
            )
            return

        env = dict(
            os.environ
        )

        env[
            "DEBIAN_FRONTEND"
        ] = "noninteractive"

        print()
        print(
            "Python.h is missing."
        )
        print(
            "Installing development headers only."
        )

        update = run(
            [
                apt_get,
                "update",
            ],
            env=env,
        )

        if update.returncode != 0:
            finish_blocked(
                "apt_update",
                (
                    "apt-get update failed. "
                    "No scientific artifact "
                    "was modified."
                ),
            )
            return

        candidates = (
            "python3.13-dev",
            "libpython3.13-dev",
        )

        for candidate in candidates:

            simulation = run(
                [
                    apt_get,
                    "-s",
                    "install",
                    "--no-install-recommends",
                    candidate,
                ],
                env=env,
            )

            if (
                simulation.returncode
                !=
                0
            ):
                continue

            install = run(
                [
                    apt_get,
                    "install",
                    "-y",
                    "--no-install-recommends",
                    candidate,
                ],
                env=env,
            )

            if (
                install.returncode
                ==
                0
            ):
                installed_package = (
                    candidate
                )
                break

        if installed_package is None:
            finish_blocked(
                "python_headers",
                (
                    "Neither python3.13-dev "
                    "nor libpython3.13-dev "
                    "could be installed."
                ),
            )
            return

    # sysconfig may have been evaluated before installation,
    # but the expected filesystem path is unchanged.
    if not python_h.is_file():
        finish_blocked(
            "python_headers",
            (
                f"Development package was "
                f"processed but {python_h} "
                f"is still missing."
            ),
        )
        return

    print()
    print(
        "Python.h             = PASS"
    )
    print(
        "installed package    =",
        (
            installed_package
            if installed_package
            is not None
            else
            "ALREADY_PRESENT"
        ),
    )

    # --------------------------------------------------------
    # 4. Verify environment did not change scientifically.
    # --------------------------------------------------------

    import torch as torch_after

    after = {
        "python":
            sys.version,

        "python_executable":
            sys.executable,

        "torch":
            torch_after.__version__,

        "torch_cuda":
            torch_after.version.cuda,

        "cuda_available":
            torch_after.cuda.is_available(),

        "gpu":
            (
                torch_after.cuda
                .get_device_name(0)
                if torch_after
                .cuda
                .is_available()
                else None
            ),
    }

    for field in (
        "python_executable",
        "torch",
        "torch_cuda",
        "cuda_available",
        "gpu",
    ):
        if (
            before[field]
            !=
            after[field]
        ):
            finish_blocked(
                "environment_after_repair",
                (
                    f"Unexpected environment "
                    f"change in {field}: "
                    f"{before[field]!r} -> "
                    f"{after[field]!r}"
                ),
            )
            return

    print(
        "PyTorch unchanged    = PASS"
    )
    print(
        "CUDA unchanged       = PASS"
    )

    # --------------------------------------------------------
    # 5. Direct Triton driver smoke.
    #
    # This deliberately executes the exact kind of native
    # helper compilation that previously failed at Python.h.
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "TRITON NATIVE DRIVER SMOKE"
    )
    print(
        "========================================"
    )

    triton_smoke = run(
        [
            sys.executable,
            "-c",
            (
                "from triton.runtime.driver "
                "import driver; "
                "d=driver.active; "
                "print('active driver =', "
                "type(d).__name__); "
                "print('device =', "
                "d.get_current_device())"
            ),
        ],
        env=os.environ.copy(),
    )

    if (
        triton_smoke.returncode
        !=
        0
    ):
        finish_blocked(
            "triton_native_driver",
            (
                "Python.h is now present, "
                "but Triton's native driver "
                "still cannot compile/load. "
                "The full compiler output is "
                "printed immediately above."
            ),
        )
        return

    print(
        "Triton native driver = PASS"
    )

    # --------------------------------------------------------
    # 6. Reproduce a CUDA autograd path.
    # --------------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "CUDA AUTOGRAD / TRITON SMOKE"
    )
    print(
        "========================================"
    )

    cuda_smoke = run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "x=torch.randn("
                "32,4,3,"
                "device='cuda',"
                "requires_grad=True); "
                "A=torch.randn("
                "32,4,3,3,"
                "device='cuda',"
                "requires_grad=True); "
                "y=(A @ "
                "x.unsqueeze(-1))"
                ".square().mean(); "
                "y.backward(); "
                "torch.cuda.synchronize(); "
                "assert x.grad is not None; "
                "assert A.grad is not None; "
                "assert torch.isfinite("
                "x.grad).all(); "
                "assert torch.isfinite("
                "A.grad).all(); "
                "print('CUDA backward = PASS')"
            ),
        ],
        env=os.environ.copy(),
    )

    if (
        cuda_smoke.returncode
        !=
        0
    ):
        finish_blocked(
            "cuda_autograd",
            (
                "Development headers are fixed, "
                "but the CUDA backward smoke "
                "still fails. The exact output "
                "is printed above."
            ),
        )
        return

    print(
        "CUDA/Triton backward = PASS"
    )

    # --------------------------------------------------------
    # 7. Mark environment repair PASS.
    # --------------------------------------------------------

    write_report(
        status="PASS",
        phase="complete",
        detail=(
            "Python development headers "
            "available and Triton native "
            "driver/CUDA backward verified."
        ),
        package=installed_package,
    )

    # --------------------------------------------------------
    # 8. Automatically rerun the full safe Block4.4 Part1.
    # --------------------------------------------------------

    if not SAFE_RUNNER.is_file():
        finish_blocked(
            "safe_runner",
            (
                "Environment repair passed, "
                "but run_block44_part1_safe.py "
                "is missing."
            ),
        )
        return

    print()
    print(
        "========================================"
    )
    print(
        "FULL BLOCK 4.4 PART 1/2 RECHECK"
    )
    print(
        "========================================"
    )

    safe = run(
        [
            sys.executable,
            str(
                SAFE_RUNNER
            ),
        ],
        env=os.environ.copy(),
    )

    # The safe runner is intentionally expected
    # to return zero even for scientific BLOCKED,
    # therefore status comes from its JSON report.
    if not SAFE_REPORT.is_file():
        finish_blocked(
            "safe_recheck",
            (
                "Safe runner finished but "
                "did not create its structured "
                "status report."
            ),
        )
        return

    safe_status = json.loads(
        SAFE_REPORT.read_text(
            encoding="utf-8"
        )
    )

    status = safe_status.get(
        "status"
    )

    print()
    print(
        "========================================"
    )
    print(
        "BLOCK 4.4 PART 1/2 FINAL STATUS"
    )
    print(
        "========================================"
    )

    print(
        "Python.h               = PASS"
    )
    print(
        "Triton native driver   = PASS"
    )
    print(
        "CUDA/Triton backward   = PASS"
    )
    print(
        "Block4.3 modified      = NO"
    )
    print(
        "cache regeneration     = NO"
    )
    print(
        "Gaussian training      = NOT STARTED"
    )
    print(
        "scientific STATUS      =",
        status,
    )

    if status == "BLOCKED":
        print(
            "phase                  =",
            safe_status.get(
                "phase"
            ),
        )
        print(
            "reason                 =",
            safe_status.get(
                "exception"
            ),
        )
        print(
            "recovery               =",
            safe_status.get(
                "recovery"
            ),
        )

    print(
        "terminal remains open  = YES"
    )


try:
    main()

except BaseException as exc:
    # Absolute last-resort catcher.
    #
    # Do not propagate a Python exception to bash.
    print()
    print(
        "========================================"
    )
    print(
        "BLOCK 4.4 REPAIR = CONTROLLED BLOCKED"
    )
    print(
        "========================================"
    )
    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )
    print()
    traceback.print_exc()
    print()
    print(
        "Block4.3 modified      = NO"
    )
    print(
        "Gaussian training      = NOT STARTED"
    )
    print(
        "terminal remains open  = YES"
    )

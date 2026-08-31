from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import torch

from iscai_stage4.ml import (
    DeterministicTrajectoryGRU,
    masked_mse_loss,
    set_global_determinism,
)


ROOT = Path(
    "/home/agni/waymo"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

BLOCK42 = (
    STAGE4
    / "reports/"
      "block42_neural_sample_gate.json"
)

FIT = (
    STAGE4
    / "artifacts/block41/fit.jsonl"
)

DEV = (
    STAGE4
    / "artifacts/block41/development.jsonl"
)

REPORT = (
    STAGE4
    / "reports/"
      "block43_preflight.json"
)

EXPECTED_BLOCK42_SHA = (
    "4f6ef90bd3455bf55f100561d66ef935"
    "49d8472112b85fce45930ec169f89fd0"
)

EXPECTED_FIT_SHA = (
    "284a61c877d937deb07137345beaabf3165690138785f9cc41f81655066d5276"
)

EXPECTED_DEV_SHA = (
    "e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c"
)


def sha256_file(path):
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


block42 = json.loads(
    BLOCK42.read_text(
        encoding="utf-8"
    )
)

if block42[
    "status"
] != "PASS":
    raise SystemExit(
        "FAIL: Block4.2 not PASS."
    )

if (
    block42[
        "implementation"
    ]["sha256"]
    !=
    EXPECTED_BLOCK42_SHA
):
    raise SystemExit(
        "FAIL: frozen Block4.2 "
        "implementation SHA mismatch."
    )

if (
    sha256_file(FIT)
    !=
    EXPECTED_FIT_SHA
):
    raise SystemExit(
        "FAIL: fit manifest changed."
    )

if (
    sha256_file(DEV)
    !=
    EXPECTED_DEV_SHA
):
    raise SystemExit(
        "FAIL: development "
        "manifest changed."
    )

if not torch.cuda.is_available():
    raise SystemExit(
        "FAIL: Stage4 frozen CUDA "
        "device unavailable."
    )

if (
    torch.version.cuda
    !=
    "13.2"
):
    raise SystemExit(
        "FAIL: expected PyTorch "
        "CUDA runtime 13.2."
    )

set_global_determinism(
    20260820
)

device = torch.device(
    "cuda:0"
)

model = (
    DeterministicTrajectoryGRU()
    .to(
        device
    )
)

target = torch.randn(
    16,
    11,
    14,
    device=device,
)

neighbors = torch.randn(
    16,
    8,
    11,
    14,
    device=device,
)

neighbor_mask = torch.ones(
    16,
    8,
    device=device,
)

map_context = torch.randn(
    16,
    10,
    device=device,
)

truth = torch.randn(
    16,
    4,
    3,
    device=device,
)

valid = torch.ones(
    16,
    4,
    device=device,
)

output = model(
    target,
    neighbors,
    neighbor_mask,
    map_context,
)

loss = masked_mse_loss(
    output,
    truth,
    valid,
)

loss.backward()

torch.cuda.synchronize()

if not torch.isfinite(
    output
).all():
    raise RuntimeError(
        "CUDA GRU forward "
        "produced non-finite values."
    )

if not torch.isfinite(
    loss
):
    raise RuntimeError(
        "CUDA GRU loss "
        "is non-finite."
    )

grads = [
    parameter.grad
    for parameter
    in model.parameters()
    if parameter.grad
    is not None
]

if not grads:
    raise RuntimeError(
        "CUDA GRU backward "
        "produced no gradients."
    )

if not all(
    bool(
        torch.isfinite(
            gradient
        ).all()
    )
    for gradient
    in grads
):
    raise RuntimeError(
        "CUDA GRU backward "
        "produced non-finite gradients."
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

count = (
    int(
        match.group(1)
    )
    if match
    else None
)

if (
    test.returncode != 0
    or
    count != 44
):
    print(
        combined
    )

    raise SystemExit(
        "FAIL: expected Stage4 "
        "preflight regression 44/44."
    )


report = {
    "stage": 4,
    "block": "4.3_preflight",
    "status": "PASS",

    "upstream": {
        "block42":
            "PASS",

        "block42_implementation_sha256":
            EXPECTED_BLOCK42_SHA,

        "fit_manifest_sha256":
            EXPECTED_FIT_SHA,

        "development_manifest_sha256":
            EXPECTED_DEV_SHA,
    },

    "environment": {
        "torch":
            torch.__version__,

        "cuda_runtime":
            torch.version.cuda,

        "cuda_available":
            True,

        "gpu":
            torch.cuda
            .get_device_name(0),
    },

    "cuda_GRU": {
        "forward":
            True,

        "masked_loss":
            True,

        "backward":
            True,

        "finite":
            True,

        "deterministic_policy":
            True,
    },

    "regression": {
        "tests_passed":
            44,

        "tests_total":
            44,
    },
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


print(
    "===== BLOCK 4.3 PREFLIGHT ====="
)

print(
    "Block4.2 upstream        = PASS"
)

print(
    "fit/dev manifests        = PASS"
)

print(
    "torch                    =",
    torch.__version__,
)

print(
    "CUDA                     =",
    torch.version.cuda,
)

print(
    "GPU                      =",
    torch.cuda
    .get_device_name(0),
)

print(
    "CUDA GRU forward         = PASS"
)

print(
    "CUDA masked loss         = PASS"
)

print(
    "CUDA GRU backward        = PASS"
)

print(
    "deterministic policy     = PASS"
)

print(
    "full Stage4 regression   = 44 / 44 PASS"
)

print(
    "STATUS = PASS"
)

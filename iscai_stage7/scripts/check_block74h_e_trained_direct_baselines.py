from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path("/home/agni/waymo")
S7 = ROOT / "iscai_stage7"

AUTH = (
    S7 / "configs/"
    "stage7_block74h_e_trained_direct_baseline_authority.json"
)

REPORT = (
    S7 / "reports/block74h/"
    "block74h_e_trained_direct_baseline_freeze.json"
)


class FailClosed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FailClosed(message)


def sha(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def state_sha(state_dict):
    h = hashlib.sha256()

    for key in sorted(state_dict):
        t = (
            state_dict[key]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode("utf-8")
        )
        h.update(
            str(t.dtype).encode("utf-8")
        )
        h.update(
            json.dumps(
                list(t.shape)
            ).encode("utf-8")
        )
        h.update(
            t.numpy()
            .tobytes(order="C")
        )

    return h.hexdigest()


require(
    AUTH.is_file(),
    "H-E authority missing.",
)

require(
    REPORT.is_file(),
    "H-E report missing.",
)

authority = json.loads(
    AUTH.read_text(
        encoding="utf-8"
    )
)

report = json.loads(
    REPORT.read_text(
        encoding="utf-8"
    )
)


print("=" * 78)
print("STAGE 7 — BLOCK 7.4H-E")
print("INDEPENDENT TRAINED DIRECT-BASELINE AUTHORITY CHECKER")
print("=" * 78)


require(
    authority.get("status")
    ==
    "FROZEN_TRAINED_DIRECT_BASELINE_AUTHORITY",
    "Authority status mismatch.",
)

require(
    report.get("status")
    ==
    "PASS_TRAINED_DIRECT_BASELINE_FREEZE",
    "Freeze report status mismatch.",
)

require(
    report["authority_sha256"]
    ==
    sha(AUTH),
    "Authority file SHA mismatch.",
)


print()
print("===== A. BOUND PROVENANCE =====")

for name, info in (
    authority[
        "provenance"
    ].items()
):
    path = Path(
        info["path"]
    )

    require(
        path.is_file(),
        f"Missing bound provenance: {name}",
    )

    require(
        sha(path)
        ==
        info["sha256"],
        f"Bound provenance changed: {name}",
    )

    print(
        "PASS",
        name,
        info["sha256"],
    )


print()
print("===== B. CHECKPOINTS =====")

for task in (
    "direct_beam",
    "direct_ADB",
):
    info = authority[
        task
    ]

    path = Path(
        info["path"]
    )

    require(
        path.is_file(),
        f"Missing {task} checkpoint.",
    )

    require(
        sha(path)
        ==
        info["file_sha256"],
        f"{task}: file SHA changed.",
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        state_sha(
            checkpoint[
                "state_dict"
            ]
        )
        ==
        info["state_dict_sha256"],
        f"{task}: state SHA changed.",
    )

    require(
        int(
            checkpoint["epoch"]
        )
        ==
        int(
            info["best_epoch"]
        ),
        f"{task}: best epoch mismatch.",
    )

    require(
        abs(
            float(
                checkpoint[
                    "development_task_loss"
                ]
            )
            -
            float(
                info[
                    "best_development_loss"
                ]
            )
        )
        <=
        1e-15,
        f"{task}: best loss mismatch.",
    )

    print(
        task,
        "| epoch =",
        info["best_epoch"],
        "| development loss =",
        info[
            "best_development_loss"
        ],
        "| state SHA =",
        info[
            "state_dict_sha256"
        ],
    )


print()
print("===== C. EXACT FINAL OUTCOMES =====")

require(
    authority[
        "direct_beam"
    ][
        "best_epoch"
    ]
    ==
    20,
    "Beam best epoch changed.",
)

require(
    abs(
        authority[
            "direct_beam"
        ][
            "best_development_loss"
        ]
        -
        1.4727941442418981
    )
    <=
    1e-15,
    "Beam development CE changed.",
)

require(
    authority[
        "direct_beam"
    ][
        "optimizer_steps"
    ]
    ==
    120,
    "Beam optimizer steps changed.",
)

require(
    authority[
        "direct_ADB"
    ][
        "best_epoch"
    ]
    ==
    4,
    "ADB best epoch changed.",
)

require(
    abs(
        authority[
            "direct_ADB"
        ][
            "best_development_loss"
        ]
        -
        0.23317452110117323
    )
    <=
    1e-15,
    "ADB development BCE changed.",
)

require(
    authority[
        "direct_ADB"
    ][
        "optimizer_steps"
    ]
    ==
    1839024,
    "ADB optimizer steps changed.",
)

print(
    "beam = epoch20 / "
    "devCE=1.4727941442418981"
)

print(
    "ADB = epoch4 / "
    "devBCE=0.23317452110117323"
)


print()
print("===== D. REPLAY PROVENANCE =====")

replay = authority[
    "runtime_replay"
]

require(
    replay[
        "interrupted_reference_execution_occurred"
    ]
    is True,
    "Interrupted reference provenance missing.",
)

require(
    replay[
        "fresh_restart_from_original_frozen_seeds"
    ]
    is True,
    "Fresh replay provenance missing.",
)

require(
    replay[
        "reference_vs_accelerated_exact_optimizer_step_parity"
    ]
    is True,
    "Exact parity provenance missing.",
)

require(
    replay[
        "exact_parity_steps"
    ]
    ==
    8,
    "Parity step count changed.",
)

require(
    replay[
        "scientific_training_parameter_change"
    ]
    is False,
    "Scientific training parameter change detected.",
)

print("fresh replay = BOUND")
print("8-step exact parity = BOUND")
print("scientific parameter change = NO")


print()
print("===== E. SCIENTIFIC BOUNDARY =====")

boundary = authority[
    "scientific_boundary"
]

for key in (
    "calibration_partition_opened_in_this_block",
    "formal_N120_raw_data_opened",
    "formal_outcomes_opened",
    "Stage4_modified",
    "Stage5_modified",
    "Stage6_modified",
):
    require(
        boundary[key] is False,
        f"Boundary violation: {key}",
    )

require(
    boundary[
        "training_complete"
    ]
    is True,
    "Training not marked complete.",
)

require(
    boundary[
        "checkpoints_frozen"
    ]
    is True,
    "Checkpoints not frozen.",
)

require(
    boundary[
        "additional_fit_training_allowed"
    ]
    is False,
    "Additional training still allowed.",
)

require(
    boundary[
        "development_reselection_allowed"
    ]
    is False,
    "Development reselection still allowed.",
)

require(
    boundary[
        "hyperparameter_change_allowed"
    ]
    is False,
    "Hyperparameter changes still allowed.",
)


print()
print("=" * 78)
print("BLOCK 7.4H-E INDEPENDENT CHECKER = PASS")
print("TRAINED DIRECT BASELINES = IMMUTABLY FROZEN")
print("DIRECT BEAM BEST EPOCH = 20")
print("DIRECT ADB BEST EPOCH = 4")
print("ADDITIONAL TRAINING = FORBIDDEN")
print("DEVELOPMENT RESELECTION = FORBIDDEN")
print("CALIBRATION PARTITION = UNOPENED")
print("FORMAL N120 RAW DATA = SEALED")
print("NEXT = HELD-OUT DIRECT-BASELINE CALIBRATION BINDING")
print("=" * 78)

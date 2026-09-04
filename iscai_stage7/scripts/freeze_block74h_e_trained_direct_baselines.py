from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path("/home/agni/waymo")

S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
S7 = ROOT / "iscai_stage7"

PREREG = (
    S7 / "configs/"
    "stage7_block74h_c_direct_training_preregistration.json"
)

ACCEL = (
    S7 / "configs/"
    "stage7_block74h_d2_runtime_acceleration_binding.json"
)

TRAINER = (
    S7 / "scripts/"
    "run_block74h_d2_direct_baseline_training_accelerated.py"
)

TRAIN_REPORT = (
    S7 / "reports/block74h/"
    "block74h_d_direct_baseline_training.json"
)

BEAM = (
    S7 / "artifacts/block74h_direct/"
    "beam/best.pt"
)

ADB = (
    S7 / "artifacts/block74h_direct/"
    "adb/best.pt"
)

OUT = (
    S7 / "configs/"
    "stage7_block74h_e_trained_direct_baseline_authority.json"
)

REPORT = (
    S7 / "reports/block74h/"
    "block74h_e_trained_direct_baseline_freeze.json"
)

EXPECTED = {
    S4 / "reports/stage4_final_closure.json":
        "570da4feb918b1025b5e85cc919360d922b471c468c85b3844f13fb7774e7c2f",

    S5 / "reports/stage5_final_closure.json":
        "c83731948749f3ca20742aaac6b7474f53c755cbc8209dd31db969d229f60610",

    S6 / "reports/stage6_final_certificate.json":
        "e3870d8c93456f0bf6470467dcbf89ee4e75e7910c6a364c487d2f30e8163cfc",

    S6 / "artifacts/stage6_to_stage7_handoff.json":
        "7456e465043fa85e257fc4322159ee44c008992888aeb8d040d4cfeead9fb449",

    PREREG:
        "fc0726e3c0d964e48b49a63d775070b1487a3716f160813391881d47cccb1333",
}


class FailClosed(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise FailClosed(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def state_sha(state_dict) -> str:
    h = hashlib.sha256()

    for key in sorted(state_dict):
        t = (
            state_dict[key]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(
            json.dumps(
                list(t.shape)
            ).encode("utf-8")
        )
        h.update(
            t.numpy().tobytes(order="C")
        )

    return h.hexdigest()


def load_json(path: Path):
    require(
        path.is_file(),
        f"Missing JSON: {path}",
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(path)


print("=" * 78)
print("STAGE 7 — BLOCK 7.4H-E")
print("TRAINED DIRECT-BASELINE AUTHORITY FREEZE")
print("READBACK ONLY — NO TRAINING — NO CALIBRATION — FORMAL N120 SEALED")
print("=" * 78)


print()
print("===== A. FROZEN UPSTREAM =====")

for path, expected in EXPECTED.items():
    require(
        path.is_file(),
        f"Missing authority: {path}",
    )

    actual = sha(path)

    require(
        actual == expected,
        (
            f"Authority changed:\n"
            f"{path}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        ),
    )

    print("PASS", path, actual)


print()
print("===== B. H-D2 EXECUTION AUTHORITY =====")

require(
    ACCEL.is_file(),
    "Acceleration binding missing.",
)

require(
    TRAINER.is_file(),
    "Accelerated trainer missing.",
)

require(
    TRAIN_REPORT.is_file(),
    "H-D training report missing.",
)

accel = load_json(ACCEL)
training = load_json(TRAIN_REPORT)

require(
    accel.get("status")
    ==
    "PASS_EXACT_EQUIVALENT_RUNTIME_ACCELERATION",
    "H-D2 acceleration status changed.",
)

require(
    accel["exact_parity"]["optimizer_steps_checked"]
    == 8,
    "H-D2 exact parity coverage changed.",
)

for key in (
    "all_losses_exact",
    "all_gradients_exact",
    "all_parameters_exact",
    "all_optimizer_states_exact",
):
    require(
        accel["exact_parity"][key] is True,
        f"H-D2 parity failure: {key}",
    )

require(
    all(
        accel[
            "unchanged_scientific_execution"
        ].values()
    ),
    "Scientific execution changed in H-D2.",
)

require(
    training.get("status")
    ==
    "PASS_DIRECT_BASELINE_TRAINING_COMPLETE",
    "Direct training report not COMPLETE.",
)

require(
    training["trainer_sha256"]
    ==
    sha(TRAINER),
    "Executed trainer SHA mismatch.",
)

print("H-D2 runtime parity = PASS")
print("training report = COMPLETE")


print()
print("===== C. DEVELOPMENT-SELECTION OUTCOMES =====")

beam_result = training["direct_beam"]
adb_result = training["direct_ADB"]

require(
    beam_result["best_epoch"] == 20,
    "Beam best epoch changed.",
)

require(
    abs(
        beam_result["best_development_loss"]
        - 1.4727941442418981
    )
    <= 1e-15,
    "Beam best development CE changed.",
)

require(
    beam_result["optimizer_steps"] == 120,
    "Beam optimizer-step count changed.",
)

require(
    adb_result["best_epoch"] == 4,
    "ADB best epoch changed.",
)

require(
    abs(
        adb_result["best_development_loss"]
        - 0.23317452110117323
    )
    <= 1e-15,
    "ADB best development BCE changed.",
)

require(
    adb_result["optimizer_steps"] == 1839024,
    "ADB optimizer-step count changed.",
)

print(
    "direct beam | best epoch =",
    beam_result["best_epoch"],
    "| dev CE =",
    beam_result["best_development_loss"],
)

print(
    "direct ADB  | best epoch =",
    adb_result["best_epoch"],
    "| dev BCE =",
    adb_result["best_development_loss"],
)


print()
print("===== D. CHECKPOINT INTEGRITY =====")

checkpoint_info = {}

for task, path, result in (
    ("direct_beam", BEAM, beam_result),
    ("direct_ADB", ADB, adb_result),
):
    require(
        path.is_file(),
        f"Missing checkpoint: {path}",
    )

    file_sha = sha(path)

    require(
        file_sha
        ==
        result["checkpoint_sha256"],
        f"{task}: checkpoint file SHA mismatch.",
    )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        ckpt["task"] == task,
        f"{task}: checkpoint task changed.",
    )

    require(
        ckpt["preregistration_sha256"]
        ==
        sha(PREREG),
        f"{task}: preregistration mismatch.",
    )

    computed_state_sha = state_sha(
        ckpt["state_dict"]
    )

    require(
        computed_state_sha
        ==
        ckpt["state_dict_sha256"]
        ==
        result["state_dict_sha256"],
        f"{task}: state_dict SHA mismatch.",
    )

    require(
        int(ckpt["epoch"])
        ==
        int(result["best_epoch"]),
        f"{task}: checkpoint epoch mismatch.",
    )

    require(
        abs(
            float(
                ckpt["development_task_loss"]
            )
            -
            float(
                result["best_development_loss"]
            )
        )
        <=
        1e-15,
        f"{task}: checkpoint development loss mismatch.",
    )

    require(
        ckpt["formal_N120_raw_data_opened"]
        is False,
        f"{task}: formal seal violation.",
    )

    require(
        ckpt["calibration_partition_opened"]
        is False,
        f"{task}: calibration opened during training.",
    )

    checkpoint_info[task] = {
        "path": str(path),
        "file_sha256": file_sha,
        "state_dict_sha256": computed_state_sha,
        "best_epoch": int(
            result["best_epoch"]
        ),
        "best_development_loss": float(
            result["best_development_loss"]
        ),
        "optimizer_steps": int(
            result["optimizer_steps"]
        ),
        "trainable_parameter_count": int(
            result["trainable_parameter_count"]
        ),
    }

    print(
        "PASS",
        task,
        file_sha,
        computed_state_sha,
    )


print()
print("===== E. SCIENTIFIC BOUNDARY =====")

boundary = training[
    "scientific_boundary"
]

require(
    boundary["training_population"]
    ==
    "fit only",
    "Training population changed.",
)

require(
    boundary[
        "model_selection_population"
    ]
    ==
    "development only",
    "Model selection changed.",
)

for key in (
    "calibration_partition_opened",
    "formal_N120_raw_data_opened",
    "formal_outcomes_opened",
    "future_GT_as_inference_input",
    "Stage4_modified",
    "Stage5_modified",
    "Stage6_modified",
    "label_shards_modified",
    "post_outcome_hyperparameter_change",
):
    require(
        boundary[key] is False,
        f"Boundary violation: {key}",
    )

require(
    boundary[
        "post_interruption_exact_equivalent_execution_replay"
    ]
    is True,
    "H-D2 replay provenance missing.",
)

require(
    boundary[
        "observed_outcomes_used_for_hyperparameter_change"
    ]
    is False,
    "Outcome-driven hyperparameter change detected.",
)

print("training population = FIT ONLY")
print("model selection = DEVELOPMENT ONLY")
print("formal N120 = SEALED")
print("calibration partition = UNOPENED")
print("post-outcome hyperparameter change = NO")


payload = {
    "stage":
        7,

    "block":
        "7.4H-E",

    "status":
        "FROZEN_TRAINED_DIRECT_BASELINE_AUTHORITY",

    "purpose":
        (
            "immutable post-training authority for "
            "development-selected direct beam and direct ADB baselines"
        ),

    "provenance": {
        "preregistration": {
            "path": str(PREREG),
            "sha256": sha(PREREG),
        },

        "runtime_acceleration_binding": {
            "path": str(ACCEL),
            "sha256": sha(ACCEL),
        },

        "executed_trainer": {
            "path": str(TRAINER),
            "sha256": sha(TRAINER),
        },

        "training_report": {
            "path": str(TRAIN_REPORT),
            "sha256": sha(TRAIN_REPORT),
        },
    },

    "direct_beam":
        checkpoint_info[
            "direct_beam"
        ],

    "direct_ADB":
        checkpoint_info[
            "direct_ADB"
        ],

    "selection_semantics": {
        "direct_beam":
            "minimum development categorical cross entropy; earliest epoch on ties",

        "direct_ADB":
            "minimum development binary cross entropy; earliest epoch on ties",

        "formal_used_for_selection":
            False,

        "calibration_used_for_selection":
            False,
    },

    "runtime_replay": {
        "interrupted_reference_execution_occurred":
            True,

        "fresh_restart_from_original_frozen_seeds":
            True,

        "reference_vs_accelerated_exact_optimizer_step_parity":
            True,

        "exact_parity_steps":
            8,

        "scientific_training_parameter_change":
            False,

        "partial_reference_run":
            "superseded_by_exact_equivalent_fresh_replay",
    },

    "scientific_boundary": {
        "training_complete":
            True,

        "checkpoints_frozen":
            True,

        "additional_fit_training_allowed":
            False,

        "development_reselection_allowed":
            False,

        "hyperparameter_change_allowed":
            False,

        "architecture_change_allowed":
            False,

        "label_change_allowed":
            False,

        "calibration_partition_opened_in_this_block":
            False,

        "formal_N120_raw_data_opened":
            False,

        "formal_outcomes_opened":
            False,

        "Stage4_modified":
            False,

        "Stage5_modified":
            False,

        "Stage6_modified":
            False,
    },

    "next": {
        "allowed":
            "held-out calibration binding/execution for trained direct baselines",

        "formal_evaluation":
            "NOT_YET_ALLOWED",

        "retraining":
            "FORBIDDEN",
    },
}


canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")

payload["content_sha256"] = (
    hashlib.sha256(
        canonical
    ).hexdigest()
)

atomic_json(
    OUT,
    payload,
)

atomic_json(
    REPORT,
    {
        "status":
            "PASS_TRAINED_DIRECT_BASELINE_FREEZE",

        "authority_path":
            str(OUT),

        "authority_sha256":
            sha(OUT),

        "authority_content_sha256":
            payload["content_sha256"],

        "beam_checkpoint_sha256":
            checkpoint_info[
                "direct_beam"
            ][
                "file_sha256"
            ],

        "ADB_checkpoint_sha256":
            checkpoint_info[
                "direct_ADB"
            ][
                "file_sha256"
            ],

        "formal_N120_raw_data_opened":
            False,

        "calibration_partition_opened":
            False,

        "retraining_allowed":
            False,
    },
)


print()
print("===== F. FROZEN AUTHORITY =====")
print("authority =", OUT)
print("authority SHA256 =", sha(OUT))
print(
    "content SHA256 =",
    payload["content_sha256"],
)
print(
    "training report SHA256 =",
    sha(TRAIN_REPORT),
)
print(
    "acceleration binding SHA256 =",
    sha(ACCEL),
)

print()
print("=" * 78)
print("BLOCK 7.4H-E FREEZE = PASS")
print("DIRECT BEAM CHECKPOINT = FROZEN")
print("DIRECT ADB CHECKPOINT = FROZEN")
print("RETRAINING / RESELECTION = FORBIDDEN")
print("CALIBRATION PARTITION = STILL UNOPENED")
print("FORMAL N120 RAW DATA = SEALED")
print("NEXT = INDEPENDENT H-E CHECKER")
print("=" * 78)

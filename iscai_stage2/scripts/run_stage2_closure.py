from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path("/home/agni/waymo")
STAGE1 = ROOT / "iscai_stage1"
STAGE2 = ROOT / "iscai_stage2"

VALIDATION_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
    "selected_validation.jsonl"
)

EXPECTED_VALIDATION_SHA256 = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

STAGE1_CLOSURE = (
    STAGE1
    / "reports/stage1a/"
    "stage1a_closure_report.json"
)

OUTPUT = (
    STAGE2
    / "reports/stage2_closure_report.json"
)


REQUIRED_REPORTS = {
    "block1c_real_ideal_observations":
        STAGE2
        / "reports/block1c_real_ideal_observations.json",

    "block2c_clean_crlb_smoke":
        STAGE2
        / "reports/block2c_clean_crlb_smoke.json",

    "block3d_degraded_scene_smoke":
        STAGE2
        / "reports/block3d_degraded_scene_smoke.json",

    "block3e_measurement_mc_gate":
        STAGE2
        / "reports/block3e_measurement_mc_gate.json",

    "block3f1_multiclass_extension_discovery":
        STAGE2
        / "reports/block3f1_multiclass_extension_discovery.json",

    "block3f_final_multiclass_coverage":
        STAGE2
        / "reports/block3f_final_multiclass_coverage_gate.json",
}


EXPECTED_EXTENSION = {
    "TYPE_CYCLIST": {
        "manifest_index_1_based": 5,
        "scenario_id": "75ae707721eb23b4",
    },

    "TYPE_PEDESTRIAN": {
        "manifest_index_1_based": 9,
        "scenario_id": "52dafd686fe77b21",
    },
}


EXPECTED_COVERAGE_SCENARIOS = [
    "b85e1bd6cc8e74c0",
    "4d82fec943ddaa44",
    "bbc29ed5e271f29b",
    "75ae707721eb23b4",
    "52dafd686fe77b21",
]


EXPECTED_MULTICLASS_COUNTS = {
    "TYPE_VEHICLE": {
        "causal_actors": 171,
        "clean_valid_measurements": 1434,
        "degraded_true_measurements": 1434,
    },

    "TYPE_PEDESTRIAN": {
        "causal_actors": 1,
        "clean_valid_measurements": 1,
        "degraded_true_measurements": 1,
    },

    "TYPE_CYCLIST": {
        "causal_actors": 3,
        "clean_valid_measurements": 17,
        "degraded_true_measurements": 17,
    },
}


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_json(
    path: Path,
) -> dict:
    require(
        path.exists(),
        f"Missing required file: {path}",
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def status_is_pass(
    report: dict,
) -> bool:
    value = str(
        report.get("status", "")
    ).upper()

    return value.startswith("PASS")


def run_regression(
    *,
    repo: Path,
    expected_tests: int,
) -> dict:
    env = os.environ.copy()

    required_pythonpath = [
        str(
            ROOT
            / "iscai_stage0/src"
        ),
        str(
            ROOT
            / "iscai_stage1/src"
        ),
        str(
            ROOT
            / "iscai_stage2/src"
        ),
    ]

    existing = env.get(
        "PYTHONPATH",
        "",
    )

    if existing:
        required_pythonpath.append(
            existing
        )

    env["PYTHONPATH"] = os.pathsep.join(
        required_pythonpath
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
    )

    combined = (
        result.stdout
        + "\n"
        + result.stderr
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        combined,
    )

    if (
        result.returncode != 0
        or match is None
    ):
        print(combined)
        raise RuntimeError(
            f"Regression failed in {repo}"
        )

    count = int(
        match.group(1)
    )

    if count != expected_tests:
        print(combined)
        raise RuntimeError(
            f"Unexpected test count in {repo}: "
            f"{count} != {expected_tests}"
        )

    if not re.search(
        r"^OK$",
        combined,
        flags=re.MULTILINE,
    ):
        print(combined)
        raise RuntimeError(
            f"Regression did not end in OK: {repo}"
        )

    return {
        "status": "PASS",
        "tests": count,
    }


def implementation_tree_sha256() -> tuple[
    str,
    list[str],
]:
    """
    Hash Stage-2 implementation inputs.

    Reports are intentionally excluded because they are
    outputs of the implementation and closure process.
    __pycache__ / .pyc are also excluded.
    """

    roots = [
        STAGE2 / "src",
        STAGE2 / "tests",
        STAGE2 / "scripts",
    ]

    files = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix == ".pyc":
                continue

            files.append(path)

    for name in (
        "README.md",
        "pyproject.toml",
    ):
        path = STAGE2 / name

        if path.exists():
            files.append(path)

    files = sorted(
        set(files),
        key=lambda p: str(
            p.relative_to(STAGE2)
        ),
    )

    h = hashlib.sha256()

    relative_paths = []

    for path in files:
        relative = str(
            path.relative_to(STAGE2)
        )

        relative_paths.append(
            relative
        )

        h.update(
            relative.encode("utf-8")
        )
        h.update(b"\0")

        with path.open("rb") as f:
            for chunk in iter(
                lambda: f.read(
                    1024 * 1024
                ),
                b"",
            ):
                h.update(chunk)

        h.update(b"\0")

    return (
        h.hexdigest(),
        relative_paths,
    )


# ------------------------------------------------------------
# Frozen validation manifest integrity
# ------------------------------------------------------------

validation_sha = sha256_file(
    VALIDATION_MANIFEST
)

require(
    validation_sha
    == EXPECTED_VALIDATION_SHA256,
    "Frozen validation manifest SHA mismatch.",
)


# ------------------------------------------------------------
# Frozen Stage-1 upstream
# ------------------------------------------------------------

stage1_closure = load_json(
    STAGE1_CLOSURE
)

require(
    stage1_closure.get("status")
    == "PASS_COMPLETE",
    "Stage-1A upstream closure is not PASS_COMPLETE.",
)

require(
    stage1_closure.get(
        "artifact_semantics"
    )
    == "causal_womd_annotation_upstream",
    "Unexpected Stage-1 artifact semantics.",
)

require(
    stage1_closure.get(
        "sensor_realistic"
    )
    is False,
    "Unexpected Stage-1 sensor-realistic claim.",
)


# ------------------------------------------------------------
# Regression
# ------------------------------------------------------------

stage1_regression = run_regression(
    repo=STAGE1,
    expected_tests=52,
)

stage2_regression = run_regression(
    repo=STAGE2,
    expected_tests=83,
)


# ------------------------------------------------------------
# Required gate reports + hashes
# ------------------------------------------------------------

reports = {}
report_hashes = {}

for name, path in REQUIRED_REPORTS.items():
    report = load_json(
        path
    )

    require(
        status_is_pass(report),
        f"Required gate did not PASS: {name}",
    )

    reports[name] = report

    report_hashes[name] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": report["status"],
    }


# ------------------------------------------------------------
# Block 3D: actual degraded behavior
# ------------------------------------------------------------

block3d = reports[
    "block3d_degraded_scene_smoke"
]

require(
    block3d["true_detections"] == 37,
    "Block 3D true-detection regression changed.",
)

require(
    block3d["missed_detections"] == 49,
    "Block 3D missed-detection regression changed.",
)

require(
    block3d["false_alarms"] == 20,
    "Block 3D false-alarm regression changed.",
)

require(
    block3d["algorithm_detection_count"] == 57,
    "Block 3D algorithm detection count changed.",
)

require(
    block3d["reproducible"] is True,
    "Block 3D reproducibility failed.",
)

require(
    block3d[
        "future_mutation_hash_identical"
    ]
    is True,
    "Block 3D future-mutation gate failed.",
)

require(
    block3d["measured_fmcw"] is False,
    "Block 3D incorrectly claims measured FMCW.",
)


# ------------------------------------------------------------
# Block 3E: covariance / CRLB Monte-Carlo
# ------------------------------------------------------------

block3e = reports[
    "block3e_measurement_mc_gate"
]

require(
    block3e["samples_per_snr"] == 30000,
    "Block 3E sample count changed.",
)

require(
    block3e[
        "crlb_snr_scaling_verified"
    ]
    is True,
    "CRLB SNR scaling gate failed.",
)

require(
    block3e["measured_fmcw"] is False,
    "Block 3E incorrectly claims measured FMCW.",
)


for snr_report in block3e["reports"]:
    for key in (
        "range_stats",
        "radial_velocity_stats",
        "azimuth_stats",
        "elevation_stats",
    ):
        stats = snr_report[key]

        require(
            abs(
                stats["normalized_mean"]
            ) <= 0.03,
            f"3E normalized mean failed: {key}",
        )

        require(
            abs(
                stats["normalized_rmse"]
                - 1.0
            ) <= 0.03,
            f"3E normalized RMSE failed: {key}",
        )

        require(
            abs(
                stats["normalized_std"]
                - 1.0
            ) <= 0.03,
            f"3E normalized std failed: {key}",
        )


# ------------------------------------------------------------
# Block 3F.1: deterministic extension selection
# ------------------------------------------------------------

block3f1 = reports[
    "block3f1_multiclass_extension_discovery"
]

require(
    block3f1["manifest_sha256"]
    == EXPECTED_VALIDATION_SHA256,
    "3F.1 manifest SHA mismatch.",
)

require(
    block3f1["missing"] == [],
    "3F.1 still has missing classes.",
)

semantics = block3f1[
    "selection_semantics"
]

require(
    semantics[
        "future_actor_states_used_for_selection"
    ]
    is False,
    "3F.1 used future actor states.",
)

require(
    semantics[
        "future_labels_used_for_selection"
    ]
    is False,
    "3F.1 used future labels.",
)

require(
    semantics["cherry_picking"]
    is False,
    "3F.1 cherry-picking flag changed.",
)

require(
    semantics[
        "first_qualifying_scenario_per_class"
    ]
    is True,
    "3F.1 first-qualifying semantics changed.",
)


for class_name, expected in (
    EXPECTED_EXTENSION.items()
):
    actual = block3f1[
        "found"
    ][class_name]

    require(
        actual[
            "manifest_record_index_1_based"
        ]
        == expected[
            "manifest_index_1_based"
        ],
        f"{class_name} manifest index changed.",
    )

    require(
        actual["scenario_id"]
        == expected["scenario_id"],
        f"{class_name} scenario ID changed.",
    )


# ------------------------------------------------------------
# Block 3F final multiclass coverage
# ------------------------------------------------------------

block3f = reports[
    "block3f_final_multiclass_coverage"
]

require(
    block3f[
        "coverage_scenario_ids"
    ]
    == EXPECTED_COVERAGE_SCENARIOS,
    "Final multiclass coverage scenario set changed.",
)

require(
    block3f[
        "algorithm_input_truth_free"
    ]
    is True,
    "Final multiclass algorithm input is not truth-free.",
)

require(
    block3f["measured_fmcw"]
    is False,
    "Final multiclass gate claims measured FMCW.",
)


for class_name, expected in (
    EXPECTED_MULTICLASS_COUNTS.items()
):
    actual = block3f[
        "coverage"
    ][class_name]

    require(
        actual["pass"] is True,
        f"{class_name} coverage did not PASS.",
    )

    for key, expected_value in (
        expected.items()
    ):
        require(
            actual[key] == expected_value,
            f"{class_name} {key} changed: "
            f"{actual[key]} != {expected_value}",
        )


coverage_semantics = block3f[
    "coverage_semantics"
]

require(
    coverage_semantics[
        "future_actor_states_used_for_selection"
    ]
    is False,
    "Final coverage used future actor states.",
)

require(
    coverage_semantics[
        "future_labels_used_for_selection"
    ]
    is False,
    "Final coverage used future labels.",
)

require(
    coverage_semantics[
        "cherry_picking"
    ]
    is False,
    "Final coverage cherry-picking flag changed.",
)


# ------------------------------------------------------------
# Freeze implementation tree
# ------------------------------------------------------------

(
    implementation_sha,
    implementation_files,
) = implementation_tree_sha256()


# ------------------------------------------------------------
# Closure report
# ------------------------------------------------------------

payload = {
    "status": "COMPLETE_FROZEN",

    "stage": "Stage 2",

    "scope": (
        "PC-FMCW-like causal track observations, "
        "measurement covariance, configurable sensing "
        "SNR, Gaussian measurement corruption, "
        "SNR-dependent missed detections, unlabeled "
        "detection frames, false alarms/clutter, "
        "and Stage-2 measurement validation"
    ),

    "upstream": {
        "stage1_status":
            stage1_closure["status"],

        "stage1_artifact_semantics":
            stage1_closure[
                "artifact_semantics"
            ],

        "stage1_sensor_realistic":
            False,

        "canonical_validation_manifest":
            str(VALIDATION_MANIFEST),

        "canonical_validation_manifest_sha256":
            validation_sha,

        "canonical_validation_records":
            44097,
    },

    "regression": {
        "stage1_tests":
            stage1_regression,

        "stage2_tests":
            stage2_regression,
    },

    "required_gate_reports":
        report_hashes,

    "gates": {
        "real_womd_ideal_observations":
            True,

        "clean_crlb_conditioned_mode":
            True,

        "gaussian_measurement_corruption":
            True,

        "snr_dependent_detection_probability":
            True,

        "missed_detections":
            True,

        "unlabeled_algorithm_detection_interface":
            True,

        "false_alarms_clutter":
            True,

        "degraded_real_scene_smoke":
            True,

        "measurement_covariance_mc_consistency":
            True,

        "crlb_snr_scaling":
            True,

        "deterministic_multiclass_extension":
            True,

        "vehicle_pipeline_coverage":
            True,

        "pedestrian_pipeline_coverage":
            True,

        "cyclist_pipeline_coverage":
            True,

        "algorithm_truth_leakage":
            False,

        "future_state_selection":
            False,

        "future_label_selection":
            False,

        "cherry_picking":
            False,
    },

    "multiclass_coverage": (
        block3f["coverage"]
    ),

    "frozen_extension": (
        EXPECTED_EXTENSION
    ),

    "degraded_smoke": {
        "true_detections": 37,
        "missed_detections": 49,
        "false_alarms": 20,
        "algorithm_detections": 57,
    },

    "measurement_mc": {
        "samples_per_snr": 30000,
        "snr_values_db":
            block3e["snr_values_db"],

        "crlb_snr_scaling_verified":
            True,
    },

    "frozen_semantics": {
        "womd_lidar_is_measured_fmcw":
            False,

        "measured_fmcw":
            False,

        "radial_velocity_source": (
            "geometry_derived_from_causal_"
            "womd_trajectory_not_measured_"
            "pointwise_lidar_doppler"
        ),

        "range_velocity_covariance": (
            "frozen_part_a_eq7_crlb_"
            "conditioned_on_stage2_sensing_snr"
        ),

        "angular_covariance": (
            "explicit_stage2_angular_"
            "uncertainty_model"
        ),

        "measurement_uncertainty_separate_from_"
        "predictive_uncertainty":
            True,

        "truth_sidecar": (
            "evaluator_only_not_algorithm_input"
        ),

        "clean_mode": (
            "no_noise_no_misses_no_false_alarms"
        ),

        "degraded_mode": (
            "gaussian_noise_plus_misses_"
            "plus_false_alarms"
        ),
    },

    "limitations": [
        (
            "Part-A CRLB range/velocity covariance is "
            "a lower-bound baseline, not a claim of "
            "calibrated real automotive optical-sensor "
            "measurement error."
        ),
        (
            "The fixed-SNR, detection-probability and "
            "false-alarm parameters used in smoke gates "
            "are controlled explicit Stage-2 assumptions, "
            "not Part-A measurements or calibrated CFAR "
            "parameters."
        ),
        (
            "The multiclass gate establishes pipeline/"
            "interface coverage only. In particular, "
            "TYPE_PEDESTRIAN has one clean-valid "
            "measurement in the deterministic extension "
            "and therefore does not constitute statistical "
            "pedestrian performance validation."
        ),
        (
            "WOMD-LiDAR is not FMCW and supplies no "
            "measured point-wise Doppler/radial velocity."
        ),
    ],

    "implementation_freeze": {
        "sha256":
            implementation_sha,

        "file_count":
            len(implementation_files),

        "files":
            implementation_files,
    },

    "next_stage": (
        "Stage 3 classical forecasting baselines "
        "using the common Stage-2 observation interface"
    ),
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage2 final closure ====="
)

print(
    "Stage1 regression       =",
    stage1_regression["tests"],
    "tests PASS",
)

print(
    "Stage2 regression       =",
    stage2_regression["tests"],
    "tests PASS",
)

print(
    "validation manifest SHA = PASS"
)

print(
    "required gate reports   =",
    len(REQUIRED_REPORTS),
    "PASS",
)

print(
    "clean mode              = PASS"
)

print(
    "degraded mode           = PASS"
)

print(
    "measurement MC          = PASS"
)

print(
    "CRLB SNR scaling        = PASS"
)

print(
    "multiclass coverage     = PASS"
)

print(
    "algorithm truth leakage = NONE"
)

print(
    "future-state selection  = NO"
)

print(
    "measured FMCW           = NO"
)

print(
    "implementation files    =",
    len(implementation_files),
)

print(
    "implementation SHA256   =",
    implementation_sha,
)

print(
    "STATUS = COMPLETE_FROZEN"
)

print(
    "report =",
    OUTPUT,
)

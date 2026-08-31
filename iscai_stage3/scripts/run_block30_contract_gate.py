from __future__ import annotations

import hashlib
import json

from pathlib import Path

from iscai_stage3.contracts import (
    forbidden_algorithm_fields,
    resolve_stage2_algorithm_schema,
)


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)

STAGE2 = Path(
    "/home/agni/waymo/iscai_stage2"
)

DATA_PREP = Path(
    "/home/agni/waymo/iscai_data_prep"
)


EXPECTED_VALIDATION_SHA256 = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_STAGE2_CLOSURE_SHA256 = (
    "44b0961128ee13b97876c97ddfb91de1"
    "b9c01d2962413a412066fb93f31701b5"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def implementation_hash() -> tuple[
    str,
    int,
]:
    roots = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
    )

    extras = (
        ROOT / "README.md",
        ROOT / "docs/stage3_scope.md",
    )

    files = []

    for base in roots:
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix == ".pyc":
                continue

            files.append(path)

    for path in extras:
        if path.exists():
            files.append(path)

    files = sorted(
        set(files),
        key=lambda p: str(
            p.relative_to(ROOT)
        ),
    )

    h = hashlib.sha256()

    for path in files:
        relative = str(
            path.relative_to(ROOT)
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

    return h.hexdigest(), len(files)


def scan_stage_boundaries():
    forbidden_downstream = (
        "iscai_stage3_panagiota",
        "iscai_stage4",
        "iscai_stage5",
        "iscai_stage6",
        "iscai_stage7",
    )

    truth_forbidden = (
        "DetectionTruthEntry",
        "DetectionTruthSidecar",
        "EvaluatorTruthSequence",
        "evaluation.truth",
    )

    downstream_offenders = []
    truth_offenders = []

    src = ROOT / "src/iscai_stage3"

    for path in src.rglob("*.py"):
        text = path.read_text(
            encoding="utf-8"
        )

        if any(
            token in text
            for token in forbidden_downstream
        ):
            downstream_offenders.append(
                str(path.relative_to(ROOT))
            )

    for package in (
        "observations",
        "association",
        "baselines",
    ):
        directory = src / package

        if not directory.exists():
            continue

        for path in directory.rglob("*.py"):
            text = path.read_text(
                encoding="utf-8"
            )

            if any(
                token in text
                for token in truth_forbidden
            ):
                truth_offenders.append(
                    str(path.relative_to(ROOT))
                )

    return (
        sorted(downstream_offenders),
        sorted(truth_offenders),
    )


schema = resolve_stage2_algorithm_schema()

forbidden_fields = (
    forbidden_algorithm_fields()
)

if forbidden_fields:
    raise SystemExit(
        "FAIL: truth/identity fields leaked "
        f"into Stage2 algorithm interface: "
        f"{forbidden_fields}"
    )


validation_manifest = (
    DATA_PREP
    / "manifests"
    / "selected_validation.jsonl"
)

stage2_closure = (
    STAGE2
    / "reports"
    / "stage2_closure_report.json"
)


validation_sha = sha256_file(
    validation_manifest
)

stage2_closure_sha = sha256_file(
    stage2_closure
)


if (
    validation_sha
    != EXPECTED_VALIDATION_SHA256
):
    raise SystemExit(
        "FAIL: frozen validation manifest "
        "changed."
    )


if (
    stage2_closure_sha
    != EXPECTED_STAGE2_CLOSURE_SHA256
):
    raise SystemExit(
        "FAIL: frozen Stage2 closure "
        "report changed."
    )


config_path = (
    ROOT
    / "configs"
    / "stage3_contract.json"
)

config = json.loads(
    config_path.read_text(
        encoding="utf-8"
    )
)


if (
    config[
        "main_information_regime"
    ][
        "association_mode"
    ]
    != "estimated_association"
):
    raise SystemExit(
        "FAIL: main association mode "
        "is not estimated association."
    )


if not config[
    "main_information_regime"
][
    "same_observation_stream_across_baselines"
]:
    raise SystemExit(
        "FAIL: common observation-stream "
        "policy disabled."
    )


if config[
    "upstream"
][
    "future_information_allowed"
]:
    raise SystemExit(
        "FAIL: future information allowed."
    )


if not config[
    "upstream"
][
    "measurement_covariance_required"
]:
    raise SystemExit(
        "FAIL: Stage2 covariance is not "
        "required."
    )


if config[
    "mht"
][
    "meaning"
] != "Multidimensional Hough Transform":
    raise SystemExit(
        "FAIL: MHT meaning is incorrect."
    )


if config[
    "imm"
][
    "required_modes"
] != [
    "CV",
    "CA",
    "CTRV",
]:
    raise SystemExit(
        "FAIL: IMM modes are incorrect."
    )


downstream_offenders, truth_offenders = (
    scan_stage_boundaries()
)


if downstream_offenders:
    raise SystemExit(
        "FAIL: downstream/legacy imports: "
        f"{downstream_offenders}"
    )


if truth_offenders:
    raise SystemExit(
        "FAIL: evaluator truth entered "
        "algorithm packages: "
        f"{truth_offenders}"
    )


impl_sha, impl_count = (
    implementation_hash()
)


report = {
    "stage": 3,
    "block": "3.0",
    "status": "PASS",

    "canonical_root": str(ROOT),

    "legacy_stage3_dependency": False,

    "upstream": {
        "validation_manifest_sha256": (
            validation_sha
        ),
        "validation_manifest_frozen": True,

        "stage2_closure_sha256": (
            stage2_closure_sha
        ),
        "stage2_closure_frozen": True,

        "measured_fmcw": False
    },

    "algorithm_interface": {
        "source": (
            "Stage2 UnlabeledDetectionFrame"
        ),
        "truth_fields": [],
        "association_main": (
            "estimated_association"
        ),
        "oracle_modes": (
            "diagnostic_only"
        ),
        "same_observation_stream_required": (
            True
        )
    },

    "stage2_schema": {
        "detection_key_field": (
            schema.detection_key_field
        ),
        "range_field": (
            schema.range_field
        ),
        "radial_velocity_field": (
            schema.radial_velocity_field
        ),
        "azimuth_field": (
            schema.azimuth_field
        ),
        "elevation_field": (
            schema.elevation_field
        ),
        "covariance_field": (
            schema.covariance_field
        ),
        "frame_timestamp_field": (
            schema.frame_timestamp_field
        ),
        "frame_detections_field": (
            schema.frame_detections_field
        )
    },

    "mht_meaning": (
        "Multidimensional Hough Transform"
    ),

    "imm_modes": [
        "CV",
        "CA",
        "CTRV"
    ],

    "truth_boundary": {
        "algorithm_truth_leakage": "NONE",
        "evaluator_only": True
    },

    "stage_boundary": {
        "legacy_imports": [],
        "downstream_imports": []
    },

    "implementation": {
        "files": impl_count,
        "sha256": impl_sha
    }
}


report_path = (
    ROOT
    / "reports"
    / "block30_contract_gate.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== Stage3 Block 3.0 gate ====="
)

print(
    "validation manifest     = PASS"
)

print(
    "Stage2 closure          = PASS"
)

print(
    "Stage2 schema           = PASS"
)

print(
    "algorithm truth leakage = NONE"
)

print(
    "legacy dependency       = NO"
)

print(
    "downstream imports      = NONE"
)

print(
    "main association mode   = "
    "estimated_association"
)

print(
    "MHT meaning             = "
    "Multidimensional Hough Transform"
)

print(
    "IMM modes               = "
    "CV / CA / CTRV"
)

print(
    "implementation files    =",
    impl_count,
)

print(
    "implementation SHA256   =",
    impl_sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    report_path,
)

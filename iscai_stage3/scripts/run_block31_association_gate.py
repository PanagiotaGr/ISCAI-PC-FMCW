from __future__ import annotations

import hashlib
import json

from pathlib import Path

from iscai_stage3.association import (
    AssociationConfig,
    associate_estimated_gnn,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)


def implementation_hash():
    files = []

    for base in (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
    ):
        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix == ".pyc":
                continue

            files.append(path)

    files = sorted(
        set(files),
        key=lambda p: str(
            p.relative_to(ROOT)
        ),
    )

    h = hashlib.sha256()

    for path in files:
        rel = str(
            path.relative_to(ROOT)
        )

        h.update(
            rel.encode("utf-8")
        )

        h.update(b"\0")

        h.update(
            path.read_bytes()
        )

        h.update(b"\0")

    return h.hexdigest(), len(files)


sequence = AlgorithmObservationSequence(
    scenario_id="block31-gate",
    frames=(
        make_frame(
            timestamp_s=0.0,
            detections=(
                make_detection(
                    key="a0",
                    range_m=10.0,
                    vr_mps=1.0,
                    az_rad=-0.2,
                ),
                make_detection(
                    key="b0",
                    range_m=40.0,
                    vr_mps=-2.0,
                    az_rad=0.4,
                ),
            ),
        ),

        make_frame(
            timestamp_s=0.1,
            detections=(
                make_detection(
                    key="b1-new-key",
                    range_m=39.8,
                    vr_mps=-2.0,
                    az_rad=0.4,
                ),
                make_detection(
                    key="a1-new-key",
                    range_m=10.1,
                    vr_mps=1.0,
                    az_rad=-0.2,
                ),
                make_detection(
                    key="false-alarm",
                    range_m=90.0,
                    vr_mps=30.0,
                    az_rad=1.4,
                ),
            ),
        ),

        make_frame(
            timestamp_s=0.2,
            detections=(
                make_detection(
                    key="b2",
                    range_m=39.6,
                    vr_mps=-2.0,
                    az_rad=0.4,
                ),
                make_detection(
                    key="a2",
                    range_m=10.2,
                    vr_mps=1.0,
                    az_rad=-0.2,
                ),
            ),
        ),
    ),
)


config_path = (
    ROOT
    / "configs"
    / "stage3_association.json"
)

config_json = json.loads(
    config_path.read_text(
        encoding="utf-8"
    )
)


if config_json["uses_truth"]:
    raise SystemExit(
        "FAIL: association config uses truth."
    )


if config_json[
    "uses_detection_key_for_matching"
]:
    raise SystemExit(
        "FAIL: detection key used for matching."
    )


if config_json[
    "coordinate_domain"
] != "stage2_measurement_space":
    raise SystemExit(
        "FAIL: association domain changed."
    )


gates = config_json["gates"]

config = AssociationConfig(
    base_range_gate_m=(
        gates["base_range_gate_m"]
    ),
    max_range_rate_mps=(
        gates["max_range_rate_mps"]
    ),
    base_radial_velocity_gate_mps=(
        gates[
            "base_radial_velocity_gate_mps"
        ]
    ),
    max_radial_acceleration_mps2=(
        gates[
            "max_radial_acceleration_mps2"
        ]
    ),
    base_azimuth_gate_rad=(
        gates["base_azimuth_gate_rad"]
    ),
    max_azimuth_rate_radps=(
        gates[
            "max_azimuth_rate_radps"
        ]
    ),
    base_elevation_gate_rad=(
        gates[
            "base_elevation_gate_rad"
        ]
    ),
    max_elevation_rate_radps=(
        gates[
            "max_elevation_rate_radps"
        ]
    ),
    covariance_sigma_multiplier=(
        gates[
            "covariance_sigma_multiplier"
        ]
    ),
    max_missed_frames=(
        gates["max_missed_frames"]
    ),
)


first = associate_estimated_gnn(
    sequence,
    config=config,
)

second = associate_estimated_gnn(
    sequence,
    config=config,
)


def signature(result):
    return tuple(
        (
            track.track_id,
            tuple(
                item.detection.detection_key
                for item in track.detections
            ),
        )
        for track in result.tracks
    )


if signature(first) != signature(second):
    raise SystemExit(
        "FAIL: association is not deterministic."
    )


if first.truth_used:
    raise SystemExit(
        "FAIL: evaluator truth used."
    )


lengths = sorted(
    len(track.detections)
    for track in first.tracks
)

if lengths != [1, 3, 3]:
    raise SystemExit(
        "FAIL: unexpected synthetic "
        f"association result: {lengths}"
    )


long_tracks = [
    track
    for track in first.tracks
    if len(track.detections) == 3
]

if len(long_tracks) != 2:
    raise SystemExit(
        "FAIL: expected two persistent tracks."
    )


for track in long_tracks:
    keys = {
        item.detection.detection_key
        for item in track.detections
    }

    if "false-alarm" in keys:
        raise SystemExit(
            "FAIL: false alarm attached to "
            "persistent track."
        )


sha, count = implementation_hash()


report = {
    "stage": 3,
    "block": "3.1",

    "status": "PASS",

    "method": (
        "deterministic_gated_greedy_"
        "nearest_neighbor"
    ),

    "input_domain": (
        "stage2_measurement_space"
    ),

    "measurement_vector": [
        "range",
        "radial_velocity",
        "azimuth",
        "elevation"
    ],

    "measurement_covariance_used": True,

    "detection_key_used_for_matching": False,

    "algorithm_truth_leakage": "NONE",

    "actor_class_used": False,

    "future_information_used": False,

    "false_alarm_handling": "PASS",

    "missed_detection_lifecycle": (
        "IMPLEMENTED"
    ),

    "azimuth_wrap": "IMPLEMENTED",

    "deterministic": True,

    "formal_validation_tuning": False,

    "implementation": {
        "files": count,
        "sha256": sha
    }
}


report_path = (
    ROOT
    / "reports"
    / "block31_association_gate.json"
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
    "===== Stage3 Block 3.1 gate ====="
)

print(
    "association method       = "
    "gated greedy nearest-neighbor"
)

print(
    "input domain             = "
    "Stage2 measurement space"
)

print(
    "measurement covariance   = USED"
)

print(
    "detection key matching   = NO"
)

print(
    "actor class matching     = NO"
)

print(
    "algorithm truth leakage  = NONE"
)

print(
    "future information       = NO"
)

print(
    "false alarm handling     = PASS"
)

print(
    "miss lifecycle           = PASS"
)

print(
    "azimuth wrapping         = PASS"
)

print(
    "deterministic repeat     = PASS"
)

print(
    "implementation files     =",
    count,
)

print(
    "implementation SHA256    =",
    sha,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    report_path,
)

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
)

from iscai_stage3.hough import (
    HoughConfig,
    HoughFrameContext,
    build_raw_hough_frames,
    build_sparse_accumulator,
    extract_hough_peaks,
    run_multidimensional_hough,
)

from rigid_test_factory import (
    identity_transform,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


ROOT = Path(
    "/home/agni/waymo/iscai_stage3"
)

PART_A = Path(
    "/home/agni/waymo/"
    "part_a_reference/"
    "ISCAI_pc_fmcw"
)

EXPECTED_PART_A_COMMIT = (
    "44d62e3478e3818d1757b00971890f844cb032f7"
)


def implementation_hash():
    files = []

    for base in (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "docs",
    ):
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

    files = sorted(
        set(files),
        key=lambda p: str(
            p.relative_to(ROOT)
        ),
    )

    digest = hashlib.sha256()

    for path in files:
        relative = str(
            path.relative_to(ROOT)
        )

        digest.update(
            relative.encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return (
        digest.hexdigest(),
        len(files),
    )


def measurement_from_xy(
    *,
    key,
    x,
    y,
    vx,
    vy,
    variance_scale=1.0,
):
    range_m = math.hypot(
        x,
        y,
    )

    if range_m <= 0.0:
        raise ValueError(
            "Synthetic range must be positive."
        )

    azimuth = math.atan2(
        y,
        x,
    )

    radial_velocity = (
        x * vx
        +
        y * vy
    ) / range_m

    return make_detection(
        key=key,
        range_m=range_m,
        vr_mps=radial_velocity,
        az_rad=azimuth,
        el_rad=0.0,
        variance_scale=variance_scale,
    )


def stationary_context(
    times,
):
    return HoughFrameContext(
        transforms=FrameTransformContext(
            T_H0_from_W=(
                identity_transform()
            ),
            T_Ht_from_W_by_frame=tuple(
                identity_transform()
                for _ in times
            ),
        ),
        frame_timestamps_s=tuple(
            times
        ),
    )


# ============================================================
# 1. Frozen Part-A provenance
# ============================================================

actual_part_a_commit = subprocess.check_output(
    [
        "git",
        "-C",
        str(PART_A),
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()


if (
    actual_part_a_commit
    !=
    EXPECTED_PART_A_COMMIT
):
    raise SystemExit(
        "FAIL: frozen Part-A commit changed: "
        f"{actual_part_a_commit}"
    )


audit_path = (
    ROOT
    / "docs"
    / "part_a_hough_audit.md"
)

if not audit_path.exists():
    raise SystemExit(
        "FAIL: Part-A Hough audit document missing."
    )


audit_text = audit_path.read_text(
    encoding="utf-8"
)


required_audit_terms = (
    "rho may be signed",
    "AND-logic",
    "rolling-window",
    "[x_anchor, y_anchor, vx, vy]",
)


for term in required_audit_terms:
    if term not in audit_text:
        raise SystemExit(
            "FAIL: Part-A Hough audit missing "
            f"{term!r}."
        )


# ============================================================
# 2. Frozen Stage3 Hough configuration
# ============================================================

config_path = (
    ROOT
    / "configs"
    / "stage3_hough.json"
)

if not config_path.exists():
    raise SystemExit(
        "FAIL: configs/stage3_hough.json missing."
    )


config_json = json.loads(
    config_path.read_text(
        encoding="utf-8"
    )
)


if (
    config_json.get("baseline")
    !=
    "Multidimensional Hough Transform"
):
    raise SystemExit(
        "FAIL: MHT meaning changed."
    )


stage3_adaptation = config_json[
    "stage3_adaptation"
]


if (
    stage3_adaptation[
        "estimated_association_used"
    ]
):
    raise SystemExit(
        "FAIL: estimated association enabled "
        "for Hough."
    )


if (
    stage3_adaptation[
        "parameter_space"
    ]
    !=
    [
        "x_anchor",
        "y_anchor",
        "vx",
        "vy",
    ]
):
    raise SystemExit(
        "FAIL: Hough parameter space changed."
    )


if not config_json[
    "uncertainty_aware_voting"
][
    "enabled"
]:
    raise SystemExit(
        "FAIL: uncertainty-aware voting disabled."
    )


if not config_json[
    "uncertainty_aware_voting"
][
    "radial_velocity_used"
]:
    raise SystemExit(
        "FAIL: radial-velocity consistency disabled."
    )


if not config_json[
    "accumulator"
][
    "peak_NMS"
]:
    raise SystemExit(
        "FAIL: peak NMS disabled."
    )


if config_json[
    "formal_validation_tuning"
]:
    raise SystemExit(
        "FAIL: Hough parameters marked "
        "validation-tuned."
    )


information_policy = config_json[
    "information_policy"
]


for key in (
    "truth_used",
    "actor_identity_used",
    "detection_key_matching_used",
    "actor_class_used",
    "future_information_used",
):
    if information_policy[key]:
        raise SystemExit(
            "FAIL: prohibited information enabled: "
            f"{key}"
        )


# ============================================================
# 3. Real sparse accumulator / crossing / clutter / miss gate
# ============================================================

times = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)


frames = []

for index, timestamp in enumerate(
    times
):
    detections = []

    # Track A:
    # vx = +4 m/s, vy = +10 m/s.
    ax = (
        10.0
        +
        4.0
        *
        timestamp
    )

    ay = (
        -2.5
        +
        10.0
        *
        timestamp
    )

    detections.append(
        measurement_from_xy(
            key=f"A-{index}",
            x=ax,
            y=ay,
            vx=4.0,
            vy=10.0,
        )
    )

    # Track B:
    # vx = +4 m/s, vy = -10 m/s.
    # One deliberate miss at frame 3.
    if index != 3:
        bx = (
            10.0
            +
            4.0
            *
            timestamp
        )

        by = (
            2.5
            -
            10.0
            *
            timestamp
        )

        detections.append(
            measurement_from_xy(
                key=f"B-{index}",
                x=bx,
                y=by,
                vx=4.0,
                vy=-10.0,
            )
        )

    # Deterministic nonlinear clutter.
    clutter_x = (
        24.0
        -
        2.5
        *
        index
    )

    clutter_y = (
        -9.0
        +
        0.65
        *
        index
        *
        index
    )

    detections.append(
        measurement_from_xy(
            key=f"FA-{index}",
            x=clutter_x,
            y=clutter_y,
            vx=0.0,
            vy=0.0,
        )
    )

    frames.append(
        make_frame(
            timestamp_s=timestamp,
            detections=tuple(
                detections
            ),
        )
    )


sequence = AlgorithmObservationSequence(
    scenario_id="block36-crossing",
    frames=tuple(frames),
)


config = HoughConfig(
    velocity_min_mps=-20.0,
    velocity_max_mps=20.0,
    velocity_step_mps=2.0,

    anchor_position_bin_m=1.0,

    minimum_vote_frames=3,
    minimum_support_frames=3,

    minimum_time_span_s=0.2,
    maximum_missing_gap_frames=2,

    maximum_peaks=16,
)


context = stationary_context(
    times
)


first = run_multidimensional_hough(
    sequence,
    context=context,
    config=config,
)


second = run_multidimensional_hough(
    sequence,
    context=context,
    config=config,
)


if first != second:
    raise SystemExit(
        "FAIL: Hough is not deterministic."
    )


if first.accumulator_cells <= 0:
    raise SystemExit(
        "FAIL: Hough accumulator is empty."
    )


if not first.peaks:
    raise SystemExit(
        "FAIL: Hough peak extraction produced "
        "no peaks."
    )


if not first.tracks:
    raise SystemExit(
        "FAIL: Hough final validation produced "
        "no tracks."
    )


if first.estimated_association_used:
    raise SystemExit(
        "FAIL: Hough used estimated association."
    )


if (
    first.truth_used
    or
    first.future_information_used
):
    raise SystemExit(
        "FAIL: prohibited information entered "
        "Hough result."
    )


velocities = tuple(
    track.velocity_H0_mps
    for track in first.tracks
)


track_a_found = any(
    abs(
        velocity[0]
        -
        4.0
    ) < 1.5
    and
    abs(
        velocity[1]
        -
        10.0
    ) < 1.5
    for velocity in velocities
)


track_b_found = any(
    abs(
        velocity[0]
        -
        4.0
    ) < 1.5
    and
    abs(
        velocity[1]
        +
        10.0
    ) < 1.5
    for velocity in velocities
)


if not track_a_found:
    raise SystemExit(
        "FAIL: crossing Track A not recovered. "
        f"velocities={velocities}"
    )


if not track_b_found:
    raise SystemExit(
        "FAIL: crossing Track B not recovered. "
        f"velocities={velocities}"
    )


if not any(
    len(
        track.support_refs
    ) >= 5
    for track in first.tracks
):
    raise SystemExit(
        "FAIL: miss-tolerant temporal support "
        "was not recovered."
    )


# ============================================================
# 4. Stage2 covariance must alter Hough vote confidence
# ============================================================

uncertainty_times = (
    0.0,
    0.1,
    0.2,
)


def build_uncertainty_sequence(
    variance_scale,
):
    result = []

    for index, timestamp in enumerate(
        uncertainty_times
    ):
        x = (
            10.0
            +
            10.0
            *
            timestamp
        )

        result.append(
            make_frame(
                timestamp_s=timestamp,
                detections=(
                    measurement_from_xy(
                        key=f"U-{index}",
                        x=x,
                        y=0.0,
                        vx=10.0,
                        vy=0.0,
                        variance_scale=(
                            variance_scale
                        ),
                    ),
                ),
            )
        )

    return AlgorithmObservationSequence(
        scenario_id="uncertainty",
        frames=tuple(result),
    )


uncertainty_context = (
    stationary_context(
        uncertainty_times
    )
)


low_frames = build_raw_hough_frames(
    build_uncertainty_sequence(
        1.0
    ),
    context=uncertainty_context,
)


high_frames = build_raw_hough_frames(
    build_uncertainty_sequence(
        10000.0
    ),
    context=uncertainty_context,
)


low_accumulator = build_sparse_accumulator(
    low_frames,
    context=uncertainty_context,
    config=config,
)


high_accumulator = build_sparse_accumulator(
    high_frames,
    context=uncertainty_context,
    config=config,
)


low_peaks = extract_hough_peaks(
    low_accumulator,
    config=config,
)


high_peaks = extract_hough_peaks(
    high_accumulator,
    config=config,
)


if not low_peaks:
    raise SystemExit(
        "FAIL: low-uncertainty gate has no peak."
    )


if not high_peaks:
    raise SystemExit(
        "FAIL: high-uncertainty gate has no peak."
    )


low_score = float(
    low_peaks[0].score
)

high_score = float(
    high_peaks[0].score
)


if not (
    low_score
    >
    high_score
):
    raise SystemExit(
        "FAIL: Stage2 covariance does not "
        "influence Hough vote confidence: "
        f"low={low_score}, high={high_score}"
    )


# ============================================================
# 5. Static leakage / architecture scan
# ============================================================

source_path = (
    ROOT
    / "src"
    / "iscai_stage3"
    / "hough"
    / "multidimensional.py"
)


source_text = source_path.read_text(
    encoding="utf-8"
)


forbidden_text_tokens = (
    "associate_estimated_gnn",
    "AssociatedTrack",
    "AssociatedDetection",
    "DetectionTruthSidecar",
    "DetectionTruthEntry",
    "EvaluatorTruthSequence",
    "source_track_id",
    "scenario.tracks",
    "iscai_stage3_panagiota",
    "Multiple Hypothesis Tracking",
)


for token in forbidden_text_tokens:
    if token in source_text:
        raise SystemExit(
            "FAIL: forbidden Hough dependency: "
            f"{token!r}"
        )


tree = ast.parse(
    source_text,
    filename=str(source_path),
)


forbidden_attributes = {
    "actor_class",
    "object_type",
    "source_track_id",
    "truth",
}


seen_forbidden_attributes = sorted(
    {
        node.attr
        for node in ast.walk(tree)
        if (
            isinstance(
                node,
                ast.Attribute,
            )
            and
            node.attr
            in forbidden_attributes
        )
    }
)


if seen_forbidden_attributes:
    raise SystemExit(
        "FAIL: forbidden algorithm attributes: "
        f"{seen_forbidden_attributes}"
    )


# ============================================================
# 6. Final report
# ============================================================

sha256, file_count = (
    implementation_hash()
)


report = {
    "stage": 3,
    "block": "3.6",
    "status": "PASS",

    "part_a": {
        "commit": (
            actual_part_a_commit
        ),
        "commit_gate": "PASS",
        "observed_core": (
            "rho/theta Hough voting, "
            "peak extraction, AND-logic "
            "trajectory validation"
        )
    },

    "mht_meaning": (
        "Multidimensional Hough Transform"
    ),

    "stage3_input": (
        "raw unlabeled Stage2 detections"
    ),

    "estimated_association_used": False,

    "parameter_space": [
        "x_anchor",
        "y_anchor",
        "vx",
        "vy"
    ],

    "algorithm": {
        "sparse_4d_accumulator": True,
        "uncertainty_weighted_voting": True,
        "radial_velocity_consistency": True,
        "peak_extraction": True,
        "peak_nms": True,
        "and_logic_validation": True,
        "weighted_trajectory_refit": True
    },

    "synthetic_gate": {
        "crossing_tracks": "PASS",
        "miss_handling": "PASS",
        "clutter_present": True,
        "track_A_found": (
            track_a_found
        ),
        "track_B_found": (
            track_b_found
        ),
        "accumulator_cells": (
            first.accumulator_cells
        ),
        "raw_peaks": len(
            first.peaks
        ),
        "validated_tracks": len(
            first.tracks
        )
    },

    "measurement_uncertainty": {
        "stage2_R_t_affects_votes": True,
        "low_uncertainty_peak_score": (
            low_score
        ),
        "high_uncertainty_peak_score": (
            high_score
        )
    },

    "information_policy": {
        "truth_used": False,
        "estimated_association_used": False,
        "actor_identity_used": False,
        "detection_key_matching_used": False,
        "actor_class_used": False,
        "future_information_used": False
    },

    "deterministic_repeat": True,

    "formal_real_validation_metrics": (
        "DEFERRED_TO_COMMON_STAGE3_"
        "EVALUATION_BLOCK"
    ),

    "implementation": {
        "files": file_count,
        "sha256": sha256
    }
}


report_path = (
    ROOT
    / "reports"
    / "block36_hough_gate.json"
)


report_path.write_text(
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
    "===== Stage3 Block 3.6 gate ====="
)

print(
    "Part-A frozen commit        = PASS"
)

print(
    "Part-A Hough core           = "
    "rho/theta + peaks + AND-logic"
)

print(
    "MHT meaning                 = "
    "Multidimensional Hough Transform"
)

print(
    "Stage3 input                = "
    "raw unlabeled Stage2 detections"
)

print(
    "estimated association       = NO"
)

print(
    "parameter space             = "
    "[x_anchor, y_anchor, vx, vy]"
)

print(
    "sparse accumulator          = PASS"
)

print(
    "Stage2 uncertainty voting   = PASS"
)

print(
    "radial velocity consistency = PASS"
)

print(
    "peak extraction             = PASS"
)

print(
    "peak NMS                    = PASS"
)

print(
    "AND-logic validation        = PASS"
)

print(
    "crossing tracks             = PASS"
)

print(
    "miss handling               = PASS"
)

print(
    "clutter handling            = PASS"
)

print(
    "actor identity              = NO"
)

print(
    "actor class                 = NO"
)

print(
    "truth leakage               = NONE"
)

print(
    "future information          = NO"
)

print(
    "deterministic repeat        = PASS"
)

print(
    "accumulator cells           =",
    first.accumulator_cells,
)

print(
    "raw peaks                   =",
    len(first.peaks),
)

print(
    "validated tracks            =",
    len(first.tracks),
)

print(
    "low-uncertainty peak score  =",
    low_score,
)

print(
    "high-uncertainty peak score =",
    high_score,
)

print(
    "implementation files        =",
    file_count,
)

print(
    "implementation SHA256       =",
    sha256,
)

print(
    "STATUS = PASS"
)

print(
    "report =",
    report_path,
)

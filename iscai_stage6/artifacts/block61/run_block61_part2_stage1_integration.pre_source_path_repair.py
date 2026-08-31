from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import copy
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback

import numpy as np


ROOT = Path("/home/agni/waymo")

S0 = ROOT / "iscai_stage0"
S1 = ROOT / "iscai_stage1"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PART1_REPORT = (
    S6
    / "reports/"
      "block61_part1_geometry_core.json"
)

BLOCK60_CLOSURE = (
    S6
    / "reports/block60_closure.json"
)

STAGE1_CLOSURE = (
    S1
    / "reports/stage1a/"
      "stage1a_closure_report.json"
)

STAGE1_PILOT_MANIFEST = (
    S1
    / "artifacts/stage1a/manifests/"
      "tiny_pilot_validation.jsonl"
)

STAGE5_CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

GEOMETRY_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "geometry.py"
)

GRID_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "grid.py"
)

WOMD_ADAPTER_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "womd_geometry.py"
)

ADB_INIT = (
    S6
    / "src/iscai_stage6/adb/"
      "__init__.py"
)

INTEGRATION_CONTRACT = (
    S6
    / "configs/"
      "block61_stage1_integration_contract.json"
)

REAL_SMOKE = (
    S6
    / "artifacts/block61/"
      "real_scene_full_box_projection.json"
)

TEST_FILE = (
    S6
    / "tests/"
      "test_block61_stage1_integration.py"
)

CLOSURE = (
    S6
    / "reports/block61_closure.json"
)

REGRESSION_LOG = (
    S6
    / "artifacts/block61/"
      "block61_full_regression.log"
)

EXPECTED = {
    "part1_report":
        (
            "d83d2f6baed18ea20dd6ba40ac085cb5"
            "dcf206e5c4624df23b3eaaf7073c3346"
        ),

    "block60_closure":
        (
            "525cf827b217cc2806bfff6547c5e241"
            "b3c9db04d188f7957acea117b309c303"
        ),

    "stage5_closure":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),

    "pilot_manifest":
        (
            "2b6fd3d7e5c411455708d325fc8595af"
            "11d99f3fcba4dde621928150695ca859"
        ),
}

SUPPORTED_CLASSES = frozenset(
    {
        "TYPE_VEHICLE",
        "TYPE_PEDESTRIAN",
        "TYPE_CYCLIST",
    }
)

MIN_FREE_GIB = 250.0

TOL_CENTER_M = 1.0e-8
TOL_BEARING_RAD = 1.0e-8
TOL_AXIS = 1.0e-10


# ============================================================
# Generic utilities
# ============================================================

def require(
    condition,
    message,
):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(
    path: Path,
):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(
    payload,
) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def write_exact_json(
    path: Path,
    payload,
):
    desired = canonical_bytes(
        payload
    )

    if path.exists():
        if path.read_bytes() != desired:
            raise RuntimeError(
                "Existing deterministic artifact "
                f"differs: {path}"
            )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(desired)

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def write_exact_text(
    path: Path,
    text: str,
):
    desired = text.encode(
        "utf-8"
    )

    if path.exists():
        if path.read_bytes() != desired:
            raise RuntimeError(
                "Existing deterministic source "
                f"differs: {path}"
            )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(desired)

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def tail_text(
    text: str,
    lines: int = 60,
) -> str:
    rows = text.splitlines()

    return "\n".join(
        rows[-lines:]
    )


def load_jsonl(
    path: Path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line in stream:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def object_to_jsonable(
    value,
):
    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "__dict__"):
        return {
            key:
                object_to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def run_stage6_tests():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S6 / "tests"),
            "-p",
            "test_block6*.py",
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    REGRESSION_LOG.write_text(
        process.stdout,
        encoding="utf-8",
    )

    return (
        process.returncode,
        process.stdout,
    )


# ============================================================
# Exact single-record WOMD access
# ============================================================

def read_scenario_at_tfrecord_offset(
    *,
    path: Path,
    record_offset: int,
):
    """
    Read exactly one TFRecord record without scanning a shard.
    """

    from waymo_open_dataset.protos import (
        scenario_pb2,
    )

    require(
        path.is_file(),
        f"Motion shard missing: {path}",
    )

    offset = int(record_offset)

    require(
        offset >= 0,
        "record_offset must be non-negative",
    )

    with path.open("rb") as stream:
        stream.seek(offset)

        length_bytes = stream.read(8)

        require(
            len(length_bytes) == 8,
            (
                "Could not read TFRecord length "
                f"at offset {offset}"
            ),
        )

        payload_length = struct.unpack(
            "<Q",
            length_bytes,
        )[0]

        length_crc = stream.read(4)

        require(
            len(length_crc) == 4,
            "Missing TFRecord length CRC",
        )

        payload = stream.read(
            payload_length
        )

        require(
            len(payload)
            ==
            payload_length,
            "Incomplete TFRecord payload",
        )

        data_crc = stream.read(4)

        require(
            len(data_crc) == 4,
            "Missing TFRecord data CRC",
        )

    scenario = scenario_pb2.Scenario()

    scenario.ParseFromString(
        payload
    )

    return (
        scenario,
        int(payload_length),
    )


# ============================================================
# Stage1 source/config discovery
# ============================================================

def discover_class_module(
    *,
    source_root: Path,
    class_name: str,
) -> tuple[
    str,
    Path,
]:
    matches = []

    marker = (
        f"class {class_name}"
    )

    for path in sorted(
        source_root.rglob("*.py")
    ):
        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except Exception:
            continue

        if marker in text:
            matches.append(path)

    require(
        len(matches) == 1,
        (
            f"Expected exactly one {class_name} "
            "definition, found: "
            + ", ".join(
                str(path)
                for path in matches
            )
        ),
    )

    path = matches[0]

    relative = path.relative_to(
        source_root
    )

    parts = list(
        relative.with_suffix("").parts
    )

    module = ".".join(parts)

    return (
        module,
        path,
    )


def rotation_matrix(
    transform,
) -> np.ndarray:
    matrix = np.asarray(
        transform.rotation,
        dtype=np.float64,
    )

    require(
        matrix.shape
        ==
        (3, 3),
        (
            "Stage1 transform rotation "
            f"has shape {matrix.shape}"
        ),
    )

    require(
        np.all(
            np.isfinite(matrix)
        ),
        "Stage1 rotation is non-finite",
    )

    return matrix


def verify_rotation(
    matrix: np.ndarray,
):
    identity_error = float(
        np.max(
            np.abs(
                matrix.T
                @
                matrix
                -
                np.eye(3)
            )
        )
    )

    determinant = float(
        np.linalg.det(matrix)
    )

    require(
        identity_error
        <=
        1.0e-9,
        (
            "Stage1 H0 rotation is not "
            "orthonormal enough: "
            f"{identity_error}"
        ),
    )

    require(
        abs(
            determinant - 1.0
        )
        <=
        1.0e-9,
        (
            "Stage1 H0 rotation determinant "
            f"is {determinant}"
        ),
    )

    return {
        "orthonormal_max_error":
            identity_error,

        "determinant":
            determinant,
    }


# ============================================================
# Stable box digest
# ============================================================

def box_digest(
    actors,
) -> str:
    rows = []

    for actor in sorted(
        actors,
        key=lambda item:
            (
                int(
                    item.track_index
                ),
                str(
                    item.track_id
                ),
            ),
    ):
        box = actor.box

        rows.append({
            "track_index":
                int(actor.track_index),

            "track_id":
                str(actor.track_id),

            "object_type":
                str(actor.object_type),

            "center_xyz": [
                float(v)
                for v in box.center_xyz
            ],

            "length_m":
                float(box.length_m),

            "width_m":
                float(box.width_m),

            "height_m":
                float(box.height_m),

            "yaw_rad":
                float(box.yaw_rad),
        })

    return sha256(
        canonical_bytes(
            rows
        )
    ).hexdigest()


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.1 PART 2/2"
    )
    print(
        "FROZEN STAGE1 INTEGRATION + REAL FULL-BOX SMOKE"
    )
    print(
        "============================================================"
    )

    required = (
        PART1_REPORT,
        BLOCK60_CLOSURE,
        STAGE1_CLOSURE,
        STAGE1_PILOT_MANIFEST,
        STAGE5_CLOSURE,
        GEOMETRY_MODULE,
        GRID_MODULE,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing prerequisite: "
            + ", ".join(missing)
        ),
    )

    protected_before = {
        str(path):
            sha256_file(path)
        for path in required
    }

    # ========================================================
    # A. Stage6 / Stage1 frozen prerequisite
    # ========================================================

    print()
    print(
        "===== A. FROZEN PREREQUISITES ====="
    )

    require(
        sha256_file(
            PART1_REPORT
        )
        ==
        EXPECTED[
            "part1_report"
        ],
        "Block6.1 Part1 SHA changed.",
    )

    require(
        sha256_file(
            BLOCK60_CLOSURE
        )
        ==
        EXPECTED[
            "block60_closure"
        ],
        "Block6.0 closure SHA changed.",
    )

    require(
        sha256_file(
            STAGE5_CLOSURE
        )
        ==
        EXPECTED[
            "stage5_closure"
        ],
        "Stage5 closure SHA changed.",
    )

    require(
        sha256_file(
            STAGE1_PILOT_MANIFEST
        )
        ==
        EXPECTED[
            "pilot_manifest"
        ],
        "Frozen Stage1 pilot manifest changed.",
    )

    part1 = load_json(
        PART1_REPORT
    )

    stage1 = load_json(
        STAGE1_CLOSURE
    )

    require(
        part1.get("status")
        ==
        "PASS_GEOMETRY_CORE",
        (
            "Block6.1 Part1 is not "
            "PASS_GEOMETRY_CORE."
        ),
    )

    require(
        stage1.get("status")
        ==
        "PASS_COMPLETE",
        (
            "Frozen Stage1 closure is not "
            "PASS_COMPLETE."
        ),
    )

    require(
        stage1.get(
            "geometry",
            {},
        ).get(
            "E0_axes"
        )
        ==
        "+x forward, +y left, +z up",
        (
            "Frozen Stage1 axis contract "
            "does not match Stage6 geometry."
        ),
    )

    require(
        stage1.get(
            "geometry",
            {},
        ).get(
            "H0"
        )
        ==
        "front_face_midpoint_surrogate",
        (
            "Frozen Stage1 H0 mode changed."
        ),
    )

    causality = stage1.get(
        "causality",
        {},
    )

    require(
        causality.get(
            "future_actor_state_dependency"
        )
        is False,
        "Stage1 future actor dependency detected.",
    )

    require(
        causality.get(
            "future_box_dependency"
        )
        is False,
        "Stage1 future box dependency detected.",
    )

    require(
        causality.get(
            "tracks_to_predict_as_realistic_input"
        )
        is False,
        (
            "Stage1 tracks_to_predict dependency "
            "detected."
        ),
    )

    print(
        "Block6.1 Part1       = EXACT PASS"
    )

    print(
        "Stage1 closure       = PASS_COMPLETE"
    )

    print(
        "H0                   = front_face_midpoint_surrogate"
    )

    print(
        "axes                 = +x forward / +y left / +z up"
    )

    print(
        "future-box dependency= NO"
    )

    print(
        "tracks_to_predict use= NO"
    )

    # ========================================================
    # B. Exact Stage1 headlamp API/config discovery
    # ========================================================

    print()
    print(
        "===== B. STAGE1 HEADLAMP API / EXTRINSIC AUDIT ====="
    )

    stage1_src = (
        S1 / "src"
    )

    (
        headlamp_module_name,
        headlamp_source,
    ) = discover_class_module(
        source_root=stage1_src,
        class_name="HeadlampSurrogateConfig",
    )

    headlamp_module = (
        importlib.import_module(
            headlamp_module_name
        )
    )

    HeadlampSurrogateConfig = getattr(
        headlamp_module,
        "HeadlampSurrogateConfig",
    )

    headlamp_config = (
        HeadlampSurrogateConfig()
    )

    headlamp_config_json = (
        object_to_jsonable(
            headlamp_config
        )
    )

    frames_source = (
        S1
        / "src/iscai_stage1/"
          "geometry/frames.py"
    )

    actor_adapter_source = (
        S1
        / "src/iscai_stage1/"
          "actors/womd_adapter.py"
    )

    history_source = (
        S1
        / "src/iscai_stage1/"
          "actors/history.py"
    )

    for path in (
        headlamp_source,
        frames_source,
        actor_adapter_source,
        history_source,
    ):
        require(
            path.is_file(),
            f"Stage1 source missing: {path}",
        )

    # Current Stage6 Box3D is yaw-oriented with z vertical.
    # This is exact for the frozen/default headlamp baseline only
    # if the headlamp frame has no roll/pitch relative to ego.
    roll = float(
        getattr(
            headlamp_config,
            "roll_rad",
            0.0,
        )
    )

    pitch = float(
        getattr(
            headlamp_config,
            "pitch_rad",
            0.0,
        )
    )

    require(
        abs(roll)
        <=
        TOL_AXIS,
        (
            "Frozen/default Stage1 headlamp roll "
            f"is non-zero ({roll}); yaw-only Box3D "
            "would be insufficient."
        ),
    )

    require(
        abs(pitch)
        <=
        TOL_AXIS,
        (
            "Frozen/default Stage1 headlamp pitch "
            f"is non-zero ({pitch}); yaw-only Box3D "
            "would be insufficient."
        ),
    )

    stage1_source_hashes = {
        "HeadlampSurrogateConfig_source":
            sha256_file(
                headlamp_source
            ),

        "frames.py":
            sha256_file(
                frames_source
            ),

        "actors/womd_adapter.py":
            sha256_file(
                actor_adapter_source
            ),

        "actors/history.py":
            sha256_file(
                history_source
            ),
    }

    print(
        "HeadlampSurrogateConfig module =",
        headlamp_module_name,
    )

    print(
        "default roll                 =",
        roll,
    )

    print(
        "default pitch                =",
        pitch,
    )

    print(
        "yaw-only Stage6 box basis    = COMPATIBLE PASS"
    )

    # ========================================================
    # C. Write Stage6 causal WOMD geometry adapter
    # ========================================================

    print()
    print(
        "===== C. CAUSAL WOMD → FULL BOX H0 ADAPTER ====="
    )

    adapter_source = r'''from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

import numpy as np

from iscai_stage1.actors.womd_adapter import (
    object_type_name,
)

from .geometry import (
    Box3D,
    project_box_to_headlamp,
    wrap_angle,
)


SUPPORTED_ADB_CLASSES = frozenset(
    {
        "TYPE_VEHICLE",
        "TYPE_PEDESTRIAN",
        "TYPE_CYCLIST",
    }
)


@dataclass(frozen=True)
class CausalADBActorBox:
    scenario_id: str
    track_index: int
    track_id: str
    object_type: str

    box: Box3D

    stage1_anchor_center_H0_m: tuple[float, float, float] | None

    stage1_anchor_bearing_H0_rad: float | None

    center_consistency_error_m: float

    source_time_index: int

    state_source: str = "current_causal"
    orientation_source: str = "current_causal"

    tracks_to_predict_used: bool = False
    objects_of_interest_used: bool = False
    future_state_used: bool = False


def _rotation_matrix(
    transform: Any,
) -> np.ndarray:

    matrix = np.asarray(
        transform.rotation,
        dtype=np.float64,
    )

    if matrix.shape != (
        3,
        3,
    ):
        raise ValueError(
            "T_H0_from_W.rotation must be 3x3"
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "T_H0_from_W.rotation contains "
            "non-finite values"
        )

    return matrix


def world_heading_to_headlamp_yaw(
    *,
    heading_W_rad: float,
    T_H0_from_W: Any,
) -> float:
    """
    Transform an actor's world-frame forward direction into H0
    and recover yaw from the projected horizontal direction.

    No future heading is used.
    """

    heading = float(
        heading_W_rad
    )

    if not math.isfinite(
        heading
    ):
        raise ValueError(
            "heading_W_rad must be finite"
        )

    forward_W = np.asarray(
        [
            math.cos(heading),
            math.sin(heading),
            0.0,
        ],
        dtype=np.float64,
    )

    forward_H0 = (
        _rotation_matrix(
            T_H0_from_W
        )
        @
        forward_W
    )

    horizontal_norm = math.hypot(
        float(
            forward_H0[0]
        ),
        float(
            forward_H0[1]
        ),
    )

    if horizontal_norm <= 1.0e-12:
        raise ValueError(
            "Transformed actor forward vector "
            "has zero horizontal norm"
        )

    return wrap_angle(
        math.atan2(
            float(
                forward_H0[1]
            ),
            float(
                forward_H0[0]
            ),
        )
    )


def build_causal_adb_actor_boxes(
    *,
    scenario: Any,
    adapted: Any,
    allowed_classes:
        frozenset[str] = SUPPORTED_ADB_CLASSES,
) -> tuple[CausalADBActorBox, ...]:
    """
    Build all anchor-valid, non-SDC vehicle/pedestrian/cyclist
    boxes from current causal WOMD geometry.

    Selection deliberately does NOT inspect:
      - tracks_to_predict,
      - objects_of_interest,
      - any state after current_time_index.
    """

    anchor = int(
        scenario.current_time_index
    )

    if int(
        adapted.anchor_index
    ) != anchor:
        raise RuntimeError(
            "Stage1 adapted anchor differs "
            "from WOMD current_time_index"
        )

    T_H0_from_W = (
        adapted.frames.T_H0_from_W
    )

    results = []

    for adapted_actor in (
        adapted.actors
    ):
        track_index = int(
            adapted_actor.track_index
        )

        if (
            track_index
            ==
            int(
                scenario.sdc_track_index
            )
        ):
            continue

        if not (
            0
            <=
            track_index
            <
            len(
                scenario.tracks
            )
        ):
            raise RuntimeError(
                "Invalid Stage1 actor track index"
            )

        track = scenario.tracks[
            track_index
        ]

        actor_class = (
            object_type_name(
                track
            )
        )

        if (
            actor_class
            not in
            allowed_classes
        ):
            continue

        if anchor >= len(
            track.states
        ):
            raise RuntimeError(
                "Anchor index exceeds track states"
            )

        state = track.states[
            anchor
        ]

        if not bool(
            state.valid
        ):
            continue

        dimensions = (
            float(state.length),
            float(state.width),
            float(state.height),
        )

        if not all(
            math.isfinite(v)
            and
            v > 0.0
            for v in dimensions
        ):
            raise ValueError(
                "Anchor-valid actor has invalid "
                "length/width/height"
            )

        center_W = (
            float(state.center_x),
            float(state.center_y),
            float(state.center_z),
        )

        center_H0_raw = (
            T_H0_from_W.apply_point(
                center_W
            )
        )

        center_H0 = tuple(
            float(v)
            for v in center_H0_raw
        )

        if not all(
            math.isfinite(v)
            for v in center_H0
        ):
            raise ValueError(
                "Transformed actor center "
                "contains non-finite values"
            )

        yaw_H0 = (
            world_heading_to_headlamp_yaw(
                heading_W_rad=(
                    float(
                        state.heading
                    )
                ),
                T_H0_from_W=(
                    T_H0_from_W
                ),
            )
        )

        box = Box3D(
            center_xyz=center_H0,
            length_m=dimensions[0],
            width_m=dimensions[1],
            height_m=dimensions[2],
            yaw_rad=yaw_H0,
        )

        stage1_center = (
            adapted_actor
            .anchor_center_H0_m
        )

        if stage1_center is None:
            consistency_error = float(
                "nan"
            )
        else:
            consistency_error = math.sqrt(
                sum(
                    (
                        float(
                            center_H0[i]
                        )
                        -
                        float(
                            stage1_center[i]
                        )
                    )
                    ** 2
                    for i in range(3)
                )
            )

        results.append(
            CausalADBActorBox(
                scenario_id=str(
                    adapted.scenario_id
                ),

                track_index=track_index,

                track_id=str(
                    track.id
                ),

                object_type=actor_class,

                box=box,

                stage1_anchor_center_H0_m=(
                    None
                    if stage1_center is None
                    else tuple(
                        float(v)
                        for v in stage1_center
                    )
                ),

                stage1_anchor_bearing_H0_rad=(
                    None
                    if (
                        adapted_actor
                        .anchor_bearing_H0_rad
                        is None
                    )
                    else float(
                        adapted_actor
                        .anchor_bearing_H0_rad
                    )
                ),

                center_consistency_error_m=(
                    float(
                        consistency_error
                    )
                ),

                source_time_index=anchor,
            )
        )

    return tuple(
        results
    )


def validate_stage1_projection_consistency(
    actor: CausalADBActorBox,
    *,
    center_tolerance_m: float = 1.0e-8,
    bearing_tolerance_rad: float = 1.0e-8,
) -> dict[str, float | bool]:
    """
    Cross-check Stage6 centroid geometry against frozen Stage1
    anchor-center and anchor-bearing outputs.
    """

    if (
        actor.stage1_anchor_center_H0_m
        is None
    ):
        raise RuntimeError(
            "Stage1 anchor center is unavailable "
            "for an anchor-valid ADB actor"
        )

    if not math.isfinite(
        actor.center_consistency_error_m
    ):
        raise RuntimeError(
            "Non-finite Stage1/Stage6 center "
            "consistency error"
        )

    if (
        actor.center_consistency_error_m
        >
        center_tolerance_m
    ):
        raise RuntimeError(
            "Stage1/Stage6 center mismatch: "
            f"{actor.center_consistency_error_m}"
        )

    projected = (
        project_box_to_headlamp(
            actor.box,
            state_source="current_causal",
            orientation_source="current_causal",
            controller_path=True,
        )
    )

    stage1_bearing = (
        actor.stage1_anchor_bearing_H0_rad
    )

    bearing_error = float(
        "nan"
    )

    if stage1_bearing is not None:
        delta = wrap_angle(
            float(
                projected.centroid.theta_rad[
                    0
                ]
            )
            -
            float(
                stage1_bearing
            )
        )

        bearing_error = abs(
            float(delta)
        )

        if (
            bearing_error
            >
            bearing_tolerance_rad
        ):
            raise RuntimeError(
                "Stage1/Stage6 bearing mismatch: "
                f"{bearing_error}"
            )

    return {
        "center_error_m":
            float(
                actor.center_consistency_error_m
            ),

        "bearing_error_rad":
            float(
                bearing_error
            ),

        "eight_corners":
            (
                projected.corners.xyz.shape
                ==
                (8, 3)
            ),

        "theta_span_positive":
            (
                projected.theta_span_rad
                >
                0.0
            ),

        "range_extent_positive":
            (
                projected.range_3d_max_m
                >
                projected.range_3d_min_m
            ),
    }


def causal_actor_boxes_sha256(
    actors,
) -> str:
    """
    Stable hash for causality/invariance tests.
    """

    payload = []

    for actor in sorted(
        actors,
        key=lambda item:
            (
                item.track_index,
                item.track_id,
            ),
    ):
        payload.append({
            "scenario_id":
                actor.scenario_id,

            "track_index":
                actor.track_index,

            "track_id":
                actor.track_id,

            "object_type":
                actor.object_type,

            "center_xyz":
                list(
                    actor.box.center_xyz
                ),

            "length_m":
                actor.box.length_m,

            "width_m":
                actor.box.width_m,

            "height_m":
                actor.box.height_m,

            "yaw_rad":
                actor.box.yaw_rad,

            "source_time_index":
                actor.source_time_index,

            "tracks_to_predict_used":
                actor.tracks_to_predict_used,

            "objects_of_interest_used":
                actor.objects_of_interest_used,

            "future_state_used":
                actor.future_state_used,
        })

    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .encode("utf-8")
    )

    return sha256(
        encoded
    ).hexdigest()
'''

    adapter_write = write_exact_text(
        WOMD_ADAPTER_MODULE,
        adapter_source,
    )

    print(
        "womd_geometry.py =",
        adapter_write,
    )

    # ========================================================
    # D. Update Stage6 ADB package exports
    # ========================================================

    print()
    print(
        "===== D. PACKAGE EXPORT CONTRACT ====="
    )

    current_init = (
        ADB_INIT.read_text(
            encoding="utf-8"
        )
        if ADB_INIT.exists()
        else ""
    )

    required_import = '''from .womd_geometry import (
    CausalADBActorBox,
    SUPPORTED_ADB_CLASSES,
    build_causal_adb_actor_boxes,
    causal_actor_boxes_sha256,
    validate_stage1_projection_consistency,
    world_heading_to_headlamp_yaw,
)
'''

    if (
        required_import
        not in
        current_init
    ):
        insert_anchor = (
            "__all__ = ("
        )

        require(
            insert_anchor
            in
            current_init,
            (
                "Could not locate __all__ in "
                "adb/__init__.py"
            ),
        )

        current_init = (
            current_init.replace(
                insert_anchor,
                required_import
                + "\n"
                + insert_anchor,
                1,
            )
        )

        additions = (
            '    "CausalADBActorBox",\n'
            '    "SUPPORTED_ADB_CLASSES",\n'
            '    "build_causal_adb_actor_boxes",\n'
            '    "causal_actor_boxes_sha256",\n'
            '    "validate_stage1_projection_consistency",\n'
            '    "world_heading_to_headlamp_yaw",\n'
        )

        current_init = (
            current_init.replace(
                insert_anchor,
                insert_anchor
                + "\n"
                + additions,
                1,
            )
        )

        # Existing file is a Stage6 Part1 artifact, so update here
        # deliberately rather than using write_exact_text.
        tmp = ADB_INIT.with_suffix(
            ".py.tmp"
        )

        tmp.write_text(
            current_init,
            encoding="utf-8",
        )

        os.replace(
            tmp,
            ADB_INIT,
        )

        init_status = "UPDATED_PART2"

    else:
        init_status = "ALREADY_UPDATED"

    print(
        "adb/__init__.py =",
        init_status,
    )

    # ========================================================
    # E. Unit tests for adapter semantics
    # ========================================================

    print()
    print(
        "===== E. ADAPTER UNIT TESTS ====="
    )

    test_source = r'''from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.womd_geometry import (
    CausalADBActorBox,
    SUPPORTED_ADB_CLASSES,
    causal_actor_boxes_sha256,
    world_heading_to_headlamp_yaw,
)


class _Transform:
    def __init__(
        self,
        rotation,
    ):
        self.rotation = rotation


class TestBlock61Stage1Integration(
    unittest.TestCase
):

    def test_required_adb_classes(self):

        self.assertEqual(
            SUPPORTED_ADB_CLASSES,
            frozenset(
                {
                    "TYPE_VEHICLE",
                    "TYPE_PEDESTRIAN",
                    "TYPE_CYCLIST",
                }
            ),
        )

    def test_identity_heading_transform(self):

        transform = _Transform(
            np.eye(3)
        )

        value = (
            world_heading_to_headlamp_yaw(
                heading_W_rad=0.4,
                T_H0_from_W=transform,
            )
        )

        self.assertAlmostEqual(
            value,
            0.4,
        )

    def test_rotated_heading_transform(self):

        angle = 0.3

        c = math.cos(-angle)
        s = math.sin(-angle)

        rotation = np.asarray(
            [
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        transform = _Transform(
            rotation
        )

        value = (
            world_heading_to_headlamp_yaw(
                heading_W_rad=angle,
                T_H0_from_W=transform,
            )
        )

        self.assertAlmostEqual(
            value,
            0.0,
            places=12,
        )

    def test_stable_actor_hash(self):

        actor = CausalADBActorBox(
            scenario_id="demo",
            track_index=2,
            track_id="abc",
            object_type="TYPE_VEHICLE",
            box=Box3D(
                center_xyz=(
                    10.0,
                    1.0,
                    0.5,
                ),
                length_m=4.0,
                width_m=2.0,
                height_m=1.5,
                yaw_rad=0.1,
            ),
            stage1_anchor_center_H0_m=(
                10.0,
                1.0,
                0.5,
            ),
            stage1_anchor_bearing_H0_rad=(
                math.atan2(
                    1.0,
                    10.0,
                )
            ),
            center_consistency_error_m=0.0,
            source_time_index=10,
        )

        first = causal_actor_boxes_sha256(
            (actor,)
        )

        second = causal_actor_boxes_sha256(
            (actor,)
        )

        self.assertEqual(
            first,
            second,
        )

    def test_provenance_flags_default_false(self):

        actor = CausalADBActorBox(
            scenario_id="demo",
            track_index=1,
            track_id="1",
            object_type="TYPE_PEDESTRIAN",
            box=Box3D(
                center_xyz=(
                    5.0,
                    0.0,
                    0.8,
                ),
                length_m=0.8,
                width_m=0.6,
                height_m=1.7,
                yaw_rad=0.0,
            ),
            stage1_anchor_center_H0_m=(
                5.0,
                0.0,
                0.8,
            ),
            stage1_anchor_bearing_H0_rad=0.0,
            center_consistency_error_m=0.0,
            source_time_index=10,
        )

        self.assertFalse(
            actor.tracks_to_predict_used
        )

        self.assertFalse(
            actor.objects_of_interest_used
        )

        self.assertFalse(
            actor.future_state_used
        )


if __name__ == "__main__":
    unittest.main()
'''

    test_write = write_exact_text(
        TEST_FILE,
        test_source,
    )

    print(
        "test file =",
        test_write,
    )

    # ========================================================
    # F. Compile implementation before real data
    # ========================================================

    print()
    print(
        "===== F. CONTROLLED COMPILE ====="
    )

    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(WOMD_ADAPTER_MODULE),
            str(ADB_INIT),
            str(TEST_FILE),
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        compile_process.returncode == 0,
        (
            "Block6.1 Part2 compile failed:\n"
            + tail_text(
                compile_process.stdout
            )
        ),
    )

    print(
        "adapter/tests compile = PASS"
    )

    # Import only after compile.
    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage6.adb.geometry import (
        project_box_to_headlamp,
        wrap_angle,
    )

    from iscai_stage6.adb.grid import (
        IlluminationGridSpec,
        rectangular_extent_cell_indices,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
        causal_actor_boxes_sha256,
        validate_stage1_projection_consistency,
    )

    # ========================================================
    # G. Exact one-real-scenario read
    # ========================================================

    print()
    print(
        "===== G. ONE REAL FROZEN STAGE1 SCENARIO ====="
    )

    pilot_rows = load_jsonl(
        STAGE1_PILOT_MANIFEST
    )

    require(
        len(pilot_rows) >= 1,
        "Frozen pilot manifest is empty.",
    )

    pilot = pilot_rows[0]

    scenario_id = str(
        pilot["scenario_id"]
    )

    source_shard = Path(
        pilot["source_shard"]
    )

    record_offset = int(
        pilot.get(
            "motion_record_offset",
            0,
        )
    )

    scenario, payload_length = (
        read_scenario_at_tfrecord_offset(
            path=source_shard,
            record_offset=record_offset,
        )
    )

    require(
        str(
            scenario.scenario_id
        )
        ==
        scenario_id,
        (
            "Exact TFRecord record identity mismatch: "
            f"expected={scenario_id}, "
            f"actual={scenario.scenario_id}"
        ),
    )

    if (
        "motion_payload_length"
        in
        pilot
    ):
        require(
            payload_length
            ==
            int(
                pilot[
                    "motion_payload_length"
                ]
            ),
            (
                "Pilot payload length mismatch."
            ),
        )

    print(
        "scenario ID        =",
        scenario_id,
    )

    print(
        "record access      = EXACT OFFSET / NO SHARD SCAN"
    )

    print(
        "current_time_index =",
        int(
            scenario.current_time_index
        ),
    )

    # ========================================================
    # H. Stage1 transform + real causal boxes
    # ========================================================

    print()
    print(
        "===== H. REAL STAGE1 H0 → STAGE6 FULL BOX ====="
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario,
            headlamp_config=headlamp_config,
        )
    )

    H0_rotation = rotation_matrix(
        adapted.frames.T_H0_from_W
    )

    rotation_audit = verify_rotation(
        H0_rotation
    )

    # Frozen baseline roll/pitch are zero, so world z must map
    # to +/- H0 z. We require same positive up direction.
    world_up_H0 = (
        H0_rotation
        @
        np.asarray(
            [0.0, 0.0, 1.0],
            dtype=np.float64,
        )
    )

    require(
        np.max(
            np.abs(
                world_up_H0
                -
                np.asarray(
                    [0.0, 0.0, 1.0]
                )
            )
        )
        <=
        TOL_AXIS,
        (
            "Frozen/default H0 does not preserve "
            "+z up; yaw-only Stage6 box geometry "
            "would be invalid."
        ),
    )

    actors = (
        build_causal_adb_actor_boxes(
            scenario=scenario,
            adapted=adapted,
        )
    )

    require(
        actors,
        (
            "Real pilot contains no anchor-valid "
            "vehicle/pedestrian/cyclist actors."
        ),
    )

    class_counts = Counter(
        actor.object_type
        for actor in actors
    )

    consistency_rows = []

    maximum_center_error = 0.0
    maximum_bearing_error = 0.0

    projected_rows = []

    for actor in actors:
        check = (
            validate_stage1_projection_consistency(
                actor,
                center_tolerance_m=TOL_CENTER_M,
                bearing_tolerance_rad=TOL_BEARING_RAD,
            )
        )

        require(
            check[
                "eight_corners"
            ],
            "Real actor did not produce 8 corners.",
        )

        require(
            check[
                "theta_span_positive"
            ],
            (
                "Real actor collapsed to "
                "centroid-only angular geometry."
            ),
        )

        maximum_center_error = max(
            maximum_center_error,
            float(
                check[
                    "center_error_m"
                ]
            ),
        )

        bearing_error = float(
            check[
                "bearing_error_rad"
            ]
        )

        if math.isfinite(
            bearing_error
        ):
            maximum_bearing_error = max(
                maximum_bearing_error,
                bearing_error,
            )

        projection = (
            project_box_to_headlamp(
                actor.box,
                state_source="current_causal",
                orientation_source="current_causal",
                controller_path=True,
            )
        )

        projected_rows.append({
            "track_index":
                actor.track_index,

            "track_id":
                actor.track_id,

            "class":
                actor.object_type,

            "center_H0_m":
                list(
                    actor.box.center_xyz
                ),

            "dimensions_lwh_m": [
                actor.box.length_m,
                actor.box.width_m,
                actor.box.height_m,
            ],

            "yaw_H0_rad":
                actor.box.yaw_rad,

            "theta_center_rad":
                projection.theta_center_rad,

            "theta_span_rad":
                projection.theta_span_rad,

            "ground_range_min_m":
                projection.ground_range_min_m,

            "ground_range_max_m":
                projection.ground_range_max_m,

            "phi_min_rad":
                projection.phi_min_rad,

            "phi_max_rad":
                projection.phi_max_rad,

            "corner_count":
                int(
                    projection.corners.xyz.shape[0]
                ),
        })

    print(
        "ADB actor boxes       =",
        len(actors),
    )

    print(
        "classes               =",
        dict(
            sorted(
                class_counts.items()
            )
        ),
    )

    print(
        "Stage1 center agreement= PASS"
    )

    print(
        "Stage1 bearing agreement= PASS"
    )

    print(
        "real full-box corners = 8/8 PER ACTOR PASS"
    )

    # ========================================================
    # I. Metadata and future-mutation leakage gates
    # ========================================================

    print()
    print(
        "===== I. STRICT CAUSALITY / SELECTION INVARIANCE ====="
    )

    baseline_hash = (
        causal_actor_boxes_sha256(
            actors
        )
    )

    # --------------------------------------------------------
    # Metadata mutation: tracks_to_predict / objects_of_interest
    # --------------------------------------------------------

    metadata_mutated = type(
        scenario
    )()

    metadata_mutated.CopyFrom(
        scenario
    )

    del metadata_mutated.tracks_to_predict[:]
    del metadata_mutated.objects_of_interest[:]

    metadata_adapted = (
        adapt_causal_womd_scenario(
            metadata_mutated,
            headlamp_config=headlamp_config,
        )
    )

    metadata_actors = (
        build_causal_adb_actor_boxes(
            scenario=metadata_mutated,
            adapted=metadata_adapted,
        )
    )

    metadata_hash = (
        causal_actor_boxes_sha256(
            metadata_actors
        )
    )

    require(
        metadata_hash
        ==
        baseline_hash,
        (
            "tracks_to_predict/objects_of_interest "
            "metadata altered Stage6 ADB actor boxes."
        ),
    )

    # --------------------------------------------------------
    # Future mutation
    # --------------------------------------------------------

    future_mutated = type(
        scenario
    )()

    future_mutated.CopyFrom(
        scenario
    )

    anchor = int(
        future_mutated.current_time_index
    )

    for track in (
        future_mutated.tracks
    ):
        for time_index in range(
            anchor + 1,
            len(track.states),
        ):
            state = track.states[
                time_index
            ]

            state.center_x += 1234.5
            state.center_y -= 987.6

            if state.length > 0.0:
                state.length *= 1.1

            state.heading += 0.7

    future_adapted = (
        adapt_causal_womd_scenario(
            future_mutated,
            headlamp_config=headlamp_config,
        )
    )

    future_actors = (
        build_causal_adb_actor_boxes(
            scenario=future_mutated,
            adapted=future_adapted,
        )
    )

    future_hash = (
        causal_actor_boxes_sha256(
            future_actors
        )
    )

    require(
        future_hash
        ==
        baseline_hash,
        (
            "Future WOMD states leaked into "
            "Stage6 causal ADB geometry."
        ),
    )

    print(
        "tracks_to_predict mutation = INVARIANT PASS"
    )

    print(
        "objects_of_interest mutation= INVARIANT PASS"
    )

    print(
        "future-state mutation       = INVARIANT PASS"
    )

    print(
        "causal geometry SHA256      =",
        baseline_hash,
    )

    # ========================================================
    # J. Diagnostic-only raster smoke
    # ========================================================

    print()
    print(
        "===== J. DIAGNOSTIC-ONLY THETA-R RASTER SMOKE ====="
    )

    maximum_range = max(
        float(
            row[
                "ground_range_max_m"
            ]
        )
        for row in projected_rows
    )

    diagnostic_range_max = max(
        1.0,
        math.ceil(
            maximum_range
            +
            1.0
        ),
    )

    diagnostic_grid = (
        IlluminationGridSpec(
            theta_min_rad=-math.pi,
            theta_max_rad=math.pi,
            range_min_m=0.0,
            range_max_m=(
                diagnostic_range_max
            ),
            n_theta=72,
            n_range=100,
        )
    )

    rasterized = 0

    for actor in actors:
        projection = (
            project_box_to_headlamp(
                actor.box,
                state_source="current_causal",
                orientation_source="current_causal",
                controller_path=True,
            )
        )

        theta_indices, range_indices = (
            rectangular_extent_cell_indices(
                diagnostic_grid,
                theta_center_rad=(
                    projection
                    .theta_center_rad
                ),
                theta_span_rad=(
                    projection
                    .theta_span_rad
                ),
                range_min_m=(
                    projection
                    .ground_range_min_m
                ),
                range_max_m=(
                    projection
                    .ground_range_max_m
                ),
            )
        )

        require(
            theta_indices.size > 0,
            (
                "Diagnostic grid lost real "
                "actor angular extent."
            ),
        )

        require(
            range_indices.size > 0,
            (
                "Diagnostic grid lost real "
                "actor range extent."
            ),
        )

        rasterized += 1

    print(
        "real actors rasterized =",
        rasterized,
    )

    print(
        "grid semantics          = DIAGNOSTIC_ONLY"
    )

    print(
        "scientific FOV frozen   = NO"
    )

    print(
        "scientific resolution   = NO"
    )

    print(
        "final grid freeze block = 6.8 PRE-FORMAL"
    )

    # ========================================================
    # K. Stage6 integration contract
    # ========================================================

    print()
    print(
        "===== K. BLOCK6.1 INTEGRATION CONTRACT ====="
    )

    integration_contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.1",

        "part":
            "2/2",

        "status":
            "FROZEN_GEOMETRY_INTERFACE",

        "Stage1": {
            "closure":
                str(
                    STAGE1_CLOSURE
                ),

            "closure_sha256":
                sha256_file(
                    STAGE1_CLOSURE
                ),

            "pilot_manifest_sha256":
                EXPECTED[
                    "pilot_manifest"
                ],

            "H0":
                "front_face_midpoint_surrogate",

            "axes":
                "+x forward, +y left, +z up",

            "headlamp_config_module":
                headlamp_module_name,

            "headlamp_config_source":
                str(
                    headlamp_source
                ),

            "headlamp_default_config":
                headlamp_config_json,

            "source_hashes":
                stage1_source_hashes,

            "default_roll_rad":
                roll,

            "default_pitch_rad":
                pitch,

            "yaw_only_Box3D_compatible":
                True,
        },

        "ADB_actor_geometry": {
            "classes": [
                "TYPE_VEHICLE",
                "TYPE_PEDESTRIAN",
                "TYPE_CYCLIST",
            ],

            "requires_anchor_valid":
                True,

            "exclude_SDC":
                True,

            "tracks_to_predict_selector":
                False,

            "objects_of_interest_selector":
                False,

            "future_validity_selector":
                False,

            "full_box":
                True,

            "corner_count":
                8,

            "dimensions":
                [
                    "length",
                    "width",
                    "height",
                ],

            "heading_source":
                "current_causal_anchor",

            "heading_transform":
                (
                    "world forward vector rotated "
                    "by frozen T_H0_from_W"
                ),

            "centroid_only":
                False,
        },

        "Stage1_consistency": {
            "center_H0_crosscheck":
                True,

            "bearing_H0_crosscheck":
                True,

            "maximum_center_error_m":
                maximum_center_error,

            "maximum_bearing_error_rad":
                maximum_bearing_error,

            "center_tolerance_m":
                TOL_CENTER_M,

            "bearing_tolerance_rad":
                TOL_BEARING_RAD,
        },

        "causality": {
            "tracks_to_predict_mutation_invariant":
                True,

            "objects_of_interest_mutation_invariant":
                True,

            "future_state_mutation_invariant":
                True,

            "future_GT_controller_use":
                False,

            "causal_actor_geometry_sha256":
                baseline_hash,
        },

        "illumination_grid": {
            "coordinate_interface":
                "theta-ground_range",

            "communication_codebook_reused":
                False,

            "final_numeric_FOV_frozen":
                False,

            "final_numeric_range_frozen":
                False,

            "final_numeric_resolution_frozen":
                False,

            "freeze_stage":
                "Block6.8 development-only pre-formal freeze",

            "reason":
                (
                    "PDF defines illumination angular/"
                    "pixel space but not a mandatory "
                    "numerical discretization; avoid "
                    "inventing PDF parameters."
                ),

            "Block6_1_diagnostic_grid":
                {
                    "status":
                        "DIAGNOSTIC_ONLY_NOT_SCIENTIFIC_GRID",

                    "theta_min_rad":
                        -math.pi,

                    "theta_max_rad":
                        math.pi,

                    "range_min_m":
                        0.0,

                    "range_max_m":
                        diagnostic_range_max,

                    "n_theta":
                        72,

                    "n_range":
                        100,
                },
        },

        "timing_clarification": {
            "Part1_text_said_numeric_grid_in_Part2":
                True,

            "superseded":
                True,

            "correct_binding_rule":
                (
                    "final ADB grid values freeze in "
                    "Block6.8 before formal evaluation"
                ),

            "scientific_change":
                False,

            "reason":
                (
                    "preserves strict audited Stage6 "
                    "pre-formal parameter-freeze plan "
                    "and avoids arbitrary early values"
                ),
        },
    }

    contract_status = (
        write_exact_json(
            INTEGRATION_CONTRACT,
            integration_contract,
        )
    )

    print(
        "integration contract =",
        contract_status,
    )

    print(
        "contract SHA256      =",
        sha256_file(
            INTEGRATION_CONTRACT
        ),
    )

    # ========================================================
    # L. Save concise real-scene evidence
    # ========================================================

    smoke_payload = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.1",

        "status":
            "PASS_REAL_FULL_BOX_PROJECTION",

        "scenario": {
            "scenario_id":
                scenario_id,

            "source_shard":
                str(
                    source_shard
                ),

            "record_offset":
                record_offset,

            "payload_length":
                payload_length,

            "access_mode":
                "exact_record_offset_no_shard_scan",
        },

        "headlamp": {
            "mode":
                "front_face_midpoint_surrogate",

            "default_config":
                headlamp_config_json,

            "rotation_audit":
                rotation_audit,

            "world_up_in_H0":
                [
                    float(v)
                    for v in world_up_H0
                ],
        },

        "actors": {
            "total_supported_anchor_valid":
                len(actors),

            "class_counts":
                dict(
                    sorted(
                        class_counts.items()
                    )
                ),

            "maximum_center_error_m":
                maximum_center_error,

            "maximum_bearing_error_rad":
                maximum_bearing_error,

            "causal_geometry_sha256":
                baseline_hash,

            "rows":
                projected_rows,
        },

        "causality": {
            "tracks_to_predict_mutation_hash":
                metadata_hash,

            "future_mutation_hash":
                future_hash,

            "baseline_hash":
                baseline_hash,

            "all_equal":
                (
                    baseline_hash
                    ==
                    metadata_hash
                    ==
                    future_hash
                ),
        },

        "diagnostic_grid": {
            "status":
                "DIAGNOSTIC_ONLY_NOT_SCIENTIFIC_GRID",

            "real_actors_rasterized":
                rasterized,

            "final_scientific_grid_frozen":
                False,
        },
    }

    smoke_status = write_exact_json(
        REAL_SMOKE,
        smoke_payload,
    )

    print(
        "real-scene evidence =",
        smoke_status,
    )

    # ========================================================
    # M. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== L. FULL STAGE6 REGRESSION ====="
    )

    rc, output = (
        run_stage6_tests()
    )

    for line in (
        output.splitlines()
    ):
        stripped = line.strip()

        if (
            stripped.startswith(
                "Ran "
            )
            or
            stripped == "OK"
            or
            stripped.startswith(
                "FAILED"
            )
        ):
            print(stripped)

    require(
        rc == 0,
        (
            "Stage6 regression failed. "
            f"Full log saved to {REGRESSION_LOG}\n"
            + tail_text(
                output,
                50,
            )
        ),
    )

    # ========================================================
    # N. Immutability / storage
    # ========================================================

    print()
    print(
        "===== M. IMMUTABILITY / STORAGE ====="
    )

    protected_after = {
        str(path):
            sha256_file(path)
        for path in required
    }

    # ADB_INIT is intentionally Stage6-modified and is not
    # in the protected prerequisite set.
    changed = [
        path
        for path in protected_before
        if (
            protected_before[path]
            !=
            protected_after[path]
        )
    ]

    require(
        not changed,
        (
            "Frozen prerequisite changed: "
            + ", ".join(changed)
        ),
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB reserve violated: "
            f"{free_gib:.3f} GiB."
        ),
    )

    print(
        "Stage1 frozen sources/artifacts = UNCHANGED"
    )

    print(
        "Stage5 closure                 = UNCHANGED"
    )

    print(
        "Block6.0 / Part1 evidence      = UNCHANGED"
    )

    print(
        "free GiB                       =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve                = PASS"
    )

    # ========================================================
    # O. Block6.1 closure
    # ========================================================

    print()
    print(
        "===== N. BLOCK6.1 CLOSURE ====="
    )

    closure_payload = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.1",

        "status":
            "PASS_COMPLETE",

        "Part1": {
            "status":
                "PASS_GEOMETRY_CORE",

            "report":
                str(
                    PART1_REPORT
                ),

            "sha256":
                EXPECTED[
                    "part1_report"
                ],
        },

        "Part2": {
            "status":
                "PASS_STAGE1_REAL_INTEGRATION",

            "integration_contract":
                str(
                    INTEGRATION_CONTRACT
                ),

            "integration_contract_sha256":
                sha256_file(
                    INTEGRATION_CONTRACT
                ),

            "real_scene_projection":
                str(
                    REAL_SMOKE
                ),

            "real_scene_projection_sha256":
                sha256_file(
                    REAL_SMOKE
                ),

            "womd_geometry_module":
                str(
                    WOMD_ADAPTER_MODULE
                ),

            "womd_geometry_sha256":
                sha256_file(
                    WOMD_ADAPTER_MODULE
                ),
        },

        "frozen_geometry_contract": {
            "H0":
                "front_face_midpoint_surrogate",

            "axes":
                "+x forward, +y left, +z up",

            "full_3D_box_projection":
                True,

            "centroid_only":
                False,

            "box_corners":
                8,

            "length_width_height":
                True,

            "current_causal_heading":
                True,

            "class_support": [
                "vehicle",
                "pedestrian",
                "cyclist",
            ],

            "tracks_to_predict_as_ADB_selector":
                False,

            "objects_of_interest_as_ADB_selector":
                False,

            "future_GT_controller_use":
                False,

            "communication_codebook_reused":
                False,
        },

        "real_data_gate": {
            "scenario_id":
                scenario_id,

            "exact_record_access":
                True,

            "dataset_scan":
                False,

            "supported_anchor_valid_actors":
                len(actors),

            "class_counts":
                dict(
                    sorted(
                        class_counts.items()
                    )
                ),

            "Stage1_center_consistency":
                "PASS",

            "Stage1_bearing_consistency":
                "PASS",

            "metadata_mutation_invariance":
                "PASS",

            "future_mutation_invariance":
                "PASS",

            "full_box_projection":
                "PASS",
        },

        "illumination_grid": {
            "interface_frozen":
                True,

            "interface":
                "theta-ground_range",

            "final_numeric_values_frozen":
                False,

            "freeze_at":
                "BLOCK6.8_PRE_FORMAL",

            "diagnostic_grid_not_scientific":
                True,
        },

        "regression": {
            "Stage6":
                "PASS",

            "full_log":
                str(
                    REGRESSION_LOG
                ),
        },

        "execution": {
            "one_real_scenario":
                True,

            "corpus_scan":
                False,

            "training":
                False,

            "trajectory_model_inference":
                False,

            "reactive_ADB_execution":
                False,

            "predictive_ADB_execution":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block6.2: static ADB and "
                "original Part-A reactive "
                "L(theta,r) adapter"
            ),
    }

    closure_status = (
        write_exact_json(
            CLOSURE,
            closure_payload,
        )
    )

    print(
        "Block6.1 closure =",
        closure_status,
    )

    print(
        "closure SHA256   =",
        sha256_file(
            CLOSURE
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.1 PART 2/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Stage1 H0 contract          = VERIFIED PASS"
    )

    print(
        "H0 axes                     = +X FORWARD / +Y LEFT / +Z UP"
    )

    print(
        "headlamp extrinsic          = FROZEN STAGE1 BASELINE"
    )

    print(
        "real WOMD scenario          = EXACT RECORD PASS"
    )

    print(
        "real full-box projection    = PASS"
    )

    print(
        "Stage1 center cross-check   = PASS"
    )

    print(
        "Stage1 bearing cross-check  = PASS"
    )

    print(
        "vehicle/ped/cyclist support = PASS"
    )

    print(
        "tracks_to_predict selector  = NO / MUTATION INVARIANT"
    )

    print(
        "objects_of_interest selector= NO / MUTATION INVARIANT"
    )

    print(
        "future-state leakage        = NO / MUTATION INVARIANT"
    )

    print(
        "centroid-only ADB           = FORBIDDEN"
    )

    print(
        "communication codebook      = NOT REUSED"
    )

    print(
        "illumination interface      = THETA-GROUND-RANGE FROZEN"
    )

    print(
        "final grid numerics         = NOT PREMATURELY FROZEN"
    )

    print(
        "final numeric freeze        = BLOCK6.8 PRE-FORMAL"
    )

    print(
        "Stage6 regression           = PASS"
    )

    print(
        "upstream modified           = NO"
    )

    print(
        "training/model inference    = NO"
    )

    print(
        "STATUS = PASS_COMPLETE"
    )

    print(
        "closure =",
        CLOSURE,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.1 PART 2/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=12
    )

    print()
    print(
        "Stage1 modified         = NO"
    )

    print(
        "Stage5 modified         = NO"
    )

    print(
        "corpus scan             = NO"
    )

    print(
        "training                = NO"
    )

    print(
        "trajectory inference    = NO"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "parameter tuning        = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no non-zero sys.exit().

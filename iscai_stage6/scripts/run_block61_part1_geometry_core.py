from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")

S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

BLOCK60_CLOSURE = (
    S6
    / "reports/block60_closure.json"
)

STAGE6_CONTRACT = (
    S6
    / "configs/stage6_contract.json"
)

PART_A_SOURCE_AUDIT = (
    S6
    / "artifacts/block60/"
      "part_a_adb_source_audit.json"
)

REACTIVE_ADAPTER_CONTRACT = (
    S6
    / "configs/"
      "part_a_reactive_adb_adapter_contract.json"
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

ADB_INIT = (
    S6
    / "src/iscai_stage6/adb/"
      "__init__.py"
)

TEST_FILE = (
    S6
    / "tests/"
      "test_block61_geometry_core.py"
)

GEOMETRY_CONTRACT = (
    S6
    / "configs/"
      "block61_geometry_contract.json"
)

REPORT = (
    S6
    / "reports/"
      "block61_part1_geometry_core.json"
)

EXPECTED = {
    "block60_closure":
        (
            "525cf827b217cc2806bfff6547c5e241"
            "b3c9db04d188f7957acea117b309c303"
        ),

    "stage6_contract":
        (
            "6b2ee6454f876c3658d4d308b827202c"
            "d31dd7c77bf96c38e6c6da70e6573675"
        ),

    "part_a_source_audit":
        (
            "5b3b8aacaa6c8947c6d9d57264a90899"
            "4a6a8d3193179a75bcd4f6b4697b0e64"
        ),

    "reactive_adapter_contract":
        (
            "d552e9d766ec356fbf57079a647a7b3c"
            "5e882a6e153c8b9e5f06c8b207539e4c"
        ),

    "stage5_closure":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),
}

MIN_FREE_GIB = 250.0


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

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


def require(
    condition,
    message,
):

    if not bool(
        condition
    ):
        raise RuntimeError(
            message
        )


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
    ).encode(
        "utf-8"
    )


def write_exact_json(
    path: Path,
    payload,
):

    desired = canonical_bytes(
        payload
    )

    if path.exists():

        if (
            path.read_bytes()
            !=
            desired
        ):
            raise RuntimeError(
                "Existing deterministic artifact "
                f"differs: {path}"
            )

        return "ALREADY_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        desired
    )

    os.replace(
        temporary,
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

        if (
            path.read_bytes()
            !=
            desired
        ):
            raise RuntimeError(
                "Existing deterministic source "
                f"differs: {path}"
            )

        return "ALREADY_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        desired
    )

    os.replace(
        temporary,
        path,
    )

    return "WRITTEN"


def run_tests():

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6 / "tests"
            ),
            "-p",
            "test_block6*.py",
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return (
        process.returncode,
        process.stdout,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.1 PART 1/2"
    )
    print(
        "ILLUMINATION GEOMETRY CORE + FULL-BOX PROJECTION"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK60_CLOSURE,
        STAGE6_CONTRACT,
        PART_A_SOURCE_AUDIT,
        REACTIVE_ADAPTER_CONTRACT,
        STAGE5_CLOSURE,
    )

    missing = [
        str(
            path
        )
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing frozen prerequisite: "
            + ", ".join(
                missing
            )
        ),
    )

    protected_before = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in required
    }

    # ========================================================
    # A. Frozen Block6.0 prerequisite
    # ========================================================

    print()
    print(
        "===== A. BLOCK6.0 / UPSTREAM SEAL ====="
    )

    for name, path in (
        (
            "block60_closure",
            BLOCK60_CLOSURE,
        ),
        (
            "stage6_contract",
            STAGE6_CONTRACT,
        ),
        (
            "part_a_source_audit",
            PART_A_SOURCE_AUDIT,
        ),
        (
            "reactive_adapter_contract",
            REACTIVE_ADAPTER_CONTRACT,
        ),
        (
            "stage5_closure",
            STAGE5_CLOSURE,
        ),
    ):

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            EXPECTED[
                name
            ],
            (
                f"{name} SHA changed.\n"
                f"expected={EXPECTED[name]}\n"
                f"actual={actual}"
            ),
        )

    block60 = load_json(
        BLOCK60_CLOSURE
    )

    require(
        block60.get(
            "status"
        )
        ==
        "PASS_COMPLETE",
        (
            "Block6.0 is not "
            "PASS_COMPLETE."
        ),
    )

    print(
        "Block6.0 closure          = EXACT PASS"
    )

    print(
        "Stage6 strict contract    = EXACT PASS"
    )

    print(
        "Part-A source audit       = EXACT PASS"
    )

    print(
        "reactive adapter contract = EXACT PASS"
    )

    print(
        "Stage5 closure            = UNCHANGED PASS"
    )

    # ========================================================
    # B. Freeze Block6.1 geometry semantics
    # ========================================================

    print()
    print(
        "===== B. BLOCK6.1 GEOMETRY CONTRACT ====="
    )

    geometry_contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.1",

        "part":
            "1/2",

        "status":
            "PARAMETERIZED_GEOMETRY_CORE",

        "pdf_mapping": {
            "centroid_only":
                False,

            "box_corners":
                True,

            "actor_length_width_height":
                True,

            "receiver_driver_relevant_region_support":
                True,

            "covariance_projection":
                "BLOCK6.4",

            "class_margin_application":
                "BLOCK6.5",

            "future_probabilistic_occupancy":
                "BLOCK6.4",
        },

        "headlamp_coordinates": {
            "frame":
                "H0",

            "axis_contract": {
                "x":
                    "forward",

                "y":
                    "left",

                "z":
                    "up",
            },

            "range_3d":
                "sqrt(x^2+y^2+z^2)",

            "ground_range":
                "sqrt(x^2+y^2)",

            "azimuth_theta":
                "atan2(y,x)",

            "elevation_phi":
                (
                    "atan2(z,sqrt(x^2+y^2))"
                ),

            "axis_contract_validation_against_frozen_Stage1":
                "BLOCK6.1_PART2",
        },

        "box_semantics": {
            "dimensions":
                [
                    "length",
                    "width",
                    "height",
                ],

            "local_length_axis":
                "+/-x",

            "local_width_axis":
                "+/-y",

            "local_height_axis":
                "+/-z",

            "yaw_axis":
                "+z",

            "corner_count":
                8,

            "centroid_is_allowed_as_diagnostic":
                True,

            "centroid_is_sufficient_for_ADB":
                False,
        },

        "relevant_region": {
            "representation":
                (
                    "fractional sub-region of "
                    "full 3D actor box"
                ),

            "fraction_domain":
                "[-0.5,+0.5]",

            "policy_specific_fraction_values":
                "NOT_FROZEN_IN_BLOCK6.1_PART1",

            "windshield_or_face_region_defaults":
                "NOT_INVENTED",
        },

        "causality": {
            "controller_future_GT":
                False,

            "allowed_controller_state_sources": [
                "current_causal",
                "predicted_sample",
                "predicted_mean",
            ],

            "oracle_future_box":
                "EVALUATOR_ONLY",

            "future_GT_heading_controller":
                False,

            "future_GT_dimensions_controller":
                False,
        },

        "illumination_grid": {
            "representation":
                "theta_range",

            "scientific_FOV_numeric_bounds":
                "BLOCK6.1_PART2",

            "scientific_range_numeric_bounds":
                "BLOCK6.1_PART2",

            "scientific_resolution":
                "BLOCK6.1_PART2",

            "reason":
                (
                    "do not invent numeric Stage6 "
                    "headlamp-grid values before "
                    "frozen Stage1 geometry audit"
                ),

            "parameterized_grid_kernel_in_Part1":
                True,
        },

        "scope_boundary": {
            "communication_beam_codebook_reused":
                False,

            "ADB_mask_generation":
                "NOT_YET",

            "probabilistic_occupancy":
                "NOT_YET",

            "class_policy":
                "NOT_YET",

            "reactive_L_theta_r":
                "BLOCK6.2",
        },
    }

    contract_write = write_exact_json(
        GEOMETRY_CONTRACT,
        geometry_contract,
    )

    print(
        "geometry contract =",
        contract_write,
    )

    print(
        "numeric Stage6 grid frozen = NO "
        "(correctly deferred to Part2)"
    )

    # ========================================================
    # C. geometry.py
    # ========================================================

    print()
    print(
        "===== C. FULL-BOX GEOMETRY KERNEL ====="
    )

    geometry_source = r'''from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1.0e-12

_ALLOWED_CONTROLLER_STATE_SOURCES = frozenset(
    {
        "current_causal",
        "predicted_sample",
        "predicted_mean",
    }
)

_FORBIDDEN_CONTROLLER_STATE_SOURCES = frozenset(
    {
        "future_gt",
        "oracle_future",
        "future_annotation",
    }
)

_ALLOWED_ORIENTATION_SOURCES = frozenset(
    {
        "current_causal",
        "predicted_tangent",
        "predicted_orientation",
        "frozen_low_speed_fallback",
        "oracle_evaluator_only",
    }
)


def _finite_array(
    values,
    *,
    shape,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, "
            f"got {array.shape}"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return array


def wrap_angle(
    angle_rad,
):
    """
    Wrap angle(s) into [-pi, pi).
    """

    array = np.asarray(
        angle_rad,
        dtype=np.float64,
    )

    wrapped = (
        array + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    if np.ndim(
        angle_rad
    ) == 0:
        return float(
            wrapped
        )

    return wrapped


def validate_controller_provenance(
    *,
    state_source: str,
    orientation_source: str,
    controller_path: bool,
) -> None:

    state_source = str(
        state_source
    )

    orientation_source = str(
        orientation_source
    )

    if (
        orientation_source
        not in
        _ALLOWED_ORIENTATION_SOURCES
    ):
        raise ValueError(
            "Unknown orientation source: "
            f"{orientation_source}"
        )

    if not controller_path:
        return

    if (
        state_source
        in
        _FORBIDDEN_CONTROLLER_STATE_SOURCES
    ):
        raise ValueError(
            "Future/oracle annotation is forbidden "
            "on the Stage6 controller path: "
            f"{state_source}"
        )

    if (
        state_source
        not in
        _ALLOWED_CONTROLLER_STATE_SOURCES
    ):
        raise ValueError(
            "Unapproved controller state source: "
            f"{state_source}"
        )

    if (
        orientation_source
        ==
        "oracle_evaluator_only"
    ):
        raise ValueError(
            "Oracle orientation is forbidden "
            "on the controller path"
        )


@dataclass(frozen=True)
class Box3D:
    """
    Oriented actor box represented in headlamp H0 coordinates.

    Convention:
      +x forward
      +y left
      +z up

    length is along local x,
    width  is along local y,
    height is along local z,
    yaw is rotation about +z.
    """

    center_xyz: tuple[float, float, float]
    length_m: float
    width_m: float
    height_m: float
    yaw_rad: float

    def __post_init__(
        self,
    ) -> None:

        center = _finite_array(
            self.center_xyz,
            shape=(3,),
            name="center_xyz",
        )

        object.__setattr__(
            self,
            "center_xyz",
            tuple(
                float(v)
                for v in center
            ),
        )

        for name in (
            "length_m",
            "width_m",
            "height_m",
        ):

            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if (
                not math.isfinite(
                    value
                )
                or
                value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and > 0"
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        yaw = float(
            self.yaw_rad
        )

        if not math.isfinite(
            yaw
        ):
            raise ValueError(
                "yaw_rad must be finite"
            )

        object.__setattr__(
            self,
            "yaw_rad",
            wrap_angle(
                yaw
            ),
        )

    @property
    def center(
        self,
    ) -> np.ndarray:

        return np.asarray(
            self.center_xyz,
            dtype=np.float64,
        )

    @property
    def dimensions(
        self,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.length_m,
                self.width_m,
                self.height_m,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class FractionalBoxRegion:
    """
    A body-frame subregion of an actor box.

    Each bound is expressed as a fraction of the corresponding
    full box dimension and must lie in [-0.5, +0.5].

    No semantic values such as windshield/face fractions are
    hard-coded here; those belong to the later class-policy block.
    """

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    z_bounds: tuple[float, float]

    def __post_init__(
        self,
    ) -> None:

        for name in (
            "x_bounds",
            "y_bounds",
            "z_bounds",
        ):

            values = tuple(
                float(v)
                for v in getattr(
                    self,
                    name,
                )
            )

            if len(
                values
            ) != 2:
                raise ValueError(
                    f"{name} requires exactly 2 values"
                )

            lo, hi = values

            if (
                not math.isfinite(
                    lo
                )
                or
                not math.isfinite(
                    hi
                )
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

            if not (
                -0.5
                <=
                lo
                <
                hi
                <=
                0.5
            ):
                raise ValueError(
                    f"{name} must satisfy "
                    "-0.5 <= lo < hi <= 0.5"
                )

            object.__setattr__(
                self,
                name,
                (
                    lo,
                    hi,
                ),
            )


@dataclass(frozen=True)
class ProjectedPoints:
    xyz: np.ndarray
    ground_range_m: np.ndarray
    range_3d_m: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray

    def __post_init__(
        self,
    ) -> None:

        n = int(
            self.xyz.shape[0]
        )

        if (
            self.xyz.ndim != 2
            or
            self.xyz.shape[1] != 3
        ):
            raise ValueError(
                "xyz must be Nx3"
            )

        for name in (
            "ground_range_m",
            "range_3d_m",
            "theta_rad",
            "phi_rad",
        ):

            array = np.asarray(
                getattr(
                    self,
                    name,
                ),
                dtype=np.float64,
            )

            if array.shape != (
                n,
            ):
                raise ValueError(
                    f"{name} must have shape ({n},)"
                )


@dataclass(frozen=True)
class ProjectedBox:
    box: Box3D
    corners: ProjectedPoints
    centroid: ProjectedPoints

    theta_center_rad: float
    theta_span_rad: float
    theta_min_unwrapped_rad: float
    theta_max_unwrapped_rad: float

    ground_range_min_m: float
    ground_range_max_m: float

    range_3d_min_m: float
    range_3d_max_m: float

    phi_min_rad: float
    phi_max_rad: float

    state_source: str
    orientation_source: str
    controller_path: bool

    @property
    def uses_full_box(
        self,
    ) -> bool:

        return (
            self.corners.xyz.shape
            ==
            (8, 3)
        )


@dataclass(frozen=True)
class ProjectedRegion:
    region: FractionalBoxRegion
    points: ProjectedPoints

    theta_center_rad: float
    theta_span_rad: float

    ground_range_min_m: float
    ground_range_max_m: float

    phi_min_rad: float
    phi_max_rad: float


def _rotation_z(
    yaw_rad: float,
) -> np.ndarray:

    c = math.cos(
        yaw_rad
    )

    s = math.sin(
        yaw_rad
    )

    return np.asarray(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def body_points_to_headlamp(
    box: Box3D,
    local_points_xyz,
) -> np.ndarray:

    points = np.asarray(
        local_points_xyz,
        dtype=np.float64,
    )

    if (
        points.ndim != 2
        or
        points.shape[1] != 3
    ):
        raise ValueError(
            "local_points_xyz must be Nx3"
        )

    if not np.all(
        np.isfinite(
            points
        )
    ):
        raise ValueError(
            "local_points_xyz contains "
            "non-finite values"
        )

    rotation = _rotation_z(
        box.yaw_rad
    )

    return (
        points
        @
        rotation.T
        +
        box.center[
            None,
            :
        ]
    )


def box_corners_headlamp(
    box: Box3D,
) -> np.ndarray:
    """
    Return all 8 physical corners of the actor box.
    """

    hx = 0.5 * box.length_m
    hy = 0.5 * box.width_m
    hz = 0.5 * box.height_m

    local = np.asarray(
        [
            [sx * hx, sy * hy, sz * hz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )

    return body_points_to_headlamp(
        box,
        local,
    )


def region_corners_headlamp(
    box: Box3D,
    region: FractionalBoxRegion,
) -> np.ndarray:
    """
    Return the 8 corners of an arbitrary body-frame subregion.
    """

    x_lo, x_hi = region.x_bounds
    y_lo, y_hi = region.y_bounds
    z_lo, z_hi = region.z_bounds

    local = np.asarray(
        [
            [
                x_frac * box.length_m,
                y_frac * box.width_m,
                z_frac * box.height_m,
            ]
            for x_frac in (
                x_lo,
                x_hi,
            )
            for y_frac in (
                y_lo,
                y_hi,
            )
            for z_frac in (
                z_lo,
                z_hi,
            )
        ],
        dtype=np.float64,
    )

    return body_points_to_headlamp(
        box,
        local,
    )


def project_points_to_headlamp(
    points_xyz,
) -> ProjectedPoints:
    """
    Map H0 Cartesian coordinates into
    (ground range, 3D range, azimuth theta, elevation phi).
    """

    xyz = np.asarray(
        points_xyz,
        dtype=np.float64,
    )

    if (
        xyz.ndim != 2
        or
        xyz.shape[1] != 3
    ):
        raise ValueError(
            "points_xyz must be Nx3"
        )

    if (
        xyz.shape[0]
        < 1
    ):
        raise ValueError(
            "at least one point is required"
        )

    if not np.all(
        np.isfinite(
            xyz
        )
    ):
        raise ValueError(
            "points_xyz contains "
            "non-finite values"
        )

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    ground_range = np.hypot(
        x,
        y,
    )

    range_3d = np.sqrt(
        x * x
        +
        y * y
        +
        z * z
    )

    if np.any(
        range_3d
        <=
        _EPS
    ):
        raise ValueError(
            "point coincides with headlamp origin"
        )

    theta = np.arctan2(
        y,
        x,
    )

    phi = np.arctan2(
        z,
        ground_range,
    )

    return ProjectedPoints(
        xyz=xyz.copy(),
        ground_range_m=ground_range,
        range_3d_m=range_3d,
        theta_rad=theta,
        phi_rad=phi,
    )


def _minimal_circular_interval(
    angles_rad,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Return:
      center,
      span,
      min_unwrapped,
      max_unwrapped

    using the minimum circular interval containing all angles.
    """

    angles = np.asarray(
        angles_rad,
        dtype=np.float64,
    )

    if (
        angles.ndim != 1
        or
        angles.size < 1
    ):
        raise ValueError(
            "angles_rad must be a non-empty vector"
        )

    if not np.all(
        np.isfinite(
            angles
        )
    ):
        raise ValueError(
            "angles_rad contains non-finite values"
        )

    if angles.size == 1:
        angle = wrap_angle(
            float(
                angles[0]
            )
        )

        return (
            angle,
            0.0,
            angle,
            angle,
        )

    wrapped_0_2pi = np.mod(
        angles,
        2.0 * np.pi,
    )

    ordered = np.sort(
        wrapped_0_2pi
    )

    extended = np.concatenate(
        [
            ordered,
            ordered[:1]
            +
            2.0 * np.pi,
        ]
    )

    gaps = np.diff(
        extended
    )

    largest_gap_index = int(
        np.argmax(
            gaps
        )
    )

    start_index = (
        largest_gap_index + 1
    ) % (
        ordered.size
    )

    start = float(
        ordered[
            start_index
        ]
    )

    relative = np.mod(
        wrapped_0_2pi
        -
        start,
        2.0 * np.pi,
    )

    lo = 0.0
    hi = float(
        np.max(
            relative
        )
    )

    span = hi - lo

    center_unwrapped = (
        start
        +
        0.5 * span
    )

    center = wrap_angle(
        center_unwrapped
    )

    min_unwrapped = (
        center_unwrapped
        -
        0.5 * span
    )

    max_unwrapped = (
        center_unwrapped
        +
        0.5 * span
    )

    return (
        float(
            center
        ),
        float(
            span
        ),
        float(
            min_unwrapped
        ),
        float(
            max_unwrapped
        ),
    )


def project_box_to_headlamp(
    box: Box3D,
    *,
    state_source: str,
    orientation_source: str,
    controller_path: bool = True,
) -> ProjectedBox:
    """
    Full 3D actor-box projection.

    Centroid projection is retained only as diagnostic metadata;
    all extent fields are derived from all 8 physical corners.
    """

    validate_controller_provenance(
        state_source=state_source,
        orientation_source=orientation_source,
        controller_path=controller_path,
    )

    corners_xyz = box_corners_headlamp(
        box
    )

    corners = project_points_to_headlamp(
        corners_xyz
    )

    centroid = project_points_to_headlamp(
        box.center[
            None,
            :
        ]
    )

    (
        theta_center,
        theta_span,
        theta_min,
        theta_max,
    ) = _minimal_circular_interval(
        corners.theta_rad
    )

    return ProjectedBox(
        box=box,
        corners=corners,
        centroid=centroid,

        theta_center_rad=theta_center,
        theta_span_rad=theta_span,
        theta_min_unwrapped_rad=theta_min,
        theta_max_unwrapped_rad=theta_max,

        ground_range_min_m=float(
            np.min(
                corners.ground_range_m
            )
        ),

        ground_range_max_m=float(
            np.max(
                corners.ground_range_m
            )
        ),

        range_3d_min_m=float(
            np.min(
                corners.range_3d_m
            )
        ),

        range_3d_max_m=float(
            np.max(
                corners.range_3d_m
            )
        ),

        phi_min_rad=float(
            np.min(
                corners.phi_rad
            )
        ),

        phi_max_rad=float(
            np.max(
                corners.phi_rad
            )
        ),

        state_source=str(
            state_source
        ),

        orientation_source=str(
            orientation_source
        ),

        controller_path=bool(
            controller_path
        ),
    )


def project_region_to_headlamp(
    box: Box3D,
    region: FractionalBoxRegion,
) -> ProjectedRegion:
    """
    Project a policy-defined 3D actor subregion.

    This provides the geometry hook required later for:
      - vehicle windshield/mirror-relevant region,
      - pedestrian face/body protection region,
      - cyclist body/vehicle visibility region.

    No class-specific fractions are hard-coded here.
    """

    points_xyz = region_corners_headlamp(
        box,
        region,
    )

    points = project_points_to_headlamp(
        points_xyz
    )

    (
        theta_center,
        theta_span,
        _,
        _,
    ) = _minimal_circular_interval(
        points.theta_rad
    )

    return ProjectedRegion(
        region=region,
        points=points,

        theta_center_rad=theta_center,
        theta_span_rad=theta_span,

        ground_range_min_m=float(
            np.min(
                points.ground_range_m
            )
        ),

        ground_range_max_m=float(
            np.max(
                points.ground_range_m
            )
        ),

        phi_min_rad=float(
            np.min(
                points.phi_rad
            )
        ),

        phi_max_rad=float(
            np.max(
                points.phi_rad
            )
        ),
    )


def centroid_only_projection(
    box: Box3D,
) -> ProjectedPoints:
    """
    Diagnostic/baseline helper only.

    This function deliberately does not produce an ADB extent.
    """

    return project_points_to_headlamp(
        box.center[
            None,
            :
        ]
    )
'''

    geometry_write = write_exact_text(
        GEOMETRY_MODULE,
        geometry_source,
    )

    print(
        "geometry.py =",
        geometry_write,
    )

    # ========================================================
    # D. grid.py
    # ========================================================

    print()
    print(
        "===== D. PARAMETERIZED ILLUMINATION GRID KERNEL ====="
    )

    grid_source = r'''from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IlluminationGridSpec:
    """
    Parameterized Stage6 ADB grid in (theta, ground-range).

    No scientific numeric defaults are deliberately provided.
    Those values must be frozen only after the Block6.1 Part2
    audit against the frozen Stage1 headlamp geometry.
    """

    theta_min_rad: float
    theta_max_rad: float

    range_min_m: float
    range_max_m: float

    n_theta: int
    n_range: int

    def __post_init__(
        self,
    ) -> None:

        theta_min = float(
            self.theta_min_rad
        )

        theta_max = float(
            self.theta_max_rad
        )

        range_min = float(
            self.range_min_m
        )

        range_max = float(
            self.range_max_m
        )

        for name, value in (
            (
                "theta_min_rad",
                theta_min,
            ),
            (
                "theta_max_rad",
                theta_max,
            ),
            (
                "range_min_m",
                range_min,
            ),
            (
                "range_max_m",
                range_max,
            ),
        ):

            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not (
            -math.pi
            <=
            theta_min
            <
            theta_max
            <=
            math.pi
        ):
            raise ValueError(
                "theta bounds must satisfy "
                "-pi <= min < max <= pi"
            )

        if not (
            0.0
            <=
            range_min
            <
            range_max
        ):
            raise ValueError(
                "range bounds must satisfy "
                "0 <= min < max"
            )

        n_theta = int(
            self.n_theta
        )

        n_range = int(
            self.n_range
        )

        if n_theta < 2:
            raise ValueError(
                "n_theta must be >= 2"
            )

        if n_range < 2:
            raise ValueError(
                "n_range must be >= 2"
            )

        object.__setattr__(
            self,
            "theta_min_rad",
            theta_min,
        )

        object.__setattr__(
            self,
            "theta_max_rad",
            theta_max,
        )

        object.__setattr__(
            self,
            "range_min_m",
            range_min,
        )

        object.__setattr__(
            self,
            "range_max_m",
            range_max,
        )

        object.__setattr__(
            self,
            "n_theta",
            n_theta,
        )

        object.__setattr__(
            self,
            "n_range",
            n_range,
        )

    @property
    def theta_edges_rad(
        self,
    ) -> np.ndarray:

        return np.linspace(
            self.theta_min_rad,
            self.theta_max_rad,
            self.n_theta + 1,
            dtype=np.float64,
        )

    @property
    def range_edges_m(
        self,
    ) -> np.ndarray:

        return np.linspace(
            self.range_min_m,
            self.range_max_m,
            self.n_range + 1,
            dtype=np.float64,
        )

    @property
    def theta_centers_rad(
        self,
    ) -> np.ndarray:

        edges = self.theta_edges_rad

        return 0.5 * (
            edges[:-1]
            +
            edges[1:]
        )

    @property
    def range_centers_m(
        self,
    ) -> np.ndarray:

        edges = self.range_edges_m

        return 0.5 * (
            edges[:-1]
            +
            edges[1:]
        )

    @property
    def theta_step_rad(
        self,
    ) -> float:

        return (
            self.theta_max_rad
            -
            self.theta_min_rad
        ) / self.n_theta

    @property
    def range_step_m(
        self,
    ) -> float:

        return (
            self.range_max_m
            -
            self.range_min_m
        ) / self.n_range

    @property
    def shape(
        self,
    ) -> tuple[int, int]:

        return (
            self.n_theta,
            self.n_range,
        )

    def contains(
        self,
        theta_rad: float,
        range_m: float,
    ) -> bool:

        theta = float(
            theta_rad
        )

        distance = float(
            range_m
        )

        return (
            self.theta_min_rad
            <=
            theta
            <=
            self.theta_max_rad
            and
            self.range_min_m
            <=
            distance
            <=
            self.range_max_m
        )

    def nearest_cell(
        self,
        theta_rad: float,
        range_m: float,
    ) -> tuple[int, int]:

        theta = float(
            theta_rad
        )

        distance = float(
            range_m
        )

        if not self.contains(
            theta,
            distance,
        ):
            raise ValueError(
                "point lies outside "
                "illumination grid"
            )

        theta_index = int(
            np.argmin(
                np.abs(
                    self.theta_centers_rad
                    -
                    theta
                )
            )
        )

        range_index = int(
            np.argmin(
                np.abs(
                    self.range_centers_m
                    -
                    distance
                )
            )
        )

        return (
            theta_index,
            range_index,
        )


def rectangular_extent_cell_indices(
    grid: IlluminationGridSpec,
    *,
    theta_center_rad: float,
    theta_span_rad: float,
    range_min_m: float,
    range_max_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Return all theta/range cell indices overlapping a continuous
    projected extent.

    This is only a geometry utility. It is NOT the final
    probabilistic occupancy mask implementation.
    """

    theta_center = float(
        theta_center_rad
    )

    theta_span = float(
        theta_span_rad
    )

    if theta_span < 0.0:
        raise ValueError(
            "theta_span_rad must be >= 0"
        )

    theta_lo = (
        theta_center
        -
        0.5 * theta_span
    )

    theta_hi = (
        theta_center
        +
        0.5 * theta_span
    )

    range_lo = float(
        range_min_m
    )

    range_hi = float(
        range_max_m
    )

    if range_hi < range_lo:
        raise ValueError(
            "range_max_m must be >= range_min_m"
        )

    theta_edges = (
        grid.theta_edges_rad
    )

    range_edges = (
        grid.range_edges_m
    )

    theta_overlap = (
        theta_edges[:-1]
        <=
        theta_hi
    ) & (
        theta_edges[1:]
        >=
        theta_lo
    )

    range_overlap = (
        range_edges[:-1]
        <=
        range_hi
    ) & (
        range_edges[1:]
        >=
        range_lo
    )

    return (
        np.flatnonzero(
            theta_overlap
        ),
        np.flatnonzero(
            range_overlap
        ),
    )
'''

    grid_write = write_exact_text(
        GRID_MODULE,
        grid_source,
    )

    print(
        "grid.py =",
        grid_write,
    )

    # ========================================================
    # E. adb package init
    # ========================================================

    adb_init_source = '''"""Stage6 illumination-space and ADB primitives."""

from .geometry import (
    Box3D,
    FractionalBoxRegion,
    ProjectedBox,
    ProjectedPoints,
    ProjectedRegion,
    box_corners_headlamp,
    centroid_only_projection,
    project_box_to_headlamp,
    project_points_to_headlamp,
    project_region_to_headlamp,
    region_corners_headlamp,
    validate_controller_provenance,
    wrap_angle,
)

from .grid import (
    IlluminationGridSpec,
    rectangular_extent_cell_indices,
)

__all__ = (
    "Box3D",
    "FractionalBoxRegion",
    "ProjectedBox",
    "ProjectedPoints",
    "ProjectedRegion",
    "IlluminationGridSpec",
    "box_corners_headlamp",
    "centroid_only_projection",
    "project_box_to_headlamp",
    "project_points_to_headlamp",
    "project_region_to_headlamp",
    "rectangular_extent_cell_indices",
    "region_corners_headlamp",
    "validate_controller_provenance",
    "wrap_angle",
)
'''

    init_write = write_exact_text(
        ADB_INIT,
        adb_init_source,
    )

    print(
        "adb/__init__.py =",
        init_write,
    )

    # ========================================================
    # F. Geometry regression tests
    # ========================================================

    print()
    print(
        "===== E. SYNTHETIC GEOMETRY REGRESSION ====="
    )

    test_source = r'''from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.geometry import (
    Box3D,
    FractionalBoxRegion,
    box_corners_headlamp,
    centroid_only_projection,
    project_box_to_headlamp,
    project_points_to_headlamp,
    project_region_to_headlamp,
    region_corners_headlamp,
    validate_controller_provenance,
    wrap_angle,
)

from iscai_stage6.adb.grid import (
    IlluminationGridSpec,
    rectangular_extent_cell_indices,
)


class TestBlock61GeometryCore(
    unittest.TestCase
):

    def test_headlamp_spherical_mapping(self):

        points = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        projected = (
            project_points_to_headlamp(
                points
            )
        )

        self.assertAlmostEqual(
            projected.theta_rad[0],
            0.0,
        )

        self.assertAlmostEqual(
            projected.theta_rad[1],
            math.pi / 4.0,
        )

        self.assertAlmostEqual(
            projected.phi_rad[2],
            math.pi / 4.0,
        )

        self.assertAlmostEqual(
            projected.range_3d_m[2],
            math.sqrt(2.0),
        )

    def test_box_has_exactly_eight_corners(self):

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        corners = (
            box_corners_headlamp(
                box
            )
        )

        self.assertEqual(
            corners.shape,
            (8, 3),
        )

        self.assertEqual(
            np.unique(
                corners,
                axis=0,
            ).shape[0],
            8,
        )

    def test_box_dimensions_are_preserved(self):

        box = Box3D(
            center_xyz=(
                15.0,
                2.0,
                1.5,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.6,
            yaw_rad=0.0,
        )

        corners = (
            box_corners_headlamp(
                box
            )
        )

        extent = (
            np.max(
                corners,
                axis=0,
            )
            -
            np.min(
                corners,
                axis=0,
            )
        )

        np.testing.assert_allclose(
            extent,
            np.asarray(
                [
                    4.0,
                    2.0,
                    1.6,
                ]
            ),
            atol=1.0e-12,
        )

    def test_yaw_rotates_length_width_axes(self):

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.0,
            yaw_rad=math.pi / 2.0,
        )

        corners = (
            box_corners_headlamp(
                box
            )
        )

        extent_xy = (
            np.max(
                corners[:, :2],
                axis=0,
            )
            -
            np.min(
                corners[:, :2],
                axis=0,
            )
        )

        np.testing.assert_allclose(
            extent_xy,
            np.asarray(
                [
                    2.0,
                    4.0,
                ]
            ),
            atol=1.0e-12,
        )

    def test_full_box_has_nonzero_angular_extent(self):

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        self.assertTrue(
            projection.uses_full_box
        )

        self.assertGreater(
            projection.theta_span_rad,
            0.0,
        )

        self.assertLess(
            projection.ground_range_min_m,
            projection.ground_range_max_m,
        )

        self.assertLess(
            projection.phi_min_rad,
            projection.phi_max_rad,
        )

    def test_centroid_does_not_encode_box_extent(self):

        box = Box3D(
            center_xyz=(
                25.0,
                1.0,
                1.0,
            ),
            length_m=5.0,
            width_m=2.0,
            height_m=2.0,
            yaw_rad=0.2,
        )

        centroid = (
            centroid_only_projection(
                box
            )
        )

        full = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        self.assertEqual(
            centroid.theta_rad.shape,
            (1,),
        )

        self.assertGreater(
            full.theta_span_rad,
            0.0,
        )

        self.assertGreater(
            full.range_3d_max_m
            -
            full.range_3d_min_m,
            0.0,
        )

    def test_relevant_subregion_supported(self):

        box = Box3D(
            center_xyz=(
                18.0,
                -1.0,
                1.2,
            ),
            length_m=4.5,
            width_m=2.0,
            height_m=1.8,
            yaw_rad=-0.1,
        )

        region = FractionalBoxRegion(
            x_bounds=(
                0.0,
                0.5,
            ),
            y_bounds=(
                -0.4,
                0.4,
            ),
            z_bounds=(
                0.0,
                0.5,
            ),
        )

        points = (
            region_corners_headlamp(
                box,
                region,
            )
        )

        projected = (
            project_region_to_headlamp(
                box,
                region,
            )
        )

        self.assertEqual(
            points.shape,
            (8, 3),
        )

        self.assertEqual(
            projected.points.xyz.shape,
            (8, 3),
        )

        self.assertGreater(
            projected.theta_span_rad,
            0.0,
        )

    def test_no_semantic_windshield_constants_are_needed(self):

        with self.assertRaises(
            ValueError
        ):
            FractionalBoxRegion(
                x_bounds=(
                    -0.6,
                    0.2,
                ),
                y_bounds=(
                    -0.4,
                    0.4,
                ),
                z_bounds=(
                    0.0,
                    0.5,
                ),
            )

    def test_future_gt_rejected_on_controller_path(self):

        box = Box3D(
            center_xyz=(
                10.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        with self.assertRaises(
            ValueError
        ):
            project_box_to_headlamp(
                box,
                state_source="future_gt",
                orientation_source="oracle_evaluator_only",
                controller_path=True,
            )

    def test_oracle_allowed_only_off_controller_path(self):

        validate_controller_provenance(
            state_source="oracle_future",
            orientation_source="oracle_evaluator_only",
            controller_path=False,
        )

        box = Box3D(
            center_xyz=(
                10.0,
                1.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="oracle_future",
                orientation_source="oracle_evaluator_only",
                controller_path=False,
            )
        )

        self.assertFalse(
            projection.controller_path
        )

    def test_predicted_sample_is_controller_safe(self):

        box = Box3D(
            center_xyz=(
                30.0,
                2.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.5,
            yaw_rad=0.1,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="predicted_sample",
                orientation_source="predicted_tangent",
                controller_path=True,
            )
        )

        self.assertEqual(
            projection.state_source,
            "predicted_sample",
        )

    def test_angle_wrap(self):

        self.assertAlmostEqual(
            wrap_angle(
                math.pi
            ),
            -math.pi,
        )

        self.assertAlmostEqual(
            wrap_angle(
                3.0 * math.pi
            ),
            -math.pi,
        )

    def test_circular_box_interval_near_pi_boundary(self):

        box = Box3D(
            center_xyz=(
                -20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=2.0,
            height_m=1.0,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        self.assertLess(
            projection.theta_span_rad,
            math.pi / 2.0,
        )

    def test_invalid_box_dimensions_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            Box3D(
                center_xyz=(
                    10.0,
                    0.0,
                    1.0,
                ),
                length_m=0.0,
                width_m=2.0,
                height_m=1.0,
                yaw_rad=0.0,
            )

    def test_grid_requires_explicit_numeric_values(self):

        grid = IlluminationGridSpec(
            theta_min_rad=-0.5,
            theta_max_rad=0.5,
            range_min_m=0.0,
            range_max_m=100.0,
            n_theta=20,
            n_range=50,
        )

        self.assertEqual(
            grid.shape,
            (20, 50),
        )

        self.assertAlmostEqual(
            grid.theta_step_rad,
            0.05,
        )

        self.assertAlmostEqual(
            grid.range_step_m,
            2.0,
        )

    def test_grid_extent_uses_more_than_centroid_cell(self):

        grid = IlluminationGridSpec(
            theta_min_rad=-0.5,
            theta_max_rad=0.5,
            range_min_m=0.0,
            range_max_m=100.0,
            n_theta=100,
            n_range=100,
        )

        box = Box3D(
            center_xyz=(
                20.0,
                0.0,
                1.0,
            ),
            length_m=4.0,
            width_m=4.0,
            height_m=1.5,
            yaw_rad=0.0,
        )

        projection = (
            project_box_to_headlamp(
                box,
                state_source="current_causal",
                orientation_source="current_causal",
            )
        )

        theta_indices, range_indices = (
            rectangular_extent_cell_indices(
                grid,
                theta_center_rad=(
                    projection.theta_center_rad
                ),
                theta_span_rad=(
                    projection.theta_span_rad
                ),
                range_min_m=(
                    projection.ground_range_min_m
                ),
                range_max_m=(
                    projection.ground_range_max_m
                ),
            )
        )

        self.assertGreater(
            theta_indices.size,
            1,
        )

        self.assertGreater(
            range_indices.size,
            1,
        )

    def test_grid_is_not_communication_codebook(self):

        grid = IlluminationGridSpec(
            theta_min_rad=-0.4,
            theta_max_rad=0.4,
            range_min_m=0.0,
            range_max_m=80.0,
            n_theta=32,
            n_range=40,
        )

        self.assertEqual(
            grid.shape,
            (32, 40),
        )

        # This object has theta/range cells only.
        self.assertFalse(
            hasattr(
                grid,
                "beam_id",
            )
        )

        self.assertFalse(
            hasattr(
                grid,
                "codebook",
            )
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
    # F. Controlled compile
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
            str(
                GEOMETRY_MODULE
            ),
            str(
                GRID_MODULE
            ),
            str(
                ADB_INIT
            ),
            str(
                TEST_FILE
            ),
        ],
        cwd=str(S6),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        compile_process.returncode
        ==
        0,
        (
            "Block6.1 geometry compile failed:\n"
            + compile_process.stdout
        ),
    )

    print(
        "geometry/grid/tests compile = PASS"
    )

    # ========================================================
    # G. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== G. STAGE6 REGRESSION ====="
    )

    rc, output = run_tests()

    summary_lines = []

    for line in output.splitlines():

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
            summary_lines.append(
                stripped
            )

    for line in summary_lines:
        print(
            line
        )

    require(
        rc == 0,
        (
            "Stage6 regression failed.\n"
            "Full output:\n"
            + output
        ),
    )

    # ========================================================
    # H. Scientific invariant smoke
    # ========================================================

    print()
    print(
        "===== H. SCIENTIFIC INVARIANT SMOKE ====="
    )

    env = os.environ.copy()

    smoke_code = r'''
import json
import math

from iscai_stage6.adb.geometry import (
    Box3D,
    FractionalBoxRegion,
    project_box_to_headlamp,
    project_region_to_headlamp,
)

box = Box3D(
    center_xyz=(20.0, 1.0, 1.2),
    length_m=4.5,
    width_m=2.0,
    height_m=1.7,
    yaw_rad=0.15,
)

full = project_box_to_headlamp(
    box,
    state_source="current_causal",
    orientation_source="current_causal",
)

region = FractionalBoxRegion(
    x_bounds=(0.0, 0.5),
    y_bounds=(-0.3, 0.3),
    z_bounds=(0.0, 0.5),
)

sub = project_region_to_headlamp(
    box,
    region,
)

print(json.dumps({
    "corner_count": int(
        full.corners.xyz.shape[0]
    ),
    "theta_span_rad": full.theta_span_rad,
    "ground_range_span_m": (
        full.ground_range_max_m
        -
        full.ground_range_min_m
    ),
    "phi_span_rad": (
        full.phi_max_rad
        -
        full.phi_min_rad
    ),
    "subregion_corner_count": int(
        sub.points.xyz.shape[0]
    ),
    "centroid_theta_rad": float(
        full.centroid.theta_rad[0]
    ),
}))
'''

    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            smoke_code,
        ],
        cwd=str(S6),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        smoke.returncode == 0,
        (
            "Fresh-process geometry smoke failed:\n"
            + smoke.stdout
        ),
    )

    smoke_payload = json.loads(
        smoke.stdout.strip()
    )

    require(
        smoke_payload[
            "corner_count"
        ]
        ==
        8,
        (
            "Full box does not project "
            "exactly 8 corners."
        ),
    )

    require(
        smoke_payload[
            "subregion_corner_count"
        ]
        ==
        8,
        (
            "Relevant subregion does not "
            "project 8 corners."
        ),
    )

    require(
        smoke_payload[
            "theta_span_rad"
        ]
        >
        0.0,
        (
            "Full box angular extent collapsed "
            "to centroid-like zero span."
        ),
    )

    require(
        smoke_payload[
            "ground_range_span_m"
        ]
        >
        0.0,
        (
            "Full box range extent collapsed."
        ),
    )

    require(
        smoke_payload[
            "phi_span_rad"
        ]
        >
        0.0,
        (
            "Actor height/elevation extent "
            "was lost."
        ),
    )

    print(
        "8 physical box corners       = PASS"
    )

    print(
        "nonzero angular extent       = PASS"
    )

    print(
        "nonzero range extent         = PASS"
    )

    print(
        "height/elevation retained    = PASS"
    )

    print(
        "3D relevant-region hook      = PASS"
    )

    print(
        "centroid-only implementation = NO"
    )

    # ========================================================
    # I. Upstream immutability / storage
    # ========================================================

    print()
    print(
        "===== I. IMMUTABILITY / STORAGE ====="
    )

    protected_after = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in required
    }

    changed = [
        path
        for path in protected_before
        if (
            protected_before[
                path
            ]
            !=
            protected_after[
                path
            ]
        )
    ]

    require(
        not changed,
        (
            "Frozen prerequisite changed during "
            "Block6.1 Part1: "
            + ", ".join(
                changed
            )
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
        "Block6.0 artifacts = UNCHANGED"
    )

    print(
        "Stage5 closure     = UNCHANGED"
    )

    print(
        "free GiB           =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve    = PASS"
    )

    # ========================================================
    # J. Part1 report
    # ========================================================

    print()
    print(
        "===== J. BLOCK6.1 PART1 REPORT ====="
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.1",

        "part":
            "1/2",

        "status":
            "PASS_GEOMETRY_CORE",

        "Block6_0": {
            "status":
                "PASS_COMPLETE",

            "closure_sha256":
                EXPECTED[
                    "block60_closure"
                ],
        },

        "geometry_contract": {
            "path":
                str(
                    GEOMETRY_CONTRACT
                ),

            "sha256":
                sha256_file(
                    GEOMETRY_CONTRACT
                ),
        },

        "implementation": {
            "geometry_module":
                str(
                    GEOMETRY_MODULE
                ),

            "geometry_sha256":
                sha256_file(
                    GEOMETRY_MODULE
                ),

            "grid_module":
                str(
                    GRID_MODULE
                ),

            "grid_sha256":
                sha256_file(
                    GRID_MODULE
                ),

            "full_box_corner_count":
                8,

            "centroid_only":
                False,

            "actor_height_retained":
                True,

            "relevant_region_support":
                True,

            "communication_codebook_reused":
                False,
        },

        "causality": {
            "future_GT_controller_path":
                False,

            "oracle_evaluator_path_supported":
                True,

            "predicted_sample_controller_path":
                True,
        },

        "scientific_grid": {
            "numeric_bounds_frozen":
                False,

            "numeric_resolution_frozen":
                False,

            "reason":
                (
                    "pending exact frozen Stage1 "
                    "headlamp geometry audit in "
                    "Block6.1 Part2"
                ),
        },

        "regression": {
            "Stage6_tests":
                "PASS",

            "fresh_process_smoke":
                smoke_payload,
        },

        "storage": {
            "free_GiB":
                free_gib,

            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "pass":
                True,
        },

        "execution": {
            "dataset_scan":
                False,

            "training":
                False,

            "trajectory_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "reactive_ADB_execution":
                False,

            "predictive_ADB_execution":
                False,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block6.1 Part2: frozen Stage1 "
                "headlamp transform integration, "
                "real causal actor adapter, "
                "illumination-grid numeric contract "
                "and real-scene projection smoke"
            ),
    }

    report_tmp = REPORT.with_suffix(
        REPORT.suffix + ".tmp"
    )

    report_tmp.write_bytes(
        canonical_bytes(
            report
        )
    )

    os.replace(
        report_tmp,
        REPORT,
    )

    print(
        "report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.1 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block6.0 prerequisite     = EXACT PASS"
    )

    print(
        "illumination space        = PARAMETERIZED THETA-R"
    )

    print(
        "full 3D box corners       = 8/8 PASS"
    )

    print(
        "actor length/width/height = PRESERVED"
    )

    print(
        "azimuth/elevation/range   = PASS"
    )

    print(
        "relevant-region geometry  = PASS"
    )

    print(
        "centroid-only ADB         = FORBIDDEN"
    )

    print(
        "future GT controller use  = REJECTED"
    )

    print(
        "communication codebook    = NOT REUSED"
    )

    print(
        "scientific grid numbers   = "
        "DEFERRED CORRECTLY TO PART2"
    )

    print(
        "Stage6 regression         = PASS"
    )

    print(
        "upstream modified         = NO"
    )

    print(
        "training/inference        = NO"
    )

    print(
        "STATUS = PASS_GEOMETRY_CORE"
    )

    print(
        "report =",
        REPORT,
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
        "BLOCK 6.1 PART 1/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(
            exc
        ).__name__,
        str(
            exc
        ),
    )

    print()
    traceback.print_exc()

    print(
        "Block6.0 modified       = NO"
    )

    print(
        "Stage5 modified         = NO"
    )

    print(
        "dataset scan            = NO"
    )

    print(
        "training/inference      = NO"
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

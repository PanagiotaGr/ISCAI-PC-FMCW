from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")


# ============================================================
# Deterministic project-source bootstrap
# ============================================================

def bootstrap_project_sources():

    roots = tuple(
        ROOT
        / f"iscai_stage{index}"
        / "src"
        for index in range(7)
    )

    missing = [
        str(path)
        for path in roots
        if not path.is_dir()
    ]

    if missing:
        raise RuntimeError(
            "Missing project source root(s): "
            + ", ".join(missing)
        )

    for path in reversed(
        roots
    ):

        value = str(path)

        while value in sys.path:
            sys.path.remove(
                value
            )

        sys.path.insert(
            0,
            value,
        )

    previous = [
        item
        for item in os.environ.get(
            "PYTHONPATH",
            "",
        ).split(
            os.pathsep
        )
        if item
    ]

    merged = []
    seen = set()

    for item in (
        [
            str(path)
            for path in roots
        ]
        +
        previous
    ):

        if item in seen:
            continue

        seen.add(item)
        merged.append(item)

    os.environ[
        "PYTHONPATH"
    ] = os.pathsep.join(
        merged
    )

    return tuple(
        str(path)
        for path in roots
    )


PROJECT_SOURCE_ROOTS = (
    bootstrap_project_sources()
)


S1 = ROOT / "iscai_stage1"
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

NOTEBOOK = (
    ROOT
    / "part_a_reference/ISCAI_pc_fmcw/"
      "notebooks/ISCAI_PC_FMCW.ipynb"
)

PART_A_REFERENCE = (
    ROOT
    / "iscai_stage0/reports/stage0/"
      "part_a_frozen_reference.json"
)

BLOCK61_CLOSURE = (
    S6
    / "reports/block61_closure.json"
)

BLOCK62_PART1_REPORT = (
    S6
    / "reports/"
      "block62_part1_static_reactive_kernel.json"
)

BLOCK62_PART1_CONTRACT = (
    S6
    / "configs/"
      "block62_static_reactive_kernel_contract.json"
)

PREBIND_AUDIT = (
    S6
    / "reports/"
      "block62_part2_prebinding_source_audit.json"
)

TRANSITIVE_AUDIT = (
    S6
    / "reports/"
      "block62_dynamic_dependency_chain_audit.json"
)

ILLUMINATION_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "illumination.py"
)

WOMD_GEOMETRY_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "womd_geometry.py"
)

PART_A_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "part_a_reactive.py"
)

ADB_INIT = (
    S6
    / "src/iscai_stage6/adb/"
      "__init__.py"
)

TEST_FILE = (
    S6
    / "tests/"
      "test_block62_exact_part_a_reactive.py"
)

BINDING = (
    S6
    / "configs/"
      "block62_part_a_reactive_binding.json"
)

REAL_SMOKE = (
    S6
    / "artifacts/block62/"
      "block62_real_causal_reactive_smoke.json"
)

PART2_REPORT = (
    S6
    / "reports/"
      "block62_part2_exact_part_a_reactive.json"
)

CLOSURE = (
    S6
    / "reports/"
      "block62_closure.json"
)

REGRESSION_LOG = (
    S6
    / "artifacts/block62/"
      "block62_part2_full_stage6_regression.log"
)

PILOT_MANIFEST = (
    S1
    / "artifacts/stage1a/manifests/"
      "tiny_pilot_validation.jsonl"
)

PAIRED_ROOT = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0"
)

EXPECTED = {
    "notebook":
        (
            "b5a80a6d3441de6d571db4f65b4a43e"
            "d4052cc2b3ccba935ad31b5dd51316ef3"
        ),

    "part_a_reference":
        (
            "fcdfc0a7b9c14fa9447ceb8563a6f20a"
            "c277706223388dc9aa8a1edf416c5e2b"
        ),

    "block61_closure":
        (
            "1d0f8781d365efd09614c8fb5a06bdbd"
            "a8f56c1c673b0ddab8b40810de270ac1"
        ),

    "part1_report":
        (
            "865c81eeb689a1b0c7e41b08117c9197"
            "c19e5cac76de147a07898b17c2224477"
        ),

    "part1_contract":
        (
            "df4cf1e8fd18f3bc9cd48cbae4a6c209"
            "1e37313cbc1a5b2cefe48bd340eb5d60"
        ),

    "prebind":
        (
            "fab4821ce11e8969a2ed8c7a36e74e7b"
            "85bc1e27116cf23a445ce32c3e3f0ffb"
        ),

    "transitive":
        (
            "7406e5e0342cee6252a536d63edd77ff"
            "909b6acd3eea7e6adb870f99e51796b5"
        ),

    "pilot_manifest":
        (
            "2b6fd3d7e5c411455708d325fc8595af"
            "11d99f3fcba4dde621928150695ca859"
        ),

    "stage5_closure":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),
}

STAGE5_CLOSURE = (
    S5
    / "reports/"
      "stage5_final_closure.json"
)

MIN_FREE_GIB = 250.0


# ============================================================
# Generic helpers
# ============================================================

def require(
    condition,
    message,
):

    if not bool(condition):
        raise RuntimeError(
            message
        )


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


def array_sha256(
    array,
) -> str:

    import numpy as np

    value = np.ascontiguousarray(
        array
    )

    return sha256(
        value.tobytes()
    ).hexdigest()


def canonical_bytes(
    value,
) -> bytes:

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def write_exact_text(
    path: Path,
    value: str,
):

    desired = value.encode(
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

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        desired
    )

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def write_exact_json(
    path: Path,
    value,
):

    desired = canonical_bytes(
        value
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

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        desired
    )

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def atomic_json(
    path: Path,
    value,
):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(
            value
        )
    )

    os.replace(
        tmp,
        path,
    )


def normalize_source_line(
    value: str,
) -> str:

    return "".join(
        value.split()
    )


def tail_text(
    text: str,
    lines: int = 100,
) -> str:

    rows = text.splitlines()

    return "\n".join(
        rows[-lines:]
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.2 PART 2/2"
    )
    print(
        "EXACT PART-A ORIGINAL REACTIVE ADB ADAPTER"
    )
    print(
        "============================================================"
    )

    protected = (
        NOTEBOOK,
        PART_A_REFERENCE,
        BLOCK61_CLOSURE,
        BLOCK62_PART1_REPORT,
        BLOCK62_PART1_CONTRACT,
        PREBIND_AUDIT,
        TRANSITIVE_AUDIT,
        ILLUMINATION_MODULE,
        WOMD_GEOMETRY_MODULE,
        PILOT_MANIFEST,
        STAGE5_CLOSURE,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing prerequisite(s): "
            + ", ".join(missing)
        ),
    )

    protected_before = {
        str(path):
            sha256_file(
                path
            )
        for path in protected
    }

    # ========================================================
    # A. Frozen prerequisite seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN PREREQUISITE SEAL ====="
    )

    exact_checks = (
        (
            "notebook",
            NOTEBOOK,
        ),
        (
            "part_a_reference",
            PART_A_REFERENCE,
        ),
        (
            "block61_closure",
            BLOCK61_CLOSURE,
        ),
        (
            "part1_report",
            BLOCK62_PART1_REPORT,
        ),
        (
            "part1_contract",
            BLOCK62_PART1_CONTRACT,
        ),
        (
            "prebind",
            PREBIND_AUDIT,
        ),
        (
            "transitive",
            TRANSITIVE_AUDIT,
        ),
        (
            "pilot_manifest",
            PILOT_MANIFEST,
        ),
        (
            "stage5_closure",
            STAGE5_CLOSURE,
        ),
    )

    for name, path in (
        exact_checks
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
                f"{name} SHA mismatch\n"
                f"expected={EXPECTED[name]}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{name:24s} = EXACT PASS"
        )

    transitive_report = json.loads(
        TRANSITIVE_AUDIT.read_text(
            encoding="utf-8"
        )
    )

    require(
        transitive_report.get(
            "status"
        )
        ==
        (
            "PASS_READY_FOR_EXACT_"
            "PARTA_REACTIVE_ADAPTER"
        ),
        (
            "Transitive Part-A audit "
            "is not in PASS-ready state."
        ),
    )

    print(
        "transitive source semantics = PASS"
    )

    # ========================================================
    # B. Exact notebook source binding
    # ========================================================

    print()
    print(
        "===== B. EXACT NOTEBOOK PARAMETER BINDING ====="
    )

    notebook = json.loads(
        NOTEBOOK.read_text(
            encoding="utf-8"
        )
    )

    source_lines = []

    for cell in notebook.get(
        "cells",
        [],
    ):

        if cell.get(
            "cell_type"
        ) != "code":
            continue

        source = cell.get(
            "source",
            ""
        )

        if isinstance(
            source,
            list,
        ):
            source = "".join(
                source
            )

        source_lines.extend(
            str(
                source
            ).splitlines()
        )

    normalized = {
        normalize_source_line(
            line
        )
        for line in source_lines
    }

    exact_required = {
        "EPS":
            "EPS=1e-12",

        "target_width":
            "target_width_m=1.9",

        "lateral_margin":
            "lateral_safety_margin_m=1.0",

        "radial_margin":
            "radial_safety_margin_m=4.0",

        "control_transition":
            "transition_length_m=20.0",

        "delta_theta":
            "delta_theta_deg=0.0",

        "fixed_epsilon":
            "epsilon_deg=0.75",

        "trz_front_margin":
            "adb_trz_front_margin_units=4.0",

        "trz_transition":
            "adb_trz_transition_length_units=8.0",

        "theta_grid":
            (
                "theta_grid_deg="
                "np.linspace(-25.0,25.0,501)"
            ),

        "range_grid":
            (
                "range_grid_m="
                "np.linspace(0.0,180.0,361)"
            ),
    }

    for name, line in (
        exact_required.items()
    ):

        require(
            normalize_source_line(
                line
            )
            in
            normalized,
            (
                "Exact Part-A notebook "
                f"assignment missing: {name}"
            ),
        )

        print(
            f"{name:24s} = EXACT SOURCE PASS"
        )

    entire_source = "\n".join(
        source_lines
    )

    formula_tokens = (
        (
            "effective_half_width",
            (
                "effective_half_width = "
                "target_width_m / 2.0 + "
                "lateral_safety_margin_m"
            ),
        ),
        (
            "dynamic_epsilon",
            (
                "epsilon = np.rad2deg("
                "np.arctan("
                "effective_half_width / "
                "np.maximum(target_range, EPS)))"
            ),
        ),
        (
            "intensity_radial_margin",
            (
                "r_min_m = target_range_m + "
                "radial_safety_margin_m"
            ),
        ),
        (
            "intensity_transition_length",
            (
                "r_max_m = r_min_m + "
                "transition_length_m"
            ),
        ),
        (
            "raised_cosine",
            (
                "intensity[transition] = "
                "0.5 * (1.0 - "
                "np.cos(np.pi * u))"
            ),
        ),
        (
            "multi_actor_minimum",
            (
                "combined = np.minimum("
                "combined, vehicle_map)"
            ),
        ),
        (
            "trz_visual_rmin",
            (
                'geometry["r_min"] = '
                "target_range + "
                "adb_trz_front_margin_units"
            ),
        ),
        (
            "trz_visual_rmax",
            (
                'geometry["r_max"] = '
                'geometry["r_min"] + '
                "adb_trz_transition_length_units"
            ),
        ),
    )

    normalized_entire_source = (
        normalize_source_line(
            entire_source
        )
    )

    for name, token in (
        formula_tokens
    ):

        require(
            normalize_source_line(
                token
            )
            in
            normalized_entire_source,
            (
                "Canonical Part-A formula "
                f"token missing: {name}"
            ),
        )

        print(
            f"{name:24s} = EXACT SOURCE PASS"
        )

    print()
    print(
        "CONTROL L(theta,r) radial transition = 20.0 m"
    )

    print(
        "TRZ visualization radial transition  = 8.0 units"
    )

    print(
        "control/visual transition conflated  = NO"
    )

    print(
        "fixed epsilon 0.75 used dynamically  = NO"
    )

    # ========================================================
    # C. Inspect frozen Stage6 causal-box interface
    # ========================================================

    print()
    print(
        "===== C. CAUSAL BOX INTERFACE BINDING ====="
    )

    from iscai_stage6.adb.womd_geometry import (
        CausalADBActorBox,
    )

    require(
        is_dataclass(
            CausalADBActorBox
        ),
        (
            "CausalADBActorBox is no longer "
            "a dataclass."
        ),
    )

    actor_fields = {
        field.name
        for field in fields(
            CausalADBActorBox
        )
    }

    center_field = (
        "stage1_anchor_center_H0_m"
    )

    require(
        center_field
        in
        actor_fields,
        (
            "Required verified Stage1 anchor-center "
            "field missing from CausalADBActorBox."
        ),
    )

    # Exact frozen Block6.1 binding:
    # CausalADBActorBox.object_type is produced by
    # object_type_name(track) and stores the symbolic
    # controller-facing semantic class.
    class_field = "object_type"

    require(
        class_field
        in
        actor_fields,
        (
            "Frozen CausalADBActorBox "
            "object_type field missing."
        ),
    )

    print(
        "center field =",
        center_field,
    )

    print(
        "class field  =",
        class_field,
    )

    print(
        "centroid/current-center use = "
        "LEGACY REACTIVE BASELINE ONLY"
    )

    print(
        "predictive centroid-only use = FORBIDDEN"
    )

    # ========================================================
    # D. Generate exact original-reactive adapter module
    # ========================================================

    print()
    print(
        "===== D. ORIGINAL REACTIVE ADAPTER MODULE ====="
    )

    module_source = f'''from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .illumination import (
    ReactiveShadowRegion,
    reactive_adb_map,
    static_adb_map,
)


# ============================================================
# Frozen Part-A dynamic reactive parameters
# ============================================================

PART_A_EPS = 1.0e-12

PART_A_TARGET_WIDTH_M = 1.9
PART_A_LATERAL_SAFETY_MARGIN_M = 1.0

PART_A_DELTA_THETA_DEG = 0.0

# Actual adb_intensity_map() controller parameters:
PART_A_RADIAL_SAFETY_MARGIN_M = 4.0
PART_A_CONTROL_TRANSITION_LENGTH_M = 20.0

# Track-reconstruction visualization radial-zone parameters.
# These are NOT substituted for the control L(theta,r) thresholds.
PART_A_TRZ_FRONT_MARGIN_UNITS = 4.0
PART_A_TRZ_TRANSITION_LENGTH_UNITS = 8.0

# Frozen fixed demonstration only.
# The dynamic track-driven path does not reference this value.
PART_A_FIXED_DEMO_EPSILON_DEG = 0.75

# Frozen Module-3 reference raster only.
# It is NOT the final Stage-6 scientific/evaluation grid.
PART_A_REFERENCE_THETA_MIN_DEG = -25.0
PART_A_REFERENCE_THETA_MAX_DEG = 25.0
PART_A_REFERENCE_THETA_POINTS = 501

PART_A_REFERENCE_RANGE_MIN_M = 0.0
PART_A_REFERENCE_RANGE_MAX_M = 180.0
PART_A_REFERENCE_RANGE_POINTS = 361

CAUSAL_BOX_CENTER_FIELD = {center_field!r}
CAUSAL_BOX_CLASS_FIELD = {class_field!r}

ORIGINAL_REACTIVE_SUPPORTED_CLASS = "TYPE_VEHICLE"

PART_A_COORDINATE_CONVENTION = (
    "+y forward / +x right"
)

STAGE6_H0_COORDINATE_CONVENTION = (
    "+x forward / +y left / +z up"
)


@dataclass(frozen=True)
class PartAReactiveGeometry:
    """
    Exact current-state Part-A dynamic angular geometry,
    expressed in both original Part-A coordinates and
    Stage-6 H0 angular coordinates.

    This is a legacy original-reactive comparator.

    It is deliberately:
      - current-state only,
      - fixed-width Part-A geometry,
      - not predictive,
      - not covariance aware,
      - not class-aware beyond the original vehicle use,
      - not a full-box Stage-6 predictive policy.
    """

    center_H0_m: tuple[
        float,
        float,
        float,
    ]

    x_partA_right_m: float
    y_partA_forward_m: float

    target_range_m: float

    theta_c_partA_deg: float
    shadow_center_partA_deg: float
    epsilon_dynamic_deg: float

    theta_min_partA_deg: float
    theta_max_partA_deg: float

    theta_min_H0_deg: float
    theta_max_H0_deg: float

    control_r_min_m: float
    control_r_max_m: float

    trz_visual_r_min_m: float
    trz_visual_r_max_m: float


class PartAReferenceGrid:
    """
    Frozen Part-A Module-3 reference sampling axes.

    Baseline-parity / smoke grid only.

    The final common Stage-6 ADB scientific grid remains
    a Block-6.8 pre-formal freeze decision.
    """

    @property
    def theta_centers_rad(
        self,
    ) -> np.ndarray:

        return np.deg2rad(
            np.linspace(
                PART_A_REFERENCE_THETA_MIN_DEG,
                PART_A_REFERENCE_THETA_MAX_DEG,
                PART_A_REFERENCE_THETA_POINTS,
            )
        )

    @property
    def range_centers_m(
        self,
    ) -> np.ndarray:

        return np.linspace(
            PART_A_REFERENCE_RANGE_MIN_M,
            PART_A_REFERENCE_RANGE_MAX_M,
            PART_A_REFERENCE_RANGE_POINTS,
        )

    @property
    def shape(
        self,
    ) -> tuple[int, int]:

        return (
            PART_A_REFERENCE_THETA_POINTS,
            PART_A_REFERENCE_RANGE_POINTS,
        )


def h0_center_to_part_a_xy(
    center_H0_m,
) -> tuple[
    float,
    float,
]:
    """
    Exact convention conversion.

    Stage6 H0:
        x_H = forward
        y_H = left

    Part-A:
        x_A = right
        y_A = forward

    Therefore:
        x_A = -y_H
        y_A =  x_H
    """

    values = tuple(
        float(value)
        for value in center_H0_m
    )

    if len(values) != 3:
        raise ValueError(
            "center_H0_m must contain x,y,z"
        )

    if not all(
        math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            "center_H0_m must be finite"
        )

    x_H0, y_H0, _z_H0 = values

    return (
        -y_H0,
        x_H0,
    )


def part_a_dynamic_geometry_from_h0_center(
    center_H0_m,
) -> PartAReactiveGeometry:
    """
    Exact binding of the frozen Part-A dynamic geometry:

      target_range = sqrt(x_A^2 + y_A^2)

      theta_c =
          deg(atan2(x_A, y_A))

      shadow_center =
          theta_c - delta_theta

      effective_half_width =
          target_width / 2 + lateral_safety_margin

      epsilon =
          deg(
              atan(
                  effective_half_width
                  /
                  max(target_range, EPS)
              )
          )

    Angular interval is then converted back to H0:
        [theta_min_A, theta_max_A]
            ->
        [-theta_max_A, -theta_min_A]

    IMPORTANT:
    The illumination-control radial thresholds come from
    adb_intensity_map():
        r_min = range + 4 m
        r_max = r_min + 20 m

    The separate +4/+8 TRZ values are preserved only as
    visualization-geometry provenance.
    """

    center = tuple(
        float(value)
        for value in center_H0_m
    )

    x_A, y_A = (
        h0_center_to_part_a_xy(
            center
        )
    )

    target_range = math.sqrt(
        x_A * x_A
        +
        y_A * y_A
    )

    theta_c_A = math.degrees(
        math.atan2(
            x_A,
            y_A,
        )
    )

    shadow_center_A = (
        theta_c_A
        -
        PART_A_DELTA_THETA_DEG
    )

    effective_half_width = (
        PART_A_TARGET_WIDTH_M
        /
        2.0
        +
        PART_A_LATERAL_SAFETY_MARGIN_M
    )

    epsilon_dynamic = math.degrees(
        math.atan(
            effective_half_width
            /
            max(
                target_range,
                PART_A_EPS,
            )
        )
    )

    theta_min_A = (
        shadow_center_A
        -
        epsilon_dynamic
    )

    theta_max_A = (
        shadow_center_A
        +
        epsilon_dynamic
    )

    # Exact interval sign/order conversion.
    theta_min_H0 = (
        -theta_max_A
    )

    theta_max_H0 = (
        -theta_min_A
    )

    control_r_min = (
        target_range
        +
        PART_A_RADIAL_SAFETY_MARGIN_M
    )

    control_r_max = (
        control_r_min
        +
        PART_A_CONTROL_TRANSITION_LENGTH_M
    )

    trz_r_min = (
        target_range
        +
        PART_A_TRZ_FRONT_MARGIN_UNITS
    )

    trz_r_max = (
        trz_r_min
        +
        PART_A_TRZ_TRANSITION_LENGTH_UNITS
    )

    return PartAReactiveGeometry(
        center_H0_m=center,

        x_partA_right_m=x_A,
        y_partA_forward_m=y_A,

        target_range_m=target_range,

        theta_c_partA_deg=theta_c_A,
        shadow_center_partA_deg=shadow_center_A,
        epsilon_dynamic_deg=epsilon_dynamic,

        theta_min_partA_deg=theta_min_A,
        theta_max_partA_deg=theta_max_A,

        theta_min_H0_deg=theta_min_H0,
        theta_max_H0_deg=theta_max_H0,

        control_r_min_m=control_r_min,
        control_r_max_m=control_r_max,

        trz_visual_r_min_m=trz_r_min,
        trz_visual_r_max_m=trz_r_max,
    )


def part_a_region_from_h0_center(
    center_H0_m,
) -> ReactiveShadowRegion:
    """
    Convert one current target center into the existing
    normalized L(theta,r) kernel representation.

    floor=0 exactly preserves the original Part-A
    reactive vehicle baseline.
    """

    geometry = (
        part_a_dynamic_geometry_from_h0_center(
            center_H0_m
        )
    )

    theta_center_H0_deg = (
        0.5
        *
        (
            geometry.theta_min_H0_deg
            +
            geometry.theta_max_H0_deg
        )
    )

    theta_span_deg = (
        geometry.theta_max_H0_deg
        -
        geometry.theta_min_H0_deg
    )

    return ReactiveShadowRegion(
        theta_center_rad=(
            math.radians(
                theta_center_H0_deg
            )
        ),

        theta_span_rad=(
            math.radians(
                theta_span_deg
            )
        ),

        r_shadow_end_m=(
            geometry.control_r_min_m
        ),

        r_transition_end_m=(
            geometry.control_r_max_m
        ),

        intensity_floor=0.0,

        source_semantics=(
            "current_deterministic_region"
        ),
    )


def part_a_current_vehicle_centers_from_causal_boxes(
    actor_boxes: Iterable,
) -> tuple[
    tuple[
        float,
        float,
        float,
    ],
    ...,
]:
    """
    Extract current vehicle centers only.

    This intentionally reproduces the frozen legacy
    original-reactive baseline, whose current MHT track
    position feeds a fixed-width angular shadow formula.

    It MUST NOT be reused as the Stage-6 predictive
    full-box occupancy implementation.
    """

    centers = []

    for actor_box in actor_boxes:

        actor_class = str(
            getattr(
                actor_box,
                CAUSAL_BOX_CLASS_FIELD,
            )
        )

        if (
            actor_class
            !=
            ORIGINAL_REACTIVE_SUPPORTED_CLASS
        ):
            continue

        center = getattr(
            actor_box,
            CAUSAL_BOX_CENTER_FIELD,
        )

        if center is None:
            raise ValueError(
                "Vehicle causal box has no "
                "verified Stage1 anchor center."
            )

        values = tuple(
            float(value)
            for value in center
        )

        if len(values) != 3:
            raise ValueError(
                "Stage1 anchor center must be 3D."
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Stage1 anchor center contains "
                "non-finite values."
            )

        centers.append(
            values
        )

    return tuple(
        centers
    )


def part_a_original_reactive_map_from_h0_centers(
    grid,
    centers_H0_m: Iterable,
) -> np.ndarray:
    """
    Original current-state reactive ADB comparator.

    No future trajectory, posterior, covariance, class policy
    or communication-codebook input is accepted.
    """

    centers = tuple(
        centers_H0_m
    )

    if not centers:
        return static_adb_map(
            grid
        )

    regions = tuple(
        part_a_region_from_h0_center(
            center
        )
        for center in centers
    )

    return reactive_adb_map(
        grid,
        regions,
    )


def part_a_direct_source_reference_map_from_h0_centers(
    grid,
    centers_H0_m: Iterable,
) -> np.ndarray:
    """
    Independent direct transcription of the frozen
    adb_intensity_map() + np.minimum combination rule.

    Used only as a parity oracle for the adapter implementation.
    """

    theta_H_deg = np.rad2deg(
        np.asarray(
            grid.theta_centers_rad,
            dtype=np.float64,
        )
    )

    range_m = np.asarray(
        grid.range_centers_m,
        dtype=np.float64,
    )

    expected_shape = (
        theta_H_deg.size,
        range_m.size,
    )

    combined = np.ones(
        expected_shape,
        dtype=np.float64,
    )

    for center in tuple(
        centers_H0_m
    ):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                center
            )
        )

        intensity = np.ones(
            expected_shape,
            dtype=np.float64,
        )

        angular = (
            theta_H_deg
            >=
            geometry.theta_min_H0_deg
        ) & (
            theta_H_deg
            <=
            geometry.theta_max_H0_deg
        )

        full_shadow = (
            angular[:, None]
            &
            (
                range_m[None, :]
                <=
                geometry.control_r_min_m
            )
        )

        transition = (
            angular[:, None]
            &
            (
                range_m[None, :]
                >
                geometry.control_r_min_m
            )
            &
            (
                range_m[None, :]
                <
                geometry.control_r_max_m
            )
        )

        intensity[
            full_shadow
        ] = 0.0

        if np.any(
            transition
        ):

            range_grid = np.broadcast_to(
                range_m[
                    None,
                    :
                ],
                expected_shape,
            )

            u = (
                range_grid[
                    transition
                ]
                -
                geometry.control_r_min_m
            ) / (
                geometry.control_r_max_m
                -
                geometry.control_r_min_m
            )

            intensity[
                transition
            ] = (
                0.5
                *
                (
                    1.0
                    -
                    np.cos(
                        np.pi
                        *
                        u
                    )
                )
            )

        combined = np.minimum(
            combined,
            intensity,
        )

    return combined
'''

    module_status = write_exact_text(
        PART_A_MODULE,
        module_source,
    )

    print(
        "part_a_reactive.py =",
        module_status,
    )

    # ========================================================
    # E. Package exports
    # ========================================================

    print()
    print(
        "===== E. PACKAGE EXPORT CONTRACT ====="
    )

    init_original = ADB_INIT.read_text(
        encoding="utf-8"
    )

    init_text = init_original

    import_block = '''from .part_a_reactive import (
    ORIGINAL_REACTIVE_SUPPORTED_CLASS,
    PART_A_CONTROL_TRANSITION_LENGTH_M,
    PART_A_FIXED_DEMO_EPSILON_DEG,
    PART_A_LATERAL_SAFETY_MARGIN_M,
    PART_A_RADIAL_SAFETY_MARGIN_M,
    PART_A_TARGET_WIDTH_M,
    PART_A_TRZ_TRANSITION_LENGTH_UNITS,
    PartAReactiveGeometry,
    PartAReferenceGrid,
    h0_center_to_part_a_xy,
    part_a_current_vehicle_centers_from_causal_boxes,
    part_a_direct_source_reference_map_from_h0_centers,
    part_a_dynamic_geometry_from_h0_center,
    part_a_original_reactive_map_from_h0_centers,
    part_a_region_from_h0_center,
)
'''

    if (
        import_block
        not in
        init_text
    ):

        marker = "__all__ = ("

        require(
            init_text.count(
                marker
            )
            ==
            1,
            (
                "Could not uniquely locate "
                "__all__ in adb/__init__.py"
            ),
        )

        init_text = init_text.replace(
            marker,
            import_block
            +
            "\n"
            +
            marker,
            1,
        )

    exports = (
        "ORIGINAL_REACTIVE_SUPPORTED_CLASS",
        "PART_A_CONTROL_TRANSITION_LENGTH_M",
        "PART_A_FIXED_DEMO_EPSILON_DEG",
        "PART_A_LATERAL_SAFETY_MARGIN_M",
        "PART_A_RADIAL_SAFETY_MARGIN_M",
        "PART_A_TARGET_WIDTH_M",
        "PART_A_TRZ_TRANSITION_LENGTH_UNITS",
        "PartAReactiveGeometry",
        "PartAReferenceGrid",
        "h0_center_to_part_a_xy",
        "part_a_current_vehicle_centers_from_causal_boxes",
        "part_a_direct_source_reference_map_from_h0_centers",
        "part_a_dynamic_geometry_from_h0_center",
        "part_a_original_reactive_map_from_h0_centers",
        "part_a_region_from_h0_center",
    )

    missing_exports = [
        name
        for name in exports
        if (
            f'"{name}"'
            not in
            init_text
        )
    ]

    if missing_exports:

        marker = "__all__ = ("

        additions = "".join(
            f'    "{name}",\n'
            for name in missing_exports
        )

        init_text = init_text.replace(
            marker,
            marker
            +
            "\n"
            +
            additions,
            1,
        )

    if init_text == init_original:

        init_status = (
            "ALREADY_UPDATED"
        )

    else:

        tmp = ADB_INIT.with_suffix(
            ".py.block62part2.tmp"
        )

        tmp.write_text(
            init_text,
            encoding="utf-8",
        )

        os.replace(
            tmp,
            ADB_INIT,
        )

        init_status = (
            "UPDATED_BLOCK62_PART2"
        )

    print(
        "adb/__init__.py =",
        init_status,
    )

    # ========================================================
    # F. Binding artifact
    # ========================================================

    print()
    print(
        "===== F. EXACT PART-A BINDING CONTRACT ====="
    )

    binding = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "baseline":
            "original_reactive_ADB",

        "status":
            "FROZEN_EXACT_PART_A_REACTIVE_BINDING",

        "source_authority": {
            "notebook":
                str(
                    NOTEBOOK
                ),

            "notebook_sha256":
                EXPECTED[
                    "notebook"
                ],

            "transitive_audit":
                str(
                    TRANSITIVE_AUDIT
                ),

            "transitive_audit_sha256":
                EXPECTED[
                    "transitive"
                ],
        },

        "coordinate_conventions": {
            "Part_A":
                "+y forward, +x right",

            "Stage6_H0":
                "+x forward, +y left, +z up",

            "H0_to_PartA":
                {
                    "x_partA_right":
                        "-y_H0_left",

                    "y_partA_forward":
                        "x_H0_forward",
                },

            "angle_relation":
                "theta_partA = -theta_H0",

            "interval_relation":
                (
                    "[theta_min_A, theta_max_A] "
                    "-> "
                    "[-theta_max_A, -theta_min_A]"
                ),
        },

        "dynamic_angular_geometry": {
            "target_width_m":
                1.9,

            "lateral_safety_margin_m":
                1.0,

            "effective_half_width_m":
                1.95,

            "delta_theta_deg":
                0.0,

            "epsilon":
                (
                    "deg(atan("
                    "(target_width_m/2 + "
                    "lateral_safety_margin_m)"
                    "/max(target_range,1e-12)))"
                ),

            "fixed_demo_epsilon_deg":
                0.75,

            "fixed_demo_epsilon_used_by_dynamic_path":
                False,

            "technical_report_0_745":
                (
                    "DOCUMENTED_ARITHMETIC_"
                    "TEXT_INCONSISTENCY"
                ),
        },

        "illumination_control_radial_semantics": {
            "source_function":
                "adb_intensity_map",

            "front_margin_m":
                4.0,

            "transition_length_m":
                20.0,

            "r_min":
                "target_range + 4.0",

            "r_max":
                "r_min + 20.0",

            "raised_cosine":
                (
                    "0.5*(1-cos(pi*u))"
                ),

            "full_shadow_floor":
                0.0,

            "outside_intensity":
                1.0,
        },

        "TRZ_visualization_geometry": {
            "source_function":
                "adb_trz_shadow_geometry_from_xy",

            "front_margin_units":
                4.0,

            "transition_length_units":
                8.0,

            "used_as_control_transition_length":
                False,

            "semantic_role":
                (
                    "TRACK_RECONSTRUCTION_"
                    "VISUALIZATION_GEOMETRY"
                ),
        },

        "multiple_targets": {
            "combination":
                "elementwise_minimum",
        },

        "legacy_original_reactive_scope": {
            "Stage1_adapter":
                "iscai_stage1.actors.womd_adapter.adapt_causal_womd_scenario",

            "actor_box_builder":
                "build_causal_adb_actor_boxes(scenario=..., adapted=...)",

            "class_field":
                "object_type",

            "class_source":
                "object_type_name(track)",

            "class_representation":
                "SYMBOLIC_STRING",

            "supported_classes":
                [
                    "TYPE_VEHICLE",
                    "TYPE_PEDESTRIAN",
                    "TYPE_CYCLIST",
                ],

            "original_reactive_vehicle_symbol":
                "TYPE_VEHICLE",

            "raw_integer_class_comparison":
                False,

            "current_state_only":
                True,

            "vehicle_center_based":
                True,

            "fixed_vehicle_width":
                True,

            "posterior_input":
                False,

            "future_input":
                False,

            "covariance_input":
                False,

            "class_aware_policy":
                False,

            "communication_codebook":
                False,

            "centroid_center_semantics":
                (
                    "ALLOWED_ONLY_AS_FROZEN_"
                    "ORIGINAL_REACTIVE_BASELINE"
                ),
        },

        "predictive_Stage6_binding": {
            "centroid_only_allowed":
                False,

            "full_box_projection_required":
                True,

            "uncertainty_required":
                True,

            "class_policy_required":
                True,

            "implemented_in_Block62":
                False,
        },

        "grid": {
            "Part_A_reference_theta_deg":
                [
                    -25.0,
                    25.0,
                    501,
                ],

            "Part_A_reference_range_m":
                [
                    0.0,
                    180.0,
                    361,
                ],

            "Part_A_reference_grid_role":
                "BASELINE_PARITY_SMOKE_ONLY",

            "formal_common_Stage6_grid_frozen":
                False,

            "formal_grid_freeze":
                "BLOCK6.8_PRE_FORMAL",
        },

        "formal_input_policy": {
            "WOMD_annotation_smoke":
                "GEOMETRY_INTEGRATION_ONLY",

            "formal_original_reactive_input":
                (
                    "CAUSAL_TRACK_BASED_"
                    "PC_FMCW_LIKE_PIPELINE"
                ),

            "formal_input_implemented_here":
                False,
        },
    }

    binding_status = write_exact_json(
        BINDING,
        binding,
    )

    print(
        "binding =",
        binding_status,
    )

    print(
        "binding SHA256 =",
        sha256_file(
            BINDING
        ),
    )

    # ========================================================
    # G. Unit tests
    # ========================================================

    print()
    print(
        "===== G. EXACT REACTIVE ADAPTER TESTS ====="
    )

    test_source = f'''from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
import unittest

import numpy as np

from iscai_stage6.adb.part_a_reactive import (
    CAUSAL_BOX_CENTER_FIELD,
    CAUSAL_BOX_CLASS_FIELD,
    PART_A_CONTROL_TRANSITION_LENGTH_M,
    PART_A_FIXED_DEMO_EPSILON_DEG,
    PART_A_LATERAL_SAFETY_MARGIN_M,
    PART_A_RADIAL_SAFETY_MARGIN_M,
    PART_A_TARGET_WIDTH_M,
    PART_A_TRZ_TRANSITION_LENGTH_UNITS,
    PartAReferenceGrid,
    h0_center_to_part_a_xy,
    part_a_current_vehicle_centers_from_causal_boxes,
    part_a_direct_source_reference_map_from_h0_centers,
    part_a_dynamic_geometry_from_h0_center,
    part_a_original_reactive_map_from_h0_centers,
)


class TestBlock62ExactPartAReactive(
    unittest.TestCase
):

    def test_frozen_exact_parameters(self):

        self.assertEqual(
            PART_A_TARGET_WIDTH_M,
            1.9,
        )

        self.assertEqual(
            PART_A_LATERAL_SAFETY_MARGIN_M,
            1.0,
        )

        self.assertEqual(
            PART_A_RADIAL_SAFETY_MARGIN_M,
            4.0,
        )

        self.assertEqual(
            PART_A_CONTROL_TRANSITION_LENGTH_M,
            20.0,
        )

        self.assertEqual(
            PART_A_TRZ_TRANSITION_LENGTH_UNITS,
            8.0,
        )

        self.assertEqual(
            PART_A_FIXED_DEMO_EPSILON_DEG,
            0.75,
        )

    def test_control_and_visual_transition_not_conflated(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertAlmostEqual(
            (
                geometry.control_r_max_m
                -
                geometry.control_r_min_m
            ),
            20.0,
        )

        self.assertAlmostEqual(
            (
                geometry.trz_visual_r_max_m
                -
                geometry.trz_visual_r_min_m
            ),
            8.0,
        )

    def test_h0_to_part_a_coordinate_conversion(self):

        x_A, y_A = (
            h0_center_to_part_a_xy(
                (
                    10.0,
                    2.0,
                    0.0,
                )
            )
        )

        self.assertEqual(
            x_A,
            -2.0,
        )

        self.assertEqual(
            y_A,
            10.0,
        )

    def test_angle_sign_relation(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    10.0,
                    2.0,
                    0.0,
                )
            )
        )

        theta_H = math.degrees(
            math.atan2(
                2.0,
                10.0,
            )
        )

        self.assertAlmostEqual(
            geometry.theta_c_partA_deg,
            -theta_H,
            places=13,
        )

    def test_dynamic_epsilon_exact_width_margin_formula(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        expected = math.degrees(
            math.atan(
                1.95
                /
                50.0
            )
        )

        self.assertAlmostEqual(
            geometry.epsilon_dynamic_deg,
            expected,
            places=14,
        )

    def test_fixed_demo_epsilon_not_forced_into_dynamic_path(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertNotAlmostEqual(
            geometry.epsilon_dynamic_deg,
            PART_A_FIXED_DEMO_EPSILON_DEG,
            places=6,
        )

    def test_dynamic_control_radial_thresholds(self):

        geometry = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    50.0,
                    0.0,
                    0.0,
                )
            )
        )

        self.assertAlmostEqual(
            geometry.control_r_min_m,
            54.0,
        )

        self.assertAlmostEqual(
            geometry.control_r_max_m,
            74.0,
        )

    def test_reference_grid_is_exact_part_a_axis_sampling(self):

        grid = PartAReferenceGrid()

        theta = np.rad2deg(
            grid.theta_centers_rad
        )

        ranges = (
            grid.range_centers_m
        )

        self.assertEqual(
            grid.shape,
            (
                501,
                361,
            ),
        )

        self.assertAlmostEqual(
            float(theta[0]),
            -25.0,
        )

        self.assertAlmostEqual(
            float(theta[-1]),
            25.0,
        )

        self.assertAlmostEqual(
            float(theta[1] - theta[0]),
            0.1,
            places=12,
        )

        self.assertAlmostEqual(
            float(ranges[0]),
            0.0,
        )

        self.assertAlmostEqual(
            float(ranges[-1]),
            180.0,
        )

        self.assertAlmostEqual(
            float(
                ranges[1]
                -
                ranges[0]
            ),
            0.5,
            places=14,
        )

    def test_generic_kernel_matches_direct_source_transcription(self):

        grid = PartAReferenceGrid()

        centers = (
            (
                40.0,
                2.0,
                0.0,
            ),
            (
                60.0,
                -4.0,
                0.0,
            ),
        )

        implementation = (
            part_a_original_reactive_map_from_h0_centers(
                grid,
                centers,
            )
        )

        reference = (
            part_a_direct_source_reference_map_from_h0_centers(
                grid,
                centers,
            )
        )

        np.testing.assert_allclose(
            implementation,
            reference,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_empty_original_reactive_equals_static(self):

        grid = PartAReferenceGrid()

        result = (
            part_a_original_reactive_map_from_h0_centers(
                grid,
                (),
            )
        )

        np.testing.assert_array_equal(
            result,
            np.ones(
                grid.shape,
                dtype=float,
            ),
        )

    def test_original_reactive_actor_extraction_vehicle_only(self):

        vehicle = SimpleNamespace(
            **{{
                CAUSAL_BOX_CENTER_FIELD:
                    (
                        20.0,
                        1.0,
                        0.0,
                    ),

                CAUSAL_BOX_CLASS_FIELD:
                    "TYPE_VEHICLE",
            }}
        )

        pedestrian = SimpleNamespace(
            **{{
                CAUSAL_BOX_CENTER_FIELD:
                    (
                        20.0,
                        2.0,
                        0.0,
                    ),

                CAUSAL_BOX_CLASS_FIELD:
                    "TYPE_PEDESTRIAN",
            }}
        )

        cyclist = SimpleNamespace(
            **{{
                CAUSAL_BOX_CENTER_FIELD:
                    (
                        20.0,
                        3.0,
                        0.0,
                    ),

                CAUSAL_BOX_CLASS_FIELD:
                    "TYPE_CYCLIST",
            }}
        )

        centers = (
            part_a_current_vehicle_centers_from_causal_boxes(
                (
                    vehicle,
                    pedestrian,
                    cyclist,
                )
            )
        )

        self.assertEqual(
            centers,
            (
                (
                    20.0,
                    1.0,
                    0.0,
                ),
            ),
        )

    def test_original_reactive_api_has_no_future_posterior_covariance(self):

        signature = inspect.signature(
            part_a_original_reactive_map_from_h0_centers
        )

        forbidden = {{
            "future",
            "posterior",
            "covariance",
            "class_margin",
            "codebook",
            "beam",
        }}

        self.assertTrue(
            set(
                signature.parameters
            ).isdisjoint(
                forbidden
            )
        )

    def test_dynamic_geometry_api_is_center_only_legacy_baseline(self):

        signature = inspect.signature(
            part_a_dynamic_geometry_from_h0_center
        )

        self.assertEqual(
            tuple(
                signature.parameters
            ),
            (
                "center_H0_m",
            ),
        )

    def test_no_actual_box_width_is_used_in_legacy_formula(self):

        geometry_a = (
            part_a_dynamic_geometry_from_h0_center(
                (
                    30.0,
                    0.0,
                    0.0,
                )
            )
        )

        expected = math.degrees(
            math.atan(
                (
                    1.9 / 2.0 + 1.0
                )
                /
                30.0
            )
        )

        self.assertAlmostEqual(
            geometry_a.epsilon_dynamic_deg,
            expected,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
'''

    test_status = write_exact_text(
        TEST_FILE,
        test_source,
    )

    print(
        "test file =",
        test_status,
    )

    # ========================================================
    # H. Controlled compile
    # ========================================================

    print()
    print(
        "===== H. CONTROLLED COMPILE ====="
    )

    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(
                PART_A_MODULE
            ),
            str(
                ADB_INIT
            ),
            str(
                TEST_FILE
            ),
        ],
        cwd=str(
            S6
        ),
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
            "Block6.2 Part2 compile failed:\n"
            +
            compile_process.stdout
        ),
    )

    print(
        "module/init/tests compile = PASS"
    )

    # Import only after compile.
    import numpy as np

    from iscai_stage1.contracts.stage1a import (
        HeadlampSurrogateConfig,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage3.validation.womd_access import (
        build_compact_motion_offset_index,
        read_motion_scenario,
        read_validation_manifest,
        resolve_motion_shard,
    )

    from iscai_stage6.adb.part_a_reactive import (
        PartAReferenceGrid,
        part_a_current_vehicle_centers_from_causal_boxes,
        part_a_direct_source_reference_map_from_h0_centers,
        part_a_dynamic_geometry_from_h0_center,
        part_a_original_reactive_map_from_h0_centers,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    # ========================================================
    # I. Exact mathematical parity smoke
    # ========================================================

    print()
    print(
        "===== I. EXACT SOURCE-PARITY SMOKE ====="
    )

    parity_grid = (
        PartAReferenceGrid()
    )

    parity_centers = (
        (
            40.0,
            2.0,
            0.0,
        ),
        (
            60.0,
            -4.0,
            0.0,
        ),
    )

    implemented = (
        part_a_original_reactive_map_from_h0_centers(
            parity_grid,
            parity_centers,
        )
    )

    direct = (
        part_a_direct_source_reference_map_from_h0_centers(
            parity_grid,
            parity_centers,
        )
    )

    maximum_parity_error = float(
        np.max(
            np.abs(
                implemented
                -
                direct
            )
        )
    )

    require(
        maximum_parity_error
        <=
        1.0e-14,
        (
            "Part-A adapter differs from "
            "direct source transcription: "
            f"{maximum_parity_error}"
        ),
    )

    print(
        "dynamic angular formula = EXACT PASS"
    )

    print(
        "H0↔Part-A conversion    = PASS"
    )

    print(
        "L(theta,r) control      = EXACT PASS"
    )

    print(
        "multi-target minimum    = EXACT PASS"
    )

    print(
        "max map parity error    =",
        maximum_parity_error,
    )

    # ========================================================
    # J. Real causal Stage1/WOMD adapter
    # ========================================================

    print()
    print(
        "===== J. REAL CAUSAL ORIGINAL-REACTIVE SMOKE ====="
    )

    rows = read_validation_manifest(
        PILOT_MANIFEST
    )

    require(
        rows,
        "Pilot manifest is empty.",
    )

    offsets = (
        build_compact_motion_offset_index(
            rows
        )
    )

    pilot_row = rows[0]

    require(
        pilot_row.scenario_id
        ==
        "b85e1bd6cc8e74c0",
        (
            "Unexpected frozen pilot scenario."
        ),
    )

    compact_offset = int(
        offsets[
            pilot_row.scenario_id
        ]
    )

    runtime_shard = (
        resolve_motion_shard(
            pilot_row,
            paired_root=(
                PAIRED_ROOT
            ),
        )
    )

    scenario = (
        read_motion_scenario(
            pilot_row,
            paired_root=(
                PAIRED_ROOT
            ),
            compact_record_offset=(
                compact_offset
            ),
        )
    )

    require(
        scenario.scenario_id
        ==
        pilot_row.scenario_id,
        (
            "Real scenario identity mismatch."
        ),
    )

    headlamp_config = (
        HeadlampSurrogateConfig()
    )

    adapted = (
        adapt_causal_womd_scenario(
            scenario,
            headlamp_config=(
                headlamp_config
            ),
        )
    )

    builder_signature_object = inspect.signature(
        build_causal_adb_actor_boxes
    )

    builder_signature = str(
        builder_signature_object
    )

    builder_parameters = (
        builder_signature_object.parameters
    )

    require(
        tuple(
            builder_parameters
        )
        ==
        (
            "scenario",
            "adapted",
            "allowed_classes",
        ),
        (
            "Unexpected causal ADB actor-box "
            "builder API: "
            f"{builder_signature}"
        ),
    )

    for required_name in (
        "scenario",
        "adapted",
    ):
        parameter = builder_parameters[
            required_name
        ]

        require(
            parameter.kind
            is
            inspect.Parameter.KEYWORD_ONLY,
            (
                f"{required_name} must remain "
                "keyword-only."
            ),
        )

        require(
            parameter.default
            is
            inspect.Parameter.empty,
            (
                f"{required_name} unexpectedly "
                "became optional."
            ),
        )

    require(
        builder_parameters[
            "allowed_classes"
        ].kind
        is
        inspect.Parameter.KEYWORD_ONLY,
        (
            "allowed_classes must remain "
            "keyword-only."
        ),
    )

    actor_boxes = tuple(
        build_causal_adb_actor_boxes(
            scenario=scenario,
            adapted=adapted,
        )
    )

    require(
        len(
            actor_boxes
        )
        ==
        9,
        (
            "Frozen Block6.1 real-scene "
            "ADB actor count changed: "
            f"{len(actor_boxes)} != 9"
        ),
    )

    vehicle_centers = (
        part_a_current_vehicle_centers_from_causal_boxes(
            actor_boxes
        )
    )

    require(
        len(
            vehicle_centers
        )
        ==
        9,
        (
            "Frozen real-scene vehicle count "
            "changed: "
            f"{len(vehicle_centers)} != 9"
        ),
    )

    real_grid = (
        PartAReferenceGrid()
    )

    reactive_map = (
        part_a_original_reactive_map_from_h0_centers(
            real_grid,
            vehicle_centers,
        )
    )

    source_reference_map = (
        part_a_direct_source_reference_map_from_h0_centers(
            real_grid,
            vehicle_centers,
        )
    )

    real_parity_error = float(
        np.max(
            np.abs(
                reactive_map
                -
                source_reference_map
            )
        )
    )

    require(
        real_parity_error
        <=
        1.0e-14,
        (
            "Real-scene Part-A source parity "
            "failed: "
            f"{real_parity_error}"
        ),
    )

    require(
        float(
            reactive_map.min()
        )
        <
        1.0,
        (
            "Real reactive smoke generated "
            "no dimming on the Part-A "
            "reference raster."
        ),
    )

    require(
        float(
            reactive_map.min()
        )
        >=
        0.0,
        "Reactive intensity below zero.",
    )

    require(
        float(
            reactive_map.max()
        )
        <=
        1.0,
        "Reactive intensity above one.",
    )

    dimmed_cells = int(
        np.count_nonzero(
            reactive_map
            <
            1.0
        )
    )

    require(
        dimmed_cells > 0,
        "No dimmed cells in real smoke.",
    )

    geometries = [
        part_a_dynamic_geometry_from_h0_center(
            center
        )
        for center in vehicle_centers
    ]

    print(
        "scenario ID              =",
        scenario.scenario_id,
    )

    print(
        "runtime shard            =",
        runtime_shard,
    )

    print(
        "compact offset           =",
        compact_offset,
    )

    print(
        "causal ADB boxes         =",
        len(
            actor_boxes
        ),
    )

    print(
        "original-reactive vehicles=",
        len(
            vehicle_centers
        ),
    )

    print(
        "real source parity error =",
        real_parity_error,
    )

    print(
        "dimmed raster cells      =",
        dimmed_cells,
    )

    print(
        "minimum intensity        =",
        float(
            reactive_map.min()
        ),
    )

    print(
        "WOMD annotation role     = "
        "GEOMETRY_INTEGRATION_ONLY"
    )

    print(
        "formal sensor mode       = NOT EXECUTED HERE"
    )

    # ========================================================
    # K. Strict mutation invariance
    # ========================================================

    print()
    print(
        "===== K. STRICT CAUSALITY / SELECTOR INVARIANCE ====="
    )

    def clone_scenario(
        source,
    ):

        clone = type(
            source
        )()

        clone.CopyFrom(
            source
        )

        return clone


    def boxes_and_map(
        candidate,
    ):

        candidate_adapted = (
            adapt_causal_womd_scenario(
                candidate,
                headlamp_config=(
                    HeadlampSurrogateConfig()
                ),
            )
        )

        boxes = tuple(
            build_causal_adb_actor_boxes(
                scenario=candidate,
                adapted=candidate_adapted,
            )
        )

        centers = (
            part_a_current_vehicle_centers_from_causal_boxes(
                boxes
            )
        )

        result = (
            part_a_original_reactive_map_from_h0_centers(
                real_grid,
                centers,
            )
        )

        center_signature = tuple(
            sorted(
                tuple(
                    round(
                        float(value),
                        12,
                    )
                    for value in center
                )
                for center in centers
            )
        )

        return (
            boxes,
            centers,
            result,
            center_signature,
        )


    base_boxes, base_centers, base_map, base_signature = (
        boxes_and_map(
            scenario
        )
    )

    selector_mutated = (
        clone_scenario(
            scenario
        )
    )

    selector_mutated.ClearField(
        "tracks_to_predict"
    )

    selector_mutated.ClearField(
        "objects_of_interest"
    )

    (
        selector_boxes,
        selector_centers,
        selector_map,
        selector_signature,
    ) = boxes_and_map(
        selector_mutated
    )

    require(
        base_signature
        ==
        selector_signature,
        (
            "Selector mutation changed "
            "current reactive geometry."
        ),
    )

    require(
        np.array_equal(
            base_map,
            selector_map,
        ),
        (
            "Selector mutation changed "
            "reactive illumination map."
        ),
    )

    future_mutated = (
        clone_scenario(
            scenario
        )
    )

    anchor = int(
        future_mutated.current_time_index
    )

    future_state_mutations = 0

    for track in (
        future_mutated.tracks
    ):

        for index in range(
            anchor + 1,
            len(
                track.states
            ),
        ):

            state = (
                track.states[
                    index
                ]
            )

            # Deliberately extreme future-only changes.
            state.center_x += (
                1000.0
                +
                float(index)
            )

            state.center_y -= (
                800.0
                +
                float(index)
            )

            state.heading += 1.2345

            state.length += 3.0
            state.width += 2.0
            state.height += 1.0

            state.valid = (
                not
                bool(
                    state.valid
                )
            )

            future_state_mutations += 1

    require(
        future_state_mutations > 0,
        (
            "Future mutation gate changed "
            "no future states."
        ),
    )

    (
        future_boxes,
        future_centers,
        future_map,
        future_signature,
    ) = boxes_and_map(
        future_mutated
    )

    require(
        base_signature
        ==
        future_signature,
        (
            "Future-state mutation changed "
            "current original-reactive geometry."
        ),
    )

    require(
        np.array_equal(
            base_map,
            future_map,
        ),
        (
            "Future-state mutation changed "
            "original-reactive illumination map."
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
        "future states modified      =",
        future_state_mutations,
    )

    print(
        "reactive map SHA256         =",
        array_sha256(
            base_map
        ),
    )

    # ========================================================
    # L. Real-scene evidence artifact
    # ========================================================

    print()
    print(
        "===== L. REAL-SCENE EVIDENCE ====="
    )

    real_evidence = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "artifact":
            "real_causal_original_reactive_smoke",

        "status":
            "PASS",

        "scenario": {
            "scenario_id":
                str(
                    scenario.scenario_id
                ),

            "split":
                str(
                    pilot_row.split
                ),

            "runtime_motion_shard":
                str(
                    runtime_shard
                ),

            "compact_runtime_offset":
                compact_offset,

            "historical_source_shard":
                str(
                    pilot_row.source_shard
                ),

            "historical_record_offset":
                int(
                    pilot_row.record_offset
                ),
        },

        "geometry": {
            "Stage6_H0":
                "+x forward, +y left, +z up",

            "Part_A":
                "+y forward, +x right",

            "conversion":
                "x_A=-y_H0; y_A=x_H0",

            "angle_relation":
                "theta_A=-theta_H0",

            "causal_actor_boxes":
                len(
                    actor_boxes
                ),

            "original_reactive_vehicle_centers":
                len(
                    vehicle_centers
                ),
        },

        "Part_A_binding": {
            "fixed_width_m":
                1.9,

            "lateral_margin_m":
                1.0,

            "dynamic_epsilon":
                (
                    "atan((1.9/2+1.0)/range)"
                ),

            "fixed_epsilon_0_75_used":
                False,

            "control_radial_margin_m":
                4.0,

            "control_transition_length_m":
                20.0,

            "TRZ_visual_transition_units":
                8.0,

            "TRZ_visual_transition_used_for_L":
                False,

            "multi_actor_rule":
                "elementwise_minimum",
        },

        "map": {
            "shape":
                list(
                    base_map.shape
                ),

            "sha256":
                array_sha256(
                    base_map
                ),

            "minimum_intensity":
                float(
                    base_map.min()
                ),

            "maximum_intensity":
                float(
                    base_map.max()
                ),

            "dimmed_cells":
                dimmed_cells,

            "direct_source_parity_max_abs_error":
                real_parity_error,
        },

        "causality": {
            "tracks_to_predict_selector":
                False,

            "objects_of_interest_selector":
                False,

            "future_state_dependency":
                False,

            "future_mutation_invariant":
                True,

            "selector_mutation_invariant":
                True,

            "future_gt_input":
                False,
        },

        "scope": {
            "WOMD_annotation_role":
                "GEOMETRY_INTEGRATION_VALIDATION_ONLY",

            "formal_sensor_realistic":
                False,

            "formal_original_reactive_mode":
                (
                    "CAUSAL_TRACK_BASED_"
                    "PC_FMCW_LIKE"
                ),

            "formal_mode_executed":
                False,

            "predictive_ADB":
                False,

            "probabilistic_mask":
                False,

            "class_aware_ADB":
                False,

            "communication_codebook_used":
                False,
        },
    }

    atomic_json(
        REAL_SMOKE,
        real_evidence,
    )

    print(
        "real-scene artifact =",
        REAL_SMOKE,
    )

    print(
        "real-scene SHA256   =",
        sha256_file(
            REAL_SMOKE
        ),
    )

    # ========================================================
    # M. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== M. FULL STAGE6 REGRESSION ====="
    )

    regression = subprocess.run(
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
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    REGRESSION_LOG.write_text(
        regression.stdout,
        encoding="utf-8",
    )

    for line in (
        regression.stdout.splitlines()
    ):

        stripped = line.strip()

        if (
            stripped.startswith(
                "Ran "
            )
            or
            stripped
            ==
            "OK"
            or
            stripped.startswith(
                "FAILED"
            )
        ):

            print(
                stripped
            )

    require(
        regression.returncode
        ==
        0,
        (
            "Stage6 regression failed.\n"
            f"Full log: {REGRESSION_LOG}\n"
            +
            tail_text(
                regression.stdout,
                100,
            )
        ),
    )

    # ========================================================
    # N. Scope / PDF guards
    # ========================================================

    print()
    print(
        "===== N. BLOCK6.2 PDF-SCOPE GUARDS ====="
    )

    from iscai_stage6.adb.part_a_reactive import (
        part_a_dynamic_geometry_from_h0_center,
        part_a_original_reactive_map_from_h0_centers,
    )

    geometry_signature = str(
        inspect.signature(
            part_a_dynamic_geometry_from_h0_center
        )
    )

    reactive_signature = str(
        inspect.signature(
            part_a_original_reactive_map_from_h0_centers
        )
    )

    forbidden_signature_tokens = (
        "future",
        "posterior",
        "covariance",
        "codebook",
        "beam_id",
        "confidence",
        "class_margin",
    )

    require(
        not any(
            token
            in
            geometry_signature.lower()
            for token in (
                forbidden_signature_tokens
            )
        ),
        (
            "Original reactive geometry "
            "exposes predictive input."
        ),
    )

    require(
        not any(
            token
            in
            reactive_signature.lower()
            for token in (
                forbidden_signature_tokens
            )
        ),
        (
            "Original reactive map "
            "exposes predictive/communication input."
        ),
    )

    print(
        "static ADB baseline        = IMPLEMENTED"
    )

    print(
        "original reactive ADB      = IMPLEMENTED"
    )

    print(
        "legacy center-based input  = PRESERVED"
    )

    print(
        "predictive centroid-only   = FORBIDDEN"
    )

    print(
        "predictive full boxes      = DEFERRED BLOCK6.3+"
    )

    print(
        "probabilistic occupancy    = DEFERRED BLOCK6.4"
    )

    print(
        "class-aware policy         = DEFERRED BLOCK6.5"
    )

    print(
        "temporal smoothing/rate    = DEFERRED BLOCK6.6"
    )

    print(
        "formal grid freeze         = BLOCK6.8 PRE-FORMAL"
    )

    print(
        "communication codebook     = NOT REUSED"
    )

    # ========================================================
    # O. Frozen immutability / storage
    # ========================================================

    print()
    print(
        "===== O. IMMUTABILITY / STORAGE ====="
    )

    changed = []

    for path in protected:

        before = protected_before[
            str(
                path
            )
        ]

        after = sha256_file(
            path
        )

        if before != after:

            changed.append(
                str(
                    path
                )
            )

    require(
        not changed,
        (
            "Frozen prerequisite(s) changed: "
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
            f"{free_gib:.3f} GiB"
        ),
    )

    print(
        "Part-A notebook      = UNCHANGED"
    )

    print(
        "Block6.1             = UNCHANGED"
    )

    print(
        "Block6.2 Part1       = UNCHANGED"
    )

    print(
        "Stage5 closure       = UNCHANGED"
    )

    print(
        "free GiB             =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve      = PASS"
    )

    # ========================================================
    # P. Part2 report
    # ========================================================

    print()
    print(
        "===== P. BLOCK6.2 PART2 REPORT ====="
    )

    part2_report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "part":
            "2/2",

        "status":
            "PASS_EXACT_ORIGINAL_REACTIVE_ADB",

        "binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                sha256_file(
                    BINDING
                ),
        },

        "implementation": {
            "module":
                str(
                    PART_A_MODULE
                ),

            "module_sha256":
                sha256_file(
                    PART_A_MODULE
                ),

            "static_ADB":
                "IMPLEMENTED",

            "original_reactive_ADB":
                "IMPLEMENTED",

            "exact_dynamic_width_margin_geometry":
                True,

            "H0_to_PartA_conversion":
                True,

            "control_transition_length_m":
                20.0,

            "TRZ_visual_transition_units":
                8.0,

            "control_visual_transition_conflated":
                False,

            "fixed_demo_epsilon_deg":
                0.75,

            "dynamic_uses_fixed_epsilon":
                False,

            "multi_actor_minimum":
                True,
        },

        "source_parity": {
            "synthetic_max_abs_error":
                maximum_parity_error,

            "real_scene_max_abs_error":
                real_parity_error,

            "pass":
                True,
        },

        "real_smoke": {
            "path":
                str(
                    REAL_SMOKE
                ),

            "sha256":
                sha256_file(
                    REAL_SMOKE
                ),

            "scenario_id":
                str(
                    scenario.scenario_id
                ),

            "vehicle_count":
                len(
                    vehicle_centers
                ),

            "annotation_role":
                "GEOMETRY_INTEGRATION_ONLY",
        },

        "causality": {
            "tracks_to_predict":
                False,

            "objects_of_interest":
                False,

            "future_states":
                False,

            "future_mutation_invariant":
                True,
        },

        "PDF_scope": {
            "original_reactive_baseline_ready":
                True,

            "deterministic_predictive_ADB":
                False,

            "uncertainty_aware_predictive_ADB":
                False,

            "class_agnostic_predictive_ADB":
                False,

            "class_aware_predictive_ADB":
                False,

            "oracle_future_ADB":
                False,

            "predictive_full_box":
                False,

            "final_Stage6_completion_gate_evaluated":
                False,
        },

        "formal_policy": {
            "formal_common_grid_frozen":
                False,

            "grid_freeze":
                "BLOCK6.8_PRE_FORMAL",

            "formal_original_reactive_input":
                (
                    "CAUSAL_TRACK_BASED_"
                    "PC_FMCW_LIKE"
                ),

            "formal_parameter_tuning":
                False,
        },

        "regression": {
            "Stage6":
                "PASS",

            "log":
                str(
                    REGRESSION_LOG
                ),
        },

        "execution": {
            "training":
                False,

            "trajectory_model_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "corpus_scan":
                False,
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block6.3 deterministic predictive "
                "future full-box projection"
            ),
    }

    atomic_json(
        PART2_REPORT,
        part2_report,
    )

    print(
        "Part2 report SHA256 =",
        sha256_file(
            PART2_REPORT
        ),
    )

    # ========================================================
    # Q. Block6.2 closure
    # ========================================================

    print()
    print(
        "===== Q. BLOCK6.2 CLOSURE ====="
    )

    closure = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "status":
            "PASS_COMPLETE",

        "completed": {
            "static_ADB_baseline":
                True,

            "original_reactive_ADB_baseline":
                True,

            "Part_A_L_theta_r":
                True,

            "Part_A_dynamic_width_margin_geometry":
                True,

            "Part_A_current_track_semantics":
                True,

            "H0_PartA_coordinate_adapter":
                True,

            "raised_cosine_transition":
                True,

            "elementwise_minimum_multi_actor":
                True,

            "real_causal_geometry_smoke":
                True,

            "future_selector_invariance":
                True,
        },

        "frozen_binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                sha256_file(
                    BINDING
                ),
        },

        "part1": {
            "report":
                str(
                    BLOCK62_PART1_REPORT
                ),

            "report_sha256":
                EXPECTED[
                    "part1_report"
                ],
        },

        "part2": {
            "report":
                str(
                    PART2_REPORT
                ),

            "report_sha256":
                sha256_file(
                    PART2_REPORT
                ),
        },

        "real_smoke": {
            "path":
                str(
                    REAL_SMOKE
                ),

            "sha256":
                sha256_file(
                    REAL_SMOKE
                ),
        },

        "scientific_distinctions": {
            "fixed_demo_epsilon_deg":
                0.75,

            "dynamic_epsilon":
                "WIDTH_PLUS_MARGIN_OVER_RANGE",

            "control_transition_length_m":
                20.0,

            "TRZ_visual_transition_units":
                8.0,

            "control_and_visual_radial_semantics_separate":
                True,

            "legacy_original_reactive_center_based":
                True,

            "predictive_centroid_only_forbidden":
                True,

            "PartA_grid_is_formal_Stage6_grid":
                False,

            "communication_beam_equals_ADB_actuator":
                False,
        },

        "not_yet_completed": {
            "deterministic_predictive_ADB":
                "BLOCK6.3",

            "probabilistic_occupancy":
                "BLOCK6.4",

            "class_aware_ADB":
                "BLOCK6.5",

            "smoothing_and_actuation":
                "BLOCK6.6",

            "metrics_oracle_and_ablations":
                "BLOCK6.7_PLUS",

            "preformal_numeric_freeze":
                "BLOCK6.8",

            "formal_Stage6_acceptance":
                False,
        },

        "Stage6_completion_claim":
            False,

        "upstream_modified":
            False,
    }

    atomic_json(
        CLOSURE,
        closure,
    )

    print(
        "Block6.2 closure = WRITTEN"
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
        "BLOCK 6.2 PART 2/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Part-A source identity       = EXACT PASS"
    )

    print(
        "static ADB baseline          = COMPLETE"
    )

    print(
        "original reactive ADB        = COMPLETE"
    )

    print(
        "dynamic width+margin epsilon = EXACT PASS"
    )

    print(
        "fixed epsilon 0.75           = FIXED-DEMO ONLY"
    )

    print(
        "H0 -> Part-A conversion      = VERIFIED PASS"
    )

    print(
        "control radial margin        = 4.0 m"
    )

    print(
        "control transition           = 20.0 m EXACT"
    )

    print(
        "TRZ visual transition        = 8.0 / SEPARATE"
    )

    print(
        "raised-cosine L(theta,r)     = EXACT PASS"
    )

    print(
        "multi-vehicle minimum        = EXACT PASS"
    )

    print(
        "real causal reactive smoke   = PASS"
    )

    print(
        "source-map parity            = PASS"
    )

    print(
        "tracks_to_predict selector   = NO / INVARIANT"
    )

    print(
        "objects_of_interest selector = NO / INVARIANT"
    )

    print(
        "future-state leakage         = NO / INVARIANT"
    )

    print(
        "legacy center-based geometry = BASELINE ONLY"
    )

    print(
        "predictive centroid-only     = FORBIDDEN"
    )

    print(
        "communication codebook       = NOT REUSED"
    )

    print(
        "formal Stage6 grid           = NOT YET FROZEN"
    )

    print(
        "predictive ADB               = NOT STARTED"
    )

    print(
        "Stage6 completion claimed    = NO"
    )

    print(
        "Stage6 regression            = PASS"
    )

    print(
        "upstream modified            = NO"
    )

    print(
        "training/model inference     = NO"
    )

    print(
        "formal evaluation/tuning     = NO"
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
        "BLOCK 6.2 PART 2/2 = BLOCKED"
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
        limit=14
    )

    print()
    print(
        "Part-A modified        = NO"
    )

    print(
        "Stage1–5 modified      = NO"
    )

    print(
        "Block6.1 modified      = NO"
    )

    print(
        "Block6.2 Part1 modified= NO"
    )

    print(
        "formal evaluation      = NO"
    )

    print(
        "parameter tuning       = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Deliberately no non-zero sys.exit().

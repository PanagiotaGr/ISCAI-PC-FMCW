from __future__ import annotations

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
# Deterministic project source bootstrap
# ============================================================

def bootstrap_project_sources():

    source_roots = tuple(
        ROOT
        / f"iscai_stage{index}"
        / "src"
        for index in range(7)
    )

    missing = [
        str(path)
        for path in source_roots
        if not path.is_dir()
    ]

    if missing:
        raise RuntimeError(
            "Missing project source root(s): "
            + ", ".join(missing)
        )

    for path in reversed(
        source_roots
    ):
        value = str(path)

        while value in sys.path:
            sys.path.remove(value)

        sys.path.insert(
            0,
            value,
        )

    old = [
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
            for path in source_roots
        ]
        +
        old
    ):
        if item not in seen:
            seen.add(item)
            merged.append(item)

    os.environ[
        "PYTHONPATH"
    ] = os.pathsep.join(
        merged
    )

    return tuple(
        str(path)
        for path in source_roots
    )


PROJECT_SOURCE_ROOTS = (
    bootstrap_project_sources()
)


S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

BLOCK61_CLOSURE = (
    S6
    / "reports/block61_closure.json"
)

BLOCK60_ADAPTER_CONTRACT = (
    S6
    / "configs/"
      "part_a_reactive_adb_adapter_contract.json"
)

BLOCK60_SOURCE_AUDIT = (
    S6
    / "artifacts/block60/"
      "part_a_adb_source_audit.json"
)

STAGE6_CONTRACT = (
    S6
    / "configs/stage6_contract.json"
)

STAGE5_CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

ILLUMINATION_MODULE = (
    S6
    / "src/iscai_stage6/adb/"
      "illumination.py"
)

ADB_INIT = (
    S6
    / "src/iscai_stage6/adb/"
      "__init__.py"
)

TEST_FILE = (
    S6
    / "tests/"
      "test_block62_static_reactive_kernel.py"
)

CONTRACT = (
    S6
    / "configs/"
      "block62_static_reactive_kernel_contract.json"
)

REPORT = (
    S6
    / "reports/"
      "block62_part1_static_reactive_kernel.json"
)

REGRESSION_LOG = (
    S6
    / "artifacts/block62/"
      "block62_part1_full_regression.log"
)

EXPECTED = {
    "block61_closure":
        (
            "1d0f8781d365efd09614c8fb5a06bdbd"
            "a8f56c1c673b0ddab8b40810de270ac1"
        ),

    "adapter_contract":
        (
            "d552e9d766ec356fbf57079a647a7b3c"
            "5e882a6e153c8b9e5f06c8b207539e4c"
        ),

    "source_audit":
        (
            "5b3b8aacaa6c8947c6d9d57264a90899"
            "4a6a8d3193179a75bcd4f6b4697b0e64"
        ),

    "stage6_contract":
        (
            "6b2ee6454f876c3658d4d308b827202c"
            "d31dd7c77bf96c38e6c6da70e6573675"
        ),

    "stage5_closure":
        (
            "c83731948749f3ca20742aaac6b7474f"
            "53c755cbc8209dd31db969d229f60610"
        ),
}

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
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

    tmp.write_bytes(desired)

    os.replace(
        tmp,
        path,
    )

    return "WRITTEN"


def tail_text(
    text: str,
    lines: int = 80,
) -> str:

    rows = text.splitlines()

    return "\n".join(
        rows[-lines:]
    )


def run_stage6_tests():

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

    REGRESSION_LOG.write_text(
        process.stdout,
        encoding="utf-8",
    )

    return (
        process.returncode,
        process.stdout,
    )


def ensure_adb_exports():

    require(
        ADB_INIT.is_file(),
        (
            "Stage6 ADB __init__.py missing: "
            f"{ADB_INIT}"
        ),
    )

    original = ADB_INIT.read_text(
        encoding="utf-8"
    )

    text = original

    import_block = '''from .illumination import (
    ReactiveShadowRegion,
    angular_shadow_mask,
    combine_illumination_maps_minimum,
    part_a_radial_profile,
    reactive_adb_map,
    single_region_illumination_map,
    static_adb_map,
)
'''

    if (
        import_block
        not in
        text
    ):

        marker = "__all__ = ("

        require(
            text.count(marker) == 1,
            (
                "Could not uniquely locate "
                "__all__ in adb/__init__.py"
            ),
        )

        text = text.replace(
            marker,
            import_block
            + "\n"
            + marker,
            1,
        )

    export_lines = (
        '    "ReactiveShadowRegion",\n'
        '    "angular_shadow_mask",\n'
        '    "combine_illumination_maps_minimum",\n'
        '    "part_a_radial_profile",\n'
        '    "reactive_adb_map",\n'
        '    "single_region_illumination_map",\n'
        '    "static_adb_map",\n'
    )

    if (
        '"ReactiveShadowRegion"'
        not in
        text
    ):

        marker = "__all__ = ("

        text = text.replace(
            marker,
            marker
            + "\n"
            + export_lines,
            1,
        )

    if text == original:
        return "ALREADY_UPDATED"

    tmp = ADB_INIT.with_suffix(
        ".py.block62.tmp"
    )

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        ADB_INIT,
    )

    return "UPDATED_BLOCK62"


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.2 PART 1/2"
    )
    print(
        "STATIC ADB + PART-A REACTIVE L(theta,r) KERNEL"
    )
    print(
        "============================================================"
    )

    protected = (
        BLOCK61_CLOSURE,
        BLOCK60_ADAPTER_CONTRACT,
        BLOCK60_SOURCE_AUDIT,
        STAGE6_CONTRACT,
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
            "Missing frozen prerequisite(s): "
            + ", ".join(missing)
        ),
    )

    protected_before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Frozen prerequisites
    # ========================================================

    print()
    print(
        "===== A. FROZEN PREREQUISITES ====="
    )

    checks = (
        (
            "block61_closure",
            BLOCK61_CLOSURE,
        ),
        (
            "adapter_contract",
            BLOCK60_ADAPTER_CONTRACT,
        ),
        (
            "source_audit",
            BLOCK60_SOURCE_AUDIT,
        ),
        (
            "stage6_contract",
            STAGE6_CONTRACT,
        ),
        (
            "stage5_closure",
            STAGE5_CLOSURE,
        ),
    )

    for key, path in checks:

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            EXPECTED[
                key
            ],
            (
                f"{key} SHA mismatch.\n"
                f"expected={EXPECTED[key]}\n"
                f"actual={actual}"
            ),
        )

    block61 = load_json(
        BLOCK61_CLOSURE
    )

    adapter_contract = load_json(
        BLOCK60_ADAPTER_CONTRACT
    )

    source_audit = load_json(
        BLOCK60_SOURCE_AUDIT
    )

    require(
        block61.get("status")
        ==
        "PASS_COMPLETE",
        (
            "Block6.1 is not PASS_COMPLETE."
        ),
    )

    require(
        adapter_contract.get(
            "status"
        )
        ==
        "FROZEN_REACTIVE_ADB_ADAPTER_CONTRACT",
        (
            "Frozen reactive adapter contract "
            "is not active."
        ),
    )

    require(
        source_audit.get(
            "status"
        )
        ==
        "PASS_PART_A_ADB_SOURCE_AUDIT",
        (
            "Part-A ADB source audit "
            "is not PASS."
        ),
    )

    print(
        "Block6.1 closure        = EXACT PASS"
    )

    print(
        "Part-A source audit     = EXACT PASS"
    )

    print(
        "reactive adapter contract= EXACT PASS"
    )

    print(
        "Stage6 strict contract  = EXACT PASS"
    )

    print(
        "Stage5 closure          = UNCHANGED PASS"
    )

    # ========================================================
    # B. Implement normalized Part-A illumination kernel
    # ========================================================

    print()
    print(
        "===== B. PART-A L(theta,r) KERNEL ====="
    )

    illumination_source = r'''from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .geometry import (
    wrap_angle,
)

from .grid import (
    IlluminationGridSpec,
)


_EPS = 1.0e-12


@dataclass(frozen=True)
class ReactiveShadowRegion:
    """
    One current deterministic ADB shadow region.

    This class deliberately describes illumination geometry only.
    It has no prediction posterior, covariance, future ground-truth,
    communication-beam or class-policy inputs.

    theta_center_rad / theta_span_rad:
        Horizontal angular shadow interval.

    r_shadow_end_m:
        Part-A/paper r_min:
        strongest attenuation applies up to this range.

    r_transition_end_m:
        Part-A/paper r_max:
        raised-cosine transition reaches full illumination here.

    intensity_floor:
        Normalized minimum illumination.
        floor=0 reproduces Eq. (5) of the reference paper.
        Non-zero floors are supported for later Stage6 policies.
    """

    theta_center_rad: float
    theta_span_rad: float

    r_shadow_end_m: float
    r_transition_end_m: float

    intensity_floor: float = 0.0

    source_semantics: str = (
        "current_deterministic_region"
    )

    def __post_init__(
        self,
    ) -> None:

        center = float(
            self.theta_center_rad
        )

        span = float(
            self.theta_span_rad
        )

        r_shadow = float(
            self.r_shadow_end_m
        )

        r_transition = float(
            self.r_transition_end_m
        )

        floor = float(
            self.intensity_floor
        )

        for name, value in (
            (
                "theta_center_rad",
                center,
            ),
            (
                "theta_span_rad",
                span,
            ),
            (
                "r_shadow_end_m",
                r_shadow,
            ),
            (
                "r_transition_end_m",
                r_transition,
            ),
            (
                "intensity_floor",
                floor,
            ),
        ):
            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not (
            0.0
            <=
            span
            <=
            2.0 * math.pi
        ):
            raise ValueError(
                "theta_span_rad must lie "
                "within [0, 2*pi]"
            )

        if r_shadow < 0.0:
            raise ValueError(
                "r_shadow_end_m must be >= 0"
            )

        if not (
            r_transition
            >
            r_shadow
        ):
            raise ValueError(
                "r_transition_end_m must be "
                "strictly greater than "
                "r_shadow_end_m"
            )

        if not (
            0.0
            <=
            floor
            <=
            1.0
        ):
            raise ValueError(
                "intensity_floor must lie "
                "within [0,1]"
            )

        source = str(
            self.source_semantics
        )

        if source != (
            "current_deterministic_region"
        ):
            raise ValueError(
                "ReactiveShadowRegion Part-A "
                "kernel accepts only "
                "current_deterministic_region "
                "semantics"
            )

        object.__setattr__(
            self,
            "theta_center_rad",
            float(
                wrap_angle(
                    center
                )
            ),
        )

        object.__setattr__(
            self,
            "theta_span_rad",
            span,
        )

        object.__setattr__(
            self,
            "r_shadow_end_m",
            r_shadow,
        )

        object.__setattr__(
            self,
            "r_transition_end_m",
            r_transition,
        )

        object.__setattr__(
            self,
            "intensity_floor",
            floor,
        )

        object.__setattr__(
            self,
            "source_semantics",
            source,
        )


def part_a_radial_profile(
    range_m,
    *,
    r_shadow_end_m: float,
    r_transition_end_m: float,
    intensity_floor: float = 0.0,
):
    """
    Normalized Part-A/paper raised-cosine radial profile.

    For floor = 0 this is exactly:

      0,
        r <= r_min

      0.5 * (
          1 - cos(
              pi * (r-r_min)/(r_max-r_min)
          )
      ),
        r_min < r < r_max

      1,
        r >= r_max

    A non-zero floor preserves the same raised-cosine shape
    while lifting the minimum intensity:

      floor + (1-floor) * raised_cosine.

    This generalization is needed later for VRU visibility
    policies but does not alter the floor=0 Part-A baseline.
    """

    r_shadow = float(
        r_shadow_end_m
    )

    r_transition = float(
        r_transition_end_m
    )

    floor = float(
        intensity_floor
    )

    if (
        not math.isfinite(
            r_shadow
        )
        or
        r_shadow < 0.0
    ):
        raise ValueError(
            "r_shadow_end_m must be finite "
            "and >= 0"
        )

    if (
        not math.isfinite(
            r_transition
        )
        or
        r_transition
        <=
        r_shadow
    ):
        raise ValueError(
            "r_transition_end_m must be finite "
            "and > r_shadow_end_m"
        )

    if (
        not math.isfinite(
            floor
        )
        or
        not (
            0.0 <= floor <= 1.0
        )
    ):
        raise ValueError(
            "intensity_floor must lie "
            "within [0,1]"
        )

    raw = np.asarray(
        range_m,
        dtype=np.float64,
    )

    scalar_input = (
        raw.ndim == 0
    )

    values = np.atleast_1d(
        raw
    )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "range_m contains non-finite values"
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "range_m must be non-negative"
        )

    profile = np.ones_like(
        values,
        dtype=np.float64,
    )

    shadow = (
        values
        <=
        r_shadow
    )

    transition = (
        values
        >
        r_shadow
    ) & (
        values
        <
        r_transition
    )

    profile[
        shadow
    ] = floor

    if np.any(
        transition
    ):

        u = (
            values[
                transition
            ]
            -
            r_shadow
        ) / (
            r_transition
            -
            r_shadow
        )

        raised_cosine = (
            0.5
            *
            (
                1.0
                -
                np.cos(
                    np.pi * u
                )
            )
        )

        profile[
            transition
        ] = (
            floor
            +
            (
                1.0
                -
                floor
            )
            *
            raised_cosine
        )

    profile = np.clip(
        profile,
        floor,
        1.0,
    )

    if scalar_input:
        return float(
            profile[0]
        )

    return profile.reshape(
        raw.shape
    )


def angular_shadow_mask(
    theta_rad,
    *,
    theta_center_rad: float,
    theta_span_rad: float,
):
    """
    Circular angular interval membership.

    Handles shadow intervals that cross -pi/+pi.
    """

    center = float(
        theta_center_rad
    )

    span = float(
        theta_span_rad
    )

    if (
        not math.isfinite(
            center
        )
        or
        not math.isfinite(
            span
        )
    ):
        raise ValueError(
            "angular parameters must be finite"
        )

    if not (
        0.0
        <=
        span
        <=
        2.0 * math.pi
    ):
        raise ValueError(
            "theta_span_rad must lie "
            "within [0,2*pi]"
        )

    raw = np.asarray(
        theta_rad,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            raw
        )
    ):
        raise ValueError(
            "theta_rad contains non-finite values"
        )

    if (
        span
        >=
        2.0 * math.pi
        -
        _EPS
    ):
        return np.ones(
            raw.shape,
            dtype=bool,
        )

    delta = wrap_angle(
        raw
        -
        wrap_angle(
            center
        )
    )

    return (
        np.abs(
            delta
        )
        <=
        0.5 * span
        +
        _EPS
    )


def static_adb_map(
    grid: IlluminationGridSpec,
) -> np.ndarray:
    """
    Static/no-dimming ADB baseline.

    Normalized intensity = 1 everywhere.
    """

    return np.ones(
        grid.shape,
        dtype=np.float64,
    )


def single_region_illumination_map(
    grid: IlluminationGridSpec,
    region: ReactiveShadowRegion,
) -> np.ndarray:
    """
    Apply one Part-A-style angular/radial shadow region.
    """

    angular = angular_shadow_mask(
        grid.theta_centers_rad,
        theta_center_rad=(
            region.theta_center_rad
        ),
        theta_span_rad=(
            region.theta_span_rad
        ),
    )

    radial = part_a_radial_profile(
        grid.range_centers_m,
        r_shadow_end_m=(
            region.r_shadow_end_m
        ),
        r_transition_end_m=(
            region.r_transition_end_m
        ),
        intensity_floor=(
            region.intensity_floor
        ),
    )

    result = static_adb_map(
        grid
    )

    result[
        angular,
        :,
    ] = radial[
        None,
        :
    ]

    return result


def combine_illumination_maps_minimum(
    maps: Iterable[
        np.ndarray
    ],
) -> np.ndarray:
    """
    Part-A multi-target combination rule:
    the strongest required dimming wins,
    implemented as elementwise minimum.
    """

    arrays = tuple(
        np.asarray(
            item,
            dtype=np.float64,
        )
        for item in maps
    )

    if not arrays:
        raise ValueError(
            "At least one illumination map "
            "is required"
        )

    shape = arrays[0].shape

    for array in arrays:

        if array.shape != shape:
            raise ValueError(
                "All illumination maps must "
                "have identical shape"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "Illumination map contains "
                "non-finite values"
            )

        if (
            np.any(
                array < 0.0
            )
            or
            np.any(
                array > 1.0
            )
        ):
            raise ValueError(
                "Normalized illumination map "
                "must lie within [0,1]"
            )

    stacked = np.stack(
        arrays,
        axis=0,
    )

    return np.min(
        stacked,
        axis=0,
    )


def reactive_adb_map(
    grid: IlluminationGridSpec,
    regions: Iterable[
        ReactiveShadowRegion
    ],
) -> np.ndarray:
    """
    Current-state reactive ADB illumination map.

    This is intentionally NOT predictive:
      - no future trajectory,
      - no posterior,
      - no covariance,
      - no communication codebook,
      - no class-aware future policy.

    Empty regions reduce exactly to the static ADB baseline.
    """

    regions = tuple(
        regions
    )

    if not regions:
        return static_adb_map(
            grid
        )

    maps = tuple(
        single_region_illumination_map(
            grid,
            region,
        )
        for region in regions
    )

    return (
        combine_illumination_maps_minimum(
            maps
        )
    )
'''

    illumination_status = (
        write_exact_text(
            ILLUMINATION_MODULE,
            illumination_source,
        )
    )

    print(
        "illumination.py =",
        illumination_status,
    )

    init_status = ensure_adb_exports()

    print(
        "adb/__init__.py =",
        init_status,
    )

    # ========================================================
    # C. Contract
    # ========================================================

    print()
    print(
        "===== C. BLOCK6.2 PART1 CONTRACT ====="
    )

    contract = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "part":
            "1/2",

        "status":
            "STATIC_AND_REACTIVE_KERNEL_CONTRACT",

        "PDF_requirements": {
            "static_ADB_baseline":
                True,

            "original_reactive_ADB_baseline":
                True,

            "preserve_Part_A_L_theta_r":
                True,

            "radial_thresholds":
                True,

            "angular_shadow_zones":
                True,

            "raised_cosine_transition":
                True,

            "minimum_illumination_floor_support":
                True,

            "arbitrary_replacement_of_ADB_model":
                False,
        },

        "mathematical_kernel": {
            "normalized_output_range":
                "[0,1]",

            "paper_equivalent_floor":
                0.0,

            "shadow_rule":
                (
                    "theta in shadow and "
                    "r <= r_min -> floor"
                ),

            "transition_rule":
                (
                    "floor + (1-floor) * "
                    "0.5*(1-cos(pi*u))"
                ),

            "u":
                (
                    "(r-r_min)/(r_max-r_min)"
                ),

            "outside_rule":
                "full illumination = 1",

            "multiple_actor_combination":
                "elementwise minimum",
        },

        "static_baseline": {
            "normalized_intensity":
                1.0,

            "adaptive_dimming":
                False,
        },

        "reactive_baseline": {
            "state_semantics":
                "current_deterministic_region",

            "future_trajectory_input":
                False,

            "predictive_covariance_input":
                False,

            "posterior_input":
                False,

            "communication_codebook_input":
                False,

            "exact_Part_A_numeric_parameter_binding":
                "BLOCK6.2_PART2",
        },

        "future_extensions": {
            "full_box_predictive_region":
                "BLOCK6.3",

            "probabilistic_occupancy":
                "BLOCK6.4",

            "class_aware_floors_and_dimming":
                "BLOCK6.5",

            "temporal_smoothing":
                "BLOCK6.6",

            "actuation_rate_limits":
                "BLOCK6.6",
        },

        "numerical_freeze": {
            "Part_A_source_parameters_frozen":
                False,

            "final_ADB_grid_frozen":
                False,

            "final_ADB_grid_freeze":
                "BLOCK6.8_PRE_FORMAL",
        },
    }

    contract_status = write_exact_json(
        CONTRACT,
        contract,
    )

    print(
        "contract =",
        contract_status,
    )

    print(
        "contract SHA256 =",
        sha256_file(
            CONTRACT
        ),
    )

    # ========================================================
    # D. Tests
    # ========================================================

    print()
    print(
        "===== D. STATIC / REACTIVE UNIT TESTS ====="
    )

    test_source = r'''from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from iscai_stage6.adb.grid import (
    IlluminationGridSpec,
)

from iscai_stage6.adb.illumination import (
    ReactiveShadowRegion,
    angular_shadow_mask,
    combine_illumination_maps_minimum,
    part_a_radial_profile,
    reactive_adb_map,
    single_region_illumination_map,
    static_adb_map,
)


class TestBlock62StaticReactiveKernel(
    unittest.TestCase
):

    def grid(self):

        return IlluminationGridSpec(
            theta_min_rad=-math.pi,
            theta_max_rad=math.pi,
            range_min_m=0.0,
            range_max_m=100.0,
            n_theta=72,
            n_range=100,
        )

    def test_static_adb_is_full_illumination(self):

        result = static_adb_map(
            self.grid()
        )

        self.assertEqual(
            result.shape,
            (72, 100),
        )

        self.assertTrue(
            np.all(
                result == 1.0
            )
        )

    def test_exact_part_a_floor_zero_shadow(self):

        values = (
            part_a_radial_profile(
                np.asarray(
                    [
                        0.0,
                        10.0,
                        20.0,
                    ]
                ),
                r_shadow_end_m=20.0,
                r_transition_end_m=40.0,
                intensity_floor=0.0,
            )
        )

        np.testing.assert_allclose(
            values,
            np.zeros(3),
            atol=1.0e-15,
        )

    def test_exact_part_a_transition_midpoint(self):

        value = part_a_radial_profile(
            30.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=0.0,
        )

        expected = (
            0.5
            *
            (
                1.0
                -
                math.cos(
                    math.pi * 0.5
                )
            )
        )

        self.assertAlmostEqual(
            value,
            expected,
            places=14,
        )

    def test_exact_part_a_transition_endpoint(self):

        value = part_a_radial_profile(
            40.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=0.0,
        )

        self.assertEqual(
            value,
            1.0,
        )

    def test_outside_transition_is_full_light(self):

        values = part_a_radial_profile(
            np.asarray(
                [
                    40.0,
                    50.0,
                    100.0,
                ]
            ),
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        np.testing.assert_allclose(
            values,
            np.ones(3),
            atol=0.0,
        )

    def test_nonzero_floor_lifts_same_curve(self):

        floor = 0.25

        start = part_a_radial_profile(
            20.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=floor,
        )

        middle = part_a_radial_profile(
            30.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=floor,
        )

        end = part_a_radial_profile(
            40.0,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
            intensity_floor=floor,
        )

        self.assertAlmostEqual(
            start,
            0.25,
        )

        self.assertAlmostEqual(
            middle,
            0.625,
        )

        self.assertAlmostEqual(
            end,
            1.0,
        )

    def test_invalid_radial_thresholds_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            part_a_radial_profile(
                20.0,
                r_shadow_end_m=30.0,
                r_transition_end_m=30.0,
            )

    def test_negative_range_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            part_a_radial_profile(
                -1.0,
                r_shadow_end_m=20.0,
                r_transition_end_m=40.0,
            )

    def test_angular_shadow_wraps_across_pi(self):

        theta = np.asarray(
            [
                math.pi - 0.05,
                -math.pi + 0.05,
                0.0,
            ]
        )

        mask = angular_shadow_mask(
            theta,
            theta_center_rad=(
                math.pi
            ),
            theta_span_rad=0.2,
        )

        self.assertTrue(
            bool(mask[0])
        )

        self.assertTrue(
            bool(mask[1])
        )

        self.assertFalse(
            bool(mask[2])
        )

    def test_outside_angular_region_stays_full(self):

        grid = self.grid()

        region = ReactiveShadowRegion(
            theta_center_rad=0.0,
            theta_span_rad=0.1,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        result = (
            single_region_illumination_map(
                grid,
                region,
            )
        )

        far_angle_index = int(
            np.argmax(
                np.abs(
                    grid.theta_centers_rad
                )
            )
        )

        self.assertTrue(
            np.all(
                result[
                    far_angle_index,
                    :
                ]
                ==
                1.0
            )
        )

    def test_single_region_map_has_dimming(self):

        grid = self.grid()

        region = ReactiveShadowRegion(
            theta_center_rad=0.0,
            theta_span_rad=0.5,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        result = (
            single_region_illumination_map(
                grid,
                region,
            )
        )

        self.assertLess(
            float(
                np.min(
                    result
                )
            ),
            1.0,
        )

        self.assertGreaterEqual(
            float(
                np.min(
                    result
                )
            ),
            0.0,
        )

        self.assertLessEqual(
            float(
                np.max(
                    result
                )
            ),
            1.0,
        )

    def test_multi_actor_combination_is_elementwise_minimum(self):

        grid = self.grid()

        first = ReactiveShadowRegion(
            theta_center_rad=-0.2,
            theta_span_rad=0.4,
            r_shadow_end_m=20.0,
            r_transition_end_m=40.0,
        )

        second = ReactiveShadowRegion(
            theta_center_rad=0.2,
            theta_span_rad=0.4,
            r_shadow_end_m=30.0,
            r_transition_end_m=50.0,
        )

        first_map = (
            single_region_illumination_map(
                grid,
                first,
            )
        )

        second_map = (
            single_region_illumination_map(
                grid,
                second,
            )
        )

        combined = (
            reactive_adb_map(
                grid,
                (
                    first,
                    second,
                ),
            )
        )

        expected = np.minimum(
            first_map,
            second_map,
        )

        np.testing.assert_allclose(
            combined,
            expected,
            atol=0.0,
        )

    def test_multi_actor_combination_order_invariant(self):

        grid = self.grid()

        first = ReactiveShadowRegion(
            theta_center_rad=-0.3,
            theta_span_rad=0.3,
            r_shadow_end_m=15.0,
            r_transition_end_m=35.0,
        )

        second = ReactiveShadowRegion(
            theta_center_rad=0.3,
            theta_span_rad=0.3,
            r_shadow_end_m=25.0,
            r_transition_end_m=45.0,
        )

        a = reactive_adb_map(
            grid,
            (
                first,
                second,
            ),
        )

        b = reactive_adb_map(
            grid,
            (
                second,
                first,
            ),
        )

        np.testing.assert_array_equal(
            a,
            b,
        )

    def test_empty_reactive_regions_equal_static_baseline(self):

        grid = self.grid()

        reactive = reactive_adb_map(
            grid,
            (),
        )

        static = static_adb_map(
            grid
        )

        np.testing.assert_array_equal(
            reactive,
            static,
        )

    def test_combiner_rejects_out_of_range_intensity(self):

        with self.assertRaises(
            ValueError
        ):
            combine_illumination_maps_minimum(
                (
                    np.asarray(
                        [
                            [1.1]
                        ]
                    ),
                )
            )

    def test_reactive_region_rejects_future_semantics(self):

        with self.assertRaises(
            ValueError
        ):
            ReactiveShadowRegion(
                theta_center_rad=0.0,
                theta_span_rad=0.2,
                r_shadow_end_m=20.0,
                r_transition_end_m=40.0,
                source_semantics="future_gt",
            )

    def test_reactive_api_has_no_predictive_or_communication_input(self):

        signature = inspect.signature(
            reactive_adb_map
        )

        names = set(
            signature.parameters
        )

        forbidden = {
            "posterior",
            "covariance",
            "future_gt",
            "codebook",
            "beam_id",
            "receiver",
        }

        self.assertTrue(
            names.isdisjoint(
                forbidden
            )
        )

    def test_region_has_no_class_policy_field(self):

        fields = set(
            ReactiveShadowRegion
            .__dataclass_fields__
        )

        forbidden = {
            "actor_class",
            "class_margin",
            "confidence_threshold",
            "closing_speed",
            "prediction_horizon",
        }

        self.assertTrue(
            fields.isdisjoint(
                forbidden
            )
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
    # E. Controlled compile
    # ========================================================

    print()
    print(
        "===== E. CONTROLLED COMPILE ====="
    )

    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(
                ILLUMINATION_MODULE
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
            "Block6.2 Part1 compile failed:\n"
            + tail_text(
                compile_process.stdout
            )
        ),
    )

    print(
        "illumination/tests compile = PASS"
    )

    # Import only after compile.
    from iscai_stage6.adb.grid import (
        IlluminationGridSpec,
    )

    from iscai_stage6.adb.illumination import (
        ReactiveShadowRegion,
        part_a_radial_profile,
        reactive_adb_map,
        static_adb_map,
    )

    # ========================================================
    # F. Exact mathematical smoke
    # ========================================================

    print()
    print(
        "===== F. PART-A EQUATION SMOKE ====="
    )

    r_min = 20.0
    r_max = 40.0

    samples = (
        0.0,
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        60.0,
    )

    actual = [
        float(
            part_a_radial_profile(
                value,
                r_shadow_end_m=r_min,
                r_transition_end_m=r_max,
                intensity_floor=0.0,
            )
        )
        for value in samples
    ]

    expected = []

    for value in samples:

        if value <= r_min:
            target = 0.0

        elif value < r_max:

            u = (
                value - r_min
            ) / (
                r_max - r_min
            )

            target = (
                0.5
                *
                (
                    1.0
                    -
                    math.cos(
                        math.pi * u
                    )
                )
            )

        else:
            target = 1.0

        expected.append(
            target
        )

    maximum_equation_error = max(
        abs(
            a - b
        )
        for a, b in zip(
            actual,
            expected,
        )
    )

    require(
        maximum_equation_error
        <=
        1.0e-14,
        (
            "Part-A raised-cosine equation "
            "mismatch: "
            f"{maximum_equation_error}"
        ),
    )

    print(
        "floor=0 Part-A formula = EXACT PASS"
    )

    print(
        "max equation error     =",
        maximum_equation_error,
    )

    # Diagnostic-only grid. Not a scientific numerical freeze.
    grid = IlluminationGridSpec(
        theta_min_rad=-math.pi,
        theta_max_rad=math.pi,
        range_min_m=0.0,
        range_max_m=100.0,
        n_theta=72,
        n_range=100,
    )

    static_map = static_adb_map(
        grid
    )

    region_a = ReactiveShadowRegion(
        theta_center_rad=-0.15,
        theta_span_rad=0.4,
        r_shadow_end_m=20.0,
        r_transition_end_m=40.0,
        intensity_floor=0.0,
    )

    region_b = ReactiveShadowRegion(
        theta_center_rad=0.15,
        theta_span_rad=0.4,
        r_shadow_end_m=30.0,
        r_transition_end_m=50.0,
        intensity_floor=0.0,
    )

    reactive_map = reactive_adb_map(
        grid,
        (
            region_a,
            region_b,
        ),
    )

    require(
        static_map.shape
        ==
        reactive_map.shape
        ==
        grid.shape,
        "ADB map shape mismatch.",
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

    require(
        float(
            reactive_map.min()
        )
        <
        1.0,
        (
            "Reactive smoke generated "
            "no shadow attenuation."
        ),
    )

    print(
        "static ADB map        = PASS"
    )

    print(
        "reactive ADB map      = PASS"
    )

    print(
        "normalized intensity  = [0,1] PASS"
    )

    print(
        "multi-region minimum  = ACTIVE"
    )

    print(
        "diagnostic grid only  = YES"
    )

    # ========================================================
    # G. Static API/causality inspection
    # ========================================================

    print()
    print(
        "===== G. REACTIVE BASELINE SCOPE GATE ====="
    )

    reactive_signature = str(
        inspect.signature(
            reactive_adb_map
        )
    )

    forbidden_words = (
        "future",
        "posterior",
        "covariance",
        "codebook",
        "beam_id",
    )

    require(
        not any(
            word
            in
            reactive_signature.lower()
            for word in forbidden_words
        ),
        (
            "Reactive ADB kernel API exposes "
            "predictive/communication input."
        ),
    )

    print(
        "current deterministic only = PASS"
    )

    print(
        "predictive posterior input  = NO"
    )

    print(
        "future GT input             = NO"
    )

    print(
        "communication codebook      = NO"
    )

    print(
        "class-aware policy          = DEFERRED"
    )

    # ========================================================
    # H. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== H. FULL STAGE6 REGRESSION ====="
    )

    rc, output = run_stage6_tests()

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
            print(stripped)

    require(
        rc == 0,
        (
            "Stage6 regression failed. "
            f"Full log: {REGRESSION_LOG}\n"
            + tail_text(
                output,
                80,
            )
        ),
    )

    # ========================================================
    # I. Immutability / storage
    # ========================================================

    print()
    print(
        "===== I. IMMUTABILITY / STORAGE ====="
    )

    protected_after = {
        str(path):
            sha256_file(path)
        for path in protected
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
        "Block6.1 closure       = UNCHANGED"
    )

    print(
        "Part-A frozen evidence = UNCHANGED"
    )

    print(
        "Stage5 closure         = UNCHANGED"
    )

    print(
        "free GiB               =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve        = PASS"
    )

    # ========================================================
    # J. Part1 report
    # ========================================================

    print()
    print(
        "===== J. BLOCK6.2 PART1 REPORT ====="
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.2",

        "part":
            "1/2",

        "status":
            "PASS_STATIC_REACTIVE_KERNEL",

        "prerequisites": {
            "Block6_1":
                "PASS_COMPLETE",

            "Block6_1_closure_sha256":
                EXPECTED[
                    "block61_closure"
                ],

            "Part_A_source_audit":
                "PASS",

            "Part_A_adapter_contract":
                "FROZEN",
        },

        "implementation": {
            "illumination_module":
                str(
                    ILLUMINATION_MODULE
                ),

            "illumination_module_sha256":
                sha256_file(
                    ILLUMINATION_MODULE
                ),

            "static_ADB":
                True,

            "reactive_L_theta_r_kernel":
                True,

            "raised_cosine":
                True,

            "angular_shadow_zone":
                True,

            "radial_thresholds":
                True,

            "minimum_floor_support":
                True,

            "multi_actor_rule":
                "elementwise_minimum",

            "normalized_intensity":
                True,
        },

        "equation_validation": {
            "paper_floor":
                0.0,

            "sample_ranges_m":
                list(
                    samples
                ),

            "maximum_absolute_error":
                maximum_equation_error,

            "pass":
                True,
        },

        "scope": {
            "original_reactive_kernel":
                "IMPLEMENTED",

            "exact_Part_A_notebook_parameter_binding":
                "PENDING_BLOCK6.2_PART2",

            "real_scene_reactive_execution":
                "PENDING_BLOCK6.2_PART2",

            "predictive_ADB":
                False,

            "probabilistic_occupancy":
                False,

            "class_aware_policy":
                False,

            "temporal_smoothing":
                False,

            "actuation_rate_limits":
                False,

            "communication_beam_reused":
                False,
        },

        "grid": {
            "final_scientific_grid_frozen":
                False,

            "diagnostic_grid_used_for_smoke":
                True,

            "final_freeze":
                "BLOCK6.8_PRE_FORMAL",
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
            "dataset_access":
                False,

            "trajectory_inference":
                False,

            "training":
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
                "Block6.2 Part2: exact frozen "
                "Part-A notebook parameter/source "
                "binding plus real causal reactive "
                "ADB smoke"
            ),
    }

    tmp = REPORT.with_suffix(
        REPORT.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(
            report
        )
    )

    os.replace(
        tmp,
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
        "BLOCK 6.2 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "Block6.1 prerequisite     = EXACT PASS"
    )

    print(
        "static ADB baseline       = IMPLEMENTED PASS"
    )

    print(
        "Part-A L(theta,r) kernel  = IMPLEMENTED PASS"
    )

    print(
        "floor=0 paper equation    = EXACT PASS"
    )

    print(
        "radial thresholds         = PASS"
    )

    print(
        "angular shadow zones      = PASS"
    )

    print(
        "raised-cosine transition  = PASS"
    )

    print(
        "minimum-floor support     = PASS"
    )

    print(
        "multi-actor combination   = ELEMENTWISE MIN PASS"
    )

    print(
        "reactive future input     = NO"
    )

    print(
        "communication codebook    = NOT REUSED"
    )

    print(
        "exact Part-A parameters   = DEFERRED TO PART2"
    )

    print(
        "predictive/class-aware    = NOT STARTED"
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
        "STATUS = PASS_STATIC_REACTIVE_KERNEL"
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
        "BLOCK 6.2 PART 1/2 = BLOCKED"
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
        "Stage1–5 modified     = NO"
    )

    print(
        "Block6.1 modified     = NO"
    )

    print(
        "dataset access        = NO"
    )

    print(
        "training/inference    = NO"
    )

    print(
        "formal evaluation     = NO"
    )

    print(
        "parameter tuning      = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

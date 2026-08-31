from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)

METRICS = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

INTERFACE = (
    S6
    / "src/iscai_stage6/adb/"
      "baseline_metric_contract.py"
)

OLD_TEST = (
    S6
    / "tests/"
      "test_block68_metric_semantics.py"
)

NEW_TEST = (
    S6
    / "tests/"
      "test_block68_metric_reconciliation.py"
)

AUTHORITY_AUDIT = (
    S6
    / "reports/"
      "block68_part2_metric_authority_audit.json"
)

OLD_PART1 = (
    S6
    / "reports/"
      "block68_part1_metric_freeze_gate.json"
)

BLOCK67 = (
    S6
    / "reports/"
      "block67_final_closure.json"
)

CLASS_POLICY = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

MARGIN_BINDING = (
    S6
    / "src/iscai_stage6/adb/"
      "class_margin_binding.py"
)

REPORT = (
    S6
    / "reports/"
      "block68_preoutcome_metric_reconciliation.json"
)

BACKUP_DIR = (
    S6
    / "artifacts/block68/"
      "preoutcome_metric_reconciliation"
)

EXPECTED_PREREG_SHA = (
    "5704494a89b4b3c7a3c19b0df8176c55"
    "0c00795ed17cf2906c343f340461429e"
)

EXPECTED_OLD_PROTOCOL_SHA = (
    "c12b95c24ba64f48ac8dc3890fa55061"
    "3ea597875f6a5a759762b88629623ac0"
)

EXPECTED_OLD_RUNTIME_SHA = (
    "fdca66a1a9af74d24baaafe4e17d12ac"
    "db2db538e037503aba6ee04684eac362"
)

EXPECTED_BLOCK67_SHA = (
    "67d168ed7678164b8d85885e6e81f41f"
    "7206bd8f754e13ec5cc28c3a65aa20e7"
)

EXPECTED_CLASS_POLICY_SHA = (
    "b998f468b98c2770c48c84a2c0aaaa21"
    "77c2d84b8fc838e80fef9b9da1ebc14b"
)

EXPECTED_PRE_REPAIR_TESTS = 196

EXPECTED_NEW_RECON_TESTS = 6

MIN_FREE_GIB = 250.0


# ============================================================
# Helpers
# ============================================================

def sha256_bytes(
    data: bytes,
) -> str:

    return sha256(
        data
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:

    return sha256_bytes(
        path.read_bytes()
    )


def load_json(
    path: Path,
):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


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


def canonical_json_bytes(
    payload,
) -> bytes:

    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_write(
    path: Path,
    data: bytes,
):

    temporary = (
        path.with_suffix(
            path.suffix
            +
            ".tmp"
        )
    )

    temporary.write_bytes(
        data
    )

    os.replace(
        temporary,
        path,
    )


def run_tests(
    pattern: str,
):

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            pattern,
        ],
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    count = None

    for line in (
        process.stdout.splitlines()
    ):

        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:

            count = int(
                match.group(1)
            )

    return (
        process.returncode,
        count,
        process.stdout,
    )


def tail(
    text: str,
    n: int = 100,
):

    return "\n".join(
        text.splitlines()[
            -n:
        ]
    )


def backup_path(
    path: Path,
) -> Path:

    relative = (
        path.relative_to(
            S6
        )
    )

    safe_name = "__".join(
        relative.parts
    )

    return (
        BACKUP_DIR
        /
        safe_name
    )


def restore(
    originals,
):

    for path, data in (
        originals.items()
    ):

        if data is None:

            if path.exists():
                path.unlink()

        else:

            atomic_write(
                path,
                data,
            )


def assert_file_unchanged(
    path: Path,
    expected_sha: str,
):

    require(
        path.is_file(),
        (
            "Protected file missing: "
            f"{path}"
        ),
    )

    require(
        sha256_file(
            path
        )
        ==
        expected_sha,
        (
            "Protected file changed: "
            f"{path}"
        ),
    )


# ============================================================
# Reconciled numerical metric runtime
# ============================================================

METRIC_SOURCE = r'''from __future__ import annotations

from dataclasses import dataclass
import numpy as np


INTENSITY_MIN = 0.0
INTENSITY_MAX = 1.0


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    mean_s: float
    median_s: float
    p95_s: float
    max_s: float


def _bool_array(
    value,
    *,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        value
    )

    if (
        array.dtype
        !=
        np.bool_
    ):
        raise TypeError(
            f"{name} must have boolean dtype"
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    return array


def _intensity_array(
    value,
    *,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if (
        np.any(
            array
            <
            INTENSITY_MIN
        )
        or
        np.any(
            array
            >
            INTENSITY_MAX
        )
    ):
        raise ValueError(
            f"{name} must lie in [0,1]"
        )

    return array


def _same_shape(
    *arrays,
):

    shapes = {
        tuple(
            np.asarray(
                item
            ).shape
        )
        for item in arrays
    }

    if len(
        shapes
    ) != 1:
        raise ValueError(
            "All arrays must have identical shape"
        )


# ============================================================
# Frozen binary support definitions
# ============================================================

def binary_dim_support(
    intensity,
) -> np.ndarray:
    """
    Frozen Block6.6 definition:

        D_tau = [I_final_tau < 1.0]
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    return (
        light
        <
        1.0
    )


# ============================================================
# 1. Mask IoU
# ============================================================

def mask_iou(
    candidate_mask,
    oracle_mask,
) -> float:
    """
    |D ∩ O| / |D ∪ O|

    Empty union = 1.
    """

    candidate = _bool_array(
        candidate_mask,
        name="candidate_mask",
    )

    oracle = _bool_array(
        oracle_mask,
        name="oracle_mask",
    )

    _same_shape(
        candidate,
        oracle,
    )

    intersection = int(
        np.count_nonzero(
            candidate
            &
            oracle
        )
    )

    union = int(
        np.count_nonzero(
            candidate
            |
            oracle
        )
    )

    if union == 0:
        return 1.0

    return float(
        intersection
        /
        union
    )


# ============================================================
# 2. Vehicle shadow-zone violation
# ============================================================

def vehicle_shadow_zone_violation(
    candidate_mask,
    vehicle_oracle_region,
) -> float:
    """
    Frozen Block6.6 definition:

        |O_vehicle \ D| / |O_vehicle|

    No vehicle cells:
        NA -> NaN
    """

    candidate = _bool_array(
        candidate_mask,
        name="candidate_mask",
    )

    vehicle = _bool_array(
        vehicle_oracle_region,
        name="vehicle_oracle_region",
    )

    _same_shape(
        candidate,
        vehicle,
    )

    denominator = int(
        np.count_nonzero(
            vehicle
        )
    )

    if denominator == 0:
        return float(
            "nan"
        )

    missed = int(
        np.count_nonzero(
            vehicle
            &
            ~candidate
        )
    )

    return float(
        missed
        /
        denominator
    )


# ============================================================
# 3. Glare-risk exposure
# ============================================================

def glare_risk_exposure(
    intensity,
    vehicle_surrogate_reference_region,
) -> float:
    """
    Mean final normalized intensity over the constructed
    vehicle front-upper/rear-upper surrogate reference cells.

    Lower is better.

    Empty evaluator support is reported as NaN.
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    region = _bool_array(
        vehicle_surrogate_reference_region,
        name=(
            "vehicle_surrogate_reference_region"
        ),
    )

    _same_shape(
        light,
        region,
    )

    if not np.any(
        region
    ):
        return float(
            "nan"
        )

    return float(
        np.mean(
            light[
                region
            ]
        )
    )


# ============================================================
# 4. Over-masking area
# ============================================================

def over_masking_area(
    candidate_mask,
    oracle_mask,
) -> float:
    """
    Frozen Block6.6 definition:

        |D \ O|
        --------------------
        |full actuator grid|

    Lower is better.
    """

    candidate = _bool_array(
        candidate_mask,
        name="candidate_mask",
    )

    oracle = _bool_array(
        oracle_mask,
        name="oracle_mask",
    )

    _same_shape(
        candidate,
        oracle,
    )

    unnecessary = (
        candidate
        &
        ~oracle
    )

    return float(
        np.count_nonzero(
            unnecessary
        )
        /
        candidate.size
    )


# ============================================================
# 5. Road illumination retention
# ============================================================

def road_illumination_retention(
    intensity,
    road_roi,
) -> float:
    """
    Frozen Block6.6 definition:

        mean(I_final over frozen road ROI)

    Higher is better.
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    road = _bool_array(
        road_roi,
        name="road_roi",
    )

    _same_shape(
        light,
        road,
    )

    if not np.any(
        road
    ):
        raise ValueError(
            "road_roi must contain at least one cell"
        )

    return float(
        np.mean(
            light[
                road
            ]
        )
    )


# ============================================================
# 6. Pedestrian visibility
# ============================================================

def pedestrian_visibility_proxy(
    intensity,
    pedestrian_future_full_box_region,
) -> float:
    """
    Frozen Block6.6 definition:

        mean(
            I_final over future-GT pedestrian
            projected full-box footprint cells
        )

    Evaluator only.
    Higher is better.

    Empty support:
        NA -> NaN
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    region = _bool_array(
        pedestrian_future_full_box_region,
        name=(
            "pedestrian_future_full_box_region"
        ),
    )

    _same_shape(
        light,
        region,
    )

    if not np.any(
        region
    ):
        return float(
            "nan"
        )

    return float(
        np.mean(
            light[
                region
            ]
        )
    )


# ============================================================
# 7. Cyclist visibility
# ============================================================

def cyclist_visibility_proxy(
    intensity,
    cyclist_future_full_box_region,
) -> float:
    """
    Frozen Block6.6 definition:

        mean(
            I_final over future-GT cyclist
            projected full-box footprint cells
        )

    Evaluator only.
    Higher is better.

    Empty support:
        NA -> NaN
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    region = _bool_array(
        cyclist_future_full_box_region,
        name=(
            "cyclist_future_full_box_region"
        ),
    )

    _same_shape(
        light,
        region,
    )

    if not np.any(
        region
    ):
        return float(
            "nan"
        )

    return float(
        np.mean(
            light[
                region
            ]
        )
    )


# ============================================================
# 8. False dimming
# ============================================================

def false_dimming(
    intensity,
    oracle_mask,
) -> float:
    """
    Frozen Block6.6 definition:

        D = [I_final < 1]

        |D \ O|
        -------
          |D|

    If |D| = 0:
        0
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    oracle = _bool_array(
        oracle_mask,
        name="oracle_mask",
    )

    _same_shape(
        light,
        oracle,
    )

    dim = binary_dim_support(
        light
    )

    dim_count = int(
        np.count_nonzero(
            dim
        )
    )

    if dim_count == 0:
        return 0.0

    unnecessary = int(
        np.count_nonzero(
            dim
            &
            ~oracle
        )
    )

    return float(
        unnecessary
        /
        dim_count
    )


# ============================================================
# 9. Temporal smoothness PDF-facing label
#
# Frozen implementation = mean absolute CHANGE RATE.
# ============================================================

def temporal_smoothness(
    intensity_sequence,
    *,
    delta_t_s,
) -> float:
    """
    Historical PDF-facing metric label retained.

    Frozen Block6.6 reported quantity:

        mean_h(
            mean_grid(
                |I_h - I_prev|
            )
            /
            delta_t_h
        )

    This is mean absolute illumination CHANGE RATE.

    Lower is better.
    """

    light = _intensity_array(
        intensity_sequence,
        name="intensity_sequence",
    )

    if light.ndim < 2:
        raise ValueError(
            "intensity_sequence must have "
            "time + grid dimensions"
        )

    transitions = (
        light.shape[0]
        -
        1
    )

    if transitions < 1:
        raise ValueError(
            "at least two frames are required"
        )

    dt = np.asarray(
        delta_t_s,
        dtype=np.float64,
    )

    if dt.ndim == 0:

        dt = np.full(
            transitions,
            float(
                dt
            ),
            dtype=np.float64,
        )

    if (
        dt.ndim != 1
        or
        dt.size != transitions
    ):
        raise ValueError(
            "delta_t_s must be scalar or "
            "one value per transition"
        )

    if (
        not np.all(
            np.isfinite(
                dt
            )
        )
        or
        np.any(
            dt <= 0.0
        )
    ):
        raise ValueError(
            "delta_t_s values must be "
            "finite and > 0"
        )

    absolute_change = np.abs(
        np.diff(
            light,
            axis=0,
        )
    )

    per_transition = np.mean(
        absolute_change.reshape(
            transitions,
            -1,
        ),
        axis=1,
    )

    return float(
        np.mean(
            per_transition
            /
            dt
        )
    )


# ============================================================
# 10. Flicker/change rate
# ============================================================

def flicker_change_rate(
    intensity_sequence,
) -> float:
    """
    Frozen Block6.6 definition:

        mean_h(
            count(I_h != I_prev)
            /
            grid_cell_count
        )

    Exact deterministic inequality.

    No additional 0.10 threshold is introduced.

    Lower is better.
    """

    light = _intensity_array(
        intensity_sequence,
        name="intensity_sequence",
    )

    if light.ndim < 2:
        raise ValueError(
            "intensity_sequence must have "
            "time + grid dimensions"
        )

    if light.shape[0] < 2:
        raise ValueError(
            "at least two frames are required"
        )

    changed = np.not_equal(
        light[1:],
        light[:-1],
    )

    per_transition = np.mean(
        changed.reshape(
            changed.shape[0],
            -1,
        ),
        axis=1,
    )

    return float(
        np.mean(
            per_transition
        )
    )


# ============================================================
# 11. Energy consumption proxy
# ============================================================

def normalized_energy_consumption(
    intensity_sequence,
) -> float:
    """
    Frozen Block6.6 definition:

        mean over horizon/grid of I_final

    This is a normalized emitted-light energy proxy.
    It is NOT a joule claim.
    """

    light = _intensity_array(
        intensity_sequence,
        name="intensity_sequence",
    )

    return float(
        np.mean(
            light
        )
    )


# ============================================================
# 12. Actuation latency
# ============================================================

def summarize_actuation_latency_s(
    latency_samples_s,
) -> LatencySummary:
    """
    Frozen Block6.6 latency semantics:

    wall-clock runtime from loaded P_occ/current geometry
    to final four-horizon illumination schedule.

    Units:
        seconds

    Reporting only.
    Not used for policy selection.
    """

    values = np.asarray(
        latency_samples_s,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or
        values.size == 0
    ):
        raise ValueError(
            "latency_samples_s must be "
            "a non-empty 1-D sequence"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "latency samples must be finite"
        )

    if np.any(
        values < 0.0
    ):
        raise ValueError(
            "latency samples must be >= 0"
        )

    return LatencySummary(
        sample_count=int(
            values.size
        ),

        mean_s=float(
            np.mean(
                values
            )
        ),

        median_s=float(
            np.median(
                values
            )
        ),

        p95_s=float(
            np.percentile(
                values,
                95.0,
            )
        ),

        max_s=float(
            np.max(
                values
            )
        ),
    )


# ============================================================
# Auxiliary preregistered diagnostic
# ============================================================

def rate_limit_safety_override_fraction(
    override_transition_mask,
) -> float:
    """
    Auxiliary Block6.6 diagnostic:

        override cells
        -------------------------------
        total actuator-cell transitions

    This is NOT one of the 12 mandatory PDF metric labels.
    """

    override = _bool_array(
        override_transition_mask,
        name="override_transition_mask",
    )

    return float(
        np.mean(
            override
        )
    )
'''


# ============================================================
# Reconciled 15 metric tests
# ============================================================

TEST_SOURCE = r'''from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.metric_semantics import (
    binary_dim_support,
    cyclist_visibility_proxy,
    false_dimming,
    flicker_change_rate,
    glare_risk_exposure,
    mask_iou,
    normalized_energy_consumption,
    over_masking_area,
    pedestrian_visibility_proxy,
    road_illumination_retention,
    summarize_actuation_latency_s,
    temporal_smoothness,
    vehicle_shadow_zone_violation,
)


class Block68MetricSemanticsTest(
    unittest.TestCase
):

    def test_01_binary_dim_support(
        self,
    ):

        value = np.asarray(
            [
                1.0,
                0.9,
                0.0,
            ]
        )

        np.testing.assert_array_equal(
            binary_dim_support(
                value
            ),
            np.asarray(
                [
                    False,
                    True,
                    True,
                ]
            ),
        )

    def test_02_mask_iou_empty(
        self,
    ):

        empty = np.zeros(
            4,
            dtype=bool,
        )

        self.assertEqual(
            mask_iou(
                empty,
                empty,
            ),
            1.0,
        )

    def test_03_vehicle_violation(
        self,
    ):

        dim = np.asarray(
            [
                True,
                False,
                False,
            ],
            dtype=bool,
        )

        oracle_vehicle = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            vehicle_shadow_zone_violation(
                dim,
                oracle_vehicle,
            ),
            0.5,
        )

    def test_04_vehicle_empty_is_na(
        self,
    ):

        dim = np.zeros(
            3,
            dtype=bool,
        )

        oracle_vehicle = np.zeros(
            3,
            dtype=bool,
        )

        self.assertTrue(
            math.isnan(
                vehicle_shadow_zone_violation(
                    dim,
                    oracle_vehicle,
                )
            )
        )

    def test_05_glare_reference_region(
        self,
    ):

        intensity = np.asarray(
            [
                0.2,
                0.4,
                1.0,
            ]
        )

        region = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertAlmostEqual(
            glare_risk_exposure(
                intensity,
                region,
            ),
            0.3,
            places=12,
        )

    def test_06_overmask_full_grid_denominator(
        self,
    ):

        dim = np.asarray(
            [
                True,
                True,
                False,
                False,
            ],
            dtype=bool,
        )

        oracle = np.asarray(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            over_masking_area(
                dim,
                oracle,
            ),
            0.25,
        )

    def test_07_road_retention(
        self,
    ):

        intensity = np.asarray(
            [
                1.0,
                0.5,
                0.0,
            ]
        )

        road = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            road_illumination_retention(
                intensity,
                road,
            ),
            0.75,
        )

    def test_08_pedestrian_visibility(
        self,
    ):

        intensity = np.asarray(
            [
                0.2,
                0.8,
                1.0,
            ]
        )

        region = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            pedestrian_visibility_proxy(
                intensity,
                region,
            ),
            0.5,
        )

    def test_09_cyclist_visibility(
        self,
    ):

        intensity = np.asarray(
            [
                0.4,
                0.6,
                1.0,
            ]
        )

        region = np.asarray(
            [
                True,
                True,
                False,
            ],
            dtype=bool,
        )

        self.assertEqual(
            cyclist_visibility_proxy(
                intensity,
                region,
            ),
            0.5,
        )

    def test_10_false_dimming_support_ratio(
        self,
    ):

        intensity = np.asarray(
            [
                0.0,
                0.5,
                1.0,
                1.0,
            ]
        )

        oracle = np.asarray(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=bool,
        )

        # D = [1,1,0,0]
        # D \ O = one cell
        # => 1/2
        self.assertEqual(
            false_dimming(
                intensity,
                oracle,
            ),
            0.5,
        )

    def test_11_false_dimming_no_dim_is_zero(
        self,
    ):

        intensity = np.ones(
            4
        )

        oracle = np.zeros(
            4,
            dtype=bool,
        )

        self.assertEqual(
            false_dimming(
                intensity,
                oracle,
            ),
            0.0,
        )

    def test_12_temporal_change_rate(
        self,
    ):

        sequence = np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    0.2,
                    0.0,
                ],
            ]
        )

        # Mean grid absolute change = 0.1.
        # dt = 0.5 s.
        # change rate = 0.2 / s.
        self.assertAlmostEqual(
            temporal_smoothness(
                sequence,
                delta_t_s=0.5,
            ),
            0.2,
            places=12,
        )

    def test_13_flicker_exact_inequality(
        self,
    ):

        sequence = np.asarray(
            [
                [
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0e-12,
                ],
            ]
        )

        self.assertEqual(
            flicker_change_rate(
                sequence
            ),
            0.5,
        )

    def test_14_energy_full_grid_mean(
        self,
    ):

        sequence = np.asarray(
            [
                [
                    0.0,
                    1.0,
                ],
                [
                    1.0,
                    0.0,
                ],
            ]
        )

        self.assertEqual(
            normalized_energy_consumption(
                sequence
            ),
            0.5,
        )

    def test_15_latency_seconds(
        self,
    ):

        summary = (
            summarize_actuation_latency_s(
                [
                    0.001,
                    0.002,
                    0.003,
                    0.004,
                ]
            )
        )

        self.assertEqual(
            summary.sample_count,
            4,
        )

        self.assertAlmostEqual(
            summary.mean_s,
            0.0025,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
'''


# ============================================================
# Additional authority/interface tests
# ============================================================

RECON_TEST_SOURCE = r'''from __future__ import annotations

import json
from pathlib import Path
import unittest

from iscai_stage6.adb.baseline_metric_contract import (
    ADBMetric,
    required_metric_interfaces,
)


ROOT = Path(
    "/home/agni/waymo"
)

S6 = (
    ROOT
    / "iscai_stage6"
)

PROTOCOL = (
    S6
    / "configs/"
      "stage6_metric_freeze_protocol.json"
)


class Block68MetricReconciliationTest(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls,
    ):

        cls.protocol = json.loads(
            PROTOCOL.read_text(
                encoding="utf-8"
            )
        )

    def test_01_authority(
        self,
    ):

        self.assertEqual(
            self.protocol[
                "authority"
            ][
                "binding_source"
            ],
            "Block6.6_preoutcome_preregistration",
        )

    def test_02_temporal_lower_is_better(
        self,
    ):

        self.assertEqual(
            self.protocol[
                "metrics"
            ][
                "temporal_smoothness"
            ][
                "direction"
            ],
            "lower_is_better",
        )

    def test_03_no_flicker_threshold(
        self,
    ):

        flicker = (
            self.protocol[
                "metrics"
            ][
                "flicker_change_rate"
            ]
        )

        self.assertNotIn(
            "delta_threshold",
            flicker,
        )

    def test_04_overmask_not_road_roi_normalized(
        self,
    ):

        interfaces = {
            metric.name:
                metric
            for metric in (
                required_metric_interfaces()
            )
        }

        self.assertFalse(
            interfaces[
                ADBMetric.OVER_MASKING_AREA
            ].requires_road_ROI
        )

    def test_05_false_dimming_not_road_roi_normalized(
        self,
    ):

        interfaces = {
            metric.name:
                metric
            for metric in (
                required_metric_interfaces()
            )
        }

        self.assertFalse(
            interfaces[
                ADBMetric.FALSE_DIMMING
            ].requires_road_ROI
        )

    def test_06_energy_not_road_roi_normalized(
        self,
    ):

        interfaces = {
            metric.name:
                metric
            for metric in (
                required_metric_interfaces()
            )
        }

        self.assertFalse(
            interfaces[
                ADBMetric.ENERGY_CONSUMPTION
            ].requires_road_ROI
        )


if __name__ == "__main__":
    unittest.main()
'''


# ============================================================
# Reconciled protocol
# ============================================================

def reconciled_protocol():

    return {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.8",

        "status":
            (
                "FROZEN_METRIC_SEMANTICS_"
                "RECONCILED_TO_BLOCK66_PREREGISTRATION"
            ),

        "source_semantics": {
            "pdf_specifies_metric_names":
                True,

            "pdf_specifies_unique_formula_for_every_metric":
                False,

            "earliest_internal_preregistration_is_binding":
                True,

            "reconciliation_performed_before_development_outcome_read":
                True,

            "reconciliation_performed_before_formal_outcome_read":
                True,
        },

        "authority": {
            "binding_source":
                "Block6.6_preoutcome_preregistration",

            "binding_path":
                str(
                    PREREG
                ),

            "binding_sha256":
                EXPECTED_PREREG_SHA,

            "development_outcomes_seen":
                False,

            "formal_outcomes_seen":
                False,

            "silent_override":
                False,

            "reason":
                (
                    "Earliest explicit pre-outcome "
                    "metric preregistration is authoritative."
                ),
        },

        "mask_semantics": {
            "binary_dim_support":
                "D_tau = [I_final_tau < 1.0]",

            "oracle_dim_support":
                (
                    "O_tau = union of constructed "
                    "future actor shadow supports"
                ),

            "measured_ADB_ground_truth":
                False,

            "oracle_controller_input":
                False,
        },

        "metrics": {
            "mask_IoU_with_constructed_oracle_future_mask": {
                "direction":
                    "higher_is_better",

                "formula":
                    "|D∩O| / |D∪O|",

                "empty_union":
                    1.0,
            },

            "vehicle_shadow_zone_violation": {
                "direction":
                    "lower_is_better",

                "formula":
                    "|O_vehicle \\ D| / |O_vehicle|",

                "no_vehicle_cells":
                    "NA",
            },

            "glare_risk_exposure": {
                "direction":
                    "lower_is_better",

                "formula":
                    (
                        "mean final normalized intensity "
                        "over vehicle constructed "
                        "front-upper/rear-upper surrogate "
                        "reference cells"
                    ),

                "empty_support_clarification":
                    "NA",
            },

            "over_masking_area": {
                "direction":
                    "lower_is_better",

                "formula":
                    "|D \\ O| / |full actuator grid|",
            },

            "road_illumination_retention": {
                "direction":
                    "higher_is_better",

                "formula":
                    "mean(I_final over frozen road ROI)",
            },

            "pedestrian_visibility_proxy": {
                "direction":
                    "higher_is_better",

                "formula":
                    (
                        "mean(I_final over future GT "
                        "pedestrian projected full-box "
                        "footprint cells)"
                    ),

                "empty_support_clarification":
                    "NA",
            },

            "cyclist_visibility_proxy": {
                "direction":
                    "higher_is_better",

                "formula":
                    (
                        "mean(I_final over future GT "
                        "cyclist projected full-box "
                        "footprint cells)"
                    ),

                "empty_support_clarification":
                    "NA",
            },

            "false_dimming": {
                "direction":
                    "lower_is_better",

                "formula":
                    "|D \\ O| / |D|; 0 when |D|=0",
            },

            "temporal_smoothness": {
                "direction":
                    "lower_is_better",

                "formula":
                    (
                        "mean_h("
                        "mean_grid(|I_h-I_prev|)"
                        "/delta_t_h)"
                    ),

                "reported_metric":
                    "mean_absolute_change_rate",
            },

            "flicker_change_rate": {
                "direction":
                    "lower_is_better",

                "formula":
                    (
                        "mean_h("
                        "count(I_h != I_prev)"
                        "/grid_cell_count)"
                    ),

                "added_threshold":
                    False,
            },

            "energy_consumption": {
                "direction":
                    "reported_not_primary_safety_target",

                "formula":
                    "mean over horizon/grid of I_final",

                "reported_semantics":
                    (
                        "normalized emitted-light energy "
                        "proxy, not joules"
                    ),
            },

            "actuation_latency": {
                "formula":
                    (
                        "wall-clock runtime from loaded "
                        "P_occ/current geometry to final "
                        "four-horizon illumination schedule"
                    ),

                "units":
                    "seconds",

                "used_for_policy_selection":
                    False,

                "summary":
                    [
                        "mean_s",
                        "median_s",
                        "p95_s",
                        "max_s",
                    ],
            },
        },

        "auxiliary_preregistered_metric": {
            "rate_limit_safety_override_fraction": {
                "direction":
                    "lower_is_better",

                "formula":
                    (
                        "override cells / total "
                        "actuator-cell transitions"
                    ),

                "mandatory_PDF_metric":
                    False,
            },
        },

        "roi_contract": {
            "road_roi": {
                "used_by":
                    [
                        "road_illumination_retention"
                    ],

                "future_truth_used_for_controller":
                    False,
            },

            "vehicle_reference_region": {
                "role":
                    "evaluator_only",

                "future_truth_controller_input":
                    False,
            },

            "pedestrian_full_box_region": {
                "role":
                    "evaluator_only",

                "future_truth_controller_input":
                    False,
            },

            "cyclist_full_box_region": {
                "role":
                    "evaluator_only",

                "future_truth_controller_input":
                    False,
            },

            "oracle_mask": {
                "role":
                    "constructed_evaluator_reference",

                "measured_ground_truth":
                    False,

                "controller_input":
                    False,

                "parameter_tuning_input":
                    False,
            },
        },

        "primary_acceptance": {
            "comparison":
                (
                    "class_aware_predictive_ADB"
                    "_vs_original_reactive_ADB"
                ),

            "vehicle_shadow_zone_violation": {
                "requirement":
                    "strict_improvement",

                "formal_rule":
                    (
                        "predictive_mean "
                        "< reactive_mean"
                    ),
            },

            "over_masking_area": {
                "requirement":
                    "non_inferiority",

                "numeric_delta":
                    None,

                "delta_source":
                    "development_only_Block6.8_Part2",

                "formal_rule":
                    (
                        "predictive_minus_reactive "
                        "<= frozen_delta"
                    ),
            },

            "pedestrian_visibility_proxy": {
                "requirement":
                    "non_inferiority",

                "numeric_delta":
                    None,

                "delta_source":
                    "development_only_Block6.8_Part2",

                "formal_rule":
                    (
                        "predictive_minus_reactive "
                        ">= -frozen_delta"
                    ),
            },

            "cyclist_visibility_proxy": {
                "requirement":
                    "non_inferiority",

                "numeric_delta":
                    None,

                "delta_source":
                    "development_only_Block6.8_Part2",

                "formal_rule":
                    (
                        "predictive_minus_reactive "
                        ">= -frozen_delta"
                    ),
            },
        },

        "part2_bound_freeze_rules": {
            "development_partition_only":
                True,

            "formal_partition_read_before_freeze":
                False,

            "formal_outcome_used_for_bound_selection":
                False,

            "all_numeric_deltas_must_be_finite":
                True,

            "all_numeric_deltas_must_be_nonnegative":
                True,

            "single_freeze_before_formal":
                True,

            "post_formal_adjustment_forbidden":
                True,
        },

        "formal_evaluation_allowed":
            False,

        "next":
            (
                "development-only runtime metric "
                "evaluation and numeric "
                "non-inferiority bound freeze"
            ),
    }


# ============================================================
# Patch Block6.7 interface
# ============================================================

def patch_interface(
    source: str,
) -> str:
    """
    Reconcile only the three interface flags that became
    inconsistent with the earlier Block6.6 formulas:

      over_masking_area
      false_dimming
      energy_consumption

    Their formulas do NOT require road-ROI normalization.
    """

    lines = source.splitlines(
        keepends=True
    )

    targets = {
        "ADBMetric.OVER_MASKING_AREA",
        "ADBMetric.FALSE_DIMMING",
        "ADBMetric.ENERGY_CONSUMPTION",
    }

    current_target = None
    patched = set()

    output = []

    for line in lines:

        detected = None

        for target in targets:

            if target in line:

                detected = target
                break

        if detected is not None:

            current_target = (
                detected
            )

        if (
            current_target
            is not None
            and
            "requires_road_ROI=True"
            in line
        ):

            line = line.replace(
                "requires_road_ROI=True",
                "requires_road_ROI=False",
                1,
            )

            patched.add(
                current_target
            )

            current_target = None

        output.append(
            line
        )

    require(
        patched
        ==
        targets,
        (
            "Could not patch exactly all three "
            "Block6.7 metric-interface ROI flags. "
            f"Patched={sorted(patched)}"
        ),
    )

    return "".join(
        output
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.8"
    )
    print(
        "CONTROLLED PRE-OUTCOME METRIC RECONCILIATION"
    )
    print(
        "============================================================"
    )

    required = (
        PREREG,
        PROTOCOL,
        METRICS,
        INTERFACE,
        OLD_TEST,
        AUTHORITY_AUDIT,
        OLD_PART1,
        BLOCK67,
        CLASS_POLICY,
        MARGIN_BINDING,
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
            "Missing reconciliation dependency: "
            +
            ", ".join(
                missing
            )
        ),
    )

    # ========================================================
    # 0. Safe idempotent readback
    # ========================================================

    if REPORT.is_file():

        existing = load_json(
            REPORT
        )

        if (
            existing.get(
                "status"
            )
            ==
            "PASS_PREOUTCOME_METRIC_RECONCILED"
        ):

            print()
            print(
                "===== EXISTING RECONCILIATION DETECTED ====="
            )

            expected = existing.get(
                "reconciled",
                {}
            )

            require(
                sha256_file(
                    PROTOCOL
                )
                ==
                expected[
                    "protocol"
                ][
                    "sha256"
                ],
                (
                    "Existing reconciled protocol "
                    "does not match report."
                ),
            )

            require(
                sha256_file(
                    METRICS
                )
                ==
                expected[
                    "metric_runtime"
                ][
                    "sha256"
                ],
                (
                    "Existing reconciled runtime "
                    "does not match report."
                ),
            )

            require(
                sha256_file(
                    INTERFACE
                )
                ==
                expected[
                    "metric_interface"
                ][
                    "sha256"
                ],
                (
                    "Existing reconciled interface "
                    "does not match report."
                ),
            )

            rc, n, output = (
                run_tests(
                    "test_*.py"
                )
            )

            require(
                rc == 0,
                (
                    "Existing reconciliation "
                    "regression failed:\n"
                    +
                    tail(
                        output
                    )
                ),
            )

            require(
                n
                ==
                existing[
                    "regression"
                ][
                    "post"
                ],
                (
                    "Existing reconciliation test "
                    "count changed."
                ),
            )

            print(
                "existing protocol      = EXACT PASS"
            )

            print(
                "existing metric runtime= EXACT PASS"
            )

            print(
                "existing interface     = EXACT PASS"
            )

            print(
                "Stage6 regression      =",
                f"{n} / {n} PASS",
            )

            print(
                "STATUS = "
                "PASS_PREOUTCOME_METRIC_RECONCILED"
            )

            print(
                "no rewrite performed = YES"
            )

            print(
                "terminal remains open = YES"
            )

            return

    # ========================================================
    # A. Authority / pre-outcome boundary
    # ========================================================

    print()
    print(
        "===== A. AUTHORITY / PRE-OUTCOME BOUNDARY ====="
    )

    require(
        sha256_file(
            PREREG
        )
        ==
        EXPECTED_PREREG_SHA,
        (
            "Block6.6 preregistration "
            "SHA changed."
        ),
    )

    require(
        sha256_file(
            PROTOCOL
        )
        ==
        EXPECTED_OLD_PROTOCOL_SHA,
        (
            "Unexpected pre-repair "
            "protocol SHA."
        ),
    )

    require(
        sha256_file(
            METRICS
        )
        ==
        EXPECTED_OLD_RUNTIME_SHA,
        (
            "Unexpected pre-repair "
            "metric runtime SHA."
        ),
    )

    require(
        sha256_file(
            BLOCK67
        )
        ==
        EXPECTED_BLOCK67_SHA,
        (
            "Block6.7 closure changed."
        ),
    )

    require(
        sha256_file(
            CLASS_POLICY
        )
        ==
        EXPECTED_CLASS_POLICY_SHA,
        (
            "Frozen class-aware policy changed."
        ),
    )

    authority = load_json(
        AUTHORITY_AUDIT
    )

    require(
        authority.get(
            "status"
        )
        ==
        (
            "BLOCKED_PREOUTCOME_"
            "METRIC_RECONCILIATION_REQUIRED"
        ),
        (
            "Authority audit status mismatch."
        ),
    )

    require(
        authority[
            "scientific_boundary"
        ][
            "development_outcomes_read"
        ]
        is False,
        (
            "Development outcomes were "
            "already read."
        ),
    )

    require(
        authority[
            "scientific_boundary"
        ][
            "formal_outcomes_read"
        ]
        is False,
        (
            "Formal outcomes were "
            "already read."
        ),
    )

    require(
        authority[
            "scientific_boundary"
        ][
            "formal_evaluation"
        ]
        is False,
        (
            "Formal evaluation already started."
        ),
    )

    print(
        "Block6.6 preregistration = EXACT PASS"
    )

    print(
        "Block6.7 closure          = EXACT PASS"
    )

    print(
        "class-aware policy        = EXACT PASS"
    )

    print(
        "development outcomes      = NOT READ"
    )

    print(
        "formal outcomes           = NOT READ"
    )

    print(
        "authority                 = BLOCK6.6"
    )

    # ========================================================
    # B. Pre-repair full regression
    # ========================================================

    print()
    print(
        "===== B. PRE-REPAIR REGRESSION ====="
    )

    rc_before, n_before, out_before = (
        run_tests(
            "test_*.py"
        )
    )

    require(
        rc_before == 0,
        (
            "Pre-repair regression failed:\n"
            +
            tail(
                out_before
            )
        ),
    )

    require(
        n_before
        ==
        EXPECTED_PRE_REPAIR_TESTS,
        (
            "Expected "
            f"{EXPECTED_PRE_REPAIR_TESTS} "
            "pre-repair tests; "
            f"got {n_before}."
        ),
    )

    print(
        "pre-repair regression =",
        f"{n_before} / {n_before} PASS",
    )

    # ========================================================
    # C. Detect unexpected old latency API dependencies
    # ========================================================

    print()
    print(
        "===== C. LEGACY METRIC API DEPENDENCY CHECK ====="
    )

    unexpected_ms_refs = []

    roots = (
        S6
        / "src",
        S6
        / "tests",
    )

    for root in roots:

        for path in root.rglob(
            "*.py"
        ):

            if path in (
                METRICS,
                OLD_TEST,
            ):
                continue

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            if (
                "summarize_actuation_latency_ms"
                in text
            ):

                unexpected_ms_refs.append(
                    str(
                        path.relative_to(
                            S6
                        )
                    )
                )

    require(
        not unexpected_ms_refs,
        (
            "Unexpected dependencies on superseded "
            "millisecond latency API: "
            +
            repr(
                unexpected_ms_refs
            )
        ),
    )

    print(
        "unexpected *_latency_ms dependencies = NONE"
    )

    # ========================================================
    # D. Build candidates in memory
    # ========================================================

    print()
    print(
        "===== D. BUILD RECONCILED CANDIDATES ====="
    )

    protocol_bytes = (
        canonical_json_bytes(
            reconciled_protocol()
        )
    )

    metric_bytes = (
        METRIC_SOURCE.encode(
            "utf-8"
        )
    )

    test_bytes = (
        TEST_SOURCE.encode(
            "utf-8"
        )
    )

    recon_test_bytes = (
        RECON_TEST_SOURCE.encode(
            "utf-8"
        )
    )

    interface_original = (
        INTERFACE.read_text(
            encoding="utf-8"
        )
    )

    interface_patched = (
        patch_interface(
            interface_original
        )
    )

    interface_bytes = (
        interface_patched.encode(
            "utf-8"
        )
    )

    # Syntax validation before any write.
    for name, text in (
        (
            "metric_semantics.py",
            METRIC_SOURCE,
        ),
        (
            "baseline_metric_contract.py",
            interface_patched,
        ),
        (
            "test_block68_metric_semantics.py",
            TEST_SOURCE,
        ),
        (
            "test_block68_metric_reconciliation.py",
            RECON_TEST_SOURCE,
        ),
    ):

        compile(
            text,
            name,
            "exec",
        )

    json.loads(
        protocol_bytes.decode(
            "utf-8"
        )
    )

    print(
        "candidate Python syntax = PASS"
    )

    print(
        "candidate protocol JSON = PASS"
    )

    # ========================================================
    # E. Verified backups
    # ========================================================

    print()
    print(
        "===== E. VERIFIED BACKUP ====="
    )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    targets = (
        PROTOCOL,
        METRICS,
        INTERFACE,
        OLD_TEST,
        NEW_TEST,
    )

    originals = {}

    for path in targets:

        originals[
            path
        ] = (
            path.read_bytes()
            if path.exists()
            else None
        )

        if path.exists():

            destination = (
                backup_path(
                    path
                )
            )

            if destination.exists():

                require(
                    destination.read_bytes()
                    ==
                    path.read_bytes(),
                    (
                        "Existing backup differs "
                        "from current pre-repair file: "
                        f"{path}"
                    ),
                )

            else:

                destination.write_bytes(
                    path.read_bytes()
                )

            require(
                destination.read_bytes()
                ==
                path.read_bytes(),
                (
                    "Backup verification failed "
                    f"for {path}"
                ),
            )

            print(
                "backup verified =",
                destination,
            )

    # Protected files that must NEVER change.
    protected_before = {
        str(
            PREREG
        ):
            sha256_file(
                PREREG
            ),

        str(
            BLOCK67
        ):
            sha256_file(
                BLOCK67
            ),

        str(
            CLASS_POLICY
        ):
            sha256_file(
                CLASS_POLICY
            ),

        str(
            MARGIN_BINDING
        ):
            sha256_file(
                MARGIN_BINDING
            ),

        str(
            AUTHORITY_AUDIT
        ):
            sha256_file(
                AUTHORITY_AUDIT
            ),

        str(
            OLD_PART1
        ):
            sha256_file(
                OLD_PART1
            ),
    }

    # ========================================================
    # F. Apply explicit pre-outcome amendment
    # ========================================================

    print()
    print(
        "===== F. APPLY EXPLICIT PRE-OUTCOME AMENDMENT ====="
    )

    amendment_applied = False

    try:

        atomic_write(
            PROTOCOL,
            protocol_bytes,
        )

        atomic_write(
            METRICS,
            metric_bytes,
        )

        atomic_write(
            INTERFACE,
            interface_bytes,
        )

        atomic_write(
            OLD_TEST,
            test_bytes,
        )

        atomic_write(
            NEW_TEST,
            recon_test_bytes,
        )

        amendment_applied = True

        print(
            "protocol semantics      = BLOCK6.6 ALIGNED"
        )

        print(
            "metric runtime          = BLOCK6.6 ALIGNED"
        )

        print(
            "Block6.7 ROI flags      = EXPLICITLY AMENDED"
        )

        print(
            "historical reports      = PRESERVED"
        )

        # ====================================================
        # G. Targeted regression
        # ====================================================

        print()
        print(
            "===== G. TARGETED RECONCILIATION TESTS ====="
        )

        rc_metric, n_metric, out_metric = (
            run_tests(
                "test_block68_metric_semantics.py"
            )
        )

        require(
            rc_metric == 0,
            (
                "Reconciled metric tests failed:\n"
                +
                tail(
                    out_metric
                )
            ),
        )

        require(
            n_metric == 15,
            (
                "Expected 15 reconciled metric tests; "
                f"got {n_metric}."
            ),
        )

        rc_recon, n_recon, out_recon = (
            run_tests(
                "test_block68_metric_reconciliation.py"
            )
        )

        require(
            rc_recon == 0,
            (
                "Metric authority tests failed:\n"
                +
                tail(
                    out_recon
                )
            ),
        )

        require(
            n_recon
            ==
            EXPECTED_NEW_RECON_TESTS,
            (
                "Expected "
                f"{EXPECTED_NEW_RECON_TESTS} "
                "reconciliation tests; "
                f"got {n_recon}."
            ),
        )

        print(
            "metric semantics tests = 15 / 15 PASS"
        )

        print(
            "authority tests        =",
            f"{n_recon} / {n_recon} PASS",
        )

        # ====================================================
        # H. Full regression
        # ====================================================

        print()
        print(
            "===== H. FULL STAGE6 REGRESSION ====="
        )

        rc_after, n_after, out_after = (
            run_tests(
                "test_*.py"
            )
        )

        require(
            rc_after == 0,
            (
                "Post-reconciliation regression failed:\n"
                +
                tail(
                    out_after
                )
            ),
        )

        expected_after = (
            n_before
            +
            EXPECTED_NEW_RECON_TESTS
        )

        require(
            n_after
            ==
            expected_after,
            (
                "Unexpected post-reconciliation "
                f"test count {n_after}; "
                f"expected {expected_after}."
            ),
        )

        print(
            "full Stage6 regression =",
            f"{n_after} / {n_after} PASS",
        )

        # ====================================================
        # I. Exact semantic readback
        # ====================================================

        print()
        print(
            "===== I. AUTHORITY / SEMANTIC READBACK ====="
        )

        protocol = load_json(
            PROTOCOL
        )

        require(
            protocol[
                "authority"
            ][
                "binding_source"
            ]
            ==
            "Block6.6_preoutcome_preregistration",
            (
                "Authority readback failed."
            ),
        )

        require(
            protocol[
                "authority"
            ][
                "development_outcomes_seen"
            ]
            is False,
            (
                "Development outcome boundary changed."
            ),
        )

        require(
            protocol[
                "authority"
            ][
                "formal_outcomes_seen"
            ]
            is False,
            (
                "Formal outcome boundary changed."
            ),
        )

        require(
            protocol[
                "metrics"
            ][
                "temporal_smoothness"
            ][
                "direction"
            ]
            ==
            "lower_is_better",
            (
                "Temporal direction "
                "was not reconciled."
            ),
        )

        require(
            protocol[
                "metrics"
            ][
                "over_masking_area"
            ][
                "formula"
            ]
            ==
            "|D \\ O| / |full actuator grid|",
            (
                "Overmask formula "
                "was not reconciled."
            ),
        )

        require(
            protocol[
                "metrics"
            ][
                "false_dimming"
            ][
                "formula"
            ]
            ==
            "|D \\ O| / |D|; 0 when |D|=0",
            (
                "False-dimming formula "
                "was not reconciled."
            ),
        )

        require(
            "delta_threshold"
            not in
            protocol[
                "metrics"
            ][
                "flicker_change_rate"
            ],
            (
                "Old flicker threshold remains."
            ),
        )

        require(
            protocol[
                "metrics"
            ][
                "vehicle_shadow_zone_violation"
            ][
                "no_vehicle_cells"
            ]
            ==
            "NA",
            (
                "Vehicle empty-support "
                "semantics changed."
            ),
        )

        require(
            protocol[
                "metrics"
            ][
                "energy_consumption"
            ][
                "formula"
            ]
            ==
            "mean over horizon/grid of I_final",
            (
                "Energy domain "
                "was not reconciled."
            ),
        )

        require(
            protocol[
                "metrics"
            ][
                "actuation_latency"
            ][
                "units"
            ]
            ==
            "seconds",
            (
                "Latency units "
                "were not reconciled."
            ),
        )

        print(
            "overmask denominator     = FULL ACTUATOR GRID"
        )

        print(
            "false dimming            = |D\\O| / |D|"
        )

        print(
            "temporal metric          = CHANGE RATE"
        )

        print(
            "temporal direction       = LOWER IS BETTER"
        )

        print(
            "flicker threshold        = NONE"
        )

        print(
            "vehicle empty support    = NA"
        )

        print(
            "energy domain            = FULL HORIZON/GRID"
        )

        print(
            "latency units            = SECONDS"
        )

        # ====================================================
        # J. Protected immutability
        # ====================================================

        print()
        print(
            "===== J. PROTECTED IMMUTABILITY ====="
        )

        protected_after = {
            str(
                PREREG
            ):
                sha256_file(
                    PREREG
                ),

            str(
                BLOCK67
            ):
                sha256_file(
                    BLOCK67
                ),

            str(
                CLASS_POLICY
            ):
                sha256_file(
                    CLASS_POLICY
                ),

            str(
                MARGIN_BINDING
            ):
                sha256_file(
                    MARGIN_BINDING
                ),

            str(
                AUTHORITY_AUDIT
            ):
                sha256_file(
                    AUTHORITY_AUDIT
                ),

            str(
                OLD_PART1
            ):
                sha256_file(
                    OLD_PART1
                ),
        }

        require(
            protected_before
            ==
            protected_after,
            (
                "Protected historical/frozen "
                "evidence changed."
            ),
        )

        print(
            "Block6.6 preregistration = UNCHANGED"
        )

        print(
            "Block6.7 closure          = UNCHANGED"
        )

        print(
            "class-aware policy        = UNCHANGED"
        )

        print(
            "m_extra binding           = UNCHANGED"
        )

        print(
            "authority audit           = UNCHANGED"
        )

        print(
            "historical Block6.8 Part1 = UNCHANGED"
        )

        # ====================================================
        # K. Storage
        # ====================================================

        print()
        print(
            "===== K. STORAGE ====="
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
                "250-GiB reserve violated."
            ),
        )

        print(
            "free GiB        =",
            round(
                free_gib,
                3,
            ),
        )

        print(
            "250-GiB reserve = PASS"
        )

        # ====================================================
        # L. Amendment report
        # ====================================================

        print()
        print(
            "===== L. RECONCILIATION FREEZE ====="
        )

        result = {
            "project":
                "Agni",

            "stage":
                6,

            "block":
                "6.8",

            "part":
                "preoutcome_metric_reconciliation",

            "status":
                "PASS_PREOUTCOME_METRIC_RECONCILED",

            "authority": {
                "source":
                    str(
                        PREREG
                    ),

                "sha256":
                    EXPECTED_PREREG_SHA,

                "rule":
                    (
                        "earliest explicit pre-outcome "
                        "preregistration is authoritative"
                    ),
            },

            "scientific_boundary": {
                "development_outcomes_read":
                    False,

                "formal_outcomes_read":
                    False,

                "formal_evaluation":
                    False,

                "policy_tuning":
                    False,

                "parameter_tuning":
                    False,
            },

            "historical_Block68_Part1": {
                "report":
                    str(
                        OLD_PART1
                    ),

                "status":
                    (
                        "PRESERVED_BUT_SUPERSEDED_BY_"
                        "PREOUTCOME_RECONCILIATION"
                    ),

                "old_protocol_sha256":
                    EXPECTED_OLD_PROTOCOL_SHA,

                "old_runtime_sha256":
                    EXPECTED_OLD_RUNTIME_SHA,
            },

            "reconciled": {
                "protocol": {
                    "path":
                        str(
                            PROTOCOL
                        ),

                    "sha256":
                        sha256_file(
                            PROTOCOL
                        ),
                },

                "metric_runtime": {
                    "path":
                        str(
                            METRICS
                        ),

                    "sha256":
                        sha256_file(
                            METRICS
                        ),
                },

                "metric_interface": {
                    "path":
                        str(
                            INTERFACE
                        ),

                    "sha256":
                        sha256_file(
                            INTERFACE
                        ),

                    "amended_flags": [
                        (
                            "over_masking_area."
                            "requires_road_ROI=false"
                        ),
                        (
                            "false_dimming."
                            "requires_road_ROI=false"
                        ),
                        (
                            "energy_consumption."
                            "requires_road_ROI=false"
                        )
                    ],
                },
            },

            "semantic_resolution": {
                "binary_dim_support":
                    "D=[I_final<1]",

                "vehicle_no_support":
                    "NA",

                "over_masking":
                    "|D\\O|/|full actuator grid|",

                "false_dimming":
                    "|D\\O|/|D|",

                "temporal_smoothness":
                    (
                        "mean_absolute_change_rate_"
                        "lower_is_better"
                    ),

                "flicker":
                    "exact_I_h_not_equal_I_prev",

                "energy":
                    "mean_full_horizon_grid",

                "latency":
                    (
                        "seconds_loaded_Pocc_to_"
                        "four_horizon_schedule"
                    ),
            },

            "regression": {
                "pre":
                    n_before,

                "metric_semantics_tests":
                    n_metric,

                "new_reconciliation_tests":
                    n_recon,

                "post":
                    n_after,

                "status":
                    "PASS",
            },

            "immutability": {
                "Block66_preregistration":
                    "UNCHANGED",

                "Block67_closure":
                    "UNCHANGED",

                "class_aware_policy":
                    "UNCHANGED",

                "class_margin_binding":
                    "UNCHANGED",

                "historical_Block68_Part1_report":
                    "UNCHANGED",
            },

            "storage": {
                "free_GiB":
                    free_gib,

                "hard_reserve_GiB":
                    MIN_FREE_GIB,

                "pass":
                    True,
            },

            "formal_evaluation_allowed":
                False,

            "next":
                (
                    "development-only metric runtime "
                    "evaluation and numeric "
                    "non-inferiority bound freeze"
                ),
        }

        atomic_write(
            REPORT,
            canonical_json_bytes(
                result
            ),
        )

        print(
            "reconciled protocol SHA =",
            sha256_file(
                PROTOCOL
            ),
        )

        print(
            "reconciled runtime SHA  =",
            sha256_file(
                METRICS
            ),
        )

        print(
            "reconciled interface SHA=",
            sha256_file(
                INTERFACE
            ),
        )

        print(
            "reconciliation report SHA=",
            sha256_file(
                REPORT
            ),
        )

    except BaseException:

        if amendment_applied:

            print()
            print(
                "============================================================"
            )
            print(
                "RECONCILIATION FAILURE -> AUTOMATIC ROLLBACK"
            )
            print(
                "============================================================"
            )

            restore(
                originals
            )

            # Ensure the report from a failed attempt cannot
            # survive as a false PASS.
            if REPORT.exists():
                REPORT.unlink()

            rc_rollback, n_rollback, rollback_output = (
                run_tests(
                    "test_*.py"
                )
            )

            print(
                "rollback regression =",
                n_rollback,
                "/",
                n_rollback,
                (
                    "PASS"
                    if rc_rollback == 0
                    else "FAIL"
                ),
            )

            if rc_rollback != 0:

                print(
                    tail(
                        rollback_output
                    )
                )

            require(
                rc_rollback == 0,
                (
                    "Automatic rollback completed "
                    "but regression does not pass."
                ),
            )

            require(
                n_rollback
                ==
                EXPECTED_PRE_REPAIR_TESTS,
                (
                    "Rollback test count mismatch: "
                    f"{n_rollback}"
                ),
            )

            print(
                "rollback state = VERIFIED PRE-REPAIR"
            )

        raise

    # ========================================================
    # M. Final
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.8 PRE-OUTCOME RECONCILIATION — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "earliest prereg authority = BLOCK6.6 EXACT"
    )

    print(
        "semantic conflicts        = RECONCILED"
    )

    print(
        "silent override           = NO"
    )

    print(
        "historical Part1          = PRESERVED / SUPERSEDED"
    )

    print(
        "Block6.7 interface change = EXPLICIT AMENDMENT"
    )

    print(
        "overmask                  = |D\\O| / FULL GRID"
    )

    print(
        "false dimming             = |D\\O| / |D|"
    )

    print(
        "temporal smoothness label = MEAN ABS CHANGE RATE"
    )

    print(
        "temporal direction        = LOWER IS BETTER"
    )

    print(
        "flicker threshold         = NONE"
    )

    print(
        "vehicle no-support        = NA"
    )

    print(
        "energy                    = FULL HORIZON/GRID MEAN"
    )

    print(
        "actuation latency         = SECONDS"
    )

    print(
        "development outcomes      = NOT READ"
    )

    print(
        "formal outcomes           = NOT READ"
    )

    print(
        "formal evaluation         = STILL BLOCKED"
    )

    print(
        "policy/parameter tuning   = NO"
    )

    print(
        "metric tests              =",
        f"{n_metric} / {n_metric} PASS",
    )

    print(
        "reconciliation tests      =",
        f"{n_recon} / {n_recon} PASS",
    )

    print(
        "full Stage6 regression    =",
        f"{n_after} / {n_after} PASS",
    )

    print(
        "STATUS = PASS_PREOUTCOME_METRIC_RECONCILED"
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
        "BLOCK 6.8 PRE-OUTCOME RECONCILIATION = BLOCKED"
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

    print()
    print(
        "automatic rollback attempted = YES"
    )

    print(
        "development outcomes read     = NO"
    )

    print(
        "formal outcomes read          = NO"
    )

    print(
        "formal evaluation             = NO"
    )

    print(
        "policy tuning                 = NO"
    )

    print(
        "parameter tuning              = NO"
    )

    print()
    print(
        "RECOVERY:"
    )

    print(
        "Do not run development evaluation."
    )

    print(
        "Do not run formal Stage6 evaluation."
    )

    print(
        "Do not manually modify the "
        "Block6.6 preregistration."
    )

    print(
        "Send this BLOCKED section for "
        "targeted diagnosis."
    )

    print(
        "terminal remains open = YES"
    )

# Intentionally no non-zero sys.exit().

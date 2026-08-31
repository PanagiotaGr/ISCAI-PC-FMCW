from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import traceback


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

SRC = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

TEST = (
    S6
    / "tests/"
      "test_block66_part3c1_class_aware_policy.py"
)

REPORT = (
    S6
    / "reports/"
      "block66_part3c1_class_aware_runtime.json"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3c1_to_part3c2_handoff.json"
)

PART3B_PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

PART3B_REPORT = (
    S6
    / "reports/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

PART3B_FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part3b_preregistration_freeze_manifest.json"
)

PART3B_HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part3b_to_part3c_handoff.json"
)

PART_A_REACTIVE = (
    S6
    / "src/iscai_stage6/adb/"
      "part_a_reactive.py"
)

ILLUMINATION = (
    S6
    / "src/iscai_stage6/adb/"
      "illumination.py"
)

PREDICTIVE_MASK = (
    S6
    / "src/iscai_stage6/adb/"
      "predictive_mask.py"
)


EXPECTED = {
    "part3b_prereg":
        (
            "5704494a89b4b3c7a3c19b0df8176c55"
            "0c00795ed17cf2906c343f340461429e"
        ),

    "part3b_report":
        (
            "1bf01b44a1cd842a1e1bf78f094da9db"
            "b50c622ff895161851e901e1d36a53ac"
        ),

    "part3b_freeze":
        (
            "a1d13bced062c2f2303b7166ab4f8b24"
            "f1ff17a236950dc569dcf16c580108c1"
        ),

    "part3b_handoff":
        (
            "60874c8a4d9c8feae5e800f89168e280"
            "7a9c859c37dab0577186055e82ad5163"
        ),

    "part_a_reactive":
        (
            "19f80f325aebc03b5257ca3c0408d1dd"
            "ea1ac3a8ff4ad875914224584e2ef173"
        ),

    "illumination":
        (
            "daeb96029ff6e1720d17cd20d3e8f001"
            "55c566eeecec3b3726f4adcc8b05d832"
        ),

    "predictive_mask":
        (
            "239611ddb1cc21b9ddd46962eb561e58e"
            "0d1474be58878f9c79c256ba1940575"
        ),
}


MODULE_TEXT = r'''
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from iscai_stage6.adb.illumination import (
    part_a_radial_profile,
)
from iscai_stage6.adb.part_a_reactive import (
    PART_A_CONTROL_TRANSITION_LENGTH_M,
    PART_A_EPS,
    PART_A_LATERAL_SAFETY_MARGIN_M,
    PART_A_RADIAL_SAFETY_MARGIN_M,
)


TYPE_VEHICLE = "TYPE_VEHICLE"
TYPE_PEDESTRIAN = "TYPE_PEDESTRIAN"
TYPE_CYCLIST = "TYPE_CYCLIST"

CORE_CLASSES = (
    TYPE_VEHICLE,
    TYPE_PEDESTRIAN,
    TYPE_CYCLIST,
)


@dataclass(frozen=True)
class MarginComputation:
    actor_class: str
    predicted_center_H0_xy_m: tuple[float, float]
    predicted_range_m: float
    lateral_predictive_sigma_m: float
    closing_speed_mps: float
    cyclist_lateral_rate_mps: float
    part_a_generic_lateral_margin_m: float
    additional_class_margin_m: float
    total_lateral_margin_m: float


@dataclass(frozen=True)
class ClassAwareComposition:
    illumination_minimum: np.ndarray
    vru_floor_guard: np.ndarray
    raw_class_aware_illumination: np.ndarray
    semantics: str = (
        "minimum_actor_intensity_then_spatial_VRU_floor_guard"
    )


@dataclass(frozen=True)
class RateLimitedSchedule:
    illumination: np.ndarray
    safety_override_mask: np.ndarray
    safety_override_count: int
    safety_override_fraction: float
    semantics: str = (
        "finite_dim_bright_rate_limit_then_VRU_floor_projection"
    )


def _finite_scalar(
    value,
    *,
    name: str,
) -> float:
    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be finite"
        )

    return result


def _validate_actor_class(
    actor_class: str,
) -> str:
    value = str(actor_class)

    if value not in CORE_CLASSES:
        raise ValueError(
            "actor_class must be one of "
            f"{CORE_CLASSES}; got {value!r}"
        )

    return value


def _xy(
    value,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape not in (
        (2,),
        (3,),
    ):
        raise ValueError(
            f"{name} must have shape (2,) or (3,); "
            f"got {array.shape}"
        )

    result = np.asarray(
        array[:2],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(result)
    ):
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return result


def _covariance_xy(
    value,
) -> np.ndarray:
    array = np.asarray(
        value,
        dtype=np.float64,
    )

    if array.shape == (3, 3):
        array = array[:2, :2]

    if array.shape != (2, 2):
        raise ValueError(
            "covariance_H0_m2 must have shape "
            "(2,2) or (3,3)"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "covariance_H0_m2 contains non-finite values"
        )

    if not np.allclose(
        array,
        array.T,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "covariance_H0_m2 must be symmetric"
        )

    eigenvalues = np.linalg.eigvalsh(
        array
    )

    if float(
        np.min(eigenvalues)
    ) < -1.0e-10:
        raise ValueError(
            "covariance_H0_m2 must be positive semidefinite"
        )

    return array


def threshold_occupancy_counts_strict_k(
    occupancy_counts,
    *,
    k: int,
    sample_count: int = 8192,
) -> np.ndarray:
    """
    Exact preregistered occupancy threshold:

        M = 1[count > k]

    This avoids any float threshold ambiguity.
    """

    if isinstance(k, bool):
        raise ValueError(
            "k must be an integer occupancy quantum"
        )

    k_value = int(k)

    if k_value != k:
        raise ValueError(
            "k must be an exact integer"
        )

    n = int(sample_count)

    if n <= 0:
        raise ValueError(
            "sample_count must be positive"
        )

    if not (
        0 <= k_value <= n - 1
    ):
        raise ValueError(
            "k must satisfy 0 <= k <= sample_count-1"
        )

    counts = np.asarray(
        occupancy_counts
    )

    if not np.issubdtype(
        counts.dtype,
        np.integer,
    ):
        raise ValueError(
            "occupancy_counts must have integer dtype"
        )

    if np.any(
        counts < 0
    ):
        raise ValueError(
            "occupancy_counts cannot be negative"
        )

    if np.any(
        counts > n
    ):
        raise ValueError(
            "occupancy_counts exceed sample_count"
        )

    return np.asarray(
        counts > k_value,
        dtype=bool,
    )


def compute_class_aware_margin(
    actor_class: str,
    *,
    current_position_H0_m,
    mean_displacement_H0_m,
    covariance_H0_m2,
    horizon_s: float,
    base_margin_m: float,
    uncertainty_multiplier: float,
    motion_multiplier: float = 0.0,
) -> MarginComputation:
    """
    Preregistered class-aware predictive margin.

    The frozen Part-A generic 1 m lateral safety margin is
    preserved and the class-aware term is ADDITIONAL.

    Vehicle:
        extra = b + u*sigma_lat + q*tau*v_close

    Pedestrian:
        extra = b + u*sigma_lat

    Cyclist:
        extra = b + u*sigma_lat + q*tau*v_lat
    """

    cls = _validate_actor_class(
        actor_class
    )

    current_xy = _xy(
        current_position_H0_m,
        name="current_position_H0_m",
    )

    displacement_xy = _xy(
        mean_displacement_H0_m,
        name="mean_displacement_H0_m",
    )

    covariance_xy = _covariance_xy(
        covariance_H0_m2
    )

    tau = _finite_scalar(
        horizon_s,
        name="horizon_s",
    )

    base = _finite_scalar(
        base_margin_m,
        name="base_margin_m",
    )

    uncertainty = _finite_scalar(
        uncertainty_multiplier,
        name="uncertainty_multiplier",
    )

    motion = _finite_scalar(
        motion_multiplier,
        name="motion_multiplier",
    )

    if tau <= 0.0:
        raise ValueError(
            "horizon_s must be > 0"
        )

    if base < 0.0:
        raise ValueError(
            "base_margin_m must be >= 0"
        )

    if uncertainty < 0.0:
        raise ValueError(
            "uncertainty_multiplier must be >= 0"
        )

    if motion < 0.0:
        raise ValueError(
            "motion_multiplier must be >= 0"
        )

    predicted_xy = (
        current_xy
        +
        displacement_xy
    )

    predicted_range = float(
        np.linalg.norm(
            predicted_xy
        )
    )

    current_range = float(
        np.linalg.norm(
            current_xy
        )
    )

    predicted_denominator = max(
        predicted_range,
        float(PART_A_EPS),
    )

    lateral_los_predicted = np.asarray(
        [
            -predicted_xy[1]
            /
            predicted_denominator,

            predicted_xy[0]
            /
            predicted_denominator,
        ],
        dtype=np.float64,
    )

    sigma_squared = float(
        lateral_los_predicted
        @
        covariance_xy
        @
        lateral_los_predicted
    )

    if sigma_squared < 0.0:
        if sigma_squared >= -1.0e-10:
            sigma_squared = 0.0
        else:
            raise ValueError(
                "negative lateral variance"
            )

    sigma_lat = math.sqrt(
        sigma_squared
    )

    closing_speed = max(
        0.0,
        (
            current_range
            -
            predicted_range
        )
        /
        tau,
    )

    current_denominator = max(
        current_range,
        float(PART_A_EPS),
    )

    lateral_los_current = np.asarray(
        [
            -current_xy[1]
            /
            current_denominator,

            current_xy[0]
            /
            current_denominator,
        ],
        dtype=np.float64,
    )

    cyclist_lateral_rate = (
        abs(
            float(
                lateral_los_current
                @
                displacement_xy
            )
        )
        /
        tau
    )

    if cls == TYPE_VEHICLE:
        additional = (
            base
            +
            uncertainty
            *
            sigma_lat
            +
            motion
            *
            tau
            *
            closing_speed
        )

    elif cls == TYPE_PEDESTRIAN:
        additional = (
            base
            +
            uncertainty
            *
            sigma_lat
        )

    else:
        additional = (
            base
            +
            uncertainty
            *
            sigma_lat
            +
            motion
            *
            tau
            *
            cyclist_lateral_rate
        )

    generic_margin = float(
        PART_A_LATERAL_SAFETY_MARGIN_M
    )

    total_margin = (
        generic_margin
        +
        additional
    )

    return MarginComputation(
        actor_class=cls,

        predicted_center_H0_xy_m=(
            float(
                predicted_xy[0]
            ),
            float(
                predicted_xy[1]
            ),
        ),

        predicted_range_m=float(
            predicted_range
        ),

        lateral_predictive_sigma_m=float(
            sigma_lat
        ),

        closing_speed_mps=float(
            closing_speed
        ),

        cyclist_lateral_rate_mps=float(
            cyclist_lateral_rate
        ),

        part_a_generic_lateral_margin_m=float(
            generic_margin
        ),

        additional_class_margin_m=float(
            additional
        ),

        total_lateral_margin_m=float(
            total_margin
        ),
    )


def angular_margin_cells(
    *,
    total_margin_m: float,
    predicted_range_m: float,
    theta_step_rad: float,
) -> int:
    """
    Convert total lateral safety expansion to frozen
    theta-grid cells using the preregistered ceil rule.
    """

    margin = _finite_scalar(
        total_margin_m,
        name="total_margin_m",
    )

    actor_range = _finite_scalar(
        predicted_range_m,
        name="predicted_range_m",
    )

    step = _finite_scalar(
        theta_step_rad,
        name="theta_step_rad",
    )

    if margin < 0.0:
        raise ValueError(
            "total_margin_m must be >= 0"
        )

    if actor_range < 0.0:
        raise ValueError(
            "predicted_range_m must be >= 0"
        )

    if step <= 0.0:
        raise ValueError(
            "theta_step_rad must be > 0"
        )

    delta_theta = math.atan2(
        margin,
        max(
            actor_range,
            float(PART_A_EPS),
        ),
    )

    return int(
        math.ceil(
            delta_theta
            /
            step
        )
    )


def dilate_mask_theta(
    mask,
    *,
    dilation_cells: int,
) -> np.ndarray:
    """
    Symmetric theta-axis dilation only.

    Expected scientific shape:
        [horizon, theta, range]

    No range dilation and no vertical/pixel dimension.
    """

    source = np.asarray(
        mask,
        dtype=bool,
    )

    if source.ndim != 3:
        raise ValueError(
            "mask must have shape "
            "[horizon, theta, range]"
        )

    cells = int(
        dilation_cells
    )

    if cells != dilation_cells:
        raise ValueError(
            "dilation_cells must be an integer"
        )

    if cells < 0:
        raise ValueError(
            "dilation_cells must be >= 0"
        )

    if cells == 0:
        return source.copy()

    result = source.copy()

    n_theta = int(
        source.shape[1]
    )

    for offset in range(
        1,
        cells + 1,
    ):
        if offset >= n_theta:
            break

        result[
            :,
            offset:,
            :,
        ] |= source[
            :,
            :-offset,
            :,
        ]

        result[
            :,
            :-offset,
            :,
        ] |= source[
            :,
            offset:,
            :,
        ]

    return result


def predictive_mask_to_illumination(
    mask,
    *,
    range_centers_m,
    intensity_floor: float,
) -> np.ndarray:
    """
    Convert class predictive support to the preserved
    Part-A raised-cosine radial illumination.

    For each active [horizon,theta] column:
        r_far = max active range center
        r_shadow_end = r_far + frozen +4 m
        r_transition_end = r_shadow_end + frozen 20 m

    The exact existing part_a_radial_profile() is reused.
    """

    binary = np.asarray(
        mask,
        dtype=bool,
    )

    if binary.ndim != 3:
        raise ValueError(
            "mask must have shape "
            "[horizon, theta, range]"
        )

    ranges = np.asarray(
        range_centers_m,
        dtype=np.float64,
    )

    if ranges.ndim != 1:
        raise ValueError(
            "range_centers_m must be one-dimensional"
        )

    if binary.shape[2] != ranges.size:
        raise ValueError(
            "mask range axis does not match "
            "range_centers_m"
        )

    if not np.all(
        np.isfinite(ranges)
    ):
        raise ValueError(
            "range_centers_m contains non-finite values"
        )

    if np.any(
        np.diff(ranges)
        <=
        0.0
    ):
        raise ValueError(
            "range_centers_m must be strictly increasing"
        )

    floor = _finite_scalar(
        intensity_floor,
        name="intensity_floor",
    )

    if not (
        0.0 <= floor <= 1.0
    ):
        raise ValueError(
            "intensity_floor must lie in [0,1]"
        )

    output = np.ones(
        binary.shape,
        dtype=np.float64,
    )

    radial_margin = float(
        PART_A_RADIAL_SAFETY_MARGIN_M
    )

    transition_length = float(
        PART_A_CONTROL_TRANSITION_LENGTH_M
    )

    horizon_count = int(
        binary.shape[0]
    )

    theta_count = int(
        binary.shape[1]
    )

    for h in range(
        horizon_count
    ):
        for t in range(
            theta_count
        ):
            active = binary[
                h,
                t,
                :,
            ]

            if not np.any(
                active
            ):
                continue

            r_far = float(
                ranges[
                    np.flatnonzero(
                        active
                    )[-1]
                ]
            )

            r_shadow_end = (
                r_far
                +
                radial_margin
            )

            r_transition_end = (
                r_shadow_end
                +
                transition_length
            )

            output[
                h,
                t,
                :,
            ] = (
                part_a_radial_profile(
                    ranges,
                    r_shadow_end_m=
                        r_shadow_end,
                    r_transition_end_m=
                        r_transition_end,
                    intensity_floor=
                        floor,
                )
            )

    return output


def compose_class_aware_illumination(
    *,
    class_illumination: Mapping[
        str,
        np.ndarray,
    ],
    class_active_masks: Mapping[
        str,
        np.ndarray,
    ],
    class_floors: Mapping[
        str,
        float,
    ],
) -> ClassAwareComposition:
    """
    Frozen overlap semantics:

      I_min = minimum over active class illumination maps

      F_vru(cell) =
          max(
              pedestrian floor if pedestrian active,
              cyclist floor if cyclist active,
              0
          )

      I_raw = max(I_min, F_vru)

    Thus a vehicle cannot force an overlapping VRU cell below
    the active VRU's non-blackout floor.
    """

    if not class_illumination:
        raise ValueError(
            "class_illumination cannot be empty"
        )

    keys = tuple(
        class_illumination.keys()
    )

    for key in keys:
        _validate_actor_class(
            key
        )

    arrays = {
        key:
            np.asarray(
                value,
                dtype=np.float64,
            )
        for key, value
        in class_illumination.items()
    }

    reference_shape = next(
        iter(
            arrays.values()
        )
    ).shape

    if len(
        reference_shape
    ) != 3:
        raise ValueError(
            "class illumination arrays must have shape "
            "[horizon,theta,range]"
        )

    for key, array in arrays.items():
        if array.shape != reference_shape:
            raise ValueError(
                "class illumination shapes differ"
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                f"{key} illumination contains "
                "non-finite values"
            )

        if (
            np.any(array < 0.0)
            or
            np.any(array > 1.0)
        ):
            raise ValueError(
                f"{key} illumination must lie in [0,1]"
            )

    minimum = np.minimum.reduce(
        [
            arrays[key]
            for key in keys
        ]
    )

    vru_guard = np.zeros(
        reference_shape,
        dtype=np.float64,
    )

    for vru_class in (
        TYPE_PEDESTRIAN,
        TYPE_CYCLIST,
    ):
        if vru_class not in arrays:
            continue

        if vru_class not in class_active_masks:
            raise ValueError(
                f"missing active mask for {vru_class}"
            )

        if vru_class not in class_floors:
            raise ValueError(
                f"missing floor for {vru_class}"
            )

        active = np.asarray(
            class_active_masks[
                vru_class
            ],
            dtype=bool,
        )

        if active.shape != reference_shape:
            raise ValueError(
                f"{vru_class} active mask shape differs"
            )

        floor = _finite_scalar(
            class_floors[
                vru_class
            ],
            name=f"{vru_class}_floor",
        )

        if not (
            0.0 < floor <= 1.0
        ):
            raise ValueError(
                "VRU floors must satisfy 0 < floor <= 1"
            )

        vru_guard = np.maximum(
            vru_guard,
            np.where(
                active,
                floor,
                0.0,
            ),
        )

    raw = np.maximum(
        minimum,
        vru_guard,
    )

    return ClassAwareComposition(
        illumination_minimum=
            np.asarray(
                minimum,
                dtype=np.float64,
            ),

        vru_floor_guard=
            np.asarray(
                vru_guard,
                dtype=np.float64,
            ),

        raw_class_aware_illumination=
            np.asarray(
                raw,
                dtype=np.float64,
            ),
    )


def temporal_smooth_schedule(
    raw_schedule,
    *,
    current_illumination,
    horizons_s: Sequence[float],
    time_constant_s: float,
) -> np.ndarray:
    """
    Frozen causal smoothing equation:

        alpha_h = 1-exp(-dt_h/T)
        I_h = alpha_h*raw_h + (1-alpha_h)*I_prev

    Initial I_prev is current causal illumination.
    """

    raw = np.asarray(
        raw_schedule,
        dtype=np.float64,
    )

    current = np.asarray(
        current_illumination,
        dtype=np.float64,
    )

    horizons = np.asarray(
        tuple(
            float(value)
            for value in horizons_s
        ),
        dtype=np.float64,
    )

    T = _finite_scalar(
        time_constant_s,
        name="time_constant_s",
    )

    if raw.ndim != 3:
        raise ValueError(
            "raw_schedule must have shape "
            "[horizon,theta,range]"
        )

    if current.shape != raw.shape[1:]:
        raise ValueError(
            "current_illumination shape mismatch"
        )

    if horizons.shape != (
        raw.shape[0],
    ):
        raise ValueError(
            "horizons_s length mismatch"
        )

    if not np.all(
        np.isfinite(raw)
    ):
        raise ValueError(
            "raw_schedule contains non-finite values"
        )

    if not np.all(
        np.isfinite(current)
    ):
        raise ValueError(
            "current_illumination contains non-finite values"
        )

    if (
        np.any(raw < 0.0)
        or
        np.any(raw > 1.0)
        or
        np.any(current < 0.0)
        or
        np.any(current > 1.0)
    ):
        raise ValueError(
            "illumination must lie in [0,1]"
        )

    if T <= 0.0:
        raise ValueError(
            "time_constant_s must be > 0"
        )

    if (
        horizons.size == 0
        or
        horizons[0] <= 0.0
        or
        np.any(
            np.diff(
                horizons
            )
            <=
            0.0
        )
    ):
        raise ValueError(
            "horizons_s must be strictly increasing "
            "positive times"
        )

    output = np.empty_like(
        raw,
        dtype=np.float64,
    )

    previous = current.copy()
    previous_time = 0.0

    for index, horizon in enumerate(
        horizons
    ):
        dt = float(
            horizon
            -
            previous_time
        )

        alpha = (
            1.0
            -
            math.exp(
                -dt
                /
                T
            )
        )

        current_output = (
            alpha
            *
            raw[
                index
            ]
            +
            (
                1.0
                -
                alpha
            )
            *
            previous
        )

        output[
            index
        ] = current_output

        previous = current_output
        previous_time = float(
            horizon
        )

    return np.clip(
        output,
        0.0,
        1.0,
    )


def apply_actuation_rate_limit(
    smoothed_schedule,
    *,
    current_illumination,
    horizons_s: Sequence[float],
    rho_dim_per_s: float,
    rho_bright_per_s: float,
    vru_floor_guard=None,
) -> RateLimitedSchedule:
    """
    Frozen finite-rate semantics:

        I_rate = clip(
            I_smooth,
            I_prev-rho_dim*dt,
            I_prev+rho_bright*dt
        )

    Then the active VRU floor is re-applied and every cell
    changed by this safety projection is counted.
    """

    smooth = np.asarray(
        smoothed_schedule,
        dtype=np.float64,
    )

    current = np.asarray(
        current_illumination,
        dtype=np.float64,
    )

    horizons = np.asarray(
        tuple(
            float(value)
            for value in horizons_s
        ),
        dtype=np.float64,
    )

    rho_dim = _finite_scalar(
        rho_dim_per_s,
        name="rho_dim_per_s",
    )

    rho_bright = _finite_scalar(
        rho_bright_per_s,
        name="rho_bright_per_s",
    )

    if smooth.ndim != 3:
        raise ValueError(
            "smoothed_schedule must have shape "
            "[horizon,theta,range]"
        )

    if current.shape != smooth.shape[1:]:
        raise ValueError(
            "current_illumination shape mismatch"
        )

    if horizons.shape != (
        smooth.shape[0],
    ):
        raise ValueError(
            "horizons_s length mismatch"
        )

    if (
        rho_dim <= 0.0
        or
        rho_bright <= 0.0
    ):
        raise ValueError(
            "rate limits must be finite and > 0"
        )

    if rho_dim < rho_bright:
        raise ValueError(
            "rho_dim_per_s must be >= rho_bright_per_s"
        )

    if (
        horizons.size == 0
        or
        horizons[0] <= 0.0
        or
        np.any(
            np.diff(
                horizons
            )
            <=
            0.0
        )
    ):
        raise ValueError(
            "horizons_s must be strictly increasing "
            "positive times"
        )

    if vru_floor_guard is None:
        guard = np.zeros_like(
            smooth,
            dtype=np.float64,
        )

    else:
        guard = np.asarray(
            vru_floor_guard,
            dtype=np.float64,
        )

        if guard.shape != smooth.shape:
            raise ValueError(
                "vru_floor_guard shape mismatch"
            )

        if (
            np.any(guard < 0.0)
            or
            np.any(guard > 1.0)
        ):
            raise ValueError(
                "vru_floor_guard must lie in [0,1]"
            )

    output = np.empty_like(
        smooth,
        dtype=np.float64,
    )

    override_mask = np.zeros(
        smooth.shape,
        dtype=bool,
    )

    previous = current.copy()
    previous_time = 0.0

    for index, horizon in enumerate(
        horizons
    ):
        dt = float(
            horizon
            -
            previous_time
        )

        lower = np.maximum(
            0.0,
            previous
            -
            rho_dim
            *
            dt,
        )

        upper = np.minimum(
            1.0,
            previous
            +
            rho_bright
            *
            dt,
        )

        limited = np.minimum(
            np.maximum(
                smooth[
                    index
                ],
                lower,
            ),
            upper,
        )

        projected = np.maximum(
            limited,
            guard[
                index
            ],
        )

        override = (
            projected
            >
            limited
        )

        override_mask[
            index
        ] = override

        output[
            index
        ] = projected

        previous = projected
        previous_time = float(
            horizon
        )

    count = int(
        np.count_nonzero(
            override_mask
        )
    )

    fraction = (
        float(count)
        /
        float(
            override_mask.size
        )
        if override_mask.size
        else 0.0
    )

    return RateLimitedSchedule(
        illumination=
            np.asarray(
                output,
                dtype=np.float64,
            ),

        safety_override_mask=
            override_mask,

        safety_override_count=
            count,

        safety_override_fraction=
            fraction,
    )
'''


TEST_TEXT = r'''
from __future__ import annotations

import math
import unittest

import numpy as np

from iscai_stage6.adb.class_aware_policy import (
    TYPE_CYCLIST,
    TYPE_PEDESTRIAN,
    TYPE_VEHICLE,
    angular_margin_cells,
    apply_actuation_rate_limit,
    compose_class_aware_illumination,
    compute_class_aware_margin,
    dilate_mask_theta,
    predictive_mask_to_illumination,
    temporal_smooth_schedule,
    threshold_occupancy_counts_strict_k,
)
from iscai_stage6.adb.part_a_reactive import (
    PART_A_LATERAL_SAFETY_MARGIN_M,
)


class StrictThresholdTests(unittest.TestCase):

    def test_strict_k_boundary(self):
        counts = np.asarray(
            [0, 1, 2, 8191, 8192],
            dtype=np.uint32,
        )

        result = (
            threshold_occupancy_counts_strict_k(
                counts,
                k=1,
                sample_count=8192,
            )
        )

        np.testing.assert_array_equal(
            result,
            np.asarray(
                [
                    False,
                    False,
                    True,
                    True,
                    True,
                ]
            ),
        )

    def test_k_zero_keeps_positive_counts(self):
        counts = np.asarray(
            [0, 1, 2],
            dtype=np.uint32,
        )

        result = (
            threshold_occupancy_counts_strict_k(
                counts,
                k=0,
            )
        )

        np.testing.assert_array_equal(
            result,
            [
                False,
                True,
                True,
            ],
        )

    def test_invalid_k_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            threshold_occupancy_counts_strict_k(
                np.zeros(
                    3,
                    dtype=np.uint32,
                ),
                k=8192,
            )


class MarginTests(unittest.TestCase):

    def test_vehicle_uncertainty_and_closing(self):
        result = compute_class_aware_margin(
            TYPE_VEHICLE,
            current_position_H0_m=(
                10.0,
                0.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                -2.0,
                0.0,
                0.0,
            ),
            covariance_H0_m2=np.diag(
                [
                    0.25,
                    4.0,
                    0.01,
                ]
            ),
            horizon_s=0.5,
            base_margin_m=0.5,
            uncertainty_multiplier=1.0,
            motion_multiplier=0.1,
        )

        self.assertAlmostEqual(
            result.lateral_predictive_sigma_m,
            2.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.closing_speed_mps,
            4.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.additional_class_margin_m,
            2.7,
            places=12,
        )

        self.assertAlmostEqual(
            result.total_lateral_margin_m,
            (
                float(
                    PART_A_LATERAL_SAFETY_MARGIN_M
                )
                +
                2.7
            ),
            places=12,
        )

    def test_pedestrian_has_no_motion_term(self):
        first = compute_class_aware_margin(
            TYPE_PEDESTRIAN,
            current_position_H0_m=(
                10.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                -5.0,
                3.0,
            ),
            covariance_H0_m2=np.eye(
                2
            ),
            horizon_s=0.5,
            base_margin_m=0.25,
            uncertainty_multiplier=2.0,
            motion_multiplier=0.0,
        )

        second = compute_class_aware_margin(
            TYPE_PEDESTRIAN,
            current_position_H0_m=(
                10.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                -5.0,
                3.0,
            ),
            covariance_H0_m2=np.eye(
                2
            ),
            horizon_s=0.5,
            base_margin_m=0.25,
            uncertainty_multiplier=2.0,
            motion_multiplier=99.0,
        )

        self.assertAlmostEqual(
            first.additional_class_margin_m,
            second.additional_class_margin_m,
            places=12,
        )

    def test_cyclist_lateral_response(self):
        result = compute_class_aware_margin(
            TYPE_CYCLIST,
            current_position_H0_m=(
                10.0,
                0.0,
            ),
            mean_displacement_H0_m=(
                0.0,
                2.0,
            ),
            covariance_H0_m2=np.eye(
                2
            ),
            horizon_s=0.5,
            base_margin_m=0.0,
            uncertainty_multiplier=1.0,
            motion_multiplier=0.5,
        )

        self.assertAlmostEqual(
            result.cyclist_lateral_rate_mps,
            4.0,
            places=12,
        )

        self.assertAlmostEqual(
            result.additional_class_margin_m,
            2.0,
            places=12,
        )

    def test_unknown_class_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            compute_class_aware_margin(
                "TYPE_UNKNOWN",
                current_position_H0_m=(
                    1.0,
                    0.0,
                ),
                mean_displacement_H0_m=(
                    0.0,
                    0.0,
                ),
                covariance_H0_m2=np.eye(
                    2
                ),
                horizon_s=0.1,
                base_margin_m=0.0,
                uncertainty_multiplier=1.0,
            )

    def test_angular_margin_uses_ceil(self):
        cells = angular_margin_cells(
            total_margin_m=1.0,
            predicted_range_m=10.0,
            theta_step_rad=0.01,
        )

        expected = int(
            math.ceil(
                math.atan2(
                    1.0,
                    10.0,
                )
                /
                0.01
            )
        )

        self.assertEqual(
            cells,
            expected,
        )


class MaskAndIlluminationTests(unittest.TestCase):

    def test_theta_dilation_only(self):
        mask = np.zeros(
            (
                1,
                5,
                4,
            ),
            dtype=bool,
        )

        mask[
            0,
            2,
            1,
        ] = True

        result = dilate_mask_theta(
            mask,
            dilation_cells=1,
        )

        expected = np.zeros_like(
            mask
        )

        expected[
            0,
            1:4,
            1,
        ] = True

        np.testing.assert_array_equal(
            result,
            expected,
        )

    def test_theta_dilation_clips_at_boundary(self):
        mask = np.zeros(
            (
                1,
                3,
                2,
            ),
            dtype=bool,
        )

        mask[
            0,
            0,
            0,
        ] = True

        result = dilate_mask_theta(
            mask,
            dilation_cells=2,
        )

        self.assertTrue(
            np.all(
                result[
                    0,
                    :,
                    0,
                ]
            )
        )

        self.assertFalse(
            np.any(
                result[
                    0,
                    :,
                    1,
                ]
            )
        )

    def test_predictive_illumination_preserves_inactive_theta(self):
        mask = np.zeros(
            (
                1,
                2,
                5,
            ),
            dtype=bool,
        )

        mask[
            0,
            0,
            1,
        ] = True

        ranges = np.asarray(
            [
                0.0,
                5.0,
                10.0,
                15.0,
                20.0,
            ]
        )

        output = predictive_mask_to_illumination(
            mask,
            range_centers_m=ranges,
            intensity_floor=0.2,
        )

        np.testing.assert_allclose(
            output[
                0,
                1,
                :,
            ],
            1.0,
            rtol=0.0,
            atol=0.0,
        )

        self.assertTrue(
            np.all(
                output
                >=
                0.2
            )
        )

        self.assertTrue(
            np.all(
                output
                <=
                1.0
            )
        )

    def test_predictive_illumination_uses_nonzero_floor(self):
        mask = np.zeros(
            (
                1,
                1,
                4,
            ),
            dtype=bool,
        )

        mask[
            0,
            0,
            0,
        ] = True

        ranges = np.asarray(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        )

        output = predictive_mask_to_illumination(
            mask,
            range_centers_m=ranges,
            intensity_floor=0.4,
        )

        self.assertAlmostEqual(
            float(
                output[
                    0,
                    0,
                    0,
                ]
            ),
            0.4,
            places=12,
        )


class CompositionTests(unittest.TestCase):

    def test_vehicle_only_uses_minimum(self):
        vehicle = np.full(
            (
                1,
                1,
                2,
            ),
            0.2,
        )

        result = compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    vehicle,
            },
            class_active_masks={},
            class_floors={},
        )

        np.testing.assert_allclose(
            result.raw_class_aware_illumination,
            vehicle,
            rtol=0.0,
            atol=0.0,
        )

    def test_pedestrian_floor_overrides_vehicle_blackout(self):
        vehicle = np.zeros(
            (
                1,
                1,
                2,
            )
        )

        pedestrian = np.full(
            (
                1,
                1,
                2,
            ),
            0.2,
        )

        active = np.asarray(
            [
                [
                    [
                        True,
                        False,
                    ]
                ]
            ],
            dtype=bool,
        )

        result = compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    vehicle,

                TYPE_PEDESTRIAN:
                    pedestrian,
            },

            class_active_masks={
                TYPE_PEDESTRIAN:
                    active,
            },

            class_floors={
                TYPE_PEDESTRIAN:
                    0.4,
            },
        )

        self.assertAlmostEqual(
            float(
                result.raw_class_aware_illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.4,
            places=12,
        )

        self.assertAlmostEqual(
            float(
                result.raw_class_aware_illumination[
                    0,
                    0,
                    1,
                ]
            ),
            0.0,
            places=12,
        )

    def test_maximum_active_vru_floor_wins(self):
        shape = (
            1,
            1,
            1,
        )

        active = np.ones(
            shape,
            dtype=bool,
        )

        result = compose_class_aware_illumination(
            class_illumination={
                TYPE_VEHICLE:
                    np.zeros(
                        shape
                    ),

                TYPE_PEDESTRIAN:
                    np.zeros(
                        shape
                    ),

                TYPE_CYCLIST:
                    np.zeros(
                        shape
                    ),
            },

            class_active_masks={
                TYPE_PEDESTRIAN:
                    active,

                TYPE_CYCLIST:
                    active,
            },

            class_floors={
                TYPE_PEDESTRIAN:
                    0.3,

                TYPE_CYCLIST:
                    0.6,
            },
        )

        self.assertAlmostEqual(
            float(
                result.raw_class_aware_illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.6,
            places=12,
        )


class TemporalTests(unittest.TestCase):

    def test_temporal_smoothing_exact_first_step(self):
        raw = np.ones(
            (
                1,
                1,
                1,
            )
        )

        current = np.zeros(
            (
                1,
                1,
            )
        )

        output = temporal_smooth_schedule(
            raw,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            time_constant_s=0.1,
        )

        expected = (
            1.0
            -
            math.exp(
                -1.0
            )
        )

        self.assertAlmostEqual(
            float(
                output[
                    0,
                    0,
                    0,
                ]
            ),
            expected,
            places=12,
        )

    def test_temporal_smoothing_is_recursive(self):
        raw = np.asarray(
            [
                [
                    [
                        0.0,
                    ]
                ],
                [
                    [
                        1.0,
                    ]
                ],
            ]
        )

        current = np.ones(
            (
                1,
                1,
            )
        )

        output = temporal_smooth_schedule(
            raw,
            current_illumination=current,
            horizons_s=(
                0.1,
                0.3,
            ),
            time_constant_s=0.2,
        )

        alpha1 = (
            1.0
            -
            math.exp(
                -0.1
                /
                0.2
            )
        )

        first = (
            1.0
            -
            alpha1
        )

        alpha2 = (
            1.0
            -
            math.exp(
                -0.2
                /
                0.2
            )
        )

        expected_second = (
            alpha2
            +
            (
                1.0
                -
                alpha2
            )
            *
            first
        )

        self.assertAlmostEqual(
            float(
                output[
                    1,
                    0,
                    0,
                ]
            ),
            expected_second,
            places=12,
        )


class RateLimitTests(unittest.TestCase):

    def test_finite_dimming_rate(self):
        smooth = np.zeros(
            (
                1,
                1,
                1,
            )
        )

        current = np.ones(
            (
                1,
                1,
            )
        )

        result = apply_actuation_rate_limit(
            smooth,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            rho_dim_per_s=2.0,
            rho_bright_per_s=1.0,
        )

        self.assertAlmostEqual(
            float(
                result.illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.8,
            places=12,
        )

    def test_finite_brightening_rate(self):
        smooth = np.ones(
            (
                1,
                1,
                1,
            )
        )

        current = np.zeros(
            (
                1,
                1,
            )
        )

        result = apply_actuation_rate_limit(
            smooth,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            rho_dim_per_s=2.0,
            rho_bright_per_s=1.0,
        )

        self.assertAlmostEqual(
            float(
                result.illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.1,
            places=12,
        )

    def test_vru_floor_safety_override_is_counted(self):
        smooth = np.zeros(
            (
                1,
                1,
                2,
            )
        )

        current = np.ones(
            (
                1,
                2,
            )
        )

        guard = np.asarray(
            [
                [
                    [
                        0.9,
                        0.0,
                    ]
                ]
            ]
        )

        result = apply_actuation_rate_limit(
            smooth,
            current_illumination=current,
            horizons_s=(
                0.1,
            ),
            rho_dim_per_s=2.0,
            rho_bright_per_s=1.0,
            vru_floor_guard=guard,
        )

        self.assertAlmostEqual(
            float(
                result.illumination[
                    0,
                    0,
                    0,
                ]
            ),
            0.9,
            places=12,
        )

        self.assertEqual(
            result.safety_override_count,
            1,
        )

        self.assertAlmostEqual(
            result.safety_override_fraction,
            0.5,
            places=12,
        )

    def test_invalid_rate_order_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            apply_actuation_rate_limit(
                np.zeros(
                    (
                        1,
                        1,
                        1,
                    )
                ),
                current_illumination=
                    np.ones(
                        (
                            1,
                            1,
                        )
                    ),
                horizons_s=(
                    0.1,
                ),
                rho_dim_per_s=1.0,
                rho_bright_per_s=2.0,
            )


if __name__ == "__main__":
    unittest.main()
'''


class ControlledBlock(RuntimeError):
    pass


def require(
    condition,
    message,
):
    if not bool(condition):
        raise ControlledBlock(
            str(message)
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def exact_seal(
    path: Path,
    expected: str,
    label: str,
):
    require(
        path.is_file(),
        f"Missing {label}: {path}",
    )

    actual = sha256_file(
        path
    )

    require(
        actual == expected,
        (
            f"{label} SHA changed: "
            f"{actual}"
        ),
    )

    print(
        f"{label:35s} = EXACT PASS"
    )

    return actual


def write_new_exact(
    path: Path,
    text: str,
):
    data = textwrap.dedent(
        text
    ).lstrip().encode(
        "utf-8"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing Part3C/1 implementation "
                f"differs and will not be overwritten: {path}"
            ),
        )

    else:
        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_bytes(
            data
        )

        temporary.replace(
            path
        )

    return sha256_file(
        path
    )


def write_json_once(
    path: Path,
    payload,
):
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing deterministic report differs: "
                f"{path}"
            ),
        )

    else:
        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_bytes(
            data
        )

        temporary.replace(
            path
        )

    return sha256_file(
        path
    )


def run(
    command,
):
    process = subprocess.run(
        command,
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "output":
            process.stdout,
    }


try:
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3C/1 — "
        "PREREGISTERED CLASS-AWARE RUNTIME OPERATORS"
    )
    print("=" * 78)

    # ========================================================
    # A. Immutable preregistration seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE PART3B PREREGISTRATION SEAL ====="
    )

    exact_seal(
        PART3B_PREREG,
        EXPECTED[
            "part3b_prereg"
        ],
        "Part3B preregistration",
    )

    exact_seal(
        PART3B_REPORT,
        EXPECTED[
            "part3b_report"
        ],
        "Part3B report",
    )

    exact_seal(
        PART3B_FREEZE,
        EXPECTED[
            "part3b_freeze"
        ],
        "Part3B freeze",
    )

    exact_seal(
        PART3B_HANDOFF,
        EXPECTED[
            "part3b_handoff"
        ],
        "Part3B -> Part3C handoff",
    )

    # Frozen Part-A components must remain byte-identical.
    exact_seal(
        PART_A_REACTIVE,
        EXPECTED[
            "part_a_reactive"
        ],
        "part_a_reactive.py",
    )

    exact_seal(
        ILLUMINATION,
        EXPECTED[
            "illumination"
        ],
        "illumination.py",
    )

    exact_seal(
        PREDICTIVE_MASK,
        EXPECTED[
            "predictive_mask"
        ],
        "predictive_mask.py",
    )

    prereg = json.loads(
        PART3B_PREREG.read_text(
            encoding="utf-8"
        )
    )

    require(
        prereg[
            "scientific_boundary"
        ][
            "numeric_policy_selected"
        ]
        is False,
        (
            "Part3B unexpectedly selected "
            "numeric policy."
        ),
    )

    require(
        prereg[
            "scientific_boundary"
        ][
            "formal_outcomes_read"
        ]
        is False,
        (
            "Part3B formal boundary changed."
        ),
    )

    print(
        "numeric policy selection = FORBIDDEN"
    )

    print(
        "formal outcomes read = NO"
    )

    # ========================================================
    # B. Write isolated runtime implementation
    # ========================================================

    print()
    print(
        "===== B. WRITE NEW ISOLATED RUNTIME MODULE ====="
    )

    source_sha = write_new_exact(
        SRC,
        MODULE_TEXT,
    )

    print(
        "module =",
        SRC,
    )

    print(
        "module SHA256 =",
        source_sha,
    )

    print(
        "existing frozen modules modified = NO"
    )

    # ========================================================
    # C. Write operator tests
    # ========================================================

    print()
    print(
        "===== C. WRITE PART3C/1 OPERATOR TESTS ====="
    )

    test_sha = write_new_exact(
        TEST,
        TEST_TEXT,
    )

    print(
        "test file =",
        TEST,
    )

    print(
        "test SHA256 =",
        test_sha,
    )

    # ========================================================
    # D. Compile
    # ========================================================

    print()
    print(
        "===== D. PYTHON COMPILE GATE ====="
    )

    compile_result = run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(
                SRC
            ),
            str(
                TEST
            ),
        ]
    )

    print(
        compile_result[
            "output"
        ]
    )

    require(
        compile_result[
            "returncode"
        ]
        ==
        0,
        "Part3C/1 compile failed.",
    )

    print(
        "compile = PASS"
    )

    # ========================================================
    # E. Focused operator tests
    # ========================================================

    print()
    print(
        "===== E. FOCUSED PART3C/1 UNIT TESTS ====="
    )

    focused = run(
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            (
                "tests."
                "test_block66_part3c1_class_aware_policy"
            ),
        ]
    )

    print(
        focused[
            "output"
        ]
    )

    require(
        focused[
            "returncode"
        ]
        ==
        0,
        (
            "Focused Part3C/1 tests failed."
        ),
    )

    focused_match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        focused[
            "output"
        ],
    )

    require(
        focused_match
        is not None,
        (
            "Could not parse focused "
            "unit-test count."
        ),
    )

    focused_count = int(
        focused_match.group(1)
    )

    require(
        focused_count
        ==
        21,
        (
            "Expected exactly 21 new "
            f"Part3C/1 tests, got {focused_count}."
        ),
    )

    print(
        "focused tests = PASS |",
        focused_count,
    )

    # ========================================================
    # F. Runtime API smoke
    # ========================================================

    print()
    print(
        "===== F. RUNTIME API SMOKE ====="
    )

    from iscai_stage6.adb.class_aware_policy import (
        angular_margin_cells,
        apply_actuation_rate_limit,
        compose_class_aware_illumination,
        compute_class_aware_margin,
        dilate_mask_theta,
        predictive_mask_to_illumination,
        temporal_smooth_schedule,
        threshold_occupancy_counts_strict_k,
    )

    callables = (
        threshold_occupancy_counts_strict_k,
        compute_class_aware_margin,
        angular_margin_cells,
        dilate_mask_theta,
        predictive_mask_to_illumination,
        compose_class_aware_illumination,
        temporal_smooth_schedule,
        apply_actuation_rate_limit,
    )

    for function in callables:
        print(
            f"{function.__name__:45s}",
            "=",
            function.__module__,
        )

    print(
        "strict-k operator       = IMPLEMENTED"
    )

    print(
        "predictive margin       = IMPLEMENTED"
    )

    print(
        "class composition       = IMPLEMENTED"
    )

    print(
        "overlap VRU priority    = IMPLEMENTED"
    )

    print(
        "temporal smoothing      = IMPLEMENTED"
    )

    print(
        "actuation-rate limiter  = IMPLEMENTED"
    )

    # ========================================================
    # G. Frozen-source continuity after implementation
    # ========================================================

    print()
    print(
        "===== G. FROZEN PART-A CONTINUITY RECHECK ====="
    )

    exact_seal(
        PART_A_REACTIVE,
        EXPECTED[
            "part_a_reactive"
        ],
        "part_a_reactive.py",
    )

    exact_seal(
        ILLUMINATION,
        EXPECTED[
            "illumination"
        ],
        "illumination.py",
    )

    exact_seal(
        PREDICTIVE_MASK,
        EXPECTED[
            "predictive_mask"
        ],
        "predictive_mask.py",
    )

    print(
        "original reactive baseline modified = NO"
    )

    # ========================================================
    # H. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== H. FULL STAGE6 REGRESSION ====="
    )

    regression = run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            "test_*.py",
        ]
    )

    print(
        "\n".join(
            regression[
                "output"
            ].splitlines()[
                -40:
            ]
        )
    )

    require(
        regression[
            "returncode"
        ]
        ==
        0,
        "Full Stage6 regression failed.",
    )

    regression_match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        regression[
            "output"
        ],
    )

    require(
        regression_match
        is not None,
        (
            "Could not parse Stage6 "
            "regression count."
        ),
    )

    regression_count = int(
        regression_match.group(1)
    )

    require(
        regression_count
        ==
        166,
        (
            "Expected 145 frozen tests + "
            "21 new tests = 166; "
            f"got {regression_count}."
        ),
    )

    print(
        "Stage6 regression = PASS |",
        regression_count,
    )

    # ========================================================
    # I. Report and handoff
    # ========================================================

    print()
    print(
        "===== I. PART3C/1 REPORT + HANDOFF ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part3C/1",

        "status":
            (
                "PASS_PREREGISTERED_CLASS_AWARE_"
                "RUNTIME_OPERATORS_IMPLEMENTED"
            ),

        "Part3B_preregistration": {
            "path":
                str(
                    PART3B_PREREG
                ),

            "sha256":
                EXPECTED[
                    "part3b_prereg"
                ],
        },

        "implementation": {
            "path":
                str(
                    SRC
                ),

            "sha256":
                source_sha,

            "new_file":
                True,

            "frozen_PartA_source_modified":
                False,
        },

        "tests": {
            "path":
                str(
                    TEST
                ),

            "sha256":
                test_sha,

            "focused_tests":
                focused_count,

            "full_Stage6_tests":
                regression_count,
        },

        "implemented_runtime_components": [
            "exact_strict_integer_k_threshold",
            "class_aware_predictive_margin_operator",
            "theta_only_predictive_mask_dilation",
            (
                "predictive_mask_to_preserved_"
                "PartA_raised_cosine_illumination"
            ),
            "class_aware_policy_composition",
            "spatial_VRU_floor_guard",
            "class_overlap_priority_rule",
            "temporal_smoothing_operator",
            "actuation_rate_limit_operator",
            "rate_limit_safety_override_accounting",
        ],

        "PartA_continuity": {
            "raised_cosine_reused":
                True,

            "generic_lateral_margin_preserved":
                True,

            "generic_radial_margin_preserved":
                True,

            "original_reactive_baseline_modified":
                False,
        },

        "actuator_boundary": {
            "axes":
                [
                    "theta",
                    "range",
                ],

            "vertical_DOF":
                False,

            "face_height_pixel_control":
                False,
        },

        "scientific_boundary": {
            "policy_sweep_executed":
                False,

            "numeric_gamma_selected":
                False,

            "numeric_margin_selected":
                False,

            "numeric_floor_selected":
                False,

            "numeric_smoothing_selected":
                False,

            "numeric_rate_selected":
                False,

            "formal_outcomes_read":
                False,

            "new_model_forward":
                False,

            "new_MC_sampling":
                False,

            "training":
                False,

            "recalibration":
                False,
        },

        "next":
            (
                "BLOCK6.6_PART3C2_MATERIALIZE_"
                "VEHICLE_GEOMETRY_SURROGATE_EVIDENCE_"
                "AND_INTEGRATION_SMOKE_WITHOUT_SWEEP"
            ),
    }

    report_sha = write_json_once(
        REPORT,
        report_payload,
    )

    handoff_payload = {
        "stage":
            6,

        "from":
            "6.6-Part3C/1",

        "to":
            "6.6-Part3C/2",

        "status":
            (
                "PASS_READY_FOR_VEHICLE_SURROGATE_"
                "EVIDENCE_AND_CONTROLLER_INTEGRATION"
            ),

        "runtime_module": {
            "path":
                str(
                    SRC
                ),

            "sha256":
                source_sha,
        },

        "Part3C2_allowed": [
            (
                "use the frozen N8192 development "
                "trajectory sample route"
            ),
            (
                "materialize geometry-based "
                "vehicle front-upper/rear-upper "
                "surrogate evidence"
            ),
            (
                "exercise class-aware runtime "
                "on synthetic/development evidence "
                "without selecting parameters"
            ),
            "add tests",
        ],

        "Part3C2_forbidden": [
            "class-policy parameter sweep",
            "numeric policy selection",
            "formal outcome read",
            "formal tuning",
            "new model inference",
            "new posterior calibration",
            "new Monte Carlo semantics",
            "future GT controller input",
            "actual windshield location claim",
            "actual mirror location claim",
            "vertical pixel actuator claim",
        ],

        "numeric_policy_selected":
            False,

        "formal_outcomes_read":
            False,
    }

    handoff_sha = write_json_once(
        HANDOFF,
        handoff_payload,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # J. Final
    # ========================================================

    print()
    print("=" * 78)
    print(
        "BLOCK 6.6 PART3C/1 — FINAL"
    )
    print("=" * 78)

    print(
        "STATUS = "
        "PASS_PREREGISTERED_CLASS_AWARE_RUNTIME_OPERATORS_IMPLEMENTED"
    )

    print(
        "strict integer-k threshold = IMPLEMENTED"
    )

    print(
        "predictive class margin    = IMPLEMENTED"
    )

    print(
        "Part-A generic margin      = PRESERVED"
    )

    print(
        "theta-only mask dilation   = IMPLEMENTED"
    )

    print(
        "raised-cosine conversion   = IMPLEMENTED / REUSED"
    )

    print(
        "class-aware composition    = IMPLEMENTED"
    )

    print(
        "VRU overlap floor guard    = IMPLEMENTED"
    )

    print(
        "temporal smoothing         = IMPLEMENTED"
    )

    print(
        "finite actuation rate      = IMPLEMENTED"
    )

    print(
        "safety override accounting = IMPLEMENTED"
    )

    print(
        "vertical actuator DOF      = NO"
    )

    print(
        "focused tests              = PASS |",
        focused_count,
    )

    print(
        "Stage6 regression          = PASS |",
        regression_count,
    )

    print(
        "policy sweep executed      = NO"
    )

    print(
        "numeric policy selected    = NO"
    )

    print(
        "formal outcomes read       = NO"
    )

    print(
        "new model forward          = NO"
    )

    print(
        "new MC sampling            = NO"
    )

    print(
        "module SHA256              =",
        source_sha,
    )

    print(
        "test SHA256                =",
        test_sha,
    )

    print(
        "report SHA256              =",
        report_sha,
    )

    print(
        "handoff SHA256             =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.6 PART3C/2 "
        "VEHICLE SURROGATE EVIDENCE + INTEGRATION"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 78)


except BaseException as exc:
    print()
    print("=" * 78)
    print(
        "BLOCK6.6 PART3C/1 — CONTROLLED BLOCK"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=25
    )

    print()
    print(
        "Part3B preregistration changed = NO"
    )

    print(
        "frozen Part-A source modified  = NO"
    )

    print(
        "policy sweep executed          = NO"
    )

    print(
        "numeric policy selected        = NO"
    )

    print(
        "formal outcomes read           = NO"
    )

    print(
        "terminal remains open          = YES"
    )

# No sys.exit().

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

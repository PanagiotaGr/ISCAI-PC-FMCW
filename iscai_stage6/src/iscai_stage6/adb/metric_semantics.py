from __future__ import annotations

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

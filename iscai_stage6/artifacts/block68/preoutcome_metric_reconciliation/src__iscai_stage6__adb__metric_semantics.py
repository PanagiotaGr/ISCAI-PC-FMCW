from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# ============================================================
# Frozen Stage6 metric conventions
#
# The PDF specifies WHICH ADB metrics are mandatory but does
# not define a unique numerical formula for every one.
#
# These formulas are therefore Stage6 implementation choices,
# frozen BEFORE development-bound selection and BEFORE formal
# evaluation.
# ============================================================

INTENSITY_MIN = 0.0
INTENSITY_MAX = 1.0

# Normalized intensity change considered one discrete
# flicker/change event.
FLICKER_DELTA_THRESHOLD = 0.10

EPS = 1.0e-12


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float


def _bool_array(
    value,
    *,
    name: str,
) -> np.ndarray:

    array = np.asarray(
        value
    )

    if array.dtype != np.bool_:
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

    if np.any(
        array
        <
        INTENSITY_MIN
    ) or np.any(
        array
        >
        INTENSITY_MAX
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

    if len(shapes) != 1:
        raise ValueError(
            "All arrays must have identical shape"
        )


def _region_mean(
    values: np.ndarray,
    region: np.ndarray,
    *,
    empty_value: float,
) -> float:

    count = int(
        np.count_nonzero(
            region
        )
    )

    if count == 0:
        return float(
            empty_value
        )

    return float(
        np.mean(
            values[
                region
            ]
        )
    )


# ============================================================
# 1. Mask IoU
# ============================================================

def mask_iou(
    candidate_mask,
    oracle_mask,
) -> float:
    """
    Binary mask IoU.

    mask=True means protected/dimmed.

    If both masks are empty, IoU=1 because they agree exactly.
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
    vehicle_required_region,
) -> float:
    """
    Fraction of required vehicle-protection cells that the
    candidate ADB fails to protect.

    Lower is better.

        missed required vehicle cells
        --------------------------------
        required vehicle cells

    Empty vehicle region returns 0 and is excluded separately
    from class-support counts by the evaluator.
    """

    candidate = _bool_array(
        candidate_mask,
        name="candidate_mask",
    )

    vehicle = _bool_array(
        vehicle_required_region,
        name="vehicle_required_region",
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
        return 0.0

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
    vehicle_required_region,
) -> float:
    """
    Mean normalized emitted intensity inside the constructed
    vehicle glare-protection region.

    Lower is better.

    Empty vehicle region returns 0.
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    vehicle = _bool_array(
        vehicle_required_region,
        name="vehicle_required_region",
    )

    _same_shape(
        light,
        vehicle,
    )

    return _region_mean(
        light,
        vehicle,
        empty_value=0.0,
    )


# ============================================================
# 4. Over-masking area
# ============================================================

def over_masking_area(
    candidate_mask,
    oracle_mask,
    road_roi,
) -> float:
    """
    Fraction of road-ROI cells unnecessarily masked relative
    to the constructed oracle future mask.

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

    road = _bool_array(
        road_roi,
        name="road_roi",
    )

    _same_shape(
        candidate,
        oracle,
        road,
    )

    denominator = int(
        np.count_nonzero(
            road
        )
    )

    if denominator == 0:
        raise ValueError(
            "road_roi must contain at least one cell"
        )

    unnecessary = (
        road
        &
        candidate
        &
        ~oracle
    )

    return float(
        np.count_nonzero(
            unnecessary
        )
        /
        denominator
    )


# ============================================================
# 5. Road illumination retention
# ============================================================

def road_illumination_retention(
    intensity,
    road_roi,
) -> float:
    """
    Mean normalized illumination retained inside road ROI.

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

    return _region_mean(
        light,
        road,
        empty_value=0.0,
    )


# ============================================================
# 6/7. VRU visibility proxies
# ============================================================

def pedestrian_visibility_proxy(
    intensity,
    pedestrian_region,
) -> float:
    """
    Mean normalized illumination retained over the constructed
    pedestrian visibility region.

    Higher is better.

    Empty pedestrian support returns NaN so the evaluator
    cannot silently interpret an absent pedestrian as perfect
    visibility.
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    region = _bool_array(
        pedestrian_region,
        name="pedestrian_region",
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

    return _region_mean(
        light,
        region,
        empty_value=float("nan"),
    )


def cyclist_visibility_proxy(
    intensity,
    cyclist_region,
) -> float:
    """
    Mean normalized illumination retained over the constructed
    cyclist visibility region.

    Higher is better.

    Empty cyclist support returns NaN.
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    region = _bool_array(
        cyclist_region,
        name="cyclist_region",
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

    return _region_mean(
        light,
        region,
        empty_value=float("nan"),
    )


# ============================================================
# 8. False dimming
# ============================================================

def false_dimming(
    intensity,
    oracle_mask,
    road_roi,
) -> float:
    """
    Mean unnecessary dimming magnitude on road cells that the
    constructed oracle does NOT request to protect.

        mean(1 - intensity)

    over road_roi AND NOT oracle_mask.

    Lower is better.
    """

    light = _intensity_array(
        intensity,
        name="intensity",
    )

    oracle = _bool_array(
        oracle_mask,
        name="oracle_mask",
    )

    road = _bool_array(
        road_roi,
        name="road_roi",
    )

    _same_shape(
        light,
        oracle,
        road,
    )

    evaluation_region = (
        road
        &
        ~oracle
    )

    if not np.any(
        evaluation_region
    ):
        return 0.0

    dimming = (
        1.0
        -
        light
    )

    return _region_mean(
        dimming,
        evaluation_region,
        empty_value=0.0,
    )


# ============================================================
# 9. Temporal smoothness
# ============================================================

def temporal_smoothness(
    intensity_sequence,
) -> float:
    """
    1 - mean absolute normalized illumination change between
    consecutive frames/candidate horizons.

    Higher is better; 1 means perfectly unchanged.

    Expected shape:
        [T, ...illumination-grid dimensions...]
    """

    light = _intensity_array(
        intensity_sequence,
        name="intensity_sequence",
    )

    if light.ndim < 2:
        raise ValueError(
            "intensity_sequence must have time + grid dimensions"
        )

    if light.shape[0] < 2:
        raise ValueError(
            "temporal metrics require at least two frames"
        )

    delta = np.abs(
        np.diff(
            light,
            axis=0,
        )
    )

    value = (
        1.0
        -
        float(
            np.mean(
                delta
            )
        )
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


# ============================================================
# 10. Flicker / change rate
# ============================================================

def flicker_change_rate(
    intensity_sequence,
    *,
    delta_threshold: float = (
        FLICKER_DELTA_THRESHOLD
    ),
) -> float:
    """
    Fraction of cell-transitions whose normalized intensity
    changes by at least the frozen threshold.

    Lower is better.
    """

    light = _intensity_array(
        intensity_sequence,
        name="intensity_sequence",
    )

    if light.ndim < 2:
        raise ValueError(
            "intensity_sequence must have time + grid dimensions"
        )

    if light.shape[0] < 2:
        raise ValueError(
            "flicker metric requires at least two frames"
        )

    threshold = float(
        delta_threshold
    )

    if (
        not math.isfinite(
            threshold
        )
        or
        threshold <= 0.0
        or
        threshold > 1.0
    ):
        raise ValueError(
            "delta_threshold must be in (0,1]"
        )

    delta = np.abs(
        np.diff(
            light,
            axis=0,
        )
    )

    return float(
        np.mean(
            delta
            >=
            threshold
        )
    )


# ============================================================
# 11. Energy consumption
# ============================================================

def normalized_energy_consumption(
    intensity_sequence,
    *,
    valid_energy_roi=None,
) -> float:
    """
    Normalized optical-energy proxy relative to full-intensity
    illumination over the same valid headlamp cells.

    0 = no emitted normalized illumination.
    1 = full normalized illumination everywhere.

    This is explicitly a normalized proxy unless a physical
    optical-power scale is supplied by a later frozen adapter.
    """

    light = _intensity_array(
        intensity_sequence,
        name="intensity_sequence",
    )

    if valid_energy_roi is None:
        return float(
            np.mean(
                light
            )
        )

    roi = _bool_array(
        valid_energy_roi,
        name="valid_energy_roi",
    )

    # Permit static ROI matching spatial dimensions or a
    # sequence ROI matching the full sequence.
    if roi.shape == light.shape:
        expanded = roi

    elif (
        light.ndim >= 2
        and
        roi.shape == light.shape[1:]
    ):
        expanded = np.broadcast_to(
            roi,
            light.shape,
        )

    else:
        raise ValueError(
            "valid_energy_roi shape incompatible with intensity_sequence"
        )

    if not np.any(
        expanded
    ):
        raise ValueError(
            "valid_energy_roi must contain at least one cell"
        )

    return float(
        np.mean(
            light[
                expanded
            ]
        )
    )


# ============================================================
# 12. Actuation latency
# ============================================================

def summarize_actuation_latency_ms(
    latency_samples_ms,
) -> LatencySummary:
    """
    Frozen latency summary from non-negative wall-clock samples.
    """

    values = np.asarray(
        latency_samples_ms,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or
        values.size == 0
    ):
        raise ValueError(
            "latency_samples_ms must be a non-empty 1-D sequence"
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

        mean_ms=float(
            np.mean(
                values
            )
        ),

        median_ms=float(
            np.median(
                values
            )
        ),

        p95_ms=float(
            np.percentile(
                values,
                95.0,
            )
        ),

        max_ms=float(
            np.max(
                values
            )
        ),
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOUND_SCALE = 1.96
BOUND_FLOOR = 0.02
BOUND_CAP = 0.05

OVERMASK_REQUIRED_SUPPORT = 120
VRU_MINIMUM_SUPPORT = 20

SUPPORTED_BOUND_METRICS = (
    "over_masking_area",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
)


@dataclass(frozen=True)
class ReactiveVariabilityBound:
    metric_name: str
    support_count: int
    reactive_mean: float
    reactive_sample_std: float
    reactive_standard_error: float
    scaled_standard_error: float
    frozen_delta: float
    semantics: str


def _validated_reactive_values(
    reactive_values,
) -> np.ndarray:

    values = np.asarray(
        tuple(reactive_values),
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "reactive_values must be one-dimensional"
        )

    if values.size == 0:
        raise ValueError(
            "reactive_values cannot be empty"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "reactive_values must contain only "
            "explicitly supported finite observations"
        )

    if (
        np.any(values < 0.0)
        or
        np.any(values > 1.0)
    ):
        raise ValueError(
            "reactive metric values must lie in [0,1]"
        )

    return values


def derive_reactive_only_noninferiority_delta(
    metric_name: str,
    reactive_values,
) -> ReactiveVariabilityBound:
    """
    Frozen Block6.8 pre-outcome rule.

    The non-inferiority tolerance is derived from the
    ORIGINAL REACTIVE comparator only:

        SE_R = sample_std(R) / sqrt(n)
        raw  = 1.96 * SE_R
        delta = clip(raw, 0.02, 0.05)

    Predictive/class-aware values are deliberately absent
    from this API, preventing target-system outcomes from
    influencing the tolerance.

    The 1.96 multiplier is a frozen variability scale.
    This function does NOT claim that delta is a formal
    confidence interval.
    """

    metric = str(metric_name)

    if metric not in SUPPORTED_BOUND_METRICS:
        raise ValueError(
            f"unsupported bound metric: {metric}"
        )

    values = _validated_reactive_values(
        reactive_values
    )

    n = int(values.size)

    if metric == "over_masking_area":

        if n != OVERMASK_REQUIRED_SUPPORT:
            raise ValueError(
                "over_masking_area requires exactly "
                f"{OVERMASK_REQUIRED_SUPPORT} "
                "development scenario observations"
            )

    else:

        if n < VRU_MINIMUM_SUPPORT:
            raise ValueError(
                f"{metric} requires at least "
                f"{VRU_MINIMUM_SUPPORT} supported "
                "development observations"
            )

    sample_std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = float(
        sample_std
        /
        np.sqrt(
            float(n)
        )
    )

    scaled = float(
        BOUND_SCALE
        *
        standard_error
    )

    delta = float(
        min(
            BOUND_CAP,
            max(
                BOUND_FLOOR,
                scaled,
            ),
        )
    )

    return ReactiveVariabilityBound(
        metric_name=metric,
        support_count=n,
        reactive_mean=float(
            np.mean(values)
        ),
        reactive_sample_std=sample_std,
        reactive_standard_error=standard_error,
        scaled_standard_error=scaled,
        frozen_delta=delta,
        semantics=(
            "reactive_comparator_only_variability_scaled_"
            "absolute_noninferiority_tolerance"
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
import math

from .class_aware_policy import (
    MarginComputation,
    angular_margin_cells,
)


@dataclass(frozen=True)
class PredictiveClassMarginBinding:
    """
    Frozen Block6.7 runtime binding.

    IMPORTANT SEMANTICS
    -------------------
    The Part-A generic lateral safety margin has already been
    represented by the Part-A reactive shadow-zone geometry.

    Therefore the Stage-6 predictive class-aware angular
    dilation MUST use only:

        MarginComputation.additional_class_margin_m

    and MUST NOT apply:

        MarginComputation.total_lateral_margin_m

    as a second dilation, because doing so would re-apply the
    frozen Part-A generic safety margin.

    `total_margin_reference_cells` is diagnostic only.
    """

    actor_class: str

    part_a_generic_lateral_margin_m: float

    additional_class_margin_m: float

    total_lateral_margin_m: float

    predictive_dilation_margin_m: float

    predicted_range_m: float

    theta_step_rad: float

    predictive_dilation_cells: int

    total_margin_reference_cells: int

    generic_margin_reapplied: bool


def _finite(
    value,
    *,
    name: str,
) -> float:

    result = float(
        value
    )

    if not math.isfinite(
        result
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    return result


def bind_predictive_class_margin(
    margin: MarginComputation,
    *,
    theta_step_rad: float,
) -> PredictiveClassMarginBinding:
    """
    Bind frozen class-aware margin computation to the
    predictive ADB angular-dilation stage.

    The argument name of `angular_margin_cells` is
    `total_margin_m` because that helper converts an arbitrary
    metric margin to cells.

    At this Stage-6 binding boundary the metric quantity
    intentionally supplied to that helper is ONLY m_extra:

        margin.additional_class_margin_m

    The total margin remains available as a diagnostic
    reference and is never used for predictive re-dilation.
    """

    if not isinstance(
        margin,
        MarginComputation,
    ):
        raise TypeError(
            "margin must be MarginComputation"
        )

    generic = _finite(
        margin.part_a_generic_lateral_margin_m,
        name=(
            "part_a_generic_lateral_margin_m"
        ),
    )

    extra = _finite(
        margin.additional_class_margin_m,
        name="additional_class_margin_m",
    )

    total = _finite(
        margin.total_lateral_margin_m,
        name="total_lateral_margin_m",
    )

    actor_range = _finite(
        margin.predicted_range_m,
        name="predicted_range_m",
    )

    theta_step = _finite(
        theta_step_rad,
        name="theta_step_rad",
    )

    if generic < 0.0:
        raise ValueError(
            "Part-A generic margin must be >= 0"
        )

    if extra < 0.0:
        raise ValueError(
            "additional class margin must be >= 0"
        )

    if total < 0.0:
        raise ValueError(
            "total lateral margin must be >= 0"
        )

    if actor_range < 0.0:
        raise ValueError(
            "predicted range must be >= 0"
        )

    if theta_step <= 0.0:
        raise ValueError(
            "theta_step_rad must be > 0"
        )

    expected_total = (
        generic
        +
        extra
    )

    if not math.isclose(
        total,
        expected_total,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "MarginComputation contract violated: "
            "total_lateral_margin_m != "
            "part_a_generic_lateral_margin_m + "
            "additional_class_margin_m"
        )

    # --------------------------------------------------------
    # SCIENTIFICALLY BINDING LINE
    #
    # Predictive angular dilation receives ONLY m_extra.
    # The generic Part-A safety margin is not reapplied.
    # --------------------------------------------------------

    predictive_cells = angular_margin_cells(
        total_margin_m=extra,
        predicted_range_m=actor_range,
        theta_step_rad=theta_step,
    )

    # Diagnostic only. Never used to drive ADB actuation.
    total_reference_cells = angular_margin_cells(
        total_margin_m=total,
        predicted_range_m=actor_range,
        theta_step_rad=theta_step,
    )

    return PredictiveClassMarginBinding(
        actor_class=str(
            margin.actor_class
        ),

        part_a_generic_lateral_margin_m=generic,

        additional_class_margin_m=extra,

        total_lateral_margin_m=total,

        predictive_dilation_margin_m=extra,

        predicted_range_m=actor_range,

        theta_step_rad=theta_step,

        predictive_dilation_cells=int(
            predictive_cells
        ),

        total_margin_reference_cells=int(
            total_reference_cells
        ),

        generic_margin_reapplied=False,
    )

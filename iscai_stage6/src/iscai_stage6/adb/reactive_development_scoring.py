from __future__ import annotations

from typing import Sequence

import numpy as np


REACTIVE_EVALUATION_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)


def _validate_horizons(
    horizons_s: Sequence[float],
) -> tuple[float, ...]:

    values = tuple(
        float(value)
        for value in horizons_s
    )

    if values != REACTIVE_EVALUATION_HORIZONS_S:
        raise ValueError(
            "Reactive development scoring is frozen to "
            "horizons (0.1, 0.3, 0.5, 1.0) seconds."
        )

    return values


def _validate_current_illumination(
    current_illumination,
) -> np.ndarray:

    current = np.asarray(
        current_illumination,
        dtype=np.float64,
    )

    if current.ndim != 2:
        raise ValueError(
            "current_illumination must be a 2-D "
            "[theta,range] Part-A reactive map"
        )

    if not np.all(
        np.isfinite(current)
    ):
        raise ValueError(
            "current_illumination contains non-finite values"
        )

    if (
        np.any(current < 0.0)
        or
        np.any(current > 1.0)
    ):
        raise ValueError(
            "current_illumination must lie in [0,1]"
        )

    return current


def hold_current_reactive_schedule(
    current_illumination,
    *,
    horizons_s: Sequence[float] = (
        0.1,
        0.3,
        0.5,
        1.0,
    ),
) -> np.ndarray:
    """
    Frozen Block6.8 single-decision evaluator semantics.

    The original reactive ADB controller produces one causal
    current-state Part-A-style illumination action at t0.

    For future-horizon comparison against predictive ADB,
    that already-made action is held unchanged over the four
    frozen Stage6 evaluation horizons.

    No future observation, future state, oracle geometry,
    future trajectory, or posterior update enters this
    operation.

    This is an evaluator representation of the t0 reactive
    decision. It is NOT a simulated closed-loop reactive
    controller with future re-observation.
    """

    _validate_horizons(
        horizons_s
    )

    current = (
        _validate_current_illumination(
            current_illumination
        )
    )

    schedule = np.repeat(
        current[
            None,
            :,
            :,
        ],
        repeats=len(
            REACTIVE_EVALUATION_HORIZONS_S
        ),
        axis=0,
    )

    return np.asarray(
        schedule,
        dtype=np.float64,
    ).copy()


def binary_dim_support(
    final_illumination_schedule,
) -> np.ndarray:
    """
    Reconciled Block6.6 metric semantics:

        D_tau = [I_final_tau < 1.0]

    Equality at exactly 1.0 is NOT dimmed support.
    """

    values = np.asarray(
        final_illumination_schedule,
        dtype=np.float64,
    )

    if values.ndim != 3:
        raise ValueError(
            "final_illumination_schedule must have shape "
            "[horizon,theta,range]"
        )

    if values.shape[0] != len(
        REACTIVE_EVALUATION_HORIZONS_S
    ):
        raise ValueError(
            "final illumination must contain exactly "
            "four frozen horizons"
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "final illumination contains non-finite values"
        )

    if (
        np.any(values < 0.0)
        or
        np.any(values > 1.0)
    ):
        raise ValueError(
            "final illumination must lie in [0,1]"
        )

    return (
        values
        <
        1.0
    )


def validate_supported_reactive_metric_values(
    values,
    *,
    metric_name: str,
) -> np.ndarray:
    """
    Validate an already support-filtered reactive metric
    vector before it is passed to the frozen NI-bound rule.

    This helper does not remove NaN/Inf values silently.
    Unsupported scenarios must be excluded explicitly by the
    evaluator before this function is called.
    """

    array = np.asarray(
        tuple(values),
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{metric_name}: metric values must be 1-D"
        )

    if array.size == 0:
        raise ValueError(
            f"{metric_name}: no supported observations"
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            f"{metric_name}: non-finite supported value"
        )

    if (
        np.any(array < 0.0)
        or
        np.any(array > 1.0)
    ):
        raise ValueError(
            f"{metric_name}: normalized values "
            "must lie in [0,1]"
        )

    return array


def scientific_input_contract() -> dict:
    """
    Machine-readable semantic boundary used by tests and
    the development runner.
    """

    return {
        "baseline":
            "original_reactive_ADB",

        "decision_time":
            "t0_current_causal_state",

        "controller_future_input":
            False,

        "future_reobservation":
            False,

        "future_controller_refresh":
            False,

        "future_truth_controller_input":
            False,

        "posterior_controller_input":
            False,

        "constructed_oracle_controller_input":
            False,

        "horizons_s":
            list(
                REACTIVE_EVALUATION_HORIZONS_S
            ),

        "schedule_semantics":
            (
                "hold_t0_current_reactive_PartA_action_"
                "unchanged_across_future_scoring_horizons"
            ),

        "metric_dim_support":
            "D_tau = [I_final_tau < 1.0]",

        "role":
            "single_decision_future_evaluation_comparator",
    }

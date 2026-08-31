from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Generic, TypeVar

from iscai_stage5.adaptive_topk_temporal import (
    AdaptiveTemporalDecision,
)

from iscai_stage5.optical_link import (
    EffectiveRateResult,
    effective_rate,
)


T = TypeVar(
    "T"
)


PARTA_CHIRP_DURATION_S = (
    10e-6
)

WOMD_SCENE_INTERVAL_S = (
    0.1
)

STAGE5_BEAM_PROBE_DURATION_S = (
    PARTA_CHIRP_DURATION_S
)

STAGE5_RATE_ACCOUNTING_FRAME_S = (
    WOMD_SCENE_INTERVAL_S
)

BEAM_PROBE_TIMING_PROVENANCE = (
    "constructed_Stage5_baseline_"
    "one_frozen_PartA_PC_FMCW_chirp_"
    "per_beam_probe"
)

RATE_FRAME_TIMING_PROVENANCE = (
    "WOMD_100ms_scene_controller_"
    "accounting_cycle_not_measured_"
    "optical_communication_frame"
)

HIGH_RATE_GROUND_TRUTH_POLICY = (
    "sub_100ms_controller_targets_use_"
    "causal_model_based_propagation_or_"
    "interpolation_not_annotated_ms_ground_truth"
)

PDF_LATENCY_FORMULA = (
    "tau_total_equals_sensing_plus_"
    "preprocessing_plus_tracking_plus_"
    "inference_plus_beam_probing_plus_"
    "actuation"
)

INFERENCE_DECOMPOSITION = (
    "tau_inference_equals_predictor_"
    "inference_plus_beam_selection"
)

RUNTIME_CLOCK = (
    "time.perf_counter_ns"
)

GPU_TIMING_POLICY = (
    "synchronize_before_and_after_GPU_segment"
)

FROZEN_POSTERIOR_HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)


@dataclass(
    frozen=True
)
class LatencyComponents:
    sensing_s: float

    preprocessing_s: float

    tracking_s: float

    predictor_inference_s: float

    beam_selection_s: float

    actuation_s: float

    data_loading_s: float = 0.0


@dataclass(
    frozen=True
)
class LatencyBudget:
    components: LatencyComponents

    probing_beam_count: int

    beam_probe_time_s: float

    beam_probing_s: float

    inference_total_s: float

    online_total_s: float

    data_loading_s: float

    wall_with_loading_s: float

    deadline_s: float

    deadline_met: bool


@dataclass(
    frozen=True
)
class LatencyPredictionRequest:
    target_offset_s: float

    target_time_semantics: str

    exact_frozen_posterior_horizon_s: float | None

    model_based_refinement_required: bool

    annotated_millisecond_ground_truth_used: bool


@dataclass(
    frozen=True
)
class TimedCallResult(
    Generic[
        T
    ]
):
    value: T

    elapsed_s: float


def _nonnegative_finite(
    value,
    *,
    name,
):
    result = float(
        value
    )

    if (
        not math.isfinite(
            result
        )
        or
        result
        <
        0.0
    ):
        raise ValueError(
            f"{name} must be finite "
            "and nonnegative."
        )

    return result


def beam_probing_time_s(
    probing_beam_count: int,
) -> float:
    k = int(
        probing_beam_count
    )

    if k < 0:
        raise ValueError(
            "probing_beam_count "
            "cannot be negative."
        )

    return float(
        k
        *
        STAGE5_BEAM_PROBE_DURATION_S
    )


def beam_training_overhead_fraction(
    probing_beam_count: int,
) -> float:
    return float(
        beam_probing_time_s(
            probing_beam_count
        )
        /
        STAGE5_RATE_ACCOUNTING_FRAME_S
    )


def effective_rate_with_frozen_timing(
    *,
    ber: float,
    probing_beam_count: int,
) -> EffectiveRateResult:
    return effective_rate(
        ber=(
            ber
        ),

        probing_beam_count=(
            probing_beam_count
        ),

        beam_probe_time_s=(
            STAGE5_BEAM_PROBE_DURATION_S
        ),

        frame_time_s=(
            STAGE5_RATE_ACCOUNTING_FRAME_S
        ),
    )


def actual_probe_count_for_adaptive_decision(
    decision: AdaptiveTemporalDecision,
) -> int:
    """
    Physical probe accounting.

    Normal operation:
        probe the adaptive mass-covering beam set.

    Loss-of-lock:
        the posterior has already declared the normal
        requested coverage unattainable before physical
        probing. The recovery schedule therefore replaces
        the normal plan:

          optional next-coarser widened probe
          +
          exhaustive current-codebook sweep.

    The diagnostic adaptive set is not charged again.
    """

    if not isinstance(
        decision,
        AdaptiveTemporalDecision,
    ):
        raise TypeError(
            "decision must be AdaptiveTemporalDecision."
        )

    if not decision.loss_of_lock:

        k = int(
            decision
            .adaptive_selection
            .k
        )

        if k <= 0:
            raise ValueError(
                "Normal adaptive decision "
                "must probe at least one beam."
            )

        return k

    if not (
        decision
        .exhaustive_fallback_active
    ):
        raise ValueError(
            "Loss-of-lock requires "
            "active exhaustive recovery."
        )

    exhaustive_count = len(
        decision
        .exhaustive_fallback_indices
    )

    if exhaustive_count <= 0:
        raise ValueError(
            "Exhaustive recovery beam set "
            "cannot be empty."
        )

    widened_count = (
        1
        if decision
        .widened_fallback
        .available
        else
        0
    )

    return int(
        widened_count
        +
        exhaustive_count
    )


def build_latency_budget(
    *,
    components: LatencyComponents,
    probing_beam_count: int,
) -> LatencyBudget:
    sensing = _nonnegative_finite(
        components.sensing_s,
        name="sensing_s",
    )

    preprocessing = _nonnegative_finite(
        components.preprocessing_s,
        name="preprocessing_s",
    )

    tracking = _nonnegative_finite(
        components.tracking_s,
        name="tracking_s",
    )

    predictor = _nonnegative_finite(
        components.predictor_inference_s,
        name="predictor_inference_s",
    )

    selection = _nonnegative_finite(
        components.beam_selection_s,
        name="beam_selection_s",
    )

    actuation = _nonnegative_finite(
        components.actuation_s,
        name="actuation_s",
    )

    loading = _nonnegative_finite(
        components.data_loading_s,
        name="data_loading_s",
    )

    probe_time = (
        STAGE5_BEAM_PROBE_DURATION_S
    )

    probing = beam_probing_time_s(
        probing_beam_count
    )

    #
    # PDF tau_inference is decomposed so that beam-selection
    # runtime remains explicitly measurable.
    #
    inference_total = (
        predictor
        +
        selection
    )

    online_total = (
        sensing
        +
        preprocessing
        +
        tracking
        +
        inference_total
        +
        probing
        +
        actuation
    )

    wall_with_loading = (
        loading
        +
        online_total
    )

    return LatencyBudget(
        components=(
            components
        ),

        probing_beam_count=int(
            probing_beam_count
        ),

        beam_probe_time_s=(
            probe_time
        ),

        beam_probing_s=(
            probing
        ),

        inference_total_s=float(
            inference_total
        ),

        online_total_s=float(
            online_total
        ),

        data_loading_s=float(
            loading
        ),

        wall_with_loading_s=float(
            wall_with_loading
        ),

        deadline_s=(
            WOMD_SCENE_INTERVAL_S
        ),

        deadline_met=(
            online_total
            <=
            WOMD_SCENE_INTERVAL_S
        ),
    )


def latency_prediction_request(
    budget: LatencyBudget,
) -> LatencyPredictionRequest:
    """
    Controller target is exactly t + tau_total.

    If tau_total coincides with one of the frozen posterior
    horizons, that posterior can be consumed directly.

    Otherwise the target remains exact, but state generation
    must use causal model-based propagation/interpolation.
    It must never fabricate annotated millisecond GT.
    """

    target = float(
        budget.online_total_s
    )

    if (
        not math.isfinite(
            target
        )
        or
        target
        <
        0.0
    ):
        raise ValueError(
            "Invalid latency target."
        )

    exact = None

    for horizon in (
        FROZEN_POSTERIOR_HORIZONS_S
    ):

        if abs(
            target
            -
            horizon
        ) <= 1e-12:

            exact = float(
                horizon
            )

            break

    refinement = (
        exact is None
    )

    if exact is not None:

        semantics = (
            "exact_frozen_posterior_horizon"
        )

    elif target < WOMD_SCENE_INTERVAL_S:

        semantics = (
            "sub_100ms_causal_local_"
            "propagation_or_interpolation"
        )

    else:

        semantics = (
            "exact_latency_target_requires_"
            "causal_model_based_refinement"
        )

    return LatencyPredictionRequest(
        target_offset_s=(
            target
        ),

        target_time_semantics=(
            semantics
        ),

        exact_frozen_posterior_horizon_s=(
            exact
        ),

        model_based_refinement_required=(
            refinement
        ),

        annotated_millisecond_ground_truth_used=(
            False
        ),
    )


def measure_call(
    function: Callable[
        [],
        T
    ],
    *,
    synchronize: Callable[
        [],
        None
    ] | None = None,
) -> TimedCallResult[
    T
]:
    """
    Generic latency primitive.

    For GPU regions, caller supplies torch.cuda.synchronize
    as the synchronize callback. This module deliberately
    does not import torch.
    """

    if synchronize is not None:
        synchronize()

    start_ns = (
        time.perf_counter_ns()
    )

    value = function()

    if synchronize is not None:
        synchronize()

    stop_ns = (
        time.perf_counter_ns()
    )

    elapsed = (
        stop_ns
        -
        start_ns
    ) / 1e9

    if elapsed < 0.0:
        raise RuntimeError(
            "Negative runtime measurement."
        )

    return TimedCallResult(
        value=(
            value
        ),

        elapsed_s=float(
            elapsed
        ),
    )

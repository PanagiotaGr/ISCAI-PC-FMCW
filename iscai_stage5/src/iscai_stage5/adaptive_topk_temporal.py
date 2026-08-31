from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage5.adaptive_topk import (
    AdaptiveTopKSelection,
    FROZEN_COVERAGE_TARGETS,
    FROZEN_NOMINAL_COVERAGE,
    adaptive_topk_probability_mass,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
    UniformAzimuthCodebook,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)


FROZEN_MC_SAMPLE_COUNT = (
    2048
)

FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE = (
    1.0
    /
    float(
        FROZEN_MC_SAMPLE_COUNT
    )
)

KMAX_RULE = (
    "full_codebook_size"
)

HYSTERESIS_RULE = (
    "retain_previous_primary_if_it_remains_"
    "inside_current_mass_covering_set"
)

SWITCH_PENALTY_RULE = (
    "unit_switch_event_indicator_no_tuned_weight"
)

NEIGHBOR_SWEEP_RULE = (
    "immediate_codebook_graph_neighbors_"
    "around_current_primary"
)

WIDENED_FALLBACK_RULE = (
    "next_coarser_frozen_codebook_"
    "nearest_center_to_current_primary"
)

LOSS_OF_LOCK_RULE = (
    "requested_coverage_not_achievable_"
    "with_available_in_support_posterior_mass"
)

LOSS_OF_LOCK_RECOVERY = (
    "widened_fallback_probe_then_"
    "exhaustive_current_codebook_sweep"
)

COARSER_CODEBOOK = {
    64:
        32,

    32:
        16,

    16:
        None,
}


@dataclass(
    frozen=True
)
class AdaptiveTemporalState:
    previous_primary_beam_index: int | None = None

    step_count: int = 0

    cumulative_switch_events: int = 0


@dataclass(
    frozen=True
)
class WidenedFallbackBeam:
    source_beam_count: int

    source_primary_index: int

    fallback_beam_count: int | None

    fallback_beam_index: int | None

    available: bool


@dataclass(
    frozen=True
)
class AdaptiveTemporalDecision:
    requested_coverage: float

    adaptive_selection: AdaptiveTopKSelection

    ordered_beam_indices: tuple[
        int,
        ...
    ]

    primary_beam_index: int

    previous_primary_retained: bool

    primary_switched: bool

    switch_penalty_units: int

    local_neighbor_probe_indices: tuple[
        int,
        ...
    ]

    loss_of_lock: bool

    loss_of_lock_reason: str | None

    widened_fallback: WidenedFallbackBeam

    exhaustive_fallback_indices: tuple[
        int,
        ...
    ]

    exhaustive_fallback_active: bool

    posterior_mass_numerical_tolerance: float

    kmax_rule: str = (
        KMAX_RULE
    )


def initial_adaptive_temporal_state():
    return AdaptiveTemporalState()


def reset_adaptive_temporal_state():
    return initial_adaptive_temporal_state()


def frozen_kmax_for_codebook(
    codebook: UniformAzimuthCodebook,
):
    """
    Freeze Kmax structurally to the full frozen codebook size.

    This introduces no tuned numerical cap and prevents Kmax
    itself from causing a false coverage failure.
    """

    value = int(
        codebook.beam_count
    )

    if value not in (
        16,
        32,
        64,
    ):
        raise ValueError(
            "Adaptive controller only supports "
            "the frozen 16/32/64 codebooks."
        )

    return value


def immediate_neighbor_indices(
    *,
    primary_beam_index: int,
    codebook: UniformAzimuthCodebook,
):
    """
    Local recovery neighborhood on the 1-D frozen beam graph.

    No tunable neighborhood radius is introduced:
    only graph-adjacent cells are used.
    """

    index = int(
        primary_beam_index
    )

    if (
        index
        <
        0
        or
        index
        >=
        codebook.beam_count
    ):
        raise ValueError(
            "primary_beam_index outside codebook."
        )

    result = []

    if index - 1 >= 0:
        result.append(
            index - 1
        )

    if index + 1 < codebook.beam_count:
        result.append(
            index + 1
        )

    return tuple(
        result
    )


def widened_fallback_for_primary(
    *,
    primary_beam_index: int,
    codebook: UniformAzimuthCodebook,
):
    """
    Widened fallback uses the next coarser already-frozen
    codebook family:

        64 -> 32
        32 -> 16
        16 -> no coarser frozen codebook

    No new beamwidth parameter is introduced.
    """

    source_count = int(
        codebook.beam_count
    )

    if source_count not in (
        16,
        32,
        64,
    ):
        raise ValueError(
            "Unsupported frozen codebook size."
        )

    source_index = int(
        primary_beam_index
    )

    if (
        source_index
        <
        0
        or
        source_index
        >=
        source_count
    ):
        raise ValueError(
            "primary_beam_index outside codebook."
        )

    fallback_count = (
        COARSER_CODEBOOK[
            source_count
        ]
    )

    if fallback_count is None:

        return WidenedFallbackBeam(
            source_beam_count=(
                source_count
            ),

            source_primary_index=(
                source_index
            ),

            fallback_beam_count=None,

            fallback_beam_index=None,

            available=False,
        )

    source_center = float(
        codebook.cells[
            source_index
        ].center_azimuth_rad
    )

    fallback_codebook = (
        build_frozen_directional_codebook(
            fallback_count
        )
    )

    fallback_cell = min(
        fallback_codebook.cells,
        key=lambda cell:
            (
                abs(
                    float(
                        cell.center_azimuth_rad
                    )
                    -
                    source_center
                ),
                int(
                    cell.index
                ),
            ),
    )

    return WidenedFallbackBeam(
        source_beam_count=(
            source_count
        ),

        source_primary_index=(
            source_index
        ),

        fallback_beam_count=(
            fallback_count
        ),

        fallback_beam_index=int(
            fallback_cell.index
        ),

        available=True,
    )


def _reorder_with_previous_primary(
    *,
    selection: AdaptiveTopKSelection,
    previous_primary_beam_index,
):
    """
    Hysteresis/persistence does not alter the mass-covering
    beam SET.

    It only keeps the previous primary first in probe order
    if that beam remains a member of the newly selected set.
    """

    indices = tuple(
        int(
            value
        )
        for value in (
            selection.beam_indices
        )
    )

    if not indices:
        raise RuntimeError(
            "Adaptive Top-K returned empty beam set."
        )

    if previous_primary_beam_index is None:

        return (
            indices,
            False,
        )

    previous = int(
        previous_primary_beam_index
    )

    if previous not in indices:

        return (
            indices,
            False,
        )

    reordered = (
        previous,
    ) + tuple(
        index
        for index in indices
        if index != previous
    )

    return (
        reordered,
        True,
    )


def adaptive_temporal_step(
    *,
    state: AdaptiveTemporalState,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
    requested_coverage: float = (
        FROZEN_NOMINAL_COVERAGE
    ),
):
    """
    Frozen Block5.5 adaptive temporal controller.

    Core set:
        smallest descending-P_b mass-covering set.

    Kmax:
        full codebook size.

    Hysteresis:
        if previous primary remains inside the new adaptive
        set, retain it as primary. The selected SET is not
        changed.

    Local recovery:
        immediate graph-neighbor candidates of primary.

    Loss of lock:
        declared if requested posterior mass cannot be
        achieved with the available in-support mass.

    Recovery:
        next-coarser frozen beam candidate when available,
        followed by exhaustive current-codebook sweep.

    No future truth, realized receiver angle, oracle label or
    formal evaluator input enters this function.
    """

    if not isinstance(
        state,
        AdaptiveTemporalState,
    ):
        raise TypeError(
            "state must be AdaptiveTemporalState."
        )

    if (
        state.step_count
        <
        0
        or
        state.cumulative_switch_events
        <
        0
    ):
        raise ValueError(
            "Adaptive temporal state counters "
            "cannot be negative."
        )

    kmax = (
        frozen_kmax_for_codebook(
            codebook
        )
    )

    adaptive = (
        adaptive_topk_probability_mass(
            probability=(
                probability
            ),

            codebook=(
                codebook
            ),

            requested_coverage=(
                requested_coverage
            ),

            kmax=(
                kmax
            ),
        )
    )

    (
        ordered,
        retained,
    ) = _reorder_with_previous_primary(
        selection=(
            adaptive
        ),

        previous_primary_beam_index=(
            state.previous_primary_beam_index
        ),
    )

    primary = int(
        ordered[
            0
        ]
    )

    previous = (
        None
        if state.previous_primary_beam_index
        is None
        else
        int(
            state.previous_primary_beam_index
        )
    )

    switched = (
        previous is not None
        and
        primary != previous
    )

    switch_units = (
        1
        if switched
        else
        0
    )

    loss_of_lock = (
        not adaptive
        .achieved_requested_coverage
    )

    if (
        adaptive
        .unattainable_due_to_outside_support
    ):
        loss_reason = (
            "outside_support_mass_prevents_"
            "requested_coverage"
        )

    elif loss_of_lock:
        loss_reason = (
            "requested_coverage_not_achieved"
        )

    else:
        loss_reason = None

    neighbors = (
        immediate_neighbor_indices(
            primary_beam_index=(
                primary
            ),

            codebook=(
                codebook
            ),
        )
    )

    widened = (
        widened_fallback_for_primary(
            primary_beam_index=(
                primary
            ),

            codebook=(
                codebook
            ),
        )
    )

    exhaustive = (
        tuple(
            range(
                codebook.beam_count
            )
        )
        if loss_of_lock
        else
        tuple()
    )

    new_state = AdaptiveTemporalState(
        previous_primary_beam_index=(
            primary
        ),

        step_count=(
            int(
                state.step_count
            )
            +
            1
        ),

        cumulative_switch_events=(
            int(
                state.cumulative_switch_events
            )
            +
            switch_units
        ),
    )

    decision = AdaptiveTemporalDecision(
        requested_coverage=float(
            requested_coverage
        ),

        adaptive_selection=(
            adaptive
        ),

        ordered_beam_indices=(
            ordered
        ),

        primary_beam_index=(
            primary
        ),

        previous_primary_retained=(
            bool(
                retained
            )
        ),

        primary_switched=(
            bool(
                switched
            )
        ),

        switch_penalty_units=(
            int(
                switch_units
            )
        ),

        local_neighbor_probe_indices=(
            neighbors
        ),

        loss_of_lock=(
            bool(
                loss_of_lock
            )
        ),

        loss_of_lock_reason=(
            loss_reason
        ),

        widened_fallback=(
            widened
        ),

        exhaustive_fallback_indices=(
            exhaustive
        ),

        exhaustive_fallback_active=(
            bool(
                loss_of_lock
            )
        ),

        posterior_mass_numerical_tolerance=(
            FROZEN_POSTERIOR_MASS_NUMERICAL_TOLERANCE
        ),
    )

    return (
        decision,
        new_state,
    )


def all_coverage_temporal_decisions(
    *,
    state: AdaptiveTemporalState,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
):
    """
    Evaluate the four preregistered q values from the same
    causal state without mutating state between q values.
    """

    return {
        q:
            adaptive_temporal_step(
                state=(
                    state
                ),

                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),

                requested_coverage=(
                    q
                ),
            )[
                0
            ]
        for q in (
            FROZEN_COVERAGE_TARGETS
        )
    }

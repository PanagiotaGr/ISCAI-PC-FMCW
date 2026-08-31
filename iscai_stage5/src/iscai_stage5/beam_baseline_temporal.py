from __future__ import annotations

from dataclasses import dataclass
import math

from iscai_stage5.beam_baselines import (
    BeamSelection,
    CONTROLLER_BASELINE_NAMES,
    EVALUATOR_ONLY_BASELINE_NAMES,
    fixed_top_k_probability,
    frozen_baseline_registry,
    previous_beam_persistence,
)

from iscai_stage5.beam_codebook import (
    BeamProbabilityMass,
    UniformAzimuthCodebook,
)


PERSISTENCE_STATE_SEMANTICS = (
    "previous_selected_beam_from_same_"
    "causal_persistence_controller"
)

PERSISTENCE_INITIALIZATION = (
    "geometry_nearest_using_current_"
    "causal_predicted_receiver_azimuth"
)

PERSISTENCE_UPDATE = (
    "selected_beam_becomes_previous_"
    "beam_for_next_controller_step"
)

PERSISTENCE_RESET = (
    "clear_previous_beam_state"
)

ORACLE_ACCESS_POLICY = (
    "evaluator_only_after_controller_decision"
)

REALIZED_RECEIVER_AZIMUTH_POLICY = (
    "evaluation_only_never_controller_input"
)


@dataclass(
    frozen=True
)
class PersistenceState:
    previous_beam_index: int | None = None

    step_count: int = 0


def initial_persistence_state() -> PersistenceState:
    return PersistenceState(
        previous_beam_index=None,
        step_count=0,
    )


def reset_persistence_state() -> PersistenceState:
    return initial_persistence_state()


def validate_controller_selection(
    *,
    selection: BeamSelection,
    codebook: UniformAzimuthCodebook,
):
    """
    Structural guard for controller-visible decisions.

    Evaluator-only/oracle selections are forbidden.
    """

    if selection.evaluation_only:
        raise ValueError(
            "Evaluator-only beam selection "
            "cannot enter controller state."
        )

    if (
        selection.policy_name
        not in
        CONTROLLER_BASELINE_NAMES
    ):
        raise ValueError(
            "Selection is not from a frozen "
            "controller baseline."
        )

    if not selection.beam_indices:
        raise ValueError(
            "Controller selection cannot be empty."
        )

    if len(
        set(
            selection.beam_indices
        )
    ) != len(
        selection.beam_indices
    ):
        raise ValueError(
            "Controller beam indices "
            "must be unique."
        )

    for index in selection.beam_indices:

        if (
            int(
                index
            )
            <
            0
            or
            int(
                index
            )
            >=
            codebook.beam_count
        ):
            raise ValueError(
                "Controller beam index "
                "outside codebook."
            )


def validate_evaluator_oracle(
    *,
    selection: BeamSelection,
    codebook: UniformAzimuthCodebook,
):
    """
    Structural guard for the evaluation-only oracle.
    """

    if not selection.evaluation_only:
        raise ValueError(
            "Oracle evaluation requires "
            "evaluation_only=True."
        )

    if (
        selection.policy_name
        not in
        EVALUATOR_ONLY_BASELINE_NAMES
    ):
        raise ValueError(
            "Selection is not a frozen "
            "evaluation-only baseline."
        )

    if len(
        selection.beam_indices
    ) != 1:
        raise ValueError(
            "Frozen oracle must return "
            "exactly one beam."
        )

    index = int(
        selection.beam_indices[
            0
        ]
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
            "Oracle beam index "
            "outside codebook."
        )


def persistence_step(
    *,
    state: PersistenceState,
    causal_predicted_azimuth_rad,
    codebook: UniformAzimuthCodebook,
):
    """
    Advance the causal previous-beam-persistence baseline.

    First controller step:
        geometry-nearest fallback.

    Later controller steps:
        reuse previous selected beam.

    The newly selected beam is stored for the next step.

    There is no future truth, realized receiver angle or
    oracle input in this controller API.
    """

    if not isinstance(
        state,
        PersistenceState,
    ):
        raise TypeError(
            "state must be PersistenceState."
        )

    if state.step_count < 0:
        raise ValueError(
            "step_count cannot be negative."
        )

    predicted_angle = float(
        causal_predicted_azimuth_rad
    )

    if not math.isfinite(
        predicted_angle
    ):
        raise ValueError(
            "causal_predicted_azimuth_rad "
            "must be finite."
        )

    selection = (
        previous_beam_persistence(
            previous_beam_index=(
                state.previous_beam_index
            ),

            codebook=(
                codebook
            ),

            fallback_predicted_azimuth_rad=(
                predicted_angle
            ),
        )
    )

    validate_controller_selection(
        selection=(
            selection
        ),

        codebook=(
            codebook
        ),
    )

    if len(
        selection.beam_indices
    ) != 1:
        raise RuntimeError(
            "Persistence baseline must "
            "select exactly one beam."
        )

    new_state = PersistenceState(
        previous_beam_index=int(
            selection.beam_indices[
                0
            ]
        ),

        step_count=(
            int(
                state.step_count
            )
            +
            1
        ),
    )

    return (
        selection,
        new_state,
    )


def fixed_probability_baseline_suite(
    *,
    probability: BeamProbabilityMass,
    codebook: UniformAzimuthCodebook,
):
    """
    Return the frozen fixed Top-1/3/5 comparison suite.

    No adaptive K selection occurs here.
    """

    selections = {}

    for k in (
        1,
        3,
        5,
    ):

        selection = (
            fixed_top_k_probability(
                probability=(
                    probability
                ),

                codebook=(
                    codebook
                ),

                k=(
                    k
                ),
            )
        )

        validate_controller_selection(
            selection=(
                selection
            ),

            codebook=(
                codebook
            ),
        )

        selections[
            k
        ] = selection

    return selections


def frozen_temporal_baseline_contract():
    registry = (
        frozen_baseline_registry()
    )

    return {
        "persistence_state":
            PERSISTENCE_STATE_SEMANTICS,

        "persistence_initialization":
            PERSISTENCE_INITIALIZATION,

        "persistence_update":
            PERSISTENCE_UPDATE,

        "persistence_reset":
            PERSISTENCE_RESET,

        "oracle_access":
            ORACLE_ACCESS_POLICY,

        "realized_receiver_azimuth":
            REALIZED_RECEIVER_AZIMUTH_POLICY,

        "adaptive_TopK":
            False,

        "blockage_aware":
            False,

        "controller_baselines":
            registry[
                "controller_baselines"
            ],

        "evaluation_only_baselines":
            registry[
                "evaluation_only_baselines"
            ],
    }

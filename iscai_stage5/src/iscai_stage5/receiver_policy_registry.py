from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite
from typing import Sequence

from iscai_stage5.receiver_selection import (
    NO_ELIGIBLE_RECEIVER,
    SELECTED_RECEIVER,
    ReceiverCandidate,
    ReceiverSelectionConfig,
    select_primary_receiver,
)


PRECEDING_CONNECTED_VEHICLE = (
    "preceding_connected_vehicle"
)

NEAREST_CAUSAL_VEHICLE_AHEAD = (
    "nearest_causal_vehicle_ahead"
)

NEAREST_CONNECTED_VEHICLE_AHEAD = (
    "nearest_connected_vehicle_ahead"
)

ONCOMING_CONNECTED_VEHICLE = (
    "oncoming_connected_vehicle"
)

FIXED_CAUSAL_TARGET = (
    "fixed_causal_target"
)

HIGHEST_EXPECTED_LINK_UTILITY = (
    "highest_expected_link_utility"
)


PRIMARY_POLICY = (
    NEAREST_CAUSAL_VEHICLE_AHEAD
)


CANONICAL_POLICIES = (
    PRECEDING_CONNECTED_VEHICLE,
    NEAREST_CAUSAL_VEHICLE_AHEAD,
    ONCOMING_CONNECTED_VEHICLE,
    FIXED_CAUSAL_TARGET,
    HIGHEST_EXPECTED_LINK_UTILITY,
)


ALIASES = {
    NEAREST_CONNECTED_VEHICLE_AHEAD:
        NEAREST_CAUSAL_VEHICLE_AHEAD,
}


@dataclass(frozen=True)
class ReceiverPolicyMetadata:
    """
    Optional CURRENT/CAUSAL metadata aligned with the
    existing ReceiverCandidate sequence.

    This metadata is controller-side only.

    It is NOT:
      - a predictor numeric feature,
      - a WOMD truth track ID,
      - tracks_to_predict,
      - future ground truth.

    association_key:
      opaque ESTIMATED-association key, if the upstream online
      association layer supplies one.

    causal_is_preceding / causal_is_oncoming:
      current causal classifications only.

    current_heading_h0_rad:
      current body heading in H0; may be used as an analytic
      oncoming fallback.

    expected_link_utility:
      causal/predicted utility only, never realized future link
      performance.
    """

    association_key: str | None = None

    causal_is_preceding: bool | None = None

    causal_is_oncoming: bool | None = None

    current_heading_h0_rad: float | None = None

    expected_link_utility: float | None = None

    def __post_init__(self) -> None:

        if (
            self.association_key
            is not None
            and not str(
                self.association_key
            )
        ):
            raise ValueError(
                "association_key must be non-empty "
                "when supplied"
            )

        for name, value in (
            (
                "current_heading_h0_rad",
                self.current_heading_h0_rad,
            ),
            (
                "expected_link_utility",
                self.expected_link_utility,
            ),
        ):
            if (
                value is not None
                and not isfinite(
                    float(value)
                )
            ):
                raise ValueError(
                    f"{name} must be finite"
                )


@dataclass(frozen=True)
class ReceiverPolicyDecision:
    policy: str

    status: str

    selected_candidate_index: int | None

    eligible_candidate_count: int

    planar_range_m: float | None

    semantics: str


def normalize_policy_name(
    policy: str,
) -> str:

    name = str(policy)

    return ALIASES.get(
        name,
        name,
    )


def _metadata_tuple(
    candidates: Sequence[ReceiverCandidate],
    metadata: Sequence[ReceiverPolicyMetadata] | None,
) -> tuple[ReceiverPolicyMetadata, ...]:

    if metadata is None:
        return tuple(
            ReceiverPolicyMetadata()
            for _ in candidates
        )

    values = tuple(metadata)

    if len(values) != len(candidates):
        raise ValueError(
            "metadata length must equal candidate length"
        )

    return values


def _planar_range(
    candidate: ReceiverCandidate,
) -> float:

    x = float(
        candidate.position_h0_m[0]
    )

    y = float(
        candidate.position_h0_m[1]
    )

    return hypot(
        x,
        y,
    )


def _existing_policy_eligible_indices(
    candidates: Sequence[ReceiverCandidate],
    selection_config: ReceiverSelectionConfig | None,
) -> tuple[int, ...]:
    """
    Reuse the EXISTING frozen Block5.1 selector as the exact shared
    eligibility oracle.

    This deliberately avoids reimplementing:
      - vehicle-only gating,
      - current causal availability,
      - association validity,
      - positive-forward H0,
      - optional frozen planar-range gate.
    """

    indices = []

    for index, candidate in enumerate(
        candidates
    ):
        if selection_config is None:
            result = select_primary_receiver(
                (candidate,)
            )
        else:
            result = select_primary_receiver(
                (candidate,),
                selection_config,
            )

        if result.status == SELECTED_RECEIVER:
            indices.append(index)

    return tuple(indices)


def _no_receiver(
    policy: str,
    eligible_count: int,
    semantics: str,
) -> ReceiverPolicyDecision:

    return ReceiverPolicyDecision(
        policy=policy,
        status=NO_ELIGIBLE_RECEIVER,
        selected_candidate_index=None,
        eligible_candidate_count=int(
            eligible_count
        ),
        planar_range_m=None,
        semantics=semantics,
    )


def _selected(
    policy: str,
    index: int,
    candidate: ReceiverCandidate,
    eligible_count: int,
    semantics: str,
) -> ReceiverPolicyDecision:

    return ReceiverPolicyDecision(
        policy=policy,
        status=SELECTED_RECEIVER,
        selected_candidate_index=int(index),
        eligible_candidate_count=int(
            eligible_count
        ),
        planar_range_m=float(
            _planar_range(candidate)
        ),
        semantics=semantics,
    )


def select_nearest_causal_vehicle_ahead(
    candidates: Sequence[ReceiverCandidate],
    *,
    selection_config: ReceiverSelectionConfig | None = None,
) -> ReceiverPolicyDecision:
    """
    Exact wrapper around the already-frozen primary Block5.1 selector.

    No policy semantics are changed here.
    """

    if selection_config is None:
        result = select_primary_receiver(
            candidates
        )
    else:
        result = select_primary_receiver(
            candidates,
            selection_config,
        )

    if result.status != SELECTED_RECEIVER:
        return _no_receiver(
            NEAREST_CAUSAL_VEHICLE_AHEAD,
            int(
                result.eligible_candidate_count
            ),
            (
                "exact_wrapper_of_existing_frozen_"
                "select_primary_receiver"
            ),
        )

    index = int(
        result.selected_candidate_index
    )

    return _selected(
        NEAREST_CAUSAL_VEHICLE_AHEAD,
        index,
        candidates[index],
        int(
            result.eligible_candidate_count
        ),
        (
            "exact_wrapper_of_existing_frozen_"
            "select_primary_receiver"
        ),
    )


def select_preceding_connected_vehicle(
    candidates: Sequence[ReceiverCandidate],
    *,
    metadata: Sequence[ReceiverPolicyMetadata],
    selection_config: ReceiverSelectionConfig | None = None,
) -> ReceiverPolicyDecision:
    """
    Select nearest shared-eligible actor that has been causally
    classified as preceding.

    Unknown classification is NOT guessed.
    """

    meta = _metadata_tuple(
        candidates,
        metadata,
    )

    shared = (
        _existing_policy_eligible_indices(
            candidates,
            selection_config,
        )
    )

    eligible = tuple(
        i
        for i in shared
        if meta[i].causal_is_preceding is True
    )

    if not eligible:
        return _no_receiver(
            PRECEDING_CONNECTED_VEHICLE,
            0,
            (
                "causal_preceding_classification_plus_"
                "frozen_shared_receiver_eligibility"
            ),
        )

    index = min(
        eligible,
        key=lambda i: (
            _planar_range(
                candidates[i]
            ),
            i,
        ),
    )

    return _selected(
        PRECEDING_CONNECTED_VEHICLE,
        index,
        candidates[index],
        len(eligible),
        (
            "causal_preceding_classification_plus_"
            "nearest_current_planar_range"
        ),
    )


def _causally_oncoming(
    metadata: ReceiverPolicyMetadata,
) -> bool:

    if (
        metadata.causal_is_oncoming
        is not None
    ):
        return bool(
            metadata.causal_is_oncoming
        )

    if (
        metadata.current_heading_h0_rad
        is None
    ):
        return False

    # H0 +x is ego-forward.
    # cos(heading)<0 means an opposite longitudinal heading component.
    return (
        cos(
            float(
                metadata.current_heading_h0_rad
            )
        )
        < 0.0
    )


def select_oncoming_connected_vehicle(
    candidates: Sequence[ReceiverCandidate],
    *,
    metadata: Sequence[ReceiverPolicyMetadata],
    selection_config: ReceiverSelectionConfig | None = None,
) -> ReceiverPolicyDecision:
    """
    Select nearest shared-eligible actor causally identified as oncoming.

    Explicit current causal classification has priority.
    Current H0 heading is an analytic fallback.
    Future heading is never an input.
    """

    meta = _metadata_tuple(
        candidates,
        metadata,
    )

    shared = (
        _existing_policy_eligible_indices(
            candidates,
            selection_config,
        )
    )

    eligible = tuple(
        i
        for i in shared
        if _causally_oncoming(
            meta[i]
        )
    )

    if not eligible:
        return _no_receiver(
            ONCOMING_CONNECTED_VEHICLE,
            0,
            (
                "current_causal_oncoming_classification_"
                "only"
            ),
        )

    index = min(
        eligible,
        key=lambda i: (
            _planar_range(
                candidates[i]
            ),
            i,
        ),
    )

    return _selected(
        ONCOMING_CONNECTED_VEHICLE,
        index,
        candidates[index],
        len(eligible),
        (
            "current_causal_oncoming_classification_"
            "plus_nearest_planar_range"
        ),
    )


def select_fixed_causal_target(
    candidates: Sequence[ReceiverCandidate],
    *,
    metadata: Sequence[ReceiverPolicyMetadata],
    fixed_association_key: str,
    selection_config: ReceiverSelectionConfig | None = None,
) -> ReceiverPolicyDecision:
    """
    Fixed target policy using an OPAQUE ESTIMATED-association key.

    The association key lives in controller-side metadata; it is not
    added to ReceiverCandidate and is not a predictor feature.

    If the selected causal target is unavailable, another actor is not
    silently substituted.
    """

    key = str(
        fixed_association_key
    )

    if not key:
        raise ValueError(
            "fixed_association_key must be non-empty"
        )

    meta = _metadata_tuple(
        candidates,
        metadata,
    )

    shared = (
        _existing_policy_eligible_indices(
            candidates,
            selection_config,
        )
    )

    matches = tuple(
        i
        for i in shared
        if meta[i].association_key == key
    )

    if len(matches) > 1:
        raise ValueError(
            "fixed_association_key is ambiguous: "
            "multiple currently eligible candidates "
            "have the same estimated-association key"
        )

    if not matches:
        return _no_receiver(
            FIXED_CAUSAL_TARGET,
            0,
            (
                "predeclared_estimated_association_key_"
                "without_actor_substitution"
            ),
        )

    index = matches[0]

    return _selected(
        FIXED_CAUSAL_TARGET,
        index,
        candidates[index],
        1,
        (
            "predeclared_estimated_association_key_"
            "without_actor_substitution"
        ),
    )


def select_highest_expected_link_utility(
    candidates: Sequence[ReceiverCandidate],
    *,
    metadata: Sequence[ReceiverPolicyMetadata],
    selection_config: ReceiverSelectionConfig | None = None,
) -> ReceiverPolicyDecision:
    """
    Highest CAUSAL/PREDICTED expected-link-utility policy.

    Realized future gain/SNR/BER/rate is evaluator-only and must never
    populate expected_link_utility.
    """

    meta = _metadata_tuple(
        candidates,
        metadata,
    )

    shared = (
        _existing_policy_eligible_indices(
            candidates,
            selection_config,
        )
    )

    eligible = tuple(
        i
        for i in shared
        if (
            meta[i].expected_link_utility
            is not None
        )
    )

    if not eligible:
        return _no_receiver(
            HIGHEST_EXPECTED_LINK_UTILITY,
            0,
            (
                "causal_predicted_link_utility_only"
            ),
        )

    index = min(
        eligible,
        key=lambda i: (
            -float(
                meta[i].expected_link_utility
            ),
            _planar_range(
                candidates[i]
            ),
            i,
        ),
    )

    return _selected(
        HIGHEST_EXPECTED_LINK_UTILITY,
        index,
        candidates[index],
        len(eligible),
        (
            "maximum_causal_predicted_link_utility_"
            "with_geometry_tie_break"
        ),
    )


def receiver_policy_registry():
    """
    Callable registry for all five Stage5 receiver-selection policies.
    """

    registry = {
        PRECEDING_CONNECTED_VEHICLE:
            select_preceding_connected_vehicle,

        NEAREST_CAUSAL_VEHICLE_AHEAD:
            select_nearest_causal_vehicle_ahead,

        ONCOMING_CONNECTED_VEHICLE:
            select_oncoming_connected_vehicle,

        FIXED_CAUSAL_TARGET:
            select_fixed_causal_target,

        HIGHEST_EXPECTED_LINK_UTILITY:
            select_highest_expected_link_utility,
    }

    registry[
        NEAREST_CONNECTED_VEHICLE_AHEAD
    ] = registry[
        NEAREST_CAUSAL_VEHICLE_AHEAD
    ]

    return registry


def select_receiver_policy(
    policy: str,
    candidates: Sequence[ReceiverCandidate],
    *,
    metadata: Sequence[ReceiverPolicyMetadata] | None = None,
    selection_config: ReceiverSelectionConfig | None = None,
    fixed_association_key: str | None = None,
) -> ReceiverPolicyDecision:
    """
    Unified dispatcher.

    For the frozen primary policy, metadata is not required and the
    existing select_primary_receiver implementation is used exactly.
    """

    canonical = normalize_policy_name(
        policy
    )

    if (
        canonical
        == NEAREST_CAUSAL_VEHICLE_AHEAD
    ):
        return (
            select_nearest_causal_vehicle_ahead(
                candidates,
                selection_config=
                    selection_config,
            )
        )

    if canonical not in CANONICAL_POLICIES:
        raise ValueError(
            f"Unknown receiver policy: {policy!r}"
        )

    if metadata is None:
        raise ValueError(
            f"{canonical} requires causal policy metadata"
        )

    if (
        canonical
        == PRECEDING_CONNECTED_VEHICLE
    ):
        return (
            select_preceding_connected_vehicle(
                candidates,
                metadata=metadata,
                selection_config=
                    selection_config,
            )
        )

    if (
        canonical
        == ONCOMING_CONNECTED_VEHICLE
    ):
        return (
            select_oncoming_connected_vehicle(
                candidates,
                metadata=metadata,
                selection_config=
                    selection_config,
            )
        )

    if canonical == FIXED_CAUSAL_TARGET:

        if fixed_association_key is None:
            raise ValueError(
                "fixed_causal_target requires "
                "fixed_association_key"
            )

        return (
            select_fixed_causal_target(
                candidates,
                metadata=metadata,
                fixed_association_key=
                    fixed_association_key,
                selection_config=
                    selection_config,
            )
        )

    if (
        canonical
        == HIGHEST_EXPECTED_LINK_UTILITY
    ):
        return (
            select_highest_expected_link_utility(
                candidates,
                metadata=metadata,
                selection_config=
                    selection_config,
            )
        )

    raise AssertionError(
        "Unreachable policy dispatcher state"
    )


def policy_contract() -> dict[str, object]:

    return {
        "primary_policy":
            PRIMARY_POLICY,

        "primary_policy_changed":
            False,

        "canonical_policies":
            list(
                CANONICAL_POLICIES
            ),

        "aliases":
            dict(ALIASES),

        "integration": {
            "candidate_type":
                "existing_iscai_stage5.receiver_selection.ReceiverCandidate",

            "primary_selector":
                "existing_select_primary_receiver",

            "policy_specific_metadata":
                "causal_sidecar_aligned_by_candidate_index",

            "existing_ReceiverCandidate_modified":
                False,

            "existing_stage3_bridge_modified":
                False,
        },

        "leakage_guards": {
            "future_truth_selector":
                False,

            "tracks_to_predict_selector":
                False,

            "perfect_WOMD_track_ID_selector":
                False,

            "realized_future_link_utility":
                False,
        },

        "fixed_causal_target": {
            "selector":
                "opaque_estimated_association_key",

            "key_is_predictor_numeric_feature":
                False,

            "key_is_WOMD_truth_track_ID":
                False,

            "silent_actor_substitution":
                False,
        },

        "preceding_connected_vehicle": {
            "classification":
                "current_causal_classification",

            "unknown_classification":
                "not_selected",
        },

        "oncoming_connected_vehicle": {
            "classification":
                "current_causal_flag_else_current_H0_heading",

            "future_heading":
                False,
        },

        "highest_expected_link_utility": {
            "utility":
                "causal_or_predicted_only",

            "future_realized_link_truth":
                False,
        },
    }

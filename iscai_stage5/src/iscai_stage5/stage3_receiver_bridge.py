from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from iscai_stage5.receiver_selection import (
    ReceiverCandidate,
)


SOURCE_SEMANTICS = (
    "current_causal_associated_actor_estimate_H0"
)


@dataclass(
    frozen=True
)
class CurrentAssociatedActorEstimateH0:
    """
    Narrow causal boundary between the frozen tracking/
    association layer and the Stage5 receiver selector.

    Only information available at the current control time
    may populate this object.

    Deliberately absent:
      - future trajectory truth,
      - tracks_to_predict,
      - objects_of_interest,
      - oracle receiver label,
      - perfect WOMD track ID,
      - future receiver heading.

    Semantic class is current actor metadata only.
    """

    semantic_class: str

    position_h0_m: tuple[
        float,
        float,
        float,
    ]

    current_available: bool = True

    association_valid: bool = True

    source_semantics: str = (
        SOURCE_SEMANTICS
    )


def _finite_position3(
    position,
) -> tuple[
    float,
    float,
    float,
]:
    values = tuple(
        float(
            value
        )
        for value in position
    )

    if len(
        values
    ) != 3:
        raise ValueError(
            "Current associated actor position "
            "must contain exactly x/y/z."
        )

    if not all(
        math.isfinite(
            value
        )
        for value in values
    ):
        raise ValueError(
            "Current associated actor position "
            "must be finite."
        )

    return values


def receiver_candidate_from_current_estimate(
    estimate: CurrentAssociatedActorEstimateH0,
) -> ReceiverCandidate:
    """
    Convert only the permitted current causal fields into
    the frozen Stage5 receiver-candidate contract.
    """

    if not isinstance(
        estimate,
        CurrentAssociatedActorEstimateH0,
    ):
        raise TypeError(
            "estimate must be a "
            "CurrentAssociatedActorEstimateH0"
        )

    if (
        estimate.source_semantics
        !=
        SOURCE_SEMANTICS
    ):
        raise ValueError(
            "Unexpected receiver-bridge "
            "source semantics."
        )

    semantic_class = str(
        estimate.semantic_class
    ).strip()

    if not semantic_class:
        raise ValueError(
            "semantic_class must be non-empty."
        )

    position = _finite_position3(
        estimate.position_h0_m
    )

    return ReceiverCandidate(
        semantic_class=(
            semantic_class
        ),

        position_h0_m=(
            position
        ),

        current_available=bool(
            estimate.current_available
        ),

        association_valid=bool(
            estimate.association_valid
        ),
    )


def receiver_candidates_from_current_estimates(
    estimates: Iterable[
        CurrentAssociatedActorEstimateH0
    ],
) -> tuple[
    ReceiverCandidate,
    ...
]:
    """
    Preserve current-estimate ordering.

    Ordering is not interpreted as identity.  The downstream
    selector may use the list index only as a deterministic
    exact-tie resolution, never as a physical/model feature.
    """

    return tuple(
        receiver_candidate_from_current_estimate(
            estimate
        )
        for estimate in estimates
    )

"""Frozen Stage-1A actor-role masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from iscai_stage1.contracts.stage1a import ObjectClass


@dataclass(frozen=True)
class ActorRoleMasks:
    is_context_actor: bool
    is_anchor_valid: bool
    is_forecasting_target_candidate: bool
    is_receiver_candidate: bool


def compute_actor_role_masks(
    *,
    validity: Sequence[bool],
    anchor_index: int,
    object_class: ObjectClass,
    is_sdc: bool,
    receiver_geometry_valid: bool,
) -> ActorRoleMasks:
    """Compute roles using causal validity only.

    Important:
    forecasting eligibility does NOT require adjacent t-1 validity
    and does NOT imply velocity_valid[anchor].
    """

    if anchor_index < 0:
        raise ValueError("anchor_index must be non-negative.")
    if anchor_index >= len(validity):
        raise IndexError("anchor_index is outside the validity sequence.")

    causal_validity = validity[: anchor_index + 1]

    is_context_actor = any(causal_validity)
    is_anchor_valid = bool(validity[anchor_index])

    causal_valid_count = sum(bool(value) for value in causal_validity)

    is_forecasting_target_candidate = (
        is_anchor_valid
        and causal_valid_count >= 2
    )

    is_receiver_candidate = (
        object_class == "TYPE_VEHICLE"
        and not is_sdc
        and is_anchor_valid
        and receiver_geometry_valid
    )

    return ActorRoleMasks(
        is_context_actor=is_context_actor,
        is_anchor_valid=is_anchor_valid,
        is_forecasting_target_candidate=is_forecasting_target_candidate,
        is_receiver_candidate=is_receiver_candidate,
    )
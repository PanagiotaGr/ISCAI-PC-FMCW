"""Causal WOMD annotation-derived actor-history canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from iscai_stage1.contracts.stage1a import Vec3, ZERO_VEC3
from iscai_stage1.actors.velocity import (
    derive_adjacent_backward_velocity,
)


@dataclass(frozen=True)
class RawObjectStateW:
    """Minimal WOMD annotation state consumed by Stage 1A.

    Deliberately excludes annotated WOMD velocity.

    The Stage-1 realistic path therefore cannot silently fall back to
    velocity_x / velocity_y from WOMD.
    """

    center_W_m: Vec3
    dimensions_lwh_m: Vec3
    heading_rad: float
    valid: bool


@dataclass(frozen=True)
class CausalActorHistory:
    """Canonical causal actor history, restricted to t <= anchor."""

    timestamps_s: tuple[float, ...]

    position_W_m: tuple[Vec3, ...]
    dimensions_lwh_m: tuple[Vec3, ...]
    heading_rad: tuple[float, ...]
    state_valid: tuple[bool, ...]

    velocity_W_mps: tuple[Vec3, ...]
    velocity_valid: tuple[bool, ...]

    # Anchor in this causal artifact is always the last local timestep.
    anchor_index: int

    # Original Scenario.current_time_index for provenance.
    source_current_time_index: int


def _validate_valid_state(state: RawObjectStateW) -> None:
    values = (
        *state.center_W_m,
        *state.dimensions_lwh_m,
        state.heading_rad,
    )

    if not all(isfinite(value) for value in values):
        raise ValueError(
            "A valid WOMD actor state contains non-finite geometry."
        )


def canonicalize_causal_actor_history(
    *,
    timestamps_seconds: Sequence[float],
    states: Sequence[RawObjectStateW],
    current_time_index: int,
) -> CausalActorHistory:
    """Create a causal artifact without inspecting any future state value.

    Only timestamps/states at indices 0..current_time_index are accessed.

    Invalid state numeric payloads are deliberately zero-filled and their
    validity is represented only through state_valid=False.
    """

    if current_time_index < 0:
        raise ValueError("current_time_index must be non-negative.")

    causal_count = current_time_index + 1

    # Important for causality:
    # require only the causal prefix to exist.
    if len(timestamps_seconds) < causal_count:
        raise ValueError(
            "timestamps_seconds does not contain the complete causal prefix."
        )

    if len(states) < causal_count:
        raise ValueError(
            "states does not contain the complete causal prefix."
        )

    causal_timestamps: list[float] = []
    positions: list[Vec3] = []
    dimensions: list[Vec3] = []
    headings: list[float] = []
    validity: list[bool] = []

    for index in range(causal_count):
        timestamp = float(timestamps_seconds[index])

        if not isfinite(timestamp):
            raise ValueError(
                f"Non-finite causal timestamp at index {index}."
            )

        state = states[index]
        is_valid = bool(state.valid)

        causal_timestamps.append(timestamp)
        validity.append(is_valid)

        if not is_valid:
            # Never preserve arbitrary annotation payload from invalid states.
            positions.append(ZERO_VEC3)
            dimensions.append(ZERO_VEC3)
            headings.append(0.0)
            continue

        _validate_valid_state(state)

        positions.append(
            (
                float(state.center_W_m[0]),
                float(state.center_W_m[1]),
                float(state.center_W_m[2]),
            )
        )
        dimensions.append(
            (
                float(state.dimensions_lwh_m[0]),
                float(state.dimensions_lwh_m[1]),
                float(state.dimensions_lwh_m[2]),
            )
        )
        headings.append(float(state.heading_rad))

    velocity = derive_adjacent_backward_velocity(
        timestamps_s=causal_timestamps,
        positions_W_m=positions,
        state_valid=validity,
    )

    return CausalActorHistory(
        timestamps_s=tuple(causal_timestamps),
        position_W_m=tuple(positions),
        dimensions_lwh_m=tuple(dimensions),
        heading_rad=tuple(headings),
        state_valid=tuple(validity),
        velocity_W_mps=velocity.velocity_W_mps,
        velocity_valid=velocity.velocity_valid,
        anchor_index=current_time_index,
        source_current_time_index=current_time_index,
    )
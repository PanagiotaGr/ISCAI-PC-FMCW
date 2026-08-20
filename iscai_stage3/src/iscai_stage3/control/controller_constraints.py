from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControllerConstraintResult:
    valid: bool
    action_count: int
    beam_switches: int
    target_switches: int
    min_priority: float
    max_priority: float


def validate_controller_history(
    *,
    history,
    beam_count: int = 8,
    max_beam_switches: int | None = None,
) -> ControllerConstraintResult:

    if not history:
        raise ValueError(
            "Empty controller history."
        )

    if beam_count <= 0:
        raise ValueError(
            "beam_count must be positive."
        )

    previous_time = None
    previous_beam = None
    previous_target = None

    beam_switches = 0
    target_switches = 0
    priorities = []


    for action in history:

        time_index = int(
            action["time_index"]
        )

        target_id = int(
            action["target_id"]
        )

        beam_id = int(
            action["beam_id"]
        )

        priority = float(
            action["priority"]
        )


        if target_id < 0:
            raise ValueError(
                "Negative target_id."
            )

        if not (
            0 <= beam_id < beam_count
        ):
            raise ValueError(
                "Beam ID outside valid range."
            )

        if not math.isfinite(
            priority
        ):
            raise ValueError(
                "Non-finite priority."
            )


        if previous_time is not None:

            if time_index != previous_time + 1:
                raise ValueError(
                    "Non-contiguous controller time."
                )

            if beam_id != previous_beam:
                beam_switches += 1

            if target_id != previous_target:
                target_switches += 1


        previous_time = time_index
        previous_beam = beam_id
        previous_target = target_id

        priorities.append(
            priority
        )


    if max_beam_switches is not None:

        if beam_switches > max_beam_switches:
            raise ValueError(
                "Beam switching constraint violated."
            )


    return ControllerConstraintResult(
        valid=True,
        action_count=len(history),
        beam_switches=beam_switches,
        target_switches=target_switches,
        min_priority=min(priorities),
        max_priority=max(priorities),
    )

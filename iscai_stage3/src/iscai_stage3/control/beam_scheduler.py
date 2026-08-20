from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BeamAction:

    target_id: int

    beam_id: int

    priority: float



def allocate_beam(
    *,
    target_id: int,
    utility: float,
    uncertainty: float,
) -> BeamAction:

    if utility < 0:
        raise ValueError(
            "Invalid utility."
        )

    if uncertainty < 0:
        raise ValueError(
            "Invalid uncertainty."
        )


    priority = (
        utility -
        uncertainty
    )


    beam_id = (
        target_id % 8
    )


    return BeamAction(
        target_id=target_id,
        beam_id=beam_id,
        priority=priority,
    )

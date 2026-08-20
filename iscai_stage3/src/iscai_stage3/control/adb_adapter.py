from __future__ import annotations

from dataclasses import dataclass


from iscai_stage3.control.temporal_adb import (
    build_temporal_schedule,
)



@dataclass(frozen=True)
class ADBTarget:

    target_id: int

    utility: float

    uncertainty_penalty: float



def prediction_to_adb_target(
    *,
    target_id: int,
    utility: float,
    uncertainty: float,
) -> ADBTarget:

    if uncertainty < 0:
        raise ValueError(
            "Negative uncertainty."
        )

    return ADBTarget(
        target_id=target_id,
        utility=utility,
        uncertainty_penalty=uncertainty,
    )



def build_adb_schedule(
    targets: tuple[ADBTarget, ...],
    *,
    horizon_s: float,
    dt: float,
):

    return build_temporal_schedule(
        targets,
        horizon_s=horizon_s,
        dt=dt,
    )

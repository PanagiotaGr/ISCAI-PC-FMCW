from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from iscai_stage2.observations.detection_set import (
    UnlabeledDetectionFrame,
)


class AssociationMode(str, Enum):
    """
    Frozen identity/association regimes.

    Oracle modes are diagnostics only.
    """

    ORACLE_IDENTITY_DIAGNOSTIC = (
        "oracle_identity_diagnostic"
    )

    ORACLE_ASSOCIATION_DIAGNOSTIC = (
        "oracle_association_diagnostic"
    )

    ESTIMATED_ASSOCIATION = (
        "estimated_association"
    )

    TRACK_BEFORE_DETECT = (
        "track_before_detect"
    )


@dataclass(frozen=True)
class AlgorithmObservationSequence:
    """
    Canonical Stage3 algorithm-facing input.

    The contained frames are the frozen Stage2
    UnlabeledDetectionFrame objects themselves.

    No track IDs.
    No actor classes.
    No truth sidecars.
    No future states.
    """

    scenario_id: str

    frames: tuple[
        UnlabeledDetectionFrame,
        ...
    ]

    source_semantics: str = (
        "stage2_degraded_algorithm_facing"
    )

    measured_fmcw: bool = False

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "scenario_id must be non-empty."
            )

        if not isinstance(
            self.frames,
            tuple,
        ):
            object.__setattr__(
                self,
                "frames",
                tuple(self.frames),
            )

        for frame in self.frames:
            if not isinstance(
                frame,
                UnlabeledDetectionFrame,
            ):
                raise TypeError(
                    "All algorithm frames must be "
                    "Stage2 "
                    "UnlabeledDetectionFrame "
                    "instances."
                )

        if self.measured_fmcw:
            raise ValueError(
                "Stage3 main interface represents "
                "PC-FMCW-like simulated "
                "observations, not measured FMCW."
            )

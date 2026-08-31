from __future__ import annotations

from dataclasses import dataclass

from iscai_stage2.observations.detection_set import (
    DetectionTruthSidecar,
)


@dataclass(frozen=True)
class EvaluatorTruthSequence:
    """
    Evaluator-only truth container.

    Baselines and association algorithms must
    never import or consume this object.
    """

    scenario_id: str

    sidecars: tuple[
        DetectionTruthSidecar,
        ...
    ]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "scenario_id must be non-empty."
            )

        if not isinstance(
            self.sidecars,
            tuple,
        ):
            object.__setattr__(
                self,
                "sidecars",
                tuple(self.sidecars),
            )

        for sidecar in self.sidecars:
            if not isinstance(
                sidecar,
                DetectionTruthSidecar,
            ):
                raise TypeError(
                    "Evaluator truth must contain "
                    "Stage2 DetectionTruthSidecar "
                    "objects only."
                )

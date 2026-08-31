from __future__ import annotations

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)


def algorithm_sequence_from_degraded_scene(
    *,
    scenario_id: str,
    scene,
) -> AlgorithmObservationSequence:
    """
    Extract only the Stage2 algorithm-facing
    frames.

    Evaluator truth sidecars are deliberately
    never accessed here.
    """

    if not hasattr(scene, "frames"):
        raise TypeError(
            "Stage2 degraded scene must expose "
            "algorithm-facing frames."
        )

    measured_fmcw = bool(
        getattr(
            scene,
            "measured_fmcw",
            False,
        )
    )

    if measured_fmcw:
        raise ValueError(
            "Stage2 scene unexpectedly claims "
            "measured FMCW."
        )

    return AlgorithmObservationSequence(
        scenario_id=scenario_id,
        frames=tuple(scene.frames),
        measured_fmcw=False,
    )

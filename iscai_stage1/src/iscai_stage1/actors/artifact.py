"""Stage-1 causal actor artifact.

Track IDs remain bookkeeping/oracle-association metadata and are never placed
inside the numeric realistic feature payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from iscai_stage1.actors.history import (
    CausalActorHistory,
    RawObjectStateW,
    canonicalize_causal_actor_history,
)
from iscai_stage1.actors.roles import (
    ActorRoleMasks,
    compute_actor_role_masks,
)
from iscai_stage1.contracts.stage1a import (
    ObjectClass,
    Stage1ArtifactSemantics,
)


@dataclass(frozen=True)
class ActorMetadata:
    scenario_id: str
    track_id: int
    object_class: ObjectClass
    is_sdc: bool


@dataclass(frozen=True)
class CausalActorArtifact:
    semantics: Stage1ArtifactSemantics
    metadata: ActorMetadata
    history: CausalActorHistory
    roles: ActorRoleMasks

    def realistic_numeric_features(self) -> dict[str, object]:
        """Return causal numeric quantities only.

        Intentionally excludes:
        - scenario_id
        - track_id
        - tracks_to_predict
        - objects_of_interest
        - annotated WOMD velocity
        """

        return {
            "timestamps_s": self.history.timestamps_s,
            "position_W_m": self.history.position_W_m,
            "dimensions_lwh_m": self.history.dimensions_lwh_m,
            "heading_rad": self.history.heading_rad,
            "state_valid": self.history.state_valid,
            "velocity_W_mps": self.history.velocity_W_mps,
            "velocity_valid": self.history.velocity_valid,
        }


def build_causal_actor_artifact(
    *,
    scenario_id: str,
    track_id: int,
    object_class: ObjectClass,
    is_sdc: bool,
    receiver_geometry_valid: bool,
    timestamps_seconds: tuple[float, ...] | list[float],
    states: tuple[RawObjectStateW, ...] | list[RawObjectStateW],
    current_time_index: int,
) -> CausalActorArtifact:
    history = canonicalize_causal_actor_history(
        timestamps_seconds=timestamps_seconds,
        states=states,
        current_time_index=current_time_index,
    )

    # Pass only the already-causal validity prefix.
    roles = compute_actor_role_masks(
        validity=history.state_valid,
        anchor_index=history.anchor_index,
        object_class=object_class,
        is_sdc=is_sdc,
        receiver_geometry_valid=receiver_geometry_valid,
    )

    return CausalActorArtifact(
        semantics=Stage1ArtifactSemantics(),
        metadata=ActorMetadata(
            scenario_id=scenario_id,
            track_id=track_id,
            object_class=object_class,
            is_sdc=is_sdc,
        ),
        history=history,
        roles=roles,
    )
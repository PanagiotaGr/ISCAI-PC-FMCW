"""Deterministic hashing for causal Stage-1 artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from iscai_stage1.actors.artifact import CausalActorArtifact


def causal_actor_payload(artifact: CausalActorArtifact) -> dict[str, object]:
    """Canonical JSON-safe payload for causal hash generation."""

    return {
        "semantics": asdict(artifact.semantics),
        "metadata": asdict(artifact.metadata),
        "history": asdict(artifact.history),
        "roles": asdict(artifact.roles),
    }


def causal_actor_sha256(artifact: CausalActorArtifact) -> str:
    payload = causal_actor_payload(artifact)

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()
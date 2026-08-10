from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from iscai_stage1.lidar.contracts import (
    CausalActorLidarArtifact,
)


def causal_actor_lidar_sha256(
    artifact: CausalActorLidarArtifact,
) -> str:
    encoded = json.dumps(
        asdict(artifact),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()

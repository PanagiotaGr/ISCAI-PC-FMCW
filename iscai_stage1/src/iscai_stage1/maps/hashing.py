from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from iscai_stage1.maps.contracts import (
    CausalMapArtifact,
)


def causal_map_payload(
    artifact: CausalMapArtifact,
) -> dict:
    return asdict(artifact)


def causal_map_sha256(
    artifact: CausalMapArtifact,
) -> str:
    payload = causal_map_payload(
        artifact
    )

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()

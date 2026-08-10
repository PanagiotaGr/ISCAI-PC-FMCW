"""Stage-1 receiver OFFSET geometry only.

No full receiver-position uncertainty is propagated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from iscai_stage1.contracts.stage1a import (
    ReceiverGeometryConfig,
    ReceiverGeometryH0,
    Vec3,
)
from iscai_stage1.geometry.rigid import (
    RigidTransform,
    mat_mul,
    rotation_z,
    transpose,
)


@dataclass(frozen=True)
class ActorAnchorStateW:
    center_w_m: Vec3
    heading_rad: float
    valid: bool


def receiver_geometry_in_H0(
    actor: ActorAnchorStateW,
    T_H0_from_W: RigidTransform,
    config: ReceiverGeometryConfig,
) -> ReceiverGeometryH0:
    """Transform body-frame receiver offset mean/covariance into H0."""

    if not actor.valid:
        raise ValueError("Receiver geometry requires a valid anchor actor state.")

    if not all(isfinite(value) for value in (*actor.center_w_m, actor.heading_rad)):
        raise ValueError("Actor receiver geometry contains non-finite values.")

    # Actor body -> W.
    R_W_from_actor = rotation_z(actor.heading_rad)

    # Actor body -> H0.
    R_H0_from_actor = mat_mul(
        T_H0_from_W.rotation,
        R_W_from_actor,
    )

    offset_mean_H0 = tuple(
        sum(
            R_H0_from_actor[row][col] * config.offset_mean_body_m[col]
            for col in range(3)
        )
        for row in range(3)
    )

    temp = mat_mul(
        R_H0_from_actor,
        config.offset_covariance_body_m2,
    )
    offset_covariance_H0 = mat_mul(
        temp,
        transpose(R_H0_from_actor),
    )

    actor_center_H0 = T_H0_from_W.apply_point(actor.center_w_m)

    receiver_point_mean_H0 = (
        actor_center_H0[0] + offset_mean_H0[0],
        actor_center_H0[1] + offset_mean_H0[1],
        actor_center_H0[2] + offset_mean_H0[2],
    )

    return ReceiverGeometryH0(
        receiver_offset_mean_H0=offset_mean_H0,  # type: ignore[arg-type]
        receiver_offset_covariance_H0=offset_covariance_H0,
        receiver_point_mean_H0=receiver_point_mean_H0,
        receiver_geometry_mode=config.mode,
        receiver_geometry_valid=True,
    )
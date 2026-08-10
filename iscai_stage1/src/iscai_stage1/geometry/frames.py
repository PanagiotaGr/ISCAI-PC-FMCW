"""Frozen W / E0 / H0 / Ht coordinate-frame definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, isfinite

from iscai_stage1.contracts.stage1a import HeadlampSurrogateConfig, Vec3
from iscai_stage1.geometry.rigid import (
    RigidTransform,
    compose,
    rotation_rpy,
    rotation_z,
)


@dataclass(frozen=True)
class SdcStateW:
    """Minimal causal SDC state needed for Stage-1 frame construction."""

    center_w_m: Vec3
    heading_rad: float
    length_m: float
    valid: bool = True


@dataclass(frozen=True)
class AnchorFrames:
    T_W_from_E0: RigidTransform
    T_E0_from_W: RigidTransform
    T_W_from_H0: RigidTransform
    T_H0_from_W: RigidTransform


@dataclass(frozen=True)
class DynamicHeadlampFrame:
    T_W_from_Ht: RigidTransform
    T_Ht_from_W: RigidTransform


def _validate_sdc_state(state: SdcStateW) -> None:
    values = (*state.center_w_m, state.heading_rad, state.length_m)

    if not state.valid:
        raise ValueError("Cannot construct frame from invalid SDC state.")
    if not all(isfinite(value) for value in values):
        raise ValueError("SDC frame state contains non-finite values.")
    if state.length_m <= 0.0:
        raise ValueError("SDC length must be positive.")


def build_T_W_from_E0(anchor: SdcStateW) -> RigidTransform:
    """E0 origin is the anchor SDC box centre."""
    _validate_sdc_state(anchor)

    return RigidTransform(
        rotation=rotation_z(anchor.heading_rad),
        translation=anchor.center_w_m,
    )


def build_T_E_from_H(
    sdc_length_m: float,
    config: HeadlampSurrogateConfig,
) -> RigidTransform:
    """Static configurable headlamp-surrogate extrinsic."""
    return RigidTransform(
        rotation=rotation_rpy(
            config.roll_rad,
            config.pitch_rad,
            config.yaw_rad,
        ),
        translation=config.translation_in_sdc_m(sdc_length_m),
    )


def build_anchor_frames(
    anchor: SdcStateW,
    config: HeadlampSurrogateConfig,
) -> AnchorFrames:
    T_W_from_E0 = build_T_W_from_E0(anchor)
    T_E0_from_W = T_W_from_E0.inverse()

    T_E0_from_H0 = build_T_E_from_H(anchor.length_m, config)
    T_W_from_H0 = compose(T_W_from_E0, T_E0_from_H0)
    T_H0_from_W = T_W_from_H0.inverse()

    return AnchorFrames(
        T_W_from_E0=T_W_from_E0,
        T_E0_from_W=T_E0_from_W,
        T_W_from_H0=T_W_from_H0,
        T_H0_from_W=T_H0_from_W,
    )


def build_dynamic_headlamp_frame(
    state_t: SdcStateW,
    config: HeadlampSurrogateConfig,
) -> DynamicHeadlampFrame:
    """Construct Ht using only the causal SDC state at timestep t."""
    T_W_from_Et = build_T_W_from_E0(state_t)
    T_Et_from_Ht = build_T_E_from_H(state_t.length_m, config)

    T_W_from_Ht = compose(T_W_from_Et, T_Et_from_Ht)

    return DynamicHeadlampFrame(
        T_W_from_Ht=T_W_from_Ht,
        T_Ht_from_W=T_W_from_Ht.inverse(),
    )


def horizontal_bearing_rad(point_H: Vec3) -> float:
    """theta = atan2(y_H, x_H): ahead=0, left>0, right<0."""
    return atan2(point_H[1], point_H[0])
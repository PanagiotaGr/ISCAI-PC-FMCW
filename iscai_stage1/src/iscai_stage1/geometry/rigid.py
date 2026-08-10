"""Minimal dependency-free rigid-transform utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from iscai_stage1.contracts.stage1a import Mat3, Vec3


IDENTITY_3: Mat3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def mat_vec(matrix: Mat3, vector: Vec3) -> Vec3:
    return tuple(
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def mat_mul(left: Mat3, right: Mat3) -> Mat3:
    return tuple(
        tuple(
            sum(left[row][k] * right[k][col] for k in range(3))
            for col in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def transpose(matrix: Mat3) -> Mat3:
    return tuple(
        tuple(matrix[col][row] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def vec_add(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def vec_sub(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def rotation_z(yaw_rad: float) -> Mat3:
    """Right-handed yaw for x-forward, y-left, z-up frames."""
    c = cos(yaw_rad)
    s = sin(yaw_rad)
    return (
        (c, -s, 0.0),
        (s, c, 0.0),
        (0.0, 0.0, 1.0),
    )


def rotation_rpy(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> Mat3:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll).

    This implementation convention is immaterial to the frozen baseline,
    whose relative rotation is identity, but makes future configuration
    deterministic.
    """
    cr, sr = cos(roll_rad), sin(roll_rad)
    cp, sp = cos(pitch_rad), sin(pitch_rad)
    cy, sy = cos(yaw_rad), sin(yaw_rad)

    rx: Mat3 = (
        (1.0, 0.0, 0.0),
        (0.0, cr, -sr),
        (0.0, sr, cr),
    )
    ry: Mat3 = (
        (cp, 0.0, sp),
        (0.0, 1.0, 0.0),
        (-sp, 0.0, cp),
    )
    rz: Mat3 = (
        (cy, -sy, 0.0),
        (sy, cy, 0.0),
        (0.0, 0.0, 1.0),
    )

    return mat_mul(mat_mul(rz, ry), rx)


@dataclass(frozen=True)
class RigidTransform:
    """Transform with convention p_dst = R_dst_from_src p_src + t."""

    rotation: Mat3
    translation: Vec3

    def apply_point(self, point_src: Vec3) -> Vec3:
        return vec_add(mat_vec(self.rotation, point_src), self.translation)

    def apply_vector(self, vector_src: Vec3) -> Vec3:
        """Rotate a vector; translation is intentionally ignored."""
        return mat_vec(self.rotation, vector_src)

    def inverse(self) -> "RigidTransform":
        rotation_inv = transpose(self.rotation)
        translation_inv = mat_vec(
            rotation_inv,
            (
                -self.translation[0],
                -self.translation[1],
                -self.translation[2],
            ),
        )
        return RigidTransform(rotation_inv, translation_inv)


def compose(
    T_A_from_B: RigidTransform,
    T_B_from_C: RigidTransform,
) -> RigidTransform:
    """Compose T_A_from_B ∘ T_B_from_C -> T_A_from_C."""
    rotation = mat_mul(T_A_from_B.rotation, T_B_from_C.rotation)
    translation = vec_add(
        mat_vec(T_A_from_B.rotation, T_B_from_C.translation),
        T_A_from_B.translation,
    )
    return RigidTransform(rotation, translation)
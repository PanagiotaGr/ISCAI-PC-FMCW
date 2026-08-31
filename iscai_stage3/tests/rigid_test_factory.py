from __future__ import annotations

from dataclasses import dataclass


def _transpose(R):
    return tuple(
        tuple(
            R[j][i]
            for j in range(3)
        )
        for i in range(3)
    )


def _matvec(R, v):
    return tuple(
        sum(
            R[i][j] * v[j]
            for j in range(3)
        )
        for i in range(3)
    )


@dataclass(frozen=True)
class FakeRigidTransform:
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    translation: tuple[
        float,
        float,
        float,
    ]

    def apply_point(self, point):
        rotated = _matvec(
            self.rotation,
            point,
        )

        return tuple(
            rotated[i]
            +
            self.translation[i]
            for i in range(3)
        )

    def apply_vector(self, vector):
        return _matvec(
            self.rotation,
            vector,
        )

    def inverse(self):
        RT = _transpose(
            self.rotation
        )

        translated = _matvec(
            RT,
            tuple(
                -x
                for x in self.translation
            ),
        )

        return FakeRigidTransform(
            rotation=RT,
            translation=translated,
        )


IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def identity_transform():
    return FakeRigidTransform(
        rotation=IDENTITY_ROTATION,
        translation=(0.0, 0.0, 0.0),
    )


def T_sensor_from_W(
    *,
    sensor_origin_W,
):
    """
    Translation-only W -> sensor transform.

    p_sensor = p_W - sensor_origin_W
    """

    return FakeRigidTransform(
        rotation=IDENTITY_ROTATION,
        translation=tuple(
            -float(x)
            for x in sensor_origin_W
        ),
    )

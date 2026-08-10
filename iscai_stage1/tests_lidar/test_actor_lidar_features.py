import unittest

import numpy as np

from iscai_stage1.lidar.actor_features import (
    _actor_box_mask,
    _stats,
)


class State:
    valid = True
    center_x = 0.0
    center_y = 0.0
    center_z = 0.0
    length = 4.0
    width = 2.0
    height = 2.0
    heading = 0.0


class TestActorLidarFeatures(unittest.TestCase):

    def test_scalar_stats(self):
        x = np.asarray(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        )

        s = _stats(x)

        self.assertEqual(s.minimum, 1.0)
        self.assertEqual(s.maximum, 3.0)
        self.assertEqual(s.mean, 2.0)

    def test_box_mask(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.9, 0.9, 0.9],
                [2.1, 0.0, 0.0],
                [0.0, 1.1, 0.0],
            ],
            dtype=np.float64,
        )

        mask, local = _actor_box_mask(
            points,
            State(),
        )

        self.assertEqual(
            mask.tolist(),
            [True, True, False, False],
        )

        np.testing.assert_allclose(
            local[0],
            [0.0, 0.0, 0.0],
        )

    def test_rotated_box(self):
        state = State()
        state.heading = np.pi / 2.0

        points = np.asarray(
            [
                [0.0, 1.9, 0.0],
                [1.1, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        mask, _ = _actor_box_mask(
            points,
            state,
        )

        self.assertEqual(
            mask.tolist(),
            [True, False],
        )


if __name__ == "__main__":
    unittest.main()

import inspect
import unittest
import numpy as np

from iscai_stage6.adb.class_aware_policy import (
    RateLimitedSchedule,
)

from iscai_stage6.adb.budgeted_residual_v2 import (
    budgeted_residual_projection,
)


def schedule(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    return RateLimitedSchedule(
        illumination=x,
        safety_override_mask=
            np.zeros_like(
                x,
                dtype=bool,
            ),
        safety_override_count=0,
        safety_override_fraction=0.0,
        semantics="unit-test",
    )


class Stage6V2BudgetedResidualTest(
    unittest.TestCase
):
    def test_budget_is_hard_spatial_bound(self):
        base = np.ones(
            (2, 5),
            dtype=float,
        )

        old = np.ones(
            (4, 2, 5),
            dtype=float,
        )

        old[:, 0, :5] = 0.2

        out = budgeted_residual_projection(
            schedule(old),
            current_illumination=base,
            budget_fraction=0.2,
        )

        # 10 cells * 0.2 -> exactly 2 extra
        extra = (
            (out.illumination < 1.0)
            &
            (base >= 1.0)[
                None,
                ...
            ]
        )

        self.assertLessEqual(
            int(
                np.max(
                    np.count_nonzero(
                        extra.reshape(
                            4,
                            -1,
                        ),
                        axis=1,
                    )
                )
            ),
            2,
        )

    def test_reactive_cells_are_held_exactly(self):
        base = np.ones(
            (2, 4),
            dtype=float,
        )

        base[
            0,
            0,
        ] = 0.35

        old = np.ones(
            (4, 2, 4),
            dtype=float,
        )

        old[
            :,
            0,
            0,
        ] = 0.1

        old[
            :,
            0,
            1,
        ] = 0.2

        out = budgeted_residual_projection(
            schedule(old),
            current_illumination=base,
            budget_fraction=0.5,
        )

        np.testing.assert_array_equal(
            out.illumination[
                :,
                0,
                0,
            ],
            np.full(
                4,
                0.35,
            ),
        )

    def test_selected_cell_keeps_exact_old_time_sequence(self):
        base = np.ones(
            (2, 4),
            dtype=float,
        )

        old = np.ones(
            (4, 2, 4),
            dtype=float,
        )

        seq = np.asarray(
            [
                0.9,
                0.7,
                0.5,
                0.4,
            ]
        )

        old[
            :,
            0,
            2,
        ] = seq

        out = budgeted_residual_projection(
            schedule(old),
            current_illumination=base,
            budget_fraction=0.5,
        )

        np.testing.assert_array_equal(
            out.illumination[
                :,
                0,
                2,
            ],
            seq,
        )

    def test_VRU_floor_guard_excludes_cell(self):
        base = np.ones(
            (2, 4),
            dtype=float,
        )

        old = np.ones(
            (4, 2, 4),
            dtype=float,
        )

        old[
            :,
            0,
            1,
        ] = 0.15

        guard = np.zeros_like(
            old
        )

        guard[
            :,
            0,
            1,
        ] = 1.0

        out = budgeted_residual_projection(
            schedule(old),
            current_illumination=base,
            vru_floor_guard=guard,
            budget_fraction=1.0,
        )

        np.testing.assert_array_equal(
            out.illumination[
                :,
                0,
                1,
            ],
            np.ones(4),
        )

    def test_controller_API_has_no_oracle_or_truth_input(self):
        sig = inspect.signature(
            budgeted_residual_projection
        )

        text = str(sig).lower()

        self.assertNotIn(
            "oracle",
            text,
        )

        self.assertNotIn(
            "truth",
            text,
        )

        self.assertNotIn(
            "future_gt",
            text,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from iscai_stage3.baselines import (
    predict_ca,
)

from iscai_stage3.geometry import (
    CartesianObservation,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from iscai_stage3.state import (
    estimate_causal_ca_state,
)

from stage2_test_factory import (
    make_detection,
)


P = (
    (0.1, 0.0, 0.0),
    (0.0, 0.1, 0.0),
    (0.0, 0.0, 0.1),
)


def obs(
    frame,
    time_s,
    x,
):
    measurement = snapshot_from_detection(
        make_detection(
            key=f"k{frame}",
            range_m=max(
                abs(x),
                0.1,
            ),
            vr_mps=0.0,
            az_rad=0.0,
        )
    )

    return CartesianObservation(
        track_id="track",
        frame_index=frame,
        timestamp_s=time_s,
        detection_key=f"k{frame}",
        position_H0_m=(
            x,
            0.0,
            0.0,
        ),
        position_covariance_H0_m2=P,
        measurement=measurement,
    )


def trajectory_x(t):
    # x(t) = 5 + 2t + 0.5*4*t^2
    return (
        5.0
        +
        2.0 * t
        +
        2.0 * t * t
    )


class TestCABaseline(unittest.TestCase):

    def test_constant_acceleration_closed_form(
        self,
    ):
        state = estimate_causal_ca_state(
            (
                obs(
                    0,
                    0.0,
                    trajectory_x(0.0),
                ),
                obs(
                    1,
                    0.1,
                    trajectory_x(0.1),
                ),
                obs(
                    2,
                    0.2,
                    trajectory_x(0.2),
                ),
            )
        )

        self.assertAlmostEqual(
            state.acceleration_H0_mps2[0],
            4.0,
            places=10,
        )

        self.assertAlmostEqual(
            state.velocity_H0_mps[0],
            2.8,
            places=10,
        )

        prediction = predict_ca(
            state,
            horizons_s=(0.3,),
        )

        self.assertAlmostEqual(
            prediction.points[0]
            .position_H0_m[0],
            trajectory_x(0.5),
            places=10,
        )

    def test_irregular_timestamps(
        self,
    ):
        def x(t):
            # a = 2 m/s²
            return (
                1.0
                +
                3.0 * t
                +
                t * t
            )

        state = estimate_causal_ca_state(
            (
                obs(0, 0.0, x(0.0)),
                obs(1, 0.15, x(0.15)),
                obs(2, 0.4, x(0.4)),
            )
        )

        self.assertAlmostEqual(
            state.acceleration_H0_mps2[0],
            2.0,
            places=10,
        )

        self.assertAlmostEqual(
            state.velocity_H0_mps[0],
            3.8,
            places=10,
        )

    def test_need_three_observations(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            estimate_causal_ca_state(
                (
                    obs(0, 0.0, 1.0),
                    obs(1, 0.1, 1.1),
                )
            )

    def test_covariance_is_symmetric(
        self,
    ):
        state = estimate_causal_ca_state(
            (
                obs(0, 0.0, 1.0),
                obs(1, 0.1, 1.1),
                obs(2, 0.2, 1.3),
            )
        )

        P9 = state.covariance_9x9

        for i in range(9):
            self.assertGreaterEqual(
                P9[i][i],
                0.0,
            )

            for j in range(9):
                self.assertAlmostEqual(
                    P9[i][j],
                    P9[j][i],
                )

    def test_deterministic_and_causal(
        self,
    ):
        state = estimate_causal_ca_state(
            (
                obs(0, 0.0, 1.0),
                obs(1, 0.1, 1.1),
                obs(2, 0.2, 1.3),
            )
        )

        first = predict_ca(
            state,
            horizons_s=(
                0.1,
                0.3,
            ),
        )

        second = predict_ca(
            state,
            horizons_s=(
                0.1,
                0.3,
            ),
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertFalse(
            first.annotated_velocity_used
        )

        self.assertFalse(
            first.annotated_acceleration_used
        )

        self.assertFalse(
            first.future_information_used
        )


if __name__ == "__main__":
    unittest.main()

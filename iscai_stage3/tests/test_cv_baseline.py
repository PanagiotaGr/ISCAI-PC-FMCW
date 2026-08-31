import unittest

from iscai_stage3.baselines import (
    predict_cv,
)

from iscai_stage3.geometry import (
    CartesianObservation,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from iscai_stage3.state import (
    estimate_causal_cv_state,
)

from stage2_test_factory import (
    make_detection,
)


P = (
    (0.1, 0.0, 0.0),
    (0.0, 0.1, 0.0),
    (0.0, 0.0, 0.1),
)


def observation(
    *,
    frame,
    time,
    x,
):
    measurement = (
        snapshot_from_detection(
            make_detection(
                key=f"k{frame}",
                range_m=max(
                    0.1,
                    abs(x),
                ),
                vr_mps=0.0,
                az_rad=0.0,
            )
        )
    )

    return CartesianObservation(
        track_id="track",
        frame_index=frame,
        timestamp_s=time,
        detection_key=f"k{frame}",
        position_H0_m=(
            x,
            0.0,
            0.0,
        ),
        position_covariance_H0_m2=P,
        measurement=measurement,
    )


class TestCVBaseline(unittest.TestCase):

    def test_cv_closed_form(self):
        state = estimate_causal_cv_state(
            (
                observation(
                    frame=0,
                    time=0.0,
                    x=10.0,
                ),
                observation(
                    frame=1,
                    time=0.1,
                    x=11.0,
                ),
            )
        )

        prediction = predict_cv(
            state,
            horizons_s=(
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

        expected_x = (
            12.0,
            14.0,
            16.0,
            21.0,
        )

        for point, expected in zip(
            prediction.points,
            expected_x,
        ):
            self.assertAlmostEqual(
                point.position_H0_m[0],
                expected,
            )

    def test_need_two_causal_observations(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            estimate_causal_cv_state(
                (
                    observation(
                        frame=0,
                        time=0.0,
                        x=10.0,
                    ),
                )
            )

    def test_state_covariance_symmetric(
        self,
    ):
        state = estimate_causal_cv_state(
            (
                observation(
                    frame=0,
                    time=0.0,
                    x=10.0,
                ),
                observation(
                    frame=1,
                    time=0.1,
                    x=11.0,
                ),
            )
        )

        P6 = state.covariance_6x6

        for i in range(6):
            self.assertGreaterEqual(
                P6[i][i],
                0.0,
            )

            for j in range(6):
                self.assertAlmostEqual(
                    P6[i][j],
                    P6[j][i],
                )

    def test_negative_horizon_rejected(
        self,
    ):
        state = estimate_causal_cv_state(
            (
                observation(
                    frame=0,
                    time=0.0,
                    x=10.0,
                ),
                observation(
                    frame=1,
                    time=0.1,
                    x=11.0,
                ),
            )
        )

        with self.assertRaises(
            ValueError
        ):
            predict_cv(
                state,
                horizons_s=(-0.1,),
            )

    def test_prediction_is_deterministic(
        self,
    ):
        state = estimate_causal_cv_state(
            (
                observation(
                    frame=0,
                    time=0.0,
                    x=10.0,
                ),
                observation(
                    frame=1,
                    time=0.1,
                    x=11.0,
                ),
            )
        )

        first = predict_cv(
            state,
            horizons_s=(
                0.1,
                0.3,
            ),
        )

        second = predict_cv(
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
            first.future_information_used
        )


if __name__ == "__main__":
    unittest.main()

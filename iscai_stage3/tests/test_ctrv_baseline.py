import math
import unittest

from iscai_stage3.baselines import (
    predict_ctrv,
)

from iscai_stage3.geometry import (
    CartesianObservation,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from iscai_stage3.state import (
    CausalCTRVState,
    estimate_causal_ctrv_state,
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
    y,
    z=0.0,
):
    measurement = snapshot_from_detection(
        make_detection(
            key=f"k{frame}",
            range_m=max(
                math.sqrt(
                    x*x + y*y + z*z
                ),
                0.1,
            ),
            vr_mps=0.0,
            az_rad=math.atan2(
                y,
                x,
            ),
        )
    )

    return CartesianObservation(
        track_id="track",
        frame_index=frame,
        timestamp_s=time_s,
        detection_key=f"k{frame}",
        position_H0_m=(
            x,
            y,
            z,
        ),
        position_covariance_H0_m2=P,
        measurement=measurement,
    )


class TestCTRVBaseline(unittest.TestCase):

    def test_straight_motion_zero_turn(
        self,
    ):
        state = estimate_causal_ctrv_state(
            (
                obs(
                    0,
                    0.0,
                    0.0,
                    0.0,
                ),
                obs(
                    1,
                    1.0,
                    1.0,
                    0.0,
                ),
                obs(
                    2,
                    2.0,
                    2.0,
                    0.0,
                ),
            )
        )

        self.assertAlmostEqual(
            state.turn_rate_radps,
            0.0,
            places=12,
        )

        self.assertAlmostEqual(
            state.heading_rad,
            0.0,
            places=12,
        )

        prediction = predict_ctrv(
            state,
            horizons_s=(1.0,),
        )

        self.assertAlmostEqual(
            prediction.points[0]
            .position_H0_m[0],
            3.0,
            places=10,
        )

        self.assertAlmostEqual(
            prediction.points[0]
            .position_H0_m[1],
            0.0,
            places=10,
        )

    def test_exact_ctrv_propagation(
        self,
    ):
        state = CausalCTRVState(
            track_id="track",
            timestamp_s=0.0,
            position_H0_m=(
                0.0,
                0.0,
                0.0,
            ),
            planar_speed_mps=10.0,
            heading_rad=0.0,
            turn_rate_radps=1.0,
            vertical_velocity_mps=0.0,
            source_frame_indices=(
                0,
                1,
                2,
            ),
        )

        prediction = predict_ctrv(
            state,
            horizons_s=(0.5,),
        )

        point = prediction.points[0]

        self.assertAlmostEqual(
            point.position_H0_m[0],
            10.0
            * math.sin(0.5),
            places=10,
        )

        self.assertAlmostEqual(
            point.position_H0_m[1],
            10.0
            * (
                1.0
                -
                math.cos(0.5)
            ),
            places=10,
        )

    def test_heading_wrap_179_to_minus179(
        self,
    ):
        h1 = math.radians(179.0)
        h2 = math.radians(-179.0)

        v1 = (
            math.cos(h1),
            math.sin(h1),
        )

        v2 = (
            math.cos(h2),
            math.sin(h2),
        )

        p0 = (
            0.0,
            0.0,
        )

        p1 = (
            p0[0] + v1[0],
            p0[1] + v1[1],
        )

        p2 = (
            p1[0] + v2[0],
            p1[1] + v2[1],
        )

        state = estimate_causal_ctrv_state(
            (
                obs(
                    0,
                    0.0,
                    p0[0],
                    p0[1],
                ),
                obs(
                    1,
                    1.0,
                    p1[0],
                    p1[1],
                ),
                obs(
                    2,
                    2.0,
                    p2[0],
                    p2[1],
                ),
            )
        )

        self.assertAlmostEqual(
            state.turn_rate_radps,
            math.radians(2.0),
            places=10,
        )

        self.assertLess(
            abs(
                state.turn_rate_radps
            ),
            0.1,
        )

    def test_stationary_low_speed_fallback(
        self,
    ):
        state = estimate_causal_ctrv_state(
            (
                obs(
                    0,
                    0.0,
                    5.0,
                    2.0,
                ),
                obs(
                    1,
                    0.1,
                    5.0,
                    2.0,
                ),
                obs(
                    2,
                    0.2,
                    5.0,
                    2.0,
                ),
            )
        )

        self.assertEqual(
            state.planar_speed_mps,
            0.0,
        )

        self.assertEqual(
            state.turn_rate_radps,
            0.0,
        )

        prediction = predict_ctrv(
            state,
            horizons_s=(1.0,),
        )

        self.assertAlmostEqual(
            prediction.points[0]
            .position_H0_m[0],
            5.0,
        )

        self.assertAlmostEqual(
            prediction.points[0]
            .position_H0_m[1],
            2.0,
        )

    def test_need_three_observations(
        self,
    ):
        with self.assertRaises(
            ValueError
        ):
            estimate_causal_ctrv_state(
                (
                    obs(
                        0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    obs(
                        1,
                        0.1,
                        1.0,
                        0.0,
                    ),
                )
            )

    def test_deterministic_and_causal(
        self,
    ):
        state = estimate_causal_ctrv_state(
            (
                obs(
                    0,
                    0.0,
                    0.0,
                    0.0,
                ),
                obs(
                    1,
                    0.1,
                    1.0,
                    0.0,
                ),
                obs(
                    2,
                    0.2,
                    2.0,
                    0.1,
                ),
            )
        )

        first = predict_ctrv(
            state,
            horizons_s=(
                0.1,
                0.3,
            ),
        )

        second = predict_ctrv(
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
            first.annotated_heading_used
        )

        self.assertFalse(
            first.future_information_used
        )


if __name__ == "__main__":
    unittest.main()

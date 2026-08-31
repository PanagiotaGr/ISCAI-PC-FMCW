import math
import unittest

import numpy as np

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.baselines import (
    predict_kalman,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.filters import (
    KalmanConfig,
    KalmanFrameContext,
    KalmanState,
    cv_process_covariance,
    cv_transition_matrix,
    filter_associated_track,
    measurement_model_and_jacobian,
    predict_kalman_state,
    update_kalman_state,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from rigid_test_factory import (
    T_sensor_from_W,
    identity_transform,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


CONFIG = KalmanConfig(
    acceleration_spectral_density_m2_s3=0.0
)


def frame_context(
    *,
    sensor_x,
    times,
):
    return KalmanFrameContext(
        transforms=FrameTransformContext(
            T_H0_from_W=(
                identity_transform()
            ),
            T_Ht_from_W_by_frame=tuple(
                T_sensor_from_W(
                    sensor_origin_W=(
                        x,
                        0.0,
                        0.0,
                    )
                )
                for x in sensor_x
            ),
        ),
        frame_timestamps_s=tuple(
            times
        ),
    )


class TestKalmanEKF(unittest.TestCase):

    def test_cv_transition_matrix(
        self,
    ):
        F = cv_transition_matrix(
            0.2
        )

        self.assertAlmostEqual(
            F[0, 3],
            0.2,
        )

        self.assertAlmostEqual(
            F[1, 4],
            0.2,
        )

        self.assertAlmostEqual(
            F[2, 5],
            0.2,
        )

    def test_process_covariance_zero_q(
        self,
    ):
        Q = cv_process_covariance(
            0.1,
            acceleration_spectral_density_m2_s3=0.0,
        )

        self.assertTrue(
            np.allclose(
                Q,
                0.0,
            )
        )

    def test_measurement_model_static_sensor(
        self,
    ):
        context = frame_context(
            sensor_x=(0.0, 0.0),
            times=(0.0, 0.1),
        )

        state = KalmanState(
            timestamp_s=0.1,
            mean_6=(
                10.0,
                0.0,
                0.0,
                3.0,
                0.0,
                0.0,
            ),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        h, _ = (
            measurement_model_and_jacobian(
                state=state,
                frame_index=1,
                context=context,
                config=CONFIG,
            )
        )

        self.assertAlmostEqual(
            h[0],
            10.0,
        )

        self.assertAlmostEqual(
            h[1],
            3.0,
        )

        self.assertAlmostEqual(
            h[2],
            0.0,
        )

        self.assertAlmostEqual(
            h[3],
            0.0,
        )

    def test_measurement_model_moving_sensor(
        self,
    ):
        context = frame_context(
            sensor_x=(0.0, 1.0),
            times=(0.0, 0.1),
        )

        # Sensor velocity = 10 m/s.
        # Actor velocity = 15 m/s.
        # Relative radial velocity = +5 m/s.
        state = KalmanState(
            timestamp_s=0.1,
            mean_6=(
                11.5,
                0.0,
                0.0,
                15.0,
                0.0,
                0.0,
            ),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        h, _ = (
            measurement_model_and_jacobian(
                state=state,
                frame_index=1,
                context=context,
                config=CONFIG,
            )
        )

        self.assertAlmostEqual(
            h[0],
            10.5,
        )

        self.assertAlmostEqual(
            h[1],
            5.0,
        )

    def test_analytic_jacobian_matches_numeric(
        self,
    ):
        context = frame_context(
            sensor_x=(0.0, 0.5),
            times=(0.0, 0.1),
        )

        base = np.asarray(
            [
                12.0,
                3.0,
                1.5,
                4.0,
                0.5,
                0.2,
            ],
            dtype=np.float64,
        )

        P = tuple(
            tuple(
                1.0 if i == j else 0.0
                for j in range(6)
            )
            for i in range(6)
        )

        state = KalmanState(
            timestamp_s=0.1,
            mean_6=tuple(base),
            covariance_6x6=P,
        )

        h, H = (
            measurement_model_and_jacobian(
                state=state,
                frame_index=1,
                context=context,
                config=CONFIG,
            )
        )

        numeric = np.zeros(
            (4, 6),
            dtype=np.float64,
        )

        eps = 1e-6

        for j in range(6):
            plus = base.copy()
            minus = base.copy()

            plus[j] += eps
            minus[j] -= eps

            hp, _ = (
                measurement_model_and_jacobian(
                    state=KalmanState(
                        timestamp_s=0.1,
                        mean_6=tuple(plus),
                        covariance_6x6=P,
                    ),
                    frame_index=1,
                    context=context,
                    config=CONFIG,
                )
            )

            hm, _ = (
                measurement_model_and_jacobian(
                    state=KalmanState(
                        timestamp_s=0.1,
                        mean_6=tuple(minus),
                        covariance_6x6=P,
                    ),
                    frame_index=1,
                    context=context,
                    config=CONFIG,
                )
            )

            delta = hp - hm

            delta[2] = (
                (delta[2] + math.pi)
                %
                (2.0 * math.pi)
                -
                math.pi
            )

            numeric[:, j] = (
                delta
                /
                (2.0 * eps)
            )

        self.assertTrue(
            np.allclose(
                H,
                numeric,
                atol=2e-5,
                rtol=2e-5,
            ),
            msg=f"\nanalytic=\n{H}\nnumeric=\n{numeric}",
        )

    def test_zero_innovation_preserves_mean(
        self,
    ):
        context = frame_context(
            sensor_x=(0.0, 1.0),
            times=(0.0, 0.1),
        )

        prior = KalmanState(
            timestamp_s=0.1,
            mean_6=(
                11.5,
                0.0,
                0.0,
                15.0,
                0.0,
                0.0,
            ),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        measurement = (
            snapshot_from_detection(
                make_detection(
                    key="m",
                    range_m=10.5,
                    vr_mps=5.0,
                    az_rad=0.0,
                    el_rad=0.0,
                )
            )
        )

        result = update_kalman_state(
            prior,
            measurement=measurement,
            frame_index=1,
            context=context,
            config=CONFIG,
        )

        self.assertTrue(
            np.allclose(
                result.posterior.mean_6,
                prior.mean_6,
                atol=1e-10,
            )
        )

    def test_stage2_R_t_changes_update_strength(
        self,
    ):
        context = frame_context(
            sensor_x=(0.0, 0.0),
            times=(0.0, 0.1),
        )

        prior = KalmanState(
            timestamp_s=0.1,
            mean_6=(
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        low_R = snapshot_from_detection(
            make_detection(
                key="low",
                range_m=11.0,
                vr_mps=0.0,
                az_rad=0.0,
                variance_scale=0.01,
            )
        )

        high_R = snapshot_from_detection(
            make_detection(
                key="high",
                range_m=11.0,
                vr_mps=0.0,
                az_rad=0.0,
                variance_scale=100.0,
            )
        )

        low = update_kalman_state(
            prior,
            measurement=low_R,
            frame_index=1,
            context=context,
            config=CONFIG,
        )

        high = update_kalman_state(
            prior,
            measurement=high_R,
            frame_index=1,
            context=context,
            config=CONFIG,
        )

        low_shift = abs(
            low.posterior.mean_6[0]
            -
            prior.mean_6[0]
        )

        high_shift = abs(
            high.posterior.mean_6[0]
            -
            prior.mean_6[0]
        )

        self.assertGreater(
            low_shift,
            high_shift,
        )

    def test_joseph_covariance_symmetric(
        self,
    ):
        context = frame_context(
            sensor_x=(0.0, 0.0),
            times=(0.0, 0.1),
        )

        prior = KalmanState(
            timestamp_s=0.1,
            mean_6=(
                10.0,
                1.0,
                0.2,
                2.0,
                0.0,
                0.0,
            ),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        measurement = (
            snapshot_from_detection(
                make_detection(
                    key="m",
                    range_m=10.2,
                    vr_mps=2.0,
                    az_rad=0.1,
                    el_rad=0.02,
                )
            )
        )

        result = update_kalman_state(
            prior,
            measurement=measurement,
            frame_index=1,
            context=context,
            config=CONFIG,
        )

        P = np.asarray(
            result.posterior
            .covariance_6x6
        )

        self.assertTrue(
            np.allclose(
                P,
                P.T,
                atol=1e-10,
            )
        )

        eigenvalues = np.linalg.eigvalsh(
            P
        )

        self.assertGreaterEqual(
            eigenvalues.min(),
            -1e-9,
        )

    def test_full_track_filter_moving_sensor(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
        )

        sensor_x = (
            0.0,
            1.0,
            2.0,
            3.0,
        )

        actor_x = (
            10.0,
            11.5,
            13.0,
            14.5,
        )

        frames = []

        for i, t in enumerate(times):
            relative_range = (
                actor_x[i]
                -
                sensor_x[i]
            )

            frames.append(
                make_frame(
                    timestamp_s=t,
                    detections=(
                        make_detection(
                            key=f"k{i}",
                            range_m=(
                                relative_range
                            ),
                            vr_mps=5.0,
                            az_rad=0.0,
                            el_rad=0.0,
                        ),
                    ),
                )
            )

        associated = (
            associate_estimated_gnn(
                AlgorithmObservationSequence(
                    scenario_id="kalman",
                    frames=tuple(frames),
                )
            )
        )

        self.assertEqual(
            len(associated.tracks),
            1,
        )

        context = frame_context(
            sensor_x=sensor_x,
            times=times,
        )

        result = filter_associated_track(
            associated.tracks[0],
            context=context,
            config=CONFIG,
        )

        self.assertEqual(
            len(result.updates),
            2,
        )

        self.assertAlmostEqual(
            result.current_state.mean_6[0],
            14.5,
            places=8,
        )

        self.assertAlmostEqual(
            result.current_state.mean_6[3],
            15.0,
            places=8,
        )

        prediction = predict_kalman(
            track_id=result.track_id,
            state=result.current_state,
            horizons_s=(0.5,),
            config=CONFIG,
        )

        self.assertAlmostEqual(
            prediction.points[0]
            .position_H0_m[0],
            22.0,
            places=8,
        )

    def test_prediction_is_deterministic_and_causal(
        self,
    ):
        state = KalmanState(
            timestamp_s=1.0,
            mean_6=(
                10.0,
                0.0,
                0.0,
                2.0,
                0.0,
                0.0,
            ),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        config = KalmanConfig()

        first = predict_kalman(
            track_id="track",
            state=state,
            horizons_s=(0.1, 0.3),
            config=config,
        )

        second = predict_kalman(
            track_id="track",
            state=state,
            horizons_s=(0.1, 0.3),
            config=config,
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertFalse(
            first.truth_used
        )

        self.assertFalse(
            first.annotated_velocity_used
        )

        self.assertFalse(
            first.future_information_used
        )


if __name__ == "__main__":
    unittest.main()

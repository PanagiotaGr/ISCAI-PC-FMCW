import unittest

import numpy as np

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.baselines import (
    predict_imm,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.filters import (
    IMMConfig,
    IMMModeState,
    KalmanFrameContext,
    combine_imm_states,
    filter_associated_track_imm,
    interact_mode_states,
    predict_imm_mode_state,
    update_imm_mode_state,
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


def diagonal_covariance(
    value=1.0,
):
    return tuple(
        tuple(
            value if i == j else 0.0
            for j in range(10)
        )
        for i in range(10)
    )


def mode_state(
    mode,
    *,
    x=0.0,
    vx=0.0,
    ax=0.0,
    omega=0.0,
    timestamp=0.0,
):
    mean = [
        0.0
        for _ in range(10)
    ]

    mean[0] = x
    mean[3] = vx
    mean[6] = ax
    mean[9] = omega

    return IMMModeState(
        mode=mode,
        timestamp_s=timestamp,
        mean_10=tuple(mean),
        covariance_10x10=(
            diagonal_covariance()
        ),
    )


def context(
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


class TestIMM(unittest.TestCase):

    def test_transition_matrix_rows_sum_to_one(
        self,
    ):
        config = IMMConfig()

        matrix = np.asarray(
            config.transition_matrix
        )

        self.assertTrue(
            np.allclose(
                matrix.sum(axis=1),
                1.0,
            )
        )

    def test_interaction_probabilities_normalize(
        self,
    ):
        states = (
            mode_state("CV", x=0.0),
            mode_state("CA", x=1.0),
            mode_state("CTRV", x=2.0),
        )

        (
            _mixed,
            predicted,
            mixing,
        ) = interact_mode_states(
            states,
            (0.2, 0.3, 0.5),
            config=IMMConfig(),
        )

        self.assertAlmostEqual(
            sum(predicted),
            1.0,
        )

        matrix = np.asarray(
            mixing
        )

        for destination in range(3):
            self.assertAlmostEqual(
                matrix[
                    :,
                    destination
                ].sum(),
                1.0,
            )

    def test_mixing_contains_between_mode_spread(
        self,
    ):
        states = (
            mode_state("CV", x=0.0),
            mode_state("CA", x=10.0),
            mode_state("CTRV", x=20.0),
        )

        combined = combine_imm_states(
            states,
            (
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ),
        )

        self.assertAlmostEqual(
            combined.mean_10[0],
            10.0,
        )

        self.assertGreater(
            combined
            .covariance_10x10[0][0],
            1.0,
        )

    def test_mode_specific_dynamics_are_distinct(
        self,
    ):
        config = IMMConfig(
            cv_acceleration_spectral_density_m2_s3=0.0,
            ca_jerk_spectral_density_m2_s5=0.0,
            ctrv_linear_acceleration_spectral_density_m2_s3=0.0,
            ctrv_turn_rate_random_walk_rad2_s3=0.0,
            mode_switch_acceleration_variance_m2_s4=0.0,
            mode_switch_turn_rate_variance_rad2_s2=0.0,
        )

        cv = predict_imm_mode_state(
            mode_state(
                "CV",
                vx=1.0,
                ax=2.0,
            ),
            target_timestamp_s=1.0,
            config=config,
        )

        ca = predict_imm_mode_state(
            mode_state(
                "CA",
                vx=1.0,
                ax=2.0,
            ),
            target_timestamp_s=1.0,
            config=config,
        )

        ctrv = predict_imm_mode_state(
            mode_state(
                "CTRV",
                vx=1.0,
                omega=1.0,
            ),
            target_timestamp_s=1.0,
            config=config,
        )

        self.assertAlmostEqual(
            cv.mean_10[0],
            1.0,
        )

        self.assertAlmostEqual(
            ca.mean_10[0],
            2.0,
        )

        self.assertNotAlmostEqual(
            ctrv.mean_10[1],
            0.0,
        )

    def test_ctrv_zero_turn_matches_cv(
        self,
    ):
        config = IMMConfig(
            cv_acceleration_spectral_density_m2_s3=0.0,
            ca_jerk_spectral_density_m2_s5=0.0,
            ctrv_linear_acceleration_spectral_density_m2_s3=0.0,
            ctrv_turn_rate_random_walk_rad2_s3=0.0,
            mode_switch_acceleration_variance_m2_s4=0.0,
            mode_switch_turn_rate_variance_rad2_s2=0.0,
        )

        cv = predict_imm_mode_state(
            mode_state(
                "CV",
                x=5.0,
                vx=3.0,
            ),
            target_timestamp_s=0.5,
            config=config,
        )

        ctrv = predict_imm_mode_state(
            mode_state(
                "CTRV",
                x=5.0,
                vx=3.0,
                omega=0.0,
            ),
            target_timestamp_s=0.5,
            config=config,
        )

        self.assertAlmostEqual(
            cv.mean_10[0],
            ctrv.mean_10[0],
            places=10,
        )

        self.assertAlmostEqual(
            ctrv.mean_10[1],
            0.0,
            places=10,
        )

    def test_measurement_likelihood_prefers_closer_mode(
        self,
    ):
        times = (
            0.0,
            0.1,
        )

        ctx = context(
            (0.0, 0.0),
            times,
        )

        close = mode_state(
            "CV",
            x=10.0,
            timestamp=0.1,
        )

        far = mode_state(
            "CA",
            x=20.0,
            timestamp=0.1,
        )

        measurement = (
            snapshot_from_detection(
                make_detection(
                    key="m",
                    range_m=10.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    el_rad=0.0,
                )
            )
        )

        close_update = (
            update_imm_mode_state(
                close,
                measurement=measurement,
                frame_index=1,
                context=ctx,
                config=IMMConfig(),
            )
        )

        far_update = (
            update_imm_mode_state(
                far,
                measurement=measurement,
                frame_index=1,
                context=ctx,
                config=IMMConfig(),
            )
        )

        self.assertGreater(
            close_update.log_likelihood,
            far_update.log_likelihood,
        )

    def test_full_filter_constant_velocity_moving_sensor(
        self,
    ):
        times = (
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        )

        sensor_x = (
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
        )

        actor_x = (
            10.0,
            11.5,
            13.0,
            14.5,
            16.0,
        )

        frames = tuple(
            make_frame(
                timestamp_s=t,
                detections=(
                    make_detection(
                        key=f"k{i}",
                        range_m=(
                            actor_x[i]
                            -
                            sensor_x[i]
                        ),
                        vr_mps=5.0,
                        az_rad=0.0,
                        el_rad=0.0,
                    ),
                ),
            )
            for i, t in enumerate(
                times
            )
        )

        associated = (
            associate_estimated_gnn(
                AlgorithmObservationSequence(
                    scenario_id="imm-cv",
                    frames=frames,
                )
            )
        )

        self.assertEqual(
            len(associated.tracks),
            1,
        )

        result = (
            filter_associated_track_imm(
                associated.tracks[0],
                context=context(
                    sensor_x,
                    times,
                ),
                config=IMMConfig(
                    cv_acceleration_spectral_density_m2_s3=0.0,
                    ca_jerk_spectral_density_m2_s5=0.0,
                    ctrv_linear_acceleration_spectral_density_m2_s3=0.0,
                    ctrv_turn_rate_random_walk_rad2_s3=0.0,
                ),
            )
        )

        self.assertEqual(
            len(result.updates),
            2,
        )

        self.assertAlmostEqual(
            sum(
                result
                .current_mode_probabilities
            ),
            1.0,
            places=12,
        )

        self.assertAlmostEqual(
            result
            .combined_current_state
            .mean_10[0],
            16.0,
            places=7,
        )

        self.assertAlmostEqual(
            result
            .combined_current_state
            .mean_10[3],
            15.0,
            places=7,
        )

        for step in result.updates:
            for mode_update in (
                step.mode_updates
            ):
                self.assertTrue(
                    mode_update
                    .measurement_covariance_used
                )

    def test_mode_probabilities_change_with_measurement_likelihood(
        self,
    ):
        times = (
            0.0,
            0.1,
        )

        ctx = context(
            (0.0, 0.0),
            times,
        )

        states = (
            mode_state(
                "CV",
                x=10.0,
                timestamp=0.0,
            ),
            mode_state(
                "CA",
                x=20.0,
                timestamp=0.0,
            ),
            mode_state(
                "CTRV",
                x=30.0,
                timestamp=0.0,
            ),
        )

        from iscai_stage3.filters import (
            imm_update_step,
        )

        measurement = (
            snapshot_from_detection(
                make_detection(
                    key="m",
                    range_m=10.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    el_rad=0.0,
                )
            )
        )

        step = imm_update_step(
            states,
            (
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
            ),
            measurement=measurement,
            frame_index=1,
            context=ctx,
            target_timestamp_s=0.1,
            config=IMMConfig(
                transition_matrix=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                cv_acceleration_spectral_density_m2_s3=0.0,
                ca_jerk_spectral_density_m2_s5=0.0,
                ctrv_linear_acceleration_spectral_density_m2_s3=0.0,
                ctrv_turn_rate_random_walk_rad2_s3=0.0,
                mode_switch_acceleration_variance_m2_s4=0.0,
                mode_switch_turn_rate_variance_rad2_s2=0.0,
            ),
        )

        self.assertGreater(
            step
            .posterior_mode_probabilities[0],
            step
            .posterior_mode_probabilities[1],
        )

        self.assertGreater(
            step
            .posterior_mode_probabilities[0],
            step
            .posterior_mode_probabilities[2],
        )

    def test_open_loop_prediction_propagates_mode_probabilities(
        self,
    ):
        states = (
            mode_state(
                "CV",
                vx=1.0,
                timestamp=1.0,
            ),
            mode_state(
                "CA",
                vx=1.0,
                timestamp=1.0,
            ),
            mode_state(
                "CTRV",
                vx=1.0,
                timestamp=1.0,
            ),
        )

        prediction = predict_imm(
            track_id="track",
            states=states,
            mode_probabilities=(
                1.0,
                0.0,
                0.0,
            ),
            horizons_s=(
                0.1,
                0.3,
                0.5,
                1.0,
            ),
            config=IMMConfig(),
        )

        self.assertEqual(
            len(prediction.points),
            4,
        )

        self.assertLess(
            prediction.points[0]
            .mode_probabilities[0],
            1.0,
        )

        for point in prediction.points:
            self.assertAlmostEqual(
                sum(
                    point.mode_probabilities
                ),
                1.0,
                places=12,
            )

    def test_prediction_deterministic_and_causal(
        self,
    ):
        states = (
            mode_state(
                "CV",
                vx=2.0,
                timestamp=1.0,
            ),
            mode_state(
                "CA",
                vx=2.0,
                timestamp=1.0,
            ),
            mode_state(
                "CTRV",
                vx=2.0,
                timestamp=1.0,
            ),
        )

        config = IMMConfig()

        first = predict_imm(
            track_id="track",
            states=states,
            mode_probabilities=(
                0.4,
                0.3,
                0.3,
            ),
            horizons_s=(
                0.1,
                0.3,
            ),
            config=config,
        )

        second = predict_imm(
            track_id="track",
            states=states,
            mode_probabilities=(
                0.4,
                0.3,
                0.3,
            ),
            horizons_s=(
                0.1,
                0.3,
            ),
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
            first.annotated_heading_used
        )

        self.assertFalse(
            first.future_information_used
        )


if __name__ == "__main__":
    unittest.main()

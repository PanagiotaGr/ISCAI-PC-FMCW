import unittest

from types import SimpleNamespace

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
    cartesianize_associated_track,
    context_from_stage1_frames,
    spherical_position_Ht,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from iscai_stage3.state import (
    estimate_causal_cv_state,
)

from rigid_test_factory import (
    T_sensor_from_W,
    identity_transform,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


class TestHeadlampReconstruction(
    unittest.TestCase
):

    def test_spherical_to_cartesian(
        self,
    ):
        measurement = (
            snapshot_from_detection(
                make_detection(
                    key="x",
                    range_m=10.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    el_rad=0.0,
                )
            )
        )

        self.assertEqual(
            spherical_position_Ht(
                measurement
            ),
            (10.0, 0.0, 0.0),
        )

    def test_static_world_actor_zero_velocity_after_ego_compensation(
        self,
    ):
        # Ego/headlamp moves from x=0 to x=1.
        # Static world actor stays at x=10.
        #
        # Therefore range changes 10 -> 9,
        # but H0 position remains 10 -> 10.

        sequence = (
            AlgorithmObservationSequence(
                scenario_id="static-actor",
                frames=(
                    make_frame(
                        timestamp_s=0.0,
                        detections=(
                            make_detection(
                                key="a0",
                                range_m=10.0,
                                vr_mps=-1.0,
                                az_rad=0.0,
                            ),
                        ),
                    ),
                    make_frame(
                        timestamp_s=0.1,
                        detections=(
                            make_detection(
                                key="a1",
                                range_m=9.0,
                                vr_mps=-1.0,
                                az_rad=0.0,
                            ),
                        ),
                    ),
                ),
            )
        )

        associated = (
            associate_estimated_gnn(
                sequence
            )
        )

        self.assertEqual(
            len(associated.tracks),
            1,
        )

        context = FrameTransformContext(
            T_H0_from_W=(
                identity_transform()
            ),
            T_Ht_from_W_by_frame=(
                T_sensor_from_W(
                    sensor_origin_W=(
                        0.0, 0.0, 0.0
                    )
                ),
                T_sensor_from_W(
                    sensor_origin_W=(
                        1.0, 0.0, 0.0
                    )
                ),
            ),
        )

        observations = (
            cartesianize_associated_track(
                associated.tracks[0],
                context=context,
            )
        )

        self.assertAlmostEqual(
            observations[0]
            .position_H0_m[0],
            10.0,
        )

        self.assertAlmostEqual(
            observations[1]
            .position_H0_m[0],
            10.0,
        )

        state = estimate_causal_cv_state(
            observations
        )

        self.assertAlmostEqual(
            state.velocity_H0_mps[0],
            0.0,
            places=10,
        )

    def test_actor_motion_recovered_when_relative_range_is_constant(
        self,
    ):
        # Headlamp: x 0 -> 1
        # Actor:    x 10 -> 11
        #
        # Relative range stays 10.
        # Naive Ht differencing would return 0.
        # Correct H0 reconstruction returns 10 m/s.

        sequence = (
            AlgorithmObservationSequence(
                scenario_id="moving-actor",
                frames=(
                    make_frame(
                        timestamp_s=0.0,
                        detections=(
                            make_detection(
                                key="a0",
                                range_m=10.0,
                                vr_mps=0.0,
                                az_rad=0.0,
                            ),
                        ),
                    ),
                    make_frame(
                        timestamp_s=0.1,
                        detections=(
                            make_detection(
                                key="a1",
                                range_m=10.0,
                                vr_mps=0.0,
                                az_rad=0.0,
                            ),
                        ),
                    ),
                ),
            )
        )

        associated = (
            associate_estimated_gnn(
                sequence
            )
        )

        context = FrameTransformContext(
            T_H0_from_W=(
                identity_transform()
            ),
            T_Ht_from_W_by_frame=(
                T_sensor_from_W(
                    sensor_origin_W=(
                        0.0, 0.0, 0.0
                    )
                ),
                T_sensor_from_W(
                    sensor_origin_W=(
                        1.0, 0.0, 0.0
                    )
                ),
            ),
        )

        observations = (
            cartesianize_associated_track(
                associated.tracks[0],
                context=context,
            )
        )

        state = estimate_causal_cv_state(
            observations
        )

        self.assertAlmostEqual(
            observations[0]
            .position_H0_m[0],
            10.0,
        )

        self.assertAlmostEqual(
            observations[1]
            .position_H0_m[0],
            11.0,
        )

        self.assertAlmostEqual(
            state.velocity_H0_mps[0],
            10.0,
            places=10,
        )

    def test_covariance_remains_valid(
        self,
    ):
        sequence = (
            AlgorithmObservationSequence(
                scenario_id="covariance",
                frames=(
                    make_frame(
                        timestamp_s=0.0,
                        detections=(
                            make_detection(
                                key="a0",
                                range_m=20.0,
                                vr_mps=0.0,
                                az_rad=0.2,
                                el_rad=0.1,
                            ),
                        ),
                    ),
                ),
            )
        )

        associated = (
            associate_estimated_gnn(
                sequence
            )
        )

        context = FrameTransformContext(
            T_H0_from_W=(
                identity_transform()
            ),
            T_Ht_from_W_by_frame=(
                identity_transform(),
            ),
        )

        observation = (
            cartesianize_associated_track(
                associated.tracks[0],
                context=context,
            )[0]
        )

        P = (
            observation
            .position_covariance_H0_m2
        )

        for i in range(3):
            self.assertGreaterEqual(
                P[i][i],
                0.0,
            )

            for j in range(3):
                self.assertAlmostEqual(
                    P[i][j],
                    P[j][i],
                )

    def test_context_from_stage1_public_fields(
        self,
    ):
        T0 = identity_transform()
        T1 = T_sensor_from_W(
            sensor_origin_W=(
                1.0, 0.0, 0.0
            )
        )

        anchor = SimpleNamespace(
            T_H0_from_W=T0
        )

        dynamic = (
            SimpleNamespace(
                T_Ht_from_W=T0
            ),
            SimpleNamespace(
                T_Ht_from_W=T1
            ),
        )

        context = (
            context_from_stage1_frames(
                anchor_frames=anchor,
                dynamic_frames=dynamic,
            )
        )

        self.assertIs(
            context.T_H0_from_W,
            T0,
        )

        self.assertIs(
            context.transform_for_frame(1),
            T1,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from iscai_stage4.data import (
    FEATURE_DIM,
    FEATURE_NAMES,
    HISTORY_STEPS,
    MAP_CONTEXT_DIM,
    MAX_NEIGHBORS,
    CausalSceneInputs,
    CausalTrackHistory,
    FutureTrajectoryLabel,
    HistoryStep,
    SupervisedTrajectorySample,
    build_model_input_payload,
)


COV = (
    (1.0, 0.1, 0.0),
    (0.1, 2.0, 0.0),
    (0.0, 0.0, 0.5),
)


def make_step(
    frame_index,
    *,
    observed=True,
    velocity_valid=True,
    x=None,
    covariance=COV,
):
    if x is None:
        x = float(
            frame_index
        )

    if not observed:
        return HistoryStep(
            frame_index=frame_index,
            timestamp_s=(
                0.1
                *
                frame_index
            ),
            position_H0_m=(
                0.0,
                0.0,
                0.0,
            ),
            velocity_H0_mps=(
                0.0,
                0.0,
                0.0,
            ),
            measurement_covariance_H0_m2=(
                (
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                )
            ),
            observed=False,
            velocity_valid=False,
        )

    return HistoryStep(
        frame_index=frame_index,
        timestamp_s=(
            0.1
            *
            frame_index
        ),
        position_H0_m=(
            x,
            1.0,
            0.0,
        ),
        velocity_H0_mps=(
            1.0,
            0.0,
            0.0,
        ),
        measurement_covariance_H0_m2=(
            covariance
        ),
        observed=True,
        velocity_valid=(
            velocity_valid
        ),
    )


def make_history(
    prediction_id,
    *,
    offset=0.0,
):
    steps = tuple(
        make_step(
            frame_index,
            x=(
                float(
                    frame_index
                )
                +
                offset
            ),
            velocity_valid=(
                frame_index
                >
                0
            ),
        )
        for frame_index
        in range(
            HISTORY_STEPS
        )
    )

    return CausalTrackHistory(
        prediction_id=prediction_id,
        steps=steps,
        latest_observed_frame_index=10,
        latest_position_H0_m=(
            10.0
            +
            offset,
            1.0,
            0.0,
        ),
    )


class TestBlock42NeuralSamples(
    unittest.TestCase
):

    def test_feature_contract_exact(self):
        self.assertEqual(
            FEATURE_DIM,
            14,
        )

        self.assertEqual(
            FEATURE_NAMES[-2:],
            (
                "observed_mask",
                "velocity_valid_mask",
            ),
        )

    def test_history_is_exactly_11(self):
        history = make_history(
            "t0"
        )

        self.assertEqual(
            len(history.steps),
            11,
        )

    def test_feature_tensor_shape(self):
        rows = (
            make_history(
                "t0"
            )
            .feature_rows()
        )

        self.assertEqual(
            len(rows),
            11,
        )

        self.assertTrue(
            all(
                len(row)
                ==
                14
                for row in rows
            )
        )

    def test_missing_step_mask(self):
        step = make_step(
            3,
            observed=False,
        )

        row = (
            step.feature_row()
        )

        self.assertEqual(
            row[-2:],
            (
                0.0,
                0.0,
            ),
        )

    def test_covariance_changes_numeric_input(self):
        first = make_step(
            1
        ).feature_row()

        second = make_step(
            1,
            covariance=(
                (4.0, 0.1, 0.0),
                (0.1, 2.0, 0.0),
                (0.0, 0.0, 0.5),
            ),
        ).feature_row()

        self.assertNotEqual(
            first,
            second,
        )

    def test_first_velocity_can_be_invalid(self):
        step = make_step(
            0,
            velocity_valid=False,
        )

        self.assertTrue(
            step.observed
        )

        self.assertFalse(
            step.velocity_valid
        )

    def test_velocity_is_numeric_model_input(self):
        row = make_step(
            2
        ).feature_row()

        self.assertEqual(
            row[3:6],
            (
                1.0,
                0.0,
                0.0,
            ),
        )

    def test_neighbor_slots_are_fixed(self):
        scene = CausalSceneInputs(
            scenario_id="demo",
            timestamps_s=tuple(
                0.1 * i
                for i in range(11)
            ),
            histories=(
                make_history(
                    "target"
                ),
                make_history(
                    "n1",
                    offset=1.0,
                ),
            ),
        )

        payload = (
            build_model_input_payload(
                scene,
                prediction_id="target",
                map_context=tuple(
                    0.0
                    for _ in range(
                        MAP_CONTEXT_DIM
                    )
                ),
            )
        )

        self.assertEqual(
            len(
                payload
                .neighbor_histories
            ),
            MAX_NEIGHBORS,
        )

    def test_neighbor_mask(self):
        scene = CausalSceneInputs(
            scenario_id="demo",
            timestamps_s=tuple(
                0.1 * i
                for i in range(11)
            ),
            histories=(
                make_history(
                    "target"
                ),
                make_history(
                    "n1",
                    offset=1.0,
                ),
            ),
        )

        payload = (
            build_model_input_payload(
                scene,
                prediction_id="target",
                map_context=tuple(
                    0.0
                    for _ in range(
                        MAP_CONTEXT_DIM
                    )
                ),
            )
        )

        self.assertEqual(
            sum(
                payload.neighbor_mask
            ),
            1.0,
        )

    def test_model_payload_has_map_context(self):
        scene = CausalSceneInputs(
            scenario_id="demo",
            timestamps_s=tuple(
                0.1 * i
                for i in range(11)
            ),
            histories=(
                make_history(
                    "target"
                ),
            ),
        )

        payload = (
            build_model_input_payload(
                scene,
                prediction_id="target",
                map_context=tuple(
                    float(i)
                    for i in range(
                        MAP_CONTEXT_DIM
                    )
                ),
            )
        )

        self.assertEqual(
            len(
                payload.map_context
            ),
            10,
        )

    def test_numeric_payload_excludes_ids(self):
        scene = CausalSceneInputs(
            scenario_id="demo",
            timestamps_s=tuple(
                0.1 * i
                for i in range(11)
            ),
            histories=(
                make_history(
                    "123456789"
                ),
            ),
        )

        payload = (
            build_model_input_payload(
                scene,
                prediction_id="123456789",
                map_context=tuple(
                    0.0
                    for _ in range(
                        MAP_CONTEXT_DIM
                    )
                ),
            )
        )

        text = str(
            payload.numeric_dict()
        )

        self.assertNotIn(
            "123456789",
            text,
        )

    def test_future_label_has_four_horizons(self):
        label = FutureTrajectoryLabel(
            positions_H0_m=(
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
            ),
            valid_mask=(
                True,
                True,
                False,
                True,
            ),
        )

        self.assertEqual(
            len(
                label.positions_H0_m
            ),
            4,
        )

        self.assertEqual(
            len(
                label.valid_mask
            ),
            4,
        )

    def test_future_mask_does_not_enter_payload(self):
        scene = CausalSceneInputs(
            scenario_id="demo",
            timestamps_s=tuple(
                0.1 * i
                for i in range(11)
            ),
            histories=(
                make_history(
                    "target"
                ),
            ),
        )

        payload = (
            build_model_input_payload(
                scene,
                prediction_id="target",
                map_context=tuple(
                    0.0
                    for _ in range(
                        MAP_CONTEXT_DIM
                    )
                ),
            )
        )

        sample = (
            SupervisedTrajectorySample(
                sample_id="demo:target",
                prediction_id="target",
                model_input=payload,
                actor_class="TYPE_VEHICLE",
                truth_track_index=42,
                historical_match_distance_m=1.0,
                historical_match_frame_index=10,
                future_label=(
                    FutureTrajectoryLabel(
                        positions_H0_m=(
                            (1.0, 0.0, 0.0),
                            (2.0, 0.0, 0.0),
                            (3.0, 0.0, 0.0),
                            (4.0, 0.0, 0.0),
                        ),
                        valid_mask=(
                            True,
                            False,
                            False,
                            True,
                        ),
                    )
                ),
            )
        )

        numeric = str(
            sample
            .model_input
            .numeric_dict()
        )

        self.assertNotIn(
            "TYPE_VEHICLE",
            numeric,
        )

        self.assertNotIn(
            "42",
            numeric,
        )

        self.assertNotIn(
            "valid_mask",
            numeric,
        )

    def test_non_symmetric_covariance_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            make_step(
                1,
                covariance=(
                    (1.0, 0.5, 0.0),
                    (0.1, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )


if __name__ == "__main__":
    unittest.main()

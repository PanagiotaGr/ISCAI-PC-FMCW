import unittest
from dataclasses import fields

from iscai_stage1.actors.artifact import (
    build_causal_actor_artifact,
)
from iscai_stage1.actors.history import (
    RawObjectStateW,
    canonicalize_causal_actor_history,
)
from iscai_stage1.artifacts.hashing import causal_actor_sha256
from iscai_stage1.contracts.stage1a import ZERO_VEC3


def state(
    x: float,
    *,
    y: float = 0.0,
    z: float = 0.0,
    valid: bool = True,
) -> RawObjectStateW:
    return RawObjectStateW(
        center_W_m=(x, y, z),
        dimensions_lwh_m=(4.5, 1.8, 1.5),
        heading_rad=0.0,
        valid=valid,
    )


class TestCausalActorHistory(unittest.TestCase):
    def test_only_causal_prefix_is_persisted(self) -> None:
        history = canonicalize_causal_actor_history(
            timestamps_seconds=(0.0, 0.1, 0.2, 0.3, 0.4),
            states=(
                state(0.0),
                state(1.0),
                state(2.0),
                state(1000.0),
                state(2000.0),
            ),
            current_time_index=2,
        )

        self.assertEqual(len(history.timestamps_s), 3)
        self.assertEqual(len(history.position_W_m), 3)
        self.assertEqual(history.anchor_index, 2)
        self.assertEqual(history.position_W_m[-1], (2.0, 0.0, 0.0))

    def test_invalid_state_numeric_payload_is_zero_filled(self) -> None:
        invalid_state = RawObjectStateW(
            center_W_m=(999999.0, -999999.0, 12345.0),
            dimensions_lwh_m=(999.0, 999.0, 999.0),
            heading_rad=123.0,
            valid=False,
        )

        history = canonicalize_causal_actor_history(
            timestamps_seconds=(0.0,),
            states=(invalid_state,),
            current_time_index=0,
        )

        self.assertFalse(history.state_valid[0])
        self.assertEqual(history.position_W_m[0], ZERO_VEC3)
        self.assertEqual(history.dimensions_lwh_m[0], ZERO_VEC3)
        self.assertEqual(history.heading_rad[0], 0.0)

    def test_annotated_velocity_is_not_part_of_realistic_state_contract(self) -> None:
        field_names = {field.name for field in fields(RawObjectStateW)}

        self.assertNotIn("velocity_x", field_names)
        self.assertNotIn("velocity_y", field_names)
        self.assertNotIn("velocity_W_mps", field_names)

    def test_forecasting_candidate_can_have_invalid_anchor_velocity(self) -> None:
        artifact = build_causal_actor_artifact(
            scenario_id="scenario-test",
            track_id=42,
            object_class="TYPE_PEDESTRIAN",
            is_sdc=False,
            receiver_geometry_valid=False,
            timestamps_seconds=(0.0, 0.1, 0.2, 0.3),
            states=(
                state(0.0, valid=True),
                state(1.0, valid=True),
                state(50.0, valid=False),
                state(3.0, valid=True),
            ),
            current_time_index=3,
        )

        self.assertTrue(
            artifact.roles.is_forecasting_target_candidate
        )
        self.assertFalse(
            artifact.history.velocity_valid[
                artifact.history.anchor_index
            ]
        )

    def test_track_id_is_metadata_not_numeric_feature(self) -> None:
        artifact = build_causal_actor_artifact(
            scenario_id="scenario-test",
            track_id=123456,
            object_class="TYPE_VEHICLE",
            is_sdc=False,
            receiver_geometry_valid=True,
            timestamps_seconds=(0.0, 0.1),
            states=(state(0.0), state(1.0)),
            current_time_index=1,
        )

        features = artifact.realistic_numeric_features()

        self.assertEqual(artifact.metadata.track_id, 123456)
        self.assertNotIn("track_id", features)
        self.assertNotIn("scenario_id", features)
        self.assertNotIn("tracks_to_predict", features)
        self.assertNotIn("objects_of_interest", features)

    def test_stage1_artifact_is_explicitly_not_sensor_realistic(self) -> None:
        artifact = build_causal_actor_artifact(
            scenario_id="scenario-test",
            track_id=1,
            object_class="TYPE_VEHICLE",
            is_sdc=False,
            receiver_geometry_valid=True,
            timestamps_seconds=(0.0, 0.1),
            states=(state(0.0), state(1.0)),
            current_time_index=1,
        )

        self.assertEqual(
            artifact.semantics.artifact_semantics,
            "causal_womd_annotation_upstream",
        )
        self.assertFalse(artifact.semantics.sensor_realistic)


class TestFutureMutationCausality(unittest.TestCase):
    def _build(
        self,
        future_x_1: float,
        future_x_2: float,
    ):
        return build_causal_actor_artifact(
            scenario_id="future-mutation-test",
            track_id=99,
            object_class="TYPE_VEHICLE",
            is_sdc=False,
            receiver_geometry_valid=True,
            timestamps_seconds=(
                0.0,
                0.1,
                0.2,  # anchor
                0.3,
                0.4,
            ),
            states=(
                state(0.0),
                state(1.0),
                state(2.0),
                state(future_x_1),
                state(future_x_2),
            ),
            current_time_index=2,
        )

    def test_future_state_mutation_does_not_change_causal_hash(self) -> None:
        original = self._build(3.0, 4.0)
        mutated = self._build(999999.0, -999999.0)

        self.assertEqual(
            causal_actor_sha256(original),
            causal_actor_sha256(mutated),
        )

    def test_future_validity_mutation_does_not_change_causal_hash(self) -> None:
        original = build_causal_actor_artifact(
            scenario_id="future-validity-test",
            track_id=101,
            object_class="TYPE_CYCLIST",
            is_sdc=False,
            receiver_geometry_valid=False,
            timestamps_seconds=(0.0, 0.1, 0.2, 0.3),
            states=(
                state(0.0),
                state(1.0),
                state(2.0),
                state(3.0, valid=False),
            ),
            current_time_index=2,
        )

        mutated = build_causal_actor_artifact(
            scenario_id="future-validity-test",
            track_id=101,
            object_class="TYPE_CYCLIST",
            is_sdc=False,
            receiver_geometry_valid=False,
            timestamps_seconds=(0.0, 0.1, 0.2, 0.3),
            states=(
                state(0.0),
                state(1.0),
                state(2.0),
                state(999.0, valid=True),
            ),
            current_time_index=2,
        )

        self.assertEqual(
            causal_actor_sha256(original),
            causal_actor_sha256(mutated),
        )

    def test_future_can_be_absent_entirely(self) -> None:
        with_future = build_causal_actor_artifact(
            scenario_id="future-truncation-test",
            track_id=7,
            object_class="TYPE_PEDESTRIAN",
            is_sdc=False,
            receiver_geometry_valid=False,
            timestamps_seconds=(0.0, 0.1, 0.2, 0.3, 0.4),
            states=(
                state(0.0),
                state(1.0),
                state(2.0),
                state(3.0),
                state(4.0),
            ),
            current_time_index=2,
        )

        causal_only = build_causal_actor_artifact(
            scenario_id="future-truncation-test",
            track_id=7,
            object_class="TYPE_PEDESTRIAN",
            is_sdc=False,
            receiver_geometry_valid=False,
            timestamps_seconds=(0.0, 0.1, 0.2),
            states=(
                state(0.0),
                state(1.0),
                state(2.0),
            ),
            current_time_index=2,
        )

        self.assertEqual(
            causal_actor_sha256(with_future),
            causal_actor_sha256(causal_only),
        )


if __name__ == "__main__":
    unittest.main()
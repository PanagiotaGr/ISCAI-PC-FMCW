import unittest
from types import SimpleNamespace

from iscai_stage1.io.womd_adapter import (
    build_stage1a_scene_from_womd,
    stage1a_scene_causal_sha256,
)


class FakeEnumValue:
    def __init__(self, name):
        self.name = name


class FakeEnumType:
    def __init__(self):
        self.values_by_number = {
            0: FakeEnumValue("TYPE_UNSET"),
            1: FakeEnumValue("TYPE_VEHICLE"),
            2: FakeEnumValue("TYPE_PEDESTRIAN"),
            3: FakeEnumValue("TYPE_CYCLIST"),
            4: FakeEnumValue("TYPE_OTHER"),
        }


class FakeField:
    def __init__(self):
        self.enum_type = FakeEnumType()


class FakeDescriptor:
    def __init__(self):
        self.fields_by_name = {
            "object_type": FakeField(),
        }


class FakeState:
    def __init__(
        self,
        x,
        *,
        y=0.0,
        z=0.0,
        length=4.0,
        width=1.8,
        height=1.5,
        heading=0.0,
        valid=True,
    ):
        self.center_x = x
        self.center_y = y
        self.center_z = z
        self.length = length
        self.width = width
        self.height = height
        self.heading = heading
        self.valid = valid

        # These intentionally exist to mimic real WOMD.
        # Stage 1A must never read them.
        self.velocity_x = 999999.0
        self.velocity_y = -999999.0


class FakeTrack:
    DESCRIPTOR = FakeDescriptor()

    def __init__(self, track_id, object_type, states):
        self.id = track_id
        self.object_type = object_type
        self.states = list(states)


class FakeScenario:
    def __init__(self, *, future_mutation=False):
        self.scenario_id = "stage1a-fake-scene"
        self.current_time_index = 3
        self.sdc_track_index = 0

        self.timestamps_seconds = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]

        self.map_features = []
        self.dynamic_map_states = [
            SimpleNamespace(lane_states=[])
            for _ in self.timestamps_seconds
        ]

        future_x = 999999.0 if future_mutation else 4.0

        self.tracks = [
            FakeTrack(
                100,
                1,
                [
                    FakeState(0.0, length=4.8),
                    FakeState(0.2, length=4.8),
                    FakeState(0.4, length=4.8),
                    FakeState(0.6, length=4.8),
                    FakeState(future_x, length=4.8),
                    FakeState(future_x + 1.0, length=4.8),
                ],
            ),
            FakeTrack(
                200,
                1,
                [
                    FakeState(10.0),
                    FakeState(10.1),
                    FakeState(10.2),
                    FakeState(10.3),
                    FakeState(future_x),
                    FakeState(future_x),
                ],
            ),
            # Forecasting candidate whose adjacent anchor velocity is invalid.
            FakeTrack(
                300,
                2,
                [
                    FakeState(20.0),
                    FakeState(20.1),
                    FakeState(999.0, valid=False),
                    FakeState(20.3),
                    FakeState(future_x),
                    FakeState(future_x),
                ],
            ),
        ]

        # Leakage traps. Adapter must never inspect these.
        self.tracks_to_predict = ["SHOULD_NOT_BE_READ"]
        self.objects_of_interest = ["SHOULD_NOT_BE_READ"]


class TestWomdAdapter(unittest.TestCase):
    def test_build_scene(self):
        scene = build_stage1a_scene_from_womd(FakeScenario())

        self.assertEqual(scene.scenario_id, "stage1a-fake-scene")
        self.assertEqual(scene.anchor_index, 3)
        self.assertEqual(scene.sdc_track_index, 0)
        self.assertEqual(len(scene.actors), 3)

    def test_only_non_sdc_vehicle_is_receiver(self):
        scene = build_stage1a_scene_from_womd(FakeScenario())

        self.assertEqual(len(scene.receivers), 1)
        self.assertEqual(scene.receivers[0].track_id, 200)
        self.assertEqual(
            scene.receivers[0].geometry.receiver_geometry_mode,
            "centroid_baseline",
        )

    def test_receiver_point_is_actor_center_plus_zero_offset(self):
        scene = build_stage1a_scene_from_womd(FakeScenario())

        receiver = scene.receivers[0].geometry

        self.assertEqual(
            receiver.receiver_offset_mean_H0,
            (0.0, 0.0, 0.0),
        )
        self.assertTrue(receiver.receiver_geometry_valid)

    def test_forecast_candidate_need_not_have_anchor_velocity(self):
        scene = build_stage1a_scene_from_womd(FakeScenario())

        pedestrian = next(
            actor
            for actor in scene.actors
            if actor.metadata.track_id == 300
        )

        self.assertTrue(
            pedestrian.roles.is_forecasting_target_candidate
        )
        self.assertFalse(
            pedestrian.history.velocity_valid[
                pedestrian.history.anchor_index
            ]
        )

    def test_scene_future_mutation_does_not_change_hash(self):
        original = build_stage1a_scene_from_womd(
            FakeScenario(future_mutation=False)
        )

        mutated = build_stage1a_scene_from_womd(
            FakeScenario(future_mutation=True)
        )

        self.assertEqual(
            stage1a_scene_causal_sha256(original),
            stage1a_scene_causal_sha256(mutated),
        )

    def test_artifacts_are_upstream_not_sensor_realistic(self):
        scene = build_stage1a_scene_from_womd(FakeScenario())

        for actor in scene.actors:
            self.assertEqual(
                actor.semantics.artifact_semantics,
                "causal_womd_annotation_upstream",
            )
            self.assertFalse(actor.semantics.sensor_realistic)


if __name__ == "__main__":
    unittest.main()
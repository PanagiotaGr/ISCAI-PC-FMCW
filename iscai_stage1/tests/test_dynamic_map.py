import unittest

from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
)
from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_anchor_frames,
)
from iscai_stage1.maps.dynamic_map import (
    canonicalize_causal_dynamic_map,
)


class FakeEnumValue:
    def __init__(self, name):
        self.name = name


class FakeEnumType:
    def __init__(self):
        self.values_by_number = {
            0: FakeEnumValue("LANE_STATE_UNKNOWN"),
            6: FakeEnumValue("LANE_STATE_GO"),
        }


class FakeField:
    def __init__(self, enum_type=None):
        self.enum_type = enum_type


class FakeDescriptor:
    def __init__(self):
        self.fields_by_name = {
            "lane": FakeField(),
            "state": FakeField(FakeEnumType()),
            "stop_point": FakeField(),
        }


class FakePoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeLaneState:
    def __init__(
        self,
        lane,
        state,
        *,
        stop_point=None,
    ):
        self.DESCRIPTOR = FakeDescriptor()
        self.lane = lane
        self.state = state
        self.stop_point = (
            stop_point
            if stop_point is not None
            else FakePoint()
        )
        self._has_stop_point = stop_point is not None

    def HasField(self, name):
        if name != "stop_point":
            raise ValueError(name)

        return self._has_stop_point


class FakeDynamicState:
    def __init__(self, lane_states=()):
        self.lane_states = list(lane_states)


class FakeScenario:
    def __init__(self, dynamic_states):
        self.current_time_index = 2
        self.timestamps_seconds = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
        ]
        self.dynamic_map_states = dynamic_states


class TestDynamicMap(unittest.TestCase):
    def setUp(self):
        anchor = SdcStateW(
            center_w_m=(10.0, 20.0, 0.0),
            heading_rad=0.0,
            length_m=4.0,
            valid=True,
        )

        self.frames = build_anchor_frames(
            anchor,
            HeadlampSurrogateConfig(),
        )

    def test_only_causal_prefix_is_kept(self):
        scenario = FakeScenario([
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState([
                FakeLaneState(999, 6),
            ]),
            FakeDynamicState([
                FakeLaneState(1000, 6),
            ]),
        ])

        result = canonicalize_causal_dynamic_map(
            scenario,
            T_H0_from_W=self.frames.T_H0_from_W,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(
            [frame.time_index for frame in result],
            [0, 1, 2],
        )

    def test_lane_state_and_enum_are_preserved(self):
        scenario = FakeScenario([
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState([
                FakeLaneState(
                    12345,
                    6,
                    stop_point=FakePoint(
                        12.0,
                        21.0,
                        0.0,
                    ),
                ),
            ]),
        ])

        result = canonicalize_causal_dynamic_map(
            scenario,
            T_H0_from_W=self.frames.T_H0_from_W,
        )

        lane_state = result[2].lane_states[0]

        self.assertEqual(lane_state.lane_id, 12345)
        self.assertEqual(
            lane_state.state_name,
            "LANE_STATE_GO",
        )

    def test_stop_point_transforms_to_H0(self):
        scenario = FakeScenario([
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState([
                FakeLaneState(
                    1,
                    6,
                    stop_point=FakePoint(
                        12.0,
                        21.0,
                        0.0,
                    ),
                ),
            ]),
        ])

        result = canonicalize_causal_dynamic_map(
            scenario,
            T_H0_from_W=self.frames.T_H0_from_W,
        )

        signal = result[2].lane_states[0]

        self.assertEqual(
            signal.stop_point_W_m,
            (12.0, 21.0, 0.0),
        )

        # H0 = world (12,20,0) for this zero-yaw setup.
        self.assertEqual(
            signal.stop_point_H0_m,
            (0.0, 1.0, 0.0),
        )

    def test_missing_stop_point_is_explicit_none(self):
        scenario = FakeScenario([
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState([
                FakeLaneState(1, 0),
            ]),
        ])

        result = canonicalize_causal_dynamic_map(
            scenario,
            T_H0_from_W=self.frames.T_H0_from_W,
        )

        signal = result[2].lane_states[0]

        self.assertIsNone(signal.stop_point_W_m)
        self.assertIsNone(signal.stop_point_H0_m)

    def test_future_contents_do_not_affect_result(self):
        original = FakeScenario([
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState([
                FakeLaneState(10, 6),
            ]),
            FakeDynamicState([
                FakeLaneState(20, 6),
            ]),
            FakeDynamicState(),
        ])

        mutated = FakeScenario([
            FakeDynamicState(),
            FakeDynamicState(),
            FakeDynamicState([
                FakeLaneState(10, 6),
            ]),
            FakeDynamicState([
                FakeLaneState(999999, 0),
            ]),
            FakeDynamicState([
                FakeLaneState(888888, 6),
            ]),
        ])

        first = canonicalize_causal_dynamic_map(
            original,
            T_H0_from_W=self.frames.T_H0_from_W,
        )

        second = canonicalize_causal_dynamic_map(
            mutated,
            T_H0_from_W=self.frames.T_H0_from_W,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
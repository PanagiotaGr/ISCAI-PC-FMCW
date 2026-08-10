import unittest
from types import SimpleNamespace

from iscai_stage1.contracts.stage1a import HeadlampSurrogateConfig
from iscai_stage1.geometry.frames import (
    SdcStateW,
    build_anchor_frames,
)
from iscai_stage1.maps.static_map import (
    canonicalize_static_map,
)


class FakeEnumValue:
    def __init__(self, name):
        self.name = name


class FakeEnumType:
    def __init__(self, mapping):
        self.values_by_number = {
            key: FakeEnumValue(value)
            for key, value in mapping.items()
        }


class FakeField:
    def __init__(self, enum_type=None):
        self.enum_type = enum_type


class FakeNestedDescriptor:
    def __init__(self, enum_name=None):
        enum_type = None

        if enum_name is not None:
            enum_type = FakeEnumType({1: enum_name})

        self.fields_by_name = {
            "type": FakeField(enum_type),
        }


class FakeOneof:
    def __init__(self, name):
        self.name = name


class FakeFeatureDescriptor:
    oneofs = (FakeOneof("feature_data"),)


class FakePoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakePolyline:
    def __init__(self, points, *, enum_name="TYPE_TEST"):
        self.polyline = points
        self.type = 1
        self.DESCRIPTOR = FakeNestedDescriptor(enum_name)


class FakeLane(FakePolyline):
    def __init__(self, points):
        super().__init__(points, enum_name="TYPE_SURFACE_STREET")
        self.speed_limit_mph = 35.0
        self.interpolating = False
        self.entry_lanes = [10, 11]
        self.exit_lanes = [20]


class FakePolygon:
    DESCRIPTOR = FakeNestedDescriptor()

    def __init__(self, points):
        self.polygon = points


class FakeStopSign:
    DESCRIPTOR = FakeNestedDescriptor()

    def __init__(self, point):
        self.position = point


class FakeFeature:
    DESCRIPTOR = FakeFeatureDescriptor()

    def __init__(self, feature_id, kind, nested):
        self.id = feature_id
        self._kind = kind
        setattr(self, kind, nested)

    def WhichOneof(self, name):
        assert name == "feature_data"
        return self._kind


class TestStaticMap(unittest.TestCase):
    def setUp(self):
        anchor = SdcStateW(
            center_w_m=(10.0, 20.0, 0.0),
            heading_rad=0.0,
            length_m=4.0,
            valid=True,
        )

        frames = build_anchor_frames(
            anchor,
            HeadlampSurrogateConfig(),
        )

        self.T_H0_from_W = frames.T_H0_from_W

    def test_lane_geometry_and_semantics(self):
        scenario = SimpleNamespace(
            map_features=[
                FakeFeature(
                    100,
                    "lane",
                    FakeLane([
                        FakePoint(12.0, 20.0),
                        FakePoint(13.0, 20.0),
                    ]),
                ),
            ]
        )

        result = canonicalize_static_map(
            scenario,
            T_H0_from_W=self.T_H0_from_W,
        )

        self.assertEqual(len(result), 1)

        lane = result[0]

        self.assertEqual(lane.feature_id, 100)
        self.assertEqual(lane.kind, "lane")
        self.assertEqual(
            lane.semantic_type,
            "TYPE_SURFACE_STREET",
        )
        self.assertEqual(lane.speed_limit_mph, 35.0)
        self.assertFalse(lane.interpolating)
        self.assertEqual(lane.entry_lane_ids, (10, 11))
        self.assertEqual(lane.exit_lane_ids, (20,))

        # H0 is 2 m ahead of the SDC centre.
        self.assertEqual(
            lane.points_H0_m[0],
            (0.0, 0.0, 0.0),
        )

    def test_polygon_geometry(self):
        scenario = SimpleNamespace(
            map_features=[
                FakeFeature(
                    200,
                    "crosswalk",
                    FakePolygon([
                        FakePoint(12.0, 20.0),
                        FakePoint(13.0, 20.0),
                        FakePoint(13.0, 21.0),
                    ]),
                ),
            ]
        )

        result = canonicalize_static_map(
            scenario,
            T_H0_from_W=self.T_H0_from_W,
        )

        self.assertEqual(result[0].kind, "crosswalk")
        self.assertEqual(len(result[0].points_W_m), 3)
        self.assertEqual(len(result[0].points_H0_m), 3)

    def test_stop_sign_position(self):
        scenario = SimpleNamespace(
            map_features=[
                FakeFeature(
                    300,
                    "stop_sign",
                    FakeStopSign(
                        FakePoint(12.0, 21.0, 0.0)
                    ),
                ),
            ]
        )

        result = canonicalize_static_map(
            scenario,
            T_H0_from_W=self.T_H0_from_W,
        )

        self.assertEqual(result[0].kind, "stop_sign")
        self.assertEqual(len(result[0].points_W_m), 1)

    def test_world_to_H0_coordinates(self):
        scenario = SimpleNamespace(
            map_features=[
                FakeFeature(
                    400,
                    "road_edge",
                    FakePolyline([
                        FakePoint(12.0, 21.0),
                    ]),
                ),
            ]
        )

        result = canonicalize_static_map(
            scenario,
            T_H0_from_W=self.T_H0_from_W,
        )

        # SDC centre W=(10,20), H0 is at W=(12,20).
        self.assertEqual(
            result[0].points_H0_m[0],
            (0.0, 1.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
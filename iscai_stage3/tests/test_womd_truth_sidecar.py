from types import SimpleNamespace
import unittest

from iscai_stage3.evaluation import (
    anchor_class_inventory,
    build_womd_truth_sidecar,
)


class T:
    def apply_point(self, p):
        return (
            p[0] - 10.0,
            p[1] + 2.0,
            p[2] - 1.0,
        )


def state(x, *, valid=True):
    return SimpleNamespace(
        center_x=x,
        center_y=0.0,
        center_z=0.0,
        valid=valid,
    )


def track(track_id, object_type, base):
    return SimpleNamespace(
        id=track_id,
        object_type=object_type,
        states=[
            state(base + i)
            for i in range(21)
        ],
    )


def scene():
    return SimpleNamespace(
        scenario_id="scene",
        current_time_index=10,
        sdc_track_index=0,
        timestamps_seconds=[
            0.1 * i
            for i in range(21)
        ],
        tracks=[
            track("SDC", 1, 0.0),
            track("V1", 1, 20.0),
            track("P1", 2, 30.0),
        ],
    )


class TestWOMDTruthSidecar(unittest.TestCase):

    def test_H0_anchor(self):
        s = scene()

        truth = build_womd_truth_sidecar(
            s,
            T_H0_from_W=T(),
            eligible_track_indices=(1,),
        )

        self.assertEqual(
            truth.tracks[0].anchor_position_H0_m,
            (20.0, 2.0, -1.0),
        )

    def test_exact_horizons(self):
        s = scene()

        truth = build_womd_truth_sidecar(
            s,
            T_H0_from_W=T(),
            eligible_track_indices=(1,),
        )

        self.assertEqual(
            tuple(
                p.horizon_s
                for p in truth.tracks[0].points
            ),
            (0.1, 0.3, 0.5, 1.0),
        )

    def test_sdc_excluded(self):
        s = scene()

        truth = build_womd_truth_sidecar(
            s,
            T_H0_from_W=T(),
            eligible_track_indices=(0, 1),
        )

        self.assertEqual(
            [x.truth_id for x in truth.tracks],
            ["V1"],
        )

    def test_future_invalidity_only_removes_point(self):
        s = scene()
        s.tracks[1].states[13].valid = False

        truth = build_womd_truth_sidecar(
            s,
            T_H0_from_W=T(),
            eligible_track_indices=(1,),
        )

        self.assertEqual(
            len(truth.tracks),
            1,
        )

        horizons = {
            p.horizon_s
            for p in truth.tracks[0].points
        }

        self.assertNotIn(0.3, horizons)
        self.assertIn(1.0, horizons)

    def test_truth_id_and_class(self):
        s = scene()

        truth = build_womd_truth_sidecar(
            s,
            T_H0_from_W=T(),
            eligible_truth_ids=("P1",),
        )

        self.assertEqual(
            truth.tracks[0].truth_id,
            "P1",
        )

        self.assertEqual(
            truth.tracks[0].actor_class,
            "TYPE_PEDESTRIAN",
        )

    def test_class_inventory_future_invariant(self):
        s = scene()

        first = anchor_class_inventory(
            s,
            eligible_track_indices=(1, 2),
        )

        for i in range(11, 21):
            s.tracks[1].states[i].center_x = 9999.0
            s.tracks[2].states[i].valid = False

        second = anchor_class_inventory(
            s,
            eligible_track_indices=(1, 2),
        )

        self.assertEqual(first, second)

        self.assertEqual(
            first,
            {
                "TYPE_PEDESTRIAN": 1,
                "TYPE_VEHICLE": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from iscai_stage3.association import (
    AssociationConfig,
    associate_estimated_gnn,
)

from iscai_stage3.contracts import (
    AlgorithmObservationSequence,
)

from stage2_test_factory import (
    make_detection,
    make_frame,
)


def sequence(*frames):
    return AlgorithmObservationSequence(
        scenario_id="synthetic",
        frames=tuple(frames),
    )


class TestEstimatedAssociation(
    unittest.TestCase
):

    def test_two_tracks_survive_order_swap(
        self,
    ):
        result = associate_estimated_gnn(
            sequence(
                make_frame(
                    timestamp_s=0.0,
                    detections=(
                        make_detection(
                            key="a0",
                            range_m=10.0,
                            vr_mps=1.0,
                            az_rad=-0.2,
                        ),
                        make_detection(
                            key="b0",
                            range_m=30.0,
                            vr_mps=-1.0,
                            az_rad=0.3,
                        ),
                    ),
                ),
                make_frame(
                    timestamp_s=0.1,
                    detections=(
                        make_detection(
                            key="b1",
                            range_m=29.9,
                            vr_mps=-1.0,
                            az_rad=0.3,
                        ),
                        make_detection(
                            key="a1",
                            range_m=10.1,
                            vr_mps=1.0,
                            az_rad=-0.2,
                        ),
                    ),
                ),
            )
        )

        lengths = sorted(
            len(track.detections)
            for track in result.tracks
        )

        self.assertEqual(
            lengths,
            [2, 2],
        )

    def test_detection_key_not_identity(
        self,
    ):
        result = associate_estimated_gnn(
            sequence(
                make_frame(
                    timestamp_s=0.0,
                    detections=(
                        make_detection(
                            key="truth-looking-A",
                            range_m=15.0,
                            vr_mps=0.0,
                            az_rad=0.0,
                        ),
                    ),
                ),
                make_frame(
                    timestamp_s=0.1,
                    detections=(
                        make_detection(
                            key="completely-different",
                            range_m=15.0,
                            vr_mps=0.0,
                            az_rad=0.0,
                        ),
                    ),
                ),
            )
        )

        self.assertEqual(
            len(result.tracks),
            1,
        )

        self.assertEqual(
            len(
                result.tracks[0].detections
            ),
            2,
        )

    def test_false_alarm_creates_short_track(
        self,
    ):
        result = associate_estimated_gnn(
            sequence(
                make_frame(
                    timestamp_s=0.0,
                    detections=(
                        make_detection(
                            key="real0",
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
                            key="real1",
                            range_m=10.0,
                            vr_mps=0.0,
                            az_rad=0.0,
                        ),
                        make_detection(
                            key="fa",
                            range_m=80.0,
                            vr_mps=25.0,
                            az_rad=1.5,
                        ),
                    ),
                ),
            )
        )

        lengths = sorted(
            len(track.detections)
            for track in result.tracks
        )

        self.assertEqual(
            lengths,
            [1, 2],
        )

    def test_track_survives_allowed_gap(
        self,
    ):
        result = associate_estimated_gnn(
            sequence(
                make_frame(
                    timestamp_s=0.0,
                    detections=(
                        make_detection(
                            key="a",
                            range_m=20.0,
                            vr_mps=0.0,
                            az_rad=0.1,
                        ),
                    ),
                ),
                make_frame(
                    timestamp_s=0.1,
                    detections=(),
                ),
                make_frame(
                    timestamp_s=0.2,
                    detections=(
                        make_detection(
                            key="c",
                            range_m=20.0,
                            vr_mps=0.0,
                            az_rad=0.1,
                        ),
                    ),
                ),
            ),
            config=AssociationConfig(
                max_missed_frames=2
            ),
        )

        self.assertEqual(
            len(result.tracks),
            1,
        )

        self.assertEqual(
            len(
                result.tracks[0].detections
            ),
            2,
        )

    def test_expired_track_does_not_reconnect(
        self,
    ):
        result = associate_estimated_gnn(
            sequence(
                make_frame(
                    timestamp_s=0.0,
                    detections=(
                        make_detection(
                            key="a",
                            range_m=20.0,
                            vr_mps=0.0,
                            az_rad=0.1,
                        ),
                    ),
                ),
                make_frame(
                    timestamp_s=0.1,
                    detections=(),
                ),
                make_frame(
                    timestamp_s=0.2,
                    detections=(),
                ),
                make_frame(
                    timestamp_s=0.3,
                    detections=(
                        make_detection(
                            key="d",
                            range_m=20.0,
                            vr_mps=0.0,
                            az_rad=0.1,
                        ),
                    ),
                ),
            ),
            config=AssociationConfig(
                max_missed_frames=1
            ),
        )

        self.assertEqual(
            len(result.tracks),
            2,
        )

    def test_azimuth_wrap_associates(
        self,
    ):
        result = associate_estimated_gnn(
            sequence(
                make_frame(
                    timestamp_s=0.0,
                    detections=(
                        make_detection(
                            key="a",
                            range_m=10.0,
                            vr_mps=0.0,
                            az_rad=(
                                math.pi - 0.01
                            ),
                        ),
                    ),
                ),
                make_frame(
                    timestamp_s=0.1,
                    detections=(
                        make_detection(
                            key="b",
                            range_m=10.0,
                            vr_mps=0.0,
                            az_rad=(
                                -math.pi
                                + 0.01
                            ),
                        ),
                    ),
                ),
            )
        )

        self.assertEqual(
            len(result.tracks),
            1,
        )

    def test_result_is_deterministic(self):
        seq = sequence(
            make_frame(
                timestamp_s=0.0,
                detections=(
                    make_detection(
                        key="a",
                        range_m=10.0,
                        vr_mps=1.0,
                        az_rad=0.0,
                    ),
                    make_detection(
                        key="b",
                        range_m=30.0,
                        vr_mps=-1.0,
                        az_rad=0.5,
                    ),
                ),
            ),
            make_frame(
                timestamp_s=0.1,
                detections=(
                    make_detection(
                        key="x",
                        range_m=10.1,
                        vr_mps=1.0,
                        az_rad=0.0,
                    ),
                    make_detection(
                        key="y",
                        range_m=29.9,
                        vr_mps=-1.0,
                        az_rad=0.5,
                    ),
                ),
            ),
        )

        first = associate_estimated_gnn(
            seq
        )

        second = associate_estimated_gnn(
            seq
        )

        signature_first = [
            (
                track.track_id,
                [
                    item.detection.detection_key
                    for item in track.detections
                ],
            )
            for track in first.tracks
        ]

        signature_second = [
            (
                track.track_id,
                [
                    item.detection.detection_key
                    for item in track.detections
                ],
            )
            for track in second.tracks
        ]

        self.assertEqual(
            signature_first,
            signature_second,
        )

        self.assertFalse(
            first.truth_used
        )


if __name__ == "__main__":
    unittest.main()

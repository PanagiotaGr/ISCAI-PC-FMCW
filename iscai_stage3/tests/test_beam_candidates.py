import unittest

from iscai_stage3.beam.candidates import (
    build_beam_candidate,
)


class TestBeamCandidates(unittest.TestCase):


    def test_valid_candidate(self):

        c = build_beam_candidate(
            scenario_id="abc",
            track_index=1,
            object_class="TYPE_VEHICLE",
            timestamp_s=1.0,
            azimuth_rad=0.1,
            elevation_rad=0.0,
            azimuth_std_rad=0.01,
            elevation_std_rad=0.01,
            confidence=0.9,
        )

        self.assertEqual(
            c.track_index,
            1,
        )


    def test_negative_uncertainty(self):

        with self.assertRaises(
            ValueError
        ):
            build_beam_candidate(
                scenario_id="abc",
                track_index=1,
                object_class="TYPE_VEHICLE",
                timestamp_s=1.0,
                azimuth_rad=0.1,
                elevation_rad=0.0,
                azimuth_std_rad=-1.0,
                elevation_std_rad=0.01,
                confidence=0.9,
            )


    def test_confidence_range(self):

        with self.assertRaises(
            ValueError
        ):
            build_beam_candidate(
                scenario_id="abc",
                track_index=1,
                object_class="TYPE_VEHICLE",
                timestamp_s=1.0,
                azimuth_rad=0.1,
                elevation_rad=0.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.01,
                confidence=2.0,
            )

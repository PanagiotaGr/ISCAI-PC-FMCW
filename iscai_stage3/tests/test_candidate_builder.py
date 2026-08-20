import unittest

from iscai_stage3.pipeline.candidate_builder import (
    build_candidate_from_observation,
)


class FakeObservation:

    geometry_valid = True
    timestamp_s = 1.0
    azimuth_rad = 0.1
    elevation_rad = 0.2


class TestCandidateBuilder(unittest.TestCase):

    def test_build_candidate(self):

        c = build_candidate_from_observation(
            scenario_id="s1",
            track_index=1,
            object_class="TYPE_VEHICLE",
            observation=FakeObservation(),
            azimuth_std_rad=0.01,
            elevation_std_rad=0.02,
            confidence=0.9,
        )

        self.assertEqual(
            c.track_index,
            1,
        )

        self.assertEqual(
            c.object_class,
            "TYPE_VEHICLE",
        )

        self.assertAlmostEqual(
            c.azimuth_rad,
            0.1,
        )


if __name__ == "__main__":
    unittest.main()

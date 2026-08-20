import unittest

from iscai_stage3.pipeline.candidate_set import (
    build_candidate_set,
)


class Obs:

    geometry_valid = True
    timestamp_s = 1.0
    azimuth_rad = 0.1
    elevation_rad = 0.2



class Actor:

    track_index = 10
    object_class = "TYPE_VEHICLE"

    observations = (
        Obs(),
    )



class TestCandidateSet(unittest.TestCase):


    def test_actor_to_candidate(self):

        result = build_candidate_set(
            scenario_id="abc",
            actor_series=(
                Actor(),
            ),
            azimuth_std_rad=0.01,
            elevation_std_rad=0.01,
            confidence=0.9,
        )


        self.assertEqual(
            len(result.candidates),
            1,
        )


        self.assertEqual(
            result.candidates[0].track_index,
            10,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)

from iscai_stage3.pipeline.smoke import (
    evaluate_causal_observable,
)


IDENTITY3 = (
    (0.01, 0.0, 0.0),
    (0.0, 0.001, 0.0),
    (0.0, 0.0, 0.001),
)


class TestStage3CausalPipeline(unittest.TestCase):

    def test_stage2_observation_to_stage3(self):

        observation = IdealCausalObservable(
            time_index=0,
            timestamp_s=0.0,

            actor_position_Ht_m=(
                20.0,
                0.0,
                0.0,
            ),

            range_m=20.0,
            azimuth_rad=0.0,
            elevation_rad=0.0,

            radial_velocity_mps=0.0,

            geometry_valid=True,
            radial_velocity_valid=True,
        )


        target = evaluate_causal_observable(
            observation=observation,
            target_id=1,
            measurement_covariance=IDENTITY3,
            priority=1.0,
        )


        self.assertEqual(
            target.target_id,
            1,
        )


        self.assertGreater(
            target.beam_confidence,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from iscai_stage2.observations.womd_ideal_adapter import (
    ActorIdealObservationSeries,
)

from iscai_stage2.observations.ideal import (
    IdealCausalObservable,
)

from iscai_stage3.pipeline.smoke import (
    evaluate_observation_series,
)


IDENTITY3 = (
    (0.01,0.0,0.0),
    (0.0,0.001,0.0),
    (0.0,0.0,0.001),
)


class TestStage3ObservationSeries(unittest.TestCase):

    def test_series_evaluation(self):

        series = ActorIdealObservationSeries(
            scenario_id="test",
            track_index=1,
            track_id="1",
            object_class="TYPE_VEHICLE",
            is_sdc=False,

            observations=(
                IdealCausalObservable(
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
                ),
            ),
        )

        result = evaluate_observation_series(
            series=series,
            measurement_covariance=IDENTITY3,
        )

        self.assertEqual(
            len(result),
            1,
        )


if __name__ == "__main__":
    unittest.main()

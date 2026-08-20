import unittest

from iscai_stage3.trajectory.beam_prediction import (
    BeamState,
    constant_velocity_predict,
)



class TestBeamPrediction(unittest.TestCase):


    def test_constant_velocity(self):

        previous = BeamState(
            timestamp_s=0.0,
            azimuth_rad=0.0,
            elevation_rad=0.0,
            azimuth_variance=0.01,
            elevation_variance=0.01,
        )


        current = BeamState(
            timestamp_s=1.0,
            azimuth_rad=0.1,
            elevation_rad=0.0,
            azimuth_variance=0.01,
            elevation_variance=0.01,
        )


        result = constant_velocity_predict(
            previous=previous,
            current=current,
            timestamp_s=2.0,
        )


        self.assertAlmostEqual(
            result.azimuth_rad,
            0.2,
        )


    def test_uncertainty_preserved(self):

        previous = BeamState(
            0.0,
            0.0,
            0.0,
            0.01,
            0.01,
        )


        current = BeamState(
            1.0,
            0.1,
            0.0,
            0.04,
            0.09,
        )


        result = constant_velocity_predict(
            previous=previous,
            current=current,
            timestamp_s=2.0,
        )


        self.assertAlmostEqual(
            result.azimuth_std_rad,
            0.2,
        )

        self.assertAlmostEqual(
            result.elevation_std_rad,
            0.3,
        )


    def test_future_time_required(self):

        previous = BeamState(
            0.0,
            0.0,
            0.0,
            0.01,
            0.01,
        )

        current = BeamState(
            1.0,
            0.1,
            0.0,
            0.01,
            0.01,
        )


        with self.assertRaises(
            ValueError
        ):
            constant_velocity_predict(
                previous=previous,
                current=current,
                timestamp_s=0.5,
            )

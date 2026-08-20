import unittest


from iscai_stage3.trajectory.gaussian import (
    GaussianTrajectory,
    GaussianTrajectoryPoint,
    propagate_constant_velocity,
)



IDENTITY3 = (
    (1.0,0.0,0.0),
    (0.0,1.0,0.0),
    (0.0,0.0,1.0),
)



class TestTrajectoryGaussian(unittest.TestCase):


    def test_cv_prediction(self):

        out = propagate_constant_velocity(
            (1.0,2.0,3.0),
            (1.0,0.0,0.0),
            2.0,
        )


        self.assertEqual(
            out,
            (3.0,2.0,3.0),
        )



    def test_timestamp_order(self):

        traj = GaussianTrajectory(
            points=(
                GaussianTrajectoryPoint(
                    timestamp_s=0.0,
                    position_H0_m=(0,0,0),
                    covariance_H0_m2=IDENTITY3,
                ),
                GaussianTrajectoryPoint(
                    timestamp_s=0.1,
                    position_H0_m=(1,0,0),
                    covariance_H0_m2=IDENTITY3,
                ),
            )
        )

        traj.validate()



    def test_non_monotonic_rejected(self):

        traj = GaussianTrajectory(
            points=(
                GaussianTrajectoryPoint(
                    timestamp_s=1.0,
                    position_H0_m=(0,0,0),
                    covariance_H0_m2=IDENTITY3,
                ),
                GaussianTrajectoryPoint(
                    timestamp_s=0.5,
                    position_H0_m=(1,0,0),
                    covariance_H0_m2=IDENTITY3,
                ),
            )
        )


        with self.assertRaises(ValueError):
            traj.validate()



if __name__ == "__main__":
    unittest.main()

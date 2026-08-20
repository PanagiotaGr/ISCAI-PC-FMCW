import unittest

from iscai_stage3.contracts.uncertainty import (
    ObservationGaussian,
    GaussianTargetState,
)


class TestUncertaintyContracts(unittest.TestCase):

    def test_observation_contract(self):

        obs = ObservationGaussian(
            range_m=20.0,
            radial_velocity_mps=1.0,
            azimuth_rad=0.1,
            elevation_rad=0.0,
            covariance_4x4=(
                (1.0,0,0,0),
                (0,1.0,0,0),
                (0,0,1.0,0),
                (0,0,0,1.0),
            ),
        )

        obs.validate()

        self.assertEqual(
            obs.range_m,
            20.0,
        )


    def test_state_contract(self):

        state = GaussianTargetState(
            position_H0_m=(1.0,2.0,3.0),
            velocity_H0_mps=(0.1,0.2,0.3),
            covariance_6x6=tuple(
                tuple(
                    1.0 if i == j else 0.0
                    for j in range(6)
                )
                for i in range(6)
            ),
        )

        self.assertEqual(
            state.position_H0_m,
            (1.0,2.0,3.0),
        )


if __name__ == "__main__":
    unittest.main()

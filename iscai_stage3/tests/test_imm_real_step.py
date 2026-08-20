import unittest


from iscai_stage3.baselines.imm_core import (
    IMMProbabilities,
)

from iscai_stage3.baselines.imm_real_step import (
    imm_real_step,
)



class TestIMMRealStep(unittest.TestCase):


    def test_real_step_updates_probabilities(self):

        result = imm_real_step(
            probabilities=IMMProbabilities(
                cv=0.33,
                ca=0.33,
                ctrv=0.34,
            ),

            position=(
                10.0,
                0.0,
                0.0,
            ),

            velocity=(
                1.0,
                0.0,
                0.0,
            ),

            acceleration=(
                0.0,
                0.0,
                0.0,
            ),

            heading_rad=0.0,

            yaw_rate_radps=0.0,

            measurement=(
                11.0,
                1.0,
                0.0,
                0.0,
            ),

            covariance=(
                (1.0,0,0,0),
                (0,1.0,0,0),
                (0,0,1.0,0),
                (0,0,0,1.0),
            ),

            dt=1.0,
        )


        self.assertAlmostEqual(
            sum(
                result
                .probabilities
                .as_tuple()
            ),
            1.0,
        )


        self.assertTrue(
            result
            .cv_log_likelihood
            >
            float("-inf")
        )


if __name__ == "__main__":
    unittest.main()


class TestIMMStableLikelihoods(unittest.TestCase):

    def test_extreme_log_likelihoods_do_not_underflow(self):

        from iscai_stage3.baselines.imm_real_step import (
            _stable_relative_likelihoods,
        )

        values = _stable_relative_likelihoods(
            cv_log_likelihood=-100000.0,
            ca_log_likelihood=-100001.0,
            ctrv_log_likelihood=-100100.0,
        )

        self.assertGreater(
            max(values),
            0.0,
        )

        self.assertAlmostEqual(
            values[0],
            1.0,
        )

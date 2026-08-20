import math
import unittest

from iscai_stage2.pc_fmcw.sensing_snr import (
    FACTORIZED_MODEL_SEMANTICS,
    FactorizedSensingSnrConfig,
    FixedSensingSnrConfig,
    SensingSnrInputs,
    factorized_sensing_snr,
    fixed_sensing_snr,
)


class TestSensingSnr(unittest.TestCase):

    def test_fixed_snr(self):
        result = fixed_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=50.0
            ),
            config=FixedSensingSnrConfig(
                snr_db=20.0
            ),
        )

        self.assertAlmostEqual(
            result.snr_db,
            20.0,
        )

        self.assertAlmostEqual(
            result.snr_linear,
            100.0,
        )

    def test_reference_point_returns_reference_snr(self):
        result = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0,
                target_coefficient_amplitude=1.0,
                visibility_power_factor=1.0,
                extra_gain_db=0.0,
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=4.0,
            ),
        )

        self.assertAlmostEqual(
            result.snr_db,
            20.0,
            places=12,
        )

    def test_range_exponent_is_explicit(self):
        near = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=4.0,
            ),
        )

        far = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=40.0
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=4.0,
            ),
        )

        self.assertAlmostEqual(
            near.snr_db - far.snr_db,
            10.0 * math.log10(16.0),
            places=12,
        )

    def test_half_amplitude_costs_6db(self):
        reference = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0,
                target_coefficient_amplitude=1.0,
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=0.0,
            ),
        )

        half = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0,
                target_coefficient_amplitude=0.5,
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=0.0,
            ),
        )

        self.assertAlmostEqual(
            reference.snr_db
            - half.snr_db,
            10.0 * math.log10(4.0),
            places=12,
        )

    def test_half_visibility_costs_3db(self):
        full = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0,
                visibility_power_factor=1.0,
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=0.0,
            ),
        )

        half = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0,
                visibility_power_factor=0.5,
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=0.0,
            ),
        )

        self.assertAlmostEqual(
            full.snr_db
            - half.snr_db,
            10.0 * math.log10(2.0),
            places=12,
        )

    def test_extra_gain_is_additive_in_db(self):
        result = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0,
                extra_gain_db=-7.5,
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=0.0,
            ),
        )

        self.assertAlmostEqual(
            result.snr_db,
            12.5,
            places=12,
        )

    def test_factorized_model_is_not_part_a_claim(self):
        result = factorized_sensing_snr(
            inputs=SensingSnrInputs(
                range_m=20.0
            ),
            config=FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=2.0,
            ),
        )

        self.assertEqual(
            result.model_semantics,
            FACTORIZED_MODEL_SEMANTICS,
        )

    def test_invalid_range_rejected(self):
        with self.assertRaises(ValueError):
            factorized_sensing_snr(
                inputs=SensingSnrInputs(
                    range_m=0.0
                ),
                config=FactorizedSensingSnrConfig(
                    reference_range_m=20.0,
                    reference_snr_db=20.0,
                    range_power_exponent=4.0,
                ),
            )

    def test_zero_visibility_not_a_valid_detection(self):
        with self.assertRaises(ValueError):
            factorized_sensing_snr(
                inputs=SensingSnrInputs(
                    range_m=20.0,
                    visibility_power_factor=0.0,
                ),
                config=FactorizedSensingSnrConfig(
                    reference_range_m=20.0,
                    reference_snr_db=20.0,
                    range_power_exponent=4.0,
                ),
            )

    def test_negative_exponent_rejected(self):
        with self.assertRaises(ValueError):
            FactorizedSensingSnrConfig(
                reference_range_m=20.0,
                reference_snr_db=20.0,
                range_power_exponent=-1.0,
            )


if __name__ == "__main__":
    unittest.main()

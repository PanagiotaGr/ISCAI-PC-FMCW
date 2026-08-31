import inspect
import math
import unittest

from iscai_stage1.contracts.stage1a import (
    HeadlampSurrogateConfig,
)

from iscai_stage5.beam_directional_gain import (
    build_frozen_directional_codebook,
)

from iscai_stage5.optical_link import (
    DBPSK_BER_MODEL,
    EFFECTIVE_RATE_MODEL,
    HEADLAMP_FRAME_SEMANTICS,
    MANDATORY_ADDITIONAL_POINTING_LOSS,
    MANDATORY_ATMOSPHERIC_LOSS,
    MANDATORY_GEOMETRIC_LOSS,
    OPTICAL_GAIN_HPBW_RULE,
    OPTICAL_GAIN_MODEL,
    PARTA_RAW_DATA_RATE_BPS,
    PARTA_REFERENCE_NOISE_POWER_NORM,
    PARTA_REFERENCE_SIGNAL_POWER_NORM,
    PARTA_REFERENCE_SNR_DB,
    PARTA_REFERENCE_SNR_LINEAR,
    POWER_UNIT_SEMANTICS,
    constructed_optical_pointing_gain,
    dbpsk_ber_from_snr,
    effective_rate,
    evaluate_optical_link,
    normalized_received_power,
    optical_gain_for_cell,
    pointing_error,
    snr_from_normalized_power,
)


class TestBlock56HeadlampContract(
    unittest.TestCase
):

    def test_01_H0_default_is_front_face_midpoint(self):
        config = (
            HeadlampSurrogateConfig()
        )

        translation = (
            config
            .translation_in_sdc_m(
                4.0
            )
        )

        self.assertEqual(
            tuple(
                float(
                    value
                )
                for value in translation
            ),
            (
                2.0,
                0.0,
                0.0,
            ),
        )

    def test_02_no_second_extrinsic_semantics(self):
        self.assertIn(
            "no_second_extrinsic",
            HEADLAMP_FRAME_SEMANTICS,
        )


class TestBlock56PartAReference(
    unittest.TestCase
):

    def test_03_raw_rate_is_one_Gbps(self):
        self.assertEqual(
            PARTA_RAW_DATA_RATE_BPS,
            1e9,
        )

    def test_04_normalized_power_reference(self):
        self.assertEqual(
            PARTA_REFERENCE_SIGNAL_POWER_NORM,
            1.0,
        )

        self.assertEqual(
            PARTA_REFERENCE_NOISE_POWER_NORM,
            0.01,
        )

        self.assertEqual(
            PARTA_REFERENCE_SNR_LINEAR,
            100.0,
        )

        self.assertEqual(
            PARTA_REFERENCE_SNR_DB,
            20.0,
        )

    def test_05_power_units_not_claimed_as_watts(self):
        self.assertIn(
            "not_absolute_watts",
            POWER_UNIT_SEMANTICS,
        )


class TestBlock56PointingGain(
    unittest.TestCase
):

    def setUp(self):
        self.codebook = (
            build_frozen_directional_codebook(
                16
            )
        )

        self.cell = (
            self.codebook.cells[
                8
            ]
        )

        self.hpbw = (
            self.cell.upper_azimuth_rad
            -
            self.cell.lower_azimuth_rad
        )

    def test_06_constructed_model_explicit(self):
        self.assertIn(
            "constructed_Stage5",
            OPTICAL_GAIN_MODEL,
        )

        self.assertIn(
            "elevation_HPBW_equals",
            OPTICAL_GAIN_HPBW_RULE,
        )

    def test_07_center_gain_is_one(self):
        gain = (
            constructed_optical_pointing_gain(
                delta_azimuth_rad=0.0,
                delta_elevation_rad=0.0,
                azimuth_hpbw_rad=(
                    self.hpbw
                ),
            )
        )

        self.assertAlmostEqual(
            gain,
            1.0,
            places=15,
        )

    def test_08_half_HPBW_azimuth_is_half_gain(self):
        gain = (
            constructed_optical_pointing_gain(
                delta_azimuth_rad=(
                    0.5
                    *
                    self.hpbw
                ),

                delta_elevation_rad=0.0,

                azimuth_hpbw_rad=(
                    self.hpbw
                ),
            )
        )

        self.assertAlmostEqual(
            gain,
            0.5,
            places=14,
        )

    def test_09_elevation_error_reduces_gain(self):
        center = (
            constructed_optical_pointing_gain(
                delta_azimuth_rad=0.0,
                delta_elevation_rad=0.0,
                azimuth_hpbw_rad=(
                    self.hpbw
                ),
            )
        )

        offset = (
            constructed_optical_pointing_gain(
                delta_azimuth_rad=0.0,
                delta_elevation_rad=(
                    0.5
                    *
                    self.hpbw
                ),
                azimuth_hpbw_rad=(
                    self.hpbw
                ),
            )
        )

        self.assertLess(
            offset,
            center,
        )

        self.assertAlmostEqual(
            offset,
            0.5,
            places=14,
        )

    def test_10_azimuth_wrap(self):
        error = pointing_error(
            receiver_azimuth_rad=(
                math.radians(
                    179.0
                )
            ),

            receiver_elevation_rad=0.0,

            beam_center_azimuth_rad=(
                math.radians(
                    -179.0
                )
            ),
        )

        self.assertAlmostEqual(
            math.degrees(
                error.delta_azimuth_rad
            ),
            -2.0,
            places=12,
        )


class TestBlock56PowerSNRBER(
    unittest.TestCase
):

    def test_11_unity_losses_reproduce_reference_power(self):
        power = normalized_received_power(
            optical_gain=1.0
        )

        self.assertEqual(
            power,
            1.0,
        )

    def test_12_mandatory_losses_are_unity(self):
        self.assertEqual(
            MANDATORY_GEOMETRIC_LOSS,
            1.0,
        )

        self.assertEqual(
            MANDATORY_ATMOSPHERIC_LOSS,
            1.0,
        )

        self.assertEqual(
            MANDATORY_ADDITIONAL_POINTING_LOSS,
            1.0,
        )

    def test_13_reference_power_reproduces_20dB(self):
        linear, db = (
            snr_from_normalized_power(
                1.0
            )
        )

        self.assertEqual(
            linear,
            100.0,
        )

        self.assertAlmostEqual(
            db,
            20.0,
            places=14,
        )

    def test_14_DBPSK_zero_SNR_is_half(self):
        self.assertAlmostEqual(
            dbpsk_ber_from_snr(
                0.0
            ),
            0.5,
            places=15,
        )

    def test_15_DBPSK_BER_decreases_with_SNR(self):
        low = (
            dbpsk_ber_from_snr(
                1.0
            )
        )

        high = (
            dbpsk_ber_from_snr(
                10.0
            )
        )

        self.assertLess(
            high,
            low,
        )

    def test_16_20dB_analytic_BER_is_effectively_zero(self):
        ber = dbpsk_ber_from_snr(
            100.0
        )

        self.assertLess(
            ber,
            1e-40,
        )

        self.assertIn(
            "Stage5_analytic_DBPSK",
            DBPSK_BER_MODEL,
        )


class TestBlock56EffectiveRate(
    unittest.TestCase
):

    def test_17_no_overhead_no_BER_returns_raw_rate(self):
        result = effective_rate(
            ber=0.0,
            probing_beam_count=0,
            beam_probe_time_s=1e-5,
            frame_time_s=0.1,
        )

        self.assertEqual(
            result.effective_rate_bps,
            1e9,
        )

    def test_18_exact_probing_overhead_formula(self):
        result = effective_rate(
            ber=0.0,
            probing_beam_count=5,
            beam_probe_time_s=1e-5,
            frame_time_s=0.1,
        )

        self.assertAlmostEqual(
            result.raw_overhead_fraction,
            0.0005,
            places=15,
        )

        self.assertAlmostEqual(
            result.effective_rate_bps,
            999_500_000.0,
            places=6,
        )

        self.assertIn(
            "K_Tbeam_over_Tframe",
            EFFECTIVE_RATE_MODEL,
        )

    def test_19_overhead_above_frame_yields_zero_rate(self):
        result = effective_rate(
            ber=0.0,
            probing_beam_count=16,
            beam_probe_time_s=0.01,
            frame_time_s=0.1,
        )

        self.assertEqual(
            result.payload_fraction,
            0.0,
        )

        self.assertEqual(
            result.effective_rate_bps,
            0.0,
        )


class TestBlock56FullChain(
    unittest.TestCase
):

    def test_20_full_chain_exact_repeat_and_no_oracle_API(self):
        codebook = (
            build_frozen_directional_codebook(
                32
            )
        )

        cell = (
            codebook.cells[
                16
            ]
        )

        kwargs = {
            "receiver_azimuth_rad":
                cell.center_azimuth_rad,

            "receiver_elevation_rad":
                0.0,

            "cell":
                cell,

            "probing_beam_count":
                3,

            "beam_probe_time_s":
                1e-5,

            "frame_time_s":
                0.1,
        }

        first = evaluate_optical_link(
            **kwargs
        )

        second = evaluate_optical_link(
            **kwargs
        )

        self.assertEqual(
            first,
            second,
        )

        self.assertAlmostEqual(
            first.optical_gain,
            1.0,
        )

        self.assertAlmostEqual(
            first.snr_db,
            20.0,
        )

        signature = str(
            inspect.signature(
                evaluate_optical_link
            )
        ).lower()

        for token in (
            "future_truth",
            "ground_truth",
            "oracle",
            "tracks_to_predict",
            "objects_of_interest",
        ):
            self.assertNotIn(
                token,
                signature,
            )


if __name__ == "__main__":
    unittest.main()

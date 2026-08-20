import math
import unittest

from iscai_stage2.pc_fmcw.covariance import (
    measurement_covariance_from_part_a_crlb,
)
from iscai_stage2.pc_fmcw.crlb import (
    notebook_range_coupling_proxy,
    part_a_eq7_crlb,
)
from iscai_stage2.pc_fmcw.reference import (
    FROZEN_PART_A,
    PART_A_GIT_COMMIT,
)
from iscai_stage2.pc_fmcw.snr import (
    awgn_noise_power_for_snr,
    snr_db_to_linear,
)


class TestPartAPcfmcwPhysics(unittest.TestCase):

    def test_frozen_reference_parameters(self):
        self.assertEqual(
            PART_A_GIT_COMMIT,
            "44d62e3478e3818d1757b00971890f844cb032f7",
        )

        self.assertEqual(
            FROZEN_PART_A.carrier_frequency_hz,
            193.4e12,
        )

        self.assertEqual(
            FROZEN_PART_A.bandwidth_hz,
            10.0e9,
        )

        self.assertEqual(
            FROZEN_PART_A.chirp_duration_s,
            10.0e-6,
        )

        self.assertEqual(
            FROZEN_PART_A.n_fast,
            131_072,
        )

        self.assertEqual(
            FROZEN_PART_A.m_chirps,
            64,
        )

    def test_range_resolution_matches_notebook(self):
        self.assertAlmostEqual(
            FROZEN_PART_A.range_resolution_m,
            0.0149896229,
            places=12,
        )

    def test_snr_db_conversion(self):
        self.assertAlmostEqual(
            snr_db_to_linear(20.0),
            100.0,
        )

    def test_notebook_awgn_power_relation(self):
        self.assertAlmostEqual(
            awgn_noise_power_for_snr(
                signal_power=2.0,
                snr_db=20.0,
            ),
            0.02,
        )

    def test_eq7_crlb_at_20_db(self):
        result = part_a_eq7_crlb(
            20.0
        )

        # Exact values from the frozen notebook formulas.
        self.assertAlmostEqual(
            result.range_std_m,
            3.6523003208840815e-05,
            places=15,
        )

        self.assertAlmostEqual(
            result.radial_velocity_std_mps,
            2.9507338424929555e-06,
            places=15,
        )

    def test_crlb_improves_with_snr(self):
        low = part_a_eq7_crlb(
            0.0
        )

        high = part_a_eq7_crlb(
            20.0
        )

        self.assertLess(
            high.range_std_m,
            low.range_std_m,
        )

        self.assertLess(
            high.radial_velocity_std_mps,
            low.radial_velocity_std_mps,
        )

    def test_closely_spaced_range_proxy(self):
        result = (
            notebook_range_coupling_proxy(
                0.02
            )
        )

        self.assertAlmostEqual(
            result.coupling_coefficient,
            0.4106064586058357,
            places=12,
        )

        self.assertAlmostEqual(
            result.std_inflation_factor,
            1.09671650584257,
            places=12,
        )

    def test_well_separated_range_proxy(self):
        result = (
            notebook_range_coupling_proxy(
                16.0
            )
        )

        self.assertLess(
            result.coupling_coefficient,
            1e-100,
        )

        self.assertAlmostEqual(
            result.std_inflation_factor,
            1.0,
        )

    def test_zero_separation_proxy_is_singular(self):
        with self.assertRaises(ValueError):
            notebook_range_coupling_proxy(
                0.0
            )

    def test_covariance_mapping(self):
        result = (
            measurement_covariance_from_part_a_crlb(
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
            )
        )

        matrix = (
            result.covariance.matrix
        )

        self.assertAlmostEqual(
            matrix[0][0],
            result.crlb.range_variance_m2,
        )

        self.assertAlmostEqual(
            matrix[1][1],
            result.crlb
            .radial_velocity_variance_m2ps2,
        )

        self.assertAlmostEqual(
            matrix[2][2],
            0.01 ** 2,
        )

        self.assertAlmostEqual(
            matrix[3][3],
            0.02 ** 2,
        )

        self.assertEqual(
            matrix[0][1],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

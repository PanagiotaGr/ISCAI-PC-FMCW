import math
import unittest

from iscai_stage2.validation.measurement_mc import (
    assert_monte_carlo_consistent,
    run_measurement_monte_carlo,
)


class TestMeasurementMonteCarlo(unittest.TestCase):

    def test_mc_matches_configured_covariance(self):
        report = (
            run_measurement_monte_carlo(
                samples=20_000,
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
                seed_base=12345,
            )
        )

        assert_monte_carlo_consistent(
            report
        )

    def test_crlb_range_std_improves_with_snr(self):
        low = (
            run_measurement_monte_carlo(
                samples=2_000,
                sensing_snr_db=0.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
                seed_base=1,
            )
        )

        high = (
            run_measurement_monte_carlo(
                samples=2_000,
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
                seed_base=1,
            )
        )

        self.assertGreater(
            low.range_stats.configured_std,
            high.range_stats.configured_std,
        )

        self.assertGreater(
            low.radial_velocity_stats
            .configured_std,

            high.radial_velocity_stats
            .configured_std,
        )

    def test_20db_to_0db_crlb_std_ratio_is_ten(self):
        low = (
            run_measurement_monte_carlo(
                samples=100,
                sensing_snr_db=0.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
            )
        )

        high = (
            run_measurement_monte_carlo(
                samples=100,
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
            )
        )

        self.assertAlmostEqual(
            (
                low.range_stats.configured_std
                /
                high.range_stats.configured_std
            ),
            10.0,
            places=12,
        )

        self.assertAlmostEqual(
            (
                low.radial_velocity_stats
                .configured_std
                /
                high.radial_velocity_stats
                .configured_std
            ),
            10.0,
            places=12,
        )

    def test_angular_std_does_not_implicitly_follow_snr(self):
        low = (
            run_measurement_monte_carlo(
                samples=100,
                sensing_snr_db=0.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
            )
        )

        high = (
            run_measurement_monte_carlo(
                samples=100,
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
            )
        )

        self.assertEqual(
            low.azimuth_stats.configured_std,
            high.azimuth_stats.configured_std,
        )

        self.assertEqual(
            low.elevation_stats.configured_std,
            high.elevation_stats.configured_std,
        )

    def test_report_is_reproducible(self):
        a = (
            run_measurement_monte_carlo(
                samples=500,
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
                seed_base=55,
            )
        )

        b = (
            run_measurement_monte_carlo(
                samples=500,
                sensing_snr_db=20.0,
                azimuth_std_rad=0.01,
                elevation_std_rad=0.02,
                seed_base=55,
            )
        )

        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()

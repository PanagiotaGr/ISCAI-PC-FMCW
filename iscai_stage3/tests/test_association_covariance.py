import unittest

from iscai_stage3.association import (
    AssociationConfig,
    association_cost,
)

from iscai_stage3.association.gating import (
    PredictedMeasurement,
)

from iscai_stage3.observations import (
    snapshot_from_detection,
)

from stage2_test_factory import (
    make_detection,
)


class TestAssociationCovariance(
    unittest.TestCase
):

    def test_larger_covariance_relaxes_gate(
        self,
    ):
        previous = snapshot_from_detection(
            make_detection(
                key="p",
                range_m=10.0,
                vr_mps=0.0,
                az_rad=0.0,
                variance_scale=1.0,
            )
        )

        low_uncertainty = (
            snapshot_from_detection(
                make_detection(
                    key="l",
                    range_m=13.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    variance_scale=0.01,
                )
            )
        )

        high_uncertainty = (
            snapshot_from_detection(
                make_detection(
                    key="h",
                    range_m=13.0,
                    vr_mps=0.0,
                    az_rad=0.0,
                    variance_scale=100.0,
                )
            )
        )

        predicted = PredictedMeasurement(
            range_m=10.0,
            radial_velocity_mps=0.0,
            azimuth_rad=0.0,
            elevation_rad=0.0,
        )

        config = AssociationConfig(
            base_range_gate_m=0.1,
            max_range_rate_mps=0.0,
            base_radial_velocity_gate_mps=1.0,
            max_radial_acceleration_mps2=0.0,
            base_azimuth_gate_rad=0.1,
            max_azimuth_rate_radps=0.0,
            base_elevation_gate_rad=0.1,
            max_elevation_rate_radps=0.0,
            covariance_sigma_multiplier=2.0,
        )

        low = association_cost(
            predicted=predicted,
            previous=previous,
            current=low_uncertainty,
            dt_s=0.1,
            config=config,
        )

        high = association_cost(
            predicted=predicted,
            previous=previous,
            current=high_uncertainty,
            dt_s=0.1,
            config=config,
        )

        self.assertGreater(
            low.normalized_range,
            high.normalized_range,
        )

        self.assertFalse(
            low.feasible
        )

        self.assertTrue(
            high.feasible
        )


if __name__ == "__main__":
    unittest.main()

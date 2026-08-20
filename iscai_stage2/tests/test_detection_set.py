import math
import unittest

from iscai_stage2.observations.contracts import (
    MeasurementCovariance,
    PcfmcwLikeObservation,
)
from iscai_stage2.observations.detection import (
    DetectionFilteredRecord,
)
from iscai_stage2.observations.detection_set import (
    FALSE_ALARM,
    TRUE_DETECTION,
    AssociatedFilteredRecord,
    FalseAlarmConfig,
    build_unlabeled_detection_frame,
    deterministic_poisson,
)


COV = MeasurementCovariance(
    matrix=(
        (0.01, 0.0, 0.0, 0.0),
        (0.0, 0.04, 0.0, 0.0),
        (0.0, 0.0, 0.001, 0.0),
        (0.0, 0.0, 0.0, 0.002),
    )
)


FA_CONFIG = FalseAlarmConfig(
    seed=20260810,

    mean_false_alarms_per_frame=3.0,

    range_min_m=1.0,
    range_max_m=100.0,

    radial_velocity_min_mps=-30.0,
    radial_velocity_max_mps=30.0,

    azimuth_min_rad=-math.pi,
    azimuth_max_rad=math.pi,

    elevation_min_rad=-0.3,
    elevation_max_rad=0.3,

    range_std_m=1.0,
    radial_velocity_std_mps=2.0,
    azimuth_std_rad=0.05,
    elevation_std_rad=0.05,
)


def valid_record(
    *,
    track_id: str = "track-1",
    time_index: int = 5,
    timestamp_s: float = 0.5,
):
    measurement = PcfmcwLikeObservation(
        scenario_id="scenario",
        track_id=track_id,
        object_class="TYPE_VEHICLE",

        time_index=time_index,
        timestamp_s=timestamp_s,

        range_m=25.0,
        radial_velocity_mps=-2.0,
        azimuth_rad=0.1,
        elevation_rad=0.02,

        covariance=COV,
        measurement_valid=True,
    )

    return DetectionFilteredRecord(
        time_index=time_index,
        timestamp_s=timestamp_s,

        upstream_measurement_valid=True,
        measurement_valid=True,

        sensing_snr_db=20.0,
        detection_probability=0.9,
        detection_uniform=0.1,

        measurement=measurement,
        invalid_reason=None,

        missed_detection_applied=False,
    )


def missed_record():
    return DetectionFilteredRecord(
        time_index=5,
        timestamp_s=0.5,

        upstream_measurement_valid=True,
        measurement_valid=False,

        sensing_snr_db=0.0,
        detection_probability=0.1,
        detection_uniform=0.9,

        measurement=None,
        invalid_reason="missed_detection",

        missed_detection_applied=True,
    )


class TestDetectionSet(unittest.TestCase):

    def test_algorithm_detection_has_no_track_or_class(self):
        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,

            associated_records=(
                AssociatedFilteredRecord(
                    track_id="secret-track",
                    object_class="TYPE_VEHICLE",
                    record=valid_record(
                        track_id="secret-track"
                    ),
                ),
            ),

            false_alarm_config=None,
        )

        detection = (
            bundle.frame.detections[0]
        )

        self.assertFalse(
            hasattr(
                detection,
                "track_id",
            )
        )

        self.assertFalse(
            hasattr(
                detection,
                "object_class",
            )
        )

        self.assertFalse(
            hasattr(
                detection,
                "source_type",
            )
        )

    def test_truth_sidecar_keeps_oracle_id(self):
        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,

            associated_records=(
                AssociatedFilteredRecord(
                    track_id="oracle-id",
                    object_class="TYPE_VEHICLE",
                    record=valid_record(
                        track_id="oracle-id"
                    ),
                ),
            ),
        )

        truth = bundle.truth.entries[0]

        self.assertEqual(
            truth.source_type,
            TRUE_DETECTION,
        )

        self.assertEqual(
            truth.source_track_id,
            "oracle-id",
        )

    def test_true_covariance_is_preserved(self):
        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,

            associated_records=(
                AssociatedFilteredRecord(
                    track_id="t",
                    object_class="TYPE_VEHICLE",
                    record=valid_record(
                        track_id="t"
                    ),
                ),
            ),
        )

        self.assertEqual(
            bundle.frame
            .detections[0]
            .covariance,
            COV,
        )

    def test_missed_detection_is_not_emitted(self):
        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,

            associated_records=(
                AssociatedFilteredRecord(
                    track_id="t",
                    object_class="TYPE_VEHICLE",
                    record=missed_record(),
                ),
            ),
        )

        self.assertEqual(
            len(
                bundle.frame.detections
            ),
            0,
        )

        self.assertEqual(
            len(
                bundle.truth.entries
            ),
            0,
        )

    def test_zero_false_alarm_mean_generates_none(self):
        config = FalseAlarmConfig(
            seed=1,

            mean_false_alarms_per_frame=0.0,

            range_min_m=1.0,
            range_max_m=10.0,

            radial_velocity_min_mps=-1.0,
            radial_velocity_max_mps=1.0,

            azimuth_min_rad=-1.0,
            azimuth_max_rad=1.0,

            elevation_min_rad=-0.2,
            elevation_max_rad=0.2,

            range_std_m=1.0,
            radial_velocity_std_mps=1.0,
            azimuth_std_rad=0.1,
            elevation_std_rad=0.1,
        )

        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,
            associated_records=(),
            false_alarm_config=config,
        )

        self.assertEqual(
            len(
                bundle.frame.detections
            ),
            0,
        )

    def test_poisson_count_is_deterministic(self):
        a = deterministic_poisson(
            mean=3.0,
            key="same-key",
        )

        b = deterministic_poisson(
            mean=3.0,
            key="same-key",
        )

        self.assertEqual(a, b)

    def test_false_alarm_generation_is_deterministic(self):
        a = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,
            associated_records=(),
            false_alarm_config=FA_CONFIG,
        )

        b = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,
            associated_records=(),
            false_alarm_config=FA_CONFIG,
        )

        self.assertEqual(a, b)

    def test_false_alarms_are_inside_configured_volume(self):
        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,
            associated_records=(),
            false_alarm_config=FA_CONFIG,
        )

        for detection in (
            bundle.frame.detections
        ):
            self.assertGreaterEqual(
                detection.range_m,
                FA_CONFIG.range_min_m,
            )

            self.assertLess(
                detection.range_m,
                FA_CONFIG.range_max_m,
            )

            self.assertGreaterEqual(
                detection.radial_velocity_mps,
                FA_CONFIG.radial_velocity_min_mps,
            )

            self.assertLess(
                detection.radial_velocity_mps,
                FA_CONFIG.radial_velocity_max_mps,
            )

            self.assertGreaterEqual(
                detection.azimuth_rad,
                FA_CONFIG.azimuth_min_rad,
            )

            self.assertLess(
                detection.azimuth_rad,
                FA_CONFIG.azimuth_max_rad,
            )

            self.assertGreaterEqual(
                detection.elevation_rad,
                FA_CONFIG.elevation_min_rad,
            )

            self.assertLess(
                detection.elevation_rad,
                FA_CONFIG.elevation_max_rad,
            )

    def test_false_alarm_truth_has_no_actor_identity(self):
        bundle = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,
            associated_records=(),
            false_alarm_config=FA_CONFIG,
        )

        for entry in (
            bundle.truth.entries
        ):
            self.assertEqual(
                entry.source_type,
                FALSE_ALARM,
            )

            self.assertIsNone(
                entry.source_track_id
            )

            self.assertIsNone(
                entry.source_object_class
            )

    def test_detection_key_is_frame_local(self):
        a = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=5,
            timestamp_s=0.5,

            associated_records=(
                AssociatedFilteredRecord(
                    track_id="same-track",
                    object_class="TYPE_VEHICLE",
                    record=valid_record(
                        track_id="same-track",
                        time_index=5,
                        timestamp_s=0.5,
                    ),
                ),
            ),
        )

        b = build_unlabeled_detection_frame(
            scenario_id="scenario",
            time_index=6,
            timestamp_s=0.6,

            associated_records=(
                AssociatedFilteredRecord(
                    track_id="same-track",
                    object_class="TYPE_VEHICLE",
                    record=valid_record(
                        track_id="same-track",
                        time_index=6,
                        timestamp_s=0.6,
                    ),
                ),
            ),
        )

        self.assertNotEqual(
            a.frame.detections[
                0
            ].detection_key,

            b.frame.detections[
                0
            ].detection_key,
        )

    def test_timestamp_mismatch_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            build_unlabeled_detection_frame(
                scenario_id="scenario",
                time_index=5,
                timestamp_s=0.7,

                associated_records=(
                    AssociatedFilteredRecord(
                        track_id="t",
                        object_class="TYPE_VEHICLE",
                        record=valid_record(),
                    ),
                ),
            )

    def test_invalid_angle_volume_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            FalseAlarmConfig(
                seed=1,

                mean_false_alarms_per_frame=1.0,

                range_min_m=1.0,
                range_max_m=10.0,

                radial_velocity_min_mps=-5.0,
                radial_velocity_max_mps=5.0,

                azimuth_min_rad=-4.0,
                azimuth_max_rad=4.0,

                elevation_min_rad=-0.2,
                elevation_max_rad=0.2,

                range_std_m=1.0,
                radial_velocity_std_mps=1.0,
                azimuth_std_rad=0.1,
                elevation_std_rad=0.1,
            )


if __name__ == "__main__":
    unittest.main()

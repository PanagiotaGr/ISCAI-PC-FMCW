import math
import unittest
from dataclasses import replace

from iscai_stage3.pipeline.real_beam_series import (
    build_causal_beam_series,
)


class DummyObservation:
    def __init__(
        self,
        *,
        timestamp_s,
        position_Ht_m,
        geometry_valid=True,
    ):
        self.timestamp_s = timestamp_s
        self.position_Ht_m = position_Ht_m
        self.geometry_valid = geometry_valid


class TestRealActorBeamSeries(unittest.TestCase):

    def test_causal_series_length(self):
        observations = tuple(
            DummyObservation(
                timestamp_s=0.1 * i,
                position_Ht_m=(10.0 + i, 1.0, 0.5),
            )
            for i in range(11)
        )

        result = build_causal_beam_series(observations)

        self.assertEqual(len(result), 11)

    def test_timestamps_preserved(self):
        observations = tuple(
            DummyObservation(
                timestamp_s=0.1 * i,
                position_Ht_m=(20.0, float(i), 0.0),
            )
            for i in range(11)
        )

        result = build_causal_beam_series(observations)

        self.assertEqual(
            tuple(x.timestamp_s for x in result),
            tuple(0.1 * i for i in range(11)),
        )

    def test_beam_coordinates_are_finite(self):
        observations = (
            DummyObservation(
                timestamp_s=0.0,
                position_Ht_m=(10.0, 2.0, 1.0),
            ),
        )

        result = build_causal_beam_series(observations)

        self.assertEqual(len(result), 1)

        beam = result[0]

        self.assertTrue(math.isfinite(beam.range_m))
        self.assertTrue(math.isfinite(beam.azimuth_rad))
        self.assertTrue(math.isfinite(beam.elevation_rad))

    def test_invalid_geometry_is_retained_explicitly(self):
        observations = (
            DummyObservation(
                timestamp_s=0.0,
                position_Ht_m=(0.0, 0.0, 0.0),
                geometry_valid=False,
            ),
        )

        result = build_causal_beam_series(observations)

        self.assertEqual(len(result), 1)
        self.assertFalse(result[0].geometry_valid)
        self.assertIsNone(result[0].range_m)
        self.assertIsNone(result[0].azimuth_rad)
        self.assertIsNone(result[0].elevation_rad)

    def test_non_monotonic_timestamps_rejected(self):
        observations = (
            DummyObservation(
                timestamp_s=0.2,
                position_Ht_m=(10.0, 0.0, 0.0),
            ),
            DummyObservation(
                timestamp_s=0.1,
                position_Ht_m=(11.0, 0.0, 0.0),
            ),
        )

        with self.assertRaises(ValueError):
            build_causal_beam_series(observations)

    def test_future_values_are_not_an_input(self):
        causal = tuple(
            DummyObservation(
                timestamp_s=0.1 * i,
                position_Ht_m=(15.0 + i, 1.0, 0.0),
            )
            for i in range(11)
        )

        first = build_causal_beam_series(causal)

        # There is deliberately no future trajectory argument in the API.
        second = build_causal_beam_series(causal)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

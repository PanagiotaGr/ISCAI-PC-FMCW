import unittest

from iscai_stage4.data.training_index import (
    TrainingScenarioRecord,
    deterministic_split,
    manifest_bytes,
    partition_counts,
    scenario_selection_hash,
)


def record(
    scenario_id: str,
    offset: int,
) -> TrainingScenarioRecord:
    return TrainingScenarioRecord(
        scenario_id=scenario_id,
        motion_shard="demo.tfrecord",
        compact_record_offset=offset,
        payload_length=100,
        selection_hash=(
            scenario_selection_hash(
                scenario_id
            )
        ),
        anchor_class_counts=(
            (
                "TYPE_VEHICLE",
                1,
            ),
        ),
    )


class TestBlock41Split(unittest.TestCase):

    def test_hash_stable(self):
        self.assertEqual(
            scenario_selection_hash(
                "abc"
            ),
            scenario_selection_hash(
                "abc"
            ),
        )

    def test_hash_changes(self):
        self.assertNotEqual(
            scenario_selection_hash(
                "abc"
            ),
            scenario_selection_hash(
                "abd"
            ),
        )

    def test_partition_counts_small(self):
        self.assertEqual(
            partition_counts(10),
            (
                8,
                1,
                1,
            ),
        )

    def test_partition_counts_real_total(self):
        self.assertEqual(
            partition_counts(
                72085
            ),
            (
                57668,
                7208,
                7209,
            ),
        )

    def test_split_exact_counts(self):
        records = tuple(
            record(
                f"s{i:03d}",
                i * 100,
            )
            for i in range(20)
        )

        split = (
            deterministic_split(
                records
            )
        )

        self.assertEqual(
            len(split["fit"]),
            16,
        )

        self.assertEqual(
            len(
                split[
                    "development"
                ]
            ),
            2,
        )

        self.assertEqual(
            len(
                split[
                    "calibration"
                ]
            ),
            2,
        )

    def test_split_input_order_invariant(self):
        first = tuple(
            record(
                f"s{i:03d}",
                i * 100,
            )
            for i in range(20)
        )

        second = tuple(
            reversed(first)
        )

        a = deterministic_split(
            first
        )

        b = deterministic_split(
            second
        )

        for key in (
            "fit",
            "development",
            "calibration",
        ):
            self.assertEqual(
                tuple(
                    x.scenario_id
                    for x in a[key]
                ),
                tuple(
                    x.scenario_id
                    for x in b[key]
                ),
            )

    def test_split_no_overlap(self):
        records = tuple(
            record(
                f"s{i:03d}",
                i * 100,
            )
            for i in range(50)
        )

        split = deterministic_split(
            records
        )

        fit = {
            x.scenario_id
            for x in split["fit"]
        }

        dev = {
            x.scenario_id
            for x in split[
                "development"
            ]
        }

        cal = {
            x.scenario_id
            for x in split[
                "calibration"
            ]
        }

        self.assertFalse(
            fit & dev
        )
        self.assertFalse(
            fit & cal
        )
        self.assertFalse(
            dev & cal
        )

        self.assertEqual(
            len(
                fit | dev | cal
            ),
            50,
        )

    def test_duplicate_ids_rejected(self):
        duplicate = (
            record(
                "same",
                0,
            ),
            record(
                "same",
                100,
            ),
        )

        with self.assertRaises(
            ValueError
        ):
            deterministic_split(
                duplicate
            )

    def test_negative_offset_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            TrainingScenarioRecord(
                scenario_id="a",
                motion_shard="x",
                compact_record_offset=-1,
                payload_length=100,
                selection_hash=(
                    scenario_selection_hash(
                        "a"
                    )
                ),
                anchor_class_counts=(),
            )

    def test_empty_scenario_id_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            scenario_selection_hash(
                ""
            )

    def test_manifest_deterministic(self):
        items = tuple(
            sorted(
                (
                    record(
                        "x",
                        0,
                    ),
                    record(
                        "y",
                        100,
                    ),
                ),
                key=lambda x:
                    x.selection_hash,
            )
        )

        a = manifest_bytes(
            items,
            partition="fit",
        )

        b = manifest_bytes(
            items,
            partition="fit",
        )

        self.assertEqual(
            a,
            b,
        )

        self.assertIn(
            b'"partition":"fit"',
            a,
        )


if __name__ == "__main__":
    unittest.main()

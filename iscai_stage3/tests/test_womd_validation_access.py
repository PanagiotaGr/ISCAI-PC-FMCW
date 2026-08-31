from hashlib import sha256
from pathlib import Path
import struct
import tempfile
import unittest

from waymo_open_dataset.protos import scenario_pb2

from iscai_stage3.validation import (
    ValidationManifestRow,
    build_compact_motion_offset_index,
    read_motion_scenario,
    resolve_motion_shard,
)


def row(
    sid,
    *,
    source="/x/validation.tfrecord-00000-of-00150",
    offset=0,
    payload=10,
):
    return ValidationManifestRow.from_mapping({
        "scenario_id": sid,
        "split": "validation",
        "source_shard": source,
        "record_offset": offset,
        "payload_length": payload,
        "motion_record_bytes": payload + 16,
        "lidar_uri": f"gs://x/{sid}.tfrecord",
        "lidar_bytes": 100,
        "pair_bytes": payload + 116,
        "selection_hash": sha256(
            sid.encode("utf-8")
        ).hexdigest(),
        "selection_policy": "all_exact_matching_validation",
    })


class TestWOMDValidationAccess(unittest.TestCase):

    def test_manifest_contract(self):
        item = row("abc")
        self.assertEqual(item.scenario_id, "abc")
        self.assertEqual(
            item.motion_record_bytes,
            item.payload_length + 16,
        )

    def test_compact_offsets(self):
        a1 = row("A1", offset=100, payload=10)
        a2 = row("A2", offset=9000, payload=20)

        offsets = build_compact_motion_offset_index(
            (a2, a1)
        )

        self.assertEqual(offsets["A1"], 0)
        self.assertEqual(offsets["A2"], 26)

    def test_offsets_reset_per_source(self):
        a = row(
            "A",
            source="/x/validation.tfrecord-00000-of-00150",
            offset=100,
        )
        b = row(
            "B",
            source="/x/validation.tfrecord-00001-of-00150",
            offset=500,
        )

        offsets = build_compact_motion_offset_index(
            (a, b)
        )

        self.assertEqual(offsets["A"], 0)
        self.assertEqual(offsets["B"], 0)

    def test_resolver_uses_paired_from_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "validation" / "motion"
            motion.mkdir(parents=True)

            expected = (
                motion
                / "paired-from-validation.tfrecord-00000-of-00150"
            )
            expected.write_bytes(b"x")

            self.assertEqual(
                resolve_motion_shard(
                    row("A"),
                    paired_root=root,
                ),
                expected,
            )

    def test_random_access_uses_compact_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = root / "validation" / "motion"
            motion.mkdir(parents=True)

            shard = (
                motion
                / "paired-from-validation.tfrecord-00000-of-00150"
            )

            scenarios = []

            for sid in ("A", "B"):
                scenario = scenario_pb2.Scenario()
                scenario.scenario_id = sid
                payload = scenario.SerializeToString()

                record = (
                    struct.pack("<Q", len(payload))
                    + b"\0\0\0\0"
                    + payload
                    + b"\0\0\0\0"
                )

                scenarios.append(
                    (sid, payload, record)
                )

            shard.write_bytes(
                scenarios[0][2]
                +
                scenarios[1][2]
            )

            a = row(
                "A",
                offset=1000,
                payload=len(scenarios[0][1]),
            )
            b = row(
                "B",
                offset=999999,
                payload=len(scenarios[1][1]),
            )

            offsets = build_compact_motion_offset_index(
                (a, b)
            )

            loaded = read_motion_scenario(
                b,
                paired_root=root,
                compact_record_offset=offsets["B"],
            )

            self.assertEqual(
                loaded.scenario_id,
                "B",
            )


if __name__ == "__main__":
    unittest.main()

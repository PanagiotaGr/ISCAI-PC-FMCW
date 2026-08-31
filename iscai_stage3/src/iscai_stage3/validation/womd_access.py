from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Iterable

from waymo_open_dataset.protos import scenario_pb2


@dataclass(frozen=True)
class ValidationManifestRow:
    scenario_id: str
    split: str
    source_shard: str

    record_offset: int
    payload_length: int
    motion_record_bytes: int

    lidar_uri: str
    lidar_bytes: int
    pair_bytes: int

    selection_hash: str
    selection_policy: str

    @classmethod
    def from_mapping(
        cls,
        row: dict,
    ) -> "ValidationManifestRow":
        required = (
            "scenario_id",
            "split",
            "source_shard",
            "record_offset",
            "payload_length",
            "motion_record_bytes",
            "lidar_uri",
            "lidar_bytes",
            "pair_bytes",
            "selection_hash",
            "selection_policy",
        )

        missing = [
            key
            for key in required
            if key not in row
        ]

        if missing:
            raise ValueError(
                f"Missing manifest fields: {missing}"
            )

        result = cls(
            scenario_id=str(row["scenario_id"]),
            split=str(row["split"]),
            source_shard=str(row["source_shard"]),
            record_offset=int(row["record_offset"]),
            payload_length=int(row["payload_length"]),
            motion_record_bytes=int(
                row["motion_record_bytes"]
            ),
            lidar_uri=str(row["lidar_uri"]),
            lidar_bytes=int(row["lidar_bytes"]),
            pair_bytes=int(row["pair_bytes"]),
            selection_hash=str(
                row["selection_hash"]
            ),
            selection_policy=str(
                row["selection_policy"]
            ),
        )

        if not result.scenario_id:
            raise ValueError(
                "Empty scenario_id."
            )

        if result.split != "validation":
            raise ValueError(
                "Only validation rows are allowed."
            )

        if result.record_offset < 0:
            raise ValueError(
                "Negative record_offset."
            )

        if result.payload_length <= 0:
            raise ValueError(
                "Invalid payload_length."
            )

        if (
            result.motion_record_bytes
            !=
            result.payload_length + 16
        ):
            raise ValueError(
                "TFRecord framing mismatch."
            )

        expected_hash = sha256(
            result.scenario_id.encode("utf-8")
        ).hexdigest()

        if (
            result.selection_hash
            !=
            expected_hash
        ):
            raise ValueError(
                "selection_hash is not "
                "SHA256(scenario_id)."
            )

        return result


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_validation_manifest(
    path: Path,
) -> tuple[
    ValidationManifestRow,
    ...
]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON on validation "
                    f"manifest line {line_number}."
                ) from exc

            rows.append(
                ValidationManifestRow
                .from_mapping(raw)
            )

    ids = [
        row.scenario_id
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate validation scenario_id."
        )

    return tuple(rows)


def deterministic_manifest_order(
    rows: Iterable[
        ValidationManifestRow
    ],
) -> tuple[
    ValidationManifestRow,
    ...
]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.selection_hash,
                row.scenario_id,
            ),
        )
    )



def build_compact_motion_offset_index(
    rows: Iterable[
        ValidationManifestRow
    ],
) -> dict[str, int]:
    """
    Reconstruct canonical offsets inside the
    compact paired-from-* motion shards.

    The frozen manifest record_offset is an
    offset in the ORIGINAL WOMD source shard.

    During paired-corpus packaging, selected
    rows from each source shard were sorted by
    original record_offset and written
    consecutively to the canonical compact
    paired-from-* shard.

    Therefore canonical compact offsets are
    cumulative selected TFRecord byte lengths,
    not the original source offsets.
    """

    grouped: dict[
        str,
        list[ValidationManifestRow],
    ] = {}

    for row in rows:
        grouped.setdefault(
            row.source_shard,
            [],
        ).append(row)

    offsets: dict[
        str,
        int,
    ] = {}

    for source_shard, source_rows in (
        grouped.items()
    ):
        ordered = sorted(
            source_rows,
            key=lambda row: (
                row.record_offset,
                row.scenario_id,
            ),
        )

        compact_offset = 0
        previous_original_end = None

        for row in ordered:
            if (
                previous_original_end
                is not None
                and
                row.record_offset
                <
                previous_original_end
            ):
                raise ValueError(
                    "Overlapping original "
                    "TFRecord intervals in "
                    f"{source_shard}."
                )

            if row.scenario_id in offsets:
                raise ValueError(
                    "Duplicate scenario_id while "
                    "building compact offsets: "
                    f"{row.scenario_id}"
                )

            offsets[
                row.scenario_id
            ] = compact_offset

            compact_offset += (
                row.motion_record_bytes
            )

            previous_original_end = (
                row.record_offset
                +
                row.motion_record_bytes
            )

    return offsets


def resolve_motion_shard(
    row: ValidationManifestRow,
    *,
    paired_root: Path,
) -> Path:
    """
    Resolve the canonical compact motion shard.

    Original:
        validation.tfrecord-XXXXX-of-00150

    Canonical:
        paired-from-validation.tfrecord-
        XXXXX-of-00150
    """

    motion_root = (
        paired_root
        /
        row.split
        /
        "motion"
    )

    if not motion_root.is_dir():
        raise FileNotFoundError(
            f"Missing motion root: {motion_root}"
        )

    source_name = Path(
        row.source_shard
    ).name

    canonical_name = (
        "paired-from-"
        +
        source_name
    )

    path = (
        motion_root
        /
        canonical_name
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Canonical compact motion shard "
            "not found: "
            f"{path}"
        )

    return path

def expected_lidar_path(
    row: ValidationManifestRow,
    *,
    paired_root: Path,
) -> Path:
    return (
        paired_root
        /
        row.split
        /
        "lidar"
        /
        row.scenario_id[:2]
        /
        f"{row.scenario_id}.tfrecord"
    )



def read_motion_scenario(
    row: ValidationManifestRow,
    *,
    paired_root: Path,
    compact_record_offset: int,
):
    """
    Random-access one Scenario from the
    canonical compact paired motion shard.

    IMPORTANT:
    compact_record_offset is reconstructed
    from the frozen manifest ordering.

    row.record_offset remains provenance for
    the deleted/original WOMD source shard and
    is deliberately NOT used against the
    canonical paired-from-* shard.
    """

    offset = int(
        compact_record_offset
    )

    if offset < 0:
        raise ValueError(
            "Negative compact record offset."
        )

    shard = resolve_motion_shard(
        row,
        paired_root=paired_root,
    )

    with shard.open("rb") as stream:
        stream.seek(offset)

        header = stream.read(12)

        if len(header) != 12:
            raise IOError(
                "Short canonical TFRecord "
                "header read."
            )

        payload_length = struct.unpack(
            "<Q",
            header[:8],
        )[0]

        if (
            payload_length
            !=
            row.payload_length
        ):
            raise ValueError(
                "Canonical payload length "
                "differs from frozen manifest: "
                f"{payload_length} != "
                f"{row.payload_length}"
            )

        payload = stream.read(
            payload_length
        )

        if (
            len(payload)
            !=
            payload_length
        ):
            raise IOError(
                "Short canonical motion "
                "payload read."
            )

        data_crc = stream.read(4)

        if len(data_crc) != 4:
            raise IOError(
                "Short canonical TFRecord "
                "data CRC read."
            )

    scenario = scenario_pb2.Scenario()

    scenario.ParseFromString(
        payload
    )

    if (
        scenario.scenario_id
        !=
        row.scenario_id
    ):
        raise ValueError(
            "Canonical Scenario ID mismatch: "
            f"{scenario.scenario_id!r} != "
            f"{row.scenario_id!r}"
        )

    return scenario


from __future__ import annotations

import json
import struct
from pathlib import Path


ROOT = Path("/waymo/data/v1_3_0_scenario")
OUT = Path(
    "/waymo/iscai_stage0/reports/stage0/"
    "split_scenario_counts.json"
)


def count_records(path: Path) -> int:
    count = 0

    with path.open("rb") as f:
        while True:
            header = f.read(12)

            if not header:
                break

            if len(header) != 12:
                raise RuntimeError(
                    f"Truncated TFRecord header: {path}"
                )

            length = struct.unpack("<Q", header[:8])[0]

            # Skip payload + data CRC.
            f.seek(length + 4, 1)
            count += 1

    return count


def main() -> None:
    result = {}

    for split in ("training", "validation", "testing"):
        files = sorted(
            (ROOT / split).glob("*.tfrecord-*")
        )

        total = 0

        for i, path in enumerate(files, 1):
            total += count_records(path)

            if i % 100 == 0:
                print(
                    split,
                    f"{i}/{len(files)} shards",
                    "scenarios:",
                    total,
                )

        result[split] = {
            "shards": len(files),
            "scenarios": total,
        }

    OUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))
    print("Report:", OUT)


if __name__ == "__main__":
    main()
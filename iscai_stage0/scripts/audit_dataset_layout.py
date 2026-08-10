#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iscai_stage0.common import load_config, write_json  # noqa: E402

TFRECORD_SUFFIXES = (".tfrecord", ".tfrecords")
ARCHIVE_SUFFIXES = (".tar", ".gz", ".zip", ".tgz")


def classify_file(path: Path) -> str:
    name = path.name.lower()
    context = "/".join(part.lower() for part in path.parts)
    if any(token in context for token in ("train", "training")):
        split = "train"
    elif any(token in context for token in ("valid", "validation")):
        split = "validation"
    elif "test" in context:
        split = "test"
    else:
        split = "unknown"

    if name.endswith(TFRECORD_SUFFIXES) or ".tfrecord-" in name:
        kind = "tfrecord"
    elif name.endswith(ARCHIVE_SUFFIXES):
        kind = "archive"
    else:
        kind = path.suffix.lower().lstrip(".") or "no_suffix"
    return f"{split}:{kind}"


def relative_tree(root: Path, max_depth: int) -> list[str]:
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        marker = "d" if path.is_dir() else "f"
        rows.append(f"{marker} {rel}")
    return rows


def build_report(root: Path, max_depth: int, sample_n: int) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")

    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    total_files = 0
    total_bytes = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total_files += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
        group = classify_file(path)
        counts[group] += 1
        if len(examples[group]) < sample_n:
            examples[group].append(str(path.relative_to(root)))

    top_dirs = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            top_dirs.append(path.name)

    return {
        "dataset_root": str(root),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "top_level_directories": top_dirs,
        "file_groups": dict(sorted(counts.items())),
        "sample_files": {k: v for k, v in sorted(examples.items())},
        "tree": relative_tree(root, max_depth=max_depth),
        "note": "Split labels here are filename heuristics only; no dataset field is considered confirmed by this report.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory the local WOMD/WOMD-LiDAR file layout without parsing schemas.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage0.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        root = args.dataset_root or Path(str(config["dataset_root"]))
        output = args.output or PROJECT_ROOT / str(config["reports_dir"]) / "dataset_layout.json"
        report = build_report(
            root=root,
            max_depth=int(config.get("max_tree_depth", 3)),
            sample_n=int(config.get("sample_files_per_group", 8)),
        )
        write_json(output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nWrote: {output}")
        return 0
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

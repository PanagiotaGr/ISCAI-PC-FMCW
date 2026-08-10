#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from iscai_stage0.common import load_config, write_json  # noqa: E402

PACKAGE_CANDIDATES = (
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "protobuf",
    "tensorflow",
    "waymo-open-dataset",
    "waymo-open-dataset-tf-2-12-0",
    "pytest",
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_report(dataset_root: Path) -> dict[str, Any]:
    disk = None
    if dataset_root.exists():
        usage = shutil.disk_usage(dataset_root)
        disk = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "container": {
            "dockerenv_present": Path("/.dockerenv").exists(),
            "cwd": os.getcwd(),
        },
        "dataset_root": {
            "path": str(dataset_root),
            "exists": dataset_root.exists(),
            "is_dir": dataset_root.is_dir(),
            "disk": disk,
        },
        "packages": {name: package_version(name) for name in PACKAGE_CANDIDATES},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage-0 runtime environment.")
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
        dataset_root = args.dataset_root or Path(str(config["dataset_root"]))
        output = args.output or PROJECT_ROOT / str(config["reports_dir"]) / "environment_manifest.json"
        report = build_report(dataset_root)
        write_json(output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nWrote: {output}")
        return 0 if report["dataset_root"]["is_dir"] else 2
    except (KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

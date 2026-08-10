from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


STAGE0_LIDAR_SCRIPT = Path(
    "/home/agni/waymo/iscai_stage0/scripts/"
    "run_stage0_lidar_gate.py"
)


def load_frozen_stage0_lidar_decoder() -> ModuleType:
    if not STAGE0_LIDAR_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Frozen Stage-0 LiDAR decoder missing: "
            f"{STAGE0_LIDAR_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "iscai_frozen_stage0_lidar_decoder",
        STAGE0_LIDAR_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not create import spec for frozen "
            "Stage-0 LiDAR decoder."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = (
        "as_transform",
        "transform_points",
        "decompress_delta",
        "range_image_to_vehicle",
        "points_in_box",
    )

    missing = [
        name
        for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            f"Frozen Stage-0 decoder missing functions: {missing}"
        )

    return module

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import struct

from waymo_open_dataset.protos import (
    scenario_pb2,
)

from iscai_stage1.actors.womd_adapter import (
    adapt_causal_womd_scenario,
)

from iscai_stage2.observations.clean_scene import (
    build_clean_observation_scene,
)

from iscai_stage2.observations.degraded_scene import (
    build_degraded_observation_scene,
    degraded_algorithm_sha256,
)

from iscai_stage2.observations.womd_ideal_adapter import (
    build_causal_dynamic_headlamp_frames,
    build_real_ideal_observation_scene,
)

from iscai_stage3.association import (
    associate_estimated_gnn,
)

from iscai_stage3.geometry import (
    FrameTransformContext,
)

from iscai_stage3.observations import (
    algorithm_sequence_from_degraded_scene,
)

from .neural_inputs import (
    build_causal_scene_inputs,
)


ROOT = Path(
    "/home/agni/waymo"
)

TRAIN_MOTION = (
    ROOT
    / "data/"
      "paired_womd_lidar_v1_3_0/"
      "training/motion"
)

STAGE2_PROFILE = (
    ROOT
    / "iscai_stage2/scripts/"
      "run_stage2_degraded_scene_smoke.py"
)


def read_training_scenario(
    record: dict,
) -> scenario_pb2.Scenario:
    path = (
        TRAIN_MOTION
        /
        record[
            "motion_shard"
        ]
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    offset = int(
        record[
            "compact_record_offset"
        ]
    )

    expected_payload_length = int(
        record[
            "payload_length"
        ]
    )

    with path.open(
        "rb"
    ) as stream:
        stream.seek(
            offset
        )

        length_bytes = (
            stream.read(8)
        )

        if len(
            length_bytes
        ) != 8:
            raise RuntimeError(
                "Invalid TFRecord "
                "length header."
            )

        payload_length = (
            struct.unpack(
                "<Q",
                length_bytes,
            )[0]
        )

        if (
            payload_length
            !=
            expected_payload_length
        ):
            raise RuntimeError(
                "Training-index payload "
                "length does not match "
                "canonical shard."
            )

        if len(
            stream.read(4)
        ) != 4:
            raise RuntimeError(
                "Missing TFRecord "
                "length CRC."
            )

        payload = stream.read(
            payload_length
        )

        if len(
            payload
        ) != payload_length:
            raise RuntimeError(
                "Incomplete TFRecord "
                "payload."
            )

        if len(
            stream.read(4)
        ) != 4:
            raise RuntimeError(
                "Missing TFRecord "
                "data CRC."
            )

    scenario = (
        scenario_pb2.Scenario()
    )

    scenario.ParseFromString(
        payload
    )

    if (
        scenario.scenario_id
        !=
        record["scenario_id"]
    ):
        raise RuntimeError(
            "Canonical training "
            "scenario ID mismatch."
        )

    return scenario


def load_frozen_stage2_configs():
    """
    Recover only the frozen top-level
    CLEAN_CONFIG and DEGRADED_CONFIG.

    The Stage2 smoke script is parsed but
    never executed.
    """

    source = (
        STAGE2_PROFILE
        .read_text(
            encoding="utf-8"
        )
    )

    tree = ast.parse(
        source,
        filename=str(
            STAGE2_PROFILE
        ),
    )

    namespace = {
        "__builtins__":
            __builtins__,
    }

    for node in tree.body:
        if isinstance(
            node,
            ast.Import,
        ):
            for alias in node.names:
                module = (
                    importlib
                    .import_module(
                        alias.name
                    )
                )

                namespace[
                    alias.asname
                    or
                    alias.name.split(
                        "."
                    )[0]
                ] = module

        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if (
                node.module
                ==
                "__future__"
            ):
                continue

            module = (
                importlib
                .import_module(
                    node.module
                )
            )

            for alias in node.names:
                if alias.name == "*":
                    continue

                namespace[
                    alias.asname
                    or
                    alias.name
                ] = getattr(
                    module,
                    alias.name,
                )

        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            if isinstance(
                node,
                ast.Assign,
            ):
                if (
                    len(node.targets)
                    !=
                    1
                    or
                    not isinstance(
                        node.targets[0],
                        ast.Name,
                    )
                ):
                    continue

                name = (
                    node.targets[0]
                    .id
                )

                value_node = (
                    node.value
                )

            else:
                if not isinstance(
                    node.target,
                    ast.Name,
                ):
                    continue

                name = (
                    node.target.id
                )

                value_node = (
                    node.value
                )

            if value_node is None:
                continue

            try:
                namespace[
                    name
                ] = eval(
                    compile(
                        ast.Expression(
                            value_node
                        ),
                        str(
                            STAGE2_PROFILE
                        ),
                        "eval",
                    ),
                    namespace,
                    namespace,
                )
            except Exception:
                continue

    if (
        "CLEAN_CONFIG"
        not in namespace
        or
        "DEGRADED_CONFIG"
        not in namespace
    ):
        raise RuntimeError(
            "Frozen Stage2 profile "
            "could not be recovered."
        )

    return (
        namespace[
            "CLEAN_CONFIG"
        ],
        namespace[
            "DEGRADED_CONFIG"
        ],
    )


def build_real_causal_inputs(
    scenario,
    *,
    clean_config,
    degraded_config,
):
    adapted = (
        adapt_causal_womd_scenario(
            scenario
        )
    )

    ideal = (
        build_real_ideal_observation_scene(
            raw_scenario=scenario,
            adapted=adapted,
            include_sdc=False,
        )
    )

    clean = (
        build_clean_observation_scene(
            ideal_scene=ideal,
            config=clean_config,
        )
    )

    degraded = (
        build_degraded_observation_scene(
            clean_scene=clean,
            config=degraded_config,
        )
    )

    algorithm_hash = (
        degraded_algorithm_sha256(
            degraded
        )
    )

    sequence = (
        algorithm_sequence_from_degraded_scene(
            scenario_id=(
                scenario.scenario_id
            ),
            scene=degraded,
        )
    )

    associated = (
        associate_estimated_gnn(
            sequence
        )
    )

    dynamic_frames, _ = (
        build_causal_dynamic_headlamp_frames(
            adapted
        )
    )

    expected = (
        int(
            scenario.current_time_index
        )
        +
        1
    )

    if (
        len(dynamic_frames)
        !=
        expected
        or
        any(
            frame is None
            for frame
            in dynamic_frames
        )
    ):
        raise RuntimeError(
            "Invalid causal dynamic "
            "headlamp frames."
        )

    timestamps = tuple(
        float(x)
        for x in (
            scenario
            .timestamps_seconds[
                :expected
            ]
        )
    )

    context = (
        FrameTransformContext(
            T_H0_from_W=(
                adapted.frames
                .T_H0_from_W
            ),
            T_Ht_from_W_by_frame=tuple(
                frame.T_Ht_from_W
                for frame
                in dynamic_frames
            ),
        )
    )

    scene_inputs = (
        build_causal_scene_inputs(
            associated.tracks,
            scenario_id=(
                scenario.scenario_id
            ),
            context=context,
            timestamps_s=(
                timestamps
            ),
        )
    )

    return {
        "adapted":
            adapted,

        "degraded":
            degraded,

        "stage2_algorithm_sha256":
            algorithm_hash,

        "association_track_count":
            len(
                associated.tracks
            ),

        "scene_inputs":
            scene_inputs,
    }

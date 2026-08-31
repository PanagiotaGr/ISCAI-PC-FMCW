from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S6 = ROOT / "iscai_stage6"

BLOCK65_HANDOFF = (
    S6
    / "artifacts/block65/"
      "block65_to_block66_handoff.json"
)

PART0B_REPORT = (
    S6
    / "reports/"
      "block66_part0b_development_partition_resolution.json"
)

BLOCK41_REPORT = (
    S4
    / "reports/"
      "block41_training_split_gate.json"
)

DEVELOPMENT_MANIFEST = (
    S4
    / "artifacts/block41/"
      "development.jsonl"
)

FORMAL_MANIFEST = (
    S3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

BLOCK64_NUMERIC = (
    S6
    / "configs/"
      "block64_mc_grid_numeric_freeze.json"
)

PARTIAL = (
    S6
    / "artifacts/block66/"
      "block66_part1a_headlamp_census.partial.jsonl"
)

ELIGIBLE_MANIFEST = (
    S6
    / "artifacts/block66/"
      "block66_part1a_headlamp_relevant_scenarios.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block66_part1a_forward_headlamp_census.json"
)


EXPECTED = {
    "block65_handoff":
        (
            "124aaedfb39867109a5ddbf5dc7ef243"
            "188df798e5ac19b4b289418ec5ffe700"
        ),

    "part0b_report":
        (
            "1e7860b9589d7ce62791c5c48802fb06"
            "f90e2ccfca5d90b8ec6b528e5e3bb032"
        ),

    "block41_report":
        (
            "428063912d90cc480d0ddaa61c82007a"
            "63e05fc0e918b0fe7da4c30ed103ff53"
        ),

    "development_manifest":
        (
            "e4689698bddd80e58add7267f791aea90"
            "ef4bef0309ca18d0d7a9e7e94fe7e5c"
        ),

    "formal_manifest":
        (
            "2208e7287ddf6439fda4597c435a9cba"
            "1d1b9d0e4c4547bc5dd92e56e8124e46"
        ),

    "block64_numeric":
        (
            "993c4248a902e7dff3a4383ac343e722"
            "2cc43ef9b9372ed2efd0ee08d6d20a73"
        ),
}


CORE_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

EXPECTED_DEVELOPMENT_SCENARIOS = 7208

GEOMETRY_EPS = 1.0e-12


class ControlledBlock(RuntimeError):
    pass


def require(
    condition: Any,
    message: str,
) -> None:
    if not bool(condition):
        raise ControlledBlock(message)


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_line(
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"


def canonical_json_bytes(
    payload: Any,
) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_exact_or_create(
    path: Path,
    payload: Any,
) -> str:
    expected = canonical_json_bytes(payload)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes() == expected,
            (
                "Existing JSON differs from "
                f"deterministic rerun: {path}"
            ),
        )
    else:
        path.write_bytes(expected)

    return sha256_file(path)


def class_count_dict(
    value,
) -> dict[str, int]:
    result = {
        name: 0
        for name in CORE_CLASSES
    }

    iterator = (
        value.items()
        if isinstance(value, dict)
        else value
    )

    for item in iterator:
        if isinstance(value, dict):
            name, count = item
        else:
            require(
                isinstance(item, (list, tuple))
                and
                len(item) == 2,
                (
                    "Malformed anchor_class_counts "
                    f"entry: {item!r}"
                ),
            )

            name, count = item

        name = str(name)

        if name in result:
            result[name] = int(count)

    return result


# ============================================================
# Exact current full-box forward-sector geometry
# ============================================================

def cross2(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        (a[0] - o[0])
        *
        (b[1] - o[1])
        -
        (a[1] - o[1])
        *
        (b[0] - o[0])
    )


def convex_hull_xy(
    points,
) -> np.ndarray:
    array = np.asarray(
        points,
        dtype=np.float64,
    )

    require(
        array.ndim == 2
        and
        array.shape[1] == 2,
        (
            "convex_hull_xy expects [N,2]."
        ),
    )

    require(
        np.all(np.isfinite(array)),
        (
            "Non-finite XY corner."
        ),
    )

    unique = sorted({
        (
            float(row[0]),
            float(row[1]),
        )
        for row in array
    })

    require(
        len(unique) >= 3,
        (
            "Full box footprint has fewer "
            "than three unique XY points."
        ),
    )

    pts = [
        np.asarray(
            item,
            dtype=np.float64,
        )
        for item in unique
    ]

    lower = []

    for point in pts:
        while (
            len(lower) >= 2
            and
            cross2(
                lower[-2],
                lower[-1],
                point,
            )
            <= 0.0
        ):
            lower.pop()

        lower.append(point)

    upper = []

    for point in reversed(pts):
        while (
            len(upper) >= 2
            and
            cross2(
                upper[-2],
                upper[-1],
                point,
            )
            <= 0.0
        ):
            upper.pop()

        upper.append(point)

    hull = np.asarray(
        lower[:-1] + upper[:-1],
        dtype=np.float64,
    )

    require(
        hull.shape[0] >= 3,
        (
            "Degenerate box ground footprint."
        ),
    )

    return hull


def clip_polygon_halfplane(
    polygon: np.ndarray,
    *,
    a: float,
    b: float,
    c: float = 0.0,
    eps: float = GEOMETRY_EPS,
) -> np.ndarray:
    """
    Sutherland-Hodgman clipping for

        a*x + b*y + c <= 0.

    Used only on current causal box geometry.
    """

    polygon = np.asarray(
        polygon,
        dtype=np.float64,
    )

    if polygon.shape[0] == 0:
        return polygon.copy()

    result = []

    def value(point):
        return (
            a * float(point[0])
            +
            b * float(point[1])
            +
            c
        )

    for index in range(
        polygon.shape[0]
    ):
        current = polygon[index]

        previous = polygon[
            index - 1
        ]

        current_value = value(
            current
        )

        previous_value = value(
            previous
        )

        current_inside = (
            current_value
            <=
            eps
        )

        previous_inside = (
            previous_value
            <=
            eps
        )

        if (
            current_inside
            !=
            previous_inside
        ):
            denominator = (
                previous_value
                -
                current_value
            )

            require(
                abs(denominator)
                >
                0.0,
                (
                    "Half-plane clipping "
                    "encountered zero denominator."
                ),
            )

            fraction = (
                previous_value
                /
                denominator
            )

            intersection = (
                previous
                +
                fraction
                *
                (
                    current
                    -
                    previous
                )
            )

            result.append(
                intersection
            )

        if current_inside:
            result.append(
                current
            )

    if not result:
        return np.empty(
            (
                0,
                2,
            ),
            dtype=np.float64,
        )

    return np.asarray(
        result,
        dtype=np.float64,
    )


def point_segment_distance_origin(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    delta = (
        b - a
    )

    denominator = float(
        np.dot(
            delta,
            delta,
        )
    )

    if denominator <= 0.0:
        return float(
            np.linalg.norm(a)
        )

    t = float(
        -np.dot(
            a,
            delta,
        )
        /
        denominator
    )

    t = min(
        1.0,
        max(
            0.0,
            t,
        ),
    )

    closest = (
        a
        +
        t
        *
        delta
    )

    return float(
        np.linalg.norm(
            closest
        )
    )


def minimum_radius_to_polygon(
    polygon: np.ndarray,
) -> float:
    polygon = np.asarray(
        polygon,
        dtype=np.float64,
    )

    require(
        polygon.ndim == 2
        and
        polygon.shape[1] == 2
        and
        polygon.shape[0] > 0,
        (
            "Invalid clipped polygon."
        ),
    )

    distances = []

    for index in range(
        polygon.shape[0]
    ):
        a = polygon[index]

        b = polygon[
            (index + 1)
            %
            polygon.shape[0]
        ]

        distances.append(
            point_segment_distance_origin(
                a,
                b,
            )
        )

    return float(
        min(distances)
    )


def box_intersects_forward_sector(
    box,
    *,
    half_angle_rad: float,
    maximum_range_m: float,
    box_corners_headlamp,
) -> bool:
    """
    Continuous intersection of the CURRENT full-box
    ground footprint with:

        |theta| <= half_angle_rad
        0 <= r <= maximum_range_m.

    Angular clipping is performed in Cartesian H0,
    avoiding the +/-pi wrap ambiguity.
    """

    corners = np.asarray(
        box_corners_headlamp(
            box
        ),
        dtype=np.float64,
    )

    require(
        corners.shape
        ==
        (
            8,
            3,
        ),
        (
            "Expected exactly eight physical "
            f"box corners; got {corners.shape}."
        ),
    )

    footprint = convex_hull_xy(
        corners[:, :2]
    )

    tangent = math.tan(
        half_angle_rad
    )

    # Upper ray:
    # y <= x*tan(alpha)
    clipped = clip_polygon_halfplane(
        footprint,
        a=-tangent,
        b=1.0,
    )

    if clipped.shape[0] == 0:
        return False

    # Lower ray:
    # y >= -x*tan(alpha)
    # -y - x*tan(alpha) <= 0
    clipped = clip_polygon_halfplane(
        clipped,
        a=-tangent,
        b=-1.0,
    )

    if clipped.shape[0] == 0:
        return False

    minimum_radius = (
        minimum_radius_to_polygon(
            clipped
        )
    )

    return bool(
        minimum_radius
        <=
        (
            maximum_range_m
            +
            GEOMETRY_EPS
        )
    )


def assign_stratum(
    relevant_counts:
        dict[str, int],
) -> str:
    if (
        relevant_counts[
            "TYPE_CYCLIST"
        ]
        >
        0
    ):
        return "cyclist"

    if (
        relevant_counts[
            "TYPE_PEDESTRIAN"
        ]
        >
        0
    ):
        return "pedestrian_no_cyclist"

    if (
        relevant_counts[
            "TYPE_VEHICLE"
        ]
        >
        0
    ):
        return "vehicle_only"

    return "none"


def load_partial(
    *,
    source_rows,
) -> list[
    dict[str, Any]
]:
    if not PARTIAL.exists():
        return []

    results = []

    for line_number, line in enumerate(
        PARTIAL.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        item = json.loads(
            line
        )

        expected_rank = len(
            results
        )

        require(
            int(
                item[
                    "source_rank"
                ]
            )
            ==
            expected_rank,
            (
                "Partial census is not "
                "contiguous at line "
                f"{line_number}."
            ),
        )

        require(
            expected_rank
            <
            len(
                source_rows
            ),
            (
                "Partial census contains "
                "too many records."
            ),
        )

        source = source_rows[
            expected_rank
        ]

        require(
            str(
                item[
                    "scenario_id"
                ]
            )
            ==
            str(
                source[
                    "scenario_id"
                ]
            ),
            (
                "Partial/source scenario "
                "mismatch."
            ),
        )

        require(
            str(
                item[
                    "selection_hash"
                ]
            )
            ==
            str(
                source[
                    "selection_hash"
                ]
            ),
            (
                "Partial/source selection "
                "hash mismatch."
            ),
        )

        results.append(
            item
        )

    return results


def main() -> None:
    print("=" * 72)
    print(
        "BLOCK 6.6 PART1A — CURRENT-CAUSAL "
        "FORWARD-HEADLAMP DEVELOPMENT CENSUS"
    )
    print("=" * 72)

    # ========================================================
    # A. Frozen upstream seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE UPSTREAM SEAL ====="
    )

    for path, expected, label in (
        (
            BLOCK65_HANDOFF,
            EXPECTED[
                "block65_handoff"
            ],
            "Block6.5 handoff",
        ),
        (
            PART0B_REPORT,
            EXPECTED[
                "part0b_report"
            ],
            "Block6.6 Part0B report",
        ),
        (
            BLOCK41_REPORT,
            EXPECTED[
                "block41_report"
            ],
            "Block4.1 split report",
        ),
        (
            DEVELOPMENT_MANIFEST,
            EXPECTED[
                "development_manifest"
            ],
            "Stage4 development manifest",
        ),
        (
            FORMAL_MANIFEST,
            EXPECTED[
                "formal_manifest"
            ],
            "formal N120 manifest",
        ),
        (
            BLOCK64_NUMERIC,
            EXPECTED[
                "block64_numeric"
            ],
            "Stage6 grid numeric freeze",
        ),
    ):
        require(
            path.is_file(),
            f"Missing {label}: {path}",
        )

        actual = sha256_file(
            path
        )

        require(
            actual == expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:28s} = EXACT PASS"
        )

    print()
    print(
        "Stage6 policy source partition = DEVELOPMENT"
    )

    print(
        "Stage4 calibration partition reused for policy tuning = NO"
    )

    # ========================================================
    # B. Development manifest + formal exclusion
    # ========================================================

    print()
    print(
        "===== B. DEVELOPMENT PARTITION BOUNDARY ====="
    )

    source_rows = [
        json.loads(
            line
        )
        for line in
        DEVELOPMENT_MANIFEST
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    require(
        len(
            source_rows
        )
        ==
        EXPECTED_DEVELOPMENT_SCENARIOS,
        (
            "Development manifest count "
            f"changed: {len(source_rows)}"
        ),
    )

    development_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in source_rows
    ]

    require(
        len(
            set(
                development_ids
            )
        )
        ==
        len(
            development_ids
        ),
        (
            "Development scenario IDs "
            "are not unique."
        ),
    )

    formal_rows = [
        json.loads(
            line
        )
        for line in
        FORMAL_MANIFEST
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]

    require(
        len(
            formal_rows
        )
        ==
        120,
        (
            "Formal manifest count changed."
        ),
    )

    formal_ids = {
        str(
            row[
                "scenario_id"
            ]
        )
        for row in formal_rows
    }

    overlap = (
        set(
            development_ids
        )
        &
        formal_ids
    )

    require(
        not overlap,
        (
            "Development partition overlaps "
            "formal N120."
        ),
    )

    print(
        "development scenarios =",
        len(
            source_rows
        ),
    )

    print(
        "formal overlap        = 0"
    )

    print(
        "formal outcomes read  = NO"
    )

    # ========================================================
    # C. Frozen headlamp-domain semantics
    # ========================================================

    print()
    print(
        "===== C. FROZEN HEADLAMP DOMAIN ====="
    )

    numeric = json.loads(
        BLOCK64_NUMERIC.read_text(
            encoding="utf-8"
        )
    )

    grid = numeric[
        "scientific_illumination_grid"
    ]

    theta = grid[
        "theta_centers_deg"
    ]

    radial = grid[
        "range_centers_m"
    ]

    require(
        float(
            theta[
                "minimum"
            ]
        )
        ==
        -25.0,
        (
            "Frozen theta minimum changed."
        ),
    )

    require(
        float(
            theta[
                "maximum"
            ]
        )
        ==
        25.0,
        (
            "Frozen theta maximum changed."
        ),
    )

    require(
        float(
            radial[
                "minimum"
            ]
        )
        ==
        0.0,
        (
            "Frozen range minimum changed."
        ),
    )

    require(
        float(
            radial[
                "maximum"
            ]
        )
        ==
        150.0,
        (
            "Frozen range maximum changed."
        ),
    )

    half_angle_rad = math.radians(
        25.0
    )

    maximum_range_m = 150.0

    print(
        "theta domain = [-25,+25] deg"
    )

    print(
        "range domain = [0,150] m"
    )

    print(
        "selection geometry = "
        "CONTINUOUS FULL-BOX FOOTPRINT / FINITE SECTOR"
    )

    print(
        "centroid selection = NO"
    )

    print(
        "raster-cell selection = NO"
    )

    print(
        "angular wrap ambiguity = AVOIDED IN CARTESIAN H0"
    )

    # ========================================================
    # D. Geometry self-check
    # ========================================================

    print()
    print(
        "===== D. CURRENT FULL-BOX GEOMETRY SELF-CHECK ====="
    )

    from iscai_stage6.adb.geometry import (
        Box3D,
        box_corners_headlamp,
    )

    front = Box3D(
        center_xyz=(
            20.0,
            0.0,
            0.0,
        ),
        length_m=4.0,
        width_m=2.0,
        height_m=1.5,
        yaw_rad=0.0,
    )

    behind = Box3D(
        center_xyz=(
            -20.0,
            0.0,
            0.0,
        ),
        length_m=4.0,
        width_m=2.0,
        height_m=1.5,
        yaw_rad=0.0,
    )

    beyond = Box3D(
        center_xyz=(
            160.0,
            0.0,
            0.0,
        ),
        length_m=4.0,
        width_m=2.0,
        height_m=1.5,
        yaw_rad=0.0,
    )

    require(
        box_intersects_forward_sector(
            front,
            half_angle_rad=(
                half_angle_rad
            ),
            maximum_range_m=(
                maximum_range_m
            ),
            box_corners_headlamp=(
                box_corners_headlamp
            ),
        ),
        (
            "Synthetic front box was "
            "not detected."
        ),
    )

    require(
        not box_intersects_forward_sector(
            behind,
            half_angle_rad=(
                half_angle_rad
            ),
            maximum_range_m=(
                maximum_range_m
            ),
            box_corners_headlamp=(
                box_corners_headlamp
            ),
        ),
        (
            "Synthetic behind box "
            "entered forward sector."
        ),
    )

    require(
        not box_intersects_forward_sector(
            beyond,
            half_angle_rad=(
                half_angle_rad
            ),
            maximum_range_m=(
                maximum_range_m
            ),
            box_corners_headlamp=(
                box_corners_headlamp
            ),
        ),
        (
            "Synthetic beyond-range box "
            "entered finite sector."
        ),
    )

    print(
        "front box   = RELEVANT PASS"
    )

    print(
        "behind box  = IRRELEVANT PASS"
    )

    print(
        "beyond box  = IRRELEVANT PASS"
    )

    # ========================================================
    # E. Resumable real development scan
    # ========================================================

    print()
    print(
        "===== E. RESUMABLE 7208-SCENARIO CAUSAL SCAN ====="
    )

    from iscai_stage4.data.real_pipeline import (
        read_training_scenario,
    )

    from iscai_stage1.actors.womd_adapter import (
        adapt_causal_womd_scenario,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
    )

    completed = load_partial(
        source_rows=(
            source_rows
        ),
    )

    start_rank = len(
        completed
    )

    print(
        "already completed =",
        start_rank,
    )

    print(
        "remaining         =",
        (
            len(
                source_rows
            )
            -
            start_rank
        ),
    )

    PARTIAL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = (
        "a"
        if PARTIAL.exists()
        else "w"
    )

    with PARTIAL.open(
        mode,
        encoding="utf-8",
    ) as output:

        for rank in range(
            start_rank,
            len(
                source_rows
            ),
        ):
            source = source_rows[
                rank
            ]

            scenario = (
                read_training_scenario(
                    source
                )
            )

            require(
                str(
                    scenario.scenario_id
                )
                ==
                str(
                    source[
                        "scenario_id"
                    ]
                ),
                (
                    "Random-access scenario ID "
                    f"mismatch at rank {rank}."
                ),
            )

            adapted = (
                adapt_causal_womd_scenario(
                    scenario
                )
            )

            boxes = (
                build_causal_adb_actor_boxes(
                    scenario=scenario,
                    adapted=adapted,
                )
            )

            actual_counts = Counter(
                str(
                    box.object_type
                )
                for box in boxes
            )

            manifest_counts = (
                class_count_dict(
                    source[
                        "anchor_class_counts"
                    ]
                )
            )

            actual_core = {
                name:
                    int(
                        actual_counts[
                            name
                        ]
                    )
                for name in CORE_CLASSES
            }

            require(
                actual_core
                ==
                manifest_counts,
                (
                    "Stage4/Stage6 current-core "
                    "count mismatch at rank "
                    f"{rank}: "
                    f"{actual_core} != "
                    f"{manifest_counts}"
                ),
            )

            relevant_counts = {
                name: 0
                for name in CORE_CLASSES
            }

            for box in boxes:
                actor_class = str(
                    box.object_type
                )

                require(
                    box.future_state_used
                    is False,
                    (
                        "Future state flag became "
                        "true in cohort census."
                    ),
                )

                require(
                    box.tracks_to_predict_used
                    is False,
                    (
                        "tracks_to_predict entered "
                        "cohort census."
                    ),
                )

                require(
                    box.objects_of_interest_used
                    is False,
                    (
                        "objects_of_interest entered "
                        "cohort census."
                    ),
                )

                relevant = (
                    box_intersects_forward_sector(
                        box.box,
                        half_angle_rad=(
                            half_angle_rad
                        ),
                        maximum_range_m=(
                            maximum_range_m
                        ),
                        box_corners_headlamp=(
                            box_corners_headlamp
                        ),
                    )
                )

                if relevant:
                    relevant_counts[
                        actor_class
                    ] += 1

            stratum = assign_stratum(
                relevant_counts
            )

            result = {
                "source_rank":
                    int(rank),

                "scenario_id":
                    str(
                        source[
                            "scenario_id"
                        ]
                    ),

                "motion_shard":
                    str(
                        source[
                            "motion_shard"
                        ]
                    ),

                "compact_record_offset":
                    int(
                        source[
                            "compact_record_offset"
                        ]
                    ),

                "payload_length":
                    int(
                        source[
                            "payload_length"
                        ]
                    ),

                "selection_hash":
                    str(
                        source[
                            "selection_hash"
                        ]
                    ),

                "current_anchor_core_counts":
                    manifest_counts,

                "headlamp_relevant_counts":
                    relevant_counts,

                "headlamp_stratum":
                    stratum,

                "current_time_index":
                    int(
                        scenario.current_time_index
                    ),

                "selection_uses_full_box":
                    True,

                "selection_uses_centroid":
                    False,

                "selection_uses_future":
                    False,

                "tracks_to_predict_used":
                    False,

                "objects_of_interest_used":
                    False,
            }

            output.write(
                canonical_line(
                    result
                )
            )

            output.flush()

            if (
                (rank + 1)
                %
                100
                ==
                0
            ):
                try:
                    import os

                    os.fsync(
                        output.fileno()
                    )
                except BaseException:
                    pass

            if (
                (rank + 1)
                %
                250
                ==
                0
                or
                rank
                ==
                len(
                    source_rows
                )
                -
                1
            ):
                print(
                    f"progress = {rank + 1}/"
                    f"{len(source_rows)}"
                )

    # ========================================================
    # F. Full census integrity
    # ========================================================

    print()
    print(
        "===== F. COMPLETE HEADLAMP CENSUS ====="
    )

    census = load_partial(
        source_rows=(
            source_rows
        ),
    )

    require(
        len(
            census
        )
        ==
        EXPECTED_DEVELOPMENT_SCENARIOS,
        (
            "Census incomplete after scan: "
            f"{len(census)}"
        ),
    )

    scenes_with_relevant_class = (
        Counter()
    )

    relevant_actor_totals = (
        Counter()
    )

    strata = Counter()

    for row in census:
        relevant = row[
            "headlamp_relevant_counts"
        ]

        for actor_class in CORE_CLASSES:
            count = int(
                relevant[
                    actor_class
                ]
            )

            relevant_actor_totals[
                actor_class
            ] += count

            if count > 0:
                scenes_with_relevant_class[
                    actor_class
                ] += 1

        strata[
            str(
                row[
                    "headlamp_stratum"
                ]
            )
        ] += 1

    class_scene_counts = {
        name:
            int(
                scenes_with_relevant_class[
                    name
                ]
            )
        for name in CORE_CLASSES
    }

    class_actor_counts = {
        name:
            int(
                relevant_actor_totals[
                    name
                ]
            )
        for name in CORE_CLASSES
    }

    stratum_counts = {
        key:
            int(
                value
            )
        for key, value in
        sorted(
            strata.items()
        )
    }

    print(
        "headlamp-relevant scenes by class =",
        class_scene_counts,
    )

    print(
        "headlamp-relevant actors by class =",
        class_actor_counts,
    )

    print(
        "headlamp causal strata =",
        stratum_counts,
    )

    for actor_class in CORE_CLASSES:
        require(
            class_scene_counts[
                actor_class
            ]
            >
            0,
            (
                "Development partition has "
                "no forward-headlamp support "
                f"for {actor_class}."
            ),
        )

        require(
            class_actor_counts[
                actor_class
            ]
            >
            0,
            (
                "Development partition has "
                "zero forward-headlamp actors "
                f"for {actor_class}."
            ),
        )

    # ========================================================
    # G. Materialize eligible-scenario manifest
    # ========================================================

    print()
    print(
        "===== G. HEADLAMP-RELEVANT DEVELOPMENT MANIFEST ====="
    )

    eligible = [
        row
        for row in census
        if (
            str(
                row[
                    "headlamp_stratum"
                ]
            )
            !=
            "none"
        )
    ]

    eligible_bytes = "".join(
        canonical_line(
            row
        )
        for row in eligible
    ).encode(
        "utf-8"
    )

    if ELIGIBLE_MANIFEST.exists():
        require(
            ELIGIBLE_MANIFEST.read_bytes()
            ==
            eligible_bytes,
            (
                "Existing eligible manifest "
                "differs from deterministic census."
            ),
        )
    else:
        ELIGIBLE_MANIFEST.write_bytes(
            eligible_bytes
        )

    eligible_sha = sha256_file(
        ELIGIBLE_MANIFEST
    )

    partial_sha = sha256_file(
        PARTIAL
    )

    print(
        "eligible scenarios =",
        len(
            eligible
        ),
    )

    print(
        "full census SHA256 =",
        partial_sha,
    )

    print(
        "eligible manifest SHA256 =",
        eligible_sha,
    )

    # Deterministic examples only; not cohort selection.
    examples = {}

    for actor_class in CORE_CLASSES:
        candidates = [
            row
            for row in eligible
            if (
                int(
                    row[
                        "headlamp_relevant_counts"
                    ][
                        actor_class
                    ]
                )
                >
                0
            )
        ]

        chosen = min(
            candidates,
            key=lambda row: (
                str(
                    row[
                        "selection_hash"
                    ]
                ),
                str(
                    row[
                        "scenario_id"
                    ]
                ),
            ),
        )

        examples[
            actor_class
        ] = {
            "scenario_id":
                str(
                    chosen[
                        "scenario_id"
                    ]
                ),

            "selection_hash":
                str(
                    chosen[
                        "selection_hash"
                    ]
                ),

            "headlamp_relevant_counts":
                chosen[
                    "headlamp_relevant_counts"
                ],
        }

    print(
        "deterministic examples ="
    )

    print(
        json.dumps(
            examples,
            indent=2,
            sort_keys=True,
        )
    )

    # ========================================================
    # H. Report / boundary
    # ========================================================

    print()
    print(
        "===== H. BLOCK6.6 PART1A REPORT ====="
    )

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part1A",

        "status":
            (
                "PASS_CAUSAL_FORWARD_HEADLAMP_"
                "DEVELOPMENT_CENSUS"
            ),

        "source_partition": {
            "name":
                "Stage4_development",

            "manifest_path":
                str(
                    DEVELOPMENT_MANIFEST
                ),

            "manifest_sha256":
                EXPECTED[
                    "development_manifest"
                ],

            "scenario_count":
                EXPECTED_DEVELOPMENT_SCENARIOS,

            "formal_overlap":
                0,

            "Stage4_calibration_reused_for_policy_selection":
                False,
        },

        "selection_semantics": {
            "time":
                "current_anchor_only",

            "classes":
                list(
                    CORE_CLASSES
                ),

            "geometry":
                (
                    "continuous_current_full_box_"
                    "ground_footprint_intersection_"
                    "with_finite_forward_headlamp_sector"
                ),

            "coordinate_frame":
                "Stage6_H0_headlamp",

            "theta_deg":
                [
                    -25.0,
                    25.0,
                ],

            "maximum_ground_range_m":
                150.0,

            "boundary_inclusive":
                True,

            "centroid_only":
                False,

            "raster_cell_selection":
                False,

            "angular_wrap_handling":
                (
                    "Cartesian_halfplane_clipping_"
                    "before_range_intersection"
                ),

            "future_states":
                False,

            "tracks_to_predict":
                False,

            "objects_of_interest":
                False,
        },

        "census": {
            "headlamp_relevant_scenes_by_class":
                class_scene_counts,

            "headlamp_relevant_actors_by_class":
                class_actor_counts,

            "mutually_exclusive_strata":
                stratum_counts,

            "eligible_scenario_count":
                len(
                    eligible
                ),
        },

        "artifacts": {
            "full_census": {
                "path":
                    str(
                        PARTIAL
                    ),

                "sha256":
                    partial_sha,
            },

            "eligible_manifest": {
                "path":
                    str(
                        ELIGIBLE_MANIFEST
                    ),

                "sha256":
                    eligible_sha,
            },
        },

        "deterministic_examples_not_cohort_selection":
            examples,

        "scientific_boundary": {
            "development_cohort_selected":
                False,

            "cohort_size_frozen":
                False,

            "class_policy_preregistered":
                False,

            "class_policy_numeric_freeze":
                False,

            "new_model_forward":
                False,

            "formal_outcomes_read":
                False,

            "formal_tuning":
                False,

            "source_modification":
                False,
        },

        "next":
            (
                "BLOCK6.6_PART1B_DETERMINISTIC_"
                "CLASS_COVERED_COHORT_SELECTION"
            ),
    }

    report_sha = (
        write_json_exact_or_create(
            REPORT,
            report_payload,
        )
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        report_sha,
    )

    # ========================================================
    # I. Final
    # ========================================================

    print()
    print("=" * 72)
    print(
        "BLOCK 6.6 PART1A — FINAL"
    )
    print("=" * 72)

    print(
        "STATUS = "
        "PASS_CAUSAL_FORWARD_HEADLAMP_DEVELOPMENT_CENSUS"
    )

    print(
        "policy-development source = STAGE4 DEVELOPMENT"
    )

    print(
        "source scenarios          =",
        EXPECTED_DEVELOPMENT_SCENARIOS,
    )

    print(
        "formal overlap            = 0"
    )

    print(
        "current full-box geometry = YES"
    )

    print(
        "centroid-only             = NO"
    )

    print(
        "future selection          = NO"
    )

    print(
        "tracks_to_predict         = NO"
    )

    print(
        "objects_of_interest       = NO"
    )

    print(
        "headlamp scenes/class     =",
        class_scene_counts,
    )

    print(
        "headlamp actors/class     =",
        class_actor_counts,
    )

    print(
        "headlamp strata           =",
        stratum_counts,
    )

    print(
        "eligible scenario count   =",
        len(
            eligible
        ),
    )

    print(
        "cohort selected           = NO"
    )

    print(
        "cohort size frozen        = NO"
    )

    print(
        "class policy frozen       = NO"
    )

    print(
        "model forward             = NO"
    )

    print(
        "formal outcomes read      = NO"
    )

    print(
        "full census SHA256        =",
        partial_sha,
    )

    print(
        "eligible manifest SHA256  =",
        eligible_sha,
    )

    print(
        "report SHA256             =",
        report_sha,
    )

    print(
        "NEXT = BLOCK6.6 PART1B "
        "DETERMINISTIC CLASS-COVERED COHORT SELECTION"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 72)


try:
    main()

except BaseException as exc:
    print()
    print("=" * 72)
    print(
        "BLOCK6.6 PART1A — CONTROLLED BLOCK"
    )
    print("=" * 72)

    print(
        "exception type =",
        type(exc).__name__,
    )

    print(
        "exception      =",
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=18
    )

    print()
    print(
        "partial census preserved =",
        PARTIAL.exists(),
    )

    print(
        "development cohort selected = NO"
    )

    print(
        "cohort size frozen          = NO"
    )

    print(
        "class-policy numeric freeze = NO"
    )

    print(
        "formal outcomes read        = NO"
    )

    print(
        "model forward               = NO"
    )

    print(
        "source modification         = NO"
    )

    print(
        "terminal remains open       = YES"
    )

# Deliberately no sys.exit().

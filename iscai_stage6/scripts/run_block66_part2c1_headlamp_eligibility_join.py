from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path("/home/agni/waymo")
S6 = ROOT / "iscai_stage6"

COHORT = (
    S6
    / "artifacts/block66/"
      "block66_class_aware_development_cohort_120.jsonl"
)

PART1A_REPORT = (
    S6
    / "reports/"
      "block66_part1a_forward_headlamp_census.json"
)

PART1B_REPORT = (
    S6
    / "reports/"
      "block66_part1b_development_cohort_freeze.json"
)

PART2B_REPORT = (
    S6
    / "reports/"
      "block66_part2b_cohort_posterior_association_census.json"
)

POSTERIOR = (
    S6
    / "artifacts/block66/"
      "block66_development_identity_safe_gaussian_posterior.jsonl"
)

MATCHES = (
    S6
    / "artifacts/block66/"
      "block66_development_association_matches.jsonl"
)

FALLBACKS = (
    S6
    / "artifacts/block66/"
      "block66_development_reactive_fallback_boxes.jsonl"
)

ELIGIBLE_MATCHES = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

ELIGIBLE_FALLBACKS = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligible_reactive_fallbacks.jsonl"
)

ELIGIBILITY = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_actor_eligibility.jsonl"
)

SCENE_SUMMARY = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_scene_summary.jsonl"
)

REPORT = (
    S6
    / "reports/"
      "block66_part2c1_headlamp_eligibility_join.json"
)

FREEZE = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_headlamp_eligibility_freeze_manifest.json"
)

HANDOFF = (
    S6
    / "artifacts/block66/"
      "block66_part2c1_to_part2c2_handoff.json"
)


EXPECTED = {
    "cohort":
        (
            "57440e3d976e7aeff44811b5eeb075dc"
            "56e0e0e070a3832ec72d354253b998c9"
        ),

    "part1a":
        (
            "553188a6d01a890e9a6c2f11e9bbfd2d"
            "8426665065c4abcc770fa7e30ca84228"
        ),

    "part1b":
        (
            "3a965b4ef3c451e7526608f2b5ceb5a1"
            "51eb08caf922a25f1b25a1ec5a12f5f0"
        ),

    "part2b":
        (
            "96dfccd808fd06d77b6f9d4b1c100f6"
            "a59db7d090a405f41b88d4fa6854c9705"
        ),

    "posterior":
        (
            "f022471fbf86c3121106011bfd7f035c5"
            "d895f2c5aeeaa2fa3a20c04b3439acd"
        ),

    "matches":
        (
            "1011d443998e7b33600911954d67c4377"
            "aefc63ec4455fb3d67b24e281661848"
        ),

    "fallbacks":
        (
            "32af5397b3ae4a3311dcf401a36bd247"
            "cfbca5fbdb9bf7e883b0a9a4e34bbb3b"
        ),
}


CORE_CLASSES = (
    "TYPE_VEHICLE",
    "TYPE_PEDESTRIAN",
    "TYPE_CYCLIST",
)

EXPECTED_HEADLAMP_ACTORS = {
    "TYPE_VEHICLE":
        890,

    "TYPE_PEDESTRIAN":
        178,

    "TYPE_CYCLIST":
        56,
}

EXPECTED_HEADLAMP_SCENES = {
    "TYPE_VEHICLE":
        116,

    "TYPE_PEDESTRIAN":
        56,

    "TYPE_CYCLIST":
        40,
}

HALF_ANGLE_RAD = math.radians(
    25.0
)

MAXIMUM_RANGE_M = 150.0

GEOMETRY_EPS = 1.0e-12


class ControlledBlock(RuntimeError):
    pass


def require(
    condition: Any,
    message: str,
):
    if not bool(condition):
        raise ControlledBlock(
            message
        )


def sha256_file(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_jsonl(
    path: Path,
):
    return [
        json.loads(line)
        for line in
        path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def canonical_line(
    payload,
):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    )


def write_jsonl_exact_or_create(
    path: Path,
    rows,
):
    data = "".join(
        canonical_line(
            row
        )
        for row in rows
    ).encode(
        "utf-8"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing JSONL differs from "
                f"deterministic rerun: {path}"
            ),
        )
    else:
        path.write_bytes(
            data
        )

    return sha256_file(
        path
    )


def write_json_exact_or_create(
    path: Path,
    payload,
):
    data = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        require(
            path.read_bytes()
            ==
            data,
            (
                "Existing JSON differs from "
                f"deterministic rerun: {path}"
            ),
        )

    else:
        path.write_bytes(
            data
        )

    return sha256_file(
        path
    )


def empty_class_counter():
    return {
        name: 0
        for name in CORE_CLASSES
    }


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
        "convex_hull_xy expects [N,2].",
    )

    require(
        np.all(
            np.isfinite(
                array
            )
        ),
        "Non-finite XY corner.",
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
            "Full box footprint has "
            "fewer than three unique points."
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

        lower.append(
            point
        )

    upper = []

    for point in reversed(
        pts
    ):
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

        upper.append(
            point
        )

    hull = np.asarray(
        lower[:-1]
        +
        upper[:-1],
        dtype=np.float64,
    )

    require(
        hull.shape[0] >= 3,
        "Degenerate ground footprint.",
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
        current = polygon[
            index
        ]

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
                abs(
                    denominator
                )
                >
                0.0,
                (
                    "Half-plane clipping "
                    "zero denominator."
                ),
            )

            fraction = (
                previous_value
                /
                denominator
            )

            result.append(
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
        b
        -
        a
    )

    denominator = float(
        np.dot(
            delta,
            delta,
        )
    )

    if denominator <= 0.0:
        return float(
            np.linalg.norm(
                a
            )
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
        "Invalid clipped polygon.",
    )

    distances = []

    for index in range(
        polygon.shape[0]
    ):
        distances.append(
            point_segment_distance_origin(
                polygon[
                    index
                ],
                polygon[
                    (
                        index + 1
                    )
                    %
                    polygon.shape[0]
                ],
            )
        )

    return float(
        min(
            distances
        )
    )


def box_intersects_forward_sector(
    box,
    *,
    box_corners_headlamp,
) -> bool:

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
            "Expected exactly 8 "
            f"box corners; got {corners.shape}."
        ),
    )

    footprint = convex_hull_xy(
        corners[
            :,
            :2
        ]
    )

    tangent = math.tan(
        HALF_ANGLE_RAD
    )

    clipped = clip_polygon_halfplane(
        footprint,
        a=-tangent,
        b=1.0,
    )

    if clipped.shape[0] == 0:
        return False

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
            MAXIMUM_RANGE_M
            +
            GEOMETRY_EPS
        )
    )


def current_box_payload(
    box,
):
    geometry = box.box

    center = [
        float(value)
        for value in
        geometry.center_xyz
    ]

    require(
        len(center) == 3
        and
        all(
            math.isfinite(
                value
            )
            for value in center
        ),
        "Invalid current box center.",
    )

    payload = {
        "center_xyz_H0_m":
            center,

        "length_m":
            float(
                geometry.length_m
            ),

        "width_m":
            float(
                geometry.width_m
            ),

        "height_m":
            float(
                geometry.height_m
            ),

        "yaw_rad":
            float(
                geometry.yaw_rad
            ),
    }

    require(
        all(
            math.isfinite(
                float(value)
            )
            for key, value
            in payload.items()
            if key
            !=
            "center_xyz_H0_m"
        ),
        "Non-finite current box geometry.",
    )

    return payload


def run_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6
                / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            S6
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        process.stdout,
    )

    return {
        "returncode":
            int(
                process.returncode
            ),

        "tests":
            (
                int(
                    match.group(1)
                )
                if match
                else None
            ),

        "tail":
            "\n".join(
                process.stdout
                .splitlines()[
                    -30:
                ]
            ),
    }


try:
    print("=" * 76)
    print(
        "BLOCK 6.6 PART2C/1 — EXACT CURRENT-HEADLAMP "
        "ACTOR ELIGIBILITY JOIN"
    )
    print("=" * 76)

    # ========================================================
    # A. Immutable upstream seal
    # ========================================================

    print()
    print(
        "===== A. IMMUTABLE UPSTREAM SEAL ====="
    )

    for path, expected, label in (
        (
            COHORT,
            EXPECTED[
                "cohort"
            ],
            "development cohort",
        ),
        (
            PART1A_REPORT,
            EXPECTED[
                "part1a"
            ],
            "Part1A headlamp census",
        ),
        (
            PART1B_REPORT,
            EXPECTED[
                "part1b"
            ],
            "Part1B cohort freeze",
        ),
        (
            PART2B_REPORT,
            EXPECTED[
                "part2b"
            ],
            "Part2B association report",
        ),
        (
            POSTERIOR,
            EXPECTED[
                "posterior"
            ],
            "identity-safe posterior",
        ),
        (
            MATCHES,
            EXPECTED[
                "matches"
            ],
            "association matches",
        ),
        (
            FALLBACKS,
            EXPECTED[
                "fallbacks"
            ],
            "reactive fallbacks",
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
            actual
            ==
            expected,
            (
                f"{label} SHA changed: "
                f"{actual}"
            ),
        )

        print(
            f"{label:28s} = EXACT PASS"
        )

    part2b = json.loads(
        PART2B_REPORT.read_text(
            encoding="utf-8"
        )
    )

    require(
        part2b.get(
            "status"
        )
        ==
        (
            "PASS_BLOCK66_PART2B_"
            "COHORT_POSTERIOR_ASSOCIATION_MATERIALIZED"
        ),
        (
            "Part2B report is not PASS."
        ),
    )

    require(
        part2b[
            "regression"
        ][
            "status"
        ]
        ==
        "PASS",
        (
            "Part2B regression not PASS."
        ),
    )

    require(
        int(
            part2b[
                "regression"
            ][
                "passed"
            ]
        )
        ==
        145,
        (
            "Part2B regression count changed."
        ),
    )

    print(
        "Part2B regression = EXACT PASS | 145"
    )

    # ========================================================
    # B. Frozen source artifacts
    # ========================================================

    print()
    print(
        "===== B. FROZEN JOIN INPUT CENSUS ====="
    )

    cohort = read_jsonl(
        COHORT
    )

    posterior = read_jsonl(
        POSTERIOR
    )

    matches = read_jsonl(
        MATCHES
    )

    fallbacks = read_jsonl(
        FALLBACKS
    )

    require(
        len(
            cohort
        )
        ==
        120,
        "Cohort N changed.",
    )

    require(
        len(
            posterior
        )
        ==
        6925,
        (
            "Posterior count changed."
        ),
    )

    require(
        len(
            matches
        )
        ==
        3937,
        (
            "Association match count changed."
        ),
    )

    require(
        len(
            fallbacks
        )
        ==
        1034,
        (
            "Reactive fallback count changed."
        ),
    )

    posterior_index = {}

    for record in posterior:
        key = (
            str(
                record[
                    "scenario_id"
                ]
            ),
            str(
                record[
                    "prediction_id"
                ]
            ),
        )

        require(
            key
            not in
            posterior_index,
            (
                "Duplicate scenario/prediction "
                f"posterior key: {key}"
            ),
        )

        posterior_index[
            key
        ] = record

    matches_by_scene = defaultdict(
        list
    )

    fallbacks_by_scene = defaultdict(
        list
    )

    for record in matches:
        matches_by_scene[
            str(
                record[
                    "scenario_id"
                ]
            )
        ].append(
            record
        )

    for record in fallbacks:
        fallbacks_by_scene[
            str(
                record[
                    "scenario_id"
                ]
            )
        ].append(
            record
        )

    print(
        "posterior records    = 6925"
    )

    print(
        "association matches  = 3937"
    )

    print(
        "reactive fallbacks   = 1034"
    )

    print(
        "new model forward    = NO"
    )

    print(
        "new MC sampling      = NO"
    )

    # ========================================================
    # C. Exact current-box geometry join
    # ========================================================

    print()
    print(
        "===== C. EXACT PART1A HEADLAMP GEOMETRY REPLAY ====="
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

    from iscai_stage6.adb.geometry import (
        box_corners_headlamp,
    )

    all_eligible = []
    eligible_matches = []
    eligible_fallbacks = []
    scene_summaries = []

    total_boxes_by_class = (
        empty_class_counter()
    )

    headlamp_by_class = (
        empty_class_counter()
    )

    matched_headlamp_by_class = (
        empty_class_counter()
    )

    fallback_headlamp_by_class = (
        empty_class_counter()
    )

    scenes_with_headlamp_class = (
        Counter()
    )

    scenes_with_matched_class = (
        Counter()
    )

    scenes_with_fallback_class = (
        Counter()
    )

    for cohort_index, row in enumerate(
        cohort
    ):
        scenario_id = str(
            row[
                "scenario_id"
            ]
        )

        scenario = (
            read_training_scenario(
                row
            )
        )

        require(
            str(
                scenario.scenario_id
            )
            ==
            scenario_id,
            (
                "Random-access scenario "
                "ID mismatch."
            ),
        )

        adapted = (
            adapt_causal_womd_scenario(
                scenario
            )
        )

        boxes = tuple(
            build_causal_adb_actor_boxes(
                scenario=scenario,
                adapted=adapted,
            )
        )

        scene_matches = (
            matches_by_scene[
                scenario_id
            ]
        )

        scene_fallbacks = (
            fallbacks_by_scene[
                scenario_id
            ]
        )

        require(
            len(
                scene_matches
            )
            +
            len(
                scene_fallbacks
            )
            ==
            len(
                boxes
            ),
            (
                "Part2B match/fallback "
                "partition does not equal "
                f"fresh box count in {scenario_id}."
            ),
        )

        route_by_box = {}

        for record in scene_matches:
            box_index = int(
                record[
                    "actor_box_index"
                ]
            )

            require(
                box_index
                not in
                route_by_box,
                (
                    "Duplicate box routing "
                    f"for {scenario_id}:{box_index}"
                ),
            )

            route_by_box[
                box_index
            ] = (
                "PREDICTIVE_MATCH",
                record,
            )

        for record in scene_fallbacks:
            box_index = int(
                record[
                    "actor_box_index"
                ]
            )

            require(
                box_index
                not in
                route_by_box,
                (
                    "Duplicate box routing "
                    f"for {scenario_id}:{box_index}"
                ),
            )

            route_by_box[
                box_index
            ] = (
                "ORIGINAL_REACTIVE_ADB",
                record,
            )

        require(
            set(
                route_by_box
            )
            ==
            set(
                range(
                    len(
                        boxes
                    )
                )
            ),
            (
                "Part2B routing does not "
                "cover every fresh causal box."
            ),
        )

        recomputed_relevant = (
            empty_class_counter()
        )

        recomputed_total = (
            empty_class_counter()
        )

        scene_matched = (
            empty_class_counter()
        )

        scene_fallback = (
            empty_class_counter()
        )

        for box_index, actor_box in enumerate(
            boxes
        ):
            object_type = str(
                actor_box.object_type
            )

            require(
                object_type
                in
                CORE_CLASSES,
                (
                    "Unexpected core-class box "
                    f"type: {object_type}"
                ),
            )

            recomputed_total[
                object_type
            ] += 1

            total_boxes_by_class[
                object_type
            ] += 1

            route, route_record = (
                route_by_box[
                    box_index
                ]
            )

            require(
                str(
                    route_record[
                        "object_type"
                    ]
                )
                ==
                object_type,
                (
                    "Fresh box class differs "
                    "from frozen Part2B route."
                ),
            )

            relevant = (
                box_intersects_forward_sector(
                    actor_box.box,
                    box_corners_headlamp=(
                        box_corners_headlamp
                    ),
                )
            )

            if not relevant:
                continue

            recomputed_relevant[
                object_type
            ] += 1

            headlamp_by_class[
                object_type
            ] += 1

            base = {
                "cohort_index":
                    int(
                        cohort_index
                    ),

                "scenario_id":
                    scenario_id,

                "actor_box_index":
                    int(
                        box_index
                    ),

                "object_type":
                    object_type,

                "current_headlamp_eligible":
                    True,

                "eligibility_time":
                    "CURRENT_CAUSAL_ANCHOR",

                "eligibility_geometry":
                    (
                        "continuous_full_box_ground_"
                        "footprint_finite_sector"
                    ),

                "theta_domain_deg":
                    [
                        -25.0,
                        25.0,
                    ],

                "maximum_range_m":
                    150.0,

                "current_box_H0":
                    current_box_payload(
                        actor_box
                    ),

                "route":
                    route,

                "future_used_for_eligibility":
                    False,

                "centroid_only":
                    False,

                "tracks_to_predict_used":
                    False,

                "objects_of_interest_used":
                    False,
            }

            if route == "PREDICTIVE_MATCH":
                prediction_id = str(
                    route_record[
                        "prediction_id"
                    ]
                )

                posterior_key = (
                    scenario_id,
                    prediction_id,
                )

                require(
                    posterior_key
                    in
                    posterior_index,
                    (
                        "Headlamp-eligible predictive "
                        "match has no frozen posterior."
                    ),
                )

                base.update({
                    "prediction_id":
                        prediction_id,

                    "association_anchor_distance_m":
                        float(
                            route_record[
                                "association_anchor_distance_m"
                            ]
                        ),

                    "posterior_key":
                        {
                            "scenario_id":
                                scenario_id,

                            "prediction_id":
                                prediction_id,
                        },

                    "posterior_sha_source":
                        EXPECTED[
                            "posterior"
                        ],
                })

                eligible_matches.append(
                    dict(
                        base
                    )
                )

                matched_headlamp_by_class[
                    object_type
                ] += 1

                scene_matched[
                    object_type
                ] += 1

            else:
                base.update({
                    "fallback":
                        "ORIGINAL_REACTIVE_ADB",
                })

                eligible_fallbacks.append(
                    dict(
                        base
                    )
                )

                fallback_headlamp_by_class[
                    object_type
                ] += 1

                scene_fallback[
                    object_type
                ] += 1

            all_eligible.append(
                base
            )

        frozen_relevant = {
            name:
                int(
                    row[
                        "headlamp_relevant_counts"
                    ][
                        name
                    ]
                )
            for name in CORE_CLASSES
        }

        require(
            recomputed_relevant
            ==
            frozen_relevant,
            (
                "Exact Part1A current-headlamp "
                "replay mismatch in scenario "
                f"{scenario_id}:\n"
                f"recomputed={recomputed_relevant}\n"
                f"frozen={frozen_relevant}"
            ),
        )

        for object_type in CORE_CLASSES:
            if (
                recomputed_relevant[
                    object_type
                ]
                >
                0
            ):
                scenes_with_headlamp_class[
                    object_type
                ] += 1

            if (
                scene_matched[
                    object_type
                ]
                >
                0
            ):
                scenes_with_matched_class[
                    object_type
                ] += 1

            if (
                scene_fallback[
                    object_type
                ]
                >
                0
            ):
                scenes_with_fallback_class[
                    object_type
                ] += 1

        scene_summaries.append({
            "cohort_index":
                int(
                    cohort_index
                ),

            "scenario_id":
                scenario_id,

            "causal_ADB_boxes_by_class":
                recomputed_total,

            "frozen_Part1A_headlamp_relevant_by_class":
                frozen_relevant,

            "recomputed_headlamp_relevant_by_class":
                recomputed_relevant,

            "eligible_predictive_matches_by_class":
                scene_matched,

            "eligible_reactive_fallbacks_by_class":
                scene_fallback,

            "Part1A_exact_parity":
                True,
        })

        if (
            (cohort_index + 1)
            %
            10
            ==
            0
            or
            cohort_index == 0
        ):
            print(
                f"[{cohort_index+1:03d}/120] "
                f"{scenario_id} | "
                f"eligible={sum(recomputed_relevant.values()):3d} | "
                f"predictive={sum(scene_matched.values()):3d} | "
                f"fallback={sum(scene_fallback.values()):3d}"
            )

    print()
    print(
        "Part1A per-scenario geometry parity = PASS | 120/120"
    )

    # ========================================================
    # D. Aggregate exact parity
    # ========================================================

    print()
    print(
        "===== D. AGGREGATE HEADLAMP ELIGIBILITY CENSUS ====="
    )

    require(
        headlamp_by_class
        ==
        EXPECTED_HEADLAMP_ACTORS,
        (
            "Aggregate headlamp actor census "
            "does not reproduce frozen Part1B."
        ),
    )

    actual_scene_support = {
        name:
            int(
                scenes_with_headlamp_class[
                    name
                ]
            )
        for name in CORE_CLASSES
    }

    require(
        actual_scene_support
        ==
        EXPECTED_HEADLAMP_SCENES,
        (
            "Aggregate headlamp scene census "
            "does not reproduce frozen Part1B."
        ),
    )

    for object_type in CORE_CLASSES:
        require(
            matched_headlamp_by_class[
                object_type
            ]
            +
            fallback_headlamp_by_class[
                object_type
            ]
            ==
            headlamp_by_class[
                object_type
            ],
            (
                "Eligible predictive/fallback "
                f"partition mismatch for {object_type}."
            ),
        )

    require(
        len(
            all_eligible
        )
        ==
        sum(
            EXPECTED_HEADLAMP_ACTORS.values()
        )
        ==
        1124,
        (
            "Total headlamp eligible actor "
            "count must equal 1124."
        ),
    )

    require(
        len(
            eligible_matches
        )
        +
        len(
            eligible_fallbacks
        )
        ==
        1124,
        (
            "Eligible predictive/fallback "
            "global partition mismatch."
        ),
    )

    print(
        "headlamp eligible actors by class =",
        headlamp_by_class,
    )

    print(
        "eligible predictive matches by class =",
        matched_headlamp_by_class,
    )

    print(
        "eligible reactive fallbacks by class =",
        fallback_headlamp_by_class,
    )

    print(
        "headlamp scenes by class =",
        actual_scene_support,
    )

    print(
        "scenes with predictive match by class =",
        {
            name:
                int(
                    scenes_with_matched_class[
                        name
                    ]
                )
            for name in CORE_CLASSES
        },
    )

    print(
        "scenes with reactive fallback by class =",
        {
            name:
                int(
                    scenes_with_fallback_class[
                        name
                    ]
                )
            for name in CORE_CLASSES
        },
    )

    all_classes_predictive = all(
        matched_headlamp_by_class[
            name
        ]
        >
        0
        for name in CORE_CLASSES
    )

    require(
        all_classes_predictive,
        (
            "At least one core class has "
            "zero headlamp-eligible "
            "predictive matches."
        ),
    )

    print(
        "all core classes have eligible predictive support = PASS"
    )

    # ========================================================
    # E. Materialize exact join artifacts
    # ========================================================

    print()
    print(
        "===== E. MATERIALIZE HEADLAMP JOIN ====="
    )

    eligibility_sha = (
        write_jsonl_exact_or_create(
            ELIGIBILITY,
            all_eligible,
        )
    )

    eligible_match_sha = (
        write_jsonl_exact_or_create(
            ELIGIBLE_MATCHES,
            eligible_matches,
        )
    )

    eligible_fallback_sha = (
        write_jsonl_exact_or_create(
            ELIGIBLE_FALLBACKS,
            eligible_fallbacks,
        )
    )

    scene_summary_sha = (
        write_jsonl_exact_or_create(
            SCENE_SUMMARY,
            scene_summaries,
        )
    )

    print(
        "eligibility manifest =",
        ELIGIBILITY,
    )

    print(
        "eligibility SHA256 =",
        eligibility_sha,
    )

    print(
        "eligible predictive matches =",
        ELIGIBLE_MATCHES,
    )

    print(
        "eligible predictive SHA256 =",
        eligible_match_sha,
    )

    print(
        "eligible reactive fallbacks =",
        ELIGIBLE_FALLBACKS,
    )

    print(
        "eligible fallback SHA256 =",
        eligible_fallback_sha,
    )

    # ========================================================
    # F. Fresh Stage6 regression
    # ========================================================

    print()
    print(
        "===== F. FRESH STAGE6 REGRESSION ====="
    )

    regression = run_regression()

    print(
        regression[
            "tail"
        ]
    )

    require(
        regression[
            "returncode"
        ]
        ==
        0,
        (
            "Fresh Stage6 regression failed."
        ),
    )

    require(
        regression[
            "tests"
        ]
        ==
        145,
        (
            "Expected exactly 145 Stage6 "
            f"tests, got {regression['tests']}."
        ),
    )

    print(
        "fresh regression = PASS | 145"
    )

    # ========================================================
    # G. Report
    # ========================================================

    print()
    print(
        "===== G. BLOCK6.6 PART2C/1 REPORT ====="
    )

    coverage_fraction_by_class = {
        name:
            float(
                matched_headlamp_by_class[
                    name
                ]
                /
                headlamp_by_class[
                    name
                ]
            )
        for name in CORE_CLASSES
    }

    report_payload = {
        "stage":
            6,

        "block":
            "6.6-Part2C/1",

        "status":
            (
                "PASS_EXACT_CURRENT_HEADLAMP_"
                "ACTOR_ELIGIBILITY_JOIN"
            ),

        "source": {
            "cohort_N":
                120,

            "Part1A_geometry":
                (
                    "continuous_current_full_box_"
                    "ground_footprint_intersection_"
                    "with_finite_forward_sector"
                ),

            "theta_deg":
                [
                    -25.0,
                    25.0,
                ],

            "maximum_range_m":
                150.0,

            "Part1A_per_scenario_exact_parity":
                True,

            "scenarios_verified":
                120,
        },

        "headlamp_eligible": {
            "actor_count":
                1124,

            "by_class":
                headlamp_by_class,

            "scene_support_by_class":
                actual_scene_support,
        },

        "predictive_matches": {
            "count":
                len(
                    eligible_matches
                ),

            "by_class":
                matched_headlamp_by_class,

            "scene_support_by_class":
                {
                    name:
                        int(
                            scenes_with_matched_class[
                                name
                            ]
                        )
                    for name in CORE_CLASSES
                },

            "coverage_fraction_of_eligible_by_class":
                coverage_fraction_by_class,

            "all_core_classes_supported":
                True,
        },

        "original_reactive_fallback": {
            "count":
                len(
                    eligible_fallbacks
                ),

            "by_class":
                fallback_headlamp_by_class,

            "scene_support_by_class":
                {
                    name:
                        int(
                            scenes_with_fallback_class[
                                name
                            ]
                        )
                    for name in CORE_CLASSES
                },
        },

        "association_boundary": {
            "association_recomputed":
                False,

            "association_threshold_added":
                False,

            "class_gate_added":
                False,

            "perfect_ID_used":
                False,

            "current_headlamp_eligibility_applied_after_frozen_association":
                True,
        },

        "artifacts": {
            "all_headlamp_eligible": {
                "path":
                    str(
                        ELIGIBILITY
                    ),

                "sha256":
                    eligibility_sha,
            },

            "eligible_predictive_matches": {
                "path":
                    str(
                        ELIGIBLE_MATCHES
                    ),

                "sha256":
                    eligible_match_sha,
            },

            "eligible_reactive_fallbacks": {
                "path":
                    str(
                        ELIGIBLE_FALLBACKS
                    ),

                "sha256":
                    eligible_fallback_sha,
            },

            "scene_summary": {
                "path":
                    str(
                        SCENE_SUMMARY
                    ),

                "sha256":
                    scene_summary_sha,
            },
        },

        "scientific_execution": {
            "new_model_forward":
                False,

            "new_MC_sampling":
                False,

            "P_occ_materialization":
                False,

            "gamma_tuning":
                False,

            "class_policy_parameter_selection":
                False,

            "training":
                False,

            "recalibration":
                False,

            "formal_evaluation":
                False,

            "formal_outcomes_read":
                False,
        },

        "next":
            (
                "BLOCK6.6_PART2C2_N8192_P_OCC_"
                "ONLY_FOR_HEADLAMP_ELIGIBLE_"
                "PREDICTIVE_MATCHES"
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
    # H. Freeze manifest + handoff
    # ========================================================

    print()
    print(
        "===== H. FREEZE MANIFEST + PART2C/2 HANDOFF ====="
    )

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.6-Part2C/1",

        "status":
            (
                "HEADLAMP_ACTOR_ELIGIBILITY_"
                "JOIN_FROZEN"
            ),

        "immutable_inputs": {
            str(
                COHORT
            ):
                EXPECTED[
                    "cohort"
                ],

            str(
                PART1A_REPORT
            ):
                EXPECTED[
                    "part1a"
                ],

            str(
                PART2B_REPORT
            ):
                EXPECTED[
                    "part2b"
                ],

            str(
                POSTERIOR
            ):
                EXPECTED[
                    "posterior"
                ],

            str(
                MATCHES
            ):
                EXPECTED[
                    "matches"
                ],

            str(
                FALLBACKS
            ):
                EXPECTED[
                    "fallbacks"
                ],
        },

        "eligible_predictive": {
            "path":
                str(
                    ELIGIBLE_MATCHES
                ),

            "sha256":
                eligible_match_sha,

            "count":
                len(
                    eligible_matches
                ),

            "by_class":
                matched_headlamp_by_class,
        },

        "eligible_reactive_fallback": {
            "path":
                str(
                    ELIGIBLE_FALLBACKS
                ),

            "sha256":
                eligible_fallback_sha,

            "count":
                len(
                    eligible_fallbacks
                ),

            "by_class":
                fallback_headlamp_by_class,
        },

        "report": {
            "path":
                str(
                    REPORT
                ),

            "sha256":
                report_sha,
        },

        "scientific_lock": {
            "association":
                "UNCHANGED",

            "headlamp_eligibility":
                "FROZEN",

            "N_MC":
                8192,

            "N_MC_executed":
                False,

            "class_policy":
                "NOT_FROZEN",

            "formal_tuning":
                "FORBIDDEN",
        },
    }

    freeze_sha = (
        write_json_exact_or_create(
            FREEZE,
            freeze_payload,
        )
    )

    handoff_payload = {
        "stage":
            6,

        "from":
            "6.6-Part2C/1",

        "to":
            "6.6-Part2C/2",

        "status":
            (
                "PASS_READY_FOR_ELIGIBLE_"
                "N8192_P_OCC_MATERIALIZATION"
            ),

        "eligible_predictive_matches": {
            "path":
                str(
                    ELIGIBLE_MATCHES
                ),

            "sha256":
                eligible_match_sha,

            "count":
                len(
                    eligible_matches
                ),

            "by_class":
                matched_headlamp_by_class,
        },

        "posterior": {
            "path":
                str(
                    POSTERIOR
                ),

            "sha256":
                EXPECTED[
                    "posterior"
                ],

            "family":
                "calibrated_Gaussian_GRU",
        },

        "frozen_MC": {
            "sample_count":
                8192,

            "seed":
                20260821,

            "grid_shape":
                [
                    501,
                    301,
                ],

            "sampler":
                (
                    "Stage5_exact_product_of_"
                    "frozen_per_horizon_Gaussian_marginals"
                ),
        },

        "required_next": [
            (
                "materialize_N8192_full_box_"
                "P_occ_only_for_eligible_predictive_matches"
            ),
            (
                "preserve_current_box_dimensions_"
                "and_samplewise_predicted_tangent_heading"
            ),
            (
                "retain_original_reactive_fallback_"
                "for_eligible_unmatched_boxes"
            ),
            (
                "record_classwise_P_occ_support_"
                "before_class_policy_preregistration"
            ),
        ],

        "forbidden_next": [
            "formal_outcome_read",
            "gamma_reoptimization",
            "perfect_ID_association",
            "future_GT_controller_input",
            "class_policy_numeric_tuning_before_P_occ_evidence",
        ],

        "upstream_freeze": {
            "eligibility_freeze_sha256":
                freeze_sha,

            "report_sha256":
                report_sha,
        },
    }

    handoff_sha = (
        write_json_exact_or_create(
            HANDOFF,
            handoff_payload,
        )
    )

    print(
        "freeze manifest =",
        FREEZE,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    # ========================================================
    # I. Final
    # ========================================================

    print()
    print("=" * 76)
    print(
        "BLOCK 6.6 PART2C/1 — FINAL"
    )
    print("=" * 76)

    print(
        "STATUS = "
        "PASS_EXACT_CURRENT_HEADLAMP_ACTOR_ELIGIBILITY_JOIN"
    )

    print(
        "Part1A per-scenario parity  = PASS | 120/120"
    )

    print(
        "headlamp eligible total     = 1124"
    )

    print(
        "headlamp eligible by class  =",
        headlamp_by_class,
    )

    print(
        "eligible predictive by class=",
        matched_headlamp_by_class,
    )

    print(
        "eligible fallback by class  =",
        fallback_headlamp_by_class,
    )

    print(
        "predictive coverage/class   =",
        coverage_fraction_by_class,
    )

    print(
        "association recomputed      = NO"
    )

    print(
        "new association threshold   = NO"
    )

    print(
        "class gate                  = NO"
    )

    print(
        "new model forward           = NO"
    )

    print(
        "N_MC=8192 executed          = NO"
    )

    print(
        "P_occ materialized          = NO"
    )

    print(
        "class policy frozen         = NO"
    )

    print(
        "formal outcomes read        = NO"
    )

    print(
        "Stage6 regression           = PASS | 145"
    )

    print(
        "eligible predictive SHA256 =",
        eligible_match_sha,
    )

    print(
        "eligible fallback SHA256   =",
        eligible_fallback_sha,
    )

    print(
        "report SHA256              =",
        report_sha,
    )

    print(
        "freeze SHA256              =",
        freeze_sha,
    )

    print(
        "handoff SHA256             =",
        handoff_sha,
    )

    print(
        "NEXT = BLOCK6.6 PART2C/2 "
        "N8192 P_OCC FOR ELIGIBLE PREDICTIVE MATCHES ONLY"
    )

    print(
        "terminal remains open = YES"
    )

    print("=" * 76)


except BaseException as exc:
    print()
    print("=" * 76)
    print(
        "BLOCK6.6 PART2C/1 — CONTROLLED BLOCK"
    )
    print("=" * 76)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=20
    )

    print()
    print(
        "association modified   = NO"
    )

    print(
        "new model forward      = NO"
    )

    print(
        "N_MC=8192 executed     = NO"
    )

    print(
        "class policy frozen    = NO"
    )

    print(
        "formal outcomes read   = NO"
    )

    print(
        "terminal remains open  = YES"
    )

# Deliberately no sys.exit().

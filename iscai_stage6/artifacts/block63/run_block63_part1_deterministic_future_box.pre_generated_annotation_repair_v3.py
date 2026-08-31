from __future__ import annotations

from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")

S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

BINDING = (
    S6
    / "configs/block63_deterministic_predictive_binding.json"
)

BINDING_FREEZE = (
    S6
    / "reports/block63_binding_freeze.json"
)

BLOCK62 = (
    S6
    / "reports/block62_closure.json"
)

GEOMETRY_MODULE = (
    S6
    / "src/iscai_stage6/adb/geometry.py"
)

WOMD_GEOMETRY_MODULE = (
    S6
    / "src/iscai_stage6/adb/womd_geometry.py"
)

STAGE5_HEADING_MODULE = (
    S5
    / "src/iscai_stage5/angular_monte_carlo.py"
)

ASSOCIATION_MODULE = (
    S6
    / "src/iscai_stage6/adb/deterministic_association.py"
)

PREDICTIVE_MODULE = (
    S6
    / "src/iscai_stage6/adb/deterministic_predictive.py"
)

TEST_FILE = (
    S6
    / "tests/test_block63_part1_deterministic_future_box.py"
)

REPORT = (
    S6
    / "reports/block63_part1_deterministic_future_box.json"
)

REGRESSION_LOG = (
    S6
    / "artifacts/block63/"
      "block63_part1_stage6_regression.log"
)

EXPECTED = {
    "binding":
        (
            "03903018109e7d5037643381ff71141a5"
            "c57f003b8bdc6b9be13bb4304468f67"
        ),

    "binding_freeze":
        (
            "6e73e58eb4a28390360e965b33582b6b4"
            "30dff149f97696494e9d504d1e0d84a"
        ),

    "block62":
        (
            "23b299b4dc6a123ce64a6ab1350b3ed9"
            "cb149c8fed176b37e9715d9935704d92"
        ),
}

MIN_FREE_GIB = 250.0

PREDICTED_STATE_SOURCE = (
    "predicted_sample"
)

PREDICTED_ORIENTATION_SOURCE = (
    "predicted_tangent"
)

CREATED_PATHS = []


# ============================================================
# Environment bootstrap
# ============================================================

def bootstrap():

    roots = tuple(
        ROOT
        / f"iscai_stage{i}"
        / "src"
        for i in range(7)
    )

    missing = [
        str(path)
        for path in roots
        if not path.is_dir()
    ]

    if missing:
        raise RuntimeError(
            "Missing project source root(s): "
            + ", ".join(missing)
        )

    for path in reversed(roots):

        value = str(path)

        while value in sys.path:
            sys.path.remove(value)

        sys.path.insert(
            0,
            value,
        )

    previous = [
        item
        for item in os.environ.get(
            "PYTHONPATH",
            "",
        ).split(
            os.pathsep
        )
        if item
    ]

    merged = []
    seen = set()

    for item in (
        [
            str(path)
            for path in roots
        ]
        +
        previous
    ):

        if item in seen:
            continue

        seen.add(item)
        merged.append(item)

    os.environ[
        "PYTHONPATH"
    ] = os.pathsep.join(
        merged
    )


# ============================================================
# Generic helpers
# ============================================================

def require(
    condition,
    message,
):

    if not bool(condition):
        raise RuntimeError(
            message
        )


def sha256_file(
    path: Path,
) -> str:

    digest = sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:

            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def canonical_bytes(
    payload,
) -> bytes:

    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n"
    ).encode(
        "utf-8"
    )


def atomic_json(
    path: Path,
    payload,
):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(
            payload
        )
    )

    os.replace(
        tmp,
        path,
    )


def write_new_or_exact(
    path: Path,
    text: str,
):

    desired = text.encode(
        "utf-8"
    )

    if path.exists():

        require(
            path.read_bytes()
            ==
            desired,
            (
                "Existing Block6.3 source differs "
                f"from deterministic candidate: {path}"
            ),
        )

        return "ALREADY_EXACT"

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        desired
    )

    os.replace(
        tmp,
        path,
    )

    CREATED_PATHS.append(
        path
    )

    return "WRITTEN"


def tail_text(
    text: str,
    lines: int = 120,
) -> str:

    rows = text.splitlines()

    return "\n".join(
        rows[-lines:]
    )


def rollback_created():

    removed = []

    for path in reversed(
        CREATED_PATHS
    ):

        try:

            if path.exists():
                path.unlink()
                removed.append(
                    str(path)
                )

        except BaseException:
            pass

    return removed


# ============================================================
# Main
# ============================================================

def main():

    bootstrap()

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 PART 1/2"
    )
    print(
        "DETERMINISTIC FUTURE FULL-BOX PROJECTION"
    )
    print(
        "============================================================"
    )

    protected = (
        BINDING,
        BINDING_FREEZE,
        BLOCK62,
        GEOMETRY_MODULE,
        WOMD_GEOMETRY_MODULE,
        STAGE5_HEADING_MODULE,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing prerequisite(s): "
            + ", ".join(missing)
        ),
    )

    protected_before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Frozen binding seal
    # ========================================================

    print()
    print(
        "===== A. FROZEN BINDING SEAL ====="
    )

    checks = (
        (
            "binding",
            BINDING,
        ),
        (
            "binding_freeze",
            BINDING_FREEZE,
        ),
        (
            "block62",
            BLOCK62,
        ),
    )

    for name, path in checks:

        actual = sha256_file(
            path
        )

        require(
            actual
            ==
            EXPECTED[name],
            (
                f"{name} SHA mismatch.\n"
                f"expected={EXPECTED[name]}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{name:18s} = EXACT PASS"
        )

    binding = json.loads(
        BINDING.read_text(
            encoding="utf-8"
        )
    )

    freeze = json.loads(
        BINDING_FREEZE.read_text(
            encoding="utf-8"
        )
    )

    require(
        binding.get("status")
        ==
        "FROZEN_READY_FOR_BLOCK63_PART1",
        (
            "Block6.3 binding is not frozen-ready."
        ),
    )

    require(
        freeze.get("status")
        ==
        "PASS_READY_FOR_BLOCK63_PART1",
        (
            "Block6.3 freeze report is not ready."
        ),
    )

    association_binding = (
        binding[
            "association_policy"
        ]
    )

    trajectory_binding = (
        binding[
            "trajectory_policy"
        ]
    )

    future_box_binding = (
        binding[
            "future_box_policy"
        ]
    )

    require(
        association_binding[
            "algorithm"
        ][
            "hard_distance_threshold_m"
        ]
        is None,
        (
            "Frozen association unexpectedly "
            "contains a distance threshold."
        ),
    )

    require(
        association_binding[
            "algorithm"
        ][
            "forced_assignment"
        ]
        is False,
        "Forced association is forbidden.",
    )

    require(
        trajectory_binding[
            "deterministic_ADB_signal"
        ]
        ==
        "Gaussian_mean_only",
        (
            "Deterministic trajectory binding changed."
        ),
    )

    require(
        trajectory_binding[
            "Stage4_prediction_semantics"
        ]
        ==
        "metric_H0_displacement",
        (
            "Stage4 prediction semantics changed."
        ),
    )

    require(
        trajectory_binding[
            "matched_box_center_used_to_recenter_prediction"
        ]
        is False,
        (
            "Annotation recentering is forbidden."
        ),
    )

    require(
        future_box_binding[
            "projection"
        ][
            "eight_physical_corners"
        ]
        is True,
        (
            "Full eight-corner projection "
            "is no longer required."
        ),
    )

    print(
        "association policy  = FROZEN PASS"
    )

    print(
        "shared mean policy  = FROZEN PASS"
    )

    print(
        "future box policy   = FROZEN PASS"
    )

    # ========================================================
    # B. Runtime API/provenance preflight BEFORE writing code
    # ========================================================

    print()
    print(
        "===== B. RUNTIME API / PROVENANCE PREFLIGHT ====="
    )

    import numpy as np

    from iscai_stage5.angular_monte_carlo import (
        resolve_samplewise_headings,
    )

    from iscai_stage6.adb.geometry import (
        Box3D,
        project_box_to_headlamp,
    )

    heading_signature = (
        inspect.signature(
            resolve_samplewise_headings
        )
    )

    required_heading_parameters = (
        "trajectory_samples_h0_m",
        "current_position_h0_m",
        "current_heading_h0_rad",
        "low_speed_threshold_mps",
    )

    require(
        tuple(
            heading_signature.parameters
        )
        ==
        required_heading_parameters,
        (
            "Frozen Stage5 heading API changed: "
            f"{heading_signature}"
        ),
    )

    # Test-only geometry; no scientific parameter is introduced.
    probe_centers = np.asarray(
        [
            [
                [25.0, 0.0, 0.0],
                [26.0, 0.0, 0.0],
                [27.0, 0.0, 0.0],
                [28.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )

    probe_headings, probe_fallback = (
        resolve_samplewise_headings(
            trajectory_samples_h0_m=(
                probe_centers
            ),
            current_position_h0_m=(
                24.0,
                0.0,
                0.0,
            ),
            current_heading_h0_rad=0.0,
        )
    )

    require(
        probe_headings.shape
        ==
        (
            1,
            4,
        ),
        (
            "Unexpected Stage5 heading output shape."
        ),
    )

    require(
        probe_fallback.shape
        ==
        (
            1,
            4,
        ),
        (
            "Unexpected Stage5 heading fallback shape."
        ),
    )

    probe_box = Box3D(
        center_xyz=(
            25.0,
            0.0,
            0.0,
        ),
        length_m=4.0,
        width_m=2.0,
        height_m=1.5,
        yaw_rad=float(
            probe_headings[
                0,
                0,
            ]
        ),
    )

    # This is an important controller-path gate. If the frozen
    # provenance validator rejects these explicit predictive
    # labels, the script blocks before scientific files are
    # written instead of silently bypassing controller checks.
    probe_projection = (
        project_box_to_headlamp(
            probe_box,
            state_source=(
                PREDICTED_STATE_SOURCE
            ),
            orientation_source=(
                PREDICTED_ORIENTATION_SOURCE
            ),
            controller_path=True,
        )
    )

    require(
        probe_projection.corners.xyz.shape
        ==
        (
            8,
            3,
        ),
        (
            "Predictive controller provenance "
            "did not produce eight physical corners."
        ),
    )

    print(
        "Stage5 heading API      = EXACT PASS"
    )

    print(
        "predictive state source =",
        PREDICTED_STATE_SOURCE,
        "PASS",
    )

    print(
        "predictive yaw source   =",
        PREDICTED_ORIENTATION_SOURCE,
        "PASS",
    )

    print(
        "controller_path=True    = PASS"
    )

    print(
        "probe corner count      = 8 PASS"
    )

    # ========================================================
    # C. Association module
    # ========================================================

    print()
    print(
        "===== C. CAUSAL ASSOCIATION MODULE ====="
    )

    association_source = r'''from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


Vec3 = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class PredictorAnchor:
    """
    Controller-side causal predictor identity.

    prediction_id is internal tracker/predictor metadata only.
    It is NOT compared with a WOMD actor ID during association.
    """

    prediction_id: str
    latest_position_H0_m: Vec3

    def __post_init__(
        self,
    ) -> None:

        if not str(
            self.prediction_id
        ):
            raise ValueError(
                "prediction_id cannot be empty."
            )

        values = tuple(
            float(value)
            for value
            in self.latest_position_H0_m
        )

        if len(values) != 3:
            raise ValueError(
                "latest_position_H0_m must be 3D."
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "latest_position_H0_m must be finite."
            )


@dataclass(frozen=True)
class ReciprocalAssociationMatch:
    predictor_index: int
    box_index: int

    prediction_id: str

    anchor_distance_m: float

    predictor_second_nearest_distance_m:
        float | None

    box_second_nearest_distance_m:
        float | None


@dataclass(frozen=True)
class ReciprocalAssociationResult:
    matches:
        tuple[
            ReciprocalAssociationMatch,
            ...,
        ]

    unmatched_predictor_indices:
        tuple[
            int,
            ...,
        ]

    unmatched_box_indices:
        tuple[
            int,
            ...,
        ]

    predictor_count: int
    box_count: int

    @property
    def matched_pair_count(
        self,
    ) -> int:

        return len(
            self.matches
        )

    @property
    def match_fraction_of_boxes(
        self,
    ) -> float:

        if self.box_count == 0:
            return 1.0

        return (
            self.matched_pair_count
            /
            self.box_count
        )

    @property
    def reactive_fallback_fraction(
        self,
    ) -> float:

        if self.box_count == 0:
            return 0.0

        return (
            len(
                self.unmatched_box_indices
            )
            /
            self.box_count
        )


def _finite_vec3(
    value,
    *,
    name: str,
) -> np.ndarray:

    result = np.asarray(
        value,
        dtype=np.float64,
    )

    if result.shape != (
        3,
    ):
        raise ValueError(
            f"{name} must have shape (3,)."
        )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _box_anchor(
    actor_box,
) -> np.ndarray | None:

    value = getattr(
        actor_box,
        "stage1_anchor_center_H0_m",
        None,
    )

    if value is None:
        return None

    return _finite_vec3(
        value,
        name=(
            "stage1_anchor_center_H0_m"
        ),
    )


def _unique_nearest(
    squared_distances: np.ndarray,
) -> tuple[
    int | None,
    float,
    float | None,
]:
    """
    Parameter-free unique nearest relation.

    Exact equal minima are considered ambiguous and therefore
    unmatched. No tolerance, hard distance gate, covariance
    gate or class gate is introduced.
    """

    values = np.asarray(
        squared_distances,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Nearest-neighbour vector must be 1D."
        )

    if values.size == 0:
        return (
            None,
            float("inf"),
            None,
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):
        raise ValueError(
            "Association distances must be finite."
        )

    minimum = float(
        np.min(
            values
        )
    )

    nearest = np.flatnonzero(
        values
        ==
        minimum
    )

    ordered = np.sort(
        values
    )

    second = (
        None
        if ordered.size < 2
        else math.sqrt(
            float(
                ordered[
                    1
                ]
            )
        )
    )

    if nearest.size != 1:

        return (
            None,
            math.sqrt(
                minimum
            ),
            second,
        )

    return (
        int(
            nearest[
                0
            ]
        ),
        math.sqrt(
            minimum
        ),
        second,
    )


def reciprocal_unique_nearest_anchor_association(
    predictor_anchors:
        Iterable[
            PredictorAnchor
        ],
    actor_boxes:
        Iterable,
) -> ReciprocalAssociationResult:
    """
    Frozen Block-6.3 association policy.

    Association basis:
      predictor latest causal H0 anchor
        <->
      Stage6 causal ADB-box Stage1 H0 anchor

    Pair acceptance:
      - predictor has one exact unique nearest box,
      - box has one exact unique nearest predictor,
      - relations are reciprocal.

    Deliberately absent:
      - hard distance threshold,
      - covariance gate,
      - class gate,
      - Hungarian/forced assignment,
      - prediction_id == WOMD track_id,
      - truth_id / truth_track_index,
      - tracks_to_predict,
      - future state information.

    A causal box with no finite Stage1 anchor remains unmatched
    and is therefore eligible for the frozen original-reactive
    fallback downstream.
    """

    predictors = tuple(
        predictor_anchors
    )

    boxes = tuple(
        actor_boxes
    )

    prediction_ids = [
        str(
            item.prediction_id
        )
        for item in predictors
    ]

    if (
        len(
            prediction_ids
        )
        !=
        len(
            set(
                prediction_ids
            )
        )
    ):
        raise ValueError(
            "Predictor IDs must be unique "
            "within one association call."
        )

    predictor_positions = tuple(
        _finite_vec3(
            item.latest_position_H0_m,
            name=(
                "predictor latest_position_H0_m"
            ),
        )
        for item in predictors
    )

    box_positions = tuple(
        _box_anchor(
            actor_box
        )
        for actor_box in boxes
    )

    valid_box_indices = tuple(
        index
        for index, position
        in enumerate(
            box_positions
        )
        if position is not None
    )

    if (
        not predictors
        or
        not valid_box_indices
    ):

        return ReciprocalAssociationResult(
            matches=(),

            unmatched_predictor_indices=(
                tuple(
                    range(
                        len(
                            predictors
                        )
                    )
                )
            ),

            unmatched_box_indices=(
                tuple(
                    range(
                        len(
                            boxes
                        )
                    )
                )
            ),

            predictor_count=len(
                predictors
            ),

            box_count=len(
                boxes
            ),
        )

    distance_squared = np.empty(
        (
            len(
                predictors
            ),
            len(
                valid_box_indices
            ),
        ),
        dtype=np.float64,
    )

    for predictor_index, predictor_position in enumerate(
        predictor_positions
    ):

        for local_box_index, box_index in enumerate(
            valid_box_indices
        ):

            difference = (
                predictor_position
                -
                box_positions[
                    box_index
                ]
            )

            distance_squared[
                predictor_index,
                local_box_index,
            ] = float(
                difference
                @
                difference
            )

    predictor_nearest = []
    predictor_distance = []
    predictor_second = []

    for predictor_index in range(
        len(
            predictors
        )
    ):

        (
            local_box,
            distance,
            second,
        ) = _unique_nearest(
            distance_squared[
                predictor_index,
                :,
            ]
        )

        predictor_nearest.append(
            local_box
        )

        predictor_distance.append(
            distance
        )

        predictor_second.append(
            second
        )

    box_nearest = []
    box_second = []

    for local_box_index in range(
        len(
            valid_box_indices
        )
    ):

        (
            predictor_index,
            _distance,
            second,
        ) = _unique_nearest(
            distance_squared[
                :,
                local_box_index,
            ]
        )

        box_nearest.append(
            predictor_index
        )

        box_second.append(
            second
        )

    matches = []

    matched_predictors = set()
    matched_boxes = set()

    for predictor_index, local_box_index in enumerate(
        predictor_nearest
    ):

        if local_box_index is None:
            continue

        if (
            box_nearest[
                local_box_index
            ]
            !=
            predictor_index
        ):
            continue

        box_index = (
            valid_box_indices[
                local_box_index
            ]
        )

        matches.append(
            ReciprocalAssociationMatch(
                predictor_index=(
                    predictor_index
                ),

                box_index=(
                    box_index
                ),

                prediction_id=(
                    predictors[
                        predictor_index
                    ].prediction_id
                ),

                anchor_distance_m=float(
                    predictor_distance[
                        predictor_index
                    ]
                ),

                predictor_second_nearest_distance_m=(
                    predictor_second[
                        predictor_index
                    ]
                ),

                box_second_nearest_distance_m=(
                    box_second[
                        local_box_index
                    ]
                ),
            )
        )

        matched_predictors.add(
            predictor_index
        )

        matched_boxes.add(
            box_index
        )

    return ReciprocalAssociationResult(
        matches=tuple(
            matches
        ),

        unmatched_predictor_indices=tuple(
            index
            for index in range(
                len(
                    predictors
                )
            )
            if index not in matched_predictors
        ),

        unmatched_box_indices=tuple(
            index
            for index in range(
                len(
                    boxes
                )
            )
            if index not in matched_boxes
        ),

        predictor_count=len(
            predictors
        ),

        box_count=len(
            boxes
        ),
    )
'''

    association_status = write_new_or_exact(
        ASSOCIATION_MODULE,
        association_source,
    )

    print(
        "deterministic_association.py =",
        association_status,
    )

    # ========================================================
    # D. Deterministic future full-box module
    # ========================================================

    print()
    print(
        "===== D. DETERMINISTIC FUTURE FULL-BOX MODULE ====="
    )

    predictive_source = f'''from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    resolve_samplewise_headings,
)

from .deterministic_association import (
    PredictorAnchor,
    ReciprocalAssociationResult,
    reciprocal_unique_nearest_anchor_association,
)

from .geometry import (
    Box3D,
    ProjectedBox,
    project_box_to_headlamp,
)


HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

PREDICTED_STATE_SOURCE = (
    {PREDICTED_STATE_SOURCE!r}
)

PREDICTED_ORIENTATION_SOURCE = (
    {PREDICTED_ORIENTATION_SOURCE!r}
)


Vec3 = tuple[
    float,
    float,
    float,
]


def _finite_vec3(
    value,
    *,
    name: str,
) -> Vec3:

    result = tuple(
        float(item)
        for item in value
    )

    if len(result) != 3:
        raise ValueError(
            f"{{name}} must be 3D."
        )

    if not all(
        math.isfinite(item)
        for item in result
    ):
        raise ValueError(
            f"{{name}} must be finite."
        )

    return result


@dataclass(frozen=True)
class DeterministicSharedMeanPrediction:
    """
    Deterministic Stage6 ablation of the frozen shared
    calibrated Gaussian trajectory posterior.

    mean_displacement_H0_m contains the Gaussian mean only.
    No predictive covariance enters this contract.
    """

    prediction_id: str

    latest_position_H0_m:
        Vec3

    mean_displacement_H0_m:
        tuple[
            Vec3,
            Vec3,
            Vec3,
            Vec3,
        ]

    def __post_init__(
        self,
    ) -> None:

        if not str(
            self.prediction_id
        ):
            raise ValueError(
                "prediction_id cannot be empty."
            )

        _finite_vec3(
            self.latest_position_H0_m,
            name=(
                "latest_position_H0_m"
            ),
        )

        if (
            len(
                self.mean_displacement_H0_m
            )
            !=
            len(
                HORIZONS_S
            )
        ):
            raise ValueError(
                "Shared mean must contain exactly "
                "the four frozen prediction horizons."
            )

        for index, value in enumerate(
            self.mean_displacement_H0_m
        ):

            _finite_vec3(
                value,
                name=(
                    "mean_displacement_H0_m"
                    f"[{{index}}]"
                ),
            )

    def association_anchor(
        self,
    ) -> PredictorAnchor:

        return PredictorAnchor(
            prediction_id=(
                self.prediction_id
            ),

            latest_position_H0_m=(
                self.latest_position_H0_m
            ),
        )

    def future_centers_H0_m(
        self,
    ) -> tuple[
        Vec3,
        Vec3,
        Vec3,
        Vec3,
    ]:
        """
        Frozen coordinate semantics:

          future_H0[h]
            =
          predictor_latest_causal_anchor_H0
            +
          Gaussian_mean_metric_H0_displacement[h]

        The matched WOMD/Stage6 box center is deliberately
        NOT used to recenter this trajectory.
        """

        anchor = np.asarray(
            self.latest_position_H0_m,
            dtype=np.float64,
        )

        displacement = np.asarray(
            self.mean_displacement_H0_m,
            dtype=np.float64,
        )

        result = (
            anchor[
                None,
                :
            ]
            +
            displacement
        )

        return tuple(
            tuple(
                float(value)
                for value in row
            )
            for row in result
        )


@dataclass(frozen=True)
class DeterministicFutureProjectedBox:
    horizon_s: float

    center_H0_m: Vec3

    yaw_H0_rad: float

    heading_fallback_used: bool

    box: Box3D

    projection: ProjectedBox


@dataclass(frozen=True)
class MatchedDeterministicActorForecast:
    predictor_index: int
    actor_box_index: int

    prediction_id: str
    actor_object_type: str

    association_anchor_distance_m: float

    future:
        tuple[
            DeterministicFutureProjectedBox,
            DeterministicFutureProjectedBox,
            DeterministicFutureProjectedBox,
            DeterministicFutureProjectedBox,
        ]


@dataclass(frozen=True)
class DeterministicPredictivePlan:
    association:
        ReciprocalAssociationResult

    matched_forecasts:
        tuple[
            MatchedDeterministicActorForecast,
            ...,
        ]

    reactive_fallback_box_indices:
        tuple[
            int,
            ...,
        ]

    unmatched_prediction_indices:
        tuple[
            int,
            ...,
        ]

    @property
    def predicted_box_count(
        self,
    ) -> int:

        return sum(
            len(
                item.future
            )
            for item
            in self.matched_forecasts
        )


def _validate_causal_actor_box(
    actor_box,
) -> None:

    for name in (
        "tracks_to_predict_used",
        "objects_of_interest_used",
        "future_state_used",
    ):

        if bool(
            getattr(
                actor_box,
                name,
                False,
            )
        ):

            raise RuntimeError(
                "Deterministic predictive controller "
                "received non-causal actor-box provenance: "
                f"{{name}}=True"
            )

    if getattr(
        actor_box,
        "stage1_anchor_center_H0_m",
        None,
    ) is None:

        raise RuntimeError(
            "Matched actor box has no causal Stage1 anchor."
        )

    box = getattr(
        actor_box,
        "box",
        None,
    )

    if box is None:
        raise RuntimeError(
            "Matched actor box has no Box3D geometry."
        )

    dimensions = (
        float(
            box.length_m
        ),
        float(
            box.width_m
        ),
        float(
            box.height_m
        ),
    )

    if not all(
        math.isfinite(value)
        and
        value > 0.0
        for value in dimensions
    ):
        raise ValueError(
            "Matched causal box has invalid dimensions."
        )

    if not math.isfinite(
        float(
            box.yaw_rad
        )
    ):
        raise ValueError(
            "Matched causal box has invalid current yaw."
        )


def build_deterministic_future_full_boxes(
    prediction:
        DeterministicSharedMeanPrediction,
    actor_box,
) -> tuple[
    DeterministicFutureProjectedBox,
    DeterministicFutureProjectedBox,
    DeterministicFutureProjectedBox,
    DeterministicFutureProjectedBox,
]:
    """
    Build the deterministic predictive ADB geometry for one
    associated actor.

    State:
      shared calibrated-Gaussian mean trajectory only.

    Center:
      predictor causal anchor + metric-H0 mean displacement.

    Dimensions:
      current causal l/w/h held explicit across horizons.

    Orientation:
      frozen Stage5 trajectory tangent rule, with current causal
      actor-box yaw used only as the low-speed carry-forward seed.

    Projection:
      full 3D Box3D -> all eight physical corners.

    Deliberately absent:
      predictive covariance,
      probabilistic occupancy,
      future GT,
      class-specific margin/floor,
      communication codebook.
    """

    _validate_causal_actor_box(
        actor_box
    )

    centers = np.asarray(
        prediction.future_centers_H0_m(),
        dtype=np.float64,
    )

    heading_samples = centers[
        None,
        :,
        :,
    ]

    headings, fallback = (
        resolve_samplewise_headings(
            trajectory_samples_h0_m=(
                heading_samples
            ),

            current_position_h0_m=(
                prediction.latest_position_H0_m
            ),

            current_heading_h0_rad=float(
                actor_box.box.yaw_rad
            ),
        )
    )

    if headings.shape != (
        1,
        4,
    ):
        raise RuntimeError(
            "Unexpected frozen Stage5 heading shape."
        )

    if fallback.shape != (
        1,
        4,
    ):
        raise RuntimeError(
            "Unexpected frozen Stage5 fallback shape."
        )

    result = []

    for horizon_index, horizon_s in enumerate(
        HORIZONS_S
    ):

        center = tuple(
            float(value)
            for value
            in centers[
                horizon_index
            ]
        )

        yaw = float(
            headings[
                0,
                horizon_index,
            ]
        )

        box = Box3D(
            center_xyz=center,

            length_m=float(
                actor_box.box.length_m
            ),

            width_m=float(
                actor_box.box.width_m
            ),

            height_m=float(
                actor_box.box.height_m
            ),

            yaw_rad=yaw,
        )

        projection = (
            project_box_to_headlamp(
                box,

                state_source=(
                    PREDICTED_STATE_SOURCE
                ),

                orientation_source=(
                    PREDICTED_ORIENTATION_SOURCE
                ),

                controller_path=True,
            )
        )

        if projection.corners.xyz.shape != (
            8,
            3,
        ):
            raise RuntimeError(
                "Future predictive box did not "
                "produce exactly eight physical corners."
            )

        result.append(
            DeterministicFutureProjectedBox(
                horizon_s=float(
                    horizon_s
                ),

                center_H0_m=center,

                yaw_H0_rad=yaw,

                heading_fallback_used=bool(
                    fallback[
                        0,
                        horizon_index,
                    ]
                ),

                box=box,

                projection=projection,
            )
        )

    return tuple(
        result
    )


def build_deterministic_predictive_plan(
    predictions:
        Iterable[
            DeterministicSharedMeanPrediction
        ],
    actor_boxes:
        Iterable,
) -> DeterministicPredictivePlan:
    """
    Associate causal predictor tracks to causal Stage6 boxes,
    then construct deterministic future full boxes.

    Unmatched causal ADB boxes are explicitly returned for the
    frozen original-reactive fallback. They are never silently
    removed from illumination control.
    """

    predictions = tuple(
        predictions
    )

    actor_boxes = tuple(
        actor_boxes
    )

    anchors = tuple(
        prediction.association_anchor()
        for prediction
        in predictions
    )

    association = (
        reciprocal_unique_nearest_anchor_association(
            anchors,
            actor_boxes,
        )
    )

    forecasts = []

    for match in association.matches:

        actor_box = actor_boxes[
            match.box_index
        ]

        prediction = predictions[
            match.predictor_index
        ]

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                actor_box,
            )
        )

        forecasts.append(
            MatchedDeterministicActorForecast(
                predictor_index=(
                    match.predictor_index
                ),

                actor_box_index=(
                    match.box_index
                ),

                prediction_id=(
                    prediction.prediction_id
                ),

                actor_object_type=str(
                    getattr(
                        actor_box,
                        "object_type",
                        ""
                    )
                ),

                association_anchor_distance_m=(
                    float(
                        match.anchor_distance_m
                    )
                ),

                future=future,
            )
        )

    return DeterministicPredictivePlan(
        association=(
            association
        ),

        matched_forecasts=tuple(
            forecasts
        ),

        reactive_fallback_box_indices=(
            association.unmatched_box_indices
        ),

        unmatched_prediction_indices=(
            association.unmatched_predictor_indices
        ),
    )
'''

    predictive_status = write_new_or_exact(
        PREDICTIVE_MODULE,
        predictive_source,
    )

    print(
        "deterministic_predictive.py =",
        predictive_status,
    )

    # ========================================================
    # E. Unit tests
    # ========================================================

    print()
    print(
        "===== E. BLOCK6.3 PART1 UNIT TESTS ====="
    )

    test_source = r'''from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from iscai_stage5.angular_monte_carlo import (
    resolve_samplewise_headings,
)

from iscai_stage6.adb.deterministic_association import (
    PredictorAnchor,
    reciprocal_unique_nearest_anchor_association,
)

from iscai_stage6.adb.deterministic_predictive import (
    HORIZONS_S,
    DeterministicSharedMeanPrediction,
    build_deterministic_future_full_boxes,
    build_deterministic_predictive_plan,
)

from iscai_stage6.adb.geometry import (
    Box3D,
)

from iscai_stage6.adb.womd_geometry import (
    CausalADBActorBox,
)


def actor_box(
    *,
    anchor,
    box_center=None,
    yaw=0.0,
    object_type="TYPE_VEHICLE",
    future_state_used=False,
    tracks_to_predict_used=False,
    objects_of_interest_used=False,
):

    if box_center is None:
        box_center = anchor

    return CausalADBActorBox(
        scenario_id="synthetic",

        track_index=0,

        track_id="womd-actor-id-not-used-for-association",

        object_type=object_type,

        box=Box3D(
            center_xyz=tuple(
                float(value)
                for value
                in box_center
            ),

            length_m=4.5,
            width_m=1.8,
            height_m=1.6,
            yaw_rad=float(
                yaw
            ),
        ),

        stage1_anchor_center_H0_m=(
            None
            if anchor is None
            else tuple(
                float(value)
                for value
                in anchor
            )
        ),

        stage1_anchor_bearing_H0_rad=None,

        center_consistency_error_m=0.0,

        source_time_index=10,

        tracks_to_predict_used=(
            tracks_to_predict_used
        ),

        objects_of_interest_used=(
            objects_of_interest_used
        ),

        future_state_used=(
            future_state_used
        ),
    )


class TestBlock63Association(
    unittest.TestCase
):

    def test_single_pair_matches_without_id_equality(self):

        predictors = (
            PredictorAnchor(
                prediction_id="s3trk-internal-17",
                latest_position_H0_m=(
                    10.0,
                    2.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    10.1,
                    2.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            len(
                result.matches
            ),
            1,
        )

        self.assertEqual(
            result.matches[
                0
            ].prediction_id,
            "s3trk-internal-17",
        )

    def test_reciprocal_two_by_two(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
            PredictorAnchor(
                "p1",
                (
                    20.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    10.2,
                    0.0,
                    0.0,
                )
            ),
            actor_box(
                anchor=(
                    19.8,
                    0.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            [
                (
                    match.predictor_index,
                    match.box_index,
                )
                for match
                in result.matches
            ],
            [
                (
                    0,
                    0,
                ),
                (
                    1,
                    1,
                ),
            ],
        )

    def test_exact_predictor_tie_is_unmatched(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    0.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    1.0,
                    0.0,
                    0.0,
                )
            ),
            actor_box(
                anchor=(
                    -1.0,
                    0.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            result.matches,
            (),
        )

        self.assertEqual(
            result.unmatched_box_indices,
            (
                0,
                1,
            ),
        )

    def test_exact_box_tie_is_unmatched(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    -1.0,
                    0.0,
                    0.0,
                ),
            ),
            PredictorAnchor(
                "p1",
                (
                    1.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    0.0,
                    0.0,
                    0.0,
                )
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            result.matches,
            (),
        )

        self.assertEqual(
            result.unmatched_box_indices,
            (
                0,
            ),
        )

    def test_missing_box_anchor_becomes_fallback(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=None,
                box_center=(
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            result.matches,
            (),
        )

        self.assertEqual(
            result.unmatched_box_indices,
            (
                0,
            ),
        )

    def test_no_class_gate(self):

        predictors = (
            PredictorAnchor(
                "p0",
                (
                    10.0,
                    0.0,
                    0.0,
                ),
            ),
        )

        boxes = (
            actor_box(
                anchor=(
                    10.0,
                    0.0,
                    0.0,
                ),
                object_type=(
                    "TYPE_PEDESTRIAN"
                ),
            ),
        )

        result = (
            reciprocal_unique_nearest_anchor_association(
                predictors,
                boxes,
            )
        )

        self.assertEqual(
            len(
                result.matches
            ),
            1,
        )

    def test_duplicate_prediction_ids_rejected(self):

        with self.assertRaises(
            ValueError
        ):
            reciprocal_unique_nearest_anchor_association(
                (
                    PredictorAnchor(
                        "same",
                        (
                            1.0,
                            0.0,
                            0.0,
                        ),
                    ),
                    PredictorAnchor(
                        "same",
                        (
                            2.0,
                            0.0,
                            0.0,
                        ),
                    ),
                ),
                (),
            )


class TestBlock63DeterministicFutureBox(
    unittest.TestCase
):

    def prediction(
        self,
        *,
        anchor=(
            10.0,
            0.0,
            0.0,
        ),
        displacement=None,
    ):

        if displacement is None:
            displacement = (
                (
                    1.0,
                    0.0,
                    0.0,
                ),
                (
                    2.0,
                    0.0,
                    0.0,
                ),
                (
                    3.0,
                    0.0,
                    0.0,
                ),
                (
                    4.0,
                    0.0,
                    0.0,
                ),
            )

        return DeterministicSharedMeanPrediction(
            prediction_id="predictor-internal",

            latest_position_H0_m=anchor,

            mean_displacement_H0_m=(
                displacement
            ),
        )

    def test_horizons_are_frozen(self):

        self.assertEqual(
            HORIZONS_S,
            (
                0.1,
                0.3,
                0.5,
                1.0,
            ),
        )

    def test_future_center_is_predictor_anchor_plus_mean(self):

        prediction = self.prediction(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        centers = (
            prediction.future_centers_H0_m()
        )

        self.assertEqual(
            centers[
                0
            ],
            (
                11.0,
                0.0,
                0.0,
            ),
        )

        self.assertEqual(
            centers[
                3
            ],
            (
                14.0,
                0.0,
                0.0,
            ),
        )

    def test_prediction_is_not_recentered_to_matched_box(self):

        prediction = self.prediction(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        box = actor_box(
            anchor=(
                10.1,
                0.0,
                0.0,
            ),

            # Intentionally very different from the predictor
            # anchor to prove that the future state is NOT
            # recentered to annotation geometry.
            box_center=(
                100.0,
                50.0,
                0.0,
            ),
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        self.assertEqual(
            future[
                0
            ].center_H0_m,
            (
                11.0,
                0.0,
                0.0,
            ),
        )

        self.assertNotEqual(
            future[
                0
            ].center_H0_m,
            (
                101.0,
                50.0,
                0.0,
            ),
        )

    def test_current_dimensions_are_held_across_horizons(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        for item in future:

            self.assertEqual(
                item.box.length_m,
                box.box.length_m,
            )

            self.assertEqual(
                item.box.width_m,
                box.box.width_m,
            )

            self.assertEqual(
                item.box.height_m,
                box.box.height_m,
            )

    def test_every_horizon_projects_eight_physical_corners(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        self.assertEqual(
            len(
                future
            ),
            4,
        )

        for item in future:

            self.assertEqual(
                item.projection.corners.xyz.shape,
                (
                    8,
                    3,
                ),
            )

            self.assertGreater(
                item.projection.theta_span_rad,
                0.0,
            )

            self.assertGreater(
                item.projection.ground_range_max_m,
                item.projection.ground_range_min_m,
            )

    def test_heading_matches_frozen_stage5_rule(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            yaw=0.25,
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        centers = np.asarray(
            prediction.future_centers_H0_m(),
            dtype=np.float64,
        )

        expected, expected_fallback = (
            resolve_samplewise_headings(
                trajectory_samples_h0_m=(
                    centers[
                        None,
                        :,
                        :,
                    ]
                ),

                current_position_h0_m=(
                    prediction.latest_position_H0_m
                ),

                current_heading_h0_rad=(
                    box.box.yaw_rad
                ),
            )
        )

        np.testing.assert_allclose(
            np.asarray(
                [
                    item.yaw_H0_rad
                    for item
                    in future
                ]
            ),
            expected[
                0
            ],
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_array_equal(
            np.asarray(
                [
                    item.heading_fallback_used
                    for item
                    in future
                ]
            ),
            expected_fallback[
                0
            ],
        )

    def test_low_speed_heading_carries_current_causal_yaw(self):

        current_yaw = 0.73

        prediction = self.prediction(
            displacement=(
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
                (
                    0.0,
                    0.0,
                    0.0,
                ),
            )
        )

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            yaw=current_yaw,
        )

        future = (
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )
        )

        for item in future:

            self.assertTrue(
                item.heading_fallback_used
            )

            self.assertAlmostEqual(
                item.yaw_H0_rad,
                current_yaw,
            )

    def test_future_state_provenance_is_rejected(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            future_state_used=True,
        )

        with self.assertRaises(
            RuntimeError
        ):
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )

    def test_tracks_to_predict_provenance_is_rejected(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            tracks_to_predict_used=True,
        )

        with self.assertRaises(
            RuntimeError
        ):
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )

    def test_objects_of_interest_provenance_is_rejected(self):

        prediction = self.prediction()

        box = actor_box(
            anchor=(
                10.0,
                0.0,
                0.0,
            ),
            objects_of_interest_used=True,
        )

        with self.assertRaises(
            RuntimeError
        ):
            build_deterministic_future_full_boxes(
                prediction,
                box,
            )

    def test_deterministic_prediction_has_no_covariance_field(self):

        signature = inspect.signature(
            DeterministicSharedMeanPrediction
        )

        self.assertNotIn(
            "covariance",
            " ".join(
                signature.parameters
            ).lower(),
        )

        self.assertNotIn(
            "scale_tril",
            " ".join(
                signature.parameters
            ).lower(),
        )

    def test_plan_exposes_reactive_fallback(self):

        prediction = self.prediction(
            anchor=(
                10.0,
                0.0,
                0.0,
            )
        )

        # Exact tie -> no accepted predictive match.
        boxes = (
            actor_box(
                anchor=(
                    9.0,
                    0.0,
                    0.0,
                )
            ),
            actor_box(
                anchor=(
                    11.0,
                    0.0,
                    0.0,
                )
            ),
        )

        plan = (
            build_deterministic_predictive_plan(
                (
                    prediction,
                ),
                boxes,
            )
        )

        self.assertEqual(
            plan.matched_forecasts,
            (),
        )

        self.assertEqual(
            plan.reactive_fallback_box_indices,
            (
                0,
                1,
            ),
        )

        self.assertEqual(
            plan.unmatched_prediction_indices,
            (
                0,
            ),
        )

    def test_matched_plan_contains_four_future_boxes(self):

        prediction = self.prediction()

        boxes = (
            actor_box(
                anchor=(
                    10.1,
                    0.0,
                    0.0,
                )
            ),
        )

        plan = (
            build_deterministic_predictive_plan(
                (
                    prediction,
                ),
                boxes,
            )
        )

        self.assertEqual(
            len(
                plan.matched_forecasts
            ),
            1,
        )

        self.assertEqual(
            plan.predicted_box_count,
            4,
        )

        self.assertEqual(
            plan.reactive_fallback_box_indices,
            (),
        )


if __name__ == "__main__":
    unittest.main()
'''

    test_status = write_new_or_exact(
        TEST_FILE,
        test_source,
    )

    print(
        "test file =",
        test_status,
    )

    # ========================================================
    # F. Controlled compile
    # ========================================================

    print()
    print(
        "===== F. CONTROLLED COMPILE ====="
    )

    compile_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(
                ASSOCIATION_MODULE
            ),
            str(
                PREDICTIVE_MODULE
            ),
            str(
                TEST_FILE
            ),
        ],
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    require(
        compile_process.returncode
        ==
        0,
        (
            "Block6.3 Part1 compile failed:\n"
            +
            compile_process.stdout
        ),
    )

    print(
        "association module = PASS"
    )

    print(
        "predictive module  = PASS"
    )

    print(
        "tests              = PASS"
    )

    # ========================================================
    # G. Dedicated Part1 tests
    # ========================================================

    print()
    print(
        "===== G. DEDICATED BLOCK6.3 PART1 TESTS ====="
    )

    dedicated = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "iscai_stage6.tests."
            "test_block63_part1_deterministic_future_box",
        ],
        cwd=str(
            ROOT
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Fallback to discovery if tests are not packaged as an
    # importable Python package.
    if dedicated.returncode != 0:

        dedicated = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(
                    S6 / "tests"
                ),
                "-p",
                (
                    "test_block63_part1_"
                    "deterministic_future_box.py"
                ),
            ],
            cwd=str(
                S6
            ),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    require(
        dedicated.returncode
        ==
        0,
        (
            "Block6.3 Part1 dedicated tests failed:\n"
            +
            tail_text(
                dedicated.stdout,
                120,
            )
        ),
    )

    for line in dedicated.stdout.splitlines():

        stripped = line.strip()

        if (
            stripped.startswith(
                "Ran "
            )
            or
            stripped
            ==
            "OK"
        ):

            print(
                stripped
            )

    # ========================================================
    # H. Independent deterministic semantic smoke
    # ========================================================

    print()
    print(
        "===== H. INDEPENDENT FUTURE-BOX SEMANTIC SMOKE ====="
    )

    from iscai_stage6.adb.deterministic_association import (
        PredictorAnchor,
        reciprocal_unique_nearest_anchor_association,
    )

    from iscai_stage6.adb.deterministic_predictive import (
        DeterministicSharedMeanPrediction,
        build_deterministic_predictive_plan,
    )

    from iscai_stage6.adb.womd_geometry import (
        CausalADBActorBox,
    )

    prediction = (
        DeterministicSharedMeanPrediction(
            prediction_id=(
                "synthetic-predictor"
            ),

            latest_position_H0_m=(
                20.0,
                0.0,
                0.0,
            ),

            mean_displacement_H0_m=(
                (
                    1.0,
                    0.0,
                    0.0,
                ),
                (
                    2.0,
                    0.2,
                    0.0,
                ),
                (
                    3.0,
                    0.5,
                    0.0,
                ),
                (
                    5.0,
                    1.0,
                    0.0,
                ),
            ),
        )
    )

    actor = CausalADBActorBox(
        scenario_id=(
            "synthetic"
        ),

        track_index=0,

        track_id=(
            "completely-different-WOMD-id"
        ),

        object_type=(
            "TYPE_VEHICLE"
        ),

        box=Box3D(
            center_xyz=(
                20.2,
                0.0,
                0.0,
            ),

            length_m=4.6,
            width_m=1.9,
            height_m=1.6,
            yaw_rad=0.0,
        ),

        stage1_anchor_center_H0_m=(
            20.2,
            0.0,
            0.0,
        ),

        stage1_anchor_bearing_H0_rad=0.0,

        center_consistency_error_m=0.0,

        source_time_index=10,
    )

    plan = (
        build_deterministic_predictive_plan(
            (
                prediction,
            ),
            (
                actor,
            ),
        )
    )

    require(
        len(
            plan.matched_forecasts
        )
        ==
        1,
        (
            "Independent synthetic association failed."
        ),
    )

    forecast = (
        plan.matched_forecasts[
            0
        ]
    )

    require(
        len(
            forecast.future
        )
        ==
        4,
        (
            "Future horizon count changed."
        ),
    )

    expected_first_center = (
        21.0,
        0.0,
        0.0,
    )

    require(
        forecast.future[
            0
        ].center_H0_m
        ==
        expected_first_center,
        (
            "Future center was incorrectly "
            "recentered to the actor box."
        ),
    )

    corner_counts = [
        int(
            item.projection
            .corners.xyz.shape[
                0
            ]
        )
        for item
        in forecast.future
    ]

    require(
        corner_counts
        ==
        [
            8,
            8,
            8,
            8,
        ],
        (
            "Not every deterministic future "
            "box retained eight corners."
        ),
    )

    print(
        "association pair count     =",
        plan.association.matched_pair_count,
    )

    print(
        "association anchor error m =",
        forecast.association_anchor_distance_m,
    )

    print(
        "WOMD track-ID equality used= NO"
    )

    print(
        "first future center H0     =",
        forecast.future[
            0
        ].center_H0_m,
    )

    print(
        "box-center recentering     = NO PASS"
    )

    print(
        "corner counts              =",
        corner_counts,
    )

    print(
        "reactive fallback boxes    =",
        plan.reactive_fallback_box_indices,
    )

    # ========================================================
    # I. Scope and API guards
    # ========================================================

    print()
    print(
        "===== I. BLOCK6.3 PART1 SCOPE GUARDS ====="
    )

    predictive_text = (
        PREDICTIVE_MODULE.read_text(
            encoding="utf-8"
        ).lower()
    )

    association_text = (
        ASSOCIATION_MODULE.read_text(
            encoding="utf-8"
        ).lower()
    )

    # Literal terms may legitimately appear in explanatory
    # docstrings, so enforce forbidden behavior structurally
    # through signatures and frozen binding rather than naïve
    # token absence.

    prediction_signature = (
        inspect.signature(
            DeterministicSharedMeanPrediction
        )
    )

    signature_text = (
        " ".join(
            prediction_signature.parameters
        ).lower()
    )

    require(
        "covariance"
        not in
        signature_text,
        (
            "Predictive covariance entered "
            "deterministic prediction contract."
        ),
    )

    require(
        "truth"
        not in
        signature_text,
        (
            "Truth metadata entered deterministic "
            "prediction contract."
        ),
    )

    association_signature = (
        inspect.signature(
            reciprocal_unique_nearest_anchor_association
        )
    )

    association_signature_text = (
        " ".join(
            association_signature.parameters
        ).lower()
    )

    for forbidden in (
        "threshold",
        "truth",
        "class",
        "covariance",
        "track_id",
    ):

        require(
            forbidden
            not in
            association_signature_text,
            (
                "Frozen association gained forbidden "
                f"parameter: {forbidden}"
            ),
        )

    print(
        "predictive covariance      = NOT USED"
    )

    print(
        "probabilistic occupancy    = NOT IMPLEMENTED"
    )

    print(
        "class-aware policy         = NOT IMPLEMENTED"
    )

    print(
        "association threshold      = NONE"
    )

    print(
        "pre-association class gate = NONE"
    )

    print(
        "perfect actor ID           = NONE"
    )

    print(
        "future GT                  = NONE"
    )

    print(
        "communication codebook     = NOT USED"
    )

    print(
        "formal grid freeze         = NO"
    )

    # ========================================================
    # J. Full Stage6 regression
    # ========================================================

    print()
    print(
        "===== J. FULL STAGE6 REGRESSION ====="
    )

    regression = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                S6 / "tests"
            ),
            "-p",
            "test_block6*.py",
        ],
        cwd=str(
            S6
        ),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    REGRESSION_LOG.write_text(
        regression.stdout,
        encoding="utf-8",
    )

    require(
        regression.returncode
        ==
        0,
        (
            "Full Stage6 regression failed.\n"
            f"Full log: {REGRESSION_LOG}\n"
            +
            tail_text(
                regression.stdout,
                140,
            )
        ),
    )

    for line in regression.stdout.splitlines():

        stripped = line.strip()

        if (
            stripped.startswith(
                "Ran "
            )
            or
            stripped
            ==
            "OK"
        ):

            print(
                stripped
            )

    # ========================================================
    # K. Frozen immutability / storage
    # ========================================================

    print()
    print(
        "===== K. IMMUTABILITY / STORAGE ====="
    )

    changed = []

    for path in protected:

        if (
            protected_before[
                str(path)
            ]
            !=
            sha256_file(
                path
            )
        ):

            changed.append(
                str(path)
            )

    require(
        not changed,
        (
            "Frozen prerequisite changed: "
            + ", ".join(changed)
        ),
    )

    free_gib = (
        shutil.disk_usage(
            ROOT
        ).free
        /
        1024**3
    )

    require(
        free_gib
        >=
        MIN_FREE_GIB,
        (
            "250-GiB reserve violated: "
            f"{free_gib:.3f} GiB"
        ),
    )

    print(
        "binding freeze      = UNCHANGED"
    )

    print(
        "Block6.2            = UNCHANGED"
    )

    print(
        "Stage5 heading code = UNCHANGED"
    )

    print(
        "Stage6 geometry      = UNCHANGED"
    )

    print(
        "free GiB             =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve      = PASS"
    )

    # ========================================================
    # L. Part1 report
    # ========================================================

    print()
    print(
        "===== L. BLOCK6.3 PART1 REPORT ====="
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "part":
            "1/2",

        "status":
            (
                "PASS_DETERMINISTIC_"
                "FUTURE_FULL_BOX"
            ),

        "binding": {
            "path":
                str(
                    BINDING
                ),

            "sha256":
                EXPECTED[
                    "binding"
                ],
        },

        "implementation": {
            "association_module":
                str(
                    ASSOCIATION_MODULE
                ),

            "association_module_sha256":
                sha256_file(
                    ASSOCIATION_MODULE
                ),

            "predictive_module":
                str(
                    PREDICTIVE_MODULE
                ),

            "predictive_module_sha256":
                sha256_file(
                    PREDICTIVE_MODULE
                ),

            "association":
                (
                    "RECIPROCAL_UNIQUE_"
                    "NEAREST_CURRENT_H0"
                ),

            "association_threshold":
                None,

            "forced_assignment":
                False,

            "class_gate":
                False,

            "perfect_actor_identity":
                False,

            "deterministic_trajectory":
                (
                    "SHARED_CALIBRATED_"
                    "GAUSSIAN_MEAN_ONLY"
                ),

            "coordinate_semantics":
                (
                    "PREDICTOR_CAUSAL_ANCHOR_H0"
                    "_PLUS_METRIC_H0_DISPLACEMENT"
                ),

            "annotation_recenter":
                False,

            "future_yaw":
                (
                    "STAGE5_TRAJECTORY_TANGENT_"
                    "LOW_SPEED_CAUSAL_CARRY_FORWARD"
                ),

            "future_dimensions":
                (
                    "CURRENT_CAUSAL_LWH_HELD"
                ),

            "future_projection":
                "FULL_8_CORNER_BOX",

            "state_source":
                PREDICTED_STATE_SOURCE,

            "orientation_source":
                PREDICTED_ORIENTATION_SOURCE,

            "controller_path":
                True,

            "unmatched_box_policy":
                (
                    "EXPLICIT_ORIGINAL_"
                    "REACTIVE_FALLBACK_INDEX"
                ),
        },

        "scope": {
            "predictive_covariance":
                False,

            "probabilistic_occupancy":
                False,

            "class_aware_policy":
                False,

            "temporal_smoothing":
                False,

            "actuation_rate_limit":
                False,

            "communication_codebook":
                False,

            "formal_grid_frozen":
                False,
        },

        "execution": {
            "training":
                False,

            "model_inference":
                False,

            "dataset_access":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,
        },

        "regression": {
            "Stage6":
                "PASS",

            "log":
                str(
                    REGRESSION_LOG
                ),
        },

        "upstream_modified":
            False,

        "next":
            (
                "Block6.3 Part2 connect frozen "
                "Stage4 Gaussian runtime on "
                "development-only causal records, "
                "validate association, then construct "
                "deterministic predictive illumination."
            ),
    }

    atomic_json(
        REPORT,
        report,
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "report SHA256 =",
        sha256_file(
            REPORT
        ),
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 PART 1/2 — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "binding prerequisite       = EXACT PASS"
    )

    print(
        "association                = "
        "RECIPROCAL UNIQUE NEAREST H0"
    )

    print(
        "association threshold      = NONE"
    )

    print(
        "forced assignment          = NO"
    )

    print(
        "perfect actor identity     = NO"
    )

    print(
        "unmatched actor handling   = "
        "EXPLICIT REACTIVE FALLBACK"
    )

    print(
        "deterministic trajectory   = SHARED GAUSSIAN MEAN"
    )

    print(
        "future center              = "
        "PREDICTOR ANCHOR + MEAN DISPLACEMENT"
    )

    print(
        "annotation recentering     = NO"
    )

    print(
        "future yaw                 = STAGE5 TANGENT RULE"
    )

    print(
        "future dimensions          = CURRENT CAUSAL L/W/H"
    )

    print(
        "future projections         = 4 HORIZONS"
    )

    print(
        "physical corners           = 8 PER FUTURE BOX"
    )

    print(
        "controller provenance      = PASS"
    )

    print(
        "predictive covariance      = NO"
    )

    print(
        "probabilistic occupancy    = NOT STARTED"
    )

    print(
        "class-aware ADB            = NOT STARTED"
    )

    print(
        "formal evaluation          = NO"
    )

    print(
        "training/model inference   = NO"
    )

    print(
        "dataset access             = NO"
    )

    print(
        "Stage6 completion claimed  = NO"
    )

    print(
        "upstream modified          = NO"
    )

    print(
        "STATUS = PASS_DETERMINISTIC_FUTURE_FULL_BOX"
    )

    print(
        "terminal remains open = YES"
    )


try:

    main()

except BaseException as exc:

    removed = rollback_created()

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 PART 1/2 = BLOCKED"
    )
    print(
        "============================================================"
    )

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    print()
    traceback.print_exc(
        limit=16
    )

    print()
    print(
        "new Block6.3 source rollback =",
        (
            "PASS"
            if removed
            else
            "NOT REQUIRED"
        ),
    )

    if removed:

        for path in removed:
            print(
                "removed =",
                path,
            )

    print(
        "Block6.2 modified       = NO"
    )

    print(
        "Stage3–5 modified       = NO"
    )

    print(
        "training/inference      = NO"
    )

    print(
        "dataset access          = NO"
    )

    print(
        "formal evaluation       = NO"
    )

    print(
        "parameter tuning        = NO"
    )

    print(
        "terminal remains open   = YES"
    )

# Deliberately no non-zero sys.exit().

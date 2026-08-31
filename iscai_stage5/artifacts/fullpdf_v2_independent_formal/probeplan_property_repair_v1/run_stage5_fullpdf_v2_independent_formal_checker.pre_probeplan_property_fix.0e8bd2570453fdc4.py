#!/usr/bin/env python3
from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ============================================================
# Frozen project paths
# ============================================================

ROOT = Path("/home/agni/waymo")

S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

RUNTIME = (
    S5
    / "src/iscai_stage5/fullpdf_v2.py"
)

PROTOCOL = (
    S5
    / "configs/stage5_fullpdf_v2_protocol.json"
)

STAGE4_LEDGER = (
    S4
    / "artifacts/fullpdf_v2/"
    "formal_prediction_ledger_run1.jsonl"
)

REPAIR_REPORT = (
    S5
    / "reports/stage5_fullpdf_v2_repair.json"
)

OUT = (
    S5
    / "artifacts/fullpdf_v2_independent_formal"
)

REPORT = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_independent_formal_checker.json"
)

CLOSURE = (
    S5
    / "reports/"
    "stage5_fullpdf_v2_final_closure.json"
)

HANDOFF = (
    OUT
    / "stage5_to_stage6_handoff_fullpdf_v2.json"
)

RUN1_LEDGER = OUT / "decision_pass_run1.jsonl"
RUN2_LEDGER = OUT / "decision_pass_run2.jsonl"
SEALED_LEDGER = OUT / "decision_ledger_sealed.jsonl"
SEAL_REPORT = OUT / "decision_ledger_seal.json"
EVALUATOR_ROWS = OUT / "formal_evaluator_rows.jsonl"
REGRESSION_LOG = OUT / "stage5_regression.log"


# ============================================================
# Frozen hashes
# ============================================================

EXPECTED_RUNTIME_SHA = (
    "e57708e9332bb40e84f18270e483d911"
    "65e26c149b80e82c8f10b4df44376249"
)

EXPECTED_PROTOCOL_SHA = (
    "91dec3cc3d3a7b72385f012d1d2bfdb8"
    "981a497e1344f585d38f3e05f2e4fa5a"
)

EXPECTED_STAGE4_LEDGER_SHA = (
    "2dd3c396d5c365677c41b3a0df6344a7"
    "c28d423daaa368dbe466bd4b17b4c2a8"
)

EXPECTED_REGRESSION_TEST_COUNT = 271

HARD_RESERVE_BYTES = 250 * (1024 ** 3)


# ============================================================
# State
# ============================================================

FORMAL_TRUTH_OPENED = False
DECISION_LEDGER_SEALED = False

GATES: dict[str, bool] = {}
EVIDENCE: dict[str, Any] = {}
EVENTS: list[dict[str, Any]] = []


class FailClosed(RuntimeError):
    pass


# ============================================================
# Generic helpers
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def json_safe(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            k: json_safe(v)
            for k, v
            in dataclasses.asdict(obj).items()
        }

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (tuple, list)):
        return [
            json_safe(v)
            for v in obj
        ]

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or obj is None:
        return obj

    if hasattr(obj, "__dict__"):
        return {
            k: json_safe(v)
            for k, v
            in vars(obj).items()
            if not k.startswith("_")
        }

    return repr(obj)


def atomic_write_json(
    path: Path,
    obj: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    text = json.dumps(
        json_safe(obj),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    tmp.write_text(
        text,
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def write_jsonl(
    path: Path,
    rows: Iterable[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in rows:
            f.write(
                canonical_json(
                    json_safe(row)
                )
            )
            f.write("\n")

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise FailClosed(message)


def record_gate(
    key: str,
    passed: bool,
    evidence: Any,
) -> None:
    GATES[key] = bool(passed)
    EVIDENCE[key] = json_safe(evidence)

    if not passed:
        raise FailClosed(
            f"Mandatory gate failed: {key}"
        )


def event(
    name: str,
    **payload: Any,
) -> None:
    EVENTS.append(
        {
            "sequence": len(EVENTS) + 1,
            "event": name,
            **json_safe(payload),
        }
    )


def recursively_find_alias(
    obj: Any,
    aliases: Iterable[str],
) -> Any:
    wanted = {
        str(x).lower()
        for x in aliases
    }

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in wanted:
                return value

        for value in obj.values():
            result = recursively_find_alias(
                value,
                wanted,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = recursively_find_alias(
                value,
                wanted,
            )

            if result is not None:
                return result

    return None


def scalar_from(
    obj: Any,
    aliases: Iterable[str],
) -> float:
    if isinstance(
        obj,
        (int, float, np.number),
    ):
        return float(obj)

    value = recursively_find_alias(
        json_safe(obj),
        aliases,
    )

    if value is None:
        raise FailClosed(
            "Unable to extract scalar "
            f"{tuple(aliases)} from {type(obj)}."
        )

    return float(value)


def vector3(value: Any) -> np.ndarray:
    arr = np.asarray(
        value,
        dtype=float,
    )

    require(
        arr.shape == (3,),
        f"Expected vector3, got {arr.shape}.",
    )

    require(
        np.isfinite(arr).all(),
        "Non-finite vector3.",
    )

    return arr


def matrix3(value: Any) -> np.ndarray:
    arr = np.asarray(
        value,
        dtype=float,
    )

    require(
        arr.shape == (3, 3),
        f"Expected 3x3 matrix, got {arr.shape}.",
    )

    require(
        np.isfinite(arr).all(),
        "Non-finite covariance.",
    )

    return arr


def normalize_angle(x: float) -> float:
    return float(
        math.atan2(
            math.sin(x),
            math.cos(x),
        )
    )


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(float(angle))
    s = math.sin(float(angle))

    return np.asarray(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def point_to_angles(
    point: Any,
) -> tuple[float, float]:
    xyz = vector3(point)

    x, y, z = (
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
    )

    horizontal = math.hypot(x, y)

    require(
        horizontal > 0.0,
        "Undefined azimuth at zero horizontal range.",
    )

    return (
        math.atan2(y, x),
        math.atan2(z, horizontal),
    )


# ============================================================
# Failure writer
# ============================================================

def fail_closed(
    reason: str,
    *,
    exception: BaseException | None = None,
) -> None:
    status = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status": "FAIL_CLOSED_FULLPDF_V2_FORMAL",
        "Stage5": (
            "REPAIRED_AND_FROZEN_PRE_FORMAL"
        ),
        "Stage6_allowed": False,
        "decision_ledger_sealed": (
            DECISION_LEDGER_SEALED
        ),
        "formal_evaluator_truth_opened": (
            FORMAL_TRUTH_OPENED
        ),
        "formal_outcomes_read": (
            FORMAL_TRUTH_OPENED
        ),
        "failure_reason": reason,
        "exception": (
            None
            if exception is None
            else repr(exception)
        ),
        "gates": GATES,
        "evidence": EVIDENCE,
        "events": EVENTS,
        "no_training": True,
        "no_retraining": True,
        "no_recalibration": True,
        "no_parameter_tuning": True,
        "no_post_outcome_tuning": True,
        "handoff_written": False,
    }

    atomic_write_json(
        REPORT,
        status,
    )

    print()
    print("=" * 72)
    print("STAGE5 FULLPDF V2 FORMAL = FAIL CLOSED")
    print("Stage6_allowed = false")
    print("reason =", reason)
    print("report =", REPORT)
    print("=" * 72)


# ============================================================
# File loading
# ============================================================

def load_json_file(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_records(
    path: Path,
    *,
    limit: int | None = None,
) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)

                if isinstance(row, dict):
                    rows.append(row)

                if (
                    limit is not None
                    and len(rows) >= limit
                ):
                    break

        return rows

    if path.suffix.lower() == ".json":
        data = load_json_file(path)

        if isinstance(data, list):
            return [
                x
                for x in data
                if isinstance(x, dict)
            ][:limit]

        if isinstance(data, dict):
            for key in (
                "rows",
                "records",
                "items",
                "samples",
                "predictions",
                "receivers",
            ):
                value = data.get(key)

                if isinstance(value, list):
                    rows = [
                        x
                        for x in value
                        if isinstance(x, dict)
                    ]

                    return (
                        rows
                        if limit is None
                        else rows[:limit]
                    )

            return [data]

    return []


# ============================================================
# Provenance helpers
# ============================================================

def discover_exact_sha_file(
    expected_sha: str,
) -> Path:
    candidates = []

    roots = [
        S4 / "reports",
        S4 / "artifacts/fullpdf_v2",
        S4 / "artifacts/block410",
    ]

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".json",
                ".jsonl",
            }:
                continue

            if path.stat().st_size > 20_000_000:
                continue

            try:
                if sha256_file(path) == expected_sha:
                    candidates.append(path)
            except OSError:
                continue

    require(
        len(candidates) == 1,
        (
            "Expected exactly one Stage4 artifact "
            f"with SHA {expected_sha}; "
            f"found {len(candidates)}: {candidates}"
        ),
    )

    return candidates[0]


def discover_horizons(
    *objects: Any,
) -> list[float]:
    found: list[tuple[float, ...]] = []

    def walk(obj: Any, key: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                low = str(k).lower()

                if (
                    "horizon" in low
                    and isinstance(v, list)
                    and len(v) == 4
                ):
                    try:
                        values = tuple(
                            float(x)
                            for x in v
                        )
                    except Exception:
                        values = ()

                    if (
                        len(values) == 4
                        and all(
                            math.isfinite(x)
                            and x > 0.0
                            for x in values
                        )
                    ):
                        found.append(values)

                walk(v, str(k))

        elif isinstance(obj, list):
            for v in obj:
                walk(v, key)

    for obj in objects:
        walk(obj)

    unique = sorted(
        set(found)
    )

    require(
        len(unique) == 1,
        (
            "Could not resolve one unique frozen "
            f"4-horizon vector. Found: {unique}"
        ),
    )

    return list(unique[0])


# ============================================================
# Causal binding discovery
# ============================================================

CAUSAL_FORBIDDEN_TOKENS = (
    "future",
    "truth",
    "ground_truth",
    "oracle",
    "tracks_to_predict",
    "objects_of_interest",
)


def safe_negative_marker(value: Any) -> bool:
    if value is False or value is None:
        return True

    if isinstance(value, str):
        upper = value.upper()

        return any(
            token in upper
            for token in (
                "NO",
                "FALSE",
                "NOT_USED",
                "NOT INPUT",
                "FORBIDDEN",
                "EVALUATOR_ONLY",
            )
        )

    return False


def has_forbidden_causal_payload(
    obj: Any,
) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            low = str(key).lower()

            if any(
                token in low
                for token in CAUSAL_FORBIDDEN_TOKENS
            ):
                if not safe_negative_marker(value):
                    return True

            if has_forbidden_causal_payload(
                value
            ):
                return True

    elif isinstance(obj, list):
        for value in obj:
            if has_forbidden_causal_payload(
                value
            ):
                return True

    return False


def class_is_vehicle(value: Any) -> bool:
    if isinstance(
        value,
        (int, np.integer),
    ):
        return int(value) == 1

    text = str(value).upper()

    return (
        "VEHICLE" in text
        or text in {"1", "TYPE_1"}
    )


def parse_dimensions(
    row: dict,
) -> tuple[float, float, float] | None:
    length = recursively_find_alias(
        row,
        (
            "length_m",
            "current_length_m",
            "anchor_length_m",
            "actor_length_m",
        ),
    )

    width = recursively_find_alias(
        row,
        (
            "width_m",
            "current_width_m",
            "anchor_width_m",
            "actor_width_m",
        ),
    )

    height = recursively_find_alias(
        row,
        (
            "height_m",
            "current_height_m",
            "anchor_height_m",
            "actor_height_m",
        ),
    )

    if (
        length is not None
        and width is not None
        and height is not None
    ):
        return (
            float(length),
            float(width),
            float(height),
        )

    vector = recursively_find_alias(
        row,
        (
            "dimensions_m",
            "actor_dimensions_m",
            "box_dimensions_m",
            "current_dimensions_m",
        ),
    )

    if vector is not None:
        arr = np.asarray(
            vector,
            dtype=float,
        )

        if arr.shape == (3,):
            return (
                float(arr[0]),
                float(arr[1]),
                float(arr[2]),
            )

    return None


def parse_causal_row(
    row: dict,
) -> dict | None:
    if has_forbidden_causal_payload(row):
        return None

    sid = recursively_find_alias(
        row,
        ("scenario_id",),
    )

    pid = recursively_find_alias(
        row,
        ("prediction_id",),
    )

    if sid is None or pid is None:
        return None

    heading = recursively_find_alias(
        row,
        (
            "heading_h0_rad",
            "current_heading_h0_rad",
            "anchor_heading_h0_rad",
            "heading_rad",
            "yaw_h0_rad",
        ),
    )

    dims = parse_dimensions(row)

    actor_class = recursively_find_alias(
        row,
        (
            "actor_class",
            "object_class",
            "semantic_class",
            "object_type",
            "class_name",
        ),
    )

    if (
        heading is None
        or dims is None
        or actor_class is None
    ):
        return None

    selected_flag = recursively_find_alias(
        row,
        (
            "selected_receiver",
            "receiver_selected",
            "is_selected_receiver",
            "is_primary_receiver",
        ),
    )

    selected_pid = recursively_find_alias(
        row,
        (
            "selected_receiver_prediction_id",
            "receiver_prediction_id",
        ),
    )

    if selected_flag is False:
        return None

    if (
        selected_pid is not None
        and str(selected_pid) != str(pid)
    ):
        return None

    if not class_is_vehicle(actor_class):
        return None

    length, width, height = dims

    if not (
        math.isfinite(float(heading))
        and length > 0.0
        and width > 0.0
        and height > 0.0
    ):
        return None

    return {
        "scenario_id": str(sid),
        "prediction_id": str(pid),
        "heading_h0_rad": float(heading),
        "length_m": float(length),
        "width_m": float(width),
        "height_m": float(height),
        "actor_class": str(actor_class),
        "selection_flag_present": (
            selected_flag is not None
            or selected_pid is not None
        ),
    }


def candidate_v2_paths(
    *,
    truth_phase: bool,
) -> list[Path]:
    paths: set[Path] = set()

    roots = [
        S4 / "artifacts/fullpdf_v2",
        S5 / "artifacts/fullpdf_v2",
        S5 / "artifacts/fullpdf_v2_repair",
        S5 / "artifacts/stage5_fullpdf_v2",
    ]

    truth_tokens = (
        "truth",
        "evaluator",
        "future",
        "target",
    )

    causal_tokens = (
        "causal",
        "binding",
        "receiver",
        "metadata",
        "current",
        "anchor",
        "identity",
    )

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".json",
                ".jsonl",
            }:
                continue

            if path.resolve() == STAGE4_LEDGER.resolve():
                continue

            low = str(path).lower()

            if any(
                token in low
                for token in (
                    "debug",
                    "invalidated",
                    "quarantine",
                    "pre_",
                )
            ):
                continue

            if truth_phase:
                if not any(
                    token in low
                    for token in truth_tokens
                ):
                    continue
            else:
                if any(
                    token in low
                    for token in truth_tokens
                ):
                    continue

                if not any(
                    token in low
                    for token in causal_tokens
                ):
                    continue

            paths.add(path)

    # Paths explicitly mentioned by the pre-formal repair report.
    if REPAIR_REPORT.is_file():
        repair = load_json_file(
            REPAIR_REPORT
        )

        def collect_strings(obj: Any):
            if isinstance(obj, dict):
                for v in obj.values():
                    collect_strings(v)
            elif isinstance(obj, list):
                for v in obj:
                    collect_strings(v)
            elif isinstance(obj, str):
                p = Path(obj)

                if (
                    p.is_absolute()
                    and p.is_file()
                    and p.suffix.lower()
                    in {".json", ".jsonl"}
                ):
                    low = str(p).lower()

                    if truth_phase:
                        if any(
                            t in low
                            for t in truth_tokens
                        ):
                            paths.add(p)
                    else:
                        if (
                            not any(
                                t in low
                                for t in truth_tokens
                            )
                            and any(
                                t in low
                                for t in causal_tokens
                            )
                        ):
                            paths.add(p)

        collect_strings(repair)

    return sorted(paths)


def resolve_causal_binding(
    ledger_keys: set[tuple[str, str]],
) -> tuple[Path, list[dict]]:
    valid: list[
        tuple[Path, str, list[dict]]
    ] = []

    for path in candidate_v2_paths(
        truth_phase=False
    ):
        try:
            probe_rows = load_records(
                path,
                limit=50,
            )
        except Exception:
            continue

        parsed_probe = [
            parse_causal_row(r)
            for r in probe_rows
        ]

        parsed_probe = [
            r
            for r in parsed_probe
            if r is not None
        ]

        if not parsed_probe:
            continue

        try:
            all_rows = load_records(path)
        except Exception:
            continue

        parsed = [
            parse_causal_row(r)
            for r in all_rows
        ]

        parsed = [
            r
            for r in parsed
            if r is not None
        ]

        if not parsed:
            continue

        # Every selected row must bind to the frozen Stage4 ledger.
        if not all(
            (
                r["scenario_id"],
                r["prediction_id"],
            )
            in ledger_keys
            for r in parsed
        ):
            continue

        by_scene: dict[str, list[dict]] = defaultdict(list)

        for row in parsed:
            by_scene[
                row["scenario_id"]
            ].append(row)

        # Exactly one selected causal receiver per represented scene.
        if not all(
            len(rows) == 1
            for rows in by_scene.values()
        ):
            continue

        # If the artifact does not contain an explicit selection
        # flag, its filename itself must explicitly identify a
        # selected/primary receiver binding.
        explicit = all(
            bool(r["selection_flag_present"])
            for r in parsed
        )

        if not explicit:
            low_name = path.name.lower()

            if not (
                "selected_receiver" in low_name
                or "primary_receiver" in low_name
            ):
                continue

        valid.append(
            (
                path,
                sha256_file(path),
                parsed,
            )
        )

    require(
        len(valid) > 0,
        (
            "No unique frozen V2 causal receiver binding "
            "could be proven without future/evaluator data."
        ),
    )

    # Equivalent duplicate copies with identical content are okay.
    grouped: dict[str, list] = defaultdict(list)

    for item in valid:
        grouped[item[1]].append(item)

    require(
        len(grouped) == 1,
        (
            "Multiple scientifically distinct causal receiver "
            f"bindings found: {[str(x[0]) for x in valid]}"
        ),
    )

    group = next(
        iter(grouped.values())
    )

    chosen = sorted(
        group,
        key=lambda x: str(x[0]),
    )[0]

    return chosen[0], chosen[2]


# ============================================================
# Runtime-object extraction
# ============================================================

def extract_az_el(
    angles: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(angles, np.ndarray):
        arr = np.asarray(
            angles,
            dtype=float,
        )

        if (
            arr.ndim == 2
            and arr.shape[1] >= 2
        ):
            return (
                arr[:, 0],
                arr[:, 1],
            )

    if isinstance(
        angles,
        (tuple, list),
    ):
        if len(angles) >= 2:
            a = np.asarray(
                angles[0],
                dtype=float,
            )
            e = np.asarray(
                angles[1],
                dtype=float,
            )

            if (
                a.ndim == 1
                and e.ndim == 1
                and len(a) == len(e)
            ):
                return a, e

    data = json_safe(angles)

    az = recursively_find_alias(
        data,
        (
            "azimuth_rad",
            "azimuth_samples_rad",
            "azimuth",
            "theta_rad",
        ),
    )

    el = recursively_find_alias(
        data,
        (
            "elevation_rad",
            "elevation_samples_rad",
            "elevation",
            "phi_rad",
        ),
    )

    require(
        az is not None
        and el is not None,
        "Could not extract azimuth/elevation samples.",
    )

    az = np.asarray(
        az,
        dtype=float,
    )

    el = np.asarray(
        el,
        dtype=float,
    )

    require(
        az.ndim == 1
        and el.ndim == 1
        and az.shape == el.shape,
        "Invalid receiver-angle sample arrays.",
    )

    return az, el


def object_dict(obj: Any) -> dict:
    data = json_safe(obj)

    require(
        isinstance(data, dict),
        f"Expected object/dict, got {type(data)}.",
    )

    return data


def posterior_probabilities(
    posterior: Any,
) -> list[float]:
    data = object_dict(posterior)

    value = recursively_find_alias(
        data,
        (
            "probabilities",
            "beam_probabilities",
            "probability_mass",
            "masses",
        ),
    )

    require(
        value is not None,
        "BeamPosterior probability vector not found.",
    )

    probs = [
        float(x)
        for x in value
    ]

    require(
        len(probs) > 0,
        "Empty beam probability vector.",
    )

    return probs


def outside_probability(
    posterior: Any,
) -> float:
    data = object_dict(posterior)

    value = recursively_find_alias(
        data,
        (
            "outside_support_probability",
            "outside_support_mass",
            "outside_probability",
        ),
    )

    require(
        value is not None,
        "Outside-support probability not found.",
    )

    return float(value)


def selection_indices(
    selection: Any,
) -> list[int]:
    data = object_dict(selection)

    value = recursively_find_alias(
        data,
        (
            "selected_indices",
            "selected_beam_indices",
            "beam_indices",
            "topk_indices",
            "indices",
        ),
    )

    require(
        value is not None,
        "AdaptiveSelection selected indices not found.",
    )

    return [
        int(x)
        for x in value
    ]


def primary_index(
    selection: Any,
) -> int:
    data = object_dict(selection)

    value = recursively_find_alias(
        data,
        (
            "primary_beam_index",
            "primary_index",
            "selected_primary_index",
            "hysteresis_primary_index",
        ),
    )

    if value is not None:
        return int(value)

    selected = selection_indices(
        selection
    )

    require(
        len(selected) > 0,
        "Selection has no primary beam.",
    )

    return int(selected[0])


def beam_record(
    beam: Any,
) -> dict:
    data = object_dict(beam)

    index = recursively_find_alias(
        data,
        ("index", "beam_index"),
    )

    center = recursively_find_alias(
        data,
        (
            "center_rad",
            "center_azimuth_rad",
            "center",
        ),
    )

    lower = recursively_find_alias(
        data,
        (
            "lower_rad",
            "lower_azimuth_rad",
            "lower",
        ),
    )

    upper = recursively_find_alias(
        data,
        (
            "upper_rad",
            "upper_azimuth_rad",
            "upper",
        ),
    )

    width = recursively_find_alias(
        data,
        (
            "width_rad",
            "beamwidth_rad",
            "width",
        ),
    )

    require(
        center is not None
        and lower is not None
        and upper is not None,
        "Beam geometry fields not found.",
    )

    if width is None:
        width = (
            float(upper)
            - float(lower)
        )

    codebook_size = recursively_find_alias(
        data,
        (
            "codebook_size",
            "size",
        ),
    )

    provenance = recursively_find_alias(
        data,
        (
            "provenance",
            "reason",
            "source",
        ),
    )

    return {
        "index": (
            None
            if index is None
            else int(index)
        ),
        "center_rad": float(center),
        "lower_rad": float(lower),
        "upper_rad": float(upper),
        "width_rad": float(width),
        "codebook_size": (
            None
            if codebook_size is None
            else int(codebook_size)
        ),
        "provenance": (
            None
            if provenance is None
            else str(provenance)
        ),
    }


def extract_plan_probes(
    plan: Any,
) -> list[Any]:
    for name in (
        "probes",
        "probe_sequence",
        "probe_plan",
    ):
        if hasattr(plan, name):
            value = getattr(plan, name)

            if callable(value):
                value = value()

            if isinstance(
                value,
                (tuple, list),
            ):
                return list(value)

    data = object_dict(plan)

    value = recursively_find_alias(
        data,
        (
            "probes",
            "probe_sequence",
        ),
    )

    require(
        isinstance(value, list),
        "ProbePlan probe sequence not found.",
    )

    return value


def probe_to_beam(
    probe: Any,
) -> Any:
    if hasattr(probe, "beam"):
        return getattr(probe, "beam")

    if isinstance(probe, dict):
        if "beam" in probe:
            return probe["beam"]

        # Some implementations serialize Beam directly.
        if any(
            key in probe
            for key in (
                "center_rad",
                "center_azimuth_rad",
            )
        ):
            return probe

    data = object_dict(probe)

    beam = recursively_find_alias(
        data,
        ("beam",),
    )

    require(
        beam is not None,
        "Could not resolve Probe -> Beam.",
    )

    return beam


# ============================================================
# Decision pass
# ============================================================

def validate_stage4_row(
    row: dict,
) -> None:
    required = {
        "scenario_id",
        "prediction_id",
        "formal_rank",
        "origin_H0_m",
        "gaussian_mean_H0_m",
        "gaussian_covariance_calibrated_H0_m2",
        "input_contract",
    }

    require(
        required.issubset(row),
        (
            "Stage4 prediction-ledger row missing "
            f"{required - set(row)}"
        ),
    )

    mean = np.asarray(
        row["gaussian_mean_H0_m"],
        dtype=float,
    )

    cov = np.asarray(
        row[
            "gaussian_covariance_"
            "calibrated_H0_m2"
        ],
        dtype=float,
    )

    require(
        mean.shape == (4, 3),
        f"Bad Gaussian mean shape {mean.shape}.",
    )

    require(
        cov.shape == (4, 3, 3),
        f"Bad covariance shape {cov.shape}.",
    )

    require(
        np.isfinite(mean).all()
        and np.isfinite(cov).all(),
        "Non-finite Stage4 prediction.",
    )

    for c in cov:
        require(
            np.allclose(
                c,
                c.T,
                atol=1e-10,
                rtol=0.0,
            ),
            "Stage4 covariance is not symmetric.",
        )

        eig = np.linalg.eigvalsh(c)

        require(
            float(np.min(eig)) >= -1e-9,
            "Stage4 covariance not PSD.",
        )

    contract = row["input_contract"]

    require(
        contract.get(
            "direct_Stage2_noisy_vr"
        ) is True,
        "Stage4 direct noisy v_r contract failed.",
    )

    require(
        contract.get(
            "measurement_R_input"
        ) is True,
        "Stage4 measurement-R contract failed.",
    )

    require(
        contract.get(
            "future_truth_input"
        ) is False,
        "Future truth entered Stage4.",
    )

    require(
        contract.get(
            "tracks_to_predict_input"
        ) is False,
        "tracks_to_predict entered Stage4.",
    )

    require(
        contract.get(
            "annotated_velocity_input"
        ) is False,
        "Annotated velocity entered primary Stage4.",
    )


def build_decision_rows(
    ledger_map: dict[
        tuple[str, str],
        dict,
    ],
    causal_rows: list[dict],
    horizons: list[float],
    protocol: dict,
) -> list[dict]:

    from iscai_stage5.fullpdf_v2 import (
        adaptive_topk,
        beam_probability_mass,
        build_codebook,
        receiver_angles,
        receiver_geometry,
        receiver_samples_h0,
        recovery_probe_plan,
    )

    modes = list(
        protocol[
            "receiver_geometry"
        ]["modes"]
    )

    codebook_sizes = [
        int(x)
        for x
        in protocol["codebooks"]["sizes"]
    ]

    q_targets = [
        float(x)
        for x
        in protocol[
            "adaptive_TopK"
        ]["coverage_targets"]
    ]

    sample_count = int(
        protocol[
            "angular_posterior"
        ]["sample_count"]
    )

    rows: list[dict] = []

    for causal in sorted(
        causal_rows,
        key=lambda x: (
            x["scenario_id"],
            x["prediction_id"],
        ),
    ):
        key = (
            causal["scenario_id"],
            causal["prediction_id"],
        )

        require(
            key in ledger_map,
            f"Causal receiver {key} not in Stage4 ledger.",
        )

        pred = ledger_map[key]

        mean = np.asarray(
            pred["gaussian_mean_H0_m"],
            dtype=float,
        )

        covariance = np.asarray(
            pred[
                "gaussian_covariance_"
                "calibrated_H0_m2"
            ],
            dtype=float,
        )

        length = float(
            causal["length_m"]
        )

        width = float(
            causal["width_m"]
        )

        height = float(
            causal["height_m"]
        )

        heading = float(
            causal["heading_h0_rad"]
        )

        for mode in modes:
            geometry = receiver_geometry(
                mode=mode,
                length_m=length,
                width_m=width,
                height_m=height,
            )

            for codebook_size in codebook_sizes:
                codebook = build_codebook(
                    codebook_size
                )

                previous_by_q: dict[
                    float,
                    int | None,
                ] = {
                    q: None
                    for q in q_targets
                }

                for hidx, horizon in enumerate(
                    horizons
                ):
                    sample_key = (
                        f"{causal['scenario_id']}|"
                        f"{causal['prediction_id']}|"
                        f"{mode}|"
                        f"{codebook_size}|"
                        f"h{hidx}"
                    )

                    samples = receiver_samples_h0(
                        actor_mean_h0_m=(
                            mean[hidx]
                        ),
                        already_calibrated_predictive_covariance_h0_m2=(
                            covariance[hidx]
                        ),
                        heading_h0_rad=heading,
                        geometry=geometry,
                        length_m=length,
                        width_m=width,
                        height_m=height,
                        sample_key=sample_key,
                        sample_count=sample_count,
                    )

                    azimuth, elevation = (
                        extract_az_el(
                            receiver_angles(
                                samples
                            )
                        )
                    )

                    require(
                        len(azimuth)
                        == sample_count,
                        "MC sample-count mismatch.",
                    )

                    posterior = (
                        beam_probability_mass(
                            azimuth_samples_rad=azimuth,
                            codebook=codebook,
                        )
                    )

                    probs = (
                        posterior_probabilities(
                            posterior
                        )
                    )

                    require(
                        len(probs)
                        == codebook_size,
                        "Beam probability length mismatch.",
                    )

                    outside = (
                        outside_probability(
                            posterior
                        )
                    )

                    total = (
                        sum(probs)
                        + outside
                    )

                    require(
                        abs(total - 1.0)
                        <= 1e-12,
                        (
                            "Beam mass + outside mass "
                            f"!=1: {total}"
                        ),
                    )

                    for q in q_targets:
                        previous = (
                            previous_by_q[q]
                        )

                        selection = adaptive_topk(
                            posterior=posterior,
                            requested_mass=q,
                            previous_primary_index=(
                                previous
                            ),
                        )

                        selected = (
                            selection_indices(
                                selection
                            )
                        )

                        require(
                            len(selected) > 0,
                            "Empty adaptive Top-K selection.",
                        )

                        require(
                            all(
                                0 <= i
                                < codebook_size
                                for i in selected
                            ),
                            "Invalid selected beam index.",
                        )

                        # Verify minimal-prefix mass contract directly
                        # from the frozen posterior probabilities.
                        order = sorted(
                            range(
                                codebook_size
                            ),
                            key=lambda i: (
                                -probs[i],
                                i,
                            ),
                        )

                        cumulative = 0.0
                        expected_min_k = None

                        for k, idx in enumerate(
                            order,
                            start=1,
                        ):
                            cumulative += (
                                probs[idx]
                            )

                            if (
                                cumulative
                                + 1e-15
                                >= q
                            ):
                                expected_min_k = k
                                break

                        if expected_min_k is not None:
                            require(
                                len(selected)
                                == expected_min_k,
                                (
                                    "adaptive_topk did not return "
                                    "minimum mass-covering K."
                                ),
                            )

                        plan = (
                            recovery_probe_plan(
                                selection=selection,
                                codebook=codebook,
                            )
                        )

                        probes = (
                            extract_plan_probes(
                                plan
                            )
                        )

                        charged_count = int(
                            plan.probing_beam_count()
                        )

                        require(
                            charged_count
                            == len(probes),
                            (
                                "ProbePlan charge count does "
                                "not equal actual probe list."
                            ),
                        )

                        selected_beams = [
                            beam_record(
                                codebook[i]
                            )
                            for i in selected
                        ]

                        charged_probes = [
                            {
                                "ordinal": i,
                                "beam": beam_record(
                                    probe_to_beam(
                                        probe
                                    )
                                ),
                                "probe": json_safe(
                                    probe
                                ),
                            }
                            for i, probe in enumerate(
                                probes
                            )
                        ]

                        primary = (
                            primary_index(
                                selection
                            )
                        )

                        previous_by_q[q] = (
                            primary
                        )

                        rows.append(
                            {
                                "schema": (
                                    "stage5_fullpdf_v2_"
                                    "causal_decision_v1"
                                ),
                                "scenario_id": (
                                    causal[
                                        "scenario_id"
                                    ]
                                ),
                                "prediction_id": (
                                    causal[
                                        "prediction_id"
                                    ]
                                ),
                                "formal_rank": (
                                    pred[
                                        "formal_rank"
                                    ]
                                ),
                                "receiver_mode": mode,
                                "codebook_size": (
                                    codebook_size
                                ),
                                "horizon_index": (
                                    hidx
                                ),
                                "horizon_s": (
                                    float(horizon)
                                ),
                                "requested_mass": (
                                    float(q)
                                ),
                                "sample_count": (
                                    sample_count
                                ),
                                "sample_key": (
                                    sample_key
                                ),
                                "previous_primary_index": (
                                    previous
                                ),
                                "primary_index": (
                                    primary
                                ),
                                "posterior": {
                                    "beam_probabilities": (
                                        probs
                                    ),
                                    "outside_support_probability": (
                                        outside
                                    ),
                                    "total_probability": (
                                        total
                                    ),
                                    "azimuth_sample_sha256": (
                                        hashlib.sha256(
                                            np.asarray(
                                                azimuth,
                                                dtype=np.float64,
                                            ).tobytes()
                                        ).hexdigest()
                                    ),
                                },
                                "selected_indices": (
                                    selected
                                ),
                                "selected_beams": (
                                    selected_beams
                                ),
                                "charged_probe_count": (
                                    charged_count
                                ),
                                "charged_probes": (
                                    charged_probes
                                ),
                                "selection": (
                                    json_safe(
                                        selection
                                    )
                                ),
                                "probe_plan": (
                                    json_safe(
                                        plan
                                    )
                                ),
                                "causal_actor_geometry": {
                                    "heading_h0_rad": (
                                        heading
                                    ),
                                    "length_m": length,
                                    "width_m": width,
                                    "height_m": height,
                                },
                                "future_truth_used": (
                                    False
                                ),
                                "tracks_to_predict_used": (
                                    False
                                ),
                                "Stage4_covariance_recalibrated": (
                                    False
                                ),
                            }
                        )

    return rows


# ============================================================
# Truth-side parsing — CALLED ONLY AFTER SEAL
# ============================================================

def select_horizon_value(
    value: Any,
    hidx: int,
    horizon: float,
) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        candidates = (
            str(horizon),
            f"{horizon:.1f}",
            f"{horizon:.3f}",
            str(hidx),
            f"h{hidx}",
        )

        for key in candidates:
            if key in value:
                return value[key]

        return None

    if isinstance(value, list):
        if len(value) == 4:
            return value[hidx]

    return value


def truth_source_for_mode(
    row: dict,
    mode: str,
) -> dict:
    root = recursively_find_alias(
        row,
        (
            "receiver_truth_by_mode",
            "truth_by_receiver_mode",
            "receiver_modes",
            "evaluator_receiver_truth",
        ),
    )

    if isinstance(root, dict):
        aliases = {
            "centroid": (
                "centroid",
                "centroid_baseline",
            ),
            "known": (
                "known",
                "known_receiver_offset",
            ),
            "uncertain": (
                "uncertain",
                "uncertain_receiver_offset",
            ),
        }

        for alias in aliases[mode]:
            if alias in root:
                value = root[alias]

                if isinstance(value, dict):
                    return value

    return row


def truth_point_or_angle(
    rows: list[dict],
    decision: dict,
) -> tuple[float, float] | None:

    mode = str(
        decision["receiver_mode"]
    )

    hidx = int(
        decision["horizon_index"]
    )

    horizon = float(
        decision["horizon_s"]
    )

    # Prefer a row explicitly carrying this horizon.
    ordered = sorted(
        rows,
        key=lambda r: (
            0
            if abs(
                float(
                    recursively_find_alias(
                        r,
                        (
                            "horizon_s",
                            "prediction_horizon_s",
                        ),
                    )
                    or -999.0
                )
                - horizon
            )
            <= 1e-6
            else 1
        ),
    )

    for row in ordered:
        source = truth_source_for_mode(
            row,
            mode,
        )

        valid = recursively_find_alias(
            source,
            (
                "valid",
                "future_valid",
                "receiver_valid",
            ),
        )

        valid = select_horizon_value(
            valid,
            hidx,
            horizon,
        )

        if valid is False:
            return None

        az = recursively_find_alias(
            source,
            (
                "receiver_azimuth_rad",
                "azimuth_rad",
                "truth_azimuth_rad",
            ),
        )

        el = recursively_find_alias(
            source,
            (
                "receiver_elevation_rad",
                "elevation_rad",
                "truth_elevation_rad",
            ),
        )

        az = select_horizon_value(
            az,
            hidx,
            horizon,
        )

        el = select_horizon_value(
            el,
            hidx,
            horizon,
        )

        if (
            az is not None
            and el is not None
            and np.isscalar(az)
            and np.isscalar(el)
        ):
            return (
                float(az),
                float(el),
            )

        receiver_point = recursively_find_alias(
            source,
            (
                "receiver_point_H0_m",
                "receiver_position_H0_m",
                "receiver_truth_H0_m",
                "future_receiver_H0_m",
            ),
        )

        receiver_point = select_horizon_value(
            receiver_point,
            hidx,
            horizon,
        )

        if receiver_point is not None:
            arr = np.asarray(
                receiver_point,
                dtype=float,
            )

            if arr.shape == (3,):
                return point_to_angles(
                    arr
                )

        # Generic future actor center.
        center = recursively_find_alias(
            row,
            (
                "future_center_H0_m",
                "future_position_H0_m",
                "future_positions_H0_m",
                "future_centers_H0_m",
                "future_actor_center_H0_m",
            ),
        )

        center = select_horizon_value(
            center,
            hidx,
            horizon,
        )

        if center is None:
            continue

        center = np.asarray(
            center,
            dtype=float,
        )

        if center.shape != (3,):
            continue

        if mode == "centroid":
            return point_to_angles(
                center
            )

        heading = recursively_find_alias(
            row,
            (
                "future_heading_H0_rad",
                "future_heading_h0_rad",
                "future_yaw_H0_rad",
                "future_yaw_h0_rad",
            ),
        )

        heading = select_horizon_value(
            heading,
            hidx,
            horizon,
        )

        if heading is None:
            continue

        geometry = (
            decision[
                "causal_actor_geometry"
            ]
        )

        length = float(
            geometry["length_m"]
        )

        if mode == "known":
            offset = np.asarray(
                [
                    -length / 2.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        elif mode == "uncertain":
            # A realized uncertain receiver placement MUST
            # have been frozen pre-formal. We do not invent
            # a new evaluator draw after seeing outcomes.
            offset = recursively_find_alias(
                row,
                (
                    "realized_receiver_offset_body_m",
                    "uncertain_receiver_offset_body_m",
                    "frozen_receiver_offset_body_m",
                ),
            )

            if offset is None:
                continue

            offset = np.asarray(
                offset,
                dtype=float,
            )

            if offset.shape != (3,):
                continue

        else:
            continue

        point = (
            center
            + rotation_z(
                float(heading)
            )
            @ offset
        )

        return point_to_angles(
            point
        )

    return None


def resolve_truth_binding(
    decision_keys: set[tuple[str, str]],
) -> tuple[Path, dict[
    tuple[str, str],
    list[dict],
]]:
    global FORMAL_TRUTH_OPENED

    require(
        DECISION_LEDGER_SEALED,
        (
            "Evaluator truth access attempted "
            "before decision-ledger seal."
        ),
    )

    FORMAL_TRUTH_OPENED = True

    event(
        "EVALUATOR_TRUTH_BOUNDARY_OPENED",
        after_decision_seal=True,
    )

    valid: list[
        tuple[
            Path,
            str,
            dict[tuple[str, str], list[dict]],
        ]
    ] = []

    for path in candidate_v2_paths(
        truth_phase=True
    ):
        try:
            rows = load_records(path)
        except Exception:
            continue

        grouped: dict[
            tuple[str, str],
            list[dict],
        ] = defaultdict(list)

        for row in rows:
            sid = recursively_find_alias(
                row,
                ("scenario_id",),
            )

            pid = recursively_find_alias(
                row,
                ("prediction_id",),
            )

            if sid is None or pid is None:
                continue

            key = (
                str(sid),
                str(pid),
            )

            if key in decision_keys:
                grouped[key].append(row)

        if not grouped:
            continue

        # Must contain actual evaluator/future endpoint data.
        sample_text = canonical_json(
            [
                json_safe(x)
                for values in list(
                    grouped.values()
                )[:5]
                for x in values[:2]
            ]
        ).lower()

        if not any(
            token in sample_text
            for token in (
                "future",
                "truth",
                "receiver_azimuth",
                "receiver_point",
                "receiver_position",
            )
        ):
            continue

        valid.append(
            (
                path,
                sha256_file(path),
                grouped,
            )
        )

    require(
        len(valid) > 0,
        (
            "No frozen V2 evaluator-truth binding "
            "could be resolved after sealing decisions."
        ),
    )

    by_sha: dict[
        str,
        list,
    ] = defaultdict(list)

    for item in valid:
        by_sha[item[1]].append(item)

    require(
        len(by_sha) == 1,
        (
            "Multiple scientifically distinct evaluator "
            f"truth bindings found: {[str(x[0]) for x in valid]}"
        ),
    )

    group = next(
        iter(by_sha.values())
    )

    chosen = sorted(
        group,
        key=lambda x: str(x[0]),
    )[0]

    return (
        chosen[0],
        chosen[2],
    )


# ============================================================
# Beam hit
# ============================================================

def beam_contains(
    beam: dict,
    azimuth: float,
) -> bool:
    lower = float(
        beam["lower_rad"]
    )

    upper = float(
        beam["upper_rad"]
    )

    value = float(
        azimuth
    )

    # Final cell includes its right boundary.
    return (
        lower <= value < upper
        or abs(
            value - upper
        )
        <= 1e-15
    )


# ============================================================
# Dynamic frozen optical-chain binding
# ============================================================

def invoke_supported(
    fn: Any,
    values: dict[str, Any],
) -> Any:
    signature = inspect.signature(
        fn
    )

    kwargs = {}

    for name, parameter in (
        signature.parameters.items()
    ):
        if name in values:
            kwargs[name] = values[name]
            continue

        if (
            parameter.default
            is not inspect.Parameter.empty
        ):
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        raise FailClosed(
            f"Unsupported required parameter "
            f"{name!r} for {fn.__module__}.{fn.__name__}"
        )

    return fn(**kwargs)


def find_optical_function(
    module: Any,
    *,
    name_tokens: tuple[str, ...],
    parameter_tokens: tuple[str, ...],
) -> Any:
    candidates = []

    for name, value in inspect.getmembers(
        module,
        inspect.isfunction,
    ):
        low = name.lower()

        if not all(
            token.lower() in low
            for token in name_tokens
        ):
            continue

        params = {
            x.lower()
            for x
            in inspect.signature(
                value
            ).parameters
        }

        if not all(
            any(
                token.lower() in p
                for p in params
            )
            for token in parameter_tokens
        ):
            continue

        candidates.append(value)

    require(
        len(candidates) == 1,
        (
            "Could not resolve one unique optical "
            f"function {name_tokens}/{parameter_tokens}; "
            f"found {[x.__name__ for x in candidates]}"
        ),
    )

    return candidates[0]


def frozen_optical_chain(
    beam: dict,
    receiver_azimuth_rad: float,
    receiver_elevation_rad: float,
    charged_probe_count: int,
) -> dict:

    import iscai_stage5.optical_link as optical
    import iscai_stage5.beam_latency as latency

    pointing = optical.pointing_error(
        receiver_azimuth_rad=(
            receiver_azimuth_rad
        ),
        receiver_elevation_rad=(
            receiver_elevation_rad
        ),
        beam_center_azimuth_rad=(
            float(beam["center_rad"])
        ),
        beam_center_elevation_rad=0.0,
    )

    delta_az = scalar_from(
        pointing,
        (
            "delta_azimuth_rad",
            "azimuth_error_rad",
        ),
    )

    delta_el = scalar_from(
        pointing,
        (
            "delta_elevation_rad",
            "elevation_error_rad",
        ),
    )

    gain_values = {
        "delta_azimuth_rad": (
            delta_az
        ),
        "delta_elevation_rad": (
            delta_el
        ),
        "azimuth_hpbw_rad": (
            float(beam["width_rad"])
        ),
        # Existing Stage5 constructed model uses
        # equal elevation/azimuth HPBW when required.
        "elevation_hpbw_rad": (
            float(beam["width_rad"])
        ),
    }

    gain = float(
        invoke_supported(
            optical.constructed_optical_pointing_gain,
            gain_values,
        )
    )

    received = invoke_supported(
        optical.normalized_received_power,
        {
            "optical_gain": gain,
        },
    )

    received_value = scalar_from(
        received,
        (
            "received_power_normalized",
            "normalized_received_power",
            "power",
            "value",
        ),
    )

    snr_fn = find_optical_function(
        optical,
        name_tokens=("snr",),
        parameter_tokens=("received_power",),
    )

    snr = invoke_supported(
        snr_fn,
        {
            "received_power_normalized": (
                received_value
            ),
            "normalized_received_power": (
                received_value
            ),
            "reference_snr_linear": getattr(
                optical,
                "PARTA_REFERENCE_SNR_LINEAR",
                None,
            ),
        },
    )

    snr_linear = scalar_from(
        snr,
        (
            "snr_linear",
            "linear",
            "snr",
        ),
    )

    try:
        snr_db = scalar_from(
            snr,
            (
                "snr_db",
                "db",
            ),
        )
    except FailClosed:
        snr_db = (
            -math.inf
            if snr_linear <= 0
            else 10.0
            * math.log10(
                snr_linear
            )
        )

    ber_fn = find_optical_function(
        optical,
        name_tokens=("ber",),
        parameter_tokens=("snr",),
    )

    ber_obj = invoke_supported(
        ber_fn,
        {
            "snr_linear": snr_linear,
            "snr": snr_linear,
        },
    )

    if isinstance(
        ber_obj,
        (int, float, np.number),
    ):
        ber = float(ber_obj)
    else:
        ber = scalar_from(
            ber_obj,
            (
                "ber",
                "bit_error_rate",
            ),
        )

    probe_time = float(
        latency.beam_probing_time_s(
            charged_probe_count
        )
    )

    rate_fn = getattr(
        latency,
        "effective_rate_with_frozen_timing",
        None,
    )

    require(
        callable(rate_fn),
        (
            "Frozen effective_rate_with_frozen_timing "
            "API unavailable."
        ),
    )

    rate_obj = invoke_supported(
        rate_fn,
        {
            "BER": ber,
            "ber": ber,
            "probing_beam_count": (
                charged_probe_count
            ),
            "probe_count": (
                charged_probe_count
            ),
            "charged_beam_count": (
                charged_probe_count
            ),
        },
    )

    rate = scalar_from(
        rate_obj,
        (
            "effective_rate_bps",
            "effective_rate",
            "rate_bps",
        ),
    )

    total_latency = None

    budget_fn = getattr(
        latency,
        "build_latency_budget",
        None,
    )

    if callable(budget_fn):
        try:
            budget = invoke_supported(
                budget_fn,
                {
                    "probing_beam_count": (
                        charged_probe_count
                    ),
                    "probe_count": (
                        charged_probe_count
                    ),
                    "charged_beam_count": (
                        charged_probe_count
                    ),
                },
            )

            total_latency = scalar_from(
                budget,
                (
                    "total_latency_s",
                    "total_frame_latency_s",
                    "total_s",
                ),
            )
        except FailClosed:
            total_latency = None

    return {
        "pointing_error_rad": {
            "azimuth": delta_az,
            "elevation": delta_el,
        },
        "optical_gain": gain,
        "received_power_normalized": (
            received_value
        ),
        "snr_linear": snr_linear,
        "snr_db": snr_db,
        "DPSK_BER": ber,
        "effective_rate_bps": rate,
        "beam_probing_latency_s": (
            probe_time
        ),
        "total_latency_s": (
            total_latency
        ),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_sealed_decisions(
    decisions: list[dict],
    truth: dict[
        tuple[str, str],
        list[dict],
    ],
    protocol: dict,
) -> tuple[list[dict], dict]:

    rows: list[dict] = []

    for decision in decisions:
        key = (
            decision["scenario_id"],
            decision["prediction_id"],
        )

        truth_rows = truth.get(
            key,
            [],
        )

        truth_angle = (
            truth_point_or_angle(
                truth_rows,
                decision,
            )
        )

        if truth_angle is None:
            rows.append(
                {
                    "scenario_id": key[0],
                    "prediction_id": key[1],
                    "receiver_mode": (
                        decision[
                            "receiver_mode"
                        ]
                    ),
                    "codebook_size": (
                        decision[
                            "codebook_size"
                        ]
                    ),
                    "horizon_s": (
                        decision["horizon_s"]
                    ),
                    "requested_mass": (
                        decision[
                            "requested_mass"
                        ]
                    ),
                    "future_endpoint_valid": (
                        False
                    ),
                }
            )

            continue

        true_azimuth, true_elevation = (
            truth_angle
        )

        selected_hit = any(
            beam_contains(
                beam,
                true_azimuth,
            )
            for beam
            in decision["selected_beams"]
        )

        charged_beams = [
            probe["beam"]
            for probe
            in decision[
                "charged_probes"
            ]
        ]

        plan_hit = any(
            beam_contains(
                beam,
                true_azimuth,
            )
            for beam in charged_beams
        )

        charged = int(
            decision[
                "charged_probe_count"
            ]
        )

        require(
            charged == len(charged_beams),
            "Evaluator attempted hidden/free beam.",
        )

        optical_results = [
            frozen_optical_chain(
                beam,
                true_azimuth,
                true_elevation,
                charged,
            )
            for beam in charged_beams
        ]

        require(
            len(optical_results) == charged,
            (
                "Optical evaluator count != "
                "charged probe count."
            ),
        )

        # Best realized link is selected ONLY among
        # beams present in the sealed charged plan.
        best_optical = max(
            optical_results,
            key=lambda x: (
                float(
                    x[
                        "effective_rate_bps"
                    ]
                ),
                float(
                    x[
                        "optical_gain"
                    ]
                ),
            ),
        )

        rows.append(
            {
                "scenario_id": key[0],
                "prediction_id": key[1],
                "receiver_mode": (
                    decision["receiver_mode"]
                ),
                "codebook_size": (
                    int(
                        decision[
                            "codebook_size"
                        ]
                    )
                ),
                "horizon_s": (
                    float(
                        decision[
                            "horizon_s"
                        ]
                    )
                ),
                "requested_mass": (
                    float(
                        decision[
                            "requested_mass"
                        ]
                    )
                ),
                "future_endpoint_valid": True,
                "truth_azimuth_rad": (
                    true_azimuth
                ),
                "truth_elevation_rad": (
                    true_elevation
                ),
                "mass_covering_set_hit": (
                    bool(selected_hit)
                ),
                "actual_charged_plan_hit": (
                    bool(plan_hit)
                ),
                "charged_probe_count": (
                    charged
                ),
                "free_probe_count": 0,
                "probing_overhead_fraction": (
                    charged
                    / int(
                        decision[
                            "codebook_size"
                        ]
                    )
                ),
                "outside_support_probability": (
                    decision[
                        "posterior"
                    ][
                        "outside_support_probability"
                    ]
                ),
                "best_charged_optical_link": (
                    best_optical
                ),
            }
        )

    expected_modes = list(
        protocol[
            "receiver_geometry"
        ]["modes"]
    )

    expected_codebooks = [
        int(x)
        for x
        in protocol["codebooks"]["sizes"]
    ]

    expected_q = [
        float(x)
        for x
        in protocol[
            "adaptive_TopK"
        ]["coverage_targets"]
    ]

    horizons = sorted(
        {
            float(x["horizon_s"])
            for x in decisions
        }
    )

    grouped: dict[
        tuple[str, int, float, float],
        list[dict],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            (
                row["receiver_mode"],
                int(
                    row["codebook_size"]
                ),
                float(row["horizon_s"]),
                float(
                    row["requested_mass"]
                ),
            )
        ].append(row)

    strata = []

    coverage_all = True
    overhead_all = True
    latency_all = True

    for mode in expected_modes:
        for size in expected_codebooks:
            for horizon in horizons:
                for q in expected_q:
                    key = (
                        mode,
                        size,
                        horizon,
                        q,
                    )

                    group = grouped.get(
                        key,
                        [],
                    )

                    valid = [
                        r
                        for r in group
                        if r.get(
                            "future_endpoint_valid"
                        )
                    ]

                    n = len(valid)

                    if n == 0:
                        empirical = None
                        mean_overhead = None
                        mean_probe_latency = None

                        coverage_pass = False
                        overhead_pass = False
                        latency_pass = False

                    else:
                        hits = sum(
                            bool(
                                r[
                                    "mass_covering_set_hit"
                                ]
                            )
                            for r in valid
                        )

                        empirical = (
                            hits / n
                        )

                        mean_overhead = float(
                            np.mean(
                                [
                                    r[
                                        "probing_overhead_fraction"
                                    ]
                                    for r in valid
                                ]
                            )
                        )

                        mean_probe_latency = float(
                            np.mean(
                                [
                                    r[
                                        "best_charged_optical_link"
                                    ][
                                        "beam_probing_latency_s"
                                    ]
                                    for r in valid
                                ]
                            )
                        )

                        # No tolerance invented after outcome:
                        # strict empirical >= requested q.
                        coverage_pass = (
                            empirical >= q
                        )

                        # Exhaustive baseline fraction = 1.
                        overhead_pass = (
                            mean_overhead < 1.0
                        )

                        latency_pass = (
                            math.isfinite(
                                mean_probe_latency
                            )
                            and mean_probe_latency
                            >= 0.0
                        )

                    coverage_all &= (
                        coverage_pass
                    )

                    overhead_all &= (
                        overhead_pass
                    )

                    latency_all &= (
                        latency_pass
                    )

                    strata.append(
                        {
                            "receiver_mode": mode,
                            "codebook_size": size,
                            "horizon_s": horizon,
                            "requested_mass": q,
                            "valid_endpoints": n,
                            "empirical_coverage": (
                                empirical
                            ),
                            "coverage_pass": (
                                coverage_pass
                            ),
                            "mean_actual_charged_overhead_fraction": (
                                mean_overhead
                            ),
                            "overhead_reduction_vs_exhaustive_pass": (
                                overhead_pass
                            ),
                            "mean_beam_probing_latency_s": (
                                mean_probe_latency
                            ),
                            "latency_pass": (
                                latency_pass
                            ),
                        }
                    )

    valid_rows = [
        r
        for r in rows
        if r.get(
            "future_endpoint_valid"
        )
    ]

    no_free_probes = (
        len(valid_rows) > 0
        and all(
            int(
                r[
                    "free_probe_count"
                ]
            )
            == 0
            for r in valid_rows
        )
    )

    optical_complete = (
        len(valid_rows) > 0
    )

    optical_fields = (
        "optical_gain",
        "received_power_normalized",
        "snr_linear",
        "snr_db",
        "DPSK_BER",
        "effective_rate_bps",
        "beam_probing_latency_s",
    )

    for row in valid_rows:
        link = row[
            "best_charged_optical_link"
        ]

        for field in optical_fields:
            if field not in link:
                optical_complete = False
                continue

            value = float(
                link[field]
            )

            if field != "snr_db":
                if not math.isfinite(value):
                    optical_complete = False

    summary = {
        "valid_evaluator_rows": (
            len(valid_rows)
        ),
        "total_decision_rows": (
            len(decisions)
        ),
        "strata": strata,
        "requested_coverage_all_strata_pass": (
            coverage_all
        ),
        "overhead_reduction_vs_exhaustive_all_strata_pass": (
            overhead_all
        ),
        "latency_all_strata_pass": (
            latency_all
        ),
        "no_free_probes": (
            no_free_probes
        ),
        "full_optical_chain_complete": (
            optical_complete
        ),
        "recovery_event_counts": {
            "local_neighbor": sum(
                any(
                    (
                        p["beam"].get(
                            "provenance"
                        )
                        == "local_neighbor_recovery"
                    )
                    for p in d[
                        "charged_probes"
                    ]
                )
                for d in decisions
            ),
            "widened_fallback": sum(
                any(
                    (
                        p["beam"].get(
                            "provenance"
                        )
                        == "widened_fallback"
                    )
                    for p in d[
                        "charged_probes"
                    ]
                )
                for d in decisions
            ),
            "exhaustive_loss_of_lock": sum(
                any(
                    (
                        p["beam"].get(
                            "provenance"
                        )
                        == "exhaustive_loss_of_lock"
                    )
                    for p in d[
                        "charged_probes"
                    ]
                )
                for d in decisions
            ),
        },
        "note": (
            "Recovery mechanisms are structurally mandatory; "
            "a zero formal occurrence is not itself a failure "
            "when the frozen regression exercises the mechanism."
        ),
    }

    return rows, summary


# ============================================================
# Main
# ============================================================

def main() -> int:
    global DECISION_LEDGER_SEALED

    print(
        "STAGE5 FULLPDF V2 "
        "INDEPENDENT FORMAL CHECKER"
    )

    print(
        "FAIL-CLOSED / NO TRAINING / "
        "NO RETUNING"
    )

    # --------------------------------------------------------
    # Scientifically safe rerun admission
    # --------------------------------------------------------
    #
    # A prior technical checker failure may be rerun ONLY when
    # it is proven to have stopped before:
    #
    #   - decision-ledger seal,
    #   - evaluator future-truth opening,
    #   - formal outcome computation.
    #
    # This does not permit rerunning a completed/partially
    # evaluated formal experiment.
    #
    if REPORT.exists():
        previous_report = load_json_file(
            REPORT
        )

        require(
            previous_report.get(
                "formal_evaluator_truth_opened"
            )
            is not True,
            (
                "Prior checker run opened evaluator truth; "
                "formal rerun prohibited."
            ),
        )

        require(
            previous_report.get(
                "formal_outcomes_read"
            )
            is not True,
            (
                "Prior checker run read formal outcomes; "
                "formal rerun prohibited."
            ),
        )

        require(
            previous_report.get(
                "decision_ledger_sealed"
            )
            is not True,
            (
                "Prior checker run sealed a decision ledger; "
                "formal rerun prohibited."
            ),
        )

        require(
            not SEALED_LEDGER.exists(),
            (
                "Existing sealed decision ledger detected; "
                "formal rerun prohibited."
            ),
        )

        require(
            not SEAL_REPORT.exists(),
            (
                "Existing decision-ledger seal report detected; "
                "formal rerun prohibited."
            ),
        )

        require(
            not EVALUATOR_ROWS.exists(),
            (
                "Existing evaluator rows detected; "
                "formal rerun prohibited."
            ),
        )

        require(
            not CLOSURE.exists(),
            (
                "Existing final Stage5 closure detected; "
                "formal rerun prohibited."
            ),
        )

        require(
            not HANDOFF.exists(),
            (
                "Existing Stage5-to-Stage6 handoff detected; "
                "formal rerun prohibited."
            ),
        )

        event(
            "PRETRUTH_TECHNICAL_RERUN_ADMITTED",
            previous_report_sha256=(
                sha256_file(REPORT)
            ),
            previous_formal_truth_opened=False,
            previous_formal_outcomes_read=False,
            previous_decision_ledger_sealed=False,
        )

    require(
        not SEALED_LEDGER.exists(),
        (
            "A sealed V2 decision ledger already exists. "
            "Refusing to overwrite."
        ),
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    event(
        "FORMAL_CHECKER_STARTED",
        evaluator_truth_opened=False,
    )

    # --------------------------------------------------------
    # 1. Storage
    # --------------------------------------------------------

    free_bytes = shutil.disk_usage(
        ROOT
    ).free

    record_gate(
        "G01_storage_hard_reserve",
        free_bytes
        >= HARD_RESERVE_BYTES,
        {
            "free_bytes": free_bytes,
            "required_bytes": (
                HARD_RESERVE_BYTES
            ),
        },
    )

    # --------------------------------------------------------
    # 2. Exact frozen hashes
    # --------------------------------------------------------

    require(
        RUNTIME.is_file(),
        f"Missing runtime: {RUNTIME}",
    )

    require(
        PROTOCOL.is_file(),
        f"Missing protocol: {PROTOCOL}",
    )

    require(
        STAGE4_LEDGER.is_file(),
        (
            "Missing Stage4 formal ledger: "
            f"{STAGE4_LEDGER}"
        ),
    )

    runtime_sha = sha256_file(
        RUNTIME
    )

    protocol_sha = sha256_file(
        PROTOCOL
    )

    stage4_ledger_sha = sha256_file(
        STAGE4_LEDGER
    )

    record_gate(
        "G02_frozen_hashes",
        (
            runtime_sha
            == EXPECTED_RUNTIME_SHA
            and protocol_sha
            == EXPECTED_PROTOCOL_SHA
            and stage4_ledger_sha
            == EXPECTED_STAGE4_LEDGER_SHA
        ),
        {
            "runtime": runtime_sha,
            "protocol": protocol_sha,
            "Stage4_prediction_ledger": (
                stage4_ledger_sha
            ),
        },
    )

    protocol = load_json_file(
        PROTOCOL
    )

    # --------------------------------------------------------
    # 3. Protocol semantic freeze
    # --------------------------------------------------------

    protocol_semantics_pass = (
        protocol.get("stage") == 5
        and protocol.get("version")
        == "fullpdf_v2"
        and protocol.get("status")
        == "FROZEN_BEFORE_STAGE5_V2_FORMAL"
        and protocol[
            "formal"
        ][
            "parameter_tuning_on_formal"
        ]
        is False
        and protocol[
            "upstream"
        ][
            "Stage4_calibration_reapplied"
        ]
        is False
        and protocol[
            "angular_posterior"
        ][
            "trajectory_covariance_input"
        ]
        == (
            "already_calibrated_"
            "Stage4_predictive_covariance"
        )
        and protocol[
            "codebooks"
        ][
            "azimuth_clipping"
        ]
        is False
        and protocol[
            "codebooks"
        ][
            "outside_support_mass"
        ]
        == "explicit"
        and protocol[
            "adaptive_TopK"
        ][
            "all_evaluated_beams_charged"
        ]
        is True
        and protocol[
            "adaptive_TopK"
        ][
            "free_hidden_probes"
        ]
        is False
    )

    record_gate(
        "G03_protocol_preoutcome_freeze",
        protocol_semantics_pass,
        protocol,
    )

    # --------------------------------------------------------
    # 4. Static no-double-calibration audit
    # --------------------------------------------------------

    runtime_tree = ast.parse(
        RUNTIME.read_text(
            encoding="utf-8"
        )
    )

    forbidden_calls = []

    forbidden_names = (
        "apply_variance_scale",
        "calibrate_stage4_horizon_covariances",
        "covariance_scaler",
        "temperature_scale",
    )

    for node in ast.walk(
        runtime_tree
    ):
        if isinstance(node, ast.Call):
            try:
                text = ast.unparse(
                    node.func
                )
            except Exception:
                text = ""

            if any(
                token in text
                for token in forbidden_names
            ):
                forbidden_calls.append(
                    text
                )

    record_gate(
        "G04_no_Stage4_recalibration",
        len(forbidden_calls) == 0,
        {
            "forbidden_runtime_calls": (
                forbidden_calls
            ),
            "covariance_semantics": (
                "already calibrated Stage4 covariance"
            ),
        },
    )

    # --------------------------------------------------------
    # 5. Public causal API audit
    # --------------------------------------------------------

    forbidden_api_tokens = (
        "truth",
        "oracle",
        "tracks_to_predict",
        "objects_of_interest",
        "future_gt",
        "ground_truth",
    )

    bad_public_args = []

    import iscai_stage5.fullpdf_v2 as fpv2

    for name, obj in inspect.getmembers(
        fpv2,
        inspect.isfunction,
    ):
        if name.startswith("_"):
            continue

        args = [
            x.lower()
            for x
            in inspect.signature(
                obj
            ).parameters
        ]

        for arg in args:
            if any(
                token in arg
                for token in forbidden_api_tokens
            ):
                bad_public_args.append(
                    f"{name}:{arg}"
                )

    record_gate(
        "G05_public_controller_API_causal",
        len(bad_public_args) == 0,
        {
            "forbidden_args": (
                bad_public_args
            )
        },
    )

    # --------------------------------------------------------
    # 6. Stage4 closure/handoff exact provenance
    # --------------------------------------------------------

    expected_s4_closure_sha = (
        protocol["upstream"][
            "Stage4_closure_sha256"
        ]
    )

    expected_s4_handoff_sha = (
        protocol["upstream"][
            "Stage4_handoff_sha256"
        ]
    )

    s4_closure_path = (
        discover_exact_sha_file(
            expected_s4_closure_sha
        )
    )

    s4_handoff_path = (
        discover_exact_sha_file(
            expected_s4_handoff_sha
        )
    )

    s4_closure = load_json_file(
        s4_closure_path
    )

    s4_handoff = load_json_file(
        s4_handoff_path
    )

    combined_s4_text = (
        canonical_json(s4_closure)
        + canonical_json(s4_handoff)
    )

    record_gate(
        "G06_Stage4_COMPLETE_FROZEN_FULLPDF_V2",
        (
            "COMPLETE_FROZEN_FULLPDF_V2"
            in combined_s4_text
            and protocol[
                "upstream"
            ][
                "Stage4_status"
            ]
            == "COMPLETE_FROZEN_FULLPDF_V2"
        ),
        {
            "closure_path": (
                str(s4_closure_path)
            ),
            "closure_sha256": (
                expected_s4_closure_sha
            ),
            "handoff_path": (
                str(s4_handoff_path)
            ),
            "handoff_sha256": (
                expected_s4_handoff_sha
            ),
        },
    )


    # --------------------------------------------------------
    # Frozen prediction-horizon contract
    # --------------------------------------------------------
    #
    # The Stage4 Full-PDF-V2 closure/handoff intentionally do
    # not duplicate the numerical horizon vector. The vector
    # is already frozen upstream in Stage3 and is also the
    # authoritative PDF main short-horizon evaluation grid:
    #
    #   100 ms / 300 ms / 500 ms / 1 s
    #
    # This is an interface/provenance resolution only.
    # No Stage5 formal outcome has been opened.
    #
    stage3_formal_path = (
        ROOT
        / "iscai_stage3/reports/"
        "block38f_formal_evaluation.json"
    )

    require(
        stage3_formal_path.is_file(),
        (
            "Missing frozen Stage3 formal horizon authority: "
            f"{stage3_formal_path}"
        ),
    )

    stage3_formal = load_json_file(
        stage3_formal_path
    )

    stage3_horizons = stage3_formal.get(
        "prediction_horizons_s"
    )

    authoritative_horizons = [
        0.1,
        0.3,
        0.5,
        1.0,
    ]

    record_gate(
        "G06B_authoritative_horizon_contract",
        stage3_horizons
        == authoritative_horizons,
        {
            "authoritative_PDF_main_short_horizons_s":
                authoritative_horizons,
            "frozen_Stage3_formal_path":
                str(stage3_formal_path),
            "frozen_Stage3_formal_sha256":
                sha256_file(stage3_formal_path),
            "frozen_Stage3_prediction_horizons_s":
                stage3_horizons,
            "Stage4_future_step_count_expected":
                4,
            "post_outcome_selection":
                False,
        },
    )

    horizons = list(
        authoritative_horizons
    )

    EVIDENCE[
        "frozen_prediction_horizons_s"
    ] = {
        "values": horizons,
        "authority":
            "authoritative_PDF_plus_frozen_Stage3_formal_interface",
        "Stage4_V2_output_steps":
            4,
    }


    # --------------------------------------------------------
    # 7. Full regression
    # --------------------------------------------------------

    event(
        "REGRESSION_STARTED",
        formal_truth_opened=False,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S5 / "tests"),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=str(S5),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    REGRESSION_LOG.write_text(
        proc.stdout,
        encoding="utf-8",
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    count = (
        None
        if match is None
        else int(match.group(1))
    )

    record_gate(
        "G07_full_Stage5_regression_271",
        (
            proc.returncode == 0
            and count
            == EXPECTED_REGRESSION_TEST_COUNT
        ),
        {
            "returncode": (
                proc.returncode
            ),
            "tests": count,
            "expected": (
                EXPECTED_REGRESSION_TEST_COUNT
            ),
            "log": (
                str(REGRESSION_LOG)
            ),
        },
    )

    # --------------------------------------------------------
    # 8. Stage4 ledger validity
    # --------------------------------------------------------

    stage4_rows = load_records(
        STAGE4_LEDGER
    )

    require(
        len(stage4_rows) > 0,
        "Empty Stage4 prediction ledger.",
    )

    ledger_map = {}

    for row in stage4_rows:
        validate_stage4_row(row)

        key = (
            str(row["scenario_id"]),
            str(row["prediction_id"]),
        )

        require(
            key not in ledger_map,
            (
                "Duplicate Stage4 prediction key "
                f"{key}"
            ),
        )

        ledger_map[key] = row

    record_gate(
        "G08_Stage4_input_contract",
        True,
        {
            "rows": len(stage4_rows),
            "unique_predictions": (
                len(ledger_map)
            ),
            "direct_noisy_Stage2_vr": (
                True
            ),
            "measurement_R_input": True,
            "future_truth_input": False,
            "tracks_to_predict_input": (
                False
            ),
            "annotated_velocity_input": (
                False
            ),
        },
    )

    # --------------------------------------------------------
    # 9. Resolve ONLY causal metadata
    # --------------------------------------------------------

    event(
        "CAUSAL_BINDING_RESOLUTION_STARTED",
        evaluator_truth_opened=False,
    )

    causal_path, causal_rows = (
        resolve_causal_binding(
            set(ledger_map)
        )
    )

    record_gate(
        "G09_unique_frozen_causal_receiver_binding",
        len(causal_rows) > 0,
        {
            "path": str(causal_path),
            "sha256": (
                sha256_file(
                    causal_path
                )
            ),
            "selected_receivers": (
                len(causal_rows)
            ),
            "future_or_truth_payload": (
                False
            ),
        },
    )

    # --------------------------------------------------------
    # 10. Frozen receiver/codebook protocol
    # --------------------------------------------------------

    record_gate(
        "G10_receiver_modes_centroid_known_uncertain",
        protocol[
            "receiver_geometry"
        ]["modes"]
        == [
            "centroid",
            "known",
            "uncertain",
        ],
        protocol[
            "receiver_geometry"
        ],
    )

    record_gate(
        "G11_codebooks_16_32_64",
        protocol[
            "codebooks"
        ]["sizes"]
        == [16, 32, 64],
        protocol["codebooks"],
    )

    record_gate(
        "G12_probability_targets",
        protocol[
            "adaptive_TopK"
        ]["coverage_targets"]
        == [
            0.9,
            0.95,
            0.975,
            0.99,
        ],
        protocol[
            "adaptive_TopK"
        ],
    )

    # --------------------------------------------------------
    # 11. EXACT CAUSAL DECISION PASS #1
    # --------------------------------------------------------

    event(
        "DECISION_PASS_1_STARTED",
        evaluator_truth_opened=False,
    )

    decisions_1 = build_decision_rows(
        ledger_map,
        causal_rows,
        horizons,
        protocol,
    )

    require(
        not FORMAL_TRUTH_OPENED,
        (
            "Formal truth was opened during "
            "decision pass 1."
        ),
    )

    write_jsonl(
        RUN1_LEDGER,
        decisions_1,
    )

    sha_run1 = sha256_file(
        RUN1_LEDGER
    )

    event(
        "DECISION_PASS_1_COMPLETE",
        sha256=sha_run1,
        rows=len(decisions_1),
        evaluator_truth_opened=False,
    )

    # --------------------------------------------------------
    # 12. EXACT CAUSAL DECISION PASS #2 — fresh state
    # --------------------------------------------------------

    event(
        "DECISION_PASS_2_STARTED",
        evaluator_truth_opened=False,
    )

    decisions_2 = build_decision_rows(
        ledger_map,
        causal_rows,
        horizons,
        protocol,
    )

    require(
        not FORMAL_TRUTH_OPENED,
        (
            "Formal truth was opened during "
            "decision pass 2."
        ),
    )

    write_jsonl(
        RUN2_LEDGER,
        decisions_2,
    )

    sha_run2 = sha256_file(
        RUN2_LEDGER
    )

    exact_repeat = (
        sha_run1 == sha_run2
        and RUN1_LEDGER.read_bytes()
        == RUN2_LEDGER.read_bytes()
    )

    record_gate(
        "G13_two_exact_repeat_causal_decision_passes",
        exact_repeat,
        {
            "run1_sha256": sha_run1,
            "run2_sha256": sha_run2,
            "rows_run1": (
                len(decisions_1)
            ),
            "rows_run2": (
                len(decisions_2)
            ),
            "future_truth_opened": (
                False
            ),
        },
    )

    # --------------------------------------------------------
    # 13. Seal decision ledger BEFORE evaluator truth
    # --------------------------------------------------------

    shutil.copyfile(
        RUN1_LEDGER,
        SEALED_LEDGER,
    )

    sealed_sha = sha256_file(
        SEALED_LEDGER
    )

    require(
        sealed_sha == sha_run1,
        "Sealed ledger hash mismatch.",
    )

    os.chmod(
        SEALED_LEDGER,
        stat.S_IRUSR
        | stat.S_IRGRP
        | stat.S_IROTH,
    )

    DECISION_LEDGER_SEALED = True

    atomic_write_json(
        SEAL_REPORT,
        {
            "stage": 5,
            "version": "fullpdf_v2",
            "status": (
                "SEALED_BEFORE_EVALUATOR_TRUTH"
            ),
            "run1_sha256": (
                sha_run1
            ),
            "run2_sha256": (
                sha_run2
            ),
            "sealed_sha256": (
                sealed_sha
            ),
            "rows": len(
                decisions_1
            ),
            "byte_exact_repeat": (
                exact_repeat
            ),
            "future_truth_opened_before_seal": (
                False
            ),
            "read_only_mode": (
                oct(
                    SEALED_LEDGER
                    .stat()
                    .st_mode
                    & 0o777
                )
            ),
        },
    )

    event(
        "DECISION_LEDGER_SEALED",
        sha256=sealed_sha,
        evaluator_truth_opened=False,
    )

    record_gate(
        "G14_decision_ledger_sealed_before_truth",
        (
            DECISION_LEDGER_SEALED
            and not FORMAL_TRUTH_OPENED
            and (
                SEALED_LEDGER
                .stat()
                .st_mode
                & 0o222
            )
            == 0
        ),
        {
            "path": (
                str(SEALED_LEDGER)
            ),
            "sha256": sealed_sha,
            "writable_bits": (
                SEALED_LEDGER
                .stat()
                .st_mode
                & 0o222
            ),
        },
    )

    # ========================================================
    # *** EVALUATOR TRUTH MAY BE OPENED ONLY BELOW THIS LINE ***
    # ========================================================

    decision_keys = {
        (
            x["scenario_id"],
            x["prediction_id"],
        )
        for x in decisions_1
    }

    truth_path, truth_map = (
        resolve_truth_binding(
            decision_keys
        )
    )

    record_gate(
        "G15_evaluator_truth_opened_only_after_seal",
        (
            FORMAL_TRUTH_OPENED
            and DECISION_LEDGER_SEALED
        ),
        {
            "truth_path": (
                str(truth_path)
            ),
            "truth_sha256": (
                sha256_file(
                    truth_path
                )
            ),
            "decision_sealed_sha256": (
                sealed_sha
            ),
        },
    )

    # --------------------------------------------------------
    # 16. Formal evaluation
    # --------------------------------------------------------

    evaluator_rows, formal_summary = (
        evaluate_sealed_decisions(
            decisions_1,
            truth_map,
            protocol,
        )
    )

    write_jsonl(
        EVALUATOR_ROWS,
        evaluator_rows,
    )

    evaluator_sha = sha256_file(
        EVALUATOR_ROWS
    )

    # --------------------------------------------------------
    # 17. Mandatory scientific gates
    # --------------------------------------------------------

    record_gate(
        "G16_requested_empirical_coverage",
        formal_summary[
            "requested_coverage_all_strata_pass"
        ],
        {
            "rule": (
                "empirical_coverage >= "
                "requested_mass; no invented tolerance"
            ),
            "strata": (
                formal_summary["strata"]
            ),
        },
    )

    record_gate(
        "G17_actual_charged_overhead_reduced_vs_exhaustive",
        formal_summary[
            "overhead_reduction_vs_exhaustive_all_strata_pass"
        ],
        {
            "comparison": (
                "actual charged probes / "
                "base codebook size < 1"
            ),
            "all_evaluated_beams_charged": (
                True
            ),
        },
    )

    record_gate(
        "G18_no_hidden_or_free_probes",
        formal_summary[
            "no_free_probes"
        ],
        {
            "optical_evaluator_scope": (
                "sealed ProbePlan beams only"
            ),
            "free_probe_count": 0,
        },
    )

    record_gate(
        "G19_latency_from_actual_charged_probe_count",
        formal_summary[
            "latency_all_strata_pass"
        ],
        {
            "source": (
                "iscai_stage5.beam_latency."
                "beam_probing_time_s"
            ),
            "count_source": (
                "sealed actual ProbePlan"
            ),
        },
    )

    record_gate(
        "G20_complete_optical_chain",
        formal_summary[
            "full_optical_chain_complete"
        ],
        {
            "chain": [
                "pointing_error",
                "optical_gain",
                "received_power",
                "SNR",
                "DPSK_BER",
                "effective_rate",
            ],
            "claim": (
                "normalized_constructed_"
                "optical_surrogate"
            ),
            "measured_optical_claim": (
                False
            ),
        },
    )

    # Structural recovery mechanisms are frozen in exact runtime
    # and regression. Their occurrence rate is reported but a zero
    # occurrence is not turned into an invented failure threshold.
    runtime_source = RUNTIME.read_text(
        encoding="utf-8"
    )

    recovery_strings = {
        "persistence": (
            "previous_primary_index"
            in runtime_source
        ),
        "hysteresis": (
            "hysteresis"
            in runtime_source.lower()
        ),
        "local_neighbor": (
            "local_neighbor_recovery"
            in runtime_source
        ),
        "widened_fallback": (
            "widened_fallback"
            in runtime_source
        ),
        "exhaustive_loss_of_lock": (
            "exhaustive_loss_of_lock"
            in runtime_source
        ),
    }

    record_gate(
        "G21_persistence_hysteresis_recovery",
        all(
            recovery_strings.values()
        ),
        {
            "structural": (
                recovery_strings
            ),
            "formal_event_counts": (
                formal_summary[
                    "recovery_event_counts"
                ]
            ),
            "regression_tests": (
                EXPECTED_REGRESSION_TEST_COUNT
            ),
        },
    )

    # --------------------------------------------------------
    # 18. Re-check frozen inputs AFTER formal outcomes
    # --------------------------------------------------------

    end_hashes = {
        "runtime": sha256_file(
            RUNTIME
        ),
        "protocol": sha256_file(
            PROTOCOL
        ),
        "Stage4_ledger": (
            sha256_file(
                STAGE4_LEDGER
            )
        ),
    }

    record_gate(
        "G22_no_post_outcome_mutation",
        (
            end_hashes["runtime"]
            == EXPECTED_RUNTIME_SHA
            and end_hashes["protocol"]
            == EXPECTED_PROTOCOL_SHA
            and end_hashes[
                "Stage4_ledger"
            ]
            == EXPECTED_STAGE4_LEDGER_SHA
        ),
        end_hashes,
    )

    # --------------------------------------------------------
    # 19. Explicit PDF requirement -> evidence matrix
    # --------------------------------------------------------

    pdf_matrix = [
        {
            "requirement": (
                "Receiver-aware posterior; not centroid only"
            ),
            "evidence": (
                "centroid/known/uncertain frozen modes"
            ),
            "gate": "G10",
            "pass": GATES[
                "G10_receiver_modes_centroid_known_uncertain"
            ],
        },
        {
            "requirement": (
                "Integrate trajectory and receiver-location uncertainty"
            ),
            "evidence": (
                "fullpdf_v2.receiver_samples_h0; "
                "2048 deterministic MC samples"
            ),
            "gate": "G03/G04",
            "pass": (
                GATES[
                    "G03_protocol_preoutcome_freeze"
                ]
                and GATES[
                    "G04_no_Stage4_recalibration"
                ]
            ),
        },
        {
            "requirement": (
                "Codebooks 16/32/64"
            ),
            "evidence": (
                "frozen protocol + decision ledger"
            ),
            "gate": "G11",
            "pass": GATES[
                "G11_codebooks_16_32_64"
            ],
        },
        {
            "requirement": (
                "Beam probability retains explicit outside-FOV mass"
            ),
            "evidence": (
                "no clipping; sum P(b)+P(outside)=1"
            ),
            "gate": "G03",
            "pass": GATES[
                "G03_protocol_preoutcome_freeze"
            ],
        },
        {
            "requirement": (
                "Adaptive minimum Top-K at "
                "90/95/97.5/99% mass"
            ),
            "evidence": (
                "runtime minimum-prefix assertion "
                "for every formal decision"
            ),
            "gate": "G12/G13",
            "pass": (
                GATES[
                    "G12_probability_targets"
                ]
                and GATES[
                    "G13_two_exact_repeat_causal_decision_passes"
                ]
            ),
        },
        {
            "requirement": (
                "Causal decisions before evaluator future truth"
            ),
            "evidence": (
                "two exact passes -> immutable seal -> "
                "truth boundary"
            ),
            "gate": "G13-G15",
            "pass": (
                GATES[
                    "G13_two_exact_repeat_causal_decision_passes"
                ]
                and GATES[
                    "G14_decision_ledger_sealed_before_truth"
                ]
                and GATES[
                    "G15_evaluator_truth_opened_only_after_seal"
                ]
            ),
        },
        {
            "requirement": (
                "Persistence, hysteresis, neighbour recovery, "
                "widened fallback, exhaustive loss-of-lock"
            ),
            "evidence": (
                "frozen runtime + 271-test regression"
            ),
            "gate": "G21",
            "pass": GATES[
                "G21_persistence_hysteresis_recovery"
            ],
        },
        {
            "requirement": (
                "All evaluated beams charged; no hidden/free probes"
            ),
            "evidence": (
                "optical evaluation restricted to sealed ProbePlan"
            ),
            "gate": "G18",
            "pass": GATES[
                "G18_no_hidden_or_free_probes"
            ],
        },
        {
            "requirement": (
                "Adaptive Top-K approximately meets "
                "requested empirical coverage"
            ),
            "evidence": (
                "strict empirical >= requested q; "
                "no post-outcome tolerance invented"
            ),
            "gate": "G16",
            "pass": GATES[
                "G16_requested_empirical_coverage"
            ],
        },
        {
            "requirement": (
                "Reduce probing overhead vs fixed/exhaustive"
            ),
            "evidence": (
                "actual charged probes vs exhaustive base codebook"
            ),
            "gate": "G17",
            "pass": GATES[
                "G17_actual_charged_overhead_reduced_vs_exhaustive"
            ],
        },
        {
            "requirement": (
                "Latency derived from actual probing workload"
            ),
            "evidence": (
                "beam_probing_time_s(actual charged count)"
            ),
            "gate": "G19",
            "pass": GATES[
                "G19_latency_from_actual_charged_probe_count"
            ],
        },
        {
            "requirement": (
                "pointing -> gain -> received power -> "
                "SNR -> DPSK BER -> effective rate"
            ),
            "evidence": (
                "frozen Stage5 constructed optical chain"
            ),
            "gate": "G20",
            "pass": GATES[
                "G20_complete_optical_chain"
            ],
        },
        {
            "requirement": (
                "No Stage4 covariance recalibration in Stage5"
            ),
            "evidence": (
                "exact runtime AST + frozen protocol"
            ),
            "gate": "G04",
            "pass": GATES[
                "G04_no_Stage4_recalibration"
            ],
        },
        {
            "requirement": (
                "No formal tuning / no post-outcome mutation"
            ),
            "evidence": (
                "frozen hashes identical before and after formal"
            ),
            "gate": "G22",
            "pass": GATES[
                "G22_no_post_outcome_mutation"
            ],
        },
    ]

    pdf_all_pass = all(
        x["pass"]
        for x in pdf_matrix
    )

    record_gate(
        "G23_PDF_requirement_matrix",
        pdf_all_pass,
        pdf_matrix,
    )

    # --------------------------------------------------------
    # 20. Final status
    # --------------------------------------------------------

    mandatory_all_pass = all(
        GATES.values()
    )

    require(
        mandatory_all_pass,
        "At least one mandatory Stage5 gate failed.",
    )

    report = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status": (
            "PASS_COMPLETE_FROZEN_FULLPDF_V2"
        ),
        "Stage5": (
            "COMPLETE_FROZEN_FULLPDF_V2"
        ),
        "Stage6_allowed": True,
        "decision_boundary": {
            "two_exact_passes": True,
            "decision_ledger_sealed": (
                True
            ),
            "future_truth_opened_only_after_seal": (
                True
            ),
            "sealed_ledger": (
                str(SEALED_LEDGER)
            ),
            "sealed_ledger_sha256": (
                sealed_sha
            ),
        },
        "provenance": {
            "Stage5_runtime": {
                "path": str(RUNTIME),
                "sha256": runtime_sha,
            },
            "Stage5_protocol": {
                "path": str(PROTOCOL),
                "sha256": protocol_sha,
            },
            "Stage4_prediction_ledger": {
                "path": (
                    str(STAGE4_LEDGER)
                ),
                "sha256": (
                    stage4_ledger_sha
                ),
            },
            "Stage4_closure": {
                "path": (
                    str(s4_closure_path)
                ),
                "sha256": (
                    expected_s4_closure_sha
                ),
            },
            "Stage4_handoff": {
                "path": (
                    str(s4_handoff_path)
                ),
                "sha256": (
                    expected_s4_handoff_sha
                ),
            },
            "causal_receiver_binding": {
                "path": (
                    str(causal_path)
                ),
                "sha256": (
                    sha256_file(
                        causal_path
                    )
                ),
            },
            "evaluator_truth_binding": {
                "path": (
                    str(truth_path)
                ),
                "sha256": (
                    sha256_file(
                        truth_path
                    )
                ),
            },
        },
        "regression": {
            "status": "PASS",
            "tests": (
                EXPECTED_REGRESSION_TEST_COUNT
            ),
            "log": (
                str(REGRESSION_LOG)
            ),
            "log_sha256": (
                sha256_file(
                    REGRESSION_LOG
                )
            ),
        },
        "formal": {
            **formal_summary,
            "evaluator_rows": (
                str(EVALUATOR_ROWS)
            ),
            "evaluator_rows_sha256": (
                evaluator_sha
            ),
        },
        "PDF_requirement_evidence_matrix": (
            pdf_matrix
        ),
        "gates": GATES,
        "events": EVENTS,
        "scientific_boundaries": {
            "training": False,
            "retraining": False,
            "recalibration": False,
            "formal_parameter_tuning": (
                False
            ),
            "post_outcome_tuning": (
                False
            ),
            "future_truth_controller_input": (
                False
            ),
            "tracks_to_predict_controller_input": (
                False
            ),
            "measured_optical_claim": (
                False
            ),
            "optical_semantics": (
                "normalized_constructed_"
                "optical_surrogate"
            ),
        },
    }

    atomic_write_json(
        REPORT,
        report,
    )

    report_sha = sha256_file(
        REPORT
    )

    closure = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status": (
            "COMPLETE_FROZEN_FULLPDF_V2"
        ),
        "PDF_compliance": "PASS",
        "Stage6_allowed": True,
        "independent_formal_report": {
            "path": str(REPORT),
            "sha256": report_sha,
        },
        "sealed_decision_ledger": {
            "path": (
                str(SEALED_LEDGER)
            ),
            "sha256": (
                sealed_sha
            ),
        },
        "formal_evaluator_rows": {
            "path": (
                str(EVALUATOR_ROWS)
            ),
            "sha256": (
                evaluator_sha
            ),
        },
        "runtime_sha256": (
            runtime_sha
        ),
        "protocol_sha256": (
            protocol_sha
        ),
        "Stage4_prediction_ledger_sha256": (
            stage4_ledger_sha
        ),
        "posterior": (
            "calibrated_full_3D_Gaussian_GRU"
        ),
        "Stage4_covariance_already_calibrated": (
            True
        ),
        "Stage5_recalibration": False,
        "no_post_outcome_tuning": True,
    }

    atomic_write_json(
        CLOSURE,
        closure,
    )

    closure_sha = sha256_file(
        CLOSURE
    )

    handoff = {
        "from_stage": 5,
        "to_stage": 6,
        "status": (
            "READY_FROM_"
            "COMPLETE_FROZEN_FULLPDF_V2"
        ),
        "Stage6_allowed": True,
        "Stage5_closure": {
            "path": (
                str(CLOSURE)
            ),
            "sha256": (
                closure_sha
            ),
        },
        "Stage5_independent_formal": {
            "path": str(REPORT),
            "sha256": report_sha,
        },
        "sealed_beam_decision_ledger": {
            "path": (
                str(SEALED_LEDGER)
            ),
            "sha256": (
                sealed_sha
            ),
        },
        "shared_trajectory_posterior": {
            "source": (
                str(STAGE4_LEDGER)
            ),
            "sha256": (
                stage4_ledger_sha
            ),
            "distribution": (
                "calibrated 3D Gaussian GRU"
            ),
            "covariance_semantics": (
                "already calibrated; "
                "DO NOT recalibrate downstream"
            ),
        },
        "receiver_modes": (
            protocol[
                "receiver_geometry"
            ]["modes"]
        ),
        "codebooks": (
            protocol[
                "codebooks"
            ]["sizes"]
        ),
        "probability_targets": (
            protocol[
                "adaptive_TopK"
            ]["coverage_targets"]
        ),
        "formal_prediction_horizons_s": (
            horizons
        ),
        "optical_semantics": (
            "normalized_constructed_"
            "optical_surrogate"
        ),
        "future_truth_controller_input": (
            False
        ),
        "tracks_to_predict_controller_input": (
            False
        ),
        "post_outcome_tuning": False,
    }

    atomic_write_json(
        HANDOFF,
        handoff,
    )

    # Freeze terminal scientific artifacts.
    for path in (
        REPORT,
        CLOSURE,
        HANDOFF,
        SEAL_REPORT,
        EVALUATOR_ROWS,
        RUN1_LEDGER,
        RUN2_LEDGER,
    ):
        os.chmod(
            path,
            stat.S_IRUSR
            | stat.S_IRGRP
            | stat.S_IROTH,
        )

    print()
    print("=" * 72)
    print(
        "STAGE5 = COMPLETE_FROZEN_FULLPDF_V2"
    )
    print("PDF compliance = PASS")
    print("Stage6_allowed = true")
    print(
        "decision ledger SHA256 =",
        sealed_sha,
    )
    print(
        "formal report SHA256   =",
        report_sha,
    )
    print(
        "closure SHA256         =",
        closure_sha,
    )
    print(
        "handoff                =",
        HANDOFF,
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except FailClosed as exc:
        fail_closed(
            str(exc),
            exception=exc,
        )

        raise SystemExit(2)

    except Exception as exc:
        fail_closed(
            (
                "Unexpected checker exception; "
                "automatic fail-closed."
            ),
            exception=exc,
        )

        traceback.print_exc()

        raise SystemExit(3)
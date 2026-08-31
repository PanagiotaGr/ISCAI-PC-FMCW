from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import sys
import traceback

from pathlib import Path
from typing import Any


ROOT = Path("/home/agni/waymo")
STAGE4 = ROOT / "iscai_stage4"
STAGE6 = ROOT / "iscai_stage6"

PART3B = (
    STAGE6
    / "configs"
    / "block66_part3b_class_aware_policy_preregistration.json"
)

PART3C_CLOSURE = (
    STAGE6
    / "reports"
    / "block66_part3c_final_closure.json"
)

PART3C_FREEZE = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_part3c_final_freeze_manifest.json"
)

PART3C_HANDOFF = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_part3c_to_block67_handoff.json"
)

CLASS_AWARE_MODULE = (
    STAGE6
    / "src"
    / "iscai_stage6"
    / "adb"
    / "class_aware_policy.py"
)

COHORT = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_class_aware_development_cohort_120.jsonl"
)

POCC_MANIFEST = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_part2c2b_eligible_actor_pocc_manifest.jsonl"
)

ELIGIBLE_MATCHES = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_part2c1_headlamp_eligible_predictive_matches.jsonl"
)

SURROGATE = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_part3c_vehicle_surrogate_evidence.jsonl"
)

POSTERIOR = (
    STAGE6
    / "artifacts"
    / "block66"
    / "block66_development_identity_safe_gaussian_posterior.jsonl"
)


EXPECTED_SHA = {
    PART3B:
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    PART3C_CLOSURE:
        "8413fd646a1bf18e15c12bf0441d1ddb1ea72ed142703020ac0d2c77dba0bb29",

    PART3C_FREEZE:
        "880c2efb0895f463aaf3350eed7730389a5871651d7ad6091cef052d95434738",

    PART3C_HANDOFF:
        "859e4d936c9e0eb04f35925382c7fa3aa86c91e1d2ff85beb86bb27e77ca4b5d",

    CLASS_AWARE_MODULE:
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    COHORT:
        "57440e3d976e7aeff44811b5eeb075dc56e0e0e070a3832ec72d354253b998c9",

    POCC_MANIFEST:
        "c361ed640b91850831d7e7177d89733fedd3e171f5f3111bcd93698c74a4a300",

    ELIGIBLE_MATCHES:
        "d7ce8769deee08286974b0327cdfd55c08ad7ea326cf9d6450a48a82b3e251b6",

    SURROGATE:
        "bfe8eb6d8b605d58f9b99d8097744b71b99ecb9ed9fd169191222375f31bc826",

    POSTERIOR:
        "f022471fbf86c3121106011bfd7f035c5d895f2c5aeeaa2fa3a20c04b3439acd",
}


EXPECTED_DEVELOPMENT_SHA = (
    "e4689698bddd80e58add7267f791aea90ef4bef0309ca18d0d7a9e7e94fe7e5c"
)


class ControlledBlock(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ControlledBlock(str(message))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def first_jsonl(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                value = json.loads(line)

                require(
                    isinstance(value, dict),
                    f"{path} first JSONL record is not an object.",
                )

                return value

    raise ControlledBlock(
        f"No records in {path}"
    )


def count_jsonl(path: Path) -> int:
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                count += 1

    return count


def flatten(
    value: Any,
    prefix: str = "",
) -> list[tuple[str, Any]]:

    out: list[tuple[str, Any]] = []

    if isinstance(value, dict):
        for key, item in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            out.extend(
                flatten(
                    item,
                    path,
                )
            )

        return out

    if isinstance(value, list):
        for index, item in enumerate(value):
            path = (
                f"{prefix}[{index}]"
            )

            out.extend(
                flatten(
                    item,
                    path,
                )
            )

        return out

    out.append(
        (
            prefix,
            value,
        )
    )

    return out


def short_repr(
    value: Any,
    maximum: int = 420,
) -> str:

    text = repr(value)

    if len(text) > maximum:
        return (
            text[:maximum]
            + "...<truncated>"
        )

    return text


def print_schema(
    title: str,
    record: dict[str, Any],
) -> None:

    print()
    print(title)
    print("keys =", sorted(record.keys()))

    for key in sorted(record.keys()):
        value = record[key]

        if isinstance(value, dict):
            print(
                f"{key}: dict keys =",
                sorted(value.keys()),
            )

        elif isinstance(value, list):
            print(
                f"{key}: list length =",
                len(value),
            )

        else:
            print(
                f"{key}:",
                short_repr(value, 220),
            )


def print_source(
    title: str,
    obj: Any,
) -> None:

    print()
    print("-" * 78)
    print(title)
    print("-" * 78)

    try:
        print(
            "signature =",
            inspect.signature(obj),
        )
    except BaseException:
        pass

    try:
        print(
            inspect.getsource(obj)
        )
    except BaseException as exc:
        print(
            "source unavailable =",
            type(exc).__name__,
            str(exc),
        )


def relevant_part3b_lines(
    payload: Any,
) -> list[str]:

    tokens = (
        "selection",
        "objective",
        "tie",
        "gamma",
        "threshold",
        "strict",
        "iou",
        "intersection",
        "union",
        "macro",
        "margin",
        "m_extra",
        "mextra",
        "atan2",
        "sigma_lat",
        "ell",
        "lateral",
        "v_close",
        "vclose",
        "v_lat",
        "vlat",
        "surrogate",
        "front_upper",
        "rear_upper",
        "floor",
        "stage",
        "candidate",
        "development",
    )

    result: list[str] = []

    for path, value in flatten(payload):
        text = (
            f"{path} = {short_repr(value)}"
        )

        lowered = text.lower()

        if any(
            token.lower() in lowered
            for token in tokens
        ):
            result.append(text)

    return result


def main() -> None:

    print("=" * 78)
    print(
        "BLOCK 6.7 PART 1/2 — READ-ONLY BINDING PREFLIGHT"
    )
    print(
        "DEVELOPMENT-ONLY CLASS-AWARE POLICY SELECTION"
    )
    print("=" * 78)

    # ==========================================================
    # A. Immutable seal
    # ==========================================================

    print()
    print(
        "===== A. IMMUTABLE BLOCK6.6 / PART3B SEAL ====="
    )

    for path, expected in EXPECTED_SHA.items():

        require(
            path.is_file(),
            f"Missing frozen file: {path}",
        )

        actual = sha256_file(path)

        require(
            actual == expected,
            (
                f"SHA mismatch for {path}\n"
                f"expected={expected}\n"
                f"actual={actual}"
            ),
        )

        print(
            f"{path.name:<62} = EXACT PASS"
        )

    print(
        "scientific source modification = NO"
    )
    print(
        "numeric policy selection       = FORBIDDEN IN PREFLIGHT"
    )
    print(
        "development outcomes read       = NO"
    )
    print(
        "future GT read                  = NO"
    )
    print(
        "formal outcomes read            = NO"
    )

    # ==========================================================
    # B. Frozen record census / schema
    # ==========================================================

    print()
    print(
        "===== B. FROZEN INPUT RECORD CENSUS ====="
    )

    counts = {
        "development cohort":
            count_jsonl(COHORT),

        "P_occ manifest":
            count_jsonl(POCC_MANIFEST),

        "eligible predictive matches":
            count_jsonl(ELIGIBLE_MATCHES),

        "vehicle surrogate evidence":
            count_jsonl(SURROGATE),

        "development posterior":
            count_jsonl(POSTERIOR),
    }

    for key, value in counts.items():
        print(
            f"{key:<36} = {value}"
        )

    require(
        counts["development cohort"] == 120,
        "Development cohort must remain exactly 120.",
    )

    require(
        counts["P_occ manifest"] == 877,
        "P_occ manifest must remain exactly 877.",
    )

    require(
        counts["eligible predictive matches"] == 877,
        "Predictive match manifest must remain exactly 877.",
    )

    require(
        counts["vehicle surrogate evidence"] == 719,
        "Vehicle surrogate evidence must remain exactly 719.",
    )

    require(
        counts["development posterior"] == 6925,
        "Development posterior must remain exactly 6925.",
    )

    # ==========================================================
    # C. Exact Part3B semantics
    # ==========================================================

    print()
    print(
        "===== C. PART3B EXACT SELECTION / MARGIN SEMANTICS ====="
    )

    prereg = load_json(
        PART3B
    )

    require(
        isinstance(prereg, dict),
        "Part3B preregistration root is not a JSON object.",
    )

    print(
        "Part3B top-level keys =",
        sorted(prereg.keys()),
    )

    relevant = (
        relevant_part3b_lines(
            prereg
        )
    )

    print(
        "relevant flattened entries =",
        len(relevant),
    )

    for line in relevant:
        print(line)

    semantic_blob = (
        "\n".join(relevant)
        .lower()
        .replace(" ", "")
    )

    print()
    print(
        "----- PREFLIGHT SEMANTIC FLAGS -----"
    )

    flags = {
        "strict_threshold_surface":
            (
                "strict"
                in semantic_blob
                and
                "threshold"
                in semantic_blob
            ),

        "iou_surface":
            "iou" in semantic_blob,

        "macro_surface":
            "macro" in semantic_blob,

        "tie_surface":
            "tie" in semantic_blob,

        "m_extra_surface":
            (
                "m_extra"
                in semantic_blob
                or
                "mextra"
                in semantic_blob
            ),

        "atan2_surface":
            "atan2" in semantic_blob,

        "sigma_lat_surface":
            (
                "sigma_lat"
                in semantic_blob
                or
                "sigmalat"
                in semantic_blob
            ),

        "closing_surface":
            (
                "v_close"
                in semantic_blob
                or
                "vclose"
                in semantic_blob
                or
                "closing"
                in semantic_blob
            ),

        "cyclist_lateral_surface":
            (
                "v_lat"
                in semantic_blob
                or
                "vlat"
                in semantic_blob
                or
                "lateral"
                in semantic_blob
            ),

        "vehicle_surrogate_surface":
            "surrogate" in semantic_blob,
    }

    for key, value in flags.items():
        print(
            f"{key:<34} = {value}"
        )

    require(
        flags["strict_threshold_surface"],
        (
            "Part3B semantic extract does not expose "
            "the frozen strict-threshold rule."
        ),
    )

    require(
        flags["iou_surface"],
        (
            "Part3B semantic extract does not expose "
            "the frozen IoU selection objective."
        ),
    )

    require(
        flags["m_extra_surface"],
        (
            "Part3B semantic extract does not expose "
            "m_extra semantics."
        ),
    )

    require(
        flags["atan2_surface"],
        (
            "Part3B semantic extract does not expose "
            "the frozen angular-margin atan2 relation."
        ),
    )

    require(
        flags["sigma_lat_surface"],
        (
            "Part3B semantic extract does not expose "
            "sigma_lat semantics."
        ),
    )

    # Do NOT resolve m_extra-vs-total by assumption here.
    # The output above is the authoritative evidence for
    # the next Part1/2 execution.

    print()
    print(
        "predictive dilation margin source = "
        "NOT GUESSED / MUST BE RESOLVED FROM ABOVE FROZEN TEXT"
    )

    # ==========================================================
    # D. Frozen runtime operator source
    # ==========================================================

    print()
    print(
        "===== D. PART3C FROZEN RUNTIME OPERATOR API ====="
    )

    import iscai_stage6.adb.class_aware_policy as cap

    required_runtime = (
        "threshold_occupancy_counts_strict_k",
        "compute_class_aware_margin",
        "angular_margin_cells",
        "dilate_mask_theta",
    )

    for name in required_runtime:
        require(
            hasattr(cap, name),
            f"Missing frozen runtime API: {name}",
        )

        obj = getattr(
            cap,
            name,
        )

        print_source(
            f"iscai_stage6.adb.class_aware_policy.{name}",
            obj,
        )

    # ==========================================================
    # E. Exact evaluator-only future-box binding
    # ==========================================================

    print()
    print(
        "===== E. EVALUATOR-ONLY FUTURE BOX API BINDING ====="
    )

    import iscai_stage6.adb.geometry as geometry
    import iscai_stage6.adb.probabilistic_occupancy as occupancy

    from iscai_stage6.adb.deterministic_predictive import (
        DeterministicFutureProjectedBox,
    )

    from iscai_stage6.adb.womd_geometry import (
        build_causal_adb_actor_boxes,
        world_heading_to_headlamp_yaw,
    )

    print_source(
        "iscai_stage6.adb.geometry.validate_controller_provenance",
        geometry.validate_controller_provenance,
    )

    print_source(
        "iscai_stage6.adb.geometry.project_box_to_headlamp",
        geometry.project_box_to_headlamp,
    )

    print_source(
        "iscai_stage6.adb.womd_geometry.world_heading_to_headlamp_yaw",
        world_heading_to_headlamp_yaw,
    )

    print()
    print(
        "build_causal_adb_actor_boxes signature =",
        inspect.signature(
            build_causal_adb_actor_boxes
        ),
    )

    print()
    print(
        "DeterministicFutureProjectedBox signature =",
        inspect.signature(
            DeterministicFutureProjectedBox
        ),
    )

    if hasattr(
        DeterministicFutureProjectedBox,
        "__dataclass_fields__",
    ):
        print(
            "DeterministicFutureProjectedBox fields =",
            list(
                DeterministicFutureProjectedBox
                .__dataclass_fields__
            ),
        )

    require(
        hasattr(
            occupancy,
            "_projected_theta_range_points",
        ),
        (
            "Frozen occupancy private adapter "
            "_projected_theta_range_points missing."
        ),
    )

    print_source(
        "iscai_stage6.adb.probabilistic_occupancy._projected_theta_range_points",
        occupancy._projected_theta_range_points,
    )

    print_source(
        "iscai_stage6.adb.probabilistic_occupancy.rasterize_projected_full_box",
        occupancy.rasterize_projected_full_box,
    )

    # ==========================================================
    # F. Exact raw-WOMD random-access route
    # ==========================================================

    print()
    print(
        "===== F. DEVELOPMENT RAW-WOMD RANDOM-ACCESS ROUTE ====="
    )

    import iscai_stage4.data.real_pipeline as real_pipeline

    require(
        hasattr(
            real_pipeline,
            "read_training_scenario",
        ),
        (
            "iscai_stage4.data.real_pipeline "
            "has no read_training_scenario."
        ),
    )

    read_training_scenario = (
        real_pipeline
        .read_training_scenario
    )

    print_source(
        "iscai_stage4.data.real_pipeline.read_training_scenario",
        read_training_scenario,
    )

    # Find the DEVELOPMENT manifest by frozen SHA,
    # without assuming its filename.
    print()
    print(
        "----- DEVELOPMENT MANIFEST RESOLUTION -----"
    )

    candidates = sorted(
        {
            *STAGE4.glob(
                "artifacts/block41/*development*.jsonl"
            ),
            *STAGE4.glob(
                "artifacts/block41/*dev*.jsonl"
            ),
        }
    )

    print(
        "candidate path count =",
        len(candidates),
    )

    resolved_dev: Path | None = None

    for path in candidates:

        if not path.is_file():
            continue

        digest = sha256_file(
            path
        )

        print(
            path,
            "| SHA256 =",
            digest,
        )

        if (
            digest
            ==
            EXPECTED_DEVELOPMENT_SHA
        ):
            resolved_dev = path

    if resolved_dev is None:

        print()
        print(
            "No name-filtered candidate matched."
        )

        block41 = (
            STAGE4
            / "artifacts"
            / "block41"
        )

        if block41.is_dir():
            print(
                "block41 directory inventory:"
            )

            for path in sorted(
                block41.iterdir()
            ):
                if path.is_file():
                    print(
                        " ",
                        path.name,
                    )

        raise ControlledBlock(
            (
                "Frozen Stage4 development manifest "
                "path was not resolved by exact SHA. "
                "This is a binding issue only; "
                "no future GT or policy outcomes were read."
            )
        )

    print(
        "resolved development manifest =",
        resolved_dev,
    )

    print(
        "resolved development SHA256 =",
        sha256_file(
            resolved_dev
        ),
    )

    development_first = (
        first_jsonl(
            resolved_dev
        )
    )

    print_schema(
        "development manifest first-record schema:",
        development_first,
    )

    # ==========================================================
    # G. Frozen join-record schemas
    # ==========================================================

    print()
    print(
        "===== G. FROZEN JOIN / POSTERIOR / SURROGATE SCHEMAS ====="
    )

    print_schema(
        "cohort first record:",
        first_jsonl(COHORT),
    )

    print_schema(
        "eligible predictive match first record:",
        first_jsonl(ELIGIBLE_MATCHES),
    )

    print_schema(
        "P_occ first manifest record:",
        first_jsonl(POCC_MANIFEST),
    )

    print_schema(
        "development posterior first record:",
        first_jsonl(POSTERIOR),
    )

    print_schema(
        "vehicle surrogate first record:",
        first_jsonl(SURROGATE),
    )

    # ==========================================================
    # H. Stage6 fresh regression
    # ==========================================================

    print()
    print(
        "===== H. FRESH STAGE6 REGRESSION ====="
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ],
        cwd=str(STAGE6),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print(
        completed.stdout,
        end="",
    )

    require(
        completed.returncode == 0,
        (
            "Fresh Stage6 unittest regression "
            f"returned {completed.returncode}."
        ),
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        completed.stdout,
    )

    require(
        match is not None,
        "Could not resolve fresh Stage6 test count.",
    )

    test_count = int(
        match.group(1)
    )

    require(
        test_count == 166,
        (
            "Expected frozen Stage6 regression "
            f"count 166, got {test_count}."
        ),
    )

    print(
        "Stage6 regression = PASS |",
        test_count,
    )

    # ==========================================================
    # Final
    # ==========================================================

    print()
    print("=" * 78)
    print(
        "BLOCK 6.7 PART 1/2 — READ-ONLY BINDING PREFLIGHT — FINAL"
    )
    print("=" * 78)
    print(
        "STATUS = PASS_BLOCK67_PART1OF2_BINDING_PREFLIGHT"
    )
    print(
        "Part3B frozen semantics          = EXTRACTED"
    )
    print(
        "Part3C runtime operators         = EXACT BOUND"
    )
    print(
        "evaluator-only box route         = EXACT BOUND"
    )
    print(
        "raw development WOMD route       = EXACT BOUND"
    )
    print(
        "development manifest             = EXACT SHA BOUND"
    )
    print(
        "P_occ actors                     = 877"
    )
    print(
        "vehicle surrogate evidence       = 719"
    )
    print(
        "policy sweep executed            = NO"
    )
    print(
        "numeric policy selected          = NO"
    )
    print(
        "development objective evaluated  = NO"
    )
    print(
        "future GT read                   = NO"
    )
    print(
        "formal outcomes read             = NO"
    )
    print(
        "training / recalibration         = NO / NO"
    )
    print(
        "new model forward                = NO"
    )
    print(
        "new MC sampling                  = NO"
    )
    print(
        "Stage6 regression                = PASS | 166"
    )
    print(
        "NEXT = BLOCK6.7 PART 1/2 RETRY — "
        "EXECUTE FROZEN DEVELOPMENT GAMMA + MARGIN SELECTION"
    )
    print(
        "terminal remains open = YES"
    )
    print("=" * 78)


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "BLOCK6.7 PART 1/2 — CONTROLLED PREFLIGHT BLOCK"
    )
    print("=" * 78)

    print(
        "exception type =",
        type(exc).__name__,
    )

    print(
        "exception =",
        str(exc),
    )

    print()
    traceback.print_exc()

    print()
    print(
        "Part3B modified               = NO"
    )
    print(
        "Part3C modified               = NO"
    )
    print(
        "P_occ modified                = NO"
    )
    print(
        "vehicle surrogate modified    = NO"
    )
    print(
        "policy sweep executed         = NO"
    )
    print(
        "numeric policy selected       = NO"
    )
    print(
        "development outcomes read     = NO"
    )
    print(
        "future GT read                = NO"
    )
    print(
        "formal outcomes read          = NO"
    )
    print(
        "terminal remains open         = YES"
    )
    print("=" * 78)

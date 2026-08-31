from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback

import numpy as np


S6 = Path("/home/agni/waymo/iscai_stage6")

# ------------------------------------------------------------------
# Frozen upstream authorities
# ------------------------------------------------------------------

PREREG = (
    S6
    / "configs/"
      "block66_part3b_class_aware_policy_preregistration.json"
)

STAGE3_FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage3_class_floor_freeze.json"
)

STAGE3_SCORING = (
    S6
    / "configs/"
      "stage6_development_stage3_floor_scoring_semantics.json"
)

DISCOVERY = (
    S6
    / "reports/"
      "block68_stage4_current_illumination_authority_discovery.json"
)

REACTIVE_CONTRACT = (
    S6
    / "configs/"
      "stage6_reactive_development_scoring_contract.json"
)

REACTIVE_FREEZE = (
    S6
    / "reports/"
      "block68_reactive_decision_ledger_freeze.json"
)

LEDGER = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_decision_ledger.jsonl"
)

T0_MAPS = (
    S6
    / "artifacts/block68/"
      "block68_original_reactive_t0_maps.npy"
)

CLASS_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "class_aware_policy.py"
)

METRIC_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "metric_semantics.py"
)

EVALUATOR_RUNTIME = (
    S6
    / "src/iscai_stage6/adb/"
      "evaluator_reference.py"
)

ROAD_ROI = (
    S6
    / "artifacts/block68/"
      "frozen_road_roi_definition.json"
)

REACTIVE_HELPER = (
    S6
    / "src/iscai_stage6/adb/"
      "reactive_development_scoring.py"
)

# ------------------------------------------------------------------
# New write-once outputs
# ------------------------------------------------------------------

BINDING = (
    S6
    / "artifacts/block68/stage4_preoutcome/"
      "stage4_current_state_scenario_binding.jsonl"
)

FREEZE = (
    S6
    / "configs/"
      "stage6_development_stage4_temporal_rate_scoring_freeze.json"
)

REPORT = (
    S6
    / "reports/"
      "block68_stage4_initial_state_scoring_freeze.json"
)

EXPECTED_TESTS = 224

EXPECTED = {
    PREREG:
        "5704494a89b4b3c7a3c19b0df8176c550c00795ed17cf2906c343f340461429e",

    STAGE3_FREEZE:
        "c8536a1a2f888ebc3310cb900626cf0c8c0c2ca3ce34c2d321120a269c548d00",

    STAGE3_SCORING:
        "a8bf30b29056562eb303f287acfcca6d8aee011f26a353897f8d18b47325e1b3",

    REACTIVE_CONTRACT:
        "454d22a2f067025fd8dde877546b7305c4357000126585c860a4a2d9ef3535d5",

    REACTIVE_FREEZE:
        "6ca868dbdf13e8a675813d4af2d3c814ce5297155126e5124ac452131670903b",

    LEDGER:
        "ac2b954571cdd0166bbcfa5bb8b2956785d1e132288dbe4af7fe2d55adeafcaa",

    T0_MAPS:
        "7c5d63c76c3d9ef74adbbd7a68e3ce7483b1dd80fd262cba06f6ce1909994993",

    CLASS_RUNTIME:
        "b998f468b98c2770c48c84a2c0aaaa2177c2d84b8fc838e80fef9b9da1ebc14b",

    METRIC_RUNTIME:
        "bf8358b5a7edfe40ffac2718ca7cd5af6e7b5fdb3ff8a2823caad6d0f40f057a",

    EVALUATOR_RUNTIME:
        "2015136fdcbfe3f837bcba5a2ee05578a4d8a01c4ecfd52e4dfac9f864349891",

    ROAD_ROI:
        "832f4bc3215fe585fee781080c98b2928157f26e8a66d22f428f15604afb75e0",

    REACTIVE_HELPER:
        "e3114093406da9e4f8088a70cfcf98843055634a6e45ef30e1f3f0420f586d7b",
}

# Already-frozen numeric lineage.
# These are carried forward only; they are NOT selected here.
STAGE1_GAMMA_FREEZE_SHA = (
    "a55f21589f55a75afcb660a8a7278e802dd9e102f4c75ca7863697359bbb2a87"
)

STAGE2_MARGIN_FREEZE_SHA = (
    "b061541f0563181cd07a91107a211b3519c6fdcc2e08243171db5b3bd9393ae8"
)

HORIZONS_S = (
    0.1,
    0.3,
    0.5,
    1.0,
)

GRID_SHAPE = (
    501,
    301,
)

MAP_STACK_SHAPE = (
    120,
    501,
    301,
)

EXPECTED_TIME_CONSTANTS_MS = [
    50,
    100,
    200,
    400,
]

EXPECTED_RATES_PER_S = [
    1,
    2,
    4,
    8,
    16,
]

EXPECTED_STAGE4_LEXICOGRAPHIC = [
    "minimize maximum normalized safety-risk component",
    "minimize mean normalized safety-risk component",
    "minimize temporal mean_absolute_change_rate",
    "minimize flicker_change_rate",
    "minimize rate-limit safety-override fraction",
    "minimize over_masking_area",
    "canonical temporal/rate tuple",
]

EXPECTED_INITIAL_STATE_TEXT = (
    "current causal class-aware reactive illumination "
    "generated from exact Part-A geometry; no future information"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows):
    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def write_once(path: Path, payload: bytes):
    if path.exists():
        existing = path.read_bytes()

        require(
            existing == payload,
            (
                "WRITE-ONCE artifact already exists "
                f"with different bytes: {path}"
            ),
        )

        return "EXISTING_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        payload
    )

    temporary.replace(
        path
    )

    return "WRITTEN"


def read_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:

        for line_number, raw in enumerate(
            stream,
            start=1,
        ):
            line = raw.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc

    return rows


def full_regression():
    process = subprocess.run(
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
        cwd=str(S6),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    count = None

    for line in process.stdout.splitlines():
        match = re.search(
            r"Ran\s+(\d+)\s+tests?",
            line,
        )

        if match:
            count = int(
                match.group(1)
            )

    require(
        process.returncode == 0,
        (
            "Stage6 regression failed:\n"
            + "\n".join(
                process.stdout.splitlines()[-120:]
            )
        ),
    )

    require(
        count == EXPECTED_TESTS,
        (
            f"Expected {EXPECTED_TESTS} tests, "
            f"observed {count}."
        ),
    )

    return count


def main():

    print("=" * 78)
    print("STAGE 6 — BLOCK 6.8")
    print("STAGE-4 INITIAL-STATE + SCORING FREEZE")
    print("WRITE-ONCE PRE-OUTCOME SEMANTIC RESOLUTION")
    print("NO 60-CANDIDATE SWEEP / NO PRIMARY GATE / NO FORMAL")
    print("=" * 78)

    # ================================================================
    # A. Exact immutable boundary
    # ================================================================

    print()
    print("===== A. EXACT IMMUTABLE BOUNDARY =====")

    for path, expected in EXPECTED.items():

        require(
            path.is_file(),
            f"Missing frozen authority: {path}",
        )

        actual = file_sha(
            path
        )

        require(
            actual == expected,
            (
                f"SHA mismatch: {path}\n"
                f"expected={expected}\n"
                f"actual  ={actual}"
            ),
        )

        print(
            path.name,
            "= EXACT PASS",
        )

    require(
        DISCOVERY.is_file(),
        f"Missing discovery report: {DISCOVERY}",
    )

    discovery = json.loads(
        DISCOVERY.read_text(
            encoding="utf-8"
        )
    )

    require(
        discovery.get("status")
        ==
        "PASS_STAGE4_CURRENT_ILLUMINATION_AUTHORITY_DISCOVERY_ONLY",
        "Stage-4 current-state discovery is not PASS.",
    )

    scientific_boundary = (
        discovery.get(
            "scientific_boundary",
            {},
        )
    )

    require(
        scientific_boundary.get(
            "Stage4_outcomes_computed"
        )
        is False,
        "Discovery says Stage-4 outcomes were already computed.",
    )

    require(
        scientific_boundary.get(
            "Stage4_winner_selected"
        )
        is False,
        "Discovery says Stage-4 winner was already selected.",
    )

    require(
        scientific_boundary.get(
            "primary_acceptance_tested"
        )
        is False,
        "Primary acceptance was unexpectedly tested.",
    )

    require(
        scientific_boundary.get(
            "formal_evaluation"
        )
        is False,
        "Formal evaluation was unexpectedly opened.",
    )

    print(
        "Stage-1 gamma         = IMMUTABLE"
    )

    print(
        "Stage-2 margins       = IMMUTABLE"
    )

    print(
        "Stage-3 floors        = IMMUTABLE"
    )

    print(
        "Stage-4 outcomes      = NOT COMPUTED"
    )

    print(
        "Stage-4 winner        = NOT SELECTED"
    )

    print(
        "primary acceptance    = NOT TESTED"
    )

    print(
        "formal evaluation     = NO"
    )

    # ================================================================
    # B. Read exact preregistration
    # ================================================================

    print()
    print("===== B. EXACT STAGE-4 PREREGISTRATION =====")

    prereg = json.loads(
        PREREG.read_text(
            encoding="utf-8"
        )
    )

    temporal = prereg[
        "temporal_smoothing"
    ]

    rate = prereg[
        "actuation_rate_limit"
    ]

    stage4 = prereg[
        "development_selection_protocol"
    ][
        "stage_4_temporal_and_rate"
    ]

    require(
        temporal[
            "candidate_time_constants_ms"
        ]
        ==
        EXPECTED_TIME_CONSTANTS_MS,
        "Temporal candidate set changed.",
    )

    require(
        temporal[
            "initial_state"
        ]
        ==
        EXPECTED_INITIAL_STATE_TEXT,
        "Preregistered initial-state wording changed.",
    )

    require(
        temporal[
            "zero_smoothing_candidate"
        ]
        is False,
        "Unexpected zero-smoothing candidate.",
    )

    require(
        rate[
            "candidate_rates_per_s"
        ]
        ==
        EXPECTED_RATES_PER_S,
        "Rate candidate set changed.",
    )

    require(
        int(
            rate[
                "rate_pair_count"
            ]
        )
        ==
        15,
        "Expected exactly 15 rate pairs.",
    )

    require(
        stage4[
            "lexicographic_selection"
        ]
        ==
        EXPECTED_STAGE4_LEXICOGRAPHIC,
        "Stage-4 lexicographic rule changed.",
    )

    require(
        stage4[
            "all_class_parameters"
        ]
        ==
        "fixed from stages 1-3",
        "Stage-4 no longer fixes Stage1-3 parameters.",
    )

    print(
        "time constants ms =",
        EXPECTED_TIME_CONSTANTS_MS,
    )

    print(
        "rate values / s   =",
        EXPECTED_RATES_PER_S,
    )

    print(
        "zero smoothing    = NO"
    )

    print(
        "class parameters  = FIXED FROM STAGES 1-3"
    )

    print(
        "selection order   = EXACT PREREGISTERED"
    )

    # ================================================================
    # C. Construct exact finite candidate space
    # ================================================================

    print()
    print("===== C. EXACT 60-CANDIDATE SPACE =====")

    rate_pairs = [
        (
            int(rho_dim),
            int(rho_bright),
        )
        for rho_dim in
        EXPECTED_RATES_PER_S
        for rho_bright in
        EXPECTED_RATES_PER_S
        if rho_dim >= rho_bright
    ]

    require(
        len(rate_pairs) == 15,
        "Finite rate pair count != 15.",
    )

    candidates = [
        {
            "time_constant_ms":
                int(time_ms),

            "time_constant_s":
                float(
                    time_ms
                    /
                    1000.0
                ),

            "rho_dim_per_s":
                int(rho_dim),

            "rho_bright_per_s":
                int(rho_bright),

            "canonical_tie_tuple":
                [
                    int(time_ms),
                    int(rho_dim),
                    int(rho_bright),
                ],
        }
        for time_ms in
        EXPECTED_TIME_CONSTANTS_MS
        for rho_dim, rho_bright in
        rate_pairs
    ]

    require(
        len(candidates) == 60,
        "Expected exactly 60 Stage-4 candidates.",
    )

    require(
        len(
            {
                (
                    row[
                        "time_constant_ms"
                    ],
                    row[
                        "rho_dim_per_s"
                    ],
                    row[
                        "rho_bright_per_s"
                    ],
                )
                for row in candidates
            }
        )
        ==
        60,
        "Duplicate Stage-4 candidate tuple.",
    )

    print(
        "rate pairs =",
        len(rate_pairs),
    )

    print(
        "T × rate pairs =",
        len(candidates),
        "EXACT",
    )

    print(
        "canonical final tie tuple = "
        "(T_ms, rho_dim_per_s, rho_bright_per_s)"
    )

    print(
        "candidate outcomes evaluated = NO"
    )

    # ================================================================
    # D. Frozen current-state maps
    # ================================================================

    print()
    print("===== D. FROZEN CAUSAL t0 ACTUATOR STATE =====")

    ledger = read_jsonl(
        LEDGER
    )

    require(
        len(ledger) == 120,
        (
            "Expected 120 reactive ledger rows, "
            f"got {len(ledger)}."
        ),
    )

    maps = np.load(
        T0_MAPS,
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        tuple(
            maps.shape
        )
        ==
        MAP_STACK_SHAPE,
        (
            "Unexpected t0 map stack shape: "
            f"{maps.shape}"
        ),
    )

    require(
        maps.dtype
        ==
        np.dtype(
            np.float64
        ),
        (
            "Unexpected t0 map dtype: "
            f"{maps.dtype}"
        ),
    )

    scenario_ids = []
    row_indices = []
    binding_rows = []

    for ledger_position, row in enumerate(
        ledger
    ):
        scenario_id = str(
            row[
                "scenario_id"
            ]
        )

        row_index = int(
            row[
                "t0_map_row_index"
            ]
        )

        map_shape = tuple(
            int(value)
            for value in
            row[
                "t0_map_shape"
            ]
        )

        map_dtype = str(
            row[
                "t0_map_dtype"
            ]
        )

        map_sha_metadata = str(
            row[
                "t0_map_sha256"
            ]
        )

        reactive = row[
            "original_reactive_ADB"
        ]

        require(
            map_shape
            ==
            GRID_SHAPE,
            (
                "Ledger map shape mismatch for "
                f"{scenario_id}: {map_shape}"
            ),
        )

        require(
            map_dtype
            ==
            "float64",
            (
                "Ledger map dtype mismatch for "
                f"{scenario_id}: {map_dtype}"
            ),
        )

        require(
            re.fullmatch(
                r"[0-9a-f]{64}",
                map_sha_metadata,
            )
            is not None,
            (
                "Invalid per-map SHA metadata for "
                f"{scenario_id}."
            ),
        )

        require(
            reactive[
                "decision_time"
            ]
            ==
            "t0_current_causal_anchor",
            (
                "Reactive decision time changed for "
                f"{scenario_id}."
            ),
        )

        require(
            reactive[
                "score_schedule"
            ]
            ==
            "hold_t0_action_at_0.1_0.3_0.5_1.0s",
            (
                "Reactive schedule semantics changed for "
                f"{scenario_id}."
            ),
        )

        require(
            0 <= row_index < 120,
            (
                "Invalid t0 map row index for "
                f"{scenario_id}: {row_index}"
            ),
        )

        current = np.asarray(
            maps[
                row_index
            ],
            dtype=np.float64,
        )

        require(
            current.shape
            ==
            GRID_SHAPE,
            (
                "Loaded current-state map shape mismatch for "
                f"{scenario_id}."
            ),
        )

        require(
            np.all(
                np.isfinite(
                    current
                )
            ),
            (
                "Non-finite current-state map for "
                f"{scenario_id}."
            ),
        )

        require(
            float(
                np.min(
                    current
                )
            )
            >=
            0.0
            and
            float(
                np.max(
                    current
                )
            )
            <=
            1.0,
            (
                "Current-state map outside [0,1] for "
                f"{scenario_id}."
            ),
        )

        scenario_ids.append(
            scenario_id
        )

        row_indices.append(
            row_index
        )

        binding_rows.append(
            {
                "scenario_id":
                    scenario_id,

                "t0_map_row_index":
                    row_index,

                "t0_map_shape":
                    [
                        501,
                        301,
                    ],

                "t0_map_dtype":
                    "float64",

                "t0_map_sha256_metadata":
                    map_sha_metadata,

                "source_artifact":
                    str(
                        T0_MAPS
                    ),

                "source_artifact_sha256":
                    EXPECTED[
                        T0_MAPS
                    ],

                "decision_time":
                    "t0_current_causal_anchor",

                "future_controller_input":
                    False,

                "stage4_I_prev_role":
                    (
                        "already-issued physical causal "
                        "Part-A reactive actuator command"
                    ),
            }
        )

    require(
        len(
            set(
                scenario_ids
            )
        )
        ==
        120,
        "Reactive ledger scenario IDs are not unique.",
    )

    require(
        sorted(
            row_indices
        )
        ==
        list(
            range(
                120
            )
        ),
        "t0 map row indices are not exactly 0..119.",
    )

    binding_rows.sort(
        key=lambda row:
            row[
                "t0_map_row_index"
            ]
    )

    print(
        "ledger scenarios = 120 / 120"
    )

    print(
        "map stack shape  =",
        maps.shape,
    )

    print(
        "map dtype        =",
        maps.dtype,
    )

    print(
        "finite/in-range  = 120 / 120 PASS"
    )

    print(
        "future input     = NO"
    )

    print(
        "whole NPY seal   = EXACT PASS"
    )

    print(
        "whole ledger seal= EXACT PASS"
    )

    # Do not guess the historical per-array serialization/hash
    # implementation. The exact NPY file SHA + exact ledger SHA already
    # seal both the contents and the recorded row-level hashes.
    print(
        "row SHA metadata = PRESERVED UNDER DUAL IMMUTABLE FILE SEALS"
    )

    # ================================================================
    # E. Explicit pre-outcome semantic resolution
    # ================================================================

    print()
    print("===== E. INITIAL-STATE SEMANTIC RESOLUTION =====")

    print(
        "literal prereg wording = "
        "current causal CLASS-AWARE reactive illumination"
    )

    print(
        "materialized actuator state = "
        "frozen current causal original Part-A reactive t0 map"
    )

    print(
        "silent equivalence claim = NO"
    )

    print(
        "resolved I_prev = already-issued causal t0 actuator command"
    )

    print(
        "class-aware target enters = I_raw + VRU floor guard"
    )

    print(
        "future information in I_prev = NO"
    )

    print(
        "new current-state controller invented = NO"
    )

    print(
        "resolution timing = BEFORE ANY STAGE-4 OUTCOME"
    )

    # ================================================================
    # F. Runtime route verification
    # ================================================================

    print()
    print("===== F. EXACT FINAL-SCHEDULE RUNTIME ROUTE =====")

    from iscai_stage6.adb.class_aware_policy import (
        apply_actuation_rate_limit,
        temporal_smooth_schedule,
    )

    smooth_signature = inspect.signature(
        temporal_smooth_schedule
    )

    rate_signature = inspect.signature(
        apply_actuation_rate_limit
    )

    require(
        "current_illumination"
        in
        smooth_signature.parameters,
        (
            "temporal_smooth_schedule lost "
            "current_illumination."
        ),
    )

    require(
        "current_illumination"
        in
        rate_signature.parameters,
        (
            "apply_actuation_rate_limit lost "
            "current_illumination."
        ),
    )

    require(
        "vru_floor_guard"
        in
        rate_signature.parameters,
        (
            "apply_actuation_rate_limit lost "
            "VRU floor guard."
        ),
    )

    runtime_source = (
        CLASS_RUNTIME.read_text(
            encoding="utf-8"
        )
    )

    require(
        (
            "Initial I_prev is current causal illumination."
            in
            runtime_source
        ),
        "Frozen smoothing I_prev contract missing.",
    )

    require(
        "previous = current.copy()"
        in
        runtime_source,
        "Frozen recursive current-state initialization missing.",
    )

    require(
        "return RateLimitedSchedule("
        in
        runtime_source,
        "Final RateLimitedSchedule constructor missing.",
    )

    print(
        "I_raw -> temporal_smooth_schedule = EXACT"
    )

    print(
        "smoothing I_prev = frozen scenario t0 map"
    )

    print(
        "smoothed -> apply_actuation_rate_limit = EXACT"
    )

    print(
        "rate I_prev = SAME frozen scenario t0 map"
    )

    print(
        "VRU guard after rate limit = REQUIRED"
    )

    print(
        "I_final = RateLimitedSchedule.illumination"
    )

    # ================================================================
    # G. Freeze exact Stage-4 scoring semantics
    # ================================================================

    print()
    print("===== G. STAGE-4 SCORING SEMANTICS =====")

    risk_components = [
        "vehicle_shadow_zone_violation",
        "glare_risk_exposure",
        "1-pedestrian_visibility_proxy",
        "1-cyclist_visibility_proxy",
        "1-road_illumination_retention",
    ]

    selection_key = [
        "max(macro_safety_risk_vector)",
        "mean(macro_safety_risk_vector)",
        "macro_temporal_mean_absolute_change_rate",
        "macro_flicker_change_rate",
        "macro_rate_limit_safety_override_fraction",
        "macro_over_masking_area",
        "time_constant_ms",
        "rho_dim_per_s",
        "rho_bright_per_s",
    ]

    print(
        "risk vector =",
        risk_components,
    )

    print(
        "metric authority = frozen Block6.6/reconciled runtime"
    )

    print(
        "candidate I_final = final rate-limited illumination"
    )

    print(
        "scenario weighting = equal scenario macro"
    )

    print(
        "class-specific NA values = excluded exactly as metric contract"
    )

    print(
        "outcome-dependent normalization = NO"
    )

    print(
        "weighted-sum objective = NO"
    )

    print(
        "epsilon tie tolerance = NONE"
    )

    print(
        "round-before-selection = NO"
    )

    print(
        "exact key ="
    )

    for item in selection_key:
        print(
            "  ",
            item,
        )

    # ================================================================
    # H. Write the immutable scenario binding
    # ================================================================

    print()
    print("===== H. WRITE-ONCE CURRENT-STATE BINDING =====")

    binding_state = write_once(
        BINDING,
        canonical_jsonl_bytes(
            binding_rows
        ),
    )

    binding_sha = file_sha(
        BINDING
    )

    print(
        "binding state =",
        binding_state,
    )

    print(
        "binding rows  =",
        len(
            binding_rows
        ),
    )

    print(
        "binding SHA   =",
        binding_sha,
    )

    # ================================================================
    # I. Write the immutable Stage-4 scoring freeze
    # ================================================================

    print()
    print("===== I. WRITE-ONCE STAGE-4 SCORING FREEZE =====")

    freeze_payload = {
        "stage":
            6,

        "block":
            "6.8_stage4_temporal_rate_preoutcome_freeze",

        "status":
            "FROZEN_STAGE4_PREOUTCOME_INITIAL_STATE_AND_SCORING",

        "scientific_boundary": {
            "Stage1_gamma":
                "IMMUTABLE",

            "Stage1_gamma_freeze_sha256":
                STAGE1_GAMMA_FREEZE_SHA,

            "Stage2_margins":
                "IMMUTABLE",

            "Stage2_margin_freeze_sha256":
                STAGE2_MARGIN_FREEZE_SHA,

            "Stage3_floors":
                "IMMUTABLE",

            "Stage3_floor_freeze_sha256":
                EXPECTED[
                    STAGE3_FREEZE
                ],

            "Stage4_outcomes_computed":
                False,

            "Stage4_winner_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,

            "post_outcome_tuning_allowed":
                False,
        },

        "frozen_stage1_gamma": {
            "TYPE_VEHICLE": {
                "k": 980,
                "gamma": 980 / 8192,
            },

            "TYPE_PEDESTRIAN": {
                "k": 499,
                "gamma": 499 / 8192,
            },

            "TYPE_CYCLIST": {
                "k": 557,
                "gamma": 557 / 8192,
            },
        },

        "frozen_stage2_additional_margin_policy": {
            "TYPE_VEHICLE": {
                "base_margin_m": 0.0,
                "uncertainty_multiplier": 0.5,
                "motion_multiplier": 0.05,
            },

            "TYPE_PEDESTRIAN": {
                "base_margin_m": 0.0,
                "uncertainty_multiplier": 0.5,
                "motion_multiplier": 0.0,
            },

            "TYPE_CYCLIST": {
                "base_margin_m": 0.0,
                "uncertainty_multiplier": 0.5,
                "motion_multiplier": 0.25,
            },

            "predictive_dilation_input":
                "additional_class_margin_m_only",

            "PartA_generic_margin":
                "preserved separately; not double applied",
        },

        "frozen_stage3_class_floors": {
            "TYPE_VEHICLE":
                0.15,

            "TYPE_PEDESTRIAN":
                1.0,

            "TYPE_CYCLIST":
                1.0,
        },

        "current_state_semantic_resolution": {
            "preregistered_literal":
                EXPECTED_INITIAL_STATE_TEXT,

            "materialized_authoritative_state":
                (
                    "already-issued current causal t0 "
                    "Part-A reactive actuator command"
                ),

            "I_prev_source":
                str(
                    T0_MAPS
                ),

            "I_prev_source_sha256":
                EXPECTED[
                    T0_MAPS
                ],

            "scenario_binding":
                str(
                    BINDING
                ),

            "scenario_binding_sha256":
                binding_sha,

            "ledger":
                str(
                    LEDGER
                ),

            "ledger_sha256":
                EXPECTED[
                    LEDGER
                ],

            "resolution_type":
                "EXPLICIT_PREOUTCOME_SEMANTIC_RESOLUTION",

            "silent_equivalence_between_original_reactive_and_class_aware_reactive":
                False,

            "new_t0_policy_invented":
                False,

            "future_information_used":
                False,

            "class_aware_policy_enters_after_I_prev_via":
                [
                    "raw_class_aware_illumination",
                    "vru_floor_guard",
                    "frozen Stage1 gamma",
                    "frozen Stage2 additional margins",
                    "frozen Stage3 class floors",
                ],

            "reason":
                (
                    "Temporal smoothing and finite-rate actuation "
                    "require the physically previous actuator command. "
                    "The only already-realized and immutably frozen "
                    "scenario-specific causal command is the exact t0 "
                    "Part-A reactive map. The previous structural all-on "
                    "map was explicitly smoke-only and is forbidden for "
                    "objective evaluation."
                ),
        },

        "horizons_s":
            list(
                HORIZONS_S
            ),

        "candidate_space": {
            "time_constants_ms":
                EXPECTED_TIME_CONSTANTS_MS,

            "rate_values_per_s":
                EXPECTED_RATES_PER_S,

            "rate_constraint":
                "rho_dim_per_s >= rho_bright_per_s",

            "rate_pair_count":
                15,

            "candidate_count":
                60,

            "candidates":
                candidates,

            "canonical_tie_tuple":
                [
                    "time_constant_ms",
                    "rho_dim_per_s",
                    "rho_bright_per_s",
                ],

            "candidate_numerics_changed_from_preregistration":
                False,
        },

        "runtime_route": {
            "raw_schedule":
                (
                    "frozen Stage1+Stage2 class masks converted through "
                    "preserved Part-A illumination and frozen Stage3 floors"
                ),

            "composition_output":
                "ClassAwareComposition.raw_class_aware_illumination",

            "smoothing":
                "temporal_smooth_schedule",

            "smoothing_initial_state":
                "scenario-specific frozen I_prev t0 map",

            "rate_limit":
                "apply_actuation_rate_limit",

            "rate_initial_state":
                "same scenario-specific frozen I_prev t0 map",

            "VRU_floor_guard":
                (
                    "ClassAwareComposition.vru_floor_guard "
                    "re-applied by rate limiter"
                ),

            "final_output":
                "RateLimitedSchedule.illumination",
        },

        "scoring": {
            "authoritative_metric_runtime":
                str(
                    METRIC_RUNTIME
                ),

            "authoritative_metric_runtime_sha256":
                EXPECTED[
                    METRIC_RUNTIME
                ],

            "evaluator_runtime":
                str(
                    EVALUATOR_RUNTIME
                ),

            "evaluator_runtime_sha256":
                EXPECTED[
                    EVALUATOR_RUNTIME
                ],

            "road_roi":
                str(
                    ROAD_ROI
                ),

            "road_roi_sha256":
                EXPECTED[
                    ROAD_ROI
                ],

            "safety_risk_vector_after_scenario_macro":
                risk_components,

            "scenario_aggregation":
                (
                    "arithmetic macro mean across frozen development "
                    "scenarios for each authoritative metric; metric-defined "
                    "NA values excluded only where contract requires"
                ),

            "outcome_dependent_normalization":
                False,

            "arbitrary_weighted_sum":
                False,

            "round_before_comparison":
                False,

            "epsilon_tie_tolerance":
                None,

            "selection_key_exact_order":
                selection_key,

            "temporal_smoothness_semantics":
                (
                    "mean_h(mean_grid(|I_h-I_prev|)/delta_t_h), "
                    "lower is better"
                ),

            "flicker_semantics":
                (
                    "mean_h(count(I_h != I_prev)/grid_cell_count), "
                    "lower is better"
                ),

            "rate_limit_safety_override_fraction":
                (
                    "override cells / total actuator-cell transitions, "
                    "lower is better"
                ),

            "over_masking_area":
                "|D \\ O| / |full actuator grid|",
        },

        "prohibited_until_winner_is_frozen": [
            "primary acceptance gate",
            "formal evaluation",
            "formal outcome read",
            "post-outcome change to I_prev semantics",
            "post-outcome change to candidate set",
            "post-outcome change to aggregation",
            "post-outcome change to selection key",
            "post-outcome retuning of Stage1 gamma",
            "post-outcome retuning of Stage2 margins",
            "post-outcome retuning of Stage3 floors",
        ],

        "next":
            (
                "Execute exactly the frozen 60 Stage-4 temporal/rate "
                "development candidates once, choose the exact "
                "lexicographic winner, authoritative replay it, "
                "and freeze it before primary acceptance."
            ),
    }

    freeze_state = write_once(
        FREEZE,
        canonical_json_bytes(
            freeze_payload
        ),
    )

    freeze_sha = file_sha(
        FREEZE
    )

    print(
        "freeze state =",
        freeze_state,
    )

    print(
        "freeze SHA   =",
        freeze_sha,
    )

    # ================================================================
    # J. Regression
    # ================================================================

    print()
    print("===== J. POST-FREEZE REGRESSION =====")

    tests = full_regression()

    print(
        "Stage6 regression =",
        f"{tests} / {tests} PASS",
    )

    # ================================================================
    # K. Final report
    # ================================================================

    report_payload = {
        "stage":
            6,

        "block":
            "6.8_stage4_initial_state_scoring_freeze",

        "status":
            "PASS_STAGE4_INITIAL_STATE_AND_SCORING_FROZEN",

        "current_state_binding": {
            "rows":
                120,

            "artifact":
                str(
                    BINDING
                ),

            "sha256":
                binding_sha,

            "t0_maps_sha256":
                EXPECTED[
                    T0_MAPS
                ],

            "ledger_sha256":
                EXPECTED[
                    LEDGER
                ],
        },

        "stage4_scoring_freeze": {
            "path":
                str(
                    FREEZE
                ),

            "sha256":
                freeze_sha,

            "candidate_count":
                60,

            "rate_pair_count":
                15,
        },

        "semantic_resolution": {
            "I_prev":
                (
                    "already-issued current causal frozen "
                    "Part-A t0 actuator map"
                ),

            "all_on_smoke_allowed":
                False,

            "future_information":
                False,

            "new_current_state_policy":
                False,

            "resolution_before_stage4_outcomes":
                True,
        },

        "scientific_boundary": {
            "Stage1_gamma":
                "IMMUTABLE",

            "Stage2_margins":
                "IMMUTABLE",

            "Stage3_floors":
                "IMMUTABLE",

            "Stage4_candidate_outcomes_computed":
                False,

            "Stage4_winner_selected":
                False,

            "primary_acceptance_tested":
                False,

            "formal_evaluation":
                False,

            "policy_tuning":
                False,
        },

        "Stage6_regression":
            tests,

        "next":
            "RUN_EXACT_60_STAGE4_TEMPORAL_RATE_DEVELOPMENT_SWEEP",
    }

    report_state = write_once(
        REPORT,
        canonical_json_bytes(
            report_payload
        ),
    )

    report_sha = file_sha(
        REPORT
    )

    print()
    print("=" * 78)
    print("BLOCK 6.8 STAGE-4 PRE-OUTCOME FREEZE — FINAL")
    print("=" * 78)

    print(
        "Stage-1 gamma             = IMMUTABLE"
    )

    print(
        "Stage-2 margins           = IMMUTABLE"
    )

    print(
        "Stage-3 floors            = IMMUTABLE"
    )

    print(
        "current I_prev scenarios  = 120 / 120"
    )

    print(
        "I_prev source             = FROZEN CAUSAL t0 ACTUATOR MAP"
    )

    print(
        "all-on structural smoke   = FORBIDDEN FOR SCORING"
    )

    print(
        "Stage-4 rate pairs        = 15 EXACT"
    )

    print(
        "Stage-4 candidates        = 60 EXACT"
    )

    print(
        "Stage-4 outcomes computed = NO"
    )

    print(
        "Stage-4 winner selected   = NO"
    )

    print(
        "primary acceptance        = NOT TESTED"
    )

    print(
        "formal evaluation         = NO"
    )

    print(
        "Stage6 regression         =",
        f"{tests} / {tests} PASS",
    )

    print(
        "binding SHA               =",
        binding_sha,
    )

    print(
        "Stage-4 freeze SHA        =",
        freeze_sha,
    )

    print(
        "report state              =",
        report_state,
    )

    print(
        "report SHA                =",
        report_sha,
    )

    print(
        "STATUS = "
        "PASS_STAGE4_INITIAL_STATE_AND_SCORING_FROZEN"
    )

    print(
        "report =",
        REPORT,
    )

    print(
        "NEXT = EXACT 60-CANDIDATE STAGE-4 SWEEP"
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:

    print()
    print("=" * 78)
    print(
        "BLOCK 6.8 STAGE-4 PRE-OUTCOME FREEZE = BLOCKED"
    )
    print("=" * 78)

    print(
        "exception =",
        type(exc).__name__,
        str(exc),
    )

    traceback.print_exc()

    print()
    print(
        "Stage-1 gamma       = STILL IMMUTABLE"
    )

    print(
        "Stage-2 margins     = STILL IMMUTABLE"
    )

    print(
        "Stage-3 floors      = STILL IMMUTABLE"
    )

    print(
        "Stage-4 sweep       = NOT ALLOWED"
    )

    print(
        "Stage-4 winner      = NOT SELECTED"
    )

    print(
        "primary acceptance  = NOT TESTED"
    )

    print(
        "formal evaluation   = NO"
    )

    print(
        "Do not alter I_prev, candidate numerics, "
        "or Stage1/2/3 policy parameters."
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no non-zero sys.exit().

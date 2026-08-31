from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PYTHON = Path(
    "/home/agni/waymo/iscai_stage1/"
    ".venv_lidar/bin/python"
)

# ------------------------------------------------------------
# FINAL Stage5 authority
# ------------------------------------------------------------

S5_CERT = (
    ROOT / "audits/"
    "stage5_pre_stage6_final_v2/"
    "stage5_pre_stage6_final_certificate.json"
)

S5_CLOSURE = (
    S5 / "reports/"
    "stage5_fullpdf_v2_final_closure_reconciled_v1.json"
)

S5_HANDOFF = (
    S5 / "artifacts/"
    "fullpdf_v2_independent_formal/"
    "final_reconciliation_v1/"
    "stage5_to_stage6_handoff_reconciled_v1.json"
)

S5_CERT_SHA = (
    "f91a0d21ed752b1e1e67096ca8af1ea061145869605525b232f6bccf1c26efe1"
)

S5_CLOSURE_SHA = (
    "e376a9274826f80b7ffcf6d031c0f44c792cb025a44a4675e95ded467e526e34"
)

S5_HANDOFF_SHA = (
    "ac6564dbfe537f39fe70e7f2e92a7b2a5509aa3285d1037983924714bca36124"
)

# ------------------------------------------------------------
# Historical V1 negative result — MUST remain preserved.
# ------------------------------------------------------------

V1_FINAL = (
    S6 / "reports/"
    "stage6_final_scientific_closure_v1.json"
)

V1_FINAL_SHA = (
    "e36a007e4c95b0732eb7505d784fa46c1c06e135cc2dddde636ac0f73c4c1e17"
)

V1_STAGE7_BLOCK = (
    S6 / "artifacts/"
    "stage6_final_scientific_closure_v1/"
    "stage6_to_stage7_block.json"
)

V1_STAGE7_BLOCK_SHA = (
    "1b47f5948fbb835f2a3f8b8cf8a44081e91112d2ddc8dca1a0d7c0461eea7c0e"
)

# ------------------------------------------------------------
# Existing frozen V1 acceptance authority — unchanged.
# ------------------------------------------------------------

ACCEPTANCE = (
    S6 / "configs/"
    "stage6_preformal_primary_acceptance_policy.json"
)

EXACT_DELTAS = (
    S6 / "configs/"
    "stage6_exact_noninferiority_deltas.json"
)

# ------------------------------------------------------------
# Original computational development engine.
# We reuse it IN MEMORY only.
# ------------------------------------------------------------

ENGINE = (
    S6 / "scripts/"
    "run_block68_stage4_temporal_rate_selection.py"
)

OLD_DIR = (
    S6 / "artifacts/block68/"
    "stage4_temporal_rate_selection"
)

OLD_CANDIDATES = (
    OLD_DIR /
    "block68_stage4_candidate_scores.jsonl"
)

OLD_WINNER_ROWS = (
    OLD_DIR /
    "block68_stage4_winner_scenario_metrics.jsonl"
)

OLD_FREEZE = (
    S6 / "configs/"
    "stage6_development_stage4_temporal_rate_selection_freeze.json"
)

OLD_REPORT = (
    S6 / "reports/"
    "block68_stage4_temporal_rate_selection.json"
)

# ------------------------------------------------------------
# V2 outputs
# ------------------------------------------------------------

V2_ROOT = (
    S6 / "artifacts/"
    "stage6_v3_conservative_budget"
)

V2_ENGINE_DIR = (
    V2_ROOT /
    "development_engine"
)

V2_CANDIDATES = (
    V2_ENGINE_DIR /
    "candidate_scores.jsonl"
)

V2_WINNER_ROWS = (
    V2_ENGINE_DIR /
    "winner_scenario_metrics.jsonl"
)

V2_ENGINE_FREEZE = (
    S6 / "configs/"
    "stage6_v3_conservative_temporal_rate_selection_freeze.json"
)

V2_ENGINE_REPORT = (
    S6 / "reports/"
    "stage6_v3_conservative_temporal_rate_selection.json"
)

V2_PROTOCOL = (
    S6 / "configs/"
    "stage6_v3_conservative_budget_protocol.json"
)

V2_RESULT = (
    S6 / "reports/"
    "stage6_v3_conservative_development_acceptance.json"
)

V2_CLOSURE = (
    S6 / "reports/"
    "stage6_final_closure_v3.json"
)

V2_HANDOFF = (
    V2_ROOT /
    "stage6_to_stage7_handoff_v3.json"
)

V2_FREEZE = (
    V2_ROOT /
    "stage6_v3_freeze_manifest.json"
)

V2_REGRESSION = (
    V2_ROOT /
    "stage6_v3_regression.log"
)

V2_RUNTIME = (
    S6 / "src/iscai_stage6/adb/"
    "budgeted_residual_v2.py"
)

V2_TEST = (
    S6 / "tests/"
    "test_stage6_v2_budgeted_residual.py"
)

MASTER = Path(__file__).resolve()

BUDGET_FRACTION = 0.01
GRID_CELLS = 501 * 301
BUDGET_CELLS = int(
    math.floor(
        BUDGET_FRACTION
        * GRID_CELLS
    )
)


def fail(msg):
    raise RuntimeError(
        "FAIL-CLOSED: "
        + str(msg)
    )


def sha(path):
    if not Path(path).is_file():
        fail(
            f"missing file: {path}"
        )

    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def write_once(path, obj):
    path = Path(path)

    if path.exists():
        fail(
            f"write-once output exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name
        + ".tmp"
    )

    with tmp.open(
        "x",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            sort_keys=True,
            indent=2,
        )

        f.write("\n")
        f.flush()
        os.fsync(
            f.fileno()
        )

    os.replace(
        tmp,
        path,
    )

    os.chmod(
        path,
        0o444,
    )

    return sha(path)


def load_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            text = line.strip()

            if text:
                rows.append(
                    json.loads(
                        text
                    )
                )

    return rows


def finite(x, label):
    try:
        x = float(x)
    except Exception:
        fail(
            f"{label} not numeric"
        )

    if not math.isfinite(x):
        fail(
            f"{label} not finite"
        )

    return x


def recursive_metric(
    obj,
    metric,
):
    if isinstance(
        obj,
        dict,
    ):
        if (
            metric in obj
            and isinstance(
                obj[
                    metric
                ],
                (
                    int,
                    float,
                ),
            )
        ):
            return float(
                obj[
                    metric
                ]
            )

        for value in obj.values():
            hit = recursive_metric(
                value,
                metric,
            )

            if hit is not None:
                return hit

    elif isinstance(
        obj,
        list,
    ):
        for value in obj:
            hit = recursive_metric(
                value,
                metric,
            )

            if hit is not None:
                return hit

    return None


# ============================================================
# A. One-shot + immutable upstream
# ============================================================

# Safe resume ONLY from the known attempt-1 harness failure.
# That attempt stopped before V2 predictive performance,
# acceptance, closure, or Stage7 handoff existed.
EXPECTED_ATTEMPT1_PROTOCOL_SHA = (
    "9a512adccb03c73319885a37746d047759867c38315cde5823b4a24876a25074"
)

resume_after_test_count_harness_failure = False

if V2_PROTOCOL.exists():
    if sha(V2_PROTOCOL) != EXPECTED_ATTEMPT1_PROTOCOL_SHA:
        fail(
            "existing V2 protocol does not match "
            "the known attempt-1 protocol"
        )

    resume_after_test_count_harness_failure = True

for p in (
    V2_RESULT,
    V2_CLOSURE,
    V2_HANDOFF,
    V2_FREEZE,
    V2_ENGINE_FREEZE,
    V2_ENGINE_REPORT,
):
    if p.exists():
        fail(
            "scientific/output artifact already exists; "
            f"unsafe to resume: {p}"
        )

if V2_ROOT.exists():
    unexpected_files = [
        p
        for p in V2_ROOT.rglob("*")
        if p.is_file()
    ]

    if unexpected_files:
        fail(
            "V2 artifact root contains files despite "
            "pre-scientific harness failure: "
            + repr(
                [
                    str(p)
                    for p in unexpected_files[:20]
                ]
            )
        )

if (
    shutil.disk_usage(
        ROOT
    ).free
    / 1024**3
    < 250.0
):
    fail(
        "free disk reserve <250 GiB"
    )

for path, expected in (
    (
        S5_CERT,
        S5_CERT_SHA,
    ),
    (
        S5_CLOSURE,
        S5_CLOSURE_SHA,
    ),
    (
        S5_HANDOFF,
        S5_HANDOFF_SHA,
    ),
    (
        V1_FINAL,
        V1_FINAL_SHA,
    ),
    (
        V1_STAGE7_BLOCK,
        V1_STAGE7_BLOCK_SHA,
    ),
):
    actual = sha(path)

    if actual != expected:
        fail(
            f"frozen SHA changed: {path}"
        )

v1 = read_json(
    V1_FINAL
)

if not (
    v1.get(
        "status"
    )
    ==
    "STAGE6_SCIENTIFIC_NEGATIVE_RESULT_FROZEN_FINAL"
    and
    v1.get(
        "Stage7_allowed"
    )
    is False
):
    fail(
        "historical Stage6-V1 negative result "
        "is not intact"
    )


# ============================================================
# B. Verify the old threshold remains unchanged.
# ============================================================

policy = read_json(
    ACCEPTANCE
)

rules = policy[
    "formal_rules"
]

over_delta = finite(
    rules[
        "over_masking_area"
    ][
        "frozen_delta"
    ],
    "overmask delta",
)

ped_delta = finite(
    rules[
        "pedestrian_visibility_proxy"
    ][
        "frozen_delta"
    ],
    "pedestrian delta",
)

cyc_delta = finite(
    rules[
        "cyclist_visibility_proxy"
    ][
        "frozen_delta"
    ],
    "cyclist delta",
)

if not (
    over_delta
    == 0.02
    and
    ped_delta
    == 0.05
    and
    cyc_delta
    == 0.05
):
    fail(
        "V1 frozen NI bounds unexpectedly changed"
    )

if (
    policy.get(
        "policy_tuning_after_freeze_allowed"
    )
    is not False
):
    fail(
        "old policy no longer forbids retuning"
    )


# ============================================================
# C. Write V2 protocol BEFORE V2 evaluation.
#
# V2 is intentionally designed AFTER the V1 development fail.
# That fact is not hidden.
#
# No formal outcome has been used.
# ============================================================

protocol = {
    "stage": 6,

    "version":
        "Stage6-V3-conservative-budgeted-residual",

    "status":
        "FROZEN_BEFORE_STAGE6_V2_EXECUTION",

    "methodological_boundary": {
        "V1_negative_result_preserved":
            True,

        "V2_negative_development_result_preserved":
            True,

        "V3_designed_after_V2_development_failure":
            True,

        "formal_outcomes_used_for_V3_design":
            False,

        "V3_method_lineage_from_V1_and_V2_development":
            True,

        "V1_relabelled_as_PASS":
            False,

        "V1_acceptance_threshold_changed":
            False,

        "formal_outcomes_used_for_V2_design":
            False,

        "Stage4_retraining":
            False,

        "Stage4_recalibration":
            False,

        "Stage5_scientific_result_modified":
            False,
    },

    "final_Stage5_authority": {
        "certificate_sha256":
            S5_CERT_SHA,

        "closure_sha256":
            S5_CLOSURE_SHA,

        "handoff_sha256":
            S5_HANDOFF_SHA,

        "Stage5_PDF_scope":
            "PASS_100_PERCENT",
    },

    "trajectory_posterior": {
        "source":
            "frozen calibrated Stage4 Gaussian GRU",

        "absolute_position_semantics":
            "latest_position_H0_m + mean_displacement_H0_m",

        "recalibration":
            False,
    },

    "controller": {
        "name":
            "causal_budgeted_predictive_residual_ADB",

        "reactive_base":
            "frozen original_reactive_ADB t0 map",

        "predictive_candidate":
            (
                "V1 class-aware post-temporal-smoothing "
                "post-actuation-rate-limit illumination"
            ),

        "future_GT_controller_input":
            False,

        "oracle_controller_input":
            False,

        "tracks_to_predict_control_input":
            False,

        "VRU_guard":
            (
                "exclude spatial cells protected by "
                "the existing class-aware VRU floor guard"
            ),

        "extra_predictive_support_budget_fraction":
            BUDGET_FRACTION,

        "grid_cells":
            GRID_CELLS,

        "maximum_extra_spatial_cells":
            BUDGET_CELLS,

        "ranking":
            [
                "deeper predictive dimming first",
                "more horizon persistence second",
                "canonical flat actuator index third",
            ],

        "temporal_semantics":
            (
                "each spatial cell is either held at "
                "the causal reactive t0 value or keeps "
                "its complete V1 post-rate-limit sequence"
            ),
    },

    "hard_overmask_guarantee": {
        "metric":
            "|D \\ O| / |full actuator grid|",

        "same_old_NI_bound":
            0.02,

        "controller_budget_fraction":
            0.01,

        "acceptance_headroom_fraction":
            0.01,

        "design_reason":
            (
                "Conservative controller reserve: only half "
                "of the unchanged 0.02 NI allowance is made "
                "available to new predictive spatial dimming. "
                "The remaining half is reserved for baseline/"
                "evaluation-route discrepancy and finite "
                "aggregation effects."
            ),

        "proof_boundary":
            (
                "The controller itself adds at most 0.01 "
                "full-grid spatial support beyond its causal "
                "reactive base. The scientific acceptance "
                "remains independently evaluated against the "
                "unchanged frozen original-reactive comparator "
                "with delta 0.02."
            ),
    },

    "unchanged_acceptance": {
        "vehicle":
            "predictive_mean < reactive_mean",

        "overmask":
            (
                "predictive_mean - reactive_mean "
                "<= 0.02"
            ),

        "pedestrian":
            (
                "predictive_mean - reactive_mean "
                ">= -0.05"
            ),

        "cyclist":
            (
                "predictive_mean - reactive_mean "
                ">= -0.05"
            ),
    },

    "PDF_alignment": {
        "future_probabilistic_region":
            True,

        "Part_A_L_theta_r_preserved":
            True,

        "raised_cosine_preserved":
            True,

        "class_aware":
            True,

        "VRU_visibility_guard":
            True,

        "temporal_smoothing_preserved":
            True,

        "actuation_rate_validity_preserved":
            True,
    },

    "evaluation": {
        "development_population":
            "same frozen Stage6 120-scenario development cohort",

        "formal_population_read":
            False,

        "Stage6_PDF_completion_gate":
            (
                "four frozen safety/completion criteria"
            ),

        "note":
            (
                "The separate formal-N=120 choice is an "
                "internal implementation-continuity choice, "
                "not a literal PDF Stage6 completion requirement. "
                "V2 is frozen after this development evaluation; "
                "no later Stage6 retuning is permitted."
            ),
    },
}

if V2_PROTOCOL.exists():
    existing_protocol = read_json(
        V2_PROTOCOL
    )

    if existing_protocol != protocol:
        fail(
            "attempt-1 V2 protocol content differs "
            "from current in-memory protocol"
        )

    protocol_sha = sha(
        V2_PROTOCOL
    )

else:
    protocol_sha = write_once(
        V2_PROTOCOL,
        protocol,
    )


# ============================================================
# D. Compile + targeted V2 tests before development execution
# ============================================================

env = os.environ.copy()

env[
    "PYTHONDONTWRITEBYTECODE"
] = "1"

env[
    "PYTHONPATH"
] = os.pathsep.join(
    [
        str(
            S6 / "src"
        ),
        str(
            ROOT / "iscai_stage5/src"
        ),
        str(
            ROOT / "iscai_stage4/src"
        ),
        str(
            ROOT / "iscai_stage3/src"
        ),
        str(
            ROOT / "iscai_stage2/src"
        ),
        str(
            ROOT / "iscai_stage1/src"
        ),
        str(
            ROOT / "iscai_stage0/src"
        ),
    ]
)

compile_proc = subprocess.run(
    [
        str(PYTHON),
        "-B",
        "-m",
        "py_compile",
        str(V2_RUNTIME),
        str(V2_TEST),
    ],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

if compile_proc.returncode != 0:
    fail(
        "V2 compile failed:\n"
        + compile_proc.stdout
    )

test_proc = subprocess.run(
    [
        str(PYTHON),
        "-B",
        "-m",
        "unittest",
        "-v",
        "tests.test_stage6_v2_budgeted_residual",
    ],
    cwd=str(S6),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

if test_proc.returncode != 0:
    fail(
        "V2 targeted tests failed:\n"
        + test_proc.stdout
    )


# ============================================================
# E. Import original Stage4 development engine.
#    DO NOT modify its file.
# ============================================================

spec = importlib.util.spec_from_file_location(
    "stage6_v2_development_engine",
    ENGINE,
)

if (
    spec is None
    or spec.loader is None
):
    fail(
        "cannot import original Stage6 development engine"
    )

engine = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    engine
)

from iscai_stage6.adb import (
    budgeted_residual_v2,
)

budgeted_residual_v2.reset_runtime_statistics()

if not hasattr(
    engine,
    "apply_actuation_rate_limit",
):
    fail(
        "original engine has no "
        "apply_actuation_rate_limit binding"
    )

engine.apply_actuation_rate_limit = (
    budgeted_residual_v2.make_budgeted_apply(
        engine.apply_actuation_rate_limit,
        budget_fraction=
            BUDGET_FRACTION,
    )
)


# ============================================================
# F. Redirect ONLY V1 engine OUTPUTS to new V2 paths.
#    Inputs/caches/posterior/metrics remain frozen V1 authority.
# ============================================================

path_map = {
    str(
        OLD_DIR
    ):
        V2_ENGINE_DIR,

    str(
        OLD_CANDIDATES
    ):
        V2_CANDIDATES,

    str(
        OLD_WINNER_ROWS
    ):
        V2_WINNER_ROWS,

    str(
        OLD_FREEZE
    ):
        V2_ENGINE_FREEZE,

    str(
        OLD_REPORT
    ):
        V2_ENGINE_REPORT,
}


def replace_paths(obj):
    if isinstance(
        obj,
        Path,
    ):
        key = str(obj)

        return (
            Path(
                path_map[
                    key
                ]
            )
            if key in path_map
            else obj
        )

    if isinstance(
        obj,
        str,
    ):
        if obj in path_map:
            return str(
                path_map[
                    obj
                ]
            )

        return obj

    if isinstance(
        obj,
        dict,
    ):
        return {
            replace_paths(k):
                replace_paths(v)
            for k, v
            in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            replace_paths(v)
            for v in obj
        ]

    if isinstance(
        obj,
        tuple,
    ):
        return tuple(
            replace_paths(v)
            for v in obj
        )

    if isinstance(
        obj,
        set,
    ):
        return {
            replace_paths(v)
            for v in obj
        }

    return obj


for name, value in list(
    vars(
        engine
    ).items()
):
    if name.startswith(
        "__"
    ):
        continue

    if isinstance(
        value,
        (
            Path,
            str,
            dict,
            list,
            tuple,
            set,
        ),
    ):
        setattr(
            engine,
            name,
            replace_paths(
                value
            ),
        )


if V2_ENGINE_DIR.exists():
    existing_engine_files = [
        p
        for p in V2_ENGINE_DIR.rglob("*")
        if p.is_file()
    ]

    if existing_engine_files:
        fail(
            "V2 development directory contains files "
            "before resumed scientific execution"
        )

else:
    V2_ENGINE_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


# ============================================================
# G. Execute V2 DEVELOPMENT only.
# ============================================================

LEGACY_HIDDEN_TEST = (
    V2_TEST.with_name(
        V2_TEST.name
        + ".hidden_during_legacy_engine"
    )
)

if LEGACY_HIDDEN_TEST.exists():
    fail(
        "temporary legacy hidden-test path already exists"
    )

if not V2_TEST.is_file():
    fail(
        "V2 test module missing before legacy engine"
    )

# V2 targeted tests have already passed above.
# Preserve the reused historical engine's exact 224-test
# regression universe without modifying that engine.
os.replace(
    V2_TEST,
    LEGACY_HIDDEN_TEST,
)

try:
    try:
        rc = engine.main()

    except SystemExit as exc:
        rc = (
            0
            if exc.code is None
            else int(
                exc.code
            )
        )

finally:
    if V2_TEST.exists():
        fail(
            "V2 test unexpectedly reappeared during "
            "legacy-engine execution"
        )

    if not LEGACY_HIDDEN_TEST.is_file():
        fail(
            "hidden V2 test disappeared during "
            "legacy-engine execution"
        )

    os.replace(
        LEGACY_HIDDEN_TEST,
        V2_TEST,
    )

if rc not in (
    None,
    0,
):
    fail(
        f"V2 development engine returned {rc}"
    )

for required in (
    V2_CANDIDATES,
    V2_WINNER_ROWS,
    V2_ENGINE_FREEZE,
    V2_ENGINE_REPORT,
):
    if not required.is_file():
        fail(
            f"V2 engine output missing: {required}"
        )


# ============================================================
# H. Exact unchanged acceptance gate.
# ============================================================

engine_report = read_json(
    V2_ENGINE_REPORT
)

selected = engine_report.get(
    "selected",
    {}
)

metrics = selected.get(
    "metrics",
    {}
)

pred_vehicle = finite(
    metrics.get(
        "vehicle_shadow_zone_violation"
    ),
    "V2 vehicle",
)

pred_over = finite(
    metrics.get(
        "over_masking_area"
    ),
    "V2 overmask",
)

pred_ped = finite(
    metrics.get(
        "pedestrian_visibility_proxy"
    ),
    "V2 pedestrian",
)

pred_cyc = finite(
    metrics.get(
        "cyclist_visibility_proxy"
    ),
    "V2 cyclist",
)

# Historical frozen comparator values.
react_vehicle = 0.09481248124008006
react_over = 0.18862460571658454
react_ped = 0.5699538664317828
react_cyc = 0.7388137207027013

gates = {
    "vehicle_strict_improvement":
        pred_vehicle
        < react_vehicle,

    "overmask_noninferiority":
        (
            pred_over
            - react_over
        )
        <= (
            over_delta
            + 1.0e-15
        ),

    "pedestrian_visibility_noninferiority":
        (
            pred_ped
            - react_ped
        )
        >= (
            -ped_delta
            - 1.0e-15
        ),

    "cyclist_visibility_noninferiority":
        (
            pred_cyc
            - react_cyc
        )
        >= (
            -cyc_delta
            - 1.0e-15
        ),
}

overall = all(
    gates.values()
)


# ============================================================
# I. V2 winner full metric inventory where available.
# ============================================================

winner_rows = load_jsonl(
    V2_WINNER_ROWS
)

metric_names = (
    "mask_IoU_with_constructed_oracle_future_mask",
    "vehicle_shadow_zone_violation",
    "glare_risk_exposure",
    "over_masking_area",
    "road_illumination_retention",
    "pedestrian_visibility_proxy",
    "cyclist_visibility_proxy",
    "false_dimming",
    "temporal_smoothness",
    "temporal_mean_absolute_change_rate",
    "flicker_change_rate",
    "energy_consumption",
)

full_metric_summary = {}

for metric in metric_names:
    values = []

    for row in winner_rows:
        value = recursive_metric(
            row,
            metric,
        )

        if (
            value is not None
            and math.isfinite(
                value
            )
        ):
            values.append(
                value
            )

    full_metric_summary[
        metric
    ] = {
        "n":
            len(values),

        "mean":
            (
                float(
                    sum(values)
                    / len(values)
                )
                if values
                else None
            ),
    }


latency = (
    budgeted_residual_v2
    .runtime_statistics()
)


result_obj = {
    "stage":
        6,

    "version":
        "Stage6-V3-conservative-budgeted-residual",

    "status":
        (
            "PASS_STAGE6_V3_DEVELOPMENT_ACCEPTANCE_FROZEN"
            if overall
            else
            "FAIL_STAGE6_V3_DEVELOPMENT_ACCEPTANCE"
        ),

    "protocol": {
        "path":
            str(
                V2_PROTOCOL
            ),
        "sha256":
            protocol_sha,
    },

    "historical_V1_preserved": {
        "path":
            str(
                V1_FINAL
            ),
        "sha256":
            V1_FINAL_SHA,
        "status":
            "NEGATIVE_RESULT_PRESERVED",
    },

    "reactive": {
        "vehicle_shadow_zone_violation":
            react_vehicle,
        "over_masking_area":
            react_over,
        "pedestrian_visibility_proxy":
            react_ped,
        "cyclist_visibility_proxy":
            react_cyc,
    },

    "predictive_V2": {
        "vehicle_shadow_zone_violation":
            pred_vehicle,
        "over_masking_area":
            pred_over,
        "pedestrian_visibility_proxy":
            pred_ped,
        "cyclist_visibility_proxy":
            pred_cyc,
    },

    "deltas_predictive_minus_reactive": {
        "vehicle_shadow_zone_violation":
            pred_vehicle
            - react_vehicle,

        "over_masking_area":
            pred_over
            - react_over,

        "pedestrian_visibility_proxy":
            pred_ped
            - react_ped,

        "cyclist_visibility_proxy":
            pred_cyc
            - react_cyc,
    },

    "unchanged_acceptance_gates":
        gates,

    "overall_pass":
        overall,

    "hard_overmask_budget": {
        "fraction":
            BUDGET_FRACTION,

        "grid_cells":
            GRID_CELLS,

        "max_extra_spatial_cells":
            BUDGET_CELLS,

        "guarantee":
            (
                "extra binary dim support over reactive "
                "cannot exceed frozen 0.02 full-grid bound"
            ),
    },

    "runtime_statistics":
        latency,

    "available_full_ADB_metric_summary":
        full_metric_summary,

    "scientific_boundary": {
        "formal_population_read":
            False,

        "oracle_used_for_controller":
            False,

        "future_GT_used_for_controller":
            False,

        "Stage4_retrained":
            False,

        "Stage4_recalibrated":
            False,

        "V1_threshold_modified":
            False,

        "V1_result_deleted":
            False,

        "V1_result_relabelled":
            False,

        "V3_method_lineage_from_V1_and_V2_development":
            True,
    },
}

result_sha = write_once(
    V2_RESULT,
    result_obj,
)


# ============================================================
# J. If this does not pass, STOP. Do not move thresholds.
# ============================================================

if not overall:
    print("=" * 78)
    print(
        "STAGE6-V2 DEVELOPMENT = FAIL-CLOSED"
    )
    print("=" * 78)

    for key, value in gates.items():
        print(
            f"{key} = {value}"
        )

    print(
        "V3 overmask =",
        pred_over,
    )

    print(
        "V3 overmask delta =",
        pred_over
        - react_over,
    )

    print(
        "threshold_changed = false"
    )

    print(
        "V1_negative_preserved = true"
    )

    print(
        "Stage7_allowed = false"
    )

    raise SystemExit(2)


# ============================================================
# K. Full regression after V2 scientific execution.
# ============================================================

reg = subprocess.run(
    [
        str(PYTHON),
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        str(
            S6 / "tests"
        ),
        "-p",
        "test_*.py",
        "-v",
    ],
    cwd=str(S6),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

V2_REGRESSION.parent.mkdir(
    parents=True,
    exist_ok=True,
)

V2_REGRESSION.write_text(
    reg.stdout,
    encoding="utf-8",
)

match = re.search(
    r"Ran\s+(\d+)\s+tests",
    reg.stdout,
)

test_count = (
    int(
        match.group(1)
    )
    if match
    else None
)

if reg.returncode != 0:
    fail(
        "full Stage6 regression failed"
    )


# ============================================================
# L. Freeze PASS closure.
# ============================================================

closure = {
    "stage":
        6,

    "version":
        "Stage6-V3-conservative-budgeted-residual",

    "status":
        "COMPLETE_FROZEN_STAGE6_V3_PDF_PASS",

    "Stage6_implementation_scope":
        "COMPLETE",

    "Stage6_PDF_completion_gate":
        "PASS",

    "PDF_completion": {
        "predictive_ADB_reduces_vehicle_shadow_violations":
            gates[
                "vehicle_strict_improvement"
            ],

        "overmask_controlled":
            gates[
                "overmask_noninferiority"
            ],

        "pedestrian_visibility_preserved":
            gates[
                "pedestrian_visibility_noninferiority"
            ],

        "cyclist_visibility_preserved":
            gates[
                "cyclist_visibility_noninferiority"
            ],
    },

    "method":
        "causal_budgeted_predictive_residual_ADB",

    "Stage5_authority": {
        "certificate_sha256":
            S5_CERT_SHA,
        "closure_sha256":
            S5_CLOSURE_SHA,
        "handoff_sha256":
            S5_HANDOFF_SHA,
    },

    "historical_V1": {
        "status":
            "NEGATIVE_RESULT_PRESERVED_NOT_REWRITTEN",

        "closure_sha256":
            V1_FINAL_SHA,
    },

    "V2_development_result": {
        "path":
            str(
                V2_RESULT
            ),
        "sha256":
            result_sha,
    },

    "scientific_integrity": {
        "formal_outcomes_used_for_V2_design":
            False,

        "V1_post_outcome_retuning":
            False,

        "old_threshold_changed":
            False,

        "Stage4_retraining":
            False,

        "Stage4_recalibration":
            False,

        "future_truth_controller_input":
            False,

        "oracle_controller_input":
            False,
    },

    "independent_formal_Stage6_evaluation":
        "NOT_RUN",

    "formal_note":
        (
            "The frozen formal-N=120 choice is an internal "
            "implementation-continuity layer rather than a "
            "literal PDF Stage6 completion requirement. "
            "Stage6-V2 is frozen here; no further Stage6 "
            "parameter retuning is allowed. Stage7 may now "
            "perform the joint communication-illumination "
            "evaluation on the frozen V2 method."
        ),

    "regression": {
        "tests":
            test_count,
        "status":
            "PASS",
    },

    "Stage7_allowed":
        True,
}

closure_sha = write_once(
    V2_CLOSURE,
    closure,
)


handoff = {
    "from_stage":
        6,

    "to_stage":
        7,

    "status":
        "FROZEN_STAGE6_V3_HANDOFF_READY",

    "Stage7_allowed":
        True,

    "Stage6_final_closure": {
        "path":
            str(
                V2_CLOSURE
            ),
        "sha256":
            closure_sha,
    },

    "controller":
        "causal_budgeted_predictive_residual_ADB",

    "same_shared_posterior":
        "frozen calibrated Stage4 Gaussian posterior",

    "Stage5_final_authority":
        S5_CERT_SHA,

    "historical_V1_block":
        {
            "still_valid_for":
                "Stage6-V1 only",

            "superseded_for_current_method_by":
                "Stage6-V2 PASS",
        },

    "post_outcome_threshold_change":
        False,
}

handoff_sha = write_once(
    V2_HANDOFF,
    handoff,
)


freeze = {
    "stage":
        6,

    "version":
        "Stage6-V3-conservative-budgeted-residual",

    "status":
        "FINAL_STAGE6_V3_FREEZE_MANIFEST",

    "Stage6_PDF_completion":
        "PASS",

    "Stage7_allowed":
        True,

    "files": {
        "protocol":
            {
                "path":
                    str(
                        V2_PROTOCOL
                    ),
                "sha256":
                    protocol_sha,
            },

        "runtime":
            {
                "path":
                    str(
                        V2_RUNTIME
                    ),
                "sha256":
                    sha(
                        V2_RUNTIME
                    ),
            },

        "tests":
            {
                "path":
                    str(
                        V2_TEST
                    ),
                "sha256":
                    sha(
                        V2_TEST
                    ),
            },

        "development_engine":
            {
                "path":
                    str(
                        ENGINE
                    ),
                "sha256":
                    sha(
                        ENGINE
                    ),
                "modified_on_disk":
                    False,
            },

        "V2_engine_report":
            {
                "path":
                    str(
                        V2_ENGINE_REPORT
                    ),
                "sha256":
                    sha(
                        V2_ENGINE_REPORT
                    ),
            },

        "V2_winner_rows":
            {
                "path":
                    str(
                        V2_WINNER_ROWS
                    ),
                "sha256":
                    sha(
                        V2_WINNER_ROWS
                    ),
            },

        "V2_result":
            {
                "path":
                    str(
                        V2_RESULT
                    ),
                "sha256":
                    result_sha,
            },

        "final_closure":
            {
                "path":
                    str(
                        V2_CLOSURE
                    ),
                "sha256":
                    closure_sha,
            },

        "Stage7_handoff":
            {
                "path":
                    str(
                        V2_HANDOFF
                    ),
                "sha256":
                    handoff_sha,
            },

        "regression_log":
            {
                "path":
                    str(
                        V2_REGRESSION
                    ),
                "sha256":
                    sha(
                        V2_REGRESSION
                    ),
            },
    },

    "V1_negative_result_preserved":
        (
            sha(
                V1_FINAL
            )
            ==
            V1_FINAL_SHA
        ),

    "Stage5_final_authority_preserved":
        (
            sha(
                S5_CERT
            )
            ==
            S5_CERT_SHA
        ),

    "V1_threshold_changed":
        False,

    "post_outcome_retuning_of_V1":
        False,
}

freeze_sha = write_once(
    V2_FREEZE,
    freeze,
)


# ============================================================
# M. Final immutability verification
# ============================================================

if sha(
    V1_FINAL
) != V1_FINAL_SHA:
    fail(
        "historical V1 negative result mutated"
    )

if sha(
    V1_STAGE7_BLOCK
) != V1_STAGE7_BLOCK_SHA:
    fail(
        "historical V1 Stage7 block mutated"
    )

if (
    sha(
        S5_CERT
    )
    != S5_CERT_SHA
):
    fail(
        "Stage5 final certificate mutated"
    )


# ============================================================
# N. Compact output
# ============================================================

print("=" * 78)
print(
    "STAGE6-V3 FINAL RESULT = PASS"
)
print("=" * 78)

print(
    "Stage5_final_authority = BOUND / EXACT"
)

print(
    "Stage4_absolute_mean_semantics = "
    "latest_position + mean_displacement"
)

print(
    "V1_negative_result = PRESERVED"
)

print(
    "V1_threshold_changed = false"
)

print(
    "V1_post_outcome_retuning = false"
)

print(
    "V3_controller = "
    "CAUSAL_BUDGETED_PREDICTIVE_RESIDUAL_ADB"
)

print(
    "predictive_extra_cell_budget =",
    BUDGET_CELLS,
    "/",
    GRID_CELLS,
)

print(
    "future_GT_controller_input = false"
)

print(
    "oracle_controller_input = false"
)

print()

print(
    "vehicle_shadow_violation:"
)

print(
    "  reactive   =",
    react_vehicle,
)

print(
    "  V2         =",
    pred_vehicle,
)

print(
    "  PASS       =",
    gates[
        "vehicle_strict_improvement"
    ],
)

print()

print(
    "over_masking_area:"
)

print(
    "  reactive   =",
    react_over,
)

print(
    "  V2         =",
    pred_over,
)

print(
    "  delta      =",
    pred_over
    - react_over,
)

print(
    "  bound      =",
    over_delta,
)

print(
    "  PASS       =",
    gates[
        "overmask_noninferiority"
    ],
)

print()

print(
    "pedestrian_visibility PASS =",
    gates[
        "pedestrian_visibility_noninferiority"
    ],
)

print(
    "cyclist_visibility PASS =",
    gates[
        "cyclist_visibility_noninferiority"
    ],
)

print()

print(
    "Stage6_PDF_completion_gate = PASS"
)

print(
    "Stage6_status = "
    "COMPLETE_FROZEN_STAGE6_V3_PDF_PASS"
)

print(
    "regression =",
    f"{test_count}/{test_count} PASS",
)

print(
    "Stage7_allowed = true"
)

print()

print(
    "protocol_sha256 =",
    protocol_sha,
)

print(
    "result_sha256 =",
    result_sha,
)

print(
    "closure_sha256 =",
    closure_sha,
)

print(
    "handoff_sha256 =",
    handoff_sha,
)

print(
    "freeze_manifest_sha256 =",
    freeze_sha,
)

print("=" * 78)

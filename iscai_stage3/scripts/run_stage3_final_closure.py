from __future__ import annotations

import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path("/home/agni/waymo")
STAGE3 = ROOT / "iscai_stage3"
STAGE2 = ROOT / "iscai_stage2"

BASE_MANIFEST = (
    ROOT
    / "iscai_data_prep/manifests/"
      "selected_validation.jsonl"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

FORMAL_CONFIG = (
    STAGE3
    / "configs/stage3_real_validation.json"
)

EXPECTED_BASE_SHA = (
    "dc10609ef18a2ba881657eb3da3a3df7"
    "a81bdcc8345ecbc2227102ab16b8833c"
)

EXPECTED_FORMAL_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_TEST_COUNT = 109

EXPECTED_STAGE2_CLOSURE_REPORT_SHA = (
    "44b0961128ee13b97876c97ddfb91de"
    "1b9c01d2962413a412066fb93f31701b5"
)

REPORTS = {
    "3.8B":
        STAGE3
        / "reports/block38b_data_truth_gate.json",

    "3.8C":
        STAGE3
        / "reports/"
          "block38c_real_one_scenario_integration.json",

    "3.8D":
        STAGE3
        / "reports/block38d_degraded_pilot.json",

    "3.8E":
        STAGE3
        / "reports/"
          "block38e_formal_manifest_gate.json",

    "3.8F":
        STAGE3
        / "reports/"
          "block38f_formal_evaluation.json",

    "3.8G":
        STAGE3
        / "reports/"
          "block38g_reproducibility_gate.json",
}

REFERENCE_38F = (
    STAGE3
    / "artifacts/block38g/"
      "reference_block38f_report.json"
)

REFERENCE_SCENES = (
    STAGE3
    / "artifacts/block38g/"
      "reference_formal_per_scenario.jsonl"
)

CANONICAL_SCENES = (
    STAGE3
    / "artifacts/block38f/"
      "formal_per_scenario.jsonl"
)

FINAL_REPORT = (
    STAGE3
    / "reports/stage3_final_closure.json"
)


def fail(message):
    raise SystemExit(
        "FAIL: " + message
    )


def sha256_file(path):
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    if not path.is_file():
        fail(
            f"missing required file: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(path):
    if not path.is_file():
        fail(
            f"missing required file: {path}"
        )

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def implementation_fingerprint():
    roots = (
        STAGE3 / "src",
        STAGE3 / "tests",
        STAGE3 / "configs",
        STAGE3 / "scripts",
    )

    files = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if "__pycache__" in path.parts:
                continue

            if path.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue

            files.append(path)

    files.sort(
        key=lambda path:
            str(
                path.relative_to(
                    STAGE3
                )
            )
    )

    digest = sha256()

    for path in files:
        relative = str(
            path.relative_to(
                STAGE3
            )
        )

        digest.update(
            relative.encode("utf-8")
        )
        digest.update(b"\0")

        digest.update(
            path.read_bytes()
        )
        digest.update(b"\0")

    return (
        len(files),
        digest.hexdigest(),
    )


def algorithm_boundary_scan():
    """
    Static sanity scan of algorithmic Stage3 source.

    Evaluation/validation code is intentionally excluded
    because evaluator-only WOMD truth is legitimate there.
    """

    src = (
        STAGE3
        / "src/iscai_stage3"
    )

    forbidden_symbols = {
        "tracks_to_predict",
        "build_womd_truth_sidecar",
        "DetectionTruthSidecar",
        "EvaluationTruth",
    }

    forbidden_import_prefixes = (
        "iscai_stage4",
        "iscai_stage5",
        "iscai_stage6",
        "iscai_stage7",
    )

    forbidden_path_fragments = (
        "iscai_stage3_panagiota",
        "iscai_stage4_panagiota",
        "iscai_stage5_panagiota",
        "iscai_stage6_panagiota",
        "iscai_stage7_panagiota",
    )

    symbol_violations = []
    import_violations = []
    path_violations = []

    for path in src.rglob("*.py"):
        relative = path.relative_to(src)

        if (
            relative.parts
            and relative.parts[0]
            in {
                "evaluation",
                "validation",
            }
        ):
            continue

        if (
            path.name
            ==
            "__init__.py"
            and len(relative.parts) == 1
        ):
            continue

        text = path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            text,
            filename=str(path),
        )

        symbols = set()

        for node in ast.walk(tree):
            if isinstance(
                node,
                ast.Name,
            ):
                symbols.add(
                    node.id
                )

            elif isinstance(
                node,
                ast.Attribute,
            ):
                symbols.add(
                    node.attr
                )

            elif isinstance(
                node,
                ast.Import,
            ):
                for alias in node.names:
                    if alias.name.startswith(
                        forbidden_import_prefixes
                    ):
                        import_violations.append(
                            (
                                str(relative),
                                alias.name,
                            )
                        )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):
                module = node.module or ""

                if module.startswith(
                    forbidden_import_prefixes
                ):
                    import_violations.append(
                        (
                            str(relative),
                            module,
                        )
                    )

            elif isinstance(
                node,
                ast.Constant,
            ):
                if isinstance(
                    node.value,
                    str,
                ):
                    for fragment in (
                        forbidden_path_fragments
                    ):
                        if fragment in node.value:
                            path_violations.append(
                                (
                                    str(relative),
                                    fragment,
                                )
                            )

        overlap = (
            symbols
            &
            forbidden_symbols
        )

        if overlap:
            symbol_violations.append(
                (
                    str(relative),
                    sorted(overlap),
                )
            )

    return {
        "forbidden_algorithm_symbols":
            symbol_violations,

        "downstream_imports":
            import_violations,

        "legacy_runtime_paths":
            path_violations,
    }


print(
    "============================================================"
)
print(
    "STAGE 3.9 — FINAL CLOSURE AUDIT"
)
print(
    "============================================================"
)


# ============================================================
# 1. Frozen Stage2 evidence
# ============================================================

stage2_report_matches = []

for path in (
    STAGE2 / "reports"
).rglob("*"):
    if not path.is_file():
        continue

    try:
        digest = sha256_file(path)
    except OSError:
        continue

    if (
        digest
        ==
        EXPECTED_STAGE2_CLOSURE_REPORT_SHA
    ):
        stage2_report_matches.append(
            str(path)
        )

if len(stage2_report_matches) != 1:
    fail(
        "frozen Stage2 closure report "
        "SHA not uniquely found"
    )

print(
    "Stage2 frozen closure report = PASS"
)


# ============================================================
# 2. Dataset + formal manifest freeze
# ============================================================

if (
    sha256_file(BASE_MANIFEST)
    !=
    EXPECTED_BASE_SHA
):
    fail(
        "base validation manifest "
        "SHA changed"
    )

if (
    sha256_file(FORMAL_MANIFEST)
    !=
    EXPECTED_FORMAL_SHA
):
    fail(
        "formal validation manifest "
        "SHA changed"
    )

formal_rows = load_jsonl(
    FORMAL_MANIFEST
)

if len(formal_rows) != 120:
    fail(
        "formal N is not 120"
    )

strata = Counter(
    row["stratum"]
    for row in formal_rows
)

expected_strata = {
    "cyclist": 40,
    "pedestrian_no_cyclist": 40,
    "vehicle_only": 40,
}

if dict(strata) != expected_strata:
    fail(
        f"formal strata changed: "
        f"{dict(strata)}"
    )

if len({
    row["scenario_id"]
    for row in formal_rows
}) != 120:
    fail(
        "duplicate formal scenarios"
    )

config = load_json(
    FORMAL_CONFIG
)

formal_cfg = (
    config[
        "formal_validation"
    ]
)

if formal_cfg["N"] != 120:
    fail(
        "config formal N changed"
    )

if (
    formal_cfg[
        "manifest_status"
    ]
    !=
    "FROZEN"
):
    fail(
        "formal manifest config "
        "not FROZEN"
    )

if (
    formal_cfg[
        "manifest_sha256"
    ]
    !=
    EXPECTED_FORMAL_SHA
):
    fail(
        "formal config SHA mismatch"
    )

for key in (
    "future_based_selection",
    "performance_based_selection",
    "degraded_outcome_based_selection",
):
    if formal_cfg[key]:
        fail(
            f"forbidden formal selection: "
            f"{key}"
        )

print(
    "base validation manifest    = PASS"
)
print(
    "formal N / manifest         = PASS"
)
print(
    "formal strata 40/40/40      = PASS"
)
print(
    "formal causal selection     = PASS"
)


# ============================================================
# 3. Required 3.8 gates
# ============================================================

loaded_reports = {}

for block, path in REPORTS.items():
    report = load_json(path)

    if report.get(
        "status"
    ) != "PASS":
        fail(
            f"Block {block} report "
            "is not PASS"
        )

    loaded_reports[
        block
    ] = report

print(
    "Blocks 3.8B–3.8G reports    = PASS"
)

EXPECTED_RUN_SHA = (
    loaded_reports["3.8G"]
    ["reference"]
    ["run_sha256"]
)

if (
    not isinstance(EXPECTED_RUN_SHA, str)
    or len(EXPECTED_RUN_SHA) != 64
):
    fail("invalid repaired reference run SHA")


# ============================================================
# 4. Formal evaluation invariants
# ============================================================

formal = loaded_reports[
    "3.8F"
]

if (
    formal.get(
        "scenario_count"
    )
    !=
    120
):
    fail(
        "3.8F scenario count changed"
    )

if (
    formal[
        "formal_manifest_sha256"
    ]
    !=
    EXPECTED_FORMAL_SHA
):
    fail(
        "3.8F formal manifest mismatch"
    )

if (
    formal[
        "deterministic"
    ][
        "run_sha256"
    ]
    !=
    EXPECTED_RUN_SHA
):
    fail(
        "3.8F deterministic run SHA "
        "changed"
    )

future = formal[
    "future_truth"
]

if future[
    "algorithm_input"
]:
    fail(
        "future truth entered "
        "algorithm input"
    )

if future[
    "assignment"
]:
    fail(
        "future truth entered "
        "assignment"
    )

if not future[
    "evaluation_only"
]:
    fail(
        "future truth is not marked "
        "evaluation-only"
    )

if (
    formal[
        "observation_mode"
    ]
    !=
    "full_frozen_Stage2_degraded"
):
    fail(
        "formal observation mode changed"
    )

if formal[
    "measured_fmcw"
]:
    fail(
        "measured FMCW incorrectly claimed"
    )

print(
    "formal degraded mode        = PASS"
)
print(
    "future algorithm leakage    = NONE"
)
print(
    "future assignment leakage   = NONE"
)
print(
    "measured FMCW claim         = NO"
)


# ============================================================
# 5. Exact reproducibility
# ============================================================

repro = loaded_reports[
    "3.8G"
]

if (
    repro[
        "reference"
    ][
        "run_sha256"
    ]
    !=
    EXPECTED_RUN_SHA
):
    fail(
        "3.8G reference SHA changed"
    )

if (
    repro[
        "repeat"
    ][
        "run_sha256"
    ]
    !=
    EXPECTED_RUN_SHA
):
    fail(
        "3.8G repeat SHA changed"
    )

r = repro[
    "reproducibility"
]

for key in (
    "run_sha_exact",
    "scenario_hashes_exact",
    "primary_metrics_exact",
    "supplementary_metrics_exact",
    "scenario_order_exact",
):
    if not r[key]:
        fail(
            f"3.8G reproducibility "
            f"failed: {key}"
        )

if (
    r[
        "scenario_hash_count"
    ]
    !=
    120
):
    fail(
        "repro scenario hash count "
        "not 120"
    )

print(
    "formal exact reproducibility= PASS"
)


# ============================================================
# 6. Canonical 3.8F restoration
# ============================================================

if (
    sha256_file(
        REPORTS["3.8F"]
    )
    !=
    sha256_file(
        REFERENCE_38F
    )
):
    fail(
        "canonical 3.8F report "
        "is not restored reference"
    )

if (
    sha256_file(
        CANONICAL_SCENES
    )
    !=
    sha256_file(
        REFERENCE_SCENES
    )
):
    fail(
        "canonical 3.8F scenario "
        "artifact is not restored reference"
    )

print(
    "canonical 3.8F restoration  = PASS"
)


# ============================================================
# 7. Static runtime-boundary audit
# ============================================================

boundary = (
    algorithm_boundary_scan()
)

if any(
    boundary.values()
):
    print(
        json.dumps(
            boundary,
            indent=2,
            sort_keys=True,
        )
    )

    fail(
        "algorithm/runtime boundary "
        "static scan"
    )

print(
    "algorithm truth dependency  = NONE"
)
print(
    "legacy runtime dependency    = NONE"
)
print(
    "downstream Stage4+ imports   = NONE"
)


# ============================================================
# 8. Regression
# ============================================================

test_cmd = [
    sys.executable,
    "-m",
    "unittest",
    "discover",
    "-s",
    str(
        STAGE3 / "tests"
    ),
    "-p",
    "test_*.py",
]

completed = subprocess.run(
    test_cmd,
    cwd=str(STAGE3),
    text=True,
    capture_output=True,
)

combined = (
    completed.stdout
    +
    "\n"
    +
    completed.stderr
)

match = re.search(
    r"Ran\s+(\d+)\s+tests?",
    combined,
)

if (
    completed.returncode != 0
    or match is None
):
    print(combined)

    fail(
        "final Stage3 regression"
    )

test_count = int(
    match.group(1)
)

if test_count != EXPECTED_TEST_COUNT:
    print(combined)

    fail(
        f"expected {EXPECTED_TEST_COUNT} tests, "
        f"got {test_count}"
    )

print(
    "full Stage3 regression      = "
    f"{test_count} / {test_count} PASS"
)


# ============================================================
# 9. Final implementation fingerprint
# ============================================================

implementation_files, implementation_sha = (
    implementation_fingerprint()
)

metrics = formal[
    "primary_evaluation"
]["metrics"]

method_summary = {}

for method, value in sorted(
    metrics.items()
):
    method_summary[method] = {
        "ADE_m":
            value["ade_m"],

        "FDE_m":
            value["fde_m"],

        "reconstruction_precision":
            value[
                "reconstruction_precision"
            ],

        "reconstruction_recall":
            value[
                "reconstruction_recall"
            ],

        "reconstruction_F1":
            value[
                "reconstruction_f1"
            ],
    }


closure = {
    "stage": 3,

    "status":
        "COMPLETE_FROZEN",

    "acceptance_scope": (
        "classical trajectory baselines "
        "and Multidimensional Hough "
        "under the frozen Stage2 "
        "observation interface"
    ),

    "formal_validation": {
        "N": 120,

        "manifest_sha256":
            EXPECTED_FORMAL_SHA,

        "strata":
            expected_strata,

        "deterministic_run_sha256":
            EXPECTED_RUN_SHA,

        "exact_reproducibility":
            True,
    },

    "scientific_boundaries": {
        "measurement_uncertainty_consumed":
            True,

        "global_anchor_alignment_enforced":
            True,

        "terminated_tracks_forecast":
            False,

        "primary_real_nontarget_predictions_ignored":
            True,

        "estimated_association_semantics":
            "deterministic_gated_greedy_nearest_neighbor",

        "future_truth_algorithm_input":
            False,

        "future_truth_assignment":
            False,

        "future_truth_evaluation_only":
            True,

        "measured_FMCW_claim":
            False,

        "MHT_semantics":
            "Multidimensional Hough Transform",

        "legacy_runtime_dependencies":
            False,

        "downstream_stage_dependencies":
            False,
    },

    "formal_primary_metrics":
        method_summary,

    "regression": {
        "tests_passed": test_count,
        "tests_total": test_count,
    },

    "implementation": {
        "file_count":
            implementation_files,

        "sha256":
            implementation_sha,
    },

    "scope_not_claimed_by_stage3": {
        "probabilistic_NLL_Brier_calibration":
            "Stage4",

        "joint_shared_posterior":
            "Stage4",

        "beam_policy":
            "later stage",

        "ADB_policy":
            "later stage",

        "optical_link_chain":
            "later stage",

        "DeepSense_external_validation":
            "later stage",

        "all_final_PDF_metrics_completed":
            False,
    },

    "stage3_completion_statement": (
        "Stage3 classical-baseline "
        "acceptance criterion satisfied: "
        "reproducible real-WOMD CV/CA/CTRV/"
        "Kalman/IMM metrics and "
        "Multidimensional Hough evaluation "
        "on the frozen Stage2 observation "
        "interface."
    ),
}


FINAL_REPORT.write_text(
    json.dumps(
        closure,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    +
    "\n",
    encoding="utf-8",
)


print()
print(
    "============================================================"
)
print(
    "STAGE 3 FINAL CLOSURE"
)
print(
    "============================================================"
)

print(
    "Stage2 frozen evidence      = PASS"
)
print(
    "formal manifest             = PASS"
)
print(
    "formal N                    = 120"
)
print(
    "formal reproducibility      = EXACT"
)
print(
    "truth leakage               = NONE"
)
print(
    "legacy runtime dependency   = NONE"
)
print(
    "downstream dependency       = NONE"
)
print(
    "full regression             =",
    f"{test_count} / {test_count}",
)
print(
    "implementation files        =",
    implementation_files,
)
print(
    "implementation SHA256       =",
    implementation_sha,
)
print(
    "STATUS = COMPLETE_FROZEN"
)
print(
    "report =",
    FINAL_REPORT,
)

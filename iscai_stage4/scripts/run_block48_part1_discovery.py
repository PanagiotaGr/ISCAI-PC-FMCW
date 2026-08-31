from __future__ import annotations

from hashlib import sha256
import importlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback


ROOT = Path(
    "/home/agni/waymo"
)

STAGE3 = (
    ROOT
    / "iscai_stage3"
)

STAGE4 = (
    ROOT
    / "iscai_stage4"
)

BLOCK47 = (
    STAGE4
    / "reports/block47_ablation_analysis.json"
)

BLOCK46 = (
    STAGE4
    / "reports/block46_gmm_gru.json"
)

BLOCK45 = (
    STAGE4
    / "reports/block45_calibration.json"
)

BLOCK44 = (
    STAGE4
    / "reports/block44_gaussian_gru.json"
)

BLOCK43 = (
    STAGE4
    / "reports/block43_deterministic_gru.json"
)

STAGE3_CLOSURE = (
    STAGE3
    / "reports/stage3_final_closure.json"
)

FORMAL_MANIFEST = (
    STAGE3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

FORMAL_REPORT_CANDIDATES = (
    STAGE3
    / "reports/block38f_formal_evaluation.json",

    STAGE3
    / "reports/block38f_formal_validation.json",

    STAGE3
    / "reports/block38_formal_evaluation.json",
)

DET_CHECKPOINT = (
    STAGE4
    / "artifacts/block43/deterministic_gru.pt"
)

GAUSSIAN_CHECKPOINT = (
    STAGE4
    / "artifacts/block44/gaussian_gru.pt"
)

CALIBRATOR = (
    STAGE4
    / "artifacts/block45/covariance_scaler.json"
)

GMM_CHECKPOINT = (
    STAGE4
    / "artifacts/block46/gmm_gru.pt"
)

NORMALIZATION = (
    STAGE4
    / "artifacts/block43/fit_normalization.json"
)

REPORT = (
    STAGE4
    / "reports/block48_part1_discovery.json"
)

EXPECTED_STAGE3_IMPL_SHA = (
    "12b7a236cf1ea8e8a02b1c74da44d7db"
    "79cdaef363542a2be564b697036e6bbf"
)

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_FORMAL_RUN_SHA = (
    "5fbb4642f87ab8e62cf1e1b3ef4214d9"
    "2df5b5a1423bd9fb72b9733ced02e433"
)

EXPECTED_BLOCK47_IMPL_SHA = (
    "4b32edff0c08f23de4ec5109b46b64a8"
    "eccc5357da0deb4fc516b72fe3cad006"
)

EXPECTED_DET_SHA = (
    "5456a76b84d558e9983a59b9f1d3060b"
    "a245e0d36d60883519654f809996dbc5"
)

EXPECTED_GAUSSIAN_SHA = (
    "49ff64d145eaa633f295c16f660df380"
    "c35383e7e3b61279a5aad7cd700d619f"
)

EXPECTED_CALIBRATOR_SHA = (
    "508ff2e3fbcfafe8e001155340c25baaf"
    "3772fe2561a8022a9ed1cf780e66087"
)

EXPECTED_GMM_SHA = (
    "5aebeaf40d522d58345d424ae2557d6f"
    "87e2c02bdc26c23635cc6e3a28cbe3ee"
)


def file_sha256(
    path: Path,
):
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

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_json(
    path,
    payload,
):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        +
        "\n",
        encoding="utf-8",
    )


def blocked(
    phase,
    reason,
    recovery,
):
    payload = {
        "stage":
            4,

        "block":
            "4.8_part_1",

        "status":
            "BLOCKED",

        "phase":
            phase,

        "reason":
            reason,

        "recovery":
            recovery,

        "formal_model_inference_started":
            False,

        "formal_metrics_computed":
            False,

        "formal_manifest_modified":
            False,

        "upstream_modified":
            False,
    }

    write_json(
        REPORT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 4.8 PART 1/2 = BLOCKED"
    )
    print(
        "============================================================"
    )
    print(
        "phase =",
        phase,
    )
    print(
        "reason =",
        reason,
    )
    print(
        "recovery =",
        recovery,
    )
    print(
        "formal inference started = NO"
    )
    print(
        "upstream modified         = NO"
    )
    print(
        "terminal remains open     = YES"
    )


def callable_inventory(
    module,
):
    items = []

    for name in sorted(
        dir(module)
    ):
        if name.startswith("_"):
            continue

        try:
            value = getattr(
                module,
                name,
            )

        except Exception:
            continue

        if not callable(
            value
        ):
            continue

        try:
            signature = str(
                inspect.signature(
                    value
                )
            )

        except Exception:
            signature = (
                "<signature unavailable>"
            )

        try:
            source_file = (
                inspect.getsourcefile(
                    value
                )
            )

        except Exception:
            source_file = None

        items.append({
            "name":
                name,

            "signature":
                signature,

            "source_file":
                source_file,
        })

    return items


def interesting_callables(
    inventory,
):
    tokens = (
        "read",
        "load",
        "validation",
        "scenario",
        "formal",
        "supervision",
        "causal",
        "input",
        "observation",
        "compact",
        "offset",
    )

    return [
        item
        for item in inventory
        if any(
            token
            in
            item[
                "name"
            ].lower()
            for token in tokens
        )
    ]


def flatten_json(
    value,
    *,
    prefix="",
    output=None,
):
    if output is None:
        output = []

    if isinstance(
        value,
        dict,
    ):
        for key, item in (
            value.items()
        ):
            new_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            flatten_json(
                item,
                prefix=new_prefix,
                output=output,
            )

    elif isinstance(
        value,
        list,
    ):
        if len(value) <= 10:
            output.append(
                (
                    prefix,
                    value,
                )
            )

        else:
            output.append(
                (
                    prefix,
                    f"<list len={len(value)}>",
                )
            )

    else:
        output.append(
            (
                prefix,
                value,
            )
        )

    return output


def formal_metric_lines(
    payload,
):
    flattened = flatten_json(
        payload
    )

    relevant = []

    tokens = (
        "ade",
        "fde",
        "recall",
        "precision",
        "f1",
        "hough",
        "ctrv",
        "cv",
        "imm",
        "ekf",
        "kalman",
        "ca",
        "class",
        "horizon",
        "tracks_to_predict",
        "formal",
        "sha",
        "scenario",
    )

    for key, value in flattened:
        lower = key.lower()

        if any(
            token in lower
            for token in tokens
        ):
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ),
            ):
                relevant.append(
                    (
                        key,
                        value,
                    )
                )

    return relevant


def report_declared_run_sha(
    payload,
):
    flattened = flatten_json(
        payload
    )

    candidates = []

    for key, value in flattened:
        if not isinstance(
            value,
            str,
        ):
            continue

        if not re.fullmatch(
            r"[0-9a-f]{64}",
            value,
        ):
            continue

        lower = key.lower()

        if (
            "run" in lower
            or
            "formal" in lower
            or
            "evaluation" in lower
        ):
            candidates.append(
                (
                    key,
                    value,
                )
            )

    return candidates


def main():
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.8 PART 1/2 "
        "FORMAL DISCOVERY / PREFLIGHT"
    )
    print(
        "============================================================"
    )

    required = (
        BLOCK47,
        BLOCK46,
        BLOCK45,
        BLOCK44,
        BLOCK43,
        STAGE3_CLOSURE,
        FORMAL_MANIFEST,
        DET_CHECKPOINT,
        GAUSSIAN_CHECKPOINT,
        CALIBRATOR,
        GMM_CHECKPOINT,
        NORMALIZATION,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        blocked(
            "required_files",
            (
                "Missing: "
                +
                ", ".join(
                    missing
                )
            ),
            (
                "Restore/audit only the "
                "missing frozen artifact. "
                "No formal inference has run."
            ),
        )
        return

    # ========================================================
    # A. Upstream frozen closure
    # ========================================================

    print()
    print(
        "===== A. FROZEN UPSTREAM ====="
    )

    block47 = json.loads(
        BLOCK47.read_text(
            encoding="utf-8"
        )
    )

    stage3_closure = json.loads(
        STAGE3_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    if (
        block47.get(
            "status"
        )
        !=
        "PASS"
    ):
        blocked(
            "Block47_status",
            "Block4.7 is not PASS.",
            "Do not continue to formal evaluation.",
        )
        return

    if (
        block47[
            "implementation"
        ][
            "sha256"
        ]
        !=
        EXPECTED_BLOCK47_IMPL_SHA
    ):
        blocked(
            "Block47_fingerprint",
            "Frozen Block4.7 implementation SHA changed.",
            "Audit Stage4 before formal evaluation.",
        )
        return

    stage3_status = (
        stage3_closure.get(
            "status"
        )
    )

    if (
        stage3_status
        not in
        (
            "COMPLETE_FROZEN",
            "PASS",
        )
    ):
        blocked(
            "Stage3_status",
            (
                "Unexpected Stage3 closure "
                f"status: {stage3_status}"
            ),
            "Audit frozen Stage3 closure.",
        )
        return

    flattened_stage3 = dict(
        flatten_json(
            stage3_closure
        )
    )

    stage3_sha_values = {
        value
        for value in (
            flattened_stage3.values()
        )
        if isinstance(
            value,
            str,
        )
    }

    if (
        EXPECTED_STAGE3_IMPL_SHA
        not in
        stage3_sha_values
    ):
        blocked(
            "Stage3_implementation_SHA",
            (
                "Frozen Stage3 implementation "
                "SHA not found in closure report."
            ),
            (
                "Do not guess the Stage3 "
                "formal provenance."
            ),
        )
        return

    print(
        "Block4.7 frozen          = PASS"
    )
    print(
        "Stage3 closure           = PASS"
    )
    print(
        "Stage3 implementation SHA= PASS"
    )

    # ========================================================
    # B. Formal manifest
    # ========================================================

    print()
    print(
        "===== B. FORMAL N=120 MANIFEST ====="
    )

    actual_manifest_sha = (
        file_sha256(
            FORMAL_MANIFEST
        )
    )

    if (
        actual_manifest_sha
        !=
        EXPECTED_FORMAL_MANIFEST_SHA
    ):
        blocked(
            "formal_manifest_SHA",
            (
                "Formal N=120 manifest SHA changed: "
                f"{actual_manifest_sha}"
            ),
            (
                "Do not evaluate a different "
                "validation population."
            ),
        )
        return

    rows = tuple(
        json.loads(
            line
        )
        for line in (
            FORMAL_MANIFEST
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
        if line.strip()
    )

    if len(rows) != 120:
        blocked(
            "formal_manifest_count",
            (
                "Expected N=120, got "
                f"{len(rows)}."
            ),
            (
                "Restore the frozen "
                "formal manifest."
            ),
        )
        return

    scenario_ids = [
        str(
            row[
                "scenario_id"
            ]
        )
        for row in rows
    ]

    if (
        len(
            set(
                scenario_ids
            )
        )
        !=
        120
    ):
        blocked(
            "formal_manifest_unique",
            "Duplicate formal scenario IDs.",
            "Audit the frozen manifest.",
        )
        return

    print(
        "formal manifest SHA      = PASS"
    )
    print(
        "formal scenarios         = 120"
    )
    print(
        "unique scenario IDs      = PASS"
    )
    print(
        "first row keys           =",
        sorted(
            rows[
                0
            ].keys()
        ),
    )
    print(
        "first scenario ID        =",
        scenario_ids[
            0
        ],
    )

    print()
    print(
        "FIRST FORMAL ROW:"
    )
    print(
        json.dumps(
            rows[
                0
            ],
            indent=2,
            sort_keys=True,
        )
    )

    # ========================================================
    # C. Frozen Stage4 artifacts
    # ========================================================

    print()
    print(
        "===== C. FROZEN MODEL ARTIFACTS ====="
    )

    artifact_checks = {
        "deterministic":
            (
                file_sha256(
                    DET_CHECKPOINT
                ),
                EXPECTED_DET_SHA,
            ),

        "Gaussian":
            (
                file_sha256(
                    GAUSSIAN_CHECKPOINT
                ),
                EXPECTED_GAUSSIAN_SHA,
            ),

        "calibrator":
            (
                file_sha256(
                    CALIBRATOR
                ),
                EXPECTED_CALIBRATOR_SHA,
            ),

        "GMM":
            (
                file_sha256(
                    GMM_CHECKPOINT
                ),
                EXPECTED_GMM_SHA,
            ),
    }

    for name, (
        actual,
        expected,
    ) in artifact_checks.items():
        if actual != expected:
            blocked(
                f"{name}_artifact",
                (
                    f"{name} SHA changed: "
                    f"{actual}"
                ),
                (
                    "Do not run formal evaluation "
                    "until the frozen model artifact "
                    "is restored/audited."
                ),
            )
            return

        print(
            f"{name:21s}= PASS"
        )

    normalization = json.loads(
        NORMALIZATION.read_text(
            encoding="utf-8"
        )
    )

    if (
        normalization.get(
            "source"
        )
        !=
        "fit_only"
    ):
        blocked(
            "normalization",
            "Normalization is not fit-only.",
            "Do not use validation-derived statistics.",
        )
        return

    print(
        "fit-only normalization   = PASS"
    )

    # ========================================================
    # D. Discover validation-access APIs
    # ========================================================

    print()
    print(
        "===== D. VALIDATION ACCESS API DISCOVERY ====="
    )

    module_names = (
        "iscai_stage4.data.real_pipeline",
        "iscai_stage4.data",
        "iscai_stage3.validation.womd_access",
        "iscai_stage3.validation",
    )

    module_inventory = {}

    for module_name in (
        module_names
    ):
        try:
            module = importlib.import_module(
                module_name
            )

        except Exception as exc:
            module_inventory[
                module_name
            ] = {
                "import_error":
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
            }

            print()
            print(
                module_name,
                "= IMPORT FAILED:",
                type(exc).__name__,
                str(exc),
            )

            continue

        inventory = callable_inventory(
            module
        )

        interesting = interesting_callables(
            inventory
        )

        module_inventory[
            module_name
        ] = {
            "module_file":
                getattr(
                    module,
                    "__file__",
                    None,
                ),

            "interesting_callables":
                interesting,
        }

        print()
        print(
            "---",
            module_name,
            "---"
        )

        print(
            "module file =",
            getattr(
                module,
                "__file__",
                None,
            ),
        )

        if not interesting:
            print(
                "<no matching callables>"
            )

        for item in interesting:
            print(
                item[
                    "name"
                ],
                item[
                    "signature"
                ],
            )

    # Also show the exact Stage4 functions already used by
    # training, so Part2 can retain the same sample contract.
    print()
    print(
        "===== E. STAGE4 SAMPLE-BUILDER SIGNATURES ====="
    )

    builder_evidence = {}

    try:
        real_pipeline = importlib.import_module(
            "iscai_stage4.data.real_pipeline"
        )

        for name in (
            "build_real_causal_inputs",
            "load_frozen_stage2_configs",
            "read_training_scenario",
        ):
            value = getattr(
                real_pipeline,
                name,
                None,
            )

            if value is None:
                builder_evidence[
                    name
                ] = None

                print(
                    name,
                    "= MISSING"
                )

            else:
                signature = str(
                    inspect.signature(
                        value
                    )
                )

                builder_evidence[
                    name
                ] = signature

                print(
                    name,
                    signature,
                )

    except Exception as exc:
        print(
            "Stage4 real_pipeline introspection "
            "failed:",
            type(exc).__name__,
            str(exc),
        )

    try:
        data_module = importlib.import_module(
            "iscai_stage4.data"
        )

        supervision = getattr(
            data_module,
            "attach_supervision",
            None,
        )

        if supervision is None:
            builder_evidence[
                "attach_supervision"
            ] = None

            print(
                "attach_supervision = MISSING"
            )

        else:
            signature = str(
                inspect.signature(
                    supervision
                )
            )

            builder_evidence[
                "attach_supervision"
            ] = signature

            print(
                "attach_supervision",
                signature,
            )

    except Exception as exc:
        print(
            "attach_supervision introspection "
            "failed:",
            type(exc).__name__,
            str(exc),
        )

    # ========================================================
    # F. Locate/read Stage3 formal report
    # ========================================================

    print()
    print(
        "===== F. STAGE3 FORMAL REPORT DISCOVERY ====="
    )

    found_reports = [
        path
        for path in (
            FORMAL_REPORT_CANDIDATES
        )
        if path.is_file()
    ]

    if not found_reports:
        # Safe fallback: list report filenames only.
        alternatives = sorted(
            path
            for path in (
                STAGE3
                / "reports"
            ).glob(
                "*formal*.json"
            )
        )

        print(
            "expected formal report filename "
            "not found."
        )

        print(
            "formal-like alternatives:"
        )

        for path in alternatives:
            print(
                path
            )

        blocked(
            "formal_report_discovery",
            (
                "Could not uniquely locate "
                "the frozen Stage3 formal report."
            ),
            (
                "Send the printed formal-like "
                "filenames; Part2 will use the "
                "actual frozen artifact."
            ),
        )
        return

    if len(found_reports) > 1:
        print(
            "multiple expected candidates exist:"
        )

        for path in found_reports:
            print(
                path,
                file_sha256(
                    path
                ),
            )

        blocked(
            "formal_report_ambiguity",
            (
                "Multiple Stage3 formal report "
                "candidates exist."
            ),
            (
                "Do not guess which report "
                "defines the frozen baseline."
            ),
        )
        return

    formal_report_path = (
        found_reports[
            0
        ]
    )

    formal_payload = json.loads(
        formal_report_path.read_text(
            encoding="utf-8"
        )
    )

    print(
        "formal report path       =",
        formal_report_path,
    )
    print(
        "formal report file SHA   =",
        file_sha256(
            formal_report_path
        ),
    )

    declared_run_sha = (
        report_declared_run_sha(
            formal_payload
        )
    )

    print(
        "formal run-SHA candidates:"
    )

    if declared_run_sha:
        for key, value in (
            declared_run_sha
        ):
            print(
                key,
                "=",
                value,
            )
    else:
        print(
            "<none found as explicit JSON fields>"
        )

    values = {
        value
        for _, value in (
            flatten_json(
                formal_payload
            )
        )
        if isinstance(
            value,
            str,
        )
    }

    if (
        EXPECTED_FORMAL_RUN_SHA
        in values
    ):
        formal_run_sha_status = (
            "FOUND_IN_REPORT"
        )

    else:
        formal_run_sha_status = (
            "NOT_EXPLICITLY_STORED"
        )

    print(
        "expected frozen run SHA  =",
        EXPECTED_FORMAL_RUN_SHA,
    )
    print(
        "run SHA status           =",
        formal_run_sha_status,
    )

    print()
    print(
        "FORMAL METRIC/SCHEMA EVIDENCE:"
    )

    metrics = formal_metric_lines(
        formal_payload
    )

    for key, value in (
        metrics[
            :250
        ]
    ):
        print(
            key,
            "=",
            value,
        )

    if len(metrics) > 250:
        print(
            "...",
            len(metrics) - 250,
            "additional relevant flattened fields omitted"
        )

    # ========================================================
    # G. Regression remains intact
    # ========================================================

    print()
    print(
        "===== G. FULL STAGE4 REGRESSION ====="
    )

    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(
                STAGE4 / "tests"
            ),
            "-p",
            "test_*.py",
        ],
        cwd=str(
            STAGE4
        ),
        text=True,
        capture_output=True,
    )

    combined = (
        test.stdout
        +
        "\n"
        +
        test.stderr
    )

    print(
        combined
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests?",
        combined,
    )

    count = (
        int(
            match.group(1)
        )
        if match
        else None
    )

    if (
        test.returncode != 0
        or
        count != 108
    ):
        blocked(
            "regression",
            (
                "Expected frozen pre-formal "
                f"regression 108/108; got {count}."
            ),
            (
                "Do not start formal inference "
                "until the existing regression "
                "is restored."
            ),
        )
        return

    # ========================================================
    # H. Freeze Part1 discovery evidence
    # ========================================================

    report = {
        "stage":
            4,

        "block":
            "4.8_part_1",

        "status":
            "PASS",

        "purpose":
            (
                "read_only_formal_contract_"
                "and_API_discovery"
            ),

        "formal_population": {
            "scenario_count":
                120,

            "manifest":
                str(
                    FORMAL_MANIFEST
                ),

            "manifest_sha256":
                actual_manifest_sha,

            "scenario_ids_unique":
                True,

            "first_row_keys":
                sorted(
                    rows[
                        0
                    ].keys()
                ),
        },

        "upstream": {
            "Stage3_status":
                stage3_status,

            "Stage3_implementation_sha256":
                EXPECTED_STAGE3_IMPL_SHA,

            "Block47_implementation_sha256":
                EXPECTED_BLOCK47_IMPL_SHA,

            "deterministic_checkpoint_sha256":
                EXPECTED_DET_SHA,

            "Gaussian_checkpoint_sha256":
                EXPECTED_GAUSSIAN_SHA,

            "calibrator_sha256":
                EXPECTED_CALIBRATOR_SHA,

            "GMM_checkpoint_sha256":
                EXPECTED_GMM_SHA,

            "normalization":
                "FIT_ONLY",
        },

        "validation_API_inventory":
            module_inventory,

        "Stage4_builder_evidence":
            builder_evidence,

        "Stage3_formal_report": {
            "path":
                str(
                    formal_report_path
                ),

            "file_sha256":
                file_sha256(
                    formal_report_path
                ),

            "expected_frozen_run_sha256":
                EXPECTED_FORMAL_RUN_SHA,

            "expected_run_sha_status":
                formal_run_sha_status,

            "declared_run_sha_candidates":
                declared_run_sha,

            "relevant_flattened_fields":
                metrics,
        },

        "Part2_pre_registered_evaluation": {
            "formal_population":
                "immutable_N120",

            "observation_mode":
                "full_frozen_stage2_degraded",

            "association":
                "frozen_stage3_estimated_gnn",

            "targets":
                (
                    "WOMD_tracks_to_predict_"
                    "evaluator_only_after_predictions"
                ),

            "models": [
                "deterministic_GRU",
                "raw_Gaussian_GRU",
                "calibrated_Gaussian_GRU",
                "GMM_ablation"
            ],

            "mandatory_probabilistic_model":
                "calibrated_Gaussian_GRU",

            "classical_baselines":
                "reuse_exact_frozen_Stage3_formal_results",

            "primary_metric":
                "planar_ADE_m",

            "horizons_s": [
                0.1,
                0.3,
                0.5,
                1.0
            ],

            "confidence_levels": [
                0.50,
                0.80,
                0.90,
                0.95,
                0.99
            ],

            "calibration_metrics": [
                "Gaussian_NLL",
                "coverage_ECE",
                "coverage_event_Brier",
                "confidence_region_coverage"
            ],

            "PDF_acceptance":
                (
                    "probabilistic_predictor_beats_"
                    "at_least_one_classical_baseline_"
                    "and_calibration_is_measured"
                ),

            "internal_project_gate":
                (
                    "calibrated_Gaussian_ADE_"
                    "less_than_frozen_Stage3_CV_ADE"
                ),

            "frozen_Stage3_CV_ADE_m":
                9.538706,

            "formal_data_used_for_training":
                False,

            "formal_data_used_for_calibrator_fit":
                False,

            "formal_data_used_for_model_selection":
                False,
        },

        "regression": {
            "tests_passed":
                108,

            "tests_total":
                108,
        },

        "formal_model_inference_started":
            False,

        "formal_metrics_computed":
            False,

        "upstream_modified":
            False,
    }

    write_json(
        REPORT,
        report,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STAGE4 BLOCK 4.8 PART 1/2 DISCOVERY GATE"
    )
    print(
        "============================================================"
    )
    print(
        "Block4.7 frozen upstream = PASS"
    )
    print(
        "Stage3 frozen upstream   = PASS"
    )
    print(
        "formal manifest N        = 120 PASS"
    )
    print(
        "formal manifest SHA      = PASS"
    )
    print(
        "formal report discovered = PASS"
    )
    print(
        "model/checkpoint hashes  = PASS"
    )
    print(
        "fit-only normalization   = PASS"
    )
    print(
        "validation API inventory = CAPTURED"
    )
    print(
        "formal schema evidence   = CAPTURED"
    )
    print(
        "formal model inference   = NOT STARTED"
    )
    print(
        "formal model selection   = NONE"
    )
    print(
        "formal calibration fit   = NONE"
    )
    print(
        "full Stage4 regression   = 108 / 108 PASS"
    )
    print(
        "STATUS = PASS"
    )
    print(
        "report =",
        REPORT,
    )
    print(
        "terminal remains open    = YES"
    )


try:
    main()

except BaseException as exc:
    blocked(
        "unexpected_discovery_error",
        (
            f"{type(exc).__name__}: "
            f"{exc}"
        ),
        (
            "Read-only Part1 discovery failed. "
            "No formal model inference or "
            "upstream modification occurred."
        ),
    )

    print()
    traceback.print_exc()

# Deliberately no sys.exit().

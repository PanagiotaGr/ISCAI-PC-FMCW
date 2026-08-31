from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

PART1 = (
    S5
    / "reports/block510_part1_acceptance_audit.json"
)

B58_FORMAL = (
    S5
    / "reports/block58_formal_evaluation.json"
)

B58_CLOSURE = (
    S5
    / "artifacts/block58/block58_final_closure.json"
)

B58_RECONCILIATION = (
    S5
    / "artifacts/block58/"
      "block58_part2_reconciliation.json"
)

B59 = (
    S5
    / "reports/block59_reproducibility.json"
)

B59_PART2 = (
    S5
    / "reports/"
      "block59_part2_terminal_contract_repair.json"
)

FORMAL_MANIFEST = (
    S3
    / "artifacts/block38e/"
      "formal_validation_120.jsonl"
)

STAGE4_CLOSURE = (
    S4
    / "reports/stage4_final_closure.json"
)

STAGE4_HANDOFF = (
    S4
    / "artifacts/block410/"
      "stage4_to_stage5_handoff.json"
)

STAGE4_FREEZE = (
    S4
    / "artifacts/block410/"
      "stage4_final_freeze_manifest.json"
)

PDF_CERT = (
    ROOT
    / "audits/pdf_compliance_stages0_4/"
      "pdf_compliance_final_certification.json"
)

HANDOFF = (
    S5
    / "artifacts/block510/"
      "stage5_to_stage6_handoff.json"
)

CLOSURE = (
    S5
    / "reports/stage5_final_closure.json"
)

FREEZE = (
    S5
    / "artifacts/block510/"
      "stage5_final_freeze_manifest.json"
)

MIN_FREE_GIB = 250.0

EXPECTED_TESTS = 257

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_B59_SHA = (
    "980bfbd2007ed798742bd7c36b5e8b6b"
    "cd0e808590c0dc5d9608db1ec6211845"
)

EXPECTED_STAGE4_CLOSURE_SHA = (
    "570da4feb918b1025b5e85cc919360d9"
    "22b471c468c85b3844f13fb7774e7c2f"
)

EXPECTED_STAGE4_HANDOFF_SHA = (
    "491bce010d35c2a394f879ecf35ed26e"
    "f1de92f7fff1465dcbf46b072a87c6fd"
)

EXPECTED_STAGE4_FREEZE_SHA = (
    "88e3f290e1f7c1d037684adc132ea8ae"
    "78af057b3e3ab564fb29516687eb25e7"
)

EXPECTED_PDF_CERT_SHA = (
    "7c94bb9deb37a5eba29c9237bfcf4b0d"
    "24e4031d9b1b22a500c5a63730bf0d06"
)


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
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


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def canonical_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def write_deterministic_json(
    path: Path,
    payload,
):
    desired = canonical_bytes(
        payload
    )

    if path.exists():
        existing = path.read_bytes()

        if existing != desired:
            raise RuntimeError(
                "Existing final artifact differs "
                f"from deterministic payload: {path}"
            )

        return "ALREADY_EXACT"

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_bytes(
        desired
    )

    os.replace(
        temporary,
        path,
    )

    return "WRITTEN"


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(value):
    return (
        isinstance(
            value,
            (int, float),
        )
        and
        not isinstance(value, bool)
        and
        math.isfinite(
            float(value)
        )
    )


def run_full_regression():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(S5 / "tests"),
            "-p",
            "test_*.py",
        ],
        cwd=str(S5),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    count = None

    for line in process.stdout.splitlines():
        stripped = line.strip()

        if stripped.startswith("Ran "):
            parts = stripped.split()

            if len(parts) >= 2:
                try:
                    count = int(parts[1])
                except Exception:
                    pass

    return (
        process.returncode,
        count,
        process.stdout,
    )


def implementation_files():
    roots = (
        S5 / "src",
        S5 / "scripts",
        S5 / "tests",
        S5 / "configs",
    )

    files = []

    for root in roots:
        if not root.exists():
            continue

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

    return sorted(
        files,
        key=lambda p:
            str(
                p.relative_to(S5)
            ),
    )


def implementation_seal():
    records = []

    aggregate = sha256()

    for path in implementation_files():
        relative = str(
            path.relative_to(S5)
        )

        digest = sha256_file(
            path
        )

        records.append({
            "path":
                relative,

            "sha256":
                digest,
        })

        aggregate.update(
            relative.encode("utf-8")
        )
        aggregate.update(b"\0")
        aggregate.update(
            digest.encode("ascii")
        )
        aggregate.update(b"\n")

    return {
        "file_count":
            len(records),

        "aggregate_sha256":
            aggregate.hexdigest(),

        "files":
            records,
    }


def exact_status(
    payload,
    expected,
    *,
    name,
):
    actual = payload.get(
        "status"
    )

    require(
        actual == expected,
        (
            f"{name} status is "
            f"{actual!r}, expected "
            f"{expected!r}."
        ),
    )


# ============================================================
# Finalizer
# ============================================================

def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.10 PART 2/2"
    )
    print(
        "CANONICAL FINAL FREEZE + STAGE5→STAGE6 HANDOFF"
    )
    print(
        "============================================================"
    )

    required = (
        PART1,
        B58_FORMAL,
        B58_CLOSURE,
        B58_RECONCILIATION,
        B59,
        B59_PART2,
        FORMAL_MANIFEST,
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE4_FREEZE,
        PDF_CERT,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing required frozen evidence: "
            + ", ".join(missing)
        ),
    )

    # ========================================================
    # A. Stage4 / formal continuity
    # ========================================================

    print()
    print(
        "===== A. FROZEN UPSTREAM CONTINUITY ====="
    )

    require(
        sha256_file(
            STAGE4_CLOSURE
        )
        ==
        EXPECTED_STAGE4_CLOSURE_SHA,
        "Stage4 closure SHA changed.",
    )

    require(
        sha256_file(
            STAGE4_HANDOFF
        )
        ==
        EXPECTED_STAGE4_HANDOFF_SHA,
        "Stage4 handoff SHA changed.",
    )

    require(
        sha256_file(
            STAGE4_FREEZE
        )
        ==
        EXPECTED_STAGE4_FREEZE_SHA,
        "Stage4 freeze SHA changed.",
    )

    require(
        sha256_file(
            PDF_CERT
        )
        ==
        EXPECTED_PDF_CERT_SHA,
        "Stages0–4 PDF certification SHA changed.",
    )

    require(
        sha256_file(
            FORMAL_MANIFEST
        )
        ==
        EXPECTED_FORMAL_MANIFEST_SHA,
        "Formal N=120 manifest SHA changed.",
    )

    print(
        "Stages0–4 PDF certification = EXACT PASS"
    )

    print(
        "Stage4 closure/handoff/freeze = EXACT PASS"
    )

    print(
        "formal N=120 manifest         = EXACT PASS"
    )

    # ========================================================
    # B. Part1 acceptance
    # ========================================================

    print()
    print(
        "===== B. BLOCK5.10 PART1 ACCEPTANCE ====="
    )

    part1 = load_json(
        PART1
    )

    exact_status(
        part1,
        "PASS_READY_FOR_FINAL_FREEZE",
        name="Block5.10 Part1",
    )

    mandatory = part1.get(
        "mandatory_acceptance",
        {}
    )

    require(
        mandatory.get("passed")
        ==
        mandatory.get("total")
        ==
        17,
        (
            "Stage5 mandatory acceptance "
            "is not 17/17."
        ),
    )

    matrix = mandatory.get(
        "matrix",
        {}
    )

    require(
        len(matrix) == 17
        and
        all(
            value is True
            for value in matrix.values()
        ),
        (
            "Stage5 acceptance matrix "
            "contains a non-PASS item."
        ),
    )

    formal_summary = part1.get(
        "formal",
        {}
    )

    coverage = float(
        formal_summary[
            "empirical_probability_coverage"
        ]
    )

    overhead_reduction = float(
        formal_summary[
            "overhead_reduction_vs_exhaustive"
        ]
    )

    require(
        coverage >= 0.95,
        "Final coverage gate failed.",
    )

    require(
        overhead_reduction > 0.0,
        "Final overhead-reduction gate failed.",
    )

    print(
        "mandatory acceptance = 17 / 17 PASS"
    )

    print(
        "empirical coverage   =",
        coverage,
        "PASS",
    )

    print(
        "overhead reduction   =",
        overhead_reduction,
        "PASS",
    )

    # ========================================================
    # C. Block5.8 formal + Block5.9 reproducibility
    # ========================================================

    print()
    print(
        "===== C. FORMAL / REPRODUCIBILITY SEAL ====="
    )

    formal58 = load_json(
        B58_FORMAL
    )

    require(
        formal58.get("status")
        ==
        "PASS",
        "Block5.8 formal report is not PASS.",
    )

    require(
        int(
            formal58[
                "formal_population"
            ][
                "N"
            ]
        )
        ==
        120,
        "Block5.8 formal population is not N=120.",
    )

    require(
        formal58[
            "formal_population"
        ][
            "same_immutable_Stage3_4_population"
        ]
        is True,
        (
            "Block5.8 formal population "
            "is not immutable Stage3/4 N=120."
        ),
    )

    reconciliation58 = load_json(
        B58_RECONCILIATION
    )

    exact_status(
        reconciliation58,
        "PASS_RECONCILED_COMPLETE",
        name="Block5.8 reconciliation",
    )

    r59 = load_json(
        B59
    )

    require(
        sha256_file(
            B59
        )
        ==
        EXPECTED_B59_SHA,
        "Block5.9 reproducibility SHA changed.",
    )

    exact_status(
        r59,
        "PASS_FROZEN",
        name="Block5.9 reproducibility",
    )

    require(
        r59.get(
            "exact_match_to_frozen_block58"
        )
        is True,
        (
            "Block5.9 no longer matches "
            "frozen Block5.8 exactly."
        ),
    )

    require(
        r59.get(
            "fresh_process_repeat_exact"
        )
        is True,
        (
            "Fresh-process reproducibility "
            "is not exact."
        ),
    )

    require(
        int(
            r59.get(
                "fresh_process_count"
            )
        )
        >=
        2,
        (
            "Fresh-process repeat "
            "count is less than two."
        ),
    )

    require(
        r59.get(
            "beam_probability_vectors_exact"
        )
        is True,
        (
            "Beam-probability vectors "
            "are not exact."
        ),
    )

    print(
        "formal N=120              = PASS"
    )

    print(
        "Block5.8 closure/reconcile= PASS"
    )

    print(
        "Block5.9                  = PASS_FROZEN"
    )

    print(
        "fresh-process repeat      = EXACT PASS"
    )

    print(
        "beam probabilities        = EXACT PASS"
    )

    # ========================================================
    # D. Pre-freeze regression
    # ========================================================

    print()
    print(
        "===== D. PRE-FREEZE FULL REGRESSION ====="
    )

    rc_before, tests_before, output_before = (
        run_full_regression()
    )

    if rc_before != 0:
        print(output_before)

    require(
        rc_before == 0,
        "Pre-freeze Stage5 regression failed.",
    )

    require(
        tests_before == EXPECTED_TESTS,
        (
            "Pre-freeze regression count "
            f"is {tests_before}, expected "
            f"{EXPECTED_TESTS}."
        ),
    )

    print(
        "pre-freeze regression =",
        f"{tests_before} / {EXPECTED_TESTS} PASS",
    )

    # ========================================================
    # E. Storage reserve
    # ========================================================

    print()
    print(
        "===== E. STORAGE RESERVE ====="
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
            "250-GiB hard reserve violated: "
            f"{free_gib:.3f} GiB free."
        ),
    )

    print(
        "free GiB =",
        round(
            free_gib,
            3,
        ),
    )

    print(
        "250-GiB reserve = PASS"
    )

    # ========================================================
    # F. Current implementation seal
    # ========================================================

    print()
    print(
        "===== F. CURRENT STAGE5 IMPLEMENTATION SEAL ====="
    )

    impl = implementation_seal()

    require(
        impl[
            "file_count"
        ]
        >
        0,
        "Stage5 implementation seal is empty.",
    )

    print(
        "implementation files =",
        impl[
            "file_count"
        ],
    )

    print(
        "implementation SHA256 =",
        impl[
            "aggregate_sha256"
        ],
    )

    # ========================================================
    # G. Stage5 → Stage6 handoff
    # ========================================================

    print()
    print(
        "===== G. STAGE5 → STAGE6 HANDOFF ====="
    )

    handoff_payload = {
        "project":
            "Agni",

        "from_stage":
            5,

        "to_stage":
            6,

        "status":
            "FROZEN_HANDOFF",

        "Stage5_status":
            "COMPLETE_FROZEN",

        "Stage6_started":
            False,

        "shared_posterior": {
            "upstream_trajectory_family":
                "calibrated_Gaussian_GRU",

            "semantics":
                (
                    "calibrated future trajectory "
                    "posterior propagated analytically/"
                    "deterministically to receiver-aware "
                    "position and angular uncertainty"
                ),

            "receiver_aware":
                True,

            "receiver_geometry_modes":
                [
                    "centroid",
                    "known",
                    "uncertain",
                ],

            "prediction_horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "measurement_uncertainty_R_t":
                (
                    "remains distinct from "
                    "predictive covariance"
                ),

            "future_ground_truth_as_controller_input":
                False,

            "tracks_to_predict_as_receiver_selector":
                False,
        },

        "communication_controller_frozen": {
            "codebooks":
                [
                    16,
                    32,
                    64,
                ],

            "adaptive_probability_targets":
                [
                    0.90,
                    0.95,
                    0.975,
                    0.99,
                ],

            "nominal_q":
                0.95,

            "primary_acceptance_geometry":
                "uncertain",

            "empirical_probability_coverage":
                coverage,

            "overhead_reduction_vs_exhaustive":
                overhead_reduction,

            "latency_fallback_controller":
                True,

            "loss_of_lock_recovery":
                (
                    "widened fallback followed "
                    "by exhaustive current-codebook sweep"
                ),

            "optical_chain":
                (
                    "pointing_error -> gain -> "
                    "received_power -> SNR -> "
                    "DPSK_BER -> effective_rate"
                ),
        },

        "Stage6_required_next_layer": {
            "name":
                "predictive_class_aware_ADB",

            "input":
                (
                    "the same frozen future "
                    "trajectory/angular posterior"
                ),

            "required_geometry":
                (
                    "future actor box/corner "
                    "projection with uncertainty"
                ),

            "required_output_space":
                (
                    "illumination pixels/angular cells "
                    "and intensity/dimming masks"
                ),

            "communication_beam_actuator_reused_as_ADB":
                False,

            "reason":
                (
                    "communication beam space and "
                    "illumination beam/pixel space "
                    "are separate downstream mappings"
                ),

            "Stage6_completion_goal":
                (
                    "reduce vehicle shadow violations "
                    "without uncontrolled over-masking "
                    "or loss of VRU visibility"
                ),
        },

        "deferred_beyond_Stage6": {
            "joint_shared_vs_independent_evaluation":
                "Stage7",

            "joint_communication_illumination_tradeoffs":
                "Stage7",

            "full_joint_end_to_end_latency":
                "Stage7",

            "statistical_failure_case_analysis":
                "Stage7",

            "DeepSense_measured_beam_power_validation":
                "Stage8",
        },

        "provenance": {
            "Stage4_handoff_path":
                str(
                    STAGE4_HANDOFF
                ),

            "Stage4_handoff_sha256":
                sha256_file(
                    STAGE4_HANDOFF
                ),

            "formal_manifest_path":
                str(
                    FORMAL_MANIFEST
                ),

            "formal_manifest_sha256":
                sha256_file(
                    FORMAL_MANIFEST
                ),

            "Block5_8_formal_report":
                str(
                    B58_FORMAL
                ),

            "Block5_8_formal_report_sha256":
                sha256_file(
                    B58_FORMAL
                ),

            "Block5_8_closure":
                str(
                    B58_CLOSURE
                ),

            "Block5_8_closure_sha256":
                sha256_file(
                    B58_CLOSURE
                ),

            "Block5_9_reproducibility":
                str(
                    B59
                ),

            "Block5_9_reproducibility_sha256":
                sha256_file(
                    B59
                ),

            "Block5_10_acceptance":
                str(
                    PART1
                ),

            "Block5_10_acceptance_sha256":
                sha256_file(
                    PART1
                ),
        },

        "scientific_execution_at_handoff": {
            "training":
                False,

            "formal_inference":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,
        },
    }

    handoff_write = write_deterministic_json(
        HANDOFF,
        handoff_payload,
    )

    handoff_sha = sha256_file(
        HANDOFF
    )

    print(
        "handoff write =",
        handoff_write,
    )

    print(
        "handoff SHA256 =",
        handoff_sha,
    )

    print(
        "Stage6 started = NO"
    )

    # ========================================================
    # H. Canonical Stage5 final closure
    # ========================================================

    print()
    print(
        "===== H. CANONICAL STAGE5 CLOSURE ====="
    )

    closure_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "status":
            "COMPLETE_FROZEN",

        "scope":
            (
                "receiver-aware probabilistic "
                "communication beam controller "
                "with optical link evaluation"
            ),

        "completion_gate": {
            "requested_coverage":
                "PASS",

            "overhead_reduction_vs_fixed_or_exhaustive":
                "PASS",

            "empirical_probability_coverage":
                coverage,

            "nominal_requested_q":
                0.95,

            "overhead_reduction_vs_exhaustive":
                overhead_reduction,
        },

        "mandatory_acceptance": {
            "passed":
                17,

            "total":
                17,

            "status":
                "PASS",
        },

        "formal_evaluation": {
            "scenario_count":
                120,

            "same_immutable_Stage3_4_population":
                True,

            "manifest_sha256":
                EXPECTED_FORMAL_MANIFEST_SHA,

            "receiver_geometry_modes":
                [
                    "centroid",
                    "known",
                    "uncertain",
                ],

            "codebooks":
                [
                    16,
                    32,
                    64,
                ],

            "prediction_horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "reported_probability_targets":
                [
                    0.90,
                    0.95,
                    0.975,
                    0.99,
                ],

            "formal_parameter_tuning":
                False,

            "posthoc_selection":
                False,
        },

        "required_components": {
            "receiver_angular_posterior":
                "PASS",

            "centroid_known_uncertain_receiver_geometry":
                "PASS",

            "beam_codebooks_and_probabilities":
                "PASS",

            "exhaustive_persistence_nearest_fixed_top1_3_5_oracle":
                "PASS",

            "adaptive_TopK":
                "PASS",

            "latency_hysteresis_fallback_reacquisition":
                "PASS",

            "optical_pointing_gain_power_SNR_DPSK_BER_effective_rate":
                "PASS",

            "reliability_overhead_curves_and_Pareto":
                "PASS",
        },

        "causality": {
            "future_truth_used_as_controller_input":
                False,

            "tracks_to_predict_used_as_receiver_selector":
                False,

            "formal_population_used_for_tuning":
                False,

            "formal_outcomes_used_for_parameter_selection":
                False,
        },

        "reproducibility": {
            "status":
                "PASS_FROZEN",

            "exact_match_to_frozen_Block5_8":
                True,

            "fresh_process_repeat_exact":
                True,

            "fresh_process_count":
                int(
                    r59[
                        "fresh_process_count"
                    ]
                ),

            "beam_probability_vectors_exact":
                True,

            "report_sha256":
                EXPECTED_B59_SHA,
        },

        "regression": {
            "tests":
                EXPECTED_TESTS,

            "passed":
                EXPECTED_TESTS,

            "status":
                "PASS",
        },

        "implementation_seal": {
            "file_count":
                impl[
                    "file_count"
                ],

            "aggregate_sha256":
                impl[
                    "aggregate_sha256"
                ],
        },

        "Stage5_to_Stage6_handoff": {
            "path":
                str(
                    HANDOFF
                ),

            "sha256":
                handoff_sha,

            "status":
                "FROZEN_HANDOFF",
        },

        "scope_boundary": {
            "predictive_ADB":
                "STAGE6",

            "joint_beam_ADB_evaluation":
                "STAGE7",

            "full_joint_end_to_end_latency":
                "STAGE7",

            "DeepSense":
                "STAGE8",

            "blockage":
                "OPTIONAL_EXTENSION",
        },

        "scientific_outputs_changed_during_closure":
            False,

        "training_during_closure":
            False,

        "formal_inference_during_closure":
            False,

        "parameter_tuning_during_closure":
            False,

        "Stage6_started":
            False,
    }

    closure_write = write_deterministic_json(
        CLOSURE,
        closure_payload,
    )

    closure_sha = sha256_file(
        CLOSURE
    )

    print(
        "closure write =",
        closure_write,
    )

    print(
        "closure SHA256 =",
        closure_sha,
    )

    print(
        "canonical Stage5 status = COMPLETE_FROZEN"
    )

    # ========================================================
    # I. Freeze manifest
    # ========================================================

    print()
    print(
        "===== I. FINAL FREEZE MANIFEST ====="
    )

    critical_artifacts = {
        "Stage4_closure":
            STAGE4_CLOSURE,

        "Stage4_to_Stage5_handoff":
            STAGE4_HANDOFF,

        "Stage4_freeze_manifest":
            STAGE4_FREEZE,

        "Stages0_4_PDF_certification":
            PDF_CERT,

        "formal_N120_manifest":
            FORMAL_MANIFEST,

        "Block5_8_formal_evaluation":
            B58_FORMAL,

        "Block5_8_final_closure":
            B58_CLOSURE,

        "Block5_8_reconciliation":
            B58_RECONCILIATION,

        "Block5_9_reproducibility":
            B59,

        "Block5_9_terminal_contract_repair":
            B59_PART2,

        "Block5_10_acceptance_audit":
            PART1,

        "Stage5_to_Stage6_handoff":
            HANDOFF,

        "Stage5_final_closure":
            CLOSURE,
    }

    critical_hashes = {
        name: {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }
        for name, path in (
            critical_artifacts.items()
        )
    }

    freeze_payload = {
        "project":
            "Agni",

        "stage":
            5,

        "status":
            "FROZEN",

        "canonical_closure": {
            "path":
                str(
                    CLOSURE
                ),

            "sha256":
                closure_sha,
        },

        "Stage5_to_Stage6_handoff": {
            "path":
                str(
                    HANDOFF
                ),

            "sha256":
                handoff_sha,
        },

        "implementation": {
            "file_count":
                impl[
                    "file_count"
                ],

            "aggregate_sha256":
                impl[
                    "aggregate_sha256"
                ],

            "files":
                impl[
                    "files"
                ],
        },

        "critical_artifacts":
            critical_hashes,

        "formal_contract": {
            "N":
                120,

            "manifest_sha256":
                EXPECTED_FORMAL_MANIFEST_SHA,

            "codebooks":
                [
                    16,
                    32,
                    64,
                ],

            "receiver_geometries":
                [
                    "centroid",
                    "known",
                    "uncertain",
                ],

            "horizons_s":
                [
                    0.1,
                    0.3,
                    0.5,
                    1.0,
                ],

            "reported_q":
                [
                    0.90,
                    0.95,
                    0.975,
                    0.99,
                ],

            "nominal_q":
                0.95,
        },

        "acceptance": {
            "mandatory_passed":
                17,

            "mandatory_total":
                17,

            "empirical_probability_coverage":
                coverage,

            "overhead_reduction_vs_exhaustive":
                overhead_reduction,
        },

        "reproducibility": {
            "status":
                "PASS_FROZEN",

            "report_sha256":
                EXPECTED_B59_SHA,

            "fresh_process_repeat_exact":
                True,
        },

        "regression": {
            "expected_tests":
                EXPECTED_TESTS,

            "status":
                "PASS",
        },

        "storage": {
            "hard_reserve_GiB":
                MIN_FREE_GIB,

            "free_GiB_at_freeze":
                free_gib,

            "reserve_gate_pass":
                True,
        },

        "freeze_semantics": {
            "Stage5_scientific_implementation_frozen":
                True,

            "Stage5_formal_outputs_frozen":
                True,

            "Stage5_acceptance_policy_frozen":
                True,

            "Stage6_started":
                False,

            "future_changes_require_explicit_new_stage_or_reopening":
                True,
        },
    }

    freeze_write = write_deterministic_json(
        FREEZE,
        freeze_payload,
    )

    freeze_sha = sha256_file(
        FREEZE
    )

    print(
        "freeze write =",
        freeze_write,
    )

    print(
        "freeze SHA256 =",
        freeze_sha,
    )

    # ========================================================
    # J. Post-write regression
    # ========================================================

    print()
    print(
        "===== J. POST-FREEZE FULL REGRESSION ====="
    )

    rc_after, tests_after, output_after = (
        run_full_regression()
    )

    if rc_after != 0:
        print(output_after)

    require(
        rc_after == 0,
        (
            "Post-freeze Stage5 regression failed."
        ),
    )

    require(
        tests_after == EXPECTED_TESTS,
        (
            "Post-freeze regression count "
            f"is {tests_after}; expected "
            f"{EXPECTED_TESTS}."
        ),
    )

    print(
        "post-freeze regression =",
        f"{tests_after} / {EXPECTED_TESTS} PASS",
    )

    # ========================================================
    # K. Final readback
    # ========================================================

    print()
    print(
        "===== K. FINAL READBACK ====="
    )

    closure_check = load_json(
        CLOSURE
    )

    handoff_check = load_json(
        HANDOFF
    )

    freeze_check = load_json(
        FREEZE
    )

    exact_status(
        closure_check,
        "COMPLETE_FROZEN",
        name="Stage5 final closure",
    )

    exact_status(
        handoff_check,
        "FROZEN_HANDOFF",
        name="Stage5→Stage6 handoff",
    )

    exact_status(
        freeze_check,
        "FROZEN",
        name="Stage5 freeze manifest",
    )

    require(
        freeze_check[
            "canonical_closure"
        ][
            "sha256"
        ]
        ==
        sha256_file(
            CLOSURE
        ),
        "Closure SHA readback mismatch.",
    )

    require(
        freeze_check[
            "Stage5_to_Stage6_handoff"
        ][
            "sha256"
        ]
        ==
        sha256_file(
            HANDOFF
        ),
        "Handoff SHA readback mismatch.",
    )

    require(
        closure_check[
            "Stage6_started"
        ]
        is False,
        "Closure incorrectly says Stage6 started.",
    )

    require(
        handoff_check[
            "Stage6_started"
        ]
        is False,
        "Handoff incorrectly says Stage6 started.",
    )

    require(
        handoff_check[
            "Stage6_required_next_layer"
        ][
            "communication_beam_actuator_reused_as_ADB"
        ]
        is False,
        (
            "Communication and ADB actuator "
            "spaces were incorrectly merged."
        ),
    )

    print(
        "closure readback = COMPLETE_FROZEN PASS"
    )

    print(
        "handoff readback = FROZEN_HANDOFF PASS"
    )

    print(
        "freeze readback  = FROZEN PASS"
    )

    print(
        "communication / ADB actuator separation = PASS"
    )

    print(
        "Stage6 started = NO"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.10 PART 2/2 — FINAL STAGE5 CLOSURE"
    )
    print(
        "============================================================"
    )

    print(
        "Blocks 5.0 → 5.10        = PASS"
    )

    print(
        "mandatory acceptance      = 17 / 17 PASS"
    )

    print(
        "formal N=120              = PASS"
    )

    print(
        "requested coverage        =",
        coverage,
        "PASS",
    )

    print(
        "overhead reduction        =",
        overhead_reduction,
        "PASS",
    )

    print(
        "reproducibility           = PASS_FROZEN"
    )

    print(
        "pre-freeze regression     =",
        f"{tests_before} / {EXPECTED_TESTS} PASS",
    )

    print(
        "post-freeze regression    =",
        f"{tests_after} / {EXPECTED_TESTS} PASS",
    )

    print(
        "implementation files      =",
        impl[
            "file_count"
        ],
    )

    print(
        "implementation SHA256     =",
        impl[
            "aggregate_sha256"
        ],
    )

    print(
        "Stage5 closure SHA256     =",
        closure_sha,
    )

    print(
        "Stage5→Stage6 handoff SHA =",
        handoff_sha,
    )

    print(
        "Stage5 freeze SHA256      =",
        freeze_sha,
    )

    print(
        "scientific outputs changed= NO"
    )

    print(
        "training/formal inference = NO"
    )

    print(
        "parameter tuning          = NO"
    )

    print(
        "Stage6 started            = NO"
    )

    print(
        "STATUS = COMPLETE_FROZEN"
    )

    print(
        "closure =",
        CLOSURE,
    )

    print(
        "handoff =",
        HANDOFF,
    )

    print(
        "freeze =",
        FREEZE,
    )

    print(
        "terminal remains open = YES"
    )


try:
    main()

except BaseException as exc:
    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.10 PART 2/2 = BLOCKED"
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
    traceback.print_exc()

    print(
        "training/formal inference = NO"
    )

    print(
        "parameter tuning          = NO"
    )

    print(
        "Stage6 implementation     = NOT STARTED"
    )

    print(
        "terminal remains open     = YES"
    )

# Deliberately no non-zero sys.exit().

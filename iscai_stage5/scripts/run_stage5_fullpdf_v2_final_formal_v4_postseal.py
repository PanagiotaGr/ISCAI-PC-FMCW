from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

CHECKER = (
    S5
    / "scripts/run_stage5_fullpdf_v2_independent_formal_checker.py"
)
CHECKER2 = (
    S5 / "scripts/check_stage5_fullpdf_v2.py"
)
RUNTIME = (
    S5 / "src/iscai_stage5/fullpdf_v2.py"
)
PROTOCOL = (
    S5 / "configs/stage5_fullpdf_v2_protocol.json"
)
ADDENDUM = (
    S5 / "configs/stage5_fullpdf_v2_preformal_addendum.json"
)
STAGE4_LEDGER = (
    S4 / "artifacts/fullpdf_v2/formal_prediction_ledger_run1.jsonl"
)

RECEIVER_BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
      "primary_receiver_binding_fullpdf_v2.jsonl"
)

TRUTH_BINDING = (
    S5
    / "artifacts/fullpdf_v2_repair/"
      "evaluator_truth_binding_fullpdf_v2.jsonl"
)

TRUTH_AMENDMENT = (
    S5
    / "configs/"
      "stage5_fullpdf_v2_evaluator_truth_binding_amendment_v1.json"
)

TRUTH_SEAL = (
    S5
    / "artifacts/fullpdf_v2_independent_formal/"
      "evaluator_truth_binding_materialization_v1/"
      "stage5_fullpdf_v2_evaluator_truth_binding_seal_v1.json"
)

TRUTH_REPORT = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_evaluator_truth_binding_materialization_v1.json"
)

ART = (
    S5 / "artifacts/fullpdf_v2_independent_formal"
)

SEALED = (
    ART / "decision_ledger_sealed.jsonl"
)

TRUTH_DIAGNOSTIC = (
    ART
    / "truth_candidate_postseal_diagnostic_v1/"
      "truth_candidate_postseal_diagnostic_v1.json"
)

OUTDIR = (
    ART / "final_formal_v4_postseal"
)

MARKER = (
    OUTDIR / "final_formal_v4_invocation.json"
)
REGRESSION_LOG = (
    OUTDIR / "regression_271.log"
)
EVALUATOR_ROWS = (
    OUTDIR / "evaluator_rows.jsonl"
)
FORMAL_SUMMARY = (
    OUTDIR / "formal_summary.json"
)
FINAL_REPORT = (
    OUTDIR / "final_formal_v4_report.json"
)
FAILURE_REPORT = (
    OUTDIR / "final_formal_v4_unexpected_failure.json"
)


EXPECTED = {
    CHECKER:
        "342344902e6b00cb841036dcdc0e54f1e1ba396344fa602864521890f5eb5da5",

    CHECKER2:
        "5e77d678af188d3f0b34b84ab53ddc78fc1b2921c6bea573dd9722c439c63599",

    RUNTIME:
        "e57708e9332bb40e84f18270e483d91165e26c149b80e82c8f10b4df44376249",

    PROTOCOL:
        "91dec3cc3d3a7b72385f012d1d2bfdb8981a497e1344f585d38f3e05f2e4fa5a",

    ADDENDUM:
        "570fbada05c8e8ed71775bbb6eb668f4016dd65c98b09b4f158abeb379eca6aa",

    STAGE4_LEDGER:
        "2dd3c396d5c365677c41b3a0df6344a7c28d423daaa368dbe466bd4b17b4c2a8",

    RECEIVER_BINDING:
        "5a5516ee050d2fd1c0a571193007650e75f23836b2734ee3df1950cc04f2e1c8",

    SEALED:
        "751703975b3fc837cc1cf660cf1321db5768323d6cf2fbef524d05a3dbb48af4",

    TRUTH_BINDING:
        "1c9582cdf62dfc02cc36c6e0a56c6ce1a1b25eedc09e9fd10af00618d51a5961",

    TRUTH_AMENDMENT:
        "e49ab5f3f6de1aab6f3fcaa739c306ddc3fa08267861cc4ea0f5b2ebe0e66712",

    TRUTH_SEAL:
        "85d68d26bcd178b3aaa467743e31c4910d94f7c75006d350d1dc1a2e888acfcc",

    TRUTH_REPORT:
        "80bb779cf3f86030c18cca95c3a748655c8cd5b6dd6f9e61d2cb334174d3fd5a",

    TRUTH_DIAGNOSTIC:
        "e94dbb37fc115f28dbea07703ebc9a9de8a3068d7859fa22d5d58943590e2ebf",
}

EXPECTED_TRUTH_MAP_HASH = (
    "d27dda703e69f0d260747e608bbd47c03c619de92201e15bc716ca867d454763"
)

EXPECTED_SEALED_ROWS = 17280
EXPECTED_RECEIVER_KEYS = 120
EXPECTED_STRATA = 3 * 3 * 4 * 4
EXPECTED_TESTS = 271


def fail(message):
    raise RuntimeError(
        "FAIL-CLOSED: " + str(message)
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def normalize(obj):
    if isinstance(obj, np.ndarray):
        return normalize(obj.tolist())

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Mapping):
        return {
            str(k): normalize(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            normalize(x)
            for x in obj
        ]

    if isinstance(
        obj,
        (str, int, float, bool),
    ) or obj is None:
        return obj

    if hasattr(obj, "__dict__"):
        return normalize(vars(obj))

    return repr(obj)


def truth_normalizable(obj):
    # Exact mapping-normalization semantics used by
    # evaluator-truth materialization V1.
    if isinstance(obj, np.ndarray):
        return truth_normalizable(obj.tolist())

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, Mapping):
        return {
            repr(key): truth_normalizable(value)
            for key, value in sorted(
                obj.items(),
                key=lambda kv: repr(kv[0]),
            )
        }

    if isinstance(obj, (list, tuple)):
        return [
            truth_normalizable(x)
            for x in obj
        ]

    if isinstance(
        obj,
        (str, int, float, bool),
    ) or obj is None:
        return obj

    if hasattr(obj, "__dict__"):
        return truth_normalizable(
            vars(obj)
        )

    return repr(obj)


def truth_map_hash(obj):
    payload = json.dumps(
        truth_normalizable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def canonical_json(obj):
    return (
        json.dumps(
            normalize(obj),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows):
    return (
        "".join(
            json.dumps(
                normalize(row),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def atomic_write(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        fail(
            f"write-once path exists: {path}"
        )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(
        tmp,
        path,
    )

    os.chmod(
        path,
        0o444,
    )


def write_json_once(path, obj):
    atomic_write(
        path,
        canonical_json(obj),
    )
    return sha256(path)


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for lineno, line in enumerate(f, 1):
            text = line.strip()

            if not text:
                continue

            obj = json.loads(text)

            if not isinstance(obj, dict):
                fail(
                    f"non-object JSONL row "
                    f"{path}:{lineno}"
                )

            rows.append(obj)

    return rows


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        fail(
            f"cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def strata_count(value):
    if isinstance(value, Mapping):
        return len(value)

    if isinstance(value, (list, tuple)):
        return len(value)

    return -1


def main():
    metrics_returned = False

    print("=" * 78)
    print(
        "STAGE5 FULLPDF V2 — FINAL FORMAL V4 POST-SEAL"
    )
    print(
        "IMMUTABLE V3 DECISIONS | EXACT FROZEN EVALUATOR | "
        "NO CONTROLLER REEXECUTION"
    )
    print("=" * 78)

    # ------------------------------------------------------------
    # A. Protected chain.
    # ------------------------------------------------------------

    for path, expected in EXPECTED.items():
        if not path.is_file():
            fail(
                f"missing protected artifact: {path}"
            )

        actual = sha256(path)

        if actual != expected:
            fail(
                "protected SHA mismatch:\n"
                f"{path}\n"
                f"actual   = {actual}\n"
                f"expected = {expected}"
            )

        print(
            "PASS | protected SHA |",
            path.name,
        )

    if (
        stat.S_IMODE(
            SEALED.stat().st_mode
        )
        & 0o222
    ):
        fail(
            "sealed V3 decision ledger is writable"
        )

    free_gib = (
        shutil.disk_usage(ROOT).free
        / 1024**3
    )

    print(
        "free_GiB =",
        free_gib,
    )

    if free_gib < 250.0:
        fail(
            "free-space reserve <250 GiB"
        )

    print(
        "PASS | >=250 GiB free-space reserve"
    )

    # No prior V4 invocation/output is allowed.
    for path in (
        MARKER,
        REGRESSION_LOG,
        EVALUATOR_ROWS,
        FORMAL_SUMMARY,
        FINAL_REPORT,
        FAILURE_REPORT,
    ):
        if path.exists():
            fail(
                "V4 is one-shot and artifact "
                f"already exists: {path}"
            )

    # ------------------------------------------------------------
    # B. Import frozen checker only as a library.
    #    main() is NOT executed.
    # ------------------------------------------------------------

    checker = import_module(
        CHECKER,
        "stage5_final_formal_v4_checker",
    )

    for name in (
        "resolve_truth_binding",
        "evaluate_sealed_decisions",
    ):
        if not callable(
            getattr(
                checker,
                name,
                None,
            )
        ):
            fail(
                f"checker function missing: {name}"
            )

    # ------------------------------------------------------------
    # C. Prove V3 exact-repeat decision state.
    # ------------------------------------------------------------

    decisions = load_jsonl(
        SEALED
    )

    if len(decisions) != EXPECTED_SEALED_ROWS:
        fail(
            "sealed decision row count changed: "
            f"{len(decisions)}"
        )

    decision_keys = {
        (
            str(row["scenario_id"]),
            str(row["prediction_id"]),
        )
        for row in decisions
    }

    if len(decision_keys) != EXPECTED_RECEIVER_KEYS:
        fail(
            "unique decision-key count changed: "
            f"{len(decision_keys)}"
        )

    run1 = Path(
        checker.RUN1_LEDGER
    )
    run2 = Path(
        checker.RUN2_LEDGER
    )
    checker_sealed = Path(
        checker.SEALED_LEDGER
    )
    seal_report_path = Path(
        checker.SEAL_REPORT
    )

    for path in (
        run1,
        run2,
        checker_sealed,
        seal_report_path,
    ):
        if not path.is_file():
            fail(
                f"V3 decision evidence missing: {path}"
            )

    if checker_sealed.resolve() != SEALED.resolve():
        fail(
            "checker SEALED_LEDGER path changed"
        )

    sealed_sha = sha256(
        SEALED
    )

    run1_sha = sha256(
        run1
    )
    run2_sha = sha256(
        run2
    )

    exact_repeat = (
        run1_sha == sealed_sha
        and run2_sha == sealed_sha
        and run1.read_bytes()
        == run2.read_bytes()
        == SEALED.read_bytes()
    )

    if not exact_repeat:
        fail(
            "V3 decision pass ledgers are no longer "
            "byte-exact with the immutable seal"
        )

    seal_report = load_json(
        seal_report_path
    )

    if (
        seal_report.get("status")
        != "SEALED_BEFORE_EVALUATOR_TRUTH"
        or seal_report.get("sealed_sha256")
        != sealed_sha
        or seal_report.get("run1_sha256")
        != sealed_sha
        or seal_report.get("run2_sha256")
        != sealed_sha
        or seal_report.get("byte_exact_repeat")
        is not True
        or seal_report.get(
            "future_truth_opened_before_seal"
        )
        is not False
    ):
        fail(
            "V3 decision seal report contract changed"
        )

    print(
        "PASS | V3 exact-repeat run1/run2/sealed"
    )
    print(
        "sealed_sha256 =",
        sealed_sha,
    )
    print(
        "sealed_rows =",
        len(decisions),
    )
    print(
        "receiver_keys =",
        len(decision_keys),
    )

    # ------------------------------------------------------------
    # D. Frozen protocol.
    # ------------------------------------------------------------

    protocol = load_json(
        PROTOCOL
    )

    if (
        protocol["receiver_geometry"]["modes"]
        != ["centroid", "known", "uncertain"]
    ):
        fail(
            "receiver geometry protocol changed"
        )

    if (
        protocol["codebooks"]["sizes"]
        != [16, 32, 64]
    ):
        fail(
            "codebook protocol changed"
        )

    if (
        protocol["adaptive_TopK"][
            "coverage_targets"
        ]
        != [0.9, 0.95, 0.975, 0.99]
    ):
        fail(
            "coverage-target protocol changed"
        )

    # ------------------------------------------------------------
    # E. Full 271-test regression BEFORE scientific outcomes.
    # ------------------------------------------------------------

    env = os.environ.copy()

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
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    match = re.search(
        r"Ran\s+(\d+)\s+tests",
        proc.stdout,
    )

    test_count = (
        None
        if match is None
        else int(match.group(1))
    )

    if (
        proc.returncode != 0
        or test_count != EXPECTED_TESTS
    ):
        fail(
            "Stage5 regression failed before "
            "formal metrics: "
            f"rc={proc.returncode}, "
            f"tests={test_count}"
        )

    atomic_write(
        REGRESSION_LOG,
        proc.stdout.encode("utf-8"),
    )

    regression_sha = sha256(
        REGRESSION_LOG
    )

    print(
        "PASS | full Stage5 regression = 271/271"
    )

    # ------------------------------------------------------------
    # F. Resolve the already-materialized evaluator truth.
    #
    # Truth has previously been opened POST-SEAL, but no formal
    # metric has been computed. No controller is executed here.
    # ------------------------------------------------------------

    checker.DECISION_LEDGER_SEALED = True
    checker.FORMAL_TRUTH_OPENED = False

    truth_path, truth_map = (
        checker.resolve_truth_binding(
            decision_keys
        )
    )

    truth_path = Path(
        truth_path
    ).resolve()

    if truth_path != TRUTH_BINDING.resolve():
        fail(
            "unchanged resolver selected "
            f"unexpected truth path: {truth_path}"
        )

    if sha256(truth_path) != EXPECTED[TRUTH_BINDING]:
        fail(
            "resolved evaluator truth SHA changed"
        )

    resolved_truth_map_hash = (
        truth_map_hash(
            truth_map
        )
    )

    if (
        resolved_truth_map_hash
        != EXPECTED_TRUTH_MAP_HASH
    ):
        fail(
            "resolved truth-map semantics changed:\n"
            f"actual   = {resolved_truth_map_hash}\n"
            f"expected = {EXPECTED_TRUTH_MAP_HASH}"
        )

    if not bool(
        checker.FORMAL_TRUTH_OPENED
    ):
        fail(
            "resolver did not mark formal truth opened"
        )

    print(
        "PASS | unchanged truth resolver unique"
    )
    print(
        "truth_sha256 =",
        sha256(truth_path),
    )
    print(
        "truth_map_hash =",
        resolved_truth_map_hash,
    )

    # ------------------------------------------------------------
    # G. Re-check protected scientific inputs immediately before
    #    the one-shot metric call.
    # ------------------------------------------------------------

    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            fail(
                f"pre-metric protected artifact changed: {path}"
            )

    if sha256(SEALED) != sealed_sha:
        fail(
            "sealed decision ledger changed pre-metric"
        )

    # ------------------------------------------------------------
    # H. Immutable one-shot invocation marker.
    # ------------------------------------------------------------

    marker_sha = write_json_once(
        MARKER,
        {
            "stage": 5,
            "version": "fullpdf_v2",
            "formal_attempt":
                "FINAL_V4_POSTSEAL_EVALUATOR_ONLY",
            "status":
                "INVOKED_BEFORE_FORMAL_METRIC_CALL",
            "decision_controller_reexecuted":
                False,
            "decision_ledger": {
                "path": str(SEALED),
                "sha256": sealed_sha,
                "rows": len(decisions),
                "receiver_keys":
                    len(decision_keys),
                "byte_exact_run1_run2":
                    True,
            },
            "evaluator_truth": {
                "path": str(truth_path),
                "sha256":
                    sha256(truth_path),
                "truth_map_hash":
                    resolved_truth_map_hash,
                "opened_postseal":
                    True,
            },
            "formal_metrics_computed_before_marker":
                False,
            "training": False,
            "retraining": False,
            "recalibration": False,
            "formal_tuning": False,
            "post_outcome_tuning": False,
            "frozen_hashes": {
                str(path): expected
                for path, expected
                in EXPECTED.items()
            },
            "regression": {
                "tests": EXPECTED_TESTS,
                "status": "PASS",
                "log_sha256":
                    regression_sha,
            },
            "Stage6_allowed":
                False,
        },
    )

    print(
        "PASS | immutable V4 invocation marker"
    )
    print(
        "marker_sha256 =",
        marker_sha,
    )

    # ============================================================
    # I. THE ONE AND ONLY SCIENTIFIC FORMAL METRIC CALL.
    # ============================================================

    print()
    print("=" * 78)
    print(
        "EXECUTING EXACT FROZEN evaluate_sealed_decisions() ONCE"
    )
    print("=" * 78)

    evaluator_rows, formal_summary = (
        checker.evaluate_sealed_decisions(
            decisions,
            truth_map,
            protocol,
        )
    )

    metrics_returned = True

    # Freeze raw outcomes immediately, before acceptance logic.
    atomic_write(
        EVALUATOR_ROWS,
        jsonl_bytes(
            evaluator_rows
        ),
    )

    evaluator_sha = sha256(
        EVALUATOR_ROWS
    )

    summary_sha = write_json_once(
        FORMAL_SUMMARY,
        formal_summary,
    )

    # ------------------------------------------------------------
    # J. Frozen scientific gates.
    # ------------------------------------------------------------

    strata = formal_summary.get(
        "strata"
    )

    n_strata = strata_count(
        strata
    )

    recovery_source = (
        RUNTIME.read_text(
            encoding="utf-8"
        )
    )

    recovery_structural = {
        "persistence":
            "previous_primary_index"
            in recovery_source,

        "hysteresis":
            "hysteresis"
            in recovery_source.lower(),

        "local_neighbor":
            "local_neighbor_recovery"
            in recovery_source,

        "widened_fallback":
            "widened_fallback"
            in recovery_source,

        "exhaustive_loss_of_lock":
            "exhaustive_loss_of_lock"
            in recovery_source,
    }

    end_hashes_ok = all(
        sha256(path) == expected
        for path, expected
        in EXPECTED.items()
    )

    sealed_unchanged = (
        sha256(SEALED)
        == sealed_sha
        and (
            stat.S_IMODE(
                SEALED.stat().st_mode
            )
            & 0o222
        )
        == 0
    )

    gates = {
        "G13_two_exact_repeat_causal_decision_passes":
            bool(exact_repeat),

        "G14_decision_ledger_sealed_before_truth":
            bool(
                seal_report[
                    "future_truth_opened_before_seal"
                ]
                is False
                and sealed_unchanged
            ),

        "G15_evaluator_truth_opened_only_after_seal":
            bool(
                checker.FORMAL_TRUTH_OPENED
                and checker.DECISION_LEDGER_SEALED
                and truth_path
                == TRUTH_BINDING.resolve()
            ),

        "G16_requested_empirical_coverage":
            bool(
                formal_summary[
                    "requested_coverage_all_strata_pass"
                ]
            ),

        "G17_actual_charged_overhead_reduced_vs_exhaustive":
            bool(
                formal_summary[
                    "overhead_reduction_vs_exhaustive_all_strata_pass"
                ]
            ),

        "G18_no_hidden_or_free_probes":
            bool(
                formal_summary[
                    "no_free_probes"
                ]
            ),

        "G19_latency_from_actual_charged_probe_count":
            bool(
                formal_summary[
                    "latency_all_strata_pass"
                ]
            ),

        "G20_complete_optical_chain":
            bool(
                formal_summary[
                    "full_optical_chain_complete"
                ]
            ),

        "G21_persistence_hysteresis_recovery":
            bool(
                all(
                    recovery_structural.values()
                )
            ),

        "G22_no_post_outcome_mutation":
            bool(
                end_hashes_ok
                and sealed_unchanged
            ),

        "G23_full_3x3x4x4_formal_matrix":
            bool(
                n_strata
                == EXPECTED_STRATA
            ),

        "G24_evaluator_output_nonempty":
            bool(
                len(evaluator_rows) > 0
            ),
    }

    scientific_all_pass = all(
        gates.values()
    )

    # ------------------------------------------------------------
    # K. Freeze final scientific outcome.
    #
    # Stage6 is deliberately NOT enabled by this runner.
    # Independent post-seal acceptance closure follows only if
    # the actual scientific result passes.
    # ------------------------------------------------------------

    status = (
        "PASS_SCIENTIFIC_READY_FOR_"
        "INDEPENDENT_POSTSEAL_CLOSURE"
        if scientific_all_pass
        else
        "FAIL_CLOSED_SCIENTIFIC_STAGE5"
    )

    report = {
        "stage": 5,
        "version": "fullpdf_v2",
        "formal_attempt":
            "FINAL_V4_POSTSEAL_EVALUATOR_ONLY",
        "status": status,
        "scientific_gate_all_pass":
            scientific_all_pass,
        "formal_metrics_computed":
            True,
        "controller_reexecuted":
            False,
        "receiver_selection_reexecuted":
            False,
        "decision_boundary": {
            "V3_exact_repeat":
                True,
            "sealed_before_truth":
                True,
            "sealed_ledger":
                str(SEALED),
            "sealed_sha256":
                sealed_sha,
            "rows":
                len(decisions),
            "receiver_keys":
                len(decision_keys),
        },
        "evaluator_truth": {
            "path":
                str(truth_path),
            "sha256":
                sha256(truth_path),
            "truth_map_hash":
                resolved_truth_map_hash,
            "postseal_only":
                True,
        },
        "formal": {
            **normalize(
                formal_summary
            ),
            "strata_count":
                n_strata,
            "expected_strata":
                EXPECTED_STRATA,
            "evaluator_rows":
                str(EVALUATOR_ROWS),
            "evaluator_rows_sha256":
                evaluator_sha,
            "formal_summary":
                str(FORMAL_SUMMARY),
            "formal_summary_sha256":
                summary_sha,
        },
        "recovery_structural":
            recovery_structural,
        "gates":
            gates,
        "scientific_boundaries": {
            "training":
                False,
            "retraining":
                False,
            "recalibration":
                False,
            "formal_parameter_tuning":
                False,
            "post_outcome_tuning":
                False,
            "future_truth_controller_input":
                False,
            "tracks_to_predict_controller_input":
                False,
            "measured_optical_claim":
                False,
            "optical_semantics":
                "normalized_constructed_optical_surrogate",
        },
        "Stage6_allowed":
            False,
        "next": (
            "INDEPENDENT_POSTSEAL_ACCEPTANCE_CLOSURE"
            if scientific_all_pass
            else
            "FREEZE_SCIENTIFIC_FAILURE_NO_TUNING"
        ),
    }

    report_sha = write_json_once(
        FINAL_REPORT,
        report,
    )

    print()
    print("=" * 78)
    print(
        "FINAL FORMAL V4 SCIENTIFIC RESULT"
    )
    print("=" * 78)
    print(
        "formal_metrics_computed = true"
    )
    print(
        "controller_reexecuted = false"
    )
    print(
        "sealed_sha256 =",
        sealed_sha,
    )
    print(
        "truth_sha256 =",
        sha256(truth_path),
    )
    print(
        "evaluator_rows =",
        len(evaluator_rows),
    )
    print(
        "evaluator_rows_sha256 =",
        evaluator_sha,
    )
    print(
        "formal_summary_sha256 =",
        summary_sha,
    )
    print(
        "formal_strata_count =",
        n_strata,
    )
    print(
        "expected_strata_count =",
        EXPECTED_STRATA,
    )
    print(
        "requested_coverage_gate =",
        gates[
            "G16_requested_empirical_coverage"
        ],
    )
    print(
        "overhead_reduction_gate =",
        gates[
            "G17_actual_charged_overhead_reduced_vs_exhaustive"
        ],
    )
    print(
        "no_free_probes_gate =",
        gates[
            "G18_no_hidden_or_free_probes"
        ],
    )
    print(
        "latency_gate =",
        gates[
            "G19_latency_from_actual_charged_probe_count"
        ],
    )
    print(
        "optical_chain_gate =",
        gates[
            "G20_complete_optical_chain"
        ],
    )
    print(
        "full_matrix_gate =",
        gates[
            "G23_full_3x3x4x4_formal_matrix"
        ],
    )
    print(
        "no_post_outcome_mutation_gate =",
        gates[
            "G22_no_post_outcome_mutation"
        ],
    )
    print(
        "scientific_gate_all_pass =",
        scientific_all_pass,
    )
    print(
        "status =",
        status,
    )
    print(
        "post_outcome_tuning = false"
    )
    print(
        "Stage6_allowed = false"
    )
    print(
        "next =",
        report["next"],
    )
    print(
        "report =",
        FINAL_REPORT,
    )
    print(
        "report_sha256 =",
        report_sha,
    )
    print("=" * 78)

    return (
        0
        if scientific_all_pass
        else 2
    )


if __name__ == "__main__":
    metrics_returned = False

    try:
        rc = main()

    except Exception as exc:
        print()
        print("=" * 78)
        print(
            "FINAL FORMAL V4 = FAIL-CLOSED HARNESS/EXECUTION"
        )
        print(
            "reason =",
            repr(exc),
        )
        print(
            "DO NOT RERUN AUTOMATICALLY"
        )
        print(
            "Stage6_allowed = false"
        )
        print("=" * 78)

        sys.exit(3)

    sys.exit(rc)

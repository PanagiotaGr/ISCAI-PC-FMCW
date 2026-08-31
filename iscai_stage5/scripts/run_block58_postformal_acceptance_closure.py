from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/agni/waymo")
STAGE5 = ROOT / "iscai_stage5"

FORMAL_REPORT = (
    STAGE5
    / "reports/block58_formal_evaluation.json"
)

PREFREEZE_REPORT = (
    STAGE5
    / "reports/block58_prefreeze_acceptance.json"
)

PRIOR_GATE = (
    STAGE5
    / "reports/block58_part2_formal_gate.json"
)

POLICY = (
    STAGE5
    / "configs/formal_stage5_acceptance_policy.json"
)

FORMAL_MANIFEST = (
    ROOT
    / "iscai_stage3/artifacts/block38e/formal_validation_120.jsonl"
)

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

ROUTE_AUDIT = (
    STAGE5
    / "artifacts/block58/formal_input_route_audit.json"
)

FORMAL_RECORDS = (
    STAGE5
    / "artifacts/block58/formal_clean_records.jsonl"
)

FORMAL_CACHE = (
    STAGE5
    / "artifacts/block58/formal_clean_cache"
)

CLOSURE = (
    STAGE5
    / "artifacts/block58/block58_final_closure.json"
)

EXPECTED_CODEBOOKS = [16, 32, 64]
EXPECTED_HORIZONS = [0.1, 0.3, 0.5, 1.0]
EXPECTED_Q_LEVELS = [0.9, 0.95, 0.975, 0.99]
EXPECTED_GEOMETRIES = ["centroid", "known", "uncertain"]

NOMINAL_Q = 0.95
PRIMARY_GEOMETRY = "uncertain"


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def file_sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(
        path.is_file(),
        f"Required JSON missing: {path}",
    )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse JSON {path}: {exc}"
        ) from exc

    require(
        isinstance(value, dict),
        f"JSON root must be object: {path}",
    )

    return value


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(
    path: Path,
    data: bytes,
) -> str:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = path.read_bytes()

        if existing == data:
            return "ALREADY_IDENTICAL"

        fail(
            "Existing frozen closure differs from the "
            "new deterministic closure. Refusing overwrite: "
            f"{path}"
        )

    temporary = path.with_name(
        path.name + f".tmp.{os.getpid()}"
    )

    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    return "CREATED"


def get_nested(
    value: dict[str, Any],
    *keys: str,
) -> Any:
    current: Any = value

    for key in keys:
        require(
            isinstance(current, dict)
            and key in current,
            "Missing required field: "
            + ".".join(keys),
        )

        current = current[key]

    return current


def same_float(
    a: Any,
    b: float,
    tol: float = 1e-12,
) -> bool:
    try:
        x = float(a)
    except Exception:
        return False

    return (
        math.isfinite(x)
        and abs(x - b) <= tol
    )


def cache_manifest() -> tuple[str, list[dict[str, Any]]]:
    require(
        FORMAL_CACHE.is_dir(),
        f"Formal cache directory missing: {FORMAL_CACHE}",
    )

    files = sorted(
        FORMAL_CACHE.glob("*.json")
    )

    require(
        len(files) == 120,
        "Formal clean cache must contain exactly "
        f"120 JSON files, found {len(files)}.",
    )

    entries: list[dict[str, Any]] = []

    combined = sha256()

    for index, path in enumerate(
        files,
        start=1,
    ):
        expected_prefix = f"{index:03d}_"

        require(
            path.name.startswith(expected_prefix),
            "Unexpected formal cache ordering/name: "
            f"expected prefix {expected_prefix}, got {path.name}",
        )

        digest = file_sha256(path)
        size = path.stat().st_size

        entry = {
            "index": index,
            "name": path.name,
            "size_bytes": size,
            "sha256": digest,
        }

        entries.append(entry)

        combined.update(
            path.name.encode("utf-8")
        )
        combined.update(b"\0")
        combined.update(
            digest.encode("ascii")
        )
        combined.update(b"\n")

    return (
        combined.hexdigest(),
        entries,
    )


def main() -> None:
    print("=" * 72)
    print("BLOCK 5.8 — POST-FORMAL ACCEPTANCE CLOSURE")
    print("=" * 72)

    # --------------------------------------------------------
    # Frozen pre-formal acceptance contract
    # --------------------------------------------------------
    prefreeze = load_json(
        PREFREEZE_REPORT
    )

    require(
        prefreeze.get("status")
        == "PASS_PREFORMAL_FREEZE",
        "Pre-formal acceptance freeze is not PASS.",
    )

    require(
        prefreeze.get(
            "formal_outcome_metrics_read"
        ) is False,
        "Pre-formal freeze claims formal outcomes were read.",
    )

    require(
        prefreeze.get(
            "formal_stage5_metrics_computed"
        ) is False,
        "Pre-formal freeze claims Stage5 metrics "
        "were already computed.",
    )

    require(
        prefreeze.get(
            "formal_results_used_for_parameter_selection"
        ) is False,
        "Formal results were used for parameter selection.",
    )

    require(
        prefreeze.get("post_hoc_tuning")
        is False,
        "Pre-formal contract indicates post-hoc tuning.",
    )

    # --------------------------------------------------------
    # Frozen formal population
    # --------------------------------------------------------
    require(
        FORMAL_MANIFEST.is_file(),
        "Frozen formal N=120 manifest is missing.",
    )

    manifest_sha = file_sha256(
        FORMAL_MANIFEST
    )

    require(
        manifest_sha
        == EXPECTED_FORMAL_MANIFEST_SHA,
        "Frozen formal N=120 manifest SHA changed.",
    )

    prefreeze_manifest = get_nested(
        prefreeze,
        "formal_manifest",
    )

    require(
        prefreeze_manifest.get("sha256")
        == manifest_sha,
        "Prefreeze manifest SHA disagrees with current "
        "frozen formal manifest.",
    )

    require(
        prefreeze_manifest.get(
            "scenario_count_contract"
        )
        == 120,
        "Prefreeze N contract is not 120.",
    )

    # --------------------------------------------------------
    # Frozen acceptance policy
    # --------------------------------------------------------
    require(
        POLICY.is_file(),
        "Frozen Stage5 acceptance policy is missing.",
    )

    policy_sha = file_sha256(
        POLICY
    )

    prefreeze_policy = get_nested(
        prefreeze,
        "acceptance_policy",
    )

    require(
        prefreeze_policy.get("sha256")
        == policy_sha,
        "Acceptance policy SHA changed after "
        "pre-formal freeze.",
    )

    require(
        same_float(
            prefreeze_policy.get("alpha"),
            0.05,
        ),
        "Frozen acceptance alpha is not 0.05.",
    )

    require(
        same_float(
            prefreeze_policy.get("nominal_q"),
            NOMINAL_Q,
        ),
        "Frozen nominal q is not 0.95.",
    )

    require(
        [
            float(v)
            for v in prefreeze_policy.get(
                "reported_q",
                [],
            )
        ]
        == EXPECTED_Q_LEVELS,
        "Frozen reported q levels changed.",
    )

    # --------------------------------------------------------
    # Successful current formal evaluation
    # --------------------------------------------------------
    formal = load_json(
        FORMAL_REPORT
    )

    require(
        formal.get("status") == "PASS",
        "Current Block5.8 formal report is not PASS.",
    )

    formal_population = get_nested(
        formal,
        "formal_population",
    )

    require(
        formal_population.get("N") == 120,
        "Current formal report population is not N=120.",
    )

    require(
        formal_population.get(
            "same_immutable_Stage3_4_population"
        )
        is True,
        "Current formal report does not confirm "
        "immutable Stage3/4 population continuity.",
    )

    require(
        formal_population.get(
            "manifest_sha256"
        )
        == manifest_sha,
        "Current formal report manifest SHA mismatch.",
    )

    scope = get_nested(
        formal,
        "scope",
    )

    require(
        scope.get(
            "formal_parameter_tuning"
        )
        is False,
        "Formal report indicates formal parameter tuning.",
    )

    frozen = get_nested(
        formal,
        "frozen_contract",
    )

    require(
        frozen.get("codebooks")
        == EXPECTED_CODEBOOKS,
        "Formal codebooks are not exactly 16/32/64.",
    )

    require(
        [
            float(v)
            for v in frozen.get(
                "q_levels",
                [],
            )
        ]
        == EXPECTED_Q_LEVELS,
        "Formal q levels differ from frozen contract.",
    )

    require(
        frozen.get(
            "receiver_geometry_modes"
        )
        == EXPECTED_GEOMETRIES,
        "Formal receiver geometry modes differ "
        "from frozen contract.",
    )

    require(
        frozen.get(
            "primary_acceptance_geometry"
        )
        == PRIMARY_GEOMETRY,
        "Primary formal acceptance geometry "
        "is not uncertain.",
    )

    require(
        frozen.get(
            "receiver_policy"
        )
        == "nearest_causal_vehicle_ahead",
        "Formal receiver-selection policy changed.",
    )

    require(
        frozen.get(
            "tracks_to_predict_receiver_selector"
        )
        is False,
        "tracks_to_predict was used as receiver selector.",
    )

    require(
        frozen.get(
            "future_truth_receiver_selector"
        )
        is False,
        "Future truth was used as receiver selector.",
    )

    require(
        frozen.get(
            "future_truth_role"
        )
        == "evaluator_only_after_controller_decision",
        "Future truth role violates causal evaluation contract.",
    )

    require(
        frozen.get(
            "probability_mass_sum_contract"
        )
        == "sum_Pb_equals_1",
        "Beam-probability mass contract changed.",
    )

    provenance = get_nested(
        formal,
        "provenance",
    )

    require(
        provenance.get(
            "acceptance_policy_sha256"
        )
        == policy_sha,
        "Formal run used a different acceptance policy SHA.",
    )

    # --------------------------------------------------------
    # Acceptance gate
    # --------------------------------------------------------
    acceptance = get_nested(
        formal,
        "acceptance",
    )

    require(
        acceptance.get(
            "primary_geometry"
        )
        == PRIMARY_GEOMETRY,
        "Acceptance primary geometry is not uncertain.",
    )

    require(
        acceptance.get(
            "coverage_pass"
        )
        is True,
        "Stage5 coverage acceptance failed.",
    )

    require(
        acceptance.get(
            "overhead_pass"
        )
        is True,
        "Stage5 overhead acceptance failed.",
    )

    require(
        acceptance.get(
            "stage5_acceptance_pass"
        )
        is True,
        "Stage5 primary acceptance gate failed.",
    )

    coverage_tests = acceptance.get(
        "coverage_tests"
    )

    require(
        isinstance(
            coverage_tests,
            list,
        ),
        "coverage_tests must be a list.",
    )

    require(
        len(coverage_tests) == 12,
        "Expected exactly 12 nominal-q coverage tests "
        "(3 codebooks x 4 horizons).",
    )

    observed_pairs: set[tuple[int, float]] = set()

    for item in coverage_tests:
        require(
            isinstance(item, dict),
            "Invalid coverage test entry.",
        )

        codebook = int(
            item.get("codebook")
        )

        horizon = float(
            item.get("horizon")
        )

        decision = item.get(
            "decision"
        )

        require(
            isinstance(decision, dict),
            "Coverage test decision missing.",
        )

        require(
            decision.get("passed")
            is True,
            "At least one nominal-q coverage test failed.",
        )

        require(
            same_float(
                decision.get(
                    "requested_q"
                ),
                NOMINAL_Q,
            ),
            "Coverage test requested_q is not frozen 0.95.",
        )

        observed_pairs.add(
            (
                codebook,
                horizon,
            )
        )

    expected_pairs = {
        (
            codebook,
            horizon,
        )
        for codebook in EXPECTED_CODEBOOKS
        for horizon in EXPECTED_HORIZONS
    }

    require(
        observed_pairs == expected_pairs,
        "Coverage tests do not span exactly "
        "16/32/64 x 0.1/0.3/0.5/1.0.",
    )

    overhead_tests = acceptance.get(
        "overhead_tests"
    )

    require(
        isinstance(
            overhead_tests,
            list,
        ),
        "overhead_tests must be a list.",
    )

    require(
        len(overhead_tests) == 3,
        "Expected exactly one overhead acceptance "
        "test per codebook.",
    )

    require(
        sorted(
            int(item.get("codebook"))
            for item in overhead_tests
        )
        == EXPECTED_CODEBOOKS,
        "Overhead tests do not cover 16/32/64.",
    )

    require(
        all(
            item.get("passed")
            is True
            for item in overhead_tests
        ),
        "At least one codebook overhead test failed.",
    )

    require(
        all(
            item.get(
                "adaptive_less_than_exhaustive"
            )
            is True
            for item in overhead_tests
        ),
        "Adaptive probing does not reduce overhead "
        "relative to exhaustive for every codebook.",
    )

    # --------------------------------------------------------
    # Receiver availability accounting
    # --------------------------------------------------------
    availability = get_nested(
        formal,
        "receiver_availability",
    )

    eligible = int(
        availability.get(
            "scenarios_with_eligible_receiver"
        )
    )

    unavailable_scenarios = int(
        availability.get(
            "scenarios_without_eligible_receiver"
        )
    )

    pred_available = int(
        availability.get(
            "selected_receiver_prediction_available"
        )
    )

    pred_unavailable = int(
        availability.get(
            "selected_receiver_prediction_unavailable"
        )
    )

    require(
        eligible + unavailable_scenarios
        == 120,
        "Receiver availability does not account "
        "for all 120 formal scenarios.",
    )

    require(
        pred_available + pred_unavailable
        == eligible,
        "Prediction availability does not account "
        "for all eligible receiver scenes.",
    )

    endpoints = availability.get(
        "valid_future_receiver_endpoints"
    )

    require(
        isinstance(endpoints, dict),
        "valid_future_receiver_endpoints missing.",
    )

    for horizon in EXPECTED_HORIZONS:
        key = str(horizon)

        require(
            key in endpoints,
            f"Missing valid endpoint count for horizon {horizon}.",
        )

        require(
            int(endpoints[key]) > 0,
            f"No valid receiver endpoints at horizon {horizon}.",
        )

    # --------------------------------------------------------
    # Formal merged records + scenario cache integrity
    # --------------------------------------------------------
    require(
        FORMAL_RECORDS.is_file(),
        "Merged formal records JSONL is missing.",
    )

    records_sha = file_sha256(
        FORMAL_RECORDS
    )

    require(
        records_sha
        == provenance.get(
            "merged_records_sha256"
        ),
        "Merged formal records SHA disagrees "
        "with formal report provenance.",
    )

    cache_sha, cache_entries = (
        cache_manifest()
    )

    # --------------------------------------------------------
    # Route audit provenance
    # --------------------------------------------------------
    require(
        ROUTE_AUDIT.is_file(),
        "Formal input route audit is missing.",
    )

    route_audit_sha = file_sha256(
        ROUTE_AUDIT
    )

    # --------------------------------------------------------
    # Prior BLOCKED gate handling.
    #
    # Never delete it. It is retained as provenance from a
    # failed earlier attempt. We only accept it as superseded
    # when it predates the successful current formal report.
    # --------------------------------------------------------
    prior_gate_summary: dict[str, Any] | None = None

    if PRIOR_GATE.is_file():
        prior_gate = load_json(
            PRIOR_GATE
        )

        gate_mtime_ns = (
            PRIOR_GATE.stat().st_mtime_ns
        )

        formal_mtime_ns = (
            FORMAL_REPORT.stat().st_mtime_ns
        )

        if prior_gate.get("status") == "BLOCKED":
            require(
                gate_mtime_ns
                < formal_mtime_ns,
                "A BLOCKED Part2 gate exists and is not "
                "older than the successful formal report. "
                "Refusing to freeze ambiguous state.",
            )

        prior_gate_summary = {
            "path":
                str(PRIOR_GATE),

            "sha256":
                file_sha256(
                    PRIOR_GATE
                ),

            "status":
                prior_gate.get(
                    "status"
                ),

            "reason":
                prior_gate.get(
                    "reason"
                ),

            "mtime_ns":
                gate_mtime_ns,

            "successful_formal_report_mtime_ns":
                formal_mtime_ns,

            "role":
                (
                    "retained_prior_attempt_artifact_"
                    "noncanonical_for_current_successful_run"
                ),
        }

    # --------------------------------------------------------
    # Useful formal summary
    # --------------------------------------------------------
    nominal = get_nested(
        formal,
        "nominal_primary_summary",
    )

    formal_sha = file_sha256(
        FORMAL_REPORT
    )

    prefreeze_sha = file_sha256(
        PREFREEZE_REPORT
    )

    closure = {
        "stage":
            5,

        "block":
            "5.8",

        "phase":
            "post_formal_acceptance_closure",

        "status":
            "PASS_FROZEN",

        "stage5_overall_closed":
            False,

        "block59_reproducibility_completed":
            False,

        "block510_final_stage5_closure_completed":
            False,

        "scientific_contract": {
            "formal_population_N":
                120,

            "same_immutable_Stage3_4_population":
                True,

            "receiver_policy":
                "nearest_causal_vehicle_ahead",

            "receiver_geometry_modes":
                EXPECTED_GEOMETRIES,

            "primary_acceptance_geometry":
                PRIMARY_GEOMETRY,

            "codebooks":
                EXPECTED_CODEBOOKS,

            "horizons_s":
                EXPECTED_HORIZONS,

            "reported_probability_mass_targets":
                EXPECTED_Q_LEVELS,

            "nominal_acceptance_q":
                NOMINAL_Q,

            "future_truth":
                "evaluator_only_after_controller_decision",

            "tracks_to_predict_receiver_selector":
                False,

            "future_truth_receiver_selector":
                False,

            "formal_parameter_tuning":
                False,

            "post_hoc_threshold_or_codebook_tuning":
                False,
        },

        "acceptance": {
            "coverage":
                "PASS",

            "overhead_reduction":
                "PASS",

            "stage5_block58_acceptance":
                "PASS",

            "coverage_test_count":
                len(
                    coverage_tests
                ),

            "overhead_test_count":
                len(
                    overhead_tests
                ),

            "nominal_primary_probability_coverage":
                nominal.get(
                    "probability_coverage"
                ),

            "nominal_primary_probing_overhead_fraction":
                nominal.get(
                    "probing_overhead_fraction"
                ),

            "nominal_primary_overhead_reduction_vs_exhaustive":
                nominal.get(
                    "overhead_reduction_vs_exhaustive"
                ),
        },

        "receiver_availability": {
            "formal_scenarios":
                120,

            "eligible_receiver_scenarios":
                eligible,

            "no_eligible_receiver_scenarios":
                unavailable_scenarios,

            "selected_receiver_prediction_available":
                pred_available,

            "selected_receiver_prediction_unavailable":
                pred_unavailable,

            "valid_future_receiver_endpoints":
                endpoints,
        },

        "provenance": {
            "formal_manifest": {
                "path":
                    str(
                        FORMAL_MANIFEST
                    ),

                "sha256":
                    manifest_sha,
            },

            "acceptance_policy": {
                "path":
                    str(
                        POLICY
                    ),

                "sha256":
                    policy_sha,
            },

            "prefreeze_report": {
                "path":
                    str(
                        PREFREEZE_REPORT
                    ),

                "sha256":
                    prefreeze_sha,
            },

            "formal_report": {
                "path":
                    str(
                        FORMAL_REPORT
                    ),

                "sha256":
                    formal_sha,
            },

            "formal_merged_records": {
                "path":
                    str(
                        FORMAL_RECORDS
                    ),

                "sha256":
                    records_sha,
            },

            "formal_cache": {
                "path":
                    str(
                        FORMAL_CACHE
                    ),

                "scenario_file_count":
                    len(
                        cache_entries
                    ),

                "combined_manifest_sha256":
                    cache_sha,
            },

            "formal_input_route_audit": {
                "path":
                    str(
                        ROUTE_AUDIT
                    ),

                "sha256":
                    route_audit_sha,
            },

            "prior_part2_gate_artifact":
                prior_gate_summary,
        },

        "next":
            "Block 5.9 exact reproducibility freeze",
    }

    encoded = canonical_bytes(
        closure
    )

    write_status = atomic_write_once(
        CLOSURE,
        encoded,
    )

    closure_sha = file_sha256(
        CLOSURE
    )

    print(
        "formal population        = 120 / SHA PASS"
    )
    print(
        "pre-formal policy        = FROZEN / SHA PASS"
    )
    print(
        "formal report            = PASS"
    )
    print(
        "coverage acceptance      = PASS"
    )
    print(
        "overhead acceptance      = PASS"
    )
    print(
        "receiver accounting      = PASS"
    )
    print(
        "causal/leakage contract  = PASS"
    )
    print(
        "codebooks                = 16 / 32 / 64"
    )
    print(
        "geometry modes           = centroid / known / uncertain"
    )
    print(
        "q levels                 = 0.90 / 0.95 / 0.975 / 0.99"
    )
    print(
        "formal cache             = 120 / integrity PASS"
    )
    print(
        "merged records SHA       = PASS"
    )

    if prior_gate_summary is not None:
        print(
            "prior BLOCKED artifact   = retained as provenance"
        )

    print(
        "closure write            =",
        write_status,
    )
    print(
        "closure SHA256           =",
        closure_sha,
    )
    print(
        "BLOCK 5.8 STATUS         = PASS / FROZEN"
    )
    print(
        "NEXT                     = Block 5.9"
    )
    print(
        "closure                  =",
        CLOSURE,
    )


if __name__ == "__main__":
    main()

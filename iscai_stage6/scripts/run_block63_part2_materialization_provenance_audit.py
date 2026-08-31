from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import traceback


ROOT = Path("/home/agni/waymo")

S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"

PART1 = (
    S6
    / "reports/block63_part1_deterministic_future_box.json"
)

BATCH_AUDIT = (
    S6
    / "reports/block63_part2_batch_row_binding_audit.json"
)

RUNTIME_AUDIT = (
    S6
    / "reports/block63_part2_runtime_route_binding_audit.json"
)

BLOCK52 = (
    S5
    / "artifacts/block52"
)

MATERIALIZATION_V1 = (
    BLOCK52
    / "dev_posterior_materialization.log"
)

MATERIALIZATION_V2 = (
    BLOCK52
    / "dev_posterior_materialization_v2.log"
)

RESOLVER_AUDIT_LOG = (
    BLOCK52
    / "dev_posterior_resolver_audit.log"
)

RESOLVER_SUMMARY_LOG = (
    BLOCK52
    / "dev_posterior_resolver_summary.log"
)

ROUTE_LOG = (
    BLOCK52
    / "actual_stage4_gaussian_route_audit.log"
)

API_LOG = (
    BLOCK52
    / "calibrated_runtime_api_audit.log"
)

DEV_JSON = (
    BLOCK52
    / "development_gaussian_posterior.json"
)

ROUTE_JSON = (
    BLOCK52
    / "actual_stage4_gaussian_route_audit.json"
)

RESOLVER_JSON = (
    BLOCK52
    / "dev_posterior_resolver_summary.json"
)

REPORT = (
    S6
    / "reports/block63_part2_materialization_provenance_audit.json"
)

EXCERPT = (
    S6
    / "artifacts/block63/"
      "block63_part2_materialization_provenance_excerpt.txt"
)


EXPECTED = {
    "part1":
        (
            "871c8a81e43048ff2ccf7ef1844ad543"
            "3a859ff15b944749e13be9160b08256a"
        ),

    "batch_audit":
        (
            "a57d848cda03f002e4b16de96314372b"
            "f3d985b15568a5b9c06528e035a9311b"
        ),

    "runtime_audit":
        (
            "00121eebc007b016865fb5b17448fbda"
            "591dd3d00b1606b24fdb279b11a81e57"
        ),
}


def require(condition, message):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:

    digest = sha256()

    with path.open("rb") as stream:

        while True:

            chunk = stream.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_text(path: Path) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def load_json(path: Path):

    return json.loads(
        read_text(path)
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
        +
        "\n"
    ).encode("utf-8")


def write_json(path: Path, payload):

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_bytes(
        canonical_bytes(payload)
    )

    os.replace(
        tmp,
        path,
    )


def contexts(
    path: Path,
    patterns,
    *,
    radius=5,
    maximum=80,
):

    text = read_text(path)

    lines = text.splitlines()

    regex = re.compile(
        "|".join(
            f"(?:{pattern})"
            for pattern in patterns
        ),
        re.IGNORECASE,
    )

    found = []

    for index, line in enumerate(lines):

        if not regex.search(line):
            continue

        lo = max(
            0,
            index - radius,
        )

        hi = min(
            len(lines),
            index + radius + 1,
        )

        found.append({
            "line":
                index + 1,

            "context":
                "\n".join(
                    f"{j + 1:05d}: {lines[j]}"
                    for j in range(
                        lo,
                        hi,
                    )
                ),
        })

        if len(found) >= maximum:
            break

    return found


def contains_any(text, values):

    lower = text.lower()

    return any(
        value.lower()
        in lower
        for value in values
    )


def contains_all(text, values):

    lower = text.lower()

    return all(
        value.lower()
        in lower
        for value in values
    )


def flattened_strings(value):

    result = []

    if isinstance(value, dict):

        for key, child in value.items():

            result.append(
                str(key)
            )

            result.extend(
                flattened_strings(child)
            )

    elif isinstance(value, list):

        for child in value:

            result.extend(
                flattened_strings(child)
            )

    else:

        result.append(
            str(value)
        )

    return result


def main():

    print(
        "============================================================"
    )
    print(
        "STAGE 6 — BLOCK 6.3 PART 2/2"
    )
    print(
        "BLOCK5.2 MATERIALIZATION PROVENANCE AUDIT"
    )
    print(
        "============================================================"
    )

    protected = (
        PART1,
        BATCH_AUDIT,
        RUNTIME_AUDIT,
        MATERIALIZATION_V1,
        MATERIALIZATION_V2,
        RESOLVER_AUDIT_LOG,
        RESOLVER_SUMMARY_LOG,
        ROUTE_LOG,
        API_LOG,
        DEV_JSON,
        ROUTE_JSON,
        RESOLVER_JSON,
    )

    missing = [
        str(path)
        for path in protected
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing frozen evidence: "
            + ", ".join(missing)
        ),
    )

    before = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Exact Stage6 audit-chain seal
    # ========================================================

    print()
    print(
        "===== A. EXACT BLOCK6.3 AUDIT CHAIN ====="
    )

    checks = (
        (
            "part1",
            PART1,
        ),
        (
            "batch_audit",
            BATCH_AUDIT,
        ),
        (
            "runtime_audit",
            RUNTIME_AUDIT,
        ),
    )

    for name, path in checks:

        actual = sha256_file(path)

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

    runtime_report = load_json(
        RUNTIME_AUDIT
    )

    require(
        runtime_report.get("status")
        ==
        "BLOCKED_RUNTIME_ROUTE_EVIDENCE_INCOMPLETE",
        (
            "Runtime-route audit no longer "
            "has expected blocked status."
        ),
    )

    require(
        runtime_report[
            "readiness"
        ][
            "exact_runtime_definitions"
        ]
        is True,
        (
            "Exact Stage4 runtime definitions "
            "are no longer proven."
        ),
    )

    require(
        runtime_report[
            "readiness"
        ][
            "causal_prediction_identity_retained"
        ]
        is True,
        (
            "Causal prediction identity "
            "is no longer proven."
        ),
    )

    print(
        "exact Stage4 runtime definitions = PASS"
    )

    print(
        "causal prediction identity        = PASS"
    )

    print(
        "only unresolved gate              = BLOCK5.2 PRODUCER ROUTE"
    )

    # ========================================================
    # B. Exact producer-log identities
    # ========================================================

    print()
    print(
        "===== B. BLOCK5.2 PRODUCER LOG IDENTITIES ====="
    )

    producer_logs = (
        MATERIALIZATION_V2,
        MATERIALIZATION_V1,
    )

    support_logs = (
        RESOLVER_AUDIT_LOG,
        RESOLVER_SUMMARY_LOG,
        ROUTE_LOG,
        API_LOG,
    )

    for path in (
        producer_logs
        +
        support_logs
    ):

        print(
            path.name,
            "| bytes =",
            path.stat().st_size,
            "| sha256 =",
            sha256_file(path),
        )

    # ========================================================
    # C. Materialization log evidence only
    # ========================================================

    print()
    print(
        "===== C. MATERIALIZATION PRODUCER EVIDENCE ====="
    )

    producer_text = "\n".join(
        read_text(path)
        for path in producer_logs
    )

    support_text = "\n".join(
        read_text(path)
        for path in support_logs
    )

    combined_text = (
        producer_text
        +
        "\n"
        +
        support_text
    )

    output_gate = (
        contains_any(
            producer_text,
            (
                "development_gaussian_posterior",
                "development posterior",
            ),
        )
        and
        contains_all(
            producer_text,
            (
                "mean_displacement_H0_m",
                "anchor_position_H0_m",
            ),
        )
    )

    calibrated_covariance_output_gate = (
        contains_any(
            producer_text,
            (
                "calibrated_covariance_H0_m2",
                "calibrated covariance",
            ),
        )
    )

    exact_probe_call_gate = (
        "probe_calibrated_output"
        in producer_text
    )

    direct_runtime_gate = (
        "build_gaussian_model"
        in producer_text
        and
        contains_any(
            producer_text,
            (
                "torch.inference_mode",
                "inference_mode",
                "model.eval",
                "model(",
            ),
        )
    )

    runtime_execution_gate = (
        exact_probe_call_gate
        or
        direct_runtime_gate
    )

    checkpoint_gate = contains_any(
        combined_text,
        (
            "gaussian_gru.pt",
            "stage4_gaussian_gru.json",
        ),
    )

    normalization_gate = contains_any(
        combined_text,
        (
            "fit_normalization.json",
            "CalibrationNormalizer",
            "fit-only normalization",
            "fit_only",
        ),
    )

    calibration_gate = contains_any(
        combined_text,
        (
            "stage4_calibration.json",
            "variance_scale",
            "variance_scale_alpha_h",
            "calibrated_covariance_H0_m2",
        ),
    )

    print(
        "posterior output materialization =",
        (
            "PASS"
            if output_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "calibrated covariance output    =",
        (
            "PASS"
            if calibrated_covariance_output_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "probe_calibrated_output call    =",
        (
            "PRESENT"
            if exact_probe_call_gate
            else
            "NOT FOUND"
        ),
    )

    print(
        "direct frozen runtime call      =",
        (
            "PRESENT"
            if direct_runtime_gate
            else
            "NOT FOUND"
        ),
    )

    print(
        "runtime execution route         =",
        (
            "PASS"
            if runtime_execution_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "Gaussian checkpoint/config      =",
        (
            "PASS"
            if checkpoint_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "fit-only normalization          =",
        (
            "PASS"
            if normalization_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "frozen covariance calibration   =",
        (
            "PASS"
            if calibration_gate
            else
            "NOT PROVEN"
        ),
    )

    # ========================================================
    # D. Exact relevant contexts
    # ========================================================

    print()
    print(
        "===== D. MATERIALIZATION LOG CONTEXTS ====="
    )

    patterns = (
        r"development_gaussian_posterior",
        r"mean_displacement_H0_m",
        r"anchor_position_H0_m",
        r"mean_absolute_H0_m",
        r"calibrated_covariance_H0_m2",
        r"probe_calibrated_output",
        r"build_gaussian_model",
        r"gaussian_gru\.pt",
        r"stage4_gaussian_gru\.json",
        r"fit_normalization\.json",
        r"CalibrationNormalizer",
        r"variance_scale",
        r"stage4_calibration\.json",
        r"inference_mode",
        r"model\.eval",
        r"load_state_dict",
        r"np\.savez",
        r"savez_compressed",
    )

    excerpt_parts = []

    context_payload = {}

    for path in producer_logs:

        found = contexts(
            path,
            patterns,
            radius=6,
            maximum=80,
        )

        context_payload[
            path.name
        ] = found

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            path
        )
        print(
            "contexts =",
            len(found),
        )
        print(
            "------------------------------------------------------------"
        )

        for index, item in enumerate(
            found,
            start=1,
        ):

            print()
            print(
                f"--- context {index} "
                f"(line {item['line']}) ---"
            )

            print(
                item[
                    "context"
                ]
            )

            excerpt_parts.append(
                "\n"
                "============================================================\n"
                f"{path}:{item['line']}\n"
                "============================================================\n"
                f"{item['context']}\n"
            )

    # ========================================================
    # E. Supporting resolver/route contexts
    # ========================================================

    print()
    print(
        "===== E. RESOLVER / ROUTE SUPPORTING EVIDENCE ====="
    )

    support_patterns = (
        r"checkpoint",
        r"gaussian_gru\.pt",
        r"stage4_gaussian_gru\.json",
        r"normalization",
        r"fit_normalization\.json",
        r"calibration",
        r"variance_scale",
        r"development_cache",
        r"development_gaussian_posterior",
        r"probe_calibrated_output",
        r"build_gaussian_model",
    )

    support_payload = {}

    for path in support_logs:

        found = contexts(
            path,
            support_patterns,
            radius=4,
            maximum=40,
        )

        support_payload[
            path.name
        ] = found

        print()
        print(
            "---",
            path.name,
            "| contexts =",
            len(found),
            "---"
        )

        for item in found[:20]:

            print()
            print(
                item[
                    "context"
                ]
            )

    # ========================================================
    # F. JSON artifact consistency
    # ========================================================

    print()
    print(
        "===== F. FROZEN JSON CONSISTENCY ====="
    )

    dev_json = load_json(
        DEV_JSON
    )

    resolver_json = load_json(
        RESOLVER_JSON
    )

    route_json = load_json(
        ROUTE_JSON
    )

    json_text = "\n".join(
        flattened_strings(
            {
                "development":
                    dev_json,

                "resolver":
                    resolver_json,

                "route":
                    route_json,
            }
        )
    )

    json_checkpoint = contains_any(
        json_text,
        (
            "gaussian_gru.pt",
            "stage4_gaussian_gru",
        ),
    )

    json_normalization = contains_any(
        json_text,
        (
            "fit_normalization",
            "fit_only",
        ),
    )

    json_calibration = contains_any(
        json_text,
        (
            "variance_scale",
            "calibrat",
        ),
    )

    json_coordinate = (
        contains_all(
            json.dumps(
                dev_json,
                sort_keys=True,
            ),
            (
                "metric_H0_displacement",
                "latest_causal_observed_target_position_H0",
            ),
        )
    )

    print(
        "JSON checkpoint/config provenance =",
        json_checkpoint,
    )

    print(
        "JSON fit-only normalization       =",
        json_normalization,
    )

    print(
        "JSON calibration provenance       =",
        json_calibration,
    )

    print(
        "JSON coordinate semantics         =",
        json_coordinate,
    )

    # ========================================================
    # G. Producer-route resolution
    # ========================================================

    print()
    print(
        "===== G. BLOCK5.2 PRODUCER ROUTE RESOLUTION ====="
    )

    strong_route = all(
        (
            output_gate,
            calibrated_covariance_output_gate,
            runtime_execution_gate,
            checkpoint_gate,
            normalization_gate,
            calibration_gate,
            json_coordinate,
        )
    )

    if strong_route:

        status = (
            "PASS_READY_FOR_SINGLE_RECORD_"
            "DEVELOPMENT_RUNTIME_BINDING"
        )

    else:

        status = (
            "BLOCKED_MATERIALIZATION_PRODUCER_"
            "EVIDENCE_INCOMPLETE"
        )

    print(
        "posterior producer output         =",
        output_gate,
    )

    print(
        "Stage4 runtime execution          =",
        runtime_execution_gate,
    )

    print(
        "checkpoint/config provenance      =",
        checkpoint_gate,
    )

    print(
        "normalization provenance          =",
        normalization_gate,
    )

    print(
        "calibration provenance            =",
        calibration_gate,
    )

    print(
        "identity reconstruction from NPZ  = NO"
    )

    print(
        "truth/perfect-ID reconstruction   = NO"
    )

    print(
        "new model inference               = NO"
    )

    # ========================================================
    # H. Immutability
    # ========================================================

    print()
    print(
        "===== H. IMMUTABILITY ====="
    )

    changed = [
        str(path)
        for path in protected
        if (
            before[
                str(path)
            ]
            !=
            sha256_file(
                path
            )
        )
    ]

    require(
        not changed,
        (
            "Frozen evidence changed: "
            + ", ".join(changed)
        ),
    )

    print(
        "Block6.3 Part1       = UNCHANGED"
    )

    print(
        "Block6.3 prior audits= UNCHANGED"
    )

    print(
        "Stage4/5 artifacts   = UNCHANGED"
    )

    print(
        "dataset access       = NO"
    )

    print(
        "model inference      = NO"
    )

    print(
        "formal evaluation    = NO"
    )

    # ========================================================
    # I. Artifacts
    # ========================================================

    EXCERPT.write_text(
        "".join(
            excerpt_parts
        ),
        encoding="utf-8",
    )

    report = {
        "project":
            "Agni",

        "stage":
            6,

        "block":
            "6.3",

        "part":
            "2/2",

        "audit":
            "BLOCK52_MATERIALIZATION_PROVENANCE",

        "status":
            status,

        "gates": {
            "posterior_output":
                output_gate,

            "calibrated_covariance_output":
                calibrated_covariance_output_gate,

            "probe_calibrated_output_call":
                exact_probe_call_gate,

            "direct_runtime_call":
                direct_runtime_gate,

            "runtime_execution":
                runtime_execution_gate,

            "checkpoint":
                checkpoint_gate,

            "fit_only_normalization":
                normalization_gate,

            "calibration":
                calibration_gate,

            "coordinate_semantics":
                json_coordinate,
        },

        "producer_logs": {
            str(path): {
                "sha256":
                    sha256_file(path),

                "contexts":
                    context_payload[
                        path.name
                    ],
            }
            for path in producer_logs
        },

        "support_contexts":
            support_payload,

        "controller_decision": {
            "anonymous_batch_controller_input":
                False,

            "arbitrary_batch_row":
                False,

            "truth_identity_reconstruction":
                False,

            "identity_preserving_runtime_required":
                True,
        },

        "scope": {
            "dataset_access":
                False,

            "model_inference":
                False,

            "formal_evaluation":
                False,

            "parameter_tuning":
                False,

            "upstream_modified":
                False,
        },

        "excerpt":
            str(
                EXCERPT
            ),
    }

    write_json(
        REPORT,
        report,
    )

    print()
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

    print(
        "excerpt =",
        EXCERPT,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 6.3 PART 2/2 MATERIALIZATION AUDIT — FINAL"
    )
    print(
        "============================================================"
    )

    print(
        "anonymous NPZ controller use = NO"
    )

    print(
        "Block5.2 producer output     =",
        (
            "PASS"
            if output_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "Stage4 Gaussian runtime      =",
        (
            "PASS"
            if runtime_execution_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "checkpoint/config            =",
        (
            "PASS"
            if checkpoint_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "fit-only normalization       =",
        (
            "PASS"
            if normalization_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "frozen calibration           =",
        (
            "PASS"
            if calibration_gate
            else
            "NOT PROVEN"
        ),
    )

    print(
        "causal identity strategy     = IN-MEMORY"
    )

    print(
        "truth/perfect ID             = NO"
    )

    print(
        "new inference                = NO"
    )

    print(
        "formal N120                  = NO"
    )

    print(
        "STATUS =",
        status,
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
        "BLOCK6.3 MATERIALIZATION PROVENANCE AUDIT = BLOCKED"
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
        limit=14
    )

    print()
    print(
        "Block6.3 Part1 modified = NO"
    )

    print(
        "Stage4/5 modified = NO"
    )

    print(
        "dataset access = NO"
    )

    print(
        "model inference = NO"
    )

    print(
        "formal evaluation = NO"
    )

    print(
        "terminal remains open = YES"
    )

# Deliberately no sys.exit().

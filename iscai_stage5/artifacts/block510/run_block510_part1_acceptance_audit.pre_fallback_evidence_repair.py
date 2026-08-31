from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path("/home/agni/waymo")
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

OUT = (
    S5
    / "reports/block510_part1_acceptance_audit.json"
)

FORMAL_MANIFEST = (
    ROOT
    / "iscai_stage3/artifacts/block38e/"
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

B50 = S5 / "reports/block50_contract_freeze.json"
B51 = S5 / "reports/block51_receiver_geometry.json"
B52 = S5 / "reports/block52_receiver_angular_posterior.json"
B53 = S5 / "reports/block53_codebook_probability.json"
B54 = S5 / "reports/block54_beam_baselines.json"
B55 = S5 / "reports/block55_adaptive_topk.json"
B56 = S5 / "reports/block56_optical_link.json"
B57 = S5 / "reports/block57_latency_controller.json"

B58_PREFREEZE = (
    S5
    / "reports/block58_prefreeze_acceptance.json"
)

B58_FORMAL = (
    S5
    / "reports/block58_formal_evaluation.json"
)

B58_RECONCILED = (
    S5
    / "reports/block58_part2_reconciled_gate.json"
)

B58_RECONCILIATION = (
    S5
    / "artifacts/block58/"
      "block58_part2_reconciliation.json"
)

B58_CLOSURE = (
    S5
    / "artifacts/block58/"
      "block58_final_closure.json"
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


EXPECTED_STAGE4 = {
    str(STAGE4_CLOSURE):
        (
            "570da4feb918b1025b5e85cc919360d9"
            "22b471c468c85b3844f13fb7774e7c2f"
        ),

    str(STAGE4_HANDOFF):
        (
            "491bce010d35c2a394f879ecf35ed26e"
            "f1de92f7fff1465dcbf46b072a87c6fd"
        ),

    str(STAGE4_FREEZE):
        (
            "88e3f290e1f7c1d037684adc132ea8ae"
            "78af057b3e3ab564fb29516687eb25e7"
        ),

    str(PDF_CERT):
        (
            "7c94bb9deb37a5eba29c9237bfcf4b0d"
            "24e4031d9b1b22a500c5a63730bf0d06"
        ),
}

EXPECTED_FORMAL_MANIFEST_SHA = (
    "2208e7287ddf6439fda4597c435a9cba"
    "1d1b9d0e4c4547bc5dd92e56e8124e46"
)

EXPECTED_B59_SHA = (
    "980bfbd2007ed798742bd7c36b5e8b6b"
    "cd0e808590c0dc5d9608db1ec6211845"
)

EXPECTED_CODEBOOKS = {16, 32, 64}
EXPECTED_HORIZONS = {0.1, 0.3, 0.5, 1.0}
EXPECTED_Q = {0.90, 0.95, 0.975, 0.99}
EXPECTED_TESTS = 257


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


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def pass_like(value):
    if value is None:
        return False

    text = str(value).upper()

    if any(
        bad in text
        for bad in (
            "BLOCK",
            "FAIL",
            "ERROR",
        )
    ):
        return False

    return any(
        good in text
        for good in (
            "PASS",
            "COMPLETE",
            "FROZEN",
        )
    )


def flatten(value, prefix=""):
    rows = []

    if isinstance(value, dict):
        for key, item in value.items():
            child = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.extend(
                flatten(
                    item,
                    child,
                )
            )

    elif isinstance(value, list):
        rows.append(
            (
                prefix,
                value,
            )
        )

        for index, item in enumerate(value):
            rows.extend(
                flatten(
                    item,
                    f"{prefix}[{index}]",
                )
            )

    else:
        rows.append(
            (
                prefix,
                value,
            )
        )

    return rows


def nested(payload, *keys):
    value = payload

    for key in keys:
        if not isinstance(value, dict):
            return None

        value = value.get(key)

    return value


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


def numeric_set(values, digits=8):
    result = set()

    for value in values:
        if finite(value):
            result.add(
                round(
                    float(value),
                    digits,
                )
            )

    return result


def joined_payload_text(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        default=str,
    ).lower()


def has_any(text, aliases):
    return any(
        alias.lower() in text
        for alias in aliases
    )


def codebook_values(payload):
    found = set()

    for path, value in flatten(payload):
        if "codebook" not in path.lower():
            continue

        values = (
            value
            if isinstance(value, list)
            else [value]
        )

        for item in values:
            if finite(item):
                integer = int(item)

                if integer in EXPECTED_CODEBOOKS:
                    found.add(integer)

    return found


def values_for_suffix(
    payload,
    suffix,
    parent=None,
):
    values = []

    for path, value in flatten(payload):
        lower = path.lower()

        if not lower.endswith(
            suffix.lower()
        ):
            continue

        if (
            parent is not None
            and parent.lower() not in lower
        ):
            continue

        if finite(value):
            values.append(
                float(value)
            )

    return values


def explicit_false_evidence(
    payloads,
    token,
):
    token = token.lower()

    for payload in payloads:
        for path, value in flatten(payload):
            if (
                token in path.lower()
                and value is False
            ):
                return True

    return False


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

    text = process.stdout

    ran = None

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("Ran "):
            parts = stripped.split()

            if len(parts) >= 2:
                try:
                    ran = int(parts[1])
                except Exception:
                    pass

    return (
        process.returncode,
        ran,
        text,
    )


# ============================================================
# Acceptance matrix
# ============================================================

def main():
    print(
        "============================================================"
    )
    print(
        "STAGE 5 — BLOCK 5.10 PART 1/2"
    )
    print(
        "FINAL ACCEPTANCE AUDIT BEFORE FREEZE"
    )
    print(
        "============================================================"
    )

    required = (
        STAGE4_CLOSURE,
        STAGE4_HANDOFF,
        STAGE4_FREEZE,
        PDF_CERT,
        FORMAL_MANIFEST,
        B50,
        B51,
        B52,
        B53,
        B54,
        B55,
        B56,
        B57,
        B58_PREFREEZE,
        B58_FORMAL,
        B58_RECONCILED,
        B58_RECONCILIATION,
        B58_CLOSURE,
        B59,
        B59_PART2,
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    require(
        not missing,
        (
            "Missing mandatory closure evidence: "
            + ", ".join(missing)
        ),
    )

    # All files below are protected during Part1.
    protected = required

    before_sha = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    # ========================================================
    # A. Frozen upstream continuity
    # ========================================================

    print()
    print(
        "===== A. STAGES 0–4 / STAGE4 HANDOFF CONTINUITY ====="
    )

    for path_string, expected in (
        EXPECTED_STAGE4.items()
    ):
        path = Path(path_string)

        require(
            sha256_file(path) == expected,
            (
                "Frozen upstream SHA mismatch: "
                f"{path}"
            ),
        )

    require(
        sha256_file(FORMAL_MANIFEST)
        ==
        EXPECTED_FORMAL_MANIFEST_SHA,
        "Formal N=120 manifest SHA changed.",
    )

    cert = load_json(PDF_CERT)

    require(
        cert.get("status")
        ==
        "100_PERCENT_PDF_COMPLIANT_THROUGH_STAGE4",
        (
            "Stages0–4 PDF compliance "
            "certification changed."
        ),
    )

    stage4 = load_json(
        STAGE4_CLOSURE
    )

    require(
        stage4.get("status")
        ==
        "COMPLETE_FROZEN",
        (
            "Stage4 is no longer "
            "COMPLETE_FROZEN."
        ),
    )

    handoff4 = load_json(
        STAGE4_HANDOFF
    )

    print(
        "Stages0–4 PDF compliance = PASS"
    )

    print(
        "Stage4 closure            = COMPLETE_FROZEN"
    )

    print(
        "Stage4→Stage5 handoff      = EXACT PASS"
    )

    print(
        "formal N=120 manifest      = EXACT PASS"
    )

    # ========================================================
    # B. Blocks 5.0–5.7
    # ========================================================

    print()
    print(
        "===== B. BLOCKS 5.0 → 5.7 ====="
    )

    reports = {
        "5.0": load_json(B50),
        "5.1": load_json(B51),
        "5.2": load_json(B52),
        "5.3": load_json(B53),
        "5.4": load_json(B54),
        "5.5": load_json(B55),
        "5.6": load_json(B56),
        "5.7": load_json(B57),
    }

    for block, report in reports.items():
        require(
            pass_like(
                report.get("status")
            ),
            (
                f"Block {block} is not PASS-like."
            ),
        )

        print(
            f"Block {block:3s} = PASS"
        )

    # ========================================================
    # C. Receiver-aware posterior / geometry
    # ========================================================

    print()
    print(
        "===== C. RECEIVER-AWARE POSTERIOR ====="
    )

    formal = load_json(
        B58_FORMAL
    )

    formal_text = joined_payload_text(
        formal
    )

    require(
        formal.get("status") == "PASS",
        "Block5.8 formal evaluation is not PASS.",
    )

    for geometry in (
        "centroid",
        "known",
        "uncertain",
    ):
        require(
            f"{geometry}|" in formal_text,
            (
                "Formal Stage5 evidence lacks "
                f"receiver geometry {geometry}."
            ),
        )

    posterior_text = (
        joined_payload_text(
            reports["5.2"]
        )
        +
        formal_text
    )

    require(
        has_any(
            posterior_text,
            (
                "angular",
                "receiver_posterior",
                "receiver posterior",
            ),
        ),
        (
            "Receiver/angular posterior "
            "evidence not found."
        ),
    )

    print(
        "receiver/angular posterior = PASS"
    )

    print(
        "centroid baseline           = PASS"
    )

    print(
        "known receiver offset       = PASS"
    )

    print(
        "uncertain receiver offset   = PASS"
    )

    # ========================================================
    # D. Codebooks + beam probabilities
    # ========================================================

    print()
    print(
        "===== D. CODEBOOKS / BEAM PROBABILITIES ====="
    )

    block53_codebooks = (
        codebook_values(
            reports["5.3"]
        )
    )

    frozen_codebooks = nested(
        formal,
        "frozen_contract",
        "codebooks",
    )

    require(
        block53_codebooks.issuperset(
            EXPECTED_CODEBOOKS
        ),
        (
            "Block5.3 does not prove "
            "16/32/64 codebooks."
        ),
    )

    require(
        isinstance(
            frozen_codebooks,
            list,
        )
        and
        {
            int(value)
            for value in frozen_codebooks
        }
        ==
        EXPECTED_CODEBOOKS,
        (
            "Formal frozen codebooks "
            "are not exactly 16/32/64."
        ),
    )

    r59 = load_json(
        B59
    )

    require(
        r59.get(
            "beam_probability_vectors_exact"
        )
        is True,
        (
            "Beam probability vectors "
            "are not reproducibly exact."
        ),
    )

    print(
        "16 / 32 / 64 codebooks    = PASS"
    )

    print(
        "beam probability vectors  = EXACT PASS"
    )

    # ========================================================
    # E. Required beam baselines
    # ========================================================

    print()
    print(
        "===== E. REQUIRED BEAM BASELINES ====="
    )

    baseline_text = (
        joined_payload_text(
            reports["5.4"]
        )
        +
        formal_text
    )

    baseline_aliases = {
        "exhaustive":
            (
                "exhaustive",
            ),

        "previous-beam persistence":
            (
                "previous_beam_persistence",
                "previous-beam",
            ),

        "geometry nearest beam":
            (
                "geometry_nearest",
                "nearest_beam",
                "nearest beam",
            ),

        "fixed Top-1":
            (
                "fixed_top1",
                "fixed_top_1",
                "top-1",
            ),

        "fixed Top-3":
            (
                "fixed_top3",
                "fixed_top_3",
                "top-3",
            ),

        "fixed Top-5":
            (
                "fixed_top5",
                "fixed_top_5",
                "top-5",
            ),

        "oracle beam set":
            (
                "oracle",
            ),
    }

    for name, aliases in (
        baseline_aliases.items()
    ):
        require(
            has_any(
                baseline_text,
                aliases,
            ),
            (
                "Required beam baseline "
                f"missing: {name}"
            ),
        )

        print(
            f"{name:28s}= PASS"
        )

    print(
        "blockage-aware Top-K      = OPTIONAL / NOT REQUIRED"
    )

    # ========================================================
    # F. Adaptive Top-K contract
    # ========================================================

    print()
    print(
        "===== F. ADAPTIVE TOP-K CONTRACT ====="
    )

    q_values = numeric_set(
        values_for_suffix(
            formal,
            ".q",
            parent="reliability_overhead",
        )
    )

    require(
        q_values.issuperset(
            numeric_set(
                EXPECTED_Q
            )
        ),
        (
            "Missing required Top-K "
            "probability-mass targets."
        ),
    )

    horizons = numeric_set(
        values_for_suffix(
            formal,
            ".horizon",
            parent="baseline_metrics",
        )
    )

    require(
        horizons.issuperset(
            numeric_set(
                EXPECTED_HORIZONS
            )
        ),
        (
            "Missing required Stage5 "
            "prediction horizons."
        ),
    )

    prefreeze = load_json(
        B58_PREFREEZE
    )

    policy = prefreeze.get(
        "acceptance_policy",
        {}
    )

    require(
        prefreeze.get("status")
        ==
        "PASS_PREFORMAL_FREEZE",
        (
            "Pre-formal acceptance "
            "policy is not frozen."
        ),
    )

    require(
        abs(
            float(
                policy.get("nominal_q")
            )
            -
            0.95
        )
        <=
        1e-12,
        "Nominal q is not 0.95.",
    )

    print(
        "q = 0.90                = PASS"
    )
    print(
        "q = 0.95 nominal        = PASS"
    )
    print(
        "q = 0.975               = PASS"
    )
    print(
        "q = 0.99                = PASS"
    )
    print(
        "horizons 0.1/0.3/0.5/1s = PASS"
    )

    # ========================================================
    # G. Latency / temporal controller / fallback
    # ========================================================

    print()
    print(
        "===== G. LATENCY / FALLBACK ====="
    )

    b57_text = joined_payload_text(
        reports["5.7"]
    )

    require(
        "latency" in b57_text,
        "Block5.7 lacks latency evidence.",
    )

    require(
        has_any(
            b57_text,
            (
                "fallback",
                "reacquisition",
                "reacquire",
                "loss_of_lock",
                "loss-of-lock",
            ),
        ),
        (
            "Block5.7 lacks fallback/"
            "reacquisition evidence."
        ),
    )

    require(
        has_any(
            b57_text,
            (
                "hysteresis",
                "persistence",
                "switch",
            ),
        ),
        (
            "Block5.7 lacks temporal "
            "beam-stability evidence."
        ),
    )

    print(
        "beam/controller latency   = PASS"
    )

    print(
        "fallback/reacquisition    = PASS"
    )

    print(
        "temporal stability logic  = PASS"
    )

    print(
        "joint beam+ADB E2E latency = DEFERRED TO STAGE7"
    )

    # ========================================================
    # H. Optical link chain
    # ========================================================

    print()
    print(
        "===== H. OPTICAL LINK CHAIN ====="
    )

    optical_text = (
        joined_payload_text(
            reports["5.6"]
        )
        +
        formal_text
    )

    optical_requirements = {
        "pointing / gain":
            (
                "pointing",
                "beam_gain",
                "optical_gain",
            ),

        "received power":
            (
                "received_power",
                "received power",
                "p_rx",
            ),

        "SNR":
            (
                "snr",
            ),

        "DPSK BER":
            (
                "ber",
                "dpsk",
            ),

        "effective rate":
            (
                "effective_rate",
                "effective rate",
            ),
    }

    for name, aliases in (
        optical_requirements.items()
    ):
        require(
            has_any(
                optical_text,
                aliases,
            ),
            (
                "Optical Stage5 chain lacks "
                f"{name} evidence."
            ),
        )

        print(
            f"{name:24s}= PASS"
        )

    # ========================================================
    # I. Formal population + no post-hoc tuning
    # ========================================================

    print()
    print(
        "===== I. FORMAL N=120 / LEAKAGE CONTRACT ====="
    )

    population = formal.get(
        "formal_population",
        {}
    )

    require(
        int(
            population.get("N")
        )
        ==
        120,
        "Formal Stage5 population is not N=120.",
    )

    require(
        population.get(
            "same_immutable_Stage3_4_population"
        )
        is True,
        (
            "Formal Stage5 population "
            "is not immutable Stage3/4 N=120."
        ),
    )

    require(
        nested(
            formal,
            "scope",
            "formal_parameter_tuning",
        )
        is False,
        (
            "Formal parameter tuning "
            "was recorded."
        ),
    )

    require(
        prefreeze.get(
            "formal_results_used_for_parameter_selection"
        )
        is False,
        (
            "Formal outcomes were used "
            "for parameter selection."
        ),
    )

    causal_reports = (
        reports["5.0"],
        reports["5.1"],
        formal,
    )

    tracks_selector_false = (
        explicit_false_evidence(
            causal_reports,
            "tracks_to_predict",
        )
    )

    require(
        tracks_selector_false,
        (
            "Could not prove from frozen Stage5 "
            "reports that tracks_to_predict is "
            "not the receiver selector."
        ),
    )

    print(
        "formal N                  = 120 PASS"
    )

    print(
        "same immutable population = PASS"
    )

    print(
        "formal parameter tuning   = NO"
    )

    print(
        "post-hoc selection        = NO"
    )

    print(
        "tracks_to_predict selector = NO PASS"
    )

    # ========================================================
    # J. PDF completion gate
    # ========================================================

    print()
    print(
        "===== J. PDF STAGE5 COMPLETION GATE ====="
    )

    primary = formal.get(
        "nominal_primary_summary",
        {}
    )

    coverage = float(
        primary.get(
            "probability_coverage"
        )
    )

    overhead_reduction = float(
        primary.get(
            "overhead_reduction_vs_exhaustive"
        )
    )

    mean_overhead = float(
        primary.get(
            "probing_overhead_fraction"
        )
    )

    beam_gain_loss = float(
        primary.get(
            "beam_gain_loss_db"
        )
    )

    switching_rate = float(
        primary.get(
            "beam_switching_rate"
        )
    )

    require(
        all(
            math.isfinite(value)
            for value in (
                coverage,
                overhead_reduction,
                mean_overhead,
                beam_gain_loss,
                switching_rate,
            )
        ),
        "Non-finite primary formal metric.",
    )

    require(
        coverage >= 0.95,
        (
            "Adaptive Top-K empirical "
            "coverage misses nominal q=0.95."
        ),
    )

    require(
        overhead_reduction > 0.0,
        (
            "Adaptive Top-K does not reduce "
            "probing overhead vs exhaustive."
        ),
    )

    print(
        "requested q              = 0.95"
    )

    print(
        "empirical coverage       =",
        coverage,
        "PASS",
    )

    print(
        "overhead reduction       =",
        overhead_reduction,
        "PASS",
    )

    print(
        "probing overhead fraction=",
        mean_overhead,
    )

    print(
        "beam-gain loss dB        =",
        beam_gain_loss,
    )

    print(
        "beam-switching rate      =",
        switching_rate,
    )

    # ========================================================
    # K. Full beam-management metric contract
    # ========================================================

    print()
    print(
        "===== K. PDF BEAM-METRIC CONTRACT ====="
    )

    metric_groups = {
        "Top-1 hit":
            (
                "fixed_top1",
                "top-1",
                "hit_rate",
            ),

        "Top-3 / Top-K hit":
            (
                "fixed_top3",
                "top-3",
                "topk",
            ),

        "average selected K":
            (
                "mean_k",
                "average_k",
            ),

        "probability coverage":
            (
                "probability_coverage",
                "\"coverage\"",
            ),

        "probing overhead":
            (
                "probing_overhead",
                "mean_probe_count",
            ),

        "overhead reduction":
            (
                "overhead_reduction",
            ),

        "beam gain loss":
            (
                "beam_gain_loss",
            ),

        "received-power loss":
            (
                "received_power_loss",
                "power_loss",
            ),

        "SNR loss":
            (
                "snr_loss",
            ),

        "BER":
            (
                "\"ber\"",
                ".ber",
            ),

        "effective rate":
            (
                "effective_rate",
            ),

        "outage":
            (
                "outage",
            ),

        "beam switching":
            (
                "beam_switch",
                "beam-switch",
            ),

        "reacquisition":
            (
                "reacquisition",
                "reacquire",
            ),

        "reliability-overhead":
            (
                "reliability_overhead",
            ),

        "Pareto frontier":
            (
                "pareto_frontier",
            ),
    }

    combined_metric_text = (
        formal_text
        +
        b57_text
        +
        optical_text
    )

    for name, aliases in (
        metric_groups.items()
    ):
        require(
            has_any(
                combined_metric_text,
                aliases,
            ),
            (
                "Missing PDF beam-management "
                f"metric: {name}"
            ),
        )

        print(
            f"{name:24s}= PASS"
        )

    # ========================================================
    # L. Block5.8 reconciliation / immutable closure
    # ========================================================

    print()
    print(
        "===== L. BLOCK5.8 FINAL EVIDENCE ====="
    )

    reconciled = load_json(
        B58_RECONCILED
    )

    reconciliation = load_json(
        B58_RECONCILIATION
    )

    closure58 = load_json(
        B58_CLOSURE
    )

    require(
        reconciled.get("status")
        ==
        "PASS_RECONCILED",
        (
            "Block5.8 Part2 reconciliation "
            "gate is not PASS."
        ),
    )

    require(
        reconciliation.get("status")
        ==
        "PASS_RECONCILED_COMPLETE",
        (
            "Block5.8 reconciliation "
            "is not complete."
        ),
    )

    require(
        pass_like(
            closure58.get("status")
        ),
        (
            "Block5.8 final closure "
            "is not PASS-like."
        ),
    )

    frozen58 = r59.get(
        "frozen_block58",
        {}
    )

    require(
        sha256_file(B58_CLOSURE)
        ==
        frozen58.get(
            "closure_sha256"
        ),
        (
            "Frozen Block5.8 closure SHA "
            "does not match Block5.9."
        ),
    )

    require(
        sha256_file(B58_FORMAL)
        ==
        frozen58.get(
            "formal_report_sha256"
        ),
        (
            "Frozen Block5.8 formal-report SHA "
            "does not match Block5.9."
        ),
    )

    print(
        "Block5.8 reconciliation = PASS"
    )

    print(
        "Block5.8 closure SHA     = EXACT PASS"
    )

    print(
        "Block5.8 formal SHA      = EXACT PASS"
    )

    # ========================================================
    # M. Block5.9 reproducibility
    # ========================================================

    print()
    print(
        "===== M. BLOCK5.9 REPRODUCIBILITY ====="
    )

    require(
        sha256_file(B59)
        ==
        EXPECTED_B59_SHA,
        (
            "Block5.9 frozen report "
            "SHA changed."
        ),
    )

    require(
        r59.get("status")
        ==
        "PASS_FROZEN",
        (
            "Block5.9 is not PASS_FROZEN."
        ),
    )

    require(
        r59.get(
            "exact_match_to_frozen_block58"
        )
        is True,
        (
            "Block5.9 exact frozen match "
            "is false."
        ),
    )

    require(
        r59.get(
            "fresh_process_repeat_exact"
        )
        is True,
        (
            "Block5.9 fresh process "
            "repeat is not exact."
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
            "Block5.9 fresh process "
            "repeat count < 2."
        ),
    )

    part2_59 = load_json(
        B59_PART2
    )

    require(
        part2_59.get("status")
        ==
        "PASS_COMPLETE",
        (
            "Block5.9 terminal-contract "
            "repair is not PASS_COMPLETE."
        ),
    )

    print(
        "canonical status          = PASS_FROZEN"
    )

    print(
        "exact frozen match        = PASS"
    )

    print(
        "fresh-process repeat      = EXACT PASS"
    )

    print(
        "test-contract repair      = PASS_COMPLETE"
    )

    # ========================================================
    # N. Full regression
    # ========================================================

    print()
    print(
        "===== N. FULL STAGE5 REGRESSION ====="
    )

    rc, test_count, test_output = (
        run_full_regression()
    )

    if rc != 0:
        print(test_output)

    require(
        rc == 0,
        (
            "Full Stage5 regression failed."
        ),
    )

    require(
        test_count == EXPECTED_TESTS,
        (
            "Unexpected Stage5 regression count: "
            f"{test_count}, expected "
            f"{EXPECTED_TESTS}."
        ),
    )

    print(
        "Stage5 regression        =",
        f"{test_count} / {EXPECTED_TESTS} PASS",
    )

    # ========================================================
    # O. Acceptance matrix
    # ========================================================

    print()
    print(
        "===== O. FINAL STAGE5 ACCEPTANCE MATRIX ====="
    )

    matrix = {
        "01_upstream_Stage4_frozen":
            True,

        "02_Stage5_contract_frozen":
            True,

        "03_receiver_angular_posterior":
            True,

        "04_receiver_geometry_centroid_known_uncertain":
            True,

        "05_codebooks_16_32_64_and_probabilities":
            True,

        "06_required_fixed_classical_beam_baselines":
            True,

        "07_adaptive_TopK_90_95_97p5_99":
            True,

        "08_latency_fallback_reacquisition":
            True,

        "09_optical_pointing_gain_power_SNR_BER_rate":
            True,

        "10_formal_immutable_N120":
            True,

        "11_requested_coverage_gate":
            coverage >= 0.95,

        "12_overhead_reduction_gate":
            overhead_reduction > 0.0,

        "13_full_beam_metric_contract":
            True,

        "14_causal_no_formal_tuning_contract":
            True,

        "15_tracks_to_predict_not_receiver_selector":
            tracks_selector_false,

        "16_exact_reproducibility":
            (
                r59.get(
                    "fresh_process_repeat_exact"
                )
                is True
            ),

        "17_full_regression":
            (
                rc == 0
                and
                test_count
                ==
                EXPECTED_TESTS
            ),
    }

    passed = sum(
        bool(value)
        for value in matrix.values()
    )

    total = len(matrix)

    require(
        passed == total,
        (
            "Stage5 acceptance matrix "
            f"is only {passed}/{total}."
        ),
    )

    for key, value in matrix.items():
        print(
            key,
            "=",
            "PASS" if value else "FAIL",
        )

    print()
    print(
        "mandatory acceptance =",
        f"{passed} / {total} PASS",
    )

    # ========================================================
    # P. Protected-artifact immutability
    # ========================================================

    print()
    print(
        "===== P. IMMUTABILITY READBACK ====="
    )

    after_sha = {
        str(path):
            sha256_file(path)
        for path in protected
    }

    changed = [
        path
        for path in before_sha
        if before_sha[path]
        != after_sha[path]
    ]

    require(
        not changed,
        (
            "Protected Stage0–5 artifact "
            "changed during acceptance audit: "
            + ", ".join(changed)
        ),
    )

    print(
        "Stages0–4 frozen evidence = UNCHANGED"
    )

    print(
        "Blocks5.0–5.7 reports     = UNCHANGED"
    )

    print(
        "Block5.8 formal evidence  = UNCHANGED"
    )

    print(
        "Block5.9 frozen evidence  = UNCHANGED"
    )

    # ========================================================
    # Q. Write Part1 acceptance audit
    # ========================================================

    payload = {
        "project":
            "Agni",

        "stage":
            5,

        "block":
            "5.10_part_1",

        "status":
            "PASS_READY_FOR_FINAL_FREEZE",

        "mandatory_acceptance": {
            "passed":
                passed,

            "total":
                total,

            "matrix":
                matrix,
        },

        "formal": {
            "N":
                120,

            "manifest_sha256":
                EXPECTED_FORMAL_MANIFEST_SHA,

            "codebooks":
                [16, 32, 64],

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

            "empirical_probability_coverage":
                coverage,

            "overhead_reduction_vs_exhaustive":
                overhead_reduction,

            "probing_overhead_fraction":
                mean_overhead,

            "beam_gain_loss_db":
                beam_gain_loss,

            "beam_switching_rate":
                switching_rate,
        },

        "reproducibility": {
            "status":
                "PASS_FROZEN",

            "report_sha256":
                EXPECTED_B59_SHA,

            "exact_match_to_frozen_block58":
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
        },

        "regression": {
            "tests":
                test_count,

            "passed":
                test_count,

            "status":
                "PASS",
        },

        "scope_boundary": {
            "Stage5":
                (
                    "receiver-aware probabilistic "
                    "communication beam controller "
                    "and optical link evaluation"
                ),

            "predictive_ADB":
                "DEFERRED_TO_STAGE6",

            "joint_beam_ADB_evaluation":
                "DEFERRED_TO_STAGE7",

            "DeepSense":
                "DEFERRED_TO_STAGE8",
        },

        "scientific_execution": {
            "training":
                False,

            "formal_inference":
                False,

            "recalibration":
                False,

            "parameter_tuning":
                False,

            "reproducibility_rerun":
                False,
        },

        "protected_artifacts_modified":
            False,

        "Stage5_final_freeze_written":
            False,

        "Stage6_handoff_written":
            False,

        "Stage6_started_by_this_block":
            False,

        "next":
            (
                "Block5.10 Part2/2: canonical "
                "Stage5 final closure, freeze "
                "manifest, and Stage5→Stage6 "
                "posterior/interface handoff"
            ),
    }

    atomic_json(
        OUT,
        payload,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BLOCK 5.10 PART 1/2 — FINAL ACCEPTANCE AUDIT"
    )
    print(
        "============================================================"
    )

    print(
        "Blocks 5.0 → 5.9          = PASS"
    )

    print(
        "mandatory acceptance      =",
        f"{passed} / {total} PASS",
    )

    print(
        "formal N=120              = PASS"
    )

    print(
        "receiver-aware posterior  = PASS"
    )

    print(
        "16/32/64 codebooks        = PASS"
    )

    print(
        "adaptive q targets        = PASS"
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
        "optical link chain        = PASS"
    )

    print(
        "latency/fallback          = PASS"
    )

    print(
        "causal/no tuning          = PASS"
    )

    print(
        "exact reproducibility     = PASS_FROZEN"
    )

    print(
        "full regression           =",
        f"{test_count} / {EXPECTED_TESTS} PASS",
    )

    print(
        "scientific outputs changed= NO"
    )

    print(
        "Stage5 final freeze       = NOT YET WRITTEN"
    )

    print(
        "Stage6 started            = NO"
    )

    print(
        "safe for Part2 final freeze = YES"
    )

    print(
        "STATUS = PASS_READY_FOR_FINAL_FREEZE"
    )

    print(
        "report =",
        OUT,
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
        "BLOCK 5.10 PART 1/2 = BLOCKED"
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
        "scientific outputs changed= NO"
    )

    print(
        "Stage5 final freeze       = NO"
    )

    print(
        "Stage6 started            = NO"
    )

    print(
        "terminal remains open     = YES"
    )

# No sys.exit() by design.

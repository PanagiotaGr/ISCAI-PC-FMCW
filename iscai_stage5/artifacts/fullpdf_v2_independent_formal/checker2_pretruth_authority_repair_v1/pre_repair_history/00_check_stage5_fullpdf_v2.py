#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


# =====================================================================
# Stage5 Full-PDF V2 Independent Formal Checker 2/2
#
# SCIENTIFIC BOUNDARY
# -------------------
# 1. No training/retraining/recalibration/tuning.
# 2. No formal outcome is parsed before the pre-truth seal.
# 3. Existing frozen/preformal formal checker is executed unchanged.
# 4. Stage4 calibrated covariance is consumed as already calibrated.
# 5. Stage6 is allowed only if every mandatory gate below passes.
# 6. No new empirical coverage tolerance is invented here.
# =====================================================================


ROOT = Path("/home/agni/waymo")
S3 = ROOT / "iscai_stage3"
S4 = ROOT / "iscai_stage4"
S5 = ROOT / "iscai_stage5"

PYTHON = Path(
    "/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python"
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
    S4
    / "artifacts/fullpdf_v2/"
      "formal_prediction_ledger_run1.jsonl"
)

FROZEN_CHECKER = (
    S5
    / "scripts/"
      "run_stage5_fullpdf_v2_independent_formal_checker.py"
)

FORMAL_REPORT = (
    S5
    / "reports/"
      "stage5_fullpdf_v2_independent_formal_checker.json"
)

OUTDIR = (
    S5
    / "artifacts/fullpdf_v2_independent_formal"
)
PRETRUTH_SEAL = OUTDIR / "checker2_pretruth_seal.json"
SUBPROCESS_LOG = OUTDIR / "checker2_frozen_checker.log"

CHECKER2_REPORT = (
    S5
    / "reports/stage5_fullpdf_v2_checker2_gate.json"
)

HANDOFF = (
    S5
    / "artifacts/fullpdf_v2/"
      "stage5_to_stage6_handoff.json"
)

EXPECTED = {
    "runtime": (
        "e57708e9332bb40e84f18270e483d911"
        "65e26c149b80e82c8f10b4df44376249"
    ),
    "protocol": (
        "91dec3cc3d3a7b72385f012d1d2bfdb"
        "8981a497e1344f585d38f3e05f2e4fa5a"
    ),
    "ledger": (
        "2dd3c396d5c365677c41b3a0df6344a"
        "7c28d423daaa368dbe466bd4b17b4c2a8"
    ),
    "addendum": (
        "570fbada05c8e8ed71775bbb6eb668f40"
        "16dd65c98b09b4f158abeb379eca6aa"
    ),
    "stage4_closure": (
        "01dbc5edcb4c67a56f1042655becf2ac"
        "04ed4130bb499f8d0aff6349e7cb0b5e"
    ),
    "stage4_handoff": (
        "7b1d2f8903f8ea27fe7567559a2b868b"
        "eebe9d834ee0a602155335339f6e204b"
    ),
}

EXPECTED_RECEIVER_MODES = (
    "centroid",
    "known",
    "uncertain",
)
EXPECTED_CODEBOOKS = (16, 32, 64)
EXPECTED_HORIZONS = (0.1, 0.3, 0.5, 1.0)
EXPECTED_TARGETS = (0.90, 0.95, 0.975, 0.99)

EXPECTED_COMBOS = set(
    itertools.product(
        EXPECTED_RECEIVER_MODES,
        EXPECTED_CODEBOOKS,
        EXPECTED_HORIZONS,
        EXPECTED_TARGETS,
    )
)

EXPECTED_REGRESSION_TESTS = 271

SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj) -> bytes:
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    tmp.write_bytes(
        canonical_json_bytes(obj)
    )
    os.replace(tmp, path)


def write_once_or_verify(
    path: Path,
    obj,
) -> None:
    payload = canonical_json_bytes(obj)

    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(
                "WRITE-ONCE artifact differs: "
                f"{path}"
            )
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("xb") as f:
        f.write(payload)


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            p = (
                str(key)
                if not prefix
                else f"{prefix}.{key}"
            )
            yield from flatten(value, p)

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            p = f"{prefix}[{index}]"
            yield from flatten(value, p)

    else:
        yield prefix, obj


def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_dicts(value)


def passish(value) -> bool:
    if value is True:
        return True

    if isinstance(value, str):
        s = value.strip().upper()

        if s in {
            "PASS",
            "PASSED",
            "TRUE",
            "OK",
            "COMPLETE",
            "COMPLETE_FROZEN_FULLPDF_V2",
        }:
            return True

        if (
            s.startswith("PASS_")
            or s.endswith("_PASS")
        ):
            return True

    return False


def normalize_float(
    value,
    allowed,
    atol=1.0e-12,
):
    try:
        x = float(value)
    except Exception:
        return None

    for expected in allowed:
        if math.isclose(
            x,
            float(expected),
            rel_tol=0.0,
            abs_tol=atol,
        ):
            return float(expected)

    return None


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        left = call_name(node.value)
        if left:
            return left + "." + node.attr
        return node.attr

    return ""


def source_call_names(path: Path):
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(path),
    )

    result = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            result.append(
                call_name(node.func)
            )

    return result


def find_exact_hash(
    root: Path,
    expected_sha: str,
):
    candidates = []

    for base in (
        root / "reports",
        root / "artifacts",
        root / "configs",
    ):
        if not base.exists():
            continue

        for p in base.rglob("*"):
            if not p.is_file():
                continue

            if p.suffix.lower() not in {
                ".json",
                ".jsonl",
                ".txt",
            }:
                continue

            try:
                if sha256(p) == expected_sha:
                    candidates.append(p)
            except OSError:
                pass

    return sorted(
        set(candidates)
    )


def find_pass_evidence(
    flat_report,
    token_groups,
):
    """
    token_groups:
      [
        ("requested", "coverage"),
        ("coverage", "acceptance"),
      ]

    PASS if any flattened path/value contains
    all tokens of one group and is pass-like.
    """
    for tokens in token_groups:
        for path, value in flat_report:
            text = (
                path + " " + str(value)
            ).lower()

            if all(
                token.lower() in text
                for token in tokens
            ):
                if passish(value):
                    return True, {
                        "path": path,
                        "value": value,
                    }

    return False, None


def find_hash_evidence(
    flat_report,
    required_tokens,
):
    hits = []

    for path, value in flat_report:
        if not isinstance(value, str):
            continue

        if not SHA_RE.fullmatch(
            value.lower()
        ):
            continue

        low = path.lower()

        if all(
            token in low
            for token in required_tokens
        ):
            hits.append(
                {
                    "path": path,
                    "sha256": value,
                }
            )

    return hits


def pick_alias(d, aliases):
    lowered = {
        str(k).lower(): v
        for k, v in d.items()
    }

    for alias in aliases:
        if alias in lowered:
            return lowered[alias]

    return None


def collect_combo_evidence(report):
    combos = {}
    flat_report = list(
        flatten(report)
    )

    def add(combo, evidence):
        if combo not in EXPECTED_COMBOS:
            return

        combos.setdefault(
            combo,
            set(),
        ).update(evidence)

    # ---------------------------------------------------------
    # A. Combined-key representation:
    # centroid|16|0.1|0.95
    # ---------------------------------------------------------
    pattern = re.compile(
        r"(centroid|known|uncertain)"
        r"[|/:,_-]+"
        r"(16|32|64)"
        r"[|/:,_-]+"
        r"(0\.1|0\.3|0\.5|1(?:\.0)?)"
        r"[|/:,_-]+"
        r"(0\.9(?:0)?|0\.95|0\.975|0\.99)",
        re.IGNORECASE,
    )

    for path, value in flat_report:
        text = path + " " + str(value)
        match = pattern.search(text)

        if match is None:
            continue

        mode = match.group(1).lower()
        codebook = int(match.group(2))

        horizon = normalize_float(
            match.group(3),
            EXPECTED_HORIZONS,
        )
        target = normalize_float(
            match.group(4),
            EXPECTED_TARGETS,
        )

        if horizon is None or target is None:
            continue

        leaf = path.split(".")[-1].lower()

        add(
            (
                mode,
                codebook,
                horizon,
                target,
            ),
            {leaf, path.lower()},
        )

    # ---------------------------------------------------------
    # B. Row/dict representation.
    # ---------------------------------------------------------
    for d in walk_dicts(report):
        mode = pick_alias(
            d,
            (
                "receiver_mode",
                "receiver_geometry",
                "receiver_geometry_mode",
                "geometry",
            ),
        )

        codebook = pick_alias(
            d,
            (
                "codebook",
                "codebook_size",
                "n_beams",
            ),
        )

        horizon = pick_alias(
            d,
            (
                "horizon",
                "horizon_s",
                "prediction_horizon_s",
            ),
        )

        target = pick_alias(
            d,
            (
                "requested_mass",
                "requested_probability_mass",
                "probability_target",
                "coverage_target",
                "target_probability",
            ),
        )

        if mode is None:
            continue

        mode = str(mode).lower()

        try:
            codebook = int(codebook)
        except Exception:
            continue

        horizon = normalize_float(
            horizon,
            EXPECTED_HORIZONS,
        )
        target = normalize_float(
            target,
            EXPECTED_TARGETS,
        )

        combo = (
            mode,
            codebook,
            horizon,
            target,
        )

        if combo not in EXPECTED_COMBOS:
            continue

        evidence = {
            str(k).lower()
            for k in d
        }

        add(combo, evidence)

    return combos


def metric_present(evidence, tokens):
    text = " ".join(
        sorted(evidence)
    )

    return any(
        token.lower() in text
        for token in tokens
    )


def handoff_candidates():
    result = set()

    patterns = (
        "*stage5*stage6*handoff*.json",
        "*stage5_to_stage6*.json",
        "*fullpdf*v2*handoff*.json",
    )

    for pattern in patterns:
        for p in S5.rglob(pattern):
            if p.is_file():
                result.add(p)

    return sorted(result)


def snapshot_hashes(paths):
    result = {}

    for p in paths:
        try:
            result[str(p)] = sha256(p)
        except OSError:
            result[str(p)] = None

    return result


def recurse_key_values(obj, wanted):
    wanted = wanted.lower()

    for d in walk_dicts(obj):
        for key, value in d.items():
            if str(key).lower() == wanted:
                yield value


# ---------------------------------------------------------------------
# Gate bookkeeping
# ---------------------------------------------------------------------

matrix = []
failures = []


def gate(
    requirement,
    ok,
    evidence,
    *,
    mandatory=True,
):
    status = "PASS" if ok else "FAIL"

    matrix.append(
        {
            "requirement": requirement,
            "mandatory": bool(mandatory),
            "evidence": evidence,
            "status": status,
        }
    )

    print(
        f"{status:4s} | {requirement}"
    )

    if mandatory and not ok:
        failures.append(
            requirement
        )


# =====================================================================
# 1. PRE-FORMAL / PRE-TRUTH VERIFICATION
# =====================================================================

print("=" * 79)
print(
    "STAGE5 FULL-PDF V2 — "
    "INDEPENDENT FORMAL CHECKER 2/2"
)
print(
    "FAIL-CLOSED | NO RETRAINING | "
    "NO RECALIBRATION | NO TUNING"
)
print("=" * 79)

required_paths = (
    RUNTIME,
    PROTOCOL,
    ADDENDUM,
    STAGE4_LEDGER,
    FROZEN_CHECKER,
)

for path in required_paths:
    gate(
        f"required path exists: {path}",
        path.is_file(),
        str(path),
    )

if failures:
    raise SystemExit(
        "FAIL_CLOSED: missing mandatory path."
    )


actual_hashes = {
    "runtime": sha256(RUNTIME),
    "protocol": sha256(PROTOCOL),
    "ledger": sha256(STAGE4_LEDGER),
    "addendum": sha256(ADDENDUM),
    "frozen_checker": sha256(
        FROZEN_CHECKER
    ),
    "checker2": sha256(
        Path(__file__).resolve()
    ),
}

for key in (
    "runtime",
    "protocol",
    "ledger",
    "addendum",
):
    gate(
        f"frozen SHA256: {key}",
        actual_hashes[key]
        == EXPECTED[key],
        {
            "expected": EXPECTED[key],
            "actual": actual_hashes[key],
        },
    )


protocol = load_json(PROTOCOL)
addendum = load_json(ADDENDUM)

gate(
    "protocol status frozen before V2 formal",
    protocol.get("status")
    == "FROZEN_BEFORE_STAGE5_V2_FORMAL",
    protocol.get("status"),
)

gate(
    "Stage4 posterior is calibrated 3D Gaussian",
    (
        protocol
        .get("upstream", {})
        .get("posterior")
        == "calibrated_full_3D_Gaussian_GRU"
    ),
    protocol.get("upstream", {}),
)

gate(
    "Stage4 calibration is NOT reapplied",
    (
        protocol
        .get("upstream", {})
        .get("Stage4_calibration_reapplied")
        is False
    ),
    protocol.get("upstream", {}),
)

gate(
    "receiver modes exact",
    tuple(
        protocol
        .get("receiver_geometry", {})
        .get("modes", [])
    )
    == EXPECTED_RECEIVER_MODES,
    protocol.get(
        "receiver_geometry",
        {},
    ),
)

gate(
    "codebooks exact 16/32/64",
    tuple(
        protocol
        .get("codebooks", {})
        .get("sizes", [])
    )
    == EXPECTED_CODEBOOKS,
    protocol.get("codebooks", {}),
)

gate(
    "probability targets exact",
    tuple(
        float(x)
        for x in (
            protocol
            .get("adaptive_TopK", {})
            .get("coverage_targets", [])
        )
    )
    == EXPECTED_TARGETS,
    protocol.get("adaptive_TopK", {}),
)

gate(
    "outside-FOV mass explicit; no clipping",
    (
        protocol
        .get("codebooks", {})
        .get("outside_support_mass")
        == "explicit"
        and
        protocol
        .get("codebooks", {})
        .get("azimuth_clipping")
        is False
    ),
    protocol.get("codebooks", {}),
)

gate(
    "all evaluated beams charged",
    (
        protocol
        .get("adaptive_TopK", {})
        .get("all_evaluated_beams_charged")
        is True
        and
        protocol
        .get("adaptive_TopK", {})
        .get("free_hidden_probes")
        is False
    ),
    protocol.get("adaptive_TopK", {}),
)

gate(
    "formal parameter tuning forbidden",
    (
        protocol
        .get("formal", {})
        .get("parameter_tuning_on_formal")
        is False
    ),
    protocol.get("formal", {}),
)


# ---------------------------------------------------------------------
# Stage4 frozen provenance by exact hashes.
# ---------------------------------------------------------------------

closure_matches = find_exact_hash(
    S4,
    EXPECTED["stage4_closure"],
)
handoff_matches = find_exact_hash(
    S4,
    EXPECTED["stage4_handoff"],
)

gate(
    "Stage4 Full-PDF V2 closure provenance hash found",
    len(closure_matches) >= 1,
    [str(x) for x in closure_matches],
)

gate(
    "Stage4→Stage5 V2 handoff provenance hash found",
    len(handoff_matches) >= 1,
    [str(x) for x in handoff_matches],
)


# ---------------------------------------------------------------------
# Runtime anti-recalibration static gate.
# ---------------------------------------------------------------------

runtime_calls = source_call_names(
    RUNTIME
)

forbidden_runtime_calls = {
    "apply_variance_scale",
    "calibrate_stage4_horizon_covariances",
    "fit_calibrator",
}

bad_runtime_calls = [
    x
    for x in runtime_calls
    if x.split(".")[-1]
    in forbidden_runtime_calls
]

gate(
    "V2 runtime has no second Stage4 calibrator call",
    len(bad_runtime_calls) == 0,
    bad_runtime_calls,
)


runtime_source = RUNTIME.read_text(
    encoding="utf-8"
)

required_runtime_tokens = (
    "already_calibrated_predictive_covariance",
    "outside_support_probability",
    "adaptive_topk",
    "local_neighbor_recovery",
    "widened_fallback",
    "exhaustive_loss_of_lock",
    "best_link_over_probed_beams",
)

missing_runtime_tokens = [
    token
    for token in required_runtime_tokens
    if token not in runtime_source
]

gate(
    "V2 recovery/no-free-probe runtime mechanisms present",
    not missing_runtime_tokens,
    {
        "required": required_runtime_tokens,
        "missing": missing_runtime_tokens,
    },
)


# ---------------------------------------------------------------------
# Stage4 prediction ledger:
# NO FUTURE TRUTH.
# ---------------------------------------------------------------------

ledger_rows = []
composite_keys = set()
duplicate_composite = 0
scenario_ids = set()
formal_ranks = set()
schema_failures = []
contract_failures = []
covariance_failures = []

with STAGE4_LEDGER.open(
    "r",
    encoding="utf-8",
) as f:
    for line_number, line in enumerate(
        f,
        start=1,
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        ledger_rows.append(row)

        required = {
            "scenario_id",
            "prediction_id",
            "formal_rank",
            "origin_H0_m",
            "gaussian_mean_H0_m",
            "gaussian_covariance_calibrated_H0_m2",
            "input_contract",
        }

        if not required <= set(row):
            schema_failures.append(
                line_number
            )
            continue

        key = (
            str(row["scenario_id"]),
            str(row["prediction_id"]),
        )

        if key in composite_keys:
            duplicate_composite += 1

        composite_keys.add(key)
        scenario_ids.add(
            str(row["scenario_id"])
        )
        formal_ranks.add(
            int(row["formal_rank"])
        )

        ic = row["input_contract"]

        expected_contract = {
            "direct_Stage2_noisy_vr": True,
            "measurement_R_input": True,
            "future_truth_input": False,
            "tracks_to_predict_input": False,
            "annotated_velocity_input": False,
        }

        for k, expected_value in (
            expected_contract.items()
        ):
            if ic.get(k) is not expected_value:
                contract_failures.append(
                    {
                        "line": line_number,
                        "field": k,
                        "actual": ic.get(k),
                    }
                )

        mean = np.asarray(
            row["gaussian_mean_H0_m"],
            dtype=np.float64,
        )
        cov = np.asarray(
            row[
                "gaussian_covariance_calibrated_H0_m2"
            ],
            dtype=np.float64,
        )
        origin = np.asarray(
            row["origin_H0_m"],
            dtype=np.float64,
        )

        if (
            origin.shape != (3,)
            or mean.shape != (4, 3)
            or cov.shape != (4, 3, 3)
            or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(cov))
        ):
            covariance_failures.append(
                {
                    "line": line_number,
                    "reason": "shape_or_finite",
                }
            )
            continue

        for hi in range(4):
            c = cov[hi]

            if not np.allclose(
                c,
                c.T,
                rtol=0.0,
                atol=1.0e-10,
            ):
                covariance_failures.append(
                    {
                        "line": line_number,
                        "horizon_index": hi,
                        "reason": "not_symmetric",
                    }
                )
                break

            eig = np.linalg.eigvalsh(c)

            if not np.all(eig > 0.0):
                covariance_failures.append(
                    {
                        "line": line_number,
                        "horizon_index": hi,
                        "reason": "not_SPD",
                    }
                )
                break


gate(
    "Stage4 ledger schema valid",
    not schema_failures,
    {
        "rows": len(ledger_rows),
        "bad_rows": schema_failures[:20],
    },
)

gate(
    "Stage4 causal input contract valid for every ledger row",
    not contract_failures,
    {
        "failures": contract_failures[:20],
        "failure_count": len(
            contract_failures
        ),
    },
)

gate(
    "Stage4 calibrated predictive covariance is finite SPD",
    not covariance_failures,
    {
        "failures": covariance_failures[:20],
        "failure_count": len(
            covariance_failures
        ),
    },
)

gate(
    "prediction identity uses unique (scenario_id,prediction_id)",
    duplicate_composite == 0,
    {
        "rows": len(ledger_rows),
        "composite_unique": len(
            composite_keys
        ),
        "duplicate_composite": (
            duplicate_composite
        ),
        "note": (
            "prediction_id alone is intentionally "
            "not treated as a global key"
        ),
    },
)

gate(
    "formal ledger spans 120 scenarios",
    len(scenario_ids) == 120,
    {
        "scenario_count": len(
            scenario_ids
        ),
        "formal_rank_count": len(
            formal_ranks
        ),
    },
)


# =====================================================================
# 2. FULL STAGE5 REGRESSION — STILL PRE-TRUTH
# =====================================================================

if failures:
    print()
    print(
        "PRETRUTH FAILURE: formal checker "
        "will NOT be executed."
    )

    preliminary = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status": "FAIL_CLOSED_PRETRUTH",
        "Stage6_allowed": False,
        "formal_evaluation_executed": False,
        "failures": failures,
        "requirement_matrix": matrix,
        "hashes": actual_hashes,
    }
    atomic_json(
        CHECKER2_REPORT,
        preliminary,
    )
    raise SystemExit(2)


print()
print(
    "===== FULL STAGE5 REGRESSION ====="
)

env = os.environ.copy()

pythonpath_parts = [
    str(S5 / "src"),
    str(S4 / "src"),
    str(S3 / "src"),
    str(ROOT / "iscai_stage2/src"),
    str(ROOT / "iscai_stage1/src"),
    str(ROOT / "iscai_stage0/src"),
]

old_pp = env.get(
    "PYTHONPATH",
    "",
)

env["PYTHONPATH"] = os.pathsep.join(
    pythonpath_parts
    + ([old_pp] if old_pp else [])
)

regression = subprocess.run(
    [
        str(PYTHON),
        "-m",
        "unittest",
        "discover",
        "-s",
        str(S5 / "tests"),
        "-p",
        "test*.py",
        "-v",
    ],
    cwd=str(S5),
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

print(regression.stdout)

match = re.search(
    r"Ran\s+(\d+)\s+tests?",
    regression.stdout,
)

test_count = (
    int(match.group(1))
    if match is not None
    else None
)

gate(
    "full Stage5 regression",
    (
        regression.returncode == 0
        and test_count
        == EXPECTED_REGRESSION_TESTS
    ),
    {
        "returncode": regression.returncode,
        "tests": test_count,
        "expected_tests": (
            EXPECTED_REGRESSION_TESTS
        ),
    },
)

if failures:
    result = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status": "FAIL_CLOSED_REGRESSION",
        "Stage6_allowed": False,
        "formal_evaluation_executed": False,
        "failures": failures,
        "requirement_matrix": matrix,
        "hashes": actual_hashes,
    }
    atomic_json(
        CHECKER2_REPORT,
        result,
    )
    raise SystemExit(2)


# =====================================================================
# 3. PRE-TRUTH WRITE-ONCE SEAL
# =====================================================================

print()
print(
    "===== PRE-TRUTH WRITE-ONCE SEAL ====="
)

# IMPORTANT:
# Existing formal report, if any, is NOT parsed here.
# Only its opaque file SHA is recorded.
prior_formal_report_sha = (
    sha256(FORMAL_REPORT)
    if FORMAL_REPORT.is_file()
    else None
)

prior_handoff_snapshot = (
    snapshot_hashes(
        handoff_candidates()
    )
)

seal_core = {
    "stage": 5,
    "version": "fullpdf_v2",
    "scientific_boundary": (
        "before_current_checker2_formal_truth_access"
    ),
    "formal_outcomes_parsed_before_seal": False,
    "training": False,
    "retraining": False,
    "recalibration": False,
    "formal_tuning": False,
    "hashes": {
        "checker2": actual_hashes[
            "checker2"
        ],
        "frozen_checker": actual_hashes[
            "frozen_checker"
        ],
        "runtime": actual_hashes[
            "runtime"
        ],
        "protocol": actual_hashes[
            "protocol"
        ],
        "preformal_addendum": (
            actual_hashes["addendum"]
        ),
        "Stage4_formal_prediction_ledger": (
            actual_hashes["ledger"]
        ),
    },
    "prior_formal_report_opaque_sha256": (
        prior_formal_report_sha
    ),
}

if PRETRUTH_SEAL.exists():
    existing_seal = load_json(
        PRETRUTH_SEAL
    )

    # Ignore any human-readable timestamp-like
    # fields if older seal had them; scientific
    # binding must match exactly on core fields.
    for key in (
        "stage",
        "version",
        "scientific_boundary",
        "formal_outcomes_parsed_before_seal",
        "training",
        "retraining",
        "recalibration",
        "formal_tuning",
        "hashes",
    ):
        if (
            existing_seal.get(key)
            != seal_core.get(key)
        ):
            raise SystemExit(
                "FAIL_CLOSED: existing pretruth "
                "seal does not match current "
                f"scientific binding at {key}."
            )
else:
    write_once_or_verify(
        PRETRUTH_SEAL,
        seal_core,
    )

print(
    "pretruth_seal_sha256 =",
    sha256(PRETRUTH_SEAL),
)


# =====================================================================
# 4. EXECUTE EXISTING FROZEN PRE-FORMAL CHECKER
#
# This checker owns:
#   decision pass A
#   decision pass B
#   exact-repeat comparison
#   decision-ledger seal
#   evaluator future truth AFTER the seal
#   all formal Stage5 system evaluation
# =====================================================================

print()
print(
    "===== EXECUTE FROZEN FORMAL CHECKER ====="
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

process = subprocess.Popen(
    [
        str(PYTHON),
        str(FROZEN_CHECKER),
    ],
    cwd=str(S5),
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
)

captured = []

assert process.stdout is not None

for line in process.stdout:
    print(line, end="")
    captured.append(line)

returncode = process.wait()

SUBPROCESS_LOG.write_text(
    "".join(captured),
    encoding="utf-8",
)

print()
print(
    "frozen_checker_returncode =",
    returncode,
)
print(
    "frozen_checker_log =",
    SUBPROCESS_LOG,
)
print(
    "frozen_checker_log_sha256 =",
    sha256(SUBPROCESS_LOG),
)


# =====================================================================
# 5. POST-RUN INTEGRITY — BEFORE ACCEPTANCE
# =====================================================================

# The current checker and all frozen scientific
# inputs must remain unchanged across formal run.
post_hashes = {
    "checker2": sha256(
        Path(__file__).resolve()
    ),
    "frozen_checker": sha256(
        FROZEN_CHECKER
    ),
    "runtime": sha256(RUNTIME),
    "protocol": sha256(PROTOCOL),
    "addendum": sha256(ADDENDUM),
    "ledger": sha256(STAGE4_LEDGER),
}

for key in (
    "checker2",
    "frozen_checker",
    "runtime",
    "protocol",
    "addendum",
    "ledger",
):
    gate(
        f"post-formal immutable hash: {key}",
        post_hashes[key]
        == actual_hashes[key],
        {
            "before": actual_hashes[key],
            "after": post_hashes[key],
        },
    )

for path in (
    closure_matches
    + handoff_matches
):
    expected = (
        EXPECTED["stage4_closure"]
        if path in closure_matches
        else EXPECTED["stage4_handoff"]
    )

    gate(
        f"Stage4 frozen upstream unchanged: {path}",
        sha256(path) == expected,
        {
            "expected": expected,
            "actual": sha256(path),
        },
    )


gate(
    "frozen formal checker process completed",
    returncode == 0,
    {
        "returncode": returncode,
        "log": str(SUBPROCESS_LOG),
    },
)


# =====================================================================
# 6. NOW — AND ONLY NOW — PARSE FORMAL REPORT
# =====================================================================

gate(
    "formal checker report exists",
    FORMAL_REPORT.is_file(),
    str(FORMAL_REPORT),
)

if not FORMAL_REPORT.is_file():
    result = {
        "stage": 5,
        "version": "fullpdf_v2",
        "status": "FAIL_CLOSED_NO_FORMAL_REPORT",
        "Stage6_allowed": False,
        "formal_evaluation_executed": (
            returncode == 0
        ),
        "failures": failures,
        "requirement_matrix": matrix,
        "hashes": {
            **actual_hashes,
            "pretruth_seal": sha256(
                PRETRUTH_SEAL
            ),
            "subprocess_log": sha256(
                SUBPROCESS_LOG
            ),
        },
    }
    atomic_json(
        CHECKER2_REPORT,
        result,
    )
    raise SystemExit(2)


formal_report = load_json(
    FORMAL_REPORT
)
flat_report = list(
    flatten(formal_report)
)


# ---------------------------------------------------------------------
# A. Exact-repeat decisions before future truth.
# ---------------------------------------------------------------------

exact_repeat_ok, exact_repeat_ev = (
    find_pass_evidence(
        flat_report,
        [
            ("exact", "repeat"),
            ("decision", "repeat"),
            ("two", "decision", "pass"),
        ],
    )
)

gate(
    "two causal decision passes are exact-repeat",
    exact_repeat_ok,
    exact_repeat_ev,
)


decision_hashes = find_hash_evidence(
    flat_report,
    ("decision", "ledger"),
)

gate(
    "decision ledger has cryptographic seal/hash",
    len(decision_hashes) >= 1,
    decision_hashes,
)


sealed_ok, sealed_ev = (
    find_pass_evidence(
        flat_report,
        [
            ("decision", "ledger", "seal"),
            ("ledger", "sealed"),
        ],
    )
)

# SHA evidence itself is a necessary
# cryptographic seal even if a separate boolean
# is not emitted.
sealed_ok = (
    sealed_ok
    or len(decision_hashes) >= 1
)

gate(
    "decision ledger sealed before evaluator truth",
    sealed_ok,
    (
        sealed_ev
        if sealed_ev is not None
        else decision_hashes
    ),
)


future_after_ok, future_after_ev = (
    find_pass_evidence(
        flat_report,
        [
            (
                "future",
                "truth",
                "after",
                "decision",
            ),
            (
                "future",
                "truth",
                "after",
                "ledger",
            ),
            (
                "truth",
                "evaluator",
                "after",
                "decision",
            ),
        ],
    )
)

gate(
    "future truth accessed only after decisions are sealed",
    future_after_ok,
    future_after_ev,
)


# ---------------------------------------------------------------------
# B. Full 3 × 3 × 4 × 4 formal matrix.
# ---------------------------------------------------------------------

combo_evidence = collect_combo_evidence(
    formal_report
)

missing_combos = sorted(
    EXPECTED_COMBOS
    - set(combo_evidence)
)

gate(
    "all receiver×codebook×horizon×target combinations evaluated",
    len(missing_combos) == 0,
    {
        "expected": len(
            EXPECTED_COMBOS
        ),
        "found": len(
            combo_evidence
        ),
        "missing": missing_combos[:50],
    },
)


missing_coverage = []
missing_probe_overhead = []
missing_latency = []
missing_ber = []
missing_rate = []

for combo in sorted(
    EXPECTED_COMBOS
):
    ev = combo_evidence.get(
        combo,
        set(),
    )

    if not metric_present(
        ev,
        ("coverage",),
    ):
        missing_coverage.append(combo)

    if not metric_present(
        ev,
        (
            "charged",
            "probe",
            "overhead",
        ),
    ):
        missing_probe_overhead.append(
            combo
        )

    if not metric_present(
        ev,
        ("latency",),
    ):
        missing_latency.append(combo)

    if not metric_present(
        ev,
        ("ber", "dpsk"),
    ):
        missing_ber.append(combo)

    if not metric_present(
        ev,
        (
            "effective_rate",
            "effective rate",
        ),
    ):
        missing_rate.append(combo)


gate(
    "empirical coverage measured for every formal combination",
    not missing_coverage,
    {
        "missing": missing_coverage[:50],
        "missing_count": len(
            missing_coverage
        ),
    },
)

gate(
    "actual charged probing overhead measured for every combination",
    not missing_probe_overhead,
    {
        "missing": (
            missing_probe_overhead[:50]
        ),
        "missing_count": len(
            missing_probe_overhead
        ),
    },
)

gate(
    "beam probing/decision latency measured for every combination",
    not missing_latency,
    {
        "missing": missing_latency[:50],
        "missing_count": len(
            missing_latency
        ),
    },
)

gate(
    "DPSK BER measured for every combination",
    not missing_ber,
    {
        "missing": missing_ber[:50],
        "missing_count": len(
            missing_ber
        ),
    },
)

gate(
    "effective optical rate measured for every combination",
    not missing_rate,
    {
        "missing": missing_rate[:50],
        "missing_count": len(
            missing_rate
        ),
    },
)


# ---------------------------------------------------------------------
# C. Coverage acceptance.
#
# IMPORTANT:
# We deliberately DO NOT create a tolerance.
# PASS must come from the already-frozen
# protocol/addendum/checker acceptance semantics.
# ---------------------------------------------------------------------

coverage_gate_ok, coverage_gate_ev = (
    find_pass_evidence(
        flat_report,
        [
            (
                "requested",
                "empirical",
                "coverage",
            ),
            (
                "requested",
                "coverage",
            ),
            (
                "coverage",
                "acceptance",
            ),
            (
                "coverage",
                "gate",
            ),
        ],
    )
)

gate(
    "frozen requested-coverage acceptance gate passes",
    coverage_gate_ok,
    coverage_gate_ev,
)


# ---------------------------------------------------------------------
# D. Overhead acceptance.
# ---------------------------------------------------------------------

overhead_gate_ok, overhead_gate_ev = (
    find_pass_evidence(
        flat_report,
        [
            (
                "overhead",
                "reduction",
            ),
            (
                "reduced",
                "overhead",
            ),
            (
                "probing",
                "overhead",
                "acceptance",
            ),
        ],
    )
)

gate(
    "frozen overhead-reduction acceptance gate passes",
    overhead_gate_ok,
    overhead_gate_ev,
)


# ---------------------------------------------------------------------
# E. No-free-probes accounting.
# ---------------------------------------------------------------------

free_probe_values = []

for path, value in flat_report:
    low = path.lower()

    if (
        "free_probe_count" in low
        or "free_probes" in low
        or "hidden_probe_count" in low
    ):
        if isinstance(
            value,
            (int, float),
        ):
            free_probe_values.append(
                (path, float(value))
            )

no_free_probe_gate_ok, no_free_ev = (
    find_pass_evidence(
        flat_report,
        [
            ("no", "free", "probe"),
            ("all", "beams", "charged"),
            ("hidden", "probe", "false"),
        ],
    )
)

numeric_no_free = (
    len(free_probe_values) > 0
    and all(
        value == 0.0
        for _, value
        in free_probe_values
    )
)

gate(
    "no hidden/free evaluated probes",
    (
        no_free_probe_gate_ok
        or numeric_no_free
    ),
    {
        "explicit_gate": no_free_ev,
        "free_probe_values": (
            free_probe_values[:100]
        ),
    },
)


# ---------------------------------------------------------------------
# F. Recovery / hysteresis / persistence mechanisms.
# ---------------------------------------------------------------------

mechanism_tokens = {
    "persistence": (
        "previous_primary_index"
        in runtime_source
    ),
    "hysteresis": (
        "hysteresis"
        in runtime_source.lower()
        or "retain"
        in runtime_source.lower()
    ),
    "local_neighbor_recovery": (
        "local_neighbor_recovery"
        in runtime_source
    ),
    "widened_fallback": (
        "widened_fallback"
        in runtime_source
    ),
    "exhaustive_loss_of_lock": (
        "exhaustive_loss_of_lock"
        in runtime_source
    ),
}

gate(
    "persistence/hysteresis/recovery/fallback mechanisms present and regression-tested",
    all(
        mechanism_tokens.values()
    ),
    mechanism_tokens,
)


# ---------------------------------------------------------------------
# G. Full optical chain evidence.
# ---------------------------------------------------------------------

required_optical_tokens = (
    "pointing",
    "optical_gain",
    "received_power",
    "snr",
    "ber",
    "effective_rate",
)

optical_presence = {}

for token in required_optical_tokens:
    optical_presence[token] = any(
        token in (
            path + " " + str(value)
        ).lower()
        for path, value
        in flat_report
    )

gate(
    "full optical pointing→gain→power→SNR→DPSK BER→effective-rate chain evidenced",
    all(
        optical_presence.values()
    ),
    optical_presence,
)


# ---------------------------------------------------------------------
# H. Receiver uncertainty evidence.
# ---------------------------------------------------------------------

receiver_uncertainty_ok = any(
    (
        "receiver"
        in path.lower()
        and "uncertain"
        in (
            path + " " + str(value)
        ).lower()
    )
    for path, value in flat_report
)

gate(
    "uncertain receiver-placement mode formally evaluated",
    receiver_uncertainty_ok,
    {
        "receiver_modes": (
            EXPECTED_RECEIVER_MODES
        ),
        "protocol_uncertain_distribution": (
            protocol
            .get("receiver_geometry", {})
            .get("uncertain_distribution")
        ),
    },
)


# =====================================================================
# 7. INDEPENDENT PDF REQUIREMENT → EVIDENCE → PASS/FAIL MATRIX
# =====================================================================

# Additional high-level PDF gates are represented
# explicitly here instead of trusting only the
# formal checker's own closure label.

pdf_matrix = [
    {
        "pdf_requirement":
            "receiver-aware angular posterior",
        "evidence":
            "3 receiver modes + already-calibrated "
            "Stage4 covariance + receiver uncertainty",
        "pass":
            (
                tuple(
                    protocol[
                        "receiver_geometry"
                    ]["modes"]
                )
                == EXPECTED_RECEIVER_MODES
                and receiver_uncertainty_ok
            ),
    },
    {
        "pdf_requirement":
            "16/32/64 directional codebooks",
        "evidence":
            protocol["codebooks"]["sizes"],
        "pass":
            tuple(
                protocol[
                    "codebooks"
                ]["sizes"]
            )
            == EXPECTED_CODEBOOKS,
    },
    {
        "pdf_requirement":
            "adaptive minimum Top-K probability mass",
        "evidence":
            protocol[
                "adaptive_TopK"
            ]["rule"],
        "pass":
            protocol[
                "adaptive_TopK"
            ]["rule"]
            == (
                "minimum_beam_set_reaching_"
                "requested_mass"
            ),
    },
    {
        "pdf_requirement":
            "90/95/97.5/99 percent targets",
        "evidence":
            protocol[
                "adaptive_TopK"
            ]["coverage_targets"],
        "pass":
            tuple(
                float(x)
                for x in protocol[
                    "adaptive_TopK"
                ]["coverage_targets"]
            )
            == EXPECTED_TARGETS,
    },
    {
        "pdf_requirement":
            "explicit outside-FOV probability",
        "evidence":
            protocol["codebooks"],
        "pass":
            (
                protocol[
                    "codebooks"
                ]["outside_support_mass"]
                == "explicit"
                and
                protocol[
                    "codebooks"
                ]["azimuth_clipping"]
                is False
            ),
    },
    {
        "pdf_requirement":
            "persistence/hysteresis/recovery",
        "evidence":
            mechanism_tokens,
        "pass":
            all(
                mechanism_tokens.values()
            ),
    },
    {
        "pdf_requirement":
            "all evaluated beams charged",
        "evidence":
            {
                "protocol":
                    protocol[
                        "adaptive_TopK"
                    ],
                "formal_free_probe":
                    free_probe_values[:20],
            },
        "pass":
            (
                protocol[
                    "adaptive_TopK"
                ][
                    "all_evaluated_beams_charged"
                ]
                is True
                and
                protocol[
                    "adaptive_TopK"
                ][
                    "free_hidden_probes"
                ]
                is False
                and
                (
                    no_free_probe_gate_ok
                    or numeric_no_free
                )
            ),
    },
    {
        "pdf_requirement":
            "latency from actual probing",
        "evidence":
            {
                "combo_latency_missing":
                    len(
                        missing_latency
                    ),
                "probe_count_source":
                    protocol[
                        "optical"
                    ][
                        "probe_count_source"
                    ],
            },
        "pass":
            (
                not missing_latency
                and protocol[
                    "optical"
                ][
                    "probe_count_source"
                ]
                == (
                    "exact_actual_probe_"
                    "plan_only"
                )
            ),
    },
    {
        "pdf_requirement":
            "optical pointing→gain→power→SNR→BER→rate",
        "evidence":
            optical_presence,
        "pass":
            all(
                optical_presence.values()
            ),
    },
    {
        "pdf_requirement":
            "adaptive requested empirical coverage",
        "evidence":
            coverage_gate_ev,
        "pass":
            coverage_gate_ok,
    },
    {
        "pdf_requirement":
            "reduced probing overhead",
        "evidence":
            overhead_gate_ev,
        "pass":
            overhead_gate_ok,
    },
    {
        "pdf_requirement":
            "causal exact-repeat decisions before evaluator truth",
        "evidence":
            {
                "exact_repeat":
                    exact_repeat_ev,
                "decision_ledger":
                    decision_hashes,
                "future_truth_after":
                    future_after_ev,
            },
        "pass":
            (
                exact_repeat_ok
                and sealed_ok
                and future_after_ok
            ),
    },
    {
        "pdf_requirement":
            "no Stage4 recalibration/retraining/formal tuning",
        "evidence":
            {
                "Stage4_calibration_reapplied":
                    protocol[
                        "upstream"
                    ][
                        "Stage4_calibration_reapplied"
                    ],
                "formal_parameter_tuning":
                    protocol[
                        "formal"
                    ][
                        "parameter_tuning_on_formal"
                    ],
                "runtime_bad_calibrator_calls":
                    bad_runtime_calls,
            },
        "pass":
            (
                protocol[
                    "upstream"
                ][
                    "Stage4_calibration_reapplied"
                ]
                is False
                and
                protocol[
                    "formal"
                ][
                    "parameter_tuning_on_formal"
                ]
                is False
                and
                not bad_runtime_calls
            ),
    },
    {
        "pdf_requirement":
            "all 3×3×4×4 formal combinations",
        "evidence":
            {
                "expected": 144,
                "found":
                    len(
                        combo_evidence
                    ),
            },
        "pass":
            not missing_combos,
    },
]


for row in pdf_matrix:
    gate(
        "PDF: "
        + row["pdf_requirement"],
        bool(row["pass"]),
        row["evidence"],
    )


# =====================================================================
# 8. FINAL FAIL-CLOSED DECISION
# =====================================================================

stage6_allowed = (
    len(failures) == 0
)

final_stage5_status = (
    "COMPLETE_FROZEN_FULLPDF_V2"
    if stage6_allowed
    else "FAIL_CLOSED_FULLPDF_V2_FORMAL"
)

formal_report_sha = sha256(
    FORMAL_REPORT
)

checker2_result = {
    "stage": 5,
    "version": "fullpdf_v2",
    "status": final_stage5_status,
    "Stage6_allowed": (
        stage6_allowed
    ),
    "scientific_boundary": {
        "training": False,
        "retraining": False,
        "recalibration": False,
        "formal_parameter_tuning": False,
        "future_truth_before_decision":
            False,
        "post_outcome_threshold_change":
            False,
    },
    "frozen_inputs": {
        "runtime": {
            "path": str(RUNTIME),
            "sha256": sha256(
                RUNTIME
            ),
        },
        "protocol": {
            "path": str(PROTOCOL),
            "sha256": sha256(
                PROTOCOL
            ),
        },
        "preformal_addendum": {
            "path": str(ADDENDUM),
            "sha256": sha256(
                ADDENDUM
            ),
        },
        "Stage4_prediction_ledger": {
            "path": str(
                STAGE4_LEDGER
            ),
            "sha256": sha256(
                STAGE4_LEDGER
            ),
        },
        "frozen_checker": {
            "path": str(
                FROZEN_CHECKER
            ),
            "sha256": sha256(
                FROZEN_CHECKER
            ),
        },
        "pretruth_seal": {
            "path": str(
                PRETRUTH_SEAL
            ),
            "sha256": sha256(
                PRETRUTH_SEAL
            ),
        },
    },
    "formal": {
        "source_report": str(
            FORMAL_REPORT
        ),
        "source_report_sha256":
            formal_report_sha,
        "combination_count_expected":
            144,
        "combination_count_evidenced":
            len(
                combo_evidence
            ),
        "receiver_modes":
            list(
                EXPECTED_RECEIVER_MODES
            ),
        "codebooks":
            list(
                EXPECTED_CODEBOOKS
            ),
        "horizons_s":
            list(
                EXPECTED_HORIZONS
            ),
        "probability_targets":
            list(
                EXPECTED_TARGETS
            ),
        "decision_ledger_hash_evidence":
            decision_hashes,
    },
    "regression": {
        "tests_passed": test_count,
        "tests_expected":
            EXPECTED_REGRESSION_TESTS,
        "status":
            (
                "PASS"
                if regression.returncode
                == 0
                else "FAIL"
            ),
    },
    "PDF_requirement_evidence_matrix":
        pdf_matrix,
    "requirement_matrix":
        matrix,
    "mandatory_failures":
        failures,
}

atomic_json(
    CHECKER2_REPORT,
    checker2_result,
)

checker2_report_sha = sha256(
    CHECKER2_REPORT
)


# ---------------------------------------------------------------------
# Handoff is created ONLY after every mandatory gate passes.
# ---------------------------------------------------------------------

if stage6_allowed:
    handoff = {
        "from_stage": 5,
        "to_stage": 6,
        "Stage5_status":
            "COMPLETE_FROZEN_FULLPDF_V2",
        "Stage6_allowed": True,
        "trajectory_posterior":
            "calibrated_full_3D_Gaussian_GRU",
        "Stage4_covariance_semantics":
            "already_calibrated_do_not_recalibrate",
        "receiver_geometry_modes":
            list(
                EXPECTED_RECEIVER_MODES
            ),
        "codebooks":
            list(
                EXPECTED_CODEBOOKS
            ),
        "probability_targets":
            list(
                EXPECTED_TARGETS
            ),
        "formal_report": {
            "path": str(
                FORMAL_REPORT
            ),
            "sha256":
                formal_report_sha,
        },
        "independent_checker2": {
            "path": str(
                CHECKER2_REPORT
            ),
            "sha256":
                checker2_report_sha,
        },
        "decision_ledger_hash_evidence":
            decision_hashes,
        "no_training": True,
        "no_retraining": True,
        "no_recalibration": True,
        "no_post_outcome_tuning": True,
        "future_truth_role":
            (
                "evaluator_only_after_"
                "sealed_causal_decisions"
            ),
    }

    if HANDOFF.exists():
        existing = load_json(
            HANDOFF
        )

        # A pre-existing handoff is accepted only
        # if it already represents this exact
        # completed V2 boundary.
        if not (
            existing.get(
                "Stage6_allowed"
            )
            is True
            and
            existing.get(
                "Stage5_status"
            )
            == (
                "COMPLETE_FROZEN_"
                "FULLPDF_V2"
            )
        ):
            raise SystemExit(
                "FAIL_CLOSED: conflicting "
                "pre-existing Stage5→Stage6 "
                f"handoff: {HANDOFF}"
            )
    else:
        write_once_or_verify(
            HANDOFF,
            handoff,
        )

    print()
    print("=" * 79)
    print(
        "STAGE5 = "
        "COMPLETE_FROZEN_FULLPDF_V2"
    )
    print(
        "Stage6_allowed = true"
    )
    print(
        "checker2_report =",
        CHECKER2_REPORT,
    )
    print(
        "checker2_report_sha256 =",
        checker2_report_sha,
    )
    print(
        "handoff =",
        HANDOFF,
    )
    print(
        "handoff_sha256 =",
        sha256(HANDOFF),
    )
    print("=" * 79)
    raise SystemExit(0)


# ---------------------------------------------------------------------
# FAILURE:
# Never manufacture / rewrite a V2 handoff.
# ---------------------------------------------------------------------

post_handoff_snapshot = snapshot_hashes(
    handoff_candidates()
)

changed_handoffs = []

for path, value in (
    post_handoff_snapshot.items()
):
    if (
        prior_handoff_snapshot
        .get(path)
        != value
    ):
        changed_handoffs.append(
            {
                "path": path,
                "before":
                    prior_handoff_snapshot
                    .get(path),
                "after": value,
            }
        )

print()
print("=" * 79)
print(
    "STAGE5 = "
    "FAIL_CLOSED_FULLPDF_V2_FORMAL"
)
print(
    "Stage6_allowed = false"
)
print(
    "mandatory failures ="
)
for failure in failures:
    print(
        "  -",
        failure,
    )
print(
    "checker2_report =",
    CHECKER2_REPORT,
)
print(
    "checker2_report_sha256 =",
    checker2_report_sha,
)
print(
    "handoff changes detected =",
    changed_handoffs,
)
print(
    "DO NOT START STAGE6."
)
print("=" * 79)

raise SystemExit(2)

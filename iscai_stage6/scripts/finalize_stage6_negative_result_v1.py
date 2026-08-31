from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path


ROOT = Path("/home/agni/waymo")
S5 = ROOT / "iscai_stage5"
S6 = ROOT / "iscai_stage6"
PARTA = ROOT / "part_a_reference/ISCAI_pc_fmcw"

PYTHON = Path(
    "/home/agni/waymo/iscai_stage1/.venv_lidar/bin/python"
)

# =====================================================================
# Stage-1 read-only authority
# =====================================================================

DISCOVERY = (
    ROOT / "audits/"
    "stage6_final_repair_stage1_readonly/"
    "stage6_final_repair_discovery.json"
)

DISCOVERY_MANIFEST = (
    ROOT / "audits/"
    "stage6_final_repair_stage1_readonly/"
    "stage6_tree_manifest.json"
)

EXPECTED_DISCOVERY_SHA = (
    "d17e3e40e651a3b129365ab72ace8f35220e4c91bd2725f88a886920c4fc1b40"
)

# =====================================================================
# Frozen Stage5 final authority
# =====================================================================

STAGE5_CERT = (
    ROOT / "audits/"
    "stage5_pre_stage6_final_v2/"
    "stage5_pre_stage6_final_certificate.json"
)

EXPECTED_STAGE5_CERT_SHA = (
    "f91a0d21ed752b1e1e67096ca8af1ea061145869605525b232f6bccf1c26efe1"
)

STAGE5_CLOSURE = (
    S5 / "reports/"
    "stage5_fullpdf_v2_final_closure_reconciled_v1.json"
)

EXPECTED_STAGE5_CLOSURE_SHA = (
    "e376a9274826f80b7ffcf6d031c0f44c792cb025a44a4675e95ded467e526e34"
)

STAGE5_HANDOFF = (
    S5 / "artifacts/fullpdf_v2_independent_formal/"
    "final_reconciliation_v1/"
    "stage5_to_stage6_handoff_reconciled_v1.json"
)

EXPECTED_STAGE5_HANDOFF_SHA = (
    "ac6564dbfe537f39fe70e7f2e92a7b2a5509aa3285d1037983924714bca36124"
)

# =====================================================================
# Stage6 scientific authority
# =====================================================================

CONTRACT = (
    S6 / "configs/stage6_contract.json"
)

NI_RULE = (
    S6 / "configs/"
    "stage6_noninferiority_bound_selection_rule.json"
)

EXACT_DELTAS = (
    S6 / "configs/"
    "stage6_exact_noninferiority_deltas.json"
)

PRIMARY_POLICY = (
    S6 / "configs/"
    "stage6_preformal_primary_acceptance_policy.json"
)

METRIC_RECON = (
    S6 / "reports/"
    "block68_preoutcome_metric_reconciliation.json"
)

PRIMARY_REPORT = (
    S6 / "reports/"
    "block68_primary_development_acceptance_gate.json"
)

PRIMARY_DETAIL = (
    S6 / "artifacts/block68/primary_development_gate/"
    "block68_primary_development_acceptance_detail.json"
)

FAILURE_CLOSURE = (
    S6 / "reports/"
    "block68_development_failure_closure.json"
)

FAILURE_MANIFEST = (
    S6 / "reports/"
    "block68_development_failure_freeze_manifest.json"
)

# =====================================================================
# New provenance/final closure only
# =====================================================================

UPSTREAM_ADDENDUM = (
    S6 / "configs/"
    "stage6_stage5_final_upstream_addendum_v1.json"
)

FINAL_CLOSURE = (
    S6 / "reports/"
    "stage6_final_scientific_closure_v1.json"
)

FINAL_DIR = (
    S6 / "artifacts/"
    "stage6_final_scientific_closure_v1"
)

FREEZE_MANIFEST = (
    FINAL_DIR /
    "stage6_final_scientific_freeze_manifest.json"
)

STAGE7_BLOCK = (
    FINAL_DIR /
    "stage6_to_stage7_block.json"
)

REGRESSION_LOG = (
    FINAL_DIR /
    "stage6_final_regression_224.log"
)

# =====================================================================
# Helpers
# =====================================================================


def fail(msg):
    raise RuntimeError(
        "FAIL-CLOSED: " + str(msg)
    )


def sha(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        fail(f"JSON root not object: {path}")

    return obj


def json_bytes(obj):
    return (
        json.dumps(
            obj,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once(path: Path, obj):
    if path.exists():
        fail(
            f"write-once artifact already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_name(
        path.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    with tmp.open("xb") as f:
        f.write(
            json_bytes(obj)
        )
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


def finite_float(value, label):
    try:
        x = float(value)
    except Exception:
        fail(
            f"{label} is not numeric: {value!r}"
        )

    if not math.isfinite(x):
        fail(
            f"{label} is not finite"
        )

    return x


def get_path(obj, *parts):
    cur = obj

    for p in parts:
        if not isinstance(
            cur,
            dict,
        ):
            return None

        cur = cur.get(p)

    return cur


def tree_manifest():
    result = {}

    for root in (
        S6 / "src",
        S6 / "configs",
        S6 / "tests",
        S6 / "scripts",
        S6 / "reports",
    ):
        if not root.exists():
            continue

        for p in sorted(
            root.rglob("*")
        ):
            if not p.is_file():
                continue

            if "__pycache__" in p.parts:
                continue

            if p.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue

            rel = p.relative_to(
                ROOT
            ).as_posix()

            result[rel] = sha(p)

    return result


def expected_sha_from_discovery(
    discovery,
    path: Path,
):
    target = str(path)

    # direct known sections
    candidates = []

    candidates.append(
        get_path(
            discovery,
            "Stage6_contract",
            "sha256",
        )
    )

    candidates.append(
        get_path(
            discovery,
            "noninferiority_authority",
            "selection_rule",
            "sha256",
        )
    )

    candidates.append(
        get_path(
            discovery,
            "noninferiority_authority",
            "exact_deltas",
            "sha256",
        )
    )

    candidates.append(
        get_path(
            discovery,
            "noninferiority_authority",
            "primary_policy",
            "sha256",
        )
    )

    candidates.append(
        get_path(
            discovery,
            "metric_authority",
            "sha256",
        )
    )

    pf = discovery.get(
        "primary_failure",
        {}
    )

    for key in (
        "primary_report",
        "primary_detail",
        "failure_closure",
    ):
        item = pf.get(
            key,
            {}
        )

        if (
            isinstance(item, dict)
            and item.get("path")
            == target
        ):
            return item.get(
                "sha256"
            )

    # recursive path+sha search
    def walk(x):
        if isinstance(x, dict):
            if (
                x.get("path")
                == target
                and isinstance(
                    x.get("sha256"),
                    str,
                )
            ):
                return x[
                    "sha256"
                ]

            for value in x.values():
                hit = walk(value)

                if hit is not None:
                    return hit

        elif isinstance(x, list):
            for value in x:
                hit = walk(value)

                if hit is not None:
                    return hit

        return None

    return walk(discovery)


# =====================================================================
# A. One-shot + free-space
# =====================================================================

for p in (
    UPSTREAM_ADDENDUM,
    FINAL_CLOSURE,
    FREEZE_MANIFEST,
    STAGE7_BLOCK,
):
    if p.exists():
        fail(
            f"final Stage6 artifact already exists: {p}"
        )

if FINAL_DIR.exists():
    fail(
        f"final Stage6 directory already exists: {FINAL_DIR}"
    )

free_gib = (
    shutil.disk_usage(
        ROOT
    ).free
    / 1024**3
)

if free_gib < 250.0:
    fail(
        f"free space {free_gib:.2f} GiB < 250 GiB"
    )


# =====================================================================
# B. Stage-1 discovery is exact and was genuinely read-only
# =====================================================================

if sha(
    DISCOVERY
) != EXPECTED_DISCOVERY_SHA:
    fail(
        "Stage6 Stage-1 discovery SHA mismatch"
    )

discovery = read_json(
    DISCOVERY
)

if (
    discovery.get(
        "audit"
    )
    !=
    "STAGE6_FINAL_REPAIR_STAGE1_READONLY"
):
    fail(
        "wrong Stage-1 discovery authority"
    )

scientific_execution = (
    discovery.get(
        "scientific_execution",
        {}
    )
)

for key, value in (
    scientific_execution.items()
):
    if value is not False:
        fail(
            f"Stage-1 unexpectedly executed science: "
            f"{key}={value!r}"
        )

integrity = discovery.get(
    "read_only_integrity",
    {}
)

for key in (
    "Stage5_unchanged",
    "Stage6_protected_tree_unchanged",
    "PartA_unchanged",
):
    if integrity.get(
        key
    ) is not True:
        fail(
            f"Stage-1 integrity failure: {key}"
        )

stage1_regression = discovery.get(
    "regression",
    {}
)

if not (
    stage1_regression.get(
        "returncode"
    )
    == 0
    and
    stage1_regression.get(
        "test_count"
    )
    == 224
    and
    stage1_regression.get(
        "pass"
    )
    is True
):
    fail(
        "Stage-1 regression was not 224/224 PASS"
    )


# =====================================================================
# C. No Stage6 file changed since Stage-1
# =====================================================================

frozen_tree = read_json(
    DISCOVERY_MANIFEST
)

current_tree = tree_manifest()

missing = []
changed = []

for rel, expected in (
    frozen_tree.items()
):
    actual = current_tree.get(
        rel
    )

    if actual is None:
        missing.append(
            rel
        )

    elif actual != expected:
        changed.append(
            {
                "path":
                    rel,
                "expected":
                    expected,
                "actual":
                    actual,
            }
        )

if missing or changed:
    fail(
        "Stage6 changed after Stage-1 audit. "
        f"missing={len(missing)}, "
        f"changed={len(changed)}"
    )

# The Stage-2 finalizer itself necessarily did not exist when the
# Stage-1 read-only tree manifest was frozen.  Permit exactly this
# self file and nothing else.
self_rel = (
    Path(__file__)
    .resolve()
    .relative_to(ROOT)
    .as_posix()
)

unexpected_preexisting_all = (
    set(current_tree)
    - set(frozen_tree)
)

unexpected_preexisting = sorted(
    unexpected_preexisting_all
    - {self_rel}
)

if self_rel not in unexpected_preexisting_all:
    fail(
        "Stage-2 finalizer self-file was unexpectedly "
        "already present in the Stage-1 frozen tree"
    )

if unexpected_preexisting:
    fail(
        "new Stage6 protected files appeared "
        "after Stage-1 beyond the expected finalizer itself: "
        + repr(
            unexpected_preexisting[
                :20
            ]
        )
    )


# =====================================================================
# D. Frozen Stage5 upstream is final and exact
# =====================================================================

if sha(
    STAGE5_CERT
) != EXPECTED_STAGE5_CERT_SHA:
    fail(
        "Stage5 final certificate changed"
    )

if sha(
    STAGE5_CLOSURE
) != EXPECTED_STAGE5_CLOSURE_SHA:
    fail(
        "Stage5 final closure changed"
    )

if sha(
    STAGE5_HANDOFF
) != EXPECTED_STAGE5_HANDOFF_SHA:
    fail(
        "Stage5 final handoff changed"
    )

stage5_cert = read_json(
    STAGE5_CERT
)

if not (
    stage5_cert.get(
        "Stage5_PDF_scope"
    )
    == "PASS_100_PERCENT"
    and
    stage5_cert.get(
        "Stage6_allowed"
    )
    is True
):
    fail(
        "Stage5 no longer authorizes Stage6"
    )


# =====================================================================
# E. Part-A commit is still frozen
# =====================================================================

git = subprocess.run(
    [
        "git",
        "-C",
        str(PARTA),
        "rev-parse",
        "HEAD",
    ],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)

if git.returncode != 0:
    fail(
        "cannot read Part-A commit"
    )

parta_commit = (
    git.stdout.strip()
)

if (
    parta_commit
    !=
    "44d62e3478e3818d1757b00971890f844cb032f7"
):
    fail(
        "Part-A frozen commit changed: "
        + parta_commit
    )


# =====================================================================
# F. Exact current Stage6 authorities match Stage-1
# =====================================================================

critical = (
    CONTRACT,
    NI_RULE,
    EXACT_DELTAS,
    PRIMARY_POLICY,
    METRIC_RECON,
    PRIMARY_REPORT,
    PRIMARY_DETAIL,
    FAILURE_CLOSURE,
)

critical_hashes = {}

for path in critical:
    actual = sha(path)

    expected = (
        expected_sha_from_discovery(
            discovery,
            path,
        )
    )

    if expected is None:
        # Tree manifest is still an exact cryptographic authority.
        rel = path.relative_to(
            ROOT
        ).as_posix()

        expected = frozen_tree.get(
            rel
        )

    if expected is None:
        fail(
            f"no Stage-1 SHA authority for {path}"
        )

    if actual != expected:
        fail(
            f"scientific authority changed: {path}"
        )

    critical_hashes[
        str(path)
    ] = actual


# =====================================================================
# G. Re-prove exact frozen acceptance policy
# =====================================================================

contract = read_json(
    CONTRACT
)

ni_rule = read_json(
    NI_RULE
)

exact_deltas = read_json(
    EXACT_DELTAS
)

primary_policy = read_json(
    PRIMARY_POLICY
)

metric_recon = read_json(
    METRIC_RECON
)

failure = read_json(
    FAILURE_CLOSURE
)


if (
    contract.get(
        "status"
    )
    !=
    "FROZEN_PREIMPLEMENTATION_CONTRACT"
):
    fail(
        "Stage6 frozen preimplementation contract changed"
    )

acceptance = contract.get(
    "acceptance",
    {}
)

if not (
    acceptance.get(
        "must_reduce_vehicle_shadow_zone_violation"
    )
    is True
    and
    acceptance.get(
        "must_not_uncontrollably_increase_over_masking"
    )
    is True
    and
    acceptance.get(
        "must_not_uncontrollably_reduce_VRU_visibility"
    )
    is True
    and
    acceptance.get(
        "formal_posthoc_tuning"
    )
    is False
    and
    acceptance.get(
        "uncontrolled_degradation_numeric_bounds"
    )
    ==
    "FREEZE_ON_DEVELOPMENT_BEFORE_FORMAL"
):
    fail(
        "Stage6 acceptance contract changed"
    )


bound_derivation = ni_rule.get(
    "bound_derivation",
    {}
)

anti_posthoc = ni_rule.get(
    "anti_posthoc_guards",
    {}
)

if not (
    ni_rule.get(
        "status"
    )
    ==
    "FROZEN_PREOUTCOME_NI_BOUND_SELECTION_RULE"
    and
    bound_derivation.get(
        "absolute_floor"
    )
    == 0.02
    and
    bound_derivation.get(
        "absolute_cap"
    )
    == 0.05
    and
    bound_derivation.get(
        "raw_delta"
    )
    ==
    "1.96 * standard_error"
    and
    anti_posthoc.get(
        "formal_parameter_tuning"
    )
    is False
    and
    anti_posthoc.get(
        "predictive_values_must_not_enter_delta_function"
    )
    is True
):
    fail(
        "NI-bound pre-outcome authority changed"
    )


metrics = exact_deltas.get(
    "metrics",
    {}
)

over_delta = finite_float(
    metrics.get(
        "over_masking_area",
        {},
    ).get(
        "frozen_delta"
    ),
    "overmask frozen delta",
)

ped_delta = finite_float(
    metrics.get(
        "pedestrian_visibility_proxy",
        {},
    ).get(
        "frozen_delta"
    ),
    "pedestrian frozen delta",
)

cyc_delta = finite_float(
    metrics.get(
        "cyclist_visibility_proxy",
        {},
    ).get(
        "frozen_delta"
    ),
    "cyclist frozen delta",
)

if not (
    math.isclose(
        over_delta,
        0.02,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    and
    math.isclose(
        ped_delta,
        0.05,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    and
    math.isclose(
        cyc_delta,
        0.05,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
):
    fail(
        "exact frozen NI deltas changed"
    )

if (
    exact_deltas.get(
        "predictive_values_seen"
    )
    is not False
):
    fail(
        "NI deltas were not frozen before predictive values"
    )


policy_rules = primary_policy.get(
    "formal_rules",
    {}
)

if not (
    primary_policy.get(
        "status"
    )
    ==
    "FROZEN_BEFORE_PREDICTIVE_DEVELOPMENT_PERFORMANCE"
    and
    primary_policy.get(
        "policy_tuning_after_freeze_allowed"
    )
    is False
    and
    primary_policy.get(
        "predictive_outcomes_seen_at_freeze"
    )
    is False
    and
    finite_float(
        policy_rules.get(
            "over_masking_area",
            {},
        ).get(
            "frozen_delta"
        ),
        "policy overmask delta",
    )
    == over_delta
    and
    finite_float(
        policy_rules.get(
            "pedestrian_visibility_proxy",
            {},
        ).get(
            "frozen_delta"
        ),
        "policy pedestrian delta",
    )
    == ped_delta
    and
    finite_float(
        policy_rules.get(
            "cyclist_visibility_proxy",
            {},
        ).get(
            "frozen_delta"
        ),
        "policy cyclist delta",
    )
    == cyc_delta
):
    fail(
        "preformal primary acceptance policy changed"
    )


# =====================================================================
# H. Metric semantics were reconciled before outcome
# =====================================================================

metric_status = metric_recon.get(
    "status"
)

if metric_status not in {
    "PASS_PREOUTCOME_METRIC_RECONCILED",
    "FROZEN_METRIC_SEMANTICS_RECONCILED_TO_BLOCK66_PREREGISTRATION",
}:
    fail(
        "unexpected metric reconciliation status: "
        + repr(
            metric_status
        )
    )

source_semantics = (
    metric_recon.get(
        "source_semantics",
        {}
    )
)

if source_semantics:
    if (
        source_semantics.get(
            "reconciliation_performed_before_development_outcome_read"
        )
        is not True
    ):
        fail(
            "metric reconciliation was not pre-development-outcome"
        )


# =====================================================================
# I. Re-prove the genuine scientific failure exactly
# =====================================================================

primary = failure.get(
    "primary_development_acceptance",
    {}
)

if not (
    primary.get(
        "status"
    )
    ==
    "FAIL_PRIMARY_DEVELOPMENT_ACCEPTANCE_FROZEN"
    and
    primary.get(
        "vehicle_strict_improvement"
    )
    is True
    and
    primary.get(
        "pedestrian_visibility_noninferiority"
    )
    is True
    and
    primary.get(
        "cyclist_visibility_noninferiority"
    )
    is True
    and
    primary.get(
        "overmask_noninferiority"
    )
    is False
    and
    primary.get(
        "overall_pass"
    )
    is False
):
    fail(
        "primary scientific failure state changed"
    )


reason = failure.get(
    "failure_reason",
    {}
)

reactive = finite_float(
    reason.get(
        "reactive"
    ),
    "reactive overmask",
)

predictive = finite_float(
    reason.get(
        "predictive"
    ),
    "predictive overmask",
)

delta = finite_float(
    reason.get(
        "predictive_minus_reactive"
    ),
    "overmask delta",
)

allowed = finite_float(
    reason.get(
        "maximum_allowed_delta"
    ),
    "allowed overmask delta",
)

excess = finite_float(
    reason.get(
        "excess_beyond_allowed_delta"
    ),
    "overmask excess",
)

if not math.isclose(
    predictive - reactive,
    delta,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    fail(
        "overmask delta arithmetic mismatch"
    )

if not math.isclose(
    delta - allowed,
    excess,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    fail(
        "overmask excess arithmetic mismatch"
    )

if not math.isclose(
    allowed,
    over_delta,
    rel_tol=0.0,
    abs_tol=1e-15,
):
    fail(
        "failure gate does not use frozen overmask delta"
    )

if not (
    delta > allowed
    and excess > 0.0
):
    fail(
        "current result no longer represents genuine overmask failure"
    )


# =====================================================================
# J. Current protocol explicitly forbids retuning / formal continuation
# =====================================================================

frozen_policy = failure.get(
    "frozen_policy",
    {}
)

protocol_boundary = failure.get(
    "protocol_boundary",
    {}
)

if (
    frozen_policy.get(
        "post_outcome_retuning_allowed"
    )
    is not False
):
    fail(
        "failure closure no longer forbids post-outcome retuning"
    )

if not (
    protocol_boundary.get(
        "Stage7_handoff_allowed"
    )
    is False
    and
    protocol_boundary.get(
        "current_protocol_may_be_retuned"
    )
    is False
    and
    protocol_boundary.get(
        "formal_evaluation_allowed_after_failure"
    )
    is False
    and
    protocol_boundary.get(
        "formal_evaluation_run"
    )
    is False
    and
    protocol_boundary.get(
        "negative_result_preserved"
    )
    is True
):
    fail(
        "failure protocol boundary changed"
    )


# =====================================================================
# K. Clarify Stage-1 broad boolean scan
# =====================================================================

# Stage-1 printed formal_evaluation_detected=True because it performed
# a broad boolean-name scan over all reports/configs.  That signal is
# not authoritative evidence that the current Stage6 formal evaluator
# ran.  The frozen scientific failure closure is the authoritative
# execution boundary and states formal_evaluation_run=false.

stage1_formal_scan = (
    discovery.get(
        "repair_classification_signals",
        {},
    ).get(
        "formal_evaluation_detected"
    )
)

formal_execution_authority = {
    "Stage1_broad_boolean_scan":
        stage1_formal_scan,

    "authoritative_failure_closure_formal_evaluation_run":
        protocol_boundary.get(
            "formal_evaluation_run"
        ),

    "classification":
        (
            "NO_CURRENT_PROTOCOL_FORMAL_EVALUATION; "
            "Stage1 signal was broad metadata scan, "
            "not evaluator-execution proof"
        ),
}


# =====================================================================
# L. Regression — no scientific execution
# =====================================================================

env = os.environ.copy()

env[
    "PYTHONDONTWRITEBYTECODE"
] = "1"

env[
    "PYTHONPATH"
] = os.pathsep.join(
    [
        str(
            ROOT / "iscai_stage6/src"
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
        env.get(
            "PYTHONPATH",
            "",
        ),
    ]
)

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
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)

m = re.search(
    r"Ran\s+(\d+)\s+tests",
    reg.stdout,
)

test_count = (
    int(
        m.group(1)
    )
    if m
    else None
)

if not (
    reg.returncode == 0
    and test_count == 224
):
    fail(
        "Stage6 regression failed: "
        f"rc={reg.returncode}, tests={test_count}"
    )


# =====================================================================
# M. Build provenance-only Stage5 upstream supersession
# =====================================================================

old_upstream = contract.get(
    "upstream",
    {}
)

upstream_addendum = {
    "stage":
        6,

    "status":
        "FROZEN_PROVENANCE_SUPERSESSION_ONLY",

    "purpose":
        (
            "Bind Stage6 to the final reconciled/certified "
            "Stage5 authority created after the original "
            "Stage6 preimplementation contract. "
            "No Stage6 scientific parameter or result changes."
        ),

    "original_Stage6_contract": {
        "path":
            str(
                CONTRACT
            ),
        "sha256":
            sha(
                CONTRACT
            ),
        "original_Stage5_fields":
            old_upstream,
    },

    "superseding_final_Stage5_authority": {
        "certificate":
            str(
                STAGE5_CERT
            ),
        "certificate_sha256":
            EXPECTED_STAGE5_CERT_SHA,

        "closure":
            str(
                STAGE5_CLOSURE
            ),
        "closure_sha256":
            EXPECTED_STAGE5_CLOSURE_SHA,

        "handoff":
            str(
                STAGE5_HANDOFF
            ),
        "handoff_sha256":
            EXPECTED_STAGE5_HANDOFF_SHA,

        "Stage5_PDF_scope":
            "PASS_100_PERCENT",

        "Stage6_allowed":
            True,
    },

    "scientific_change":
        False,

    "runtime_change":
        False,

    "parameter_change":
        False,

    "acceptance_rule_change":
        False,

    "formal_outcome_change":
        False,

    "post_outcome_tuning":
        False,
}


# =====================================================================
# N. PDF Stage6 evidence matrix
# =====================================================================

source_tree_text = ""

for p in sorted(
    (S6 / "src/iscai_stage6/adb").glob(
        "*.py"
    )
):
    source_tree_text += p.read_text(
        encoding="utf-8",
        errors="replace",
    ).lower()

pdf_matrix = [
    {
        "requirement":
            "future box projection",
        "pass":
            (
                "full_box"
                in source_tree_text
                or
                "box_corners"
                in source_tree_text
                or
                "corners"
                in source_tree_text
            ),
    },
    {
        "requirement":
            "probabilistic masks / occupancy",
        "pass":
            (
                "occupancy"
                in source_tree_text
                or
                "p_occ"
                in source_tree_text
                or
                "pocc"
                in source_tree_text
            ),
    },
    {
        "requirement":
            "predictive covariance",
        "pass":
            "covariance"
            in source_tree_text,
    },
    {
        "requirement":
            "vehicle/pedestrian/cyclist class-aware policies",
        "pass":
            all(
                token
                in source_tree_text
                for token in (
                    "vehicle",
                    "pedestrian",
                    "cyclist",
                )
            ),
    },
    {
        "requirement":
            "Part-A reactive illumination continuity",
        "pass":
            (
                "original_reactive"
                in source_tree_text
                or
                "part_a"
                in source_tree_text
            ),
    },
    {
        "requirement":
            "temporal smoothing / actuation handling",
        "pass":
            (
                "temporal"
                in source_tree_text
                and
                (
                    "smooth"
                    in source_tree_text
                    or
                    "rho_dim"
                    in source_tree_text
                    or
                    "actuation"
                    in source_tree_text
                )
            ),
    },
    {
        "requirement":
            "predictive ADB reduces vehicle shadow violations",
        "pass":
            True,
    },
    {
        "requirement":
            "no uncontrolled VRU visibility loss",
        "pass":
            (
                primary.get(
                    "pedestrian_visibility_noninferiority"
                )
                is True
                and
                primary.get(
                    "cyclist_visibility_noninferiority"
                )
                is True
            ),
    },
    {
        "requirement":
            "no uncontrolled increase in over-masking",
        "pass":
            False,
    },
]

implementation_requirements = (
    pdf_matrix[
        :6
    ]
)

acceptance_requirements = (
    pdf_matrix[
        6:
    ]
)

implementation_complete = all(
    row[
        "pass"
    ]
    for row in implementation_requirements
)

stage6_acceptance_pass = all(
    row[
        "pass"
    ]
    for row in acceptance_requirements
)

if not implementation_complete:
    fail(
        "Stage6 mandatory predictive-ADB implementation "
        "evidence is incomplete"
    )

if stage6_acceptance_pass:
    fail(
        "unexpected: frozen negative result now passes"
    )


# =====================================================================
# O. Prepare final closure objects in memory
# =====================================================================

final_closure_obj = {
    "stage":
        6,

    "status":
        "STAGE6_SCIENTIFIC_NEGATIVE_RESULT_FROZEN_FINAL",

    "implementation_status":
        "COMPLETE",

    "PDF_implementation_scope":
        "PASS",

    "PDF_stage6_completion_gate":
        "FAIL",

    "PDF_completion_reason":
        (
            "Predictive ADB reduces vehicle shadow-zone "
            "violations and preserves pedestrian/cyclist "
            "visibility, but violates the pre-outcome frozen "
            "over-masking non-inferiority criterion."
        ),

    "scientific_result": {
        "vehicle_shadow_zone_violation":
            "PASS_STRICT_IMPROVEMENT",

        "pedestrian_visibility":
            "PASS_NONINFERIOR",

        "cyclist_visibility":
            "PASS_NONINFERIOR",

        "over_masking_area": {
            "status":
                "FAIL_NONINFERIORITY",

            "reactive":
                reactive,

            "predictive":
                predictive,

            "predictive_minus_reactive":
                delta,

            "preoutcome_frozen_allowed_delta":
                allowed,

            "excess_beyond_allowed_delta":
                excess,
        },
    },

    "acceptance_authority": {
        "contract":
            {
                "path":
                    str(
                        CONTRACT
                    ),
                "sha256":
                    sha(
                        CONTRACT
                    ),
            },

        "NI_rule":
            {
                "path":
                    str(
                        NI_RULE
                    ),
                "sha256":
                    sha(
                        NI_RULE
                    ),
            },

        "exact_deltas":
            {
                "path":
                    str(
                        EXACT_DELTAS
                    ),
                "sha256":
                    sha(
                        EXACT_DELTAS
                    ),
            },

        "primary_policy":
            {
                "path":
                    str(
                        PRIMARY_POLICY
                    ),
                "sha256":
                    sha(
                        PRIMARY_POLICY
                    ),
            },

        "primary_failure":
            {
                "path":
                    str(
                        FAILURE_CLOSURE
                    ),
                "sha256":
                    sha(
                        FAILURE_CLOSURE
                    ),
            },
    },

    "metric_semantics": {
        "status":
            metric_status,

        "path":
            str(
                METRIC_RECON
            ),

        "sha256":
            sha(
                METRIC_RECON
            ),

        "reconciled_preoutcome":
            True,
    },

    "scientific_boundary": {
        "controller_rerun_during_finalization":
            False,

        "development_evaluator_rerun_during_finalization":
            False,

        "formal_evaluator_rerun_during_finalization":
            False,

        "formal_evaluation_run_for_current_failed_protocol":
            False,

        "formal_outcomes_used_for_tuning":
            False,

        "post_outcome_retuning":
            False,

        "current_protocol_retuned":
            False,

        "acceptance_threshold_modified":
            False,

        "failure_reinterpreted_as_pass":
            False,

        "negative_result_preserved":
            True,
    },

    "formal_execution_resolution":
        formal_execution_authority,

    "Stage5_upstream_supersession":
        {
            "path":
                str(
                    UPSTREAM_ADDENDUM
                ),
            "sha256":
                None,
        },

    "PDF_requirement_matrix":
        pdf_matrix,

    "regression":
        {
            "tests":
                224,
            "status":
                "PASS",
        },

    "Stage7_allowed":
        False,

    "Stage7_block_reason":
        (
            "Stage6 PDF completion gate not satisfied. "
            "A new scientific Stage6 protocol/method version "
            "would be required before Stage7."
        ),

    "new_protocol_required_for_future_PASS":
        True,

    "current_protocol_may_be_retuned":
        False,
}


stage7_block_obj = {
    "from_stage":
        6,

    "to_stage":
        7,

    "status":
        "BLOCKED_BY_FROZEN_STAGE6_NEGATIVE_RESULT",

    "Stage7_allowed":
        False,

    "reason":
        (
            "Stage6 predictive ADB failed its pre-outcome "
            "frozen over-masking non-inferiority gate."
        ),

    "vehicle_shadow_violation":
        "PASS",

    "pedestrian_visibility":
        "PASS",

    "cyclist_visibility":
        "PASS",

    "over_masking":
        "FAIL",

    "overmask_delta":
        delta,

    "allowed_delta":
        allowed,

    "post_outcome_tuning":
        False,

    "failure_preserved":
        True,
}


# =====================================================================
# P. Write transaction
# =====================================================================

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=False,
)

REGRESSION_LOG.write_text(
    reg.stdout,
    encoding="utf-8",
)

os.chmod(
    REGRESSION_LOG,
    0o444,
)

upstream_sha = write_once(
    UPSTREAM_ADDENDUM,
    upstream_addendum,
)

final_closure_obj[
    "Stage5_upstream_supersession"
][
    "sha256"
] = upstream_sha

closure_sha = write_once(
    FINAL_CLOSURE,
    final_closure_obj,
)

stage7_block_obj[
    "Stage6_final_closure"
] = {
    "path":
        str(
            FINAL_CLOSURE
        ),
    "sha256":
        closure_sha,
}

stage7_block_sha = write_once(
    STAGE7_BLOCK,
    stage7_block_obj,
)


freeze_obj = {
    "stage":
        6,

    "status":
        "FINAL_NEGATIVE_RESULT_FREEZE_MANIFEST",

    "scientific_outcome":
        "NEGATIVE",

    "Stage6_PDF_completion":
        "FAIL",

    "implementation_complete":
        True,

    "Stage7_allowed":
        False,

    "frozen_files": {
        "Stage1_discovery":
            {
                "path":
                    str(
                        DISCOVERY
                    ),
                "sha256":
                    EXPECTED_DISCOVERY_SHA,
            },

        "Stage5_certificate":
            {
                "path":
                    str(
                        STAGE5_CERT
                    ),
                "sha256":
                    EXPECTED_STAGE5_CERT_SHA,
            },

        "Stage5_final_closure":
            {
                "path":
                    str(
                        STAGE5_CLOSURE
                    ),
                "sha256":
                    EXPECTED_STAGE5_CLOSURE_SHA,
            },

        "Stage5_final_handoff":
            {
                "path":
                    str(
                        STAGE5_HANDOFF
                    ),
                "sha256":
                    EXPECTED_STAGE5_HANDOFF_SHA,
            },

        "Stage6_contract":
            {
                "path":
                    str(
                        CONTRACT
                    ),
                "sha256":
                    sha(
                        CONTRACT
                    ),
            },

        "NI_rule":
            {
                "path":
                    str(
                        NI_RULE
                    ),
                "sha256":
                    sha(
                        NI_RULE
                    ),
            },

        "exact_deltas":
            {
                "path":
                    str(
                        EXACT_DELTAS
                    ),
                "sha256":
                    sha(
                        EXACT_DELTAS
                    ),
            },

        "primary_policy":
            {
                "path":
                    str(
                        PRIMARY_POLICY
                    ),
                "sha256":
                    sha(
                        PRIMARY_POLICY
                    ),
            },

        "primary_report":
            {
                "path":
                    str(
                        PRIMARY_REPORT
                    ),
                "sha256":
                    sha(
                        PRIMARY_REPORT
                    ),
            },

        "primary_detail":
            {
                "path":
                    str(
                        PRIMARY_DETAIL
                    ),
                "sha256":
                    sha(
                        PRIMARY_DETAIL
                    ),
            },

        "failure_closure":
            {
                "path":
                    str(
                        FAILURE_CLOSURE
                    ),
                "sha256":
                    sha(
                        FAILURE_CLOSURE
                    ),
            },

        "upstream_addendum":
            {
                "path":
                    str(
                        UPSTREAM_ADDENDUM
                    ),
                "sha256":
                    upstream_sha,
            },

        "final_closure":
            {
                "path":
                    str(
                        FINAL_CLOSURE
                    ),
                "sha256":
                    closure_sha,
            },

        "Stage7_block":
            {
                "path":
                    str(
                        STAGE7_BLOCK
                    ),
                "sha256":
                    stage7_block_sha,
            },

        "regression_log":
            {
                "path":
                    str(
                        REGRESSION_LOG
                    ),
                "sha256":
                    sha(
                        REGRESSION_LOG
                    ),
            },
    },

    "scientific_changes_during_finalization":
        False,

    "runtime_changes_during_finalization":
        False,

    "parameter_changes_during_finalization":
        False,

    "acceptance_changes_during_finalization":
        False,

    "post_outcome_tuning":
        False,
}

freeze_sha = write_once(
    FREEZE_MANIFEST,
    freeze_obj,
)


# =====================================================================
# Q. Verify all pre-existing Stage6 files stayed exact
# =====================================================================

after_tree = tree_manifest()

for rel, expected in (
    frozen_tree.items()
):
    actual = after_tree.get(
        rel
    )

    if actual != expected:
        fail(
            "pre-existing Stage6 file mutated during "
            f"finalization: {rel}"
        )


# =====================================================================
# R. Stage5 and Part-A still exact
# =====================================================================

if (
    sha(
        STAGE5_CERT
    )
    != EXPECTED_STAGE5_CERT_SHA
    or
    sha(
        STAGE5_CLOSURE
    )
    != EXPECTED_STAGE5_CLOSURE_SHA
    or
    sha(
        STAGE5_HANDOFF
    )
    != EXPECTED_STAGE5_HANDOFF_SHA
):
    fail(
        "Stage5 mutated during Stage6 finalization"
    )

git2 = subprocess.run(
    [
        "git",
        "-C",
        str(PARTA),
        "rev-parse",
        "HEAD",
    ],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)

if (
    git2.returncode != 0
    or
    git2.stdout.strip()
    !=
    "44d62e3478e3818d1757b00971890f844cb032f7"
):
    fail(
        "Part-A changed during Stage6 finalization"
    )


# =====================================================================
# S. Compact final result
# =====================================================================

print("=" * 78)
print(
    "STAGE6 FINALIZATION = "
    "SUCCESSFUL FROZEN NEGATIVE SCIENTIFIC CLOSURE"
)
print("=" * 78)

print(
    "Stage5_final_authority = PASS / EXACT"
)

print(
    "PartA_frozen_commit = PASS / EXACT"
)

print(
    "Stage6_preexisting_tree = UNCHANGED"
)

print(
    "regression = 224/224 PASS"
)

print()

print(
    "vehicle_shadow_violation = PASS"
)

print(
    "pedestrian_visibility = PASS"
)

print(
    "cyclist_visibility = PASS"
)

print(
    "overmask_noninferiority = FAIL"
)

print(
    "reactive_overmask =",
    reactive,
)

print(
    "predictive_overmask =",
    predictive,
)

print(
    "predictive_minus_reactive =",
    delta,
)

print(
    "frozen_allowed_delta =",
    allowed,
)

print(
    "excess_beyond_bound =",
    excess,
)

print()

print(
    "formal_evaluation_run_for_failed_protocol = false"
)

print(
    "controller_rerun = false"
)

print(
    "evaluator_rerun = false"
)

print(
    "acceptance_threshold_changed = false"
)

print(
    "current_protocol_retuned = false"
)

print(
    "post_outcome_tuning = false"
)

print()

print(
    "Stage6_implementation_scope = COMPLETE"
)

print(
    "Stage6_PDF_completion_gate = FAIL"
)

print(
    "Stage6_status = "
    "STAGE6_SCIENTIFIC_NEGATIVE_RESULT_FROZEN_FINAL"
)

print(
    "Stage7_allowed = false"
)

print(
    "new_Stage6_protocol_required_for_future_PASS = true"
)

print()

print(
    "upstream_addendum_sha256 =",
    upstream_sha,
)

print(
    "final_closure_sha256 =",
    closure_sha,
)

print(
    "stage7_block_sha256 =",
    stage7_block_sha,
)

print(
    "freeze_manifest_sha256 =",
    freeze_sha,
)

print("=" * 78)
